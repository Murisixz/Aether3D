"""GPU data layouts and packing, checked without a GPU.

The byte layouts the CPU writes must match the WGSL structs the shaders read.
These tests compute each struct's member offsets from the WGSL source with
WGSL's alignment rules (a small independent layout calculator below) and
compare them with the offsets the ``pack_*`` functions write to.
"""

import re

import numpy as np
import pytest

from aether3d import Scene, create_sphere
from aether3d.rendering.color import linear_to_srgb, quantize
from aether3d.rendering.gpu import resources, shaders
from aether3d.rendering.scene_data import extract_render_scene, resolve_camera
from aether3d.rendering.settings import ShadowSettings
from aether3d.rendering.shadows import plan_shadows

# (size, alignment) of WGSL types used in host-shared structs
SCALARS = {"f32": (4, 4), "u32": (4, 4), "i32": (4, 4)}


def wgsl_type(name: str, source: str | None = None) -> tuple[int, int]:
    name = name.strip()
    if name in SCALARS:
        return SCALARS[name]
    m = re.fullmatch(r"vec([234])<(f32|u32|i32)>", name)
    if m:
        n = int(m.group(1))
        return (4 * n, 8 if n == 2 else 16)
    if name == "mat4x4<f32>":
        return (64, 16)
    m = re.fullmatch(r"array<(.+),\s*(\d+)>", name)
    if m:
        size, align = wgsl_type(m.group(1))
        stride = (size + align - 1) // align * align
        return (stride * int(m.group(2)), align)
    if source is not None and re.search(r"struct\s+" + name + r"\s*\{", source):
        offsets, size = struct_layout(source, name)  # a nested struct
        return (size, _struct_align(source, name))
    raise AssertionError(f"unexpected WGSL type {name}")


def _struct_align(source: str, struct: str) -> int:
    body = re.search(r"struct\s+" + struct + r"\s*\{(.*?)\};", source, re.S).group(1)
    kinds = [
        line.split("//")[0].strip().rstrip(",").partition(":")[2] for line in body.splitlines()
    ]
    return max(wgsl_type(k, source)[1] for k in kinds if k.strip())


def struct_layout(source: str, struct: str) -> tuple[dict[str, int], int]:
    body = re.search(r"struct\s+" + struct + r"\s*\{(.*?)\};", source, re.S).group(1)
    offsets, offset, largest = {}, 0, 0
    for line in body.splitlines():
        line = line.split("//")[0].strip().rstrip(",")
        if not line:
            continue
        name, _, kind = line.partition(":")
        size, align = wgsl_type(kind, source)
        offset = (offset + align - 1) // align * align
        offsets[name.strip()] = offset
        offset += size
        largest = max(largest, align)
    return offsets, (offset + largest - 1) // largest * largest


SCENE = shaders.module_source("scene")
FINALIZE = shaders.module_source("finalize", True)


def test_instance_layout_matches_the_shader():
    offsets, size = struct_layout(SCENE, "Instance")
    assert size == 4 * resources.INSTANCE_FLOATS == 128
    assert offsets == {
        "model": 0,
        "normal": 48,
        "base_color": 96,
        "specular": 108,
        "shininess": 112,
        "flags": 116,
        "index": 120,
        "colors": 124,  # (Stage 16: was "_pad"; the first vertex colour, HAS_COLORS)
    }


def test_frame_light_shadow_and_finalize_layouts_match_the_shaders():
    frame, size = struct_layout(SCENE, "Frame")
    assert size == 4 * resources.FRAME_FLOATS
    assert (frame["forward"], frame["ambient"], frame["light_count"]) == (64, 80, 96)
    assert (frame["pcf_radius"], frame["shadow_texel"]) == (100, 104)
    assert (frame["eye"], frame["atmo"]) == (112, 4 * 32)  # pack_frame: atmosphere at [32:80]
    light, size = struct_layout(SCENE, "Light")
    assert size == 4 * resources.LIGHT_FLOATS
    assert [light[k] for k in ("radiance", "kind", "vector", "range", "falloff", "shadow")] == [
        0,
        12,
        16,
        28,
        32,
        36,
    ]
    assert light["cascades"] == 40  # pack_lights: word 10
    shadow, size = struct_layout(SCENE, "ShadowMap")
    assert size == 4 * resources.SHADOW_FLOATS
    assert (shadow["bias"], shadow["normal_offset"], shadow["split"]) == (64, 68, 72)
    # (Stage 16: the finalize pass also gets the camera basis and the atmosphere; it was
    # {"size": 0, "depth": 16} and 32 bytes.)
    params, size = struct_layout(FINALIZE, "Params")
    assert params == {"size": 0, "depth": 16, "forward": 32, "right": 48, "up": 64, "atmo": 80}
    assert size == 4 * (20 + resources.ATMO_FLOATS)


def test_atmosphere_layout_matches_the_shader_and_the_packing():
    atmo, size = struct_layout(SCENE, "Atmo")
    assert size == 4 * resources.ATMO_FLOATS
    names = [
        "flags",
        "sun",
        "zenith",
        "horizon",
        "ground",
        "glow",
        "fog",
        "fog_color",
        "fog_sun",
        "ground_ambient",
        "background",
        "_pad",
    ]
    assert atmo == {name: 16 * k for k, name in enumerate(names)}
    scene = lit_scene()
    scene.world.atmosphere.configure(
        enabled=True,
        sun="Sun",
        fog_density=0.002,
        fog_height=5.0,
        fog_falloff=0.1,
        ambient="hemisphere",
    )
    rs = extract(scene)
    data = resources.pack_atmosphere(rs)
    words = data.view(np.uint32)
    assert words[0] == 1 and words[2] == 1  # enabled, hemisphere ambient
    assert data[4 * 0 + 7] == 1.0  # a sun
    np.testing.assert_allclose(
        data[4:7], -np.array([0.3, 0.4, -1.0]) / np.linalg.norm([0.3, 0.4, -1]), rtol=1e-6
    )
    assert data[atmo["fog"] // 4] == pytest.approx(0.002)
    assert data[atmo["fog"] // 4 + 1] == pytest.approx(0.1)
    assert data[atmo["fog"] // 4 + 3] == 1.0
    frame = resources.pack_frame(rs, 16 / 9, None)
    np.testing.assert_array_equal(frame[32:80], data)
    off = resources.pack_atmosphere(extract(lit_scene()))
    assert not off.any()


def lit_scene():
    scene = Scene("Packing")
    ball = create_sphere(1.0, 16, 8)
    scene.add_object("A", mesh=ball, position=(1e6, 0, 0)).appearance.configure(
        color="#ff8000", specular=0.5, shininess=20
    )
    scene.add_object("B", mesh=ball, position=(1e6 + 3, 0, 0), scale=(-1, 1, 1))
    scene.add_cube("C", size=1, position=(1e6, 3, 0))
    scene.add_point_light("Bulb", position=(1e6, -2, 3), intensity=5, range=9)
    scene.add_directional_light("Sun", direction=(0.3, 0.4, -1))
    scene.add_camera("Cam", position=(1e6 + 1, -10, 2), look_at=(1e6, 0, 0))
    return scene


def extract(scene, aspect=16 / 9):
    return extract_render_scene(scene, resolve_camera(scene, None), aspect=aspect)


def test_instances_are_camera_relative_and_keep_their_precision():
    rs = extract(lit_scene())
    eye = rs.camera.frame.position
    order, draws = resources.draw_order(rs, "none")
    data = resources.pack_instances(rs, order, eye)
    words = data.view(np.uint32)
    for row, index in enumerate(order):
        inst = rs.instances[index]
        model = data[row, :12].reshape(3, 4).astype(np.float64)
        # The translation is world - eye, computed before rounding: about 1 unit,
        # exact to float32, although the world coordinates are about 1e6.
        assert np.allclose(model[:, 3], inst.world[:3, 3] - eye, atol=1e-6)
        assert np.allclose(model[:, :3], inst.world[:3, :3], atol=1e-6)
        assert words[row, 30] == index
        assert words[row, 29] & resources.MIRRORED == (1 if inst.mirrored else 0)
        assert words[row, 29] & resources.HAS_NORMALS == resources.HAS_NORMALS
    assert data[list(order).index(0), 27] == pytest.approx(0.5)  # specular of "A"


def test_shared_meshes_become_one_instanced_draw():
    rs = extract(lit_scene())
    order, draws = resources.draw_order(rs, "none")
    assert [(m, f, c) for m, _, f, c in draws] == [(0, 0, 2), (1, 2, 1)]  # sphere x2, cube
    order, draws = resources.draw_order(rs, "back")  # culling: mirrored drawn separately
    assert [(m, mirrored, c) for m, mirrored, _, c in draws] == [
        (0, False, 1),
        (0, True, 1),
        (1, False, 1),
    ]
    keep = np.array([True, False, True])
    order, draws = resources.draw_order(rs, "none", keep)
    assert order.tolist() == [0, 2] and [c for *_, c in draws] == [1, 1]


def test_lights_and_shadow_layers():
    rs = extract(lit_scene())
    eye = rs.camera.frame.position
    plan = plan_shadows(rs, ShadowSettings(), 16 / 9)
    data = resources.pack_lights(rs, eye, plan)
    words, signed = data.view(np.uint32), data.view(np.int32)
    point = [i for i, light in enumerate(rs.lights) if light.type == "point"][0]
    sun = 1 - point
    assert words[point, 3] == 1 and words[sun, 3] == 0
    assert np.allclose(data[point, 4:7], rs.lights[point].position - eye)
    assert data[point, 7] == 9 and words[point, 8] == 1
    assert signed[point, 9] == -1 and signed[sun, 9] == 0
    maps = resources.pack_shadow_maps(plan)
    assert maps.shape == (1, resources.SHADOW_FLOATS)
    assert np.allclose(maps[0, :16].reshape(4, 4).T, plan.maps[0].matrix)


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_reversed_depth_maps_near_to_one_and_far_to_zero(projection):
    scene = Scene()
    scene.add_cube("C")
    cam = scene.add_camera("Cam", projection=projection, near=0.5, far=200, position=(0, -5, 0))
    cam.look_at((0, 0, 0))
    rs = extract(scene)
    vp = resources.view_projection(rs, 16 / 9)
    standard = rs.camera.camera.projection_matrix(16 / 9) @ rs.camera.frame.view_matrix()
    f = rs.camera.frame
    for depth, expected in ((0.5, 1.0), (200.0, 0.0), (20.0, None)):
        relative = depth * f.forward + 0.3 * f.right - 0.2 * f.up
        clip = vp @ np.append(relative, 1.0)
        reference = standard @ np.append(relative + f.position, 1.0)
        assert np.allclose(clip[:2] / clip[3], reference[:2] / reference[3])
        z = clip[2] / clip[3]
        assert z == pytest.approx(1.0 - reference[2] / reference[3], abs=1e-12)
        if expected is not None:
            assert z == pytest.approx(expected, abs=1e-12)


def test_gpu_srgb_encoding_rounds_exactly_like_the_cpu():
    thresholds = resources.srgb_thresholds()
    rng = np.random.default_rng(5)
    values = np.concatenate(
        [
            rng.random(200_000).astype(np.float32),
            (rng.random(50_000) * 0.01).astype(np.float32),
            np.array([0, 1, 2, -1, 1e-9, 0.0031308, 0.5, np.inf], dtype=np.float32),
        ]
    )
    # The finalize shader's binary search counts the thresholds at or below a value.
    gpu = np.searchsorted(thresholds[1:], values, side="right")
    cpu = quantize(linear_to_srgb(values.astype(np.float64)))
    assert np.array_equal(gpu, cpu)


def test_output_layout_is_aligned_and_decodes():
    layout = resources.OutputLayout.build(7, 5, 3, 4)
    sizes = layout.sizes()
    starts = [layout.image, layout.linear, layout.aux, layout.counts, layout.queries]
    assert all(s % resources.ALIGN == 0 for s in starts)
    ends = [s + sizes[k] for s, k in zip(starts, sizes, strict=True)]
    assert all(e <= s for e, s in zip(ends, starts[1:], strict=False)) and ends[-1] <= layout.total
    raw = np.zeros(layout.total, dtype=np.uint8)
    raw[layout.image : layout.image + 4] = (10, 20, 30, 0)
    raw[layout.aux : layout.aux + 12] = np.array([2.5, 0, 0.25], np.float32).view(np.uint8)
    raw[layout.aux + 4 : layout.aux + 8] = np.array([-1], np.int32).view(np.uint8)
    raw[layout.counts : layout.counts + 12] = np.array([4, 5, 6], np.uint32).view(np.uint8)
    out = layout.decode(raw)
    assert out["image"][0, 0].tolist() == [10, 20, 30]
    assert out["depth"][0, 0] == 2.5 and out["object_index"][0, 0] == -1
    assert out["coverage"][0, 0] == 0.25 and out["counts"].tolist() == [4, 5, 6]


def test_shader_modules_are_assembled_from_the_bundled_files_only():
    for kind in ("scene", "shadow"):
        text = shaders.module_source(kind)
        assert text.startswith(shaders.shader_source("common"))
    single = shaders.module_source("finalize", False)
    multi = shaders.module_source("finalize", True)
    assert not re.search(r"\{\{[A-Z_]+\}\}", single + multi)  # every placeholder filled
    assert "texture_multisampled_2d<f32>" in multi and "texture_multisampled_2d" not in single
    with pytest.raises(ValueError):
        shaders.module_source("anything else")


def test_writable_mesh_arrays_are_never_cached():
    rs = extract(lit_scene())
    mesh = rs.meshes[0]
    assert resources._mesh_key(mesh) is not None
    writable = type(mesh)(mesh.positions.copy(), mesh.triangles, mesh.normals)
    assert resources._mesh_key(writable) is None


def test_cache_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("AETHER3D_GPU_CACHE_MB", "64")
    assert resources.mesh_budget() == 64 * 2**20
    monkeypatch.setenv("AETHER3D_GPU_CACHE_MB", "lots")
    with pytest.raises(Exception) as info:
        resources.mesh_budget()
    assert "AETHER3D_GPU_CACHE_MB" in str(info.value)
