"""Baking: closed-form references for every map, the render of a baked normal map against
the high-resolution surface it came from, determinism and errors."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Mesh, Scene, TexturingError
from aether3d.primitives import create_sphere
from aether3d.texturing import bake, rasterize_uv
from aether3d.texturing.dilate import dilate


def plane(size=2.0, n=8, z=0.0) -> Mesh:
    xs, ys = np.meshgrid(
        np.linspace(-size / 2, size / 2, n + 1), np.linspace(-size / 2, size / 2, n + 1)
    )
    p = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], 1)
    uv = (p[:, :2] + size / 2) / size
    t = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            t += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return Mesh(p, t, normals=np.tile([0, 0, 1.0], (p.shape[0], 1)), uvs=uv)


# --------------------------------------------------------------------------- #
# Rasterization and padding
# --------------------------------------------------------------------------- #


def test_uv_rasterization_covers_each_sample_once_with_valid_barycentrics():
    m = plane(n=7)
    uv = m.uvs.astype(np.float64)[m.triangles]
    cov = rasterize_uv(uv, 50, 40, samples=2)
    assert cov.overlaps == 0
    assert cov.sample.size == 50 * 40 * 4  # the unit square, every sample exactly once
    assert np.all(cov.bary >= -1e-12) and np.allclose(cov.bary.sum(axis=1), 1.0)
    # Barycentrics reproduce the sample positions, up to the snapping of corners to 1/256
    # of a sample.
    corners = uv[cov.triangle]
    p = np.einsum("nk,nkc->nc", cov.bary, corners)
    row, col = np.divmod(cov.sample, 100)
    assert np.allclose(p[:, 0], (col + 0.5) / 100, rtol=0, atol=1 / 256 / 100)
    assert np.allclose(p[:, 1], 1 - (row + 0.5) / 80, rtol=0, atol=1 / 256 / 80)


def test_overlapping_uvs_take_the_lowest_triangle():
    uv = np.array([[[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]]], float)
    cov = rasterize_uv(uv, 8, 8)
    assert np.all(cov.triangle == 0) and cov.overlaps == cov.sample.size


def test_dilation_grows_ring_by_ring():
    v = np.zeros((7, 7, 1))
    cov = np.zeros((7, 7), bool)
    v[3, 3] = 5.0
    cov[3, 3] = True
    out, grown = dilate(v, cov, 2)
    assert grown[1:6, 1:6].all() and not grown[0].any()
    assert np.allclose(out[grown], 5.0)


# --------------------------------------------------------------------------- #
# Maps against closed forms
# --------------------------------------------------------------------------- #


def test_a_surface_baked_onto_itself_gives_a_flat_normal_map():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 24, 12))
    result = bake(
        scene, ball, maps=["normal", "object_normal"], resolution=64, samples=1, padding=0
    )
    hit = result.hit > 0
    nmap = result.maps["normal"][hit][:, :3] * 2 - 1
    assert np.allclose(nmap, [0, 0, 1], atol=1e-6)
    assert result.report["miss_fraction"] < 1e-3


def test_height_of_a_parallel_plane():
    scene = Scene()
    low = scene.add_object("Low", mesh=plane())
    scene.add_object("High", mesh=plane(z=0.05))
    result = bake(
        scene,
        low,
        "High",
        ["height"],
        resolution=32,
        samples=1,
        cage=0.1,
        depth=0.1,
        height_range=0.1,
        padding=0,
    )
    h = result.maps["height"][result.hit > 0][:, 0]
    assert np.allclose(h, 0.5 + 0.5 * 0.5, atol=1e-9)


def test_ambient_occlusion_under_a_parallel_ceiling():
    # A plane under a large parallel ceiling at height h, AO distance D: rays leaving at
    # angle theta hit the ceiling at h / cos(theta), so the cosine-weighted visible fraction
    # is (h / D)^2.
    scene = Scene()
    floor = scene.add_object("Floor", mesh=plane(size=0.5, n=4))
    ceiling = plane(size=200.0, n=1, z=0.25)
    flipped = Mesh(ceiling.positions, ceiling.triangles[:, ::-1], uvs=ceiling.uvs)
    scene.add_object("Ceiling", mesh=flipped)
    result = bake(
        scene,
        floor,
        ["Floor", "Ceiling"],
        ["ao"],
        resolution=16,
        samples=1,
        ao_rays=256,
        ao_distance=0.5,
        cage=0.01,
        depth=0.01,
        padding=0,
    )
    ao = result.maps["ao"][result.hit > 0][:, 0]
    assert abs(ao.mean() - 0.25) < 0.01


def test_curvature_of_a_sphere():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(2.0, 64, 32))
    result = bake(
        scene, ball, maps=["curvature"], resolution=64, samples=1, curvature_radius=1.0, padding=0
    )
    k = result.maps["curvature"][result.hit > 0][:, 0]
    # k = 1/2: value 0.5 + 0.5 * 1/2 = 0.75 (the discrete estimate is within a few percent).
    assert abs(np.median(k) - 0.75) < 0.01


def test_thickness_of_a_slab():
    scene = Scene()
    top = plane(size=0.4, n=4, z=0.0)
    bottom = plane(size=50.0, n=1, z=-0.1)
    # The target is the slab's top; the source is the slab (top facing up, bottom down).
    target = scene.add_object("Top", mesh=top)
    flipped = Mesh(bottom.positions, bottom.triangles[:, ::-1], uvs=bottom.uvs)
    scene.add_object("Bottom", mesh=flipped)
    result = bake(
        scene,
        target,
        ["Top", "Bottom"],
        ["thickness"],
        resolution=8,
        samples=1,
        thickness_rays=256,
        thickness_distance=0.4,
        cage=0.01,
        depth=0.01,
        padding=0,
    )
    got = result.maps["thickness"][result.hit > 0][:, 0].mean()
    # E[min(d / cos(theta), D)] / D over cosine-weighted directions (cos^2 uniform).
    u = (np.arange(200000) + 0.5) / 200000
    expected = np.minimum(0.1 / np.sqrt(1 - u), 0.4).mean() / 0.4
    assert abs(got - expected) < 0.01


def test_material_channels_and_ids():
    scene = Scene()
    low = scene.add_object("Low", mesh=plane())
    high = scene.add_object("High", mesh=plane(z=0.01))
    red = scene.create_material("Red", base_color="#ff0000", roughness=0.3, metallic=1.0)
    high.material = red
    result = bake(
        scene,
        low,
        high,
        ["base_color", "roughness", "metallic", "material_id"],
        resolution=8,
        samples=1,
        padding=0,
    )
    hit = result.hit > 0
    assert np.allclose(result.maps["base_color"][hit][:, :3], [1, 0, 0], atol=1e-12)
    assert np.allclose(result.maps["roughness"][hit][:, 0], 0.3)
    assert np.allclose(result.maps["metallic"][hit][:, 0], 1.0)
    ids = result.maps["material_id"][hit][:, :3]
    assert np.allclose(ids, ids[0])


# --------------------------------------------------------------------------- #
# A baked normal map renders like the surface it came from
# --------------------------------------------------------------------------- #


def bumpy_sphere(radius=1.0, seg=192, rings=96) -> Mesh:
    m = create_sphere(radius, seg, rings)
    p = m.positions.astype(np.float64)
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    r = radius * (1 + 0.03 * np.sin(9 * u[:, 0]) * np.sin(9 * u[:, 1]) * np.cos(7 * u[:, 2]))
    q = u * r[:, None]
    from aether3d.uv.mikktspace import face_normals

    t = m.triangles.astype(np.int64)
    fn = face_normals(q.astype(np.float32), t).astype(np.float64)
    n = np.zeros_like(q)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(length > 1e-12, n / np.maximum(length, 1e-300), u)  # poles: radial
    return Mesh(q, m.triangles, normals=n)


def test_baked_normal_map_renders_like_the_high_resolution_surface():
    scene = Scene()
    scene.render_settings.default_lights = False
    high = scene.add_object("High", mesh=bumpy_sphere())
    low = scene.add_object("Low", mesh=create_sphere(1.0, 32, 16))
    low.unwrap(resolution=512, padding=4)
    result = bake(scene, low, high, ["normal"], resolution=512, samples=2, cage=0.05, depth=0.05)
    assert result.report["miss_fraction"] == 0.0
    tex = result.to_texture(scene, "normal", "Baked normal")
    mat = scene.create_material("Baked", roughness=0.8)
    mat.set_texture("normal", tex)
    low.material = mat
    scene.add_directional_light("Key", direction=(-0.5, 0.7, -0.6), intensity=2.0)
    scene.add_camera("Cam", position=(0, -3.2, 0.6), look_at=(0, 0, 0), fov=40)
    plain = scene.materials.material("Baked")
    del plain
    low.visible, high.visible = True, False
    baked = scene.render(width=160, height=160, samples=2)
    high.visible, low.visible = True, False
    reference = scene.render(width=160, height=160, samples=2)
    low.material = None
    low.visible, high.visible = True, False
    unmapped = scene.render(width=160, height=160, samples=2)
    both = (baked.object_index >= 0) & (reference.object_index >= 0)
    inner = both
    for _ in range(3):  # away from the silhouettes, which a normal map cannot move
        inner = inner & np.roll(inner, 1, 0) & np.roll(inner, -1, 0)
        inner = inner & np.roll(inner, 1, 1) & np.roll(inner, -1, 1)
    err_baked = np.abs(baked.linear - reference.linear)[inner].mean()
    err_plain = np.abs(unmapped.linear - reference.linear)[inner].mean()
    # Measured 0.21: what remains is parallax (the 3% bumps are not displaced) and the
    # filtering of the 8-bit map.
    assert err_baked < 0.25 * err_plain


def test_the_bake_is_the_exact_inverse_of_the_renderers_transform():
    from aether3d.texturing.bake import tangent_space

    rng = np.random.default_rng(5)
    n = rng.normal(size=(1000, 3))  # interpolated (unnormalized) normals
    t = rng.normal(size=(1000, 3))
    t -= n * ((t * n).sum(1) / (n * n).sum(1))[:, None] * rng.uniform(0.9, 1.0, (1000, 1))
    sign = np.where(rng.random(1000) < 0.5, -1.0, 1.0)
    target = rng.normal(size=(1000, 3))
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    tn = tangent_space(target, t, n, sign)
    b = sign[:, None] * np.cross(n, t)
    mapped = t * tn[:, 0:1] + b * tn[:, 1:2] + n * tn[:, 2:3]  # the renderers' transform
    mapped /= np.linalg.norm(mapped, axis=1, keepdims=True)
    assert np.allclose(mapped, target, atol=1e-9)


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def test_bakes_are_deterministic():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 16, 8))
    a = bake(scene, ball, maps=["ao", "normal"], resolution=32, ao_rays=16)
    b = bake(scene, ball, maps=["ao", "normal"], resolution=32, ao_rays=16)
    for name in a.maps:
        assert np.array_equal(a.maps[name], b.maps[name])


def test_misses_are_reported():
    scene = Scene()
    low = scene.add_object("Low", mesh=plane())
    scene.add_object("High", mesh=plane(z=0.5))
    result = bake(scene, low, "High", ["normal"], resolution=16, cage=0.1, depth=0.1)
    assert [d.code for d in result.diagnostics] == ["BAKE_MISSES"]
    assert result.report["miss_fraction"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"maps": ["sparkle"]}, "INVALID_ARGUMENT"),
        ({"resolution": 0}, "INVALID_ARGUMENT"),
        ({"samples": 9}, "INVALID_ARGUMENT"),
        ({"resolution": 16384, "samples": 2}, "BAKE_TOO_LARGE"),
        ({"cage": -1}, "INVALID_ARGUMENT"),
        ({"cage": 0, "depth": 0}, "INVALID_ARGUMENT"),
        ({"ao_rays": 0, "maps": ["ao"]}, "INVALID_ARGUMENT"),
        ({"sources": ["Nobody"]}, "OBJECT_NOT_FOUND"),
        ({"backend": "quantum"}, "INVALID_ARGUMENT"),
    ],
)
def test_invalid_bakes_are_refused(kwargs, code):
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 8, 4))
    with pytest.raises(TexturingError) as exc:
        bake(scene, ball, **{"maps": ["normal"], **kwargs})
    assert exc.value.code == code


def test_a_target_without_uvs_is_refused():
    scene = Scene()
    m = create_sphere(1.0, 8, 4)
    obj = scene.add_object("Bare", mesh=Mesh(m.positions, m.triangles))
    with pytest.raises(TexturingError) as exc:
        bake(scene, obj)
    assert exc.value.code == "NO_UVS"


def test_cancellation():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 8, 4))
    with pytest.raises(TexturingError) as exc:
        bake(scene, ball, resolution=16, should_stop=lambda: True)
    assert exc.value.code == "CANCELLED"


def test_scene_bake_applies_maps_in_one_undo_step():
    scene = Scene()
    low = scene.add_object("Low", mesh=plane())
    scene.add_object("High", mesh=plane(z=0.01))
    before = len(scene.history.entries)
    result = scene.bake(low, "High", ["normal", "ao", "roughness"], resolution=16, apply=True)
    applied = result.report["applied"]
    assert set(applied["textures"]) == {"normal", "ao", "roughness"}
    material = applied["material"]
    assert low.material is material
    assert set(material.textures) == {"normal", "occlusion", "roughness"}
    assert len(scene.history.entries) == before + 1
    scene.history.undo()
    assert low.material is None and not scene.materials.textures


def test_gpu_bake_agrees_with_the_cpu(gpu):
    scene = Scene()
    low = scene.add_object("Low", mesh=create_sphere(1.0, 24, 12))
    low.unwrap(resolution=128)
    scene.add_object("High", mesh=create_sphere(1.0, 96, 48))
    scene.add_object("Floor", mesh=plane(4.0, 4, z=-1.05))
    options = {"maps": ["normal", "ao", "thickness", "height"], "resolution": 128, "ao_rays": 16}
    cpu = bake(scene, low, ["High", "Floor"], backend="cpu", **options)
    on_gpu = bake(scene, low, ["High", "Floor"], backend="gpu", adapter=gpu.adapter, **options)
    assert on_gpu.report["backend"] == "gpu" and cpu.report["backend"] == "cpu"
    assert np.array_equal(cpu.hit, on_gpu.hit)
    for name in ("normal", "height"):
        assert np.abs(cpu.maps[name] - on_gpu.maps[name]).max() < 1e-5, name
    # AO and thickness: the same rays; float32 rounding may flip the odd grazing hit.
    for name in ("ao", "thickness"):
        diff = np.abs(cpu.maps[name] - on_gpu.maps[name])[..., 0]
        assert diff.mean() < 1e-3 and np.mean(diff > 1e-4) < 0.01, name


def test_gpu_backend_without_a_gpu_is_refused(monkeypatch):
    from aether3d.texturing import gpu_bake

    def unavailable(*args, **kwargs):
        raise TexturingError("GPU_UNAVAILABLE", "no GPU here", "backend")

    monkeypatch.setattr(gpu_bake, "GPUTracer", unavailable)
    scene = Scene()
    low = scene.add_object("Low", mesh=create_sphere(1.0, 12, 6))
    low.unwrap(resolution=64)
    with pytest.raises(TexturingError) as exc:
        bake(scene, low, maps=["normal"], resolution=64, backend="gpu")
    assert exc.value.code == "GPU_UNAVAILABLE"
    assert bake(scene, low, maps=["normal"], resolution=64).report["backend"] == "cpu"
