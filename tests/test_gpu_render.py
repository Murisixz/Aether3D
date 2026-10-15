"""Real GPU renders, checked against the CPU renderer, the ray caster and analytic values.

Every test here draws on a real graphics adapter and reads the framebuffer
back (the ``gpu`` fixture skips them, with the reason, when none works; CI sets
``AETHER3D_REQUIRE_GPU=1`` on a job with Mesa's lavapipe so they must run).

Tolerances, and why (GPU rasterizers legitimately differ from Aether3D's CPU
rasterizer in sub-pixel details):

* **Parity mode** (``supersample=s, msaa=1``) samples exactly the CPU's pixel
  positions. The object seen at each pixel matches except at triangle edges,
  where hardware snaps vertices to its own sub-pixel grid (1/256 pixel on
  NVIDIA and llvmpipe; Vulkan only requires 1/16) and computes in float32:
  mismatches must lie on object boundaries and stay under 0.5% of pixels.
* **Colors**: away from edges, 8-bit sRGB values within 2 codes (observed: 1);
  linear radiance within 2e-3 (observed: 2e-4 on an RTX 4060, 3e-6 on
  llvmpipe -- hardware interpolates attributes from snapped vertices).
* **Depth**: relative error below 1e-3 away from edges (observed: 1e-4).
* **MSAA** uses the device's sample pattern, not the CPU's grid, and shades
  once per pixel (at the centroid of the covered samples) instead of once per
  sample. Where shading is smooth the results agree within 2 codes; inside
  sharp highlights and on creases they legitimately differ by up to the local
  shading variation. Object coverage agrees within 1% of the samples.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from render_reference import reference_image  # noqa: E402
from test_render_scene import boundary_mask, hostile_scene  # noqa: E402

from aether3d import Expr, RenderError, Scene, create_cube, create_sphere, import_gltf  # noqa: E402
from aether3d.rendering.gpu import device as gpu_device  # noqa: E402

DATA = Path(__file__).parent / "data"


def interior(*maps):
    """Pixels whose 3x3 neighbourhood sees a single object in every given object map."""
    mask = np.ones(maps[0].shape, dtype=bool)
    for ids in maps:
        mask &= ~boundary_mask(ids)
    return mask


def local_variation(image):
    """Largest 8-bit difference between each pixel and its 3x3 neighbours."""
    h, w, _ = image.shape
    img = image.astype(int)
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros((h, w), dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            other = pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            out = np.maximum(out, np.abs(other - img).max(axis=-1))
    return out


def assert_parity(gpu_result, cpu_result, *, codes=2, linear=2e-3, depth=1e-3):
    g, c = gpu_result, cpu_result
    mismatch = g.object_index != c.object_index
    assert np.all(boundary_mask(c.object_index)[mismatch]), "a coverage difference away from edges"
    assert mismatch.mean() < 0.005
    inner = interior(c.object_index, g.object_index) & ~mismatch
    diff = np.abs(g.image.astype(int) - c.image.astype(int)).max(axis=-1)
    assert diff[inner].max(initial=0) <= codes
    assert np.abs(g.linear - c.linear).max(axis=-1)[inner].max(initial=0) <= linear
    fg = inner & (c.object_index >= 0)
    rel = np.abs(g.depth[fg] - c.depth[fg]) / c.depth[fg]
    assert rel.max(initial=0) < depth
    assert np.all(np.isinf(g.depth[g.object_index < 0]))
    for go, co in zip(g.objects, c.objects, strict=True):
        assert go["name"] == co["name"]
        assert (
            abs(go["visible_samples"] - co["visible_samples"]) <= 0.01 * co["visible_samples"] + 2
        )
    return mismatch.mean()


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
@pytest.mark.parametrize("samples", [1, 2, 3])
def test_parity_mode_reproduces_the_cpu_renderer(gpu, projection, samples):
    scene = hostile_scene(projection)
    cpu = scene.render(width=128, height=80, samples=samples)
    result = gpu.render(scene, width=128, height=80, samples=samples, supersample=samples, msaa=1)
    assert result.backend["name"] == "gpu"
    assert result.backend["antialiasing"] == {
        "msaa": 1,
        "supersample": samples,
        "samples_per_pixel": samples * samples,
    }
    assert result.stats["samples"] == 128 * 80 * samples * samples
    assert_parity(result, cpu)
    assert result.stats["samples_covered"] == sum(o["visible_samples"] for o in result.objects)
    # Coverage: identical, except that a sample exactly on a silhouette may fall on either
    # side (Apple's Metal rasterizer did so for 1 of 10,240 pixels): at most one sample,
    # only next to background or partially covered pixels.
    diff = np.abs(result.coverage - cpu.coverage)
    assert diff.max() <= 1 / samples**2 + 1e-6
    changed = diff > 1e-6
    near_silhouette = ~interior(np.where(cpu.coverage < 1, -1, 0))
    assert np.all(near_silhouette[changed]) and changed.mean() < 0.005


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_gpu_matches_the_independent_ray_caster(gpu, projection):
    scene = hostile_scene(projection)
    result = gpu.render(scene, width=120, height=80, samples=1, default_lights=False)
    ref_img, ref_ids, ref_depth, _, _ = reference_image(scene, scene.active_camera, 120, 80)
    mismatch = result.object_index != ref_ids
    assert np.all(boundary_mask(ref_ids)[mismatch]) and mismatch.mean() < 0.01
    inner = interior(ref_ids, result.object_index)
    assert np.abs(result.linear - ref_img).max(axis=-1)[inner].max() < 2e-3
    fg = inner & (ref_ids >= 0)
    assert (np.abs(result.depth[fg] - ref_depth[fg]) / ref_depth[fg]).max() < 1e-3


def test_msaa_antialiases_edges_and_agrees_with_the_cpu_inside(gpu):
    scene = hostile_scene()
    cpu = scene.render(width=160, height=100, samples=2)
    for msaa in [m for m in gpu.msaa if m > 1]:
        result = gpu.render(scene, width=160, height=100, msaa=msaa)
        assert result.backend["antialiasing"]["msaa"] == msaa
        inner = interior(cpu.object_index, result.object_index)
        inner &= result.object_index == cpu.object_index
        diff = np.abs(result.image.astype(int) - cpu.image.astype(int)).max(axis=-1)
        variation = local_variation(cpu.image)
        smooth = inner & (variation <= 8)
        assert smooth.sum() > 0.8 * inner.sum() and diff[smooth].max() <= 2
        # MSAA shades once per pixel: highlights and creases differ within their variation.
        assert np.all(diff[inner] <= np.maximum(2, variation[inner] // 2 + 2))
        # Edge pixels mix their objects: fractional coverage exists only along silhouettes.
        partial = (result.coverage > 0) & (result.coverage < 1)
        assert partial.any() and np.all(boundary_mask(cpu.object_index)[partial])
        assert set(np.round(result.coverage.ravel() * msaa, 4)) <= set(range(msaa + 1))
        for go, co in zip(result.objects, cpu.objects, strict=True):
            fraction = go["visible_samples"] / (160 * 100 * msaa)
            assert fraction == pytest.approx(co["visible_samples"] / (160 * 100 * 4), abs=0.01)


def test_default_antialiasing_uses_samples_squared_per_pixel(gpu):
    scene = hostile_scene()
    for samples, expected in ((1, (1, 1)), (2, (4, 1)), (3, (1, 3)), (4, (4, 2))):
        result = gpu.render(scene, width=40, height=24, samples=samples)
        aa = result.backend["antialiasing"]
        assert (aa["msaa"], aa["supersample"]) == expected
        assert aa["samples_per_pixel"] == samples * samples
        assert result.settings["samples"] == samples


def analytic_quad(color="#808080", light=None, ambient=0.0, projection="orthographic"):
    """A square facing the camera, filling the middle of the view."""
    scene = Scene("Quad")
    plane = scene.add_plane("Quad", size=2)
    plane.rotation_euler = (90, 0, 0)  # +Z normal turned toward -Y, the camera side
    plane.color = color
    scene.world.configure(background="#000000", ambient_intensity=ambient)
    scene.add_camera(
        "Cam", projection=projection, ortho_scale=4, fov=60, position=(0, -5, 0), look_at=(0, 0, 0)
    )
    if light is not None:
        scene.add_directional_light("Sun", direction=light)
    return scene


@pytest.mark.parametrize("angle", [0, 30, 60, 89])
def test_lambert_shading_matches_the_formula(gpu, angle):
    rad = np.radians(angle)
    scene = analytic_quad(light=(np.sin(rad), np.cos(rad), 0.0), ambient=0.0)
    result = gpu.render(scene, width=32, height=32, samples=1)
    base = (0x80 / 255 + 0.055) / 1.055
    base = base**2.4
    center = result.linear[16, 16]
    assert np.allclose(center, base * np.cos(rad), rtol=1e-5, atol=1e-7)


def test_point_light_falloff_range_and_specular_peak(gpu):
    scene = analytic_quad(color="#ffffff")
    scene.get_by_name("Quad").appearance.configure(specular=0.5, shininess=50)
    scene.add_point_light("Bulb", position=(0, -2, 0), intensity=4, range=10)
    result = gpu.render(scene, width=33, height=33, samples=1)
    # The pixel at the center sees the quad's center: distance 2 along the normal,
    # where light, view and normal align: diffuse I/d^2 * window plus specular.
    window = (1 - (2 / 10) ** 4) ** 2
    expected = 4 / 4 * window * (1 + 0.5)
    assert result.linear[16, 16] == pytest.approx([expected] * 3, rel=1e-4)
    off = result.linear[16, 30]  # farther from the light: dimmer
    assert np.all(off < result.linear[16, 16])


def test_exposure_background_and_srgb_encoding(gpu):
    scene = analytic_quad(color="#404040", ambient=1.0)
    scene.world.background = "#336699"
    plain = gpu.render(scene, width=16, height=16, samples=1)
    brighter = gpu.render(scene, width=16, height=16, samples=1, exposure=1.5)
    assert plain.pixel(0, 0) == (0x33, 0x66, 0x99)  # the background code round-trips exactly
    assert np.allclose(brighter.linear[8, 8], plain.linear[8, 8] * 2**1.5, rtol=1e-6)
    from aether3d.rendering.color import linear_to_srgb, quantize

    # The GPU encodes 8-bit sRGB exactly as the CPU renderer would from the same values.
    assert np.array_equal(brighter.image, quantize(linear_to_srgb(brighter.linear)))
    assert np.array_equal(plain.image, quantize(linear_to_srgb(plain.linear)))


def test_projection_places_known_points_on_known_pixels(gpu):
    # Orthographic 4 units across 64 pixels: the 2-unit quad covers exactly columns 16-47.
    result = gpu.render(analytic_quad(ambient=1.0), width=64, height=64, samples=1)
    covered = result.object_index == 0
    assert (
        covered[:, 16:48][16:48].all() and not covered[:, :16].any() and not covered[:, 48:].any()
    )
    assert not covered[:16].any() and not covered[48:].any()
    # Orthographic depth comes from a float32 depth buffer spanning near..far (0.1..1000):
    # its absolute resolution is (far - near) / 2^24, about 6e-5 units.
    assert result.depth[32, 32] == pytest.approx(5.0, abs=4 * 1000 / 2**24)
    persp = gpu.render(analytic_quad(ambient=1.0, projection="perspective"), width=64, height=64)
    # A 60-degree view at distance 5 is 2 * 5 * tan(30) = 5.774 units across.
    width = (persp.object_index[32] == 0).sum()
    assert width == pytest.approx(64 * 2 / (2 * 5 * np.tan(np.radians(30))), abs=1.0)


def test_every_geometry_source_renders_like_the_cpu(gpu):
    scene = Scene("Sources")
    scene.add_plane("Ground", size=30).color = "#5a8f3c"
    imported = import_gltf(DATA / "khronos" / "Box" / "glTF-Binary" / "Box.glb").scene
    box = next(o for o in imported if o.mesh is not None)
    scene.add_object("Imported", mesh=box.mesh, position=(-3, 0, 1)).color = "#aa5533"
    edited = scene.add_cube("Edited", size=1.5, position=(0, 0, 0.75))
    with edited.edit_mesh() as edit:
        top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
        edit.extrude_faces(top, 1.0)
    scene.procedural.add_parameter("count", 3, min=1, max=8)
    row = scene.add_geometry("Row", "box", width=0.6, depth=0.6, height=0.6)
    row.add_modifier("array", count=Expr("count"), relative_offset=(1.6, 0, 0))
    scene.add_object("Procedural", geometry=row, position=(2, 0, 0.3)).color = "#3366cc"
    empty = scene.add_object("Empty", position=(0, 3, 0))
    scene.add_object("Child", mesh=create_sphere(0.5, 16, 8), parent=empty).color = "#dddd33"
    scene.add_cube("Hidden", size=3, position=(0, -2, 1)).visible = False
    scene.add_camera("Cam", position=(3, -9, 5), look_at=(0.5, 0, 0.5), fov=50)
    scene.add_directional_light("Sun", direction=(0.3, 0.6, -1), intensity=1.5)
    cpu = scene.render(width=120, height=80, samples=2)
    result = gpu.render(scene, width=120, height=80, samples=2, supersample=2)
    assert_parity(result, cpu)
    visible = {o["name"] for o in result.objects if o["visible_samples"] > 0}
    assert visible == {"Ground", "Imported", "Edited", "Procedural", "Child"}


def test_procedural_changes_render_immediately_and_nothing_is_modified(gpu):
    scene = Scene()
    scene.procedural.add_parameter("count", 2, min=1, max=6)
    row = scene.add_geometry("Row", "box", width=0.8, depth=0.8, height=0.8)
    row.add_modifier("array", count=Expr("count"), relative_offset=(1.5, 0, 0))
    obj = scene.add_object("Row", geometry=row)
    scene.add_camera("Cam", position=(3, -10, 4), look_at=(3, 0, 0), fov=50)
    document = json.dumps(scene.to_dict(), sort_keys=True)
    two = gpu.render(scene, width=96, height=48, samples=1)
    scene.procedural.set_parameter("count", 5)
    five = gpu.render(scene, width=96, height=48, samples=1)
    assert two.objects[0]["triangles"] == 24 and five.objects[0]["triangles"] == 60
    assert five.stats["pixels_covered"] > two.stats["pixels_covered"]
    assert five.backend["resources"]["mesh_uploads"] == 1  # the new evaluation, uploaded
    assert_parity(five, scene.render(width=96, height=48, samples=1))
    scene.procedural.set_parameter("count", 2)
    again = gpu.render(scene, width=96, height=48, samples=1)
    assert np.array_equal(again.image, two.image)
    assert obj.is_procedural and [m.name for m in row.modifiers] == ["Array"]
    assert json.dumps(scene.to_dict(), sort_keys=True) == document


def test_undo_and_redo_render_exactly_on_the_gpu_with_warm_caches(gpu):
    scene = Scene()
    scene.procedural.add_parameter("count", 2, min=1, max=6)
    row = scene.add_geometry("Row", "box", width=0.8, depth=0.8, height=0.8)
    row.add_modifier("array", count=Expr("count"), relative_offset=(1.5, 0, 0))
    scene.add_object("Row", geometry=row)
    scene.add_sphere(name="Ball", position=(1, -2, 1))
    scene.add_camera("Cam", position=(3, -10, 4), look_at=(3, 0, 0), fov=50)
    two = gpu.render(scene, width=96, height=48, samples=1)
    scene.procedural.set_parameter("count", 5)
    scene.get_by_name("Ball").mesh = create_cube(1.5)
    scene.get_by_name("Ball").color = "#ff2020"
    five = gpu.render(scene, width=96, height=48, samples=1)
    scene.history.undo(3)
    again = gpu.render(scene, width=96, height=48, samples=1)
    assert np.array_equal(again.image, two.image)
    assert again.backend["resources"]["mesh_uploads"] == 0  # the old meshes are still cached
    scene.history.redo(3)
    assert np.array_equal(gpu.render(scene, width=96, height=48, samples=1).image, five.image)


def test_meshes_are_uploaded_once_and_shared_by_instances(gpu):
    scene = Scene()
    mesh = create_sphere(0.4, 16, 8)
    for i in range(10):
        for j in range(10):
            scene.add_object(f"Ball {i} {j}", mesh=mesh, position=(i, j, 0))
    scene.add_camera("Cam", position=(4.5, -12, 9), look_at=(4.5, 4.5, 0))
    first = gpu.render(scene, width=64, height=48, samples=1)
    resources = first.backend["resources"]
    assert first.stats["meshes"] == 1 and first.stats["instances"] == 100
    assert first.stats["draw_calls"] == 1  # one instanced draw for all hundred
    second = gpu.render(scene, width=64, height=48, samples=1)
    assert second.backend["resources"]["mesh_uploads"] == 0
    assert np.array_equal(second.image, first.image)
    assert resources["mesh_cache"]["meshes"] >= 1


def test_freed_meshes_leave_the_gpu_cache(gpu):
    import gc

    scene = Scene()
    obj = scene.add_object("Thing", mesh=create_sphere(1.0, 32, 16))
    scene.add_camera("Cam", position=(0, -6, 0), look_at=(0, 0, 0))
    gpu.render(scene, width=32, height=32, samples=1)
    context, _ = gpu_device.get_context(gpu.adapter)
    positions = obj.mesh.positions
    entries = context.meshes._entries.values()
    serials = {serial for refs, _, serial in entries if refs[0]() is positions}
    assert len(serials) == 1  # the sphere is resident
    del positions
    obj.mesh = create_cube(1.0)  # the scene no longer uses the sphere...
    gc.collect()
    result = gpu.render(scene, width=32, height=32, samples=1)
    assert result.backend["resources"]["mesh_uploads"] == 1  # the cube
    # ... but its history does (undo brings the sphere back), so the sphere stays cached.
    assert serials & {serial for _, _, serial in context.meshes._entries.values()}
    scene.history.clear()  # now nothing refers to the sphere
    gc.collect()
    result = gpu.render(scene, width=32, height=32, samples=1)
    assert result.backend["resources"]["mesh_uploads"] == 0
    assert not serials & {serial for _, _, serial in context.meshes._entries.values()}


@pytest.mark.parametrize("cull", ["back", "front"])
def test_culling_with_mirrored_instances_matches_the_cpu(gpu, cull):
    scene = hostile_scene()
    cpu = scene.render(width=128, height=80, samples=1, cull=cull)
    result = gpu.render(scene, width=128, height=80, samples=1, cull=cull, msaa=1)
    assert_parity(result, cpu)


def test_hard_cases_camera_inside_clipping_and_extreme_coordinates(gpu):
    room = Scene()
    room.add_cube("Room", size=10).color = "#c8b89a"
    room.add_sphere("Ball", radius=1, position=(0, 3, 0)).color = "#3355cc"
    room.add_camera("Cam", position=(0, -2, 0), look_at=(0, 3, 0), near=0.1, far=50, fov=90)
    room.add_point_light("Lamp", position=(1, 0, 3), intensity=10)
    inside = gpu.render(room, width=80, height=60, samples=1)
    assert inside.stats["pixels_covered"] == 80 * 60
    assert_parity(inside, room.render(width=80, height=60, samples=1))

    far = Scene()
    far.add_plane("Huge", size=1e7).color = "#557755"
    far.add_cube("Far", size=1e4, position=(0, 1e6, 5e3)).color = "#aa3333"
    far.add_cube("Tiny", size=1e-4, position=(0, -1, 0))
    cam = far.add_camera("Cam", position=(0, -10, 2), near=1e-2, far=1e7)
    cam.look_at((0, 0, 0))
    result = gpu.render(far, width=64, height=48)
    assert np.isfinite(result.linear).all()
    assert {o["name"] for o in result.objects if o["visible_samples"]} >= {"Huge", "Far"}

    offset = Scene()  # a detailed object a million units from the origin
    offset.add_cube("Crate", size=1, position=(1e6, 1e6, 0)).color = "#cc6633"
    offset.add_camera("Cam", position=(1e6 + 2, 1e6 - 3, 1.5), look_at=(1e6, 1e6, 0))
    near_origin = Scene()
    near_origin.add_cube("Crate", size=1, position=(0, 0, 0)).color = "#cc6633"
    near_origin.add_camera("Cam", position=(2, -3, 1.5), look_at=(0, 0, 0))
    a = gpu.render(offset, width=64, height=48, samples=1)
    b = gpu.render(near_origin, width=64, height=48, samples=1)
    # Camera-relative transforms: far from the origin renders like near it.
    assert (a.object_index != b.object_index).mean() < 0.01
    assert np.abs(a.image.astype(int) - b.image.astype(int)).max() <= 2


def test_degenerate_triangles_and_objects_outside_the_view(gpu):
    from aether3d import Mesh

    scene = Scene()
    flat = Mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1]], [[0, 1, 2], [0, 0, 3], [0, 1, 3]])
    scene.add_object("Degenerate", mesh=flat)
    scene.add_cube("Behind", size=1, position=(0, -20, 0))
    scene.add_cube("Partly", size=4, position=(3.5, 5, 0))
    scene.add_camera("Cam", position=(0, -8, 0), look_at=(0, 0, 0))
    cpu = scene.render(width=64, height=48, samples=1)
    result = gpu.render(scene, width=64, height=48, samples=1)
    assert_parity(result, cpu)
    seen = {o["name"]: o["visible_samples"] for o in result.objects}
    assert seen["Behind"] == 0 and seen["Partly"] > 0 and seen["Degenerate"] > 0


def test_thousands_of_objects(gpu):
    scene = Scene()
    cube = create_cube(0.7)
    for i in range(50):
        for j in range(50):
            obj = scene.add_object(f"C{i}_{j}", mesh=cube, position=(i - 25, j - 25, (i * j) % 3))
            obj.color = (i / 50, j / 50, 0.5)
    scene.add_directional_light("Sun", direction=(-0.3, 0.5, -1))
    cam = scene.add_camera("Cam", fov=40)
    cam.frame(direction=(-0.5, 1, -0.8), aspect=16 / 9)
    cpu = scene.render(width=160, height=90, samples=1)
    result = gpu.render(scene, width=160, height=90, samples=1)
    assert result.stats["instances"] == 2500 and result.stats["draw_calls"] == 1
    assert_parity(result, cpu)


def test_lighting_without_lights_and_without_geometry(gpu):
    lone = Scene()
    lone.add_cube("Crate")
    lone.add_camera("Cam", position=(3, -4, 2), look_at=(0, 0, 0))
    default = gpu.render(lone, width=48, height=32, samples=1)
    assert "DEFAULT_LIGHTING" in [d.code for d in default.diagnostics]
    assert_parity(default, lone.render(width=48, height=32, samples=1))
    dark = gpu.render(lone, width=48, height=32, samples=1, default_lights=False)
    assert "NO_LIGHTS" in [d.code for d in dark.diagnostics]
    empty = Scene()
    empty.world.background = "#123456"
    empty.add_camera("Cam")
    nothing = gpu.render(empty, width=8, height=8)
    assert "NOTHING_TO_RENDER" in [d.code for d in nothing.diagnostics]
    assert np.all(nothing.image == (0x12, 0x34, 0x56)) and nothing.objects == []


def test_results_are_deterministic_saveable_and_json(gpu, tmp_path):
    from PIL import Image

    scene = hostile_scene()
    a = gpu.render(scene, width=96, height=64)
    b = gpu.render(scene, width=96, height=64)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.linear, b.linear)
    path = a.save(tmp_path / "gpu.png")
    with Image.open(path) as png:  # an independent decoder
        assert png.size == (96, 64) and png.mode == "RGB"
        assert np.array_equal(np.asarray(png), a.image)
    report = json.loads(json.dumps(a.to_dict()))
    assert report["backend"]["name"] == "gpu" and report["backend"]["api"] == "wgpu"
    assert report["backend"]["adapter"]["name"] == gpu.name
    timings = report["stats"]["timings"]
    for key in ("evaluate", "extract", "init", "upload", "prepare", "gpu", "readback", "total"):
        assert timings[key] >= 0
    if gpu.info["timestamps"]:  # GPU time of each pass, measured on the GPU itself
        assert set(a.stats["gpu_timings"]) == {"scene", "finalize"}
        assert all(t > 0 for t in a.stats["gpu_timings"].values())
        shadowed = gpu.render(scene, width=96, height=64, shadows=True)
        assert set(shadowed.stats["gpu_timings"]) == {"shadow", "scene", "finalize"}
    assert a.linear.dtype == np.float32 and a.depth.dtype == np.float32
    assert a.object_index.dtype == np.int32 and a.coverage.dtype == np.float32


def test_render_errors_leave_no_image(gpu, tmp_path):
    scene = hostile_scene()
    with pytest.raises(RenderError) as info:
        gpu.render(scene, width=64, height=48, shadows={"lights": ["Nope"]})
    assert info.value.code == "INVALID_SHADOWS"
    assert list(tmp_path.iterdir()) == []


def bar_coverage(width, height, half_length, half_width, angle, extent, supersample=16):
    """Exact-enough area of a rotated rectangle inside each pixel (dense supersampling)."""
    n = supersample
    xs = (np.arange(width * n) + 0.5) / (width * n) * 2 - 1
    ys = 1 - (np.arange(height * n) + 0.5) / (height * n) * 2
    x, y = np.meshgrid(xs * extent * width / height, ys * extent)
    c, s = np.cos(angle), np.sin(angle)
    u, v = x * c + y * s, -x * s + y * c
    inside = (np.abs(u) <= half_length) & (np.abs(v) <= half_width)
    return inside.reshape(height, n, width, n).mean(axis=(1, 3))


def test_msaa_coverage_of_thin_diagonal_geometry_matches_its_area(gpu):
    """A bar 3/4 pixel wide at 27 degrees: MSAA tracks its true area, one sample cannot."""
    width, height, extent = 96, 64, 2.0  # orthographic: 4 units across 64 pixels vertically
    pixel = 2 * extent / height
    angle = np.radians(27.0)
    scene = Scene("Thin")
    bar = scene.add_cube("Bar", size=1)
    bar.scale = (5.0, 0.75 * pixel, 0.1)  # 5 units long, three quarters of a pixel wide
    bar.rotation_euler = (0, 0, 27.0)
    bar.color = "#ffffff"
    scene.world.configure(background="#000000", ambient_intensity=1.0)
    cam = scene.add_camera(
        "Top", projection="orthographic", ortho_scale=2 * extent, position=(0, 0, 10)
    )
    cam.look_along((0, 0, -1), up=(0, 1, 0))
    area = bar_coverage(width, height, 2.5, 0.375 * pixel, angle, extent)
    crossed = area > 0.02  # pixels the bar really passes through
    aliased = gpu.render(scene, width=width, height=height, msaa=1, default_lights=False)
    assert (aliased.coverage[crossed] == 0).mean() > 0.2  # one sample per pixel: gaps
    aliased_error = np.abs(aliased.coverage - area)[crossed].mean()
    for count in [m for m in gpu.msaa if m >= 4]:
        result = gpu.render(scene, width=width, height=height, msaa=count, default_lights=False)
        assert set(np.round(result.coverage.ravel() * count, 4)) <= set(range(count + 1))
        assert (result.coverage[area > 0.5] > 0).all()  # no gaps where the bar really is
        error = np.abs(result.coverage - area)[crossed]
        assert error.mean() < aliased_error / 2 and error.mean() < 0.15
        # White on black in linear light: the pixel's brightness is its coverage.
        assert np.allclose(result.linear[..., 0], result.coverage, atol=1e-3)


def test_saved_projects_render_on_both_backends_without_changes(gpu, tmp_path):
    """A version 2 project (Aether3D 0.4.0) renders on CPU and GPU; nothing in it changes."""
    import importlib.util as util

    if util.find_spec("manifold3d") is None:
        pytest.skip("the castle uses booleans (manifold3d)")
    from aether3d import Scene as _Scene

    source = DATA / "v0.4.0" / "castle.aether"
    scene = _Scene.load(source)
    before = json.dumps(scene.to_dict(), sort_keys=True)
    stacks = {g.name: [m.name for m in g.modifiers] for g in scene.procedural.geometries}
    options = {"width": 128, "height": 72, "samples": 2, "frame": True}
    cpu = scene.render(**options)
    parity = gpu.render(scene, supersample=2, msaa=1, **options)
    msaa = gpu.render(scene, shadows=True, **options)
    assert_parity(parity, cpu)
    assert msaa.stats["pixels_covered"] > 0.2 * 128 * 72
    assert json.dumps(scene.to_dict(), sort_keys=True) == before  # rendering changed nothing
    scene.save(tmp_path / "castle.aether")
    again = _Scene.load(tmp_path / "castle.aether")
    assert {g.name: [m.name for m in g.modifiers] for g in again.procedural.geometries} == stacks
    assert all(o.is_procedural for o in again if o.name in stacks)
    reparity = gpu.render(again, supersample=2, msaa=1, **options)
    assert np.array_equal(reparity.image, parity.image)
