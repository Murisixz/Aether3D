"""Rendering whole scenes: every geometry source, an independent reference, robustness.

The reference comparisons use ``render_reference.py``, a brute-force ray caster
that shares no code with the rasterizer. Documented tolerances:

* object visible at each pixel center: identical, except pixels whose center
  lies within the rasterizer's 1/256-pixel snapping of a silhouette (those
  must be on an object boundary in the reference image);
* depth: relative error below 1e-5;
* linear radiance: absolute error below 1e-4 (interior pixels agree to float32
  precision; up to ~1e-5 occurs where barycentrics are clamped at edges) --
  far below one 8-bit display step.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from render_reference import reference_image  # noqa: E402

from aether3d import (  # noqa: E402
    Expr,
    RenderError,
    RenderLimits,
    Scene,
    create_sphere,
    import_gltf,
)

DATA = Path(__file__).parent / "data"


def boundary_mask(ids: np.ndarray) -> np.ndarray:
    h, w = ids.shape
    pad = np.pad(ids, 1, mode="edge")
    mask = np.zeros_like(ids, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            mask |= pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] != ids
    return mask


def crease_mask(normals: np.ndarray) -> np.ndarray:
    """Pixels whose 3x3 neighbourhood (in the reference) sees surfaces of different normals."""
    h, w, _ = normals.shape
    pad = np.pad(normals, ((1, 1), (1, 1), (0, 0)), mode="edge")
    mask = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            other = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            mask |= np.einsum("ijk,ijk->ij", other, normals) < 1 - 1e-9
    return mask


def compare_with_reference(scene: Scene, width: int = 120, height: int = 80) -> dict:
    result = scene.render(width=width, height=height, samples=1, default_lights=False)
    ref_img, ref_ids, ref_depth, _, ref_normals = reference_image(
        scene, scene.active_camera, width, height
    )
    ours = result.object_index
    mismatch = ours != ref_ids
    assert np.all(boundary_mask(ref_ids)[mismatch]), "a mismatch away from silhouettes"
    assert mismatch.mean() < 0.01
    same = ~mismatch & (ref_ids >= 0)
    rel = np.abs(result.depth[same] - ref_depth[same]) / ref_depth[same]
    assert rel.max() < 1e-5
    # A pixel center exactly on an edge between faces of one object may be given to
    # either face (both are correct); only such crease pixels may differ in radiance.
    error = np.abs(result.linear - ref_img).max(axis=-1)
    shading_mismatch = ~mismatch & (error >= 1e-4)
    assert np.all(crease_mask(ref_normals)[shading_mismatch]), "a shading error away from edges"
    assert shading_mismatch.mean() < 0.01
    return {"result": result, "mismatches": int(mismatch.sum()), "hits": int((ref_ids >= 0).sum())}


def hostile_scene(projection: str = "perspective") -> Scene:
    """Non-uniform scale, a mirrored shared mesh, a scaled rotated parent, near clipping."""
    scene = Scene("Hostile")
    scene.add_plane("Ground", size=200).color = "#5a8f3c"  # reaches behind the camera
    cube = scene.add_cube("Cube", size=1.5, position=(-1.2, 0.5, 0.75))
    cube.rotation_euler = (0, 0, 30)
    cube.color = "#c0392b"
    ball = create_sphere(0.6, 24, 12)
    a = scene.add_object("Ellipsoid", mesh=ball, position=(1.2, 0.0, 0.9), scale=(1.8, 0.7, 1.1))
    a.color = "#2e86c1"
    a.appearance.configure(specular=0.6, shininess=40)
    scene.add_object(
        "Mirrored", mesh=ball, position=(0.2, 2.2, 0.7), scale=(-1, 1, 1)
    ).color = "#f1c40f"
    pivot = scene.add_object("Pivot", position=(0, 3.5, 0.2), scale=(2, 1, 1))
    pivot.rotation_euler = (0, 0, 45)
    child = scene.add_cube("Child", size=0.6, parent=pivot, position=(0.6, 0, 0.5))
    child.rotation_euler = (10, 20, 0)
    child.color = "#8e44ad"
    if projection == "perspective":
        scene.add_camera(
            "Cam", position=(3.5, -5.5, 2.4), look_at=(0, 1, 0.6), fov=50, near=0.05, far=500
        )
    else:
        scene.add_camera(
            "Cam",
            projection="orthographic",
            ortho_scale=8,
            position=(3.5, -5.5, 4),
            look_at=(0, 1, 0.6),
            near=0,
            far=100,
        )
    scene.add_directional_light("Sun", direction=(-0.4, 0.7, -0.9), intensity=1.6, color="#fff0d8")
    scene.add_point_light("Bulb", position=(0.5, -1.5, 1.8), intensity=3, color="#ffd0a0", range=8)
    scene.world.ambient_intensity = 0.15
    return scene


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_matches_independent_ray_caster(projection):
    out = compare_with_reference(hostile_scene(projection))
    assert out["hits"] > 5000
    result = out["result"]
    assert result.stats["clipped"] >= 1  # the ground plane crosses the near plane
    names = {o["name"]: o["visible_samples"] for o in result.objects}
    assert all(names[n] > 0 for n in ("Ground", "Cube", "Ellipsoid", "Mirrored", "Child"))


def test_camera_inside_geometry_and_near_clipping():
    scene = Scene()
    room = scene.add_cube("Room", size=10)  # the camera stands inside, seeing back faces
    room.color = "#c8b89a"
    scene.add_sphere("Ball", radius=1, position=(0, 3, 0)).color = "#3355cc"
    scene.add_camera("Cam", position=(0, -2, 0), look_at=(0, 3, 0), near=0.1, far=50, fov=90)
    scene.add_point_light("Lamp", position=(1, 0, 3), intensity=10)
    out = compare_with_reference(scene, 80, 60)
    assert out["result"].stats["pixels_covered"] == 80 * 60  # walls everywhere, no holes


def test_rotated_camera_hierarchy_matches_reference():
    scene = hostile_scene()
    scene.remove(scene.get_by_name("Cam"))
    rig = scene.add_object("Rig", position=(0, -4, 2), scale=(0.5, 2.0, 1.5))
    rig.rotation_euler = (15, -10, 35)
    cam = scene.add_camera("Cam", parent=rig, fov=60, near=0.05)
    cam.look_at((0, 1, 0.5), up=(0.2, 0.0, 1.0))
    compare_with_reference(scene)


def test_occlusion_does_not_depend_on_scene_order():
    def build(order):
        scene = Scene()
        boxes = {"Near": (0, -1, 0), "Far": (0.4, 2, 0.3)}
        for name in order:
            scene.add_cube(name, size=1.5, position=boxes[name]).color = (
                "#ff0000" if name == "Near" else "#0000ff"
            )
        scene.add_camera("Cam", position=(0, -6, 0), look_at=(0, 0, 0))
        return scene.render(width=64, height=48)

    a, b = build(["Near", "Far"]), build(["Far", "Near"])
    assert np.array_equal(a.image, b.image)
    center = a.pixel(32, 24)
    assert center[0] > 100 and center[2] < 30  # the near (red) box hides the far one


def test_perspective_foreshortening_versus_orthographic():
    def widths(projection):
        # Both cubes are centered on the view axis horizontally (stacked vertically), so
        # their widest rows show exactly their front faces.
        scene = Scene()
        scene.add_cube("Near", size=1, position=(0, -2, -1.0))
        scene.add_cube("Far", size=1, position=(0, 4, 1.2))
        cam = scene.add_camera(
            "Cam", projection=projection, position=(0, -8, 0), fov=40, ortho_scale=6
        )
        cam.look_along((0, 1, 0))
        r = scene.render(width=300, height=300, samples=1)
        return [int((r.object_index == k).sum(axis=1).max()) for k in (0, 1)]

    near, far = widths("perspective")
    # Front faces at distances 5.5 and 11.5: apparent widths scale with 1 / distance.
    assert near / far == pytest.approx(11.5 / 5.5, rel=0.03)
    near_o, far_o = widths("orthographic")
    assert near_o == far_o


def test_objects_behind_the_camera_or_hidden_do_not_render():
    scene = Scene()
    scene.add_cube("Behind", size=2, position=(0, 5, 0))
    hidden = scene.add_cube("Hidden", size=2, position=(0, -5, 0))
    hidden.visible = False
    scene.add_camera("Cam", position=(0, 0, 0))  # looks along -Y
    scene.add_cube("Seen", size=1, position=(0, -6, 0))
    r = scene.render(width=32, height=32)
    seen = {o["name"]: o["visible_samples"] for o in r.objects}
    assert seen == {"Behind": 0, "Seen": seen["Seen"]} and seen["Seen"] > 0


def test_extreme_coordinates_render_without_errors():
    scene = Scene()
    scene.add_plane("Huge", size=1e7).color = "#557755"
    scene.add_cube("Far", size=1e4, position=(0, 1e6, 5e3)).color = "#aa3333"
    scene.add_cube("Tiny", size=1e-4, position=(0, -1, 0))
    cam = scene.add_camera("Cam", position=(0, -10, 2), near=1e-2, far=1e7)
    cam.look_at((0, 0, 0))
    r = scene.render(width=64, height=48)
    assert np.isfinite(r.linear).all() and r.stats["pixels_covered"] > 0
    assert {o["name"] for o in r.objects if o["visible_samples"]} >= {"Huge", "Far"}


def test_shared_mesh_instances_have_their_own_transform_color_and_visibility():
    scene = Scene()
    mesh = create_sphere(0.8, 24, 12)
    colors = {"Red": "#ff0000", "Green": "#00ff00", "Blue": "#0000ff"}
    for i, (name, color) in enumerate(colors.items()):
        scene.add_object(name, mesh=mesh, position=(i * 2.0 - 2.0, 0, 0)).color = color
    scene.get_by_name("Green").visible = False
    scene.add_camera("Cam", position=(0, -8, 0), look_at=(0, 0, 0))
    r = scene.render(width=96, height=48, samples=1)
    assert r.stats["meshes"] == 1 and r.stats["instances"] == 2  # one mesh, two instances
    names = [o["name"] for o in r.objects]
    assert names == ["Red", "Blue"]
    red = r.image[r.object_index == 0].mean(axis=0)
    blue = r.image[r.object_index == 1].mean(axis=0)
    assert red[0] > 4 * max(red[1], red[2]) and blue[2] > 4 * max(blue[0], blue[1])
    # Each instance appears where its own transform puts it.
    cols = [np.nonzero(r.object_index == k)[1].mean() for k in (0, 1)]
    assert cols[0] < 48 < cols[1]
    assert np.all(r.object_index[:, 40:56] == -1)  # the hidden middle one is not drawn


def test_gltf_imported_meshes_render():
    scene = import_gltf(DATA / "khronos" / "Box" / "glTF-Binary" / "Box.glb").scene
    r = scene.render(width=64, height=64, frame=True)
    assert r.camera["temporary"] and "CAMERA_FRAMED" in [d.code for d in r.diagnostics]
    assert r.stats["pixels_covered"] > 400
    orientation = import_gltf(
        DATA / "khronos" / "OrientationTest" / "glTF-Binary" / "OrientationTest.glb",
        allow_lossy=True,
    ).scene
    cam = orientation.add_camera("Cam", fov=45)
    cam.frame(direction=(0.3, 1.0, -0.4), aspect=100 / 70)
    out = compare_with_reference(orientation, 100, 70)
    assert out["hits"] > 1000


def test_edited_meshes_render_their_committed_geometry():
    scene = Scene()
    crate = scene.add_cube("Crate", size=2)
    scene.add_camera("Cam", position=(4, -6, 4), look_at=(0, 0, 0.5))
    before = scene.render(width=64, height=48, samples=1)
    with crate.edit_mesh() as edit:
        top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
        edit.extrude_faces(top, 1.5)
    after = scene.render(width=64, height=48, samples=1)
    assert after.stats["pixels_covered"] > before.stats["pixels_covered"]
    compare_with_reference(scene, 64, 48)


def test_procedural_objects_render_the_current_evaluation_without_side_effects():
    scene = Scene()
    scene.procedural.add_parameter("count", 2, min=1, max=6)
    row = scene.add_geometry("Row", "box", width=0.8, depth=0.8, height=0.8)
    row.add_modifier("array", count=Expr("count"), relative_offset=(1.5, 0, 0))
    obj = scene.add_object("Row", geometry=row)
    scene.add_camera("Cam", position=(3, -10, 4), look_at=(3, 0, 0), fov=50)
    document = json.dumps(scene.to_dict(), sort_keys=True)
    two = scene.render(width=96, height=48, samples=1)
    scene.procedural.set_parameter("count", 5)
    five = scene.render(width=96, height=48, samples=1)
    assert two.objects[0]["triangles"] == 24 and five.objects[0]["triangles"] == 60
    assert five.stats["pixels_covered"] > two.stats["pixels_covered"]
    # Rendering bakes nothing and changes nothing.
    assert obj.is_procedural and [m.name for m in row.modifiers] == ["Array"]
    scene.procedural.set_parameter("count", 2)
    assert json.dumps(scene.to_dict(), sort_keys=True) == document
    assert np.array_equal(scene.render(width=96, height=48, samples=1).image, two.image)


def test_failing_procedural_geometry_is_reported():
    scene = Scene()
    bad = scene.add_geometry("Bad", "box")
    obj = scene.add_object("Bad", geometry=bad)
    scene.add_camera("Cam", position=(0, -5, 0))
    from aether3d.procedural.modifiers import MODIFIER_TYPES

    original = type(MODIFIER_TYPES["mirror"]).apply
    bad.add_modifier("mirror")
    scene.procedural.clear_cache()
    try:
        type(MODIFIER_TYPES["mirror"]).apply = lambda *a: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        with pytest.raises(RenderError) as info:
            scene.render(width=8, height=8)
    finally:
        type(MODIFIER_TYPES["mirror"]).apply = original
    assert info.value.code == "GEOMETRY_EVALUATION_FAILED"
    assert info.value.location == "objects[Bad]" and "MODIFIER_FAILED" in info.value.message
    obj.visible = False
    assert scene.render(width=8, height=8).stats["instances"] == 0


def test_rendering_is_deterministic():
    scene = hostile_scene()
    a = scene.render(width=80, height=60)
    b = scene.render(width=80, height=60)
    assert np.array_equal(a.image, b.image) and a.png_bytes() == b.png_bytes()
    assert np.array_equal(a.linear, b.linear)


def test_resource_limits_and_invalid_settings():
    scene = hostile_scene()
    with pytest.raises(RenderError) as info:
        scene.render(width=20000, height=10)
    assert info.value.code == "INVALID_SETTINGS"
    with pytest.raises(RenderError) as info:
        scene.render(width=4000, height=4000, samples=4)
    assert info.value.code == "RESOURCE_LIMIT" and "samples" in info.value.message
    with pytest.raises(RenderError) as info:
        scene.render(width=64, height=64, limits=RenderLimits(max_triangles=100))
    assert info.value.code == "RESOURCE_LIMIT"
    for bad in (
        {"samples": 5},
        {"samples": 0},
        {"cull": "sideways"},
        {"exposure": float("nan")},
        {"width": 0},
        {"width": 1.5},
    ):
        with pytest.raises(RenderError) as info:
            scene.render(**bad)
        assert info.value.code == "INVALID_SETTINGS"


def test_width_or_height_alone_keeps_the_shape():
    scene = hostile_scene()
    scene.get_by_name("Cam").camera.aspect_ratio = 2.0
    r = scene.render(width=100)
    assert (r.width, r.height) == (100, 50)
    scene.get_by_name("Cam").camera.aspect_ratio = None
    r = scene.render(height=90)  # saved settings are 1280x720 (16:9)
    assert (r.width, r.height) == (160, 90)
    r = scene.render(width=64, height=64)
    assert "ASPECT_MISMATCH" not in [d.code for d in r.diagnostics]
    scene.get_by_name("Cam").camera.aspect_ratio = 2.0
    r = scene.render(width=64, height=64)
    assert "ASPECT_MISMATCH" in [d.code for d in r.diagnostics]


def test_render_result_report_and_save(tmp_path):
    scene = hostile_scene()
    r = scene.render(width=64, height=40)
    report = r.to_dict()
    json.dumps(report)
    assert report["output"] is None and report["camera"]["name"] == "Cam"
    assert set(report["stats"]["timings"]) >= {
        "evaluate",
        "extract",
        "geometry",
        "raster",
        "shade",
        "output",
        "total",
    }
    assert report["camera"]["vertical_fov"] == pytest.approx(50)
    path = r.save(tmp_path / "out.png")
    assert path.read_bytes() == r.png_bytes() and r.to_dict()["output"] == str(path)
    with pytest.raises(RenderError) as info:
        r.save(tmp_path / "out.jpg")
    assert info.value.code == "INVALID_OUTPUT"
    with pytest.raises(RenderError) as info:
        r.save(tmp_path / "missing-dir" / "out.png")
    assert info.value.code == "OUTPUT_ERROR"
    assert not (tmp_path / "missing-dir").exists()
