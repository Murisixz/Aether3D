"""The retained render world gives exactly the render scenes of the full extraction.

``extract_render_scene`` brings the scene's retained render world up to date from the
history's change journal (``rendering/retained.py``). After every kind of change --
transforms deep in a hierarchy, structure, visibility, appearance, materials and textures
(also changed on disk), face groups, meshes, lights, cameras, procedural geometry,
transactions, rollbacks, undo and redo -- its render scene must be the full extraction's
(``_extract_full``), byte for byte, and its errors the same errors.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
import threading

import numpy as np
import pytest
from parity import assert_bitwise, assert_render_scenes_equal

import aether3d as a3
from aether3d import Mesh, Scene, _features, create_cube, create_sphere, history
from aether3d.errors import RenderError
from aether3d.rendering.png import write_png
from aether3d.rendering.scene_data import RenderCamera, _extract_full, extract_render_scene

# Some changes overflow world matrices on purpose (renders then fail, the same way).
pytestmark = [
    pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
]


def save_png(path, value: int, size: int = 4):
    arr = np.full((size, size, 4), 255, np.uint8)
    arr[..., :3] = value
    write_png(path, arr)
    return path


def grouped_cube(size: float = 1.0) -> Mesh:
    """A cube whose six faces are face groups 0-5."""
    cube = create_cube(size)
    return Mesh(
        cube.positions, cube.triangles, cube.normals, cube.uvs, groups=np.repeat(np.arange(6), 2)
    )


def rich_scene(tmp_path) -> Scene:
    s = Scene("Retained")
    s.add_image_texture("Photo", save_png(tmp_path / "photo.png", 180))
    s.add_procedural_texture(
        "Checks", {"type": "checker", "scale": 2, "color1": "#202020", "color2": "#e0e0e0"}
    )
    s.create_material("Stone", base_color=(0.5, 0.5, 0.5))
    tiles = s.create_material("Tiles", base_color="#ffffff")
    tiles.set_texture("base_color", "Photo")
    painted = s.create_material("Painted", base_color="#ffffff")
    painted.set_texture("base_color", "Checks", coordinates="world")
    base = s.add_cube(name="Base", size=2, position=(0, 0, 1))
    ball = s.add_sphere(name="Ball", parent=base, position=(0, 0, 2))
    s.add_cube(name="Cap", size=0.5, parent=ball, position=(0, 0, 1))
    s.assign_material("Base", "Stone")
    s.add_plane(name="Ground", size=20)
    grouped = s.add_object("Grouped", mesh=grouped_cube(), position=(3, 0, 1))
    s.assign_material(grouped, "Tiles", group=1)
    s.assign_material(grouped, "Painted", group=2)
    cube = create_cube(1.0)
    s.add_object("NoUV", mesh=Mesh(cube.positions, cube.triangles), position=(-3, 0, 1))
    s.assign_material("NoUV", "Tiles")
    shared = create_sphere(0.4, 8, 4)
    pivot = s.add_object("Pivot", position=(0, 4, 0))
    s.add_object("Shared1", mesh=shared, parent=pivot, position=(1, 0, 0))
    s.add_object("Shared2", mesh=shared, parent=pivot, position=(-1, 0, 0))
    s.add_point_light(name="Lamp", position=(2, 2, 6))
    s.add_directional_light(name="Sun", direction=(-0.3, 0.5, -1.0))
    s.add_camera(name="Cam2", position=(-6, -6, 4), look_at=(0, 0, 0))
    s.add_camera(name="Cam", position=(8, -8, 5), look_at=(0, 0, 0))
    s.add_procedural("Tower", "cylinder", radius=1, depth=4, segments=8, position=(5, 5, 0))
    s.procedural.add_parameter("height", 3.0, min=1, max=10)
    s.history.clear()
    return s


def outcome(extract, scene, camera, **kw):
    try:
        return extract(scene, camera, **kw), None
    except RenderError as exc:
        return None, (exc.code, exc.message, exc.location)


def compare(scene: Scene, camera=None, **kw) -> None:
    """The retained world's render scene (or error) is the full extraction's."""
    camera = camera if camera is not None else scene.active_camera
    kw.setdefault("aspect", 4 / 3)
    kept, kept_error = outcome(extract_render_scene, scene, camera, **kw)
    full, full_error = outcome(_extract_full, scene, camera, **kw)
    assert kept_error == full_error
    if full is not None:
        assert_render_scenes_equal(kept, full, "retained vs full extraction")
        if not scene.history.in_progress and all(
            _features.enabled(v)
            for v in ("AETHER3D_RETAINED_RENDER_WORLD", "AETHER3D_MATERIAL_CACHE")
        ):  # the retained world was used and is current
            world = scene._derived()["render_world"]
            assert world.revision == scene.revision


def random_actions(s: Scene, rng: random.Random, tmp_path) -> list:
    def pick():
        return rng.choice(list(s))

    def meshes():
        return [o for o in s if o.mesh is not None] or [pick()]

    def transaction():
        with s.transaction("several"):
            for obj in rng.sample(list(s), min(3, len(s))):
                obj.position = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 3))
            rng.choice(meshes()).color = (rng.random(), rng.random(), rng.random())

    def failing_transaction():
        with pytest.raises(RuntimeError), s.transaction("fails"):
            pick().position = (9.0, 9.0, 9.0)
            s.add_cube(name=f"Doomed{rng.randrange(10**6)}")
            raise RuntimeError("rolled back")

    def rewrite_image():
        path = tmp_path / "photo.png"
        save_png(path, rng.randrange(256), size=rng.choice([4, 8]))
        stamp = os.stat(path).st_mtime_ns + rng.randrange(1, 10) * 10**9
        os.utime(path, ns=(stamp, stamp))

    def huge():  # a world matrix overflows: renders fail (the same way) until undone
        obj = rng.choice(meshes())
        obj.scale = (1e300, 1.0, 1.0)
        if obj.parent is not None:
            obj.parent.scale = (1e300, 1.0, 1.0)

    def flatten():  # a singular world transform: ZERO_WORLD_SCALE
        rng.choice(meshes()).scale = (1.0, 0.0, 1.0)

    def sun_sideways():  # zero scale along the light's forward axis
        s.get_by_name("Sun").scale = (1.0, 0.0, 1.0)

    def group_material():
        s.assign_material(
            "Grouped", rng.choice([None, "Stone", "Tiles", "Painted"]), group=rng.randrange(6)
        )

    return [
        lambda: s.add_cube(position=(rng.uniform(-5, 5), rng.uniform(-5, 5), 0)),
        lambda: (
            s.add_object(f"S{rng.randrange(10**6)}", mesh=s.get_by_name("Shared1").mesh)
            if "Shared1" in [o.name for o in s]
            else None
        ),
        lambda: s.remove(pick()),
        lambda: s.duplicate(pick()),
        lambda: setattr(pick(), "position", (rng.random(), rng.random(), rng.random())),
        lambda: setattr(pick(), "rotation_euler", (rng.uniform(0, 90), rng.uniform(0, 90), 0)),
        lambda: setattr(pick(), "scale", (rng.uniform(0.5, 2), rng.uniform(-2, -0.5), 1.0)),
        lambda: setattr(pick(), "name", f"N{rng.randrange(10**6)}"),
        lambda: pick().set_parent(rng.choice([None, pick()])),
        lambda: setattr(pick(), "visible", rng.random() < 0.5),
        lambda: setattr(rng.choice(meshes()), "color", (rng.random(), rng.random(), rng.random())),
        lambda: setattr(rng.choice(meshes()).appearance, "specular", rng.random()),
        lambda: s.assign_material(pick(), rng.choice([None, "Stone", "Tiles", "Painted"])),
        group_material,
        lambda: setattr(s.materials["Stone"], "roughness", rng.random()),
        lambda: (
            setattr(s.materials["Tiles"], "name", f"Tiles{rng.randrange(100)}")
            if "Tiles" in [m.name for m in s.materials.materials]
            else None
        ),
        lambda: setattr(s.world, "background", (rng.random(), rng.random(), rng.random())),
        lambda: setattr(s.get_by_name("Lamp").light, "intensity", rng.uniform(0.1, 3)),
        lambda: setattr(s.get_by_name("Cam").camera, "fov", rng.uniform(20, 80)),
        lambda: setattr(s, "active_camera", rng.choice(["Cam", "Cam2"])),
        lambda: s.procedural.set_parameter("height", rng.uniform(1, 10)),
        lambda: s.procedural.geometry("Tower").set_source(segments=rng.randrange(3, 12)),
        lambda: setattr(rng.choice(meshes()), "mesh", create_cube(rng.uniform(0.5, 2))),
        transaction,
        failing_transaction,
        rewrite_image,
        huge,
        flatten,
        sun_sideways,
        lambda: s.history.undo() if s.history.can_undo else None,
        lambda: s.history.undo() if s.history.can_undo else None,
        lambda: s.history.redo() if s.history.can_redo else None,
    ]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_random_changes_keep_the_retained_world_exact(seed, tmp_path):
    rng = random.Random(seed)
    s = rich_scene(tmp_path)
    compare(s)
    actions = random_actions(s, rng, tmp_path)
    for step in range(60):
        with contextlib.suppress(a3.AetherError):
            rng.choice(actions)()
        compare(s)
        if step % 7 == 0:  # another camera, an auto-framed one, other render settings
            compare(s, s.get_by_name("Cam2") if "Cam2" in [o.name for o in s] else None)
            frame = s.active_camera
            if frame is not None:
                compare(
                    s,
                    RenderCamera(None, "temporary", a3.Camera(), _frame(frame)),
                    default_lights=False,
                    aspect=2.0,
                )


def _frame(camera_object):
    from aether3d.rendering.camera import frame_from_matrix

    try:
        return frame_from_matrix(np.asarray(camera_object.world_matrix))[0]
    except RenderError:
        return frame_from_matrix(np.eye(4))[0]


def test_changes_inside_an_operation_are_rendered_without_the_world(tmp_path):
    s = rich_scene(tmp_path)
    compare(s)
    world = s._derived()["render_world"]
    revision = world.revision
    with s.transaction("Open"):
        s.get_by_name("Base").position = (4.0, 0.0, 0.0)
        compare(s)  # the full extraction runs: the change is not journaled yet
        assert world.revision == revision
        moved = extract_render_scene(s, s.active_camera, aspect=4 / 3)
        assert any(i.name == "Base" and i.world[0, 3] == 4.0 for i in moved.instances)
    compare(s)
    assert s._derived()["render_world"] is world and world.revision == s.revision


def test_errors_are_the_full_extractions(tmp_path, monkeypatch):
    s = rich_scene(tmp_path)
    compare(s)
    base = s.get_by_name("Base")
    base.scale = (1e300, 1.0, 1.0)
    s.get_by_name("Ball").scale = (1e300, 1.0, 1.0)
    with pytest.raises(RenderError) as info:
        extract_render_scene(s, s.active_camera)
    assert info.value.code == "NONFINITE_TRANSFORM"
    compare(s)
    s.history.undo()
    s.history.undo()
    compare(s)
    # A failing procedural evaluation (a modifier broken at run time: the evaluation is
    # deterministic otherwise, so the world evaluates again only when something changes,
    # here the object being shown again).
    from aether3d.procedural.modifiers import MODIFIER_TYPES

    s.procedural.geometry("Tower").add_modifier("mirror")
    compare(s)
    tower = s.get_by_name("Tower")
    tower.visible = False
    compare(s)
    s.procedural.clear_cache()
    original = type(MODIFIER_TYPES["mirror"]).apply
    monkeypatch.setattr(
        type(MODIFIER_TYPES["mirror"]),
        "apply",
        lambda *a: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    tower.visible = True
    with pytest.raises(RenderError) as info:
        extract_render_scene(s, s.active_camera)
    assert info.value.code == "GEOMETRY_EVALUATION_FAILED"
    compare(s)
    monkeypatch.setattr(type(MODIFIER_TYPES["mirror"]), "apply", original)
    s.procedural.clear_cache()
    compare(s)
    # A camera transform that cannot be used.
    s.get_by_name("Cam").scale = (1.0, 0.0, 1.0)
    compare(s)


def test_switches_select_the_full_extraction(tmp_path, monkeypatch):
    s = rich_scene(tmp_path)
    monkeypatch.setenv("AETHER3D_RETAINED_RENDER_WORLD", "0")
    compare(s)
    assert "render_world" not in s._derived()
    monkeypatch.setenv("AETHER3D_RETAINED_RENDER_WORLD", "1")
    monkeypatch.setenv("AETHER3D_MATERIAL_CACHE", "0")
    extract_render_scene(s, s.active_camera)
    assert "render_world" not in s._derived()  # it needs the material cache
    monkeypatch.delenv("AETHER3D_MATERIAL_CACHE")
    compare(s)
    assert "render_world" in s._derived()


def test_a_journal_that_does_not_reach_back_rebuilds_the_world(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "JOURNAL_CHANGES", 8)
    s = rich_scene(tmp_path)
    compare(s)
    first = s._derived()["render_world"]
    with s.transaction("Many"):
        for k in range(10):
            s.add_cube(name=f"Extra{k}", position=(k, -3.0, 0.0))
    s.get_by_name("Base").position = (1.0, 2.0, 3.0)  # (journaled, but trimmed)
    s.get_by_name("Ground").color = (0.1, 0.9, 0.1)
    s.get_by_name("Shared1").visible = False
    assert s.history._changes_since(first.revision) is None
    compare(s)
    assert s._derived()["render_world"] is first  # the same world, rebuilt
    s.history.undo()
    compare(s)


def test_an_unknown_aspect_rebuilds_the_world(tmp_path, monkeypatch):
    s = rich_scene(tmp_path)
    compare(s)
    s.get_by_name("Base").position = (1.0, 1.0, 1.0)
    real = s.history._changes_since
    unknown = history._Attr("future", "_future")
    monkeypatch.setattr(s.history, "_changes_since", lambda r: (real(r) or []) + [(unknown, s)])
    compare(s)


def test_the_render_scene_shares_only_read_only_arrays(tmp_path):
    s = rich_scene(tmp_path)
    rs = extract_render_scene(s, s.active_camera)
    inst = rs.instances[0]
    for a in (inst.world, inst.normal_matrix, inst.base_color):
        assert not a.flags.writeable
    for name in ("world", "normal_matrix", "mesh", "material", "base_color"):
        assert not getattr(rs.instance_arrays, name).flags.writeable
    again = extract_render_scene(s, s.active_camera)
    assert again.instances is rs.instances and again.instance_arrays is rs.instance_arrays


def test_an_unchanged_frame_reuses_everything_and_one_move_rebuilds_one_instance(tmp_path):
    s = rich_scene(tmp_path)
    first = extract_render_scene(s, s.active_camera)
    s.get_by_name("Ground").position = (0.0, 0.0, -0.5)
    second = extract_render_scene(s, s.active_camera)
    rebuilt = [a for a, b in zip(first.instances, second.instances, strict=True) if a is not b]
    assert [i.name for i in rebuilt] == ["Ground"]
    s.get_by_name("Base").position = (0.0, 1.0, 1.0)  # moves its whole subtree
    third = extract_render_scene(s, s.active_camera)
    rebuilt = {a.name for a, b in zip(second.instances, third.instances, strict=True) if a is not b}
    assert rebuilt == {"Base", "Ball", "Cap"}
    compare(s)


def test_many_colors_over_time_do_not_grow_the_world(tmp_path):
    s = rich_scene(tmp_path)
    obj = s.get_by_name("Ground")
    for k in range(3000):
        obj.color = (k / 3000, 0.5, 0.5)
        extract_render_scene(s, s.active_camera)
    world = s._derived()["render_world"]
    assert len(world.key_info) < 2100
    compare(s)


def test_concurrent_renders_get_correct_scenes(tmp_path):
    s = rich_scene(tmp_path)
    expected = _extract_full(s, s.active_camera, aspect=4 / 3)
    errors: list[BaseException] = []

    def work():
        try:
            for _ in range(15):
                rs = extract_render_scene(s, s.active_camera, aspect=4 / 3)
                assert_render_scenes_equal(rs, expected, "concurrent render")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]


def test_retained_and_full_renders_are_identical_images(tmp_path, monkeypatch):
    s = rich_scene(tmp_path)
    kept = s.render(width=48, height=36, backend="cpu")
    s.get_by_name("Base").position = (0.5, 0.0, 1.0)
    kept = s.render(width=48, height=36, backend="cpu")
    monkeypatch.setenv("AETHER3D_RETAINED_RENDER_WORLD", "0")
    full = s.render(width=48, height=36, backend="cpu")
    for name in ("image", "linear", "depth", "object_index", "coverage"):
        assert_bitwise(getattr(kept, name), getattr(full, name), name)
    assert kept.objects == full.objects
    assert [d.to_dict() for d in kept.diagnostics] == [d.to_dict() for d in full.diagnostics]


def test_gpu_renders_are_identical_with_and_without_the_world(gpu, tmp_path, monkeypatch):
    s = rich_scene(tmp_path)
    gpu.render(s, width=48, height=36, shadows=True)
    s.get_by_name("Ball").position = (0.0, 0.5, 2.0)
    kept = gpu.render(s, width=48, height=36, shadows=True)
    monkeypatch.setenv("AETHER3D_RETAINED_RENDER_WORLD", "0")
    full = gpu.render(s, width=48, height=36, shadows=True)
    for name in ("image", "linear", "depth", "object_index", "coverage"):
        assert_bitwise(getattr(kept, name), getattr(full, name), name)
    assert kept.objects == full.objects


def test_world_matrix_arithmetic_is_the_full_extractions_for_subtrees(tmp_path):
    """A moved parent recomputes its subtree level by level; the result is bitwise the
    full stack's."""
    s = rich_scene(tmp_path)
    extract_render_scene(s, s.active_camera)
    rng = random.Random(9)
    for _ in range(20):
        obj = rng.choice([s.get_by_name(n) for n in ("Base", "Ball", "Pivot")])
        obj.rotation_euler = (rng.uniform(-180, 180), rng.uniform(-90, 90), 0.0)
        obj.scale = (rng.uniform(0.1, 3), rng.uniform(0.1, 3), rng.uniform(-3, -0.1))
        compare(s)
    world = s._derived()["render_world"]
    from aether3d.rendering.scene_data import _world_stack

    objects, stack = _world_stack(s)
    assert [id(o) for o in objects] == [id(o) for o in world.objects]
    assert_bitwise(world.world, stack, "world matrices")
    assert math.isfinite(float(np.sum(world.world)))


def test_render_scenes_narrowed_to_some_objects_have_their_own_arrays(tmp_path):
    """Spatial queries on rendered surfaces narrow the scene's render scene to the objects
    asked about; its instance arrays must describe those instances (not the whole scene's)."""
    from aether3d.spatial.index import get_index
    from aether3d.spatial.rays import rendered_scene

    s = rich_scene(tmp_path)
    index = get_index(s)
    entries = [e for e in index.entries if e.obj.name in ("Ball", "Ground", "Grouped")]
    ps, _ = rendered_scene(index, entries)
    rs = ps.rs
    assert [i.name for i in rs.instances] == ["Ball", "Ground", "Grouped"]
    arrays = rs.instance_arrays
    assert_bitwise(arrays.world, np.stack([i.world for i in rs.instances]), "narrowed arrays")
    assert arrays.mesh.tolist() == [i.mesh for i in rs.instances]
    assert rs.triangle_count == sum(rs.meshes[i.mesh].triangles.shape[0] for i in rs.instances)


def test_scenes_with_nothing_to_render(tmp_path):
    s = Scene("Empty")
    camera = s.add_camera(name="Cam", position=(0, -5, 2), look_at=(0, 0, 0))
    compare(s)  # NOTHING_TO_RENDER, the default light
    s.add_point_light(name="Lamp", position=(0, 0, 3))
    compare(s)
    cube = s.add_cube(name="Cube")
    compare(s)
    cube.visible = False
    compare(s)
    s.remove(cube)
    compare(s)
    compare(s, camera, default_lights=False)
    s.remove(s.get_by_name("Lamp"))
    compare(s, camera, default_lights=False)  # NO_LIGHTS
