"""The operations history: undo, redo, transactions, checkpoints, limits and persistence.

The central check: every kind of change, undone, gives back exactly the project
document it started from -- the same JSON, through ``scene.to_dict()`` -- and the same
Python objects; redone, exactly the document it produced. The test suite runs with
``history.STRICT`` (conftest.py), so every undo, redo and rollback also re-checks the
scene's internal indexes, and a change recorded outside an operation fails.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

import aether3d as a3
from aether3d import HistoryError, Scene, create_cube, create_sphere
from aether3d import history as history_module


def doc(scene: Scene) -> str:
    return json.dumps(scene.to_dict(), sort_keys=True)


def catalog_scene() -> Scene:
    """A scene with a bit of everything; its history starts empty."""
    s = Scene("Catalog")
    cube = s.add_cube(name="Cube", size=2, position=(0, 0, 1))
    s.add_sphere(name="Ball", parent=cube, position=(0, 0, 2))
    s.add_plane(name="Ground", size=20)
    s.add_camera(name="Cam", position=(8, -8, 5), look_at=(0, 0, 0))
    s.add_point_light(name="Lamp", position=(2, 2, 6))
    stone = s.create_material("Stone", base_color=(0.5, 0.5, 0.5))
    s.assign_material("Cube", stone)
    s.add_procedural_texture("Dots", {"type": "noise", "scale": 3})
    s.add_procedural("Tower", "cylinder", radius=1, depth=4, segments=8, position=(5, 0, 0))
    s.procedural.add_parameter("height", 3.0, min=1, max=10)
    s.history.clear()
    return s


def by(scene: Scene, name: str):
    return scene.get_by_name(name)


def _edit_mesh(s: Scene) -> None:
    with by(s, "Ground").edit_mesh() as e:
        e.extrude_faces(e.select_faces(normal=((0, 0, 1), 1)), 1.0)


def _transaction(s: Scene) -> None:
    with s.transaction("Several changes"):
        s.add_cube(name="A")
        by(s, "A").position = (1, 1, 1)
        s.remove(by(s, "Lamp"))
        s.world.background = "#000000"


CHANGES = {
    # objects and hierarchy
    "add a cube": lambda s: s.add_cube(name="New"),
    "add an object under a parent": lambda s: s.add_object(
        "M", mesh=create_cube(1), parent=by(s, "Cube")
    ),
    "remove with children": lambda s: s.remove(by(s, "Cube")),
    "remove, keeping children": lambda s: s.remove(by(s, "Cube"), recursive=False),
    "remove the active camera": lambda s: s.remove(by(s, "Cam")),
    "duplicate with children": lambda s: s.duplicate("Cube"),
    "duplicate a procedural object": lambda s: s.duplicate("Tower"),
    "duplicate a camera": lambda s: s.duplicate("Cam"),
    "rename": lambda s: setattr(by(s, "Cube"), "name", "Box"),
    "unparent": lambda s: by(s, "Ball").set_parent(None),
    "reparent keeping the world transform": lambda s: by(s, "Ball").set_parent(
        by(s, "Ground"), keep_world_transform=True
    ),
    # transforms
    "position": lambda s: setattr(by(s, "Ball"), "position", (1, 2, 3)),
    "euler rotation": lambda s: setattr(by(s, "Ball"), "rotation_euler", (10, 20, 30)),
    "scale": lambda s: setattr(by(s, "Cube"), "scale", 3),
    "transform object": lambda s: setattr(by(s, "Cube"), "transform", a3.Transform((1, 1, 1))),
    "transform in place": lambda s: setattr(by(s, "Cube").transform, "position", (7, 0, 0)),
    "world position": lambda s: by(s, "Ball").set_world_position((4, 4, 4)),
    "look at": lambda s: by(s, "Cam").look_at((1, 1, 0)),
    # appearance
    "visibility": lambda s: setattr(by(s, "Ground"), "visible", False),
    "color": lambda s: setattr(by(s, "Ground"), "color", "#ff0000"),
    "appearance object": lambda s: setattr(
        by(s, "Ground"), "appearance", a3.Appearance(color="#00ff00", specular=0.5)
    ),
    # cameras and lights
    "camera field": lambda s: setattr(by(s, "Cam").camera, "fov", 25),
    "camera fields": lambda s: by(s, "Cam").camera.configure(near=1, far=50),
    "frame a camera": lambda s: by(s, "Cam").frame(),
    "light field": lambda s: setattr(by(s, "Lamp").light, "intensity", 7),
    "no active camera": lambda s: setattr(s, "active_camera", None),
    "add an active camera": lambda s: s.add_camera(name="Cam2", active=True, position=(1, 1, 1)),
    "add a sun": lambda s: s.add_directional_light(name="Sun", direction=(0.2, 0.3, -1)),
    # world and render settings
    "background": lambda s: setattr(s.world, "background", "#123456"),
    "sky environment": lambda s: s.world.configure(environment={"type": "sky", "intensity": 2}),
    "environment field": lambda s: setattr(s.world.environment, "intensity", 3.5),
    "world object": lambda s: setattr(s, "world", a3.World(background="#abcdef")),
    "render settings": lambda s: s.render_settings.configure(width=640, height=360),
    "render settings object": lambda s: setattr(
        s, "render_settings", a3.RenderSettings(width=100, height=50)
    ),
    "scene name": lambda s: setattr(s, "name", "Renamed"),
    # meshes
    "mesh": lambda s: setattr(by(s, "Ground"), "mesh", create_sphere(1)),
    "mesh edit session": _edit_mesh,
    # materials and textures
    "preset material": lambda s: s.create_material("Wood", preset="wood"),
    "material parameters": lambda s: s.materials["Stone"].configure(roughness=0.2),
    "material parameter": lambda s: setattr(s.materials["Stone"], "roughness", 0.9),
    "material name": lambda s: setattr(s.materials["Stone"], "name", "Granite"),
    "material model": lambda s: setattr(s.materials["Stone"], "model", "simple"),
    "assign a material": lambda s: s.assign_material(["Ground", "Ball"], "Stone"),
    "unassign a material": lambda s: setattr(by(s, "Cube"), "material", None),
    "face group material": lambda s: s.assign_material("Ball", "Stone", group=1),
    "remove a used material": lambda s: s.materials.remove_material("Stone", force=True),
    "procedural texture slot": lambda s: s.materials["Stone"].set_procedural_texture(
        "base_color", "checker", scale=4
    ),
    "bind a texture": lambda s: s.materials["Stone"].set_texture("roughness", "Dots", scale=2),
    "texture name": lambda s: setattr(s.materials.texture("Dots"), "name", "Spots"),
    "texture graph": lambda s: s.materials.texture("Dots").set_params(scale=9),
    "remove a texture": lambda s: s.materials.remove_texture("Dots"),
    "colors to materials": lambda s: s.materials.convert_appearances(),
    # procedural geometry
    "add a parameter": lambda s: s.procedural.add_parameter("width", 3.0),
    "set a parameter": lambda s: s.procedural.set_parameter("height", 5.0),
    "geometry source": lambda s: s.procedural.geometry("Tower").set_source(radius=2),
    "add a modifier": lambda s: s.procedural.geometry("Tower").add_modifier("bevel", width=0.05),
    "bake": lambda s: by(s, "Tower").bake(),
    "detach geometry": lambda s: by(s, "Tower").set_geometry(None),
    "add a procedural object": lambda s: s.add_procedural("Pillar", "box", position=(0, 5, 0)),
    "procedural undo": lambda s: s.procedural.undo(),  # undoes the parameter of the setup
    # grouped
    "transaction": _transaction,
}


@pytest.mark.parametrize("name", list(CHANGES))
def test_every_kind_of_change_is_undone_and_redone_exactly(name):
    s = catalog_scene()
    objects = {o.id: o for o in s}
    materials = {m.id: m for m in s.materials.materials}
    before = doc(s)
    revision = s.revision
    CHANGES[name](s)
    after = doc(s)
    assert after != before
    assert len(s.history) == 1  # one call (or transaction), one step
    entry = s.history.entries[0]
    assert entry.label and entry.changes >= 1 and entry.paths and entry.bytes > 0
    assert s.revision > revision
    for _ in range(2):
        s.history.undo()
        assert doc(s) == before
        assert all(s.get(oid) is obj for oid, obj in objects.items())  # the same objects
        assert all(s.materials.material(mid) is m for mid, m in materials.items())
        s.history.redo()
        assert doc(s) == after
    assert len(s.history) == 1 and not s.history.can_redo  # undo/redo recorded nothing


def test_undo_brings_back_the_same_objects_handles_and_values():
    s = catalog_scene()
    cube, ball, cam = by(s, "Cube"), by(s, "Ball"), by(s, "Cam")
    camera, transform = cam.camera, cube.transform
    stone = s.materials["Stone"]
    tower = s.procedural.geometry("Tower")
    bevel = tower.add_modifier("bevel", width=0.05)
    s.remove(cube)
    s.materials.remove_material(stone, force=True)
    s.procedural.remove_modifier(tower, bevel)
    camera.fov = 20
    assert cube.scene is None and ball.scene is None and stone.library is None
    s.history.undo(4)
    assert cube.scene is s and ball.scene is s and ball.parent is cube
    assert stone.library is s.materials and cube.material is stone
    assert tower.modifiers == [bevel]  # the modifier handle the caller holds
    assert cam.camera is camera and camera.fov == 40.0  # edited in place, restored in place
    assert cube.transform is transform
    transform.position = (3, 3, 3)  # still recorded: the restored object's own transform
    assert s.history.entries[-1].label == "Set position of 'Cube'"


def test_a_long_random_chain_undoes_and_redoes_through_every_state():
    rng = random.Random(9)
    s = catalog_scene()
    states = [doc(s)]
    steps = 0

    def pick_object():
        return rng.choice(list(s))

    actions = [
        lambda: s.add_cube(position=(rng.uniform(-5, 5), 0, 0)),
        lambda: s.remove(pick_object()),
        lambda: s.duplicate(pick_object()),
        lambda: setattr(pick_object(), "position", (rng.random(), rng.random(), rng.random())),
        lambda: setattr(pick_object(), "rotation_euler", (rng.uniform(0, 90), 0, 0)),
        lambda: setattr(pick_object(), "name", f"N{rng.randrange(10**6)}"),
        lambda: pick_object().set_parent(rng.choice([None, pick_object()])),
        lambda: setattr(pick_object(), "visible", rng.random() < 0.5),
        lambda: setattr(pick_object(), "color", (rng.random(), rng.random(), rng.random())),
        lambda: s.assign_material(pick_object(), rng.choice([None, "Stone"])),
        lambda: setattr(s.materials["Stone"], "roughness", rng.random()),
        lambda: setattr(s.world, "ambient_intensity", rng.random()),
        lambda: s.procedural.set_parameter("height", rng.uniform(1, 10)),
        lambda: s.procedural.geometry("Tower").set_source(segments=rng.randrange(3, 12)),
        lambda: s.add_procedural(None, "box", position=(0, rng.uniform(-5, 5), 0)),
        lambda: setattr(pick_object(), "mesh", create_cube(rng.uniform(0.5, 2))),
    ]
    failures = 0
    while steps < 150:
        before = doc(s)
        count = len(s.history)
        try:
            rng.choice(actions)()
        except a3.AetherError:
            failures += 1  # e.g. a cycle, a camera with a mesh, a procedural object's mesh
            assert doc(s) == before and len(s.history) == count  # failures change nothing
            continue
        after = doc(s)
        if after == before:
            assert len(s.history) == count  # nothing changed, nothing recorded
            continue
        steps += 1
        states.append(after)
        assert len(s.history) == count + 1
    assert failures > 0
    for k in range(steps, 0, -1):
        assert doc(s) == states[k]
        s.history.undo()
    assert doc(s) == states[0]
    for k in range(1, steps + 1):
        s.history.redo()
        assert doc(s) == states[k]
    s.history.undo(steps)  # in one call, all or nothing
    assert doc(s) == states[0]
    s.history.redo(steps)
    assert doc(s) == states[-1]
    assert s.history.verify()["ok"]


def test_failed_operations_leave_no_step_and_no_change():
    s = catalog_scene()
    before, revision = doc(s), s.revision
    failing = [
        lambda: setattr(by(s, "Cube"), "name", "Ball"),  # name taken
        lambda: by(s, "Cube").set_parent(by(s, "Ball")),  # cycle
        lambda: setattr(by(s, "Cam").camera, "fov", 500),
        lambda: s.add_light("directional", "Sun", direction=(0, 0, 0)),
        lambda: setattr(by(s, "Cam"), "mesh", create_cube(1)),
        lambda: s.materials["Stone"].configure(roughness=7),
        lambda: s.materials.remove_material("Stone"),  # in use
        lambda: s.procedural.set_parameter("height", 50.0),  # above max
        lambda: s.add_procedural("Tower", "box"),  # name taken
        lambda: s.duplicate("No such object"),
        lambda: s.history.undo(),  # nothing to undo
    ]
    for action in failing:
        with pytest.raises(a3.AetherError):
            action()
        assert doc(s) == before and len(s.history) == 0 and s.revision == revision


def test_a_transaction_is_one_step_and_rolls_everything_back_on_failure():
    s = catalog_scene()
    before = doc(s)
    with s.transaction("Build") as tx:
        a = s.add_cube(name="A")
        a.position = (1, 2, 3)
        s.materials["Stone"].roughness = 0.1
    assert tx.entry is not None and tx.entry.label == "Build" and len(s.history) == 1
    built = doc(s)
    with pytest.raises(RuntimeError), s.transaction("Fails"):
        s.remove(by(s, "Cube"))
        s.world.background = "#ffffff"
        raise RuntimeError("stop")
    assert doc(s) == built and len(s.history) == 1
    s.history.undo()
    assert doc(s) == before and a.scene is None


def test_nested_transactions_are_savepoints():
    s = catalog_scene()
    with s.transaction("Outer"):
        by(s, "Cube").position = (1, 0, 0)
        try:
            with s.transaction("Inner"):
                by(s, "Cube").position = (2, 0, 0)
                s.remove(by(s, "Lamp"))
                raise ValueError("inner fails")
        except ValueError:
            pass
        # Only the inner block was rolled back.
        assert by(s, "Cube").position == (1.0, 0.0, 0.0) and "Lamp" in {o.name for o in s}
        by(s, "Ball").visible = False
    assert [e.label for e in s.history.entries] == ["Outer"]
    s.history.undo()
    assert by(s, "Cube").position == (0.0, 0.0, 1.0) and by(s, "Ball").visible


def test_rollback_inside_a_transaction():
    s = catalog_scene()
    before = doc(s)
    with s.transaction("Try") as tx:
        s.add_cube(name="Temp")
        tx.rollback()
        assert doc(s) == before
        by(s, "Cube").scale = 2
    assert [e.label for e in s.history.entries] == ["Try"]
    with pytest.raises(HistoryError) as exc:
        tx.rollback()
    assert exc.value.code == "TRANSACTION_CLOSED"


def test_a_new_change_discards_the_redo_branch_and_its_checkpoints():
    s = catalog_scene()
    by(s, "Cube").position = (1, 0, 0)
    s.history.checkpoint("one")
    by(s, "Cube").position = (2, 0, 0)
    s.history.checkpoint("two")
    s.history.undo()
    assert s.history.can_redo and set(s.history.checkpoints) == {"one", "two"}
    by(s, "Ball").visible = False  # a new branch
    assert not s.history.can_redo and set(s.history.checkpoints) == {"one"}
    with pytest.raises(HistoryError) as exc:
        s.history.redo()
    assert exc.value.code == "NOTHING_TO_REDO"


def test_checkpoints_go_backward_and_forward():
    s = catalog_scene()
    s.history.checkpoint("start")
    states = {"start": doc(s)}
    for i, name in enumerate(["a", "b", "c"]):
        by(s, "Cube").position = (i + 1, 0, 0)
        s.history.checkpoint(name)
        states[name] = doc(s)
    start = s.history.checkpoints["start"]  # the index of the step behind each state
    assert s.history.checkpoints == {"start": start, "a": start + 1, "b": start + 2, "c": start + 3}
    assert len(s.history.goto("a")) == 2 and doc(s) == states["a"]
    assert len(s.history.goto("c")) == 2 and doc(s) == states["c"]
    assert len(s.history.goto("start")) == 3 and doc(s) == states["start"]
    assert s.history.goto("start") == []
    with pytest.raises(HistoryError) as exc:
        s.history.goto("nowhere")
    assert exc.value.code == "UNKNOWN_CHECKPOINT" and "start" in exc.value.hint
    s.history.remove_checkpoint("b")
    assert "b" not in s.history.checkpoints


def test_limits_drop_the_oldest_steps_and_unreachable_checkpoints():
    s = catalog_scene()
    s.history.checkpoint("origin")
    s.history.max_entries = 5
    for i in range(12):
        by(s, "Cube").position = (i, 0, 0)
    assert len(s.history) == 5 and s.history.dropped == 7
    assert "origin" not in s.history.checkpoints  # its state can no longer be reached
    s.history.undo(5)
    assert by(s, "Cube").position == (6.0, 0.0, 0.0)
    with pytest.raises(HistoryError):
        s.history.undo()
    s.history.configure(max_entries=2)
    assert len(s.history) == 0 and len(s.history.redo_entries) == 5  # undo steps go first
    with pytest.raises(HistoryError) as exc:
        s.history.configure(max_entries=-1)
    assert exc.value.code == "INVALID_ARGUMENT"


def test_memory_counts_what_only_the_history_holds():
    s = Scene("Memory")
    obj = s.add_object("Big", mesh=create_sphere(1, 256, 128))
    mesh_bytes = obj.mesh.positions.nbytes + obj.mesh.triangles.nbytes
    s.history.clear()
    small = s.history.nbytes
    obj.position = (1, 0, 0)  # the step refers to no mesh
    assert s.history.nbytes - small < 2000
    obj.mesh = create_cube(1)  # the step now holds the old big mesh, used by nothing else
    held = s.history.nbytes
    assert held >= mesh_bytes
    s.history.undo()  # the big mesh is the scene's again: the history holds the cube
    assert s.history.nbytes < held
    s.history.redo()
    s.history.max_bytes = mesh_bytes // 2  # too small for the old mesh: its step goes
    assert s.history.nbytes <= mesh_bytes // 2
    assert s.history.dropped >= 1 and all(e.label != "Set mesh of 'Big'" for e in s.history.entries)


def test_disabled_history_records_nothing_and_operations_stay_atomic():
    s = catalog_scene()
    by(s, "Cube").position = (1, 0, 0)
    with s.history.disabled():
        assert len(s.history) == 0  # disabling clears: older steps could no longer be undone
        by(s, "Cube").position = (2, 0, 0)
        before = doc(s)
        with pytest.raises(RuntimeError), s.transaction("Fails"):
            s.add_cube(name="X")
            raise RuntimeError
        assert doc(s) == before  # rollback works without recording
    assert s.history.enabled and len(s.history) == 0
    by(s, "Cube").position = (3, 0, 0)
    assert len(s.history) == 1


def test_history_errors_have_codes():
    s = catalog_scene()
    cases = [
        (lambda: s.history.undo(), "NOTHING_TO_UNDO"),
        (lambda: s.history.redo(), "NOTHING_TO_REDO"),
        (lambda: s.history.undo(0), "INVALID_ARGUMENT"),
        (lambda: s.history.undo(True), "INVALID_ARGUMENT"),
        (lambda: s.history.checkpoint(""), "INVALID_ARGUMENT"),
        (lambda: s.history.remove_checkpoint("x"), "UNKNOWN_CHECKPOINT"),
        (lambda: setattr(s.history, "enabled", "yes"), "INVALID_ARGUMENT"),
    ]
    for action, code in cases:
        with pytest.raises(HistoryError) as exc:
            action()
        assert exc.value.code == code and exc.value.to_dict()["code"] == code
    by(s, "Cube").position = (1, 0, 0)
    with s.transaction("Open"):
        with pytest.raises(HistoryError) as exc:
            s.history.undo()
        assert exc.value.code == "TRANSACTION_OPEN"


def test_entries_describe_the_changes_and_serialize():
    s = catalog_scene()
    cube = by(s, "Cube")
    cube.position = (1, 0, 0)
    s.world.background = "#102030"
    first, second = s.history.entries
    assert first.paths == (f"objects[{cube.id}].transform",)
    assert second.paths == ("world",) and second.index > first.index
    data = s.history.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert [e["label"] for e in data["entries"]] == [
        "Set position of 'Cube'",
        "Set world background",
    ]
    s.history.undo()
    assert [e.label for e in s.history.redo_entries] == ["Set world background"]


def test_scripts_are_one_step_each_and_can_name_it():
    s = catalog_scene()
    a3.run_scene_script(
        s,
        {
            "format": "aether3d.scene-script",
            "version": 1,
            "label": "Set up the shot",
            "operations": [
                {"op": "add_camera", "name": "Shot", "position": [5, -5, 3], "look_at": [0, 0, 0]},
                {"op": "set_world", "background": "#223344"},
            ],
        },
    )
    a3.run_edit_script(
        s,
        {
            "format": "aether3d.edit-script",
            "version": 1,
            "edits": [
                {
                    "object": "Ground",
                    "operations": [
                        {"op": "select", "kind": "faces", "normal": [[0, 0, 1], 1], "as": "top"},
                        {"op": "extrude_faces", "faces": {"selection": "top"}, "distance": 1},
                    ],
                }
            ],
        },
    )
    a3.run_procedural_script(
        s,
        {
            "format": "aether3d.procedural-script",
            "version": 1,
            "operations": [{"op": "set_parameter", "name": "height", "value": 4.0}],
        },
    )
    labels = [e.label for e in s.history.entries]
    assert labels == ["Set up the shot", "Edit script (1 edit)", "Procedural script (1 operation)"]
    with pytest.raises(a3.ScriptError):
        a3.run_scene_script(
            s, {"format": "aether3d.scene-script", "version": 1, "label": 3, "operations": [{}]}
        )


def test_the_procedural_systems_own_undo_is_a_step_of_the_scene_history():
    s = catalog_scene()
    before = doc(s)
    s.procedural.set_parameter("height", 6.0)
    changed = doc(s)
    s.procedural.undo()  # the procedural stack's own undo, recorded as a scene step
    assert doc(s) == before
    assert [e.label for e in s.history.entries][-1] == "Undo procedural change"
    s.history.undo()  # undo the procedural undo
    assert doc(s) == changed
    s.history.undo()
    assert doc(s) == before


def test_loaded_and_imported_scenes_start_with_an_empty_history(tmp_path):
    s = catalog_scene()
    by(s, "Cube").position = (1, 2, 3)
    s.save(tmp_path / "scene.aether")
    for loaded in (Scene.load(tmp_path / "scene.aether"), Scene.from_dict(s.to_dict())):
        assert len(loaded.history) == 0 and loaded.revision == 0
    s.export(tmp_path / "scene.glb", allow_lossy=True)
    imported = Scene.import_file(tmp_path / "scene.glb", allow_lossy=True)
    assert len(imported.history) == 0
    before = doc(imported)
    target = next(o for o in imported if o.mesh is not None)
    target.scale = 2
    imported.remove(target)
    imported.history.undo(2)
    assert doc(imported) == before


def test_the_history_is_saved_with_the_project_and_replays_exactly(tmp_path):
    s = catalog_scene()
    ball = by(s, "Ball")
    s.remove(by(s, "Cube"))  # removed objects travel in the history file
    s.materials.remove_material("Stone", force=True)
    s.add_cube(name="Fresh").mesh = create_sphere(0.5)
    s.history.checkpoint("cleared")
    s.world.environment.intensity = 2.0
    s.history.undo()  # an undone step travels too
    project = tmp_path / "scene.aether"
    s.save(project, history=True)
    assert (tmp_path / "scene.aether.history").exists()
    t = Scene.load(project, history=True)
    assert doc(t) == doc(s) and len(t.history) == 4 and len(t.history.redo_entries) == 1
    assert t.history.checkpoints == s.history.checkpoints
    assert [e.label for e in t.history.entries] == [e.label for e in s.history.entries]
    t.history.redo()
    assert t.world.environment.intensity == 2.0
    t.history.undo(5)
    s.history.undo(4)
    assert doc(t) == doc(s) and by(t, "Ball").parent is by(t, "Cube")
    assert ball.parent is by(s, "Cube")
    assert t.history.verify()["ok"] and s.history.verify()["ok"]


def test_a_history_that_does_not_match_its_project_is_refused(tmp_path):
    s = catalog_scene()
    by(s, "Cube").position = (5, 5, 5)
    project = tmp_path / "scene.aether"
    s.save(project, history=True)
    s.save(tmp_path / "other.aether")  # saved without a history
    with pytest.raises(HistoryError) as exc:
        Scene.load(tmp_path / "other.aether", history=True)
    assert exc.value.code == "NO_HISTORY"
    other = Scene.load(project)
    other.name = "Changed elsewhere"
    other.save(project)  # without its history: the old one no longer matches
    with pytest.raises(HistoryError) as exc:
        Scene.load(project, history=True)
    assert exc.value.code == "HISTORY_MISMATCH"
    assert Scene.load(project).name == "Changed elsewhere"  # the project still loads


def test_a_tampered_history_file_is_refused_and_changes_nothing(tmp_path):
    s = catalog_scene()
    by(s, "Cube").position = (5, 5, 5)
    s.remove(by(s, "Lamp"))
    project = tmp_path / "scene.aether"
    s.save(project, history=True)
    file = tmp_path / "scene.aether.history"
    data = json.loads(file.read_text())
    good = json.dumps(data)
    corruptions = [
        lambda d: d["undo"][0]["changes"][0].update(aspect="object.nothing"),
        lambda d: d["undo"][1]["changes"][0].update(target="no-such-object"),
        lambda d: d.update(format_version=99),
        lambda d: d["undo"][0]["changes"][0].update(before=[[0, 0], [0, 0, 0, 1], [1, 1, 1]]),
        lambda d: d["undo"][0]["changes"][0].update(after=[[9, 9, 9], [0, 0, 0, 1], [1, 1, 1]]),
        lambda d: d["undo"][0]["changes"][0].update(before=[[0, 0, 1], [0, 0, 0, 2], [1, 1, 1]]),
    ]
    for corrupt in corruptions:
        d = json.loads(good)
        corrupt(d)
        file.write_text(json.dumps(d))
        t = Scene.load(project)
        before = doc(t)
        with pytest.raises(HistoryError) as exc:
            t.history.load(file, project_dir=tmp_path)
        assert exc.value.code in ("INVALID_HISTORY", "UNSUPPORTED_HISTORY")
        assert doc(t) == before and len(t.history) == 0
    file.write_text("{not json")
    with pytest.raises(HistoryError) as exc:
        Scene.load(project, history=True)
    assert exc.value.code == "INVALID_HISTORY"


def test_renders_after_undo_match_the_original_pixels():
    s = catalog_scene()
    options = {"width": 48, "height": 32}
    path = {"samples": 2, "bounces": 2}
    raster = s.render(**options).image
    traced = s.render(**options, backend="path", path=path).image
    s.procedural.set_parameter("height", 8.0)
    by(s, "Cube").mesh = create_sphere(1)
    s.materials["Stone"].base_color = (1.0, 0.1, 0.1)
    s.world.background = "#ffffff"
    assert not np.array_equal(s.render(**options).image, raster)
    s.history.undo(4)
    assert np.array_equal(s.render(**options).image, raster)
    assert np.array_equal(s.render(**options, backend="path", path=path).image, traced)


def test_caches_stay_valid_across_undo():
    s = catalog_scene()
    system = s.procedural
    system.mesh(system.geometry("Tower").id)
    system.geometry("Tower").set_source(segments=16)
    system.mesh(system.geometry("Tower").id)
    hits = system.stats.hits
    s.history.undo()  # the definitions of before: their evaluation is still cached
    system.mesh(system.geometry("Tower").id)
    assert system.stats.hits > hits
    version = s.materials.version
    s.materials["Stone"].roughness = 0.3
    s.history.undo()
    s.history.redo()
    assert s.materials.version > version + 1  # undo and redo are changes for renderers too


def test_duplicate_copies_hierarchies_and_shares_meshes():
    s = catalog_scene()
    cube = by(s, "Cube")
    copy = s.duplicate(cube)
    assert copy.name == "Cube.001" and copy.parent is None and copy.id != cube.id
    (ball_copy,) = copy.children
    assert ball_copy.name == "Ball.001" and ball_copy.mesh is by(s, "Ball").mesh
    assert copy.material is cube.material and copy.position == cube.position
    tower = s.duplicate("Tower", "Tower B", parent=cube, children=False)
    assert tower.geometry is by(s, "Tower").geometry and tower.parent is cube
    cam = s.duplicate("Cam")
    assert cam.camera is not by(s, "Cam").camera and cam.camera.fov == by(s, "Cam").camera.fov
    assert s.active_camera is by(s, "Cam")
    inside = s.duplicate(cube, parent=cube)  # a copy inside the original is not copied again
    assert [c.name for c in inside.children] == ["Ball.002", "Tower B.001"]
    with pytest.raises(a3.NameConflictError):
        s.duplicate(cube, "Ball")
    assert len(s.history) == 4


def test_strict_mode_reports_changes_recorded_outside_an_operation():
    s = catalog_scene()
    with pytest.raises(AssertionError):
        s.history._touch(history_module.VISIBLE, by(s, "Cube"))


def test_the_cli_undoes_and_redoes_changes_across_processes(tmp_path, capsys):
    from aether3d.cli import main

    project = tmp_path / "shot.aether"
    catalog_scene().save(project)

    def run(*argv):
        code = main([*argv, "--json"])
        out = capsys.readouterr().out
        return code, json.loads(out)

    code, data = run("history", "show", str(project))
    assert code == 1 and data["error"]["code"] == "NO_HISTORY"
    assert run("history", "start", str(project))[0] == 0
    original = doc(Scene.load(project))
    code, data = run("object", "set", str(project), "Cube", "--position", "4", "5", "6")
    assert code == 0 and data["history"]["steps"] == 1
    moved = doc(Scene.load(project))
    assert run("history", "checkpoint", str(project), "moved")[0] == 0
    code, data = run("scene", "set", str(project), "--background", "#ff0000")
    assert code == 0 and data["history"]["steps"] == 2
    code, data = run("history", "undo", str(project), "-n", "2")
    assert code == 0 and [e["label"] for e in data["undone"]] == [
        "aether3d scene set",
        "aether3d object set",
    ]
    assert doc(Scene.load(project)) == original
    code, data = run("history", "goto", str(project), "moved")
    assert code == 0 and doc(Scene.load(project)) == moved
    code, data = run("history", "verify", str(project))
    assert code == 0 and data["ok"]
    code, data = run("history", "redo", str(project), "--dry-run")
    assert code == 0 and doc(Scene.load(project)) == moved  # a dry run writes nothing
    code, data = run("history", "show", str(project))
    assert [e["label"] for e in data["entries"]] == ["aether3d object set"]
    assert [e["label"] for e in data["redo"]] == ["aether3d scene set"]
    # A project changed without its history: the old history is set aside, a new one starts.
    scene = Scene.load(project)
    scene.name = "Elsewhere"
    scene.save(project)
    code, data = run("object", "set", str(project), "Cube", "--visible", "false")
    assert code == 0 and "HISTORY_MISMATCH" in data["history"]["note"]
    assert data["history"]["steps"] == 1


def test_the_cli_starts_a_history_with_the_history_flag(tmp_path, capsys):
    from aether3d.cli import main

    project = tmp_path / "flag.aether"
    catalog_scene().save(project)
    assert main(["object", "set", str(project), "Cube", "--scale", "2", "2", "2"]) == 0
    assert not Path(str(project) + ".history").exists()  # no history unless asked
    assert main(["object", "set", str(project), "Cube", "--scale", "3", "3", "3", "--history"]) == 0
    assert Path(str(project) + ".history").exists()
    assert main(["history", "undo", str(project)]) == 0
    assert by(Scene.load(project), "Cube").scale == (2.0, 2.0, 2.0)
    capsys.readouterr()


def test_merging_another_scene_is_one_undoable_step():
    s = catalog_scene()
    other = catalog_scene()
    other.materials["Stone"].set_texture("roughness", "Dots")
    before = doc(s)
    copies = s.merge(other, parent="Ground")
    assert [c.name for c in copies] == [
        "Cube.001",
        "Ground.001",
        "Cam.001",
        "Lamp.001",
        "Tower.001",
    ]
    assert (
        all(c.parent is by(s, "Ground") for c in copies) and by(s, "Ball.001").parent is copies[0]
    )
    stone = s.materials["Stone.001"]  # copied, renamed, its texture followed
    assert copies[0].material is stone and stone.texture("roughness").name == "Dots.001"
    assert by(s, "Tower.001").geometry is None and by(s, "Tower.001").mesh is not None  # static
    assert s.active_camera is by(s, "Cam")  # this scene's active camera stays
    stone.roughness = 0.05
    assert other.materials["Stone"].roughness != 0.05  # copies are independent
    assert [e.label for e in s.history.entries] == [
        "Merge 'Catalog' (6 objects)",
        "Set roughness of material 'Stone.001'",
    ]
    s.history.undo(2)
    assert doc(s) == before
    s.history.redo(2)
    assert by(s, "Cube.001").material is stone


def test_importing_a_gltf_file_into_a_scene_can_be_undone(tmp_path):
    asset = catalog_scene()
    asset.export(tmp_path / "asset.glb", allow_lossy=True)
    s = catalog_scene()
    before = doc(s)
    imported = s.merge(tmp_path / "asset.glb", allow_lossy=True)
    assert imported and len(s.history) == 1
    with pytest.raises(FileNotFoundError):
        s.merge(tmp_path / "missing.glb")
    assert len(s.history) == 1  # a failed import changes nothing
    s.history.undo()
    assert doc(s) == before
    s.merge(s)  # a scene into itself: each object once
    assert len(s) == 2 * len(catalog_scene())


def test_the_cli_merges_a_file_into_a_project_undoably(tmp_path, capsys):
    from aether3d.cli import main

    project = tmp_path / "base.aether"
    catalog_scene().save(project)
    catalog_scene().export(tmp_path / "asset.glb", allow_lossy=True)
    original = doc(Scene.load(project))
    argv = ["scene", "merge", str(project), str(tmp_path / "asset.glb"), "--allow-lossy"]
    assert main([*argv, "--history", "--parent", "Ground"]) == 0
    merged = Scene.load(project)
    assert len(merged) > len(catalog_scene()) and by(merged, "Cube.001").parent.name == "Ground"
    assert main(["history", "undo", str(project)]) == 0
    assert doc(Scene.load(project)) == original
    capsys.readouterr()


def test_a_step_that_cannot_be_applied_changes_nothing():
    s = Scene("Conflict")
    s.add_cube(name="C")
    a = s.add_cube(name="A")
    a.name = "B"
    other = s.get_by_name("C")
    # A change outside the history (private attributes): undoing the rename would now
    # give two objects the name "A".
    del s._by_name["C"]
    other._name = "A"
    s._by_name["A"] = other
    before = doc(s)
    with pytest.raises(HistoryError) as exc:
        s.history.undo()
    assert exc.value.code == "HISTORY_CONFLICT"
    assert doc(s) == before and a.name == "B" and s.get_by_name("A") is other
    assert len(s.history) == 3  # the step is still there, not applied


def test_edits_of_a_removed_object_are_steps_too():
    s = catalog_scene()
    before = doc(s)
    lamp = by(s, "Lamp")
    s.remove(lamp)
    lamp.position = (5, 5, 5)  # undo of the removal must not bring this back unrecorded
    lamp.light.intensity = 9
    assert [e.label for e in s.history.entries][1:] == [
        "Set position of 'Lamp'",
        "Set light intensity of 'Lamp'",
    ]
    s.history.undo(3)
    assert doc(s) == before and lamp.scene is s and lamp.light.intensity == 1.0
    s.history.redo(3)
    assert lamp.scene is None and lamp.position == (5.0, 5.0, 5.0)
