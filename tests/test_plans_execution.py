"""Executing scene plans: atomicity, history, modes, idempotency, cancellation, chunks,
determinism and parity with the per-object API."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from aether3d import PlanError, Scene, create_cube, create_sphere
from aether3d.history import check_scene
from aether3d.plans import ID_NAMESPACE


def plan(**sections) -> dict:
    return {"format": "aether3d.scene-plan", "version": 1, **sections}


def village(count: int = 4, **extra) -> dict:
    """A small plan with resources, prefabs, patterns, lights, a camera and operations."""
    return plan(
        label="Village",
        materials={"red": {"base_color": "#a03020"}, "stone": {"preset": "stone"}},
        meshes={
            "block": {"primitive": "cube", "size": 2},
            "roof": {"primitive": "cone", "segments": 4},
        },
        prefabs={
            "house": {
                "tags": ["house"],
                "children": {
                    "walls": {"mesh": "@block", "position": [0, 0, 1], "material": "@stone"},
                    "roof": {"mesh": "@roof", "position": [0, 0, 3], "material": "@red"},
                },
            }
        },
        objects={
            "ground": {"mesh": {"primitive": "plane", "size": 50}, "color": "#336633"},
            "houses": {"prefab": "@house", "array": {"count": count, "offset": [5, 0, 0]}},
            "sun": {"light": {"type": "directional"}, "direction": [-0.3, 0.4, -1]},
            "cam": {
                "camera": {"fov": 50},
                "position": [10, -25, 12],
                "look_at": [8, 0, 0],
                "active": True,
            },
        },
        operations=[{"op": "update", "select": "@houses[0]", "set": {"translate": [0, 2, 0]}}],
        **extra,
    )


def _document(scene: Scene) -> dict:
    return scene.to_dict()


def _without_ids(doc: dict) -> dict:
    """A project document with object ids replaced by their names (for parity checks)."""
    objects = doc["scene"]["objects"]
    names = {o["id"]: o["name"] for o in objects}
    out = []
    for o in objects:
        o = dict(o)
        o["id"] = o["name"]
        if o.get("parent") is not None:
            o["parent"] = names[o["parent"]]
        out.append(o)
    doc = {**doc, "scene": {**doc["scene"], "objects": out}}
    if doc["scene"].get("active_camera") is not None:
        doc["scene"]["active_camera"] = names[doc["scene"]["active_camera"]]
    return doc


# --------------------------------------------------------------------------- #
# One step, all or nothing
# --------------------------------------------------------------------------- #


def test_a_plan_is_one_undoable_step():
    scene = Scene("History")
    scene.add_cube(name="Before")
    before = _document(scene)
    result = scene.execute_plan(village())
    after = _document(scene)
    assert result.status == "applied" and result.step["label"] == "Village"
    assert [e.label for e in scene.history.entries] == ["Add object 'Before'", "Village"]
    assert result.revision == {"before": 1, "after": 2}
    scene.history.undo()
    assert _document(scene) == before
    scene.history.redo()
    assert _document(scene) == after
    check_scene(scene)


@pytest.mark.parametrize(
    "breaking, code, pointer",
    [
        # a material that the library refuses (after textures were created)
        (
            {
                "textures": {"t": {"procedural": {"type": "checker"}}},
                "materials": {"bad": {"roughness": 2}},
            },
            "OPERATION_FAILED",
            "/materials/bad",
        ),
        # a geometry whose modifier fails
        (
            {
                "geometries": {
                    "g": {
                        "primitive": "box",
                        "modifiers": [{"type": "array", "params": {"count": 0}}],
                    }
                }
            },
            "OPERATION_FAILED",
            "/geometries/g/modifiers/0",
        ),
        # an operation after every object was created
        (
            {
                "operations": [
                    {
                        "op": "update",
                        "select": {"tags": ["house"], "expect": 99},
                        "set": {"visible": False},
                    }
                ]
            },
            "SELECTION_MISMATCH",
            "/operations/0/select",
        ),
        # a camera framing nothing
        (
            {
                "operations": [
                    {
                        "op": "frame_camera",
                        "camera": "@cam",
                        "targets": {"tags": ["none"], "expect": {"min": 0}},
                    }
                ]
            },
            "OPERATION_FAILED",
            "/operations/0",
        ),
    ],
)
def test_a_failing_plan_changes_nothing(breaking, code, pointer):
    scene = Scene("Atomic")
    scene.add_cube(name="Existing")
    scene.create_material("Kept")
    before, revision, steps = _document(scene), scene.revision, len(scene.history)
    doc = village()
    for key, value in breaking.items():  # add to the village's sections
        doc[key] = {**doc.get(key, {}), **value} if isinstance(value, dict) else value
    with pytest.raises(PlanError) as info:
        scene.execute_plan(doc)
    assert info.value.code == code and info.value.pointer == pointer, info.value
    assert _document(scene) == before
    assert scene.revision == revision and len(scene.history) == steps
    assert scene._procedural is None or not scene.procedural.geometries
    check_scene(scene)


def test_engine_errors_are_reported_with_their_own_code():
    scene = Scene("Errors")
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(geometries={"g": {"primitive": "box", "params": {"width": -1}}}))
    error = info.value
    assert error.code == "OPERATION_FAILED" and "ProceduralError" in error.message
    assert error.details["error"]["code"] == "PARAMETER_OUT_OF_RANGE"
    assert error.to_dict()["details"]["error"]["type"] == "ProceduralError"


def test_dry_runs_report_exactly_and_change_nothing():
    scene = Scene("Dry")
    before, revision = _document(scene), scene.revision
    doc = village(queries=[{"query": "overlaps", "objects": "@houses/walls"}])
    result = scene.execute_plan(doc, mode="dry_run")
    assert result.status == "dry_run" and result.counts["objects"] == 4 * 3 + 3
    assert result.queries[0]["result"] == []  # answered in the would-be scene
    assert _document(scene) == before and scene.revision == revision
    assert not scene.history.can_undo and len(scene) == 0
    assert result.symbols["houses"]["count"] == 4


def test_validation_against_a_scene_estimates_and_checks_references():
    scene = Scene("Validate")
    scene.add_cube(name="Ground")
    result = scene.execute_plan(village(), mode="validate")
    assert result.status == "validated" and len(scene) == 1
    estimate = result.estimate
    assert estimate["objects"] == 15 and estimate["meshes"] == 3
    assert estimate["triangles"] == 4 * (12 + 6) + 2  # walls and cone roofs, and the ground
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"x": {"parent": "Missing"}}), mode="validate")
    assert info.value.code == "UNKNOWN_REFERENCE"


def test_expected_revision():
    scene = Scene("Revision")
    scene.add_cube(name="A")
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"b": {}}), expected_revision=0)
    assert info.value.code == "REVISION_MISMATCH" and info.value.details == {
        "revision": 1,
        "expected": 0,
    }
    assert scene.execute_plan(plan(objects={"b": {}}, expected_revision=1)).ok
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(expected_revision=5), expected_revision=6)
    assert info.value.code == "INVALID_VALUE"


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_resending_a_plan_with_its_key_does_not_apply_it_twice():
    scene = Scene("Idempotent")
    doc = village(idempotency_key="village-1")
    first = scene.execute_plan(doc)
    count = len(scene)
    again = scene.execute_plan(doc)
    assert again.status == "already_applied" and len(scene) == count
    assert again.symbols == first.symbols and len(scene.history) == 1
    with pytest.raises(PlanError) as info:
        scene.execute_plan(village(count=5, idempotency_key="village-1"))
    assert info.value.code == "IDEMPOTENCY_KEY_REUSED"
    scene.history.undo()
    with pytest.raises(PlanError) as info:
        scene.execute_plan(doc)
    assert info.value.code == "IDEMPOTENCY_CONFLICT"


def test_keys_give_deterministic_ids_that_survive_other_processes(tmp_path):
    doc = village(idempotency_key="village-2")
    scene = Scene("First")
    result = scene.execute_plan(doc)
    first_id = result.symbols["houses"]["objects"][0]["id"]
    import uuid

    assert first_id == str(uuid.uuid5(ID_NAMESPACE, "village-2\x1fhouses[0]"))
    path = tmp_path / "village.aether"
    scene.save(path)
    later = Scene.load(path)  # another session: no in-process memory of the plan
    again = later.execute_plan(doc)
    assert again.status == "already_applied" and len(later) == len(scene)
    later.remove(later.get(first_id))
    with pytest.raises(PlanError) as info:
        later.execute_plan(doc)
    assert info.value.code == "IDEMPOTENCY_CONFLICT" and info.value.details["present"] > 0
    other = Scene("Other project")  # the same key elsewhere builds the same ids
    other.execute_plan(doc)
    assert other.get(first_id).name == "houses 0"


def test_plans_with_the_same_key_are_deterministic():
    a, b = Scene("A"), Scene("B")
    doc = (
        village(idempotency_key="same", objects=None) if False else village(idempotency_key="same")
    )
    a.execute_plan(doc)
    b.execute_plan(doc)
    da, db = _document(a), _document(b)
    da["scene"]["name"] = db["scene"]["name"]
    for m in da["scene"]["materials"] + db["scene"]["materials"]:
        m.pop("id", None)
    assert _without_ids(da)["scene"]["objects"] == _without_ids(db)["scene"]["objects"]
    assert [o.id for o in a] == [o.id for o in b]


# --------------------------------------------------------------------------- #
# Cancellation, timeouts and progress
# --------------------------------------------------------------------------- #


def _big(n: int = 3000) -> dict:
    return plan(
        meshes={"block": {"primitive": "cube"}},
        objects={"blocks": {"mesh": "@block", "grid": {"count": [n // 30, 30]}}},
        operations=[
            {
                "op": "update",
                "select": {"tags": ["none"], "expect": {"min": 0}},
                "set": {"visible": True},
            }
        ],
    )


def test_cancellation_rolls_back_at_a_safe_point():
    scene = Scene("Cancel")
    scene.add_cube(name="Existing")
    before = _document(scene)
    stop = threading.Event()
    stop.set()
    with pytest.raises(PlanError) as info:
        scene.execute_plan(_big(), cancel=stop)
    assert info.value.code == "CANCELLED" and _document(scene) == before
    calls = []

    def cancel_later() -> bool:
        calls.append(1)
        return len(calls) > 3  # after objects were created

    with pytest.raises(PlanError) as info:
        scene.execute_plan(_big(9000), cancel=cancel_later)
    assert info.value.code == "CANCELLED"
    assert _document(scene) == before and len(scene.history) == 1


def test_progress_is_reported_and_timeouts_roll_back():
    scene = Scene("Progress")
    seen = []
    scene.execute_plan(
        _big(9000), progress=lambda done, total, msg: seen.append((done, total, msg))
    )
    assert seen and all(a[0] <= b[0] for a, b in zip(seen, seen[1:], strict=False))
    assert seen[-1][0] == seen[-1][1] or seen[-1][0] >= 9000
    assert any("created" in m for _, _, m in seen)
    other = Scene("Timeout")
    with pytest.raises(PlanError) as info:
        other.execute_plan(_big(9000), timeout=1e-6)
    assert info.value.code == "TIMEOUT" and len(other) == 0
    doc = _big(9000)
    doc["options"] = {"timeout": 0.001}
    with pytest.raises(PlanError) as info:
        other.execute_plan(doc)
    assert info.value.code == "TIMEOUT"


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


def _chunked(on_failure: str = "keep", fail: bool = False) -> dict:
    doc = plan(
        label="Rows",
        idempotency_key="rows-1",
        options={"chunk_size": 100, "on_chunk_failure": on_failure},
        materials={"m": {"base_color": "#888888"}},
        meshes={"block": {"primitive": "cube"}},
        objects={"rows": {"mesh": "@block", "material": "@m", "grid": {"count": [35, 10]}}},
    )
    if fail:
        doc["operations"] = [{"op": "update", "select": "Missing", "set": {"visible": False}}]
    return doc


def test_chunked_plans_are_one_step_per_chunk_with_a_checkpoint():
    scene = Scene("Chunks")
    result = scene.execute_plan(_chunked())
    assert result.chunks["total"] == 4 and result.chunks["completed"] == 4
    assert [e.label for e in scene.history.entries] == [f"Rows (chunk {k}/4)" for k in range(1, 5)]
    assert result.chunks["checkpoint"] == "plan:rows-1:before"
    assert len(scene) == 350
    scene.history.goto(result.chunks["checkpoint"])
    assert len(scene) == 0 and len(scene.materials.materials) == 0


@pytest.mark.parametrize("policy", ["keep", "rollback"])
def test_a_failing_chunk_keeps_or_rolls_back_the_completed_ones(policy):
    scene = Scene("Chunk failure")
    with pytest.raises(PlanError) as info:
        scene.execute_plan(_chunked(policy, fail=True))
    report = info.value.details["chunks"]
    assert report["failed"] == 4 and report["completed"] == 3
    if policy == "keep":
        assert len(scene) == 300 and report["checkpoint"] == "plan:rows-1:before"
        resumed = scene.execute_plan(_chunked())  # the fixed plan, same key: resumes
        assert resumed.chunks["skipped"] == 3 and resumed.chunks["completed"] == 1
        assert len(scene) == 350
        assert {o.material.name for o in scene} == {"m"}
    else:
        assert report["rolled_back"] and len(scene) == 0


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def test_update_operations():
    scene = Scene("Updates")
    base = scene.add_cube(name="Base", position=(10, 0, 0))
    other = scene.add_cube(name="Other")
    crate = scene.add_cube(name="Crate", position=(1, 2, 3))
    crate.metadata = {"keep": 1, "drop": 2}
    scene.create_material("Blue", base_color="#0000ff")
    crate_changes = {
        "translate": [1, 0, 0],
        "scale_by": 2,
        "add_tags": ["moved"],
        "metadata": {"drop": None, "new": "x"},
        "name": "Crate 2",
    }
    reparent = {"parent": "Base", "keep_world_transform": True}
    blue = {"material": "Blue", "rotation_euler": [0, 0, 90]}
    nothing = {"tags": ["nothing"], "expect": {"min": 0}}
    result = scene.execute_plan(
        plan(
            operations=[
                {"op": "update", "select": "Crate", "set": crate_changes},
                {"op": "update", "select": ["Crate 2"], "set": reparent},
                {"op": "update", "select": {"names": ["Other"]}, "set": blue},
                {"op": "update", "select": nothing, "set": {"visible": False}},
            ]
        )
    )
    assert crate.name == "Crate 2" and crate.parent is base
    assert crate.world_position == pytest.approx((2, 2, 3))  # kept while re-parented
    assert crate.position == pytest.approx((-8, 2, 3)) and crate.scale == (2.0, 2.0, 2.0)
    assert crate.tags == ("moved",) and crate.metadata == {"keep": 1, "new": "x"}
    assert other.material.name == "Blue" and other.rotation_euler == pytest.approx((0, 0, 90))
    assert [r["matched"] for r in result.operations] == [1, 1, 1, 0]
    assert result.warnings[0]["code"] == "EMPTY_SELECTION"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(
            plan(operations=[{"op": "update", "select": {"kind": "mesh"}, "set": {"name": "X"}}])
        )
    assert info.value.code == "INVALID_VALUE" and info.value.operation == 0


def test_delete_operations_need_explicit_objects_or_an_expected_count():
    scene = Scene("Deletes")
    group = scene.add_object("Group")
    for k in range(3):
        scene.add_cube(name=f"Child {k}", parent=group)
    result = validate_issues(plan(operations=[{"op": "delete", "select": {"name_glob": "Child*"}}]))
    assert result[0]["code"] == "INVALID_VALUE" and result[0]["operation"] == 0
    result = scene.execute_plan(
        plan(
            operations=[
                {"op": "delete", "select": {"name_glob": "Child*", "expect": 3}, "recursive": False}
            ]
        )
    )
    assert result.operations[0]["removed"] == 3 and len(scene) == 1
    scene.history.undo()
    result = scene.execute_plan(plan(operations=[{"op": "delete", "select": "Group"}]))
    assert result.operations[0]["removed"] == 4 and len(scene) == 0


def validate_issues(doc: dict) -> list[dict]:
    from aether3d import validate_plan

    return validate_plan(doc).issues


def test_parameter_camera_and_setting_operations():
    scene = Scene("Settings")
    result = scene.execute_plan(
        plan(
            procedural={"parameters": {"width": {"value": 2.0, "min": 0.5}}},
            geometries={"wall": {"primitive": "box", "params": {"width": {"expr": "width"}}}},
            objects={
                "wall_object": {"geometry": "@wall", "name": "Wall"},
                "cam_a": {"camera": {}, "position": [0, -10, 0]},
                "cam_b": {"camera": {}, "position": [5, -10, 0], "look_at": [0, 0, 0]},
            },
            operations=[
                {"op": "set_parameters", "values": {"width": 4.0}},
                {"op": "set_active_camera", "camera": "@cam_b"},
                {"op": "frame_camera", "targets": "@wall_object"},
            ],
            world={"background": "#102030", "ambient_intensity": 0.2},
            render_settings={"width": 320, "height": 200},
        )
    )
    wall = scene.get_by_name("Wall")
    assert wall.mesh.bounds()[1][0] == pytest.approx(2.0)
    assert scene.active_camera.name == "cam_b"
    assert result.operations[2]["camera"]["name"] == "cam_b"
    assert scene.world.ambient_intensity == 0.2 and scene.render_settings.width == 320


def test_queries_answer_with_plan_references():
    scene = Scene("Queries")
    result = scene.execute_plan(
        plan(
            objects={
                "floor": {"mesh": {"primitive": "plane", "size": 10}},
                "crates": {
                    "mesh": {"primitive": "cube"},
                    "array": {"count": 2, "offset": [0.5, 0, 0]},
                    "position": [0, 0, 0.5],
                },
            },
            queries=[
                {"query": "overlaps", "objects": "@crates"},
                {"query": "distance", "a": "@crates[0]", "b": "@floor"},
                {"query": "bounds", "object": "Nope"},
            ],
        )
    )
    overlaps, distance, error = result.queries
    assert len(overlaps["result"]) == 1  # the two crates intersect
    assert distance["result"]["distance"] == pytest.approx(0)
    assert error["error"]["code"] == "OBJECT_NOT_FOUND"


# --------------------------------------------------------------------------- #
# Parity with the per-object API
# --------------------------------------------------------------------------- #


def test_a_plan_builds_what_the_per_object_api_builds():
    by_plan = Scene("Parity")
    by_plan.execute_plan(
        plan(
            materials={"red": {"base_color": "#ff2010"}},
            objects={
                "ground": {"mesh": {"primitive": "plane", "size": 20}, "color": "#40a040"},
                "crates": {
                    "mesh": {"primitive": "cube", "size": 1},
                    "grid": {"count": [3, 2], "spacing": 1.5},
                    "position": [0, 0, 0.5],
                    "rotation_euler": [0, 0, 15],
                    "material": "@red",
                },
                "ball": {
                    "mesh": {"primitive": "sphere", "radius": 0.5, "segments": 16, "rings": 8},
                    "position": [0, -2, 0.5],
                },
                "sun": {
                    "light": {"type": "directional", "intensity": 1.5},
                    "direction": [-0.3, 0.5, -1],
                },
                "cam": {"camera": {"fov": 45}, "position": [6, -9, 5], "look_at": [1.5, 0.75, 0.5]},
            },
        )
    )
    by_api = Scene("Parity")
    ground = by_api.add_plane(name="ground", size=20)
    ground.color = "#40a040"
    red = by_api.create_material("red", base_color="#ff2010")
    cube = create_cube(1.0)
    for k in range(6):
        x, y = k % 3, k // 3
        by_api.add_object(
            f"crates {k}", mesh=cube, position=(x * 1.5, y * 1.5, 0.5)
        ).rotation_euler = (0, 0, 15)
        by_api.get_by_name(f"crates {k}").material = red
    by_api.add_object("ball", mesh=create_sphere(0.5, 16, 8), position=(0, -2, 0.5))
    by_api.add_directional_light("sun", direction=(-0.3, 0.5, -1), intensity=1.5)
    by_api.add_camera("cam", fov=45, position=(6, -9, 5), look_at=(1.5, 0.75, 0.5))
    a, b = _without_ids(by_plan.to_dict()), _without_ids(by_api.to_dict())
    for doc in (a, b):
        for m in doc["scene"]["materials"]:
            m.pop("id")
        for o in doc["scene"]["objects"]:
            if o.get("material"):
                o["material"] = "red"
    assert a == b
    image_a = by_plan.render(width=96, height=64).image
    image_b = by_api.render(width=96, height=64).image
    assert np.array_equal(image_a, image_b)


def test_large_plans_scale_linearly():
    def build(n: int) -> float:
        scene = Scene("Large")
        doc = plan(
            meshes={"b": {"primitive": "cube"}},
            objects={"g": {"mesh": "@b", "grid": {"count": [n // 50, 50]}}},
        )
        start = time.perf_counter()
        scene.execute_plan(doc)
        elapsed = time.perf_counter() - start
        assert len(scene) == n
        return elapsed

    build(1000)
    small, large = min(build(2000) for _ in range(2)), min(build(16000) for _ in range(2))
    assert large < 16 * small + 0.1, (small, large)  # 8x the objects: far from 64x


def test_plans_print_nothing(capsys):
    Scene("Quiet").execute_plan(village(queries=[{"query": "describe"}]))
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_plan_built_scenes_save_and_load(tmp_path):
    scene = Scene("Saved")
    scene.execute_plan(village())
    path = tmp_path / "saved.aether"
    scene.save(path)
    loaded = Scene.load(path)
    assert loaded.to_dict() == scene.to_dict()
    assert loaded.to_dict()["format_version"] == 6


def test_gpu_draws_instances_of_a_shared_mesh_in_one_call(gpu):
    scene = Scene("Instancing")
    scene.execute_plan(
        plan(
            meshes={"block": {"primitive": "cube", "size": 0.8}},
            objects={
                "blocks": {"mesh": "@block", "grid": {"count": [20, 20, 2]}, "color": "#c08040"},
                "cam": {"camera": {}, "position": [10, -25, 20], "look_at": [10, 10, 0]},
            },
        )
    )
    result = gpu.render(scene, width=160, height=90)
    assert result.backend["name"] == "gpu"
    assert result.stats["draw_calls"] == 1  # 800 objects sharing one mesh: one instanced draw
