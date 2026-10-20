"""Corrections: the action allowlist, exact frames, policy, atomic apply, verification,
rollback, undo and redo, preview, persistence (Stage 12)."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import FeedbackError, FeedbackSession, Scene
from aether3d.feedback import CorrectionPolicy, check_actions
from aether3d.feedback.corrections import bind, compile_actions
from aether3d.plans import validate_plan
from aether3d.serialization import scene_to_dict

EXPECT = {"format": "aether3d.expectations", "version": 1}


def room() -> Scene:
    scene = Scene("Room")
    scene.add_plane("Floor", size=10)
    scene.add_cube("Table", size=1, position=(0, 0, 0.5)).tags = ["furniture"]
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, 1.35))  # floats 0.25
    scene.add_camera("Cam", position=(3, -5, 3), look_at=(0, 0, 0.6))
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=1.0)
    return scene


def session(scene, *checks, **kw):
    return FeedbackSession(
        scene, {**EXPECT, "checks": list(checks), "defaults": False}, width=160, height=90, **kw
    )


VASE = {"check": "rests_on", "objects": "Vase", "support": "Table"}


# --------------------------------------------------------------------------- #
# The allowlist
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "action",
    [
        {"action": "delete", "object": "Vase"},
        {"action": "run_python", "code": "import os"},
        {"action": "translate", "object": "Vase", "offset": [0, 0, 1], "shell": "ls"},
        {"action": "translate", "object": "Vase"},
        {"action": "translate", "object": "Vase", "offset": [0, 0, 1e9]},
        {"action": "translate", "object": "Vase", "offset": [0, 0, float("nan")]},
        {"action": "scale", "object": "Vase", "factor": 0},
        {"action": "rotate", "object": "Vase"},
        {"action": "rotate", "object": "Vase", "rotation": [0, 0, 0, 2]},
        {"action": "set_camera", "camera": "Cam"},
        {"action": "add_light", "type": "laser", "direction": [0, 0, -1]},
        {"action": "add_light", "type": "point"},
        {"action": "set_exposure", "exposure": 99},
        {"action": "frame_camera", "targets": "@vase"},
        ["translate"],
        "translate",
    ],
)
def test_actions_outside_the_allowlist_or_malformed_are_refused(action):
    with pytest.raises(FeedbackError) as exc:
        check_actions([action])
    assert exc.value.issues and exc.value.issues[0]["pointer"].startswith("/actions/0")


def test_every_allowed_action_checks_binds_and_compiles():
    scene = room()
    scene.create_material("Clay", base_color="#b5651d")
    scene.procedural.add_parameter("count", 3, min=1, max=10)
    actions = check_actions(
        [
            {"action": "translate", "object": "Vase", "offset": [0, 0, -0.25]},
            {"action": "place", "object": "Table", "position": [0.1, 0, 0.5]},
            {"action": "rotate", "object": "Table", "rotation_euler": [0, 0, 10]},
            {"action": "scale", "object": "Table", "factor": [1, 1, 1.1]},
            {"action": "set_visibility", "object": "Floor", "visible": True},
            {"action": "assign_material", "object": "Table", "material": "Clay"},
            {"action": "set_color", "object": "Vase", "color": "#336699"},
            {"action": "frame_camera", "camera": "Cam", "targets": ["Table", "Vase"]},
            {"action": "aim_camera", "camera": "Cam", "look_at": "Vase"},
            {"action": "dolly_camera", "camera": "Cam", "distance": 0.5},
            {"action": "set_camera", "camera": "Cam", "fov": 35, "near": 0.05},
            {"action": "set_light", "light": "Sun", "intensity": 1.5},
            {
                "action": "add_light",
                "type": "point",
                "name": "Fill",
                "position": [1, -2, 2],
                "intensity": 3,
            },
            {"action": "set_exposure", "exposure": 0.5},
            {"action": "set_ambient", "intensity": 0.2},
            {"action": "set_parameter", "name": "count", "value": 4},
        ]
    )
    bound = [bind(scene, a) for a in actions]
    compiled = compile_actions(bound, "all", scene.revision)
    plan, script = compiled["plan"], compiled["scene_script"]
    assert validate_plan(plan, scene).ok
    assert [op["op"] for op in script["operations"]] == ["set_camera", "set_light", "add_light"]
    assert plan["render_settings"] == {"exposure": 0.5} and plan["world"] == {
        "ambient_intensity": 0.2
    }
    fs = session(scene)
    with pytest.raises(FeedbackError) as exc:  # procedural changes are off by default
        fs.apply(actions=actions, verify=False)
    assert "allow_procedural" in exc.value.message
    result = fs.apply(actions=actions, verify=False, policy={"allow_procedural": True})
    assert result.status == "applied" and result.step["label"].startswith("Feedback")
    assert scene.get_by_name("Table").material.name == "Clay"
    assert scene.get_by_name("Cam").camera.fov == 35
    assert scene.get_by_name("Fill").light.intensity == 3
    assert scene.procedural.parameters["count"].value == 4
    assert scene.render_settings.exposure == 0.5


def test_binding_refuses_what_cannot_apply():
    scene = room()
    scene.create_material("Clay")
    scene.assign_material("Table", "Clay")
    cases = [
        ({"action": "translate", "object": "Nope", "offset": [0, 0, 1]}, "OBJECT_NOT_FOUND"),
        ({"action": "assign_material", "object": "Vase", "material": "Gold"}, "MATERIAL_NOT_FOUND"),
        ({"action": "set_color", "object": "Table", "color": "#ff0000"}, "MATERIAL_ASSIGNED"),
        ({"action": "set_light", "light": "Vase", "intensity": 2}, "NOT_A_LIGHT"),
        ({"action": "aim_camera", "camera": "Vase", "look_at": [0, 0, 0]}, "NOT_A_CAMERA"),
        ({"action": "set_camera", "camera": "Cam", "near": 50, "far": 10}, "INVALID_VALUE"),
        ({"action": "set_parameter", "name": "nope", "value": 1}, "PARAMETER_NOT_FOUND"),
        (
            {"action": "add_light", "type": "point", "name": "Vase", "position": [0, 0, 1]},
            "NAME_CONFLICT",
        ),
    ]
    for action, code in cases:
        (a,) = check_actions([action])
        with pytest.raises(FeedbackError) as exc:
            bind(scene, a)
        assert exc.value.code == code, action


@pytest.mark.parametrize("mirror", [1.0, -1.0])
def test_world_offsets_are_exact_under_rotated_scaled_parents(mirror):
    scene = room()
    parent = scene.add_object("Rig", position=(1, 2, 0.5))
    parent.rotation_euler = (20, -35, 60)
    parent.scale = (2.0 * mirror, 0.5, 3.0)
    child = scene.add_cube("Child", size=0.3, parent=parent, position=(0.2, -0.1, 0.4))
    before = np.array(child.world_position)
    offset = np.array([0.123, -0.456, 0.789])
    fs = session(scene)
    fs.apply(
        actions=[{"action": "translate", "object": "Child", "offset": offset.tolist()}],
        verify=False,
    )
    assert np.allclose(np.array(child.world_position) - before, offset, atol=1e-12)
    target = [3.0, -1.0, 2.0]
    fs.apply(actions=[{"action": "place", "object": "Child", "position": target}], verify=False)
    assert np.allclose(child.world_position, target, atol=1e-12)
    local = child.position
    fs.apply(
        actions=[{"action": "translate", "object": "Child", "offset": [0, 0, 1], "frame": "local"}],
        verify=False,
    )
    assert np.allclose(child.position, (local[0], local[1], local[2] + 1), atol=1e-12)


def test_a_zero_scale_parent_refuses_world_offsets():
    scene = room()
    parent = scene.add_object("Flat", position=(0, 0, 0))
    parent.scale = (1, 1, 0)
    scene.add_cube("Child", size=0.3, parent=parent)
    (a,) = check_actions([{"action": "translate", "object": "Child", "offset": [0, 0, 1]}])
    with pytest.raises(FeedbackError) as exc:
        bind(scene, a)
    assert exc.value.code == "SINGULAR_TRANSFORM"


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def test_policy_limits_and_categories():
    scene = room()
    fs = session(scene)
    with pytest.raises(FeedbackError) as exc:
        fs.apply(actions=[{"action": "translate", "object": "Vase", "offset": [0, 0, 50]}])
    assert exc.value.code == "NOT_ALLOWED" and "max_translation" in exc.value.message
    with pytest.raises(FeedbackError) as exc:
        fs.apply(actions=[{"action": "set_exposure", "exposure": 6}])
    assert "max_exposure_change" in exc.value.message
    with pytest.raises(FeedbackError) as exc:
        fs.apply(actions=[{"action": "rotate", "object": "Vase", "rotation_euler": [0, 0, 170]}])
    assert "max_rotation" in exc.value.message
    with pytest.raises(FeedbackError) as exc:
        fs.apply(
            actions=[{"action": "dolly_camera", "distance": 0.2}], policy={"allow_camera": False}
        )
    assert "allow_camera" in exc.value.message
    with pytest.raises(FeedbackError) as exc:
        fs.apply(
            actions=[
                {"action": "translate", "object": o, "offset": [0, 0, 0.01]}
                for o in ("Vase", "Table", "Floor")
            ],
            policy={"max_objects": 2},
        )
    assert "max_objects" in exc.value.message
    with pytest.raises(FeedbackError) as exc:
        CorrectionPolicy.coerce({"auto_apply": ["approval"], "max_scale": 0.5, "nope": 1})
    assert {i["pointer"] for i in exc.value.issues} == {
        "/policy/auto_apply/0",
        "/policy/max_scale",
        "/policy/nope",
    }
    assert scene.get_by_name("Vase").position == (0.2, 0.1, 1.35)  # nothing changed


# --------------------------------------------------------------------------- #
# Proposals and applying them
# --------------------------------------------------------------------------- #


def test_a_safe_proposal_applies_as_one_exact_undo_step():
    scene = room()
    fs = session(scene, VASE)
    vase = scene.get_by_name("Vase")
    doc_before = scene_to_dict(scene)
    corrections = fs.propose()
    (p,) = corrections.proposals
    assert p.safety == "safe" and p.policy["auto"] and p.compiled["plan"]["expected_revision"]
    assert p.expected[0]["before"] == pytest.approx(0.25, abs=1e-6) and p.expected[0]["after"] == 0
    steps = len(scene.history)
    result = fs.apply(corrections)
    assert result.status == "applied" and len(scene.history) == steps + 1
    assert result.verification["decision"] == "kept"
    assert result.verification["targeted_resolved"] == list(p.findings)
    assert vase.position[2] == pytest.approx(1.1, abs=1e-6)
    doc_after = scene_to_dict(scene)
    scene.history.undo()
    assert scene_to_dict(scene) == doc_before and scene.get_by_name("Vase") is vase
    scene.history.redo()
    assert scene_to_dict(scene) == doc_after


def test_applying_refuses_changed_scenes_and_unapproved_proposals():
    scene = room()
    fs = session(scene, VASE, {"check": "no_overlap", "objects": "Table", "against": "Vase"})
    scene.get_by_name("Vase").position = (0.6, 0.1, 1.0)  # floating 0 but into the table edge
    corrections = fs.propose()
    with pytest.raises(FeedbackError) as exc:
        fs.apply(corrections, expected_revision=corrections.revision - 1)
    assert exc.value.code == "REVISION_MISMATCH"
    scene.get_by_name("Floor").tags = ["ground"]
    with pytest.raises(FeedbackError) as exc:
        fs.apply(corrections)
    assert exc.value.code == "STALE_PROPOSALS"
    with pytest.raises(FeedbackError) as exc:
        fs.apply(fs.propose(), ["P99"])
    assert exc.value.code == "UNKNOWN_PROPOSAL"


def test_one_change_per_target_per_step():
    scene = room()
    scene.get_by_name("Vase").position = (0.2, 0.1, 1.06)  # sunken 0.04
    fs = session(
        scene,
        VASE,
        {"check": "upright", "objects": "Vase"},
        policy={"auto_apply": ["safe", "verify"]},
    )
    scene.get_by_name("Vase").rotation_euler = (8, 0, 0)
    corrections = fs.propose()
    assert len(corrections.proposals) == 2
    result = fs.apply(corrections)
    assert len(result.applied) == 1
    assert result.rejected and "same target" in result.rejected[0]["reasons"][0]


def test_a_correction_that_breaks_a_passing_check_is_undone():
    scene = room()
    scene.add_cube("Lamp", size=0.4, position=(2, 0, 0.2))
    fs = session(
        scene,
        {"check": "no_overlap", "objects": "Lamp", "against": "Table"},
        {"check": "rests_on", "objects": "Lamp", "support": "Floor"},
    )
    before = scene_to_dict(scene)
    steps, revision = len(scene.history), scene.revision
    result = fs.apply(actions=[{"action": "translate", "object": "Lamp", "offset": [-1.6, 0, 0]}])
    assert result.status == "rolled_back"
    assert "regression" in result.reason
    assert [n["code"] for n in result.verification["new_failures"]] == ["INTERSECTION"]
    assert scene_to_dict(scene) == before
    assert scene.revision > revision  # undo is a change of revision, like any
    assert len(scene.history) == steps and scene.history.can_redo
    result = fs.apply(
        actions=[{"action": "translate", "object": "Lamp", "offset": [-1.6, 0, 0]}],
        policy={"rollback_on_regression": False},
    )
    assert result.status == "applied"


def test_an_ineffective_proposal_is_undone():
    scene = room()
    fs = session(scene, VASE, policy={"auto_apply": ["safe", "verify"]})
    corrections = fs.propose()
    (p,) = corrections.proposals
    # Sabotage the proposal: a move that leaves the vase floating as high.
    p.actions = ({"action": "translate", "object": p.actions[0]["object"], "offset": [0.05, 0, 0]},)
    result = fs.apply(corrections)
    assert result.status == "rolled_back" and "ineffective" in result.reason


def test_preview_changes_nothing():
    scene = room()
    fs = session(scene, VASE)
    before = scene_to_dict(scene)
    revision, steps = scene.revision, len(scene.history)
    corrections = fs.propose()
    preview = fs.preview(corrections)
    assert preview["comparison"]["resolved"] == list(corrections.proposals[0].findings)
    assert preview["after"]["errors"] == 0 and preview["observation"].stale
    assert 0 < preview["timings"]["diagnose"] <= preview["timings"]["total"]
    assert scene_to_dict(scene) == before
    assert scene.revision == revision and len(scene.history) == steps
    assert fs.apply(corrections).status == "applied"  # the proposals are still current


def test_corrections_keep_identity_hierarchy_procedural_nodes_and_materials(tmp_path):
    scene = room()
    clay = scene.create_material("Clay")
    tower = scene.add_procedural("Tower", "cylinder", radius=0.2, depth=1.0, segments=8)
    tower.position = (2, 2, 0.4)  # sunk 0.1
    tower.material = clay
    cap = scene.add_cube("Cap", size=0.1, parent=tower, position=(0, 0, 0.55))
    fs = session(scene, {"check": "rests_on", "objects": "Tower", "support": "Floor"})
    ids = {o.name: o.id for o in scene}
    node = tower.geometry
    cap_world = np.array(cap.world_position)
    result = fs.apply(fs.propose())
    assert result.status == "applied"
    assert tower.position[2] == pytest.approx(0.5, abs=1e-6)
    assert tower.geometry is node and tower.material is clay and cap.parent is tower
    assert np.allclose(np.array(cap.world_position) - cap_world, (0, 0, 0.1), atol=1e-6)
    assert {o.name: o.id for o in scene} == ids
    path = tmp_path / "room.aether"
    scene.save(path, history=True)
    loaded = Scene.load(path, history=True)
    assert scene_to_dict(loaded, tmp_path) == scene_to_dict(scene, tmp_path)
    assert loaded.history.entries[-1].label.startswith("Feedback:")
    loaded.history.undo()
    assert loaded.get_by_name("Tower").position[2] == pytest.approx(0.4)


def test_agent_actions_are_kept_unless_they_regress():
    scene = room()
    fs = session(scene, VASE)
    result = fs.apply(actions=[{"action": "set_exposure", "exposure": 0.3}])
    assert result.status == "applied" and result.verification["decision"] == "kept"
    with pytest.raises(FeedbackError) as exc:
        fs.apply(fs.propose(), actions=[{"action": "set_exposure", "exposure": 0}])
    assert exc.value.code == "INVALID_ARGUMENT"
