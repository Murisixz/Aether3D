"""Animation for agents: scripts (every operation, atomicity, limits, cancellation, fuzzed
input, the schema), the CLI, scene plans, MCP tools and the feedback loop."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
from test_animation_motion import humanoid

from aether3d import FeedbackSession, Scene
from aether3d.animation.schema import animation_script_schema
from aether3d.animation.script import (
    OPERATIONS,
    AnimationScriptError,
    check_script,
    run_animation_script,
)
from aether3d.cli import main
from aether3d.plans import execute_plan
from aether3d.serialization import scene_to_dict

FORMAT = {"format": "aether3d.animation-script", "version": 1}


def stage() -> Scene:
    s = Scene("Stage")
    s.add_plane("Floor", size=12)
    s.add_sphere("Ball", position=(0, 0, 1))
    s.add_cube("Box", position=(2, 0, 0.5))
    humanoid(s)
    cam = s.add_camera("Cam", position=(4, -6, 2), look_at=(0, 0, 1))
    s.add_camera("Wide", position=(0, -12, 4), look_at=(0, 0, 1))
    s.active_camera = cam
    return s


def script(*ops) -> dict:
    return {**FORMAT, "operations": list(ops)}


BROAD = script(
    {"op": "timeline", "fps": 24, "start": 0, "end": 4},
    {"op": "add_marker", "name": "hit", "time": 0.5},
    {"op": "sections", "sections": [{"name": "intro", "start": 0, "end": 2}]},
    {"op": "keyframe", "target": "Ball", "path": "transform.position", "time": 0,
     "value": [0, 0, 1]},
    {"op": "keyframe", "target": "Ball", "path": "transform.position", "time": 1,
     "value": [2, 0, 1], "interpolation": "sine", "ease": "in_out"},
    {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
     "keys": [[0, 1], [0.5, 0.2], [1, 1]], "extrapolation": "cycle"},
    {"op": "edit_keys", "target": "Ball", "path": "transform.position.z", "edit": "scale_time",
     "factor": 2},
    {"op": "add_clip", "name": "Spin"},
    {"op": "keyframe", "clip": "Spin", "path": "transform.rotation_euler.z", "time": 0,
     "value": 0},
    {"op": "keyframe", "clip": "Spin", "path": "transform.rotation_euler.z", "time": 1,
     "value": 90},
    {"op": "add_layer", "target": "Box", "name": "base"},
    {"op": "add_strip", "target": "Box", "layer": "base", "clip": "Spin", "repeat": 2},
    {"op": "generate", "armature": "Hero", "motion": "wave", "name": "Hello"},
    {"op": "add_constraint", "owner": "Cam", "constraint": {"name": "aim", "type": "look_at",
                                                             "target": {"object": "Ball"}}},
    {"op": "add_path", "name": "Track", "points": [[0, -6, 2], [4, -4, 2], [6, 0, 2]]},
    {"op": "follow_path", "object": "Wide", "path": "Track", "start": 0, "end": 4},
    {"op": "shots", "shots": [{"camera": "Wide", "start": 0, "end": 2},
                              {"camera": "Cam", "start": 2, "end": 4}]},
    {"op": "camera_move", "camera": "Cam", "move": "zoom", "amount": 1.5, "start": 2, "end": 4},
    {"op": "simplify", "target": "Ball", "path": "transform.position.x", "tolerance": 1e-3},
    {"op": "report", "target": "Ball", "path": "transform.position.z"},
    {"op": "diagnose"},
    {"op": "sample", "target": "Ball", "path": "transform.position.z", "times": [0, 0.5, 1]},
    {"op": "set_time", "time": 1.5},
)  # fmt: skip


def test_a_broad_script_applies_as_one_step():
    s = stage()
    out = run_animation_script(s, BROAD)
    assert [o["op"] for o in out["operations"]] == [o["op"] for o in BROAD["operations"]]
    names = {c["name"] for c in out["animation"]["clips"]}
    assert {"Ball Action", "Spin", "Hello"} <= names
    report = next(o for o in out["operations"] if o["op"] == "report")["result"]
    assert report["keys"] == 3 and report["range"] == [0.0, 2.0]  # (scaled by 2)
    sample = next(o for o in out["operations"] if o["op"] == "sample")["result"]
    assert sample[0] == pytest.approx(1.0)
    assert s.animation.time == 1.5
    json.dumps(out)
    assert s.history.can_undo
    s.history.undo()
    assert s.animation.clips() == []


@pytest.mark.parametrize(
    "bad, code, where",
    [
        ({"op": "keyframe", "target": "Nope", "path": "transform.position.x", "time": 0,
          "value": 1}, "TARGET_MISSING", "operations[1].target"),
        ({"op": "set_channel", "target": "Ball", "path": "transform.wobble", "keys": [[0, 1]]},
         "UNKNOWN_PROPERTY", "operations[1]"),
        ({"op": "set_channel", "target": "Ball", "path": "transform.position.x",
          "keys": [[0, 1], [0, 2], [0, 3]]}, "INVALID_KEYS", "operations[1]"),
        ({"op": "generate", "armature": "Hero", "motion": "moonwalk"}, "INVALID_ARGUMENT",
         "operations[1]"),
        ({"op": "teleport"}, "INVALID_SCRIPT", "operations[1].op"),
        ({"op": "keyframe", "target": "Ball", "path": "transform.position.x", "time": 0,
          "value": 1, "shell": "ls"}, "INVALID_SCRIPT", "operations[1]"),
        ({"op": "edit_keys", "target": "Ball", "path": "transform.position.x", "edit": "explode"},
         "UNKNOWN_OPERATION", "operations[1]"),
    ],
)  # fmt: skip
def test_failures_change_nothing_and_say_where(bad, code, where):
    s = stage()
    s.animation.time = 0.25
    before = json.dumps(scene_to_dict(s), sort_keys=True)
    steps = repr(s.history)
    ok = {"op": "keyframe", "target": "Ball", "path": "transform.position.x", "time": 0,
          "value": 3}  # fmt: skip
    with pytest.raises(AnimationScriptError) as info:
        run_animation_script(s, script(ok, bad))
    assert info.value.code == code
    assert info.value.location.startswith(where)
    assert json.dumps(scene_to_dict(s), sort_keys=True) == before
    assert s.animation.time == 0.25 and repr(s.history) == steps


def test_limits_and_cancellation_roll_back():
    s = stage()
    many = {"op": "set_channel", "target": "Ball", "path": "transform.position.x",
            "keys": [[t * 0.01, t] for t in range(500)]}  # fmt: skip
    with pytest.raises(AnimationScriptError) as info:
        run_animation_script(s, {**script(many), "limits": {"max_keys": 100}})
    assert info.value.code == "BUDGET_EXCEEDED"
    with pytest.raises(AnimationScriptError) as info:
        run_animation_script(s, script(many), should_stop=lambda: True)
    assert info.value.code == "CANCELLED"
    assert s.animation.clips() == []
    with pytest.raises(AnimationScriptError):
        run_animation_script(s, {**script(many), "limits": {"max_seconds": -1}})


def test_fuzzed_scripts_fail_cleanly():
    """Random corruptions of a valid script: either it runs, or it raises an
    AnimationScriptError and the scene is exactly as before -- never anything else."""
    rng = random.Random(17)
    values = [None, True, -1, 0, 1e308, float("nan"), "", "x" * 300, [], {}, [1, 2], {"a": 1},
              "Ball", "Hero", "transform.position.x", [[0, 1]], -1e9]  # fmt: skip
    ops = BROAD["operations"]
    for trial in range(120):
        doc = copy.deepcopy(script(*rng.sample(ops, 4)))
        for _ in range(rng.randint(1, 3)):
            step = rng.choice(doc["operations"])
            key = rng.choice(list(step))
            if rng.random() < 0.15:
                del step[key]
            else:
                step[key] = rng.choice(values)
        s = stage()
        before = json.dumps(scene_to_dict(s), sort_keys=True)
        try:
            run_animation_script(s, doc)
        except AnimationScriptError as exc:
            assert exc.code and exc.location, trial
            assert json.dumps(scene_to_dict(s), sort_keys=True) == before, trial


def test_the_schema_agrees_with_the_checker():
    jsonschema = pytest.importorskip("jsonschema")
    schema = animation_script_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    check_script(BROAD)
    assert not list(validator.iter_errors(BROAD))
    for op, (_, required, _, _) in OPERATIONS.items():
        doc = script({"op": op, **{k: "Ball" for k in required}})
        assert any(e for e in validator.iter_errors({**doc, "operations": [{"op": op, "nope": 1}]}))
        with pytest.raises(AnimationScriptError):
            check_script({**doc, "operations": [{"op": op, "nope": 1}]})
    assert list(validator.iter_errors(script({"op": "teleport"})))


def test_the_cli_runs_reports_graphs_and_renders(tmp_path, capsys):
    s = stage()
    path = tmp_path / "stage.aether"
    s.save(path)
    doc = tmp_path / "bounce.json"
    doc.write_text(json.dumps(script(
        {"op": "timeline", "fps": 12, "start": 0, "end": 1},
        {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
         "keys": [[0, 1], [0.5, 0.3], [1, 1]]},
    )))  # fmt: skip

    def run_json(*argv):
        code = main([*argv, "--json"])
        return code, json.loads(capsys.readouterr().out)

    code, out = run_json("animation", "run", str(path), str(doc))
    assert code == 0 and out["animation"]["clips"][0]["keys"] == 3
    code, out = run_json("animation", "info", str(path))
    assert out["bindings"][0]["target"] == "Ball"
    code, out = run_json("animation", "report", str(path), "--target", "Ball", "--path",
                         "transform.position.z")  # fmt: skip
    assert out["keys"] == 3 and out["overshoot"] == []
    code, out = run_json("animation", "graph", str(path), str(tmp_path / "g.png"), "--target",
                         "Ball", "--path", "transform.position.z", "--width", "300")  # fmt: skip
    assert code == 0 and (tmp_path / "g.png").read_bytes()[:4] == b"\x89PNG"
    code, out = run_json("animation", "render", str(path), str(tmp_path / "frames"), "--width",
                         "64", "--height", "48", "--step", "4", "--backend", "cpu",
                         "--contact-sheet", str(tmp_path / "sheet.png"))  # fmt: skip
    assert code == 0 and len(out["frames"]) == 4 and (tmp_path / "sheet.png").exists()
    code, out = run_json("export", str(path), str(tmp_path / "out.glb"), "--animation", "clips")
    assert code == 0 and out["animations"][0]["name"] == "Ball Action"
    code, out = run_json("animation", "schema")
    assert out["$id"] == "aether3d://schema/animation-script"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(script({"op": "keyframe", "target": "Nope", "path": "visible",
                                      "time": 0, "value": True})))  # fmt: skip
    code, out = run_json("animation", "run", str(path), str(bad))
    assert code == 1 and out["error"]["location"] == "operations[0].target"


def test_scene_plans_carry_animation_scripts():
    s = Scene("Plan")
    plan = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "meshes": {"sphere": {"primitive": "sphere"}},
        "objects": {"ball": {"mesh": "@sphere", "name": "Ball"}},
        "operations": [{"op": "animation", "script": script(
            {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
             "keys": [[0, 1], [1, 0]]})}],
    }  # fmt: skip
    result = execute_plan(s, plan).to_dict()
    assert result["operations"][0]["animation_clips"] == 1
    bad = copy.deepcopy(plan)
    bad["operations"][0]["script"]["operations"][0]["op"] = "fly"
    from aether3d.errors import PlanError

    with pytest.raises(PlanError) as info:
        execute_plan(Scene("P"), bad)
    assert info.value.to_dict()["issues"][0]["pointer"].endswith("script/operations/0/op")


def test_feedback_finds_sliding_feet_and_plants_them():
    s = stage()
    hero = s.get_by_name("Hero")
    a = s.animation
    a.set_timeline(start=0, end=2)
    a.set_channel(hero, 'joints["Hips"].position.y', [(0, 0.0), (2, 0.2)], interpolation="linear")
    a.set_channel(hero, 'joints["Hips"].position.z', [(0, 0.95), (2, 0.91)], interpolation="linear")
    a.set_channel(hero, 'joints["Spine"].rotation_euler.x', [(0, 0.0), (1, 10.0), (2, 0.0)],
                  interpolation="linear")  # fmt: skip
    fs = FeedbackSession(
        s,
        {"format": "aether3d.expectations", "version": 1, "defaults": False, "checks": [
            {"check": "foot_contacts", "objects": "Hero", "feet": ["Foot.L", "Foot.R"],
             "height": 0.1, "speed": 0.5},
            {"check": "animation_continuity", "objects": "Hero"}]},
        width=96, height=64,
    )  # fmt: skip
    found = {f.code for f in fs.diagnose().findings}
    assert found == {"FOOT_SLIDING", "FOOT_PENETRATION", "MOTION_DISCONTINUITY"}
    blocked = fs.propose().to_dict()  # (animation corrections are opt-in)
    held = next(p for p in blocked["proposals"] if p["actions"][0]["action"] == "plant_feet")
    assert "allow_animation" in held["policy"]["violations"][0]
    proposals = fs.propose(policy={"allow_animation": True}).to_dict()["proposals"]
    plant = next(p for p in proposals if p["actions"][0]["action"] == "plant_feet")
    smooth = next(p for p in proposals if p["actions"][0]["action"] == "smooth_keys")
    assert set(plant["actions"][0]["legs"]) == {"Foot.L", "Foot.R"}
    assert plant["safety"] == "verify" and smooth["safety"] == "approval"
    result = fs.apply(ids=[plant["id"]], approve=[plant["id"]],
                      policy={"allow_animation": True}).to_dict()  # fmt: skip
    assert result["status"] == "applied"
    after = {f.code: f.magnitude for f in fs.diagnose().findings}
    assert "FOOT_SLIDING" not in after and "FOOT_PENETRATION" not in after
    assert after["MOTION_DISCONTINUITY"] == pytest.approx(20.0)  # approval: not applied
    assert np.isfinite(result["verification"]["after"]["score"])
