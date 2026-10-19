"""Rigging through animation scripts and the command line (Stage 18): data only, one
undo step, all or nothing."""

from __future__ import annotations

import json

import pytest

from aether3d import Scene, load_scene
from aether3d.animation.script import AnimationScriptError, run_animation_script
from aether3d.cli import main
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface


def script(*ops):
    return {"format": "aether3d.animation-script", "version": 1, "operations": list(ops)}


@pytest.fixture(scope="module")
def body_mesh():
    return implicit_surface(humanoid_shapes(), 0.018)


def test_a_rigging_script_is_one_step_and_reports(body_mesh):
    s = Scene("Script")
    s.add_object("Body", mesh=body_mesh)
    n = len(s.history)
    out = run_animation_script(s, script(
        {"op": "detect_landmarks", "meshes": ["Body"]},
        {"op": "autorig", "meshes": [{"object": "Body"}], "name": "Rig"},
        {"op": "rename_joint", "armature": "Rig", "joint": "Head", "name": "Skull"},
        {"op": "edit_weights", "mesh": "Body", "operations": [{"op": "normalize"}]},
        {"op": "add_control", "armature": "Rig", "name": "look2", "kind": "gaze",
         "joint": "Skull"},
        {"op": "generate", "armature": "Rig", "motion": "idle"},
        {"op": "pose_character", "armature": "Rig", "time": 0.5,
         "commands": [{"do": "raise_arm", "side": "L", "angle": 45}]},
        {"op": "ik_fk", "armature": "Rig", "limb": "arm_ik.R", "mode": "ik", "time": 0.25},
        {"op": "inspect_weights", "mesh": "Body"},
        {"op": "diagnose_rig", "armature": "Rig", "poses": ["elbow_bend_120.L"],
         "animation": False},
        {"op": "apply_rig_fix", "armature": "Rig", "fixes": "safe"},
    ))  # fmt: skip
    assert len(s.history) == n + 1
    ops = out["operations"]
    assert ops[0]["result"]["landmarks"]["hip.L"]["confidence"] > 0
    assert ops[1]["result"]["armature"] == "Rig"
    assert ops[7]["result"]["mismatch"] < 1e-9
    assert ops[8]["result"]["valid"]
    assert "elbow_bend_120.L" in ops[9]["result"]["poses"]
    rig = s.animation.rig(s.get_by_name("Rig"))
    assert rig.index("Skull") is not None and rig.control("look2").kind == "gaze"
    json.dumps(out)  # (everything reported is JSON)
    s.history.undo()
    assert len(s) == 1 and s.animation.rig(s.get_by_name("Body")) is None


def test_failing_rigging_scripts_change_nothing(body_mesh):
    s = Scene("Fail")
    s.add_object("Body", mesh=body_mesh)
    rev = s.revision
    for bad, where in (
        ([{"op": "autorig", "meshes": ["Body"], "name": "Rig"},
          {"op": "remove_joint", "armature": "Rig", "joint": "Nope"}], "operations[1]"),
        ([{"op": "autorig", "meshes": []}], "operations[0].meshes"),
        ([{"op": "autorig", "meshes": ["Body"], "template": "octopus"}], "operations[0]"),
        ([{"op": "pose_character", "armature": "Body", "commands": [{"do": "fly"}]}],
         "operations[0]"),
        ([{"op": "edit_rig", "armature": "Body", "operations": [], "evil": "eval()"}],
         "operations[0]"),
    ):  # fmt: skip
        with pytest.raises(AnimationScriptError) as e:
            run_animation_script(s, script(*bad))
        assert e.value.location.startswith(where), (e.value.location, where)
        assert s.revision == rev and len(s) == 1


def test_the_rig_command_line(body_mesh, tmp_path, capsys):
    s = Scene("CLI")
    s.add_object("Body", mesh=body_mesh)
    path = tmp_path / "hero.aether"
    s.save(path)

    def run_json(*argv):
        code = main([*argv, "--json"])
        return code, json.loads(capsys.readouterr().out)

    code, out = run_json("rig", "landmarks", str(path), "--mesh", "Body")
    assert code == 0 and "knee.L" in out["landmarks"]
    code, out = run_json("rig", "autorig", str(path), "--mesh", "Body", "--name", "Hero")
    assert code == 0 and out["armature"] == "Hero"
    code, out = run_json("rig", "info", str(path), "--armature", "Hero")
    assert code == 0 and out["skins"][0]["mesh"] == "Body"
    code, out = run_json("rig", "weights", str(path), "--mesh", "Body", "--joint", "Hand.L")
    assert code == 0 and len(out["weights"]) == body_mesh.vertex_count
    code, out = run_json("rig", "diagnose", str(path), "--armature", "Hero", "--pose",
                         "knee_bend_120.L")  # fmt: skip
    assert code == 0 and list(out["poses"]) == ["knee_bend_120.L"]
    code, out = run_json("rig", "fix", str(path), "--armature", "Hero", "--dry-run")
    assert code == 0 and out["dry_run"]
    code, out = run_json("rig", "preview", str(path), str(tmp_path / "s.svg"), "--armature",
                         "Hero")  # fmt: skip
    assert code == 0 and (tmp_path / "s.svg").read_text().startswith("<svg")
    size = ("--width", "64", "--height", "80")
    code, out = run_json("rig", "preview", str(path), str(tmp_path / "w.png"), "--armature",
                         "Hero", "--mode", "weights", *size)  # fmt: skip
    assert code == 0 and (tmp_path / "w.png").read_bytes()[:4] == b"\x89PNG"
    code, out = run_json("rig", "guide")
    assert "Rigid-part characters need no bones" in out["guide"]
    # The revision's verbs: test (the stress suite, a sheet), pose, ikfk, bind, transfer,
    # edit and create -- each one undo step, saved.
    code, out = run_json("rig", "test", str(path), "--armature", "Hero", "--sheet",
                         str(tmp_path / "sheet.png"))  # fmt: skip
    assert code == 0 and "elbow_bend_140.L" in out["poses"]
    assert {"joint_ring", "new_creases", "folded_triangles"} <= set(
        out["poses"]["elbow_bend_140.L"]["Body"])  # fmt: skip
    assert out["sheet"]["poses"][0] == "elbow_bend_140.L"
    assert (tmp_path / "sheet.png").read_bytes()[:4] == b"\x89PNG"
    code, out = run_json("rig", "pose", str(path), "--armature", "Hero", "--time", "1", "--command",
                         '{"do": "raise_arm", "side": "L", "angle": 120}', "--command",
                         '{"do": "roll_foot", "side": "R", "angle": 25}')  # fmt: skip
    assert code == 0 and [c["do"] for c in out["commands"]] == ["raise_arm", "roll_foot"]
    assert out["commands"][0]["clavicle_angle"] == 15.0
    code, out = run_json("rig", "ikfk", str(path), "--armature", "Hero", "--limb", "arm_ik.R",
                         "--mode", "ik", "--time", "2")  # fmt: skip
    assert code == 0 and out["mismatch"] < 1e-9
    code, out = run_json("rig", "bind", str(path), "--mesh", "Body", "--armature", "Hero",
                         "--smooth", "8")  # fmt: skip
    assert code == 0 and out["technique"].startswith("bone heat")
    loaded = load_scene(path)
    skin = loaded.animation._owned[("skin", loaded.get_by_name("Body")._id)]._definition
    assert json.loads(skin.smooth)["iterations"] == 8 and skin.normals == "surface"
    code, out = run_json("rig", "edit", str(path), "--armature", "Hero", "--op",
                         '{"op": "rename_joint", "joint": "Head", "name": "Skull"}')  # fmt: skip
    assert code == 0
    assert load_scene(path).animation.rig("Hero").index("Skull") is not None
    rig_file = tmp_path / "rig.json"
    rig_file.write_text(json.dumps({"joints": [{"name": "Root", "position": [0, 0, 1]}]}))
    code, out = run_json("rig", "create", str(path), "--armature", "Prop", "--rig",
                         str(rig_file))  # fmt: skip
    assert code == 0 and out["joints"] == ["Root"]
    code, out = run_json("rig", "pose", str(path), "--armature", "Hero", "--command",
                         '{"do": "fly"}')  # fmt: skip
    assert code != 0 and out["error"]["code"] == "INVALID_ARGUMENT"
    code, out = run_json("rig", "info", str(path), "--armature", "Nope")
    assert code != 0
    code, out = run_json("rig", "info", str(path), "--armature", "Body")
    assert code != 0 and out["error"]["code"] == "RIG_MISSING"


def test_the_rig_commands_do_not_load_the_mcp_sdk(tmp_path):
    """``aether3d rig`` works without the optional MCP SDK (the summary lives in the
    engine, not in the server package)."""
    import subprocess
    import sys

    s = Scene("Bare")
    s.animation.add_rig(s.add_object("Arm"), [{"name": "Root", "position": [0, 0, 1]}])
    path = tmp_path / "bare.aether"
    s.save(path)
    code = (
        "import json, sys; from aether3d.cli import main; "
        f"main(['rig', 'info', {str(path)!r}, '--armature', 'Arm', '--json']); "
        "main(['rig', 'guide', '--json']); "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'mcp'), file=sys.stderr)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stderr.strip() == "[]"
    assert '"armature": "Arm"' in out.stdout
