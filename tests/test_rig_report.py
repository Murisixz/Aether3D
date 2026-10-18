"""Rig diagnostics, verified corrections and previews (Stage 18)."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation.rig import make_skin
from aether3d.errors import AnimationError
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface


@pytest.fixture(scope="module")
def body_mesh():
    return implicit_surface(humanoid_shapes(), 0.016)


def rigged(body_mesh):
    s = Scene("Diag")
    body = s.add_object("Body", mesh=body_mesh)
    rep = s.animation.autorig(body)
    return s, s.get_by_name(rep["armature"]), body


def test_diagnosis_measures_poses_without_changing_anything(body_mesh):
    s, arm, body = rigged(body_mesh)
    a = s.animation
    rev = s.revision
    d = a.diagnose_rig(arm, animation=False)
    assert s.revision == rev
    assert set(d["poses"]) >= {"elbow_bend_120.L", "knee_bend_120.R", "wrist_twist_90.L",
                               "torso_twist_45", "head_turn_60"}  # fmt: skip
    for pose, meshes in d["poses"].items():
        rec = meshes["Body"]
        assert 0.4 < rec["joint_ring"] < 1.2, pose
        assert "volume_change" in rec  # (the implicit body is closed)
    assert all(f["basis"] == "numeric" for f in d["findings"])
    assert sum(d["summary"].values()) >= len(d["findings"])


def test_linear_twists_collapse_and_dual_quaternions_do_not(body_mesh):
    """The candy wrapper: a 90-degree wrist twist without twist helpers, measured."""
    s, arm, body = rigged(body_mesh)
    a = s.animation
    raw = a.rig(arm).to_json()
    raw["constraints"] = [c for c in raw["constraints"] if c["type"] != "twist"]
    a.set_rig(arm, raw)
    sk = a._owned[("skin", body.id)]._definition
    rig = a.rig(arm)
    tw = {rig.joints[rig.index(n)].id for n in ("ForeArmTwist.L", "ForeArmTwist.R")}
    dense = sk.dense()
    fore = {rig.joints[rig.index(n)].id: rig.joints[rig.index(n.replace("Twist", ""))].id
            for n in ("ForeArmTwist.L", "ForeArmTwist.R")}  # fmt: skip
    for c, j in enumerate(sk.joints):  # the twist helpers' weights back to the forearms
        if j in tw:
            dense[:, sk.joints.index(fore[j])] += dense[:, c]
            dense[:, c] = 0.0
    idx = np.argsort(-dense, axis=1)[:, :4].astype(np.int32)
    w = np.take_along_axis(dense, idx, axis=1)
    with a.batch("No twist"):
        a._set(a._owned[("skin", body.id)], definition=make_skin(
            sk.object, sk.rig, list(sk.joints), sk.inverse_bind, idx, w, "linear"))  # fmt: skip
    lin = a.diagnose_rig(arm, poses=["wrist_twist_90.L"], animation=False)
    ring_lin = lin["poses"]["wrist_twist_90.L"]["Body"]["joint_ring"]
    with a.batch("DQS"):
        cur = a._owned[("skin", body.id)]._definition
        a._set(a._owned[("skin", body.id)], definition=cur.replace(method="dual_quaternion"))
    dq = a.diagnose_rig(arm, poses=["wrist_twist_90.L"], animation=False)
    ring_dq = dq["poses"]["wrist_twist_90.L"]["Body"]["joint_ring"]
    assert ring_dq > ring_lin
    assert ring_dq > 0.95


def test_bind_mismatch_is_found_and_fixed_by_a_verified_rebind(body_mesh):
    s, arm, body = rigged(body_mesh)
    a = s.animation
    a.move_joint(arm, "Shin.L", [0.0, 0.0, 0.03], space="parent", skins="keep_bind")
    d = a.diagnose_rig(arm, poses=None, animation=False)
    found = [f for f in d["findings"] if f["code"] == "BIND_MISMATCH"]
    assert found and found[0]["magnitude"] > 0.01
    props = a.propose_rig_fix(arm, diagnosis=d)
    rebind = [p for p in props["proposals"] if p["code"] == "BIND_MISMATCH"]
    assert rebind and rebind[0]["safety"] == "verify"
    n = len(s.history)
    rep = a.apply_rig_fix(arm, rebind)
    assert rep["applied"] == [rebind[0]["id"]] and not rep["reverted"]
    assert len(s.history) == n + 1
    after = a.diagnose_rig(arm, poses=None, animation=False)
    assert not any(f["code"] == "BIND_MISMATCH" for f in after["findings"])
    s.history.undo()
    assert any(f["code"] == "BIND_MISMATCH"
               for f in a.diagnose_rig(arm, poses=None, animation=False)["findings"])  # fmt: skip


def test_fixes_that_do_not_help_are_undone_and_dry_runs_change_nothing(body_mesh):
    s, arm, body = rigged(body_mesh)
    a = s.animation
    harmful = {"id": "fix-test", "code": "WEIGHT_LEAK", "safety": "verify",
               "title": "the hand takes the thigh", "target": {"mesh": "Body"},
               "operations": [{"call": "edit_weights", "mesh": "Body", "operations": [
                   {"op": "assign", "joint": "Hand.L", "value": 1.0,
                    "select": {"role": "thigh.L"}}]}]}  # fmt: skip
    rev = s.revision
    weights = a._owned[("skin", body.id)]._definition.weights.copy()
    rep = a.apply_rig_fix(arm, [harmful])
    assert rep["reverted"] == ["fix-test"] and rep["applied"] == []
    assert rep["why"]["fix-test"]["worse"]  # (it made things worse: undone)
    assert np.array_equal(a._owned[("skin", body.id)]._definition.weights, weights)
    assert s.revision == rev
    rep = a.apply_rig_fix(arm, [{**harmful, "safety": "approval"}], dry_run=True)
    assert rep["applied"] == ["fix-test"] and rep["dry_run"] and s.revision == rev
    with pytest.raises(AnimationError):
        a.apply_rig_fix(arm, [{"code": "X", "operations": [{"call": "rm -rf"}]}])


def test_collapse_fix_blends_dual_quaternions_and_improves(body_mesh):
    s, arm, body = rigged(body_mesh)
    a = s.animation
    d = a.diagnose_rig(arm, poses=["knee_bend_120.L"], animation=False)
    before = d["poses"]["knee_bend_120.L"]["Body"]["joint_ring"]
    fix = {"id": "fix-blend", "code": "JOINT_COLLAPSE", "safety": "verify", "title": "blend",
           "target": {"mesh": "Body", "pose": "knee_bend_120.L", "joints": ["Shin.L"]},
           "operations": [{"call": "skin_method", "mesh": "Body", "method": "blended",
                           "dq": {"joint": "Shin.L", "share": 1.0}}]}  # fmt: skip
    a.apply_rig_fix(arm, [fix], verify=False)
    sk = a._owned[("skin", body.id)]._definition
    assert sk.method == "blended" and sk.dq.max() == 1.0 and sk.dq.min() == 0.0
    after = a.diagnose_rig(arm, poses=["knee_bend_120.L"], animation=False)
    assert after["poses"]["knee_bend_120.L"]["Body"]["joint_ring"] > before


def test_timeline_checks_find_flips_unreachable_targets_and_sliding(body_mesh):
    s, arm, body = rigged(body_mesh)
    a = s.animation
    a.generate(arm, "walk", forward="-y")
    tgt = s.get_by_name(f"{arm.name} hand_ik.L")
    a.keyframe(tgt, "transform.position", 0.0, [3.0, 0.0, 1.0])  # far out of reach
    a.keyframe(arm, 'constraints["arm_ik.L"].influence', 0.0, 1.0)
    d = a.diagnose_rig(arm, poses=None)
    codes = {f["code"] for f in d["findings"]}
    assert "IK_UNREACHABLE" in codes
    assert d["timeline"]["feet"]  # measured
    assert d["timeline"]["ik_worst"]["arm_ik.L"]["residual"] > 1.0


def test_previews_render_without_touching_the_scene(body_mesh, tmp_path):
    s, arm, body = rigged(body_mesh)
    a = s.animation
    rev = s.revision
    n = len(s)
    for kw in ({"mode": "pose"}, {"mode": "weights"}, {"mode": "weights", "joint": "Hand.L"},
               {"mode": "stretch", "pose": "elbow_bend_120.L"}, {"mode": "skeleton"}):  # fmt: skip
        out = a.rig_preview(arm, width=96, height=128, path=tmp_path / "p.png", **kw)
        img = out["image"]
        assert img.shape[:2] == (128, 96)
        assert img[..., :3].std() > 1  # (something drawn)
    assert "<svg" in out["svg"] and "UpperArm.L" in out["svg"] and "hand_ik.L" in out["svg"]
    assert (tmp_path / "p.png").stat().st_size > 100
    assert s.revision == rev and len(s) == n
    with pytest.raises(AnimationError):
        a.rig_preview(arm, mode="stretch")
    with pytest.raises(AnimationError):
        a.rig_preview(arm, mode="xray")
