"""Rig diagnostics in the feedback loop (Stage 18): a ``rig`` expectation check, its
corrections (``rig_fix``: the rig's own proposals, measured again after each change) and
the data checks at the boundary."""

from __future__ import annotations

import pytest

from aether3d import Scene
from aether3d.errors import AnimationError
from aether3d.feedback import FeedbackSession
from aether3d.feedback.corrections import check_actions
from aether3d.feedback.schema import actions_schema, expectations_schema
from aether3d.feedback.session import FeedbackError
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface


@pytest.fixture(scope="module")
def body_mesh():
    return implicit_surface(humanoid_shapes(), 0.018)


def rigged(mesh, *, soft_knee=False):
    s = Scene("Feedback")
    body = s.add_object("Body", mesh=mesh)
    s.animation.autorig(body, name="Rig")
    if soft_knee:  # (weights smoothed by hand far too wide: the knee collapses when bent)
        smooth = {"op": "smooth", "iterations": 200, "joints": ["Thigh.L", "Shin.L"]}
        s.animation.edit_weights(body, [smooth, smooth])
    return s


def expectations(*checks):
    return {"format": "aether3d.expectations", "version": 1, "defaults": False,
            "checks": list(checks)}  # fmt: skip


RIG = {"check": "rig", "id": "rig", "objects": "Rig", "poses": ["knee_bend_120.L"],
       "timeline": False}  # fmt: skip


def test_the_rig_check_finds_corrects_and_verifies(body_mesh):
    s = rigged(body_mesh, soft_knee=True)
    fs = FeedbackSession(s, expectations(RIG), width=48, height=48)
    findings = fs.diagnose().findings
    collapse = next(f for f in findings if f.code == "JOINT_COLLAPSE")
    assert collapse.kind == "rig" and collapse.unit == "fraction lost"
    assert 0.25 < collapse.magnitude < 0.5  # (the skin keeps 50..75 % of its size)
    assert collapse.key.endswith("|knee_bend_120.L/Body/Shin.L")
    assert collapse.evidence["rig"]["pose"] == "knee_bend_120.L"
    held = fs.propose().to_dict()["proposals"]  # (rig corrections are animation: opt-in)
    assert held and all("allow_animation" in p["policy"]["violations"][0] for p in held)
    policy = {"allow_animation": True}
    proposals = fs.propose(policy=policy).to_dict()["proposals"]
    fix = next(p for p in proposals if collapse.key in p["findings"])
    action = fix["actions"][0]
    assert action["action"] == "rig_fix" and fix["safety"] == "verify"
    assert action["fixes"][0]["operations"][0]["call"] == "skin_method"
    assert not fix["policy"]["auto"]  # (verify: applied when asked)
    rev = s.revision
    result = fs.apply(ids=[fix["id"]], policy=policy).to_dict()
    assert result["status"] == "applied", result
    verification = result["verification"]
    assert verification["decision"] == "kept" and collapse.key in verification["resolved"]
    assert not verification["new_failures"]
    skin = s.animation._owned[("skin", s.get_by_name("Body")._id)]._definition
    assert skin.method == "blended" and 0.0 < skin.dq.max() <= 1.0
    assert 0.0 < float((skin.dq > 0).mean()) < 0.5  # (only around the knee)
    s.history.undo()
    assert s.revision != rev and s.animation._owned[
        ("skin", s.get_by_name("Body")._id)]._definition.method == "linear"  # fmt: skip


def test_rig_fixes_that_do_not_help_are_undone(body_mesh):
    s = rigged(body_mesh)
    a = s.animation
    diag = a.diagnose_rig("Rig", poses=["knee_bend_120.L"], animation=False)
    useless = {"code": "INTERPENETRATION", "safety": "verify", "title": "Smooth nothing",
               "operations": [{"call": "edit_weights", "mesh": "Body",
                               "operations": [{"op": "normalize"}]}],
               "target": {"mesh": "Body", "pose": "knee_bend_120.L"}}  # fmt: skip
    assert any(f["code"] == "INTERPENETRATION" for f in diag["findings"])
    steps = len(s.history)
    report = a.apply_rig_fix("Rig", [useless])
    assert report["applied"] == [] and report["reverted"] == [None]
    assert len(s.history) == steps  # (the fix left no step)


def test_rig_fix_actions_and_proposals_are_checked_data(body_mesh):
    s = rigged(body_mesh)
    a = s.animation
    steps = len(s.history)
    good = a.propose_rig_fix("Rig", poses=["knee_bend_120.L"], animation=False)["proposals"]
    assert check_actions([{"action": "rig_fix", "armature": "Rig", "fixes": good}])
    assert check_actions([{"action": "rig_fix", "armature": "Rig", "fixes": "safe"}])
    for bad in (
        {"action": "rig_fix", "armature": "Rig", "fixes": "everything"},
        {"action": "rig_fix", "armature": "Rig", "fixes": [{"code": "X"}]},
        {"action": "rig_fix", "armature": "Rig", "fixes": []},
        {"action": "rig_fix", "armature": "Rig"},
    ):
        with pytest.raises(FeedbackError):
            check_actions([bad])
    op = {"call": "skin_method", "mesh": "Body", "method": "blended",
          "dq": {"joint": "Shin.L", "share": 0.5}}  # fmt: skip
    for change, where in (
        ({"call": "eval"}, "operations[0]"),
        ({"source": "import os"}, "operations[0]"),
        ({"method": "magic"}, "operations[0].method"),
        ({"dq": {"share": 2.0}}, "operations[0].dq.share"),
        ({"dq": {"share": "auto"}}, "operations[0].dq"),
        ({"dq": {"joint": "Nope"}}, "operations[0].dq.joint"),
        ({"mesh": "Rig"}, "operations[0].mesh"),
    ):
        fix = {"code": "JOINT_COLLAPSE", "operations": [{**op, **change}]}
        with pytest.raises(AnimationError) as e:
            a.apply_rig_fix("Rig", [fix])
        assert e.value.location == f"fixes[0].{where}", (change, e.value.location)
    with pytest.raises(AnimationError):
        a.apply_rig_fix("Rig", [{"code": "X", "operations": [op], "run": "rm -rf /"}])
    with pytest.raises(AnimationError):
        a.apply_rig_fix("Rig", [{"code": "X", "operations": [{"call": "pole",
                                                             "constraint": "nope"}]}])  # fmt: skip
    assert len(s.history) == steps  # (nothing changed)
    fs = FeedbackSession(s, expectations(RIG), width=48, height=48)
    forged = {"code": "JOINT_COLLAPSE", "operations": [{**op, "mesh": "Nope"}]}
    with pytest.raises(FeedbackError) as e:
        fs.apply(actions=[{"action": "rig_fix", "armature": "Rig", "fixes": [forged]}],
                 policy={"allow_animation": True})  # fmt: skip
    assert e.value.location == "/actions/0/fixes/0"
    assert len(s.history) == steps


def test_a_rig_check_on_an_unrigged_object_says_so(body_mesh):
    s = Scene("Plain")
    s.add_object("Body", mesh=body_mesh)
    fs = FeedbackSession(s, expectations({**RIG, "objects": "Body"}), width=48, height=48)
    d = fs.diagnose()
    assert [f.code for f in d.findings] == ["RIG_MISSING"]
    out = fs.propose(policy={"allow_animation": True}).to_dict()
    assert out["proposals"] == []
    schema = expectations_schema()
    rig = next(c for c in schema["properties"]["checks"]["items"]["oneOf"]
               if c["properties"]["check"].get("const") == "rig")  # fmt: skip
    assert set(rig["properties"]) >= {"objects", "poses", "timeline", "ignore"}
    actions = actions_schema()["items"]["oneOf"]
    assert any(v["properties"]["action"]["const"] == "rig_fix" for v in actions)
    with pytest.raises(FeedbackError):
        FeedbackSession(s, expectations({**RIG, "poses": "all of them"}))
