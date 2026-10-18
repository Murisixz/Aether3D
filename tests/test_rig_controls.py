"""Controls, IK/FK switching with matched poses, and high-level posing (Stage 18)."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.errors import AnimationError
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface


@pytest.fixture(scope="module")
def body_mesh():
    return implicit_surface(humanoid_shapes(), 0.016)


def rigged(body_mesh, name="Hero"):
    s = Scene(name)
    body = s.add_object("Body", mesh=body_mesh)
    rep = s.animation.autorig(body)
    return s, s.get_by_name(rep["armature"]), body


def chain_positions(a, arm, names, t):
    j = a.pose_joints(arm, t, space="world")
    return np.array([j[n][:3, 3] for n in names])


@pytest.mark.parametrize("limb, names", [
    ("leg_ik.L", ["Thigh.L", "Shin.L", "Foot.L"]),
    ("arm_ik.R", ["UpperArm.R", "ForeArm.R", "Hand.R"]),
])  # fmt: skip
def test_ik_fk_switches_keep_the_pose(body_mesh, limb, names):
    s, arm, _ = rigged(body_mesh)
    a = s.animation
    period = a.generate(arm, "walk", forward="-y")["period"]
    t_ik, t_fk = 0.3 * period, 0.75 * period
    before_ik = chain_positions(a, arm, names, t_ik)
    n = len(s.history)
    rep = a.ik_fk(arm, limb, "ik", t_ik)
    assert len(s.history) == n + 1
    assert rep["mismatch"] < 1e-9 and rep["residual"] < 1e-9
    assert np.allclose(chain_positions(a, arm, names, t_ik), before_ik, atol=1e-9)
    # In IK, the limb follows its target: the target is keyed where the FK pose was.
    assert a.rigging(t_ik + 0.05).residuals[f"{arm.name}/{limb}"] < 1e-6
    before_fk = chain_positions(a, arm, names, t_fk)
    rep = a.ik_fk(arm, limb, "fk", t_fk)
    assert rep["mismatch"] < 1e-9
    assert np.allclose(chain_positions(a, arm, names, t_fk), before_fk, atol=1e-9)
    # The influence is 0 before, 1 between and 0 after the switches.
    path = f'constraints["{limb}"].influence'
    v = a.sample(arm, path, [t_ik - 0.01, 0.5 * (t_ik + t_fk), t_fk + 0.01])
    assert np.allclose(v, [0.0, 1.0, 0.0])
    s.history.undo()
    assert np.allclose(a.sample(arm, path, [0.5 * (t_ik + t_fk)]), [1.0])


def test_blended_switches_and_errors(body_mesh):
    s, arm, _ = rigged(body_mesh)
    a = s.animation
    a.generate(arm, "walk", forward="-y")
    rep = a.ik_fk(arm, "foot_ik.L", "ik", 0.5, blend=0.2)  # (a control names its limb)
    assert rep["limb"] == "leg_ik.L" and rep["mismatch"] < 1e-9
    v = a.sample(arm, 'constraints["leg_ik.L"].influence', [0.3, 0.4, 0.5])
    assert np.allclose(v, [0.0, 0.5, 1.0])
    for args in (("nope", "ik", 0.5), ("leg_ik.L", "both", 0.5), ("gaze", "ik", 0.5)):
        with pytest.raises(AnimationError):
            a.ik_fk(arm, *args)


def test_controls_on_a_hand_built_rig():
    """Stage 17 rigs (identity joint frames, no controls) gain IK and gaze controls."""
    s = Scene("Built")
    arm = s.add_object("Arm")
    a = s.animation
    a.add_rig(arm, [
        {"name": "Hips", "position": [0, 0, 1]},
        {"name": "Chest", "parent": "Hips", "position": [0, 0, 0.4]},
        {"name": "Head", "parent": "Chest", "position": [0, 0, 0.3], "tail": [0, 0, 0.2]},
        {"name": "UpperArm.L", "parent": "Chest", "position": [0.2, 0, 0.2]},
        {"name": "ForeArm.L", "parent": "UpperArm.L", "position": [0, -0.02, -0.3]},
        {"name": "Hand.L", "parent": "ForeArm.L", "position": [0, 0.02, -0.25],
         "tail": [0, 0, -0.1]},
    ])  # fmt: skip
    rep = a.add_control(arm, "hand_ik.L", "ik", chain=["UpperArm.L", "ForeArm.L", "Hand.L"])
    assert len(rep["objects"]) == 2  # a target and a pole
    rig = a.rig(arm)
    assert rig.control("hand_ik.L").kind == "ik_target"
    assert rig.control("hand_ik.L_pole").kind == "pole"
    tgt = s.get_by_name("Arm hand_ik.L")
    tgt.position = (0.45, -0.15, 1.35)
    a.keyframe(arm, 'constraints["hand_ik.L_ik"].influence', 0.0, 1.0)
    assert a.rigging(0.0).residuals["Arm/hand_ik.L_ik"] < 1e-9
    a.add_control(arm, "look", "gaze", joint="Head")
    rig = a.rig(arm)
    gaze = s.get(rig.control("look").object)
    gaze.position = (2.0, -1.0, 1.7)
    a.keyframe(arm, 'constraints["look"].influence', 0.0, 1.0)
    head = a.pose_joints(arm, 0.0, space="world")["Head"]
    axis = rig.constraints[-1].p["axis"]
    i, sign = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1),
               "-z": (2, -1)}[axis]  # fmt: skip
    aim = head[:3, i] * sign
    want = np.array([2.0, -1.0, 1.7]) - head[:3, 3]
    assert np.dot(aim / np.linalg.norm(aim), want / np.linalg.norm(want)) > 0.9999
    for bad in ({"kind": "ik", "chain": ["Hips"]}, {"kind": "gaze"}, {"kind": "spin"}):
        with pytest.raises(AnimationError):
            a.add_control(arm, "x", **bad)


def test_pose_commands(body_mesh):
    s, arm, _ = rigged(body_mesh)
    a = s.animation
    m = arm.world_matrix
    goal = (m @ np.array([0.3, -0.35, 1.3, 1.0]))[:3]
    rep = a.pose_character(arm, [
        {"do": "reach", "hand": "L", "target": goal.tolist()},
        {"do": "look_at", "target": goal.tolist()},
    ], time=1.0)  # fmt: skip
    reach = rep["commands"][0]
    assert reach["reachable"] and reach["residual"] < 1e-6 and reach["lean_degrees"] == 0
    hand = a.pose_joints(arm, 1.0, space="world")["Hand.L"][:3, 3]
    assert np.linalg.norm(hand - goal) < 1e-6
    # Out of reach: the chest leans toward the goal, the report says how far it is.
    far = (m @ np.array([0.2, -0.9, 1.2, 1.0]))[:3]
    rep = a.pose_character(arm, [{"do": "reach", "hand": "R", "target": far.tolist()}], time=2.0)
    assert not rep["commands"][0]["reachable"] and rep["commands"][0]["lean_degrees"] > 0
    # Raise, lean, crouch with planted feet, aim the hips, rotate any joint (from rest).
    feet0 = chain_positions(a, arm, ["Foot.L", "Foot.R"], 3.0)
    rep = a.pose_character(arm, [
        {"do": "raise_arm", "side": "R", "angle": 90, "direction": "forward"},
        {"do": "lean", "angle": 20, "direction": "forward"},
        {"do": "crouch", "depth": 0.25},
        {"do": "aim_hips", "yaw": 15},
        {"do": "rotate", "joint": "Head", "euler": [0, 0, 30]},
    ], time=3.0)  # fmt: skip
    j = a.pose_joints(arm, 3.0, space="armature")
    upper = j["ForeArm.R"][:3, 3] - j["UpperArm.R"][:3, 3]
    assert abs(upper[2]) < 0.15 * np.linalg.norm(upper) and upper[1] < 0  # forward, level
    feet1 = chain_positions(a, arm, ["Foot.L", "Foot.R"], 3.0)
    assert np.allclose(feet0, feet1, atol=1e-6)  # planted while the hips went down
    rig = a.rig(arm)
    hips0 = rig.rest_model()[rig.index("Hips")][2, 3]  # (keys hold before their time)
    assert j["Hips"][2, 3] == pytest.approx(hips0 - 0.25, abs=1e-9)
    assert rep["residuals"]["leg_ik.L"] < 1e-6 and rep["residuals"]["leg_ik.R"] < 1e-6
    chest = j["Chest"][:3, 3] - j["Spine"][:3, 3]
    assert chest[1] < -0.02  # leaning forward (-y)
    with pytest.raises(AnimationError):
        a.pose_character(arm, [{"do": "fly"}], time=1.0)
    with pytest.raises(AnimationError):
        a.pose_character(arm, [{"do": "rotate", "joint": "Tail"}], time=1.0)
