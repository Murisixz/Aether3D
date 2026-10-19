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
    assert np.degrees(np.arctan2(-chest[1], chest[2])) > 5.0  # leaning forward (-y)
    with pytest.raises(AnimationError):
        a.pose_character(arm, [{"do": "fly"}], time=1.0)
    with pytest.raises(AnimationError):
        a.pose_character(arm, [{"do": "rotate", "joint": "Tail"}], time=1.0)


def test_a_held_crouch_keeps_the_feet_where_they_were_planted(body_mesh):
    """Planting feet measures them where they are: already planted by IK while the hips
    are low, they stay on the floor (not where a straight FK leg would put them)."""
    s, arm, _ = rigged(body_mesh)
    a = s.animation
    a.generate(arm, "idle", forward="-y")
    feet = []
    for t, depth in ((0.5, 0.0), (1.2, 0.4), (1.9, 0.4), (2.6, 0.0)):
        rep = a.pose_character(arm, [{"do": "crouch", "depth": depth}], time=t)
        assert max(rep["commands"][0]["residuals"].values()) < 2e-3, (t, rep)
        feet.append(np.array([rep["commands"][0]["feet"][s_] for s_ in ("L", "R")]))
    for f in feet[1:]:
        assert np.abs(f - feet[0]).max() < 3e-3  # (the same spot, on the floor)
    hips = chain_positions(a, arm, ["Hips"], 1.5)[0]
    rest = chain_positions(a, arm, ["Hips"], 0.5)[0]
    assert rest[2] - hips[2] == pytest.approx(0.4, abs=0.02)


def test_switches_keep_every_joint_including_twist_helpers(body_mesh):
    """FK -> IK -> FK with a hand target turned in between: no joint moves or turns at
    either switch -- helpers driven by twist constraints included (they are evaluated
    after the IK that moves their source joint), and a relaxed limb is not rolled."""
    from aether3d.animation.rigging import constraint_order

    s, arm, _ = rigged(body_mesh)
    a = s.animation
    a.generate(arm, "idle", forward="-y")
    rig = a.rig(arm)
    order = [rig.constraints[i].name for i in constraint_order(rig)]
    assert order.index("forearm_twist.R") > order.index("arm_ik.R")
    a.ik_fk(arm, "arm_ik.R", "ik", 0.5)
    c = next(c for c in rig.constraints if c.name == "arm_ik.R")
    target = s._objects[c.p["target"]["object"]]
    from aether3d.animation import rotation as R

    q0 = np.asarray(a.sample(target, "transform.rotation", [0.5])[0], float)
    turned = R.multiply(q0[None], R.axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(70.0)))[0]
    a.keyframe(target, "transform.rotation", 1.2, list(turned))
    for t, mode in ((0.5, None), (1.8, "fk")):
        if mode:
            a.ik_fk(arm, "arm_ik.R", mode, t)
        before = a.pose_joints(arm, t - 1e-6, space="world")
        after = a.pose_joints(arm, t, space="world")
        for name, m in after.items():
            assert np.abs(m - before[name]).max() < 1e-4, (t, name)


@pytest.fixture(scope="module")
def figure_rig():
    """The anatomical figure, rigged without a skin (its hands show their fingers)."""
    from aether3d.sculpt.figure import figure_mesh

    s = Scene("Figure")
    body = s.add_object("Body", mesh=figure_mesh(0.008))
    rep = s.animation.autorig(body, name="Rig", bind=False)
    return s, rep


def test_raising_an_arm_elevates_the_clavicle_and_points_the_arm_exactly(figure_rig):
    from aether3d.animation import rotation as R
    from aether3d.animation.controls import shoulder_share

    s, _ = figure_rig
    a = s.animation
    j0 = a.pose_joints("Rig", 0.0, space="armature")
    n = len(s.history)
    for t, side, angle, direction, shoulder in (
        (1.0, "L", 90.0, "forward", "auto"),
        (2.0, "R", 150.0, "side", "auto"),
        (3.0, "L", 120.0, "side", 0.0),
    ):
        rep = a.pose_character("Rig", [{"do": "raise_arm", "side": side, "angle": angle,
                                        "direction": direction, "shoulder": shoulder}],
                               time=t)["commands"][0]  # fmt: skip
        assert rep["clavicle"] == f"Shoulder.{side}"
        assert rep["clavicle_angle"] == pytest.approx(shoulder_share(angle, shoulder))
        j = a.pose_joints("Rig", t, space="armature")
        out = np.array([1.0 if side == "L" else -1.0, 0.0, 0.0])
        fwd = np.array([0.0, -1.0, 0.0])
        axis = np.cross([0.0, 0.0, -1.0], fwd if direction == "forward" else out)
        turn = R.matrices(R.axis_angle(axis / np.linalg.norm(axis), np.radians(angle)))[0]
        d0 = j0[f"ForeArm.{side}"][:3, 3] - j0[f"UpperArm.{side}"][:3, 3]
        d1 = j[f"ForeArm.{side}"][:3, 3] - j[f"UpperArm.{side}"][:3, 3]
        assert np.abs(d1 - turn @ d0).max() < 1e-9  # (exactly where asked)
        rise = j[f"UpperArm.{side}"][2, 3] - j0[f"UpperArm.{side}"][2, 3]
        if shoulder == 0.0:
            assert abs(rise) < 1e-9
        else:
            assert 0.01 < rise < 0.06  # (the shoulder comes up a few centimetres)
    assert shoulder_share(20.0) == 0.0 and shoulder_share(90.0) == pytest.approx(10.0)
    assert shoulder_share(180.0) == pytest.approx(25.0)
    assert len(s.history) == n + 3
    for _ in range(3):
        s.history.undo()


def test_fingers_curl_into_a_fist_point_and_spread(figure_rig):
    s, _ = figure_rig
    a = s.animation
    rig = a.rig("Rig")
    tails = rig.tails()

    def tips(t):
        j = a.pose_joints("Rig", t, space="armature")
        inv = np.linalg.inv(j["Hand.L"])
        out = {}
        for cap in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            k = rig.index(f"{cap}3.L")
            tip = inv @ j[f"{cap}3.L"] @ np.append(tails[k], 1.0)
            out[cap] = (tip[:3], (inv @ j[f"{cap}1.L"])[:3, 3])
        return out

    rest = tips(0.0)
    rep = a.pose_character("Rig", [{"do": "curl_fingers", "side": "L", "shape": "fist"}],
                           time=1.0)["commands"][0]  # fmt: skip
    assert len(rep["joints"]) == 15
    fist = tips(1.0)
    for cap in ("Index", "Middle", "Ring", "Pinky"):
        reach0 = np.linalg.norm(rest[cap][0] - rest[cap][1])
        reach1 = np.linalg.norm(fist[cap][0] - fist[cap][1])
        assert reach1 < 0.45 * reach0, cap  # (curled up to its knuckle)
    a.pose_character("Rig", [{"do": "curl_fingers", "side": "L", "shape": "point"}], time=2.0)
    point = tips(2.0)
    assert np.abs(point["Index"][0] - rest["Index"][0]).max() < 1e-9  # (straight)
    assert np.linalg.norm(point["Middle"][0] - point["Middle"][1]) < 0.45 * np.linalg.norm(
        rest["Middle"][0] - rest["Middle"][1])  # fmt: skip
    a.pose_character("Rig", [{"do": "curl_fingers", "side": "L", "amount": 0.0,
                              "spread": 15}], time=3.0)  # fmt: skip
    spread = tips(3.0)
    gap0 = np.linalg.norm(rest["Index"][0] - rest["Pinky"][0])
    assert np.linalg.norm(spread["Index"][0] - spread["Pinky"][0]) > gap0 + 0.01
    with pytest.raises(AnimationError) as e:
        a.pose_character("Rig", [{"do": "curl_fingers", "fingers": ["toe"]}], time=1.0)
    assert e.value.code == "INVALID_ARGUMENT"
    for _ in range(3):
        s.history.undo()


def test_feet_roll_on_their_heel_ball_and_toe_tip(figure_rig):
    s, _ = figure_rig
    a = s.animation
    rig = a.rig("Rig")
    lm = rig.landmark_dict
    rest = rig.rest_model()

    def carried(j, joint, name):
        k = rig.index(joint)
        return (j[joint] @ np.linalg.solve(rest[k], np.append(lm[name], 1.0)))[:3]

    n = len(s.history)
    # (each roll at its own time: a roll is absolute, whatever was keyed before it)
    for t, angle, fixed, moves in (
        (1.0, 20.0, ("ball.L", "toe_tip.L"), "heel.L"),
        (2.0, -15.0, ("heel.L",), "toe_tip.L"),
        (3.0, 45.0, ("toe_tip.L",), "heel.L"),
        (4.0, 20.0, ("ball.L", "toe_tip.L"), "heel.L"),
    ):
        rep = a.pose_character("Rig", [{"do": "roll_foot", "side": "L", "angle": angle}],
                               time=t)  # fmt: skip
        assert rep["residuals"]["leg_ik.L"] < 1e-6
        j = a.pose_joints("Rig", t, space="armature")
        for name in fixed:
            joint = "Foot.L" if name == "heel.L" else "Toe.L"
            assert np.abs(carried(j, joint, name) - lm[name]).max() < 2e-3, (angle, name)
        joint = "Foot.L" if moves == "heel.L" else "Toe.L"
        assert carried(j, joint, moves)[2] - lm[moves][2] > 0.02, (angle, moves)
    j1, j4 = (a.pose_joints("Rig", t, space="armature") for t in (1.0, 4.0))
    assert np.abs(j1["Foot.L"] - j4["Foot.L"]).max() < 1e-9  # (the same roll, the same pose)
    assert len(s.history) == n + 4
    for _ in range(4):
        s.history.undo()
