"""Motion tools: cycle-aware handles, curve reports, keyframe reduction and resampling,
graph images, mirroring, root motion, foot contacts and planting, retargeting, baking,
trajectories, diagnostics and the motion generators -- measured, not assumed."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation import FCurve
from aether3d.animation import rotation as R
from aether3d.animation.fitting import fit_samples, resample, simplify
from aether3d.animation.generators import find_roles
from aether3d.animation.graph import graph_png, graph_svg
from aether3d.animation.motion import mirror_clip
from aether3d.animation.report import continuity, curve_report, extrema, overshoot
from aether3d.errors import AnimationError
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import world_matrices

PX = 1 / 16


def humanoid(s: Scene, name: str = "Hero", tpose: bool = True):
    """Hips at 0.95, thigh 0.45, shin 0.42, ankle 0.08 above the ground, toes on it."""
    arm = s.add_object(name)
    joints = [
        {"name": "Hips", "position": [0, 0, 0.95]},
        {"name": "Spine", "parent": "Hips", "position": [0, 0, 0.1]},
        {"name": "Chest", "parent": "Spine", "position": [0, 0, 0.2]},
        {"name": "Neck", "parent": "Chest", "position": [0, 0, 0.25]},
        {"name": "Head", "parent": "Neck", "position": [0, 0, 0.1], "tail": [0, 0, 0.22]},
    ]
    for side, sx in (("L", -1), ("R", 1)):
        if tpose:
            arm_tail, fore_pos = [0.28 * sx, 0, 0], [0.28 * sx, 0, 0]
            fore_tail, hand_pos, hand_tail = [0.25 * sx, 0, 0], [0.25 * sx, 0, 0], [0.08 * sx, 0, 0]
        else:
            arm_tail, fore_pos = [0, 0, -0.28], [0, 0, -0.28]
            fore_tail, hand_pos, hand_tail = [0, 0, -0.25], [0, 0, -0.25], [0, 0, -0.08]
        joints += [
            {"name": f"UpperArm.{side}", "parent": "Chest", "position": [0.18 * sx, 0, 0.2],
             "tail": arm_tail},
            {"name": f"ForeArm.{side}", "parent": f"UpperArm.{side}", "position": fore_pos,
             "tail": fore_tail},
            {"name": f"Hand.{side}", "parent": f"ForeArm.{side}", "position": hand_pos,
             "tail": hand_tail},
            {"name": f"Thigh.{side}", "parent": "Hips", "position": [0.1 * sx, 0, 0],
             "tail": [0, 0, -0.45]},
            {"name": f"Shin.{side}", "parent": f"Thigh.{side}", "position": [0, 0, -0.45],
             "tail": [0, 0, -0.42]},
            {"name": f"Foot.{side}", "parent": f"Shin.{side}", "position": [0, 0, -0.42],
             "tail": [0, 0.14, -0.08]},
        ]  # fmt: skip
    s.animation.add_rig(arm, joints)
    return arm


def blocky(s: Scene, name: str = "Steve"):
    """Minecraft proportions: one joint per limb (no knees, elbows or feet)."""
    arm = s.add_object(name)
    s.animation.add_rig(
        arm,
        [
            {"name": "Hips", "position": [0, 0, 12 * PX]},
            {"name": "Body", "parent": "Hips", "position": [0, 0, 0], "tail": [0, 0, 12 * PX]},
            {"name": "Head", "parent": "Body", "position": [0, 0, 12 * PX], "tail": [0, 0, 8 * PX]},
            *[
                {"name": f"Arm.{side}", "parent": "Body", "position": [x * PX, 0, 10 * PX],
                 "tail": [0, 0, -12 * PX]}
                for side, x in (("L", -6), ("R", 6))
            ],
            *[
                {"name": f"Leg.{side}", "parent": "Hips", "position": [x * PX, 0, 0],
                 "tail": [0, 0, -12 * PX]}
                for side, x in (("L", -2), ("R", 2))
            ],
        ],
    )  # fmt: skip
    return arm


# --------------------------------------------------------------------------------------- #
# Cycle-aware automatic handles
# --------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["cycle", "cycle_offset"])
def test_automatic_handles_are_smooth_across_a_loop(mode):
    keys = [(0.0, 0.0), (0.3, 1.0), (0.5, 0.2), (0.8, -0.7), (1.0, 0.0 if mode == "cycle" else 0.5)]
    c = FCurve.from_keys(keys, handles="auto").replace(extrapolation=mode)
    v, d, a = c.evaluate(np.array([1.0 - 1e-9, 1.0 + 1e-9]), order=2)
    assert d[0] == pytest.approx(d[1], abs=1e-6)  # C1 across the wrap
    # The end keys take the slope of the parabola through their wrapped neighbours.
    delta = 0.5 if mode == "cycle_offset" else 0.0
    t_prev, v_prev = 0.8 - 1.0, -0.7 - delta
    s1, s2 = (0.0 - v_prev) / (0.0 - t_prev), (1.0 - 0.0) / 0.3
    expect = (s1 * 0.3 + s2 * 0.2) / 0.5
    assert c.slope(0, "right") == pytest.approx(expect, rel=1e-12)
    flat = FCurve.from_keys(keys, handles="auto")  # constant ends: one-sided slopes
    assert flat.slope(0, "right") == pytest.approx(1.0 / 0.3, rel=1e-12)


def test_mirrored_ends_are_flat():
    c = FCurve.from_keys([(0, 0), (1, 2), (2, 1)], handles="auto").replace(extrapolation="mirror")
    assert c.slope(0, "right") == 0.0 and c.slope(2, "left") == 0.0


# --------------------------------------------------------------------------------------- #
# Curve reports
# --------------------------------------------------------------------------------------- #


def test_reports_find_corners_steps_extrema_and_overshoot_exactly():
    c = FCurve.from_keys(
        [
            {"time": 0, "value": 0, "interpolation": "linear"},
            {"time": 1, "value": 2, "interpolation": "linear"},
            {"time": 2, "value": 1, "interpolation": "constant"},
            {"time": 3, "value": 3, "interpolation": "back", "ease": "out"},
            {"time": 4, "value": 4},
        ]
    )
    found = {(d["key"], d["order"], d["intended"]) for d in continuity(c)}
    assert (1, 1, False) in found  # a linear corner: velocity jumps by -3
    assert (3, 0, True) in found  # the constant key's step (value 1 -> 3)
    corner = next(d for d in continuity(c) if d["key"] == 1)
    assert corner["jump"] == pytest.approx(-3.0, abs=1e-12)
    # back-out overshoot: the closed form 1 + 4 s^3 / (27 (s + 1)^2) of the unit span
    s_ = 1.70158
    peak = 1.0 + 4 * s_**3 / (27 * (s_ + 1) ** 2)
    over = overshoot(c)
    assert len(over) == 1 and over[0]["segment"] == 3
    assert over[0]["value"] == pytest.approx(3.0 + peak, abs=1e-9)
    ex = extrema(c)
    assert any(e["kind"] == "max" and e["segment"] == 3 for e in ex)
    rep = curve_report(c, samples=5)
    assert rep["keys"] == 5 and rep["interpolation"]["linear"] == 2
    assert len(rep["samples"]["time"]) == 5
    json.dumps(rep)


def test_auto_clamped_curves_report_no_overshoot():
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.uniform(0.1, 1.0, 30))
    v = rng.normal(size=30)
    c = FCurve.from_arrays(t, v, handles="auto_clamped")
    assert overshoot(c, relative=1e-9) == []
    loose = FCurve.from_arrays(t, v, handles="auto")
    assert overshoot(loose, relative=1e-9)  # (Catmull-Rom slopes overshoot)


# --------------------------------------------------------------------------------------- #
# Reduction, resampling, fitting
# --------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("tolerance", [1e-2, 1e-4])
def test_simplify_keeps_every_point_within_tolerance(tolerance):
    t = np.linspace(0, 10, 601)
    v = np.sin(t) + 0.3 * np.sin(3.1 * t)
    c = FCurve.from_arrays(t, v, handles="auto")
    out, rep = simplify(c, tolerance)
    dense = np.linspace(0, 10, 100_001)
    true = float(np.max(np.abs(out.evaluate(dense) - c.evaluate(dense))))
    assert rep["keys_after"] < rep["keys_before"] / 5
    assert rep["max_error"] <= tolerance
    assert true <= tolerance * 1.05  # (measured between the samples too)


def test_simplify_keeps_steps_and_rejects_bad_tolerances():
    c = FCurve.from_keys(
        [
            {"time": 0, "value": 0},
            {"time": 1, "value": 1, "interpolation": "constant"},
            {"time": 2, "value": 5},
            {"time": 3, "value": 6},
        ]
    )
    out, _ = simplify(c, 1e-6)
    assert out.evaluate(1.999) == 1.0 and out.evaluate(2.0) == 5.0
    for bad in (0, -1, float("nan"), True):
        with pytest.raises(AnimationError):
            simplify(c, bad)


def test_resample_puts_keys_on_frames_with_exact_values_and_slopes():
    c = FCurve.from_keys([(0, 0), (1, 3), (2, 1)], handles="auto")
    r = resample(c, "24000/1001")
    assert len(r) == 48  # frames 0..47 within 0..2 s at 23.976 fps
    assert np.allclose(r.values, c.evaluate(r.times), atol=0)
    v, d = c.evaluate(r.times, order=1)
    assert np.allclose([r.slope(i, "right") for i in range(len(r))], d, rtol=1e-12, atol=1e-12)


def test_fit_samples_follows_motion_capture_like_data():
    rng = np.random.default_rng(9)
    t = np.linspace(0, 4, 481)
    v = np.cumsum(rng.normal(scale=0.01, size=t.size)) + np.sin(2 * t)
    curve, rep = fit_samples(t, v, 0.02)
    assert np.max(np.abs(curve.evaluate(t) - v)) <= 0.02
    assert rep["keys"] < 200
    with pytest.raises(AnimationError):
        fit_samples(t[::-1], v, 0.1)


# --------------------------------------------------------------------------------------- #
# Graph images
# --------------------------------------------------------------------------------------- #


def test_graphs_are_well_formed_and_deterministic():
    a = FCurve.from_keys([(0, 0), (0.4, 1.2), (1.0, 0.8), (1.6, 0.0)], handles="auto")
    b = FCurve.from_keys([(0, 0), (0.5, 1), (1.6, 0)], interpolation="linear")
    svg = graph_svg({"eased": a, "linear <b>": b}, title="Arm & leg")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    texts = [e.text for e in root.iter() if e.tag.endswith("text")]
    assert "linear <b>" in texts and "Arm & leg" in texts  # (escaped in the file)
    assert svg == graph_svg({"eased": a, "linear <b>": b}, title="Arm & leg")
    png = graph_png({"eased": a}, plots=("value", "velocity"), width=400, plot_height=120)
    assert png.dtype == np.uint8 and png.shape[1] == 400 and png.shape[2] == 3
    assert np.count_nonzero(png.min(axis=2) < 200) > 500  # ink was drawn
    assert np.array_equal(png, graph_png({"eased": a}, plots=("value", "velocity"), width=400,
                                         plot_height=120))  # fmt: skip
    with pytest.raises(AnimationError):
        graph_svg({"a": a}, plots=("jerkiness",))


def test_scene_graphs_take_channels_or_a_clip():
    s = Scene("G")
    cube = s.add_cube(name="Cube")
    a = s.animation
    a.keyframe(cube, "transform.position.x", 0, 0)
    a.keyframe(cube, "transform.position.x", 1, 2)
    a.keyframe(cube, "transform.position.z", 0.5, 1)
    svg = a.graph([(cube, "transform.position.x")])
    assert "Cube: transform.position.x" in svg
    img = a.graph(clip=a.action(cube), format="png", width=300)
    assert img.shape[1] == 300


# --------------------------------------------------------------------------------------- #
# Mirroring, root motion
# --------------------------------------------------------------------------------------- #


def test_mirroring_swaps_sides_reflects_and_is_an_involution():
    s = Scene("M")
    arm = humanoid(s, tpose=False)
    a = s.animation
    clip = a.add_clip("Reach")
    clip.keyframe('joints["UpperArm.L"].rotation', 0.0, [0, 0, 0, 1])
    clip.keyframe(
        'joints["UpperArm.L"].rotation', 1.0, R.from_euler([[30.0, 10.0, -20.0]])[0].tolist()
    )
    clip.set_channel('joints["Hips"].position.x', [(0, 0.0), (1, 0.2)])
    clip.set_channel('joints["Spine"].rotation_euler.y', [(0, 0.0), (1, 15.0)])
    rig = a.rig(arm)
    m = mirror_clip(clip.data, rig)
    assert (
        'joints["UpperArm.R"].rotation' in m.paths
        and 'joints["UpperArm.L"].rotation' not in m.paths
    )
    q = m.channel('joints["UpperArm.R"].rotation').evaluate(1.0)
    e = R.to_euler(q[None])[0]
    assert np.allclose(e, [30.0, -10.0, 20.0], atol=1e-9)
    assert m.channel('joints["Hips"].position.x').evaluate(1.0) == pytest.approx(-0.2)
    assert m.channel('joints["Spine"].rotation_euler.y').evaluate(1.0) == pytest.approx(-15.0)
    back = mirror_clip(m, rig)
    assert back.same(clip.data)
    handle = a.mirror_clip(clip, arm)
    assert handle.name == "Reach Mirrored"


def test_root_motion_extraction_keeps_the_world_motion():
    s = Scene("Root")
    arm = humanoid(s)
    a = s.animation
    a.set_timeline(start=0, end=2)
    clip = a.add_clip("Stride")
    clip.set_channel('joints["Hips"].position.y', [(0, 0.0), (2, 1.6)], interpolation="linear")
    clip.set_channel('joints["Hips"].position.x', [(0, 0.0), (1, 0.05), (2, 0.0)])
    a.set_action(arm, clip)
    before = [a.pose_joints(arm, t, "world")["Hips"][:3, 3].copy() for t in (0.3, 1.1, 2.0)]
    rep = a.extract_root_motion(clip, "Hips")
    assert rep["moved"]["y"] == pytest.approx(1.6)
    after = [a.pose_joints(arm, t, "world")["Hips"][:3, 3].copy() for t in (0.3, 1.1, 2.0)]
    assert np.allclose(before, after, atol=1e-12)
    joint_y = a.clip(clip.name).channel('joints["Hips"].position.y')
    assert joint_y.evaluate(1.5) == 0.0
    a.extract_root_motion(clip, "Hips", mode="in_place")  # (already extracted: nothing left)
    with pytest.raises(AnimationError):
        a.extract_root_motion(clip, "Hips", mode="sideways")


# --------------------------------------------------------------------------------------- #
# Retargeting
# --------------------------------------------------------------------------------------- #


def test_retargeting_matches_bone_directions_across_proportions_and_rest_poses():
    s = Scene("Retarget")
    src = humanoid(s, "Small", tpose=True)
    tgt = humanoid(s, "Big", tpose=False)
    tgt_rig = s.animation.rig(tgt).to_json()
    for j in tgt_rig["joints"]:  # 1.6x bigger
        j["position"] = [1.6 * x for x in j["position"]]
        if "tail" in j:
            j["tail"] = [1.6 * x for x in j["tail"]]
    s.animation.set_rig(tgt, tgt_rig)
    a = s.animation
    clip = a.add_clip("Kick")
    clip.keyframe('joints["Thigh.L"].rotation', 0.0, [0, 0, 0, 1])
    clip.keyframe('joints["Thigh.L"].rotation', 1.0, R.from_euler([[50.0, 0, 0]])[0].tolist())
    clip.keyframe('joints["UpperArm.R"].rotation', 1.0, R.from_euler([[0, 0, -40.0]])[0].tolist())
    clip.set_channel('joints["Hips"].position.y', [(0, 0.0), (1, 0.5)], interpolation="linear")
    rep = a.retarget(src, tgt, "Kick", fps=10)
    assert rep["root_scale"] == pytest.approx(1.6)
    assert set(rep["rest_mismatch"]) == {"UpperArm.L", "UpperArm.R", "ForeArm.L", "ForeArm.R",
                                         "Hand.L", "Hand.R"}  # T-pose vs arms down  # fmt: skip
    a.set_action(src, "Kick")
    a.set_action(tgt, rep["clip"])
    ps, pt = a.pose_joints(src, 1.0, "armature"), a.pose_joints(tgt, 1.0, "armature")
    rs, rt = a.rig(src), a.rig(tgt)

    def direction(pose, rig, name):
        v = pose[name][:3, :3] @ rig.tails()[rig.index(name)]
        return v / np.linalg.norm(v)

    for name in ("Thigh.L", "Shin.L", "Foot.L"):
        assert np.allclose(direction(ps, rs, name), direction(pt, rt, name), atol=1e-9)

    # The arm turns by the same rotation (armature space) from its own, different rest.
    def turn(pose, rig, name):
        rest = rig.rest_model()[rig.index(name)][:3, :3]
        posed = pose[name][:3, :3]
        return R.from_matrices(
            (posed @ np.linalg.inv(rest))[None]
            / np.cbrt(np.linalg.det(posed @ np.linalg.inv(rest)))
        )[0]

    q_src, q_tgt = turn(ps, rs, "UpperArm.R"), turn(pt, rt, "UpperArm.R")
    assert abs(float(q_src @ q_tgt)) == pytest.approx(1.0, abs=1e-12)
    assert pt["Hips"][1, 3] == pytest.approx(0.5 * 1.6)
    with pytest.raises(AnimationError):
        a.retarget(src, tgt, "Kick", mapping={"Thigh.L": "Nope"})


def test_role_detection_reads_common_naming_schemes():
    s = Scene("Names")
    arm = s.add_object("Mixamo")
    names = [("mixamorig:Hips", None), ("mixamorig:Spine", "mixamorig:Hips"),
             ("mixamorig:Spine1", "mixamorig:Spine"), ("mixamorig:Spine2", "mixamorig:Spine1"),
             ("mixamorig:Neck", "mixamorig:Spine2"), ("mixamorig:Head", "mixamorig:Neck"),
             ("mixamorig:LeftShoulder", "mixamorig:Spine2"),
             ("mixamorig:LeftArm", "mixamorig:LeftShoulder"),
             ("mixamorig:LeftForeArm", "mixamorig:LeftArm"),
             ("mixamorig:LeftHand", "mixamorig:LeftForeArm"),
             ("mixamorig:LeftUpLeg", "mixamorig:Hips"),
             ("mixamorig:LeftLeg", "mixamorig:LeftUpLeg"),
             ("mixamorig:LeftFoot", "mixamorig:LeftLeg"),
             ("mixamorig:LeftToeBase", "mixamorig:LeftFoot")]  # fmt: skip
    s.animation.add_rig(arm, [{"name": n, **({"parent": p} if p else {}), "position": [0, 0, 0.1]}
                              for n, p in names])  # fmt: skip
    roles = find_roles(s.animation.rig(arm))
    assert roles["hips"] == "mixamorig:Hips" and roles["spine"] == "mixamorig:Spine"
    assert roles["chest"] == "mixamorig:Spine2"
    assert roles["upper_arm.L"] == "mixamorig:LeftArm"  # not the clavicle
    assert roles["lower_arm.L"] == "mixamorig:LeftForeArm"
    assert roles["thigh.L"] == "mixamorig:LeftUpLeg" and roles["shin.L"] == "mixamorig:LeftLeg"
    assert roles["foot.L"] == "mixamorig:LeftFoot"
    b = blocky(s)
    roles = find_roles(s.animation.rig(b))
    assert roles["thigh.R"] == "Leg.R" and "shin.R" not in roles and roles["spine"] == "Body"
    with pytest.raises(AnimationError):
        find_roles(s.animation.rig(b), {"elbow": "Arm.L"})


# --------------------------------------------------------------------------------------- #
# Generators, contacts, planting
# --------------------------------------------------------------------------------------- #


def contacts(a, arm, feet, end):
    rep = a.foot_contacts(arm, feet, start=0.0, end=end, rate=120, height=0.02, speed=0.25)
    return (max(f["sliding_max"] for f in rep["feet"].values()),
            max(f["penetration"] for f in rep["feet"].values()), rep)  # fmt: skip


def test_generated_walk_loops_plants_its_feet_and_is_editable():
    s = Scene("Walk")
    arm = humanoid(s)
    a = s.animation
    rep = a.generate(arm, "walk")
    period = rep["period"]
    assert rep["ik_residual"] == pytest.approx(0.0, abs=1e-9)
    assert rep["max_error"]["degrees"] <= 0.05
    # Seamless: the pose one period later is the same (the armature moved on by a stride).
    p0 = a.pose_joints(arm, 0.3, "armature")
    p1 = a.pose_joints(arm, 0.3 + period, "armature")
    for name in ("Thigh.L", "Shin.R", "Foot.L", "Hips", "UpperArm.R"):
        assert np.allclose(p0[name], p1[name], atol=1e-9)
    w0 = world_matrices(s, animation_overlay(s, 0.3))[arm.id][:3, 3]
    w1 = world_matrices(s, animation_overlay(s, 0.3 + period))[arm.id][:3, 3]
    assert w1[1] - w0[1] == pytest.approx(rep["travel"], abs=1e-9)
    slide, pen, _ = contacts(a, arm, ["Foot.L", "Foot.R"], 2 * period)
    assert slide < 0.003 and pen < 0.001  # metres (fitting within 0.05 degrees)
    clip = a.clip(rep["clip"])
    meta = clip.to_json()["metadata"]["generator"]
    assert meta["kind"] == "walk" and meta["version"] == 1
    curve = clip.channel('joints["Thigh.L"].rotation_euler.x')
    assert isinstance(curve, FCurve) and curve.extrapolation == ("cycle", "cycle")
    clip.edit_keys('joints["Thigh.L"].rotation_euler.x', "scale_values", factor=1.2)  # editable


def test_generated_jump_flies_a_ballistic_arc():
    s = Scene("Jump")
    arm = humanoid(s)
    a = s.animation
    rep = a.generate(arm, "jump", height=0.5, gravity=9.81)
    hips = [a.pose_joints(arm, t, "armature")["Hips"][2, 3] for t in (rep["takeoff"], rep["apex"])]
    rise = 0.5 * 0.87  # height is in leg lengths (this rig: 0.87)
    assert hips[1] - hips[0] == pytest.approx(rise, abs=2e-3)
    assert rep["landing"] - rep["takeoff"] == pytest.approx(2 * math.sqrt(2 * rise / 9.81))
    slide, pen, _ = contacts(a, arm, ["Foot.L", "Foot.R"], rep["duration"])
    assert slide < 0.003 and pen < 0.001


@pytest.mark.parametrize("tpose", [True, False])
def test_a_wave_raises_the_forearm_and_swings_it_side_to_side(tpose):
    s = Scene("Wave")
    arm = humanoid(s, tpose=tpose)
    a = s.animation
    rep = a.generate(arm, "wave", side="L", waves=2, rate=2.0)
    forward = np.array([0.0, 1.0, 0.0])  # (the generators' default forward, +y)

    def forearm(t):
        pose = a.pose_joints(arm, t, "armature")
        d = pose["Hand.L"][:3, 3] - pose["ForeArm.L"][:3, 3]
        return d / np.linalg.norm(d)

    up = forearm(0.45)  # (raised; the swing starts here)
    assert up[2] > math.cos(math.radians(11)) and up @ forward > 0.1  # vertical, leaning forward
    # (quarter and three-quarter swings: +-22 degrees about the forward axis, opposite sides)
    swing = [math.degrees(math.atan2(forearm(0.45 + q / 2.0)[0], forearm(0.45 + q / 2.0)[2]))
             for q in (0.25, 0.75)]  # fmt: skip
    assert swing[0] == pytest.approx(-swing[1], abs=1.0) and 20 < abs(swing[0]) < 24
    assert forearm(rep["duration"])[2] < -0.99  # (lowered again, hanging)


@pytest.mark.parametrize("kind", ["idle", "run", "land", "wave", "head_turn"])
def test_every_motion_generates_on_a_blocky_rig_without_sinking(kind):
    s = Scene("Blocky")
    arm = blocky(s)
    a = s.animation
    rep = a.generate(arm, kind)
    end = rep["duration"] * (2 if rep["cyclic"] else 1)
    slide, pen, _ = contacts(a, arm, ["Leg.L", "Leg.R"], end)
    assert pen < 0.001
    assert slide < 0.01  # (a straight leg steps off its toe at the end of a landing)
    assert rep["roles"]["thigh.L"] == "Leg.L"


def test_generator_arguments_are_checked_and_failures_change_nothing():
    s = Scene("Bad")
    arm = humanoid(s)
    a = s.animation
    revision = s.revision
    for kind, options in (("dance", {}), ("walk", {"cadence": 99}), ("walk", {"speed": 1.0}),
                          ("jump", {"height": float("nan")}), ("walk", {"forward": "+z"}),
                          ("walk", {"tolerance": 0}), ("land", {"style": "dramatic"})):  # fmt: skip
        with pytest.raises(AnimationError):
            a.generate(arm, kind, **options)
    assert s.revision == revision and a.clips() == []
    empty = s.add_object("NoRig")
    with pytest.raises(AnimationError):
        a.generate(empty, "walk")


def test_planting_feet_removes_sliding():
    s = Scene("Plant")
    arm = humanoid(s)
    a = s.animation
    a.set_timeline(start=0.0, end=2.0)
    # The hips glide forward (and a little down) while the legs stay still: the feet skate
    # along. Planted where they start, they stay within the legs' reach.
    a.set_channel(arm, 'joints["Hips"].position.y', [(0, 0.0), (2, 0.2)], interpolation="linear")
    a.set_channel(arm, 'joints["Hips"].position.z', [(0, 0.95), (2, 0.91)], interpolation="linear")
    slide, _, rep = contacts(a, arm, ["Foot.L", "Foot.R"], 2.0)
    assert slide == pytest.approx(0.2, abs=0.01)
    out = a.plant_feet(arm, {"Foot.L": ["Thigh.L", "Shin.L", "Foot.L"],
                             "Foot.R": ["Thigh.R", "Shin.R", "Foot.R"]},
                       height=0.1, speed=0.5)  # fmt: skip
    assert out["before"]["Foot.L"] == pytest.approx(0.2, abs=0.01)
    assert out["after"]["Foot.L"] < 0.002 and out["after"]["Foot.R"] < 0.002
    assert s.history.undo()  # one step
    assert a.constraints(arm) in ((), [], None) or len(a.constraints(arm)) == 0


def test_planting_a_rushed_walk_pins_whole_stances_without_pushing_feet_into_the_ground():
    # The root motion scaled by 1.3: the body outruns its steps and the feet skate, fast
    # enough that a 0.3 m/s contact test breaks each stance into pieces. Planting finds the
    # whole stances, pins each in one touch-down pose (position and orientation of one
    # instant), and never lowers a heel or toe below the ground.
    s = Scene("Rushed")
    arm = humanoid(s)
    a = s.animation
    rep = a.generate(arm, "walk")
    a.set_timeline(start=0.0, end=3.0)
    a.edit_keys(arm, "transform.position.y", "scale_values", clip=rep["clip"], factor=1.3,
                pivot=0.0)  # fmt: skip
    feet = ["Foot.L", "Foot.R"]
    before = a.foot_contacts(arm, feet)
    out = a.plant_feet(arm, {f: [f"Thigh.{f[-1]}", f"Shin.{f[-1]}", f] for f in feet})
    after = a.foot_contacts(arm, feet)
    cycles = 3.0 / rep["period"]
    for f in feet:
        assert before["feet"][f]["sliding_max"] > 0.03  # (skating)
        assert len(out["contacts"][f]) <= math.ceil(cycles) + 1  # (whole stances, not pieces)
        assert after["feet"][f]["sliding_max"] < 0.25 * before["feet"][f]["sliding_max"]
        assert after["feet"][f]["penetration"] <= max(before["feet"][f]["penetration"], 5e-4)
        assert out["before"][f] == before["feet"][f]["sliding_max"]  # (measured alike)


def test_contact_measurements_share_the_feedback_checks_defaults():
    import inspect

    from aether3d.animation.motion import foot_contacts
    from aether3d.feedback.schema import expectations_schema

    defaults = {k: p.default for k, p in inspect.signature(foot_contacts).parameters.items()}
    check = next(c for c in expectations_schema()["properties"]["checks"]["items"]["oneOf"]
                 if c["properties"]["check"].get("const") == "foot_contacts")  # fmt: skip
    for key in ("height", "speed"):
        assert check["properties"][key]["default"] == defaults[key]


# --------------------------------------------------------------------------------------- #
# Cameras, baking, trajectories, diagnostics
# --------------------------------------------------------------------------------------- #


def test_follow_camera_keeps_its_offset_and_aims_at_the_target():
    s = Scene("Follow")
    car = s.add_cube(name="Car")
    cam = s.add_camera(name="Cam", position=(0, -5, 2))
    a = s.animation
    a.set_timeline(start=0, end=3)
    a.set_channel(car, "transform.position.x", [(0, 0.0), (3, 6.0)])
    rigid = a.follow_camera(cam, car, offset=(0, -6, 2.5), lag=0.0, aim=(0, 0, 0.5))
    assert rigid["max_error"] <= 0.005
    for t in (0.0, 1.3, 3.0):
        worlds = world_matrices(s, animation_overlay(s, t))
        car_pos = worlds[car.id][:3, 3]
        cam_w = worlds[cam.id]
        assert np.allclose(cam_w[:3, 3], car_pos + [0, -6, 2.5], atol=0.006)
        forward = -cam_w[:3, 1] / np.linalg.norm(cam_w[:3, 1])
        to = car_pos + [0, 0, 0.5] - cam_w[:3, 3]
        assert np.allclose(forward, to / np.linalg.norm(to), atol=1e-9)
    lagged = a.follow_camera(cam, car, offset=(0, -6, 2.5), lag=0.5)
    x_mid = world_matrices(s, animation_overlay(s, 1.5))[cam.id][0, 3]
    car_mid = world_matrices(s, animation_overlay(s, 1.5))[car.id][0, 3]
    assert x_mid < car_mid  # behind: it lags
    assert lagged["max_error"] <= 0.005


def test_baking_turns_constraints_into_plain_keys():
    s = Scene("Bake")
    ball = s.add_cube(name="Ball")
    eye = s.add_cube(name="Eye", position=(0, -3, 1))
    a = s.animation
    a.set_timeline(start=0, end=2, fps=24)
    a.set_channel(ball, "transform.position.x", [(0, -2.0), (2, 2.0)])
    a.add_constraint(eye, {"name": "watch", "type": "look_at", "target": {"object": ball.id}})
    want = {t: world_matrices(s, animation_overlay(s, t))[eye.id].copy() for t in (0.5, 1.25)}
    rep = a.bake([eye])
    a.remove_constraint(eye, "watch")
    a.set_action(eye, rep["clips"]["Eye"])
    for t, m in want.items():  # on frames: exact up to float rounding
        if abs(t * 24 - round(t * 24)) < 1e-9:
            got = world_matrices(s, animation_overlay(s, t))[eye.id]
            assert np.allclose(got, m, atol=1e-9)


def test_trajectories_and_diagnostics_measure_motion():
    s = Scene("Traj")
    cube = s.add_cube(name="Cube")
    a = s.animation
    a.set_timeline(start=0, end=2)
    a.set_channel(cube, "transform.position.x", [(0, 0.0), (2, 4.0)], interpolation="linear")
    tr = a.trajectory(cube, rate=120)
    assert tr["path_length"] == pytest.approx(4.0, abs=1e-9)
    assert tr["speed"]["max"] == pytest.approx(2.0, abs=1e-9)
    a.set_channel(cube, "transform.position.y", [(0, 0.0), (1, 1.0), (2, 1.0)],
                  interpolation="linear")  # fmt: skip
    found = a.diagnose()
    codes = [f["code"] for f in found["findings"]]
    assert "VELOCITY_DISCONTINUITY" in codes
    json.dumps(found)
