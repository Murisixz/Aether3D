"""World-space motion quality (``anim.motion_report``): every detector on a motion made
to have (or not have) its defect, and the report's contract (sampled, read-only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aether3d import Scene
from aether3d.serialization import scene_to_dict


def codes(report, code):
    return [f for f in report["findings"] if f["code"] == code]


def stage(end=4.0):
    scene = Scene()
    scene.animation.set_timeline(fps=24, start=0, end=end)
    return scene, scene.animation


def test_acceleration_steps_where_curves_are_only_c1():
    scene, anim = stage()
    keys = [(0, 0), (0.7, 1.0), (1.3, 1.6), (2.0, 3.5), (3.0, 4.0), (4, 4.2)]
    for name, smoothing in (("Local", "local"), ("Continuous", "continuous")):
        anim.set_channel(scene.add_cube(name, size=0.2), "transform.position.y", keys,
                         handles="auto", smoothing=smoothing)  # fmt: skip
    r = anim.motion_report(thresholds={"acceleration_step": 0.5})
    local = {round(f["time"], 2) for f in codes(r, "ACCELERATION_STEP") if f["location"] == "Local"}
    smooth = [f for f in codes(r, "ACCELERATION_STEP") if f["location"] == "Continuous"]
    assert {0.7, 1.3, 2.0, 3.0} <= local
    assert smooth == []  # C2 at every automatic key
    by_name = {t["target"]: t for t in r["targets"]}
    assert by_name["Continuous"]["jitter"]["position_mm"] < 0.05
    assert r["basis"] == "sampled" and r["thresholds"]["acceleration_step"] == 0.5


def test_teleports_velocity_jumps_and_pauses():
    scene, anim = stage()
    anim.set_channel(scene.add_cube("Jump", size=0.2), "transform.position.x",
                     [(0, 0), (2, 1), {"time": 2, "value": 3}, (4, 4)])  # fmt: skip
    anim.set_channel(scene.add_cube("Kink", size=0.2), "transform.position.x",
                     [{"time": 0, "value": 0, "interpolation": "linear"},
                      {"time": 2, "value": 2, "interpolation": "linear"},
                      {"time": 4, "value": 0}])  # fmt: skip
    anim.set_channel(scene.add_cube("Hitch", size=0.2), "transform.position.x",
                     [(0, 0), (1.5, 1), (1.7, 1), (4, 3)])  # fmt: skip
    r = anim.motion_report()
    jump = codes(r, "POSITION_JUMP")
    assert (
        len(jump) == 1
        and jump[0]["location"] == "Jump"
        and jump[0]["time"] == pytest.approx(2.0, abs=0.01)
    )
    assert not [f for f in codes(r, "VELOCITY_JUMP") if f["location"] == "Jump"]
    kink = [f for f in codes(r, "VELOCITY_JUMP") if f["location"] == "Kink"]
    assert len(kink) == 1 and kink[0]["time"] == pytest.approx(2.0, abs=0.02)
    assert kink[0]["magnitude"] == pytest.approx(2.0, rel=0.05)  # +1 -> -1 m/s
    pauses = [f for f in codes(r, "PAUSE") if f["location"] == "Hitch"]
    assert len(pauses) == 1 and 1.4 < pauses[0]["time"] < 1.7
    assert pauses[0]["severity"] == "info"


def test_jitter_and_rotation_snaps():
    scene, anim = stage()
    shaky = scene.add_cube("Shaky", size=0.2)
    anim.set_channel(shaky, "transform.position.x", [(0, 0), (4, 2)],
                     modifiers=[{"type": "noise", "strength": 0.004, "scale": 0.02}])  # fmt: skip
    calm = scene.add_cube("Calm", size=0.2)
    anim.set_channel(calm, "transform.position.x", [(0, 0), (4, 2)])
    snap = scene.add_cube("Snap", size=0.2)
    anim.set_channel(snap, "transform.rotation_euler.z",
                     [(0, 0), (2, 10), {"time": 2, "value": 80}, (4, 90)])  # fmt: skip
    r = anim.motion_report()
    jit = [f for f in codes(r, "JITTER") if f["location"] == "Shaky"]
    assert len(jit) == 1 and jit[0]["position_mm"] > 0.5  # the threshold
    assert not [f for f in codes(r, "JITTER") if f["location"] == "Calm"]
    snaps = [f for f in codes(r, "ROTATION_SNAP") if f["location"] == "Snap"]
    assert len(snaps) == 1 and snaps[0]["magnitude"] == pytest.approx(70.0, abs=1.0)


def test_loop_seams():
    scene, anim = stage(end=2.0)
    good = scene.add_cube("Good", size=0.2)
    anim.set_channel(good, "transform.position.z", [(0, 0), (0.5, 0.3), (1.2, -0.2), (2, 0)],
                     handles="auto", extrapolation="cycle")  # fmt: skip
    bad = scene.add_cube("Bad", size=0.2)
    anim.set_channel(
        bad,
        "transform.position.z",
        [(0, 0), (0.5, 0.3), (1.2, -0.2), (2, 0)],
        handles="auto",
        extrapolation="linear",
    )  # natural ends: unequal slopes
    r = anim.motion_report(loop=True)
    seams = {f["location"] for f in codes(r, "LOOP_SEAM")}
    assert "Bad" in seams and "Good" not in seams


def test_pole_flips_of_an_ik_leg():
    scene, anim = stage(end=2.0)
    leg = scene.add_object("Leg")
    scene.add_object("Target", position=(0, 0.2, 0.3))  # within reach: bent
    pole = scene.add_object("Pole", position=(0, 1, 0.5))
    anim.add_rig(leg, [
        {"name": "Hip", "position": [0, 0, 1]},
        {"name": "Knee", "parent": "Hip", "position": [0, 0.05, -0.45]},
        {"name": "Ankle", "parent": "Knee", "position": [0, -0.05, -0.45]},
    ], constraints=[{"name": "LegIK", "type": "ik", "chain": ["Hip", "Knee", "Ankle"],
                     "target": {"object": "Target"}, "pole": {"object": "Pole"},
                     "solver": "two_bone"}])  # fmt: skip
    anim.set_channel(pole, "transform.position.y",
                     [{"time": 0, "value": 1.0, "interpolation": "constant"},
                      {"time": 1.0, "value": -1.0}])  # fmt: skip
    r = anim.motion_report([{"object": "Leg", "joints": ["Hip", "Knee", "Ankle"]}])
    flips = codes(r, "POLE_FLIP")
    assert len(flips) == 1 and flips[0]["time"] == pytest.approx(1.0, abs=0.02)
    assert flips[0]["magnitude"] > 150.0
    still = anim.motion_report([{"object": "Leg", "joints": "all"}], end=0.9)
    assert not codes(still, "POLE_FLIP")


def test_ground_penetration_is_measured_on_vertices():
    scene, anim = stage(end=2.0)
    ball = scene.add_sphere("Ball", radius=0.25, position=(0, 0, 1))
    anim.set_channel(ball, "transform.position.z", [(0, 1.0), (1, 0.2), (2, 1.0)])
    r = anim.motion_report(thresholds={"ground": 0.0})
    hits = codes(r, "GROUND_PENETRATION")
    assert len(hits) == 1 and hits[0]["magnitude"] == pytest.approx(0.05, abs=0.003)
    assert hits[0]["time"] == pytest.approx(1.0, abs=0.04)


def test_cameras_framing_obstruction_and_cuts():
    scene, anim = stage(end=3.0)
    hero = scene.add_cube("Hero", size=0.5, position=(0, 0, 0.25))
    anim.set_channel(hero, "transform.position.x", [(0, -1), (3, 6)])
    cam = scene.add_camera("Cam", position=(0, -6, 1.5), look_at=(0, 0, 0.25))
    scene.add_cube("Wall", size=1.0, position=(1.5, -3, 0.5))
    r = anim.motion_report(["Cam"], subject="Hero")
    out = codes(r, "SUBJECT_OUT_OF_FRAME")
    assert out and out[0]["severity"] == "error"
    assert codes(r, "SUBJECT_OBSTRUCTED")
    framing = r["targets"][0]["framing"]
    assert framing["aim_error_deg"]["max"] > 40.0
    # Shots: a cut is intended, never a jump.
    cam2 = scene.add_camera("Cam2", position=(8, -6, 2), look_at=(0, 0, 0))
    anim.set_shots([{"camera": "Cam", "start": 0, "end": 1.5, "name": "A"},
                    {"camera": "Cam2", "start": 1.5, "end": 3.0, "name": "B"}])  # fmt: skip
    del cam, cam2
    shots = anim.motion_report([{"camera": "shots"}])
    assert codes(shots, "CUT") and not codes(shots, "POSITION_JUMP")


def test_the_report_changes_nothing():
    scene, anim = stage()
    cube = scene.add_cube("Cube", size=0.2)
    anim.set_channel(cube, "transform.position.x", [(0, 0), (2, 1), (4, 0)])
    anim.time = 1.25
    before = json.dumps(scene_to_dict(scene), sort_keys=True)
    revision = scene.revision
    report = anim.motion_report(subject="Cube")
    assert json.dumps(scene_to_dict(scene), sort_keys=True) == before
    assert scene.revision == revision and anim.time == 1.25
    assert report["targets"][0]["target"] == "Cube"
    with pytest.raises(Exception) as err:
        anim.motion_report(thresholds={"nope": 1})
    assert getattr(err.value, "code", "") == "INVALID_ARGUMENT"
    assert np.isfinite(report["targets"][0]["speed"]["max"])


def test_feedback_measures_motion_and_keeps_only_corrections_that_help():
    from aether3d.feedback import FeedbackSession

    scene, anim = stage()
    cube = scene.add_cube("Cube", size=0.2)
    anim.set_channel(cube, "transform.position.y",
                     [(0, 0), (0.7, 1.0), (1.3, 1.6), (2.0, 3.5), (3.0, 4.0), (4, 4.2)],
                     handles="auto", smoothing="local")  # fmt: skip
    loop = scene.add_cube("Loop", size=0.2, position=(1, 0, 0))
    anim.set_channel(loop, "transform.position.z", [(0, 0), (0.8, 0.6), (2.0, -0.2), (4, 0)],
                     handles="auto", extrapolation="linear")  # fmt: skip
    expectations = {
        "format": "aether3d.expectations", "version": 1, "defaults": False,
        "checks": [
            {"check": "motion_quality", "objects": "Cube", "level": "info", "severity": "warning",
             "thresholds": {"acceleration_step": 0.5}},
            {"check": "motion_quality", "objects": "Loop", "loop": True, "severity": "warning"},
        ],
    }  # fmt: skip
    fs = FeedbackSession(scene, expectations, width=64, height=48)
    found = {(f.code, f.subjects[0]["name"]) for f in fs.diagnose().findings}
    assert ("ACCELERATION_STEP", "Cube") in found and ("LOOP_SEAM", "Loop") in found
    blocked = fs.propose().to_dict()["proposals"]
    assert all("allow_animation" in p["policy"]["violations"][0] for p in blocked)
    policy = {"allow_animation": True, "auto_apply": ["safe", "verify"]}
    proposals = fs.propose(policy=policy).to_dict()["proposals"]
    smoothing = next(p for p in proposals if p["actions"][0]["action"] == "curve_smoothing")
    cyclic = next(p for p in proposals if p["actions"][0]["action"] == "make_cyclic")
    assert smoothing["safety"] == cyclic["safety"] == "verify"
    result = fs.apply(ids=[smoothing["id"], cyclic["id"]], policy=policy).to_dict()
    assert result["status"] == "applied", result
    assert anim.channel(cube, "transform.position.y").smoothing == "continuous"
    assert anim.channel(loop, "transform.position.z").extrapolation == ("cycle", "cycle")
    after = {(f.code, f.subjects[0]["name"]) for f in fs.diagnose().findings}
    assert ("LOOP_SEAM", "Loop") not in after
    steps_after = [f for f in fs.diagnose().findings if f.code == "ACCELERATION_STEP"]
    assert len(steps_after) < sum(1 for c, n in found if c == "ACCELERATION_STEP")
    scene.history.undo()  # one step: both corrections
    assert anim.channel(cube, "transform.position.y").smoothing == "local"
