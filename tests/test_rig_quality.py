"""Rig quality, measured against references (the Stage 18 revision): bone heat weights
against Blender's Automatic Weights on the same mesh and bones, the anatomical figure's
deformation under the stress suite against bounds that v0.19's rigs fail, and landmarks
against the figure's known joints. The bounds come from measurements recorded in
research/2026-10-stage18-revision/DESIGN.md; each assertion says what v0.19 measured."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene

DATA = Path(__file__).parent / "data" / "blender_heat"


def test_heat_weights_agree_with_blenders_automatic_weights():
    """The same mesh and bone segments Blender 5.2 got (its Automatic Weights, exported by
    the revision's Blender experiment): the dominant joint agrees on 99.5 % of the
    vertices or more (measured 99.8 %; v0.19's geodesic weights: 82.6 %), the weights
    differ by 0.004 or less on average (measured 0.0013; v0.19: 0.194)."""
    from aether3d.animation.heat import heat_weights
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    case = json.loads((DATA / "case.json").read_text())
    mesh = implicit_surface(humanoid_shapes(), 0.012)
    p = np.ascontiguousarray(mesh.positions, np.float32)
    assert hashlib.sha256(p.tobytes()).hexdigest() == case["positions_sha256"], (
        "the humanoid recipe's mesh changed: regenerate the Blender reference (see case.json)"
    )
    z = np.load(DATA / "weights.npz")
    blender = np.zeros(tuple(z["shape"]))
    blender[z["rows"], z["cols"]] = z["values"]
    blender /= np.maximum(blender.sum(axis=1, keepdims=True), 1e-12)
    names = case["joints"]
    heads, tails = np.array(case["heads"]), np.array(case["tails"])
    # (the side rule on the sources, as Aether3D's generator applies it)
    side = np.array([1 if n.endswith(".L") else -1 if n.endswith(".R") else 0 for n in names])
    size = float(np.max(np.ptp(p, axis=0)))
    x = p[:, 0][:, None]
    allowed = ~(((side > 0) & (x < -0.02 * size)) | ((side < 0) & (x > 0.02 * size)))
    ours, _ = heat_weights(p.astype(np.float64), np.asarray(mesh.triangles), heads, tails,
                           allowed=allowed)  # fmt: skip
    agree = float((ours.argmax(axis=1) == blender.argmax(axis=1)).mean())
    mad = float(np.abs(ours - blender).sum(axis=1).mean() / 2)
    assert agree >= 0.995, agree
    assert mad <= 0.004, mad


@pytest.fixture(scope="module")
def figure():
    from aether3d.sculpt.figure import figure_mesh

    return figure_mesh(0.008)


def stress(mesh, **weights):
    s = Scene("Figure")
    body = s.add_object("Body", mesh=mesh)
    s.animation.autorig(body, name="Rig", weights=weights or None)
    rep = s.animation.diagnose_rig("Rig", poses="stress", animation=False, max_findings=500)
    return {pose: e["Body"] for pose, e in rep["poses"].items()}, rep


# The figure under the stress suite, auto-rigged with the defaults (heat weights, eight
# influences, linear blend skinning): per pose kind, the least share of its size the skin
# around the joint keeps, the most new creases and the worst edge stretch. v0.19 measured
# (its own rig of this mesh): elbows and knees 0.73 to 0.75 kept and 44 to 91 creases;
# shoulders 96 and 97 creases and a stretch up to 12.3; hips 133 to 273 creases; the head
# turn 0.91 kept and 79 creases.
BOUNDS = {
    "elbow_bend": (0.85, 20, 2.5),
    "knee_bend": (0.85, 20, 2.5),
    "hip_flex": (0.95, 90, 4.5),
    "shoulder_raise": (0.88, 40, 4.5),
    "wrist_twist": (0.97, 3, 2.0),
    "torso_twist": (0.95, 3, 2.5),
    "spine_bend": (0.98, 3, 2.0),
    "head_turn": (0.93, 5, 2.0),
}


def test_the_figure_deforms_within_measured_bounds(figure):
    poses, rep = stress(figure)
    assert len(poses) == 13
    for pose, rec in poses.items():
        kind = pose.rsplit("_", 1)[0]
        keep, creases, stretch = BOUNDS[kind]
        assert rec["joint_ring"] >= keep, (pose, rec)
        assert rec["new_creases"] <= creases, (pose, rec)
        assert rec["worst_stretch"] <= stretch, (pose, rec)
    folds = sum(r["folded_triangles"] for r in poses.values())
    assert folds <= 25, folds  # (v0.19: 209 over the suite)
    codes = {f["code"] for f in rep["findings"]}
    assert "JOINT_COLLAPSE" not in codes  # (v0.19: at its elbows and knees)


def test_eight_influences_keep_the_torso_smooth_and_smoothing_rescues_four(figure):
    """Heat weights keep up to eight influences. Capped at four (as most imported skins
    are), the twisted torso creases -- 65 new creases and 15 folds measured, 1 and 0 with
    eight -- and so does the bent spine (107; 0); corrective smoothing relaxes a
    four-influence skin (0 and 2) and more than halves its worst stretch at a raised
    shoulder (9.1 to 3.6)."""
    default, _ = stress(figure)
    four, _ = stress(figure, influences=4)
    smoothed, _ = stress(figure, influences=4, corrective=True)
    for pose in ("torso_twist_60", "spine_bend_45"):
        assert default[pose]["new_creases"] <= 3, pose
        assert four[pose]["new_creases"] >= 30, pose
        assert smoothed[pose]["new_creases"] <= 5, pose
    shoulder = "shoulder_raise_150.L"
    assert four[shoulder]["worst_stretch"] > 1.5 * smoothed[shoulder]["worst_stretch"]


FIGURES = {
    "a": {},
    "t": {"arm_angle": 88},
    "heavy": {"muscle": 1.4, "belly": 1.5, "limbs": 1.2, "chest": 1.2, "height": 1.68},
    "stylized": {"head": 1.7, "legs": 0.8, "limbs": 1.3, "arm_angle": 45},
    # (thighs that nearly touch: the voxels part the legs 9 cm under the crotch, so the
    # crotch is measured on the mesh -- the hip was 5 % of the height off)
    "slim": {"muscle": 0.7, "legs": 1.08, "hips": 0.9, "height": 1.85},
    # (a hand against the thigh is not found: the other's fingers go too -- one hand
    # had kept them unchecked)
    "arms_down": {"arm_angle": 8, "elbow_bend": 4},
}
MAJOR = ("hip.L", "knee.L", "ankle.L", "shoulder.L", "elbow.L", "wrist.L", "neck")
FINGER_KEYS = tuple(f"{f}.{k}" for f in ("thumb", "index", "middle", "ring", "pinky")
                    for k in ("1", "2", "3", "tip"))  # fmt: skip


@pytest.mark.parametrize("name", list(FIGURES))
def test_landmarks_find_the_figures_joints(name):
    """Major joints within 2.6 % of the height (measured at most 2.52 % over nine body
    types and three resolutions); the fingers, when the mesh shows them, within 2.5 %."""
    from aether3d.sculpt.figure import figure_joints, figure_mesh

    o = FIGURES[name]
    s = Scene(name)
    body = s.add_object("Body", mesh=figure_mesh(0.008, **o))
    found = s.animation.detect_landmarks(body)["landmarks"]
    truth = figure_joints(**o)
    height = o.get("height", 1.75)

    def err(k):
        return float(np.linalg.norm(np.array(found[k]["position"]) - truth[k])) / height

    for k in MAJOR:
        assert err(k) < 0.026, (k, err(k))
    present = [f"{k}.{s_}" for k in FINGER_KEYS for s_ in "LR" if f"{k}.{s_}" in found]
    assert len(present) in (0, 2 * len(FINGER_KEYS))  # (all the fingers or none)
    for k in present:
        assert err(k) < 0.025, (k, err(k))


def test_misnamed_fingers_are_left_out():
    """At 12 mm a stylized figure's fused fingers were found on both hands alike -- but
    named one off (the middle finger's tip 4 % of the height from the hand's tip, its
    joints 2.5 to 4.3 % off): a middle finger whose tip is not the hand's tip drops them."""
    from aether3d.sculpt.figure import figure_mesh

    s = Scene("Fused")
    o = {"head": 1.7, "legs": 0.8, "limbs": 1.3, "arm_angle": 45}
    body = s.add_object("Body", mesh=figure_mesh(0.012, **o))
    rep = s.animation.detect_landmarks(body)
    assert not any(k.startswith(("index.", "middle.", "thumb.")) for k in rep["landmarks"])
    warn = [d for d in rep["diagnostics"] if d["code"] == "FINGERS_UNCERTAIN"]
    assert warn and "middle.L" in warn[0]["fingers"]


def test_unresolved_fingers_are_left_out_and_said_so():
    """At 12 mm voxels the fingers (8 mm across) are not resolved: measured, the hands
    disagreed by 2 to 7 % of the height -- they get no finger landmarks, and a warning."""
    from aether3d.sculpt.figure import figure_mesh

    s = Scene("Coarse")
    body = s.add_object("Body", mesh=figure_mesh(0.012))
    rep = s.animation.detect_landmarks(body)
    assert not any(k.startswith(("index.", "thumb.")) for k in rep["landmarks"])
    warn = [d for d in rep["diagnostics"] if d["code"] == "FINGERS_UNCERTAIN"]
    assert warn and warn[0]["fingers"]
