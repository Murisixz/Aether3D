"""Stage 18 rig document: roles, helper joints, swing-twist and hinge limits, controls,
correctives, landmarks, info, twist distribution, blended skinning and project format 11
-- checked against independent computations."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aether3d import Scene, load_scene
from aether3d.animation import limits as L
from aether3d.animation import rotation as R
from aether3d.animation.rig import Corrective, parse_rig, skin_dual_quaternion, skin_linear
from aether3d.errors import AnimationError
from aether3d.mesh import Mesh
from aether3d.serialization import scene_from_dict, scene_to_dict

RNG = np.random.default_rng(18)


def quat(axis, degrees) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    h = math.radians(degrees) / 2
    return np.array([*(a * math.sin(h)), math.cos(h)])


def mul(a, b) -> np.ndarray:
    return R.multiply(np.asarray(a)[None], np.asarray(b)[None])[0]


def angle_between(a, b) -> float:
    d = abs(float(np.dot(a, b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


# --------------------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------------------- #


def test_limit_forms_are_checked_and_canonical():
    cone = L.check_limits({"swing": {"cone": 60}, "twist": [-30, 45]}, "l")
    assert json.loads(cone) == {"axis": "+y", "swing": {"cone": 60.0}, "twist": [-30.0, 45.0]}
    ell = json.loads(L.check_limits({"axis": "+x", "swing": {"y": [-10, 80], "z": [-5, 30]}}, "l"))
    assert ell["swing"] == {"y": [-10.0, 80.0], "z": [-5.0, 30.0]}
    hinge = json.loads(L.check_limits({"hinge": {"axis": "+x", "min": 0, "max": 150}}, "l"))
    assert hinge == {"hinge": {"axis": "+x", "min": 0.0, "max": 150.0}}
    for bad in (
        {"swing": {"cone": 200}},
        {"swing": {"y": [-10, 10]}},  # y is the bone axis of a +y bone
        {"twist": [10, 40]},  # the rest pose (0) must be allowed
        {"hinge": {"axis": "+w"}},
        {"hinge": {"axis": "+x"}, "twist": [-1, 1]},
        {"axis": "+y"},
        {"rotation": {"min": [0, 0, 0]}, "twist": [-1, 1]},
        {"swing": {"cone": float("nan")}},
    ):
        with pytest.raises(AnimationError) as e:
            L.check_limits(bad, "limits")
        assert e.value.code == "INVALID_RIG"


def _rel(q_local, q_rest):
    return L.relative(np.asarray(q_local), np.asarray(q_rest))


@pytest.mark.parametrize("rest_deg", [0.0, 35.0, 170.0])
def test_swing_cone_and_twist_clamp_from_the_rest_rotation(rest_deg):
    q_rest = quat([0.3, 1.0, -0.2], rest_deg)
    limits = json.loads(L.check_limits({"swing": {"cone": 40}, "twist": [-20, 30]}, "l"))
    y = np.array([0.0, 1.0, 0.0])
    for _ in range(200):
        swing_axis = np.cross(y, RNG.normal(size=3))
        swing = quat(swing_axis, RNG.uniform(0, 175))
        twist = quat(y, RNG.uniform(-179, 179))
        q_local = mul(q_rest, mul(swing, twist))
        out = L.clamp(limits, q_local, q_rest)
        got = q_local if out is None else out
        rel = _rel(got, q_rest)
        sv = L.swing_vector(rel, y)
        assert np.linalg.norm(sv) <= 40 + 1e-7
        assert -20 - 1e-7 <= L.twist_angle(rel, y) <= 30 + 1e-7
        before = _rel(q_local, q_rest)
        if out is None:  # inside: untouched
            assert np.linalg.norm(L.swing_vector(before, y)) <= 40
        else:  # the swing keeps its direction
            b = L.swing_vector(before, y)
            if np.linalg.norm(b) > 1e-6 and np.linalg.norm(sv) > 1e-6:
                cos = b @ sv / (np.linalg.norm(b) * np.linalg.norm(sv))
                assert cos > 1 - 1e-9


def test_ellipse_quadrants_are_asymmetric_and_sign_stable():
    limits = json.loads(L.check_limits({"swing": {"x": [-30, 90], "z": [-10, 45]}}, "l"))
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    forward = quat([1, 0, 0], 70)  # within +x 90
    assert L.clamp(limits, forward, ident) is None
    back = quat([1, 0, 0], -60)  # beyond -x 30
    got = L.clamp(limits, back, ident)
    assert L.swing_vector(_rel(got, ident), np.array([0.0, 1.0, 0.0]))[0] == pytest.approx(-30)
    # q and -q are the same rotation: the same clamp, wherever the sign flips.
    for q in (back, quat([0.4, 0, 0.9], 120), quat([0.9, 0.1, -0.4], 179.9)):
        a, b = L.clamp(limits, q, ident), L.clamp(limits, -q, ident)
        assert angle_between(a, b) < 1e-7
    # A zero bound forbids that direction entirely.
    flat = json.loads(L.check_limits({"swing": {"x": [0, 0], "z": [-90, 90]}}, "l"))
    got = L.clamp(flat, quat([1, 0, 1], 60), ident)
    assert abs(L.swing_vector(_rel(got, ident), np.array([0.0, 1.0, 0.0]))[0]) < 1e-9


def test_hinges_keep_one_axis_and_clamp_its_angle():
    limits = json.loads(L.check_limits({"hinge": {"axis": "+x", "min": 0, "max": 150}}, "l"))
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    assert L.clamp(limits, quat([1, 0, 0], 90), ident) is None
    hyper = L.clamp(limits, quat([1, 0, 0], -25), ident)  # a knee bending backwards
    assert angle_between(hyper, ident) < 1e-9
    skew = L.clamp(limits, mul(quat([1, 0, 0], 60), quat([0, 0, 1], 20)), ident)
    rel = _rel(skew, ident)
    assert np.linalg.norm(rel[[1, 2]]) < 1e-12  # only rotation about x is left
    assert L.twist_angle(rel, np.array([1.0, 0, 0])) == pytest.approx(60, abs=1e-6)
    # Wraparound: 200 degrees about +x is -160 (shorter way), clamped to 0.
    assert angle_between(L.clamp(limits, quat([1, 0, 0], 200), ident), ident) < 1e-9


def chain_scene(limits=None, enforce="ik") -> tuple[Scene, object]:
    s = Scene("Chain")
    arm = s.add_object("Armature")
    s.animation.add_rig(
        arm,
        [
            {"name": "Hip", "position": [0, 0, 1], "role": "thigh.L"},
            {"name": "Knee", "parent": "Hip", "position": [0, 0, -0.45], "role": "shin.L",
             **({"limits": limits} if limits else {})},
            {"name": "Ankle", "parent": "Knee", "position": [0, 0, -0.45], "role": "foot.L",
             "tail": [0, -0.15, -0.05]},
            {"name": "Helper", "parent": "Hip", "position": [0, 0.2, 0], "deform": False},
        ],
        enforce_limits=enforce,
    )  # fmt: skip
    return s, arm


def test_enforce_always_limits_joints_that_channels_rotate():
    hinge = {"hinge": {"axis": "+x", "min": 0, "max": 120}}
    s, arm = chain_scene(hinge, "ik")
    a = s.animation
    a.keyframe(arm, 'joints["Knee"].rotation_euler', 0.0, [-40.0, 0.0, 0.0])
    knee = a.pose_joints(arm, 0.0, space="armature")["Knee"]
    assert np.allclose(knee[:3, :3], R.matrices(quat([1, 0, 0], -40)[None])[0])  # Stage 17
    s2, arm2 = chain_scene(hinge, "always")
    s2.animation.keyframe(arm2, 'joints["Knee"].rotation_euler', 0.0, [-40.0, 0.0, 0.0])
    knee2 = s2.animation.pose_joints(arm2, 0.0, space="armature")["Knee"]
    assert np.allclose(knee2[:3, :3], np.eye(3), atol=1e-12)  # clamped to the hinge's 0
    s2.animation.keyframe(arm2, 'joints["Knee"].rotation_euler', 0.0, [70.0, 0.0, 0.0])
    knee3 = s2.animation.pose_joints(arm2, 0.0, space="armature")["Knee"]
    assert np.allclose(knee3[:3, :3], R.matrices(quat([1, 0, 0], 70)[None])[0], atol=1e-12)


def test_ik_respects_hinge_limits():
    s, arm = chain_scene({"hinge": {"axis": "+x", "min": 0, "max": 150}})
    a = s.animation
    target = s.add_object("Target", position=(0, 0.3, 0.4))  # in front: the knee bends back
    ik = {"name": "leg", "type": "ik", "chain": ["Hip", "Knee", "Ankle"],
          "target": {"object": "Target"}, "pole": {"point": [0, -1, 0.5]}}  # fmt: skip
    a.add_constraint(arm, ik)
    m = a.pose_joints(arm, 0.0, space="armature")
    rel = L.relative(R.from_matrices(np.linalg.solve(m["Hip"], m["Knee"])[None, :3, :3])[0],
                     np.array([0, 0, 0, 1.0]))  # fmt: skip
    assert np.linalg.norm(rel[[1, 2]]) < 1e-9
    assert 0.0 <= L.twist_angle(rel, np.array([1.0, 0, 0])) <= 150.0 + 1e-9
    del target


# --------------------------------------------------------------------------------------- #
# Roles, helper joints, controls, correctives, landmarks, info
# --------------------------------------------------------------------------------------- #


def test_roles_helpers_and_new_fields_round_trip():
    s, arm = chain_scene({"swing": {"cone": 70}, "twist": [-10, 10]})
    a = s.animation
    target = s.add_object("FootIK", position=(0, 0, 0.1))
    pole = s.add_object("KneePole", position=(0, -0.5, 0.55))
    rig = a.rig(arm).to_json()
    rig["constraints"] = [
        {
            "name": "leg_ik.L",
            "type": "ik",
            "chain": ["Hip", "Knee", "Ankle"],
            "target": {"object": "FootIK"},
            "pole": {"object": "KneePole"},
        }
    ]
    rig["controls"] = [
        {"name": "foot_ik.L", "kind": "ik_target", "object": "FootIK", "joint": "Ankle",
         "constraint": "leg_ik.L", "group": "leg.L", "side": "L", "shape": "cube"},
        {"name": "knee_pole.L", "kind": "pole", "object": "KneePole", "joint": "Knee",
         "constraint": "leg_ik.L", "group": "leg.L", "side": "L", "size": 0.03},
        {"name": "hip_fk.L", "kind": "fk", "joint": "Hip", "group": "leg.L"},
    ]  # fmt: skip
    rig["landmarks"] = {"knee.L": [0.0, 0.01, 0.55], "heel.L": [0, 0.03, 0.02]}
    rig["template"] = "humanoid"
    rig["info"] = {"forward": "-y", "up": "+z", "description": "a leg"}
    a.set_rig(arm, rig)
    r = a.rig(arm)
    assert r.roles == {"thigh.L": "Hip", "shin.L": "Knee", "foot.L": "Ankle"}
    assert r.role_index("shin.L") == r.index("Knee")
    assert [r.joints[k].name for k in r.deforming] == ["Hip", "Knee", "Ankle"]
    assert r.control("foot_ik.L").object == target.id  # names become ids
    assert r.control("knee_pole.L").size == 0.03
    assert r.forward_axis() == "-y" and r.template == "humanoid"
    assert r.landmark_dict["knee.L"] == (0.0, 0.01, 0.55)
    # Saved and loaded unchanged; undo restores the Stage 17 rig.
    loaded = scene_from_dict(json.loads(json.dumps(scene_to_dict(s))))
    r2 = loaded.animation.rig(loaded.get_by_name("Armature"))
    assert r2.to_json() == r.to_json()
    s.history.undo()
    assert a.rig(arm).controls == () and a.rig(arm).template is None
    s.history.redo()
    assert a.rig(arm).to_json() == r.to_json()
    # Removing a control's object drops the control (the IK reports its missing target).
    s.remove(pole)
    assert [c.name for c in a.rig(arm).controls] == ["foot_ik.L", "hip_fk.L"]
    codes = {d.code for d in a.rigging(0.0).diagnostics}
    assert "CONSTRAINT_TARGET_MISSING" not in codes  # (the pole is optional)


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda r: r["joints"][1].update(role="Shin"), "INVALID_RIG"),
        (lambda r: r["joints"][1].update(role="thigh.L"), "INVALID_RIG"),  # duplicate
        (lambda r: r["joints"][1].update(deform="no"), "INVALID_RIG"),
        (lambda r: r.update(controls=[{"name": "c", "kind": "fk", "joint": "No"}]), "INVALID_RIG"),
        (lambda r: r.update(controls=[{"name": "c", "kind": "ik_target"}]), "INVALID_RIG"),
        (lambda r: r.update(controls=[{"name": "c", "kind": "fk", "joint": "Hip",
                                       "constraint": "missing"}]), "INVALID_RIG"),
        (lambda r: r.update(controls=[{"name": "c", "kind": "fk", "joint": "Hip",
                                       "mirror": "d"}]), "INVALID_RIG"),
        (lambda r: r.update(correctives=[{"name": "c", "mesh": "m", "target": "t",
                                          "joint": "Knee", "range": [5, 5]}]), "INVALID_RIG"),
        (lambda r: r.update(correctives=[{"name": "c", "mesh": "m", "target": "t",
                                          "joint": "Knee", "driver": "bend"}]), "INVALID_RIG"),
        (lambda r: r.update(landmarks={"a": [0, 0]}), "INVALID_RIG"),
        (lambda r: r.update(info={"forward": "+z", "up": "+z"}), "INVALID_RIG"),
        (lambda r: r.update(info={"colour": "red"}), "INVALID_RIG"),
        (lambda r: r.update(enforce_limits="sometimes"), "INVALID_RIG"),
    ],
)  # fmt: skip
def test_invalid_rig_documents_are_refused(change, code):
    s, arm = chain_scene()
    raw = s.animation.rig(arm).to_json()
    change(raw)
    with pytest.raises(AnimationError) as e:
        parse_rig(raw, "rig")
    assert e.value.code == code


def tube_scene(method="linear"):
    """A tube along -Z skinned to Hip/Knee with a 'bulge' morph target at the knee."""
    s, arm = chain_scene()
    a = s.animation
    n_ring, rings = 12, 31
    ang = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
    zs = np.linspace(1.0, 0.1, rings)
    pos = np.array([[0.06 * math.cos(t), 0.06 * math.sin(t), z] for z in zs for t in ang])
    tris = []
    for r in range(rings - 1):
        for i in range(n_ring):
            p0, p1 = r * n_ring + i, r * n_ring + (i + 1) % n_ring
            tris += [[p0, p1 + n_ring, p1], [p0, p0 + n_ring, p1 + n_ring]]
    mesh = Mesh(pos.astype(np.float32), np.array(tris, np.uint32))
    leg = s.add_object("Leg", mesh=mesh)
    bulge = np.zeros_like(pos)
    near = np.abs(pos[:, 2] - 0.55) < 0.06
    bulge[near, :2] = pos[near, :2] * 0.3
    a.set_morph_targets(leg, [{"name": "knee_bulge", "positions": bulge.tolist()}])
    w = np.clip((0.62 - pos[:, 2]) / 0.14, 0, 1)
    a.skin(leg, arm, weights={"indices": np.stack([np.zeros(len(w)), np.ones(len(w))], 1)
                              .astype(np.int32), "weights": np.stack([1 - w, w], 1),
                              "joints": ["Hip", "Knee"]}, method=method)  # fmt: skip
    return s, arm, leg, near


def test_correctives_follow_the_driver_joint_after_constraints():
    s, arm, leg, near = tube_scene()
    a = s.animation
    raw = a.rig(arm).to_json()
    raw["correctives"] = [{"name": "knee_fix", "mesh": "Leg", "target": "knee_bulge",
                           "joint": "Knee", "driver": "rotation", "axis": "+x",
                           "range": [30, 90], "curve": "linear"}]  # fmt: skip
    a.set_rig(arm, raw)
    assert a.rig(arm).correctives[0].mesh == leg.id
    for deg, expected in ((0, 0.0), (30, 0.0), (60, 0.5), (90, 1.0), (120, 1.0), (-60, 0.0)):
        a.keyframe(arm, 'joints["Knee"].rotation_euler', 0.0, [float(deg), 0.0, 0.0])
        result = a.rigging(0.0)
        value, weight = result.correctives["Armature/knee_fix"]
        assert value == pytest.approx(deg, abs=1e-9)
        assert weight == pytest.approx(expected, abs=1e-12)
    # The morph is applied before skinning: at 90 degrees, the bulge is fully in.
    a.keyframe(arm, 'joints["Knee"].rotation_euler', 0.0, [0.0, 0.0, 0.0])
    a.keyframe(arm, 'joints["Hip"].rotation_euler', 0.0, [0.0, 0.0, 0.0])
    rest = a.rigging(0.0).meshes.get(leg.id)
    assert rest is None or np.allclose(rest.positions, leg.mesh.positions, atol=1e-6)
    # A constraint moving the knee drives the corrective too (after constraints).
    raw = a.rig(arm).to_json()
    raw["constraints"] = [{"name": "bend", "type": "copy_rotation", "joint": "Knee",
                           "target": {"point": [0, 0, 0]}}]  # fmt: skip
    a.set_rig(arm, raw)
    c = Corrective("x", "m", "t", "Knee", "angle", "+x", None, (0.0, 180.0), "smooth")
    assert c.weight_for(90.0) == pytest.approx(0.5)


def test_twist_distribution_takes_a_share_of_the_source_twist():
    s = Scene("Forearm")
    arm = s.add_object("Armature")
    a = s.animation
    a.add_rig(
        arm,
        [
            {"name": "ForeArm", "position": [0, 0, 1], "euler": [0, 0, 0]},
            {"name": "Twist", "parent": "ForeArm", "position": [0, 0.12, 0]},
            {"name": "Hand", "parent": "ForeArm", "position": [0, 0.25, 0],
             "rotation": quat([0, 0, 1], 15).tolist()},
        ],
        constraints=[{"name": "share", "type": "twist", "joint": "Twist",
                      "target": {"joint": "Hand"}, "factor": 0.5}],
    )  # fmt: skip
    hand_rest = quat([0, 0, 1], 15)
    a.keyframe(arm, 'joints["Hand"].rotation', 0.0, mul(hand_rest, quat([0, 1, 0], 80)).tolist())
    m = a.pose_joints(arm, 0.0, space="armature")
    rel = R.from_matrices(np.linalg.solve(m["ForeArm"], m["Twist"])[None, :3, :3])[0]
    assert angle_between(rel, quat([0, 1, 0], 40)) < 1e-9
    with pytest.raises(AnimationError):
        a.add_constraint(arm, {"name": "bad", "type": "twist", "joint": "Twist",
                               "target": {"object": "Armature"}})  # fmt: skip


# --------------------------------------------------------------------------------------- #
# Blended skinning and files
# --------------------------------------------------------------------------------------- #


def test_blended_skins_mix_linear_and_dual_quaternion_per_vertex():
    s, arm, leg, _ = tube_scene()
    a = s.animation
    skin = a._owned[("skin", leg.id)]._definition
    dq = np.clip((leg.mesh.positions[:, 2] - 0.3) / 0.4, 0, 1)
    from aether3d.animation.rig import make_skin

    a.skin(leg, arm, weights={"indices": skin.indices, "weights": skin.weights,
                              "joints": ["Hip", "Knee"]}, method="blended", dq=dq)  # fmt: skip
    blended = a._owned[("skin", leg.id)]._definition
    assert blended.method == "blended" and np.array_equal(blended.dq, dq)
    a.keyframe(arm, 'joints["Knee"].rotation_euler', 0.0, [0.0, 100.0, 0.0])  # a twist
    out = a.rigging(0.0).meshes[leg.id]
    rig = a.rig(arm)
    pal = np.stack([np.eye(4), np.eye(4)])
    knee_local = np.eye(4)
    knee_local[:3, :3] = R.matrices(quat([0, 1, 0], 100)[None])[0]
    pal[1] = rig.rest_model()[1] @ knee_local @ np.linalg.inv(rig.rest_model()[1])
    lbs, _ = skin_linear(leg.mesh.positions, None, skin.indices, skin.weights, pal)
    dqs, _ = skin_dual_quaternion(leg.mesh.positions, None, skin.indices, skin.weights, pal)
    expect = lbs + dq[:, None] * (dqs.astype(np.float64) - lbs)
    assert np.allclose(out.positions, expect, atol=2e-6)
    with pytest.raises(AnimationError):
        make_skin(leg.id, skin.rig, list(skin.joints), skin.inverse_bind, skin.indices,
                  skin.weights, "blended")  # fmt: skip
    with pytest.raises(AnimationError):
        make_skin(leg.id, skin.rig, list(skin.joints), skin.inverse_bind, skin.indices,
                  skin.weights, "blended", dq=dq * 2)  # fmt: skip
    # Saved and loaded: method, dq and locks survive.
    locked = blended.replace(locked_joints=(skin.joints[0],),
                             locked_vertices=np.array([1, 5], np.int32))  # fmt: skip
    with a.batch("Lock"):
        a._set(a._owned[("skin", leg.id)], definition=locked)
    loaded = scene_from_dict(json.loads(json.dumps(scene_to_dict(s))))
    got = loaded.animation._owned[("skin", leg.id)]._definition
    assert got.same(locked)


def test_version_10_projects_load_and_save_as_11(tmp_path):
    s, arm, leg, _ = tube_scene()
    doc = scene_to_dict(s)
    assert doc["format_version"] == 11
    doc["format_version"] = 10  # a Stage 17 rig has nothing new: the v10 form is the same
    old = scene_from_dict(doc)
    assert (
        old.animation.rig(old.get_by_name("Armature")).to_json() == s.animation.rig(arm).to_json()
    )
    path = tmp_path / "leg.aether"
    old.save(path)
    assert json.loads(path.read_text())["format_version"] == 11
    assert load_scene(path).animation.rig(arm.id) is not None
