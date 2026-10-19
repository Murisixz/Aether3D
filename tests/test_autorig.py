"""Automatic rigging (Stage 18): implicit test bodies, landmark detection against the
modeled joint positions (A-pose, T-pose, arms down, stylized, a child, a wide stance,
rotated, scaled, in parts), the template rigs it builds, their controls, weights and
motion -- and the honest fallbacks when the geometry cannot decide."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d import Scene, ValidationError, create_cube
from aether3d.animation import rotation as R
from aether3d.errors import AnimationError
from aether3d.sculpt.implicit import (
    HUMANOID_DEFAULTS,
    humanoid_shapes,
    implicit_surface,
    quadruped_shapes,
)

VOXEL = 0.014


def truth(**o):
    """Joint positions of :func:`humanoid_shapes` (the recipe's own construction)."""
    d = dict(HUMANOID_DEFAULTS)
    d.update(o)
    H = d["height"]
    ankle_z = 0.039 * H
    hip_z = ankle_z + (0.51 * H - ankle_z) * d["legs"]
    knee_z = ankle_z + 0.49 * (hip_z - ankle_z)
    head_rz = 0.066 * H * d["head"]
    chin = H - head_rz - 0.85 * head_rz
    neck_base = chin - 0.045 * H
    shoulder_z = neck_base - 0.012 * H
    hw, sw = 0.052 * H * d["hips"], 0.105 * H * d["shoulders"]
    ang, bend = math.radians(d["arm_angle"]), math.radians(d["elbow_bend"])
    sh = np.array([sw, 0.0, shoulder_z - 0.01 * H])
    down = np.array([math.sin(ang), 0.0, -math.cos(ang)])
    elbow = sh + 0.186 * H * d["arms"] * down
    d2 = down * math.cos(bend) + np.array([0.0, -1.0, 0.0]) * math.sin(bend)
    wrist = elbow + 0.146 * H * d["arms"] * d2
    xf = hw + d["stance"] / 2
    kb = math.radians(d["knee_bend"])
    return {
        "hip.L": np.array([hw, 0, hip_z]),
        "knee.L": np.array([0.5 * (hw + xf), -math.sin(kb) * 0.1 * H, knee_z]),
        "ankle.L": np.array([xf, 0, ankle_z]),
        "shoulder.L": sh,
        "elbow.L": elbow,
        "wrist.L": wrist,
        "neck": np.array([0, 0, neck_base]),
    }


VARIANTS = {
    "a_pose": {},
    "t_pose": {"arm_angle": 88},
    "arms_down": {"arm_angle": 6},
    "stylized": {"head": 1.7, "legs": 0.75, "limbs": 1.3, "arm_angle": 40},
    "child": {"height": 1.2, "head": 1.3},
    "wide_bent": {"stance": 0.3, "arm_angle": 60, "knee_bend": 15, "elbow_bend": 25},
}


@pytest.fixture(scope="module")
def bodies():
    return {k: implicit_surface(humanoid_shapes(**v), VOXEL) for k, v in VARIANTS.items()}


def test_implicit_surfaces_are_closed_and_validated():
    mesh = implicit_surface([{"type": "sphere", "center": [0, 0, 0], "radius": 0.5}], 0.02)
    p = mesh.positions.astype(float)
    assert np.allclose(np.linalg.norm(p, axis=1), 0.5, atol=0.003)
    # Closed and outward: every edge is shared by two triangles; the volume is positive.
    t = mesh.triangles.astype(np.int64)
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    volume = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6
    assert volume == pytest.approx(4 / 3 * math.pi * 0.125, rel=0.02)
    assert np.all(np.sum(mesh.normals * p, axis=1) > 0)  # gradient normals point out
    capsule = implicit_surface([{"type": "capsule", "a": [0, 0, 0], "b": [0, 0, 1],
                                 "radius": 0.2, "radius_b": 0.1}], 0.02)  # fmt: skip
    z = capsule.positions[:, 2]
    assert z.min() == pytest.approx(-0.2, abs=0.01) and z.max() == pytest.approx(1.1, abs=0.01)
    hole = {"type": "sphere", "center": [0, 0, 0], "radius": 1, "subtract": True}
    for bad in ([], [{"type": "torus"}], [{"type": "sphere", "center": [0, 0], "radius": 1}],
                [hole]):  # fmt: skip
        with pytest.raises(ValidationError):
            implicit_surface(bad, 0.05)
    with pytest.raises(ValidationError):
        implicit_surface([{"type": "sphere", "center": [0, 0, 0], "radius": 1}], 1e-4)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_landmarks_match_the_modeled_joints(bodies, name):
    s = Scene(name)
    body = s.add_object("Body", mesh=bodies[name])
    rep = s.animation.detect_landmarks(body)
    H = VARIANTS[name].get("height", 1.75)
    errors = {
        k: float(np.linalg.norm(np.array(rep["landmarks"][k]["position"]) - v))
        for k, v in truth(**VARIANTS[name]).items()
    }
    assert max(errors.values()) < 0.04 * H, errors  # within 4 % of the height
    assert rep["frame"]["forward"] == pytest.approx([0, -1, 0], abs=1e-6)
    assert rep["frame"]["up"] == pytest.approx([0, 0, 1], abs=1e-6)
    pose = {"t_pose": "T-pose", "arms_down": "arms down"}.get(name, "A-pose")
    assert rep["pose"] == pose
    for lm in rep["landmarks"].values():
        assert 0 <= lm["confidence"] <= 1 and lm["method"] and lm["note"]


def test_landmarks_follow_the_character_in_any_placement(bodies):
    """Rotated about the vertical, moved and scaled: the same landmarks, transformed."""
    s = Scene("Placed")
    q = R.from_euler(np.array([0.0, 0.0, 137.0]))[0]
    body = s.add_object("Body", mesh=bodies["a_pose"], position=(3.0, -2.0, 0.5),
                        rotation=tuple(q.tolist()), scale=1.6)  # fmt: skip
    rep = s.animation.detect_landmarks(body)
    m = body.world_matrix
    fwd = m[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert rep["frame"]["forward"] == pytest.approx(fwd / np.linalg.norm(fwd), abs=0.02)
    for k, v in truth().items():
        expect = (m @ np.append(v, 1.0))[:3]
        got = np.array(rep["landmarks"][k]["position"])
        assert np.linalg.norm(got - expect) < 0.04 * 1.75 * 1.6, k


def test_lying_meshes_are_measured_with_the_given_up_axis(bodies):
    s = Scene("Lying")
    q = R.from_euler(np.array([90.0, 0.0, 0.0]))[0]  # Z-up body turned onto its back
    body = s.add_object("Body", mesh=bodies["t_pose"], rotation=tuple(q.tolist()))
    rep = s.animation.detect_landmarks(body, up="-y")
    m = body.world_matrix
    for k in ("hip.L", "shoulder.L", "wrist.L"):
        expect = (m @ np.append(truth(arm_angle=88)[k], 1.0))[:3]
        assert np.linalg.norm(np.array(rep["landmarks"][k]["position"]) - expect) < 0.07, k


def test_overrides_and_diagnostics(bodies):
    s = Scene("Over")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    rep = s.animation.detect_landmarks(body, landmarks={"knee.L": [0.1, -0.02, 0.5]})
    assert rep["landmarks"]["knee.L"] == {"position": [0.1, -0.02, 0.5], "confidence": 1.0,
                                          "method": "override", "note": "given"}  # fmt: skip
    # A sword in the hand: an extra protrusion is reported, the hands still found.
    sword = implicit_surface([{"type": "capsule", "a": [0.5, -0.08, 0.86], "b": [1.2, -0.3, 0.86],
                               "radius": 0.02}], VOXEL)  # fmt: skip
    s.add_object("Sword", mesh=sword)
    rep = s.animation.detect_landmarks([body, s.get_by_name("Sword")])
    assert any(d["code"] == "EXTRA_EXTREMITY" for d in rep["diagnostics"])
    with pytest.raises(AnimationError):
        s.animation.detect_landmarks(body, template="octopus")
    with pytest.raises(AnimationError):
        s.animation.detect_landmarks(body, up="sideways")


def test_autorig_builds_a_controlled_weighted_humanoid(bodies):
    s = Scene("Rigged")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    n = len(s.history)
    rep = s.animation.autorig(body)
    assert len(s.history) == n + 1  # one undo step
    arm = s.get_by_name(rep["armature"])
    a = s.animation
    rig = a.rig(arm)
    expect = {"Hips", "Spine", "Chest", "Neck", "Head", "Shoulder.L", "UpperArm.L", "ForeArm.L",
              "ForeArmTwist.L", "Hand.L", "Thigh.R", "Shin.R", "Foot.R", "Toe.R"}  # fmt: skip
    assert expect <= set(rig.names)
    assert rig.roles["lower_arm.L"] == "ForeArm.L" and rig.template == "humanoid"
    assert rig.forward_axis() == "-y"
    # Bone frames: +Y along each bone; +X the bend axis (elbows and knees bend positive).
    model = rig.rest_model()
    for name, child in (("Thigh.L", "Shin.L"), ("UpperArm.R", "ForeArm.R"), ("Spine", "Chest")):
        k = rig.index(name)
        d = model[rig.index(child)][:3, 3] - model[k][:3, 3]
        assert np.dot(model[k][:3, 1], d / np.linalg.norm(d)) > 0.999
    # Controls: objects under the armature, IK starting in FK (influence 0).
    for c in ("hand_ik.L", "foot_ik.R", "knee_pole.L", "elbow_pole.R", "gaze"):
        ctl = rig.control(c)
        assert ctl is not None and s.get(ctl.object).parent is arm
    ik = {c.name: c for c in rig.constraints}
    assert ik["leg_ik.L"].influence == 0.0 and ik["arm_ik.R"].p["solver"] == "two_bone"
    assert ik["forearm_twist.L"].type == "twist"
    # The skin uses deforming joints only, all vertices weighted.
    w = a.inspect_weights(body)
    assert w["valid"] and w["max_influences"] <= 8  # (heat weights keep up to eight)
    assert not any(f["code"] in ("CROSS_SIDE_WEIGHTS", "HELPER_JOINT_WEIGHTED")
                   for f in w["findings"])  # fmt: skip
    # Nothing moved at rest; undo removes everything.
    rest = a.rigging(0.0).meshes.get(body.id)
    assert rest is None or np.allclose(rest.positions, body.mesh.positions, atol=1e-5)
    s.history.undo()
    assert len(s) == 1  # the armature and the control objects are gone with the rig
    assert ("skin", body.id) not in a._owned


def test_generated_motion_and_ik_drive_the_autorig(bodies):
    s = Scene("Moving")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    rep = s.animation.autorig(body)
    arm = s.get_by_name(rep["armature"])
    a = s.animation
    g = a.generate(arm, "walk", forward="-y")
    c = a.foot_contacts(arm, ["Foot.L", "Foot.R"], start=0, end=2 * g["period"], height=0.03)
    for foot in c["feet"].values():
        assert foot["sliding_max"] < 0.01 and foot["penetration"] < 0.005 and foot["contacts"]
    # Elbows and knees bend the anatomical way when IK reaches.
    s2 = Scene("Reach")
    b2 = s2.add_object("Body", mesh=bodies["a_pose"])
    arm2 = s2.get_by_name(s2.animation.autorig(b2)["armature"])
    a2 = s2.animation
    tgt = s2.get_by_name(f"{arm2.name} hand_ik.L")
    tgt.position = (0.3, -0.4, 1.2)
    a2.keyframe(arm2, 'constraints["arm_ik.L"].influence', 0.0, 1.0)
    rr = a2.rigging(0.0)
    assert rr.residuals[f"{arm2.name}/arm_ik.L"] < 1e-6
    j = a2.pose_joints(arm2, 0.0, space="world")
    shoulder, elbow, hand = (j[n][:3, 3] for n in ("UpperArm.L", "ForeArm.L", "Hand.L"))
    mid = 0.5 * (shoulder + hand)
    assert (elbow - mid)[1] > 0.0  # the elbow points back, never forward


def test_stylized_child_and_bent_bodies_rig(bodies):
    for name in ("stylized", "child", "wide_bent"):
        s = Scene(name)
        body = s.add_object("Body", mesh=bodies[name])
        rep = s.animation.autorig(body, template="stylized" if name == "stylized" else "humanoid")
        assert rep["weights"]["Body"]["unreached_vertices"] == 0
        assert rep["confidence"] >= 0.5


def test_blocky_characters_rig_part_by_part():
    s = Scene("Blocky")
    px = 1 / 16
    parts = []
    for c, size in (((0, 0, 18 * px), (8 * px, 4 * px, 12 * px)),
                    ((0, 0, 28 * px), (8 * px, 8 * px, 8 * px)),
                    ((6 * px, 0, 18 * px), (4 * px, 4 * px, 12 * px)),
                    ((-6 * px, 0, 18 * px), (4 * px, 4 * px, 12 * px)),
                    ((2 * px, 0, 6 * px), (4 * px, 4 * px, 12 * px)),
                    ((-2 * px, 0, 6 * px), (4 * px, 4 * px, 12 * px))):  # fmt: skip
        b = create_cube(1.0)
        parts.append((b.positions.astype(float) * size + c, b.triangles.astype(np.int64)))
    from aether3d.mesh import Mesh

    pos = np.concatenate([p for p, _ in parts])
    tris = np.concatenate([t + sum(len(q) for q, _ in parts[:i]) for i, (_, t) in enumerate(parts)])
    body = s.add_object("Steve", mesh=Mesh(pos.astype(np.float32), tris.astype(np.uint32)))
    rep = s.animation.autorig(body, template="blocky", forward="-y")
    arm = s.get_by_name(rep["armature"])
    rig = s.animation.rig(arm)
    assert set(rig.names) == {"Hips", "Body", "Head", "Arm.L", "Arm.R", "Leg.L", "Leg.R"}
    sk = s.animation._owned[("skin", body.id)]._definition
    assert np.all(np.sort(sk.dense(), axis=1)[:, -1] == 1.0)  # every block rigid
    head = rig.rest_model()[rig.index("Head")][:3, 3]
    assert head[2] == pytest.approx(24 * px, abs=1.5 * px)  # the head turns on the body's top
    g = s.animation.generate(arm, "walk", forward="-y")
    assert g["keys"] > 0
    # The same blocks as separate objects: attached by default, and measured (they are
    # the body), so the skeleton is the single mesh's.
    s2 = Scene("Blocks")
    blocks = [s2.add_object(f"Block{i}", mesh=Mesh(p.astype(np.float32), t.astype(np.uint32)))
              for i, (p, t) in enumerate(parts)]  # fmt: skip
    rep2 = s2.animation.autorig(blocks, template="blocky", forward="-y")
    assert rep2["measured"] == [b.name for b in blocks]
    assert set(rep2["parts"].values()) == {"attach"}
    assert np.allclose([j["position"] for j in rep2["joints"]],
                       [j["position"] for j in rep["joints"]], atol=1e-9)  # fmt: skip


def test_quadrupeds_rig():
    s = Scene("Dog")
    body = s.add_object("Dog", mesh=implicit_surface(quadruped_shapes(), 0.02))
    rep = s.animation.autorig(body, template="quadruped", forward="-y")
    rig = s.animation.rig(s.get_by_name(rep["armature"]))
    assert {"Pelvis", "Spine", "Chest", "Neck", "Head", "Tail", "FrontLeg.L", "BackFoot.R"} <= set(
        rig.names
    )
    assert rep["weights"]["Dog"]["unreached_vertices"] == 0


def test_dry_runs_change_nothing_and_parts_bind_their_way(bodies):
    s = Scene("Parts")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    helmet = s.add_object("Helmet", mesh=implicit_surface(
        [{"type": "ellipsoid", "center": [0, 0, 1.64], "radii": [0.11, 0.12, 0.13]},
         {"type": "ellipsoid", "center": [0, 0, 1.6], "radii": [0.1, 0.11, 0.1],
          "subtract": True}], 0.012))  # fmt: skip
    n = len(s.history)
    plan = s.animation.autorig([body, helmet], dry_run=True)
    assert plan["dry_run"] and len(s.history) == n and plan["joints"]
    rep = s.animation.autorig([body, helmet], parts={"Helmet": "rigid"})
    assert rep["weights"]["Helmet"]["mode"] == "rigid"
    sk = s.animation._owned[("skin", helmet.id)]._definition
    rig = s.animation._items[sk.rig]._definition
    dominant = {rig.joints[rig.index_of_id(sk.joints[c])].name
                for c in np.argmax(sk.dense(), axis=1)}  # fmt: skip
    assert dominant == {"Head"}
    with pytest.raises(AnimationError):
        s.animation.autorig(body, parts={"Nope": "rigid"})


def test_clothing_and_armour_do_not_move_the_skeleton(bodies):
    """The landmarks are measured on the body: the pieces ``parts`` transfers, attaches
    or binds rigidly follow it and leave its joints exactly where the bare body puts
    them (measured with it, a helmet over the head raises the skull's pivot)."""
    s = Scene("Dressed")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    helmet = s.add_object("Helmet", mesh=implicit_surface(
        [{"type": "ellipsoid", "center": [0, 0, 1.64], "radii": [0.11, 0.12, 0.13]},
         {"type": "ellipsoid", "center": [0, 0, 1.6], "radii": [0.1, 0.11, 0.1],
          "subtract": True}], 0.012))  # fmt: skip
    pack = s.add_object("Pack", mesh=implicit_surface(
        [{"type": "box", "center": [0, 0.17, 1.25], "size": [0.3, 0.12, 0.36]}], 0.02))  # fmt: skip
    bare = s.animation.autorig(body, dry_run=True)
    dressed = s.animation.autorig(
        [body, helmet, pack], parts={"Helmet": "attach", "Pack": "rigid"}, dry_run=True
    )
    assert bare["measured"] == dressed["measured"] == ["Body"]
    assert [j["position"] for j in dressed["joints"]] == [j["position"] for j in bare["joints"]]
    assert [d["code"] for d in dressed["diagnostics"]] == [d["code"] for d in bare["diagnostics"]]
    # (generated with the body, the pieces are measured with it and move its joints)
    mixed = s.animation.autorig([body, helmet, pack], dry_run=True)
    assert mixed["measured"] == ["Body", "Helmet", "Pack"]

    def head(r):
        return next(j["position"] for j in r["joints"] if j["name"] == "Head")

    assert abs(head(mixed)[2] - head(bare)[2]) > 0.01
    # Every part skipped but one: that one is measured, whatever its mode.
    only = s.animation.autorig(
        [body, helmet], parts={"Body": "skip", "Helmet": "attach"}, dry_run=True
    )
    assert only["measured"] == ["Helmet"]


def test_mirrored_pieces_attach_to_mirrored_joints(bodies):
    """A plate over a shoulder is as near the clavicle (which ends there) as the upper arm
    (which starts there), up to rounding: such ties (within a micrometre) go to the bone
    nearest the root, on both sides alike -- the strictly nearest bone had sent a
    showcase's two plates to different bones."""
    s = Scene("Plates")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    plan = s.animation.autorig(body, dry_run=True)
    at = {j["name"]: np.array(j["position"]) for j in plan["joints"]}
    plates = []
    for side in ("L", "R"):
        j, fore = at[f"UpperArm.{side}"], at[f"ForeArm.{side}"]
        u = (fore - j) / np.linalg.norm(fore - j)
        v = (j - at[f"Shoulder.{side}"]) / np.linalg.norm(j - at[f"Shoulder.{side}"])
        n = v - (v @ u) * u
        n /= np.linalg.norm(n)
        # (beyond the clavicle's end, over the upper arm's start: 5e-7 m nearer the arm)
        c = j + 0.035 * n + 1.9e-4 * u
        plates.append(s.add_object(f"Plate.{side}", mesh=create_cube(1.0), position=tuple(c),
                                   scale=(0.1, 0.1, 0.04)))  # fmt: skip
    rep = s.animation.autorig([body, *plates], parts={p.name: "attach" for p in plates})
    assert rep["weights"]["Plate.L"]["joint"] == "Shoulder.L"
    assert rep["weights"]["Plate.R"]["joint"] == "Shoulder.R"


def test_attached_parts_move_with_the_placed_character(bodies):
    """Attached hard pieces are parented to the armature (where they are), so placing or
    turning the character afterwards keeps them on their joints; undo restores the
    hierarchy."""
    from aether3d.rendering.overlay import animation_overlay
    from aether3d.rendering.scene_data import world_matrices

    s = Scene("Placed")
    body = s.add_object("Body", mesh=bodies["a_pose"])
    helmet = s.add_object(
        "Helmet",
        mesh=implicit_surface(
            [{"type": "ellipsoid", "center": [0, 0.01, 1.66], "radii": [0.12, 0.13, 0.12]}], 0.02
        ),
    )
    a = s.animation
    rep = a.autorig([body, helmet], name="Rig", parts={"Helmet": "attach"})
    assert rep["weights"]["Helmet"] == {"mode": "attach", "method": "attachment",
                                        "joint": "Head", "parented": True}  # fmt: skip
    arm = s.get_by_name("Rig")
    assert helmet.parent is arm
    rest = helmet.world_matrix
    head0 = a.pose_joints(arm, 0.0, space="world")["Head"]
    arm.rotation_euler = (0.0, 0.0, 90.0)
    arm.position = (-2.6, 1.0, 0.0)
    a.generate(arm, "walk", forward="-y")
    for t in (0.0, 0.35, 0.7):
        head = a.pose_joints(arm, t, space="world")["Head"]
        w = world_matrices(s, animation_overlay(s, t))[helmet.id]
        # the helmet keeps its place relative to the head, wherever the character goes
        assert np.allclose(w, head @ np.linalg.inv(head0) @ rest, atol=1e-9), t
    for _ in range(4):  # (the walk, the position, the rotation, the auto-rig)
        s.history.undo()
    assert helmet.parent is None and "Rig" not in [o.name for o in s.objects]
