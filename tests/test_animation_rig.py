"""Rigs: skeletons, forward kinematics, attachments, skins (linear and dual quaternion),
morph targets, IK and constraints -- against independent computations."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from parity import assert_render_scenes_equal

from aether3d import Scene, math3d
from aether3d.animation import rotation as R
from aether3d.animation.rig import parse_rig, skin_dual_quaternion, skin_linear
from aether3d.errors import AnimationError
from aether3d.mesh import Mesh
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import _extract_full, extract_render_scene, world_matrices

RNG = np.random.default_rng(31)


def arm_scene() -> Scene:
    s = Scene("Arm")
    arm = s.add_object("Armature", position=(0.5, 0.0, 0.0))
    s.animation.add_rig(
        arm,
        [
            {"name": "Shoulder", "position": [0, 0, 1]},
            {"name": "Elbow", "parent": "Shoulder", "position": [0, 1, 0]},
            {"name": "Wrist", "parent": "Elbow", "position": [0, 1, 0], "tail": [0, 0.2, 0]},
        ],
    )
    return s


def cylinder(n_ring: int = 16, rings: int = 21, radius: float = 0.2, length: float = 2.0) -> Mesh:
    """A capped-free tube along +Y from 0 to ``length`` (rings x n_ring vertices)."""
    ang = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
    ys = np.linspace(0, length, rings)
    pos = np.array(
        [[radius * math.cos(a), y, radius * math.sin(a)] for y in ys for a in ang], np.float32
    )
    nrm = np.array([[math.cos(a), 0.0, math.sin(a)] for _ in ys for a in ang], np.float32)
    tris = []
    for r in range(rings - 1):
        for i in range(n_ring):
            a, b = r * n_ring + i, r * n_ring + (i + 1) % n_ring
            c, d = a + n_ring, b + n_ring
            tris += [[a, c, b], [b, c, d]]
    return Mesh(pos, np.array(tris, np.uint32), normals=nrm)


# --------------------------------------------------------------------------------------- #
# Skeletons
# --------------------------------------------------------------------------------------- #


def test_forward_kinematics_matches_independent_matrix_products():
    joints = []
    for i in range(12):
        joints.append(
            {
                "name": f"J{i}",
                "parent": None if i == 0 else f"J{(i - 1) // 2}",
                "position": RNG.normal(0, 1, 3).tolist(),
                "rotation": R.normalize(RNG.normal(size=4))[0].tolist(),
                "scale": RNG.uniform(0.5, 1.5, 3).tolist(),
            }
        )
    rig = parse_rig({"object": "x", "joints": list(reversed(joints))}, "rig")  # any order in
    model = rig.rest_model()
    expected = {}
    for j in joints:  # parents first in this list
        local = math3d.compose_matrix(j["position"], j["rotation"], j["scale"])
        expected[j["name"]] = local if j["parent"] is None else expected[j["parent"]] @ local
    for k, j in enumerate(rig.joints):
        assert np.allclose(model[k], expected[j.name], atol=1e-12)
    assert all(rig.joints[k].parent < k for k in range(len(rig.joints)))


@pytest.mark.parametrize(
    ("joints", "code"),
    [
        ([{"name": "A", "parent": "B"}, {"name": "B", "parent": "A"}], "SKELETON_CYCLE"),
        ([{"name": "A", "parent": "Z"}], "INVALID_RIG"),
        ([{"name": "A"}, {"name": "A"}], "INVALID_RIG"),
        ([{"name": "A", "scale": [1, 0, 1]}], "INVALID_RIG"),
        ([{"name": 'A"B'}], "INVALID_NAME"),
        ([], "INVALID_RIG"),
    ],
)
def test_invalid_skeletons_are_refused(joints, code):
    with pytest.raises(AnimationError) as err:
        parse_rig({"object": "x", "joints": joints}, "rig")
    assert err.value.code == code


def test_joint_channels_pose_the_skeleton_and_follow_the_armature():
    s = arm_scene()
    a = s.animation
    arm = s.get_by_name("Armature")
    a.keyframe(arm, 'joints["Shoulder"].rotation_euler.z', 0.0, 0.0, interpolation="linear")
    a.keyframe(arm, 'joints["Shoulder"].rotation_euler.z', 1.0, 90.0, interpolation="linear")
    a.keyframe(arm, "transform.position.x", 0.0, 0.5, interpolation="linear")
    a.keyframe(arm, "transform.position.x", 1.0, 2.5, interpolation="linear")
    wrist = a.pose_joints(arm, 1.0)["Wrist"][:3, 3]
    assert np.allclose(wrist, [2.5 - 2.0, 0.0, 1.0], atol=1e-12)  # turned 90 degrees, moved
    half = a.pose_joints(arm, 0.5, space="armature")["Elbow"][:3, 3]
    assert np.allclose(half, [-math.sin(math.pi / 4), math.cos(math.pi / 4), 1.0], atol=1e-12)


def test_attachments_follow_their_joints_through_any_hierarchy():
    s = arm_scene()
    a = s.animation
    arm = s.get_by_name("Armature")
    upper = s.add_cube(name="Upper", size=0.2, position=(0.5, 0.5, 1.0))
    hand = s.add_cube(
        name="Hand", size=0.1, parent=upper, position=(0.0, 1.5, 0.0)
    )  # child of Upper
    a.attach(upper, arm, "Shoulder")
    a.attach(hand, arm, "Wrist")
    a.keyframe(arm, 'joints["Shoulder"].rotation_euler.z', 0.0, 0.0, interpolation="linear")
    a.keyframe(arm, 'joints["Shoulder"].rotation_euler.z', 1.0, 90.0, interpolation="linear")
    a.keyframe(arm, 'joints["Elbow"].rotation_euler.z', 0.0, 0.0, interpolation="linear")
    a.keyframe(arm, 'joints["Elbow"].rotation_euler.z', 1.0, 90.0, interpolation="linear")
    w = world_matrices(s, animation_overlay(s, 1.0))
    joints = a.pose_joints(arm, 1.0)
    rest = s.animation.rig(arm).rest_model()
    a_rest = world_matrices(s)[arm.id]
    for obj, jname, k in ((upper, "Shoulder", 0), (hand, "Wrist", 2)):
        expected = joints[jname] @ np.linalg.inv(a_rest @ rest[k]) @ world_matrices(s)[obj.id]
        assert np.allclose(w[obj.id], expected, atol=1e-12)
    assert upper.position == (0.5, 0.5, 1.0)  # authored: untouched
    # The armature's ancestors cannot follow its joints.
    parent = s.add_object("Parent")
    arm.set_parent(parent)
    with pytest.raises(AnimationError) as err:
        a.attach(parent, arm, "Elbow")
    assert err.value.code == "ATTACHMENT_CYCLE"


# --------------------------------------------------------------------------------------- #
# Skins
# --------------------------------------------------------------------------------------- #


def reference_lbs(pos, idx, w, palette):
    out = np.zeros((pos.shape[0], 3))
    for v in range(pos.shape[0]):
        m = sum(w[v, j] * palette[idx[v, j]] for j in range(idx.shape[1]))
        out[v] = m[:3, :3] @ pos[v].astype(np.float64) + m[:3, 3]
    return out


def test_linear_blend_skinning_matches_a_per_vertex_reference_and_keeps_normals_outward():
    n, m = 400, 6
    pos = RNG.normal(size=(n, 3)).astype(np.float32)
    nrm = R.normalize(np.concatenate([RNG.normal(size=(n, 3)), np.zeros((n, 1))], axis=1))[
        :, :3
    ].astype(np.float32)
    idx = RNG.integers(0, m, (n, 4)).astype(np.int32)
    w = RNG.random((n, 4))
    w /= w.sum(axis=1, keepdims=True)
    palette = np.zeros((m, 4, 4))
    for j in range(m):
        q = R.normalize(RNG.normal(size=4))[0]
        scale = RNG.uniform(0.5, 2.0, 3) * (-1 if j == 0 else 1)  # one mirrored joint
        palette[j] = math3d.compose_matrix(RNG.normal(size=3), q, scale)
    p, nn = skin_linear(pos, nrm, idx, w, palette)
    assert np.allclose(p, reference_lbs(pos, idx, w, palette), atol=1e-5)
    # Normals: the inverse transpose of the blended linear part, normalized, outward.
    for v in range(0, n, 37):
        mm = sum(w[v, j] * palette[idx[v, j]] for j in range(4))[:3, :3]
        expect = np.linalg.inv(mm).T @ nrm[v].astype(np.float64)
        expect /= np.linalg.norm(expect)
        assert np.allclose(nn[v], expect, atol=1e-5)


def test_bind_pose_reproduces_the_mesh_and_rigid_weights_move_exactly_with_their_joint():
    s = arm_scene()
    a = s.animation
    arm = s.get_by_name("Armature")
    tube = s.add_object("Tube", parent=arm, position=(0.0, 0.0, 1.0), mesh=cylinder())
    report = a.skin(tube, arm, weights="nearest")
    assert report["influences"] == 1 and report["joints_used"] >= 2
    rest = a.rigging(0.0).meshes[tube.id]
    assert np.allclose(rest.positions, tube.mesh.positions, atol=2e-6)
    a.keyframe(arm, 'joints["Elbow"].rotation_euler.x', 0.0, 0.0, interpolation="linear")
    a.keyframe(arm, 'joints["Elbow"].rotation_euler.x', 1.0, 90.0, interpolation="linear")
    bent = a.rigging(1.0).meshes[tube.id].positions
    skin = a.rig(arm)
    names = [j.name for j in skin.joints]
    weights = s.animation._skins()[tube.id]
    elbow = names.index("Elbow")
    # Vertices bound to the elbow rotate 90 degrees about X around the elbow (y = 1).
    rows = np.flatnonzero(
        weights.indices[:, 0] == [skin.index_of_id(j) for j in weights.joints].index(elbow)
    )
    p = tube.mesh.positions[rows].astype(np.float64)
    expect = np.stack([p[:, 0], 1.0 - p[:, 2], (p[:, 1] - 1.0)], axis=1)
    assert np.allclose(bent[rows], expect, atol=2e-6)


def test_dual_quaternion_skinning_keeps_volume_where_linear_blending_collapses():
    mesh = cylinder(rings=41)
    pos = mesh.positions
    y = pos[:, 1].astype(np.float64)
    w1 = np.clip((y - 0.6) / 0.8, 0.0, 1.0)  # a smooth blend around the middle
    idx = np.stack([np.zeros(len(y), np.int32), np.ones(len(y), np.int32)], axis=1)
    w = np.stack([1.0 - w1, w1], axis=1)
    twist = math3d.compose_matrix(
        (0, 0, 0), math3d.quat_from_axis_angle((0, 1, 0), 170.0), (1, 1, 1)
    )
    palette = np.stack([np.eye(4), twist])
    lbs, _ = skin_linear(pos, mesh.normals, idx, w, palette)
    dqs, _ = skin_dual_quaternion(pos, mesh.normals, idx, w, palette)
    mid = np.abs(y - 1.0) < 0.03
    r_lbs = np.hypot(lbs[mid, 0], lbs[mid, 2]).min()
    r_dqs = np.hypot(dqs[mid, 0], dqs[mid, 2]).min()
    assert r_lbs < 0.2 * 0.2  # the candy wrapper: the tube pinches
    assert r_dqs == pytest.approx(0.2, abs=2e-3)  # rigid blending keeps the radius
    assert np.allclose(dqs[y < 0.5], pos[y < 0.5], atol=1e-6)  # untouched where weight is 0


def test_morph_targets_blend_before_skinning():
    s = arm_scene()
    a = s.animation
    arm = s.get_by_name("Armature")
    tube = s.add_object("Tube", parent=arm, position=(0.0, 0.0, 1.0), mesh=cylinder())
    n = tube.mesh.vertex_count
    bulge = np.zeros((n, 3))
    bulge[:, 0] = tube.mesh.positions[:, 0] * 0.5  # radius x 1.5 along x
    a.set_morph_targets(tube, [{"name": "bulge", "positions": bulge.tolist()}])
    a.keyframe(tube, 'morph["bulge"]', 0.0, 0.0, interpolation="linear")
    a.keyframe(tube, 'morph["bulge"]', 1.0, 1.0, interpolation="linear")
    half = a.rigging(0.5).meshes[tube.id].positions
    assert np.allclose(half[:, 0], tube.mesh.positions[:, 0] * 1.25, atol=1e-6)
    a.skin(tube, arm, weights="nearest")
    a.keyframe(arm, "transform.position.z", 0.0, 0.0, interpolation="linear")
    a.keyframe(arm, "transform.position.z", 1.0, 0.0, interpolation="linear")
    both = a.rigging(1.0).meshes[tube.id].positions
    assert np.allclose(both[:, 0], tube.mesh.positions[:, 0] * 1.5, atol=1e-5)


# --------------------------------------------------------------------------------------- #
# IK and constraints
# --------------------------------------------------------------------------------------- #


def test_two_bone_ik_reaches_bends_toward_its_pole_and_reports_residuals():
    s = arm_scene()
    a = s.animation
    arm = s.get_by_name("Armature")
    target = s.add_object("Target", position=(0.5 + 1.2, 1.0, 1.0))
    s.add_object("Pole", position=(0.5, 0.5, 3.0))
    a.add_constraint(
        arm,
        {
            "name": "reach",
            "type": "ik",
            "chain": ["Shoulder", "Elbow", "Wrist"],
            "target": {"object": "Target"},
            "pole": {"object": "Pole"},
        },
    )
    joints = a.pose_joints(arm, 0.0)
    assert np.allclose(joints["Wrist"][:3, 3], target.world_position, atol=1e-12)
    for j in ("Elbow", "Wrist"):  # bone lengths kept
        assert np.linalg.norm(
            joints[j][:3, 3] - joints["Shoulder" if j == "Elbow" else "Elbow"][:3, 3]
        ) == pytest.approx(1.0)
    assert joints["Elbow"][2, 3] > 1.0  # bent up, toward the pole
    assert a.rigging(0.0).residuals["Armature/reach"] == pytest.approx(0.0, abs=1e-12)
    target.position = (0.5 + 5.0, 0.0, 1.0)  # out of reach: fully extended toward it
    joints = a.pose_joints(arm, 0.0)
    assert np.allclose(joints["Wrist"][:3, 3], [0.5 + 2.0, 0.0, 1.0], atol=1e-9)
    assert a.rigging(0.0).residuals["Armature/reach"] == pytest.approx(3.0, abs=1e-9)


def test_fabrik_chains_and_joint_limits():
    s = Scene("Chain")
    arm = s.add_object("Armature")
    joints = [{"name": "J0"}] + [
        {"name": f"J{i}", "parent": f"J{i - 1}", "position": [0, 1, 0]} for i in range(1, 6)
    ]
    a = s.animation
    a.add_rig(arm, joints)
    s.add_object("Goal", position=(2.0, 2.5, 1.0))
    a.add_constraint(
        arm,
        {
            "name": "tip",
            "type": "ik",
            "chain": [f"J{i}" for i in range(6)],
            "solver": "fabrik",
            "target": {"object": "Goal"},
            "tolerance": 1e-9,
            "iterations": 200,
        },
    )
    tip = a.pose_joints(arm, 0.0)["J5"][:3, 3]
    assert np.allclose(tip, [2.0, 2.5, 1.0], atol=1e-6)
    # A joint limit (Euler degrees, local) is enforced after the solve and the miss reported.
    rig = a.rig(arm).to_json()
    rig["joints"][1]["limits"] = {"rotation": {"min": [None, None, -5.0], "max": [None, None, 5.0]}}
    a.set_rig(arm, rig)
    joints_now = a.pose_joints(arm, 0.0)
    local = np.linalg.inv(joints_now["J0"]) @ joints_now["J1"]
    e = R.to_euler(np.array(math3d.decompose_matrix(local)[1]))[0]
    assert -5.0 - 1e-9 <= e[2] <= 5.0 + 1e-9


def test_object_constraints_look_at_copy_child_of_and_distance():
    s = Scene("C")
    cam = s.add_camera(name="Cam", position=(0, -10, 2))
    car = s.add_cube(name="Car", position=(0, 0, 0))
    follower = s.add_cube(name="Follower", position=(3, 0, 0))
    tether = s.add_sphere(name="Tether", position=(10, 0, 0))
    a = s.animation
    a.keyframe(car, "transform.position", 0.0, (0, 0, 0), interpolation="linear")
    a.keyframe(car, "transform.position", 2.0, (6, 4, 0), interpolation="linear")
    a.set_constraints(cam, [{"name": "aim", "type": "look_at", "target": {"object": "Car"}}])
    a.set_constraints(follower, [{"name": "ride", "type": "child_of", "target": {"object": "Car"}}])
    a.set_constraints(
        tether,
        [{"name": "leash", "type": "limit_distance", "target": {"object": "Car"}, "max": 2.0}],
    )
    w = world_matrices(s, animation_overlay(s, 1.0))
    car_pos = np.array([3.0, 2.0, 0.0])
    forward = -w[cam.id][:3, 1]  # cameras look along local -Y
    to_car = car_pos - w[cam.id][:3, 3]
    assert np.allclose(
        forward / np.linalg.norm(forward), to_car / np.linalg.norm(to_car), atol=1e-12
    )
    assert np.allclose(w[follower.id][:3, 3], car_pos + [3, 0, 0], atol=1e-12)  # rides along
    assert np.linalg.norm(w[tether.id][:3, 3] - car_pos) == pytest.approx(2.0)
    # Influence animates: half way, the follower is half way to riding along.
    a.keyframe(follower, 'constraints["ride"].influence', 0.0, 0.5)
    w = world_matrices(s, animation_overlay(s, 1.0))
    assert np.allclose(w[follower.id][:3, 3], [3.0 + 1.5, 1.0, 0.0], atol=1e-12)


def test_constraint_cycles_are_refused():
    s = Scene("Cycle")
    s.add_cube(name="A")
    s.add_cube(name="B")
    a = s.animation
    a.set_constraints("A", [{"type": "copy_location", "target": {"object": "B"}}])
    with pytest.raises(AnimationError) as err:
        a.set_constraints("B", [{"type": "copy_location", "target": {"object": "A"}}])
    assert err.value.code == "CONSTRAINT_CYCLE"
    assert a.constraints("B") == []  # refused: nothing changed


# --------------------------------------------------------------------------------------- #
# Persistence, history, renders
# --------------------------------------------------------------------------------------- #


def rigged_scene() -> Scene:
    s = arm_scene()
    a = s.animation
    arm = s.get_by_name("Armature")
    tube = s.add_object("Tube", parent=arm, position=(0.0, 0.0, 1.0), mesh=cylinder())
    a.skin(tube, arm, weights="envelope", influences=2)
    s.add_object("Target", position=(1.7, 1.0, 1.5))
    a.add_constraint(
        arm,
        {
            "name": "reach",
            "type": "ik",
            "chain": ["Shoulder", "Elbow", "Wrist"],
            "target": {"object": "Target"},
        },
    )
    a.keyframe("Target", "transform.position.z", 0.0, 1.5, interpolation="linear")
    a.keyframe("Target", "transform.position.z", 2.0, 0.5, interpolation="linear")
    cam = s.add_camera(name="Cam", position=(4, -6, 3), look_at=(0.5, 1, 1))
    s.active_camera = cam
    return s


def test_rigs_skins_and_constraints_round_trip_and_undo():
    s = rigged_scene()
    path_doc = json.loads(json.dumps(s.to_dict()))
    loaded = Scene.from_dict(path_doc)
    for t in (0.0, 0.7, 2.0):
        a = s.animation.rigging(t)
        b = loaded.animation.rigging(t)
        tube = s.get_by_name("Tube").id
        assert a.meshes[tube].positions.tobytes() == b.meshes[tube].positions.tobytes()
    s.remove(s.get_by_name("Armature"))  # its rig and skin go with it
    assert s.animation._owned == {}
    s.history.undo()
    assert {k for k, _ in s.animation._owned} == {"rig", "skin"}


def test_retained_world_equals_full_extraction_with_skins_and_attachments():
    s = rigged_scene()
    cam = s.get_by_name("Cam")
    for t in (0.0, 1.3, 0.4, 2.0, 1.3):
        overlay = animation_overlay(s, t)
        assert_render_scenes_equal(
            extract_render_scene(s, cam, aspect=1.5, overlay=overlay),
            _extract_full(s, cam, aspect=1.5, overlay=overlay),
        )
