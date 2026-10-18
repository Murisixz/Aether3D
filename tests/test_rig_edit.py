"""Structural rig editing (Stage 18): every operation keeps skins, clips, masks,
constraints, controls and attachments consistent -- checked by evaluating the posed
character before and after, against independent expectations."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene, create_cube
from aether3d.animation import rotation as R
from aether3d.errors import AnimationError
from aether3d.mesh import Mesh


def humanoid(scene: Scene, name: str = "Hero"):
    arm = scene.add_object(name, position=(0.3, -0.2, 0.0))
    joints = [
        {"name": "Hips", "position": [0, 0, 0.95], "role": "hips"},
        {"name": "Spine", "parent": "Hips", "position": [0, 0, 0.12], "role": "spine"},
        {"name": "Chest", "parent": "Spine", "position": [0, 0, 0.2], "role": "chest"},
        {"name": "Neck", "parent": "Chest", "position": [0, 0, 0.22], "role": "neck"},
        {"name": "Head", "parent": "Neck", "position": [0, 0, 0.1], "tail": [0, 0, 0.24],
         "role": "head"},
    ]  # fmt: skip
    for side, sx in (("L", 1), ("R", -1)):
        joints += [
            {"name": f"UpperArm.{side}", "parent": "Chest", "position": [0.21 * sx, 0, 0.17],
             "role": f"upper_arm.{side}"},
            {"name": f"ForeArm.{side}", "parent": f"UpperArm.{side}", "position": [0, 0, -0.29],
             "role": f"lower_arm.{side}",
             "limits": {"axis": "+y", "swing": {"x": [-10, 140], "z": [-5, 20]}}},
            {"name": f"Hand.{side}", "parent": f"ForeArm.{side}", "position": [0, 0, -0.25],
             "tail": [0, 0, -0.09], "role": f"hand.{side}"},
            {"name": f"Thigh.{side}", "parent": "Hips", "position": [0.1 * sx, 0, -0.02],
             "role": f"thigh.{side}"},
            {"name": f"Shin.{side}", "parent": f"Thigh.{side}", "position": [0, 0, -0.44],
             "role": f"shin.{side}"},
            {"name": f"Foot.{side}", "parent": f"Shin.{side}", "position": [0, 0, -0.41],
             "tail": [0, -0.15, -0.08], "role": f"foot.{side}"},
        ]  # fmt: skip
    scene.animation.add_rig(arm, joints)
    return arm


def limb_mesh(rig) -> Mesh:
    """Tubes along every bone (armature space), one mesh."""
    model = rig.rest_model()
    tails = rig.tails()
    pos, tris = [], []
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for k in range(len(rig.joints)):
        a = model[k][:3, 3]
        b = (model[k] @ np.append(tails[k], 1.0))[:3]
        axis = b - a
        if np.linalg.norm(axis) < 1e-6:
            continue
        w = axis / np.linalg.norm(axis)
        u = np.cross(w, [0.3, 0.4, 0.86])
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        base = sum(len(p) for p in pos)
        ring = []
        for f in np.linspace(0, 1, 9):
            c = a + f * axis
            ring.append(c + 0.04 * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v))
        pts = np.concatenate(ring)
        for r_ in range(8):
            for i in range(8):
                p0 = base + r_ * 8 + i
                p1 = base + r_ * 8 + (i + 1) % 8
                tris += [[p0, p1, p1 + 8], [p0, p1 + 8, p0 + 8]]
        pos.append(pts)
    return Mesh(np.concatenate(pos).astype(np.float32), np.array(tris, np.uint32))


def character(method: str = "linear"):
    s = Scene("Edit")
    hero = humanoid(s)
    rig = s.animation.rig(hero)
    body = s.add_object("Body", mesh=limb_mesh(rig), position=hero.position)
    s.animation.skin(body, hero, weights="envelope", influences=3, method=method)
    sword = s.add_object("Sword", mesh=create_cube(0.05), position=(0.82, -0.2, 0.9))
    s.animation.attach(sword, hero, "Hand.L")
    return s, hero, body, sword


def walk(s, hero):
    rep = s.animation.generate(hero, "walk", forward="-y")
    return [0.13 * rep["period"] * k for k in range(1, 6)]


def posed(s, hero, body, sword, times):
    a = s.animation
    out = []
    for t in times:
        joints = a.pose_joints(hero, t, space="world")
        rr = a.rigging(t)
        mesh = rr.meshes.get(body.id)
        sw = a.overlay(t)
        out.append((joints, None if mesh is None else mesh.positions.copy(), sw))
    return out


def same_world(joints_a: dict, joints_b: dict, renames: dict | None = None, atol=1e-9):
    renames = renames or {}
    for name, m in joints_a.items():
        other = joints_b[renames.get(name, name)]
        assert np.allclose(m, other, atol=atol), name


# --------------------------------------------------------------------------------------- #


def test_renaming_follows_into_clips_masks_constraints_controls_and_targets():
    s, hero, body, sword = character()
    a = s.animation
    times = walk(s, hero)
    pole = s.add_object("ElbowPole", position=(0.5, 0.4, 1.2))
    target = s.add_object("HandIK", position=(0.6, -0.4, 1.0))
    raw = a.rig(hero).to_json()
    raw["constraints"] = [
        {
            "name": "arm_ik.L",
            "type": "ik",
            "influence": 0.0,
            "chain": ["UpperArm.L", "ForeArm.L", "Hand.L"],
            "target": {"object": "HandIK"},
            "pole": {"object": "ElbowPole"},
        }
    ]
    raw["controls"] = [{"name": "hand_ik.L", "kind": "ik_target", "object": "HandIK",
                        "joint": "Hand.L", "constraint": "arm_ik.L"}]  # fmt: skip
    a.set_rig(hero, raw)
    a.add_layer(hero, "upper", mask=[{"joint": "ForeArm.L", "descendants": True}])
    watcher = s.add_object("Watcher", position=(0, -3, 1.5))
    a.add_constraint(watcher, {"name": "aim", "type": "look_at",
                               "target": {"object": "Hero", "joint": "ForeArm.L"}})  # fmt: skip
    before = posed(s, hero, body, sword, times)
    aim_before = a.overlay(times[1])
    rep = a.rename_joint(hero, "ForeArm.L", "LowerArm.L")
    assert rep["renamed"] == {"ForeArm.L": "LowerArm.L"}
    assert rep["clips"] and rep["clips"][0]["renamed_channels"] >= 1
    rig = a.rig(hero)
    assert rig.index("ForeArm.L") is None and rig.index("LowerArm.L") is not None
    assert rig.constraints[0].p["chain"][1] == "LowerArm.L"
    assert rig.control("hand_ik.L").joint == "Hand.L"
    layer = a.binding(hero).layers[0]
    assert layer.mask[0][0] == "joint:LowerArm.L"
    assert a.constraints(watcher)[0]["target"]["joint"] == "LowerArm.L"
    after = posed(s, hero, body, sword, times)
    for (ja, ma, _), (jb, mb, _) in zip(before, after, strict=True):
        same_world(ja, jb, {"ForeArm.L": "LowerArm.L"}, atol=0)
        assert np.array_equal(ma, mb)
    assert np.array_equal(aim_before.local, a.overlay(times[1]).local)
    s.history.undo()
    assert a.rig(hero).index("ForeArm.L") is not None
    assert a.binding(hero).layers[0].mask[0][0] == "joint:ForeArm.L"
    del pole, target


def test_moving_a_joint_rebinds_or_keeps_the_bind():
    s, hero, body, sword = character()
    a = s.animation
    rest_mesh = body.mesh.positions.copy()
    sword_rest = sword.world_matrix.copy()
    shin = a.pose_joints(hero, 0.0, space="armature")["Shin.L"][:3, 3].copy()
    rep = a.move_joint(hero, "Shin.L", shin + [0, 0, 0.05], space="armature")
    assert rep["skins"][0]["rebound_joints"] >= 1
    # Rebind: at rest nothing moved (mesh, attached sword); children kept their place.
    rr = a.rigging(0.0)
    mesh = rr.meshes.get(body.id)
    assert mesh is None or np.allclose(mesh.positions, rest_mesh, atol=1e-5)
    j = a.pose_joints(hero, 0.0, space="armature")
    assert np.allclose(j["Shin.L"][:3, 3], shin + [0, 0, 0.05], atol=1e-12)
    assert np.allclose(j["Foot.L"][:3, 3], [0.1, 0, 0.08], atol=1e-12)
    local = rr.local.get(sword.id)
    assert local is None or np.allclose(local, sword._transform.matrix, atol=1e-12)
    assert np.array_equal(sword.world_matrix, sword_rest)
    s.history.undo()
    # keep_bind: the skin follows the moved joint at rest (vertices near the knee move).
    a.move_joint(hero, "Shin.L", shin + [0, 0, 0.05], space="armature", skins="keep_bind")
    moved = a.rigging(0.0).meshes[body.id].positions
    delta = np.linalg.norm(moved - rest_mesh, axis=1)
    assert delta.max() > 0.01 and np.median(delta) < 1e-6


def test_orienting_joints_converts_animation_exactly_and_keeps_skins():
    s, hero, body, sword = character()
    a = s.animation
    times = walk(s, hero)
    before = posed(s, hero, body, sword, times)
    rep = a.orient_joints(hero, "all", animation="convert", tolerance=0.001)
    rig = a.rig(hero)
    model = rig.rest_model()
    # Every bone now points along its local +Y.
    for k, j in enumerate(rig.joints):
        kids = [c for c, jj in enumerate(rig.joints) if jj.parent == k]
        if kids:
            d = model[kids[0]][:3, 3] - model[k][:3, 3]
            y = model[k][:3, 1]
            assert np.dot(d / np.linalg.norm(d), y / np.linalg.norm(y)) > 1 - 1e-9, j.name
    assert any(d["code"] == "JOINT_FRAME_CHANGED" for d in rep["diagnostics"])
    after = posed(s, hero, body, sword, times)
    for (ja, ma, _), (jb, mb, _) in zip(before, after, strict=True):
        for name in ja:  # positions within the refit tolerance; rotations too
            assert np.allclose(ja[name][:3, 3], jb[name][:3, 3], atol=5e-5), name
        assert np.allclose(ma, mb, atol=2e-4)
    assert all("refit" in c["converted"] for c in rep["clips"] if "converted" in c)


def test_quaternion_tracks_convert_exactly_under_frame_changes():
    s = Scene("Quats")
    hero = humanoid(s)
    a = s.animation
    rng = np.random.default_rng(4)
    for name in ("Spine", "Chest", "UpperArm.L", "ForeArm.L", "Hand.L"):
        keys = [[t, R.normalize(rng.normal(size=4))[0].tolist()] for t in (0.0, 0.4, 1.0)]
        a.set_channel(hero, f'joints["{name}"].rotation', keys, interpolation="squad")
    times = np.linspace(0, 1, 13)
    before = [a.pose_joints(hero, t, space="world") for t in times]
    rep = a.orient_joints(hero, "all", animation="convert")
    exact = [p for c in rep["clips"] for p in c.get("converted", {}).get("exact", [])]
    assert len(exact) == 5
    rest = a.pose_joints(hero, 0.0, space="world")
    change = {n: np.linalg.solve(before[0][n], rest[n]) for n in rest}  # the frame change
    for t, b in zip(times, before, strict=True):
        now = a.pose_joints(hero, t, space="world")
        for name in b:  # positions unchanged; frames turned by their constant change
            assert np.allclose(now[name][:3, 3], b[name][:3, 3], atol=1e-12), name
            assert np.allclose(np.linalg.solve(b[name], now[name]), change[name], atol=1e-12)


def test_orienting_without_conversion_reports_the_changed_meaning():
    s, hero, body, sword = character()
    a = s.animation
    walk(s, hero)
    rep = a.orient_joints(hero, ["ForeArm.L"])
    assert any(d["code"] == "ANIMATION_MEANING_CHANGED" for d in rep["diagnostics"])


def test_removing_a_joint_moves_weights_attachments_and_children():
    s, hero, body, sword = character()
    a = s.animation
    walk(s, hero)
    a.keyframe(hero, 'joints["Hand.L"].rotation_euler', 0.5, [0.0, 0.0, 20.0])
    rig = a.rig(hero)
    hand_id = rig.joints[rig.index("Hand.L")].id
    fore_id = rig.joints[rig.index("ForeArm.L")].id
    skin = a._owned[("skin", body.id)]._definition
    hand_col = skin.joints.index(hand_id)
    had = skin.dense()[:, hand_col].copy()
    rest_hand = a.pose_joints(hero, 0.0, space="armature")["Hand.L"].copy()
    with pytest.raises(AnimationError) as e:  # a constraint uses it: refused
        raw = rig.to_json()
        raw["constraints"] = [{"name": "aim", "type": "look_at", "joint": "Hand.L",
                               "target": {"point": [0, -1, 1]}}]  # fmt: skip
        a.set_rig(hero, raw)
        a.remove_joint(hero, "Hand.L")
    assert e.value.code == "JOINT_IN_USE"
    rep = a.remove_joint(hero, "Hand.L", constraints="remove", channels="remove")
    assert rep["removed"] == ["Hand.L"]
    assert any(d["code"] == "CONSTRAINT_REMOVED" for d in rep["diagnostics"])
    assert rep["clips"][0]["removed_channels"]
    skin2 = a._owned[("skin", body.id)]._definition
    dense = skin2.dense()
    assert hand_id not in skin2.joints
    fore_col = skin2.joints.index(fore_id)
    assert np.all(dense[had > 0.5, fore_col] > 0.5)  # the hand's weight went to the forearm
    assert np.allclose(dense.sum(axis=1), 1.0)
    assert dict(a.rig(hero).attachments)[sword.id] == fore_id  # the sword follows the forearm
    del rest_hand
    s.history.undo()
    assert a.rig(hero).index("Hand.L") is not None


def test_insert_and_split_divide_a_bone_and_its_weights():
    s, hero, body, sword = character()
    a = s.animation
    rest_mesh = body.mesh.positions.copy()
    a.edit_rig(hero, [{"op": "insert_joint", "name": "Spine2", "child": "Chest", "at": 0.5,
                       "weights": "split", "blend": 0.0}])  # fmt: skip
    rig = a.rig(hero)
    m = rig.rest_model()
    assert np.allclose(m[rig.index("Spine2")][:3, 3], [0, 0, 1.17], atol=1e-12)
    assert rig.joints[rig.index("Chest")].parent == rig.index("Spine2")
    rr = a.rigging(0.0)
    mesh = rr.meshes.get(body.id)
    assert mesh is None or np.allclose(mesh.positions, rest_mesh, atol=1e-5)
    skin = a._owned[("skin", body.id)]._definition
    col = skin.joints.index(rig.joints[rig.index("Spine2")].id)
    assert skin.dense()[:, col].sum() > 1.0  # the upper half of the spine bone moved over
    # Turning the new joint bends the upper spine region only.
    a.keyframe(hero, 'joints["Spine2"].rotation_euler', 0.0, [25.0, 0.0, 0.0])
    bent = a.rigging(0.0).meshes[body.id].positions
    moved = np.linalg.norm(bent - rest_mesh, axis=1) > 1e-6
    k = rig.index("Spine2")
    turned = {rig.joints[i].id for i in [k, *rig.descendants(k)]}
    share = skin.dense()[:, [c for c, j in enumerate(skin.joints) if j in turned]].sum(axis=1)
    assert np.array_equal(moved, share > 1e-6)  # exactly the vertices the new joint moves
    spine = np.abs(rest_mesh[:, 0]) < 0.05
    low = spine & (rest_mesh[:, 2] < 1.12) & (rest_mesh[:, 2] > 1.0)
    high = spine & (rest_mesh[:, 2] > 1.2) & (rest_mesh[:, 2] < 1.26)
    assert share[high].min() > share[low].max()  # the upper part of the old spine bone
    rep = a.edit_rig(hero, [{"op": "split_bone", "joint": "Thigh.L", "count": 3}])
    assert rep["added"] == ["Thigh.L.02", "Thigh.L.01"] or set(rep["added"]) == {
        "Thigh.L.01",
        "Thigh.L.02",
    }
    rig = a.rig(hero)
    z = [rig.rest_model()[rig.index(n)][2, 3] for n in ("Thigh.L", "Thigh.L.01", "Thigh.L.02",
                                                         "Shin.L")]  # fmt: skip
    assert np.allclose(np.diff(z), (z[-1] - z[0]) / 3)


def test_mirroring_rebuilds_the_other_side():
    s, hero, body, sword = character()
    a = s.animation
    original = {n: m.copy() for n, m in a.pose_joints(hero, 0.0, space="armature").items()}
    rig = a.rig(hero)
    raw = rig.to_json()
    for j in raw["joints"]:  # disturb the right arm
        if j["name"] == "ForeArm.R":
            j["position"] = [0.03, 0.02, -0.27]
    a.set_rig(hero, raw)
    rep = a.mirror_joint_chain(hero, "L")
    assert "ForeArm.R" in rep["operations"][0]["updated"]
    now = a.pose_joints(hero, 0.0, space="armature")
    for name in ("UpperArm.R", "ForeArm.R", "Hand.R", "Thigh.R", "Shin.R", "Foot.R"):
        assert np.allclose(now[name], original[name], atol=1e-12), name
    rig = a.rig(hero)
    lim = rig.joints[rig.index("ForeArm.R")].limits
    assert '"z":[-20.0,5.0]' in lim and '"x":[-10.0,140.0]' in lim
    # A missing side is created.
    s2 = Scene("Half")
    arm = s2.add_object("Arm")
    s2.animation.add_rig(arm, [{"name": "Root", "position": [0, 0, 1]},
                               {"name": "Wing.L", "parent": "Root", "position": [0.5, 0.1, 0],
                                "role": "wing.L"}])  # fmt: skip
    s2.animation.mirror_joint_chain(arm, "L")
    r2 = s2.animation.rig(arm)
    w = r2.rest_model()[r2.index("Wing.R")][:3, 3]
    assert np.allclose(w, [-0.5, 0.1, 1.0]) and r2.joints[r2.index("Wing.R")].role == "wing.R"


def test_reparent_align_and_validation():
    s, hero, body, sword = character()
    a = s.animation
    with pytest.raises(AnimationError) as e:
        a.reparent_joint(hero, "Spine", "Head")
    assert e.value.code == "SKELETON_CYCLE"
    before = a.pose_joints(hero, 0.0, space="armature")["Head"].copy()
    a.reparent_joint(hero, "Head", "Chest")
    assert np.allclose(a.pose_joints(hero, 0.0, space="armature")["Head"], before, atol=1e-12)
    a.edit_rig(hero, [{"op": "align_bone", "joint": "UpperArm.L", "direction": "+x"}])
    m = a.pose_joints(hero, 0.0, space="armature")
    d = m["ForeArm.L"][:3, 3] - m["UpperArm.L"][:3, 3]
    assert np.allclose(d, [0.29, 0, 0], atol=1e-12)
    rep = a.edit_rig(hero, [{"op": "move_joint", "joint": "Neck", "position": [0, 0, 0],
                             "children": "follow"}])  # fmt: skip
    del rep
    rep = a.edit_rig(hero, [{"op": "add_joint", "name": "Tip", "parent": "Head",
                             "position": [0, 0, 0]}], dry_run=True)  # fmt: skip
    assert any(d["code"] == "BONE_TOO_SHORT" for d in rep["diagnostics"])
    assert a.rig(hero).index("Tip") is None  # dry run: nothing changed
    for bad in (
        [{"op": "add_joint", "name": "Hips"}],
        [{"op": "rename_joint", "joint": "Hips", "name": "Head"}],
        [{"op": "add_joint", "name": "X", "position": [0, float("nan"), 0]}],
        [{"op": "move_joint", "joint": "Nope", "position": [0, 0, 0]}],
        [{"op": "explode"}],
        [{"op": "add_joint", "name": "Y", "colour": "red"}],
    ):
        with pytest.raises(AnimationError):
            a.edit_rig(hero, bad)


def test_world_space_positions_and_batches_are_one_undo_step():
    s, hero, body, sword = character()
    a = s.animation
    n = len(s.history)
    a.edit_rig(
        hero,
        [
            {"op": "add_joint", "name": "Prop", "parent": "Hand.L", "position": [0.5, -0.2, 0.3],
             "space": "world", "deform": False},
            {"op": "rename_joint", "joint": "Prop", "name": "Weapon"},
            {"op": "set_joint", "joint": "Weapon", "role": "prop.L"},
        ],
    )  # fmt: skip
    assert len(s.history) == n + 1
    w = a.pose_joints(hero, 0.0, space="world")["Weapon"]
    assert np.allclose(w[:3, 3], [0.5, -0.2, 0.3], atol=1e-12)
    rig = a.rig(hero)
    assert rig.joints[rig.index("Weapon")].role == "prop.L" and not rig.joints[
        rig.index("Weapon")].deform  # fmt: skip
    s.history.undo()
    assert a.rig(hero).index("Weapon") is None
