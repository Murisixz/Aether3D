"""Auto-rigged characters with Stage 17's animation tools (Stage 18): the generators'
F-curve clips, layers with masks and NLA strips, retargeting from a Stage 17 rig and
mirrored clips act on them exactly as on hand-built rigs -- one evaluation, no second
animation system."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface


@pytest.fixture(scope="module")
def body_mesh():
    return implicit_surface(humanoid_shapes(), 0.02)


def hero(mesh):
    s = Scene("Compat")
    body = s.add_object("Body", mesh=mesh)
    s.animation.autorig(body, name="Hero")
    return s, s.get_by_name("Hero")


def local(a, arm, t):
    """Each joint's matrix relative to its parent at time ``t``."""
    model = a.pose_joints(arm, t, space="armature")
    rig = a.rig(arm)
    out = {}
    for j in rig.joints:
        m = model[j.name]
        out[j.name] = m if j.parent < 0 else np.linalg.solve(model[rig.joints[j.parent].name], m)
    return out


def direction(a, arm, t, joint, child):
    """The world direction from a joint to its child at time ``t``."""
    w = a.pose_joints(arm, t, space="world")
    d = w[child][:3, 3] - w[joint][:3, 3]
    return d / np.linalg.norm(d)


def angle(a, src, tgt, t, joint, child) -> float:
    """Degrees between a bone's world direction in two characters at time ``t``."""
    d = direction(a, src, t, joint, child) @ direction(a, tgt, t, joint, child)
    return float(np.degrees(np.arccos(np.clip(d, -1.0, 1.0))))


def stage17_rig(scene, name, x):
    """Stage 17's hand-built humanoid (examples/animation_common.py): identity joint
    frames, no roles, no helpers."""
    arm = scene.add_object(name, position=(x, 0.0, 0.0))
    joints = [
        {"name": "Hips", "position": [0, 0, 0.95]},
        {"name": "Spine", "parent": "Hips", "position": [0, 0, 0.12]},
        {"name": "Chest", "parent": "Spine", "position": [0, 0, 0.2]},
        {"name": "Neck", "parent": "Chest", "position": [0, 0, 0.22]},
        {"name": "Head", "parent": "Neck", "position": [0, 0, 0.1], "tail": [0, 0, 0.24]},
    ]
    for side, sx in (("L", 1), ("R", -1)):
        joints += [
            {"name": f"UpperArm.{side}", "parent": "Chest", "position": [0.21 * sx, 0, 0.17],
             "tail": [0, 0, -0.29]},
            {"name": f"ForeArm.{side}", "parent": f"UpperArm.{side}", "position": [0, 0, -0.29],
             "tail": [0, 0, -0.25]},
            {"name": f"Hand.{side}", "parent": f"ForeArm.{side}", "position": [0, 0, -0.25],
             "tail": [0, 0, -0.09]},
            {"name": f"Thigh.{side}", "parent": "Hips", "position": [0.1 * sx, 0, -0.02],
             "tail": [0, 0, -0.44]},
            {"name": f"Shin.{side}", "parent": f"Thigh.{side}", "position": [0, 0, -0.44],
             "tail": [0, 0, -0.41]},
            {"name": f"Foot.{side}", "parent": f"Shin.{side}", "position": [0, 0, -0.41],
             "tail": [0, -0.15, -0.08]},
        ]  # fmt: skip
    scene.animation.add_rig(arm, joints)
    return arm


def test_layers_and_strips_drive_an_auto_rigged_character(body_mesh):
    s, arm = hero(body_mesh)
    a = s.animation
    walk = a.generate(arm, "walk", forward="-y", travel=False)["clip"]
    wave = a.generate(arm, "wave", forward="-y", bind=False)["clip"]
    t = 0.7
    walking = local(a, arm, t)
    a.set_action(arm, wave)
    waving = local(a, arm, t)
    a.set_action(arm, walk)
    a.add_layer(arm, "wave", mask=[{"joint": "UpperArm.R", "descendants": True}])
    a.add_strip(arm, "wave", wave)
    both = local(a, arm, t)
    for j in ("UpperArm.R", "ForeArm.R"):  # (the masked joints the wave keys)
        assert np.allclose(both[j], waving[j], atol=1e-9), j
    for j in ("Hips", "Thigh.L", "Shin.R", "UpperArm.L", "Spine", "Head"):
        assert np.allclose(both[j], walking[j], atol=1e-9), j
    assert not np.allclose(walking["UpperArm.R"], waving["UpperArm.R"], atol=1e-3)
    a.update_layer(arm, "wave", weight=0.0)
    assert np.allclose(local(a, arm, t)["UpperArm.R"], walking["UpperArm.R"], atol=1e-9)
    s.history.undo()  # (the weight change, one step)
    assert np.allclose(local(a, arm, t)["UpperArm.R"], waving["UpperArm.R"], atol=1e-9)


def test_stage17_clips_retarget_onto_an_auto_rigged_character(body_mesh):
    s, arm = hero(body_mesh)
    a = s.animation
    src = stage17_rig(s, "Old", 3.0)
    walk = a.generate(src, "walk", forward="-y", travel=False)["clip"]
    rep = a.retarget(src, arm, walk)
    shared = {"Hips", "Spine", "Chest", "Neck", "Head"} | {
        f"{b}.{side}" for b in ("UpperArm", "ForeArm", "Hand", "Thigh", "Shin", "Foot")
        for side in "LR"
    }  # fmt: skip
    assert shared <= set(rep["mapped"])
    assert {"Shoulder.L", "ForeArmTwist.R", "Toe.L"} <= set(rep["unmapped_target"])
    # Rotations are mapped relative to each rig's rest: the legs and the spine (the same
    # rests) point the same way all along; the arms differ by their rests' difference (an
    # A-pose against hanging arms), which the report names.
    assert set(rep["rest_mismatch"]) == {"ForeArm.L", "ForeArm.R", "Hand.L", "Hand.R"}
    a.set_action(src, None)  # (both at rest)
    rest_arm = angle(a, src, arm, 0.0, "UpperArm.L", "ForeArm.L")
    a.set_action(src, walk)
    a.set_action(arm, rep["clip"])
    for t in np.linspace(0.1, 1.0, 7):
        for joint, child in (("Thigh.L", "Shin.L"), ("Shin.R", "Foot.R"), ("Spine", "Chest")):
            assert angle(a, src, arm, t, joint, child) < 3.0, (t, joint)
        assert abs(angle(a, src, arm, t, "UpperArm.L", "ForeArm.L") - rest_arm) < 1.0, t


def test_mirrored_clips_swap_the_sides_of_an_auto_rigged_character(body_mesh):
    s, arm = hero(body_mesh)
    a = s.animation
    walk = a.generate(arm, "walk", forward="-y", travel=False)["clip"]
    t = 0.45
    left = direction(a, arm, t, "Thigh.L", "Shin.L")
    right = direction(a, arm, t, "Thigh.R", "Shin.R")
    a.set_action(arm, a.mirror_clip(walk, arm))
    flip = np.array([-1.0, 1.0, 1.0])  # (the rig's mirror plane is x = 0, facing -y)
    assert np.allclose(direction(a, arm, t, "Thigh.R", "Shin.R"), left * flip, atol=0.02)
    assert np.allclose(direction(a, arm, t, "Thigh.L", "Shin.L"), right * flip, atol=0.02)
