"""Automatic rigging: an unrigged character in, a rigged, weighted, controllable one out.

:func:`autorig` measures the meshes (:mod:`.landmarks`), builds a **template** skeleton
on the landmarks, orients every bone, adds limits, IK/FK controls and twist helpers, and
binds every mesh with automatic weights (:mod:`.weights`) -- as **one undo step**, with
a report of every landmark's confidence and every decision. Everything it makes is
ordinary, editable data: joints (:mod:`.rig_edit` edits them), constraints, control
objects, skins.

Templates:

* ``humanoid`` / ``stylized`` -- Hips, Spine, Chest, Neck, Head; Shoulder (clavicle),
  UpperArm, ForeArm (with a ForeArmTwist helper taking half the wrist's twist), Hand;
  Thigh, Shin, Foot, Toe -- sides ``.L``/``.R``; names the motion generators recognize,
  and explicit ``role`` s. ``stylized`` differs in its weight policy (tighter blends).
* ``blocky`` -- Minecraft-style characters: Hips, Body, Head, Arm.L/R, Leg.L/R, pivots at
  the blocks' tops, each block bound rigidly.
* ``quadruped`` -- Pelvis, Spine, Chest, Neck, Head, Tail, and four legs of three joints
  (hip, knee, foot) each.

**Joint frames** (``orientation="bone"``): +Y along each bone toward its child; for the
spine, neck, head and arms +Z leans toward the character's front, for legs toward its
back, for feet and toes upward -- so +X is every limb's bend axis and a positive rotation
about it bends the limb the anatomical way (elbows forward, knees backward). Limits are
swing-twist cones and ellipses and hinges in those frames (applied to joints IK moves,
Stage 17's rule; ``enforce_limits="always"`` applies them to every pose).
``orientation="armature"`` keeps every joint frame aligned with the armature (identity
rest rotations), as Stage 17's hand-built rigs.

**Controls**: the armature is the root; FK controls are the joints themselves; IK targets
for the hands and feet, poles for elbows and knees and a gaze target are empty objects
parented to the armature, feeding ``two_bone`` IK and ``look_at`` constraints whose
influence starts at 0 (FK): generated motions play unchanged, and
``ik_fk`` / ``pose_character`` switch and pose with matched results.

The armature is created facing the character (its local -Y forward, +Z up, origin under
the body on the ground) unless one is given.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import landmarks as LM
from . import rotation as R

if TYPE_CHECKING:
    from .system import AnimationSystem

TEMPLATES = ("humanoid", "stylized", "blocky", "quadruped")
PART_MODES = ("geodesic", "transfer", "rigid", "attach", "skip")
"""How each mesh binds: geodesic weights (the default), weights transferred from the
first geodesic part (clothing), one joint for the whole part as a skin (``rigid``) or as a
rigid attachment to the nearest joint (``attach``), or not at all (``skip``)."""
FORWARD = np.array([0.0, -1.0, 0.0])
BACKWARD = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


# --------------------------------------------------------------------------------------- #
# Templates: joints on landmarks
# --------------------------------------------------------------------------------------- #


def _humanoid_joints(det: LM.Detection, twist: bool) -> list[dict[str, Any]]:
    P = det.point
    J: list[dict[str, Any]] = [
        {"name": "Hips", "at": P("pelvis"), "role": "hips", "lean": FORWARD, "end": "Spine",
         "limits": {"swing": {"cone": 40}, "twist": [-40, 40]}},
        {"name": "Spine", "parent": "Hips", "at": P("spine"), "role": "spine", "lean": FORWARD,
         "end": "Chest", "limits": {"swing": {"cone": 45}, "twist": [-40, 40]}},
        {"name": "Chest", "parent": "Spine", "at": P("chest"), "role": "chest", "lean": FORWARD,
         "end": "Neck", "limits": {"swing": {"cone": 40}, "twist": [-40, 40]}},
        {"name": "Neck", "parent": "Chest", "at": P("neck"), "role": "neck", "lean": FORWARD,
         "end": "Head", "limits": {"swing": {"cone": 50}, "twist": [-60, 60]}},
        {"name": "Head", "parent": "Neck", "at": P("head"), "role": "head", "lean": FORWARD,
         "tip": P("head_top"), "limits": {"swing": {"cone": 45}, "twist": [-70, 70]}},
    ]  # fmt: skip
    for side, sign in (("L", 1.0), ("R", -1.0)):
        out = -1.0 if side == "L" else 1.0  # (a left thigh abducts toward -z swing)
        elbow, wrist = P(f"elbow.{side}"), P(f"wrist.{side}")
        J += [
            {"name": f"Shoulder.{side}", "parent": "Chest", "at": P(f"clavicle.{side}"),
             "role": f"clavicle.{side}", "lean": FORWARD, "end": f"UpperArm.{side}",
             "limits": {"swing": {"cone": 25}, "twist": [-10, 10]}},
            {"name": f"UpperArm.{side}", "parent": f"Shoulder.{side}", "at": P(f"shoulder.{side}"),
             "role": f"upper_arm.{side}", "lean": FORWARD, "end": f"ForeArm.{side}",
             "limits": {"swing": {"cone": 130}, "twist": [-90, 90]}},
            {"name": f"ForeArm.{side}", "parent": f"UpperArm.{side}", "at": elbow,
             "role": f"lower_arm.{side}", "lean": FORWARD,
             "end": f"ForeArmTwist.{side}" if twist else f"Hand.{side}",
             "limits": {"hinge": {"axis": "+x", "min": 0, "max": 150}}},
        ]  # fmt: skip
        if twist:
            J.append(
                {
                    "name": f"ForeArmTwist.{side}",
                    "parent": f"ForeArm.{side}",
                    "at": 0.5 * (elbow + wrist),
                    "role": f"forearm_twist.{side}",
                    "lean": FORWARD,
                    "tip": wrist,
                }  # fmt: skip
            )
        J += [
            {"name": f"Hand.{side}", "parent": f"ForeArm.{side}", "at": wrist,
             "role": f"hand.{side}", "lean": FORWARD, "tip": P(f"hand_tip.{side}"),
             "limits": {"swing": {"x": [-70, 80], "z": [-30, 30]}, "twist": [-25, 25]}},
            {"name": f"Thigh.{side}", "parent": "Hips", "at": P(f"hip.{side}"),
             "role": f"thigh.{side}", "lean": BACKWARD, "end": f"Shin.{side}",
             "limits": {"swing": {"x": [-120, 35],
                                  "z": [-45, 25] if out < 0 else [-25, 45]},
                        "twist": [-40, 40]}},
            {"name": f"Shin.{side}", "parent": f"Thigh.{side}", "at": P(f"knee.{side}"),
             "role": f"shin.{side}", "lean": BACKWARD, "end": f"Foot.{side}",
             "limits": {"hinge": {"axis": "+x", "min": 0, "max": 150}}},
            {"name": f"Foot.{side}", "parent": f"Shin.{side}", "at": P(f"ankle.{side}"),
             "role": f"foot.{side}", "lean": UP, "end": f"Toe.{side}",
             "limits": {"swing": {"x": [-50, 30], "z": [-25, 25]}, "twist": [-15, 15]}},
            {"name": f"Toe.{side}", "parent": f"Foot.{side}", "at": P(f"ball.{side}"),
             "role": f"toe.{side}", "lean": UP, "tip": P(f"toe_tip.{side}"),
             "limits": {"hinge": {"axis": "+x", "min": -30, "max": 60}}},
        ]  # fmt: skip
        del sign
    return J


def _blocky_joints(det: LM.Detection) -> list[dict[str, Any]]:
    P = det.point
    J = [
        {"name": "Hips", "at": P("pelvis"), "role": "hips", "lean": FORWARD,
         "tip": P("pelvis") + [0, 0, 0.05 * det.height]},
        {"name": "Body", "parent": "Hips", "at": P("pelvis"), "role": "spine", "lean": FORWARD,
         "tip": P("neck")},
        {"name": "Head", "parent": "Body", "at": P("head"), "role": "head", "lean": FORWARD,
         "tip": P("head_top")},
    ]  # fmt: skip
    for side in ("L", "R"):
        J += [
            {"name": f"Arm.{side}", "parent": "Body", "at": P(f"shoulder.{side}"),
             "role": f"upper_arm.{side}", "lean": FORWARD, "tip": P(f"hand_tip.{side}")},
            {"name": f"Leg.{side}", "parent": "Hips", "at": P(f"hip.{side}"),
             "role": f"thigh.{side}", "lean": BACKWARD,
             "tip": np.array([P(f"hip.{side}")[0], 0.0, 0.0])},
        ]  # fmt: skip
    return J


def _quadruped_joints(det: LM.Detection) -> list[dict[str, Any]]:
    P = det.point
    lm = det.landmarks
    J = [
        {"name": "Pelvis", "at": P("pelvis"), "role": "hips", "lean": UP, "end": "Spine"},
        {"name": "Spine", "parent": "Pelvis", "at": P("spine"), "role": "spine", "lean": UP,
         "end": "Chest"},
        {"name": "Chest", "parent": "Spine", "at": P("chest"), "role": "chest", "lean": UP,
         "end": "Neck"},
        {"name": "Neck", "parent": "Chest", "at": P("neck"), "role": "neck", "lean": UP,
         "end": "Head"},
        {"name": "Head", "parent": "Neck", "at": P("head"), "role": "head", "lean": UP,
         "tip": P("head_top")},
    ]  # fmt: skip
    if "tail_tip" in lm:
        root = P("pelvis") + 0.3 * (P("tail_tip") - P("pelvis"))
        J.append({"name": "Tail", "parent": "Pelvis", "at": root, "role": "tail", "lean": UP,
                  "tip": P("tail_tip")})  # fmt: skip
    for end, parent in (("front", "Chest"), ("back", "Pelvis")):
        cap = end.capitalize()
        for side in ("L", "R"):
            if f"{end}_hip.{side}" not in lm:
                continue
            foot = P(f"{end}_foot.{side}")
            J += [
                {"name": f"{cap}Leg.{side}", "parent": parent, "at": P(f"{end}_hip.{side}"),
                 "role": f"{end}_thigh.{side}", "lean": FORWARD, "end": f"{cap}Shin.{side}"},
                {"name": f"{cap}Shin.{side}", "parent": f"{cap}Leg.{side}",
                 "at": P(f"{end}_knee.{side}"), "role": f"{end}_shin.{side}", "lean": FORWARD,
                 "end": f"{cap}Foot.{side}"},
                {"name": f"{cap}Foot.{side}", "parent": f"{cap}Shin.{side}", "at": foot,
                 "role": f"{end}_foot.{side}", "lean": UP,
                 "tip": foot + [0, -0.05 * det.height, 0]},
            ]  # fmt: skip
    return J


# --------------------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------------------- #


def _frames(joints: list[dict[str, Any]], orientation: str) -> dict[str, np.ndarray]:
    """World (canonical) frames: columns X, Y, Z of each joint."""
    by = {j["name"]: j for j in joints}
    out = {}
    for j in joints:
        if orientation == "armature":
            out[j["name"]] = np.eye(3)
            continue
        end = by[j["end"]]["at"] if "end" in j else j.get("tip")
        if end is None or np.linalg.norm(end - j["at"]) < 1e-9:
            out[j["name"]] = np.eye(3)
            continue
        y = (end - j["at"]) / np.linalg.norm(end - j["at"])
        lean = j["lean"]
        for hint in (lean, UP, FORWARD):
            z = hint - (hint @ y) * y
            if np.linalg.norm(z) > 0.3:
                break
        z = z / np.linalg.norm(z)
        x = np.cross(y, z)
        out[j["name"]] = np.stack([x, y, z], axis=1)
    return out


# --------------------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------------------- #


def autorig(
    system: AnimationSystem,
    meshes: Any,
    *,
    armature: Any = None,
    name: str | None = None,
    template: str = "humanoid",
    landmarks: dict[str, Any] | None = None,
    up: Any = "+z",
    forward: Any = "auto",
    resolution: int = 128,
    symmetry: str = "auto",
    orientation: str = "bone",
    twist: bool = True,
    controls: bool = True,
    bind: bool = True,
    parts: dict[str, str] | None = None,
    weights: dict[str, Any] | None = None,
    skinning: str = "linear",
    enforce_limits: str = "ik",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rig the given meshes (see the module docstring); one undo step. Returns a report:
    landmarks with confidences, the joints, controls and constraints made, the weights
    of every part, diagnostics and timings. ``dry_run``: detect and plan only."""
    if template not in TEMPLATES:
        raise _err("INVALID_ARGUMENT", f"template is one of {', '.join(TEMPLATES)}", "template")
    if orientation not in ("bone", "armature"):
        raise _err("INVALID_ARGUMENT", "orientation is 'bone' or 'armature'", "orientation")
    for key, value in (("twist", twist), ("controls", controls), ("bind", bind)):
        if not isinstance(value, bool):
            raise _err("INVALID_ARGUMENT", f"{key} is true or false", key)
    t0 = time.perf_counter()
    pos, tris, objs = LM.gather(system, meshes)
    detect_template = "humanoid" if template == "stylized" else template
    det = LM.detect(pos, tris, template=detect_template, up=up, forward=forward,
                    resolution=resolution, symmetry=symmetry, overrides=landmarks)  # fmt: skip
    t1 = time.perf_counter()
    if template in ("humanoid", "stylized"):
        joints = _humanoid_joints(det, twist)
    elif template == "blocky":
        joints = _blocky_joints(det)
    else:
        joints = _quadruped_joints(det)
    frames = _frames(joints, orientation)
    report = det.report()
    report["template"] = template
    report["timings"] = {"detect_s": round(t1 - t0, 4)}
    report["joints"] = [
        {"name": j["name"], "parent": j.get("parent"), "role": j.get("role"),
         "position": [round(float(v), 6) for v in det.world(j["at"])]}
        for j in joints
    ]  # fmt: skip
    part_modes = _part_modes(system, objs, parts, template)
    report["parts"] = {o.name: m for o, m in part_modes}
    if dry_run:
        report["dry_run"] = True
        return report
    with system._op(f"Auto-rig {objs[0].name!r}"):
        arm = _armature(system, armature, name or f"{objs[0].name} Rig", det)
        raw, made = _rig_json(system, arm, joints, frames, det, template, twist, controls,
                              enforce_limits)  # fmt: skip
        system.set_rig(arm, raw)
        report["armature"] = arm.name
        report["controls"] = made
        rig = system.rig(arm)
        report["constraints"] = [c.name for c in rig.constraints]
        t2 = time.perf_counter()
        report["weights"] = {}
        if bind:
            opts = dict(weights or {})
            default = {"stylized": "stylized", "blocky": "rigid"}.get(template, "organic")
            policy = opts.pop("policy", default)
            if template == "blocky":
                opts.setdefault("rigid_parts", "all")
            body = [o for o, m in part_modes if m == "geodesic"]
            for obj, mode in part_modes:
                if mode == "skip":
                    continue
                if mode == "geodesic":
                    rep = system.generate_weights(obj, arm, policy=policy, context=body,
                                                  skinning=skinning, **opts)  # fmt: skip
                elif mode == "rigid":  # the whole part follows its dominant joint
                    rep = system.generate_weights(obj, arm, policy="rigid", rigid_parts="all",
                                                  context=body, **opts)  # fmt: skip
                    system.edit_weights(obj, [{"op": "rigid", "per": "selection"}])
                    sk = system._owned[("skin", obj._id)]._definition
                    rep["joint"] = rig.joints[
                        rig.index_of_id(sk.joints[int(sk.indices[0, 0])])
                    ].name
                elif mode == "transfer":
                    if not body:
                        raise _err("INVALID_ARGUMENT", "transfer needs a geodesic body part",
                                   "parts")  # fmt: skip
                    rep = system.transfer_weights(obj, body[0], skinning=skinning)
                else:  # attach: the joint nearest the part's centre
                    rep = _attach_part(system, obj, arm)
                report["weights"][obj.name] = _brief(rep, mode)
        report["timings"]["bind_s"] = round(time.perf_counter() - t2, 4)
    report["timings"]["total_s"] = round(time.perf_counter() - t0, 4)
    return report


def _brief(rep: dict[str, Any], mode: str) -> dict[str, Any]:
    keep = ("method", "policy", "technique", "voxel_size", "unreached_vertices", "rigid_parts",
            "joints_used", "influences", "far_vertices", "unmatched_parts", "confidence",
            "distance", "joint", "timings")  # fmt: skip
    return {"mode": mode, **{k: rep[k] for k in keep if k in rep}}


def _part_modes(system, objs, parts, template) -> list[tuple[Any, str]]:
    """Each mesh's binding: ``parts`` by name, else ``geodesic`` (all parts voxelized
    together, so clothing over a body takes the body's distances)."""
    parts = dict(parts or {})
    out = []
    for obj in objs:
        mode = parts.pop(obj.name, parts.pop(obj._id, "geodesic"))
        if mode not in PART_MODES:
            raise _err("INVALID_ARGUMENT", f"a part's mode is one of {', '.join(PART_MODES)}",
                       f"parts.{obj.name}")  # fmt: skip
        out.append((obj, mode))
    if parts:
        raise _err("INVALID_ARGUMENT", f"parts names meshes not rigged: {sorted(parts)}", "parts")
    del template
    return out


def _attach_part(system, obj, arm) -> dict[str, Any]:
    from ..rendering.scene_data import world_matrices

    rig = system.rig(arm)
    worlds = world_matrices(system.scene)
    c = np.asarray(obj.mesh.positions, np.float64).mean(axis=0)
    c = (worlds[obj._id] @ np.append(c, 1.0))[:3]
    model = worlds[arm._id] @ rig.rest_model()
    tails = rig.tails()
    best, joint = np.inf, rig.joints[0].name
    for k, j in enumerate(rig.joints):
        if not j.deform:
            continue
        a = model[k][:3, 3]
        b = (model[k] @ np.append(tails[k], 1.0))[:3]
        t = np.clip((c - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-12), 0, 1)
        d = float(np.linalg.norm(c - (a + t * (b - a))))
        if d < best:
            best, joint = d, j.name
    system.attach(obj, arm, joint)
    return {"joint": joint, "method": "attachment"}


def _armature(system, armature, name, det) -> Any:
    if armature is not None:
        arm = system._object(armature, "armature")
        if system.rig(arm) is not None:
            raise _err("DUPLICATE_BINDING", f"{arm.name!r} already has a rig", "armature",
                       hint="pass another armature or none (one is created)")  # fmt: skip
        return arm
    frame = det.rotation.T  # columns: the canonical axes in world space
    q = R.from_matrices(frame[None])[0]
    return system.scene.add_object(
        _unique_name(system.scene, name), position=tuple(det.origin.tolist()),
        rotation=tuple(q.tolist()),
    )  # fmt: skip


def _unique_name(scene: Any, base: str) -> str:
    name, k = base, 1
    while scene._by_name.get(name) is not None:
        k += 1
        name = f"{base}.{k:03d}"
    return name


def _rig_json(system, arm, joints, frames, det, template, twist, controls, enforce) -> tuple:
    from ..rendering.scene_data import world_matrices
    from .constraints import decompose

    a_world = world_matrices(system.scene)[arm._id]
    a_inv = np.linalg.inv(a_world)
    model = {}
    for j in joints:
        m = np.eye(4)
        m[:3, :3] = det.rotation.T @ frames[j["name"]]  # canonical -> world
        m[:3, 3] = det.world(j["at"])
        model[j["name"]] = a_inv @ m  # armature space
    out = []
    for j in joints:
        parent = j.get("parent")
        local = (
            model[j["name"]] if parent is None else np.linalg.solve(model[parent], model[j["name"]])
        )
        p, q, s = decompose(local)
        if q[3] < 0:
            q = -q
        rec: dict[str, Any] = {"name": j["name"], "parent": parent, "position": p.tolist(),
                               "rotation": q.tolist(), "role": j.get("role")}  # fmt: skip
        if not np.allclose(s, 1.0, atol=1e-9):
            rec["scale"] = s.tolist()
        end = j.get("tip")
        if end is not None:
            tip_arm = a_inv @ np.append(det.world(end), 1.0)
            rec["tail"] = np.linalg.solve(model[j["name"]], tip_arm)[:3].tolist()
        if "limits" in j:
            rec["limits"] = j["limits"]
        if j.get("deform") is False:
            rec["deform"] = False
        out.append(rec)
    by_name = {j["name"]: j for j in out}
    # The forearm's bone ends at its twist helper (their bones split the forearm).
    constraints: list[dict[str, Any]] = []
    made: list[dict[str, Any]] = []
    ctrl: list[dict[str, Any]] = []
    if template in ("humanoid", "stylized"):
        for side in ("L", "R"):
            if twist:
                tw = model[f"ForeArmTwist.{side}"][:3, 3]
                fa = model[f"ForeArm.{side}"]
                by_name[f"ForeArm.{side}"]["tail"] = np.linalg.solve(fa, np.append(tw, 1.0))[
                    :3
                ].tolist()
                constraints.append({"name": f"forearm_twist.{side}", "type": "twist",
                                    "joint": f"ForeArmTwist.{side}",
                                    "target": {"joint": f"Hand.{side}"}, "axis": "+y",
                                    "factor": 0.5})  # fmt: skip
        if controls:
            ctrl, made, cons = _humanoid_controls(system, arm, model, det)
            constraints += cons
    elif template in ("blocky", "quadruped") and controls:
        ctrl = [{"name": f"{j['name'].lower()}_fk", "kind": "fk", "joint": j["name"],
                 "group": "body"} for j in out]  # fmt: skip
    info = {
        "forward": "-y",
        "up": "+z",
        "source": "autorig",
        "style": template,
        "description": f"auto-rigged {template} ({det.info.get('pose', 'rest pose')}), height "
        f"{det.height:.3f}",
    }
    if "pose" in det.info:
        info["pose"] = det.info["pose"]
    lms = {name: (a_inv @ np.append(det.world(l_.position), 1.0))[:3].tolist()
           for name, l_ in det.landmarks.items()}  # fmt: skip
    raw = {
        "joints": out,
        "constraints": constraints,
        "controls": ctrl,
        "landmarks": lms,
        "template": template,
        "info": info,
        "enforce_limits": enforce,
    }
    return raw, made


def _humanoid_controls(system, arm, model, det) -> tuple[list, list, list]:
    """IK targets, poles and the gaze target: empties under the armature, at rest where
    the joints are (IK and look-at influence 0: FK until switched)."""
    from .constraints import decompose

    H = det.height
    ctrl: list[dict[str, Any]] = [
        {"name": "root", "kind": "root", "object": arm._id, "group": "root", "shape": "circle",
         "size": 0.3 * H},
        {"name": "pelvis", "kind": "fk", "joint": "Hips", "group": "spine", "shape": "cube"},
        {"name": "chest", "kind": "fk", "joint": "Chest", "group": "spine", "shape": "circle"},
        {"name": "head", "kind": "fk", "joint": "Head", "group": "head", "shape": "circle"},
    ]  # fmt: skip
    made = []
    cons = []

    def empty(label: str, m: np.ndarray, shape: str, size: float) -> Any:
        p, q, _ = decompose(m)
        q = q if q[3] >= 0 else -q
        obj = system.scene.add_object(_unique_name(system.scene, f"{arm.name} {label}"),
                                      parent=arm, position=tuple(p.tolist()),
                                      rotation=tuple(q.tolist()))  # fmt: skip
        obj.metadata = {"rig_control": label, "shape": shape, "size": size}
        made.append(obj.name)
        return obj

    for side in ("L", "R"):
        group_a, group_l = f"arm.{side}", f"leg.{side}"
        hand = model[f"Hand.{side}"]
        elbow = model[f"ForeArm.{side}"][:3, 3]
        arm_len = float(np.linalg.norm(elbow - model[f"UpperArm.{side}"][:3, 3]))
        t_hand = empty(f"hand_ik.{side}", hand, "cube", 0.06 * H)
        pole_m = np.eye(4)
        pole_m[:3, 3] = elbow + BACKWARD * 1.5 * arm_len
        t_elbow = empty(f"elbow_pole.{side}", pole_m, "diamond", 0.03 * H)
        foot = model[f"Foot.{side}"]
        knee = model[f"Shin.{side}"][:3, 3]
        leg_len = float(np.linalg.norm(knee - model[f"Thigh.{side}"][:3, 3]))
        t_foot = empty(f"foot_ik.{side}", foot, "cube", 0.07 * H)
        kpole = np.eye(4)
        kpole[:3, 3] = knee + FORWARD * 1.5 * leg_len
        t_knee = empty(f"knee_pole.{side}", kpole, "diamond", 0.03 * H)
        cons += [
            {"name": f"arm_ik.{side}", "type": "ik", "influence": 0.0,
             "chain": [f"UpperArm.{side}", f"ForeArm.{side}", f"Hand.{side}"],
             "target": {"object": t_hand._id}, "pole": {"object": t_elbow._id},
             "solver": "two_bone", "end": "target"},
            {"name": f"leg_ik.{side}", "type": "ik", "influence": 0.0,
             "chain": [f"Thigh.{side}", f"Shin.{side}", f"Foot.{side}"],
             "target": {"object": t_foot._id}, "pole": {"object": t_knee._id},
             "solver": "two_bone", "end": "target"},
        ]  # fmt: skip
        mirror = "R" if side == "L" else "L"
        ctrl += [
            {"name": f"hand_ik.{side}", "kind": "ik_target", "object": t_hand._id,
             "joint": f"Hand.{side}", "constraint": f"arm_ik.{side}", "group": group_a,
             "side": side, "shape": "cube", "size": 0.06 * H, "mirror": f"hand_ik.{mirror}"},
            {"name": f"elbow_pole.{side}", "kind": "pole", "object": t_elbow._id,
             "joint": f"ForeArm.{side}", "constraint": f"arm_ik.{side}", "group": group_a,
             "side": side, "shape": "diamond", "size": 0.03 * H,
             "mirror": f"elbow_pole.{mirror}"},
            {"name": f"foot_ik.{side}", "kind": "ik_target", "object": t_foot._id,
             "joint": f"Foot.{side}", "constraint": f"leg_ik.{side}", "group": group_l,
             "side": side, "shape": "cube", "size": 0.07 * H, "mirror": f"foot_ik.{mirror}"},
            {"name": f"knee_pole.{side}", "kind": "pole", "object": t_knee._id,
             "joint": f"Shin.{side}", "constraint": f"leg_ik.{side}", "group": group_l,
             "side": side, "shape": "diamond", "size": 0.03 * H, "mirror": f"knee_pole.{mirror}"},
        ]  # fmt: skip
        for jn, grp in ((f"UpperArm.{side}", group_a), (f"ForeArm.{side}", group_a),
                        (f"Hand.{side}", group_a), (f"Thigh.{side}", group_l),
                        (f"Shin.{side}", group_l), (f"Foot.{side}", group_l)):  # fmt: skip
            ctrl.append({"name": f"{jn.split('.')[0].lower()}_fk.{side}", "kind": "fk",
                         "joint": jn, "group": grp, "side": side})  # fmt: skip
    head = model["Head"]
    gaze_m = np.eye(4)
    gaze_m[:3, 3] = head[:3, 3] + FORWARD * 1.2 * H
    t_gaze = empty("gaze", gaze_m, "sphere", 0.05 * H)
    cons.append({"name": "gaze", "type": "look_at", "joint": "Head", "influence": 0.0,
                 "target": {"object": t_gaze._id}, "axis": "+z", "up": "+y",
                 "world_up": [0.0, 0.0, 1.0]})  # fmt: skip
    ctrl.append({"name": "gaze", "kind": "gaze", "object": t_gaze._id, "joint": "Head",
                 "constraint": "gaze", "group": "head", "shape": "sphere",
                 "size": 0.05 * H})  # fmt: skip
    return ctrl, made, cons
