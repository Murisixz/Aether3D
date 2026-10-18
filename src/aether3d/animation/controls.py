"""Controls, IK/FK switching and high-level posing.

* :func:`add_control` gives any rig (auto-rigged or hand-built, Stage 17's included) a
  control: an IK limb (a target object, an optional pole object and a ``two_bone`` or
  ``fabrik`` constraint, influence 0 so it starts in FK), a gaze (a target and a
  ``look_at`` constraint), an FK control (a joint) or the root (the armature).
* :func:`ik_fk` switches a limb between FK and IK at a time with **matched poses**: to IK,
  the target (and pole) are keyed where the FK pose puts the end joint (and its bend), then
  the constraint's influence is keyed to 1; to FK, the joints are keyed with the rotations
  the IK solution gives, then the influence is keyed to 0. Either way the pose does not
  jump; the report measures it (``mismatch``: the largest joint displacement across the
  switch, metres; ``rotation_mismatch``: the largest joint turn, degrees -- an IK limb
  with a ``bend_axis`` keeps its bones' twist aligned with the bend plane, so an FK pose
  whose bones were twisted otherwise turns that far about the bones).
* :func:`pose_character` turns commands into keys on controls and joints -- ``reach``
  (a hand to a point or object, leaning the spine when out of reach), ``plant_feet``,
  ``look_at``, ``raise_arm``, ``lean``, ``crouch``, ``aim_hips``, ``rotate`` (any joint,
  relative to its rest) -- and reports reachability, IK residuals and the limits hit.

Joint rotation channels hold **absolute** local rotations (the rest rotation included),
as glTF and Stage 17 do; ``rotate`` and the other commands compose with the rest
rotation (``rest * delta``), so a delta of zero is the rest pose.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import rotation as R

if TYPE_CHECKING:
    from .system import AnimationSystem

KINDS = ("ik", "pole", "gaze", "fk", "root")
COMMANDS = ("reach", "plant_feet", "look_at", "raise_arm", "lean", "crouch", "aim_hips", "rotate")


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


def _vec(value: Any, n: int, location: str) -> np.ndarray:
    try:
        v = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err("INVALID_ARGUMENT", f"expected {n} numbers", location) from None
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise _err("INVALID_ARGUMENT", f"expected {n} finite numbers", location)
    return v


def _num(value: Any, location: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _err("INVALID_ARGUMENT", "a finite number", location)
    if not lo <= value <= hi:
        raise _err("INVALID_ARGUMENT", f"a number within {lo:g}..{hi:g}", location)
    return float(value)


# --------------------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------------------- #


def _rigging(system: AnimationSystem, time: float, influences: dict[str, float] | None = None,
             arm_id: str | None = None):  # fmt: skip
    """The rigging pass at ``time`` with some of an armature's constraint influences
    overridden, and the world matrices of every object then."""
    from ..rendering.overlay import RenderOverlay
    from ..rendering.scene_data import world_matrices
    from .rigging import evaluate_rigging

    pose = system.evaluate(time)
    if influences:
        merged = {k: dict(v) for k, v in pose.influences.items()}
        merged.setdefault(arm_id, {}).update(influences)
        pose = dataclasses.replace(pose, influences=merged, extra={})
        result = evaluate_rigging(system, pose)
    else:
        result = system.rigging(time)
    ids = tuple(result.local)
    overlay = RenderOverlay(
        ids=ids, local=np.array([result.local[i] for i in ids]).reshape(-1, 4, 4)
    )
    return result, world_matrices(system.scene, overlay)


def _quat(m: np.ndarray) -> np.ndarray:
    lin = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    q = R.from_matrices(lin[None])[0]
    return q if q[3] >= 0 else -q


def _set_object_world(system, obj, world: np.ndarray, worlds, time: float, key: bool) -> None:
    """Put ``obj`` at ``world`` at ``time``: keys (``key``) or its authored transform."""
    from .constraints import decompose

    parent = obj.parent
    p_world = np.eye(4) if parent is None else worlds[parent.id]
    local = np.linalg.solve(p_world, world)
    p, q, _ = decompose(local)
    q = q if q[3] >= 0 else -q
    if key:
        system.keyframe(obj, "transform.position", time, p.tolist())
        system.keyframe(obj, "transform.rotation", time, q.tolist())
    else:
        obj.position = tuple(p.tolist())
        obj.rotation = tuple(q.tolist())


def _key_joint_rotation(system, arm, rig, name: str, q: np.ndarray, time: float) -> None:
    """Key a joint's absolute local rotation, in the representation its action uses."""
    binding = system.binding(arm)
    clip = None
    if binding is not None and binding.action is not None:
        clip = system._items[binding.action]._definition
    euler_paths = [f'joints["{name}"].rotation_euler.{c}' for c in "xyz"]
    if clip is not None and any(clip.channel(p) is not None for p in euler_paths):
        current = np.array(
            [
                float(clip.channel(p).evaluate(np.array([time]))[0])
                if clip.channel(p) is not None
                else 0.0
                for p in euler_paths
            ]
        )
        e = R.to_euler(q[None])[0]
        e = R.euler_filter(np.stack([current, e]))[1]
        system.keyframe(arm, f'joints["{name}"].rotation_euler', time, e.tolist())
    else:
        system.keyframe(arm, f'joints["{name}"].rotation', time, q.tolist())


def _key_influence(system, arm, name: str, before: float, after: float, time: float,
                   blend: float) -> None:  # fmt: skip
    path = f'constraints["{name}"].influence'
    if blend > 0:
        system.keyframe(arm, path, max(0.0, time - blend), before, interpolation="linear")
        system.keyframe(arm, path, time, after, interpolation="linear")
        return
    eps = 1e-4
    if time - eps >= 0.0:
        system.keyframe(arm, path, time - eps, before, interpolation="constant")
    system.keyframe(arm, path, time, after, interpolation="constant")


# --------------------------------------------------------------------------------------- #
# Adding controls
# --------------------------------------------------------------------------------------- #


def add_control(
    system: AnimationSystem,
    armature: Any,
    name: str,
    kind: str,
    *,
    chain: list[str] | None = None,
    joint: str | None = None,
    constraint: str | None = None,
    pole: bool | list[float] = True,
    target: Any = None,
    group: str = "",
    side: str | None = None,
    influence: float = 0.0,
    space: str = "armature",
) -> dict[str, Any]:
    """Add a control to ``armature``'s rig (one undo step); see the module docstring.
    ``kind``: ``ik`` (``chain``, root to end; ``pole``: true, false or a point),
    ``pole`` (for an existing ``constraint``), ``gaze`` (``joint``), ``fk`` (``joint``)
    or ``root``. Control objects go under the armature (``space="armature"``) or beside it
    (``"world"``: they stay put when the armature moves, for planted feet)."""
    from ..rendering.scene_data import world_matrices
    from .autorig import _unique_name

    if kind not in KINDS:
        raise _err("INVALID_ARGUMENT", f"kind is one of {', '.join(KINDS)}", "kind")
    if space not in ("armature", "world"):
        raise _err("INVALID_ARGUMENT", "space is 'armature' or 'world'", "space")
    arm = system._object(armature, "armature")
    rig = system.rig(arm)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    if rig.control(name) is not None:
        raise _err("DUPLICATE_NAME", f"the rig has a control {name!r}", "name")
    influence = _num(influence, "influence", 0.0, 1.0)
    worlds = world_matrices(system.scene)
    a_world = worlds[arm._id]
    model = rig.rest_model()
    parent = arm if space == "armature" else arm.parent
    raw = rig.to_json()
    made: list[str] = []
    controls = list(raw.get("controls", []))
    constraints = list(raw.get("constraints", []))

    def empty(label: str, world: np.ndarray) -> Any:
        from .constraints import decompose

        p_world = np.eye(4) if parent is None else worlds[parent._id]
        p, q, _ = decompose(np.linalg.solve(p_world, world))
        q = q if q[3] >= 0 else -q
        obj = system.scene.add_object(
            _unique_name(system.scene, f"{arm.name} {label}"), parent=parent,
            position=tuple(p.tolist()), rotation=tuple(q.tolist()),
        )  # fmt: skip
        obj.metadata = {"rig_control": label}
        made.append(obj.name)
        return obj

    with system._op(f"Add control {name!r} to {arm.name!r}"):
        if kind == "ik":
            if not isinstance(chain, list) or len(chain) < 2:
                raise _err("INVALID_ARGUMENT", "an ik control names a chain of 2+ joints", "chain")
            idx = [rig.index(n) for n in chain]
            if any(k is None for k in idx):
                raise _err("JOINT_MISSING", "a chain joint is not in the rig", "chain")
            end = a_world @ model[idx[-1]]
            tgt = empty(name, end) if target is None else system._object(target, "target")
            cname = constraint or f"{name}_ik"
            spec: dict[str, Any] = {
                "name": cname, "type": "ik", "chain": list(chain), "influence": influence,
                "target": {"object": tgt._id}, "end": "target",
                "solver": "two_bone" if len(chain) == 3 else "fabrik",
            }  # fmt: skip
            controls.append({"name": name, "kind": "ik_target", "object": tgt._id,
                             "joint": chain[-1], "constraint": cname, "group": group,
                             **({"side": side} if side else {})})  # fmt: skip
            if pole is not False and len(chain) >= 3:
                mid = (a_world @ model[idx[len(idx) // 2]])[:3, 3]
                if isinstance(pole, (list, tuple, np.ndarray)):
                    where = _vec(pole, 3, "pole")
                else:
                    a = (a_world @ model[idx[0]])[:3, 3]
                    b = end[:3, 3]
                    off = mid - (
                        a + ((mid - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12) * (b - a)
                    )
                    if np.linalg.norm(off) < 1e-6:  # a straight chain: bend toward the front
                        fwd = a_world[:3, :3] @ _forward_vec(rig)
                        off = fwd
                    where = mid + off / np.linalg.norm(off) * float(np.linalg.norm(b - a))
                pm = np.eye(4)
                pm[:3, 3] = where
                pobj = empty(f"{name}_pole", pm)
                spec["pole"] = {"object": pobj._id}
                controls.append({"name": f"{name}_pole", "kind": "pole", "object": pobj._id,
                                 "joint": chain[len(chain) // 2], "constraint": cname,
                                 "group": group, **({"side": side} if side else {})})  # fmt: skip
            constraints.append(spec)
        elif kind == "pole":
            if constraint is None:
                raise _err("INVALID_ARGUMENT", "a pole control names its ik constraint",
                           "constraint")  # fmt: skip
            spec = next((c for c in constraints if c["name"] == constraint), None)
            if spec is None or spec["type"] != "ik":
                raise _err("CONSTRAINT_MISSING", f"no ik constraint {constraint!r}", "constraint")
            where = _vec(pole, 3, "pole") if isinstance(pole, (list, tuple)) else None
            if where is None:
                raise _err("INVALID_ARGUMENT", "pole is the pole's point [x, y, z] (world)", "pole")
            pm = np.eye(4)
            pm[:3, 3] = where
            pobj = empty(name, pm)
            spec["pole"] = {"object": pobj._id}
            controls.append({"name": name, "kind": "pole", "object": pobj._id,
                             "constraint": constraint, "group": group})  # fmt: skip
        elif kind == "gaze":
            k = rig.index(joint) if isinstance(joint, str) else None
            if k is None:
                raise _err("JOINT_MISSING", "a gaze control names its joint", "joint")
            head = a_world @ model[k]
            fwd = a_world[:3, :3] @ _forward_vec(rig)
            gm = np.eye(4)
            gm[:3, 3] = head[:3, 3] + fwd / np.linalg.norm(fwd) * 1.5
            tgt = empty(name, gm) if target is None else system._object(target, "target")
            # Look along the joint's axis that points forward at rest.
            local_fwd = np.linalg.solve(head[:3, :3], fwd)
            axis = _nearest_axis(local_fwd)
            up_axis = _nearest_axis(np.linalg.solve(head[:3, :3], a_world[:3, 2]), exclude=axis)
            cname = constraint or name
            constraints.append({"name": cname, "type": "look_at", "joint": joint,
                                "influence": influence, "target": {"object": tgt._id},
                                "axis": axis, "up": up_axis,
                                "world_up": [0.0, 0.0, 1.0]})  # fmt: skip
            controls.append({"name": name, "kind": "gaze", "object": tgt._id, "joint": joint,
                             "constraint": cname, "group": group or "head"})  # fmt: skip
        elif kind == "fk":
            if not isinstance(joint, str) or rig.index(joint) is None:
                raise _err("JOINT_MISSING", "an fk control names its joint", "joint")
            controls.append({"name": name, "kind": "fk", "joint": joint, "group": group,
                             **({"side": side} if side else {})})  # fmt: skip
        else:
            controls.append({"name": name, "kind": "root", "object": arm._id, "group": "root"})
        raw["controls"] = controls
        raw["constraints"] = constraints
        system.set_rig(arm, raw)
    return {"control": name, "kind": kind, "objects": made}


def _forward_vec(rig) -> np.ndarray:
    axes = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}
    return np.array(axes.get(rig.forward_axis("-y"), (0, -1, 0)), np.float64)


def _nearest_axis(v: np.ndarray, exclude: str | None = None) -> str:
    names = ("+x", "-x", "+y", "-y", "+z", "-z")
    vecs = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
    scores = vecs @ (v / max(np.linalg.norm(v), 1e-12))
    for i in np.argsort(-scores):
        if exclude is None or names[i][1] != exclude[1]:
            return names[i]
    return "+z"


# --------------------------------------------------------------------------------------- #
# IK/FK switching
# --------------------------------------------------------------------------------------- #


def _limb(rig, limb: str) -> tuple[Any, Any, Any]:
    """``(ik constraint, target control, pole control)`` named by a constraint or control."""
    cons = {c.name: c for c in rig.constraints}
    if limb in cons:
        c = cons[limb]
    else:
        ctl = rig.control(limb)
        if ctl is None or ctl.constraint is None or ctl.constraint not in cons:
            raise _err("CONSTRAINT_MISSING", f"no ik limb {limb!r} (a constraint or a control)",
                       "limb")  # fmt: skip
        c = cons[ctl.constraint]
    if c.type != "ik":
        raise _err("INVALID_ARGUMENT", f"{c.name!r} is not an ik constraint", "limb")
    target = next(
        (k for k in rig.controls if k.constraint == c.name and k.kind == "ik_target"), None
    )
    pole = next((k for k in rig.controls if k.constraint == c.name and k.kind == "pole"), None)
    return c, target, pole


def ik_fk(
    system: AnimationSystem,
    armature: Any,
    limb: str,
    mode: str,
    time: float,
    *,
    blend: float = 0.0,
) -> dict[str, Any]:
    """Switch ``limb`` (an ik constraint or one of its controls) to ``mode`` (``"ik"`` or
    ``"fk"``) at ``time`` with a matched pose (keys written; one undo step)."""
    from .time import check_time

    if mode not in ("ik", "fk"):
        raise _err("INVALID_ARGUMENT", "mode is 'ik' or 'fk'", "mode")
    time = check_time(time, "time")
    blend = _num(blend, "blend", 0.0, 60.0)
    arm = system._object(armature, "armature")
    rig = system.rig(arm)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    c, target, pole = _limb(rig, limb)
    p = c.p
    chain = [rig.index(n) for n in p["chain"]]
    scene = system.scene
    if mode == "ik":
        result, worlds = _rigging(system, time, {c.name: 0.0}, arm._id)
        _, model = result.joints[arm._id]
        a_world = worlds[arm._id]
        with system._op(f"Switch {c.name!r} to IK"):
            tgt_ref = p["target"]
            if "object" not in tgt_ref:
                raise _err("INVALID_ARGUMENT", "the limb's target is not an object", "limb")
            tgt = scene._objects.get(tgt_ref["object"]) or scene._by_name.get(tgt_ref["object"])
            end = a_world @ model[chain[-1]]
            if p["end"] == "target":
                world = end
            else:
                world = np.eye(4)
                world[:3, 3] = end[:3, 3]
            _set_object_world(system, tgt, world, worlds, time, True)
            if blend > 0:
                _set_object_world(system, tgt, world, worlds, max(0.0, time - blend), True)
            if p.get("pole") and "object" in p["pole"] and len(chain) >= 3:
                pobj = scene._objects.get(p["pole"]["object"])
                a = (a_world @ model[chain[0]])[:3, 3]
                mid = (a_world @ model[chain[len(chain) // 2]])[:3, 3]
                b = end[:3, 3]
                ab = b - a
                off = mid - (a + ((mid - a) @ ab) / max(ab @ ab, 1e-12) * ab)
                if np.linalg.norm(off) > 1e-6:
                    pm = np.eye(4)
                    pm[:3, 3] = mid + off / np.linalg.norm(off) * float(np.linalg.norm(ab))
                    _set_object_world(system, pobj, pm, worlds, time, True)
            _key_influence(system, arm, c.name, 0.0, 1.0, time, blend)
        before = model
    else:
        result, worlds = _rigging(system, time, {c.name: 1.0}, arm._id)
        _, model = result.joints[arm._id]
        parents = rig.parents
        with system._op(f"Switch {c.name!r} to FK"):
            for k in chain:
                par = np.eye(4) if parents[k] < 0 else model[parents[k]]
                local = np.linalg.solve(par, model[k])
                _key_joint_rotation(system, arm, rig, rig.joints[k].name, _quat(local), time)
            _key_influence(system, arm, c.name, 1.0, 0.0, time, blend)
        before = model
    after_result = system.rigging(time)
    _, after = after_result.joints[arm._id]
    mismatch = float(max(np.linalg.norm(after[k][:3, 3] - before[k][:3, 3]) for k in chain))
    turned = 0.0
    for k in chain:
        qa, qb = _quat(before[k]), _quat(after[k])
        turned = max(turned, math.degrees(2 * math.acos(min(1.0, abs(float(qa @ qb))))))
    residual = after_result.residuals.get(f"{arm.name}/{c.name}")
    return {
        "limb": c.name,
        "mode": mode,
        "time": time,
        "blend": blend,
        "mismatch": mismatch,
        "rotation_mismatch": round(turned, 6),
        **({"residual": residual} if residual is not None else {}),
        "target": None if target is None else target.name,
        "pole": None if pole is None else pole.name,
    }


# --------------------------------------------------------------------------------------- #
# High-level posing
# --------------------------------------------------------------------------------------- #


def _rest_rotation(rig, name: str) -> np.ndarray:
    k = rig.index(name)
    if k is None:
        raise _err("JOINT_MISSING", f"no joint {name!r}", "joint")
    return np.array(rig.joints[k].rotation, np.float64)


def _role_joint(rig, role: str, fallback: list[str]) -> str | None:
    from .generators import find_roles

    k = rig.role_index(role)
    if k is not None:
        return rig.joints[k].name
    roles = find_roles(rig)
    if role in roles:
        return roles[role]
    for n in fallback:
        if rig.index(n) is not None:
            return n
    return None


def _current_local(system, arm, rig, name: str, time: float) -> np.ndarray:
    pose = system.evaluate(time)
    posed = pose.joints.get(arm._id, {}).get(name)
    if posed is not None and posed[1] is not None:
        return np.asarray(posed[1], np.float64)
    return _rest_rotation(rig, name)


def _rotate(system, arm, rig, name: str, euler: np.ndarray, time: float,
            relative_to: str = "rest") -> dict[str, Any]:  # fmt: skip
    base = _rest_rotation(rig, name) if relative_to == "rest" else _current_local(
        system, arm, rig, name, time)  # fmt: skip
    q = R.multiply(base[None], R.from_euler(np.asarray(euler, np.float64)))[0]
    q = q / np.linalg.norm(q)
    _key_joint_rotation(system, arm, rig, name, q if q[3] >= 0 else -q, time)
    return {"joint": name, "euler": [float(v) for v in euler]}


def pose_character(
    system: AnimationSystem, armature: Any, commands: list[Any], *, time: float = 0.0
) -> dict[str, Any]:
    """Apply posing ``commands`` at ``time`` as keys (one undo step); see the module
    docstring. Returns a report per command (reach, residuals, limits)."""
    from .time import check_time

    time = check_time(time, "time")
    arm = system._object(armature, "armature")
    if system.rig(arm) is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    if not isinstance(commands, list) or not 1 <= len(commands) <= 1000:
        raise _err("INVALID_ARGUMENT", "commands is a list of 1..1000", "commands")
    out = []
    with system._op(f"Pose {arm.name!r}"):
        for i, cmd in enumerate(commands):
            loc = f"commands[{i}]"
            if not isinstance(cmd, dict) or cmd.get("do") not in COMMANDS:
                raise _err("INVALID_ARGUMENT", f"a command is {{do: {' | '.join(COMMANDS)}, ...}}",
                           loc)  # fmt: skip
            rig = system.rig(arm)
            out.append(_COMMANDS[cmd["do"]](system, arm, rig, cmd, time, loc))
    final = system.rigging(time)
    residuals = {k.split("/", 1)[1]: round(v, 6) for k, v in final.residuals.items()
                 if k.startswith(f"{arm.name}/")}  # fmt: skip
    return {"armature": arm.name, "time": time, "commands": out, "residuals": residuals,
            "diagnostics": [d.to_dict() for d in final.diagnostics]}  # fmt: skip


def _side(cmd: dict, loc: str) -> str:
    side = cmd.get("side", cmd.get("hand", cmd.get("foot", "L")))
    if side not in ("L", "R"):
        raise _err("INVALID_ARGUMENT", "side is 'L' or 'R'", f"{loc}.side")
    return side


def _point(system, value: Any, loc: str) -> np.ndarray:
    if isinstance(value, dict) and "object" in value:
        obj = system._object(value["object"], f"{loc}.object")
        return obj.world_matrix[:3, 3].copy()
    if isinstance(value, str):
        return system._object(value, loc).world_matrix[:3, 3].copy()
    return _vec(value, 3, loc)


def _ensure_limb(system, arm, rig, kind: str, side: str) -> str:
    """The ik constraint of a limb (adding controls to rigs that have none)."""
    cname = f"{kind}_ik.{side}"
    if any(c.name == cname for c in rig.constraints):
        return cname
    if kind == "arm":
        chain = [_role_joint(rig, f"{r}.{side}", []) for r in ("upper_arm", "lower_arm", "hand")]
    else:
        chain = [_role_joint(rig, f"{r}.{side}", []) for r in ("thigh", "shin", "foot")]
    if any(c is None for c in chain):
        raise _err("ROLE_MISSING", f"the rig has no {side} {kind} chain", "commands")
    label = "hand_ik" if kind == "arm" else "foot_ik"
    add_control(system, arm, f"{label}.{side}", "ik", chain=chain, constraint=cname,
                group=f"{kind}.{side}", side=side)  # fmt: skip
    return cname


def _cmd_reach(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    side = _side(cmd, loc)
    goal = _point(system, cmd.get("target"), f"{loc}.target")
    cname = _ensure_limb(system, arm, rig, "arm", side)
    rig = system.rig(arm)
    c, target, _ = _limb(rig, cname)
    chain = c.p["chain"]
    result, worlds = _rigging(system, time, {cname: 0.0}, arm._id)
    _, model = result.joints[arm._id]
    a_world = worlds[arm._id]
    idx = [rig.index(n) for n in chain]
    shoulder = (a_world @ model[idx[0]])[:3, 3]
    length = sum(
        float(np.linalg.norm(model[idx[i + 1]][:3, 3] - model[idx[i]][:3, 3]))
        for i in range(len(idx) - 1)
    )
    distance = float(np.linalg.norm(goal - shoulder))
    lean = 0.0
    if distance > 0.98 * length and cmd.get("lean", True):
        # Out of reach: lean the chest toward the goal (at most 30 degrees).
        chest = _role_joint(rig, "chest", ["Chest", "Spine"])
        if chest is not None:
            k = rig.index(chest)
            base = (a_world @ model[k])[:3, 3]
            need = distance - 0.95 * length
            up = (a_world @ model[k])[:3, 3] - (a_world @ model[0])[:3, 3]
            lever = max(float(np.linalg.norm(shoulder - base)), 1e-6)
            lean = min(30.0, math.degrees(need / lever))
            lean_dir = goal - base
            lean_dir -= (lean_dir @ a_world[:3, 2]) * a_world[:3, 2]
            del up
            if np.linalg.norm(lean_dir) > 1e-6:
                axis_w = np.cross(a_world[:3, 2], lean_dir / np.linalg.norm(lean_dir))
                m = a_world @ model[k]
                axis_local = np.linalg.solve(m[:3, :3], axis_w)
                axis_local /= np.linalg.norm(axis_local)
                qd = R.axis_angle(axis_local, math.radians(lean))[0]
                base_q = _current_local(system, arm, rig, chest, time)
                q = R.multiply(base_q[None], qd[None])[0]
                _key_joint_rotation(system, arm, rig, chest, q / np.linalg.norm(q), time)
    tgt = system.scene._objects[c.p["target"]["object"]]
    world = np.eye(4)
    hand = a_world @ model[idx[-1]]
    world[:3, :3] = hand[:3, :3]
    world[:3, 3] = goal
    _set_object_world(system, tgt, world, worlds, time, True)
    path = f'constraints["{cname}"].influence'
    system.keyframe(arm, path, time, 1.0, interpolation="constant")
    rr = system.rigging(time)
    return {"do": "reach", "side": side, "distance": round(distance, 6),
            "arm_length": round(length, 6), "reachable": distance <= length,
            "lean_degrees": round(lean, 3),
            "residual": round(rr.residuals.get(f"{arm.name}/{cname}", 0.0), 6)}  # fmt: skip


def _cmd_plant_feet(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    ground = cmd.get("ground")
    out = {}
    for side in ("L", "R"):
        cname = _ensure_limb(system, arm, rig, "leg", side)
        rig = system.rig(arm)
        c, _, _ = _limb(rig, cname)
        result, worlds = _rigging(system, time, {cname: 0.0}, arm._id)
        _, model = result.joints[arm._id]
        a_world = worlds[arm._id]
        foot = a_world @ model[rig.index(c.p["chain"][-1])]
        world = foot.copy()
        if ground is not None:
            g = _num(ground, f"{loc}.ground")
            world[2, 3] = max(world[2, 3], g + (foot[2, 3] - _sole(rig, model, c, a_world)))
        tgt = system.scene._objects[c.p["target"]["object"]]
        _set_object_world(system, tgt, world, worlds, time, True)
        system.keyframe(arm, f'constraints["{cname}"].influence', time, 1.0,
                        interpolation="constant")  # fmt: skip
        out[side] = [round(float(v), 6) for v in world[:3, 3]]
    return {"do": "plant_feet", "feet": out}


def _sole(rig, model, c, a_world) -> float:
    k = rig.index(c.p["chain"][-1])
    tails = rig.tails()
    tip = (a_world @ model[k] @ np.append(tails[k], 1.0))[:3]
    return float(min(tip[2], (a_world @ model[k])[2, 3]))


def _cmd_look_at(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    goal = _point(system, cmd.get("target"), f"{loc}.target")
    gaze = next((k for k in rig.controls if k.kind == "gaze"), None)
    if gaze is None:
        head = _role_joint(rig, "head", ["Head"])
        if head is None:
            raise _err("ROLE_MISSING", "the rig has no head joint", loc)
        add_control(system, arm, "gaze", "gaze", joint=head)
        rig = system.rig(arm)
        gaze = rig.control("gaze")
    _, worlds = _rigging(system, time)
    obj = system.scene._objects[gaze.object]
    m = np.eye(4)
    m[:3, 3] = goal
    _set_object_world(system, obj, m, worlds, time, True)
    system.keyframe(arm, f'constraints["{gaze.constraint}"].influence', time,
                    _num(cmd.get("weight", 1.0), f"{loc}.weight", 0.0, 1.0),
                    interpolation="constant")  # fmt: skip
    return {"do": "look_at", "target": [round(float(v), 6) for v in goal]}


def _cmd_raise_arm(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    side = _side(cmd, loc)
    angle = _num(cmd.get("angle", 90.0), f"{loc}.angle", -180.0, 180.0)
    direction = cmd.get("direction", "forward")
    if direction not in ("forward", "side"):
        raise _err("INVALID_ARGUMENT", "direction is 'forward' or 'side'", f"{loc}.direction")
    name = _role_joint(rig, f"upper_arm.{side}", [f"UpperArm.{side}"])
    if name is None:
        raise _err("ROLE_MISSING", f"the rig has no {side} upper arm", loc)
    return {"do": "raise_arm", **_turn_in_armature(system, arm, rig, name, direction, side, angle,
                                                    time)}  # fmt: skip


def _turn_in_armature(system, arm, rig, name, direction, side, angle, time) -> dict[str, Any]:
    """Turn a joint relative to its rest about an armature axis: ``forward`` raises a
    limb toward the front, ``side`` away from the body."""
    k = rig.index(name)
    model = rig.rest_model()
    fwd = _forward_vec(rig)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    out = -right if side == "L" else right  # the character's left is -right
    # (a hanging limb swings toward the front, or away from the body)
    axis_a = np.cross(-up, fwd) if direction == "forward" else np.cross(-up, out)
    axis_local = np.linalg.solve(model[k][:3, :3], axis_a / np.linalg.norm(axis_a))
    axis_local /= np.linalg.norm(axis_local)
    qd = R.axis_angle(axis_local, math.radians(angle))[0]
    rest = _rest_rotation(rig, name)
    q = R.multiply(rest[None], qd[None])[0]
    _key_joint_rotation(system, arm, rig, name, q / np.linalg.norm(q), time)
    return {"joint": name, "angle": angle, "direction": direction}


def _cmd_lean(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    angle = _num(cmd.get("angle", 15.0), f"{loc}.angle", -90.0, 90.0)
    direction = cmd.get("direction", "forward")
    if direction not in ("forward", "back", "left", "right"):
        raise _err("INVALID_ARGUMENT", "direction is forward, back, left or right",
                   f"{loc}.direction")  # fmt: skip
    joints = [j for j in (_role_joint(rig, "spine", []), _role_joint(rig, "chest", [])) if j]
    if not joints:
        raise _err("ROLE_MISSING", "the rig has no spine", loc)
    fwd = _forward_vec(rig)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    toward = {"forward": fwd, "back": -fwd, "left": -right, "right": right}[direction]
    axis_a = np.cross(up, toward)
    model = rig.rest_model()
    for name in joints:
        k = rig.index(name)
        axis_local = np.linalg.solve(model[k][:3, :3], axis_a)
        axis_local /= np.linalg.norm(axis_local)
        qd = R.axis_angle(axis_local, math.radians(angle / len(joints)))[0]
        q = R.multiply(_rest_rotation(rig, name)[None], qd[None])[0]
        _key_joint_rotation(system, arm, rig, name, q / np.linalg.norm(q), time)
    return {"do": "lean", "angle": angle, "direction": direction, "joints": joints}


def _cmd_crouch(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    depth = _num(cmd.get("depth", 0.25), f"{loc}.depth", 0.0, 10.0)
    hips = _role_joint(rig, "hips", ["Hips"])
    if hips is None:
        raise _err("ROLE_MISSING", "the rig has no hips", loc)
    planted = _cmd_plant_feet(system, arm, rig, {"do": "plant_feet"}, time, loc)
    k = rig.index(hips)
    rig = system.rig(arm)
    parent = rig.parents[k]
    pos = np.array(rig.joints[k].position, np.float64)
    down = np.array([0.0, 0.0, -depth])
    if parent >= 0:  # (the hips' position is in its parent's frame)
        down = np.linalg.solve(rig.rest_model()[parent][:3, :3], down)
    system.keyframe(arm, f'joints["{hips}"].position', time, (pos + down).tolist())
    rr = system.rigging(time)
    res = {k2.split("/", 1)[1]: round(v, 6) for k2, v in rr.residuals.items()
           if "leg_ik" in k2 and k2.startswith(f"{arm.name}/")}  # fmt: skip
    return {"do": "crouch", "depth": depth, "feet": planted["feet"], "residuals": res}


def _cmd_aim_hips(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    yaw = _num(cmd.get("yaw", 0.0), f"{loc}.yaw", -180.0, 180.0)
    hips = _role_joint(rig, "hips", ["Hips"])
    if hips is None:
        raise _err("ROLE_MISSING", "the rig has no hips", loc)
    k = rig.index(hips)
    model = rig.rest_model()
    axis_local = np.linalg.solve(model[k][:3, :3], np.array([0.0, 0.0, 1.0]))
    axis_local /= np.linalg.norm(axis_local)
    qd = R.axis_angle(axis_local, math.radians(yaw))[0]
    q = R.multiply(_rest_rotation(rig, hips)[None], qd[None])[0]
    _key_joint_rotation(system, arm, rig, hips, q / np.linalg.norm(q), time)
    return {"do": "aim_hips", "yaw": yaw, "joint": hips}


def _cmd_rotate(system, arm, rig, cmd, time, loc) -> dict[str, Any]:
    name = cmd.get("joint")
    if not isinstance(name, str) or rig.index(name) is None:
        raise _err("JOINT_MISSING", f"no joint {name!r}", f"{loc}.joint")
    euler = _vec(cmd.get("euler", [0, 0, 0]), 3, f"{loc}.euler")
    relative = cmd.get("relative", "rest")
    if relative not in ("rest", "current"):
        raise _err("INVALID_ARGUMENT", "relative is 'rest' or 'current'", f"{loc}.relative")
    return {"do": "rotate", **_rotate(system, arm, rig, name, euler, time, relative)}


_COMMANDS = {
    "reach": _cmd_reach,
    "plant_feet": _cmd_plant_feet,
    "look_at": _cmd_look_at,
    "raise_arm": _cmd_raise_arm,
    "lean": _cmd_lean,
    "crouch": _cmd_crouch,
    "aim_hips": _cmd_aim_hips,
    "rotate": _cmd_rotate,
}
