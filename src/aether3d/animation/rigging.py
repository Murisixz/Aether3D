"""The rigging pass: rigs, constraints, attachments and skins after the channels.

Given the channels' pose at a time (:class:`~.evaluate.AnimationPose`), this computes
what the scene shows: every object's local matrix (animated, attached, constrained),
deformed meshes and joint matrices. Order (the documented contract):

1. objects' local matrices from their channels; world matrices from them;
2. **units** in dependency order -- a rig (its joints from their channels, forward
   kinematics, its joint constraints in order, then its attachments) or an object's
   constraint list -- where a unit runs after every unit that moves one of its targets
   (or a target's ancestor); a cycle is reported and its units skipped;
3. **correctives** read the final joint rotations and add to morph target weights;
4. morph targets, then skins (linear, dual quaternion or blended) deform their meshes with
   the final matrices.

Nothing here changes the scene; the result only feeds the render overlay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from . import ik
from . import rotation as R
from .constraints import Constraint, apply_constraint, blend_matrices, decompose
from .rig import Rig, forward_kinematics, local_matrices, skin_dual_quaternion, skin_linear

if TYPE_CHECKING:
    from .evaluate import AnimationPose
    from .system import AnimationSystem


@dataclass
class RiggingResult:
    local: dict[str, np.ndarray] = field(default_factory=dict)  # object id -> local matrix
    meshes: dict[str, Any] = field(default_factory=dict)  # object id -> deformed Mesh
    joints: dict[str, tuple[Rig, np.ndarray]] = field(
        default_factory=dict
    )  # armature -> (rig, model)
    residuals: dict[str, float] = field(default_factory=dict)  # "Armature/constraint" -> distance
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # "Armature/corrective" -> (driver angle in degrees, weight added to the morph target)
    correctives: dict[str, tuple[float, float]] = field(default_factory=dict)
    # mesh object id -> {morph target: weight the correctives add} (what glTF bakes)
    morph_extra: dict[str, dict[str, float]] = field(default_factory=dict)


def _diag(code: str, message: str, location: str) -> Diagnostic:
    return Diagnostic(code, WARNING, message, location)


class _Worlds:
    """World matrices of the scene under changing local overrides (updated per subtree,
    with the scene hierarchy's arithmetic)."""

    def __init__(self, scene: Any, local: dict[str, np.ndarray]) -> None:
        from ..rendering.overlay import RenderOverlay
        from ..rendering.scene_data import world_matrices

        self.scene = scene
        self.local = local
        ids = tuple(local)
        overlay = RenderOverlay(ids=ids, local=np.array([local[i] for i in ids]).reshape(-1, 4, 4))
        self.world = {k: v.copy() for k, v in world_matrices(scene, overlay).items()}
        self._order: dict[str, int] | None = None

    def order(self) -> dict[str, int]:
        """Each object's position in the scene's order (cached per scene revision)."""
        if self._order is None:
            self._order = _cached(self.scene, "scene_order", _scene_order)
        return self._order

    def set_local(self, obj: Any, matrix: np.ndarray) -> None:
        from ..rendering.scene_data import _mat4_mul

        self.local[obj._id] = matrix
        stack = [obj]
        while stack:
            o = stack.pop()
            parent = o._parent
            loc = self.local.get(o._id)
            if loc is None:
                loc = o._transform.matrix
            self.world[o._id] = (
                loc.copy() if parent is None else _mat4_mul(self.world[parent._id], loc)
            )
            stack.extend(reversed(o._children))


def evaluate_rigging(
    system: AnimationSystem, pose: AnimationPose, *, deform: bool = True
) -> RiggingResult:
    """The rigging pass for ``pose`` (see the module docstring); ``deform=False`` stops
    before the meshes (correctives, morph targets, skins): joints, objects and residuals
    only, for measuring motion."""
    from .system import canonical_quats

    scene = system.scene
    result = RiggingResult()
    from ..rendering.scene_data import _compose_locals

    if pose.ids:
        mats = _compose_locals(pose.position, canonical_quats(pose.rotation), pose.scale)
        result.local = {oid: mats[k] for k, oid in enumerate(pose.ids)}
    rigs = system._rigs()
    object_constraints = system._object_constraints()
    skins = system._skins()
    morphs = system._morphs() if hasattr(system, "_morphs") else {}
    if not rigs and not object_constraints and not skins and not morphs:
        return result
    worlds = _Worlds(scene, dict(result.local))
    authored: dict[str, np.ndarray] | None = None

    # Units: ("rig", armature id) and ("object", owner id); what each one moves.
    moves: dict[tuple[str, str], set[str]] = {}
    reads: dict[tuple[str, str], set[str]] = {}
    for arm, (_item, rig) in rigs.items():
        key = ("rig", arm)
        moves[key] = {arm_child for arm_child, _ in rig.attachments}
        reads[key] = _target_objects(rig.constraints, scene) | {arm}
    for owner, (_item, constraints) in object_constraints.items():
        key = ("object", owner)
        moves[key] = {owner}
        reads[key] = _target_objects(constraints, scene)
    order, cyclic = _unit_order(scene, moves, reads)
    for unit in cyclic:
        result.diagnostics.append(
            _diag(
                "CONSTRAINT_CYCLE",
                f"{unit[0]} {unit[1]} depends on itself through constraint targets; it is skipped",
                f"animation.{unit[0]}s[{unit[1]}]",
            )
        )
    for kind, oid in order:
        if kind == "rig":
            item, rig = rigs[oid]
            if authored is None and rig.attachments:
                authored = _authored_worlds(system)
            _run_rig(system, pose, rig, oid, worlds, authored, result)
        else:
            _run_object_constraints(system, pose, oid, object_constraints[oid][1], worlds, result)
    result.local = worlds.local
    if not deform:
        return result
    extra = _corrective_weights(system, rigs, morphs, result)
    result.morph_extra = extra
    # Deformation: morph targets then skins, with the final matrices.
    for oid, skin in skins.items():
        _run_skin(system, pose, oid, skin, worlds, result, extra.get(oid))
    for oid in morphs:
        if oid not in result.meshes:
            mesh = _morphed(system, pose, oid, extra.get(oid))
            if mesh is not None:
                result.meshes[oid] = mesh
    return result


def _corrective_weights(
    system: AnimationSystem, rigs: dict, morphs: dict, result: RiggingResult
) -> dict[str, dict[str, float]]:
    """Mesh object id -> {morph target: weight to add}, from every rig's correctives
    (their driver joints' final rotations)."""
    out: dict[str, dict[str, float]] = {}
    for arm, (_item, rig) in rigs.items():
        if not rig.correctives:
            continue
        found = result.joints.get(arm)
        if found is None:
            continue
        _, model = found
        rest_rot = rig.rest_trs()[1]
        parents = rig.parents
        arm_obj = system.scene._objects.get(arm)
        arm_name = arm if arm_obj is None else arm_obj.name
        for c in rig.correctives:
            k = rig.index(c.joint)
            morph = morphs.get(c.mesh)
            if k is None or morph is None or c.target not in morph.names:
                result.diagnostics.append(
                    _diag(
                        "CORRECTIVE_TARGET_MISSING",
                        f"corrective {c.name!r} of {arm_name!r} drives a morph target its "
                        "mesh does not have",
                        f"objects[{arm_name}].correctives[{c.name}]",
                    )
                )
                continue
            parent_m = np.eye(4) if parents[k] < 0 else model[parents[k]]
            _, q, _ = decompose(np.linalg.solve(parent_m, model[k]))
            value = c.value(q, rest_rot[k])
            weight = c.weight_for(value)
            result.correctives[f"{arm_name}/{c.name}"] = (value, weight)
            if weight != 0.0:
                bucket = out.setdefault(c.mesh, {})
                bucket[c.target] = bucket.get(c.target, 0.0) + weight
    return out


def _cached(scene: Any, name: str, make: Any) -> Any:
    """``make(scene)``, cached on the animation program per scene revision (read only)."""
    from .evaluate import scene_key

    key = scene_key(scene)
    cache = scene.animation._program().rest_cache
    cached = cache.get(name)
    if key is not None and cached is not None and cached[0] == key:
        return cached[1]
    value = make(scene)
    if key is not None:
        cache[name] = (key, value)
    return value


def _scene_order(scene: Any) -> dict[str, int]:
    return {o._id: k for k, o in enumerate(scene)}


def _authored_worlds(system: AnimationSystem) -> dict[str, np.ndarray]:
    """World matrices of the authored scene (the rest of attachments)."""
    from ..rendering.scene_data import world_matrices

    return _cached(system.scene, "authored_worlds", world_matrices)


def _target_objects(constraints: tuple[Constraint, ...], scene: Any) -> set[str]:
    out = set()
    for c in constraints:
        p = c.p
        for key in ("target", "pole"):
            t = p.get(key)
            if t and "object" in t:
                obj = scene._objects.get(t["object"]) or scene._by_name.get(t["object"])
                if obj is not None:
                    out.add(obj._id)
    return out


def _unit_order(scene: Any, moves: dict, reads: dict) -> tuple[list, list]:
    """Units in dependency order (a unit after the units that move what it reads or an
    ancestor of it), and the units on cycles."""
    mover: dict[str, list] = {}
    for unit, objs in moves.items():
        for o in objs:
            mover.setdefault(o, []).append(unit)
    deps: dict[Any, set] = {u: set() for u in moves}
    for unit, objs in reads.items():
        for oid in objs:
            obj = scene._objects.get(oid)
            chain = [obj] if obj is not None else []
            if obj is not None:
                chain.extend(obj.iter_ancestors())
            for o in chain:
                for other in mover.get(o._id, ()):
                    if other != unit:
                        deps[unit].add(other)
    order = []
    done: set = set()
    remaining = sorted(deps, key=lambda u: (u[0] != "rig", u[1]))
    while remaining:
        progressed = False
        rest = []
        for u in remaining:
            if deps[u] <= done:
                order.append(u)
                done.add(u)
                progressed = True
            else:
                rest.append(u)
        if not progressed:
            return order, rest
        remaining = rest
    return order, []


def _target_world(
    target: dict[str, Any] | None, worlds: _Worlds, result: RiggingResult, scene: Any,
    rig_model: tuple[Rig, np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:  # fmt: skip
    """The world matrix of a constraint target (``None``: missing)."""
    if target is None:
        return None
    if "point" in target:
        m = np.eye(4)
        m[:3, 3] = target["point"]
        if target.get("space") == "armature" and rig_model is not None:
            m = rig_model[2] @ m
        return m
    if "object" in target:
        obj = scene._objects.get(target["object"]) or scene._by_name.get(target["object"])
        if obj is None:
            return None
        if "joint" in target:
            found = result.joints.get(obj._id)
            if found is None:
                return None
            rig, model = found
            k = rig.index(target["joint"])
            if k is None:
                return None
            return worlds.world[obj._id] @ model[k]
        return worlds.world[obj._id]
    if "joint" in target and rig_model is not None:
        rig, model, armature_world = rig_model
        k = rig.index(target["joint"])
        return None if k is None else armature_world @ model[k]
    return None


def _run_rig(
    system: AnimationSystem,
    pose: AnimationPose,
    rig: Rig,
    arm_id: str,
    worlds: _Worlds,
    authored: dict[str, np.ndarray] | None,
    result: RiggingResult,
) -> None:
    scene = system.scene
    arm = scene._objects.get(arm_id)
    if arm is None:
        result.diagnostics.append(
            _diag("TARGET_MISSING", "a rig's armature object is gone", f"animation.rigs[{arm_id}]")
        )
        return
    pos, rot, scl = (a.copy() for a in rig.rest_trs())
    posed = pose.joints.get(arm_id, {})
    unknown = []
    for name, (p, q, s) in posed.items():
        k = rig.index(name)
        if k is None:
            unknown.append(name)
            continue
        if p is not None:
            pos[k] = p
        if q is not None:
            rot[k] = q
        if s is not None:
            scl[k] = s
    if unknown:
        result.diagnostics.append(
            _diag(
                "JOINT_MISSING",
                f"channels animate joints the rig of {arm.name!r} does not have: "
                f"{sorted(unknown)[:8]}",
                f"objects[{arm.name}]",
            )
        )
    local = local_matrices(pos, rot, scl)
    parents, depth = rig.parents, rig.depth
    model = forward_kinematics(local, parents, depth)
    armature_world = worlds.world[arm_id]
    influences = pose.influences.get(arm_id, {})
    for i in constraint_order(rig):
        c = rig.constraints[i]
        if c.mute:
            continue
        w = influences.get(c.name, c.influence)
        if w <= 0.0:
            continue
        try:
            local, model = _joint_constraint(
                c, w, rig, local, model, armature_world, worlds, result, scene, arm
            )
        except Exception as exc:  # noqa: BLE001 - a failing constraint is reported, not raised
            result.diagnostics.append(
                _diag(
                    "CONSTRAINT_FAILED",
                    f"constraint {c.name!r} of {arm.name!r} failed: {exc}",
                    f"objects[{arm.name}].constraints[{c.name}]",
                )
            )
    if rig.enforce_limits == "always":
        limited = False
        for k, j in enumerate(rig.joints):
            if j.limits != "null":
                new = _limit(rig, k, local[k])
                if new is not local[k]:
                    local = local if limited else local.copy()
                    local[k] = new
                    limited = True
        if limited:
            model = forward_kinematics(local, parents, depth)
    result.joints[arm_id] = (rig, model)
    # Attachments: W' = A' J' (A J)^-1 W (rest: authored).
    if rig.attachments:
        assert authored is not None
        rest_model = rig.rest_model()
        a_rest = authored[arm_id]
        for oid, jid in _attachments_in_order(worlds.order(), rig.attachments):
            obj = scene._objects.get(oid)
            k = rig.index_of_id(jid)
            if obj is None or k is None:
                result.diagnostics.append(
                    _diag(
                        "ATTACHMENT_MISSING",
                        f"an attachment of {arm.name!r} lost its object or joint",
                        f"objects[{arm.name}]",
                    )
                )
                continue
            delta = armature_world @ model[k] @ np.linalg.inv(a_rest @ rest_model[k])
            w_new = delta @ authored[oid]
            parent = obj._parent
            p_world = np.eye(4) if parent is None else worlds.world[parent._id]
            worlds.set_local(obj, np.linalg.solve(p_world, w_new))


def constraint_order(rig: Rig) -> list[int]:
    """The rig's joint constraints in dependency order: a constraint after those that
    write what it reads -- a twist reads its source joint's own rotation, any other a
    target joint's world matrix (that joint and its ancestors) -- the list's order
    otherwise (a cycle keeps it). Cached on the rig."""
    found = rig._cache.get("constraint_order")
    if found is not None:
        return found
    cons = rig.constraints
    writes: list[set[int]] = []
    reads: list[set[int]] = []
    for c in cons:
        p = c.p
        if c.type == "ik":
            w = {rig.index(n) for n in p["chain"]}
        else:
            w = {rig.index(p["joint"])} if isinstance(p.get("joint"), str) else set()
        r: set[int] = set()
        for key in ("target", "pole"):
            t = p.get(key)
            k = rig.index(t["joint"]) if isinstance(t, dict) and "joint" in t else None
            if k is None:
                continue
            if c.type == "twist":
                r.add(k)
            else:
                r.add(k)
                r.update(_ancestors(rig, k))
        writes.append({k for k in w if k is not None})
        reads.append(r)
    deps = [
        {j for j in range(len(cons)) if j != i and writes[j] & reads[i]} for i in range(len(cons))
    ]
    order: list[int] = []
    done: set[int] = set()
    while len(order) < len(cons):
        ready = [i for i in range(len(cons)) if i not in done and deps[i] <= done]
        if not ready:  # a cycle: the rest in list order
            order.extend(i for i in range(len(cons)) if i not in done)
            break
        order.append(ready[0])
        done.add(ready[0])
    rig._cache["constraint_order"] = order
    return order


def _ancestors(rig: Rig, k: int) -> set[int]:
    out = set()
    p = rig.parents[k]
    while p >= 0:
        out.add(int(p))
        p = rig.parents[p]
    return out


def _attachments_in_order(order: dict[str, int], attachments: tuple) -> list:
    return sorted(attachments, key=lambda a: order.get(a[0], -1))


def _joint_constraint(
    c: Constraint, w: float, rig: Rig, local: np.ndarray, model: np.ndarray,
    armature_world: np.ndarray, worlds: _Worlds, result: RiggingResult, scene: Any, arm: Any,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    p = c.p
    if c.type == "twist":  # (local rotations only: no targets, no armature space)
        return _twist(p, w, rig, local)
    inv_a = np.linalg.inv(armature_world)
    rig_model = (rig, model, armature_world)
    target = _target_world(p.get("target"), worlds, result, scene, rig_model)
    if target is None:
        result.diagnostics.append(
            _diag(
                "CONSTRAINT_TARGET_MISSING",
                f"the target of {c.name!r} is missing; it does nothing",
                f"objects[{arm.name}].constraints[{c.name}]",
            )
        )
        return local, model
    target_a = inv_a @ target  # in armature space
    new_local = local.copy()
    parents = rig.parents
    if c.type == "ik":
        chain = [rig.index(n) for n in p["chain"]]
        positions = model[chain][:, :3, 3].copy()
        goal = target_a[:3, 3]
        pole_w = (
            _target_world(p.get("pole"), worlds, result, scene, rig_model)
            if p.get("pole")
            else None
        )
        pole = None if pole_w is None else (inv_a @ pole_w)[:3, 3]
        if p["solver"] == "two_bone":
            mid, end, residual = ik.two_bone(
                positions[0],
                positions[1],
                positions[2],
                goal,
                pole,
                stretch=p["stretch"],
                soft=p["soft"],
            )
            new_pos = np.stack([positions[0], mid, end])
        elif p["solver"] == "fabrik":
            new_pos, residual, _ = ik.fabrik(
                positions, goal, tolerance=p["tolerance"], iterations=p["iterations"], pole=pole
            )
        else:
            new_pos, residual, _ = ik.ccd(
                positions, goal, tolerance=p["tolerance"], iterations=p["iterations"]
            )
        deltas = ik.chain_rotations(positions, new_pos)
        new_model = model.copy()
        acc = np.array([0.0, 0.0, 0.0, 1.0])
        for i, k in enumerate(chain):
            if i < len(deltas):
                acc = R.multiply(deltas[i][None], acc[None])[0]
            rot = R.matrices(acc[None])[0]
            m = model[k].copy()
            m[:3, :3] = rot @ model[k][:3, :3]
            m[:3, 3] = new_pos[i]
            if i == len(chain) - 1 and p["end"] == "target":
                _, s = model[k][:3, :3], np.linalg.norm(model[k][:3, :3], axis=0)
                m[:3, :3] = target_a[:3, :3] / np.linalg.norm(target_a[:3, :3], axis=0) * s
            new_model[k] = m
        if p.get("bend_axis") and len(chain) == 3:
            _align_bend(new_model, chain, new_pos, p["bend_axis"])
        limited = False
        for i, k in enumerate(chain):
            if i == len(chain) - 1 and p["end"] == "keep":
                break
            parent_m = np.eye(4) if parents[k] < 0 else new_model[parents[k]]
            free = np.linalg.solve(parent_m, new_model[k])
            new_local[k] = _limit(rig, k, free)
            limited |= new_local[k] is not free
        if limited:  # the residual is where the limited chain's end really is
            reached = forward_kinematics(new_local, rig.parents, rig.depth)[chain[-1]][:3, 3]
            residual = float(np.linalg.norm(reached - goal))
        result.residuals[f"{arm.name}/{c.name}"] = float(residual)
    else:
        k = rig.index(p["joint"])
        parent_m = np.eye(4) if parents[k] < 0 else model[parents[k]]
        owner = model[k]
        out = apply_constraint(c, owner, target_a, target_a[:3, 3], parent=parent_m)
        new_local[k] = _limit(rig, k, np.linalg.solve(parent_m, out))
    if w < 1.0:
        for k in range(local.shape[0]):
            if not np.array_equal(new_local[k], local[k]):
                new_local[k] = blend_matrices(local[k], new_local[k], w)
    new_model = forward_kinematics(new_local, rig.parents, rig.depth)
    return new_local, new_model


def _twist(p: dict[str, Any], w: float, rig: Rig, local: np.ndarray):
    """A twist constraint: the joint turns about its ``axis`` by ``factor`` times the source
    joint's twist about it (relative to rest; :func:`.limits.twist_angle`); the influence
    scales the angle (the same rotation as blending toward it, along the shortest arc).
    Scalar arithmetic: two of these run per character and frame."""
    from . import limits as L

    src = rig.index(p["target"]["joint"])
    k = rig.index(p["joint"])
    m = local[src][:3, :3]
    qx, qy, qz, qw = _quat_of((m / np.linalg.norm(m, axis=0)).tolist())
    rx, ry, rz, rw = rig.rest_trs()[1][src].tolist()
    # q_rest^-1 q (Hamilton), then w >= 0
    x = rw * qx - rx * qw - ry * qz + rz * qy
    y = rw * qy + rx * qz - ry * qw - rz * qx
    z = rw * qz - rx * qy + ry * qx - rz * qw
    v = rw * qw + rx * qx + ry * qy + rz * qz
    n = math.sqrt(x * x + y * y + z * z + v * v)
    if v < 0.0:
        n = -n
    axis = L.AXES[p["axis"]]
    ax, ay, az = axis.tolist()
    proj = (x * ax + y * ay + z * az) / n
    w_rel = v / n
    if abs(proj) < 1e-15 and abs(w_rel) < 1e-15:
        return local, forward_kinematics(local, rig.parents, rig.depth)
    angle = 2.0 * math.atan2(proj, w_rel) * p["factor"] * w
    new_local = local.copy()
    if angle != 0.0:
        # The joint's own rotation turned about its axis: T R S -> T (R turn) S.
        mk = local[k][:3, :3]
        scale = np.linalg.norm(mk, axis=0)
        c, s_ = math.cos(angle), math.sin(angle)
        kx = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
        turn = np.eye(3) + s_ * kx + (1.0 - c) * (kx @ kx)
        new_local[k][:3, :3] = ((mk / scale) @ turn) * scale
    return new_local, forward_kinematics(new_local, rig.parents, rig.depth)


def _quat_of(m: list) -> tuple[float, float, float, float]:
    """``(x, y, z, w)`` of a 3x3 rotation (rows; Shepperd's method, as math3d)."""
    (a, b, c), (d, e, f), (g, h, i) = m
    tr = a + e + i
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (h - f) / s, (c - g) / s, (d - b) / s, 0.25 * s
    if a > e and a > i:
        s = math.sqrt(1.0 + a - e - i) * 2.0
        return 0.25 * s, (b + d) / s, (c + g) / s, (h - f) / s
    if e > i:
        s = math.sqrt(1.0 + e - a - i) * 2.0
        return (b + d) / s, 0.25 * s, (f + h) / s, (c - g) / s
    s = math.sqrt(1.0 + i - a - e) * 2.0
    return (c + g) / s, (f + h) / s, 0.25 * s, (d - b) / s


def _align_bend(model: np.ndarray, chain: list, pos: np.ndarray, axis: str) -> None:
    """Twist the first two bones of a chain about themselves so their ``axis`` lies
    along the bend plane's normal ``d1 x d2`` (positions unchanged): a positive rotation
    about it then bends the chain the way it bends now."""
    from .limits import AXES

    d1 = pos[1] - pos[0]
    d2 = pos[2] - pos[1]
    n = np.cross(d1, d2)
    nn = float(np.linalg.norm(n))
    scale_len = float(np.linalg.norm(d1) * np.linalg.norm(d2))
    if nn < 1e-9 * scale_len or nn == 0.0:
        return  # a straight chain: no plane
    # A slightly bent limb needs no alignment (its limits allow a little sideways swing),
    # and its plane is poorly defined: the alignment fades in between 20 and 45 degrees
    # of bend. So switching a relaxed FK limb to IK does not roll its bones (the visible
    # pose is kept), while deep bends (crouches, reaches) bend about the bend axis.
    bend = math.degrees(math.atan2(nn, float(d1 @ d2)))
    fade = min(1.0, max(0.0, (bend - 20.0) / 25.0))
    fade = fade * fade * (3.0 - 2.0 * fade)
    if fade <= 0.0:
        return
    n /= nn
    a_local = AXES[axis]
    for k, d in ((chain[0], d1), (chain[1], d2)):
        lin = model[k][:3, :3]
        scale = np.linalg.norm(lin, axis=0)
        rot = lin / scale
        bone = d / np.linalg.norm(d)
        current = rot @ a_local
        cur = current - (current @ bone) * bone
        want = n - (n @ bone) * bone
        if np.linalg.norm(cur) < 1e-9 or np.linalg.norm(want) < 1e-9:
            continue
        cur /= np.linalg.norm(cur)
        want /= np.linalg.norm(want)
        angle = math.atan2(float(np.cross(cur, want) @ bone), float(cur @ want)) * fade
        turn = R.matrices(R.axis_angle(bone, angle))[0]
        model[k][:3, :3] = (turn @ rot) * scale


def _limit(rig: Rig, k: int, local: np.ndarray) -> np.ndarray:
    """A joint's local matrix with its rotation limits (Euler XYZ degrees) applied."""
    from .rig import joint_limits

    limits = joint_limits(rig, k)
    if not limits:
        return local
    if "rotation" not in limits:  # swing-twist or hinge, from the rest rotation
        from . import limits as L
        from .constraints import compose

        p, q, s = decompose(local)
        clamped = L.clamp(limits, q, rig.rest_trs()[1][k])
        return local if clamped is None else compose(p, clamped, s)
    rot = limits["rotation"]
    p, q, s = decompose(local)
    e = R.to_euler(q[None])[0]
    changed = False
    for i in range(3):
        lo, hi = rot["min"][i], rot["max"][i]
        if lo is not None and e[i] < lo:
            e[i], changed = lo, True
        if hi is not None and e[i] > hi:
            e[i], changed = hi, True
    if not changed:
        return local
    from .constraints import compose

    return compose(p, R.from_euler(e)[0], s)


def _run_object_constraints(
    system: AnimationSystem,
    pose: AnimationPose,
    oid: str,
    constraints: tuple,
    worlds: _Worlds,
    result: RiggingResult,
) -> None:
    scene = system.scene
    obj = scene._objects.get(oid)
    if obj is None:
        return
    parent = obj._parent
    p_world = np.eye(4) if parent is None else worlds.world[parent._id]
    world = worlds.world[oid].copy()
    influences = pose.influences.get(oid, {})
    for c in constraints:
        if c.mute:
            continue
        w = influences.get(c.name, c.influence)
        if w <= 0.0:
            continue
        if c.type == "follow_path":
            offset = influences.get(f"{c.name}.offset", c.p["offset"])
            out = system._follow_path(c, world, offset)
            if out is None:
                result.diagnostics.append(
                    _diag(
                        "CONSTRAINT_TARGET_MISSING",
                        f"the path of {c.name!r} is missing",
                        f"objects[{obj.name}].constraints[{c.name}]",
                    )
                )
                continue
        else:
            target = _target_world(c.p.get("target"), worlds, result, scene, None)
            if target is None and c.type not in ("limit_location", "limit_rotation", "limit_scale"):
                result.diagnostics.append(
                    _diag(
                        "CONSTRAINT_TARGET_MISSING",
                        f"the target of {c.name!r} is missing; it does nothing",
                        f"objects[{obj.name}].constraints[{c.name}]",
                    )
                )
                continue
            if c.type == "child_of" and c.p["inverse"] is None:
                target_rest = _target_rest(c, scene)
                c = _with_inverse(c, target_rest)
            out = apply_constraint(
                c, world, target, None if target is None else target[:3, 3], parent=p_world
            )
        world = blend_matrices(world, out, w)
    worlds.set_local(obj, np.linalg.solve(p_world, world))


def _target_rest(c: Constraint, scene: Any) -> np.ndarray:
    from ..rendering.scene_data import world_matrices

    t = c.p["target"]
    obj = scene._objects.get(t.get("object", "")) or scene._by_name.get(t.get("object", ""))
    return world_matrices(scene)[obj._id] if obj is not None else np.eye(4)


def _with_inverse(c: Constraint, target_rest: np.ndarray) -> Constraint:
    import json

    p = c.p
    p["inverse"] = np.linalg.inv(target_rest).tolist()
    return Constraint(
        c.name, c.type, json.dumps(p, sort_keys=True, separators=(",", ":")), c.influence, c.mute
    )


def _run_skin(
    system: AnimationSystem,
    pose: AnimationPose,
    oid: str,
    skin: Any,
    worlds: _Worlds,
    result: RiggingResult,
    extra: dict[str, float] | None = None,
) -> None:
    from ..mesh import Mesh

    scene = system.scene
    obj = scene._objects.get(oid)
    item = system._items.get(skin.rig)
    if obj is None or item is None or not item._alive:
        result.diagnostics.append(
            _diag("SKIN_MISSING", "a skin lost its mesh object or rig", f"animation.skins[{oid}]")
        )
        return
    rig = item._definition
    found = result.joints.get(rig.object)
    if found is None:
        return
    _, model = found
    base = _morphed(system, pose, oid, extra)
    mesh = base if base is not None else obj.mesh
    if mesh is None or mesh.vertex_count != skin.vertex_count:
        result.diagnostics.append(
            _diag(
                "SKIN_MESH_MISMATCH",
                f"the skin of {obj.name!r} was bound to {skin.vertex_count} vertices; its mesh has "
                f"{0 if mesh is None else mesh.vertex_count} (rebind after editing the mesh)",
                f"objects[{obj.name}]",
            )
        )
        return
    idx = [rig.index_of_id(j) for j in skin.joints]
    if any(i is None for i in idx):
        result.diagnostics.append(
            _diag(
                "SKIN_JOINT_MISSING",
                f"the skin of {obj.name!r} uses joints the rig lost",
                f"objects[{obj.name}]",
            )
        )
        return
    a_world = worlds.world[rig.object]
    m_world_inv = np.linalg.inv(worlds.world[oid])
    palette = m_world_inv @ a_world @ model[idx] @ skin.inverse_bind
    if skin.method == "linear":
        pos, nrm = skin_linear(mesh.positions, mesh.normals, skin.indices, skin.weights, palette)
    else:
        pos, nrm = skin_dual_quaternion(
            mesh.positions, mesh.normals, skin.indices, skin.weights, palette,
            skin.dq if skin.method == "blended" else None,
        )  # fmt: skip
    if not np.all(np.isfinite(pos)):
        result.diagnostics.append(
            _diag(
                "SKIN_NONFINITE",
                f"skinning {obj.name!r} gave non-finite positions",
                f"objects[{obj.name}]",
            )
        )
        return
    result.meshes[oid] = Mesh._from_arrays(
        pos, mesh.triangles, nrm, mesh.uvs, mesh.groups, mesh.colors
    )


def _morphed(
    system: AnimationSystem, pose: AnimationPose, oid: str, extra: dict[str, float] | None = None
) -> Any:
    """The morphed mesh of ``oid`` (``None``: no morph targets or all weights 0);
    ``extra``: weights correctives add to the animated (or default) ones."""
    morphs = system._morphs() if hasattr(system, "_morphs") else {}
    morph = morphs.get(oid)
    if morph is None:
        return None
    weights = pose.morphs.get(oid, {})
    if extra:
        weights = dict(weights)
        for name, w in extra.items():
            if name in morph.names:
                base = weights.get(name, morph.defaults[morph.names.index(name)])
                weights[name] = base + w
    return morph.apply(system.scene._objects[oid].mesh, weights)
