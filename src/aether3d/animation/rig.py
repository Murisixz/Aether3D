"""Skeletons, rigs, rigid attachments and skins.

A **rig** is a skeleton (joints with stable ids, unique names, parents and rest
transforms) attached to an *armature object*: joint matrices are in the armature
object's space, so moving the armature (root motion, placement) moves the skeleton.
Joint channels of clips bound to the armature object (``joints["NAME"].rotation`` ...)
pose it; joint constraints (IK, look-at, limits...) adjust it; then

* **attachments** move rigid parts with joints -- an attached object follows its joint's
  motion relative to the rest pose (``W' = A' J' (A J)^-1 W``: armature and joint now,
  times their inverse at rest, times the object's authored world matrix), whatever the
  scene hierarchy; blocky characters animate without remodelling;
* **skins** deform meshes smoothly: per vertex up to 8 joints with normalized weights,
  ``v' = sum_k w_k S_k v`` with the skinning matrices ``S_j = W_m'^-1 A' J'_j B_j``
  (``B_j``: the inverse bind matrix, mesh space at bind to joint space at bind), normals
  by the cofactor of the blended linear part (correct for non-uniform scale and
  mirroring), or dual quaternion blending (``method: "dual_quaternion"``, Kavan et al.
  2008) for rigid joint motion without the volume loss of linear blending.

Conventions: joints are stored parents first; a joint's rest transform is relative to its
parent joint (the root's to the armature object); its ``tail`` (the bone's far end in its
own space, default: toward its first child, else ``(0, 0.1, 0)``) gives bones a length and
direction for display and solvers.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import paths as P
from . import rotation as R
from .constraints import Constraint, parse_constraints

MAX_JOINTS = 4096
MAX_INFLUENCES = 8
MAX_SKIN_VERTICES = 10_000_000
SKIN_CHUNK = 65536
METHODS = ("linear", "dual_quaternion", "blended")


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _vec(value: Any, n: int, location: str, what: str) -> tuple[float, ...]:
    try:
        arr = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err("INVALID_RIG", f"{what} is {n} numbers", location) from None
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise _err("INVALID_RIG", f"{what} is {n} finite numbers", location)
    return tuple(float(x) for x in arr)


@dataclass(frozen=True)
class Joint:
    id: str
    name: str
    parent: int  # index of the parent joint (-1: a root)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tail: tuple[float, float, float] | None = None
    limits: str = "null"  # canonical JSON (see :mod:`.limits`)
    role: str | None = None  # semantic role: "hips", "upper_arm.L", ... (unique per rig)
    deform: bool = True  # False: a helper joint (controls, IK, twist sources): no weights
    # The part of the bone (fractions from head to tail) its skin weights cover: a bone
    # split for twist (the upper arm's shoulder half on its counter-twisting helper) keeps
    # its whole length for animation and weighs only on its own part.
    span: tuple[float, float] = (0.0, 1.0)

    def to_json(self, joints: tuple[Joint, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "parent": None if self.parent < 0 else joints[self.parent].name,
            "position": list(self.position),
            "rotation": list(self.rotation),
        }
        if self.scale != (1.0, 1.0, 1.0):
            out["scale"] = list(self.scale)
        if self.tail is not None:
            out["tail"] = list(self.tail)
        if self.limits != "null":
            out["limits"] = json.loads(self.limits)
        if self.role is not None:
            out["role"] = self.role
        if not self.deform:
            out["deform"] = False
        if self.span != (0.0, 1.0):
            out["span"] = list(self.span)
        return out


CONTROL_KINDS = ("ik_target", "pole", "gaze", "fk", "root", "custom")
CONTROL_SHAPES = ("sphere", "cube", "diamond", "circle", "arrow", "none")
CORRECTIVE_DRIVERS = ("angle", "swing", "twist", "rotation")
ENFORCE_LIMITS = ("ik", "always")
INFO_KEYS = ("forward", "up", "description", "source", "style", "pose")
MAX_CONTROLS = 4096
MAX_CORRECTIVES = 1024
MAX_LANDMARKS = 1024
_ROLE = re.compile(r"^[a-z][a-z0-9_]*(\.[A-Za-z0-9_]+)*$")


@dataclass(frozen=True)
class Control:
    """A named animation control of a rig. Controls the animator moves are ordinary scene
    objects (``object``: IK targets, poles, gaze targets, the root), so every transform
    channel, layer, path and export works on them; ``fk`` controls are the joints
    themselves (their channels). ``constraint`` names the rig constraint the control
    feeds (an ``ik`` or ``look_at``); ``group`` and ``side`` organize controls (``arm.L``,
    ``leg.R``, ``spine``, ``head``); ``shape`` and ``size`` are display hints."""

    name: str
    kind: str
    object: str | None = None  # scene object id (targets, poles, gaze, root)
    joint: str | None = None  # the joint it drives or follows (by name)
    constraint: str | None = None  # the rig constraint it feeds (by name)
    group: str = ""
    side: str | None = None  # "L" | "R" | None
    shape: str = "sphere"
    size: float = 0.05
    mirror: str | None = None  # the control on the other side (by name)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind}
        for key in ("object", "joint", "constraint", "side", "mirror"):
            v = getattr(self, key)
            if v is not None:
                out[key] = v
        if self.group:
            out["group"] = self.group
        if self.shape != "sphere":
            out["shape"] = self.shape
        if self.size != 0.05:
            out["size"] = self.size
        return out


@dataclass(frozen=True)
class Corrective:
    """A pose-driven corrective: the weight of morph target ``target`` of mesh object
    ``mesh`` follows an angle of ``joint`` (from its rest rotation, after constraints):
    ``driver`` ``angle`` (the whole rotation), ``swing`` (how far the bone ``axis``
    turned; with ``direction``, the signed part toward that axis), ``twist`` (signed,
    about ``axis``) or ``rotation`` (the signed rotation-vector component along ``axis``:
    a hinge's bend). ``range`` (degrees) maps to 0..1, shaped by ``curve`` (``linear`` or
    ``smooth``), times ``weight``; the result adds to the target's animated weight."""

    name: str
    mesh: str  # mesh object id
    target: str  # morph target name
    joint: str  # driver joint name
    driver: str = "rotation"
    axis: str = "+x"
    direction: str | None = None
    range: tuple[float, float] = (0.0, 90.0)
    curve: str = "smooth"
    weight: float = 1.0

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "mesh": self.mesh,
            "target": self.target,
            "joint": self.joint,
            "driver": self.driver,
            "axis": self.axis,
            "range": list(self.range),
            "curve": self.curve,
        }
        if self.direction is not None:
            out["direction"] = self.direction
        if self.weight != 1.0:
            out["weight"] = self.weight
        return out

    def value(self, q_local: Any, q_rest: Any) -> float:
        """The driver's angle (degrees) for a joint rotation."""
        from . import limits as L

        q = L.relative(np.asarray(q_local, np.float64), np.asarray(q_rest, np.float64))
        if self.driver == "angle":
            return math.degrees(2.0 * math.acos(min(1.0, float(q[3]))))
        axis = L.AXES[self.axis]
        if self.driver == "twist":
            return L.twist_angle(q, axis)
        if self.driver == "swing":
            sv = L.swing_vector(q, axis)
            if self.direction is None:
                return float(np.linalg.norm(sv))
            return float(sv @ L.AXES[self.direction])
        # rotation: the rotation vector's component along the axis
        n = float(np.linalg.norm(q[:3]))
        if n < 1e-15:
            return 0.0
        angle = 2.0 * math.atan2(n, float(q[3]))
        return math.degrees(angle) * float(q[:3] @ axis) / n

    def weight_for(self, value: float) -> float:
        a, b = self.range
        u = (value - a) / (b - a)
        u = min(1.0, max(0.0, u))
        if self.curve == "smooth":
            u = u * u * (3.0 - 2.0 * u)
        return u * self.weight


@dataclass(frozen=True)
class Rig:
    """A skeleton on an armature object (immutable; see the module docstring)."""

    object: str
    joints: tuple[Joint, ...]
    attachments: tuple[tuple[str, str], ...] = ()  # (object id, joint id)
    constraints: tuple[Constraint, ...] = ()
    mirror: tuple[tuple[str, str], ...] = ((".L", ".R"), ("Left", "Right"), ("_l", "_r"))
    controls: tuple[Control, ...] = ()
    correctives: tuple[Corrective, ...] = ()
    landmarks: tuple[tuple[str, tuple[float, float, float]], ...] = ()  # armature space
    template: str | None = None
    info: str = "null"  # canonical JSON: {forward, up, description, source, style, pose}
    enforce_limits: str = "ik"  # "ik": limits on joints IK moves (Stage 17); "always"
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # -- structure -------------------------------------------------------------------------

    def index(self, name: str) -> int | None:
        found = self._cache.get("index")
        if found is None:
            found = self._cache["index"] = {j.name: k for k, j in enumerate(self.joints)}
        return found.get(name)

    def index_of_id(self, jid: str) -> int | None:
        found = self._cache.get("ids")
        if found is None:
            found = self._cache["ids"] = {j.id: k for k, j in enumerate(self.joints)}
        return found.get(jid)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def parents(self) -> np.ndarray:
        found = self._cache.get("parents")
        if found is None:
            found = np.array([j.parent for j in self.joints], np.int64)
            found.flags.writeable = False
            self._cache["parents"] = found
        return found

    @property
    def depth(self) -> np.ndarray:
        found = self._cache.get("depth")
        if found is None:
            d = np.zeros(len(self.joints), np.int64)
            for k, j in enumerate(self.joints):
                d[k] = 0 if j.parent < 0 else d[j.parent] + 1
            d.flags.writeable = False
            found = self._cache["depth"] = d
        return found

    def rest_trs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        found = self._cache.get("rest_trs")
        if found is None:
            pos = np.array([j.position for j in self.joints], np.float64).reshape(-1, 3)
            rot = np.array([j.rotation for j in self.joints], np.float64).reshape(-1, 4)
            scl = np.array([j.scale for j in self.joints], np.float64).reshape(-1, 3)
            for a in (pos, rot, scl):
                a.flags.writeable = False
            found = self._cache["rest_trs"] = (pos, rot, scl)
        return found

    def rest_local(self) -> np.ndarray:
        found = self._cache.get("rest_local")
        if found is None:
            pos, rot, scl = self.rest_trs()
            found = local_matrices(pos, rot, scl)
            found.flags.writeable = False
            self._cache["rest_local"] = found
        return found

    def rest_model(self) -> np.ndarray:
        """Rest joint matrices in armature space."""
        found = self._cache.get("rest_model")
        if found is None:
            found = forward_kinematics(self.rest_local(), self.parents, self.depth)
            found.flags.writeable = False
            self._cache["rest_model"] = found
        return found

    def tails(self) -> np.ndarray:
        """Each joint's bone end in its own space (``tail``, else toward its first child,
        else ``(0, 0.1, 0)``)."""
        found = self._cache.get("tails")
        if found is None:
            n = len(self.joints)
            out = np.zeros((n, 3))
            first_child = [-1] * n
            for k, j in enumerate(self.joints):
                if j.parent >= 0 and first_child[j.parent] < 0:
                    first_child[j.parent] = k
            for k, j in enumerate(self.joints):
                if j.tail is not None:
                    out[k] = j.tail
                elif first_child[k] >= 0:
                    out[k] = self.joints[first_child[k]].position
                else:
                    out[k] = (0.0, 0.1, 0.0)
            out.flags.writeable = False
            found = self._cache["tails"] = out
        return found

    def deform_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Each joint's weighting segment at rest in armature space ``(starts, ends)``:
        the part of its bone (head to tail) its ``span`` covers."""
        found = self._cache.get("deform_segments")
        if found is None:
            model = self.rest_model()
            tails = self.tails()
            span = np.array([j.span for j in self.joints], np.float64).reshape(-1, 2)
            a = np.einsum("nij,nj->ni", model[:, :3, :3], span[:, :1] * tails) + model[:, :3, 3]
            b = np.einsum("nij,nj->ni", model[:, :3, :3], span[:, 1:] * tails) + model[:, :3, 3]
            for arr in (a, b):
                arr.flags.writeable = False
            found = self._cache["deform_segments"] = (a, b)
        return found

    def descendants(self, k: int) -> list[int]:
        out = []
        parents = self.parents
        inside = {k}
        for i in range(k + 1, len(self.joints)):
            if int(parents[i]) in inside:
                inside.add(i)
                out.append(i)
        return out

    def role_index(self, role: str) -> int | None:
        """The joint with semantic ``role`` (``"hips"``, ``"upper_arm.L"``...), or ``None``."""
        found = self._cache.get("roles")
        if found is None:
            found = self._cache["roles"] = {
                j.role: k for k, j in enumerate(self.joints) if j.role is not None
            }
        return found.get(role)

    @property
    def roles(self) -> dict[str, str]:
        """Role -> joint name."""
        return {j.role: j.name for j in self.joints if j.role is not None}

    @property
    def deforming(self) -> tuple[int, ...]:
        """Indices of the joints that may carry skin weights (``deform``)."""
        return tuple(k for k, j in enumerate(self.joints) if j.deform)

    def control(self, name: str) -> Control | None:
        for c in self.controls:
            if c.name == name:
                return c
        return None

    @property
    def info_dict(self) -> dict[str, Any]:
        return {} if self.info == "null" else json.loads(self.info)

    @property
    def landmark_dict(self) -> dict[str, tuple[float, float, float]]:
        return dict(self.landmarks)

    def forward_axis(self, default: str = "-y") -> str:
        """The direction the character faces in armature space (``info.forward``)."""
        return self.info_dict.get("forward", default)

    def mirror_name(self, name: str) -> str | None:
        """The joint name on the other side (``Arm.L`` -> ``Arm.R``), or ``None``."""
        for left, right in self.mirror:
            for a, b in ((left, right), (right, left)):
                if name.endswith(a):
                    return name[: -len(a)] + b
                if name.startswith(a):
                    return b + name[len(a) :]
        return None

    # -- serialization ---------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "object": self.object,
            "joints": [j.to_json(self.joints) for j in self.joints],
        }
        if self.attachments:
            out["attachments"] = [
                {"object": oid, "joint": self.joints[self.index_of_id(jid)].name}  # type: ignore[index]
                for oid, jid in self.attachments
            ]
        if self.constraints:
            out["constraints"] = [c.to_json() for c in self.constraints]
        if self.mirror != Rig.__dataclass_fields__["mirror"].default:
            out["mirror"] = [list(p) for p in self.mirror]
        if self.controls:
            out["controls"] = [c.to_json() for c in self.controls]
        if self.correctives:
            out["correctives"] = [c.to_json() for c in self.correctives]
        if self.landmarks:
            out["landmarks"] = {name: list(p) for name, p in self.landmarks}
        if self.template is not None:
            out["template"] = self.template
        if self.info != "null":
            out["info"] = json.loads(self.info)
        if self.enforce_limits != "ik":
            out["enforce_limits"] = self.enforce_limits
        return out

    def replace(self, **changes: Any) -> Rig:
        """A copy with some fields changed (caches are not shared)."""
        fields = {f: getattr(self, f) for f in Rig.__dataclass_fields__ if f != "_cache"}
        fields.update(changes)
        return Rig(**fields)


def parse_rig(raw: Any, location: str) -> Rig:
    """A checked rig: joints parents first (reordered when needed), names unique, ids
    unique (generated when missing), no cycles."""
    if isinstance(raw, Rig):
        return raw
    if not isinstance(raw, dict) or set(raw) - set(RIG_KEYS):
        raise _err("INVALID_RIG", f"a rig is {{{', '.join(RIG_KEYS)}}}", location)
    obj = raw.get("object")
    if not isinstance(obj, str) or not obj:
        raise _err("INVALID_RIG", "a rig names its armature object", f"{location}.object")
    joints_raw = raw.get("joints")
    if not isinstance(joints_raw, list) or not 1 <= len(joints_raw) <= MAX_JOINTS:
        raise _err("INVALID_RIG", f"a rig has 1..{MAX_JOINTS} joints", f"{location}.joints")
    rows = []
    names: dict[str, int] = {}
    ids: set[str] = set()
    roles: set[str] = set()
    for i, j in enumerate(joints_raw):
        loc = f"{location}.joints[{i}]"
        if not isinstance(j, dict) or set(j) - set(JOINT_KEYS):
            raise _err(
                "INVALID_RIG",
                "a joint is {id, name, parent, position, rotation, scale, tail, limits, role, "
                "deform, span}",
                loc,
            )
        name = P.check_name(j.get("name"), "a joint name", f"{loc}.name")
        if name in names:
            raise _err("INVALID_RIG", f"two joints are named {name!r}", f"{loc}.name")
        names[name] = i
        jid = j.get("id") or str(uuid.uuid4())
        if not isinstance(jid, str) or jid in ids:
            raise _err("INVALID_RIG", "joint ids must be unique strings", f"{loc}.id")
        ids.add(jid)
        parent = j.get("parent")
        if parent is not None:
            P.check_name(parent, "a parent joint name", f"{loc}.parent")
        pos = _vec(j.get("position", (0, 0, 0)), 3, f"{loc}.position", "position")
        if "euler" in j and "rotation" in j:
            raise _err("INVALID_RIG", "give rotation or euler, not both", loc)
        if "euler" in j:
            rot = tuple(
                R.from_euler(np.array(_vec(j["euler"], 3, f"{loc}.euler", "euler")))[0].tolist()
            )
        else:
            rot = _vec(j.get("rotation", (0, 0, 0, 1)), 4, f"{loc}.rotation", "rotation")
        norm = math.sqrt(sum(c * c for c in rot))
        if norm < 1e-9:
            raise _err("INVALID_RIG", "a rotation must not be zero", f"{loc}.rotation")
        if abs(norm - 1.0) > 1e-15:
            rot = tuple(c / norm for c in rot)
        scl = _vec(j.get("scale", (1, 1, 1)), 3, f"{loc}.scale", "scale")
        if any(abs(s) < 1e-12 for s in scl):
            raise _err("INVALID_RIG", "a joint's rest scale must not be zero", f"{loc}.scale")
        tail = None if j.get("tail") is None else _vec(j["tail"], 3, f"{loc}.tail", "tail")
        limits = j.get("limits")
        limits_json = "null"
        if limits is not None:
            limits_json = _check_limits(limits, f"{loc}.limits")
        role = j.get("role")
        if role is not None:
            if not isinstance(role, str) or len(role) > 64 or not _ROLE.match(role):
                raise _err(
                    "INVALID_RIG",
                    "a role is lower-case words with an optional side: 'hips', 'upper_arm.L'",
                    f"{loc}.role",
                )
            if role in roles:
                raise _err("INVALID_RIG", f"two joints have the role {role!r}", f"{loc}.role")
            roles.add(role)
        deform = j.get("deform", True)
        if not isinstance(deform, bool):
            raise _err("INVALID_RIG", "deform is true or false", f"{loc}.deform")
        span = (0.0, 1.0)
        if j.get("span") is not None:
            sa, sb = _vec(j["span"], 2, f"{loc}.span", "span")
            if not (0.0 <= sa < sb <= 1.0 and sb - sa >= 0.05):
                raise _err(
                    "INVALID_RIG",
                    "span is [start, end]: fractions of the bone, 0 <= start < end <= 1, "
                    "at least 0.05 apart",
                    f"{loc}.span",
                )
            span = (sa, sb)
        rows.append((jid, name, parent, pos, rot, scl, tail, limits_json, role, deform, span))
    # Order parents first (stable), refusing unknown parents and cycles.
    for _, name, parent, *_ in rows:
        if parent is not None and parent not in names:
            raise _err("INVALID_RIG", f"joint {name!r} has an unknown parent {parent!r}", location)
    order: list[int] = []
    placed: set[str] = set()
    remaining = list(range(len(rows)))
    while remaining:
        progressed = False
        rest = []
        for i in remaining:
            parent = rows[i][2]
            if parent is None or parent in placed:
                order.append(i)
                placed.add(rows[i][1])
                progressed = True
            else:
                rest.append(i)
        if not progressed:
            cyc = sorted(rows[i][1] for i in rest)
            raise _err(
                "SKELETON_CYCLE", f"the joints {cyc[:6]} form a parent cycle", f"{location}.joints"
            )
        remaining = rest
    new_index = {rows[i][1]: k for k, i in enumerate(order)}
    joints = tuple(
        Joint(
            rows[i][0],
            rows[i][1],
            -1 if rows[i][2] is None else new_index[rows[i][2]],
            rows[i][3],
            rows[i][4],
            rows[i][5],
            rows[i][6],
            rows[i][7],
            rows[i][8],
            rows[i][9],
            rows[i][10],
        )
        for i in order
    )
    rig = Rig(obj, joints)
    attachments = []
    att_raw = raw.get("attachments", [])
    if not isinstance(att_raw, list) or len(att_raw) > 100_000:
        raise _err("INVALID_RIG", "attachments is a list", f"{location}.attachments")
    seen_objects = set()
    for i, a in enumerate(att_raw):
        loc = f"{location}.attachments[{i}]"
        if not isinstance(a, dict) or set(a) != {"object", "joint"}:
            raise _err("INVALID_RIG", "an attachment is {object, joint}", loc)
        k = rig.index(a["joint"]) if isinstance(a["joint"], str) else None
        if k is None:
            raise _err("INVALID_RIG", f"no joint {a['joint']!r}", f"{loc}.joint")
        if not isinstance(a["object"], str) or a["object"] in seen_objects or a["object"] == obj:
            raise _err(
                "INVALID_RIG", "each object attaches once, and never the armature", f"{loc}.object"
            )
        seen_objects.add(a["object"])
        attachments.append((a["object"], joints[k].id))
    constraints = parse_constraints(
        raw.get("constraints", []), f"{location}.constraints", joints=True
    )
    for i, c in enumerate(constraints):
        p = c.p
        for name in [*(p.get("chain") or []), p.get("joint")]:
            if name is not None and rig.index(name) is None:
                raise _err(
                    "INVALID_CONSTRAINT",
                    f"no joint {name!r} in the rig",
                    f"{location}.constraints[{i}]",
                )
        chain = p.get("chain")
        if chain:
            idx = [rig.index(n) for n in chain]
            for a, b in zip(idx, idx[1:], strict=False):
                if joints[b].parent != a:  # type: ignore[index]
                    raise _err(
                        "INVALID_CONSTRAINT",
                        "an IK chain lists consecutive joints, each the parent of the next",
                        f"{location}.constraints[{i}].chain",
                    )
        for key in ("target", "pole"):
            t = p.get(key)
            if t and "joint" in t and "object" not in t and rig.index(t["joint"]) is None:
                raise _err(
                    "INVALID_CONSTRAINT",
                    f"no joint {t['joint']!r} in the rig",
                    f"{location}.constraints[{i}].{key}",
                )
    mirror = raw.get("mirror")
    if mirror is None:
        mirror_t = Rig.__dataclass_fields__["mirror"].default
    else:
        if not isinstance(mirror, list) or not all(
            isinstance(m, list) and len(m) == 2 and all(isinstance(s, str) and s for s in m)
            for m in mirror
        ):
            raise _err(
                "INVALID_RIG", "mirror is a list of [left, right] name parts", f"{location}.mirror"
            )
        mirror_t = tuple((a, b) for a, b in mirror)
    base = Rig(obj, joints, tuple(attachments), constraints, mirror_t)
    controls = _parse_controls(raw.get("controls", []), base, f"{location}.controls")
    correctives = _parse_correctives(raw.get("correctives", []), base, f"{location}.correctives")
    landmarks = _parse_landmarks(raw.get("landmarks"), f"{location}.landmarks")
    template = raw.get("template")
    if template is not None:
        template = P.check_name(template, "a template name", f"{location}.template")
    info = _parse_info(raw.get("info"), f"{location}.info")
    enforce = raw.get("enforce_limits", "ik")
    if enforce not in ENFORCE_LIMITS:
        raise _err(
            "INVALID_RIG",
            "enforce_limits is 'ik' (limits on joints IK moves) or 'always'",
            f"{location}.enforce_limits",
        )
    if not (controls or correctives or landmarks or template or info != "null" or enforce != "ik"):
        return base
    return base.replace(
        controls=controls,
        correctives=correctives,
        landmarks=landmarks,
        template=template,
        info=info,
        enforce_limits=enforce,
    )


RIG_KEYS = (
    "object",
    "joints",
    "attachments",
    "constraints",
    "mirror",
    "controls",
    "correctives",
    "landmarks",
    "template",
    "info",
    "enforce_limits",
)
JOINT_KEYS = (
    "id",
    "name",
    "parent",
    "position",
    "rotation",
    "euler",
    "scale",
    "tail",
    "limits",
    "role",
    "deform",
    "span",
)
CONTROL_KEYS = (
    "name",
    "kind",
    "object",
    "joint",
    "constraint",
    "group",
    "side",
    "shape",
    "size",
    "mirror",
)
CORRECTIVE_KEYS = (
    "name",
    "mesh",
    "target",
    "joint",
    "driver",
    "axis",
    "direction",
    "range",
    "curve",
    "weight",
)


def _opt_str(raw: dict, key: str, location: str, what: str) -> str | None:
    v = raw.get(key)
    if v is None:
        return None
    return P.check_name(v, what, f"{location}.{key}")


def _parse_controls(raw: Any, rig: Rig, location: str) -> tuple[Control, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_CONTROLS:
        raise _err("INVALID_RIG", f"controls is a list of at most {MAX_CONTROLS}", location)
    out = []
    names: set[str] = set()
    constraint_names = {c.name for c in rig.constraints}
    for i, c in enumerate(raw):
        loc = f"{location}[{i}]"
        if isinstance(c, Control):
            c = c.to_json()
        if not isinstance(c, dict) or set(c) - set(CONTROL_KEYS):
            raise _err("INVALID_RIG", f"a control is {{{', '.join(CONTROL_KEYS)}}}", loc)
        name = P.check_name(c.get("name"), "a control name", f"{loc}.name")
        if name in names:
            raise _err("INVALID_RIG", f"two controls are named {name!r}", f"{loc}.name")
        names.add(name)
        kind = c.get("kind")
        if kind not in CONTROL_KINDS:
            raise _err("INVALID_RIG", f"kind is one of {', '.join(CONTROL_KINDS)}", f"{loc}.kind")
        obj = c.get("object")
        if obj is not None and (not isinstance(obj, str) or not obj):
            raise _err("INVALID_RIG", "object is a scene object id", f"{loc}.object")
        if kind == "fk" and obj is not None:
            raise _err("INVALID_RIG", "an fk control is its joint (no object)", f"{loc}.object")
        if kind != "fk" and obj is None:
            raise _err("INVALID_RIG", f"a {kind} control is a scene object", f"{loc}.object")
        joint = _opt_str(c, "joint", loc, "a joint name")
        if joint is not None and rig.index(joint) is None:
            raise _err("INVALID_RIG", f"no joint {joint!r} in the rig", f"{loc}.joint")
        if kind == "fk" and joint is None:
            raise _err("INVALID_RIG", "an fk control names its joint", f"{loc}.joint")
        constraint = _opt_str(c, "constraint", loc, "a constraint name")
        if constraint is not None and constraint not in constraint_names:
            raise _err("INVALID_RIG", f"no rig constraint {constraint!r}", f"{loc}.constraint")
        group = c.get("group", "")
        if not isinstance(group, str) or len(group) > 64:
            raise _err("INVALID_RIG", "group is a short name", f"{loc}.group")
        side = c.get("side")
        if side not in (None, "L", "R"):
            raise _err("INVALID_RIG", "side is 'L', 'R' or null", f"{loc}.side")
        shape = c.get("shape", "sphere")
        if shape not in CONTROL_SHAPES:
            raise _err(
                "INVALID_RIG", f"shape is one of {', '.join(CONTROL_SHAPES)}", f"{loc}.shape"
            )
        size = c.get("size", 0.05)
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or not (math.isfinite(size) and size > 0)
        ):
            raise _err("INVALID_RIG", "size is a positive number", f"{loc}.size")
        mirror = _opt_str(c, "mirror", loc, "a control name")
        out.append(
            Control(name, kind, obj, joint, constraint, group, side, shape, float(size), mirror)
        )
    for i, c in enumerate(out):
        if c.mirror is not None and c.mirror not in names:
            raise _err("INVALID_RIG", f"no control {c.mirror!r} to mirror", f"{location}[{i}]")
    return tuple(out)


def _parse_correctives(raw: Any, rig: Rig, location: str) -> tuple[Corrective, ...]:
    from .limits import AXES

    if not isinstance(raw, list) or len(raw) > MAX_CORRECTIVES:
        raise _err("INVALID_RIG", f"correctives is a list of at most {MAX_CORRECTIVES}", location)
    out = []
    names: set[str] = set()
    for i, c in enumerate(raw):
        loc = f"{location}[{i}]"
        if isinstance(c, Corrective):
            c = c.to_json()
        if not isinstance(c, dict) or set(c) - set(CORRECTIVE_KEYS):
            raise _err("INVALID_RIG", f"a corrective is {{{', '.join(CORRECTIVE_KEYS)}}}", loc)
        name = P.check_name(c.get("name"), "a corrective name", f"{loc}.name")
        if name in names:
            raise _err("INVALID_RIG", f"two correctives are named {name!r}", f"{loc}.name")
        names.add(name)
        mesh = c.get("mesh")
        if not isinstance(mesh, str) or not mesh:
            raise _err("INVALID_RIG", "mesh is a mesh object id", f"{loc}.mesh")
        target = P.check_name(c.get("target"), "a morph target name", f"{loc}.target")
        joint = P.check_name(c.get("joint"), "a joint name", f"{loc}.joint")
        if rig.index(joint) is None:
            raise _err("INVALID_RIG", f"no joint {joint!r} in the rig", f"{loc}.joint")
        driver = c.get("driver", "rotation")
        if driver not in CORRECTIVE_DRIVERS:
            raise _err(
                "INVALID_RIG",
                f"driver is one of {', '.join(CORRECTIVE_DRIVERS)}",
                f"{loc}.driver",
            )
        axis = c.get("axis", "+y" if driver in ("swing", "twist") else "+x")
        if axis not in AXES:
            raise _err("INVALID_RIG", f"axis is one of {', '.join(AXES)}", f"{loc}.axis")
        direction = c.get("direction")
        if direction is not None and (
            driver != "swing" or direction not in AXES or direction[1] == axis[1]
        ):
            raise _err(
                "INVALID_RIG",
                "direction (a swing's signed part) is an axis other than the bone axis",
                f"{loc}.direction",
            )
        rng = c.get("range", [0.0, 90.0])
        if (
            not isinstance(rng, (list, tuple))
            or len(rng) != 2
            or not all(
                not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                for v in rng
            )
            or rng[0] == rng[1]
        ):
            raise _err("INVALID_RIG", "range is [from, to] degrees (different)", f"{loc}.range")
        curve = c.get("curve", "smooth")
        if curve not in ("linear", "smooth"):
            raise _err("INVALID_RIG", "curve is 'linear' or 'smooth'", f"{loc}.curve")
        weight = c.get("weight", 1.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
        ):
            raise _err("INVALID_RIG", "weight is a finite number", f"{loc}.weight")
        out.append(
            Corrective(
                name,
                mesh,
                target,
                joint,
                driver,
                axis,
                direction,
                (float(rng[0]), float(rng[1])),
                curve,
                float(weight),
            )
        )
    return tuple(out)


def _parse_landmarks(raw: Any, location: str) -> tuple:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):  # (the dataclass form: pairs)
        raw = dict(raw)
    if not isinstance(raw, dict) or len(raw) > MAX_LANDMARKS:
        raise _err(
            "INVALID_RIG", f"landmarks map names to [x, y, z] (at most {MAX_LANDMARKS})", location
        )
    out = []
    for name, p in raw.items():
        P.check_name(name, "a landmark name", f"{location}.{name}")
        out.append((name, _vec(p, 3, f"{location}.{name}", "a landmark")))
    return tuple(out)


def _parse_info(raw: Any, location: str) -> str:
    from .limits import AXES

    if raw is None:
        return "null"
    if isinstance(raw, str) and raw.startswith("{"):
        raw = json.loads(raw)
    if not isinstance(raw, dict) or set(raw) - set(INFO_KEYS):
        raise _err("INVALID_RIG", f"info has the keys {', '.join(INFO_KEYS)}", location)
    out: dict[str, Any] = {}
    for key, v in raw.items():
        if key in ("forward", "up"):
            if v not in AXES:
                raise _err("INVALID_RIG", f"{key} is one of {', '.join(AXES)}", f"{location}.{key}")
        elif not isinstance(v, str) or len(v) > 2000:
            raise _err("INVALID_RIG", f"{key} is a text of at most 2000 characters", location)
        out[key] = v
    if "forward" in out and "up" in out and out["forward"][1] == out["up"][1]:
        raise _err("INVALID_RIG", "forward and up must be different axes", location)
    return json.dumps(out, sort_keys=True, separators=(",", ":")) if out else "null"


def _check_limits(raw: Any, location: str) -> str:
    if isinstance(raw, dict) and set(raw) == {"rotation"}:
        return _check_limits_euler(raw, location)  # (Stage 17's form, unchanged)
    from .limits import check_limits

    return check_limits(raw, location)


def _check_limits_euler(raw: Any, location: str) -> str:
    if not isinstance(raw, dict) or set(raw) - {"rotation"}:
        raise _err("INVALID_RIG", "limits are {rotation: {min, max}} (Euler XYZ degrees)", location)
    rot = raw.get("rotation")
    if not isinstance(rot, dict) or set(rot) - {"min", "max"}:
        raise _err("INVALID_RIG", "rotation limits are {min, max}", f"{location}.rotation")
    out: dict[str, Any] = {}
    for key in ("min", "max"):
        v = rot.get(key, [None, None, None])
        if not isinstance(v, list) or len(v) != 3:
            raise _err(
                "INVALID_RIG", f"{key} is [x, y, z] (null: no bound)", f"{location}.rotation.{key}"
            )
        out[key] = [
            None if x is None else float(_vec([x], 1, f"{location}.rotation.{key}", key)[0])
            for x in v
        ]
    for i in range(3):
        if (
            out["min"][i] is not None
            and out["max"][i] is not None
            and out["min"][i] > out["max"][i]
        ):
            raise _err("INVALID_RIG", "a rotation limit's min exceeds its max", location)
    return json.dumps({"rotation": out}, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------------------- #


def local_matrices(pos: np.ndarray, rot: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """``T R S`` per joint with the scene's arithmetic (rotations as stored canonically)."""
    from ..rendering.scene_data import _compose_locals
    from .system import canonical_quats

    if pos.shape[0] == 0:
        return np.zeros((0, 4, 4))
    return _compose_locals(pos, canonical_quats(rot), scl)


def forward_kinematics(local: np.ndarray, parents: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Joint matrices in armature space (``parent @ local``, parents first), with the
    scene hierarchy's arithmetic (and its native kernel when installed)."""
    from ..rendering.scene_data import _world_from_local

    if local.shape[0] == 0:
        return local.copy()
    return _world_from_local(
        np.ascontiguousarray(local).copy(),
        np.asarray(parents, np.int64),
        np.asarray(depth, np.int64),
    )


# --------------------------------------------------------------------------------------- #
# Skins
# --------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Skin:
    """Smooth binding of a mesh object to a rig (immutable)."""

    object: str  # the mesh object
    rig: str  # the rig item id
    joints: tuple[str, ...]  # joint ids, palette order
    inverse_bind: np.ndarray  # (m, 4, 4): mesh space at bind -> joint space at bind
    indices: np.ndarray  # (n, k) int32 into joints
    weights: np.ndarray  # (n, k) float64, rows summing to 1
    method: str = "linear"
    dq: np.ndarray | None = None  # (n,) float64 0..1: the DQS share per vertex ("blended")
    locked_joints: tuple[str, ...] = ()  # joint ids whose weights edits keep
    locked_vertices: np.ndarray | None = None  # (l,) int32 sorted: vertices edits keep
    # "joints": rest normals carried by the blended joint matrices (Stage 17, glTF's
    # rule); "surface": rest normals turned as the deformed surface turns around each
    # vertex (creases and twists shade as they are shaped).
    normals: str = "joints"
    smooth: str = "null"  # canonical JSON of the corrective smoothing (``SMOOTH_KEYS``)

    @property
    def vertex_count(self) -> int:
        return int(self.indices.shape[0])

    def dense(self) -> np.ndarray:
        """``(n, m)`` weights over the palette (for editing and inspection)."""
        n, m = self.vertex_count, len(self.joints)
        out = np.zeros((n, m))
        rows = np.repeat(np.arange(n), self.indices.shape[1])
        np.add.at(out, (rows, self.indices.reshape(-1)), self.weights.reshape(-1))
        return out

    def to_json(self, rig: Rig | None = None) -> dict[str, Any]:
        names = (
            [rig.joints[rig.index_of_id(j)].name for j in self.joints]  # type: ignore[index]
            if rig is not None and all(rig.index_of_id(j) is not None for j in self.joints)
            else list(self.joints)
        )
        return {
            "object": self.object,
            "rig": self.rig,
            "joints": list(self.joints),
            "joint_names": names,
            "inverse_bind": self.inverse_bind.reshape(-1, 16).tolist(),
            "indices": self.indices.tolist(),
            "weights": self.weights.tolist(),
            "method": self.method,
            **({} if self.dq is None else {"dq": self.dq.tolist()}),
            **({} if self.normals == "joints" else {"normals": self.normals}),
            **({} if self.smooth == "null" else {"smooth": json.loads(self.smooth)}),
            **(
                {}
                if not self.locked_joints and self.locked_vertices is None
                else {
                    "locks": {
                        "joints": list(self.locked_joints),
                        "vertices": []
                        if self.locked_vertices is None
                        else self.locked_vertices.tolist(),
                    }
                }
            ),
        }

    def same(self, other: Skin) -> bool:
        def arrays(a: np.ndarray | None, b: np.ndarray | None) -> bool:
            return (a is None) == (b is None) and (a is None or a.tobytes() == b.tobytes())

        return (
            self.object == other.object
            and self.rig == other.rig
            and self.joints == other.joints
            and self.method == other.method
            and self.inverse_bind.tobytes() == other.inverse_bind.tobytes()
            and self.indices.tobytes() == other.indices.tobytes()
            and self.weights.tobytes() == other.weights.tobytes()
            and arrays(self.dq, other.dq)
            and self.locked_joints == other.locked_joints
            and arrays(self.locked_vertices, other.locked_vertices)
            and self.normals == other.normals
            and self.smooth == other.smooth
        )

    def replace(self, **changes: Any) -> Skin:
        fields = {f: getattr(self, f) for f in Skin.__dataclass_fields__}
        fields.update(changes)
        return Skin(**fields)


def make_skin(
    obj: str,
    rig_id: str,
    joints: list[str],
    inverse_bind: Any,
    indices: Any,
    weights: Any,
    method: str = "linear",
    location: str = "skin",
    *,
    dq: Any = None,
    locks: Any = None,
    normals: str = "joints",
    smooth: Any = None,
) -> Skin:
    """A checked skin: weights non-negative, normalized per vertex (rows summing to 0 are
    refused), at most :data:`MAX_INFLUENCES` per vertex, indices within the palette;
    ``dq`` (``blended``: each vertex's DQS share, 0..1), ``locks`` (``{"joints": [ids],
    "vertices": [indices]}``: what weight edits keep), ``normals`` (``joints`` or
    ``surface``), ``smooth`` (corrective smoothing: :func:`check_smooth`)."""
    m = len(joints)
    if not 1 <= m <= MAX_JOINTS or len(set(joints)) != m:
        raise _err(
            "INVALID_SKIN", f"a skin uses 1..{MAX_JOINTS} distinct joints", f"{location}.joints"
        )
    ib = (
        np.asarray(inverse_bind, np.float64).reshape(-1, 4, 4)
        if np.size(inverse_bind)
        else np.zeros((0, 4, 4))
    )
    if ib.shape != (m, 4, 4) or not np.all(np.isfinite(ib)):
        raise _err(
            "INVALID_SKIN",
            "inverse_bind holds one finite 4x4 matrix per joint",
            f"{location}.inverse_bind",
        )
    idx = np.asarray(indices)
    w = np.asarray(weights, np.float64)
    if idx.ndim == 1:
        idx = idx[:, None]
    if w.ndim == 1:
        w = w[:, None]
    if idx.shape != w.shape or idx.ndim != 2 or not 1 <= idx.shape[1] <= MAX_INFLUENCES:
        raise _err(
            "INVALID_SKIN",
            f"indices and weights are (vertices, 1..{MAX_INFLUENCES}) arrays",
            location,
        )
    if idx.shape[0] > MAX_SKIN_VERTICES:
        raise _err("LIMIT_EXCEEDED", f"more than {MAX_SKIN_VERTICES} skinned vertices", location)
    if idx.dtype.kind not in "iu" or (idx.size and (idx.min() < 0 or idx.max() >= m)):
        raise _err(
            "INVALID_SKIN", "indices name joints of the skin (0..joints-1)", f"{location}.indices"
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise _err("INVALID_SKIN", "weights must be finite and non-negative", f"{location}.weights")
    total = w.sum(axis=1)
    if np.any(total <= 0):
        k = int(np.flatnonzero(total <= 0)[0])
        raise _err("INVALID_SKIN", f"vertex {k} has no weight", f"{location}.weights[{k}]")
    w = w / total[:, None]
    if method not in METHODS:
        raise _err("INVALID_SKIN", f"method is one of {', '.join(METHODS)}", f"{location}.method")
    idx = idx.astype(np.int32)
    n = idx.shape[0]
    dq_arr = None
    if method == "blended":
        if dq is None:
            raise _err(
                "INVALID_SKIN", "a blended skin gives dq: each vertex's DQS share 0..1", location
            )
        dq_arr = np.asarray(dq, np.float64).reshape(-1)
        if dq_arr.shape != (n,) or not np.all(np.isfinite(dq_arr)):
            raise _err("INVALID_SKIN", f"dq is {n} finite numbers", f"{location}.dq")
        if dq_arr.size and (dq_arr.min() < 0.0 or dq_arr.max() > 1.0):
            raise _err("INVALID_SKIN", "dq values are within 0..1", f"{location}.dq")
        dq_arr = dq_arr.copy()
    elif dq is not None:
        raise _err("INVALID_SKIN", "dq applies to method 'blended'", f"{location}.dq")
    locked_joints: tuple[str, ...] = ()
    locked_vertices = None
    if locks is not None:
        if not isinstance(locks, dict) or set(locks) - {"joints", "vertices"}:
            raise _err("INVALID_SKIN", "locks are {joints, vertices}", f"{location}.locks")
        lj = locks.get("joints", [])
        if not isinstance(lj, (list, tuple)) or any(j not in joints for j in lj):
            raise _err(
                "INVALID_SKIN", "locked joints are joint ids of the skin", f"{location}.locks"
            )
        locked_joints = tuple(sorted(set(lj), key=list(joints).index))
        lv = locks.get("vertices", [])
        arr = np.asarray(lv if lv is not None else [], np.int64).reshape(-1)
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            raise _err("INVALID_SKIN", f"locked vertices are 0..{n - 1}", f"{location}.locks")
        if arr.size:
            locked_vertices = np.unique(arr).astype(np.int32)
    if normals not in NORMALS:
        raise _err("INVALID_SKIN", f"normals is one of {', '.join(NORMALS)}", f"{location}.normals")
    smooth_json = check_smooth(smooth, f"{location}.smooth")
    for a in (ib, idx, w, dq_arr, locked_vertices):
        if a is not None:
            a.flags.writeable = False
    return Skin(
        obj, rig_id, tuple(joints), ib, idx, w, method, dq_arr, locked_joints, locked_vertices,
        normals, smooth_json,
    )  # fmt: skip


NORMALS = ("joints", "surface")
SMOOTH_KEYS = ("iterations", "strength", "region", "weights")
SMOOTH_REGIONS = ("blend", "all")
SMOOTH_WEIGHTS = ("cotangent", "uniform")


def check_smooth(raw: Any, location: str = "smooth") -> str:
    """Corrective smoothing of a skin as canonical JSON (``"null"``: none):
    ``{"iterations": 1..100 (default 8), "strength": 0..1 (default 0.5), "region":
    "blend" (where weights blend, fading out where one joint holds a vertex: the
    default) or "all", "weights": "cotangent" (the default: smoothing along the surface,
    no drift across it) or "uniform"}`` -- see :func:`delta_mush`."""
    if raw is None:
        return "null"
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(SMOOTH_KEYS):
        raise _err("INVALID_SKIN", f"smooth is {{{', '.join(SMOOTH_KEYS)}}} or null", location)
    it = raw.get("iterations", 8)
    if isinstance(it, bool) or not isinstance(it, int) or not 1 <= it <= 100:
        raise _err("INVALID_SKIN", "smooth.iterations is 1..100", f"{location}.iterations")
    st = raw.get("strength", 0.5)
    if isinstance(st, bool) or not isinstance(st, (int, float)) or not 0.0 < float(st) <= 1.0:
        raise _err("INVALID_SKIN", "smooth.strength is in (0, 1]", f"{location}.strength")
    region = raw.get("region", "blend")
    if region not in SMOOTH_REGIONS:
        raise _err(
            "INVALID_SKIN",
            f"smooth.region is one of {', '.join(SMOOTH_REGIONS)}",
            f"{location}.region",
        )
    weights = raw.get("weights", "cotangent")
    if weights not in SMOOTH_WEIGHTS:
        raise _err(
            "INVALID_SKIN",
            f"smooth.weights is one of {', '.join(SMOOTH_WEIGHTS)}",
            f"{location}.weights",
        )
    return json.dumps(
        {"iterations": it, "strength": float(st), "region": region, "weights": weights},
        sort_keys=True,
        separators=(",", ":"),
    )


def skin_linear(
    positions: np.ndarray,
    normals: np.ndarray | None,
    indices: np.ndarray,
    weights: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Linear blend skinning (the NumPy reference; float32 in and out, float64 inside).

    Per vertex: ``M = w_0 P_0 + w_1 P_1 + ...`` (in influence order), ``v' = M v``
    (``((m00 x + m01 y) + m02 z) + m03`` per row), ``n' = cof(M_3x3) n`` normalized.
    ``palette`` is ``(m, 3, 4)`` (or ``(m, 4, 4)``)."""
    from .. import _native

    pal = np.ascontiguousarray(palette[:, :3, :4], np.float64)
    n = positions.shape[0]
    out_p = np.empty((n, 3), np.float32)
    out_n = None if normals is None else np.empty((n, 3), np.float32)
    native = _native.module()
    if native is not None and hasattr(native, "skin_linear"):
        native.skin_linear(
            np.ascontiguousarray(positions, np.float32),
            None if normals is None else np.ascontiguousarray(normals, np.float32),
            np.ascontiguousarray(indices, np.int32),
            np.ascontiguousarray(weights, np.float64),
            pal,
            out_p,
            out_n,
            _native.threads(),
        )
        return out_p, out_n
    k = indices.shape[1]
    for start in range(0, n, SKIN_CHUNK):
        sl = slice(start, min(n, start + SKIN_CHUNK))
        idx = indices[sl]
        w = weights[sl]
        m = w[:, 0, None, None] * pal[idx[:, 0]]
        for j in range(1, k):
            m = m + w[:, j, None, None] * pal[idx[:, j]]
        p = positions[sl].astype(np.float64)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        out_p[sl, 0] = ((m[:, 0, 0] * x + m[:, 0, 1] * y) + m[:, 0, 2] * z) + m[:, 0, 3]
        out_p[sl, 1] = ((m[:, 1, 0] * x + m[:, 1, 1] * y) + m[:, 1, 2] * z) + m[:, 1, 3]
        out_p[sl, 2] = ((m[:, 2, 0] * x + m[:, 2, 1] * y) + m[:, 2, 2] * z) + m[:, 2, 3]
        if normals is not None:
            nr = normals[sl].astype(np.float64)
            out_n[sl] = _cofactor_normals(m, nr)  # type: ignore[index]
    return out_p, out_n


def _cofactor_normals(m: np.ndarray, nr: np.ndarray) -> np.ndarray:
    """``cof(M) n`` normalized (``cof`` = det x inverse transpose: right for any
    invertible linear part, mirrored ones included)."""
    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    c00, c01, c02 = e * i - f * h, f * g - d * i, d * h - e * g
    c10, c11, c12 = c * h - b * i, a * i - c * g, b * g - a * h
    c20, c21, c22 = b * f - c * e, c * d - a * f, a * e - b * d
    x, y, z = nr[:, 0], nr[:, 1], nr[:, 2]
    nx = (c00 * x + c01 * y) + c02 * z
    ny = (c10 * x + c11 * y) + c12 * z
    nz = (c20 * x + c21 * y) + c22 * z
    det = (a * c00 + b * c01) + c * c02
    sign = np.where(det < 0.0, -1.0, 1.0)  # a mirrored blend keeps normals outward
    length = np.sqrt((nx * nx + ny * ny) + nz * nz)
    inv = np.where(length > 0.0, sign / np.where(length > 0.0, length, 1.0), 0.0)
    return np.stack([nx * inv, ny * inv, nz * inv], axis=1).astype(np.float32)


def dual_quaternion_palette(palette: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per joint: the rotation ``q`` (unit quaternion, ``x, y, z, w``), the dual part
    ``d = 0.5 (t, 0) q`` and the scale/shear ``K = R^T L`` (row-major 3x3) of each skinning
    matrix ``S = T R K`` (``R`` the closest rotation to ``L``, by the polar
    decomposition; Kavan et al. 2008's scale stage)."""
    pal = np.asarray(palette[:, :3, :4], np.float64)
    m = pal.shape[0]
    lin = pal[:, :, :3]
    u, _, vt = np.linalg.svd(lin)
    rot = u @ vt
    flip = np.linalg.det(rot) < 0
    if np.any(flip):
        u[flip, :, -1] *= -1.0
        rot = u @ vt
    k_mat = np.einsum("nji,njk->nik", rot, lin)  # R^T lin
    q = R.from_matrices(rot)  # (m, 4)
    t = pal[:, :, 3]
    tq = np.concatenate([t, np.zeros((m, 1))], axis=1)
    d = 0.5 * R.multiply(tq, q)
    return (
        np.ascontiguousarray(q, np.float64),
        np.ascontiguousarray(d, np.float64),
        np.ascontiguousarray(k_mat.reshape(m, 9), np.float64),
    )


def skin_dual_quaternion(
    positions: np.ndarray,
    normals: np.ndarray | None,
    indices: np.ndarray,
    weights: np.ndarray,
    palette: np.ndarray,
    blend: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dual quaternion skinning (Kavan et al. 2008) with Kavan's scale stage: each
    skinning matrix ``S = T R K`` is split into a rigid part ``(R, T)`` and a scale/shear
    part ``K``; ``K`` is blended linearly and applied first, the rigid parts as dual
    quaternions (antipodality resolved against the first influence). Normals: the
    cofactor of the blended ``K`` (signed by its determinant), then the rotation.

    ``blend`` (per vertex, 0..1) mixes with linear blend skinning: ``p = p_lbs + b (p_dq -
    p_lbs)`` (exactly LBS at 0 and DQS at 1), normals likewise, renormalized.

    The NumPy reference: per vertex in a fixed operation order (no reductions, no BLAS),
    which the native kernel repeats bit for bit."""
    from .. import _native

    q, d, kmat = dual_quaternion_palette(palette)
    pal = np.ascontiguousarray(palette[:, :3, :4], np.float64).reshape(-1, 12)
    n = positions.shape[0]
    out_p = np.empty((n, 3), np.float32)
    out_n = None if normals is None else np.empty((n, 3), np.float32)
    native = _native.module()
    if native is not None and hasattr(native, "skin_dual_quaternion"):
        native.skin_dual_quaternion(
            np.ascontiguousarray(positions, np.float32),
            None if normals is None else np.ascontiguousarray(normals, np.float32),
            np.ascontiguousarray(indices, np.int32),
            np.ascontiguousarray(weights, np.float64),
            q,
            d,
            kmat,
            None if blend is None else pal,
            None if blend is None else np.ascontiguousarray(blend, np.float64),
            out_p,
            out_n,
            _native.threads(),
        )
        return out_p, out_n
    for start in range(0, n, SKIN_CHUNK):
        sl = slice(start, min(n, start + SKIN_CHUNK))
        p64, n64 = _dq_chunk(
            positions[sl], None if normals is None else normals[sl], indices[sl], weights[sl],
            q, d, kmat,
        )  # fmt: skip
        if blend is not None:
            b = np.asarray(blend[sl], np.float64)
            pl, nl = _lbs_chunk(positions[sl], None if normals is None else normals[sl],
                                indices[sl], weights[sl], pal)  # fmt: skip
            p64 = _mix(pl, p64, b)
            if n64 is not None:
                n64 = _unit(_mix(nl, n64, b))
        out_p[sl] = p64.astype(np.float32)
        if n64 is not None:
            out_n[sl] = n64.astype(np.float32)  # type: ignore[index]
    return out_p, out_n


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = a + t[:, None] * (b - a)
    out = np.where((t == 0.0)[:, None], a, out)
    return np.where((t == 1.0)[:, None], b, out)


def _unit(v: np.ndarray) -> np.ndarray:
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    length = np.sqrt((x * x + y * y) + z * z)
    inv = np.where(length > 0.0, 1.0 / np.where(length > 0.0, length, 1.0), 0.0)
    return np.stack([x * inv, y * inv, z * inv], axis=1)


def _lbs_chunk(pos, nrm, idx, w, pal):
    """Linear blend skinning of a chunk in float64 (``skin_linear``'s arithmetic)."""
    k = idx.shape[1]
    m = w[:, 0, None] * pal[idx[:, 0]]
    for j in range(1, k):
        m = m + w[:, j, None] * pal[idx[:, j]]
    p = pos.astype(np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    out = np.stack(
        [
            ((m[:, 0] * x + m[:, 1] * y) + m[:, 2] * z) + m[:, 3],
            ((m[:, 4] * x + m[:, 5] * y) + m[:, 6] * z) + m[:, 7],
            ((m[:, 8] * x + m[:, 9] * y) + m[:, 10] * z) + m[:, 11],
        ],
        axis=1,
    )
    if nrm is None:
        return out, None
    a, b, c = m[:, 0], m[:, 1], m[:, 2]
    d_, e, f = m[:, 4], m[:, 5], m[:, 6]
    g, h, i = m[:, 8], m[:, 9], m[:, 10]
    return out, _cofactor_apply(a, b, c, d_, e, f, g, h, i, nrm.astype(np.float64))


def _cofactor_apply(a, b, c, d, e, f, g, h, i, nr):
    """``sign(det) cof(M) n`` normalized, in float64."""
    c00, c01, c02 = e * i - f * h, f * g - d * i, d * h - e * g
    c10, c11, c12 = c * h - b * i, a * i - c * g, b * g - a * h
    c20, c21, c22 = b * f - c * e, c * d - a * f, a * e - b * d
    x, y, z = nr[:, 0], nr[:, 1], nr[:, 2]
    nx = (c00 * x + c01 * y) + c02 * z
    ny = (c10 * x + c11 * y) + c12 * z
    nz = (c20 * x + c21 * y) + c22 * z
    det = (a * c00 + b * c01) + c * c02
    sign = np.where(det < 0.0, -1.0, 1.0)
    length = np.sqrt((nx * nx + ny * ny) + nz * nz)
    inv = np.where(length > 0.0, sign / np.where(length > 0.0, length, 1.0), 0.0)
    return np.stack([nx * inv, ny * inv, nz * inv], axis=1)


def _dq_chunk(pos, nrm, idx, w, q, d, kmat):
    """Dual quaternion skinning of a chunk in float64 (the reference operation order)."""
    k = idx.shape[1]
    i0 = idx[:, 0]
    q0x, q0y, q0z, q0w = q[i0, 0], q[i0, 1], q[i0, 2], q[i0, 3]
    zeros = np.zeros(idx.shape[0])
    rx, ry, rz, rw = zeros, zeros, zeros, zeros
    dx, dy, dz, dw = zeros, zeros, zeros, zeros
    kb = [zeros] * 9
    for j in range(k):
        ij = idx[:, j]
        wj = w[:, j]
        qx, qy, qz, qw = q[ij, 0], q[ij, 1], q[ij, 2], q[ij, 3]
        dot = ((qx * q0x + qy * q0y) + qz * q0z) + qw * q0w
        s = np.where(dot < 0.0, -1.0, 1.0) * wj
        rx, ry, rz, rw = rx + s * qx, ry + s * qy, rz + s * qz, rw + s * qw
        dx = dx + s * d[ij, 0]
        dy = dy + s * d[ij, 1]
        dz = dz + s * d[ij, 2]
        dw = dw + s * d[ij, 3]
        kb = [kb[e] + wj * kmat[ij, e] for e in range(9)]
    norm = np.sqrt(((rx * rx + ry * ry) + rz * rz) + rw * rw)
    norm = np.where(norm > 0.0, norm, 1.0)
    rx, ry, rz, rw = rx / norm, ry / norm, rz / norm, rw / norm
    dx, dy, dz, dw = dx / norm, dy / norm, dz / norm, dw / norm
    p = pos.astype(np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    px = (kb[0] * x + kb[1] * y) + kb[2] * z
    py = (kb[3] * x + kb[4] * y) + kb[5] * z
    pz = (kb[6] * x + kb[7] * y) + kb[8] * z
    ox, oy, oz = _rotate(rx, ry, rz, rw, px, py, pz)
    # translation = 2 * vector part of (dual * conj(real))
    cx, cy, cz, cw = -rx, -ry, -rz, rw
    tx = 2.0 * (((dw * cx + dx * cw) + dy * cz) - dz * cy)
    ty = 2.0 * (((dw * cy - dx * cz) + dy * cw) + dz * cx)
    tz = 2.0 * (((dw * cz + dx * cy) - dy * cx) + dz * cw)
    out = np.stack([ox + tx, oy + ty, oz + tz], axis=1)
    if nrm is None:
        return out, None
    nk = _cofactor_apply(*kb, nrm.astype(np.float64))
    nx, ny, nz = _rotate(rx, ry, rz, rw, nk[:, 0], nk[:, 1], nk[:, 2])
    return out, _unit(np.stack([nx, ny, nz], axis=1))


def _rotate(ux, uy, uz, w, vx, vy, vz):
    """``v`` rotated by the unit quaternion ``(u, w)`` (``rotation.rotate``'s order)."""
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)
    return (
        (vx + w * tx) + (uy * tz - uz * ty),
        (vy + w * ty) + (uz * tx - ux * tz),
        (vz + w * tz) + (ux * ty - uy * tx),
    )


def face_normal_sums(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per vertex, the sum of its triangles' cross products ``(b - a) x (c - a)`` (an
    area-weighted normal), accumulated corner by corner in triangle order."""
    p = np.asarray(p, np.float64)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    u, v = b - a, c - a
    fx = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    fy = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    fz = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    n = len(p)
    out = np.zeros((n, 3))
    for k, f in enumerate((fx, fy, fz)):
        out[:, k] = (np.bincount(t[:, 0], f, n) + np.bincount(t[:, 1], f, n)) + np.bincount(
            t[:, 2], f, n
        )
    return out


def surface_normals(
    rest_frame: np.ndarray, posed: np.ndarray, triangles: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Normals that turn as the deformed surface turns: each rest normal rotated by the
    smallest rotation taking the vertex's rest area-weighted normal ``rest_frame`` (unit)
    to its posed one. Identity at rest (the authored normals stay), and a crease or a
    twist shades as the facets around it are shaped; a vertex whose surface turned over
    (an antiparallel normal) takes the posed area-weighted normal itself."""
    b = face_normal_sums(posed, triangles)
    length = np.sqrt(np.sum(b * b, axis=1))
    b = np.where(length[:, None] > 0, b / np.where(length > 0, length, 1.0)[:, None], rest_frame)
    a = rest_frame
    x = np.asarray(normals, np.float64)
    c = np.sum(a * b, axis=1)
    v = np.cross(a, b)
    # R x = c x + v x x + v (v . x) / (1 + c)   (the rotation about a x b taking a to b)
    turned = (
        c[:, None] * x
        + np.cross(v, x)
        + v * (np.sum(v * x, axis=1) / np.maximum(1.0 + c, 1e-12))[:, None]
    )
    out = np.where((c > -0.95)[:, None], turned, b)
    norm = np.sqrt(np.sum(out * out, axis=1))
    return (out / np.where(norm > 0, norm, 1.0)[:, None]).astype(np.float32)


# --------------------------------------------------------------------------------------- #
# Corrective smoothing (delta mush)
# --------------------------------------------------------------------------------------- #

MUSH_BLEND = 0.15  # 1 - the largest weight at which a vertex is fully smoothed


def mush_edges(n: int, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The smoothing graph: directed edges ``(dst, src)`` (both ways of every mesh edge,
    sorted by ``dst`` then ``src``), each vertex's degree, and each vertex's first
    neighbour (the lowest-numbered, ``-1`` for none: the tangent's reference)."""
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    a = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    b = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)
    dst = np.concatenate([pairs[:, 0], pairs[:, 1]])
    src = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.lexsort((src, dst))
    dst, src = dst[order], src[order]
    deg = np.bincount(dst, minlength=n)
    first = np.full(n, -1, np.int64)
    starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]]) if dst.size else np.zeros(0, int)
    first[dst[starts]] = src[starts]
    return np.stack([dst, src]), deg, first


def mush_weights(
    p: np.ndarray, t: np.ndarray, graph: np.ndarray, deg: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """The smoothing's weight per directed edge of ``graph`` and their sum per vertex:
    ``uniform`` (1), or ``cotangent`` -- the rest surface's cotangent weights, negative
    ones (edges across obtuse corners) clamped to 0, so each step is an average that
    moves vertices along the surface's normal only where it curves: an irregular mesh's
    vertices do not drift along it, so the rest detail is small and lies across the
    surface (a vertex whose weights all vanish smooths uniformly)."""
    dst, src = graph
    n = len(p)
    if kind == "uniform":
        return np.ones(len(dst)), np.asarray(deg, np.float64)
    from .heat import laplacian

    rows, cols, vals, _ = laplacian(np.asarray(p, np.float64), t)
    at = np.searchsorted(dst * n + src, rows * n + cols)
    w = np.maximum(np.bincount(at, -vals, len(dst)), 0.0)
    wsum = np.bincount(dst, w, n)
    flat = (wsum <= 0.0) & (np.asarray(deg) > 0)
    if flat.any():
        w = np.where(flat[dst], 1.0, w)
        wsum = np.bincount(dst, w, n)
    return w, wsum


def _mush_smooth(
    x: np.ndarray, graph: np.ndarray, w: np.ndarray, wsum: np.ndarray, it: int, lam: float
):
    """``it`` rounds of ``x += lam (weighted mean of neighbours - x)``; each sum in edge
    order (a bincount: what the native kernel's loop repeats)."""
    dst, src = graph
    n = len(x)
    has = wsum > 0
    div = np.where(has, wsum, 1.0)
    out = np.array(x, np.float64)
    for _ in range(it):
        nxt = out.copy()
        for k in range(3):
            mean = np.bincount(dst, w * out[src, k], n) / div
            nxt[:, k] = np.where(has, out[:, k] + lam * (mean - out[:, k]), out[:, k])
        out = nxt
    return out


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _unit3(x, y, z):
    length = np.sqrt((x * x + y * y) + z * z)
    ok = length > 0.0
    inv = np.where(ok, 1.0 / np.where(ok, length, 1.0), 0.0)
    return x * inv, y * inv, z * inv, length


def _mush_corners(sm: np.ndarray, t: np.ndarray):
    """Per triangle corner (corner ``c`` of triangle ``f`` at ``3 f + c``) an orthonormal
    frame of the smoothed surface -- the corner's bisector, the face normal x bisector,
    the face normal: nine arrays -- and its weight, the sine of the corner's angle (0 for
    a degenerate corner, whose frame is undefined)."""
    v = t.reshape(-1)
    a = t[:, [1, 2, 0]].reshape(-1)
    b = t[:, [2, 0, 1]].reshape(-1)
    ux, uy, uz, _ = _unit3(sm[a, 0] - sm[v, 0], sm[a, 1] - sm[v, 1], sm[a, 2] - sm[v, 2])
    wx, wy, wz, _ = _unit3(sm[b, 0] - sm[v, 0], sm[b, 1] - sm[v, 1], sm[b, 2] - sm[v, 2])
    nx, ny, nz, sine = _unit3(*_cross(ux, uy, uz, wx, wy, wz))
    xx, xy, xz, _ = _unit3(ux + wx, uy + wy, uz + wz)
    yx, yy, yz = _cross(nx, ny, nz, xx, xy, xz)
    return (xx, xy, xz, yx, yy, yz, nx, ny, nz), sine


def mush_rest(
    positions: np.ndarray, triangles: np.ndarray, weights: np.ndarray, spec: dict[str, Any]
) -> dict[str, Any]:
    """What delta mush keeps of the rest pose: the smoothing graph, the rest detail (each
    vertex's offset from the smoothed rest surface, in the frame of each of its triangle
    corners) and the smoothing mask (``region`` ``blend``: full where the vertex's largest
    weight is at most ``1 - MUSH_BLEND``, none where one joint holds it, smooth in between
    and spread over the surface; ``all``: everywhere)."""
    p0 = np.asarray(positions, np.float64)
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    n = len(p0)
    graph, deg, _ = mush_edges(n, t)
    w, wsum = mush_weights(p0, t, graph, deg, spec.get("weights", "cotangent"))
    it, lam = int(spec["iterations"]), float(spec["strength"])
    sm = _mush_smooth(p0, graph, w, wsum, it, lam)
    (xx, xy, xz, yx, yy, yz, nx, ny, nz), _ = _mush_corners(sm, t)
    v = t.reshape(-1)
    dx, dy, dz = p0[v, 0] - sm[v, 0], p0[v, 1] - sm[v, 1], p0[v, 2] - sm[v, 2]
    delta = np.stack(
        [
            (xx * dx + xy * dy) + xz * dz,
            (yx * dx + yy * dy) + yz * dz,
            (nx * dx + ny * dy) + nz * dz,
        ],
        axis=1,
    )
    if spec.get("region", "blend") == "all":
        mask = np.ones(n)
    else:
        top = np.asarray(weights, np.float64).max(axis=1)
        u = np.clip((1.0 - top) / MUSH_BLEND, 0.0, 1.0)
        mask = u * u * (3.0 - 2.0 * u)
        spread = _mush_smooth(np.stack([mask, mask, mask], axis=1), graph, w, wsum, it, 0.5)
        mask = np.maximum(mask, np.clip(spread[:, 0] * 1.5, 0.0, 1.0))
    return {"graph": np.ascontiguousarray(graph), "weights": w, "wsum": wsum, "triangles": t,
            "delta": delta, "mask": mask, "iterations": it, "strength": lam}  # fmt: skip


def delta_mush(positions: np.ndarray, rest: dict[str, Any]) -> np.ndarray:
    """Delta mush of skinned positions (Mancewicz et al. 2014), with the per-corner frames
    of Blender's Corrective Smooth: the posed surface is smoothed as the rest one was, and
    each vertex's rest detail is added back in the frames of its triangle corners on the
    smoothed posed surface, averaged by the corners' weights (one frame per vertex turns
    with whichever edge defines it, and shears put folds in the detail), then blended by
    the mask -- LBS pinching, folds and crumples relax while the modeled detail stays. The
    input at rest (to rounding). The NumPy reference of the native kernel (the same
    operations in the same order: bitwise the same result)."""
    from .. import _native

    p = np.ascontiguousarray(positions, np.float64)
    t = rest["triangles"]
    native = _native.module()
    if native is not None and hasattr(native, "delta_mush"):
        out = np.empty((len(p), 3), np.float32)
        native.delta_mush(
            p, rest["graph"][0], rest["graph"][1], rest["weights"], rest["wsum"],
            np.ascontiguousarray(t, np.int64), np.ascontiguousarray(rest["delta"], np.float64),
            np.ascontiguousarray(rest["mask"], np.float64), int(rest["iterations"]),
            float(rest["strength"]), out, _native.threads(),
        )  # fmt: skip
        return out
    n = len(p)
    sm = _mush_smooth(
        p, rest["graph"], rest["weights"], rest["wsum"], rest["iterations"], rest["strength"]
    )
    (xx, xy, xz, yx, yy, yz, nx, ny, nz), w = _mush_corners(sm, t)
    d = rest["delta"]
    d0, d1, d2 = d[:, 0], d[:, 1], d[:, 2]
    v = t.reshape(-1)
    wsum = np.bincount(v, w, n)
    has = wsum > 0.0
    div = np.where(has, wsum, 1.0)
    o = []
    for k, (ex, ey, en) in enumerate(((xx, yx, nx), (xy, yy, ny), (xz, yz, nz))):
        acc = np.bincount(v, w * ((ex * d0 + ey * d1) + en * d2), n)
        o.append(np.where(has, sm[:, k] + acc / div, p[:, k]))
    m = rest["mask"]
    out = np.stack([p[:, k] + m * (o[k] - p[:, k]) for k in range(3)], axis=1)
    return out.astype(np.float32)


def joint_limits(rig: Rig, k: int) -> dict[str, Any] | None:
    limits = rig.joints[k].limits
    return None if limits == "null" else json.loads(limits)


__all__ = [
    "Joint",
    "METHODS",
    "Rig",
    "Skin",
    "forward_kinematics",
    "local_matrices",
    "make_skin",
    "parse_rig",
    "skin_dual_quaternion",
    "skin_linear",
]


# --------------------------------------------------------------------------------------- #
# Morph targets
# --------------------------------------------------------------------------------------- #

MAX_MORPH_TARGETS = 256


@dataclass(frozen=True)
class Morph:
    """Morph targets (blend shapes) of a mesh object: per target, position (and optional
    normal) offsets for every vertex; ``A' = A + sum_m w_m dA_m`` (glTF's rule), applied
    before skinning. Weights animate as ``morph["NAME"]``; ``defaults`` are the rest
    weights."""

    object: str
    names: tuple[str, ...]
    positions: np.ndarray  # (t, n, 3) float32 offsets
    normals: np.ndarray | None  # (t, n, 3) float32 offsets or None
    defaults: tuple[float, ...]

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[1])

    def apply(self, mesh: Any, weights: dict[str, float]) -> Any:
        """``mesh`` with the targets blended in (``None`` when every weight is 0)."""
        from ..mesh import Mesh

        if mesh is None or mesh.vertex_count != self.vertex_count:
            return None
        w = np.array([weights.get(n, d) for n, d in zip(self.names, self.defaults, strict=True)])
        if not np.any(w):
            return None
        pos = mesh.positions.astype(np.float64)
        for k in np.flatnonzero(w).tolist():
            pos = pos + w[k] * self.positions[k].astype(np.float64)
        nrm = mesh.normals
        if nrm is not None and self.normals is not None:
            nn = nrm.astype(np.float64)
            for k in np.flatnonzero(w).tolist():
                nn = nn + w[k] * self.normals[k].astype(np.float64)
            length = np.linalg.norm(nn, axis=1)
            nrm = (nn / np.where(length > 0, length, 1.0)[:, None]).astype(np.float32)
        return Mesh._from_arrays(
            pos.astype(np.float32), mesh.triangles, nrm, mesh.uvs, mesh.groups, mesh.colors
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "object": self.object,
            "targets": [
                {
                    "name": n,
                    "positions": self.positions[k].tolist(),
                    **({} if self.normals is None else {"normals": self.normals[k].tolist()}),
                    **({} if self.defaults[k] == 0.0 else {"default": self.defaults[k]}),
                }
                for k, n in enumerate(self.names)
            ],
        }
        return out

    def same(self, other: Morph) -> bool:
        return (
            self.object == other.object
            and self.names == other.names
            and self.defaults == other.defaults
            and self.positions.tobytes() == other.positions.tobytes()
            and (self.normals is None) == (other.normals is None)
            and (self.normals is None or self.normals.tobytes() == other.normals.tobytes())
        )


def make_morph(obj: str, targets: Any, vertex_count: int, location: str = "morph") -> Morph:
    """Checked morph targets: ``[{name, positions: (n, 3) offsets, normals?, default?}]``."""
    if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_MORPH_TARGETS:
        raise _err("INVALID_MORPH", f"1..{MAX_MORPH_TARGETS} targets", location)
    names, pos, nrm, defaults = [], [], [], []
    with_normals = None
    for i, t in enumerate(targets):
        loc = f"{location}.targets[{i}]"
        if not isinstance(t, dict) or set(t) - {"name", "positions", "normals", "default"}:
            raise _err("INVALID_MORPH", "a target is {name, positions, normals, default}", loc)
        name = P.check_name(t.get("name"), "a morph target name", f"{loc}.name")
        if name in names:
            raise _err("INVALID_MORPH", f"two targets are named {name!r}", f"{loc}.name")
        p = np.asarray(t.get("positions"), np.float64)
        if p.shape != (vertex_count, 3) or not np.all(np.isfinite(p)):
            raise _err(
                "INVALID_MORPH",
                f"positions are ({vertex_count}, 3) finite offsets",
                f"{loc}.positions",
            )
        has = t.get("normals") is not None
        if with_normals is None:
            with_normals = has
        elif with_normals != has:
            raise _err(
                "INVALID_MORPH", "give normals for every target or for none", f"{loc}.normals"
            )
        if has:
            q = np.asarray(t["normals"], np.float64)
            if q.shape != (vertex_count, 3) or not np.all(np.isfinite(q)):
                raise _err(
                    "INVALID_MORPH",
                    f"normals are ({vertex_count}, 3) finite offsets",
                    f"{loc}.normals",
                )
            nrm.append(q.astype(np.float32))
        d = t.get("default", 0.0)
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(float(d)):
            raise _err("INVALID_MORPH", "default is a finite weight", f"{loc}.default")
        names.append(name)
        pos.append(p.astype(np.float32))
        defaults.append(float(d))
    positions = np.stack(pos)
    positions.flags.writeable = False
    normals = None
    if with_normals:
        normals = np.stack(nrm)
        normals.flags.writeable = False
    return Morph(obj, tuple(names), positions, normals, tuple(defaults))
