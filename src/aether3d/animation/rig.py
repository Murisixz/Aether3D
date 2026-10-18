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
METHODS = ("linear", "dual_quaternion")


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
    limits: str = "null"  # canonical JSON: {"rotation": {"min": [...], "max": [...]}}

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
        return out


@dataclass(frozen=True)
class Rig:
    """A skeleton on an armature object (immutable; see the module docstring)."""

    object: str
    joints: tuple[Joint, ...]
    attachments: tuple[tuple[str, str], ...] = ()  # (object id, joint id)
    constraints: tuple[Constraint, ...] = ()
    mirror: tuple[tuple[str, str], ...] = ((".L", ".R"), ("Left", "Right"), ("_l", "_r"))
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

    def descendants(self, k: int) -> list[int]:
        out = []
        parents = self.parents
        inside = {k}
        for i in range(k + 1, len(self.joints)):
            if int(parents[i]) in inside:
                inside.add(i)
                out.append(i)
        return out

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
        return out


def parse_rig(raw: Any, location: str) -> Rig:
    """A checked rig: joints parents first (reordered when needed), names unique, ids
    unique (generated when missing), no cycles."""
    if isinstance(raw, Rig):
        return raw
    if not isinstance(raw, dict) or set(raw) - {
        "object",
        "joints",
        "attachments",
        "constraints",
        "mirror",
    }:
        raise _err(
            "INVALID_RIG", "a rig is {object, joints, attachments, constraints, mirror}", location
        )
    obj = raw.get("object")
    if not isinstance(obj, str) or not obj:
        raise _err("INVALID_RIG", "a rig names its armature object", f"{location}.object")
    joints_raw = raw.get("joints")
    if not isinstance(joints_raw, list) or not 1 <= len(joints_raw) <= MAX_JOINTS:
        raise _err("INVALID_RIG", f"a rig has 1..{MAX_JOINTS} joints", f"{location}.joints")
    rows = []
    names: dict[str, int] = {}
    ids: set[str] = set()
    for i, j in enumerate(joints_raw):
        loc = f"{location}.joints[{i}]"
        if not isinstance(j, dict) or set(j) - {
            "id",
            "name",
            "parent",
            "position",
            "rotation",
            "euler",
            "scale",
            "tail",
            "limits",
        }:
            raise _err(
                "INVALID_RIG",
                "a joint is {id, name, parent, position, rotation, scale, tail, limits}",
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
        rows.append((jid, name, parent, pos, rot, scl, tail, limits_json))
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
    return Rig(obj, joints, tuple(attachments), constraints, mirror_t)


def _check_limits(raw: Any, location: str) -> str:
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

    @property
    def vertex_count(self) -> int:
        return int(self.indices.shape[0])

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
        }

    def same(self, other: Skin) -> bool:
        return (
            self.object == other.object
            and self.rig == other.rig
            and self.joints == other.joints
            and self.method == other.method
            and self.inverse_bind.tobytes() == other.inverse_bind.tobytes()
            and self.indices.tobytes() == other.indices.tobytes()
            and self.weights.tobytes() == other.weights.tobytes()
        )


def make_skin(
    obj: str,
    rig_id: str,
    joints: list[str],
    inverse_bind: Any,
    indices: Any,
    weights: Any,
    method: str = "linear",
    location: str = "skin",
) -> Skin:
    """A checked skin: weights non-negative, normalized per vertex (rows summing to 0 are
    refused), at most :data:`MAX_INFLUENCES` per vertex, indices within the palette."""
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
    for a in (ib, idx, w):
        a.flags.writeable = False
    return Skin(obj, rig_id, tuple(joints), ib, idx, w, method)


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


def skin_dual_quaternion(
    positions: np.ndarray,
    normals: np.ndarray | None,
    indices: np.ndarray,
    weights: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dual quaternion skinning (Kavan et al. 2008) with Kavan's scale stage: each
    skinning matrix ``S = T R K`` is split into a rigid part ``(R, T)`` and a scale/shear
    part ``K``; ``K`` is blended linearly and applied first, the rigid parts as dual
    quaternions (antipodality resolved against the first influence)."""
    pal = np.asarray(palette[:, :3, :4], np.float64)
    m = pal.shape[0]
    lin = pal[:, :, :3]
    # Polar decomposition per joint: lin = R K (R the closest rotation).
    u, _, vt = np.linalg.svd(lin)
    rot = u @ vt
    flip = np.linalg.det(rot) < 0
    if np.any(flip):
        u[flip, :, -1] *= -1.0
        rot = u @ vt
    k_mat = np.einsum("nji,njk->nik", rot, lin)  # R^T lin
    q = R.from_matrices(rot)  # (m, 4)
    t = pal[:, :, 3]
    # Dual part: 0.5 * (t, 0) * q.
    tq = np.concatenate([t, np.zeros((m, 1))], axis=1)
    d = 0.5 * R.multiply(tq, q)
    n = positions.shape[0]
    out_p = np.empty((n, 3), np.float32)
    out_n = None if normals is None else np.empty((n, 3), np.float32)
    kk = indices.shape[1]
    for start in range(0, n, SKIN_CHUNK):
        sl = slice(start, min(n, start + SKIN_CHUNK))
        idx = indices[sl]
        w = weights[sl]
        q0 = q[idx[:, 0]]
        real = np.zeros((idx.shape[0], 4))
        dual = np.zeros((idx.shape[0], 4))
        kblend = np.zeros((idx.shape[0], 3, 3))
        for j in range(kk):
            qj = q[idx[:, j]]
            s = np.where(np.sum(qj * q0, axis=1) < 0.0, -1.0, 1.0) * w[:, j]
            real = real + s[:, None] * qj
            dual = dual + s[:, None] * d[idx[:, j]]
            kblend = kblend + w[:, j, None, None] * k_mat[idx[:, j]]
        norm = np.sqrt(np.sum(real * real, axis=1))
        norm = np.where(norm > 0.0, norm, 1.0)
        real = real / norm[:, None]
        dual = dual / norm[:, None]
        p = positions[sl].astype(np.float64)
        p = np.einsum("nij,nj->ni", kblend, p)
        rp = R.rotate(real, p)
        # translation = 2 * dual * conj(real) (vector part)
        trans = 2.0 * R.multiply(dual, R.conjugate(real))[:, :3]
        out_p[sl] = (rp + trans).astype(np.float32)
        if normals is not None:
            nr = normals[sl].astype(np.float64)
            # Normals: the inverse transpose of K, then the rotation.
            kin = np.linalg.inv(kblend).transpose(0, 2, 1)
            nn = R.rotate(real, np.einsum("nij,nj->ni", kin, nr))
            length = np.linalg.norm(nn, axis=1)
            out_n[sl] = (nn / np.where(length > 0, length, 1.0)[:, None]).astype(np.float32)  # type: ignore[index]
    return out_p, out_n


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
