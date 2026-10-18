"""Constraints: rules that adjust an object's or a joint's evaluated transform.

A constraint list belongs to an **owner** -- a scene object (an item of kind
``constraints``) or a rig (its joint constraints) -- and applies in order, each blended
with what came before by its ``influence`` (0..1, animatable as
``constraints["NAME"].influence``). Constraints read **targets**:

* ``{"object": ID_OR_NAME}`` -- a scene object's evaluated world transform;
* ``{"joint": NAME}`` -- a joint of the same rig (joint constraints), or
  ``{"object": ARMATURE, "joint": NAME}`` -- a joint of another rig;
* ``{"point": [x, y, z]}`` -- a fixed point in world space (``"space": "armature"``: in
  the rig's space).

Types:

* ``ik`` (joint): ``chain`` (joint names root to end), ``target``, ``pole``, ``solver``
  (``two_bone`` for three joints, ``fabrik``, ``ccd``), ``iterations``, ``tolerance``,
  ``stretch``, ``soft``, ``end`` (``keep``: the end joint keeps its local rotation;
  ``target``: it takes the target's world rotation). The residual distance is reported.
* ``look_at`` (object or joint): point ``axis`` (``-y`` by default: cameras and lights
  look along local -Y; bones point along ``+y``) at the target, keeping ``up`` (owner
  axis, ``+z``) toward ``world_up`` (``[0, 0, 1]``); ``target_offset`` (world axes) moves
  the aim point off the target's origin (a character's chest rather than its feet).
* ``copy_location`` / ``copy_rotation`` / ``copy_scale`` / ``copy_transforms``: the
  target's world values (``axes``: which of ``xyz``; ``offset``: added in world space).
* ``child_of``: follow the target as if parented to it, from where it was when the
  constraint was made (``inverse``: the target's world matrix then, inverted; computed
  when not given).
* ``limit_location`` / ``limit_rotation`` (Euler XYZ degrees) / ``limit_scale``: clamp
  local values to ``min``/``max`` (per axis, ``null`` for no bound).
* ``limit_distance``: keep the distance to the target within ``min``..``max``.
* ``twist`` (joint): the owner joint turns about its ``axis`` (``+y``: along the bone) by
  ``factor`` times the twist of the target joint (``{"joint": NAME}``) from its rest
  rotation -- twist distribution: forearm or upper-arm twist joints take part of a
  wrist's or shoulder's twist, so skins twist along the limb instead of at one joint.

Evaluation order: the units (objects with constraints, rigs) run in dependency order --
a unit reads the transforms of its targets after their own units ran -- and a cycle is
refused when it is made (``CONSTRAINT_CYCLE``). A missing target makes the constraint do
nothing and is reported (``CONSTRAINT_TARGET_MISSING``); the scene is never changed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import paths as P
from . import rotation as R

TYPES = (
    "ik",
    "look_at",
    "copy_location",
    "copy_rotation",
    "copy_scale",
    "copy_transforms",
    "child_of",
    "limit_location",
    "limit_rotation",
    "limit_scale",
    "limit_distance",
    "follow_path",
    "twist",
)
JOINT_ONLY = ("ik", "twist")
MAX_CONSTRAINTS = 256
MAX_CHAIN = 64
AXES = {
    "+x": (0, 1.0),
    "-x": (0, -1.0),
    "+y": (1, 1.0),
    "-y": (1, -1.0),
    "+z": (2, 1.0),
    "-z": (2, -1.0),
}

_PARAMS: dict[str, dict[str, Any]] = {
    "ik": {
        "required": ("chain", "target"),
        "optional": {
            "pole": None,
            "solver": "two_bone",
            "iterations": 32,
            "tolerance": 1e-6,
            "stretch": 1.0,
            "soft": 0.0,
            "end": "keep",
        },
    },
    "look_at": {
        "required": ("target",),
        "optional": {
            "axis": "-y",
            "up": "+z",
            "world_up": [0.0, 0.0, 1.0],
            "target_offset": [0.0, 0.0, 0.0],
        },
    },
    "copy_location": {
        "required": ("target",),
        "optional": {"axes": "xyz", "offset": [0.0, 0.0, 0.0]},
    },
    "copy_rotation": {"required": ("target",), "optional": {}},
    "copy_scale": {"required": ("target",), "optional": {"axes": "xyz"}},
    "copy_transforms": {"required": ("target",), "optional": {}},
    "child_of": {"required": ("target",), "optional": {"inverse": None}},
    "limit_location": {
        "required": (),
        "optional": {"min": [None, None, None], "max": [None, None, None]},
    },
    "limit_rotation": {
        "required": (),
        "optional": {"min": [None, None, None], "max": [None, None, None]},
    },
    "limit_scale": {
        "required": (),
        "optional": {"min": [None, None, None], "max": [None, None, None]},
    },
    "limit_distance": {"required": ("target",), "optional": {"min": 0.0, "max": None}},
    "follow_path": {
        "required": ("path",),
        "optional": {"offset": 0.0, "follow": True, "axis": "+y", "up": "+z", "loop": False},
    },
    "twist": {"required": ("target",), "optional": {"axis": "+y", "factor": 0.5}},
}


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _num(
    value: Any, location: str, what: str, lo: float = -math.inf, hi: float = math.inf
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err("INVALID_CONSTRAINT", f"{what} must be a number, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v) or not lo <= v <= hi:
        raise _err(
            "INVALID_CONSTRAINT", f"{what} must be finite and within {lo:g}..{hi:g}", location
        )
    return v


@dataclass(frozen=True)
class Constraint:
    name: str
    type: str
    params: str  # canonical JSON of the checked parameters
    influence: float = 1.0
    mute: bool = False

    @property
    def p(self) -> dict[str, Any]:
        return json.loads(self.params)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type, **self.p}
        if self.influence != 1.0:
            out["influence"] = self.influence
        if self.mute:
            out["mute"] = True
        return out


def check_target(raw: Any, location: str, *, joints: bool) -> dict[str, Any]:
    if not isinstance(raw, dict) or not raw:
        raise _err(
            "INVALID_CONSTRAINT",
            'a target is {"object": ...}, {"joint": ...} or {"point": [x, y, z]}',
            location,
        )
    unknown = set(raw) - {"object", "joint", "point", "space"}
    if unknown:
        raise _err("INVALID_CONSTRAINT", f"unknown target fields {sorted(unknown)}", location)
    if "point" in raw:
        if set(raw) - {"point", "space"}:
            raise _err("INVALID_CONSTRAINT", "a point target has only point and space", location)
        pt = raw["point"]
        if not isinstance(pt, (list, tuple)) or len(pt) != 3:
            raise _err("INVALID_CONSTRAINT", "point is [x, y, z]", f"{location}.point")
        space = raw.get("space", "world")
        if space not in ("world", "armature"):
            raise _err("INVALID_CONSTRAINT", "space is 'world' or 'armature'", f"{location}.space")
        return {
            "point": [_num(v, f"{location}.point[{i}]", "a coordinate") for i, v in enumerate(pt)],
            "space": space,
        }
    if "space" in raw:
        raise _err("INVALID_CONSTRAINT", "space applies to point targets", location)
    out: dict[str, Any] = {}
    if "object" in raw:
        if not isinstance(raw["object"], str) or not raw["object"]:
            raise _err("INVALID_CONSTRAINT", "object is an id or a name", f"{location}.object")
        out["object"] = raw["object"]
    if "joint" in raw:
        out["joint"] = P.check_name(raw["joint"], "a joint name", f"{location}.joint")
        if not joints and "object" not in out:
            raise _err(
                "INVALID_CONSTRAINT",
                "an object's constraint names a joint of a rig: {object: ARMATURE, joint: NAME}",
                location,
            )
    return out


def parse_constraint(raw: Any, location: str, *, joints: bool) -> Constraint:
    """A checked constraint (``joints``: it belongs to a rig)."""
    if isinstance(raw, Constraint):
        return raw
    if not isinstance(raw, dict):
        raise _err("INVALID_CONSTRAINT", "a constraint is an object", location)
    kind = raw.get("type")
    if kind not in TYPES:
        raise _err(
            "INVALID_CONSTRAINT",
            f"unknown constraint type {kind!r}; one of {', '.join(TYPES)}",
            f"{location}.type",
        )
    if kind in JOINT_ONLY and not joints:
        raise _err(
            "INVALID_CONSTRAINT",
            f"{kind} constraints belong to rigs (joint chains)",
            f"{location}.type",
        )
    spec = _PARAMS[kind]
    allowed = {"name", "type", "influence", "mute", "joint", *spec["required"], *spec["optional"]}
    unknown = set(raw) - allowed
    if unknown:
        raise _err("INVALID_CONSTRAINT", f"unknown {kind} parameters {sorted(unknown)}", location)
    missing = [k for k in spec["required"] if k not in raw]
    if missing:
        raise _err("INVALID_CONSTRAINT", f"{kind} needs {', '.join(missing)}", location)
    name = P.check_name(raw.get("name", kind), "a constraint name", f"{location}.name")
    p: dict[str, Any] = {
        **spec["optional"],
        **{k: raw[k] for k in raw if k in spec["required"] or k in spec["optional"]},
    }
    if joints and kind != "ik":
        # Joint constraints other than IK name their owner joint.
        if "joint" not in raw:
            raise _err("INVALID_CONSTRAINT", f"a rig's {kind} constraint names its joint", location)
        p["joint"] = P.check_name(raw["joint"], "a joint name", f"{location}.joint")
    elif "joint" in raw:
        raise _err(
            "INVALID_CONSTRAINT", "joint names the owner of a rig's constraint", f"{location}.joint"
        )
    for key in ("target", "pole"):
        if p.get(key) is not None:
            p[key] = check_target(p[key], f"{location}.{key}", joints=joints)
    if kind == "ik":
        chain = p["chain"]
        if not isinstance(chain, list) or not 2 <= len(chain) <= MAX_CHAIN:
            raise _err(
                "INVALID_CONSTRAINT",
                f"chain lists 2..{MAX_CHAIN} joints, root to end",
                f"{location}.chain",
            )
        p["chain"] = [
            P.check_name(j, "a joint name", f"{location}.chain[{i}]") for i, j in enumerate(chain)
        ]
        if p["solver"] not in ("two_bone", "fabrik", "ccd"):
            raise _err(
                "INVALID_CONSTRAINT", "solver is two_bone, fabrik or ccd", f"{location}.solver"
            )
        if p["solver"] == "two_bone" and len(chain) != 3:
            raise _err(
                "INVALID_CONSTRAINT", "two_bone solves chains of three joints", f"{location}.chain"
            )
        it = p["iterations"]
        if isinstance(it, bool) or not isinstance(it, int) or not 1 <= it <= 1000:
            raise _err(
                "INVALID_CONSTRAINT", "iterations is an integer 1..1000", f"{location}.iterations"
            )
        p["tolerance"] = _num(p["tolerance"], f"{location}.tolerance", "tolerance", 0.0, 1e3)
        p["stretch"] = _num(p["stretch"], f"{location}.stretch", "stretch", 1.0, 10.0)
        p["soft"] = _num(p["soft"], f"{location}.soft", "soft", 0.0, 0.5)
        if p["end"] not in ("keep", "target"):
            raise _err("INVALID_CONSTRAINT", "end is 'keep' or 'target'", f"{location}.end")
    if kind in ("look_at", "follow_path"):
        for key in ("axis", "up"):
            if p[key] not in AXES:
                raise _err(
                    "INVALID_CONSTRAINT", f"{key} is one of {', '.join(AXES)}", f"{location}.{key}"
                )
        if AXES[p["axis"]][0] == AXES[p["up"]][0]:
            raise _err("INVALID_CONSTRAINT", "axis and up must be different axes", location)
    if kind == "look_at":
        wu = p["world_up"]
        if not isinstance(wu, (list, tuple)) or len(wu) != 3:
            raise _err("INVALID_CONSTRAINT", "world_up is [x, y, z]", f"{location}.world_up")
        p["world_up"] = [_num(v, f"{location}.world_up[{i}]", "world_up") for i, v in enumerate(wu)]
        if math.hypot(*p["world_up"]) < 1e-9:
            raise _err("INVALID_CONSTRAINT", "world_up must not be zero", f"{location}.world_up")
        off = p["target_offset"]
        if not isinstance(off, (list, tuple)) or len(off) != 3:
            raise _err(
                "INVALID_CONSTRAINT", "target_offset is [x, y, z]", f"{location}.target_offset"
            )
        p["target_offset"] = [
            _num(v, f"{location}.target_offset[{i}]", "target_offset") for i, v in enumerate(off)
        ]
    if kind in ("copy_location", "copy_scale"):
        axes = p["axes"]
        if (
            not isinstance(axes, str)
            or not axes
            or set(axes) - set("xyz")
            or len(set(axes)) != len(axes)
        ):
            raise _err("INVALID_CONSTRAINT", "axes is some of 'xyz'", f"{location}.axes")
    if kind == "copy_location":
        off = p["offset"]
        if not isinstance(off, (list, tuple)) or len(off) != 3:
            raise _err("INVALID_CONSTRAINT", "offset is [x, y, z]", f"{location}.offset")
        p["offset"] = [_num(v, f"{location}.offset[{i}]", "offset") for i, v in enumerate(off)]
    if kind.startswith("limit_") and kind != "limit_distance":
        for key in ("min", "max"):
            vals = p[key]
            if not isinstance(vals, (list, tuple)) or len(vals) != 3:
                raise _err(
                    "INVALID_CONSTRAINT",
                    f"{key} is [x, y, z] (null: no bound)",
                    f"{location}.{key}",
                )
            p[key] = [
                None if v is None else _num(v, f"{location}.{key}[{i}]", key)
                for i, v in enumerate(vals)
            ]
        for i in range(3):
            lo, hi = p["min"][i], p["max"][i]
            if lo is not None and hi is not None and lo > hi:
                raise _err("INVALID_CONSTRAINT", "min must not exceed max", location)
    if kind == "limit_distance":
        p["min"] = _num(p["min"], f"{location}.min", "min", 0.0)
        if p["max"] is not None:
            p["max"] = _num(p["max"], f"{location}.max", "max", 0.0)
            if p["max"] < p["min"]:
                raise _err("INVALID_CONSTRAINT", "min must not exceed max", location)
    if kind == "child_of" and p["inverse"] is not None:
        inv = np.asarray(p["inverse"], np.float64)
        if inv.shape != (4, 4) or not np.all(np.isfinite(inv)):
            raise _err(
                "INVALID_CONSTRAINT", "inverse is a 4x4 matrix (rows)", f"{location}.inverse"
            )
        p["inverse"] = inv.tolist()
    if kind == "twist":
        t = p["target"]
        if set(t) != {"joint"}:
            raise _err(
                "INVALID_CONSTRAINT",
                "a twist constraint's target is a joint of the same rig: {joint: NAME}",
                f"{location}.target",
            )
        if p["axis"] not in AXES:
            raise _err(
                "INVALID_CONSTRAINT", f"axis is one of {', '.join(AXES)}", f"{location}.axis"
            )
        p["factor"] = _num(p["factor"], f"{location}.factor", "factor", -1.0, 1.0)
    if kind == "follow_path":
        if not isinstance(p["path"], str) or not p["path"]:
            raise _err("INVALID_CONSTRAINT", "path is a path item (id or name)", f"{location}.path")
        p["offset"] = _num(p["offset"], f"{location}.offset", "offset")
        for key in ("follow", "loop"):
            if not isinstance(p[key], bool):
                raise _err("INVALID_CONSTRAINT", f"{key} is true or false", f"{location}.{key}")
    influence = _num(raw.get("influence", 1.0), f"{location}.influence", "influence", 0.0, 1.0)
    mute = raw.get("mute", False)
    if not isinstance(mute, bool):
        raise _err("INVALID_CONSTRAINT", "mute is true or false", f"{location}.mute")
    return Constraint(
        name, kind, json.dumps(p, sort_keys=True, separators=(",", ":")), influence, mute
    )


def parse_constraints(raw: Any, location: str, *, joints: bool) -> tuple[Constraint, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_CONSTRAINTS:
        raise _err(
            "INVALID_CONSTRAINT", f"constraints is a list of at most {MAX_CONSTRAINTS}", location
        )
    out = tuple(parse_constraint(c, f"{location}[{i}]", joints=joints) for i, c in enumerate(raw))
    names = [c.name for c in out]
    if len(set(names)) != len(names):
        raise _err("INVALID_CONSTRAINT", "constraint names must be unique per owner", location)
    return out


# --------------------------------------------------------------------------------------- #
# Matrix helpers (evaluation)
# --------------------------------------------------------------------------------------- #


def decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(position, rotation (unit quaternion), scale)`` of an affine matrix (the closest
    rotation when there is shear; negative determinant -> negative x scale)."""
    from .. import math3d

    try:
        p, q, s = math3d.decompose_matrix(m)
    except Exception:  # noqa: BLE001 - singular: keep the translation
        return m[:3, 3].copy(), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3)
    return np.array(p), np.array(q), np.array(s)


def compose(p: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = R.matrices(np.asarray(q, np.float64)[None])[0] * np.asarray(s, np.float64)
    m[:3, 3] = p
    return m


def blend_matrices(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """``a`` moved toward ``b`` by ``w``: positions and scales linearly, rotations by slerp."""
    if w >= 1.0:
        return b
    if w <= 0.0:
        return a
    pa, qa, sa = decompose(a)
    pb, qb, sb = decompose(b)
    q = R.slerp(qa[None], qb[None], np.array([w]))[0]
    return compose(pa + w * (pb - pa), q, sa + w * (sb - sa))


def look_rotation(forward: np.ndarray, up_hint: np.ndarray, axis: str, up: str) -> np.ndarray:
    """A rotation matrix whose owner ``axis`` points along ``forward`` and whose ``up``
    axis leans toward ``up_hint``."""
    f = np.asarray(forward, np.float64)
    n = np.linalg.norm(f)
    if n < 1e-12:
        return np.eye(3)
    f = f / n
    u = np.asarray(up_hint, np.float64)
    u = u - (u @ f) * f
    if np.linalg.norm(u) < 1e-9:  # looking along up: any perpendicular
        u = np.cross(f, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(f, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    ai, asign = AXES[axis]
    ui, usign = AXES[up]
    cols = [None, None, None]
    cols[ai] = asign * f
    cols[ui] = usign * u
    k = 3 - ai - ui
    # The third axis completes a right-handed basis.
    e = [np.zeros(3), np.zeros(3), np.zeros(3)]
    e[ai], e[ui] = cols[ai], cols[ui]
    third = np.cross(e[(k + 1) % 3], e[(k + 2) % 3])
    cols[k] = third
    return np.stack(cols, axis=1)


def apply_constraint(
    c: Constraint,
    owner: np.ndarray,
    target: np.ndarray | None,
    target_pos: np.ndarray | None,
    *,
    parent: np.ndarray | None = None,
    rest_local: np.ndarray | None = None,
) -> np.ndarray:
    """The owner's world matrix after a non-IK constraint (``target``: the target's world
    matrix, ``target_pos``: its position; ``parent``: the owner's parent world matrix,
    for local limits)."""
    p = c.p
    kind = c.type
    if kind == "look_at":
        pos = owner[:3, 3]
        aim = target_pos + np.asarray(p["target_offset"], np.float64)
        rot = look_rotation(aim - pos, np.asarray(p["world_up"]), p["axis"], p["up"])
        _, _, s = decompose(owner)
        out = owner.copy()
        out[:3, :3] = rot * s
        return out
    if kind == "copy_location":
        out = owner.copy()
        for axis in p["axes"]:
            i = "xyz".index(axis)
            out[i, 3] = target_pos[i] + p["offset"][i]
        return out
    if kind in ("copy_rotation", "copy_scale", "copy_transforms"):
        po, qo, so = decompose(owner)
        pt, qt, st = decompose(target)
        if kind == "copy_rotation":
            return compose(po, qt, so)
        if kind == "copy_scale":
            s = so.copy()
            for axis in p["axes"]:
                i = "xyz".index(axis)
                s[i] = st[i]
            return compose(po, qo, s)
        return target.copy()
    if kind == "child_of":
        inv = np.asarray(p["inverse"], np.float64) if p["inverse"] is not None else np.eye(4)
        return target @ inv @ owner
    if kind in ("limit_location", "limit_rotation", "limit_scale"):
        par = np.eye(4) if parent is None else parent
        local = np.linalg.solve(par, owner)
        pl, ql, sl = decompose(local)
        lo, hi = p["min"], p["max"]
        if kind == "limit_location":
            for i in range(3):
                if lo[i] is not None:
                    pl[i] = max(pl[i], lo[i])
                if hi[i] is not None:
                    pl[i] = min(pl[i], hi[i])
        elif kind == "limit_scale":
            for i in range(3):
                if lo[i] is not None:
                    sl[i] = max(sl[i], lo[i])
                if hi[i] is not None:
                    sl[i] = min(sl[i], hi[i])
        else:
            e = R.to_euler(ql[None])[0]
            for i in range(3):
                if lo[i] is not None:
                    e[i] = max(e[i], lo[i])
                if hi[i] is not None:
                    e[i] = min(e[i], hi[i])
            ql = R.from_euler(e)[0]
        return par @ compose(pl, ql, sl)
    if kind == "limit_distance":
        pos = owner[:3, 3]
        d = pos - target_pos
        dist = float(np.linalg.norm(d))
        lo, hi = p["min"], p["max"]
        want = dist
        if dist < lo:
            want = lo
        if hi is not None and dist > hi:
            want = hi
        if want == dist or dist < 1e-12:
            return owner
        out = owner.copy()
        out[:3, 3] = target_pos + d * (want / dist)
        return out
    raise AssertionError(kind)  # pragma: no cover


def targets_of(constraints: tuple[Constraint, ...]) -> list[dict[str, Any]]:
    out = []
    for c in constraints:
        p = c.p
        for key in ("target", "pole"):
            t = p.get(key)
            if t is not None:
                out.append(t)
    return out
