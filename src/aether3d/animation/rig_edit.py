"""Structural rig editing: joints added, removed, moved, oriented, renamed, reparented,
inserted, split and mirrored -- with everything that depends on them kept consistent.

:func:`edit_rig` applies a list of operations to one rig as **one undo step** (all or
nothing; ``dry_run`` reports without changing anything). Operations (``op``):

* ``add_joint`` -- ``name``, ``parent``, ``position`` (``space``: ``parent`` -- relative
  to the parent joint, the default -- ``armature`` or ``world``), ``rotation`` | ``euler``,
  ``scale``, ``tail``, ``role``, ``deform``, ``limits``. A new joint has no weights.
* ``remove_joint`` -- ``joint``; its children move to its parent (their rest placement
  kept); skin weights and attachments go to ``weights_to`` (default: the parent);
  controls and correctives on it are removed; constraints using it are refused unless
  ``constraints: "remove"``.
* ``move_joint`` -- ``joint``, ``position`` (``space`` as above); ``children``:
  ``keep_world`` (they stay where they are, the default) or ``follow``.
* ``orient_joint`` -- ``joint`` (or ``joints``: names or ``"all"``): point the local
  ``aim`` axis (``+y``) at the bone's end (first child, else tail) with the ``up`` axis
  (``+z``) toward ``up_vector`` (armature space; default the rig's forward axis, else
  ``-y``), then turn by ``roll`` degrees about the aim. Children keep their placement.
* ``rename_joint`` -- ``joint``, ``name``.
* ``reparent_joint`` -- ``joint``, ``parent`` (``null``: a root); ``keep``: ``world``
  (the default) or ``local``. Cycles are refused.
* ``insert_joint`` -- ``name`` and ``child``: a joint between ``child`` and its parent at
  fraction ``at`` (0.5) of the bone; ``weights``: ``split`` (vertices of the parent bone
  beyond ``at`` move to the new joint, over a smooth band of ``blend`` (0.1) of the bone)
  or ``none`` (the default).
* ``split_bone`` -- ``joint`` and ``count`` (2..16): ``count - 1`` joints evenly between
  the joint and its first child (``names`` optional), weights split along the bone.
* ``mirror_joints`` -- ``joints`` (names) or ``side`` (``"L"``/``"R"``): create or update
  each joint's mirror image (``mirror`` name pairs; reflection ``x -> -x`` in armature
  space) with mirrored roles and limits.
* ``align_bone`` -- ``joint`` and ``direction`` (an axis such as ``"-z"`` or ``[x, y,
  z]``, armature space): the joint's first child (and its subtree) moves so the bone
  points that way, keeping its length.
* ``set_joint`` -- ``joint`` and any of ``role``, ``deform``, ``limits``, ``tail``.

**Dependent data** (reported, never changed silently):

* skins bound to the rig: ``skins="rebind"`` (default) keeps the rest deformation exactly
  -- each changed joint's inverse bind matrix becomes ``J_new^-1 J_old B`` -- so the mesh
  stays where it is and bends about the new joints; ``"keep_bind"`` keeps the inverse
  binds (the mesh follows the moved joints at rest). Removed joints' weights go to their
  target; new joints have none (``split`` excepted).
* clips animating the armature (its action and strips, and unbound clips naming only
  this rig's joints) and layer masks: renamed joints are renamed in channel paths and
  masks. Channels of removed joints are kept (reported ``JOINT_MISSING`` at evaluation)
  unless ``channels="remove"``. A changed joint frame (``orient_joint``) changes what its
  rotation channels and its children's channels mean: ``animation="convert"`` rewrites
  them -- quaternion tracks exactly (left and right products by constants commute with
  slerp and squad), Euler and position curves by resampling at 120 Hz and refitting within
  ``tolerance`` (the error is reported) -- and ``"keep"`` (the default) reports them.
* constraints (chains, owners, targets -- of this rig and any constraint targeting its
  joints), controls, correctives and attachments follow renames and removals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from ..errors import AnimationError
from . import paths as P
from . import rotation as R
from .constraints import compose, decompose

if TYPE_CHECKING:
    from .system import AnimationSystem

OPS = (
    "add_joint",
    "remove_joint",
    "move_joint",
    "orient_joint",
    "rename_joint",
    "reparent_joint",
    "insert_joint",
    "split_bone",
    "mirror_joints",
    "align_bone",
    "set_joint",
)
SKIN_POLICIES = ("rebind", "keep_bind")
ANIMATION_POLICIES = ("keep", "convert")
CHANNEL_POLICIES = ("keep", "remove")
MAX_OPERATIONS = 10_000
MIN_BONE = 1e-6
_AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


def _vec(value: Any, n: int, location: str, what: str) -> np.ndarray:
    try:
        a = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err("INVALID_ARGUMENT", f"{what} is {n} numbers", location) from None
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise _err("INVALID_ARGUMENT", f"{what} is {n} finite numbers", location)
    return a


def _direction(value: Any, location: str) -> np.ndarray:
    if isinstance(value, str):
        if value not in _AXES:
            raise _err("INVALID_ARGUMENT", f"an axis is one of {', '.join(_AXES)}", location)
        return np.array(_AXES[value])
    d = _vec(value, 3, location, "a direction")
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise _err("INVALID_ARGUMENT", "a direction must not be zero", location)
    return d / n


# --------------------------------------------------------------------------------------- #
# The joint table (mutable while editing)
# --------------------------------------------------------------------------------------- #


@dataclass
class _Row:
    id: str
    name: str
    parent: str | None
    pos: np.ndarray
    rot: np.ndarray
    scl: np.ndarray
    tail: np.ndarray | None
    limits: str
    role: str | None
    deform: bool


class _Table:
    def __init__(self, rig: Any) -> None:
        self.rows: list[_Row] = []
        for j in rig.joints:
            self.rows.append(
                _Row(
                    j.id,
                    j.name,
                    None if j.parent < 0 else rig.joints[j.parent].name,
                    np.array(j.position, np.float64),
                    np.array(j.rotation, np.float64),
                    np.array(j.scale, np.float64),
                    None if j.tail is None else np.array(j.tail, np.float64),
                    j.limits,
                    j.role,
                    j.deform,
                )
            )

    def find(self, name: Any, location: str) -> _Row:
        if isinstance(name, str):
            for r in self.rows:
                if r.name == name:
                    return r
        raise _err("JOINT_MISSING", f"no joint {name!r} in the rig", location)

    def get(self, name: str | None) -> _Row | None:
        if name is None:
            return None
        for r in self.rows:
            if r.name == name:
                return r
        return None

    def children(self, name: str) -> list[_Row]:
        return [r for r in self.rows if r.parent == name]

    def subtree(self, name: str) -> list[_Row]:
        out, stack = [], [name]
        while stack:
            n = stack.pop()
            for c in self.children(n):
                out.append(c)
                stack.append(c.name)
        return out

    @staticmethod
    def local(r: _Row) -> np.ndarray:
        return compose(r.pos, r.rot, r.scl)

    def models(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}

        def model(r: _Row) -> np.ndarray:
            found = out.get(r.name)
            if found is not None:
                return found
            parent = self.get(r.parent)
            m = self.local(r) if parent is None else model(parent) @ self.local(r)
            out[r.name] = m
            return m

        for r in self.rows:
            model(r)
        return out

    def set_model(self, r: _Row, m: np.ndarray, models: dict[str, np.ndarray]) -> None:
        parent = self.get(r.parent)
        local = m if parent is None else np.linalg.solve(models[parent.name], m)
        p, q, s = decompose(local)
        r.pos, r.rot, r.scl = p, _canonical(q), _clean_scale(s, r.scl)

    def unique(self, name: str, location: str) -> str:
        P.check_name(name, "a joint name", location)
        if self.get(name) is not None:
            raise _err("INVALID_RIG", f"a joint is already named {name!r}", location)
        return name


def _canonical(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / math.sqrt(float(q @ q))
    return -q if q[3] < 0 else q


def _clean_scale(s: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Scales that were exactly 1 stay exactly 1 through a decomposition."""
    s = np.asarray(s, np.float64).copy()
    near = np.abs(s - before) < 1e-12
    s[near] = before[near]
    return s


def _quat_of(m: np.ndarray) -> np.ndarray:
    lin = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    return _canonical(R.from_matrices(lin[None])[0])


# --------------------------------------------------------------------------------------- #
# Edit state: what happened to joints (for the dependent data)
# --------------------------------------------------------------------------------------- #


@dataclass
class _State:
    table: _Table
    armature_world: np.ndarray
    old_models: dict[str, np.ndarray]  # by joint id: rest model before the edit
    renames: dict[str, str] = field(default_factory=dict)  # old name -> new name (final)
    removed: dict[str, str | None] = field(default_factory=dict)  # removed id -> target id
    frame_right: dict[str, np.ndarray] = field(default_factory=dict)  # id -> C (q' = q C)
    frame_left: dict[str, np.ndarray] = field(default_factory=dict)  # id -> L (q' = L q)
    splits: list[tuple[str, str, float, float]] = field(default_factory=list)
    drop_constraints: set[str] = field(default_factory=set)
    report: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    def warn(self, code: str, message: str, location: str, severity: str = WARNING) -> None:
        self.diagnostics.append(Diagnostic(code, severity, message, location))


def _position(st: _State, raw: dict, parent: _Row | None, location: str) -> np.ndarray:
    """A requested joint position as an armature-space point."""
    p = _vec(raw.get("position", (0, 0, 0)), 3, f"{location}.position", "position")
    space = raw.get("space", "parent")
    if space == "parent":
        if parent is None:
            return p
        return (st.table.models()[parent.name] @ np.append(p, 1.0))[:3]
    if space == "armature":
        return p
    if space == "world":
        return (np.linalg.solve(st.armature_world, np.append(p, 1.0)))[:3]
    raise _err("INVALID_ARGUMENT", "space is 'parent', 'armature' or 'world'", f"{location}.space")


def _check_keys(raw: dict, allowed: set[str], location: str) -> None:
    unknown = set(raw) - allowed - {"op"}
    if unknown:
        raise _err(
            "INVALID_ARGUMENT",
            f"unknown arguments {sorted(unknown)} of {raw.get('op')}",
            location,
            hint=f"arguments: {', '.join(sorted(allowed))}",
        )


# --------------------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------------------- #


def _op_add_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(
        raw,
        {"name", "parent", "position", "space", "rotation", "euler", "scale", "tail", "role",
         "deform", "limits", "id"},
        loc,
    )  # fmt: skip
    t = st.table
    name = t.unique(raw.get("name"), f"{loc}.name")
    parent = None if raw.get("parent") is None else t.find(raw["parent"], f"{loc}.parent")
    point = _position(st, raw, parent, loc)
    if "rotation" in raw and "euler" in raw:
        raise _err("INVALID_ARGUMENT", "give rotation or euler, not both", loc)
    if "euler" in raw:
        q = R.from_euler(_vec(raw["euler"], 3, f"{loc}.euler", "euler"))[0]
    else:
        q = _vec(raw.get("rotation", (0, 0, 0, 1)), 4, f"{loc}.rotation", "rotation")
        if np.linalg.norm(q) < 1e-9:
            raise _err("INVALID_ARGUMENT", "a rotation must not be zero", f"{loc}.rotation")
    scl = _vec(raw.get("scale", (1, 1, 1)), 3, f"{loc}.scale", "scale")
    models = t.models()
    space = raw.get("space", "parent")
    rot_m = R.matrices(_canonical(q)[None])[0]
    if space != "parent" and parent is not None:  # an orientation in armature/world space
        if space == "world":
            rot_m = _unit_columns(st.armature_world[:3, :3]).T @ rot_m
        rot_m = _unit_columns(models[parent.name][:3, :3]).T @ rot_m
    elif space == "world":
        rot_m = _unit_columns(st.armature_world[:3, :3]).T @ rot_m
    local_q = _canonical(R.from_matrices(rot_m[None])[0])
    m_parent = np.eye(4) if parent is None else models[parent.name]
    local_p = np.linalg.solve(m_parent, np.append(point, 1.0))[:3]
    tail = None if raw.get("tail") is None else _vec(raw["tail"], 3, f"{loc}.tail", "tail")
    from .rig import _check_limits

    limits = "null" if raw.get("limits") is None else _check_limits(raw["limits"], f"{loc}.limits")
    deform = raw.get("deform", True)
    if not isinstance(deform, bool):
        raise _err("INVALID_ARGUMENT", "deform is true or false", f"{loc}.deform")
    import uuid

    jid = raw.get("id") or str(uuid.uuid4())
    if any(r.id == jid for r in t.rows):
        raise _err("INVALID_RIG", "joint ids must be unique", f"{loc}.id")
    t.rows.append(
        _Row(jid, name, None if parent is None else parent.name, local_p, local_q, scl, tail,
             limits, raw.get("role"), deform)
    )  # fmt: skip
    st.added.append(name)
    st.report.append({"op": "add_joint", "joint": name, "position": point.tolist()})


def _unit_columns(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=0)


def _op_remove_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "weights_to", "constraints"}, loc)
    t = st.table
    r = t.find(raw.get("joint"), f"{loc}.joint")
    if raw.get("constraints", "refuse") not in ("refuse", "remove"):
        raise _err("INVALID_ARGUMENT", "constraints is 'refuse' or 'remove'", f"{loc}.constraints")
    to = raw.get("weights_to", "parent")
    target = t.get(r.parent) if to == "parent" else t.find(to, f"{loc}.weights_to")
    if target is r:
        raise _err("INVALID_ARGUMENT", "weights go to another joint", f"{loc}.weights_to")
    models = t.models()
    for c in t.children(r.name):  # children keep their rest placement
        m = models[c.name]
        c.parent = r.parent
        t.set_model(c, m, models)
    t.rows.remove(r)
    st.removed[r.id] = None if target is None else target.id
    if raw.get("constraints") == "remove":
        st.drop_constraints.add(r.name)
    st.report.append(
        {
            "op": "remove_joint",
            "joint": r.name,
            "weights_to": None if target is None else target.name,
        }
    )


def _op_move_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "position", "space", "children"}, loc)
    t = st.table
    r = t.find(raw.get("joint"), f"{loc}.joint")
    children = raw.get("children", "keep_world")
    if children not in ("keep_world", "follow"):
        raise _err("INVALID_ARGUMENT", "children is 'keep_world' or 'follow'", f"{loc}.children")
    parent = t.get(r.parent)
    point = _position(st, raw, parent, loc)
    models = t.models()
    old = models[r.name]
    new = old.copy()
    new[:3, 3] = point
    kids = t.children(r.name)
    keep = {c.name: models[c.name] for c in kids} if children == "keep_world" else {}
    t.set_model(r, new, models)
    models = t.models()
    for c in kids:
        if c.name in keep:
            t.set_model(c, keep[c.name], models)
    st.report.append(
        {"op": "move_joint", "joint": r.name, "from": old[:3, 3].tolist(), "to": point.tolist()}
    )


def _bone_end(t: _Table, r: _Row, models: dict[str, np.ndarray]) -> np.ndarray | None:
    kids = t.children(r.name)
    if kids:
        return models[kids[0].name][:3, 3]
    if r.tail is not None:
        return (models[r.name] @ np.append(r.tail, 1.0))[:3]
    return None


def _op_orient_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "joints", "aim", "up", "up_vector", "roll"}, loc)
    from .constraints import AXES, look_rotation

    t = st.table
    if "joint" in raw and "joints" in raw:
        raise _err("INVALID_ARGUMENT", "give joint or joints", loc)
    if "joints" in raw:
        names = [r.name for r in t.rows] if raw["joints"] == "all" else raw["joints"]
        if not isinstance(names, list):
            raise _err("INVALID_ARGUMENT", "joints is a list of names or 'all'", f"{loc}.joints")
    else:
        names = [raw.get("joint")]
    aim, up = raw.get("aim", "+y"), raw.get("up", "+z")
    for key, v in (("aim", aim), ("up", up)):
        if v not in AXES:
            raise _err("INVALID_ARGUMENT", f"{key} is one of {', '.join(AXES)}", f"{loc}.{key}")
    if AXES[aim][0] == AXES[up][0]:
        raise _err("INVALID_ARGUMENT", "aim and up must be different axes", loc)
    default_up = rig.forward_axis("-y")
    up_vector = _direction(raw.get("up_vector", default_up), f"{loc}.up_vector")
    roll = raw.get("roll", 0.0)
    if isinstance(roll, bool) or not isinstance(roll, (int, float)) or not math.isfinite(roll):
        raise _err("INVALID_ARGUMENT", "roll is degrees", f"{loc}.roll")
    oriented = []
    for i, name in enumerate(names):
        r = t.find(name, f"{loc}.joints[{i}]" if "joints" in raw else f"{loc}.joint")
        models = t.models()
        m = models[r.name]
        end = _bone_end(t, r, models)
        if end is None or float(np.linalg.norm(end - m[:3, 3])) < MIN_BONE:
            st.warn(
                "BONE_TOO_SHORT",
                f"joint {r.name!r} has no bone to aim along; its frame is kept",
                f"{loc}.joint",
            )
            continue
        rot = look_rotation(end - m[:3, 3], up_vector, aim, up)
        if roll:
            ai, sgn = AXES[aim]
            axis = np.zeros(3)
            axis[ai] = sgn
            rot = rot @ R.matrices(R.axis_angle(axis, math.radians(float(roll))))[0]
        old_q = _quat_of(m)
        new_q = _canonical(R.from_matrices(rot[None])[0])
        c = _canonical(R.multiply(R.conjugate(old_q)[None], new_q[None])[0])  # old^-1 new
        if np.allclose(c, [0, 0, 0, 1], atol=1e-14):
            continue
        kids = {k.name: models[k.name] for k in t.children(r.name)}
        new = m.copy()
        scale = np.linalg.norm(m[:3, :3], axis=0)
        new[:3, :3] = rot * scale
        t.set_model(r, new, models)
        models = t.models()
        for k in t.children(r.name):
            t.set_model(k, kids[k.name], models)
        if r.tail is not None:
            r.tail = R.rotate(R.conjugate(c)[None], r.tail[None])[0]
        _compose_frame(st, r.id, c, [k.id for k in t.children(r.name)])
        if r.limits != "null":
            st.warn(
                "JOINT_FRAME_CHANGED",
                f"joint {r.name!r} has limits measured in its old frame; check them",
                f"{loc}.joint",
            )
        for k in t.children(r.name):
            if k.limits != "null" and "rotation" in json.loads(k.limits):
                st.warn(
                    "JOINT_FRAME_CHANGED",
                    f"joint {k.name!r} has Euler limits relative to its parent's frame; check them",
                    f"{loc}.joint",
                )
        oriented.append(r.name)
    st.report.append({"op": "orient_joint", "joints": oriented, "aim": aim, "up": up})


def _compose_frame(st: _State, jid: str, c: np.ndarray, child_ids: list[str]) -> None:
    """Record a frame change ``C`` of joint ``jid``: its rotation channels become
    ``q C``, its children's ``C^-1 q`` (and their positions ``C^-1 p``)."""
    prev = st.frame_right.get(jid, np.array([0.0, 0.0, 0.0, 1.0]))
    st.frame_right[jid] = R.multiply(prev[None], c[None])[0]
    inv = R.conjugate(c)
    for cid in child_ids:
        prev_l = st.frame_left.get(cid, np.array([0.0, 0.0, 0.0, 1.0]))
        st.frame_left[cid] = R.multiply(inv[None], prev_l[None])[0]


def _op_rename_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "name"}, loc)
    t = st.table
    r = t.find(raw.get("joint"), f"{loc}.joint")
    new = t.unique(raw.get("name"), f"{loc}.name")
    old = r.name
    for c in t.children(old):
        c.parent = new
    r.name = new
    # Chain renames: a -> b then b -> c is a -> c.
    original = next((o for o, n in st.renames.items() if n == old), old)
    st.renames[original] = new
    st.report.append({"op": "rename_joint", "from": old, "to": new})


def _op_reparent_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "parent", "keep"}, loc)
    t = st.table
    r = t.find(raw.get("joint"), f"{loc}.joint")
    parent = None if raw.get("parent") is None else t.find(raw["parent"], f"{loc}.parent")
    keep = raw.get("keep", "world")
    if keep not in ("world", "local"):
        raise _err("INVALID_ARGUMENT", "keep is 'world' or 'local'", f"{loc}.keep")
    if parent is not None and (parent is r or parent in t.subtree(r.name)):
        raise _err(
            "SKELETON_CYCLE",
            f"{parent.name!r} is {r.name!r} or below it: reparenting would make a cycle",
            f"{loc}.parent",
        )
    models = t.models()
    m = models[r.name]
    r.parent = None if parent is None else parent.name
    if keep == "world":
        t.set_model(r, m, models)
    st.report.append(
        {"op": "reparent_joint", "joint": r.name, "parent": None if parent is None else parent.name}
    )


def _op_insert_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"name", "child", "at", "weights", "blend", "role", "deform"}, loc)
    t = st.table
    child = t.find(raw.get("child"), f"{loc}.child")
    parent = t.get(child.parent)
    if parent is None:
        raise _err("INVALID_ARGUMENT", "the child joint has no parent to insert under", loc)
    at = _fraction(raw.get("at", 0.5), f"{loc}.at", 1e-3, 1 - 1e-3)
    mode = raw.get("weights", "none")
    if mode not in ("none", "split"):
        raise _err("INVALID_ARGUMENT", "weights is 'none' or 'split'", f"{loc}.weights")
    blend = _fraction(raw.get("blend", 0.1), f"{loc}.blend", 0.0, 1.0)
    name = t.unique(raw.get("name"), f"{loc}.name")
    models = t.models()
    a, b = models[parent.name][:3, 3], models[child.name][:3, 3]
    if float(np.linalg.norm(b - a)) < MIN_BONE:
        raise _err("BONE_TOO_SHORT", f"the bone {parent.name!r} has no length", loc)
    m = models[parent.name].copy()
    m[:3, 3] = a + at * (b - a)
    import uuid

    row = _Row(str(uuid.uuid4()), name, parent.name, np.zeros(3), np.array([0, 0, 0, 1.0]),
               np.ones(3), None, "null", raw.get("role"), raw.get("deform", True))  # fmt: skip
    t.rows.append(row)
    t.set_model(row, m, models)
    child_m = models[child.name]
    child.parent = name
    t.set_model(child, child_m, t.models())
    if mode == "split":
        st.splits.append((parent.id, row.id, at, blend))
    st.added.append(name)
    st.report.append({"op": "insert_joint", "joint": name, "between": [parent.name, child.name]})


def _fraction(value: Any, location: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
        raise _err("INVALID_ARGUMENT", f"a fraction within {lo:g}..{hi:g}", location)
    return float(value)


def _op_split_bone(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "count", "names", "blend"}, loc)
    t = st.table
    r = t.find(raw.get("joint"), f"{loc}.joint")
    count = raw.get("count", 2)
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 16:
        raise _err("INVALID_ARGUMENT", "count is 2..16", f"{loc}.count")
    kids = t.children(r.name)
    if not kids:
        raise _err("INVALID_ARGUMENT", f"{r.name!r} has no child bone to split", f"{loc}.joint")
    names = raw.get("names") or [f"{r.name}.{i:02d}" for i in range(1, count)]
    if not isinstance(names, list) or len(names) != count - 1:
        raise _err("INVALID_ARGUMENT", f"names lists {count - 1} new joints", f"{loc}.names")
    child = kids[0]
    blend = raw.get("blend", 0.1)
    # Insert from the far end so each insertion splits what is left of the bone.
    for i in range(count - 1, 0, -1):
        _op_insert_joint(
            st,
            {"op": "insert_joint", "name": names[i - 1], "child": child.name, "at": i / (i + 1),
             "weights": "split", "blend": blend},
            loc,
            rig,
        )  # fmt: skip
        child = t.find(names[i - 1], loc)
    st.report.append({"op": "split_bone", "joint": r.name, "new": names})


def _mirror_limits(limits: str) -> str:
    """A joint's limits seen in its mirror image (``x -> -x``: rotations about x keep
    their sign, about y and z flip)."""
    if limits == "null":
        return limits
    lim = json.loads(limits)

    def flip(rng: list[float]) -> list[float]:
        return [-rng[1], -rng[0]]

    if "rotation" in lim:
        mn, mx = lim["rotation"]["min"], lim["rotation"]["max"]
        new_min, new_max = list(mn), list(mx)
        for i in (1, 2):
            new_min[i] = None if mx[i] is None else -mx[i]
            new_max[i] = None if mn[i] is None else -mn[i]
        lim = {"rotation": {"min": new_min, "max": new_max}}
    elif "hinge" in lim:
        h = dict(lim["hinge"])
        if h["axis"][1] != "x":
            h["min"], h["max"] = -h["max"], -h["min"]
        lim = {"hinge": h}
    else:
        letter = lim["axis"][1]
        if "twist" in lim and letter != "x":
            lim["twist"] = flip(lim["twist"])
        swing = lim.get("swing")
        if swing and "cone" not in swing:
            lim["swing"] = {k: (v if k == "x" else flip(v)) for k, v in swing.items()}
    return json.dumps(lim, sort_keys=True, separators=(",", ":"))


def _mirror_role(role: str | None) -> str | None:
    if role is None:
        return None
    if role.endswith(".L"):
        return role[:-2] + ".R"
    if role.endswith(".R"):
        return role[:-2] + ".L"
    return role


def _op_mirror_joints(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joints", "side"}, loc)
    t = st.table
    if ("joints" in raw) == ("side" in raw):
        raise _err("INVALID_ARGUMENT", "give joints or side", loc)
    if "side" in raw:
        if raw["side"] not in ("L", "R"):
            raise _err("INVALID_ARGUMENT", "side is 'L' or 'R'", f"{loc}.side")
        tokens = [pair[0 if raw["side"] == "L" else 1] for pair in rig.mirror]
        sources = [
            r for r in t.rows if any(r.name.endswith(s) or r.name.startswith(s) for s in tokens)
        ]
    else:
        if not isinstance(raw["joints"], list) or not raw["joints"]:
            raise _err("INVALID_ARGUMENT", "joints is a list of names", f"{loc}.joints")
        sources = [t.find(n, f"{loc}.joints[{i}]") for i, n in enumerate(raw["joints"])]
    order = {r.name: k for k, r in enumerate(t.rows)}
    sources.sort(key=lambda r: _depth(t, r) * 100000 + order[r.name])
    flip = np.diag([-1.0, 1.0, 1.0, 1.0])
    made, updated = [], []
    import uuid

    for src in sources:
        target_name = rig.mirror_name(src.name)
        if target_name is None or target_name == src.name:
            st.warn("MIRROR_UNNAMED", f"{src.name!r} has no mirror name", f"{loc}.joints")
            continue
        models = t.models()
        m = flip @ models[src.name] @ flip
        parent_name = None if src.parent is None else (rig.mirror_name(src.parent) or src.parent)
        if parent_name is not None and t.get(parent_name) is None:
            parent_name = src.parent
        dst = t.get(target_name)
        if dst is None:
            ident = np.array([0, 0, 0, 1.0])
            dst = _Row(str(uuid.uuid4()), target_name, parent_name, np.zeros(3), ident,
                       np.ones(3), None, "null", None, src.deform)  # fmt: skip
            t.rows.append(dst)
            made.append(target_name)
            st.added.append(target_name)
        else:
            if dst.parent != parent_name and parent_name is not None:
                if t.get(parent_name) in t.subtree(dst.name):
                    raise _err("SKELETON_CYCLE", f"mirroring {src.name!r} makes a cycle", loc)
                dst.parent = parent_name
            updated.append(target_name)
        kids = {k.name: models[k.name] for k in t.children(dst.name)}
        t.set_model(dst, m, models)
        models = t.models()
        for k in t.children(dst.name):
            if k.name in kids and rig.mirror_name(k.name) not in {s.name for s in sources}:
                t.set_model(k, kids[k.name], models)
        dst.tail = None if src.tail is None else _mirror_tail(src, dst, t)
        dst.limits = _mirror_limits(src.limits)
        role = _mirror_role(src.role)
        if role is not None and not any(r.role == role for r in t.rows if r is not dst):
            dst.role = role
        dst.deform = src.deform
    st.report.append({"op": "mirror_joints", "created": made, "updated": updated})


def _mirror_tail(src: _Row, dst: _Row, t: _Table) -> np.ndarray:
    models = t.models()
    flip = np.diag([-1.0, 1.0, 1.0])
    tip = models[src.name] @ np.append(src.tail, 1.0)
    tip_m = flip @ tip[:3]
    return np.linalg.solve(models[dst.name], np.append(tip_m, 1.0))[:3]


def _depth(t: _Table, r: _Row) -> int:
    d = 0
    p = t.get(r.parent)
    while p is not None:
        d += 1
        p = t.get(p.parent)
    return d


def _op_align_bone(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "direction"}, loc)
    t = st.table
    r = t.find(raw.get("joint"), f"{loc}.joint")
    kids = t.children(r.name)
    if not kids:
        raise _err("INVALID_ARGUMENT", f"{r.name!r} has no child bone", f"{loc}.joint")
    d = _direction(raw.get("direction"), f"{loc}.direction")
    models = t.models()
    a = models[r.name][:3, 3]
    child = kids[0]
    b = models[child.name][:3, 3]
    length = float(np.linalg.norm(b - a))
    new = models[child.name].copy()
    new[:3, 3] = a + d * length
    t.set_model(child, new, models)
    st.report.append({"op": "align_bone", "joint": r.name, "child": child.name})


def _op_set_joint(st: _State, raw: dict, loc: str, rig: Any) -> None:
    _check_keys(raw, {"joint", "role", "deform", "limits", "tail"}, loc)
    from .rig import _check_limits

    r = st.table.find(raw.get("joint"), f"{loc}.joint")
    if "role" in raw:
        r.role = raw["role"]
    if "deform" in raw:
        if not isinstance(raw["deform"], bool):
            raise _err("INVALID_ARGUMENT", "deform is true or false", f"{loc}.deform")
        r.deform = raw["deform"]
    if "limits" in raw:
        r.limits = (
            "null" if raw["limits"] is None else _check_limits(raw["limits"], f"{loc}.limits")
        )
    if "tail" in raw:
        r.tail = None if raw["tail"] is None else _vec(raw["tail"], 3, f"{loc}.tail", "tail")
    st.report.append(
        {"op": "set_joint", "joint": r.name, "changed": sorted(set(raw) - {"op", "joint"})}
    )


_HANDLERS = {
    "add_joint": _op_add_joint,
    "remove_joint": _op_remove_joint,
    "move_joint": _op_move_joint,
    "orient_joint": _op_orient_joint,
    "rename_joint": _op_rename_joint,
    "reparent_joint": _op_reparent_joint,
    "insert_joint": _op_insert_joint,
    "split_bone": _op_split_bone,
    "mirror_joints": _op_mirror_joints,
    "align_bone": _op_align_bone,
    "set_joint": _op_set_joint,
}


# --------------------------------------------------------------------------------------- #
# Building the new rig and its dependent data
# --------------------------------------------------------------------------------------- #


def _rename_name(name: str, renames: dict[str, str]) -> str:
    return renames.get(name, name)


def _new_rig_json(st: _State, rig: Any) -> dict[str, Any]:
    t = st.table
    renames = st.renames
    removed_names = {j.name for j in rig.joints if j.id in st.removed}  # names before the edit
    joints = []
    for r in t.rows:
        j: dict[str, Any] = {
            "id": r.id,
            "name": r.name,
            "parent": r.parent,
            "position": r.pos.tolist(),
            "rotation": r.rot.tolist(),
        }
        if not np.array_equal(r.scl, np.ones(3)):
            j["scale"] = r.scl.tolist()
        if r.tail is not None:
            j["tail"] = r.tail.tolist()
        if r.limits != "null":
            j["limits"] = json.loads(r.limits)
        if r.role is not None:
            j["role"] = r.role
        if not r.deform:
            j["deform"] = False
        joints.append(j)
    raw = rig.to_json()
    raw["joints"] = joints
    alive_ids = {r.id for r in t.rows}
    name_of_id = {r.id: r.name for r in t.rows}
    # Attachments follow their joint (ids), removed joints' to the weight target.
    atts = []
    for oid, jid in rig.attachments:
        while jid in st.removed and st.removed[jid] is not None:
            jid = st.removed[jid]
        if jid in alive_ids:
            atts.append({"object": oid, "joint": name_of_id[jid]})
        else:
            st.warn("ATTACHMENT_DROPPED", f"object {oid} lost its joint", "attachments")
    raw["attachments"] = atts
    # Constraints: renamed; those using removed joints are dropped (when allowed).
    cons = []
    for c in raw.get("constraints", []):
        used = _constraint_joints(c)
        gone = sorted(used & removed_names)
        if gone:
            if not (set(gone) <= st.drop_constraints):
                raise _err(
                    "JOINT_IN_USE",
                    f"constraint {c['name']!r} uses {gone}; remove it first or pass "
                    "constraints: 'remove'",
                    "operations",
                )
            st.warn("CONSTRAINT_REMOVED", f"constraint {c['name']!r} used {gone}", "constraints")
            continue
        cons.append(_rename_constraint(c, renames))
    raw["constraints"] = cons
    kept_constraints = {c["name"] for c in cons}
    controls = []
    for c in raw.get("controls", []):
        if c.get("joint") in removed_names:
            st.warn("CONTROL_REMOVED", f"control {c['name']!r} drove a removed joint", "controls")
            continue
        c = dict(c)
        if c.get("joint") is not None:
            c["joint"] = _rename_name(c["joint"], renames)
        if c.get("constraint") is not None and c["constraint"] not in kept_constraints:
            c.pop("constraint")
        controls.append(c)
    names = {c["name"] for c in controls}
    for c in controls:
        if c.get("mirror") is not None and c["mirror"] not in names:
            c.pop("mirror")
    raw["controls"] = controls
    correctives = []
    for c in raw.get("correctives", []):
        if c["joint"] in removed_names:
            st.warn("CORRECTIVE_REMOVED", f"corrective {c['name']!r} lost its joint", "correctives")
            continue
        correctives.append({**c, "joint": _rename_name(c["joint"], renames)})
    raw["correctives"] = correctives
    return raw


def _constraint_joints(c: dict[str, Any]) -> set[str]:
    out = set(c.get("chain") or [])
    if c.get("joint"):
        out.add(c["joint"])
    for key in ("target", "pole"):
        t = c.get(key)
        if isinstance(t, dict) and "joint" in t and "object" not in t:
            out.add(t["joint"])
    return out


def _rename_constraint(c: dict[str, Any], renames: dict[str, str]) -> dict[str, Any]:
    c = dict(c)
    if c.get("chain"):
        c["chain"] = [_rename_name(n, renames) for n in c["chain"]]
    if c.get("joint"):
        c["joint"] = _rename_name(c["joint"], renames)
    for key in ("target", "pole"):
        t = c.get(key)
        if isinstance(t, dict) and "joint" in t and "object" not in t:
            c[key] = {**t, "joint": _rename_name(t["joint"], renames)}
    return c


def _rename_path(path: str, renames: dict[str, str]) -> str:
    if path.startswith('joints["'):
        end = path.index('"]', 8)
        name = path[8:end]
        if name in renames:
            return f'joints["{renames[name]}"]' + path[end + 2 :]
    return path


def _path_joint(path: str) -> str | None:
    if path.startswith('joints["'):
        return path[8 : path.index('"]', 8)]
    return None


def edit_rig(
    system: AnimationSystem,
    armature: Any,
    operations: list[Any],
    *,
    skins: str = "rebind",
    animation: str = "keep",
    channels: str = "keep",
    tolerance: float = 0.01,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply structural ``operations`` to ``armature``'s rig (see the module docstring);
    one undo step. Returns a report: per operation, the dependent changes (skins, clips,
    masks), warnings and the validation of the result."""
    from ..rendering.scene_data import world_matrices
    from .rig import parse_rig

    obj = system._object(armature, "armature")
    rig = system.rig(obj)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig (add_rig first)", "armature")
    if skins not in SKIN_POLICIES:
        raise _err("INVALID_ARGUMENT", f"skins is one of {', '.join(SKIN_POLICIES)}", "skins")
    if animation not in ANIMATION_POLICIES:
        raise _err(
            "INVALID_ARGUMENT", f"animation is one of {', '.join(ANIMATION_POLICIES)}", "animation"
        )
    if channels not in CHANNEL_POLICIES:
        raise _err(
            "INVALID_ARGUMENT", f"channels is one of {', '.join(CHANNEL_POLICIES)}", "channels"
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0 < tolerance <= 10
    ):
        raise _err("INVALID_ARGUMENT", "tolerance is 0..10 degrees", "tolerance")
    if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_OPERATIONS:
        raise _err("INVALID_ARGUMENT", f"operations is a list of 1..{MAX_OPERATIONS}", "operations")
    worlds = world_matrices(system.scene)
    old_models = {j.id: rig.rest_model()[k].copy() for k, j in enumerate(rig.joints)}
    st = _State(_Table(rig), worlds[obj._id], old_models)
    for i, raw in enumerate(operations):
        loc = f"operations[{i}]"
        if not isinstance(raw, dict) or raw.get("op") not in _HANDLERS:
            raise _err(
                "INVALID_ARGUMENT",
                f"an operation is {{op, ...}} with op one of {', '.join(OPS)}",
                loc,
            )
        _HANDLERS[raw["op"]](st, raw, loc, rig)
    new_raw = _new_rig_json(st, rig)
    new_raw["object"] = obj._id
    new_rig = parse_rig(new_raw, "rig")
    _validate(st, new_rig)
    plan = _Plan(system, obj, rig, new_rig, st, skins, animation, channels, float(tolerance))
    plan.prepare(worlds)
    report = {
        "armature": obj.name,
        "operations": st.report,
        "joints": {"before": len(rig.joints), "after": len(new_rig.joints)},
        "renamed": dict(st.renames),
        "removed": [j.name for j in rig.joints if j.id in st.removed],
        "added": list(st.added),
        "skins": plan.skin_report,
        "clips": plan.clip_report,
        "dry_run": bool(dry_run),
        "diagnostics": [d.to_dict() for d in st.diagnostics],
    }
    if not dry_run:
        plan.apply()
    return report


def _validate(st: _State, rig: Any) -> None:
    model = rig.rest_model()
    for k, j in enumerate(rig.joints):
        if not np.all(np.isfinite(model[k])):
            raise _err("INVALID_RIG", f"joint {j.name!r} has a non-finite rest", "operations")
        if j.parent >= 0:
            length = float(np.linalg.norm(model[k][:3, 3] - model[j.parent][:3, 3]))
            if length < MIN_BONE:
                st.warn(
                    "BONE_TOO_SHORT",
                    f"joint {j.name!r} sits on its parent {rig.joints[j.parent].name!r}",
                    f"joints[{j.name}]",
                )


@dataclass
class _Plan:
    system: Any
    obj: Any
    old: Any
    new: Any
    st: _State
    skins: str
    animation: str
    channels: str
    tolerance: float
    skin_updates: list[tuple[Any, Any]] = field(default_factory=list)
    clip_updates: list[tuple[Any, Any]] = field(default_factory=list)
    binding_updates: list[tuple[Any, Any]] = field(default_factory=list)
    constraint_updates: list[tuple[Any, Any]] = field(default_factory=list)
    attached_moves: list[tuple[Any, np.ndarray]] = field(default_factory=list)
    skin_report: list[dict[str, Any]] = field(default_factory=list)
    clip_report: list[dict[str, Any]] = field(default_factory=list)

    def prepare(self, worlds: dict[str, np.ndarray]) -> None:
        self._skins(worlds)
        self._clips()
        self._masks_and_targets()
        if self.skins == "keep_bind":
            self._move_attached(worlds)

    # -- skins -----------------------------------------------------------------------------

    def _skins(self, worlds: dict[str, np.ndarray]) -> None:
        system, st = self.system, self.st
        rig_item = system._owned[("rig", self.obj._id)]
        new_model = {j.id: self.new.rest_model()[k] for k, j in enumerate(self.new.joints)}
        for (kind, oid), item in list(system._owned.items()):
            if kind != "skin" or item._definition.rig != rig_item.id:
                continue
            skin = item._definition
            dense = skin.dense()
            palette = list(skin.joints)
            ib = [skin.inverse_bind[i] for i in range(len(palette))]
            moved_weight = 0.0
            # Removed joints: weights to their targets (added to the palette if needed).
            for col, jid in enumerate(list(palette)):
                if jid not in st.removed:
                    continue
                target = st.removed[jid]
                while target in st.removed and st.removed[target] is not None:
                    target = st.removed[target]
                if target is None:
                    if np.any(dense[:, col] > 0):
                        raise _err(
                            "JOINT_IN_USE",
                            "a root joint with skin weights needs weights_to",
                            "operations",
                        )
                    continue
                if target not in palette:
                    palette.append(target)
                    ib.append(self._bind_for(skin, target, worlds, new_model))
                    dense = np.concatenate([dense, np.zeros((dense.shape[0], 1))], axis=1)
                tc = palette.index(target)
                moved_weight += float(dense[:, col].sum())
                dense[:, tc] += dense[:, col]
                dense[:, col] = 0.0
            # Splits: part of a parent bone's weights to the inserted joint.
            for parent_id, new_id, at, blend in st.splits:
                if parent_id not in palette:
                    continue
                if new_id not in palette:
                    palette.append(new_id)
                    ib.append(self._bind_for(skin, new_id, worlds, new_model))
                    dense = np.concatenate([dense, np.zeros((dense.shape[0], 1))], axis=1)
                share = self._split_share(
                    skin, oid, parent_id, new_id, at, blend, worlds, new_model
                )
                pc, nc = palette.index(parent_id), palette.index(new_id)
                part = dense[:, pc] * share
                dense[:, nc] += part
                dense[:, pc] -= part
            keep_cols = [c for c, jid in enumerate(palette) if jid not in st.removed]
            palette = [palette[c] for c in keep_cols]
            dense = dense[:, keep_cols]
            ib = [ib[c] for c in keep_cols]
            rebound = 0
            if self.skins == "rebind":
                for c, jid in enumerate(palette):
                    old = st.old_models.get(jid)
                    new = new_model[jid]
                    if old is not None and not np.array_equal(old, new):
                        ib[c] = np.linalg.solve(new, old @ ib[c])
                        rebound += 1
            k = skin.indices.shape[1]
            idx, w = _top_k(dense, max(k, 1))
            new_skin = skin.replace(
                joints=tuple(palette),
                inverse_bind=_frozen(np.array(ib)),
                indices=_frozen(idx),
                weights=_frozen(w),
                locked_joints=tuple(j for j in skin.locked_joints if j in palette),
            )
            self.skin_updates.append((item, new_skin))
            self.skin_report.append(
                {
                    "mesh": system.scene._objects[oid].name,
                    "rebound_joints": rebound,
                    "weight_moved": round(moved_weight, 6),
                    "joints": len(palette),
                }
            )

    def _bind_for(self, skin, jid, worlds, new_model) -> np.ndarray:
        """An inverse bind for a joint new to a skin: the joint's rest in mesh space."""
        m_rest = worlds[skin.object]
        a_rest = worlds[self.obj._id]
        return np.linalg.inv(new_model[jid]) @ np.linalg.inv(a_rest) @ m_rest

    def _split_share(self, skin, oid, parent_id, new_id, at, blend, worlds, new_model):
        mesh = self.system.scene._objects[oid].mesh
        pos = mesh.positions.astype(np.float64)
        to_arm = np.linalg.inv(worlds[self.obj._id]) @ worlds[oid]
        p = pos @ to_arm[:3, :3].T + to_arm[:3, 3]
        a = self.st.old_models[parent_id][:3, 3]
        b = new_model[new_id][:3, 3]
        # The bone's far end: where the original child was (the new joint sits at ``at``).
        end = a + (b - a) / at
        axis = end - a
        u = (p - a) @ axis / float(axis @ axis)
        if blend <= 0:
            return (u >= at).astype(np.float64)
        x = np.clip((u - (at - blend / 2)) / blend, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    # -- clips, masks, constraint targets --------------------------------------------------

    def _armature_clips(self) -> tuple[list[Any], list[str]]:
        """Clip items animating this armature, and the shared ones left alone."""
        system = self.system
        mine: dict[str, Any] = {}
        users: dict[str, set[str]] = {}
        for (kind, tid), item in system._bindings.items():
            if kind != P.OBJECT:
                continue
            for cid in item._definition.clips():
                users.setdefault(cid, set()).add(tid)
        for cid, objs in users.items():
            if self.obj._id in objs:
                mine[cid] = objs
        names = set(self.old.names)
        rigs = system._rigs()
        for item in system._clip_names.values():
            if item.id in users:
                continue
            joints = {_path_joint(p) for p, _ in item._definition.channels} - {None}
            if not joints or not joints <= names:
                continue
            owners = [
                a for a, (_i, r) in rigs.items() if all(r.index(j) is not None for j in joints)
            ]
            if owners == [self.obj._id]:
                mine[item.id] = {self.obj._id}
        shared = []
        out = []
        for cid, objs in mine.items():
            item = system._items[cid]
            others = objs - {self.obj._id}
            if any(("rig", o) in system._owned for o in others):
                shared.append(item._name)
                continue
            out.append(item)
        return out, shared

    def _clips(self) -> None:
        from .clip import Clip

        st = self.st
        renames = st.renames
        removed = {j.name for j in self.old.joints if j.id in st.removed}
        name_to_id = {j.name: j.id for j in self.old.joints}
        items, shared = self._armature_clips()
        for name in shared:
            st.warn(
                "CLIP_SHARED",
                f"clip {name!r} also animates another rig; it is left unchanged",
                "clips",
            )
        for item in items:
            clip = item._definition
            changed = False
            report: dict[str, Any] = {"clip": item._name}
            new_items = []
            dropped, kept_orphans = [], []
            converted: dict[str, list[str]] = {}
            pending_frames: dict[str, list[tuple[str, Any]]] = {}
            for path, ch in clip.channels:
                joint = _path_joint(path)
                if joint is not None and joint in removed:
                    if self.channels == "remove":
                        dropped.append(path)
                        changed = True
                        continue
                    kept_orphans.append(path)
                new_path = _rename_path(path, renames)
                if new_path != path:
                    changed = True
                jid = name_to_id.get(joint) if joint is not None else None
                if jid is not None and (jid in st.frame_right or jid in st.frame_left):
                    pending_frames.setdefault(jid, []).append((new_path, ch))
                    continue
                new_items.append((new_path, ch))
            for jid, chans in pending_frames.items():
                if self.animation == "keep":
                    new_items.extend(chans)
                    st.warn(
                        "ANIMATION_MEANING_CHANGED",
                        f"clip {item._name!r}: {[p for p, _ in chans]} were authored in a joint "
                        "frame that changed (animation='convert' rewrites them)",
                        "clips",
                    )
                    continue
                new_chans, errors = _convert_frames(
                    chans, st.frame_left.get(jid), st.frame_right.get(jid), self.tolerance
                )
                new_items.extend(new_chans)
                for kind, paths in errors.items():
                    converted.setdefault(kind, []).extend(paths)
                changed = True
            if dropped:
                report["removed_channels"] = dropped
            if kept_orphans:
                report["orphaned_channels"] = kept_orphans
                st.warn(
                    "CHANNELS_ORPHANED",
                    f"clip {item._name!r} keeps channels of removed joints: {kept_orphans[:6]}",
                    "clips",
                )
            if converted:
                report["converted"] = converted
            renamed = [p for p, _ in clip.channels if _rename_path(p, renames) != p]
            if renamed:
                report["renamed_channels"] = len(renamed)
            if changed:
                new_clip = Clip(
                    clip.target, tuple(sorted(new_items, key=lambda x: x[0])), clip.events,
                    clip.range, clip.metadata,
                )  # fmt: skip
                self.clip_updates.append((item, new_clip))
                self.clip_report.append(report)

    def _masks_and_targets(self) -> None:
        import dataclasses

        renames = self.st.renames
        if not renames:
            return
        system = self.system
        item = system._bindings.get((P.OBJECT, self.obj._id))
        if item is not None:
            b = item._definition
            layers = []
            changed = False
            for layer in b.layers:
                mask = []
                for key, w, desc in layer.mask:
                    new = key
                    if key.startswith("joint:") and key[6:] in renames:
                        new = "joint:" + renames[key[6:]]
                    elif key.startswith('joints["'):
                        new = _rename_path(key, renames)
                    changed |= new != key
                    mask.append((new, w, desc))
                layers.append(dataclasses.replace(layer, mask=tuple(mask)))
            if changed:
                self.binding_updates.append((item, dataclasses.replace(b, layers=tuple(layers))))
        # Constraints elsewhere that target this rig's joints ({object: ARM, joint: NAME}).
        from .constraints import parse_constraints

        for (kind, oid), other in system._owned.items():
            if kind not in ("constraints", "rig") or (kind == "rig" and oid == self.obj._id):
                continue
            cons = other._definition.constraints
            raw = [c.to_json() for c in cons]
            changed = False
            for c in raw:
                for key in ("target", "pole"):
                    t = c.get(key)
                    if (
                        isinstance(t, dict)
                        and t.get("object") == self.obj._id
                        and t.get("joint") in renames
                    ):
                        c[key] = {**t, "joint": renames[t["joint"]]}
                        changed = True
            if changed:
                parsed = parse_constraints(raw, "constraints", joints=kind == "rig")
                self.constraint_updates.append(
                    (other, dataclasses.replace(other._definition, constraints=parsed))
                )

    def _move_attached(self, worlds: dict[str, np.ndarray]) -> None:
        """``keep_bind``: attached parts move with their joints' new rest."""
        a = worlds[self.obj._id]
        new_model = {j.id: self.new.rest_model()[k] for k, j in enumerate(self.new.joints)}
        for oid, jid in self.new.attachments:
            old = self.st.old_models.get(jid)
            if old is None or np.array_equal(old, new_model[jid]):
                continue
            delta = a @ new_model[jid] @ np.linalg.inv(a @ old)
            self.attached_moves.append((self.system.scene._objects[oid], delta @ worlds[oid]))

    # -- apply -----------------------------------------------------------------------------

    def apply(self) -> None:
        system = self.system
        rig_item = system._owned[("rig", self.obj._id)]
        with system._op(f"Edit rig of {self.obj.name!r}"):
            system._set(rig_item, definition=self.new)
            for item, skin in self.skin_updates:
                system._set(item, definition=skin)
            for item, clip in self.clip_updates:
                system._set(item, definition=clip)
            for item, binding in self.binding_updates:
                system._set(item, definition=binding)
            for item, definition in self.constraint_updates:
                system._set(item, definition=definition)
            for obj, world in self.attached_moves:
                parent = obj.parent
                local = world if parent is None else np.linalg.solve(parent.world_matrix, world)
                p, q, s = decompose(local)
                obj.position = tuple(p.tolist())
                obj.rotation = tuple(_canonical(q).tolist())
                obj.scale = tuple(s.tolist())
            system._check_cycles()


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _top_k(dense: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` largest weights per row (ties: lower joint first), renormalized; rows
    with no weight keep joint 0 (refused upstream)."""
    k = min(k, dense.shape[1])
    order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(dense, order, axis=1)
    w = np.where(w > 0, w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0
    if np.any(empty):
        w[empty, 0] = 1.0
        total = w.sum(axis=1, keepdims=True)
    return order.astype(np.int32), w / total


# --------------------------------------------------------------------------------------- #
# Converting channels to a new joint frame
# --------------------------------------------------------------------------------------- #


def _convert_frames(
    chans: list[tuple[str, Any]],
    left: np.ndarray | None,
    right: np.ndarray | None,
    tolerance: float,
) -> tuple[list[tuple[str, Any]], dict[str, list[str]]]:
    """Channels of one joint rewritten for its new frame: rotations ``L q C``, positions
    ``L p`` (``L``: the parent's frame change inverted, ``C``: the joint's own)."""
    from .fitting import fit_samples
    from .tracks import RotationTrack

    ident = np.array([0.0, 0.0, 0.0, 1.0])
    lq = ident if left is None else left
    rq = ident if right is None else right
    out: list[tuple[str, Any]] = []
    notes: dict[str, list[str]] = {}
    by_field: dict[str, dict[str, Any]] = {}
    for path, ch in chans:
        spec = P.parse(path, P.OBJECT)
        if isinstance(ch, RotationTrack):
            a = ch.arrays()
            q = R.multiply(R.multiply(lq[None], a["quats"]), rq[None])
            tin = R.multiply(R.multiply(lq[None], a["in"]), rq[None])
            tout = R.multiply(R.multiply(lq[None], a["out"]), rq[None])
            out.append((path, ch.replace({"quats": q, "in": tin, "out": tout})))
            notes.setdefault("exact", []).append(path)
            continue
        by_field.setdefault(f"{spec.name}|{spec.field}", {})[spec.component] = (path, ch)
    for key, comps in by_field.items():
        name, fld = key.split("|")
        if fld == "rotation_euler" or fld == "position":
            if fld == "position" and left is None:
                out.extend(comps.values())
                continue
            curves = {c: v[1] for c, v in comps.items()}
            times = _sample_times([c for c in curves.values()])
            base = {0: 0.0, 1: 0.0, 2: 0.0}
            vals = np.stack(
                [
                    curves[c].evaluate(times) if c in curves else np.full(times.shape, base[c])
                    for c in range(3)
                ],
                axis=1,
            )
            if fld == "rotation_euler":
                q = R.from_euler(vals)
                q = R.multiply(R.multiply(lq[None], q), rq[None])
                new_vals = R.euler_filter(R.to_euler(q))
            else:
                new_vals = R.rotate(lq[None], vals)
            for c in range(3):
                path = f'joints["{name}"].{fld}.{"xyz"[c]}'
                if c not in curves and np.allclose(new_vals[:, c], base[c], atol=1e-12):
                    continue
                curve, info = fit_samples(
                    times,
                    new_vals[:, c],
                    tolerance if fld == "rotation_euler" else tolerance * 1e-3,
                )
                old = curves.get(c)
                if old is not None:
                    curve = curve.replace(extrapolation=list(old.extrapolation))
                out.append((path, curve))
                notes.setdefault("refit", []).append(f"{path} (max error {info['max_error']:.3g})")
        else:  # scale: unchanged by a rotation of frames only when uniform
            out.extend(comps.values())
            notes.setdefault("unchanged", []).extend(v[0] for v in comps.values())
    return out, notes


def _sample_times(curves: list[Any]) -> np.ndarray:
    lo = min(c.range[0] for c in curves)
    hi = max(c.range[1] for c in curves)
    keys = np.unique(np.concatenate([np.asarray(c.times) for c in curves]))
    if hi <= lo:
        return np.array([lo, lo + 1e-3])
    dense = np.linspace(lo, hi, max(2, int(math.ceil((hi - lo) * 120)) + 1))
    return np.unique(np.concatenate([keys, dense]))
