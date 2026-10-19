"""glTF export of skeletons, skins, morph targets and animations.

**Skeletons.** A rig's joints become glTF nodes under the armature's node (rest TRS,
converted like every node, see :mod:`.axes`), named after the joints and marked in
``extras.aether3d.joint``. Objects attached rigidly to a joint become children of its
node (their rest pose relative to it), with their Aether3D parent kept in extras.

**Skins.** A skinned mesh is carried by its own root node with no transform (glTF
ignores a skinned mesh node's transform and parents, and the Khronos validator warns
about either), marked ``extras.aether3d.skinned_mesh_of``; the object keeps its own
node, transform and children. ``inverseBindMatrices`` are the skin's exactly (they
have glTF's meaning: mesh space at bind to joint space; ``C M C^T`` converts them).
``JOINTS_n``/``WEIGHTS_n`` carry up to eight influences, strongest first, unused slots
at joint 0 with weight 0, weights renormalized in float32.

**Morph targets** become primitive ``targets`` (``POSITION`` and ``NORMAL`` offsets),
with ``mesh.weights`` (the defaults) and ``mesh.extras.targetNames``.

**Animations** (``animation=``):

* ``"clips"`` -- every clip with a target (the action or a strip of a binding, or the
  one rig whose joints a clip names) as a glTF animation of its own, played alone;
* ``"actions"`` -- only the clips playing as actions;
* ``"bake"`` -- what the scene shows (layers, constraints, IK, paths) sampled on the
  timeline's frames into ``LINEAR`` keys, reduced within the tolerances;
* ``"auto"`` (default) -- ``"bake"`` when layers, constraints, IK or paths shape the
  motion, else ``"clips"``; ``"none"``.

Clip channels map to glTF exactly where glTF can say the same thing: ``STEP`` for
constant keys, ``LINEAR`` for linear keys and for slerped rotations (glTF's ``LINEAR``
rotation is the shortest-path slerp; a deliberately long arc is split at its middle),
``CUBICSPLINE`` with one-sided slopes as in/out tangents for unweighted Bezier keys
(cubic Hermite in time) and imported glTF cubics. Everything else -- easings, weighted
handles, Euler rotations, squad, timing curves, modifiers, extrapolation inside the
range -- is fitted: ``CUBICSPLINE`` keys with the motion's exact values and slopes are
added where the error is largest until every check point is within tolerance
(``position_tolerance`` scene units, ``rotation_tolerance`` degrees,
``weight_tolerance``). The largest error is reported per animation; an instantaneous
jump becomes a transition of ``JUMP_SECONDS``. Values are float32 in the file.
Visibility, camera, light, material and constraint channels have no glTF equivalent:
lossy diagnostics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from . import axes

if TYPE_CHECKING:
    from ..scene import Scene

FLOAT = 5126
UNSIGNED_BYTE = 5121
UNSIGNED_SHORT = 5123
MODES = ("auto", "clips", "actions", "bake", "none")
JUMP_SECONDS = 1e-4
MAX_KEYS_PER_CHANNEL = 200_000
CHECKS_PER_INTERVAL = 16
_C4 = np.eye(4)
_C4[:3, :3] = axes.AXIS_MATRIX


def matrix_to_gltf(m: np.ndarray) -> np.ndarray:
    return _C4 @ m @ _C4.T


def _diag(code: str, severity: str, message: str, location: str | None = None, **kw) -> Diagnostic:
    return Diagnostic(code, severity, message, location, **kw)


# --------------------------------------------------------------------------------------- #
# Skeletons, skins, morph targets
# --------------------------------------------------------------------------------------- #


@dataclass
class Rigging:
    """What of a scene's animation document reaches the glTF nodes."""

    system: Any = None
    rigs: dict[str, Any] = field(default_factory=dict)  # armature id -> Rig
    rig_owner: dict[str, str] = field(default_factory=dict)  # rig item id -> armature id
    skins: dict[str, Any] = field(default_factory=dict)  # mesh object id -> Skin
    morphs: dict[str, Any] = field(default_factory=dict)  # mesh object id -> Morph
    attached: dict[str, tuple[str, int]] = field(default_factory=dict)  # id -> (arm, joint)
    joint_nodes: dict[str, list[int]] = field(default_factory=dict)  # arm id -> node per joint
    carriers: dict[str, int] = field(default_factory=dict)  # skinned object id -> node
    mesh_node: dict[str, int] = field(default_factory=dict)  # object id -> node with its mesh
    object_node: dict[str, int] = field(default_factory=dict)  # object id -> its own node


def collect(scene: Scene) -> Rigging:
    system = getattr(scene, "_animation", None)
    out = Rigging(system=system)
    if system is None:
        return out
    for (kind, oid), item in system._owned.items():
        if not item._alive or oid not in scene._objects:
            continue
        if kind == "rig":
            out.rigs[oid] = item._definition
            out.rig_owner[item.id] = oid
        elif kind == "morph":
            out.morphs[oid] = item._definition
    for (kind, oid), item in system._owned.items():
        if kind != "skin" or not item._alive or oid not in scene._objects:
            continue
        if item._definition.rig in out.rig_owner:
            out.skins[oid] = item._definition
    for arm, rig in out.rigs.items():
        for obj_id, joint_id in rig.attachments:
            k = rig.index_of_id(joint_id)
            if obj_id in scene._objects and k is not None and obj_id not in out.skins:
                out.attached[obj_id] = (arm, k)
    return out


def rig_diagnostics(scene: Scene, rigging: Rigging, diagnostics: list[Diagnostic]) -> None:
    """What glTF cannot carry of the rigs: skinning other than linear blend (lossy), and
    the authoring data kept only as Aether3D extras or not exported (information)."""
    if not rigging.rigs:
        return
    for oid, skin in sorted(rigging.skins.items(), key=lambda kv: scene._objects[kv[0]].name):
        if skin.method != "linear":
            name = scene._objects[oid].name
            diagnostics.append(_diag(
                "SKINNING_METHOD_LOSSY", WARNING,
                f"{name!r} deforms with {skin.method.replace('_', ' ')} skinning; glTF skins "
                "are linear blend skinning: exported with the same weights, it bends with "
                "less volume at twisting and bending joints",
                f"/skins/{skin_name(scene, oid)}", lossy=True,
            ))  # fmt: skip
        if skin.smooth != "null":
            name = scene._objects[oid].name
            diagnostics.append(_diag(
                "SKIN_SMOOTHING_LOSSY", WARNING,
                f"{name!r} relaxes its skinning with corrective smoothing (delta mush); glTF "
                "skins have none: exported with the same weights, its folds and pinches come "
                "back",
                f"/skins/{skin_name(scene, oid)}", lossy=True,
            ))  # fmt: skip
        if skin.normals == "surface":
            name = scene._objects[oid].name
            diagnostics.append(_diag(
                "SKIN_NORMALS_DIFFER", INFO,
                f"{name!r} shades with normals that follow its deformed surface; glTF viewers "
                "carry the rest normals with the joints (the geometry is the same)",
                f"/skins/{skin_name(scene, oid)}",
            ))  # fmt: skip
    kept, dropped = set(), set()
    for rig in rigging.rigs.values():
        if any(j.role for j in rig.joints):
            kept.add("joint roles")
        if any(j.limits != "null" for j in rig.joints):
            kept.add("joint limits")
        if any(not j.deform for j in rig.joints):
            kept.add("helper joints")
        if any(j.span != (0.0, 1.0) for j in rig.joints):
            kept.add("joint spans")
        if rig.controls:
            dropped.add("control definitions (their objects are exported as nodes)")
        if rig.correctives:
            dropped.add("corrective drivers (their shapes' weights are baked)")
        if rig.landmarks or rig.template:
            dropped.add("landmarks and template")
        if rig.constraints:
            dropped.add("constraints (their effect is baked)")
    if kept or dropped:
        parts = []
        if kept:
            parts.append("kept as Aether3D extras on the joint nodes: " + ", ".join(sorted(kept)))
        if dropped:
            parts.append("not exported: " + ", ".join(sorted(dropped)))
        diagnostics.append(_diag("RIG_AUTHORING_DATA", INFO, "; ".join(parts), "/nodes"))


def trs_to_gltf(pos: Any, rot: Any, scl: Any) -> dict[str, list[float]]:
    node: dict[str, list[float]] = {}
    pos, rot, scl = tuple(map(float, pos)), tuple(map(float, rot)), tuple(map(float, scl))
    if pos != (0.0, 0.0, 0.0):
        node["translation"] = list(axes.vector_to_gltf(pos))
    if rot != (0.0, 0.0, 0.0, 1.0):
        node["rotation"] = list(axes.rotation_to_gltf(rot))
    if scl != (1.0, 1.0, 1.0):
        node["scale"] = list(axes.scale_to_gltf(scl))
    return node


def add_skeletons(
    scene: Scene,
    rigging: Rigging,
    nodes: list[dict[str, Any]],
    node_of: dict[str, int],
    roots: list[int],
    extras_key: str,
) -> None:
    """Append joint nodes, move attached objects under their joints (in place)."""
    from ..animation.constraints import decompose
    from ..rendering.scene_data import world_matrices

    if not rigging.rigs:
        return
    worlds = world_matrices(scene)  # authored (rest) worlds
    for arm, rig in rigging.rigs.items():
        base = len(nodes)
        indices = list(range(base, base + len(rig.joints)))
        rigging.joint_nodes[arm] = indices
        for k, joint in enumerate(rig.joints):
            node: dict[str, Any] = {"name": joint.name}
            node.update(trs_to_gltf(joint.position, joint.rotation, joint.scale))
            children = [base + c for c, j in enumerate(rig.joints) if j.parent == k]
            if children:
                node["children"] = children
            stored: dict[str, Any] = {"joint": joint.name, "armature": arm}
            # Stage 18 joint semantics (Aether3D's own: other tools ignore extras).
            if joint.tail is not None:
                stored["tail"] = [float(v) for v in joint.tail]
            if joint.role is not None:
                stored["role"] = joint.role
            if not joint.deform:
                stored["deform"] = False
            if joint.span != (0.0, 1.0):
                stored["span"] = [float(v) for v in joint.span]
            if joint.limits != "null":
                stored["limits"] = json.loads(joint.limits)
            node["extras"] = {extras_key: stored}
            nodes.append(node)
        arm_node = nodes[node_of[arm]]
        arm_node["children"] = arm_node.get("children", []) + [
            base + k for k, j in enumerate(rig.joints) if j.parent < 0
        ]
    model_of = {arm: rig.rest_model() for arm, rig in rigging.rigs.items()}
    for obj_id, (arm, k) in rigging.attached.items():
        n = node_of[obj_id]
        obj = scene._objects[obj_id]
        # Detach the node from where it was, re-hang it under the joint's node.
        for other in nodes:
            if n in other.get("children", []):
                other["children"] = [c for c in other["children"] if c != n]
                if not other["children"]:
                    del other["children"]
        if n in roots:
            roots.remove(n)
        joint_node = nodes[rigging.joint_nodes[arm][k]]
        joint_node["children"] = joint_node.get("children", []) + [n]
        local = np.linalg.inv(worlds[arm] @ model_of[arm][k]) @ worlds[obj_id]
        pos, rot, scl = decompose(local)
        node = nodes[n]
        for key in ("translation", "rotation", "scale"):
            node.pop(key, None)
        node.update(trs_to_gltf(pos, rot, scl))
        stored = node.setdefault("extras", {}).setdefault(extras_key, {})
        stored["attached"] = {
            "joint": rigging.rigs[arm].joints[k].name,
            "parent": obj.parent.id if obj.parent is not None else None,
            "translation": list(obj.position),
            "rotation": list(obj.rotation),
            "scale": list(obj.scale),
        }
    # (node order: an attached node may now come before its new parent: glTF allows it)


def skin_attributes(skin: Any, source: np.ndarray | None, buffers: Any) -> dict[str, int]:
    """``JOINTS_n``/``WEIGHTS_n`` accessors of ``skin`` (``source``: vertex map)."""
    idx = np.asarray(skin.indices, np.int64)
    w = np.asarray(skin.weights, np.float64)
    if source is not None:
        idx, w = idx[source], w[source]
    n, k = idx.shape
    # Merge repeated joints of a vertex, strongest first, unused slots at joint 0.
    merged_i = np.zeros((n, k), np.int64)
    merged_w = np.zeros((n, k))
    order = np.argsort(-w, axis=1, kind="stable")
    si = np.take_along_axis(idx, order, axis=1)
    sw = np.take_along_axis(w, order, axis=1)
    for col in range(k):
        same = np.zeros(n, bool)
        for prev in range(col):
            same |= (si[:, prev] == si[:, col]) & (sw[:, prev] > 0)
        merged_w[:, col] = np.where(same, 0.0, sw[:, col])
        merged_i[:, col] = np.where(same | (sw[:, col] <= 0), 0, si[:, col])
        for prev in range(col):
            hit = (si[:, prev] == si[:, col]) & (sw[:, prev] > 0) & (sw[:, col] > 0)
            merged_w[hit, prev] += sw[hit, col]
    sets = (k + 3) // 4
    width = sets * 4
    full_i = np.zeros((n, width), np.int64)
    full_w = np.zeros((n, width))
    full_i[:, :k] = merged_i
    full_w[:, :k] = merged_w
    full_w[full_w <= 0] = 0.0
    full_i[full_w <= 0] = 0
    # Renormalize in float32 so the float32 weights sum to 1 as closely as possible.
    w32 = (full_w / full_w.sum(axis=1, keepdims=True)).astype(np.float32)
    rest = w32[:, 1:].astype(np.float64).sum(axis=1)
    w32[:, 0] = (1.0 - rest).astype(np.float32)
    joint_type = UNSIGNED_BYTE if len(skin.joints) <= 256 else UNSIGNED_SHORT
    jdtype = np.uint8 if joint_type == UNSIGNED_BYTE else np.uint16
    out = {}
    for s in range(sets):
        cols = slice(4 * s, 4 * s + 4)
        out[f"JOINTS_{s}"] = buffers.add(
            np.ascontiguousarray(full_i[:, cols].astype(jdtype)), joint_type, "VEC4", 34962, False
        )
        out[f"WEIGHTS_{s}"] = buffers.add(
            np.ascontiguousarray(w32[:, cols]), FLOAT, "VEC4", 34962, False
        )
    return out


def morph_targets(morph: Any, source: np.ndarray | None, buffers: Any, with_normals: bool):
    """Primitive ``targets`` of ``morph``."""
    targets = []
    for t in range(len(morph.names)):
        pos = np.asarray(morph.positions[t], np.float32)
        if source is not None:
            pos = pos[source]
        entry = {"POSITION": buffers.add(axes.vectors_to_gltf(pos), FLOAT, "VEC3", 34962, True)}
        if with_normals and morph.normals is not None:
            nrm = np.asarray(morph.normals[t], np.float32)
            if source is not None:
                nrm = nrm[source]
            entry["NORMAL"] = buffers.add(axes.vectors_to_gltf(nrm), FLOAT, "VEC3", 34962, False)
        targets.append(entry)
    return targets


def add_skins(
    scene: Scene,
    rigging: Rigging,
    nodes: list[dict[str, Any]],
    node_of: dict[str, int],
    buffers: Any,
) -> list[dict[str, Any]]:
    """glTF ``skins`` (one per skinned object), wired to the carrier nodes."""
    skins = []
    for obj_id, skin in rigging.skins.items():
        if obj_id not in rigging.carriers:
            continue
        arm = rigging.rig_owner[skin.rig]
        rig = rigging.rigs[arm]
        joints = [rigging.joint_nodes[arm][rig.index_of_id(j)] for j in skin.joints]
        ibm = np.array([matrix_to_gltf(m) for m in skin.inverse_bind])
        data = np.ascontiguousarray(ibm.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32))
        accessor = buffers.add(data, FLOAT, "MAT4", None, False)
        skins.append(
            {"joints": joints, "inverseBindMatrices": accessor, "name": skin_name(scene, obj_id)}
        )
        nodes[rigging.carriers[obj_id]]["skin"] = len(skins) - 1
    return skins


def skin_name(scene: Scene, obj_id: str) -> str:
    return f"{scene._objects[obj_id].name} Skin"


# --------------------------------------------------------------------------------------- #
# Animation channels: exact where glTF can say it, else fitted
# --------------------------------------------------------------------------------------- #


@dataclass
class Channel:
    node: int
    path: str  # translation | rotation | scale | weights
    times: np.ndarray
    values: np.ndarray  # (k, c) Aether3D axes
    interpolation: str  # LINEAR | STEP | CUBICSPLINE
    in_tan: np.ndarray | None = None
    out_tan: np.ndarray | None = None
    error: float = 0.0
    exact: bool = True
    extrapolation: tuple[str, str] | None = None  # (kept in extras when not constant)


class _Source:
    """A vector property as functions of time: ``value(t) -> (n, c)`` and
    ``derivative(t) -> (n, c)`` (right derivatives), its key times and how glTF can
    hold it exactly (``"step"``, ``"linear"``, ``"cubic"`` or ``None``)."""

    def __init__(
        self,
        value,
        derivative,
        keys: np.ndarray,
        exact: str | None,
        rotation: bool,
        extrapolates: bool = False,
    ):
        self.value = value
        self.derivative = derivative
        self.keys = keys
        self.exact = exact
        self.rotation = rotation
        self.extrapolates = extrapolates  # (beyond its keys it is not held constant)
        self.extrapolation: tuple[str, str] | None = None  # (common to its curves)


def _curve_kind(curve: Any) -> str | None:
    """How glTF holds an F-curve exactly: ``step``, ``linear``, ``cubic`` or ``None``."""
    from ..animation.curves import _SEG_BEZIER, _SEG_CONSTANT, _SEG_HERMITE, _SEG_JUMP, _SEG_LINEAR

    if curve.modifiers:
        return None
    kinds = set(curve.compiled().kind.tolist())
    if not kinds:
        return "step"
    if _SEG_JUMP in kinds:
        return None
    if kinds <= {_SEG_CONSTANT}:
        return "step"
    if kinds <= {_SEG_LINEAR}:
        return "linear"
    if kinds <= {_SEG_LINEAR, _SEG_HERMITE}:
        return "cubic"
    if _SEG_BEZIER in kinds:
        return None  # weighted handles: not a cubic in time
    return None


def _combine_kinds(kinds: list[str | None]) -> str | None:
    """The interpolation able to hold every component exactly (``None``: fit)."""
    present = [k for k in kinds if k != "constant"]
    if any(k is None for k in present):
        return None
    if not present:
        return "step"
    kinds_set = set(present)
    if kinds_set == {"step"}:
        return "step"
    if "step" in kinds_set:
        return None
    if kinds_set == {"linear"}:
        return "linear"
    return "cubic"


def scalar_source(curves: list[Any], rest: list[float]) -> _Source:
    """A vector from per-component F-curves (``None``: the rest value, constant)."""

    def value(t: np.ndarray) -> np.ndarray:
        out = np.empty((t.size, len(curves)))
        for c, curve in enumerate(curves):
            out[:, c] = rest[c] if curve is None else curve.evaluate(t)
        return out

    def derivative(t: np.ndarray) -> np.ndarray:
        out = np.zeros((t.size, len(curves)))
        for c, curve in enumerate(curves):
            if curve is not None:
                d = curve.evaluate(t, order=1)[1]
                out[:, c] = np.where(np.isfinite(d), d, 0.0)
        return out

    keys = np.unique(np.concatenate([c.times for c in curves if c is not None] or [np.zeros(0)]))
    kinds = ["constant" if c is None else _curve_kind(c) for c in curves]
    moving = any(c is not None and c.extrapolation != ("constant", "constant") for c in curves)
    src = _Source(value, derivative, keys, _combine_kinds(kinds), False, moving)
    modes = {tuple(c.extrapolation) for c in curves if c is not None}
    src.extrapolation = modes.pop() if len(modes) == 1 else None
    return src


def track_source(track: Any) -> _Source:
    from ..animation import rotation as R
    from ..animation.tracks import CONSTANT, CUBIC, SLERP, TIMING_LINEAR

    def value(t: np.ndarray) -> np.ndarray:
        return np.asarray(track.evaluate(t)).reshape(-1, 4)

    def derivative(t: np.ndarray) -> np.ndarray:
        q, omega = track.evaluate(t, order=1)
        q = np.asarray(q).reshape(-1, 4)
        omega = np.asarray(omega).reshape(-1, 3)
        w = np.concatenate([omega, np.zeros((omega.shape[0], 1))], axis=1)
        return 0.5 * R.multiply(w, q)

    a = track.arrays()
    interp = set(a["interpolation"][:-1].tolist()) if len(track) > 1 else {CONSTANT}
    linear_timing = bool(np.all(a["timing"][:-1] == TIMING_LINEAR)) if len(track) > 1 else True
    exact = None
    if interp <= {CONSTANT}:
        exact = "step"
    elif interp <= {SLERP} and linear_timing:
        exact = "slerp"
    elif interp <= {CUBIC} and linear_timing:
        exact = "glcubic"
    moving = tuple(track.extrapolation) != ("constant", "constant")
    src = _Source(value, derivative, np.asarray(track.times, np.float64), exact, True, moving)
    src.track = track  # type: ignore[attr-defined]
    src.extrapolation = tuple(track.extrapolation)
    return src


def euler_source(curves: list[Any], rest: list[float]) -> _Source:
    from ..animation import rotation as R

    def angles(t: np.ndarray) -> np.ndarray:
        out = np.empty((t.size, 3))
        for c, curve in enumerate(curves):
            out[:, c] = rest[c] if curve is None else curve.evaluate(t)
        return out

    def value(t: np.ndarray) -> np.ndarray:
        return R.from_euler(angles(t))

    def derivative(t: np.ndarray) -> np.ndarray:
        h = 1e-6
        a, b = R.from_euler(angles(t - h)), R.from_euler(angles(t + h))
        b = R.align(a, b)
        return (b - a) / (2 * h)

    keys = np.unique(np.concatenate([c.times for c in curves if c is not None] or [np.zeros(0)]))
    src = _Source(value, derivative, keys, None, True, True)
    modes = {tuple(c.extrapolation) for c in curves if c is not None}
    mode = modes.pop() if len(modes) == 1 else None
    # (quaternion tracks repeat but cannot extrapolate linearly)
    src.extrapolation = mode if mode is not None and "linear" not in mode else None
    return src


def _error(a: np.ndarray, b: np.ndarray, rotation: bool) -> np.ndarray:
    if rotation:
        na = a / np.linalg.norm(a, axis=1, keepdims=True)
        nb = b / np.linalg.norm(b, axis=1, keepdims=True)
        d = np.clip(np.abs(np.sum(na * nb, axis=1)), 0.0, 1.0)
        return np.degrees(2.0 * np.arccos(d))
    return np.max(np.abs(a - b), axis=1)


def _hermite(t: np.ndarray, times, values, ins, outs) -> np.ndarray:
    k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    span = times[k + 1] - times[k]
    s = ((t - times[k]) / span)[:, None]
    s2, s3 = s * s, s * s * s
    return (
        (2 * s3 - 3 * s2 + 1) * values[k]
        + (s3 - 2 * s2 + s) * span[:, None] * outs[k]
        + (-2 * s3 + 3 * s2) * values[k + 1]
        + (s3 - s2) * span[:, None] * ins[k + 1]
    )


def _align_signs(values: np.ndarray, *tangents: np.ndarray) -> None:
    for i in range(1, values.shape[0]):
        if float(values[i] @ values[i - 1]) < 0.0:
            values[i] = -values[i]
            for tan in tangents:
                tan[i] = -tan[i]


def fit_source(src: _Source, t0: float, t1: float, tol: float) -> tuple[Channel, list[str]]:
    """Keys holding ``src`` on ``[t0, t1]``: exact when possible, else fitted."""
    notes: list[str] = []
    keys = src.keys[(src.keys >= t0) & (src.keys <= t1)]
    times = np.unique(np.concatenate([[t0], keys, [t1]]))
    exact = src.exact
    if exact and src.extrapolates and src.keys.size and (t0 < src.keys[0] or t1 > src.keys[-1]):
        exact = None  # (the range reaches extrapolation glTF would hold constant)
    if exact == "step":
        values = src.value(times)
        if src.rotation:
            _align_signs(values)
        return Channel(-1, "", times, values, "STEP"), notes
    if exact == "linear":
        values = src.value(times)
        jumps = _jump_times(src, times)
        if jumps.size == 0:
            return Channel(-1, "", times, values, "LINEAR"), notes
    if exact == "slerp":
        track = src.track  # type: ignore[attr-defined]
        times, values = _slerp_keys(track, t0, t1)
        return Channel(-1, "", times, values, "LINEAR"), notes
    if exact == "glcubic":
        track = src.track  # type: ignore[attr-defined]
        a = track.arrays()
        inside = (a["times"] >= t0) & (a["times"] <= t1)
        if inside.all() and t0 == a["times"][0] and t1 == a["times"][-1]:
            values, ins, outs = a["quats"].copy(), a["in"].copy(), a["out"].copy()
            return Channel(-1, "", a["times"].copy(), values, "CUBICSPLINE", ins, outs), notes
    # Hermite keys with the motion's own values and one-sided slopes, refined until every
    # check point is within the tolerance (or the key budget is spent).
    if times.size < 2:
        times = np.array([t0, t0 + max(t1 - t0, 1e-3)])
    jumps = _jump_times(src, times)
    if jumps.size:
        pre = jumps - JUMP_SECONDS
        times = np.unique(np.concatenate([times, pre[pre > t0]]))
        notes.append("JUMP_APPROXIMATED")
    rounds = 0
    while True:
        rounds += 1
        values = src.value(times)
        outs = src.derivative(times)
        # Left slopes: the derivative just before each key, extrapolated to the key
        # (second order: exact to rounding for the cubic segments of a curve).
        gap = np.diff(times, prepend=times[0] - 1.0)
        step = np.minimum(1e-7 * np.maximum(1.0, np.abs(times)), 0.25 * gap)
        ins = 2.0 * src.derivative(times - step) - src.derivative(times - 2.0 * step)
        ins[0] = outs[0]
        if src.rotation:
            _align_signs(values, ins, outs)
        n = times.size
        frac = np.arange(1, CHECKS_PER_INTERVAL + 1) / (CHECKS_PER_INTERVAL + 1)
        checks = (times[:-1, None] + (times[1:] - times[:-1])[:, None] * frac[None, :]).reshape(-1)
        want = src.value(checks)
        got = _hermite(checks, times, values, ins, outs)
        err = _error(got, want, src.rotation).reshape(n - 1, CHECKS_PER_INTERVAL)
        worst = err.max(axis=1)
        bad = np.flatnonzero((worst > tol) & (np.diff(times) > 2 * JUMP_SECONDS))
        if bad.size == 0 or n >= MAX_KEYS_PER_CHANNEL:
            if bad.size and n >= MAX_KEYS_PER_CHANNEL:
                notes.append("KEY_BUDGET_SPENT")
            # Hermite keys of a cubic-in-time curve at its own keys hold it exactly.
            held = exact == "cubic" and rounds == 1 and not jumps.size
            ch = Channel(-1, "", times, values, "CUBICSPLINE", ins, outs, float(err.max()), held)
            return ch, notes
        add = checks.reshape(n - 1, CHECKS_PER_INTERVAL)[bad, err[bad].argmax(axis=1)]
        times = np.unique(np.concatenate([times, add]))


def _jump_times(src: _Source, times: np.ndarray) -> np.ndarray:
    """Key times where the value jumps (the left limit differs from the value)."""
    if times.size < 2:
        return np.zeros(0)
    inner = times[1:]
    left = src.value(inner - 1e-9 * np.maximum(1.0, np.abs(inner)))
    right = src.value(inner)
    gap = _error(left, right, src.rotation)
    scale = 1e-6 if not src.rotation else 1e-4
    return inner[gap > scale]


def _slerp_keys(track: Any, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Slerp keys on ``[t0, t1]`` whose shortest arcs are the track's own arcs: its keys
    in the range, the range's ends, and the middle of every arc the track takes the long
    way round (each half is then under 180 degrees, where shortest and stored agree)."""
    from ..animation import rotation as R

    times = np.asarray(track.times, np.float64)
    bounds = np.unique(np.concatenate([[t0], times[(times > t0) & (times < t1)], [t1]]))
    out_t = [bounds[0]]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if not b > a:
            continue
        qa, qm, qb = np.asarray(track.evaluate(np.array([a, 0.5 * (a + b), b]))).reshape(3, 4)
        short = R.slerp(qa[None], qb[None], np.array([0.5]), shortest=True)[0]
        if abs(float(short @ qm)) < 1.0 - 1e-9:
            out_t.append(0.5 * (a + b))
        out_t.append(b)
    times_out = np.array(out_t)
    vals = np.asarray(track.evaluate(times_out)).reshape(-1, 4).copy()
    _align_signs(vals)
    return times_out, vals


def reduce_linear(
    times: np.ndarray, values: np.ndarray, tol: float, rotation: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Baked samples as ``LINEAR`` keys: the fewest (greedily) within ``tol``."""
    from ..animation import rotation as R

    n = times.size
    if n <= 2:
        return times, values, 0.0
    keep = [0, n - 1]
    while True:
        idx = np.array(sorted(keep))
        k = np.clip(np.searchsorted(idx, np.arange(n), side="right") - 1, 0, idx.size - 2)
        a, b = idx[k], idx[k + 1]
        u = (times - times[a]) / (times[b] - times[a])
        if rotation:
            got = R.slerp(values[a], values[b], u, shortest=True)
        else:
            got = values[a] + (values[b] - values[a]) * u[:, None]
        err = _error(got, values, rotation)
        j = int(err.argmax())
        if err[j] <= tol or j in keep:
            return times[idx], values[idx], float(err.max())
        keep.append(j)


# --------------------------------------------------------------------------------------- #
# Planning animations
# --------------------------------------------------------------------------------------- #

_TRS = {"position": "translation", "rotation": "rotation", "scale": "scale"}


def _needs_bake(system: Any) -> list[str]:
    """Reasons the scene's motion is more than its clips played alone."""
    reasons = []
    for item in system._bindings.values():
        b = item._definition
        if not b.mute and any(layer.strips and not layer.mute for layer in b.layers):
            reasons.append("NLA layers")
            break
    for (kind, _), item in system._owned.items():
        if not item._alive:
            continue
        if kind == "constraints" and item._definition.constraints:
            reasons.append("constraints")
        elif kind == "rig" and item._definition.constraints:
            reasons.append("rig constraints (IK)")
        if kind == "rig" and item._definition.correctives:
            reasons.append("corrective shapes")
    return sorted(set(reasons))


def _clip_targets(scene: Scene, rigging: Rigging) -> dict[str, list[tuple[str, bool]]]:
    """Clip id -> ``[(object id, as its action?)]``."""
    from ..animation import paths as P

    system = rigging.system
    out: dict[str, list[tuple[str, bool]]] = {}
    for (kind, tid), item in system._bindings.items():
        if kind != P.OBJECT or tid not in scene._objects:
            continue
        b = item._definition
        if b.action is not None:
            out.setdefault(b.action, []).append((tid, not b.mute))
        for layer in b.layers:
            for strip in layer.strips:
                out.setdefault(strip.clip, [])
                if all(t != tid for t, _ in out[strip.clip]):
                    out[strip.clip].append((tid, False))
    # Unbound clips naming joints: the one rig having them all.
    for item in system._clip_names.values():
        if item.id in out:
            continue
        joints = set()
        for path, _ in item._definition.channels:
            if path.startswith("joints["):
                joints.add(P.parse(path, P.OBJECT).name)
        if not joints:
            continue
        found = [
            arm
            for arm, rig in rigging.rigs.items()
            if all(rig.index(j) is not None for j in joints)
        ]
        if len(found) == 1:
            out[item.id] = [(found[0], False)]
    return out


def clip_sources(
    scene: Scene, rigging: Rigging, clip: Any, target: str
) -> tuple[dict[tuple[int, str], _Source], list[str]]:
    """``(node, glTF path) -> source`` for one clip played on ``target``, and the paths
    glTF cannot carry."""
    from ..animation import paths as P
    from ..animation import rotation as R
    from ..animation.curves import FCurve

    obj = scene._objects[target]
    rig = rigging.rigs.get(target)
    groups: dict[tuple[str, str], dict[str, Any]] = {}  # (owner, field) -> pieces
    lost: list[str] = []
    for path, ch in clip.channels:
        spec = P.parse(path, clip.target)
        if spec.group == "joint":
            if rig is None or rig.index(spec.name) is None:
                lost.append(path)
                continue
            owner = ("joint", spec.name)
        elif spec.group == "transform":
            owner = ("object", target)
        elif spec.group == "morph":
            owner = ("morph", target)
        else:
            lost.append(path)
            continue
        if spec.group == "morph":
            groups.setdefault(owner, {})[spec.name] = ch
            continue
        fieldname = "rotation" if spec.field in ("rotation", "rotation_euler") else spec.field
        piece = groups.setdefault((owner[0] + ":" + owner[1], fieldname), {})
        if spec.field == "rotation":
            piece["track"] = ch
        elif spec.field == "rotation_euler":
            piece.setdefault("euler", [None, None, None])[spec.component] = ch
        else:
            piece.setdefault("parts", [None, None, None])[spec.component] = ch
    sources: dict[tuple[int, str], _Source] = {}
    for key, piece in groups.items():
        if key[0] == "morph":
            morph = rigging.morphs.get(target)
            if morph is None or target not in rigging.mesh_node:
                lost.extend(f'morph["{n}"]' for n in piece)
                continue
            curves = [piece.get(n) for n in morph.names]
            for n in piece:
                if n not in morph.names:
                    lost.append(f'morph["{n}"]')
            sources[(rigging.mesh_node[target], "weights")] = scalar_source(
                curves, list(morph.defaults)
            )
            continue
        owner, fieldname = key
        kind, name = owner.split(":", 1)
        if kind == "joint":
            k = rig.index(name)
            node = rigging.joint_nodes[target][k]
            pos, rot, scl = rig.rest_trs()
            rest = {"position": pos[k], "rotation": rot[k], "scale": scl[k]}
        else:
            node = rigging.object_node[target]
            rest = {"position": obj.position, "rotation": obj.rotation, "scale": obj.scale}
        if fieldname == "rotation":
            if "track" in piece:
                sources[(node, "rotation")] = track_source(piece["track"])
            else:
                rest_e = R.to_euler(np.asarray(rest["rotation"], np.float64)[None])[0]
                sources[(node, "rotation")] = euler_source(piece["euler"], list(rest_e))
        else:
            parts = piece["parts"]
            for c in range(3):
                if parts[c] is not None and not isinstance(parts[c], FCurve):
                    parts[c] = None
            sources[(node, _TRS[fieldname])] = scalar_source(
                parts, [float(v) for v in rest[fieldname]]
            )
    return sources, lost


def bake_sources(
    scene: Scene, rigging: Rigging, node_parent: dict[int, int], fps: Any, t0: float, t1: float
) -> tuple[dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], list[str]]:
    """Sampled ``(node, path) -> (times, values)`` of what the scene shows."""
    from ..animation.constraints import decompose
    from ..animation.time import FrameRate, frame_range
    from ..rendering.scene_data import world_matrices

    system = rigging.system
    rate = system.fps if fps is None else FrameRate(fps, "fps")
    frames = frame_range(rate, t0, t1)
    times = rate.frame_times(frames)
    if times.size == 0 or times[0] > t0:
        times = np.unique(np.concatenate([[t0], times]))
    if times[-1] < t1:
        times = np.concatenate([times, [t1]])
    object_node = rigging.object_node
    node_world: dict[int, list[np.ndarray]] = {}
    morph_w: dict[str, list[np.ndarray]] = {}
    for t in times.tolist():
        overlay = system.overlay(t)
        worlds = world_matrices(scene, overlay)
        rig_eval = system.rigging(t) if rigging.rigs else None
        for oid, n in object_node.items():
            node_world.setdefault(n, []).append(worlds[oid])
        for arm, nodes in rigging.joint_nodes.items():
            if rig_eval is not None and arm in rig_eval.joints:
                model = rig_eval.joints[arm][1]
            else:
                model = rigging.rigs[arm].rest_model()
            for k, n in enumerate(nodes):
                node_world.setdefault(n, []).append(worlds[arm] @ model[k])
        pose = system.evaluate(t)
        for oid, morph in rigging.morphs.items():
            got = pose.morphs.get(oid, {})
            # (corrective shapes add to the animated weights: what the scene shows)
            add = rig_eval.morph_extra.get(oid, {}) if rig_eval is not None else {}
            morph_w.setdefault(oid, []).append(
                np.array(
                    [
                        got.get(nm, d) + add.get(nm, 0.0)
                        for nm, d in zip(morph.names, morph.defaults, strict=True)
                    ]
                )
            )
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for n, mats in node_world.items():
        w = np.array(mats)
        parent = node_parent.get(n)
        if parent is not None and parent in node_world:
            local = np.linalg.solve(np.array(node_world[parent]), w)
        else:
            local = w
        trs = [decompose(m) for m in local]
        pos = np.array([x[0] for x in trs])
        rot = np.array([x[1] for x in trs])
        scl = np.array([x[2] for x in trs])
        _align_signs(rot)
        for path, vals in (("translation", pos), ("rotation", rot), ("scale", scl)):
            if np.max(np.abs(vals - vals[0])) > 1e-10:
                out[(n, path)] = (times, vals)
    for oid, ws in morph_w.items():
        vals = np.array(ws)
        if oid in rigging.mesh_node and np.max(np.abs(vals - vals[0])) > 1e-12:
            out[(rigging.mesh_node[oid], "weights")] = (times, vals)
    return out, []


def encode_channel(ch: Channel, buffers: Any) -> tuple[dict[str, Any], int, int]:
    """Accessors of a channel (glTF axes, float32): ``(sampler, input, output)``."""
    t = ch.times.astype(np.float32)
    keep = np.concatenate([[True], np.diff(t.astype(np.float64)) > 0])
    vals = ch.values[keep]
    ins = ch.in_tan[keep] if ch.in_tan is not None else None
    outs = ch.out_tan[keep] if ch.out_tan is not None else None
    t = t[keep]

    def convert(a: np.ndarray) -> np.ndarray:
        if ch.path == "translation":
            return axes.vectors_to_gltf(a.astype(np.float64))
        if ch.path == "scale":
            return a[:, [0, 2, 1]].astype(np.float64)
        if ch.path == "rotation":
            out = np.empty_like(a, dtype=np.float64)
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = a[:, 0], a[:, 2], -a[:, 1], a[:, 3]
            return out + 0.0
        return a.astype(np.float64)

    v = convert(vals)
    if ch.path == "rotation":
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
    kind = {"translation": "VEC3", "scale": "VEC3", "rotation": "VEC4", "weights": "SCALAR"}[
        ch.path
    ]
    if ch.interpolation == "CUBICSPLINE":
        a_, b_ = convert(ins), convert(outs)
        data = np.stack([a_, v, b_], axis=1).reshape(-1, v.shape[1])
    else:
        data = v
    if ch.path == "weights":
        data = data.reshape(-1, 1)
    inp = buffers.add(t.reshape(-1, 1), FLOAT, "SCALAR", None, True)
    out = buffers.add(np.ascontiguousarray(data.astype(np.float32)), FLOAT, kind, None, False)
    return {"input": inp, "output": out, "interpolation": ch.interpolation}, inp, out


def build_animations(
    scene: Scene,
    rigging: Rigging,
    node_parent: dict[int, int],
    buffers: Any,
    diagnostics: list[Diagnostic],
    *,
    mode: str = "auto",
    fps: Any = None,
    position_tolerance: float = 1e-4,
    rotation_tolerance: float = 0.01,
    weight_tolerance: float = 1e-4,
) -> list[dict[str, Any]]:
    """glTF ``animations`` of the scene (see the module docstring)."""
    system = rigging.system
    if mode == "none" or system is None or not (system._bindings or system._owned):
        return []
    if not system._bindings:
        return []
    tol = {"translation": position_tolerance, "scale": position_tolerance,
           "rotation": rotation_tolerance, "weights": weight_tolerance}  # fmt: skip
    reasons = _needs_bake(system)
    if mode == "auto":
        mode = "bake" if reasons else "clips"
        if reasons:
            diagnostics.append(_diag(
                "ANIMATION_BAKED", INFO,
                f"the motion depends on {', '.join(reasons)}: exported as baked keys "
                "(animation='clips' exports the clips themselves)",
                "/animations",
            ))  # fmt: skip
    elif mode in ("clips", "actions") and reasons:
        diagnostics.append(_diag(
            "ANIMATION_NOT_BAKED", WARNING,
            f"{', '.join(reasons)} shape the motion but clips are exported alone "
            "(animation='bake' exports what the scene shows)",
            "/animations",
        ))  # fmt: skip
    if mode in ("clips", "actions"):  # (baking reports every channel it cannot represent)
        _check_bound_properties(system, diagnostics)
    animations: list[dict[str, Any]] = []
    if mode == "bake":
        t0, t1 = system.start, system.end
        if t0 < 0:
            diagnostics.append(_diag("NEGATIVE_TIMES_CLIPPED", WARNING,
                                     "glTF times start at 0: the timeline before 0 is not exported",
                                     "/animations", lossy=True))  # fmt: skip
            t0 = 0.0
        if not t1 > t0:
            return []
        sampled, _ = bake_sources(scene, rigging, node_parent, fps, t0, t1)
        channels = []
        worst: dict[str, float] = {}
        for (node, path), (times, vals) in sorted(sampled.items()):
            kt, kv, err = reduce_linear(times, vals, tol[path], path == "rotation")
            worst[path] = max(worst.get(path, 0.0), err)
            channels.append(Channel(node, path, kt, kv, "LINEAR", error=err, exact=False))
        _check_property_channels(scene, system, None, diagnostics)
        if channels:
            animations.append(_encode_animation("Scene", channels, buffers))
            animations[-1]["extras"] = {"aether3d": {"baked": True, "max_error": worst}}
        return animations
    targets = _clip_targets(scene, rigging)
    for name, item in sorted(system._clip_names.items(), key=lambda kv: kv[1]._serial):
        users = targets.get(item.id, [])
        if mode == "actions":
            users = [u for u in users if u[1]]
        if not users:
            if mode == "clips":
                diagnostics.append(_diag("CLIP_WITHOUT_TARGET", INFO,
                                         f"clip {name!r} plays on no object: not exported",
                                         f"/animations/{name}"))  # fmt: skip
            continue
        clip = item._definition
        t0, t1 = clip.frame_range()
        if t0 < 0:
            diagnostics.append(_diag("NEGATIVE_TIMES_CLIPPED", WARNING,
                                     f"clip {name!r} has keys before 0, which glTF cannot hold",
                                     f"/animations/{name}", lossy=True))  # fmt: skip
            t0 = 0.0
        if not t1 >= t0:
            continue
        channels = []
        worst = {}
        approximated = 0
        for target, _ in users:
            sources, lost = clip_sources(scene, rigging, clip, target)
            if lost:
                diagnostics.append(_diag(
                    "ANIMATED_PROPERTY_NOT_EXPORTED", WARNING,
                    f"clip {name!r}: glTF animates node transforms and morph weights only; "
                    f"not exported: {', '.join(sorted(set(lost))[:8])}",
                    f"/animations/{name}", count=len(set(lost)), lossy=True,
                ))  # fmt: skip
            for (node, path), src in sorted(sources.items(), key=lambda kv: kv[0]):
                ch, notes = fit_source(src, t0, max(t1, t0), tol[path])
                ch.node, ch.path = node, path
                spans = src.keys.size and src.keys[0] == t0 and src.keys[-1] == t1
                if src.extrapolation not in (None, ("constant", "constant")) and spans:
                    ch.extrapolation = src.extrapolation
                channels.append(ch)
                worst[path] = max(worst.get(path, 0.0), ch.error)
                approximated += 0 if ch.exact else 1
                for note in notes:
                    diagnostics.append(_diag(note, WARNING if note != "JUMP_APPROXIMATED" else INFO,
                                             _NOTE_TEXT[note], f"/animations/{name}"))  # fmt: skip
        if channels:
            anim = _encode_animation(name, channels, buffers)
            anim["extras"] = {
                "aether3d": {
                    "clip": name,
                    "actions": [t for t, as_action in users if as_action],
                    "fitted_channels": approximated,
                    "max_error": worst,
                }
            }
            animations.append(anim)
    return animations


_NOTE_TEXT = {
    "JUMP_APPROXIMATED": f"a jump in value is held as a {JUMP_SECONDS:g} s transition "
    "(glTF key times must increase)",
    "KEY_BUDGET_SPENT": "a channel needs more keys than the budget to meet the tolerance; "
    "the reported error exceeds it",
}


def _check_property_channels(scene, system, clip, diagnostics) -> None:
    """Channels glTF cannot animate (baking: constraint influences shape the baked
    transforms, so they are represented; cameras, lights and materials are not)."""
    from ..animation import paths as P

    lost = set()
    for item in system._clip_names.values():
        for path, _ in item._definition.channels:
            spec = P.parse(path, item._definition.target)
            if spec.group in ("transform", "joint", "morph"):
                continue
            if spec.group == "constraint":
                continue  # (baked: their effect is in the joint and node keys)
            lost.add(path)
    if lost:
        diagnostics.append(_diag(
            "ANIMATED_PROPERTY_NOT_EXPORTED", WARNING,
            "glTF animates node transforms and morph weights only; not exported: "
            + ", ".join(sorted(lost)[:8]),
            "/animations", count=len(lost), lossy=True,
        ))  # fmt: skip


def _check_bound_properties(system, diagnostics) -> None:
    """Bindings of the world and of materials (0.20.0: reported, not dropped silently):
    glTF has no world, and animates no material property here."""
    from ..animation import paths as P

    counts = {P.WORLD: 0, P.MATERIAL: 0}
    for item in system._binding_items():
        kind = item._definition.target[0]
        if kind in counts and not item._definition.mute and item._definition.clips():
            counts[kind] += 1
    lost = (["the world (background, ambient light, environment, atmosphere)"]
            if counts[P.WORLD] else []) + (
        [f"{counts[P.MATERIAL]} material(s)"] if counts[P.MATERIAL] else [])  # fmt: skip
    if lost:
        diagnostics.append(_diag(
            "ANIMATED_PROPERTY_NOT_EXPORTED", WARNING,
            "glTF animates node transforms and morph weights only; the animation of "
            + " and ".join(lost) + " is not exported",
            "/animations", count=counts[P.WORLD] + counts[P.MATERIAL], lossy=True,
        ))  # fmt: skip


def _encode_animation(name: str, channels: list[Channel], buffers: Any) -> dict[str, Any]:
    samplers, out_channels = [], []
    for ch in channels:
        sampler, _, _ = encode_channel(ch, buffers)
        samplers.append(sampler)
        entry: dict[str, Any] = {
            "sampler": len(samplers) - 1,
            "target": {"node": ch.node, "path": ch.path},
        }
        if ch.extrapolation is not None:  # (glTF holds; Aether3D repeats as authored)
            entry["extras"] = {"aether3d": {"extrapolation": list(ch.extrapolation)}}
        out_channels.append(entry)
    return {"name": name, "channels": out_channels, "samplers": samplers}


def check_mode(mode: Any) -> str:
    from ..errors import ValidationError

    if mode not in MODES:
        raise ValidationError(f"animation must be one of {', '.join(MODES)}, got {mode!r}")
    return mode


def tolerance_check(name: str, value: Any) -> float:
    from ..errors import ValidationError

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (math.isfinite(value) and value > 0)
    ):
        raise ValidationError(f"{name} must be a positive number, got {value!r}")
    return float(value)
