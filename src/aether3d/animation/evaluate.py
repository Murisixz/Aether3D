"""Evaluating a scene's animation at a time: a compiled program, vectorized per frame.

``evaluate(system, t)`` returns an immutable :class:`AnimationPose` -- the values the
animation gives every animated property at ``t`` -- without changing the scene.

**Compilation** (once per animation revision): every binding's action and strips become
*contributions* (a clip played with a time mapping, a weight, a layer mode and a rank:
the order in which it blends), and every channel of a contribution's clip a *query*
into one :class:`~.batch.CurveBatch` (scalars) or :class:`~.batch.RotationBatch`
(quaternions), aimed at a *slot* (one animated property component of one target).
Euler channels are grouped per owner and turned into quaternion queries. Additive layers
store each channel's reference value (the clip at the start of its range); accumulating
strips store each channel's change across a cycle.

**A frame**: the strips' clip times and weights (vectorized over strips), one batched
curve evaluation and one batched rotation evaluation for every query, then blending
layer by layer from the targets' authored (rest) values, which are cached per scene
revision: a replace layer averages its strips by weight and blends the average over what
is below with influence ``min(sum w, 1) * layer weight * mask`` (``slerp`` for
rotations); an additive layer adds ``w * layer weight * mask * (x - ref)`` (rotations:
``base * (ref^-1 x)^(w f)``). See :mod:`.nla`. Values outside a property's valid range
are clamped and
reported (``VALUE_CLAMPED``). Nothing is kept between frames: evaluating ``t`` gives the
same bits whatever was evaluated before.

**Order** (the documented contract): action, then layers bottom to top, strips in order
within a layer; then rigs and constraints (``rig.py``), then deformation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from . import paths as P
from . import rotation as R
from .batch import CurveBatch, RotationBatch
from .curves import FCurve
from .nla import Strip, layer_weight, strip_times
from .tracks import RotationTrack

if TYPE_CHECKING:
    from .system import AnimationSystem

REPLACE, ADDITIVE = 0, 1


@dataclass
class _Contribution:
    binding: int  # index into program.targets
    clip: Any  # Clip
    strip: Strip | None  # None: the action (clip time = scene time)
    layer: Any  # Layer or None
    rank: int  # the layer's position in its binding's stack (0: the action)
    mode: int
    clip_range: tuple[float, float]
    strip_rank: int = 0  # the strip's position within its layer


@dataclass
class Program:
    """A compiled animation document (see the module docstring)."""

    revision: int
    targets: list[tuple[str, str]] = field(default_factory=list)  # (kind, id) per binding
    contributions: list[_Contribution] = field(default_factory=list)
    # scalar queries
    curves: CurveBatch | None = None
    q_curve: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    q_contrib: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    q_slot: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    q_mask: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_delta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slots: list[tuple[int, P.PathSpec]] = field(default_factory=list)  # (target, spec)
    # rotation queries (quaternion tracks, and Euler triples)
    tracks: RotationBatch | None = None
    r_track: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    r_contrib: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    r_slot: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    r_mask: np.ndarray = field(default_factory=lambda: np.zeros(0))
    r_ref: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    r_delta_log: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Euler queries: three scalar curves (or the rest value) per (contribution, owner)
    e_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))  # -1: rest
    e_contrib: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    e_slot: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    e_mask: np.ndarray = field(default_factory=lambda: np.zeros(0))
    e_ref: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    e_delta_log: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rot_slots: list[tuple[int, str | None]] = field(default_factory=list)  # (target, joint)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    rest_cache: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class AnimationPose:
    """The animation's values at one time (read-only; see :func:`evaluate`).

    * ``transforms``: object id -> ``(position, rotation, scale)`` (local; objects with
      any animated transform channel; ``rotation`` a unit quaternion),
    * ``visible``: object id -> bool,
    * ``fields``: ``(object id, "camera" | "light" | "appearance")`` -> ``{field: value}``,
    * ``materials``: material id -> ``{parameter: value}``,
    * ``joints``: rig object id -> joint name -> ``(position, rotation, scale)``,
    * ``morphs``: object id -> ``{target name: weight}``,
    * ``influences``: object id -> ``{constraint name: influence}``.
    """

    time: float
    revision: int
    ids: tuple[str, ...]
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    visible: dict[str, bool]
    fields: dict[tuple[str, str], dict[str, Any]]
    materials: dict[str, dict[str, Any]]
    joints: dict[str, dict[str, tuple]]
    morphs: dict[str, dict[str, float]]
    influences: dict[str, dict[str, float]]
    diagnostics: tuple[Diagnostic, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def transforms(self) -> dict[str, tuple[tuple, tuple, tuple]]:
        return {
            oid: (
                tuple(self.position[k].tolist()),
                tuple(self.rotation[k].tolist()),
                tuple(self.scale[k].tolist()),
            )
            for k, oid in enumerate(self.ids)
        }

    @property
    def empty(self) -> bool:
        return not (
            self.ids or self.visible or self.fields or self.materials or self.joints
            or self.morphs or self.extra.get("local")
        )  # fmt: skip


# --------------------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------------------- #


def compile_program(system: AnimationSystem) -> Program:
    """Compile the animation document (bindings, clips) into a :class:`Program`."""
    prog = Program(system.revision)
    scene = system.scene
    curves: list[FCurve] = []
    curve_index: dict[int, int] = {}
    tracks: list[RotationTrack] = []
    track_index: dict[int, int] = {}
    slot_index: dict[tuple[int, str], int] = {}
    rot_index: dict[tuple[int, str | None], int] = {}

    def curve_id(c: FCurve) -> int:
        k = curve_index.get(id(c))
        if k is None:
            k = curve_index[id(c)] = len(curves)
            curves.append(c)
        return k

    def track_id(tr: RotationTrack) -> int:
        k = track_index.get(id(tr))
        if k is None:
            k = track_index[id(tr)] = len(tracks)
            tracks.append(tr)
        return k

    def slot(target: int, spec: P.PathSpec) -> int:
        key = (target, spec.path)
        k = slot_index.get(key)
        if k is None:
            k = slot_index[key] = len(prog.slots)
            prog.slots.append((target, spec))
        return k

    def rot_slot(target: int, joint: str | None) -> int:
        key = (target, joint)
        k = rot_index.get(key)
        if k is None:
            k = rot_index[key] = len(prog.rot_slots)
            prog.rot_slots.append(key)
        return k

    q: dict[str, list] = {n: [] for n in ("curve", "contrib", "slot", "mask", "ref", "delta")}
    r: dict[str, list] = {n: [] for n in ("track", "contrib", "slot", "mask", "ref", "delta")}
    e: dict[str, list] = {n: [] for n in ("curves", "contrib", "slot", "mask", "ref", "delta")}
    for item in system._binding_items():
        binding = item._definition
        kind, tid = binding.target
        if binding.mute:
            continue
        if kind == P.OBJECT and tid not in scene._objects:
            prog.diagnostics.append(_missing(item, "object", tid))
            continue
        if kind == P.MATERIAL and scene.materials.find_material(tid) is None:
            prog.diagnostics.append(_missing(item, "material", tid))
            continue
        b = len(prog.targets)
        prog.targets.append((kind, tid))
        rig_item = system._owned.get(("rig", tid)) if kind == P.OBJECT else None
        rig = rig_item._definition if rig_item is not None else None
        entries: list[tuple[Any, Strip | None, Any, int, int, int]] = []
        if binding.action is not None:
            entries.append((binding.action, None, None, REPLACE, 0, 0))
        for li, layer in enumerate(binding.layers):
            if layer.mute:
                continue
            for si, strip in enumerate(layer.strips):
                if not strip.mute:
                    mode = ADDITIVE if layer.mode == "additive" else REPLACE
                    entries.append((strip.clip, strip, layer, mode, li + 1, si))
        for clip_id, strip, layer, mode, rank, strip_rank in entries:
            clip_item = system._items.get(clip_id)
            if clip_item is None or not clip_item._alive:
                prog.diagnostics.append(
                    Diagnostic(
                        "CLIP_MISSING",
                        WARNING,
                        f"{item._name!r} plays a clip that no longer exists ({clip_id}); "
                        "it is skipped",
                        f"animation.bindings[{item._name}]",
                    )
                )
                continue
            clip = clip_item._definition
            if clip.target != kind:
                prog.diagnostics.append(
                    Diagnostic(
                        "CLIP_TARGET_MISMATCH",
                        WARNING,
                        f"clip {clip_item._name!r} animates {clip.target}s, not {kind}s; skipped",
                        f"animation.bindings[{item._name}]",
                    )
                )
                continue
            ci = len(prog.contributions)
            crange = clip.frame_range()
            prog.contributions.append(
                _Contribution(b, clip, strip, layer, rank, mode, crange, strip_rank)
            )
            eulers: dict[str | None, list[int]] = {}
            for path, channel in clip.channels:
                spec = P.parse(path, clip.target)
                assert not isinstance(spec, list)
                mask = _mask_weight(layer, spec, rig)
                if mask <= 0.0:
                    continue
                if spec.kind == P.ROTATION:
                    rs = rot_slot(b, spec.name if spec.group == "joint" else None)
                    tr = channel
                    r["track"].append(track_id(tr))
                    r["contrib"].append(ci)
                    r["slot"].append(rs)
                    r["mask"].append(mask)
                    r["ref"].append(
                        tr.evaluate(crange[0])
                        if mode == ADDITIVE
                        else np.array([0.0, 0.0, 0.0, 1.0])
                    )
                    if strip is not None and strip.accumulate:
                        a, z = _trim(strip, crange)
                        qa, qz = tr.evaluate(a), tr.evaluate(z)
                        r["delta"].append(R.log(R.multiply(R.align(qa, qz), R.conjugate(qa))))
                    else:
                        r["delta"].append(np.zeros(3))
                    continue
                if spec.field == "rotation_euler":
                    owner = spec.name if spec.group == "joint" else None
                    triple = eulers.setdefault(owner, [-1, -1, -1, mask])
                    triple[spec.component] = curve_id(channel)
                    continue
                s = slot(b, spec)
                q["curve"].append(curve_id(channel))
                q["contrib"].append(ci)
                q["slot"].append(s)
                q["mask"].append(mask)
                q["ref"].append(float(channel.evaluate(crange[0])) if mode == ADDITIVE else 0.0)
                if strip is not None and strip.accumulate:
                    a, z = _trim(strip, crange)
                    q["delta"].append(float(channel.evaluate(z)) - float(channel.evaluate(a)))
                else:
                    q["delta"].append(0.0)
            for owner, (cx, cy, cz, mask) in eulers.items():
                e["curves"].append([cx, cy, cz])
                e["contrib"].append(ci)
                e["slot"].append(rot_slot(b, owner))
                e["mask"].append(mask)
                e["ref"].append(None)  # (filled below once the batch exists)
                e["delta"].append(None)
    prog.curves = CurveBatch(curves)
    prog.tracks = RotationBatch(tracks)
    prog.q_curve = np.array(q["curve"], np.int64)
    prog.q_contrib = np.array(q["contrib"], np.int64)
    prog.q_slot = np.array(q["slot"], np.int64)
    prog.q_mask = np.array(q["mask"], np.float64)
    prog.q_ref = np.array(q["ref"], np.float64)
    prog.q_delta = np.array(q["delta"], np.float64)
    prog.r_track = np.array(r["track"], np.int64)
    prog.r_contrib = np.array(r["contrib"], np.int64)
    prog.r_slot = np.array(r["slot"], np.int64)
    prog.r_mask = np.array(r["mask"], np.float64)
    prog.r_ref = np.array(r["ref"], np.float64).reshape(-1, 4)
    prog.r_delta_log = np.array(r["delta"], np.float64).reshape(-1, 3)
    prog.e_curves = np.array(e["curves"], np.int64).reshape(-1, 3)
    prog.e_contrib = np.array(e["contrib"], np.int64)
    prog.e_slot = np.array(e["slot"], np.int64)
    prog.e_mask = np.array(e["mask"], np.float64)
    # Euler references and per-cycle changes need the rest Euler angles of their owners,
    # which belong to the scene (they are filled per evaluation, see _euler_quats).
    prog.e_ref = np.zeros((len(e["contrib"]), 4))
    prog.e_delta_log = np.zeros((len(e["contrib"]), 3))
    return prog


def _missing(item: Any, kind: str, tid: str) -> Diagnostic:
    return Diagnostic(
        "TARGET_MISSING",
        WARNING,
        f"animation {item._name!r} targets a {kind} that no longer exists ({tid}); it is skipped",
        f"animation.bindings[{item._name}]",
    )


def _trim(strip: Strip, crange: tuple[float, float]) -> tuple[float, float]:
    a = crange[0] if strip.clip_start is None else strip.clip_start
    z = crange[1] if strip.clip_end is None else strip.clip_end
    return a, z


def _mask_weight(layer: Any, spec: P.PathSpec, rig: Any = None) -> float:
    """The weight a layer's mask gives a channel (1 without a mask; 0 when not listed). A
    joint entry with ``descendants`` covers the joint's whole subtree in ``rig``."""
    if layer is None or not layer.mask:
        return 1.0
    best = 0.0
    for key, w, desc in layer.mask:
        if key.startswith("joint:"):
            if spec.group != "joint":
                continue
            if spec.name == key[6:] or (desc and _descends(rig, spec.name, key[6:])):
                best = max(best, w)
        elif spec.path == key or spec.path.startswith(key + ".") or spec.path.startswith(key + "["):
            best = max(best, w)
    return best


def _descends(rig: Any, joint: str, ancestor: str) -> bool:
    """Whether ``joint`` lies below ``ancestor`` in ``rig``."""
    if rig is None:
        return False
    k, a = rig.index(joint), rig.index(ancestor)
    if k is None or a is None:
        return False
    parents = rig.parents
    p = int(parents[k])
    while p >= 0:
        if p == a:
            return True
        p = int(parents[p])
    return False


# --------------------------------------------------------------------------------------- #
# Rest values
# --------------------------------------------------------------------------------------- #


def _rest_scalar(
    scene: Any, target: tuple[str, str], spec: P.PathSpec, joints: Any
) -> float | None:
    kind, tid = target
    if kind == P.MATERIAL:
        m = scene.materials.find_material(tid)
        value = m.params.get(spec.field)
        if spec.component is not None:
            return float(value[spec.component])
        return float(value) if value is not None else None
    obj = scene._objects[tid]
    g = spec.group
    if g == "transform":
        t = obj._transform
        src = {"position": t._position, "scale": t._scale}.get(spec.field)
        if spec.field == "rotation_euler":
            return float(R.to_euler(np.array(t._rotation))[0][spec.component])
        return float(src[spec.component])
    if g == "visible":
        return 1.0 if obj._visible else 0.0
    if g == "camera":
        cam = obj._camera
        if cam is None:
            return None
        return float(getattr(cam, spec.field))
    if g == "light":
        light = obj._light
        if light is None:
            return None
        v = getattr(light, spec.field)
        return float(v[spec.component]) if spec.component is not None else float(v)
    if g == "appearance":
        if obj._camera is not None or obj._light is not None:
            return None
        v = getattr(obj._appearance, spec.field)
        return float(v[spec.component]) if spec.component is not None else float(v)
    if g == "joint":
        return joints.rest_component(tid, spec) if joints is not None else None
    if g in ("morph", "constraint"):
        return (
            joints.rest_extra(tid, spec) if joints is not None else (0.0 if g == "morph" else 1.0)
        )
    return None


def _rest_values(prog: Program, system: AnimationSystem) -> dict[str, Any]:
    """Rest values of every slot (cached on the program per scene revision)."""
    scene = system.scene
    key = scene.revision
    cached = prog.rest_cache.get("rest")
    if cached is not None and cached[0] == key:
        return cached[1]
    rigs = system._rig_rest() if hasattr(system, "_rig_rest") else None
    values = np.zeros(len(prog.slots))
    valid = np.ones(len(prog.slots), bool)
    for k, (b, spec) in enumerate(prog.slots):
        v = _rest_scalar(scene, prog.targets[b], spec, rigs)
        if v is None:
            valid[k] = False
        else:
            values[k] = v
    rot = np.zeros((len(prog.rot_slots), 4))
    rot_valid = np.ones(len(prog.rot_slots), bool)
    rest_euler = np.zeros((len(prog.rot_slots), 3))
    for k, (b, joint) in enumerate(prog.rot_slots):
        kind, tid = prog.targets[b]
        if kind != P.OBJECT:
            rot_valid[k] = False
            continue
        if joint is None:
            rot[k] = scene._objects[tid]._transform._rotation
        else:
            found = rigs.rest_rotation(tid, joint) if rigs is not None else None
            if found is None:
                rot_valid[k] = False
                rot[k] = (0.0, 0.0, 0.0, 1.0)
                continue
            rot[k] = found
    if len(prog.rot_slots):
        rest_euler = R.to_euler(rot)
    out = {
        "scalar": values,
        "valid": valid,
        "rot": rot,
        "rot_valid": rot_valid,
        "euler": rest_euler,
    }
    # Euler references (additive layers) and per-cycle changes (accumulating strips).
    if prog.e_contrib.size:
        ref_t = np.array([prog.contributions[c].clip_range[0] for c in prog.e_contrib.tolist()])
        prog.e_ref[:] = _euler_quats(prog, rest_euler, np.arange(prog.e_contrib.size), ref_t)
        deltas = np.zeros((prog.e_contrib.size, 3))
        for i, c in enumerate(prog.e_contrib.tolist()):
            con = prog.contributions[c]
            if con.strip is not None and con.strip.accumulate:
                a, z = _trim(con.strip, con.clip_range)
                qa = _euler_quats(prog, rest_euler, np.array([i]), np.array([a]))
                qz = _euler_quats(prog, rest_euler, np.array([i]), np.array([z]))
                deltas[i] = R.log(R.multiply(R.align(qa, qz), R.conjugate(qa)))[0]
        prog.e_delta_log[:] = deltas
    prog.rest_cache["rest"] = (key, out)
    return out


def _euler_quats(
    prog: Program, rest_euler: np.ndarray, rows: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Quaternions of Euler queries ``rows`` at clip ``times`` (missing axes: rest)."""
    curves = prog.e_curves[rows]
    ang = rest_euler[prog.e_slot[rows]].copy()
    batch = prog.curves
    for axis in range(3):
        have = curves[:, axis] >= 0
        if np.any(have):
            ang[have, axis] = batch.evaluate(curves[have, axis], times[have])[0]
    return R.from_euler(ang)


# --------------------------------------------------------------------------------------- #
# A frame
# --------------------------------------------------------------------------------------- #


def _contribution_times(
    prog: Program, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip time, weight, cycle and layer weight of every contribution at scene time t."""
    n = len(prog.contributions)
    ctime = np.empty(n)
    weight = np.empty(n)
    cycle = np.zeros(n)
    lw = np.ones(n)
    tt = np.array([t])
    for k, con in enumerate(prog.contributions):
        if con.strip is None:
            ctime[k] = t
            weight[k] = 1.0
            continue
        c, w, cyc, _ = strip_times(con.strip, con.clip_range, tt)
        ctime[k], weight[k], cycle[k] = c[0], w[0], cyc[0]
        lw[k] = layer_weight(con.layer, t)
    return ctime, weight, cycle, lw


def evaluate(system: AnimationSystem, t: float, prog: Program | None = None) -> AnimationPose:
    """The animation of ``system`` at scene time ``t`` (see the module docstring)."""
    if prog is None:
        prog = system._program()
    rest = _rest_values(prog, system)
    diags: list[Diagnostic] = list(prog.diagnostics)
    ctime, cweight, ccycle, clw = _contribution_times(prog, t)
    ranks = np.array([c.rank for c in prog.contributions], np.int64)
    strip_ranks = np.array([c.strip_rank for c in prog.contributions], np.int64)
    modes = np.array([c.mode for c in prog.contributions], np.int64)

    # Scalars, layer by layer. A replace layer averages its strips by their weights and
    # blends the average over what is below with influence min(sum of weights, 1) x the
    # layer's weight x its mask: crossfades between strips never show the rest pose. An
    # additive layer adds every strip's w x (value - reference).
    values = rest["scalar"].copy()
    nslots = len(prog.slots)
    if prog.q_curve.size:
        qt = ctime[prog.q_contrib]
        x = prog.curves.evaluate(prog.q_curve, qt)[0] + ccycle[prog.q_contrib] * prog.q_delta
        w = cweight[prog.q_contrib]
        f = clw[prog.q_contrib] * prog.q_mask
        qr = ranks[prog.q_contrib]
        qm = modes[prog.q_contrib]
        for rank in np.unique(qr).tolist():
            sel = (qr == rank) & (w > 0.0) & (f > 0.0)
            if not np.any(sel):
                continue
            s = prog.q_slot[sel]
            if qm[sel][0] == ADDITIVE:
                values += np.bincount(s, w[sel] * f[sel] * (x[sel] - prog.q_ref[sel]), nslots)
                continue
            wsum = np.bincount(s, w[sel], nslots)
            xsum = np.bincount(s, w[sel] * x[sel], nslots)
            fs = np.zeros(nslots)
            fs[s] = f[sel]  # (one value per slot within a layer)
            live = wsum > 0.0
            avg = xsum[live] / wsum[live]
            influence = np.minimum(wsum[live], 1.0) * fs[live]
            # (full influence plays the curves exactly: v + 1 x (x - v) may round off x)
            values[live] = np.where(
                influence == 1.0, avg, values[live] + influence * (avg - values[live])
            )
    touched = np.zeros(len(prog.slots), bool)
    touched[prog.q_slot] = True

    # Rotations.
    rot = rest["rot"].copy()
    rot_touched = np.zeros(len(prog.rot_slots), bool)
    entries: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    if prog.r_track.size:
        qr_t = ctime[prog.r_contrib]
        x = prog.tracks.evaluate(prog.r_track, qr_t)
        cyc = ccycle[prog.r_contrib]
        if np.any(prog.r_delta_log != 0.0):
            x = R.multiply(R.exp(prog.r_delta_log * cyc[:, None]), x)
        entries.append(
            (prog.r_contrib, prog.r_slot, x, prog.r_mask, prog.r_ref, ranks[prog.r_contrib])
        )
    if prog.e_contrib.size:
        rows = np.arange(prog.e_contrib.size)
        x = _euler_quats(prog, rest["euler"], rows, ctime[prog.e_contrib])
        cyc = ccycle[prog.e_contrib]
        if np.any(prog.e_delta_log != 0.0):
            x = R.multiply(R.exp(prog.e_delta_log * cyc[:, None]), x)
        entries.append(
            (prog.e_contrib, prog.e_slot, x, prog.e_mask, prog.e_ref, ranks[prog.e_contrib])
        )
    if entries:
        contrib = np.concatenate([en[0] for en in entries])
        slots = np.concatenate([en[1] for en in entries])
        xs = np.concatenate([en[2] for en in entries])
        masks = np.concatenate([en[3] for en in entries])
        refs = np.concatenate([en[4] for en in entries])
        rk = np.concatenate([en[5] for en in entries])
        sk = strip_ranks[contrib]
        w = cweight[contrib]
        f = clw[contrib] * masks
        md = modes[contrib]
        nrot = len(prog.rot_slots)
        for rank in np.unique(rk).tolist():
            layer_sel = (rk == rank) & (w > 0.0) & (f > 0.0)
            if not np.any(layer_sel):
                continue
            additive = md[layer_sel][0] == ADDITIVE
            acc = np.zeros((nrot, 4))
            acc[:, 3] = 1.0
            wacc = np.zeros(nrot)
            fs = np.zeros(nrot)
            for srank in np.unique(sk[layer_sel]).tolist():
                sel = layer_sel & (sk == srank)
                s = slots[sel]
                if additive:  # compose each strip's delta, scaled, on top
                    rot[s] = R.normalize(R.additive(rot[s], xs[sel], refs[sel], w[sel] * f[sel]))
                    continue
                first = wacc[s] == 0.0
                mix = np.where(first, 1.0, w[sel] / np.where(first, 1.0, wacc[s] + w[sel]))
                acc[s] = np.where(first[:, None], xs[sel], R.blend(acc[s], xs[sel], mix))
                wacc[s] = wacc[s] + w[sel]
                fs[s] = f[sel]
            if not additive:
                live = np.flatnonzero(wacc > 0.0)
                influence = np.minimum(wacc[live], 1.0) * fs[live]
                full = (influence == 1.0)[:, None]
                rot[live] = R.normalize(
                    np.where(full, acc[live], R.blend(rot[live], acc[live], influence))
                )
        rot_touched[slots] = True

    # Assemble the pose.
    scene = system.scene
    visible: dict[str, bool] = {}
    fields: dict[tuple[str, str], dict[str, Any]] = {}
    materials: dict[str, dict[str, Any]] = {}
    joints: dict[str, dict[str, list]] = {}
    morphs: dict[str, dict[str, float]] = {}
    influences: dict[str, dict[str, float]] = {}
    clamped: dict[str, int] = {}
    skipped: dict[str, int] = {}
    # Object transforms, all at once (by far the most slots): see _transform_plan.
    plan = _transform_plan(prog)
    base = _transform_base(prog, plan, system)
    row_hit = np.zeros(len(plan["row_b"]), bool)
    position_rows = base["position"].copy()
    rotation_rows = base["rotation"].copy()
    scale_rows = base["scale"].copy()
    if plan["tf_k"].size:
        k_tf = plan["tf_k"]
        on = touched[k_tf]
        bad = on & ~rest["valid"][k_tf]
        v_tf = values[k_tf]
        bad |= on & ~np.isfinite(v_tf)
        _count(skipped, plan["tf_path"], bad)
        use = on & ~bad
        lo, hi = plan["tf_lo"], plan["tf_hi"]
        outside = use & ((v_tf < lo) | (v_tf > hi))
        _count(clamped, plan["tf_path"], outside)
        v_tf = np.where(outside, np.minimum(np.maximum(v_tf, lo), hi), v_tf)
        rows, comps, fields_ = plan["tf_row"][use], plan["tf_comp"][use], plan["tf_field"][use]
        vals = v_tf[use]
        at_pos = fields_ == 0
        position_rows[rows[at_pos], comps[at_pos]] = vals[at_pos]
        scale_rows[rows[~at_pos], comps[~at_pos]] = vals[~at_pos]
        row_hit[rows] = True
    if plan["or_k"].size:
        k_or = plan["or_k"]
        use = rot_touched[k_or] & rest["rot_valid"][k_or]
        rows = plan["or_row"][use]
        rotation_rows[rows] = rot[k_or[use]]
        row_hit[rows] = True
    for k in np.flatnonzero(touched & ~plan["is_tf"]).tolist():
        b, spec = prog.slots[k]
        kind, tid = prog.targets[b]
        if not rest["valid"][k]:
            skipped[spec.path] = skipped.get(spec.path, 0) + 1
            continue
        v = float(values[k])
        if not math.isfinite(v):
            skipped[spec.path] = skipped.get(spec.path, 0) + 1
            continue
        if v < spec.lo or v > spec.hi:
            v = min(max(v, spec.lo), spec.hi)
            clamped[spec.path] = clamped.get(spec.path, 0) + 1
        g = spec.group
        if kind == P.MATERIAL:
            entry = materials.setdefault(tid, {})
            material = scene.materials.find_material(tid)
            _set_component(entry, spec, v, lambda m=material, f=spec.field: list(m.params[f]))
            continue
        if g == "visible":
            visible[tid] = v > 0.5
        elif g in ("camera", "light", "appearance"):
            entry = fields.setdefault((tid, g), {})
            obj = scene._objects[tid]

            def current(obj=obj, g=g, spec=spec):
                holder = (
                    obj._camera
                    if g == "camera"
                    else obj._light
                    if g == "light"
                    else obj._appearance
                )
                return list(getattr(holder, spec.field))

            _set_component(entry, spec, v, current)
        elif g == "joint":
            jt = joints.setdefault(tid, {}).setdefault(spec.name, [None, None, None])
            slot = {"position": 0, "scale": 2}[spec.field]
            if jt[slot] is None:
                jt[slot] = list(system._rig_rest().rest_trs(tid, spec.name)[slot])
            jt[slot][spec.component] = v
        elif g == "morph":
            morphs.setdefault(tid, {})[spec.name] = v
        elif g == "constraint":
            key = spec.name if spec.field == "influence" else f"{spec.name}.{spec.field}"
            influences.setdefault(tid, {})[key] = v
    for k in np.flatnonzero(rot_touched & ~plan["is_or"]).tolist():
        b, joint = prog.rot_slots[k]
        kind, tid = prog.targets[b]
        if not rest["rot_valid"][k]:
            continue
        jt = joints.setdefault(tid, {}).setdefault(joint, [None, None, None])
        jt[1] = rot[k]
    # Objects with an animated transform component: unanimated components authored.
    ids = [tid for tid, hit in zip(base["ids"], row_hit.tolist(), strict=True) if hit]
    position = np.ascontiguousarray(position_rows[row_hit])
    rotation = rotation_rows[row_hit]
    if rotation.size:
        rotation = R.normalize(rotation)
    scale = np.ascontiguousarray(scale_rows[row_hit])
    for arr in (position, rotation, scale):
        arr.flags.writeable = False
    joint_out = {tid: {name: tuple(v) for name, v in jts.items()} for tid, jts in joints.items()}
    for path, count in sorted(clamped.items()):
        diags.append(
            Diagnostic(
                "VALUE_CLAMPED",
                INFO,
                f"{count} animated value(s) of {path} were outside the property's range and "
                "were clamped for this evaluation",
                path,
            )
        )
    for path, count in sorted(skipped.items()):
        diags.append(
            Diagnostic(
                "CHANNEL_NOT_APPLICABLE",
                WARNING,
                f"{count} channel(s) {path} target(s) without that property (e.g. a camera "
                "field on a mesh object); ignored",
                path,
            )
        )
    return AnimationPose(
        float(t),
        prog.revision,
        tuple(ids),
        position,
        rotation,
        scale,
        visible,
        fields,
        materials,
        joint_out,
        morphs,
        influences,
        tuple(diags),
    )


def _count(counts: dict[str, int], path_ids: tuple[list[str], np.ndarray], mask: np.ndarray):
    """Add to ``counts`` (path -> count) the slots of ``mask``."""
    if not np.any(mask):
        return
    names, ids = path_ids
    for i, n in zip(*np.unique(ids[mask], return_counts=True), strict=True):
        counts[names[int(i)]] = counts.get(names[int(i)], 0) + int(n)


def _transform_plan(prog: Program) -> dict[str, Any]:
    """Where the object transform slots go in the pose (static: cached on the program).

    Rows are the bound objects with a transform slot or an object rotation slot, in
    binding order (the pose's order); ``tf_*`` describe the scalar slots (position and
    scale components) and ``or_*`` the object rotation slots."""
    found = prog.rest_cache.get("transform_plan")
    if found is not None:
        return found
    rows: dict[int, int] = {}
    tf_k, tf_b, tf_field, tf_comp, tf_lo, tf_hi, tf_path = [], [], [], [], [], [], []
    path_names: list[str] = []
    path_index: dict[str, int] = {}
    for k, (b, spec) in enumerate(prog.slots):
        if spec.group != "transform" or prog.targets[b][0] != P.OBJECT:
            continue
        tf_k.append(k)
        tf_b.append(b)
        tf_field.append(0 if spec.field == "position" else 2)
        tf_comp.append(spec.component)
        tf_lo.append(spec.lo)
        tf_hi.append(spec.hi)
        if spec.path not in path_index:
            path_index[spec.path] = len(path_names)
            path_names.append(spec.path)
        tf_path.append(path_index[spec.path])
    or_k, or_b = [], []
    for k, (b, joint) in enumerate(prog.rot_slots):
        if joint is None and prog.targets[b][0] == P.OBJECT:
            or_k.append(k)
            or_b.append(b)
    for b in sorted(set(tf_b) | set(or_b)):
        rows[b] = len(rows)
    is_tf = np.zeros(len(prog.slots), bool)
    is_tf[tf_k] = True
    is_or = np.zeros(len(prog.rot_slots), bool)
    is_or[or_k] = True
    plan = {
        "row_b": list(rows),
        "tf_k": np.array(tf_k, np.int64),
        "tf_row": np.array([rows[b] for b in tf_b], np.int64),
        "tf_field": np.array(tf_field, np.int64),
        "tf_comp": np.array(tf_comp, np.int64),
        "tf_lo": np.array(tf_lo, np.float64),
        "tf_hi": np.array(tf_hi, np.float64),
        "tf_path": (path_names, np.array(tf_path, np.int64)),
        "or_k": np.array(or_k, np.int64),
        "or_row": np.array([rows[b] for b in or_b], np.int64),
        "is_tf": is_tf,
        "is_or": is_or,
    }
    prog.rest_cache["transform_plan"] = plan
    return plan


def _transform_base(prog: Program, plan: dict[str, Any], system: AnimationSystem) -> dict:
    """The authored transforms of the plan's rows (cached per scene revision)."""
    key = system.scene.revision
    cached = prog.rest_cache.get("transform_base")
    if cached is not None and cached[0] == key:
        return cached[1]
    objects = system.scene._objects
    ids, pos, rot, scl = [], [], [], []
    for b in plan["row_b"]:
        tid = prog.targets[b][1]
        tf = objects[tid]._transform
        ids.append(tid)
        pos.append(tf._position)
        rot.append(tf._rotation)
        scl.append(tf._scale)
    base = {
        "ids": ids,
        "position": np.array(pos, np.float64).reshape(-1, 3),
        "rotation": np.array(rot, np.float64).reshape(-1, 4),
        "scale": np.array(scl, np.float64).reshape(-1, 3),
    }
    prog.rest_cache["transform_base"] = (key, base)
    return base


def _set_component(entry: dict[str, Any], spec: P.PathSpec, value: float, current) -> None:
    if spec.component is None:
        entry[spec.field] = value
        return
    vec = entry.get(spec.field)
    if vec is None:
        vec = entry[spec.field] = list(current())
    vec[spec.component] = value
