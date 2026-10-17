"""Keyframe reduction, resampling, fitting and baking.

* :func:`simplify` reduces a curve's keys to a bounded error: starting from its first and
  last keys, it repeatedly adds a key where the reduced curve is farthest from the
  original (measured on dense samples: every original key plus ``density`` points per
  original segment), until the largest deviation is at most ``tolerance``. Each kept key
  takes the original's value and exact slope (Hermite handles), so the error falls as
  the fourth power of the spacing on smooth motion. Kept steps (constant keys) stay steps.
* :func:`resample` puts keys on a frame grid (value and slope of the original).
* :func:`fit_samples` turns dense samples (motion capture, simulation output) into a curve
  within ``tolerance`` of every sample (slopes from the samples' centred differences).
* :func:`bake` samples what the scene *shows* -- after layers, rigs and constraints -- for
  objects (local transforms) or joints (local to their parent joint) into ordinary keys
  of a new clip, optionally simplified: the way to export NLA, IK and constraints to
  formats that only carry keys.

Errors reported are measured on the samples used (``basis: "sampled"``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from .curves import CONSTANT, FCurve
from .time import FrameRate, frame_range

if TYPE_CHECKING:
    from .system import AnimationSystem

MAX_FIT_SAMPLES = 2_000_000


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _hermite_curve(
    times: np.ndarray, values: np.ndarray, slopes: np.ndarray, steps: np.ndarray | None = None
) -> FCurve:
    """Keys with free, unweighted handles of the given slopes (``steps``: constant keys)."""
    n = times.shape[0]
    interp = np.full(n, 2, np.uint8)
    if steps is not None:
        interp[steps] = CONSTANT
    keys = []
    for i in range(n):
        keys.append(
            {
                "time": float(times[i]),
                "value": float(values[i]),
                "interpolation": int(interp[i]),
                "left": {"type": "free", "slope": float(slopes[i])},
                "right": {"type": "free", "slope": float(slopes[i])},
            }
        )
    return FCurve.from_keys(keys)


def _reduce(
    t: np.ndarray, v: np.ndarray, slope_of, tolerance: float, max_keys: int, fixed: np.ndarray
) -> tuple[np.ndarray, float]:
    """Indices (into the samples) of the kept keys and the largest deviation."""
    keep = sorted({0, t.shape[0] - 1, *fixed.tolist()})
    worst = math.inf
    while True:
        idx = np.array(keep)
        curve = _hermite_curve(t[idx], v[idx], slope_of(idx))
        err = np.abs(curve.evaluate(t) - v)
        worst = float(err.max())
        if worst <= tolerance or len(keep) >= max_keys:
            return idx, worst
        k = int(err.argmax())
        if k in keep:  # (cannot improve: the worst sample is a key)
            return idx, worst
        keep = sorted({*keep, k})


def simplify(
    curve: FCurve, tolerance: float, *, density: int = 16, max_keys: int | None = None
) -> tuple[FCurve, dict[str, Any]]:
    """A curve within ``tolerance`` of ``curve`` with few keys, and a report
    ``{keys_before, keys_after, max_error, basis}``."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not tolerance > 0:
        raise _err("INVALID_ARGUMENT", "tolerance is a positive number", "tolerance")
    n = len(curve)
    if n <= 2:
        return curve, {"keys_before": n, "keys_after": n, "max_error": 0.0, "basis": "sampled"}
    if curve.modifiers:
        raise _err(
            "INVALID_ARGUMENT", "simplify a curve without modifiers (bake them first)", "curve"
        )
    times = curve.times
    pieces = [times[:1]]
    for k in range(n - 1):
        a, b = times[k], times[k + 1]
        if b > a:
            pieces.append(np.linspace(a, b, density + 1)[1:])
    t = np.unique(np.concatenate(pieces))
    if t.size > MAX_FIT_SAMPLES:
        raise _err("LIMIT_EXCEEDED", "too many samples: lower density", "density")
    v, d = curve.evaluate(t, order=1)
    d = np.where(np.isfinite(d), d, 0.0)
    steps = np.flatnonzero(curve.interpolation == CONSTANT)
    jumps = np.flatnonzero(np.diff(times) == 0)
    fixed_times = np.concatenate(
        [
            times[steps],
            times[np.minimum(steps + 1, n - 1)],
            times[jumps],
            times[np.minimum(jumps + 1, n - 1)],
        ]
    )
    fixed = np.flatnonzero(np.isin(t, fixed_times))
    if steps.size or jumps.size:
        # Steps and jumps are kept exactly: the keys of the original around them.
        out = _with_steps(curve, tolerance, density, max_keys)
        return out
    idx, worst = _reduce(t, v, lambda i: d[i], float(tolerance), max_keys or t.size, fixed)
    result = _hermite_curve(t[idx], v[idx], d[idx])
    result = result.replace(extrapolation=list(curve.extrapolation))
    return result, {
        "keys_before": n,
        "keys_after": int(idx.size),
        "max_error": worst,
        "basis": "sampled",
    }


def _with_steps(
    curve: FCurve, tolerance: float, density: int, max_keys: int | None
) -> tuple[FCurve, dict[str, Any]]:
    """Simplify each run of smooth segments between steps or jumps separately."""
    times = curve.times
    n = len(curve)
    breaks = sorted(
        {
            0,
            n - 1,
            *np.flatnonzero(curve.interpolation == CONSTANT).tolist(),
            *(np.flatnonzero(curve.interpolation == CONSTANT) + 1).tolist(),
            *np.flatnonzero(np.diff(times) == 0).tolist(),
            *(np.flatnonzero(np.diff(times) == 0) + 1).tolist(),
        }
    )
    breaks = [b for b in breaks if 0 <= b < n]
    keys: list[dict[str, Any]] = []
    worst = 0.0
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        if curve.interpolation[a] == CONSTANT or times[b] == times[a]:
            keys.append(curve.key(a))
            continue
        sub = curve.replace(
            {name: arr[a : b + 1] for name, arr in curve.arrays().items()}, extrapolation="constant"
        )
        reduced, rep = simplify(sub, tolerance, density=density, max_keys=max_keys)
        worst = max(worst, rep["max_error"])
        keys.extend(reduced.keys()[:-1])
    keys.append(curve.key(n - 1))
    out = FCurve.from_keys(keys).replace(extrapolation=list(curve.extrapolation))
    return out, {"keys_before": n, "keys_after": len(out), "max_error": worst, "basis": "sampled"}


def resample(
    curve: FCurve, fps: Any, *, start: float | None = None, end: float | None = None
) -> FCurve:
    """Keys on every frame of ``fps`` in ``start..end`` (default: the key range), each with
    the original's value and slope."""
    rate = FrameRate(fps, "fps")
    a, b = curve.range
    a = a if start is None else float(start)
    b = b if end is None else float(end)
    frames = frame_range(rate, a, b)
    if frames.size < 1:
        raise _err("INVALID_RANGE", "no frame in the range", "range")
    t = rate.frame_times(frames)
    v, d = curve.evaluate(t, order=1)
    d = np.where(np.isfinite(d), d, 0.0)
    return _hermite_curve(t, v, d).replace(extrapolation=list(curve.extrapolation))


def fit_samples(
    times: Any, values: Any, tolerance: float, *, max_keys: int | None = None
) -> tuple[FCurve, dict[str, Any]]:
    """A curve through dense samples within ``tolerance`` of every sample."""
    t = np.asarray(times, np.float64).reshape(-1)
    v = np.asarray(values, np.float64).reshape(-1)
    if t.shape != v.shape or t.size < 2 or t.size > MAX_FIT_SAMPLES:
        raise _err("INVALID_ARGUMENT", f"2..{MAX_FIT_SAMPLES} samples of equal lengths", "samples")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))) or np.any(np.diff(t) <= 0):
        raise _err("INVALID_ARGUMENT", "sample times must increase and values be finite", "samples")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not tolerance > 0:
        raise _err("INVALID_ARGUMENT", "tolerance is a positive number", "tolerance")
    d = np.gradient(v, t)
    idx, worst = _reduce(
        t, v, lambda i: d[i], float(tolerance), max_keys or t.size, np.zeros(0, np.int64)
    )
    return _hermite_curve(t[idx], v[idx], d[idx]), {
        "samples": int(t.size),
        "keys": int(idx.size),
        "max_error": worst,
        "basis": "sampled",
    }


def bake(
    system: AnimationSystem,
    targets: list[Any],
    *,
    start: float | None = None,
    end: float | None = None,
    fps: Any = None,
    joints: bool = True,
    tolerance: float | None = None,
    name: str = "Baked",
) -> dict[str, Any]:
    """Sample the evaluated local transforms of ``targets`` (objects; with ``joints``,
    also every joint of their rigs) on a frame grid into a new clip per target, keyed
    as position curves, quaternion tracks (slerp, shortest path) and scale curves;
    ``tolerance`` simplifies the curves. Returns ``{clips: {target name: clip name},
    frames, max_error}``. The clips are created (one undo step); actions are unchanged."""
    from ..rendering.scene_data import world_matrices
    from . import rotation as R
    from .constraints import decompose
    from .tracks import RotationTrack

    rate = system.fps if fps is None else FrameRate(fps, "fps")
    a = system.start if start is None else float(start)
    b = system.end if end is None else float(end)
    frames = frame_range(rate, a, b)
    if frames.size < 2:
        raise _err("INVALID_RANGE", "baking needs two frames or more", "range")
    times = rate.frame_times(frames)
    objs = [system._object(t, f"targets[{i}]") for i, t in enumerate(targets)]
    local: dict[str, list[np.ndarray]] = {o._id: [] for o in objs}
    joint_local: dict[str, dict[str, list[np.ndarray]]] = {}
    for t in times.tolist():
        overlay = system.overlay(t)
        worlds = world_matrices(system.scene, overlay)
        rigging = system.rigging(t) if joints else None
        for o in objs:
            parent = o._parent
            p = np.eye(4) if parent is None else worlds[parent._id]
            local[o._id].append(np.linalg.solve(p, worlds[o._id]))
            if rigging is not None and o._id in rigging.joints:
                rig, model = rigging.joints[o._id]
                per = joint_local.setdefault(o._id, {})
                for k, j in enumerate(rig.joints):
                    parent_m = np.eye(4) if j.parent < 0 else model[j.parent]
                    per.setdefault(j.name, []).append(np.linalg.solve(parent_m, model[k]))
    report: dict[str, Any] = {"clips": {}, "frames": int(frames.size), "max_error": 0.0}

    def channels_of(mats: list[np.ndarray], prefix: str) -> dict[str, Any]:
        trs = [decompose(m) for m in mats]
        pos = np.array([x[0] for x in trs])
        rot = np.array([x[1] for x in trs])
        scl = np.array([x[2] for x in trs])
        for i in range(1, rot.shape[0]):
            rot[i] = R.align(rot[i - 1], rot[i])
        out: dict[str, Any] = {}
        for axis, i in (("x", 0), ("y", 1), ("z", 2)):
            for field, arr in (("position", pos), ("scale", scl)):
                if field == "scale" and np.allclose(arr[:, i], 1.0, atol=1e-12):
                    continue
                vals = arr[:, i]
                curve = FCurve.from_arrays(times, vals, handles="auto_clamped")
                if tolerance is not None:
                    curve, rep = fit_samples(times, vals, tolerance)
                    report["max_error"] = max(report["max_error"], rep["max_error"])
                out[f"{prefix}{field}.{axis}"] = curve
        out[f"{prefix}rotation"] = RotationTrack.from_arrays(times, rot)
        return out

    with system._op(f"Bake {len(objs)} object(s)"):
        for o in objs:
            chans = channels_of(local[o._id], "transform.")
            for jname, mats in joint_local.get(o._id, {}).items():
                chans.update(channels_of(mats, f'joints["{jname}"].'))
            clip_name = system._unique_clip_name(f"{o.name} {name}")
            handle = system.add_clip(clip_name)
            for path, ch in sorted(chans.items()):
                handle.set_channel(path, ch)
            report["clips"][o.name] = clip_name
    return report
