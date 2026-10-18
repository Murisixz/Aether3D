"""Curve and motion diagnostics: what a motion does, measured.

**Curve reports** (:func:`curve_report`) describe one F-curve exactly where the
mathematics allows: key counts, interpolation and handle types, corrected handles;
**continuity** at every key from the two segments' exact one-sided limits (value C0,
velocity C1, acceleration C2 -- a step from a ``constant`` key or a jump pair is
reported as intended); **extrema** found from the sign changes of the analytic velocity
and refined by bisection; per-segment **overshoot** beyond the range of the segment's two
keys; the largest velocity, acceleration and jerk.

**Trajectory reports** (:func:`trajectory_report`) follow an object or a joint in world
space through the evaluated scene (after layers, rigs and constraints), sampled at a
rate: path length, speed, acceleration and jerk (central differences of evaluated
positions: ``sampled``), angular speed, and a smoothness measure (the log dimensionless
jerk of Balasubramanian et al., 2012: closer to 0 is smoother).

:func:`diagnose` runs the checks over a scene's animation and returns **findings**
``{code, severity, location, time, magnitude, message, basis}`` with ``basis``
``"exact"`` (from the curves' closed forms) or ``"sampled"``. A finding is a measurement,
not an aesthetic judgement: a deliberate snap is flagged as a velocity discontinuity and
it is for the animator (or the agent) to decide.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .curves import _SEG_CONSTANT, _SEG_JUMP, HANDLE_TYPES, INTERPOLATIONS, FCurve, _segment_values

if TYPE_CHECKING:
    from .system import AnimationSystem

MAX_SAMPLES = 200_000


def _limits(curve: FCurve, k: int) -> dict[str, list[float]]:
    """Value, velocity and acceleration at key ``k`` from the left and from the right."""
    c = curve.compiled()
    t = np.array([curve.times[k]])
    n = len(curve)
    out: dict[str, list[float]] = {}
    if k > 0 and c.kind[k - 1] != _SEG_JUMP:
        left = _segment_values(c, np.array([k - 1]), t, 2)
        out["left"] = [float(left[0][0]), float(left[1][0]), float(left[2][0])]
    elif k > 0:  # a jump pair: the left limit is the previous segment's end
        if k > 1:
            left = _segment_values(c, np.array([k - 2]), t, 2)
            out["left"] = [float(left[0][0]), float(left[1][0]), float(left[2][0])]
    if k < n - 1:
        right = _segment_values(c, np.array([k]), t, 2)
        if c.kind[k] == _SEG_JUMP:
            right = (
                _segment_values(c, np.array([k + 1]), t, 2)
                if k + 1 < n - 1
                else [np.array([curve.values[k + 1]]), np.zeros(1), np.zeros(1)]
            )
        out["right"] = [float(right[0][0]), float(right[1][0]), float(right[2][0])]
    return out


def continuity(curve: FCurve, *, tolerance: float = 1e-9) -> list[dict[str, Any]]:
    """Discontinuities at the keys: ``{key, time, order (0 value, 1 velocity, 2
    acceleration), jump, intended}`` (``intended``: a constant key's step or a jump pair)."""
    c = curve.compiled()
    out = []
    n = len(curve)
    scale = max(1.0, float(np.max(np.abs(curve.values))))
    for k in range(1, n - 1):
        lim = _limits(curve, k)
        if "left" not in lim or "right" not in lim:
            continue
        left, right = lim["left"], lim["right"]
        intended_step = (
            c.kind[k - 1] == _SEG_CONSTANT
            or (k + 1 < n and curve.times[k + 1] == curve.times[k])
            or curve.times[k - 1] == curve.times[k]
        )
        for order in range(3):
            jump = right[order] - left[order]
            if not (math.isfinite(jump)) or abs(jump) > tolerance * scale * (1.0 + 10.0 * order):
                out.append(
                    {
                        "key": k,
                        "time": float(curve.times[k]),
                        "order": order,
                        "jump": jump if math.isfinite(jump) else float("inf"),
                        "left": left[order],
                        "right": right[order],
                        "intended": bool(intended_step and order <= 1),
                    }
                )
                break  # report the lowest order that breaks
    return out


def extrema(curve: FCurve, samples_per_segment: int = 64) -> list[dict[str, Any]]:
    """Local extrema inside segments (velocity crossing zero), refined by bisection on
    the analytic velocity: ``{time, value, kind: "max" | "min", segment}``."""
    n = len(curve)
    if n < 2:
        return []
    out = []
    for k in range(n - 1):
        t0, t1 = float(curve.times[k]), float(curve.times[k + 1])
        if t1 <= t0:
            continue
        ts = np.linspace(t0, t1, samples_per_segment + 1)[1:-1]
        if ts.size == 0:
            continue
        v, d = curve.evaluate(ts, order=1)
        sign = np.sign(d)
        for i in np.flatnonzero(sign[:-1] * sign[1:] < 0).tolist():
            lo, hi = float(ts[i]), float(ts[i + 1])
            dlo = rising = float(d[i])  # (the sign before the crossing names the extremum)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                dm = curve.evaluate(mid, order=1)[1]
                if (dm < 0) == (dlo < 0):
                    lo, dlo = mid, dm
                else:
                    hi = mid
            tm = 0.5 * (lo + hi)
            vm = float(curve.evaluate(tm))
            kind = "max" if rising > 0 else "min"
            out.append({"time": tm, "value": vm, "kind": kind, "segment": k})
    return out


def overshoot(curve: FCurve, *, relative: float = 1e-3) -> list[dict[str, Any]]:
    """Segments whose values leave the range of their two keys by more than ``relative``
    of the keys' difference (or 1e-9 when the keys are equal): ``{segment, time, value,
    beyond}``."""
    out = []
    for e in extrema(curve):
        k = e["segment"]
        a, b = float(curve.values[k]), float(curve.values[k + 1])
        lo, hi = min(a, b), max(a, b)
        beyond = max(e["value"] - hi, lo - e["value"], 0.0)
        if beyond > max(relative * abs(b - a), 1e-9):
            out.append({"segment": k, "time": e["time"], "value": e["value"], "beyond": beyond})
    return out


def curve_report(curve: FCurve, *, samples: int = 0, rate: float = 240.0) -> dict[str, Any]:
    """Everything measurable about one curve (see the module docstring). ``samples``:
    also return that many evenly spaced samples of value, velocity and acceleration."""
    n = len(curve)
    t0, t1 = curve.range
    comp = curve.compiled()
    report: dict[str, Any] = {
        "keys": n,
        "range": [t0, t1],
        "interpolation": _counts([INTERPOLATIONS[i] for i in curve.interpolation.tolist()]),
        "handles": _counts(
            [HANDLE_TYPES[i] for i in curve.array("left_type").tolist()]
            + [HANDLE_TYPES[i] for i in curve.array("right_type").tolist()]
        ),
        "weighted": int(np.count_nonzero(curve.array("weighted"))),
        "corrected_handles": np.flatnonzero(comp.corrected).tolist(),
        "extrapolation": list(curve.extrapolation),
        "modifiers": [m.type for m in curve.modifiers],
    }
    if n >= 2 and t1 > t0:
        count = int(min(MAX_SAMPLES, max(64, (t1 - t0) * rate)))
        ts = np.linspace(t0, t1, count)
        v, d, a, j = curve.evaluate(ts, order=3)
        finite_d = np.where(np.isfinite(d), np.abs(d), 0.0)
        finite_a = np.where(np.isfinite(a), np.abs(a), 0.0)
        finite_j = np.where(np.isfinite(j), np.abs(j), 0.0)
        report["value"] = {"min": float(v.min()), "max": float(v.max())}
        report["velocity"] = {"max": float(finite_d.max()), "at": float(ts[int(finite_d.argmax())])}
        report["acceleration"] = {
            "max": float(finite_a.max()),
            "at": float(ts[int(finite_a.argmax())]),
        }
        report["jerk"] = {"max": float(finite_j.max()), "at": float(ts[int(finite_j.argmax())])}
        report["vertical_tangents"] = int(np.count_nonzero(~np.isfinite(d)))
        report["discontinuities"] = continuity(curve)
        report["extrema"] = extrema(curve)
        report["overshoot"] = overshoot(curve)
        if samples:
            k = int(min(max(samples, 2), 10_000))
            st = np.linspace(t0, t1, k)
            sv, sd, sa = curve.evaluate(st, order=2)
            report["samples"] = {
                "time": st.tolist(),
                "value": sv.tolist(),
                "velocity": sd.tolist(),
                "acceleration": sa.tolist(),
            }
    return report


def _counts(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------------------- #


def sample_world(
    system: AnimationSystem, target: Any, times: np.ndarray, joint: str | None = None
) -> np.ndarray:
    """World matrices ``(n, 4, 4)`` of an object (or one of its rig's joints) at ``times``."""
    from ..rendering.scene_data import world_matrices

    if joint is not None:
        return sample_joints(system, target, times, [joint])[joint]
    obj = system._object(target, "target")
    out = np.empty((len(times), 4, 4))
    for i, t in enumerate(times.tolist()):
        out[i] = world_matrices(system.scene, system.overlay(t))[obj._id]
    return out


def sample_joints(
    system: AnimationSystem, target: Any, times: np.ndarray, joints: list[str]
) -> dict[str, np.ndarray]:
    """World matrices ``(n, 4, 4)`` of several joints of an object's rig at ``times``,
    from one evaluation of the scene per time."""
    from ..rendering.scene_data import world_matrices

    obj = system._object(target, "target")
    out = {j: np.empty((len(times), 4, 4)) for j in joints}
    for i, t in enumerate(times.tolist()):
        w = world_matrices(system.scene, system.overlay(t))[obj._id]
        found = system.rigging(t).joints.get(obj._id)
        if found is None:
            raise _missing_joint(obj.name, joints[0])
        rig, model = found
        for joint in joints:
            k = rig.index(joint)
            if k is None:
                raise _missing_joint(obj.name, joint)
            out[joint][i] = w @ model[k]
    return out


def _missing_joint(name: str, joint: str) -> Exception:
    from ..errors import AnimationError

    return AnimationError("JOINT_MISSING", f"{name!r} has no rig joint {joint!r}", "joint")


def trajectory_report(
    system: AnimationSystem,
    target: Any,
    *,
    start: float | None = None,
    end: float | None = None,
    rate: float = 60.0,
    joint: str | None = None,
) -> dict[str, Any]:
    """Motion of an object or joint in world space (see the module docstring)."""
    from ..errors import AnimationError

    s = system.start if start is None else float(start)
    e = system.end if end is None else float(end)
    if not (math.isfinite(s) and math.isfinite(e) and e > s):
        raise AnimationError("INVALID_RANGE", "a trajectory needs start < end", "range")
    if not 1.0 <= rate <= 10_000.0:
        raise AnimationError("INVALID_ARGUMENT", "rate is 1..10000 samples per second", "rate")
    count = int(min(20_000, max(8, round((e - s) * rate) + 1)))
    ts = np.linspace(s, e, count)
    worlds = sample_world(system, target, ts, joint)
    pos = worlds[:, :3, 3]
    h = ts[1] - ts[0]
    vel = np.gradient(pos, h, axis=0)
    acc = np.gradient(vel, h, axis=0)
    jerk = np.gradient(acc, h, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(acc, axis=1)
    jk = np.linalg.norm(jerk, axis=1)
    length = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    duration = e - s
    peak = float(speed.max())
    ldlj = None
    if peak > 1e-12:
        dimless = (duration**3 / peak**2) * float(np.trapezoid(jk**2, ts))
        ldlj = -math.log(max(dimless, 1e-300))
    from .rotation import align, conjugate, from_matrices, log, multiply

    rots = from_matrices(
        worlds[:, :3, :3] / np.linalg.norm(worlds[:, :3, :3], axis=1, keepdims=True)
    )
    for i in range(1, len(rots)):
        rots[i] = align(rots[i - 1], rots[i])
    ang = np.linalg.norm(2.0 * log(multiply(rots[1:], conjugate(rots[:-1]))), axis=1) / h
    return {
        "range": [s, e],
        "samples": count,
        "basis": "sampled",
        "path_length": length,
        "displacement": float(np.linalg.norm(pos[-1] - pos[0])),
        "speed": {"max": peak, "mean": float(speed.mean()), "at": float(ts[int(speed.argmax())])},
        "acceleration": {"max": float(accel.max()), "at": float(ts[int(accel.argmax())])},
        "jerk": {"max": float(jk.max()), "rms": float(np.sqrt(np.mean(jk**2)))},
        "angular_speed": {"max": float(ang.max()) if ang.size else 0.0},
        "smoothness_ldlj": ldlj,
        "start": pos[0].tolist(),
        "end": pos[-1].tolist(),
    }


# --------------------------------------------------------------------------------------- #
# Diagnose
# --------------------------------------------------------------------------------------- #


def diagnose(
    system: AnimationSystem,
    *,
    overshoot_relative: float = 0.05,
    include_intended: bool = False,
    max_findings: int = 500,
) -> dict[str, Any]:
    """Findings over every channel of every clip and the evaluation's own reports."""
    findings: list[dict[str, Any]] = []

    def add(code, severity, location, message, *, time=None, magnitude=None, basis="exact"):
        if len(findings) < max_findings:
            findings.append(
                {
                    "code": code,
                    "severity": severity,
                    "location": location,
                    "time": time,
                    "magnitude": magnitude,
                    "message": message,
                    "basis": basis,
                }
            )

    for clip in system.clips():
        for path, ch in clip.data.channels:
            loc = f"clips[{clip.name}].channels[{path}]"
            if not isinstance(ch, FCurve):
                q = ch.quats
                norms = np.linalg.norm(q, axis=1)
                if np.any(np.abs(norms - 1.0) > 1e-9):
                    add(
                        "NON_UNIT_QUATERNION",
                        "warning",
                        loc,
                        "rotation keys that are not unit quaternions",
                    )
                continue
            for d in continuity(ch):
                if d["intended"] and not include_intended:
                    continue
                code = (
                    "VALUE_DISCONTINUITY",
                    "VELOCITY_DISCONTINUITY",
                    "ACCELERATION_DISCONTINUITY",
                )[d["order"]]
                sev = "warning" if d["order"] < 2 else "info"
                add(
                    code,
                    sev,
                    f"{loc}.keys[{d['key']}]",
                    f"{('value', 'velocity', 'acceleration')[d['order']]} jumps by "
                    f"{d['jump']:.6g} at {d['time']:.6g} s",
                    time=d["time"],
                    magnitude=abs(d["jump"]),
                )
            for o in overshoot(ch, relative=overshoot_relative):
                add(
                    "OVERSHOOT",
                    "info",
                    f"{loc}.segments[{o['segment']}]",
                    f"the curve passes its keys' range by {o['beyond']:.6g} at {o['time']:.6g} s",
                    time=o["time"],
                    magnitude=o["beyond"],
                )
            corrected = np.flatnonzero(ch.compiled().corrected).tolist()
            for k in corrected:
                add(
                    "HANDLES_CORRECTED",
                    "info",
                    f"{loc}.segments[{k}]",
                    "handles reach past their segment; they are scaled back at evaluation",
                )
            t0, t1 = ch.range
            if t0 < system.start - 1e-12 or t1 > system.end + 1e-12:
                add(
                    "KEYS_OUTSIDE_TIMELINE",
                    "info",
                    loc,
                    f"keys span {t0:g}..{t1:g} s; "
                    f"the timeline is {system.start:g}..{system.end:g} s",
                )
    pose = system.evaluate(system.start)
    for d in pose.diagnostics:
        add(d.code, d.severity, d.location or "animation", d.message, basis="exact")
    if system._owned:
        times = np.linspace(system.start, system.end, 9)
        worst: dict[str, tuple[float, float]] = {}
        for t in times.tolist():
            rigging = system.rigging(t)
            for name, r in rigging.residuals.items():
                if r > worst.get(name, (0.0, 0.0))[0]:
                    worst[name] = (r, t)
            for d in rigging.diagnostics:
                add(
                    d.code,
                    d.severity,
                    d.location or "animation",
                    d.message,
                    time=t,
                    basis="sampled",
                )
        for name, (r, t) in worst.items():
            if r > 1e-6:
                add(
                    "IK_UNREACHED",
                    "info",
                    f"constraints[{name}]",
                    f"the IK target stays {r:.6g} away at {t:g} s (out of reach)",
                    time=t,
                    magnitude=r,
                    basis="sampled",
                )
    counts = _counts([f["severity"] for f in findings])
    return {"findings": findings, "counts": counts, "truncated": len(findings) >= max_findings}
