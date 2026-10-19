"""Motion quality in world space: what the evaluated scene does, not what one curve says.

:func:`motion_report` samples the evaluated animation (layers, strips, rigs, constraints,
IK, paths, shots: what renders show) at ``rate`` samples per second and measures, for
objects, rig joints and cameras:

* **jumps** -- velocity and acceleration discontinuities, measured as the difference of
  one-sided least-squares fits (lines over ``window`` seconds on each side of every
  sample), so smooth variation cancels and only breaks remain: ``POSITION_JUMP`` (a
  teleport: travel within one sample far beyond the speed around it), ``VELOCITY_JUMP``
  (velocity changes by more than ``velocity_jump`` of the peak speed), ``ACCELERATION_STEP``
  (acceleration steps by more than ``acceleration_step``: a key where the curve is only C1
  -- often harmless, and what foot plants physically are), ``ROTATION_SNAP`` /
  ``ANGULAR_VELOCITY_JUMP`` for orientations;
* **jitter** -- motion above ``jitter_hz`` (the residual of a Gaussian low-pass of the
  trajectory), as RMS millimetres and degrees: ``JITTER``, ``CAMERA_JITTER``;
* **pauses** -- the motion stopping for ``pause_min``..``pause_max`` seconds between moves
  (``PAUSE``, info: holds are often intended);
* **pole flips** -- the bend plane of an IK chain turning over within one sample
  (``POLE_FLIP``);
* **loop seams** -- with ``loop``, the velocity and acceleration at the end of the range
  against its start (``LOOP_SEAM``);
* **ground penetration** -- with ``ground`` (a height), the lowest vertex of each mesh
  object (deformed skins included) below it (``GROUND_PENETRATION``);
* **cameras** -- the above for the camera, and with a ``subject`` (an object or a point):
  aim error, the subject's place in the frame and how fast it swims there
  (``FRAMING_DRIFT``), ``SUBJECT_OUT_OF_FRAME``, ``SUBJECT_OBSTRUCTED`` (line of sight at
  up to 32 sampled times). The shot sequence's camera (``{"camera": "shots"}``) jumps at its
  cuts by design: those are ``CUT`` (info), not jumps.

Every number is sampled (``basis: "sampled"``) and every threshold used is in the report:
the report measures; whether a snap or a pause is wanted is the animator's (or agent's)
decision. Nothing in the scene changes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import AnimationError

DEFAULTS: dict[str, float | None] = {
    "velocity_jump": 0.25,
    "velocity_jump_min": 0.05,
    "acceleration_step": 4.0,
    "position_jump": 0.02,
    "jitter_hz": 8.0,
    "jitter_mm": 0.5,
    "jitter_deg": 0.05,
    "snap_deg": 15.0,
    "angular_jump": 0.5,
    "angular_jump_min": 90.0,
    "pause_speed": 0.02,
    "pause_min": 0.06,
    "pause_max": 0.5,
    "window": 0.05,
    "ground": None,
    "ground_tolerance": 0.002,
    "frame_margin": 0.0,
    "drift": 0.25,
}
"""Default thresholds (units: m, s, degrees, fractions of a target's own peak)."""
MAX_SAMPLES = 20_000
MAX_TARGETS = 512
MAX_SIGHT_SAMPLES = 32


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


# --------------------------------------------------------------------------------------- #
# Signal tools
# --------------------------------------------------------------------------------------- #


def _one_sided_fits(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """For every sample ``i`` of ``x`` ((n, d)), the value at ``i`` of a least-squares line
    fitted to the ``w`` samples before it (excluding ``i``) and to the ``w`` after it, by
    running sums (NaN where a side has fewer than 2 samples)."""
    n = x.shape[0]
    idx = np.arange(n, dtype=np.float64)
    c0 = np.concatenate([np.zeros((1,)), np.cumsum(np.ones(n))])
    c1 = np.concatenate([np.zeros((1,)), np.cumsum(idx)])
    c2 = np.concatenate([np.zeros((1,)), np.cumsum(idx * idx)])
    cx = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    cix = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(idx[:, None] * x, axis=0)])

    def fit(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        # sums over samples lo..hi-1; the line's value at i
        s0 = c0[hi] - c0[lo]
        s1 = c1[hi] - c1[lo]
        s2 = c2[hi] - c2[lo]
        sx = cx[hi] - cx[lo]
        six = cix[hi] - cix[lo]
        det = s0 * s2 - s1 * s1
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (s0[:, None] * six - s1[:, None] * sx) / det[:, None]
            icpt = (sx - slope * s1[:, None]) / s0[:, None]
            out = icpt + slope * idx[:, None]
        out[s0 < 2] = np.nan
        return out

    lo_b = np.maximum(idx - w, 0).astype(int)
    hi_b = idx.astype(int)
    lo_f = (idx + 1).astype(int)
    hi_f = np.minimum(idx + 1 + w, n).astype(int)
    return fit(lo_b, hi_b), fit(lo_f, hi_f)


def _peaks(score: np.ndarray, threshold: float, spread: int) -> list[int]:
    """Indices of local maxima of ``score`` above ``threshold``, at least ``spread`` apart."""
    above = np.flatnonzero(np.nan_to_num(score, nan=0.0) > threshold)
    picks: list[int] = []
    for i in above.tolist():
        if picks and i - picks[-1] <= spread:
            if score[i] > score[picks[-1]]:
                picks[-1] = i
            continue
        picks.append(i)
    return picks


def _neighbour_median(x: np.ndarray, k: int) -> np.ndarray:
    """For every element of ``x`` (1D), the median of the ``2k`` elements around it,
    itself excluded (edges: those that exist) -- a reference a spike cannot inflate."""
    n = x.shape[0]
    pad = np.concatenate([np.full(k, np.nan), x, np.full(k, np.nan)])
    rows = [pad[j : j + n] for j in range(2 * k + 1) if j != k]
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(np.stack(rows, 1), axis=1)
    return np.nan_to_num(out, nan=0.0)


def _savgol_residual(x: np.ndarray, m: int) -> np.ndarray:
    """``x`` ((n, d)) minus its cubic Savitzky-Golay smoothing over ``2m + 1`` samples
    (polynomials up to degree 3 pass unchanged, so only roughness remains; the ``m``
    samples at each end: 0)."""
    n = x.shape[0]
    out = np.zeros_like(x)
    if n < 2 * m + 1 or m < 2:
        return out
    i = np.arange(-m, m + 1, dtype=np.float64)
    c = 3.0 * (3.0 * m * m + 3.0 * m - 1.0 - 5.0 * i * i) / ((4.0 * m * m - 1.0) * (2.0 * m + 3.0))
    smooth = np.zeros((n - 2 * m, x.shape[1]))
    for j, cj in enumerate(c.tolist()):  # a fixed order of additions
        smooth += cj * x[j : j + n - 2 * m]
    out[m : n - m] = x[m : n - m] - smooth
    return out


def _gaussian_lowpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the first axis (``sigma`` in samples), ends reflected."""
    if sigma < 0.5:
        return x.copy()
    radius = int(math.ceil(3.0 * sigma))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    pad = np.concatenate([x[radius:0:-1], x, x[-2 : -radius - 2 : -1]], axis=0)
    if pad.shape[0] < x.shape[0] + 2 * radius:  # too short to reflect: hold the ends
        pad = np.concatenate([np.repeat(x[:1], radius, 0), x, np.repeat(x[-1:], radius, 0)], axis=0)
    out = np.zeros_like(x)
    for j, wj in enumerate(w.tolist()):  # a fixed order of additions
        out += wj * pad[j : j + x.shape[0]]
    return out


def _quaternions(rot: np.ndarray) -> np.ndarray:
    """Sign-continuous unit quaternions of rotation matrices ((n, 3, 3), scale removed)."""
    from .rotation import align, from_matrices

    q = from_matrices(rot)
    for i in range(1, len(q)):
        q[i] = align(q[i - 1], q[i])
    return q


def _angular_velocity(q: np.ndarray, h: float) -> np.ndarray:
    """World angular velocity vectors (rad/s) between samples (n - 1, 3)."""
    from .rotation import conjugate, log, multiply

    return 2.0 * log(multiply(q[1:], conjugate(q[:-1]))) / h


# --------------------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------------------- #


class _Findings:
    def __init__(self, limit: int) -> None:
        self.items: list[dict[str, Any]] = []
        self.limit = limit
        self.dropped = 0

    def add(self, code, severity, location, message, *, time=None, magnitude=None, **extra):
        if len(self.items) >= self.limit:
            self.dropped += 1
            return
        item = {
            "code": code,
            "severity": severity,
            "location": location,
            "time": None if time is None else float(time),
            "magnitude": None if magnitude is None else float(magnitude),
            "message": message,
            "basis": "sampled",
        }
        item.update(extra)
        self.items.append(item)


def _resolve_targets(system: Any, targets: Any) -> list[dict[str, Any]]:
    """Normalized target entries: ``{"kind": object|joint|camera|shots, ...}``."""
    scene = system.scene
    out: list[dict[str, Any]] = []
    if targets is None:
        seen = set()
        for (kind, oid), _binding in system.bindings():
            if kind == "object" and oid in scene._objects and oid not in seen:
                seen.add(oid)
                out.append({"object": oid})
        for (kind, oid), _item in list(system._owned.items()):
            if kind in ("rig", "constraints") and oid in scene._objects and oid not in seen:
                seen.add(oid)
                out.append({"object": oid})
        rigs = [e for e in out if system.rig(e["object"]) is not None]
        for e in rigs:
            out.append({"object": e["object"], "joints": "all"})
        if system._owned.get(("sequence", "sequence")) is not None:
            out.append({"camera": "shots"})
    else:
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        for i, raw in enumerate(targets):
            loc = f"targets[{i}]"
            if isinstance(raw, dict):
                unknown = set(raw) - {"object", "joint", "joints", "camera", "subject"}
                if unknown:
                    raise _err("INVALID_ARGUMENT", f"unknown target fields {sorted(unknown)}", loc)
                if raw.get("camera") == "shots":
                    out.append({"camera": "shots", "subject": raw.get("subject")})
                    continue
                ref = raw.get("object", raw.get("camera"))
                if ref is None:
                    raise _err("INVALID_ARGUMENT", "a target names an object", loc)
                obj = system._object(ref, f"{loc}.object")
                entry: dict[str, Any] = {"object": obj._id}
                if "joint" in raw:
                    entry["joints"] = [raw["joint"]]
                elif "joints" in raw:
                    entry["joints"] = raw["joints"]
                if "subject" in raw:
                    entry["subject"] = raw["subject"]
                out.append(entry)
            else:
                obj = system._object(raw, loc)
                out.append({"object": obj._id})
    if len(out) > MAX_TARGETS:
        raise _err("LIMIT_EXCEEDED", f"at most {MAX_TARGETS} targets", "targets")
    return out


def motion_report(
    system: Any,
    targets: Any = None,
    *,
    start: float | None = None,
    end: float | None = None,
    rate: float = 120.0,
    subject: Any = None,
    loop: bool = False,
    thresholds: dict[str, Any] | None = None,
    max_findings: int = 200,
    include_series: bool = False,
) -> dict[str, Any]:
    """World-space motion quality of ``targets`` (see the module docstring)."""
    s = system.start if start is None else float(start)
    e = system.end if end is None else float(end)
    if not (math.isfinite(s) and math.isfinite(e) and e > s):
        raise _err("INVALID_RANGE", "a motion report needs start < end", "range")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 10.0 <= rate <= 2000:
        raise _err("INVALID_ARGUMENT", "rate is 10..2000 samples per second", "rate")
    if not isinstance(max_findings, int) or not 1 <= max_findings <= 10_000:
        raise _err("INVALID_ARGUMENT", "max_findings is 1..10000", "max_findings")
    th = dict(DEFAULTS)
    for k, v in (thresholds or {}).items():
        if k not in DEFAULTS:
            raise _err("INVALID_ARGUMENT", f"unknown threshold {k!r}", f"thresholds.{k}")
        if v is not None and (
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        ):
            raise _err("INVALID_ARGUMENT", f"threshold {k} must be a number", f"thresholds.{k}")
        th[k] = v
    entries = _resolve_targets(system, targets)
    count = int(min(MAX_SAMPLES, max(16, round((e - s) * rate) + 1)))
    ts = np.linspace(s, e, count)
    h = float(ts[1] - ts[0])
    scene = system.scene
    # One evaluation per time for every target.
    object_ids = sorted({x["object"] for x in entries if "object" in x})
    joint_sets: dict[str, list[str] | None] = {}
    for x in entries:
        if "joints" in x:
            joint_sets[x["object"]] = None if x["joints"] == "all" else list(x["joints"])
    worlds = {oid: np.empty((count, 4, 4)) for oid in object_ids}
    joint_worlds: dict[str, dict[str, np.ndarray]] = {}
    shots = any(x.get("camera") == "shots" for x in entries)
    shot_frames: list[Any] = []
    for i, t in enumerate(ts.tolist()):
        posed = system._posed_worlds(t)
        for oid in object_ids:
            worlds[oid][i] = posed[oid]
        if joint_sets:
            rr = system.rigging(t, deform=False)
            for oid, names in joint_sets.items():
                found = rr.joints.get(oid)
                if found is None:
                    raise _err(
                        "JOINT_MISSING", f"{scene._objects[oid].name!r} has no rig", "targets"
                    )
                rig, model = found
                if names is None:
                    names = [j.name for j in rig.joints]
                    joint_sets[oid] = names
                table = joint_worlds.setdefault(oid, {n: np.empty((count, 4, 4)) for n in names})
                for name in names:
                    k = rig.index(name)
                    if k is None:
                        raise _err(
                            "JOINT_MISSING",
                            f"{scene._objects[oid].name!r} has no joint {name!r}",
                            "targets",
                        )
                    table[name][i] = posed[oid] @ model[k]
        if shots:
            shot_frames.append(_shot_camera(system, t))
    findings = _Findings(max_findings)
    reports: list[dict[str, Any]] = []
    series: dict[str, Any] = {}
    for x in entries:
        if x.get("camera") == "shots":
            frames = shot_frames
            if not frames or all(f is None for f in frames):
                findings.add("NO_SHOTS", "info", "shots", "the scene has no shot sequence")
                continue
            mats = np.array([f[0] if f is not None else np.eye(4) for f in frames])
            cuts = [
                i
                for i in range(1, count)
                if frames[i] is not None
                and frames[i - 1] is not None
                and frames[i][1] != frames[i - 1][1]
                and "->" not in frames[i][1]
                and "->" not in frames[i - 1][1]
            ]
            rep = _measure(
                "shots", mats, ts, h, th, findings, camera=True, cuts=cuts,
                subject=x.get("subject", subject), system=system, loop=loop,
                fov=[f[2] if f is not None else 40.0 for f in frames],
            )  # fmt: skip
            reports.append({"target": "shots", "kind": "camera", **rep["summary"]})
            if include_series:
                series["shots"] = rep["series"]
            continue
        obj = scene._objects[x["object"]]
        if "joints" in x:
            table = joint_worlds.get(x["object"], {})
            for name, mats in table.items():
                rep = _measure(f"{obj.name}.joints[{name}]", mats, ts, h, th, findings, loop=loop)
                reports.append(
                    {"target": obj.name, "joint": name, "kind": "joint", **rep["summary"]}
                )
                if include_series:
                    series[f"{obj.name}/{name}"] = rep["series"]
            _pole_flips(system, obj, table, ts, findings)
            continue
        is_camera = obj.camera is not None
        rep = _measure(
            obj.name, worlds[x["object"]], ts, h, th, findings, camera=is_camera,
            subject=x.get("subject", subject) if is_camera else None, system=system,
            loop=loop, fov=[obj.camera.fov] * count if is_camera else None,
        )  # fmt: skip
        reports.append(
            {"target": obj.name, "kind": "camera" if is_camera else "object", **rep["summary"]}
        )
        if include_series:
            series[obj.name] = rep["series"]
    if th["ground"] is not None:
        _ground_penetration(system, entries, ts, th, findings)
    counts: dict[str, int] = {}
    for f in findings.items:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    out = {
        "range": [s, e],
        "rate": float(rate),
        "samples": count,
        "basis": "sampled",
        "thresholds": th,
        "targets": reports,
        "findings": findings.items,
        "counts": counts,
        "truncated": findings.dropped > 0,
    }
    if include_series:
        out["series"] = series
    return out


def _shot_camera(system: Any, t: float):
    """``(world matrix, shot label, vertical fov)`` of the shot sequence's camera at ``t``
    (a cut camera or a blend's virtual camera), or ``None``."""
    from .cinematics import shot_camera

    item = system._owned.get(("sequence", "sequence"))
    if item is None:
        return None
    shot, prev, _w = item._definition.at(t)
    if shot is None:
        return None
    posed = system._posed_worlds(t)
    overlay = system.overlay(t)
    fields = {} if overlay is None else dict(overlay.fields)
    found = shot_camera(system, t, posed, fields)
    if found is None:
        return None
    label = shot.name if prev is None else f"{prev.name}->{shot.name}"
    if isinstance(found, str):
        cam = system.scene._objects[found]
        values = fields.get((cam._id, "camera"))
        fov = float(values.get("fov", cam.camera.fov)) if values else float(cam.camera.fov)
        return np.array(posed[found], dtype=np.float64), label, fov
    frame = found.frame
    m = np.eye(4)
    m[:3, 0] = frame.right
    m[:3, 1] = -np.asarray(frame.forward)
    m[:3, 2] = frame.up
    m[:3, 3] = frame.position
    return m, label, float(found.camera.fov)


def _measure(
    label: str,
    mats: np.ndarray,
    ts: np.ndarray,
    h: float,
    th: dict[str, Any],
    findings: _Findings,
    *,
    camera: bool = False,
    cuts: list[int] | None = None,
    subject: Any = None,
    system: Any = None,
    loop: bool = False,
    fov: list[float] | None = None,
) -> dict[str, Any]:
    n = mats.shape[0]
    pos = mats[:, :3, 3].copy()
    lin = mats[:, :3, :3]
    rot = lin / np.maximum(np.linalg.norm(lin, axis=1, keepdims=True), 1e-300)
    q = _quaternions(rot)
    cut_set = set(cuts or [])
    vel = np.gradient(pos, h, axis=0)
    acc = np.gradient(vel, h, axis=0)
    jerk = np.gradient(acc, h, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    peak = float(speed.max())
    omega = _angular_velocity(q, h) if n > 1 else np.zeros((0, 3))
    ang_speed = np.degrees(np.linalg.norm(omega, axis=1)) if omega.size else np.zeros(0)
    ang_peak = float(ang_speed.max()) if ang_speed.size else 0.0
    w = max(3, int(round(th["window"] / h)))
    spread = 2 * w
    # Position jumps: one sample's travel against the travel around it.
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    local = _neighbour_median(step, 4)
    jumps = np.flatnonzero(step > np.maximum(th["position_jump"], 8.0 * local)).tolist()
    for i in jumps:
        if i + 1 in cut_set:
            findings.add(
                "CUT", "info", label, f"a cut at {ts[i + 1]:.4g} s", time=ts[i + 1],
                magnitude=step[i],
            )  # fmt: skip
            continue
        findings.add(
            "POSITION_JUMP",
            "error",
            label,
            f"moves {step[i]:.4g} m within one sample at {ts[i + 1]:.4g} s",
            time=ts[i + 1],
            magnitude=step[i],
        )
    # Velocity jumps and acceleration steps: differences of one-sided fits.
    vb, vf = _one_sided_fits(vel, w)
    dv = np.linalg.norm(vf - vb, axis=1)
    lim_v = max(th["velocity_jump"] * peak, th["velocity_jump_min"])
    near_cut = np.zeros(n, bool)
    for c in [*cut_set, *(i + 1 for i in jumps)]:  # cuts and teleports: already reported
        near_cut[max(0, c - spread) : c + spread + 1] = True
    edge = w + 3  # the range's ends: one-sided fits (and differences) incomplete
    near_cut[:edge] = near_cut[max(0, n - edge) :] = True
    vjumps = _peaks(np.where(near_cut, 0.0, dv), lim_v, spread)
    for i in vjumps:
        findings.add(
            "VELOCITY_JUMP",
            "warning",
            label,
            f"velocity changes by {dv[i]:.4g} m/s at {ts[i]:.4g} s (peak speed {peak:.4g})",
            time=ts[i],
            magnitude=dv[i],
        )
    ab, af = _one_sided_fits(acc, w)
    da = np.linalg.norm(af - ab, axis=1)
    near_v = near_cut.copy()
    for i in vjumps:  # a velocity jump's acceleration impulse: already reported
        near_v[max(0, i - spread) : i + spread + 1] = True
    steps = _peaks(np.where(near_v, 0.0, da), th["acceleration_step"], spread)
    for i in steps:
        findings.add(
            "ACCELERATION_STEP",
            "info",
            label,
            f"acceleration steps by {da[i]:.4g} m/s^2 at {ts[i]:.4g} s",
            time=ts[i],
            magnitude=da[i],
        )
    # Rotation: snaps (one sample's turn) and angular-velocity jumps.
    turn = ang_speed * h  # degrees per sample
    if turn.size:
        local_t = _neighbour_median(turn, 4)
        snaps = np.flatnonzero(turn > np.maximum(th["snap_deg"], 8.0 * local_t)).tolist()
        for i in snaps:
            near_cut[max(0, i + 1 - spread) : i + 2 + spread] = True
            if i + 1 in cut_set:
                continue
            findings.add(
                "ROTATION_SNAP",
                "warning",
                label,
                f"turns {turn[i]:.4g} degrees within one sample at {ts[i + 1]:.4g} s",
                time=ts[i + 1],
                magnitude=turn[i],
            )
        om = np.degrees(omega)
        ob, of = _one_sided_fits(om, w)
        dw = np.linalg.norm(of - ob, axis=1)
        lim_w = max(th["angular_jump"] * ang_peak, th["angular_jump_min"])
        near = near_cut[: dw.shape[0]]
        for i in _peaks(np.where(near, 0.0, dw), lim_w, spread):
            findings.add(
                "ANGULAR_VELOCITY_JUMP",
                "warning",
                label,
                f"angular velocity changes by {dw[i]:.4g} deg/s at {ts[i]:.4g} s",
                time=ts[i],
                magnitude=dw[i],
            )
    # Jitter: what a cubic local fit over about one period of jitter_hz leaves (smooth
    # motion -- polynomials up to degree 3 -- leaves nothing; cuts and jumps split it).
    m = max(2, int(round(0.5 / (th["jitter_hz"] * h))))
    breaks = set(cut_set) | {i + 1 for i in jumps} | set(vjumps)
    segments = _split(n, breaks)
    res_p = []
    res_a = []
    for a, b in segments:
        # (corners at the breaks are reported as jumps: leave a window out of each side)
        a, b = a + (w if a else 0), b - (w if b < n else 0)
        if b - a < 2 * m + 1:
            continue
        res_p.append(np.linalg.norm(_savgol_residual(pos[a:b], m), axis=1)[m : b - a - m])
        # Quaternion components (sign-continuous) are smooth and never wrap; a small
        # residual r of a unit quaternion is a turn of 2 |r| radians.
        rq = np.linalg.norm(_savgol_residual(q[a:b], m), axis=1)[m : b - a - m]
        res_a.append(np.degrees(2.0 * rq))
    jit_mm = float(np.sqrt(np.mean(np.concatenate(res_p) ** 2)) * 1000.0) if res_p else 0.0
    jit_deg = float(np.sqrt(np.mean(np.concatenate(res_a) ** 2))) if res_a else 0.0
    if jit_mm > th["jitter_mm"] or jit_deg > th["jitter_deg"]:
        findings.add(
            "CAMERA_JITTER" if camera else "JITTER",
            "warning",
            label,
            f"{jit_mm:.3g} mm and {jit_deg:.3g} degrees RMS of motion above {th['jitter_hz']:g} Hz",
            magnitude=max(jit_mm / th["jitter_mm"], jit_deg / th["jitter_deg"]),
            position_mm=jit_mm,
            angle_deg=jit_deg,
        )
    # Pauses between moves.
    if peak > 1e-6:
        slow = speed < th["pause_speed"] * peak
        k = 0
        while k < n:
            if not slow[k]:
                k += 1
                continue
            j = k
            while j < n and slow[j]:
                j += 1
            dur = (j - k) * h
            moving_before = k > 0 and speed[max(0, k - spread) : k].max() > 0.1 * peak
            moving_after = j < n and speed[j : j + spread].max() > 0.1 * peak
            if th["pause_min"] <= dur <= th["pause_max"] and moving_before and moving_after:
                findings.add(
                    "PAUSE",
                    "info",
                    label,
                    f"the motion stops for {dur:.3g} s at {ts[k]:.4g} s between moves",
                    time=ts[k],
                    magnitude=dur,
                )
            k = j
    # Loop seam: the end against the start.
    seam = None
    if loop and n > 4:
        seam = {
            "velocity": float(np.linalg.norm(vel[-1] - vel[0])),
            "acceleration": float(np.linalg.norm(acc[-1] - acc[0])),
            "angular_velocity": float(np.linalg.norm(np.degrees(omega[-1] - omega[0])))
            if omega.size
            else 0.0,
        }
        if seam["velocity"] > lim_v or seam["angular_velocity"] > max(
            th["angular_jump"] * ang_peak, th["angular_jump_min"]
        ):
            findings.add(
                "LOOP_SEAM",
                "warning",
                label,
                f"looping, velocity jumps by {seam['velocity']:.4g} m/s and angular velocity "
                f"by {seam['angular_velocity']:.4g} deg/s at the seam",
                time=float(ts[-1]),
                magnitude=seam["velocity"],
            )
    ldlj = None
    duration = float(ts[-1] - ts[0])
    jk = np.linalg.norm(jerk, axis=1)
    if peak > 1e-9:
        dimless = (duration**3 / peak**2) * float(np.trapezoid(jk**2, ts))
        ldlj = -math.log(max(dimless, 1e-300))
    summary: dict[str, Any] = {
        "path_length": float(step.sum()),
        "speed": {"max": peak, "mean": float(speed.mean())},
        "acceleration": {"max": float(np.linalg.norm(acc, axis=1).max())},
        "jerk": {"max": float(jk.max()), "rms": float(np.sqrt(np.mean(jk**2)))},
        "angular_speed": {"max": ang_peak},
        "jitter": {"position_mm": jit_mm, "angle_deg": jit_deg},
        "acceleration_steps": len(steps),
        "smoothness_ldlj": ldlj,
    }
    if seam is not None:
        summary["loop_seam"] = seam
    if camera and subject is not None and system is not None:
        summary["framing"] = _framing(label, mats, ts, subject, system, th, findings, fov, cuts)
    return {
        "summary": summary,
        "series": {
            "time": ts[:: max(1, n // 600)].tolist(),
            "speed": speed[:: max(1, n // 600)].tolist(),
            "acceleration": np.linalg.norm(acc, axis=1)[:: max(1, n // 600)].tolist(),
        },
    }


def _split(n: int, cuts: set[int]) -> list[tuple[int, int]]:
    bounds = [0, *sorted(c for c in cuts if 0 < c < n), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _subject_points(system: Any, subject: Any, ts: np.ndarray) -> np.ndarray:
    """The subject's world position at every time (a point, or an object's bounds
    centre as animated)."""
    if isinstance(subject, (list, tuple)) and len(subject) == 3:
        p = np.asarray(subject, dtype=np.float64)
        if not np.all(np.isfinite(p)):
            raise _err("INVALID_ARGUMENT", "a subject point must be finite", "subject")
        return np.repeat(p[None, :], ts.shape[0], 0)
    obj = system._object(subject, "subject")
    centre = np.zeros(4)
    centre[3] = 1.0
    mesh = obj.mesh
    if mesh is not None and len(mesh.positions):
        lo, hi = mesh.positions.min(axis=0), mesh.positions.max(axis=0)
        centre[:3] = 0.5 * (lo + hi)
    out = np.empty((ts.shape[0], 3))
    for i, t in enumerate(ts.tolist()):
        out[i] = (system._posed_worlds(t)[obj._id] @ centre)[:3]
    return out


def _framing(label, mats, ts, subject, system, th, findings, fov, cuts) -> dict[str, Any]:
    """Aim error, screen position and its drift, out-of-frame and obstruction."""
    from ..rendering.camera import Camera  # noqa: F401 - conventions only

    target = _subject_points(system, subject, ts)
    eye = mats[:, :3, 3]
    lin = mats[:, :3, :3]
    rot = lin / np.maximum(np.linalg.norm(lin, axis=1, keepdims=True), 1e-300)
    forward = -rot[:, :, 1]  # cameras look along local -Y
    right = rot[:, :, 0]
    up = rot[:, :, 2]
    d = target - eye
    dist = np.linalg.norm(d, axis=1)
    dn = d / np.maximum(dist, 1e-12)[:, None]
    aim = np.degrees(np.arccos(np.clip(np.sum(dn * forward, axis=1), -1.0, 1.0)))
    depth = np.sum(d * forward, axis=1)
    aspect = _aspect(system)
    tan_y = np.tan(np.radians(np.asarray(fov if fov is not None else [40.0] * len(ts))) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.sum(d * right, axis=1) / (depth * tan_y * aspect)
        sy = np.sum(d * up, axis=1) / (depth * tan_y)
    behind = depth <= 0
    margin = th["frame_margin"]
    out_of_frame = behind | (np.abs(sx) > 1.0 - margin) | (np.abs(sy) > 1.0 - margin)
    if np.any(out_of_frame):
        i = int(np.flatnonzero(out_of_frame)[0])
        frac = float(out_of_frame.mean())
        findings.add(
            "SUBJECT_OUT_OF_FRAME",
            "error" if frac > 0.1 else "warning",
            label,
            f"the subject leaves the frame {frac * 100:.3g} % of the time (first at {ts[i]:.4g} s)",
            time=ts[i],
            magnitude=frac,
        )
    # Drift: how fast the subject swims across the frame (frame half-widths per second),
    # away from cuts.
    h = float(ts[1] - ts[0])
    screen = np.stack([sx, sy], 1)
    vel = np.linalg.norm(np.gradient(screen, h, axis=0), axis=1)
    for c in cuts or []:
        vel[max(0, c - 2) : c + 3] = 0.0
    vel = np.where(np.isfinite(vel), vel, 0.0)
    drift = float(np.percentile(vel, 95)) if vel.size else 0.0
    if drift > th["drift"]:
        findings.add(
            "FRAMING_DRIFT",
            "info",
            label,
            f"the subject moves across the frame at {drift:.3g} half-widths/s (95th percentile)",
            magnitude=drift,
        )
    # Obstruction: line of sight at up to 32 sampled times.
    blocked = []
    if system is not None:
        picks = np.linspace(0, len(ts) - 1, min(MAX_SIGHT_SAMPLES, len(ts))).round().astype(int)
        exclude = []
        if not (isinstance(subject, (list, tuple)) and len(subject) == 3):
            exclude = [system._object(subject, "subject")._id]
        anim = system
        before = anim.time
        try:
            for i in picks.tolist():
                anim.time = float(ts[i])
                sight = system.scene.line_of_sight(
                    eye[i].tolist(), target[i].tolist(), exclude=exclude or None
                )
                if not sight.clear:
                    blocked.append(float(ts[i]))
        finally:
            anim.time = before
        if blocked:
            findings.add(
                "SUBJECT_OBSTRUCTED",
                "warning",
                label,
                f"something lies between the camera and the subject at {len(blocked)} of "
                f"{len(picks)} sampled times (first at {blocked[0]:.4g} s)",
                time=blocked[0],
                magnitude=len(blocked) / len(picks),
            )
    return {
        "aim_error_deg": {"mean": float(np.mean(aim)), "max": float(np.max(aim))},
        "screen": {
            "mean": [float(np.nanmean(sx)), float(np.nanmean(sy))],
            "max_offset": float(np.nanmax(np.hypot(sx, sy))),
        },
        "drift": drift,
        "distance": {"min": float(dist.min()), "max": float(dist.max())},
        "out_of_frame": float(out_of_frame.mean()),
        "obstructed_samples": len(blocked),
    }


def _aspect(system: Any) -> float:
    r = system.scene.render
    try:
        return float(r.width) / float(r.height)
    except Exception:  # noqa: BLE001
        return 16.0 / 9.0


def _pole_flips(system: Any, obj: Any, table: dict[str, np.ndarray], ts, findings) -> None:
    """IK chains whose bend plane turns over within one sample."""
    rig = system.rig(obj)
    if rig is None:
        return
    for c in rig.constraints:
        if c.type != "ik" or c.mute:
            continue
        chain = c.p.get("chain", [])
        if len(chain) < 3 or any(j not in table for j in chain[:3]):
            continue
        a = table[chain[0]][:, :3, 3]
        b = table[chain[1]][:, :3, 3]
        e = table[chain[-1]][:, :3, 3]
        n = np.cross(b - a, e - b)
        size = np.linalg.norm(n, axis=1)
        ref = np.linalg.norm(b - a, axis=1) * np.linalg.norm(e - b, axis=1)
        bent = size > 0.05 * np.maximum(ref, 1e-12)
        nn = n / np.maximum(size, 1e-300)[:, None]
        dots = np.sum(nn[1:] * nn[:-1], axis=1)
        flips = np.flatnonzero((dots < 0.0) & bent[1:] & bent[:-1])
        for i in flips.tolist():
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(dots[i])))))
            findings.add(
                "POLE_FLIP",
                "warning",
                f"{obj.name}.constraints[{c.name}]",
                f"the {chain[0]}..{chain[-1]} chain's bend plane turns {angle:.3g} degrees "
                f"within one sample at {ts[i + 1]:.4g} s",
                time=ts[i + 1],
                magnitude=angle,
            )


def _ground_penetration(system: Any, entries, ts, th, findings) -> None:
    """The lowest vertex of each mesh object below the ground height (sampled at <= 30
    Hz; skins deformed)."""
    scene = system.scene
    ground = float(th["ground"])
    tol = float(th["ground_tolerance"])
    ids = sorted({x["object"] for x in entries if "object" in x and "joints" not in x})
    meshes = [oid for oid in ids if scene._objects[oid].mesh is not None]
    # skinned meshes bound to the rigs of the targets
    for (kind, oid), _item in list(system._owned.items()):
        if kind == "skin" and oid in scene._objects and oid not in meshes:
            meshes.append(oid)
    if not meshes:
        return
    step = max(1, int(round((ts[1] - ts[0]) ** -1 / 30.0)))
    worst: dict[str, tuple[float, float]] = {}
    for t in ts[::step].tolist():
        overlay = system.overlay(t)
        posed = system._posed_worlds(t)
        for oid in meshes:
            obj = scene._objects[oid]
            mesh = obj.mesh if overlay is None else overlay.meshes.get(oid, obj.mesh)
            if mesh is None or not len(mesh.positions):
                continue
            w = posed[oid]
            p = np.asarray(mesh.positions, dtype=np.float64)
            z = p[:, 0] * w[2, 0] + p[:, 1] * w[2, 1] + p[:, 2] * w[2, 2] + w[2, 3]
            depth = ground - float(z.min())
            if depth > worst.get(oid, (0.0, 0.0))[0]:
                worst[oid] = (depth, t)
    for oid, (depth, t) in worst.items():
        if depth > tol:
            findings.add(
                "GROUND_PENETRATION",
                "warning",
                scene._objects[oid].name,
                f"goes {depth * 1000:.3g} mm below the ground at {t:.4g} s",
                time=t,
                magnitude=depth,
            )
