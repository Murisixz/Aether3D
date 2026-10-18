"""Joint limits: Euler bounds (Stage 17), swing-twist cones and ellipses, and hinges.

A joint's ``limits`` take one of three forms:

* ``{"rotation": {"min": [x, y, z], "max": [x, y, z]}}`` -- Euler XYZ degrees of the
  joint's local rotation (Stage 17; ``null`` for no bound). Unchanged.
* ``{"axis": "+y", "swing": {...}, "twist": [min, max]}`` -- measured **from the rest
  rotation** (``q_rel = q_rest^-1 q_local``) and split into a *twist* about the bone
  ``axis`` (default ``+y``: bones point along their local +Y) and a *swing* that moves
  the axis (Dobrowolski 2015). ``swing`` is ``{"cone": degrees}`` (a circular cone) or
  bounds per perpendicular axis, ``{"x": [min, max], "z": [min, max]}`` for a ``+y``
  bone: the swing's rotation vector (degrees) must lie inside the ellipse quadrants these
  bounds span (Baerlocher and Boulic 2001; asymmetric ranges -- a shoulder lifting
  further forward than back -- are four quarter ellipses). Either part may be omitted.
* ``{"hinge": {"axis": "+x", "min": deg, "max": deg, "play": deg}}`` -- only rotation
  about one local axis, from the rest rotation, within ``min..max`` (an elbow or a knee):
  swing out of the hinge plane beyond ``play`` (default 0) is removed.

Every new-form range must contain 0 (the rest pose is allowed). Clamping is stable at
quaternion sign changes and wraparound: ``q_rel`` is taken with ``w >= 0`` (the shorter
rotation), so twist angles lie in ``-180..180`` and swings in ``0..180`` degrees; a
swing outside its ellipse is scaled back radially onto it (stable, the direction kept).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import rotation as R

AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}
_LETTERS = "xyz"


def _err(message: str, location: str) -> AnimationError:
    return AnimationError("INVALID_RIG", message, location)


def _deg(value: Any, location: str, lo: float = -180.0, hi: float = 180.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err(f"expected a number of degrees, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v) or not lo <= v <= hi:
        raise _err(f"degrees must be within {lo:g}..{hi:g}", location)
    return v


def _range(value: Any, location: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _err("a range is [min, max] in degrees", location)
    lo = _deg(value[0], f"{location}[0]")
    hi = _deg(value[1], f"{location}[1]")
    if not lo <= 0.0 <= hi:
        raise _err("a range must contain 0 (the rest pose): min <= 0 <= max", location)
    return [lo, hi]


def perpendicular(axis: str) -> tuple[str, str]:
    """The two other axis letters of a bone axis (``"+y"`` -> ``("x", "z")``)."""
    letter = axis[1]
    a, b = (c for c in _LETTERS if c != letter)
    return a, b


def check_limits(raw: Any, location: str) -> str:
    """Canonical JSON of checked limits (see the module docstring)."""
    if not isinstance(raw, dict) or not raw:
        raise _err("limits are {rotation}, {axis, swing, twist} or {hinge}", location)
    keys = set(raw)
    if "hinge" in keys:
        if keys != {"hinge"}:
            raise _err("a hinge limit has only {hinge: {axis, min, max}}", location)
        h = raw["hinge"]
        loc = f"{location}.hinge"
        if not isinstance(h, dict) or set(h) - {"axis", "min", "max", "play"}:
            raise _err("hinge is {axis, min, max, play}", loc)
        axis = h.get("axis", "+x")
        if axis not in AXES:
            raise _err(f"axis is one of {', '.join(AXES)}", f"{loc}.axis")
        lo, hi = _range([h.get("min", -180.0), h.get("max", 180.0)], loc)
        out = {"axis": axis, "min": lo, "max": hi}
        if h.get("play", 0.0):
            out["play"] = _deg(h["play"], f"{loc}.play", 0.0, 45.0)
        return _canonical({"hinge": out})
    if keys - {"axis", "swing", "twist"} or not keys & {"swing", "twist"}:
        raise _err(
            "limits are {rotation: {min, max}}, {axis, swing, twist} or {hinge: {axis, min, max}}",
            location,
        )
    axis = raw.get("axis", "+y")
    if axis not in AXES:
        raise _err(f"axis is one of {', '.join(AXES)}", f"{location}.axis")
    out: dict[str, Any] = {"axis": axis}
    if raw.get("swing") is not None:
        s = raw["swing"]
        loc = f"{location}.swing"
        a, b = perpendicular(axis)
        if not isinstance(s, dict) or not s:
            raise _err(f"swing is {{cone: degrees}} or {{{a}: [min, max], {b}: [min, max]}}", loc)
        if "cone" in s:
            if set(s) != {"cone"}:
                raise _err("a cone swing has only {cone: degrees}", loc)
            cone = _deg(s["cone"], f"{loc}.cone", 0.0, 180.0)
            out["swing"] = {"cone": cone}
        else:
            if set(s) - {a, b}:
                raise _err(f"swing bounds are about the axes {a} and {b} (bone axis {axis})", loc)
            out["swing"] = {k: _range(s.get(k, [-180.0, 180.0]), f"{loc}.{k}") for k in (a, b)}
    if raw.get("twist") is not None:
        out["twist"] = _range(raw["twist"], f"{location}.twist")
    return _canonical(out)


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------- #
# Measuring and clamping
# --------------------------------------------------------------------------------------- #


def relative(q_local: np.ndarray, q_rest: np.ndarray) -> np.ndarray:
    """``q_rest^-1 q_local`` with ``w >= 0`` (the rotation from the rest pose)."""
    q = R.multiply(R.conjugate(np.asarray(q_rest, np.float64)), np.asarray(q_local, np.float64))
    q = q / math.sqrt(float(q @ q))
    return -q if q[3] < 0.0 else q


def twist_angle(q_rel: np.ndarray, axis: np.ndarray) -> float:
    """Signed twist (degrees) about ``axis`` of a rotation with ``w >= 0``."""
    p = float(q_rel[:3] @ axis)
    if abs(p) < 1e-15 and abs(q_rel[3]) < 1e-15:
        return 0.0
    return math.degrees(2.0 * math.atan2(p, q_rel[3]))


def swing_vector(q_rel: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The swing part's rotation vector (degrees; perpendicular to ``axis``)."""
    swing, _ = R.swing_twist(q_rel[None], axis[None])
    s = swing[0]
    if s[3] < 0.0:
        s = -s
    v = s[:3] - float(s[:3] @ axis) * axis  # (exactly perpendicular)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.zeros(3)
    angle = 2.0 * math.atan2(n, s[3])
    return v / n * math.degrees(angle)


def angles(q_local: Any, q_rest: Any, axis: str = "+y") -> dict[str, Any]:
    """``{"swing": degrees, "twist": degrees, "swing_vector": [x, y, z]}`` of a joint's
    rotation from its rest pose (for drivers and diagnostics)."""
    a = AXES[axis]
    q = relative(np.asarray(q_local, np.float64), np.asarray(q_rest, np.float64))
    sv = swing_vector(q, a)
    return {
        "swing": float(np.linalg.norm(sv)),
        "twist": twist_angle(q, a),
        "swing_vector": [float(x) for x in sv],
        "angle": math.degrees(2.0 * math.acos(min(1.0, float(q[3])))),
    }


def _axis_rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    h = math.radians(degrees) / 2.0
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)])


def _from_vector(v: np.ndarray) -> np.ndarray:
    """The rotation with rotation vector ``v`` (degrees)."""
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return _axis_rotation(v / n, n)


def clamp(limits: dict[str, Any], q_local: np.ndarray, q_rest: np.ndarray) -> np.ndarray | None:
    """The local rotation within swing-twist or hinge ``limits`` (``None``: already
    inside, nothing to change). Euler limits are applied by the rigging pass."""
    q = relative(q_local, q_rest)
    if "hinge" in limits:
        h = limits["hinge"]
        axis = AXES[h["axis"]]
        angle = twist_angle(q, axis)
        swing = swing_vector(q, axis)
        clamped = min(max(angle, h["min"]), h["max"])
        play = h.get("play", 0.0)
        norm = float(np.linalg.norm(swing))
        if clamped == angle and norm <= max(play, 1e-9):
            return None
        kept = swing * (play / norm) if norm > play else swing  # (the allowed play remains)
        new_rel = R.multiply(_from_vector(kept), _axis_rotation(axis, clamped))
    else:
        axis_name = limits["axis"]
        axis = AXES[axis_name]
        twist = twist_angle(q, axis)
        swing = swing_vector(q, axis)
        changed = False
        if "twist" in limits:
            lo, hi = limits["twist"]
            t2 = min(max(twist, lo), hi)
            changed |= t2 != twist
            twist = t2
        if "swing" in limits:
            new_swing = _clamp_swing(limits["swing"], swing, axis_name)
            if new_swing is not None:
                swing, changed = new_swing, True
        if not changed:
            return None
        new_rel = R.multiply(_from_vector(swing), _axis_rotation(axis, twist))
    out = R.multiply(np.asarray(q_rest, np.float64), new_rel)
    return out / math.sqrt(float(out @ out))


def _clamp_swing(spec: dict[str, Any], v: np.ndarray, axis: str) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    if "cone" in spec:
        cone = spec["cone"]
        if norm <= cone:
            return None
        return v * (cone / norm)
    a, b = perpendicular(axis)
    ia, ib = _LETTERS.index(a), _LETTERS.index(b)
    ca, cb = float(v[ia]), float(v[ib])
    ra = spec[a][1] if ca >= 0.0 else -spec[a][0]
    rb = spec[b][1] if cb >= 0.0 else -spec[b][0]
    # A zero bound allows no swing that way: that component goes first.
    if ra <= 0.0:
        ca = 0.0
    if rb <= 0.0:
        cb = 0.0
    ta = 0.0 if ra <= 0.0 else ca / ra
    tb = 0.0 if rb <= 0.0 else cb / rb
    e = ta * ta + tb * tb
    if e <= 1.0 and ca == float(v[ia]) and cb == float(v[ib]):
        return None
    f = 1.0 if e <= 1.0 else 1.0 / math.sqrt(e)
    out = np.zeros(3)
    out[ia] = ca * f
    out[ib] = cb * f
    return out


def is_euler(limits: dict[str, Any] | None) -> bool:
    return bool(limits) and "rotation" in limits
