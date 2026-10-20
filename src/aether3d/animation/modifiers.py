"""F-curve modifiers: composable functions applied to a curve, in order, with derivatives.

A curve's modifiers form a chain: ``[m1, m2]`` evaluates ``m2(m1(curve))``. A *value*
modifier maps the value (``m(c)(t) = g(c(t), t)``); a *time* modifier maps time
(``m(c)(t) = c(h(t))``), so a time modifier applies to everything before it in the chain
(``[noise, stepped]`` holds the noise too; ``[stepped, noise]`` does not). Every modifier
has an ``influence`` (0..1) and an optional ``range`` ``[start, end]`` with ``blend_in``
and ``blend_out`` ramps inside it; outside the range it does nothing. ``mute`` skips it.

Types (parameters):

* ``offset`` (``value``): ``v + value``.
* ``scale`` (``factor``, ``pivot`` = 0): ``pivot + (v - pivot) * factor``.
* ``limits`` (``min``, ``max``, either optional): clamp; derivatives are 0 where clamped.
* ``noise`` (``strength``, ``scale`` s, ``offset`` = 0, ``octaves`` = 1, ``roughness`` =
  0.5, ``seed`` = 0): ``v + strength * N((t + offset) / scale)``, ``N`` smooth (C2,
  quintic-fade gradient noise, about -1..1) and deterministic in ``seed``.
* ``wave`` (``amplitude``, ``period`` s, ``phase`` = 0 s, ``decay`` = 0 /s, ``start`` = 0
  s): ``v + amplitude * exp(-decay * max(t - start, 0)) * sin(2 pi (t - phase) / period)``
  -- breathing, hovering, damped follow-through.
* ``envelope`` (``reference``, ``points``: ``[[time, gain, offset], ...]``):
  ``reference + (v - reference) * G(t) + O(t)``, ``G``, ``O`` piecewise linear (constant
  outside the points).
* ``smooth`` (``sigma`` s): Gaussian smoothing, ``sum_j w_j v(t + tau_j)`` over 15
  fixed nodes within 3 sigma (seekable: no state between frames).
* ``generator`` (``coefficients`` ``[c0, c1, ...]``, up to degree 10; ``additive`` = true):
  ``v + sum_k c_k t^k`` (``additive: false`` replaces the value) -- drifts, ramps, trends.
* ``function`` (``function``: ``sin``, ``cos``, ``sinc``, ``sqrt`` or ``ln``; ``amplitude``
  = 1, ``phase_multiplier`` = 1, ``phase_offset`` = 0, ``value_offset`` = 0, ``additive`` =
  true): ``v + amplitude * f(phase_multiplier * t + phase_offset) + value_offset``, as
  Blender's *Built-In Function* modifier (``sinc(x) = sin(x) / x``; ``sqrt`` and ``ln`` give
  0 outside their domain).
* ``stepped`` (``step`` s, ``offset`` = 0 s): time held on steps of ``step`` -- sample and
  hold, on-twos/threes animation.
* ``cycles`` (``before``, ``after``: ``repeat`` (default), ``repeat_offset``, ``mirror`` or
  ``none``; ``count_before``, ``count_after``: whole cycles, 0 = without end): repeats the
  curve's key range outside it, as Blender's *Cycles* modifier: each repetition of
  ``repeat_offset`` is shifted by the keys' value change, ``mirror`` runs every other
  repetition backwards, and after the counted cycles time passes through (the curve's own
  extrapolation). The end of a cycle belongs to it. As in Blender, a ``cycles`` modifier
  first in the chain, repeating without end on both sides (no range, full influence),
  makes the automatic handles cycle-aware, like cyclic extrapolation.
* ``time_offset`` (``offset`` s): ``c(t - offset)``.
* ``time_scale`` (``factor``, ``pivot`` = 0 s): ``c(pivot + (t - pivot) * factor)``.

Derivatives are exact for each modifier's definition (chain and product rules; the
ramps and envelopes are piecewise linear, so their second derivatives are 0 between
breakpoints).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError

if TYPE_CHECKING:
    from .curves import FCurve

MAX_MODIFIERS = 16
MAX_SMOOTH = 2
SMOOTH_NODES = 7  # each side: 2 * 7 + 1 = 15 nodes
MAX_OCTAVES = 8
MAX_ENVELOPE_POINTS = 1000

TYPES: dict[str, dict[str, Any]] = {
    "offset": {"required": ("value",), "optional": {}},
    "scale": {"required": ("factor",), "optional": {"pivot": 0.0}},
    "limits": {"required": (), "optional": {"min": None, "max": None}},
    "noise": {
        "required": ("strength", "scale"),
        "optional": {"offset": 0.0, "octaves": 1, "roughness": 0.5, "seed": 0},
    },
    "wave": {
        "required": ("amplitude", "period"),
        "optional": {"phase": 0.0, "decay": 0.0, "start": 0.0},
    },
    "envelope": {"required": ("points",), "optional": {"reference": 0.0}},
    "smooth": {"required": ("sigma",), "optional": {}},
    "generator": {"required": ("coefficients",), "optional": {"additive": True}},
    "function": {
        "required": ("function",),
        "optional": {
            "amplitude": 1.0,
            "phase_multiplier": 1.0,
            "phase_offset": 0.0,
            "value_offset": 0.0,
            "additive": True,
        },
    },
    "stepped": {"required": ("step",), "optional": {"offset": 0.0}},
    "cycles": {
        "required": (),
        "optional": {"before": "repeat", "after": "repeat", "count_before": 0, "count_after": 0},
    },
    "time_offset": {"required": ("offset",), "optional": {}},
    "time_scale": {"required": ("factor",), "optional": {"pivot": 0.0}},
}
TIME_TYPES = ("stepped", "time_offset", "time_scale", "cycles")
FUNCTIONS = ("sin", "cos", "sinc", "sqrt", "ln")
CYCLE_MODES = ("repeat", "repeat_offset", "mirror", "none")
MAX_DEGREE = 10
MAX_CYCLES = 1_000_000
COMMON = ("type", "influence", "range", "blend_in", "blend_out", "mute")


def _err(message: str, location: str, code: str = "INVALID_MODIFIER") -> AnimationError:
    return AnimationError(code, message, location)


def _num(
    value: Any,
    location: str,
    *,
    lo: float = -math.inf,
    hi: float = math.inf,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err(f"must be a number, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v):
        raise _err(f"must be finite, got {value!r}", location)
    if positive and v <= 0.0:
        raise _err(f"must be greater than 0, got {value!r}", location)
    if not lo <= v <= hi:
        raise _err(f"must be within {lo:g}..{hi:g}, got {value!r}", location)
    return v


@dataclass(frozen=True)
class Modifier:
    """One modifier (immutable). ``params`` holds the type's parameters, checked."""

    type: str
    params: tuple[tuple[str, Any], ...]
    influence: float = 1.0
    range: tuple[float, float] | None = None
    blend_in: float = 0.0
    blend_out: float = 0.0
    mute: bool = False
    _extra: Any = field(default=None, compare=False, repr=False)

    def p(self, name: str) -> Any:
        for k, v in self.params:
            if k == name:
                return v
        raise KeyError(name)  # pragma: no cover

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for k, v in self.params:
            if v is not None:
                out[k] = [list(p) for p in v] if k == "points" else v
        if self.influence != 1.0:
            out["influence"] = self.influence
        if self.range is not None:
            out["range"] = list(self.range)
            if self.blend_in:
                out["blend_in"] = self.blend_in
            if self.blend_out:
                out["blend_out"] = self.blend_out
        if self.mute:
            out["mute"] = True
        return out


def parse_modifier(raw: Any, location: str) -> Modifier:
    if isinstance(raw, Modifier):
        return raw
    if not isinstance(raw, dict):
        raise _err("a modifier is an object with a type", location)
    kind = raw.get("type")
    if kind not in TYPES:
        raise _err(f"unknown modifier type {kind!r}; one of {', '.join(TYPES)}", f"{location}.type")
    spec = TYPES[kind]
    allowed = set(COMMON) | set(spec["required"]) | set(spec["optional"])
    unknown = set(raw) - allowed
    if unknown:
        raise _err(f"unknown {kind} parameters {sorted(unknown)}", location)
    missing = [k for k in spec["required"] if k not in raw]
    if missing:
        raise _err(f"{kind} needs {', '.join(missing)}", location)
    values: dict[str, Any] = dict(spec["optional"])
    for k in spec["required"]:
        values[k] = raw[k]
    for k in spec["optional"]:
        if k in raw:
            values[k] = raw[k]
    loc = location
    # Type-specific checks.
    if kind == "offset":
        values["value"] = _num(values["value"], f"{loc}.value")
    elif kind == "scale":
        values["factor"] = _num(values["factor"], f"{loc}.factor")
        values["pivot"] = _num(values["pivot"], f"{loc}.pivot")
    elif kind == "limits":
        for k in ("min", "max"):
            if values[k] is not None:
                values[k] = _num(values[k], f"{loc}.{k}")
        if values["min"] is None and values["max"] is None:
            raise _err("limits needs min or max", loc)
        if (
            values["min"] is not None
            and values["max"] is not None
            and values["min"] > values["max"]
        ):
            raise _err("min must not exceed max", loc)
    elif kind == "noise":
        values["strength"] = _num(values["strength"], f"{loc}.strength")
        values["scale"] = _num(values["scale"], f"{loc}.scale", positive=True)
        values["offset"] = _num(values["offset"], f"{loc}.offset")
        octaves = values["octaves"]
        if (
            isinstance(octaves, bool)
            or not isinstance(octaves, int)
            or not 1 <= octaves <= MAX_OCTAVES
        ):
            raise _err(f"octaves must be an integer 1..{MAX_OCTAVES}", f"{loc}.octaves")
        values["roughness"] = _num(values["roughness"], f"{loc}.roughness", lo=0.0, hi=1.0)
        seed = values["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise _err("seed must be an integer 0..2**32-1", f"{loc}.seed")
    elif kind == "wave":
        values["amplitude"] = _num(values["amplitude"], f"{loc}.amplitude")
        values["period"] = _num(values["period"], f"{loc}.period", positive=True)
        values["phase"] = _num(values["phase"], f"{loc}.phase")
        values["decay"] = _num(values["decay"], f"{loc}.decay", lo=0.0)
        values["start"] = _num(values["start"], f"{loc}.start")
    elif kind == "envelope":
        values["reference"] = _num(values["reference"], f"{loc}.reference")
        pts = values["points"]
        if not isinstance(pts, list) or not 1 <= len(pts) <= MAX_ENVELOPE_POINTS:
            raise _err(
                f"points must be a list of 1..{MAX_ENVELOPE_POINTS} [time, gain, offset]",
                f"{loc}.points",
            )
        clean = []
        for i, p in enumerate(pts):
            if not isinstance(p, (list, tuple)) or len(p) != 3:
                raise _err("a point is [time, gain, offset]", f"{loc}.points[{i}]")
            clean.append(tuple(_num(x, f"{loc}.points[{i}][{j}]") for j, x in enumerate(p)))
        if any(clean[i][0] >= clean[i + 1][0] for i in range(len(clean) - 1)):
            raise _err("point times must increase", f"{loc}.points")
        values["points"] = tuple(clean)
    elif kind == "smooth":
        values["sigma"] = _num(values["sigma"], f"{loc}.sigma", positive=True)
    elif kind == "stepped":
        values["step"] = _num(values["step"], f"{loc}.step", positive=True)
        values["offset"] = _num(values["offset"], f"{loc}.offset")
    elif kind == "generator":
        coef = values["coefficients"]
        if not isinstance(coef, (list, tuple)) or not 1 <= len(coef) <= MAX_DEGREE + 1:
            raise _err(
                f"coefficients must be a list of 1..{MAX_DEGREE + 1} numbers (c0 first)",
                f"{loc}.coefficients",
            )
        values["coefficients"] = tuple(
            _num(c, f"{loc}.coefficients[{i}]") for i, c in enumerate(coef)
        )
        if not isinstance(values["additive"], bool):
            raise _err("additive must be true or false", f"{loc}.additive")
    elif kind == "function":
        if values["function"] not in FUNCTIONS:
            raise _err(f"function must be one of {', '.join(FUNCTIONS)}", f"{loc}.function")
        for k in ("amplitude", "phase_multiplier", "phase_offset", "value_offset"):
            values[k] = _num(values[k], f"{loc}.{k}")
        if not isinstance(values["additive"], bool):
            raise _err("additive must be true or false", f"{loc}.additive")
    elif kind == "cycles":
        for k in ("before", "after"):
            if values[k] not in CYCLE_MODES:
                raise _err(f"{k} must be one of {', '.join(CYCLE_MODES)}", f"{loc}.{k}")
        for k in ("count_before", "count_after"):
            c = values[k]
            if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= MAX_CYCLES:
                raise _err(f"{k} must be an integer 0..{MAX_CYCLES} (0: no end)", f"{loc}.{k}")
    elif kind == "time_offset":
        values["offset"] = _num(values["offset"], f"{loc}.offset")
    elif kind == "time_scale":
        values["factor"] = _num(values["factor"], f"{loc}.factor")
        values["pivot"] = _num(values["pivot"], f"{loc}.pivot")
    influence = _num(raw.get("influence", 1.0), f"{loc}.influence", lo=0.0, hi=1.0)
    rng = raw.get("range")
    blend_in = blend_out = 0.0
    if rng is not None:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise _err("range is [start, end]", f"{loc}.range")
        a, b = (_num(x, f"{loc}.range[{i}]") for i, x in enumerate(rng))
        if not a < b:
            raise _err("range start must be before its end", f"{loc}.range")
        blend_in = _num(raw.get("blend_in", 0.0), f"{loc}.blend_in", lo=0.0)
        blend_out = _num(raw.get("blend_out", 0.0), f"{loc}.blend_out", lo=0.0)
        if blend_in + blend_out > b - a:
            raise _err("blend_in + blend_out exceed the range", loc)
        rng = (a, b)
    elif "blend_in" in raw or "blend_out" in raw:
        raise _err("blend_in/blend_out need a range", loc)
    mute = raw.get("mute", False)
    if not isinstance(mute, bool):
        raise _err("mute must be true or false", f"{loc}.mute")
    params = tuple((k, values[k]) for k in (*spec["required"], *spec["optional"]))
    return Modifier(kind, params, influence, rng, blend_in, blend_out, mute)


def parse_modifiers(raw: Any, location: str) -> tuple[Modifier, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise _err("modifiers must be a list", location)
    if len(raw) > MAX_MODIFIERS:
        raise _err(f"at most {MAX_MODIFIERS} modifiers", location, "LIMIT_EXCEEDED")
    mods = tuple(parse_modifier(m, f"{location}[{i}]") for i, m in enumerate(raw))
    if sum(1 for m in mods if m.type == "smooth" and not m.mute) > MAX_SMOOTH:
        raise _err(f"at most {MAX_SMOOTH} smooth modifiers per curve", location, "LIMIT_EXCEEDED")
    return mods


# --------------------------------------------------------------------------------------- #
# Noise (deterministic 1D gradient noise with analytic derivatives)
# --------------------------------------------------------------------------------------- #

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _hash_gradient(i: np.ndarray, seed: int) -> np.ndarray:
    """A gradient in [-1, 1) for integer lattice points ``i`` (splitmix64 finalizer)."""
    with np.errstate(over="ignore"):
        z = i.astype(np.int64).view(np.uint64) + np.uint64(seed) * _GOLDEN + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (2.0 / 2.0**53) - 1.0


def gradient_noise(x: np.ndarray, seed: int, order: int) -> list[np.ndarray]:
    """``[n, n', n'', n''']`` of 1D gradient noise (quintic fade, about -1..1)."""
    i = np.floor(x)
    f = x - i
    g0 = _hash_gradient(i, seed)
    g1 = _hash_gradient(i + 1.0, seed)
    fade = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    d = g1 * (f - 1.0) - g0 * f  # b - a
    c = g1 - g0  # (b - a)'
    out = [2.0 * (g0 * f + fade * d)]
    if order >= 1:
        f1 = 30.0 * f * f * (f - 1.0) * (f - 1.0)
        out.append(2.0 * (g0 + f1 * d + fade * c))
    if order >= 2:
        f2 = 60.0 * f * (2.0 * f * f - 3.0 * f + 1.0)
        out.append(2.0 * (f2 * d + 2.0 * f1 * c))
    if order >= 3:
        f3 = 60.0 * (6.0 * f * f - 6.0 * f + 1.0)
        out.append(2.0 * (f3 * d + 3.0 * f2 * c))
    return out


def _fractal(x: np.ndarray, m: Modifier, order: int) -> list[np.ndarray]:
    octaves = m.p("octaves")
    rough = m.p("roughness")
    seed = m.p("seed")
    total = [np.zeros_like(x) for _ in range(order + 1)]
    norm = 0.0
    amp = 1.0
    freq = 1.0
    for o in range(octaves):
        n = gradient_noise(x * freq + 37.17 * o, seed + o, order)
        for k in range(order + 1):
            total[k] = total[k] + amp * n[k] * freq**k
        norm += amp
        amp *= rough
        freq *= 2.0
    return [v / norm for v in total]


# --------------------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------------------- #


def _weight(m: Modifier, t: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray] | None:
    """``(w, w')`` of the influence and range ramps (``w'' = 0`` between kinks), or
    ``None`` when the modifier applies fully everywhere."""
    if m.range is None and m.influence == 1.0:
        return None
    w = np.full_like(t, m.influence)
    dw = np.zeros_like(t)
    if m.range is not None:
        a, b = m.range
        ramp = np.where((t >= a) & (t <= b), 1.0, 0.0)
        dramp = np.zeros_like(t)
        if m.blend_in > 0.0:
            inside = (t >= a) & (t < a + m.blend_in)
            ramp = np.where(inside, (t - a) / m.blend_in, ramp)
            dramp = np.where(inside, 1.0 / m.blend_in, dramp)
        if m.blend_out > 0.0:
            inside = (t > b - m.blend_out) & (t <= b)
            ramp = np.where(inside, (b - t) / m.blend_out, ramp)
            dramp = np.where(inside, -1.0 / m.blend_out, dramp)
        w = w * ramp
        dw = m.influence * dramp
    return w, dw


def _blend(
    base: list[np.ndarray], mod: list[np.ndarray], wd: tuple[np.ndarray, np.ndarray] | None
) -> list[np.ndarray]:
    if wd is None:
        return mod
    w, dw = wd
    out = [base[0] + w * (mod[0] - base[0])]
    if len(base) > 1:
        out.append((1.0 - w) * base[1] + w * mod[1] + dw * (mod[0] - base[0]))
    if len(base) > 2:
        out.append((1.0 - w) * base[2] + w * mod[2] + 2.0 * dw * (mod[1] - base[1]))
    if len(base) > 3:
        out.append((1.0 - w) * base[3] + w * mod[3] + 3.0 * dw * (mod[2] - base[2]))
    return out


def _piecewise_linear(points: tuple, column: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([p[0] for p in points])
    ys = np.array([p[column] for p in points])
    val = np.interp(t, xs, ys)
    if len(points) == 1:
        return val, np.zeros_like(t)
    k = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 2)
    slope = (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k])
    inside = (t >= xs[0]) & (t < xs[-1])
    return val, np.where(inside, slope, 0.0)


def _apply_value(m: Modifier, v: list[np.ndarray], t: np.ndarray, order: int) -> list[np.ndarray]:
    kind = m.type
    if kind == "offset":
        return [v[0] + m.p("value"), *v[1:]]
    if kind == "scale":
        k = m.p("factor")
        pivot = m.p("pivot")
        return [pivot + (v[0] - pivot) * k, *(x * k for x in v[1:])]
    if kind == "limits":
        lo, hi = m.p("min"), m.p("max")
        val = v[0]
        clamped = np.zeros(val.shape, bool)
        if lo is not None:
            clamped |= val < lo
            val = np.maximum(val, lo)
        if hi is not None:
            clamped |= val > hi
            val = np.minimum(val, hi)
        return [val, *(np.where(clamped, 0.0, x) for x in v[1:])]
    if kind == "noise":
        scale = m.p("scale")
        n = _fractal((t + m.p("offset")) / scale, m, order)
        s = m.p("strength")
        return [v[k] + s * n[k] / scale**k for k in range(order + 1)]
    if kind == "wave":
        amp, period, phase = m.p("amplitude"), m.p("period"), m.p("phase")
        decay, start = m.p("decay"), m.p("start")
        omega = 2.0 * math.pi / period
        x = omega * (t - phase)
        s, c = np.sin(x), np.cos(x)
        live = t > start
        lam = np.where(live, decay, 0.0)
        env = np.exp(-lam * np.where(live, t - start, 0.0))
        # w(t) = amp * env * sin(x): derivatives of e^{-lam tau} sin(omega tau + .)
        alpha, beta = np.ones_like(t), np.zeros_like(t)  # coefficient of (sin, cos)
        out = [v[0] + amp * env * (alpha * s + beta * c)]
        for k in range(1, order + 1):
            alpha, beta = -lam * alpha - omega * beta, -lam * beta + omega * alpha
            out.append(v[k] + amp * env * (alpha * s + beta * c))
        return out
    if kind == "generator":
        coef = m.p("coefficients")
        # Horner's rule for the polynomial and its derivatives (a fixed order of operations).
        poly = [np.zeros_like(t) for _ in range(order + 1)]
        for c in reversed(coef):
            for k in range(order, 0, -1):
                poly[k] = poly[k] * t + k * poly[k - 1]
            poly[0] = poly[0] * t + c
        if m.p("additive"):
            return [v[k] + poly[k] for k in range(order + 1)]
        return poly
    if kind == "function":
        f = _function(m.p("function"), m.p("phase_multiplier") * t + m.p("phase_offset"), order)
        amp, mult = m.p("amplitude"), m.p("phase_multiplier")
        out = [amp * f[0] + m.p("value_offset")]
        scale = amp
        for k in range(1, order + 1):
            scale = scale * mult
            out.append(scale * f[k])
        if m.p("additive"):
            return [v[k] + out[k] for k in range(order + 1)]
        return out
    if kind == "envelope":
        ref = m.p("reference")
        g, dg = _piecewise_linear(m.p("points"), 1, t)
        o, do = _piecewise_linear(m.p("points"), 2, t)
        out = [ref + (v[0] - ref) * g + o]
        if order >= 1:
            out.append(dg * (v[0] - ref) + g * v[1] + do)
        if order >= 2:
            out.append(2.0 * dg * v[1] + g * v[2])
        if order >= 3:
            out.append(3.0 * dg * v[2] + g * v[3])
        return out
    raise AssertionError(kind)  # pragma: no cover


def _function(name: str, x: np.ndarray, order: int) -> list[np.ndarray]:
    """A built-in function and its first ``order`` derivatives (0 outside the domains of
    ``sqrt`` and ``ln``)."""
    if name in ("sin", "cos"):
        s, c = np.sin(x), np.cos(x)
        seq = [s, c, -s, -c] if name == "sin" else [c, -s, -c, s]
        return seq[: order + 1]
    if name == "sinc":
        s, c = np.sin(x), np.cos(x)
        small = np.abs(x) < 1e-3
        xs = np.where(small, 1.0, x)
        x2 = x * x
        # sin(x)/x and its derivatives; a Taylor series near 0 (no cancellation).
        f = [np.where(small, 1.0 - x2 / 6.0 + x2 * x2 / 120.0, s / xs)]
        if order >= 1:
            f.append(np.where(small, -x / 3.0 + x * x2 / 30.0, (x * c - s) / (xs * xs)))
        if order >= 2:
            third = (-x2 * s - 2.0 * x * c + 2.0 * s) / (xs * xs * xs)
            f.append(np.where(small, -1.0 / 3.0 + x2 / 10.0, third))
        if order >= 3:
            f.append(
                np.where(
                    small,
                    x / 5.0,
                    (-x2 * x * c + 3.0 * x2 * s + 6.0 * x * c - 6.0 * s) / (xs * xs * xs * xs),
                )
            )
        return f
    live = x > 0.0
    xp = np.where(live, x, 1.0)
    if name == "sqrt":
        r = np.sqrt(xp)
        f = [np.where(live, r, 0.0)]
        terms = [0.5 / r, -0.25 / (r * xp), 0.375 / (r * xp * xp)]
    else:  # ln
        f = [np.where(live, np.log(xp), 0.0)]
        terms = [1.0 / xp, -1.0 / (xp * xp), 2.0 / (xp * xp * xp)]
    f.extend(np.where(live, terms[k], 0.0) for k in range(order))
    return f


def _cycle_map(m: Modifier, curve: FCurve, t: np.ndarray):
    """Time inside the curve's key range, the value offset and the time direction of the
    cycles modifier ``m`` at ``t`` (``None``: the keys span no time)."""
    t0, t1 = curve.range
    period = t1 - t0
    if not period > 0.0:
        return None
    delta = float(curve.values[-1] - curve.values[0])
    local = t.copy()
    offset = np.zeros_like(t)
    sign = np.ones_like(t)
    for side, mode, count in (
        ("before", m.p("before"), m.p("count_before")),
        ("after", m.p("after"), m.p("count_after")),
    ):
        if mode == "none":
            continue
        sel = t > t1 if side == "after" else t < t0
        if not np.any(sel):
            continue
        beyond = t[sel] - t1 if side == "after" else t0 - t[sel]
        cycle = np.ceil(beyond / period)  # 1 for the first repetition (its end included)
        u = beyond - (cycle - 1.0) * period  # (0, period] into the repetition
        mirrored = (cycle % 2.0) == 1.0 if mode == "mirror" else np.zeros_like(cycle, bool)
        if side == "after":
            inside = np.where(mirrored, t1 - u, t0 + u)
            shift = cycle * delta
        else:
            inside = np.where(mirrored, t0 + u, t1 - u)
            shift = -cycle * delta
        keep = np.ones_like(cycle, bool) if count == 0 else cycle <= count
        local[sel] = np.where(keep, inside, t[sel])
        if mode == "repeat_offset":
            offset[sel] = np.where(keep, shift, 0.0)
        sign[sel] = np.where(keep & mirrored, -1.0, 1.0)
    return local, offset, sign


def cyclic_handles(modifiers: tuple) -> tuple[int, int] | None:
    """The extrapolation codes a leading endless ``cycles`` modifier gives the automatic
    handles (Blender's rule), or ``None``."""
    if not modifiers:
        return None
    m = modifiers[0]
    if (
        m.type != "cycles"
        or m.mute
        or m.range is not None
        or m.influence != 1.0
        or m.p("count_before")
        or m.p("count_after")
        or m.p("before") not in ("repeat", "repeat_offset")
        or m.p("after") not in ("repeat", "repeat_offset")
    ):
        return None
    code = {"repeat": 2, "repeat_offset": 3}
    return code[m.p("before")], code[m.p("after")]


_SMOOTH_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _smooth_nodes(k: int = SMOOTH_NODES) -> tuple[np.ndarray, np.ndarray]:
    found = _SMOOTH_CACHE.get(k)
    if found is None:
        x = np.arange(-k, k + 1, dtype=np.float64) * (3.0 / k)  # in sigmas
        w = np.exp(-0.5 * x * x)
        found = _SMOOTH_CACHE[k] = (x, w / w.sum())
    return found


def evaluate_with_modifiers(curve: FCurve, t: np.ndarray, order: int) -> list[np.ndarray]:
    """The curve's value and derivatives at ``t`` with its modifier chain applied."""
    mods = curve.modifiers

    def chain(level: int, tt: np.ndarray) -> list[np.ndarray]:
        if level == 0:
            return curve.evaluate_base(tt, order)
        m = mods[level - 1]
        if m.mute:
            return chain(level - 1, tt)
        wd = _weight(m, tt, order)
        if m.type in TIME_TYPES:
            if m.type == "stepped":
                step, off = m.p("step"), m.p("offset")
                h = off + np.floor((tt - off) / step) * step
                inner = chain(level - 1, h)
                mod = [inner[0], *(np.zeros_like(tt) for _ in range(order))]
            elif m.type == "cycles":
                mapped = _cycle_map(m, curve, tt)
                if mapped is None:
                    mod = chain(level - 1, tt)
                else:
                    h, shift, sign = mapped
                    inner = chain(level - 1, h)
                    mod = [inner[0] + shift]
                    mod.extend(inner[j] * (sign if j % 2 else 1.0) for j in range(1, order + 1))
            else:
                if m.type == "time_offset":
                    h, k = tt - m.p("offset"), 1.0
                else:
                    pivot, k = m.p("pivot"), m.p("factor")
                    h = pivot + (tt - pivot) * k
                inner = chain(level - 1, h)
                mod = [inner[0], *(inner[j] * k**j for j in range(1, order + 1))]
            base = chain(level - 1, tt) if wd is not None else mod
            return _blend(base, mod, wd)
        if m.type == "smooth":
            x, w = _smooth_nodes()
            sigma = m.p("sigma")
            shifted = (tt[None, :] + (x * sigma)[:, None]).reshape(-1)
            inner = chain(level - 1, shifted)
            n = tt.shape[0]
            # A fixed-order sum per element (never BLAS: its blocking could make a time's
            # result depend on the other times evaluated with it).
            mod = []
            for comp in inner:
                rows = comp.reshape(len(x), n)
                acc = w[0] * rows[0]
                for j in range(1, len(x)):
                    acc = acc + w[j] * rows[j]
                mod.append(acc)
            base = chain(level - 1, tt) if wd is not None else mod
            return _blend(base, mod, wd)
        base = chain(level - 1, tt)
        mod = _apply_value(m, base, tt, order)
        return _blend(base, mod, wd)

    return chain(len(mods), t)


Factory = Callable[..., Modifier]
