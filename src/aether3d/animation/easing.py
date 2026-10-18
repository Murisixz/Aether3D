"""Easing functions for keyframe segments, with exact derivatives.

An eased segment from ``(t0, v0)`` to ``(t1, v1)`` takes the value
``v0 + (v1 - v0) * E(u)``, ``u = (t - t0) / (t1 - t0)``. ``E`` is built from a base
*ease-in* function ``f`` (``f(0) = 0``, ``f(1) = 1``) and a mode:

* ``in``      ``E(u) = f(u)`` (slow start),
* ``out``     ``E(u) = 1 - f(1 - u)`` (slow end),
* ``in_out``  ``E(u) = f(2u) / 2`` for ``u < 1/2``, else ``1 - f(2 - 2u) / 2``.

The base functions are Robert Penner's easing equations (BSD licence), the set animation
tools share: ``sine``, ``quad``, ``cubic``, ``quart``, ``quint``, ``expo``, ``circ``,
``back`` (overshoot ``s``, 1.70158 by default: about 10 %), ``elastic`` (amplitude ``a >=
1`` and period ``p`` in segment units) and ``bounce``. Two are made exact at both ends:
``expo`` is the normalized ``(2^(10u) - 1) / 1023`` (Penner's ``2^(10(u - 1))`` misses 0
by 2^-10), and ``elastic`` subtracts Penner's residual at ``u = 0`` linearly, so every
eased segment passes through both of its keys exactly.

First, second and third derivatives (``E'``, ``E''``, ``E'''`` with respect to ``u``) are
closed forms, so velocity, acceleration and jerk of eased motion are exact. ``circ`` has
an infinite slope at its steep end (a deliberate vertical tangent), and ``bounce`` has
velocity jumps at its impacts (deliberate); the curve reports say so.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import AnimationError

EASINGS = ("sine", "quad", "cubic", "quart", "quint", "expo", "circ", "back", "elastic", "bounce")
MODES = ("in", "out", "in_out")
DEFAULT_BACK = 1.70158
DEFAULT_AMPLITUDE = 1.0
DEFAULT_PERIOD = 0.3

_LN2_10 = 10.0 * math.log(2.0)
_EXPO_SCALE = 1.0 / 1023.0
_HALF_PI = 0.5 * math.pi


def easing_index(name: str, location: str = "easing") -> int:
    if name not in EASINGS:
        raise AnimationError(
            "INVALID_EASING",
            f"unknown easing {name!r}",
            location,
            hint=f"one of {', '.join(EASINGS)}",
        )
    return EASINGS.index(name)


def mode_index(name: str, location: str = "ease") -> int:
    if name not in MODES:
        raise AnimationError(
            "INVALID_EASING", f"unknown easing mode {name!r}", location, hint="in, out or in_out"
        )
    return MODES.index(name)


def _elastic(u: np.ndarray, a: np.ndarray, p: np.ndarray, order: int) -> list[np.ndarray]:
    """Penner's elastic ease-in and derivatives, residual at ``u = 0`` removed."""
    a = np.maximum(a, 1.0)
    omega = 2.0 * math.pi / p
    s = np.arcsin(1.0 / a) / omega
    out = []
    for uu, want in ((u, order), (np.zeros_like(u), 0)):
        amp = a * np.exp(_LN2_10 * (uu - 1.0))
        phi = (uu - 1.0 - s) * omega
        sin, cos = np.sin(phi), np.cos(phi)
        alpha, beta = -np.ones_like(uu), np.zeros_like(uu)  # f = amp (alpha sin + beta cos)
        values = [amp * (alpha * sin + beta * cos)]
        for _ in range(want):
            alpha, beta = _LN2_10 * alpha - omega * beta, _LN2_10 * beta + omega * alpha
            values.append(amp * (alpha * sin + beta * cos))
        out.append(values)
    values, at_zero = out
    residual = at_zero[0]
    values[0] = values[0] - (1.0 - u) * residual
    if order >= 1:
        values[1] = values[1] + residual
    return values


_B = 7.5625


def _bounce_out(u: np.ndarray, order: int) -> list[np.ndarray]:
    """Penner's bounce ease-out (four parabolas) and derivatives."""
    c = np.select(
        [u < 1.0 / 2.75, u < 2.0 / 2.75, u < 2.5 / 2.75],
        [0.0, 1.5 / 2.75, 2.25 / 2.75],
        2.625 / 2.75,
    )
    k = np.select([u < 1.0 / 2.75, u < 2.0 / 2.75, u < 2.5 / 2.75], [0.0, 0.75, 0.9375], 0.984375)
    d = u - c
    values = [_B * d * d + k]
    if order >= 1:
        values.append(2.0 * _B * d)
    if order >= 2:
        values.append(np.full_like(u, 2.0 * _B))
    if order >= 3:
        values.append(np.zeros_like(u))
    values[0] = np.where(u >= 1.0, 1.0, np.where(u <= 0.0, 0.0, values[0]))
    return values


def base_in(
    kind: int,
    u: np.ndarray,
    order: int,
    back: np.ndarray | None = None,
    amplitude: np.ndarray | None = None,
    period: np.ndarray | None = None,
) -> list[np.ndarray]:
    """``[f, f', f'', f''']`` (up to ``order``) of the ease-in function ``kind`` at ``u``."""
    name = EASINGS[kind]
    zero = np.zeros_like(u)
    if name == "sine":
        x = _HALF_PI * u
        s, c = np.sin(x), np.cos(x)
        vals = [1.0 - c, _HALF_PI * s, _HALF_PI**2 * c, -(_HALF_PI**3) * s]
    elif name in ("quad", "cubic", "quart", "quint"):
        n = {"quad": 2, "cubic": 3, "quart": 4, "quint": 5}[name]
        vals = []
        coef = 1.0
        for k in range(4):
            if n - k < 0:
                vals.append(zero)
            else:
                vals.append(coef * u ** (n - k))
            coef *= n - k
    elif name == "expo":
        e = np.exp(_LN2_10 * u) * _EXPO_SCALE
        vals = [e - _EXPO_SCALE, _LN2_10 * e, _LN2_10**2 * e, _LN2_10**3 * e]
    elif name == "circ":
        w = np.maximum(1.0 - u * u, 0.0)
        r = np.sqrt(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = [
                1.0 - r,
                np.where(w > 0.0, u / r, np.inf),
                np.where(w > 0.0, 1.0 / (w * r), np.inf),
                np.where(w > 0.0, 3.0 * u / (w * w * r), np.inf),
            ]
    elif name == "back":
        s = np.full_like(u, DEFAULT_BACK) if back is None else back
        vals = [
            (s + 1.0) * u * u * u - s * u * u,
            3.0 * (s + 1.0) * u * u - 2.0 * s * u,
            6.0 * (s + 1.0) * u - 2.0 * s,
            6.0 * (s + 1.0) + zero,
        ]
    elif name == "elastic":
        a = np.full_like(u, DEFAULT_AMPLITUDE) if amplitude is None else amplitude
        p = np.full_like(u, DEFAULT_PERIOD) if period is None else period
        return _elastic(u, a, p, order)
    else:  # bounce: ease-in = 1 - out(1 - u)
        out = _bounce_out(1.0 - u, order)
        vals = [1.0 - out[0]]
        sign = 1.0
        for k in range(1, order + 1):
            vals.append(sign * out[k])
            sign = -sign
        return vals
    return vals[: order + 1]


def evaluate(
    kind: int,
    mode: int,
    u: np.ndarray,
    order: int = 0,
    back: np.ndarray | None = None,
    amplitude: np.ndarray | None = None,
    period: np.ndarray | None = None,
) -> list[np.ndarray]:
    """``[E, E', E'', E''']`` (up to ``order``) of easing ``kind`` in ``mode`` at ``u``
    (``0 <= u <= 1``); derivatives are with respect to ``u``."""
    u = np.asarray(u, dtype=np.float64)
    m = MODES[mode]
    if m == "in":
        return base_in(kind, u, order, back, amplitude, period)
    if m == "out":
        f = base_in(kind, 1.0 - u, order, back, amplitude, period)
        out = [1.0 - f[0]]
        sign = 1.0
        for k in range(1, order + 1):
            out.append(sign * f[k])
            sign = -sign
        return out
    first = u < 0.5
    lo = base_in(kind, np.where(first, 2.0 * u, 0.0), order, back, amplitude, period)
    hi = base_in(kind, np.where(first, 1.0, 2.0 - 2.0 * u), order, back, amplitude, period)
    out = [np.where(first, 0.5 * lo[0], 1.0 - 0.5 * hi[0])]
    scale = 1.0
    sign = 1.0
    for k in range(1, order + 1):
        out.append(np.where(first, scale * lo[k], sign * scale * hi[k]))
        scale *= 2.0
        sign = -sign
    return out
