"""Rotation tracks: quaternion keyframes with rotation-aware interpolation and timing.

A :class:`RotationTrack` animates an orientation. It is not four independent F-curves:
interpolating quaternion components and renormalizing changes the angular speed along
the way and can take the long way round. Each key holds a unit quaternion (its sign kept
as given), and segment ``k`` (key ``k`` to ``k + 1``) uses key ``k``'s settings:

* ``interpolation``: ``constant`` (hold), ``slerp`` (constant angular speed along the
  great arc), ``squad`` (Shoemake's spherical cubic through the neighbouring keys:
  continuous angular velocity at keys), or ``cubic`` (glTF's CUBICSPLINE: component-wise
  Hermite with the keys' ``in``/``out`` tangents, normalized -- what glTF files carry);
* ``path``: ``shortest`` (default: through ``-q_{k+1}`` when the dot product is negative)
  or ``stored`` (the quaternions as given: a deliberate long rotation, up to 360 degrees);
* ``timing``: how time advances through the arc -- ``linear`` (default), an easing
  (``{"easing": "sine", "ease": "in_out"}``, see :mod:`.easing`) or a timing Bezier
  ``{"bezier": [x1, y1, x2, y2]}`` (CSS-style ``cubic-bezier``; ``0 <= x1, x2 <= 1``,
  ``y`` free: values outside ``[0, 1]`` overshoot along the arc).

Angular velocity (radians per second, parent space) and angular acceleration are exact
for ``constant`` and ``slerp`` (``theta * axis * u'(t)``, ``theta * axis * u''(t)``); for
``squad`` and ``cubic`` they are computed by fourth-order central differences of the
evaluated orientation (``numerical`` in reports). Extrapolation: ``constant``, ``cycle``,
``cycle_offset`` (each cycle composed with the rotation from the first to the last key:
a turning walk keeps turning), ``mirror``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import easing as _easing
from . import rotation as R

INTERPOLATIONS = ("constant", "slerp", "squad", "cubic")
CONSTANT, SLERP, SQUAD, CUBIC = range(4)
PATHS = ("shortest", "stored")
EXTRAPOLATIONS = ("constant", "cycle", "cycle_offset", "mirror")
TIMING_LINEAR, TIMING_EASING, TIMING_BEZIER = range(3)
MAX_KEYS = 1_000_000
UNIT_TOLERANCE = 1e-6


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _index(table, value, location, code):
    if isinstance(value, str) and value in table:
        return table.index(value)
    if (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and 0 <= value < len(table)
    ):
        return int(value)
    raise _err(code, f"{value!r} is not one of {', '.join(table)}", location)


def _quat(value: Any, location: str) -> np.ndarray:
    try:
        q = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err(
            "INVALID_ROTATION", f"a rotation is [x, y, z, w], got {value!r}", location
        ) from None
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise _err(
            "INVALID_ROTATION", f"a rotation is four finite numbers, got {value!r}", location
        )
    n = float(np.sqrt(q @ q))
    if n < 1e-9:
        raise _err("INVALID_ROTATION", "a rotation quaternion must not be zero", location)
    return q / n if abs(n - 1.0) > 1e-15 else q


def _timing(raw: Any, location: str) -> tuple[int, tuple[float, ...]]:
    """``(kind, params)``: linear; easing (index, mode, back, amplitude, period); bezier."""
    if raw is None or raw == "linear":
        return TIMING_LINEAR, (0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(raw, str):
        raw = {"easing": raw}
    if not isinstance(raw, dict):
        raise _err("INVALID_TIMING", "timing is 'linear', an easing or {bezier: [...]}", location)
    if "bezier" in raw:
        if set(raw) != {"bezier"}:
            raise _err("INVALID_TIMING", "a bezier timing has only 'bezier'", location)
        b = raw["bezier"]
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise _err("INVALID_TIMING", "bezier timing is [x1, y1, x2, y2]", location)
        vals = []
        for i, x in enumerate(b):
            if (
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(float(x))
            ):
                raise _err(
                    "INVALID_TIMING",
                    "bezier timing values must be finite numbers",
                    f"{location}.bezier[{i}]",
                )
            vals.append(float(x))
        if not (0.0 <= vals[0] <= 1.0 and 0.0 <= vals[2] <= 1.0):
            raise _err("INVALID_TIMING", "bezier timing x1 and x2 must be within 0..1", location)
        if not (-10.0 <= vals[1] <= 10.0 and -10.0 <= vals[3] <= 10.0):
            raise _err("INVALID_TIMING", "bezier timing y1 and y2 must be within -10..10", location)
        return TIMING_BEZIER, (vals[0], vals[1], vals[2], vals[3], 0.0)
    unknown = set(raw) - {"easing", "ease", "back", "amplitude", "period"}
    if unknown or "easing" not in raw:
        raise _err(
            "INVALID_TIMING",
            "an easing timing is {easing, ease, back, amplitude, period}",
            location,
        )
    e = _easing.easing_index(raw["easing"], f"{location}.easing")
    m = _easing.mode_index(raw.get("ease", "in_out"), f"{location}.ease")
    params = []
    for name, default, lo, hi in (
        ("back", _easing.DEFAULT_BACK, 0.0, 100.0),
        ("amplitude", _easing.DEFAULT_AMPLITUDE, 1.0, 100.0),
        ("period", _easing.DEFAULT_PERIOD, 1e-3, 10.0),
    ):
        v = raw.get(name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
            raise _err(
                "INVALID_TIMING", f"{name} must be within {lo:g}..{hi:g}", f"{location}.{name}"
            )
        params.append(float(v))
    return TIMING_EASING, (float(e), float(m), *params)


def _timing_json(kind: int, p: np.ndarray) -> Any:
    if kind == TIMING_LINEAR:
        return "linear"
    if kind == TIMING_BEZIER:
        return {"bezier": [float(p[0]), float(p[1]), float(p[2]), float(p[3])]}
    out: dict[str, Any] = {"easing": _easing.EASINGS[int(p[0])], "ease": _easing.MODES[int(p[1])]}
    name = out["easing"]
    if name == "back":
        out["back"] = float(p[2])
    if name == "elastic":
        out["amplitude"], out["period"] = float(p[3]), float(p[4])
    return out


def _bezier_timing(p: np.ndarray, s: np.ndarray, order: int) -> list[np.ndarray]:
    """``u(s)`` of a CSS-style timing Bezier (control x within [0, 1]: monotone)."""
    from .curves import solve_bezier_time

    x1, y1, x2, y2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xc = np.stack([3 * x1 - 3 * x2 + 1, -6 * x1 + 3 * x2, 3 * x1, np.zeros_like(x1)], axis=1)
    yc = np.stack([3 * y1 - 3 * y2 + 1, -6 * y1 + 3 * y2, 3 * y1, np.zeros_like(y1)], axis=1)
    w = solve_bezier_time(xc, s, np.ones_like(s))
    a, b, c = yc[:, 0], yc[:, 1], yc[:, 2]
    out = [((a * w + b) * w + c) * w]
    if order >= 1:
        xa, xb, xcc = xc[:, 0], xc[:, 1], xc[:, 2]
        q = (3 * xa * w + 2 * xb) * w + xcc
        p1 = (3 * a * w + 2 * b) * w + c
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(q != 0.0, p1 / q, 0.0))
            if order >= 2:
                qq = 6 * xa * w + 2 * xb
                pp = 6 * a * w + 2 * b
                out.append(np.where(q != 0.0, (pp * q - p1 * qq) / (q * q * q), 0.0))
    return out


def timing_values(
    kind: np.ndarray, params: np.ndarray, s: np.ndarray, order: int
) -> list[np.ndarray]:
    """``u(s)`` and its derivatives with respect to ``s`` for each segment's timing."""
    out = [s.copy()] + [
        np.ones_like(s) if o == 1 else np.zeros_like(s) for o in range(1, order + 1)
    ]
    sel = kind == TIMING_EASING
    if np.any(sel):
        p = params[sel]
        res = [np.empty(int(sel.sum())) for _ in range(order + 1)]
        for e in np.unique(p[:, 0]).astype(int).tolist():
            for m in np.unique(p[p[:, 0] == e, 1]).astype(int).tolist():
                mm = (p[:, 0] == e) & (p[:, 1] == m)
                ev = _easing.evaluate(e, m, s[sel][mm], order, p[mm, 2], p[mm, 3], p[mm, 4])
                for o in range(order + 1):
                    res[o][mm] = ev[o]
        for o in range(order + 1):
            out[o][sel] = res[o]
    sel = kind == TIMING_BEZIER
    if np.any(sel):
        ev = _bezier_timing(params[sel], s[sel], order)
        for o in range(order + 1):
            out[o][sel] = ev[o]
    return out


class RotationTrack:
    """An immutable quaternion keyframe track; see the module docstring."""

    __slots__ = (
        "_times",
        "_quats",
        "_interp",
        "_path",
        "_timing",
        "_timing_params",
        "_in",
        "_out",
        "_before",
        "_after",
        "_squad",
    )

    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("use RotationTrack.from_keys(...)")

    @classmethod
    def from_keys(
        cls,
        keys: Iterable[Any],
        *,
        interpolation: str = "slerp",
        extrapolation: Any = "constant",
        location: str = "track",
    ) -> RotationTrack:
        """``keys``: ``(time, [x, y, z, w])`` pairs or dicts with ``time``, ``rotation``
        (or ``euler``: XYZ degrees), and optionally ``interpolation``, ``path``,
        ``timing``, ``in``/``out`` (cubic tangents, quaternion units per second)."""
        default = _index(
            INTERPOLATIONS, interpolation, f"{location}.interpolation", "INVALID_INTERPOLATION"
        )
        rows = []
        for i, raw in enumerate(keys):
            loc = f"{location}.keys[{i}]"
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                raw = {"time": raw[0], "rotation": raw[1]}
            if not isinstance(raw, dict):
                raise _err("INVALID_KEYS", "a key is (time, rotation) or an object", loc)
            unknown = set(raw) - {
                "time",
                "rotation",
                "euler",
                "interpolation",
                "path",
                "timing",
                "in",
                "out",
            }
            if unknown:
                raise _err("INVALID_KEYS", f"unknown key fields {sorted(unknown)}", loc)
            t = raw.get("time")
            if (
                isinstance(t, bool)
                or not isinstance(t, (int, float))
                or not math.isfinite(float(t))
            ):
                raise _err("INVALID_KEYS", "a key needs a finite time", f"{loc}.time")
            if ("rotation" in raw) == ("euler" in raw):
                raise _err("INVALID_KEYS", "a key needs rotation or euler (not both)", loc)
            if "euler" in raw:
                e = np.asarray(raw["euler"], np.float64).reshape(-1)
                if e.shape != (3,) or not np.all(np.isfinite(e)):
                    raise _err(
                        "INVALID_ROTATION",
                        "euler is three finite angles in degrees",
                        f"{loc}.euler",
                    )
                q = R.from_euler(e)[0]
            else:
                q = _quat(raw["rotation"], f"{loc}.rotation")
            interp = _index(
                INTERPOLATIONS,
                raw.get("interpolation", INTERPOLATIONS[default]),
                f"{loc}.interpolation",
                "INVALID_INTERPOLATION",
            )
            path = _index(PATHS, raw.get("path", "shortest"), f"{loc}.path", "INVALID_KEYS")
            tk, tp = _timing(raw.get("timing"), f"{loc}.timing")
            tin = np.zeros(4) if raw.get("in") is None else _vec4(raw["in"], f"{loc}.in")
            tout = np.zeros(4) if raw.get("out") is None else _vec4(raw["out"], f"{loc}.out")
            rows.append((float(t), q, interp, path, tk, tp, tin, tout))
            if len(rows) > MAX_KEYS:
                raise _err("LIMIT_EXCEEDED", f"more than {MAX_KEYS} keys", f"{location}.keys")
        if not rows:
            raise _err("INVALID_KEYS", "a track needs at least one key", f"{location}.keys")
        rows.sort(key=lambda r: r[0])
        times = np.array([r[0] for r in rows])
        if np.any(np.diff(times) <= 0):
            raise _err(
                "INVALID_KEYS", "rotation key times must increase strictly", f"{location}.keys"
            )
        return cls._make(
            times,
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows], np.uint8),
            np.array([r[3] for r in rows], np.uint8),
            np.array([r[4] for r in rows], np.uint8),
            np.array([r[5] for r in rows]),
            np.array([r[6] for r in rows]),
            np.array([r[7] for r in rows]),
            _extrapolation(extrapolation, f"{location}.extrapolation"),
            location,
        )

    @classmethod
    def from_arrays(
        cls,
        times: Any,
        quats: Any,
        *,
        interpolation: str = "slerp",
        extrapolation: Any = "constant",
        location: str = "track",
    ) -> RotationTrack:
        t = np.asarray(times, np.float64).reshape(-1)
        q = np.asarray(quats, np.float64).reshape(-1, 4)
        n = t.shape[0]
        if n == 0 or q.shape[0] != n:
            raise _err(
                "INVALID_KEYS", "times and quaternions must be equal, non-empty lists", location
            )
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
            raise _err("INVALID_KEYS", "times and quaternions must be finite", location)
        if np.any(np.diff(t) <= 0):
            raise _err("INVALID_KEYS", "rotation key times must increase strictly", location)
        norms = np.linalg.norm(q, axis=1)
        if np.any(norms < 1e-9):
            raise _err("INVALID_ROTATION", "zero quaternion", location)
        q = np.where((np.abs(norms - 1.0) > 1e-15)[:, None], q / norms[:, None], q)
        interp = np.full(
            n,
            _index(
                INTERPOLATIONS, interpolation, f"{location}.interpolation", "INVALID_INTERPOLATION"
            ),
            np.uint8,
        )
        extrap = _extrapolation(extrapolation, f"{location}.extrapolation")
        zeros4 = np.zeros((n, 4))
        return cls._make(
            t, q, interp, np.zeros(n, np.uint8), np.zeros(n, np.uint8), np.zeros((n, 5)),
            zeros4, zeros4.copy(), extrap, location,
        )  # fmt: skip

    @classmethod
    def _make(
        cls, times, quats, interp, path, timing, timing_params, tin, tout, extrap, location
    ) -> RotationTrack:
        self = object.__new__(cls)
        if times.shape[0] > MAX_KEYS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_KEYS} keys", location)
        self._times = _frozen(times)
        self._quats = _frozen(quats)
        self._interp = _frozen(interp)
        self._path = _frozen(path)
        self._timing = _frozen(timing)
        self._timing_params = _frozen(timing_params)
        self._in = _frozen(tin)
        self._out = _frozen(tout)
        self._before, self._after = extrap
        if (self._before or self._after) and not times[-1] > times[0]:
            raise _err(
                "INVALID_EXTRAPOLATION", "cyclic extrapolation needs two keys or more", location
            )
        self._squad = None
        return self

    # -- reading ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._times.shape[0])

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def quats(self) -> np.ndarray:
        return self._quats

    @property
    def range(self) -> tuple[float, float]:
        return float(self._times[0]), float(self._times[-1])

    @property
    def extrapolation(self) -> tuple[str, str]:
        return EXTRAPOLATIONS[self._before], EXTRAPOLATIONS[self._after]

    def key(self, index: int) -> dict[str, Any]:
        i = int(index)
        out: dict[str, Any] = {
            "time": float(self._times[i]),
            "rotation": self._quats[i].tolist(),
            "interpolation": INTERPOLATIONS[self._interp[i]],
        }
        if self._path[i]:
            out["path"] = PATHS[self._path[i]]
        if self._timing[i] != TIMING_LINEAR:
            out["timing"] = _timing_json(int(self._timing[i]), self._timing_params[i])
        if self._interp[i] == CUBIC or (i > 0 and self._interp[i - 1] == CUBIC):
            out["in"] = self._in[i].tolist()
            out["out"] = self._out[i].tolist()
        return out

    def keys(self) -> list[dict[str, Any]]:
        return [self.key(i) for i in range(len(self))]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "times": self._times.copy(),
            "quats": self._quats.copy(),
            "interpolation": self._interp.copy(),
            "path": self._path.copy(),
            "timing": self._timing.copy(),
            "timing_params": self._timing_params.copy(),
            "in": self._in.copy(),
            "out": self._out.copy(),
        }

    def replace(
        self,
        arrays: dict[str, np.ndarray] | None = None,
        *,
        extrapolation: Any = None,
        location: str = "track",
    ) -> RotationTrack:
        a = self.arrays()
        if arrays:
            for k, v in arrays.items():
                if k not in a:
                    raise _err("INVALID_ARGUMENT", f"unknown track field {k!r}", location)
                a[k] = np.array(v, dtype=a[k].dtype)
        order = np.argsort(a["times"], kind="stable")
        a = {k: v[order] for k, v in a.items()}
        t = a["times"]
        if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)):
            raise _err("INVALID_KEYS", "rotation key times must increase strictly", location)
        q = a["quats"]
        norms = np.linalg.norm(q, axis=1)
        if not np.all(np.isfinite(q)) or np.any(norms < 1e-9):
            raise _err("INVALID_ROTATION", "rotations must be finite and non-zero", location)
        q = np.where((np.abs(norms - 1.0) > 1e-15)[:, None], q / norms[:, None], q)
        extrap = (
            (self._before, self._after)
            if extrapolation is None
            else _extrapolation(extrapolation, f"{location}.extrapolation")
        )
        return RotationTrack._make(
            t,
            q,
            a["interpolation"],
            a["path"],
            a["timing"],
            a["timing_params"],
            a["in"],
            a["out"],
            extrap,
            location,
        )

    # -- evaluation ------------------------------------------------------------------------

    def _aligned(self) -> np.ndarray:
        """Keys with each one's sign chosen per the path rule relative to the previous
        aligned key (the effective quaternion sequence)."""
        q = self._quats.copy()
        flip = 1.0
        for k in range(1, q.shape[0]):
            if self._path[k - 1] == 0:  # shortest: the hemisphere of the previous key
                flip = -1.0 if float(q[k - 1] @ self._quats[k]) < 0.0 else 1.0
            # stored: the same sign change as the previous key (the stored arc kept)
            q[k] = self._quats[k] * flip
        return q

    def _controls(self) -> np.ndarray:
        found = self._squad
        if found is None:
            q = self._aligned()
            n = q.shape[0]
            prev = np.concatenate([q[:1], q[:-1]]) if n > 1 else q
            nxt = np.concatenate([q[1:], q[-1:]]) if n > 1 else q
            found = R.squad_controls(prev, q, nxt) if n > 1 else q.copy()
            object.__setattr__(self, "_squad", _frozen(found))
        return found

    def evaluate(self, times: Any, order: int = 0) -> Any:
        """Orientation (``(n, 4)``; one row for a scalar time), with ``order >= 1`` also
        the angular velocity (rad/s, parent space) and ``order >= 2`` the angular
        acceleration: ``(q, omega, alpha)``."""
        if isinstance(order, bool) or order not in (0, 1, 2):
            raise _err("INVALID_ARGUMENT", f"order must be 0..2, got {order!r}", "order")
        scalar = np.ndim(times) == 0
        t = np.asarray(times, np.float64).reshape(-1)
        if not np.all(np.isfinite(t)):
            raise _err("INVALID_TIME", "times must be finite", "times")
        q, w, a = self._evaluate(t, order)
        if scalar:
            q, w, a = q[0], (None if w is None else w[0]), (None if a is None else a[0])
        if order == 0:
            return q
        return (q, w) if order == 1 else (q, w, a)

    def _evaluate(self, t: np.ndarray, order: int):
        times = self._times
        n = times.shape[0]
        first, last = float(times[0]), float(times[-1])
        period = last - first
        tt = t.copy()
        pre = np.zeros((t.shape[0], 4))
        pre[:, 3] = 1.0  # rotation applied before (cycle_offset)
        sign = np.ones_like(t)
        aligned = self._aligned()
        for mask, mode in ((t < first, self._before), (t > last, self._after)):
            if not np.any(mask) or mode == 0:
                continue
            rel = (t[mask] - first) / period
            cyc = np.floor(rel)
            local = np.clip(t[mask] - first - cyc * period, 0.0, period)
            if mode == 3:
                odd = (np.abs(cyc) % 2.0) == 1.0
                tt[mask] = np.where(odd, last - local, first + local)
                sign[mask] = np.where(odd, -1.0, 1.0)
            else:
                tt[mask] = first + local
                if mode == 2:
                    delta = R.multiply(aligned[-1], R.conjugate(aligned[0]))  # first -> last
                    pre[mask] = R.exp(R.log(delta)[None, :] * cyc[:, None])
        tt = np.clip(tt, first, last)
        if n == 1:
            q = np.repeat(self._quats[:1], t.shape[0], axis=0)
            zero = np.zeros((t.shape[0], 3))
            return q, (zero if order >= 1 else None), (zero.copy() if order >= 2 else None)
        k = np.clip(np.searchsorted(times, tt, side="right") - 1, 0, n - 2)
        span = times[k + 1] - times[k]
        s = np.clip((tt - times[k]) / span, 0.0, 1.0)
        kind = self._interp[k]
        tim = timing_values(self._timing[k], self._timing_params[k], s, 2)
        u, du, ddu = tim
        q0 = aligned[k]
        q1 = aligned[k + 1]
        out = np.empty((t.shape[0], 4))
        sel = kind == CONSTANT
        out[sel] = q0[sel]
        sel = kind == SLERP
        if np.any(sel):
            out[sel] = R.slerp(q0[sel], q1[sel], u[sel], shortest=False)
        sel = kind == SQUAD
        if np.any(sel):
            c = self._controls()
            out[sel] = R.squad(q0[sel], q1[sel], c[k[sel]], c[k[sel] + 1], u[sel])
        sel = kind == CUBIC
        if np.any(sel):
            out[sel] = self._cubic(k[sel], u[sel], span[sel])
        # At a key exactly: the key itself (in the track's continuous sign sequence).
        at = tt == times[k]
        out[at] = aligned[k[at]]
        at_last = tt == last
        out[at_last] = aligned[-1]
        out = R.multiply(pre, out)
        omega = alpha = None
        if order >= 1:
            omega = np.zeros((t.shape[0], 3))
            alpha = np.zeros((t.shape[0], 3)) if order >= 2 else None
            sl = kind == SLERP
            if np.any(sl):
                vel = R.slerp_velocity(q0[sl], q1[sl], shortest=False) / span[sl][:, None]
                omega[sl] = vel * du[sl][:, None]
                if alpha is not None:
                    alpha[sl] = vel * (ddu[sl] / span[sl])[:, None]
            num = (kind == SQUAD) | (kind == CUBIC)
            if np.any(num):
                omega[num], acc = self._numerical(tt[num], order)
                if alpha is not None:
                    alpha[num] = acc
            if np.any(~(pre[:, 3] == 1.0)):
                omega = R.rotate(pre, omega)
                if alpha is not None:
                    alpha = R.rotate(pre, alpha)
            omega = omega * sign[:, None]
        return out, omega, alpha

    def _cubic(self, k: np.ndarray, u: np.ndarray, span: np.ndarray) -> np.ndarray:
        """glTF CUBICSPLINE: ``(2u^3 - 3u^2 + 1) v_k + td (u^3 - 2u^2 + u) b_k + (-2u^3 +
        3u^2) v_k+1 + td (u^3 - u^2) a_k+1``, normalized (tangents as stored)."""
        u2, u3 = u * u, u * u * u
        v0 = self._quats[k]
        v1 = self._quats[k + 1]
        b = self._out[k]
        a = self._in[k + 1]
        r = (
            (2 * u3 - 3 * u2 + 1)[:, None] * v0
            + (span * (u3 - 2 * u2 + u))[:, None] * b
            + (-2 * u3 + 3 * u2)[:, None] * v1
            + (span * (u3 - u2))[:, None] * a
        )
        return R.normalize(r)

    def _numerical(self, t: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Angular velocity/acceleration by fourth-order central differences (h = 1e-4 of
        the segment, inside the segment)."""
        times = self._times
        n = times.shape[0]
        k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, n - 2)
        span = times[k + 1] - times[k]
        h = 1e-4 * span
        lo, hi = times[k], times[k + 1]
        c = np.clip(t, lo + 2 * h, hi - 2 * h)
        samples = [self._evaluate(c + j * h, 0)[0] for j in (-2, -1, 0, 1, 2)]
        base = samples[2]
        rel = [2.0 * R.log(R.multiply(R.align(base, s), R.conjugate(base))) for s in samples]
        omega = (rel[0] - 8 * rel[1] + 8 * rel[3] - rel[4]) / (12 * h)[:, None]
        acc = None
        if order >= 2:
            acc = (-rel[0] + 16 * rel[1] - 30 * rel[2] + 16 * rel[3] - rel[4]) / (12 * h * h)[
                :, None
            ]
        return omega, acc

    # -- serialization ---------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        n = len(self)
        keys: dict[str, Any] = {"time": self._times.tolist(), "rotation": self._quats.tolist()}
        interp = [INTERPOLATIONS[i] for i in self._interp.tolist()]
        keys["interpolation"] = interp[0] if len(set(interp)) == 1 else interp
        if np.any(self._path):
            keys["path"] = [PATHS[i] for i in self._path.tolist()]
        if np.any(self._timing):
            keys["timing"] = [
                _timing_json(int(self._timing[i]), self._timing_params[i]) for i in range(n)
            ]
        if np.any(self._interp == CUBIC):
            keys["in"] = self._in.tolist()
            keys["out"] = self._out.tolist()
        out: dict[str, Any] = {"keys": keys}
        if self._before or self._after:
            out["extrapolation"] = (
                EXTRAPOLATIONS[self._before]
                if self._before == self._after
                else [EXTRAPOLATIONS[self._before], EXTRAPOLATIONS[self._after]]
            )
        return out

    @classmethod
    def from_json(cls, data: Any, location: str = "track") -> RotationTrack:
        if not isinstance(data, dict) or set(data) - {"keys", "extrapolation"}:
            raise _err("INVALID_TRACK", "a rotation track is {keys, extrapolation}", location)
        keys = data.get("keys")
        if isinstance(keys, list):
            return cls.from_keys(
                keys, extrapolation=data.get("extrapolation", "constant"), location=location
            )
        if not isinstance(keys, dict):
            raise _err("INVALID_TRACK", "keys must be columns or a list", f"{location}.keys")
        unknown = set(keys) - {"time", "rotation", "interpolation", "path", "timing", "in", "out"}
        if unknown:
            raise _err(
                "INVALID_TRACK", f"unknown key columns {sorted(unknown)}", f"{location}.keys"
            )
        times = keys.get("time")
        rots = keys.get("rotation")
        if not isinstance(times, list) or not isinstance(rots, list) or len(times) != len(rots):
            raise _err(
                "INVALID_KEYS",
                "time and rotation must be lists of equal length",
                f"{location}.keys",
            )
        n = len(times)

        def col(name: str, default: Any) -> list:
            v = keys.get(name, default)
            items = v if isinstance(v, list) else [v] * n
            if len(items) != n:
                raise _err(
                    "INVALID_KEYS", f"{name} must have {n} entries", f"{location}.keys.{name}"
                )
            return items

        rows = []
        interp = col("interpolation", "slerp")
        paths = col("path", "shortest")
        timing = keys.get("timing")
        timing = timing if isinstance(timing, list) else [timing] * n
        if len(timing) != n:
            raise _err("INVALID_KEYS", f"timing must have {n} entries", f"{location}.keys.timing")
        tin = keys.get("in")
        tout = keys.get("out")
        for i in range(n):
            row: dict[str, Any] = {
                "time": times[i],
                "rotation": rots[i],
                "interpolation": interp[i],
                "path": paths[i],
                "timing": timing[i],
            }
            if tin is not None:
                row["in"] = tin[i]
            if tout is not None:
                row["out"] = tout[i]
            rows.append(row)
        return cls.from_keys(
            rows, extrapolation=data.get("extrapolation", "constant"), location=location
        )

    def same(self, other: RotationTrack) -> bool:
        if self is other:
            return True
        if not isinstance(other, RotationTrack) or len(self) != len(other):
            return False
        if (self._before, self._after) != (other._before, other._after):
            return False
        names = ("_times", "_quats", "_interp", "_path", "_timing", "_timing_params", "_in", "_out")
        return all(getattr(self, a).tobytes() == getattr(other, a).tobytes() for a in names)

    def __repr__(self) -> str:
        return f"RotationTrack({len(self)} keys)"


def _vec4(value: Any, location: str) -> np.ndarray:
    v = np.asarray(value, np.float64).reshape(-1)
    if v.shape != (4,) or not np.all(np.isfinite(v)):
        raise _err("INVALID_KEYS", "a tangent is four finite numbers", location)
    return v


def _extrapolation(value: Any, location: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise _err("INVALID_EXTRAPOLATION", "give one mode or [before, after]", location)
        return (
            _index(EXTRAPOLATIONS, value[0], f"{location}[0]", "INVALID_EXTRAPOLATION"),
            _index(EXTRAPOLATIONS, value[1], f"{location}[1]", "INVALID_EXTRAPOLATION"),
        )
    k = _index(EXTRAPOLATIONS, value, location, "INVALID_EXTRAPOLATION")
    return k, k
