"""F-curves: keyframed scalar functions of time, with exact evaluation and derivatives.

An :class:`FCurve` is immutable: editing returns a new curve (so the history keeps curves
by reference and caches keyed by a curve stay valid). Its keys are stored as structure of
arrays (times, values, per-key interpolation, easing mode and parameters, handle types and
handle offsets), all read-only.

**Keys and segments.** Times are non-decreasing; two keys may share a time (a *jump*: the
curve is right-continuous there -- its value at the time is the second key's, its limit
from the left the first's), never three. Segment ``k`` runs from key ``k`` to key
``k + 1`` and uses key ``k``'s interpolation:

* ``constant`` -- ``v_k`` (a step at the next key);
* ``linear`` -- the straight line;
* ``bezier`` -- the cubic Bezier through ``(t_k, v_k)``, its right handle, the next key's
  left handle and ``(t_{k+1}, v_{k+1})``;
* an easing (``sine`` ... ``bounce``, see :mod:`.easing`) with mode ``in``/``out``/``in_out``.

**Handles** are offsets ``(dt, dv)`` from their key (left: ``dt <= 0``, right: ``dt >= 0``),
so they move with their key. Types (per side):

* ``free`` -- as given; ``aligned`` -- the same slope as the key's other handle;
* ``vector`` -- pointing at the neighbouring key (a third of the way);
* ``auto`` -- the slope of the parabola through the key and its two neighbours
  (non-uniform Catmull-Rom; the secant at the ends): C1-continuous motion;
* ``auto_clamped`` -- ``auto``, but flat at local extrema and at the ends, and limited by
  the Fritsch-Carlson condition (SIAM J. Numer. Anal. 17(2), 1980), which guarantees that
  each segment's values stay between its two keys' values: **no overshoot**.

A key is *unweighted* by default: its handles' time extents are a third of the adjacent
intervals (kept so when keys move: only slopes are stored in effect), which makes the
segment a cubic Hermite curve in time, evaluated without inversion. A *weighted* key keeps
its handles' explicit time extents (longer handles hold the motion longer). As in
Blender's ``BKE_fcurve_correct_bezpart``, a handle reaching past the segment's other end
is scaled back along its own direction to the segment's span at evaluation (the stored
handle is unchanged and the report says so); this keeps time monotonic in every segment,
so the parameter of a time is unique. It is found by Newton's method safeguarded by
bisection (deterministic, only ``+ - * /``), the value by Horner's rule.

**Derivatives.** ``evaluate(times, order=3)`` returns the value and the first three
derivatives with respect to time (velocity, acceleration, jerk), exactly: from the
polynomials for Bezier segments (``v' = y'/x'``, ``v'' = (y''x' - y'x'')/x'^3`` and the
next), from the easings' closed forms, through extrapolation and modifiers. A vertical
handle (``dt = 0``, ``dv != 0``) gives an infinite derivative at its key (reported).

**Extrapolation** before the first and after the last key: ``constant`` (hold),
``linear`` (the end handle's slope for Bezier ends, else the end segment's), ``cycle``
(repeat), ``cycle_offset`` (repeat, each cycle shifted by the value change across the
keys: walk cycles that advance), ``mirror`` (repeat back and forth).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import easing as _easing

INTERPOLATIONS = ("constant", "linear", "bezier", *_easing.EASINGS)
CONSTANT, LINEAR, BEZIER = 0, 1, 2
_FIRST_EASING = 3
HANDLE_TYPES = ("free", "aligned", "vector", "auto", "auto_clamped")
FREE, ALIGNED, VECTOR, AUTO, AUTO_CLAMPED = range(5)
EXTRAPOLATIONS = ("constant", "linear", "cycle", "cycle_offset", "mirror")

MAX_KEYS = 1_000_000
MAX_ABS_TIME = 1.0e9
MAX_ABS_VALUE = 1.0e15
NEWTON_ITERATIONS = 64

# Segment kinds of a compiled curve.
_SEG_CONSTANT, _SEG_LINEAR, _SEG_HERMITE, _SEG_BEZIER, _SEG_EASE, _SEG_JUMP = range(6)

_FIELDS = (
    "times",
    "values",
    "interpolation",
    "ease",
    "left_type",
    "right_type",
    "left_dt",
    "left_dv",
    "right_dt",
    "right_dv",
    "weighted",
    "back",
    "amplitude",
    "period",
)


def _err(code: str, message: str, location: str, **kw: Any) -> AnimationError:
    return AnimationError(code, message, location, **kw)


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.flags.writeable:
        a = a.copy() if a.base is not None else a
        a.flags.writeable = False
    return a


def _number(value: Any, location: str, what: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err("INVALID_KEYS", f"{what} must be a number, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v):
        raise _err("INVALID_KEYS", f"{what} must be finite, got {value!r}", location)
    return v


def _index(table: Sequence[str], value: Any, location: str, code: str) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if 0 <= int(value) < len(table):
            return int(value)
    elif isinstance(value, str) and value in table:
        return table.index(value)
    raise _err(code, f"{value!r} is not one of {', '.join(table)}", location)


class FCurve:
    """An immutable keyframed curve; see the module docstring.

    Build one with :meth:`from_keys` (key dicts or ``(time, value)`` pairs) or
    :meth:`from_arrays`; read keys with :meth:`keys`; evaluate with :meth:`evaluate`.
    """

    __slots__ = (*(f"_{f}" for f in _FIELDS), "_before", "_after", "_modifiers", "_compiled")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:  # pragma: no cover - use the constructors
        raise TypeError("use FCurve.from_keys(...) or FCurve.from_arrays(...)")

    @classmethod
    def _make(
        cls,
        arrays: dict[str, np.ndarray],
        before: int,
        after: int,
        modifiers: tuple = (),
        location: str = "curve",
        recompute: bool = True,
    ) -> FCurve:
        self = object.__new__(cls)
        a = dict(arrays)
        _check_arrays(a, location)
        if recompute:
            _recompute_handles(a)
        for name in _FIELDS:
            object.__setattr__(self, f"_{name}", _frozen(a[name]))
        self._before = before
        self._after = after
        self._modifiers = tuple(modifiers)
        self._compiled = None
        if (before >= 2 or after >= 2) and not self._times[-1] > self._times[0]:
            raise _err(
                "INVALID_EXTRAPOLATION",
                "cyclic extrapolation needs keys spanning a positive time",
                f"{location}.extrapolation",
            )
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_before", "_after", "_modifiers", "_compiled") or name.startswith("_"):
            object.__setattr__(self, name, value)
        else:  # pragma: no cover
            raise AttributeError("FCurve is immutable")

    @classmethod
    def from_keys(
        cls,
        keys: Iterable[Any],
        *,
        interpolation: str = "bezier",
        handles: str = "auto_clamped",
        extrapolation: Any = "constant",
        modifiers: Iterable[Any] = (),
        location: str = "curve",
    ) -> FCurve:
        """A curve through ``keys``: ``(time, value)`` pairs or dicts with ``time``,
        ``value`` and optionally ``interpolation``, ``ease``, ``left`` / ``right``
        (``{"type", "dt", "dv"}`` or a type name), ``weighted``, ``back``,
        ``amplitude``, ``period``. ``interpolation`` and ``handles`` are the defaults."""
        default_interp = _index(
            INTERPOLATIONS, interpolation, f"{location}.interpolation", "INVALID_INTERPOLATION"
        )
        default_handle = _index(HANDLE_TYPES, handles, f"{location}.handles", "INVALID_HANDLE")
        rows = []
        for i, raw in enumerate(keys):
            rows.append(_key_row(raw, f"{location}.keys[{i}]", default_interp, default_handle))
            if len(rows) > MAX_KEYS:
                raise _err("LIMIT_EXCEEDED", f"more than {MAX_KEYS} keys", f"{location}.keys")
        if not rows:
            raise _err("INVALID_KEYS", "a curve needs at least one key", f"{location}.keys")
        order = sorted(range(len(rows)), key=lambda k: rows[k]["times"])  # stable
        arrays: dict[str, Any] = {
            name: np.array([rows[k][name] for k in order], dtype=_DTYPES[name]) for name in _FIELDS
        }
        arrays["_slope_left"] = np.array([rows[k]["_slope_left"] for k in order], bool)
        arrays["_slope_right"] = np.array([rows[k]["_slope_right"] for k in order], bool)
        before, after = _extrapolation(extrapolation, f"{location}.extrapolation")
        mods = _modifiers(modifiers, f"{location}.modifiers")
        return cls._make(arrays, before, after, mods, location)

    @classmethod
    def from_arrays(
        cls,
        times: Any,
        values: Any,
        *,
        interpolation: Any = "bezier",
        handles: Any = "auto_clamped",
        extrapolation: Any = "constant",
        modifiers: Iterable[Any] = (),
        location: str = "curve",
    ) -> FCurve:
        """A curve from arrays of times and values (sorted), with one interpolation and
        handle type for every key (names, or arrays of indices)."""
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if t.shape != v.shape:
            raise _err("INVALID_KEYS", "times and values differ in length", f"{location}.keys")
        n = t.shape[0]
        if n == 0:
            raise _err("INVALID_KEYS", "a curve needs at least one key", f"{location}.keys")
        if n > MAX_KEYS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_KEYS} keys", f"{location}.keys")

        def column(value: Any, table: Sequence[str], code: str, what: str) -> np.ndarray:
            if isinstance(value, str):
                return np.full(n, _index(table, value, f"{location}.{what}", code), np.uint8)
            arr = np.asarray(value)
            if arr.shape != (n,) or arr.dtype.kind not in "iu":
                raise _err(code, f"{what} must be a name or {n} indices", f"{location}.{what}")
            if arr.size and (arr.min() < 0 or arr.max() >= len(table)):
                raise _err(code, f"{what} index out of range", f"{location}.{what}")
            return arr.astype(np.uint8)

        handle = column(handles, HANDLE_TYPES, "INVALID_HANDLE", "handles")
        arrays = {
            "times": t.copy(),
            "values": v.copy(),
            "interpolation": column(
                interpolation, INTERPOLATIONS, "INVALID_INTERPOLATION", "interpolation"
            ),
            "ease": np.full(n, 2, np.uint8),
            "left_type": handle.copy(),
            "right_type": handle.copy(),
            "left_dt": np.zeros(n),
            "left_dv": np.zeros(n),
            "right_dt": np.zeros(n),
            "right_dv": np.zeros(n),
            "weighted": np.zeros(n, bool),
            "back": np.full(n, _easing.DEFAULT_BACK),
            "amplitude": np.full(n, _easing.DEFAULT_AMPLITUDE),
            "period": np.full(n, _easing.DEFAULT_PERIOD),
        }
        before, after = _extrapolation(extrapolation, f"{location}.extrapolation")
        return cls._make(
            arrays, before, after, _modifiers(modifiers, f"{location}.modifiers"), location
        )

    @classmethod
    def constant_value(cls, value: float, time: float = 0.0) -> FCurve:
        return cls.from_keys([(time, value)], interpolation="constant")

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return int(self._times.shape[0])

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def interpolation(self) -> np.ndarray:
        return self._interpolation

    @property
    def extrapolation(self) -> tuple[str, str]:
        return EXTRAPOLATIONS[self._before], EXTRAPOLATIONS[self._after]

    @property
    def modifiers(self) -> tuple:
        return self._modifiers

    @property
    def range(self) -> tuple[float, float]:
        """``(first key time, last key time)``."""
        return float(self._times[0]), float(self._times[-1])

    def array(self, name: str) -> np.ndarray:
        """A read-only key array (``times``, ``values``, ``left_dt``, ...)."""
        if name not in _FIELDS:
            raise _err("INVALID_ARGUMENT", f"unknown key field {name!r}", "field")
        return getattr(self, f"_{name}")

    def arrays(self) -> dict[str, np.ndarray]:
        """Copies of every key array (writable), for building an edited curve."""
        return {name: getattr(self, f"_{name}").copy() for name in _FIELDS}

    def key(self, index: int) -> dict[str, Any]:
        """Key ``index`` as a dict (the form :meth:`from_keys` accepts)."""
        n = len(self)
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not -n <= index < n
        ):
            raise _err("INVALID_KEY_INDEX", f"no key {index!r} (the curve has {n})", "index")
        i = int(index) % n
        out: dict[str, Any] = {
            "time": float(self._times[i]),
            "value": float(self._values[i]),
            "interpolation": INTERPOLATIONS[self._interpolation[i]],
            "left": {
                "type": HANDLE_TYPES[self._left_type[i]],
                "dt": float(self._left_dt[i]),
                "dv": float(self._left_dv[i]),
            },
            "right": {
                "type": HANDLE_TYPES[self._right_type[i]],
                "dt": float(self._right_dt[i]),
                "dv": float(self._right_dv[i]),
            },
            "weighted": bool(self._weighted[i]),
        }
        if self._interpolation[i] >= _FIRST_EASING:
            out["ease"] = _easing.MODES[self._ease[i]]
            name = INTERPOLATIONS[self._interpolation[i]]
            if name == "back":
                out["back"] = float(self._back[i])
            if name == "elastic":
                out["amplitude"] = float(self._amplitude[i])
                out["period"] = float(self._period[i])
        return out

    def keys(self) -> list[dict[str, Any]]:
        return [self.key(i) for i in range(len(self))]

    def slope(self, index: int, side: str) -> float:
        """The slope ``dv/dt`` of a handle (``inf`` for a vertical one)."""
        i = int(index)
        dt = float(self._right_dt[i] if side == "right" else self._left_dt[i])
        dv = float(self._right_dv[i] if side == "right" else self._left_dv[i])
        if dt == 0.0:
            return 0.0 if dv == 0.0 else math.copysign(math.inf, dv if side == "right" else -dv)
        return dv / dt

    # ------------------------------------------------------------------ #
    # Editing (each returns a new curve)
    # ------------------------------------------------------------------ #

    def replace(
        self,
        arrays: dict[str, np.ndarray] | None = None,
        *,
        extrapolation: Any = None,
        modifiers: Any = None,
        location: str = "curve",
        recompute: bool = True,
    ) -> FCurve:
        """A curve with some arrays, the extrapolation or the modifiers replaced (keys
        re-sorted by time, stably; automatic handles recomputed)."""
        a = self.arrays()
        if arrays:
            for name, value in arrays.items():
                if name not in _FIELDS:
                    raise _err("INVALID_ARGUMENT", f"unknown key field {name!r}", location)
                a[name] = np.asarray(value, dtype=_DTYPES[name]).reshape(-1).copy()
        n = a["times"].shape[0]
        if any(a[name].shape != (n,) for name in _FIELDS):
            raise _err("INVALID_KEYS", "key arrays differ in length", f"{location}.keys")
        order = np.argsort(a["times"], kind="stable")
        if not np.array_equal(order, np.arange(n)):
            a = {name: a[name][order] for name in _FIELDS}
        before, after = (
            (self._before, self._after)
            if extrapolation is None
            else _extrapolation(extrapolation, f"{location}.extrapolation")
        )
        mods = (
            self._modifiers if modifiers is None else _modifiers(modifiers, f"{location}.modifiers")
        )
        return FCurve._make(a, before, after, mods, location, recompute)

    def with_keys(self, keys: Iterable[Any], *, location: str = "curve") -> FCurve:
        """The same extrapolation and modifiers with other keys."""
        new = FCurve.from_keys(keys, location=location)
        return new.replace(
            extrapolation=self.extrapolation, modifiers=self._modifiers, location=location
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def compiled(self) -> _Compiled:
        c = self._compiled
        if c is None:
            c = _compile(self)
            object.__setattr__(self, "_compiled", c)
        return c

    def evaluate(self, times: Any, order: int = 0) -> Any:
        """The value (``order=0``) or ``(value, velocity, acceleration, jerk)[: order+1]``
        at ``times`` (a number or an array)."""
        if isinstance(order, bool) or order not in (0, 1, 2, 3):
            raise _err("INVALID_ARGUMENT", f"order must be 0..3, got {order!r}", "order")
        scalar = np.ndim(times) == 0
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(t)):
            raise _err("INVALID_TIME", "times must be finite", "times")
        from .modifiers import evaluate_with_modifiers

        out = evaluate_with_modifiers(self, t, order)
        if scalar:
            out = [float(o[0]) for o in out]
        return out[0] if order == 0 else tuple(out)

    def evaluate_base(self, t: np.ndarray, order: int) -> list[np.ndarray]:
        """Keys, interpolation and extrapolation only (no modifiers)."""
        return _evaluate(self.compiled(), self, t, order)

    def __call__(self, time: float) -> float:
        return float(self.evaluate(time))

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_json(self) -> dict[str, Any]:
        """Columnar JSON (a field the same for every key is one value)."""
        n = len(self)
        keys: dict[str, Any] = {
            "time": self._times.tolist(),
            "value": self._values.tolist(),
        }

        def column(arr: np.ndarray, table: Sequence[str] | None = None, default: Any = None) -> Any:
            values = arr.tolist()
            if table is not None:
                values = [table[v] for v in values]
            first = values[0]
            if all(v == first for v in values):
                return None if first == default else first
            return values

        for name, table, default in (
            ("interpolation", INTERPOLATIONS, None),
            ("ease", _easing.MODES, "in_out"),
            ("left_type", HANDLE_TYPES, None),
            ("right_type", HANDLE_TYPES, None),
            ("weighted", None, False),
            ("back", None, _easing.DEFAULT_BACK),
            ("amplitude", None, _easing.DEFAULT_AMPLITUDE),
            ("period", None, _easing.DEFAULT_PERIOD),
        ):
            value = column(getattr(self, f"_{name}"), table, default)
            if value is not None:
                keys[name] = value
        # Handles: only those of free/aligned sides (and of weighted keys) carry data.
        stored = (self._left_type <= ALIGNED) | (self._right_type <= ALIGNED) | self._weighted
        if np.any(stored):
            rows = np.flatnonzero(stored)
            keys["handles"] = {
                "keys": rows.tolist(),
                "left": np.stack([self._left_dt[rows], self._left_dv[rows]], 1).tolist(),
                "right": np.stack([self._right_dt[rows], self._right_dv[rows]], 1).tolist(),
            }
        out: dict[str, Any] = {"keys": keys}
        if self._before or self._after:
            out["extrapolation"] = (
                EXTRAPOLATIONS[self._before]
                if self._before == self._after
                else [EXTRAPOLATIONS[self._before], EXTRAPOLATIONS[self._after]]
            )
        if self._modifiers:
            out["modifiers"] = [m.to_json() for m in self._modifiers]
        del n
        return out

    @classmethod
    def from_json(cls, data: Any, location: str = "curve") -> FCurve:
        if not isinstance(data, dict):
            raise _err("INVALID_CURVE", "a curve must be an object", location)
        unknown = set(data) - {"keys", "extrapolation", "modifiers"}
        if unknown:
            raise _err("INVALID_CURVE", f"unknown curve fields {sorted(unknown)}", location)
        keys = data.get("keys")
        if isinstance(keys, list):  # a list of key dicts or pairs
            return cls.from_keys(
                keys,
                extrapolation=data.get("extrapolation", "constant"),
                modifiers=data.get("modifiers", ()),
                location=location,
            )
        if not isinstance(keys, dict):
            raise _err(
                "INVALID_CURVE", "keys must be an object of columns or a list", f"{location}.keys"
            )
        allowed = {
            "time", "value", "interpolation", "ease", "left_type", "right_type", "weighted",
            "back", "amplitude", "period", "handles",
        }  # fmt: skip
        unknown = set(keys) - allowed
        if unknown:
            raise _err(
                "INVALID_CURVE", f"unknown key columns {sorted(unknown)}", f"{location}.keys"
            )
        times = keys.get("time")
        values = keys.get("value")
        if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
            raise _err(
                "INVALID_KEYS", "time and value must be lists of equal length", f"{location}.keys"
            )
        n = len(times)
        if n == 0:
            raise _err("INVALID_KEYS", "a curve needs at least one key", f"{location}.keys")
        if n > MAX_KEYS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_KEYS} keys", f"{location}.keys")
        t = np.array(
            [_number(x, f"{location}.keys.time[{i}]", "time") for i, x in enumerate(times)]
        )
        v = np.array([_number(x, f"{location}.keys.value[{i}]") for i, x in enumerate(values)])

        def column(name: str, table: Sequence[str] | None, default: Any, dtype: Any) -> np.ndarray:
            raw = keys.get(name, default)
            loc = f"{location}.keys.{name}"
            items = raw if isinstance(raw, list) else [raw] * n
            if len(items) != n:
                raise _err("INVALID_KEYS", f"{name} must have {n} entries", loc)
            if table is not None:
                return np.array(
                    [_index(table, x, f"{loc}[{i}]", "INVALID_KEYS") for i, x in enumerate(items)],
                    dtype,
                )
            if dtype is bool:
                if not all(isinstance(x, bool) for x in items):
                    raise _err("INVALID_KEYS", f"{name} must be booleans", loc)
                return np.array(items, bool)
            return np.array([_number(x, f"{loc}[{i}]", name) for i, x in enumerate(items)], dtype)

        arrays = {
            "times": t,
            "values": v,
            "interpolation": column("interpolation", INTERPOLATIONS, "bezier", np.uint8),
            "ease": column("ease", _easing.MODES, "in_out", np.uint8),
            "left_type": column("left_type", HANDLE_TYPES, "auto_clamped", np.uint8),
            "right_type": column("right_type", HANDLE_TYPES, "auto_clamped", np.uint8),
            "weighted": column("weighted", None, False, bool),
            "back": column("back", None, _easing.DEFAULT_BACK, np.float64),
            "amplitude": column("amplitude", None, _easing.DEFAULT_AMPLITUDE, np.float64),
            "period": column("period", None, _easing.DEFAULT_PERIOD, np.float64),
            "left_dt": np.zeros(n),
            "left_dv": np.zeros(n),
            "right_dt": np.zeros(n),
            "right_dv": np.zeros(n),
        }
        handles = keys.get("handles")
        if handles is not None:
            hloc = f"{location}.keys.handles"
            if not isinstance(handles, dict) or set(handles) != {"keys", "left", "right"}:
                raise _err("INVALID_KEYS", "handles must be {keys, left, right}", hloc)
            rows = handles["keys"]
            if (
                not isinstance(rows, list)
                or len(rows) != len(handles["left"])
                or len(rows) != len(handles["right"])
            ):
                raise _err("INVALID_KEYS", "handles lists must have equal lengths", hloc)
            for j, k in enumerate(rows):
                if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n:
                    raise _err(
                        "INVALID_KEYS", f"handle key index {k!r} out of range", f"{hloc}.keys[{j}]"
                    )
                for side in ("left", "right"):
                    pair = handles[side][j]
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise _err("INVALID_KEYS", "a handle is [dt, dv]", f"{hloc}.{side}[{j}]")
                    arrays[f"{side}_dt"][k] = _number(pair[0], f"{hloc}.{side}[{j}][0]", "dt")
                    arrays[f"{side}_dv"][k] = _number(pair[1], f"{hloc}.{side}[{j}][1]", "dv")
        if np.any(np.diff(t) < 0):
            raise _err("INVALID_KEYS", "key times must be non-decreasing", f"{location}.keys.time")
        before, after = _extrapolation(
            data.get("extrapolation", "constant"), f"{location}.extrapolation"
        )
        mods = _modifiers(data.get("modifiers", ()), f"{location}.modifiers")
        return cls._make(arrays, before, after, mods, location)

    # ------------------------------------------------------------------ #
    # Equality (exact) and display
    # ------------------------------------------------------------------ #

    def same(self, other: FCurve) -> bool:
        """Bitwise equality of keys, extrapolation and modifiers."""
        if self is other:
            return True
        if not isinstance(other, FCurve) or len(self) != len(other):
            return False
        if (self._before, self._after) != (other._before, other._after):
            return False
        if [m.to_json() for m in self._modifiers] != [m.to_json() for m in other._modifiers]:
            return False
        return all(
            getattr(self, f"_{f}").tobytes() == getattr(other, f"_{f}").tobytes() for f in _FIELDS
        )

    def __repr__(self) -> str:
        t0, t1 = self.range
        return f"FCurve({len(self)} keys, {t0:g}..{t1:g} s)"


_DTYPES = {
    "times": np.float64,
    "values": np.float64,
    "interpolation": np.uint8,
    "ease": np.uint8,
    "left_type": np.uint8,
    "right_type": np.uint8,
    "left_dt": np.float64,
    "left_dv": np.float64,
    "right_dt": np.float64,
    "right_dv": np.float64,
    "weighted": bool,
    "back": np.float64,
    "amplitude": np.float64,
    "period": np.float64,
}


# --------------------------------------------------------------------------------------- #
# Validation of keys
# --------------------------------------------------------------------------------------- #


def _handle_spec(raw: Any, default: int, location: str) -> tuple[int, float | None, float | None]:
    if raw is None:
        return default, None, None
    if isinstance(raw, str):
        return _index(HANDLE_TYPES, raw, location, "INVALID_HANDLE"), None, None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (
            FREE,
            _number(raw[0], f"{location}[0]", "dt"),
            _number(raw[1], f"{location}[1]", "dv"),
        )
    if not isinstance(raw, dict):
        raise _err(
            "INVALID_HANDLE", "a handle is a type name, [dt, dv] or {type, dt, dv}", location
        )
    unknown = set(raw) - {"type", "dt", "dv", "slope"}
    if unknown:
        raise _err("INVALID_HANDLE", f"unknown handle fields {sorted(unknown)}", location)
    kind = _index(
        HANDLE_TYPES,
        raw.get("type", "free" if ("dt" in raw or "slope" in raw) else HANDLE_TYPES[default]),
        f"{location}.type",
        "INVALID_HANDLE",
    )
    dt = _number(raw["dt"], f"{location}.dt", "dt") if "dt" in raw else None
    if "slope" in raw:
        if "dv" in raw:
            raise _err("INVALID_HANDLE", "give dv or slope, not both", location)
        slope = _number(raw["slope"], f"{location}.slope", "slope")
        return kind, dt, ("slope", slope)  # type: ignore[return-value]
    dv = _number(raw["dv"], f"{location}.dv", "dv") if "dv" in raw else None
    return kind, dt, dv


def _key_row(raw: Any, location: str, default_interp: int, default_handle: int) -> dict[str, Any]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        raw = {"time": raw[0], "value": raw[1]}
    if not isinstance(raw, dict):
        raise _err("INVALID_KEYS", "a key is (time, value) or an object", location)
    allowed = {
        "time",
        "value",
        "interpolation",
        "ease",
        "left",
        "right",
        "weighted",
        "back",
        "amplitude",
        "period",
        "handles",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise _err("INVALID_KEYS", f"unknown key fields {sorted(unknown)}", location)
    if "time" not in raw or "value" not in raw:
        raise _err("INVALID_KEYS", "a key needs a time and a value", location)
    t = _number(raw["time"], f"{location}.time", "time")
    v = _number(raw["value"], f"{location}.value")
    if abs(t) > MAX_ABS_TIME:
        raise _err("INVALID_KEYS", f"time beyond +-{MAX_ABS_TIME:g} s", f"{location}.time")
    if abs(v) > MAX_ABS_VALUE:
        raise _err("INVALID_KEYS", f"value beyond +-{MAX_ABS_VALUE:g}", f"{location}.value")
    interp = (
        _index(
            INTERPOLATIONS,
            raw["interpolation"],
            f"{location}.interpolation",
            "INVALID_INTERPOLATION",
        )
        if "interpolation" in raw
        else default_interp
    )
    ease = _index(_easing.MODES, raw.get("ease", "in_out"), f"{location}.ease", "INVALID_EASING")
    both = raw.get("handles")
    lt, ldt, ldv = _handle_spec(raw.get("left", both), default_handle, f"{location}.left")
    rt, rdt, rdv = _handle_spec(raw.get("right", both), default_handle, f"{location}.right")
    weighted = raw.get("weighted", False)
    if not isinstance(weighted, bool):
        raise _err("INVALID_KEYS", "weighted must be true or false", f"{location}.weighted")
    back = _number(raw.get("back", _easing.DEFAULT_BACK), f"{location}.back", "back")
    amplitude = _number(
        raw.get("amplitude", _easing.DEFAULT_AMPLITUDE), f"{location}.amplitude", "amplitude"
    )
    period = _number(raw.get("period", _easing.DEFAULT_PERIOD), f"{location}.period", "period")
    if not 0.0 <= back <= 100.0:
        raise _err("INVALID_EASING", "back overshoot must be within 0..100", f"{location}.back")
    if not 1.0 <= amplitude <= 100.0:
        raise _err(
            "INVALID_EASING", "elastic amplitude must be within 1..100", f"{location}.amplitude"
        )
    if not 1e-3 <= period <= 10.0:
        raise _err(
            "INVALID_EASING", "elastic period must be within 0.001..10", f"{location}.period"
        )

    def resolve(dt, dv, side):
        if dv is not None and isinstance(dv, tuple):  # ("slope", s)
            slope = dv[1]
            return dt, None, slope
        return dt, dv, None

    ldt, ldv, lslope = resolve(ldt, ldv, "left")
    rdt, rdv, rslope = resolve(rdt, rdv, "right")
    for side, dt in (("left", ldt), ("right", rdt)):
        if dt is not None and ((side == "left" and dt > 0) or (side == "right" and dt < 0)):
            raise _err(
                "INVALID_HANDLE",
                f"a {side} handle must point {'back' if side == 'left' else 'forward'} in time "
                f"(dt {'<=' if side == 'left' else '>='} 0), got dt={dt!r}",
                f"{location}.{side}.dt",
            )
    return {
        "times": t,
        "values": v,
        "interpolation": interp,
        "ease": ease,
        "left_type": lt,
        "right_type": rt,
        # Explicit offsets; slopes become offsets once intervals are known (below).
        "left_dt": np.nan if ldt is None else ldt,
        "left_dv": (np.nan if ldv is None else ldv) if lslope is None else lslope,
        "right_dt": np.nan if rdt is None else rdt,
        "right_dv": (np.nan if rdv is None else rdv) if rslope is None else rslope,
        "weighted": weighted,
        "back": back,
        "amplitude": amplitude,
        "period": period,
        "_slope_left": lslope is not None,
        "_slope_right": rslope is not None,
    }


def _check_arrays(a: dict[str, np.ndarray], location: str) -> None:
    t = a["times"]
    n = t.shape[0]
    if n == 0:
        raise _err("INVALID_KEYS", "a curve needs at least one key", f"{location}.keys")
    for name in ("times", "values"):
        if not np.all(np.isfinite(a[name])):
            raise _err("INVALID_KEYS", f"key {name} must be finite", f"{location}.keys")
    if np.any(np.abs(t) > MAX_ABS_TIME):
        raise _err("INVALID_KEYS", f"key times beyond +-{MAX_ABS_TIME:g} s", f"{location}.keys")
    if np.any(np.abs(a["values"]) > MAX_ABS_VALUE):
        raise _err("INVALID_KEYS", f"key values beyond +-{MAX_ABS_VALUE:g}", f"{location}.keys")
    d = np.diff(t)
    if np.any(d < 0):
        raise _err("INVALID_KEYS", "key times must be non-decreasing", f"{location}.keys")
    zero = d == 0
    if np.any(zero[1:] & zero[:-1]):
        k = int(np.flatnonzero(zero[1:] & zero[:-1])[0])
        raise _err(
            "INVALID_KEYS",
            f"three keys share the time {t[k]!r}: at most two (a jump)",
            f"{location}.keys[{k}]",
        )
    if np.any(a["interpolation"] >= len(INTERPOLATIONS)) or np.any(
        a["left_type"] >= len(HANDLE_TYPES)
    ):
        raise _err("INVALID_KEYS", "interpolation or handle type out of range", f"{location}.keys")
    if np.any(a["right_type"] >= len(HANDLE_TYPES)) or np.any(a["ease"] >= len(_easing.MODES)):
        raise _err("INVALID_KEYS", "handle type or easing mode out of range", f"{location}.keys")
    for side, sign in (("left", -1.0), ("right", 1.0)):
        dt = a[f"{side}_dt"]
        bad = np.isfinite(dt) & (dt * sign < 0)
        if np.any(bad):
            k = int(np.flatnonzero(bad)[0])
            raise _err(
                "INVALID_HANDLE",
                f"a {side} handle must point {'back' if sign < 0 else 'forward'} in time",
                f"{location}.keys[{k}].{side}",
            )
    if not (np.all(np.isfinite(a["back"])) and np.all(np.isfinite(a["amplitude"]))):
        raise _err("INVALID_EASING", "easing parameters must be finite", f"{location}.keys")
    if not np.all(np.isfinite(a["period"])) or np.any(a["period"] <= 0):
        raise _err("INVALID_EASING", "elastic periods must be positive", f"{location}.keys")


# --------------------------------------------------------------------------------------- #
# Handles
# --------------------------------------------------------------------------------------- #


def _recompute_handles(a: dict[str, Any]) -> None:
    """Fill and update handle offsets from handle types (in place).

    Missing (NaN) offsets are filled; ``slope``-given handles (rows of a key table) turn
    into offsets; automatic types are recomputed; unweighted keys get time extents of a
    third of their intervals (their slopes kept)."""
    t = a["times"].astype(np.float64)
    v = a["values"].astype(np.float64)
    n = t.shape[0]
    slope_left = a.pop("_slope_left", None)
    slope_right = a.pop("_slope_right", None)
    ldt = np.asarray(a["left_dt"], np.float64).copy()
    ldv = np.asarray(a["left_dv"], np.float64).copy()
    rdt = np.asarray(a["right_dt"], np.float64).copy()
    rdv = np.asarray(a["right_dv"], np.float64).copy()
    lt = np.asarray(a["left_type"], np.uint8)
    rt = np.asarray(a["right_type"], np.uint8)
    weighted = np.asarray(a["weighted"], bool)

    # Intervals to the neighbours (0: none, or a jump partner).
    prev_dt = np.zeros(n)
    next_dt = np.zeros(n)
    if n > 1:
        d = np.diff(t)
        prev_dt[1:] = d
        next_dt[:-1] = d
    has_prev = prev_dt > 0
    has_next = next_dt > 0
    sec_prev = np.zeros(n)
    sec_next = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        if n > 1:
            s = np.diff(v) / np.where(np.diff(t) > 0, np.diff(t), 1.0)
            sec_prev[1:] = np.where(prev_dt[1:] > 0, s, 0.0)
            sec_next[:-1] = np.where(next_dt[:-1] > 0, s, 0.0)
    # Third-of-interval time extents (the end without a neighbour mirrors the other side;
    # a lone key gets 1/3 s for display and extrapolation).
    left_len = np.where(has_prev, prev_dt / 3.0, np.where(has_next, next_dt / 3.0, 1.0 / 3.0))
    right_len = np.where(has_next, next_dt / 3.0, np.where(has_prev, prev_dt / 3.0, 1.0 / 3.0))

    # Slope-given handles: offsets from the slopes.
    if slope_left is not None:
        sl = (
            np.asarray(slope_left, bool)
            if not isinstance(slope_left, bool)
            else np.full(n, slope_left)
        )
        sr = (
            np.asarray(slope_right, bool)
            if not isinstance(slope_right, bool)
            else np.full(n, slope_right)
        )
        use = sl & ~np.isfinite(ldt)
        ldt[use] = -left_len[use]
        ldv[sl] = ldv[sl] * ldt[sl]
        use = sr & ~np.isfinite(rdt)
        rdt[use] = right_len[use]
        rdv[sr] = rdv[sr] * rdt[sr]

    # Automatic slopes.
    both = has_prev & has_next
    with np.errstate(divide="ignore", invalid="ignore"):
        parabola = np.where(
            both,
            (sec_prev * next_dt + sec_next * prev_dt) / np.where(both, prev_dt + next_dt, 1.0),
            np.where(has_prev, sec_prev, np.where(has_next, sec_next, 0.0)),
        )
    clamped = parabola.copy()
    extremum = both & (sec_prev * sec_next <= 0.0)
    clamped[extremum | ~both] = 0.0
    # Fritsch-Carlson: alpha^2 + beta^2 <= 9 on every segment makes the cubic Hermite
    # segment monotone (no overshoot). Each key takes the smaller of the factors its two
    # segments ask for, so every segment satisfies the bound (and no order of
    # processing matters). Slopes keep their signs (zero at extrema, from the rule above).
    if n > 1:
        d = sec_next[:-1]
        live = next_dt[:-1] > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.where(live & (d != 0.0), clamped[:-1] / np.where(d != 0.0, d, 1.0), 0.0)
            beta = np.where(live & (d != 0.0), clamped[1:] / np.where(d != 0.0, d, 1.0), 0.0)
            r2 = alpha * alpha + beta * beta
            tau = np.where(r2 > 9.0, 3.0 / np.sqrt(np.where(r2 > 9.0, r2, 1.0)), 1.0)
        factor = np.ones(n)
        factor[:-1] = np.minimum(factor[:-1], tau)
        factor[1:] = np.minimum(factor[1:], tau)
        clamped = clamped * factor

    def fill(
        side: str,
        types: np.ndarray,
        dt: np.ndarray,
        dv: np.ndarray,
        length: np.ndarray,
        sign: float,
    ) -> None:
        auto = types == AUTO
        dt[auto] = sign * length[auto]
        dv[auto] = parabola[auto] * dt[auto]
        ac = types == AUTO_CLAMPED
        dt[ac] = sign * length[ac]
        dv[ac] = clamped[ac] * dt[ac]
        vec = types == VECTOR
        dt[vec] = sign * length[vec]
        target = sec_prev if side == "left" else sec_next
        dv[vec] = target[vec] * dt[vec]
        # Free and aligned handles without offsets: flat, a third of the interval.
        missing = ~np.isfinite(dt)
        dt[missing] = sign * length[missing]
        missing_v = ~np.isfinite(dv)
        dv[missing_v] = 0.0
        # Unweighted free/aligned handles: time extents of a third of the interval,
        # slopes kept.
        manual = ((types == FREE) | (types == ALIGNED)) & ~weighted
        with np.errstate(divide="ignore", invalid="ignore"):
            old_dt = dt[manual]
            slope = np.where(
                old_dt != 0.0, dv[manual] / np.where(old_dt != 0.0, old_dt, 1.0), np.nan
            )
        new_dt = sign * length[manual]
        # A vertical handle (dt = 0, dv != 0) stays vertical: its slope is infinite. An
        # unchanged extent keeps its offset bit for bit (recomputing is idempotent).
        vertical = ~np.isfinite(slope)
        dv_new = np.where(vertical | (new_dt == old_dt), dv[manual], slope * new_dt)
        dt[manual] = np.where(vertical, dt[manual], new_dt)
        dv[manual] = dv_new

    fill("left", lt, ldt, ldv, left_len, -1.0)
    fill("right", rt, rdt, rdv, right_len, 1.0)

    # Aligned handles take the other side's slope (their own time extent kept).
    for side_a in ("left", "right"):
        types_a = lt if side_a == "left" else rt
        types_b = rt if side_a == "left" else lt
        dta, dva = (ldt, ldv) if side_a == "left" else (rdt, rdv)
        dtb, dvb = (rdt, rdv) if side_a == "left" else (ldt, ldv)
        follow = (types_a == ALIGNED) & (types_b != ALIGNED)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope_b = np.where(dtb != 0.0, dvb / np.where(dtb != 0.0, dtb, 1.0), 0.0)
        dva[follow] = slope_b[follow] * dta[follow]
    # Both aligned: the right handle leads (set_handle aligns the other side explicitly).
    pair = (lt == ALIGNED) & (rt == ALIGNED)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_r = np.where(rdt != 0.0, rdv / np.where(rdt != 0.0, rdt, 1.0), 0.0)
    ldv[pair] = slope_r[pair] * ldt[pair]

    a["left_dt"], a["left_dv"], a["right_dt"], a["right_dv"] = ldt, ldv, rdt, rdv


# --------------------------------------------------------------------------------------- #
# Compilation and evaluation
# --------------------------------------------------------------------------------------- #


class _Compiled:
    """Per-segment data of a curve, ready for vectorized evaluation."""

    __slots__ = (
        "t",
        "v",
        "kind",
        "x",
        "y",
        "span",
        "easing",
        "mode",
        "back",
        "amplitude",
        "period",
        "corrected",
        "slope_before",
        "slope_after",
    )


def _compile(curve: FCurve) -> _Compiled:
    t = curve._times
    v = curve._values
    n = t.shape[0]
    m = max(n - 1, 0)
    c = _Compiled()
    c.t = t
    c.v = v
    interp = curve._interpolation[:m]
    span = np.diff(t)
    kind = np.full(m, _SEG_CONSTANT, np.uint8)
    kind[interp == LINEAR] = _SEG_LINEAR
    bez = interp == BEZIER
    weighted = curve._weighted
    w_seg = (weighted[:-1] | weighted[1:]) if n > 1 else np.zeros(0, bool)
    third = span / 3.0
    exact = (
        (curve._right_dt[:m] == third) & (curve._left_dt[1:] == -third)
        if n > 1
        else np.zeros(0, bool)
    )
    kind[bez & ~w_seg & exact] = _SEG_HERMITE
    kind[bez & (w_seg | ~exact)] = _SEG_BEZIER
    kind[interp >= _FIRST_EASING] = _SEG_EASE
    kind[span == 0.0] = _SEG_JUMP
    c.kind = _frozen(kind)
    c.span = _frozen(span)
    # Bezier control points (handles corrected so time stays monotonic).
    p1t = curve._right_dt[:m].copy()
    p1v = curve._right_dv[:m].copy()
    p2t = curve._left_dt[1:].copy() if n > 1 else np.zeros(0)
    p2v = curve._left_dv[1:].copy() if n > 1 else np.zeros(0)
    corrected = np.zeros(m, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        over1 = (p1t > span) & (span > 0)
        f1 = np.where(over1, span / np.where(over1, p1t, 1.0), 1.0)
        over2 = (-p2t > span) & (span > 0)
        f2 = np.where(over2, span / np.where(over2, -p2t, 1.0), 1.0)
    p1t, p1v = p1t * f1, p1v * f1
    p2t, p2v = p2t * f2, p2v * f2
    corrected |= (over1 | over2) & (kind == _SEG_BEZIER)
    c.corrected = _frozen(corrected)
    v0, v1 = v[:m], v[1:] if n > 1 else np.zeros(0)

    # Power basis: P(u) = ((a u + b) u + c) u + d for the four control values.
    def basis(q0, q1, q2, q3):
        return np.stack(
            [
                -q0 + 3.0 * q1 - 3.0 * q2 + q3,
                3.0 * q0 - 6.0 * q1 + 3.0 * q2,
                -3.0 * q0 + 3.0 * q1,
                q0,
            ],
            axis=1,
        )

    # x relative to the segment start (better conditioned); y absolute.
    zeros = np.zeros(m)
    c.x = _frozen(basis(zeros, p1t, span + p2t, span))
    # Hermite segments use the slopes on a third-of-interval time extent (unweighted keys
    # are stored that way already; the slope is what counts).
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(p1t != 0.0, p1v / np.where(p1t != 0.0, p1t, 1.0), 0.0)
        s2 = np.where(p2t != 0.0, p2v / np.where(p2t != 0.0, p2t, 1.0), 0.0)
    herm = kind == _SEG_HERMITE
    q1 = np.where(herm, v0 + s1 * span / 3.0, v0 + p1v)
    q2 = np.where(herm, v1 - s2 * span / 3.0, v1 + p2v)
    c.y = _frozen(basis(v0, q1, q2, v1))
    c.easing = _frozen(
        np.where(interp >= _FIRST_EASING, interp - _FIRST_EASING, 0).astype(np.uint8)
    )
    c.mode = _frozen(curve._ease[:m].copy())
    c.back = curve._back[:m]
    c.amplitude = curve._amplitude[:m]
    c.period = curve._period[:m]

    def end_slope(at_start: bool) -> float:
        if n < 2:
            dt = curve._left_dt[0] if at_start else curve._right_dt[0]
            dv = curve._left_dv[0] if at_start else curve._right_dv[0]
            return float(dv / dt) if dt != 0.0 else 0.0
        k = 0 if at_start else m - 1
        if span[k] == 0.0:
            return 0.0
        if interp[k] == BEZIER:
            i = 0 if at_start else n - 1
            dt = curve._left_dt[i] if at_start else curve._right_dt[i]
            dv = curve._left_dv[i] if at_start else curve._right_dv[i]
            if dt != 0.0:
                return float(dv / dt)
            return float(_bezier_end_slope(c, k, at_start))
        return float((v[k + 1] - v[k]) / span[k]) if interp[k] != CONSTANT else 0.0

    c.slope_before = end_slope(True)
    c.slope_after = end_slope(False)
    return c


def _bezier_end_slope(c: _Compiled, k: int, at_start: bool) -> float:
    xa, xb, xc, _ = c.x[k]
    ya, yb, yc, _ = c.y[k]
    if at_start:
        dx, dy = xc, yc
    else:
        dx, dy = 3 * xa + 2 * xb + xc, 3 * ya + 2 * yb + yc
    return dy / dx if dx != 0 else 0.0


def _horner(coef: np.ndarray, u: np.ndarray, order: int) -> list[np.ndarray]:
    """``P, P', P'', P'''`` of the power-basis cubics ``coef`` (rows a, b, c, d) at ``u``."""
    a, b, c, d = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3]
    out = [((a * u + b) * u + c) * u + d]
    if order >= 1:
        out.append((3.0 * a * u + 2.0 * b) * u + c)
    if order >= 2:
        out.append(6.0 * a * u + 2.0 * b)
    if order >= 3:
        out.append(6.0 * a)
    return out


def solve_bezier_time(x: np.ndarray, target: np.ndarray, span: np.ndarray) -> np.ndarray:
    """The parameter ``u`` in ``[0, 1]`` with ``x(u) = target`` for monotone cubics ``x``
    (power basis rows, ``x(0) = 0``, ``x(1) = span``): Newton's method safeguarded by
    bisection, per element, at most :data:`NEWTON_ITERATIONS` steps, stopping when the
    bracket collapses or the residual is within 4 ulp of the span. Only ``+ - * /`` and
    comparisons: deterministic everywhere."""
    a, b, c, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.clip(np.where(span > 0, target / np.where(span > 0, span, 1.0), 0.0), 0.0, 1.0)
    lo = np.zeros_like(u)
    hi = np.ones_like(u)
    tol = 4.0 * np.finfo(np.float64).eps * np.maximum(span, 1e-300)
    done = span <= 0.0
    eps4 = 4.0 * np.finfo(np.float64).eps
    for _ in range(NEWTON_ITERATIONS):
        if np.all(done):
            break
        f = ((a * u + b) * u + c) * u + d - target
        deriv = (3.0 * a * u + 2.0 * b) * u + c
        # Converged: an exact hit, or a residual within a few ulp whose Newton correction
        # is negligible too (where x' vanishes -- a vertical tangent in time -- a small
        # residual says little about u, and bisection goes on to collapse the bracket).
        conv = (f == 0.0) | ((np.abs(f) <= tol) & (np.abs(f) <= eps4 * np.abs(deriv)))
        done = done | conv
        lo = np.where(~done & (f < 0.0), u, lo)
        hi = np.where(~done & (f > 0.0), u, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = u - f / deriv
        inside = (deriv > 0.0) & (newton > lo) & (newton < hi)
        nxt = np.where(inside, newton, 0.5 * (lo + hi))
        collapse = (hi - lo) <= 2.0 * np.finfo(np.float64).eps
        u = np.where(done, u, nxt)
        done = done | collapse
    return u


def _segment_values(c: _Compiled, k: np.ndarray, t: np.ndarray, order: int) -> list[np.ndarray]:
    """Values (and time derivatives) of segments ``k`` at times ``t`` inside them."""
    kind = c.kind[k]
    t0 = c.t[k]
    span = c.span[k]
    v0 = c.v[k]
    v1 = c.v[k + 1]
    out = [np.empty_like(t) for _ in range(order + 1)]
    for o in out[1:]:
        o.fill(0.0)

    # constant (and jumps, which are never evaluated inside: they have no interior)
    sel = (kind == _SEG_CONSTANT) | (kind == _SEG_JUMP)
    out[0][sel] = v0[sel]

    sel = kind == _SEG_LINEAR
    if np.any(sel):
        u = (t[sel] - t0[sel]) / span[sel]
        out[0][sel] = v0[sel] + (v1[sel] - v0[sel]) * u
        if order >= 1:
            out[1][sel] = (v1[sel] - v0[sel]) / span[sel]

    sel = kind == _SEG_HERMITE
    if np.any(sel):
        u = (t[sel] - t0[sel]) / span[sel]
        ys = _horner(c.y[k[sel]], u, order)
        out[0][sel] = ys[0]
        scale = 1.0 / span[sel]
        factor = scale.copy()
        for o in range(1, order + 1):
            out[o][sel] = ys[o] * factor
            factor = factor * scale

    sel = kind == _SEG_BEZIER
    if np.any(sel):
        ks = k[sel]
        u = solve_bezier_time(c.x[ks], t[sel] - t0[sel], span[sel])
        ys = _horner(c.y[ks], u, order)
        out[0][sel] = ys[0]
        if order >= 1:
            xs = _horner(c.x[ks], u, order)
            q = xs[1]
            p = ys[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[1][sel] = np.where(q != 0.0, p / q, np.copysign(np.inf, p) * (p != 0.0))
                if order >= 2:
                    pp, qq = ys[2], xs[2]
                    out[2][sel] = (pp * q - p * qq) / (q * q * q)
                if order >= 3:
                    ppp, qqq = ys[3], xs[3]
                    out[3][sel] = (
                        ppp * q * q - p * qqq * q - 3.0 * qq * pp * q + 3.0 * p * qq * qq
                    ) / (q**5)

    sel = kind == _SEG_EASE
    if np.any(sel):
        ks = k[sel]
        u = np.clip((t[sel] - t0[sel]) / span[sel], 0.0, 1.0)
        dv = v1[sel] - v0[sel]
        eas = c.easing[ks]
        mod = c.mode[ks]
        res = [np.empty_like(u) for _ in range(order + 1)]
        for e in np.unique(eas).tolist():
            for md in np.unique(mod[eas == e]).tolist():
                m = (eas == e) & (mod == md)
                ev = _easing.evaluate(
                    e, md, u[m], order, c.back[ks][m], c.amplitude[ks][m], c.period[ks][m]
                )
                for o in range(order + 1):
                    res[o][m] = ev[o]
        out[0][sel] = np.where(u == 0.0, v0[sel], v0[sel] + dv * res[0])  # exact at keys
        scale = 1.0 / span[sel]
        factor = dv * scale
        for o in range(1, order + 1):
            out[o][sel] = res[o] * factor
            factor = factor * scale
    return out


def _evaluate(c: _Compiled, curve: FCurve, t: np.ndarray, order: int) -> list[np.ndarray]:
    """Evaluate keys + interpolation + extrapolation at ``t`` (no modifiers)."""
    times = c.t
    n = times.shape[0]
    first, last = float(times[0]), float(times[-1])
    out = [np.empty_like(t) for _ in range(order + 1)]
    for o in out[1:]:
        o.fill(0.0)
    tt = t.copy()
    offset = np.zeros_like(t)  # value offset (cycle_offset)
    sign = np.ones_like(t)  # time direction (mirror)
    period = last - first
    before = t < first
    after = t > last
    for mask, mode in ((before, curve._before), (after, curve._after)):
        if not np.any(mask) or mode < 2:
            continue
        rel = (t[mask] - first) / period
        cycle = np.floor(rel)
        local = t[mask] - first - cycle * period
        local = np.clip(local, 0.0, period)
        if mode == 4:  # mirror: odd cycles run backwards
            odd = (np.abs(cycle) % 2.0) == 1.0
            tt[mask] = np.where(odd, last - local, first + local)
            sign[mask] = np.where(odd, -1.0, 1.0)
        else:
            tt[mask] = first + local
            if mode == 3:
                offset[mask] = cycle * (float(c.v[-1]) - float(c.v[0]))
    inside = (tt >= first) & (tt <= last)
    if n == 1:
        out[0][:] = c.v[0]
    else:
        idx = np.searchsorted(times, tt[inside], side="right") - 1
        idx = np.clip(idx, 0, n - 2)
        # At the last key the value is the last key's (its segment is the last one).
        seg = _segment_values(c, idx, tt[inside], order)
        at_last = tt[inside] == last
        seg[0] = np.where(at_last, c.v[-1], seg[0])
        if order >= 1 and np.any(at_last):
            # Derivatives at the last key: the left limit (end of the last segment).
            k_last = np.full(int(np.count_nonzero(at_last)), n - 2)
            lim = _segment_values(c, k_last, tt[inside][at_last], order)
            for o in range(1, order + 1):
                seg[o][at_last] = lim[o]
        for o in range(order + 1):
            out[o][inside] = seg[o]
    # Hold / linear extrapolation outside.
    for mask, mode, end in ((t < first, curve._before, 0), (t > last, curve._after, 1)):
        if not np.any(mask) or mode >= 2:
            continue
        key_v = float(c.v[0] if end == 0 else c.v[-1])
        key_t = first if end == 0 else last
        if mode == 1:
            slope = c.slope_before if end == 0 else c.slope_after
            out[0][mask] = key_v + slope * (t[mask] - key_t)
            if order >= 1:
                out[1][mask] = slope
            for o in range(2, order + 1):
                out[o][mask] = 0.0
        else:
            out[0][mask] = key_v
            for o in range(1, order + 1):
                out[o][mask] = 0.0
    if np.any(offset):
        out[0] = out[0] + offset
    if np.any(sign < 0):
        for o in range(1, order + 1, 2):
            out[o] = out[o] * sign
    return out


# --------------------------------------------------------------------------------------- #
# Extrapolation and modifiers (parsing)
# --------------------------------------------------------------------------------------- #


def _extrapolation(value: Any, location: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise _err("INVALID_EXTRAPOLATION", "give one mode or [before, after]", location)
        a = _index(EXTRAPOLATIONS, value[0], f"{location}[0]", "INVALID_EXTRAPOLATION")
        b = _index(EXTRAPOLATIONS, value[1], f"{location}[1]", "INVALID_EXTRAPOLATION")
        return a, b
    k = _index(EXTRAPOLATIONS, value, location, "INVALID_EXTRAPOLATION")
    return k, k


def _modifiers(value: Any, location: str) -> tuple:
    from .modifiers import parse_modifiers

    return parse_modifiers(value, location)
