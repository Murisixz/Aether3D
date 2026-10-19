"""Clips and keyframe editing: the programmable graph-editor operations.

A :class:`Clip` is reusable animation data: channels keyed by property path (scalar
:class:`~.curves.FCurve`s and quaternion :class:`~.tracks.RotationTrack`s) and **events**
(named times with JSON data: contacts, footsteps, sound cues). Clips are immutable;
every edit returns a new clip, and paths are resolved through :mod:`.paths`, so a clip
holds only animatable properties of its target kind (``object`` or ``material``).

The key operations (:func:`edit_curve`) are pure functions on curves. Keys are selected
by indices, by a time range ``{"from": t0, "to": t1}`` (inclusive), or ``"all"``:

* ``insert`` (time, value; a key at that exact time gets the new value), ``delete``,
  ``set`` (value, time, interpolation, ease, back/amplitude/period, weighted),
  ``handle`` (one side: type, ``dt``/``dv`` or ``slope``; an aligned pair turns
  together), ``handle_type``;
* ``move`` (``dt``, ``dv``), ``scale_time`` / ``scale_values`` (``factor``, ``pivot``),
  ``reverse`` (time mirrored about a pivot: handles swap sides, ``in``/``out`` easings
  swap), ``quantize`` (snap times to a frame rate), ``distribute`` (evenly between two
  times), ``retime`` (a piecewise-linear time map; ``velocity: "scale"`` keeps the
  shape in the new time, ``"preserve"`` keeps the slopes);
* ``interpolation`` / ``easing`` on a selection; ``extrapolation``; ``modifiers``;
  ``smoothing`` (how the channel's automatic handles are computed);
* timing (0.20.0): ``insert`` with ``keep_shape`` (de Casteljau: the curve does not
  change), ``ease`` (``ease_in``/``ease_out``: flat handles over that fraction of the
  interval, ``slope`` for a moving key), ``anticipate`` (a counter-move of ``amount`` of
  the change at fraction ``at`` of the move), ``overshoot`` (follow-through: ``amount``
  beyond the end, settling over ``settle`` seconds in ``oscillations`` swings shrinking by
  ``decay``), ``hold`` (a moving hold of ``duration``, ``drift`` toward the next key,
  inserting time unless ``shift`` is false), ``smooth`` (key values averaged by a
  Gaussian of ``sigma`` seconds; the ends kept).

Scaling time scales weighted handles' extents with it, so slopes (velocities) scale by
``1 / factor`` -- the motion is the same, faster or slower -- unless ``velocity:
"preserve"`` asks to keep them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import easing as _easing
from . import paths as P
from .curves import (
    ALIGNED,
    HANDLE_TYPES,
    INTERPOLATIONS,
    FCurve,
)
from .time import FrameRate
from .tracks import RotationTrack

MAX_CHANNELS = 100_000
MAX_EVENTS = 100_000
MAX_EVENT_DATA = 16_384  # bytes of JSON per event


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _num(value: Any, location: str, what: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err("INVALID_ARGUMENT", f"{what} must be a number, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v):
        raise _err("INVALID_ARGUMENT", f"{what} must be finite, got {value!r}", location)
    return v


@dataclass(frozen=True)
class Event:
    """A named time in a clip (or on the timeline) with optional JSON data."""

    time: float
    name: str
    data: str = "null"  # canonical JSON text

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"time": self.time, "name": self.name}
        if self.data != "null":
            out["data"] = json.loads(self.data)
        return out


def parse_event(raw: Any, location: str) -> Event:
    if isinstance(raw, Event):
        return raw
    if (
        not isinstance(raw, dict)
        or set(raw) - {"time", "name", "data"}
        or "time" not in raw
        or "name" not in raw
    ):
        raise _err("INVALID_EVENT", "an event is {time, name, data}", location)
    t = _num(raw["time"], f"{location}.time", "time")
    name = P.check_name(raw["name"], "an event name", f"{location}.name")
    try:
        data = json.dumps(raw.get("data"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise _err("INVALID_EVENT", "event data must be finite JSON", f"{location}.data") from None
    if len(data) > MAX_EVENT_DATA:
        raise _err(
            "LIMIT_EXCEEDED", f"event data beyond {MAX_EVENT_DATA} bytes", f"{location}.data"
        )
    return Event(t, name, data)


Channel = FCurve | RotationTrack


@dataclass(frozen=True)
class Clip:
    """Immutable animation data (see the module docstring). ``channels`` is sorted by
    path; ``range`` is explicit or ``None`` (the keys' extent)."""

    target: str = P.OBJECT
    channels: tuple[tuple[str, Channel], ...] = ()
    events: tuple[Event, ...] = ()
    range: tuple[float, float] | None = None
    metadata: str = "{}"
    _index: dict[str, int] = field(default=None, compare=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {p: i for i, (p, _) in enumerate(self.channels)})

    # -- reading ---------------------------------------------------------------------------

    def channel(self, path: str) -> Channel | None:
        i = self._index.get(path)
        return None if i is None else self.channels[i][1]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(p for p, _ in self.channels)

    def key_range(self) -> tuple[float, float] | None:
        lo, hi = math.inf, -math.inf
        for _, ch in self.channels:
            a, b = ch.range
            lo, hi = min(lo, a), max(hi, b)
        for e in self.events:
            lo, hi = min(lo, e.time), max(hi, e.time)
        return None if lo > hi else (lo, hi)

    def frame_range(self) -> tuple[float, float]:
        """The explicit range, else the keys' extent, else ``(0, 0)``."""
        if self.range is not None:
            return self.range
        r = self.key_range()
        return (0.0, 0.0) if r is None else r

    # -- editing (new clips) ---------------------------------------------------------------

    def with_channel(self, path: str, channel: Channel | None, location: str = "clip") -> Clip:
        spec = P.parse(path, self.target, location=f"{location}.channels")
        assert not isinstance(spec, list)
        if channel is not None:
            want = RotationTrack if spec.kind == P.ROTATION else FCurve
            if not isinstance(channel, want):
                raise _err(
                    "INVALID_CHANNEL",
                    f"{spec.path} takes a {'rotation track' if want is RotationTrack else 'curve'}",
                    f"{location}.channels[{spec.path}]",
                )
        items = dict(self.channels)
        if channel is None:
            items.pop(spec.path, None)
        else:
            items[spec.path] = channel
        if len(items) > MAX_CHANNELS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_CHANNELS} channels", location)
        _check_rotation_modes(items, location)
        return Clip(
            self.target, tuple(sorted(items.items())), self.events, self.range, self.metadata
        )

    def with_events(self, events: Iterable[Any], location: str = "clip") -> Clip:
        evs = tuple(
            sorted(
                (parse_event(e, f"{location}.events[{i}]") for i, e in enumerate(events)),
                key=lambda e: (e.time, e.name),
            )
        )
        if len(evs) > MAX_EVENTS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_EVENTS} events", location)
        return Clip(self.target, self.channels, evs, self.range, self.metadata)

    def with_range(self, rng: Any, location: str = "clip") -> Clip:
        return Clip(
            self.target, self.channels, self.events, _range(rng, f"{location}.range"), self.metadata
        )

    def map_channels(self, fn) -> Clip:
        """A clip with ``fn(path, channel)`` applied to every channel (``None`` drops it)."""
        items = []
        for path, ch in self.channels:
            new = fn(path, ch)
            if new is not None:
                items.append((path, new))
        return Clip(self.target, tuple(items), self.events, self.range, self.metadata)

    # -- serialization ---------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "target": self.target,
            "channels": {
                path: (
                    {"rotation": ch.to_json()} if isinstance(ch, RotationTrack) else ch.to_json()
                )
                for path, ch in self.channels
            },
        }
        if self.events:
            out["events"] = [e.to_json() for e in self.events]
        if self.range is not None:
            out["range"] = list(self.range)
        if self.metadata != "{}":
            out["metadata"] = json.loads(self.metadata)
        return out

    @classmethod
    def from_json(cls, data: Any, location: str = "clip") -> Clip:
        if not isinstance(data, dict):
            raise _err("INVALID_CLIP", "a clip is an object", location)
        unknown = set(data) - {"target", "channels", "events", "range", "metadata"}
        if unknown:
            raise _err("INVALID_CLIP", f"unknown clip fields {sorted(unknown)}", location)
        target = data.get("target", P.OBJECT)
        if target not in (P.OBJECT, P.MATERIAL):
            raise _err("INVALID_CLIP", "target is 'object' or 'material'", f"{location}.target")
        clip = Clip(target)
        channels = data.get("channels", {})
        if not isinstance(channels, dict):
            raise _err(
                "INVALID_CLIP", "channels is an object keyed by path", f"{location}.channels"
            )
        items: dict[str, Channel] = {}
        for path, raw in channels.items():
            loc = f"{location}.channels[{path}]"
            spec = P.parse(path, target, location=loc)
            assert not isinstance(spec, list)
            if spec.kind == P.ROTATION:
                if not isinstance(raw, dict) or set(raw) != {"rotation"}:
                    raise _err("INVALID_CHANNEL", f"{spec.path} is {{rotation: track}}", loc)
                items[spec.path] = RotationTrack.from_json(raw["rotation"], loc)
            else:
                items[spec.path] = FCurve.from_json(raw, loc)
        if len(items) > MAX_CHANNELS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_CHANNELS} channels", location)
        _check_rotation_modes(items, location)
        clip = Clip(target, tuple(sorted(items.items())))
        if "events" in data:
            if not isinstance(data["events"], list):
                raise _err("INVALID_CLIP", "events is a list", f"{location}.events")
            clip = clip.with_events(data["events"], location)
        if data.get("range") is not None:
            clip = clip.with_range(data["range"], location)
        if "metadata" in data:
            try:
                meta = json.dumps(
                    data["metadata"], sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            except (TypeError, ValueError):
                raise _err(
                    "INVALID_CLIP", "metadata must be JSON", f"{location}.metadata"
                ) from None
            if not isinstance(data["metadata"], dict) or len(meta) > MAX_EVENT_DATA:
                raise _err(
                    "INVALID_CLIP", "metadata is a small JSON object", f"{location}.metadata"
                )
            clip = Clip(clip.target, clip.channels, clip.events, clip.range, meta)
        return clip

    def same(self, other: Clip) -> bool:
        if self is other:
            return True
        if not isinstance(other, Clip) or (self.target, self.events, self.range, self.metadata) != (
            other.target,
            other.events,
            other.range,
            other.metadata,
        ):
            return False
        if self.paths != other.paths:
            return False
        return all(a.same(b) for (_, a), (_, b) in zip(self.channels, other.channels, strict=True))


def _range(rng: Any, location: str) -> tuple[float, float] | None:
    if rng is None:
        return None
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise _err("INVALID_RANGE", "a range is [start, end]", location)
    a, b = _num(rng[0], f"{location}[0]", "start"), _num(rng[1], f"{location}[1]", "end")
    if not a < b:
        raise _err("INVALID_RANGE", "a range's start must be before its end", location)
    return (a, b)


def _check_rotation_modes(items: dict[str, Channel], location: str) -> None:
    """One rotation representation per object or joint within a clip."""
    quats = {p.rsplit(".", 1)[0] for p in items if p.endswith(".rotation")}
    eulers = {p.rsplit(".", 2)[0] for p in items if ".rotation_euler." in p}
    both = quats & eulers
    if both:
        owner = sorted(both)[0]
        raise _err(
            "ROTATION_MODE_CONFLICT",
            f"{owner} is animated by both a quaternion track and Euler curves in one clip",
            location,
        )


# --------------------------------------------------------------------------------------- #
# Selections and key operations
# --------------------------------------------------------------------------------------- #


def select(curve: FCurve | RotationTrack, selection: Any, location: str = "keys") -> np.ndarray:
    """Key indices of a selection: ``"all"``, a list of indices (negative from the end),
    or ``{"from": t0, "to": t1}`` (times, inclusive)."""
    n = len(curve)
    if selection is None or selection == "all":
        return np.arange(n)
    if isinstance(selection, dict):
        if set(selection) - {"from", "to"}:
            raise _err("INVALID_SELECTION", "a time selection is {from, to}", location)
        lo = (
            _num(selection.get("from", -math.inf), f"{location}.from", "from")
            if "from" in selection
            else -math.inf
        )
        hi = (
            _num(selection.get("to", math.inf), f"{location}.to", "to")
            if "to" in selection
            else math.inf
        )
        t = curve.times
        return np.flatnonzero((t >= lo) & (t <= hi))
    if isinstance(selection, (int, np.integer)) and not isinstance(selection, bool):
        selection = [selection]
    if not isinstance(selection, (list, tuple)) or not selection:
        raise _err("INVALID_SELECTION", "select keys by 'all', indices or {from, to}", location)
    out = []
    for i, k in enumerate(selection):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not -n <= int(k) < n:
            raise _err(
                "INVALID_SELECTION", f"no key {k!r} (the channel has {n})", f"{location}[{i}]"
            )
        out.append(int(k) % n)
    return np.unique(np.array(out, np.int64))


def _key_defaults(curve: FCurve, time: float) -> dict[str, Any]:
    """Interpolation and handle types for a new key: the previous key's, else the next's."""
    t = curve.times
    i = int(np.searchsorted(t, time, side="right")) - 1
    i = min(max(i, 0), len(curve) - 1)
    return {
        "interpolation": int(curve.interpolation[i]),
        "left_type": int(curve.array("left_type")[i]),
        "right_type": int(curve.array("right_type")[i]),
        "ease": int(curve.array("ease")[i]),
    }


def insert_key(
    curve: FCurve | None, time: float, value: float, location: str = "keys", **options: Any
) -> FCurve:
    """``curve`` with a key at ``time`` (a new curve when ``curve`` is ``None``)."""
    time = _num(time, f"{location}.time", "time")
    value = _num(value, f"{location}.value")
    if curve is None:
        row = {"time": time, "value": value, **options}
        return FCurve.from_keys([row], location=location)
    a = curve.arrays()
    hit = np.flatnonzero(a["times"] == time)
    if hit.size:
        k = int(hit[-1])
        a["values"][k] = value
        new = curve.replace(a, location=location)
        return _apply_key_options(new, [k], options, location) if options else new
    d = _key_defaults(curve, time)
    k = int(np.searchsorted(a["times"], time, side="right"))
    row = {
        "times": time,
        "values": value,
        "interpolation": d["interpolation"],
        "ease": d["ease"],
        "left_type": d["left_type"],
        "right_type": d["right_type"],
        "left_dt": np.nan,
        "left_dv": np.nan,
        "right_dt": np.nan,
        "right_dv": np.nan,
        "weighted": False,
        "back": _easing.DEFAULT_BACK,
        "amplitude": _easing.DEFAULT_AMPLITUDE,
        "period": _easing.DEFAULT_PERIOD,
    }
    for name, v in row.items():
        a[name] = np.insert(a[name], k, v)
    new = curve.replace(a, location=location)
    return _apply_key_options(new, [k], options, location) if options else new


def _apply_key_options(
    curve: FCurve, keys: list[int], options: dict[str, Any], location: str
) -> FCurve:
    """Set per-key fields from a key dict (interpolation, ease, handles, ...)."""
    rows = curve.keys()
    for k in keys:
        row = rows[k]
        for name, value in options.items():
            if name in ("left", "right") and isinstance(value, dict) and "type" not in value:
                value = {**value, "type": "free"}
            row[name] = value
    return curve.with_keys(rows, location=location)


def edit_curve(curve: FCurve, op: str, args: dict[str, Any], location: str = "operation") -> FCurve:
    """Apply one key operation (see the module docstring) and return the new curve."""
    n = len(curve)
    if op == "insert":
        _only(args, {"time", "value", "keep_shape", *_KEY_FIELDS}, location)
        time = _num(args.get("time"), f"{location}.time", "time")
        keep = args.get("keep_shape", False)
        if not isinstance(keep, bool):
            raise _err("INVALID_ARGUMENT", "keep_shape is true or false", f"{location}.keep_shape")
        options = {k: v for k, v in args.items() if k in _KEY_FIELDS}
        if keep:
            if "value" in args or options:
                raise _err(
                    "INVALID_ARGUMENT",
                    "a shape-keeping key takes its value and handles from the curve",
                    location,
                )
            return insert_keeping_shape(curve, time, location)
        value = args.get("value")
        value = float(curve.evaluate(time)) if value is None else value
        return insert_key(curve, time, value, location, **options)
    if op == "smoothing":
        _only(args, {"smoothing"}, location)
        return curve.replace(smoothing=args.get("smoothing"), location=location)
    sel = select(curve, args.get("keys", "all"), f"{location}.keys")
    if op == "ease":
        _only(args, {"keys", "ease_in", "ease_out", "slope"}, location)
        return ease_keys(curve, sel, args, location)
    if op == "anticipate":
        _only(args, {"keys", "amount", "at"}, location)
        return anticipate(curve, sel, args, location)
    if op == "overshoot":
        _only(args, {"keys", "amount", "settle", "oscillations", "decay"}, location)
        return overshoot(curve, sel, args, location)
    if op == "hold":
        _only(args, {"keys", "duration", "drift", "shift"}, location)
        return hold(curve, sel, args, location)
    if op == "smooth":
        _only(args, {"keys", "sigma", "keep_ends"}, location)
        return smooth_values(curve, sel, args, location)
    if op == "delete":
        if sel.size >= n:
            raise _err(
                "INVALID_SELECTION",
                "a channel keeps at least one key (remove the channel instead)",
                location,
            )
        keep = np.setdiff1d(np.arange(n), sel)
        a = {k: v[keep] for k, v in curve.arrays().items()}
        return curve.replace(a, location=location)
    if op == "set":
        allowed = {
            "keys",
            "time",
            "value",
            "interpolation",
            "ease",
            "back",
            "amplitude",
            "period",
            "weighted",
        }
        _only(args, allowed, location)
        fields = {k: v for k, v in args.items() if k != "keys"}
        if "time" in fields and sel.size != 1:
            raise _err(
                "INVALID_SELECTION", "set a time on one key at a time (or use move)", location
            )
        return _apply_key_options(curve, sel.tolist(), fields, location)
    if op == "handle":
        _only(args, {"keys", "side", "type", "dt", "dv", "slope", "weighted"}, location)
        side = args.get("side")
        if side not in ("left", "right", "both"):
            raise _err("INVALID_HANDLE", "side is left, right or both", f"{location}.side")
        return _set_handles(curve, sel, side, args, location)
    if op == "handle_type":
        _only(args, {"keys", "side", "type"}, location)
        side = args.get("side", "both")
        kind = args.get("type")
        if kind not in HANDLE_TYPES:
            raise _err(
                "INVALID_HANDLE", f"type is one of {', '.join(HANDLE_TYPES)}", f"{location}.type"
            )
        a = curve.arrays()
        for s in ("left", "right") if side == "both" else (side,):
            if s not in ("left", "right"):
                raise _err("INVALID_HANDLE", "side is left, right or both", f"{location}.side")
            a[f"{s}_type"][sel] = HANDLE_TYPES.index(kind)
        return curve.replace(a, location=location)
    if op == "interpolation":
        _only(args, {"keys", "interpolation", "ease", "back", "amplitude", "period"}, location)
        fields = {k: v for k, v in args.items() if k != "keys"}
        if fields.get("interpolation") not in INTERPOLATIONS:
            raise _err(
                "INVALID_INTERPOLATION",
                f"interpolation is one of {', '.join(INTERPOLATIONS)}",
                location,
            )
        return _apply_key_options(curve, sel.tolist(), fields, location)
    if op == "move":
        _only(args, {"keys", "dt", "dv"}, location)
        a = curve.arrays()
        a["times"][sel] = a["times"][sel] + _num(args.get("dt", 0.0), f"{location}.dt", "dt")
        a["values"][sel] = a["values"][sel] + _num(args.get("dv", 0.0), f"{location}.dv", "dv")
        return curve.replace(a, location=location)
    if op == "scale_time":
        _only(args, {"keys", "factor", "pivot", "velocity"}, location)
        f = _num(args.get("factor"), f"{location}.factor", "factor")
        if f == 0.0:
            raise _err("INVALID_ARGUMENT", "a time scale factor cannot be 0", f"{location}.factor")
        pivot = _num(args.get("pivot", 0.0), f"{location}.pivot", "pivot")
        out = _scale_time(curve, sel, f, pivot, args.get("velocity", "scale"), location)
        return out
    if op == "scale_values":
        _only(args, {"keys", "factor", "pivot"}, location)
        f = _num(args.get("factor"), f"{location}.factor", "factor")
        pivot = _num(args.get("pivot", 0.0), f"{location}.pivot", "pivot")
        a = curve.arrays()
        a["values"][sel] = pivot + (a["values"][sel] - pivot) * f
        a["left_dv"][sel] *= f
        a["right_dv"][sel] *= f
        return curve.replace(a, location=location)
    if op == "reverse":
        _only(args, {"keys", "pivot"}, location)
        if sel.size != n:
            raise _err("INVALID_SELECTION", "reverse applies to every key of a channel", location)
        lo, hi = curve.range
        pivot = _num(args.get("pivot", 0.5 * (lo + hi)), f"{location}.pivot", "pivot")
        return reverse(curve, pivot, location)
    if op == "quantize":
        _only(args, {"keys", "fps"}, location)
        rate = FrameRate(args.get("fps", 24), f"{location}.fps")
        a = curve.arrays()
        a["times"][sel] = [rate.snap(t) for t in a["times"][sel].tolist()]
        return _dedupe(curve, a, location)
    if op == "distribute":
        _only(args, {"keys", "start", "end"}, location)
        if sel.size < 2:
            raise _err("INVALID_SELECTION", "distribute needs two keys or more", location)
        a = curve.arrays()
        start = _num(args.get("start", a["times"][sel[0]]), f"{location}.start", "start")
        end = _num(args.get("end", a["times"][sel[-1]]), f"{location}.end", "end")
        if not start < end:
            raise _err("INVALID_ARGUMENT", "distribute needs start < end", location)
        a["times"][sel] = np.linspace(start, end, sel.size)
        return curve.replace(a, location=location)
    if op == "retime":
        _only(args, {"keys", "map", "velocity"}, location)
        return retime(curve, args.get("map"), args.get("velocity", "scale"), f"{location}")
    raise _err("UNKNOWN_OPERATION", f"unknown key operation {op!r}", location)


_KEY_FIELDS = {
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

KEY_OPERATIONS = (
    "insert",
    "set",
    "delete",
    "handle",
    "handle_type",
    "interpolation",
    "move",
    "scale_time",
    "scale_values",
    "reverse",
    "quantize",
    "distribute",
    "retime",
    "ease",
    "anticipate",
    "overshoot",
    "hold",
    "smooth",
    "smoothing",
)


def _only(args: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(args) - allowed
    if unknown:
        raise _err("INVALID_ARGUMENT", f"unknown arguments {sorted(unknown)}", location)


def _set_handles(
    curve: FCurve, sel: np.ndarray, side: str, args: dict[str, Any], location: str
) -> FCurve:
    a = curve.arrays()
    kind = args.get("type")
    if kind is not None and kind not in HANDLE_TYPES:
        raise _err(
            "INVALID_HANDLE", f"type is one of {', '.join(HANDLE_TYPES)}", f"{location}.type"
        )
    if "dv" in args and "slope" in args:
        raise _err("INVALID_HANDLE", "give dv or slope, not both", location)
    if "weighted" in args:
        if not isinstance(args["weighted"], bool):
            raise _err("INVALID_HANDLE", "weighted is true or false", f"{location}.weighted")
        a["weighted"][sel] = args["weighted"]
    sides = ("left", "right") if side == "both" else (side,)
    for s in sides:
        sign = -1.0 if s == "left" else 1.0
        types = a[f"{s}_type"]
        if kind is not None:
            types[sel] = HANDLE_TYPES.index(kind)
        elif "dt" in args or "dv" in args or "slope" in args:
            # Setting an offset makes an automatic handle free (aligned stays aligned).
            auto = (types[sel] != ALIGNED) & (types[sel] != 0)
            types[sel[auto]] = 0
        if "dt" in args:
            dt = _num(args["dt"], f"{location}.dt", "dt")
            if dt * sign < 0:
                raise _err(
                    "INVALID_HANDLE",
                    f"a {s} handle must point {'back' if s == 'left' else 'forward'} in time",
                    f"{location}.dt",
                )
            if side == "both" and s == "left":
                dt = -abs(dt)
            a[f"{s}_dt"][sel] = dt
        if "slope" in args:
            slope = _num(args["slope"], f"{location}.slope", "slope")
            a[f"{s}_dv"][sel] = slope * a[f"{s}_dt"][sel]
        elif "dv" in args:
            dv = _num(args["dv"], f"{location}.dv", "dv")
            a[f"{s}_dv"][sel] = -dv if (side == "both" and s == "left") else dv
    # An aligned pair turns together: the other side takes the edited side's slope.
    if side in ("left", "right"):
        other = "right" if side == "left" else "left"
        pair = sel[(a["left_type"][sel] == ALIGNED) & (a["right_type"][sel] == ALIGNED)]
        if pair.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = a[f"{side}_dv"][pair] / a[f"{side}_dt"][pair]
            ok = np.isfinite(slope)
            a[f"{other}_dv"][pair[ok]] = slope[ok] * a[f"{other}_dt"][pair[ok]]
    return curve.replace(a, location=location)


def _scale_time(
    curve: FCurve, sel: np.ndarray, f: float, pivot: float, velocity: Any, location: str
) -> FCurve:
    if velocity not in ("scale", "preserve"):
        raise _err("INVALID_ARGUMENT", "velocity is 'scale' or 'preserve'", f"{location}.velocity")
    a = curve.arrays()
    a["times"][sel] = pivot + (a["times"][sel] - pivot) * f
    if f < 0:  # reversed selection: handles swap sides
        if sel.size != len(curve):
            raise _err(
                "INVALID_ARGUMENT", "a negative factor reverses: apply it to every key", location
            )
        return reverse(
            curve.replace(a, location=location, recompute=False),
            pivot,
            location,
            already_mirrored=True,
        )
    for s in ("left", "right"):
        a[f"{s}_dt"][sel] *= f
        if velocity == "preserve":
            a[f"{s}_dv"][sel] *= f
    return curve.replace(a, location=location)


def reverse(
    curve: FCurve, pivot: float, location: str = "keys", *, already_mirrored: bool = False
) -> FCurve:
    """Time mirrored about ``pivot``: key ``j`` of the result is key ``n-1-j``; its
    segment takes the interpolation of the segment it was; handles swap sides; easing
    ``in`` and ``out`` swap."""
    a = curve.arrays()
    n = len(curve)
    order = np.arange(n)[::-1]
    b = {k: v[order].copy() for k, v in a.items()}
    if not already_mirrored:
        b["times"] = 2.0 * pivot - b["times"]
    # Handles: the new left handle is the old right one mirrored in time.
    b["left_dt"], b["right_dt"] = -b["right_dt"], -b["left_dt"]
    b["left_dv"], b["right_dv"] = b["right_dv"].copy(), b["left_dv"].copy()
    b["left_type"], b["right_type"] = b["right_type"].copy(), b["left_type"].copy()
    # Segment j of the result was segment n-2-j: key j takes key (j+1)'s settings in b.
    for name in ("interpolation", "ease", "back", "amplitude", "period"):
        col = b[name]
        col[:-1] = col[1:].copy()
    ease = b["ease"]
    swapped = ease.copy()
    swapped[ease == 0] = 1
    swapped[ease == 1] = 0
    b["ease"] = swapped
    return curve.replace(b, location=location)


def _dedupe(curve: FCurve, a: dict[str, np.ndarray], location: str) -> FCurve:
    """After snapping: keys landing on one time keep the last (two at most: a jump)."""
    order = np.argsort(a["times"], kind="stable")
    a = {k: v[order] for k, v in a.items()}
    t = a["times"]
    keep = np.ones(t.shape[0], bool)
    run = 1
    for i in range(1, t.shape[0]):
        if t[i] == t[i - 1]:
            run += 1
            if run > 2:
                keep[i - 1] = False
        else:
            run = 1
    a = {k: v[keep] for k, v in a.items()}
    return curve.replace(a, location=location)


def retime(
    curve: FCurve, mapping: Any, velocity: Any = "scale", location: str = "retime"
) -> FCurve:
    """Keys moved through a piecewise-linear time map ``[[old, new], ...]`` (increasing
    in both): ``velocity="scale"`` keeps the shape in the new time (handle extents follow
    the local stretch, so velocities scale), ``"preserve"`` keeps the slopes."""
    if velocity not in ("scale", "preserve"):
        raise _err("INVALID_ARGUMENT", "velocity is 'scale' or 'preserve'", f"{location}.velocity")
    if not isinstance(mapping, (list, tuple)) or len(mapping) < 2:
        raise _err(
            "INVALID_ARGUMENT",
            "a time map is [[old, new], ...] with two points or more",
            f"{location}.map",
        )
    pts = []
    for i, p in enumerate(mapping):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise _err("INVALID_ARGUMENT", "a time map point is [old, new]", f"{location}.map[{i}]")
        pts.append(
            (
                _num(p[0], f"{location}.map[{i}][0]", "old"),
                _num(p[1], f"{location}.map[{i}][1]", "new"),
            )
        )
    old = np.array([p[0] for p in pts])
    new = np.array([p[1] for p in pts])
    if np.any(np.diff(old) <= 0) or np.any(np.diff(new) <= 0):
        raise _err(
            "INVALID_ARGUMENT", "a time map must increase in both columns", f"{location}.map"
        )
    a = curve.arrays()
    t = a["times"]
    slope = np.diff(new) / np.diff(old)
    k = np.clip(np.searchsorted(old, t, side="right") - 1, 0, len(old) - 2)
    local = np.where(t < old[0], slope[0], np.where(t > old[-1], slope[-1], slope[k]))
    a["times"] = np.interp(t, old, new, left=np.nan, right=np.nan)
    below, above = t < old[0], t > old[-1]
    a["times"][below] = new[0] + (t[below] - old[0]) * slope[0]
    a["times"][above] = new[-1] + (t[above] - old[-1]) * slope[-1]
    for s in ("left", "right"):
        a[f"{s}_dt"] = a[f"{s}_dt"] * local
        if velocity == "preserve":
            a[f"{s}_dv"] = a[f"{s}_dv"] * local
    return curve.replace(a, location=location)


# --------------------------------------------------------------------------------------- #
# Timing operations: shape-preserving insertion, eases, anticipation, overshoot, holds
# --------------------------------------------------------------------------------------- #


def _control_points(curve: FCurve, k: int) -> np.ndarray:
    """The four (time, value) control points of segment ``k`` as evaluated (handles
    reaching past the segment scaled back, as the evaluation does)."""
    a = {name: curve.array(name) for name in ("left_dt", "left_dv", "right_dt", "right_dv")}
    t, v = curve.times, curve.values
    span = float(t[k + 1] - t[k])
    p1t, p1v = float(a["right_dt"][k]), float(a["right_dv"][k])
    p2t, p2v = float(a["left_dt"][k + 1]), float(a["left_dv"][k + 1])
    if p1t > span:
        p1t, p1v = span, p1v * span / p1t
    if -p2t > span:
        p2t, p2v = -span, p2v * span / -p2t
    return np.array(
        [
            [t[k], v[k]],
            [t[k] + p1t, v[k] + p1v],
            [t[k + 1] + p2t, v[k + 1] + p2v],
            [t[k + 1], v[k + 1]],
        ],
        dtype=np.float64,
    )


def _freeze_handles(a: dict[str, np.ndarray], keys: Iterable[int]) -> None:
    """Make the handles of ``keys`` free and weighted with their current offsets, so the
    segments on both sides keep their shapes whatever changes around them."""
    for k in keys:
        for side in ("left", "right"):
            if a[f"{side}_type"][k] != ALIGNED:
                a[f"{side}_type"][k] = HANDLE_TYPES.index("free")
        a["weighted"][k] = True


def insert_keeping_shape(curve: FCurve, time: float, location: str = "insert") -> FCurve:
    """``curve`` with a key at ``time`` that leaves its shape unchanged: a Bezier segment
    is split by de Casteljau's construction (the new key gets the two halves' handles,
    aligned; its neighbours' handles become free and weighted, keeping their exact
    offsets), a linear or constant segment gets a key of its kind on it. Eased segments
    cannot be split into two of their kind (``CANNOT_KEEP_SHAPE``)."""
    from .curves import BEZIER, CONSTANT, LINEAR, solve_bezier_time

    t = curve.times
    n = len(curve)
    if not t[0] < time < t[-1] or np.any(t == time):
        value = float(curve.evaluate(time))
        return insert_key(curve, time, value, location)
    k = int(np.searchsorted(t, time, side="right")) - 1
    interp = int(curve.interpolation[k])
    a = curve.arrays()
    if interp in (CONSTANT, LINEAR):
        value = float(a["values"][k]) if interp == CONSTANT else float(curve.evaluate(time))
        return insert_key(curve, time, value, location, interpolation=INTERPOLATIONS[interp])
    if interp != BEZIER:
        raise AnimationError(
            "CANNOT_KEEP_SHAPE",
            f"segment {k} is eased ({INTERPOLATIONS[interp]}): it cannot be split into two "
            "segments of its kind",
            location,
            hint="convert it to bezier first (the interpolation operation), or insert without "
            "keep_shape",
        )
    p = _control_points(curve, k)
    span = float(t[k + 1] - t[k])
    x = p[:, 0] - p[0, 0]
    basis = np.array(
        [
            [
                -x[0] + 3 * x[1] - 3 * x[2] + x[3],
                3 * x[0] - 6 * x[1] + 3 * x[2],
                -3 * x[0] + 3 * x[1],
                x[0],
            ]
        ]
    )
    u = float(solve_bezier_time(basis, np.array([time - t[k]]), np.array([span]))[0])
    p01, p12, p23 = p[0] + u * (p[1] - p[0]), p[1] + u * (p[2] - p[1]), p[2] + u * (p[3] - p[2])
    p012, p123 = p01 + u * (p12 - p01), p12 + u * (p23 - p12)
    mid = p012 + u * (p123 - p012)
    mid[0] = time  # exactly the requested time (u solved it to rounding)
    frozen = [k, k + 1]
    if curve.smoothing == "local":
        # The local rule's clamping factor of a key reads its neighbours' automatic slopes
        # (even of keys with other handles): keep the next neighbours too.
        frozen += [j for j in (k - 1, k + 2) if 0 <= j < len(curve)]
    _freeze_handles(a, frozen)
    a["right_dt"][k], a["right_dv"][k] = p01 - p[0]
    a["left_dt"][k + 1], a["left_dv"][k + 1] = p23 - p[3]
    row = {
        "times": time,
        "values": float(mid[1]),
        "interpolation": BEZIER,
        "ease": int(a["ease"][k]),
        "left_type": ALIGNED,
        "right_type": ALIGNED,
        "left_dt": float(p012[0] - time),
        "left_dv": float(p012[1] - mid[1]),
        "right_dt": float(p123[0] - time),
        "right_dv": float(p123[1] - mid[1]),
        "weighted": True,
        "back": _easing.DEFAULT_BACK,
        "amplitude": _easing.DEFAULT_AMPLITUDE,
        "period": _easing.DEFAULT_PERIOD,
    }
    for name, v in row.items():
        a[name] = np.insert(a[name], k + 1, v)
    del n
    return curve.replace(a, location=location)


def _fraction(args: dict[str, Any], name: str, default: float, location: str, hi: float = 1.0):
    value = _num(args.get(name, default), f"{location}.{name}", name)
    if not 0.0 <= value <= hi:
        raise _err("INVALID_ARGUMENT", f"{name} must be within 0..{hi:g}", f"{location}.{name}")
    return value


def ease_keys(curve: FCurve, sel: np.ndarray, args: dict[str, Any], location: str) -> FCurve:
    """Flat (or ``slope``) handles whose time extents are ``ease_in`` / ``ease_out`` times
    the adjacent interval (weighted keys): longer extents linger longer near the key --
    the "influence" of an ease."""
    a = curve.arrays()
    t = a["times"]
    n = t.shape[0]
    slope = _num(args.get("slope", 0.0), f"{location}.slope", "slope")
    sides = []
    if "ease_in" in args:
        sides.append(("left", _fraction(args, "ease_in", 1 / 3, location)))
    if "ease_out" in args:
        sides.append(("right", _fraction(args, "ease_out", 1 / 3, location)))
    if not sides:
        raise _err("INVALID_ARGUMENT", "give ease_in, ease_out or both (0..1)", location)
    for k in sel.tolist():
        for side, frac in sides:
            if side == "left":
                gap = t[k] - t[k - 1] if k > 0 else (t[k + 1] - t[k] if k + 1 < n else 1.0)
                dt = -frac * gap
            else:
                gap = t[k + 1] - t[k] if k + 1 < n else (t[k] - t[k - 1] if k > 0 else 1.0)
                dt = frac * gap
            a[f"{side}_type"][k] = ALIGNED
            a[f"{side}_dt"][k] = dt
            a[f"{side}_dv"][k] = slope * dt
        if len(sides) == 1:  # the other side keeps its extent and takes the slope
            other = "right" if sides[0][0] == "left" else "left"
            a[f"{other}_type"][k] = ALIGNED
            a[f"{other}_dv"][k] = slope * a[f"{other}_dt"][k]
        a["weighted"][k] = True
    return curve.replace(a, location=location)


def _insert_rows(curve: FCurve, rows: list[tuple[float, float]], location: str) -> FCurve:
    out = curve
    for time, value in rows:
        out = insert_key(out, time, value, location)
    return out


def anticipate(curve: FCurve, sel: np.ndarray, args: dict[str, Any], location: str) -> FCurve:
    """Before the move that starts at each selected key, a counter-move: a key at
    fraction ``at`` (0.3) of the segment, ``amount`` (0.15) of the move's change the other
    way. The move then starts from there, faster."""
    amount = _fraction(args, "amount", 0.15, location, hi=10.0)
    at = _fraction(args, "at", 0.3, location, hi=0.9)
    t, v = curve.times, curve.values
    rows = []
    for k in sel.tolist():
        if k + 1 >= len(curve) or not t[k + 1] > t[k]:
            raise _err("INVALID_SELECTION", f"key {k} starts no move (no later key)", location)
        rows.append(
            (float(t[k] + at * (t[k + 1] - t[k])), float(v[k] - amount * (v[k + 1] - v[k])))
        )
    return _insert_rows(curve, rows, location)


def overshoot(curve: FCurve, sel: np.ndarray, args: dict[str, Any], location: str) -> FCurve:
    """After the move that ends at each selected key, follow-through: the motion passes
    the key, swings ``amount`` (0.1) of the move's change beyond it, and settles back over
    ``settle`` seconds in ``oscillations`` (1) swings shrinking by ``decay`` (0.4)."""
    amount = _fraction(args, "amount", 0.1, location, hi=10.0)
    decay = _fraction(args, "decay", 0.4, location)
    settle = _num(args.get("settle", 0.3), f"{location}.settle", "settle")
    count = args.get("oscillations", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 16:
        raise _err(
            "INVALID_ARGUMENT", "oscillations is an integer 1..16", f"{location}.oscillations"
        )
    if not settle > 0.0:
        raise _err("INVALID_ARGUMENT", "settle must be > 0 seconds", f"{location}.settle")
    t, v = curve.times, curve.values
    rows = []
    for k in sel.tolist():
        if k == 0 or not t[k] > t[k - 1]:
            raise _err("INVALID_SELECTION", f"key {k} ends no move (no earlier key)", location)
        end = t[k] + settle
        if k + 1 < len(curve) and not end < t[k + 1]:
            raise _err(
                "INVALID_ARGUMENT",
                f"the settle after key {k} ({settle:g} s) reaches the next key",
                f"{location}.settle",
            )
        change = float(v[k] - v[k - 1])
        step = settle / (count + 1)
        for i in range(count):
            swing = amount * change * decay**i * (1.0 if i % 2 == 0 else -1.0)
            rows.append((float(t[k] + (i + 1) * step), float(v[k] + swing)))
        rows.append((float(end), float(v[k])))
    return _insert_rows(curve, rows, location)


def hold(curve: FCurve, sel: np.ndarray, args: dict[str, Any], location: str) -> FCurve:
    """A (moving) hold after each selected key: a key ``duration`` later, ``drift`` (0) of
    the way toward the next key; with ``shift`` (default) every later key moves later by
    ``duration`` (time is inserted), else the hold must end before the next key."""
    duration = _num(args.get("duration"), f"{location}.duration", "duration")
    if not duration > 0.0:
        raise _err("INVALID_ARGUMENT", "duration must be > 0 seconds", f"{location}.duration")
    drift = _fraction(args, "drift", 0.0, location)
    shift = args.get("shift", True)
    if not isinstance(shift, bool):
        raise _err("INVALID_ARGUMENT", "shift is true or false", f"{location}.shift")
    out = curve
    for k in sorted(sel.tolist(), reverse=True):  # later keys first: indices stay valid
        t, v = out.times, out.values
        nxt = float(v[k + 1]) if k + 1 < len(out) else float(v[k])
        target = float(v[k] + drift * (nxt - v[k]))
        if shift:
            a = out.arrays()
            a["times"][k + 1 :] = a["times"][k + 1 :] + duration
            out = out.replace(a, location=location)
        elif k + 1 < len(out) and not t[k] + duration < t[k + 1]:
            raise _err("INVALID_ARGUMENT", f"the hold after key {k} reaches the next key", location)
        out = insert_key(out, float(out.times[k] + duration), target, location)
    return out


def smooth_values(curve: FCurve, sel: np.ndarray, args: dict[str, Any], location: str) -> FCurve:
    """The selected keys' values replaced by a Gaussian average (in time, ``sigma``
    seconds) of the keys around them; the curve's first and last keys keep their values
    unless ``keep_ends`` is false. Times and handle types are kept."""
    sigma = _num(args.get("sigma"), f"{location}.sigma", "sigma")
    if not sigma > 0.0:
        raise _err("INVALID_ARGUMENT", "sigma must be > 0 seconds", f"{location}.sigma")
    keep_ends = args.get("keep_ends", True)
    if not isinstance(keep_ends, bool):
        raise _err("INVALID_ARGUMENT", "keep_ends is true or false", f"{location}.keep_ends")
    a = curve.arrays()
    t, v = a["times"], a["values"].copy()
    n = t.shape[0]
    new = v.copy()
    for k in sel.tolist():
        if keep_ends and k in (0, n - 1):
            continue
        lo = int(np.searchsorted(t, t[k] - 3.0 * sigma, side="left"))
        hi = int(np.searchsorted(t, t[k] + 3.0 * sigma, side="right"))
        d = (t[lo:hi] - t[k]) / sigma
        w = np.exp(-0.5 * d * d)
        acc = 0.0
        norm = 0.0
        for wi, vi in zip(w.tolist(), v[lo:hi].tolist(), strict=True):  # a fixed order
            acc += wi * vi
            norm += wi
        new[k] = acc / norm
    a["values"] = new
    return curve.replace(a, location=location)


__all__ = [
    "KEY_OPERATIONS",
    "Clip",
    "Event",
    "edit_curve",
    "insert_key",
    "parse_event",
    "retime",
    "reverse",
    "select",
]
