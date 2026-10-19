"""Non-linear animation: bindings, layers and strips (clip instances).

A **binding** animates one target (an object, or a material) with:

* an ``action`` -- a clip played in scene time as is (``clip time = scene time``): what
  keyframing writes into, at the bottom of the stack;
* ``layers`` above it, bottom to top. A layer is ``replace`` (its value is blended over
  what is below by its weight) or ``additive`` (it adds ``w * (value - reference)``, a
  clip's change relative to its reference pose; rotations compose ``delta^w``), with an
  optional **mask** of property paths or joints (each with a weight; a joint mask entry
  can include its descendants) and a weight that may be animated by an F-curve.

A **strip** places a clip in time: it starts at ``start`` (scene time), plays the clip's
``[clip_start, clip_end]`` (default: the clip's range) ``repeat`` times (fractional
counts end mid-cycle) at ``speed`` (``reverse`` plays it backwards), so it lasts
``(clip_end - clip_start) * repeat / speed`` seconds. ``blend_in``/``blend_out`` ramp its
weight up after its start and down before its end (overlapping strips of a layer then
crossfade), ``before``/``after`` choose what it does outside its time (``nothing`` or
``hold`` its first/last frame), ``accumulate`` adds each repetition's change (root motion
that keeps walking, turns that keep turning), a ``weight`` curve animates its weight, and
a ``time`` curve remaps strip time to clip time explicitly (any retiming).

**Blending.** A replace layer averages its active strips by their weights (``sum w x /
sum w``) and blends that average over the layers below with influence ``min(sum w, 1)``
times its weight and mask: two strips crossfading with weights ``1 - s`` and ``s`` give
``(1 - s) A + s B`` exactly, never the rest pose, while a lone strip fading in blends
from what is below. Rotations average by successive ``slerp`` (``x_i`` weighted ``w_i /
(W + w_i)``). An additive layer adds every strip's ``w (value - reference)`` and composes
rotation deltas ``(reference^-1 x)^w`` in strip order.

Clip time ``c`` of a strip at scene time ``t`` (``local = (t - start) * speed``,
``L = clip_end - clip_start``): cycle ``k = floor(local / L)`` (the last cycle ends at
``clip_end`` exactly), ``c = clip_start + (local - k L)``, mirrored for ``reverse``. All
of this is evaluated directly at ``t``: seeking anywhere costs the same, and nothing
depends on frames evaluated before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import AnimationError
from . import paths as P
from .curves import FCurve

MAX_LAYERS = 64
MAX_STRIPS = 10_000
MAX_MASK = 10_000
MODES = ("replace", "additive")
HOLDS = ("nothing", "hold")


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _num(
    value: Any,
    location: str,
    what: str,
    *,
    lo: float = -math.inf,
    hi: float = math.inf,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err("INVALID_ARGUMENT", f"{what} must be a number, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v) or (positive and v <= 0) or not lo <= v <= hi:
        bound = "> 0" if positive else f"within {lo:g}..{hi:g}"
        raise _err(
            "INVALID_ARGUMENT", f"{what} must be finite and {bound}, got {value!r}", location
        )
    return v


BLENDS = ("linear", "smooth", "smoother")
"""Shapes of a strip's blend ramps: linear (the velocity of a crossfade jumps where a ramp
starts and ends), smooth (3 r^2 - 2 r^3: C1), smoother (6 r^5 - 15 r^4 + 10 r^3: C2)."""


def _shape(r: np.ndarray, shape: Any) -> np.ndarray:
    """A ramp ``r`` in [0, 1] through its blend shape (per element when ``shape`` is an
    array): the same operations whatever the batch, so tables equal single strips."""
    smooth = r * r * (3.0 - 2.0 * r)
    smoother = r * r * r * (r * (r * 6.0 - 15.0) + 10.0)
    return np.where(shape == 2, smoother, np.where(shape == 1, smooth, r))


@dataclass(frozen=True)
class Strip:
    clip: str  # clip item id
    name: str = "strip"
    start: float = 0.0
    clip_start: float | None = None
    clip_end: float | None = None
    speed: float = 1.0
    repeat: float = 1.0
    reverse: bool = False
    blend_in: float = 0.0
    blend_out: float = 0.0
    before: str = "nothing"
    after: str = "nothing"
    weight: float = 1.0
    weight_curve: FCurve | None = None
    time_curve: FCurve | None = None
    accumulate: bool = False
    mute: bool = False
    blend_curve: str = "linear"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"clip": self.clip, "name": self.name, "start": self.start}
        for k in ("clip_start", "clip_end"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        defaults = Strip("")
        for k in (
            "speed",
            "repeat",
            "reverse",
            "blend_in",
            "blend_out",
            "before",
            "after",
            "weight",
            "accumulate",
            "mute",
            "blend_curve",
        ):
            if getattr(self, k) != getattr(defaults, k):
                out[k] = getattr(self, k)
        if self.weight_curve is not None:
            out["weight_curve"] = self.weight_curve.to_json()
        if self.time_curve is not None:
            out["time_curve"] = self.time_curve.to_json()
        return out

    def same(self, other: Strip) -> bool:
        a, b = self.to_json(), other.to_json()
        return a == b


def parse_strip(raw: Any, location: str, clip_ids: Any = None) -> Strip:
    if isinstance(raw, Strip):
        return raw
    if not isinstance(raw, dict):
        raise _err("INVALID_STRIP", "a strip is an object", location)
    allowed = {
        "clip", "name", "start", "clip_start", "clip_end", "speed", "repeat", "reverse", "blend_in",
        "blend_out", "before", "after", "weight", "weight_curve", "time_curve", "accumulate",
        "mute", "blend_curve",
    }  # fmt: skip
    unknown = set(raw) - allowed
    if unknown:
        raise _err("INVALID_STRIP", f"unknown strip fields {sorted(unknown)}", location)
    clip = raw.get("clip")
    if not isinstance(clip, str) or not clip:
        raise _err("INVALID_STRIP", "a strip needs a clip (id or name)", f"{location}.clip")
    if clip_ids is not None:
        clip = clip_ids(clip, f"{location}.clip")
    name = P.check_name(raw.get("name", "strip"), "a strip name", f"{location}.name")
    start = _num(raw.get("start", 0.0), f"{location}.start", "start")
    cs = raw.get("clip_start")
    ce = raw.get("clip_end")
    cs = None if cs is None else _num(cs, f"{location}.clip_start", "clip_start")
    ce = None if ce is None else _num(ce, f"{location}.clip_end", "clip_end")
    if cs is not None and ce is not None and not cs < ce:
        raise _err("INVALID_STRIP", "clip_start must be before clip_end", location)
    speed = _num(raw.get("speed", 1.0), f"{location}.speed", "speed", positive=True, hi=1e6)
    repeat = _num(raw.get("repeat", 1.0), f"{location}.repeat", "repeat", positive=True, hi=1e9)
    blend_in = _num(raw.get("blend_in", 0.0), f"{location}.blend_in", "blend_in", lo=0.0)
    blend_out = _num(raw.get("blend_out", 0.0), f"{location}.blend_out", "blend_out", lo=0.0)
    weight = _num(raw.get("weight", 1.0), f"{location}.weight", "weight", lo=0.0, hi=1.0)
    blend = raw.get("blend_curve", "linear")
    if blend not in BLENDS:
        raise _err(
            "INVALID_STRIP",
            f"blend_curve is one of {', '.join(BLENDS)}",
            f"{location}.blend_curve",
        )
    out: dict[str, Any] = {}
    for key in ("reverse", "accumulate", "mute"):
        v = raw.get(key, False)
        if not isinstance(v, bool):
            raise _err("INVALID_STRIP", f"{key} is true or false", f"{location}.{key}")
        out[key] = v
    for key in ("before", "after"):
        v = raw.get(key, "nothing")
        if v not in HOLDS:
            raise _err("INVALID_STRIP", f"{key} is 'nothing' or 'hold'", f"{location}.{key}")
        out[key] = v
    curves = {}
    for key in ("weight_curve", "time_curve"):
        v = raw.get(key)
        if v is None:
            curves[key] = None
        elif isinstance(v, FCurve):
            curves[key] = v
        else:
            curves[key] = (
                FCurve.from_json(v, f"{location}.{key}")
                if isinstance(v, dict)
                else FCurve.from_keys(v, location=f"{location}.{key}")
            )
    return Strip(
        clip, name, start, cs, ce, speed, repeat, out["reverse"], blend_in, blend_out,
        out["before"], out["after"], weight, curves["weight_curve"], curves["time_curve"],
        out["accumulate"], out["mute"], blend,
    )  # fmt: skip


@dataclass(frozen=True)
class Layer:
    name: str
    mode: str = "replace"
    weight: float = 1.0
    weight_curve: FCurve | None = None
    mask: tuple[
        tuple[str, float, bool], ...
    ] = ()  # (path prefix or joint name, weight, descendants)
    strips: tuple[Strip, ...] = ()
    mute: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "mode": self.mode}
        if self.weight != 1.0:
            out["weight"] = self.weight
        if self.weight_curve is not None:
            out["weight_curve"] = self.weight_curve.to_json()
        if self.mask:
            out["mask"] = [
                {
                    "path" if not m[0].startswith("joint:") else "joint": m[0].removeprefix(
                        "joint:"
                    ),
                    "weight": m[1],
                    **({"descendants": True} if m[2] else {}),
                }
                for m in self.mask
            ]
        out["strips"] = [s.to_json() for s in self.strips]
        if self.mute:
            out["mute"] = True
        return out


def parse_layer(raw: Any, location: str, clip_ids: Any = None) -> Layer:
    if isinstance(raw, Layer):
        return raw
    if not isinstance(raw, dict):
        raise _err("INVALID_LAYER", "a layer is an object", location)
    unknown = set(raw) - {"name", "mode", "weight", "weight_curve", "mask", "strips", "mute"}
    if unknown:
        raise _err("INVALID_LAYER", f"unknown layer fields {sorted(unknown)}", location)
    name = P.check_name(raw.get("name", "Layer"), "a layer name", f"{location}.name")
    mode = raw.get("mode", "replace")
    if mode not in MODES:
        raise _err("INVALID_LAYER", "mode is 'replace' or 'additive'", f"{location}.mode")
    weight = _num(raw.get("weight", 1.0), f"{location}.weight", "weight", lo=0.0, hi=1.0)
    wc = raw.get("weight_curve")
    if wc is not None and not isinstance(wc, FCurve):
        wc = (
            FCurve.from_json(wc, f"{location}.weight_curve")
            if isinstance(wc, dict)
            else FCurve.from_keys(wc, location=f"{location}.weight_curve")
        )
    mask_raw = raw.get("mask", [])
    if not isinstance(mask_raw, list) or len(mask_raw) > MAX_MASK:
        raise _err(
            "INVALID_LAYER", f"mask is a list of at most {MAX_MASK} entries", f"{location}.mask"
        )
    mask = []
    for i, m in enumerate(mask_raw):
        loc = f"{location}.mask[{i}]"
        if isinstance(m, str):
            m = {"path": m}
        if (
            not isinstance(m, dict)
            or set(m) - {"path", "joint", "weight", "descendants"}
            or (("path" in m) == ("joint" in m))
        ):
            raise _err("INVALID_LAYER", "a mask entry is {path | joint, weight, descendants}", loc)
        w = _num(m.get("weight", 1.0), f"{loc}.weight", "weight", lo=0.0, hi=1.0)
        desc = m.get("descendants", False)
        if not isinstance(desc, bool):
            raise _err("INVALID_LAYER", "descendants is true or false", f"{loc}.descendants")
        if "joint" in m:
            key = "joint:" + P.check_name(m["joint"], "a joint name", f"{loc}.joint")
        else:
            path = m["path"]
            if not isinstance(path, str) or not path:
                raise _err(
                    "INVALID_LAYER",
                    "a mask path is a property path or a prefix of one",
                    f"{loc}.path",
                )
            if desc:
                raise _err("INVALID_LAYER", "descendants applies to joints", loc)
            key = path
        mask.append((key, w, desc))
    strips_raw = raw.get("strips", [])
    if not isinstance(strips_raw, list) or len(strips_raw) > MAX_STRIPS:
        raise _err(
            "INVALID_LAYER", f"strips is a list of at most {MAX_STRIPS}", f"{location}.strips"
        )
    strips = tuple(
        parse_strip(s, f"{location}.strips[{i}]", clip_ids) for i, s in enumerate(strips_raw)
    )
    mute = raw.get("mute", False)
    if not isinstance(mute, bool):
        raise _err("INVALID_LAYER", "mute is true or false", f"{location}.mute")
    return Layer(name, mode, weight, wc, tuple(mask), strips, mute)


@dataclass(frozen=True)
class Binding:
    """What animates one target: ``target`` is ``("object" | "material", id)``, or
    ``("world", "world")`` for the scene's world (0.20.0)."""

    target: tuple[str, str]
    action: str | None = None  # clip item id
    layers: tuple[Layer, ...] = ()
    mute: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {self.target[0]: self.target[1]}
        if self.action is not None:
            out["action"] = self.action
        if self.layers:
            out["layers"] = [layer.to_json() for layer in self.layers]
        if self.mute:
            out["mute"] = True
        return out

    def clips(self) -> set[str]:
        out = set() if self.action is None else {self.action}
        for layer in self.layers:
            out.update(s.clip for s in layer.strips)
        return out


def parse_binding(raw: Any, location: str, clip_ids: Any = None) -> Binding:
    if isinstance(raw, Binding):
        return raw
    if not isinstance(raw, dict):
        raise _err("INVALID_BINDING", "a binding is an object", location)
    unknown = set(raw) - {"object", "material", "world", "action", "layers", "mute"}
    if unknown:
        raise _err("INVALID_BINDING", f"unknown binding fields {sorted(unknown)}", location)
    kinds = [k for k in ("object", "material", "world") if k in raw]
    if len(kinds) != 1:
        raise _err("INVALID_BINDING", "a binding has an object, a material or the world", location)
    kind = kinds[0]
    tid = raw[kind]
    if kind == "world":
        if tid != "world":
            raise _err("INVALID_BINDING", 'the world\'s binding is {"world": "world"}',
                       f"{location}.world")  # fmt: skip
    elif not isinstance(tid, str) or not tid:
        raise _err("INVALID_BINDING", f"{kind} is an id", f"{location}.{kind}")
    action = raw.get("action")
    if action is not None:
        if not isinstance(action, str):
            raise _err("INVALID_BINDING", "action is a clip id or name", f"{location}.action")
        if clip_ids is not None:
            action = clip_ids(action, f"{location}.action")
    layers_raw = raw.get("layers", [])
    if not isinstance(layers_raw, list) or len(layers_raw) > MAX_LAYERS:
        raise _err(
            "INVALID_BINDING", f"layers is a list of at most {MAX_LAYERS}", f"{location}.layers"
        )
    layers = tuple(
        parse_layer(layer, f"{location}.layers[{i}]", clip_ids)
        for i, layer in enumerate(layers_raw)
    )
    names = [layer.name for layer in layers]
    if len(set(names)) != len(names):
        raise _err(
            "INVALID_BINDING", "layer names must be unique within a binding", f"{location}.layers"
        )
    mute = raw.get("mute", False)
    if not isinstance(mute, bool):
        raise _err("INVALID_BINDING", "mute is true or false", f"{location}.mute")
    return Binding((kind, tid), action, layers, mute)


# --------------------------------------------------------------------------------------- #
# Time mapping
# --------------------------------------------------------------------------------------- #


def strip_length(strip: Strip, clip_range: tuple[float, float]) -> float:
    """Scene seconds the strip plays (its time curve's range when it has one)."""
    if strip.time_curve is not None:
        a, b = strip.time_curve.range
        return b - a
    cs, ce = _trim(strip, clip_range)
    return (ce - cs) * strip.repeat / strip.speed


def _trim(strip: Strip, clip_range: tuple[float, float]) -> tuple[float, float]:
    cs = clip_range[0] if strip.clip_start is None else strip.clip_start
    ce = clip_range[1] if strip.clip_end is None else strip.clip_end
    return cs, ce


def strip_times(
    strip: Strip, clip_range: tuple[float, float], t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For scene times ``t``: ``(clip time, weight, cycle, d clip time / d t)``. A weight
    of 0 means the strip does not contribute at that time."""
    t = np.asarray(t, np.float64)
    if strip.mute:
        z = np.zeros_like(t)
        return z, z.copy(), z.copy(), z.copy()
    cs, ce = _trim(strip, clip_range)
    span = ce - cs
    if strip.time_curve is not None:
        a, b = strip.time_curve.range
        length = b - a
        local = t - strip.start
        c, dc = strip.time_curve.evaluate(np.clip(local + a, a, b), order=1)
        cycle = np.zeros_like(t)
    else:
        length = span * strip.repeat / strip.speed if span > 0 else 0.0
        local = (t - strip.start) * strip.speed
        if span > 0:
            cycle = np.floor(local / span)
            total = span * strip.repeat
            within = local - cycle * span
            # The end of the last cycle is the end of the clip, not the start of a cycle.
            last = np.minimum(np.floor(total / span - 1e-12), np.ceil(strip.repeat) - 1)
            at_end = local >= total
            cycle = np.where(at_end, last, np.clip(cycle, 0, last))
            within = np.where(at_end, total - last * span, np.clip(local - cycle * span, 0.0, span))
            within = np.where(local < 0, 0.0, within)
        else:
            cycle = np.zeros_like(t)
            within = np.zeros_like(t)
        c = ce - within if strip.reverse else cs + within
        dc = np.full_like(t, -strip.speed if strip.reverse else strip.speed)
    inside = (t >= strip.start) & (t <= strip.start + length)
    before = t < strip.start
    after = t > strip.start + length
    w = np.where(inside, 1.0, 0.0)
    if strip.before == "hold":
        w = np.where(before, 1.0, w)
    if strip.after == "hold":
        w = np.where(after, 1.0, w)
    dc = np.where(inside, dc, 0.0)
    # Ramps (inside the strip's time), shaped by the strip's blend.
    shape = BLENDS.index(strip.blend_curve)
    if strip.blend_in > 0:
        ramp = (t - strip.start) / strip.blend_in
        w = np.where(inside & (ramp < 1.0), _shape(np.clip(ramp, 0.0, 1.0), shape), w)
    if strip.blend_out > 0:
        ramp = (strip.start + length - t) / strip.blend_out
        w = np.where(
            inside & (ramp < 1.0), np.minimum(w, _shape(np.clip(ramp, 0.0, 1.0), shape)), w
        )
    w = w * strip.weight
    if strip.weight_curve is not None:
        w = w * np.clip(strip.weight_curve.evaluate(t - strip.start), 0.0, 1.0)
    return c, w, cycle, dc


class StripTable:
    """Plain strips (no time or weight curve) as arrays, timed in one pass: the clip
    time, weight and cycle of every strip at a scene time. The same operations as
    :func:`strip_times`, element by element, so the results are bit for bit its own."""

    def __init__(self, strips: list[tuple[Strip, tuple[float, float]]]) -> None:
        n = len(strips)
        f = {k: np.zeros(n) for k in ("start", "speed", "cs", "ce", "span", "length", "total",
                                      "last", "blend_in", "blend_out", "weight")}  # fmt: skip
        b = {k: np.zeros(n, bool) for k in ("reverse", "hold_before", "hold_after", "mute")}
        shapes = np.zeros(n, np.int64)
        for i, (strip, clip_range) in enumerate(strips):
            assert strip.time_curve is None and strip.weight_curve is None
            cs, ce = _trim(strip, clip_range)
            span = ce - cs
            f["start"][i], f["speed"][i] = strip.start, strip.speed
            f["cs"][i], f["ce"][i], f["span"][i] = cs, ce, span
            if span > 0:
                total = span * strip.repeat
                f["length"][i] = span * strip.repeat / strip.speed
                f["total"][i] = total
                f["last"][i] = np.minimum(np.floor(total / span - 1e-12), np.ceil(strip.repeat) - 1)
            f["blend_in"][i], f["blend_out"][i] = strip.blend_in, strip.blend_out
            f["weight"][i] = strip.weight
            b["reverse"][i], b["mute"][i] = strip.reverse, strip.mute
            b["hold_before"][i] = strip.before == "hold"
            b["hold_after"][i] = strip.after == "hold"
            shapes[i] = BLENDS.index(strip.blend_curve)
        self.shape = shapes
        self.__dict__.update(f)
        self.__dict__.update(b)
        self.cycling = self.span > 0
        self.safe_span = np.where(self.cycling, self.span, 1.0)
        self.ramp_in = self.blend_in > 0
        self.ramp_out = self.blend_out > 0
        self.safe_in = np.where(self.ramp_in, self.blend_in, 1.0)
        self.safe_out = np.where(self.ramp_out, self.blend_out, 1.0)
        self.end = self.start + self.length

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(clip time, weight, cycle)`` of every strip at scene time ``t``."""
        start, span, last, total = self.start, self.span, self.last, self.total
        local = (t - start) * self.speed
        cycle = np.floor(local / self.safe_span)
        at_end = local >= total
        cycle = np.where(at_end, last, np.clip(cycle, 0, last))
        within = np.where(at_end, total - last * span, np.clip(local - cycle * span, 0.0, span))
        within = np.where(local < 0, 0.0, within)
        cycle = np.where(self.cycling, cycle, 0.0)
        within = np.where(self.cycling, within, 0.0)
        c = np.where(self.reverse, self.ce - within, self.cs + within)
        inside = (t >= start) & (t <= self.end)
        w = np.where(inside, 1.0, 0.0)
        w = np.where(self.hold_before & (t < start), 1.0, w)
        w = np.where(self.hold_after & (t > self.end), 1.0, w)
        ramp = (t - start) / self.safe_in
        w = np.where(
            self.ramp_in & inside & (ramp < 1.0), _shape(np.clip(ramp, 0.0, 1.0), self.shape), w
        )
        ramp = (self.end - t) / self.safe_out
        w = np.where(
            self.ramp_out & inside & (ramp < 1.0),
            np.minimum(w, _shape(np.clip(ramp, 0.0, 1.0), self.shape)),
            w,
        )
        w = w * self.weight
        mute = self.mute
        return np.where(mute, 0.0, c), np.where(mute, 0.0, w), np.where(mute, 0.0, cycle)


def layer_weight(layer: Layer, t: float) -> float:
    if layer.mute:
        return 0.0
    w = layer.weight
    if layer.weight_curve is not None:
        w *= float(np.clip(layer.weight_curve.evaluate(t), 0.0, 1.0))
    return w
