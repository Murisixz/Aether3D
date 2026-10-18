"""Time in Aether3D animation: seconds, exact frame rates, frames and timecode.

**Seconds are canonical.** Keyframes, strips, markers and events store times in seconds
(float64). Frames are a view of time through a frame rate, so changing the timeline's
frame rate never changes a motion, and sub-frame and negative times are ordinary values.

**Frame rates are exact rationals** (:class:`FrameRate`): ``24``, ``25``, ``30``,
``"30000/1001"`` (29.97, NTSC), ``"24000/1001"`` (23.976). The time of frame ``n`` is the
rational ``n * den / num`` rounded once to float64 (``frame_times`` multiplies exactly,
then divides once: the correctly rounded value for every integer frame below 2**53), and
the frame at a time is decided on the exact rational value of the float -- so
``frame_at(frame_time(n)) == n`` for every frame, at every rate, with no drift over hours
of timeline.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np

from ..errors import AnimationError

MAX_NUMERATOR = 1_000_000
MAX_DENOMINATOR = 100_000
MIN_FPS = Fraction(1, 1000)
MAX_FPS = Fraction(10_000)

# Decimal spellings of the NTSC-family rates, which are really n * 1000/1001.
_NTSC = {
    "23.976": (24000, 1001),
    "23.98": (24000, 1001),
    "29.97": (30000, 1001),
    "47.952": (48000, 1001),
    "59.94": (60000, 1001),
    "119.88": (120000, 1001),
}


def _bad(message: str, location: str, code: str = "INVALID_FRAME_RATE") -> AnimationError:
    return AnimationError(code, message, location)


class FrameRate:
    """An exact frame rate ``numerator / denominator`` frames per second.

    Accepts an integer (``24``), a ``"num/den"`` string (``"30000/1001"``), a
    ``[num, den]`` pair, a :class:`~fractions.Fraction`, another ``FrameRate``, or a float
    that is an integer (``25.0``) or an NTSC decimal (``29.97`` -> ``30000/1001``).
    """

    __slots__ = ("_num", "_den")

    def __init__(self, rate: Any = 24, location: str = "fps") -> None:
        num, den = _parse_rate(rate, location)
        self._num = num
        self._den = den

    @property
    def numerator(self) -> int:
        return self._num

    @property
    def denominator(self) -> int:
        return self._den

    @property
    def fraction(self) -> Fraction:
        return Fraction(self._num, self._den)

    @property
    def fps(self) -> float:
        """Frames per second as a float (``29.97002997...`` for ``30000/1001``)."""
        return self._num / self._den

    @property
    def frame_duration(self) -> float:
        """Seconds per frame (``den / num``, rounded once)."""
        return self._den / self._num

    def frame_time(self, frame: float) -> float:
        """The time in seconds of ``frame`` (fractional frames give sub-frame times)."""
        if isinstance(frame, bool) or not isinstance(frame, (int, float, np.integer, np.floating)):
            raise _bad(f"frame must be a number, got {frame!r}", "frame", "INVALID_TIME")
        if isinstance(frame, (int, np.integer)):
            return float(Fraction(int(frame) * self._den, self._num))
        f = float(frame)
        if not math.isfinite(f):
            raise _bad(f"frame must be finite, got {frame!r}", "frame", "INVALID_TIME")
        return float(Fraction(f) * self._den / self._num)

    def frame_times(self, frames: Any) -> np.ndarray:
        """Times of integer ``frames`` (array), each the exact rational rounded once."""
        f = np.asarray(frames)
        if f.dtype.kind not in "iu":
            raise _bad("frames must be integers", "frames", "INVALID_TIME")
        if f.size and int(np.max(np.abs(f))) * self._den >= 2**53:
            raise _bad("frame numbers are too large for exact times", "frames", "INVALID_TIME")
        return (f.astype(np.float64) * float(self._den)) / float(self._num)

    def frame_at(self, time: float, rounding: str = "nearest") -> int:
        """The frame at ``time`` seconds: ``"nearest"`` (ties to the later frame),
        ``"floor"`` (the frame showing at that time) or ``"ceil"``."""
        t = _finite_time(time, "time")
        if rounding not in ("nearest", "floor", "ceil"):
            raise _bad(f"rounding must be nearest, floor or ceil, got {rounding!r}", "rounding")
        exact = Fraction(t) * self._num / self._den
        nearest = math.floor(exact + Fraction(1, 2))
        # A frame's time is the double nearest to its exact rational time, which may lie
        # just below or above it: that double is the frame, whatever the rounding.
        if rounding == "nearest" or float(Fraction(nearest * self._den, self._num)) == t:
            return nearest
        return math.floor(exact) if rounding == "floor" else math.ceil(exact)

    def snap(self, time: float) -> float:
        """``time`` moved to the nearest frame time."""
        return self.frame_time(self.frame_at(time))

    def is_on_frame(self, time: float) -> bool:
        """Whether ``time`` is exactly (as a float) the time of a frame."""
        return self.frame_time(self.frame_at(time)) == float(time)

    def timecode(self, time: float) -> str:
        """``HH:MM:SS:FF`` non-drop-frame timecode of ``time`` (frames counted at the
        nominal integer rate, ``ceil(fps)``; negative times get a leading ``-``)."""
        frame = self.frame_at(time, "floor")
        sign = "-" if frame < 0 else ""
        frame = abs(frame)
        nominal = math.ceil(self.fraction)
        ff = frame % nominal
        seconds = frame // nominal
        return f"{sign}{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}:{ff:02d}"

    def to_json(self) -> Any:
        return self._num if self._den == 1 else f"{self._num}/{self._den}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameRate):
            return NotImplemented
        return self._num == other._num and self._den == other._den

    def __hash__(self) -> int:
        return hash((self._num, self._den))

    def __repr__(self) -> str:
        return f"FrameRate({self.to_json()!r})"


def _parse_rate(rate: Any, location: str) -> tuple[int, int]:
    if isinstance(rate, FrameRate):
        return rate._num, rate._den
    value: Fraction | None = None
    if isinstance(rate, bool):
        value = None
    elif isinstance(rate, (int, np.integer)):
        value = Fraction(int(rate))
    elif isinstance(rate, Fraction):
        value = rate
    elif isinstance(rate, (float, np.floating)):
        f = float(rate)
        if math.isfinite(f):
            text = f"{f:.3f}".rstrip("0").rstrip(".")
            if f == int(f):
                value = Fraction(int(f))
            elif text in _NTSC:
                value = Fraction(*_NTSC[text])
            else:
                raise _bad(
                    f"frame rate {f!r} is neither an integer nor an NTSC rate",
                    location,
                )
    elif isinstance(rate, str):
        text = rate.strip()
        if text in _NTSC:
            value = Fraction(*_NTSC[text])
        else:
            parts = text.split("/")
            try:
                if len(parts) == 2:
                    value = Fraction(int(parts[0]), int(parts[1]))
                elif len(parts) == 1:
                    value = Fraction(int(parts[0]))
            except (ValueError, ZeroDivisionError):
                value = None
    elif isinstance(rate, (list, tuple)) and len(rate) == 2:
        n, d = rate
        integers = all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in (n, d))
        if integers and d != 0:
            value = Fraction(int(n), int(d))
    if value is None:
        raise _bad(
            f"frame rate must be an integer, 'num/den', [num, den] or an NTSC decimal, "
            f"got {rate!r}",
            location,
        )
    if not (MIN_FPS <= value <= MAX_FPS):
        raise _bad(f"frame rate {value} is outside {MIN_FPS}..{MAX_FPS}", location)
    if value.numerator > MAX_NUMERATOR or value.denominator > MAX_DENOMINATOR:
        raise _bad(f"frame rate {value} has too large a numerator or denominator", location)
    return value.numerator, value.denominator


def _finite_time(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise AnimationError("INVALID_TIME", f"time must be a number, got {value!r}", location)
    t = float(value)
    if not math.isfinite(t):
        raise AnimationError("INVALID_TIME", f"time must be finite, got {value!r}", location)
    return t


def check_time(value: Any, location: str = "time") -> float:
    """``value`` as a finite float time in seconds, or :class:`AnimationError`."""
    return _finite_time(value, location)


def frame_range(rate: FrameRate, start: float, end: float, step: int = 1) -> np.ndarray:
    """The frames whose times lie in ``[start, end]`` (both ends included), every
    ``step``-th from the first, as an int64 array."""
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise AnimationError("INVALID_TIME", f"step must be an integer >= 1, got {step!r}", "step")
    a = rate.frame_at(check_time(start, "start"), "ceil")
    b = rate.frame_at(check_time(end, "end"), "floor")
    if b < a:
        return np.zeros(0, dtype=np.int64)
    return np.arange(a, b + 1, int(step), dtype=np.int64)
