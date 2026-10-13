"""Lights: the light component of a scene object.

This is a deliberately simple, **non-physically-based** lighting model (see
``docs/RENDERING.md`` for the exact equations):

* ``"directional"`` -- parallel light travelling along the light object's
  forward axis (local -Y, like cameras). ``intensity`` is a unitless
  multiplier: a white light of intensity 1 shining straight onto a surface
  makes it appear exactly its base color.
* ``"point"`` -- light emitted from the object's world position. ``intensity``
  is the same multiplier measured at a distance of 1 unit; with
  ``falloff="inverse_square"`` (default) it falls off as ``1 / d^2``, with
  ``"none"`` it stays constant. An optional ``range`` fades it smoothly to zero
  at that distance with the window ``clamp(1 - (d / range)^4, 0, 1)^2``.

Ambient light is a scene-wide setting (``scene.world``), not a light object.
Lights do not cast shadows in this renderer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import RenderError
from .color import Color, parse_color

LIGHT_TYPES = ("directional", "point")
FALLOFFS = ("inverse_square", "none")
MAX_INTENSITY = 1e9

_FIELDS = ("type", "color", "intensity", "range", "falloff")


class Light:
    """The light component of a scene object (``obj.light``).

    * ``type`` -- ``"directional"`` or ``"point"``.
    * ``color`` -- display (sRGB) color, e.g. ``"#fff2d6"`` or ``(1, 0.95, 0.85)``.
    * ``intensity`` -- unitless multiplier, ``>= 0``.
    * ``range`` -- point lights: distance at which the light fades out, or ``None``.
    * ``falloff`` -- point lights: ``"inverse_square"`` (default) or ``"none"``.
    """

    __slots__ = tuple(f"_{name}" for name in _FIELDS)

    def __init__(
        self,
        type: str = "directional",
        color: Any = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        range: float | None = None,
        falloff: str = "inverse_square",
    ) -> None:
        values = _validate(
            {
                "type": type,
                "color": color,
                "intensity": intensity,
                "range": range,
                "falloff": falloff,
            },
            "light",
        )
        for name in _FIELDS:
            setattr(self, f"_{name}", values[name])

    def configure(self, **changes: Any) -> Light:
        """Change several parameters at once; validated together (all or nothing)."""
        unknown = sorted(set(changes) - set(_FIELDS))
        if unknown:
            raise RenderError(
                "INVALID_LIGHT",
                f"unknown light parameter(s): {', '.join(unknown)}",
                "light",
                hint=f"parameters: {', '.join(_FIELDS)}",
            )
        values = _validate({**self.to_dict(), **changes}, "light")
        for name in _FIELDS:
            setattr(self, f"_{name}", values[name])
        return self

    type = property(lambda self: self._type, lambda self, v: self.configure(type=v))
    color = property(lambda self: self._color, lambda self, v: self.configure(color=v))
    intensity = property(lambda self: self._intensity, lambda self, v: self.configure(intensity=v))
    range = property(lambda self: self._range, lambda self, v: self.configure(range=v))
    falloff = property(lambda self: self._falloff, lambda self, v: self.configure(falloff=v))

    def copy(self) -> Light:
        return Light(**self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, f"_{name}") for name in _FIELDS}
        data["color"] = list(self._color)
        return data

    @classmethod
    def from_dict(cls, data: Any, location: str = "light") -> Light:
        if not isinstance(data, dict):
            raise RenderError("INVALID_LIGHT", "expected an object", location)
        missing = [f for f in _FIELDS if f not in data]
        unknown = sorted(set(data) - set(_FIELDS))
        if missing or unknown:
            what = f"missing {', '.join(missing)}" if missing else f"unknown {', '.join(unknown)}"
            raise RenderError("INVALID_LIGHT", f"light keys: {what}", location)
        light = cls.__new__(cls)
        values = _validate(dict(data), location)
        for name in _FIELDS:
            setattr(light, f"_{name}", values[name])
        return light

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Light):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        extra = ""
        if self._type == "point":
            extra = f", falloff={self._falloff}" + (
                f", range={self._range:g}" if self._range is not None else ""
            )
        return f"Light({self._type}, color={self._color}, intensity={self._intensity:g}{extra})"


def _validate(values: dict[str, Any], location: str) -> dict[str, Any]:
    if values["type"] not in LIGHT_TYPES:
        raise RenderError(
            "INVALID_LIGHT",
            f"type must be one of {', '.join(LIGHT_TYPES)}, got {values['type']!r}",
            f"{location}.type",
        )
    try:
        color: Color = parse_color(values["color"], f"{location}.color")
    except RenderError as exc:
        raise RenderError("INVALID_LIGHT", exc.message, exc.location) from None
    intensity = values["intensity"]
    if (
        isinstance(intensity, bool)
        or not isinstance(intensity, (int, float, np.integer, np.floating))
        or not math.isfinite(float(intensity))
        or not 0.0 <= float(intensity) <= MAX_INTENSITY
    ):
        raise RenderError(
            "INVALID_LIGHT",
            f"intensity must be a finite number in [0, {MAX_INTENSITY:.0e}], got {intensity!r}",
            f"{location}.intensity",
        )
    rng = values["range"]
    if rng is not None:
        if (
            isinstance(rng, bool)
            or not isinstance(rng, (int, float, np.integer, np.floating))
            or not math.isfinite(float(rng))
            or float(rng) <= 0.0
        ):
            raise RenderError(
                "INVALID_LIGHT",
                f"range must be a number > 0 or null, got {rng!r}",
                f"{location}.range",
            )
        rng = float(rng)
    if values["falloff"] not in FALLOFFS:
        raise RenderError(
            "INVALID_LIGHT",
            f"falloff must be one of {', '.join(FALLOFFS)}, got {values['falloff']!r}",
            f"{location}.falloff",
        )
    return {
        "type": values["type"],
        "color": color,
        "intensity": float(intensity),
        "range": rng,
        "falloff": values["falloff"],
    }
