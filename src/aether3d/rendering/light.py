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
The CPU renderer draws no shadows; the GPU backend renders shadow maps for
directional lights when asked (``render(backend="gpu", shadows=True)``).

Light sizes are used by the path tracer only (``render(backend="path")``),
which traces shadow rays toward points on the light: ``radius`` makes a point
light a glowing sphere and ``angle`` gives a directional light the angular
diameter of a sun disk, so shadows get soft edges (penumbrae). Both default
to 0 (a point; parallel rays), and the raster backends ignore them.

0.20.0 adds (measured against Blender 5.2's Cycles: research/2026-10-stage19/blender/
e3_lights):

* ``"spot"`` -- a point light masked to a cone along the light's front (local -Y):
  ``spot_size`` is the cone's full angle in degrees, ``spot_blend`` (0..1) softens its edge
  with ``smoothstep((cos t - cos h) / ((1 - cos h) blend))`` (``h`` the half angle), as
  Blender; inside the cone it shines as the point light of the same intensity does.
* ``"area"`` -- a one-sided Lambertian emitter facing the light's front: ``shape``
  (``square``, ``rectangle``, ``disk``, ``ellipse``), ``size`` along local X and
  ``size_y`` along local Z (the second size of rectangles and ellipses). Its radiance is
  ``pi I / A`` (``A`` its area): far away it lights a surface facing it as the point light
  of the same intensity would. ``spread`` (degrees, default 180) narrows its emission as a
  grid of slats in front of it would (a honeycomb): toward an angle ``a`` from its normal
  the radiance is ``L max(1 - tan a / tan(spread / 2), 0) / (1 - h / tan h)`` (``h =
  spread / 2``), which keeps its power; this is what Blender's ``spread`` measures as.
* ``units`` -- ``"aether"`` (``intensity`` as above) or ``"watts"``: Blender's light
  units -- a sun's irradiance in W/m^2 (intensity ``S / pi``), the power of point and
  spot lights in W (``P / (4 pi^2)``: radiant intensity ``P / (4 pi)``), and of area
  lights (``P / pi^2``: radiance ``P / (pi A)``).
* ``temperature`` (kelvins) with ``use_temperature`` -- the colour is multiplied by the
  black body's (:mod:`.blackbody`: luminance 1, so only the hue changes); ``exposure``
  (stops) multiplies the intensity by ``2 ** exposure``; ``shadow`` -- whether the light is
  occluded (``False``: it shines through geometry, an artistic light).
* Light linking: ``include`` -- object ids; when not empty, the light lights only those
  objects and their descendants; ``exclude`` -- object ids it does not light (with their
  descendants). Every backend honours it, at every path vertex (a linked light's light
  reflected off its receivers still reaches other objects in the path tracers).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import RenderError
from ..history import recording
from .color import Color, parse_color

LIGHT_TYPES = ("directional", "point", "spot", "area")
FALLOFFS = ("inverse_square", "none")
SHAPES = ("square", "rectangle", "disk", "ellipse")
UNITS = ("aether", "watts")
MAX_INTENSITY = 1e9
MAX_ANGLE = 180.0
"""Largest angular diameter of a directional light, in degrees."""
MAX_SIZE = 1e6
MAX_LINKS = 100_000
"""Most object ids a light's ``include`` or ``exclude`` lists."""

_FIELDS = (
    "type",
    "color",
    "intensity",
    "range",
    "falloff",
    "radius",
    "angle",
    "spot_size",
    "spot_blend",
    "shape",
    "size",
    "size_y",
    "spread",
    "units",
    "temperature",
    "use_temperature",
    "exposure",
    "shadow",
    "include",
    "exclude",
)
DEFAULTS_0_20 = {
    "spot_size": 45.0,
    "spot_blend": 0.15,
    "shape": "square",
    "size": 1.0,
    "size_y": 1.0,
    "spread": 180.0,
    "units": "aether",
    "temperature": 6500.0,
    "use_temperature": False,
    "exposure": 0.0,
    "shadow": True,
    "include": (),
    "exclude": (),
}
"""The fields 0.20.0 adds and their defaults (lights of older documents keep their look)."""


class Light:
    """The light component of a scene object (``obj.light``).

    * ``type`` -- ``"directional"``, ``"point"``, ``"spot"`` or ``"area"``. Directional,
      spot and area lights shine along the object's front (local -Y).
    * ``color`` -- display (sRGB) color, e.g. ``"#fff2d6"`` or ``(1, 0.95, 0.85)``.
    * ``intensity`` -- ``>= 0``, in ``units``.
    * ``range`` -- point and spot lights: distance at which the light fades out, or ``None``.
    * ``falloff`` -- point and spot lights: ``"inverse_square"`` (default) or ``"none"``.
    * ``radius`` -- point and spot lights: radius of the emitting sphere (``>= 0``,
      default 0); soft shadows in the path tracers.
    * ``angle`` -- directional lights: angular diameter in degrees
      (``0..180``, default 0; the sun is about 0.53); soft shadows in the
      path tracers.
    * ``spot_size`` -- spot lights: the cone's full angle in degrees (``1..180``,
      default 45); ``spot_blend`` -- its soft edge (``0..1``, default 0.15): the
      cone fades with a smoothstep over that share of its cosine range, as in Blender.
    * ``shape`` -- area lights: ``"square"``, ``"rectangle"``, ``"disk"`` or
      ``"ellipse"``; ``size`` (width or diameter, along local X) and ``size_y``
      (rectangles and ellipses, along local Z), in metres. Area lights are one-sided
      Lambertian emitters; ``spread`` (``1..180`` degrees, default 180) narrows them.
    * ``units`` -- ``"aether"`` (0.x: the radiance factor) or ``"watts"`` (Blender's
      light power: suns in W/m^2, others in W).
    * ``temperature`` -- kelvins (``800..12000``, default 6500); with
      ``use_temperature`` the light's colour is multiplied by the black body's
      (:mod:`~aether3d.rendering.blackbody`, luminance 1).
    * ``exposure`` -- scales the intensity by ``2^exposure``.
    * ``shadow`` -- ``False``: nothing blocks this light (path tracers trace no shadow
      rays for it; shadow maps skip it).
    * ``include`` / ``exclude`` -- light linking: object ids this light lights only /
      does not light (each with its descendants); empty by default (every object).
    """

    __slots__ = (*(f"_{name}" for name in _FIELDS), "_owner")

    def __init__(
        self,
        type: str = "directional",
        color: Any = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        range: float | None = None,
        falloff: str = "inverse_square",
        radius: float = 0.0,
        angle: float = 0.0,
        **more: Any,
    ) -> None:
        unknown = sorted(set(more) - set(DEFAULTS_0_20))
        if unknown:
            raise RenderError(
                "INVALID_LIGHT",
                f"unknown light parameter(s): {', '.join(unknown)}",
                "light",
                hint=f"parameters: {', '.join(_FIELDS)}",
            )
        values = _validate(
            {
                "type": type,
                "color": color,
                "intensity": intensity,
                "range": range,
                "falloff": falloff,
                "radius": radius,
                "angle": angle,
                **DEFAULTS_0_20,
                **more,
            },
            "light",
        )
        for name in _FIELDS:
            setattr(self, f"_{name}", values[name])
        self._owner = None  # the object this belongs to (records its changes)

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
        with recording(self, changes):
            for name in _FIELDS:
                setattr(self, f"_{name}", values[name])
        return self

    type = property(lambda self: self._type, lambda self, v: self.configure(type=v))
    color = property(lambda self: self._color, lambda self, v: self.configure(color=v))
    intensity = property(lambda self: self._intensity, lambda self, v: self.configure(intensity=v))
    range = property(lambda self: self._range, lambda self, v: self.configure(range=v))
    falloff = property(lambda self: self._falloff, lambda self, v: self.configure(falloff=v))
    radius = property(lambda self: self._radius, lambda self, v: self.configure(radius=v))
    angle = property(lambda self: self._angle, lambda self, v: self.configure(angle=v))
    spot_size = property(lambda self: self._spot_size, lambda self, v: self.configure(spot_size=v))
    spot_blend = property(
        lambda self: self._spot_blend, lambda self, v: self.configure(spot_blend=v)
    )
    shape = property(lambda self: self._shape, lambda self, v: self.configure(shape=v))
    size = property(lambda self: self._size, lambda self, v: self.configure(size=v))
    size_y = property(lambda self: self._size_y, lambda self, v: self.configure(size_y=v))
    spread = property(lambda self: self._spread, lambda self, v: self.configure(spread=v))
    units = property(lambda self: self._units, lambda self, v: self.configure(units=v))
    temperature = property(
        lambda self: self._temperature, lambda self, v: self.configure(temperature=v)
    )
    use_temperature = property(
        lambda self: self._use_temperature, lambda self, v: self.configure(use_temperature=v)
    )
    exposure = property(lambda self: self._exposure, lambda self, v: self.configure(exposure=v))
    shadow = property(lambda self: self._shadow, lambda self, v: self.configure(shadow=v))
    include = property(lambda self: self._include, lambda self, v: self.configure(include=v))
    exclude = property(lambda self: self._exclude, lambda self, v: self.configure(exclude=v))

    def area(self) -> float:
        """An area light's emitting area in square metres (0 for other lights)."""
        if self._type != "area":
            return 0.0
        w = self._size
        h = self._size if self._shape in ("square", "disk") else self._size_y
        return w * h * (math.pi / 4.0 if self._shape in ("disk", "ellipse") else 1.0)

    def intensity_value(self) -> float:
        """The intensity in Aether3D's units after ``units`` and ``exposure``: what the
        renderers' formulas take (a facing white surface lit to radiance ``I`` at 1 m)."""
        i = self._intensity * 2.0**self._exposure
        if self._units == "watts":
            if self._type == "directional":
                i = i / math.pi
            elif self._type == "area":
                i = i / (math.pi * math.pi)
            else:
                i = i / (4.0 * math.pi * math.pi)
        return i

    def linear_color(self) -> tuple[float, float, float]:
        """The colour in linear light, times the black body's when ``use_temperature``."""
        from .blackbody import blackbody
        from .color import to_linear

        c = to_linear(self._color)
        if self._use_temperature:
            b = blackbody(self._temperature)
            return (float(c[0] * b[0]), float(c[1] * b[1]), float(c[2] * b[2]))
        return (float(c[0]), float(c[1]), float(c[2]))

    def copy(self) -> Light:
        return Light(**self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, f"_{name}") for name in _FIELDS}
        data["color"] = list(self._color)
        data["include"] = list(self._include)
        data["exclude"] = list(self._exclude)
        return data

    @classmethod
    def from_dict(cls, data: Any, location: str = "light") -> Light:
        if not isinstance(data, dict):
            raise RenderError("INVALID_LIGHT", "expected an object", location)
        # (the fields of 0.20.0 default when missing: documents written before them)
        missing = [f for f in _FIELDS if f not in data and f not in DEFAULTS_0_20]
        unknown = sorted(set(data) - set(_FIELDS))
        if missing or unknown:
            what = f"missing {', '.join(missing)}" if missing else f"unknown {', '.join(unknown)}"
            raise RenderError("INVALID_LIGHT", f"light keys: {what}", location)
        light = cls.__new__(cls)
        values = _validate({**DEFAULTS_0_20, **data}, location)
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
        if self._type == "spot":
            extra = f", spot_size={self._spot_size:g}, blend={self._spot_blend:g}"
        elif self._type == "area":
            extra = f", {self._shape} {self._size:g}x{self._size_y:g}"
        elif self._type == "point":
            extra = f", falloff={self._falloff}" + (
                f", range={self._range:g}" if self._range is not None else ""
            )
            if self._radius:
                extra += f", radius={self._radius:g}"
        elif self._angle:
            extra = f", angle={self._angle:g}"
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
    sizes = {}
    for name, hi in (("radius", 1e9), ("angle", MAX_ANGLE)):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= hi
        ):
            raise RenderError(
                "INVALID_LIGHT",
                f"{name} must be a finite number in [0, {hi:g}], got {value!r}",
                f"{location}.{name}",
            )
        sizes[name] = float(value)
    out = {
        "type": values["type"],
        "color": color,
        "intensity": float(intensity),
        "range": rng,
        "falloff": values["falloff"],
        **sizes,
    }
    for name, lo, hi in (
        ("spot_size", 0.1, 180.0),
        ("spot_blend", 0.0, 1.0),
        ("size", 1e-6, MAX_SIZE),
        ("size_y", 1e-6, MAX_SIZE),
        ("spread", 1.0, 180.0),
        ("temperature", 100.0, 100_000.0),
        ("exposure", -40.0, 40.0),
    ):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or not lo <= float(value) <= hi
        ):
            raise RenderError(
                "INVALID_LIGHT",
                f"{name} must be a number in [{lo:g}, {hi:g}], got {value!r}",
                f"{location}.{name}",
            )
        out[name] = float(value)
    for name, table in (("shape", SHAPES), ("units", UNITS)):
        if values[name] not in table:
            raise RenderError(
                "INVALID_LIGHT",
                f"{name} must be one of {', '.join(table)}, got {values[name]!r}",
                f"{location}.{name}",
            )
        out[name] = values[name]
    for name in ("use_temperature", "shadow"):
        if not isinstance(values[name], bool):
            raise RenderError(
                "INVALID_LIGHT", f"{name} must be true or false", f"{location}.{name}"
            )
        out[name] = values[name]
    for name in ("include", "exclude"):
        ids = values[name]
        if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple)):
            raise RenderError(
                "INVALID_LIGHT", f"{name} must be a list of object ids", f"{location}.{name}"
            )
        if len(ids) > MAX_LINKS or not all(isinstance(i, str) and i for i in ids):
            raise RenderError(
                "INVALID_LIGHT",
                f"{name} must be a list of at most {MAX_LINKS} object ids (strings)",
                f"{location}.{name}",
                hint="link objects by id: light.include = [obj.id]",
            )
        out[name] = tuple(dict.fromkeys(ids))  # in order, without repeats
    return out
