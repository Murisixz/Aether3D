"""Object appearance, world (background and ambient light), render settings and limits.

All colors are display (sRGB) colors; see :mod:`aether3d.rendering.color`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import RenderError
from .color import Color, parse_color

DEFAULT_COLOR: Color = (0.8, 0.8, 0.8)
DEFAULT_BACKGROUND: Color = (0.2, 0.2, 0.2)
CULL_MODES = ("none", "back", "front")
MAX_SAMPLES_PER_AXIS = 4


def _number(value: Any, code: str, field: str, location: str, lo: float, hi: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not lo <= float(value) <= hi
    ):
        raise RenderError(
            code, f"{field} must be a finite number in [{lo:g}, {hi:g}], got {value!r}", location
        )
    return float(value)


def _integer(value: Any, code: str, field: str, location: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise RenderError(
            code, f"{field} must be an integer in [{lo}, {hi}], got {value!r}", location
        )
    return int(value)


def _color(value: Any, code: str, location: str) -> Color:
    try:
        return parse_color(value, location)
    except RenderError as exc:
        raise RenderError(code, exc.message, exc.location) from None


class _Validated:
    """Small base for validated value objects with ``configure`` and dict round trips."""

    _FIELDS: tuple[str, ...] = ()
    _CODE = ""
    __slots__ = ()

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        raise NotImplementedError

    def _assign(self, values: dict[str, Any]) -> None:
        for name in self._FIELDS:
            object.__setattr__(self, f"_{name}", values[name])

    def configure(self, **changes: Any):
        """Change several values at once; validated together (all or nothing)."""
        unknown = sorted(set(changes) - set(self._FIELDS))
        if unknown:
            raise RenderError(
                self._CODE,
                f"unknown parameter(s): {', '.join(unknown)}",
                hint=f"parameters: {', '.join(self._FIELDS)}",
            )
        self._assign(self._check({**self.to_dict(), **changes}, self._location()))
        return self

    def _location(self) -> str:
        return type(self).__name__.lower()

    def to_dict(self) -> dict[str, Any]:
        data = {}
        for name in self._FIELDS:
            value = getattr(self, f"_{name}")
            data[name] = list(value) if isinstance(value, tuple) else value
        return data

    @classmethod
    def from_dict(cls, data: Any, location: str):
        if not isinstance(data, dict):
            raise RenderError(cls._CODE, "expected an object", location)
        missing = [f for f in cls._FIELDS if f not in data]
        unknown = sorted(set(data) - set(cls._FIELDS))
        if missing or unknown:
            what = f"missing {', '.join(missing)}" if missing else f"unknown {', '.join(unknown)}"
            raise RenderError(cls._CODE, f"keys: {what}", location)
        obj = cls.__new__(cls)
        obj._assign(obj._check(dict(data), location))
        return obj

    def copy(self):
        return type(self).from_dict(self.to_dict(), self._location())

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.to_dict() == other.to_dict()  # type: ignore[union-attr]

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{type(self).__name__}({fields})"


def _prop(name: str):
    return property(
        lambda self: getattr(self, f"_{name}"), lambda self, v: self.configure(**{name: v})
    )


class Appearance(_Validated):
    """How a mesh object looks: a solid base color plus an optional simple highlight.

    * ``color`` -- display (sRGB) base color (default light gray ``(0.8, 0.8, 0.8)``).
    * ``specular`` -- strength of a Blinn-Phong highlight in ``[0, 1]`` (default 0: matte).
    * ``shininess`` -- highlight exponent in ``[1, 10000]`` (larger = smaller, sharper).

    This is not a material system: there are no textures or physically based
    parameters yet.
    """

    _FIELDS = ("color", "specular", "shininess")
    _CODE = "INVALID_APPEARANCE"
    __slots__ = ("_color", "_specular", "_shininess")

    def __init__(self, color: Any = DEFAULT_COLOR, specular: float = 0.0, shininess: float = 32.0):
        self._assign(
            self._check(
                {"color": color, "specular": specular, "shininess": shininess}, "appearance"
            )
        )

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        return {
            "color": _color(values["color"], self._CODE, f"{location}.color"),
            "specular": _number(
                values["specular"], self._CODE, "specular", f"{location}.specular", 0.0, 1.0
            ),
            "shininess": _number(
                values["shininess"], self._CODE, "shininess", f"{location}.shininess", 1.0, 10000.0
            ),
        }

    color = _prop("color")
    specular = _prop("specular")
    shininess = _prop("shininess")

    @property
    def is_default(self) -> bool:
        return self == Appearance()


class World(_Validated):
    """Scene-wide background and ambient light (``scene.world``).

    * ``background`` -- display (sRGB) color behind all geometry.
    * ``ambient_color`` -- display color of the uniform ambient light.
    * ``ambient_intensity`` -- unitless ambient strength (``>= 0``): every
      surface receives ``base color x ambient color x ambient_intensity``
      (linear light), regardless of orientation.
    """

    _FIELDS = ("background", "ambient_color", "ambient_intensity")
    _CODE = "INVALID_WORLD"
    __slots__ = ("_background", "_ambient_color", "_ambient_intensity")

    def __init__(
        self,
        background: Any = DEFAULT_BACKGROUND,
        ambient_color: Any = (1.0, 1.0, 1.0),
        ambient_intensity: float = 0.1,
    ) -> None:
        self._assign(
            self._check(
                {
                    "background": background,
                    "ambient_color": ambient_color,
                    "ambient_intensity": ambient_intensity,
                },
                "world",
            )
        )

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        return {
            "background": _color(values["background"], self._CODE, f"{location}.background"),
            "ambient_color": _color(
                values["ambient_color"], self._CODE, f"{location}.ambient_color"
            ),
            "ambient_intensity": _number(
                values["ambient_intensity"],
                self._CODE,
                "ambient_intensity",
                f"{location}.ambient_intensity",
                0.0,
                1e6,
            ),
        }

    background = _prop("background")
    ambient_color = _prop("ambient_color")
    ambient_intensity = _prop("ambient_intensity")


class RenderSettings(_Validated):
    """Persistent output settings (``scene.render_settings``); any can be overridden per render.

    * ``width``, ``height`` -- image size in pixels.
    * ``samples`` -- antialiasing: ``samples x samples`` supersamples per pixel
      (1-4), averaged in linear light.
    * ``exposure`` -- exposure in stops (EV): linear values are multiplied by
      ``2 ** exposure`` before display encoding.
    * ``cull`` -- ``"none"`` (default; both sides of every triangle render,
      back faces lit with flipped normals), ``"back"`` or ``"front"``.
    * ``default_lights`` -- when the scene has no light objects, light it with
      a built-in key light from the camera's upper left (reported as
      ``DEFAULT_LIGHTING``). Set to ``False`` to render with ambient light only.
    """

    _FIELDS = ("width", "height", "samples", "exposure", "cull", "default_lights")
    _CODE = "INVALID_SETTINGS"
    __slots__ = ("_width", "_height", "_samples", "_exposure", "_cull", "_default_lights")

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        samples: int = 2,
        exposure: float = 0.0,
        cull: str = "none",
        default_lights: bool = True,
    ) -> None:
        self._assign(
            self._check(
                {
                    "width": width,
                    "height": height,
                    "samples": samples,
                    "exposure": exposure,
                    "cull": cull,
                    "default_lights": default_lights,
                },
                "render_settings",
            )
        )

    def _location(self) -> str:
        return "render_settings"

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        cull = values["cull"]
        if cull not in CULL_MODES:
            raise RenderError(
                self._CODE,
                f"cull must be one of {', '.join(CULL_MODES)}, got {cull!r}",
                f"{location}.cull",
            )
        if not isinstance(values["default_lights"], bool):
            raise RenderError(
                self._CODE,
                f"default_lights must be true or false, got {values['default_lights']!r}",
                f"{location}.default_lights",
            )
        return {
            "width": _integer(
                values["width"],
                self._CODE,
                "width",
                f"{location}.width",
                1,
                RenderLimits.max_dimension,
            ),
            "height": _integer(
                values["height"],
                self._CODE,
                "height",
                f"{location}.height",
                1,
                RenderLimits.max_dimension,
            ),
            "samples": _integer(
                values["samples"],
                self._CODE,
                "samples",
                f"{location}.samples",
                1,
                MAX_SAMPLES_PER_AXIS,
            ),
            "exposure": _number(
                values["exposure"], self._CODE, "exposure", f"{location}.exposure", -20.0, 20.0
            ),
            "cull": cull,
            "default_lights": values["default_lights"],
        }

    width = _prop("width")
    height = _prop("height")
    samples = _prop("samples")
    exposure = _prop("exposure")
    cull = _prop("cull")
    default_lights = _prop("default_lights")


@dataclass(frozen=True)
class RenderLimits:
    """Safety limits checked before any memory is allocated (raise ``RESOURCE_LIMIT``).

    * ``max_dimension`` -- largest image width or height in pixels.
    * ``max_samples`` -- largest number of supersamples (``width x height x samples^2``);
      the renderer needs about 8 bytes per sample plus a few bytes per pixel.
    * ``max_triangles`` -- largest number of triangles to render (counting every instance).
    """

    max_dimension: int = 16384
    max_samples: int = 40_000_000
    max_triangles: int = 20_000_000
