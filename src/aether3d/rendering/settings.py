"""Object appearance, world (background and ambient light), render settings and limits.

All colors are display (sRGB) colors; see :mod:`aether3d.rendering.color`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import RenderError
from ..history import recording
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
    """Small base for validated value objects with ``configure`` and dict round trips.

    A value object that belongs to a scene (an object's appearance, the world, the
    render settings) has an owner, and its changes are recorded in the scene's history.
    """

    _FIELDS: tuple[str, ...] = ()
    _CODE = ""
    __slots__ = ("_owner",)

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
        values = self._check({**self.to_dict(), **changes}, self._location())
        with recording(self, changes):
            self._assign(values)
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


ENVIRONMENT_TYPES = ("ambient", "uniform", "sky")
DEFAULT_ZENITH: Color = (0.22, 0.42, 0.78)
DEFAULT_HORIZON: Color = (0.78, 0.85, 0.93)
DEFAULT_GROUND: Color = (0.32, 0.30, 0.27)


class Environment(_Validated):
    """Light arriving from infinitely far away (``scene.world.environment``).

    Used by the path tracer (``render(backend="path")``); the raster backends
    keep using the world's background and ambient light unchanged.

    * ``type`` -- ``"ambient"`` (default): a uniform environment whose
      radiance is the world's ambient light (``ambient_color x
      ambient_intensity``, linear), the physical reading of the raster
      ambient term; camera rays that miss show ``world.background``.
      ``"uniform"``: radiance ``color x intensity`` from every direction.
      ``"sky"``: a procedural gradient from ``ground_color`` (below the
      horizon) through ``horizon_color`` to ``zenith_color`` (straight up,
      +Z), times ``intensity``. With ``"uniform"`` and ``"sky"`` every ray
      that misses the scene -- camera rays included -- sees the environment.
    * ``color`` -- display color of the uniform environment.
    * ``intensity`` -- radiance multiplier (``>= 0``) of ``"uniform"`` and ``"sky"``.
    * ``zenith_color``, ``horizon_color``, ``ground_color`` -- display colors of the sky.

    See ``docs/RENDER_QUALITY.md`` for the sky formula.
    """

    _FIELDS = ("type", "color", "intensity", "zenith_color", "horizon_color", "ground_color")
    _CODE = "INVALID_WORLD"
    __slots__ = tuple(f"_{name}" for name in _FIELDS)

    def __init__(
        self,
        type: str = "ambient",
        color: Any = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        zenith_color: Any = DEFAULT_ZENITH,
        horizon_color: Any = DEFAULT_HORIZON,
        ground_color: Any = DEFAULT_GROUND,
    ) -> None:
        self._assign(
            self._check(
                {
                    "type": type,
                    "color": color,
                    "intensity": intensity,
                    "zenith_color": zenith_color,
                    "horizon_color": horizon_color,
                    "ground_color": ground_color,
                },
                "world.environment",
            )
        )

    def _location(self) -> str:
        return "world.environment"

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        kind = values["type"]
        if kind not in ENVIRONMENT_TYPES:
            raise RenderError(
                self._CODE,
                f"type must be one of {', '.join(ENVIRONMENT_TYPES)}, got {kind!r}",
                f"{location}.type",
            )
        out: dict[str, Any] = {"type": kind}
        for name in ("color", "zenith_color", "horizon_color", "ground_color"):
            out[name] = _color(values[name], self._CODE, f"{location}.{name}")
        out["intensity"] = _number(
            values["intensity"], self._CODE, "intensity", f"{location}.intensity", 0.0, 1e6
        )
        return {name: out[name] for name in self._FIELDS}

    type = _prop("type")
    color = _prop("color")
    intensity = _prop("intensity")
    zenith_color = _prop("zenith_color")
    horizon_color = _prop("horizon_color")
    ground_color = _prop("ground_color")


class World(_Validated):
    """Scene-wide background, ambient light and environment (``scene.world``).

    * ``background`` -- display (sRGB) color behind all geometry.
    * ``ambient_color`` -- display color of the uniform ambient light.
    * ``ambient_intensity`` -- unitless ambient strength (``>= 0``): every
      surface receives ``base color x ambient color x ambient_intensity``
      (linear light), regardless of orientation.
    * ``environment`` -- an :class:`Environment`: what the path tracer sees
      and is lit by where rays leave the scene. Assign an ``Environment`` or
      a dict of its fields (missing fields keep their current values).
    """

    _FIELDS = ("background", "ambient_color", "ambient_intensity", "environment")
    _CODE = "INVALID_WORLD"
    __slots__ = ("_background", "_ambient_color", "_ambient_intensity", "_environment")

    def __init__(
        self,
        background: Any = DEFAULT_BACKGROUND,
        ambient_color: Any = (1.0, 1.0, 1.0),
        ambient_intensity: float = 0.1,
        environment: Any = None,
    ) -> None:
        self._assign(
            self._check(
                {
                    "background": background,
                    "ambient_color": ambient_color,
                    "ambient_intensity": ambient_intensity,
                    "environment": Environment() if environment is None else environment,
                },
                "world",
            )
        )

    def configure(self, **changes: Any):
        env = changes.get("environment")
        if isinstance(env, dict):  # partial updates keep the other environment fields
            changes = {**changes, "environment": {**self._environment.to_dict(), **env}}
        return super().configure(**changes)

    def _assign(self, values: dict[str, Any]) -> None:
        super()._assign(values)
        self._environment._owner = (self, None)  # its changes are changes of the world

    def _edit_value(self, aspect: Any, fields: Any) -> Any:
        return recording(self, [f"environment.{f}" for f in fields])

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["environment"] = self._environment.to_dict()
        return data

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        env = values["environment"]
        if isinstance(env, Environment):
            env = env.copy()
        elif isinstance(env, dict):
            env = Environment.from_dict(env, f"{location}.environment")
        else:
            raise RenderError(
                self._CODE,
                f"environment must be an Environment or a dict, got {env!r}",
                f"{location}.environment",
            )
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
            "environment": env,
        }

    background = _prop("background")
    ambient_color = _prop("ambient_color")
    ambient_intensity = _prop("ambient_intensity")
    environment = _prop("environment")


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


MAX_SHADOW_MAPS = 4
"""Most directional lights that cast shadows in one render (one shadow map each)."""


class ShadowSettings(_Validated):
    """Shadow-map settings for one render (GPU backend only; not saved in projects).

    Pass ``shadows=True`` (these defaults), a ``ShadowSettings`` or a dict of
    its fields to :func:`~aether3d.rendering.render`.

    * ``lights`` -- ``None`` (default): every directional light casts shadows
      (at most :data:`MAX_SHADOW_MAPS`, the brightest first); or a list of
      directional light object ids or names.
    * ``map_size`` -- width and height of each shadow map in texels
      (256-16384, default 2048).
    * ``bias`` -- constant depth bias in shadow-map texels (default 0.5).
    * ``normal_offset`` -- extra texels the receiving point moves along its
      face normal (scaled by the sine of the light angle), beyond the
      ``sqrt(2) * (pcf_radius + 1)`` texels that keep flat surfaces from
      shadowing themselves (default 0.5).
    * ``pcf_radius`` -- percentage-closer filtering: ``(2r + 1)^2`` bilinear
      depth comparisons around each point (0-4, default 1; 0 is the
      hardest edge the map can give).

    See ``docs/GPU.md`` for how the shadow map is fitted and what these mean.
    """

    _FIELDS = ("lights", "map_size", "bias", "normal_offset", "pcf_radius")
    _CODE = "INVALID_SHADOWS"
    __slots__ = ("_lights", "_map_size", "_bias", "_normal_offset", "_pcf_radius")

    def __init__(
        self,
        lights: Any = None,
        map_size: int = 2048,
        bias: float = 0.5,
        normal_offset: float = 0.5,
        pcf_radius: int = 1,
    ) -> None:
        self._assign(
            self._check(
                {
                    "lights": lights,
                    "map_size": map_size,
                    "bias": bias,
                    "normal_offset": normal_offset,
                    "pcf_radius": pcf_radius,
                },
                "shadows",
            )
        )

    def _location(self) -> str:
        return "shadows"

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        lights = values["lights"]
        if lights is not None:
            if isinstance(lights, str) or not isinstance(lights, (list, tuple)):
                raise RenderError(
                    self._CODE,
                    f"lights must be null or a list of light ids or names, got {lights!r}",
                    f"{location}.lights",
                )
            if len(lights) > MAX_SHADOW_MAPS:
                raise RenderError(
                    self._CODE,
                    f"at most {MAX_SHADOW_MAPS} lights can cast shadows, got {len(lights)}",
                    f"{location}.lights",
                )
            for i, ref in enumerate(lights):
                if not isinstance(ref, str) or not ref:
                    raise RenderError(
                        self._CODE,
                        f"light references must be non-empty strings, got {ref!r}",
                        f"{location}.lights[{i}]",
                    )
            lights = tuple(lights)
        return {
            "lights": lights,
            "map_size": _integer(
                values["map_size"], self._CODE, "map_size", f"{location}.map_size", 256, 16384
            ),
            "bias": _number(values["bias"], self._CODE, "bias", f"{location}.bias", 0.0, 100.0),
            "normal_offset": _number(
                values["normal_offset"],
                self._CODE,
                "normal_offset",
                f"{location}.normal_offset",
                0.0,
                100.0,
            ),
            "pcf_radius": _integer(
                values["pcf_radius"], self._CODE, "pcf_radius", f"{location}.pcf_radius", 0, 4
            ),
        }

    lights = _prop("lights")
    map_size = _prop("map_size")
    bias = _prop("bias")
    normal_offset = _prop("normal_offset")
    pcf_radius = _prop("pcf_radius")

    @classmethod
    def coerce(cls, value: Any) -> ShadowSettings | None:
        """``None``/``False`` -> no shadows; ``True`` -> defaults; a dict or instance as given."""
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, cls):
            return value.copy()
        if isinstance(value, dict):
            unknown = sorted(set(value) - set(cls._FIELDS))
            if unknown:
                raise RenderError(
                    cls._CODE,
                    f"unknown shadow setting(s): {', '.join(unknown)}",
                    "shadows",
                    hint=f"settings: {', '.join(cls._FIELDS)}",
                )
            return cls(**value)
        raise RenderError(
            cls._CODE,
            f"shadows must be true, false, a dict or a ShadowSettings, got {value!r}",
            "shadows",
        )


INTEGRATORS = ("path", "ao")
MAX_PATH_SAMPLES = 1_000_000
MAX_BOUNCES = 64


class PathSettings(_Validated):
    """Path tracer settings for one render (``backend="path"``; not saved in projects).

    Pass a :class:`PathSettings` or a dict of its fields as ``path=`` to
    :func:`~aether3d.rendering.render`.

    * ``samples`` -- paths traced per pixel (default 16). Noise falls as
      ``1 / sqrt(samples)``.
    * ``bounces`` -- indirect bounces (default 4): ``0`` is direct lighting
      only (camera hit, lights and emitters seen from it); each bounce adds
      light that reflected once more.
    * ``seed`` -- random seed (default 0). The same scene, settings and seed
      give the same image.
    * ``light_samples`` -- shadow rays per light (and per emissive-surface
      sample) at every path vertex (default 1).
    * ``russian_roulette`` -- the path vertex (0: the surface the camera
      sees) from which dim paths may stop early, with their survivors
      weighted up (unbiased; default 3).
    * ``clamp`` -- largest radiance one sample may carry in indirect light
      (``0``: off, the default). Clamping removes fireflies but loses energy
      (biased).
    * ``integrator`` -- ``"path"`` (default) or ``"ao"``: a fast preview that
      traces camera rays and ambient occlusion only (the image is the AO
      pass; see ``ao``).
    * ``workers`` -- CPU processes to trace with: ``0`` (default) picks one
      per CPU (at most 8) for renders big enough to gain from it, ``1``
      traces in the calling process, ``N`` uses ``N``. Parallel tracing forks
      processes and is available on Linux; elsewhere renders use one
      process. The image does not depend on it.
    """

    _FIELDS = (
        "samples",
        "bounces",
        "seed",
        "light_samples",
        "russian_roulette",
        "clamp",
        "integrator",
        "workers",
    )
    _CODE = "INVALID_PATH_SETTINGS"
    __slots__ = tuple(f"_{name}" for name in _FIELDS)

    def __init__(
        self,
        samples: int = 16,
        bounces: int = 4,
        seed: int = 0,
        light_samples: int = 1,
        russian_roulette: int = 3,
        clamp: float = 0.0,
        integrator: str = "path",
        workers: int = 0,
    ) -> None:
        self._assign(
            self._check(
                {
                    "samples": samples,
                    "bounces": bounces,
                    "seed": seed,
                    "light_samples": light_samples,
                    "russian_roulette": russian_roulette,
                    "clamp": clamp,
                    "integrator": integrator,
                    "workers": workers,
                },
                "path",
            )
        )

    def _location(self) -> str:
        return "path"

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        integrator = values["integrator"]
        if integrator not in INTEGRATORS:
            raise RenderError(
                self._CODE,
                f"integrator must be one of {', '.join(INTEGRATORS)}, got {integrator!r}",
                f"{location}.integrator",
            )
        c = self._CODE
        return {
            "samples": _integer(
                values["samples"], c, "samples", f"{location}.samples", 1, MAX_PATH_SAMPLES
            ),
            "bounces": _integer(
                values["bounces"], c, "bounces", f"{location}.bounces", 0, MAX_BOUNCES
            ),
            "seed": _integer(values["seed"], c, "seed", f"{location}.seed", 0, 2**63 - 1),
            "light_samples": _integer(
                values["light_samples"], c, "light_samples", f"{location}.light_samples", 1, 64
            ),
            "russian_roulette": _integer(
                values["russian_roulette"],
                c,
                "russian_roulette",
                f"{location}.russian_roulette",
                0,
                MAX_BOUNCES,
            ),
            "clamp": _number(values["clamp"], c, "clamp", f"{location}.clamp", 0.0, 1e12),
            "integrator": integrator,
            "workers": _integer(values["workers"], c, "workers", f"{location}.workers", 0, 256),
        }

    samples = _prop("samples")
    bounces = _prop("bounces")
    seed = _prop("seed")
    light_samples = _prop("light_samples")
    russian_roulette = _prop("russian_roulette")
    clamp = _prop("clamp")
    integrator = _prop("integrator")
    workers = _prop("workers")

    @classmethod
    def coerce(cls, value: Any) -> PathSettings:
        """``None`` -> defaults; a dict or instance as given."""
        return _coerce(cls, value, "path")


class AOSettings(_Validated):
    """Ambient occlusion for one render (``backend="path"``; not saved in projects).

    AO is the cosine-weighted fraction of the hemisphere above the surface
    seen by each camera ray that is blocked by geometry within ``distance``.
    It is an artistic approximation, not light transport: it is computed as
    a separate ``"ao"`` pass and never multiplied into the path-traced image.
    Pass ``ao=True``, an :class:`AOSettings` or a dict of its fields.

    * ``samples`` -- AO rays per camera sample (default 4).
    * ``distance`` -- reach in scene units (default 1): geometry farther than
      this does not occlude.
    * ``strength`` -- ``0..1`` (default 1): the pass stores ``1 - strength x
      occlusion``.

    Rays use the path settings' ``seed``, so AO is reproducible too.
    """

    _FIELDS = ("samples", "distance", "strength")
    _CODE = "INVALID_AO_SETTINGS"
    __slots__ = ("_samples", "_distance", "_strength")

    def __init__(self, samples: int = 4, distance: float = 1.0, strength: float = 1.0) -> None:
        self._assign(
            self._check({"samples": samples, "distance": distance, "strength": strength}, "ao")
        )

    def _location(self) -> str:
        return "ao"

    def _check(self, values: dict[str, Any], location: str) -> dict[str, Any]:
        c = self._CODE
        distance = _number(values["distance"], c, "distance", f"{location}.distance", 0.0, 1e12)
        if distance <= 0.0:
            raise RenderError(c, "distance must be > 0", f"{location}.distance")
        return {
            "samples": _integer(values["samples"], c, "samples", f"{location}.samples", 1, 1024),
            "distance": distance,
            "strength": _number(
                values["strength"], c, "strength", f"{location}.strength", 0.0, 1.0
            ),
        }

    samples = _prop("samples")
    distance = _prop("distance")
    strength = _prop("strength")

    @classmethod
    def coerce(cls, value: Any) -> AOSettings | None:
        """``None``/``False`` -> no AO; ``True`` -> defaults; a dict or instance as given."""
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        return _coerce(cls, value, "ao")


def _coerce(cls, value: Any, location: str):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value.copy()
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(cls._FIELDS))
        if unknown:
            raise RenderError(
                cls._CODE,
                f"unknown {location} setting(s): {', '.join(unknown)}",
                location,
                hint=f"settings: {', '.join(cls._FIELDS)}",
            )
        return cls(**value)
    raise RenderError(
        cls._CODE,
        f"{location} must be a dict or a {cls.__name__}, got {value!r}",
        location,
    )


@dataclass(frozen=True)
class RenderLimits:
    """Safety limits checked before any memory is allocated (raise ``RESOURCE_LIMIT``).

    * ``max_dimension`` -- largest image width or height in pixels.
    * ``max_samples`` -- largest number of supersamples (``width x height x samples^2``);
      the renderer needs about 8 bytes per sample plus a few bytes per pixel.
    * ``max_triangles`` -- largest number of triangles to render (counting every instance).
    * ``max_gpu_memory`` -- largest estimated GPU memory, in bytes, for one
      GPU render's images, shadow maps and output buffers (meshes are cached
      separately within their own budget).
    * ``max_path_rays`` -- largest number of rays one path-traced render may
      trace (an upper estimate from resolution, samples, bounces and lights).
    * ``max_path_memory`` -- largest estimated host memory, in bytes, of one
      path-traced render (accumulation buffers, shading data, the BVH and one
      batch of paths).
    """

    max_dimension: int = 16384
    max_samples: int = 40_000_000
    max_triangles: int = 20_000_000
    max_gpu_memory: int = 4 * 2**30
    max_path_rays: int = 100_000_000_000
    max_path_memory: int = 8 * 2**30
