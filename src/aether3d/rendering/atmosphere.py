"""The outdoor atmosphere on the CPU: sky with a sun glow, height fog, hemisphere ambient.

``scene.world.atmosphere`` (:class:`~aether3d.rendering.settings.Atmosphere`) is resolved
once per render into an :class:`AtmosphereModel` (linear colours, the sun direction, the
camera-relative fog height). The CPU rasterizer and the CPU path tracer use these
functions; the GPU backends evaluate the same formulas in ``gpu/wgsl/atmosphere.wgsl``.

* **Sky** (direction ``d``, ``z = d.z``): the environment -- for ``"sky"``,
  ``mix(horizon, zenith, sqrt(z))`` above the horizon and ``mix(horizon, ground,
  min(1, 4|z|))`` below, times ``intensity``; ``"uniform"``: its colour; ``"ambient"``: the
  world's background -- plus the sun glow ``glow * (g^size + 0.15 g^(size / 8))``,
  ``g = max(d . sun, 0)``, plus the horizon glow ``horizon_glow * glow_colour *
  (1 - min(2.5 |z|, 1))^4 * (0.25 + 0.75 a^2)``, ``a`` the cosine between ``d``'s and the
  sun's horizontal directions.
* **Fog** along a camera ray from the eye ``c`` in direction ``d`` to distance ``t``:
  density ``rho(z) = density * e^(-falloff (z - height))`` integrates exactly to
  ``tau = rho(c.z) (1 - e^(-falloff d.z t)) / (falloff d.z)`` (``rho(c.z) t`` when
  ``falloff d.z t`` is tiny; ``t = inf`` gives ``rho(c.z) / (falloff d.z)`` looking up and
  infinite optical depth otherwise); the colour becomes ``L T + F (1 - T)`` with
  ``T = e^(-tau)`` and ``F = fog_colour + fog_sun * sun_colour * g^power``.
* **Hemisphere ambient** (rasterizers): ``mix(ground_ambient, ambient, 0.5 + 0.5 n.z)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color import to_linear

ENV_KINDS = {"ambient": 0, "uniform": 1, "sky": 2}


@dataclass(frozen=True)
class AtmosphereModel:
    sun: np.ndarray | None  # (3,) unit, toward the sun
    env_kind: int
    zenith: np.ndarray  # linear x intensity ("uniform": the colour)
    horizon: np.ndarray
    ground: np.ndarray
    background: np.ndarray
    glow: np.ndarray  # linear glow colour x strength
    glow_size: float
    horizon_glow: float
    sky_background: bool
    fog_density: float
    fog_falloff: float
    fog_rel: float  # eye z - fog height
    fog_color: np.ndarray
    fog_sun: np.ndarray  # linear sun colour x strength
    fog_power: float
    hemisphere: bool
    ground_ambient: np.ndarray  # linear x ambient intensity

    @property
    def fog(self) -> bool:
        return self.fog_density > 0.0


def resolve(rs: Any) -> AtmosphereModel | None:
    """The render scene's atmosphere (``rs.atmosphere``: the settings and the world's
    ambient intensity), or ``None`` when it is off."""
    atm = getattr(rs, "atmosphere", None)
    if atm is None or not atm["settings"].enabled:
        return None
    return build(
        atm["settings"],
        rs.environment,
        rs.background,
        rs.lights,
        rs.camera.frame.position,
        atm["ambient_intensity"],
    )


def sun_light(atm: Any, lights: Any) -> Any:
    """The light giving the sun's direction (named, or the brightest directional one)."""
    directional = [li for li in lights if li.type == "directional"]
    if atm.sun is not None:
        for li in directional:
            if atm.sun in (li.name, li.object_id):
                return li
        return None
    if not directional:
        return None
    lum = [
        0.2126 * li.radiance[0] + 0.7152 * li.radiance[1] + 0.0722 * li.radiance[2]
        for li in directional
    ]
    return directional[int(np.argmax(lum))]


def build(
    atm: Any,
    env: Any,
    background: np.ndarray,
    lights: Any,
    eye: np.ndarray,
    ambient_intensity: float,
) -> AtmosphereModel:
    li = sun_light(atm, lights)
    sun = None if li is None else -np.asarray(li.direction, dtype=np.float64)
    kind = ENV_KINDS.get(getattr(env, "type", "ambient"), 0)
    intensity = float(getattr(env, "intensity", 1.0)) if env is not None else 1.0
    if kind == 2:
        zenith = to_linear(env.zenith_color) * intensity
        horizon = to_linear(env.horizon_color) * intensity
        ground = to_linear(env.ground_color) * intensity
    elif kind == 1:
        zenith = horizon = ground = to_linear(env.color) * intensity
    else:
        zenith = horizon = ground = np.asarray(background, dtype=np.float64)
    return AtmosphereModel(
        sun=sun,
        env_kind=kind,
        zenith=np.asarray(zenith, dtype=np.float64),
        horizon=np.asarray(horizon, dtype=np.float64),
        ground=np.asarray(ground, dtype=np.float64),
        background=np.asarray(background, dtype=np.float64),
        glow=to_linear(atm.sun_glow_color) * atm.sun_glow,
        glow_size=float(atm.sun_glow_size),
        horizon_glow=float(atm.horizon_glow),
        sky_background=bool(atm.sky),
        fog_density=float(atm.fog_density),
        fog_falloff=float(atm.fog_falloff),
        fog_rel=float(eye[2]) - float(atm.fog_height),
        fog_color=to_linear(atm.fog_color),
        fog_sun=to_linear(atm.sun_glow_color) * atm.fog_sun,
        fog_power=float(atm.fog_sun_power),
        hemisphere=atm.ambient == "hemisphere",
        ground_ambient=to_linear(atm.ground_ambient) * float(ambient_intensity),
    )


def sky(model: AtmosphereModel, d: np.ndarray) -> np.ndarray:
    """``(N, 3)`` sky radiance in directions ``d`` (unit, ``(N, 3)``)."""
    z = d[:, 2]
    if model.env_kind == 2:
        up = np.sqrt(np.clip(z, 0.0, 1.0))[:, None]
        down = np.minimum(1.0, 4.0 * np.abs(z))[:, None]
        above = model.horizon + up * (model.zenith - model.horizon)
        below = model.horizon + down * (model.ground - model.horizon)
        base = np.where((z >= 0.0)[:, None], above, below)
    else:
        base = np.broadcast_to(model.zenith, d.shape).copy()
    return base + glow(model, d)


def glow(model: AtmosphereModel, d: np.ndarray) -> np.ndarray:
    """``(N, 3)`` the sun glow and horizon glow added to the sky in directions ``d``."""
    base = np.zeros(d.shape)
    if model.sun is not None:
        g = np.maximum(d @ model.sun, 0.0)
        halo = np.power(g, model.glow_size) + 0.15 * np.power(g, max(model.glow_size / 8.0, 1.0))
        base = base + halo[:, None] * model.glow
        if model.horizon_glow > 0.0:
            sh = model.sun[:2]
            sl = float(np.hypot(sh[0], sh[1]))
            dh = d[:, :2]
            dl = np.hypot(dh[:, 0], dh[:, 1])
            a = np.where((dl > 1e-9) & (sl > 1e-9), (dh @ sh) / np.maximum(dl * sl, 1e-30), 0.0)
            a = np.maximum(a, 0.0)
            t = 1.0 - np.minimum(2.5 * np.abs(d[:, 2]), 1.0)
            band = t * t * t * t * (0.25 + 0.75 * a * a)
            unit = model.glow / max(float(np.max(model.glow)), 1e-9)
            base = base + (model.horizon_glow * band)[:, None] * unit
    return base


def fog(model: AtmosphereModel, color: np.ndarray, d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``color`` (``(N, 3)``) seen through the fog along directions ``d`` over distances
    ``t`` (``inf``: the sky)."""
    if not model.fog:
        return color
    trans = transmittance(model, d, t)
    f = np.broadcast_to(model.fog_color, color.shape)
    if model.sun is not None and float(np.max(model.fog_sun)) > 0.0:
        g = np.maximum(d @ model.sun, 0.0)
        f = f + np.power(g, model.fog_power)[:, None] * model.fog_sun
    return color * trans[:, None] + f * (1.0 - trans)[:, None]


def transmittance(model: AtmosphereModel, d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``(N,)`` the fraction of light crossing the fog along directions ``d`` over ``t``."""
    if not model.fog:
        return np.ones(d.shape[0])
    b = model.fog_falloff
    rho = model.fog_density * (np.exp(-b * model.fog_rel) if b > 0.0 else 1.0)
    dz = d[:, 2]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        finite = np.isfinite(t)
        tt = np.where(finite, t, 0.0)
        if b > 0.0:
            x = b * dz * tt
            small = np.abs(x) < 1e-4
            exact = rho * (1.0 - np.exp(-x)) / np.where(small, 1.0, b * dz)
            tau = np.where(small, rho * tt * (1.0 - 0.5 * x), exact)
            sky_tau = np.where(dz > 1e-4, rho / (b * np.maximum(dz, 1e-4)), np.inf)
        else:
            tau = rho * tt
            sky_tau = np.full(dz.shape, np.inf)
        tau = np.where(finite, tau, sky_tau)
        return np.exp(-np.minimum(tau, 700.0))


def ambient_at(model: AtmosphereModel | None, ambient: np.ndarray, nz: np.ndarray) -> Any:
    """Per-sample ambient light ``(N, 3)`` for the hemisphere mode, else ``None``."""
    if model is None or not model.hemisphere:
        return None
    w = (0.5 + 0.5 * np.clip(nz, -1.0, 1.0))[:, None]
    return model.ground_ambient + w * (np.asarray(ambient) - model.ground_ambient)
