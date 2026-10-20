"""Display transforms: exposure, an optional tone mapper, sRGB encoding (all backends).

Every renderer produces scene-referred linear light. The output stage then:

1. multiplies by ``2 ** exposure``;
2. applies the tone mapper (``tonemap``):

   * ``"none"`` (default) -- nothing: values are clipped to ``[0, 1]`` by the
     encoding, exactly as Aether3D always did (legacy renders are unchanged);
   * ``"neutral"`` -- the Khronos PBR Neutral tone mapper (Khronos Group,
     2024; the formula of the published specification). It first subtracts
     an offset from all three channels: 0.04 wherever the smallest channel
     is at least 0.08 (the 4% Fresnel reflection a PBR surface adds under
     white light, so base colors display as authored), less below that
     (``x - 6.25 x^2`` for the smallest channel ``x``, keeping black black).
     Colors whose largest channel then stays below 0.76 are otherwise kept;
     brighter ones are compressed smoothly toward 1 and slightly
     desaturated, so highlights, light sources and bright skies keep their
     shape instead of clipping;
   * ``"film"`` (0.20.0) -- a filmic view transform with a path to white: the colour is
     drawn toward grey by :data:`FILM_INSET` (rows summing to 1: grey stays grey), each
     channel goes through a tone curve of its log2 exposure around middle grey 0.18 --
     a monotone cubic (Fritsch and Carlson) through :data:`FILM_DISPLAY` at
     :data:`FILM_EV` -- and the display-linear result through :data:`FILM_OUTSET`, then
     the sRGB encoding. Middle grey shows at 0.4613 (sRGB code 118), +6.5 stops reaches
     white, -10 stops black; saturated highlights go to white without hue skews. Its
     numbers were fitted to Blender 5.2's AgX view transform, measured on grey and seven
     colour ramps from -8 to +10 stops (research/2026-10-stage19/film_fit/fit.py: RMS
     display error 0.008, 0.038 at most): Aether3D's own transform, not Blender's;

3. encodes with the sRGB transfer function (``"film"``: its own last step) and quantizes
   to 8 bits (16 with ``bits=16``, for masters).

``RenderResult.linear`` holds the values after step 1; HDR files
(``.hdr``, ``.pfm``) store them too, before tone mapping.
"""

from __future__ import annotations

import numpy as np

from ..errors import RenderError
from .color import linear_to_srgb, quantize

TONEMAPS = ("none", "neutral", "film")
NEUTRAL_START = 0.8 - 0.04
NEUTRAL_DESATURATION = 0.15


def check_tonemap(value) -> str:
    if value not in TONEMAPS:
        raise RenderError(
            "INVALID_SETTINGS",
            f"tonemap must be one of {', '.join(TONEMAPS)}, got {value!r}",
            "tonemap",
        )
    return value


def neutral(color: np.ndarray) -> np.ndarray:
    """Khronos PBR Neutral tone mapping of linear RGB (``(..., 3)``)."""
    c = np.asarray(color, dtype=np.float64)
    x = c.min(axis=-1)
    offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    c = c - offset[..., None]
    peak = c.max(axis=-1)
    d = 1.0 - NEUTRAL_START
    compress = peak >= NEUTRAL_START
    safe_peak = np.where(compress, peak, 1.0)
    new_peak = 1.0 - d * d / (safe_peak + d - NEUTRAL_START)
    scaled = c * (new_peak / safe_peak)[..., None]
    g = 1.0 - 1.0 / (NEUTRAL_DESATURATION * (safe_peak - new_peak) + 1.0)
    mixed = scaled * (1.0 - g)[..., None] + (new_peak * g)[..., None]
    return np.where(compress[..., None], mixed, c)


FILM_EV = np.arange(-12.0, 12.25, 0.5)
"""Exposures (stops around middle grey 0.18) of the film curve's points."""
FILM_DISPLAY = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 4.6e-05, 0.000244, 0.000717, 0.001648, 0.003265, 0.005889,
    0.009992, 0.016277, 0.025827, 0.040218, 0.05884, 0.080506, 0.105884, 0.135743,
    0.170979, 0.21262, 0.262031, 0.320422, 0.387506, 0.461323, 0.53591, 0.605484,
    0.668328, 0.724233, 0.773507, 0.816195, 0.853583, 0.886341, 0.915084, 0.940378,
    0.962684, 0.982449, 0.999443, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
])  # fmt: skip
"""Display (sRGB-encoded) values of grey at :data:`FILM_EV`."""
FILM_INSET = np.array([
    [0.65247195, 0.26081387, 0.08671418],
    [0.09555971, 0.80626328, 0.09817701],
    [0.08496163, 0.17664461, 0.73839376],
])  # fmt: skip
FILM_OUTSET = np.array([
    [1.4924381, -0.32973379, -0.16270432],
    [-0.11961517, 1.11057682, 0.00903835],
    [-0.15369894, -0.2418455, 1.39554444],
])  # fmt: skip


def _fritsch_carlson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Slopes of the monotone cubic Hermite interpolant through ``(x, y)``."""
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.concatenate([[delta[0]], 0.5 * (delta[:-1] + delta[1:]), [delta[-1]]])
    for k in range(len(delta)):
        if delta[k] == 0.0:
            m[k] = m[k + 1] = 0.0
            continue
        a, b = m[k] / delta[k], m[k + 1] / delta[k]
        t = a * a + b * b
        if t > 9.0:
            tau = 3.0 / np.sqrt(t)
            m[k], m[k + 1] = tau * a * delta[k], tau * b * delta[k]
    return m


_FILM_SLOPES = _fritsch_carlson(FILM_EV, FILM_DISPLAY)


def film_curve(ev: np.ndarray) -> np.ndarray:
    """The film tone curve: display values of grey ``ev`` stops from middle grey."""
    x = np.clip(ev, FILM_EV[0], FILM_EV[-1])
    k = np.clip(np.searchsorted(FILM_EV, x, side="right") - 1, 0, len(FILM_EV) - 2)
    h = FILM_EV[k + 1] - FILM_EV[k]
    t = (x - FILM_EV[k]) / h
    t2, t3 = t * t, t * t * t
    y0, y1 = FILM_DISPLAY[k], FILM_DISPLAY[k + 1]
    m0, m1 = _FILM_SLOPES[k] * h, _FILM_SLOPES[k + 1] * h
    y = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * y1
    y = y + (t3 - t2) * m1
    return np.clip(y, 0.0, 1.0)


def film(color: np.ndarray) -> np.ndarray:
    """The film view transform: display (sRGB-encoded) values of linear RGB."""
    from .color import srgb_to_linear

    c = np.asarray(color, dtype=np.float64)
    x = _mat3(FILM_INSET, c)
    ev = np.log2(np.maximum(x, 1e-12) / 0.18)
    y = _mat3(FILM_OUTSET, srgb_to_linear(film_curve(ev)))
    return linear_to_srgb(np.clip(y, 0.0, 1.0))


def _mat3(m: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``m @ c`` per pixel, as explicit ordered sums (no BLAS: the same bits anywhere)."""
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return np.stack([m[i, 0] * r + m[i, 1] * g + m[i, 2] * b for i in range(3)], axis=-1)


def display(linear: np.ndarray, tonemap: str = "none", bits: int = 8) -> np.ndarray:
    """8-bit (or 16-bit) sRGB codes of exposed linear light (in row blocks: bounded
    temporaries)."""
    out = np.empty(linear.shape, dtype=np.uint8 if bits == 8 else np.uint16)
    for r0 in range(0, linear.shape[0], 128):
        block = linear[r0 : r0 + 128]
        if tonemap == "film":
            codes = film(block)
        else:
            if tonemap == "neutral":
                block = neutral(block)
            codes = linear_to_srgb(block)
        if bits == 8:
            out[r0 : r0 + 128] = quantize(codes)
        else:
            out[r0 : r0 + 128] = np.rint(np.clip(codes, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return out


def color_info(exposure: float, tonemap: str) -> dict:
    """How a render's linear light became its PNG (for ``RenderResult.color``)."""
    return {
        "working_space": "linear sRGB (Rec. 709 primaries, D65)",
        "exposure": exposure,
        "exposure_scale": 2.0**exposure,
        "tonemap": tonemap,
        "output": "sRGB transfer function, 8 bits per channel (values clipped to [0, 1])"
        if tonemap != "film"
        else "the film view transform (a tone curve with a path to white), sRGB, 8 bits",
        "hdr_output": "linear values after exposure (.hdr Radiance RGBE, .pfm float32)",
    }
