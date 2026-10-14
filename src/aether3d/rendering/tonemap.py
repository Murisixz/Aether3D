"""Display transforms: exposure, an optional tone mapper, sRGB encoding (all backends).

Every renderer produces scene-referred linear light. The output stage then:

1. multiplies by ``2 ** exposure``;
2. applies the tone mapper (``tonemap``):

   * ``"none"`` (default) -- nothing: values are clipped to ``[0, 1]`` by the
     encoding, exactly as Aether3D always did (legacy renders are unchanged);
   * ``"neutral"`` -- the Khronos PBR Neutral tone mapper (Khronos Group,
     2024; the formula of the published specification): colors whose
     channels stay below about 0.76 are kept (apart from a tiny toe that
     lifts nothing and darkens near-black by at most 0.04), brighter ones
     are compressed smoothly toward 1 and slightly desaturated, so
     highlights, light sources and bright skies keep their shape instead of
     clipping;

3. encodes with the sRGB transfer function and quantizes to 8 bits.

``RenderResult.linear`` holds the values after step 1; HDR files
(``.hdr``, ``.pfm``) store them too, before tone mapping.
"""

from __future__ import annotations

import numpy as np

from ..errors import RenderError
from .color import linear_to_srgb, quantize

TONEMAPS = ("none", "neutral")
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


def display(linear: np.ndarray, tonemap: str = "none") -> np.ndarray:
    """8-bit sRGB codes of exposed linear light (in row blocks: bounded temporaries)."""
    out = np.empty(linear.shape, dtype=np.uint8)
    for r0 in range(0, linear.shape[0], 128):
        block = linear[r0 : r0 + 128]
        if tonemap == "neutral":
            block = neutral(block)
        out[r0 : r0 + 128] = quantize(linear_to_srgb(block))
    return out


def color_info(exposure: float, tonemap: str) -> dict:
    """How a render's linear light became its PNG (for ``RenderResult.color``)."""
    return {
        "working_space": "linear sRGB (Rec. 709 primaries, D65)",
        "exposure": exposure,
        "exposure_scale": 2.0**exposure,
        "tonemap": tonemap,
        "output": "sRGB transfer function, 8 bits per channel (values clipped to [0, 1])",
        "hdr_output": "linear values after exposure (.hdr Radiance RGBE, .pfm float32)",
    }
