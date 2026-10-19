"""Colours of black bodies: a light's ``temperature`` as linear Rec. 709 RGB.

Planck's law ``B(lambda, T) = 2 h c^2 / lambda^5 / (exp(h c / (lambda k T)) - 1)`` is
integrated against the CIE 1931 2-degree colour-matching functions (Wyman, Sloan and
Shirley's multi-lobe fit, "Simple Analytic Approximations to the CIE XYZ Color Matching
Functions", JCGT 2(2), 2013) from 360 to 830 nm in 1 nm steps, converted from XYZ to linear
Rec. 709 (sRGB primaries, D65 white; no chromatic adaptation), channels outside the gamut
clipped to 0, and scaled to a luminance of 1 -- so a temperature changes a light's hue,
never its brightness. Temperatures are clamped to 800..12,000 K.

Measured against Blender 5.2's light temperatures and Blackbody node (which also clamp to
800..12,000 K; research/2026-10-stage19/blender/e3_lights, tests/test_lights.py): the
same hue within 1.1 % per channel from 2,700 to 12,000 K (0.5 % from 4,500 K), 2 % at
2,000 K, 2.6 % at 1,850 K and 6.2 % at 1,000 to 1,500 K, where the fit of the matching
functions is least accurate (the far red). 6,500 K is not white (D65 is not a black
body): ``(1.042, 0.984, 1.035)``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

MIN_TEMPERATURE = 800.0
MAX_TEMPERATURE = 12_000.0

_XYZ_TO_REC709 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def _lobe(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _cmf(nm: np.ndarray) -> np.ndarray:
    """``(3, n)``: x, y, z colour matching functions at wavelengths ``nm``."""
    x = (
        1.056 * _lobe(nm, 599.8, 37.9, 31.0)
        + 0.362 * _lobe(nm, 442.0, 16.0, 26.7)
        - 0.065 * _lobe(nm, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _lobe(nm, 568.8, 46.9, 40.5) + 0.286 * _lobe(nm, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(nm, 437.0, 11.8, 36.0) + 0.681 * _lobe(nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z])


@lru_cache(maxsize=4096)
def blackbody(temperature: float) -> tuple[float, float, float]:
    """Linear Rec. 709 RGB of a black body at ``temperature`` kelvins, luminance 1."""
    t = min(max(float(temperature), MIN_TEMPERATURE), MAX_TEMPERATURE)
    nm = np.arange(360.0, 831.0, 1.0)
    m = nm * 1e-9
    c1, c2 = 3.741771852e-16, 1.438776877e-2  # 2 pi h c^2 (W m^2), h c / k (m K)
    radiance = c1 / (m**5 * np.expm1(c2 / (m * t)))
    xyz = _cmf(nm) @ radiance
    rgb = np.maximum(_XYZ_TO_REC709 @ xyz, 0.0)
    lum = float(_LUMINANCE @ rgb)
    if not lum > 0.0 or not math.isfinite(lum):  # pragma: no cover - the range keeps it > 0
        return (1.0, 1.0, 1.0)
    out = rgb / lum
    return (float(out[0]), float(out[1]), float(out[2]))
