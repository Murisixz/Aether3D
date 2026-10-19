"""A denoiser for path-traced images (0.20.0): ``render(..., denoise=True)``.

An edge-avoiding a-trous wavelet filter (Dammertz et al. 2010, "Edge-Avoiding A-Trous
Wavelet Transform for fast Global Illumination Filtering") with the luminance
edge-stopping of SVGF scaled by each pixel's estimated variance (Schied et al. 2017,
"Spatiotemporal Variance-Guided Filtering"; the spatial part only, one frame):

1. **Demodulate**: the beauty divided by the albedo pass (at least :data:`MIN_ALBEDO`), so
   textures are not blurred -- only the light falling on them is filtered.
2. **Filter** :data:`ITERATIONS` times with a 5 x 5 B3-spline kernel whose taps spread by
   ``2^i``; each tap ``q`` of pixel ``p`` is weighted by ``max(0, n_p . n_q)^sigma_normal``
   (normals), ``exp(-|z_p - z_q| / (sigma_depth |grad z . (p - q)| + eps))`` (view depth;
   the background only mixes with itself) and ``exp(-|l_p - l_q| / (sigma_luminance
   sqrt(var_p) + eps))`` (luminance, with ``var_p`` the variance of ``p``'s estimate,
   blurred over 3 x 3, carried through the iterations as the filtered estimate's).
3. **Remodulate** by the albedo.

Measured against converged renders (``tests/test_denoise.py``, two scenes, 2 to 64
samples): 20 to 35 % less display-space error than the raw image -- what 1.5 to 2 times
the samples would give. It does not replace samples: stronger settings (wider
luminance tolerance, more iterations) blur more than they remove.

It is a *biased* estimator: it trades noise for blur where the features cannot tell
detail from noise (fine shadows, caustics, glossy reflections of small things). The raw
image stays available (``result.passes["beauty"]``); the result's ``linear`` and
``image`` are the denoised ones and the report says so (``backend["denoiser"]`` and a
``DENOISED`` diagnostic). Deterministic: the same passes give the same bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import RenderError

ITERATIONS = 3
MIN_ALBEDO = 0.01
KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class DenoiseSettings:
    """``iterations`` (1-8), ``sigma_normal`` (exponent), ``sigma_depth``,
    ``sigma_luminance`` (in standard deviations of the noise)."""

    iterations: int = ITERATIONS
    sigma_normal: float = 128.0
    sigma_depth: float = 1.0
    sigma_luminance: float = 2.0

    @classmethod
    def coerce(cls, value: Any) -> DenoiseSettings | None:
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise RenderError("INVALID_DENOISE", "denoise is true, false or an object", "denoise")
        unknown = sorted(set(value) - {"iterations", "sigma_normal", "sigma_depth",
                                       "sigma_luminance"})  # fmt: skip
        if unknown:
            raise RenderError(
                "INVALID_DENOISE", f"unknown denoise setting(s): {', '.join(unknown)}", "denoise"
            )
        it = value.get("iterations", ITERATIONS)
        if isinstance(it, bool) or not isinstance(it, int) or not 1 <= it <= 8:
            raise RenderError("INVALID_DENOISE", "iterations is 1..8", "denoise.iterations")
        out = {"iterations": it}
        for key, lo, hi in (("sigma_normal", 0.0, 1e4), ("sigma_depth", 1e-6, 1e6),
                            ("sigma_luminance", 1e-6, 1e6)):  # fmt: skip
            v = value.get(key, getattr(cls, key))
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise RenderError(
                    "INVALID_DENOISE", f"{key} is a number in [{lo:g}, {hi:g}]", f"denoise.{key}"
                )
            out[key] = float(v)
        return cls(**out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "a-trous (edge-avoiding, variance-guided)",
            "iterations": self.iterations,
            "sigma_normal": self.sigma_normal,
            "sigma_depth": self.sigma_depth,
            "sigma_luminance": self.sigma_luminance,
            "biased": True,
        }


def _pad(a: np.ndarray, r: int) -> np.ndarray:
    widths = [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, widths, mode="edge")


def _blur3(v: np.ndarray) -> np.ndarray:
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    p = _pad(v, 1)
    h, w = v.shape
    rows = sum(k[i] * p[i : i + h, :] for i in range(3))
    return sum(k[j] * rows[:, j : j + w] for j in range(3))


def denoise(passes: dict[str, np.ndarray], settings: DenoiseSettings | None = None) -> np.ndarray:
    """``(H, W, 3) float32``: the denoised beauty of a path tracer's ``passes``."""
    s = settings or DenoiseSettings()
    beauty = np.asarray(passes["beauty"], np.float32)
    h, w = beauty.shape[:2]
    depth = np.asarray(passes["depth"], np.float32)
    background = ~np.isfinite(depth)
    albedo = np.maximum(np.asarray(passes["albedo"], np.float32), MIN_ALBEDO)
    albedo = np.where(background[..., None], 1.0, albedo).astype(np.float32)  # (the sky)
    normal = np.asarray(passes["normal"], np.float32)
    var_rgb = np.asarray(passes["variance"], np.float32)
    var_rgb = np.where(np.isfinite(var_rgb), var_rgb, 0.0)
    color = beauty / albedo
    lum_alb = np.maximum(albedo @ LUMINANCE, MIN_ALBEDO)
    var = (var_rgb @ (LUMINANCE * LUMINANCE)) / (lum_alb * lum_alb)  # of the demodulated luminance
    z = np.where(background, 0.0, depth).astype(np.float32)
    gz = np.zeros_like(z)  # |dz/dx| + |dz/dy| (central differences, pixels of geometry)
    gz[:, 1:-1] += 0.5 * np.abs(z[:, 2:] - z[:, :-2])
    gz[1:-1, :] += 0.5 * np.abs(z[2:, :] - z[:-2, :])
    gz = np.where(background, 0.0, gz)
    eps = np.float32(1e-6)
    for i in range(s.iterations):
        step = 2**i
        r = 2 * step
        cp, np_, zp, bp, vp = (_pad(a, r) for a in (color, normal, z, background, var))
        lum = color @ LUMINANCE
        lp = _pad(lum, r)
        sigma = s.sigma_luminance * np.sqrt(np.maximum(_blur3(var), 0.0)) + eps
        acc = np.zeros_like(color)
        wsum = np.zeros((h, w), np.float32)
        vacc = np.zeros((h, w), np.float32)
        for a in range(5):
            dy = (a - 2) * step
            for b in range(5):
                dx = (b - 2) * step
                sl = (slice(r + dy, r + dy + h), slice(r + dx, r + dx + w))
                kq = KERNEL[a] * KERNEL[b]
                if dy == 0 and dx == 0:
                    wq = np.full((h, w), kq, np.float32)
                else:
                    nd = np.maximum((normal * np_[sl]).sum(axis=2), 0.0)
                    wn = nd**s.sigma_normal if s.sigma_normal > 0 else 1.0
                    dist = gz * (abs(dx) + abs(dy)) * s.sigma_depth + eps
                    wz = np.exp(-np.abs(z - zp[sl]) / dist)
                    wz = np.where(background == bp[sl], np.where(background, 1.0, wz), 0.0)
                    wl = np.exp(-np.abs(lum - lp[sl]) / sigma)
                    wq = (kq * wn * wz * wl).astype(np.float32)
                acc += wq[..., None] * cp[sl]
                wsum += wq
                vacc += wq * wq * vp[sl]
        color = acc / np.maximum(wsum, eps)[..., None]
        var = vacc / np.maximum(wsum * wsum, eps)
    return (color * albedo).astype(np.float32)
