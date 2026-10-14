"""The path tracer's BSDF: Stage 7's metallic-roughness BRDF, evaluated and sampled.

The reflectance is exactly the one the rasterizers shade with (see
:mod:`aether3d.rendering.surface` and ``docs/MATERIALS.md``), divided by pi
to make it a BRDF (the rasterizers' light intensities carry that factor of
pi; see ``docs/RENDER_QUALITY.md``):

    f(l, v) = (1 - F) c_diff / pi + F D V

with ``c_diff = base (1 - metallic)``, ``F0 = 0.08 specular (1 - metallic) +
base metallic``, Schlick's ``F(v.h)``, the GGX distribution ``D`` and the
height-correlated Smith visibility ``V = G2 / (4 n.l n.v)`` with
``alpha = roughness^2`` (roughness at least :data:`MIN_ROUGHNESS`, as on the
rasterizers). ``"simple"`` materials are Lambertian (``c_diff = base``) when
their ``specular`` is 0; otherwise they get a dielectric GGX lobe with
``F0 = 0.08 specular`` and the roughness that matches their Blinn-Phong
exponent (``alpha^2 = 2 / (shininess + 2)``, as the glTF exporter writes
them).

**Sampling** picks the specular lobe with probability ``p_s`` (the
luminance of ``F(n.v)`` against the diffuse albedo, kept in ``[0.1, 0.9]``
when both lobes exist) and otherwise the diffuse lobe:

* diffuse: cosine-weighted hemisphere, ``pdf_d = n.l / pi``;
* specular: GGX visible normals (Heitz 2018) reflected about ``h``,
  ``pdf_s = G1(v) D(h) / (4 n.v)``.

Every sample is weighted by ``f n.l / pdf`` with the mixture density
``pdf = p_s pdf_s + (1 - p_s) pdf_d`` (one-sample MIS over the lobes), so
the estimator is unbiased whichever lobe produced the direction. Light
sampling combines with the same ``pdf`` through the power heuristic.

Transmission (glass, refraction) is not modelled: every material reflects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..surface import LUMINANCE, MIN_ROUGHNESS
from .sampling import cosine_hemisphere, frame, ggx_visible_normal, to_local, to_world

MIN_COS = 1e-6
"""Directions closer than this to the tangent plane carry no light."""
LOBE_RANGE = (0.1, 0.9)
"""Bounds of the specular lobe's selection probability when both lobes exist."""


def roughness_from_shininess(shininess: np.ndarray) -> np.ndarray:
    """GGX perceptual roughness matching a Blinn-Phong exponent (``alpha^2 = 2/(n+2)``)."""
    return (2.0 / (np.asarray(shininess, dtype=np.float64) + 2.0)) ** 0.25


@dataclass
class BSDF:
    """The BSDF at ``m`` surface points, in a local frame around the shading normal."""

    t: np.ndarray  # (m, 3) tangent
    b: np.ndarray  # (m, 3) bitangent
    n: np.ndarray  # (m, 3) shading normal, on the side of the outgoing direction
    wo: np.ndarray  # (m, 3) local outgoing (view) direction, z > 0
    c_diff: np.ndarray  # (m, 3)
    f0: np.ndarray  # (m, 3)
    alpha: np.ndarray  # (m,) GGX alpha = roughness^2
    specular: np.ndarray  # (m,) bool: has a specular lobe
    p_spec: np.ndarray  # (m,) probability of sampling the specular lobe

    @classmethod
    def build(
        cls,
        n: np.ndarray,
        wo_world: np.ndarray,
        base: np.ndarray,
        metallic: np.ndarray,
        roughness: np.ndarray,
        specular: np.ndarray,
        has_specular: np.ndarray,
    ) -> BSDF:
        """``n`` must face ``wo_world`` (``n . wo > 0``); see the module docstring."""
        metallic = np.clip(metallic, 0.0, 1.0)[:, None]
        rough = np.clip(roughness, MIN_ROUGHNESS, 1.0)
        c_diff = base * (1.0 - metallic)
        f0 = 0.08 * specular[:, None] * (1.0 - metallic) + base * metallic
        f0 = np.where(has_specular[:, None], f0, 0.0)
        t, b = frame(n)
        wo = to_local(wo_world, t, b, n)
        wo[:, 2] = np.maximum(wo[:, 2], MIN_COS)
        ndv = wo[:, 2]
        f_view = f0 + (1.0 - f0) * ((1.0 - ndv) ** 5)[:, None]
        w_spec = np.where(has_specular, f_view @ LUMINANCE, 0.0)
        w_diff = (c_diff @ LUMINANCE) * (1.0 - w_spec)
        total = w_spec + w_diff
        p = np.where(total > 0.0, w_spec / np.where(total > 0.0, total, 1.0), 0.0)
        both = has_specular & (w_diff > 0.0)
        p = np.where(both, np.clip(p, *LOBE_RANGE), np.where(has_specular, 1.0, 0.0))
        # A specular-only surface with no diffuse albedo samples only the lobe it has.
        p = np.where(has_specular & (w_diff <= 0.0), 1.0, p)
        return cls(t, b, n, wo, c_diff, f0, rough * rough, has_specular.copy(), p)

    # -- evaluation ---------------------------------------------------------------------

    def _eval_local(self, wi: np.ndarray, rows=slice(None)):
        """``(f, pdf)`` for local incoming directions ``wi`` at ``rows``."""
        wo = self.wo[rows]
        ndl = wi[:, 2]
        ndv = wo[:, 2]
        ok = ndl > MIN_COS
        ndl_c = np.maximum(ndl, MIN_COS)
        c_diff = self.c_diff[rows]
        alpha = self.alpha[rows]
        a2 = alpha * alpha
        spec = self.specular[rows]
        h = wi + wo
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
        ndh = np.maximum(h[:, 2], 0.0)
        vdh = np.clip((wo * h).sum(axis=1), 0.0, 1.0)
        f0 = self.f0[rows]
        fresnel = f0 + (1.0 - f0) * ((1.0 - vdh) ** 5)[:, None]
        fresnel = np.where(spec[:, None], fresnel, 0.0)
        d = ndh * ndh * (a2 - 1.0) + 1.0
        dist = a2 / (math.pi * d * d)
        vis = 0.5 / np.maximum(
            ndl_c * np.sqrt(ndv * ndv * (1.0 - a2) + a2)
            + ndv * np.sqrt(ndl_c * ndl_c * (1.0 - a2) + a2),
            1e-300,
        )
        f = (1.0 - fresnel) * c_diff / math.pi + fresnel * (dist * vis)[:, None]
        g1 = 2.0 * ndv / (ndv + np.sqrt(a2 + (1.0 - a2) * ndv * ndv))
        pdf_spec = g1 * dist / (4.0 * ndv)
        p = self.p_spec[rows]
        pdf = p * pdf_spec + (1.0 - p) * ndl_c / math.pi
        f = np.where(ok[:, None], f, 0.0)
        pdf = np.where(ok, pdf, 0.0)
        return f, pdf

    def eval(self, wi_world: np.ndarray, rows=slice(None)):
        """``(f, pdf, cos)`` for world incoming directions (unit) at ``rows``."""
        wi = to_local(wi_world, self.t[rows], self.b[rows], self.n[rows])
        f, pdf = self._eval_local(wi, rows)
        return f, pdf, np.maximum(wi[:, 2], 0.0)

    # -- sampling ----------------------------------------------------------------------

    def sample(self, u_lobe: np.ndarray, u1: np.ndarray, u2: np.ndarray):
        """Sample incoming directions.

        Returns ``(wi_world, weight = f cos / pdf, pdf, valid, specular)`` where
        ``specular`` tells which lobe produced each direction.
        """
        spec = u_lobe < self.p_spec
        wi = cosine_hemisphere(u1, u2)
        if spec.any():
            h = ggx_visible_normal(self.wo[spec], self.alpha[spec], u1[spec], u2[spec])
            wo = self.wo[spec]
            wi[spec] = 2.0 * (wo * h).sum(axis=1, keepdims=True) * h - wo
        f, pdf = self._eval_local(wi)
        valid = (wi[:, 2] > MIN_COS) & (pdf > 0.0)
        weight = np.where(valid[:, None], f * (wi[:, 2] / np.where(valid, pdf, 1.0))[:, None], 0.0)
        return to_world(wi, self.t, self.b, self.n), weight, pdf, valid, spec

    def albedo(self) -> np.ndarray:
        """Directional albedo estimate used for the albedo pass: ``c_diff + F(n.v)``."""
        ndv = self.wo[:, 2]
        f_view = self.f0 + (1.0 - self.f0) * ((1.0 - ndv) ** 5)[:, None]
        return self.c_diff * (1.0 - f_view) + f_view
