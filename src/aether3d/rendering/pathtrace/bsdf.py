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

**Transmission** (0.20.0, ``transmission`` x ``(1 - metallic)`` of a standard material,
``w_g``): the BSDF is ``(1 - w_g) f + w_g f_glass``, with ``f_glass`` a rough dielectric
interface of relative index ``eta`` (``ior`` entering the surface's outward side,
``1 / ior`` leaving it): GGX reflection ``F D G2 / (4 |n.l| n.v)`` and refraction
``base (1 - F) D G2 |l.h| |v.h| / (|n.l| n.v (l.h + v.h / eta)^2 eta^2)`` (Walter et al.
2007; radiance), with the exact dielectric Fresnel ``F(v.h, eta)`` (1 beyond the critical
angle: total internal reflection). The refracted light is tinted by the base colour at
every crossing, as Blender's Principled BSDF does; volumes do not absorb. The glass lobe
is picked with probability ``w_g``, then reflection with probability ``F`` of a visible
normal; densities are the lobes' mixture, as above.
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


def fresnel_dielectric(cos_i: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Unpolarized Fresnel reflectance of light arriving at ``cos_i >= 0`` on an interface
    of relative index ``eta`` (transmitted side over incident side); 1 for total internal
    reflection."""
    c = np.clip(cos_i, 0.0, 1.0)
    sin2_t = (1.0 - c * c) / (eta * eta)
    tir = sin2_t >= 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
    r_par = (eta * c - cos_t) / np.maximum(eta * c + cos_t, 1e-300)
    r_perp = (c - eta * cos_t) / np.maximum(c + eta * cos_t, 1e-300)
    return np.where(tir, 1.0, 0.5 * (r_par * r_par + r_perp * r_perp))


def _lambda(w: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Smith's Lambda for GGX at local directions ``w``."""
    c2 = np.maximum(w[:, 2] * w[:, 2], 1e-300)
    tan2 = np.maximum(1.0 - c2, 0.0) / c2
    return 0.5 * (np.sqrt(1.0 + a2 * tan2) - 1.0)


def glass_eval(wo, wi, alpha, eta, tint):
    """``(f, pdf)`` of the rough dielectric lobe (local frame, ``wo.z > 0``) for incoming
    ``wi`` on either side; see the module docstring."""
    a2 = alpha * alpha
    cos_o = wo[:, 2]
    cos_i = wi[:, 2]
    reflect = cos_i > 0.0
    etap = np.where(reflect, 1.0, eta)
    wm = wi * etap[:, None] + wo
    len2 = (wm * wm).sum(axis=1)
    ok = (np.abs(cos_i) > MIN_COS) & (len2 > 1e-300)
    wm = wm / np.sqrt(np.maximum(len2, 1e-300))[:, None]
    wm = np.where((wm[:, 2] < 0.0)[:, None], -wm, wm)
    dot_i = (wm * wi).sum(axis=1)
    dot_o = (wm * wo).sum(axis=1)
    ok &= (dot_i * cos_i > 0.0) & (dot_o * cos_o > 0.0)  # (no back-facing microfacets)
    f_r = fresnel_dielectric(dot_o, eta)
    nh = wm[:, 2]
    d = nh * nh * (a2 - 1.0) + 1.0
    dist = a2 / (math.pi * d * d)
    g2 = 1.0 / (1.0 + _lambda(wo, a2) + _lambda(wi, a2))
    g1 = 1.0 / (1.0 + _lambda(wo, a2))
    visible = g1 * np.maximum(dot_o, 0.0) * dist / np.maximum(cos_o, 1e-300)
    abs_i = np.maximum(np.abs(cos_i), 1e-300)
    denom = (dot_i + dot_o / etap) ** 2
    denom = np.maximum(denom, 1e-300)
    f_refl = dist * g2 * f_r / (4.0 * abs_i * cos_o)
    f_tran = dist * (1.0 - f_r) * g2 * np.abs(dot_i * dot_o) / (abs_i * cos_o * denom)
    f_tran = f_tran / (etap * etap)
    pdf_refl = visible / (4.0 * np.maximum(np.abs(dot_o), 1e-300)) * f_r
    pdf_tran = visible * np.abs(dot_i) / denom * (1.0 - f_r)
    f = np.where(reflect[:, None], f_refl[:, None], f_tran[:, None] * tint)
    pdf = np.where(reflect, pdf_refl, pdf_tran)
    f = np.where(ok[:, None], f, 0.0)
    return f, np.where(ok, pdf, 0.0)


def glass_sample(wo, alpha, eta, u_choice, u1, u2):
    """Local incoming directions of the rough dielectric lobe: a visible GGX normal, then
    reflection with probability ``F`` (``u_choice``), else refraction; and whether each
    is valid (a reflection above the surface, a refraction below it)."""
    wm = ggx_visible_normal(wo, alpha, u1, u2)
    cos_m = (wo * wm).sum(axis=1)
    f_r = fresnel_dielectric(cos_m, eta)
    reflected = 2.0 * cos_m[:, None] * wm - wo
    sin2_t = (1.0 - cos_m * cos_m) / (eta * eta)
    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
    refracted = -wo / eta[:, None] + (cos_m / eta - cos_t)[:, None] * wm
    take_reflect = u_choice < f_r
    wi = np.where(take_reflect[:, None], reflected, refracted)
    valid = np.where(take_reflect, wi[:, 2] > MIN_COS, (wi[:, 2] < -MIN_COS) & (sin2_t < 1.0))
    return wi, valid


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
    glass: np.ndarray | None = None  # (m,) w_g, the glass lobe's weight (None: no glass)
    eta: np.ndarray | None = None  # (m,) relative index for the side of wo
    tint: np.ndarray | None = None  # (m, 3) transmission tint (the base colour)

    @property
    def transmits(self) -> np.ndarray:
        """Rows whose BSDF also transmits (light from below the surface counts)."""
        if self.glass is None:
            return np.zeros(self.n.shape[0], dtype=bool)
        return self.glass > 0.0

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
        transmission: np.ndarray | None = None,
        ior: np.ndarray | None = None,
        entering: np.ndarray | None = None,
    ) -> BSDF:
        """``n`` must face ``wo_world`` (``n . wo > 0``); see the module docstring.
        ``transmission`` and ``ior`` per point, ``entering``: whether ``wo`` is on the
        surface's outward side."""
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
        out = cls(t, b, n, wo, c_diff, f0, rough * rough, has_specular.copy(), p)
        if transmission is not None and np.any(transmission > 0.0):
            w_g = np.clip(transmission, 0.0, 1.0) * (1.0 - metallic[:, 0])
            ior = np.asarray(ior, np.float64)
            out.glass = w_g
            out.eta = np.where(entering, ior, 1.0 / ior)
            out.tint = base.copy()
        return out

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
        if self.glass is None:
            return f, pdf, np.maximum(wi[:, 2], 0.0)
        f, pdf = self._with_glass(wi, f, pdf, rows)
        cos = np.where(self.transmits[rows], np.abs(wi[:, 2]), np.maximum(wi[:, 2], 0.0))
        return f, pdf, cos

    def _with_glass(self, wi, f, pdf, rows=slice(None)):
        """Mix in the glass lobe where it exists: ``(1 - w_g) f + w_g f_glass``."""
        idx = np.arange(self.n.shape[0])[rows]
        w = self.glass[idx]
        g = np.flatnonzero(w > 0.0)
        if not g.size:
            return f, pdf
        f = f.copy()
        pdf = pdf.copy()
        k = idx[g]
        fg, pg = glass_eval(self.wo[k], wi[g], self.alpha[k], self.eta[k], self.tint[k])
        wg = w[g]
        f[g] = (1.0 - wg)[:, None] * f[g] + wg[:, None] * fg
        pdf[g] = (1.0 - wg) * pdf[g] + wg * pg
        return f, pdf

    # -- sampling ----------------------------------------------------------------------

    def sample(self, u_lobe: np.ndarray, u1: np.ndarray, u2: np.ndarray):
        """Sample incoming directions.

        Returns ``(wi_world, weight = f cos / pdf, pdf, valid, specular)`` where
        ``specular`` tells which lobe produced each direction.
        """
        if self.glass is not None:
            return self._sample_with_glass(u_lobe, u1, u2)
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

    def _sample_with_glass(self, u_lobe, u1, u2):
        """``sample`` with glass lobes: the glass lobe is picked with probability ``w_g``
        (the rest of ``u_lobe`` then picks reflection or refraction), else the opaque
        lobes as before (with ``u_lobe`` rescaled)."""
        w = self.glass
        glass = u_lobe < w
        rest = np.where(glass, 0.0, (u_lobe - w) / np.where(w < 1.0, 1.0 - w, 1.0))
        spec = (rest < self.p_spec) & ~glass
        wi = cosine_hemisphere(u1, u2)
        if spec.any():
            h = ggx_visible_normal(self.wo[spec], self.alpha[spec], u1[spec], u2[spec])
            wo = self.wo[spec]
            wi[spec] = 2.0 * (wo * h).sum(axis=1, keepdims=True) * h - wo
        good = np.ones(wi.shape[0], dtype=bool)
        if glass.any():
            g = np.flatnonzero(glass)
            choice = u_lobe[g] / np.maximum(w[g], 1e-300)
            wi[g], good[g] = glass_sample(
                self.wo[g], self.alpha[g], self.eta[g], choice, u1[g], u2[g]
            )
        f, pdf = self._eval_local(wi)
        f, pdf = self._with_glass(wi, f, pdf)
        cos = np.where(self.transmits, np.abs(wi[:, 2]), wi[:, 2])
        valid = good & (cos > MIN_COS) & (pdf > 0.0)
        weight = np.where(valid[:, None], f * (cos / np.where(valid, pdf, 1.0))[:, None], 0.0)
        return to_world(wi, self.t, self.b, self.n), weight, pdf, valid, spec | glass

    def albedo(self) -> np.ndarray:
        """Directional albedo estimate used for the albedo pass: ``c_diff + F(n.v)``
        (glass: its tint)."""
        ndv = self.wo[:, 2]
        f_view = self.f0 + (1.0 - self.f0) * ((1.0 - ndv) ** 5)[:, None]
        out = self.c_diff * (1.0 - f_view) + f_view
        if self.glass is not None:
            w = self.glass[:, None]
            out = (1.0 - w) * out + w * self.tint
        return out
