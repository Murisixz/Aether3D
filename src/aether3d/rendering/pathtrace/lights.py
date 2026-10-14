"""Light sources and next-event estimation (direct light sampling) for the path tracer.

**Units.** A light of intensity ``I`` and linear color ``c`` lights a white
Lambertian surface facing it to radiance ``c I`` on the rasterizers (their
``E (n.l)`` term). With the BRDF ``base / pi`` that is an irradiance of
``pi c I`` (directional lights) and a radiant intensity of ``pi c I``
(point lights, attenuated by ``1/d^2`` with ``falloff="inverse_square"``,
not at all with ``"none"``, and windowed by ``range`` as on the
rasterizers), so a directly lit diffuse surface has the same radiance on
every backend.

**Analytic lights** are sampled at every path vertex, ``light_samples``
shadow rays each:

* directional, ``angle = 0``: the single direction toward the light;
* directional, ``angle > 0``: a disk of uniform radiance ``L`` filling the
  cone of half-angle ``a = angle / 2``, sampled uniformly in that cone
  (``pdf = 1 / Omega``, ``Omega = 2 pi (1 - cos a)``). Its irradiance on a
  surface facing it is ``pi L sin^2 a``; choosing ``L = c I / sin^2 a``
  keeps it ``pi c I``, so each sample contributes ``f cos x L Omega = f cos
  x pi c I x 2 / (1 + cos a)``;
* point, ``radius = 0``: the direction to the light, shadow ray up to it;
* point, ``radius > 0``: a sphere of uniform radiance sampled uniformly in
  the cone it subtends (``sin a = radius / d``), shadow rays up to its
  surface. Facing it, a surface receives ``pi L sin^2 a``; ``L`` is chosen so
  this equals the point light's ``pi c I / d^2`` (with ``falloff`` and
  ``range`` applied at the center distance ``d``): every sample contributes
  ``f cos x E x 2 (1 - cos a) / sin^2 a``.

So sizing a light keeps how strongly it lights surfaces facing it and only
softens shadows (and spreads light around a bit near the source).

These lights cannot be hit by rays (they are not geometry), so their
direct-light estimates need no MIS weight.

**Emissive surfaces** (materials with ``emissive`` color x strength) are
area lights: radiance ``Le`` leaves each point of the front face (both
faces of double-sided materials). Next-event estimation picks one emissive
triangle per sample with probability proportional to its power estimate
(area x luminance of the emissive factor, doubled when double-sided), then
a uniform point on it, with the solid-angle density
``p = P(triangle) / area x d^2 / |cos theta_light|``. Rays the BSDF sends
toward the same surfaces also find them, so both strategies are weighted
with Veach's power heuristic; together they count every emitter once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import surface
from ..surface import LUMINANCE, MIN_LIGHT_DISTANCE
from .sampling import (
    SLOT_LIGHTS,
    dimension,
    frame,
    power_heuristic,
    to_world,
    uniform,
    uniform_cone,
    uniform_triangle,
)

SHADOW_END = 1.0 - 1e-7
"""Shadow rays stop this fraction short of the sampled point on the light."""


@dataclass
class Emitters:
    gtri: np.ndarray  # (E,) global triangle ids
    inst: np.ndarray
    prim: np.ndarray
    material: np.ndarray
    normal: np.ndarray  # (E, 3) outward unit geometric normal
    area: np.ndarray  # (E,)
    pmf: np.ndarray  # (E,) selection probability
    cdf: np.ndarray  # (E,) cumulative pmf (last = 1)
    double_sided: np.ndarray  # (E,) bool
    lookup: np.ndarray  # (T,) global triangle -> emitter index (-1: none)

    @property
    def count(self) -> int:
        return int(self.gtri.shape[0])


@dataclass
class LightSet:
    kind: np.ndarray  # (L,) 0: directional, 1: point
    radiance: np.ndarray  # (L, 3) linear color x intensity
    position: np.ndarray  # (L, 3)
    to_light: np.ndarray  # (L, 3) directional: unit direction toward the light
    cos_max: np.ndarray  # (L,) directional: cos(angle / 2)
    radius: np.ndarray  # (L,)
    range: np.ndarray  # (L,) inf: no range window
    inverse_square: np.ndarray  # (L,) bool
    emitters: Emitters | None

    @classmethod
    def build(cls, ps) -> LightSet:
        lights = ps.rs.lights
        n = len(lights)
        kind = np.array([0 if light.type == "directional" else 1 for light in lights], dtype=int)
        radiance = np.array([light.radiance for light in lights], dtype=np.float64).reshape(n, 3)
        position = np.zeros((n, 3))
        to_light = np.zeros((n, 3))
        cos_max = np.ones(n)
        radius = np.zeros(n)
        rng = np.full(n, np.inf)
        inverse_square = np.zeros(n, dtype=bool)
        for k, light in enumerate(lights):
            if light.type == "directional":
                to_light[k] = -np.asarray(light.direction, dtype=np.float64)
                cos_max[k] = math.cos(math.radians(light.angle) * 0.5)
            else:
                position[k] = light.position
                radius[k] = light.radius
                rng[k] = np.inf if light.range is None else light.range
                inverse_square[k] = light.falloff == "inverse_square"
        return cls(
            kind,
            radiance,
            position,
            to_light,
            cos_max,
            radius,
            rng,
            inverse_square,
            _emitters(ps),
        )

    @property
    def count(self) -> int:
        return int(self.kind.shape[0])

    def to_dict(self) -> dict:
        return {
            "lights": self.count,
            "sized_lights": int(
                np.count_nonzero((self.radius > 0) | ((self.kind == 0) & (self.cos_max < 1.0)))
            ),
            "emissive_triangles": 0 if self.emitters is None else self.emitters.count,
        }

    # ------------------------------------------------------------- sampling --

    def sample_analytic(self, index: int, p: np.ndarray, keys, vertex: int, sample: int):
        """Sample light ``index`` from points ``p``: ``(wi, target, scale)``.

        ``sample`` numbers the light samples of a vertex (it picks the random
        dimensions). ``target`` is the shadow-ray end point (``None``:
        infinitely far) and ``scale`` the unoccluded irradiance per unit
        ``cos`` (``(m, 3)``).
        """
        m = p.shape[0]
        slot = SLOT_LIGHTS + 3 * sample
        if self.kind[index] == 0:
            axis = np.broadcast_to(self.to_light[index], (m, 3))
            if self.cos_max[index] < 1.0:
                u1 = uniform(keys, dimension(vertex, slot))
                u2 = uniform(keys, dimension(vertex, slot + 1))
                local = uniform_cone(u1, u2, np.full(m, self.cos_max[index]))
                t, b = frame(np.ascontiguousarray(axis))
                wi = to_world(local, t, b, axis)
            else:
                wi = axis.copy()
            # Uniform radiance over the disk: pi c I at normal incidence (see the docstring).
            disk = 2.0 / (1.0 + self.cos_max[index])
            scale = np.broadcast_to(math.pi * self.radiance[index] * disk, (m, 3)).copy()
            return wi, None, scale
        center = self.position[index]
        v = center - p
        dist = np.linalg.norm(v, axis=1)
        axis = v / np.maximum(dist, 1e-300)[:, None]
        r = self.radius[index]
        target = np.broadcast_to(center, (m, 3)).copy()
        wi = axis
        sphere = np.ones(m)
        if r > 0.0:
            outside = dist > r
            if outside.any():
                rows = np.flatnonzero(outside)
                u1 = uniform(keys[rows], dimension(vertex, slot))
                u2 = uniform(keys[rows], dimension(vertex, slot + 1))
                sin2 = (r / dist[rows]) ** 2
                cmax = np.sqrt(np.maximum(0.0, 1.0 - sin2))
                local = uniform_cone(u1, u2, cmax)
                t, b = frame(axis[rows])
                w = to_world(local, t, b, axis[rows])
                # Where the sampled direction enters the sphere.
                proj = (w * v[rows]).sum(axis=1)
                disc = np.maximum(proj * proj - (dist[rows] ** 2 - r * r), 0.0)
                hit = np.maximum(proj - np.sqrt(disc), 0.0)
                wi = axis.copy()
                wi[rows] = w
                target[rows] = p[rows] + w * hit[:, None]
                sphere[rows] = 2.0 * (1.0 - cmax) / sin2
        d = np.maximum(dist, MIN_LIGHT_DISTANCE)
        atten = 1.0 / (d * d) if self.inverse_square[index] else np.ones(m)
        rng = self.range[index]
        if np.isfinite(rng):
            ratio = d / rng
            atten = atten * np.clip(1.0 - ratio**4, 0.0, 1.0) ** 2
        scale = (math.pi * self.radiance[index])[None, :] * (atten * sphere)[:, None]
        return wi, target, scale

    def sample_emitter(self, ps, p: np.ndarray, keys, vertex: int, sample: int):
        """Sample points on emissive triangles: ``(wi, target, le_over_pdf, pdf, gtri, inst)``.

        ``sample`` numbers the light samples of a vertex (after the analytic
        lights'). ``le_over_pdf`` is ``Le / p`` (``(m, 3)``, zero where nothing
        is emitted toward ``p``); ``pdf`` the solid-angle density ``p``.
        """
        em = self.emitters
        slot = SLOT_LIGHTS + 3 * sample
        u0 = uniform(keys, dimension(vertex, slot))
        u1 = uniform(keys, dimension(vertex, slot + 1))
        u2 = uniform(keys, dimension(vertex, slot + 2))
        k = np.minimum(np.searchsorted(em.cdf, u0, side="right"), em.count - 1)
        bary = uniform_triangle(u1, u2)
        gtri = em.gtri[k]
        p0, p1, p2 = ps.corners(gtri)
        x = bary[:, 0:1] * p0 + bary[:, 1:2] * p1 + bary[:, 2:3] * p2
        v = x - p
        dist2 = (v * v).sum(axis=1)
        dist = np.sqrt(dist2)
        wi = v / np.maximum(dist, 1e-300)[:, None]
        cos_l = -(em.normal[k] * wi).sum(axis=1)
        emits = (cos_l > 0.0) | em.double_sided[k]
        abs_cos = np.abs(cos_l)
        ok = emits & (abs_cos > 1e-12) & (dist > 0.0)
        pdf = np.where(ok, em.pmf[k] / em.area[k] * dist2 / np.where(ok, abs_cos, 1.0), 0.0)
        toward = np.where(cos_l > 0.0, 1.0, -1.0)
        le = self.emission(ps, gtri, em.inst[k], bary, toward)
        le_over_pdf = np.where(ok[:, None], le / np.where(ok, pdf, 1.0)[:, None], 0.0)
        return wi, x, le_over_pdf, pdf, gtri, em.inst[k]

    def emission(self, ps, gtri, inst, bary, toward) -> np.ndarray:
        """Emitted radiance at points of emissive triangles (textures and alpha masks)."""
        mat = ps.MAT[gtri]
        out = np.zeros((gtri.shape[0], 3))
        zero = np.zeros((gtri.shape[0], 3))
        for index in np.unique(mat):
            rows = np.flatnonzero(mat == index)
            material = ps.rs.materials[int(index)]
            if not material.textures and not material.masked:
                out[rows] = material.emissive
                continue
            g, i = gtri[rows], inst[rows]
            normal, _ = ps.geometric_normal(g, i)
            s = surface.Samples(g, i, bary[rows], zero[rows], zero[rows], normal, toward[rows])
            sv = surface.evaluate_material(ps.ctx, s, material)
            le = sv.emissive
            if material.masked:
                le = np.where((sv.alpha >= material.alpha_cutoff)[:, None], le, 0.0)
            out[rows] = le
        return out

    def emitter_pdf(self, it, gtri: np.ndarray) -> np.ndarray:
        """Solid-angle density with which next-event estimation picks these hit points."""
        em = self.emitters
        k = em.lookup[gtri]
        ok = k >= 0
        cos_l = np.abs((it.ng * it.d).sum(axis=1))
        kk = np.where(ok, k, 0)
        safe = np.where(cos_l > 1e-12, cos_l, 1.0)
        return np.where(ok & (cos_l > 1e-12), em.pmf[kk] / em.area[kk] * it.t * it.t / safe, 0.0)


def _emitters(ps) -> Emitters | None:
    mats = ps.rs.materials
    glowing = np.array([bool(np.any(m.emissive > 0.0)) for m in mats], dtype=bool)
    if not glowing.any() or ps.triangle_count == 0:
        return None
    gtri = np.flatnonzero(glowing[ps.MAT])
    if not gtri.size:
        return None
    inst = ps.INST[gtri].astype(np.int64)
    normal, area = ps.geometric_normal(gtri, inst)
    material = ps.MAT[gtri]
    emissive = np.array([m.emissive for m in mats], dtype=np.float64).reshape(-1, 3)
    double = np.array([m.double_sided for m in mats], dtype=bool)
    power = area * (emissive[material] @ LUMINANCE) * np.where(double[material], 2.0, 1.0)
    keep = power > 0.0
    if not keep.any():
        return None
    gtri, inst, normal, area, material, power = (
        a[keep] for a in (gtri, inst, normal, area, material, power)
    )
    pmf = power / power.sum()
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    lookup = np.full(ps.triangle_count, -1, dtype=np.int64)
    lookup[gtri] = np.arange(gtri.size)
    prim = gtri - ps.tri_offset[inst]
    return Emitters(gtri, inst, prim, material, normal, area, pmf, cdf, double[material], lookup)


def mis_weight_emission(lights: LightSet, it, pdf_bsdf: np.ndarray) -> np.ndarray:
    """MIS weight of emission found by BSDF sampling (power heuristic against NEE)."""
    if lights.emitters is None:
        return np.ones(it.gtri.shape[0])
    pdf_light = lights.emitter_pdf(it, it.gtri)
    return np.where(pdf_light > 0.0, power_heuristic(pdf_bsdf, pdf_light), 1.0)
