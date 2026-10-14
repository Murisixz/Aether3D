"""The path tracer's BSDF and sample warps, checked by independent numerical integration.

Integrals over the hemisphere are computed here with a deterministic
midpoint rule on a fine grid in ``(cos theta, phi)`` (uniform in solid
angle), which shares nothing with the importance sampling under test and
resolves even the sharpest lobes tested (roughness 0.2: GGX alpha 0.04).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d.rendering import surface
from aether3d.rendering.pathtrace.bsdf import BSDF
from aether3d.rendering.pathtrace.sampling import (
    cosine_hemisphere,
    frame,
    path_keys,
    power_heuristic,
    uniform,
    uniform_cone,
    uniform_triangle,
)

N = 400_000


def uniform_hemisphere(rng, n):
    z = rng.random(n)
    phi = 2 * math.pi * rng.random(n)
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def quadrature(view, base, metallic, roughness, fn, n_cos=1024, n_phi=2048):
    """Midpoint-rule integral over the hemisphere of ``fn(f, pdf, cos)`` (per channel)."""
    z = (np.arange(n_cos) + 0.5) / n_cos
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * math.pi
    cell = (1.0 / n_cos) * (2 * math.pi / n_phi)
    total = 0.0
    rows = 64
    bsdf = make_bsdf(rows * n_phi, view, base, metallic, roughness)
    for z0 in range(0, n_cos, rows):
        zz = np.repeat(z[z0 : z0 + rows], n_phi)
        pp = np.tile(phi, rows)
        r = np.sqrt(np.maximum(0.0, 1 - zz * zz))
        w = np.stack([r * np.cos(pp), r * np.sin(pp), zz], axis=1)
        f, pdf, cos = bsdf.eval(w)
        total = total + fn(f, pdf, cos).sum(axis=0) * cell
    return total


def make_bsdf(n, view, base, metallic, roughness, specular=0.5, has_specular=True):
    normal = np.tile([0.0, 0.0, 1.0], (n, 1))
    wo = np.tile(view / np.linalg.norm(view), (n, 1))
    return BSDF.build(
        normal,
        wo,
        np.tile(base, (n, 1)),
        np.full(n, metallic),
        np.full(n, roughness),
        np.full(n, specular),
        np.full(n, has_specular),
    )


CASES = [
    # base, metallic, roughness, view direction
    ((0.8, 0.8, 0.8), 0.0, 1.0, (0.0, 0.0, 1.0)),
    ((0.8, 0.5, 0.2), 0.0, 0.5, (0.3, 0.0, 1.0)),
    ((1.0, 1.0, 1.0), 1.0, 0.2, (0.5, 0.2, 1.0)),
    ((0.9, 0.6, 0.3), 1.0, 0.6, (1.0, 0.0, 0.4)),
    ((0.5, 0.5, 0.5), 0.5, 0.35, (0.0, 1.0, 0.3)),
]


@pytest.mark.parametrize("base,metallic,roughness,view", CASES)
def test_sampling_weights_estimate_the_same_integral_as_uniform_sampling(
    base, metallic, roughness, view
):
    """E[f cos / pdf] over BSDF samples equals the directional albedo integral."""
    rng = np.random.default_rng(1)
    view, base = np.array(view, float), np.array(base)
    bsdf = make_bsdf(N, view, base, metallic, roughness)
    wi, weight, pdf, valid, _ = bsdf.sample(rng.random(N), rng.random(N), rng.random(N))
    importance = weight.mean(axis=0)
    # Independent reference: the integral of f cos over the hemisphere by quadrature.
    reference = quadrature(view, base, metallic, roughness, lambda f, p, c: f * c[:, None])
    np.testing.assert_allclose(importance, reference, rtol=0.01, atol=1e-3)
    # Energy: a passive surface never reflects more than it receives.
    assert np.all(reference <= 1.0 + 0.01)
    assert np.all(importance <= 1.0 + 0.01)


@pytest.mark.parametrize("base,metallic,roughness,view", CASES)
def test_the_sampling_density_is_the_density_of_the_samples(base, metallic, roughness, view):
    """The returned pdf integrates to the probability of valid samples and matches a histogram."""
    rng = np.random.default_rng(2)
    view, base = np.array(view, float), np.array(base)
    bsdf = make_bsdf(N, view, base, metallic, roughness)
    wi, _, pdf, valid, _ = bsdf.sample(rng.random(N), rng.random(N), rng.random(N))
    integral = quadrature(view, base, metallic, roughness, lambda f, p, c: p)
    assert integral == pytest.approx(valid.mean(), abs=0.005)
    # Histogram over polar-angle bands vs the density integrated over each band.
    edges = np.linspace(0.0, 1.0, 9)  # in cos(theta)
    counts, _ = np.histogram(wi[valid, 2], bins=edges)
    expected = []
    for k in range(8):
        lo, hi = edges[k], edges[k + 1]
        band = quadrature(
            view,
            base,
            metallic,
            roughness,
            lambda f, p, c, lo=lo, hi=hi: p * ((c >= lo) & (c < hi)),
        )
        expected.append(band * N)
    expected = np.array(expected)
    big = expected > 2000
    np.testing.assert_allclose(counts[big], expected[big], rtol=0.05)
    # The density each sample reports is the density at its direction.
    _, again, _ = bsdf.eval(wi)
    np.testing.assert_allclose(again[valid], pdf[valid], rtol=1e-9)


def test_lambertian_reflection_matches_the_rasterizers_diffuse_term():
    """f = base / pi: a white light of intensity 1 (irradiance pi) gives radiance = base."""
    bsdf = make_bsdf(4, np.array([0.0, 0.3, 1.0]), np.array([0.6, 0.4, 0.2]), 0, 1, 0, False)
    light = np.tile([0.2, -0.1, 1.0], (4, 1))
    light /= np.linalg.norm(light, axis=1, keepdims=True)
    f, _, cos = bsdf.eval(light)
    np.testing.assert_allclose(f * math.pi, np.tile([0.6, 0.4, 0.2], (4, 1)), rtol=1e-12)
    assert np.all(cos > 0)


def test_the_standard_brdf_times_pi_is_the_raster_shading():
    """pi f (n.l) for one light equals surface.shade_standard's light term exactly."""
    from aether3d.rendering.scene_data import RenderLight, RenderScene
    from aether3d.rendering.surface import SurfaceValues

    rng = np.random.default_rng(3)
    n = 50
    base = rng.random((n, 3))
    metallic = rng.random(n)
    rough = rng.uniform(0.05, 1.0, n)
    view = rng.normal(size=(n, 3))
    view[:, 2] = np.abs(view[:, 2]) + 0.2
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    light_dir = np.array([0.3, -0.2, 0.9])
    light_dir /= np.linalg.norm(light_dir)
    normal = np.tile([0.0, 0.0, 1.0], (n, 1))
    rs = RenderScene(
        (),
        (),
        (RenderLight(None, "L", "directional", np.ones(3), None, -light_dir, None, "none"),),
        None,
        np.zeros(3),
        np.zeros(3),
    )
    sv = SurfaceValues(base, np.ones(n), metallic, rough, np.ones(n), np.zeros((n, 3)), normal)
    raster = surface.shade_standard(rs, sv, 0.5, normal, np.zeros((n, 3)), view)
    bsdf = BSDF.build(normal, view, base, metallic, rough, np.full(n, 0.5), np.ones(n, bool))
    f, _, cos = bsdf.eval(np.tile(light_dir, (n, 1)))
    np.testing.assert_allclose(math.pi * f * cos[:, None], raster, rtol=1e-9, atol=1e-12)


def test_warps_have_their_documented_densities():
    rng = np.random.default_rng(4)
    u1, u2 = rng.random(N), rng.random(N)
    d = cosine_hemisphere(u1, u2)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    assert d[:, 2].mean() == pytest.approx(2 / 3, abs=3e-3)  # E[cos] under cos/pi
    cos_max = np.full(N, math.cos(0.3))
    c = uniform_cone(u1, u2, cos_max)
    assert c[:, 2].min() >= cos_max[0] - 1e-12
    assert c[:, 2].mean() == pytest.approx((1 + cos_max[0]) / 2, abs=1e-3)
    b = uniform_triangle(u1, u2)
    np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-12)
    assert b.min() >= 0 and np.allclose(b.mean(axis=0), 1 / 3, atol=2e-3)
    t, bt = frame(d)
    for a, bb in ((t, bt), (t, d), (bt, d)):
        np.testing.assert_allclose((a * bb).sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(t, bt), d, atol=1e-12)


def test_power_heuristic_weights_sum_to_one():
    a = np.array([0.0, 1.0, 2.0, 1e6])
    b = np.array([1.0, 1.0, 0.0, 3.0])
    np.testing.assert_allclose(power_heuristic(a, b) + power_heuristic(b, a), 1.0)


def test_random_numbers_depend_only_on_seed_pixel_sample_and_dimension():
    keys = path_keys(7, np.arange(1000), np.full(1000, 3))
    again = path_keys(7, np.arange(1000)[::-1], np.full(1000, 3))[::-1]
    assert np.array_equal(uniform(keys, 5), uniform(again, 5))
    values = uniform(keys, 5)
    assert values.min() >= 0.0 and values.max() < 1.0
    assert not np.array_equal(values, uniform(keys, 6))
    assert not np.array_equal(values, uniform(path_keys(8, np.arange(1000), np.full(1000, 3)), 5))
    big = uniform(path_keys(0, np.arange(200_000), np.zeros(200_000)), 1)
    assert big.mean() == pytest.approx(0.5, abs=3e-3)
    hist, _ = np.histogram(big, bins=10, range=(0, 1))
    assert hist.min() > 19_000 and hist.max() < 21_000
