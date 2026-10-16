"""TriangleSurface (nearest points, rays) and the Morton BVH, against brute force."""

import numpy as np
import pytest

from aether3d import create_sphere
from aether3d.rendering.raytrace.bvh import build_bvh_morton, morton_codes, triangle_bounds
from aether3d.spatial.geometry import bvh_box_pairs, closest_points_on_triangles
from aether3d.spatial.surface import TriangleSurface, _moller_trumbore


def soup(rng, n=300):
    """Random triangles in a unit cube (overlapping, degenerate-free)."""
    a = rng.uniform(-1, 1, size=(n, 3))
    b = a + rng.normal(scale=0.2, size=(n, 3))
    c = a + rng.normal(scale=0.2, size=(n, 3))
    pts = np.concatenate([a, b, c])
    tris = np.arange(3 * n).reshape(3, n).T
    return pts, tris


def test_morton_bvh_box_pairs_match_brute_force():
    rng = np.random.default_rng(0)
    pts, tris = soup(rng, 500)
    lo, hi = triangle_bounds(pts, tris)
    for leaf in (1, 4, 16):
        bvh = build_bvh_morton(lo, hi, leaf)
        assert sorted(bvh.order.tolist()) == list(range(len(tris)))
        q = rng.uniform(-1.2, 1.2, size=(40, 3))
        qlo, qhi = q - 0.1, q + 0.1
        qi, pi = bvh_box_pairs(bvh, lo, hi, qlo, qhi)
        brute = {
            (i, j)
            for i in range(len(q))
            for j in np.flatnonzero(np.all((qlo[i] <= hi) & (qhi[i] >= lo), axis=1))
        }
        assert set(zip(qi.tolist(), pi.tolist(), strict=True)) == brute


def test_morton_codes_interleave_bits():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    codes = morton_codes(p, np.zeros(3), np.ones(3))
    assert codes[0] == 0
    # x occupies bits 0, 3, 6 ...; y bits 1, 4, ...; z bits 2, 5, ...
    assert codes[1] & np.uint64(1) and not codes[1] & np.uint64(2)
    assert codes[2] & np.uint64(2) and codes[3] & np.uint64(4)
    assert codes[1] < codes[2] < codes[3]


@pytest.mark.parametrize("seed", [1, 2])
def test_nearest_points_are_exact(seed):
    rng = np.random.default_rng(seed)
    pts, tris = soup(rng)
    surface = TriangleSurface(pts, tris)
    q = rng.uniform(-1.5, 1.5, size=(200, 3))
    got = surface.nearest(q)
    for i, p in enumerate(q):
        cp, _ = closest_points_on_triangles(p, surface.a, surface.b, surface.c)
        d = np.sqrt(((cp - p) ** 2).sum(axis=1))
        assert got.distance[i] == d.min()  # bit for bit: the same kernel decides
        assert d[got.triangles[i]] == d.min()
    capped = surface.nearest(q, max_distance=0.05)
    far = got.distance > 0.05
    assert np.all(capped.triangles[far] == -1) and np.all(capped.triangles[~far] >= 0)


def test_rays_find_the_closest_hit_from_either_side():
    rng = np.random.default_rng(4)
    sphere = create_sphere(1.0, 48, 24)
    surface = TriangleSurface(sphere.positions, sphere.triangles)
    o = rng.normal(size=(300, 3)) * 3
    d = rng.normal(size=(300, 3))
    hit = surface.raycast(o, d)
    for i in range(len(o)):
        t, *_ = _moller_trumbore(
            np.repeat(o[i : i + 1], len(surface.a), 0),
            np.repeat(d[i : i + 1], len(surface.a), 0),
            surface.a,
            surface.b,
            surface.c,
        )
        t = t[t > 0]
        expected = t.min() if len(t) else np.inf
        assert hit.t[i] == expected
    inside = surface.raycast(np.zeros((5, 3)), rng.normal(size=(5, 3)))
    assert np.all(np.isfinite(inside.t)) and not inside.front.any()  # back faces
    limited = surface.raycast(o, d, tmax=0.5)
    assert np.all(limited.t[np.isfinite(limited.t)] < 0.5)
