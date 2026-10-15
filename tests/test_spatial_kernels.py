"""Stage 10 geometric kernels against independent references (aether3d.spatial.geometry)."""

from __future__ import annotations

import numpy as np
import pytest
from spatial_helpers import (
    random_triangles,
    ref_closest_on_triangle,
    ref_triangle_meets_box,
    ref_triangles_intersect,
)

import aether3d as a3
from aether3d.rendering.raytrace.bvh import build_bvh
from aether3d.spatial import geometry as geo

RNG_SEED = 20261015


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_closest_points_on_triangles_match_the_reference():
    rng = np.random.default_rng(RNG_SEED)
    tris = random_triangles(rng, 400)
    points = rng.normal(scale=2.0, size=(400, 3))
    for p, tri in zip(points, tris, strict=True):
        got, bary = geo.closest_points_on_triangles(p, tri[0:1], tri[1:2], tri[2:3])
        want = ref_closest_on_triangle(p, *tri)
        assert np.allclose(got[0], want, atol=1e-9)
        assert np.isclose(bary[0].sum(), 1.0) and np.all(bary[0] >= -1e-12)
        assert np.allclose(bary[0] @ tri, got[0], atol=1e-9)


def test_closest_points_on_degenerate_triangles():
    p = np.array([[0.3, 2.0, 0.0]])
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    for c in (np.array([[2.0, 0.0, 0.0]]), a.copy(), b.copy()):  # collinear / coincident
        got, bary = geo.closest_points_on_triangles(p, a, b, c)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(bary))
        assert np.allclose(got[0], [0.3, 0.0, 0.0])
    point = np.zeros((1, 3))
    got, _ = geo.closest_points_on_triangles(p, point, point, point)
    assert np.allclose(got, 0.0)


def test_segment_distances_are_the_minimum_over_both_segments():
    rng = np.random.default_rng(RNG_SEED + 1)
    p1, q1, p2, q2 = rng.normal(size=(4, 300, 3))
    q1[:20] = p1[:20]  # degenerate segments (points)
    q2[10:30] = p2[10:30]
    d2, c1, c2 = geo.segment_distances(p1, q1, p2, q2)
    s = np.linspace(0.0, 1.0, 201)
    for k in range(300):
        a = p1[k] + s[:, None] * (q1[k] - p1[k])
        b = p2[k] + s[:, None] * (q2[k] - p2[k])
        sampled = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min()
        d = np.sqrt(d2[k])
        assert d <= sampled + 1e-12  # never above any pair of points
        step = 0.005 * (np.linalg.norm(q1[k] - p1[k]) + np.linalg.norm(q2[k] - p2[k]))
        assert d >= sampled - step - 1e-12
        assert np.isclose(np.linalg.norm(c1[k] - c2[k]), d)


def test_triangle_intersection_matches_mollers_interval_test():
    rng = np.random.default_rng(RNG_SEED + 2)
    t1 = random_triangles(rng, 3000, spread=0.4)
    t2 = random_triangles(rng, 3000, spread=0.4)
    got = geo.triangles_intersect(t1[:, 0], t1[:, 1], t1[:, 2], t2[:, 0], t2[:, 1], t2[:, 2])
    want = np.array([ref_triangles_intersect(a, b) for a, b in zip(t1, t2, strict=True)])
    assert 300 < want.sum() < 2700  # both outcomes are well represented
    assert np.array_equal(got, want)


def test_triangle_intersection_special_cases():
    def check(t1, t2, expected):
        t1, t2 = np.asarray(t1, float)[None], np.asarray(t2, float)[None]
        got = geo.triangles_intersect(t1[:, 0], t1[:, 1], t1[:, 2], t2[:, 0], t2[:, 1], t2[:, 2])
        assert bool(got[0]) is expected, (t1, t2)
        back = geo.triangles_intersect(t2[:, 0], t2[:, 1], t2[:, 2], t1[:, 0], t1[:, 1], t1[:, 2])
        assert bool(back[0]) is expected

    base = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    check(base, [(0.2, 0.2, 0), (2, 0.2, 0), (0.2, 2, 0)], True)  # coplanar overlap
    check(base, [(2, 2, 0), (3, 2, 0), (2, 3, 0)], False)  # coplanar apart
    check(base, [(1, 0, 0), (2, 0, 0), (1, 1, 0)], True)  # coplanar, sharing a vertex
    check(base, [(0, 0, 0), (1, 0, 0), (0.5, 0, 1)], True)  # sharing an edge, at an angle
    check(base, [(0.2, 0.2, 0), (0.3, 0.2, 1), (0.2, 0.3, 1)], True)  # touching at a vertex
    check(base, [(0.2, 0.2, 1e-9), (0.3, 0.2, 1), (0.2, 0.3, 1)], False)  # just above
    check(base, [(0.2, 0.2, -1), (0.3, 0.2, 1), (0.2, 0.3, 1)], True)  # piercing
    # Parallel edges (zero cross products) in parallel planes, offset.
    check(base, [(0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5)], False)
    # Degenerate triangles: a segment piercing, a segment beside, a point inside, a point off.
    check(base, [(0.2, 0.2, -1), (0.2, 0.2, 1), (0.2, 0.2, 1)], True)
    check(base, [(2, 2, -1), (2, 2, 1), (2, 2, 1)], False)
    check(base, [(0.2, 0.2, 0)] * 3, True)
    check(base, [(1.0, 1.0, 0)] * 3, False)  # in the plane, outside the triangle


def test_triangle_distances_are_exact_and_zero_when_intersecting():
    rng = np.random.default_rng(RNG_SEED + 3)
    t1 = random_triangles(rng, 200, spread=2.0)
    t2 = random_triangles(rng, 200, spread=2.0)
    dist, p1, p2 = geo.triangle_distances(
        *(t1[:, k] for k in range(3)), *(t2[:, k] for k in range(3))
    )
    hit = np.array([ref_triangles_intersect(a, b) for a, b in zip(t1, t2, strict=True)])
    assert np.all(dist[hit] == 0.0)
    g = np.linspace(0, 1, 41)
    s, t = np.meshgrid(g, g)
    keep = s + t <= 1
    s, t = s[keep], t[keep]
    for k in range(200):
        a = t1[k, 0] + s[:, None] * (t1[k, 1] - t1[k, 0]) + t[:, None] * (t1[k, 2] - t1[k, 0])
        b = t2[k, 0] + s[:, None] * (t2[k, 1] - t2[k, 0]) + t[:, None] * (t2[k, 2] - t2[k, 0])
        sampled = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min()
        assert dist[k] <= sampled + 1e-9
        if not hit[k]:
            assert np.isclose(np.linalg.norm(p1[k] - p2[k]), dist[k])
            assert dist[k] >= sampled - 0.1  # the grid is coarse; exactness is checked above
        else:  # both points are one common point of the two triangles
            assert np.allclose(p1[k], p2[k])
            for tri, q in ((t1[k], p1[k]), (t2[k], p2[k])):
                assert np.linalg.norm(ref_closest_on_triangle(q, *tri) - q) < 1e-9


def test_triangles_against_boxes_match_polygon_clipping():
    rng = np.random.default_rng(RNG_SEED + 4)
    tris = random_triangles(rng, 2000, spread=1.5)
    lo, hi = np.array([-0.5, -0.3, -0.4]), np.array([0.6, 0.5, 0.3])
    verts = geo.box_vertices(lo, hi)
    got = geo.triangles_intersect_convex(
        tris[:, 0], tris[:, 1], tris[:, 2], verts, geo.BOX_NORMALS, geo.BOX_NORMALS
    )
    want = np.array([ref_triangle_meets_box(t, lo, hi) for t in tris])
    assert 200 < want.sum() < 1800
    assert np.array_equal(got, want)


def test_winding_numbers_of_closed_and_open_meshes():
    cube = a3.create_cube(2.0)
    P = cube.positions.astype(float)
    a, b, c = (P[cube.triangles[:, k]] for k in range(3))
    pts = np.array([[0, 0, 0], [0.99, -0.99, 0.5], [1.01, 0, 0], [5, 5, 5]])
    assert np.allclose(geo.winding_numbers(pts, a, b, c), [1, 1, 0, 0], atol=1e-9)
    # reversed winding: -1 inside
    assert np.allclose(geo.winding_numbers(pts[:1], a, c, b), [-1])
    # an open mesh (the cube without its top) gives a fraction at the center
    keep = ~np.all(np.isclose(np.stack([a, b, c])[..., 2], 1.0), axis=0)
    w = geo.winding_numbers(pts[:1], a[keep], b[keep], c[keep])
    assert 0.7 < w[0] < 0.9  # 5 of 6 faces: 5/6 of the full solid angle


def test_component_representatives_weld_equal_positions():
    cube = a3.create_cube(1.0)  # 24 vertices: each face has its own (welded: one component)
    assert geo.component_representatives(cube.positions, cube.triangles).size == 1
    two = cube.positions.astype(np.float64)
    positions = np.concatenate([two, two + (5.0, 0.0, 0.0)])
    triangles = np.concatenate([cube.triangles, cube.triangles + 24])
    reps = geo.component_representatives(positions, triangles)
    assert reps.tolist() == [0, 24]


def test_bvh_box_pairs_find_every_overlapping_box():
    rng = np.random.default_rng(RNG_SEED + 5)
    lo = rng.uniform(-10, 10, size=(500, 3))
    hi = lo + rng.uniform(0, 2, size=(500, 3))
    qlo = rng.uniform(-10, 10, size=(60, 3))
    qhi = qlo + rng.uniform(0, 4, size=(60, 3))
    q, p = geo.bvh_box_pairs(build_bvh(lo, hi), lo, hi, qlo, qhi)
    got = set(zip(q.tolist(), p.tolist(), strict=True))
    overlap = np.all((qlo[:, None] <= hi[None]) & (qhi[:, None] >= lo[None]), axis=2)
    want = set(zip(*np.nonzero(overlap), strict=True))
    assert got == {(int(i), int(j)) for i, j in want}


@pytest.mark.parametrize("count", [1, 7, 100])
def test_box_pair_distances_bound_point_distances(count):
    rng = np.random.default_rng(RNG_SEED + count)
    lo1 = rng.uniform(-3, 3, size=(count, 3))
    hi1 = lo1 + rng.uniform(0, 2, size=(count, 3))
    lo2 = rng.uniform(-3, 3, size=(count, 3))
    hi2 = lo2 + rng.uniform(0, 2, size=(count, 3))
    near, far = geo.box_pair_distances(lo1, hi1, lo2, hi2)
    a = lo1[:, None] + rng.uniform(size=(count, 200, 3)) * (hi1 - lo1)[:, None]
    b = lo2[:, None] + rng.uniform(size=(count, 200, 3)) * (hi2 - lo2)[:, None]
    d = np.linalg.norm(a - b, axis=2)
    assert np.all(d >= near[:, None] - 1e-12) and np.all(d <= far[:, None] + 1e-12)
