"""Exact orientation predicates: the float filter never decides a wrong sign, and the
segment-triangle test is exact on coplanar and touching configurations."""

from fractions import Fraction

import numpy as np

from aether3d.spatial.predicates import orient2d, orient3d, segment_crosses_triangle


def exact3(a, b, c, d) -> int:
    u = [Fraction(float(a[k])) - Fraction(float(d[k])) for k in range(3)]
    v = [Fraction(float(b[k])) - Fraction(float(d[k])) for k in range(3)]
    w = [Fraction(float(c[k])) - Fraction(float(d[k])) for k in range(3)]
    det = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return (det > 0) - (det < 0)


def exact2(a, b, c) -> int:
    det = (Fraction(float(a[0])) - Fraction(float(c[0]))) * (
        Fraction(float(b[1])) - Fraction(float(c[1]))
    ) - (Fraction(float(a[1])) - Fraction(float(c[1]))) * (
        Fraction(float(b[0])) - Fraction(float(c[0]))
    )
    return (det > 0) - (det < 0)


def test_orient3d_matches_exact_arithmetic_on_nearly_coplanar_points():
    rng = np.random.default_rng(3)
    n = 3000
    a, b, c = rng.normal(size=(3, n, 3))
    # Points on the plane through a, b, c (rounded to floats: on, just above or below).
    s, t = rng.uniform(-2, 2, size=(2, n, 1))
    d = a + s * (b - a) + t * (c - a)
    d[: n // 3] = a[: n // 3]  # exactly coplanar duplicates
    got = orient3d(a, b, c, d)
    want = np.array([exact3(a[i], b[i], c[i], d[i]) for i in range(n)])
    assert np.array_equal(got, want)
    assert (want == 0).sum() >= n // 3 and (want != 0).any()


def test_orient2d_matches_exact_arithmetic_on_nearly_collinear_points():
    rng = np.random.default_rng(4)
    n = 3000
    a, b = rng.normal(size=(2, n, 2)) * 1e3
    t = rng.uniform(-1, 2, size=(n, 1))
    c = a + t * (b - a)
    got = orient2d(a, b, c)
    want = np.array([exact2(a[i], b[i], c[i]) for i in range(n)])
    assert np.array_equal(got, want)


def test_the_filter_decides_clear_cases_and_huge_values_take_the_exact_path():
    a = np.array([[0.0, 0, 0], [1e200, 0, 0]])
    b = np.array([[1.0, 0, 0], [0, 1e200, 0]])
    c = np.array([[0.0, 1, 0], [0, 0, 0]])
    d = np.array([[0.0, 0, 1], [0, 0, 1e200]])
    assert orient3d(a, b, c, d).tolist() == [-1, -1]


def test_segment_triangle_cases():
    tri = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])

    def crosses(p, q):
        return bool(segment_crosses_triangle([p], [q], tri[[0]], tri[[1]], tri[[2]])[0])

    assert crosses([0.2, 0.2, -1], [0.2, 0.2, 1])  # through the interior
    assert not crosses([2, 2, -1], [2, 2, 1])  # misses
    assert crosses([0.5, 0.5, -1], [0.5, 0.5, 1])  # through the edge (closed triangle)
    assert crosses([0, 0, -1], [0, 0, 1])  # through a vertex
    assert crosses([0.2, 0.2, 0], [0.2, 0.2, 1])  # starts on the triangle
    assert not crosses([0.2, 0.2, 1e-300], [0.2, 0.2, 1])  # starts just above it
    # Coplanar: crossing an edge, inside, touching a vertex, outside.
    assert crosses([-1, 0.5, 0], [0.5, 0.2, 0])
    assert crosses([0.1, 0.1, 0], [0.2, 0.2, 0])
    assert crosses([-1, 1, 0], [1, -1, 0])  # touches only the corner (0, 0)
    assert not crosses([-1, 2, 0], [-0.5, 2, 0])
    # The far edge of a neighbour sharing the vertex (0, 0, 0): folded flat onto the
    # triangle it meets it; folded away by a hair it does not.
    assert crosses([0.6, 0.1, 0], [0.1, 0.6, 0])
    assert not crosses([0.6, 0.1, 1e-9], [0.1, 0.6, 1e-9])


def test_parallel_translated_triangles_do_not_intersect():
    # Two marching-cubes triangles one cell apart: parallel planes, so a float
    # ray-triangle determinant is all rounding. The exact test finds no contact.
    from aether3d.spatial.geometry import triangles_intersect

    a = np.array(
        [
            [0.8799999356269836, -0.8200000524520874, -0.2184751331806183],
            [0.9299999475479126, -0.8366247415542603, -0.17000000178813934],
            [0.8966246247291565, -0.8700000643730164, -0.17000000178813934],
        ]
    )
    b = np.array(
        [
            [0.9299999475479126, -0.7700000405311584, -0.2184751331806183],
            [0.9799999594688416, -0.7866247296333313, -0.17000000178813934],
            [0.9466246366500854, -0.8200000524520874, -0.17000000178813934],
        ]
    )
    for x, y in ((a, b), (b, a)):
        assert not triangles_intersect(x[[0]], x[[1]], x[[2]], y[[0]], y[[1]], y[[2]])[0]
