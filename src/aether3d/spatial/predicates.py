"""Exact orientation predicates, vectorized: a float filter, exact arithmetic when unsure.

``orient3d(a, b, c, d)`` is the sign of ``det[a - d, b - d, c - d]``: positive when ``d``
lies below the plane through ``a``, ``b``, ``c`` (which appear counter-clockwise seen from
above), negative above, zero exactly when the four points are coplanar. ``orient2d(a, b,
c)`` is the sign of ``det[a - c, b - c]``: positive when ``a``, ``b``, ``c`` turn
counter-clockwise.

Each determinant is first evaluated in float64; Shewchuk's static error bounds ("Adaptive
Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", Discrete &
Computational Geometry 18, 1997: ``ccwerrboundA = (3 + 16 eps) eps`` and
``o3derrboundA = (7 + 56 eps) eps`` times the permanent) certify the sign when the value
is farther from zero than the bound. The remaining rows -- coplanar or nearly so -- are
evaluated exactly with rational arithmetic (``fractions.Fraction``: every float is a
rational number), so every answer is the exact sign for the given float inputs. Inputs
must be finite; values whose products could overflow or underflow (magnitudes beyond
``1e140`` or below ``1e-140`` but not zero) always take the exact path.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

EPS = 2.0**-53
CCW_BOUND = (3.0 + 16.0 * EPS) * EPS
O3D_BOUND = (7.0 + 56.0 * EPS) * EPS
_HUGE, _TINY = 1e140, 1e-140


def _unsafe(*arrays: np.ndarray) -> np.ndarray:
    """Rows with magnitudes for which the float filter's bound is not valid."""
    out = np.zeros(arrays[0].shape[0], dtype=bool)
    for x in arrays:
        ax = np.abs(x)
        out |= ((ax > _HUGE) | ((ax < _TINY) & (ax > 0))).any(axis=1)
    return out


def orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact signs (-1, 0, 1) of ``det[a - c, b - c]`` for rows of 2D points."""
    a, b, c = (np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in (a, b, c))
    with np.errstate(over="ignore", invalid="ignore"):  # such rows take the exact path
        left = (a[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1])
        right = (a[:, 1] - c[:, 1]) * (b[:, 0] - c[:, 0])
        det = left - right
        bound = CCW_BOUND * (np.abs(left) + np.abs(right))
        sign = np.sign(det).astype(np.int8)
        unsure = ~(np.abs(det) > bound) | _unsafe(a, b, c)
    for i in np.flatnonzero(unsure):
        ax, ay = Fraction(a[i, 0]), Fraction(a[i, 1])
        bx, by = Fraction(b[i, 0]), Fraction(b[i, 1])
        cx, cy = Fraction(c[i, 0]), Fraction(c[i, 1])
        value = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
        sign[i] = (value > 0) - (value < 0)
    return sign


def orient3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Exact signs (-1, 0, 1) of ``det[a - d, b - d, c - d]`` for rows of 3D points."""
    a, b, c, d = (np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in (a, b, c, d))
    with np.errstate(over="ignore", invalid="ignore"):  # such rows take the exact path
        ad, bd, cd = a - d, b - d, c - d
        bdxcdy = bd[:, 0] * cd[:, 1]
        cdxbdy = cd[:, 0] * bd[:, 1]
        cdxady = cd[:, 0] * ad[:, 1]
        adxcdy = ad[:, 0] * cd[:, 1]
        adxbdy = ad[:, 0] * bd[:, 1]
        bdxady = bd[:, 0] * ad[:, 1]
        det = (
            ad[:, 2] * (bdxcdy - cdxbdy)
            + bd[:, 2] * (cdxady - adxcdy)
            + cd[:, 2] * (adxbdy - bdxady)
        )
        permanent = (
            (np.abs(bdxcdy) + np.abs(cdxbdy)) * np.abs(ad[:, 2])
            + (np.abs(cdxady) + np.abs(adxcdy)) * np.abs(bd[:, 2])
            + (np.abs(adxbdy) + np.abs(bdxady)) * np.abs(cd[:, 2])
        )
        sign = np.sign(det).astype(np.int8)
        unsure = ~(np.abs(det) > O3D_BOUND * permanent) | _unsafe(a, b, c, d)
    for i in np.flatnonzero(unsure):
        pa = [Fraction(x) for x in a[i]]
        pb = [Fraction(x) for x in b[i]]
        pc = [Fraction(x) for x in c[i]]
        pd = [Fraction(x) for x in d[i]]
        u = [pa[k] - pd[k] for k in range(3)]
        v = [pb[k] - pd[k] for k in range(3)]
        w = [pc[k] - pd[k] for k in range(3)]
        value = (
            u[2] * (v[0] * w[1] - v[1] * w[0])
            + v[2] * (w[0] * u[1] - w[1] * u[0])
            + w[2] * (u[0] * v[1] - u[1] * v[0])
        )
        sign[i] = (value > 0) - (value < 0)
    return sign


def _drop_axis(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The axis to drop to project each triangle's plane onto 2D without degenerating
    (the largest component of its normal; dropping a coordinate is exact)."""
    n = np.cross(b - a, c - a)
    return np.argmax(np.abs(n), axis=1)


def _project(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    keep = np.array([[1, 2], [0, 2], [0, 1]])[axis]
    return np.take_along_axis(points, keep, axis=1)


def segment_crosses_triangle(
    p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Whether each closed segment ``p q`` meets triangle ``a b c`` (closed), exactly.

    Non-coplanar rows: the segment's ends are not strictly on one side of the plane, and
    the line through ``p q`` passes inside or on the triangle (the three edge
    orientations do not disagree). Coplanar rows are decided in the triangle's plane:
    an end inside the triangle, or the segment crossing or touching one of its edges.
    """
    p, q, a, b, c = (np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in (p, q, a, b, c))
    sp = orient3d(a, b, c, p)
    sq = orient3d(a, b, c, q)
    out = np.zeros(len(p), dtype=bool)
    general = ~((sp == 0) & (sq == 0)) & (sp * sq <= 0)
    rows = np.flatnonzero(general)
    if len(rows):
        o1 = orient3d(p[rows], q[rows], a[rows], b[rows])
        o2 = orient3d(p[rows], q[rows], b[rows], c[rows])
        o3 = orient3d(p[rows], q[rows], c[rows], a[rows])
        pos = (o1 >= 0) & (o2 >= 0) & (o3 >= 0)
        neg = (o1 <= 0) & (o2 <= 0) & (o3 <= 0)
        out[rows] = pos | neg
    flat = np.flatnonzero((sp == 0) & (sq == 0))
    if len(flat):
        axis = _drop_axis(a[flat], b[flat], c[flat])
        P, Q = _project(p[flat], axis), _project(q[flat], axis)
        A, B, C = _project(a[flat], axis), _project(b[flat], axis), _project(c[flat], axis)
        hit = _point_in_triangle_2d(P, A, B, C) | _point_in_triangle_2d(Q, A, B, C)
        for u, v in ((A, B), (B, C), (C, A)):
            hit |= _segments_meet_2d(P, Q, u, v)
        out[flat] = hit
    return out


def _point_in_triangle_2d(p, a, b, c) -> np.ndarray:
    s = orient2d(a, b, c)
    o1, o2, o3 = orient2d(a, b, p), orient2d(b, c, p), orient2d(c, a, p)
    pos = (o1 >= 0) & (o2 >= 0) & (o3 >= 0)
    neg = (o1 <= 0) & (o2 <= 0) & (o3 <= 0)
    return (s != 0) & np.where(s > 0, pos, neg)


def _segments_meet_2d(p1, q1, p2, q2) -> np.ndarray:
    d1 = orient2d(p2, q2, p1)
    d2 = orient2d(p2, q2, q1)
    d3 = orient2d(p1, q1, p2)
    d4 = orient2d(p1, q1, q2)
    proper = (d1 * d2 < 0) & (d3 * d4 < 0)

    def on(a, b, x, d):
        within = (
            (np.minimum(a[:, 0], b[:, 0]) <= x[:, 0])
            & (x[:, 0] <= np.maximum(a[:, 0], b[:, 0]))
            & (np.minimum(a[:, 1], b[:, 1]) <= x[:, 1])
            & (x[:, 1] <= np.maximum(a[:, 1], b[:, 1]))
        )
        return (d == 0) & within

    return (
        proper | on(p2, q2, p1, d1) | on(p2, q2, q1, d2) | on(p1, q1, p2, d3) | on(p1, q1, q2, d4)
    )
