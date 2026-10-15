"""Independent brute-force references for the Stage 10 spatial query tests (plain NumPy,
sharing no code with aether3d.spatial)."""

from __future__ import annotations

import numpy as np


def ref_closest_on_segment(p, a, b):
    d = b - a
    dd = float(d @ d)
    t = 0.0 if dd == 0.0 else min(1.0, max(0.0, float((p - a) @ d) / dd))
    return a + t * d


def ref_closest_on_triangle(p, a, b, c):
    """Projection onto the plane when it falls inside, else the nearest edge point."""
    e0, e1 = b - a, c - a
    m = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]])
    rhs = np.array([(p - a) @ e0, (p - a) @ e1])
    if abs(np.linalg.det(m)) > 1e-14:
        s, t = np.linalg.solve(m, rhs)
        if s >= 0 and t >= 0 and s + t <= 1:
            return a + s * e0 + t * e1
    candidates = [ref_closest_on_segment(p, x, y) for x, y in ((a, b), (b, c), (c, a))]
    return min(candidates, key=lambda q: float((q - p) @ (q - p)))


def ref_triangles_intersect(t1, t2):
    """Moller's interval test (1997) for triangles in general position (not coplanar)."""

    def interval(tri, dist, direction):
        proj = tri @ direction
        signs = np.sign(dist)
        # the vertex alone on its side of the other plane
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            if signs[k] != signs[i] and signs[k] != signs[j] and signs[i] == signs[j]:
                break
        else:
            return None
        ta = proj[k] + (proj[i] - proj[k]) * dist[k] / (dist[k] - dist[i])
        tb = proj[k] + (proj[j] - proj[k]) * dist[k] / (dist[k] - dist[j])
        return min(ta, tb), max(ta, tb)

    n1 = np.cross(t1[1] - t1[0], t1[2] - t1[0])
    n2 = np.cross(t2[1] - t2[0], t2[2] - t2[0])
    d1 = (t1 - t2[0]) @ n2  # t1's vertices against t2's plane
    d2 = (t2 - t1[0]) @ n1
    if np.all(d1 > 0) or np.all(d1 < 0) or np.all(d2 > 0) or np.all(d2 < 0):
        return False
    direction = np.cross(n1, n2)
    i1 = interval(t1, d1, direction)
    i2 = interval(t2, d2, direction)
    assert i1 is not None and i2 is not None
    return not (i1[1] < i2[0] or i2[1] < i1[0])


def ref_triangle_meets_box(tri, lo, hi):
    """Sutherland-Hodgman: clip the triangle by the box's six planes (boundaries kept)."""
    poly = [np.asarray(v, dtype=float) for v in tri]
    for axis in range(3):
        for sign, bound in ((1.0, hi[axis]), (-1.0, -lo[axis])):
            out = []
            for k, cur in enumerate(poly):
                prev = poly[k - 1]
                fc = sign * cur[axis] - bound
                fp = sign * prev[axis] - bound
                if fc <= 0:
                    if fp > 0:
                        out.append(prev + (cur - prev) * (fp / (fp - fc)))
                    out.append(cur)
                elif fp <= 0:
                    out.append(prev + (cur - prev) * (fp / (fp - fc)))
            poly = out
            if not poly:
                return False
    return True


def random_triangles(rng, n, spread=1.0):
    centers = rng.uniform(-spread, spread, size=(n, 1, 3))
    return centers + rng.normal(scale=0.6, size=(n, 3, 3))
