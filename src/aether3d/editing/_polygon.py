"""Planar-polygon geometry used by the editing engine.

All functions take polygon vertex positions as an ``(n, 3)`` array in loop
order (counter-clockwise when seen from the front, i.e. against the normal).
"""

from __future__ import annotations

import math

import numpy as np


def newell_normal(points: np.ndarray) -> np.ndarray:
    """Newell's area vector: direction = front normal, length = 2 x area.

    Robust for non-planar and concave polygons; zero for degenerate ones.
    (Computed in plain Python floats: faster than NumPy for small polygons.)
    """
    pts = (
        points.tolist() if isinstance(points, np.ndarray) else [list(map(float, q)) for q in points]
    )
    nx = ny = nz = 0.0
    x1, y1, z1 = pts[-1]
    for x2, y2, z2 in pts:
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
        x1, y1, z1 = x2, y2, z2
    return np.array([nx, ny, nz])


def polygon_area(points: np.ndarray) -> float:
    return float(np.linalg.norm(newell_normal(points)) / 2.0)


def unit(v: np.ndarray) -> np.ndarray | None:
    """``v`` scaled to length 1, or ``None`` if it has (near-)zero or non-finite length."""
    if len(v) == 3:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        length = math.sqrt(x * x + y * y + z * z)
        if length < 1e-300 or not math.isfinite(length):
            return None
        return np.array([x / length, y / length, z / length])
    length = float(np.linalg.norm(v))
    if length < 1e-300 or not math.isfinite(length):
        return None
    return v / length


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal ``(u, v)`` spanning the plane, with ``u x v == normal``."""
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not length > 1e-300:
        raise ValueError("cannot build a plane basis from a zero normal")
    nx, ny, nz = nx / length, ny / length, nz / length
    hx, hy, hz = (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0)
    ux, uy, uz = hy * nz - hz * ny, hz * nx - hx * nz, hx * ny - hy * nx  # helper x n
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ul, uy / ul, uz / ul
    vx, vy, vz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux  # n x u
    return np.array([ux, uy, uz]), np.array([vx, vy, vz])


def project(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """2D coordinates in the polygon plane; front-facing loops become counter-clockwise."""
    u, v = plane_basis(normal)
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p @ u, p @ v], axis=1)


def _project_list(points: np.ndarray, normal: np.ndarray) -> list[tuple[float, float]]:
    u, v = plane_basis(normal)
    ux, uy, uz = u.tolist()
    vx, vy, vz = v.tolist()
    pts = points.tolist() if isinstance(points, np.ndarray) else points
    return [(x * ux + y * uy + z * uz, x * vx + y * vy + z * vz) for x, y, z in pts]


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _segments_intersect(p1, p2, q1, q2, eps: float) -> bool:
    """True if closed segments p1p2 and q1q2 touch or cross."""
    d1, d2 = _cross2(q1, q2, p1), _cross2(q1, q2, p2)
    d3, d4 = _cross2(p1, p2, q1), _cross2(p1, p2, q2)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True

    def on_segment(a, b, c) -> bool:  # c collinear with ab and within its box
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )

    return (
        (abs(d1) <= eps and on_segment(q1, q2, p1))
        or (abs(d2) <= eps and on_segment(q1, q2, p2))
        or (abs(d3) <= eps and on_segment(p1, p2, q1))
        or (abs(d4) <= eps and on_segment(p1, p2, q2))
    )


def _scale_eps(pts2d: np.ndarray) -> float:
    extent = float(np.max(np.ptp(pts2d, axis=0))) if len(pts2d) else 0.0
    return 1e-12 * max(extent, 1e-300) ** 2


def is_simple(pts2d: np.ndarray) -> bool:
    """True if the closed 2D polygon does not touch or cross itself."""
    n = len(pts2d)
    if n < 3:
        return False
    eps = _scale_eps(pts2d)
    for i in range(n):
        a1, a2 = pts2d[i], pts2d[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue  # adjacent edges share an endpoint by construction
            if _segments_intersect(a1, a2, pts2d[j], pts2d[(j + 1) % n], eps):
                return False
    # Adjacent edges folding back onto each other (a spike) also make a polygon non-simple.
    for i in range(n):
        prev, cur, nxt = pts2d[i - 1], pts2d[i], pts2d[(i + 1) % n]
        if abs(_cross2(prev, cur, nxt)) <= eps and np.dot(prev - cur, nxt - cur) > 0:
            return False
    return True


def point_in_polygon(pt: np.ndarray, pts2d: np.ndarray) -> bool:
    """Even-odd test (points on the boundary may go either way)."""
    inside = False
    x, y = pt
    n = len(pts2d)
    for i in range(n):
        (x1, y1), (x2, y2) = pts2d[i], pts2d[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def chord_inside(pts2d: np.ndarray, i: int, j: int) -> bool:
    """True if the segment between non-adjacent vertices i and j lies inside the polygon."""
    n = len(pts2d)
    eps = _scale_eps(pts2d)
    a, b = pts2d[i], pts2d[j]
    for k in range(n):
        k2 = (k + 1) % n
        if k in (i, j) or k2 in (i, j):
            continue
        if _segments_intersect(a, b, pts2d[k], pts2d[k2], eps):
            return False
    # A chord touching a vertex of the polygon (other than its ends) is rejected above.
    return point_in_polygon((a + b) / 2.0, pts2d)


def _min_angle(a, b, c) -> float:
    best = math.pi
    for (px, py), (qx, qy), (rx, ry) in ((a, b, c), (b, c, a), (c, a, b)):
        ux, uy, vx, vy = qx - px, qy - py, rx - px, ry - py
        nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
        if nu == 0 or nv == 0:
            return 0.0
        best = min(best, math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))))
    return best


def _c2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def triangulate(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon by ear clipping; indices refer to ``points``.

    Triangles keep the polygon's winding. Collinear vertices (for example
    vertices inserted on an edge) are handled. Among valid ears the one with
    the largest minimum angle is clipped first, which is deterministic and
    avoids slivers. Raises ``ValueError`` if the polygon is degenerate or
    self-intersecting.
    """
    n = len(points)
    if n < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    normal = newell_normal(points)
    if unit(normal) is None:
        raise ValueError("polygon has zero area")
    if n == 3:
        return [(0, 1, 2)]
    pts = _project_list(points, normal)
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    eps = 1e-12 * max(extent, 1e-300) ** 2
    remaining = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        m = len(remaining)
        best, best_quality = None, -1.0
        for k in range(m):
            i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _c2(a, b, c) <= eps:
                continue  # reflex or collinear tip
            blocked = False
            for other in remaining:
                if other in (i0, i1, i2):
                    continue
                p = pts[other]
                if _c2(a, b, p) >= -eps and _c2(b, c, p) >= -eps and _c2(c, a, p) >= -eps:
                    blocked = True
                    break
            if blocked:
                continue
            quality = _min_angle(a, b, c)
            if quality > best_quality + 1e-12:
                best, best_quality = k, quality
            if m > 64:  # keep large polygons O(n^2): take the first valid ear
                break
        if best is None:
            raise ValueError(
                "polygon is self-intersecting or degenerate; it cannot be triangulated"
            )
        m = len(remaining)
        triangles.append((remaining[best - 1], remaining[best], remaining[(best + 1) % m]))
        del remaining[best]
    a, b, c = (pts[i] for i in remaining)
    if _c2(a, b, c) <= eps:
        raise ValueError("polygon is self-intersecting or degenerate; it cannot be triangulated")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def affine_fit(points: np.ndarray, values: np.ndarray):
    """Least-squares affine map from the polygon plane to attribute values.

    Returns ``f(target_points) -> values``. It reproduces any attribute that
    varies affinely across a planar polygon exactly (e.g. UVs of a planar
    face mapped without perspective), and is the natural generalisation of
    barycentric interpolation to n-gons.
    """
    p = np.asarray(points, dtype=np.float64)
    normal = newell_normal(p)
    if unit(normal) is None:
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        return lambda targets: np.tile(mean, (len(np.atleast_2d(targets)), 1))
    u, v = plane_basis(normal)
    origin = p.mean(axis=0)

    def coords(q):
        q = np.atleast_2d(np.asarray(q, dtype=np.float64)) - origin
        return np.stack([q @ u, q @ v, np.ones(len(q))], axis=1)

    solution, *_ = np.linalg.lstsq(coords(p), np.asarray(values, dtype=np.float64), rcond=None)
    return lambda targets: coords(targets) @ solution
