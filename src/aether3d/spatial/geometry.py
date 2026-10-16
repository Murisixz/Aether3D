"""Exact geometric kernels of the spatial queries, vectorized with NumPy (float64).

Every function works on arrays of primitives at once. Triangles are given by
their corners ``a``, ``b``, ``c`` (each ``(n, 3)``).

* :func:`closest_points_on_triangles` -- the point of each triangle nearest to a
  query point (Ericson, *Real-Time Collision Detection*, 5.1.5: Voronoi regions of
  the vertices, edges and face), with its barycentric coordinates.
* :func:`segment_distances` -- closest points of pairs of segments (Ericson 5.1.9).
* :func:`triangles_intersect` -- whether pairs of triangles share a point (touching
  counts: triangles are closed sets). A separating axis proves a pair disjoint;
  the rest are decided constructively (an edge crossing the other triangle, or a
  zero vertex-triangle or edge-edge distance).
* :func:`triangle_distances` -- exact distances between pairs of triangles: zero if
  they intersect, else the least of the six vertex-triangle and nine edge-edge
  distances (the closest points of two disjoint triangles lie on a vertex of one
  or on two edges).
* :func:`triangles_intersect_convex` -- whether triangles meet a convex polyhedron
  (an axis-aligned box or a view frustum), by separating axes: the polyhedron's
  face normals, the triangle's normal and the cross products of their edges.
* :func:`winding_numbers` -- the generalized winding number of a closed or open
  triangle mesh at points (the sum of the triangles' signed solid angles over
  ``4 pi``; Van Oosterom and Strackee 1983): 1 inside and 0 outside a closed,
  consistently oriented mesh (-1 inside a mirrored one), fractional for open ones.
* :func:`bvh_box_pairs` -- every pair of a query box and a primitive of a
  :class:`~aether3d.rendering.raytrace.BVH` whose boxes overlap (a breadth-first
  traversal, vectorized over all query/node pairs at once): the candidates that
  exact tests then decide.
"""

from __future__ import annotations

import numpy as np

CHUNK = 262_144
"""Primitive pairs processed at once (bounds temporary memory)."""


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def triangle_normals(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unnormalized normals ``(b - a) x (c - a)`` (the winding's front side)."""
    return np.cross(b - a, c - a)


def closest_points_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point of each triangle to ``p`` (``(3,)`` or ``(n, 3)``) and its barycentrics.

    Degenerate triangles (segments, points) work too: the regions then collapse
    onto their edges and vertices.
    """
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), a.shape)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    n = a.shape[0]
    bary = np.zeros((n, 3))
    done = np.zeros(n, dtype=bool)

    def take(mask: np.ndarray, w: np.ndarray) -> None:
        nonlocal done
        mask = mask & ~done
        bary[mask] = w[mask]
        done |= mask

    one = np.ones(n)
    zero = np.zeros(n)
    take((d1 <= 0) & (d2 <= 0), np.stack([one, zero, zero], axis=1))  # vertex a
    take((d3 >= 0) & (d4 <= d3), np.stack([zero, one, zero], axis=1))  # vertex b
    take((d6 >= 0) & (d5 <= d6), np.stack([zero, zero, one], axis=1))  # vertex c
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        take((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - v, v, zero], axis=1))  # edge ab
        w = d2 / (d2 - d6)
        take((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - w, zero, w], axis=1))  # edge ac
        u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        take(
            (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
            np.stack([zero, 1 - u, u], axis=1),
        )  # edge bc
        denom = va + vb + vc
        vv = vb / denom
        ww = vc / denom
        face = np.stack([1 - vv - ww, vv, ww], axis=1)
    rest = ~done
    bary[rest] = face[rest]
    # Degenerate triangles (coincident or collinear corners) can make the region
    # formulas divide zero by zero: their nearest point is on one of their edges.
    bad = ~np.isfinite(bary).all(axis=1)
    for k in np.flatnonzero(bad):
        bary[k] = _degenerate_bary(p[k], a[k], b[k], c[k])
    points = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    return points, bary


def _degenerate_bary(p, a, b, c) -> np.ndarray:
    best, out = np.inf, np.array([1.0, 0.0, 0.0])
    for i, j, (s, e) in ((0, 1, (a, b)), (0, 2, (a, c)), (1, 2, (b, c))):
        d = e - s
        dd = float(d @ d)
        t = 0.0 if dd == 0.0 else min(1.0, max(0.0, float((p - s) @ d) / dd))
        q = s + t * d
        dist = float((p - q) @ (p - q))
        if dist < best:
            best = dist
            out = np.zeros(3)
            out[i], out[j] = 1.0 - t, t
    return out


def segment_distances(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Squared distances between pairs of segments ``p1 q1`` and ``p2 q2``, and the closest
    points on each (Ericson 5.1.9, including degenerate segments)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    c, b = _dot(d1, r), _dot(d1, d2)
    eps = 1e-300
    n = p1.shape[0]
    s = np.zeros(n)
    t = np.zeros(n)
    both = (a <= eps) & (e <= eps)
    first = (a <= eps) & ~both
    second = (e <= eps) & ~both & ~first
    general = ~(both | first | second)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(first, np.clip(f / e, 0.0, 1.0), t)
        s = np.where(second, np.clip(-c / a, 0.0, 1.0), s)
        denom = a * e - b * b
        sg = np.where(denom != 0.0, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0)
        tg = (b * sg + f) / e
        low = tg < 0.0
        high = tg > 1.0
        sg = np.where(low, np.clip(-c / a, 0.0, 1.0), sg)
        sg = np.where(high, np.clip((b - c) / a, 0.0, 1.0), sg)
        tg = np.clip(tg, 0.0, 1.0)
    s = np.where(general, sg, s)
    t = np.where(general, tg, t)
    s = np.nan_to_num(s)
    t = np.nan_to_num(t)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    diff = c1 - c2
    return _dot(diff, diff), c1, c2


def _project(v: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min and max of the projections of point sets ``v`` ``(n, k, 3)`` on ``axis`` ``(n, 3)``."""
    d = v[..., 0] * axis[:, None, 0] + v[..., 1] * axis[:, None, 1] + v[..., 2] * axis[:, None, 2]
    return d.min(axis=1), d.max(axis=1)


def _separated(u: np.ndarray, v: np.ndarray, axis: np.ndarray, slack: np.ndarray) -> np.ndarray:
    """Whether the projections of ``u`` and ``v`` on ``axis`` are apart by more than
    ``slack * |axis|`` (a zero axis separates nothing)."""
    lo1, hi1 = _project(u, axis)
    lo2, hi2 = _project(v, axis)
    gap = slack * np.sqrt(_dot(axis, axis))
    return (hi1 < lo2 - gap) | (hi2 < lo1 - gap)


TOUCH = 1e-12
"""Relative scale below which triangles count as touching (rounding of exact contacts)."""


def triangles_intersect(
    a1: np.ndarray,
    b1: np.ndarray,
    c1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    """Whether each pair of triangles shares a point (closed triangles: touching counts).

    A separating axis among the two normals and the nine edge cross products
    proves a pair disjoint. Every other pair is decided constructively: two
    triangles meet exactly when an edge of one crosses the other (their
    intersection is a segment whose ends lie on edges; decided by exact
    orientation predicates, :mod:`aether3d.spatial.predicates`) or, for touching
    pairs, when a vertex-triangle or edge-edge distance is zero -- within
    ``TOUCH`` times the coordinates' scale, the rounding of exact contacts.
    Degenerate triangles (segments, points) are handled the same way.
    """
    n = a1.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    u = np.stack([a1, b1, c1], axis=1)
    v = np.stack([a2, b2, c2], axis=1)
    scale = np.maximum(np.abs(u).max(axis=(1, 2)), np.abs(v).max(axis=(1, 2))) + 1.0
    slack = TOUCH * scale
    sep = _separated(u, v, np.cross(b1 - a1, c1 - a1), slack)
    sep |= _separated(u, v, np.cross(b2 - a2, c2 - a2), slack)
    e1 = (b1 - a1, c1 - b1, a1 - c1)
    e2 = (b2 - a2, c2 - b2, a2 - c2)
    for x in e1:
        for y in e2:
            sep |= _separated(u, v, np.cross(x, y), slack)
    out = np.zeros(n, dtype=bool)
    rows = np.flatnonzero(~sep)
    if rows.size:
        t1 = (a1[rows], b1[rows], c1[rows])
        t2 = (a2[rows], b2[rows], c2[rows])
        hit = np.zeros(rows.size, dtype=bool)
        # Exact orientation predicates: a float ray-triangle test misfires on nearly
        # parallel planes (its determinant is then all rounding).
        from .predicates import segment_crosses_triangle

        for (p, q), tri in (
            ((t1[0], t1[1]), t2),
            ((t1[1], t1[2]), t2),
            ((t1[2], t1[0]), t2),
            ((t2[0], t2[1]), t1),
            ((t2[1], t2[2]), t1),
            ((t2[2], t2[0]), t1),
        ):
            hit |= segment_crosses_triangle(p, q, *tri)
        rest = np.flatnonzero(~hit)
        if rest.size:
            d, _, _ = triangle_distances_raw(*(x[rest] for x in (*t1, *t2)))
            hit[rest] = d <= TOUCH * scale[rows][rest]
        out[rows] = hit
    return out


def point_triangle_distances(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    points, _ = closest_points_on_triangles(p, a, b, c)
    diff = points - p
    return _dot(diff, diff), points


def triangle_distances(
    a1: np.ndarray,
    b1: np.ndarray,
    c1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    c2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact distances between pairs of triangles and a closest point on each.

    Intersecting pairs (see :func:`triangles_intersect`) have distance 0 and a
    common point (:func:`contact_points`) as both closest points.
    """
    dist, p1, p2 = triangle_distances_raw(a1, b1, c1, a2, b2, c2)
    hit = triangles_intersect(a1, b1, c1, a2, b2, c2)
    if hit.any():
        dist[hit] = 0.0
        point = contact_points(a1[hit], b1[hit], c1[hit], a2[hit], b2[hit], c2[hit])
        p1[hit] = point
        p2[hit] = point
    return dist, p1, p2


def triangles_intersect_convex(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    vertices: np.ndarray,
    face_normals: np.ndarray,
    edge_directions: np.ndarray,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Whether each triangle meets a convex polyhedron (closed sets, within ``tolerance``).

    ``vertices`` ``(k, 3)``: the polyhedron's corners; ``face_normals`` ``(f, 3)``
    and ``edge_directions`` ``(e, 3)``: its distinct face normal and edge
    directions (a box: 3 and 3; a frustum: up to 6 and 6).
    """
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    tri = np.stack([a, b, c], axis=1)
    poly = np.broadcast_to(vertices, (n, *vertices.shape))
    tol = np.full(n, float(tolerance))
    sep = np.zeros(n, dtype=bool)
    for normal in face_normals:
        sep |= _separated(tri, poly, np.broadcast_to(normal, (n, 3)), tol)
    sep |= _separated(tri, poly, np.cross(b - a, c - a), tol)
    for edge in (b - a, c - b, a - c):
        for direction in edge_directions:
            sep |= _separated(tri, poly, np.cross(edge, direction), tol)
    return ~sep


BOX_NORMALS = np.eye(3)


def box_vertices(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def winding_numbers(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Generalized winding numbers of the triangles ``a b c`` at ``points`` ``(m, 3)``."""
    m = points.shape[0]
    out = np.zeros(m)
    if a.shape[0] == 0:
        return out
    step = max(1, CHUNK // max(1, m))
    for s in range(0, a.shape[0], step):
        sl = slice(s, s + step)
        ra = a[None, sl] - points[:, None]
        rb = b[None, sl] - points[:, None]
        rc = c[None, sl] - points[:, None]
        la = np.sqrt(_dot(ra, ra))
        lb = np.sqrt(_dot(rb, rb))
        lc = np.sqrt(_dot(rc, rc))
        num = _dot(ra, np.cross(rb, rc))
        den = la * lb * lc + _dot(ra, rb) * lc + _dot(rb, rc) * la + _dot(rc, ra) * lb
        out += np.arctan2(num, den).sum(axis=1)
    return out / (2.0 * np.pi)


def bvh_box_pairs(
    bvh, lo: np.ndarray, hi: np.ndarray, qlo: np.ndarray, qhi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(query, primitive)`` index pairs whose boxes overlap (closed boxes): the BVH's
    primitives have boxes ``lo``, ``hi``; the queries ``qlo``, ``qhi``."""
    nq = qlo.shape[0]
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if nq == 0 or bvh.order.size == 0:
        return empty
    root = np.all((qlo <= bvh.hi) & (qhi >= bvh.lo), axis=1)
    q = np.flatnonzero(root).astype(np.int64)
    node = np.zeros(q.size, dtype=np.int64)
    out_q: list[np.ndarray] = []
    out_p: list[np.ndarray] = []
    while q.size:
        leaf = bvh.child[node, 0] < 0
        if leaf.any():
            lq, ln = q[leaf], node[leaf]
            counts = bvh.count[ln]
            total = int(counts.sum())
            if total:
                starts = np.repeat(bvh.first[ln], counts)
                offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                out_q.append(np.repeat(lq, counts))
                out_p.append(bvh.order[starts + offsets])
        iq, inode = q[~leaf], node[~leaf]
        next_q, next_node = [], []
        for j in (0, 1):
            clo = bvh.child_lo[inode, j]
            chi = bvh.child_hi[inode, j]
            ok = np.all((qlo[iq] <= chi) & (qhi[iq] >= clo), axis=1)
            next_q.append(iq[ok])
            next_node.append(bvh.child[inode[ok], j])
        q = np.concatenate(next_q) if next_q else np.zeros(0, dtype=np.int64)
        node = np.concatenate(next_node) if next_node else np.zeros(0, dtype=np.int64)
    if not out_q:
        return empty
    q, p = np.concatenate(out_q), np.concatenate(out_p)
    ok = np.all((qlo[q] <= hi[p]) & (qhi[q] >= lo[p]), axis=1)
    return q[ok], p[ok]


def box_point_distances(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple:
    """Squared distances from point ``p`` to boxes ``lo``, ``hi`` (``(n, 3)``): the nearest
    and the farthest point of each box."""
    near = np.clip(p, lo, hi) - p
    far = np.maximum(np.abs(lo - p), np.abs(hi - p))
    return _dot(near, near), _dot(far, far)


def box_pair_distances(lo1, hi1, lo2, hi2) -> tuple:
    """Distances between pairs of boxes: the least, and the greatest between any two of their
    points (an upper bound of the distance between anything inside them)."""
    gap = np.maximum(np.maximum(lo1 - hi2, lo2 - hi1), 0.0)
    span = np.maximum(np.abs(hi1 - lo2), np.abs(hi2 - lo1))
    return np.sqrt(_dot(gap, gap)), np.sqrt(_dot(span, span))


def segment_triangle_points(p, q, a, b, c) -> tuple[np.ndarray, np.ndarray]:
    """Where segments ``p q`` cross triangles ``a b c`` (Moller-Trumbore on the segment's
    parameter in [0, 1], boundaries included); returns ``(crossing, point)``."""
    d = q - p
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = _dot(e1, h)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = 1.0 / det
        s = p - a
        u = f * _dot(s, h)
        qv = np.cross(s, e1)
        v = f * _dot(d, qv)
        t = f * _dot(e2, qv)
        ok = (det != 0.0) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)
    ok &= np.isfinite(t)
    point = p + d * np.where(ok, t, 0.0)[:, None]
    return ok, point


def contact_points(a1, b1, c1, a2, b2, c2) -> np.ndarray:
    """A common point of each pair of intersecting triangles: where an edge of one crosses
    the other, else (coplanar pairs) the nearest vertex/edge points (``(n, 3)``)."""
    n = a1.shape[0]
    out = np.zeros((n, 3))
    done = np.zeros(n, dtype=bool)
    for (p, q), tri in (
        ((a1, b1), (a2, b2, c2)),
        ((b1, c1), (a2, b2, c2)),
        ((c1, a1), (a2, b2, c2)),
        ((a2, b2), (a1, b1, c1)),
        ((b2, c2), (a1, b1, c1)),
        ((c2, a2), (a1, b1, c1)),
    ):
        ok, point = segment_triangle_points(p, q, *tri)
        new = ok & ~done
        out[new] = point[new]
        done |= new
    if not done.all():
        rest = np.flatnonzero(~done)
        _, p1, _ = triangle_distances_raw(
            a1[rest], b1[rest], c1[rest], a2[rest], b2[rest], c2[rest]
        )
        out[rest] = p1
    return out


def triangle_distances_raw(a1, b1, c1, a2, b2, c2):
    """Least vertex-triangle and edge-edge distance of triangle pairs and its points (the
    distance of disjoint triangles; for intersecting ones only an upper bound)."""
    n = a1.shape[0]
    best = np.full(n, np.inf)
    p1 = np.zeros((n, 3))
    p2 = np.zeros((n, 3))

    def keep(d2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> None:
        better = d2 < best
        best[better] = d2[better]
        p1[better] = q1[better]
        p2[better] = q2[better]

    for vtx in (a1, b1, c1):
        d2, q = point_triangle_distances(vtx, a2, b2, c2)
        keep(d2, vtx, q)
    for vtx in (a2, b2, c2):
        d2, q = point_triangle_distances(vtx, a1, b1, c1)
        keep(d2, q, vtx)
    for s1, e1 in ((a1, b1), (b1, c1), (c1, a1)):
        for s2, e2 in ((a2, b2), (b2, c2), (c2, a2)):
            d2, q1, q2 = segment_distances(s1, e1, s2, e2)
            keep(d2, q1, q2)
    return np.sqrt(best), p1, p2


def component_labels(triangles: np.ndarray, vertex_count: int) -> np.ndarray:
    """Connected-component label (``0 .. k - 1``, in order of each component's lowest
    vertex) of every vertex of a triangle mesh; unused vertices get ``-1``."""
    t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    labels = np.full(vertex_count, -1, dtype=np.int64)
    if not len(t):
        return labels
    u = np.concatenate([t[:, 0], t[:, 1]])
    v = np.concatenate([t[:, 1], t[:, 2]])
    parent = np.arange(vertex_count, dtype=np.int64)
    while True:
        pu, pv = parent[u], parent[v]
        differ = pu != pv
        if not differ.any():
            break
        np.minimum.at(
            parent, np.maximum(pu[differ], pv[differ]), np.minimum(pu[differ], pv[differ])
        )
        while True:  # pointer jumping: every vertex points at its root
            up = parent[parent]
            if np.array_equal(up, parent):
                break
            parent = up
    used = np.unique(t)
    roots, inverse = np.unique(parent[used], return_inverse=True)
    labels[used] = inverse.reshape(-1)
    return labels


def component_representatives(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """One vertex (the lowest index) of every connected component of a triangle mesh, with
    vertices at exactly the same position welded (as mesh inspection does)."""
    if triangles.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    used = np.unique(triangles)
    keys = np.ascontiguousarray(positions[used] + 0.0)
    _, weld = np.unique(
        keys.view(np.dtype((np.void, keys.dtype.itemsize * 3))), return_inverse=True
    )
    weld = weld.reshape(-1)
    welded = np.full(positions.shape[0], -1, dtype=np.int64)
    welded[used] = weld
    w = welded[triangles]
    u = np.concatenate([w[:, 0], w[:, 1]])
    v = np.concatenate([w[:, 1], w[:, 2]])
    parent = np.arange(int(weld.max()) + 1, dtype=np.int64)
    while True:
        pu, pv = parent[u], parent[v]
        differ = pu != pv
        if not differ.any():
            break
        lo = np.minimum(pu[differ], pv[differ])
        hi = np.maximum(pu[differ], pv[differ])
        np.minimum.at(parent, hi, lo)
        while True:  # pointer jumping: every vertex points at its root
            up = parent[parent]
            if np.array_equal(up, parent):
                break
            parent = up
    root = parent[welded[used]]
    _, first = np.unique(root, return_index=True)
    return np.sort(used[first])
