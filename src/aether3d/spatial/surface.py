"""Exact nearest-point and ray queries on one triangle mesh, for many points at once.

:class:`TriangleSurface` holds float64 corners and a linear (Morton) BVH
(:func:`~aether3d.rendering.raytrace.bvh.build_bvh_morton`). It serves the modeling and
sculpting tools that project points onto a surface (shrinkwrap, stroke projection,
remeshing); the scene-wide queries of :mod:`aether3d.spatial.queries` keep their own
index.

* :meth:`TriangleSurface.nearest` is exact: each point first descends the BVH greedily
  to the nearer child and takes the nearest triangle of that leaf as an upper bound on
  its distance; the BVH is then walked level by
  level, keeping only nodes whose box is no farther than the best distance found so far
  and measuring the triangles of the leaves reached
  (:func:`~aether3d.spatial.geometry.closest_points_on_triangles`). A node farther than
  the best distance cannot hold a nearer triangle, so the result is the nearest one
  (lowest id on ties).
* :meth:`TriangleSurface.raycast` returns the closest hit of each ray with
  ``tmin < t < tmax`` (Moeller-Trumbore in float64 on the candidate triangles whose boxes
  the ray's segment crosses), from either side of a triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rendering.raytrace.bvh import build_bvh_morton, morton_codes, triangle_bounds
from .geometry import closest_points_on_triangles

BATCH = 2048
"""Points or rays processed together (bounds temporary memory)."""


@dataclass
class Nearest:
    """Closest surface points: ``points`` ``(n, 3)``, ``triangles`` ``(n,)`` (``-1``: none
    within ``max_distance``), barycentric ``bary`` ``(n, 3)`` and ``distance`` ``(n,)``."""

    points: np.ndarray
    triangles: np.ndarray
    bary: np.ndarray
    distance: np.ndarray


@dataclass
class RayHit:
    """Closest hits: ``t`` ``(n,)`` (``inf``: none), ``triangles`` (``-1``: none),
    ``bary`` ``(n, 3)``, ``points`` ``(n, 3)`` and ``front`` (the ray meets the
    triangle's front side, against its winding normal)."""

    t: np.ndarray
    triangles: np.ndarray
    bary: np.ndarray
    points: np.ndarray
    front: np.ndarray


class TriangleSurface:
    def __init__(self, positions: np.ndarray, triangles: np.ndarray) -> None:
        self.positions = np.ascontiguousarray(positions, dtype=np.float64)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.a = self.positions[self.triangles[:, 0]]
        self.b = self.positions[self.triangles[:, 1]]
        self.c = self.positions[self.triangles[:, 2]]
        self.lo, self.hi = triangle_bounds(self.positions, self.triangles)
        self.bvh = build_bvh_morton(self.lo, self.hi)
        centroid = (self.a + self.b + self.c) / 3.0 if len(self.triangles) else np.zeros((0, 3))
        if len(centroid):
            self._box = (centroid.min(axis=0), centroid.max(axis=0))
            codes = morton_codes(centroid, *self._box)
            self._sorted = np.argsort(codes, kind="stable")
            self._codes = codes[self._sorted]

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    def nearest(
        self, points: np.ndarray, max_distance: float = np.inf, *, batch: int = BATCH
    ) -> Nearest:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n = len(p)
        out_t = np.full(n, -1, dtype=np.int64)
        out_p = np.full((n, 3), np.nan)
        out_b = np.zeros((n, 3))
        out_d = np.full(n, np.inf)
        if n == 0 or self.triangle_count == 0:
            return Nearest(out_p, out_t, out_b, out_d)
        bvh = self.bvh
        cap = 2 * bvh.depth + 4
        for s in range(0, n, batch):
            q = p[s : s + batch]
            m = len(q)
            best_d = np.full(m, float(max_distance))
            best_t = np.full(m, -1, dtype=np.int64)
            best_p = np.full((m, 3), np.nan)
            best_b = np.zeros((m, 3))
            # One traversal stack per point, run in lockstep: pop a node, skip it if its
            # box is farther than the point's best distance, measure a leaf's triangles,
            # or push an inner node's children (the nearer one on top).
            stack = np.zeros((m, cap), dtype=np.int64)
            bound = np.zeros((m, cap))
            top = np.zeros(m, dtype=np.int64)
            root = _box_distance(q, bvh.lo[None], bvh.hi[None])
            top[root <= best_d] = 1
            bound[:, 0] = root
            active = np.flatnonzero(top > 0)
            while active.size:
                top[active] -= 1
                node = stack[active, top[active]]
                lb = bound[active, top[active]]
                live = lb <= best_d[active]
                active, node = active[live], node[live]
                leaf = bvh.child[node, 0] < 0
                if leaf.any():
                    lp, ln = active[leaf], node[leaf]
                    counts = bvh.count[ln]
                    total = int(counts.sum())
                    starts = np.repeat(bvh.first[ln], counts)
                    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                    qi = np.repeat(lp, counts)
                    ti = bvh.order[starts + offsets]
                    cp, bary = closest_points_on_triangles(
                        q[qi], self.a[ti], self.b[ti], self.c[ti]
                    )
                    dist = np.sqrt(((cp - q[qi]) ** 2).sum(axis=1))
                    order = np.lexsort((ti, dist, qi))
                    qi, ti, dist = qi[order], ti[order], dist[order]
                    cp, bary = cp[order], bary[order]
                    firsts = np.flatnonzero(np.r_[True, qi[1:] != qi[:-1]])
                    rows = qi[firsts]
                    better = (dist[firsts] < best_d[rows]) | (
                        (dist[firsts] == best_d[rows]) & (ti[firsts] < best_t[rows])
                    )
                    sel = firsts[better]
                    rows = qi[sel]
                    best_d[rows], best_t[rows] = dist[sel], ti[sel]
                    best_p[rows], best_b[rows] = cp[sel], bary[sel]
                ip, inode = active[~leaf], node[~leaf]
                if ip.size:
                    d0 = _box_distance(q[ip], bvh.child_lo[inode, 0], bvh.child_hi[inode, 0])
                    d1 = _box_distance(q[ip], bvh.child_lo[inode, 1], bvh.child_hi[inode, 1])
                    near_first = d0 <= d1
                    far = np.where(near_first, bvh.child[inode, 1], bvh.child[inode, 0])
                    near = np.where(near_first, bvh.child[inode, 0], bvh.child[inode, 1])
                    far_d = np.where(near_first, d1, d0)
                    near_d = np.where(near_first, d0, d1)
                    for kid, dk in ((far, far_d), (near, near_d)):
                        ok = dk <= best_d[ip]
                        rows = ip[ok]
                        stack[rows, top[rows]] = kid[ok]
                        bound[rows, top[rows]] = dk[ok]
                        top[rows] += 1
                active = np.flatnonzero(top > 0)
            found = best_t >= 0
            rows = s + np.flatnonzero(found)
            out_t[rows] = best_t[found]
            out_p[rows] = best_p[found]
            out_b[rows] = best_b[found]
            out_d[rows] = best_d[found]
        return Nearest(out_p, out_t, out_b, out_d)

    def raycast(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        tmin: float = 0.0,
        tmax: float | np.ndarray = np.inf,
    ) -> RayHit:
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        d = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        n = len(o)
        t_out = np.full(n, np.inf)
        tri_out = np.full(n, -1, dtype=np.int64)
        bary_out = np.zeros((n, 3))
        front_out = np.zeros(n, dtype=bool)
        limit = np.broadcast_to(np.asarray(tmax, dtype=np.float64), (n,)).copy()
        bvh = self.bvh
        cap = 2 * bvh.depth + 4
        for s in range(0, n if self.triangle_count else 0, BATCH):
            ro, rd = o[s : s + BATCH], d[s : s + BATCH]
            m = len(ro)
            with np.errstate(divide="ignore"):
                inv = 1.0 / rd
            best = limit[s : s + BATCH].copy()
            stack = np.zeros((m, cap), dtype=np.int64)
            entry = np.zeros((m, cap))
            top = np.zeros(m, dtype=np.int64)
            t_root = _slab(ro, inv, bvh.lo[None], bvh.hi[None], tmin, best)
            top[np.isfinite(t_root)] = 1
            entry[:, 0] = t_root
            active = np.flatnonzero(top > 0)
            while active.size:
                top[active] -= 1
                node = stack[active, top[active]]
                te = entry[active, top[active]]
                live = te <= best[active]
                active, node = active[live], node[live]
                leaf = bvh.child[node, 0] < 0
                if leaf.any():
                    lr, ln = active[leaf], node[leaf]
                    counts = bvh.count[ln]
                    total = int(counts.sum())
                    starts = np.repeat(bvh.first[ln], counts)
                    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                    r = np.repeat(lr, counts)
                    tri = bvh.order[starts + offsets]
                    t, u, v, front = _moller_trumbore(
                        ro[r], rd[r], self.a[tri], self.b[tri], self.c[tri]
                    )
                    good = (t > tmin) & (t <= best[r])
                    if good.any():
                        r, tri, t = r[good], tri[good], t[good]
                        u, v, front = u[good], v[good], front[good]
                        order = np.lexsort((tri, t, r))
                        r, tri, t = r[order], tri[order], t[order]
                        u, v, front = u[order], v[order], front[order]
                        firsts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
                        rr = r[firsts]
                        g = s + rr
                        better = (t[firsts] < best[rr]) | (
                            (t[firsts] == best[rr]) & (tri[firsts] < tri_out[g])
                        )
                        sel = firsts[better]
                        rr, g = r[sel], s + r[sel]
                        best[rr] = t[sel]
                        t_out[g] = t[sel]
                        tri_out[g] = tri[sel]
                        bary_out[g] = np.stack([1 - u[sel] - v[sel], u[sel], v[sel]], axis=1)
                        front_out[g] = front[sel]
                ir, inode = active[~leaf], node[~leaf]
                if ir.size:
                    t0 = _slab(
                        ro[ir],
                        inv[ir],
                        bvh.child_lo[inode, 0],
                        bvh.child_hi[inode, 0],
                        tmin,
                        best[ir],
                    )
                    t1 = _slab(
                        ro[ir],
                        inv[ir],
                        bvh.child_lo[inode, 1],
                        bvh.child_hi[inode, 1],
                        tmin,
                        best[ir],
                    )
                    near_first = t0 <= t1
                    far = np.where(near_first, bvh.child[inode, 1], bvh.child[inode, 0])
                    near = np.where(near_first, bvh.child[inode, 0], bvh.child[inode, 1])
                    far_t = np.where(near_first, t1, t0)
                    near_t = np.where(near_first, t0, t1)
                    for kid, tk in ((far, far_t), (near, near_t)):
                        ok = np.isfinite(tk)
                        rows = ir[ok]
                        stack[rows, top[rows]] = kid[ok]
                        entry[rows, top[rows]] = tk[ok]
                        top[rows] += 1
                active = np.flatnonzero(top > 0)
        points = o + np.where(np.isfinite(t_out), t_out, 0.0)[:, None] * d
        points[tri_out < 0] = np.nan
        return RayHit(t_out, tri_out, bary_out, points, front_out)


def _box_distance(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance from points to axis-aligned boxes (0 inside)."""
    d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return np.sqrt((d * d).sum(axis=1))


def _slab(o, inv, lo, hi, tmin, tmax) -> np.ndarray:
    """Entry parameter of each ray into its box within ``[tmin, tmax]`` (inf: misses)."""
    with np.errstate(invalid="ignore"):
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
    # A zero direction component inside the slab gives nan (0 * inf): no constraint.
    near = np.nanmax(np.where(np.isnan(t0), -np.inf, np.minimum(t0, t1)), axis=1)
    far = np.nanmin(np.where(np.isnan(t0), np.inf, np.maximum(t0, t1)), axis=1)
    near = np.maximum(near, tmin)
    far = np.minimum(far, tmax)
    hit = near <= far * (1 + 1e-12) + 1e-300
    return np.where(hit, near, np.inf)


def _moller_trumbore(o, d, a, b, c):
    """Ray parameters ``t`` (inf on a miss) and barycentric ``u``, ``v`` (of ``b``, ``c``)."""
    e1, e2 = b - a, c - a
    pvec = np.cross(d, e2)
    det = (e1 * pvec).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tvec = o - a
        u = (tvec * pvec).sum(axis=1) * inv
        qvec = np.cross(tvec, e1)
        v = (d * qvec).sum(axis=1) * inv
        t = (e2 * qvec).sum(axis=1) * inv
        miss = (det == 0) | ~np.isfinite(t) | (u < 0) | (v < 0) | (u + v > 1)
    t = np.where(miss, np.inf, t)
    # det = -d . (e1 x e2): the ray meets the front (travels against the winding normal)
    # exactly when det > 0.
    front = det > 0
    return t, u, v, front
