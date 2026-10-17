"""Triangulation of planar polygons with holes, many polygons per call.

Ear clipping on doubly linked vertex rings (Eberly, "Triangulation by Ear Clipping",
2002; the hole bridging, z-order acceleration and recovery passes popularized by the
Mapbox *earcut* library, reimplemented here from the algorithm's description):

1. The outer ring is made counter-clockwise and holes clockwise.
2. Holes are joined to the outer ring in order of their leftmost vertex: a ray from that
   vertex towards -x finds the nearest edge; the visible vertex of minimal angle inside
   the triangle it defines becomes the bridge (two duplicated vertices).
3. Ears are clipped: a convex vertex whose triangle contains no reflex vertex. Rings with
   more than 80 vertices index vertices on a z-order curve so that an ear test visits
   only vertices inside the ear's bounding box.
4. When no ear is found: consecutive duplicate points are removed, then local
   self-intersections cured, then the ring is split along a valid diagonal and each half
   triangulated. Collinear vertices are never dropped (they are shared with neighbouring
   faces in watertight meshes; dropping one would leave a T-junction) unless every
   other recovery failed.

Arithmetic is plain float64 ``+ - *`` and comparisons in a fixed order, so the native
kernel (``aether3d_native.triangulate``) produces exactly the same triangles
(``tests/test_native.py``). Output triangles are counter-clockwise.

``triangulate(points, ring_starts, polygon_starts)``: ``points`` (n, 2); rings are
``points[ring_starts[k]:ring_starts[k + 1]]``; polygon ``j`` owns rings
``polygon_starts[j]..polygon_starts[j + 1] - 1``, the first being its outer ring. Returns
``(triangles (t, 3) int64 indices into points, polygon of each triangle (t,) int64)``.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from ..errors import CadError

Z_ORDER_THRESHOLD = 80
MAX_POINTS = 1 << 24


def triangulate(
    points: np.ndarray, ring_starts: np.ndarray, polygon_starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    rings = np.ascontiguousarray(ring_starts, dtype=np.int64)
    polys = np.ascontiguousarray(polygon_starts, dtype=np.int64)
    _check(pts, rings, polys)
    native = _native.module()
    if native is not None:
        cap = len(pts) + 2 * (len(rings) - 1)  # every triangle removes a vertex or bridge
        tris = np.empty((cap, 3), dtype=np.int64)
        owner = np.empty(cap, dtype=np.int64)
        count = native.triangulate(pts, rings, polys, tris, owner)
        return tris[:count].copy(), owner[:count].copy()
    return triangulate_numpy(pts, rings, polys)


def _check(pts: np.ndarray, rings: np.ndarray, polys: np.ndarray) -> None:
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise CadError("INVALID_POLYGON", "points must be an (n, 2) array")
    if len(pts) > MAX_POINTS:
        raise CadError("LIMIT_EXCEEDED", f"more than {MAX_POINTS} polygon points")
    if not np.all(np.isfinite(pts)):
        raise CadError("INVALID_POLYGON", "polygon points must be finite")
    if rings.ndim != 1 or len(rings) < 1 or rings[0] != 0 or rings[-1] != len(pts):
        raise CadError("INVALID_POLYGON", "ring_starts must run from 0 to len(points)")
    if np.any(np.diff(rings) < 0):
        raise CadError("INVALID_POLYGON", "ring_starts must not decrease")
    nrings = len(rings) - 1
    if polys.ndim != 1 or len(polys) < 1 or polys[0] != 0 or polys[-1] != nrings:
        raise CadError("INVALID_POLYGON", "polygon_starts must run from 0 to the ring count")
    if np.any(np.diff(polys) < 1):
        raise CadError("INVALID_POLYGON", "every polygon needs an outer ring")


def triangulate_numpy(
    pts: np.ndarray, rings: np.ndarray, polys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The reference implementation (the native kernel's twin)."""
    xs = pts[:, 0].tolist()
    ys = pts[:, 1].tolist()
    out: list[int] = []
    owner: list[int] = []
    rl = rings.tolist()
    pl = polys.tolist()
    for j in range(len(pl) - 1):
        r0, r1 = pl[j], pl[j + 1]
        e = _Earcut(xs, ys)
        e.run([(rl[k], rl[k + 1]) for k in range(r0, r1)])
        out += e.triangles
        owner += [j] * (len(e.triangles) // 3)
    return np.array(out, dtype=np.int64).reshape(-1, 3), np.array(owner, dtype=np.int64)


class _Earcut:
    """One polygon. Nodes live in parallel lists (index = node id)."""

    def __init__(self, xs: list[float], ys: list[float]) -> None:
        self.X = xs
        self.Y = ys
        self.i: list[int] = []
        self.x: list[float] = []
        self.y: list[float] = []
        self.prev: list[int] = []
        self.next: list[int] = []
        self.z: list[int] = []
        self.pz: list[int] = []
        self.nz: list[int] = []
        self.triangles: list[int] = []
        self.min_x = self.min_y = 0.0
        self.inv_size = 0.0

    # -- nodes ----------------------------------------------------------------------

    def node(self, vi: int) -> int:
        n = len(self.i)
        self.i.append(vi)
        self.x.append(self.X[vi])
        self.y.append(self.Y[vi])
        self.prev.append(n)
        self.next.append(n)
        self.z.append(0)
        self.pz.append(-1)
        self.nz.append(-1)
        return n

    def insert_after(self, vi: int, last: int) -> int:
        n = self.node(vi)
        if last >= 0:
            nx = self.next[last]
            self.next[n] = nx
            self.prev[n] = last
            self.prev[nx] = n
            self.next[last] = n
        return n

    def remove(self, p: int) -> None:
        nx, pv = self.next[p], self.prev[p]
        self.prev[nx] = pv
        self.next[pv] = nx
        if self.pz[p] >= 0:
            self.nz[self.pz[p]] = self.nz[p]
        if self.nz[p] >= 0:
            self.pz[self.nz[p]] = self.pz[p]

    def ring(self, start: int, end: int, ccw: bool) -> int:
        """A ring as a linked list in the requested orientation; returns a node or -1."""
        area = 0.0
        j = end - 1
        for k in range(start, end):
            area += (self.X[j] - self.X[k]) * (self.Y[k] + self.Y[j])
            j = k
        # area > 0 here means the ring is counter-clockwise (twice its area, y up).
        last = -1
        if ccw == (area > 0):
            for k in range(start, end):
                last = self.insert_after(k, last)
        else:
            for k in range(end - 1, start - 1, -1):
                last = self.insert_after(k, last)
        if last >= 0 and self.equals(last, self.next[last]):
            self.remove(last)
            last = self.next[last]
        return last

    # -- predicates ---------------------------------------------------------------------

    def cross(self, a: int, b: int, c: int) -> float:
        x, y = self.x, self.y
        return (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])

    def equals(self, a: int, b: int) -> bool:
        return self.x[a] == self.x[b] and self.y[a] == self.y[b]

    def in_triangle(self, ax, ay, bx, by, cx, cy, px, py) -> bool:
        return (
            (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0
            and (cx - bx) * (py - by) - (cy - by) * (px - bx) >= 0
            and (ax - cx) * (py - cy) - (ay - cy) * (px - cx) >= 0
        )

    def locally_inside(self, a: int, b: int) -> bool:
        """Whether the diagonal a -> b starts into the polygon's interior at a."""
        an, ap = self.next[a], self.prev[a]
        if self.cross(ap, a, an) > 0:  # convex
            return self.cross(a, an, b) >= 0 and self.cross(a, b, ap) >= 0
        return self.cross(a, an, b) > 0 or self.cross(a, b, ap) > 0

    def middle_inside(self, a: int, b: int) -> bool:
        p = a
        inside = False
        px = (self.x[a] + self.x[b]) / 2
        py = (self.y[a] + self.y[b]) / 2
        while True:
            q = self.next[p]
            if (self.y[p] > py) != (self.y[q] > py) and self.y[q] != self.y[p]:
                xc = (self.x[q] - self.x[p]) * (py - self.y[p]) / (self.y[q] - self.y[p]) + self.x[
                    p
                ]
                if px < xc:
                    inside = not inside
            p = q
            if p == a:
                return inside

    def on_segment(self, p: int, q: int, r: int) -> bool:
        x, y = self.x, self.y
        return min(x[p], x[r]) <= x[q] <= max(x[p], x[r]) and min(y[p], y[r]) <= y[q] <= max(
            y[p], y[r]
        )

    def intersects(self, p1: int, q1: int, p2: int, q2: int) -> bool:
        o1 = _sign(self.cross(p1, q1, p2))
        o2 = _sign(self.cross(p1, q1, q2))
        o3 = _sign(self.cross(p2, q2, p1))
        o4 = _sign(self.cross(p2, q2, q1))
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        return o4 == 0 and self.on_segment(p2, q1, q2)

    def intersects_polygon(self, a: int, b: int) -> bool:
        p = a
        ia, ib = self.i[a], self.i[b]
        while True:
            q = self.next[p]
            if (
                self.i[p] != ia
                and self.i[q] != ia
                and self.i[p] != ib
                and self.i[q] != ib
                and self.intersects(p, q, a, b)
            ):
                return True
            p = q
            if p == a:
                return False

    def valid_diagonal(self, a: int, b: int) -> bool:
        return (
            self.i[self.next[a]] != self.i[b]
            and self.i[self.prev[a]] != self.i[b]
            and not self.intersects_polygon(a, b)
            and (
                (
                    self.locally_inside(a, b)
                    and self.locally_inside(b, a)
                    and self.middle_inside(a, b)
                    and (
                        self.cross(self.prev[a], a, self.prev[b]) != 0
                        or self.cross(a, self.prev[b], b) != 0
                    )
                )
                or (
                    self.equals(a, b)
                    and self.cross(self.prev[a], a, self.next[a]) < 0
                    and self.cross(self.prev[b], b, self.next[b]) < 0
                )
            )
        )

    # -- ears -------------------------------------------------------------------------

    def is_ear(self, ear: int) -> bool:
        a, b, c = self.prev[ear], ear, self.next[ear]
        if self.cross(a, b, c) <= 0:
            return False
        x, y = self.x, self.y
        ax, ay, bx, by, cx, cy = x[a], y[a], x[b], y[b], x[c], y[c]
        p = self.next[c]
        while p != a:
            if (
                not (x[p] == ax and y[p] == ay)
                and not (x[p] == cx and y[p] == cy)
                and not (x[p] == bx and y[p] == by)
                and self.in_triangle(ax, ay, bx, by, cx, cy, x[p], y[p])
                and self.cross(self.prev[p], p, self.next[p]) <= 0
            ):
                return False
            p = self.next[p]
        return True

    def is_ear_hashed(self, ear: int) -> bool:
        a, b, c = self.prev[ear], ear, self.next[ear]
        if self.cross(a, b, c) <= 0:
            return False
        x, y = self.x, self.y
        ax, ay, bx, by, cx, cy = x[a], y[a], x[b], y[b], x[c], y[c]
        x0 = min(ax, bx, cx)
        y0 = min(ay, by, cy)
        x1 = max(ax, bx, cx)
        y1 = max(ay, by, cy)
        min_z = self.z_order(x0, y0)
        max_z = self.z_order(x1, y1)

        def blocks(p: int) -> bool:
            return (
                x0 <= x[p] <= x1
                and y0 <= y[p] <= y1
                and p != a
                and p != c
                and not (x[p] == ax and y[p] == ay)
                and not (x[p] == cx and y[p] == cy)
                and not (x[p] == bx and y[p] == by)
                and self.in_triangle(ax, ay, bx, by, cx, cy, x[p], y[p])
                and self.cross(self.prev[p], p, self.next[p]) <= 0
            )

        p = self.pz[ear]
        n = self.nz[ear]
        while p >= 0 and self.z[p] >= min_z and n >= 0 and self.z[n] <= max_z:
            if blocks(p) or blocks(n):
                return False
            p = self.pz[p]
            n = self.nz[n]
        while p >= 0 and self.z[p] >= min_z:
            if blocks(p):
                return False
            p = self.pz[p]
        while n >= 0 and self.z[n] <= max_z:
            if blocks(n):
                return False
            n = self.nz[n]
        return True

    def z_order(self, px: float, py: float) -> int:
        xi = int((px - self.min_x) * self.inv_size)
        yi = int((py - self.min_y) * self.inv_size)
        xi = (xi | (xi << 8)) & 0x00FF00FF
        xi = (xi | (xi << 4)) & 0x0F0F0F0F
        xi = (xi | (xi << 2)) & 0x33333333
        xi = (xi | (xi << 1)) & 0x55555555
        yi = (yi | (yi << 8)) & 0x00FF00FF
        yi = (yi | (yi << 4)) & 0x0F0F0F0F
        yi = (yi | (yi << 2)) & 0x33333333
        yi = (yi | (yi << 1)) & 0x55555555
        return xi | (yi << 1)

    def index_curve(self, start: int) -> None:
        nodes = []
        p = start
        while True:
            if self.z[p] == 0:
                self.z[p] = self.z_order(self.x[p], self.y[p])
            nodes.append(p)
            p = self.next[p]
            if p == start:
                break
        nodes.sort(key=lambda k: self.z[k])  # stable, like the native kernel's sort
        for k, nd in enumerate(nodes):
            self.pz[nd] = nodes[k - 1] if k > 0 else -1
            self.nz[nd] = nodes[k + 1] if k + 1 < len(nodes) else -1

    # -- the algorithm ----------------------------------------------------------------

    def filter_duplicates(self, start: int, end: int = -1) -> int:
        """Remove vertices equal to their successor (``filterPoints`` without the
        collinear test: collinear vertices are shared with neighbouring faces)."""
        return self._filter(start, end, collinear=False)

    def filter_collinear(self, start: int) -> int:
        """Last resort: also drop vertices on the segment between their neighbours."""
        return self._filter(start, -1, collinear=True)

    def _filter(self, start: int, end: int, collinear: bool) -> int:
        if start < 0:
            return start
        if end < 0:
            end = start
        p = start
        while True:
            again = False
            nx = self.next[p]
            if self.equals(p, nx) or (collinear and self.cross(self.prev[p], p, nx) == 0):
                self.remove(p)
                p = end = self.prev[p]
                if p == self.next[p]:
                    break
                again = True
            else:
                p = nx
            if not again and p == end:
                break
        return end

    def cure_local_intersections(self, start: int) -> int:
        p = start
        while True:
            a = self.prev[p]
            b = self.next[self.next[p]]
            if (
                not self.equals(a, b)
                and self.intersects(a, p, self.next[p], b)
                and self.locally_inside(a, b)
                and self.locally_inside(b, a)
            ):
                self.triangles += [self.i[a], self.i[p], self.i[b]]
                self.remove(self.next[p])
                self.remove(p)
                p = start = b
            p = self.next[p]
            if p == start:
                break
        return self.filter_duplicates(p)

    def split_polygon(self, a: int, b: int) -> int:
        a2 = self.node(self.i[a])
        b2 = self.node(self.i[b])
        an, bp = self.next[a], self.prev[b]
        self.next[a] = b
        self.prev[b] = a
        self.next[a2] = an
        self.prev[an] = a2
        self.next[b2] = a2
        self.prev[a2] = b2
        self.next[bp] = b2
        self.prev[b2] = bp
        return b2

    def split_earcut(self, start: int, hashed: bool) -> None:
        a = start
        while True:
            b = self.next[self.next[a]]
            while b != self.prev[a]:
                if self.i[a] != self.i[b] and self.valid_diagonal(a, b):
                    c = self.split_polygon(a, b)
                    a = self.filter_duplicates(a, self.next[a])
                    c = self.filter_duplicates(c, self.next[c])
                    self.earcut_linked(a, hashed, 0)
                    self.earcut_linked(c, hashed, 0)
                    return
                b = self.next[b]
            a = self.next[a]
            if a == start:
                break
        # No valid diagonal: drop collinear points and clip what remains.
        self.earcut_linked(self.filter_collinear(start), hashed, 3)

    def earcut_linked(self, ear: int, hashed: bool, stage: int) -> None:
        if ear < 0:
            return
        if stage == 0 and hashed:
            self.index_curve(ear)
        stop = ear
        while self.prev[ear] != self.next[ear]:
            prv, nxt = self.prev[ear], self.next[ear]
            if self.is_ear_hashed(ear) if hashed else self.is_ear(ear):
                self.triangles += [self.i[prv], self.i[ear], self.i[nxt]]
                self.remove(ear)
                ear = self.next[nxt]
                stop = ear
                continue
            ear = nxt
            if ear == stop:
                if stage == 0:
                    self.earcut_linked(self.filter_duplicates(ear), hashed, 1)
                elif stage == 1:
                    ear = self.cure_local_intersections(self.filter_duplicates(ear))
                    self.earcut_linked(ear, hashed, 2)
                elif stage == 2:
                    self.split_earcut(ear, hashed)
                elif stage == 3:
                    # Degenerate remainder (zero area): nothing sensible to clip.
                    pass
                break

    def leftmost(self, start: int) -> int:
        p = start
        best = start
        while True:
            if self.x[p] < self.x[best] or (self.x[p] == self.x[best] and self.y[p] < self.y[best]):
                best = p
            p = self.next[p]
            if p == start:
                return best

    def find_bridge(self, hole: int, outer: int) -> int:
        """A vertex of the outer ring visible from the hole's leftmost vertex ``hole``."""
        p = outer
        hx, hy = self.x[hole], self.y[hole]
        qx = -float("inf")
        m = -1
        while True:
            q = self.next[p]
            if hy <= self.y[p] and hy >= self.y[q] and self.y[q] != self.y[p]:
                x = self.x[p] + (hy - self.y[p]) * (self.x[q] - self.x[p]) / (self.y[q] - self.y[p])
                if x <= hx and x > qx:
                    qx = x
                    m = p if self.x[p] < self.x[q] else q
                    if x == hx:
                        return m
            p = q
            if p == outer:
                break
        if m < 0:
            return -1
        stop = m
        mx, my = self.x[m], self.y[m]
        tan_min = float("inf")
        p = m
        while True:
            px, py = self.x[p], self.y[p]
            if hx >= px >= mx and hx != px:
                inside = (
                    self.in_triangle(hx, hy, mx, my, qx, hy, px, py)
                    if hy < my
                    else self.in_triangle(qx, hy, mx, my, hx, hy, px, py)
                )
                if inside:
                    tan = abs(hy - py) / (hx - px)
                    if self.locally_inside(p, hole) and (
                        tan < tan_min
                        or (
                            tan == tan_min
                            and (px > self.x[m] or (px == self.x[m] and self._sector(m, p)))
                        )
                    ):
                        m = p
                        tan_min = tan
            p = self.next[p]
            if p == stop:
                return m

    def _sector(self, m: int, p: int) -> bool:
        return (
            self.cross(self.prev[m], m, self.prev[p]) > 0
            and self.cross(self.next[p], m, self.next[m]) > 0
        )

    def run(self, rings: list[tuple[int, int]]) -> None:
        r0, r1 = rings[0]
        if r1 - r0 < 3:
            return
        outer = self.ring(r0, r1, True)
        if outer < 0 or self.next[outer] == self.prev[outer]:
            return
        holes = []
        for s, e in rings[1:]:
            if e - s < 3:
                continue
            h = self.ring(s, e, False)
            if h >= 0 and self.next[h] != h:
                holes.append(self.leftmost(h))
        holes.sort(key=lambda h: (self.x[h], self.y[h]))
        for h in holes:
            bridge = self.find_bridge(h, outer)
            if bridge < 0:
                continue
            back = self.split_polygon(bridge, h)
            self.filter_duplicates(back, self.next[back])
            outer = self.filter_duplicates(bridge, self.next[bridge])
        count = 0
        p = outer
        min_x = max_x = self.x[outer]
        min_y = max_y = self.y[outer]
        while True:
            count += 1
            min_x = min(min_x, self.x[p])
            max_x = max(max_x, self.x[p])
            min_y = min(min_y, self.y[p])
            max_y = max(max_y, self.y[p])
            p = self.next[p]
            if p == outer:
                break
        hashed = count > Z_ORDER_THRESHOLD
        if hashed:
            self.min_x, self.min_y = min_x, min_y
            size = max(max_x - min_x, max_y - min_y)
            self.inv_size = 32767.0 / size if size != 0 else 0.0
        self.earcut_linked(outer, hashed, 0)


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)
