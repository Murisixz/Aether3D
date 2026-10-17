"""Constrained Delaunay refinement of 2D triangulations (for curved-face tessellation).

Start from any triangulation of a polygon with holes (``triangulate.py``); Lawson edge
flips that never flip a boundary edge turn it into the constrained Delaunay
triangulation; points inserted inside triangles (split into three, then flipped) refine
it. Distances are measured in a scaled metric (``sx``, ``sy``) so that a surface's
parameter space can be refined as if it were isotropic in space.

Predicates are float64 with a relative guard: a flip is only made when the opposite
point is clearly inside the circumcircle, so rounding can never make two flips undo each
other (the loop always terminates; worst case a triangle stays non-Delaunay).
"""

from __future__ import annotations

import numpy as np


class TriMesh2D:
    def __init__(self, points: np.ndarray, triangles: np.ndarray, sx: float = 1.0, sy: float = 1.0):
        self.p: list[list[float]] = [[float(x) * sx, float(y) * sy] for x, y in points]
        self.sx, self.sy = sx, sy
        self.tri: list[list[int]] = [list(map(int, t)) for t in triangles]
        self.alive: list[bool] = [True] * len(self.tri)
        self.edge_map: dict[tuple[int, int], int] = {}  # directed edge -> triangle
        for t, (a, b, c) in enumerate(self.tri):
            self.edge_map[(a, b)] = t
            self.edge_map[(b, c)] = t
            self.edge_map[(c, a)] = t
        # Boundary (constrained) edges: those without an opposite directed edge.
        self.fixed: set[tuple[int, int]] = set()
        for a, b in self.edge_map:
            if (b, a) not in self.edge_map:
                self.fixed.add((min(a, b), max(a, b)))

    # -- predicates -----------------------------------------------------------------

    def orient(self, a: int, b: int, c: int) -> float:
        pa, pb, pc = self.p[a], self.p[b], self.p[c]
        return (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])

    def incircle(self, a: int, b: int, c: int, d: int) -> float:
        pa, pb, pc, pd = self.p[a], self.p[b], self.p[c], self.p[d]
        adx, ady = pa[0] - pd[0], pa[1] - pd[1]
        bdx, bdy = pb[0] - pd[0], pb[1] - pd[1]
        cdx, cdy = pc[0] - pd[0], pc[1] - pd[1]
        ad = adx * adx + ady * ady
        bd = bdx * bdx + bdy * bdy
        cd = cdx * cdx + cdy * cdy
        det = (
            adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx)
        )
        perm = (
            (abs(bdx * cdy) + abs(bdy * cdx)) * ad
            + (abs(adx * cdy) + abs(ady * cdx)) * bd
            + (abs(adx * bdy) + abs(ady * bdx)) * cd
        )
        return det if abs(det) > 1e-9 * perm else 0.0

    # -- flips ------------------------------------------------------------------------

    def _set(self, t: int, a: int, b: int, c: int) -> None:
        self.tri[t] = [a, b, c]
        self.edge_map[(a, b)] = t
        self.edge_map[(b, c)] = t
        self.edge_map[(c, a)] = t

    def legalize(self, edges: list[tuple[int, int]], limit: int = 1_000_000) -> None:
        stack = list(edges)
        count = 0
        while stack and count < limit:
            count += 1
            a, b = stack.pop()
            if (min(a, b), max(a, b)) in self.fixed:
                continue
            t1 = self.edge_map.get((a, b))
            t2 = self.edge_map.get((b, a))
            if t1 is None or t2 is None or not self.alive[t1] or not self.alive[t2]:
                continue
            c = next(v for v in self.tri[t1] if v != a and v != b)
            d = next(v for v in self.tri[t2] if v != a and v != b)
            # t1 = (a, b, c) ccw; t2 = (b, a, d) ccw. Flip if d is inside circle(a, b, c)
            # and the quad a-d-b-c is convex.
            if self.incircle(a, b, c, d) <= 0:
                continue
            if self.orient(c, a, d) <= 0 or self.orient(d, b, c) <= 0:
                continue
            for e in ((a, b), (b, a)):
                self.edge_map.pop(e, None)
            self._set(t1, c, a, d)
            self._set(t2, d, b, c)
            stack += [(a, d), (d, b), (b, c), (c, a)]

    def make_delaunay(self) -> None:
        self.legalize([e for e in self.edge_map if e[0] < e[1]])

    # -- insertion ----------------------------------------------------------------------

    def locate(self, x: float, y: float, start: int = 0) -> int:
        """A triangle containing (x, y) (scaled coordinates), by walking; -1 if outside."""
        t = start if 0 <= start < len(self.tri) and self.alive[start] else self._any()
        for _ in range(4 * len(self.tri) + 8):
            a, b, c = self.tri[t]
            moved = False
            for u, v in ((a, b), (b, c), (c, a)):
                pu, pv = self.p[u], self.p[v]
                if (pv[0] - pu[0]) * (y - pu[1]) - (pv[1] - pu[1]) * (x - pu[0]) < 0:
                    nt = self.edge_map.get((v, u))
                    if nt is None:
                        return -1
                    t = nt
                    moved = True
                    break
            if not moved:
                return t
        return -1

    def _any(self) -> int:
        for t, ok in enumerate(self.alive):
            if ok:
                return t
        return -1

    def insert(self, x: float, y: float, start: int = 0) -> int:
        """Insert a point given in *unscaled* coordinates; returns its index or -1."""
        sx, sy = x * self.sx, y * self.sy
        t = self.locate(sx, sy, start)
        if t < 0:
            return -1
        a, b, c = self.tri[t]
        self.p.append([sx, sy])
        n = len(self.p) - 1
        # Refuse points (nearly) on an edge: they would make degenerate triangles.
        if self.orient(a, b, n) <= 0 or self.orient(b, c, n) <= 0 or self.orient(c, a, n) <= 0:
            self.p.pop()
            return -1
        for e in ((a, b), (b, c), (c, a)):
            self.edge_map.pop(e, None)
        self._set(t, a, b, n)
        self.tri.append([0, 0, 0])
        self.alive.append(True)
        t2 = len(self.tri) - 1
        self._set(t2, b, c, n)
        self.tri.append([0, 0, 0])
        self.alive.append(True)
        t3 = len(self.tri) - 1
        self._set(t3, c, a, n)
        self.legalize([(a, b), (b, c), (c, a)])
        return n

    def points(self) -> np.ndarray:
        arr = np.array(self.p, dtype=np.float64)
        arr[:, 0] /= self.sx
        arr[:, 1] /= self.sy
        return arr

    def triangles(self) -> np.ndarray:
        return np.array(
            [t for t, ok in zip(self.tri, self.alive, strict=True) if ok], dtype=np.int64
        )
