"""Edge slide and vertex slide: move vertices along the edges next to them.

Edge slide moves the vertices of chains of edges (loops or open chains) along their
*rails*: for each vertex, the edge leaving the chain on the chosen side. The side of a
chain is defined by its faces: walking the chain from its lowest-id end (or, for a closed
loop, from its lowest-id vertex toward its lower-id neighbour), side ``+1`` holds the
faces that run the chain's direction, ``-1`` the others. ``factor`` in ``[-1, 1]`` moves
each vertex that fraction of the way along its rail on side ``sign(factor)``; ``even``
moves every vertex the same distance, ``|factor|`` times the shortest rail.

When several edges leave a vertex on one side (a pole), the rail is the one most nearly
perpendicular to the chain there (lowest edge id on ties). A vertex with no edge on that
side (a chain running along the side's boundary, or a triangle corner) cannot slide:
``SLIDE_NO_RAIL``.

UVs follow the vertex (``correct_uvs``): in faces containing the rail, the corner is
interpolated along the rail edge, which is exact; in other faces it takes the same value
when the vertex has one UV there (no seam), and is otherwise left as it was and reported
(``UV_SEAM_STRETCHED``). Normals of the faces involved are recomputed at commit.
"""

from __future__ import annotations

import math

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import EditError, ValidationError
from . import _polygon as poly
from .core import UV, Elements, operation


class SlideMixin:
    @operation
    def slide_edges(
        self, edges: Elements, factor: float, *, even: bool = False, correct_uvs: bool = True
    ):
        """Slide chains of edges along their rails (see the module docs).

        ``data["vertices"]`` maps each moved vertex to the far end of its rail.
        """
        eids = self._elements(edges, "edges")
        f = float(factor)
        if not (-1.0 <= f <= 1.0) or math.isnan(f):
            raise ValidationError("factor must be between -1 and 1")
        if f == 0.0:
            return {"vertices": {}}
        side = 1 if f > 0 else -1
        rails: dict[int, int] = {}  # vertex -> rail's far end
        for chain in self._edge_chains(eids):
            for v, prev, nxt, direction in chain:
                rails[v] = self._rail(v, prev, nxt, direction, side)
        lengths = {v: float(np.linalg.norm(self._pos[r] - self._pos[v])) for v, r in rails.items()}
        shortest = min(lengths.values())
        targets: dict[int, np.ndarray] = {}
        fractions: dict[int, float] = {}
        for v, r in rails.items():
            t = abs(f) if not even else abs(f) * shortest / lengths[v]
            fractions[v] = t
            targets[v] = self._pos[v] + t * (self._pos[r] - self._pos[v])
        self._slide(targets, rails, fractions, correct_uvs)
        return {"vertices": dict(rails)}

    @operation
    def slide_vertex(self, vertex: int, toward: int, factor: float, *, correct_uvs: bool = True):
        """Move ``vertex`` the fraction ``factor`` (0..1) along its edge to ``toward``."""
        (v,) = self._elements(vertex, "vertices")
        (r,) = self._elements(toward, "vertices")
        if self.edge_between(v, r) is None:
            raise EditError(
                "NOT_ADJACENT",
                f"vertices {v} and {r} are not joined by an edge",
                elements={"vertices": [v, r]},
            )
        t = float(factor)
        if not (0.0 <= t < 1.0):
            raise ValidationError("factor must be in [0, 1)")
        target = self._pos[v] + t * (self._pos[r] - self._pos[v])
        self._slide({v: target}, {v: r}, {v: t}, correct_uvs)
        return {"vertex": v}

    # -- helpers ---------------------------------------------------------------------

    def _edge_chains(self, eids: list[int]) -> list[list[tuple[int, int | None, int | None, int]]]:
        """Chains of the edges as ``(vertex, previous, next, chain direction sign)``."""
        adj: dict[int, list[int]] = {}
        for e in eids:
            a, b = self._edge_verts[e]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        forks = [v for v, ns in adj.items() if len(ns) > 2]
        if forks:
            raise EditError(
                "SLIDE_NOT_A_CHAIN",
                "the edges branch: each vertex may have at most two selected edges",
                elements={"vertices": sorted(forks)},
            )
        seen: set[int] = set()
        chains = []
        ends = sorted(v for v, ns in adj.items() if len(ns) == 1)
        for start in ends + sorted(adj):
            if start in seen:
                continue
            closed = len(adj[start]) == 2
            order = [start]
            seen.add(start)
            prev, cur = None, start
            nxt_choice = min(adj[start])
            while True:
                candidates = [x for x in adj[cur] if x != prev and x not in seen]
                if cur == start and prev is None:
                    candidates = [nxt_choice]
                if not candidates:
                    break
                prev, cur = cur, candidates[0]
                order.append(cur)
                seen.add(cur)
            items = []
            m = len(order)
            for i, v in enumerate(order):
                p = order[i - 1] if (i > 0 or closed) else None
                q = order[(i + 1) % m] if (i < m - 1 or closed) else None
                items.append((v, p, q, 1))
            chains.append(items)
        return chains

    def _faces_running(self, a: int, b: int) -> list[int]:
        """Faces that run the edge a -> b in their loop."""
        e = self.edge_between(a, b)
        out = []
        for f in self._edge_faces.get(e, []):
            loop = self._faces[f]
            if loop[(loop.index(a) + 1) % len(loop)] == b:
                out.append(f)
        return out

    def _rail(self, v: int, prev: int | None, nxt: int | None, _sign: int, side: int) -> int:
        """The far end of the edge leaving the chain at ``v`` on ``side``."""
        # Faces on side +1 run the chain's direction (prev -> v -> nxt).
        side_faces: set[int] = set()
        if nxt is not None:
            side_faces.update(
                self._faces_running(v, nxt) if side > 0 else self._faces_running(nxt, v)
            )
        if prev is not None:
            side_faces.update(
                self._faces_running(prev, v) if side > 0 else self._faces_running(v, prev)
            )
        chain_neighbours = {x for x in (prev, nxt) if x is not None}
        if not side_faces:
            raise EditError(
                "SLIDE_NO_RAIL",
                f"vertex {v} has no face on that side of the chain",
                elements={"vertices": [v]},
                hint="slide the other way (negative factor)",
            )
        # Walk the fan of v from the side's faces, across edges not in the chain.
        fan_faces = set(side_faces)
        stack = list(side_faces)
        candidates: set[int] = set()
        while stack:
            f = stack.pop()
            loop = self._faces[f]
            k = loop.index(v)
            for u in (loop[k - 1], loop[(k + 1) % len(loop)]):
                if u in chain_neighbours:
                    continue
                candidates.add(u)
                e = self.edge_between(v, u)
                for g in self._edge_faces[e]:
                    if g not in fan_faces:
                        fan_faces.add(g)
                        stack.append(g)
        if not candidates:
            raise EditError(
                "SLIDE_NO_RAIL",
                f"no edge leaves the chain at vertex {v} on that side",
                elements={"vertices": [v]},
            )
        if len(candidates) == 1:
            return next(iter(candidates))
        tangent = np.zeros(3)
        for x, s in ((prev, -1.0), (nxt, 1.0)):
            if x is not None:
                d = poly.unit(self._pos[x] - self._pos[v])
                if d is not None:
                    tangent += s * d
        t = poly.unit(tangent)

        def score(u: int) -> tuple[float, int]:
            d = poly.unit(self._pos[u] - self._pos[v])
            if d is None or t is None:
                return (1.0, self.edge_between(v, u))
            return (abs(float(np.dot(d, t))), self.edge_between(v, u))

        return min(candidates, key=score)

    def _slide(
        self,
        targets: dict[int, np.ndarray],
        rails: dict[int, int],
        fractions: dict[int, float],
        correct_uvs: bool,
    ) -> None:
        vids = sorted(targets)
        faces = sorted({g for v in vids for g in self._faces_of_vertex(v)})
        before = {g: self._face_normal_vec(g) for g in faces}
        self._set_positions(vids, np.array([targets[v] for v in vids]))
        for g in faces:
            nrm = self._face_normal_vec(g)
            pts = self._pos[self._faces[g]]
            if (
                nrm is None
                or (before[g] is not None and float(np.dot(nrm, before[g])) <= 0)
                or not poly.is_simple(poly.project(pts, nrm))
            ):
                raise EditError(
                    "SLIDE_FOLDS_FACE",
                    f"sliding that far folds face {g}",
                    elements={"faces": [g]},
                    hint="use a smaller factor",
                )
        if not correct_uvs or UV not in self._layers:
            return
        layer = self._layers[UV]
        stretched = []
        for v in vids:
            r, t = rails[v], fractions[v]
            rail_value = None
            new_corners: dict[int, list] = {}
            for g in sorted(self._faces_of_vertex(v)):
                loop = self._faces[g]
                if r in loop:
                    k, kr = loop.index(v), loop.index(r)
                    a, b = np.asarray(layer.data[g][k]), np.asarray(layer.data[g][kr])
                    value = tuple(float(x) for x in a + t * (b - a))
                    if rail_value is None:
                        rail_value = (tuple(layer.data[g][k]), value)
                    values = list(layer.data[g])
                    values[k] = value
                    new_corners[g] = values
            for g in sorted(self._faces_of_vertex(v)):
                if g in new_corners:
                    continue
                loop = self._faces[g]
                k = loop.index(v)
                if rail_value is not None and _same(layer.data[g][k], rail_value[0]):
                    values = list(layer.data[g])
                    values[k] = rail_value[1]
                    new_corners[g] = values
                else:
                    stretched.append(g)
            for g, values in new_corners.items():
                self._set_corners(g, UV, values)
        if stretched:
            self._diagnose(
                Diagnostic(
                    "UV_SEAM_STRETCHED",
                    INFO,
                    "faces across a UV seam from the slide keep their corner UVs and stretch",
                    None,
                    len(set(stretched)),
                    tuple(sorted(set(stretched))[:10]),
                )
            )


def _same(a, b) -> bool:
    return all(abs(x - y) <= 1e-12 for x, y in zip(a, b, strict=True))
