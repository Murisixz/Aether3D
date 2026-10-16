"""Filling holes and bridging edge loops.

Both work on **boundary loops**: chains of boundary edges (edges with one face). A new
face along a boundary edge must run the edge opposite to the face already there, which
fixes the orientation of every fill and bridge face, so the result is consistently wound.

Fill
----
``fill_holes`` closes each loop with one polygon (``"ngon"``), a fan of triangles around
the loop's centroid (``"fan"``), or a minimum-weight triangulation (``"triangulate"``):
the dynamic program of Barequet and Sharir with Liepa's weight ("Filling Holes in
Meshes", SGP 2003), minimizing first the largest dihedral angle between adjacent
triangles (including the faces around the hole) and then the total area. ``"auto"``
makes a polygon of a planar, simple loop, triangulates loops of up to
:data:`MAX_DP_LOOP` vertices, and fans larger ones.

Bridge
------
``bridge_edge_loops`` joins two boundary loops (closed or open) with a strip of faces.
Oriented by their faces, the two loops run in opposite directions across the bridge, so
loop A's ``a_i`` meets loop B's ``b_(offset - i)``; the offset minimizing the summed squared
rung lengths is used (``twist`` adds to it). Equal vertex counts give quads (``segments``
rows, linearly interpolated); different counts give triangles by merging the two loops'
normalized arc lengths.

Corner attributes come from the faces around the loop (as extrusion walls do), so UVs of
the new faces may have zero area, which is reported (``UV_DEGENERATE_FACES``).
"""

from __future__ import annotations

import math

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import int_at_least
from . import _polygon as poly
from .core import UV, Elements, Selection, operation

MAX_DP_LOOP = 64
"""Loops longer than this are not triangulated by dynamic programming (O(n^3))."""


class FillMixin:
    # -- boundary loops --------------------------------------------------------------

    def _boundary_edge_loops(self, edges: list[int]) -> list[list[int]]:
        """Group boundary edges into chains; each chain lists its edges in order, running
        each edge opposite to its face (the direction a new face along it must run)."""
        nxt: dict[int, list[tuple[int, int]]] = {}
        for e in edges:
            faces = self._edge_faces.get(e, [])
            if len(faces) != 1:
                continue
            a, b = self._edge_verts[e]
            loop = self._faces[faces[0]]
            if loop[(loop.index(a) + 1) % len(loop)] == b:
                a, b = b, a  # the face runs a -> b; the new face runs b -> a
            nxt.setdefault(a, []).append((b, e))
        for v, outs in nxt.items():
            if len(outs) > 1:
                raise EditError(
                    "AMBIGUOUS_BOUNDARY",
                    f"vertex {v} is where several boundary loops touch",
                    elements={"vertices": [v]},
                    hint="fill or bridge the loops one at a time, or weld that vertex apart",
                )
        has_prev = {b for outs in nxt.values() for b, _ in outs}
        chains: list[list[int]] = []
        seen: set[int] = set()
        # Open chains first (they start at a vertex no chain edge enters), then cycles.
        starts = sorted(v for v in nxt if v not in has_prev) + sorted(nxt)
        for s in starts:
            if s in seen or s not in nxt:
                continue
            chain: list[int] = []
            v = s
            while v in nxt and v not in seen:
                seen.add(v)
                b, e = nxt[v][0]
                chain.append(e)
                v = b
            if chain:
                chains.append(chain)
        return chains

    def _chain_vertices(self, chain: list[int]) -> tuple[list[int], bool]:
        """Vertices of a chain in the new faces' direction and whether it is closed."""
        verts: list[int] = []
        for k, e in enumerate(chain):
            a, b = self._edge_verts[e]
            if k == 0:
                if len(chain) > 1:
                    na, nb = self._edge_verts[chain[1]]
                    first = a if b in (na, nb) else b
                else:
                    faces = self._edge_faces[e]
                    loop = self._faces[faces[0]]
                    first = b if loop[(loop.index(a) + 1) % len(loop)] == b else a
                verts.append(first)
            last = verts[-1]
            verts.append(b if last == a else a)
        closed = len(verts) > 2 and verts[0] == verts[-1]
        return (verts[:-1] if closed else verts), closed

    def _vertex_attrs_on_boundary(self, v: int) -> dict:
        """Corner attributes of ``v`` in the lowest-id face that uses it."""
        f = min(self._faces_of_vertex(v))
        return self._corners_of(f)[self._faces[f].index(v)]

    # -- fill ------------------------------------------------------------------------

    @operation
    def fill_holes(self, edges: Elements | None = None, *, mode: str = "auto", max_sides: int = 0):
        """Close boundary loops (all of them, or those made of ``edges``) with faces.

        ``mode``: ``"auto"``, ``"ngon"``, ``"fan"`` or ``"triangulate"`` (see the module
        docs). Loops with more than ``max_sides`` edges are skipped when ``max_sides > 0``.
        Open chains are not loops and are skipped (reported). ``data["faces"]`` are the new
        faces, ``data["loops"]`` the filled loops' vertex lists.
        """
        if mode not in ("auto", "ngon", "fan", "triangulate"):
            raise ValidationError(f"mode must be auto, ngon, fan or triangulate, got {mode!r}")
        if isinstance(max_sides, bool) or not isinstance(max_sides, int) or max_sides < 0:
            raise ValidationError("max_sides must be an integer >= 0")
        if edges is None:
            candidates = sorted(e for e, fs in self._edge_faces.items() if len(fs) == 1)
        else:
            candidates = self._elements(edges, "edges")
            inner = [e for e in candidates if len(self._edge_faces[e]) != 1]
            if inner:
                raise EditError(
                    "NOT_BOUNDARY",
                    f"{len(inner)} edges are not boundary edges",
                    elements={"edges": inner},
                    hint="select boundary edges (mesh.boundary_edges())",
                )
        new_faces: list[int] = []
        filled: list[list[int]] = []
        skipped: list[int] = []
        for chain in self._boundary_edge_loops(candidates):
            verts, closed = self._chain_vertices(chain)
            if not closed or len(verts) < 3:
                skipped.append(chain[0])
                continue
            if max_sides and len(verts) > max_sides:
                skipped.append(chain[0])
                continue
            new_faces += self._fill_loop(chain, mode)
            filled.append(verts)
        if skipped:
            self._diagnose(
                Diagnostic(
                    "HOLES_NOT_FILLED",
                    INFO,
                    "open chains and loops longer than max_sides were not filled",
                    None,
                    len(skipped),
                    tuple(skipped[:10]),
                )
            )
        return {"faces": Selection.of(faces=new_faces), "loops": filled}

    def _fill_loop(self, chain: list[int], mode: str) -> list[int]:
        verts, closed = self._chain_vertices(chain)
        if not closed:
            return []
        pts = self._pos[verts]
        m = len(verts)
        nrm = poly.newell_normal(pts)
        simple = poly.unit(nrm) is not None and poly.is_simple(poly.project(pts, nrm))
        planar = simple and _plane_deviation(pts) <= 1e-9 * max(self._scale(), 1e-300)
        if mode == "auto":
            mode = "ngon" if planar else ("triangulate" if m <= MAX_DP_LOOP else "fan")
        neighbours = sorted({f for e in chain for f in self._edge_faces[e]})
        smooth = all(self._smooth[f] for f in neighbours)
        group = self._group.get(neighbours[0], 0)
        attrs = [self._vertex_attrs_on_boundary(v) for v in verts]
        made: list[int] = []
        if mode == "ngon":
            if not simple:
                raise EditError(
                    "FILL_NOT_SIMPLE",
                    "the loop is not a simple polygon seen along its normal",
                    elements={"vertices": verts[:10]},
                    hint="use mode='triangulate' or 'fan'",
                )
            made.append(
                self._new_face(verts, self._split_corners(attrs), smooth=smooth, group=group)
            )
        elif mode == "fan":
            c = self._new_vertex(pts.mean(axis=0))
            mid = self._mix([(a, 1.0 / m) for a in attrs])
            for i in range(m):
                j = (i + 1) % m
                made.append(
                    self._new_face(
                        [c, verts[i], verts[j]],
                        self._split_corners([mid, attrs[i], attrs[j]]),
                        smooth=smooth,
                        group=group,
                    )
                )
        else:
            if m > MAX_DP_LOOP:
                raise EditError(
                    "FILL_TOO_LARGE",
                    f"the loop has {m} vertices; triangulation handles up to {MAX_DP_LOOP}",
                    hint="use mode='fan' or 'ngon'",
                )
            outer = [self._face_normal_vec(self._edge_faces[e][0]) for e in chain]
            for a, b, c in _min_weight_triangulation(pts, outer):
                made.append(
                    self._new_face(
                        [verts[a], verts[b], verts[c]],
                        self._split_corners([attrs[a], attrs[b], attrs[c]]),
                        smooth=smooth,
                        group=group,
                    )
                )
        eps = self._area_eps()
        bad = [f for f in made if poly.polygon_area(self._pos[self._faces[f]]) <= eps]
        if bad:
            raise EditError(
                "DEGENERATE_RESULT",
                "filling would create zero-area faces",
                elements={"faces": bad},
                hint="try another mode",
            )
        self._report_uv_area(made)
        return made

    def _report_uv_area(self, faces: list[int]) -> None:
        if UV not in self._layers or not faces:
            return
        flat = []
        for f in faces:
            uv = np.asarray(self._layers[UV].data[f], dtype=np.float64)
            x, y = uv[:, 0], uv[:, 1]
            area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            if area <= 1e-12:
                flat.append(f)
        if flat:
            self._diagnose(
                Diagnostic(
                    "UV_DEGENERATE_FACES",
                    WARNING,
                    "new faces take the UVs of the boundary they close and have zero UV area",
                    None,
                    len(flat),
                    tuple(flat[:10]),
                )
            )

    # -- bridge ----------------------------------------------------------------------

    @operation
    def bridge_edge_loops(self, edges: Elements, *, segments: int = 1, twist: int = 0):
        """Join two boundary loops (or two open boundary chains) with a strip of faces.

        ``segments`` rows of quads (equal vertex counts); with different counts the
        strip is one row of triangles. ``twist`` shifts the pairing by that many
        vertices. ``data["faces"]`` are the new faces, ``data["vertices"]`` the new
        vertices of intermediate rows.
        """
        segments = int_at_least(segments, 1, "segments")
        if segments > 256:
            raise ValidationError("segments must be at most 256")
        if isinstance(twist, bool) or not isinstance(twist, int):
            raise ValidationError("twist must be an integer")
        eids = self._elements(edges, "edges")
        inner = [e for e in eids if len(self._edge_faces[e]) != 1]
        if inner:
            raise EditError(
                "NOT_BOUNDARY",
                f"{len(inner)} edges are not boundary edges",
                elements={"edges": inner},
                hint="bridging joins open boundaries; delete the faces inside the loops first",
            )
        chains = self._boundary_edge_loops(eids)
        if len(chains) != 2:
            raise EditError(
                "BRIDGE_NEEDS_TWO_LOOPS",
                f"the edges form {len(chains)} boundary loops or chains; bridging needs 2",
                elements={"edges": eids[:20]},
            )
        (va, closed_a), (vb, closed_b) = (self._chain_vertices(c) for c in chains)
        if closed_a != closed_b:
            raise EditError(
                "BRIDGE_NEEDS_TWO_LOOPS",
                "one chain is closed and the other open",
                elements={"edges": eids[:20]},
            )
        if set(va) & set(vb):
            raise EditError(
                "BRIDGE_LOOPS_TOUCH",
                "the two loops share vertices",
                elements={"vertices": sorted(set(va) & set(vb))},
            )
        neighbours = sorted({f for c in chains for e in c for f in self._edge_faces[e]})
        smooth = all(self._smooth[f] for f in neighbours)
        group = self._group.get(min(self._edge_faces[e][0] for e in chains[0]), 0)
        pa, pb = self._pos[va], self._pos[vb]
        na, nb = len(va), len(vb)
        # Both chains run in their new faces' direction, so across the bridge B runs the
        # other way: a_i pairs with b_(offset - i).
        if closed_a:
            if na == nb:
                idx = (np.arange(na)[:, None] - np.arange(na)[None, :]) % na  # [off, i]
                cost = ((pa[None, :, :] - pb[idx]) ** 2).sum(axis=2).sum(axis=1)
                offset = int(np.argmin(cost)) + twist
                b_order = [vb[(offset - i) % nb] for i in range(nb)]
            else:
                start = int(np.argmin(((pb - pa[0]) ** 2).sum(axis=1))) + twist
                b_order = [vb[(start - i) % nb] for i in range(nb)]
        else:
            b_order = vb[::-1]
        attrs_a = [self._vertex_attrs_on_boundary(v) for v in va]
        attrs_b = {v: self._vertex_attrs_on_boundary(v) for v in vb}
        made: list[int] = []
        created: list[int] = []
        if na == nb:
            rows = [list(va)]
            row_attrs = [attrs_a]
            for k in range(1, segments):
                t = k / segments
                ids, vals = [], []
                for i in range(na):
                    x = self._new_vertex((1 - t) * self._pos[va[i]] + t * self._pos[b_order[i]])
                    created.append(x)
                    ids.append(x)
                    vals.append(self._mix([(attrs_a[i], 1 - t), (attrs_b[b_order[i]], t)]))
                rows.append(ids)
                row_attrs.append(vals)
            rows.append(list(b_order))
            row_attrs.append([attrs_b[v] for v in b_order])
            count = na if closed_a else na - 1
            for k in range(segments):
                r0, r1, a0, a1 = rows[k], rows[k + 1], row_attrs[k], row_attrs[k + 1]
                for i in range(count):
                    j = (i + 1) % na
                    loop = [r0[i], r0[j], r1[j], r1[i]]
                    vals = [a0[i], a0[j], a1[j], a1[i]]
                    made.append(
                        self._new_face(loop, self._split_corners(vals), smooth=smooth, group=group)
                    )
        else:
            if segments > 1:
                raise EditError(
                    "BRIDGE_SEGMENTS_NEED_EQUAL_COUNTS",
                    f"the loops have {na} and {nb} vertices; segments > 1 needs equal counts",
                    hint="use segments=1, or subdivide one loop's edges first",
                )
            for loop_ids in _zipper(pa, self._pos[b_order], closed_a):
                ids = [va[i] if side == "a" else b_order[i] for side, i in loop_ids]
                vals = [attrs_a[i] if side == "a" else attrs_b[b_order[i]] for side, i in loop_ids]
                made.append(
                    self._new_face(ids, self._split_corners(vals), smooth=smooth, group=group)
                )
        eps = self._area_eps()
        bad = [f for f in made if poly.polygon_area(self._pos[self._faces[f]]) <= eps]
        if bad:
            raise EditError(
                "DEGENERATE_RESULT",
                "the bridge would create zero-area faces",
                elements={"faces": bad},
                hint="check that the loops face each other, or use twist",
            )
        nm = sorted(
            {
                self._edge_by_pair[key]
                for f in made
                for key in self._loop_edges(self._faces[f])
                if len(self._edge_faces[self._edge_by_pair[key]]) > 2
            }
        )
        if nm:
            raise EditError(
                "NON_MANIFOLD_RESULT",
                "the bridge would give edges more than two faces",
                elements={"edges": nm},
            )
        self._report_uv_area(made)
        return {"faces": Selection.of(faces=made), "vertices": Selection.of(created)}


def _plane_deviation(points: np.ndarray) -> float:
    nrm = poly.unit(poly.newell_normal(points))
    if nrm is None:
        return math.inf
    return float(np.abs((points - points.mean(axis=0)) @ nrm).max())


def _zipper(pa: np.ndarray, pb: np.ndarray, closed: bool) -> list[list[tuple[str, int]]]:
    """Triangles joining polylines A and B (B already paired in the bridge's direction),
    advancing along whichever has the smaller normalized arc length next."""

    def params(p: np.ndarray) -> np.ndarray:
        q = np.vstack([p, p[:1]]) if closed else p
        seg = np.sqrt(((q[1:] - q[:-1]) ** 2).sum(axis=1))
        total = float(seg.sum()) or 1.0
        return np.concatenate([[0.0], np.cumsum(seg)]) / total

    ta, tb = params(pa), params(pb)
    na, nb = len(pa), len(pb)
    steps_a = na if closed else na - 1
    steps_b = nb if closed else nb - 1
    i = j = 0
    faces: list[list[tuple[str, int]]] = []
    while i < steps_a or j < steps_b:
        advance_a = j >= steps_b or (i < steps_a and ta[i + 1] <= tb[j + 1])
        if advance_a:
            faces.append([("a", i % na), ("a", (i + 1) % na), ("b", j % nb)])
            i += 1
        else:
            faces.append([("a", i % na), ("b", (j + 1) % nb), ("b", j % nb)])
            j += 1
    return faces


def _min_weight_triangulation(
    pts: np.ndarray, outer: list[np.ndarray | None]
) -> list[tuple[int, int, int]]:
    """Liepa's minimum-weight triangulation of a closed polygon (vertices in the new
    faces' winding). ``outer[i]`` is the normal of the face beyond edge (i, i + 1).
    Weights compare (largest dihedral angle, total area) lexicographically."""
    m = len(pts)

    def tri_normal(i: int, j: int, k: int) -> np.ndarray | None:
        return poly.unit(poly.cross3(pts[j] - pts[i], pts[k] - pts[i]))

    def angle(n1, n2) -> float:
        if n1 is None or n2 is None:
            return math.pi
        return math.acos(max(-1.0, min(1.0, float(np.dot(n1, n2)))))

    def area(i: int, j: int, k: int) -> float:
        return 0.5 * float(np.linalg.norm(poly.cross3(pts[j] - pts[i], pts[k] - pts[i])))

    inf = (math.inf, math.inf)
    best: dict[tuple[int, int], tuple[float, float]] = {}
    choice: dict[tuple[int, int], int] = {}
    for i in range(m - 1):
        best[(i, i + 1)] = (0.0, 0.0)
    for span in range(2, m):
        for i in range(m - span):
            k = i + span
            winner, pick = inf, -1
            for j in range(i + 1, k):
                wl, wr = best[(i, j)], best[(j, k)]
                n = tri_normal(i, j, k)
                if n is None:
                    continue
                dihedral = 0.0
                for a, b in ((i, j), (j, k)):
                    if b == a + 1:
                        dihedral = max(dihedral, angle(n, outer[a]))
                    else:
                        c = choice[(a, b)]
                        dihedral = max(dihedral, angle(n, tri_normal(a, c, b)))
                if span == m - 1:  # the closing edge (0, m - 1) borders the outer face
                    dihedral = max(dihedral, angle(n, outer[m - 1]))
                w = (max(wl[0], wr[0], dihedral), wl[1] + wr[1] + area(i, j, k))
                if w[0] < winner[0] - 1e-12 or (
                    abs(w[0] - winner[0]) <= 1e-12 and w[1] < winner[1]
                ):
                    winner, pick = w, j
            if pick < 0:
                raise EditError(
                    "TRIANGULATION_FAILED", "the hole cannot be triangulated (degenerate loop)"
                )
            best[(i, k)] = winner
            choice[(i, k)] = pick
    out: list[tuple[int, int, int]] = []
    stack = [(0, m - 1)]
    while stack:
        i, k = stack.pop()
        if k - i < 2:
            continue
        j = choice[(i, k)]
        out.append((i, j, k))
        stack += [(i, j), (j, k)]
    return out
