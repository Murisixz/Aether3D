"""Edge bevel (chamfer).

Geometry
--------
In every face adjacent to a bevelled edge, the new edge runs parallel to the
original one at perpendicular distance ``width`` inside the face:

* where two bevelled edges meet inside one face, the corner moves to the
  intersection of their offset lines (a mitre);
* where a bevelled edge meets an unbevelled edge, the corner slides along the
  unbevelled edge to where the offset line crosses it (if both faces of that
  edge request a position, the average is used and ``BEVEL_UNEVEN_WIDTH`` is
  reported when they differ);
* every bevelled edge is replaced by a quad (the chamfer face); the hole left
  at a vertex where three or more new points meet is filled by a polygon
  (e.g. the triangle at a box corner whose three edges are bevelled), or --
  when that polygon would be flat-degenerate and only one bevelled edge
  reaches the vertex -- merged into the chamfer face.

Supported topology: bevelled edges must have exactly two faces, and every
vertex they touch must have a closed, manifold, consistently wound fan of
faces (so open boundaries are not supported). Only single-segment (chamfer)
bevels are implemented. Widths that would fold or invert faces raise
``BEVEL_WIDTH_TOO_LARGE``; nothing is changed in any error case.
"""

from __future__ import annotations

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import EditError, ValidationError
from . import _polygon as poly
from .core import Elements, Selection, operation


class BevelMixin:
    @operation
    def bevel_edges(self, edges: Elements, width: float, *, segments: int = 1):
        """Chamfer edges by ``width`` (see the module documentation for the exact geometry).

        ``data["faces"]`` are the new chamfer faces, ``data["corner_faces"]``
        the polygons filling vertex corners, and ``data["vertices"]`` the new
        vertices.
        """
        eids = self._elements(edges, "edges")
        w = float(width)
        if not w > 0:
            raise ValidationError("width must be > 0")
        if segments != 1:
            raise EditError(
                "UNSUPPORTED_OPERATION",
                "only single-segment (chamfer) bevels are implemented",
                hint="use segments=1",
            )
        bevelled = set(eids)
        for e in eids:
            if len(self._edge_faces[e]) != 2:
                raise EditError(
                    "NOT_MANIFOLD",
                    f"edge {e} has {len(self._edge_faces[e])} faces; bevelled edges need exactly 2",
                    elements={"edges": [e]},
                )
        edge_ends = {e: self._edge_verts[e] for e in eids}  # edges disappear while rebuilding
        vertices = sorted({v for e in eids for v in edge_ends[e]})
        fans = {v: self._closed_fan(v) for v in vertices}
        normal_cache: dict[int, np.ndarray] = {}

        def normal(f: int) -> np.ndarray:
            if f not in normal_cache:
                n = self._face_normal_vec(f)
                if n is None:
                    raise EditError(
                        "DEGENERATE_INPUT", f"face {f} has zero area", elements={"faces": [f]}
                    )
                normal_cache[f] = n
            return normal_cache[f]

        def inward(f: int, a: int, b: int) -> np.ndarray:
            """Unit vector inside face f perpendicular to its directed edge a -> b."""
            d = poly.unit(np.cross(normal(f), self._pos[b] - self._pos[a]))
            if d is None:
                raise EditError(
                    "DEGENERATE_INPUT", f"face {f} has a zero-length edge", elements={"faces": [f]}
                )
            return d

        # 1. Point requests per vertex.
        slide_requests: dict[tuple[int, int], list[np.ndarray]] = {}  # (v, edge) -> positions
        mitre: dict[tuple[int, int], np.ndarray] = {}  # (v, face) -> position
        for v in vertices:
            pv = self._pos[v]
            for f, e_in, e_out, p, q in fans[v]:
                b_in, b_out = e_in in bevelled, e_out in bevelled
                if b_in and b_out:
                    d_in, d_out = inward(f, p, v), inward(f, v, q)
                    m = poly.unit(d_in + d_out)
                    if m is None or float(np.dot(m, d_in)) < 1e-3:
                        raise EditError(
                            "BEVEL_WIDTH_TOO_LARGE",
                            f"bevelled edges fold back at vertex {v}",
                            elements={"vertices": [v]},
                        )
                    mitre[(v, f)] = pv + w / float(np.dot(m, d_in)) * m
                elif b_in or b_out:
                    d = inward(f, p, v) if b_in else inward(f, v, q)
                    other, edge = (q, e_out) if b_in else (p, e_in)
                    u = poly.unit(self._pos[other] - pv)
                    c = float(np.dot(u, d)) if u is not None else 0.0
                    if c < 1e-3:
                        raise EditError(
                            "BEVEL_WIDTH_TOO_LARGE",
                            f"edge {edge} runs (almost) parallel to a "
                            f"bevelled edge at vertex {v}; the corner cannot slide along it",
                            elements={"edges": [edge]},
                        )
                    slide_requests.setdefault((v, edge), []).append(pv + w / c * u)
        slide: dict[tuple[int, int], np.ndarray] = {}
        uneven = []
        for key, points in slide_requests.items():
            slide[key] = np.mean(points, axis=0)
            if len(points) == 2 and np.linalg.norm(points[0] - points[1]) > 1e-9 * self._scale():
                uneven.append(key[1])
        if uneven:
            self._diagnose(
                Diagnostic(
                    "BEVEL_UNEVEN_WIDTH",
                    INFO,
                    "two faces requested different slide positions on an edge; their "
                    "average was used, so the width there is approximate",
                    None,
                    len(uneven),
                    tuple(sorted(uneven)[:10]),
                )
            )
        # Slide points must stay on their edge (and not cross one coming from the other end).
        for (v, e), point in slide.items():
            a, b = self._edge_verts[e]
            other = b if a == v else a
            length = float(np.linalg.norm(self._pos[other] - self._pos[v]))
            used = float(np.linalg.norm(point - self._pos[v]))
            if (other, e) in slide:
                used += float(np.linalg.norm(slide[(other, e)] - self._pos[other]))
            if used >= length * (1 - 1e-9):
                raise EditError(
                    "BEVEL_WIDTH_TOO_LARGE",
                    f"width {w} is too large for edge {e}",
                    elements={"edges": [e]},
                    hint="use a smaller width",
                )

        # 2. Corner replacement lists (in each face's loop direction).
        point_id: dict[tuple, int] = {}
        new_vertices: list[int] = []

        def vertex_for(key: tuple, position: np.ndarray) -> int:
            if key not in point_id:
                point_id[key] = self._new_vertex(position)
                new_vertices.append(point_id[key])
            return point_id[key]

        replacement: dict[tuple[int, int], list[int]] = {}  # (v, face) -> new vertex ids
        retained: set[int] = set()
        for v in vertices:
            for f, e_in, e_out, _p, _q in fans[v]:
                if (v, f) in mitre:
                    replacement[(v, f)] = [vertex_for(("m", v, f), mitre[(v, f)])]
                    continue
                seq = []
                for e in (e_in, e_out):
                    if e in bevelled:
                        continue
                    if (v, e) in slide:
                        seq.append(vertex_for(("s", v, e), slide[(v, e)]))
                    else:
                        seq.append(v)
                        retained.add(v)
                replacement[(v, f)] = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]

        # 3. Rebuild the faces around the bevelled vertices (attributes interpolated in-face).
        affected = sorted({f for v in vertices for f, *_ in fans[v]})
        corner_attrs: dict[tuple[int, int], dict] = {}  # (face, vertex id) -> attributes
        new_loops: dict[int, tuple[list[int], list[dict]]] = {}
        for f in affected:
            loop, corners = self._faces[f], self._corners_of(f)
            fits_targets = []
            new_loop: list[int] = []
            new_corners: list[dict | None] = []
            for v, c in zip(loop, corners, strict=True):
                if (v, f) in replacement:
                    for x in replacement[(v, f)]:
                        new_loop.append(x)
                        if x == v:
                            new_corners.append(c)
                        else:
                            new_corners.append(None)
                            fits_targets.append((len(new_corners) - 1, x))
                else:
                    new_loop.append(v)
                    new_corners.append(c)
            if fits_targets:
                values = self._affine_attrs(f, np.array([self._pos[x] for _, x in fits_targets]))
                for (slot, _), attrs in zip(fits_targets, values, strict=True):
                    new_corners[slot] = attrs
            for x, attrs in zip(new_loop, new_corners, strict=True):
                corner_attrs[(f, x)] = attrs
            new_loops[f] = (new_loop, new_corners)
        for f, (new_loop, _corners) in new_loops.items():
            if len(set(new_loop)) != len(new_loop) or len(new_loop) < 3:
                raise EditError(
                    "BEVEL_UNSUPPORTED_TOPOLOGY",
                    f"face {f} would become invalid",
                    elements={"faces": [f]},
                )
        original_normals = {f: normal(f) for f in affected}
        for f, (new_loop, new_corners) in new_loops.items():
            self._replace_face(f, new_loop, self._split_corners(new_corners))

        # 4. Chamfer faces.
        chamfers: dict[int, int] = {}
        chamfer_loops: dict[int, list[int]] = {}
        chamfer_ends: dict[int, tuple] = {}  # edge -> (original direction a->b, chamfer ends)
        chamfer_corners: dict[int, list[dict]] = {}
        edge_faces = self._edge_faces_index(fans)
        directed = self._directed_index(fans)
        for e in eids:
            f, g = edge_faces[e]
            a, b = directed[(e, f)]  # f ran a -> b
            xf_a, xf_b = (
                self._point_in(a, f, replacement),
                self._point_in(b, f, replacement, last=False),
            )
            xg_a, xg_b = (
                self._point_in(a, g, replacement, last=False),
                self._point_in(b, g, replacement),
            )
            loop = [xf_b, xf_a, xg_a, xg_b]
            chamfer_loops[e] = loop
            chamfer_ends[e] = (self._pos[b] - self._pos[a], xf_a, xf_b, xg_a, xg_b)
            chamfer_corners[e] = [
                corner_attrs[(f, xf_b)],
                corner_attrs[(f, xf_a)],
                corner_attrs[(g, xg_a)],
                corner_attrs[(g, xg_b)],
            ]

        # 5. Corner polygons (holes at vertices where 3+ new points meet).
        corner_faces = []
        # Pending merges of degenerate corners into chamfers; one chamfer can receive
        # a merge at each of its two ends, so keep a list per edge.
        merges: list[tuple[int, int, list[int], list[dict]]] = []
        for v in vertices:
            seq: list[int] = []
            owner: dict[int, int] = {}
            for f, *_ in fans[v]:
                for x in replacement[(v, f)]:
                    owner.setdefault(x, f)
                    if not seq or seq[-1] != x:
                        seq.append(x)
            while len(seq) > 1 and seq[0] == seq[-1]:
                seq.pop()
            if len(seq) < 3:
                continue
            hole = seq[::-1]  # faces run the fan forwards; the hole must run the other way
            attrs = [corner_attrs[(owner[x], x)] for x in hole]
            area = poly.polygon_area(self._pos[hole])
            if area > self._area_eps() and poly.is_simple(
                poly.project(self._pos[hole], poly.newell_normal(self._pos[hole]))
            ):
                smooth = all(self._smooth[f] for f, *_ in fans[v])
                corner_faces.append(self._new_face(hole, self._split_corners(attrs), smooth=smooth))
                continue
            touching = [e for e in eids if v in edge_ends[e]]
            if len(touching) != 1:
                raise EditError(
                    "BEVEL_UNSUPPORTED_VERTEX",
                    f"the corner at vertex {v} would be a degenerate polygon",
                    elements={"vertices": [v]},
                    hint="bevel fewer edges at this vertex",
                )
            merges.append((touching[0], v, hole, attrs))
        for e, v, hole, attrs in merges:
            loop, values = chamfer_loops[e], chamfer_corners[e]
            # The chamfer crosses the hole along one segment; replace it by the hole's other path.
            n = len(hole)
            for k in range(len(loop)):
                x, y = loop[k], loop[(k + 1) % len(loop)]
                if x in hole and y in hole and hole[(hole.index(y) + 1) % n] == x:
                    i = hole.index(x)
                    path = [hole[(i + j) % n] for j in range(1, n)]  # from after x round to y
                    path_attrs = [attrs[(i + j) % n] for j in range(1, n)]
                    chamfer_loops[e] = loop[: k + 1] + path[:-1] + loop[k + 1 :]
                    chamfer_corners[e] = values[: k + 1] + path_attrs[:-1] + values[k + 1 :]
                    break
            else:  # pragma: no cover - the segment always exists by construction
                raise EditError(
                    "BEVEL_UNSUPPORTED_VERTEX",
                    f"cannot close the corner at vertex {v}",
                    elements={"vertices": [v]},
                )
        for e in eids:
            loop = chamfer_loops[e]
            if len(set(loop)) != len(loop):
                raise EditError(
                    "BEVEL_UNSUPPORTED_TOPOLOGY",
                    f"the chamfer of edge {e} would repeat a vertex",
                    elements={"edges": [e]},
                )
        for e in eids:
            f, g = edge_faces[e]
            chamfers[e] = self._new_face(
                chamfer_loops[e],
                self._split_corners(chamfer_corners[e]),
                smooth=self._smooth[f] and self._smooth[g],
            )

        for v in vertices:
            if v not in retained and self._alive[v] and not self._vert_edges[v]:
                self._delete_vertex(v)

        # 6. Verify: chamfer edges keep their direction (a reversed one means the width
        # carried a face past its opposite side), nothing folded, nothing degenerate.
        for e, (direction, xf_a, xf_b, xg_a, xg_b) in chamfer_ends.items():
            if (
                float(np.dot(self._pos[xf_b] - self._pos[xf_a], direction)) <= 0
                or float(np.dot(self._pos[xg_b] - self._pos[xg_a], direction)) <= 0
            ):
                raise EditError(
                    "BEVEL_WIDTH_TOO_LARGE",
                    f"width {w} collapses the faces next to edge {e}",
                    elements={"edges": [e]},
                    hint="use a smaller width",
                )
        eps = self._area_eps()
        for f in affected:
            n = self._face_normal_vec(f)
            if (
                n is None
                or float(np.dot(n, original_normals[f])) <= 0
                or not poly.is_simple(poly.project(self._pos[self._faces[f]], original_normals[f]))
            ):
                raise EditError(
                    "BEVEL_WIDTH_TOO_LARGE",
                    f"width {w} folds face {f}",
                    elements={"faces": [f]},
                    hint="use a smaller width",
                )
        bad = [
            f
            for f in [*chamfers.values(), *corner_faces]
            if poly.polygon_area(self._pos[self._faces[f]]) <= eps
        ]
        if bad:
            raise EditError(
                "BEVEL_WIDTH_TOO_LARGE",
                "the bevel would create zero-area faces",
                elements={"faces": bad},
            )
        return {
            "faces": Selection.of(faces=chamfers.values()),
            "corner_faces": Selection.of(faces=corner_faces),
            "vertices": Selection.of(new_vertices),
        }

    # -- helpers -------------------------------------------------------------------

    def _closed_fan(self, v: int) -> list[tuple[int, int, int, int, int]]:
        """Faces around ``v`` in order as ``(face, edge_in, edge_out, p, q)``.

        ``face`` runs ``p -> v -> q``; ``edge_out`` of one entry is ``edge_in`` of
        the next. Requires a closed, manifold, consistently wound fan.
        """
        edges = self._vert_edges[v]
        if len(edges) < 3:
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"vertex {v} has only {len(edges)} edges; bevelling needs at least 3",
                elements={"vertices": [v]},
            )
        if any(len(self._edge_faces[e]) != 2 for e in edges):
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"vertex {v} is on an open boundary or a non-manifold edge",
                elements={"vertices": [v]},
                hint="bevelling at boundaries is not supported",
            )
        faces = self._faces_of_vertex(v)
        start = min(faces)
        fan = []
        f = start
        while True:
            loop = self._faces[f]
            k = loop.index(v)
            p, q = loop[k - 1], loop[(k + 1) % len(loop)]
            e_in, e_out = self._edge_id(p, v), self._edge_id(v, q)
            fan.append((f, e_in, e_out, p, q))
            g_faces = self._edge_faces[e_out]
            g = g_faces[0] if g_faces[1] == f else g_faces[1]
            lg = self._faces[g]
            if lg[(lg.index(v) - 1)] != q:
                raise EditError(
                    "BEVEL_UNSUPPORTED_VERTEX",
                    f"faces around vertex {v} are wound inconsistently",
                    elements={"vertices": [v]},
                )
            f = g
            if f == start:
                break
            if len(fan) > len(faces):  # pragma: no cover - defensive
                break
        if len(fan) != len(faces) or len(fan) != len(edges):
            raise EditError(
                "BEVEL_UNSUPPORTED_VERTEX",
                f"vertex {v} is non-manifold (its faces form more than one fan)",
                elements={"vertices": [v]},
            )
        return fan

    @staticmethod
    def _edge_faces_index(fans) -> dict[int, tuple[int, int]]:
        """For each edge, its two faces in fan order (from the first fan listing both).

        Built once per bevel (``fans`` is not modified afterwards), so lookups are
        constant-time instead of a scan over every fan.
        """
        index: dict[int, tuple[int, int]] = {}
        for fan in fans.values():
            per_edge: dict[int, list[int]] = {}
            for f, e_in, e_out, *_ in fan:
                for e in dict.fromkeys((e_in, e_out)):
                    per_edge.setdefault(e, []).append(f)
            for e, faces in per_edge.items():
                if len(faces) == 2 and e not in index:
                    index[e] = (faces[0], faces[1])
        return index

    @staticmethod
    def _directed_index(fans) -> dict[tuple[int, int], tuple[int, int]]:
        """``(edge, face) -> (a, b)``: the endpoints of the edge with face ``f`` running a -> b."""
        index: dict[tuple[int, int], tuple[int, int]] = {}
        for v, fan in fans.items():
            for face, e_in, e_out, p, q in fan:
                index.setdefault((e_out, face), (v, q))
                index.setdefault((e_in, face), (p, v))
        return index

    @staticmethod
    def _point_in(v: int, f: int, replacement, last: bool = True) -> int:
        points = replacement[(v, f)]
        return points[-1] if last else points[0]
