"""Face subdivision and loop cuts."""

from __future__ import annotations

from ..errors import EditError, ValidationError
from ..math3d import int_at_least
from .core import Elements, Selection, operation

MAX_CUTS = 100


def _cuts(value: int) -> int:
    cuts = int_at_least(value, 1, "cuts")
    if cuts > MAX_CUTS:
        raise ValidationError(f"cuts must be at most {MAX_CUTS}")
    return cuts


class SubdivideMixin:
    @operation
    def subdivide_faces(self, faces: Elements, cuts: int = 1):
        """Subdivide faces without changing their shape.

        Each edge of the selected faces gets ``cuts`` evenly spaced vertices
        (neighbouring faces receive them too, so the mesh stays connected).
        Quads become a ``(cuts+1) x (cuts+1)`` grid of quads (bilinear),
        triangles ``(cuts+1)^2`` triangles (barycentric), and larger polygons
        (``cuts=1`` only) a fan of quads around their center. Attributes are
        interpolated the same way; stored normals are kept and interpolated
        (subdivision refines the facets but not the intended smooth surface).
        ``data["faces"]`` maps each face to its pieces (the first keeps the
        original id).
        """
        fids = self._elements(faces, "faces")
        cuts = _cuts(cuts)
        for f in fids:
            if len(self._faces[f]) > 4 and cuts > 1:
                raise EditError(
                    "UNSUPPORTED_TOPOLOGY",
                    f"face {f} has {len(self._faces[f])} sides; "
                    "polygons with more than 4 sides support cuts=1 only",
                    elements={"faces": [f]},
                    hint="use cuts=1 or triangulate first",
                )
        originals = {
            f: (list(self._faces[f]), self._corners_of(f), self._smooth[f], self._group.get(f, 0))
            for f in fids
        }
        params = [k / (cuts + 1) for k in range(1, cuts + 1)]
        points: dict[tuple[int, int], list[int]] = {}
        for f in fids:
            loop = originals[f][0]
            for k in range(len(loop)):
                a, b = loop[k], loop[(k + 1) % len(loop)]
                key = (min(a, b), max(a, b))
                if key not in points:
                    points[key] = self._split_edge(self._edge_by_pair[key], key[0], params)
        pieces: dict[int, list[int]] = {}
        new_vertices: list[int] = []
        for f in fids:
            loop, corners, smooth, group = originals[f]

            def side(a: int, b: int) -> list[int]:
                ids = points[(min(a, b), max(a, b))]
                return [a, *(ids if a < b else ids[::-1]), b]

            if len(loop) == 4:
                result = self._subdivide_quad(f, loop, corners, side, cuts, new_vertices)
            elif len(loop) == 3:
                result = self._subdivide_triangle(f, loop, corners, side, cuts, new_vertices)
            else:
                result = self._subdivide_polygon(f, loop, corners, side, new_vertices)
            built = []
            for k, (piece_loop, piece_corners) in enumerate(result):
                split = self._split_corners(piece_corners)
                if k == 0:
                    self._replace_face(f, piece_loop, split, dirty=False)
                    built.append(f)
                else:
                    built.append(
                        self._new_face(piece_loop, split, smooth=smooth, group=group, dirty=False)
                    )
            pieces[f] = built
        return {"faces": pieces, "vertices": Selection.of(new_vertices)}

    def _subdivide_quad(self, f, loop, corners, side, cuts, new_vertices):
        m = cuts + 1
        c = self._pos[loop]
        grid = [
            [None] * (m + 1) for _ in range(m + 1)
        ]  # grid[i][j], i along c0->c1, j along c0->c3
        s0, s1, s2, s3 = (
            side(loop[0], loop[1]),
            side(loop[1], loop[2]),
            side(loop[3], loop[2]),
            side(loop[0], loop[3]),
        )
        for k in range(m + 1):
            grid[k][0], grid[m][k], grid[k][m], grid[0][k] = s0[k], s1[k], s2[k], s3[k]

        def weights(i, j):
            u, v = i / m, j / m
            return [(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]

        for i in range(1, m):
            for j in range(1, m):
                w = weights(i, j)
                grid[i][j] = self._new_vertex(sum(wk * ck for wk, ck in zip(w, c, strict=True)))
                new_vertices.append(grid[i][j])
        out = []
        for j in range(m):
            for i in range(m):
                idx = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                out.append(
                    (
                        [grid[a][b] for a, b in idx],
                        [self._mix(list(zip(corners, weights(a, b), strict=True))) for a, b in idx],
                    )
                )
        return out

    def _subdivide_triangle(self, f, loop, corners, side, cuts, new_vertices):
        m = cuts + 1
        c = self._pos[loop]
        grid: dict[tuple[int, int], int] = {}
        s0, s1, s2 = side(loop[0], loop[1]), side(loop[1], loop[2]), side(loop[0], loop[2])
        for k in range(m + 1):
            grid[(k, 0)], grid[(m - k, k)], grid[(0, k)] = s0[k], s1[k], s2[k]

        def weights(i, j):
            u, v = i / m, j / m
            return [1 - u - v, u, v]

        for i in range(1, m):
            for j in range(1, m - i):
                w = weights(i, j)
                grid[(i, j)] = self._new_vertex(sum(wk * ck for wk, ck in zip(w, c, strict=True)))
                new_vertices.append(grid[(i, j)])
        out = []

        def tri(idx):
            return (
                [grid[k] for k in idx],
                [self._mix(list(zip(corners, weights(*k), strict=True))) for k in idx],
            )

        for j in range(m):
            for i in range(m - j):
                out.append(tri([(i, j), (i + 1, j), (i, j + 1)]))
                if i + j < m - 1:
                    out.append(tri([(i + 1, j), (i + 1, j + 1), (i, j + 1)]))
        return out

    def _subdivide_polygon(self, f, loop, corners, side, new_vertices):
        n = len(loop)
        center = self._new_vertex(self._pos[loop].mean(axis=0))
        new_vertices.append(center)
        center_attrs = self._mix([(c, 1.0 / n) for c in corners])
        mids = [side(loop[k], loop[(k + 1) % n])[1] for k in range(n)]
        mid_attrs = [self._mix([(corners[k], 0.5), (corners[(k + 1) % n], 0.5)]) for k in range(n)]
        return [
            (
                [loop[k], mids[k], center, mids[k - 1]],
                [corners[k], mid_attrs[k], center_attrs, mid_attrs[k - 1]],
            )
            for k in range(n)
        ]

    @operation
    def loop_cut(self, edge: int, cuts: int = 1, *, factor: float | None = None):
        """Cut new edge loop(s) across the ring of quads through ``edge``.

        The ring runs across consecutive quads (see :meth:`edge_ring`) and
        stops at a boundary, a non-quad face, or where it closes. Each ring
        edge gets ``cuts`` evenly spaced vertices -- or, with ``cuts=1``, one
        vertex at ``factor`` (0..1) measured from the lower-id vertex of
        ``edge`` -- and every quad of the ring is split along them. Faces at
        the ends of an open ring keep their shape and gain the new vertices.
        ``data["edges"]`` are the new loop edges.
        """
        (start_edge,) = self._elements(edge, "edges")
        cuts = _cuts(cuts)
        if factor is not None:
            if cuts != 1:
                raise ValidationError("factor can only be used with cuts=1")
            if not 0.0 < float(factor) < 1.0:
                raise ValidationError("factor must be strictly between 0 and 1")
            params = [float(factor)]
        else:
            params = [k / (cuts + 1) for k in range(1, cuts + 1)]
        ring, closed = self._ring_ordered(start_edge)
        quads = [face for _, face in ring if face is not None]
        if not quads:
            raise EditError(
                "NO_QUAD_RING",
                f"edge {start_edge} has no adjacent quad to cut across",
                elements={"edges": [start_edge]},
                hint="loop cuts need quads; triangulated meshes can use join_triangles first",
            )
        edges = [e for e, _ in ring]
        if len(set(edges)) != len(edges) or len(set(quads)) != len(quads):
            raise EditError(
                "UNSUPPORTED_TOPOLOGY",
                f"the quad ring through edge {start_edge} crosses the same quad or edge twice",
                elements={"edges": sorted(set(edges))},
            )
        # Propagate a consistent start vertex along the ring from the given edge.
        start: dict[int, int] = {start_edge: self._edge_verts[start_edge][0]}
        index = edges.index(start_edge)

        def across(e_from: int, s_from: int, face: int, e_to: int) -> int:
            loop = self._faces[face]
            k = loop.index(s_from)
            n = len(loop)
            nxt, prv = loop[(k + 1) % n], loop[k - 1]
            if self.edge_between(s_from, nxt) == e_from:  # edge runs s -> next; s connects to prev
                return prv
            return nxt

        for i in range(index, len(ring) - 1):
            e, face = ring[i]
            start[edges[i + 1]] = across(e, start[e], face, edges[i + 1])
        for i in range(index - 1, -1, -1):
            e, face = ring[i]
            start[e] = across(edges[i + 1], start[edges[i + 1]], face, e)
        if closed:
            e, face = ring[-1]
            if across(e, start[e], face, edges[0]) != start[edges[0]]:
                raise EditError(
                    "NON_ORIENTABLE_RING",
                    "the quad ring twists (Mobius strip); the cut is ambiguous",
                    elements={"edges": edges},
                )

        originals = {face: (list(self._faces[face]), self._corners_of(face)) for face in quads}
        chains: dict[int, list[int]] = {}
        for e in edges:
            a, b = self._edge_verts[e]
            s = start[e]
            other = b if s == a else a
            chains[e] = [s, *self._split_edge(e, s, params), other]
        new_edges: set[int] = set()
        pieces: dict[int, list[int]] = {}
        for i, (e, face) in enumerate(ring):
            if face is None:
                continue
            e_next = edges[(i + 1) % len(edges)]
            loop, corners = originals[face]
            A, B = chains[e], chains[e_next]
            n = len(loop)
            k = loop.index(A[0])
            forward = loop[(k + 1) % n] == A[-1]
            # Bilinear attribute weights: u along the ring edge, v across the quad.
            pos_in_loop = {v: j for j, v in enumerate(loop)}
            ca0, ca1 = corners[pos_in_loop[A[0]]], corners[pos_in_loop[A[-1]]]
            cb0, cb1 = corners[pos_in_loop[B[0]]], corners[pos_in_loop[B[-1]]]
            ts = [0.0, *params, 1.0]

            def attrs(t, v, ca0=ca0, ca1=ca1, cb0=cb0, cb1=cb1):
                return self._mix(
                    [(ca0, (1 - t) * (1 - v)), (ca1, t * (1 - v)), (cb0, (1 - t) * v), (cb1, t * v)]
                )

            built = []
            for m in range(len(A) - 1):
                if forward:
                    strip = [A[m], A[m + 1], B[m + 1], B[m]]
                    values = [
                        attrs(ts[m], 0),
                        attrs(ts[m + 1], 0),
                        attrs(ts[m + 1], 1),
                        attrs(ts[m], 1),
                    ]
                else:
                    strip = [A[m + 1], A[m], B[m], B[m + 1]]
                    values = [
                        attrs(ts[m + 1], 0),
                        attrs(ts[m], 0),
                        attrs(ts[m], 1),
                        attrs(ts[m + 1], 1),
                    ]
                split = self._split_corners(values)
                if m == 0:
                    self._replace_face(face, strip, split, dirty=False)
                    built.append(face)
                else:
                    built.append(
                        self._new_face(
                            strip,
                            split,
                            smooth=self._smooth[face],
                            group=self._group.get(face, 0),
                            dirty=False,
                        )
                    )
            pieces[face] = built
            for m in range(1, len(A) - 1):
                new_edges.add(self._edge_id(A[m], B[m]))
        return {
            "edges": Selection.of(edges=new_edges),
            "vertices": Selection.of(v for chain in chains.values() for v in chain[1:-1]),
            "faces": pieces,
            "closed": closed,
        }
