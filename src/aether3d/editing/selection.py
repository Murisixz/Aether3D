"""Element selection: by id, geometric predicates, normals, adjacency, loops and rings."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..errors import EditError, ValidationError
from ..math3d import vec3
from . import _polygon as poly
from .core import Elements, Selection

KINDS = ("vertices", "edges", "faces")


class SelectionMixin:
    """Selection queries. All criteria are combined with AND.

    Locations: a vertex's position, an edge's midpoint, a face's center (mean
    of its vertices). For edges and faces, ``test="all"`` or ``"any"`` applies
    the ``box``/``sphere``/``plane`` criteria to their vertices instead.
    ``space="world"`` evaluates locations and normals with the session's
    object world transform. ``expect=n`` raises ``SELECTION_COUNT_MISMATCH``
    unless exactly ``n`` elements match -- a cheap guard against selecting
    the wrong elements. Results are sorted id sets; ordering never matters.
    """

    # -- public queries ------------------------------------------------------

    def select_all(self) -> Selection:
        return Selection.of(self.vertex_ids(), self.edge_ids(), self.face_ids())

    def select_vertices(self, ids: Iterable[int] | None = None, **criteria: Any) -> Selection:
        """Select vertices; see :meth:`select` for criteria."""
        return self.select("vertices", ids, **criteria)

    def select_edges(self, ids: Iterable[int] | None = None, **criteria: Any) -> Selection:
        """Select edges; see :meth:`select` for criteria."""
        return self.select("edges", ids, **criteria)

    def select_faces(self, ids: Iterable[int] | None = None, **criteria: Any) -> Selection:
        """Select faces; see :meth:`select` for criteria."""
        return self.select("faces", ids, **criteria)

    def select(
        self,
        kind: str,
        ids: Iterable[int] | None = None,
        *,
        box: tuple | None = None,
        sphere: tuple | None = None,
        plane: dict | None = None,
        normal: tuple | None = None,
        nearest: Iterable[float] | None = None,
        boundary: bool | None = None,
        sides: int | None = None,
        group: int | None = None,
        predicate: Callable[[int, tuple], bool] | None = None,
        test: str = "center",
        space: str = "local",
        expect: int | None = None,
    ) -> Selection:
        """Select elements of ``kind`` (``"vertices"``, ``"edges"`` or ``"faces"``).

        * ``ids`` -- explicit ids (all must exist).
        * ``box=(min_xyz, max_xyz)`` -- location inside the box (inclusive).
        * ``sphere=(center_xyz, radius)`` -- location within the sphere.
        * ``plane={"point", "normal", "side", "tolerance"}`` -- ``side`` is
          ``"front"`` (default, signed distance > tolerance), ``"back"`` or
          ``"on"`` (|distance| <= tolerance).
        * ``normal=(direction, max_angle_degrees)`` -- faces: face normal;
          vertices/edges: average normal of adjacent faces.
        * ``nearest=xyz`` -- the single nearest element; raises
          ``AMBIGUOUS_SELECTION`` if several are equally near.
        * ``boundary=True/False`` -- on (or off) an open boundary.
        * ``sides=n`` -- faces with exactly ``n`` vertices.
        * ``group=k`` -- faces in face group ``k`` (see :meth:`set_group`).
        * ``predicate(id, location) -> bool`` -- arbitrary Python test.
        """
        if kind not in KINDS:
            raise ValidationError(f"kind must be one of {KINDS}, got {kind!r}")
        if test not in ("center", "all", "any"):
            raise ValidationError(f"test must be 'center', 'all' or 'any', got {test!r}")
        if space not in ("local", "world"):
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        if sides is not None and kind != "faces":
            raise ValidationError("sides= only applies to faces")
        if group is not None and kind != "faces":
            raise ValidationError("group= only applies to faces")
        all_ids = {"vertices": self.vertex_ids, "edges": self.edge_ids, "faces": self.face_ids}[
            kind
        ]()
        if ids is not None:
            candidates = self._require(
                self._ids(ids, kind),
                kind,
                {"vertices": self.has_vertex, "edges": self.has_edge, "faces": self.has_face}[kind],
            )
        else:
            candidates = all_ids
        candidates = np.asarray(candidates, dtype=np.int64)
        to_space = self._space_transform(space)

        def locations(ids: np.ndarray) -> np.ndarray:
            if kind == "vertices":
                pts = self._pos[ids]
            elif kind == "edges":
                pairs = np.array([self._edge_verts[int(e)] for e in ids]).reshape(-1, 2)
                pts = (self._pos[pairs[:, 0]] + self._pos[pairs[:, 1]]) / 2.0
            else:
                pts = np.array([self._pos[self._faces[int(f)]].mean(axis=0) for f in ids]).reshape(
                    -1, 3
                )
            return to_space(pts)

        def element_vertices(i: int) -> list[int]:
            if kind == "vertices":
                return [i]
            if kind == "edges":
                return list(self._edge_verts[i])
            return list(self._faces[i])

        def spatial(test_points: Callable[[np.ndarray], np.ndarray]) -> None:
            nonlocal candidates
            if len(candidates) == 0:
                return
            if test == "center" or kind == "vertices":
                keep = test_points(locations(candidates))
            else:
                combine = np.all if test == "all" else np.any
                keep = np.array(
                    [
                        bool(combine(test_points(to_space(self._pos[element_vertices(int(i))]))))
                        for i in candidates
                    ]
                )
            candidates = candidates[keep]

        if box is not None:
            lo, hi = (np.asarray(vec3(b, "box corner")) for b in box)
            spatial(lambda p: np.all((p >= lo) & (p <= hi), axis=1))
        if sphere is not None:
            center, radius = sphere
            c, r = np.asarray(vec3(center, "sphere center")), float(radius)
            spatial(lambda p: np.linalg.norm(p - c, axis=1) <= r)
        if plane is not None:
            point = np.asarray(vec3(plane.get("point", (0, 0, 0)), "plane point"))
            n = poly.unit(np.asarray(vec3(plane["normal"], "plane normal")))
            if n is None:
                raise ValidationError("plane normal must be non-zero")
            side = plane.get("side", "front")
            tol = float(plane.get("tolerance", 1e-6 * self._scale()))
            if side not in ("front", "back", "on"):
                raise ValidationError(f"plane side must be front/back/on, got {side!r}")

            def plane_test(p: np.ndarray) -> np.ndarray:
                d = (p - point) @ n
                return (
                    d > tol
                    if side == "front"
                    else (d < -tol if side == "back" else np.abs(d) <= tol)
                )

            spatial(plane_test)
        if normal is not None:
            direction, max_angle = normal
            d = poly.unit(np.asarray(vec3(direction, "normal direction")))
            if d is None:
                raise ValidationError("normal direction must be non-zero")
            cos_limit = math.cos(math.radians(float(max_angle)))
            normal_space = self._normal_transform(space)
            keep = []
            for i in candidates:
                en = self._element_normal(kind, int(i))
                en = poly.unit(normal_space(en)) if en is not None else None
                keep.append(en is not None and float(np.dot(en, d)) >= cos_limit - 1e-12)
            candidates = candidates[np.array(keep, dtype=bool)] if len(candidates) else candidates
        if boundary is not None:
            keep = [self._on_boundary(kind, int(i)) == boundary for i in candidates]
            candidates = candidates[np.array(keep, dtype=bool)] if len(candidates) else candidates
        if sides is not None:
            candidates = np.array(
                [f for f in candidates if len(self._faces[int(f)]) == int(sides)], dtype=np.int64
            )
        if group is not None:
            candidates = np.array(
                [f for f in candidates if self._group.get(int(f), 0) == int(group)], dtype=np.int64
            )
        if predicate is not None:
            locs = locations(candidates) if len(candidates) else np.zeros((0, 3))
            keep = [
                bool(predicate(int(i), tuple(locs[k].tolist()))) for k, i in enumerate(candidates)
            ]
            candidates = candidates[np.array(keep, dtype=bool)] if len(candidates) else candidates
        if nearest is not None and len(candidates):
            target = np.asarray(vec3(nearest, "nearest point"))
            dist = np.linalg.norm(locations(candidates) - target, axis=1)
            best = float(dist.min())
            tied = candidates[dist <= best + 1e-9 * max(self._scale(), 1.0)]
            if len(tied) > 1:
                raise EditError(
                    "AMBIGUOUS_SELECTION",
                    f"{len(tied)} {kind} are equally near {tuple(target.tolist())}",
                    elements={kind: [int(t) for t in tied]},
                    hint="add a criterion or use explicit ids",
                )
            candidates = tied
        result = Selection(**{kind: frozenset(int(i) for i in candidates)})
        if expect is not None and len(candidates) != int(expect):
            raise EditError(
                "SELECTION_COUNT_MISMATCH",
                f"expected {expect} {kind} but the criteria matched {len(candidates)}",
                elements={kind: [int(i) for i in candidates]},
            )
        return result

    # -- conversions -----------------------------------------------------------

    def to_vertices(self, selection: Selection) -> Selection:
        """All vertices used by the selected elements."""
        return Selection.of(vertices=self._vertices_of(selection))

    def to_edges(self, selection: Selection) -> Selection:
        """Selected edges, plus all edges of selected faces and edges between selected vertices."""
        edges = set(self._require_edges(selection.edges))
        for f in self._require_faces(selection.faces):
            edges.update(self._edge_id(a, b) for a, b in self._loop_edges(self._faces[f]))
        verts = set(self._require_vertices(selection.vertices))
        for v in verts:
            for e in self._vert_edges[v]:
                a, b = self._edge_verts[e]
                if a in verts and b in verts:
                    edges.add(e)
        return Selection.of(edges=edges)

    def to_faces(self, selection: Selection) -> Selection:
        """Selected faces, plus faces whose vertices are all selected (directly or via edges)."""
        faces = set(self._require_faces(selection.faces))
        verts = set(self._require_vertices(selection.vertices))
        for e in self._require_edges(selection.edges):
            verts.update(self._edge_verts[e])
        for v in verts:
            for f in self._faces_of_vertex(v):
                if all(u in verts for u in self._faces[f]):
                    faces.add(f)
        return Selection.of(faces=faces)

    # -- adjacency -------------------------------------------------------------

    def linked_faces(self, faces: Elements) -> Selection:
        """All faces connected to ``faces`` through shared edges."""
        seen = set(self._elements(faces, "faces"))
        stack = list(seen)
        while stack:
            f = stack.pop()
            for key in self._loop_edges(self._faces[f]):
                for g in self._edge_faces[self._edge_by_pair[key]]:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        return Selection.of(faces=seen)

    def grow(self, selection: Selection) -> Selection:
        """Add every element adjacent (through a shared vertex) to the selection."""
        verts = set(self._vertices_of(selection))
        faces = set()
        for v in verts:
            faces |= self._faces_of_vertex(v)
        out = Selection.of(faces=faces)
        if selection.vertices:
            out = out | Selection.of(vertices={u for f in faces for u in self._faces[f]})
        if selection.edges:
            out = out | self.to_edges(Selection.of(faces=faces))
        return out if selection.faces else Selection(out.vertices, out.edges, frozenset())

    def region_boundary(self, faces: Elements) -> Selection:
        """Edges used by exactly one of the given faces (the region's outline)."""
        counts: dict[int, int] = {}
        for f in self._elements(faces, "faces"):
            for key in self._loop_edges(self._faces[f]):
                e = self._edge_by_pair[key]
                counts[e] = counts.get(e, 0) + 1
        return Selection.of(edges=[e for e, c in counts.items() if c == 1])

    def boundary_edges(self) -> Selection:
        """Edges with exactly one face (open boundaries of the mesh)."""
        return Selection.of(edges=[e for e, f in self._edge_faces.items() if len(f) == 1])

    def non_manifold_edges(self) -> Selection:
        return Selection.of(edges=[e for e, f in self._edge_faces.items() if len(f) > 2])

    def edge_loop(self, edge: int) -> Selection:
        """The edge loop through ``edge``.

        Interior loops continue straight through vertices with exactly four
        edges (taking the edge that shares no face with the current one) and
        stop at poles, boundaries or where they close. A boundary edge yields
        its boundary loop instead.
        """
        return Selection.of(edges=self._edge_loop_ordered(edge))

    def edge_ring(self, edge: int) -> Selection:
        """The ring of edges opposite ``edge`` across consecutive quads."""
        return Selection.of(edges=[e for e, _ in self._ring_ordered(edge)[0]])

    # -- helpers ---------------------------------------------------------------

    def _space_transform(self, space: str) -> Callable[[np.ndarray], np.ndarray]:
        if space == "local":
            return lambda p: p
        m = self.world_matrix
        return lambda p: p @ m[:3, :3].T + m[:3, 3]

    def _normal_transform(self, space: str) -> Callable[[np.ndarray], np.ndarray]:
        if space == "local":
            return lambda n: n
        linear = self.world_matrix[:3, :3]
        try:
            inv_t = np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            raise EditError(
                "SINGULAR_WORLD_TRANSFORM",
                "the object's world transform is singular; world-space normals are undefined",
            ) from None
        return lambda n: inv_t @ n

    def _element_normal(self, kind: str, i: int) -> np.ndarray | None:
        if kind == "faces":
            return self._face_normal_vec(i)
        faces = self._faces_of_vertex(i) if kind == "vertices" else self._edge_faces[i]
        total = np.zeros(3)
        for f in faces:
            total += poly.newell_normal(self._pos[self._faces[f]])  # area weighted
        return poly.unit(total)

    def _on_boundary(self, kind: str, i: int) -> bool:
        if kind == "edges":
            return len(self._edge_faces[i]) == 1
        if kind == "vertices":
            return any(len(self._edge_faces[e]) == 1 for e in self._vert_edges[i])
        return any(
            len(self._edge_faces[self._edge_by_pair[key]]) == 1
            for key in self._loop_edges(self._faces[i])
        )

    def _edge_loop_ordered(self, edge: int) -> list[int]:
        start = self._require_edges([edge])[0]
        boundary = len(self._edge_faces[start]) == 1

        def step(e: int, v: int) -> int | None:
            edges_at_v = self._vert_edges[v]
            if boundary:
                options = [x for x in edges_at_v if x != e and len(self._edge_faces[x]) == 1]
                return options[0] if len(options) == 1 else None
            if len(edges_at_v) != 4 or any(len(self._edge_faces[x]) != 2 for x in edges_at_v):
                return None
            faces_e = set(self._edge_faces[e])
            options = [x for x in edges_at_v if x != e and not faces_e & set(self._edge_faces[x])]
            return options[0] if len(options) == 1 else None

        loop = [start]
        seen = {start}
        a, b = self._edge_verts[start]
        for direction_vertex, prepend in ((b, False), (a, True)):
            e, v = start, direction_vertex
            while True:
                nxt = step(e, v)
                if nxt is None:
                    break
                if nxt in seen:  # closed loop
                    return loop
                seen.add(nxt)
                (loop.insert(0, nxt) if prepend else loop.append(nxt))
                x, y = self._edge_verts[nxt]
                e, v = nxt, (y if x == v else x)
        return loop

    def _ring_ordered(self, edge: int) -> tuple[list[tuple[int, int | None]], bool]:
        """Ring edges in order as ``(edge, face_after)`` pairs, and whether it closes.

        ``face_after`` is the quad between this edge and the next one (``None``
        for the last edge of an open ring).
        """
        start = self._require_edges([edge])[0]

        def walk(e: int, face: int) -> tuple[list[tuple[int, int]], bool]:
            out: list[tuple[int, int]] = []
            seen = {start}
            while True:
                loop = self._faces[face]
                if len(loop) != 4 or len(self._edge_faces[e]) > 2:
                    return out, False
                keys = self._loop_edges(loop)
                k = keys.index(self._edge_verts[e])
                opposite = self._edge_by_pair[keys[(k + 2) % 4]]
                if opposite in seen and opposite != start:
                    return out, False  # the ring folds back onto itself: stop here
                seen.add(opposite)
                out.append((face, opposite))
                if opposite == start:
                    return out, True
                faces = self._edge_faces[opposite]
                if len(faces) != 2:
                    return out, False
                face = faces[0] if faces[1] == face else faces[1]
                e = opposite

        faces = list(self._edge_faces[start])
        if not faces:
            return [(start, None)], False
        forward, closed = walk(start, faces[0])
        if closed:
            ring = [(start, forward[0][0])]
            for i, (_face, opposite) in enumerate(forward[:-1]):
                ring.append((opposite, forward[i + 1][0]))
            return ring, True
        backward: list[tuple[int, int]] = []
        if len(faces) == 2:
            backward, _ = walk(start, faces[1])
        # Assemble: ... b2 -[f]- b1 -[f]- start -[f]- f1 -[f]- f2 ...
        ring: list[tuple[int, int | None]] = []
        back_edges = [opp for _, opp in backward]
        back_faces = [face for face, _ in backward]
        for i in range(len(back_edges) - 1, -1, -1):
            ring.append((back_edges[i], back_faces[i]))
        if forward:
            ring.append((start, forward[0][0]))
            for i, (_, opposite) in enumerate(forward):
                ring.append((opposite, forward[i + 1][0] if i + 1 < len(forward) else None))
        else:
            ring.append((start, None))
        return ring, False
