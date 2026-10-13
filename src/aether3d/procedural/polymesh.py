"""PolyMesh: the immutable polygon-mesh value that flows through procedural graphs.

A procedural chain passes polygons (not triangles) from modifier to modifier,
so operations such as Catmull-Clark subdivision or bevel see the real quads
and n-gons. The layout is fully vectorized (compressed sparse rows):

* ``positions``     -- ``(V, 3) float64``
* ``face_offsets``  -- ``(F + 1,) int64``; face ``f`` owns corners
  ``face_offsets[f]:face_offsets[f + 1]``
* ``face_vertices`` -- ``(C,) int64``; the vertex of each corner, counter-
  clockwise seen from the front of its face (Aether3D's winding convention)
* ``uvs``           -- ``(C, 2)`` per-corner texture coordinates, or ``None``
* ``normals``       -- ``(C, 3)`` per-corner normals, or ``None``
* ``normals_valid`` -- ``(F,) bool``: faces whose stored corner normals are
  still exact (transforms keep them valid; reshaping operations clear them)
* ``smooth``        -- ``(F,) bool`` shading flag
* ``sharp_edges``   -- ``(S, 2) int64`` vertex pairs (``a < b``) marked sharp

When a PolyMesh is converted to a triangle :class:`~aether3d.Mesh`, normals of
faces whose stored normals are no longer valid are recomputed with the same
flat / smooth / sharp-edge shading model as the mesh editing engine.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..diagnostics import ERROR, INFO, MAX_SAMPLES, WARNING, Diagnostic
from ..editing import _polygon as poly
from ..editing.validation import TopologyReport
from ..mesh import Mesh


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


class PolyMeshError(ValueError):
    """Raised for structurally invalid PolyMesh data (an internal invariant)."""


class PolyMesh:
    __slots__ = (
        "positions",
        "face_offsets",
        "face_vertices",
        "uvs",
        "normals",
        "normals_valid",
        "smooth",
        "sharp_edges",
        "_derived",
    )

    def __init__(
        self,
        positions: Any,
        face_offsets: Any,
        face_vertices: Any,
        *,
        uvs: Any | None = None,
        normals: Any | None = None,
        normals_valid: Any | None = None,
        smooth: Any | None = None,
        sharp_edges: Any | None = None,
    ) -> None:
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        offsets = np.asarray(face_offsets, dtype=np.int64).reshape(-1)
        fv = np.asarray(face_vertices, dtype=np.int64).reshape(-1)
        if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != len(fv):
            raise PolyMeshError("face_offsets must start at 0 and end at the corner count")
        sizes = np.diff(offsets)
        if np.any(sizes < 3):
            raise PolyMeshError("every face needs at least 3 corners")
        if len(fv) and (fv.min() < 0 or fv.max() >= len(pos)):
            raise PolyMeshError("face vertex index out of range")
        f = len(sizes)
        c = len(fv)
        self.positions = _readonly(pos)
        self.face_offsets = _readonly(offsets)
        self.face_vertices = _readonly(fv)
        self.uvs = None if uvs is None else _readonly(np.asarray(uvs, np.float64).reshape(c, 2))
        self.normals = (
            None if normals is None else _readonly(np.asarray(normals, np.float64).reshape(c, 3))
        )
        if self.normals is None:
            valid = np.zeros(f, dtype=bool)
        elif normals_valid is None:
            valid = np.ones(f, dtype=bool)
        else:
            valid = np.asarray(normals_valid, dtype=bool).reshape(f)
        self.normals_valid = _readonly(valid)
        self.smooth = _readonly(
            np.zeros(f, bool) if smooth is None else np.asarray(smooth, bool).reshape(f)
        )
        sharp = np.zeros((0, 2), np.int64) if sharp_edges is None else np.asarray(sharp_edges)
        sharp = np.sort(sharp.astype(np.int64).reshape(-1, 2), axis=1)
        self.sharp_edges = _readonly(np.unique(sharp, axis=0) if len(sharp) else sharp)
        self._derived: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def face_count(self) -> int:
        return len(self.face_offsets) - 1

    @property
    def corner_count(self) -> int:
        return len(self.face_vertices)

    @property
    def face_sizes(self) -> np.ndarray:
        return self._get("sizes", lambda: np.diff(self.face_offsets))

    @property
    def corner_face(self) -> np.ndarray:
        return self._get(
            "corner_face", lambda: np.repeat(np.arange(self.face_count), self.face_sizes)
        )

    @property
    def next_corner(self) -> np.ndarray:
        def build():
            cf = self.corner_face
            local = np.arange(self.corner_count) - self.face_offsets[cf]
            return self.face_offsets[cf] + (local + 1) % self.face_sizes[cf]

        return self._get("next", build)

    @property
    def prev_corner(self) -> np.ndarray:
        def build():
            cf = self.corner_face
            local = np.arange(self.corner_count) - self.face_offsets[cf]
            return self.face_offsets[cf] + (local - 1) % self.face_sizes[cf]

        return self._get("prev", build)

    def _get(self, key: str, build):
        if key not in self._derived:
            self._derived[key] = build()
        return self._derived[key]

    def with_(self, **changes: Any) -> PolyMesh:
        """A copy with some arrays replaced (others shared; arrays are read-only)."""
        fields = {
            "positions": self.positions,
            "face_offsets": self.face_offsets,
            "face_vertices": self.face_vertices,
            "uvs": self.uvs,
            "normals": self.normals,
            "normals_valid": self.normals_valid,
            "smooth": self.smooth,
            "sharp_edges": self.sharp_edges,
        }
        fields.update(changes)
        return PolyMesh(
            fields.pop("positions"),
            fields.pop("face_offsets"),
            fields.pop("face_vertices"),
            **fields,
        )

    @classmethod
    def empty(cls, uvs: bool = True, normals: bool = True) -> PolyMesh:
        return cls(
            np.zeros((0, 3)),
            [0],
            [],
            uvs=np.zeros((0, 2)) if uvs else None,
            normals=np.zeros((0, 3)) if normals else None,
        )

    @classmethod
    def from_faces(
        cls,
        positions: Any,
        faces: list[list[int]],
        *,
        uvs: list[list[tuple]] | None = None,
        normals: list[list[tuple]] | None = None,
        smooth: Any = None,
        sharp_edges: Any = None,
    ) -> PolyMesh:
        """Build from a list of vertex loops (and optional per-face corner lists)."""
        sizes = [len(f) for f in faces]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        fv = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(offsets[-1]))
        flat_uv = None if uvs is None else np.array([uv for f in uvs for uv in f], np.float64)
        flat_n = None if normals is None else np.array([n for f in normals for n in f], np.float64)
        return cls(
            positions,
            offsets,
            fv,
            uvs=flat_uv,
            normals=flat_n,
            smooth=smooth,
            sharp_edges=sharp_edges,
        )

    # ------------------------------------------------------------------ #
    # Topology (vectorized)
    # ------------------------------------------------------------------ #

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(edge_vertices (E, 2), corner_edge (C,), edge_face_count (E,))``.

        Corner ``c`` owns the edge from its vertex to the next corner's vertex.
        """

        def build():
            a = self.face_vertices
            b = self.face_vertices[self.next_corner]
            keys = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
            if len(keys) == 0:
                return np.zeros((0, 2), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
            uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            return uniq, inverse.reshape(-1), counts

        return self._get("edges", build)

    @property
    def face_normals(self) -> np.ndarray:
        """``(F, 3)`` unnormalized Newell area vectors (length = 2 x face area)."""

        def build():
            if self.face_count == 0:
                return np.zeros((0, 3))
            p = self.positions[self.face_vertices]
            q = self.positions[self.face_vertices[self.next_corner]]
            terms = np.stack(
                [
                    (p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2]),
                    (p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0]),
                    (p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]),
                ],
                axis=1,
            )
            return np.add.reduceat(terms, self.face_offsets[:-1], axis=0)

        return self._get("face_normals", build)

    @property
    def face_areas(self) -> np.ndarray:
        return self._get("face_areas", lambda: np.linalg.norm(self.face_normals, axis=1) / 2.0)

    def scale(self) -> float:
        if self.vertex_count == 0:
            return 1.0
        return max(float(np.max(np.ptp(self.positions, axis=0))), 1e-12)

    def area_eps(self) -> float:
        return 1e-12 * self.scale() ** 2

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.vertex_count == 0:
            return None
        used = np.unique(self.face_vertices) if self.face_count else np.arange(self.vertex_count)
        p = self.positions[used] if len(used) else self.positions
        return p.min(axis=0), p.max(axis=0)

    # ------------------------------------------------------------------ #
    # Normals (shading model shared with the mesh editing engine)
    # ------------------------------------------------------------------ #

    def corner_normals(self) -> np.ndarray:
        """``(C, 3)`` normals: stored where valid, recomputed elsewhere.

        Recomputation follows the flat/smooth/sharp model: a flat face's corners
        use the face normal; a smooth corner uses the angle-weighted average of
        the smooth faces around its vertex reachable without crossing a sharp,
        boundary or non-manifold edge. Smooth corners whose fan contains any
        face with invalid normals are recomputed as a whole (so a reshaped face
        never leaves a stale neighbour normal behind).
        """
        c = self.corner_count
        if c == 0:
            return np.zeros((0, 3))
        fn = self.face_normals
        length = np.linalg.norm(fn, axis=1)
        unit = np.divide(fn, length[:, None], out=np.zeros_like(fn), where=length[:, None] > 1e-300)
        cf = self.corner_face
        valid_faces = (
            self.normals_valid if self.normals is not None else np.zeros(self.face_count, bool)
        )
        result = np.zeros((c, 3))
        if self.normals is not None:
            result[:] = self.normals
        # Smooth groups: corners at the same vertex joined across manifold, non-sharp edges
        # between two smooth faces (vectorized label propagation).
        edge_vertices, corner_edge, edge_counts = self.edges
        sharp = np.zeros(len(edge_vertices), dtype=bool)
        if len(self.sharp_edges) and len(edge_vertices):
            sharp_keys = self.sharp_edges[:, 0] * (self.vertex_count + 1) + self.sharp_edges[:, 1]
            edge_keys = edge_vertices[:, 0] * (self.vertex_count + 1) + edge_vertices[:, 1]
            sharp = np.isin(edge_keys, sharp_keys)
        smooth_corner = self.smooth[cf]
        joinable = (edge_counts[corner_edge] == 2) & ~sharp[corner_edge]
        # A 2-face edge owned by corners c1 and c2 links the corners of its two faces
        # that sit at the same vertex: (c1, next(c2)) and (next(c1), c2) when the faces
        # are wound consistently, (c1, c2) and (next(c1), next(c2)) when they are not.
        labels = np.arange(c)
        a, b = _paired_corners(self, joinable & smooth_corner)
        labels = _propagate(labels, a, b)
        # Angle weights.
        p = self.positions[self.face_vertices]
        u = self.positions[self.face_vertices[self.prev_corner]] - p
        w = self.positions[self.face_vertices[self.next_corner]] - p
        nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
        cosang = np.einsum("ij,ij->i", u, w) / np.maximum(nu * nw, 1e-300)
        angle = np.arccos(np.clip(cosang, -1.0, 1.0))
        contrib = unit[cf] * angle[:, None]
        sums = np.zeros((c, 3))
        np.add.at(sums, labels, contrib)
        group_invalid = np.zeros(c, dtype=bool)
        np.logical_or.at(group_invalid, labels, ~valid_faces[cf])
        smooth_values = sums[labels]
        smooth_len = np.linalg.norm(smooth_values, axis=1)
        smooth_unit = np.divide(
            smooth_values,
            smooth_len[:, None],
            out=np.zeros_like(smooth_values),
            where=smooth_len[:, None] > 1e-300,
        )
        flat = ~smooth_corner
        recompute_flat = flat & ~valid_faces[cf]
        recompute_smooth = smooth_corner & group_invalid[labels]
        result[recompute_flat] = unit[cf][recompute_flat]
        result[recompute_smooth] = smooth_unit[recompute_smooth]
        # Fallback for zero-area faces / cancelled fans.
        bad = np.linalg.norm(result, axis=1) < 1e-12
        if bad.any():
            result[bad] = (0.0, 0.0, 1.0)
        return result

    # ------------------------------------------------------------------ #
    # Conversion to and from the triangle Mesh / EditableMesh
    # ------------------------------------------------------------------ #

    def triangulate(self) -> tuple[np.ndarray, list[Diagnostic]]:
        """``(T, 3)`` corner-index triangles preserving winding; plus diagnostics.

        Triangles stay as they are; quads are split along the diagonal that
        keeps both halves front-facing (the shorter one when both do); larger
        polygons use ear clipping. Zero-area faces are fan-split and reported;
        a non-degenerate polygon that cannot be triangulated (self-intersecting)
        raises :class:`PolyMeshError`.
        """
        sizes = self.face_sizes
        off = self.face_offsets[:-1]
        out: list[np.ndarray] = []
        diagnostics: list[Diagnostic] = []
        tri = np.flatnonzero(sizes == 3)
        out.append(np.stack([off[tri], off[tri] + 1, off[tri] + 2], axis=1))
        quad = np.flatnonzero(sizes == 4)
        if len(quad):
            c0 = off[quad]
            c = [c0, c0 + 1, c0 + 2, c0 + 3]
            p = [self.positions[self.face_vertices[k]] for k in c]
            qn = self.face_normals[quad]

            def tri_ok(i, j, k):
                n = np.cross(p[j] - p[i], p[k] - p[i])
                return np.einsum("ij,ij->i", n, qn) > 0

            split_a = tri_ok(0, 1, 2) & tri_ok(0, 2, 3)  # diagonal 0-2
            split_b = tri_ok(1, 2, 3) & tri_ok(1, 3, 0)  # diagonal 1-3
            d02 = np.linalg.norm(p[2] - p[0], axis=1)
            d13 = np.linalg.norm(p[3] - p[1], axis=1)
            use_b = split_b & (~split_a | (d13 < d02 * (1 - 1e-12)))
            ta = np.stack([c[0], c[1], c[2], c[0], c[2], c[3]], axis=1).reshape(-1, 3)
            tb = np.stack([c[1], c[2], c[3], c[1], c[3], c[0]], axis=1).reshape(-1, 3)
            chosen = np.where(np.repeat(use_b, 2)[:, None], tb, ta)
            out.append(chosen)
            degenerate = ~split_a & ~split_b
            if degenerate.any():
                diagnostics.append(self._degenerate_diag(quad[degenerate]))
        big = np.flatnonzero(sizes > 4)
        eps = self.area_eps()
        degenerate_big = []
        for f in big:
            s, e = int(self.face_offsets[f]), int(self.face_offsets[f + 1])
            corners = np.arange(s, e)
            if self.face_areas[f] <= eps:
                degenerate_big.append(f)
                out.append(np.stack([np.full(e - s - 2, s), corners[1:-1], corners[2:]], axis=1))
                continue
            try:
                local = poly.triangulate(self.positions[self.face_vertices[s:e]])
            except ValueError as exc:
                raise PolyMeshError(f"face {f} cannot be triangulated: {exc}") from None
            out.append(corners[np.asarray(local, dtype=np.int64)])
        if degenerate_big:
            diagnostics.append(self._degenerate_diag(np.asarray(degenerate_big)))
        triangles = np.concatenate(out) if out else np.zeros((0, 3), np.int64)
        return triangles.astype(np.int64).reshape(-1, 3), diagnostics

    @staticmethod
    def _degenerate_diag(faces: np.ndarray) -> Diagnostic:
        return Diagnostic(
            "ZERO_AREA_FACES",
            WARNING,
            "zero-area polygons were split into zero-area triangles",
            "faces",
            len(faces),
            tuple(int(f) for f in faces[:MAX_SAMPLES]),
        )

    def to_mesh(self) -> tuple[Mesh, list[Diagnostic]]:
        """Triangulate into an immutable :class:`Mesh` (deterministic vertex order)."""
        triangles, diagnostics = self.triangulate()
        if self.corner_count == 0:
            return (
                Mesh(
                    np.zeros((0, 3)),
                    np.zeros((0, 3), np.int64),
                    normals=np.zeros((0, 3)) if self.normals is not None else None,
                    uvs=np.zeros((0, 2)) if self.uvs is not None else None,
                ),
                diagnostics,
            )
        cols = [self.face_vertices[:, None].astype(np.float64)]
        normals = self.corner_normals() if self.normals is not None else None
        if normals is not None:
            normals = normals.astype(np.float32).astype(np.float64)  # dedupe at output precision
            cols.append(normals)
        uvs = self.uvs.astype(np.float32).astype(np.float64) if self.uvs is not None else None
        if uvs is not None:
            cols.append(uvs)
        rows = np.ascontiguousarray(np.hstack(cols))
        _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(first, kind="stable")  # first-use order: deterministic, stable
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        corner_to_vertex = rank[inverse]
        firsts = first[order]
        mesh = Mesh(
            self.positions[self.face_vertices[firsts]],
            corner_to_vertex[triangles],
            normals=normals[firsts] if normals is not None else None,
            uvs=uvs[firsts] if uvs is not None else None,
        )
        return mesh, diagnostics

    @classmethod
    def from_mesh(cls, mesh: Mesh, *, quads: bool = True) -> PolyMesh:
        """Polygon form of a triangle mesh (exact welding; quads rebuilt as when editing)."""
        from ..editing import EditableMesh

        return cls.from_editable(EditableMesh.from_mesh(mesh, quads=quads))

    @classmethod
    def from_editable(cls, em) -> PolyMesh:
        """Snapshot of an :class:`~aether3d.editing.EditableMesh` (ids are compacted)."""
        fids = sorted(em._faces)
        used = sorted({v for f in fids for v in em._faces[f]})
        remap = np.full(em._nv, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        loops = [em._faces[f] for f in fids]
        sizes = [len(loop) for loop in loops]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        fv = remap[np.fromiter((v for loop in loops for v in loop), np.int64, int(offsets[-1]))]
        uvs = normals = None
        if "uv" in em._layers:
            data = em._layers["uv"].data
            uvs = np.array([uv for f in fids for uv in data[f]], np.float64).reshape(-1, 2)
        if "normal" in em._layers:
            data = em._layers["normal"].data
            normals = np.array([n for f in fids for n in data[f]], np.float64).reshape(-1, 3)
        sharp = [em._edge_verts[e] for e in em._sharp]
        sharp_arr = remap[np.array(sharp, np.int64).reshape(-1, 2)] if sharp else None
        return cls(
            em._pos[used] if used else np.zeros((0, 3)),
            offsets,
            fv,
            uvs=uvs,
            normals=normals,
            normals_valid=np.array([f not in em._dirty_faces for f in fids], bool),
            smooth=np.array([em._smooth.get(f, False) for f in fids], bool),
            sharp_edges=sharp_arr,
        )

    def to_editable(self):
        """An :class:`~aether3d.editing.EditableMesh` with the same polygons."""
        from ..editing import EditableMesh

        em = EditableMesh(uvs=self.uvs is not None, normals=self.normals is not None)
        em._bulk_vertices(self.positions)
        uv = self.uvs.tolist() if self.uvs is not None else None
        nr = self.normals.tolist() if self.normals is not None else None
        offsets = self.face_offsets.tolist()
        fv = self.face_vertices.tolist()
        smooth = self.smooth.tolist()
        valid = self.normals_valid.tolist()
        for f in range(self.face_count):
            s, e = offsets[f], offsets[f + 1]
            corners = {}
            if uv is not None:
                corners["uv"] = uv[s:e]
            if nr is not None:
                corners["normal"] = nr[s:e]
            em._new_face(fv[s:e], corners, smooth=smooth[f], dirty=not valid[f])
        for a, b in self.sharp_edges.tolist():
            e = em.edge_between(a, b)
            if e is not None:
                em._sharp.add(e)
        return em

    # ------------------------------------------------------------------ #
    # Validation (vectorized; same codes as the editing validator)
    # ------------------------------------------------------------------ #

    def validate(self) -> TopologyReport:
        """Topology checks on the polygon mesh (see ``docs/PROCEDURAL.md``)."""
        diagnostics: list[Diagnostic] = []

        def add(code: str, severity: str, message: str, kind: str, ids: np.ndarray) -> None:
            ids = np.asarray(ids)
            if len(ids):
                diagnostics.append(
                    Diagnostic(
                        code,
                        severity,
                        message,
                        kind,
                        int(len(ids)),
                        tuple(int(i) for i in ids[:MAX_SAMPLES]),
                    )
                )

        add(
            "NON_FINITE_POSITIONS",
            ERROR,
            "vertex positions contain NaN or infinity",
            "vertices",
            np.flatnonzero(~np.all(np.isfinite(self.positions), axis=1)),
        )
        add(
            "ZERO_AREA_FACES",
            WARNING,
            "faces have (near-)zero area",
            "faces",
            np.flatnonzero(self.face_areas <= self.area_eps()),
        )
        edge_vertices, corner_edge, counts = self.edges
        add(
            "BOUNDARY_EDGES",
            INFO,
            "edges with one face (open boundary)",
            "edges",
            np.flatnonzero(counts == 1),
        )
        add(
            "NON_MANIFOLD_EDGES",
            WARNING,
            "edges shared by more than two faces",
            "edges",
            np.flatnonzero(counts > 2),
        )
        # Winding: on a 2-face edge the two owning corners must run opposite ways.
        inconsistent = np.zeros(0, np.int64)
        if len(corner_edge):
            order = np.argsort(corner_edge, kind="stable")
            sorted_edges = corner_edge[order]
            start = np.flatnonzero(np.r_[True, sorted_edges[1:] != sorted_edges[:-1]])
            group = np.diff(np.r_[start, len(order)])
            two = start[group == 2]
            c1, c2 = order[two], order[two + 1]
            same = self.face_vertices[c1] == self.face_vertices[c2]
            inconsistent = corner_edge[c1[same]]
        add(
            "INCONSISTENT_WINDING",
            WARNING,
            "the two faces of these edges are wound the same way",
            "edges",
            inconsistent,
        )
        used = np.zeros(self.vertex_count, bool)
        used[self.face_vertices] = True
        add(
            "LOOSE_VERTICES",
            INFO,
            "vertices not used by any face",
            "vertices",
            np.flatnonzero(~used),
        )
        nm_vertices = self._non_manifold_vertices()
        add(
            "NON_MANIFOLD_VERTICES",
            WARNING,
            "faces around these vertices form more than one fan",
            "vertices",
            nm_vertices,
        )
        f = self.face_count
        summary = {
            "vertex_count": int(used.sum()),
            "edge_count": int(len(edge_vertices)),
            "face_count": f,
            "face_sizes": {
                str(k) if k < 5 else "5+": int(v)
                for k, v in sorted(_size_histogram(self.face_sizes).items())
            },
            "triangle_count": int(np.sum(self.face_sizes - 2)),
            "boundary_edge_count": int(np.sum(counts == 1)),
            "non_manifold_edge_count": int(np.sum(counts > 2)),
            "closed": bool(f) and not np.any(counts == 1),
            "edge_manifold": not np.any(counts > 2),
            "vertex_manifold": len(nm_vertices) == 0,
            "consistently_oriented": len(inconsistent) == 0,
            "euler_characteristic": int(used.sum()) - int(len(edge_vertices)) + f,
        }
        return TopologyReport(summary, diagnostics)

    def _non_manifold_vertices(self) -> np.ndarray:
        c = self.corner_count
        if c == 0:
            return np.zeros(0, np.int64)
        _, corner_edge, counts = self.edges
        a, b = _paired_corners(self, counts[corner_edge] == 2)
        labels = _propagate(np.arange(c), a, b)
        pairs = np.unique(np.stack([self.face_vertices, labels], axis=1), axis=0)
        fans = np.bincount(pairs[:, 0], minlength=self.vertex_count)
        return np.flatnonzero(fans > 1)

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    def content_hash(self) -> str:
        """SHA-256 of all arrays (stable across runs and platforms)."""

        def build():
            h = hashlib.sha256()
            for arr in (
                self.positions,
                self.face_offsets,
                self.face_vertices,
                self.smooth,
                self.normals_valid,
                self.sharp_edges,
            ):
                h.update(np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes())
            for arr in (self.uvs, self.normals):
                h.update(b"-" if arr is None else np.ascontiguousarray(arr).astype("<f8").tobytes())
            return h.hexdigest()

        return self._get("hash", build)

    def __repr__(self) -> str:
        return f"PolyMesh(vertices={self.vertex_count}, faces={self.face_count})"


def _paired_corners(pm: PolyMesh, usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner pairs at the same vertex across 2-face edges whose owning corner is ``usable``."""
    _, corner_edge, _ = pm.edges
    if len(corner_edge) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(corner_edge, kind="stable")
    sorted_edges = corner_edge[order]
    start = np.flatnonzero(np.r_[True, sorted_edges[1:] != sorted_edges[:-1]])
    group = np.diff(np.r_[start, len(order)])
    two = start[group == 2]
    c1, c2 = order[two], order[two + 1]
    ok = usable[c1] & usable[c2]
    c1, c2 = c1[ok], c2[ok]
    n1, n2 = pm.next_corner[c1], pm.next_corner[c2]
    same = pm.face_vertices[c1] == pm.face_vertices[c2]  # inconsistently wound pair
    a = np.concatenate([c1, n1])
    b = np.concatenate([np.where(same, c2, n2), np.where(same, n2, c2)])
    return a, b


def _propagate(labels: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component labels (minimum member) by vectorized label propagation."""
    if len(a) == 0:
        return labels
    while True:
        m = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, labels):
            return labels
        labels = new


def _size_histogram(sizes: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    values, counts = np.unique(np.minimum(sizes, 5), return_counts=True)
    for v, n in zip(values.tolist(), counts.tolist(), strict=True):
        out[v] = n
    return out


def mesh_hash(mesh: Mesh) -> str:
    """SHA-256 of a Mesh's arrays (positions, triangles, normals, UVs)."""
    h = hashlib.sha256()
    for arr in (mesh.positions, mesh.triangles, mesh.normals, mesh.uvs):
        h.update(b"-" if arr is None else np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
