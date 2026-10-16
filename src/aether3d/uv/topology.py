"""Welded topology of a render mesh for UV work: edges, mates, seams and chart vertices.

A :class:`~aether3d.Mesh` splits vertices wherever an attribute differs (a UV seam, a
hard edge), so its triangles do not share vertices across those edges. UV work needs the
surface itself: vertices are **welded** by exact position, and every triangle corner
``(t, k)`` points at a welded vertex. Half-edge ``h = 3 t + k`` runs from corner ``k`` to
corner ``k + 1`` of triangle ``t``.

* An edge with exactly two half-edges of opposite directions is **manifold**; its two
  half-edges are each other's ``mate``. Boundary edges (one half-edge), non-manifold edges
  (three or more) and edges whose two triangles disagree on orientation have no mate.
* Across a mated edge, the render mesh may still be cut: the corners on the two sides can
  have different UVs (a **UV seam**), different normals (a **hard edge**) or different face
  groups (a **group border**).
* **Chart vertices.** Within a set of faces (a chart), the corners of one welded vertex are
  the same chart vertex when the faces around it are connected through uncut mated edges
  inside the chart -- so a bow-tie vertex, or a vertex on a cut that reaches into the
  chart, becomes several chart vertices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mesh import Mesh


def union_labels(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of ``n`` nodes joined by pairs ``(a[i], b[i])``: every node's
    label is the smallest node of its component."""
    parent = np.arange(n, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.size == 0:
        return parent
    while True:
        pa, pb = parent[a], parent[b]
        differ = pa != pb
        if not differ.any():
            return parent
        np.minimum.at(
            parent, np.maximum(pa[differ], pb[differ]), np.minimum(pa[differ], pb[differ])
        )
        while True:
            up = parent[parent]
            if np.array_equal(up, parent):
                break
            parent = up


def dense_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Labels renumbered ``0 .. k - 1`` in order of first appearance of each label value's
    smallest member (i.e. ascending label value); returns ``(labels, k)``."""
    uniq, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64), int(uniq.shape[0])


def weld_positions(positions: np.ndarray) -> tuple[np.ndarray, int]:
    """Welded id of every vertex (equal positions share an id; ``-0.0`` equals ``0.0``);
    ids are numbered by first occurrence."""
    p = np.ascontiguousarray(np.asarray(positions, dtype=np.float32) + np.float32(0.0))
    if p.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), 0
    view = p.view(np.dtype((np.void, p.dtype.itemsize * 3))).reshape(-1)
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    # Number welded vertices by their first occurrence (stable, input-order ids).
    rank = np.empty(first.shape[0], dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.shape[0], dtype=np.int64)
    return rank[inverse.reshape(-1)], int(first.shape[0])


@dataclass
class Topology:
    """The welded surface of a mesh (see the module docstring). Arrays are read-only."""

    positions: np.ndarray  # (V, 3) float64, render vertices
    triangles: np.ndarray  # (T, 3) int64, render vertex ids
    uvs: np.ndarray | None  # (V, 2) float64 or None
    normals: np.ndarray | None  # (V, 3) float64 or None
    groups: np.ndarray | None  # (T,) int64 or None
    weld: np.ndarray  # (V,) welded id of each render vertex
    welded_count: int
    corners: np.ndarray  # (T, 3) welded id of each corner
    wpos: np.ndarray  # (welded_count, 3) float64 welded positions
    edge: np.ndarray  # (3T,) edge id of each half-edge
    edges: np.ndarray  # (E, 2) welded endpoints (min, max)
    edge_faces: np.ndarray  # (E,) half-edges on the edge
    mate: np.ndarray  # (3T,) the opposite half-edge, or -1
    face_normal: np.ndarray  # (T, 3) unit (zero for degenerate faces)
    face_area: np.ndarray  # (T,) 3D area
    uv_seam: np.ndarray  # (3T,) mated half-edge whose two sides have different UVs
    hard: np.ndarray  # (3T,) mated half-edge whose two sides have different normals
    group_border: np.ndarray  # (3T,) mated half-edge between two face groups
    dihedral: np.ndarray  # (3T,) angle between the two faces' normals (radians), 0 unmated

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> Topology:
        positions = mesh.positions.astype(np.float64)
        tris = mesh.triangles.astype(np.int64)
        weld, count = weld_positions(mesh.positions)
        corners = weld[tris]
        wpos = np.zeros((count, 3), dtype=np.float64)
        if count:
            wpos[weld] = positions
        T = tris.shape[0]
        a = corners.reshape(-1)
        b = corners[:, [1, 2, 0]].reshape(-1)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo * max(count, 1) + hi
        uniq, edge, counts = np.unique(key, return_inverse=True, return_counts=True)
        edge = edge.reshape(-1).astype(np.int64)
        edges = np.stack([uniq // max(count, 1), uniq % max(count, 1)], axis=1).astype(np.int64)
        edge_faces = counts.astype(np.int64)
        # Mates: edges with exactly two half-edges running in opposite directions.
        mate = np.full(3 * T, -1, dtype=np.int64)
        two = edge_faces[edge] == 2
        h = np.flatnonzero(two)
        if h.size:
            order = np.argsort(edge[h], kind="stable")
            hs = h[order].reshape(-1, 2)
            opposite = a[hs[:, 0]] == b[hs[:, 1]]
            hs = hs[opposite & (a[hs[:, 0]] != a[hs[:, 1]])]
            mate[hs[:, 0]] = hs[:, 1]
            mate[hs[:, 1]] = hs[:, 0]
        p = positions[tris]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        length = np.linalg.norm(cross, axis=1)
        area = 0.5 * length
        normal = np.divide(
            cross, length[:, None], out=np.zeros_like(cross), where=length[:, None] > 0
        )
        uvs = None if mesh.uvs is None else mesh.uvs.astype(np.float64)
        normals = None if mesh.normals is None else mesh.normals.astype(np.float64)
        groups = None if mesh.groups is None else mesh.groups.astype(np.int64)
        hv = tris.reshape(-1)  # render vertex at the start of each half-edge
        hn = tris[:, [1, 2, 0]].reshape(-1)  # ... and at its end
        mated = np.flatnonzero(mate >= 0)
        m = mate[mated]
        uv_seam = np.zeros(3 * T, dtype=bool)
        hard = np.zeros(3 * T, dtype=bool)
        border = np.zeros(3 * T, dtype=bool)
        dihedral = np.zeros(3 * T, dtype=np.float64)
        if mated.size:
            # h runs a -> b, its mate b -> a: corner a of h's face meets the mate's end.
            if uvs is not None:
                uv_seam[mated] = np.any(uvs[hv[mated]] != uvs[hn[m]], axis=1) | np.any(
                    uvs[hn[mated]] != uvs[hv[m]], axis=1
                )
            if normals is not None:
                hard[mated] = np.any(normals[hv[mated]] != normals[hn[m]], axis=1) | np.any(
                    normals[hn[mated]] != normals[hv[m]], axis=1
                )
            if groups is not None:
                border[mated] = groups[mated // 3] != groups[m // 3]
            cosine = np.clip((normal[mated // 3] * normal[m // 3]).sum(axis=1), -1.0, 1.0)
            dihedral[mated] = np.arccos(cosine)
        for arr in (
            positions,
            tris,
            weld,
            corners,
            wpos,
            edge,
            edges,
            edge_faces,
            mate,
            normal,
            area,
        ):
            arr.flags.writeable = False
        return cls(
            positions,
            tris,
            uvs,
            normals,
            groups,
            weld,
            count,
            corners,
            wpos,
            edge,
            edges,
            edge_faces,
            mate,
            normal,
            area,
            uv_seam,
            hard,
            border,
            dihedral,
        )

    def chart_vertices(self, chart: np.ndarray, cut: np.ndarray) -> tuple[np.ndarray, int]:
        """Chart-vertex id of every corner (``(T, 3)``) for the face partition ``chart``
        (``(T,)``) and the cut half-edges ``cut`` (``(3T,)`` bool); ids are numbered by
        their lowest corner."""
        T = self.triangle_count
        h = np.flatnonzero((self.mate >= 0) & ~cut)
        m = self.mate[h]
        keep = (h < m) & (chart[h // 3] == chart[m // 3])
        h, m = h[keep], m[keep]
        # h runs a -> b in face h//3; its mate b -> a. Corner of a in h's face is h % 3,
        # in the mate's face it is (m % 3 + 1) % 3; likewise for b.
        fh, kh = h // 3, h % 3
        fm, km = m // 3, m % 3
        ca1 = 3 * fh + kh
        ca2 = 3 * fm + (km + 1) % 3
        cb1 = 3 * fh + (kh + 1) % 3
        cb2 = 3 * fm + km
        labels = union_labels(3 * T, np.r_[ca1, cb1], np.r_[ca2, cb2])
        ids, count = dense_labels(labels)
        return ids.reshape(T, 3), count

    def face_components(self, cut: np.ndarray, faces: np.ndarray | None = None) -> np.ndarray:
        """Component label (smallest face id) of every face, joining faces across uncut mated
        edges (only faces where ``faces`` is true, when given; others keep their own id)."""
        h = np.flatnonzero((self.mate >= 0) & ~cut)
        m = self.mate[h]
        keep = h < m
        if faces is not None:
            keep &= faces[h // 3] & faces[m // 3]
        return union_labels(self.triangle_count, h[keep] // 3, m[keep] // 3)
