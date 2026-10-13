"""Polygon-level topology validation of an :class:`EditableMesh`.

Implemented checks (codes and severities are listed in ``docs/EDITING.md``):
internal adjacency consistency, invalid face loops, non-finite positions,
zero-area faces, self-intersecting (non-simple) polygons, non-planar
polygons, concave polygons, duplicate faces, coincident vertices (exactly
equal positions), loose vertices, boundary edges, non-manifold edges,
non-manifold vertices (faces around a vertex forming several fans), and
inconsistent winding across edges.

Not implemented: intersections between *different* faces (self-intersection
of the surface), and overlap of faces that do not share vertices. Reports
therefore never claim a mesh is free of self-intersections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import ERROR, INFO, MAX_SAMPLES, WARNING, Diagnostic, worst_severity
from . import _polygon as poly

if TYPE_CHECKING:
    from .core import MeshCore

NONPLANAR_RELATIVE = 1e-5
"""A polygon is non-planar if a vertex is farther than this x its size from its best-fit plane."""


@dataclass
class TopologyReport:
    """Result of :meth:`EditableMesh.validate`."""

    summary: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def status(self) -> str:
        return worst_severity(self.diagnostics)

    @property
    def ok(self) -> bool:
        """True when there are no error-severity findings."""
        return self.status != "error"

    def codes(self) -> set[str]:
        return {d.code for d in self.diagnostics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": dict(self.summary),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "not_evaluated": [
                "self-intersection between different faces",
                "overlap of unconnected faces",
            ],
        }


def validate(mesh: MeshCore) -> TopologyReport:
    diagnostics: list[Diagnostic] = []

    def add(code: str, severity: str, message: str, kind: str, ids: list[int]) -> None:
        if ids:
            ids = sorted(ids)
            diagnostics.append(
                Diagnostic(code, severity, message, kind, len(ids), tuple(ids[:MAX_SAMPLES]))
            )

    broken = mesh._consistency_errors()
    if broken:
        diagnostics.append(
            Diagnostic("BROKEN_ADJACENCY", ERROR, "; ".join(broken[:5]), None, len(broken))
        )
    alive = mesh.vertex_ids()
    positions = mesh._pos
    add(
        "NON_FINITE_POSITIONS",
        ERROR,
        "vertex positions contain NaN or infinity",
        "vertices",
        [v for v in alive if not np.all(np.isfinite(positions[v]))],
    )

    eps = mesh._area_eps()
    zero, self_intersecting, nonplanar, concave = [], [], [], []
    for f, loop in mesh._faces.items():
        pts = positions[loop]
        normal = poly.newell_normal(pts)
        if np.linalg.norm(normal) / 2 <= eps:
            zero.append(f)
            continue
        if len(loop) == 3:
            continue
        n = normal / np.linalg.norm(normal)
        size = float(np.max(np.ptp(pts, axis=0)))
        if float(np.max(np.abs((pts - pts.mean(axis=0)) @ n))) > NONPLANAR_RELATIVE * size:
            nonplanar.append(f)
        projected = poly.project(pts, normal)
        if not poly.is_simple(projected):
            self_intersecting.append(f)
            continue
        cross = [
            (projected[(k + 1) % len(loop)] - projected[k])[0]
            * (projected[(k + 2) % len(loop)] - projected[k])[1]
            - (projected[(k + 1) % len(loop)] - projected[k])[1]
            * (projected[(k + 2) % len(loop)] - projected[k])[0]
            for k in range(len(loop))
        ]
        if min(cross) < -1e-12 * size**2:
            concave.append(f)
    add("ZERO_AREA_FACES", WARNING, "faces have (near-)zero area", "faces", zero)
    add(
        "SELF_INTERSECTING_FACES",
        ERROR,
        "face outlines cross themselves; they cannot be triangulated",
        "faces",
        self_intersecting,
    )
    add(
        "NONPLANAR_FACES",
        WARNING,
        "polygons are not planar; their triangulation affects the shape",
        "faces",
        nonplanar,
    )
    add(
        "CONCAVE_FACES",
        INFO,
        "polygons are concave (valid, triangulated by ear clipping)",
        "faces",
        concave,
    )

    seen: dict[frozenset, int] = {}
    duplicates = []
    for f in sorted(mesh._faces):
        key = frozenset(mesh._faces[f])
        if key in seen:
            duplicates.append(f)
        else:
            seen[key] = f
    add(
        "DUPLICATE_FACES",
        WARNING,
        "faces use the same vertices as an earlier face",
        "faces",
        duplicates,
    )

    if alive:
        p = np.ascontiguousarray(positions[alive] + 0.0)
        keys = p.view(np.dtype((np.void, p.dtype.itemsize * 3))).ravel()
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        coincident = [alive[i] for i in np.flatnonzero(counts[inverse.reshape(-1)] > 1)]
        add(
            "COINCIDENT_VERTICES",
            WARNING,
            "distinct vertices share exactly the same position",
            "vertices",
            coincident,
        )
    add(
        "LOOSE_VERTICES",
        INFO,
        "vertices are not used by any face",
        "vertices",
        [v for v in alive if not mesh._vert_edges[v]],
    )

    boundary = [e for e, fs in mesh._edge_faces.items() if len(fs) == 1]
    non_manifold = [e for e, fs in mesh._edge_faces.items() if len(fs) > 2]
    add("BOUNDARY_EDGES", INFO, "edges with one face (open boundary)", "edges", boundary)
    add("NON_MANIFOLD_EDGES", WARNING, "edges shared by more than two faces", "edges", non_manifold)

    inconsistent = []
    for e, fs in mesh._edge_faces.items():
        if len(fs) != 2:
            continue
        a, b = mesh._edge_verts[e]
        directions = []
        for f in fs:
            loop = mesh._faces[f]
            directions.append(loop[(loop.index(a) + 1) % len(loop)] == b)
        if directions[0] == directions[1]:
            inconsistent.append(e)
    add(
        "INCONSISTENT_WINDING",
        WARNING,
        "the two faces of these edges are wound in the same direction",
        "edges",
        inconsistent,
    )

    nm_vertices = [v for v in alive if mesh._vert_edges[v] and _fan_count(mesh, v) > 1]
    add(
        "NON_MANIFOLD_VERTICES",
        WARNING,
        "faces around these vertices form more than one fan (surfaces touch at a single vertex)",
        "vertices",
        nm_vertices,
    )

    summary = mesh.summary()
    summary.update(
        euler_characteristic=summary["vertex_count"]
        - summary["edge_count"]
        + summary["face_count"],
        closed=not boundary and bool(mesh._faces),
        edge_manifold=not non_manifold,
        vertex_manifold=not nm_vertices,
        consistently_oriented=not inconsistent,
    )
    return TopologyReport(summary, diagnostics)


def _fan_count(mesh: MeshCore, v: int) -> int:
    """Number of edge-connected groups of faces around ``v`` (1 for a manifold vertex)."""
    faces = mesh._faces_of_vertex(v)
    parent = {f: f for f in faces}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in mesh._vert_edges[v]:
        fs = mesh._edge_faces[e]
        if len(fs) == 2:  # only manifold edges connect a fan
            ra, rb = find(fs[0]), find(fs[1])
            if ra != rb:
                parent[ra] = rb
    return len({find(f) for f in faces})
