"""Geometry inspection: structured, machine-readable reports on meshes and scenes.

:func:`inspect_mesh` analyses a single :class:`Mesh`; :func:`inspect_scene`
analyses every mesh in a scene once (shared meshes are reported once) and adds
world-space information per object. Both return report objects with a
``to_dict()`` method producing plain JSON-compatible data.

What is and is not evaluated
----------------------------
Topology is computed on *welded* vertices: vertices are merged only when their
positions are bit-identical (``-0.0`` equals ``0.0``); no distance tolerance is
applied. On that welded graph the report evaluates, per edge:

* boundary edges (used by exactly one triangle) -> ``closed``
* edges used by more than two triangles -> ``edge_manifold``
* edges traversed in the same direction by both adjacent triangles
  -> ``consistently_oriented``

and, per vertex, whether the triangles around it form a single fan connected
through manifold edges (``vertex_manifold``; a "bowtie" where two surfaces
touch at one vertex is not). Self-intersections are **not** evaluated, so
reports never claim a mesh is free of them or "watertight" in the general
sense. The enclosed ``volume`` is reported only when the mesh is closed,
edge-manifold and consistently oriented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .diagnostics import ERROR, INFO, MAX_SAMPLES, WARNING, Diagnostic, worst_severity
from .math3d import compose_matrix, decompose_matrix
from .mesh import Mesh

if TYPE_CHECKING:
    from .scene import Scene

NORMAL_UNIT_TOLERANCE = 1e-3
"""Normals whose length differs from 1 by more than this are reported."""

DEGENERATE_AREA_RELATIVE = 1e-12
"""Triangles with area <= this x (bounding-box diagonal)^2 count as zero-area."""


def _samples(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(mask)[:MAX_SAMPLES])


def _bounds_dict(points: np.ndarray) -> dict[str, list[float]] | None:
    if points.shape[0] == 0:
        return None
    lo = points.min(axis=0).astype(np.float64)
    hi = points.max(axis=0).astype(np.float64)
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "size": (hi - lo).tolist(),
        "center": ((lo + hi) / 2.0).tolist(),
    }


def _triangle_cross(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Per-triangle cross product (b - a) x (c - a); its length is twice the area."""
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return np.cross(b - a, c - a)


@dataclass
class MeshGeometryReport:
    """Result of :func:`inspect_mesh`. See ``docs/INSPECTION.md`` for every field."""

    vertex_count: int
    triangle_count: int
    attributes: dict[str, bool]
    bounds: dict[str, list[float]] | None
    surface_area: float
    uv_bounds: dict[str, list[float]] | None
    topology: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``"ok"``, ``"warning"`` or ``"error"`` (worst diagnostic severity)."""
        return worst_severity(self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "attributes": dict(self.attributes),
            "bounds": self.bounds,
            "surface_area": self.surface_area,
            "uv_bounds": self.uv_bounds,
            "topology": dict(self.topology),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def inspect_mesh(mesh: Mesh, *, location: str | None = None) -> MeshGeometryReport:
    """Analyse ``mesh`` and return a :class:`MeshGeometryReport`.

    ``location`` is attached to every diagnostic (e.g. ``"meshes[2]"``).
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"expected a Mesh, got {type(mesh).__name__}")
    positions = mesh.positions.astype(np.float64)
    triangles = mesh.triangles.astype(np.int64)
    n_vertices, n_triangles = mesh.vertex_count, mesh.triangle_count
    diagnostics: list[Diagnostic] = []

    def report(code: str, severity: str, message: str, mask: np.ndarray | None = None) -> None:
        count = None if mask is None else int(mask.sum())
        samples = () if mask is None else _samples(mask)
        diagnostics.append(Diagnostic(code, severity, message, location, count, samples))

    # -- data validity (guaranteed by Mesh, verified anyway) ---------------
    for name, array in (
        ("positions", mesh.positions),
        ("normals", mesh.normals),
        ("uvs", mesh.uvs),
    ):
        if array is not None:
            bad = ~np.all(np.isfinite(array), axis=1)
            if bad.any():
                report(f"NON_FINITE_{name.upper()}", ERROR, f"vertex {name} contain NaN/inf", bad)
    index_bad = np.any(triangles >= n_vertices, axis=1) if n_triangles else np.zeros(0, bool)
    if index_bad.any():
        report("INDEX_OUT_OF_RANGE", ERROR, "triangles reference missing vertices", index_bad)
        triangles = triangles[~index_bad]

    if n_triangles == 0:
        report("EMPTY_MESH", INFO, "mesh has no triangles")

    # -- triangle quality --------------------------------------------------
    bounds = _bounds_dict(positions)
    diag_sq = float(np.sum(np.square(np.subtract(bounds["max"], bounds["min"])))) if bounds else 0.0
    cross = _triangle_cross(positions, triangles) if len(triangles) else np.zeros((0, 3))
    double_area = np.linalg.norm(cross, axis=1)
    surface_area = float(double_area.sum() / 2.0)

    repeated = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    zero_area = ~repeated & (double_area / 2.0 <= DEGENERATE_AREA_RELATIVE * max(diag_sq, 1e-300))
    if repeated.any():
        report(
            "DEGENERATE_TRIANGLES_REPEATED_INDEX",
            WARNING,
            "triangles use the same vertex index more than once",
            repeated,
        )
    if zero_area.any():
        report(
            "DEGENERATE_TRIANGLES_ZERO_AREA", WARNING, "triangles have (near-)zero area", zero_area
        )

    if len(triangles):
        canonical = np.sort(triangles, axis=1)
        _, first, counts = np.unique(canonical, axis=0, return_index=True, return_counts=True)
        duplicate = np.ones(len(triangles), dtype=bool)
        duplicate[first] = False  # every occurrence after the first is a duplicate
        if duplicate.any():
            report(
                "DUPLICATE_TRIANGLES",
                WARNING,
                "triangles reference the same three vertices as an earlier triangle",
                duplicate,
            )

    referenced = np.zeros(n_vertices, dtype=bool)
    referenced[triangles.ravel()] = True
    if n_vertices and not referenced.all():
        report("UNREFERENCED_VERTICES", INFO, "vertices are not used by any triangle", ~referenced)

    # -- normals -----------------------------------------------------------
    if mesh.normals is not None and n_vertices:
        normals = mesh.normals.astype(np.float64)
        length = np.linalg.norm(normals, axis=1)
        zero = length < 1e-12
        non_unit = ~zero & (np.abs(length - 1.0) > NORMAL_UNIT_TOLERANCE)
        if zero.any():
            report("ZERO_LENGTH_NORMALS", WARNING, "vertex normals have zero length", zero)
        if non_unit.any():
            report(
                "NON_UNIT_NORMALS",
                WARNING,
                f"vertex normals deviate from unit length by more than {NORMAL_UNIT_TOLERANCE}",
                non_unit,
            )
        if len(triangles):
            vertex_normal_sum = normals[triangles].sum(axis=1)
            facing = np.einsum("ij,ij->i", cross, vertex_normal_sum)
            usable = ~(repeated | zero_area) & (np.linalg.norm(vertex_normal_sum, axis=1) > 1e-12)
            opposed = usable & (facing < 0.0)
            if opposed.any():
                report(
                    "NORMALS_OPPOSE_WINDING",
                    WARNING,
                    "triangle winding (counter-clockwise = front) points away from "
                    "its vertex normals",
                    opposed,
                )

    uv_bounds = None
    if mesh.uvs is not None and n_vertices:
        uv_bounds = _bounds_dict(mesh.uvs.astype(np.float64))
        if uv_bounds is not None:
            del uv_bounds["center"], uv_bounds["size"]

    topology = _topology(mesh, triangles[~(repeated | zero_area)] if len(triangles) else triangles)
    if topology["boundary_edge_count"]:
        diagnostics.append(
            Diagnostic(
                "OPEN_BOUNDARY",
                INFO,
                "mesh has boundary edges (it is not closed); "
                "expected for open surfaces like planes",
                location,
                topology["boundary_edge_count"],
            )
        )
    if topology["non_manifold_edge_count"]:
        diagnostics.append(
            Diagnostic(
                "NON_MANIFOLD_EDGES",
                WARNING,
                "edges are shared by more than two triangles",
                location,
                topology["non_manifold_edge_count"],
            )
        )
    if topology["non_manifold_vertex_count"]:
        diagnostics.append(
            Diagnostic(
                "NON_MANIFOLD_VERTICES",
                WARNING,
                "triangles around these vertices form more than one fan "
                "(surfaces touch at a single vertex)",
                location,
                topology["non_manifold_vertex_count"],
            )
        )
    if topology["inconsistent_edge_count"]:
        diagnostics.append(
            Diagnostic(
                "INCONSISTENT_ORIENTATION",
                WARNING,
                "adjacent triangles have opposite winding across shared edges",
                location,
                topology["inconsistent_edge_count"],
            )
        )
    if topology["volume"] is not None and topology["volume"] < 0.0:
        diagnostics.append(
            Diagnostic(
                "INSIDE_OUT",
                WARNING,
                "closed mesh has negative volume: its triangles face inwards",
                location,
            )
        )

    return MeshGeometryReport(
        vertex_count=n_vertices,
        triangle_count=n_triangles,
        attributes={
            "positions": True,
            "normals": mesh.normals is not None,
            "uvs": mesh.uvs is not None,
        },
        bounds=bounds,
        surface_area=surface_area,
        uv_bounds=uv_bounds,
        topology=topology,
        diagnostics=diagnostics,
    )


def _topology(mesh: Mesh, triangles: np.ndarray) -> dict[str, Any]:
    """Edge statistics on the exactly-welded vertex graph (degenerate triangles excluded)."""
    result: dict[str, Any] = {
        "vertex_welding": "exact_position",
        "welded_vertex_count": 0,
        "edge_count": 0,
        "boundary_edge_count": 0,
        "non_manifold_edge_count": 0,
        "inconsistent_edge_count": 0,
        "non_manifold_vertex_count": 0,
        "closed": False,
        "edge_manifold": True,
        "consistently_oriented": True,
        "vertex_manifold": True,
        "volume": None,
    }
    if mesh.vertex_count == 0:
        return result
    # "+ 0" turns -0.0 into 0.0 so that both weld together.
    points = np.ascontiguousarray(mesh.positions + np.float32(0.0))
    keys = points.view(np.dtype((np.void, points.dtype.itemsize * 3))).ravel()
    _, weld = np.unique(keys, return_inverse=True)
    weld = weld.reshape(-1)
    result["welded_vertex_count"] = int(weld.max()) + 1
    if len(triangles) == 0:
        return result

    w = weld[triangles]
    w = w[(w[:, 0] != w[:, 1]) & (w[:, 1] != w[:, 2]) & (w[:, 0] != w[:, 2])]
    if len(w) == 0:
        return result
    n = np.int64(result["welded_vertex_count"])
    a = np.concatenate([w[:, 0], w[:, 1], w[:, 2]]).astype(np.int64)
    b = np.concatenate([w[:, 1], w[:, 2], w[:, 0]]).astype(np.int64)
    undirected = np.minimum(a, b) * n + np.maximum(a, b)
    directed = a * n + b
    edge_keys, edge_index, edge_counts = np.unique(
        undirected, return_inverse=True, return_counts=True
    )
    # How many distinct directions each undirected edge is traversed in.
    unique_directed, first = np.unique(directed, return_index=True)
    directions = np.bincount(edge_index.reshape(-1)[first], minlength=len(edge_keys))

    boundary = int(np.sum(edge_counts == 1))
    non_manifold = int(np.sum(edge_counts > 2))
    inconsistent = int(np.sum((edge_counts == 2) & (directions == 1)))
    result.update(
        edge_count=int(len(edge_keys)),
        boundary_edge_count=boundary,
        non_manifold_edge_count=non_manifold,
        inconsistent_edge_count=inconsistent,
        closed=boundary == 0,
        edge_manifold=non_manifold == 0,
        consistently_oriented=inconsistent == 0,
    )
    nm_vertices = _non_manifold_vertices(w, edge_index.reshape(-1), edge_counts)
    result.update(non_manifold_vertex_count=nm_vertices, vertex_manifold=nm_vertices == 0)
    if boundary == 0 and non_manifold == 0 and inconsistent == 0:
        p = mesh.positions.astype(np.float64)
        signed = np.einsum(
            "ij,ij->i", p[triangles[:, 0]], np.cross(p[triangles[:, 1]], p[triangles[:, 2]])
        )
        result["volume"] = float(signed.sum() / 6.0)
    return result


def _non_manifold_vertices(w: np.ndarray, edge_index: np.ndarray, edge_counts: np.ndarray) -> int:
    """Count vertices whose triangles form more than one fan (connected via manifold edges).

    ``w`` are welded, non-degenerate triangles; ``edge_index`` maps each of the
    ``3T`` triangle edges (block layout: edge k of triangle t is entry
    ``k*T + t``) to its undirected edge. Corners (also in block layout) are
    linked across manifold edges and grouped by vectorized label propagation.
    """
    t_count = len(w)
    entries = np.flatnonzero(edge_counts[edge_index] == 2)
    if len(entries):
        order = entries[np.argsort(edge_index[entries], kind="stable")]
        e1, e2 = order[0::2], order[1::2]
        blk1, t1 = np.divmod(e1, t_count)
        blk2, t2 = np.divmod(e2, t_count)
        u1, u2 = w[t1, blk1], w[t2, blk2]
        cu1, cv1 = blk1 * t_count + t1, ((blk1 + 1) % 3) * t_count + t1
        cu2, cv2 = blk2 * t_count + t2, ((blk2 + 1) % 3) * t_count + t2
        same = u1 == u2  # both triangles list the edge starting at the same vertex
        a = np.concatenate([cu1, cv1])
        b = np.concatenate([np.where(same, cu2, cv2), np.where(same, cv2, cu2)])
    else:
        a = b = np.zeros(0, dtype=np.int64)
    labels = np.arange(3 * t_count)
    while True:
        m = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, labels):
            break
        labels = new
    vertex_of_corner = w.T.reshape(-1)
    pairs = np.unique(np.stack([vertex_of_corner, labels], axis=1), axis=0)
    fans = np.bincount(pairs[:, 0])
    return int(np.sum(fans > 1))


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #


@dataclass
class SceneGeometryReport:
    """Result of :func:`inspect_scene`. See ``docs/INSPECTION.md`` for every field."""

    scene_name: str
    summary: dict[str, Any]
    world_bounds: dict[str, list[float]] | None
    meshes: list[MeshGeometryReport]
    mesh_users: list[list[str]]
    objects: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def all_diagnostics(self) -> list[Diagnostic]:
        """Scene-level diagnostics followed by every mesh's diagnostics."""
        return self.diagnostics + [d for m in self.meshes for d in m.diagnostics]

    @property
    def status(self) -> str:
        """``"ok"``, ``"warning"`` or ``"error"`` across the scene and all meshes."""
        return worst_severity(self.all_diagnostics)

    def to_dict(self) -> dict[str, Any]:
        meshes = []
        for index, (report, users) in enumerate(zip(self.meshes, self.mesh_users, strict=True)):
            meshes.append({"index": index, "used_by": list(users), **report.to_dict()})
        return {
            "status": self.status,
            "scene": self.scene_name,
            "summary": dict(self.summary),
            "world_bounds": self.world_bounds,
            "objects": [dict(o) for o in self.objects],
            "meshes": meshes,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def inspect_scene(scene: Scene) -> SceneGeometryReport:
    """Analyse every object and mesh of ``scene``.

    Each distinct mesh is inspected once, in order of first use (the same
    order used by ``.aether`` files). Objects reference meshes by index.
    """
    mesh_index: dict[int, int] = {}
    meshes: list[Mesh] = []
    users: list[list[str]] = []
    objects: list[dict[str, Any]] = []
    diagnostics: list[Diagnostic] = []
    world_points: list[np.ndarray] = []
    totals = {"vertices": 0, "triangles": 0}

    for obj in scene:
        world = obj.world_matrix
        det = float(np.linalg.det(world[:3, :3]))
        loc = f"objects[{obj.id}]"
        sheared = False
        if abs(det) < 1e-12:
            diagnostics.append(
                Diagnostic(
                    "ZERO_WORLD_SCALE",
                    WARNING,
                    f"object {obj.name!r} has a singular world transform (zero scale); "
                    "its geometry collapses",
                    loc,
                )
            )
        else:
            position, rotation, scale = decompose_matrix(world)
            sheared = not np.allclose(
                compose_matrix(position, rotation, scale), world, rtol=1e-9, atol=1e-9
            )
            if sheared:
                diagnostics.append(
                    Diagnostic(
                        "SHEARED_WORLD_TRANSFORM",
                        INFO,
                        f"object {obj.name!r} inherits non-uniform scale through a rotated parent; "
                        "its world transform contains shear "
                        "(world_rotation/world_scale are approximate)",
                        loc,
                    )
                )
        if det < 0.0:
            diagnostics.append(
                Diagnostic(
                    "MIRRORED_WORLD_TRANSFORM",
                    INFO,
                    f"object {obj.name!r} has a mirrored world transform (negative determinant); "
                    "its triangle winding appears reversed in world space",
                    loc,
                )
            )

        entry: dict[str, Any] = {
            "id": obj.id,
            "name": obj.name,
            "parent": None if obj.parent is None else obj.parent.id,
            "mesh": None,
            "world_matrix_determinant": det,
            "mirrored": det < 0.0,
            "sheared": sheared,
            "world_bounds": None,
            "world_surface_area": None,
        }
        mesh = obj.mesh
        if mesh is not None:
            key = id(mesh)
            if key not in mesh_index:
                mesh_index[key] = len(meshes)
                meshes.append(mesh)
                users.append([])
            entry["mesh"] = mesh_index[key]
            users[mesh_index[key]].append(obj.id)
            totals["vertices"] += mesh.vertex_count
            totals["triangles"] += mesh.triangle_count
            if mesh.vertex_count:
                local = mesh.positions.astype(np.float64)
                points = local @ world[:3, :3].T + world[:3, 3]
                world_points.append(points)
                entry["world_bounds"] = _bounds_dict(points)
                if mesh.triangle_count:
                    area = np.linalg.norm(_triangle_cross(points, mesh.triangles), axis=1).sum()
                    entry["world_surface_area"] = float(area / 2.0)
                else:
                    entry["world_surface_area"] = 0.0
        objects.append(entry)

    reports = [inspect_mesh(m, location=f"meshes[{i}]") for i, m in enumerate(meshes)]
    all_points = np.concatenate(world_points) if world_points else np.zeros((0, 3))
    summary = {
        "object_count": len(objects),
        "mesh_object_count": sum(1 for o in objects if o["mesh"] is not None),
        "unique_mesh_count": len(meshes),
        "instanced_vertex_count": totals["vertices"],
        "instanced_triangle_count": totals["triangles"],
        "unique_vertex_count": sum(m.vertex_count for m in meshes),
        "unique_triangle_count": sum(m.triangle_count for m in meshes),
    }
    return SceneGeometryReport(
        scene_name=scene.name,
        summary=summary,
        world_bounds=_bounds_dict(all_points),
        meshes=reports,
        mesh_users=users,
        objects=objects,
        diagnostics=diagnostics,
    )
