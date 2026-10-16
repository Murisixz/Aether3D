"""Modifier types: non-destructive operations from one PolyMesh to another.

Every modifier type declares a parameter schema (see :func:`modifier_types`)
and implements ``apply(mesh, values, context) -> (mesh, diagnostics)``. Modifiers
never mutate their input (PolyMesh arrays are read-only) and raise
:class:`ProceduralError` with a stable code for unsupported input instead of
producing damaged geometry. See ``docs/PROCEDURAL.md`` for the exact
mathematics and attribute rules of each type.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..diagnostics import INFO, MAX_SAMPLES, WARNING, Diagnostic
from ..errors import EditError
from ..math3d import quat_from_euler, quat_to_matrix3
from .params import ParamSpec, ProceduralError
from .polymesh import PolyMesh

MAX_FACES = 5_000_000
"""Upper bound on the faces any modifier may produce (guards runaway parameters)."""


@dataclass
class ModifierContext:
    """What a modifier may use besides its input: a location for errors and operands."""

    location: str
    evaluate_operand: Callable[[Any, str], PolyMesh] | None = None


class ModifierType:
    name: str = ""
    description: str = ""
    params: tuple[ParamSpec, ...] = ()

    def check(self, values: dict, location: str) -> None:
        """Cross-parameter validation (raise ProceduralError)."""

    def apply(self, mesh: PolyMesh, values: dict, ctx: ModifierContext):
        raise NotImplementedError

    def references(self, values: dict) -> list[Any]:
        """Operand references (geometry ids/names) this modifier depends on."""
        return []

    def schema(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "description": self.description,
            "params": [p.to_json() for p in self.params],
        }


def _diag(code: str, severity: str, message: str, location: str, ids=None) -> Diagnostic:
    ids = [] if ids is None else [int(i) for i in np.asarray(ids).ravel()]
    return Diagnostic(code, severity, message, location, len(ids) or None, tuple(ids[:MAX_SAMPLES]))


def _limit(faces: int, location: str) -> None:
    if faces > MAX_FACES:
        raise ProceduralError(
            "PROCEDURAL_LIMIT",
            f"the result would have {faces} faces (limit {MAX_FACES})",
            location,
            hint="reduce counts, levels or segments",
        )


def _affine(
    translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), pivot=(0, 0, 0)
) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix3(quat_from_euler(rotation)) * np.asarray(scale, dtype=np.float64)
    p = np.asarray(pivot, dtype=np.float64)
    m[:3, 3] = np.asarray(translation, dtype=np.float64) + p - m[:3, :3] @ p
    return m


def reverse_corners(mesh: PolyMesh) -> np.ndarray:
    """Permutation re-winding every face (keeps each face's first corner)."""
    cf = mesh.corner_face
    start = mesh.face_offsets[cf]
    local = np.arange(mesh.corner_count) - start
    new_local = np.where(local == 0, 0, mesh.face_sizes[cf] - local)
    perm = np.empty(mesh.corner_count, dtype=np.int64)
    perm[start + new_local] = np.arange(mesh.corner_count)
    return perm


def transform_mesh(mesh: PolyMesh, m: np.ndarray, location: str) -> PolyMesh:
    """Apply an affine matrix; mirrored transforms re-wind faces so they stay outward-facing."""
    linear = m[:3, :3]
    det = float(np.linalg.det(linear))
    if abs(det) < 1e-300 or not math.isfinite(det):
        raise ProceduralError(
            "DEGENERATE_TRANSFORM",
            "the transform collapses space (zero scale)",
            location,
            hint="use non-zero scale factors",
        )
    positions = mesh.positions @ linear.T + m[:3, 3]
    normals = None
    if mesh.normals is not None:
        n = mesh.normals @ np.linalg.inv(linear)  # rows: (A^-T n)^T
        length = np.linalg.norm(n, axis=1, keepdims=True)
        normals = np.divide(n, length, out=np.array(mesh.normals, copy=True), where=length > 1e-300)
    if det > 0:
        return mesh.with_(positions=positions, normals=normals)
    perm = reverse_corners(mesh)
    return mesh.with_(
        positions=positions,
        face_vertices=mesh.face_vertices[perm],
        uvs=None if mesh.uvs is None else mesh.uvs[perm],
        normals=None if normals is None else normals[perm],
    )


def concat_groups(meshes: list[PolyMesh]) -> np.ndarray | None:
    """Face groups of meshes placed one after the other (missing groups are group 0)."""
    if all(m.groups is None for m in meshes):
        return None
    return np.concatenate(
        [m.groups if m.groups is not None else np.zeros(m.face_count, np.int64) for m in meshes]
    )


def concat(meshes: list[PolyMesh]) -> PolyMesh:
    """Disjoint union of meshes (layers kept only if every mesh has them; face groups if any
    has them)."""
    meshes = [m for m in meshes if m.face_count or m.vertex_count]
    if not meshes:
        return PolyMesh.empty()
    vertex_base = np.cumsum([0] + [m.vertex_count for m in meshes[:-1]])
    corner_base = np.cumsum([0] + [m.corner_count for m in meshes[:-1]])
    offsets = np.concatenate(
        [[0]] + [m.face_offsets[1:] + cb for m, cb in zip(meshes, corner_base, strict=True)]
    )
    has_uv = all(m.uvs is not None for m in meshes)
    has_n = all(m.normals is not None for m in meshes)
    return PolyMesh(
        np.concatenate([m.positions for m in meshes]),
        offsets,
        np.concatenate([m.face_vertices + vb for m, vb in zip(meshes, vertex_base, strict=True)]),
        uvs=np.concatenate([m.uvs for m in meshes]) if has_uv else None,
        normals=np.concatenate([m.normals for m in meshes]) if has_n else None,
        normals_valid=np.concatenate([m.normals_valid for m in meshes]) if has_n else None,
        smooth=np.concatenate([m.smooth for m in meshes]),
        sharp_edges=np.concatenate(
            [m.sharp_edges + vb for m, vb in zip(meshes, vertex_base, strict=True)]
        ),
        groups=concat_groups(meshes),
    )


def _drop_degenerate_faces(mesh: PolyMesh, fv: np.ndarray) -> tuple[PolyMesh, np.ndarray]:
    """Apply a new corner->vertex array, collapsing repeated consecutive vertices.

    Faces left with fewer than 3 distinct corners are removed. Returns the new
    mesh and the ids of removed faces.
    """
    nxt = mesh.next_corner
    keep_corner = fv != fv[nxt]
    sizes = (
        np.add.reduceat(keep_corner.astype(np.int64), mesh.face_offsets[:-1])
        if mesh.face_count
        else []
    )
    sizes = np.asarray(sizes)
    keep_face = sizes >= 3
    # A face that repeats a vertex non-consecutively is not a simple polygon: drop it too.
    if keep_face.any():
        cf = mesh.corner_face
        kept = keep_corner & keep_face[cf]
        pairs = np.unique(np.stack([cf[kept], fv[kept]], axis=1), axis=0)
        distinct = np.bincount(pairs[:, 0], minlength=mesh.face_count)
        keep_face &= distinct == sizes
    removed = np.flatnonzero(~keep_face)
    corner_keep = keep_corner & keep_face[mesh.corner_face]
    new_sizes = sizes[keep_face]
    offsets = np.concatenate([[0], np.cumsum(new_sizes)])
    out = PolyMesh(
        mesh.positions,
        offsets,
        fv[corner_keep],
        uvs=None if mesh.uvs is None else mesh.uvs[corner_keep],
        normals=None if mesh.normals is None else mesh.normals[corner_keep],
        normals_valid=mesh.normals_valid[keep_face] if mesh.normals is not None else None,
        smooth=mesh.smooth[keep_face],
        sharp_edges=mesh.sharp_edges,
        groups=None if mesh.groups is None else mesh.groups[keep_face],
    )
    return out, removed


def compact(mesh: PolyMesh) -> PolyMesh:
    """Remove vertices no face uses (renumbering vertices, keeping their order)."""
    used = np.zeros(mesh.vertex_count, bool)
    used[mesh.face_vertices] = True
    if used.all():
        return mesh
    remap = np.cumsum(used) - 1
    sharp = mesh.sharp_edges
    if len(sharp):
        sharp = remap[sharp[used[sharp].all(axis=1)]]
    return mesh.with_(
        positions=mesh.positions[used], face_vertices=remap[mesh.face_vertices], sharp_edges=sharp
    )


def _invalidate_faces_touching(mesh: PolyMesh, vertices: np.ndarray) -> np.ndarray:
    if mesh.normals is None or len(vertices) == 0:
        return mesh.normals_valid
    mark = np.zeros(mesh.vertex_count, bool)
    mark[vertices] = True
    touching = np.zeros(mesh.face_count, bool)
    np.logical_or.at(touching, mesh.corner_face, mark[mesh.face_vertices])
    return mesh.normals_valid & ~touching


def weld(
    mesh: PolyMesh, candidates: np.ndarray, group: np.ndarray, distance: float
) -> tuple[PolyMesh, int]:
    """Merge candidate vertices from *different* groups closer than ``distance``.

    Clusters are transitive and merge to their lowest vertex id (keeping its
    position). Returns the welded mesh and the number of merged vertices.
    """
    if len(candidates) == 0:
        return mesh, 0
    pts = mesh.positions[candidates]
    cell = max(distance, 1e-12)
    keys = np.floor(pts / cell).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i, key in enumerate(map(tuple, keys.tolist())):
        buckets.setdefault(key, []).append(i)
    parent = list(range(len(candidates)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for key, members in buckets.items():
        near = [
            m
            for o in offsets
            for m in buckets.get((key[0] + o[0], key[1] + o[1], key[2] + o[2]), ())
        ]
        near_arr = np.asarray(near)
        for i in members:
            close = near_arr[np.linalg.norm(pts[near_arr] - pts[i], axis=1) <= distance]
            for j in close.tolist():
                if group[candidates[j]] != group[candidates[i]]:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    target = np.arange(mesh.vertex_count)
    merged = 0
    for i in range(len(candidates)):
        r = find(i)
        if r != i:
            target[candidates[i]] = candidates[r]
            merged += 1
    if not merged:
        return mesh, 0
    moved = np.flatnonzero(target != np.arange(mesh.vertex_count))
    sharp = target[mesh.sharp_edges] if len(mesh.sharp_edges) else mesh.sharp_edges
    if len(sharp):
        sharp = sharp[sharp[:, 0] != sharp[:, 1]]
    mesh = mesh.with_(
        sharp_edges=sharp,
        normals_valid=_invalidate_faces_touching(
            mesh, np.unique(np.concatenate([moved, target[moved]]))
        ),
    )
    out, _ = _drop_degenerate_faces(mesh, target[mesh.face_vertices])
    return compact(out), merged


def _require_topology(mesh: PolyMesh, location: str, what: str, *, manifold: bool = True) -> None:
    report = mesh.validate()
    s = report.summary
    if manifold and not s["edge_manifold"]:
        raise ProceduralError(
            "UNSUPPORTED_TOPOLOGY",
            f"{what} needs edge-manifold input (no edge with more than two faces)",
            location,
            details={"report": report.to_dict()},
        )
    if not s["consistently_oriented"]:
        raise ProceduralError(
            "INCONSISTENT_WINDING",
            f"{what} needs consistently wound input",
            location,
            details={"report": report.to_dict()},
        )


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #


class TransformModifier(ModifierType):
    name = "transform"
    description = (
        "Affine transform of the geometry: scale and rotate (XYZ Euler degrees) about a pivot, "
        "then translate. Mirroring scales re-wind faces; stored normals are transformed exactly."
    )
    params = (
        ParamSpec("translation", "vec3", (0.0, 0.0, 0.0), description="offset added last"),
        ParamSpec("rotation", "vec3", (0.0, 0.0, 0.0), description="XYZ Euler angles in degrees"),
        ParamSpec("scale", "vec3", (1.0, 1.0, 1.0), description="per-axis scale (non-zero)"),
        ParamSpec("pivot", "vec3", (0.0, 0.0, 0.0), description="center of rotation and scale"),
    )

    def check(self, values: dict, location: str) -> None:
        if any(s == 0 for s in values["scale"]):
            raise ProceduralError(
                "DEGENERATE_TRANSFORM", "scale factors must be non-zero", f"{location}.scale"
            )

    def apply(self, mesh, values, ctx):
        m = _affine(values["translation"], values["rotation"], values["scale"], values["pivot"])
        return transform_mesh(mesh, m, ctx.location), []


# --------------------------------------------------------------------------- #
# Mirror
# --------------------------------------------------------------------------- #


class MirrorModifier(ModifierType):
    name = "mirror"
    description = (
        "Adds a mirrored copy across the planes through `origin` perpendicular to each axis in "
        "`axes` (applied in X, Y, Z order). With `merge`, vertices within `merge_threshold` of a "
        "plane are snapped onto it and shared by both halves, closing the seam."
    )
    params = (
        ParamSpec(
            "axes",
            "enum",
            "x",
            choices=("x", "y", "z", "xy", "xz", "yz", "xyz"),
            description="mirror axes",
        ),
        ParamSpec("origin", "vec3", (0.0, 0.0, 0.0), description="a point on every mirror plane"),
        ParamSpec("merge", "bool", True, description="weld vertices on the mirror plane"),
        ParamSpec(
            "merge_threshold",
            "float",
            1e-4,
            min=0.0,
            max=1e6,
            description="distance from the plane within which vertices are welded",
        ),
        ParamSpec("mirror_u", "bool", False, description="mirror U (u -> 1 - u) on the copy"),
        ParamSpec("mirror_v", "bool", False, description="mirror V (v -> 1 - v) on the copy"),
    )

    def apply(self, mesh, values, ctx):
        diagnostics = []
        for axis_name in values["axes"]:
            axis = "xyz".index(axis_name)
            mesh, diags = self._mirror(mesh, axis, values, ctx.location)
            diagnostics += diags
        return mesh, diagnostics

    def _mirror(self, mesh: PolyMesh, axis: int, values: dict, location: str):
        _limit(2 * mesh.face_count, location)
        origin = values["origin"][axis]
        threshold = values["merge_threshold"] if values["merge"] else -1.0
        positions = np.array(mesh.positions)
        d = positions[:, axis] - origin
        on_plane = np.abs(d) <= threshold
        positions[on_plane, axis] = origin
        diagnostics = []
        tolerance = max(threshold, 0.0)
        if np.any(d > tolerance) and np.any(d < -tolerance):
            diagnostics.append(
                _diag(
                    "MIRROR_OVERLAP",
                    WARNING,
                    f"geometry lies on both sides of the {'xyz'[axis]} mirror plane: the "
                    "mirrored copy overlaps it (and vertices merged on the plane can make "
                    "the result non-manifold); mirror geometry that lies on one side",
                    location,
                )
            )
        v = mesh.vertex_count
        copy_index = np.where(on_plane, np.arange(v), v + np.cumsum(~on_plane) - 1)
        new_positions = positions[~on_plane].copy()
        new_positions[:, axis] = 2 * origin - new_positions[:, axis]
        face_on_plane = np.ones(mesh.face_count, bool)
        np.logical_and.at(face_on_plane, mesh.corner_face, on_plane[mesh.face_vertices])
        # With merging, a face on the plane whose every edge also borders an off-plane
        # face ends up sandwiched between the two halves (e.g. the cap of a half
        # model): it is removed, and reported. Other faces on the plane are kept once.
        interior = np.zeros(mesh.face_count, bool)
        if values["merge"] and face_on_plane.any():
            _, corner_edge, _ = mesh.edges
            off_plane = np.zeros(int(corner_edge.max()) + 1, np.int64)
            np.add.at(off_plane, corner_edge, ~face_on_plane[mesh.corner_face])
            interior = face_on_plane.copy()
            np.logical_and.at(interior, mesh.corner_face, off_plane[corner_edge] > 0)
        if interior.any():
            diagnostics.append(
                _diag(
                    "MIRROR_INTERIOR_FACES_REMOVED",
                    INFO,
                    "faces on the mirror plane that would lie inside the merged result "
                    "were removed",
                    location,
                    np.flatnonzero(interior),
                )
            )
        if not values["merge"]:
            exactly_on = np.ones(mesh.face_count, bool)
            np.logical_and.at(exactly_on, mesh.corner_face, (d == 0)[mesh.face_vertices])
            if exactly_on.any():
                diagnostics.append(
                    _diag(
                        "MIRROR_COINCIDENT_FACES",
                        WARNING,
                        "faces lying in the mirror plane coincide with their mirrored copies "
                        "(merge is off)",
                        location,
                        np.flatnonzero(exactly_on),
                    )
                )
        if (face_on_plane & ~interior).any():
            diagnostics.append(
                _diag(
                    "MIRROR_FACES_ON_PLANE",
                    WARNING,
                    "faces lying in the mirror plane "
                    "are not duplicated (they would coincide with their mirror image)",
                    location,
                    np.flatnonzero(face_on_plane & ~interior),
                )
            )
        base = mesh.with_(positions=positions)
        perm = reverse_corners(base)
        reflected = base.with_(
            positions=np.concatenate([positions, new_positions]),
            face_vertices=copy_index[base.face_vertices[perm]],
            uvs=None if base.uvs is None else self._mirror_uv(base.uvs[perm], values),
            normals=None if base.normals is None else self._reflect(base.normals[perm], axis),
            sharp_edges=copy_index[base.sharp_edges] if len(base.sharp_edges) else base.sharp_edges,
        )
        keep = ~face_on_plane
        original = base.with_(positions=np.concatenate([positions, new_positions]))
        if interior.any():
            original = _select_faces(original, ~interior)
        copy = _select_faces(reflected, keep)
        out = concat_shared(original, copy)
        seam = np.flatnonzero(on_plane)
        out = out.with_(normals_valid=_invalidate_faces_touching(out, seam))
        return compact(out), diagnostics

    @staticmethod
    def _reflect(normals: np.ndarray, axis: int) -> np.ndarray:
        n = np.array(normals)
        n[:, axis] = -n[:, axis]
        return n

    @staticmethod
    def _mirror_uv(uvs: np.ndarray, values: dict) -> np.ndarray:
        uv = np.array(uvs)
        if values["mirror_u"]:
            uv[:, 0] = 1 - uv[:, 0]
        if values["mirror_v"]:
            uv[:, 1] = 1 - uv[:, 1]
        return uv


def _select_faces(mesh: PolyMesh, keep: np.ndarray) -> PolyMesh:
    corner_keep = keep[mesh.corner_face]
    offsets = np.concatenate([[0], np.cumsum(mesh.face_sizes[keep])])
    return PolyMesh(
        mesh.positions,
        offsets,
        mesh.face_vertices[corner_keep],
        uvs=None if mesh.uvs is None else mesh.uvs[corner_keep],
        normals=None if mesh.normals is None else mesh.normals[corner_keep],
        normals_valid=mesh.normals_valid[keep] if mesh.normals is not None else None,
        smooth=mesh.smooth[keep],
        sharp_edges=mesh.sharp_edges,
        groups=None if mesh.groups is None else mesh.groups[keep],
    )


def concat_shared(a: PolyMesh, b: PolyMesh) -> PolyMesh:
    """Faces of ``a`` and ``b`` over the same vertex array (``a.positions``)."""
    has_uv = a.uvs is not None and b.uvs is not None
    has_n = a.normals is not None and b.normals is not None
    return PolyMesh(
        a.positions,
        np.concatenate([a.face_offsets, b.face_offsets[1:] + a.corner_count]),
        np.concatenate([a.face_vertices, b.face_vertices]),
        uvs=np.concatenate([a.uvs, b.uvs]) if has_uv else None,
        normals=np.concatenate([a.normals, b.normals]) if has_n else None,
        normals_valid=np.concatenate([a.normals_valid, b.normals_valid]) if has_n else None,
        smooth=np.concatenate([a.smooth, b.smooth]),
        sharp_edges=np.concatenate([a.sharp_edges, b.sharp_edges]),
        groups=concat_groups([a, b]),
    )


# --------------------------------------------------------------------------- #
# Array
# --------------------------------------------------------------------------- #


class ArrayModifier(ModifierType):
    name = "array"
    description = (
        "Repeats the geometry `count` times. Each copy is the previous one moved by the step "
        "transform: translate by relative_offset x (bounding-box size) + constant_offset after "
        "rotating/scaling about `pivot`. With `merge`, open-boundary vertices of neighbouring "
        "copies closer than `merge_distance` are welded."
    )
    params = (
        ParamSpec(
            "count",
            "int",
            2,
            min=1,
            max=100_000,
            description="number of copies (including the first)",
        ),
        ParamSpec(
            "relative_offset",
            "vec3",
            (1.0, 0.0, 0.0),
            description="step as a fraction of the bounding-box size per axis",
        ),
        ParamSpec(
            "constant_offset", "vec3", (0.0, 0.0, 0.0), description="additional step in units"
        ),
        ParamSpec(
            "rotation", "vec3", (0.0, 0.0, 0.0), description="per-step rotation (XYZ Euler degrees)"
        ),
        ParamSpec("scale", "vec3", (1.0, 1.0, 1.0), description="per-step scale (non-zero)"),
        ParamSpec(
            "pivot", "vec3", (0.0, 0.0, 0.0), description="center of the per-step rotation/scale"
        ),
        ParamSpec(
            "merge",
            "bool",
            False,
            description="weld touching boundary vertices of neighbouring copies",
        ),
        ParamSpec(
            "merge_distance", "float", 1e-4, min=0.0, max=1e6, description="welding distance"
        ),
    )

    def check(self, values: dict, location: str) -> None:
        if any(s == 0 for s in values["scale"]):
            raise ProceduralError(
                "DEGENERATE_TRANSFORM", "scale factors must be non-zero", f"{location}.scale"
            )

    def apply(self, mesh, values, ctx):
        count = values["count"]
        _limit(count * mesh.face_count, ctx.location)
        if count == 1 or mesh.vertex_count == 0:
            return mesh, []
        bounds = mesh.bounds()
        size = bounds[1] - bounds[0] if bounds is not None else np.zeros(3)
        step = _affine(
            np.asarray(values["relative_offset"]) * size + np.asarray(values["constant_offset"]),
            values["rotation"],
            values["scale"],
            values["pivot"],
        )
        copies, m = [mesh], np.eye(4)
        for _ in range(1, count):
            m = step @ m
            copies.append(transform_mesh(mesh, m, ctx.location))
        out = concat(copies)
        diagnostics = []
        if values["merge"]:
            _, corner_edge, counts = mesh.edges
            boundary_vertices = np.unique(mesh.face_vertices[counts[corner_edge] == 1])
            group = np.repeat(np.arange(count), mesh.vertex_count)
            candidates = (
                boundary_vertices[None, :] + (np.arange(count) * mesh.vertex_count)[:, None]
            ).ravel()
            before = out.face_count
            out, merged = weld(out, candidates, group, values["merge_distance"])
            diagnostics.append(
                _diag(
                    "ARRAY_MERGED", INFO, f"welded {merged} vertices between copies", ctx.location
                )
            )
            if out.face_count != before:
                diagnostics.append(
                    _diag(
                        "FACES_COLLAPSED",
                        WARNING,
                        f"{before - out.face_count} faces collapsed while welding and were removed",
                        ctx.location,
                    )
                )
        return out, diagnostics


# --------------------------------------------------------------------------- #
# Solidify
# --------------------------------------------------------------------------- #


def vertex_normals(mesh: PolyMesh) -> np.ndarray:
    """Angle-weighted vertex normals (unit; zero where undefined)."""
    fn = mesh.face_normals
    length = np.linalg.norm(fn, axis=1, keepdims=True)
    unit = np.divide(fn, length, out=np.zeros_like(fn), where=length > 1e-300)
    p = mesh.positions[mesh.face_vertices]
    u = mesh.positions[mesh.face_vertices[mesh.prev_corner]] - p
    w = mesh.positions[mesh.face_vertices[mesh.next_corner]] - p
    cosang = np.einsum("ij,ij->i", u, w) / np.maximum(
        np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300
    )
    angle = np.arccos(np.clip(cosang, -1.0, 1.0))
    sums = np.zeros((mesh.vertex_count, 3))
    np.add.at(sums, mesh.face_vertices, unit[mesh.corner_face] * angle[:, None])
    n = np.linalg.norm(sums, axis=1, keepdims=True)
    return np.divide(sums, n, out=np.zeros_like(sums), where=n > 1e-12)


class SolidifyModifier(ModifierType):
    name = "solidify"
    description = (
        "Gives surfaces thickness: the surface is offset along vertex normals into two shells "
        "`thickness` apart (offset -1 keeps the original as the outer side, +1 as the inner side, "
        "0 centers it); open boundaries are closed with rim quads. Even thickness compensates for "
        "corners."
    )
    params = (
        ParamSpec(
            "thickness", "float", 0.1, min=-1e6, max=1e6, description="shell thickness (non-zero)"
        ),
        ParamSpec(
            "offset",
            "float",
            -1.0,
            min=-1.0,
            max=1.0,
            description="where the original surface sits",
        ),
        ParamSpec(
            "even_thickness",
            "bool",
            False,
            description="keep the wall thickness constant at corners",
        ),
        ParamSpec("rim", "bool", True, description="close open boundaries with rim faces"),
    )

    def check(self, values: dict, location: str) -> None:
        if values["thickness"] == 0:
            raise ProceduralError(
                "INVALID_PARAMETER", "thickness must be non-zero", f"{location}.thickness"
            )

    def apply(self, mesh, values, ctx):
        if mesh.face_count == 0:
            return mesh, []
        _require_topology(mesh, ctx.location, "solidify")
        _limit(2 * mesh.face_count + mesh.corner_count, ctx.location)
        vn = vertex_normals(mesh)
        used = np.zeros(mesh.vertex_count, bool)
        used[mesh.face_vertices] = True
        undefined = np.flatnonzero(used & (np.linalg.norm(vn, axis=1) < 0.5))
        if len(undefined):
            raise ProceduralError(
                "UNDEFINED_NORMAL",
                "some vertices have no defined normal (their faces cancel out or have zero area)",
                ctx.location,
                details={"vertices": undefined[:MAX_SAMPLES].tolist()},
            )
        factor = np.ones(mesh.vertex_count)
        if values["even_thickness"]:
            fn = mesh.face_normals / np.maximum(
                np.linalg.norm(mesh.face_normals, axis=1, keepdims=True), 1e-300
            )
            dots = np.einsum("ij,ij->i", vn[mesh.face_vertices], fn[mesh.corner_face])
            total = np.zeros(mesh.vertex_count)
            count = np.zeros(mesh.vertex_count)
            np.add.at(total, mesh.face_vertices, dots)
            np.add.at(count, mesh.face_vertices, 1)
            mean = np.divide(total, count, out=np.ones_like(total), where=count > 0)
            factor = 1.0 / np.clip(mean, 0.25, 1.0)
        t, off = values["thickness"], values["offset"]
        outer = mesh.positions + vn * (t * (1 + off) / 2) * factor[:, None]
        inner = mesh.positions - vn * (t * (1 - off) / 2) * factor[:, None]
        v = mesh.vertex_count
        perm = reverse_corners(mesh)
        faces_outer = mesh.with_(
            positions=np.concatenate([outer, inner]),
            normals_valid=np.zeros(mesh.face_count, bool) if mesh.normals is not None else None,
        )
        faces_inner = faces_outer.with_(
            face_vertices=mesh.face_vertices[perm] + v,
            uvs=None if mesh.uvs is None else mesh.uvs[perm],
            normals=None if mesh.normals is None else mesh.normals[perm],
            sharp_edges=mesh.sharp_edges + v,
        )
        out = concat_shared(faces_outer, faces_inner)
        diagnostics = []
        _, corner_edge, counts = mesh.edges
        boundary = np.flatnonzero(counts[corner_edge] == 1)
        if values["rim"] and len(boundary):
            a = mesh.face_vertices[boundary]
            b = mesh.face_vertices[mesh.next_corner[boundary]]
            loops = np.stack([b, a, a + v, b + v], axis=1)
            k = len(boundary)
            rim_uv = None
            if mesh.uvs is not None:
                ua, ub = mesh.uvs[boundary], mesh.uvs[mesh.next_corner[boundary]]
                rim_uv = np.stack([ub, ua, ua, ub], axis=1).reshape(-1, 2)
                diagnostics.append(
                    _diag(
                        "UV_DEGENERATE_FACES",
                        WARNING,
                        "rim faces copy the UVs of the boundary edge and have zero UV area",
                        ctx.location,
                    )
                )
            rims = PolyMesh(
                out.positions,
                np.arange(0, 4 * k + 1, 4),
                loops.ravel(),
                uvs=rim_uv,
                normals=np.zeros((4 * k, 3)) if out.normals is not None else None,
                normals_valid=np.zeros(k, bool) if out.normals is not None else None,
                smooth=np.zeros(k, bool),
                # A rim face belongs to the group of the face whose boundary edge it closes.
                groups=None if mesh.groups is None else mesh.groups[mesh.corner_face[boundary]],
            )
            out = concat_shared(out, rims)
        elif len(boundary):
            diagnostics.append(
                _diag(
                    "SOLIDIFY_OPEN",
                    INFO,
                    "rim is disabled: the result has open boundaries",
                    ctx.location,
                )
            )
        return out, diagnostics


# --------------------------------------------------------------------------- #
# Subdivision surface (Catmull-Clark)
# --------------------------------------------------------------------------- #


class SubdivisionModifier(ModifierType):
    name = "subdivision"
    description = (
        "Catmull-Clark subdivision surface: every n-gon becomes n quads and vertices move toward "
        "the smooth limit surface. Open boundaries follow the cubic B-spline boundary rule. UVs "
        "are subdivided linearly per face (seams are kept). Needs edge-manifold, consistently "
        "wound input."
    )
    params = (
        ParamSpec("levels", "int", 1, min=0, max=6, description="number of subdivision steps"),
        ParamSpec("shade_smooth", "bool", True, description="mark the result smooth-shaded"),
    )

    def apply(self, mesh, values, ctx):
        levels = values["levels"]
        if levels == 0 or mesh.face_count == 0:
            return mesh, []
        _require_topology(mesh, ctx.location, "subdivision")
        _limit(int(mesh.corner_count * 4 ** (levels - 1)), ctx.location)
        for _ in range(levels):
            mesh = catmull_clark(compact(mesh))
        if values["shade_smooth"]:
            mesh = mesh.with_(smooth=np.ones(mesh.face_count, bool))
        return mesh, []


def catmull_clark(mesh: PolyMesh) -> PolyMesh:
    """One Catmull-Clark step on an edge-manifold polygon mesh."""
    v = mesh.vertex_count
    p = mesh.positions
    fv = mesh.face_vertices
    cf = mesh.corner_face
    sizes = mesh.face_sizes
    edge_vertices, corner_edge, counts = mesh.edges
    e = len(edge_vertices)
    face_points = np.add.reduceat(p[fv], mesh.face_offsets[:-1], axis=0) / sizes[:, None]
    midpoints = (p[edge_vertices[:, 0]] + p[edge_vertices[:, 1]]) / 2
    face_sum = np.zeros((e, 3))
    np.add.at(face_sum, corner_edge, face_points[cf])
    boundary_edge = counts == 1
    edge_points = np.where(
        boundary_edge[:, None],
        midpoints,
        (p[edge_vertices[:, 0]] + p[edge_vertices[:, 1]] + face_sum) / 4,
    )
    # Vertex points.
    n_faces = np.bincount(fv, minlength=v).astype(np.float64)
    f_sum = np.zeros((v, 3))
    np.add.at(f_sum, fv, face_points[cf])
    n_edges = np.bincount(edge_vertices.ravel(), minlength=v).astype(np.float64)
    r_sum = np.zeros((v, 3))
    np.add.at(r_sum, edge_vertices[:, 0], midpoints)
    np.add.at(r_sum, edge_vertices[:, 1], midpoints)
    with np.errstate(invalid="ignore", divide="ignore"):
        favg = f_sum / n_faces[:, None]
        ravg = r_sum / n_edges[:, None]
        nn = n_faces[:, None]
        interior = (favg + 2 * ravg + (nn - 3) * p) / nn
    b_edges = edge_vertices[boundary_edge]
    b_count = np.bincount(b_edges.ravel(), minlength=v)
    b_sum = np.zeros((v, 3))
    np.add.at(b_sum, b_edges[:, 0], p[b_edges[:, 1]])
    np.add.at(b_sum, b_edges[:, 1], p[b_edges[:, 0]])
    boundary_rule = (b_sum + 6 * p) / 8
    vertex_points = np.where(
        (b_count == 0)[:, None], interior, np.where((b_count == 2)[:, None], boundary_rule, p)
    )
    positions = np.concatenate([vertex_points, edge_points, face_points])
    prev = mesh.prev_corner
    nxt = mesh.next_corner
    quads = np.stack([fv, v + corner_edge, v + e + cf, v + corner_edge[prev]], axis=1)
    uvs = None
    if mesh.uvs is not None:
        uv = mesh.uvs
        face_uv = np.add.reduceat(uv, mesh.face_offsets[:-1], axis=0) / sizes[:, None]
        uvs = np.stack([uv, (uv + uv[nxt]) / 2, face_uv[cf], (uv[prev] + uv) / 2], axis=1).reshape(
            -1, 2
        )
    sharp = np.zeros((0, 2), np.int64)
    if len(mesh.sharp_edges):
        key = edge_vertices[:, 0] * (v + 1) + edge_vertices[:, 1]
        skey = mesh.sharp_edges[:, 0] * (v + 1) + mesh.sharp_edges[:, 1]
        hit = np.flatnonzero(np.isin(key, skey))
        if len(hit):
            ends = edge_vertices[hit]
            sharp = np.concatenate(
                [np.stack([ends[:, 0], v + hit], 1), np.stack([ends[:, 1], v + hit], 1)]
            )
    c = mesh.corner_count
    return PolyMesh(
        positions,
        np.arange(0, 4 * c + 1, 4),
        quads.ravel(),
        uvs=uvs,
        normals=np.zeros((4 * c, 3)) if mesh.normals is not None else None,
        normals_valid=np.zeros(c, bool) if mesh.normals is not None else None,
        smooth=mesh.smooth[cf],
        sharp_edges=sharp,
        groups=None if mesh.groups is None else mesh.groups[cf],
    )


# --------------------------------------------------------------------------- #
# Bevel (reuses the mesh editing engine)
# --------------------------------------------------------------------------- #


class BevelModifier(ModifierType):
    name = "bevel"
    description = (
        "Bevels edges with the mesh editing engine's bevel (same geometry, attribute and "
        "topology rules): a chamfer, or with `segments` > 1 a rounded strip shaped by "
        "`profile` (0.5: circular arc). `limit` selects all edges, edges whose faces meet at "
        "`angle` degrees or more, or edges marked sharp."
    )
    params = (
        ParamSpec(
            "width", "float", 0.05, min=1e-9, max=1e6, description="chamfer width within each face"
        ),
        ParamSpec(
            "segments",
            "int",
            1,
            min=1,
            max=64,
            description="faces across each bevelled edge (1: a chamfer)",
        ),
        ParamSpec(
            "profile",
            "float",
            0.5,
            min=0.01,
            max=0.99,
            description="shape of the cross-section: 0.5 circular, 0.25 straight, toward 1 square",
        ),
        ParamSpec(
            "limit", "enum", "angle", choices=("all", "angle", "sharp"), description="which edges"
        ),
        ParamSpec(
            "angle",
            "float",
            30.0,
            min=0.0,
            max=180.0,
            description="minimum angle for limit='angle'",
        ),
    )

    def apply(self, mesh, values, ctx):
        if mesh.face_count == 0:
            return mesh, []
        em = mesh.to_editable()
        if values["limit"] == "all":
            edges = em.edge_ids()
        elif values["limit"] == "sharp":
            edges = sorted(em._sharp)
        else:
            cos_limit = math.cos(math.radians(values["angle"]))
            edges = []
            for eid, faces in em._edge_faces.items():
                if len(faces) == 2:
                    n0, n1 = em._face_normal_vec(faces[0]), em._face_normal_vec(faces[1])
                    if (
                        n0 is not None
                        and n1 is not None
                        and float(np.dot(n0, n1)) <= cos_limit + 1e-12
                    ):
                        edges.append(eid)
        if not edges:
            return mesh, [
                _diag(
                    "BEVEL_NO_EDGES",
                    INFO,
                    "no edges matched the limit; nothing was bevelled",
                    ctx.location,
                )
            ]
        try:
            result = em.bevel_edges(
                sorted(edges),
                values["width"],
                segments=values["segments"],
                profile=values["profile"],
            )
        except EditError as exc:
            raise ProceduralError(
                exc.code,
                exc.message,
                ctx.location,
                hint=exc.hint,
                details={"elements": exc.elements},
            ) from None
        return PolyMesh.from_editable(em), list(result.diagnostics)


# --------------------------------------------------------------------------- #
# Boolean (manifold3d backend)
# --------------------------------------------------------------------------- #


class BooleanModifier(ModifierType):
    name = "boolean"
    description = (
        "Robust solid boolean (union, difference, intersection) with another procedural "
        "geometry, using the Manifold library (manifold3d, Apache-2.0), whose results are always "
        "manifold. Both inputs must be closed, edge- and vertex-manifold, consistently wound, "
        "outward-facing solids. UVs are carried through; the result is triangulated."
    )
    params = (
        ParamSpec(
            "operation",
            "enum",
            "difference",
            choices=("union", "difference", "intersection"),
            description="boolean operation",
        ),
        ParamSpec(
            "operand",
            "operand",
            None,
            required=True,
            description='{"geometry": id or name, "parameters": {overrides}}',
        ),
        ParamSpec("operand_translation", "vec3", (0.0, 0.0, 0.0), description="operand placement"),
        ParamSpec(
            "operand_rotation", "vec3", (0.0, 0.0, 0.0), description="operand rotation (XYZ deg)"
        ),
        ParamSpec("operand_scale", "vec3", (1.0, 1.0, 1.0), description="operand scale (non-zero)"),
    )

    def check(self, values: dict, location: str) -> None:
        operand = values["operand"]
        if (
            not isinstance(operand, dict)
            or "geometry" not in operand
            or not set(operand) <= {"geometry", "parameters"}
        ):
            raise ProceduralError(
                "INVALID_PARAMETER",
                'operand must be {"geometry": id or name, "parameters": {...}}',
                f"{location}.operand",
            )
        if not isinstance(operand.get("parameters", {}), dict):
            raise ProceduralError(
                "INVALID_PARAMETER",
                "operand parameters must be an object",
                f"{location}.operand.parameters",
            )
        if any(s == 0 for s in values["operand_scale"]):
            raise ProceduralError(
                "DEGENERATE_TRANSFORM",
                "operand_scale must be non-zero",
                f"{location}.operand_scale",
            )

    def references(self, values: dict) -> list[Any]:
        operand = values.get("operand")
        return [operand["geometry"]] if isinstance(operand, dict) and "geometry" in operand else []

    def apply(self, mesh, values, ctx):
        try:
            import manifold3d
        except ImportError:
            raise ProceduralError(
                "BOOLEAN_BACKEND_UNAVAILABLE",
                "the boolean modifier needs the manifold3d package",
                ctx.location,
                hint="pip install 'aether3d[boolean]'",
            ) from None
        operand = ctx.evaluate_operand(values["operand"], f"{ctx.location}.operand")
        m = _affine(
            values["operand_translation"], values["operand_rotation"], values["operand_scale"]
        )
        operand = transform_mesh(operand, m, ctx.location)
        for which, solid in (("input", mesh), ("operand", operand)):
            _require_solid(
                solid, f"{ctx.location}" + ("" if which == "input" else ".operand"), which
            )
        diagnostics = []
        with_uv = mesh.uvs is not None and operand.uvs is not None
        if (mesh.uvs is None) != (operand.uvs is None):
            diagnostics.append(
                _diag(
                    "UVS_DROPPED",
                    WARNING,
                    "only one boolean input has UVs; the result has none",
                    ctx.location,
                )
            )
        a = _to_manifold(manifold3d, mesh, with_uv, 0)
        b = _to_manifold(manifold3d, operand, with_uv, mesh.face_count)
        op = values["operation"]
        result = a + b if op == "union" else (a - b if op == "difference" else a ^ b)
        status = result.status()
        if str(status) not in ("Error.NoError", "NoError"):
            raise ProceduralError(
                "BOOLEAN_FAILED", f"the boolean backend reported {status}", ctx.location
            )
        out = _from_manifold(
            result.to_mesh(),
            with_uv,
            np.concatenate([mesh.smooth, operand.smooth]),
            mesh.normals is not None,
            concat_groups([mesh, operand]),
        )
        if out.face_count == 0:
            diagnostics.append(
                _diag("BOOLEAN_EMPTY_RESULT", WARNING, f"the {op} is empty", ctx.location)
            )
        elif out.groups is None and concat_groups([mesh, operand]) is not None:
            diagnostics.append(
                _diag(
                    "FACE_GROUPS_DROPPED",
                    WARNING,
                    "the boolean backend did not report source faces; face groups were lost",
                    ctx.location,
                )
            )
        return out, diagnostics


def _require_solid(mesh: PolyMesh, location: str, which: str) -> None:
    report = mesh.validate()
    s = report.summary
    problems = [
        name
        for name, ok in (
            ("closed", s["closed"]),
            ("edge-manifold", s["edge_manifold"]),
            ("vertex-manifold", s["vertex_manifold"]),
            ("consistently wound", s["consistently_oriented"]),
        )
        if not ok
    ]
    if not problems and mesh.face_count:
        m, _ = mesh.to_mesh()
        from ..inspection import inspect_mesh

        volume = inspect_mesh(m).topology["volume"]
        if volume is None or volume <= 0:
            problems.append("outward-facing (positive volume)")
    if problems or mesh.face_count == 0:
        raise ProceduralError(
            "BOOLEAN_INPUT_NOT_SOLID",
            f"the boolean {which} is not a closed solid: it is not "
            + ", ".join(problems or ["empty"]),
            location,
            hint="booleans need closed, manifold, outward-facing meshes (primitives, solidify)",
            details={"report": report.to_dict()},
        )


def _to_manifold(manifold3d, mesh: PolyMesh, with_uv: bool, face_base: int):
    tris, _ = mesh.triangulate()
    corners = tris.ravel()
    props = [mesh.positions[mesh.face_vertices[corners]]]
    if with_uv:
        props.append(mesh.uvs[corners])
    vert_properties = np.ascontiguousarray(np.hstack(props), dtype=np.float32)
    vertex = mesh.face_vertices[corners]
    first = np.full(mesh.vertex_count, -1, np.int64)
    order = np.arange(len(vertex))
    first_seen = np.unique(vertex, return_index=True)[1]
    first[vertex[first_seen]] = first_seen
    merge_from = order[first[vertex] != order]
    merge_to = first[vertex[merge_from]]
    tri_index = np.arange(len(tris))
    face_of_tri = mesh.corner_face[tris[:, 0]]
    gl = manifold3d.Mesh(
        vert_properties=vert_properties,
        tri_verts=np.ascontiguousarray(
            tri_index.repeat(3).reshape(-1, 3) * 3 + np.array([0, 1, 2]), np.uint32
        ),
        merge_from_vert=np.ascontiguousarray(merge_from, np.uint32),
        merge_to_vert=np.ascontiguousarray(merge_to, np.uint32),
        face_id=np.ascontiguousarray(face_of_tri + face_base, np.uint32),
    )
    return manifold3d.Manifold(gl)


def _from_manifold(
    gl, with_uv: bool, smooth: np.ndarray, normals: bool, groups: np.ndarray | None = None
) -> PolyMesh:
    props = np.asarray(gl.vert_properties, dtype=np.float64)
    tris = np.asarray(gl.tri_verts, dtype=np.int64).reshape(-1, 3)
    if len(tris) == 0:
        return PolyMesh.empty(uvs=with_uv, normals=normals)
    topo = np.arange(len(props))
    mf, mt = np.asarray(gl.merge_from_vert, np.int64), np.asarray(gl.merge_to_vert, np.int64)
    topo[mf] = mt
    uniq, vertex_of = np.unique(topo, return_inverse=True)
    vertex_of = vertex_of.reshape(-1)
    face_id = np.asarray(gl.face_id, dtype=np.int64)
    k = len(tris)
    return PolyMesh(
        props[uniq, :3],
        np.arange(0, 3 * k + 1, 3),
        vertex_of[tris].ravel(),
        uvs=props[tris.ravel(), 3:5] if with_uv else None,
        normals=np.zeros((3 * k, 3)) if normals else None,
        normals_valid=np.zeros(k, bool) if normals else None,
        smooth=smooth[face_id] if len(face_id) == k else np.zeros(k, bool),
        # Manifold reports every output triangle's source face, so groups follow it.
        groups=None if groups is None else (groups[face_id] if len(face_id) == k else None),
    )


MODIFIER_TYPES: dict[str, ModifierType] = {
    t.name: t
    for t in (
        TransformModifier(),
        MirrorModifier(),
        ArrayModifier(),
        SolidifyModifier(),
        SubdivisionModifier(),
        BevelModifier(),
        BooleanModifier(),
    )
}
