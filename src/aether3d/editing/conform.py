"""Shrinkwrap: move vertices onto another surface.

``shrinkwrap(vertices, target)`` projects vertices onto a target triangle mesh -- an
immutable :class:`~aether3d.Mesh`, or (in an editing session) another scene object, in
which case both objects' world transforms are taken into account. Two modes:

* ``"nearest"``: each vertex moves to the exact closest point of the target
  (:class:`~aether3d.spatial.surface.TriangleSurface`), then ``offset`` along the
  target's face normal there (the side the vertex came from: positive offsets keep a
  vertex outside when it was outside);
* ``"project"``: each vertex moves along ``direction`` (both ways unless
  ``both_ways=False``) to the first hit, then ``offset`` along the hit face's normal.

Vertices farther than ``max_distance``, or without a hit, stay where they are and are
reported (``SHRINKWRAP_MISSED``). The faces around moved vertices get recomputed normals
at commit; UVs are kept (the texture follows the vertices).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import vec3
from ..mesh import Mesh
from . import _polygon as poly
from .core import Elements, Selection, operation


class ConformMixin:
    @operation
    def shrinkwrap(
        self,
        vertices: Elements,
        target: Any,
        *,
        mode: str = "nearest",
        direction: Iterable[float] | None = None,
        both_ways: bool = True,
        offset: float = 0.0,
        max_distance: float | None = None,
        target_matrix: Any = None,
    ):
        """Move vertices onto ``target`` (see the module docs). ``target_matrix`` maps the
        target's coordinates into this mesh's local space (default: identity, or the
        objects' world transforms when ``target`` is a scene object)."""
        from ..spatial.surface import TriangleSurface

        vids = self._vertices_of(vertices) if not isinstance(vertices, int) else [vertices]
        vids = self._require_vertices(vids)
        if not vids:
            raise EditError("EMPTY_SELECTION", "no vertices given")
        mesh, matrix = self._resolve_target(target, target_matrix)
        if mesh.triangle_count == 0:
            raise EditError("EMPTY_TARGET", "the target has no triangles")
        pts = mesh.positions.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.all(np.isfinite(pts)):
            raise EditError("NON_FINITE_RESULT", "the target's positions are not finite here")
        surface = TriangleSurface(pts, mesh.triangles.astype(np.int64))
        limit = np.inf if max_distance is None else float(max_distance)
        if not limit > 0:
            raise ValidationError("max_distance must be > 0")
        off = float(offset)
        if not np.isfinite(off):
            raise ValidationError("offset must be finite")
        p = self._pos[vids]
        face_n = np.cross(surface.b - surface.a, surface.c - surface.a)
        lengths = np.sqrt((face_n**2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            face_n = face_n / lengths[:, None]
        if mode == "nearest":
            hit = surface.nearest(p, max_distance=limit)
            ok = hit.triangles >= 0
            new = hit.points.copy()
            n = face_n[np.maximum(hit.triangles, 0)]
            # Offset toward the side the vertex was on.
            side = np.sign(((p - hit.points) * n).sum(axis=1))
            side[side == 0] = 1.0
            new = new + (off * side)[:, None] * n
        elif mode == "project":
            if direction is None:
                raise ValidationError("mode='project' needs a direction")
            dvec = poly.unit(np.asarray(vec3(direction, "direction"), dtype=np.float64))
            if dvec is None:
                raise ValidationError("direction must be non-zero")
            dirs = np.broadcast_to(dvec, p.shape)
            fwd = surface.raycast(p, dirs, tmax=limit)
            t, tri = fwd.t, fwd.triangles
            if both_ways:
                back = surface.raycast(p, -dirs, tmax=limit)
                use_back = back.t < t
                t = np.where(use_back, -back.t, t)
                tri = np.where(use_back, back.triangles, tri)
            ok = tri >= 0
            new = p + np.where(ok, t, 0.0)[:, None] * dirs
            n = face_n[np.maximum(tri, 0)]
            new = new + off * n
        else:
            raise ValidationError(f"mode must be 'nearest' or 'project', got {mode!r}")
        ok &= np.isfinite(new).all(axis=1)
        moved = [v for v, good in zip(vids, ok, strict=True) if good]
        missed = [v for v, good in zip(vids, ok, strict=True) if not good]
        if moved:
            self._set_positions(moved, new[ok])
        if missed:
            self._diagnose(
                Diagnostic(
                    "SHRINKWRAP_MISSED",
                    INFO,
                    "vertices without a target point within reach stayed where they were",
                    None,
                    len(missed),
                    tuple(missed[:10]),
                )
            )
        return {"moved": Selection.of(moved), "missed": Selection.of(missed)}

    def _resolve_target(self, target: Any, target_matrix: Any) -> tuple[Mesh, np.ndarray]:
        from .transform import _matrix4

        matrix = np.eye(4) if target_matrix is None else _matrix4(target_matrix)
        if isinstance(target, Mesh):
            return target, matrix
        mesh = getattr(target, "mesh", None)
        world = getattr(target, "world_matrix", None)
        if isinstance(mesh, Mesh) and world is not None:
            if target_matrix is None:
                try:
                    inv = np.linalg.inv(self.world_matrix)
                except np.linalg.LinAlgError:
                    raise EditError(
                        "SINGULAR_WORLD_TRANSFORM", "the edited object's transform is singular"
                    ) from None
                matrix = inv @ np.asarray(world, dtype=np.float64)
            return mesh, matrix
        raise ValidationError("target must be a Mesh or a scene object with a mesh")
