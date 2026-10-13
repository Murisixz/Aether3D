"""Geometry transformations of vertices, edges and faces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import quat_from_axis_angle, quat_to_matrix3, scale3, vec3
from . import _polygon as poly
from .core import NORMAL, Elements, operation


def _matrix4(value: Any) -> np.ndarray:
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValidationError("matrix must be a finite 4x4 matrix")
    if not np.allclose(m[3], (0, 0, 0, 1)):
        raise ValidationError("matrix must be affine (last row 0, 0, 0, 1)")
    return m


class TransformMixin:
    """Translate, rotate, scale or arbitrarily transform selected elements.

    ``elements`` is a :class:`Selection` (all its vertices, edges and faces
    move) or an iterable of vertex ids. ``space="world"`` interprets offsets,
    axes, pivots and matrices in world space using the object's world
    transform (which may be rotated, non-uniformly scaled or mirrored); the
    edit is converted exactly to local space as ``M^-1 A M``.

    ``pivot`` is ``"center"`` (mean of the moved vertices, the default),
    ``"origin"``, or a point. Transforms that mirror (negative determinant)
    reverse the winding of every face of each connected piece that moves
    entirely, so mirrored pieces stay outward-facing; mirrored parts still
    attached to unmoved faces keep their winding (and consistent orientation).
    Faces that become zero-area raise ``DEGENERATE_RESULT`` (and nothing
    changes) unless ``allow_degenerate=True``.
    """

    @operation
    def translate(self, elements: Elements, offset: Iterable[float], *, space: str = "local"):
        """Move elements by ``offset``."""
        m = np.eye(4)
        m[:3, 3] = vec3(offset, "offset")
        return self._transform_vertices(self._vertices_of(elements), m, space, False)

    @operation
    def rotate(
        self,
        elements: Elements,
        angle: float,
        axis: Iterable[float] = (0.0, 0.0, 1.0),
        *,
        pivot: str | Iterable[float] = "center",
        space: str = "local",
    ):
        """Rotate elements ``angle`` degrees counter-clockwise about ``axis`` through ``pivot``."""
        vids = self._vertices_of(elements)
        rotation = np.eye(4)
        rotation[:3, :3] = quat_to_matrix3(quat_from_axis_angle(axis, float(angle)))
        return self._transform_vertices(
            vids, self._about(vids, rotation, pivot, space), space, False
        )

    @operation
    def scale(
        self,
        elements: Elements,
        factor: float | Iterable[float],
        *,
        pivot: str | Iterable[float] = "center",
        space: str = "local",
        allow_degenerate: bool = False,
    ):
        """Scale elements by ``factor`` (number or per-axis) about ``pivot``."""
        vids = self._vertices_of(elements)
        s = np.diag([*scale3(factor, "factor"), 1.0])
        return self._transform_vertices(
            vids, self._about(vids, s, pivot, space), space, allow_degenerate
        )

    @operation
    def transform(
        self,
        elements: Elements,
        matrix: Any,
        *,
        space: str = "local",
        allow_degenerate: bool = False,
    ):
        """Apply an affine 4x4 ``matrix`` (column vectors) to the elements' vertices."""
        vids = self._vertices_of(elements)
        return self._transform_vertices(vids, _matrix4(matrix), space, allow_degenerate)

    @operation
    def set_positions(self, positions: dict[int, Iterable[float]], *, space: str = "local"):
        """Place vertices at explicit positions (``{vertex_id: xyz}``)."""
        if not isinstance(positions, dict) or not positions:
            raise ValidationError("positions must be a non-empty {vertex_id: xyz} dict")
        vids = self._require_vertices(list(positions))
        target = np.array([vec3(positions[v], f"position of vertex {v}") for v in vids])
        if space == "world":
            target = self._world_to_local(target)
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        self._set_positions(vids, target)
        self._check_degenerate(vids, False)
        return {"vertices": vids}

    @operation
    def move_along_normals(self, elements: Elements, distance: float):
        """Move each vertex by ``distance`` along its vertex normal (area-weighted face normals).

        Positive distances move outward (in front of the faces).
        """
        vids = self._vertices_of(elements)
        d = float(distance)
        offsets = np.zeros((len(vids), 3))
        for i, v in enumerate(vids):
            total = np.zeros(3)
            for f in self._faces_of_vertex(v):
                total += poly.newell_normal(self._pos[self._faces[f]])
            n = poly.unit(total)
            if n is None:
                raise EditError(
                    "UNDEFINED_NORMAL",
                    f"vertex {v} has no faces with a defined normal",
                    elements={"vertices": [v]},
                )
            offsets[i] = d * n
        self._set_positions(vids, self._pos[vids] + offsets)
        self._check_degenerate(vids, False)
        return {"vertices": vids}

    # -- helpers ----------------------------------------------------------------

    def _about(self, vids: list[int], m: np.ndarray, pivot: Any, space: str) -> np.ndarray:
        if isinstance(pivot, str):
            if pivot == "origin":
                center = np.zeros(3)
            elif pivot == "center":
                pts = self._pos[vids] if space == "local" else self.world_positions(vids)
                center = pts.mean(axis=0)
            else:
                raise ValidationError(f"pivot must be 'center', 'origin' or a point, got {pivot!r}")
        else:
            center = np.asarray(vec3(pivot, "pivot"))
        to, back = np.eye(4), np.eye(4)
        to[:3, 3], back[:3, 3] = -center, center
        return back @ m @ to

    def _world_to_local(self, points: np.ndarray) -> np.ndarray:
        try:
            inv = np.linalg.inv(self.world_matrix)
        except np.linalg.LinAlgError:
            raise EditError(
                "SINGULAR_WORLD_TRANSFORM",
                "the object's world transform is singular; world-space edits are undefined",
                hint="use space='local'",
            ) from None
        return points @ inv[:3, :3].T + inv[:3, 3]

    def _transform_vertices(
        self, vids: list[int], m: np.ndarray, space: str, allow_degenerate: bool
    ) -> dict[str, Any]:
        if not vids:
            raise EditError("EMPTY_SELECTION", "no elements to transform")
        if space == "world":
            world = self.world_matrix
            try:
                m = np.linalg.inv(world) @ m @ world
            except np.linalg.LinAlgError:
                raise EditError(
                    "SINGULAR_WORLD_TRANSFORM",
                    "the object's world transform is singular; world-space edits are undefined",
                    hint="use space='local'",
                ) from None
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        linear = m[:3, :3]
        moved = set(vids)
        new_positions = self._pos[vids] @ linear.T + m[:3, 3]
        if not np.all(np.isfinite(new_positions)):
            raise EditError("NON_FINITE_RESULT", "the transform produces non-finite positions")
        touched: set[int] = set()
        for v in vids:
            touched |= self._faces_of_vertex(v)
        inside = sorted(f for f in touched if all(u in moved for u in self._faces[f]))
        partial = sorted(touched - set(inside))
        det = float(np.linalg.det(linear))
        self._set_positions(vids, new_positions, dirty=False)
        self._mark_dirty(partial)
        flipped: list[int] = []
        if abs(det) <= 1e-300:
            self._mark_dirty(inside)
        else:
            # A mirror reverses orientation. Re-wind faces only where an entire
            # connected piece of the mesh was mirrored; a mirrored part that is still
            # attached to unmoved faces keeps its winding (its inside has not changed
            # sides), so orientation stays consistent across the mesh.
            rewind = self._whole_components(inside) if det < 0 else set()
            normal_matrix = np.linalg.inv(linear).T
            for f in inside:
                # Winding-derived normals transform as det(A) A^-T n; re-wound faces drop
                # the sign together with the winding.
                sign = 1.0 if (det > 0 or f in rewind) else -1.0
                if NORMAL in self._layers:
                    normals = []
                    for n in self._layers[NORMAL].data[f]:
                        t = poly.unit(sign * (normal_matrix @ np.asarray(n)))
                        normals.append(tuple(t.tolist()) if t is not None else n)
                    self._set_corners(f, NORMAL, normals)
                if f in rewind:
                    loop = self._faces[f]
                    corners = self._corners_of(f)
                    order = [0] + list(range(len(loop) - 1, 0, -1))
                    self._replace_face(
                        f,
                        [loop[i] for i in order],
                        self._split_corners([corners[i] for i in order]),
                        dirty=False,  # geometry and normals are exact
                    )
                    flipped.append(f)
        if flipped:
            self._diagnose(
                Diagnostic(
                    "WINDING_REVERSED",
                    INFO,
                    "the transform mirrors whole connected pieces of the mesh; their faces "
                    "were re-wound to stay outward-facing",
                    None,
                    len(flipped),
                    tuple(flipped[:10]),
                )
            )
        self._check_degenerate(vids, allow_degenerate)
        return {"vertices": vids, "determinant": det, "reversed_faces": flipped}

    def _whole_components(self, faces: list[int]) -> set[int]:
        """Faces of ``faces`` whose entire edge-connected component is in ``faces``."""
        pool = set(faces)
        whole: set[int] = set()
        seen: set[int] = set()
        for start in faces:
            if start in seen:
                continue
            component, stack, complete = {start}, [start], True
            seen.add(start)
            while stack:
                f = stack.pop()
                for key in self._loop_edges(self._faces[f]):
                    for g in self._edge_faces[self._edge_by_pair[key]]:
                        if g not in component:
                            component.add(g)
                            seen.add(g)
                            stack.append(g)
                            if g not in pool:
                                complete = False
            if complete:
                whole |= component
        return whole

    def _check_degenerate(self, vids: list[int], allow: bool) -> None:
        if allow:
            return
        eps = self._area_eps()
        faces: set[int] = set()
        for v in vids:
            faces |= self._faces_of_vertex(v)
        bad = sorted(f for f in faces if poly.polygon_area(self._pos[self._faces[f]]) <= eps)
        if bad:
            raise EditError(
                "DEGENERATE_RESULT",
                f"{len(bad)} faces would have zero area",
                elements={"faces": bad},
                hint="use a different transform, or pass allow_degenerate=True",
            )
