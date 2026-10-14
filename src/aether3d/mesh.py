"""Indexed triangle mesh: Aether3D's internal geometry representation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ValidationError

POSITION_DTYPE = np.float32
INDEX_DTYPE = np.uint32
GROUP_DTYPE = np.uint16
MAX_GROUP = 2**16 - 1
"""Largest face group index."""


def _attribute(value: Any, components: int, name: str, vertex_count: int | None) -> np.ndarray:
    try:
        arr = np.array(value, dtype=POSITION_DTYPE)
    except (TypeError, ValueError):
        raise ValidationError(f"mesh {name} must be numeric") from None
    if arr.ndim == 1 and arr.size % components == 0:
        arr = arr.reshape(-1, components)
    if arr.ndim != 2 or arr.shape[1] != components:
        raise ValidationError(f"mesh {name} must have shape (N, {components}), got {arr.shape}")
    if vertex_count is not None and arr.shape[0] != vertex_count:
        raise ValidationError(
            f"mesh {name} has {arr.shape[0]} entries but the mesh has {vertex_count} vertices"
        )
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"mesh {name} must contain only finite numbers")
    arr.flags.writeable = False
    return arr


class Mesh:
    """An immutable indexed triangle mesh.

    Data is stored in contiguous NumPy arrays, which are the layout GPUs and
    native (Rust/C++) code consume directly:

    * ``positions`` -- ``(N, 3) float32`` vertex positions (required)
    * ``triangles`` -- ``(M, 3) uint32`` vertex indices, counter-clockwise
      winding when seen from the front (outside) face
    * ``normals``   -- ``(N, 3) float32`` unit normals, or ``None``
    * ``uvs``       -- ``(N, 2) float32`` texture coordinates, or ``None``
    * ``groups``    -- ``(M,) uint16`` face group of every triangle, or
      ``None`` (all triangles in group 0). Objects can give each group its
      own material (``obj.group_materials``); glTF primitives import as groups.

    Arrays are copied on construction and marked read-only, so a mesh can be
    safely shared by several scene objects. To change geometry, build a new
    ``Mesh``. Flat ``(N*3,)`` inputs are accepted and reshaped.
    """

    __slots__ = ("_positions", "_triangles", "_normals", "_uvs", "_groups")

    def __init__(
        self,
        positions: Any,
        triangles: Any,
        normals: Any | None = None,
        uvs: Any | None = None,
        groups: Any | None = None,
    ) -> None:
        self._positions = _attribute(positions, 3, "positions", None)
        n = self._positions.shape[0]
        self._normals = None if normals is None else _attribute(normals, 3, "normals", n)
        self._uvs = None if uvs is None else _attribute(uvs, 2, "uvs", n)
        self._triangles = self._validate_triangles(triangles, n)
        self._groups = None if groups is None else self._validate_groups(groups)

    def _validate_groups(self, value: Any) -> np.ndarray:
        raw = np.asarray(value)
        count = self._triangles.shape[0]
        if raw.size == 0 and count == 0:
            raw = raw.reshape(0).astype(np.int64)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValidationError("mesh groups must contain integer group indices")
        raw = raw.reshape(-1)
        if raw.shape[0] != count:
            raise ValidationError(
                f"mesh groups has {raw.shape[0]} entries but the mesh has {count} triangles"
            )
        if raw.size and (raw.min() < 0 or raw.max() > MAX_GROUP):
            raise ValidationError(
                f"mesh groups must be in [0, {MAX_GROUP}], got range [{raw.min()}, {raw.max()}]"
            )
        arr = raw.astype(GROUP_DTYPE)
        arr.flags.writeable = False
        return arr

    @staticmethod
    def _validate_triangles(value: Any, vertex_count: int) -> np.ndarray:
        raw = np.asarray(value)
        if raw.size == 0:
            raw = raw.reshape(0, 3).astype(np.int64)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValidationError("mesh triangles must contain integer vertex indices")
        if raw.ndim == 1 and raw.size % 3 == 0:
            raw = raw.reshape(-1, 3)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValidationError(f"mesh triangles must have shape (M, 3), got {raw.shape}")
        if raw.size and (raw.min() < 0 or raw.max() >= vertex_count):
            raise ValidationError(
                f"mesh triangle indices must be in [0, {vertex_count - 1}], "
                f"got range [{raw.min()}, {raw.max()}]"
            )
        arr = raw.astype(INDEX_DTYPE)
        arr.flags.writeable = False
        return arr

    # -- data --------------------------------------------------------------

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def triangles(self) -> np.ndarray:
        return self._triangles

    @property
    def normals(self) -> np.ndarray | None:
        return self._normals

    @property
    def uvs(self) -> np.ndarray | None:
        return self._uvs

    @property
    def groups(self) -> np.ndarray | None:
        """Face group of every triangle, or ``None`` (one group, 0)."""
        return self._groups

    @property
    def group_ids(self) -> tuple[int, ...]:
        """The face groups used by the mesh, ascending (``(0,)`` without groups)."""
        if self._groups is None:
            return (0,) if self.triangle_count else ()
        return tuple(int(g) for g in np.unique(self._groups))

    def with_groups(self, groups: Any | None) -> Mesh:
        """A copy of this mesh (sharing its arrays) with other face groups."""
        mesh = Mesh.__new__(Mesh)
        mesh._positions, mesh._triangles = self._positions, self._triangles
        mesh._normals, mesh._uvs = self._normals, self._uvs
        mesh._groups = None if groups is None else mesh._validate_groups(groups)
        return mesh

    @property
    def vertex_count(self) -> int:
        return int(self._positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self._triangles.shape[0])

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Axis-aligned bounding box ``(min_xyz, max_xyz)`` in mesh-local space."""
        if self.vertex_count == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        lo = self._positions.min(axis=0).astype(np.float64).tolist()
        hi = self._positions.max(axis=0).astype(np.float64).tolist()
        return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])

    # -- value semantics ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mesh):
            return NotImplemented

        def same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
            if a is None or b is None:
                return a is b
            return a.shape == b.shape and bool(np.array_equal(a, b))

        return (
            same(self._positions, other._positions)
            and same(self._triangles, other._triangles)
            and same(self._normals, other._normals)
            and same(self._uvs, other._uvs)
            and same(self._groups, other._groups)
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        extras = [name for name in ("normals", "uvs", "groups") if getattr(self, name) is not None]
        return (
            f"Mesh(vertices={self.vertex_count}, triangles={self.triangle_count}"
            + (f", attributes={'+'.join(extras)}" if extras else "")
            + ")"
        )
