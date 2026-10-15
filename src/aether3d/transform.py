"""Local transform (position, rotation, scale) of a scene object."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

import numpy as np

from . import math3d
from .history import TRANSFORM
from .math3d import IDENTITY_QUAT, Quat, Vec3

_UNRECORDED = contextlib.nullcontext()


class Transform:
    """A translation / rotation / scale triple, relative to a parent space.

    Components are stored as immutable tuples of Python floats, so reading
    ``transform.position`` never hands out a mutable view that could be
    changed by accident. Assign a new value to change a component::

        t = Transform()
        t.position = (1, 2, 3)
        t.rotation_euler = (0, 0, 90)   # degrees, XYZ order
        t.scale = 2                     # uniform scale shorthand

    See :mod:`aether3d.math3d` for the coordinate and rotation conventions.

    An object's own transform (``obj.transform``) records every change in its
    scene's history; standalone transforms record nothing.
    """

    __slots__ = ("_position", "_rotation", "_scale", "_owner")

    def __init__(
        self,
        position: Iterable[float] = (0.0, 0.0, 0.0),
        rotation: Iterable[float] = IDENTITY_QUAT,
        scale: float | Iterable[float] = (1.0, 1.0, 1.0),
    ) -> None:
        self._position: Vec3 = math3d.vec3(position, "position")
        self._rotation: Quat = math3d.quat(rotation, "rotation")
        self._scale: Vec3 = math3d.scale3(scale, "scale")
        self._owner = None  # the SceneObject this is the transform of

    # -- components --------------------------------------------------------

    @property
    def position(self) -> Vec3:
        """Translation ``(x, y, z)``."""
        return self._position

    @position.setter
    def position(self, value: Iterable[float]) -> None:
        position = math3d.vec3(value, "position")
        with self._edit("position"):
            self._position = position

    @property
    def rotation(self) -> Quat:
        """Rotation as a unit quaternion ``(x, y, z, w)``; normalized on assignment."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Iterable[float]) -> None:
        rotation = math3d.quat(value, "rotation")
        with self._edit("rotation"):
            self._rotation = rotation

    @property
    def rotation_euler(self) -> Vec3:
        """Rotation as XYZ Euler angles in degrees (a view of :attr:`rotation`)."""
        return math3d.quat_to_euler(self._rotation)

    @rotation_euler.setter
    def rotation_euler(self, value: Iterable[float]) -> None:
        rotation = math3d.quat_from_euler(value)
        with self._edit("rotation"):
            self._rotation = rotation

    @property
    def scale(self) -> Vec3:
        """Scale ``(x, y, z)``. Assigning a single number sets a uniform scale."""
        return self._scale

    @scale.setter
    def scale(self, value: float | Iterable[float]) -> None:
        scale = math3d.scale3(value, "scale")
        with self._edit("scale"):
            self._scale = scale

    def _edit(self, what: str):
        owner = self._owner
        if owner is None:
            return _UNRECORDED
        return owner._edit(TRANSFORM, f"Set {what} of {owner._name!r}")

    # -- matrices ----------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """The 4x4 matrix ``T @ R @ S`` (a fresh array on every access)."""
        return math3d.compose_matrix(self._position, self._rotation, self._scale)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Transform:
        """Decompose an affine matrix into a transform (see ``decompose_matrix``)."""
        position, rotation, scale = math3d.decompose_matrix(matrix)
        return cls(position, rotation, scale)

    # -- value semantics ---------------------------------------------------

    def copy(self) -> Transform:
        return Transform(self._position, self._rotation, self._scale)

    def is_close(self, other: Transform, tolerance: float = 1e-6) -> bool:
        """Approximate equality; ``q`` and ``-q`` count as the same rotation."""
        q_dot = abs(float(np.dot(self._rotation, other._rotation)))
        return (
            np.allclose(self._position, other._position, atol=tolerance)
            and np.allclose(self._scale, other._scale, atol=tolerance)
            and abs(1.0 - q_dot) <= tolerance
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return (
            self._position == other._position
            and self._rotation == other._rotation
            and self._scale == other._scale
        )

    __hash__ = None  # type: ignore[assignment]  # mutable

    def __repr__(self) -> str:
        return (
            f"Transform(position={self._position}, rotation={self._rotation}, scale={self._scale})"
        )
