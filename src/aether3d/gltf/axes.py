"""Exact conversion between Aether3D (+Z up) and glTF (+Y up) coordinates.

Both systems are right-handed. Aether3D is +Z up; glTF is +Y up with the
front of an asset facing +Z. The conversion is the rotation of -90 degrees
about the X axis -- the same convention Blender uses -- so a scene moves between
Aether3D and Blender via glTF without changing its coordinates::

    glTF (x, y, z) = (x_a, z_a, -y_a)        Aether3D (x, y, z) = (x_g, -z_g, y_g)

``AXIS_MATRIX`` (``C``) maps Aether3D vectors to glTF. Because ``C`` is a proper
rotation (determinant +1):

* positions, directions and normals are multiplied by ``C`` (normals need no
  inverse-transpose, and triangle winding is unchanged);
* a local transform ``M`` becomes ``C M C^T``. For ``M = T R S`` this is again
  a TRS transform: ``t' = C t``, the quaternion's vector part is rotated by
  ``C`` (``w`` unchanged), and the scale axes are permuted
  ``(sx, sy, sz) -> (sx, sz, sy)``. Non-uniform scale therefore stays exact;
* hierarchies convert node by node: ``C (M1 M2) C^T = (C M1 C^T)(C M2 C^T)``,
  so world transforms are converted consistently.

``C`` only permutes components and flips a sign, so every conversion here is
exact in floating point.

Texture coordinates use a bottom-left origin in Aether3D (as in Blender and
OpenGL) and a top-left origin in glTF, so ``v`` is flipped: ``v' = 1 - v``.
"""

from __future__ import annotations

import numpy as np

from ..math3d import Quat, Vec3

AXIS_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
"""3x3 rotation taking Aether3D (+Z up) vectors to glTF (+Y up)."""


def vector_to_gltf(v: Vec3) -> Vec3:
    x, y, z = v
    return (x, z, -y + 0.0)  # "+ 0.0" avoids emitting -0.0


def vector_from_gltf(v: Vec3) -> Vec3:
    x, y, z = v
    return (x, -z + 0.0, y)


def rotation_to_gltf(q: Quat) -> Quat:
    x, y, z, w = q
    return (x, z, -y + 0.0, w)


def rotation_from_gltf(q: Quat) -> Quat:
    x, y, z, w = q
    return (x, -z + 0.0, y, w)


def scale_to_gltf(s: Vec3) -> Vec3:
    x, y, z = s
    return (x, z, y)


def scale_from_gltf(s: Vec3) -> Vec3:
    x, y, z = s
    return (x, z, y)


def vectors_to_gltf(array: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3)`` array of positions or normals (dtype preserved)."""
    out = np.empty_like(array)
    out[:, 0] = array[:, 0]
    out[:, 1] = array[:, 2]
    out[:, 2] = -array[:, 1]
    return out + array.dtype.type(0)


def vectors_from_gltf(array: np.ndarray) -> np.ndarray:
    """Inverse of :func:`vectors_to_gltf`."""
    out = np.empty_like(array)
    out[:, 0] = array[:, 0]
    out[:, 1] = -array[:, 2]
    out[:, 2] = array[:, 1]
    return out + array.dtype.type(0)


def flip_uvs(uvs: np.ndarray) -> np.ndarray:
    """Convert texture coordinates between bottom-left and top-left origins (float32 out)."""
    out = uvs.astype(np.float64)
    out[:, 1] = 1.0 - out[:, 1]
    return out.astype(np.float32)
