"""Small, dependency-light 3D math helpers built on NumPy.

Conventions (used consistently across Aether3D):

* **Coordinate system:** right-handed, +Z up, +X right; the front of an
  asset faces -Y (as in Blender, and matching glTF's +Z front via
  :mod:`aether3d.gltf.axes`).
  One unit is one meter by convention.
* **Vectors** are column vectors; a point ``p`` is transformed as ``M @ p``.
* **Matrices** are 4x4 ``float64`` NumPy arrays. A transform is composed as
  ``T @ R @ S`` (scale first, then rotate, then translate).
* **Quaternions** are unit quaternions stored as ``(x, y, z, w)`` (the glTF
  order). They are canonicalized so that ``w >= 0``.
* **Euler angles** are in degrees, applied in ``XYZ`` order about the fixed
  (world) axes: first X, then Y, then Z, i.e. ``R = Rz @ Ry @ Rx``.

The public functions here operate on plain tuples / NumPy arrays and hold no
state, which keeps them easy to replace with a native (Rust/C++) backend later.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from .errors import ValidationError

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _finite_array(value: Any, size: int, what: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{what} must be a sequence of {size} numbers, got {value!r}")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{what} must be a sequence of {size} numbers, got {value!r}"
        ) from None
    if arr.shape != (size,):
        raise ValidationError(f"{what} must have exactly {size} components, got {value!r}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{what} must contain only finite numbers, got {value!r}")
    return arr


def vec3(value: Iterable[float], what: str = "vector") -> Vec3:
    """Validate ``value`` as a finite 3-component vector and return it as a tuple."""
    x, y, z = _finite_array(value, 3, what).tolist()
    return (x, y, z)


def scale3(value: float | Iterable[float], what: str = "scale") -> Vec3:
    """Like :func:`vec3`, but a single number is expanded to a uniform scale."""
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        value = (value, value, value)
    return vec3(value, what)  # type: ignore[arg-type]


def quat(value: Iterable[float], what: str = "rotation") -> Quat:
    """Validate ``value`` as an ``(x, y, z, w)`` quaternion; return it normalized."""
    arr = _finite_array(value, 4, what)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise ValidationError(f"{what} quaternion must be non-zero, got {value!r}")
    # Leave already-unit quaternions untouched so normalization is idempotent and
    # independent of component order (dividing by a norm of 1 +/- 1 ulp would
    # perturb the last bits).
    if abs(norm - 1.0) > 1e-15:
        arr = arr / norm
    if arr[3] < 0.0:
        arr = -arr
    x, y, z, w = (float(c) + 0.0 for c in arr)  # "+ 0.0" folds -0.0 into 0.0
    return (x, y, z, w)


def positive_number(value: Any, what: str) -> float:
    """Validate ``value`` as a finite number strictly greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValidationError(f"{what} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{what} must be a finite number greater than 0, got {value!r}")
    return result


def int_at_least(value: Any, minimum: int, what: str) -> int:
    """Validate ``value`` as an integer ``>= minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{what} must be an integer, got {value!r}")
    if value < minimum:
        raise ValidationError(f"{what} must be at least {minimum}, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# Quaternions
# --------------------------------------------------------------------------- #


def quat_multiply(a: Iterable[float], b: Iterable[float]) -> Quat:
    """Hamilton product ``a * b`` (applying ``b`` first, then ``a``)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return quat(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quat_from_axis_angle(axis: Iterable[float], degrees: float) -> Quat:
    """Quaternion rotating ``degrees`` counter-clockwise about ``axis``."""
    a = np.asarray(vec3(axis, "axis"))
    length = float(np.linalg.norm(a))
    if length < _EPS:
        raise ValidationError("axis must be non-zero")
    half = math.radians(degrees) / 2.0
    x, y, z = (a / length * math.sin(half)).tolist()
    return quat((x, y, z, math.cos(half)))


def quat_from_euler(euler_degrees: Iterable[float]) -> Quat:
    """Quaternion from XYZ Euler angles in degrees (see module conventions)."""
    rx, ry, rz = vec3(euler_degrees, "euler angles")
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), rx)
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), ry)
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), rz)
    return quat_multiply(qz, quat_multiply(qy, qx))


def quat_to_matrix3(q: Iterable[float]) -> np.ndarray:
    """3x3 rotation matrix for a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix3_to_quat(m: np.ndarray) -> Quat:
    """Unit quaternion for a 3x3 rotation matrix (Shepperd's method)."""
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    return quat(q)


def quat_to_euler(q: Iterable[float]) -> Vec3:
    """XYZ Euler angles in degrees for a quaternion (see module conventions).

    Near gimbal lock (Y = +/-90 degrees) the Z angle is reported as 0 and the
    full remaining rotation is attributed to X.
    """
    m = quat_to_matrix3(q)
    sin_y = -m[2, 0]
    y = math.asin(max(-1.0, min(1.0, sin_y)))
    if abs(sin_y) < 1.0 - 1e-9:
        x = math.atan2(m[2, 1], m[2, 2])
        z = math.atan2(m[1, 0], m[0, 0])
    else:
        x = math.atan2(-m[1, 2], m[1, 1])
        z = 0.0
    return (math.degrees(x) + 0.0, math.degrees(y) + 0.0, math.degrees(z) + 0.0)


# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #


def compose_matrix(
    position: Iterable[float] = (0.0, 0.0, 0.0),
    rotation: Iterable[float] = IDENTITY_QUAT,
    scale: Iterable[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Build the 4x4 matrix ``T @ R @ S``."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quat_to_matrix3(rotation) * np.asarray(scale3(scale), dtype=np.float64)
    m[:3, 3] = vec3(position, "position")
    return m


def decompose_matrix(matrix: np.ndarray) -> tuple[Vec3, Quat, Vec3]:
    """Split an affine 4x4 matrix into ``(position, rotation, scale)``.

    The result is exact for any matrix produced by :func:`compose_matrix`
    (and products of such matrices with uniform scale). Matrices containing
    shear -- which arise from non-uniform scale combined with rotation in a
    hierarchy -- cannot be represented as TRS; the closest rotation is
    returned in that case.

    Raises :class:`ValidationError` if the matrix is singular (e.g. zero scale),
    because rotation is then undefined.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValidationError("matrix must be a finite 4x4 matrix")
    basis = m[:3, :3]
    det = float(np.linalg.det(basis))
    if abs(det) < _EPS:
        raise ValidationError("matrix is singular (zero scale); it cannot be decomposed")
    scale = np.linalg.norm(basis, axis=0)
    if det < 0.0:
        scale[0] = -scale[0]
    rot = basis / scale
    # Snap to the nearest proper rotation to absorb numerical noise and shear.
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] = -u[:, -1]
        rot = u @ vt
    sx, sy, sz = scale.tolist()
    return vec3(m[:3, 3], "position"), matrix3_to_quat(rot), (sx, sy, sz)


def transform_point(matrix: np.ndarray, point: Iterable[float]) -> Vec3:
    """Apply an affine 4x4 matrix to a 3D point."""
    p = np.append(np.asarray(vec3(point, "point")), 1.0)
    x, y, z, _ = (np.asarray(matrix, dtype=np.float64) @ p).tolist()
    return (x, y, z)
