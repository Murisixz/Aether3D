"""Quaternion arrays for animation: interpolation, blending, Euler conversion, unwrapping.

Conventions are Aether3D's (``math3d``): unit quaternions ``(x, y, z, w)``, Hamilton
products (``a * b`` applies ``b`` first), Euler angles XYZ in degrees about the fixed axes
(``R = Rz Ry Rx``). Unlike authored transforms, animation keys keep their **signs**: ``q``
and ``-q`` are the same orientation but not the same path. Interpolation from ``a`` to
``b`` takes the short way (through ``-b`` when ``a . b < 0``) unless a key asks for its
quaternions to be used as stored, which allows a deliberate long way round (up to 360
degrees per segment; multi-turn spins use several keys or Euler channels).

* ``slerp(a, b, u)`` -- constant angular speed along the great arc (Shoemake 1985); near
  identical orientations (angle below 1e-6 rad) it falls back to normalized linear
  interpolation, whose error there is below 1e-18.
* ``log`` / ``exp`` -- ``q = exp(v)`` with ``v = (angle / 2) axis``.
* ``squad_controls`` -- Shoemake's intermediate quaternions for C1 spherical cubics.
* ``blend`` / ``additive`` -- layer composition: ``slerp(base, x, w)`` and
  ``base * (ref^-1 x)^w``.
"""

from __future__ import annotations

import math

import numpy as np

SMALL_ANGLE = 1e-6


def as_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q.reshape(-1, 4) if q.ndim == 1 else q


def normalize(q: np.ndarray) -> np.ndarray:
    q = as_quats(q)
    n = np.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    return q / np.where(n > 0.0, n, 1.0)[:, None]


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products ``a * b`` (row by row, broadcasting)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(a, np.float64) * np.asarray(b, np.float64), axis=-1)


def align(reference: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``q`` or ``-q``, whichever is in the hemisphere of ``reference``."""
    s = np.where(dot(reference, q) < 0.0, -1.0, 1.0)
    return q * s[..., None]


def log(q: np.ndarray) -> np.ndarray:
    """``v`` with ``exp(v) = q``: ``(angle / 2) * axis`` (angle in ``[0, 2 pi]`` for unit
    ``q`` as stored -- ``-q`` gives the long way round)."""
    q = np.asarray(q, np.float64)
    v = q[..., :3]
    s = np.sqrt(np.sum(v * v, axis=-1))
    half = np.arctan2(s, q[..., 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(s > 1e-300, half / np.where(s > 1e-300, s, 1.0), 1.0)
    return v * k[..., None]


def exp(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    half = np.sqrt(np.sum(v * v, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(half > 1e-300, np.sin(half) / np.where(half > 1e-300, half, 1.0), 1.0)
    return np.concatenate([v * k[..., None], np.cos(half)[..., None]], axis=-1)


def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation angle in radians between orientations ``a`` and ``b`` (0..pi), from
    ``atan2`` of the relative rotation (accurate near 0, unlike ``acos`` of a dot)."""
    rel = multiply(conjugate(a), b)
    s = np.sqrt(np.sum(rel[..., :3] ** 2, axis=-1))
    return 2.0 * np.arctan2(s, np.abs(rel[..., 3]))


def slerp(
    a: np.ndarray, b: np.ndarray, u: np.ndarray, shortest: np.ndarray | bool = True
) -> np.ndarray:
    """Spherical linear interpolation of unit quaternions (rows), ``u`` per row (values
    outside ``[0, 1]`` extrapolate along the same great arc)."""
    a = as_quats(a)
    b = as_quats(b)
    u = np.asarray(u, np.float64).reshape(-1)
    d = np.sum(a * b, axis=1)
    flip = (d < 0.0) & np.asarray(shortest, bool)
    b = np.where(flip[:, None], -b, b)
    d = np.where(flip, -d, d)
    d = np.clip(d, -1.0, 1.0)
    theta = np.arccos(d)
    sin_t = np.sin(theta)
    small = theta < SMALL_ANGLE
    with np.errstate(divide="ignore", invalid="ignore"):
        wa = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(small, 1.0, sin_t))
        wb = np.where(small, u, np.sin(u * theta) / np.where(small, 1.0, sin_t))
    out = a * wa[:, None] + b * wb[:, None]
    return np.where(small[:, None], normalize(out), out)


def slerp_velocity(a: np.ndarray, b: np.ndarray, shortest: np.ndarray | bool = True) -> np.ndarray:
    """Angular velocity per unit ``u`` of ``slerp(a, b, u)`` (constant along the arc):
    ``theta * axis``, in the frame the quaternions rotate into (parent space)."""
    a = as_quats(a)
    b = as_quats(b)
    d = np.sum(a * b, axis=1)
    flip = (d < 0.0) & np.asarray(shortest, bool)
    b = np.where(flip[:, None], -b, b)
    rel = multiply(b, conjugate(a))  # b a^-1: the rotation from a to b, applied after a
    return 2.0 * log(rel)


def squad_controls(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Shoemake's intermediate quaternion ``s_i = q_i exp(-(log(q_i^-1 q_{i+1}) +
    log(q_i^-1 q_{i-1})) / 4)`` (inputs already hemisphere-aligned)."""
    inv = conjugate(cur)
    la = log(multiply(inv, nxt))
    lb = log(multiply(inv, prev))
    return multiply(cur, exp(-(la + lb) / 4.0))


def squad(
    q0: np.ndarray, q1: np.ndarray, s0: np.ndarray, s1: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """``slerp(slerp(q0, q1, u), slerp(s0, s1, u), 2u(1 - u))`` (no hemisphere flips:
    the controls are aligned with their keys)."""
    p = slerp(q0, q1, u, shortest=False)
    s = slerp(s0, s1, u, shortest=False)
    return slerp(p, s, 2.0 * u * (1.0 - u), shortest=False)


def blend(base: np.ndarray, other: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``slerp(base, other, w)`` the short way (layer and crossfade blending)."""
    return slerp(base, other, weight, shortest=True)


def additive(
    base: np.ndarray, value: np.ndarray, reference: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """``base * (reference^-1 value)^w``: the clip's rotation relative to its reference
    pose, scaled by ``w`` and applied on top of ``base`` (local space)."""
    delta = multiply(conjugate(reference), align(reference, value))
    w = np.asarray(weight, np.float64).reshape(-1)
    scaled = exp(log(delta) * w[:, None])
    return multiply(base, scaled)


# --------------------------------------------------------------------------------------- #
# Euler angles (XYZ, degrees, extrinsic -- math3d's convention)
# --------------------------------------------------------------------------------------- #


def from_euler(degrees: np.ndarray) -> np.ndarray:
    """Quaternions of XYZ Euler angles in degrees, ``q = qz * qy * qx`` (rows)."""
    e = np.radians(np.asarray(degrees, np.float64).reshape(-1, 3)) * 0.5
    cx, sx = np.cos(e[:, 0]), np.sin(e[:, 0])
    cy, sy = np.cos(e[:, 1]), np.sin(e[:, 1])
    cz, sz = np.cos(e[:, 2]), np.sin(e[:, 2])
    # qz * qy = (0, 0, sz, cz) * (0, sy, 0, cy)
    zy = np.stack([-sz * sy, cz * sy, sz * cy, cz * cy], axis=1)
    qx = np.stack([sx, np.zeros_like(sx), np.zeros_like(sx), cx], axis=1)
    return multiply(zy, qx)


def to_euler(q: np.ndarray) -> np.ndarray:
    """XYZ Euler angles in degrees (``math3d.quat_to_euler``'s branch choice)."""
    q = normalize(q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m20 = 2 * (x * z - y * w)
    m21 = 2 * (y * z + x * w)
    m22 = 1 - 2 * (x * x + y * y)
    m10 = 2 * (x * y + z * w)
    m00 = 1 - 2 * (y * y + z * z)
    m12 = 2 * (y * z - x * w)
    m11 = 1 - 2 * (x * x + z * z)
    sin_y = np.clip(-m20, -1.0, 1.0)
    ry = np.arcsin(sin_y)
    regular = np.abs(sin_y) < 1.0 - 1e-9
    rx = np.where(regular, np.arctan2(m21, m22), np.arctan2(-m12, m11))
    rz = np.where(regular, np.arctan2(m10, m00), 0.0)
    return np.degrees(np.stack([rx, ry, rz], axis=1))


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    """A sequence of angles (degrees) with multiples of 360 added so that consecutive
    values differ by at most 180 (``numpy.unwrap`` in degrees)."""
    v = np.asarray(values, np.float64)
    if v.size < 2:
        return v.copy()
    return np.degrees(np.unwrap(np.radians(v)))


def euler_filter(angles: np.ndarray) -> np.ndarray:
    """Make a sequence of XYZ Euler triples continuous: for each, choose between the two
    equivalent triples ``(x, y, z)`` and ``(x + 180, 180 - y, z + 180)`` and multiples of
    360 per axis, minimizing the distance to the previous triple (the rotations are
    unchanged)."""
    e = np.asarray(angles, np.float64).reshape(-1, 3).copy()
    for i in range(1, e.shape[0]):
        prev = e[i - 1]
        best = None
        for cand in (e[i], np.array([e[i, 0] + 180.0, 180.0 - e[i, 1], e[i, 2] + 180.0])):
            c = cand + 360.0 * np.round((prev - cand) / 360.0)
            d = float(np.sum(np.abs(c - prev)))
            if best is None or d < best[0] - 1e-12:
                best = (d, c)
        e[i] = best[1]  # type: ignore[index]
    return e


def matrices(q: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` rotation matrices (``math3d.quat_to_matrix3``'s formula)."""
    q = as_quats(q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def from_matrices(m: np.ndarray) -> np.ndarray:
    """Unit quaternions (``w >= 0``) of ``(n, 3, 3)`` rotation matrices (Shepperd)."""
    m = np.asarray(m, np.float64).reshape(-1, 3, 3)
    n = m.shape[0]
    q = np.empty((n, 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    case0 = tr > 0.0
    case1 = ~case0 & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    case2 = ~case0 & ~case1 & (m[:, 1, 1] > m[:, 2, 2])
    case3 = ~case0 & ~case1 & ~case2
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2.0
        q[case0] = np.stack(
            [
                (m[:, 2, 1] - m[:, 1, 2]) / s,
                (m[:, 0, 2] - m[:, 2, 0]) / s,
                (m[:, 1, 0] - m[:, 0, 1]) / s,
                0.25 * s,
            ],
            axis=1,
        )[case0]
        s = np.sqrt(np.maximum(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 0.0)) * 2.0
        q[case1] = np.stack(
            [
                0.25 * s,
                (m[:, 0, 1] + m[:, 1, 0]) / s,
                (m[:, 0, 2] + m[:, 2, 0]) / s,
                (m[:, 2, 1] - m[:, 1, 2]) / s,
            ],
            axis=1,
        )[case1]
        s = np.sqrt(np.maximum(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], 0.0)) * 2.0
        q[case2] = np.stack(
            [
                (m[:, 0, 1] + m[:, 1, 0]) / s,
                0.25 * s,
                (m[:, 1, 2] + m[:, 2, 1]) / s,
                (m[:, 0, 2] - m[:, 2, 0]) / s,
            ],
            axis=1,
        )[case2]
        s = np.sqrt(np.maximum(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], 0.0)) * 2.0
        q[case3] = np.stack(
            [
                (m[:, 0, 2] + m[:, 2, 0]) / s,
                (m[:, 1, 2] + m[:, 2, 1]) / s,
                0.25 * s,
                (m[:, 1, 0] - m[:, 0, 1]) / s,
            ],
            axis=1,
        )[case3]
    q = normalize(q)
    return np.where((q[:, 3] < 0.0)[:, None], -q, q)


def axis_angle(axis: np.ndarray, radians: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, np.float64).reshape(-1, 3)
    n = np.linalg.norm(a, axis=1)
    a = a / np.where(n > 0, n, 1.0)[:, None]
    h = 0.5 * np.asarray(radians, np.float64).reshape(-1)
    return np.concatenate([a * np.sin(h)[:, None], np.cos(h)[:, None]], axis=1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``numpy.cross`` of 3-vectors (rows, broadcasting): the same products and
    differences, bit for bit, without its per-call overhead (small arrays, hot paths)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (rows) by quaternions ``q`` (rows or one)."""
    q = np.asarray(q, np.float64)
    v = np.asarray(v, np.float64)
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


def swing_twist(q: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose ``q = swing * twist`` with ``twist`` about ``axis`` (unit, rows)."""
    q = as_quats(q)
    a = np.asarray(axis, np.float64).reshape(-1, 3)
    p = np.sum(q[:, :3] * a, axis=1)[:, None] * a
    twist = np.concatenate([p, q[:, 3:4]], axis=1)
    n = np.linalg.norm(twist, axis=1)
    twist = np.where(
        (n > 1e-12)[:, None],
        twist / np.where(n > 1e-12, n, 1.0)[:, None],
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    swing = multiply(q, conjugate(twist))
    return swing, twist


TWO_PI = 2.0 * math.pi
