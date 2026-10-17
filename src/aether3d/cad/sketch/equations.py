"""Equation kernels of the sketch solver: residuals and exact Jacobians, vectorized.

Every constraint compiles to equations of a few *kernels*. A kernel evaluates ``m``
equations at once from a matrix ``V`` (m, k) of the variables they read (gathered by an
index matrix) and constant columns ``C`` (m, c); it returns the residuals (m,) and the
partial derivatives (m, k). All residuals are lengths in metres -- angle equations are
multiplied by the sketch's characteristic length -- so one tolerance and one scaling fit
every row.

Derivatives are analytic (checked against finite differences in the tests). Where a
gradient is undefined (a zero-length line, two coincident points of a distance), it is
taken as zero and the solver's damping and rank analysis handle the degenerate row.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-300


def _safe(n: np.ndarray) -> np.ndarray:
    return np.where(n > _TINY, n, 1.0)


def linear(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``sum(w_j v_j) - c0`` with weights in ``c[:, 1:]``."""
    w = c[:, 1:]
    return np.sum(w * v, axis=1) - c[:, 0], w.copy()


def dist(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``|p - q| - c0``; V = [px, py, qx, qy]."""
    dx, dy = v[:, 0] - v[:, 2], v[:, 1] - v[:, 3]
    n = np.hypot(dx, dy)
    s = _safe(n)
    gx, gy = np.where(n > _TINY, dx / s, 0.0), np.where(n > _TINY, dy / s, 0.0)
    jac = np.stack([gx, gy, -gx, -gy], axis=1)
    return n - c[:, 0], jac


def dist_var(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``|p - q| - (w1 v1 + w2 v2) - c0``; V = [px, py, qx, qy, v1, v2], C = [c0, w1, w2]."""
    r, jd = dist(v[:, :4], c[:, :1])
    r = r - c[:, 1] * v[:, 4] - c[:, 2] * v[:, 5]
    jac = np.concatenate([jd, -c[:, 1:2], -c[:, 2:3]], axis=1)
    return r, jac


def dist_diff(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``|p - q| - |r - s| - c0``; V = [p, q, r, s] (8 columns)."""
    r1, j1 = dist(v[:, :4], np.zeros((len(v), 1)))
    r2, j2 = dist(v[:, 4:], np.zeros((len(v), 1)))
    return r1 - r2 - c[:, 0], np.concatenate([j1, -j2], axis=1)


def point_line(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of ``p`` from the line ``a -> b`` minus ``c0 + w v``.

    V = [ax, ay, bx, by, px, py, var], C = [c0, w]; positive on the left of ``a -> b``.
    """
    ax, ay, bx, by, px, py = (v[:, i] for i in range(6))
    dx, dy = bx - ax, by - ay
    ex, ey = px - ax, py - ay
    cr = dx * ey - dy * ex
    n = np.hypot(dx, dy)
    s = _safe(n)
    r = cr / s - c[:, 0] - c[:, 1] * v[:, 6]
    # d cr: wrt p = (-dy, dx); wrt d (i.e. b) = (ey, -ex); wrt a = -(d b) - (d p).
    dcr_b = np.stack([ey, -ex], axis=1)
    dcr_p = np.stack([-dy, dx], axis=1)
    dn_b = np.stack([dx, dy], axis=1) / s[:, None]
    db = dcr_b / s[:, None] - (cr / s**2)[:, None] * dn_b
    dp = dcr_p / s[:, None]
    da = -db - dp
    jac = np.concatenate([da, db, dp, -c[:, 1:2]], axis=1)
    jac[n <= _TINY] = 0.0
    return r, jac


def point_line_mid(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of the midpoint of ``p q`` from the line ``a -> b`` minus ``c0``.

    V = [ax, ay, bx, by, px, py, qx, qy]."""
    m = 0.5 * (v[:, 4:6] + v[:, 6:8])
    vv = np.concatenate([v[:, :4], m, np.zeros((len(v), 1))], axis=1)
    r, j = point_line(vv, np.concatenate([c[:, :1], np.zeros((len(v), 1))], axis=1))
    jac = np.concatenate([j[:, :4], 0.5 * j[:, 4:6], 0.5 * j[:, 4:6]], axis=1)
    return r, jac


def perp_dot(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``dot(b - a, q - p) / |b - a| - c0``; V = [a, b, p, q] (8 columns)."""
    dx, dy = v[:, 2] - v[:, 0], v[:, 3] - v[:, 1]
    ex, ey = v[:, 6] - v[:, 4], v[:, 7] - v[:, 5]
    dot = dx * ex + dy * ey
    n = np.hypot(dx, dy)
    s = _safe(n)
    r = dot / s - c[:, 0]
    db = np.stack([ex, ey], axis=1) / s[:, None] - (dot / s**3)[:, None] * np.stack(
        [dx, dy], axis=1
    )
    dq = np.stack([dx, dy], axis=1) / s[:, None]
    jac = np.concatenate([-db, db, -dq, dq], axis=1)
    jac[n <= _TINY] = 0.0
    return r, jac


def lines_angle(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``L * sin(phi - theta)`` with ``phi`` the signed angle from line 1 to line 2.

    V = [a1, b1, a2, b2] (8 columns), C = [L, cos theta, sin theta]. theta = 0: parallel;
    pi/2: perpendicular.
    """
    d1x, d1y = v[:, 2] - v[:, 0], v[:, 3] - v[:, 1]
    d2x, d2y = v[:, 6] - v[:, 4], v[:, 7] - v[:, 5]
    cr = d1x * d2y - d1y * d2x
    dt = d1x * d2x + d1y * d2y
    n1, n2 = np.hypot(d1x, d1y), np.hypot(d2x, d2y)
    s1, s2 = _safe(n1), _safe(n2)
    big_l, ct, st = c[:, 0], c[:, 1], c[:, 2]
    f = (cr * ct - dt * st) / (s1 * s2)
    k = big_l / (s1 * s2)
    g1x = k * (d2y * ct - d2x * st) - big_l * f * d1x / s1**2
    g1y = k * (-d2x * ct - d2y * st) - big_l * f * d1y / s1**2
    g2x = k * (-d1y * ct - d1x * st) - big_l * f * d2x / s2**2
    g2y = k * (d1x * ct - d1y * st) - big_l * f * d2y / s2**2
    jac = np.stack([-g1x, -g1y, g1x, g1y, -g2x, -g2y, g2x, g2y], axis=1)
    jac[(n1 <= _TINY) | (n2 <= _TINY)] = 0.0
    return big_l * f, jac


def line_angle(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``L * sin(alpha_d - alpha)`` for the absolute direction of line ``a -> b``.

    V = [ax, ay, bx, by], C = [L, cos alpha, sin alpha]."""
    dx, dy = v[:, 2] - v[:, 0], v[:, 3] - v[:, 1]
    n = np.hypot(dx, dy)
    s = _safe(n)
    big_l, ca, sa = c[:, 0], c[:, 1], c[:, 2]
    num = dy * ca - dx * sa
    f = num / s
    gx = big_l * (-sa / s - num * dx / s**3)
    gy = big_l * (ca / s - num * dy / s**3)
    jac = np.stack([-gx, -gy, gx, gy], axis=1)
    jac[n <= _TINY] = 0.0
    return big_l * f, jac


KERNELS = {
    "linear": linear,
    "dist": dist,
    "dist_var": dist_var,
    "dist_diff": dist_diff,
    "point_line": point_line,
    "point_line_mid": point_line_mid,
    "perp_dot": perp_dot,
    "lines_angle": lines_angle,
    "line_angle": line_angle,
}
