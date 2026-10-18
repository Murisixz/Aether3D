"""Inverse kinematics solvers on joint chains (positions in one space, e.g. the rig's).

* :func:`two_bone` -- the analytic solution for a three-joint chain (shoulder-elbow-wrist,
  hip-knee-ankle) by the law of cosines, bending in the plane of the root, the target and
  a **pole** point (where the elbow or knee points). Unreachable targets are approached
  as far as the chain reaches (fully extended toward them); ``stretch`` lets the bones
  lengthen up to a factor instead. ``soft`` eases the last fraction of the reach so the
  middle joint does not snap straight (``soft = 0``: exact, the default).
* :func:`fabrik` -- FABRIK (Aristidou and Lasenby, Graphical Models 73(5), 2011) for
  chains of any length: forward and backward reaching passes that keep bone lengths,
  until the end is within ``tolerance`` of the target or ``iterations`` are done.
* :func:`ccd` -- cyclic coordinate descent (Wang and Chen 1991): each joint, end to root,
  rotates the end toward the target.

Each returns the new joint **positions** and the remaining distance to the target (the
residual, reported to callers); :func:`chain_rotations` turns positions into rotation
deltas that align each bone with its new direction by the smallest rotation (which keeps
the joints' roll as stable as the motion allows). Everything is deterministic: fixed
iteration orders, no randomness.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros_like(v)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation (unit quaternion ``x, y, z, w``) taking direction ``a`` to
    direction ``b`` (identity for zero vectors; 180 degrees about a perpendicular axis for
    opposite ones)."""
    a = _unit(np.asarray(a, np.float64))
    b = _unit(np.asarray(b, np.float64))
    if not a.any() or not b.any():
        return np.array([0.0, 0.0, 0.0, 1.0])
    d = float(a @ b)
    if d >= 1.0 - 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d <= -1.0 + 1e-15:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = _unit(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    c = np.cross(a, b)
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def two_bone(
    root: np.ndarray,
    mid: np.ndarray,
    end: np.ndarray,
    target: np.ndarray,
    pole: np.ndarray | None = None,
    *,
    stretch: float = 1.0,
    soft: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(new mid, new end, residual)`` for a three-joint chain reaching ``target``.

    ``pole`` -- a point the middle joint bends toward (default: the current bend);
    ``stretch`` -- the largest factor bones may lengthen by to reach (1: rigid);
    ``soft`` -- fraction of the reach (0..0.5) over which full extension is eased.
    """
    root = np.asarray(root, np.float64)
    mid = np.asarray(mid, np.float64)
    end = np.asarray(end, np.float64)
    target = np.asarray(target, np.float64)
    a = float(np.linalg.norm(mid - root))
    b = float(np.linalg.norm(end - mid))
    if a < EPS or b < EPS:
        return mid.copy(), end.copy(), float(np.linalg.norm(end - target))
    to = target - root
    d = float(np.linalg.norm(to))
    direction = to / d if d > EPS else _unit(end - root)
    reach = a + b
    scale = 1.0
    if d > reach:
        scale = min(d / reach, max(stretch, 1.0))
    a_s, b_s = a * scale, b * scale
    reach_s = a_s + b_s
    lo = abs(a_s - b_s) + 1e-9 * reach_s
    dist = min(max(d, lo), reach_s)
    if soft > 0.0 and d < reach_s:
        # Ease the approach to full extension: distances beyond (1 - soft) of the reach
        # map smoothly (exponentially) toward it, never quite reaching it.
        start = (1.0 - soft) * reach_s
        if dist > start:
            span = reach_s - start
            dist = start + span * (1.0 - math.exp(-(dist - start) / span))
    # Bend plane from the pole (or the current bend), perpendicular to the direction.
    hint = (np.asarray(pole, np.float64) - root) if pole is not None else (mid - root)
    perp = hint - (hint @ direction) * direction
    if np.linalg.norm(perp) < 1e-9 * max(a, 1.0):
        # Pole on the line: keep the current bend plane, else any perpendicular.
        perp = (mid - root) - ((mid - root) @ direction) * direction
        if np.linalg.norm(perp) < 1e-9 * max(a, 1.0):
            perp = np.cross(direction, [0.0, 0.0, 1.0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(direction, [1.0, 0.0, 0.0])
    perp = _unit(perp)
    cos_a = (a_s * a_s + dist * dist - b_s * b_s) / (2.0 * a_s * dist)
    cos_a = min(1.0, max(-1.0, cos_a))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    new_mid = root + a_s * (cos_a * direction + sin_a * perp)
    new_end = root + dist * direction
    if d <= EPS:
        new_end = new_mid + b_s * _unit(new_mid - root)
    residual = float(np.linalg.norm(new_end - target))
    return new_mid, new_end, residual


def fabrik(
    positions: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = 1e-6,
    iterations: int = 32,
    pole: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """``(positions, residual, iterations used)`` of FABRIK on a chain (root fixed)."""
    p = np.array(positions, np.float64)
    target = np.asarray(target, np.float64)
    n = p.shape[0]
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    root = p[0].copy()
    total = float(lengths.sum())
    if float(np.linalg.norm(target - root)) >= total:
        direction = _unit(target - root)
        for i in range(1, n):
            p[i] = p[i - 1] + lengths[i - 1] * direction
        return p, float(np.linalg.norm(p[-1] - target)), 0
    if pole is not None:  # start bent toward the pole: a stable, chosen solution
        pole = np.asarray(pole, np.float64)
        for i in range(1, n - 1):
            p[i] = p[i] + 1e-3 * lengths[i - 1] * _unit(pole - p[i])
    used = 0
    for step in range(1, iterations + 1):
        used = step
        p[-1] = target
        for i in range(n - 2, -1, -1):  # backward: from the end
            p[i] = p[i + 1] + lengths[i] * _unit(p[i] - p[i + 1])
        p[0] = root
        for i in range(1, n):  # forward: from the root
            p[i] = p[i - 1] + lengths[i - 1] * _unit(p[i] - p[i - 1])
        if float(np.linalg.norm(p[-1] - target)) <= tolerance:
            break
    return p, float(np.linalg.norm(p[-1] - target)), used


def ccd(
    positions: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = 1e-6,
    iterations: int = 32,
) -> tuple[np.ndarray, float, int]:
    """``(positions, residual, iterations used)`` of cyclic coordinate descent."""
    from . import rotation as R

    p = np.array(positions, np.float64)
    target = np.asarray(target, np.float64)
    n = p.shape[0]
    used = 0
    for step in range(1, iterations + 1):
        used = step
        for i in range(n - 2, -1, -1):
            q = rotation_between(p[-1] - p[i], target - p[i])
            rel = p[i + 1 :] - p[i]
            p[i + 1 :] = p[i] + R.rotate(np.broadcast_to(q, (rel.shape[0], 4)), rel)
        if float(np.linalg.norm(p[-1] - target)) <= tolerance:
            break
    return p, float(np.linalg.norm(p[-1] - target)), used


def chain_rotations(old: np.ndarray, new: np.ndarray) -> list[np.ndarray]:
    """For each bone ``i`` (joint ``i`` to ``i + 1``), the world rotation delta that turns
    its old direction into its new one, applied in chain order (each bone's delta given
    the deltas of the bones before it): ``delta_i`` is the smallest rotation from the
    bone's direction after its parents turned to its new direction."""
    from . import rotation as R

    old = np.asarray(old, np.float64)
    new = np.asarray(new, np.float64)
    deltas = []
    acc = np.array([0.0, 0.0, 0.0, 1.0])
    for i in range(old.shape[0] - 1):
        before = R.rotate(acc[None], (old[i + 1] - old[i])[None])[0]
        q = rotation_between(before, new[i + 1] - new[i])
        deltas.append(q)
        acc = R.multiply(q[None], acc[None])[0]
    return deltas
