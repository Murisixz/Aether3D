"""Measurements of a humanoid body for landmark detection: exact probes and sections.

:mod:`.landmarks` finds a character's extremities and limb centrelines on a voxelized
solid (coarse but robust to any mesh). Joint centres need finer measurements than a
voxel: this module measures the **mesh itself**, in the detection's canonical frame (up
``+z``, facing ``-y``, the character's left ``+x``, the mirror plane at ``x = 0``, the
soles at ``z = 0``):

* :class:`Probe` -- *reach*: the distance from a point inside the body along a direction
  to the surface, by Aether3D's watertight ray traversal against the mesh's triangles
  (the voxel solid only decides whether a point is inside);
* horizontal **sections** of the midline (the front, back and side surfaces at a height),
  from which the head, chin, neck, the neck's base and the spine's depth are read;
* **limb profiles** -- reaches around a limb's axis line (up, down, forward, back, in,
  out) at stations along it, from which shoulders, hips, wrists and elbows are read.

Every estimate is a measurement with a stated method; the anthropometric fractions used
where the surface shows nothing (straight knees and elbows, the head's pivot below the
crown) are named in the landmark's note.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])
BACK = np.array([0.0, 1.0, 0.0])


class Probe:
    """Exact distances from points inside a body to its surface, along directions."""

    def __init__(self, positions: np.ndarray, triangles: np.ndarray, inside: Any) -> None:
        from ..rendering.raytrace.accel import BLASCache, SceneAccel
        from ..rendering.scene_data import RenderMesh

        pos = np.ascontiguousarray(np.asarray(positions, np.float64).astype(np.float32))
        tris = np.ascontiguousarray(np.asarray(triangles).astype(np.uint32))
        pos.setflags(write=False)
        tris.setflags(write=False)
        rs = types.SimpleNamespace(
            meshes=(RenderMesh(pos, tris, None),),
            instances=(types.SimpleNamespace(mesh=0, world=np.eye(4)),),
        )
        self.accel = SceneAccel.build(rs, cache=BLASCache(budget=256 * 2**20))
        self.inside = inside  # (n, 3) canonical points -> (n,) bool
        p64 = np.asarray(positions, np.float64)
        t64 = np.asarray(triangles, np.int64)
        self.normals = np.cross(p64[t64[:, 1]] - p64[t64[:, 0]], p64[t64[:, 2]] - p64[t64[:, 0]])

    def reach(self, points: np.ndarray, directions: np.ndarray, most: float) -> np.ndarray:
        """Distance from each point along its (unit) direction to the first surface, ``0``
        for points outside the body: outside the voxel solid, or where the ray meets no
        surface within ``most`` or first meets one it enters (a point in a gap the voxels
        filled, between fingers or under an arm: from inside a closed body every ray
        leaves through the surface)."""
        p = np.atleast_2d(np.asarray(points, np.float64))
        d = np.atleast_2d(np.asarray(directions, np.float64))
        d = np.ascontiguousarray(np.broadcast_to(d, p.shape))
        hits = self.accel.closest_hit(np.ascontiguousarray(p), d, 0.0, float(most))
        hit = hits.inst >= 0
        leaving = np.zeros(p.shape[0], bool)
        if hit.any():
            n = self.normals[hits.prim[hit]]
            leaving[hit] = np.einsum("ij,ij->i", n, d[hit]) > 0.0
        return np.where(self.inside(p) & leaving, hits.t, 0.0)

    def around(
        self, points: np.ndarray, axis: np.ndarray, ref: np.ndarray, most: float, n: int = 8
    ) -> np.ndarray:
        """Reaches from each point in ``n`` directions perpendicular to ``axis`` (the
        first along ``ref`` projected off the axis, then turning about the axis):
        ``(points, n)``."""
        a = axis / np.linalg.norm(axis)
        u = ref - (ref @ a) * a
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        v = np.cross(a, u)
        p = np.atleast_2d(points)
        out = np.empty((p.shape[0], n))
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            d = np.cos(ang) * u + np.sin(ang) * v
            out[:, k] = self.reach(p, np.broadcast_to(d, p.shape), most)
        return out

    def radius(
        self, points: np.ndarray, axes: np.ndarray, ref: np.ndarray, most: float, n: int = 8
    ) -> np.ndarray:
        """The mean reach of each point in ``n`` directions perpendicular to its own axis
        (a limb's mean radius at points of its centreline; 0 outside the body)."""
        p = np.atleast_2d(np.asarray(points, np.float64))
        a = np.atleast_2d(np.asarray(axes, np.float64))
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        r = np.broadcast_to(np.asarray(ref, np.float64), a.shape)
        u = r - np.sum(r * a, axis=1, keepdims=True) * a
        bad = np.linalg.norm(u, axis=1) < 1e-6
        if bad.any():  # (the reference along the axis: any perpendicular)
            alt = np.cross(a[bad], np.array([1.0, 0.0, 0.0]))
            alt[np.linalg.norm(alt, axis=1) < 1e-6] = np.cross(
                a[bad][np.linalg.norm(alt, axis=1) < 1e-6], np.array([0.0, 1.0, 0.0])
            )
            u[bad] = alt
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        v = np.cross(a, u)
        total = np.zeros(p.shape[0])
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            total += self.reach(p, np.cos(ang) * u + np.sin(ang) * v, most)
        return total / n


def first_exit(probe: Probe, start: np.ndarray, direction: np.ndarray, most: float) -> float:
    """How far a point can move from ``start`` along ``direction`` and stay inside."""
    return float(probe.reach(start[None], direction[None], most)[0])


def plateau_start(values: np.ndarray, fraction: float = 0.97) -> int:
    """The first index whose value reaches ``fraction`` of the maximum."""
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return 0
    return int(np.flatnonzero(v >= fraction * float(v.max()))[0])


# --------------------------------------------------------------------------------------- #
# Midline sections
# --------------------------------------------------------------------------------------- #


class Midline:
    """Horizontal sections through the mirror plane: at height ``z``, the front and back
    surfaces (``y``) through a centre found iteratively, and the half width (``x``)."""

    def __init__(self, probe: Probe, height: float) -> None:
        self.probe = probe
        self.H = height

    def section(self, z: np.ndarray, y0: np.ndarray) -> dict[str, np.ndarray]:
        """``front``, ``back`` (y of the surfaces), ``centre`` and ``half_width`` at each
        height (``y0``: a point inside the body at that height, near its middle)."""
        z = np.asarray(z, np.float64)
        y = np.asarray(y0, np.float64).copy()
        most = 0.6 * self.H
        for _ in range(2):  # recentre on the midpoint of front and back
            p = np.stack([np.zeros_like(z), y, z], axis=1)
            f = self.probe.reach(p, -BACK, most)
            b = self.probe.reach(p, BACK, most)
            inside = (f > 0) | (b > 0)
            y = np.where(inside, y + 0.5 * (b - f), y)
        p = np.stack([np.zeros_like(z), y, z], axis=1)
        f = self.probe.reach(p, -BACK, most)
        b = self.probe.reach(p, BACK, most)
        w = self.probe.reach(p, LEFT, most)
        w2 = self.probe.reach(p, -LEFT, most)
        return {
            "front": y - f,
            "back": y + b,
            "centre": y,
            "half_width": 0.5 * (w + w2),
            "inside": (f > 0) & (b > 0),
        }


def smooth1d(v: np.ndarray, k: int = 2) -> np.ndarray:
    """A moving average over ``2k + 1`` samples (ends: shorter windows)."""
    if v.size <= 2 * k:
        return v.copy()
    c = np.concatenate([[0.0], np.cumsum(v)])
    out = np.empty_like(v, dtype=np.float64)
    for i in range(v.size):
        lo, hi = max(0, i - k), min(v.size, i + k + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


# --------------------------------------------------------------------------------------- #
# Fingers: measured on the mesh surface (thinner than the voxels)
# --------------------------------------------------------------------------------------- #

# Phalanges as fractions of a finger's length from its knuckle (proximal, middle, distal:
# 0.48, 0.29, 0.23 in Buryanov and Kotiuk 2010), and the thumb's from its base
# (metacarpal, proximal, distal).
PHALANGES = (0.49, 0.28, 0.23)
THUMB = (0.42, 0.32, 0.26)
KNUCKLE_IN = 0.12  # of a finger's free length: from where it parts to its knuckle


def hand_fingers(
    positions: np.ndarray,
    triangles: np.ndarray,
    wrist: np.ndarray,
    tip: np.ndarray,
    H: float,
) -> dict[str, Any]:
    """The fingers of one hand, from its mesh: ``{"count", "tips", "fingers": {name:
    [knuckle, middle joint, end joint, tip]}, "thumb": [base, middle, end, tip] | None,
    "mitten": bool}`` (fingers named by their distance from the thumb: index nearest,
    whatever way the hand faces). The hand's surface beyond the wrist is searched for
    fingertips (local maxima of the distance along the surface from the wrist); each
    finger's centreline is the centroids of rings of equal surface distance from its tip,
    up to where the ring widens into the palm; its knuckle lies on that axis
    :data:`KNUCKLE_IN` of the finger's free length further in."""
    from .weights import mesh_edges, surface_distances

    p = np.asarray(positions, np.float64)
    t = np.asarray(triangles, np.int64)
    axis = tip - wrist
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return {"count": 0, "tips": [], "fingers": {}, "thumb": None, "mitten": False}
    axis = axis / length
    rel = p - wrist
    along = rel @ axis
    near = (along > -0.01 * H) & (np.linalg.norm(rel, axis=1) < 1.35 * length)
    idx = np.flatnonzero(near)
    if idx.size < 50:
        return {"count": 0, "tips": [], "fingers": {}, "thumb": None, "mitten": False}
    local = np.full(len(p), -1, np.int64)
    local[idx] = np.arange(idx.size)
    keep_t = np.all(local[t] >= 0, axis=1)
    lt = local[t[keep_t]]
    q = p[idx]
    ea, eb = mesh_edges(lt)
    # Connected to the tip (not the thigh the hand hangs by).
    seed_tip = int(np.argmin(np.linalg.norm(q - tip, axis=1)))
    reach_tip = surface_distances(q, ea, eb, np.array([seed_tip]), 3.0 * length)
    part = np.isfinite(reach_tip)
    wrist_ring = np.flatnonzero(part & (np.abs((q - wrist) @ axis) < 0.01 * H))
    if wrist_ring.size == 0:
        wrist_ring = np.array([int(np.argmin(np.where(part, (q - wrist) @ axis, np.inf)))])
    g = surface_distances(q, ea, eb, wrist_ring, 3.0 * length)
    g = np.where(part, g, -np.inf)
    # Fingertips: the farthest points from the wrist, each suppressing its neighbourhood.
    order = np.argsort(-g)
    gmax = float(g[order[0]])
    tips: list[int] = []
    for v in order.tolist():
        if g[v] < 0.45 * gmax or len(tips) >= 6:
            break
        if all(np.linalg.norm(q[v] - q[u]) > 0.013 * H for u in tips):
            # a local maximum: nothing within a finger's width along the surface (not
            # across the gap to the next finger) is farther from the wrist
            ball = surface_distances(q, ea, eb, np.array([v]), 0.012 * H) < 0.012 * H
            if g[ball].max() <= g[v] + 1e-12:
                tips.append(v)
    out: dict[str, Any] = {"count": len(tips), "tips": [q[v].copy() for v in tips]}
    if len(tips) < 4:
        out.update(fingers={}, thumb=None, mitten=True)
        return out
    chains = []
    edge_len = float(np.median(np.linalg.norm(q[ea] - q[eb], axis=1))) if ea.size else 0.0
    step = max(0.004 * H, 0.75 * edge_len)
    for v in tips:
        d = surface_distances(q, ea, eb, np.array([v]), 0.13 * H)
        cents, spreads = [], []
        k = 1
        while k * step < 0.12 * H:
            ring = _contour(q, ea, eb, d, k * step)
            if len(ring) < 4:
                break
            c = ring.mean(axis=0)
            spread = float(np.mean(np.linalg.norm(ring - c, axis=1)))
            if spreads and spread > 1.9 * float(np.median(spreads[: max(2, len(spreads) // 2)])):
                break
            cents.append(c)
            spreads.append(spread)
            k += 1
        chains.append((v, np.array(cents) if cents else q[[v]], spreads))
    # How well the mesh shows the fingers: their radius (the rings near their tips) in
    # edges of the hand's mesh.
    radii = [float(np.median(sp[:3])) for _, _, sp in chains if sp]
    out["edge"] = edge_len
    out["finger_radius"] = float(np.median(radii)) if radii else 0.0
    # The thumb: its finger's base lies nearest the wrist.
    bases = [ch[1][-1] for ch in chains]
    thumb_i = int(np.argmin([float((b - wrist) @ axis) for b in bases]))
    fingers = {}
    others = [i for i in range(len(chains)) if i != thumb_i]
    others.sort(key=lambda i: float(np.linalg.norm(bases[i] - bases[thumb_i])))
    names = ("index", "middle", "ring", "pinky")
    for name, i in zip(names, others[:4], strict=False):
        v, cents, _ = chains[i]
        tip_p = q[v]
        free = float(np.linalg.norm(cents[-1] - tip_p)) if len(cents) else 0.0
        dirn = tip_p - cents[-1] if len(cents) else axis
        dirn = dirn / max(float(np.linalg.norm(dirn)), 1e-12)
        # (fingers modeled from their knuckles part there; a webbed hand parts a third of
        # the proximal phalanx further out: the knuckle lies between the two)
        knuckle = cents[-1] - dirn * (KNUCKLE_IN * free)
        total = free * (1.0 + KNUCKLE_IN)
        poly = np.vstack([knuckle[None], cents[::-1], tip_p[None]])
        sp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
        sp = sp * (total / max(float(sp[-1]), 1e-12))
        joints = [knuckle]
        acc = 0.0
        for f in PHALANGES[:2]:
            acc += f * total
            joints.append(np.stack([np.interp(acc, sp, poly[:, j]) for j in range(3)]))
        joints.append(tip_p.copy())
        fingers[name] = joints
    v, cents, _ = chains[thumb_i]
    tip_p = q[v]
    base = cents[-1] if len(cents) else tip_p
    # The thumb's base joint lies in the heel of the hand, beside the wrist.
    cmc = wrist + 0.35 * (base - wrist)
    poly = np.vstack([cmc[None], cents[::-1], tip_p[None]])
    sp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    joints = [cmc]
    acc = 0.0
    for f in THUMB[:2]:
        acc += f * float(sp[-1])
        joints.append(np.stack([np.interp(acc, sp, poly[:, j]) for j in range(3)]))
    joints.append(tip_p.copy())
    out.update(fingers=fingers, thumb=joints, mitten=False)
    return out


def _contour(q: np.ndarray, ea: np.ndarray, eb: np.ndarray, d: np.ndarray, level: float):
    """The points where the surface distance ``d`` crosses ``level`` along the edges (a
    ring around a finger, however coarse its triangles)."""
    da, db = d[ea], d[eb]
    ok = np.isfinite(da) & np.isfinite(db)
    lo, hi = np.minimum(da, db), np.maximum(da, db)
    cross = ok & (lo < level) & (hi >= level) & (hi > lo)
    a, b = ea[cross], eb[cross]
    t = (level - d[a]) / (d[b] - d[a])
    return q[a] + t[:, None] * (q[b] - q[a])
