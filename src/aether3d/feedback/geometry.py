"""Exact geometric measurements behind the placement checks and their corrections.

Everything here reads the scene's spatial index (``aether3d.spatial``: world-space
vertices, per-object BVHs, the watertight ray test) and changes nothing.

* :func:`vertical_contact` -- how an object sits on a support: the vertical gap between
  them and how deep one penetrates the other, from vertical rays cast from the vertices
  of each against the other (front and back faces tell entering from leaving a surface).
  Exact for configurations where the extreme is reached at a vertex of one object over a
  face of the other -- an object standing on a floor or a table top; an edge resting
  across an edge can be measured too large (reported as ``method``).
* :func:`classify_contact` -- whether two objects whose surfaces meet merely touch: they
  do when a translation of at most ``tolerance`` along one of the six axis directions
  separates them, decided by the exact triangle-intersection test on translated copies.
  Otherwise they penetrate, and :func:`separation` finds, along each direction, a
  translation that separates them (bisection between ``tolerance`` and the translation
  that separates their world boxes, keeping the separated end: every reported
  translation is verified separated by the exact test).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..spatial.index import Entry, SpatialIndex

MAX_RAY_POINTS = 8192
"""Vertices per object used as ray origins (more are subsampled with a fixed stride)."""
AXES = (
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
)


def _points(entry: Entry) -> tuple[np.ndarray, bool]:
    pts = entry.points
    if pts.shape[0] <= MAX_RAY_POINTS:
        return pts, True
    stride = math.ceil(pts.shape[0] / MAX_RAY_POINTS)
    return pts[::stride], False


def cast(
    index: SpatialIndex, entries: list[Entry], origins: np.ndarray, direction: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Entry]]:
    """First hits of rays from ``origins`` along ``direction`` (unit) against ``entries``:
    ``(ray, instance, t, front, ents)`` -- the ray's index, the hit entry's index in
    ``ents``, the distance and whether the ray arrived from the triangle's front side."""
    from ..spatial.queries import _trace

    n = origins.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    if not n or not entries:
        return empty, empty, np.zeros(0), np.zeros(0, dtype=bool), []
    d = np.broadcast_to(np.asarray(direction, dtype=np.float64), (n, 3)).copy()
    hits, ents = _trace(index, entries, origins, d, 0.0, np.inf, "geometry", first=True)
    if not len(hits):
        return empty, empty, np.zeros(0), np.zeros(0, dtype=bool), ents
    front = np.zeros(len(hits), dtype=bool)
    for k in range(len(hits)):
        e = ents[int(hits.inst[k])]
        tri = e.mesh.triangles[int(hits.prim[k])].astype(np.int64)
        a, b, c = e.vertices[tri[0]], e.vertices[tri[1]], e.vertices[tri[2]]
        normal = np.cross(b - a, c - a) * e.orientation
        front[k] = float(normal @ d[0]) < 0.0
    return hits.ray, hits.inst, hits.t, front, ents


def vertical_contact(
    index: SpatialIndex, subject: Entry, supports: list[Entry], tolerance: float
) -> dict[str, Any]:
    """How ``subject`` sits on ``supports`` along the vertical (see the module docstring).

    Returns ``gap`` (the least vertical distance from the subject down to a support's
    surface, ``inf`` when nothing is below), ``penetration`` (the greatest depth of a
    vertex of one inside the other, measured vertically; 0 if none), the supports involved
    and ``status``: ``"resting"``, ``"floating"``, ``"sunken"`` or ``"not_above"``.

    Rays start ``eps`` (a billionth of the coordinates' scale) behind each vertex, so a
    vertex lying on a surface sees it at distance ``eps`` instead of starting on it.
    """
    zhat = np.array([0.0, 0.0, 1.0])
    pts, exact = _points(subject)
    lo, hi = subject.box
    scale = max(1.0, float(np.max(np.abs(np.concatenate([lo, hi])))))
    eps = 1e-9 * scale
    gap, gap_by, gap_point = math.inf, None, None
    depth, depth_by = 0.0, None

    def nearest_front(origins, targets, direction):
        ray, inst, t, front, ents = cast(index, targets, origins, direction)
        if not front.any():
            return None
        k = int(np.argmin(np.where(front, t, np.inf)))
        return max(float(t[k]) - eps, 0.0), ents[int(inst[k])], origins[int(ray[k])]

    def deepest_back(origins, targets, direction):
        ray, inst, t, front, ents = cast(index, targets, origins, direction)
        if front.all():
            return None
        k = int(np.argmax(np.where(~front, t, -np.inf)))
        return max(float(t[k]) - eps, 0.0), ents[int(inst[k])]

    # From the subject's vertices: down onto a support's top (front face) is a gap; up to a
    # support's surface from inside it (back face) is a penetration.
    found = nearest_front(pts + eps * zhat, supports, -zhat)
    if found is not None:
        gap, gap_by, origin = found
        gap_point = origin + eps * zhat - (gap + eps) * zhat
    found = deepest_back(pts - eps * zhat, supports, zhat)
    if found is not None and found[0] > depth:
        depth, depth_by = found
    # From each support's vertices under the subject: up onto the subject's bottom (front
    # face) is a gap; down out of the subject from inside it (back face) is a penetration.
    for support in supports:
        slo, shi = support.box
        if np.any(slo[:2] > hi[:2] + tolerance) or np.any(shi[:2] < lo[:2] - tolerance):
            continue  # their footprints do not overlap
        spts, exact_b = _points(support)
        exact = exact and exact_b
        under = (
            (spts[:, 0] >= lo[0] - tolerance)
            & (spts[:, 0] <= hi[0] + tolerance)
            & (spts[:, 1] >= lo[1] - tolerance)
            & (spts[:, 1] <= hi[1] + tolerance)
        )
        spts = spts[under]
        if not spts.size:
            continue
        found = nearest_front(spts - eps * zhat, [subject], zhat)
        if found is not None and found[0] < gap:
            gap, gap_by = found[0], support
            gap_point = found[2] + eps * zhat
        found = deepest_back(spts + eps * zhat, [subject], -zhat)
        if found is not None and found[0] > depth:
            depth, depth_by = found[0], support
    if depth > tolerance:
        status, gap, gap_point = "sunken", 0.0, None
    elif depth > 0.0:  # in contact (within the tolerance): any gap elsewhere is beside it
        status, gap, gap_point = "resting", 0.0, None
    elif math.isfinite(gap):
        status = "floating" if gap > tolerance else "resting"
    elif _touching(index, subject, supports, tolerance):
        status, gap = "resting", 0.0
    else:
        status = "not_above"
    out: dict[str, Any] = {
        "status": status,
        "gap": gap,
        "penetration": depth,
        "gap_support": None if gap_by is None else _ref(gap_by),
        "penetrated_support": None if depth_by is None else _ref(depth_by),
        "method": "vertical rays from both objects' vertices (watertight ray test)",
        "exact": exact,
    }
    if gap_point is not None and math.isfinite(gap):
        out["gap_point"] = [float(v) for v in gap_point]
    return out


def _touching(index: SpatialIndex, subject: Entry, supports: list[Entry], tol: float) -> bool:
    from ..spatial.queries import _surfaces_meet

    return any(_surfaces_meet(subject, s, tol) for s in supports)


def _ref(entry: Entry) -> dict[str, str]:
    return {"id": entry.obj.id, "name": entry.obj.name}


class Shifted:
    """An index entry translated by ``offset`` (for exact intersection tests of a moved
    copy without rebuilding its BVH): the same triangles, BVH topology and order."""

    def __init__(self, entry: Entry, offset: np.ndarray) -> None:
        self.entry = entry
        self.offset = np.asarray(offset, dtype=np.float64)
        self.obj = entry.obj
        self.bvh = entry.bvh
        self._nodes: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def box(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.entry.box
        return lo + self.offset, hi + self.offset

    @property
    def node_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        if self._nodes is None:
            lo, hi = self.entry.node_boxes
            self._nodes = (lo + self.offset, hi + self.offset)
        return self._nodes

    def corners(self, triangles: Any = None):
        a, b, c = self.entry.corners(triangles)
        return a + self.offset, b + self.offset, c + self.offset


def meets(a: Any, b: Any) -> bool:
    """Whether two (possibly shifted) entries' surfaces share a point (exact)."""
    from ..spatial.queries import _surfaces_meet

    return _surfaces_meet(a, b, 0.0)


def classify_contact(a: Entry, b: Entry, tolerance: float) -> dict[str, Any]:
    """Two entries whose surfaces meet: ``{"contact": True, "axis": [...]}`` when moving
    ``a`` by ``tolerance`` along an axis direction separates them, else
    ``{"contact": False}`` (they penetrate)."""
    for axis in AXES:
        offset = np.asarray(axis) * max(tolerance, 0.0)
        if tolerance > 0.0 and not meets(Shifted(a, offset), b):
            return {"contact": True, "axis": list(axis)}
    return {"contact": False}


def box_separation(a: Entry, b: Entry, direction: np.ndarray) -> float:
    """The translation of ``a`` along unit axis ``direction`` that separates the world
    boxes (and therefore the objects) by a tiny clearance."""
    alo, ahi = a.box
    blo, bhi = b.box
    k = int(np.flatnonzero(direction)[0])
    sign = float(direction[k])
    span = max(float(np.max(np.abs(np.concatenate([alo, ahi, blo, bhi])))), 1.0)
    clearance = 1e-9 * span
    if sign > 0:
        return max(0.0, float(bhi[k] - alo[k])) + clearance
    return max(0.0, float(ahi[k] - blo[k])) + clearance


def separation(
    a: Entry, b: Entry, directions: Any, *, tolerance: float, steps: int = 24
) -> list[dict[str, Any]]:
    """For each unit axis ``direction``: a translation of ``a`` that separates it from
    ``b`` (bisection between ``tolerance`` and the box-separating translation, keeping the
    end verified separated by the exact test), sorted by distance."""
    out = []
    for direction in directions:
        d = np.asarray(direction, dtype=np.float64)
        hi = box_separation(a, b, d)
        lo = min(tolerance, hi)
        if lo > 0.0 and not meets(Shifted(a, d * lo), b):
            hi = lo
        else:
            for _ in range(steps):
                if hi - lo <= max(tolerance * 0.25, 1e-12):
                    break
                mid = 0.5 * (lo + hi)
                if meets(Shifted(a, d * mid), b):
                    lo = mid
                else:
                    hi = mid
        out.append({"direction": [float(v) for v in d], "distance": float(hi)})
    out.sort(key=lambda r: (r["distance"], r["direction"]))
    return out
