"""Exact geometric measurements behind the placement checks and their corrections.

Everything here reads the scene's spatial index (``aether3d.spatial``: world-space
vertices, per-object BVHs, the watertight ray test) and changes nothing.

* :func:`vertical_contact` -- how an object sits on a support: how far it must rise to
  stop penetrating the supports it crosses (exact: the triangle test decides every step
  of a bisection), and otherwise the vertical gap down to the nearest support, from
  vertical rays cast from the vertices of each against the other (exact where the gap is
  reached at a vertex over a face -- an object standing on a floor or a table top; an
  edge resting across an edge can be measured too large).
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
    inst = np.asarray(hits.inst, dtype=np.int64)
    prim = np.asarray(hits.prim, dtype=np.int64)
    for k in np.unique(inst):  # one vectorized pass per object hit
        rows = np.flatnonzero(inst == k)
        e = ents[int(k)]
        a, b, c = e.corners(prim[rows])
        u, v = b - a, c - a
        normal = np.stack(
            [
                u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
                u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
            ],
            axis=1,
        )
        front[rows] = (normal @ d[0]) * e.orientation < 0.0
    return hits.ray, hits.inst, hits.t, front, ents


def vertical_contact(
    index: SpatialIndex, subject: Entry | list[Entry], supports: list[Entry], tolerance: float
) -> dict[str, Any]:
    """How ``subject`` sits on ``supports`` along the vertical (see the module docstring).

    ``subject`` may be several entries: the parts of a group (a prefab of blocks), which
    rest as one.

    * **Penetration** -- for every support the subject's surfaces cross (the exact triangle
      test), the upward translation of the subject that separates them, found by bisection
      and verified separated by the same test; ``penetration`` is the largest. This is
      exact for any configuration (a leg piercing a rug has no vertex inside it).
    * **Gap** -- when nothing is crossed: the least vertical distance from the subject
      down to a support's top (front faces), with vertical rays from the subject's vertices
      down and from the supports' vertices (under the subject) up. Exact where the
      extreme is at a vertex over a face; an edge resting across an edge can be measured
      too large.

    ``status``: ``"sunken"`` (penetration beyond ``tolerance``), ``"resting"`` (touching,
    or a gap within it), ``"floating"`` or ``"not_above"`` (nothing below).
    """
    zhat = np.array([0.0, 0.0, 1.0])
    subjects = as_list(subject)
    depth, depth_by, touching = 0.0, None, None
    lift_axis = [(0.0, 0.0, 1.0)]
    for support in supports:
        if not meets(subjects, [support]):
            continue
        lift = separation(subjects, [support], lift_axis, tolerance=tolerance)[0]["distance"]
        if lift <= tolerance:
            touching = touching or support
        elif lift > depth:
            depth, depth_by = lift, support
    found_points = [_points(e) for e in subjects]
    exact = all(f[1] for f in found_points)
    out: dict[str, Any] = {
        "method": "penetration: the upward translation that separates the surfaces (exact "
        "triangle test, bisection); gap: vertical rays from both objects' vertices",
    }
    if depth > tolerance:
        out.update(
            status="sunken",
            gap=0.0,
            penetration=depth,
            gap_support=None,
            penetrated_support=_ref(depth_by),
            exact=True,
        )
        return out
    if touching is not None:
        out.update(
            status="resting",
            gap=0.0,
            penetration=0.0,
            gap_support=_ref(touching),
            penetrated_support=None,
            exact=True,
        )
        return out
    pts = np.concatenate([f[0] for f in found_points])
    lo, hi = union_box(subjects)
    scale = max(1.0, float(np.max(np.abs(np.concatenate([lo, hi])))))
    eps = 1e-9 * scale
    gap, gap_by, gap_point = math.inf, None, None

    def nearest_front(origins, targets, direction):
        ray, inst, t, front, ents = cast(index, targets, origins, direction)
        if not front.any():
            return None
        k = int(np.argmin(np.where(front, t, np.inf)))
        return max(float(t[k]) - eps, 0.0), ents[int(inst[k])], origins[int(ray[k])]

    found = nearest_front(pts + eps * zhat, supports, -zhat)
    if found is not None:
        gap, gap_by, origin = found
        gap_point = origin - gap * zhat
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
        found = nearest_front(spts - eps * zhat, subjects, zhat)
        if found is not None and found[0] < gap:
            gap, gap_by = found[0], support
            gap_point = found[2] + eps * zhat
    status = "not_above"
    if math.isfinite(gap):
        status = "floating" if gap > tolerance else "resting"
    out.update(
        status=status,
        gap=gap,
        penetration=0.0,
        gap_support=None if gap_by is None else _ref(gap_by),
        penetrated_support=None,
        exact=exact,
    )
    if gap_point is not None and math.isfinite(gap):
        out["gap_point"] = [float(v) for v in gap_point]
    return out


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


def as_list(entries: Any) -> list[Any]:
    return list(entries) if isinstance(entries, (list, tuple)) else [entries]


def union_box(entries: Any) -> tuple[np.ndarray, np.ndarray]:
    """The world box around several entries."""
    boxes = [e.box for e in as_list(entries)]
    return (
        np.min(np.array([b[0] for b in boxes]), axis=0),
        np.max(np.array([b[1] for b in boxes]), axis=0),
    )


def shift(entries: Any, offset: Any) -> list[Shifted]:
    return [Shifted(e, np.asarray(offset, dtype=np.float64)) for e in as_list(entries)]


def meets(a: Any, b: Any, tolerance: float = 0.0, offset: Any = None) -> bool:
    """Whether the surfaces of two entries -- or two groups of entries -- share a point (or
    come within ``tolerance``), with ``a`` moved by ``offset``: exact, on the triangles of
    the pairs of parts whose boxes overlap (only those parts are shifted, and only their
    BVHs are built)."""
    from ..spatial.queries import _surfaces_meet

    first, second = as_list(a), as_list(b)
    move = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    alo = np.array([e.box[0] for e in first]) + move - tolerance
    ahi = np.array([e.box[1] for e in first]) + move + tolerance
    blo = np.array([e.box[0] for e in second])
    bhi = np.array([e.box[1] for e in second])
    near = np.all(alo[:, None, :] <= bhi[None, :, :], axis=2) & np.all(
        blo[None, :, :] <= ahi[:, None, :], axis=2
    )
    moved: dict[int, Any] = {}
    for i, j in zip(*np.nonzero(near), strict=True):
        if first[i].obj is second[j].obj:
            continue
        part = first[i] if offset is None else moved.setdefault(i, Shifted(first[i], move))
        if _surfaces_meet(part, second[j], tolerance):
            return True
    return False


def classify_contact(a: Any, b: Any, tolerance: float) -> dict[str, Any]:
    """Two entries (or groups) whose surfaces meet: ``{"contact": True, "axis": [...]}`` when
    moving ``a`` by ``tolerance`` along an axis direction separates them, else
    ``{"contact": False}`` (they penetrate)."""
    if tolerance > 0.0:
        for axis in AXES:
            if not meets(a, b, offset=np.asarray(axis) * tolerance):
                return {"contact": True, "axis": list(axis)}
    return {"contact": False}


def box_separation(a: Any, b: Any, direction: np.ndarray) -> float:
    """The translation of ``a`` along unit axis ``direction`` that separates the world
    boxes (and therefore the objects) by a tiny clearance."""
    alo, ahi = union_box(a)
    blo, bhi = union_box(b)
    k = int(np.flatnonzero(direction)[0])
    sign = float(direction[k])
    span = max(float(np.max(np.abs(np.concatenate([alo, ahi, blo, bhi])))), 1.0)
    clearance = 1e-9 * span
    if sign > 0:
        return max(0.0, float(bhi[k] - alo[k])) + clearance
    return max(0.0, float(ahi[k] - blo[k])) + clearance


def separation(
    a: Any, b: Any, directions: Any, *, tolerance: float, steps: int = 24
) -> list[dict[str, Any]]:
    """For each unit axis ``direction``: a translation of ``a`` (an entry or a group) that
    separates it from ``b`` (bisection between ``tolerance`` and the box-separating
    translation, keeping the end verified separated by the exact test), sorted by
    distance."""
    out = []
    for direction in directions:
        d = np.asarray(direction, dtype=np.float64)
        hi = box_separation(a, b, d)
        lo = min(tolerance, hi)
        if lo > 0.0 and not meets(a, b, offset=d * lo):
            hi = lo
        else:
            for _ in range(steps):
                if hi - lo <= max(tolerance * 0.25, 1e-12):
                    break
                mid = 0.5 * (lo + hi)
                if meets(a, b, offset=d * mid):
                    lo = mid
                else:
                    hi = mid
        out.append({"direction": [float(v) for v in d], "distance": float(hi)})
    out.sort(key=lambda r: (r["distance"], r["direction"]))
    return out
