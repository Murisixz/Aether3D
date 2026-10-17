"""Walls in plan: location lines, joins and exact footprints.

A wall is a straight location line from ``start`` to ``end`` (plan coordinates, metres),
a thickness, and where the thickness lies relative to the line (``"center"``, or
``"left"``/``"right"`` of the direction of travel). Its footprint -- the polygon its solid
is the vertical extrusion of -- comes from the joins with other walls:

* **End joins** (L, Y, and junctions of three or more walls): the wall ends meeting at a
  point are sorted by direction; each pair of neighbours meets at the intersection of
  the left face line of one with the right face line of the other, and every end is
  closed through the junction point. The footprints then tile the junction exactly:
  no overlap, no gap, for any thicknesses and angles (a kinked miter when thicknesses
  differ). Nearly collinear neighbours (within 1e-9 rad) are joined straight.
* **T joins**: a wall whose end lies on another wall's location line (not at its ends)
  stops at that wall's near face; the through wall is unchanged.
* **Free ends** are square.

Everything is computed in closed form from line intersections in float64; the result
records which joins were made, so a moved wall shows which neighbours it affects.
Walls crossing each other away from their ends are reported (``WALLS_CROSS``) rather
than silently overlapped; zero-length walls and thickness <= 0 are refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import CadError


@dataclass
class WallLine:
    id: str
    start: np.ndarray
    end: np.ndarray
    thickness: float
    location: str = "center"  # center | left | right

    @property
    def direction(self) -> np.ndarray:
        d = self.end - self.start
        return d / np.linalg.norm(d)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def normal(self) -> np.ndarray:  # left of travel
        d = self.direction
        return np.array([-d[1], d[0]])

    def offsets(self) -> tuple[float, float]:
        """Signed distances of the left and right faces from the location line."""
        t = self.thickness
        if self.location == "left":
            return t, 0.0
        if self.location == "right":
            return 0.0, -t
        return t / 2, -t / 2


@dataclass
class Footprint:
    wall: str
    polygon: np.ndarray  # (n, 2), counter-clockwise
    # Parameter range [a, b] (metres along the location line from its start) where both
    # faces are straight and complete: openings must stay inside it.
    clear: tuple[float, float]
    joins: list[dict] = field(default_factory=list)


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _line_intersection(p, d, q, e):
    den = _cross(d, e)
    if abs(den) < 1e-12:
        return None
    t = _cross(q - p, e) / den
    return p + t * d


def footprints(walls: list[WallLine], tol: float) -> dict[str, Footprint]:
    by_id = {w.id: w for w in walls}
    for w in walls:
        if not w.length > tol:
            raise CadError("DEGENERATE_WALL", f"wall {w.id!r} has (almost) no length", w.id)
        if not w.thickness > 0:
            raise CadError("INVALID_WALL", f"wall {w.id!r} needs a positive thickness", w.id)
    # Ends: (wall, which end, point, outgoing direction).
    ends = []
    for w in walls:
        ends.append((w.id, 0, w.start, w.direction))
        ends.append((w.id, 1, w.end, -w.direction))
    # Cluster end points.
    clusters: list[list[int]] = []
    for i, (_, _, p, _) in enumerate(ends):
        for cl in clusters:
            if np.linalg.norm(ends[cl[0]][2] - p) <= tol:
                cl.append(i)
                break
        else:
            clusters.append([i])
    # Face lines at each end, oriented outward from the junction.
    # For an end leaving along u: its left face (seen leaving) and right face.
    corner: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    joins: dict[str, list[dict]] = {w.id: [] for w in walls}
    for cl in clusters:
        if len(cl) < 2:
            continue
        center = np.mean([ends[i][2] for i in cl], axis=0)
        order = sorted(cl, key=lambda i: math.atan2(ends[i][3][1], ends[i][3][0]))
        n = len(order)
        for k in range(n):
            i, j = order[k], order[(k + 1) % n]  # j is counter-clockwise after i
            wi, ei = ends[i][0], ends[i][1]
            wj, ej = ends[j][0], ends[j][1]
            ui, uj = ends[i][3], ends[j][3]
            # Left face of i (seen leaving the junction) and right face of j.
            li = _face_line(by_id[wi], ei, "left", center)
            rj = _face_line(by_id[wj], ej, "right", center)
            if abs(_cross(ui, uj)) < 1e-9 and float(ui @ uj) < 0:
                # Collinear and opposite: a straight continuation.
                p = li[0] - float((li[0] - center) @ ui) * ui
                q = rj[0] - float((rj[0] - center) @ uj) * uj
                corner.setdefault((wi, ei), {})["left"] = p
                corner.setdefault((wj, ej), {})["right"] = q
                continue
            x = _line_intersection(li[0], ui, rj[0], uj)
            if x is None:
                x = 0.5 * (li[0] + rj[0])
            corner.setdefault((wi, ei), {})["left"] = x
            corner.setdefault((wj, ej), {})["right"] = x
        for i in order:
            corner.setdefault((ends[i][0], ends[i][1]), {})["center"] = center
            joins[ends[i][0]].append(
                {
                    "end": "start" if ends[i][1] == 0 else "end",
                    "with": [ends[o][0] for o in order if o != i],
                    "kind": "L" if n == 2 else "junction",
                }
            )
    # T joins: free ends lying on another wall's location line.
    for i, (wid, e, p, u) in enumerate(ends):
        if any(i in cl and len(cl) > 1 for cl in clusters):
            continue
        for other in walls:
            if other.id == wid:
                continue
            d = other.direction
            t = float((p - other.start) @ d)
            dist = abs(_cross(d, p - other.start))
            if tol < t < other.length - tol and dist <= max(tol, other.thickness / 2 + tol):
                # Stop at the face of ``other`` that faces this wall.
                lo, ro = other.offsets()
                side = 1.0 if float((p - u * 1.0 - other.start) @ other.normal) >= 0 else -1.0
                # The wall arrives from the side its body is on (opposite to u at the end).
                body_side = float(u @ other.normal)
                off = lo if body_side > 0 else ro
                face_p = other.start + other.normal * off
                w = by_id[wid]
                for name in ("left", "right"):
                    fl = _face_line(w, e, name, p)
                    x = _line_intersection(fl[0], u, face_p, d)
                    if x is None:
                        raise CadError(
                            "WALLS_PARALLEL",
                            f"walls {wid!r} and {other.id!r} are parallel at a T",
                            wid,
                        )
                    corner.setdefault((wid, e), {})[name] = x
                joins[wid].append(
                    {"end": "start" if e == 0 else "end", "with": [other.id], "kind": "T"}
                )
                joins[other.id].append({"at": t, "with": [wid], "kind": "T-through"})
                del side
                break
    out = {}
    for w in walls:
        lo, ro = w.offsets()
        nrm = w.normal
        pts = []
        # Start end (outgoing u = direction): its "right" (seen leaving) is the wall's right.
        c0 = corner.get((w.id, 0), {})
        c1 = corner.get((w.id, 1), {})
        s_right = c0.get("right", w.start + nrm * ro)
        s_left = c0.get("left", w.start + nrm * lo)
        # End end (outgoing -direction): seen leaving, left/right swap relative to the wall.
        e_left_wall = c1.get("right", w.end + nrm * lo)
        e_right_wall = c1.get("left", w.end + nrm * ro)
        pts.append(s_right)
        pts.append(e_right_wall)
        if "center" in c1:
            pts.append(c1["center"])
        pts.append(e_left_wall)
        pts.append(s_left)
        if "center" in c0:
            pts.append(c0["center"])
        poly = _dedupe(np.array(pts), tol)
        area = 0.5 * float(
            np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
        )
        if area < 0:
            poly = poly[::-1]
        # Clear range: both faces straight between the ends' inner-most corners.
        d = w.direction
        a = max(float((s_right - w.start) @ d), float((s_left - w.start) @ d), 0.0)
        b = min(float((e_right_wall - w.start) @ d), float((e_left_wall - w.start) @ d), w.length)
        out[w.id] = Footprint(w.id, poly, (a, b), joins[w.id])
    _check_crossings(walls, out, tol)
    return out


def _face_line(w: WallLine, end: int, which: str, center) -> tuple[np.ndarray, np.ndarray]:
    """A face line of a wall end, as (point, direction leaving the junction). ``which`` is
    left/right *seen leaving the junction along the wall*."""
    lo, ro = w.offsets()
    nrm = w.normal
    base = w.start if end == 0 else w.end
    u = w.direction if end == 0 else -w.direction
    # Seen leaving the junction, an end's left face is the wall's left face at its
    # start and its right face at its end.
    off = (lo if which == "left" else ro) if end == 0 else (ro if which == "left" else lo)
    return base + nrm * off, u


def _dedupe(poly: np.ndarray, tol: float) -> np.ndarray:
    keep = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.array(keep)


def _check_crossings(walls, fps, tol) -> None:
    lines = [(w.id, w.start, w.end) for w in walls]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, p0, p1 = lines[i]
            b, q0, q1 = lines[j]
            d, e = p1 - p0, q1 - q0
            den = _cross(d, e)
            if abs(den) < 1e-12:
                continue
            t = _cross(q0 - p0, e) / den
            s = _cross(q0 - p0, d) / den
            la, lb = np.linalg.norm(d), np.linalg.norm(e)
            inner_a = tol / la < t < 1 - tol / la
            inner_b = tol / lb < s < 1 - tol / lb
            if inner_a and inner_b:
                raise CadError(
                    "WALLS_CROSS",
                    f"walls {a!r} and {b!r} cross away from their ends",
                    a,
                    hint="split one of them at the crossing (two walls meeting there)",
                )
