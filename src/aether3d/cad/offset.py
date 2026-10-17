"""Exact offsets of profile loops made of segments and arcs.

Offsetting a segment gives a parallel segment and an arc a concentric arc, exactly. At
each corner the offset pieces are trimmed or extended to meet: lines and circles are
intersected in closed form (the intersection nearest the original corner is taken), which
is what offset planes and cylinders do in 3D (a drafted or shelled prism). Where two
pieces met tangentially, their offsets still meet tangentially at the offset point.

The offset is refused (``OFFSET_TOPOLOGY_CHANGE``) when it would change the loop's
structure: a piece reversing or vanishing, an arc collapsing, or pieces crossing --
an exact offset with a different topology is a different shape, never silently produced.
Splines cannot be offset exactly (``UNSUPPORTED_GEOMETRY``).
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import CadError
from .geom2d import Arc, Segment, Spline, angle_of, intersect
from .sketch.profile import Loop, Piece

TAU = 2.0 * math.pi


def _tangent_at(c, t: float) -> np.ndarray:
    d = c.tangent(t)
    return d / np.linalg.norm(d)


def offset_loop(loop: Loop, distance: float, tol: float) -> Loop:
    """The loop offset by ``distance`` to the left of its direction of travel."""
    if distance == 0:
        return loop
    pieces = loop.pieces
    for p in pieces:
        if isinstance(p.curve, Spline):
            raise CadError(
                "UNSUPPORTED_GEOMETRY",
                f"the spline {p.source!r} cannot be offset exactly",
                hint="use lines and arcs in profiles that are offset (draft, shell)",
            )
    off = [p.curve.offset(distance) for p in pieces]
    n = len(off)
    corners: list[np.ndarray] = []
    for i in range(n):
        a, b = off[i], off[(i + 1) % n]
        orig = pieces[i].curve.end
        ta, tb = _tangent_at(pieces[i].curve, 1.0), _tangent_at(pieces[(i + 1) % n].curve, 0.0)
        if abs(ta[0] * tb[1] - ta[1] * tb[0]) < 1e-12 and float(ta @ tb) > 0:
            corners.append(a.end)  # tangent continuation: the offsets meet there
            continue
        hits = intersect(a, b, tol, extend=True)
        if not hits:
            raise CadError(
                "OFFSET_TOPOLOGY_CHANGE",
                f"the offset pieces of {pieces[i].source!r} and "
                f"{pieces[(i + 1) % n].source!r} no longer meet",
                hint="use a smaller offset (draft angle, shell thickness)",
            )
        best = min(hits, key=lambda h: float(np.linalg.norm(h.point - orig)))
        corners.append(best.point)
    out = []
    for i in range(n):
        start, end = corners[i - 1], corners[i]
        c = off[i]
        orig = pieces[i].curve
        if isinstance(c, Segment):
            new = Segment(start, end)
            d0 = orig.p1 - orig.p0
            d1 = new.p1 - new.p0
            if float(d0 @ d1) <= 0 or new.length <= tol:
                raise CadError(
                    "OFFSET_TOPOLOGY_CHANGE",
                    f"the offset of segment {pieces[i].source!r} would vanish or reverse",
                    hint="use a smaller offset",
                )
        else:
            a0 = angle_of(start - c.center)
            a1 = angle_of(end - c.center)
            sweep = (a1 - a0) % TAU if c.sweep > 0 else -((a0 - a1) % TAU)
            if abs(sweep) < 1e-12 or abs(abs(sweep) - abs(orig.sweep)) > math.pi:
                raise CadError(
                    "OFFSET_TOPOLOGY_CHANGE",
                    f"the offset of arc {pieces[i].source!r} would vanish or wrap around",
                    hint="use a smaller offset",
                )
            new = Arc(c.center, c.radius, a0, sweep)
        out.append(Piece(new, pieces[i].source, pieces[i].index))
    result = Loop(out)
    # Non-adjacent pieces must not cross.
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if intersect(out[i].curve, out[j].curve, tol):
                raise CadError(
                    "OFFSET_TOPOLOGY_CHANGE",
                    f"the offset loop crosses itself ({out[i].source!r}, {out[j].source!r})",
                    hint="use a smaller offset",
                )
    if (result.area > 0) != (loop.area > 0):
        raise CadError("OFFSET_TOPOLOGY_CHANGE", "the offset loop turned inside out")
    return result
