"""Ruled lofts between planar sections.

Consecutive sections must have loops with the same number of pieces (after closed loops
are split in two); corresponding pieces are joined by ruled surfaces, exactly:

* segment to segment: a bilinear patch (a degree 1 x 1 NURBS surface; a plane when the
  segments are coplanar);
* arc to arc (or arc to segment): a rational quadratic x linear NURBS surface whose
  boundaries are the exact arcs (Piegl & Tiller A7.1) -- straight rulings between
  corresponding points.

The two end sections become planar caps. Pieces are matched in order starting from each
loop's first piece; loops are oriented counter-clockwise first.
"""

from __future__ import annotations

import math

import numpy as np

from ...errors import CadError
from ..geom2d import Arc, Segment
from ..geom3d import Frame, Line3, NurbsSurface, Plane
from ..nurbs import BSplineSurface, circle_arc_nurbs
from ..sketch.profile import Loop
from .build import Builder, _curve3, _split_closed
from .topology import Body


def _piece_nurbs(c, frame: Frame, segments: int):
    """3D control points and weights of a piece as a rational quadratic with ``segments``
    Bezier spans (segments are elevated: their middle points at the chords' midpoints)."""
    if isinstance(c, Arc):
        center = frame.to_world(np.array([*c.center, 0.0]))
        n = circle_arc_nurbs(center, frame.x, frame.y, c.radius, c.start_angle, c.sweep)
        if (len(n.points) - 1) // 2 != segments:
            # Re-split the arc into the required number of spans.
            pts, w = [], []
            for k in range(segments):
                a0 = c.start_angle + c.sweep * k / segments
                a1 = c.start_angle + c.sweep * (k + 1) / segments
                part = circle_arc_nurbs(center, frame.x, frame.y, c.radius, a0, a1 - a0)
                if (len(part.points) - 1) // 2 != 1:
                    raise CadError("LOFT_UNSUPPORTED", "an arc span exceeds 90 degrees")
                if k == 0:
                    pts.append(part.points[0])
                    w.append(1.0)
                pts += [part.points[1], part.points[2]]
                w += [float(part.weights[1]) if part.weights is not None else 1.0, 1.0]
            return np.array(pts), np.array(w)
        return n.points, (n.weights if n.weights is not None else np.ones(len(n.points)))
    if isinstance(c, Segment):
        p0 = frame.to_world(np.array([*c.p0, 0.0]))
        p1 = frame.to_world(np.array([*c.p1, 0.0]))
        pts = [p0]
        for k in range(segments):
            a = p0 + (p1 - p0) * k / segments
            b = p0 + (p1 - p0) * (k + 1) / segments
            pts += [0.5 * (a + b), b]
        return np.array(pts), np.ones(len(pts))
    raise CadError("LOFT_UNSUPPORTED", "loft sections are made of lines and arcs")


def ruled_loft(sections: list[tuple[Loop, Frame]], *, prefix: str = "loft", group: int = 0) -> Body:
    loops = []
    for loop, frame in sections:
        lp = _split_closed(loop if loop.area > 0 else loop.reversed())
        loops.append((lp, frame))
    n = len(loops[0][0].pieces)
    if any(len(lp.pieces) != n for lp, _ in loops):
        raise CadError(
            "LOFT_UNSUPPORTED",
            "loft sections must have the same number of edges",
            hint=f"counts: {[len(lp.pieces) for lp, _ in loops]}",
        )
    b = Builder()
    verts = []
    for si, (lp, fr) in enumerate(loops):
        verts.append(
            [b.vertex((si, k), fr.to_world(np.array([*lp.pieces[k].start, 0.0]))) for k in range(n)]
        )
    section_edges = []
    for si, (lp, fr) in enumerate(loops):
        row = []
        for k in range(n):
            c = lp.pieces[k].curve
            mid = fr.to_world(np.array([*c.point(0.5), 0.0]))
            row.append(
                b.edge(
                    ("sec", si, k), _curve3(c, fr, 0.0), verts[si][k], verts[si][(k + 1) % n], mid
                )
            )
        section_edges.append(row)
    rails = []
    for si in range(len(loops) - 1):
        row = []
        for k in range(n):
            a, c = b.verts[verts[si][k]], b.verts[verts[si + 1][k]]
            row.append(
                b.edge(
                    ("rail", si, k), Line3(a, c - a), verts[si][k], verts[si + 1][k], 0.5 * (a + c)
                )
            )
        rails.append(row)
    for si in range(len(loops) - 1):
        (la, fa), (lb, fb) = loops[si], loops[si + 1]
        for k in range(n):
            ca, cb = la.pieces[k].curve, lb.pieces[k].curve
            spans = 1
            for c in (ca, cb):
                if isinstance(c, Arc):
                    spans = max(spans, math.ceil(abs(c.sweep) / (math.pi / 2) - 1e-9))
            pa, wa = _piece_nurbs(ca, fa, spans)
            pb, wb = _piece_nurbs(cb, fb, spans)
            knots = (
                [0.0, 0.0, 0.0]
                + [x for i in range(1, spans) for x in (i / spans, i / spans)]
                + [1.0, 1.0, 1.0]
            )
            ctrl = np.stack([pa, pb], axis=1)
            weights = np.stack([wa, wb], axis=1)
            surf = NurbsSurface(BSplineSurface(2, 1, knots, [0, 0, 1, 1], ctrl, weights))
            loop = [
                section_edges[si][k],
                rails[si][(k + 1) % n],
                (section_edges[si + 1][k][0], not section_edges[si + 1][k][1]),
                (rails[si][k][0], not rails[si][k][1]),
            ]
            # Outward: right of travel along section a, for a counter-clockwise loop.
            t2 = ca.tangent(0.5)
            right = fa.direction_to_world(np.array([t2[1], -t2[0], 0.0]))
            mid = fa.to_world(np.array([*ca.point(0.5), 0.0]))
            uv = surf.inverse(mid[None])[0]
            normal = surf.normal(np.array([uv[0]]), np.array([uv[1]]))[0]
            sense = bool(float(normal @ right) > 0)
            b.face(surf, sense, [loop], f"{prefix}:side/{la.pieces[k].source}.{si}", group)
    # Caps: the first section seen from below, the last from above (their frames' z).
    first, last = loops[0][1], loops[-1][1]
    direction = loops[-1][1].origin - loops[0][1].origin
    flip_first = float(first.z @ direction) < 0
    flip_last = float(last.z @ direction) < 0
    cap0 = [(e, not f) for e, f in reversed(section_edges[0])]
    cap1 = list(section_edges[-1])
    b.face(
        Plane(first),
        flip_first,
        [cap0 if not flip_first else list(section_edges[0])],
        f"{prefix}:start",
        group,
    )
    b.face(
        Plane(last),
        not flip_last,
        [cap1 if not flip_last else [(e, not f) for e, f in reversed(section_edges[-1])]],
        f"{prefix}:end",
        group,
    )
    return b.body()
