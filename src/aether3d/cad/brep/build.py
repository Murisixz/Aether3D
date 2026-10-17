"""Exact B-rep construction: extrusions, revolutions and primitive solids.

Every builder produces an exact :class:`~aether3d.cad.brep.topology.Body` from analytic
input -- no tessellation is involved:

* ``extrude(region, frame, start, end, draft=0)``: segments sweep planes, arcs cylinders
  (cones when drafted), splines linear-extrusion surfaces; the caps are planes bounded by
  the region's loops. Draft tapers the profile by ``tan(draft)`` per unit of height
  (positive: the profile shrinks toward ``end``), computed with the exact loop offset.
* ``revolve(region, frame, axis_point, axis_dir, angle)``: segments sweep planes,
  cylinders or cones, arcs spheres or tori, splines revolution surfaces; partial
  revolutions get planar caps. Full revolutions are made of two half turns, so no face
  wraps onto itself (no seam edges).
* ``box``, ``cylinder``, ``cone``, ``sphere``, ``torus`` -- built by the two above.

Face names follow the persistent naming scheme: ``"<prefix>:side/<entity>[.<piece>]"``,
``"<prefix>:start"``/``":end"`` caps (extrusions), ``"<prefix>:rev/<entity>.<half>"``
and ``":cap0"``/``":cap1"`` (revolutions). Coordinates are metres; ``frame`` is the
sketch plane (x, y in the sketch, z its normal).
"""

from __future__ import annotations

import math

import numpy as np

from ...errors import CadError
from ..geom2d import Arc, Segment, Spline
from ..geom3d import Circle3, Cone, Cylinder, Frame, Line3, Nurbs3, Plane, Revolution, Sphere, Torus
from ..geom3d import Extrusion as ExtrusionSurface
from ..nurbs import BSplineCurve
from ..offset import offset_loop
from ..sketch.profile import Loop, Piece, Region
from ..tolerance import Tolerance
from .topology import Body, Edge, Face, edge_range

TAU = 2.0 * math.pi


class Builder:
    """Accumulates shared vertices, edges and faces by key."""

    def __init__(self) -> None:
        self.verts: list[np.ndarray] = []
        self.vkeys: dict = {}
        self.edges: list[Edge] = []
        self.ekeys: dict = {}
        self.faces: list[Face] = []

    def vertex(self, key, p) -> int:
        if key in self.vkeys:
            return self.vkeys[key]
        self.verts.append(np.asarray(p, dtype=np.float64).reshape(3))
        self.vkeys[key] = len(self.verts) - 1
        return self.vkeys[key]

    def edge(self, key, curve, a: int, b: int, pm, name: str = "") -> tuple[int, bool]:
        """The edge (shared by key) from vertex ``a`` to ``b`` through ``pm``; returns
        ``(edge index, forward)`` for a coedge running from ``a`` to ``b``."""
        if key in self.ekeys:
            ei = self.ekeys[key]
            return ei, self.edges[ei].v0 == a
        if a == b:
            raise CadError("INTERNAL", "closed edges are built from two halves")
        pa, pb = self.verts[a], self.verts[b]
        t0, t1 = edge_range(curve, pa, np.asarray(pm, dtype=np.float64), pb)
        q0 = curve.point(t0)
        forward = bool(np.linalg.norm(q0 - pa) <= np.linalg.norm(q0 - pb))
        v0, v1 = (a, b) if forward else (b, a)
        self.edges.append(Edge(curve, t0, t1, v0, v1, name))
        self.ekeys[key] = len(self.edges) - 1
        return len(self.edges) - 1, forward

    def face(
        self, surface, sense: bool, loops, name: str, group: int = 0, window: float | None = None
    ) -> int:
        self.faces.append(
            Face(surface, bool(sense), tuple(tuple(lp) for lp in loops), name, group, window)
        )
        return len(self.faces) - 1

    def body(self, kind: str = "solid") -> Body:
        return Body(
            np.array(self.verts) if self.verts else np.zeros((0, 3)),
            tuple(self.edges),
            tuple(self.faces),
            (tuple(range(len(self.faces))),),
            kind,
        )


def _piece_name(p: Piece) -> str:
    return f"{p.source}" if p.index == 0 else f"{p.source}.{p.index}"


def _split_closed(loop: Loop) -> Loop:
    """Loops of one closed piece are split in two (every loop then has two vertices)."""
    if len(loop.pieces) != 1:
        return loop
    p = loop.pieces[0]
    c = p.curve
    return Loop([Piece(c.sub(0.0, 0.5), p.source, 0), Piece(c.sub(0.5, 1.0), p.source, 1)])


def _curve3(c, frame: Frame, z: float):
    """A 2D sketch curve lifted to height ``z`` above the sketch plane."""
    if isinstance(c, Segment):
        p0 = frame.to_world(np.array([c.p0[0], c.p0[1], z]))
        d = frame.direction_to_world(np.array([*(c.p1 - c.p0), 0.0]))
        return Line3(p0, d)
    if isinstance(c, Arc):
        center = frame.to_world(np.array([c.center[0], c.center[1], z]))
        return Circle3(Frame(center, frame.x, frame.y, frame.z), c.radius)
    if isinstance(c, Spline):
        cv = c.curve
        pts = frame.to_world(np.c_[cv.points, np.full(len(cv.points), z)])
        return Nurbs3(BSplineCurve(cv.degree, cv.knots, pts, cv.weights))
    raise CadError("INVALID_GEOMETRY", f"cannot lift {c!r}")


def _to3(frame: Frame, p2, z: float) -> np.ndarray:
    return frame.to_world(np.array([p2[0], p2[1], z]))


def _sense(surface, point: np.ndarray, outward: np.ndarray) -> bool:
    uv = surface.inverse(point[None])[0]
    n = surface.normal(uv[0], uv[1]).reshape(3)
    return bool(float(n @ outward) > 0)


def extrude(
    region: Region,
    frame: Frame,
    start: float,
    end: float,
    *,
    draft: float = 0.0,
    prefix: str = "extrude",
    group: int = 0,
) -> Body:
    """A prism (or drafted prism) from a planar region between heights start < end."""
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise CadError("INVALID_EXTENT", "an extrusion needs end > start", prefix)
    h = end - start
    if not -math.radians(89) < draft < math.radians(89):
        raise CadError("INVALID_DRAFT", "the draft angle must be within (-89, 89) degrees", prefix)
    size = max(float(np.max(np.abs(np.concatenate([lp.polyline(1e9) for lp in region.loops])))), h)
    tol = Tolerance.for_scale(size + abs(start) + abs(end)).resolution
    loops = [_split_closed(lp) for lp in region.loops]
    tops = loops
    if draft != 0.0:
        tops = [offset_loop(lp, h * math.tan(draft), tol) for lp in loops]
    b = Builder()
    side_names: set[str] = set()
    bottom_loops, top_loops = [], []
    for li, (lb, lt) in enumerate(zip(loops, tops, strict=True)):
        n = len(lb.pieces)
        vb = [b.vertex(("b", li, k), _to3(frame, lb.pieces[k].start, start)) for k in range(n)]
        vt = [b.vertex(("t", li, k), _to3(frame, lt.pieces[k].start, end)) for k in range(n)]
        bot_coedges, top_coedges = [], []
        for k in range(n):
            pb, pt = lb.pieces[k], lt.pieces[k]
            k1 = (k + 1) % n
            name = _piece_name(pb)
            cb = _curve3(pb.curve, frame, start)
            ct = _curve3(pt.curve, frame, end)
            eb = b.edge(("eb", li, k), cb, vb[k], vb[k1], _to3(frame, pb.curve.point(0.5), start))
            et = b.edge(("et", li, k), ct, vt[k], vt[k1], _to3(frame, pt.curve.point(0.5), end))
            ev0 = b.edge(
                ("ev", li, k),
                Line3(b.verts[vb[k]], b.verts[vt[k]] - b.verts[vb[k]]),
                vb[k],
                vt[k],
                0.5 * (b.verts[vb[k]] + b.verts[vt[k]]),
            )
            ev1 = b.edge(
                ("ev", li, k1),
                Line3(b.verts[vb[k1]], b.verts[vt[k1]] - b.verts[vb[k1]]),
                vb[k1],
                vt[k1],
                0.5 * (b.verts[vb[k1]] + b.verts[vt[k1]]),
            )
            bot_coedges.append(eb)
            top_coedges.append(et)
            # Side surface.
            c = pb.curve
            if isinstance(c, Segment):
                d3 = frame.direction_to_world(np.array([*(c.p1 - c.p0), 0.0]))
                p0 = _to3(frame, c.p0, start)
                p1t = b.verts[vt[k1]]
                normal = np.cross(d3, p1t - b.verts[vb[k1]])
                surf = Plane(Frame.from_axes(p0, normal, d3))
            elif isinstance(c, Arc):
                center = _to3(frame, c.center, start)
                rt = lt.pieces[k].curve.radius
                if draft == 0.0:
                    surf = Cylinder(Frame(center, frame.x, frame.y, frame.z), c.radius)
                else:
                    slope = (rt - c.radius) / h
                    if slope > 0:
                        surf = Cone(
                            Frame(center, frame.x, frame.y, frame.z), c.radius, math.atan(slope)
                        )
                    else:
                        fr = Frame.from_axes(center, -frame.z, frame.x)
                        surf = Cone(fr, c.radius, math.atan(-slope))
            else:
                if draft != 0.0:
                    raise CadError("UNSUPPORTED_GEOMETRY", "splines cannot be drafted", prefix)
                surf = ExtrusionSurface(cb, frame.z * 1.0)
            # Outward direction at the piece's middle: right of travel.
            tmid = c.tangent(0.5)
            right2 = np.array([tmid[1], -tmid[0]])
            outward = frame.direction_to_world(np.array([*right2, 0.0]))
            mid = _to3(frame, c.point(0.5), start + 0.5 * h if draft == 0.0 else start)
            sense = _sense(surf, mid, outward / np.linalg.norm(outward))
            fname = f"{prefix}:side/{name}"
            while fname in side_names:
                fname += "'"
            side_names.add(fname)
            loop = [eb, ev1, (et[0], not et[1]), (ev0[0], not ev0[1])]
            window = float(surf.inverse(mid[None])[0, 0]) if surf.periodic_u else None
            b.face(surf, sense, [loop], fname, group, window)
        bottom_loops.append([(e, not f) for e, f in reversed(bot_coedges)])
        top_loops.append(top_coedges)
    cap0 = Plane(Frame(frame.origin + start * frame.z, frame.x, frame.y, frame.z))
    cap1 = Plane(Frame(frame.origin + end * frame.z, frame.x, frame.y, frame.z))
    b.face(cap0, False, bottom_loops, f"{prefix}:start", group)
    b.face(cap1, True, top_loops, f"{prefix}:end", group)
    return b.body()


def _rho_h(frame: Frame, axis_point, axis_dir) -> tuple[Frame, callable]:
    """The revolution frame: z along the axis, x toward the profile (in the sketch plane)."""
    a = frame.to_world(np.array([axis_point[0], axis_point[1], 0.0]))
    d = frame.direction_to_world(np.array([axis_dir[0], axis_dir[1], 0.0]))
    d = d / np.linalg.norm(d)
    x = np.cross(d, frame.z)  # in the sketch plane, perpendicular to the axis
    fr = Frame.from_axes(a, d, x)
    return fr


def revolve(
    region: Region,
    frame: Frame,
    axis_point,
    axis_dir,
    angle: float = TAU,
    *,
    prefix: str = "revolve",
    group: int = 0,
) -> Body:
    """A solid of revolution of a planar region about an axis in its plane."""
    if not (0 < angle <= TAU + 1e-12):
        raise CadError("INVALID_ANGLE", "a revolution's angle must be in (0, 360] degrees", prefix)
    angle = min(angle, TAU)
    full = angle >= TAU - 1e-12
    ap = np.asarray(axis_point, dtype=np.float64)
    ad = np.asarray(axis_dir, dtype=np.float64)
    if np.linalg.norm(ad) == 0:
        raise CadError("DEGENERATE_GEOMETRY", "the axis direction is zero", prefix)
    ad2 = ad / np.linalg.norm(ad)
    # Side of the axis (2D): signed distance of profile points.
    pts = np.concatenate(
        [lp.polyline(1e-6 * max(1.0, region.outer.bbox()[1].max())) for lp in region.loops]
    )
    sd = (pts[:, 0] - ap[0]) * ad2[1] - (pts[:, 1] - ap[1]) * ad2[0]
    size = float(np.max(np.abs(pts))) + 1e-3
    tol = Tolerance.for_scale(size).resolution
    if np.any(sd > tol) and np.any(sd < -tol):
        raise CadError(
            "REVOLVE_CROSSES_AXIS",
            "the profile lies on both sides of the axis",
            prefix,
            hint="keep the profile on one side (it may touch the axis)",
        )
    if np.all(sd <= tol):
        ad2 = -ad2  # profile on the right: revolve about the reversed axis
    rf = _rho_h(frame, ap, ad2)
    # Profile point -> (rho, h) and back.

    def rh(p2) -> np.ndarray:
        p3 = frame.to_world(np.array([p2[0], p2[1], 0.0]))
        loc = rf.to_local(p3)
        return np.array([math.hypot(loc[0], loc[1]), loc[2]])

    spans = [(0.0, math.pi), (math.pi, TAU)] if full else [(0.0, angle)]
    b = Builder()
    loops = [_split_closed(lp) for lp in region.loops]

    def rot_point(p2, u: float) -> np.ndarray:
        r, hh = rh(p2)
        return rf.to_world(np.array([r * math.cos(u), r * math.sin(u), hh]))

    def on_axis(p2) -> bool:
        return rh(p2)[0] <= tol

    names: set[str] = set()
    sorted({u for s in spans for u in s})
    for li, lp in enumerate(loops):
        n = len(lp.pieces)
        for k in range(n):
            piece = lp.pieces[k]
            c = piece.curve
            nk = (k + 1) % n
            pa, pb = piece.start, piece.end
            ka, kb = (li, k), (li, nk)
            if isinstance(c, Segment) and on_axis(pa) and on_axis(pb):
                continue  # lies on the axis: no face
            surf = _revolved_surface(c, rh, rf, prefix, piece)
            for hi, (u0, u1) in enumerate(spans):
                # Vertices (points on the axis are single poles).
                def vid(key, p2, u):
                    if on_axis(p2):
                        return b.vertex(("pole", key), rot_point(p2, 0.0))
                    return b.vertex(("v", key, round(u % TAU, 12)), rot_point(p2, u))

                a0, b0 = vid(ka, pa, u0), vid(kb, pb, u0)
                a1, b1 = vid(ka, pa, u1), vid(kb, pb, u1)
                mid2 = c.point(0.5)
                e_start = b.edge(
                    ("prof", li, k, round(u0 % TAU, 12)),
                    _rot_curve(c, rf, rh, u0, frame),
                    a0,
                    b0,
                    rot_point(mid2, u0),
                )
                e_end = b.edge(
                    ("prof", li, k, round(u1 % TAU, 12)),
                    _rot_curve(c, rf, rh, u1, frame),
                    a1,
                    b1,
                    rot_point(mid2, u1),
                )
                loop = [e_start]
                if not on_axis(pb):
                    loop.append(_circle_edge(b, rf, rh, kb, pb, b0, b1, u0, u1))
                loop.append((e_end[0], not e_end[1]))
                if not on_axis(pa):
                    ce = _circle_edge(b, rf, rh, ka, pa, a0, a1, u0, u1)
                    loop.append((ce[0], not ce[1]))
                # Outward: rotate the profile's right-of-travel direction.
                um = 0.5 * (u0 + u1)
                t2 = c.tangent(0.5)
                right2 = np.array([t2[1], -t2[0]])
                p_on = rot_point(mid2, um)
                q_on = rot_point(np.asarray(mid2) + 1e-6 * right2 / np.linalg.norm(right2), um)
                outward = (q_on - p_on) / np.linalg.norm(q_on - p_on)
                sense = _sense(surf, p_on, outward)
                if not _loop_ccw(b, loop, outward):
                    loop = [(e, not f) for e, f in reversed(loop)]
                fname = f"{prefix}:rev/{_piece_name(piece)}" + (f".{hi}" if full else "")
                while fname in names:
                    fname += "'"
                names.add(fname)
                window = None
                if surf.periodic_u:
                    window = float(surf.inverse(p_on[None])[0, 0])
                b.face(surf, sense, [loop], fname, group, window)
    if not full:
        for ci, u in enumerate((0.0, angle)):
            cap_loops = []
            for li, lp in enumerate(loops):
                n = len(lp.pieces)
                co = []
                for k in range(n):
                    piece = lp.pieces[k]
                    pa, pb = piece.start, piece.end
                    ka, kb = (li, k), (li, (k + 1) % n)

                    def vid(key, p2, uu=u):
                        if on_axis(p2):
                            return b.vertex(("pole", key), rot_point(p2, 0.0))
                        return b.vertex(("v", key, round(uu % TAU, 12)), rot_point(p2, uu))

                    a_, b_ = vid(ka, pa), vid(kb, pb)
                    co.append(
                        b.edge(
                            ("prof", li, k, round(u % TAU, 12)),
                            _rot_curve(piece.curve, rf, rh, u, frame),
                            a_,
                            b_,
                            rot_point(piece.curve.point(0.5), u),
                        )
                    )
                cap_loops.append(co)
            # The cap plane: the sketch plane rotated by u; outward is -e_u at u=0, +e_u at end.
            e_u = rf.direction_to_world(np.array([-math.sin(u), math.cos(u), 0.0]))
            outward = -e_u if ci == 0 else e_u
            x_axis = rf.direction_to_world(np.array([math.cos(u), math.sin(u), 0.0]))
            plane = Plane(Frame.from_axes(rf.origin, outward, x_axis))
            oriented = []
            for li, co in enumerate(cap_loops):
                want_ccw = li == 0  # the region's outer loop; the others are holes
                ccw = _loop_ccw(b, co, outward)
                oriented.append(co if ccw == want_ccw else [(e, not f) for e, f in reversed(co)])
            b.face(plane, True, oriented, f"{prefix}:cap{ci}", group)
    return b.body()


def _loop_ccw(b: Builder, loop, normal: np.ndarray) -> bool:
    """Whether a loop runs counter-clockwise seen from ``normal`` (Newell's area vector)."""
    pts = []
    for ei, fwd in loop:
        e = b.edges[ei]
        ts = np.linspace(e.t0, e.t1, 17)
        p = e.curve.point(ts)
        pts.append(p if fwd else p[::-1])
    p = np.concatenate(pts)
    area = np.zeros(3)
    q = np.roll(p, -1, axis=0)
    area = np.sum(np.cross(p, q), axis=0) * 0.5
    return bool(float(area @ normal) > 0)


def _circle_edge(b: Builder, rf: Frame, rh, key, p2, va, vb, u0, u1):
    r, hh = rh(p2)
    center = rf.to_world(np.array([0.0, 0.0, hh]))
    circ = Circle3(Frame(center, rf.x, rf.y, rf.z), r)
    um = 0.5 * (u0 + u1)
    pm = rf.to_world(np.array([r * math.cos(um), r * math.sin(um), hh]))
    return b.edge(("circ", key, round(u0, 12), round(u1, 12)), circ, va, vb, pm)


def _rot_curve(c, rf: Frame, rh, u: float, frame: Frame):
    """The profile curve rotated by ``u`` about the axis."""
    rot = np.eye(4)
    cu, su = math.cos(u), math.sin(u)
    r_local = np.array([[cu, -su, 0], [su, cu, 0], [0, 0, 1]])
    rot[:3, :3] = rf.rotation @ r_local @ rf.rotation.T
    rot[:3, 3] = rf.origin - rot[:3, :3] @ rf.origin
    base = _curve3(c, frame, 0.0)
    return base.transformed(rot) if u != 0.0 else base


def _revolved_surface(c, rh, rf: Frame, prefix: str, piece: Piece):
    if isinstance(c, Segment):
        (r0, h0), (r1, h1) = rh(c.p0), rh(c.p1)
        if abs(h1 - h0) <= 1e-15 * max(1.0, abs(h0)):
            center = rf.to_world(np.array([0.0, 0.0, h0]))
            return Plane(Frame(center, rf.x, rf.y, rf.z))
        if abs(r1 - r0) <= 1e-15 * max(1.0, r0):
            return Cylinder(Frame(rf.origin, rf.x, rf.y, rf.z), r0)
        slope = (r1 - r0) / (h1 - h0)
        center = rf.to_world(np.array([0.0, 0.0, h0]))
        if slope > 0:
            return Cone(Frame(center, rf.x, rf.y, rf.z), r0, math.atan(slope))
        fr = Frame.from_axes(center, -rf.z, rf.x)
        return Cone(fr, r0, math.atan(-slope))
    if isinstance(c, Arc):
        rc, hc = rh(c.center)
        center = rf.to_world(np.array([0.0, 0.0, hc]))
        if rc <= 1e-12 * max(1.0, c.radius):
            return Sphere(Frame(center, rf.x, rf.y, rf.z), c.radius)
        if rc <= c.radius:
            raise CadError(
                "UNSUPPORTED_GEOMETRY",
                f"revolving arc {piece.source!r} would make a self-intersecting torus",
                prefix,
            )
        return Torus(Frame(center, rf.x, rf.y, rf.z), rc, c.radius)
    return Revolution(rf, _profile_nurbs(c, rf, rh))


def _profile_nurbs(c: Spline, rf: Frame, rh) -> Nurbs3:
    cv = c.curve
    rhs = np.array([rh(p) for p in cv.points])
    pts = rf.to_world(np.c_[rhs[:, 0], np.zeros(len(rhs)), rhs[:, 1]])
    return Nurbs3(BSplineCurve(cv.degree, cv.knots, pts, cv.weights))


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def rectangle_region(x0: float, y0: float, x1: float, y1: float, name: str = "rect") -> Region:
    c = [np.array(p, float) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    names = ("bottom", "right", "top", "left")
    return Region(
        Loop([Piece(Segment(c[i], c[(i + 1) % 4]), f"{name}.{names[i]}", 0) for i in range(4)])
    )


def circle_region(cx: float, cy: float, r: float, name: str = "circle") -> Region:
    return Region(Loop([Piece(Arc(np.array([cx, cy]), r, 0.0, TAU), name, 0)]))


def box(size, *, frame: Frame | None = None, prefix: str = "box", group: int = 0) -> Body:
    sx, sy, sz = (float(v) for v in size)
    if min(sx, sy, sz) <= 0:
        raise CadError("INVALID_SIZE", "box sizes must be positive", prefix)
    frame = frame or Frame.world()
    return extrude(
        rectangle_region(-sx / 2, -sy / 2, sx / 2, sy / 2, "face"),
        frame,
        -sz / 2,
        sz / 2,
        prefix=prefix,
        group=group,
    )


def cylinder(
    radius: float, height: float, *, frame: Frame | None = None, prefix="cylinder", group=0
) -> Body:
    if radius <= 0 or height <= 0:
        raise CadError("INVALID_SIZE", "radius and height must be positive", prefix)
    frame = frame or Frame.world()
    return extrude(
        circle_region(0, 0, radius, "wall"),
        frame,
        -height / 2,
        height / 2,
        prefix=prefix,
        group=group,
    )


def sphere(radius: float, *, frame: Frame | None = None, prefix="sphere", group=0) -> Body:
    if radius <= 0:
        raise CadError("INVALID_SIZE", "the radius must be positive", prefix)
    frame = frame or Frame.world()
    # Half disk in the sketch's xy plane, revolved about the sketch y axis.
    arc = Arc(np.zeros(2), radius, -math.pi / 2, math.pi)
    seg = Segment(np.array([0.0, radius]), np.array([0.0, -radius]))
    region = Region(Loop([Piece(arc, "surface", 0), Piece(seg, "axis", 0)]))
    sk = Frame(frame.origin, frame.x, frame.z, -frame.y)  # the sketch plane contains z
    return revolve(region, sk, (0.0, 0.0), (0.0, 1.0), prefix=prefix, group=group)


def cone(
    radius0: float, radius1: float, height: float, *, frame=None, prefix="cone", group=0
) -> Body:
    if radius0 < 0 or radius1 < 0 or max(radius0, radius1) <= 0 or height <= 0:
        raise CadError("INVALID_SIZE", "radii >= 0 (one positive) and height > 0", prefix)
    frame = frame or Frame.world()
    pts = [(0.0, 0.0), (radius0, 0.0), (radius1, height), (0.0, height)]
    pieces = []
    names = ("base", "side", "top", "axis")
    for i in range(4):
        a, b_ = np.array(pts[i]), np.array(pts[(i + 1) % 4])
        if np.linalg.norm(b_ - a) > 0:
            pieces.append(Piece(Segment(a, b_), names[i], 0))
    sk = Frame(frame.origin, frame.x, frame.z, -frame.y)
    return revolve(Region(Loop(pieces)), sk, (0.0, 0.0), (0.0, 1.0), prefix=prefix, group=group)


def torus(major: float, minor: float, *, frame=None, prefix="torus", group=0) -> Body:
    if not major > minor > 0:
        raise CadError("INVALID_SIZE", "a torus needs major > minor > 0", prefix)
    frame = frame or Frame.world()
    sk = Frame(frame.origin, frame.x, frame.z, -frame.y)
    return revolve(
        circle_region(major, 0.0, minor, "surface"),
        sk,
        (0.0, 0.0),
        (0.0, 1.0),
        prefix=prefix,
        group=group,
    )
