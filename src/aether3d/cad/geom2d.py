"""Analytic 2D curves: segments, circular arcs, circles and B-splines.

Used by sketches (profiles, trimming, snapping, offsets) and drawings. Every curve is
parameterized on ``t`` in ``[0, 1]`` (a circle's ``t`` is the fraction of a turn from its
start angle) and offers points, tangents, length, closest points, splitting, reversal,
offsetting and adaptive sampling. Arcs keep a signed sweep: positive counter-clockwise.

Intersections are computed in closed form for lines and circles; tangencies are
recognised within the tolerance policy's ``resolution`` (a tangent line meets a circle
in one point, reported with ``tangent=True``) instead of producing two nearly equal
points or none depending on rounding. Splines are intersected by recursive subdivision
of their control polygons followed by Newton refinement on both curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import CadError
from .nurbs import BSplineCurve

TAU = 2.0 * math.pi
_GX, _GW = np.polynomial.legendre.leggauss(10)


def integrate(f, a: float, b: float, rtol: float = 1e-12, depth: int = 24) -> float:
    """Adaptive Gauss-Legendre quadrature of a vectorized ``f`` over ``[a, b]``: halves are
    refined until two levels agree to ``rtol`` (or ``depth`` levels)."""

    def gauss(lo: float, hi: float) -> float:
        u = 0.5 * (hi - lo) * _GX + 0.5 * (hi + lo)
        return 0.5 * (hi - lo) * float(np.sum(_GW * f(u)))

    total = 0.0
    stack = [(a, b, gauss(a, b), 0)]
    budget = 1 << 14
    while stack:
        lo, hi, whole, level = stack.pop()
        mid = 0.5 * (lo + hi)
        left, right = gauss(lo, mid), gauss(mid, hi)
        budget -= 1
        if (
            level >= depth
            or budget <= 0
            or abs(left + right - whole) <= rtol * max(abs(left + right), 1e-300)
        ):
            total += left + right
        else:
            stack.append((lo, mid, left, level + 1))
            stack.append((mid, hi, right, level + 1))
    return total


def _v(p) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64).reshape(2)
    return a


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def norm(a: np.ndarray) -> float:
    return float(math.hypot(a[0], a[1]))


def angle_of(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


@dataclass(frozen=True)
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    kind: str = field(default="segment", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "p0", _v(self.p0))
        object.__setattr__(self, "p1", _v(self.p1))

    def point(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.p0 + np.multiply.outer(t, self.p1 - self.p0)

    def tangent(self, t) -> np.ndarray:
        d = self.p1 - self.p0
        return np.broadcast_to(d, (*np.shape(t), 2)).copy()

    def curvature(self, t) -> np.ndarray:
        return np.zeros(np.shape(t))

    @property
    def length(self) -> float:
        return norm(self.p1 - self.p0)

    @property
    def start(self) -> np.ndarray:
        return self.p0

    @property
    def end(self) -> np.ndarray:
        return self.p1

    def reversed(self) -> Segment:
        return Segment(self.p1, self.p0)

    def sub(self, t0: float, t1: float) -> Segment:
        return Segment(self.point(t0), self.point(t1))

    def closest(self, p) -> tuple[float, np.ndarray]:
        p = _v(p)
        d = self.p1 - self.p0
        dd = float(d @ d)
        t = 0.0 if dd == 0 else min(1.0, max(0.0, float((p - self.p0) @ d) / dd))
        return t, self.point(t)

    def offset(self, distance: float) -> Segment:
        """Offset to the left (positive) of the direction of travel."""
        d = self.p1 - self.p0
        n = np.array([-d[1], d[0]]) / norm(d)
        return Segment(self.p0 + distance * n, self.p1 + distance * n)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return np.minimum(self.p0, self.p1), np.maximum(self.p0, self.p1)

    def sample(self, chord: float, angle: float = 0.26) -> np.ndarray:
        return np.array([0.0, 1.0])

    def signed_area_term(self) -> float:
        """Contribution to the enclosed signed area (Green's theorem: 1/2 x dy - y dx)."""
        return 0.5 * cross(self.p0, self.p1)


@dataclass(frozen=True)
class Arc:
    """A circular arc: center, radius, start angle and signed sweep (radians).

    A full circle has ``|sweep| == 2 pi``.
    """

    center: np.ndarray
    radius: float
    start_angle: float
    sweep: float
    kind: str = field(default="arc", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _v(self.center))
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise CadError("DEGENERATE_GEOMETRY", "an arc's radius must be positive")
        if not (0 < abs(self.sweep) <= TAU + 1e-12):
            raise CadError("DEGENERATE_GEOMETRY", "an arc's sweep must be in (0, 360] degrees")

    @property
    def closed(self) -> bool:
        return abs(abs(self.sweep) - TAU) <= 1e-12

    def angle(self, t) -> np.ndarray:
        return self.start_angle + np.asarray(t, dtype=np.float64) * self.sweep

    def point(self, t) -> np.ndarray:
        a = self.angle(t)
        return self.center + self.radius * np.stack([np.cos(a), np.sin(a)], axis=-1)

    def tangent(self, t) -> np.ndarray:
        a = self.angle(t)
        s = self.sweep * self.radius
        return s * np.stack([-np.sin(a), np.cos(a)], axis=-1)

    def curvature(self, t) -> np.ndarray:
        """Signed curvature (positive when turning left)."""
        return np.full(np.shape(t), math.copysign(1.0 / self.radius, self.sweep))

    @property
    def length(self) -> float:
        return abs(self.sweep) * self.radius

    @property
    def start(self) -> np.ndarray:
        return self.point(0.0)

    @property
    def end(self) -> np.ndarray:
        return self.point(1.0)

    @property
    def mid(self) -> np.ndarray:
        return self.point(0.5)

    def reversed(self) -> Arc:
        return Arc(self.center, self.radius, self.start_angle + self.sweep, -self.sweep)

    def sub(self, t0: float, t1: float) -> Arc:
        return Arc(self.center, self.radius, float(self.angle(t0)), (t1 - t0) * self.sweep)

    def param_of_angle(self, a: float) -> float:
        """``t`` of the point at angle ``a`` (may be outside [0, 1] if not on the arc)."""
        d = (a - self.start_angle) * math.copysign(1.0, self.sweep)
        d = math.fmod(d, TAU)
        if d < 0:
            d += TAU
        return d / abs(self.sweep)

    def closest(self, p) -> tuple[float, np.ndarray]:
        p = _v(p)
        v = p - self.center
        if norm(v) == 0:
            return 0.0, self.start
        t = self.param_of_angle(angle_of(v))
        if t <= 1.0:
            return t, self.point(t)
        # Outside the arc: the nearer end (t past 1 wraps toward the start).
        d0 = norm(self.start - p)
        d1 = norm(self.end - p)
        return (0.0, self.start) if d0 <= d1 else (1.0, self.end)

    def offset(self, distance: float) -> Arc:
        """Offset to the left of the direction of travel (inward for counter-clockwise)."""
        r = self.radius - math.copysign(distance, self.sweep)
        if r <= 0:
            raise CadError(
                "OFFSET_COLLAPSE",
                f"offsetting an arc of radius {self.radius:g} m by {distance:g} m collapses it",
            )
        return Arc(self.center, r, self.start_angle, self.sweep)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        pts = [self.start, self.end]
        for k in range(4):
            a = k * math.pi / 2
            if self.param_of_angle(a) <= 1.0:
                pts.append(self.center + self.radius * np.array([math.cos(a), math.sin(a)]))
        pts = np.array(pts)
        return pts.min(axis=0), pts.max(axis=0)

    def segments_for(self, chord: float, angle: float = 0.26) -> int:
        """Segments needed so the polyline is within ``chord`` and ``angle`` of the arc."""
        sweep = abs(self.sweep)
        n_angle = math.ceil(sweep / max(angle, 1e-3))
        if chord >= self.radius:
            n_chord = 1
        else:
            n_chord = math.ceil(sweep / (2.0 * math.acos(1.0 - chord / self.radius)))
        return int(min(max(n_angle, n_chord, 1 if not self.closed else 3), 1 << 16))

    def sample(self, chord: float, angle: float = 0.26) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.segments_for(chord, angle) + 1)

    def signed_area_term(self) -> float:
        # 1/2 * integral (x dy - y dx) along the arc, exactly.
        c, r, a0, s = self.center, self.radius, self.start_angle, self.sweep
        a1 = a0 + s
        term = r * r * s + r * (
            c[0] * (math.sin(a1) - math.sin(a0)) - c[1] * (math.cos(a1) - math.cos(a0))
        )
        return 0.5 * float(term)


@dataclass(frozen=True)
class Spline:
    """A planar B-spline curve (``curve`` has 2D control points), on t in [0, 1]."""

    curve: BSplineCurve
    kind: str = field(default="spline", init=False)

    def __post_init__(self) -> None:
        if self.curve.dim != 2:
            raise CadError("INVALID_SPLINE", "a sketch spline has 2D control points")

    def _u(self, t) -> np.ndarray:
        lo, hi = self.curve.domain
        return lo + np.asarray(t, dtype=np.float64) * (hi - lo)

    def point(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        out = self.curve.evaluate(np.atleast_1d(self._u(t)))
        return out.reshape(*t.shape, 2)

    def tangent(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        lo, hi = self.curve.domain
        out = self.curve.derivatives(np.atleast_1d(self._u(t)), 1)[1] * (hi - lo)
        return out.reshape(*t.shape, 2)

    def curvature(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        d = self.curve.derivatives(self._u(t), 2)
        c1, c2 = d[1], d[2]
        num = c1[:, 0] * c2[:, 1] - c1[:, 1] * c2[:, 0]
        den = np.maximum(np.hypot(c1[:, 0], c1[:, 1]) ** 3, 1e-300)
        return num / den

    @property
    def start(self) -> np.ndarray:
        return self.point(0.0)

    @property
    def end(self) -> np.ndarray:
        return self.point(1.0)

    @property
    def length(self) -> float:
        lo, hi = self.curve.domain
        knots = np.unique(self.curve.knots[(self.curve.knots >= lo) & (self.curve.knots <= hi)])

        def speed(u):
            d = self.curve.derivatives(u, 1)[1]
            return np.hypot(d[:, 0], d[:, 1])

        return sum(integrate(speed, a, b) for a, b in zip(knots[:-1], knots[1:], strict=True))

    def reversed(self) -> Spline:
        c = self.curve
        lo, hi = c.domain
        knots = (lo + hi) - c.knots[::-1]
        weights = None if c.weights is None else c.weights[::-1]
        return Spline(BSplineCurve(c.degree, knots, c.points[::-1], weights))

    def sub(self, t0: float, t1: float) -> Spline:
        """The piece between ``t0 < t1`` (by knot insertion to full multiplicity)."""
        if t1 < t0:
            return self.sub(t1, t0).reversed()
        c = self.curve
        p = c.degree
        lo, hi = c.domain
        for t in (t0, t1):
            u = lo + t * (hi - lo)
            if lo < u < hi:
                mult = int(np.sum(c.knots == u))
                if mult < p:
                    c = c.insert_knot(u, p - mult)
        u0, u1 = lo + t0 * (hi - lo), lo + t1 * (hi - lo)
        k = c.knots
        i0 = 0 if u0 <= lo else int(np.searchsorted(k, u0, side="left")) - 1
        i1 = len(c.points) - 1 if u1 >= hi else int(np.searchsorted(k, u1, side="left")) - 1
        pts = c.points[i0 : i1 + 1]
        w = None if c.weights is None else c.weights[i0 : i1 + 1]
        inner = k[(k > u0) & (k < u1)]
        knots = np.concatenate([[u0] * (p + 1), inner, [u1] * (p + 1)])
        return Spline(BSplineCurve(p, knots, pts, w))

    def closest(self, p) -> tuple[float, np.ndarray]:
        u, pt = self.curve.closest(_v(p))
        lo, hi = self.curve.domain
        return (u - lo) / (hi - lo), pt

    def offset(self, distance: float):
        raise CadError("UNSUPPORTED_GEOMETRY", "splines cannot be offset exactly")

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        # Convex hull property: the control polygon bounds the curve.
        return self.curve.points.min(axis=0), self.curve.points.max(axis=0)

    def sample(self, chord: float, angle: float = 0.26) -> np.ndarray:
        """Parameters of a polyline within ``chord`` of the curve (recursive bisection)."""
        ts = list(np.linspace(0.0, 1.0, 4 * (len(self.curve.points) - 1) + 1))
        out = [ts[0]]
        stack = list(reversed(list(zip(ts[:-1], ts[1:], strict=True))))
        while stack and len(out) < 1 << 16:
            a, b = stack.pop()
            m = 0.5 * (a + b)
            pa, pb, pm = self.point(np.array([a, m, b]))
            dev = abs(cross(pb - pa, pm - pa)) / max(norm(pb - pa), 1e-300)
            ta, tb = self.tangent(np.array([a, b]))
            turn = abs(math.atan2(cross(ta, tb), float(ta @ tb)))
            if (dev > chord or turn > angle) and b - a > 1e-9:
                stack.append((m, b))
                stack.append((a, m))
            else:
                out.append(b)
        return np.array(out)

    def signed_area_term(self) -> float:
        lo, hi = self.curve.domain
        knots = np.unique(self.curve.knots[(self.curve.knots >= lo) & (self.curve.knots <= hi)])

        def term(u):
            d = self.curve.derivatives(u, 1)
            return d[0][:, 0] * d[1][:, 1] - d[0][:, 1] * d[1][:, 0]

        return 0.5 * sum(integrate(term, a, b) for a, b in zip(knots[:-1], knots[1:], strict=True))


Curve2D = Segment | Arc | Spline


def arc_from_points(start, mid, end) -> Arc:
    """The arc from ``start`` through ``mid`` to ``end``."""
    a, b, c = _v(start), _v(mid), _v(end)
    d = 2.0 * cross(b - a, c - a)
    if d == 0:
        raise CadError("DEGENERATE_GEOMETRY", "the three points are collinear")
    ab, ac = float((b - a) @ (b - a)), float((c - a) @ (c - a))
    ux = ((c - a)[1] * ab - (b - a)[1] * ac) / d
    uy = ((b - a)[0] * ac - (c - a)[0] * ab) / d
    center = a + np.array([ux, uy])
    r = norm(a - center)
    a0, a1 = angle_of(a - center), angle_of(c - center)
    ccw = cross(b - a, c - a) > 0
    sweep = (a1 - a0) % TAU if ccw else -((a0 - a1) % TAU)
    if sweep == 0:
        sweep = TAU if ccw else -TAU
    return Arc(center, r, a0, sweep)


def arc_from_center(center, start, end, ccw: bool = True) -> Arc:
    """The arc about ``center`` from ``start`` to the angle of ``end`` (radius of start)."""
    c, s, e = _v(center), _v(start), _v(end)
    r = norm(s - c)
    if r == 0:
        raise CadError("DEGENERATE_GEOMETRY", "an arc's start coincides with its center")
    a0, a1 = angle_of(s - c), angle_of(e - c)
    sweep = (a1 - a0) % TAU if ccw else -((a0 - a1) % TAU)
    if sweep == 0:
        sweep = TAU if ccw else -TAU
    return Arc(c, r, a0, sweep)


# --------------------------------------------------------------------------- #
# Intersections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Hit:
    """An intersection: parameters on both curves, the point, and whether it is a tangency
    or the start of a collinear/co-circular overlap (``overlap`` = (t_a_end, t_b_end))."""

    ta: float
    tb: float
    point: np.ndarray
    tangent: bool = False
    overlap: tuple[float, float] | None = None


def _line_params(p, d, q, e):
    den = cross(d, e)
    w = q - p
    return den, cross(w, e), cross(w, d)


def intersect(a, b, tol: float, *, extend: bool = False) -> list[Hit]:
    """Intersections of two curves (within ``tol`` metres for tangency and endpoints).

    With ``extend``, segments are treated as infinite lines and arcs as full circles
    (parameters may fall outside [0, 1]): used for trimming and extending.
    """
    if a.kind == "segment" and b.kind == "segment":
        return _seg_seg(a, b, tol, extend)
    if a.kind == "segment" and b.kind == "arc":
        return _seg_arc(a, b, tol, extend)
    if a.kind == "arc" and b.kind == "segment":
        return [Hit(h.tb, h.ta, h.point, h.tangent) for h in _seg_arc(b, a, tol, extend)]
    if a.kind == "arc" and b.kind == "arc":
        return _arc_arc(a, b, tol, extend)
    return _generic(a, b, tol)


def _within(t: float, length: float, tol: float, extend: bool) -> bool:
    if extend:
        return True
    slack = tol / max(length, 1e-300)
    return -slack <= t <= 1.0 + slack


def _clamp01(t: float) -> float:
    return min(1.0, max(0.0, t))


def _seg_seg(a: Segment, b: Segment, tol: float, extend: bool) -> list[Hit]:
    d, e = a.p1 - a.p0, b.p1 - b.p0
    la, lb = norm(d), norm(e)
    if la == 0 or lb == 0:
        return []
    den, na, nb = _line_params(a.p0, d, b.p0, e)
    # Parallel when the lines' directions differ by less than the resolution over length.
    if abs(den) <= tol * max(la, lb) / max(la * lb, 1e-300) * la * lb / max(la, lb):
        # Collinear? distance of b.p0 from line a.
        dist = abs(cross(b.p0 - a.p0, d)) / la
        if dist > tol:
            return []
        t0 = float((b.p0 - a.p0) @ d) / (la * la)
        t1 = float((b.p1 - a.p0) @ d) / (la * la)
        lo, hi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
        if extend or hi < lo - tol / la:
            return []
        if (hi - lo) * la <= tol:  # touching at one point
            t = 0.5 * (lo + hi)
            p = a.point(t)
            return [Hit(t, _clamp01(b.closest(p)[0]), p, True)]
        pa, pb = a.point(lo), a.point(hi)
        return [
            Hit(lo, b.closest(pa)[0], pa, False, (hi, b.closest(pb)[0])),
        ]
    ta, tb = na / den, nb / den
    if not (_within(ta, la, tol, extend) and _within(tb, lb, tol, extend)):
        return []
    if not extend:
        ta, tb = _clamp01(ta), _clamp01(tb)
    return [Hit(ta, tb, a.point(ta))]


def _seg_arc(s: Segment, c: Arc, tol: float, extend: bool) -> list[Hit]:
    d = s.p1 - s.p0
    ls = norm(d)
    if ls == 0:
        return []
    u = d / ls
    w = s.p0 - c.center
    # Foot of the perpendicular from the center.
    tf = -float(w @ u)
    foot = s.p0 + tf * u
    h = norm(foot - c.center)
    r = c.radius
    out: list[Hit] = []
    if abs(h - r) <= tol:  # tangent
        cands = [(tf, True)]
    elif h > r:
        return []
    else:
        k = math.sqrt(max(r * r - h * h, 0.0))
        cands = [(tf - k, False), (tf + k, False)]
    for dist, tangent in cands:
        ta = dist / ls
        p = s.p0 + dist * u
        if tangent:
            # The exact tangent point lies on the circle.
            v = p - c.center
            n = norm(v)
            if n > 0:
                p = c.center + v * (r / n)
        tb = c.param_of_angle(angle_of(p - c.center))
        if not extend:
            if not _within(ta, ls, tol, False):
                continue
            if tb > 1.0:
                # Near the start of the arc (wrapped)?
                if (TAU - tb * abs(c.sweep)) * r <= tol:
                    tb = 0.0
                elif (tb - 1.0) * abs(c.sweep) * r <= tol:
                    tb = 1.0
                else:
                    continue
            ta = _clamp01(ta)
        out.append(Hit(ta, tb, p, tangent))
    return out


def _arc_arc(a: Arc, b: Arc, tol: float, extend: bool) -> list[Hit]:
    v = b.center - a.center
    dist = norm(v)
    ra, rb = a.radius, b.radius
    if dist <= tol and abs(ra - rb) <= tol:
        if extend:
            return []
        # Co-circular arcs: report the overlap of their angular ranges by endpoints.
        hits = []
        for p in (b.start, b.end):
            t = a.param_of_angle(angle_of(p - a.center))
            if t <= 1.0 + tol / a.length:
                hits.append(Hit(_clamp01(t), b.closest(p)[0], p, True))
        for p in (a.start, a.end):
            t = b.param_of_angle(angle_of(p - b.center))
            if t <= 1.0 + tol / b.length:
                hits.append(Hit(a.closest(p)[0], _clamp01(t), p, True))
        return _dedupe(hits, tol)
    if dist <= tol:
        return []
    if dist > ra + rb + tol or dist < abs(ra - rb) - tol:
        return []
    tangent = abs(dist - (ra + rb)) <= tol or abs(dist - abs(ra - rb)) <= tol
    u = v / dist
    x = (dist * dist + ra * ra - rb * rb) / (2 * dist)
    if tangent:
        pts = [a.center + u * math.copysign(ra, x)]
    else:
        y = math.sqrt(max(ra * ra - x * x, 0.0))
        n = np.array([-u[1], u[0]])
        pts = [a.center + u * x + n * y, a.center + u * x - n * y]
    out = []
    for p in pts:
        ta = a.param_of_angle(angle_of(p - a.center))
        tb = b.param_of_angle(angle_of(p - b.center))
        if not extend:
            ta = _snap_arc_t(a, ta, tol)
            tb = _snap_arc_t(b, tb, tol)
            if ta is None or tb is None:
                continue
        out.append(Hit(ta, tb, p, tangent))
    return out


def _snap_arc_t(c: Arc, t: float, tol: float) -> float | None:
    if t <= 1.0:
        return t
    if (t - 1.0) * abs(c.sweep) * c.radius <= tol:
        return 1.0
    if (TAU - t * abs(c.sweep)) * c.radius <= tol:
        return 0.0
    return None


def _dedupe(hits: list[Hit], tol: float) -> list[Hit]:
    out: list[Hit] = []
    for h in hits:
        if all(norm(h.point - o.point) > tol for o in out):
            out.append(h)
    return out


def _generic(a, b, tol: float, depth: int = 0) -> list[Hit]:
    """Intersections involving splines: subdivision of parameter boxes, then Newton."""
    found: list[Hit] = []
    stack = [(0.0, 1.0, 0.0, 1.0, 0)]
    while stack:
        a0, a1, b0, b1, level = stack.pop()
        pa = a.point(np.linspace(a0, a1, 9))
        pb = b.point(np.linspace(b0, b1, 9))
        # Bounding boxes of the samples, grown by the chord deviation estimate.
        grow = tol + max(_chord_bound(pa), _chord_bound(pb))
        if np.any(pa.min(0) - grow > pb.max(0)) or np.any(pb.min(0) - grow > pa.max(0)):
            continue
        if (
            level >= 40
            or (a1 - a0) * _len_est(pa) + (b1 - b0) * _len_est(pb) < tol * 4
            or (level >= 6 and _flat(pa, tol) and _flat(pb, tol))
        ):
            hit = _newton(a, b, 0.5 * (a0 + a1), 0.5 * (b0 + b1), tol)
            if hit is not None and all(norm(hit.point - h.point) > tol for h in found):
                found.append(hit)
            continue
        am, bm = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
        stack += [
            (a0, am, b0, bm, level + 1),
            (a0, am, bm, b1, level + 1),
            (am, a1, b0, bm, level + 1),
            (am, a1, bm, b1, level + 1),
        ]
        if len(stack) > 20000:
            raise CadError("INTERSECTION_FAILED", "spline intersection did not converge")
    return sorted(found, key=lambda h: h.ta)


def _chord_bound(p: np.ndarray) -> float:
    d = p[-1] - p[0]
    n = norm(d)
    if n == 0:
        return float(np.max(np.linalg.norm(p - p[0], axis=1)))
    off = np.abs((p[:, 0] - p[0, 0]) * d[1] - (p[:, 1] - p[0, 1]) * d[0]) / n
    return float(off.max())


def _len_est(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _flat(p: np.ndarray, tol: float) -> bool:
    return _chord_bound(p) <= tol


def _newton(a, b, ta: float, tb: float, tol: float) -> Hit | None:
    for _ in range(50):
        pa, pb = a.point(ta), b.point(tb)
        da, db = a.tangent(ta), b.tangent(tb)
        f = pa - pb
        jac = np.array([[da[0], -db[0]], [da[1], -db[1]]])
        det = float(np.linalg.det(jac))
        if abs(det) < 1e-300:
            break
        step = np.linalg.solve(jac, f)
        ta, tb = ta - float(step[0]), tb - float(step[1])
        if norm(f) < tol * 1e-3:
            break
    pa, pb = a.point(ta), b.point(tb)
    if norm(pa - pb) > tol:
        return None
    slack_a, slack_b = tol / max(a.length, 1e-300), tol / max(b.length, 1e-300)
    if not (-slack_a <= ta <= 1 + slack_a and -slack_b <= tb <= 1 + slack_b):
        return None
    ta, tb = _clamp01(ta), _clamp01(tb)
    da, db = a.tangent(ta), b.tangent(tb)
    sin = abs(cross(da, db)) / max(norm(da) * norm(db), 1e-300)
    return Hit(ta, tb, 0.5 * (pa + pb), sin < 1e-9)


def polyline(curve, chord: float, angle: float = 0.26) -> np.ndarray:
    """Points of a polyline within ``chord`` of the curve (both ends included)."""
    return curve.point(curve.sample(chord, angle))
