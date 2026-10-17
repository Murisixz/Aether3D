"""Intersection curves of surface pairs, for recovering boolean edges exactly.

``recover_curve(s1, s2, points, tol)`` returns the analytic curve both surfaces share
near the given polyline -- or ``None`` when the pair has no closed form here:

=====================  =====================================================
pair                   curve
=====================  =====================================================
plane - plane          line
plane - cylinder       circle (plane across the axis), line (along it), ellipse
plane - cone           circle (across the axis), lines (through the apex), ellipse
plane - sphere         circle
plane - torus          circle (a parallel: plane across the axis)
coaxial quadrics       circle (parallels of surfaces of revolution sharing an axis)
=====================  =====================================================

A candidate is accepted only if every polyline point lies within ``tol`` of it (the
polyline is the boolean's approximation of the true intersection, so ``tol`` is the
boolean tessellation's chord plus margin). Otherwise the caller traces the intersection
numerically (``trace``): every point is pulled onto both surfaces by Newton steps, and a
cubic B-spline interpolates them; the largest distance of the spline from both surfaces,
measured between the points, becomes the edge's tolerance.
"""

from __future__ import annotations

import math

import numpy as np

from ..geom3d import Circle3, Cone, Cylinder, Ellipse3, Frame, Line3, Nurbs3, Plane, Sphere
from ..nurbs import interpolate

_AXIAL = ("cylinder", "cone", "sphere", "torus", "revolution")


def _axis(s):
    return s.frame.origin, s.frame.z


def _fits(curve, points: np.ndarray, tol: float) -> bool:
    t = curve.inverse(points)
    q = curve.point(t)
    return bool(np.max(np.linalg.norm(q - points, axis=1)) <= tol)


def _plane_plane(p1: Plane, p2: Plane, pts: np.ndarray):
    n1, n2 = p1.frame.z, p2.frame.z
    d = np.cross(n1, n2)
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
        return None
    d /= nd
    # Point on both planes closest to the polyline's center.
    c = pts.mean(axis=0)
    a = np.array([n1, n2, d])
    b = np.array([n1 @ p1.frame.origin, n2 @ p2.frame.origin, d @ c])
    origin = np.linalg.solve(a, b)
    return [Line3(origin, d)]


def _plane_cylinder(p: Plane, cyl: Cylinder, pts: np.ndarray):
    n = p.frame.z
    o, a = _axis(cyl)
    r = cyl.radius
    na = float(n @ a)
    if abs(abs(na) - 1.0) < 1e-12:
        t = float(n @ (p.frame.origin - o)) / na
        return [Circle3(Frame.from_axes(o + t * a, a, cyl.frame.x), r)]
    if abs(na) < 1e-12:
        # Lines along the axis where the plane cuts the circle.
        dist = float(n @ (o - p.frame.origin))
        if abs(dist) > r * (1.0 + 1e-9):
            return None  # (a plane tangent to the cylinder gives one line: k = 0 below)
        foot = o - dist * n
        side = np.cross(a, n)
        k = math.sqrt(max(r * r - dist * dist, 0.0))
        return [Line3(foot + k * side, a), Line3(foot - k * side, a)]
    t = float(n @ (p.frame.origin - o)) / na
    center = o + t * a
    major_dir = a - na * n
    major_dir /= np.linalg.norm(major_dir)
    return [Ellipse3(Frame.from_axes(center, n, major_dir), r / abs(na), r)]


def _plane_cone(p: Plane, cone: Cone, pts: np.ndarray):
    n = p.frame.z
    o, a = _axis(cone)
    na = float(n @ a)
    if abs(abs(na) - 1.0) < 1e-12:
        v = float(n @ (p.frame.origin - o)) / na
        r = cone.radius + v * cone.slope
        if r <= 0:
            return None
        return [Circle3(Frame.from_axes(o + v * a, a, cone.frame.x), r)]
    apex = o + cone.apex_v * a
    if abs(float(n @ (apex - p.frame.origin))) < 1e-12 * max(1.0, cone.radius):
        # Plane through the apex: generator lines.
        out = []
        loc = cone.frame.to_local(pts)
        us = np.unique(np.round(np.arctan2(loc[:, 1], loc[:, 0]), 9))
        for u in us[:2]:
            d = cone.point(np.array([u]), np.array([1.0]))[0] - apex
            out.append(Line3(apex, d))
        return out or None
    # Ellipse when the plane is steeper than the generators (cuts every generator).
    half = abs(cone.half_angle)
    tilt = math.acos(min(1.0, abs(na)))  # angle between plane normal and axis
    if tilt >= math.pi / 2 - half - 1e-12:
        return None  # parabola or hyperbola: traced
    # Fit the ellipse's frame from the section: center and axes by the points (exact
    # conic parameters from the polyline would be approximate; derive them analytically).
    # Section of x^2 + y^2 = (r0 + z tan h)^2 by the plane; work in the cone frame.
    loc_o = cone.frame.to_local(p.frame.origin)
    ln = n @ cone.frame.rotation  # plane normal in the cone frame
    # Parameterize plane points as P = q + s e1 + t e2 (e1 along the steepest direction).
    e1 = np.array([0.0, 0.0, 1.0]) - ln[2] * ln
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ln, e1)
    q = loc_o  # a point of the plane (cone frame)
    k = cone.slope
    r0 = cone.radius

    # Quadratic form in (s, t): (q + s e1 + t e2)_xy^2 - (r0 + k (q + s e1 + t e2)_z)^2 = 0
    def comps(v):
        return np.array([v[0], v[1]]), v[2]

    qxy, qz = comps(q)
    e1xy, e1z = comps(e1)
    e2xy, e2z = comps(e2)
    big_a = e1xy @ e1xy - k * k * e1z * e1z
    big_c = e2xy @ e2xy - k * k * e2z * e2z
    big_b = 2 * (e1xy @ e2xy - k * k * e1z * e2z)
    big_d = 2 * (qxy @ e1xy - k * e1z * (r0 + k * qz))
    big_e = 2 * (qxy @ e2xy - k * e2z * (r0 + k * qz))
    big_f = qxy @ qxy - (r0 + k * qz) ** 2
    if abs(big_b) > 1e-9 * (abs(big_a) + abs(big_c)):
        return None
    if big_a <= 0 or big_c <= 0:
        return None
    s0, t0 = -big_d / (2 * big_a), -big_e / (2 * big_c)
    rhs = big_d * big_d / (4 * big_a) + big_e * big_e / (4 * big_c) - big_f
    if rhs <= 0:
        return None
    ra, rb = math.sqrt(rhs / big_a), math.sqrt(rhs / big_c)
    center_loc = q + s0 * e1 + t0 * e2
    center = cone.frame.to_world(center_loc)
    ax1 = cone.frame.direction_to_world(e1)
    ax2 = cone.frame.direction_to_world(e2)
    if ra >= rb:
        return [Ellipse3(Frame.from_axes(center, n, ax1), ra, rb)]
    return [Ellipse3(Frame.from_axes(center, n, ax2), rb, ra)]


def _plane_sphere(p: Plane, sph: Sphere, pts: np.ndarray):
    n = p.frame.z
    o = sph.frame.origin
    d = float(n @ (o - p.frame.origin))
    if abs(d) >= sph.radius:
        return None
    center = o - d * n
    return [Circle3(Frame.from_axes(center, n, None), math.sqrt(sph.radius**2 - d * d))]


def _coaxial(s1, s2, pts: np.ndarray, tol: float):
    if s1.kind not in _AXIAL or s2.kind not in _AXIAL:
        return None
    o1, a1 = _axis(s1)
    o2, a2 = _axis(s2)
    if abs(abs(float(a1 @ a2)) - 1.0) > 1e-12:
        return None
    if float(np.linalg.norm(np.cross(o2 - o1, a1))) > tol:
        return None
    loc = s1.frame.to_local(pts)
    h = float(np.mean(loc[:, 2]))
    r = float(np.mean(np.hypot(loc[:, 0], loc[:, 1])))
    if r <= 0:
        return None
    return [Circle3(Frame.from_axes(s1.frame.to_world(np.array([0, 0, h])), a1, s1.frame.x), r)]


def recover_curve(s1, s2, pts: np.ndarray, tol: float):
    """The analytic curve of ``s1`` and ``s2`` through the polyline ``pts``, or None."""
    k1, k2 = s1.kind, s2.kind
    cands = None
    if k1 == "plane" and k2 == "plane":
        cands = _plane_plane(s1, s2, pts)
    elif "plane" in (k1, k2):
        p, o = (s1, s2) if k1 == "plane" else (s2, s1)
        if o.kind == "cylinder":
            cands = _plane_cylinder(p, o, pts)
        elif o.kind == "cone":
            cands = _plane_cone(p, o, pts)
        elif o.kind == "sphere":
            cands = _plane_sphere(p, o, pts)
        elif o.kind in ("torus", "revolution"):
            # Parallels: plane across the axis.
            cands = _coaxial_plane(p, o, pts, tol)
    else:
        cands = _coaxial(s1, s2, pts, tol)
    if not cands:
        return None
    for c in cands:
        if _fits(c, pts, tol):
            return c
    return None


def _coaxial_plane(p: Plane, o, pts: np.ndarray, tol: float):
    n = p.frame.z
    a = o.frame.z
    if abs(abs(float(n @ a)) - 1.0) > 1e-12:
        return None
    loc = o.frame.to_local(pts)
    r = float(np.mean(np.hypot(loc[:, 0], loc[:, 1])))
    h = float(n @ (p.frame.origin - o.frame.origin)) / float(n @ a)
    return [Circle3(Frame.from_axes(o.frame.to_world(np.array([0, 0, h])), a, o.frame.x), r)]


def refine_onto(s1, s2, p: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Points moved onto both surfaces (alternating projections, then Newton on the
    pair of distance functions)."""
    q = np.array(p, dtype=np.float64)
    for _ in range(iterations):
        uv1 = s1.inverse(q)
        p1 = s1.point(uv1[:, 0], uv1[:, 1])
        n1 = s1.normal(uv1[:, 0], uv1[:, 1])
        uv2 = s2.inverse(q)
        p2 = s2.point(uv2[:, 0], uv2[:, 1])
        n2 = s2.normal(uv2[:, 0], uv2[:, 1])
        # Solve for a correction in the span of n1, n2 that zeroes both signed distances.
        d1 = np.sum((q - p1) * n1, axis=1)
        d2 = np.sum((q - p2) * n2, axis=1)
        c = np.sum(n1 * n2, axis=1)
        det = 1.0 - c * c
        ok = det > 1e-12
        a = np.where(ok, (d1 - c * d2) / np.where(ok, det, 1.0), d1)
        b = np.where(ok, (d2 - c * d1) / np.where(ok, det, 1.0), 0.0)
        step = a[:, None] * n1 + b[:, None] * n2
        q = q - step
        if float(np.max(np.linalg.norm(step, axis=1))) < 1e-15 * max(1.0, float(np.max(np.abs(q)))):
            break
    return q


def trace(s1, s2, pts: np.ndarray) -> tuple[Nurbs3, float]:
    """A cubic B-spline on both surfaces through the polyline's refined points, with the
    measured deviation from the surfaces."""
    q = refine_onto(s1, s2, pts)
    # Remove repeated points.
    keep = [0]
    for i in range(1, len(q)):
        if np.linalg.norm(q[i] - q[keep[-1]]) > 1e-12:
            keep.append(i)
    q = q[keep]
    if len(q) > 400:
        q = q[np.unique(np.linspace(0, len(q) - 1, 400).round().astype(int))]
    curve = Nurbs3(interpolate(q, 3 if len(q) >= 4 else len(q) - 1))
    lo, hi = curve.curve.domain
    ts = np.linspace(lo, hi, 4 * len(q))
    mids = curve.point(ts)
    dev = float(max(np.max(s1.distance(mids)), np.max(s2.distance(mids))))
    return curve, dev
