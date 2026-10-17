"""Precision geometry foundations: units, expressions, NURBS, 2D/3D geometry, offsets and
polygon triangulation -- checked against closed forms and independent computations."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aether3d.cad import geom2d, geom3d, nurbs
from aether3d.cad.expr import Expression, evaluate_value
from aether3d.cad.offset import offset_loop
from aether3d.cad.sketch.profile import Loop, Piece
from aether3d.cad.tolerance import TessellationTolerance, Tolerance
from aether3d.cad.triangulate import triangulate, triangulate_numpy
from aether3d.cad.units import (
    ANGLE,
    AREA,
    LENGTH,
    Quantity,
    format_quantity,
    parse_angle,
    parse_length,
    parse_quantity,
)
from aether3d.errors import CadError

# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "metres"),
    [
        ("12.5 mm", Fraction(125, 10000)),
        ("3/8 in", Fraction(3, 8) * Fraction(254, 10000)),
        ("2 1/2 in", Fraction(5, 2) * Fraction(254, 10000)),
        ("3' 4\"", Fraction(40) * Fraction(254, 10000)),
        ("3'-4 1/2\"", Fraction(81, 2) * Fraction(254, 10000)),
        ("3ft4in", Fraction(40) * Fraction(254, 10000)),
        ("3 ft 4 in", Fraction(40) * Fraction(254, 10000)),
        ('4"', Fraction(4) * Fraction(254, 10000)),
        ("-0.75 ft", -Fraction(3, 4) * Fraction(3048, 10000)),
        ("1e-3 m", Fraction(1, 1000)),
        ("2 yd", 2 * Fraction(9144, 10000)),
        ("7 um", Fraction(7, 10**6)),
    ],
)
def test_lengths_parse_exactly_rounded_once(text, metres):
    # The exact rational value rounded once to the nearest float64.
    assert parse_length(text) == float(metres)


def test_international_inch_and_foot_are_exact():
    assert Quantity.of(1, "in").value == 0.0254
    assert Quantity.of(1, "ft").value == 0.3048
    assert Quantity.of(1, "ft").to("in") == 12.0
    assert Quantity.of(254, "mm").to("in") == 10.0
    # Round trip through a unit and back.
    for v in (0.1, 1 / 3, 12.7, 1e-5, 123456.789):
        assert Quantity.of(v, "in").to("in") == pytest.approx(v, rel=2e-16)


def test_angles_and_dms():
    assert parse_angle("45 deg") == math.radians(45)
    assert parse_angle("90°") == math.pi / 2
    assert parse_angle(30) == math.radians(30)  # bare numbers default to degrees
    dms = parse_quantity("30° 15' 10\"", expected=ANGLE).to("deg")
    assert dms == pytest.approx(30 + 15 / 60 + 10 / 3600, abs=1e-12)
    assert parse_angle("0.5 rad") == 0.5


def test_areas_volumes_densities():
    assert parse_quantity("12 m2").dimension == AREA
    assert parse_quantity("12 m^2").value == 12.0
    assert parse_quantity("12 m²").value == 12.0
    assert parse_quantity("1 ft2").value == pytest.approx(0.3048**2, rel=1e-15)
    assert parse_quantity("2400 kg/m3").value == 2400.0


@pytest.mark.parametrize(
    "bad",
    ["", "mm", "12 parsecs", "1/0 in", "nan mm", "inf m", "12 mm mm", "x" * 300, "--3 m"],
)
def test_bad_quantities_are_refused(bad):
    with pytest.raises(CadError):
        parse_length(bad)


def test_unit_mismatches_are_errors():
    with pytest.raises(CadError) as e:
        parse_quantity("45 deg", expected=LENGTH)
    assert e.value.code == "UNIT_MISMATCH"
    with pytest.raises(CadError) as e:
        Quantity.of(1, "m") + Quantity.of(1, "deg")
    assert e.value.code == "UNIT_MISMATCH"
    with pytest.raises(CadError) as e:
        parse_quantity(12, expected=LENGTH)  # a bare number needs a unit or a default
    assert e.value.code == "UNIT_REQUIRED"
    assert parse_quantity(12, expected=LENGTH, default_unit="mm").value == 0.012


def test_formatting():
    assert format_quantity(Quantity.of(12.5, "mm")) == "12.5 mm"
    assert format_quantity(Quantity.of(40.5, "in"), style="architectural") == "3'-4 1/2\""
    assert format_quantity(Quantity.of(40.5, "in"), style="fractional") == '40 1/2"'
    assert format_quantity(Quantity.of(1 / 32, "in"), style="architectural", denominator=16) == (
        "0'-0\""  # 1/32 rounds to even (0) at 1/16
    )
    assert format_quantity(parse_quantity("45 deg")) == "45°"
    assert format_quantity(Quantity.of(1234.5678, "mm"), "m", precision=2) == "1.23 m"


# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #


def test_unit_aware_expressions():
    w = {"width": Quantity(1.0), "rise": Quantity(0.18), "run": Quantity(0.28)}
    assert Expression("width / 2 - 10 mm").evaluate(w).value == pytest.approx(0.49)
    assert Expression("(1/2) in").evaluate(w).value == pytest.approx(0.0127, rel=1e-15)
    assert Expression("2 * width + 3' 4\"").evaluate(w).to("in") == pytest.approx(2 / 0.0254 + 40)
    ang = Expression("atan2(rise, run)").evaluate(w)
    assert ang.dimension == ANGLE
    assert ang.value == pytest.approx(math.atan2(0.18, 0.28))
    assert Expression("width if width > 2 m else 3 m").evaluate(w).value == 3.0
    assert Expression("round(width / 1 mm)").evaluate(w).value == 1000.0
    assert evaluate_value("=width * 2", w, expected=LENGTH).value == 2.0
    assert evaluate_value("25 mm", w, expected=LENGTH).value == 0.025


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os')",
        "width.real",
        "width[0]",
        "open('x')",
        "(lambda: 1)()",
        "[x for x in width]",
        "width + 3 deg",
        "sin(width)",
        "2 ** 1000 ** 1000",
        "exec('1')",
        "_private",
        "max(width, k=1)",
        "f'{width}'",
        "width +" + " width +" * 400 + " width",
    ],
)
def test_expressions_refuse_anything_but_arithmetic(source):
    with pytest.raises(CadError):
        Expression(source).evaluate({"width": Quantity(1.0)})


def test_unknown_parameters_are_named():
    with pytest.raises(CadError) as e:
        Expression("depth * 2").evaluate({})
    assert e.value.code == "UNKNOWN_PARAMETER"


# --------------------------------------------------------------------------- #
# Tolerances
# --------------------------------------------------------------------------- #


def test_tolerance_policy_scales_with_the_model():
    small = Tolerance.for_scale(0.01)
    site = Tolerance.for_scale(4.0e6)
    assert small.resolution == 1e-10
    assert site.resolution == pytest.approx(2**10 * 2**-53 * 4e6)
    assert site.resolution < 1e-6  # still sub-micrometre on a 4,000 km site
    assert small.solve < small.resolution and site.solve < site.resolution
    with pytest.raises(CadError):
        TessellationTolerance(chord=-1)
    with pytest.raises(CadError):
        TessellationTolerance(angle=math.radians(80))


# --------------------------------------------------------------------------- #
# NURBS
# --------------------------------------------------------------------------- #


def test_rational_circle_is_exact_and_derivatives_match():
    c = nurbs.circle_arc_nurbs(np.zeros(3), np.eye(3)[0], np.eye(3)[1], 2.0, 0.3, 2 * math.pi)
    t = np.linspace(0, 1, 1001)
    assert np.abs(np.linalg.norm(c.evaluate(t), axis=1) - 2.0).max() < 1e-14
    # Away from the double knots (C0 in parameter), compare with finite differences.
    t = np.linspace(0.01, 0.24, 37)
    d = c.derivatives(t, 2)
    h = 1e-6
    fd1 = (c.evaluate(t + h) - c.evaluate(t - h)) / (2 * h)
    np.testing.assert_allclose(d[1], fd1, rtol=0, atol=1e-8 * np.abs(d[1]).max())
    fd2 = (c.evaluate(t + 1e-4) - 2 * c.evaluate(t) + c.evaluate(t - 1e-4)) / 1e-8
    np.testing.assert_allclose(d[2], fd2, rtol=0, atol=1e-5 * np.abs(d[2]).max())


def test_bspline_matches_bernstein_and_knot_insertion_keeps_the_curve():
    pts = np.array([[0, 0], [1, 2], [3, 2], [4, 0]], float)
    c = nurbs.BSplineCurve(3, [0, 0, 0, 0, 1, 1, 1, 1], pts)
    t = np.linspace(0, 1, 50)
    b = np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3], axis=1)
    np.testing.assert_allclose(c.evaluate(t), b @ pts, atol=1e-15)
    c2 = c.insert_knot(0.3, 2).insert_knot(0.7)
    np.testing.assert_allclose(c2.evaluate(t), c.evaluate(t), atol=1e-14)


def test_interpolation_passes_through_its_points():
    rng = np.random.default_rng(3)
    pts = rng.random((9, 3))
    c = nurbs.interpolate(pts)
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    params = np.concatenate([[0], np.cumsum(chords) / chords.sum()])
    np.testing.assert_allclose(c.evaluate(params), pts, atol=1e-12)


def test_rational_surface_is_a_cylinder_and_derivatives_match():
    w = math.sqrt(0.5)
    s = nurbs.BSplineSurface(
        2,
        1,
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1],
        np.array([[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]], [[0, 1, 0], [0, 1, 1]]], float),
        np.array([[1, 1], [w, w], [1, 1]]),
    )
    rng = np.random.default_rng(1)
    u, v = rng.random(50), rng.random(50)
    p = s.evaluate(u, v)
    assert np.abs(np.hypot(p[:, 0], p[:, 1]) - 1).max() < 1e-15
    d = s.derivatives(u, v, 2)
    h = 1e-6
    np.testing.assert_allclose(
        d[1, 0], (s.evaluate(u + h, v) - s.evaluate(u - h, v)) / 2e-6, atol=1e-8
    )
    fuv = (
        s.evaluate(u + 1e-4, v + 1e-4)
        - s.evaluate(u + 1e-4, v - 1e-4)
        - s.evaluate(u - 1e-4, v + 1e-4)
        + s.evaluate(u - 1e-4, v - 1e-4)
    ) / 4e-8
    np.testing.assert_allclose(d[1, 1], fuv, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": 0, "knots": [0, 1], "points": [[0, 0], [1, 1]]},
        {"degree": 12, "knots": [0] * 14 + [1] * 14, "points": [[0, 0]] * 13},
        {"degree": 2, "knots": [0, 0, 0, 1, 1], "points": [[0, 0], [1, 1], [2, 0]]},
        {"degree": 1, "knots": [0, 0, 1, 1], "points": [[0, np.nan], [1, 1]]},
        {"degree": 1, "knots": [0, 1, 0.5, 1], "points": [[0, 0], [1, 1]]},
        {"degree": 1, "knots": [0, 0, 1, 1], "points": [[0, 0], [1, 1]], "weights": [1, -1]},
    ],
)
def test_malformed_splines_are_refused(kwargs):
    with pytest.raises(CadError):
        nurbs.BSplineCurve(**kwargs)


# --------------------------------------------------------------------------- #
# 2D geometry
# --------------------------------------------------------------------------- #


def test_line_circle_tangency_is_one_point():
    c = geom2d.Arc([0, 0], 1.0, 0.0, 2 * math.pi)
    hits = geom2d.intersect(geom2d.Segment([-2, 1], [2, 1]), c, 1e-10)
    assert len(hits) == 1 and hits[0].tangent
    np.testing.assert_allclose(hits[0].point, [0, 1], atol=1e-15)
    # Just inside: two points; just outside: none.
    assert len(geom2d.intersect(geom2d.Segment([-2, 1 - 1e-6], [2, 1 - 1e-6]), c, 1e-10)) == 2
    assert not geom2d.intersect(geom2d.Segment([-2, 1 + 1e-6], [2, 1 + 1e-6]), c, 1e-10)


def test_circle_circle_intersections_closed_form():
    a = geom2d.Arc([0, 0], 1, 0, 2 * math.pi)
    b = geom2d.Arc([1.5, 0], 1, 0, 2 * math.pi)
    pts = sorted((tuple(h.point) for h in geom2d.intersect(a, b, 1e-10)), key=lambda p: p[1])
    y = math.sqrt(1 - 0.75**2)
    np.testing.assert_allclose(pts, [(0.75, -y), (0.75, y)], atol=1e-15)
    touch = geom2d.intersect(a, geom2d.Arc([2, 0], 1, 0, 2 * math.pi), 1e-10)
    assert len(touch) == 1 and touch[0].tangent


def test_collinear_overlap_and_parallel_segments():
    s1 = geom2d.Segment([0, 0], [2, 0])
    hits = geom2d.intersect(s1, geom2d.Segment([1, 0], [3, 0]), 1e-12)
    assert len(hits) == 1 and hits[0].overlap is not None
    assert not geom2d.intersect(s1, geom2d.Segment([0, 1e-6], [2, 1e-6]), 1e-12)


def test_arc_from_three_points_and_signed_area():
    arc = geom2d.arc_from_points([1, 0], [0, 1], [-1, 0])
    assert arc.radius == pytest.approx(1)
    assert arc.sweep == pytest.approx(math.pi)
    # Green's term of a quarter circle about the origin is its sector area.
    assert geom2d.Arc([0, 0], 2, 0, math.pi / 2).signed_area_term() == pytest.approx(math.pi)


def test_spline_length_and_intersections():
    sp = geom2d.Spline(nurbs.interpolate(np.array([[0, 0], [1, 2], [2, -1], [3, 1], [4, 0.0]])))
    dense = sp.point(np.linspace(0, 1, 200001))
    ref = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
    assert sp.length == pytest.approx(ref, rel=1e-9)
    hits = geom2d.intersect(sp, geom2d.Segment([0, 0.5], [4, 0.5]), 1e-10)
    assert len(hits) == 4
    for h in hits:
        assert abs(h.point[1] - 0.5) < 1e-9
        assert np.linalg.norm(sp.point(h.ta) - h.point) < 1e-9


def test_spline_pieces_match_the_whole():
    sp = geom2d.Spline(nurbs.interpolate(np.array([[0, 0], [1, 2], [2, -1], [3, 1], [4, 0.0]])))
    for t0, t1 in ((0.2, 0.7), (0, 0.5), (0.3, 1.0)):
        sub = sp.sub(t0, t1)
        ts = np.linspace(0, 1, 11)
        np.testing.assert_allclose(sub.point(ts), sp.point(t0 + ts * (t1 - t0)), atol=1e-13)


# --------------------------------------------------------------------------- #
# 3D geometry
# --------------------------------------------------------------------------- #


def _surfaces():
    rng = np.random.default_rng(0)
    f = geom3d.Frame.from_axes(rng.random(3), rng.random(3) - 0.5, rng.random(3))
    prof = geom3d.Nurbs3(
        nurbs.interpolate(
            f.to_world(np.c_[0.5 + rng.random(5) * 0.3, np.zeros(5), np.linspace(0, 1, 5)])
        )
    )
    return [
        (geom3d.Plane(f), (-1, 1), (-1, 1)),
        (geom3d.Cylinder(f, 0.7), (-3, 3), (-1, 1)),
        (geom3d.Cone(f, 0.5, 0.3), (-3, 3), (0, 1)),
        (geom3d.Sphere(f, 0.8), (-3, 3), (-1.2, 1.2)),
        (geom3d.Torus(f, 1.0, 0.3), (-3, 3), (-3, 3)),
        (geom3d.Revolution(f, prof), (-3, 3), (0.05, 0.95)),
    ]


@pytest.mark.parametrize("index", range(6))
def test_surface_derivatives_normals_and_inverse(index):
    s, (u0, u1), (v0, v1) = _surfaces()[index]
    rng = np.random.default_rng(index)
    u, v = rng.uniform(u0, u1, 40), rng.uniform(v0, v1, 40)
    p = s.point(u, v)
    su, sv = s.derivatives(u, v)
    h = 1e-6
    np.testing.assert_allclose(su, (s.point(u + h, v) - s.point(u - h, v)) / (2 * h), atol=1e-8)
    np.testing.assert_allclose(sv, (s.point(u, v + h) - s.point(u, v - h)) / (2 * h), atol=1e-8)
    n = np.cross(su, sv)
    n /= np.linalg.norm(n, axis=1)[:, None]
    np.testing.assert_allclose(s.normal(u, v), n, atol=1e-12)
    uv = s.inverse(p)
    np.testing.assert_allclose(s.point(uv[:, 0], uv[:, 1]), p, atol=1e-12)


def test_frames_are_right_handed_and_round_trip():
    f = geom3d.Frame.from_axes([1, 2, 3], [0, 1, 1], [1, 0, 0])
    assert np.allclose(np.cross(f.x, f.y), f.z)
    pts = np.random.default_rng(0).random((10, 3))
    np.testing.assert_allclose(f.to_local(f.to_world(pts)), pts, atol=1e-15)
    with pytest.raises(CadError):
        geom3d.Frame.from_axes([0, 0, 0], [0, 0, 1], [0, 0, 2])


# --------------------------------------------------------------------------- #
# Offsets
# --------------------------------------------------------------------------- #


def _rounded_rectangle(w, h, r):
    s = geom2d.Segment
    a = geom2d.Arc
    pieces = [
        s([r, 0], [w - r, 0]),
        a([w - r, r], r, -math.pi / 2, math.pi / 2),
        s([w, r], [w, h - r]),
        a([w - r, h - r], r, 0, math.pi / 2),
        s([w - r, h], [r, h]),
        a([r, h - r], r, math.pi / 2, math.pi / 2),
        s([0, h - r], [0, r]),
        a([r, r], r, math.pi, math.pi / 2),
    ]
    return Loop([Piece(p, f"e{i}", 0) for i, p in enumerate(pieces)])


def test_inward_offset_of_a_rounded_rectangle_is_exact():
    loop = _rounded_rectangle(4.0, 2.0, 0.5)
    inner = offset_loop(loop, 0.2, 1e-10)
    # Area of a rounded rectangle: w h - (4 - pi) r^2.
    w, h, r = 3.6, 1.6, 0.3
    assert inner.area == pytest.approx(w * h - (4 - math.pi) * r * r, rel=1e-13)


def test_offsets_that_change_topology_are_refused():
    loop = _rounded_rectangle(4.0, 2.0, 0.5)
    with pytest.raises(CadError) as e:
        offset_loop(loop, 0.6, 1e-10)  # the corner arcs (r = 0.5) would collapse
    assert e.value.code in ("OFFSET_COLLAPSE", "OFFSET_TOPOLOGY_CHANGE")


# --------------------------------------------------------------------------- #
# Triangulation
# --------------------------------------------------------------------------- #


def _area(tri, pts):
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    return 0.5 * (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


def _polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_triangulation_of_polygons_with_holes_covers_them_exactly():
    n = 64
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
    hole = np.c_[2 + np.cos(th), 2 + np.sin(th)][::-1]
    pts = np.vstack([outer, hole])
    tri, owner = triangulate(pts, np.array([0, 4, 4 + n]), np.array([0, 2]))
    area = _area(tri, pts)
    assert area.min() > 0
    assert area.sum() == pytest.approx(16 - abs(_polygon_area(hole)), rel=1e-14)
    assert len(tri) == n + 4  # n + 4 vertices, one hole: (V + 2H - 2) triangles


def test_triangulation_keeps_collinear_boundary_points():
    r = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]], float)
    tri, _ = triangulate(r, np.array([0, 6]), np.array([0, 1]))
    assert sorted(set(tri.ravel().tolist())) == list(range(6))
    assert _area(tri, r).sum() == pytest.approx(2.0)


def test_triangulation_of_large_star_uses_z_order_and_is_exact():
    k = 500
    rad = np.where(np.arange(2 * k) % 2 == 0, 1.0, 0.4)
    a = np.linspace(0, 2 * np.pi, 2 * k, endpoint=False)
    star = np.c_[rad * np.cos(a), rad * np.sin(a)]
    tri, _ = triangulate_numpy(star, np.array([0, 2 * k]), np.array([0, 1]))
    assert len(tri) == 2 * k - 2
    assert _area(tri, star).min() > 0
    assert _area(tri, star).sum() == pytest.approx(_polygon_area(star), rel=1e-13)


def test_triangulation_inputs_are_validated():
    with pytest.raises(CadError):
        triangulate(np.array([[0, 0], [1, np.nan], [0, 1]]), np.array([0, 3]), np.array([0, 1]))
    with pytest.raises(CadError):
        triangulate(np.zeros((3, 2)), np.array([0, 2]), np.array([0, 1]))
    with pytest.raises(CadError):
        triangulate(np.zeros((3, 2)), np.array([0, 3]), np.array([0, 0]))
