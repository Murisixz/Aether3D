"""Exact B-reps: builders, validity, tessellation accuracy, mass properties against closed
forms, and booleans (Manifold topology, exact geometry recovered)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d.cad.brep import build
from aether3d.cad.brep.properties import mass_properties
from aether3d.cad.brep.tessellate import max_deviation, tessellate
from aether3d.cad.brep.topology import Body, validate
from aether3d.cad.geom3d import Frame
from aether3d.cad.sketch.profile import build_profile
from aether3d.cad.tolerance import TessellationTolerance
from aether3d.errors import CadError


def P(i, x, y):  # noqa: N802
    return {"id": i, "type": "point", "at": [x, y]}


def L(i, a, b):  # noqa: N802
    return {"id": i, "type": "line", "start": a, "end": b}


def plate_region():
    ents = [
        P("a", 0, 0),
        P("b", 4, 0),
        P("c", 4, 2),
        P("d", 0, 2),
        L("l1", "a", "b"),
        L("l2", "b", "c"),
        L("l3", "c", "d"),
        L("l4", "d", "a"),
        P("hc", 1, 1),
        {"id": "h", "type": "circle", "center": "hc", "radius": 0.5},
    ]
    return build_profile({"entities": ents}).regions[0]


def _watertight(t) -> bool:
    e = np.sort(
        np.concatenate([t.triangles[:, [0, 1]], t.triangles[:, [1, 2]], t.triangles[:, [2, 0]]]),
        axis=1,
    )
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _mesh_volume(t) -> float:
    p = t.positions[t.triangles]
    return float(np.sum(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])))) / 6


PI = math.pi
CASES = {
    "box": (lambda: build.box((1, 2, 3)), 6.0, 22.0),
    "cylinder": (lambda: build.cylinder(0.5, 2), PI * 0.25 * 2, 2 * PI * 0.5 * 2 + 2 * PI * 0.25),
    "sphere": (lambda: build.sphere(1.0), 4 / 3 * PI, 4 * PI),
    "cone": (lambda: build.cone(1.0, 0.4, 2.0), PI * 2 / 3 * (1 + 0.4 + 0.16), None),
    "torus": (lambda: build.torus(2.0, 0.5), 2 * PI**2 * 2 * 0.25, 4 * PI**2 * 2 * 0.5),
    "plate": (
        lambda: build.extrude(plate_region(), Frame.world(), 0, 0.5),
        (8 - PI / 4) * 0.5,
        None,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_builders_are_valid_and_measure_exactly(name):
    make, volume, area = CASES[name]
    body = make()
    report = validate(body)
    assert report["valid"], report["problems"]
    m = mass_properties(body)
    assert m.volume == pytest.approx(volume, rel=1e-13)
    if area is not None:
        assert m.area == pytest.approx(area, rel=1e-13)
    assert m.error_volume < 1e-12 * max(1.0, volume)


def test_inertia_of_box_cylinder_and_sphere():
    m = mass_properties(build.box((1, 2, 3)))
    np.testing.assert_allclose(
        np.diag(m.inertia), [6 * 13 / 12, 6 * 10 / 12, 6 * 5 / 12], rtol=1e-13
    )
    np.testing.assert_allclose(m.inertia - np.diag(np.diag(m.inertia)), 0, atol=1e-13)
    r, h = 0.5, 2.0
    v = PI * r * r * h
    m = mass_properties(build.cylinder(r, h))
    np.testing.assert_allclose(
        np.diag(m.inertia), [v * (3 * r * r + h * h) / 12] * 2 + [v * r * r / 2], rtol=1e-13
    )
    m = mass_properties(build.sphere(1.0))
    np.testing.assert_allclose(np.diag(m.inertia), [0.4 * 4 / 3 * PI] * 3, rtol=1e-13)
    np.testing.assert_allclose(m.centroid, 0, atol=1e-14)


def test_revolution_matches_pappus_and_drafted_prism_matches_frustum_sum():
    reg = plate_region()
    area = 8 - PI / 4
    cy = (8 * 1 - PI / 4 * 1) / area  # centroid height of the plate with its hole
    body = build.revolve(reg, Frame.world(), (0, -1), (1, 0), PI / 2)
    assert validate(body)["valid"]
    assert mass_properties(body).volume == pytest.approx(area * (cy + 1) * PI / 2, rel=1e-13)
    # A drafted circular boss: a cone frustum, exactly.
    boss = build.extrude(
        build.circle_region(0, 0, 1.0), Frame.world(), 0, 1.0, draft=math.radians(10)
    )
    r1 = 1.0 - math.tan(math.radians(10))
    assert mass_properties(boss).volume == pytest.approx(PI / 3 * (1 + r1 + r1 * r1), rel=1e-13)


def test_far_from_origin_bodies_keep_their_digits():
    far = Frame.from_axes([4e6, 3e6, 100], [0, 0, 1], [1, 0, 0])
    m = mass_properties(build.box((1, 2, 3), frame=far))
    assert m.volume == pytest.approx(6.0, rel=1e-14)
    np.testing.assert_allclose(m.centroid, [4e6, 3e6, 100], rtol=0, atol=1e-8)


@pytest.mark.parametrize("name", sorted(CASES))
def test_tessellations_are_watertight_and_within_the_chord(name):
    body = CASES[name][0]()
    tol = TessellationTolerance(chord=2e-3, relative=None)
    t = tessellate(body, tol)
    assert _watertight(t)
    assert max_deviation(body, t) <= tol.chord * 1.0001
    # Mesh volume converges to the exact one (inscribed: smaller, within ~ area * chord).
    exact = CASES[name][1]
    assert abs(_mesh_volume(t) - exact) <= mass_properties(body).area * tol.chord
    mesh = t.to_mesh()
    assert mesh.triangle_count == len(t.triangles)


def test_body_transforms_and_serialization_round_trip():
    body = build.extrude(plate_region(), Frame.world(), 0, 0.5)
    rot = np.eye(4)
    c, s = math.cos(0.7), math.sin(0.7)
    rot[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    rot[:3, 3] = [5, -3, 2]
    moved = body.transformed(rot)
    assert validate(moved)["valid"]
    assert mass_properties(moved).volume == pytest.approx(mass_properties(body).volume, rel=1e-13)
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    flipped = body.transformed(mirror)
    assert validate(flipped)["valid"]
    assert mass_properties(flipped).volume == pytest.approx(mass_properties(body).volume, rel=1e-13)
    again = Body.from_dict(body.to_dict())
    assert validate(again)["valid"]
    assert mass_properties(again).volume == mass_properties(body).volume


def test_revolve_refuses_profiles_across_the_axis():
    with pytest.raises(CadError) as e:
        build.revolve(plate_region(), Frame.world(), (0, 1), (1, 0), PI)
    assert e.value.code == "REVOLVE_CROSSES_AXIS"


# --------------------------------------------------------------------------- #
# Booleans
# --------------------------------------------------------------------------- #

manifold3d = pytest.importorskip("manifold3d")

from aether3d.cad.brep.boolean import boolean  # noqa: E402


def test_plate_minus_cylinder_recovers_circles_exactly():
    plate = build.box((4, 2, 0.5))
    tool = build.cylinder(0.5, 1.0, frame=Frame.from_axes([1, 0, 0], [0, 0, 1]))
    r = boolean(plate, tool, "difference")
    assert validate(r)["valid"]
    assert mass_properties(r).volume == pytest.approx((8 - PI / 4) * 0.5, rel=1e-13)
    curves = r.summary()["curves"]
    assert curves.get("circle") == 4 and "nurbs" not in curves
    assert r.summary()["max_edge_tolerance"] == 0.0


def test_box_unions_intersections_and_coplanar_merges():
    a = build.box((1, 1, 1))
    b = build.box((1, 1, 1), frame=Frame.from_axes([0.5, 0.5, 0], [0, 0, 1]))
    assert mass_properties(boolean(a, b, "union")).volume == pytest.approx(1.75, rel=1e-14)
    assert mass_properties(boolean(a, b, "intersection")).volume == pytest.approx(0.25, rel=1e-14)
    c = build.box((1, 1, 1), frame=Frame.from_axes([1, 0, 0], [0, 0, 1]))
    joined = boolean(a, c, "union")
    assert len(joined.faces) == 6  # coplanar faces of both boxes merged
    assert mass_properties(joined).volume == pytest.approx(2.0, rel=1e-14)


def test_oblique_hole_has_exact_elliptic_edges():
    plate = build.box((4, 2, 0.5))
    tool = build.cylinder(0.3, 3.0, frame=Frame.from_axes([0, 0, 0], [1, 0, 1]))
    r = boolean(plate, tool, "difference")
    assert validate(r)["valid"]
    assert r.summary()["curves"].get("ellipse") == 4
    # A cylinder crossing a slab of thickness t at 45 degrees removes pi r^2 t / cos 45.
    assert mass_properties(r).volume == pytest.approx(4 - PI * 0.09 * 0.5 * math.sqrt(2), rel=1e-12)


def test_internal_void_becomes_a_second_shell():
    r = boolean(build.box((2, 2, 2)), build.sphere(0.6), "difference")
    assert len(r.shells) == 2
    assert mass_properties(r).volume == pytest.approx(8 - 4 / 3 * PI * 0.216, rel=1e-13)


def test_crossing_cylinders_get_traced_edges_with_measured_tolerance():
    big = build.cylinder(0.5, 3)
    small = build.cylinder(0.3, 3, frame=Frame.from_axes([0, 0, 0], [1, 0, 0]))
    r = boolean(big, small, "union")
    assert validate(r)["valid"]
    s = r.summary()
    assert s["curves"].get("nurbs", 0) > 0
    assert 0 < s["max_edge_tolerance"] < 1e-6
    assert any("traced" in n for n in r.notes)
    # Independent volume: V(A) + V(B) - integral of 4 sqrt((R^2 - y^2)(r^2 - y^2)) dy.
    x, w = np.polynomial.legendre.leggauss(200)
    th = (x + 1) * PI / 4  # y = r sin(theta), theta in [0, pi/2]
    big_r, small_r = 0.5, 0.3
    integrand = 4 * np.sqrt(big_r**2 - (small_r * np.sin(th)) ** 2) * small_r**2 * np.cos(th) ** 2
    inter = 2 * float(np.sum(w * integrand) * PI / 4)
    expected = PI * big_r**2 * 3 + PI * small_r**2 * 3 - inter
    assert mass_properties(r).volume == pytest.approx(expected, rel=1e-8)


def test_empty_results_and_non_solids_are_errors():
    a = build.box((1, 1, 1))
    far = build.box((1, 1, 1), frame=Frame.from_axes([5, 0, 0], [0, 0, 1]))
    with pytest.raises(CadError) as e:
        boolean(a, far, "intersection")
    assert e.value.code == "EMPTY_RESULT"
    with pytest.raises(CadError):
        boolean(a, far, "xor")
