"""The sketch solver and profiles: classic configurations with hand-derived answers,
degrees of freedom, redundancy and conflicts with blame, branch stability, structure,
and region extraction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d.cad.sketch.equations import KERNELS
from aether3d.cad.sketch.model import validate
from aether3d.cad.sketch.profile import build_profile
from aether3d.cad.sketch.solve import solve_sketch
from aether3d.cad.sketch.structure import decompose, hopcroft_karp, tarjan_order
from aether3d.errors import CadError


def P(i, x, y):  # noqa: N802
    return {"id": i, "type": "point", "at": [x, y]}


def L(i, a, b):  # noqa: N802
    return {"id": i, "type": "line", "start": a, "end": b}


def C(t, *refs, **kw):  # noqa: N802
    cid = kw.pop("id", None) or f"{t}_{'_'.join(refs)}"
    return {"id": cid, "type": t, "refs": list(refs), **kw}


def at(sol, pid):
    return next(e["at"] for e in sol.definition["entities"] if e["id"] == pid)


def rect(x=(0, 0), w=(1.1, 0.1), h=(0.9, 0.6), d=(-0.1, 0.5)):
    return [
        P("p1", *x),
        P("p2", *w),
        P("p3", *h),
        P("p4", *d),
        L("l1", "p1", "p2"),
        L("l2", "p2", "p3"),
        L("l3", "p3", "p4"),
        L("l4", "p4", "p1"),
    ]


HV = [C("horizontal", "l1"), C("horizontal", "l3"), C("vertical", "l2"), C("vertical", "l4")]


# --------------------------------------------------------------------------- #
# Kernels and structure
# --------------------------------------------------------------------------- #

KERNEL_SPECS = {
    "linear": (4, [0, 1, -1, 2, 0.5]),
    "dist": (4, [0.3]),
    "dist_var": (6, [0.1, 1.0, -1.0]),
    "dist_diff": (8, [0.05]),
    "point_line": (7, [0.1, 1.0]),
    "point_line_mid": (8, [0.2]),
    "perp_dot": (8, [0.1]),
    "lines_angle": (8, [2.0, math.cos(0.7), math.sin(0.7)]),
    "line_angle": (4, [2.0, math.cos(0.3), math.sin(0.3)]),
}


@pytest.mark.parametrize("name", sorted(KERNEL_SPECS))
def test_kernel_jacobians_match_finite_differences(name):
    k, c = KERNEL_SPECS[name]
    rng = np.random.default_rng(len(name))
    v = rng.random((25, k))
    cs = np.tile(c, (25, 1)).astype(float)
    r, jac = KERNELS[name](v, cs)
    h = 1e-7
    for j in range(k):
        vp, vm = v.copy(), v.copy()
        vp[:, j] += h
        vm[:, j] -= h
        fd = (KERNELS[name](vp, cs)[0] - KERNELS[name](vm, cs)[0]) / (2 * h)
        np.testing.assert_allclose(jac[:, j], fd, atol=2e-7)


def test_matching_and_dulmage_mendelsohn():
    # rows: r0 {c0}, r1 {c0, c1}, r2 {c1}  -> over-determined (3 rows, 2 columns)
    d = decompose([[0], [0, 1], [1]], 2)
    assert sorted(d.over_rows) == [0, 1, 2] and d.structural_rank == 2
    # A chain: r0 {c0}, r1 {c0, c1}, r2 {c1, c2}: well determined, blocks in order.
    d = decompose([[0], [0, 1], [1, 2]], 3)
    assert not d.over_rows and not d.under_cols
    assert [b[0] for b in d.blocks] == [[0], [1], [2]]
    # An under-determined row: r0 {c0, c1}.
    d = decompose([[0, 1]], 2)
    assert d.under_rows == [0] and sorted(d.under_cols) == [0, 1]
    rm, cm = hopcroft_karp([[0, 1], [0], [1, 2]], 3)
    assert sum(1 for x in rm if x >= 0) == 3
    assert tarjan_order([[1], [0], [0]]) == [[0, 1], [2]]


# --------------------------------------------------------------------------- #
# Classic sketches
# --------------------------------------------------------------------------- #


def test_rectangle_degrees_of_freedom_then_fully_constrained():
    s = solve_sketch({"entities": rect(), "constraints": HV}, {})
    assert s.status == "under_constrained" and s.dof == 4
    cons = [
        *HV,
        C("fix", "p1"),
        C("length", "l1", value=1, id="w"),
        C("length", "l2", value=0.5, id="h"),
    ]
    s = solve_sketch({"entities": rect(), "constraints": cons}, {"w": 1.0, "h": 0.5})
    assert s.status == "fully_constrained" and s.dof == 0
    assert s.max_residual < 1e-14
    np.testing.assert_allclose(
        [at(s, p) for p in ("p1", "p2", "p3", "p4")],
        [[0, 0], [1, 0], [1, 0.5], [0, 0.5]],
        atol=1e-14,
    )
    assert s.free_points == []


def test_free_entities_are_reported():
    s = solve_sketch({"entities": rect(), "constraints": [*HV, C("fix", "p1")]}, {})
    assert s.dof == 2
    assert "p1" not in s.free_points and {"p2", "p3"} <= set(s.free_points)


def test_triangle_by_three_sides_3_4_5():
    ents = [
        P("a", 0, 0),
        P("b", 3.1, 0.2),
        P("c", 0.1, 4.2),
        L("ab", "a", "b"),
        L("bc", "b", "c"),
        L("ca", "c", "a"),
    ]
    cons = [
        C("fix", "a"),
        C("horizontal", "ab"),
        C("length", "ab", id="d1", value=3),
        C("length", "bc", id="d2", value=5),
        C("length", "ca", id="d3", value=4),
    ]
    s = solve_sketch({"entities": ents, "constraints": cons}, {"d1": 3, "d2": 5, "d3": 4})
    assert s.status == "fully_constrained"
    np.testing.assert_allclose(at(s, "c"), [0, 4], atol=1e-13)
    # Shrinking d3 keeps the triangle on the same side (law of cosines: x = (9+6.25-25)/6).
    s2 = solve_sketch(
        {"entities": s.definition["entities"], "constraints": cons},
        {"d1": 3, "d2": 5, "d3": 2.5},
        previous_values={"d1": 3, "d2": 5, "d3": 4},
    )
    x = (9 + 6.25 - 25) / 6
    np.testing.assert_allclose(at(s2, "c"), [x, math.sqrt(6.25 - x * x)], atol=1e-12)
    assert not s2.warnings


def test_redundant_constraint_is_blamed_with_its_dependencies():
    cons = [
        *HV,
        C("fix", "p1"),
        C("length", "l1", id="w", value=1),
        C("length", "l2", id="h", value=0.5),
    ]
    base = solve_sketch({"entities": rect(), "constraints": cons}, {"w": 1, "h": 0.5})
    more = [*cons, C("length", "l3", id="w2", value=1)]
    s = solve_sketch(
        {"entities": base.definition["entities"], "constraints": more}, {"w": 1, "h": 0.5, "w2": 1}
    )
    assert s.status == "redundant" and s.dof == 0
    assert s.redundant == [{"constraint": "w2", "depends_on": ["v1", "v2", "w"]}] or (
        s.redundant[0]["constraint"] == "w2" and "w" in s.redundant[0]["depends_on"]
    )


def test_conflicting_constraint_is_reported_not_applied():
    cons = [
        *HV,
        C("fix", "p1"),
        C("length", "l1", id="w", value=1),
        C("length", "l2", id="h", value=0.5),
        C("length", "l3", id="w2", value=1.2),
    ]
    s = solve_sketch({"entities": rect(), "constraints": cons}, {"w": 1, "h": 0.5, "w2": 1.2})
    assert s.status == "conflicting"
    conflict = s.conflicts[0]
    assert conflict["constraint"] == "w2" and "w" in conflict["conflicts_with"]
    assert conflict["residual"] == pytest.approx(0.1, rel=0.2) or conflict["residual"] > 0.01


def test_parallel_and_perpendicular_conflict():
    ents = [
        P("a", 0, 0),
        P("b", 1, 0.1),
        P("c", 0, 1),
        P("d", 1, 1.2),
        L("l1", "a", "b"),
        L("l2", "c", "d"),
    ]
    s = solve_sketch(
        {
            "entities": ents,
            "constraints": [
                C("parallel", "l1", "l2", id="pa"),
                C("perpendicular", "l1", "l2", id="pe"),
            ],
        },
        {},
    )
    assert s.status == "conflicting"
    assert s.conflicts[0]["constraint"] == "pe" and s.conflicts[0]["conflicts_with"] == ["pa"]


def test_tangent_arc_line_and_radius():
    ents = [
        P("c", 0, 0),
        P("s", 1, 0),
        P("e", 0, 1.05),
        {"id": "a1", "type": "arc", "center": "c", "start": "s", "end": "e", "radius": 1.0},
        P("q", -2, 1.1),
        L("l", "e", "q"),
    ]
    cons = [C("tangent", "a1", "l"), C("radius", "a1", id="r", value=2)]
    s = solve_sketch({"entities": ents, "constraints": cons}, {"r": 2})
    assert s.dof == 5 and s.max_residual < 1e-12
    c, e, q = (np.array(at(s, k)) for k in ("c", "e", "q"))
    assert np.linalg.norm(e - c) == pytest.approx(2, abs=1e-12)
    assert abs(float((q - e) @ (e - c))) < 1e-12  # radius perpendicular to the line


def test_circles_tangent_externally_and_concentric_equal():
    ents = [
        P("c1", 0, 0),
        {"id": "k1", "type": "circle", "center": "c1", "radius": 1.0},
        P("c2", 2.5, 0.3),
        {"id": "k2", "type": "circle", "center": "c2", "radius": 0.7},
    ]
    cons = [
        C("fix", "k1"),
        C("tangent", "k1", "k2", side=1),
        C("radius", "k2", id="r", value=0.5),
        C("horizontal", "c1", "c2"),
    ]
    s = solve_sketch({"entities": ents, "constraints": cons}, {"r": 0.5})
    assert s.status == "fully_constrained"
    np.testing.assert_allclose(at(s, "c2"), [1.5, 0], atol=1e-13)


def test_symmetric_midpoint_point_on_and_angle():
    ents = [
        P("a", -1.2, 0.1),
        P("b", 0.9, -0.2),
        P("o1", 0, -1),
        P("o2", 0.1, 1),
        L("axis", "o1", "o2"),
        P("m", 0.3, 0.4),
        L("ab", "a", "b"),
    ]
    cons = [
        C("fix", "o1"),
        C("vertical", "axis"),
        C("symmetric", "a", "b", "axis"),
        C("midpoint", "m", "ab"),
        C("distance", "a", "b", id="d", value=2),
    ]
    s = solve_sketch({"entities": ents, "constraints": cons}, {"d": 2})
    a, b, m = (np.array(at(s, k)) for k in ("a", "b", "m"))
    assert abs(a[0] + b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12
    np.testing.assert_allclose(m, (a + b) / 2, atol=1e-13)
    assert np.linalg.norm(b - a) == pytest.approx(2, abs=1e-12)


def test_angles_between_lines_and_to_the_axis():
    ents = [P("o", 0, 0), P("x", 2, 0.1), P("y", 1.5, 1.2), L("l1", "o", "x"), L("l2", "o", "y")]
    cons = [
        C("fix", "o"),
        C("angle", "l1", id="a1", value=math.radians(10), side=1),
        C("angle", "l1", "l2", id="a2", value=math.radians(35), side=1),
        C("length", "l1", id="n1", value=2),
        C("length", "l2", id="n2", value=1),
    ]
    vals = {"a1": math.radians(10), "a2": math.radians(35), "n1": 2, "n2": 1}
    s = solve_sketch({"entities": ents, "constraints": cons}, vals)
    assert s.status == "fully_constrained"
    np.testing.assert_allclose(
        at(s, "x"), [2 * math.cos(math.radians(10)), 2 * math.sin(math.radians(10))], atol=1e-12
    )
    np.testing.assert_allclose(
        at(s, "y"), [math.cos(math.radians(45)), math.sin(math.radians(45))], atol=1e-12
    )


def test_reference_dimensions_measure_without_driving():
    cons = [
        *HV,
        C("fix", "p1"),
        C("length", "l1", id="w", value=1),
        C("length", "l2", id="h", value=0.5),
        C("length", "l3", id="ref", value=0.0, driving=False),
    ]
    s = solve_sketch({"entities": rect(), "constraints": cons}, {"w": 1, "h": 0.5})
    assert s.status == "fully_constrained"
    assert s.reference["ref"] == pytest.approx(1.0, abs=1e-14)


def test_solving_is_deterministic_and_does_not_mutate_input():
    defn = {
        "entities": rect(),
        "constraints": [*HV, C("fix", "p1"), C("length", "l1", id="w", value=1)],
    }
    frozen = repr(defn)
    a = solve_sketch(defn, {"w": 1})
    b = solve_sketch(defn, {"w": 1})
    assert repr(defn) == frozen
    assert a.definition == b.definition


def test_random_dimension_edits_never_flip_a_corner():
    # An L-shaped outline driven by four dimensions, edited 60 times at random.
    pts = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]
    ents = [P(f"p{i}", *p) for i, p in enumerate(pts)] + [
        L(f"l{i}", f"p{i}", f"p{(i + 1) % 6}") for i in range(6)
    ]
    cons = [C("fix", "p0")]
    for i in range(6):
        cons.append(C("horizontal" if i % 2 == 0 else "vertical", f"l{i}"))
    dims = {"a": ("l0", 4), "b": ("l1", 1), "c": ("l2", 3), "d": ("l3", 2)}
    for k, (line, v) in dims.items():
        cons.append(C("length", line, id=k, value=v))
    values = {k: float(v) for k, (_, v) in dims.items()}
    defn = {"entities": ents, "constraints": cons}
    rng = np.random.default_rng(7)
    for _ in range(60):
        prev = dict(values)
        a = rng.uniform(2.5, 6.0)
        c = rng.uniform(0.2, a - 0.3)
        values = {"a": a, "c": c, "b": rng.uniform(0.3, 2), "d": rng.uniform(0.5, 3)}
        s = solve_sketch(defn, values, previous_values=prev)
        assert s.converged and s.status == "fully_constrained"
        assert not [w for w in s.warnings if w["code"] == "BRANCH_CHANGED"]
        defn = {"entities": s.definition["entities"], "constraints": cons}
        p = np.array([at(s, f"p{i}") for i in range(6)])
        # The outline stays counter-clockwise (positive area) with these exact sides.
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        assert area > 0


@pytest.mark.parametrize(
    ("defn", "code"),
    [
        ({"entities": [P("p", 0, 0), P("p", 1, 1)]}, "DUPLICATE_ID"),
        ({"entities": [L("l", "a", "b")]}, "UNKNOWN_ENTITY"),
        ({"entities": [P("p", 0, float("nan"))]}, "INVALID_VALUE"),
        ({"entities": [P("p", 0, 1e12)]}, "VALUE_TOO_LARGE"),
        (
            {"entities": [P("a", 0, 0)], "constraints": [C("parallel", "a", "a")]},
            "INVALID_CONSTRAINT",
        ),
        ({"entities": [P("a", 0, 0)], "constraints": [C("length", "a")]}, "INVALID_CONSTRAINT"),
        ({"entities": [{"id": "x", "type": "blob"}]}, "INVALID_SKETCH"),
        ({"entities": [P("a b", 0, 0)]}, "INVALID_ID"),
    ],
)
def test_invalid_definitions_are_located(defn, code):
    with pytest.raises(CadError) as e:
        validate(defn)
    assert e.value.code == code
    assert e.value.location


def test_values_are_checked():
    ents = [P("a", 0, 0), P("b", 1, 0), L("l", "a", "b")]
    for v in (0.0, -1.0, float("nan")):
        with pytest.raises(CadError):
            solve_sketch(
                {"entities": ents, "constraints": [C("length", "l", id="k", value=v)]}, {"k": v}
            )


def test_cancellation_and_timeouts_change_nothing():
    defn = {"entities": rect(), "constraints": [*HV, C("length", "l1", id="w", value=3)]}
    with pytest.raises(CadError) as e:
        solve_sketch(defn, {"w": 3}, should_stop=lambda: True)
    assert e.value.code == "CANCELLED"
    with pytest.raises(CadError) as e:
        solve_sketch(defn, {"w": 3}, deadline=0.0)
    assert e.value.code == "TIMEOUT"


def test_large_chain_sketch_solves_fast_in_small_blocks():
    n = 400
    ents = [P(f"p{i}", i * 1.1 + 0.05 * (i % 3), 0.07 * (i % 5)) for i in range(n + 1)]
    ents += [L(f"l{i}", f"p{i}", f"p{i + 1}") for i in range(n)]
    cons = [C("fix", "p0")]
    values = {}
    for i in range(n):
        cons.append(C("horizontal", f"l{i}"))
        cons.append(C("length", f"l{i}", id=f"d{i}", value=1))
        values[f"d{i}"] = 1.0
    s = solve_sketch({"entities": ents, "constraints": cons}, values)
    assert s.status == "fully_constrained"
    assert s.structure["largest_block"] <= 2
    assert at(s, f"p{n}")[0] == pytest.approx(n, abs=1e-9)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def _rect(n, x0, y0, x1, y1):
    return [
        P(n + "a", x0, y0),
        P(n + "b", x1, y0),
        P(n + "c", x1, y1),
        P(n + "d", x0, y1),
        L(n + "1", n + "a", n + "b"),
        L(n + "2", n + "b", n + "c"),
        L(n + "3", n + "c", n + "d"),
        L(n + "4", n + "d", n + "a"),
    ]


def _circle(n, x, y, r):
    return [P(n + "c", x, y), {"id": n, "type": "circle", "center": n + "c", "radius": r}]


def test_profile_plate_with_hole():
    pr = build_profile({"entities": _rect("r", 0, 0, 4, 2) + _circle("h", 1, 1, 0.5)})
    assert len(pr.regions) == 1
    assert pr.regions[0].area == pytest.approx(8 - math.pi * 0.25, rel=1e-14)
    assert len(pr.regions[0].holes) == 1


def test_profile_union_of_overlapping_outlines_and_nested_islands():
    pr = build_profile({"entities": _rect("r", 0, 0, 4, 2) + _rect("s", 2, 1, 6, 3)})
    assert [round(r.area, 12) for r in pr.regions] == [14.0]
    pr = build_profile(
        {"entities": _rect("r", 0, 0, 10, 10) + _rect("s", 2, 2, 8, 8) + _rect("t", 4, 4, 6, 6)}
    )
    assert sorted(round(r.area, 12) for r in pr.regions) == [4.0, 64.0]


def test_profile_selection_by_point_and_dividing_lines():
    ents = _rect("r", 0, 0, 4, 2) + [P("m1", 2, 0), P("m2", 2, 2), L("mid", "m1", "m2")]
    assert [r.area for r in build_profile({"entities": ents}).regions] == pytest.approx([8.0])
    assert [
        r.area for r in build_profile({"entities": ents}, select=[[1, 1]]).regions
    ] == pytest.approx([4.0])
    with pytest.raises(CadError) as e:
        build_profile({"entities": ents}, select=[[9, 9]])
    assert e.value.code == "NO_REGION_AT_POINT"


def test_profile_open_chains_are_reported_and_construction_ignored():
    with pytest.raises(CadError) as e:
        build_profile({"entities": [P("a", 0, 0), P("b", 1, 0), L("l", "a", "b")]})
    assert e.value.code == "OPEN_PROFILE"
    ents = _circle("c", 0, 0, 1) + [P("a", -2, 1), P("b", 2, 1), L("t", "a", "b")]
    pr = build_profile({"entities": ents})
    assert pr.regions[0].area == pytest.approx(math.pi, rel=1e-14)
    assert len(pr.open_ends) == 2
    ents = _rect("r", 0, 0, 4, 2)
    ents.append({**L("x", "ra", "rc"), "construction": True})
    assert build_profile({"entities": ents}).regions[0].area == pytest.approx(8.0)


def test_profile_with_arcs_and_tangent_contacts():
    ents = _rect("r", 0, 0, 4, 2) + _circle("h", 4, 1, 1)
    assert build_profile({"entities": ents}).regions[0].area == pytest.approx(
        8 + math.pi / 2, rel=1e-13
    )
