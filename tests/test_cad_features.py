"""Part features against closed-form volumes: holes, fillets, chamfers, shells, patterns,
mirrors, revolutions, sweeps, lofts and drafts; references, errors and multi-body parts."""

from __future__ import annotations

import math

import pytest

import aether3d
from aether3d.errors import CadError

pytest.importorskip("manifold3d")

PI = math.pi


def doc_mm():
    scene = aether3d.Scene("features")
    return scene, scene.cad


def plate(doc, w=80, d=40, t=10, name="Plate"):
    sk = doc.sketch(f"{name} profile")
    sk.rectangle((0, 0), (w, d))
    part = doc.part(name)
    part.extrude(sk, t, id="base")
    return part


def vol(part, body="main"):
    m = part.measure(body)
    assert m["valid"], m
    return m["volume"]


def test_counterbore_countersink_and_drill_point_holes():
    scene, doc = doc_mm()
    p = plate(doc)
    base = 80 * 40 * 10
    p.hole(
        "base:end", [[20, 20, 10]], 8, kind="counterbore", cbore_diameter=14, cbore_depth=4, id="cb"
    )
    expected = base - (PI * 49 * 4 + PI * 16 * 6)
    assert vol(p) == pytest.approx(expected, rel=1e-12)
    p.hole(
        "base:end",
        [[60, 20, 10]],
        8,
        kind="countersink",
        csink_diameter=16,
        csink_angle="90 deg",
        id="cs",
    )
    h = (8 - 4) / math.tan(PI / 4)
    expected -= PI * h * (64 + 32 + 16) / 3 + PI * 16 * (10 - h)
    assert vol(p) == pytest.approx(expected, rel=1e-12)
    p.hole("base:end", [[40, 10, 10]], 6, depth=5, tip_angle="118 deg", id="blind")
    tip = 3 / math.tan(math.radians(59))
    expected -= PI * 9 * 5 + PI * 9 * tip / 3
    assert vol(p) == pytest.approx(expected, rel=1e-12)
    names = {f["name"].split("#")[0] for f in p.faces()}
    assert {"cb:rev/cbore.0", "cs:rev/csink.0", "blind:rev/tip.0"} <= names


def test_holes_must_be_on_their_face():
    scene, doc = doc_mm()
    p = plate(doc)
    with pytest.raises(CadError) as e:
        p.hole("base:end", [[100, 20, 10]], 8)
    assert e.value.code == "HOLE_OFF_FACE"
    with pytest.raises(CadError) as e:
        p.hole("base:side/r1", [[20, 0, 5]], 4, kind="counterbore", cbore_diameter=3, cbore_depth=1)
    assert e.value.code == "INVALID_VALUE"


def test_convex_and_concave_fillets_and_chamfers():
    scene, doc = doc_mm()
    p = plate(doc)
    p.fillet([{"edge": ["base:end", "base:side/r2"]}], 3, id="round")
    assert vol(p) == pytest.approx(32000 - (1 - PI / 4) * 9 * 40, rel=1e-12)
    p.chamfer([{"edge": ["base:end", "base:side/r4"]}], 2, id="bevel")
    assert vol(p) == pytest.approx(32000 - (1 - PI / 4) * 9 * 40 - 2 * 2 / 2 * 40, rel=1e-12)
    faces = {f["name"]: f["surface"] for f in p.faces()}
    assert faces["round:side/round"] == "cylinder" and faces["bevel:side/bevel"] == "plane"
    # A concave edge: an L section; the fillet adds material.
    scene2, doc2 = doc_mm()
    sk = doc2.sketch("L")
    sk.polyline([(0, 0), (40, 0), (40, 10), (10, 10), (10, 30), (0, 30)])
    part = doc2.part("Angle")
    part.extrude(sk, 20, id="base")
    v0 = vol(part)
    part.fillet([{"edge": ["base:side/l3", "base:side/l4"]}], 4, id="inner")
    assert vol(part) == pytest.approx(v0 + (1 - PI / 4) * 16 * 20, rel=1e-12)


def test_blending_edges_that_meet_or_have_slanted_ends_is_refused():
    scene, doc = doc_mm()
    p = plate(doc)
    with pytest.raises(CadError) as e:
        p.fillet(
            [{"edge": ["base:end", "base:side/r1"]}, {"edge": ["base:end", "base:side/r2"]}], 2
        )
    assert e.value.code == "FILLET_CORNER_UNSUPPORTED"
    with pytest.raises(CadError) as e:
        p.fillet([{"edge": ["base:side/r1", "base:side/r2"]}], 50)
    assert e.value.code in ("FILLET_UNSUPPORTED", "EMPTY_RESULT", "BOOLEAN_FAILED")


def test_shell_of_a_rounded_box_is_an_exact_offset():
    scene, doc = doc_mm()
    sk = doc.sketch("Housing")
    sk.rectangle((0, 0), (60, 40))
    part = doc.part("Box")
    part.extrude(sk, 30, id="base")
    part.shell("base", 2, open=("end",), id="shell")
    outer = 60 * 40 * 30
    inner = 56 * 36 * 28
    assert vol(part) == pytest.approx(outer - inner, rel=1e-12)
    with pytest.raises(CadError) as e:
        part.set("shell", thickness=25)
    assert e.value.code in ("OFFSET_TOPOLOGY_CHANGE", "OFFSET_COLLAPSE", "SHELL_TOO_THICK")


def test_patterns_and_mirrors_of_holes():
    scene, doc = doc_mm()
    p = plate(doc, 100, 40, 10)
    p.hole("base:end", [[10, 20, 10]], 6, id="h")
    p.pattern(["h"], kind="linear", direction=[1, 0, 0], count=5, spacing=20)
    assert vol(p) == pytest.approx(40000 - 5 * PI * 9 * 10, rel=1e-12)
    p.mirror({"point": [50, 20, 0], "normal": [0, 1, 0]}, features=["h"], id="m")
    # Mirrored across the plate's center line: onto itself (no new material removed).
    assert vol(p) == pytest.approx(40000 - 5 * PI * 9 * 10, rel=1e-12)


def test_circular_pattern():
    scene, doc = doc_mm()
    sk = doc.sketch("Disc")
    sk.circle((0, 0), 50)
    p = doc.part("Flange")
    p.extrude(sk, 8, id="base")
    p.hole("base:end", [[35, 0, 8]], 8, id="bolt")
    p.pattern(["bolt"], kind="circular", axis={"point": [0, 0, 0], "direction": [0, 0, 1]}, count=6)
    assert vol(p) == pytest.approx(PI * 2500 * 8 - 6 * PI * 16 * 8, rel=1e-12)


def test_revolve_feature_about_a_sketch_line():
    scene, doc = doc_mm()
    sk = doc.sketch("Ring section", plane="XZ")
    sk.rectangle((20, 0), (10, 15))
    sk.line((0, -5), (0, 30), id="axis", construction=True)
    p = doc.part("Ring")
    p.revolve(sk, "axis", "360 deg", id="rev")
    assert vol(p) == pytest.approx(PI * (30**2 - 20**2) * 15, rel=1e-12)
    p.set("rev", angle="90 deg")
    assert vol(p) == pytest.approx(PI * (30**2 - 20**2) * 15 / 4, rel=1e-12)


def test_sweep_along_a_line_and_an_arc():
    scene, doc = doc_mm()
    path = doc.sketch("Path", plane="XY")
    path.line((0, 0), (50, 0), id="a")
    pid = path.arc((50, 30), (50, 0), (80, 30), id="b")
    del pid
    prof = doc.sketch("Section", plane={"origin": [0, 0, 0], "z": [1, 0, 0], "x": [0, 1, 0]})
    prof.circle((0, 0), 5)
    p = doc.part("Pipe")
    p.add_feature({"type": "sweep", "id": "sw", "profile": "Section", "path": "Path"})
    length = 50 + 30 * PI / 2
    assert vol(p) == pytest.approx(PI * 25 * length, rel=1e-10)


def test_ruled_lofts_are_frustums():
    scene, doc = doc_mm()
    a = doc.sketch("Bottom", plane="XY")
    a.rectangle((-20, -20), (40, 40))
    b = doc.sketch("Top", plane={"origin": [0, 0, 30], "z": [0, 0, 1], "x": [1, 0, 0]})
    b.rectangle((-10, -10), (20, 20))
    p = doc.part("Hopper")
    p.add_feature({"type": "loft", "id": "loft", "sections": ["Bottom", "Top"]})
    a1, a2 = 1600, 400
    assert vol(p) == pytest.approx(30 / 3 * (a1 + a2 + math.sqrt(a1 * a2)), rel=1e-10)
    c = doc.sketch("C1", plane="XY")
    c.circle((0, 0), 20)
    d = doc.sketch("C2", plane={"origin": [0, 0, 40], "z": [0, 0, 1], "x": [1, 0, 0]})
    d.circle((0, 0), 10)
    q = doc.part("Cone")
    q.add_feature({"type": "loft", "id": "loft", "sections": ["C1", "C2"]})
    assert vol(q) == pytest.approx(PI * 40 / 3 * (400 + 200 + 100), rel=1e-9)


def test_drafted_extrusion_volume():
    scene, doc = doc_mm()
    sk = doc.sketch("Boss")
    sk.rectangle((0, 0), (40, 20))
    p = doc.part("Boss part")
    p.extrude(sk, 10, draft="5 deg", id="base")
    k = math.tan(math.radians(5))
    expected = 40 * 20 * 10 - (40 + 20) * k * 10**2 + 4 * k * k * 10**3 / 3
    assert vol(p) == pytest.approx(expected, rel=1e-12)


def test_multibody_parts_and_body_booleans():
    scene, doc = doc_mm()
    p = plate(doc)
    p.add_feature(
        {
            "type": "box",
            "id": "blk",
            "size": [10, 10, 30],
            "frame": {"origin": [40, 20, 15]},
            "operation": "new",
            "body": "post",
        }
    )
    assert set(p.result.bodies) == {"main", "post"}
    p.add_feature(
        {"type": "boolean", "id": "join", "op": "union", "target": "main", "tool": "post"}
    )
    assert set(p.result.bodies) == {"main"}
    assert vol(p) == pytest.approx(32000 + 10 * 10 * 20, rel=1e-12)


def test_reference_errors_list_candidates():
    scene, doc = doc_mm()
    p = plate(doc, 100, 40, 10)
    p.hole("base:end", [[10, 20, 10], [90, 20, 10]], 6, id="h")
    with pytest.raises(CadError) as e:
        p.fillet([{"edge": ["base:end", "nothing"]}], 1)
    assert e.value.code == "REFERENCE_LOST" and "base:end" in e.value.hint
    # Moving a hole keeps its name; deleting the generating feature loses the reference.
    p.set("h", points=[[15, 20, 10], [85, 20, 10]])
    assert any(f["name"].startswith("h.0:") for f in p.faces())


def test_editing_an_early_feature_reuses_later_cached_work():
    scene, doc = doc_mm()
    p = plate(doc)
    p.hole("base:end", [[20, 20, 10]], 8, id="h1")
    p.fillet([{"edge": ["base:end", "base:side/r2"]}], 3, id="f")
    cache = doc.__dict__["_feature_cache"]
    n = len(cache)
    p.set("f", radius=2)  # only the last feature reruns
    assert len(cache) == n + 1
