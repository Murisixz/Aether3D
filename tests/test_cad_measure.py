"""Measurements of exact geometry (closed-form checks) and snapping in sketches."""

from __future__ import annotations

import math

import pytest

import aether3d
from aether3d.errors import CadError

pytest.importorskip("manifold3d")


def plate():
    scene = aether3d.Scene("measure")
    doc = scene.cad
    sk = doc.sketch("Profile")
    sk.rectangle((0, 0), (80, 40))
    p = doc.part("Plate")
    p.extrude(sk, 10, id="base")
    p.hole("base:end", [[20, 20, 10]], 8, id="h")
    p.hole("base:end", [[60, 20, 10]], 6, id="g")
    return doc


def test_single_entities_are_described_exactly():
    doc = plate()
    top = doc.measure({"part": "Plate", "face": "base:end"})
    assert top["surface"] == "plane" and top["normal"] == [0.0, 0.0, 1.0]
    assert top["area"] == pytest.approx(80 * 40 - math.pi * 16 - math.pi * 9, rel=1e-12)
    wall = doc.measure({"part": "Plate", "face": "h:rev/wall.0"})
    assert wall["surface"] == "cylinder" and wall["radius"] == pytest.approx(4.0, rel=1e-15)
    assert wall["area"] == pytest.approx(math.pi * 8 * 10 / 2, rel=1e-12)
    edge = doc.measure({"part": "Plate", "edge": ["base:end", "base:side/r1"]})
    assert edge == {
        "reference": "Plate/main:edge",
        "kind": "edge",
        "curve": "line",
        "length": pytest.approx(80.0, rel=1e-15),
    }


def test_relations_between_entities():
    doc = plate()
    m = doc.measure({"part": "Plate", "face": "base:end"}, {"part": "Plate", "face": "base:start"})
    assert m["angle_deg"] == 0.0 and m["parallel_distance"] == pytest.approx(10.0, rel=1e-14)
    assert m["distance"] == pytest.approx(10.0, rel=1e-14)
    m = doc.measure(
        {"part": "Plate", "face": "base:end"}, {"part": "Plate", "face": "base:side/r2"}
    )
    assert m["angle_deg"] == pytest.approx(90.0, abs=1e-12) and m["distance"] == 0.0
    # Two hole axes: 40 mm apart; the walls' closest points: 40 - 4 - 3.
    m = doc.measure(
        {"part": "Plate", "face": "h:rev/wall.0"}, {"part": "Plate", "face": "g:rev/wall.1"}
    )
    assert m["axis_distance"] == pytest.approx(40.0, rel=1e-14)
    assert m["distance"] == pytest.approx(40 - 4 - 3, abs=m["bound"])
    # A point above the plate: to the top face's plane and to the face itself.
    m = doc.measure({"point": [20, 20, 30]}, {"part": "Plate", "face": "base:end"})
    assert m["parallel_distance"] == pytest.approx(20.0, rel=1e-14)
    # Above the hole: the closest point of the (trimmed) top face is on the hole's rim.
    assert m["distance"] == pytest.approx(math.hypot(20, 4), rel=1e-9)
    m = doc.measure({"sketch": "Profile", "entity": "r1"}, {"sketch": "Profile", "entity": "r2"})
    assert m["angle_deg"] == pytest.approx(90.0, abs=1e-12)


def test_bad_references_are_refused():
    doc = plate()
    for ref, code in (
        ({"part": "Plate", "face": "nothing"}, "REFERENCE_LOST"),
        ({"part": "Plate"}, "INVALID_REFERENCE"),
        ({"sketch": "Profile", "entity": "zz"}, "UNKNOWN_ENTITY"),
        ("Plate", "INVALID_REFERENCE"),
        ({"point": [1, 2, 3, 4]}, "INVALID_VALUE"),
    ):
        with pytest.raises(CadError) as e:
            doc.measure(ref)
        assert e.value.code == code


def test_snapping_priorities_and_exact_points():
    doc = aether3d.Scene("snap").cad
    sk = doc.sketch("S")
    sk.rectangle((0, 0), (80, 40))
    sk.circle((20, 20), 4)
    sk.line((0, 0), (80, 40), id="diag")
    assert doc.snap("S", 79.5, 0.3) == {
        "kind": "endpoint",
        "point": [80.0, 0.0],
        "ref": "r1",
        "distance": pytest.approx(math.hypot(0.5, 0.3)),
    }
    assert doc.snap("S", 40.2, 20.1)["kind"] == "midpoint"
    q = doc.snap("S", 24.2, 20.1)
    assert q["kind"] == "quadrant" and q["point"] == [24.0, 20.0]
    # The diagonal crosses the right edge's top end and the circle nowhere; the
    # rectangle's sides meet the diagonal at the corners (end points win there).
    hit = doc.snap("S", 79.9, 39.8)
    assert hit["kind"] == "endpoint" and hit["point"] == [80.0, 40.0]
    g = doc.snap("S", 33, 17, grid=5, kinds=["grid"])
    assert g["kind"] == "grid" and g["point"] == [35.0, 15.0]
    assert doc.snap("S", 33, 17, kinds=["endpoint"]) is None
    with pytest.raises(CadError):
        doc.snap("S", 0, 0, kinds=["magic"])


def test_snapping_to_an_intersection():
    doc = aether3d.Scene("snap").cad
    sk = doc.sketch("S")
    sk.line((0, 0), (10, 10), id="a")
    sk.line((0, 10), (10, 0), id="b")
    hit = doc.snap("S", 5.1, 4.95)
    assert hit["kind"] == "intersection" and hit["point"] == pytest.approx([5.0, 5.0], abs=1e-15)
