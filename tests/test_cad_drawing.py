"""Drawings (plans, sections, elevations, isometric views; SVG, DXF and PNG) and DXF
sketch exchange, against closed-form geometry; optional ezdxf audit of the output."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import aether3d
from aether3d.cad.interop.dxf import read_dxf
from aether3d.errors import CadError

pytest.importorskip("manifold3d")

PI = math.pi
DATA = Path(__file__).parent / "data" / "dxf"


def bracket():
    scene = aether3d.Scene("bracket")
    doc = scene.cad
    sk = doc.sketch("P")
    sk.rectangle((0, 0), (80, 40))
    p = doc.part("Bracket")
    p.extrude(sk, 10, id="base")
    p.hole("base:end", [[20, 20, 10]], 8, kind="counterbore", cbore_diameter=14, cbore_depth=4)
    p.hole("base:end", [[60, 20, 10]], 8, kind="countersink", csink_diameter=16)
    p.fillet([{"edge": ["base:end", "base:side/r2"]}], 3)
    return scene, doc


def _area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _hatch_area(h):
    return abs(_area(h.loops[0])) - sum(abs(_area(lp)) for lp in h.loops[1:])


def test_top_view_draws_exact_circles_and_front_view_hidden_lines():
    scene, doc = bracket()
    doc.drawing("Top", "top", ["Bracket"])
    top = doc.result("Top")
    # Every point of the hole outlines lies on a circle about a hole centre (the curves
    # are exact, sampled): radii 4, 7 (counterbore) and 8 (countersink) mm.
    radii = set()
    for pl in top.polylines:
        for c in ((0.02, 0.02), (0.06, 0.02)):
            r = np.hypot(pl.points[:, 0] - c[0], pl.points[:, 1] - c[1])
            if r.max() - r.min() < 1e-12 and r.max() < 0.01:
                radii.add(round(float(r.mean()) * 1000, 9))
    assert {4.0, 7.0, 8.0} <= radii
    assert [d.text for d in top.dimensions] == ["80 mm", "40 mm"]
    doc.drawing("Front", "front", ["Bracket"])
    front = doc.result("Front")
    hidden = [p for p in front.polylines if p.style == "hidden"]
    # The holes' walls are behind the front face: hidden vertical segments at their
    # silhouettes (a countersink's continues up its cone, in one polyline).
    xs = set()
    for p in hidden:
        d = np.diff(p.points, axis=0)
        for k in np.flatnonzero((np.abs(d[:, 0]) < 1e-12) & (np.abs(d[:, 1]) > 0)):
            xs.add(round(float(p.points[k, 0]) * 1000, 6))
    for x in (16, 24, 13, 27, 56, 64):
        assert any(abs(v - x) < 1e-6 for v in xs), (x, xs)


def test_section_fill_is_the_exact_cut_area():
    scene, doc = bracket()
    doc.drawing("A-A", "section", ["Bracket"], plane={"origin": [0, 20, 0], "normal": [0, -1, 0]})
    d = doc.result("A-A")
    area = sum(_hatch_area(h) for h in d.hatches) * 1e6
    cbore = 14 * 4 + 8 * 6
    h = (16 - 8) / 2 / math.tan(PI / 4)
    csink = (16 + 8) / 2 * h + 8 * (10 - h)
    fillet = (1 - PI / 4) * 9
    expected = 80 * 10 - cbore - csink - fillet
    # The fillet's quarter circle is sampled within the drawing's chord.
    assert area == pytest.approx(expected, rel=2e-4)
    assert any(p.style == "section" for p in d.polylines)
    assert d.notes == []


def test_isometric_view_hides_back_edges_and_has_no_auto_dimensions():
    scene, doc = bracket()
    doc.drawing("Iso", "iso", ["Bracket"])
    d = doc.result("Iso")
    assert d.dimensions == [] and all(p.style == "visible" for p in d.polylines)
    # The bottom face's far corner (0, 40, 0) is hidden: no visible line ends there.
    from aether3d.cad.drawing.views import ViewFrame

    corner = ViewFrame.named("iso").project(np.array([0.0, 0.04, 0.0]))[0]
    ends = np.concatenate([p.points[[0, -1]] for p in d.polylines])
    assert np.min(np.linalg.norm(ends - corner, axis=1)) > 1e-4


def house():
    from test_cad_interop import full_house

    return full_house()


def test_plan_hatches_walls_cut_through_openings_and_labels_rooms():
    scene, doc = house()
    doc.drawing("Ground plan", "plan", level="Ground")
    d = doc.result("Ground plan")
    floor = doc.result("House Ground")
    gross = sum(float(abs(_area(w.footprint))) for w in floor.walls.values())
    # At 1.2 m the cut passes through every door and window opening.
    open_area = sum(o.width * floor.walls[o.wall].line.thickness for o in floor.openings.values())
    walls = sum(_hatch_area(h) for h in d.hatches if h.layer == "section-wall")
    assert walls == pytest.approx(gross - open_area, rel=1e-12)
    texts = [t.text for t in d.texts]
    assert "Living" in texts and "Bedroom" in texts
    assert any(t.startswith("001") and t.endswith("m²") for t in texts)
    swings = [p for p in d.polylines if p.layer == "door-swing"]
    assert len(swings) == 2 * 2  # a leaf and an arc per door
    assert [x.text for x in d.dimensions] == ["10.3 m", "8.3 m"]
    assert d.notes == []


def test_drawings_follow_model_edits_and_undo():
    scene, doc = house()
    doc.drawing("Ground plan", "plan", level="Ground")
    before = doc.result("Ground plan")
    arcs0 = [p.points for p in before.polylines if p.layer == "door-swing" and len(p.points) > 2]
    floor = doc.item("House Ground")
    defn = dict(floor._definition)
    defn["openings"] = [
        {**o, "position": 5.0} if o.get("id") == "front" else o for o in defn["openings"]
    ]
    doc.set_definition(floor, defn)
    after = doc.result("Ground plan")
    arcs1 = [p.points for p in after.polylines if p.layer == "door-swing" and len(p.points) > 2]
    shift = {round(float(np.mean(b[:, 0] - a[:, 0])), 9) for a, b in zip(arcs0, arcs1, strict=True)}
    assert shift == {0.0, 2.0}
    scene.history.undo()
    again = doc.result("Ground plan")
    arcs2 = [p.points for p in again.polylines if p.layer == "door-swing" and len(p.points) > 2]
    assert all(np.array_equal(a, b) for a, b in zip(arcs0, arcs2, strict=True))


def test_outputs_are_valid_and_deterministic():
    scene, doc = bracket()
    doc.drawing("Top", "top", ["Bracket"])
    d = doc.result("Top")
    svg = d.to_svg()
    root = ET.fromstring(svg)
    assert root.get("width") == "420mm" and root.get("viewBox") == "0 0 420 297"
    dxf = d.to_dxf("mm")
    g = read_dxf(dxf)
    assert g.unit_m == 0.001 and g.count() > 0
    # Real size: the plate's outline spans 80 x 40 mm.
    pts = np.concatenate([np.array([a, b]) for a, b in g.lines])
    assert np.ptp(pts[:, 0]) >= 80 - 1e-9
    a = d.to_png(60)
    b = doc.result("Top").to_png(60)
    assert a.dtype == np.uint8 and a.shape == (
        int(math.ceil(297 / 25.4 * 60)),
        int(math.ceil(420 / 25.4 * 60)),
        3,
    )
    assert hashlib.sha256(a.tobytes()).digest() == hashlib.sha256(b.tobytes()).digest()
    assert a.min() == 0 and a.max() == 255


def test_drawing_definitions_are_validated():
    scene, doc = bracket()
    for bad, code in (
        ({"view": "perspective", "items": ["Bracket"]}, "INVALID_VALUE"),
        ({"view": "top", "items": []}, "INVALID_VALUE"),
        ({"view": "top", "items": ["P"]}, "INVALID_VALUE"),
        ({"view": "top", "items": ["Bracket"], "scale": "1:0"}, "INVALID_VALUE"),
        ({"view": "section", "items": ["Bracket"]}, "INVALID_VALUE"),
        ({"view": "top", "items": ["Nope"]}, "UNKNOWN_ITEM"),
    ):
        with pytest.raises(CadError) as e:
            doc.add("drawing", "D", bad)
        assert e.value.code == code, bad
    assert doc.items("drawing") == []


# --------------------------------------------------------------------------- #
# DXF sketches
# --------------------------------------------------------------------------- #


def test_dxf_written_by_ezdxf_imports_exactly():
    # Written by ezdxf (tests/data/dxf/README.md): bulged LWPOLYLINE corners, a mirrored
    # (extrusion -Z) arc, text and a block reference (skipped, and reported).
    doc = aether3d.Scene("dxf").cad
    sk = doc.import_dxf((DATA / "ezdxf_plate.dxf").read_text(), "Plate")
    assert sk.import_report["skipped"] == {"TEXT": 1, "INSERT": 1}
    [region] = doc.result("Plate").regions(None)
    plate = 100 * 60 - (4 - PI) * 100
    holes = PI * 64 + 20 * 10 + PI * 25
    assert region.area * 1e6 == pytest.approx(plate - holes, rel=1e-13)
    only = doc.import_dxf((DATA / "ezdxf_plate.dxf").read_text(), "Outline", layers=["OUTLINE"])
    assert only.import_report["layers"] == ["OUTLINE"]
    [outline] = doc.result("Outline").regions(None)
    assert outline.area * 1e6 == pytest.approx(plate, rel=1e-13)


def test_sketch_dxf_round_trip():
    doc = aether3d.Scene("rt").cad
    sk = doc.sketch("S")
    sk.rectangle((0, 0), (80, 40))
    sk.circle((20, 20), 4)
    for unit in ("mm", "in", "m"):
        text = doc.export_sketch_dxf("S", unit)
        back = doc.import_dxf(text, f"S {unit}")
        [a] = doc.result("S").regions(None)
        [b] = doc.result(back.name).regions(None)
        assert b.area == pytest.approx(a.area, rel=1e-12)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("0\nSECTION\n2\nENTITIES\nx\nLINE\n", "INVALID_DXF"),
        (
            "0\nSECTION\n2\nENTITIES\n0\nLINE\n10\nnan\n20\n0\n11\n1\n21\n1\n0\nENDSEC\n",
            "INVALID_DXF",
        ),
        ("0\nSECTION\n2\nENTITIES\n0\nCIRCLE\n10\n0\n20\n0\n40\n-1\n0\nENDSEC\n", "INVALID_DXF"),
        ("0\nSECTION\n2\nHEADER\n0\nENDSEC\n", "INVALID_DXF"),
        ("0\nSECTION\n2\nENTITIES\n0\nTEXT\n1\nhi\n0\nENDSEC\n", "UNIT_REQUIRED"),
        (
            "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n"
            "0\nTEXT\n1\nhi\n0\nENDSEC\n",
            "NOTHING_TO_IMPORT",
        ),
    ],
)
def test_untrusted_dxf_is_refused_with_structured_errors(text, code):
    doc = aether3d.Scene("bad").cad
    with pytest.raises(CadError) as e:
        doc.import_dxf(text, "S")
    assert e.value.code == code
    assert doc.items("sketch") == []


VALIDATOR = os.environ.get("AETHER3D_VALIDATOR_PYTHON")

_AUDIT = """
import json, sys
import ezdxf
doc = ezdxf.readfile(sys.argv[1])
auditor = doc.audit()
print(json.dumps({"errors": len(auditor.errors), "fixes": len(auditor.fixes),
                  "entities": len(doc.modelspace())}))
"""


@pytest.mark.skipif(not VALIDATOR, reason="set AETHER3D_VALIDATOR_PYTHON (with ezdxf)")
def test_ezdxf_audits_drawing_and_sketch_dxf(tmp_path):
    import json

    scene, doc = house()
    doc.drawing("Plan", "plan", level="Ground")
    files = {"plan.dxf": doc.result("Plan").to_dxf("mm")}
    s2 = aether3d.Scene("s").cad
    sk = s2.sketch("S")
    sk.rectangle((0, 0), (80, 40))
    sk.circle((20, 20), 4)
    files["sketch.dxf"] = s2.export_sketch_dxf("S", "mm")
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text)
        out = subprocess.run(
            [VALIDATOR, "-c", _AUDIT, str(path)], capture_output=True, text=True, timeout=300
        )
        assert out.returncode == 0, out.stderr[-2000:]
        r = json.loads(out.stdout.strip().splitlines()[-1])
        assert r["errors"] == 0 and r["entities"] > 0, (name, r)
