"""STEP and IFC exchange: exact round trips, files written by another kernel (Open
CASCADE), untrusted input, and -- when an independent validator is installed --
validation by Open CASCADE and IfcOpenShell.

The validator tests run only when ``AETHER3D_VALIDATOR_PYTHON`` names a Python that has
``OCP`` (cadquery-ocp) and ``ifcopenshell``; neither is a dependency of Aether3D.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

import aether3d
from aether3d.cad.brep import build
from aether3d.cad.brep.properties import mass_properties
from aether3d.cad.brep.topology import validate
from aether3d.cad.interop.step import read_step, write_step
from aether3d.errors import CadError

pytest.importorskip("manifold3d")

DATA = Path(__file__).parent / "data" / "step"
PI = math.pi


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
    return scene, doc, p


def loft():
    scene = aether3d.Scene("loft")
    doc = scene.cad
    doc.sketch("C1").circle((0, 0), 20)
    doc.sketch("C2", plane={"origin": [0, 0, 40], "z": [0, 0, 1], "x": [1, 0, 0]}).circle(
        (0, 0), 10
    )
    p = doc.part("Loft")
    p.add_feature({"type": "loft", "id": "loft", "sections": ["C1", "C2"]})
    return p.body()


def _same_volume(a, b, rel=1e-12):
    va, vb = mass_properties(a).volume, mass_properties(b).volume
    assert vb == pytest.approx(va, rel=rel)


# --------------------------------------------------------------------------- #
# STEP
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("units", ["mm", "m", "in"])
def test_step_round_trip_is_exact_for_analytic_and_nurbs_faces(units):
    _, _, p = bracket()
    cases = {
        "bracket": p.body(),
        "sphere": build.sphere(0.02),
        "torus": build.torus(0.03, 0.01),
        "loft": loft(),
    }
    for name, body in cases.items():
        text = write_step([(name, body)], units=units, name=name)
        [(back_name, back)] = read_step(text)
        assert back_name == name
        assert validate(back)["valid"], name
        _same_volume(body, back, rel=1e-12)
        assert back.summary()["surfaces"] == body.summary()["surfaces"], name
        lo, hi = body.bounds
        blo, bhi = back.bounds
        assert max(abs(blo - lo).max(), abs(bhi - hi).max()) < 1e-12


CONE_H = 6 / math.tan(0.380506377112)


def test_step_reader_reads_open_cascade_files_exactly():
    # Volumes computed by Open CASCADE itself (tests/data/step/README.md).
    expected = {
        "occt_cylinder": (PI * 25 * 20, None),
        "occt_plate_hole": (80 * 40 * 10 - PI * 16 * 10, None),
        "occt_sphere": (4 / 3 * PI * 7**3, (1e-3, 2e-3, 3e-3)),
        "occt_hemisphere": (2 / 3 * PI * 10**3, (0.0, 0.0, -3.75e-3)),
        # The file stores the half angle to 12 digits (0.380506377112), which puts the
        # apex 1.6e-11 mm above 15 mm: the exact volume of the file's cone.
        "occt_cone": (PI * 36 * CONE_H / 3, (0.0, 0.0, CONE_H / 4 * 1e-3)),
    }
    for name, (volume_mm3, centroid) in expected.items():
        [(_, body)] = read_step((DATA / f"{name}.step").read_text())
        assert validate(body)["valid"], name
        m = mass_properties(body)
        assert m.volume * 1e9 == pytest.approx(volume_mm3, rel=1e-12), name
        if centroid is not None:
            assert m.centroid == pytest.approx(centroid, abs=1e-15)


def test_imported_seam_faces_tessellate_within_the_chord():
    from aether3d.cad.brep.tessellate import max_deviation, tessellate

    for name in ("occt_cylinder", "occt_plate_hole", "occt_sphere", "occt_hemisphere", "occt_cone"):
        [(_, body)] = read_step((DATA / f"{name}.step").read_text())
        t = tessellate(body)
        assert max_deviation(body, t) <= t.chord * 1.0001, name


def test_booleans_on_imported_bodies_are_exact():
    # Seam faces from another kernel become bands and caps without seam edges after a
    # boolean (their loops wrap around the axis); volumes stay closed-form exact.
    doc = aether3d.Scene("imported").cad
    [zone] = doc.import_step((DATA / "occt_hemisphere.step").read_text(), name="Zone")
    zone.add_feature(
        {
            "type": "box",
            "id": "cut",
            "size": [40, 40, 10],
            "operation": "cut",
            "frame": {"origin": [0, 0, -11]},
        }
    )  # keeps -6 <= z <= 0: a spherical zone
    assert zone.measure("main")["volume"] == pytest.approx(PI * (100 * 6 - 6**3 / 3), rel=1e-12)
    [cone] = doc.import_step((DATA / "occt_cone.step").read_text(), name="Frustum")
    cone.add_feature(
        {
            "type": "box",
            "id": "top",
            "size": [40, 40, 10],
            "operation": "cut",
            "frame": {"origin": [0, 0, 15]},
        }
    )  # keeps 0 <= z <= 10
    r2 = 6 * (1 - 10 / 15)
    assert cone.measure("main")["volume"] == pytest.approx(
        PI * 10 / 3 * (36 + 6 * r2 + r2 * r2), rel=1e-12
    )
    [bored] = doc.import_step((DATA / "occt_hemisphere.step").read_text(), name="Bored")
    bored.add_feature(
        {"type": "cylinder", "id": "bore", "radius": 2, "height": 40, "operation": "cut"}
    )  # through the pole: the sphere meets the bore in an exact circle
    h = math.sqrt(100 - 4)
    removed = PI * 4 * h + PI * (10 - h) ** 2 * (30 - (10 - h)) / 3
    m = bored.measure("main")
    assert m["valid"] and m["volume"] == pytest.approx(2 / 3 * PI * 1000 - removed, rel=1e-12)
    for part in (zone, cone, bored):
        assert part.measure("main")["valid"]
        assert len(doc.body_mesh(part.item, "main").triangles) > 0


def test_document_import_and_export_of_step_is_one_undoable_step():
    scene = aether3d.Scene("import")
    doc = scene.cad
    [part] = doc.import_step((DATA / "occt_plate_hole.step").read_text(), name="Plate")
    assert part.name == "Plate"
    m = part.measure("main")
    assert m["valid"] and m["volume"] == pytest.approx(80 * 40 * 10 - PI * 16 * 10, rel=1e-12)
    # An imported solid is a feature like any other: holes can be added to it.
    part.add_feature(
        {
            "type": "cylinder",
            "id": "pin",
            "radius": 3,
            "height": 30,
            "operation": "cut",
            "frame": {"origin": [60, 20, 5]},
        }
    )
    assert part.measure("main")["volume"] == pytest.approx(
        80 * 40 * 10 - PI * 16 * 10 - PI * 9 * 10, rel=1e-12
    )
    text = doc.export_step()
    [(_, back)] = read_step(text)
    assert mass_properties(back).volume * 1e9 == pytest.approx(part.measure("main")["volume"])
    scene.history.undo()
    scene.history.undo()
    assert doc.items("part") == []


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("hello", "INVALID_STEP"),
        ("ISO-10303-21;\nHEADER;\nENDSEC;\n", "INVALID_STEP"),
        ("ISO-10303-21;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.));\n", "INVALID_STEP"),
        (
            "ISO-10303-21;\nDATA;\n#1=FOO((((((((((((((((((((((((((((((((((((((((((((((((((("
            "(((((((((((((((((((((((((((((((((((((((((((((((((((1)));\nENDSEC;\n",
            "INVALID_STEP",
        ),
        ("ISO-10303-21;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\n", "UNSUPPORTED_STEP"),
        ("ISO-10303-21;\nDATA;\n#1=MANIFOLD_SOLID_BREP('',#2);\nENDSEC;\n", "INVALID_STEP"),
        # A reference cycle is caught by the entity type checks before it is followed.
        ("ISO-10303-21;\nDATA;\n#1=MANIFOLD_SOLID_BREP('',#1);\nENDSEC;\n", "UNSUPPORTED_STEP"),
        (
            "ISO-10303-21;\nDATA;\n#1=NEXT_ASSEMBLY_USAGE_OCCURRENCE('');\nENDSEC;\n",
            "UNSUPPORTED_STEP",
        ),
        ("ISO-10303-21;\nDATA;\n#1=CARTESIAN_POINT('',(1E999,0.,0.));\nENDSEC;\n", "INVALID_STEP"),
    ],
)
def test_untrusted_step_text_is_refused_with_structured_errors(text, code):
    with pytest.raises(CadError) as e:
        read_step(text)
    assert e.value.code == code


def test_step_size_limit():
    from aether3d.cad.interop import step

    with pytest.raises(CadError) as e:
        read_step("ISO-10303-21;" + " " * (step.MAX_FILE_BYTES + 1))
    assert e.value.code == "LIMIT_EXCEEDED"


def test_solid_feature_refuses_invalid_breps():
    scene = aether3d.Scene("bad")
    doc = scene.cad
    good = build.box([0.01, 0.01, 0.01]).to_dict()
    broken = json.loads(json.dumps(good))
    broken["faces"] = broken["faces"][:-1]  # an open shell
    for brep, code in ((broken, "INVALID_GEOMETRY"), ({"edges": 3}, "INVALID_VALUE")):
        with pytest.raises(CadError) as e:
            doc.add("part", "P", {"features": [{"type": "solid", "id": "s", "brep": brep}]})
        assert e.value.code == code
        assert doc.items("part") == []


# --------------------------------------------------------------------------- #
# IFC
# --------------------------------------------------------------------------- #


def full_house():
    from test_arch import house

    scene, doc, b, g = house()
    b.slab("Ground", outline=g, thickness=0.25, name="Slab")
    b.roof(
        "First",
        outline=g,
        type="gable",
        slope="30 deg",
        overhang=0.5,
        ridge="x",
        thickness=0.2,
        name="Roof",
    )
    b.stair(
        "Ground",
        "First",
        start=[1, 1],
        direction=[0, 1],
        width=0.9,
        tread=0.28,
        riser=0.175,
        name="Stair",
    )
    b.columns(
        "Ground",
        to="First",
        profile={"shape": "circle", "diameter": 0.3},
        grid={"x": [12, 15], "y": [0, 4]},
        name="Cols",
    )
    b.beams(
        "First",
        [[[12, 0], [12, 4]]],
        profile={"shape": "I", "width": 0.2, "depth": 0.4, "web": 0.01, "flange": 0.015},
        name="Beams",
    )
    return scene, doc


def test_ifc_export_round_trips_geometry_quantities_and_ids():
    scene, doc = full_house()
    text, stats = doc.export_ifc(name="House")
    assert stats == {
        "IfcWall": 5,
        "IfcOpeningElement": 4,
        "IfcDoor": 2,
        "IfcWindow": 2,
        "IfcSpace": 2,
        "IfcSlab": 1,
        "IfcRoof": 1,
        "IfcStair": 1,
        "IfcColumn": 4,
        "IfcBeam": 1,
    }
    # Deterministic: the same document writes the same entities (only the time stamp
    # in the header differs).
    again, _ = doc.export_ifc(name="House")
    assert text.split("DATA;")[1] == again.split("DATA;")[1]
    elements = doc.import_ifc(text, create_parts=False)
    ids = [e.global_id for e in elements]
    assert len(ids) == len(set(ids)) and all(len(i) == 22 for i in ids)
    floor = doc.result("House Ground")
    by_name = {(e.ifc_class, e.name): e for e in elements}
    for wid, wr in floor.walls.items():
        e = by_name[("IFCWALL", wid)]
        q = e.properties["Qto_WallBaseQuantities"]
        # The imported solid (extrusion minus openings) has the exported net volume.
        assert mass_properties(e.body).volume == pytest.approx(q["NetVolume"], rel=1e-12)
        assert e.properties["Pset_WallCommon"]["IsExternal"] is bool(wr.exterior)
        assert e.storey == "Ground"
    for name in ("Slab", "Roof", "Stair"):
        [e] = [x for x in elements if x.name == name]
        body = doc.result(name).bodies[0][1]
        assert mass_properties(e.body).volume == pytest.approx(
            mass_properties(body).volume, rel=1e-12
        )
    cols = [e for e in elements if e.ifc_class == "IFCCOLUMN"]
    assert len(cols) == 4
    for e in cols:
        assert mass_properties(e.body).volume == pytest.approx(PI * 0.15**2 * 3.0, rel=1e-12)
    [beam] = [e for e in elements if e.ifc_class == "IFCBEAM"]
    i_area = 2 * 0.2 * 0.015 + (0.4 - 0.03) * 0.01
    assert mass_properties(beam.body).volume == pytest.approx(i_area * 4.0, rel=1e-12)


def test_ifc_import_creates_parts_and_refuses_other_schemas():
    scene, doc = full_house()
    text, _ = doc.export_ifc()
    other = aether3d.Scene("other").cad
    elements = other.import_ifc(text)
    parts = other.items("part")
    assert len(parts) == sum(1 for e in elements if e.body is not None) == 19
    assert parts[0]._definition["source"]["format"] == "ifc"
    with pytest.raises(CadError) as e:
        other.import_ifc(text.replace("FILE_SCHEMA(('IFC4'))", "FILE_SCHEMA(('IFC2X3'))"))
    assert e.value.code == "UNSUPPORTED_IFC"
    with pytest.raises(CadError) as e:
        other.import_ifc("not ifc")
    assert e.value.code == "INVALID_IFC"


# --------------------------------------------------------------------------- #
# Independent validators (optional)
# --------------------------------------------------------------------------- #

VALIDATOR = os.environ.get("AETHER3D_VALIDATOR_PYTHON")
needs_validator = pytest.mark.skipif(
    not VALIDATOR, reason="set AETHER3D_VALIDATOR_PYTHON to a Python with OCP and ifcopenshell"
)

_OCCT_CHECK = """
import json, sys
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
r = STEPControl_Reader()
assert r.ReadFile(sys.argv[1]) == 1
r.TransferRoots()
s = r.OneShape()
p = GProp_GProps()
BRepGProp.VolumeProperties_s(s, p, 1e-12)
print(json.dumps({"valid": bool(BRepCheck_Analyzer(s).IsValid()), "volume": p.Mass()}))
"""

_IFC_CHECK = """
import json, sys
import ifcopenshell, ifcopenshell.validate
f = ifcopenshell.open(sys.argv[1])
log = ifcopenshell.validate.json_logger()
ifcopenshell.validate.validate(f, log, express_rules=True)
counts = {t: len(f.by_type(t)) for t in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSpace")}
print(json.dumps({"issues": [str(s.get("message"))[:200] for s in log.statements],
                  "counts": counts}))
"""


def _run(script: str, path: Path) -> dict:
    out = subprocess.run(
        [VALIDATOR, "-c", script, str(path)], capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@needs_validator
def test_open_cascade_accepts_exported_step(tmp_path):
    _, _, p = bracket()
    for name, body in (("bracket", p.body()), ("torus", build.torus(0.03, 0.01)), ("loft", loft())):
        path = tmp_path / f"{name}.step"
        path.write_text(write_step([(name, body)], units="mm"))
        r = _run(_OCCT_CHECK, path)
        assert r["valid"], name
        assert r["volume"] == pytest.approx(mass_properties(body).volume * 1e9, rel=1e-9)


@needs_validator
def test_ifcopenshell_validates_exported_ifc(tmp_path):
    scene, doc = full_house()
    text, stats = doc.export_ifc()
    path = tmp_path / "house.ifc"
    path.write_text(text)
    r = _run(_IFC_CHECK, path)
    assert r["issues"] == []
    assert r["counts"] == {t: stats[t] for t in r["counts"]}
