"""Architecture: wall joins, openings, rooms, slabs, roofs, stairs, columns, beams,
quantities against independent arithmetic, incremental edits and sketch-driven plans."""

from __future__ import annotations

import math

import numpy as np
import pytest

import aether3d
from aether3d.arch.quantities import schedule, to_csv, wall_quantities
from aether3d.arch.walls import WallLine, footprints
from aether3d.cad.brep.properties import mass_properties
from aether3d.cad.brep.topology import validate
from aether3d.errors import CadError

pytest.importorskip("manifold3d")


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


# --------------------------------------------------------------------------- #
# Joins in plan
# --------------------------------------------------------------------------- #


def test_l_join_tiles_the_corner_for_equal_and_unequal_thicknesses():
    for t1, t2 in ((0.3, 0.3), (0.3, 0.12)):
        walls = [
            WallLine("a", np.array([0.0, 0]), np.array([5.0, 0]), t1),
            WallLine("b", np.array([5.0, 0]), np.array([5.0, 4]), t2),
        ]
        fp = footprints(walls, 1e-9)
        # The footprints tile the L outline (no gap, no overlap): its area is
        # (5 + t2/2) t1 + (4 + t1/2) t2 - t1 t2 = 5 t1 + 4 t2.
        total = _area(fp["a"].polygon) + _area(fp["b"].polygon)
        assert total == pytest.approx(5 * t1 + 4 * t2, rel=1e-12)
        # The shared miter: both footprints contain the corner points.
        corner_out = np.array([5 + t2 / 2, -t1 / 2])
        assert min(np.linalg.norm(fp["a"].polygon - corner_out, axis=1)) < 1e-12
        assert min(np.linalg.norm(fp["b"].polygon - corner_out, axis=1)) < 1e-12


def test_junction_of_three_walls_and_t_join():
    walls = [
        WallLine("a", np.array([0.0, 0]), np.array([4.0, 0]), 0.2),
        WallLine("b", np.array([4.0, 0]), np.array([8.0, 0]), 0.2),
        WallLine("c", np.array([4.0, 0]), np.array([4.0, 3]), 0.1),
    ]
    fp = footprints(walls, 1e-9)
    total = sum(_area(fp[w].polygon) for w in "abc")
    # Union: the long wall (8.0 + square ends) and the branch above it.
    assert total == pytest.approx(8.0 * 0.2 + (3 - 0.1) * 0.1, rel=1e-12)
    # T: c ends on the middle of a long wall.
    walls = [
        WallLine("long", np.array([0.0, 0]), np.array([8.0, 0]), 0.3),
        WallLine("t", np.array([4.0, 3]), np.array([4.0, 0]), 0.1),
    ]
    fp = footprints(walls, 1e-9)
    assert _area(fp["t"].polygon) == pytest.approx((3 - 0.15) * 0.1, rel=1e-12)
    assert _area(fp["long"].polygon) == pytest.approx(8 * 0.3, rel=1e-12)
    assert fp["t"].joins[0]["kind"] == "T"


def test_crossing_walls_are_reported():
    walls = [
        WallLine("a", np.array([0.0, 0]), np.array([4.0, 0]), 0.2),
        WallLine("b", np.array([2.0, -1]), np.array([2.0, 1]), 0.2),
    ]
    with pytest.raises(CadError) as e:
        footprints(walls, 1e-9)
    assert e.value.code == "WALLS_CROSS"


# --------------------------------------------------------------------------- #
# A small house
# --------------------------------------------------------------------------- #


def house(scene=None):
    scene = scene or aether3d.Scene("house")
    doc = scene.cad
    doc.set_units(length="m")
    with doc.batch("House"):
        b = doc.building(
            "House",
            levels=[("Ground", 0, 3.0), ("First", 3.0, 3.0)],
            wall_types={
                "Ext": {"thickness": "300 mm", "function": "exterior"},
                "Int": {"thickness": "120 mm"},
            },
            families={
                "D90": {"kind": "door", "width": "900 mm", "height": "2100 mm"},
                "W12": {
                    "kind": "window",
                    "width": "1200 mm",
                    "height": "1400 mm",
                    "sill": "900 mm",
                    "mullions": 1,
                },
            },
            criteria={
                "riser_min": "150 mm",
                "riser_max": "190 mm",
                "tread_min": "250 mm",
                "blondel_min": "600 mm",
                "blondel_max": "650 mm",
            },
        )
        g = b.floor("Ground")
        ext = g.walls([(0, 0), (10, 0), (10, 8), (0, 8)], type="Ext")
        g.wall((6, 0), (6, 8), type="Int", id="mid")
        g.door(ext[0], "D90", position=3.0, id="front")
        g.window(ext[1], "W12", position=4.0, id="east")
        g.window(ext[2], "W12", position=2.0, id="north")
        g.door("mid", "D90", position=4.0, id="inner")
        g.room("Living", (3, 4), number="001", id="living")
        g.room("Bedroom", (8, 4), number="002", id="bed")
    return scene, doc, b, g


def test_walls_quantities_equal_their_solids():
    scene, doc, b, g = house()
    r = g.result
    for w in r.walls.values():
        assert validate(w.body)["valid"]
        q = wall_quantities(w, r.openings)
        assert mass_properties(w.body).volume == pytest.approx(q["NetVolume"], rel=1e-12)
    w1 = r.walls["w1"]
    q = wall_quantities(w1, r.openings)
    # 10 m wall, 300 mm, mitered at both ends: footprint (10.3 + 9.7) / 2 * 0.3 = 3.0 m2.
    assert q["GrossFootprintArea"] == pytest.approx(3.0, rel=1e-12)
    assert q["GrossVolume"] == pytest.approx(9.0, rel=1e-12)
    assert q["NetVolume"] == pytest.approx(9.0 - 0.9 * 2.1 * 0.3, rel=1e-12)
    assert q["NetSideArea"] == pytest.approx(30.0 - 0.9 * 2.1, rel=1e-12)


def test_room_areas_are_net_of_wall_faces():
    scene, doc, b, g = house()
    rooms = g.result.rooms
    assert rooms["living"].area == pytest.approx((6 - 0.15 - 0.06) * 7.7, rel=1e-12)
    assert rooms["bed"].area == pytest.approx((10 - 0.15 - 6 - 0.06) * 7.7, rel=1e-12)
    assert rooms["living"].perimeter == pytest.approx(2 * (5.79 + 7.7), rel=1e-12)


def test_moving_a_wall_moves_its_doors_and_changes_the_rooms():
    scene, doc, b, g = house()
    obj = doc.show(g)
    door_before = g.result.openings["inner"].placement.copy()
    g.set("mid", start=[5, 0], end=[5, 8])
    rooms = g.result.rooms
    assert rooms["living"].area == pytest.approx((5 - 0.15 - 0.06) * 7.7, rel=1e-12)
    door_after = g.result.openings["inner"].placement
    np.testing.assert_allclose(door_after[:3, 3] - door_before[:3, 3], [-1, 0, 0], atol=1e-12)
    assert obj.children  # walls and fillers shown as children
    scene.history.undo()
    assert g.result.rooms["living"].area == pytest.approx((6 - 0.15 - 0.06) * 7.7, rel=1e-12)


def test_invalid_openings_are_refused_with_codes():
    scene, doc, b, g = house()
    with pytest.raises(CadError) as e:
        g.window("w1", "W12", position=3.2)
    assert e.value.code == "OPENING_OVERLAP"
    with pytest.raises(CadError) as e:
        g.door("w1", "D90", position=9.8)
    assert e.value.code == "OPENING_OUTSIDE_WALL"
    with pytest.raises(CadError) as e:
        g.window("w2", "W12", position=4.0, height="2.5 m")
    assert e.value.code in ("OPENING_TOO_TALL", "OPENING_OVERLAP")
    with pytest.raises(CadError) as e:
        g.room("Nowhere", (20, 20))
    assert e.value.code == "ROOM_NOT_ENCLOSED"


def test_slab_roof_stair_columns_and_beams():
    scene, doc, b, g = house()
    slab = b.slab("Ground", outline=g, thickness=0.25, name="Slab")
    r = doc.result(slab)
    assert r.quantities["GrossArea"] == pytest.approx(10.3 * 8.3, rel=1e-12)
    assert mass_properties(r.bodies[0][1]).volume == pytest.approx(10.3 * 8.3 * 0.25, rel=1e-12)
    roof = b.roof(
        "First",
        outline=g,
        type="gable",
        slope="30 deg",
        overhang=0.5,
        ridge="x",
        thickness=0.2,
        name="Roof",
    )
    rr = doc.result(roof)
    assert validate(rr.bodies[0][1])["valid"]
    w, d = 10.3 + 1.0, 8.3 + 1.0
    # A gable shell of vertical thickness t / cos(a): two sloped slabs.
    assert rr.quantities["ProjectedArea"] == pytest.approx(w * d, rel=1e-12)
    assert rr.quantities["GrossArea"] == pytest.approx(w * d / math.cos(math.radians(30)), rel=1e-9)
    hip = b.roof("First", outline=g, type="hip", slope="30 deg", thickness=0.2, name="Hip")
    assert validate(doc.result(hip).bodies[0][1])["valid"]
    st = b.stair(
        "Ground",
        "First",
        start=[1, 1],
        direction=[0, 1],
        width=0.9,
        tread=0.28,
        riser=0.175,
        name="Stair",
    )
    q = doc.result(st).quantities
    assert q["NumberOfRisers"] == 17 and q["RiserHeight"] == pytest.approx(3 / 17)
    assert doc.result(st).info["criteria_met"]
    with pytest.raises(CadError) as e:
        b.stair(
            "Ground",
            "First",
            start=[2, 1],
            direction=[0, 1],
            tread=0.2,
            riser=0.25,
            enforce=True,
            name="Steep",
        )
    assert e.value.code == "STAIR_CRITERIA"
    cols = b.columns(
        "Ground",
        to="First",
        profile={"shape": "rect", "width": 0.3, "depth": 0.3},
        grid={"x": [12, 15], "y": [0, 4, 8]},
        name="Cols",
    )
    assert doc.result(cols).quantities["NetVolume"] == pytest.approx(6 * 0.09 * 3.0, rel=1e-12)
    obj = doc.show(cols)
    meshes = {id(c.mesh) for c in obj.children}
    assert len(obj.children) == 6 and len(meshes) == 1  # one shared mesh
    beams = b.beams(
        "First",
        [[[12, 0], [12, 8]], [[15, 0], [15, 8]]],
        profile={"shape": "I", "width": 0.2, "depth": 0.4, "web": 0.01, "flange": 0.015},
        name="Beams",
    )
    area = 2 * 0.2 * 0.015 + 0.01 * (0.4 - 0.03)
    assert doc.result(beams).quantities["NetVolume"] == pytest.approx(area * 16, rel=1e-12)


def test_schedules_and_csv():
    scene, doc, b, g = house()
    rows = schedule(doc)
    kinds = [r["element"] for r in rows]
    assert kinds.count("wall") == 5 and kinds.count("door") == 2 and kinds.count("window") == 2
    assert kinds.count("space") == 2
    text = to_csv(rows)
    assert text.splitlines()[0].startswith("element,")
    assert len(text.splitlines()) == len(rows) + 1


def test_floor_plan_driven_by_a_dimensioned_sketch():
    scene = aether3d.Scene("plan")
    doc = scene.cad
    doc.set_units(length="m")
    with doc.batch("Plan"):
        doc.parameter("bedroom_width", "4 m", dimension="length")
        sk = doc.sketch("Plan")
        outer = sk.rectangle((0, 0), (10, 8), prefix="o")
        sk.constrain("fix", "p1")
        sk.dimension("length", outer[0], value="10 m", id="L")
        sk.dimension("length", outer[1], value="8 m", id="D")
        mid = sk.line((6, 0), (6, 8), id="m")
        sk.constrain("vertical", mid)
        sk.constrain("point_on", "p5", outer[0])
        sk.constrain("point_on", "p6", outer[2])
        sk.dimension("distance", "p5", "p2", value="=bedroom_width", id="bw", along="x")
        b = doc.building("Flat", levels=[("L0", 0, 2.8)], wall_types={"W": {"thickness": "200 mm"}})
        f = b.floor("L0")
        f.walls_from_sketch("Plan")
        f.room("Bedroom", (8, 4), id="bed")
    assert f.result.rooms["bed"].area == pytest.approx((4 - 0.1 - 0.1) * (8 - 0.2), rel=1e-12)
    doc.parameter("bedroom_width", "5 m", dimension="length")
    assert f.result.rooms["bed"].area == pytest.approx((5 - 0.2) * 7.8, rel=1e-12)
    assert sk.measure("L") == pytest.approx(10.0)
