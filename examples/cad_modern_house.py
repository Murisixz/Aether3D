"""A two-storey modern house: levels, a cantilever on pilotis, a stair, by Aether3D.

Run from the repository root:

    python examples/cad_modern_house.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

The ground floor (10 x 8 m) carries a first floor that cantilevers 4 m to the east onto
two pilotis; a flat roof with an overhang; slabs (the first one with the stair's
opening); a straight stair checked against design criteria *given here* (riser 150-190
mm, tread at least 250 mm, 2R + T 600-650 mm -- no building code is assumed or claimed).

Checks (exact, from the B-reps and parameters):

* the stair meets the stated criteria, and the opening leaves at least 2.0 m of headroom
  above the stair's nosing line where the slab begins, and ends where the stair arrives;
* the pilotis stop exactly at the underside of the first slab (no overlap, no gap);
* the gross floor areas of the two levels are the outsides of their exterior walls;
* every product of the IFC4 export, read back, has its B-rep's volume.

Writes ``cad_modern_house.png`` (a render above the first floor plan, a section through
the stair and the south elevation), ``cad_modern_house.ifc``, the drawings as SVG and
``cad_modern_house.json``. Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cad_common import (  # noqa: E402
    caption,
    column,
    daylight,
    drawing_image,
    fit_width,
    ground,
    meshes_of,
    render,
    row,
    save_json,
    save_png,
)

from aether3d import Scene, __version__  # noqa: E402
from aether3d.cad.brep.properties import mass_properties  # noqa: E402

ITEMS = ("Ground floor", "First floor", "Ground slab", "First slab", "Roof", "Stair",
         "Pilotis")  # fmt: skip
STAIR = {"x": 8.5, "y0": 0.6, "width": 1.0, "tread": 0.28, "riser": 0.175}
OPENING = (1.5, 0.6 + 16 * 0.28)  # y from .. to: headroom start, the stair's arrival


def build(scene: Scene):
    lib = scene.materials
    lib.create_material("Plaster", base_color="#ecebe6", roughness=0.85)
    lib.create_material("Partition", base_color="#f3f1ec", roughness=0.9)
    lib.create_material("Concrete", base_color="#9d9a94", roughness=0.75)
    lib.create_material("Membrane", base_color="#3b3d40", roughness=0.6)
    lib.create_material("Oak", preset="wood")
    lib.create_material("Frame", base_color="#2c2e31", roughness=0.35, metallic=0.7)
    lib.create_material("Glass", base_color="#3e5666", roughness=0.03)
    doc = scene.cad
    doc.set_units(length="m")
    with doc.batch("House"):
        b = doc.building(
            "House",
            levels=[("Ground", 0, 3.0), ("First", 3.0, 3.0), ("Roof", 6.0, 1.0)],
            wall_types={
                "Ext": {"thickness": "300 mm", "function": "exterior", "material": "Plaster"},
                "Int": {"thickness": "120 mm", "material": "Partition"},
            },
            families={
                "D100": {"kind": "door", "width": "1000 mm", "height": "2200 mm"},
                "D80": {"kind": "door", "width": "800 mm", "height": "2100 mm"},
                "W300": {"kind": "window", "width": "3000 mm", "height": "2300 mm",
                         "sill": "200 mm", "mullions": 2},
                "W180": {"kind": "window", "width": "1800 mm", "height": "1500 mm",
                         "sill": "900 mm", "mullions": 1},
                "W60": {"kind": "window", "width": "600 mm", "height": "1500 mm",
                        "sill": "900 mm"},
            },  # fmt: skip
            criteria={"riser_min": "150 mm", "riser_max": "190 mm", "tread_min": "250 mm",
                      "blondel_min": "600 mm", "blondel_max": "650 mm"},  # fmt: skip
            materials={"slab": "Concrete", "roof": "Membrane", "stair": "Concrete",
                       "column": "Concrete", "leaf": "Oak", "frame": "Frame", "glass": "Glass"},
        )  # fmt: skip
        g = b.floor("Ground", name="Ground floor")
        e = g.walls([(0, 0), (10, 0), (10, 8), (0, 8)], type="Ext")
        g.wall((0, 5), (5, 5), type="Int", id="utility")
        g.door(e[1], "D100", position=5.5, id="entry")
        g.door("utility", "D80", position=2.0, id="d_util")
        g.window(e[0], "W300", position=2.5, id="g_s1")
        g.window(e[0], "W300", position=6.5, id="g_s2")
        g.window(e[3], "W180", position=4.0, id="g_w")
        g.window(e[2], "W180", position=3.0, id="g_n")
        g.room("Living / kitchen", (5, 2.5), number="G01")
        g.room("Utility", (2.5, 6.5), number="G02")
        f = b.floor("First", name="First floor")
        e2 = f.walls([(0, 0), (14, 0), (14, 8), (0, 8)], type="Ext")
        f.wall((6, 0), (6, 8), type="Int", id="f_mid")
        f.wall((10, 0), (10, 8), type="Int", id="f_east")
        f.door("f_mid", "D80", position=6.5, id="f_d1", flip=True)
        f.door("f_east", "D80", position=6.5, id="f_d2")
        f.window(e2[0], "W300", position=3.0, id="f_s1")
        f.window(e2[0], "W180", position=8.0, id="f_s2")
        f.window(e2[0], "W300", position=12.0, id="f_s3")
        f.window(e2[1], "W180", position=4.0, id="f_e")
        f.window(e2[2], "W60", position=11.0, id="f_n1")
        f.window(e2[2], "W180", position=3.0, id="f_n2")
        f.room("Bedroom", (3, 4), number="F01")
        f.room("Landing", (8, 1), number="F02")
        f.room("Studio", (12, 4), number="F03")
        x0, x1 = STAIR["x"] - 0.55, STAIR["x"] + 0.55
        b.slab("Ground", outline=g, thickness=0.25, name="Ground slab")
        b.slab("First", outline=f, thickness=0.25, name="First slab",
               holes=[[[x0, OPENING[0]], [x1, OPENING[0]], [x1, OPENING[1]],
                       [x0, OPENING[1]]]])  # fmt: skip
        b.roof("Roof", outline=f, type="flat", thickness=0.3, overhang=0.6, name="Roof")
        b.stair("Ground", "First", start=[STAIR["x"], STAIR["y0"]], direction=[0, 1],
                width=STAIR["width"], tread=STAIR["tread"], riser=STAIR["riser"],
                name="Stair")  # fmt: skip
        b.columns("Ground", height=2.75, points=[[13.6, 0.4], [13.6, 7.6]],
                  profile={"shape": "circle", "diameter": 0.3}, name="Pilotis")  # fmt: skip
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="smaller images, fewer samples")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    report: dict = {"aether3d": __version__}

    scene = Scene("Modern house")
    t0 = time.perf_counter()
    doc = build(scene)
    report["build_seconds"] = time.perf_counter() - t0

    stair = doc.info("Stair")
    report["stair"] = {"checks": stair["checks"], "quantities": stair.get("quantities")}
    if not stair["criteria_met"]:
        failures.append("the stair misses the stated criteria")
    q = doc.result("Stair").quantities
    r, t = q["RiserHeight"], q["TreadLength"]
    # The nosing line rises r per t from the first riser (r above y0); where the opening
    # begins, the slab's underside (2.75 m) must clear it by 2.0 m.
    nosing = r + (OPENING[0] - STAIR["y0"]) * r / t
    headroom = 2.75 - nosing
    arrival = STAIR["y0"] + (q["NumberOfRisers"] - 1) * t
    report["headroom_m"] = headroom
    report["arrival_y_m"] = arrival
    if headroom < 2.0:
        failures.append(f"headroom {headroom:.3f} m under the opening's edge")
    if abs(arrival - OPENING[1]) > 1e-12:
        failures.append(f"the stair arrives at y {arrival}, the opening ends at {OPENING[1]}")
    slab_bottom = min(b.transformed(m).bounds[0][2] for _, b, m in doc.result("First slab").bodies)
    tops = [b.transformed(m).bounds[1][2] for _, b, m in doc.result("Pilotis").bodies]
    report["pilotis_top_m"] = tops
    report["first_slab_underside_m"] = slab_bottom
    if any(abs(z - slab_bottom) > 1e-12 for z in tops):
        failures.append(f"pilotis tops {tops} vs slab underside {slab_bottom}")
    from aether3d.arch.api import FloorHandle

    areas = {}
    for name, expected in (("Ground floor", 10.3 * 8.3), ("First floor", 14.3 * 8.3)):
        a = FloorHandle(doc, doc.item(name)).quantities()["gross_floor_area"]
        areas[name] = a
        if abs(a - expected) > 1e-9:
            failures.append(f"{name}: gross area {a} m2, expected {expected}")
    report["gross_floor_areas_m2"] = areas

    ifc, stats = doc.export_ifc(name="Modern house")
    (out / "cad_modern_house.ifc").write_text(ifc)
    report["ifc_entities"] = stats
    # The B-reps' volumes by (storey, IFC name): walls are named by their id (ids repeat
    # on different floors), elements by their item (columns: item and member).
    own = {}
    for it in ITEMS:
        res = doc.result(it)
        if hasattr(res, "walls"):
            for w in res.walls.values():
                own[(res.level.name, w.id)] = mass_properties(w.body).volume
            continue
        d = doc.item(it).definition
        storey = d.get("level", d.get("from"))
        for name, body, _ in res.bodies:
            label = f"{it} {name}" if it == "Pilotis" else it
            own[(storey, label)] = mass_properties(body).volume
    worst, compared = 0.0, 0
    for e in doc.import_ifc(ifc, create_parts=False):
        key = (e.storey, e.name)
        if e.body is not None and key in own:
            worst = max(worst, abs(mass_properties(e.body).volume - own[key]))
            compared += 1
    report["ifc_products_compared"] = compared
    if compared != len(own):
        failures.append(f"compared {compared} IFC products with {len(own)} B-reps")
    report["ifc_volume_max_difference_m3"] = worst
    if worst > 1e-9:
        failures.append(f"an IFC product's volume differs by {worst} m3")

    doc.drawing("First floor plan", "plan", level="First", title="House: first floor plan")
    doc.drawing("Section B-B", "section", list(ITEMS), title="House: section B-B (stair)",
                plane={"origin": [STAIR["x"], 0, 0], "normal": [-1, 0, 0]})  # fmt: skip
    doc.drawing("South elevation", "front", list(ITEMS), title="House: south elevation")
    sheets = [doc.result(n) for n in ("First floor plan", "Section B-B", "South elevation")]
    for d in sheets:
        report.setdefault("drawings", {})[d.title] = d.summary()
        if d.notes:
            failures.append(f"{d.title}: {d.notes}")
    (out / "cad_modern_house_section.svg").write_text(sheets[1].to_svg())
    (out / "cad_modern_house_plan.svg").write_text(sheets[0].to_svg())

    shown = [doc.show(i) for i in ITEMS]
    ground(scene, size=90.0, z=-0.25, color="#8d9a78")
    daylight(scene, sun=(0.5, 0.75, -0.55))
    cam = scene.add_camera("Camera", fov=30)
    w, h = (720, 405) if args.quick else (1920, 1080)
    cam.frame(targets=meshes_of(shown), direction=(-0.62, 0.9, -0.3), margin=0.06,
              aspect=w / h)  # fmt: skip
    image, backend, seconds = render(scene, cam, w, h, args.backend, 16 if args.quick else 384)
    report["render"] = {"backend": backend, "seconds": seconds, "size": [w, h]}
    third = (w - 16) // 3
    bottom = row([caption(drawing_image(d, third), d.title) for d in sheets])
    picture = column([caption(image, f"Render ({backend})"), fit_width(bottom, w)])
    save_png(out / "cad_modern_house.png", picture)
    report["failures"] = failures
    save_json(out / "cad_modern_house.json", report)
    print(
        f"house: {sum(stats.values())} IFC products, stair {q['NumberOfRisers']} risers "
        f"of {r * 1000:.1f} mm, headroom {headroom:.2f} m; render {backend} {seconds:.1f} s"
    )
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
