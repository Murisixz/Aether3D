"""A flat's floor plan: walls, openings, rooms, quantities, IFC, by Aether3D.

Run from the repository root:

    python examples/cad_floor_plan.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

A 12 x 9 m flat on one level: exterior walls along the lines of a dimensioned footprint
sketch (its length and depth are parameters), interior partitions of another type,
doors and windows from families, five rooms found from the walls around their points.

Checks (exact, from the parameters and the B-reps):

* the rooms' net areas and the walls' footprints tile the gross floor area (the outside
  of the exterior walls) exactly;
* every wall's net volume (the IFC base quantity, from the parameters) equals the
  measured volume of its B-rep with the openings cut out;
* the IFC4 export, read back, gives every wall the same net volume.

Writes ``cad_floor_plan.png`` (the plan at 1:50 beside a cut-away render),
``cad_floor_plan.svg`` and ``.dxf`` (the plan), ``cad_floor_plan.ifc``,
``cad_floor_plan_schedule.csv`` (walls, doors, windows, rooms) and
``cad_floor_plan.json``. Exits with status 1 if a check fails.
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
from aether3d.arch.quantities import schedule, to_csv  # noqa: E402
from aether3d.cad.brep.properties import mass_properties  # noqa: E402

FAMILIES = {
    "D80": {"kind": "door", "width": "800 mm", "height": "2100 mm"},
    "D90": {"kind": "door", "width": "900 mm", "height": "2100 mm"},
    "W150": {"kind": "window", "width": "1500 mm", "height": "1400 mm", "sill": "900 mm",
             "mullions": 1},
    "W240": {"kind": "window", "width": "2400 mm", "height": "1600 mm", "sill": "700 mm",
             "mullions": 2},
    "W60": {"kind": "window", "width": "600 mm", "height": "600 mm", "sill": "1500 mm"},
}  # fmt: skip


def build(scene: Scene):
    """The flat, as one undo step. Returns the document, the building and the floor."""
    lib = scene.materials
    lib.create_material("Plaster", base_color="#e9e4da", roughness=0.85)
    lib.create_material("Partition", base_color="#f3f1ec", roughness=0.9)
    lib.create_material("Oak", preset="wood")
    lib.create_material("Frame", base_color="#3a3c40", roughness=0.4, metallic=0.6)
    lib.create_material("Glass", base_color="#4f6878", roughness=0.04)  # opaque, reflective
    lib.create_material("Screed", base_color="#9c968c", roughness=0.8)
    doc = scene.cad
    doc.set_units(length="m")
    with doc.batch("Flat"):
        b = doc.building(
            "Flat",
            levels=[("Level 1", 0, 2.8)],
            wall_types={
                "Ext": {"thickness": "250 mm", "function": "exterior", "material": "Plaster"},
                "Int": {"thickness": "100 mm", "material": "Partition"},
            },
            families=FAMILIES,
            materials={"leaf": "Oak", "frame": "Frame", "glass": "Glass", "slab": "Screed"},
        )
        # The exterior walls follow a dimensioned sketch of the footprint (its dimensions
        # are parameters); the partitions are placed by coordinates.
        doc.parameter("length", "12 m")
        doc.parameter("depth", "9 m")
        footprint = doc.sketch("Footprint")
        sides = footprint.rectangle((0, 0), (12, 9))
        footprint.constrain("fix", "p1")
        footprint.dimension("length", sides[0], value="=length")
        footprint.dimension("length", sides[1], value="=depth")
        g = b.floor("Level 1", name="Level 1")
        ext = g.walls_from_sketch(footprint, type="Ext")
        g.wall((7, 0), (7, 9), type="Int", id="spine")
        g.wall((0, 5), (7, 5), type="Int", id="hall")
        g.wall((7, 4.5), (12, 4.5), type="Int", id="beds")
        g.wall((3, 5), (3, 9), type="Int", id="bath")
        g.door(ext[0], "D90", position=2.0, id="entry")
        g.door("spine", "D80", position=2.2, id="d_bed1", flip=True)
        g.door("spine", "D80", position=6.8, id="d_bed2", flip=True)
        g.door("hall", "D80", position=1.5, id="d_bath")
        g.door("hall", "D90", position=5.0, id="d_kitchen")
        g.window(ext[0], "W240", position=4.6, id="w_living")
        g.window(ext[0], "W150", position=9.5, id="w_bed1")
        g.window(ext[2], "W150", position=2.5, id="w_bed2")
        g.window(ext[2], "W150", position=7.0, id="w_kitchen")
        g.window(ext[2], "W60", position=10.5, id="w_bath")
        g.window(ext[3], "W150", position=6.5, id="w_living2")
        for name, point, number in (("Living", (3.5, 2.5), "01"), ("Kitchen", (5, 7), "02"),
                                    ("Bath", (1.5, 7), "03"), ("Bedroom 1", (9.5, 2.2), "04"),
                                    ("Bedroom 2", (9.5, 6.7), "05")):  # fmt: skip
            g.room(name, point, number=number)
        b.slab("Level 1", outline=g, thickness=0.2, name="Slab")
    return doc, b, g


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

    scene = Scene("Flat")
    t0 = time.perf_counter()
    doc, building, floor = build(scene)
    report["build_seconds"] = time.perf_counter() - t0
    sketch = doc.info("Footprint")
    report["footprint_sketch"] = {k: sketch[k] for k in ("status", "dof", "reference")}
    if sketch["status"] != "fully_constrained":
        failures.append(f"the footprint sketch is {sketch['status']}")
    q = floor.quantities()
    report["totals"] = q["totals"]
    report["gross_floor_area_m2"] = q["gross_floor_area"]
    rooms = sum(r["quantities"]["NetFloorArea"] for r in q["rooms"])
    walls = sum(w["quantities"]["GrossFootprintArea"] for w in q["walls"])
    report["tiling"] = {"rooms_m2": rooms, "wall_footprints_m2": walls}
    if abs(rooms + walls - 12.25 * 9.25) > 1e-9:
        failures.append(f"rooms {rooms} + walls {walls} do not tile 12.25 x 9.25")
    res = doc.result("Level 1")
    worst = 0.0
    for w in q["walls"]:
        measured = mass_properties(res.walls[w["id"]].body).volume
        worst = max(worst, abs(measured - w["quantities"]["NetVolume"]))
    report["wall_volume_max_difference_m3"] = worst
    if worst > 1e-9:
        failures.append(f"a wall's B-rep volume differs from its quantity by {worst}")

    rows = schedule(doc)
    (out / "cad_floor_plan_schedule.csv").write_text(to_csv(rows))
    report["schedule_rows"] = len(rows)
    ifc, stats = doc.export_ifc(name="Flat")
    (out / "cad_floor_plan.ifc").write_text(ifc)
    report["ifc_entities"] = stats
    back = {e.name: e for e in doc.import_ifc(ifc, create_parts=False)}
    for w in q["walls"]:
        e = back[w["id"]]
        if abs(mass_properties(e.body).volume - w["quantities"]["NetVolume"]) > 1e-9:
            failures.append(f"IFC wall {w['id']} volume differs")

    doc.drawing("Plan", "plan", level="Level 1", title="Flat: level 1 plan", scale="1:50")
    plan = doc.result("Plan")
    report["plan"] = plan.summary()
    if plan.notes:
        failures.append(f"plan notes: {plan.notes}")
    (out / "cad_floor_plan.svg").write_text(plan.to_svg())
    (out / "cad_floor_plan.dxf").write_text(plan.to_dxf("mm"))

    # The cut-away: walls, doors, windows and slab, no roof, from above.
    shown = [doc.show("Level 1"), doc.show("Slab")]
    ground(scene, size=60.0, z=-0.2, color="#8f9a7a")
    daylight(scene, sun=(0.35, 0.55, -0.76))
    cam = scene.add_camera("Camera", fov=30)
    w, h = (560, 420) if args.quick else (1400, 1050)
    cam.frame(targets=meshes_of(shown), direction=(0.45, 0.75, -0.95), margin=0.04,
              aspect=w / h)  # fmt: skip
    image, backend, seconds = render(scene, cam, w, h, args.backend, 16 if args.quick else 256)
    report["render"] = {"backend": backend, "seconds": seconds, "size": [w, h]}
    sheet = fit_width(drawing_image(plan, 2 * w), w * 4 // 3)
    picture = row([caption(sheet, "Plan 1:50 (Aether3D drawing)"),
                   caption(image, f"Cut-away render ({backend})")])  # fmt: skip
    save_png(out / "cad_floor_plan.png", picture)
    report["failures"] = failures
    save_json(out / "cad_floor_plan.json", report)
    print(f"flat: {q['totals']['walls']} walls, {q['totals']['doors']} doors, "
          f"{q['totals']['windows']} windows, {len(q['rooms'])} rooms ({rooms:.2f} m2 net of "
          f"{q['gross_floor_area']:.4f} m2 gross); IFC {sum(stats.values())} products")  # fmt: skip
    print(f"render {backend} {seconds:.1f} s")
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
