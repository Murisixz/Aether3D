"""A parametric grid building and incremental edits, by Aether3D.

Run from the repository root:

    python examples/cad_grid_building.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

A four-storey frame on a 4 x 3 bay grid: per level a slab, 20 columns at the grid's
intersections, 31 beams along its lines under the slab above, and a core of walls with a
door. Every grid coordinate is an expression over two parameters (``bay_x``, ``bay_y``),
so the whole frame follows them.

Edits, each one undoable step, timed and counted (``evaluations``: items re-evaluated):

1. a door moves in the level 2 core -- only that floor re-evaluates;
2. ``bay_x`` grows from 7.5 m to 9 m -- every grid-dependent item re-evaluates, nothing
   else (the parameters and other untouched items keep their results);
3. undo of both, then redo -- the history restores the definitions and the items they
   affect re-evaluate (results still in the document's bounded result cache are reused:
   ``cache_hits``); the counts are reported, not assumed.

Checks: beam and column counts, the beams' total length and the slabs' area against the
grid's closed forms, before and after; the level 2 plan after the edits; undo restores
the first state exactly.

Writes ``cad_grid_building.png`` (renders before and after the bay edit, the level 2
plan), ``cad_grid_building.json`` (timings, counts, quantities). Exits with status 1 if a
check fails.
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

LEVELS, HEIGHT, NX, NY = 4, 3.6, 4, 3
SLAB, BEAM_DEPTH = 0.25, 0.6


def build(scene: Scene):
    lib = scene.materials
    lib.create_material("Concrete", base_color="#a29e97", roughness=0.8)
    lib.create_material("Core", base_color="#c9c4ba", roughness=0.85)
    lib.create_material("Door", base_color="#8a5a3a", roughness=0.6)
    doc = scene.cad
    doc.set_units(length="m")
    with doc.batch("Grid building"):
        doc.parameter("bay_x", "7.5 m")
        doc.parameter("bay_y", "6 m")
        b = doc.building(
            "Office",
            levels=[(f"L{k}", k * HEIGHT, HEIGHT) for k in range(LEVELS)] + [("Roof", 14.4, 1)],
            wall_types={"Core": {"thickness": "250 mm", "material": "Core"}},
            families={"D": {"kind": "door", "width": "1000 mm", "height": "2100 mm"}},
            materials={"slab": "Concrete", "column": "Concrete", "beam": "Concrete",
                       "leaf": "Door", "frame": "Door"},
        )  # fmt: skip
        xs = [f"={i}*bay_x" for i in range(NX + 1)]
        ys = [f"={j}*bay_y" for j in range(NY + 1)]
        outline = [["=0", "=0"], [xs[-1], "=0"], [xs[-1], ys[-1]], ["=0", ys[-1]]]
        lines = [[[xs[i], y], [xs[i + 1], y]] for i in range(NX) for y in ys]
        lines += [[[x, ys[j]], [x, ys[j + 1]]] for j in range(NY) for x in xs]
        for k in range(LEVELS):
            level, above = f"L{k}", (f"L{k + 1}" if k + 1 < LEVELS else "Roof")
            core = b.floor(level, name=f"Core {level}")
            corners = [
                ["=1.5*bay_x", "=1.2*bay_y"],
                ["=2.5*bay_x", "=1.2*bay_y"],
                ["=2.5*bay_x", "=1.8*bay_y"],
                ["=1.5*bay_x", "=1.8*bay_y"],
            ]
            # Between the grid lines (no column inside it), up to the slab above.
            walls = [core.wall(corners[i], corners[(i + 1) % 4], type="Core",
                               height=HEIGHT - SLAB, id=f"c{i + 1}") for i in range(4)]  # fmt: skip
            core.door(walls[0], "D", position=1.5, id="door")
            b.slab(level, boundary=outline, thickness=SLAB, name=f"Slab {level}")
            b.columns(level, height=HEIGHT - SLAB, grid={"x": xs, "y": ys},
                      profile={"shape": "rect", "width": 0.5, "depth": 0.5},
                      name=f"Columns {level}")  # fmt: skip
            b.beams(above, lines, profile={"shape": "rect", "width": 0.35, "depth": BEAM_DEPTH},
                    top_offset=-SLAB, name=f"Beams {level}")  # fmt: skip
        b.slab("Roof", boundary=outline, thickness=SLAB, name="Roof slab")
    return doc


def quantities(doc) -> dict:
    beams = [doc.result(f"Beams L{k}").quantities for k in range(LEVELS)]
    cols = [doc.result(f"Columns L{k}").quantities for k in range(LEVELS)]
    slabs = [doc.result(f"Slab L{k}").quantities for k in range(LEVELS)]
    return {
        "beam_count": sum(q["Count"] for q in beams),
        "beam_length_m": sum(q["Length"] for q in beams),
        "column_count": sum(q["Count"] for q in cols),
        "slab_area_m2": sum(q["GrossArea"] for q in slabs),
    }


def expected(bx: float, by: float) -> dict:
    return {
        "beam_count": LEVELS * (NX * (NY + 1) + NY * (NX + 1)),
        "beam_length_m": LEVELS * (NX * bx * (NY + 1) + NY * by * (NX + 1)),
        "column_count": LEVELS * (NX + 1) * (NY + 1),
        "slab_area_m2": LEVELS * NX * bx * NY * by,
    }


def compare(tag, got, want, failures):
    for k, v in want.items():
        if abs(got[k] - v) > 1e-9 * max(1.0, abs(v)):
            failures.append(f"{tag}: {k} is {got[k]}, expected {v}")


def timed(doc, fn):
    s0, t0 = dict(doc.stats), time.perf_counter()
    fn()
    return {"seconds": time.perf_counter() - t0,
            "evaluations": doc.stats["evaluations"] - s0["evaluations"],
            "cache_hits": doc.stats["cache_hits"] - s0["cache_hits"]}  # fmt: skip


def snapshot(scene, doc, args, label):
    """A render of the frame from the fixed camera (shown objects are removed after it)."""
    shown = [doc.show(i) for i in [i.name for i in doc.items()
                                   if i.kind in ("floor", "slab", "columns", "beams")]]  # fmt: skip
    w, h = (480, 300) if args.quick else (1200, 750)
    image, backend, seconds = render(scene, scene.get_by_name("Camera"), w, h, args.backend,
                                     16 if args.quick else 256)  # fmt: skip
    for obj in shown:
        scene.remove(obj)
    return caption(image, f"{label} ({backend})"), {"backend": backend, "seconds": seconds}


def place_camera(scene, doc, args):
    """One camera for both renders, framed on the larger (edited) frame: the bay edit
    shows as a change of size."""
    w, h = (480, 300) if args.quick else (1200, 750)
    cam = scene.add_camera("Camera", fov=30)
    doc.parameter("bay_x", "9 m")  # one step
    with scene.history.transaction("Frame the edit"):  # another
        shown = [doc.show(i) for i in [i.name for i in doc.items()
                                       if i.kind in ("slab", "columns")]]  # fmt: skip
        cam.frame(targets=meshes_of(shown), direction=(-0.7, 0.85, -0.45), margin=0.04,
                  aspect=w / h)  # fmt: skip
    # Both steps are undone; the camera keeps its pose.
    pose = (cam.position, cam.rotation)
    scene.history.undo()
    scene.history.undo()
    cam.position, cam.rotation = pose
    return cam


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

    scene = Scene("Grid building")
    t0 = time.perf_counter()
    doc = build(scene)
    report["build"] = {"seconds": time.perf_counter() - t0, "items": len(doc.items()),
                       "evaluations": doc.stats["evaluations"]}  # fmt: skip
    first = quantities(doc)
    report["before"] = first
    compare("before", first, expected(7.5, 6.0), failures)
    ground(scene, size=160.0, z=-SLAB, color="#8e9a7c")
    daylight(scene, sun=(0.4, 0.7, -0.6))
    place_camera(scene, doc, args)
    before_img, report["render_before"] = snapshot(scene, doc, args, "Before: bay_x 7.5 m")

    def move_door():
        floor = doc.item("Core L2")
        d = dict(floor.definition)
        d["openings"] = [{**d["openings"][0], "position": 4.0}]
        doc.set_definition(floor, d)

    report["edit_door"] = timed(doc, move_door)
    report["edit_bay"] = timed(doc, lambda: doc.parameter("bay_x", "9 m"))
    after = quantities(doc)
    report["after"] = after
    compare("after", after, expected(9.0, 6.0), failures)
    if report["edit_door"]["evaluations"] != 1:
        failures.append(f"moving a door re-evaluated {report['edit_door']['evaluations']} items")
    # Undo and redo, each followed by reading every quantity (so any re-evaluation they
    # cause is counted).
    got = {}
    report["undo"] = timed(doc, lambda: (scene.history.undo(), scene.history.undo(),
                                         got.update(undone=quantities(doc))))  # fmt: skip
    if got["undone"] != first:
        failures.append("undo did not restore the first state")
    report["redo"] = timed(doc, lambda: (scene.history.redo(), scene.history.redo(),
                                         got.update(redone=quantities(doc))))  # fmt: skip
    if got["redone"] != after:
        failures.append("redo did not restore the edited state")
    doc.drawing("Level 2 plan", "plan", level="L2", title="Office: level 2 plan (bay_x 9 m)",
                items=["Core L2", "Slab L2", "Columns L2", "Beams L1"], sheet="A2")  # fmt: skip
    plan = doc.result("Level 2 plan")
    report["plan"] = plan.summary()
    if plan.notes:
        failures.append(f"plan notes: {plan.notes}")
    after_img, report["render_after"] = snapshot(scene, doc, args, "After: bay_x 9 m")

    w = before_img.shape[1]
    renders = row([before_img, after_img])
    sheet = caption(fit_width(drawing_image(plan, 2 * w), renders.shape[1]), plan.title)
    save_png(out / "cad_grid_building.png", column([renders, sheet]))
    report["failures"] = failures
    save_json(out / "cad_grid_building.json", report)
    b, d = report["build"], report["edit_door"]
    print(f"build: {b['items']} items in {b['seconds']:.3f} s; door edit: {d['evaluations']} "
          f"evaluation, {d['seconds'] * 1000:.1f} ms; bay edit: "
          f"{report['edit_bay']['evaluations']} evaluations, "
          f"{report['edit_bay']['seconds'] * 1000:.1f} ms; undo: "
          f"{report['undo']['evaluations']} evaluations, "
          f"{report['undo']['cache_hits']} cache hits")  # fmt: skip
    print(f"beams {after['beam_count']}, {after['beam_length_m']:.3f} m; columns "
          f"{after['column_count']}; slabs {after['slab_area_m2']:.3f} m2")  # fmt: skip
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
