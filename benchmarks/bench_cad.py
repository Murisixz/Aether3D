"""Precision modeling benchmarks (Stage 15): sketches, features, booleans, tessellation,
assemblies, buildings, drawings, exchange and the triangulation kernel.

Run from the repository root:

    python benchmarks/bench_cad.py [--json OUT.json] [--label NAME] [--quick]
        [--only sketch,...] [--compare SAVED.json]

Every group (and every size of a sized group) runs in its own Python process, so memory
peaks and caches of one never influence another; each reports its peak RSS (``VmHWM`` on
Linux). Timings are medians of repeated runs after a warm-up where a run is short, single
runs where it is long (``bench_engine.median_time``). Inputs are generated, so every run
measures the same models. Results depend on the machine: compare runs on one machine only.

Groups:

* ``sketch:N`` -- a row of N rectangles, each with 4 horizontal/vertical constraints, 2
  dimensions and 2 placement dimensions (8 N constraints, fully constrained): the first
  solve (structure analysis included) and an incremental solve after one dimension edit.
* ``features`` -- the L-bracket of ``examples/cad_bracket.py`` built, a parameter edit
  (everything downstream re-runs), and a feature edit near the end (only that feature).
* ``booleans:N`` -- a plate with N holes (one feature of N points): build time, faces.
* ``tessellate`` -- a sphere, a torus and the bracket at chord tolerances 1e-3, 1e-4 and
  1e-5 of their size: triangles and seconds; the bracket's mass properties.
* ``assembly:N`` -- a plate with N holes and N pins, each mated (concentric + coincident):
  solve time, degrees of freedom; exact clash detection (N = 4 only: each pin touches its
  hole, a B-rep intersection of coincident cylinders per pair).
* ``building:NxL`` -- L levels of an N x N grid of 5 m rooms (a wall per bay side, meeting
  at the grid's nodes; a door per room): build time, element counts; one door moved on
  one level (incremental) vs the whole building regenerated from its definitions (cold).
* ``drawing`` -- a plan, a section and an isometric view of ``building:6`` (2 levels).
* ``exchange`` -- STEP export and import of the bracket, IFC4 export and import of
  ``building:6``.
* ``triangulate`` -- 600 random polygons with holes (66k points): the native kernel and its
  NumPy twin (bitwise-equal output).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "tests"))

from bench_engine import compare, machine, median_time, ms, peak_rss_mib  # noqa: E402

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def rectangles(n: int) -> dict:
    """A sketch definition: n rectangles in a row, fully constrained (metres)."""
    ents, cons = [], []
    for i in range(n):
        x0 = i * 0.03
        pts = [(x0, 0.0), (x0 + 0.02, 0.0), (x0 + 0.02, 0.01), (x0, 0.01)]
        for k, p in enumerate(pts):
            ents.append({"id": f"p{i}_{k}", "type": "point", "at": list(p)})
        for k in range(4):
            ents.append({"id": f"l{i}_{k}", "type": "line", "start": f"p{i}_{k}",
                         "end": f"p{i}_{(k + 1) % 4}"})  # fmt: skip
            cons.append({"id": f"hv{i}_{k}", "type": "horizontal" if k % 2 == 0 else "vertical",
                         "refs": [f"l{i}_{k}"]})  # fmt: skip
        cons.append({"id": f"w{i}", "type": "length", "refs": [f"l{i}_0"], "value": 20})
        cons.append({"id": f"h{i}", "type": "length", "refs": [f"l{i}_1"], "value": 10})
        if i == 0:
            cons.append({"id": "fix", "type": "fix", "refs": ["p0_0"]})
        else:
            for axis in ("x", "y"):
                cons.append({"id": f"d{i}{axis}", "type": "distance", "along": axis,
                             "refs": [f"p{i - 1}_0", f"p{i}_0"],
                             "value": 30 if axis == "x" else 0})  # fmt: skip
    return {"plane": "XY", "entities": ents, "constraints": cons}


def building(scene, n: int, levels: int):
    doc = scene.cad
    doc.set_units(length="m")
    with doc.batch("Building"):
        b = doc.building("B", levels=[(f"L{k}", 3.0 * k, 3.0) for k in range(levels)],
                         wall_types={"W": {"thickness": "200 mm"}},
                         families={"D": {"kind": "door", "width": "900 mm",
                                         "height": "2100 mm"}})  # fmt: skip
        for k in range(levels):
            f = b.floor(f"L{k}", name=f"Floor {k}")
            # Walls meet at the grid's nodes (one wall per bay side: walls may not cross).
            for i in range(n + 1):
                for j in range(n):
                    f.wall((5.0 * j, 5.0 * i), (5.0 * j + 5, 5.0 * i), type="W", id=f"h{i}_{j}")
                    f.wall((5.0 * i, 5.0 * j), (5.0 * i, 5.0 * j + 5), type="W", id=f"v{i}_{j}")
            for i in range(n):
                for j in range(n):
                    f.door(f"h{j}_{i}", "D", position=2.5, id=f"d{i}_{j}")
                    f.room(f"R{i}.{j}", (5.0 * i + 2.5, 5.0 * j + 2.5))
    return doc


def bracket():
    from cad_bracket import build

    from aether3d import Scene

    scene = Scene("Bracket")
    scene.materials.create_material("Steel", base_color="#888888")
    doc, part = build(scene)
    return scene, doc, part


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #


def group_sketch(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    n = int(size)
    defn = rectangles(n)
    scene = Scene("Sketch")
    doc = scene.cad
    t0 = time.perf_counter()
    doc.add("sketch", "S", defn)
    first = time.perf_counter() - t0
    info = doc.info("S")
    values = iter(range(21, 10_000))

    def edit():
        item = doc.item("S")
        d = json.loads(json.dumps(item.definition))
        for c in d["constraints"]:
            if c["id"] == f"w{n // 2}":
                c["value"] = next(values)
        doc.set_definition(item, d)

    edit_s = median_time(edit, 3 if n >= 500 else 7)
    return {
        "constraints": len(defn["constraints"]),
        "equations_dof": info["dof"],
        "status": info["status"],
        "first_solve_ms": ms(first),
        "dimension_edit_ms": ms(edit_s),
        "blocks": info["structure"]["blocks"],
    }


def group_features(quick: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    scene, doc, part = bracket()
    build_s = time.perf_counter() - t0
    widths = iter((52, 54, 56, 58, 60, 62, 64))
    param = median_time(lambda: doc.parameter("width", f"{next(widths)} mm"), 3, warmup=0)
    radii = iter((2.0, 2.5, 3.0, 3.5, 4.0))
    last = median_time(lambda: part.set("edge", distance=next(radii)), 3, warmup=0)
    return {
        "faces": part.measure("main")["faces"],
        "build_ms": ms(build_s),
        "parameter_edit_ms": ms(param),
        "last_feature_edit_ms": ms(last),
    }


def group_booleans(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    n = int(size)
    side = math.ceil(math.sqrt(n))
    scene = Scene("Holes")
    doc = scene.cad
    doc.set_units(length="mm")
    sk = doc.sketch("P")
    sk.rectangle((0, 0), (20 * side, 20 * side))
    p = doc.part("Plate")
    p.extrude(sk, 5, id="base")
    pts = [[10 + 20 * (k % side), 10 + 20 * (k // side), 5] for k in range(n)]
    t0 = time.perf_counter()
    p.hole("base:end", pts, 6, id="h")
    s = time.perf_counter() - t0
    m = p.measure("main")
    return {"holes": n, "seconds": round(s, 4), "faces": m["faces"], "valid": m["valid"]}


def group_tessellate(quick: bool) -> dict[str, Any]:
    from aether3d.cad.brep import build
    from aether3d.cad.brep.properties import mass_properties
    from aether3d.cad.brep.tessellate import tessellate
    from aether3d.cad.tolerance import TessellationTolerance

    _, _, part = bracket()
    bodies = {"sphere": build.sphere(0.05), "torus": build.torus(0.05, 0.015),
              "bracket": part.body()}  # fmt: skip
    out: dict[str, Any] = {}
    for name, body in bodies.items():
        for rel in (1e-3, 1e-4, 1e-5) if not quick else (1e-3, 1e-4):
            tol = TessellationTolerance(chord=1.0, angle=math.radians(30), relative=rel)
            t = tessellate(body, tol)
            s = median_time(lambda b=body, tol=tol: tessellate(b, tol), 3)
            out[f"{name}@{rel:g}"] = {"triangles": len(t.triangles), "ms": ms(s)}
    out["bracket_mass_properties_ms"] = ms(median_time(lambda: mass_properties(part.body()), 5))
    return out


def group_assembly(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene
    from aether3d.cad.assembly import clashes

    n = int(size)
    scene = Scene("Assembly")
    doc = scene.cad
    doc.set_units(length="mm")
    with doc.batch("Parts"):
        sk = doc.sketch("P")
        sk.rectangle((0, 0), (20 * n, 20))
        plate = doc.part("Plate")
        plate.extrude(sk, 10, id="base")
        plate.hole("base:end", [[10 + 20 * k, 10, 10] for k in range(n)], 8, id="h")
        pin = doc.part("Pin")
        pin.add_feature({"type": "cylinder", "id": "shaft", "radius": 4, "height": 30,
                         "frame": {"origin": [0, 0, 15]}})  # fmt: skip
        pin.add_feature({"type": "cylinder", "id": "head", "radius": 7, "height": 4,
                         "frame": {"origin": [0, 0, 32]}, "operation": "add"})  # fmt: skip
    comps = [{"id": "plate", "source": "Plate", "fixed": True}]
    mates = []
    for k in range(n):
        comps.append({"id": f"pin{k}", "source": "Pin",
                      "placement": {"origin": [20 * k + 3, 7, 40]}})  # fmt: skip
        mates.append({"id": f"c{k}", "type": "concentric",
                      "a": {"component": "plate", "face": f"h.{k}:rev/wall.0"},
                      "b": {"component": f"pin{k}", "face": "shaft:side/wall.0"}})  # fmt: skip
        mates.append({"id": f"f{k}", "type": "coincident",
                      "a": {"component": "plate", "face": "base:end"},
                      "b": {"component": f"pin{k}", "face": "head:start"}})  # fmt: skip
    t0 = time.perf_counter()
    doc.add("assembly", "A", {"components": comps, "mates": mates})
    solve = time.perf_counter() - t0
    info = doc.info("A")
    out = {"components": n + 1, "mates": 2 * n, "solve_ms": ms(solve),
           "status": info.get("status"), "dof": info.get("dof")}  # fmt: skip
    if n <= 4:
        # Exact clash detection: one B-rep intersection per pair of overlapping boxes
        # (each pin touches its hole: coincident cylinders, classified as contact).
        t0 = time.perf_counter()
        found = clashes(doc, "A")
        out["clash_ms"] = ms(time.perf_counter() - t0)
        out["contacts"] = sum(1 for c in found if c["kind"] != "interference")
        out["interferences"] = sum(1 for c in found if c["kind"] == "interference")
    return out


def group_building(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene
    from aether3d.arch.api import FloorHandle

    n, levels = (int(x) for x in size.split("x"))
    scene = Scene("Building")
    t0 = time.perf_counter()
    doc = building(scene, n, levels)
    build_s = time.perf_counter() - t0
    q = FloorHandle(doc, doc.item("Floor 0")).quantities()["totals"]
    positions = iter((1.0, 1.5, 2.0, 2.5, 3.0))

    def move_door():
        item = doc.item(f"Floor {levels // 2}")
        d = json.loads(json.dumps(item.definition))
        d["openings"][0]["position"] = next(positions)
        doc.set_definition(item, d)

    s0 = dict(doc.stats)
    edit = median_time(move_door, 3, warmup=0)
    evaluations = (doc.stats["evaluations"] - s0["evaluations"]) / 3
    definitions = doc.to_json()
    t0 = time.perf_counter()
    cold = Scene("Cold")
    cold.cad._load(definitions)
    cold.cad.regenerate_all()
    cold_s = time.perf_counter() - t0
    return {
        "levels": levels,
        "walls": q["walls"] * levels,
        "doors": q["doors"] * levels,
        "rooms": n * n * levels,
        "build_s": round(build_s, 3),
        "door_edit_ms": ms(edit),
        "door_edit_evaluations": evaluations,
        "cold_regeneration_s": round(cold_s, 3),
    }


def group_drawing(quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    scene = Scene("Drawings")
    doc = building(scene, 6, 2)
    out = {}
    for name, view, extra in (
        ("plan", "plan", {"level": "L0"}),
        ("section", "section", {"plane": {"origin": [15, 12.5, 0], "normal": [0, -1, 0]},
                                "items": ["Floor 0", "Floor 1"]}),
        ("iso", "iso", {"items": ["Floor 0", "Floor 1"]}),
    ):  # fmt: skip
        t0 = time.perf_counter()
        doc.drawing(name, view, **extra)
        d = doc.result(name)
        s = time.perf_counter() - t0
        t0 = time.perf_counter()
        png = d.to_png(100)
        out[name] = {"ms": ms(s), "polylines": len(d.polylines), "hatches": len(d.hatches),
                     "rays": d.info.get("visibility_rays"), "png_ms": ms(time.perf_counter() - t0),
                     "png_pixels": int(png.shape[0] * png.shape[1])}  # fmt: skip
    return out


def group_exchange(quick: bool) -> dict[str, Any]:
    from aether3d import Scene
    from aether3d.cad.interop.step import read_step

    _, doc, _ = bracket()
    t0 = time.perf_counter()
    step = doc.export_step(["Bracket"])
    export_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    read_step(step)
    import_s = time.perf_counter() - t0
    scene = Scene("IFC")
    bdoc = building(scene, 6, 2)
    t0 = time.perf_counter()
    ifc, stats = bdoc.export_ifc()
    ifc_export = time.perf_counter() - t0
    t0 = time.perf_counter()
    elements = bdoc.import_ifc(ifc, create_parts=False)
    ifc_import = time.perf_counter() - t0
    return {"step_bytes": len(step), "step_export_ms": ms(export_s), "step_import_ms": ms(import_s),
            "ifc_bytes": len(ifc), "ifc_products": sum(stats.values()) - stats.get(
                "IfcOpeningElement", 0), "ifc_export_s": round(ifc_export, 3),
            "ifc_import_s": round(ifc_import, 3), "ifc_elements_read": len(elements)}  # fmt: skip


def group_triangulate(quick: bool) -> dict[str, Any]:
    import numpy as np
    from test_native import _polygons

    from aether3d import _native
    from aether3d.cad.triangulate import triangulate, triangulate_numpy

    pts, rings, polys = _polygons(np.random.default_rng(0), 600)
    out = {"polygons": len(polys) - 1, "points": len(pts)}
    if _native.module() is not None:
        out["native_ms"] = ms(median_time(lambda: triangulate(pts, rings, polys), 5))
    out["numpy_ms"] = ms(median_time(lambda: triangulate_numpy(pts, rings, polys), 3))
    a = triangulate(pts, rings, polys)
    b = triangulate_numpy(pts, rings, polys)
    out["triangles"] = len(a[0])
    out["bitwise_equal"] = bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    return out


SIZED = {
    "sketch": (group_sketch, ("2", "12", "125", "625")),
    "booleans": (group_booleans, ("4", "25", "100")),
    "assembly": (group_assembly, ("4", "16", "64")),
    "building": (group_building, ("4x2", "10x4", "10x10")),
}
QUICK_SIZES = {"sketch": ("2", "125"), "booleans": ("4", "25"), "assembly": ("4",),
               "building": ("4x2",)}  # fmt: skip
GROUPS = {
    "features": group_features,
    "tessellate": group_tessellate,
    "drawing": group_drawing,
    "exchange": group_exchange,
    "triangulate": group_triangulate,
}


def run_group(name: str, quick: bool) -> dict[str, Any]:
    group, _, size = name.partition(":")
    result = SIZED[group][0](size, quick) if size else GROUPS[group](quick)
    result["peak_rss_mib"] = peak_rss_mib()
    return result


def jobs(only: list[str], quick: bool) -> list[str]:
    out = []
    for name in only:
        group, _, size = name.partition(":")
        if group in GROUPS and not size:
            out.append(group)
        elif group in SIZED:
            sizes = (size,) if size else (QUICK_SIZES[group] if quick else SIZED[group][1])
            out += [f"{group}:{s}" for s in sizes]
        else:
            raise SystemExit(f"unknown group {name!r}")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join([*SIZED, *GROUPS]))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print("BENCH-RESULT " + json.dumps(run_group(args.worker, args.quick)))
        return 0
    groups = jobs([g for g in args.only.split(",") if g], args.quick)
    data: dict[str, Any] = {"label": args.label, "machine": machine(), "groups": {}}
    for group in groups:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", group]
        if args.quick:
            command.append("--quick")
        t0 = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BENCH-RESULT ")]
        if proc.returncode != 0 or not lines:
            data["groups"][group] = {"error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"{group}: FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        data["groups"][group] = json.loads(lines[-1][len("BENCH-RESULT ") :])
        print(f"{group} ({time.perf_counter() - t0:.0f} s): {json.dumps(data['groups'][group])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        compare(data, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
