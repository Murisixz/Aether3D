"""A parametric L-bracket: exact features, drawings, STEP, an edit, by Aether3D.

Run from the repository root:

    python examples/cad_bracket.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

1. **Sketch.** The bracket's section, fully constrained: horizontal and vertical lines,
   a fixed corner and four driving dimensions from parameters (length, height,
   thickness).
2. **Features.** Extruded symmetrically by ``width``; a concave fillet at the root, two
   counterbored bolt holes in the base, two holes in the upright, a chamfer, a rib.
   Every face has a persistent name (``base_holes.1:rev/cbore.0``).
3. **Checks.** The solid is valid; its volume (boundary quadrature on the exact surfaces)
   agrees with the volume of a fine tessellation within the tessellation's chord, and with
   its re-imported STEP export exactly; the bolt holes are ``2 x 15`` mm apart, measured
   between their exact axes.
4. **Edit.** The thickness and width parameters change: the sketch, the features and the
   drawings regenerate (the regeneration statistics are recorded), the holes keep their
   names.
5. **Drawings and render.** Front (hidden lines), top, a section through the bolt holes
   and an isometric view; the part rendered by Aether3D's path tracer (the GPU one when
   available).

Writes ``cad_bracket.png`` (the render beside the drawing sheet), ``cad_bracket.json``,
``cad_bracket.step``, ``cad_bracket_front.svg`` and ``cad_bracket_section.dxf``.
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cad_common import (  # noqa: E402
    caption,
    column,
    daylight,
    drawing_image,
    fit_width,
    ground,
    render,
    row,
    save_json,
    save_png,
)

from aether3d import Scene, __version__  # noqa: E402
from aether3d.cad.brep.tessellate import tessellate  # noqa: E402
from aether3d.cad.interop.step import read_step  # noqa: E402
from aether3d.cad.tolerance import TessellationTolerance  # noqa: E402


def build(scene: Scene):
    doc = scene.cad
    doc.set_units(length="mm")
    with doc.batch("Bracket"):
        for name, value in (("length", 80), ("height", 60), ("t", 8), ("width", 50),
                            ("bolt", 8)):  # fmt: skip
            doc.parameter(name, f"{value} mm")
        sk = doc.sketch("Section", plane="XZ")
        lines = sk.polyline([(0, 0), (80, 0), (80, 8), (8, 8), (8, 60), (0, 60)])
        sk.constrain("fix", "p1")
        for i, lid in enumerate(lines):
            sk.constrain("horizontal" if i % 2 == 0 else "vertical", lid)
        sk.dimension("length", lines[0], value="=length", id="d_length")
        sk.dimension("length", lines[1], value="=t", id="d_base")
        sk.dimension("length", lines[4], value="=t", id="d_wall")
        sk.dimension("length", lines[5], value="=height", id="d_height")
        part = doc.part("Bracket", materials={"all": "Steel"})
        part.add_feature({"type": "extrude", "id": "body", "sketch": "Section",
                          "extent": {"type": "symmetric", "distance": "=width"}})  # fmt: skip
        part.fillet([{"edge": ["body:side/l3", "body:side/l4"]}], 6, id="root")
        part.hole("body:side/l3", [[55, -15, 8], [55, 15, 8]], "=bolt", kind="counterbore",
                  cbore_diameter=14, cbore_depth=3, id="base_holes")  # fmt: skip
        part.hole("body:side/l4", [[8, -12, 42], [8, 12, 42]], 6.6, id="wall_holes")
        part.chamfer([{"edge": ["body:side/l2", "body:side/l3"]}], 2, id="edge")
        rib = doc.sketch("Rib", plane="XZ")
        rib.polyline([(8, 8), (40, 8), (8, 40)])
        part.add_feature({"type": "extrude", "id": "rib", "sketch": "Rib", "operation": "add",
                          "extent": {"type": "symmetric", "distance": 6}})  # fmt: skip
    return doc, part


def drawings(doc):
    """Four views on A4 sheets at the largest standard scale that fits (2:1 here)."""
    doc.drawing("Front", "front", ["Bracket"], title="L-bracket: front", sheet="A4")
    doc.drawing("Top", "top", ["Bracket"], title="L-bracket: top", sheet="A4")
    doc.drawing("Section A-A", "section", ["Bracket"], title="L-bracket: section A-A",
                plane={"origin": [0, 15, 0], "normal": [0, -1, 0]}, sheet="A4")  # fmt: skip
    doc.drawing("Iso", "iso", ["Bracket"], title="L-bracket: isometric", sheet="A4")


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

    scene = Scene("L-bracket")
    scene.materials.create_material("Steel", preset="metal")
    t0 = time.perf_counter()
    doc, part = build(scene)
    report["build_seconds"] = time.perf_counter() - t0
    sketch = doc.info("Section")
    report["sketch"] = {k: sketch[k] for k in ("status", "dof", "max_residual", "iterations")}
    if sketch["status"] != "fully_constrained":
        failures.append(f"the section is {sketch['status']}")

    def check_part(tag: str) -> dict:
        info = part.measure("main")
        body = part.body()
        fine = tessellate(body, TessellationTolerance(chord=2e-6, angle=math.radians(5),
                                                       relative=None))  # fmt: skip
        p = fine.positions[fine.triangles]
        mesh_volume = float(np.sum(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])))
                            / 6) * 1e9  # fmt: skip
        step = doc.export_step(["Bracket"], units="mm")
        (out / "cad_bracket.step").write_text(step)
        [(_, back)] = read_step(step)
        from aether3d.cad.brep.properties import mass_properties

        step_volume = mass_properties(back).volume * 1e9
        axes = doc.measure({"part": "Bracket", "face": "base_holes.0:rev/wall.0"},
                           {"part": "Bracket", "face": "base_holes.1:rev/wall.0"})  # fmt: skip
        entry = {
            "valid": info["valid"],
            "faces": info["faces"],
            "volume_mm3": info["volume"],
            "volume_error_estimate_mm3": info["volume_estimated_error"],
            "tessellation_volume_mm3": mesh_volume,
            "tessellation_triangles": len(fine.triangles),
            "step_bytes": len(step),
            "step_reimported_volume_mm3": step_volume,
            "bolt_axis_distance_mm": axes["axis_distance"],
            "surfaces": part.body().summary()["surfaces"],
        }
        if not info["valid"]:
            failures.append(f"{tag}: the solid is not valid")
        if abs(mesh_volume - info["volume"]) > 1e-3 * info["volume"]:
            failures.append(f"{tag}: tessellation volume {mesh_volume} vs {info['volume']}")
        if abs(step_volume - info["volume"]) > 1e-9 * info["volume"]:
            failures.append(f"{tag}: STEP round trip volume {step_volume} vs {info['volume']}")
        if abs(axes["axis_distance"] - 30.0) > 1e-9:
            failures.append(f"{tag}: bolt holes {axes['axis_distance']} mm apart")
        return entry

    report["first"] = check_part("first")
    drawings(doc)
    # The edit: thicker and wider. Everything downstream regenerates; nothing else.
    before = dict(doc.stats)
    t0 = time.perf_counter()
    with doc.batch("Thicker, wider"):
        doc.parameter("t", "10 mm")
        doc.parameter("width", "60 mm")
    report["edit"] = {
        "seconds": time.perf_counter() - t0,
        "evaluations": doc.stats["evaluations"] - before["evaluations"],
        "items": len(doc.items()),
    }
    report["edited"] = check_part("edited")
    if report["edited"]["volume_mm3"] <= report["first"]["volume_mm3"]:
        failures.append("the edit did not add material")
    names = [f["name"] for f in part.faces()]
    if "base_holes.1:rev/cbore.0" not in names:
        failures.append("a hole lost its name in the edit")

    # Drawings.
    sheets = {}
    for name in ("Front", "Top", "Section A-A", "Iso"):
        d = doc.result(name)
        sheets[name] = d
        report.setdefault("drawings", {})[name] = d.summary()
        if d.notes:
            failures.append(f"drawing {name}: {d.notes}")
    (out / "cad_bracket_front.svg").write_text(sheets["Front"].to_svg())
    (out / "cad_bracket_section.dxf").write_text(sheets["Section A-A"].to_dxf("mm"))

    # Render: the part on a ground plane, under a sun and a sky.
    obj = doc.show("Bracket")
    ground(scene, size=0.6)
    daylight(scene)
    cam = scene.add_camera("Camera", fov=28)
    w, h = (480, 360) if args.quick else (1200, 900)
    cam.frame(targets=[obj], direction=(-0.75, 0.95, -0.55), margin=0.12, aspect=w / h)
    image, backend, seconds = render(scene, cam, w, h, args.backend, 16 if args.quick else 256)
    report["render"] = {"backend": backend, "seconds": seconds, "size": [w, h]}

    sheet_w = w // 2
    tiles = {k: drawing_image(d, sheet_w) for k, d in sheets.items()}
    grid = column([row([tiles["Front"], tiles["Top"]], 0),
                   row([tiles["Section A-A"], tiles["Iso"]], 0)], 0)  # fmt: skip
    grid = fit_width(grid, w)
    picture = row([caption(image, f"Render ({backend})"), caption(grid, "Drawings")])
    save_png(out / "cad_bracket.png", picture)
    report["failures"] = failures
    save_json(out / "cad_bracket.json", report)
    e = report["edited"]
    print(
        f"bracket: {e['faces']} faces, {e['volume_mm3']:.3f} mm3 (tessellation "
        f"{e['tessellation_volume_mm3']:.3f}, STEP {e['step_reimported_volume_mm3']:.6f})"
    )
    print(f"edit: {report['edit']['evaluations']} evaluations in {report['edit']['seconds']:.3f} s;"
          f" render {backend} {seconds:.1f} s")  # fmt: skip
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
