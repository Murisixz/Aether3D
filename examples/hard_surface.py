"""Hard-surface modeling with the Stage 13 operators: a sci-fi power cell, rendered by Aether3D.

Run from the repository root:

    python examples/hard_surface.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

Every step is an ``EditableMesh`` operation (each one also runs from a JSON edit script,
``aether3d edit``), applied in an edit session that commits one undo step per part:

1. Body: a box whose eight rim edges are rounded by a three-segment bevel (the corners
   where three rounded edges meet get a patch), then its top inset and pushed in.
2. A duct through the body: both end faces inset and opened, and ``bridge_edge_loops``
   joins the two holes with three rings of quads.
3. Panel lines: two ``knife_project`` cuts across the front (only faces facing the cut,
   so they wrap around the rounded corners and stop at the sides), and the band between
   them inset and pressed in.
4. A chamfer: ``bisect`` along a slanted plane, clearing the corner and filling the cut.
5. Shading: ``set_auto_smooth`` (creases sharp by angle) and ``set_weighted_normals``
   (large flat faces stay flat next to rounded edges).
6. A knob from ``add_lathe``; a bezel ring from ``spin`` of a square profile, turned
   outward by ``recalculate_normals``.
7. A cap: a cylinder primitive with a creased Catmull-Clark subdivision (crease 3 keeps
   its rims sharp while the rest rounds off).
8. Checks: every part reaches the ``solid`` validity tier (closed, manifold, outward, no
   self-intersection: the exact test), and ``cleanup`` finds nothing to repair.

Writes ``hard_surface.png`` (two views) and ``hard_surface.json`` (each operation with
what it created and its time, the validity of each part) to OUTPUT_DIR. Exits with
status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import Scene, __version__
from aether3d.rendering.png import encode_png
from aether3d.sculpt.validate import object_report

HERE = Path(__file__).resolve().parent


def render(scene, camera, width: int, height: int, backend: str):
    """A render with sun shadows where the backend draws them (the GPU rasterizer); the
    CPU rasterizer has none."""
    from aether3d import RenderError

    options = {"width": width, "height": height, "camera": camera, "backend": backend}
    if backend != "cpu":
        try:
            return scene.render(**options, shadows=True)
        except RenderError as exc:
            if exc.code != "BACKEND_UNSUPPORTED":
                raise
    return scene.render(**options)


class Steps:
    """Runs operations, recording what each created and how long it took."""

    def __init__(self) -> None:
        self.log: list[dict] = []

    def __call__(self, part: str, name: str, op, *args, **kwargs):
        t0 = time.perf_counter()
        result = op(*args, **kwargs)
        created = result.created if isinstance(result.created, dict) else {}
        self.log.append(
            {
                "part": part,
                "operation": name,
                "created": {k: len(v) for k, v in created.items()},
                "diagnostics": [d.code for d in result.diagnostics],
                "ms": round(1000 * (time.perf_counter() - t0), 2),
            }
        )
        return result


def on(axis: int, value: float, tol: float = 1e-6):
    return lambda _i, p: abs(p[axis] - value) < tol


def build_body(scene: Scene, step: Steps):
    body = scene.add_object("Power cell")
    with body.edit_mesh() as m:
        m.add_box((2.0, 1.2, 0.8))

        # 1. A rounded rim (the eight top and vertical edges), then the recess: inset the
        # flat top and push it in. Bevelling first keeps the corners clean -- an inset's
        # diagonal edge at a corner would cross the rounded patch there.
        def rim(_i, p):
            corner = abs(abs(p[0]) - 1) < 1e-6 and abs(abs(p[1]) - 0.6) < 1e-6
            top_edge = abs(p[2] - 0.4) < 1e-6 and (
                abs(abs(p[0]) - 1) < 1e-6 or abs(abs(p[1]) - 0.6) < 1e-6
            )
            return corner or top_edge

        edges = m.select("edges", predicate=rim, expect=8)
        step("body", "bevel_edges", m.bevel_edges, edges, 0.08, segments=3, profile=0.5)
        top = m.select("faces", normal=((0, 0, 1), 1), box=((-1, -1, 0.39), (1, 1, 0.41)))
        step("body", "inset_faces", m.inset_faces, top, 0.1)
        step("body", "extrude_faces", m.extrude_faces, top, -0.08)  # the inset's inner face
        m.set_group(m.select("faces", box=((-0.83, -0.43, 0.31), (0.83, 0.43, 0.41))), 1)
        # 2. A duct along x: open both ends and bridge them.
        ends = m.select("faces", predicate=lambda _i, p: abs(abs(p[0]) - 1) < 1e-6, expect=2)
        step("body", "inset_faces", m.inset_faces, ends, 0.25)
        holes = ends  # an inset keeps the original faces inside the new rings
        step("body", "delete_faces", m.delete_faces, holes)
        loops = m.select("edges", boundary=True)
        step("body", "bridge_edge_loops", m.bridge_edge_loops, loops, segments=3)
        m.set_group(
            m.select(
                "faces",
                predicate=lambda _i, p: abs(p[0]) < 0.999 and abs(p[1]) < 0.28 and abs(p[2]) < 0.08,
            ),
            3,
        )
        # 3. Two panel lines across the front (and around its rounded corners), and the
        # band between them pressed in.
        for z in (-0.25, 0.15):
            line = [(-1.2, -0.7, z), (1.2, -0.7, z)]
            step("body", "knife_project", m.knife_project, line, (0, 1, 0), through=False)
        panel = m.select(
            "faces",
            normal=((0, -1, 0), 1),
            box=((-0.59, -0.61, -0.24), (0.59, -0.59, 0.14)),
            expect=1,
        )
        step("body", "inset_faces", m.inset_faces, panel, 0.03)
        step("body", "extrude_faces", m.extrude_faces, panel, -0.025)  # the inner face
        m.set_group(panel, 2)
        # 4. A chamfered back-bottom edge.
        step(
            "body",
            "bisect",
            m.bisect,
            (0, 0.6, -0.25),
            (0, 1, -1),
            clear="front",
            fill=True,
        )
        # 5. Shading.
        step("body", "set_auto_smooth", m.set_auto_smooth, 40)
        step("body", "set_weighted_normals", m.set_weighted_normals, weight=100)
        cleanup = step("body", "cleanup", m.cleanup)
    return body, cleanup.data["report"]


def build_knob(scene: Scene, step: Steps):
    knob = scene.add_object("Knob", position=(0.45, 0.0, 0.32))
    profile = [(0.0, 0.0), (0.16, 0.0), (0.16, 0.05), (0.14, 0.07), (0.14, 0.15), (0.11, 0.18)]
    profile.append((0.0, 0.18))
    with knob.edit_mesh() as m:
        step("knob", "add_lathe", m.add_lathe, profile, segments=48)
        step("knob", "set_auto_smooth", m.set_auto_smooth, 35)
    bezel = scene.add_object("Bezel", position=(0.45, 0.0, 0.32))
    with bezel.edit_mesh() as m:
        m.add_grid(1, 1, size=0.04)  # a square in the XY plane...
        square = m.select("faces", expect=1)
        m.rotate(square, 90, (1, 0, 0))  # ...stood up in XZ...
        m.translate(square, (0.21, 0.0, 0.02))  # ...beside the knob
        step("bezel", "spin", m.spin, square, 360, 48)
        step("bezel", "recalculate_normals", m.recalculate_normals)  # facing outward
        step("bezel", "set_auto_smooth", m.set_auto_smooth, 50)
    return knob, bezel


def build_cap(scene: Scene, step: Steps):
    t0 = time.perf_counter()
    geometry = scene.add_geometry("Cap", "cylinder", radius=0.22, depth=0.16, segments=24)
    geometry.add_modifier("subdivision", levels=2, crease=3)
    step.log.append(
        {
            "part": "cap",
            "operation": "subdivision (levels 2, crease 3)",
            "created": {"triangles": geometry.evaluate().triangle_count},
            "diagnostics": [],
            "ms": round(1000 * (time.perf_counter() - t0), 2),
        }
    )
    return scene.add_object("Cap", geometry=geometry, position=(-0.45, 0.0, 0.40))


def stage(scene: Scene, body, knob, bezel, cap) -> None:
    paint = scene.create_material("Paint", base_color="#d9822b", metallic=0.3, roughness=0.35)
    dark = scene.create_material("Recess", base_color="#2b2f36", roughness=0.6)
    glow = scene.create_material(
        "Panel glow", base_color="#1b2a30", emissive="#38c6e8", emissive_strength=0.8
    )
    duct = scene.create_material("Duct", base_color="#444a52", metallic=0.8, roughness=0.3)
    rubber = scene.create_material("Rubber", base_color="#1d1d1f", roughness=0.8)
    chrome = scene.create_material("Chrome", base_color="#d8dde3", metallic=1.0, roughness=0.15)
    cap_paint = scene.create_material("Cap paint", base_color="#2f6db5", roughness=0.4)
    scene.assign_material(body, paint)
    scene.assign_material(body, dark, group=1)
    scene.assign_material(body, glow, group=2)
    scene.assign_material(body, duct, group=3)
    scene.assign_material(knob, rubber)
    scene.assign_material(bezel, chrome)
    scene.assign_material(cap, cap_paint)
    floor = scene.add_plane("Floor", size=12, position=(0, 0, -0.4))
    scene.assign_material(floor, scene.create_material("Floor", base_color="#9aa0a6"))
    scene.add_directional_light("Key", direction=(-0.5, 0.7, -1.0), intensity=2.4)
    scene.add_directional_light("Fill", direction=(0.8, 0.3, -0.4), intensity=0.6)
    scene.add_directional_light("Back light", direction=(0.4, -0.9, -0.5), intensity=1.2)


def render_views(scene: Scene, width: int, height: int, backend: str):
    views = []
    cam = scene.add_camera("Front")
    cam.frame(
        ["Power cell", "Knob", "Cap"],
        direction=(-3.2, 3.4, -2.2),
        margin=0.08,
        aspect=width / height,
    )
    back = scene.add_camera("Back")
    back.frame(["Power cell"], direction=(2.6, -3.4, -0.7), margin=0.08, aspect=width / height)
    for camera in (cam, back):
        views.append(render(scene, camera, width, height, backend))
    return views


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="smaller images (CI)")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (400, 300) if args.quick else (800, 600)
    failures: list[str] = []
    scene = Scene("Hard surface")
    step = Steps()
    t0 = time.perf_counter()
    body, cleanup = build_body(scene, step)
    knob, bezel = build_knob(scene, step)
    cap = build_cap(scene, step)
    seconds = time.perf_counter() - t0
    stage(scene, body, knob, bezel, cap)

    parts = {}
    for obj in (body, knob, bezel, cap):
        report = object_report(obj, self_intersections=True)
        s = report["summary"]
        parts[obj.name] = {
            "tier": s["tier"],
            "triangles": s["triangle_count"],
            "components": s["components"],
            "volume": s.get("volume"),
            "min_angle_degrees": round(s["min_angle_degrees"], 2),
        }
        if s["tier"] != "solid":
            failures.append(f"{obj.name} is {s['tier']}, not solid")
    repairs = {e["step"]: e["fixed"] + e.get("remaining", 0) for e in cleanup}
    if any(repairs.values()):
        failures.append(f"cleanup found something to repair: {repairs}")
    lo, hi = body.mesh.bounds()
    if not (math.isclose(hi[0], 1.0, abs_tol=1e-5) and math.isclose(lo[2], -0.4, abs_tol=1e-5)):
        failures.append(f"the body's bounds changed: {lo} {hi}")

    t1 = time.perf_counter()
    views = render_views(scene, width, height, args.backend)
    render_seconds = time.perf_counter() - t1
    image = np.concatenate([v.image for v in views], axis=1)
    png = encode_png(image, text={"Software": f"Aether3D {__version__}"})
    (out / "hard_surface.png").write_bytes(png)
    report = {
        "aether3d": __version__,
        "modeling_seconds": round(seconds, 3),
        "operations": step.log,
        "parts": parts,
        "cleanup": cleanup,
        "render": {
            "backend": views[0].backend,
            "size": [width, height],
            "seconds": round(render_seconds, 2),
        },
        "failures": failures,
    }
    (out / "hard_surface.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for entry in step.log:
        made = ", ".join(f"{v} {k}" for k, v in entry["created"].items()) or "-"
        print(f"  {entry['part']:<6} {entry['operation']:<32} {made:<40} {entry['ms']:8.2f} ms")
    for name, info in parts.items():
        print(f"{name}: {info['tier']}, {info['triangles']} triangles")
    print(f"rendered on {views[0].backend['name']} in {render_seconds:.2f} s")
    print(f"wrote {out / 'hard_surface.png'} and {out / 'hard_surface.json'}")
    for failure in failures:
        print("FAILED:", failure, file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
