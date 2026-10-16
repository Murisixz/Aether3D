"""A furnished room with first-draft mistakes, found and corrected from its renders (Stage 12).

Run from the repository root:

    python examples/feedback_room.py [OUTPUT_DIR] [--backend auto|gpu|cpu] [--quick]

1. **Build** a living room with one scene plan: floor, two walls, a rug, a table and two
   chairs (groups of boards), a sofa, a standing lamp, a painting and a vase -- with the
   mistakes of a first draft: the vase floats above the table, a chair is sunk into the
   floor, the lamp stands partly inside the back wall.
2. **State the expectations** as data: furniture stands on whatever is below it (the
   rug or the floor), the vase on the table; nothing intersects the walls; the camera
   shows the main pieces; the exposure. Nothing says the painting must rest on anything:
   it hangs, and is not flagged.
3. **Observe and diagnose**: the findings, measured from the triangles and the render
   (gaps, penetrations, separations, visible fractions), with where each one is in the
   image.
4. **Correct** with the feedback loop: each correction is a typed action compiled to a
   scene plan, applied as one undo step, verified on a new render and kept only if it
   helps (here: safe and verify corrections are automatic).
5. **Check the result**: every expectation holds; the history returns exactly to the
   first draft (and back).

Writes ``feedback_room.png`` (the first draft and the corrected room, the corrected
objects marked) and ``feedback_room.json`` (findings, proposals, every loop iteration,
measurements, timings, the renderer) to OUTPUT_DIR (default ``examples/output``).
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import FeedbackSession, Scene, __version__
from aether3d.feedback import overlay, run_loop
from aether3d.rendering.png import encode_png
from aether3d.serialization import scene_to_dict

HERE = Path(__file__).resolve().parent

PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "label": "Living room (first draft)",
    "materials": {
        "oak": {"preset": "wood", "color": "#9a6b3f"},
        "fabric": {"preset": "fabric", "color": "#5b7896"},
        "plaster": {"base_color": "#e8e2d6", "roughness": 0.9},
        "floorboards": {"preset": "wood", "color": "#b98a5a", "scale": 0.6},
        "rug": {"preset": "fabric", "color": "#a33b2f"},
        "brass": {"preset": "metal", "color": "#c9a24b"},
        "shade": {"base_color": "#f3e7c9", "roughness": 0.8},
        "ceramic": {"base_color": "#2f6f8f", "roughness": 0.3},
        "canvas": {"base_color": "#d9a441", "roughness": 0.7},
    },
    "meshes": {"board": {"primitive": "cube", "size": 1}},
    "prefabs": {
        "table": {
            "tags": ["furniture"],
            "children": {
                "top": {
                    "mesh": "@board",
                    "material": "@oak",
                    "position": [0, 0, 0.73],
                    "scale": [1.4, 0.8, 0.04],
                },
                "legs": {
                    "mesh": "@board",
                    "material": "@oak",
                    "scale": [0.06, 0.06, 0.71],
                    "position": [0, 0, 0.355],
                    "instances": [
                        {"position": [-0.62, -0.32, 0]},
                        {"position": [0.62, -0.32, 0]},
                        {"position": [-0.62, 0.32, 0]},
                        {"position": [0.62, 0.32, 0]},
                    ],
                },
            },
        },
        "chair": {
            "tags": ["furniture"],
            "children": {
                "seat": {
                    "mesh": "@board",
                    "material": "@oak",
                    "position": [0, 0, 0.45],
                    "scale": [0.45, 0.45, 0.04],
                },
                "back": {
                    "mesh": "@board",
                    "material": "@oak",
                    "position": [0, 0.205, 0.72],
                    "scale": [0.45, 0.04, 0.5],
                },
                "legs": {
                    "mesh": "@board",
                    "material": "@oak",
                    "scale": [0.04, 0.04, 0.43],
                    "position": [0, 0, 0.215],
                    "instances": [
                        {"position": [-0.2, -0.2, 0]},
                        {"position": [0.2, -0.2, 0]},
                        {"position": [-0.2, 0.2, 0]},
                        {"position": [0.2, 0.2, 0]},
                    ],
                },
            },
        },
        "lamp": {
            "tags": ["furniture"],
            "children": {
                "base": {
                    "mesh": {"primitive": "cylinder", "radius": 0.18, "depth": 0.04},
                    "material": "@brass",
                    "position": [0, 0, 0.02],
                },
                "pole": {
                    "mesh": {"primitive": "cylinder", "radius": 0.02, "depth": 1.5},
                    "material": "@brass",
                    "position": [0, 0, 0.79],
                },
                "shade": {
                    "mesh": {
                        "primitive": "cone",
                        "radius_bottom": 0.25,
                        "radius_top": 0.14,
                        "depth": 0.3,
                    },
                    "material": "@shade",
                    "position": [0, 0, 1.6],
                },
            },
        },
    },
    "objects": {
        "Floor": {"mesh": {"primitive": "plane", "size": 7}, "material": "@floorboards"},
        "back_wall": {
            "name": "Back wall",
            "mesh": "@board",
            "material": "@plaster",
            "position": [0, 3.1, 1.4],
            "scale": [7, 0.2, 2.8],
            "tags": ["wall"],
        },
        "left_wall": {
            "name": "Left wall",
            "mesh": "@board",
            "material": "@plaster",
            "position": [-3.6, 0, 1.4],
            "scale": [0.2, 6.4, 2.8],
            "tags": ["wall"],
        },
        "Rug": {
            "mesh": "@board",
            "material": "@rug",
            "position": [0.3, 0.2, 0.01],
            "scale": [3.2, 2.2, 0.02],
            "tags": ["furniture"],
        },
        "Table": {"prefab": "@table", "position": [0.4, 0.3, 0.02]},
        "chair_a": {
            "name": "Chair A",
            "prefab": "@chair",
            "position": [0.1, -0.55, 0.02],
            "rotation_euler": [0, 0, 180],
        },
        "chair_b": {
            "name": "Chair B",
            "prefab": "@chair",
            "position": [1.25, 0.3, -0.06],
            "rotation_euler": [0, 0, -90],
        },
        "Sofa": {
            "mesh": {"primitive": "box", "size": [2.2, 0.9, 0.8]},
            "material": "@fabric",
            "position": [-1.6, 2.4, 0.4],
            "tags": ["furniture"],
        },
        "Lamp": {"prefab": "@lamp", "position": [1.9, 2.95, 0]},
        "Painting": {
            "mesh": "@board",
            "material": "@canvas",
            "position": [-1.6, 2.98, 1.75],
            "scale": [1.2, 0.04, 0.8],
        },
        "Vase": {
            "mesh": {"primitive": "cylinder", "radius": 0.08, "depth": 0.28},
            "material": "@ceramic",
            "position": [0.55, 0.35, 1.06],
        },
        "Sun": {
            "light": {"type": "directional", "intensity": 1.6, "angle": 2},
            "direction": [-0.45, 0.55, -0.7],
        },
        "Fill": {
            "light": {"type": "point", "intensity": 4, "range": 9},
            "position": [1.5, -2.5, 2.4],
        },
        "Camera": {
            "camera": {"fov": 52},
            "position": [3.4, -3.6, 2.3],
            "look_at": [0, 1.0, 0.7],
            "active": True,
        },
    },
}

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "Living room",
    "views": {"main": {"camera": "Camera"}},
    "checks": [
        {"id": "standing", "check": "rests_on", "objects": {"tags": ["furniture"]}},
        {"id": "vase-on-table", "check": "rests_on", "objects": "Vase", "support": "Table"},
        {
            "id": "clear-of-walls",
            "check": "no_overlap",
            "objects": {"tags": ["furniture"]},
            "against": {"tags": ["wall"]},
            "fix": {"move": "objects"},
        },
        {"id": "upright", "check": "upright", "objects": {"tags": ["furniture"]}},
        {
            "id": "in-view",
            "check": "visible",
            "objects": ["Sofa", "Table", "Lamp", "Vase"],
            "min_visible_fraction": 0.3,
        },
        {
            "id": "light",
            "check": "exposure",
            "mean_luminance": {"min": 0.05, "max": 0.6},
            "max_clipped_fraction": 0.05,
        },
    ],
}


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    out = np.full((h, left.shape[1] + gap + right.shape[1], 3), 255, dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] + gap :] = right
    return out


def findings_table(diagnosis) -> list[dict]:
    return [
        {
            "id": f.id,
            "check": f.check,
            "code": f.code,
            "basis": f.basis,
            "exact": f.exact,
            "subjects": [s["name"] for s in f.subjects],
            "magnitude": f.magnitude,
            "unit": f.unit,
            "message": f.message,
        }
        for f in diagnosis.findings
        if f.severity in ("error", "warning")
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="smaller images (CI)")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (480, 270) if args.quick else (960, 540)
    failures: list[str] = []
    report: dict = {"aether3d": __version__}

    # 1. Build the first draft (one plan, one undo step).
    t0 = time.perf_counter()
    scene = Scene("Living room")
    built = scene.execute_plan(PLAN)
    report["build"] = {"objects": built.counts["objects"], "seconds": time.perf_counter() - t0}
    draft = scene_to_dict(scene)

    # 2-3. Expectations, observation and diagnosis.
    feedback = FeedbackSession(
        scene,
        EXPECTATIONS,
        backend=args.backend,
        width=width,
        height=height,
        policy={"auto_apply": ["safe", "verify"]},
    )
    before = feedback.diagnose()
    report["before"] = {
        "accepted": before.accepted,
        "errors": before.errors,
        "findings": findings_table(before),
        "timings": before.timings,
    }
    # The objects to correct: the subjects of the error findings (not their supports).
    corrected = list(
        dict.fromkeys(
            s["id"]
            for f in before.findings
            if f.severity == "error"
            for s in f.subjects
            if s["role"] in ("subject", "either")
        )
    )
    first = feedback.observe("main", focus=corrected)
    image_before = overlay.annotate(first.image, first.index, first._marker_boxes())
    report["renderer"] = first.renderer
    proposals = feedback.propose(before)
    report["proposals"] = [
        {
            "id": p.id,
            "title": p.title,
            "safety": p.safety,
            "auto": p.policy["auto"],
            "actions": p.actions,
            "rationale": p.rationale,
        }
        for p in proposals.proposals
    ]

    # 4. The loop: corrections applied one step each, verified on new renders.
    t0 = time.perf_counter()
    loop = run_loop(feedback, budget={"max_iterations": 6})
    report["loop"] = {
        "stop_reason": loop.stop_reason,
        "seconds": time.perf_counter() - t0,
        "budget_used": loop.budget["used"],
        "iterations": [
            {
                "iteration": it["iteration"],
                "errors": it["diagnosis"]["errors"],
                "applied": (it.get("applied") or {}).get("applied"),
                "status": (it.get("applied") or {}).get("status"),
                "decision": ((it.get("applied") or {}).get("verification") or {}).get("decision"),
                "changes": (it.get("applied") or {}).get("changes"),
                "seconds": it["seconds"],
            }
            for it in loop.iterations
        ],
    }
    after = feedback.diagnose()
    report["after"] = {
        "accepted": after.accepted,
        "errors": after.errors,
        "findings": findings_table(after),
    }
    last = feedback.observe("main", focus=corrected)
    image_after = overlay.annotate(last.image, last.index, last._marker_boxes())

    # 5. Checks: the expectations hold, measured again; the history goes back and forth.
    if loop.stop_reason != "accepted" or not after.accepted:
        failures.append(f"the loop stopped with {loop.stop_reason}, accepted={after.accepted}")
    codes = {f["code"] for f in report["before"]["findings"]}
    for code in ("FLOATING", "SUNKEN", "INTERSECTION"):
        if code not in codes:
            failures.append(f"the first draft should show {code}")
    vase_bottom = scene.get_by_name("Vase").bounds().aabb.min[2]
    table_top = scene.get_by_name("Table/top").bounds().aabb.max[2]
    if abs(vase_bottom - table_top) > 1e-4:  # measured independently: world boxes
        failures.append(f"the vase's bottom is {vase_bottom - table_top:+.5f} m off the table")
    rug_top = scene.get_by_name("Rug").bounds().aabb.max[2]
    for name, ground in (
        ("Table", rug_top),
        ("Chair A", rug_top),
        ("Chair B", rug_top),
        ("Lamp", 0.0),
    ):
        low = scene.get_by_name(name).bounds(children=True).aabb.min[2]
        if abs(low - ground) > 1e-4:
            failures.append(f"{name} stands at z={low:.5f}, not at {ground:.5f}")
    if after.findings:  # not even the informational scan for penetrations finds anything
        failures.append(f"the corrected room still shows {[f.code for f in after.findings]}")
    chair, table = scene.get_by_name("Chair B"), scene.get_by_name("Table")
    clash = scene.overlaps(
        objects=[o.id for o in chair.iter_descendants()],
        against=[o.id for o in table.iter_descendants()],
    )
    if clash:  # measured independently: the exact triangle test of the spatial queries
        failures.append(f"Chair B intersects the table: {[(c.a.name, c.b.name) for c in clash]}")
    final = scene_to_dict(scene)
    scene.history.checkpoint("corrected")
    scene.history.goto(loop.checkpoint)
    if scene_to_dict(scene) != draft:
        failures.append("going back to the loop's checkpoint does not restore the first draft")
    scene.history.goto("corrected")
    if scene_to_dict(scene) != final:
        failures.append("going forward again does not restore the corrected room")
    report["history"] = [e.label for e in scene.history.entries][-6:]

    png = encode_png(
        side_by_side(image_before, image_after),
        text={"Software": f"Aether3D {__version__}"},
    )
    (out / "feedback_room.png").write_bytes(png)
    report["failures"] = failures
    (out / "feedback_room.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"first draft: {before.errors} error(s): " + ", ".join(sorted(codes)))
    for it in report["loop"]["iterations"]:
        print(
            f"  iteration {it['iteration']}: {it['errors']} error(s), applied {it['applied']} "
            f"({it['decision']})"
        )
    print(
        f"loop: {loop.stop_reason} in {report['loop']['seconds']:.2f} s on "
        f"{first.renderer['backend']} ({first.renderer['device']})"
    )
    print(f"wrote {out / 'feedback_room.png'} and {out / 'feedback_room.json'}")
    for failure in failures:
        print(f"CHECK FAILED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
