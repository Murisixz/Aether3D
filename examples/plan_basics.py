"""Scene plans from Python: build, check and change a scene in one call each (Stage 11).

Run from the repository root:

    python examples/plan_basics.py [OUTPUT_DIR] [--quick]

A *scene plan* is JSON-compatible data (``aether3d.scene-plan``, version 1) that builds
or changes a scene in one call: it is checked as a whole, then applied as one undo step,
all or nothing. The same plans work from the CLI (``aether3d execute-plan``) and from AI
agents over MCP (``aether3d mcp serve``; see ``mcp_client.py``). This example:

1. checks a plan without a scene (``validate_plan``) and with one (a dry run);
2. builds a market square with it: a paved square and a fountain, a row of market stalls
   from one prefab with varied canopies, a wall with a gate left open by ``where``,
   scattered trees, lamps on a circle, the sun and a camera -- one call, one undo step;
3. sends a plan with mistakes: every problem comes back at once, with a stable code, a
   JSON pointer and a hint, and nothing changes;
4. changes the square with a plan of operations (it names the revision it was written
   for), and shows that a plan for an older revision is refused;
5. adds benches with an idempotency key, and sends the same plan again: it applies once;
6. undoes the last two plans (one step each) and redoes them;
7. renders a small preview with the CPU rasterizer.

Writes ``plan_basics.png`` and ``plan_basics.json``. Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from aether3d import PlanError, Scene, validate_plan

HERE = Path(__file__).resolve().parent

PLAN = "aether3d.scene-plan"

SQUARE = {
    "format": PLAN,
    "version": 1,
    "label": "Market square",
    "parameters": {"stalls": 6},
    "materials": {
        "stone": {"preset": "stone"},
        "wood": {"preset": "wood"},
        "grass": {"preset": "grass"},
        "red": {"base_color": "#b8322a"},
        "blue": {"base_color": "#2f5fa8"},
        "gold": {"base_color": "#d9a32e"},
        "leaves": {"base_color": "#3f7d2e"},
        "water": {"base_color": "#3d7fb3", "roughness": 0.1},
    },
    "meshes": {
        "block": {"primitive": "cube", "size": 1},
        "post": {"primitive": "box", "size": [0.12, 0.12, 2.3]},
    },
    "prefabs": {
        # A stall: its canopy, with the counter and four posts under it.
        "stall": {
            "mesh": {"primitive": "box", "size": [2.8, 1.8, 0.1]},
            "tags": ["stall"],
            "children": {
                "counter": {
                    "mesh": {"primitive": "box", "size": [2.4, 0.8, 1]},
                    "material": "@wood",
                    "position": [0, -0.4, -1.85],
                },
                "posts": {
                    "mesh": "@post",
                    "material": "@wood",
                    "instances": {
                        "positions": [
                            [-1.3, -0.8, -1.2],
                            [1.3, -0.8, -1.2],
                            [-1.3, 0.8, -1.2],
                            [1.3, 0.8, -1.2],
                        ]
                    },
                },
            },
        },
        "tree": {
            "tags": ["tree"],
            "children": {
                "trunk": {
                    "mesh": {"primitive": "cylinder", "radius": 0.2, "depth": 2, "segments": 8},
                    "color": "#6b4a2b",
                    "position": [0, 0, 1],
                },
                "crown": {
                    "mesh": {"primitive": "sphere", "radius": 1.2, "segments": 12, "rings": 8},
                    "material": "@leaves",
                    "position": [0, 0, 2.8],
                },
            },
        },
    },
    "objects": {
        "ground": {"mesh": {"primitive": "plane", "size": 80}, "material": "@grass"},
        "square": {
            "mesh": {"primitive": "box", "size": [24, 24, 0.1]},
            "material": "@stone",
            "position": [0, 0, 0.05],
        },
        "fountain": {
            "mesh": {"primitive": "cylinder", "radius": 2, "depth": 0.8, "segments": 32},
            "material": "@stone",
            "position": [0, 0, 0.5],
            "tags": ["fountain"],
            "children": {
                "water": {
                    "mesh": {"primitive": "cylinder", "radius": 1.8, "depth": 0.1, "segments": 32},
                    "material": "@water",
                    "position": [0, 0, 0.36],
                }
            },
        },
        "stalls": {
            "prefab": "@stall",
            "name": "Stall {i}",
            "position": [-8.5, -7, 2.4],
            "array": {"count": {"expr": "stalls"}, "offset": [3.4, 0, 0]},
            "vary": {"material": {"cycle": ["@red", "@blue", "@gold"]}},
        },
        "wall": {
            "mesh": "@block",
            "material": "@stone",
            "tags": ["wall"],
            "position": [-12, 12.5, 0.5],
            "grid": {"count": [25, 1, 3]},
            "where": {"expr": "not (x >= 11 and x <= 13 and z <= 1)"},  # the gate
        },
        "trees": {
            "prefab": "@tree",
            "name": "Tree {i}",
            "scatter": {
                "count": 14,
                "seed": 4,
                "box": {"min": [-26, 16], "max": [26, 30]},
                "min_distance": 3.5,
            },
            "vary": {"scale": {"uniform": [0.8, 1.3], "seed": 2}},
        },
        "lamps": {
            "mesh": {"primitive": "cylinder", "radius": 0.08, "depth": 3, "segments": 8},
            "color": "#303030",
            "tags": ["lamp"],
            "circle": {"count": 8, "radius": 6, "z": 1.5},
        },
        "sun": {"light": {"type": "directional", "intensity": 2.2}, "direction": [-0.4, 0.5, -0.8]},
        "camera": {
            "camera": {"fov": 45},
            "position": [18, -26, 15],
            "look_at": [0, 2, 0],
            "active": True,
        },
    },
    "world": {"background": "#a8cbe8"},
    # Answered right after the plan is applied, in the same call.
    "queries": [
        {"query": "overlaps", "objects": "@stalls", "against": ["@fountain", "@lamps"]},
        {"query": "raycast", "origin": [0, 0, 20], "direction": [0, 0, -1]},
    ],
}

# Four mistakes: an undefined symbol, a string for a number, an empty grid, and a delete
# by filters without saying how many objects it means.
BROKEN = {
    "format": PLAN,
    "version": 1,
    "label": "Benches (with mistakes)",
    "objects": {
        "bench": {"mesh": "@seat", "position": [0, 4, "0.3"]},
        "posts": {"mesh": {"primitive": "cylinder"}, "grid": {"count": [0, 2]}},
    },
    "operations": [{"op": "delete", "select": {"tags": ["stall"]}}],
}


def change_plan(revision: int) -> dict:
    """Edits by name and tag: existing things are named without "@"."""
    return {
        "format": PLAN,
        "version": 1,
        "label": "Close two stalls, paint the lamps, fell two trees",
        "expected_revision": revision,
        "materials": {"grey": {"base_color": "#8a8a8a"}},
        "operations": [
            {
                "op": "update",
                "select": {"names": ["Stall 0", "Stall 5"]},
                "set": {"add_tags": ["closed"], "material": "@grey"},
            },
            {
                "op": "update",
                "select": {"tags": ["lamp"], "expect": 8},
                "set": {"color": "#2b4a3a"},
            },
            {"op": "delete", "select": ["Tree 3", "Tree 7"], "recursive": True},
        ],
        "queries": [{"query": "nearest", "point": [0, 0, 1], "count": 3}],
    }


BENCHES = {
    "format": PLAN,
    "version": 1,
    "label": "Benches",
    "idempotency_key": "market-benches-1",
    "objects": {
        "benches": {
            "mesh": {"primitive": "box", "size": [1.8, 0.5, 0.45]},
            "material": "wood",  # the material the first plan made, by name
            "tags": ["bench"],
            "circle": {
                "count": 4,
                "radius": 3.6,
                "z": 0.225,
                "orient": "tangent",
                "start_angle": 45,
            },
        }
    },
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="a smaller preview")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    report: dict = {}

    def check(ok: bool, what: str) -> None:
        print(f"   {'ok' if ok else 'FAILED'}: {what}")
        if not ok:
            problems.append(what)

    print("1. Check a plan before building")
    static = validate_plan(SQUARE)
    print(f"   without a scene: valid={static.ok}, at most {static.estimate['objects']} objects")
    check(static.ok, "the market square plan is valid")
    scene = Scene("Market square")
    dry = scene.execute_plan(SQUARE, mode="dry_run")
    print(f"   dry run: {dry.counts['objects']} objects, scene still has {len(scene)}")
    check(dry.status == "dry_run" and len(scene) == 0, "a dry run changes nothing")

    print("2. Build the square: one call, one undo step")
    started = time.perf_counter()
    built = scene.execute_plan(SQUARE)
    seconds = time.perf_counter() - started
    print(
        f"   {built.counts['objects']} objects in {seconds * 1e3:.1f} ms: {built.counts['by_kind']}"
    )
    print(f"   history step {built.step['index']}: {built.step['label']!r}")
    overlaps, below = built.queries
    print(f"   stalls touching the fountain or a lamp: {overlaps['result']}")
    print(f"   straight down from above the center: {below['result']['object']['name']}")
    check(built.status == "applied" and len(scene) == dry.counts["objects"], "applied")
    check(overlaps["result"] == [], "no stall stands in the fountain or on a lamp")
    check(below["result"]["object"]["name"] == "fountain/water", "the fountain is at the center")
    names = {o.name for o in scene}
    gate = {"wall 12", "wall 37"} & names  # x = 12, z = 0 and 1 (x varies fastest)
    check(not gate and "wall 62" in names, "the gate is left open by where, with a lintel")
    report["built"] = {"objects": built.counts["objects"], "ms": round(seconds * 1e3, 1)}

    print("3. A plan with mistakes: every problem at once, nothing changes")
    revision = scene.revision
    try:
        scene.execute_plan(BROKEN)
        check(False, "the broken plan is refused")
    except PlanError as error:
        for issue in error.issues:
            print(f"   {issue['code']:<18} {issue['pointer']:<28} {issue['message']}")
        codes = sorted(i["code"] for i in error.issues)
        check(
            codes == ["INVALID_VALUE", "INVALID_VALUE", "OUT_OF_RANGE", "UNKNOWN_REFERENCE"],
            "four problems, each with its code and pointer",
        )
        report["errors"] = error.issues
    check(scene.revision == revision, "the scene did not change")

    print("4. Change the square with operations")
    plan = change_plan(scene.revision)
    changed = scene.execute_plan(plan)
    for op in changed.operations:
        print(f"   {op['op']}: {op['matched']} object(s)")
    check([op["matched"] for op in changed.operations] == [2, 8, 2], "the edits hit what they name")
    names = {o.name for o in scene}
    check(not {"Tree 3", "Tree 3/trunk", "Tree 7/crown"} & names, "two trees removed")
    try:
        scene.execute_plan(plan)  # written for the revision before the change
        check(False, "a plan for an old revision is refused")
    except PlanError as error:
        print(f"   resent: {error.code}: {error.message}")
        check(error.code == "REVISION_MISMATCH", "a plan for an old revision is refused")

    print("5. Retries are safe with an idempotency key")
    first = scene.execute_plan(BENCHES)
    again = scene.execute_plan(BENCHES)
    print(f"   first: {first.status} ({first.counts['objects']} benches); again: {again.status}")
    benches = [o for o in scene if o.has_tag("bench")]
    check(first.status == "applied" and again.status == "already_applied", "applied once")
    check(len(benches) == 4 and all(o.material.name == "wood" for o in benches), "4 wooden benches")

    print("6. Undo the last two plans, then redo them")
    count = len(scene)
    undone = scene.history.undo(2)
    print(f"   undone: {[entry.label for entry in undone]}")
    check(
        len(scene) == count - 4 + 6 and "Tree 3" in {o.name for o in scene},
        "the trees are back and the benches gone",
    )
    scene.history.redo(2)
    check(len(scene) == count, "redo brings the edits back")

    print("7. A preview (CPU rasterizer)")
    width, height = (320, 180) if args.quick else (960, 540)
    result = scene.render(width=width, height=height, backend="cpu")
    image = result.save(out / "plan_basics.png")
    print(f"   {result.backend['name']} {width}x{height}: {image}")
    check(result.backend["name"] == "cpu", "rendered on the CPU, and reported so")

    report.update(
        objects=len(scene),
        history=[entry.label for entry in scene.history.entries],
        problems=problems,
    )
    (out / "plan_basics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {image} and plan_basics.json")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
