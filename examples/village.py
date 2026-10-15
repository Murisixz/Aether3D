"""A village of 11,270 objects from reusable structures, built with one plan (Stage 11).

Run from the repository root:

    python examples/village.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

The plan is the ``village`` example that ships with Aether3D (``aether3d://examples/village``
over MCP; ``aether3d/plans/examples/village.json`` in the package): three house prefabs
-- walls, roof, door, windows, chimney and a fenced front yard with a gate -- in blocks
of nine with trees and street lamps, on an 8x8 street grid, with a park, a plaza and 273
cars with varied paint. 11,270 objects that share 20 meshes.

1. Checks the plan without a scene, then builds the village: one call, one undo step.
2. Looks at what it made: counts by tag, and the questions the plan asks (no car stands
   on a lamp post).
3. Paints one door of one house with a small plan. The edit costs what one object costs,
   not what the village costs: it changes that object only and records a small undo step.
4. Undoes and redoes the edit, then the whole village.
5. Renders a preview. The GPU rasterizer draws the village in one instanced draw call per
   mesh; the CPU rasterizer is the portable fallback (``--backend cpu``).

Writes ``village.png`` and ``village.json`` (counts, times, draw calls). Times are of
this run on this machine; ``benchmarks/bench_plans.py`` measures them properly. Exits
with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import resources
from pathlib import Path

from aether3d import Scene, validate_plan
from aether3d.plans import select_objects

HERE = Path(__file__).resolve().parent

DOOR = "Block 3-0/houses 4/door"


def ms(seconds: float) -> float:
    return round(seconds * 1e3, 2)


def state(scene: Scene) -> dict:
    """What an edit could change, for every object (to see what one did)."""
    return {
        o.id: (o.name, o.position, o.rotation, o.scale, o.visible, o.material and o.material.id)
        for o in scene
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="a smaller preview")
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    report: dict = {}

    def check(ok: bool, what: str) -> None:
        print(f"   {'ok' if ok else 'FAILED'}: {what}")
        if not ok:
            problems.append(what)

    print("1. Check the plan, then build the village")
    text = (resources.files("aether3d.plans") / "examples" / "village.json").read_text("utf-8")
    plan = json.loads(text)
    started = time.perf_counter()
    static = validate_plan(plan)
    checked = time.perf_counter() - started
    print(f"   {len(text) / 1024:.1f} KiB of JSON, checked in {ms(checked)} ms: ", end="")
    print(f"at most {static.estimate['objects']} objects (where-filters make fewer)")
    scene = Scene("Village")
    started = time.perf_counter()
    built = scene.execute_plan(plan)
    seconds = time.perf_counter() - started
    counts = built.counts
    print(f"   {counts['objects']} objects in {seconds:.2f} s, one undo step: {counts['by_kind']}")
    print(f"   {counts['meshes']} meshes shared by {counts['mesh_objects_sharing_a_mesh']} objects")
    timings = {k: ms(v) for k, v in built.timings.items()}
    print(f"   stages (ms): {timings}")
    check(built.status == "applied" and counts["objects"] == len(scene) == 11270, "11,270 objects")
    report["build"] = {
        "objects": counts["objects"],
        "seconds": round(seconds, 3),
        "stages_ms": timings,
    }

    print("2. What it made")
    tags = {
        tag: len(select_objects(scene, {"tags": [tag]}))
        for tag in ("house", "fence", "tree", "lamp", "car")
    }
    print(f"   by tag: {tags}")
    overlaps, nearest = built.queries
    print(f"   cars on lamp posts: {overlaps['result']}; nearest the fountain: ", end="")
    print([n["object"]["name"] for n in nearest["result"]])
    check(tags["house"] == 62 * 9 and tags["car"] == 273, "558 houses and 273 cars")
    check(overlaps["result"] == [], "no car stands on a lamp post")
    report["tags"] = tags

    print("3. Paint one door")
    door = scene.get_by_name(DOOR)
    before = state(scene)
    edit = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "label": "Paint one door green",
        "expected_revision": scene.revision,
        "materials": {"door_green": {"base_color": "#2f6f3a"}},
        "operations": [{"op": "update", "select": DOOR, "set": {"material": "@door_green"}}],
    }
    started = time.perf_counter()
    edited = scene.execute_plan(edit)
    edit_seconds = time.perf_counter() - started
    after = state(scene)
    changed = [after[k][0] for k in before if after[k] != before[k]]
    step = edited.step
    ratio = seconds / edit_seconds
    print(f"   {ms(edit_seconds)} ms ({ratio:.0f}x less than the build); changed: {changed}")
    print(
        f"   undo step: {step['changes']} change(s), {step['bytes']} bytes (the build's: ", end=""
    )
    print(f"{built.step['changes']} changes, {built.step['bytes'] / 1e6:.1f} MB)")
    check(changed == [DOOR] and door.material.name == "door_green", "only that door changed")
    check(step["changes"] <= 2 and step["bytes"] < 4096, "its undo step is tiny")
    report["edit"] = {"ms": ms(edit_seconds), "changed": changed, "step": step}

    print("4. Undo and redo")
    started = time.perf_counter()
    scene.history.undo()
    undo_edit = time.perf_counter() - started
    check(door.material.name != "door_green", "undo restores the door")
    scene.history.redo()
    started = time.perf_counter()
    scene.history.undo(2)
    undo_all = time.perf_counter() - started
    check(len(scene) == 0, "two more undos remove the village")
    started = time.perf_counter()
    scene.history.redo(2)
    redo_all = time.perf_counter() - started
    print(f"   undo the edit: {ms(undo_edit)} ms; undo the village: {ms(undo_all)} ms; ", end="")
    print(f"redo both: {ms(redo_all)} ms")
    check(state(scene) == after, "redo brings back exactly the edited village")
    report["history_ms"] = {
        "undo_edit": ms(undo_edit),
        "undo_all": ms(undo_all),
        "redo_all": ms(redo_all),
    }

    print("5. A preview")
    width, height = (320, 180) if args.quick else (1280, 720)
    started = time.perf_counter()
    result = scene.render(width=width, height=height, backend=args.backend)
    render_seconds = time.perf_counter() - started
    name = result.backend["name"]
    calls = result.stats.get("draw_calls")
    where = f"GPU, {result.backend['adapter']['name']}" if name == "gpu" else "CPU"
    print(f"   {width}x{height} on the {where} in {render_seconds:.2f} s (first render: ", end="")
    print("includes creating the GPU device)" if name == "gpu" else "the CPU rasterizer)")
    if name == "gpu":
        print(f"   {calls} draw calls for {counts['by_kind']['mesh']} mesh objects")
        check(calls <= counts["meshes"], "at most one draw call per mesh")
    if args.backend == "gpu":
        check(name == "gpu", "rendered on the GPU, as asked")
    image = result.save(out / "village.png")
    report["render"] = {
        "backend": name,
        "size": [width, height],
        "seconds": round(render_seconds, 3),
        "draw_calls": calls,
    }

    report["problems"] = problems
    (out / "village.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {image} and village.json")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
