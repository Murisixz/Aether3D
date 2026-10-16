"""Visual feedback (Stage 12): what observing, diagnosing and correcting cost.

Run from the repository root:

    python benchmarks/bench_feedback.py [--json OUT.json] [--quick | --smoke] [--no-gpu]
        [--adapter SPEC]

Groups (medians of repeated runs after warm-up, the garbage collector paused while each
runs, as in ``bench_engine.py``):

* ``stats`` -- the NumPy per-object reductions over an object-index buffer (the language
  decision's evidence): 640 x 360 and 1920 x 1080 with 3,000 objects;
* ``observe`` -- one observation (render, object-index and depth readback, statistics) of
  the furnished room (11 objects + groups), the castle (2,190 objects) and the village
  (11,270 objects) at 640 x 360, on the CPU and the GPU rasterizers: cold (the first after
  building), after one object moved (a new revision), and reused (same revision); the
  retained render world against the full extraction after one move;
* ``diagnose`` -- the checks of the room's and the castle's expectations, observations
  already made (what a diagnosis adds to the renders);
* ``correct`` -- applying one verified correction on the room (the step, a new render and
  a new diagnosis), and a preview;
* ``loop`` -- the whole loop from the first draft to acceptance: the room and the garden
  view (examples/feedback_room.py, examples/feedback_composition.py).

``--smoke`` runs every group once, on the room only, with small images: it checks that
the benchmark works (CI), and its numbers mean nothing.

Results depend on the machine: compare runs on the same machine only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "examples"))

from bench_engine import machine, median_time  # noqa: E402


def ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


SIZE = {"width": 640, "height": 360}
"""The observations' size (``--smoke`` makes it smaller)."""


# ---------------------------------------------------------------------------------------


def group_stats(quick: bool, smoke: bool = False) -> dict[str, Any]:
    from functools import partial

    from aether3d.feedback.observe import image_statistics, luminance_of, pixel_statistics

    rng = np.random.default_rng(1)
    repeat = 1 if smoke else 3 if quick else 7
    out = {}
    for w, h in ((160, 90),) if smoke else ((640, 360), (1920, 1080)):
        n = 300 if smoke else 3000
        index = np.full((h, w), -1, np.int32)
        for k in range(n):
            x0, y0 = rng.integers(0, w - 30), rng.integers(0, h - 30)
            index[y0 : y0 + rng.integers(1, 30), x0 : x0 + rng.integers(1, 30)] = k
        depth = rng.random((h, w)).astype(np.float32)
        image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        lum, source = luminance_of(image, None)
        out[f"{w}x{h}"] = {
            "objects": n,
            "pixel_statistics_ms": ms(
                median_time(partial(pixel_statistics, index, depth, lum, n), repeat)
            ),
            "luminance_ms": ms(median_time(partial(luminance_of, image, None), repeat)),
            "image_statistics_ms": ms(
                median_time(partial(image_statistics, image, index, lum, source), repeat)
            ),
        }
    return out


def _scenes(quick: bool, smoke: bool = False):
    import feedback_room
    import mcp_feedback_agent

    from aether3d import Scene

    room = Scene("Room")
    room.execute_plan(feedback_room.PLAN)
    if smoke:
        return {"room": room}
    castle = Scene("Castle")
    castle.execute_plan(mcp_feedback_agent.first_draft())
    scenes = {"room": room, "castle": castle}
    if not quick:
        village = Scene("Village")
        village.execute_plan(
            json.loads((ROOT / "src/aether3d/plans/examples/village.json").read_text())
        )
        scenes["village"] = village
    return scenes


def _mover(scene):
    """An object to move by a millimeter (a new revision, one object changed)."""
    for obj in scene:
        if obj.mesh is not None and obj.parent is None and obj.name not in ("Floor", "ground"):
            return obj
    return next(o for o in scene if o.mesh is not None)


def _move_and_observe(scene, mover, kw, step):
    """Move one object by a millimeter (back and forth: a new revision) and observe."""
    from aether3d.feedback import observe

    mover.position = (mover.position[0] + step[0], *mover.position[1:])
    step[0] = -step[0]
    return observe(scene, **kw)


def group_observe(
    quick: bool, backends: list[str], adapter: str | None, smoke: bool = False
) -> dict[str, Any]:
    from functools import partial

    from aether3d.feedback import ObservationStore, observe

    out: dict[str, Any] = {}
    repeat = 1 if smoke else 3 if quick else 7
    for name, scene in _scenes(quick, smoke).items():
        row: dict[str, Any] = {"objects": len(scene)}
        camera = scene.active_camera
        mover = _mover(scene)
        for backend in backends:
            kw = {**SIZE, "backend": backend}
            if backend == "gpu" and adapter:
                kw["adapter"] = adapter
            t0 = time.perf_counter()
            obs = observe(scene, **kw)
            cold = time.perf_counter() - t0
            moved = partial(_move_and_observe, scene, mover, kw, [0.001])
            store = ObservationStore()
            observe(scene, store=store, **kw)
            entry = {
                "cold_ms": ms(cold),
                "render_ms": ms(obs.timings["render"]),
                "metrics_ms": ms(obs.timings["metrics"]),
                "one_moved_ms": ms(median_time(moved, repeat)),
                "reused_ms": ms(median_time(partial(observe, scene, store=store, **kw), 5)),
            }
            os.environ["AETHER3D_RETAINED_RENDER_WORLD"] = "0"
            try:
                entry["one_moved_full_extraction_ms"] = ms(median_time(moved, min(repeat, 5)))
            finally:
                del os.environ["AETHER3D_RETAINED_RENDER_WORLD"]
            if backend == "gpu":
                entry["adapter"] = obs.renderer.get("adapter", {}).get("name")
            row[backend] = entry
        row["camera"] = camera.name if camera is not None else None
        out[name] = row
    return out


def group_diagnose(quick: bool, backend: str, smoke: bool = False) -> dict[str, Any]:
    import feedback_room
    import mcp_feedback_agent

    from aether3d import FeedbackSession, Scene

    out = {}
    cases = {
        "room": (feedback_room.PLAN, feedback_room.EXPECTATIONS),
        "castle": (mcp_feedback_agent.first_draft(), mcp_feedback_agent.EXPECTATIONS),
    }
    if smoke:
        del cases["castle"]
    for name, (plan, expectations) in cases.items():
        scene = Scene(name)
        scene.execute_plan(plan)
        fs = FeedbackSession(scene, expectations, backend=backend, **SIZE)
        first = fs.diagnose()  # renders the views

        def checks(fs=fs):
            fs._diagnoses.clear()  # the same renders, the checks again
            return fs.diagnose()

        out[name] = {
            "objects": len(scene),
            "checks": len(fs.expectations.checks),
            "findings": len(first.findings),
            "first_diagnosis_ms": ms(first.timings["total"]),
            "checks_ms": ms(median_time(checks, 1 if smoke else 3 if quick else 5)),
            "reused_ms": ms(median_time(fs.diagnose, 5)),
        }
    return out


def group_correct(quick: bool, backend: str, smoke: bool = False) -> dict[str, Any]:
    import feedback_room

    from aether3d import FeedbackSession, Scene

    def fresh():
        scene = Scene("Room")
        scene.execute_plan(feedback_room.PLAN)
        return FeedbackSession(
            scene,
            feedback_room.EXPECTATIONS,
            backend=backend,
            policy={"auto_apply": ["safe", "verify"]},
            **SIZE,
        )

    applies, previews, proposes = [], [], []
    for _ in range(1 if smoke else 3 if quick else 5):
        fs = fresh()
        corrections = fs.propose()  # diagnoses first
        t0 = time.perf_counter()
        fs.propose(corrections.diagnosis)
        proposes.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fs.preview(corrections)
        previews.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        result = fs.apply(corrections)
        applies.append(time.perf_counter() - t0)
        assert result.status == "applied", result.to_dict()
    return {
        "propose_ms": ms(float(np.median(proposes))),
        "preview_ms": ms(float(np.median(previews))),
        "apply_and_verify_ms": ms(float(np.median(applies))),
        "applied_proposals": len(result.applied),
    }


def group_loop(quick: bool, backend: str, smoke: bool = False) -> dict[str, Any]:
    import feedback_composition
    import feedback_room

    from aether3d import FeedbackSession, Scene
    from aether3d.feedback import run_loop

    out = {}
    for name, module, policy in (
        ("room", feedback_room, {"auto_apply": ["safe", "verify"]}),
        (
            "garden",
            feedback_composition,
            {"auto_apply": ["safe", "verify"], "max_translation": 40, "max_exposure_change": 6},
        ),
    ):
        times, iterations, renders = [], [], []
        for _ in range(1 if smoke else 3 if quick else 5):
            scene = Scene(name)
            scene.execute_plan(module.PLAN)
            fs = FeedbackSession(scene, module.EXPECTATIONS, backend=backend, policy=policy, **SIZE)
            t0 = time.perf_counter()
            report = run_loop(fs, budget={"max_iterations": 8})
            times.append(time.perf_counter() - t0)
            assert report.stop_reason == "accepted", report.stop_reason
            iterations.append(len(report.iterations))
            renders.append(report.budget["used"]["renders"])
        out[name] = {
            "loop_ms": ms(float(np.median(times))),
            "iterations": int(np.median(iterations)),
            "renders": int(np.median(renders)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the results here")
    parser.add_argument("--quick", action="store_true", help="fewer repeats, no village")
    parser.add_argument("--smoke", action="store_true", help="once, the room only (CI)")
    parser.add_argument("--no-gpu", action="store_true", help="CPU only")
    parser.add_argument("--adapter", help="GPU adapter (default: the best)")
    args = parser.parse_args(argv)
    backends = ["cpu"] if args.no_gpu else ["cpu", "gpu"]
    if not args.no_gpu:
        from aether3d.rendering.gpu import gpu_status

        if not gpu_status(adapter=args.adapter)["available"]:
            backends = ["cpu"]
    fast = backends[-1]
    smoke = args.smoke
    if smoke:
        SIZE.update(width=480, height=270)
    t0 = time.perf_counter()
    results: dict[str, Any] = {"machine": machine(), "groups": {}}
    groups = results["groups"]
    groups["stats"] = group_stats(args.quick, smoke)
    print("stats", json.dumps(groups["stats"]))
    groups["observe"] = group_observe(args.quick, backends, args.adapter, smoke)
    print("observe", json.dumps(groups["observe"]))
    groups["diagnose"] = {b: group_diagnose(args.quick, b, smoke) for b in backends}
    print("diagnose", json.dumps(groups["diagnose"]))
    groups["correct"] = {b: group_correct(args.quick, b, smoke) for b in backends}
    print("correct", json.dumps(groups["correct"]))
    groups["loop"] = {b: group_loop(args.quick, b, smoke) for b in backends}
    print("loop", json.dumps(groups["loop"]))
    results["seconds"] = round(time.perf_counter() - t0, 1)
    results["backends"] = backends
    results["fast_backend"] = fast
    results["size"] = dict(SIZE)
    results["mode"] = "smoke" if smoke else "quick" if args.quick else "full"
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
