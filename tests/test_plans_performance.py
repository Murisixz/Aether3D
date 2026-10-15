"""Scene plans scale as designed: regression tests that do not assume a machine's speed.

Times are compared with each other on the same machine (the median of a few runs, with
generous margins), never with fixed budgets; sizes are deterministic and are checked
exactly. ``benchmarks/bench_plans.py`` measures the actual speed.
"""

from __future__ import annotations

import statistics
import time

from aether3d import Scene

PLAN = "aether3d.scene-plan"


def blocks(n: int) -> dict:
    return {
        "format": PLAN,
        "version": 1,
        "meshes": {"block": {"primitive": "cube", "size": 0.5}},
        "objects": {
            "blocks": {
                "mesh": "@block",
                "repeat": n,
                "position": [{"expr": "i % 100"}, {"expr": "floor(i / 100)"}, 0.25],
            }
        },
    }


def median_seconds(fn, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def test_building_scales_linearly():
    def build(n):
        return lambda: Scene("Blocks").execute_plan(blocks(n))

    build(500)()  # warm up
    small, large = median_seconds(build(1000), 3), median_seconds(build(8000), 3)
    assert large < 16 * small + 0.1, (small, large)  # 8x the objects: linear is 8x, not 64x


def test_undo_and_redo_of_a_plan_scale_linearly():
    def cycle(n):
        scene = Scene("Blocks")
        scene.execute_plan(blocks(n))

        def go():
            scene.history.undo()
            scene.history.redo()

        return go

    small, large = median_seconds(cycle(1000)), median_seconds(cycle(8000))
    assert large < 16 * small + 0.05, (small, large)


def test_a_small_edit_costs_the_same_in_a_large_scene():
    def edit(n):
        scene = Scene("Blocks")
        scene.execute_plan(blocks(n))
        names = iter(range(10**6))

        def go():
            k = next(names) % n
            scene.execute_plan(
                {
                    "format": PLAN,
                    "version": 1,
                    "operations": [
                        {"op": "update", "select": f"blocks {k}", "set": {"color": "#ff0000"}}
                    ],
                }
            )

        return go

    edit(200)()  # warm up
    small, large = median_seconds(edit(500), 9), median_seconds(edit(16000), 9)
    assert large < 3 * small + 0.002, (small, large)  # 32x the scene: about the same


def test_a_plans_history_is_one_step_of_linear_size():
    sizes = {}
    for n in (1000, 4000):
        scene = Scene("Blocks")
        result = scene.execute_plan(blocks(n))
        assert len(scene.history) == 1 and result.step["index"] == 1
        sizes[n] = result.step["bytes"]
        assert sizes[n] == scene.history.nbytes
    assert sizes[1000] < 1000 * 1000  # under 1 KB per object
    assert 3.5 < sizes[4000] / sizes[1000] < 4.5
