"""The automatic feedback loop: acceptance, budgets, cancellation, approvals and resuming,
stagnation, cycles, determinism (Stage 12). Renders use the CPU rasterizer."""

from __future__ import annotations

import threading

import pytest

from aether3d import FeedbackError, FeedbackSession, Scene
from aether3d.feedback import FeedbackBudget, run_loop
from aether3d.serialization import scene_to_dict

EXPECT = {"format": "aether3d.expectations", "version": 1}


def room() -> Scene:
    scene = Scene("Room")
    scene.add_plane("Floor", size=10)
    scene.add_cube("Table", size=1, position=(0, 0, 0.5))
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, 1.35))  # floats 0.25 over the table
    scene.add_cube("Chair", size=0.5, position=(0, -1.2, 0.15))  # sunk 0.1 into the floor
    scene.add_cube("Lamp", size=0.4, position=(0.55, 0.6, 0.2))  # 0.15 into the table's side
    scene.add_camera("Cam", position=(3, -5, 3), look_at=(0, 0, 0.6))
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=1.0)
    return scene


CHECKS = [
    {"check": "rests_on", "objects": "Vase", "support": "Table"},
    {"check": "rests_on", "objects": ["Chair", "Lamp"], "support": "Floor"},
    {"check": "no_overlap", "objects": "Lamp", "against": "Table"},
    {"check": "visible", "objects": ["Table", "Vase", "Chair", "Lamp"]},
]


def session(scene, checks=CHECKS, **kw):
    document = {**EXPECT, "views": {"main": {"camera": "Cam"}}, "checks": checks}
    kw.setdefault("width", 160)
    kw.setdefault("height", 90)
    return FeedbackSession(scene, document, **kw)


def test_the_loop_corrects_a_room_and_every_step_can_be_undone():
    scene = room()
    start = scene_to_dict(scene)
    fs = session(scene, policy={"auto_apply": ["safe", "verify"]})
    report = run_loop(fs, budget={"max_iterations": 6})
    assert report.stop_reason == "accepted", report.to_dict()
    assert report.final["accepted"]
    assert scene.get_by_name("Vase").position[2] == pytest.approx(1.1, abs=1e-6)
    assert scene.get_by_name("Chair").position[2] == pytest.approx(0.25, abs=1e-6)
    lamp = scene.get_by_name("Lamp")
    assert lamp.position[0] >= 0.7 - 1e-4 or lamp.position[1] >= 0.7 - 1e-4  # out of the table
    applied = [it["applied"] for it in report.iterations if "applied" in it]
    assert all(a["status"] == "applied" for a in applied)
    labels = [e.label for e in scene.history.entries if e.label.startswith("Feedback")]
    assert len(labels) == sum(1 for a in applied)  # one undo step per application
    scene.history.goto(report.checkpoint)
    assert scene_to_dict(scene) == start
    used = report.budget["used"]
    assert used["iterations"] == len(report.iterations) and used["renders"] >= 1


def test_safe_only_policies_stop_with_pending_approvals_and_resume():
    scene = room()
    fs = session(scene)  # auto_apply: safe only
    report = run_loop(fs)
    assert report.stop_reason == "awaiting_approval"
    pending = {p["title"] for p in report.pending}
    assert any("Separate 'Lamp'" in t for t in pending)
    assert scene.get_by_name("Vase").position[2] == pytest.approx(1.1, abs=1e-6)  # safe: done
    corrections = fs.propose()
    lamp = [p for p in corrections.proposals if "Lamp" in p.title][0]
    assert lamp.safety == "verify" and not lamp.policy["auto"]
    result = fs.apply(corrections, [lamp.id])  # naming a verify proposal approves it
    assert result.status == "applied"
    report = run_loop(fs, resume=report.id)
    assert report.stop_reason == "accepted"


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        ({"max_iterations": 1}, "max_iterations"),
        ({"max_renders": 2}, "budget_renders"),
        ({"max_pixels": 2 * 160 * 90}, "budget_pixels"),
        ({"max_seconds": 1e-9}, "budget_time"),
        ({"max_memory": 2**20}, "budget_memory"),
    ],
)
def test_budgets_stop_the_loop(budget, reason):
    scene = room()
    scene.get_by_name("Vase").rotation_euler = (6, 0, 0)  # two corrections of one object:
    checks = [*CHECKS, {"check": "upright", "objects": "Vase"}]  # one per iteration
    fs = session(scene, checks, policy={"auto_apply": ["safe", "verify"]})
    if reason == "budget_memory":
        fs.observe(None, width=640, height=360)  # > 1 MiB of observations kept
    report = run_loop(fs, budget=budget)
    assert report.stop_reason == reason


def test_cancellation_stops_before_changing_anything():
    scene = room()
    start = scene_to_dict(scene)
    fs = session(scene)
    cancel = threading.Event()
    cancel.set()
    report = run_loop(fs, cancel=cancel)
    assert report.stop_reason == "cancelled" and not report.iterations
    assert scene_to_dict(scene) == start

    calls = []

    def progress(done, total, message):
        calls.append(message)
        if "applying" in message:
            cancel.set()

    cancel.clear()
    report = run_loop(fs, cancel=cancel, progress=progress)
    assert report.stop_reason == "cancelled"
    assert scene_to_dict(scene) == start  # cancelled before its first change
    assert calls[0].endswith("diagnosing")


def test_a_correction_that_does_not_help_is_not_tried_again():
    scene = Scene("Rug")
    scene.add_plane("Floor", size=10)
    rug = scene.add_cube("Rug", size=1, position=(0, 0, 0.6))
    rug.scale = (1, 1, 0.05)
    scene.add_camera("Cam", position=(3, -5, 3), look_at=(0, 0, 0.3))
    # Only rests on the floor if it also stays inside a box that the floor is below:
    # every lowering regresses the region check and is undone.
    checks = [
        {"check": "rests_on", "objects": "Rug", "support": "Floor"},
        {
            "check": "inside_region",
            "objects": "Rug",
            "region": {"box": {"min": [-5, -5, 0.5], "max": [5, 5, 3]}},
        },
    ]
    fs = session(scene, checks, policy={"auto_apply": ["safe", "verify"]})
    start = scene_to_dict(scene)
    report = run_loop(fs, budget={"max_iterations": 6})
    assert report.stop_reason in ("stagnated", "no_safe_progress")
    first = report.iterations[0]["applied"]
    assert first["status"] == "rolled_back" and "regression" in first["reason"]
    assert any("skipped_tried" in it for it in report.iterations[1:])
    assert scene_to_dict(scene) == start


def test_a_finding_that_comes_back_stops_the_loop():
    scene = Scene("Shelf")
    scene.add_plane("Floor", size=10)
    scene.add_cube("Table", size=1, position=(0, 0, 0.5))
    scene.add_cube("Cup", size=0.2, position=(0, 0, 1.75))
    scene.add_camera("Cam", position=(3, -5, 3), look_at=(0, 0, 0.6))
    checks = [  # contradictory: resting on the table puts it below the region
        {"check": "rests_on", "objects": "Cup", "support": "Table"},
        {
            "check": "inside_region",
            "objects": "Cup",
            "region": {"box": {"min": [-1, -1, 1.5], "max": [1, 1, 2]}},
        },
    ]
    fs = session(scene, checks, policy={"auto_apply": ["safe", "verify"], "verify": False})
    report = run_loop(fs, budget={"max_iterations": 8})
    assert report.stop_reason == "cycle_detected"
    assert report.iterations[-1]["returned"]


def test_the_loop_is_deterministic():
    reports = []
    for _ in range(2):
        scene = room()
        fs = session(scene, policy={"auto_apply": ["safe", "verify"]})
        report = run_loop(fs, budget={"max_iterations": 6})
        steps = [
            (a["applied"], [{k: v for k, v in c.items() if k != "id"} for c in a["changes"]])
            for it in report.iterations
            if (a := it.get("applied")) is not None
        ]
        placement = {o.name: (o.position, o.rotation, o.scale) for o in scene}
        reports.append((report.stop_reason, steps, placement, scene.render_settings.to_dict()))
    assert reports[0] == reports[1]  # bit for bit: same decisions, same final placement


def test_budget_documents_are_checked():
    with pytest.raises(FeedbackError) as exc:
        FeedbackBudget.coerce({"max_iterations": 0, "max_seconds": -1, "nope": 1})
    assert {i["pointer"] for i in exc.value.issues} == {
        "/budget/max_iterations",
        "/budget/max_seconds",
        "/budget/nope",
    }
    with pytest.raises(FeedbackError) as exc:
        run_loop(session(room()), resume="loop-unknown")
    assert exc.value.code == "UNKNOWN_LOOP"


def test_the_loop_on_the_gpu_makes_the_same_decisions(gpu):
    results = []
    for backend in ("cpu", "gpu"):
        scene = room()
        fs = session(
            scene,
            backend=backend,
            adapter=gpu.adapter if backend == "gpu" else None,
            policy={"auto_apply": ["safe", "verify"]},
        )
        report = run_loop(fs, budget={"max_iterations": 6})
        results.append(
            (report.stop_reason, [it.get("applied", {}).get("applied") for it in report.iterations])
        )
    assert results[0] == results[1]
