"""Showcase 2 (Stage 17): graph-curve repair, measured before and after -- and the same
edit made three ways (Python, a JSON animation script, MCP) with identical results.

A crane lowers a crate onto a shelf; its keys were made carelessly:

* automatic (Catmull-Rom) handles **overshoot**: the crate dips below the shelf's top
  before it settles -- it would pass through the shelf;
* one **linear** key in the middle of the smooth move makes a **velocity corner** (a jerk);
* the swing is "captured" data: dense keys with **jitter**.

The engine measures the problems exactly -- overshoot from the extrema of the curve's
closed form, the corner from the two segments' one-sided derivatives, the jitter from
the jerk -- then the repair is a short list of graph-editor operations: automatic clamped
handles (the Fritsch-Carlson limit guarantees no overshoot), Bezier interpolation on the
corner key, and for the jitter a Gaussian smoothing modifier (sigma 0.06 s) baked by
resampling, then reduced within 0.05 degrees. The repair is applied

1. from Python (``scene.animation.edit_keys`` / ``simplify``),
2. as an ``aether3d.animation-script`` through the CLI (``aether3d animation run``),
3. over MCP (``run_script``) by a real server process, as an agent would,

and the three repaired projects are compared: their curves must be identical.

Run from the repository root (MCP needs ``pip install "aether3d[mcp]"``):

    python examples/animation_curve_repair.py [OUTPUT_DIR] [--quick]

Writes to ``OUTPUT_DIR`` (default ``examples/output/animation/curve_repair``):
``crane_before.aether``, ``repair_script.json``, ``crane_after_{python,json,mcp}.aether``,
``graph_before.png``/``.svg``, ``graph_after.png``/``.svg``, ``graphs_side_by_side.png``,
``onion_before.png``/``onion_after.png`` (eight frames layered: the motion's path),
``report.json`` (the measurements before and after, and the comparison).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from animation_common import (  # noqa: E402
    backend,
    save_png,
    use_backend,
    write_json,
)

from aether3d import Scene  # noqa: E402
from aether3d.animation.frames import contact_sheet  # noqa: E402

SHELF_TOP = 1.2
CRATE_HALF = 0.25
PATHS = ("transform.position.z", "transform.position.x", "transform.rotation_euler.y")


def build() -> Scene:
    s = Scene("Crane")
    s.world.background = (0.83, 0.86, 0.9)
    s.add_plane("Floor", size=12).color = (0.55, 0.56, 0.58)
    shelf = s.add_cube(
        "Shelf", size=1.0, position=(1.5, 0, SHELF_TOP / 2), scale=(1.6, 1.2, SHELF_TOP)
    )
    shelf.color = (0.55, 0.36, 0.2)
    crate = s.add_cube("Crate", size=2 * CRATE_HALF, position=(-2.0, 0, 2.6))
    crate.color = (0.85, 0.62, 0.24)
    hook = s.add_cube("Cable", size=1.0, position=(-2.0, 0, 4.4), scale=(0.04, 0.04, 3.0))
    hook.color = (0.15, 0.15, 0.15)
    s.add_directional_light("Sun", direction=(-0.4, 0.6, -0.8), intensity=2.2)
    cam = s.add_camera("Cam", position=(0.2, -7.5, 2.6), look_at=(0.0, 0, 1.6), fov=42)
    s.active_camera = cam
    a = s.animation
    a.set_timeline(fps=24, start=0.0, end=4.0)
    # Height: carelessly keyed with automatic (overshooting) handles and a linear key.
    a.set_channel(crate, "transform.position.z", [
        {"time": 0.0, "value": 2.6}, {"time": 1.2, "value": 2.9},
        {"time": 2.0, "value": 2.2, "interpolation": "linear"},
        {"time": 2.6, "value": SHELF_TOP + CRATE_HALF},
        {"time": 4.0, "value": SHELF_TOP + CRATE_HALF},
    ], handles="auto")  # fmt: skip
    a.set_channel(crate, "transform.position.x", [(0.0, -2.0), (1.6, 0.4), (2.6, 1.5), (4.0, 1.5)],
                  handles="auto")  # fmt: skip
    # The swing: "captured" at 48 keys per second, with jitter.
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 4.0, 193)
    swing = 14 * np.sin(2 * np.pi * t / 2.6) * np.exp(-t / 1.8) + rng.normal(scale=0.6, size=t.size)
    captured = [(float(x), float(y)) for x, y in zip(t, swing, strict=True)]
    a.set_channel(crate, "transform.rotation_euler.y", captured,
                  handles="auto")  # fmt: skip
    a.set_channel(hook, "transform.position.x", [(0.0, -2.0), (1.6, 0.4), (2.6, 1.5), (4.0, 1.5)],
                  handles="auto")  # fmt: skip
    return s


REPAIR = {
    "format": "aether3d.animation-script",
    "version": 1,
    "label": "Repair the crane's curves",
    "operations": [
        {"op": "edit_keys", "target": "Crate", "path": "transform.position.z",
         "edit": "handle_type", "keys": "all", "side": "both", "type": "auto_clamped"},
        {"op": "edit_keys", "target": "Crate", "path": "transform.position.z",
         "edit": "interpolation", "keys": [2], "interpolation": "bezier"},
        {"op": "edit_keys", "target": "Crate", "path": "transform.position.x",
         "edit": "handle_type", "keys": "all", "side": "both", "type": "auto_clamped"},
        {"op": "set_modifiers", "target": "Crate", "path": "transform.rotation_euler.y",
         "modifiers": [{"type": "smooth", "sigma": 0.06}]},
        {"op": "resample", "target": "Crate", "path": "transform.rotation_euler.y", "fps": 48},
        {"op": "simplify", "target": "Crate", "path": "transform.rotation_euler.y",
         "tolerance": 0.05},
    ],
}  # fmt: skip


def repair_python(s: Scene) -> None:
    """The same six operations, from Python."""
    a = s.animation
    crate = s.get_by_name("Crate")
    with a.batch("Repair the crane's curves"):
        a.edit_keys(crate, "transform.position.z", "handle_type", keys="all", side="both",
                    type="auto_clamped")  # fmt: skip
        a.edit_keys(
            crate, "transform.position.z", "interpolation", keys=[2], interpolation="bezier"
        )
        a.edit_keys(crate, "transform.position.x", "handle_type", keys="all", side="both",
                    type="auto_clamped")  # fmt: skip
        a.set_modifiers(crate, "transform.rotation_euler.y", [{"type": "smooth", "sigma": 0.06}])
        a.resample(crate, "transform.rotation_euler.y", fps=48)  # (bakes the smoothing)
        a.simplify(crate, "transform.rotation_euler.y", 0.05)


def measure(s: Scene) -> dict:
    a = s.animation
    crate = s.get_by_name("Crate")
    out = {}
    for path in PATHS:
        rep = a.curve_report(crate, path)
        out[path] = {
            "keys": rep["keys"],
            "overshoot": rep["overshoot"],
            "unintended_corners": [d for d in rep["discontinuities"] if not d["intended"]],
            "jerk_max": rep["jerk"]["max"],
            "velocity_max": rep["velocity"]["max"],
        }
    z = a.channel(crate, "transform.position.z")
    minima = [e["value"] for e in a.curve_report(crate, "transform.position.z")["extrema"]
              if e["kind"] == "min"]  # fmt: skip
    lowest = min(minima) if minima else None
    t = np.linspace(2.0, 4.0, 2001)
    out["crate_bottom_min_after_2s_m"] = float(np.min(z.evaluate(t)) - CRATE_HALF)
    out["shelf_top_m"] = SHELF_TOP
    out["penetrates_shelf"] = bool(out["crate_bottom_min_after_2s_m"] < SHELF_TOP - 1e-9)
    out["lowest_extremum"] = lowest
    out["findings"] = [
        {k: f[k] for k in ("code", "location", "time", "magnitude")}
        for f in a.diagnose(overshoot_relative=0.01)["findings"]
    ]
    return out


def graphs(scene: Scene, crate, when: str):
    """Value, velocity and acceleration: the height and travel (m), and the swing (deg)."""
    a = scene.animation
    moves = a.graph([(crate, p) for p in PATHS[:2]], format="png", width=900,
                    title=f"Crate height and travel, {when} (m)")  # fmt: skip
    swing = a.graph([(crate, PATHS[2])], format="png", width=900,
                    title=f"Crate swing, {when} (degrees)")  # fmt: skip
    return moves, swing


def onion(s: Scene, path: Path, frames: int = 9) -> None:
    """The motion's path: frames rendered at evenly spaced times, combined by the darkest
    value of each pixel (so every pose shows)."""
    acc = None
    for k in range(frames):
        r = s.render(time=4.0 * k / (frames - 1), width=640, height=360, backend=backend())
        img = r.image.astype(np.float64)
        acc = img if acc is None else np.minimum(acc, img)  # (darkest: every pose shows)
    save_png(
        path,
        acc.astype(np.uint8),
        "Aether3D renders at nine times, combined (darkest value per pixel)",
    )


async def repair_mcp(workspace: Path, project: str) -> dict:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace)],
        env=dict(os.environ),
    )
    async with Client(server) as client:

        async def call(tool, args=None):
            r = await client.call_tool(tool, args or {})
            if r.is_error:
                raise RuntimeError(json.dumps(r.structured_content))
            return r.structured_content

        await call("scene_open", {"path": project})
        before = await call("inspect_animation", {"what": "curve", "target": "Crate",
                                                  "path": "transform.position.z"})  # fmt: skip
        ran = await call("run_script", {"script": REPAIR})
        after = await call("inspect_animation", {"what": "curve", "target": "Crate",
                                                 "path": "transform.position.z"})  # fmt: skip
        await call("scene_save", {"path": "crane_after_mcp.aether", "overwrite": True})
        return {"overshoot_before": len(before["overshoot"]),
                "overshoot_after": len(after["overshoot"]),
                "operations": len(ran["operations"])}  # fmt: skip


def curves_of(path: Path) -> dict:
    s = Scene.load(path)
    crate = s.get_by_name("Crate")
    return {p: s.animation.channel(crate, p).to_json() for p in PATHS}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output/animation/curve_repair"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args(argv)
    use_backend(args.backend)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"backend": backend()}

    scene = build()
    scene.save(out / "crane_before.aether")
    (out / "repair_script.json").write_text(json.dumps(REPAIR, indent=2) + "\n")
    report["before"] = measure(scene)
    crate = scene.get_by_name("Crate")
    before = graphs(scene, crate, "before repair")
    graph = "a graph drawn by Aether3D: value, velocity and acceleration"
    save_png(out / "graph_before.png", before[0], graph)
    save_png(out / "graph_swing_before.png", before[1], graph)
    (out / "graph_before.svg").write_text(scene.animation.graph([(crate, p) for p in PATHS[:2]]))
    onion(scene, out / "onion_before.png")

    # 1. Python
    repair_python(scene)
    scene.save(out / "crane_after_python.aether")
    report["after"] = measure(scene)
    after = graphs(scene, crate, "after repair")
    save_png(out / "graph_after.png", after[0], graph)
    save_png(out / "graph_swing_after.png", after[1], graph)
    (out / "graph_after.svg").write_text(scene.animation.graph([(crate, p) for p in PATHS[:2]]))
    labels = ["height, travel: before", "after", "swing: before", "after"]
    sheet = contact_sheet([before[0], after[0], before[1], after[1]], columns=2, labels=labels)
    save_png(out / "graphs_side_by_side.png", sheet, "graphs drawn by Aether3D, before and after")
    onion(scene, out / "onion_after.png")

    # 2. The JSON script, through the CLI (a separate process).
    json_project = out / "crane_after_json.aether"
    subprocess.run(
        [sys.executable, "-m", "aether3d", "animation", "run", str(out / "crane_before.aether"),
         str(out / "repair_script.json"), "--output", str(json_project), "--json"],
        check=True, capture_output=True, cwd=ROOT,
    )  # fmt: skip

    # 3. MCP (a server process; the workspace is the output directory).
    try:
        import anyio

        report["mcp"] = anyio.run(repair_mcp, out, "crane_before.aether")
    except ImportError:
        report["mcp"] = {"skipped": "the MCP SDK is not installed"}
    python_curves = curves_of(out / "crane_after_python.aether")
    json_curves = curves_of(json_project)
    report["same_python_json"] = python_curves == json_curves
    if (out / "crane_after_mcp.aether").exists():
        report["same_python_mcp"] = python_curves == curves_of(out / "crane_after_mcp.aether")
    write_json(out / "report.json", report)
    print(json.dumps({k: report[k] for k in report if k.startswith("same") or k == "mcp"}))
    print("overshoot before/after:",
          len(report["before"]["transform.position.z"]["overshoot"]),
          len(report["after"]["transform.position.z"]["overshoot"]))  # fmt: skip
    print("penetrates the shelf before/after:", report["before"]["penetrates_shelf"],
          report["after"]["penetrates_shelf"])  # fmt: skip
    ok = report["same_python_json"] and report.get("same_python_mcp", True)
    if not ok or report["after"]["penetrates_shelf"]:
        print("FAILED: the routes differ or the crate still goes into the shelf")
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
