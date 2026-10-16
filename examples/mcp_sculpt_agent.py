"""An agent sculpts over MCP and checks the result with the visual feedback tools (Stage 13).

Needs the MCP SDK (``pip install "aether3d[mcp]"``). Run from the repository root:

    python examples/mcp_sculpt_agent.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

This script plays an AI agent connected to ``aether3d mcp serve``. There is no language
model here: every step is the tool call an agent would make, with the arguments it would
send, so the run is reproducible. Nothing is judged visually by this file: the checks
are the engine's measurements (geometry and renders), never a scripted "vision" verdict.

1. Orient: ``capabilities`` (topic ``sculpt``), the guide ``aether3d://sculpt/guide`` and
   the schema ``aether3d://schema/sculpt-script``.
2. Build a bust on a pedestal with ``execute_scene_plan``; ``inspect_mesh`` the ball
   that will become the head (tier ``solid``, with the exact self-intersection test).
3. A first sculpt script with a mistake an agent can make -- a voxel remesh of the open
   ground plane -- is refused with a structured error (``VOXEL_NOT_WATERTIGHT``, its
   location in the script, a hint), and nothing changes (the scene revision is the same).
4. The corrected script: a voxel remesh then an isotropic remesh for an even base,
   symmetric strokes (brow, eye sockets, nose, cheeks, mouth), the chin pulled down with
   ``grab``, ``require: "solid"`` so a broken result would be refused, and limits. One
   undo step.
5. ``observe_scene`` looks at the bust; ``diagnose_scene`` with expectations: the head's
   mesh is ``solid`` (``mesh_validity``), it rests on the pedestal, it is visible. The
   pulled chin now sinks into the pedestal: the diagnosis says so, with the depth.
6. ``propose_corrections`` and ``apply_corrections``: the bust is raised by the measured
   depth, verified on a new render, as one undo step. A last diagnosis is accepted.
7. ``history`` lists the steps; ``scene_save``.

Writes ``mcp_sculpt_agent.png`` (the ball, and the sculpted bust after the correction)
and ``mcp_sculpt_agent.json`` (every call in brief with its time) to OUTPUT_DIR; the
project stays in ``OUTPUT_DIR/mcp_sculpt_workspace``. Exits with status 1 if a check
fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    import anyio
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
except ImportError:  # pragma: no cover - the message is the point
    sys.exit('this example needs the MCP SDK: pip install "aether3d[mcp]"')

import numpy as np

from aether3d import __version__

HERE = Path(__file__).resolve().parent

PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "label": "Bust on a pedestal",
    "materials": {
        "clay": {"base_color": "#c49a74", "roughness": 0.75},
        "stone": {"base_color": "#8d8f93", "roughness": 0.85},
    },
    "objects": {
        "ground": {"mesh": {"primitive": "plane", "size": 12}, "color": "#5d6168"},
        "pedestal": {
            "mesh": {"primitive": "cylinder", "radius": 0.45, "depth": 1.0, "segments": 48},
            "position": [0, 0, 0.5],
            "material": "@stone",
        },
        "head": {
            "mesh": {"primitive": "sphere", "radius": 0.5, "segments": 48, "rings": 24},
            "position": [0, 0, 1.5],
            "material": "@clay",
        },
        "key": {"light": {"type": "directional", "intensity": 2.4}, "direction": [0.5, 0.8, -0.7]},
        "rim": {"light": {"type": "directional", "intensity": 0.8}, "direction": [-0.6, -0.4, -0.4]},  # noqa: E501
        "camera": {"camera": {"fov": 35}, "position": [1.6, -3.6, 2.1], "look_at": [0, 0, 1.35]},
    },
}  # fmt: skip

SYM = {"symmetry": "x"}


def taper(points):
    pressure = np.interp(np.linspace(0, 1, len(points)), [0, 0.5, 1], [0.3, 1.0, 0.3])
    return [{"position": p, "pressure": float(q)} for p, q in zip(points, pressure, strict=True)]


# World coordinates: the head's centre is (0, 0, 1.5); it faces -Y.
STROKES = [
    {"op": "voxel_remesh", "voxel_size": 0.025},
    {"op": "remesh", "target_length": 0.03, "iterations": 3},  # even triangles to sculpt
    {
        "op": "stroke",
        "brush": {"kind": "draw", "radius": 0.11, "strength": 0.12, **SYM},
        "path": taper(
            [[0.04, -0.47, 1.68], [0.14, -0.45, 1.7], [0.24, -0.41, 1.68], [0.32, -0.36, 1.64]]
        ),
    },
    {
        "op": "stroke",
        "brush": {"kind": "draw", "radius": 0.09, "strength": 0.3, "direction": -1, **SYM},
        "path": [[0.15, -0.46, 1.58]],
    },
    {
        "op": "stroke",
        "brush": {"kind": "draw", "radius": 0.07, "strength": 0.12, **SYM},
        "path": taper([[0.0, -0.5, 1.58], [0.0, -0.53, 1.48], [0.0, -0.55, 1.42]]),
    },
    {
        "op": "stroke",
        "brush": {"kind": "clay", "radius": 0.1, "strength": 0.25, **SYM},
        "path": taper([[0.16, -0.46, 1.44], [0.26, -0.4, 1.46], [0.34, -0.33, 1.48]]),
    },
    {
        "op": "stroke",
        "brush": {"kind": "crease", "radius": 0.05, "strength": 0.3, **SYM},
        "path": taper([[0.0, -0.5, 1.34], [0.08, -0.48, 1.34], [0.15, -0.45, 1.36]]),
    },
    {
        "op": "stroke",
        "brush": {"kind": "grab", "radius": 0.22, "strength": 1.0, **SYM},
        "path": [[0.0, -0.4, 1.12], [0.0, -0.42, 0.95]],
    },
    {"op": "check", "self_intersections": True},
]

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "Bust",
    "views": {"main": {"camera": "camera"}},
    "checks": [
        {"id": "sound", "check": "mesh_validity", "objects": "head", "tier": "solid"},
        {"id": "on-pedestal", "check": "rests_on", "objects": "head", "support": "pedestal",
         "fix": {"move": "objects"}},
        {"id": "seen", "check": "visible", "objects": "head", "min_visible_fraction": 0.8},
    ],
}  # fmt: skip


class Agent:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.transcript: list[dict] = []

    async def call(self, name: str, arguments: dict | None = None, *, error: bool = False):
        t0 = time.perf_counter()
        result = await self.client.call_tool(name, arguments or {})
        seconds = time.perf_counter() - t0
        data = result.structured_content
        images = [base64.b64decode(c.data) for c in result.content if c.type == "image"]
        self.transcript.append(
            {
                "tool": name,
                "seconds": round(seconds, 4),
                "error": result.is_error,
                "images": len(images),
                "brief": brief(name, data),
            }
        )
        if result.is_error != error:
            raise RuntimeError(f"{name}: {data}")
        return data, images


def brief(name: str, data: dict) -> dict:
    if "error" in data:
        e = data["error"]
        return {k: e.get(k) for k in ("code", "location", "hint")}
    if name == "inspect_mesh":
        s = data["summary"]
        return {k: s.get(k) for k in ("tier", "triangle_count", "components", "volume")}
    if name == "run_script":
        commit = data["sculpts"][0]["commit"]
        return {
            "changed": commit["changed"],
            "triangles": commit["mesh"]["triangle_count"],
            "tier": commit["report"]["summary"]["tier"],
            "operations": [op["op"] for op in data["sculpts"][0]["operations"]],
        }
    if name == "diagnose_scene":
        return {
            "accepted": data["accepted"],
            "findings": [(f["code"], f["basis"], f.get("magnitude")) for f in data["findings"]],
        }
    if name == "propose_corrections":
        return {"proposals": [(p["id"], p["safety"], p["title"]) for p in data["proposals"]]}
    if name == "apply_corrections":
        v = data.get("verification") or {}
        return {"status": data["status"], "decision": v.get("decision")}
    if name == "observe_scene":
        return {"renderer": data["observations"][0]["renderer"]}
    if name == "history":
        return {"steps": [s["label"] for s in data["steps"]]}
    return {k: data[k] for k in list(data)[:3]}


def decode(png: bytes) -> np.ndarray:
    from aether3d.materials.images import decode_image

    return np.asarray(decode_image(png).pixels)[..., :3]


async def run(workspace: Path, backend: str, size: tuple[int, int]) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace),
              "--feedback-backend", backend],
        env=dict(os.environ),
    )  # fmt: skip
    out: dict = {}
    async with Client(params) as client:
        agent = Agent(client)
        caps, _ = await agent.call("capabilities", {"topic": "sculpt"})
        guide = await client.read_resource("aether3d://sculpt/guide")
        schema = await client.read_resource("aether3d://schema/sculpt-script")
        out["orient"] = {
            "brushes": len(caps["brushes"]),
            "operations": caps["operations"],
            "guide_chars": len(guide.contents[0].text),
            "schema_bytes": len(schema.contents[0].text),
        }
        await agent.call("scene_new", {"path": "bust.aether"})
        await agent.call("execute_scene_plan", {"plan": PLAN})
        out["ball"], _ = await agent.call(
            "inspect_mesh", {"object": "head", "self_intersections": True}
        )
        width, height = size
        look = {"images": ["image"], "width": width, "height": height}
        _, images = await agent.call("observe_scene", look)
        out["ball_png"] = images[0]

        info, _ = await agent.call("scene_info")
        revision = info["revision"]
        wrong = {
            "format": "aether3d.sculpt-script",
            "version": 1,
            "sculpts": [{"object": "ground", "operations": [STROKES[0]]}],
        }
        refused, _ = await agent.call("run_script", {"script": wrong}, error=True)
        out["refused"] = refused["error"]
        info, _ = await agent.call("scene_info")
        out["unchanged"] = info["revision"] == revision

        script = {
            "format": "aether3d.sculpt-script",
            "version": 1,
            "label": "Sculpt the bust",
            "limits": {"max_dabs": 5000, "max_triangles": 400000, "max_seconds": 120},
            "sculpts": [{"object": "head", "require": "solid", "operations": STROKES}],
        }
        out["sculpt"], _ = await agent.call("run_script", {"script": script})
        out["head"], _ = await agent.call(
            "inspect_mesh", {"object": "head", "self_intersections": True}
        )
        await agent.call(
            "observe_scene", {**look, "view": {"focus": {"objects": "head"}}, "focus": "head"}
        )
        out["diagnosis"], _ = await agent.call("diagnose_scene", {"expectations": EXPECTATIONS})
        proposals, _ = await agent.call("propose_corrections", {"preview": True})
        out["proposals"] = proposals
        chosen = [p["id"] for p in proposals["proposals"] if p["safety"] != "approval"]
        out["applied"], _ = await agent.call(
            "apply_corrections",
            {"proposals": chosen, "expected_revision": out["diagnosis"]["revision"]},
        )
        out["final"], _ = await agent.call("diagnose_scene", {})
        _, images = await agent.call("observe_scene", look)
        out["bust_png"] = images[0]
        out["history"], _ = await agent.call("history", {"action": "list"})
        await agent.call("scene_save", {})
        out["transcript"] = agent.transcript
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="smaller images (CI)")
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    args = parser.parse_args(argv)
    out = Path(args.output)
    workspace = out / "mcp_sculpt_workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    size = (360, 360) if args.quick else (720, 720)
    t0 = time.perf_counter()
    result = anyio.run(run, workspace, args.backend, size)
    seconds = time.perf_counter() - t0

    failures = []
    if result["ball"]["summary"]["tier"] != "solid":
        failures.append("the ball should be solid")
    if result["refused"]["code"] != "VOXEL_NOT_WATERTIGHT" or not result["unchanged"]:
        failures.append(
            f"the wrong script should be refused, changing nothing: {result['refused']}"
        )
    if result["head"]["summary"]["tier"] != "solid":
        failures.append(f"the sculpted head is {result['head']['summary']['tier']}")
    codes = {(f["code"], f["basis"]) for f in result["diagnosis"]["findings"]}
    if not any(code == "SUNKEN" or code == "PENETRATING" for code, _ in codes):
        failures.append(f"the pulled chin should sink into the pedestal: {codes}")
    if any(f["check"] == "sound" for f in result["diagnosis"]["findings"]):
        failures.append("the sculpted head should be a sound (solid) mesh")
    if result["applied"]["status"] != "applied":
        failures.append(f"apply_corrections: {result['applied']['status']}")
    if not result["final"]["accepted"]:
        failures.append(f"not accepted at the end: {brief('diagnose_scene', result['final'])}")
    steps = [s["label"] for s in result["history"]["steps"]]
    if "Sculpt the bust" not in steps:
        failures.append(f"the sculpt should be one undo step: {steps}")

    from aether3d.rendering.png import encode_png

    left, right = decode(result.pop("ball_png")), decode(result.pop("bust_png"))
    image = np.full((left.shape[0], left.shape[1] + 8 + right.shape[1], 3), 255, dtype=np.uint8)
    image[:, : left.shape[1]] = left
    image[: right.shape[0], left.shape[1] + 8 :] = right
    (out / "mcp_sculpt_agent.png").write_bytes(
        encode_png(image, text={"Software": f"Aether3D {__version__}"})
    )
    report = {
        "aether3d": __version__,
        "seconds": seconds,
        "orient": result["orient"],
        "refused": result["refused"],
        "sculpt": brief("run_script", result["sculpt"]),
        "head": brief("inspect_mesh", result["head"]),
        "diagnosis": brief("diagnose_scene", result["diagnosis"]),
        "proposals": brief("propose_corrections", result["proposals"]),
        "applied": brief("apply_corrections", result["applied"]),
        "final": brief("diagnose_scene", result["final"]),
        "history": steps,
        "transcript": result["transcript"],
        "failures": failures,
    }
    (out / "mcp_sculpt_agent.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for step in result["transcript"]:
        what = json.dumps(step["brief"])[:110]
        print(f"  {step['tool']:<20} {step['seconds'] * 1000:8.1f} ms  {what}")
    print(f"{len(result['transcript'])} tool calls in {seconds:.2f} s")
    for failure in failures:
        print(f"CHECK FAILED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
