"""An agent's visual feedback loop over MCP: build, look, check, correct, verify (Stage 12).

Needs the MCP SDK (``pip install "aether3d[mcp]"``). Run from the repository root:

    python examples/mcp_feedback_agent.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

This script plays an AI agent connected to ``aether3d mcp serve``. There is no language
model here: every step is the tool call an agent would make, so the run is reproducible.
**The one "visual evaluation" it sends is scripted by this file, not produced by a model
looking at an image** -- it says so in its provenance (``image_input: false``), and the
engine labels its finding an *agent* judgment, never a visual observation.

1. Orient: ``capabilities`` (topic ``feedback``) and the guide ``aether3d://feedback/guide``.
2. Build the Minecraft-style castle (2,190 blocks, one ``execute_scene_plan``) with a first
   draft's mistakes: the north-east tower stands 3 m above the ground, a tree grows inside
   the keep's wall.
3. Look: ``observe_scene`` returns the render and an annotated copy (numbered markers on
   the towers), with each tower's measured pixels and geometry.
4. Check: ``diagnose_scene`` with expectations -- towers stand on the ground, trees keep
   clear of the keep, the keep is visible, the exposure is sane -- and the scripted
   evaluation, grounded by the engine in the render's object index.
5. Correct: ``propose_corrections`` (with a preview that changes nothing), then
   ``apply_corrections`` of the deterministic proposals with ``expected_revision``: one
   undo step, verified on a new render. The scripted evaluation's suggestion stays an
   approval-only proposal, and the agent leaves it.
6. ``feedback_loop`` for what remains, then a last look and ``scene_save``.

Writes ``mcp_feedback_agent.png`` (before and after, towers and trees marked) and
``mcp_feedback_agent.json`` (every call in brief with its time and result) to OUTPUT_DIR;
the project stays in the server's workspace (``OUTPUT_DIR/mcp_feedback_workspace``).
Exits with status 1 if a check fails.
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
ROOT = HERE.parent

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "Castle",
    "views": {"main": {"camera": "camera"}},
    "checks": [
        # The ground plane lies 2 cm below z = 0 (the castle plan's choice): a tolerance.
        {
            "id": "towers-stand",
            "check": "rests_on",
            "objects": {"tags": ["tower"]},
            "support": "ground",
            "tolerance": 0.05,
        },
        {
            "id": "trees-clear",
            "check": "no_overlap",
            "objects": {"tags": ["tree"]},
            "against": "keep",
            "fix": {"move": "objects"},
        },
        {"id": "keep-seen", "check": "visible", "objects": "keep", "min_visible_fraction": 0.5},
        {"id": "light", "check": "exposure", "mean_luminance": {"min": 0.05, "max": 0.8}},
    ],
}


def first_draft() -> dict:
    plan = json.loads((ROOT / "src/aether3d/plans/examples/castle.json").read_text())
    plan["label"] = "Castle (first draft)"
    plan["objects"]["trees"]["instances"][1]["position"] = [-3, -0.6, 0]  # in the keep's wall
    plan["objects"]["towers"]["instances"][2]["position"] = [15, 15, 3]  # 3 m up
    return plan


class Agent:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.transcript: list[dict] = []

    async def call(self, name: str, arguments: dict | None = None) -> tuple[dict, list[bytes]]:
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
        if result.is_error:
            raise RuntimeError(f"{name}: {data['error']}")
        return data, images


def brief(name: str, data: dict) -> dict:
    if "error" in data:
        return {"error": data["error"].get("code")}
    if name == "diagnose_scene":
        return {
            "accepted": data["accepted"],
            "summary": data["summary"],
            "findings": [(f["code"], f["basis"], f.get("magnitude")) for f in data["findings"]],
        }
    if name == "propose_corrections":
        return {
            "proposals": [(p["id"], p["safety"], p["title"]) for p in data["proposals"]],
            "unresolved": [(u["code"], u["reason"]) for u in data["unresolved"]],
        }
    if name == "apply_corrections":
        v = data.get("verification") or {}
        return {
            "status": data["status"],
            "applied": data["applied"],
            "resolved": v.get("resolved"),
            "decision": v.get("decision"),
        }
    if name == "feedback_loop":
        return {"stop_reason": data["stop_reason"], "iterations": len(data["iterations"])}
    if name == "observe_scene":
        return {
            "observations": [o["id"] for o in data["observations"]],
            "renderer": data["observations"][0]["renderer"],
        }
    return {k: data[k] for k in list(data)[:4]}


def decode(png: bytes) -> np.ndarray:
    """RGB pixels of a PNG (Aether3D's own decoder)."""
    from aether3d.materials.images import decode_image

    return np.asarray(decode_image(png).pixels)[..., :3]


async def run(workspace: Path, backend: str, size: tuple[int, int]) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "aether3d",
            "mcp",
            "serve",
            "--workspace",
            str(workspace),
            "--feedback-backend",
            backend,
        ],
        env=dict(os.environ),
    )
    out: dict = {}
    async with Client(params) as client:
        agent = Agent(client)
        caps, _ = await agent.call("capabilities", {"topic": "feedback"})
        guide = await client.read_resource("aether3d://feedback/guide")
        out["orient"] = {
            "checks": len(caps["checks"]),
            "actions": len(caps["actions"]),
            "guide_chars": len(guide.contents[0].text),
        }
        await agent.call("scene_new", {"path": "castle.aether"})
        await agent.call("execute_scene_plan", {"plan": first_draft()})
        width, height = size
        look = {
            "images": ["image", "annotated"],
            "width": width,
            "height": height,
            "focus": {"tags": ["tower"]},
        }
        seen, images = await agent.call("observe_scene", look)
        out["before_png"] = images[1]
        obs = seen["observations"][0]
        tower = next(o for o in obs["objects"] if o["name"] == "Tower NE")
        scripted = {  # SCRIPTED by this example: no model looked at the image
            "format": "aether3d.vision-evaluation",
            "version": 1,
            "observation": obs["id"],
            "image_sha256": obs["image"]["sha256"],
            "provenance": {
                "provider": "scripted-example",
                "image_input": False,
                "method": "written by examples/mcp_feedback_agent.py, not a model",
            },
            "findings": [
                {
                    "category": "floating",
                    "severity": "warning",
                    "confidence": 0.9,
                    "summary": "the north-east tower looks as if it hovers",
                    "region": tower["screen_bounds"],
                    "objects": [tower["id"]],
                    "suggestion": {
                        "action": "translate",
                        "object": tower["id"],
                        "offset": [0, 0, -3],
                    },
                }
            ],
        }
        diagnosis, _ = await agent.call(
            "diagnose_scene", {"expectations": EXPECTATIONS, "evaluations": [scripted]}
        )
        out["diagnosis"] = diagnosis
        proposals, preview = await agent.call(
            "propose_corrections", {"evaluations": [scripted], "preview": True}
        )
        out["proposals"] = proposals
        # The agent's decision: apply the deterministic proposals (safe, and verify ones,
        # which it approves by naming them); leave the scripted suggestion unapproved.
        chosen = [
            p["id"]
            for p in proposals["proposals"]
            if p["basis"] not in ("vision", "agent") and p["safety"] != "approval"
        ]
        applied, _ = await agent.call(
            "apply_corrections",
            {"proposals": chosen, "expected_revision": diagnosis["revision"]},
        )
        out["applied"] = applied
        loop, _ = await agent.call(
            "feedback_loop", {"budget": {"max_iterations": 4}, "include_image": False}
        )
        out["loop"] = loop
        final, _ = await agent.call("diagnose_scene", {})
        out["final"] = final
        seen, images = await agent.call("observe_scene", {**look, "images": ["annotated"]})
        out["after_png"] = images[0]
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
    workspace = out / "mcp_feedback_workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    size = (480, 270) if args.quick else (960, 540)
    t0 = time.perf_counter()
    result = anyio.run(run, workspace, args.backend, size)
    seconds = time.perf_counter() - t0

    failures = []
    codes = {(f["code"], f["basis"]) for f in result["diagnosis"]["findings"]}
    for expected in (
        ("FLOATING", "geometry"),
        ("INTERSECTION", "geometry"),
        ("VISION_FLOATING", "agent"),
    ):
        if expected not in codes:
            failures.append(f"the first diagnosis should report {expected}")
    by_safety = {p["safety"] for p in result["proposals"]["proposals"]}
    if "approval" not in by_safety:
        failures.append("the scripted suggestion should be an approval-only proposal")
    if result["applied"]["status"] != "applied":
        failures.append(f"apply_corrections: {result['applied']['status']}")
    if not result["final"]["accepted"]:
        failures.append("the castle does not meet its expectations at the end")
    if result["proposals"]["preview"]["comparison"]["resolved"] == []:
        failures.append("the preview should resolve findings")

    from aether3d.rendering.png import encode_png

    left, right = decode(result.pop("before_png")), decode(result.pop("after_png"))

    h = max(left.shape[0], right.shape[0])
    image = np.full((h, left.shape[1] + 8 + right.shape[1], 3), 255, dtype=np.uint8)
    image[: left.shape[0], : left.shape[1]] = left
    image[: right.shape[0], left.shape[1] + 8 :] = right
    (out / "mcp_feedback_agent.png").write_bytes(
        encode_png(image, text={"Software": f"Aether3D {__version__}"})
    )
    report = {
        "aether3d": __version__,
        "seconds": seconds,
        "orient": result["orient"],
        "first_diagnosis": brief("diagnose_scene", result["diagnosis"]),
        "proposals": brief("propose_corrections", result["proposals"]),
        "preview_resolved": result["proposals"]["preview"]["comparison"]["resolved"],
        "applied": brief("apply_corrections", result["applied"]),
        "loop": brief("feedback_loop", result["loop"]),
        "final": brief("diagnose_scene", result["final"]),
        "transcript": result["transcript"],
        "failures": failures,
    }
    (out / "mcp_feedback_agent.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for step in result["transcript"]:
        what = json.dumps(step["brief"])[:110]
        print(f"  {step['tool']:<20} {step['seconds'] * 1000:8.1f} ms  {what}")
    print(
        f"{len(result['transcript'])} tool calls in {seconds:.2f} s; final: "
        f"{'accepted' if result['final']['accepted'] else 'NOT accepted'}"
    )
    for failure in failures:
        print(f"CHECK FAILED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
