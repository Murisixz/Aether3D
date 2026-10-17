"""An agent models a part over MCP, hits a constraint conflict and recovers (Stage 15).

Needs the MCP SDK (``pip install "aether3d[mcp]"``). Run from the repository root:

    python examples/mcp_cad_agent.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

This script plays an AI agent connected to ``aether3d mcp serve``. There is no language
model here: every step is the tool call an agent would make, with the arguments it would
send, so the run is reproducible. Nothing is judged visually by this file: the checks
are the engine's exact measurements. (Whether an agent can *see* the images the tools
return depends on its client and model; the server only delivers them.)

1. Orient: ``capabilities`` (topic ``cad``), the guide and the CAD script schema.
2. Model: one CAD script -- parameters, a fully constrained profile, a plate with two
   holes -- ending with ``expect`` checks (thickness, validity, constraint status).
3. Mistake: the agent adds a diagonal dimension of 120 mm to a 100 x 40 rectangle. The
   script is refused (``SKETCH_CONFLICT``) with the conflicting constraints named and the
   residual; nothing changes (same revision).
4. Recovery: the agent makes the diagonal a reference dimension (``driving: false``) --
   the engine measures it instead -- and checks it equals sqrt(100^2 + 40^2) mm.
5. Inspect: ``inspect_cad`` with a drawing image; ``diagnose_scene`` with the CAD
   feedback checks (``cad_valid``, ``cad_dimension``) -- accepted.
6. Exchange: ``export_scene`` to STEP and SVG, ``import_cad`` of the STEP back as a
   second part with the same volume; ``history`` (one step per script); ``scene_save``.

Writes ``mcp_cad_agent.png`` (the drawing ``inspect_cad`` returned, the render
``observe_scene`` returned) and ``mcp_cad_agent.json`` (every call in brief, with its
time). Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cad_common import caption, fit_width, row, save_json, save_png  # noqa: E402

from aether3d import __version__  # noqa: E402

PLATE = {
    "format": "aether3d.cad-script",
    "version": 1,
    "label": "Mounting plate",
    "operations": [
        {"op": "units", "length": "mm"},
        {"op": "parameter", "name": "length", "value": "100 mm"},
        {"op": "parameter", "name": "thickness", "value": "6 mm"},
        {"op": "add", "kind": "sketch", "name": "Outline", "definition": {
            "plane": "XY",
            "entities": [
                {"id": "a", "type": "point", "at": [0, 0]},
                {"id": "b", "type": "point", "at": [90, 0]},
                {"id": "c", "type": "point", "at": [90, 45]},
                {"id": "d", "type": "point", "at": [0, 45]},
                {"id": "bottom", "type": "line", "start": "a", "end": "b"},
                {"id": "right", "type": "line", "start": "b", "end": "c"},
                {"id": "top", "type": "line", "start": "c", "end": "d"},
                {"id": "left", "type": "line", "start": "d", "end": "a"}],
            "constraints": [
                {"id": "origin", "type": "fix", "refs": ["a"]},
                {"id": "h1", "type": "horizontal", "refs": ["bottom"]},
                {"id": "h2", "type": "horizontal", "refs": ["top"]},
                {"id": "v1", "type": "vertical", "refs": ["right"]},
                {"id": "v2", "type": "vertical", "refs": ["left"]},
                {"id": "len", "type": "length", "refs": ["bottom"], "value": "=length"},
                {"id": "wid", "type": "length", "refs": ["right"], "value": 40}]}},
        {"op": "add", "kind": "part", "name": "Plate", "definition": {"features": [
            {"type": "extrude", "id": "base", "sketch": "Outline",
             "extent": {"type": "blind", "distance": "=thickness"}},
            {"type": "hole", "id": "holes", "face": "base:end", "kind": "countersink",
             "points": [[30, 20, 6], [70, 20, 6]], "diameter": 6.6, "csink_diameter": 12},
            {"type": "chamfer", "id": "edge", "edges": [{"edge": ["base:end", "base:side/right"]}],
             "distance": 1.5}]}},
        {"op": "show", "item": "Plate"},
        {"op": "expect", "info": "Outline", "checks": {"status": "fully_constrained"}},
        {"op": "expect", "info": "Plate", "checks": {"bodies.main.valid": True}},
        {"op": "expect", "measure": {"a": {"part": "Plate", "face": "base:end"},
                                     "b": {"part": "Plate", "face": "base:start"}},
         "checks": {"parallel_distance": {"value": 6, "tolerance": 1e-9}}},
    ],
}  # fmt: skip

STAGE = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "label": "Stage",
    "objects": {
        "table": {"mesh": {"primitive": "plane", "size": 0.6}, "color": "#6f7378",
                  "position": [0.05, 0.02, -0.0005]},
        "key": {"light": {"type": "directional", "intensity": 2.6},
                "direction": [0.45, 0.7, -0.6]},
        "fill": {"light": {"type": "directional", "intensity": 0.6},
                 "direction": [-0.6, -0.3, -0.5]},
        "camera": {"camera": {"fov": 30}, "position": [0.16, -0.12, 0.13],
                   "look_at": [0.05, 0.02, 0.0]},
    },
}  # fmt: skip

DIAGONAL = [
    {"op": "append", "item": "Outline", "list": "entities",
     "values": [{"id": "diag", "type": "line", "start": "a", "end": "c", "construction": True}]},
    {"op": "append", "item": "Outline", "list": "constraints",
     "values": [{"id": "diag_len", "type": "length", "refs": ["diag"], "value": 120}]},
]  # fmt: skip

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "The plate as specified",
    "defaults": False,
    "checks": [
        {"check": "cad_valid", "fully_constrained": True},
        {"check": "cad_dimension", "reference": {"part": "Plate", "face": "holes.0:rev/wall.0"},
         "other": {"part": "Plate", "face": "holes.1:rev/wall.0"},
         "quantity": "axis_distance", "value": 40, "tolerance": 1e-9},
    ],
}  # fmt: skip


def script(ops, label):
    return {"format": "aether3d.cad-script", "version": 1, "label": label, "operations": ops}


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
        self.transcript.append({"tool": name, "seconds": round(seconds, 4),
                                "error": result.is_error, "images": len(images),
                                "brief": brief(name, data)})  # fmt: skip
        if result.is_error != error:
            raise RuntimeError(f"{name}: {data}")
        return data, images


def brief(name: str, data: dict) -> dict:
    if "error" in data:
        e = data["error"]
        return {k: e.get(k) for k in ("code", "location", "message")}
    if name == "run_script":
        return {"operations": [op["op"] for op in data["operations"]]}
    if name == "inspect_cad":
        if "items" in data:
            return {"items": [i["name"] for i in data["items"]]}
        return {k: data.get(k) for k in ("kind", "name", "status", "dof", "reference")}
    if name == "diagnose_scene":
        return {"accepted": data["accepted"],
                "findings": [(f["code"], f["basis"]) for f in data["findings"]]}  # fmt: skip
    if name == "history":
        return {"steps": [s["label"] for s in data["steps"]]}
    if name == "observe_scene":
        return {"renderer": data["observations"][0]["renderer"]["backend"]}
    return {k: data[k] for k in list(data)[:3]}


def decode(png: bytes) -> np.ndarray:
    from aether3d.materials.images import decode_image

    return np.asarray(decode_image(png).pixels)[..., :3]


async def run(workspace: Path, backend: str, size: int) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace),
              "--feedback-backend", backend],
        env=dict(os.environ),
    )  # fmt: skip
    out: dict = {}
    async with Client(params) as client:
        agent = Agent(client)
        caps, _ = await agent.call("capabilities", {"topic": "cad"})
        guide = await client.read_resource("aether3d://cad/guide")
        schema = await client.read_resource("aether3d://schema/cad-script")
        out["orient"] = {"operations": caps["operations"], "features": caps["features"],
                         "guide_chars": len(guide.contents[0].text),
                         "schema_bytes": len(schema.contents[0].text)}  # fmt: skip
        await agent.call("scene_new", {"path": "plate.aether"})
        await agent.call("run_script", {"script": PLATE})
        await agent.call("execute_scene_plan", {"plan": STAGE})

        before, _ = await agent.call("scene_info")
        refused, _ = await agent.call("run_script", {"script": script(DIAGONAL, "Diagonal")},
                                      error=True)  # fmt: skip
        out["refused"] = refused["error"]
        after, _ = await agent.call("scene_info")
        out["unchanged"] = after["revision"] == before["revision"]

        # The agent reads the conflict and turns its diagonal into a reference dimension.
        fixed = json.loads(json.dumps(DIAGONAL))
        fixed[1]["values"][0].update(driving=False, value=0)
        fixed.append({"op": "expect", "info": "Outline",
                      "checks": {"status": "fully_constrained"}})  # fmt: skip
        recovery = script(fixed, "Reference diagonal")
        out["recovered"], _ = await agent.call("run_script", {"script": recovery})
        out["outline"], _ = await agent.call("inspect_cad", {"item": "Outline"})
        args = {"item": "Plate", "image": "drawing", "view": "top"}
        out["plate"], images = await agent.call("inspect_cad", args)
        out["drawing_png"] = images[0]
        out["diagnosis"], _ = await agent.call("diagnose_scene", {"expectations": EXPECTATIONS})
        _, images = await agent.call("observe_scene", {"images": ["image"], "width": size,
                                                       "height": size * 3 // 4})  # fmt: skip
        out["render_png"] = images[0]
        out["step"], _ = await agent.call("export_scene", {"path": "plate.step",
                                                           "items": ["Plate"]})  # fmt: skip
        await agent.call("run_script", {"script": script(
            [{"op": "add", "kind": "drawing", "name": "Plate top",
              "definition": {"view": "top", "items": ["Plate"]}}], "Drawing")})  # fmt: skip
        out["svg"], _ = await agent.call("export_scene", {"path": "plate_top.svg",
                                                          "item": "Plate top"})  # fmt: skip
        out["import"], _ = await agent.call("import_cad", {"path": "plate.step",
                                                           "name": "Plate check"})  # fmt: skip
        out["check"], _ = await agent.call("inspect_cad", {"item": "Plate check"})
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
    workspace = out / "mcp_cad_workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    size = 480 if args.quick else 960
    t0 = time.perf_counter()
    result = anyio.run(run, workspace, args.backend, size)
    seconds = time.perf_counter() - t0

    failures = []
    refused = result["refused"]
    if refused["code"] != "SKETCH_CONFLICT" or not result["unchanged"]:
        failures.append(f"the conflicting dimension should be refused, changing nothing: "
                        f"{refused}")  # fmt: skip
    named = refused.get("details", {}).get("conflicts", [{}])[0]
    if named.get("constraint") != "diag_len" or not named.get("conflicts_with"):
        failures.append(f"the conflict should name diag_len and its partners: {named}")
    # The reference value is the document's display text (3 decimals in mm).
    measured = result["outline"]["reference"]["diag_len"]
    if abs(float(measured.split()[0]) - math.hypot(100, 40)) > 5e-4:
        failures.append(f"the diagonal measures {measured}, expected {math.hypot(100, 40):.4f}")
    if result["outline"]["status"] != "fully_constrained":
        failures.append(f"the outline is {result['outline']['status']}")
    if not result["diagnosis"]["accepted"]:
        failures.append(f"not accepted: {brief('diagnose_scene', result['diagnosis'])}")
    v1 = result["plate"]["bodies"]["main"]["volume"]
    v2 = result["check"]["bodies"]["main"]["volume"]
    if abs(v1 - v2) > 1e-9 * v1:
        failures.append(f"the STEP round trip changed the volume: {v1} -> {v2}")
    steps = [s["label"] for s in result["history"]["steps"]]
    for label in ("Mounting plate", "Reference diagonal"):
        if steps.count(label) != 1:
            failures.append(f"{label!r} should be one undo step: {steps}")
    if "Diagonal" in steps:
        failures.append("the refused script left a step")

    drawing = decode(result.pop("drawing_png"))
    render = decode(result.pop("render_png"))
    drawing = fit_width(drawing, render.shape[1] * 4 // 3)
    picture = row([caption(drawing, "inspect_cad image (top view drawing)"),
                   caption(render, "observe_scene image")])  # fmt: skip
    save_png(out / "mcp_cad_agent.png", picture)
    report = {
        "aether3d": __version__,
        "seconds": seconds,
        "orient": result["orient"],
        "refused": refused,
        "diagonal_reference": measured,
        "diagnosis": brief("diagnose_scene", result["diagnosis"]),
        "volumes_mm3": {"plate": v1, "reimported": v2},
        "history": steps,
        "transcript": result["transcript"],
        "failures": failures,
    }
    save_json(out / "mcp_cad_agent.json", report)
    for entry in result["transcript"]:
        print(f"  {entry['tool']:<16} {entry['seconds']:7.3f} s  {entry['brief']}")
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
