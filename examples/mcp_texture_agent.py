"""An agent unwraps, checks, corrects and paints over MCP (Stage 14).

Needs the MCP SDK (``pip install "aether3d[mcp]"``). Run from the repository root:

    python examples/mcp_texture_agent.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

This script plays an AI agent connected to ``aether3d mcp serve``. There is no language
model here: every step is the tool call an agent would make, with the arguments it would
send, so the run is reproducible. Nothing is judged visually by this file: the checks are
the engine's measurements, never a scripted "vision" verdict.

1. Orient: ``capabilities`` (topic ``texturing``), ``aether3d://texturing/guide`` and the
   schema ``aether3d://schema/texture-script``.
2. Build a crate and a barrel with ``execute_scene_plan``.
3. A first texture script with two mistakes an agent can make:

   * it binds a texture by a file path -- refused (``TEXTURE_NOT_FOUND`` at
     ``operations[3].texture``: scripts never read files), and nothing changes;
   * once fixed, it runs -- but gives the barrel a *planar* projection, which lays its
     front and back on the same UVs.

4. Check: ``inspect_uvs`` (overlaps image) and ``diagnose_scene`` with a ``uv_quality``
   expectation find the overlap exactly (``UV_OVERLAP``, basis ``geometry``).
5. Correct: a second script unwraps the barrel (charts, no overlaps), then paints both
   objects with surface strokes -- the barrel's iron hoops as rings around it, the crate's
   cross braces -- and fills. ``diagnose_scene`` again: accepted.
6. ``observe_scene`` renders the result; ``history`` lists one step per script;
   ``scene_save``.

Writes ``mcp_texture_agent.png`` (the barrel's planar UVs with the overlap in red, its
unwrapped UVs with the painted texture, the final render) and ``mcp_texture_agent.json``
(every call in brief with its time). Exits with status 1 if a check fails.
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
    "label": "Crate and barrel",
    "objects": {
        "ground": {"mesh": {"primitive": "plane", "size": 12}, "color": "#5d6168"},
        "crate": {"mesh": {"primitive": "cube", "size": 0.8}, "position": [-0.6, 0, 0.4]},
        "barrel": {
            "mesh": {"primitive": "cylinder", "radius": 0.35, "depth": 1.0, "segments": 48},
            "position": [0.6, 0.1, 0.5],
        },
        "key": {"light": {"type": "directional", "intensity": 2.4}, "direction": [0.5, 0.8, -0.7]},
        "rim": {"light": {"type": "directional", "intensity": 0.8}, "direction": [-0.6, -0.4, -0.4]},  # noqa: E501
        "camera": {"camera": {"fov": 35}, "position": [1.4, -3.4, 2.0], "look_at": [0, 0, 0.45]},
    },
}  # fmt: skip

FIRST = {
    "format": "aether3d.texture-script",
    "version": 1,
    "label": "Map and texture",
    "operations": [
        {"op": "unwrap", "object": "crate", "resolution": 512, "padding": 6},
        {"op": "project_uvs", "object": "barrel", "projection": "planar", "axis": "y"},
        {"op": "add_texture", "name": "Crate wood", "width": 512,
         "background": "#9a6b3c", "layers": [{"name": "Braces"}]},
        {"op": "bind", "object": "crate", "slot": "base_color", "texture": "crate_wood.png"},
        {"op": "add_texture", "name": "Barrel wood", "width": 512,
         "background": "#7a4a26", "layers": [{"name": "Hoops"}]},
        {"op": "bind", "object": "barrel", "slot": "base_color", "texture": "Barrel wood",
         "factor": "#ffffff"},
    ],
}  # fmt: skip

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "checks": [
        {"id": "clean-uvs", "check": "uv_quality", "objects": ["crate", "barrel"],
         "resolution": 512, "max_density_spread": 0.6},
        {"id": "seen", "check": "visible", "objects": ["crate", "barrel"]},
    ],
}  # fmt: skip


def ring(z: float, radius: float = 0.355, n: int = 49) -> list[list[float]]:
    a = np.linspace(0, 2 * np.pi, n)
    return [[0.6 + radius * np.cos(t), 0.1 + radius * np.sin(t), z] for t in a]


SECOND = {
    "format": "aether3d.texture-script",
    "version": 1,
    "label": "Unwrap the barrel and paint",
    "limits": {"max_seconds": 120, "max_dabs": 20000},
    "operations": [
        {"op": "unwrap", "object": "barrel", "resolution": 512, "padding": 6},
        {"op": "uv_report", "object": "barrel", "resolution": 512},
        *[
            {"op": "stroke", "object": "barrel", "texture": "Barrel wood", "layer": "Hoops",
             "brush": {"radius": 0.035, "color": "#3b3d40", "hardness": 0.9, "spacing": 0.2},
             "points": ring(z)}
            for z in (0.12, 0.38, 0.62, 0.88)
        ],
        *[
            {"op": "stroke", "object": "crate", "texture": "Crate wood", "layer": "Braces",
             "brush": {"radius": 0.05, "color": "#6b4423", "hardness": 0.85},
             "points": points}
            for points in (
                [[-0.95, -0.401, 0.05], [-0.25, -0.401, 0.75]],
                [[-0.95, -0.401, 0.75], [-0.25, -0.401, 0.05]],
                [[-0.999, -0.35, 0.05], [-0.999, 0.35, 0.75]],
                [[-0.999, -0.35, 0.75], [-0.999, 0.35, 0.05]],
            )
        ],
        {"op": "fill", "object": "barrel", "texture": "Barrel wood", "layer": "Hoops",
         "color": "#5a3a1c", "where": {"islands": [0]}, "opacity": 0.25},
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
    if name == "inspect_uvs":
        keys = ("islands", "flipped_triangles", "overlapping_triangles", "stacked_pair_count")
        return {k: data.get(k) for k in keys}
    if name == "run_script":
        return {"operations": [op["op"] for op in data["operations"]]}
    if name == "diagnose_scene":
        return {
            "accepted": data["accepted"],
            "findings": [(f["code"], f["basis"], f.get("magnitude")) for f in data["findings"]],
        }
    if name == "observe_scene":
        return {"renderer": data["observations"][0]["renderer"]}
    if name == "history":
        return {"steps": [s["label"] for s in data["steps"]]}
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
        caps, _ = await agent.call("capabilities", {"topic": "texturing"})
        guide = await client.read_resource("aether3d://texturing/guide")
        schema = await client.read_resource("aether3d://schema/texture-script")
        out["orient"] = {
            "operations": caps["operations"],
            "bake_maps": caps["bake_maps"],
            "guide_chars": len(guide.contents[0].text),
            "schema_bytes": len(schema.contents[0].text),
        }
        await agent.call("scene_new", {"path": "workshop.aether"})
        await agent.call("execute_scene_plan", {"plan": PLAN})

        info, _ = await agent.call("scene_info")
        refused, _ = await agent.call("run_script", {"script": FIRST}, error=True)
        out["refused"] = refused["error"]
        after, _ = await agent.call("scene_info")
        out["unchanged"] = after["revision"] == info["revision"]

        fixed = json.loads(json.dumps(FIRST))
        fixed["operations"][3] = {"op": "bind", "object": "crate", "slot": "base_color",
                                  "texture": "Crate wood", "factor": "#ffffff"}  # fmt: skip
        out["first"], _ = await agent.call("run_script", {"script": fixed})
        out["planar"], images = await agent.call(
            "inspect_uvs", {"object": "barrel", "image": "overlaps", "size": size}
        )
        out["planar_png"] = images[0]
        await agent.call("observe_scene", {"images": ["image"], "width": size, "height": size})
        out["diagnosis"], _ = await agent.call("diagnose_scene", {"expectations": EXPECTATIONS})

        out["second"], _ = await agent.call("run_script", {"script": SECOND})
        out["unwrapped"], images = await agent.call(
            "inspect_uvs",
            {"object": "barrel", "image": "texture", "texture": "Barrel wood", "size": size},
        )
        out["unwrapped_png"] = images[0]
        out["final"], _ = await agent.call("diagnose_scene", {"expectations": EXPECTATIONS})
        _, images = await agent.call(
            "observe_scene", {"images": ["image"], "width": size, "height": size}
        )
        out["render_png"] = images[0]
        out["texture"], _ = await agent.call("inspect_texture", {"texture": "Barrel wood",
                                                                  "preview_size": 0})  # fmt: skip
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
    workspace = out / "mcp_texture_workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    size = 360 if args.quick else 720
    t0 = time.perf_counter()
    result = anyio.run(run, workspace, args.backend, size)
    seconds = time.perf_counter() - t0

    failures = []
    if result["refused"]["code"] != "TEXTURE_NOT_FOUND" or not result["unchanged"]:
        failures.append(f"the file-path bind should be refused, changing nothing: "
                        f"{result['refused']}")  # fmt: skip
    if result["refused"]["location"] != "operations[3].texture":
        failures.append(f"refusal location: {result['refused']['location']}")
    if result["planar"]["overlapping_triangles"] == 0:
        failures.append("the planar projection should overlap the barrel's front and back")
    found = {(f["code"], f["subjects"][0]["name"]) for f in result["diagnosis"]["findings"]}
    if ("UV_OVERLAP", "barrel") not in found or any(n == "crate" for _, n in found):
        failures.append(f"the diagnosis should find the barrel's overlap only: {found}")
    if result["unwrapped"]["overlapping_triangles"] or result["unwrapped"]["flipped_triangles"]:
        failures.append("the unwrapped barrel should have clean UVs")
    if not result["final"]["accepted"]:
        failures.append(f"not accepted at the end: {brief('diagnose_scene', result['final'])}")
    steps = [s["label"] for s in result["history"]["steps"]]
    for label in ("Map and texture", "Unwrap the barrel and paint"):
        if steps.count(label) != 1:
            failures.append(f"{label!r} should be one undo step: {steps}")

    from aether3d.rendering.png import encode_png

    panels = [decode(result.pop(k)) for k in ("planar_png", "unwrapped_png", "render_png")]
    image = np.concatenate(panels, axis=1)
    (out / "mcp_texture_agent.png").write_bytes(
        encode_png(image, text={"Software": f"Aether3D {__version__}"})
    )
    report = {
        "aether3d": __version__,
        "seconds": seconds,
        "orient": result["orient"],
        "refused": result["refused"],
        "planar": brief("inspect_uvs", result["planar"]),
        "diagnosis": brief("diagnose_scene", result["diagnosis"]),
        "unwrapped": brief("inspect_uvs", result["unwrapped"]),
        "final": brief("diagnose_scene", result["final"]),
        "texture": result["texture"]["texture"],
        "history": steps,
        "transcript": result["transcript"],
        "failures": failures,
    }
    (out / "mcp_texture_agent.json").write_text(json.dumps(report, indent=2, default=str))
    for entry in result["transcript"]:
        print(f"  {entry['tool']:<18} {entry['seconds']:7.3f} s  {entry['brief']}")
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
