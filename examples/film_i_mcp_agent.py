"""Showcase I -- an agent lights, checks, corrects, animates and renders over MCP.

Needs the MCP SDK (``pip install "aether3d[mcp]"``). This script plays an AI agent
connected to ``aether3d mcp serve``: there is no language model here, every step is the
tool call an agent would make, so the run is reproducible (and nothing claims a model
looked at an image).

1. ``scene_new``; ``execute_scene_plan`` builds a product shot in one call and one undo
   step: a backdrop, a plinth, two products (a terracotta ball, a ceramic vase) grouped, a
   camera.
2. ``run_script`` (a scene script) lights the group with a ``low_key`` light rig -- a
   deliberately harsh first choice (16:1).
3. ``diagnose_scene`` with expectations: the products' lighting ratio between 2:1 and 6:1
   (the ``lighting`` check: direct light measured on their visible surface), their
   outline separated from the backdrop, the key and the rim reaching them, the exposure
   sane. The ratio fails.
4. ``propose_corrections`` and ``apply_corrections`` of the proposal (``verify``: kept only
   because a new render measured it better), then ``diagnose_scene`` again: accepted.
5. ``run_script`` (an animation script) keys a slow camera arc around the products, and
   ``inspect_animation`` (``what: motion``) measures it in world space.
6. ``render_image`` with the film path: the GPU path tracer (the CPU one with
   ``--quick``), Sobol' samples, a Blackman-Harris filter, adaptive sampling, the
   denoiser and the film view transform; ``scene_save``.

Writes ``before_after.png`` (the observation before and after the correction),
``final.png`` (the film render), ``transcript.json`` (every call in brief, with its time)
and ``report.json``; the project stays in the server's workspace.

    .venv/bin/python examples/film_i_mcp_agent.py [--quick]
"""

from __future__ import annotations

import base64
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

from film_common import OUT, arguments, gpu_available, rel, write_report  # noqa: E402

from aether3d.rendering.png import write_png  # noqa: E402

PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "label": "Product shot",
    "materials": {
        "ceramic": {"base_color": "#ece6da", "roughness": 0.3},
        "terracotta": {"base_color": "#c2643c", "roughness": 0.65},
        "matte": {"base_color": "#e8e4dc", "roughness": 0.6},
        "grey": {"base_color": "#5d626b", "roughness": 0.9},
    },
    "objects": {
        "floor": {"mesh": {"primitive": "plane", "size": 12}, "material": "@grey"},
        "wall": {"mesh": {"primitive": "plane", "size": 12}, "material": "@grey",
                 "position": [0, 2.2, 3], "rotation_euler": [90, 0, 0]},
        "plinth": {"mesh": {"primitive": "cube", "size": 1}, "scale": [1.2, 0.8, 0.5],
                   "position": [0, 0, 0.25], "material": "@matte"},
        "products": {"position": [0, 0, 0.5], "tags": ["product"], "children": {
            "orb": {"mesh": {"primitive": "sphere", "radius": 0.17, "segments": 64,
                             "rings": 32}, "position": [-0.22, 0.0, 0.17],
                    "material": "@terracotta"},
            "vase": {"mesh": {"primitive": "cylinder", "radius": 0.09, "depth": 0.42,
                              "segments": 64}, "position": [0.22, 0.06, 0.21],
                     "material": "@ceramic"}}},
        "cam": {"camera": {"fov": 26}, "position": [0.2, -3.1, 1.15],
                "look_at": [0, 0, 0.68], "active": True},
    },
}  # fmt: skip
EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "Product shot",
    "views": {"main": {"camera": "cam"}},
    "default_view": "main",
    "defaults": False,
    "checks": [
        {"id": "light", "check": "lighting", "objects": "products",
         "ratio": {"min": 2, "max": 6}, "max_merged_fraction": 0.35,
         "lights": ["Key", "Rim"]},
        {"id": "exposure", "check": "exposure", "mean_luminance": {"min": 0.03, "max": 0.7},
         "max_clipped_fraction": 0.05, "severity": "warning"},
    ],
}  # fmt: skip
ORBIT = {
    "format": "aether3d.animation-script",
    "version": 1,
    "operations": [
        {"op": "set_channel", "target": "cam", "path": "transform.position.x",
         "keys": [[0, 0.2], [3, 1.4]], "interpolation": "sine", "ease": "in_out"},
        {"op": "set_channel", "target": "cam", "path": "transform.position.y",
         "keys": [[0, -3.1], [3, -2.8]], "interpolation": "sine", "ease": "in_out"},
        {"op": "add_constraint", "owner": "cam",
         "constraint": {"type": "look_at", "target": {"object": "products"},
                        "target_offset": [0, 0, 0.18]}},
        {"op": "timeline", "fps": 60, "start": 0, "end": 3},
    ],
}  # fmt: skip


class Agent:
    """Tool calls with a transcript (each call's time and a brief of its reply)."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.transcript: list[dict] = []

    async def call(self, name: str, arguments: dict | None = None) -> tuple[dict, list[bytes]]:
        t0 = time.perf_counter()
        result = await self.client.call_tool(name, arguments or {})
        data = result.structured_content
        images = [base64.b64decode(c.data) for c in result.content if c.type == "image"]
        self.transcript.append({"tool": name, "seconds": round(time.perf_counter() - t0, 3),
                                "error": result.is_error, "images": len(images),
                                "brief": brief(name, data)})  # fmt: skip
        if result.is_error:
            raise RuntimeError(f"{name}: {data['error']}")
        return data, images


def brief(name: str, data: dict) -> dict:
    if "error" in data:
        return {"error": data["error"].get("code")}
    if name == "diagnose_scene":
        light = next((c for c in data["checks"] if c["id"] == "light"), {})
        return {"accepted": data["accepted"],
                "findings": [(f["code"], f["basis"], f.get("magnitude")) for f in data["findings"]],
                "ratio": (light.get("measured") or {}).get("ratio")}  # fmt: skip
    if name == "propose_corrections":
        return {"proposals": [(p["id"], p["safety"], p["title"]) for p in data["proposals"]]}
    if name == "apply_corrections":
        v = data.get("verification") or {}
        return {"status": data["status"], "decision": v.get("decision")}
    if name == "inspect_animation":
        return {"findings": [f["code"] for f in data.get("findings", [])]}
    return {k: data[k] for k in list(data)[:3] if not isinstance(data[k], (list, dict))}


def decode(png: bytes) -> np.ndarray:
    from aether3d.materials.images import decode_image

    return np.asarray(decode_image(png).pixels)[..., :3]


async def run(workspace: Path, quick: bool) -> dict:
    backend = "cpu" if quick else ("gpu" if gpu_available() else "cpu")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace),
              "--feedback-backend", backend],
        env=dict(os.environ),
    )  # fmt: skip
    out: dict = {}
    size = (320, 180) if quick else (960, 540)
    async with Client(params) as client:
        agent = Agent(client)
        await agent.call("scene_new", {"path": "product.aether"})
        built, _ = await agent.call("execute_scene_plan", {"plan": PLAN})
        out["plan"] = {"status": built["status"], "objects": built["counts"]["objects"]}
        rig, _ = await agent.call("run_script", {"script": {
            "format": "aether3d.scene-script", "version": 1,
            "operations": [{"op": "light_rig", "subject": "products", "preset": "low_key",
                            "temperature": 5200}]}})  # fmt: skip
        out["rig"] = rig.get("results", rig)
        look = {"images": ["image"], "width": size[0], "height": size[1]}
        _, before = await agent.call("observe_scene", look)
        first, _ = await agent.call("diagnose_scene", {"expectations": EXPECTATIONS})
        out["first"] = brief("diagnose_scene", first)
        proposals, _ = await agent.call("propose_corrections", {})
        out["proposals"] = brief("propose_corrections", proposals)
        chosen = [p["id"] for p in proposals["proposals"] if p["safety"] in ("safe", "verify")]
        request = {"proposals": chosen, "expected_revision": first["revision"]}
        applied, _ = await agent.call("apply_corrections", request)
        out["applied"] = brief("apply_corrections", applied)
        final, _ = await agent.call("diagnose_scene", {})
        out["final"] = brief("diagnose_scene", final)
        _, after = await agent.call("observe_scene", look)
        await agent.call("run_script", {"script": ORBIT})
        motion, _ = await agent.call("inspect_animation", {"what": "motion", "targets": ["cam"],
                                                           "subject": "products"})  # fmt: skip
        out["camera_motion"] = brief("inspect_animation", motion)
        render = {"path": "final.png", "width": size[0], "height": size[1], "time": 1.5,
                  "spp": 16 if quick else 256, "sampler": "sobol", "filter": "blackman_harris",
                  "adaptive": 0.03, "denoise": True, "tonemap": "film",
                  "backend": "path" if quick or backend == "cpu" else "gpu_path"}  # fmt: skip
        rendered, _ = await agent.call("render_image", render)
        out["render"] = {k: rendered.get(k) for k in ("backend", "seconds", "diagnostics")}
        await agent.call("scene_save", {})
        out["transcript"] = agent.transcript
        out["images"] = {"before": before[0], "after": after[0]}
    return out


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "i_mcp_agent"
    out.mkdir(parents=True, exist_ok=True)
    workspace = out / "workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    t0 = time.perf_counter()
    result = anyio.run(run, workspace, args.quick)
    images = result.pop("images")
    left, right = decode(images["before"]), decode(images["after"])
    gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
    write_png(out / "before_after.png", np.concatenate([left, gap, right], axis=1))
    shutil.copyfile(workspace / "final.png", out / "final.png")
    write_report(out / "transcript.json", {"calls": result.pop("transcript")})
    report = {"showcase": "I agent workflow over MCP", "quick": args.quick, **result,
              "before_after": rel(out / "before_after.png"), "final_png": rel(out / "final.png"),
              "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
    write_report(out / "report.json", report)
    return report


if __name__ == "__main__":
    data = main()
    print(f"I agent: {data['seconds']} s, first {data['first']} -> final {data['final']}")
