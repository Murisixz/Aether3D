"""Showcase 6 (Stage 17): an agent animates over MCP -- build, rig, generate, measure, make
a mistake, find it, correct it, verify, render, export -- with a full transcript.

A real ``aether3d mcp serve`` process (stdio, workspace-confined) is driven by a scripted
client that calls tools the way an agent would and decides only from their structured
JSON results:

1. ``capabilities``, ``scene_new``; ``execute_scene_plan`` builds a ground, a light, a
   camera, an armature and sixteen box parts of a robot (one undo step);
2. ``run_script`` (an animation script): a 17-joint rig, the parts attached, a generated
   walk with root motion, a damped follow camera -- one undo step, all or nothing;
3. ``inspect_animation`` (contacts, trajectory): the feet are planted, the distance
   covered is measured; ``history`` checkpoint;
4. a director's note -- "cover the distance 30% faster" -- and the naive edit: the root
   motion's keys scaled by 1.3 (``edit_keys``), so the body outruns its steps;
5. ``diagnose_scene`` with expectations (foot contacts at the default limits, continuity):
   FOOT_SLIDING, with the measured sliding as evidence;
6. ``propose_corrections`` (animation corrections opt in: ``allow_animation``) proposes
   ``plant_feet``; ``apply_corrections`` applies it as one undo step and verifies it: it
   is kept if it improves the finding without breaking a passing check;
7. measured again: better, still sliding -- the cause is the edit, so the agent goes back
   to the checkpoint and retimes the whole cycle instead (an NLA strip at speed 1.3 that
   accumulates root motion: steps and travel speed up together), refits the follow
   camera, measures (contacts, distance) and diagnoses until accepted;
8. refusals: a script reaching for a Python attribute through a property path, a key of
   the wrong type and a save path outside the workspace are refused with located errors,
   and the scene's revision is unchanged;
9. ``render_preview`` at three times (the server saves the PNGs in the workspace),
   ``export_scene`` (GLB), ``scene_save``, ``history``.

**The client does not look at images.** It is deterministic code standing in for an
agent: it reads only the structured results; the preview images it receives are saved for
people and never analysed (the transcript marks each as ``inspected: false``). A
vision-capable agent could look at them; this one does not, and claims nothing about them.

Run from the repository root (needs ``pip install "aether3d[mcp]"``):

    python examples/animation_mcp_agent.py [OUTPUT_DIR]

Writes to ``OUTPUT_DIR`` (default ``examples/output/animation/mcp_agent``; also the
server's workspace): ``transcript.json`` (every call: arguments, structured result,
milliseconds), ``transcript.md`` (readable), ``preview_*.png``, ``robot.glb``,
``robot.aether`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from animation_common import validate_glb, write_json  # noqa: E402

FEET = ["Foot.L", "Foot.R"]
PARTS = {  # name: (joint, center at rest, size) -- a robot of boxes (faces -Y)
    "Pelvis": ("Hips", (0, 0, 0.95), (0.3, 0.18, 0.14)),
    "Torso": ("Chest", (0, 0, 1.25), (0.36, 0.2, 0.42)),
    "Head": ("Head", (0, 0, 1.7), (0.22, 0.24, 0.26)),
}
for _side, _x in (("L", 1), ("R", -1)):
    PARTS.update({
        f"UpperArm.{_side}": (f"UpperArm.{_side}", (0.21 * _x, 0, 1.3), (0.09, 0.09, 0.29)),
        f"ForeArm.{_side}": (f"ForeArm.{_side}", (0.21 * _x, 0, 1.025), (0.08, 0.08, 0.25)),
        f"Thigh.{_side}": (f"Thigh.{_side}", (0.1 * _x, 0, 0.71), (0.12, 0.12, 0.44)),
        f"Shin.{_side}": (f"Shin.{_side}", (0.1 * _x, 0, 0.285), (0.1, 0.1, 0.41)),
        f"Foot.{_side}": (f"Foot.{_side}", (0.1 * _x, -0.06, 0.04), (0.1, 0.24, 0.08)),
    })  # fmt: skip


def joints() -> list[dict]:
    out = [
        {"name": "Hips", "position": [0, 0, 0.95]},
        {"name": "Spine", "parent": "Hips", "position": [0, 0, 0.12]},
        {"name": "Chest", "parent": "Spine", "position": [0, 0, 0.2]},
        {"name": "Neck", "parent": "Chest", "position": [0, 0, 0.22]},
        {"name": "Head", "parent": "Neck", "position": [0, 0, 0.1], "tail": [0, 0, 0.24]},
    ]
    for side, x in (("L", 1), ("R", -1)):
        out += [
            {"name": f"UpperArm.{side}", "parent": "Chest", "position": [0.21 * x, 0, 0.17],
             "tail": [0, 0, -0.29]},
            {"name": f"ForeArm.{side}", "parent": f"UpperArm.{side}", "position": [0, 0, -0.29],
             "tail": [0, 0, -0.25]},
            {"name": f"Hand.{side}", "parent": f"ForeArm.{side}", "position": [0, 0, -0.25]},
            {"name": f"Thigh.{side}", "parent": "Hips", "position": [0.1 * x, 0, -0.02],
             "tail": [0, 0, -0.44]},
            {"name": f"Shin.{side}", "parent": f"Thigh.{side}", "position": [0, 0, -0.44],
             "tail": [0, 0, -0.41]},
            {"name": f"Foot.{side}", "parent": f"Shin.{side}", "position": [0, 0, -0.41],
             "tail": [0, -0.15, -0.08]},
        ]  # fmt: skip
    return out


def plan() -> dict:
    colors = {"Pelvis": "#3a4250", "Torso": "#d8892c", "Head": "#e7e2d6"}
    objects: dict[str, Any] = {
        "ground": {"name": "Ground", "mesh": {"primitive": "plane", "size": 30},
                   "color": "#6f8a5a"},
        "sun": {"name": "Sun", "light": {"type": "directional", "intensity": 2.4},
                "direction": [-0.4, 0.55, -0.8]},
        "cam": {"name": "Cam", "camera": {"fov": 40}, "position": [3.2, -3.0, 1.5],
                "look_at": [0, 0, 1.0]},
        "hero": {"name": "Hero"},
    }  # fmt: skip
    for name, (_, center, size) in PARTS.items():
        color = colors.get(name, "#d8892c" if "Arm" in name else "#3a4250")
        objects[name.replace(".", "_").lower()] = {
            "name": name, "mesh": {"primitive": "box", "size": list(size)},
            "position": list(center), "color": color}  # fmt: skip
    return {"format": "aether3d.scene-plan", "version": 1, "label": "Robot parts",
            "objects": objects}  # fmt: skip


def rig_script() -> dict:
    ops: list[dict] = [{"op": "timeline", "fps": 24, "start": 0, "end": 4},
                       {"op": "add_rig", "armature": "Hero", "joints": joints()}]  # fmt: skip
    ops += [{"op": "attach", "object": name, "armature": "Hero", "joint": joint}
            for name, (joint, _, _) in PARTS.items()]  # fmt: skip
    ops += [
        {"op": "generate", "armature": "Hero", "motion": "walk", "forward": "-y",
         "name": "Walk"},
        {"op": "follow_camera", "camera": "Cam", "target": "Hero", "offset": [3.2, -3.0, 1.5],
         "lag": 0.3, "aim": [0, 0, 1.0], "start": 0, "end": 4},
    ]  # fmt: skip
    return {"format": "aether3d.animation-script", "version": 1, "label": "Rig and walk",
            "operations": ops}  # fmt: skip


def faster_script() -> dict:
    """The naive edit: the root motion's values scaled by 1.3 (the body outruns its feet)."""
    return {"format": "aether3d.animation-script", "version": 1, "label": "Cover it faster",
            "operations": [{"op": "edit_keys", "target": "Hero", "clip": "Walk",
                            "path": "transform.position.y", "edit": "scale_values",
                            "factor": 1.3, "pivot": 0.0}]}  # fmt: skip


def retime_script() -> dict:
    """The edit that keeps the feet: the whole cycle 30% faster (an NLA strip at speed
    1.3 accumulating each cycle's root motion), and the follow camera fitted again."""
    strip = {"clip": "Walk", "speed": 1.3, "repeat": 8, "accumulate": True}
    return {"format": "aether3d.animation-script", "version": 1, "label": "Walk 30% faster",
            "operations": [
                {"op": "set_binding", "target": "Hero",
                 "data": {"layers": [{"name": "walk", "strips": [strip]}]}},
                {"op": "follow_camera", "camera": "Cam", "target": "Hero",
                 "offset": [3.2, -3.0, 1.5], "lag": 0.3, "aim": [0, 0, 1.0], "start": 0,
                 "end": 4}]}  # fmt: skip


EXPECT = {"format": "aether3d.expectations", "version": 1, "defaults": False, "checks": [
    {"check": "foot_contacts", "objects": "Hero", "feet": FEET},  # (default limits)
    {"check": "animation_continuity", "objects": "Hero"}]}  # fmt: skip


def _brief(tool: str, data: dict) -> dict:
    """The parts of a structured result a reader needs (the transcript keeps all)."""
    if data is None:
        return {}
    if "error" in data:
        e = data["error"]
        return {"refused": e.get("code"), "location": e.get("location"),
                "message": e.get("message")}  # fmt: skip
    if tool == "capabilities":
        return {"aether3d": data.get("aether3d"), "tools": len(data.get("tools", {}))}
    if tool == "history":
        return {"action": data.get("action"), "moved": len(data.get("moved", [])),
                "steps": [st["label"] for st in data.get("steps", [])],
                "checkpoints": data.get("checkpoints")}  # fmt: skip
    if tool == "inspect_animation" and data.get("what") == "trajectory":
        return {"path_length_m": round(data["path_length"], 4),
                "displacement_m": round(data["displacement"], 4)}  # fmt: skip
    if tool == "inspect_animation" and data.get("what") == "contacts":
        return {f: {"contacts": len(d["contacts"]), "sliding_max_m": round(d["sliding_max"], 4),
                    "penetration_m": round(d["penetration"], 4)}
                for f, d in data["feet"].items()}  # fmt: skip
    if tool in ("diagnose_scene", "propose_corrections"):
        out: dict[str, Any] = {}
        findings = data.get("findings") or data.get("diagnosis", {}).get("findings", [])
        out["findings"] = [f"{f['code']} ({f.get('magnitude')})" for f in findings]
        if "proposals" in data:
            out["proposals"] = [{"id": p["id"], "action": p["actions"][0]["action"],
                                 "safety": p["safety"]} for p in data["proposals"]]  # fmt: skip
        return out
    if tool == "apply_corrections":
        v = data.get("verification") or {}
        return {"status": data.get("status"), "applied": data.get("applied"),
                "decision": v.get("decision"), "reason": v.get("reason"),
                "findings": [{"code": f["code"], "before": round(f["before"], 4),
                              "after": round(f["after"], 4)}
                             for f in v.get("persisting", [])],
                "new_failures": [f["code"] for f in v.get("new_failures", [])]}  # fmt: skip
    if tool == "run_script":
        return {"operations": len(data.get("operations", []))}
    keep = ("revision", "path", "bytes", "width", "height", "saved", "objects", "status")
    return {k: data[k] for k in keep if k in data}


async def agent(workspace: Path) -> dict:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace)],
        env=dict(os.environ),
    )
    transcript: list[dict] = []
    report: dict[str, Any] = {}
    async with Client(server) as client:

        async def call(tool: str, args: dict | None = None, *, expect_error: bool = False):
            t0 = time.perf_counter()
            r = await client.call_tool(tool, args or {})
            ms = round(1000 * (time.perf_counter() - t0), 1)
            images = [c for c in r.content if getattr(c, "type", None) == "image"]
            entry = {"step": len(transcript) + 1, "tool": tool, "arguments": args or {},
                     "ok": not r.is_error, "ms": ms, "result": r.structured_content,
                     "images": [{"mime": c.mime_type, "inspected": False}
                                for c in images]}  # fmt: skip
            transcript.append(entry)
            if r.is_error != expect_error:
                raise RuntimeError(f"{tool}: {json.dumps(r.structured_content)[:600]}")
            return r.structured_content

        caps = await call("capabilities")
        report["server"] = {k: caps[k] for k in ("version", "tools") if k in caps}
        await call("scene_new", {"name": "Robot walk"})
        await call("execute_scene_plan", {"plan": plan()})
        await call("run_script", {"script": rig_script()})
        contacts = {"what": "contacts", "target": "Hero", "feet": FEET, "start": 0.5, "end": 4}
        path = {"what": "trajectory", "target": "Hero", "start": 0, "end": 4}
        allow = {"allow_animation": True}  # (animation corrections are opt-in)
        report["walk"] = _brief("inspect_animation", await call("inspect_animation", contacts))
        report["walk_distance_m"] = round((await call("inspect_animation", path))["path_length"], 4)
        await call("history", {"action": "checkpoint", "name": "walk"})
        await call("run_script", {"script": faster_script()})
        report["rushed"] = _brief("inspect_animation", await call("inspect_animation", contacts))
        diagnosis = await call("diagnose_scene", {"expectations": EXPECT, "include_image": False})
        report["diagnosis"] = _brief("diagnose_scene", diagnosis)
        held = await call("propose_corrections", {"expectations": EXPECT})
        report["proposals_default_policy"] = _brief("propose_corrections", held)
        proposals = await call("propose_corrections", {"expectations": EXPECT, "policy": allow})
        report["proposals"] = _brief("propose_corrections", proposals)
        plant = next(p for p in proposals["proposals"]
                     if p["actions"][0]["action"] == "plant_feet")  # fmt: skip
        ids = [plant["id"]]
        applied = await call("apply_corrections", {"proposals": ids, "approve": ids,
                                                   "policy": allow})  # fmt: skip
        report["applied"] = _brief("apply_corrections", applied)
        report["planted"] = _brief("inspect_animation", await call("inspect_animation", contacts))
        again = await call("diagnose_scene", {"expectations": EXPECT, "include_image": False})
        report["diagnosis_after_planting"] = _brief("diagnose_scene", again)

        # The edit is the cause: back to the checkpoint, and retime the whole cycle.
        await call("history", {"action": "goto", "name": "walk"})
        await call("run_script", {"script": retime_script()})
        report["retimed"] = _brief("inspect_animation", await call("inspect_animation", contacts))
        report["retimed_distance_m"] = round(
            (await call("inspect_animation", path))["path_length"], 4
        )
        final = await call("diagnose_scene", {"expectations": EXPECT, "include_image": False})
        report["final_diagnosis"] = _brief("diagnose_scene", final)
        report["accepted"] = final.get("accepted")

        # Refusals: nothing changes.
        before = (await call("scene_info"))["revision"]

        def script(*ops: dict) -> dict:
            return {"script": {"format": "aether3d.animation-script", "version": 1,
                               "operations": list(ops)}}  # fmt: skip

        reach = await call("run_script", script(
            {"op": "keyframe", "target": "Hero", "path": "__class__.__init__", "time": 1,
             "value": 0}), expect_error=True)  # fmt: skip
        typed = await call("run_script", script(
            {"op": "set_channel", "target": "Hero", "path": "transform.position.z",
             "keys": [[0, 0], [1, 2]]},
            {"op": "keyframe", "target": "Hero", "path": "transform.position.z", "time": 2,
             "value": "high"}), expect_error=True)  # fmt: skip
        escape = await call("scene_save", {"path": "../outside.aether"}, expect_error=True)
        after = (await call("scene_info"))["revision"]
        report["refusals"] = {
            "attribute_path": reach.get("error", reach),
            "wrong_type_after_a_valid_operation": typed.get("error", typed),
            "path_outside_workspace": escape.get("error", escape),
            "revision_unchanged": before == after,
        }

        for t in (0.6, 1.8, 3.2):
            preview = {"time": t, "width": 640, "height": 360, "shadows": True,
                       "save_to": f"preview_{t:.1f}s.png", "overwrite": True}  # fmt: skip
            await call("render_preview", preview)
        exported = await call("export_scene", {"path": "robot.glb", "animation": "clips",
                                               "overwrite": True})  # fmt: skip
        report["export"] = {k: exported.get(k) for k in ("path", "bytes", "animations", "skins")}
        await call("scene_save", {"path": "robot.aether", "overwrite": True})
        steps = await call("history", {"action": "list"})
        report["history"] = {k: v for k, v in steps.items() if k != "scene"}
    report["calls"] = len(transcript)
    report["images_received"] = sum(len(e["images"]) for e in transcript)
    report["images_inspected_by_the_client"] = 0
    return {"transcript": transcript, "report": report}


def markdown(transcript: list[dict], report: dict) -> str:
    lines = [
        "# MCP agent transcript (Stage 17 showcase 6)",
        "",
        "A scripted client (deterministic code standing in for an agent) drove a real "
        "`aether3d mcp serve` process. It decided only from structured JSON results; the "
        f"{report['images_received']} images it received were saved, never analysed.",
        "",
        "| # | tool | ok | ms | what came back |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in transcript:
        brief = _brief(e["tool"], e["result"] or {})
        if e["images"]:
            brief["images"] = f"{len(e['images'])} (not inspected)"
        text = json.dumps(brief, separators=(", ", ": "))[:220].replace("|", "\\|")
        lines.append(f"| {e['step']} | `{e['tool']}` | {'yes' if e['ok'] else 'refused'} | "
                     f"{e['ms']} | {text} |")  # fmt: skip
    lines += ["", "## Outcome", "", "```json", json.dumps(report, indent=2), "```", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output/animation/mcp_agent"))
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        import anyio
        import mcp  # noqa: F401
    except ImportError:
        print("needs the MCP SDK: pip install 'aether3d[mcp]'")
        return 2
    result = anyio.run(agent, out)
    report = result["report"]
    report["gltf_validator"] = validate_glb(out / "robot.glb")
    write_json(out / "transcript.json", result["transcript"])
    (out / "transcript.md").write_text(markdown(result["transcript"], report), encoding="utf-8")
    write_json(out / "report.json", report)
    keys = ("walk", "walk_distance_m", "rushed", "diagnosis", "applied", "planted",
            "diagnosis_after_planting", "retimed", "retimed_distance_m", "final_diagnosis",
            "accepted", "refusals")  # fmt: skip
    print(json.dumps({k: report.get(k) for k in keys}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
