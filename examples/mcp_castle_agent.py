"""An agent's whole workflow over MCP: build a castle, find its mistakes, fix them (Stage 11).

Needs the MCP SDK (``pip install "aether3d[mcp]"``). Run from the repository root:

    python examples/mcp_castle_agent.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

This script plays the part of an AI agent connected to ``aether3d mcp serve``. There is
no language model here: each step is the MCP tool call an agent would make, with the
arguments it would send, so the run is reproducible. The first draft of the castle has
two mistakes on purpose, of the kind a first draft has; the point is the loop that finds
and fixes them.

1. Orient: the server's capabilities, the scene plan guide and the castle example (MCP
   resources an agent reads before writing a plan).
2. Build: one ``execute_scene_plan`` call makes the Minecraft-style castle -- 2,190
   blocks and parts, one undo step -- from the example plan with two mistakes: a tree
   planted inside the keep's wall, and a corner tower 3 m above the ground.
3. Inspect: one ``spatial_query`` call with a batch of questions -- which trees intersect
   the keep, where each tower's lowest point is, whether the gate is clear. The answers
   are computed from the triangles and change nothing.
4. Fix: a small plan of operations moves the tree and lowers the tower, and asks the same
   questions again in the same call: no problems left. ``expected_revision`` makes the
   fix refuse to apply to a scene that changed since the inspection.
5. Undo the fix (the problems are back), redo it (they are gone): ``history``.
6. A quick preview with ``render_preview``: the GPU rasterizer when it works, the CPU one
   otherwise; the reply says which, and never path traces.
7. Save the project with its history, export a binary glTF (.glb), and open the saved
   project again: its objects and its undo history come back.

Writes ``mcp_castle_agent.png`` (the preview) and ``mcp_castle_agent.json`` (every call
with its arguments in brief, its time and what it found) to OUTPUT_DIR; the project and
the .glb stay in the server's workspace, ``OUTPUT_DIR/mcp_castle_workspace``. Exits with
status 1 if a check fails.
"""

from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
PLAN = "aether3d.scene-plan"
TOWERS = ["Tower SW", "Tower SE", "Tower NE", "Tower NW"]

# The questions of step 3, asked again by the fix (selectors, not hundreds of names).
TREES_IN_THE_KEEP = {
    "query": "overlaps",
    "objects": {"name_glob": "Tree */*"},
    "against": {"under": "keep"},
}
TOWER_BOTTOMS = [{"query": "bounds", "object": t, "children": True} for t in TOWERS]
GATE_IS_CLEAR = {"query": "line_of_sight", "start": [0, -30, 2], "end": [0, -8, 2]}


def first_draft(castle: dict) -> dict:
    """The example castle with two mistakes, as an agent's first draft might have."""
    plan = json.loads(json.dumps(castle))
    plan["label"] = "Castle (first draft)"
    plan["objects"]["trees"]["instances"][1]["position"] = [-3, -0.6, 0]  # in the keep's wall
    plan["objects"]["towers"]["instances"][2]["position"] = [15, 15, 3]  # floating
    return plan


class Agent:
    """Calls tools, times them, and keeps a transcript."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.transcript: list[dict] = []

    async def call(self, tool: str, arguments: dict | None = None, *, brief: str = "") -> dict:
        started = time.perf_counter()
        result = await self.client.call_tool(tool, arguments or {})
        ms = (time.perf_counter() - started) * 1e3
        reply = result.structured_content
        self.transcript.append(
            {
                "tool": tool,
                "arguments": brief or arguments,
                "ms": round(ms, 1),
                "ok": not result.is_error,
            }
        )
        if result.is_error:
            raise RuntimeError(f"{tool}: {reply['error']['code']}: {reply['error']['message']}")
        print(f"   -> {tool} ({ms:.0f} ms)")
        return reply

    def note(self, **facts) -> None:
        self.transcript[-1]["found"] = facts


def problems_in(answers: list[dict]) -> dict:
    """What the batch of step 3 says, as facts: overlapping trees, floating towers."""
    overlaps, *bottoms = answers
    trees = sorted(
        {pair[side]["name"].split("/")[0] for pair in overlaps["result"] for side in ("a", "b")}
        - {"keep"}
    )
    floating = {
        tower: round(b["result"]["aabb"]["min"][2], 3)
        for tower, b in zip(TOWERS, bottoms, strict=True)
        if b["result"]["aabb"]["min"][2] > 0.05  # the ground's top is at z = 0
    }
    return {"trees_in_the_keep": trees, "floating_towers": floating}


async def run(output: Path, workspace: Path, quick: bool, backend: str) -> int:
    problems: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(f"   {'ok' if ok else 'FAILED'}: {what}")
        if not ok:
            problems.append(what)

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace)],
        env=dict(os.environ),
    )
    async with Client(server) as client:
        agent = Agent(client)

        print("1. Orient")
        info = await agent.call("capabilities")
        print(f"   Aether3D {info['aether3d']}: {len(info['tools'])} tools, {info['conventions']}")
        guide = (await client.read_resource("aether3d://guide")).contents[0].text
        castle = json.loads(
            (await client.read_resource("aether3d://examples/castle")).contents[0].text
        )
        print(f"   read the guide ({len(guide)} characters) and the castle example")

        print("2. Build the castle with one plan (with two mistakes)")
        await agent.call("scene_new", {"name": "Castle"})
        built = await agent.call(
            "execute_scene_plan", {"plan": first_draft(castle)}, brief="the castle (first draft)"
        )
        server_ms = built["timings"]["total"] * 1e3
        objects = built["counts"]["objects"]
        print(f"   {objects} objects, one undo step ({server_ms:.0f} ms in the server)")
        agent.note(objects=built["counts"]["objects"], server_ms=round(server_ms, 1))
        check(built["counts"]["objects"] == 2190, "2,190 objects")

        print("3. Inspect")
        answers = await agent.call(
            "spatial_query",
            {"requests": [TREES_IN_THE_KEEP, *TOWER_BOTTOMS, GATE_IS_CLEAR]},
            brief="trees in the keep, the towers' lowest points, the gate",
        )
        found = problems_in(answers["results"][:5])
        gate = answers["results"][5]["result"]
        agent.note(**found, gate_clear=gate["clear"])
        print(f"   trees inside the keep: {found['trees_in_the_keep']}")
        print(f"   towers above the ground: {found['floating_towers']}")
        print(f"   the gate is clear: {gate['clear']}")
        check(
            found == {"trees_in_the_keep": ["Tree 1"], "floating_towers": {"Tower NE": 3.0}},
            "found both",
        )
        check(gate["clear"], "nothing blocks the gate")

        print("4. Fix, and check in the same call")
        fix = {
            "format": PLAN,
            "version": 1,
            "label": "Fix: move Tree 1 out of the keep, lower Tower NE",
            "expected_revision": answers["revision"],
            "operations": [
                {"op": "update", "select": "Tree 1", "set": {"position": [-8, -4, 0]}},
                {"op": "update", "select": "Tower NE", "set": {"translate": [0, 0, -3]}},
            ],
            "queries": [TREES_IN_THE_KEEP, *TOWER_BOTTOMS],
        }
        fixed = await agent.call("execute_scene_plan", {"plan": fix}, brief=fix["label"])
        after = problems_in(fixed["queries"])
        agent.note(**after)
        check(after == {"trees_in_the_keep": [], "floating_towers": {}}, "no problems left")

        print("5. Undo the fix, then redo it")
        undone = await agent.call("history", {"action": "undo"})
        again = await agent.call("spatial_query", {"requests": [TREES_IN_THE_KEEP, *TOWER_BOTTOMS]})
        back = problems_in(again["results"])
        print(f"   undid {undone['moved'][0]['label']!r}: problems back: {back}")
        check(back == found, "undo brings the mistakes back")
        await agent.call("history", {"action": "redo"})
        again = await agent.call("spatial_query", {"requests": [TREES_IN_THE_KEEP, *TOWER_BOTTOMS]})
        check(problems_in(again["results"]) == after, "redo fixes them again")

        print("6. A quick preview")
        width, height = (320, 180) if quick else (960, 540)
        preview = await agent.call(
            "render_preview",
            {
                "width": width,
                "height": height,
                "backend": backend,
                "save_to": "castle_preview.png",
                "overwrite": True,
            },
        )
        device = f"GPU ({preview['adapter']['name']})" if preview["device"] == "gpu" else "CPU"
        print(f"   {width}x{height} on the {device}")
        agent.note(
            device=preview["device"],
            backend=preview["backend"],
            draw_calls=preview.get("draw_calls"),
        )
        if backend == "gpu":
            check(preview["device"] == "gpu", "rendered on the GPU, as asked")

        print("7. Save, export, reopen")
        saved = await agent.call(
            "scene_save", {"path": "castle.aether", "overwrite": True, "history": True}
        )
        exported = await agent.call("export_scene", {"path": "castle.glb", "overwrite": True})
        size = exported["bytes"] / 1e6
        print(f"   {saved['saved']}; {exported['path']}: {size:.1f} MB, {exported['nodes']} nodes")
        reopened = await agent.call("scene_open", {"path": "castle.aether", "history": True})
        check(reopened["scene"]["objects"] == 2190, "the saved project has every object")
        steps = [s["label"] for s in (await agent.call("history", {"action": "list"}))["steps"]]
        print(f"   its history came back with it: {steps}")
        check(steps == ["Castle (first draft)", fix["label"]], "the history was saved with it")

    shutil.copyfile(workspace / "castle_preview.png", output / "mcp_castle_agent.png")
    calls = agent.transcript
    report = {
        "calls": calls,
        "tool_calls": len(calls),
        "total_ms": round(sum(c["ms"] for c in calls), 1),
        "problems": problems,
    }
    (output / "mcp_castle_agent.json").write_text(json.dumps(report, indent=2) + "\n")
    seconds = report["total_ms"] / 1e3
    print(f"{len(calls)} tool calls in {seconds:.1f} s; wrote mcp_castle_agent.png and .json")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="a smaller preview")
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    workspace = output / "mcp_castle_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return anyio.run(run, output, workspace, args.quick, args.backend)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
