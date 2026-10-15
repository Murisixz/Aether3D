"""Talk to the Aether3D MCP server from Python, the way an AI agent's host does (Stage 11).

Needs the MCP SDK (``pip install "aether3d[mcp]"``). Run from the repository root:

    python examples/mcp_client.py [WORKSPACE] [--backend auto|gpu|cpu]

Starts ``aether3d mcp serve --workspace WORKSPACE`` as a subprocess and speaks MCP to it
over stdio with the official SDK's client:

1. lists the server's tools and resources;
2. starts a scene and builds it with one scene plan -- the ``basics`` example, read from
   the server's ``aether3d://examples/basics`` resource;
3. asks spatial questions (what is nearest to a point, what a ray hits);
4. renders a preview: the GPU rasterizer when it works, the CPU one otherwise, and the
   reply says which (``--backend`` picks one);
5. saves the project.

Every tool returns structured JSON (also as text, for clients that read only text);
errors come back as tool results with a code and a hint. The server only reads and
writes inside WORKSPACE (default: ``examples/output/mcp_workspace``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import anyio
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
except ImportError:  # pragma: no cover - the message is the point
    sys.exit('this example needs the MCP SDK: pip install "aether3d[mcp]"')

HERE = Path(__file__).resolve().parent


async def call(client: Client, tool: str, arguments: dict | None = None) -> dict:
    """Call a tool; print and return its structured reply (an error raises)."""
    result = await client.call_tool(tool, arguments or {})
    reply = result.structured_content
    if result.is_error:
        raise RuntimeError(f"{tool} failed: {reply['error']['code']}: {reply['error']['message']}")
    return reply


async def run(workspace: Path, backend: str) -> int:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace)],
        env=dict(os.environ),
    )
    async with Client(server) as client:
        print(f"connected to {client.server_info.name} {client.server_info.version}")

        print("1. What the server offers")
        tools = (await client.list_tools()).tools
        print(f"   {len(tools)} tools: {', '.join(t.name for t in tools)}")
        resources = (await client.list_resources()).resources
        print(f"   resources: {', '.join(str(r.uri) for r in resources)}")

        print("2. One plan builds the scene")
        await call(client, "scene_new", {"name": "Courtyard"})
        example = await client.read_resource("aether3d://examples/basics")
        plan = json.loads(example.contents[0].text)
        built = await call(client, "execute_scene_plan", {"plan": plan})
        revision = built["revision"]
        print(f"   {built['status']}: {built['counts']['objects']} objects in one undo step")
        print(f"   (revision {revision['before']} -> {revision['after']})")
        print(f"   pillars touching each other: {built['queries'][0]['result']}")

        print("3. Spatial questions")
        answers = await call(
            client,
            "spatial_query",
            {
                "requests": [
                    {"query": "nearest", "point": [0, 0, 1], "count": 2},
                    {"query": "raycast", "origin": [0, -20, 1.5], "direction": [0, 1, 0]},
                ]
            },
        )
        nearest, ray = (a["result"] for a in answers["results"])
        print(f"   nearest to the center: {[n['object']['name'] for n in nearest]}")
        print(f"   a ray along +Y hits {ray['object']['name']} at {ray['distance']:.2f} m")

        print("4. A preview")
        preview = await client.call_tool(
            "render_preview",
            {
                "width": 640,
                "height": 360,
                "backend": backend,
                "save_to": "preview.png",
                "overwrite": True,
            },
        )
        report = preview.structured_content
        if preview.is_error:
            print(f"   no preview: {report['error']['message']}")
            return 1
        image = [c for c in preview.content if c.type == "image"][0]
        where = f"GPU, {report['adapter']['name']}" if report["device"] == "gpu" else "CPU"
        print(f"   {report['width']}x{report['height']} on the {where}: {report['saved']}")
        print(f"   ({len(image.data)} base64 characters of PNG came back in the reply)")
        if "fallback" in report:
            print(f"   the GPU was not used: {report['fallback']}")

        print("5. Save")
        saved = await call(client, "scene_save", {"path": "courtyard.aether", "overwrite": True})
        print(f"   wrote {saved['saved']} ({saved['scene']['objects']} objects)")
    print("done")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace", nargs="?", default=str(HERE / "output" / "mcp_workspace"))
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return anyio.run(run, workspace, args.backend)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
