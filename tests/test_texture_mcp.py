"""Texturing through the MCP server (Stage 14): texture scripts in run_script, the
read-only inspect_uvs and inspect_texture tools with their images, the texturing
capabilities topic, guide and schema resources, and errors that change nothing."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
jsonschema = pytest.importorskip("jsonschema")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "objects": {
        "Crate": {"mesh": {"primitive": "cube", "size": 1}},
        "Ball": {"mesh": {"primitive": "sphere", "segments": 24, "rings": 12},
                 "position": [2, 0, 0]},
    },
}  # fmt: skip

SCRIPT = {
    "format": "aether3d.texture-script",
    "version": 1,
    "label": "Paint the crate",
    "operations": [
        {"op": "unwrap", "object": "Crate", "resolution": 256, "padding": 4},
        {"op": "add_texture", "name": "Crate color", "width": 256, "layers": [{"name": "Ink"}]},
        {"op": "bind", "object": "Crate", "slot": "base_color", "texture": "Crate color",
         "factor": "#ffffff"},
        {"op": "stroke", "object": "Crate", "texture": "Crate color",
         "brush": {"radius": 0.2, "color": "#cc2200"}, "points": [[0.5, -0.2, 0], [0.5, 0.2, 0]]},
    ],
}  # fmt: skip


def session(workspace: Path, body):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace)],
        env=dict(os.environ),
    )

    async def main():
        async with Client(params, mode="legacy") as client:
            return await body(client)

    return anyio.run(main)


async def call(client, name, arguments=None, *, error=False):
    result = await client.call_tool(name, arguments or {})
    assert result.is_error == error, result.structured_content
    images = [base64.b64decode(c.data) for c in result.content if c.type == "image"]
    return result.structured_content, images


def test_texturing_through_mcp(tmp_path):
    async def body(client):
        out = {}
        out["tools"] = [t.name for t in (await client.list_tools()).tools]
        await call(client, "scene_new")
        await call(client, "execute_scene_plan", {"plan": PLAN})
        out["run"], _ = await call(client, "run_script", {"script": SCRIPT})
        out["uvs"], out["uv_images"] = await call(
            client, "inspect_uvs", {"object": "Crate", "image": "overlaps", "size": 256}
        )
        out["tex"], out["tex_images"] = await call(
            client, "inspect_texture", {"texture": "Crate color", "preview_size": 128}
        )
        out["layout_tex"], out["layout_images"] = await call(
            client, "inspect_uvs", {"object": "Crate", "image": "texture",
                                    "texture": "Crate color", "size": 128},
        )  # fmt: skip
        bad = {
            "format": "aether3d.texture-script",
            "version": 1,
            "operations": [
                {"op": "bind", "object": "Ball", "slot": "base_color",
                 "texture": str(tmp_path / "secret.png")},
            ],
        }  # fmt: skip
        out["file"], _ = await call(client, "run_script", {"script": bad}, error=True)
        out["missing"], _ = await call(client, "inspect_uvs", {"object": "Nobody"}, error=True)
        out["history"], _ = await call(client, "history", {"action": "list"})
        out["topic"], _ = await call(client, "capabilities", {"topic": "texturing"})
        guide = await client.read_resource("aether3d://texturing/guide")
        out["guide"] = guide.contents[0].text
        schema = await client.read_resource("aether3d://schema/texture-script")
        out["schema"] = json.loads(schema.contents[0].text)
        return out

    out = session(tmp_path, body)
    assert {"inspect_uvs", "inspect_texture"} <= set(out["tools"])
    assert out["run"]["format"] == "aether3d.texture-script"
    assert [r["op"] for r in out["run"]["operations"]] == [
        "unwrap",
        "add_texture",
        "bind",
        "stroke",
    ]
    assert out["uvs"]["flipped_triangles"] == 0 and out["uvs"]["overlapping_triangles"] == 0
    assert out["uv_images"] and out["uv_images"][0].startswith(b"\x89PNG")
    assert out["layout_images"][0].startswith(b"\x89PNG")
    assert out["tex"]["texture"]["name"] == "Crate color"
    assert out["tex"]["users"] == [{"material": out["tex"]["users"][0]["material"],
                                    "slot": "base_color"}]  # fmt: skip
    assert out["tex_images"][0].startswith(b"\x89PNG")
    assert out["file"]["error"]["code"] == "TEXTURE_NOT_FOUND"
    assert out["file"]["error"]["location"] == "operations[0].texture"
    assert out["missing"]["error"]["code"] == "NOT_FOUND"
    labels = [s["label"] for s in out["history"]["steps"]]
    assert labels[-1] == "Paint the crate" and labels.count("Paint the crate") == 1
    assert "stroke" in out["topic"]["operations"] and "normal" in out["topic"]["bake_maps"]
    assert out["guide"].startswith("# Aether3D texturing")
    jsonschema.Draft202012Validator.check_schema(out["schema"])
    assert not (tmp_path / "secret.png").exists()
