"""Worlds through the MCP server (a real server process): world scripts, inspect_world,
the guide and the schema."""

import json

import pytest

pytest.importorskip("mcp")
jsonschema = pytest.importorskip("jsonschema")

from test_mcp import call, session  # noqa: E402

SCRIPT = {
    "format": "aether3d.world-script",
    "operations": [
        {
            "op": "create",
            "name": "Isle",
            "preset": "temperate",
            "seed": 2,
            "size": 1024.0,
            "overrides": {"hydrology": {"cell": 32.0}},
        },
        {"op": "place_structure", "item": {"id": "hut", "kind": "house", "at": [10, 10]}},
    ],
}


def test_worlds_over_mcp(tmp_path):
    async def body(client):
        out = {}
        await call(client, "scene_new", {})
        out["run"] = await call(client, "run_script", {"script": SCRIPT})
        out["summary"] = await call(client, "inspect_world", {"what": "summary"})
        out["sample"] = await call(
            client,
            "inspect_world",
            {"what": "sample", "world": "Isle", "points": [[0, 0], [10, 10]]},
        )
        out["outside"] = await call(
            client,
            "inspect_world",
            {"what": "sample", "world": "Isle", "points": [[1e6, 0]]},
            error=True,
        )
        out["bad"] = await call(
            client,
            "run_script",
            {"script": {"format": "aether3d.world-script", "operations": [{"op": "explode"}]}},
            error=True,
        )
        guide = await client.read_resource("aether3d://worlds/guide")
        out["guide"] = guide.contents[0].text
        schema = await client.read_resource("aether3d://schema/world-script")
        out["schema"] = json.loads(schema.contents[0].text)
        out["preview"] = await call(
            client, "render_preview", {"backend": "cpu", "width": 64, "height": 36, "frame": True}
        )
        return out

    out = session(tmp_path, body)
    assert out["run"]["format"] == "aether3d.world-script"
    assert [o["op"] for o in out["run"]["operations"]] == ["create", "place_structure"]
    worlds = out["summary"]["worlds"]
    assert worlds[0]["name"] == "Isle" and worlds[0]["structures"] == 1
    assert "alpine_valley" in out["summary"]["presets"]
    assert len(out["sample"]["height"]) == 2
    assert out["outside"]["error"]["code"] == "OUTSIDE_WORLD"
    assert out["bad"]["error"]["code"] == "INVALID_SCRIPT"
    assert "world-script" in out["guide"]
    jsonschema.Draft202012Validator.check_schema(out["schema"])
    jsonschema.validate(SCRIPT, out["schema"])
