"""The animation MCP surface: animation scripts through run_script, inspect_animation
(summary, exact curve reports, contacts, a graph image), renders at a time, glTF export
with animation, and the guide and schema resources -- against a real server process."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp")

from test_mcp import call, session  # noqa: E402

SCRIPT = {
    "format": "aether3d.animation-script",
    "version": 1,
    "label": "Bounce",
    "operations": [
        {"op": "timeline", "fps": 24, "start": 0, "end": 2},
        {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
         "keys": [[0, 1.5], [0.5, 0.25], [1, 1.5]], "extrapolation": "cycle"},
        {"op": "set_channel", "target": "Ball", "path": "transform.position.x",
         "keys": [[0, -1], [2, 1]], "interpolation": "linear"},
    ],
}  # fmt: skip
PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "meshes": {"sphere": {"primitive": "sphere"}, "ground": {"primitive": "plane"}},
    "objects": {
        "floor": {"mesh": "@ground", "name": "Floor", "scale": [8, 8, 1]},
        "ball": {"mesh": "@sphere", "name": "Ball", "position": [0, 0, 1.5], "scale": 0.5},
        "cam": {"camera": {}, "name": "Cam", "position": [0, -7, 2], "look_at": [0, 0, 1],
                "active": True},
    },
}  # fmt: skip


def test_animate_inspect_render_and_export(tmp_path):
    (tmp_path / "out").mkdir()

    async def body(client):
        out = {}
        await call(client, "scene_new")
        await call(client, "execute_scene_plan", {"plan": PLAN})
        out["run"] = await call(client, "run_script", {"script": SCRIPT})
        out["summary"] = await call(client, "inspect_animation")
        out["curve"] = await call(client, "inspect_animation", {
            "what": "curve", "target": "Ball", "path": "transform.position.z"})  # fmt: skip
        r = await client.call_tool("inspect_animation", {
            "what": "graph", "target": "Ball", "path": "transform.position.z"})  # fmt: skip
        out["graph"] = [c for c in r.content if c.type == "image"]
        a = await client.call_tool("render_preview", {"time": 0.0, "width": 96, "height": 64})
        b = await client.call_tool("render_preview", {"time": 0.5, "width": 96, "height": 64})
        out["frames"] = [
            [c for c in a.content if c.type == "image"][0].data,
            [c for c in b.content if c.type == "image"][0].data,
        ]
        out["export"] = await call(client, "export_scene", {
            "path": "out/bounce.glb", "animation": "clips"})  # fmt: skip
        out["bad"] = await call(client, "run_script", {"script": {**SCRIPT, "operations": [
            {"op": "keyframe", "target": "Nope", "path": "visible", "time": 0,
             "value": True}]}}, error=True)  # fmt: skip
        out["after"] = await call(client, "inspect_animation")
        guide = await client.read_resource("aether3d://animation/guide")
        schema = await client.read_resource("aether3d://schema/animation-script")
        out["guide"] = guide.contents[0].text
        out["schema"] = json.loads(schema.contents[0].text)
        return out

    out = session(tmp_path, body)
    assert out["run"]["format"] == "aether3d.animation-script"
    clips = out["summary"]["clips"]
    assert clips == [{"name": "Ball Action", "channels": 2, "keys": 5, "range": [0.0, 2.0]}]
    assert out["curve"]["keys"] == 3 and out["curve"]["discontinuities"] == []
    assert len(out["graph"]) == 1 and out["graph"][0].mime_type == "image/png"
    assert out["frames"][0] != out["frames"][1]  # the ball moved
    assert out["export"]["animations"][0]["name"] == "Ball Action"
    assert (tmp_path / "out" / "bounce.glb").exists()
    assert out["bad"]["error"]["code"] == "TARGET_MISSING"
    assert out["after"]["clips"] == clips  # (the failed script changed nothing)
    assert "aether3d.animation-script" in out["guide"]
    assert out["schema"]["$id"] == "aether3d://schema/animation-script"
