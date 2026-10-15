"""The MCP server, tested as a client uses it: ``aether3d mcp serve`` in a subprocess,
spoken to over stdio by the official MCP SDK client (both protocol eras) and by raw
JSON-RPC (to check that stdout carries protocol messages only).

These tests never touch a GPU (renders use ``backend: "cpu"``): the portable CI jobs run
them; ``test_gpu_preview`` takes the ``gpu`` fixture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
jsonschema = pytest.importorskip("jsonschema")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

BASICS = json.loads(
    (Path(__file__).parents[1] / "src/aether3d/plans/examples/basics.json").read_text()
)


def _params(workspace: Path, *extra: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "aether3d", "mcp", "serve", "--workspace", str(workspace), *extra],
        env=dict(os.environ),  # PYTHONPATH and the test switches reach the server
    )


def session(workspace: Path, body, *extra: str, mode: str = "legacy"):
    """Run ``await body(client)`` against a fresh server process."""

    async def main():
        async with Client(_params(workspace, *extra), mode=mode) as client:
            return await body(client)

    return anyio.run(main)


async def call(client, name: str, arguments: dict | None = None, *, error: bool = False) -> dict:
    result = await client.call_tool(name, arguments or {})
    assert result.is_error == error, result.structured_content
    text = [c for c in result.content if c.type == "text"]
    assert json.loads(text[-1].text) == result.structured_content  # the same JSON, as text
    return result.structured_content


# --------------------------------------------------------------------------- #
# Discovery and the protocol
# --------------------------------------------------------------------------- #


def test_tools_are_discoverable_with_strict_schemas(tmp_path):
    async def body(client):
        assert client.server_info.name == "aether3d"
        assert "execute_scene_plan" in (client.instructions or "")
        return (await client.list_tools()).tools

    tools = session(tmp_path, body)
    names = [t.name for t in tools]
    assert names == [
        "scene_new", "scene_open", "scene_save", "scene_info", "objects_list",
        "execute_scene_plan", "check_scene_plan", "spatial_query", "render_preview",
        "render_image", "history", "export_scene", "run_script", "diagnostics", "capabilities",
    ]  # fmt: skip
    for tool in tools:
        schema = tool.input_schema
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert "$ref" not in json.dumps(schema)  # self-contained for every client
        assert len(tool.description) > 80 and tool.annotations.open_world_hint is False
    by_name = {t.name: t for t in tools}
    assert by_name["spatial_query"].annotations.read_only_hint
    assert not by_name["execute_scene_plan"].annotations.read_only_hint
    assert by_name["scene_save"].annotations.destructive_hint
    size = len(json.dumps([t.model_dump(mode="json", by_alias=True) for t in tools]))
    assert size < 40_000  # the tool list stays small in a model's context


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_both_protocol_eras(tmp_path, mode):
    async def body(client):
        info = await call(client, "capabilities")
        return client.protocol_version, info

    version, info = session(tmp_path, body, mode=mode)
    assert version == ("2025-11-25" if mode == "legacy" else "2026-07-28")
    assert info["aether3d"] and info["workspace"] == "." and info["scene"] == {"open": False}


def _rpc(id_: int, method: str, params: dict | None = None) -> dict:
    message: dict = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _tool(id_: int, name: str, arguments: dict) -> dict:
    return _rpc(id_, "tools/call", {"name": name, "arguments": arguments})


def test_stdout_carries_only_protocol_messages(tmp_path):
    """Raw JSON-RPC over the pipes: every stdout line is one JSON-RPC message."""
    initialize = _rpc(
        1,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "raw", "version": "0"},
        },
    )
    preview = {"backend": "cpu", "width": 64, "height": 36}
    requests = [
        _rpc(2, "tools/list"),
        _tool(3, "scene_new", {}),
        _tool(4, "execute_scene_plan", {"plan": BASICS}),
        _tool(5, "render_preview", preview),
        _tool(6, "nope", {}),
        _rpc(7, "resources/read", {"uri": "aether3d://guide"}),
        _rpc(8, "ping"),
    ]
    command = [sys.executable, "-m", "aether3d", "mcp", "serve", "--workspace", str(tmp_path)]
    proc = subprocess.Popen(
        [*command, "--log-level", "info"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
    )
    lines: list[str] = []

    def exchange(message: dict) -> dict:
        proc.stdin.write((json.dumps(message) + "\n").encode())
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline().decode("utf-8")
            assert line, proc.stderr.read().decode()[-2000:]
            lines.append(line)
            reply = json.loads(line)  # every line is one JSON message...
            assert reply.get("jsonrpc") == "2.0"  # ...of JSON-RPC
            if reply.get("id") == message["id"]:
                return reply

    try:
        replies = {1: exchange(initialize)}
        proc.stdin.write(b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')
        for request in requests:
            replies[request["id"]] = exchange(request)
        proc.stdin.close()  # the client leaves: the server stops cleanly
        assert proc.wait(timeout=30) == 0
        stderr = proc.stderr.read().decode()
    finally:
        if proc.poll() is None:  # pragma: no cover - only if the test failed
            proc.kill()
    assert proc.stdout.read() == b""
    assert replies[1]["result"]["serverInfo"]["name"] == "aether3d"
    assert len(replies[2]["result"]["tools"]) == 15
    assert replies[4]["result"]["structuredContent"]["counts"]["objects"] == 44
    image = replies[5]["result"]["content"][0]
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    assert replies[6]["error"]["code"] == -32602  # an unknown tool is a protocol error
    assert replies[7]["result"]["contents"][0]["mimeType"] == "text/markdown"
    assert replies[8]["result"] == {}
    assert len(lines) == 8  # nothing but the replies
    assert "execute_scene_plan done" in stderr  # logs went to stderr


# --------------------------------------------------------------------------- #
# A workflow
# --------------------------------------------------------------------------- #


PILLARS = [f"pillars {k}" for k in range(8)]
CRATE_VS_PILLARS = {"query": "overlaps", "objects": ["crate"], "against": PILLARS}


def test_build_inspect_fix_undo_preview_save_export(tmp_path):
    async def body(client):
        out = {}
        await call(client, "scene_new", {"name": "Yard", "path": "yard.aether"})
        result = await call(client, "execute_scene_plan", {"plan": BASICS})
        out["plan"] = result
        revision = result["revision"]["after"]
        # a mistake: a crate inside a pillar, found by a query
        bad = {
            "format": "aether3d.scene-plan",
            "version": 1,
            "label": "Crate",
            "objects": {
                "crate": {"mesh": {"primitive": "cube", "size": 1}, "position": [6, 0, 1.5]}
            },
            "queries": [CRATE_VS_PILLARS],
        }
        result = await call(
            client, "execute_scene_plan", {"plan": bad, "expected_revision": revision}
        )
        out["overlaps"] = result["queries"][0]["result"]
        fix = {
            "format": "aether3d.scene-plan",
            "version": 1,
            "label": "Move the crate",
            "operations": [{"op": "update", "select": "crate", "set": {"translate": [0, -3, 0]}}],
            "queries": [CRATE_VS_PILLARS],
        }
        out["fixed"] = (await call(client, "execute_scene_plan", {"plan": fix}))["queries"][0][
            "result"
        ]
        out["undo"] = await call(client, "history", {"action": "undo"})
        again = await call(
            client,
            "spatial_query",
            {
                "requests": [
                    CRATE_VS_PILLARS,
                    # the same question with selectors instead of names
                    {
                        "query": "overlaps",
                        "objects": {"names": ["crate"]},
                        "against": {"name_glob": "pillars *", "expect": 8},
                    },
                    {"query": "overlaps", "objects": {"tags": ["none"], "expect": {"min": 1}}},
                ]
            },
        )
        out["after_undo"], out["by_selectors"], out["unmet"] = again["results"]
        out["redo"] = await call(client, "history", {"action": "redo"})
        out["list"] = await call(
            client,
            "objects_list",
            {"select": {"name_glob": "wall*"}, "limit": 10, "fields": ["name", "world_position"]},
        )
        out["page2"] = await call(
            client,
            "objects_list",
            {"select": {"name_glob": "wall*"}, "limit": 10, "cursor": out["list"]["next_cursor"]},
        )
        out["preview"] = await client.call_tool(
            "render_preview",
            {"backend": "cpu", "width": 160, "height": 90, "save_to": "preview.png"},
        )
        out["save"] = await call(client, "scene_save", {})
        out["export"] = await call(client, "export_scene", {"path": "yard.glb"})
        out["info"] = await call(client, "scene_info")
        return out

    out = session(tmp_path, body)
    assert out["plan"]["status"] == "applied" and out["plan"]["counts"]["objects"] == 44
    assert [{o["a"]["name"], o["b"]["name"]} for o in out["overlaps"]] == [{"crate", "pillars 0"}]
    assert out["fixed"] == []
    assert [m["label"] for m in out["undo"]["moved"]] == ["Move the crate"]
    assert len(out["after_undo"]["result"]) >= 1
    assert out["by_selectors"]["result"] == out["after_undo"]["result"]
    assert out["unmet"]["error"]["code"] == "SELECTION_MISMATCH"
    assert out["unmet"]["error"]["pointer"] == "/requests/2/objects"
    assert out["redo"]["scene"]["dirty"]
    assert out["list"]["total"] == 33 and len(out["list"]["objects"]) == 10
    assert out["page2"]["offset"] == 10 and out["page2"]["objects"][0]["name"] == "wall 10"
    preview = out["preview"]
    assert not preview.is_error and preview.content[0].type == "image"
    report = preview.structured_content
    assert (
        report["backend"] == "cpu"
        and report["device"] == "cpu"
        and report["saved"] == "preview.png"
    )
    assert out["save"]["saved"] == "yard.aether" and not out["save"]["scene"]["dirty"]
    assert out["export"]["path"] == "yard.glb" and (tmp_path / "yard.glb").stat().st_size > 1000
    assert out["info"]["counts"]["objects"] == 45 and out["info"]["history"]["steps"] == 3
    from aether3d import Scene

    assert len(Scene.load(tmp_path / "yard.aether")) == 45


def test_errors_are_tool_errors_with_pointers(tmp_path):
    async def body(client):
        out = {}
        out["no_scene"] = await call(client, "execute_scene_plan", {"plan": BASICS}, error=True)
        await call(client, "scene_new")
        bad = {
            "format": "aether3d.scene-plan",
            "version": 1,
            "objects": {"a": {"mesh": "@m", "colour": 1}},
        }
        out["plan"] = await call(client, "execute_scene_plan", {"plan": bad}, error=True)
        out["args"] = await call(client, "objects_list", {"limit": 0, "extra": 1}, error=True)
        out["check"] = await call(client, "check_scene_plan", {"plan": bad})
        out["info"] = await call(client, "scene_info")
        return out

    out = session(tmp_path, body)
    assert out["no_scene"]["error"]["code"] == "NO_SCENE"
    error = out["plan"]["error"]
    assert error["type"] == "PlanError" and len(error["issues"]) == 2
    assert {i["pointer"] for i in error["issues"]} == {"/objects/a/colour", "/objects/a/mesh"}
    assert out["args"]["error"]["code"] == "INVALID_ARGUMENTS"
    assert {i["pointer"] for i in out["args"]["error"]["issues"]} == {"/", "/limit"}
    assert not out["check"]["ok"] and len(out["check"]["issues"]) == 2
    assert out["info"]["counts"]["objects"] == 0 and out["info"]["revision"] == 0


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def test_paths_stay_in_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.aether"
    from aether3d import Scene

    Scene("Secret").save(secret)
    (workspace / "link.aether").symlink_to(secret)
    (workspace / "out").mkdir()
    (workspace / "out" / "escape").symlink_to(tmp_path)
    Scene("Mine").save(workspace / "mine.aether")

    async def body(client):
        codes = {}
        for name, args in {
            "dotdot": ("scene_open", {"path": "../secret.aether"}),
            "absolute": ("scene_open", {"path": str(secret)}),
            "link": ("scene_open", {"path": "link.aether"}),
            "suffix": ("scene_open", {"path": "mine.py"}),
            "missing": ("scene_open", {"path": "nope.aether"}),
            "device": ("scene_open", {"path": "/dev/null.aether"}),
        }.items():
            codes[name] = (await call(client, args[0], args[1], error=True))["error"]["code"]
        await call(client, "scene_new")
        for name, args in {
            "write_through_dir_link": ("scene_save", {"path": "out/escape/x.aether"}),
            "write_dotdot": ("export_scene", {"path": "../x.glb"}),
            "overwrite": ("scene_save", {"path": "mine.aether"}),
            "no_folder": ("scene_save", {"path": "missing/x.aether"}),
            "new_on_existing": ("scene_new", {"path": "mine.aether", "discard_unsaved": True}),
        }.items():
            codes[name] = (await call(client, args[0], args[1], error=True))["error"]["code"]
        replaced = await call(client, "scene_save", {"path": "mine.aether", "overwrite": True})
        return codes, replaced

    codes, replaced = session(workspace, body)
    assert codes == {
        "dotdot": "PATH_OUTSIDE_WORKSPACE",
        "absolute": "PATH_OUTSIDE_WORKSPACE",
        "link": "PATH_OUTSIDE_WORKSPACE",
        "suffix": "SUFFIX_NOT_ALLOWED",
        "missing": "NOT_FOUND",
        "device": "PATH_OUTSIDE_WORKSPACE",
        "write_through_dir_link": "PATH_OUTSIDE_WORKSPACE",
        "write_dotdot": "PATH_OUTSIDE_WORKSPACE",
        "overwrite": "FILE_EXISTS",
        "no_folder": "PATH_INVALID",
        "new_on_existing": "FILE_EXISTS",
    }
    assert replaced["saved"] == "mine.aether"
    assert len(Scene.load(secret)) == 0  # untouched


def test_projects_changed_on_disk_are_not_overwritten_and_read_only_mode(tmp_path):
    from aether3d import Scene

    Scene("Shared").save(tmp_path / "shared.aether")

    async def body(client):
        await call(client, "scene_open", {"path": "shared.aether"})
        await call(client, "execute_scene_plan", {"plan": BASICS})
        other = Scene.load(tmp_path / "shared.aether")  # someone else changes the file
        other.add_cube(name="Theirs")
        other.save(tmp_path / "shared.aether")
        refused = await call(client, "scene_save", {}, error=True)
        saved = await call(client, "scene_save", {"overwrite": True})
        return refused, saved

    refused, saved = session(tmp_path, body)
    assert refused["error"]["code"] == "PROJECT_CHANGED_ON_DISK"
    assert saved["saved"] == "shared.aether"

    async def read_only(client):
        await call(client, "scene_open", {"path": "shared.aether"})
        return await call(client, "export_scene", {"path": "x.glb"}, error=True)

    error = session(tmp_path, read_only, "--read-only")
    assert error["error"]["code"] == "READ_ONLY" and not (tmp_path / "x.glb").exists()


def test_messages_do_not_reveal_the_workspace_location(tmp_path):
    async def body(client):
        await call(client, "scene_new")
        bad = {
            "format": "aether3d.scene-plan",
            "version": 1,
            "textures": {"t": {"image": "nope.png"}},
        }
        return await call(client, "execute_scene_plan", {"plan": bad}, error=True)

    error = session(tmp_path, body)
    assert str(tmp_path) not in json.dumps(error)
    assert error["error"]["code"] == "OPERATION_FAILED"


def test_render_limits_and_path_tracing_budget(tmp_path):
    async def body(client):
        await call(client, "scene_new")
        await call(client, "execute_scene_plan", {"plan": BASICS})
        out = {}
        out["too_big"] = await call(
            client, "render_preview", {"width": 1920, "height": 1920}, error=True
        )
        out["budget"] = await call(
            client,
            "render_image",
            {"path": "big.png", "backend": "path", "width": 4096, "height": 4096, "spp": 64},
            error=True,
        )
        out["image"] = await call(
            client,
            "render_image",
            {"path": "small.png", "backend": "path", "width": 48, "height": 27, "spp": 2},
        )
        return out

    out = session(tmp_path, body)
    assert out["too_big"]["error"]["code"] == "LIMIT_EXCEEDED"
    assert out["budget"]["error"]["code"] == "LIMIT_EXCEEDED"
    assert out["image"]["backend"] == "path" and out["image"]["device"] == "cpu"
    assert (tmp_path / "small.png").exists()

    async def no_path(client):
        await call(client, "scene_new")
        return await call(client, "render_image", {"path": "p.png", "backend": "path"}, error=True)

    assert session(tmp_path, no_path, "--no-path-tracing")["error"]["code"] == "NOT_ALLOWED"


# --------------------------------------------------------------------------- #
# Long work: cancellation, progress, retries
# --------------------------------------------------------------------------- #

BIG = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "label": "Many",
    "meshes": {"b": {"primitive": "cube", "size": 0.5}},
    "objects": {"blocks": {"mesh": "@b", "grid": {"count": [300, 300]}}},
}


def test_a_cancelled_plan_rolls_back_and_the_server_goes_on(tmp_path):
    async def body(client):
        await call(client, "scene_new")
        with pytest.raises(mcp.MCPError):  # the client gives up and cancels the request
            await client.call_tool("execute_scene_plan", {"plan": BIG}, read_timeout_seconds=0.3)
        await anyio.sleep(0.2)
        return await call(client, "scene_info")

    info = session(tmp_path, body)
    assert info["counts"]["objects"] == 0 and info["revision"] == 0
    assert info["history"]["steps"] == 0


def test_progress_notifications_and_idempotent_retries(tmp_path):
    progress: list[tuple] = []

    async def on_progress(value, total, message):
        progress.append((value, total, message))

    async def body(client):
        await call(client, "scene_new")
        keyed = {**BIG, "idempotency_key": "many-1"}
        result = await client.call_tool(
            "execute_scene_plan", {"plan": keyed}, progress_callback=on_progress
        )
        again = await call(client, "execute_scene_plan", {"plan": keyed})
        return result.structured_content, again

    first, again = session(tmp_path, body)
    assert first["status"] == "applied" and first["counts"]["objects"] == 90_000
    assert again["status"] == "already_applied" and again["scene"]["objects"] == 90_000
    assert len(progress) >= 3 and progress[-1][0] <= progress[-1][1]
    assert any("created" in (m or "") for _, _, m in progress)


# --------------------------------------------------------------------------- #
# Resources, prompts, scripts, diagnostics
# --------------------------------------------------------------------------- #


def test_resources_and_prompts(tmp_path):
    async def body(client):
        out = {}
        out["before"] = [r.uri for r in (await client.list_resources()).resources]
        await call(client, "scene_new")
        await call(client, "execute_scene_plan", {"plan": BASICS})
        await call(client, "render_preview", {"backend": "cpu", "width": 32, "height": 18})
        out["after"] = [r.uri for r in (await client.list_resources()).resources]
        out["templates"] = [
            t.uri_template for t in (await client.list_resource_templates()).resource_templates
        ]
        schema = await client.read_resource("aether3d://schema/scene-plan")
        out["schema"] = json.loads(schema.contents[0].text)
        out["examples"] = {}
        for name in ("basics", "castle", "village"):
            example = await client.read_resource(f"aether3d://examples/{name}")
            out["examples"][name] = json.loads(example.contents[0].text)
        obj = await client.read_resource("aether3d://scene/objects/ground")
        out["object"] = json.loads(obj.contents[0].text)
        preview = await client.read_resource("aether3d://render/last-preview")
        out["preview"] = preview.contents[0]
        out["prompts"] = [p.name for p in (await client.list_prompts()).prompts]
        prompt = await client.get_prompt("from_example", {"example": "basics"})
        out["prompt"] = prompt.messages[0].content.text
        with pytest.raises(mcp.MCPError):
            await client.get_prompt("build_scene", {})
        return out

    out = session(tmp_path, body)
    assert out["before"][:2] == ["aether3d://guide", "aether3d://schema/scene-plan"]
    assert (
        "aether3d://scene/summary" in out["after"]
        and "aether3d://render/last-preview" in out["after"]
    )
    assert "aether3d://scene/objects/{ref}" in out["templates"]
    assert out["schema"]["$id"] == "urn:aether3d:schema:scene-plan:1"
    assert out["examples"]["basics"] == BASICS
    assert {plan["label"] for plan in out["examples"].values()} == {
        BASICS["label"],
        "Minecraft castle",
        "Village",
    }
    assert all(f"aether3d://examples/{name}" in out["before"] for name in out["examples"])
    assert out["object"]["name"] == "ground" and out["object"]["kind"] == "mesh"
    assert out["preview"].mime_type == "image/png" and len(out["preview"].blob) > 100
    assert out["prompts"] == ["build_scene", "inspect_and_fix", "from_example"]
    assert "aether3d.scene-plan" in out["prompt"]


def test_scripts_and_diagnostics(tmp_path):
    async def body(client):
        await call(client, "scene_new")
        await call(
            client,
            "execute_scene_plan",
            {
                "plan": {
                    "format": "aether3d.scene-plan",
                    "version": 1,
                    "objects": {"Block": {"mesh": {"primitive": "cube"}}},
                }
            },
        )
        out = {}
        top = {"op": "select", "kind": "faces", "normal": [[0, 0, 1], 1], "expect": 1, "as": "top"}
        extrude = {"op": "extrude_faces", "faces": {"selection": "top"}, "distance": 1}
        out["edit"] = await call(client, "run_script", {"script": {
            "format": "aether3d.edit-script", "version": 1,
            "edits": [{"object": "Block", "operations": [top, extrude]}],
        }})  # fmt: skip
        out["procedural"] = await call(client, "run_script", {"script": {
            "format": "aether3d.procedural-script", "version": 1,
            "operations": [{"op": "add_parameter", "name": "h", "value": 2.0}],
        }})  # fmt: skip
        out["scene"] = await call(client, "run_script", {"script": {
            "format": "aether3d.scene-script", "version": 1,
            "operations": [{"op": "set_world", "background": "#203040"}],
        }})  # fmt: skip
        out["bad"] = await call(
            client, "run_script", {"script": {"format": "aether3d.other"}}, error=True
        )
        out["diagnostics"] = await call(
            client, "diagnostics", {"checks": ["geometry", "materials", "history"]}
        )
        out["history"] = await call(client, "history", {"action": "list"})
        return out

    out = session(tmp_path, body)
    assert out["edit"]["format"] == "aether3d.edit-script"
    assert out["scene"]["scene"]["revision"] == 4
    assert out["bad"]["error"]["code"] == "INVALID_ARGUMENTS"  # the schema's enum
    assert out["diagnostics"]["history"]["ok"] is True
    assert [s["label"] for s in out["history"]["steps"]][0] == "Scene plan"


def test_server_opens_a_project_at_start(tmp_path):
    from aether3d import Scene

    scene = Scene("Start")
    scene.add_cube(name="Cube")
    scene.save(tmp_path / "start.aether", history=True)

    async def body(client):
        return await call(client, "scene_info")

    info = session(tmp_path, body, "--project", "start.aether")
    assert info["project"] == "start.aether" and info["counts"]["objects"] == 1
    assert info["history"]["steps"] == 1  # its history file came along


def test_gpu_preview(tmp_path, gpu):
    async def body(client):
        await call(client, "scene_new")
        await call(client, "execute_scene_plan", {"plan": BASICS})
        return await client.call_tool(
            "render_preview", {"backend": "gpu", "width": 320, "height": 180}
        )

    adapter = os.environ.get("AETHER3D_TEST_GPU_ADAPTER")
    result = session(tmp_path, body, *(("--gpu-adapter", adapter) if adapter else ()))
    report = result.structured_content
    assert report["backend"] == "gpu" and report["device"] == "gpu"
    if adapter:
        assert adapter.lower() in report["adapter"]["name"].lower()
    assert report["adapter"]["name"] and report["draw_calls"] == 3  # 3 meshes, instanced
    assert result.content[0].type == "image"


def test_errors_never_show_paths_of_this_machine(tmp_path):
    from aether3d import PlanError, RenderError
    from aether3d.mcp.session import ServerConfig, Session
    from aether3d.mcp.tools import error_payload

    session = Session(ServerConfig(workspace=tmp_path))
    inside = tmp_path / "scenes" / "castle.aether"
    home = Path.home() / "private" / "notes.txt"
    # an engine error: workspace paths become relative, the home folder becomes ~
    engine = error_payload(RenderError("X", f"cannot use {inside} nor {home}"), session)
    message = engine["error"]["message"]  # paths keep the platform's separators
    assert str(Path("scenes", "castle.aether")) in message and str(tmp_path) not in message
    assert str(Path.home()) not in message and str(Path("~", "private", "notes.txt")) in message
    # a bug: no absolute path at all, but the exception type stays
    bug = error_payload(OSError(f"[Errno 2] No such file: '{home}' (from /usr/lib/x.py)"), session)
    assert bug["error"]["code"] == "INTERNAL_ERROR"
    assert bug["error"]["message"] == "OSError: [Errno 2] No such file: '<path>' (from <path>)"
    # plan errors keep their JSON pointers
    plan = error_payload(PlanError("INVALID_VALUE", "bad", "/objects/towers/grid/count"), session)
    assert plan["error"]["pointer"] == "/objects/towers/grid/count"


def test_long_answers_are_cut_short_and_large_replies_refused(tmp_path):
    crowd = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "objects": {
            "crates": {
                "mesh": {"primitive": "cube"},
                "repeat": 150,
                "position": [{"expr": "i * 0.5"}, 0, 0.5],
            }
        },
        "queries": [{"query": "overlaps", "objects": "@crates"}],
    }

    async def body(client):
        out = {}
        await call(client, "scene_new")
        out["built"] = await call(client, "execute_scene_plan", {"plan": crowd, "max_results": 10})
        # a change is reported in full, whatever its size
        out["dry"] = await call(
            client,
            "execute_scene_plan",
            {
                "plan": {**crowd, "options": {"on_name_conflict": "rename"}},
                "mode": "dry_run",
                "max_results": 10000,
            },
        )
        out["answers"] = await call(
            client,
            "spatial_query",
            {"requests": [{"query": "overlaps"}, {"query": "describe"}], "max_results": 5},
        )
        out["too_large"] = await call(
            client,
            "spatial_query",
            {"request": {"query": "overlaps"}, "max_results": 10000},
            error=True,
        )
        out["list"] = await call(
            client,
            "objects_list",
            {
                "limit": 500,
                "fields": ["id", "name", "position", "rotation_euler", "world_position", "bounds"],
            },
            error=True,
        )
        return out

    out = session(tmp_path, body, "--max-reply-kib", "8")
    built = out["built"]["queries"][0]
    pairs = built["total"]
    assert built["truncated"] and len(built["result"]) == 10 and pairs > 100
    dry = out["dry"]["queries"][0]
    assert "truncated" not in dry and len(dry["result"]) >= pairs
    assert len(json.dumps(out["dry"])) > 8 * 1024
    overlaps, described = out["answers"]["results"]
    assert len(overlaps["result"]) == 5 and overlaps["total"] == pairs
    assert len(described["result"]["objects"]) == 5 and described["total"] == 150
    for refused in (out["too_large"], out["list"]):
        assert refused["error"]["code"] == "RESULT_TOO_LARGE"
        assert refused["error"]["details"]["limit"] == 8 * 1024
    assert "max_results" in out["too_large"]["error"]["hint"]
    assert "limit" in out["list"]["error"]["hint"]


def test_unicode_names_round_trip(tmp_path):
    name = "Château ★ 城/塔"
    plan = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "label": "Ünïcode",
        "materials": {"stone": {"name": "Pierre ✓", "base_color": "#8a8a8a"}},
        "objects": {"keep": {"mesh": {"primitive": "cube"}, "name": name, "material": "@stone"}},
    }

    async def body(client):
        out = {}
        await call(client, "scene_new", {"name": "Été"})
        out["plan"] = await call(client, "execute_scene_plan", {"plan": plan})
        out["list"] = await call(
            client,
            "objects_list",
            {"select": {"name_glob": "Château*"}, "fields": ["name", "material"]},
        )
        result = await client.call_tool(
            "spatial_query", {"request": {"query": "bounds", "object": name}}
        )
        out["text"] = result.content[-1].text
        out["saved"] = await call(client, "scene_save", {"path": "été.aether"})
        return out

    out = session(tmp_path, body)
    assert out["plan"]["label"] == "Ünïcode"
    listed = out["list"]["objects"]
    assert [o["name"] for o in listed] == [name]
    assert "Pierre ✓" in json.dumps(listed[0]["material"], ensure_ascii=False)
    assert name in out["text"]  # UTF-8 text, not \\u escapes
    assert out["saved"]["saved"] == "été.aether"
    from aether3d import Scene

    assert Scene.load(tmp_path / "été.aether").get_by_name(name).material.name == "Pierre ✓"


@pytest.mark.parametrize("kind", ["escaped", "linked"])
def test_workspace_paths_are_hidden_however_they_are_spelled(tmp_path, kind):
    """JSON escapes a Windows path's backslashes, and macOS reaches temporary folders
    through a link (/var -> /private/var): the workspace is hidden either way."""
    from aether3d import RenderError
    from aether3d.mcp.session import ServerConfig, Session
    from aether3d.mcp.tools import error_payload

    if kind == "escaped":
        if os.name != "posix":
            pytest.skip("such a folder name needs a POSIX file system")
        root = tmp_path / 'back\\slash "quoted"'
        root.mkdir()
    else:
        real = tmp_path / "real"
        real.mkdir()
        root = tmp_path / "link"
        try:
            root.symlink_to(real, target_is_directory=True)
        except OSError:  # pragma: no cover - Windows without the privilege
            pytest.skip("cannot create a symbolic link here")
    session = Session(ServerConfig(workspace=root))
    payload = error_payload(
        RenderError("X", f"cannot read {root / 'scenes' / 'a.aether'}"), session
    )
    message = payload["error"]["message"]
    assert message == "cannot read scenes/a.aether" or message.endswith("scenes\\a.aether")
    assert str(root) not in json.dumps(payload) and str(root.resolve()) not in json.dumps(payload)
