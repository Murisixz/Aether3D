"""The visual-feedback MCP tools, through a real server process (Stage 12).

The server runs with ``--feedback-backend cpu``: these tests never touch a GPU (the
portable CI jobs run them). The vision evaluations sent here are FIXTURES written by the
test -- they exercise the protocol and the grounding, not any model's vision.
"""

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
    "meshes": {"block": {"primitive": "cube", "size": 1}},
    "objects": {
        "floor": {"mesh": {"primitive": "plane", "size": 10}, "color": "#777777"},
        "table": {"mesh": "@block", "position": [0, 0, 0.5]},
        "vase": {
            "mesh": {"primitive": "cylinder", "radius": 0.1, "depth": 0.3},
            "position": [0, 0, 1.4],
        },
        "sun": {"light": {"type": "directional", "intensity": 2}, "direction": [-0.4, 0.5, -1]},
        "cam": {
            "camera": {"fov": 40},
            "position": [4, -5, 3],
            "look_at": [0, 0, 0.6],
            "active": True,
        },
    },
}
EXPECT = {
    "format": "aether3d.expectations",
    "version": 1,
    "views": {"main": {"camera": "cam"}},
    "checks": [
        {"id": "vase-on-table", "check": "rests_on", "objects": "vase", "support": "table"},
        {"id": "seen", "check": "visible", "objects": ["table", "vase"]},
    ],
}


def run(workspace: Path, body, *extra: str):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "aether3d",
            "mcp",
            "serve",
            "--workspace",
            str(workspace),
            "--feedback-backend",
            "cpu",
            *extra,
        ],
        env=dict(os.environ),
    )

    async def main():
        async with Client(params) as client:
            return await body(client)

    return anyio.run(main)


async def call(client, name, arguments=None, *, error=False):
    result = await client.call_tool(name, arguments or {})
    assert result.is_error == error, result.structured_content
    text = [c for c in result.content if c.type == "text"]
    assert json.loads(text[-1].text) == result.structured_content
    images = [base64.b64decode(c.data) for c in result.content if c.type == "image"]
    return result.structured_content, images


async def build(client):
    await call(client, "scene_new", {"path": "room.aether"})
    await call(client, "execute_scene_plan", {"plan": PLAN})


def test_observe_returns_images_markers_and_crops(tmp_path):
    async def body(client):
        await build(client)
        data, images = await call(
            client,
            "observe_scene",
            {
                "images": ["image", "annotated", "ids"],
                "width": 200,
                "height": 120,
                "focus": ["vase"],
                "occlusion": True,
                "crops": [{"rect": [0.25, 0.25, 0.75, 0.75], "scale": 2}],
                "views": [{}, {"preset": "top"}],
            },
        )
        return data, images

    data, images = run(tmp_path, body)
    assert len(images) == len(data["images"]) == 8  # 2 views x (3 kinds + 1 crop)
    assert all(png.startswith(b"\x89PNG") for png in images)
    kinds = [i["kind"] for i in data["images"]]
    assert kinds == ["image", "annotated", "ids", "crop"] * 2
    first = data["observations"][0]
    assert first["renderer"]["backend"] == "cpu" and first["renderer"]["device"] == "cpu"
    assert [m["name"] for m in first["markers"]] == ["vase"]  # the focus objects are marked
    vase = next(o for o in first["objects"] if o["name"] == "vase")
    assert vase["geometry"]["frustum"] == "inside" and vase["geometry"]["occlusion"]["computed"]
    assert data["observations"][1]["view"]["kind"] == "preset"


def test_the_whole_workflow_over_mcp(tmp_path):
    async def body(client):
        await build(client)
        obs, _ = await call(client, "observe_scene", {"images": ["annotated"]})
        oid = obs["observations"][0]["id"]
        sha = obs["observations"][0]["image"]["sha256"]
        vase_id = next(o["id"] for o in obs["observations"][0]["objects"] if o["name"] == "vase")
        region = next(
            o["screen_bounds"] for o in obs["observations"][0]["objects"] if o["name"] == "vase"
        )
        fixture = {  # a FIXTURE evaluation (no model looked at the image)
            "format": "aether3d.vision-evaluation",
            "version": 1,
            "observation": oid,
            "image_sha256": sha,
            "provenance": {"provider": "test-fixture", "image_input": False},
            "findings": [
                {
                    "category": "floating",
                    "severity": "error",
                    "confidence": 0.9,
                    "summary": "the vase hovers",
                    "region": region,
                    "objects": [vase_id],
                }
            ],
        }
        diag, images = await call(
            client,
            "diagnose_scene",
            {"expectations": EXPECT, "evaluations": [fixture], "include_image": True},
        )
        status, _ = await call(client, "scene_info")
        proposals, preview_images = await call(client, "propose_corrections", {"preview": True})
        after_preview, _ = await call(client, "scene_info")
        stale, _ = await call(
            client, "apply_corrections", {"expected_revision": diag["revision"] - 1}, error=True
        )
        applied, _ = await call(
            client,
            "apply_corrections",
            {"expected_revision": diag["revision"], "include_image": True},
        )
        undo, _ = await call(client, "history", {"action": "undo"})
        loop, loop_images = await call(client, "feedback_loop", {"budget": {"max_iterations": 3}})
        return (
            diag,
            images,
            status,
            proposals,
            preview_images,
            after_preview,
            stale,
            applied,
            loop,
            loop_images,
        )

    (
        diag,
        images,
        status,
        proposals,
        preview_images,
        after_preview,
        stale,
        applied,
        loop,
        loop_images,
    ) = run(tmp_path, body)
    codes = {f["code"]: f for f in diag["findings"]}
    assert codes["FLOATING"]["basis"] == "geometry"
    assert codes["VISION_FLOATING"]["basis"] == "agent"  # no image input declared
    assert codes["VISION_FLOATING"]["evidence"]["grounding"] == "grounded"
    assert len(images) == 1 and diag["image"]["kind"] == "annotated"
    assert not diag["accepted"] and diag["summary"]["errors"] == 1
    (p,) = [x for x in proposals["proposals"] if x["safety"] == "safe"]
    assert p["compiled"]["plan"]["format"] == "aether3d.scene-plan"
    assert proposals["preview"]["comparison"]["resolved"] == p["findings"]
    assert len(preview_images) == 1
    assert after_preview["revision"] == status["revision"]  # the preview changed nothing
    assert stale["error"]["code"] == "REVISION_MISMATCH"
    assert applied["status"] == "applied" and applied["verification"]["decision"] == "kept"
    assert applied["image"]["kind"] == "annotated"
    assert loop["stop_reason"] == "accepted" and loop["checkpoint"]
    assert len(loop_images) == 1


def test_agent_actions_approvals_and_refusals(tmp_path):
    async def body(client):
        await build(client)
        await call(client, "diagnose_scene", {"expectations": EXPECT})
        bad, _ = await call(
            client,
            "apply_corrections",
            {"actions": [{"action": "delete", "object": "vase"}]},
            error=True,
        )
        schema_error, _ = await call(
            client, "apply_corrections", {"actions": [{"object": "vase"}]}, error=True
        )
        far, _ = await call(
            client,
            "apply_corrections",
            {"actions": [{"action": "translate", "object": "vase", "offset": [0, 0, 500]}]},
            error=True,
        )
        ok, _ = await call(
            client,
            "apply_corrections",
            {"actions": [{"action": "set_exposure", "exposure": 0.25}], "label": "brighter"},
        )
        return bad, schema_error, far, ok

    bad, schema_error, far, ok = run(tmp_path, body)
    assert bad["error"]["code"] == "INVALID_VALUE" and bad["error"]["issues"]
    assert schema_error["error"]["code"] == "INVALID_ARGUMENTS"
    assert far["error"]["code"] == "NOT_ALLOWED"
    assert ok["status"] == "applied" and ok["step"]["label"] == "brighter"


def test_feedback_resources_and_schemas(tmp_path):
    async def body(client):
        await build(client)
        obs, _ = await call(client, "observe_scene", {"width": 64, "height": 36})
        oid = obs["observations"][0]["id"]
        listed = (await client.list_resources()).resources
        templates = (await client.list_resource_templates()).resource_templates
        guide = await client.read_resource("aether3d://feedback/guide")
        schemas = {
            name: json.loads(
                (await client.read_resource(f"aether3d://schema/{name}")).contents[0].text
            )
            for name in ("expectations", "vision-evaluation", "corrections")
        }
        report = await client.read_resource(f"aether3d://observations/{oid}")
        png = await client.read_resource(f"aether3d://observations/{oid}/annotated")
        caps, _ = await call(client, "capabilities", {"topic": "feedback"})
        return listed, templates, guide, schemas, report, png, caps

    listed, templates, guide, schemas, report, png, caps = run(tmp_path, body)
    uris = {str(r.uri) for r in listed}
    assert {"aether3d://feedback/guide", "aether3d://schema/expectations"} <= uris
    assert "aether3d://observations/{id}/{kind}" in {t.uri_template for t in templates}
    assert "## Expectations" in guide.contents[0].text
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(EXPECT, schemas["expectations"])
    assert json.loads(report.contents[0].text)["format"] == "aether3d.observation"
    assert base64.b64decode(png.contents[0].blob).startswith(b"\x89PNG")
    assert "rests_on" in caps["checks"] and "translate" in caps["actions"]


def test_limits_and_unknown_observations(tmp_path):
    async def body(client):
        await build(client)
        too_many, _ = await call(
            client,
            "observe_scene",
            {
                "views": [{}, {"preset": "top"}, {"preset": "front"}, {"preset": "iso"}],
                "images": ["image", "annotated", "ids"],
                "width": 64,
                "height": 36,
            },
            error=True,
        )
        huge, _ = await call(client, "observe_scene", {"width": 1920, "height": 1920}, error=True)
        try:
            await client.read_resource("aether3d://observations/obs-nope")
            unknown = None
        except Exception as exc:  # an MCP error, not a crash
            unknown = str(exc)
        return too_many, huge, unknown

    too_many, huge, unknown = run(tmp_path, body)
    assert too_many["error"]["code"] == "LIMIT_EXCEEDED"
    assert huge["error"]["code"] == "LIMIT_EXCEEDED"
    assert unknown and "obs-nope" in unknown and "/" not in unknown.split("obs-nope")[0][-3:]


def test_the_loop_waits_for_approval_and_resumes(tmp_path):
    expect = {
        **EXPECT,
        "checks": [{"check": "no_overlap", "objects": "vase", "against": "table"}],
    }

    async def body(client):
        await build(client)
        await call(
            client,
            "execute_scene_plan",
            {
                "plan": {
                    "format": "aether3d.scene-plan",
                    "version": 1,
                    "operations": [
                        {"op": "update", "select": "vase", "set": {"position": [0.5, 0, 0.9]}}
                    ],
                }
            },
        )
        first, _ = await call(
            client, "feedback_loop", {"expectations": expect, "include_image": False}
        )
        pending = first["pending_approval"]
        applied, _ = await call(client, "apply_corrections", {"proposals": [pending[0]["id"]]})
        resumed, _ = await call(
            client, "feedback_loop", {"resume": first["id"], "include_image": False}
        )
        return first, applied, resumed

    first, applied, resumed = run(tmp_path, body)
    assert first["stop_reason"] == "awaiting_approval" and first["pending_approval"]
    assert applied["status"] == "applied"
    assert resumed["stop_reason"] == "accepted" and resumed["id"] == first["id"]
