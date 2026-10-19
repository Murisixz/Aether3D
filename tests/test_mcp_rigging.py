"""The rigging MCP surface (Stage 18): an agent opens an unrigged character, measures it,
rigs and binds it with a script, animates and poses it, diagnoses, fixes, previews and
exports it -- every step through the server, against a real server process."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp")

from test_mcp import call, session  # noqa: E402

from aether3d import Scene  # noqa: E402
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface  # noqa: E402


def script(*ops, label="Rig"):
    return {"format": "aether3d.animation-script", "version": 1, "label": label,
            "operations": list(ops)}  # fmt: skip


def test_an_agent_rigs_animates_verifies_and_exports(tmp_path):
    s = Scene("Hero")
    s.add_object("Body", mesh=implicit_surface(humanoid_shapes(), 0.02))
    s.save(tmp_path / "hero.aether")
    (tmp_path / "out").mkdir()

    async def body(client):
        out = {}
        await call(client, "scene_open", {"path": "hero.aether"})
        out["landmarks"] = await call(client, "inspect_rig", {"what": "landmarks",
                                                              "meshes": ["Body"]})  # fmt: skip
        out["rig"] = await call(client, "run_script", {"script": script(
            {"op": "autorig", "meshes": ["Body"], "name": "Hero Rig"},
            {"op": "generate", "armature": "Hero Rig", "motion": "walk", "forward": "-y"},
            {"op": "ik_fk", "armature": "Hero Rig", "limb": "leg_ik.L", "mode": "ik",
             "time": 0.4},
            {"op": "pose_character", "armature": "Hero Rig", "time": 1.5,
             "commands": [{"do": "look_at", "target": [0.5, -3.0, 1.6]}]},
        )})  # fmt: skip
        out["summary"] = await call(client, "inspect_rig", {"target": "Hero Rig"})
        out["weights"] = await call(client, "inspect_rig", {"what": "weights", "target": "Body"})
        out["diagnose"] = await call(client, "inspect_rig", {
            "what": "diagnose", "target": "Hero Rig", "pose": "elbow_bend_120.L"})  # fmt: skip
        out["fixes"] = await call(client, "inspect_rig", {"what": "fixes", "target": "Hero Rig"})
        out["apply"] = await call(client, "run_script", {"script": script(
            {"op": "apply_rig_fix", "armature": "Hero Rig", "fixes": "safe"})})  # fmt: skip
        r = await client.call_tool("inspect_rig", {"what": "preview", "target": "Hero Rig",
                                                   "mode": "skeleton"})  # fmt: skip
        out["preview"] = [c for c in r.content if c.type == "image"]
        out["export"] = await call(client, "export_scene", {"path": "out/hero.glb",
                                                            "animation": "bake"})  # fmt: skip
        out["bad"] = await call(client, "run_script", {"script": script(
            {"op": "edit_rig", "armature": "Hero Rig",
             "operations": [{"op": "reparent_joint", "joint": "Hips", "parent": "Head"}]})},
            error=True)  # fmt: skip
        out["undo"] = await call(client, "history", {"action": "undo"})
        guide = await client.read_resource("aether3d://rigging/guide")
        out["guide"] = guide.contents[0].text
        schema = await client.read_resource("aether3d://schema/animation-script")
        out["schema"] = json.loads(schema.contents[0].text)
        return out

    out = session(tmp_path, body)
    lm = out["landmarks"]["landmarks"]
    assert {"hip.L", "knee.L", "shoulder.R", "wrist.R", "neck"} <= set(lm)
    ops = out["rig"]["operations"]
    assert ops[0]["result"]["armature"] == "Hero Rig"
    assert ops[2]["result"]["mismatch"] < 1e-9
    summary = out["summary"]
    assert summary["template"] == "humanoid" and summary["skins"][0]["mesh"] == "Body"
    assert any(c["kind"] == "ik_target" for c in summary["controls"])
    assert out["weights"]["valid"]
    assert "elbow_bend_120.L" in out["diagnose"]["poses"]
    assert isinstance(out["fixes"]["proposals"], list)
    assert len(out["preview"]) == 1 and out["preview"][0].mime_type == "image/png"
    assert out["export"]["path"].endswith("hero.glb")
    assert not any(d["lossy"] for d in out["export"]["diagnostics"])  # (glTF's own skinning)
    assert out["bad"]["error"]["code"] in ("SKELETON_CYCLE", "INVALID_SCRIPT")
    assert "Rigid-part characters need no bones" in out["guide"]
    ops_schema = out["schema"]["properties"]["operations"]["items"]["oneOf"]
    names = {o["properties"]["op"]["const"] for o in ops_schema}
    assert {"autorig", "edit_weights", "pose_character", "apply_rig_fix", "diagnose_rig"} <= names
