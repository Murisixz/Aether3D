"""The precision-modeling MCP tools: CAD scripts through run_script, inspect_cad (with a
drawing image), CAD exports through export_scene and import_cad confined to the
workspace, the cad
capabilities topic and resources -- against a real server process."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("manifold3d")

from test_mcp import call, session  # noqa: E402

DATA = Path(__file__).parent / "data"
BRACKET = json.loads((DATA / "cad" / "bracket_script.json").read_text())


def test_cad_script_inspect_export_import_round_trip(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "in.step").write_text((DATA / "step" / "occt_plate_hole.step").read_text())
    (tmp_path / "plate.dxf").write_text((DATA / "dxf" / "ezdxf_plate.dxf").read_text())

    async def body(client):
        out = {}
        await call(client, "scene_new")
        out["run"] = await call(client, "run_script", {"script": BRACKET})
        out["doc"] = await call(client, "inspect_cad")
        r = await client.call_tool("inspect_cad", {"item": "Bracket", "image": "drawing"})
        out["inspect"] = r.structured_content
        out["image"] = [c for c in r.content if c.type == "image"]
        await call(
            client,
            "run_script",
            {
                "script": {
                    "format": "aether3d.cad-script",
                    "version": 1,
                    "operations": [
                        {
                            "op": "add",
                            "kind": "drawing",
                            "name": "Top",
                            "definition": {"view": "top", "items": ["Bracket"]},
                        }
                    ],
                }
            },
        )
        out["svg"] = await call(client, "export_scene", {"path": "out/top.svg", "item": "Top"})
        out["step"] = await call(
            client, "export_scene", {"path": "out/bracket.step", "items": ["Bracket"]}
        )
        out["again"] = await call(client, "export_scene", {"path": "out/bracket.step"}, error=True)
        out["outside"] = await call(client, "export_scene", {"path": "../x.step"}, error=True)
        out["import"] = await call(client, "import_cad", {"path": "in.step", "name": "Plate"})
        out["dxf"] = await call(client, "import_cad", {"path": "plate.dxf"})
        out["plate"] = await call(client, "inspect_cad", {"item": "Plate"})
        out["failed"] = await call(
            client,
            "run_script",
            {
                "script": {
                    "format": "aether3d.cad-script",
                    "version": 1,
                    "operations": [
                        {"op": "parameter", "name": "width", "value": "90 mm"},
                        {"op": "expect", "info": "Bracket", "checks": {"bodies.main.volume": 1}},
                    ],
                }
            },
            error=True,
        )
        out["after"] = await call(client, "inspect_cad", {"item": "width"})
        out["caps"] = await call(client, "capabilities", {"topic": "cad"})
        guide = await client.read_resource("aether3d://cad/guide")
        schema = await client.read_resource("aether3d://schema/cad-script")
        out["guide"] = guide.contents[0].text
        out["schema"] = json.loads(schema.contents[0].text)
        return out

    out = session(tmp_path, body)
    assert out["run"]["format"] == "aether3d.cad-script"
    assert [i["name"] for i in out["doc"]["items"]] == ["width", "thickness", "Profile", "Bracket"]
    assert out["doc"]["items"][2]["status"] == "fully_constrained"
    faces = [f["name"] for f in out["inspect"]["faces"]["main"]]
    assert "base:end" in faces and "h.1:rev/wall.0" in faces  # hole h at its second point
    assert out["inspect"]["bodies"]["main"]["valid"]
    assert len(out["image"]) == 1 and out["image"][0].mime_type == "image/png"
    assert "the report's numbers are the authority" in out["inspect"]["image"]
    assert out["svg"]["path"] == "out/top.svg" and (tmp_path / "out" / "top.svg").exists()
    assert (tmp_path / "out" / "bracket.step").read_text().startswith("ISO-10303-21;")
    assert out["again"]["error"]["code"] == "FILE_EXISTS"
    assert out["outside"]["error"]["code"] == "PATH_OUTSIDE_WORKSPACE"
    assert out["import"]["parts"] == ["Plate"]
    assert out["dxf"]["sketch"] == "plate" and out["dxf"]["skipped"] == {"TEXT": 1, "INSERT": 1}
    assert out["plate"]["bodies"]["main"]["volume"] == pytest.approx(31497.345175425602, rel=1e-12)
    # A failed expectation rolls the whole script back: width keeps its value.
    assert out["failed"]["error"]["code"] == "EXPECTATION_FAILED"
    assert out["after"]["display"] == "100 mm"
    assert "extrude" in out["caps"]["features"] and "plan" in out["caps"]["drawing_views"]
    assert out["schema"]["$id"] == "aether3d://schema/cad-script"
    assert "aether3d.cad-script" in out["guide"]
