"""``aether3d cad run|info|export|import|schema``: atomic scripts, failed expectations
(exit 3, nothing saved), exports by suffix, imports, structured errors."""

import json
from pathlib import Path

import pytest

from aether3d import Scene
from aether3d.cli import main

pytest.importorskip("manifold3d")

DATA = Path(__file__).parent / "data"
SCRIPT = DATA / "cad" / "bracket_script.json"


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "shop.aether"
    Scene("Shop").save(path)
    return path


def test_run_info_export_import(project, tmp_path, capsys):
    code, out = run_json(capsys, "cad", "run", str(project), str(SCRIPT))
    assert code == 0 and out["ok"] and len(out["operations"]) == 12
    code, out = run_json(capsys, "cad", "info", str(project))
    assert [i["name"] for i in out["items"]] == ["width", "thickness", "Profile", "Bracket"]
    code, out = run_json(capsys, "cad", "info", str(project), "Bracket")
    assert out["bodies"]["main"]["valid"]
    step = tmp_path / "bracket.step"
    code, out = run_json(capsys, "cad", "export", str(project), str(step), "--item", "Bracket")
    assert code == 0 and step.read_text().startswith("ISO-10303-21;")
    code, out = run_json(capsys, "cad", "export", str(project), str(step))
    assert code == 1 and out["error"]["type"] == "FileExistsError"
    code, out = run_json(capsys, "cad", "import", str(project), str(step), "--name", "Copy")
    assert code == 0 and out["parts"] == ["Copy"]
    scene = Scene.load(project)
    a = scene.cad.part("Bracket").measure("main")["volume"]
    b = scene.cad.part("Copy").measure("main")["volume"]
    assert b == pytest.approx(a, rel=1e-12)
    dxf = DATA / "dxf" / "ezdxf_plate.dxf"
    code, out = run_json(capsys, "cad", "import", str(project), str(dxf), "--name", "Plate")
    assert code == 0 and out["sketch"] == "Plate" and out["circles"] == 1


def test_drawing_exports(project, tmp_path, capsys):
    script = json.loads(SCRIPT.read_text())
    script["operations"].append({"op": "add", "kind": "drawing", "name": "Front", "definition": {
        "view": "front", "items": ["Bracket"]}})  # fmt: skip
    path = tmp_path / "s.json"
    path.write_text(json.dumps(script))
    assert run_json(capsys, "cad", "run", str(project), str(path))[0] == 0
    for suffix in (".svg", ".dxf", ".png"):
        out_file = tmp_path / f"front{suffix}"
        code, out = run_json(capsys, "cad", "export", str(project), str(out_file),
                             "--item", "Front", "--dpi", "40")  # fmt: skip
        assert code == 0 and out_file.stat().st_size == out["bytes"] > 100
    assert (tmp_path / "front.png").read_bytes().startswith(b"\x89PNG")
    code, out = run_json(capsys, "cad", "export", str(project), str(tmp_path / "x.svg"),
                         "--item", "Bracket")  # fmt: skip
    assert code == 1 and "not a drawing" in out["error"]["message"]


def test_failed_expectations_save_nothing(project, tmp_path, capsys):
    run_json(capsys, "cad", "run", str(project), str(SCRIPT))
    before = project.read_bytes()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "aether3d.cad-script", "version": 1, "operations": [
        {"op": "parameter", "name": "width", "value": "1 m"},
        {"op": "expect", "info": "Bracket",
         "checks": {"bodies.main.volume": {"max": 1}}}]}))  # fmt: skip
    code, out = run_json(capsys, "cad", "run", str(project), str(bad))
    assert code == 3 and out["error"]["code"] == "EXPECTATION_FAILED" and not out["saved"]
    assert project.read_bytes() == before
    bad.write_text(json.dumps({"format": "aether3d.cad-script", "version": 1, "operations": [
        {"op": "remove", "item": "Nothing"}]}))  # fmt: skip
    code, out = run_json(capsys, "cad", "run", str(project), str(bad))
    assert code == 1 and out["error"]["type"] == "CadError"
    assert out["error"]["location"].startswith("operations[0]")


def test_schema(capsys):
    code, out = run_json(capsys, "cad", "schema")
    assert code == 0 and out["$id"] == "aether3d://schema/cad-script"
