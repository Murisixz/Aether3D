import json
import subprocess
import sys

import pytest

from aether3d import Scene
from aether3d.cli import main


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def run_json(capsys, *argv):
    code, out, _ = run(capsys, *argv, "--json")
    return code, json.loads(out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("CLI Scene")
    group = scene.add_object("Group", position=(1, 0, 0))
    scene.add_cube("Cube", size=2, parent=group, position=(0, 0, 1))
    scene.add_sphere("Sphere")
    path = tmp_path / "project.aether"
    scene.save(path)
    return path


def test_new_creates_valid_empty_project(tmp_path, capsys):
    path = tmp_path / "fresh.aether"
    code, out, _ = run(capsys, "new", str(path))
    assert code == 0 and "fresh" in out
    scene = Scene.load(path)
    assert scene.name == "fresh" and len(scene) == 0


def test_new_with_name_and_json(tmp_path, capsys):
    path = tmp_path / "x.aether"
    code, data = run_json(capsys, "new", str(path), "--name", "My World")
    assert code == 0
    assert data == {
        "ok": True,
        "command": "new",
        "path": str(path),
        "scene": "My World",
        "format_version": 1,
    }
    assert Scene.load(path).name == "My World"


def test_new_refuses_to_overwrite(project, capsys):
    before = project.read_bytes()
    code, data = run_json(capsys, "new", str(project))
    assert code == 1
    assert data["ok"] is False and data["error"]["type"] == "FileExistsError"
    assert project.read_bytes() == before
    code, _ = run_json(capsys, "new", str(project), "--force")
    assert code == 0 and len(Scene.load(project)) == 0


def test_validate_ok(project, capsys):
    code, out, _ = run(capsys, "validate", str(project))
    assert code == 0 and out.startswith("OK:")
    code, data = run_json(capsys, "validate", str(project))
    assert code == 0
    assert data["valid"] is True
    assert data["object_count"] == 3 and data["mesh_count"] == 2


def test_validate_reports_location_of_errors(project, capsys):
    doc = json.loads(project.read_text())
    doc["scene"]["objects"][1]["transform"]["scale"] = [1, 1]
    project.write_text(json.dumps(doc))
    code, out, err = run(capsys, "validate", str(project))
    assert code == 1 and out == ""
    assert err.startswith("error: scene.objects[1].transform.scale:")
    code, data = run_json(capsys, "validate", str(project))
    assert code == 1
    assert data["error"]["type"] == "ProjectFormatError"
    assert data["error"]["location"] == "scene.objects[1].transform.scale"


@pytest.mark.parametrize("command", ["validate", "inspect"])
def test_missing_file(tmp_path, capsys, command):
    code, data = run_json(capsys, command, str(tmp_path / "missing.aether"))
    assert code == 1
    assert data["error"]["type"] == "FileNotFoundError"


def test_invalid_json_file(tmp_path, capsys):
    path = tmp_path / "broken.aether"
    path.write_text("{")
    code, data = run_json(capsys, "validate", str(path))
    assert code == 1 and "invalid JSON" in data["error"]["message"]


def test_inspect_json(project, capsys):
    code, data = run_json(capsys, "inspect", str(project))
    assert code == 0
    scene = Scene.load(project)
    assert data["scene"]["name"] == "CLI Scene"
    assert data["scene"]["object_count"] == 3
    assert data["scene"]["roots"] == [o.id for o in scene.roots]
    by_name = {o["name"]: o for o in data["objects"]}
    assert [o["name"] for o in data["objects"]] == ["Group", "Cube", "Sphere"]
    cube = by_name["Cube"]
    assert cube["parent"] == by_name["Group"]["id"]
    assert by_name["Group"]["children"] == [cube["id"]]
    assert cube["depth"] == 1
    assert cube["transform"]["position"] == [0.0, 0.0, 1.0]
    assert cube["world_position"] == [1.0, 0.0, 1.0]
    assert cube["mesh"]["vertex_count"] == 24
    assert cube["mesh"]["bounds"] == {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    assert by_name["Group"]["mesh"] is None


def test_inspect_human_tree(project, capsys):
    code, out, _ = run(capsys, "inspect", str(project))
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == 'Scene "CLI Scene" (3 objects, format v1)'
    assert lines[1].startswith("├── Group")
    assert lines[2].startswith("│   └── Cube") and "mesh=24v/12t" in lines[2]
    assert lines[3].startswith("└── Sphere")


def test_inspect_empty(tmp_path, capsys):
    path = tmp_path / "e.aether"
    Scene("E").save(path)
    _, out, _ = run(capsys, "inspect", str(path))
    assert "(empty)" in out


def test_usage_errors_exit_2(capsys):
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        main(["frobnicate"])
    assert info.value.code == 2


def test_python_dash_m_entry_point(project):
    result = subprocess.run(
        [sys.executable, "-m", "aether3d", "validate", str(project), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_inspect_falls_back_to_ascii_tree(project, capsys, monkeypatch):
    import io

    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="ascii"))
    from aether3d import cli

    assert cli._tree_glyphs() == ("|-- ", "`-- ", "|   ")
