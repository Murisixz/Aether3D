import json
import subprocess
import sys
from pathlib import Path

import pytest

from aether3d import FORMAT_VERSION, Scene
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
        "format_version": FORMAT_VERSION,
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
    assert lines[0] == f'Scene "CLI Scene" (3 objects, format v{FORMAT_VERSION})'
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


# --------------------------------------------------------------------------- #
# Interchange and geometry commands
# --------------------------------------------------------------------------- #

KHRONOS = Path(__file__).parent / "data" / "khronos"


def test_export_and_import_commands_round_trip(project, tmp_path, capsys):
    glb = tmp_path / "out.glb"
    code, data = run_json(capsys, "export", str(project), str(glb))
    assert code == 0
    assert data["format"] == "glb" and data["source_format"] == "aether"
    assert data["node_count"] == 3 and data["mesh_count"] == 2
    assert [n["name"] for n in data["nodes"]] == ["Group", "Cube", "Sphere"]

    back = tmp_path / "back.aether"
    code, data = run_json(capsys, "import", str(glb), str(back))
    assert code == 0 and data["lossy"] is False and data["diagnostics"] == []
    assert all(o["id_source"] == "extras" for o in data["objects"])
    original, restored = Scene.load(project), Scene.load(back)
    assert [o.id for o in restored] == [o.id for o in original]
    assert [o.transform for o in restored] == [o.transform for o in original]


def test_export_refuses_overwrite_and_bad_extension(project, tmp_path, capsys):
    target = tmp_path / "x.glb"
    target.write_bytes(b"keep")
    code, data = run_json(capsys, "export", str(project), str(target))
    assert code == 1 and data["error"]["type"] == "FileExistsError"
    assert target.read_bytes() == b"keep"
    assert run_json(capsys, "export", str(project), str(target), "--force")[0] == 0
    code, data = run_json(capsys, "export", str(project), str(tmp_path / "x.obj"))
    assert code == 1 and data["error"]["type"] == "ValidationError"


def test_import_reports_lossy_content_and_writes_nothing(tmp_path, capsys):
    source = KHRONOS / "SimpleSkin/glTF-Embedded/SimpleSkin.gltf"
    dest = tmp_path / "skin.aether"
    code, data = run_json(capsys, "import", str(source), str(dest))
    assert code == 1 and not dest.exists()
    assert data["error"]["type"] == "UnsupportedFeatureError"
    assert [d["code"] for d in data["error"]["diagnostics"]] == ["SKIN_IGNORED"]
    code, data = run_json(capsys, "import", str(source), str(dest), "--allow-lossy")
    assert code == 0 and data["lossy"] is True and dest.exists()


def test_import_errors_have_code_and_location(tmp_path, capsys):
    bad = tmp_path / "bad.gltf"
    bad.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": 4, "uri": "../x"}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}],
                "bufferViews": [{"buffer": 0, "byteLength": 12}],
            }
        )  # fmt: skip
    )
    code, data = run_json(capsys, "validate", str(bad))
    assert code == 1
    assert data["error"] == {
        "type": "AssetFormatError",
        "message": data["error"]["message"],
        "location": "/buffers/0/uri",
        "code": "UNSAFE_URI",
    }
    code, out, err = run(capsys, "import", str(bad), str(tmp_path / "x.aether"))
    assert code == 1 and err.startswith("error: /buffers/0/uri: refusing")


def test_validate_and_inspect_accept_gltf(capsys):
    box = KHRONOS / "Box/glTF-Binary/Box.glb"
    code, data = run_json(capsys, "validate", str(box))
    assert code == 0 and data["source_format"] == "glb"
    assert [d["code"] for d in data["import_diagnostics"]] == []  # Box's material is imported
    code, out, _ = run(capsys, "inspect", str(box))
    assert code == 0 and "imported from GLB" in out.splitlines()[0]


def test_geometry_command(project, capsys):
    code, data = run_json(capsys, "geometry", str(project))
    assert code == 0
    report = data["geometry"]
    assert report["status"] == "ok" and report["summary"]["unique_mesh_count"] == 2
    cube = next(m for m in report["meshes"] if m["vertex_count"] == 24)
    assert cube["topology"]["closed"] and cube["topology"]["volume"] == pytest.approx(8.0)
    code, out, _ = run(capsys, "geometry", str(project))
    assert code == 0 and out.startswith('Geometry of "CLI Scene": OK')


def test_geometry_exit_code_3_on_errors(project, capsys, monkeypatch):
    from aether3d import inspection
    from aether3d.diagnostics import Diagnostic

    original = inspection.inspect_mesh

    def with_error(mesh, *, location=None):
        report = original(mesh, location=location)
        report.diagnostics.append(Diagnostic("NON_FINITE_POSITIONS", "error", "test", location))
        return report

    monkeypatch.setattr(inspection, "inspect_mesh", with_error)
    code, data = run_json(capsys, "geometry", str(project))
    assert code == 3 and data["ok"] is True and data["geometry"]["status"] == "error"


def test_geometry_of_gltf_with_allow_lossy(capsys):
    modes = KHRONOS / "MeshPrimitiveModes/glTF/MeshPrimitiveModes.gltf"
    assert run_json(capsys, "geometry", str(modes))[0] == 1
    code, data = run_json(capsys, "geometry", str(modes), "--allow-lossy")
    assert code == 0 and data["lossy"] is True
    assert data["geometry"]["summary"]["unique_triangle_count"] == 16
