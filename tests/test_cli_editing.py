"""CLI access to topology queries and edit scripts."""

import io
import json
import subprocess
import sys

import pytest

from aether3d import Scene, inspect_mesh
from aether3d.cli import main


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("Workshop")
    scene.add_cube("Block", size=2)
    path = tmp_path / "w.aether"
    scene.save(path)
    return path


def write_script(tmp_path, operations, obj="Block", **extra):
    path = tmp_path / "edit.json"
    path.write_text(
        json.dumps(
            {
                "format": "aether3d.edit-script",
                "version": 1,
                "edits": [{"object": obj, "operations": operations, **extra}],
            }
        )
    )
    return path


EXTRUDE = [
    {"op": "select", "kind": "faces", "normal": [[0, 0, 1], 1], "expect": 1, "as": "top"},
    {"op": "extrude_faces", "faces": {"selection": "top"}, "distance": 1},
]


def test_topology_command(project, capsys):
    code, data = run_json(capsys, "topology", str(project), "--object", "Block", "--elements")
    assert code == 0
    summary = data["topology"]["summary"]
    assert (summary["vertex_count"], summary["edge_count"], summary["face_count"]) == (8, 12, 6)
    assert summary["closed"] and summary["vertex_manifold"] and summary["euler_characteristic"] == 2
    assert len(data["elements"]["faces"]) == 6 and data["object"]["name"] == "Block"
    code, data = run_json(capsys, "topology", str(project), "--object", "Block", "--triangles")
    assert data["topology"]["summary"]["face_count"] == 12
    code, data = run_json(capsys, "topology", str(project), "--object", "Nope")
    assert code == 1 and data["error"]["type"] == "ObjectNotFoundError"


def test_edit_in_place(project, tmp_path, capsys):
    script = write_script(tmp_path, EXTRUDE)
    code, data = run_json(capsys, "edit", str(project), str(script))
    assert code == 0 and data["output"] == str(project)
    assert data["edits"][0]["commit"]["changed"]
    block = Scene.load(project).get_by_name("Block")
    assert inspect_mesh(block.mesh).topology["volume"] == pytest.approx(12)


def test_edit_to_output_and_dry_run(project, tmp_path, capsys):
    before = project.read_bytes()
    script = write_script(tmp_path, EXTRUDE)
    out = tmp_path / "out.aether"
    code, data = run_json(capsys, "edit", str(project), str(script), "--output", str(out))
    assert code == 0 and project.read_bytes() == before and out.exists()
    code, data = run_json(capsys, "edit", str(project), str(script), "--dry-run")
    assert code == 0 and data["dry_run"] and data["output"] is None
    assert project.read_bytes() == before


def test_edit_from_stdin(project, tmp_path, capsys, monkeypatch):
    doc = {
        "format": "aether3d.edit-script",
        "version": 1,
        "edits": [{"object": "Block", "operations": EXTRUDE}],
    }
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(doc).encode())))
    code, data = run_json(capsys, "edit", str(project), "-")
    assert code == 0 and data["edits"][0]["commit"]["changed"]


def test_edit_failure_reports_location_and_writes_nothing(project, tmp_path, capsys):
    before = project.read_bytes()
    script = write_script(
        tmp_path, [*EXTRUDE, {"op": "inset_faces", "faces": [0], "thickness": 50}]
    )
    code, data = run_json(capsys, "edit", str(project), str(script))
    assert code == 1 and project.read_bytes() == before
    error = data["error"]
    assert error["type"] == "ScriptError" and error["code"] == "INSET_TOO_LARGE"
    assert error["location"] == "edits[0].operations[2]" and error["elements"]["faces"]
    code = main(["edit", str(project), str(script)])
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("error: edits[0].operations[2]:")


def test_edit_rejects_bad_input(project, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    code, data = run_json(capsys, "edit", str(project), str(bad))
    assert code == 1 and data["error"]["code"] == "INVALID_JSON"
    code, data = run_json(capsys, "edit", str(project), str(tmp_path / "missing.json"))
    assert code == 1 and data["error"]["type"] == "FileNotFoundError"
    glb = tmp_path / "x.glb"
    Scene().export(glb)
    code, data = run_json(capsys, "edit", str(glb), str(bad))
    assert code == 1 and "import" in data["error"]["message"]


def test_topology_exit_code_3_on_errors(project, capsys, monkeypatch):
    from aether3d.diagnostics import Diagnostic
    from aether3d.editing.mesh import EditableMesh

    real = EditableMesh.validate

    def failing(self):
        report = real(self)
        report.diagnostics.append(Diagnostic("SELF_INTERSECTING_FACES", "error", "injected"))
        return report

    monkeypatch.setattr(EditableMesh, "validate", failing)
    code, data = run_json(capsys, "topology", str(project), "--object", "Block")
    assert code == 3 and data["ok"] is True and data["topology"]["status"] == "error"


def test_cli_end_to_end_subprocess(project, tmp_path):
    script = write_script(tmp_path, EXTRUDE)
    result = subprocess.run(
        [sys.executable, "-m", "aether3d", "edit", str(project), str(script), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True
