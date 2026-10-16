"""``aether3d sculpt run|check|schema``: atomic scripts, dry runs, history files, tiers
and exit codes, structured errors."""

import json

import pytest

from aether3d import Mesh, Scene, create_sphere
from aether3d.cli import main

STROKE = {
    "op": "stroke",
    "brush": {"kind": "inflate", "radius": 0.5, "strength": 0.2, "symmetry": "x"},
    "path": [[0.4, -0.9, 0.0]],
}


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("Studio")
    scene.add_object("Ball", mesh=create_sphere(1.0, 32, 16))
    scene.add_plane("Floor", size=4)
    path = tmp_path / "studio.aether"
    scene.save(path)
    return path


def write(tmp_path, *operations, obj="Ball", **entry):
    script = {
        "format": "aether3d.sculpt-script",
        "version": 1,
        "sculpts": [{"object": obj, "operations": list(operations), **entry}],
    }
    path = tmp_path / "script.json"
    path.write_text(json.dumps(script))
    return path


def test_run_saves_once_and_dry_run_writes_nothing(tmp_path, capsys, project):
    before = project.read_bytes()
    script = write(tmp_path, STROKE, require="closed")
    code, out = run_json(capsys, "sculpt", "run", str(project), str(script), "--dry-run")
    assert code == 0 and out["dry_run"] and out["output"] is None
    assert out["sculpts"][0]["commit"]["changed"]
    assert project.read_bytes() == before
    code, out = run_json(capsys, "sculpt", "run", str(project), str(script), "--history")
    assert code == 0 and out["output"] == str(project)
    after = Scene.load(project)
    assert after.get_by_name("Ball").mesh.positions.tobytes() != before
    code, out = run_json(capsys, "history", "show", str(project))
    assert code == 0 and out["entries"][-1]["label"] == "aether3d sculpt run script.json"


def test_failures_are_structured_and_change_nothing(tmp_path, capsys, project):
    before = project.read_bytes()
    script = write(tmp_path, STROKE, {"op": "voxel_remesh", "voxel_size": 0.1}, obj="Floor")
    code, out = run_json(capsys, "sculpt", "run", str(project), str(script))
    assert code != 0 and not out["ok"]
    error = out["error"]
    assert error["code"] == "VOXEL_NOT_WATERTIGHT"
    assert error["location"] == "sculpts[0].operations[1]" and error["hint"]
    assert project.read_bytes() == before


def test_check_reports_tiers_and_exit_codes(tmp_path, capsys, project):
    code, out = run_json(capsys, "sculpt", "check", str(project))
    tiers = {r["object"]["name"]: r["summary"]["tier"] for r in out["objects"]}
    assert code == 0 and tiers == {"Ball": "closed", "Floor": "manifold"}
    code, out = run_json(capsys, "sculpt", "check", str(project), "--require", "closed")
    assert code == 3 and out["below"] == ["Floor"]
    code, out = run_json(
        capsys,
        "sculpt",
        "check",
        str(project),
        "--object",
        "Ball",
        "--self-intersections",
        "--require",
        "solid",
    )
    assert code == 0 and out["objects"][0]["summary"]["self_intersecting_pairs"] == 0
    code, out = run_json(capsys, "sculpt", "check", str(project), "--object", "Nobody")
    assert code != 0 and out["error"]["type"] == "ObjectNotFoundError"


def test_check_finds_broken_meshes(tmp_path, capsys):
    import numpy as np

    ball = create_sphere(1.0, 16, 8)
    p = np.concatenate([ball.positions, ball.positions + np.float32([0.5, 0, 0])])
    t = np.concatenate([ball.triangles, ball.triangles + len(ball.positions)])
    scene = Scene()
    scene.add_object("Pair", mesh=Mesh(p, t))
    path = tmp_path / "pair.aether"
    scene.save(path)
    code, out = run_json(
        capsys, "sculpt", "check", str(path), "--self-intersections", "--require", "solid"
    )
    summary = out["objects"][0]["summary"]
    assert code == 3 and summary["tier"] == "closed" and summary["components"] == 2
    assert summary["self_intersecting_pairs"] > 0


def test_schema(capsys):
    code = main(["sculpt", "schema"])
    schema = json.loads(capsys.readouterr().out)
    assert code == 0 and schema["$id"] == "aether3d://schema/sculpt-script"
