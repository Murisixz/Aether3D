"""``aether3d texture run|report|bake|schema``: atomic scripts, dry runs, reports and exit
codes, baked PNG files, structured errors."""

import json

import pytest

from aether3d import Mesh, Scene, create_sphere
from aether3d.cli import main
from aether3d.primitives import create_cube

SCRIPT = {
    "format": "aether3d.texture-script",
    "version": 1,
    "operations": [
        {"op": "unwrap", "object": "Crate", "resolution": 128},
        {"op": "add_texture", "name": "Paint", "width": 128, "layers": [{"name": "Ink"}]},
        {"op": "bind", "object": "Crate", "slot": "base_color", "texture": "Paint"},
        {
            "op": "stroke",
            "object": "Crate",
            "texture": "Paint",
            "brush": {"radius": 0.2, "color": "#ff0000"},
            "points": [[0.5, 0.0, 0.0]],
        },
    ],
}


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("Workshop")
    scene.add_object("Crate", mesh=create_cube(1.0))
    scene.add_object("Ball", mesh=create_sphere(0.5, 16, 8), position=(2, 0, 0))
    scene.add_object("Ball high", mesh=create_sphere(0.5, 48, 24), position=(2, 0, 0))
    path = tmp_path / "workshop.aether"
    scene.save(path)
    return path


def write(tmp_path, script):
    path = tmp_path / "script.json"
    path.write_text(json.dumps(script))
    return path


def test_run_saves_once_and_dry_run_writes_nothing(tmp_path, capsys, project):
    before = project.read_bytes()
    script = write(tmp_path, SCRIPT)
    code, data = run_json(capsys, "texture", "run", str(project), str(script), "--dry-run")
    assert code == 0 and data["dry_run"] and project.read_bytes() == before
    assert [r["op"] for r in data["operations"]] == ["unwrap", "add_texture", "bind", "stroke"]
    code, data = run_json(capsys, "texture", "run", str(project), str(script))
    assert code == 0 and data["output"]
    scene = Scene.load(project)
    paint = scene.materials.texture("Paint")
    assert paint.painted and paint.layer("Ink").tiles  # the stroke was saved
    assert scene.get_by_name("Crate").material.textures["base_color"] is not None


def test_run_reports_errors_with_locations(tmp_path, capsys, project):
    before = project.read_bytes()
    bad = json.loads(json.dumps(SCRIPT))
    bad["operations"][3]["brush"]["falloff"] = "box"
    code, data = run_json(capsys, "texture", "run", str(project), str(write(tmp_path, bad)))
    assert code != 0 and data["error"]["location"] == "operations[3].brush.falloff"
    assert project.read_bytes() == before


def test_report_and_require_clean(tmp_path, capsys, project):
    code, data = run_json(capsys, "texture", "report", str(project))
    assert code == 0 and {o["object"]["name"] for o in data["objects"]} >= {"Crate", "Ball"}
    # Stack two triangles of the ball on top of each other in UV: unclean.
    scene = Scene.load(project)
    ball = scene.get_by_name("Ball")
    m = ball.mesh
    uvs = m.uvs.copy()
    t0, t1 = m.triangles[0], m.triangles[len(m.triangles) // 2]
    uvs[t1] = uvs[t0] + [[0.01, 0.0]]
    ball.mesh = Mesh(m.positions, m.triangles, normals=m.normals, uvs=uvs)
    scene.save(project)
    code, data = run_json(capsys, "texture", "report", str(project), "--require-clean")
    assert code == 3 and "Ball" in data["unclean"]


def test_bake_writes_pngs_and_refuses_to_overwrite(tmp_path, capsys, project):
    out = tmp_path / "maps"
    args = ["texture", "bake", str(project), "--target", "Ball", "--source", "Ball high"]
    args += ["--map", "normal", "--map", "ao", "--resolution", "64", "--ao-rays", "4"]
    args += ["--backend", "cpu", "--out", str(out)]
    code, data = run_json(capsys, *args)
    assert code == 0 and set(data["files"]) == {"normal", "ao"}
    assert (out / "Ball_normal.png").read_bytes().startswith(b"\x89PNG")
    code, data = run_json(capsys, *args)
    assert code != 0 and "already exists" in data["error"]["message"]
    code, data = run_json(capsys, *args, "--force", "--apply")
    assert code == 0 and data["output"]
    scene = Scene.load(project)
    assert scene.get_by_name("Ball").material.textures["normal"] is not None


def test_bake_needs_an_output(capsys, project):
    code, data = run_json(capsys, "texture", "bake", str(project), "--target", "Ball")
    assert code != 0 and "--out" in data["error"]["message"]


def test_schema(capsys):
    code, data = run_json(capsys, "texture", "schema")
    assert code == 0 and data["properties"]["format"]["const"] == "aether3d.texture-script"
