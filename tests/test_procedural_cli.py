"""Procedural scripts (declarative JSON) and the ``aether3d procedural`` commands."""

import io
import json
import subprocess
import sys

import pytest

from aether3d import Scene, ScriptError, run_procedural_script
from aether3d.cli import main

FORMAT = {"format": "aether3d.procedural-script", "version": 1}

CASTLE_WALL = [
    {"op": "add_parameter", "name": "width", "value": 10.0, "min": 2, "max": 50},
    {"op": "add_parameter", "name": "towers", "value": 4, "min": 2, "max": 12},
    {"op": "add_rule", "name": "even", "expr": "towers % 2 == 0", "message": "towers must be even"},
    {
        "op": "add_geometry",
        "name": "Wall",
        "primitive": "box",
        "params": {"width": {"expr": "width"}, "depth": 1, "height": 3},
    },
    {
        "op": "add_geometry",
        "name": "Tower",
        "primitive": "cylinder",
        "params": {"radius": 1, "depth": 5, "segments": 8},
        "define": [{"name": "spacing", "value": {"expr": "width / (towers - 1)"}, "type": "float"}],
    },
    {
        "op": "add_modifier",
        "geometry": "Tower",
        "type": "array",
        "name": "Row",
        "params": {
            "count": {"expr": "towers"},
            "relative_offset": [0, 0, 0],
            "constant_offset": {"expr": "(spacing, 0, 0)"},
        },
    },
    {"op": "create_object", "name": "Wall", "geometry": "Wall"},
    {"op": "create_object", "name": "Towers", "geometry": "Tower", "position": [-5, 0, 0]},
    {"op": "validate", "require_ok": True},
]


def definitions(scene) -> str:
    info = scene.procedural.info()
    info.pop("cache")
    return json.dumps(info, sort_keys=True)


def script(*operations):
    return {**FORMAT, "operations": list(operations)}


def test_script_builds_and_reports():
    scene = Scene()
    result = run_procedural_script(scene, script(*CASTLE_WALL))
    assert [r["op"] for r in result["operations"]] == [op["op"] for op in CASTLE_WALL]
    assert result["operations"][-1]["report"]["status"] == "ok"
    towers = scene.get_by_name("Towers")
    assert towers.is_procedural and towers.mesh.triangle_count == 4 * 28
    run_procedural_script(scene, script({"op": "set_parameter", "name": "towers", "value": 6}))
    assert towers.mesh.triangle_count == 6 * 28
    info = run_procedural_script(
        scene, script({"op": "info", "geometry": "Tower", "evaluate": True})
    )
    data = info["operations"][0]["info"]
    assert data["parameters"]["spacing"]["resolved"] == pytest.approx(2.0)
    assert data["evaluated"]["mesh"]["triangle_count"] == 6 * 28


@pytest.mark.parametrize(
    ("operations", "location", "code"),
    [
        (
            [{"op": "set_parameter", "name": "towers", "value": 5}],
            "operations[0]: parameters.rules[even]",
            "CONSTRAINT_VIOLATED",
        ),
        (
            [
                {
                    "op": "set_modifier",
                    "geometry": "Tower",
                    "modifier": "Row",
                    "params": {"count": 0},
                }
            ],
            "operations[0]: geometry[Tower].modifiers[Row].params.count",
            "PARAMETER_OUT_OF_RANGE",
        ),
        (
            [{"op": "add_modifier", "geometry": "Tower", "type": "explode"}],
            "operations[0]",
            "UNKNOWN_MODIFIER",
        ),
        (
            [
                {"op": "create_object", "name": "New", "geometry": "Wall"},
                {"op": "remove_geometry", "geometry": "Nope"},
            ],
            "operations[1]",
            "UNKNOWN_GEOMETRY",
        ),
        ([{"op": "remove_geometry", "geometry": "Wall"}], "operations[0]", "GEOMETRY_IN_USE"),
        ([{"op": "teleport"}], "operations[0].op", "UNKNOWN_OPERATION"),
        ([{"op": "set_parameter", "name": "towers"}], "operations[0]", "INVALID_SCRIPT"),
        (
            [{"op": "set_parameter", "name": "towers", "value": 4, "extra": 1}],
            "operations[0]",
            "INVALID_SCRIPT",
        ),
        (
            [{"op": "set_parameter", "name": "width", "value": {"expr": "import os"}}],
            "operations[0]",
            "INVALID_EXPRESSION",
        ),
        (
            [{"op": "add_geometry", "name": "X", "primitive": "box", "source": "Wall"}],
            "operations[0]",
            "INVALID_SOURCE",
        ),
        ([{"op": "bake", "object": "Ghost"}], "operations[0]", "UNKNOWN_OBJECT"),
    ],
)
def test_failing_scripts_change_nothing(operations, location, code):
    scene = Scene()
    run_procedural_script(scene, script(*CASTLE_WALL))
    before = definitions(scene)
    names = [o.name for o in scene]
    revision = scene.procedural.revision
    with pytest.raises(ScriptError) as info:
        run_procedural_script(
            scene, script({"op": "set_parameter", "name": "width", "value": 20.0}, *operations)
        )
    err = info.value
    shifted = location.replace("operations[1]", "operations[2]").replace(
        "operations[0]", "operations[1]"
    )
    assert err.code == code and err.location.startswith(shifted), (err.code, err.location)
    assert definitions(scene) == before
    assert [o.name for o in scene] == names and scene.procedural.revision == revision


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"format": "aether3d.edit-script", "version": 1, "operations": [{"op": "info"}]},
        {**FORMAT, "version": 2, "operations": [{"op": "info"}]},
        {**FORMAT, "operations": []},
        {**FORMAT, "operations": [{"op": "info"}], "extra": True},
        {**FORMAT, "operations": ["info"]},
    ],
)
def test_malformed_scripts(bad):
    with pytest.raises(ScriptError):
        run_procedural_script(Scene(), bad)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "castle.aether"
    Scene("Castle").save(path)
    script_path = tmp_path / "build.json"
    script_path.write_text(json.dumps(script(*CASTLE_WALL)))
    assert main(["procedural", "run", str(path), str(script_path)]) == 0
    return path


def test_cli_run_inspect_set_validate(project, capsys, tmp_path):
    capsys.readouterr()
    code, data = run_json(capsys, "procedural", "inspect", str(project), "--evaluate")
    assert code == 0 and data["ok"] and data["command"] == "procedural inspect"
    assert {g["name"] for g in data["geometry"]} == {"Wall", "Tower"}
    assert data["parameters"]["towers"]["resolved"] == 4
    assert {o["name"] for o in data["objects"]} == {"Wall", "Towers"}

    code, data = run_json(capsys, "procedural", "set", str(project), "towers=6", "width=12")
    assert code == 0 and data["output"] == str(project)
    loaded = Scene.load(project)
    assert loaded.get_by_name("Towers").mesh.triangle_count == 6 * 28

    code, data = run_json(capsys, "procedural", "set", str(project), "towers=7")
    assert code == 1 and data["error"]["code"] == "CONSTRAINT_VIOLATED"
    assert data["error"]["location"].startswith("operations[0]: parameters.rules[even]")
    assert Scene.load(project).procedural.parameters["towers"].value == 6  # file untouched

    code, data = run_json(capsys, "procedural", "set", str(project), "width=30", "--dry-run")
    assert code == 0 and data["output"] is None
    assert Scene.load(project).procedural.parameters["width"].value == 12

    code, data = run_json(
        capsys, "procedural", "set", str(project), "spacing=3", "--geometry", "Tower"
    )
    assert code == 0
    code, data = run_json(capsys, "procedural", "validate", str(project))
    assert code == 0 and data["status"] == "ok"

    code, data = run_json(capsys, "inspect", str(project))
    objects = {o["name"]: o for o in data["objects"]}
    assert objects["Towers"]["geometry"]["name"] == "Tower"
    assert objects["Towers"]["mesh"]["triangle_count"] == 6 * 28
    code, text = main(["inspect", str(project)]), capsys.readouterr().out
    assert "procedural=Tower" in text

    code, text = main(["procedural", "inspect", str(project)]), capsys.readouterr().out
    assert code == 0 and "towers: int = 6" in text and "Row [array]" in text


def test_cli_validate_reports_geometry_errors_with_exit_3(tmp_path, capsys):
    path = tmp_path / "bad.aether"
    scene = Scene()
    scene.add_geometry("Sheet", "box")
    scene.save(path)
    # Make the definition fail at evaluation only (valid structure, impossible geometry).
    doc = json.loads(path.read_text())
    doc["scene"]["geometry"][0]["modifiers"].append(
        {
            "id": "m1",
            "name": "Bevel",
            "type": "bevel",
            "enabled": True,
            "params": {"width": 5.0, "limit": "all"},
        }
    )
    path.write_text(json.dumps(doc))
    code, data = run_json(capsys, "procedural", "validate", str(path))
    assert code == 3 and data["status"] == "error"
    assert data["geometry"][0]["error"]["code"] == "BEVEL_WIDTH_TOO_LARGE"
    # Commands that need the evaluated mesh fail cleanly (exit 1, JSON error).
    doc["scene"]["objects"].append(
        {
            "id": "o1",
            "name": "Sheet",
            "parent": None,
            "mesh": None,
            "geometry": doc["scene"]["geometry"][0]["id"],
            "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
        }
    )
    path.write_text(json.dumps(doc))
    code, data = run_json(capsys, "geometry", str(path))
    assert code == 1 and data["error"]["type"] == "ProceduralError"
    assert data["error"]["location"].startswith("geometry[Sheet].modifiers[Bevel]")


def test_cli_bake(project, capsys):
    capsys.readouterr()
    code, data = run_json(capsys, "procedural", "bake", str(project), "--object", "Towers")
    assert code == 0 and [b["name"] for b in data["baked"]] == ["Towers"]
    scene = Scene.load(project)
    assert not scene.get_by_name("Towers").is_procedural
    assert scene.get_by_name("Towers").mesh.triangle_count == 4 * 28
    assert scene.get_by_name("Wall").is_procedural
    code, data = run_json(capsys, "procedural", "bake", str(project), "--all")
    assert code == 0 and [b["name"] for b in data["baked"]] == ["Wall"]
    code, data = run_json(capsys, "procedural", "bake", str(project))
    assert code == 1


def test_cli_types_and_usage_errors(capsys, tmp_path):
    code, data = run_json(capsys, "procedural", "types")
    assert code == 0
    assert {t["type"] for t in data["modifiers"]} == {
        "transform",
        "mirror",
        "array",
        "solidify",
        "subdivision",
        "bevel",
        "boolean",
    }
    assert {t["type"] for t in data["primitives"]} >= {"box", "grid", "cylinder", "cone", "torus"}
    with pytest.raises(SystemExit) as info:
        main(["procedural"])
    assert info.value.code == 2
    code, data = run_json(capsys, "procedural", "inspect", str(tmp_path / "missing.aether"))
    assert code == 1 and data["error"]["type"] == "FileNotFoundError"
    code, data = run_json(capsys, "procedural", "run", str(tmp_path / "x.glb"), "-")
    assert code == 1


def test_cli_run_from_stdin(tmp_path, monkeypatch, capsys):
    path = tmp_path / "s.aether"
    Scene().save(path)
    payload = json.dumps(script(*CASTLE_WALL)).encode()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    code, data = run_json(capsys, "procedural", "run", str(path), "-")
    assert code == 0 and len(data["operations"]) == len(CASTLE_WALL)


def test_cli_entry_point(project):
    out = subprocess.run(
        [sys.executable, "-m", "aether3d.cli", "procedural", "validate", str(project), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0 and json.loads(out.stdout)["status"] == "ok"


def test_creating_many_objects_with_a_script_scales_linearly():
    """Each step is checked, but a created object needs no scan of every object (which
    made N creations cost O(N^2)): 8x the objects cost about 8x the time."""
    import time

    def run(n):
        operations = [
            {"op": "add_geometry", "name": "Block", "primitive": "box", "params": {"width": 0.9}},
            *(
                {"op": "create_object", "name": f"b{i}", "geometry": "Block", "position": [i, 0, 0]}
                for i in range(n)
            ),
        ]
        scene = Scene("Blocks")
        start = time.perf_counter()
        run_procedural_script(scene, script(*operations))
        seconds = time.perf_counter() - start
        assert len(scene) == n and len(scene.history) == 1
        return seconds

    run(100)  # warm up
    small, large = min(run(500) for _ in range(2)), min(run(4000) for _ in range(2))
    assert large < 16 * small + 0.05, (small, large)


def test_a_script_that_removes_a_used_node_still_fails_at_that_step():
    scene = Scene("Orphans")
    with pytest.raises(ScriptError) as info:
        run_procedural_script(
            scene,
            script(
                {"op": "add_geometry", "name": "Block", "primitive": "box", "params": {}},
                {"op": "create_object", "name": "A", "geometry": "Block"},
                {"op": "remove_geometry", "geometry": "Block"},
            ),
        )
    assert "operations[2]" in str(info.value) and len(scene) == 0
