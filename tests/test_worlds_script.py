"""World scripts, their schema, limits and the command line; inputs are untrusted."""

import copy
import json
import random

import pytest

from aether3d import Scene
from aether3d.cli import main
from aether3d.errors import WorldError
from aether3d.worlds.script import FORMAT, run_world_script, schema

jsonschema = pytest.importorskip("jsonschema")

SCRIPT = {
    "format": FORMAT,
    "version": 1,
    "label": "A valley",
    "operations": [
        {
            "op": "create",
            "name": "Valley",
            "preset": "temperate",
            "seed": 4,
            "size": 1024.0,
            "overrides": {"hydrology": {"cell": 32.0}},
        },
        {
            "op": "add_feature",
            "feature": {
                "type": "lake",
                "id": "tarn",
                "shape": {"circle": [200, 100, 60]},
                "level": 30.0,
            },
        },
        {"op": "place_structure", "item": {"id": "lookout", "kind": "tower", "at": [0, 0]}},
        {"op": "configure", "patch": {"vegetation": {"density": 0.5}}},
    ],
    "queries": [
        {"query": "sample", "points": [[0, 0], [200, 100]], "fields": ["height", "water"]},
        {"query": "structures"},
        {"query": "info"},
    ],
}


def test_a_script_builds_a_world_in_one_step_and_answers_queries():
    scene = Scene("S")
    out = run_world_script(scene, copy.deepcopy(SCRIPT))
    assert [o["op"] for o in out["operations"]] == [
        "create",
        "add_feature",
        "place_structure",
        "configure",
    ]
    assert out["queries"][0]["water"][1] == 30.0
    assert out["queries"][1]["items"][0]["id"] == "lookout"
    assert len(scene.history.entries) == 1 and scene.history.entries[-1].label == "A valley"
    scene.history.undo()
    assert len(scene.worlds) == 0


def test_scene_plans_carry_world_scripts():
    from aether3d import execute_plan
    from aether3d.errors import PlanError

    plan = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "operations": [{"op": "world", "script": copy.deepcopy(SCRIPT)}],
    }
    scene = Scene("Plan")
    op = execute_plan(scene, plan).to_dict()["operations"][0]
    assert op["world_operations"] == 4 and op["worlds"] == ["Valley"]
    assert op["world_queries"][1]["items"][0]["id"] == "lookout"
    assert len(scene.history.entries) == 1  # the plan, world script included, is one step
    bad = copy.deepcopy(plan)
    bad["operations"][0]["script"]["operations"][0]["op"] = "terraform"
    with pytest.raises(PlanError) as info:
        execute_plan(Scene("P"), bad)
    assert info.value.to_dict()["issues"][0]["pointer"].endswith("script/operations/0/op")


def test_failing_scripts_change_nothing_and_name_the_operation():
    scene = Scene("S")
    run_world_script(scene, copy.deepcopy(SCRIPT))
    before = scene.worlds.get("Valley").definition
    bad = {
        "format": FORMAT,
        "world": "Valley",
        "operations": [
            {"op": "place_structure", "item": {"id": "hut", "kind": "house", "at": [5, 5]}},
            {
                "op": "add_feature",
                "feature": {"type": "lake", "id": "tarn", "shape": {"circle": [0, 0, 5]}},
            },
        ],
    }
    with pytest.raises(WorldError) as info:
        run_world_script(scene, bad)
    assert info.value.code == "DUPLICATE_ID" and info.value.location.startswith("operations[1]")
    assert scene.worlds.get("Valley").definition == before
    cases = [
        (
            {"format": FORMAT, "operations": [{"op": "explode"}]},
            "operations[0].op",
            "INVALID_SCRIPT",
        ),
        (
            {"format": FORMAT, "operations": [{"op": "configure", "world": "Nope", "patch": {}}]},
            "operations[0].world",
            "UNKNOWN_WORLD",
        ),
        (
            {
                "format": FORMAT,
                "world": "Valley",
                "operations": [
                    {"op": "configure", "patch": {"terrain": {"mountains": {"height": -5}}}}
                ],
            },
            "operations[0].definition.terrain.mountains.height",
            "INVALID_DEFINITION",
        ),
        ({"format": "aether3d.cad-script", "operations": []}, "format", "INVALID_SCRIPT"),
        ({"format": FORMAT, "operations": [{"op": "create"}]}, "operations[0]", "INVALID_SCRIPT"),
        (
            {"format": FORMAT, "operations": [{"op": "create", "name": "X", "colour": 1}]},
            "operations[0].colour",
            "INVALID_SCRIPT",
        ),
    ]
    for script, where, code in cases:
        with pytest.raises(WorldError) as info:
            run_world_script(scene, script)
        assert info.value.code == code and info.value.location == where, (script, info.value)


def test_the_schema_is_valid_and_accepts_the_examples():
    s = schema()
    jsonschema.Draft202012Validator.check_schema(s)
    jsonschema.validate(SCRIPT, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"format": FORMAT, "operations": [{"op": "explode"}]}, s)


def test_mutated_scripts_raise_world_errors_only_and_change_nothing():
    rng = random.Random(16)
    scene = Scene("Fuzz")
    run_world_script(scene, copy.deepcopy(SCRIPT))
    doc = json.dumps(scene.worlds.get("Valley").definition, sort_keys=True)
    values = [None, -1, 0, 1e308, float("nan"), "x", [], {}, [1, 2], True, "Valley", [[1e9, 0]]]
    for _ in range(150):
        script = copy.deepcopy(SCRIPT)
        script["operations"] = script["operations"][1:]
        script["world"] = "Valley"
        for _ in range(rng.randint(1, 3)):
            step = rng.choice(script["operations"] + script["queries"])
            if not step:
                continue
            key = rng.choice(sorted(step))
            if rng.random() < 0.3:
                del step[key]
            elif isinstance(step[key], dict) and step[key]:
                inner = rng.choice(sorted(step[key]))
                step[key][inner] = rng.choice(values)
            else:
                step[key] = rng.choice(values)
        try:
            run_world_script(scene, script)
        except WorldError:
            pass
        else:
            scene.history.undo()  # (a valid mutation: put it back)
        assert json.dumps(scene.worlds.get("Valley").definition, sort_keys=True) == doc


def test_limits_protect_against_excessive_work():
    scene = Scene("L")
    with pytest.raises(WorldError) as info:
        run_world_script(scene, {"format": FORMAT, "operations": [{"op": "remove"}] * 513})
    assert info.value.code == "INVALID_SCRIPT"
    with pytest.raises(WorldError) as info:
        scene.worlds.create("Huge", "temperate", size=1e9)
    assert info.value.code == "INVALID_DEFINITION"
    w = scene.worlds.create("W", "temperate", seed=1, size=1024.0)
    with pytest.raises(WorldError) as info:
        w.sample([[0.0, 0.0]] * 1_000_001)
    assert info.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(WorldError) as info:
        w.ground_path([[0, 0], [400, 0]], spacing=0.001)
    assert info.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(WorldError) as info:
        run_world_script(
            scene,
            {
                "format": FORMAT,
                "limits": {"max_seconds": 1e-9},
                "world": "W",
                "operations": [{"op": "configure", "patch": {}}] * 3,
            },
        )
    assert info.value.code == "BUDGET_EXCEEDED"


def test_scripts_never_touch_files(tmp_path):
    """No operation takes a path: the only file a world can write is an explicit export,
    which is not part of the script vocabulary."""
    from aether3d.worlds.script import OPERATIONS, QUERIES

    words = {k for req, opt in OPERATIONS.values() for k in req | opt}
    words |= {k for req, opt in QUERIES.values() for k in req | opt}
    assert not any("path" in w or "file" in w for w in words)


def test_the_command_line(tmp_path, capsys):
    project = tmp_path / "p.aether"
    assert main(["new", str(project)]) == 0
    capsys.readouterr()
    script = tmp_path / "w.json"
    script.write_text(json.dumps(SCRIPT))
    assert main(["world", "run", str(project), str(script), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["worlds"][0]["name"] == "Valley"
    assert main(["world", "info", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["world"] == "Valley"
    assert (
        main(["world", "sample", str(project), "--at", "0,0", "--field", "height", "--json"]) == 0
    )
    assert len(json.loads(capsys.readouterr().out)["height"]) == 1
    glb = tmp_path / "r.glb"
    code = main(
        [
            "world",
            "export",
            str(project),
            str(glb),
            "--region",
            "-32,-32,32,32",
            "--level",
            "2",
            "--include",
            "terrain,water",
            "--json",
        ]
    )
    assert code == 0 and glb.stat().st_size > 1000
    capsys.readouterr()
    for sub in ("presets", "blocks", "schema", "guide"):
        assert main(["world", sub, "--json"]) == 0
        capsys.readouterr()
    assert main(["world", "sample", str(project), "--at", "1e9,0", "--json"]) != 0
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False and err["error"].get("code") == "OUTSIDE_WORLD"
    # Lists that start with a minus parse on every Python (argparse reads "-1,2" as an
    # option before 3.14 unless it is glued: --at=-1,2), spaced or glued.
    capsys.readouterr()
    for args in (["--at", "-10,5"], ["--at=-10,5"]):
        assert main(["world", "sample", str(project), *args, "--field", "height", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["x"] == [-10.0]
