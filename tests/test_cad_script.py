"""CAD scripts: the guide's examples run and are schema-valid; structure errors, rollback,
budgets, cancellation; units in scripts; list edits."""

from __future__ import annotations

import json
import re
from importlib import resources

import pytest

import aether3d
from aether3d.cad.schema import cad_script_schema
from aether3d.cad.script import check_script, run_cad_script
from aether3d.errors import CadError

pytest.importorskip("manifold3d")
jsonschema = pytest.importorskip("jsonschema")

GUIDE = resources.files("aether3d.cad").joinpath("guide.md").read_text(encoding="utf-8")


def _blocks():
    return re.findall(r"```json\n(.*?)\n```", GUIDE, re.S)


def _script(ops, **extra):
    return {"format": "aether3d.cad-script", "version": 1, "operations": ops, **extra}


def _valid(script):
    jsonschema.validate(script, cad_script_schema())
    return script


def test_the_guides_examples_run_and_match_the_schema():
    blocks = _blocks()
    assert len(blocks) == 4
    part = _valid(json.loads(blocks[0]))
    scene = aether3d.Scene("guide")
    out = run_cad_script(scene, part)
    assert [op["op"] for op in out["operations"]][-2:] == ["expect", "expect"]
    edits = _valid(_script(json.loads("[" + blocks[1] + "]")))
    run_cad_script(scene, edits)
    doc = scene.cad
    assert doc.result("Profile").point2("p2")[0] == pytest.approx(0.1, rel=1e-12)
    house = _valid(_script(json.loads("[" + blocks[2] + "]")))
    out = run_cad_script(scene, house)
    rows = out["operations"][-1]["result"]
    assert {r["element"] for r in rows} == {"wall"} and len(rows) == 5
    drawing = _valid(_script([json.loads(blocks[3])]))
    run_cad_script(scene, drawing)
    d = doc.result("Ground plan")
    assert d.scale_label() == "1:100" and d.notes == []


def test_lengths_in_scripts_use_the_document_unit():
    scene = aether3d.Scene("units")
    run_cad_script(scene, _script([
        {"op": "units", "length": "in"},
        {"op": "add", "kind": "sketch", "name": "S", "definition": {"entities": [
            {"id": "c", "type": "point", "at": [1, "25.4 mm"]},
            {"id": "k", "type": "circle", "center": "c", "radius": 0.5}]}},
        {"op": "append", "item": "S", "list": "entities",
         "values": [{"id": "q", "type": "point", "at": [2, 0]}]},
        {"op": "update", "item": "S", "list": "entities", "id": "q", "changes": {"at": [3, 0]}},
    ]))  # fmt: skip
    res = scene.cad.result("S")
    assert res.point2("c") == pytest.approx([0.0254, 0.0254], rel=1e-15)
    assert res.point2("q") == pytest.approx([0.0762, 0.0], rel=1e-15)
    assert res.entity("k")["radius"] == pytest.approx(0.0127, rel=1e-15)


def test_a_failing_step_rolls_back_everything():
    scene = aether3d.Scene("rollback")
    script = json.loads(_data("bracket_script.json").read_text())
    run_cad_script(scene, script)
    doc = scene.cad
    before = json.dumps(doc.to_json(), sort_keys=True)
    for ops, code, where in (
        ([{"op": "parameter", "name": "width", "value": "150 mm"},
          {"op": "update", "item": "Bracket", "list": "features", "id": "h",
           "changes": {"points": [[500, 20, 10]]}}], "HOLE_OFF_FACE", "operations[1]"),
        ([{"op": "remove", "item": "Profile"}], "ITEM_IN_USE", "operations[0]"),
        ([{"op": "delete", "item": "Bracket", "list": "features", "ids": ["nope"]}],
         "UNKNOWN_ELEMENT", "operations[0]"),
        ([{"op": "append", "item": "Bracket", "list": "walls", "values": [{}]}],
         "INVALID_SCRIPT", "operations[0].list"),
        ([{"op": "parameter", "name": "thickness", "value": "1 mm"},
          {"op": "expect", "measure": {
              "a": {"part": "Bracket", "face": "base:end"},
              "b": {"part": "Bracket", "face": "base:start"}},
           "checks": {"parallel_distance": {"min": 5}}}], "EXPECTATION_FAILED", "operations[1]"),
    ):  # fmt: skip
        with pytest.raises(CadError) as e:
            run_cad_script(scene, _script(ops))
        assert e.value.code == code, (code, e.value)
        assert e.value.location.startswith(where), e.value.location
        assert json.dumps(doc.to_json(), sort_keys=True) == before


def test_structure_is_checked_before_anything_runs():
    for bad, where in (
        ({"format": "aether3d.cad-script", "version": 2, "operations": [{"op": "info"}]},
         "script.version"),
        (_script([]), "script.operations"),
        (_script([{"op": "explode"}]), "operations[0].op"),
        (_script([{"op": "add", "kind": "part", "name": "P"}]), "operations[0]"),
        (_script([{"op": "info", "extra": 1}]), "operations[0]"),
        (_script([{"op": "info"}], limits={"max_seconds": -1}), "script.limits.max_seconds"),
        (_script([{"op": "info"}], limits={"max_bytes": 1}), "script.limits"),
    ):  # fmt: skip
        with pytest.raises(CadError) as e:
            check_script(bad)
        assert e.value.code == "INVALID_SCRIPT" and e.value.location == where


def test_budgets_and_cancellation():
    scene = aether3d.Scene("budget")
    ops = [{"op": "parameter", "name": f"p{k}", "value": k} for k in range(20)]
    with pytest.raises(CadError) as e:
        run_cad_script(scene, _script(ops, limits={"max_items": 10}))
    assert e.value.code == "LIMIT_EXCEEDED" and scene.cad.items() == []
    with pytest.raises(CadError) as e:
        run_cad_script(scene, _script(ops), should_stop=lambda: True)
    assert e.value.code == "CANCELLED" and scene.cad.items() == []
    with pytest.raises(CadError) as e:
        run_cad_script(scene, _script(ops, limits={"max_seconds": 1e-9}))
    assert e.value.code == "BUDGET_EXCEEDED" and scene.cad.items() == []


def test_one_undo_step():
    scene = aether3d.Scene("undo")
    run_cad_script(scene, json.loads(_data("bracket_script.json").read_text()))
    assert len(scene.cad.items()) == 4
    scene.history.undo()
    assert scene.cad.items() == [] and len(scene) == 0
    scene.history.redo()
    assert scene.cad.part("Bracket").measure("main")["valid"]


def _data(name):
    from pathlib import Path

    return Path(__file__).parent / "data" / "cad" / name


def test_scene_plans_run_cad_scripts_in_their_step():
    from aether3d.errors import PlanError
    from aether3d.plans import execute_plan

    scene = aether3d.Scene("plan")
    script = json.loads(_data("bracket_script.json").read_text())
    plan = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "operations": [
            {"op": "cad", "script": script},
            {"op": "update", "select": {"names": ["Bracket"]}, "set": {"position": [0, 0, 1]}},
        ],
    }
    from aether3d.plans import plan_schema

    jsonschema.validate(plan, plan_schema())
    result = execute_plan(scene, plan).to_dict()
    assert result["operations"][0]["cad_items"] == 4
    assert scene.get_by_name("Bracket").position == (0.0, 0.0, 1.0)
    bad = {"format": "aether3d.scene-plan", "version": 1, "operations": [
        {"op": "cad", "script": _script([{"op": "explode"}])}]}  # fmt: skip
    with pytest.raises(PlanError) as e:
        execute_plan(scene, bad)
    assert e.value.pointer == "/operations/0/script/operations/0/op"
