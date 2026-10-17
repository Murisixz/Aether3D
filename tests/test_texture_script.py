"""Texture scripts: a whole texturing workflow as JSON, all or nothing, with limits, stable
error locations, and no way to reach files or code."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest

from aether3d import Scene, TexturingError
from aether3d.primitives import create_cube, create_sphere
from aether3d.texturing import run_texture_script


def scene_with_crate():
    scene = Scene()
    scene.add_object("Crate", mesh=create_cube(1.0))
    scene.add_object("Ball", mesh=create_sphere(0.5, 24, 12), position=(2, 0, 0))
    scene.add_object("Ball high", mesh=create_sphere(0.5, 64, 32), position=(2, 0, 0))
    scene.add_camera("Cam", position=(0, -4, 1), look_at=(0, 0, 0))
    return scene


WORKFLOW = {
    "format": "aether3d.texture-script",
    "version": 1,
    "label": "Texture the crate",
    "operations": [
        {"op": "unwrap", "object": "Crate", "resolution": 256, "padding": 4},
        {
            "op": "add_texture",
            "name": "Crate color",
            "width": 256,
            "height": 256,
            "background": [0.54, 0.42, 0.28, 1],
            "layers": [{"name": "Dirt"}],
        },
        {"op": "bind", "object": "Crate", "slot": "base_color", "texture": "Crate color",
         "factor": "#ffffff"},
        {"op": "fill", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
         "color": "#202020", "where": {"triangles": [0, 1]}, "opacity": 0.5},
        {"op": "stroke", "object": "Crate", "texture": "Crate color",
         "brush": {"radius": 0.15, "color": "#ff0000"},
         "points": [[0.5, -0.2, 0.0], [0.5, 0.2, 0.0]]},
        {"op": "stroke", "object": "Crate", "texture": "Crate color", "space": "view",
         "camera": "Cam", "view_size": [128, 128],
         "brush": {"radius": 6, "color": "#00ff00"}, "points": [[60, 70], [70, 70]]},
        {"op": "gradient", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
         "start": [0, 0, -0.5], "end": [0, 0, 0.5], "colors": ["#000000", "#ffffff"],
         "opacity": 0.2},
        {"op": "uv_report", "object": "Crate", "resolution": 256},
        {"op": "bake", "target": "Ball", "sources": ["Ball high"], "maps": ["normal", "ao"],
         "resolution": 64, "ao_rays": 8, "backend": "cpu", "apply": True},
        {"op": "texture_info", "texture": "Crate color"},
    ],
}  # fmt: skip


def test_a_whole_workflow_is_one_undo_step():
    scene = scene_with_crate()
    original = scene.get_by_name("Crate").mesh
    n = len(scene.history.entries)
    out = run_texture_script(scene, copy.deepcopy(WORKFLOW))
    assert [r["op"] for r in out["operations"]] == [s["op"] for s in WORKFLOW["operations"]]
    assert len(scene.history.entries) == n + 1
    assert scene.history.entries[-1].label == "Texture the crate"
    crate = scene.get_by_name("Crate")
    assert crate.mesh.uvs is not None
    assert crate.material.base_color == (1.0, 1.0, 1.0)
    doc = scene.materials.texture("Crate color")
    assert [layer.name for layer in doc.layers] == ["Dirt"]
    stroke = out["operations"][4]
    assert stroke["dabs"] > 1 and stroke["texels"] > 0
    assert out["operations"][5]["texels"] > 0  # the view stroke reached the visible side
    ball = scene.get_by_name("Ball")
    assert out["operations"][8]["applied"]["textures"].keys() == {"normal", "ao"}
    assert scene.materials.texture(ball.material.textures["normal"].texture).name.startswith("Ball")
    scene.history.undo()
    assert crate.mesh is original
    assert scene.materials.find_texture("Crate color") is None
    assert ball.material is None


def test_a_failing_step_rolls_everything_back_and_says_where():
    scene = scene_with_crate()
    before = scene.get_by_name("Crate").mesh
    script = copy.deepcopy(WORKFLOW)
    script["operations"][4]["brush"]["radius"] = -1
    with pytest.raises(TexturingError) as exc:
        run_texture_script(scene, script)
    assert exc.value.location.startswith("operations[4]")
    assert scene.get_by_name("Crate").mesh is before
    assert scene.materials.find_texture("Crate color") is None
    assert scene.get_by_name("Crate").material is None


def test_texture_references_are_never_file_paths(tmp_path):
    from aether3d.rendering.png import encode_png

    png = tmp_path / "secret.png"
    png.write_bytes(encode_png(np.zeros((4, 4, 3), np.uint8)))
    scene = scene_with_crate()
    script = {
        "format": "aether3d.texture-script",
        "version": 1,
        "operations": [
            {"op": "bind", "object": "Crate", "slot": "base_color", "texture": str(png)}
        ],
    }
    with pytest.raises(TexturingError) as exc:
        run_texture_script(scene, script)
    assert exc.value.code == "TEXTURE_NOT_FOUND"
    assert not scene.materials.textures


def test_limits_stop_a_script():
    scene = scene_with_crate()
    script = copy.deepcopy(WORKFLOW)
    script["limits"] = {"max_texels": 1000}
    with pytest.raises(TexturingError) as exc:
        run_texture_script(scene, script)
    assert exc.value.code == "BUDGET_EXCEEDED" and exc.value.location == "operations[1]"
    script["limits"] = {"max_dabs": 3}
    with pytest.raises(TexturingError) as exc:
        run_texture_script(scene, script)
    assert exc.value.code == "BUDGET_EXCEEDED" and exc.value.location == "operations[4]"
    assert scene.materials.find_texture("Crate color") is None


@pytest.mark.parametrize(
    "mutate, location",
    [
        (lambda s: s.update(format="aether3d.sculpt-script"), "script.format"),
        (lambda s: s.update(version=2), "script.version"),
        (lambda s: s.update(operations=[]), "script.operations"),
        (lambda s: s.update(extra=1), "script"),
        (lambda s: s["operations"][0].update(op="explode"), "operations[0].op"),
        (lambda s: s["operations"][0].update(should_stop=True), "operations[0]"),
        (lambda s: s["operations"][8].update(progress="x"), "operations[8]"),
        (lambda s: s["operations"][8].update(adapter="x"), "operations[8]"),
        (lambda s: s["operations"][4].update(view={"x": 1}), "operations[4]"),
        (lambda s: s.update(limits={"max_seconds": -1}), "script.limits.max_seconds"),
        (lambda s: s["operations"][0].update(object="Nope"), "operations[0].object"),
    ],
)
def test_malformed_scripts_are_refused_with_locations(mutate, location):
    scene = scene_with_crate()
    script = copy.deepcopy(WORKFLOW)
    mutate(script)
    with pytest.raises(TexturingError) as exc:
        run_texture_script(scene, script)
    assert exc.value.location == location
    assert len(scene.history.entries) == 0 or scene.materials.find_texture("Crate color") is None


def _mutations(rng: random.Random, value):
    """Random structural damage to a JSON document."""
    junk = [None, True, -1, 0, 1e309 if False else 1e30, "", "x" * 3, [], {}, [1, 2], {"a": 1}]
    if isinstance(value, dict) and value:
        key = rng.choice(list(value))
        out = dict(value)
        r = rng.random()
        if r < 0.3:
            out[key] = rng.choice(junk)
        elif r < 0.45:
            del out[key]
        else:
            out[key] = _mutations(rng, value[key])
        return out
    if isinstance(value, list) and value:
        out = list(value)
        i = rng.randrange(len(out))
        if rng.random() < 0.4:
            out[i] = rng.choice(junk)
        else:
            out[i] = _mutations(rng, out[i])
        return out
    return rng.choice(junk)


def test_fuzzed_scripts_fail_cleanly_or_succeed_atomically():
    rng = random.Random(14)
    base = copy.deepcopy(WORKFLOW)
    base["operations"] = [op for op in base["operations"] if op["op"] != "bake"]
    for _ in range(60):
        scene = scene_with_crate()
        script = _mutations(rng, json.loads(json.dumps(base)))
        n = len(scene.history.entries)
        try:
            run_texture_script(scene, script)
        except TexturingError as exc:
            assert exc.code and isinstance(exc.location, (str, type(None)))
            assert len(scene.history.entries) == n
            assert scene.materials.find_texture("Crate color") is None
        else:
            assert len(scene.history.entries) == n + 1


def test_the_schema_accepts_what_the_executor_accepts():
    jsonschema = pytest.importorskip("jsonschema")
    from aether3d.texturing.schema import texture_script_schema

    schema = texture_script_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(WORKFLOW)
    rng = random.Random(7)
    base = copy.deepcopy(WORKFLOW)
    base["operations"] = [op for op in base["operations"] if op["op"] != "bake"]
    accepted = 0
    for _ in range(150):
        script = _mutations(rng, json.loads(json.dumps(base)))
        try:
            run_texture_script(scene_with_crate(), script)
        except TexturingError:
            continue
        accepted += 1
        errors = [e.message for e in validator.iter_errors(script)]
        assert not errors, (script, errors)
    assert accepted > 0
