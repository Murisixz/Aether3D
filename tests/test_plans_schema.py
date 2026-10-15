"""The plan JSON Schema and Aether3D's own checker agree.

The schema describes the format's shape; ``parse_plan`` checks the shape and much more
(references, cycles, names, expressions, limits). So for every document: if the checker
accepts it, the schema must accept it, and if the schema refuses it, the checker must
refuse it too. Both directions are tested on valid plans and on random mutations of them.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from aether3d import PlanError, plan_schema
from aether3d.plans import parse_plan

try:
    import jsonschema
except ImportError:  # the job without optional dependencies
    jsonschema = None

needs_jsonschema = pytest.mark.skipif(jsonschema is None, reason="jsonschema is not installed")


def plan(**sections) -> dict:
    return {"format": "aether3d.scene-plan", "version": 1, **sections}


VALID = [
    plan(),
    plan(
        label="Castle",
        description="walls and towers",
        idempotency_key="castle-1",
        expected_revision=3,
        options={
            "on_name_conflict": "rename",
            "tags": ["castle"],
            "result_ids": "all",
            "timeout": 30,
        },
        parameters={"h": 6, "w": {"expr": "h * 2"}, "up": [0, 0, 1], "flag": True},
        procedural={"parameters": {"width": {"value": 2, "min": 1}}, "set": {}},
        meshes={
            "block": {"primitive": "cube", "size": 1},
            "slab": {"primitive": "box", "size": [4, 1, {"expr": "h"}]},
            "ball": {"primitive": "sphere", "radius": 0.5, "segments": 12, "rings": 6},
            "tri": {
                "data": {"positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "triangles": [[0, 1, 2]]}
            },
            "copy": {"from_object": "Ground"},
        },
        textures={
            "dots": {"procedural": {"type": "checker"}},
            "wood": {"image": "wood.png", "embed": True},
        },
        materials={
            "stone": {"preset": "stone", "scale": 2},
            "red": {
                "base_color": "#a03020",
                "roughness": 0.8,
                "textures": {"base_color": {"texture": "@dots"}},
            },
        },
        geometries={
            "tower": {
                "primitive": "cylinder",
                "params": {"radius": 2},
                "parameters": {"height": {"value": 9}},
                "modifiers": [{"type": "transform", "params": {"translation": [0, 0, 4]}}],
            },
            "tall": {"source": "@tower", "overrides": {"height": 12}},
        },
        prefabs={"house": {"tags": ["house"], "children": {"walls": {"mesh": "@block"}}}},
        objects={
            "castle": {"position": [0, 0, 0], "tags": ["castle"], "metadata": {"era": 1200}},
            "walls": {
                "parent": "@castle",
                "mesh": "@block",
                "material": "@stone",
                "grid": {"count": [10, 1, {"expr": "h"}], "spacing": [1, 1, 1], "center": True},
                "name": "Brick {x}-{z}",
            },
            "towers": {
                "geometry": "@tower",
                "circle": {"count": 4, "radius": 12, "orient": "outward"},
            },
            "houses": {
                "prefab": "@house",
                "scatter": {
                    "count": 20,
                    "seed": 3,
                    "box": {"min": [-50, -50], "max": [50, 50]},
                    "min_distance": 4,
                },
                "vary": {"rotation_euler": {"uniform": [[0, 0, 0], [0, 0, 360]], "seed": 1}},
            },
            "row": {
                "array": {
                    "count": 5,
                    "offset": [2, 0, 0],
                    "rotation_step": [0, 0, 10],
                    "scale_step": 1.1,
                }
            },
            "fence": {
                "line": {"from": [0, 0, 0], "to": [10, 0, 0], "count": 6},
                "color": [0.4, 0.3, 0.2],
            },
            "props": {
                "mesh": "@ball",
                "instances": [
                    {"position": [1, 2, 3], "name": "Prop"},
                    {"position": [3, 2, 1], "material": None},
                ],
            },
            "cols": {"instances": {"positions": [[0, 0, 0], [1, 1, 1]], "scales": [1, [1, 2, 1]]}},
            "rep": {
                "repeat": 3,
                "position": {"expr": "(i, 0, 0)"},
                "vary": {"color": {"cycle": ["#fff", "#000"]}},
            },
            "sun": {"light": {"type": "directional", "intensity": 2}, "direction": [0, 0, -1]},
            "cam": {
                "camera": {"fov": 40},
                "position": [10, -10, 5],
                "look_at": "@castle",
                "active": True,
                "frame": True,
            },
        },
        operations=[
            {
                "op": "update",
                "select": "@walls[0]",
                "set": {"translate": [0, 0, 1], "add_tags": ["gate"]},
            },
            {
                "op": "update",
                "select": {"tags": ["house"], "expect": {"min": 1}},
                "set": {"scale_by": 1.1},
            },
            {"op": "delete", "select": {"refs": ["@row[4]"]}, "recursive": False},
            {"op": "set_parameters", "values": {"width": 3}},
            {"op": "frame_camera", "camera": "@cam", "targets": {"name_glob": "Brick*"}},
            {"op": "set_active_camera", "camera": "@cam"},
        ],
        world={"background": "#9cc6e8"},
        render_settings={"width": 640, "height": 360},
        active_camera="@cam",
        queries=[{"query": "overlaps", "objects": "@houses"}, {"query": "describe"}],
    ),
]


def _schema_accepts(doc: dict) -> bool:
    return jsonschema.Draft202012Validator(plan_schema()).is_valid(doc)


def _parser_accepts(doc: dict) -> bool:
    try:
        parse_plan(doc)
    except PlanError:
        return False
    return True


@needs_jsonschema
def test_the_schema_is_a_valid_draft_2020_12_schema():
    jsonschema.Draft202012Validator.check_schema(plan_schema())
    assert len(json.dumps(plan_schema())) < 60_000


@needs_jsonschema
@pytest.mark.parametrize("doc", VALID, ids=["minimal", "everything"])
def test_valid_plans_satisfy_both(doc):
    assert _parser_accepts(doc)
    assert _schema_accepts(doc)


def _paths(value, prefix=()):
    yield prefix
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _paths(v, (*prefix, k))
    elif isinstance(value, list):
        for k, v in enumerate(value):
            yield from _paths(v, (*prefix, k))


def _mutate(doc: dict, rng: random.Random) -> dict:
    out = copy.deepcopy(doc)
    paths = [p for p in _paths(out) if p]
    path = rng.choice(paths)
    parent = out
    for key in path[:-1]:
        parent = parent[key]
    key = path[-1]
    kind = rng.randrange(5)
    if kind == 0 and isinstance(parent, dict):
        del parent[key]
    elif kind == 1:
        parent[key] = rng.choice([None, True, -1, 0.5, "text", [], {}, [1, 2], {"x": 1}])
    elif kind == 2 and isinstance(parent, dict):
        parent[f"unknown_{rng.randrange(9)}"] = 1
    elif kind == 3 and isinstance(parent[key], str):
        parent[key] = parent[key] + "?"
    else:
        parent[key] = copy.deepcopy(rng.choice(paths) and parent[key])
    return out


@needs_jsonschema
def test_random_mutations_never_split_the_checker_from_the_schema():
    rng = random.Random(11)
    checked = refused = 0
    for _ in range(400):
        doc = _mutate(VALID[1], rng)
        parser, schema = _parser_accepts(doc), _schema_accepts(doc)
        if parser:
            assert schema, json.dumps(doc)[:2000]
        if not schema:
            assert not parser, json.dumps(doc)[:2000]
            refused += 1
        checked += 1
    assert checked == 400 and refused > 50  # the mutations do break things


def test_mutated_plans_either_apply_or_fail_with_a_plan_error_and_change_nothing():
    """Execution never fails with anything but ``PlanError``, and a failure is atomic."""
    from aether3d import PlanLimits, Scene

    base = copy.deepcopy(VALID[1])
    base["textures"].pop("wood")  # an image file
    base.pop("expected_revision")
    rng = random.Random(5)
    applied = refused = 0
    for _ in range(200):
        doc = _mutate(base, rng)
        scene = Scene("Fuzz")
        scene.add_cube(name="Ground")
        before = scene.to_dict()
        try:
            scene.execute_plan(doc, limits=PlanLimits(max_objects=5000))
            applied += 1
        except PlanError:
            assert scene.to_dict() == before
            assert len(scene.history) == 1
            refused += 1
    assert applied > 50 and refused > 50
