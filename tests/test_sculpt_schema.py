"""The sculpt-script JSON Schema against the checker and executor: examples validate,
every mutated document the executor accepts is schema-valid, and no mutation crashes it
(only structured errors)."""

import copy
import json

import numpy as np
import pytest

from aether3d import AetherError, Scene, create_sphere
from aether3d.sculpt.schema import sculpt_script_schema
from aether3d.sculpt.script import run_sculpt_script

jsonschema = pytest.importorskip("jsonschema")

DOCUMENT = {
    "format": "aether3d.sculpt-script",
    "version": 1,
    "label": "Fuzzed",
    "limits": {"max_dabs": 500, "max_triangles": 100000, "max_seconds": 30},
    "sculpts": [
        {
            "object": "Ball",
            "require": "closed",
            "check": False,
            "scope": "object",
            "operations": [
                {"op": "set_mask", "value": 0.5, "mode": "add", "sphere": [[0, 0, -1], 0.5]},
                {
                    "op": "stroke",
                    "brush": {
                        "kind": "draw",
                        "radius": 0.3,
                        "strength": 0.2,
                        "falloff": [[0, 1], [0.5, 0.5], [1, 0]],
                        "symmetry": "x",
                        "automask": ["topology"],
                        "pressure": "both",
                    },
                    "path": [[0.2, -0.9, 0.1], {"position": [0.3, -0.9, 0.3], "pressure": 0.5}],
                    "space": "world",
                    "project": "nearest",
                },
                {
                    "op": "stroke",
                    "brush": {
                        "kind": "clay",
                        "radius": 0.3,
                        "dyntopo": {"detail": 0.3, "mode": "relative", "relax": 0.5},
                    },
                    "path": [[0, 0, 1]],
                },
                {"op": "invert_mask"},
                {"op": "clear_mask"},
                {"op": "remesh", "target_length": 0.2, "iterations": 1, "flips": "delaunay"},
                {"op": "symmetrize", "axis": "x", "direction": "-"},
                {"op": "check", "self_intersections": False},
                {"op": "undo"},
                {"op": "redo"},
                {"op": "voxel_remesh", "voxel_size": 0.15, "mode": "solid"},
            ],
        }
    ],
}


def scene() -> Scene:
    s = Scene()
    s.add_object("Ball", mesh=create_sphere(1.0, 16, 8))
    return s


def valid(doc) -> bool:
    return jsonschema.Draft202012Validator(sculpt_script_schema()).is_valid(doc)


def test_the_schema_is_a_valid_json_schema_and_the_example_runs():
    jsonschema.Draft202012Validator.check_schema(sculpt_script_schema())
    assert valid(DOCUMENT)
    out = run_sculpt_script(scene(), copy.deepcopy(DOCUMENT))
    assert out["sculpts"][0]["commit"]["changed"]
    json.dumps(out)  # plain JSON


def _mutations(document, rng, limit=300):
    replacements = [None, True, "x", -1, 0, 1e9, 0.5, [], {}, [1, 2, 3], "draw"]
    paths = []

    def walk(value, path):
        paths.append(path)
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, (*path, k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, (*path, i))

    walk(document, ())
    out = []
    for _ in range(limit):
        doc = copy.deepcopy(document)
        path = paths[int(rng.integers(len(paths)))]
        if not path:
            continue
        parent = doc
        for key in path[:-1]:
            parent = parent[key]
        choice = rng.integers(3)
        if choice == 0 and isinstance(parent, dict):
            del parent[path[-1]]
        elif choice == 1 and isinstance(parent, dict):
            parent["unknown_key"] = 1
        else:
            parent[path[-1]] = replacements[int(rng.integers(len(replacements)))]
        out.append(doc)
    return out


def test_mutations_never_crash_and_accepted_documents_are_schema_valid():
    rng = np.random.default_rng(13)
    accepted = rejected = 0
    for doc in _mutations(DOCUMENT, rng):
        try:
            run_sculpt_script(scene(), doc)
        except AetherError:
            rejected += 1
            continue
        accepted += 1
        assert valid(doc), json.dumps(doc)[:400]
    assert accepted > 10 and rejected > 100
