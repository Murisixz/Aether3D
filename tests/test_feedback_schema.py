"""The feedback documents' JSON Schemas agree with their checkers (Stage 12).

The checkers are the authority; the schemas describe them for clients. On valid documents
both accept; on up to 600 mutations of each (removed keys, unknown keys, wrong types,
values out of range) every document the schema refuses the checker refuses too -- the schema
never promises more than the engine accepts. (The checker also refuses what a schema
cannot express: min above max, unknown view names, duplicate ids.)
"""

from __future__ import annotations

import contextlib
import copy
import random

import pytest

from aether3d import FeedbackError, Scene
from aether3d.feedback import check_actions, check_evaluation, observe, parse_expectations
from aether3d.feedback.schema import actions_schema, evaluation_schema, expectations_schema

jsonschema = pytest.importorskip("jsonschema")

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "Everything",
    "description": "every check kind",
    "views": {
        "main": {"camera": "Cam"},
        "top": {"preset": "top", "objects": {"tags": ["furniture"]}, "margin": 0.1},
        "close": {"focus": {"objects": "Lamp", "direction": [0, 1, -0.3], "fov": 30}},
        "free": {"virtual": {"position": [0, -6, 2], "look_at": [0, 0, 0.5], "near": 0.05}},
    },
    "default_view": "main",
    "checks": [
        {"check": "exists", "objects": {"tags": ["chair"]}, "count": {"min": 2, "max": 6}},
        {
            "check": "visible",
            "objects": ["Lamp", "Sofa"],
            "view": "main",
            "min_pixels": 10,
            "min_visible_fraction": 0.5,
            "in_frame": True,
            "severity": "warning",
        },
        {
            "check": "framing",
            "objects": "Sofa",
            "region": [0.2, 0.2, 0.8, 0.8],
            "fill": {"min": 0.1},
            "height": {"max": 0.7},
        },
        {
            "check": "rests_on",
            "objects": "Vase",
            "support": "Table",
            "tolerance": 0.001,
            "fix": {"move": "objects", "allow": ["translate"]},
        },
        {
            "check": "no_overlap",
            "objects": {"tags": ["furniture"]},
            "against": "Wall",
            "tolerance": 0.0001,
            "allow_contact": True,
            "fix": {"move": "objects", "camera": False},
        },
        {
            "check": "inside_region",
            "objects": "Rug",
            "region": {"box": {"min": [-3, -3, 0], "max": [3, 3, 2]}},
        },
        {"check": "distance", "a": "Lamp", "b": "Sofa", "min": 0.2, "max": 1.5},
        {
            "check": "relation",
            "a": "Lamp",
            "b": "Table",
            "predicate": "near",
            "near_distance": 1.0,
            "expect": True,
        },
        {"check": "upright", "objects": "Lamp", "max_tilt": 3, "axis": [0, 0, 1]},
        {"check": "material", "objects": "Rug", "material": "Wool", "label": "rug"},
        {
            "check": "exposure",
            "view": "main",
            "mean_luminance": {"min": 0.1, "max": 0.6},
            "over": "image",
            "max_clipped_fraction": 0.02,
            "max_black_fraction": 0.3,
            "min_coverage": 0.2,
        },
        {"check": "camera_clipping", "view": "close", "objects": "Lamp", "id": "clip"},
        {"check": "mesh_validity", "objects": {"tags": ["furniture"]}, "tier": "solid"},
    ],
    "acceptance": {"max_errors": 0, "max_warnings": 3, "require_vision": False},
    "defaults": True,
}

ACTIONS = [
    {"action": "translate", "object": "Vase", "offset": [0, 0, -0.1], "frame": "world"},
    {"action": "place", "object": "Vase", "position": [0, 0, 1], "frame": "local"},
    {"action": "rotate", "object": "Vase", "rotation_euler": [0, 0, 10]},
    {"action": "rotate", "object": "Vase", "rotation": [0, 0, 0, 1]},
    {"action": "scale", "object": "Vase", "factor": 1.5},
    {"action": "scale", "object": "Vase", "factor": [1, 1, 2]},
    {"action": "set_visibility", "object": "Vase", "visible": False},
    {"action": "assign_material", "object": "Vase", "material": "Clay"},
    {"action": "set_color", "object": "Vase", "color": "#336699"},
    {"action": "frame_camera", "camera": "Cam", "targets": {"tags": ["a"]}, "margin": 0.1},
    {"action": "aim_camera", "camera": "Cam", "look_at": [0, 0, 1]},
    {"action": "aim_camera", "look_at": "Vase"},
    {"action": "dolly_camera", "camera": "Cam", "distance": -1},
    {"action": "set_camera", "camera": "Cam", "fov": 35, "near": 0.1, "far": 100},
    {"action": "set_light", "light": "Sun", "intensity": 2, "color": "#ffffff"},
    {"action": "add_light", "type": "directional", "direction": [0, 0, -1], "intensity": 1},
    {"action": "add_light", "type": "point", "name": "Fill", "position": [0, 0, 3]},
    {"action": "set_exposure", "exposure": -1},
    {"action": "set_ambient", "intensity": 0.2, "color": [1, 1, 1]},
    {"action": "set_parameter", "name": "count", "value": 3},
]


def _mutations(document, rng, limit=400):
    """Deterministic mutations of every part of a JSON document."""
    replacements = [None, True, "x", -1, 1e9, 0.5, [], {}, [1, 2, 3]]
    paths = []

    def walk(value, path):
        paths.append(path)
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, (*path, k))
        elif isinstance(value, list):
            for k, v in enumerate(value):
                walk(v, (*path, k))

    walk(document, ())
    out = []
    for path in paths:
        parent = document
        for key in path[:-1]:
            parent = parent[key]
        target = parent if not path else parent[path[-1]]
        for r in replacements:
            if not path:
                continue
            m = copy.deepcopy(document)
            p = m
            for key in path[:-1]:
                p = p[key]
            p[path[-1]] = copy.deepcopy(r)
            out.append(m)
        if isinstance(target, dict):
            for k in list(target):
                m = copy.deepcopy(document)
                p = m
                for key in path:
                    p = p[key]
                del p[k]
                out.append(m)
            m = copy.deepcopy(document)
            p = m
            for key in path:
                p = p[key]
            p["zzz_unknown"] = 1
            out.append(m)
    rng.shuffle(out)
    return out[:limit]


def _agree(valid, check, schema, rng, limit=400):
    validator = jsonschema.Draft202012Validator(schema)
    assert not list(validator.iter_errors(valid))
    check(valid)
    refused = 0
    for m in _mutations(valid, rng, limit):
        if list(validator.iter_errors(m)):
            refused += 1
            with pytest.raises(FeedbackError):
                check(m)
        else:  # the schema allows it: if the checker refuses, it is a semantic rule
            with contextlib.suppress(FeedbackError):
                check(m)
    assert refused > limit // 4  # the mutations do exercise the schema


def test_expectations_schema_agrees_with_the_checker():
    jsonschema.Draft202012Validator.check_schema(expectations_schema())
    _agree(EXPECTATIONS, parse_expectations, expectations_schema(), random.Random(12), 600)


def test_actions_schema_agrees_with_the_checker():
    jsonschema.Draft202012Validator.check_schema(actions_schema())
    _agree(ACTIONS, check_actions, actions_schema(), random.Random(13), 600)


def test_evaluation_schema_agrees_with_the_checker():
    scene = Scene("S")
    scene.add_cube("Crate", position=(0, 0, 0.5))
    scene.add_camera("Cam", position=(0, -5, 1), look_at=(0, 0, 0.5))
    obs = observe(scene, width=64, height=36)
    valid = {
        "format": "aether3d.vision-evaluation",
        "version": 1,
        "observation": obs.id,
        "image_sha256": obs.sha256,
        "provenance": {
            "provider": "fixture",
            "model": "none",
            "image_input": False,
            "method": "written by the test",
        },
        "request": "a crate",
        "overall": {"matches_request": True, "score": 0.8, "confidence": 0.5, "summary": "ok"},
        "findings": [
            {
                "id": "a",
                "category": "placement",
                "severity": "warning",
                "confidence": 0.7,
                "summary": "s",
                "region": [1, 1, 30, 30],
                "objects": ["Crate"],
                "suggestion": {"action": "translate", "object": "Crate", "offset": [0, 0, 1]},
            },
            {
                "category": "lighting",
                "confidence": 0.2,
                "summary": "t",
                "region_normalized": [0, 0, 0.5, 0.5],
            },
        ],
    }
    jsonschema.Draft202012Validator.check_schema(evaluation_schema())
    _agree(valid, lambda d: check_evaluation(d, obs), evaluation_schema(), random.Random(14), 600)
