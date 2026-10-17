"""The JSON Schema (draft 2020-12) of animation scripts, built from the checker's tables.

The checker (:func:`~aether3d.animation.script.check_script` and the methods each
operation reaches) is the authority; the schema describes the same documents for agents
and editors (every operation, its required and optional arguments), and
``tests/test_animation_script.py`` checks that the two agree on generated documents.
"""

from __future__ import annotations

from typing import Any

from .curves import EXTRAPOLATIONS, HANDLE_TYPES, INTERPOLATIONS
from .easing import MODES
from .generators import KINDS
from .script import FORMAT_ID, LIMIT_KEYS, MAX_OPERATIONS, OPERATIONS, VERSION

_NUMBER = {"type": "number"}
_TIME = {"type": "number", "description": "seconds"}
_REF = {
    "description": "an object by name, or {object: name or id} / {material: name or id}",
    "oneOf": [
        {"type": "string", "minLength": 1},
        {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "properties": {
                "object": {"type": "string"},
                "material": {"type": "string"},
                "id": {"type": "string"},
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ],
}
_PATH = {
    "type": "string",
    "description": "an animatable path: transform.position(.x), transform.rotation, "
    'transform.rotation_euler.z, joints["NAME"].rotation, morph["NAME"], camera.fov, '
    "light.intensity, appearance.color, material.roughness, visible, "
    'constraints["NAME"].influence, ...',
}
_KEYS = {
    "type": "array",
    "description": "keys: [time, value] pairs or {time, value, interpolation, left, right, ...}",
    "items": {"oneOf": [{"type": "array", "minItems": 2, "maxItems": 2}, {"type": "object"}]},
}

_ARGUMENTS: dict[str, Any] = {
    "target": _REF,
    "object": _REF,
    "armature": _REF,
    "camera": _REF,
    "owner": _REF,
    "mesh": _REF,
    "source": _REF,
    "path": _PATH,
    "time": _TIME,
    "start": _TIME,
    "end": _TIME,
    "keys": _KEYS,
    "interpolation": {"enum": list(INTERPOLATIONS) + ["constant", "slerp", "squad", "cubic"]},
    "ease": {"enum": list(MODES)},
    "handles": {"enum": list(HANDLE_TYPES)},
    "extrapolation": {
        "oneOf": [
            {"enum": list(EXTRAPOLATIONS)},
            {
                "type": "array",
                "items": {"enum": list(EXTRAPOLATIONS)},
                "minItems": 2,
                "maxItems": 2,
            },
        ]
    },
    "motion": {"enum": list(KINDS)},
    "modifiers": {
        "type": "array",
        "description": "F-curve modifiers: {type: offset|scale|limits|noise|wave|envelope|"
        "smooth|stepped|time_offset|time_scale, ...parameters, influence, range, mute}",
        "items": {"type": "object"},
        "maxItems": 16,
    },
    "fps": {"description": "frames per second: a number or a fraction like '30000/1001'"},
    "tolerance": {"type": "number", "exclusiveMinimum": 0},
    "clip": {"type": ["string", "null"], "description": "a clip name"},
    "name": {"type": "string", "minLength": 1, "maxLength": 128},
    "times": {"type": "array", "items": _NUMBER, "minItems": 1, "maxItems": 10_000},
    "targets": {"type": "array", "items": _REF, "minItems": 1},
    "feet": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "joints": {
        "type": "array",
        "description": "rig joints: {name, parent, position, rotation, euler, scale, tail, limits}",
    },
    "edit": {
        "enum": [
            "set",
            "delete",
            "handle",
            "handle_type",
            "interpolation",
            "move",
            "scale_time",
            "scale_values",
            "reverse",
            "quantize",
            "distribute",
            "retime",
        ],
    },  # fmt: skip
}


def _operation(op: str) -> dict[str, Any]:
    _, required, optional, read_only = OPERATIONS[op]
    props: dict[str, Any] = {"op": {"const": op}}
    for key in (*required, *optional):
        props[key] = _ARGUMENTS.get(key, {})
    return {
        "type": "object",
        "description": ("read-only: " if read_only else "") + op.replace("_", " "),
        "properties": props,
        "required": ["op", *required],
        "additionalProperties": False,
    }


def animation_script_schema() -> dict[str, Any]:
    """The JSON Schema of ``aether3d.animation-script`` documents."""
    limits = {
        "type": "object",
        "properties": {
            "max_seconds": {"type": "number", "exclusiveMinimum": 0},
            "max_keys": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    }
    assert set(limits["properties"]) == set(LIMIT_KEYS)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "aether3d://schema/animation-script",
        "title": "Aether3D animation script",
        "description": "Ordered animation operations applied atomically (one undo step).",
        "type": "object",
        "properties": {
            "format": {"const": FORMAT_ID},
            "version": {"const": VERSION},
            "label": {"type": "string", "minLength": 1, "maxLength": 200},
            "limits": limits,
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPERATIONS,
                "items": {"oneOf": [_operation(op) for op in OPERATIONS]},
            },
        },
        "required": ["format", "version", "operations"],
        "additionalProperties": False,
    }
