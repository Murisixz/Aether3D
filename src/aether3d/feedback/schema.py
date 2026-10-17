"""JSON Schemas (draft 2020-12) of the feedback documents.

Hand-written, like the scene plan's: the checkers in :mod:`.expectations`,
:mod:`.vision` and :mod:`.corrections` are the authority, and the tests check that the
schemas accept what the checkers accept and refuse what they refuse on mutated documents
(``tests/test_feedback_schema.py``). The MCP server publishes them as resources.
"""

from __future__ import annotations

from typing import Any

from . import corrections, expectations, vision

DRAFT = "https://json-schema.org/draft/2020-12/schema"

_NUMBER = {"type": "number"}
_VEC3 = {"type": "array", "items": _NUMBER, "minItems": 3, "maxItems": 3}
_RECT = {"type": "array", "items": _NUMBER, "minItems": 4, "maxItems": 4}
_RANGE = {
    "type": "object",
    "properties": {"min": _NUMBER, "max": _NUMBER},
    "additionalProperties": False,
    "minProperties": 1,
}
_SELECTOR = {
    "description": "objects: an id or name, a list of them, or a selector object {ids, names, "
    "tags, any_tags, name_glob, kind, under, visible, material, expect}",
    "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        {"type": "object"},
    ],
}
_LENS = {
    "projection": {"enum": ["perspective", "orthographic"]},
    "fov": {"type": "number", "minimum": 0.01, "maximum": 179},
    "fov_axis": {"enum": ["vertical", "horizontal", "auto"]},
    "ortho_scale": {"type": "number", "minimum": 0},
    "near": {"type": "number", "minimum": 0},
    "far": {"type": "number", "minimum": 0},
}


def view_schema() -> dict[str, Any]:
    from .views import PRESETS

    name = {"type": "string", "minLength": 1, "maxLength": 128}
    return {
        "description": "a view: {camera} (a scene camera; {} is the active one), {virtual} "
        "(a camera that is not in the scene), {focus} (framed on objects) or {preset}",
        "type": "object",
        "oneOf": [
            {
                "properties": {"camera": {"type": ["string", "null"]}, "name": name},
                "additionalProperties": False,
            },
            {
                "properties": {
                    "name": name,
                    "virtual": {
                        "type": "object",
                        "properties": {
                            "position": _VEC3,
                            "look_at": _VEC3,
                            "direction": _VEC3,
                            "up": _VEC3,
                            **_LENS,
                        },
                        "required": ["position"],
                        "additionalProperties": False,
                    },
                },
                "required": ["virtual"],
                "additionalProperties": False,
            },
            {
                "properties": {
                    "name": name,
                    "focus": {
                        "type": "object",
                        "properties": {
                            "objects": _SELECTOR,
                            "direction": _VEC3,
                            "up": _VEC3,
                            "margin": {"type": "number", "minimum": 0, "maximum": 0.9},
                            "from_camera": {"type": "string"},
                            **_LENS,
                        },
                        "required": ["objects"],
                        "additionalProperties": False,
                    },
                },
                "required": ["focus"],
                "additionalProperties": False,
            },
            {
                "properties": {
                    "name": name,
                    "preset": {"enum": list(PRESETS)},
                    "objects": _SELECTOR,
                    "margin": {"type": "number", "minimum": 0, "maximum": 0.9},
                },
                "required": ["preset"],
                "additionalProperties": False,
            },
        ],
    }


def _check_schema(kind: str) -> dict[str, Any]:
    spec = expectations.CHECKS[kind]
    fields: dict[str, Any] = {
        "check": {"const": kind},
        "id": {"type": "string", "minLength": 1, "maxLength": 64},
        "severity": {"enum": list(expectations.SEVERITIES)},
        "label": {"type": "string", "maxLength": 300},
        "fix": {
            "type": "object",
            "properties": {
                "move": {"enum": list(expectations.MOVE_CHOICES)},
                "camera": {"type": "boolean"},
                "allow": {"type": "array", "items": {"enum": list(corrections.ACTIONS)}},
            },
            "additionalProperties": False,
        },
    }
    types = {
        "objects": _SELECTOR,
        "against": _SELECTOR,
        "support": _SELECTOR,
        "a": {"type": "string", "minLength": 1},
        "b": {"type": "string", "minLength": 1},
        "view": {"type": "string", "minLength": 1},
        "count": _RANGE,
        "min_pixels": {"type": "integer", "minimum": 0},
        "min_visible_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "in_frame": {"type": "boolean"},
        "fill": _RANGE,
        "height": _RANGE,
        "tolerance": {"type": "number", "minimum": 0},
        "allow_contact": {"type": "boolean"},
        "min": {"type": "number", "minimum": 0},
        "max": {"type": "number", "minimum": 0},
        "predicate": {"enum": list(expectations.RELATION_PREDICATES)},
        "expect": {"type": "boolean"},
        "near_distance": {"type": "number", "minimum": 0},
        "max_tilt": {"type": "number", "minimum": 0, "maximum": 180},
        "axis": _VEC3,
        "material": {"type": "string", "minLength": 1},
        "mean_luminance": _RANGE,
        "over": {"enum": ["objects", "image"]},
        "max_clipped_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "max_black_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "min_coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "tier": {"enum": list(expectations.MESH_TIERS), "default": "closed"},
        "resolution": {"type": "integer", "minimum": 16, "maximum": 16384, "default": 1024},
        "max_flipped": {"type": "integer", "minimum": 0, "default": 0},
        "max_overlapping": {"type": "integer", "minimum": 0, "default": 0},
        "max_stretch": {"type": "number", "minimum": 1},
        "max_density_spread": {"type": "number", "minimum": 0},
        "min_gap": {"type": "number", "minimum": 0, "description": "texels"},
        "items": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        "fully_constrained": {"type": "boolean"},
        "reference": {"type": "object", "description": "a CAD reference (docs/CAD.md)"},
        "other": {"type": "object", "description": "a second CAD reference"},
        "quantity": {"enum": list(expectations.CAD_QUANTITIES)},
        "value": {"type": "number"},
    }
    for key in spec["fields"]:
        if key == "region":
            fields[key] = (
                {**_RECT, "description": "normalized [x0, y0, x1, y1] of the image"}
                if kind == "framing"
                else {
                    "type": "object",
                    "properties": {
                        "box": {
                            "type": "object",
                            "properties": {"min": _VEC3, "max": _VEC3},
                            "required": ["min", "max"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["box"],
                    "additionalProperties": False,
                }
            )
        else:
            fields[key] = types[key]
    return {
        "type": "object",
        "description": spec["summary"],
        "properties": fields,
        "required": ["check", *spec["required"]],
        "additionalProperties": False,
    }


def expectations_schema() -> dict[str, Any]:
    """The JSON Schema of ``aether3d.expectations`` version 1."""
    return {
        "$schema": DRAFT,
        "$id": "https://aether3d.dev/schemas/expectations-1.json",
        "title": "Aether3D expectations",
        "type": "object",
        "properties": {
            "format": {"const": expectations.FORMAT},
            "version": {"const": expectations.VERSION},
            "label": {"type": "string", "minLength": 1, "maxLength": 200},
            "description": {"type": "string", "maxLength": 2000},
            "views": {
                "type": "object",
                "additionalProperties": view_schema(),
                "maxProperties": expectations.MAX_VIEWS,
            },
            "default_view": {"type": ["string", "null"]},
            "checks": {
                "type": "array",
                "items": {"oneOf": [_check_schema(k) for k in expectations.CHECKS]},
                "maxItems": expectations.MAX_CHECKS,
            },
            "acceptance": {
                "type": "object",
                "properties": {
                    "max_errors": {"type": "integer", "minimum": 0},
                    "max_warnings": {"type": ["integer", "null"], "minimum": 0},
                    "require_vision": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "defaults": {"type": "boolean"},
        },
        "required": ["format", "version"],
        "additionalProperties": False,
    }


def action_schema() -> dict[str, Any]:
    """One correction action (the allowlist)."""
    obj = {"type": "string", "minLength": 1}
    types: dict[str, Any] = {
        "object": obj,
        "camera": obj,
        "light": obj,
        "material": obj,
        "name": obj,
        "offset": _VEC3,
        "position": _VEC3,
        "direction": _VEC3,
        "rotation_euler": _VEC3,
        "rotation": {"type": "array", "items": _NUMBER, "minItems": 4, "maxItems": 4},
        "frame": {"enum": ["world", "local"]},
        "factor": {
            "anyOf": [
                {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
                {
                    "type": "array",
                    "items": {"type": "number", "exclusiveMinimum": 0},
                    "minItems": 3,
                    "maxItems": 3,
                },
            ]
        },
        "visible": {"type": "boolean"},
        "color": {"anyOf": [{"type": "string"}, _VEC3]},
        "targets": _SELECTOR,
        "margin": {"type": "number", "minimum": 0, "maximum": 0.9},
        "look_at": {"anyOf": [obj, _VEC3]},
        "distance": _NUMBER,
        "fov": {"type": "number", "minimum": 0.01, "maximum": 179},
        "near": {"type": "number", "exclusiveMinimum": 0},
        "far": {"type": "number", "exclusiveMinimum": 0},
        "intensity": {"type": "number", "minimum": 0, "maximum": 1e6},
        "type": {"enum": ["directional", "point"]},
        "exposure": {"type": "number", "minimum": -20, "maximum": 20},
        "value": {"type": ["number", "string", "boolean"]},
    }
    variants = []
    for name, spec in corrections.ACTIONS.items():
        variants.append(
            {
                "type": "object",
                "description": spec["summary"],
                "properties": {"action": {"const": name}, **{k: types[k] for k in spec["fields"]}},
                "required": ["action", *spec["required"]],
                "additionalProperties": False,
            }
        )
    return {"oneOf": variants}


def actions_schema() -> dict[str, Any]:
    return {
        "$schema": DRAFT,
        "$id": "https://aether3d.dev/schemas/correction-actions-1.json",
        "title": "Aether3D correction actions",
        "type": "array",
        "items": action_schema(),
        "minItems": 1,
        "maxItems": 100,
    }


def evaluation_schema() -> dict[str, Any]:
    """The JSON Schema of ``aether3d.vision-evaluation`` version 1."""
    return {
        "$schema": DRAFT,
        "$id": "https://aether3d.dev/schemas/vision-evaluation-1.json",
        "title": "Aether3D vision evaluation",
        "type": "object",
        "properties": {
            "format": {"const": vision.FORMAT},
            "version": {"const": vision.VERSION},
            "observation": {"type": "string", "minLength": 1, "maxLength": 64},
            "image_sha256": {"type": "string", "maxLength": 64},
            "provenance": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "minLength": 1, "maxLength": 64},
                    "model": {"type": "string", "minLength": 1, "maxLength": 128},
                    "image_input": {
                        "type": "boolean",
                        "description": "whether the model received the image (declared)",
                    },
                    "method": {"type": "string", "maxLength": 500},
                },
                "required": ["provider"],
                "additionalProperties": False,
            },
            "request": {"type": "string", "maxLength": 2000},
            "overall": {
                "type": "object",
                "properties": {
                    "matches_request": {"type": ["boolean", "null"]},
                    "score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "summary": {"type": "string", "maxLength": 2000},
                },
                "additionalProperties": False,
            },
            "findings": {
                "type": "array",
                "maxItems": vision.MAX_FINDINGS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "category": {"enum": list(vision.CATEGORIES)},
                        "severity": {"enum": list(vision.SEVERITIES)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "summary": {"type": "string", "minLength": 1, "maxLength": 500},
                        "region": {**_RECT, "description": "pixels [x0, y0, x1, y1)"},
                        "region_normalized": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "objects": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "maxItems": vision.MAX_OBJECTS,
                        },
                        "suggestion": action_schema(),
                    },
                    "required": ["category", "summary", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["format", "version", "observation", "provenance"],
        "additionalProperties": False,
    }


def policy_schema() -> dict[str, Any]:
    switches = ("verify", "rollback_on_regression", "rollback_if_ineffective")
    flags = {
        k: {"type": "boolean"}
        for k in corrections.CorrectionPolicy.FIELDS
        if k.startswith("allow_") or k in switches
    }
    return {
        "type": "object",
        "description": "what corrections may do and which apply without approval",
        "properties": {
            "auto_apply": {"type": "array", "items": {"enum": ["safe", "verify"]}, "maxItems": 2},
            **flags,
            "max_actions": {"type": "integer", "minimum": 1, "maximum": 1000},
            "max_objects": {"type": "integer", "minimum": 1, "maximum": 1000},
            "max_translation": {"type": ["number", "null"], "minimum": 0},
            "max_rotation": {"type": "number", "minimum": 0, "maximum": 180},
            "max_scale": {"type": "number", "minimum": 1, "maximum": 1000},
            "max_exposure_change": {"type": "number", "minimum": 0, "maximum": 20},
            "max_light_factor": {"type": "number", "minimum": 1, "maximum": 1000},
            "min_vision_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    }


def budget_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "bounds of a feedback loop",
        "properties": {
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 100},
            "max_renders": {"type": "integer", "minimum": 1, "maximum": 10000},
            "max_pixels": {"type": "integer", "minimum": 1},
            "max_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 86400},
            "max_memory": {"type": "integer", "minimum": 1048576},
            "max_critic_calls": {"type": "integer", "minimum": 0, "maximum": 100},
            "max_cost": {"type": "number", "minimum": 0},
            "patience": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "additionalProperties": False,
    }
