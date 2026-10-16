"""The JSON Schema (draft 2020-12) of sculpt scripts, built from the checkers' constants.

The checker (:func:`~aether3d.sculpt.script.check_script` and the brush, stroke and
operation validators it reaches) is the authority; the schema describes the same
documents for agents and editors, and ``tests/test_sculpt_schema.py`` checks that the
two agree on generated documents.
"""

from __future__ import annotations

from typing import Any

from .brushes import AUTOMASKS, FALLOFFS, KINDS, PRESSURE_TARGETS, SMOOTH_METHODS
from .script import FORMAT_ID, LIMIT_KEYS, VERSION
from .validate import TIERS
from .voxel import MAX_VOXELS

_NUMBER = {"type": "number"}
_VEC3 = {"type": "array", "items": _NUMBER, "minItems": 3, "maxItems": 3}
_POSITIVE = {"type": "number", "exclusiveMinimum": 0}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """An optional argument: ``null`` means not given."""
    return {"oneOf": [schema, {"type": "null"}]}


def _brush() -> dict[str, Any]:
    falloff_points = {
        "type": "array",
        "items": {"type": "array", "items": _NUMBER, "minItems": 2, "maxItems": 2},
        "minItems": 2,
        "maxItems": 64,
        "description": "a custom curve: [x, y] points, x from 0 to 1 increasing, y in [0, 1]",
    }
    dyntopo = {
        "type": "object",
        "description": "dynamic topology inside the brush (not for grab, elastic, layer, mask)",
        "properties": {
            "detail": {
                **_POSITIVE,
                "description": "target edge length: a fraction of the radius (relative, "
                "0.01..1) or world units (constant)",
            },
            "mode": {"enum": ["relative", "constant"], "default": "relative"},
            "refine": {"enum": ["subdivide", "collapse", "both"], "default": "both"},
            "flip": {"type": "boolean", "default": True},
            "relax": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "max_splits": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
        },
        "required": ["detail"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "kind": {"enum": list(KINDS)},
            "radius": {**_POSITIVE, "description": "world units"},
            "strength": {"type": "number", "minimum": 0, "maximum": 10, "default": 0.5},
            "direction": {"enum": [1, -1], "default": 1},
            "falloff": {"oneOf": [{"enum": list(FALLOFFS)}, falloff_points]},
            "hardness": {"type": "number", "minimum": 0, "maximum": 0.99, "default": 0},
            "spacing": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 10,
                "default": 0.1,
                "description": "dab spacing as a fraction of the radius",
            },
            "normal": {"enum": ["area", "vertex", "view"], "default": "area"},
            "normal_radius": {"type": "number", "minimum": 0.01, "maximum": 2, "default": 1},
            "falloff_shape": {"enum": ["sphere", "projected"], "default": "sphere"},
            "geodesic": {"type": "boolean", "default": False},
            "front_faces_only": {"type": "boolean", "default": False},
            "automask": {"type": "array", "items": {"enum": list(AUTOMASKS)}, "uniqueItems": True},
            "use_mask": {"type": "boolean", "default": True},
            "symmetry": {
                "type": "string",
                "pattern": "^(?!.*(.).*\\1)[xyz]{0,3}$",
                "description": "mirror axes (local planes through the origin), e.g. 'x'",
            },
            "pressure": {"enum": list(PRESSURE_TARGETS), "default": "strength"},
            "smooth_method": {"enum": list(SMOOTH_METHODS), "default": "laplacian"},
            "poisson_ratio": {"type": "number", "minimum": 0, "maximum": 0.499, "default": 0.4},
            "dyntopo": {"oneOf": [dyntopo, {"type": "null"}]},
        },
        "required": ["kind", "radius"],
        "additionalProperties": False,
    }


def _path() -> dict[str, Any]:
    sample = {
        "oneOf": [
            _VEC3,
            {
                "type": "object",
                "properties": {
                    "position": _VEC3,
                    "pressure": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["position"],
                "additionalProperties": False,
            },
        ]
    }
    return {"type": "array", "items": sample, "minItems": 1, "maxItems": 100_000}


def _operations() -> list[dict[str, Any]]:
    def op(name: str, props: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"op": {"const": name}, **props},
            "required": ["op", *required],
            "additionalProperties": False,
        }

    region = {  # null: not given
        "vertices": _nullable({"type": "array", "items": {"type": "integer", "minimum": 0}}),
        "sphere": _nullable(
            {
                "type": "array",
                "prefixItems": [_VEC3, {"type": "number", "minimum": 0}],
                "minItems": 2,
                "maxItems": 2,
            }
        ),
        "box": _nullable({"type": "array", "items": _VEC3, "minItems": 2, "maxItems": 2}),
        "groups": _nullable({"type": "array", "items": {"type": "integer"}}),
    }
    return [
        op(
            "stroke",
            {
                "brush": _brush(),
                "path": _path(),
                "space": {"enum": ["world", "local"], "default": "world"},
                "project": {"enum": ["nearest", "ray", "none"], "default": "nearest"},
                "view": _nullable(_VEC3),
                "resample": {"type": "boolean", "default": True},
            },
            ("brush", "path"),
        ),
        op(
            "set_mask",
            {
                "value": {"type": "number", "minimum": 0, "maximum": 1, "default": 1},
                "mode": {"enum": ["set", "add", "subtract"], "default": "set"},
                **region,
            },
        ),
        op("invert_mask", {}),
        op("clear_mask", {}),
        op(
            "remesh",
            {
                "target_length": {**_POSITIVE, "description": "world units"},
                "iterations": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "flips": {"enum": ["valence", "delaunay", "none"], "default": "valence"},
                "relax": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "project": {"type": "boolean", "default": True},
            },
            ("target_length",),
        ),
        op(
            "voxel_remesh",
            {
                "voxel_size": {**_POSITIVE, "description": "world units"},
                "mode": {"enum": ["solid", "shell"], "default": "solid"},
                "thickness": _nullable(
                    {**_POSITIVE, "description": "shell only, at least two voxels"}
                ),
                "max_voxels": _nullable({"type": "integer", "minimum": 512, "default": MAX_VOXELS}),
            },
            ("voxel_size",),
        ),
        op(
            "symmetrize",
            {
                "axis": {"enum": ["x", "y", "z"], "default": "x"},
                "direction": {"enum": ["+", "-"], "default": "+"},
                "mode": {"enum": ["auto", "positions", "mirror"], "default": "auto"},
                "tolerance": _nullable({"type": "number", "minimum": 0}),
            },
        ),
        op("undo", {}),
        op("redo", {}),
        op("check", {"self_intersections": {"type": "boolean", "default": False}}),
    ]


def sculpt_script_schema() -> dict[str, Any]:
    """The JSON Schema of ``aether3d.sculpt-script`` version 1 documents."""
    obj_ref = {
        "oneOf": [
            {"type": "string", "minLength": 1},
            {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                "minProperties": 1,
                "maxProperties": 1,
                "additionalProperties": False,
            },
        ]
    }
    limits = {
        "type": "object",
        "properties": {
            key: (
                {"type": "integer", "minimum": 1}
                if key != "max_seconds"
                else {**_POSITIVE, "description": "wall-clock seconds"}
            )
            for key in LIMIT_KEYS
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "aether3d://schema/sculpt-script",
        "title": "Aether3D sculpt script",
        "description": "Brush strokes, masks, remeshing and symmetrize on objects' meshes, "
        "applied as one undo step (all or nothing). See docs/MODELING.md.",
        "type": "object",
        "properties": {
            "format": {"const": FORMAT_ID},
            "version": {"const": VERSION},
            "label": {"type": "string", "minLength": 1, "pattern": "\\S"},
            "limits": limits,
            "sculpts": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "object": obj_ref,
                        "require": {"enum": list(TIERS)},
                        "check": {"type": "boolean", "default": False},
                        "scope": {"enum": ["object", "shared"], "default": "object"},
                        "operations": {"type": "array", "items": {"oneOf": _operations()}},
                    },
                    "required": ["object", "operations"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["format", "version", "sculpts"],
        "additionalProperties": False,
    }
