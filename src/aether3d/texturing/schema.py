"""The JSON Schema (draft 2020-12) of texture scripts, built from the checker's tables.

The checker (:func:`~aether3d.texturing.script.check_script` and the operations it runs) is
the authority; the schema describes the same documents for agents and editors: which
operations exist, their required and optional arguments, and the types and ranges of the
common ones. ``tests/test_texture_script.py`` checks that every generated document the
executor accepts is schema-valid.
"""

from __future__ import annotations

from typing import Any

from ..materials.material import SLOTS
from ..materials.painted import BLEND_MODES, ENCODINGS, LAYER_KINDS, MAX_LAYERS
from ..uv import PROJECTIONS
from ..uv.charts import METHODS, SPLIT_BY
from ..uv.pack import MAX_RESOLUTION
from .bake import MAPS
from .paint import BRUSH_KINDS, FALLOFFS, MAX_POINTS, SPACES
from .script import _EDIT_UVS, FORMAT_ID, LIMIT_KEYS, OPERATIONS, VERSION

_NUMBER = {"type": "number"}
_POSITIVE = {"type": "number", "exclusiveMinimum": 0}
_UNIT = {"type": "number", "minimum": 0, "maximum": 1}
_BOOL = {"type": "boolean"}
_STRING = {"type": "string", "minLength": 1}
_INTS = {"type": "array", "items": {"type": "integer"}}
_COLOR = {
    "oneOf": [
        {"type": "string", "pattern": "^#?[0-9a-fA-F]{6}$"},
        {"type": "array", "items": _UNIT, "minItems": 1, "maxItems": 4},
    ],
    "description": "'#rrggbb' (display sRGB) or 1-4 numbers in [0, 1] (stored encoding)",
}
_OBJECT = {
    "oneOf": [
        _STRING,
        {
            "type": "object",
            "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": False,
        },
    ],
    "description": "object name, or {id} / {name}",
}
_RESOLUTION = {
    "oneOf": [
        {"type": "integer", "minimum": 1, "maximum": MAX_RESOLUTION},
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": MAX_RESOLUTION},
            "minItems": 2,
            "maxItems": 2,
        },
    ]
}


def _brush() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"enum": list(BRUSH_KINDS), "default": "paint"},
            "radius": {**_POSITIVE, "description": "object units (surface), pixels (view), "
                       "texels (uv)"},
            "strength": _UNIT,
            "opacity": {**_UNIT, "description": "the most a stroke can lay"},
            "hardness": _UNIT,
            "falloff": {"enum": list(FALLOFFS)},
            "spacing": {"type": "number", "minimum": 0.01, "maximum": 10},
            "color": _COLOR,
            "blend": {"enum": list(BLEND_MODES)},
            "normal_limit": {"type": "number", "minimum": 0, "maximum": 180},
            "backfaces": _BOOL,
            "symmetry": {"type": "string", "pattern": "^(?!.*(.).*\\1)[xyz]{0,3}$"},
            "clone_offset": {"type": "array", "items": _NUMBER, "minItems": 2, "maxItems": 2},
            "blur_radius": {"type": "integer", "minimum": 1, "maximum": 16},
        },
        "additionalProperties": False,
    }  # fmt: skip


def _properties() -> dict[str, Any]:
    """Schemas of argument names that mean the same in every operation that takes them
    (names whose shape depends on the operation, such as ``rotate``, are left open)."""
    layer = {
        "type": "object",
        "properties": {
            "name": _STRING,
            "kind": {"enum": list(LAYER_KINDS)},
            "blend": {"enum": list(BLEND_MODES)},
            "opacity": _UNIT,
            "visible": _BOOL,
            "value": _COLOR,
        },
        "additionalProperties": False,
    }
    return {
        "object": _OBJECT,
        "target": _OBJECT,
        "source": _OBJECT,
        "camera": _OBJECT,
        "sources": {"type": "array", "items": _OBJECT, "minItems": 1},
        "texture": {**_STRING, "description": "a texture id or name (never a file)"},
        "layer": {**_STRING, "description": "a layer id or name"},
        "name": _STRING,
        "material": _STRING,
        "projection": {"enum": list(PROJECTIONS)},
        "operation": {"enum": list(_EDIT_UVS)},
        "method": {"enum": list(METHODS)},
        "split_by": {
            "oneOf": [
                {"enum": list(SPLIT_BY)},
                {"type": "array", "items": {"enum": list(SPLIT_BY)}, "uniqueItems": True},
            ]
        },
        "resolution": _RESOLUTION,
        "width": {"type": "integer", "minimum": 1, "maximum": MAX_RESOLUTION},
        "height": {"type": "integer", "minimum": 1, "maximum": MAX_RESOLUTION},
        "encoding": {"enum": list(ENCODINGS)},
        "background": _COLOR,
        "layers": {"type": "array", "items": layer, "maxItems": MAX_LAYERS},
        "index": {"type": "integer", "minimum": 0},
        "kind": {"enum": list(LAYER_KINDS)},
        "blend": {"enum": list(BLEND_MODES)},
        "opacity": _UNIT,
        "visible": _BOOL,
        "value": _COLOR,
        "color": _COLOR,
        "colors": {"type": "array", "items": _COLOR, "minItems": 2, "maxItems": 2},
        "start": {"type": "array", "items": _NUMBER, "minItems": 3, "maxItems": 3},
        "end": {"type": "array", "items": _NUMBER, "minItems": 3, "maxItems": 3},
        "slot": {"enum": list(SLOTS)},
        "channel": {"enum": ["r", "g", "b", "a", "rgb"]},
        "brush": _brush(),
        "points": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_POINTS,
            "items": {"type": "array", "items": _NUMBER, "minItems": 2, "maxItems": 3},
            "description": "object-space [x, y, z] (surface), pixel [x, y] (view) or [u, v] (uv)",
        },
        "pressure": {"type": "array", "items": _UNIT},
        "space": {"enum": list(SPACES), "default": "surface"},
        "coordinates": {"enum": ["world", "object"], "default": "world"},
        "view_size": {
            "type": "array",
            "items": {"type": "integer", "minimum": 16, "maximum": 4096},
            "minItems": 2,
            "maxItems": 2,
        },
        "where": {
            "type": "object",
            "properties": {k: _INTS for k in ("islands", "groups", "triangles")},
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": False,
        },
        "targets": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"texture": _STRING, "layer": _STRING, "color": _COLOR},
                "required": ["texture"],
                "additionalProperties": False,
            },
        },
        "maps": {"type": "array", "items": {"enum": list(MAPS)}, "minItems": 1},
        "apply": _BOOL,
        "unwrap": {"enum": ["auto", True, False]},
        "own_material": _BOOL,
        "samples": {"type": "integer", "minimum": 1, "maximum": 8},
        "padding": {"type": "integer", "minimum": 0},
        "cage": _POSITIVE,
        "depth": _POSITIVE,
        "ao_rays": {"type": "integer", "minimum": 1},
        "thickness_rays": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0},
        "backend": {"enum": ["auto", "cpu", "gpu"]},
        "angle_limit": _NUMBER,
        "max_stretch": _NUMBER,
        "iterations": {"type": "integer", "minimum": 0},
        "texels_per_unit": _POSITIVE,
        "islands": _INTS,
        "faces": _INTS,
        "pack": _BOOL,
        "keep_aspect": _BOOL,
        "keep_density": _BOOL,
        "allow_overlap": _BOOL,
        "islands_limit": {"type": "integer", "minimum": 0},
    }


def texture_script_schema() -> dict[str, Any]:
    """The JSON Schema of ``aether3d.texture-script`` version 1 documents."""
    props = _properties()
    operations = []
    for op, (required, optional) in OPERATIONS.items():
        names = sorted(required | optional)
        operations.append(
            {
                "type": "object",
                "title": op,
                "properties": {"op": {"const": op}, **{k: props.get(k, {}) for k in names}},
                "required": ["op", *sorted(required)],
                "additionalProperties": False,
            }
        )
    limits = {
        "type": "object",
        "properties": {
            k: (
                {"type": "integer", "minimum": 1}
                if k != "max_seconds"
                else {**_POSITIVE, "description": "wall-clock seconds"}
            )
            for k in LIMIT_KEYS
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "aether3d://schema/texture-script",
        "title": "Aether3D texture script",
        "description": "UV unwrapping and editing, painted textures and layers, brush "
        "strokes, fills, gradients, bakes and texture transfer, applied as one undo step "
        "(all or nothing). Scripts never read or write files. See docs/TEXTURING.md.",
        "type": "object",
        "properties": {
            "format": {"const": FORMAT_ID},
            "version": {"const": VERSION},
            "label": {"type": "string", "minLength": 1, "pattern": "\\S"},
            "limits": limits,
            "operations": {"type": "array", "minItems": 1, "items": {"oneOf": operations}},
        },
        "required": ["format", "version", "operations"],
        "additionalProperties": False,
    }
