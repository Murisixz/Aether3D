"""The JSON Schema (draft 2020-12) of scene plans, ``aether3d.scene-plan`` version 1.

The schema documents the format and lets any JSON Schema tool check a plan's shape.
Aether3D's own checker (:func:`aether3d.plans.parse_plan`) enforces the same rules and
more (references, cycles, names, limits, expressions), with a JSON pointer and a stable
code for every problem; the tests check that both accept and refuse the same plans.
"""

from __future__ import annotations

import copy
from typing import Any

from .model import (
    CAMERA_FIELDS,
    FORMAT_ID,
    GENERATORS,
    LIGHT_FIELDS,
    MESH_PRIMITIVES,
    VARY_FIELDS,
    VERSION,
)

SCHEMA_ID = "urn:aether3d:schema:scene-plan:1"
SYMBOL_PATTERN = "^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$"


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def _numbers(count: int) -> dict[str, Any]:
    """An array of exactly ``count`` numbers."""
    return {"type": "array", "items": {"type": "number"}, "minItems": count, "maxItems": count}


def _obj(properties: dict[str, Any], required: tuple = (), **extra: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    schema.update(extra)
    return schema


def _symbols(value: dict[str, Any], description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "propertyNames": {"pattern": SYMBOL_PATTERN},
        "additionalProperties": value,
    }


def _defs() -> dict[str, Any]:
    num = _ref("number")
    vec = _ref("vec3")
    defs: dict[str, Any] = {
        "expr": _obj(
            {"expr": {"type": "string", "minLength": 1, "maxLength": 1000}},
            ("expr",),
            description="A safe expression over plan parameters (and, per instance, i, n, u, "
            "x, y, z, angle): numbers, + - * / // % **, comparisons, and/or/not, "
            "a if c else b, tuples, min max abs floor ceil round sqrt sin cos tan clamp "
            "int float (degrees), pi.",
        ),
        "number": {"oneOf": [{"type": "number"}, _ref("expr")]},
        "vec3": {
            "oneOf": [
                {"type": "array", "items": num, "minItems": 3, "maxItems": 3},
                _ref("expr"),
            ],
            "description": "[x, y, z] (components may be expressions) or one expression",
        },
        "scale": {"oneOf": [num, vec], "description": "uniform number or [x, y, z]"},
        "reference": {
            "oneOf": [
                {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 512,
                    "description": '"@symbol", "@symbol[index]", "@symbol/child" for what this '
                    "plan creates; otherwise an existing id or name",
                },
                _obj({"id": {"type": "string", "minLength": 1}}, ("id",)),
                _obj({"name": {"type": "string", "minLength": 1}}, ("name",)),
            ]
        },
        "color": {
            "oneOf": [
                {"type": "string", "pattern": "^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"},
                {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 3,
                    "maxItems": 3,
                },
            ],
            "description": "display (sRGB) color: #rrggbb, #rgb or [r, g, b] in 0..1",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 64},
            "maxItems": 64,
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "description": "object name; templates may use {i}, {i:3} (zero-padded), {n}, "
            "{x}, {y}, {z}, {parent}, {symbol}",
        },
    }
    defs["mesh"] = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "primitive": {"enum": list(MESH_PRIMITIVES)},
                    "size": {"oneOf": [num, vec]},
                    "radius": num,
                    "width": num,
                    "depth": num,
                    "height": num,
                    "segments": {"type": "integer", "minimum": 1, "maximum": 1024},
                    "rings": {"type": "integer", "minimum": 2, "maximum": 1024},
                    "radius_bottom": num,
                    "radius_top": num,
                    "caps": {"type": "boolean"},
                    "smooth": {"type": "boolean"},
                    "major_radius": num,
                    "minor_radius": num,
                    "major_segments": {"type": "integer", "minimum": 3, "maximum": 1024},
                    "minor_segments": {"type": "integer", "minimum": 3, "maximum": 1024},
                    "x_size": num,
                    "y_size": num,
                    "x_segments": {"type": "integer", "minimum": 1, "maximum": 1024},
                    "y_segments": {"type": "integer", "minimum": 1, "maximum": 1024},
                },
                "required": ["primitive"],
                "additionalProperties": False,
                "description": "a primitive (parameters as in aether3d procedural types; cube: "
                "size; box: size [x, y, z] or width/depth/height; sphere: radius, segments, rings)",
            },
            _obj(
                {
                    "data": _obj(
                        {
                            "positions": {"type": "array", "items": {"type": "array"}},
                            "triangles": {"type": "array", "items": {"type": "array"}},
                            "normals": {"type": "array"},
                            "uvs": {"type": "array"},
                            "groups": {"type": "array"},
                        },
                        ("positions", "triangles"),
                    )
                },
                ("data",),
            ),
            _obj({"from_object": _ref("reference")}, ("from_object",)),
        ]
    }
    sampler = {
        "wrap": {},
        "filter": {"enum": ["linear", "nearest"]},
        "mipmaps": {"type": "boolean"},
        "color_space": {"enum": ["auto", "srgb", "linear"]},
    }
    defs["texture"] = {
        "oneOf": [
            _obj(
                {
                    "name": {"type": "string"},
                    "image": {"type": "string", "description": "PNG/JPEG path inside the project"},
                    "embed": {"type": "boolean"},
                    **sampler,
                },
                ("image",),
            ),
            _obj(
                {
                    "name": {"type": "string"},
                    "procedural": {
                        "type": ["object", "string"],
                        "description": "a texture node graph or one node type",
                    },
                },
                ("procedural",),
            ),
        ]
    }
    defs["material"] = {
        "type": "object",
        "description": "preset (stone, brick, wood, metal, grass, fabric) with its options, or "
        "model (standard, simple) with parameters (base_color, roughness, metallic, ...); "
        "textures maps slots to bindings",
        "properties": {
            "name": {"type": "string"},
            "preset": {"enum": ["stone", "brick", "wood", "metal", "grass", "fabric"]},
            "model": {"enum": ["standard", "simple"]},
            "textures": {"type": "object", "additionalProperties": {"type": "object"}},
        },
        "additionalProperties": True,
    }
    defs["geometry"] = {
        "type": "object",
        "description": "a procedural geometry node: exactly one of primitive (+ params), source "
        "(+ overrides, after) and mesh; parameters; modifiers [{type, name, enabled, params}]",
        "properties": {
            "name": {"type": "string"},
            "primitive": {"type": "string"},
            "params": {"type": "object"},
            "source": _ref("reference"),
            "overrides": {"type": "object"},
            "after": {"type": ["string", "integer"]},
            "mesh": _ref("reference"),
            "parameters": {"type": "object"},
            "modifiers": {
                "type": "array",
                "items": _obj(
                    {
                        "type": {"type": "string"},
                        "name": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "params": {"type": "object"},
                    },
                    ("type",),
                ),
            },
        },
        "additionalProperties": False,
    }
    count = {"oneOf": [{"type": "integer", "minimum": 1}, _ref("expr")]}
    defs["patterns"] = {
        "grid": _obj(
            {
                "count": {"type": "array", "items": count, "minItems": 2, "maxItems": 3},
                "spacing": {
                    "oneOf": [
                        num,
                        {"type": "array", "items": num, "minItems": 2, "maxItems": 3},
                        _ref("expr"),
                    ]
                },
                "center": {"type": "boolean"},
            },
            ("count",),
        ),
        "array": _obj(
            {"count": count, "offset": vec, "rotation_step": vec, "scale_step": _ref("scale")},
            ("count",),
        ),
        "circle": _obj(
            {
                "count": count,
                "radius": num,
                "start_angle": num,
                "end_angle": num,
                "orient": {"enum": ["none", "outward", "inward", "tangent"]},
                "z": num,
            },
            ("count", "radius"),
        ),
        "line": _obj({"from": vec, "to": vec, "count": count}, ("from", "to", "count")),
        "scatter": _obj(
            {
                "count": count,
                "seed": {"type": "integer", "minimum": 0},
                "box": _obj(
                    {
                        "min": _numbers(2),
                        "max": _numbers(2),
                    },
                    ("min", "max"),
                ),
                "circle": _obj(
                    {
                        "center": _numbers(2),
                        "radius": {"type": "number", "minimum": 0},
                    },
                    ("radius",),
                ),
                "z": num,
                "min_distance": {"type": "number", "minimum": 0},
                "allow_fewer": {"type": "boolean"},
                "max_attempts": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            ("count",),
        ),
        "instances": {
            "oneOf": [
                {
                    "type": "array",
                    "minItems": 1,
                    "items": _obj(
                        {
                            "position": vec,
                            "rotation_euler": vec,
                            "scale": _ref("scale"),
                            "name": _ref("name"),
                            "color": _ref("color"),
                            "material": {"oneOf": [_ref("reference"), {"type": "null"}]},
                            "visible": {"type": "boolean"},
                            "tags": _ref("tags"),
                            "metadata": {"type": "object"},
                        }
                    ),
                },
                _obj(
                    {
                        "positions": {"type": "array", "items": vec, "minItems": 1},
                        "rotations_euler": {"type": "array", "items": vec},
                        "scales": {"type": "array", "items": _ref("scale")},
                    },
                    ("positions",),
                ),
            ]
        },
        "repeat": count,
    }
    generator = {
        "type": "object",
        "description": "exactly one of: cycle [values], choose [values] (+ weights, seed), "
        "uniform [low, high] (+ seed), linear [start, end], values [one per instance], "
        "expr (per instance)",
        "properties": {
            **{g: {} for g in GENERATORS},
            "expr": {"type": "string"},
            "seed": {"type": "integer", "minimum": 0},
            "weights": {"type": "array", "items": {"type": "number", "minimum": 0}},
        },
        "additionalProperties": False,
        "minProperties": 1,
    }
    defs["generator"] = generator
    patterns = defs.pop("patterns")
    defs["object"] = {
        "type": "object",
        "description": "one object, or many with a pattern (grid, array, circle, line, scatter, "
        "instances, repeat). At most one of mesh, geometry, prefab, camera, light (none: an "
        "empty group). children: objects created under each instance.",
        "properties": {
            "name": _ref("name"),
            "id": {"type": "string", "minLength": 1, "maxLength": 128},
            "mesh": {"oneOf": [_ref("reference"), _ref("mesh")]},
            "geometry": _ref("reference"),
            "prefab": _ref("reference"),
            "camera": _obj({k: {} for k in CAMERA_FIELDS}),
            "light": _obj({k: {} for k in LIGHT_FIELDS}),
            "parent": _ref("reference"),
            "position": vec,
            "rotation_euler": vec,
            "rotation": _numbers(4),
            "scale": _ref("scale"),
            "look_at": {"oneOf": [vec, _ref("reference")]},
            "look_along": vec,
            "direction": vec,
            "up": vec,
            "material": {"oneOf": [_ref("reference"), {"type": "null"}]},
            "group_materials": {
                "type": "object",
                "propertyNames": {"pattern": "^[0-9]+$"},
                "additionalProperties": {"oneOf": [_ref("reference"), {"type": "null"}]},
            },
            "color": _ref("color"),
            "specular": {"type": "number", "minimum": 0, "maximum": 1},
            "shininess": {"type": "number", "minimum": 1, "maximum": 10000},
            "visible": {"type": "boolean"},
            "tags": _ref("tags"),
            "metadata": {"type": "object"},
            "active": {"type": "boolean"},
            "frame": {
                "oneOf": [
                    {"const": True},
                    _obj({"targets": _ref("selector"), "direction": vec, "margin": num}),
                ]
            },
            "vary": {
                "type": "object",
                "propertyNames": {"enum": list(VARY_FIELDS)},
                "additionalProperties": _ref("generator"),
            },
            "where": {
                "allOf": [_ref("expr")],
                "description": "pattern instances for which this expression is false are not "
                "created (e.g. a door: not (x >= 4 and x <= 5 and z <= 2))",
            },
            "children": _symbols(_ref("object"), "child objects, by symbol"),
            **patterns,
        },
        "additionalProperties": False,
    }
    defs["selector"] = {
        "oneOf": [
            _ref("reference"),
            {"type": "array", "items": _ref("reference"), "minItems": 1},
            {
                "type": "object",
                "properties": {
                    "refs": {"type": "array", "items": {"type": "string"}},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "names": {"type": "array", "items": {"type": "string"}},
                    "tags": _ref("tags"),
                    "any_tags": _ref("tags"),
                    "name_glob": {"type": "string", "description": 'glob, e.g. "Wall*"'},
                    "kind": {"enum": ["mesh", "procedural", "camera", "light", "empty"]},
                    "under": _ref("reference"),
                    "visible": {"type": "boolean"},
                    "material": _ref("reference"),
                    "created": {"type": "boolean"},
                    "expect": {
                        "oneOf": [
                            {"type": "integer", "minimum": 0},
                            _obj(
                                {
                                    "min": {"type": "integer", "minimum": 0},
                                    "max": {"type": "integer", "minimum": 0},
                                }
                            ),
                        ]
                    },
                },
                "additionalProperties": False,
                "minProperties": 1,
            },
        ]
    }
    update_fields = {
        "name": {"type": "string"},
        "position": vec,
        "rotation_euler": vec,
        "rotation": _numbers(4),
        "scale": _ref("scale"),
        "translate": vec,
        "scale_by": _ref("scale"),
        "color": _ref("color"),
        "specular": {"type": "number", "minimum": 0, "maximum": 1},
        "shininess": {"type": "number", "minimum": 1, "maximum": 10000},
        "material": {"oneOf": [_ref("reference"), {"type": "null"}]},
        "visible": {"type": "boolean"},
        "tags": _ref("tags"),
        "add_tags": _ref("tags"),
        "remove_tags": _ref("tags"),
        "metadata": {"type": "object", "description": "entries to set; null removes one"},
        "parent": {"oneOf": [_ref("reference"), {"type": "null"}]},
        "keep_world_transform": {"type": "boolean"},
        "look_at": {"oneOf": [vec, _ref("reference")]},
        "look_along": vec,
        "up": vec,
    }
    defs["operation"] = {
        "oneOf": [
            _obj(
                {
                    "op": {"const": "update"},
                    "select": _ref("selector"),
                    "set": _obj(update_fields, minProperties=1),
                },
                ("op", "select", "set"),
            ),
            _obj(
                {
                    "op": {"const": "delete"},
                    "select": _ref("selector"),
                    "recursive": {"type": "boolean"},
                },
                ("op", "select"),
            ),
            _obj(
                {
                    "op": {"const": "set_parameters"},
                    "values": {"type": "object"},
                    "geometry": _ref("reference"),
                },
                ("op", "values"),
            ),
            _obj(
                {
                    "op": {"const": "frame_camera"},
                    "camera": _ref("reference"),
                    "targets": _ref("selector"),
                    "direction": vec,
                    "margin": num,
                },
                ("op",),
            ),
            _obj(
                {"op": {"const": "set_active_camera"}, "camera": _ref("reference")},
                ("op", "camera"),
            ),
            _obj(
                {
                    "op": {"const": "cad"},
                    "script": {
                        "type": "object",
                        "description": "an aether3d.cad-script (schema: "
                        "aether3d://schema/cad-script): sketches, parts, buildings, drawings",
                        "properties": {"format": {"const": "aether3d.cad-script"}},
                        "required": ["format", "version", "operations"],
                    },
                },
                ("op", "script"),
            ),
            _obj(
                {
                    "op": {"const": "animation"},
                    "script": {
                        "type": "object",
                        "description": "an aether3d.animation-script (schema: "
                        "aether3d://schema/animation-script): keys, clips, rigs, generators",
                        "properties": {"format": {"const": "aether3d.animation-script"}},
                        "required": ["format", "version", "operations"],
                    },
                },
                ("op", "script"),
            ),
        ]
    }
    return defs


def plan_schema() -> dict[str, Any]:
    """The JSON Schema of a scene plan (a fresh copy)."""
    return copy.deepcopy(_SCHEMA)


def _plan() -> dict[str, Any]:
    from ..spatial.request import QUERIES

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "Aether3D scene plan",
        "description": "A declarative, versioned description of scene changes, executed "
        "atomically by Aether3D (docs/SCENE_PLANS.md).",
        "type": "object",
        "properties": {
            "format": {"const": FORMAT_ID},
            "version": {"const": VERSION},
            "label": {"type": "string", "minLength": 1, "maxLength": 200},
            "description": {"type": "string", "maxLength": 4000},
            "expected_revision": {"type": ["integer", "null"], "minimum": 0},
            "idempotency_key": {"type": ["string", "null"], "minLength": 1, "maxLength": 128},
            "options": _obj(
                {
                    "on_name_conflict": {"enum": ["error", "rename"]},
                    "tags": _ref("tags"),
                    "result_ids": {"enum": ["auto", "all", "none"]},
                    "chunk_size": {"type": ["integer", "null"], "minimum": 100, "maximum": 1000000},
                    "on_chunk_failure": {"enum": ["keep", "rollback"]},
                    "timeout": {"type": ["number", "null"], "exclusiveMinimum": 0},
                }
            ),
            "parameters": {
                "type": "object",
                "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}$"},
                "additionalProperties": {
                    "oneOf": [
                        {"type": "number"},
                        {"type": "boolean"},
                        _numbers(3),
                        _ref("expr"),
                    ]
                },
            },
            "procedural": {
                "oneOf": [
                    _obj({"parameters": {"type": "object"}, "set": {"type": "object"}}),
                    {"type": "null"},
                ]
            },
            "meshes": _symbols(_ref("mesh"), "meshes shared by objects, by symbol"),
            "textures": _symbols(_ref("texture"), "textures, by symbol"),
            "materials": _symbols(_ref("material"), "materials, by symbol"),
            "geometries": _symbols(_ref("geometry"), "procedural geometry nodes, by symbol"),
            "prefabs": _symbols(
                _ref("object"),
                "reusable structures: an object definition (without placement) that objects "
                'instantiate with "prefab": "@symbol"',
            ),
            "objects": _symbols(_ref("object"), "objects to create, by symbol"),
            "operations": {"type": "array", "items": _ref("operation")},
            "world": {
                "type": ["object", "null"],
                "description": "background, ambient_color, ambient_intensity, environment",
            },
            "render_settings": {
                "type": ["object", "null"],
                "description": "width, height, samples, exposure, cull, default_lights",
            },
            "active_camera": {"oneOf": [_ref("reference"), {"type": "null"}]},
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"query": {"enum": list(QUERIES)}},
                    "required": ["query"],
                    "description": "a spatial query request as for aether3d query (plan "
                    'references "@symbol" become object ids)',
                },
            },
        },
        "required": ["format", "version"],
        "additionalProperties": False,
        "$defs": _defs(),
    }


_SCHEMA = _plan()
