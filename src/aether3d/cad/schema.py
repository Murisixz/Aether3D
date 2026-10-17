"""The JSON Schema (draft 2020-12) of CAD scripts, built from the checker's tables.

The checker (:func:`~aether3d.cad.script.check_script` and the item kinds' own checks)
is the authority; the schema describes the same documents for agents and editors: the
operations and their arguments, the item kinds and the shapes of their definitions
(sketch entities and constraints, part features, assembly mates, building elements,
drawings). ``tests/test_cad_script.py`` checks that the scripts the executor accepts are
schema-valid.
"""

from __future__ import annotations

from typing import Any

from .assembly import MATE_TYPES
from .drawing.kind import VIEW_NAMES
from .features import FEATURE_TYPES
from .features import OPERATIONS as FEATURE_OPERATIONS
from .measure import SNAP_KINDS
from .script import FORMAT_ID, LIMIT_KEYS, LISTS, OPERATIONS, VERSION
from .sketch.model import CONSTRAINT_TYPES, ENTITY_TYPES
from .units import ANGLE_UNITS, LENGTH_UNITS

_STRING = {"type": "string", "minLength": 1}
_ID = {"type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[^:|#/ ]+$"}
_LENGTH = {
    "oneOf": [{"type": "number"}, {"type": "string", "minLength": 1}],
    "description": "a number in the document's unit, a quantity ('3/8 in', '2 m') or "
    "'=expression' over parameters",
}
_POINT2 = {"type": "array", "items": _LENGTH, "minItems": 2, "maxItems": 2}
_POINT3 = {"type": "array", "items": _LENGTH, "minItems": 2, "maxItems": 3}
_REF = {
    "type": "object",
    "description": "{part, face|edge|vertex[, body, near]}, {sketch, point|entity} or "
    "{point: [x, y, z]}",
    "properties": {
        "part": _STRING,
        "body": _STRING,
        "face": _STRING,
        "edge": {
            "oneOf": [_STRING, {"type": "array", "items": _STRING, "minItems": 2, "maxItems": 2}]
        },
        "vertex": _POINT3,
        "near": _POINT3,
        "sketch": _STRING,
        "entity": _STRING,
        "point": {"oneOf": [_STRING, _POINT3]},
    },
    "additionalProperties": False,
}

_ENTITY = {
    "type": "object",
    "properties": {
        "id": _ID,
        "type": {"enum": list(ENTITY_TYPES)},
        "at": {**_POINT2, "description": "point position (document unit in scripts)"},
        "start": _ID,
        "end": _ID,
        "center": _ID,
        "radius": _LENGTH,
        "degree": {"type": "integer", "minimum": 1, "maximum": 5},
        "points": {"type": "array", "items": _ID},
        "construction": {"type": "boolean"},
    },
    "required": ["id", "type"],
}
_CONSTRAINT = {
    "type": "object",
    "properties": {
        "id": _ID,
        "type": {"enum": list(CONSTRAINT_TYPES)},
        "refs": {"type": "array", "items": _ID, "minItems": 1, "maxItems": 3},
        "value": {**_LENGTH, "description": "lengths, or angles ('30 deg')"},
        "driving": {"type": "boolean"},
        "along": {"enum": ["x", "y"]},
        "side": {"type": "integer"},
    },
    "required": ["id", "type", "refs"],
}
_PLANE = {
    "anyOf": [
        {"enum": ["XY", "XZ", "YZ"]},
        {"type": "object", "description": "{origin, z, x}"},
        {**_STRING, "description": "a frame item's name"},
    ]
}
DEFINITIONS: dict[str, dict[str, Any]] = {
    "parameter": {
        "type": "object",
        "properties": {
            "value": _LENGTH,
            "dimension": {"enum": ["length", "angle", "ratio", "area", "volume"]},
        },
        "required": ["value"],
    },
    "frame": {"type": "object", "description": "{origin, z, x} or {base, offset, rotate}"},
    "sketch": {
        "type": "object",
        "properties": {
            "plane": _PLANE,
            "entities": {"type": "array", "items": _ENTITY},
            "constraints": {"type": "array", "items": _CONSTRAINT},
        },
    },
    "part": {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _ID,
                        "type": {"enum": list(FEATURE_TYPES)},
                        "operation": {"enum": list(FEATURE_OPERATIONS)},
                        "body": _STRING,
                        "suppressed": {"type": "boolean"},
                    },
                    "required": ["id", "type"],
                },
            },
            "materials": {"type": "object", "additionalProperties": {"type": "string"}},
            "color": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        },
    },
    "assembly": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _ID,
                        "source": _STRING,
                        "fixed": {"type": "boolean"},
                        "placement": {"type": "object"},
                    },
                    "required": ["id", "source"],
                },
            },
            "mates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _ID,
                        "type": {"enum": [*MATE_TYPES, "fixed"]},
                        "a": {"type": "object"},
                        "b": {"type": "object"},
                        "offset": _LENGTH,
                        "value": _LENGTH,
                        "flip": {"type": "boolean"},
                    },
                    "required": ["id", "type"],
                },
            },
        },
    },
    "building": {
        "type": "object",
        "properties": {
            "levels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": _STRING, "elevation": _LENGTH, "height": _LENGTH},
                    "required": ["name", "elevation", "height"],
                },
            },
            "wall_types": {"type": "object"},
            "families": {"type": "object"},
            "criteria": {"type": "object"},
            "grids": {"type": "array"},
            "site": {"type": "object"},
            "materials": {"type": "object"},
        },
        "required": ["levels"],
    },
    "floor": {
        "type": "object",
        "properties": {
            "building": _STRING,
            "level": _STRING,
            "walls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _ID,
                        "start": _POINT2,
                        "end": _POINT2,
                        "type": _STRING,
                        "line": {"type": "object"},
                        "height": _LENGTH,
                        "base_offset": _LENGTH,
                        "location": {"enum": ["center", "left", "right"]},
                    },
                    "required": ["id"],
                },
            },
            "openings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _ID,
                        "wall": _ID,
                        "position": _LENGTH,
                        "family": _STRING,
                        "flip": {"type": "boolean"},
                    },
                    "required": ["id", "wall", "position", "family"],
                },
            },
            "rooms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _ID,
                        "name": _STRING,
                        "number": {"type": "string"},
                        "point": _POINT2,
                    },
                    "required": ["id", "name", "point"],
                },
            },
        },
        "required": ["building", "level"],
    },
    "slab": {"type": "object", "required": ["building", "level"]},
    "roof": {
        "type": "object",
        "required": ["building", "level", "type"],
        "properties": {"type": {"enum": ["flat", "shed", "gable", "hip"]}},
    },
    "stair": {"type": "object", "required": ["building", "from", "to", "start", "direction"]},
    "columns": {"type": "object", "required": ["building", "from"]},
    "beams": {"type": "object", "required": ["building", "level", "lines"]},
    "drawing": {
        "type": "object",
        "properties": {
            "view": {"enum": list(VIEW_NAMES)},
            "items": {"type": "array", "items": _STRING, "minItems": 1},
            "level": _STRING,
            "cut_height": _LENGTH,
            "plane": {"type": "object"},
            "hidden": {"type": "boolean"},
            "scale": {"type": ["string", "null"], "pattern": "^[0-9]+:[0-9]+$"},
            "sheet": {"enum": ["A4", "A3", "A2", "A1"]},
            "dimensions": {"oneOf": [{"enum": ["auto", "none"]}, {"type": "array"}]},
            "title": {"type": "string"},
        },
        "required": ["view", "items"],
    },
}


SCHEDULE_KINDS = (
    "all",
    "walls",
    "doors",
    "windows",
    "rooms",
    "slab",
    "roof",
    "stair",
    "columns",
    "beams",
)
OP_PROPERTIES = {"schedule": {"kind": {"enum": list(SCHEDULE_KINDS)}}}


def _properties() -> dict[str, Any]:
    return {
        "length": {"enum": list(LENGTH_UNITS)},
        "angle": {"enum": list(ANGLE_UNITS)},
        "name": _STRING,
        "value": _LENGTH,
        "dimension": {"enum": ["length", "angle", "ratio", "area", "volume"]},
        "kind": {"enum": list(DEFINITIONS)},
        "definition": {"type": "object"},
        "item": _STRING,
        "changes": {"type": "object"},
        "list": {"enum": sorted({x for v in LISTS.values() for x in v})},
        "values": {"type": "array", "minItems": 1},
        "id": {"oneOf": [_STRING, {"type": "array"}]},
        "ids": {"type": "array", "minItems": 1},
        "object": _STRING,
        "a": _REF,
        "b": _REF,
        "sketch": _STRING,
        "x": _LENGTH,
        "y": _LENGTH,
        "radius": _LENGTH,
        "grid": _LENGTH,
        "kinds": {"type": "array", "items": {"enum": list(SNAP_KINDS)}},
        "building": _STRING,
        "checks": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 100,
            "description": "dotted path in the measured or info data -> expected value, or "
            "{value, tolerance} / {min, max} / {equals}",
        },
        "measure": {
            "type": "object",
            "properties": {"a": _REF, "b": _REF},
            "required": ["a"],
            "additionalProperties": False,
        },
        "info": _STRING,
        "message": _STRING,
    }


def cad_script_schema() -> dict[str, Any]:
    """The JSON Schema of ``aether3d.cad-script`` version 1 documents."""
    props = _properties()
    operations = []
    for op, (required, optional) in OPERATIONS.items():
        names = sorted(required | optional)
        own = OP_PROPERTIES.get(op, {})
        schema: dict[str, Any] = {
            "type": "object",
            "title": op,
            "properties": {
                "op": {"const": op},
                **{k: own.get(k, props.get(k, {})) for k in names},
            },
            "required": ["op", *sorted(required)],
            "additionalProperties": False,
        }
        if op == "add":
            schema["allOf"] = [
                {
                    "if": {"properties": {"kind": {"const": kind}}},
                    "then": {"properties": {"definition": defn}},
                }
                for kind, defn in DEFINITIONS.items()
            ]
        operations.append(schema)
    limits = {
        "type": "object",
        "properties": {
            k: {"type": "integer", "minimum": 1}
            if k != "max_seconds"
            else {"type": "number", "exclusiveMinimum": 0, "description": "wall-clock seconds"}
            for k in LIMIT_KEYS
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "aether3d://schema/cad-script",
        "title": "Aether3D CAD script",
        "description": "Parametric sketches (constraints, dimensions), parts (features), "
        "assemblies (mates), buildings (levels, walls, openings, rooms, slabs, roofs, "
        "stairs, columns, beams) and drawings, edited as data and applied as one undo step "
        "(all or nothing); measurements and checks. Scripts never read or write files. "
        "See docs/CAD.md.",
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
