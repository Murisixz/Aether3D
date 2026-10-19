"""The JSON Schema (draft 2020-12) of animation scripts, built from the checker's tables.

The checker (:func:`~aether3d.animation.script.check_script` and the methods each
operation reaches) is the authority; the schema describes the same documents for agents
and editors (every operation, its required and optional arguments), and
``tests/test_animation_script.py`` checks that the two agree on generated documents.
"""

from __future__ import annotations

from typing import Any

from .clip import KEY_OPERATIONS
from .curves import EXTRAPOLATIONS, HANDLE_TYPES, INTERPOLATIONS, SMOOTHING
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
    "meshes": {
        "description": "mesh objects (one or a list): the character's body, clothing, parts",
        "oneOf": [_REF, {"type": "array", "items": _REF, "minItems": 1, "maxItems": 256}],
    },
    "context": {"type": "array", "items": _REF, "description": "more meshes voxelized with it"},
    "template": {"enum": ["humanoid", "stylized", "blocky", "quadruped"]},
    "landmarks": {
        "type": "object",
        "description": "landmark overrides {name: [x, y, z] world}: hip.L, knee.L, ankle.L, "
        "shoulder.L, elbow.L, wrist.L, neck, head, pelvis, ... (see detect_landmarks)",
    },
    "parts": {
        "type": "object",
        "description": "per mesh name, how it binds: generate (weights.method, default heat), "
        "heat or geodesic (that method), transfer (from the first generated part), rigid, "
        "attach (parented to its nearest joint) or skip; the joints are measured on the "
        "generated parts only (the body: clothing and props do not move them)",
        "additionalProperties": {
            "enum": ["generate", "heat", "geodesic", "transfer", "rigid", "attach", "skip"]
        },
    },
    "operations": {
        "type": "array",
        "description": "edit_rig: {op: add_joint|remove_joint|move_joint|orient_joint|"
        "rename_joint|reparent_joint|insert_joint|split_bone|mirror_joints|align_bone|set_joint, "
        "...}; edit_weights: {op: assign|add|subtract|scale|replace|normalize|prune|cap|smooth|"
        "mirror|rigid|lock|unlock, select, falloff, ...}",
        "items": {"type": "object"},
        "minItems": 1,
    },
    "commands": {
        "type": "array",
        "description": "{do: reach|plant_feet|look_at|raise_arm|lean|crouch|aim_hips|rotate|"
        "curl_fingers|roll_foot, ...}: raise_arm {side, angle, direction: forward|side, "
        "shoulder: auto|0..1}; curl_fingers {side, shape: curl|fist|point, amount, fingers, "
        "spread}; roll_foot {side, angle (<0 heel, >0 ball then toe), break, ground}",
        "items": {"type": "object"},
        "minItems": 1,
    },
    "fixes": {
        "description": "proposals from propose_rig_fix, or 'safe' / 'verify'",
        "oneOf": [{"enum": ["safe", "verify"]}, {"type": "array", "items": {"type": "object"}}],
    },
    "root": _REF,
    "roles": {
        "type": "object",
        "description": "rigid parts by role: {head|torso|arm.L|arm.R|leg.L|leg.R: object} "
        "(default: found by name under the root, sides measured)",
        "additionalProperties": _REF,
    },
    "to": {"enum": ["skeleton", "rigid"]},
    "hide_original": {"type": "boolean"},
    "policy": {"enum": ["organic", "stylized", "rigid"]},
    "skinning": {"enum": ["linear", "dual_quaternion", "blended"]},
    "method": {
        "type": "string",
        "description": "skin: linear | dual_quaternion | blended; generate_weights: heat (the "
        "default: bone heat) | geodesic | envelope | nearest",
    },
    "normals": {
        "enum": ["joints", "surface"],
        "description": "joints: rest normals turned by the joints (glTF's); surface: turned "
        "as the deformed surface turns",
    },
    "corrective": {
        "type": ["object", "boolean", "null"],
        "description": "corrective smoothing (delta mush) for the skin: {iterations 1..100, "
        "strength (0, 1], region: blend | all, weights: cotangent | uniform}, true for the "
        "defaults, null for none",
    },
    "smooth": {
        "description": "skin: corrective smoothing (delta mush), as generate_weights' "
        "corrective, or null; generate_weights and transfer_weights: smoothing passes over "
        "the weights",
    },
    "diffusion": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": "heat weights: the Laplacian's scale (1: Blender's balance)",
    },
    "visibility": {
        "type": "boolean",
        "description": "heat weights: only bones a vertex sees heat it (the default)",
    },
    "poses": {
        "description": "diagnostic poses: suite (the everyday range), stress (the extremes), "
        "or a list of suite names and custom {name, turns, joint} poses",
        "oneOf": [
            {"enum": ["suite", "stress"]},
            {"type": "array", "maxItems": 64},
            {"type": "null"},
        ],
    },
    "kind": {"enum": ["ik", "pole", "gaze", "fk", "root"]},
    "skins": {"enum": ["rebind", "keep_bind"]},
    "edit": {"enum": list(KEY_OPERATIONS)},
    "smoothing": {
        "enum": list(SMOOTHING),
        "description": "automatic handles: continuous (Blender's Continuous Acceleration, "
        "the default for new curves: acceleration continuous at automatic keys), blender "
        "(Blender's handles without smoothing), local (Aether3D's own rule before 0.20.0)",
    },
    "keep_shape": {"type": "boolean", "description": "insert: a key that keeps the curve"},
    "blend_curve": {
        "enum": ["linear", "smooth", "smoother"],
        "description": "strip: the shape of its blend ramps (smooth: C1, smoother: C2 "
        "crossfades; linear: velocity jumps where a ramp starts and ends)",
    },
    "ease_in": {"type": "number", "minimum": 0, "maximum": 1},
    "ease_out": {"type": "number", "minimum": 0, "maximum": 1},
    "amount": {"type": "number", "minimum": 0, "maximum": 10},
    "at": {"type": "number", "minimum": 0, "maximum": 0.9},
    "settle": {"type": "number", "exclusiveMinimum": 0},
    "oscillations": {"type": "integer", "minimum": 1, "maximum": 16},
    "decay": {"type": "number", "minimum": 0, "maximum": 1},
    "duration": {"type": "number", "exclusiveMinimum": 0},
    "drift": {"type": "number", "minimum": 0, "maximum": 1},
    "shift": {"type": "boolean"},
    "sigma": {"type": "number", "exclusiveMinimum": 0},
    "keep_ends": {"type": "boolean"},
    "measure": {
        "type": "array",
        "maxItems": 512,
        "description": "motion: objects (names, ids) or {object, joint | joints: [...] | "
        '"all"} or {camera: "shots"}; default: every animated object, rig and the shots',
        "items": {"oneOf": [{"type": "string"}, {"type": "object"}]},
    },
    "subject": {
        "description": "motion: what a camera should frame: an object or a point [x, y, z]",
    },
    "loop": {"type": "boolean", "description": "motion: compare the range's end with its start"},
    "thresholds": {
        "type": "object",
        "description": "motion: thresholds to override (velocity_jump, acceleration_step, "
        "position_jump, jitter_hz, jitter_mm, jitter_deg, snap_deg, angular_jump, pause_speed, "
        "pause_min, pause_max, window, ground, ground_tolerance, frame_margin, drift)",
    },
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
