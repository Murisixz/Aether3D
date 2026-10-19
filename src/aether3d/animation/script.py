"""Declarative animation scripts: ordered JSON operations, safe to accept from untrusted input.

A script is plain data -- no code is ever evaluated, no attribute is looked up by name
from the document, and every operation maps to one whitelisted method with checked
arguments::

    {
      "format": "aether3d.animation-script",
      "version": 1,
      "label": "Animate the hero",               # optional: the undo step's name
      "limits": {"max_seconds": 60, "max_keys": 500000},
      "operations": [
        {"op": "timeline", "fps": 24, "start": 0, "end": 4},
        {"op": "keyframe", "target": "Ball", "path": "transform.position",
         "time": 0, "value": [0, 0, 1], "interpolation": "bezier"},
        {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
         "keys": [[0, 1], [0.5, 0], [1, 1]], "interpolation": "linear",
         "extrapolation": "cycle"},
        {"op": "edit_keys", "target": "Ball", "path": "transform.position.z",
         "edit": "scale_time", "factor": 0.5},
        {"op": "generate", "armature": "Hero", "motion": "walk", "cadence": 2.0},
        {"op": "camera_orbit", "camera": "Cam", "target": "Hero", "start": 0, "end": 4},
        {"op": "report", "target": "Ball", "path": "transform.position.z"}
      ]
    }

Objects are named (``"Hero"``), or given as ``{"object": NAME_OR_ID}``; materials as
``{"material": NAME_OR_ID}``. Operations (see ``guide.md`` and the schema for every
argument): ``timeline``, ``set_time``, ``markers``, ``add_marker``, ``sections``,
``add_clip``, ``remove_clip``, ``rename_clip``, ``clip_range``, ``clip_events``,
``keyframe``, ``set_channel``, ``edit_keys``, ``set_action``, ``set_binding``,
``add_layer``, ``update_layer``, ``remove_layer``, ``add_strip``, ``update_strip``,
``remove_strip``, ``add_rig``, ``set_rig``, ``attach``, ``detach``, ``skin``, ``unbind``,
``add_constraint``, ``set_constraints``, ``remove_constraint``, ``morph_targets``,
``add_path``, ``remove_path``, ``follow_path``, ``shots``, ``add_shot``, ``camera_move``,
``camera_orbit``, ``follow_camera``, ``generate``, ``set_modifiers``, ``simplify``,
``resample``, ``bake``,
``mirror_clip``, ``extract_root_motion``, ``plant_feet``, ``retarget``, ``apply_pose``;
rigging (Stage 18): ``autorig``, ``edit_rig``, ``add_joint``, ``remove_joint``,
``move_joint``, ``orient_joints``, ``rename_joint``, ``reparent_joint``,
``mirror_joint_chain``, ``generate_weights``, ``edit_weights``, ``normalize_weights``,
``transfer_weights``, ``add_control``, ``ik_fk``, ``pose_character``, ``apply_rig_fix``;
rigid-part characters (no bones): ``generate_rigid``, ``convert_character``;
read-only: ``report``, ``trajectory``, ``diagnose``, ``contacts``, ``sample``,
``detect_landmarks``, ``inspect_weights``, ``diagnose_rig``, ``propose_rig_fix``,
``rigid_parts``.

:func:`run_animation_script` applies everything or nothing, as one step of the scene's
history: if an operation fails, a limit is reached or the run is cancelled, the scene is
exactly as before (the playhead too). Errors carry a stable ``code``, the ``location``
of the operation's argument (``operations[3].keys[2]``) and a hint.
"""

from __future__ import annotations

import inspect
import math
import time
from typing import TYPE_CHECKING, Any

from ..errors import AetherError

if TYPE_CHECKING:
    from ..scene import Scene

FORMAT_ID = "aether3d.animation-script"
VERSION = 1
MAX_OPERATIONS = 10_000
LIMIT_KEYS = ("max_seconds", "max_keys")
DEFAULT_LIMITS = {"max_seconds": 300.0, "max_keys": 5_000_000}


class AnimationScriptError(AetherError, ValueError):
    """A script is malformed or one of its operations failed: ``code`` (the underlying
    error's, or ``INVALID_SCRIPT``), ``location`` and ``details`` (``hint``, ...)."""

    def __init__(
        self, message: str, location: str, code: str = "INVALID_SCRIPT", details: dict | None = None
    ):
        self.message = message
        self.location = location
        self.code = code
        self.details = details or {}
        super().__init__(f"{location}: [{code}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            **self.details,
        }


# --------------------------------------------------------------------------------------- #
# The operation table: op -> (method, required, optional, read_only)
# --------------------------------------------------------------------------------------- #

_KEY_OPTIONS = (
    "interpolation", "ease", "left", "right", "weighted", "back", "amplitude", "period",
    "handles", "path", "timing",
)  # fmt: skip
_CURVE_OPTIONS = ("interpolation", "handles", "extrapolation", "modifiers", "ease")
_STRIP_OPTIONS = (
    "name", "start", "clip_start", "clip_end", "speed", "repeat", "reverse", "blend_in",
    "blend_out", "before", "after", "weight", "weight_curve", "time_curve", "accumulate",
    "mute",
)  # fmt: skip
_LAYER_CHANGES = ("name", "mode", "weight", "weight_curve", "mask", "mute", "index")
_EDIT_ARGUMENTS = (
    "keys", "time", "value", "interpolation", "ease", "back", "amplitude", "period",
    "weighted", "side", "type", "dt", "dv", "slope", "factor", "pivot", "velocity", "fps",
    "start", "end", "map",
)  # fmt: skip

OPERATIONS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], bool]] = {
    "timeline": ("set_timeline", (), ("fps", "start", "end"), False),
    "set_time": ("_set_time", ("time",), (), False),
    "markers": ("set_markers", ("markers",), (), False),
    "add_marker": ("add_marker", ("name", "time"), ("data",), False),
    "sections": ("set_sections", ("sections",), (), False),
    "add_clip": ("add_clip", ("name",), ("data",), False),
    "remove_clip": ("remove_clip", ("clip",), (), False),
    "rename_clip": ("_rename_clip", ("clip", "name"), (), False),
    "clip_range": ("_clip_range", ("clip", "range"), (), False),
    "clip_events": ("_clip_events", ("clip", "events"), (), False),
    "keyframe": ("keyframe", ("path", "time", "value"), ("target", "clip", *_KEY_OPTIONS), False),
    "set_channel": ("set_channel", ("path", "keys"), ("target", "clip", *_CURVE_OPTIONS), False),
    "edit_keys": ("edit_keys", ("path", "edit"), ("target", "clip", *_EDIT_ARGUMENTS), False),
    "set_action": ("set_action", ("target", "clip"), (), False),
    "set_binding": ("set_binding", ("target", "data"), (), False),
    "add_layer": ("add_layer", ("target", "name"), ("mode", "weight", "mask", "index"), False),
    "update_layer": ("update_layer", ("target", "name"), _LAYER_CHANGES, False),
    "remove_layer": ("remove_layer", ("target", "name"), (), False),
    "add_strip": ("add_strip", ("target", "layer", "clip"), _STRIP_OPTIONS, False),
    "update_strip": (
        "update_strip",
        ("target", "layer", "index"),
        ("clip", *_STRIP_OPTIONS),
        False,
    ),
    "remove_strip": ("remove_strip", ("target", "layer", "index"), (), False),
    "add_rig": ("add_rig", ("armature", "joints"), ("attachments", "constraints", "mirror"), False),
    "set_rig": ("set_rig", ("armature", "data"), (), False),
    "attach": ("attach", ("object", "armature", "joint"), (), False),
    "detach": ("detach", ("object",), (), False),
    "skin": (
        "skin",
        ("mesh", "armature"),
        ("weights", "joints", "influences", "method", "dq", "normals", "smooth"),
        False,
    ),
    "unbind": ("unbind", ("mesh",), (), False),
    "add_constraint": ("add_constraint", ("owner", "constraint"), (), False),
    "set_constraints": ("set_constraints", ("owner", "constraints"), (), False),
    "remove_constraint": ("remove_constraint", ("owner", "name"), (), False),
    "morph_targets": ("set_morph_targets", ("object", "targets"), (), False),
    "add_path": (
        "add_path",
        ("name",),
        ("points", "kind", "closed", "handles", "center", "radius", "normal", "start"),
        False,
    ),  # fmt: skip
    "remove_path": ("remove_path", ("path",), (), False),
    "follow_path": (
        "follow_path",
        ("object", "path", "start", "end"),
        ("offsets", "easing", "ease", "follow", "axis", "up", "loop", "name"),
        False,
    ),  # fmt: skip
    "shots": ("set_shots", ("shots",), (), False),
    "add_shot": ("add_shot", ("camera", "start", "end"), ("name", "blend"), False),
    "camera_move": (
        "camera_move",
        ("camera", "move", "amount", "start", "end"),
        ("easing", "ease"),
        False,
    ),  # fmt: skip
    "camera_orbit": (
        "camera_orbit",
        ("camera", "target", "start", "end"),
        ("degrees", "easing", "ease", "name"),
        False,
    ),  # fmt: skip
    "follow_camera": (
        "follow_camera",
        ("camera", "target"),
        ("start", "end", "offset", "lag", "aim", "tolerance", "name"),
        False,
    ),  # fmt: skip
    "generate": (
        "generate",
        ("armature", "motion"),
        (
            "name",
            "start",
            "mapping",
            "forward",
            "travel",
            "tolerance",
            "bind",
            "amplitude",
            "period",
            "cadence",
            "stride",
            "duty",
            "lift",
            "gravity",
            "height",
            "distance",
            "style",
            "knee",
            "side",
            "waves",
            "rate",
            "yaw",
            "pitch",
            "turn",
            "hold",
            "back",
        ),
        False,
    ),  # fmt: skip
    "set_modifiers": ("set_modifiers", ("path", "modifiers"), ("target", "clip"), False),
    "simplify": (
        "simplify",
        ("path", "tolerance"),
        ("target", "clip", "density", "max_keys"),
        False,
    ),
    "resample": ("resample", ("path",), ("target", "clip", "fps", "start", "end"), False),
    "bake": ("bake", ("targets",), ("start", "end", "fps", "joints", "tolerance", "name"), False),
    "mirror_clip": ("mirror_clip", ("clip", "armature"), ("name",), False),
    "extract_root_motion": ("extract_root_motion", ("clip", "joint"), ("mode", "yaw"), False),
    "plant_feet": (
        "plant_feet",
        ("armature", "legs"),
        ("start", "end", "rate", "ground", "height", "speed", "blend", "pole"),
        False,
    ),  # fmt: skip
    "retarget": (
        "retarget",
        ("source", "target", "clip"),
        ("mapping", "root", "fps", "start", "end", "name"),
        False,
    ),  # fmt: skip
    "apply_pose": ("apply_pose", (), ("time",), False),
    # Stage 18: rigging.
    "autorig": (
        "autorig",
        ("meshes",),
        (
            "armature",
            "name",
            "template",
            "landmarks",
            "up",
            "forward",
            "resolution",
            "symmetry",
            "orientation",
            "twist",
            "controls",
            "bind",
            "parts",
            "weights",
            "skinning",
            "enforce_limits",
            "dry_run",
        ),
        False,
    ),  # fmt: skip
    "edit_rig": (
        "edit_rig",
        ("armature", "operations"),
        ("skins", "animation", "channels", "tolerance", "dry_run"),
        False,
    ),
    "add_joint": (
        "add_joint",
        ("armature", "name"),
        (
            "parent",
            "position",
            "space",
            "rotation",
            "euler",
            "scale",
            "tail",
            "role",
            "deform",
            "limits",
        ),
        False,
    ),  # fmt: skip
    "remove_joint": (
        "remove_joint",
        ("armature", "joint"),
        ("weights_to", "constraints", "channels"),
        False,
    ),  # fmt: skip
    "move_joint": (
        "move_joint",
        ("armature", "joint", "position"),
        ("space", "children", "skins"),
        False,
    ),  # fmt: skip
    "orient_joints": (
        "orient_joints",
        ("armature",),
        ("joints", "aim", "up", "up_vector", "roll", "animation", "tolerance"),
        False,
    ),
    "rename_joint": ("rename_joint", ("armature", "joint", "name"), (), False),
    "reparent_joint": ("reparent_joint", ("armature", "joint", "parent"), ("keep",), False),
    "mirror_joint_chain": ("mirror_joint_chain", ("armature",), ("side",), False),
    "generate_weights": (
        "generate_weights",
        ("mesh", "armature"),
        (
            "method",
            "policy",
            "influences",
            "joints",
            "resolution",
            "voxel_size",
            "smooth",
            "hops",
            "sides",
            "rigid_parts",
            "skinning",
            "dq",
            "max_seconds",
            "context",
            "diffusion",
            "visibility",
            "normals",
            "corrective",
        ),
        False,
    ),  # fmt: skip
    "generate_rigid": (
        "generate_rigid",
        ("root", "motion"),
        (
            "name",
            "start",
            "roles",
            "forward",
            "travel",
            "tolerance",
            "bind",
            "amplitude",
            "period",
            "cadence",
            "stride",
            "duty",
            "lift",
            "gravity",
            "height",
            "distance",
            "style",
            "knee",
            "side",
            "waves",
            "rate",
            "yaw",
            "pitch",
            "turn",
            "hold",
            "back",
        ),
        False,
    ),
    "convert_character": (
        "convert_character",
        ("source", "to"),
        ("roles", "forward", "name", "hide_original", "start", "end", "tolerance"),
        False,
    ),
    "rigid_parts": ("rigid_parts", ("root",), ("roles", "forward"), True),
    "edit_weights": ("edit_weights", ("mesh", "operations"), ("dry_run",), False),
    "normalize_weights": ("normalize_weights", ("mesh",), (), False),
    "transfer_weights": (
        "transfer_weights",
        ("mesh", "source"),
        ("max_distance", "influences", "unmatched", "smooth", "skinning"),
        False,
    ),
    "add_control": (
        "add_control",
        ("armature", "name", "kind"),
        ("chain", "joint", "constraint", "pole", "target", "group", "side", "influence", "space"),
        False,
    ),  # fmt: skip
    "ik_fk": ("ik_fk", ("armature", "limb", "mode", "time"), ("blend",), False),
    "pose_character": ("pose_character", ("armature", "commands"), ("time",), False),
    "apply_rig_fix": (
        "apply_rig_fix",
        ("armature", "fixes"),
        ("dry_run", "verify", "poses", "animation"),
        False,
    ),
    "detect_landmarks": (
        "detect_landmarks",
        ("meshes",),
        ("template", "up", "forward", "resolution", "symmetry", "landmarks"),
        True,
    ),
    "inspect_weights": ("inspect_weights", ("mesh",), ("joint", "max_list"), True),
    "diagnose_rig": (
        "diagnose_rig",
        ("armature",),
        ("poses", "time", "animation", "max_findings"),
        True,
    ),  # fmt: skip
    "propose_rig_fix": ("propose_rig_fix", ("armature",), ("poses", "time", "animation"), True),
    "report": ("curve_report", ("path",), ("target", "clip", "samples"), True),
    "trajectory": ("trajectory", ("target",), ("start", "end", "rate", "joint"), True),
    "diagnose": ("diagnose", (), ("overshoot_relative", "include_intended", "max_findings"), True),
    "contacts": (
        "foot_contacts",
        ("armature", "feet"),
        ("start", "end", "rate", "ground", "height", "speed", "min_duration"),
        True,
    ),  # fmt: skip
    "sample": ("sample", ("target", "path", "times"), ("order",), True),
}
# Script argument -> method parameter (where the names differ).
_RENAMED = {
    "edit_keys": {"edit": "op"},
    "generate": {"motion": "kind"},
    "skin": {"mesh": "mesh_object"},
    "unbind": {"mesh": "mesh_object"},
    "morph_targets": {"object": "mesh_object"},
    "attach": {"object": "obj"},
    "detach": {"object": "obj"},
    "follow_path": {"object": "obj"},
    "remove_path": {"path": "ref"},
    "remove_clip": {"clip": "ref"},
    "mirror_clip": {},
    "add_rig": {},
    "generate_weights": {"mesh": "mesh_object"},
    "edit_weights": {"mesh": "mesh_object"},
    "normalize_weights": {"mesh": "mesh_object"},
    "inspect_weights": {"mesh": "mesh_object"},
    "transfer_weights": {"mesh": "target"},
    "generate_rigid": {"motion": "kind", "roles": "parts"},
    "convert_character": {"roles": "parts"},
    "rigid_parts": {"roles": "parts"},
}
# Arguments holding lists of object references.
_REF_LISTS = ("meshes", "context")
# Arguments naming scene objects (or materials), checked to be references.
_REFS = ("target", "object", "armature", "camera", "owner", "mesh", "source", "root")
MAX_SAMPLE_TIMES = 10_000


# --------------------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------------------- #


def _object(value: Any, loc: str) -> dict:
    if not isinstance(value, dict):
        raise AnimationScriptError(f"expected an object, got {type(value).__name__}", loc)
    return value


def _limits(raw: Any) -> dict[str, float]:
    raw = _object(raw, "script.limits")
    unknown = sorted(set(raw) - set(LIMIT_KEYS))
    if unknown:
        raise AnimationScriptError(f"unknown limit(s): {', '.join(unknown)}", "script.limits")
    out = dict(DEFAULT_LIMITS)
    for key, value in raw.items():
        integer = key == "max_keys"
        ok = isinstance(value, int) if integer else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok or not math.isfinite(value) or value <= 0:
            kind = "an integer" if integer else "a number"
            raise AnimationScriptError(f"{key} must be {kind} > 0", f"script.limits.{key}")
        out[key] = value
    return out


def _ref(value: Any, loc: str) -> None:
    if isinstance(value, str) and value.strip():
        return
    if isinstance(value, dict) and len(value) == 1:
        ((key, inner),) = value.items()
        if key in ("object", "material", "id", "name") and isinstance(inner, str) and inner.strip():
            return
    raise AnimationScriptError(
        'expected a name, or {"object": name or id} / {"material": name or id}', loc
    )


def check_script(script: Any) -> dict:
    """Check a script's structure (everything that does not need the scene); returns it."""
    script = _object(script, "script")
    allowed = {"format", "version", "label", "limits", "operations"}
    unknown = sorted(set(script) - allowed)
    if unknown:
        raise AnimationScriptError(f"unknown key(s): {', '.join(unknown)}", "script")
    for key in ("format", "version", "operations"):
        if key not in script:
            raise AnimationScriptError(f"missing key {key!r}", "script")
    if script["format"] != FORMAT_ID:
        raise AnimationScriptError(f"format must be {FORMAT_ID!r}", "script.format")
    if isinstance(script["version"], bool) or script["version"] != VERSION:
        raise AnimationScriptError(
            f"unsupported version {script['version']!r} (supported: {VERSION})", "script.version"
        )
    if "label" in script and (
        not isinstance(script["label"], str)
        or not script["label"].strip()
        or len(script["label"]) > 200
    ):
        raise AnimationScriptError("label must be a non-empty string (<= 200)", "script.label")
    _limits(script.get("limits", {}))
    ops = script["operations"]
    if not isinstance(ops, list) or not ops:
        raise AnimationScriptError("operations must be a non-empty array", "script.operations")
    if len(ops) > MAX_OPERATIONS:
        raise AnimationScriptError(
            f"too many operations ({len(ops)} > {MAX_OPERATIONS})", "script.operations"
        )
    for k, step in enumerate(ops):
        loc = f"operations[{k}]"
        step = _object(step, loc)
        op = step.get("op")
        if not isinstance(op, str) or op not in OPERATIONS:
            raise AnimationScriptError(
                f"unknown op {op!r}", f"{loc}.op", details={"hint": "ops: " + ", ".join(OPERATIONS)}
            )
        _, required, optional, _ = OPERATIONS[op]
        args = {key: v for key, v in step.items() if key != "op"}
        unknown = sorted(set(args) - set(required) - set(optional))
        if unknown:
            raise AnimationScriptError(
                f"{op} takes no {', '.join(unknown)}", loc,
                details={"hint": f"{op} takes: {', '.join((*required, *optional)) or 'nothing'}"},
            )  # fmt: skip
        missing = [key for key in required if key not in args]
        if missing:
            raise AnimationScriptError(f"{op} needs {', '.join(missing)}", loc)
        for key in _REFS:
            if key in args:
                _ref(args[key], f"{loc}.{key}")
        for key in _REF_LISTS:
            if key in args:
                refs = args[key] if isinstance(args[key], list) else [args[key]]
                if not refs or len(refs) > 256:
                    raise AnimationScriptError(f"{key} names 1..256 objects", f"{loc}.{key}")
                for i, r in enumerate(refs):
                    _ref(r, f"{loc}.{key}[{i}]")
        if "roles" in args:
            roles = args["roles"]
            if not isinstance(roles, dict) or not 1 <= len(roles) <= 16:
                raise AnimationScriptError("roles is {role: object}", f"{loc}.roles")
            for role, r in roles.items():
                _ref(r, f"{loc}.roles.{role}")
        if op == "sample":
            times = args["times"]
            if not isinstance(times, list) or not 1 <= len(times) <= MAX_SAMPLE_TIMES:
                raise AnimationScriptError(
                    f"times is a list of 1..{MAX_SAMPLE_TIMES} times", f"{loc}.times"
                )
        if op == "bake" and (not isinstance(args["targets"], list) or not args["targets"]):
            raise AnimationScriptError("targets is a non-empty list", f"{loc}.targets")
    return script


# --------------------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------------------- #


def _fail(exc: Exception, loc: str) -> AnimationScriptError:
    code = getattr(exc, "code", None) or type(exc).__name__
    details: dict[str, Any] = {}
    if getattr(exc, "hint", None):
        details["hint"] = exc.hint
    inner = getattr(exc, "location", None)
    where = (
        f"{loc}.{inner}" if isinstance(inner, str) and inner and not inner.startswith(loc) else loc
    )
    message = getattr(exc, "message", None) or str(exc)
    return AnimationScriptError(message, where, code, details)


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "name") and hasattr(value, "id") and not isinstance(value, (str, int)):
        return {"name": value.name}
    return value


def _count_keys(system: Any) -> int:
    total = 0
    for item in system._clip_names.values():
        for _, ch in item._definition.channels:
            total += len(ch)
    return total


def _call(system: Any, op: str, args: dict[str, Any], loc: str) -> Any:
    method_name, _, _, _ = OPERATIONS[op]
    renamed = _RENAMED.get(op, {})
    kwargs = {}
    for k, v in args.items():
        if k in _REFS and isinstance(v, dict) and set(v) <= {"id", "name"}:
            v = next(iter(v.values()))  # {"id": ..} / {"name": ..}: the string itself
        if k in _REF_LISTS:
            v = [
                next(iter(r.values())) if isinstance(r, dict) else r
                for r in (v if isinstance(v, list) else [v])
            ]
        if k == "roles":
            v = {role: next(iter(r.values())) if isinstance(r, dict) else r
                 for role, r in v.items()}  # fmt: skip
        kwargs[renamed.get(k, k)] = v
    if method_name == "_set_time":
        system.time = kwargs["time"]
        return {"time": system.time}
    if method_name in ("_rename_clip", "_clip_range", "_clip_events"):
        handle = system.clip(kwargs.pop("clip"))
        if method_name == "_rename_clip":
            handle.rename(kwargs["name"])
        elif method_name == "_clip_range":
            handle.set_range(kwargs["range"])
        else:
            handle.set_events(kwargs["events"])
        return {"clip": handle.name}
    if op in (
        "keyframe",
        "set_channel",
        "edit_keys",
        "report",
        "simplify",
        "resample",
        "set_modifiers",
    ):
        if "target" not in kwargs and "clip" not in kwargs:
            raise AnimationScriptError(f"{op} needs a target or a clip", loc)
        if op == "keyframe" and "clip" in kwargs:
            handle = system.clip(kwargs.pop("clip"))
            kwargs.pop("target", None)
            handle.keyframe(kwargs.pop("path"), kwargs.pop("time"), kwargs.pop("value"), **kwargs)
            return {"clip": handle.name}
        kwargs.setdefault("target", None)
    if op == "set_action" and kwargs["clip"] is not None and not isinstance(kwargs["clip"], str):
        raise AnimationScriptError("clip is a clip name (or null)", f"{loc}.clip")
    method = getattr(system, method_name)
    try:
        inspect.signature(method).bind(**kwargs)
    except TypeError as exc:
        raise AnimationScriptError(str(exc), loc, "INVALID_ARGUMENT") from None
    return method(**kwargs)


def run_animation_script(
    scene: Scene,
    script: dict[str, Any],
    *,
    label: str | None = None,
    should_stop: Any = None,
) -> dict[str, Any]:
    """Apply an animation script to ``scene`` (everything or nothing; see the module
    docs). Returns ``{"operations": [...], "animation": summary}``."""
    script = check_script(script)
    limits = _limits(script.get("limits", {}))
    ops = script["operations"]
    label = label or script.get("label") or f"Animation script ({len(ops)} operations)"
    system = scene.animation
    deadline = time.monotonic() + float(limits["max_seconds"])
    before_time = system.time
    results = []
    try:
        with scene.history.transaction(label):
            for k, step in enumerate(ops):
                loc = f"operations[{k}]"
                if should_stop is not None and should_stop():
                    raise AnimationScriptError("the script was cancelled", loc, "CANCELLED")
                if time.monotonic() > deadline:
                    raise AnimationScriptError(
                        f"the script ran longer than max_seconds ({limits['max_seconds']} s)",
                        loc,
                        "BUDGET_EXCEEDED",
                        {"hint": "raise limits.max_seconds or split the script"},
                    )
                op = step["op"]
                args = {key: v for key, v in step.items() if key != "op"}
                try:
                    out = _call(system, op, args, loc)
                except AnimationScriptError:
                    raise
                except (AetherError, ValueError, TypeError, KeyError) as exc:
                    raise _fail(exc, loc) from None
                keys = _count_keys(system)
                if keys > limits["max_keys"]:
                    raise AnimationScriptError(
                        f"the animation holds {keys} keys (max_keys {int(limits['max_keys'])})",
                        loc,
                        "BUDGET_EXCEEDED",
                        {"hint": "simplify curves or raise limits.max_keys"},
                    )
                entry: dict[str, Any] = {"op": op}
                if out is not None:
                    if hasattr(out, "name") and hasattr(out, "paths"):  # a clip handle
                        entry["clip"] = out.name
                    else:
                        entry["result"] = _jsonable(out)
                results.append(entry)
    except BaseException:
        system._time = before_time  # (the playhead is not in the history)
        raise
    return {"operations": results, "animation": summary(scene)}


def summary(scene: Scene) -> dict[str, Any]:
    """A compact account of a scene's animation (for agents)."""
    system = getattr(scene, "_animation", None)
    if system is None:
        return {"clips": [], "bindings": 0}
    objects = scene._objects
    clips = []
    for item in sorted(system._clip_names.values(), key=lambda i: i._serial):
        data = item._definition
        rng = data.frame_range()
        clips.append({"name": item._name, "channels": len(data.channels),
                      "keys": sum(len(ch) for _, ch in data.channels),
                      "range": [rng[0], rng[1]]})  # fmt: skip
    bindings = []
    for (kind, tid), item in system._bindings.items():
        b = item._definition
        name = objects[tid].name if kind == "object" and tid in objects else tid
        action = system._items.get(b.action)._name if b.action in system._items else None
        bindings.append({"target": name, "action": action,
                         "layers": [layer.name for layer in b.layers], "mute": b.mute})  # fmt: skip
    owned: dict[str, int] = {}
    for (kind, _), item in system._owned.items():
        if item._alive:
            owned[kind] = owned.get(kind, 0) + 1
    return {
        "timeline": {
            "fps": system.fps.to_json(),
            "start": system.start,
            "end": system.end,
            "time": system.time,
        },  # fmt: skip
        "clips": clips,
        "bindings": bindings,
        "rigs": owned.get("rig", 0),
        "skins": owned.get("skin", 0),
        "constraints": owned.get("constraints", 0),
        "morphs": owned.get("morph", 0),
        "paths": len(system._path_names),
        "shots": len(system.shots()),
    }
