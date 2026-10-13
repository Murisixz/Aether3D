"""Declarative JSON scene-setup scripts: cameras, lights, appearance, world, render settings.

Scripts are data -- nothing in them is executed as code. Format::

    {
      "format": "aether3d.scene-script",
      "version": 1,
      "operations": [
        {"op": "add_camera", "name": "Main", "position": [8, -12, 6], "look_at": [0, 0, 1]},
        {"op": "add_light", "type": "directional", "direction": [-0.4, 0.6, -1], "intensity": 2},
        {"op": "set_object", "object": "Crate", "color": "#c0392b"},
        {"op": "set_world", "background": "#9cc6e8"},
        {"op": "set_render_settings", "width": 1920, "height": 1080, "samples": 3}
      ]
    }

Operations (``object``/``camera``/``light``/``parent`` are object ids or names;
``look_at`` is a point ``[x, y, z]`` or an object):

* ``add_camera`` -- ``name``, ``projection``, ``fov``, ``fov_axis``, ``ortho_scale``,
  ``near``, ``far``, ``aspect_ratio``, ``parent``, ``position``, ``rotation_euler``,
  ``look_at``, ``up``, ``active``; ``frame: true`` then frames it on ``frame_targets``
  (default: all visible geometry) along ``frame_direction`` with ``frame_margin``.
* ``add_light`` -- ``type`` (``directional``/``point``), ``name``, ``color``, ``intensity``,
  ``range``, ``falloff``, ``parent``, ``position``, ``rotation_euler``, ``direction``,
  ``look_at``, ``up``.
* ``set_camera`` / ``set_light`` -- ``camera``/``light`` plus any of their parameters.
* ``set_object`` -- ``object`` plus ``position``, ``rotation_euler``, ``scale``,
  ``look_at`` or ``look_along`` (with ``up``), ``color``, ``specular``, ``shininess``,
  ``visible``.
* ``frame_camera`` -- ``camera`` (default: the active camera), ``targets``, ``direction``,
  ``up``, ``margin``, ``fit_clipping``.
* ``set_active_camera`` -- ``camera`` (or ``null``).
* ``set_world`` -- ``background``, ``ambient_color``, ``ambient_intensity``.
* ``set_render_settings`` -- ``width``, ``height``, ``samples``, ``exposure``, ``cull``,
  ``default_lights``.
* ``remove`` -- ``object`` (and its children).
* ``info`` -- read-only summary of cameras, lights, world and settings.

:func:`run_scene_script` is atomic: if any operation fails, every change the
script made is undone (created objects are removed, removed objects are put
back, and changed values are restored in place) and a :class:`ScriptError`
names the failing operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..editing.script import ScriptError
from ..errors import AetherError, RenderError

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

FORMAT_ID = "aether3d.scene-script"
VERSION = 1
MAX_OPERATIONS = 10_000

_CAMERA = {"projection", "fov", "fov_axis", "ortho_scale", "near", "far", "aspect_ratio"}
_LIGHT = {"color", "intensity", "range", "falloff"}
_PLACE = {"parent", "position", "rotation_euler", "look_at", "up"}
_SPECS: dict[str, tuple[set[str], set[str]]] = {
    "add_camera": (
        set(),
        {"name", "active", "frame", "frame_targets", "frame_direction", "frame_margin"}
        | _CAMERA
        | _PLACE,
    ),
    "add_light": (set(), {"type", "name", "direction"} | _LIGHT | _PLACE),
    "set_camera": ({"camera"}, set(_CAMERA)),
    "set_light": ({"light"}, {"type"} | _LIGHT),
    "set_object": (
        {"object"},
        {
            "position",
            "rotation_euler",
            "scale",
            "look_at",
            "look_along",
            "up",
            "color",
            "specular",
            "shininess",
            "visible",
        },
    ),
    "frame_camera": (set(), {"camera", "targets", "direction", "up", "margin", "fit_clipping"}),
    "set_active_camera": ({"camera"}, set()),
    "set_world": (set(), {"background", "ambient_color", "ambient_intensity"}),
    "set_render_settings": (
        set(),
        {"width", "height", "samples", "exposure", "cull", "default_lights"},
    ),
    "remove": ({"object"}, set()),
    "info": (set(), set()),
}


def run_scene_script(scene: Scene, script: Any) -> dict[str, Any]:
    """Apply a scene-setup script atomically; returns per-operation results."""
    if not isinstance(script, dict):
        raise ScriptError("the script must be a JSON object", "script")
    unknown = sorted(set(script) - {"format", "version", "operations"})
    if unknown:
        raise ScriptError(f"unknown key(s): {', '.join(unknown)}", "script")
    if script.get("format") != FORMAT_ID:
        raise ScriptError(f"format must be {FORMAT_ID!r}", "script.format")
    if script.get("version") != VERSION:
        raise ScriptError(
            f"unsupported version {script.get('version')!r} (supported: {VERSION})",
            "script.version",
        )
    operations = script.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ScriptError("operations must be a non-empty array", "script.operations")
    if len(operations) > MAX_OPERATIONS:
        raise ScriptError(f"too many operations (limit {MAX_OPERATIONS})", "script.operations")
    journal = _Journal(scene)
    results = []
    try:
        for i, step in enumerate(operations):
            results.append(_run(scene, step, f"operations[{i}]", journal))
    except BaseException:
        journal.rollback()
        raise
    return {"operations": results}


class _Journal:
    """Snapshot of everything a script can change, for exact rollback."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.created: list[SceneObject] = []
        self.removed: list[tuple[SceneObject, SceneObject | None, int]] = []
        self.objects = {
            o.id: (
                o,
                (o.transform.position, o.transform.rotation, o.transform.scale),
                o.appearance.to_dict(),
                o.visible,
                None if o.camera is None else o.camera.to_dict(),
                None if o.light is None else o.light.to_dict(),
            )
            for o in scene
        }
        self.active = scene._active_camera
        self.world = scene.world.to_dict()
        self.settings = scene.render_settings.to_dict()

    def rollback(self) -> None:
        scene = self.scene
        for obj in reversed(self.created):
            if obj.scene is scene:
                scene.remove(obj)
        for obj, parent, index in reversed(self.removed):
            scene._restore_subtree(obj, parent, index)
        for obj, (pos, rot, scale), appearance, visible, camera, light in self.objects.values():
            obj.transform.position, obj.transform.rotation, obj.transform.scale = pos, rot, scale
            obj.appearance.configure(**appearance)
            obj._visible = visible
            if camera is not None:
                obj.camera.configure(**camera)
            if light is not None:
                obj.light.configure(**light)
        scene._active_camera = self.active
        scene.world.configure(**self.world)
        scene.render_settings.configure(**self.settings)


def _run(scene: Scene, step: Any, loc: str, journal: _Journal) -> dict[str, Any]:
    if not isinstance(step, dict) or not isinstance(step.get("op"), str):
        raise ScriptError('each operation must be an object with an "op" string', loc)
    op = step["op"]
    if op not in _SPECS:
        raise ScriptError(
            f"unknown operation {op!r}",
            f"{loc}.op",
            "UNKNOWN_OPERATION",
            {"hint": "one of " + ", ".join(sorted(_SPECS))},
        )
    required, optional = _SPECS[op]
    keys = set(step) - {"op"}
    missing, extra = sorted(required - keys), sorted(keys - required - optional)
    if missing or extra:
        what = f"missing {', '.join(missing)}" if missing else f"unknown key(s) {', '.join(extra)}"
        raise ScriptError(f"{op}: {what}", loc)
    try:
        return {"op": op, **_dispatch(scene, op, step, journal)}
    except ScriptError as exc:
        raise ScriptError(exc.message, f"{loc}: {exc.location}", exc.code, exc.details) from None
    except RenderError as exc:
        subject = next(
            (step[k] for k in ("camera", "light", "object") if isinstance(step.get(k), str)), None
        )
        where = exc.location or ""
        if subject is not None and where.split(".")[0] in ("camera", "light", "appearance"):
            where = f"objects[{subject}].{where}"
        details = {"hint": exc.hint} if exc.hint else {}
        raise ScriptError(
            exc.message, f"{loc}: {where}" if where else loc, exc.code, details
        ) from None
    except AetherError as exc:
        code = getattr(exc, "code", "INVALID_ARGUMENT")
        message = getattr(exc, "message", None) or str(exc)
        location = getattr(exc, "location", None)
        details = {}
        if getattr(exc, "hint", None):
            details["hint"] = exc.hint
        raise ScriptError(
            message, f"{loc}: {location}" if location else loc, code, details
        ) from None
    except (TypeError, ValueError) as exc:
        raise ScriptError(str(exc), loc, "INVALID_ARGUMENT") from None


def _obj(scene: Scene, ref: Any) -> SceneObject:
    return scene._resolve_object(ref)


def _target(scene: Scene, value: Any) -> Any:
    return _obj(scene, value) if isinstance(value, str) else value


def _ref(obj: SceneObject) -> dict[str, str]:
    return {"id": obj.id, "name": obj.name}


def _dispatch(scene: Scene, op: str, s: dict[str, Any], journal: _Journal) -> dict[str, Any]:
    if op == "add_camera":
        params = {k: s[k] for k in _CAMERA if k in s}
        obj = scene.add_camera(
            s.get("name"),
            parent=_obj(scene, s["parent"]) if "parent" in s else None,
            position=s.get("position"),
            look_at=_target(scene, s["look_at"]) if "look_at" in s else None,
            up=s.get("up", (0.0, 0.0, 1.0)),
            active=s.get("active"),
            **params,
        )
        journal.created.append(obj)
        if "rotation_euler" in s:
            obj.rotation_euler = s["rotation_euler"]
        result: dict[str, Any] = {"object": _ref(obj), "active": scene.active_camera is obj}
        if s.get("frame", False):
            framing = obj.frame(
                s.get("frame_targets"),
                direction=s.get("frame_direction"),
                up=s.get("up", (0.0, 0.0, 1.0)),
                margin=s.get("frame_margin", 0.05),
            )
            result["framing"] = framing.to_dict()
        return result
    if op == "add_light":
        params = {k: s[k] for k in _LIGHT if k in s}
        obj = scene.add_light(
            s.get("type", "directional"),
            s.get("name"),
            parent=_obj(scene, s["parent"]) if "parent" in s else None,
            position=s.get("position"),
            direction=s.get("direction"),
            look_at=_target(scene, s["look_at"]) if "look_at" in s else None,
            up=s.get("up", (0.0, 0.0, 1.0)),
            **params,
        )
        journal.created.append(obj)
        if "rotation_euler" in s:
            obj.rotation_euler = s["rotation_euler"]
        return {"object": _ref(obj)}
    if op == "set_camera":
        obj = _obj(scene, s["camera"])
        if obj.camera is None:
            raise ScriptError(f"object {obj.name!r} is not a camera", "camera", "NOT_A_CAMERA")
        obj.camera.configure(**{k: s[k] for k in _CAMERA if k in s})
        return {"object": _ref(obj), "camera": obj.camera.to_dict()}
    if op == "set_light":
        obj = _obj(scene, s["light"])
        if obj.light is None:
            raise ScriptError(f"object {obj.name!r} is not a light", "light", "NOT_A_LIGHT")
        obj.light.configure(**{k: s[k] for k in {"type"} | _LIGHT if k in s})
        return {"object": _ref(obj), "light": obj.light.to_dict()}
    if op == "set_object":
        obj = _obj(scene, s["object"])
        if "look_at" in s and "look_along" in s:
            raise ScriptError("give look_at or look_along, not both", "object")
        appearance = {k: s[k] for k in ("color", "specular", "shininess") if k in s}
        if appearance:
            obj.appearance.configure(**appearance)
        if "visible" in s:
            obj.visible = s["visible"]
        if "scale" in s:
            obj.scale = s["scale"]
        if "rotation_euler" in s:
            obj.rotation_euler = s["rotation_euler"]
        if "position" in s:
            obj.position = s["position"]
        up = s.get("up", (0.0, 0.0, 1.0))
        if "look_at" in s:
            obj.look_at(_target(scene, s["look_at"]), up=up)
        if "look_along" in s:
            obj.look_along(s["look_along"], up=up)
        return {"object": _ref(obj), **_object_summary(obj)}
    if op == "frame_camera":
        if "camera" in s:
            obj = _obj(scene, s["camera"])
        else:
            obj = scene.active_camera
            if obj is None:
                raise ScriptError("there is no active camera to frame", "camera", "NO_CAMERA")
        targets = s.get("targets")
        framing = obj.frame(
            targets,
            direction=s.get("direction"),
            up=s.get("up", (0.0, 0.0, 1.0)),
            margin=s.get("margin", 0.05),
            fit_clipping=s.get("fit_clipping", True),
        )
        return {"object": _ref(obj), "framing": framing.to_dict()}
    if op == "set_active_camera":
        scene.active_camera = s["camera"]
        active = scene.active_camera
        return {"active_camera": None if active is None else _ref(active)}
    if op == "set_world":
        scene.world.configure(
            **{k: s[k] for k in ("background", "ambient_color", "ambient_intensity") if k in s}
        )
        return {"world": scene.world.to_dict()}
    if op == "set_render_settings":
        scene.render_settings.configure(**{k: s[k] for k in _SPECS[op][1] if k in s})
        return {"render_settings": scene.render_settings.to_dict()}
    if op == "remove":
        obj = _obj(scene, s["object"])
        parent = obj.parent
        siblings = scene.roots if parent is None else parent.children
        journal.removed.append((obj, parent, siblings.index(obj)))
        removed = [obj.id, *(d.id for d in obj.iter_descendants())]
        scene.remove(obj)
        return {"removed": removed}
    if op == "info":
        return rendering_summary(scene)
    raise AssertionError(op)  # pragma: no cover


def _object_summary(obj: SceneObject) -> dict[str, Any]:
    t = obj.transform
    return {
        "kind": obj.kind,
        "position": list(t.position),
        "rotation_euler": list(t.rotation_euler),
        "scale": list(t.scale),
        "appearance": obj.appearance.to_dict(),
        "visible": obj.visible,
    }


def rendering_summary(scene: Scene) -> dict[str, Any]:
    """Cameras (with world placement), lights, world and render settings, JSON-compatible."""
    import numpy as np

    from .camera import frame_from_matrix
    from .scene_data import world_matrices

    worlds = world_matrices(scene)
    cameras = []
    for obj in scene.cameras:
        entry: dict[str, Any] = {
            **_ref(obj),
            "active": scene.active_camera is obj,
            **obj.camera.to_dict(),
        }
        try:
            frame, _ = frame_from_matrix(worlds[obj.id], obj.name)
            entry.update(frame.to_dict())
        except AetherError as exc:
            entry["error"] = str(exc)
        cameras.append(entry)
    lights = []
    for obj in scene.lights:
        world = worlds[obj.id]
        entry = {
            **_ref(obj),
            "visible": obj.visible,
            **obj.light.to_dict(),
            "position": world[:3, 3].tolist(),
        }
        direction = -world[:3, 1]
        length = float(np.linalg.norm(direction))
        entry["direction"] = (direction / length).tolist() if length > 1e-12 else None
        lights.append(entry)
    active = scene.active_camera
    return {
        "active_camera": None if active is None else _ref(active),
        "cameras": cameras,
        "lights": lights,
        "world": scene.world.to_dict(),
        "render_settings": scene.render_settings.to_dict(),
    }
