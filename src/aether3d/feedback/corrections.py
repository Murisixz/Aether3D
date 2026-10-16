"""Corrections: typed, validated, explainable changes proposed for findings.

**Actions** are the only changes a correction can make (:data:`ACTIONS`, an allowlist):
moving, turning or scaling an object, showing it, assigning a material or color, framing,
aiming or moving a camera, its lens, a light, the exposure, the ambient light, a
procedural parameter. Nothing else -- no deletion, no topology edit, no file, no network,
no code. An action names its target by id or name and states its coordinate frame
(``"world"`` by default: offsets are converted exactly through the parent's world matrix).

Actions **compile** to the engine's own operations: a scene plan (``update``,
``frame_camera``, ``set_parameters``, ``world``, ``render_settings``) and, for camera
lenses and lights, a scene script (``set_camera``, ``set_light``, ``add_light``). The
compiled documents are part of every proposal, so an agent can read exactly what will
run, and run it itself. Applying runs them in **one history transaction**: one undo step,
all or nothing, refused when the scene's revision is not the proposal's.

**Proposals** are made from findings by planners, one per finding code. Each carries its
target, the findings it addresses, the evidence, the expected effect (the finding's
magnitude after), what verification will check, alternatives, and a **safety** class:

* ``"safe"`` -- a deterministic, exact repair of an explicitly stated expectation
  (lower a floating object by its measured gap, assign the stated material, unhide an
  object that must be visible, move a clipping plane, stand an object upright);
* ``"verify"`` -- deterministic but approximate, or with side effects to check (separate
  intersecting objects, re-frame or aim a camera, change the exposure): applied
  automatically only when the policy allows, and kept only if verification shows the
  findings improved and nothing regressed;
* ``"approval"`` -- subjective or ambiguous (which of two objects to move, hiding an
  occluder, adding a light, anything a vision model or an agent suggested): never applied
  without explicit approval.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, FeedbackError
from ._doc import BAD, Doc, join

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject
    from .checks import Diagnosis, Finding

FORMAT = "aether3d.corrections"
VERSION = 1
SAFETY = ("safe", "verify", "approval")
MAX_COORDINATE = 1e7

ACTIONS: dict[str, dict[str, Any]] = {
    "translate": {
        "fields": ("object", "offset", "frame"),
        "required": ("object", "offset"),
        "category": "placement",
        "summary": "move an object by an offset (frame: world, default, or local: its parent's)",
    },
    "place": {
        "fields": ("object", "position", "frame"),
        "required": ("object", "position"),
        "category": "placement",
        "summary": "put an object's origin at a position (frame: world or local)",
    },
    "rotate": {
        "fields": ("object", "rotation_euler", "rotation"),
        "required": ("object",),
        "category": "placement",
        "summary": "set an object's local rotation (Euler degrees XYZ, or a quaternion)",
    },
    "scale": {
        "fields": ("object", "factor"),
        "required": ("object", "factor"),
        "category": "placement",
        "summary": "multiply an object's scale by a factor (a number or [x, y, z])",
    },
    "set_visibility": {
        "fields": ("object", "visible"),
        "required": ("object", "visible"),
        "category": "visibility",
        "summary": "show or hide an object",
    },
    "assign_material": {
        "fields": ("object", "material"),
        "required": ("object", "material"),
        "category": "materials",
        "summary": "assign an existing material (by name or id) to an object",
    },
    "set_color": {
        "fields": ("object", "color"),
        "required": ("object", "color"),
        "category": "materials",
        "summary": "set the color of an object without a material",
    },
    "frame_camera": {
        "fields": ("camera", "targets", "direction", "margin"),
        "required": (),
        "category": "camera",
        "summary": "move a camera (default: the active one) so the targets fill its frame",
    },
    "aim_camera": {
        "fields": ("camera", "look_at"),
        "required": ("look_at",),
        "category": "camera",
        "summary": "turn a camera toward a point or an object",
    },
    "dolly_camera": {
        "fields": ("camera", "distance"),
        "required": ("distance",),
        "category": "camera",
        "summary": "move a camera along its view direction (positive: forward)",
    },
    "set_camera": {
        "fields": ("camera", "fov", "near", "far"),
        "required": (),
        "category": "camera",
        "summary": "change a camera's field of view or clipping planes",
    },
    "set_light": {
        "fields": ("light", "intensity", "color"),
        "required": ("light",),
        "category": "lighting",
        "summary": "change a light's intensity or color",
    },
    "add_light": {
        "fields": ("type", "name", "direction", "position", "intensity", "color"),
        "required": ("type",),
        "category": "lighting",
        "summary": "add a directional or point light",
    },
    "set_exposure": {
        "fields": ("exposure",),
        "required": ("exposure",),
        "category": "lighting",
        "summary": "set the render settings' exposure (stops)",
    },
    "set_ambient": {
        "fields": ("intensity", "color"),
        "required": (),
        "category": "lighting",
        "summary": "set the world's ambient light",
    },
    "set_parameter": {
        "fields": ("name", "value"),
        "required": ("name", "value"),
        "category": "procedural",
        "summary": "set a procedural scene parameter",
    },
}


def check_action(doc: Doc, value: Any, ptr: str) -> dict[str, Any] | None:
    """Check one action's shape (references are resolved later, against the scene)."""
    obj = doc.object(value, ptr, 'an action {"action": name, ...}')
    if obj is None:
        return None
    name = doc.enum(obj.get("action"), join(ptr, "action"), tuple(ACTIONS))
    if name is BAD:
        return None
    spec = ACTIONS[name]
    before = doc.issues.total
    doc.keys(obj, ptr, ("action", *spec["fields"]), spec["required"])
    out: dict[str, Any] = {"action": name}
    for key in spec["fields"]:
        if key not in obj:
            continue
        p = join(ptr, key)
        v = obj[key]
        if key in ("object", "camera", "light", "material", "name"):
            r = doc.string(v, p)
        elif key in ("offset", "position", "direction", "rotation_euler"):
            r = doc.vector(v, p, nonzero=key == "direction")
            if r is not BAD and max(abs(c) for c in r) > MAX_COORDINATE:
                doc.issues.add("OUT_OF_RANGE", f"values beyond {MAX_COORDINATE:g}", p)
                r = BAD
            r = BAD if r is BAD else list(r)
        elif key == "rotation":
            r = doc.vector(v, p, 4)
            if r is not BAD and abs(math.hypot(*r) - 1.0) > 1e-6:
                doc.issues.add("INVALID_VALUE", "a rotation quaternion has length 1", p)
                r = BAD
            r = BAD if r is BAD else list(r)
        elif key == "frame":
            r = doc.enum(v, p, ("world", "local"))
        elif key == "factor":
            if isinstance(v, list):
                r = doc.vector(v, p)
                if r is not BAD and min(r) <= 0.0:
                    doc.issues.add("OUT_OF_RANGE", "scale factors are positive", p)
                    r = BAD
                r = BAD if r is BAD else list(r)
            else:
                r = doc.number(v, p, lo=0.0, lo_open=True, hi=1000.0)
        elif key == "visible":
            r = doc.boolean(v, p)
        elif key == "color":
            r = doc.color(v, p)
        elif key == "targets":
            r = doc.selector(v, p)
        elif key == "margin":
            r = doc.number(v, p, lo=0.0, hi=0.9)
        elif key == "look_at":
            if isinstance(v, str):
                r = doc.string(v, p)
            else:
                r = doc.vector(v, p)
                r = BAD if r is BAD else list(r)
        elif key == "distance":
            r = doc.number(v, p, lo=-MAX_COORDINATE, hi=MAX_COORDINATE)
        elif key == "fov":
            r = doc.number(v, p, lo=0.01, hi=179.0)
        elif key in ("near", "far"):
            r = doc.number(v, p, lo=0.0, lo_open=True, hi=MAX_COORDINATE)
        elif key == "intensity":
            r = doc.number(v, p, lo=0.0, hi=1e6)
        elif key == "type":
            r = doc.enum(v, p, ("directional", "point"))
        elif key == "exposure":
            r = doc.number(v, p, lo=-20.0, hi=20.0)
        elif key == "value":
            if isinstance(v, (bool, int, float)):
                r = v if isinstance(v, bool) else doc.number(v, p)
            else:
                r = doc.string(v, p)
        else:  # pragma: no cover - every field is handled above
            r = BAD
        if r is not BAD:
            out[key] = r
    if name == "rotate" and ("rotation" in obj) == ("rotation_euler" in obj):
        doc.issues.add("INVALID_VALUE", "give rotation_euler or rotation (one of them)", ptr)
    if name == "set_camera" and not ({"fov", "near", "far"} & set(obj)):
        doc.issues.add("MISSING_KEY", "give fov, near or far", ptr)
    if name == "set_light" and not ({"intensity", "color"} & set(obj)):
        doc.issues.add("MISSING_KEY", "give intensity or color", ptr)
    if name == "set_ambient" and not ({"intensity", "color"} & set(obj)):
        doc.issues.add("MISSING_KEY", "give intensity or color", ptr)
    if name == "add_light" and ("direction" in obj) == ("position" in obj):
        doc.issues.add(
            "INVALID_VALUE", "a light needs direction (directional) or position (point)", ptr
        )
    if doc.issues.total > before:
        return None
    return out


def check_actions(actions: Any, ptr: str = "/actions") -> list[dict[str, Any]]:
    """A list of actions, checked as a whole (``FeedbackError`` lists every problem)."""
    doc = Doc()
    items = doc.array(actions, ptr, min_items=1, max_items=100)
    out = []
    for k, item in enumerate(items or []):
        a = check_action(doc, item, join(ptr, k))
        if a is not None:
            out.append(a)
    doc.issues.raise_if_any("invalid actions")
    return out


# --------------------------------------------------------------------------- #
# Binding: actions against the scene
# --------------------------------------------------------------------------- #


@dataclass
class Bound:
    """An action resolved against the scene: its target, the change, its compilation."""

    action: dict[str, Any]
    category: str
    target: Any
    changes: list[dict[str, Any]]
    plan_op: dict[str, Any] | None = None
    plan_section: tuple[str, dict[str, Any]] | None = None
    script_op: dict[str, Any] | None = None
    magnitude: dict[str, float] = field(default_factory=dict)


def _object(scene: Scene, ref: Any, what: str, ptr: str) -> SceneObject:
    obj = scene._objects.get(ref) if isinstance(ref, str) else None
    if obj is None and isinstance(ref, str):
        obj = scene._by_name.get(ref)
    if obj is None:
        raise FeedbackError("OBJECT_NOT_FOUND", f"no {what} {ref!r} in the scene", ptr)
    return obj


def _vec(v: Any) -> list[float]:
    return [float(c) for c in v]


def _parent_world(obj: SceneObject) -> np.ndarray | None:
    from ..rendering.scene_data import world_matrices

    if obj.parent is None:
        return None
    return world_matrices(obj._require_scene())[obj.parent.id]


def world_to_local_offset(obj: SceneObject, offset: Any, ptr: str = "") -> list[float]:
    """The change of ``obj.position`` that moves it by ``offset`` in world space (exact
    float64 solve through the parent's world matrix)."""
    v = np.asarray(offset, dtype=np.float64)
    parent = _parent_world(obj)
    if parent is None:
        return _vec(v)
    try:
        local = np.linalg.solve(parent[:3, :3], v)
    except np.linalg.LinAlgError:
        raise FeedbackError(
            "SINGULAR_TRANSFORM",
            f"the parent of {obj.name!r} has a zero world scale: world offsets cannot be converted",
            ptr,
        ) from None
    if not np.all(np.isfinite(local)):
        raise FeedbackError("SINGULAR_TRANSFORM", "the offset is not finite in local space", ptr)
    return _vec(local)


def world_to_local_point(obj: SceneObject, point: Any, ptr: str = "") -> list[float]:
    p = np.append(np.asarray(point, dtype=np.float64), 1.0)
    parent = _parent_world(obj)
    if parent is None:
        return _vec(p[:3])
    try:
        local = np.linalg.solve(parent, p)
    except np.linalg.LinAlgError:
        raise FeedbackError(
            "SINGULAR_TRANSFORM", f"the parent of {obj.name!r} has a zero world scale", ptr
        ) from None
    return _vec(local[:3])


def bind(scene: Scene, action: dict[str, Any], ptr: str = "") -> Bound:
    """Resolve and compile one checked action (``FeedbackError`` when it cannot apply)."""
    name = action["action"]
    category = ACTIONS[name]["category"]
    sel = None
    if "object" in action:
        target = _object(scene, action["object"], "object", f"{ptr}/object")
        sel = {"ids": [target.id]}
    if name == "translate":
        frame = action.get("frame", "world")
        world_offset = action["offset"]
        if frame == "world":
            local = world_to_local_offset(target, world_offset, ptr)
        else:
            local = list(world_offset)
            parent = _parent_world(target)
            world_offset = _vec(local if parent is None else parent[:3, :3] @ np.asarray(local))
        before = _vec(target.world_position)
        after = _vec(np.asarray(before) + np.asarray(world_offset))
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "world_position",
                    "before": before,
                    "after": after,
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"translate": local}},
            magnitude={"translation": float(np.linalg.norm(world_offset))},
        )
    if name == "place":
        frame = action.get("frame", "world")
        before = _vec(target.world_position)
        if frame == "world":
            local = world_to_local_point(target, action["position"], ptr)
            after = list(action["position"])
        else:
            local = list(action["position"])
            parent = _parent_world(target)
            after = local if parent is None else _vec((parent @ np.append(local, 1.0))[:3])
        moved = float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "world_position",
                    "before": before,
                    "after": after,
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"position": local}},
            magnitude={"translation": moved},
        )
    if name == "rotate":
        from ..math3d import quat_from_euler

        before = list(target.rotation)
        if "rotation" in action:
            q = tuple(action["rotation"])
            change = {"rotation": list(q)}
        else:
            q = quat_from_euler(action["rotation_euler"])
            change = {"rotation_euler": list(action["rotation_euler"])}
        angle = _quat_angle(before, q)
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "rotation",
                    "before": before,
                    "after": list(q),
                }
            ],
            plan_op={"op": "update", "select": sel, "set": change},
            magnitude={"rotation": angle},
        )
    if name == "scale":
        f = action["factor"]
        factor = [f, f, f] if not isinstance(f, list) else f
        before = list(target.scale)
        after = [a * b for a, b in zip(before, factor, strict=True)]
        worst = max(max(x, 1.0 / x) for x in factor)
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "scale",
                    "before": before,
                    "after": after,
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"scale_by": factor}},
            magnitude={"scale": worst},
        )
    if name == "set_visibility":
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "visible",
                    "before": target.visible,
                    "after": action["visible"],
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"visible": action["visible"]}},
        )
    if name == "assign_material":
        library = scene.materials
        material = next(
            (m for m in library.materials if action["material"] in (m.id, m.name)), None
        )
        if material is None:
            raise FeedbackError(
                "MATERIAL_NOT_FOUND", f"no material {action['material']!r}", f"{ptr}/material"
            )
        current = target.material
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "material",
                    "before": None if current is None else current.name,
                    "after": material.name,
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"material": {"id": material.id}}},
        )
    if name == "set_color":
        if target.material is not None:
            raise FeedbackError(
                "MATERIAL_ASSIGNED",
                f"{target.name!r} has a material: its color comes from it",
                f"{ptr}/color",
                hint="change the material (or assign another one)",
            )
        return Bound(
            action,
            category,
            target,
            [
                {
                    "object": target.name,
                    "id": target.id,
                    "field": "color",
                    "before": list(target.color),
                    "after": action["color"],
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"color": action["color"]}},
        )
    if name in ("frame_camera", "aim_camera", "dolly_camera", "set_camera"):
        return _bind_camera(scene, action, category, ptr)
    if name in ("set_light", "add_light"):
        return _bind_light(scene, action, category, ptr)
    if name == "set_exposure":
        before = scene.render_settings.exposure
        return Bound(
            action,
            category,
            None,
            [
                {
                    "object": None,
                    "field": "render_settings.exposure",
                    "before": before,
                    "after": action["exposure"],
                }
            ],
            plan_section=("render_settings", {"exposure": action["exposure"]}),
            magnitude={"exposure": abs(action["exposure"] - before)},
        )
    if name == "set_ambient":
        section: dict[str, Any] = {}
        changes = []
        if "intensity" in action:
            section["ambient_intensity"] = action["intensity"]
            changes.append(
                {
                    "object": None,
                    "field": "world.ambient_intensity",
                    "before": scene.world.ambient_intensity,
                    "after": action["intensity"],
                }
            )
        if "color" in action:
            section["ambient_color"] = action["color"]
            changes.append(
                {
                    "object": None,
                    "field": "world.ambient_color",
                    "before": list(scene.world.ambient_color),
                    "after": action["color"],
                }
            )
        return Bound(action, category, None, changes, plan_section=("world", section))
    if name == "set_parameter":
        procedural = scene._procedural
        if procedural is None or action["name"] not in procedural.parameters:
            raise FeedbackError(
                "PARAMETER_NOT_FOUND", f"no procedural parameter {action['name']!r}", ptr
            )
        before = procedural.parameters[action["name"]].value
        return Bound(
            action,
            category,
            None,
            [
                {
                    "object": None,
                    "field": f"procedural.{action['name']}",
                    "before": _plain(before),
                    "after": action["value"],
                }
            ],
            plan_op={"op": "set_parameters", "values": {action["name"]: action["value"]}},
        )
    raise FeedbackError(
        "UNSUPPORTED_ACTION", f"unsupported action {name!r}", ptr
    )  # pragma: no cover


def _plain(value: Any) -> Any:
    from ..procedural.params import Expr

    if isinstance(value, Expr):
        return {"expr": value.source}
    if isinstance(value, tuple):
        return list(value)
    return value


def _quat_angle(a: Any, b: Any) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _camera_object(scene: Scene, ref: Any, ptr: str) -> SceneObject:
    if ref is None:
        camera = scene.active_camera
        if camera is None:
            raise FeedbackError("NO_CAMERA", "the scene has no active camera", ptr)
        return camera
    camera = _object(scene, ref, "camera", ptr)
    if camera.camera is None:
        raise FeedbackError("NOT_A_CAMERA", f"{camera.name!r} is not a camera", ptr)
    return camera


def _bind_camera(scene: Scene, action: dict[str, Any], category: str, ptr: str) -> Bound:
    from ..rendering.camera import frame_from_matrix
    from ..rendering.scene_data import world_matrices

    camera = _camera_object(scene, action.get("camera"), f"{ptr}/camera")
    name = action["action"]
    sel = {"ids": [camera.id]}
    before = _vec(camera.world_position)
    if name == "frame_camera":
        op: dict[str, Any] = {"op": "frame_camera", "camera": {"id": camera.id}}
        if "targets" in action:
            op["targets"] = action["targets"]
        if "direction" in action:
            op["direction"] = action["direction"]
        if "margin" in action:
            op["margin"] = action["margin"]
        return Bound(
            action,
            category,
            camera,
            [
                {
                    "object": camera.name,
                    "id": camera.id,
                    "field": "camera placement",
                    "before": before,
                    "after": "framed on the targets",
                }
            ],
            plan_op=op,
        )
    if name == "aim_camera":
        look = action["look_at"]
        if isinstance(look, str):
            point = _object(scene, look, "object", f"{ptr}/look_at")
            value: Any = {"id": point.id}
            where = _vec(point.world_position)
        else:
            value, where = list(look), list(look)
        frame, _ = frame_from_matrix(world_matrices(scene)[camera.id])
        new_dir = np.asarray(where) - frame.position
        norm = float(np.linalg.norm(new_dir))
        if norm < 1e-12:
            raise FeedbackError("INVALID_VALUE", "the camera would look at its own position", ptr)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(frame.forward @ new_dir) / norm))))
        return Bound(
            action,
            category,
            camera,
            [
                {
                    "object": camera.name,
                    "id": camera.id,
                    "field": "view direction",
                    "before": _vec(frame.forward),
                    "after": _vec(new_dir / norm),
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"look_at": value}},
            magnitude={"rotation": angle},
        )
    if name == "dolly_camera":
        frame, _ = frame_from_matrix(world_matrices(scene)[camera.id])
        offset = frame.forward * float(action["distance"])
        local = world_to_local_offset(camera, offset, ptr)
        return Bound(
            action,
            category,
            camera,
            [
                {
                    "object": camera.name,
                    "id": camera.id,
                    "field": "world_position",
                    "before": before,
                    "after": _vec(np.asarray(before) + offset),
                }
            ],
            plan_op={"op": "update", "select": sel, "set": {"translate": local}},
            magnitude={"translation": abs(float(action["distance"]))},
        )
    # set_camera: the lens, through a scene script
    changes = []
    op = {"op": "set_camera", "camera": camera.id}
    for key in ("fov", "near", "far"):
        if key in action:
            op[key] = action[key]
            changes.append(
                {
                    "object": camera.name,
                    "id": camera.id,
                    "field": f"camera.{key}",
                    "before": getattr(camera.camera, key),
                    "after": action[key],
                }
            )
    trial = camera.camera.copy()
    try:
        trial.configure(**{k: op[k] for k in ("fov", "near", "far") if k in op})
    except AetherError as exc:
        raise FeedbackError(
            "INVALID_VALUE", getattr(exc, "message", str(exc)), f"{ptr}/camera"
        ) from None
    return Bound(action, category, camera, changes, script_op=op)


def _bind_light(scene: Scene, action: dict[str, Any], category: str, ptr: str) -> Bound:
    if action["action"] == "add_light":
        op = {"op": "add_light", "type": action["type"]}
        for key in ("name", "direction", "position", "intensity", "color"):
            if key in action:
                op[key] = action[key]
        if action.get("name") and action["name"] in scene._by_name:
            raise FeedbackError("NAME_CONFLICT", f"{action['name']!r} exists", f"{ptr}/name")
        return Bound(
            action,
            category,
            None,
            [
                {
                    "object": action.get("name", "new light"),
                    "field": "light",
                    "before": None,
                    "after": {k: v for k, v in op.items() if k != "op"},
                }
            ],
            script_op=op,
        )
    light = _object(scene, action["light"], "light", f"{ptr}/light")
    if light.light is None:
        raise FeedbackError("NOT_A_LIGHT", f"{light.name!r} is not a light", f"{ptr}/light")
    op = {"op": "set_light", "light": light.id}
    changes = []
    magnitude = {}
    for key in ("intensity", "color"):
        if key in action:
            op[key] = action[key]
            before = getattr(light.light, key)
            changes.append(
                {
                    "object": light.name,
                    "id": light.id,
                    "field": f"light.{key}",
                    "before": list(before) if isinstance(before, tuple) else before,
                    "after": action[key],
                }
            )
    if "intensity" in action:
        old = max(light.light.intensity, 1e-12)
        new = max(action["intensity"], 1e-12)
        magnitude["light_factor"] = max(new / old, old / new)
    return Bound(action, category, light, changes, script_op=op, magnitude=magnitude)


# --------------------------------------------------------------------------- #
# Compiling a set of bound actions
# --------------------------------------------------------------------------- #


def compile_actions(bound: list[Bound], label: str, revision: int) -> dict[str, Any]:
    """The plan and scene script that perform ``bound`` (in this order), as documents."""
    operations = []
    sections: dict[str, dict[str, Any]] = {}
    script_ops = []
    for b in bound:
        if b.plan_op is not None:
            operations.append(b.plan_op)
        if b.plan_section is not None:
            name, values = b.plan_section
            sections.setdefault(name, {}).update(values)
        if b.script_op is not None:
            script_ops.append(b.script_op)
    plan = None
    if operations or sections:
        plan = {
            "format": "aether3d.scene-plan",
            "version": 1,
            "label": label,
            "expected_revision": revision,
        }
        if operations:
            plan["operations"] = operations
        plan.update(sections)
    script = None
    if script_ops:
        script = {"format": "aether3d.scene-script", "version": 1, "operations": script_ops}
    return {"plan": plan, "scene_script": script}


def run_compiled(scene: Scene, compiled: dict[str, Any], label: str, base_dir: Any = None):
    """Apply compiled corrections as ONE history step (all or nothing); returns its entry."""
    from ..plans import execute_plan
    from ..rendering.script import run_scene_script

    with scene.transaction(label) as tx:
        if compiled.get("plan") is not None:
            execute_plan(scene, compiled["plan"], base_dir=base_dir)
        if compiled.get("scene_script") is not None:
            run_scene_script(scene, compiled["scene_script"], base_dir=base_dir)
    return tx.entry


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

CATEGORY_SWITCH = {
    "camera": "allow_camera",
    "lighting": "allow_lighting",
    "materials": "allow_materials",
    "visibility": "allow_visibility",
    "procedural": "allow_procedural",
    "placement": "allow_placement",
}


@dataclass(frozen=True)
class CorrectionPolicy:
    """What corrections may do, and which are applied without asking.

    * ``auto_apply`` -- safety classes applied automatically (default ``("safe",)``;
      ``"approval"`` can never be automatic).
    * ``allow_*`` -- categories of actions allowed at all.
    * Limits per application: ``max_actions``, ``max_objects`` (objects changed),
      ``max_translation`` (meters; default a quarter of the scene's world-box diagonal,
      at least 1 m), ``max_rotation`` (degrees), ``max_scale`` (factor, both ways),
      ``max_exposure_change`` (stops), ``max_light_factor``.
    * ``verify`` -- re-observe and re-check after applying; ``rollback_on_regression``
      undoes the step when a check that passed fails afterwards;
      ``rollback_if_ineffective`` undoes it when the findings it addressed did not improve.
    * ``min_vision_confidence`` -- vision findings below it make no proposals.
    """

    auto_apply: tuple[str, ...] = ("safe",)
    allow_placement: bool = True
    allow_camera: bool = True
    allow_lighting: bool = True
    allow_materials: bool = True
    allow_visibility: bool = True
    allow_procedural: bool = False
    max_actions: int = 20
    max_objects: int = 10
    max_translation: float | None = None
    max_rotation: float = 90.0
    max_scale: float = 4.0
    max_exposure_change: float = 4.0
    max_light_factor: float = 8.0
    verify: bool = True
    rollback_on_regression: bool = True
    rollback_if_ineffective: bool = True
    min_vision_confidence: float = 0.6

    FIELDS = (
        "auto_apply",
        "allow_placement",
        "allow_camera",
        "allow_lighting",
        "allow_materials",
        "allow_visibility",
        "allow_procedural",
        "max_actions",
        "max_objects",
        "max_translation",
        "max_rotation",
        "max_scale",
        "max_exposure_change",
        "max_light_factor",
        "verify",
        "rollback_on_regression",
        "rollback_if_ineffective",
        "min_vision_confidence",
    )

    @classmethod
    def coerce(cls, value: Any, ptr: str = "/policy") -> CorrectionPolicy:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        doc = Doc()
        obj = doc.object(value, ptr, "a correction policy")
        out: dict[str, Any] = {}
        if obj is not None:
            doc.keys(obj, ptr, cls.FIELDS)
            for key, v in obj.items():
                p = join(ptr, key)
                if key == "auto_apply":
                    items = doc.array(v, p, max_items=3)
                    chosen = []
                    for i, item in enumerate(items or []):
                        s = doc.enum(item, join(p, i), ("safe", "verify"))
                        if s is not BAD:
                            chosen.append(s)
                    out[key] = tuple(chosen)
                elif key.startswith("allow_") or key in (
                    "verify",
                    "rollback_on_regression",
                    "rollback_if_ineffective",
                ):
                    r = doc.boolean(v, p)
                    if r is not BAD:
                        out[key] = r
                elif key in ("max_actions", "max_objects"):
                    r = doc.integer(v, p, lo=1, hi=1000)
                    if r is not BAD:
                        out[key] = r
                elif key == "max_translation":
                    r = None if v is None else doc.number(v, p, lo=0.0)
                    if r is not BAD:
                        out[key] = r
                elif key == "max_rotation":
                    r = doc.number(v, p, lo=0.0, hi=180.0)
                    if r is not BAD:
                        out[key] = r
                elif key in ("max_scale", "max_light_factor"):
                    r = doc.number(v, p, lo=1.0, hi=1000.0)
                    if r is not BAD:
                        out[key] = r
                elif key == "max_exposure_change":
                    r = doc.number(v, p, lo=0.0, hi=20.0)
                    if r is not BAD:
                        out[key] = r
                elif key == "min_vision_confidence":
                    r = doc.number(v, p, lo=0.0, hi=1.0)
                    if r is not BAD:
                        out[key] = r
        doc.issues.raise_if_any("invalid policy")
        return cls(**out)

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.FIELDS}
        out["auto_apply"] = list(self.auto_apply)
        return out

    def translation_limit(self, scene: Scene) -> float:
        if self.max_translation is not None:
            return self.max_translation
        try:
            b = scene.bounds()
        except AetherError:
            b = None
        if b is None:
            return 1.0
        size = float(np.linalg.norm(np.asarray(b.aabb.max) - np.asarray(b.aabb.min)))
        return max(1.0, 0.25 * size)

    def violations(self, scene: Scene, bound: list[Bound]) -> list[str]:
        """Why this set of actions is not allowed (empty: allowed)."""
        out = []
        if len(bound) > self.max_actions:
            out.append(f"{len(bound)} actions exceed max_actions {self.max_actions}")
        objects = {b.target.id for b in bound if b.target is not None}
        if len(objects) > self.max_objects:
            out.append(f"{len(objects)} objects changed exceed max_objects {self.max_objects}")
        limit = self.translation_limit(scene)
        for b in bound:
            switch = CATEGORY_SWITCH.get(b.category)
            if switch is not None and not getattr(self, switch):
                out.append(f"{b.action['action']}: {b.category} changes are not allowed ({switch})")
            m = b.magnitude
            if m.get("translation", 0.0) > limit:
                out.append(
                    f"{b.action['action']}: a move of {m['translation']:.4g} m exceeds "
                    f"max_translation {limit:.4g} m"
                )
            if m.get("rotation", 0.0) > self.max_rotation:
                out.append(
                    f"{b.action['action']}: a turn of {m['rotation']:.3g} degrees exceeds "
                    f"max_rotation {self.max_rotation:g}"
                )
            if m.get("scale", 1.0) > self.max_scale:
                out.append(
                    f"{b.action['action']}: a scale factor beyond max_scale {self.max_scale:g}"
                )
            if m.get("exposure", 0.0) > self.max_exposure_change:
                out.append(
                    f"set_exposure: a change of {m['exposure']:.3g} stops exceeds "
                    f"max_exposure_change {self.max_exposure_change:g}"
                )
            if m.get("light_factor", 1.0) > self.max_light_factor:
                limit = self.max_light_factor
                out.append(f"set_light: an intensity factor beyond max_light_factor {limit:g}")
        return out


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #


@dataclass
class Proposal:
    """A proposed correction (see the module docstring)."""

    id: str
    title: str
    findings: tuple[str, ...]  # finding keys
    actions: tuple[dict[str, Any], ...]
    safety: str
    basis: str
    rationale: str
    confidence: float | None = None
    expected: list[dict[str, Any]] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    compiled: dict[str, Any] | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)
    targets: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        text = json.dumps(self.actions, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def to_dict(self, *, compiled: bool = True) -> dict[str, Any]:
        from .checks import _clean

        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "safety": self.safety,
            "basis": self.basis,
            "findings": list(self.findings),
            "actions": [dict(a) for a in self.actions],
            "rationale": self.rationale,
            "expected": _clean(self.expected),
            "validation": self.validation,
            "changes": _clean(self.changes),
            "policy": self.policy,
            "signature": self.signature,
        }
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.alternatives:
            out["alternatives"] = _clean(self.alternatives)
        if self.notes:
            out["notes"] = self.notes
        if compiled and self.compiled is not None:
            out["compiled"] = self.compiled
        return out


@dataclass
class CorrectionSet:
    """Proposals for one diagnosis, with what could not be corrected automatically."""

    id: str
    diagnosis: Diagnosis
    revision: int
    proposals: list[Proposal]
    unresolved: list[dict[str, Any]]
    policy: CorrectionPolicy

    def get(self, pid: str) -> Proposal:
        for p in self.proposals:
            if p.id == pid:
                return p
        raise FeedbackError(
            "UNKNOWN_PROPOSAL",
            f"no proposal {pid!r} in {self.id}",
            "proposals",
            hint=", ".join(p.id for p in self.proposals) or "there are no proposals",
        )

    def to_dict(self, *, compiled: bool = True) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "id": self.id,
            "diagnosis": self.diagnosis.id,
            "revision": self.revision,
            "stale": self.diagnosis.scene.revision != self.revision,
            "proposals": [p.to_dict(compiled=compiled) for p in self.proposals],
            "unresolved": self.unresolved,
            "policy": self.policy.to_dict(),
        }


@dataclass
class Draft:
    title: str
    findings: list[Finding]
    actions: list[dict[str, Any]]
    safety: str
    rationale: str
    expected: list[dict[str, Any]] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: float | None = None
    basis: str | None = None
    group: str | None = None  # drafts with the same group merge (one camera framing)


SAFETY_RANK = {"safe": 0, "verify": 1, "approval": 2}


def propose(diagnosis: Diagnosis, policy: CorrectionPolicy | None = None) -> CorrectionSet:
    """Proposals for the failing findings of ``diagnosis`` (deterministic order)."""
    from .planners import plan_finding

    policy = policy or CorrectionPolicy()
    scene = diagnosis.scene
    if diagnosis.stale:
        raise FeedbackError(
            "STALE_DIAGNOSIS",
            f"the scene changed since diagnosis {diagnosis.id} (revision {diagnosis.revision}, "
            f"now {scene.revision})",
            "diagnosis",
            hint="diagnose again",
        )
    drafts: list[Draft] = []
    unresolved: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    for finding in diagnosis.findings:
        if finding.severity == "info" and finding.basis not in ("vision", "agent"):
            continue
        found, reason = plan_finding(diagnosis, finding, policy, context)
        if found:
            drafts.extend(found)
        else:
            unresolved.append(
                {
                    "finding": finding.id,
                    "key": finding.key,
                    "code": finding.code,
                    "reason": reason,
                    "hint": finding.hint,
                }
            )
    drafts = _merge(drafts)
    drafts.sort(key=lambda d: (SAFETY_RANK[d.safety], _first_index(diagnosis, d)))
    proposals = []
    for k, d in enumerate(drafts):
        p = _finish(scene, diagnosis, d, f"P{k + 1}", policy)
        proposals.append(p)
    digest = hashlib.sha256(
        json.dumps([p.signature for p in proposals] + [diagnosis.id]).encode()
    ).hexdigest()[:12]
    return CorrectionSet(
        f"corr-{digest}", diagnosis, diagnosis.revision, proposals, unresolved, policy
    )


def _first_index(diagnosis: Diagnosis, draft: Draft) -> int:
    keys = [f.key for f in diagnosis.findings]
    return min((keys.index(f.key) for f in draft.findings if f.key in keys), default=0)


def _merge(drafts: list[Draft]) -> list[Draft]:
    out: list[Draft] = []
    groups: dict[str, Draft] = {}
    for d in drafts:
        if d.group is None:
            out.append(d)
            continue
        first = groups.get(d.group)
        if first is None:
            groups[d.group] = d
            out.append(d)
            continue
        first.findings.extend(f for f in d.findings if f not in first.findings)
        first.expected.extend(d.expected)
        if d.actions and first.actions and d.actions[0].get("targets") is not None:
            merged = _union_targets(first.actions[0].get("targets"), d.actions[0]["targets"])
            first.actions[0]["targets"] = merged
    return out


def _union_targets(a: Any, b: Any) -> list[str]:
    items = []
    for t in (a, b):
        if t is None:
            continue
        for x in t if isinstance(t, list) else [t]:
            if x not in items:
                items.append(x)
    return items


def _finish(
    scene: Scene, diagnosis: Diagnosis, d: Draft, pid: str, policy: CorrectionPolicy
) -> Proposal:
    bases = {f.basis for f in d.findings}
    basis = d.basis or (
        "vision"
        if "vision" in bases
        else "agent"
        if "agent" in bases
        else "geometry"
        if "geometry" in bases
        else next(iter(sorted(bases)), "scene")
    )
    p = Proposal(
        id=pid,
        title=d.title,
        findings=tuple(f.key for f in d.findings),
        actions=tuple(d.actions),
        safety=d.safety,
        basis=basis,
        rationale=d.rationale,
        confidence=d.confidence,
        expected=d.expected,
        validation=d.validation,
        alternatives=d.alternatives,
        notes=list(d.notes),
    )
    try:
        bound = [bind(scene, a, f"/actions/{k}") for k, a in enumerate(d.actions)]
    except FeedbackError as exc:
        p.notes.append(f"cannot apply as proposed: [{exc.code}] {exc.message}")
        p.safety = "approval"
        p.policy = {"auto": False, "violations": [exc.message]}
        return p
    p.changes = [c for b in bound for c in b.changes]
    p.targets = tuple(dict.fromkeys(b.target.id for b in bound if b.target is not None))
    p.compiled = compile_actions(bound, f"Feedback: {d.title}", diagnosis.revision)
    violations = policy.violations(scene, bound)
    auto = not violations and p.safety in policy.auto_apply and p.safety != "approval"
    p.policy = {"auto": auto, "violations": violations}
    return p
