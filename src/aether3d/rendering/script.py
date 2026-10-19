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
* ``add_light`` -- ``type`` (``directional``/``point``/``spot``/``area``), ``name``,
  ``color``, ``intensity``, ``range``, ``falloff``, ``radius``, ``angle``, ``spot_size``,
  ``spot_blend``, ``shape``, ``size``, ``size_y``, ``spread``, ``units``, ``temperature``,
  ``use_temperature``, ``exposure``, ``shadow``, ``include``, ``exclude`` (object ids or
  names), ``parent``, ``position``, ``rotation_euler``, ``direction``, ``look_at``, ``up``.
* ``light_rig`` -- ``subject`` and a ``preset`` (``three_point``, ``loop``,
  ``rembrandt``, ``butterfly``, ``split``, ``high_key``, ``low_key``, ``silhouette``);
  ``camera``, ``time``, ``distance``, ``level``, ``ratio``, ``rim``, ``temperature``,
  ``prefix`` (:func:`aether3d.rendering.studio.light_rig`).
* ``set_camera`` / ``set_light`` -- ``camera``/``light`` plus any of their parameters.
* ``set_object`` -- ``object`` plus ``position``, ``rotation_euler``, ``scale``,
  ``look_at`` or ``look_along`` (with ``up``), ``color``, ``specular``, ``shininess``,
  ``visible``; ``tags`` (replaces them), ``add_tags``, ``remove_tags``, ``metadata``
  (entries to set; ``null`` removes one).
* ``frame_camera`` -- ``camera`` (default: the active camera), ``targets``, ``direction``,
  ``up``, ``margin``, ``fit_clipping``.
* ``set_active_camera`` -- ``camera`` (or ``null``).
* ``set_world`` -- ``background``, ``ambient_color``, ``ambient_intensity``,
  ``environment`` (an object of environment fields; the others keep their values).
* ``set_render_settings`` -- ``width``, ``height``, ``samples``, ``exposure``, ``cull``,
  ``default_lights``.
* ``remove`` -- ``object`` (and its children).
* ``info`` -- read-only summary of cameras, lights, world and settings.
* ``query`` -- a read-only spatial query: ``query`` names it and the other keys are
  its arguments, as in :func:`aether3d.spatial.run_query` (``{"op": "query",
  "query": "raycast", "origin": [0, 0, 10], "direction": [0, 0, -1]}``). It sees the
  scene as the operations before it left it.

Materials and textures (``material``/``texture`` are ids or names):

* ``create_material`` -- ``name``, ``model``, ``preset`` and any material
  parameters or preset options (``{"op": "create_material", "name": "Wall",
  "preset": "brick", "scale": 2, "roughness": 0.9}``).
* ``set_material`` -- ``material`` plus parameters, ``model`` or a new ``name``.
* ``remove_material`` -- ``material`` (``force`` unassigns it first).
* ``assign_material`` -- ``objects`` (one or a list), ``material`` (``null``
  unassigns) and optionally a face ``group``.
* ``set_texture`` -- ``material``, ``slot`` and one of ``texture`` (an
  existing one), ``image`` (a PNG/JPEG path), ``procedural`` (a node graph or
  node) or ``"clear": true``; binding options ``channel``, ``coordinates``,
  ``projection``, ``offset``, ``scale``, ``rotation``, ``blend``; for new
  textures ``texture_name``, ``embed``, ``wrap``, ``filter``, ``mipmaps``,
  ``color_space``.
* ``add_texture`` -- ``name`` and ``image`` or ``procedural`` (plus ``embed`` and
  sampler settings).
* ``set_texture_settings`` -- ``texture`` plus ``name``, sampler settings,
  ``embed: true`` or a new ``graph``.
* ``remove_texture`` -- ``texture`` (``force`` removes its bindings).
* ``convert_colors`` -- ``objects`` (default: all) and ``dedupe``: object
  colors become equivalent simple materials.
* ``materials`` -- read-only summary of materials and textures.

Image paths in a script are resolved inside the project's directory and may
not leave it (scripts may come from anywhere); the CLI's own material
commands accept paths as typed.

:func:`run_scene_script` is atomic: if any operation fails, every change the
script made is undone (created objects are removed, removed objects are put
back, and changed values are restored in place) and a :class:`ScriptError`
names the failing operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..editing.script import ScriptError
from ..errors import AetherError, MaterialError, RenderError

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

FORMAT_ID = "aether3d.scene-script"
VERSION = 1
MAX_OPERATIONS = 10_000

_CAMERA = {
    "projection", "fov", "fov_axis", "ortho_scale", "near", "far", "aspect_ratio", "sensor",
    "shift_x", "shift_y", "fstop", "focus_distance", "focus_object", "aperture_blades",
    "aperture_rotation", "aperture_ratio", "focal_length",
}  # fmt: skip
_LIGHT = {
    "color", "intensity", "range", "falloff", "radius", "angle", "spot_size", "spot_blend",
    "shape", "size", "size_y", "spread", "units", "temperature", "use_temperature",
    "exposure", "shadow", "include", "exclude",
}  # fmt: skip
_PLACE = {"parent", "position", "rotation_euler", "look_at", "up"}
_SPECS: dict[str, tuple[set[str], set[str]]] = {
    "add_camera": (
        set(),
        {"name", "active", "frame", "frame_targets", "frame_direction", "frame_margin"}
        | _CAMERA
        | _PLACE,
    ),
    "add_light": (set(), {"type", "name", "direction"} | _LIGHT | _PLACE),
    "light_rig": (
        {"subject"},
        {"preset", "camera", "time", "distance", "level", "ratio", "rim", "temperature", "prefix"},
    ),  # fmt: skip
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
            "tags",
            "add_tags",
            "remove_tags",
            "metadata",
        },
    ),
    "frame_camera": (set(), {"camera", "targets", "direction", "up", "margin", "fit_clipping"}),
    "set_active_camera": ({"camera"}, set()),
    "set_world": (set(), {"background", "ambient_color", "ambient_intensity", "environment"}),
    "set_render_settings": (
        set(),
        {"width", "height", "samples", "exposure", "cull", "default_lights"},
    ),
    "remove": ({"object"}, set()),
    "info": (set(), set()),
    "query": ({"query"}, set()),  # its arguments are checked by spatial.run_query
}


def _material_specs() -> dict[str, tuple[set[str], set[str]]]:
    from ..materials.material import PARAM_SPECS, TextureBinding
    from ..materials.presets import COMMON_OPTIONS, PRESETS

    params = set(PARAM_SPECS)
    preset_options = set(COMMON_OPTIONS) | {o for _, _, extra in PRESETS.values() for o in extra}
    sampler = {"wrap", "filter", "mipmaps", "color_space"}
    return {
        "create_material": (set(), {"name", "model", "preset"} | params | preset_options),
        "set_material": ({"material"}, {"name", "model"} | params),
        "remove_material": ({"material"}, {"force"}),
        "assign_material": ({"objects", "material"}, {"group"}),
        "set_texture": (
            {"material", "slot"},
            {"texture", "image", "procedural", "clear", "texture_name", "embed"}
            | set(TextureBinding.FIELDS)
            | sampler,
        ),
        "add_texture": ({"name"}, {"image", "procedural", "embed"} | sampler),
        "set_texture_settings": ({"texture"}, {"name", "embed", "graph"} | sampler),
        "remove_texture": ({"texture"}, {"force"}),
        "convert_colors": (set(), {"objects", "dedupe"}),
        "materials": (set(), set()),
    }


_SPECS.update(_material_specs())


def run_scene_script(
    scene: Scene,
    script: Any,
    *,
    base_dir: Any = None,
    trusted_paths: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    """Apply a scene-setup script atomically; returns per-operation results.

    The whole script is one step of ``scene.history`` (``label`` names it). Image
    paths are resolved inside ``base_dir`` (default: the project's directory, else
    the current one) and may not leave it, unless ``trusted_paths`` (paths typed
    by the user, as the CLI's material commands pass them).
    """
    if not isinstance(script, dict):
        raise ScriptError("the script must be a JSON object", "script")
    unknown = sorted(set(script) - {"format", "version", "operations", "label"})
    if unknown:
        raise ScriptError(f"unknown key(s): {', '.join(unknown)}", "script")
    if "label" in script and (not isinstance(script["label"], str) or not script["label"].strip()):
        raise ScriptError("label must be a non-empty string", "script.label")
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
    from pathlib import Path

    root = Path(base_dir) if base_dir is not None else scene.materials.base_dir or Path.cwd()
    journal = _Journal((root.absolute(), trusted_paths))
    results = []
    # One step of the scene's history, all or nothing: if an operation fails, the
    # transaction restores everything the earlier ones changed.
    count = len(operations)
    label = label or script.get("label") or f"Scene script ({count} operation{'s' * (count != 1)})"
    with scene.history.transaction(label):
        for i, step in enumerate(operations):
            results.append(_run(scene, step, f"operations[{i}]", journal))
    return {"operations": results}


class _Journal:
    """What operations of one script run share: where image paths may point."""

    def __init__(self, paths: tuple) -> None:
        self.paths = paths


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
    if op == "query":
        extra = []
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


def _links(scene: Scene, params: dict[str, Any]) -> dict[str, Any]:
    """Light linking lists of object ids or names, as ids."""
    for key in ("include", "exclude"):
        if key in params:
            refs = params[key]
            if not isinstance(refs, list):
                raise ScriptError(f"{key} is a list of objects (ids or names)", key)
            params[key] = [_obj(scene, r).id for r in refs]
    return params


def _lens(scene: Scene, params: dict[str, Any]) -> dict[str, Any]:
    """Camera parameters with ``focus_object`` given by name or id resolved to its id."""
    if params.get("focus_object") is not None:
        params = {**params, "focus_object": _obj(scene, params["focus_object"]).id}
    return params


def _dispatch(scene: Scene, op: str, s: dict[str, Any], journal: _Journal) -> dict[str, Any]:
    if op == "query":
        from ..spatial import run_query

        return run_query(scene, {k: v for k, v in s.items() if k != "op"})
    if op == "add_camera":
        params = _lens(scene, {k: s[k] for k in _CAMERA if k in s})
        obj = scene.add_camera(
            s.get("name"),
            parent=_obj(scene, s["parent"]) if "parent" in s else None,
            position=s.get("position"),
            look_at=_target(scene, s["look_at"]) if "look_at" in s else None,
            up=s.get("up", (0.0, 0.0, 1.0)),
            active=s.get("active"),
            **params,
        )
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
    if op == "light_rig":
        options = {k: s[k] for k in ("preset", "time", "distance", "level", "ratio", "rim",
                                     "temperature", "prefix") if k in s}  # fmt: skip
        if "camera" in s:
            options["camera"] = _obj(scene, s["camera"])
        return scene.light_rig(_obj(scene, s["subject"]), **options)
    if op == "add_light":
        params = _links(scene, {k: s[k] for k in _LIGHT if k in s})
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
        if "rotation_euler" in s:
            obj.rotation_euler = s["rotation_euler"]
        return {"object": _ref(obj)}
    if op == "set_camera":
        obj = _obj(scene, s["camera"])
        if obj.camera is None:
            raise ScriptError(f"object {obj.name!r} is not a camera", "camera", "NOT_A_CAMERA")
        obj.camera.configure(**_lens(scene, {k: s[k] for k in _CAMERA if k in s}))
        return {"object": _ref(obj), "camera": obj.camera.to_dict()}
    if op == "set_light":
        obj = _obj(scene, s["light"])
        if obj.light is None:
            raise ScriptError(f"object {obj.name!r} is not a light", "light", "NOT_A_LIGHT")
        obj.light.configure(**_links(scene, {k: s[k] for k in {"type"} | _LIGHT if k in s}))
        return {"object": _ref(obj), "light": obj.light.to_dict()}
    if op == "set_object":
        obj = _obj(scene, s["object"])
        if "look_at" in s and "look_along" in s:
            raise ScriptError("give look_at or look_along, not both", "object")
        appearance = {k: s[k] for k in ("color", "specular", "shininess") if k in s}
        if appearance:
            obj._check_no_material(next(iter(appearance)))
            obj.appearance.configure(**appearance)
        if "visible" in s:
            obj.visible = s["visible"]
        if "tags" in s:
            obj.tags = _string_list(s["tags"], "tags")
        if "add_tags" in s:
            obj.add_tags(*_string_list(s["add_tags"], "add_tags"))
        if "remove_tags" in s:
            obj.remove_tags(*_string_list(s["remove_tags"], "remove_tags"))
        if "metadata" in s:
            if not isinstance(s["metadata"], dict):
                raise ScriptError("metadata must be an object", "metadata")
            for key, value in s["metadata"].items():  # null removes an entry
                if value is None:
                    obj.remove_metadata(key)
                else:
                    obj.set_metadata(key, value)
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
        result = {"object": _ref(obj), **_object_summary(obj)}
        if any(k in s for k in ("tags", "add_tags", "remove_tags", "metadata")):
            result.update(tags=list(obj.tags), metadata=obj.metadata)
        return result
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
        scene.world.configure(**{k: s[k] for k in _SPECS[op][1] if k in s})
        return {"world": scene.world.to_dict()}
    if op == "set_render_settings":
        scene.render_settings.configure(**{k: s[k] for k in _SPECS[op][1] if k in s})
        return {"render_settings": scene.render_settings.to_dict()}
    if op == "remove":
        obj = _obj(scene, s["object"])
        removed = [obj.id, *(d.id for d in obj.iter_descendants())]
        scene.remove(obj)
        return {"removed": removed}
    if op == "info":
        return rendering_summary(scene)
    if op in _MATERIAL_OPS:
        return _material_op(scene, op, s, journal)
    raise AssertionError(op)  # pragma: no cover


_MATERIAL_OPS = set(_material_specs())


def _image_path(value: Any, journal: _Journal) -> Any:
    """A script's image path: inside the project directory unless paths are trusted."""
    from ..materials.textures import check_relative_path

    root, trusted = journal.paths
    if not isinstance(value, str) or not value:
        raise MaterialError("INVALID_ARGUMENT", "image must be a file path", "image")
    if trusted:
        return value
    rel = check_relative_path(value, "image")
    path = (root / rel).absolute()
    if not path.resolve().is_relative_to(root.resolve()):
        raise MaterialError("UNSAFE_PATH", f"image {value!r} leaves the project directory", "image")
    return path


def _material_ref(material: Any) -> dict[str, str]:
    return {"id": material.id, "name": material.name}


def _material_op(scene: Scene, op: str, s: dict[str, Any], journal: _Journal) -> dict[str, Any]:
    library = scene.materials
    if op == "create_material":
        options = {k: v for k, v in s.items() if k not in ("op", "name", "model", "preset")}
        if "preset" in s:
            material = library.create_material(s.get("name"), preset=s["preset"], **options)
        else:
            material = library.create_material(
                s.get("name"), model=s.get("model", "standard"), **options
            )
        return {"material": _material_ref(material), "textures": sorted(material.textures)}
    if op == "set_material":
        material = library.material(s["material"])
        if "name" in s:
            material.name = s["name"]
        changes = {k: v for k, v in s.items() if k not in ("op", "material", "name")}
        if changes:
            material.configure(**changes)
        return {"material": _material_ref(material), "parameters": material.params}
    if op == "remove_material":
        material = library.material(s["material"])
        ref = _material_ref(material)
        unassigned = library.remove_material(material, force=bool(s.get("force", False)))
        return {"removed": ref, "unassigned": unassigned}
    if op == "assign_material":
        objects = s["objects"] if isinstance(s["objects"], list) else [s["objects"]]
        targets = library.assign(objects, s["material"], group=s.get("group"))
        material = None if s["material"] is None else _material_ref(library.material(s["material"]))
        return {
            "material": material,
            "objects": [_ref(o) for o in targets],
            "group": s.get("group"),
        }
    if op == "set_texture":
        material = library.material(s["material"])
        slot = s["slot"]
        sources = [k for k in ("texture", "image", "procedural", "clear") if k in s]
        if len(sources) != 1:
            raise MaterialError(
                "INVALID_ARGUMENT",
                "give exactly one of texture, image, procedural and clear",
                "set_texture",
            )
        if "clear" in s:
            material.clear_texture(slot)
            return {"material": _material_ref(material), "slot": slot, "texture": None}
        from ..materials.material import TextureBinding

        binding = {k: s[k] for k in TextureBinding.FIELDS if k in s}
        sampler = {k: s[k] for k in ("wrap", "filter", "mipmaps", "color_space") if k in s}
        if "texture" in s:
            if sampler or "embed" in s or "texture_name" in s:
                raise MaterialError(
                    "INVALID_ARGUMENT",
                    "sampler settings, embed and texture_name apply to new textures; use "
                    "set_texture_settings for an existing one",
                    "set_texture",
                )
            texture = library.texture(s["texture"])
        elif "image" in s:
            texture = library.add_image_texture(
                s.get("texture_name"),
                _image_path(s["image"], journal),
                embed=bool(s.get("embed", False)),
                **sampler,
            )
        else:
            texture = library.add_procedural_texture(
                s.get("texture_name") or library._unique_texture_name(f"{material.name} {slot}"),
                s["procedural"],
            )
        result = material.set_texture(slot, texture, **binding)
        return {
            "material": _material_ref(material),
            "slot": slot,
            "texture": {"id": texture.id, "name": texture.name, "type": texture.kind},
            "binding": result.to_dict(),
        }
    if op == "add_texture":
        sampler = {k: s[k] for k in ("wrap", "filter", "mipmaps", "color_space") if k in s}
        if ("image" in s) == ("procedural" in s):
            raise MaterialError("INVALID_ARGUMENT", "give image or procedural", "add_texture")
        if "image" in s:
            texture = library.add_image_texture(
                s["name"],
                _image_path(s["image"], journal),
                embed=bool(s.get("embed", False)),
                **sampler,
            )
        else:
            if sampler or "embed" in s:
                raise MaterialError(
                    "INVALID_ARGUMENT", "sampler settings and embed apply to images", "add_texture"
                )
            texture = library.add_procedural_texture(s["name"], s["procedural"])
        return {"texture": texture.info()}
    if op == "set_texture_settings":
        texture = library.texture(s["texture"])
        if "name" in s:
            texture.name = s["name"]
        sampler = {k: s[k] for k in ("wrap", "filter", "mipmaps", "color_space") if k in s}
        if texture.kind == "image":
            if "graph" in s:
                raise MaterialError(
                    "INVALID_ARGUMENT", "graph applies to procedural textures", "graph"
                )
            if sampler:
                texture.configure(**sampler)
            if s.get("embed"):
                texture.embed()
        else:
            if sampler or "embed" in s:
                raise MaterialError(
                    "INVALID_ARGUMENT", "sampler settings and embed apply to images", "texture"
                )
            if "graph" in s:
                texture.set_graph(s["graph"])
        return {"texture": texture.info()}
    if op == "remove_texture":
        texture = library.texture(s["texture"])
        info = {"id": texture.id, "name": texture.name}
        unbound = library.remove_texture(texture, force=bool(s.get("force", False)))
        return {"removed": info, "unbound_materials": unbound}
    if op == "convert_colors":
        objects = s.get("objects")
        created = library.convert_appearances(objects, dedupe=s.get("dedupe", True))
        return {"created": [_material_ref(m) for m in created]}
    if op == "materials":
        return library.inspect()
    raise AssertionError(op)  # pragma: no cover


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ScriptError(f"{key} must be an array of strings", key)
    return value


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
