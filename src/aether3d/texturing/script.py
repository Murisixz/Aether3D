"""Declarative texture scripts: UVs, painted textures, strokes and bakes as ordered JSON
operations, safe to accept from untrusted input.

A script is plain data (no code is ever evaluated, no file is read or written)::

    {
      "format": "aether3d.texture-script",
      "version": 1,
      "label": "Paint the crate",                     # optional: the undo step's name
      "limits": {"max_seconds": 60, "max_texels": 16777216, "max_dabs": 50000},
      "operations": [
        {"op": "unwrap", "object": "Crate", "resolution": 1024, "padding": 8},
        {"op": "add_texture", "name": "Crate color", "width": 1024, "height": 1024,
         "background": "#8a6a48", "layers": [{"name": "Dirt"}]},
        {"op": "bind", "object": "Crate", "slot": "base_color", "texture": "Crate color",
         "factor": "#ffffff"},
        {"op": "fill", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
         "color": "#3b2a1a", "where": {"groups": [2]}, "opacity": 0.5},
        {"op": "stroke", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
         "brush": {"radius": 0.05, "color": "#202020", "strength": 0.6},
         "points": [[0.5, -0.5, 0.2], [0.5, -0.5, 0.4]]},
        {"op": "bake", "target": "Crate", "sources": ["Crate high"],
         "maps": ["normal", "ao"], "resolution": 1024, "apply": true},
        {"op": "uv_report", "object": "Crate"}
      ]
    }

Operations (``op``; every other key is an argument, checked against the operation's own
list):

* UVs: ``unwrap``, ``project_uvs`` (``projection``: planar, box, cylindrical, spherical),
  ``edit_uvs`` (``operation``: transform, align, normalize, texel_density, pack, stitch),
  ``uv_report`` (read-only).
* Textures: ``add_texture`` (a painted texture, optionally with ``layers``),
  ``add_layer``, ``set_layer``, ``remove_layer``, ``clear_layer``, ``flatten``,
  ``texture_info`` (read-only), ``bind`` (a texture to a slot of the object's material,
  created when it has none; ``factor`` sets the value the texture multiplies).
* Painting: ``stroke`` (``space``: surface -- object-space points --, uv, or view with
  ``camera`` and ``view_size``), ``fill``, ``gradient``.
* Baking: ``bake`` (``apply`` binds the maps), ``transfer`` (unwrap + bake a source's
  material onto a target).

:func:`run_texture_script` applies everything or nothing, as one step of the scene's
history. Errors are :class:`~aether3d.errors.TexturingError` with a stable ``code`` and the
``location`` of the step (``operations[3].brush.radius``).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from ..errors import AetherError, ObjectNotFoundError, TexturingError

if TYPE_CHECKING:
    from ..scene import Scene

FORMAT_ID = "aether3d.texture-script"
VERSION = 1
MAX_OPERATIONS = 10_000
LIMIT_KEYS = ("max_seconds", "max_texels", "max_dabs")
DEFAULT_LIMITS = {"max_seconds": 600.0, "max_texels": 4 * 8192 * 8192, "max_dabs": 1_000_000}

_UNWRAP = {
    "seams", "split_by", "angle_limit", "method", "max_stretch", "iterations", "faces", "pack",
    "resolution", "padding", "rotate", "texels_per_unit",
}  # fmt: skip
_PROJECT = {"axis", "center", "pack", "resolution", "padding", "rotate", "texels_per_unit"}
_EDIT_UVS = {
    "transform": {"islands", "translate", "rotate", "scale", "mirror", "pivot"},
    "align": {"islands", "mode"},
    "normalize": {"islands", "keep_aspect"},
    "texel_density": {"texels_per_unit", "resolution", "islands"},
    "pack": {"resolution", "padding", "rotate", "texels_per_unit", "keep_density"},
    "stitch": {"edges", "allow_overlap"},
}
_BAKE = {
    "maps", "resolution", "samples", "cage", "depth", "padding", "ao_rays", "ao_distance",
    "thickness_rays", "thickness_distance", "curvature_radius", "height_range", "seed",
    "backend",
}  # fmt: skip
_SAMPLER = {"wrap", "wrap_u", "wrap_v", "filter", "mipmaps"}
_LAYER = {"name", "kind", "blend", "opacity", "visible", "value"}
FACTOR_PARAM = {
    "base_color": "base_color",
    "roughness": "roughness",
    "metallic": "metallic",
    "emissive": "emissive",
    "opacity": "opacity",
    "normal": "normal_strength",
    "occlusion": "occlusion_strength",
}

OPERATIONS: dict[str, tuple[set[str], set[str]]] = {
    # op: (required keys, optional keys) -- besides "op"
    "unwrap": ({"object"}, _UNWRAP),
    "project_uvs": ({"object", "projection"}, _PROJECT),
    "edit_uvs": ({"object", "operation"}, set().union(*_EDIT_UVS.values())),
    "uv_report": ({"object"}, {"resolution", "islands_limit"}),
    "add_texture": (
        {"name"},
        {"width", "height", "encoding", "background", "layers"} | _SAMPLER,
    ),
    "add_layer": ({"texture"}, _LAYER | {"index"}),
    "set_layer": ({"texture", "layer"}, {"name", "blend", "opacity", "visible", "value"}),
    "remove_layer": ({"texture", "layer"}, set()),
    "clear_layer": ({"texture", "layer"}, set()),
    "flatten": ({"texture"}, set()),
    "texture_info": ({"texture"}, set()),
    "bind": ({"object", "slot", "texture"}, {"material", "channel", "factor"}),
    "stroke": (
        {"object", "brush", "points"},
        {
            "texture",
            "layer",
            "targets",
            "space",
            "coordinates",
            "pressure",
            "padding",
            "camera",
            "view_size",
        },
    ),
    "fill": ({"object", "texture", "color"}, {"layer", "where", "opacity", "blend", "padding"}),
    "gradient": (
        {"object", "texture", "start", "end", "colors"},
        {"layer", "opacity", "coordinates"},
    ),
    "bake": ({"target"}, _BAKE | {"sources", "apply", "material"}),
    "transfer": (
        {"source", "target"},
        {
            "maps",
            "resolution",
            "unwrap",
            "own_material",
            "samples",
            "cage",
            "depth",
            "padding",
            "backend",
        },
    ),
}
READ_ONLY = ("uv_report", "texture_info")


def _err(message: str, location: str, code: str = "INVALID_SCRIPT", **kw: Any) -> TexturingError:
    return TexturingError(code, message, location, **kw)


def _dict(value: Any, loc: str) -> dict:
    if not isinstance(value, dict):
        raise _err(f"expected an object, got {type(value).__name__}", loc)
    return value


def _keys(obj: dict, allowed: set[str], loc: str, required: set[str] = frozenset()) -> None:
    unknown = sorted(set(obj) - allowed - required)
    if unknown:
        raise _err(f"unknown key(s): {', '.join(unknown)}", loc)
    missing = sorted(required - set(obj))
    if missing:
        raise _err(f"missing key(s): {', '.join(missing)}", loc)


def _limits(raw: Any) -> dict[str, float]:
    raw = _dict(raw, "script.limits")
    _keys(raw, set(LIMIT_KEYS), "script.limits")
    out = dict(DEFAULT_LIMITS)
    for key, value in raw.items():
        integer = key != "max_seconds"
        ok = isinstance(value, int) if integer else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok or not math.isfinite(value) or value <= 0:
            kind = "an integer" if integer else "a number"
            raise _err(f"{key} must be {kind} > 0", f"script.limits.{key}")
        out[key] = value
    return out


def check_script(script: Any) -> dict:
    """Check a script's structure (everything that does not need the scene); returns it."""
    script = _dict(script, "script")
    _keys(script, {"label", "limits"}, "script", {"format", "version", "operations"})
    if script["format"] != FORMAT_ID:
        raise _err(f"format must be {FORMAT_ID!r}", "script.format")
    if isinstance(script["version"], bool) or script["version"] != VERSION:
        raise _err(
            f"unsupported version {script['version']!r} (supported: {VERSION})", "script.version"
        )
    if "label" in script and (not isinstance(script["label"], str) or not script["label"].strip()):
        raise _err("label must be a non-empty string", "script.label")
    _limits(script.get("limits", {}))
    ops = script["operations"]
    if not isinstance(ops, list) or not ops:
        raise _err("operations must be a non-empty array", "script.operations")
    if len(ops) > MAX_OPERATIONS:
        raise _err(f"too many operations ({len(ops)} > {MAX_OPERATIONS})", "script.operations")
    for k, step in enumerate(ops):
        loc = f"operations[{k}]"
        step = _dict(step, loc)
        op = step.get("op")
        if not isinstance(op, str) or op not in OPERATIONS:
            raise _err(f"op must be one of {', '.join(OPERATIONS)}", f"{loc}.op")
        required, optional = OPERATIONS[op]
        _keys(step, optional | {"op"}, loc, required)
    return script


class _Budget:
    def __init__(self, limits: dict[str, float], should_stop=None) -> None:
        self.limits = limits
        self.should_stop = should_stop
        self.deadline = time.monotonic() + limits["max_seconds"]
        self.texels = 0
        self.dabs = 0

    def stop(self) -> bool:
        if self.should_stop is not None and self.should_stop():
            raise TexturingError("CANCELLED", "the texture script was cancelled")
        if time.monotonic() > self.deadline:
            raise TexturingError(
                "BUDGET_EXCEEDED",
                f"the script ran longer than max_seconds ({self.limits['max_seconds']} s)",
                hint="raise limits.max_seconds or split the script",
            )
        return False

    def add_texels(self, n: int, loc: str) -> None:
        self.texels += int(n)
        if self.texels > self.limits["max_texels"]:
            raise _err(
                f"the script's textures and bakes total {self.texels:,} texels "
                f"(max_texels {int(self.limits['max_texels']):,})",
                loc,
                "BUDGET_EXCEEDED",
                hint="use smaller resolutions or raise limits.max_texels",
            )

    def add_dabs(self, n: int, loc: str) -> None:
        self.dabs += int(n)
        if self.dabs > self.limits["max_dabs"]:
            raise _err(
                f"the script's strokes made {self.dabs:,} dabs "
                f"(max_dabs {int(self.limits['max_dabs']):,})",
                loc,
                "BUDGET_EXCEEDED",
                hint="space the dabs further apart or raise limits.max_dabs",
            )


def _object(scene: Scene, ref: Any, loc: str):
    try:
        if isinstance(ref, str):
            try:
                return scene.get_by_name(ref)
            except ObjectNotFoundError:
                return scene.get(ref)
        ref = _dict(ref, loc)
        _keys(ref, {"id", "name"}, loc)
        if len(ref) != 1 or not all(isinstance(v, str) for v in ref.values()):
            raise _err("give exactly one of id or name (a string)", loc)
        return scene.get(ref["id"]) if "id" in ref else scene.get_by_name(ref["name"])
    except ObjectNotFoundError:
        raise _err(f"object {ref!r} not found", loc, "OBJECT_NOT_FOUND") from None


def _texture(scene: Scene, ref: Any, loc: str, painted: bool = True):
    from ..materials.painted import PaintedTexture

    if not isinstance(ref, str) or not ref:
        raise _err("a texture is named by its id or name (a string)", loc)
    tex = scene.materials.find_texture(ref)  # never a file path: scripts read no files
    if tex is None:
        raise _err(f"texture {ref!r} not found", loc, "TEXTURE_NOT_FOUND")
    if painted and not isinstance(tex, PaintedTexture):
        raise _err(
            f"texture {tex.name!r} is not a painted texture",
            loc,
            "NOT_PAINTED",
            hint="create one with add_texture",
        )
    return tex


def _relocate(exc: AetherError, loc: str) -> TexturingError:
    code = getattr(exc, "code", None) or type(exc).__name__
    message = getattr(exc, "message", None) or str(exc)
    inner = getattr(exc, "location", None)
    where = (
        f"{loc}.{inner}"
        if isinstance(inner, str) and inner and not inner.startswith(("operations[", "script"))
        else (inner or loc)
    )
    return TexturingError(
        code,
        message,
        where,
        hint=getattr(exc, "hint", None),
        details=getattr(exc, "details", None)
        if isinstance(getattr(exc, "details", None), dict)
        else None,
    )


def run_texture_script(
    scene: Scene, script: dict[str, Any], *, label: str | None = None, should_stop=None
) -> dict[str, Any]:
    """Apply a texture script to ``scene`` (everything or nothing; see the module docs).
    ``should_stop`` (a callable) cancels it (``CANCELLED``, rolled back) when it returns
    true. Returns ``{"operations": [...]}``: each step's result."""
    script = check_script(script)
    budget = _Budget(_limits(script.get("limits", {})), should_stop)
    ops = script["operations"]
    label = label or script.get("label") or f"Texture script ({len(ops)} operations)"
    results = []
    t0 = time.perf_counter()
    with scene.history.transaction(label):
        for k, step in enumerate(ops):
            loc = f"operations[{k}]"
            budget.stop()
            try:
                result = _run(scene, step, loc, budget)
            except TexturingError as exc:
                if exc.location and exc.location.startswith(("operations[", "script")):
                    raise
                raise _relocate(exc, loc) from None
            except AetherError as exc:
                raise _relocate(exc, loc) from None
            results.append({"op": step["op"], **result})
    return {"operations": results, "seconds": time.perf_counter() - t0}


def _run(scene: Scene, step: dict, loc: str, budget: _Budget) -> dict[str, Any]:
    from ..materials.painted import Layer

    op = step["op"]
    args = {k: v for k, v in step.items() if k != "op"}
    if op == "unwrap":
        obj = _object(scene, args.pop("object"), f"{loc}.object")
        from .. import uv

        result = uv.unwrap(obj._uv_mesh(), should_stop=budget.stop, **args)
        obj._set_uv_mesh(result.mesh, "Unwrap UVs")
        return {**result.report, "diagnostics": [d.to_dict() for d in result.diagnostics]}
    if op == "project_uvs":
        obj = _object(scene, args.pop("object"), f"{loc}.object")
        return obj.project_uvs(args.pop("projection"), **args)
    if op == "edit_uvs":
        obj = _object(scene, args.pop("object"), f"{loc}.object")
        operation = args.pop("operation")
        allowed = _EDIT_UVS.get(operation)
        if allowed is None:
            raise _err(f"operation must be one of {', '.join(_EDIT_UVS)}", f"{loc}.operation")
        _keys(args, allowed, loc)
        return obj.edit_uvs(operation, **args)
    if op == "uv_report":
        obj = _object(scene, args.pop("object"), f"{loc}.object")
        return obj.uv_report(**args)
    if op == "add_texture":
        layers = args.pop("layers", [])
        if not isinstance(layers, list):
            raise _err("layers must be an array", f"{loc}.layers")
        width = args.get("width", 1024)
        height = args.get("height", width)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            budget.add_texels(width * height, loc)
        tex = scene.add_painted_texture(args.pop("name"), **args)
        for i, layer in enumerate(layers):
            layer = _dict(layer, f"{loc}.layers[{i}]")
            _keys(layer, _LAYER, f"{loc}.layers[{i}]")
            tex.add_layer(**layer)
        return {"texture": tex.info()}
    if op in ("add_layer", "set_layer", "remove_layer", "clear_layer", "flatten", "texture_info"):
        tex = _texture(scene, args.pop("texture"), f"{loc}.texture")
        if op == "add_layer":
            layer: Layer = tex.add_layer(**args)
            return {"layer": layer.id, "texture": tex.info()}
        if op == "set_layer":
            tex.set_layer(args.pop("layer"), **args)
        elif op == "remove_layer":
            tex.remove_layer(args.pop("layer"))
        elif op == "clear_layer":
            tex.clear_layer(args.pop("layer"))
        elif op == "flatten":
            tex.flatten()
        return {"texture": tex.info()}
    if op == "bind":
        return _bind(scene, args, loc)
    if op == "stroke":
        return _stroke(scene, args, loc, budget)
    if op == "fill":
        from .paint import fill

        obj = _object(scene, args.pop("object"), f"{loc}.object")
        target = _target(scene, args.pop("texture"), args.pop("layer", None), loc)
        return fill(
            scene,
            obj,
            target,
            args.pop("color"),
            where=args.get("where"),
            opacity=args.get("opacity", 1.0),
            blend_mode=args.get("blend", "normal"),
            padding=args.get("padding", 2),
        )
    if op == "gradient":
        from .paint import gradient

        obj = _object(scene, args.pop("object"), f"{loc}.object")
        target = _target(scene, args.pop("texture"), args.pop("layer", None), loc)
        return gradient(
            scene,
            obj,
            target,
            args["start"],
            args["end"],
            args["colors"],
            coordinates=args.get("coordinates", "world"),
            opacity=args.get("opacity", 1.0),
        )
    if op == "bake":
        from ..uv.unwrap import _check_resolution
        from .bake import apply_bake, bake

        target = _object(scene, args.pop("target"), f"{loc}.target")
        sources = args.pop("sources", None)
        if sources is not None:
            if not isinstance(sources, list):
                raise _err("sources must be an array of objects", f"{loc}.sources")
            sources = [_object(scene, s, f"{loc}.sources[{i}]") for i, s in enumerate(sources)]
        apply = args.pop("apply", False)
        if not isinstance(apply, bool):
            raise _err("apply must be true or false", f"{loc}.apply")
        material = args.pop("material", None)
        w, h = _check_resolution(args.get("resolution", 1024))
        budget.add_texels(w * h, loc)
        maps = args.pop("maps", ["normal"])
        result = bake(scene, target, sources, maps, should_stop=budget.stop, **args)
        out = {
            "report": result.report,
            "diagnostics": [d.to_dict() for d in result.diagnostics],
        }
        if apply:
            mat = None
            if material is not None:
                if not isinstance(material, str):
                    raise _err("material is a material id or name", f"{loc}.material")
                mat = scene.materials.material(material)
            applied = apply_bake(scene, target, result, material=mat)
            out["applied"] = {
                "material": applied["material"].name,
                "textures": {k: t.name for k, t in applied["textures"].items()},
            }
        return out
    if op == "transfer":
        from ..uv.unwrap import _check_resolution
        from .transfer import DEFAULT_MAPS, transfer_textures

        w, h = _check_resolution(args.get("resolution", 1024))
        budget.add_texels(w * h, loc)
        source = _object(scene, args.pop("source"), f"{loc}.source")
        target = _object(scene, args.pop("target"), f"{loc}.target")
        maps = args.pop("maps", list(DEFAULT_MAPS))
        return transfer_textures(scene, source, target, maps, should_stop=budget.stop, **args)
    raise _err(f"unknown op {op!r}", f"{loc}.op")  # unreachable: checked


def _target(scene: Scene, texture: Any, layer: Any, loc: str):
    from .paint import Target

    tex = _texture(scene, texture, f"{loc}.texture")
    if layer is None:
        paint = [x for x in tex.layers if x.kind == "paint"]
        if not paint:
            raise _err(
                f"texture {tex.name!r} has no paint layer",
                f"{loc}.layer",
                "NO_PAINT_LAYER",
                hint="add one with add_layer",
            )
        layer = paint[-1].id
    elif not isinstance(layer, str):
        raise _err("layer is a layer id or name", f"{loc}.layer")
    tex.layer(layer)
    return Target(tex, layer, None)


def _stroke(scene: Scene, args: dict, loc: str, budget: _Budget) -> dict[str, Any]:
    from .paint import PaintView, Target, _color, stroke

    obj = _object(scene, args.pop("object"), f"{loc}.object")
    targets = args.pop("targets", None)
    if targets is not None:
        if "texture" in args or "layer" in args:
            raise _err("give either texture (and layer) or targets", loc)
        if not isinstance(targets, list) or not targets:
            raise _err("targets must be a non-empty array", f"{loc}.targets")
        items = []
        for i, t in enumerate(targets):
            tl = f"{loc}.targets[{i}]"
            t = _dict(t, tl)
            _keys(t, {"layer", "color"}, tl, {"texture"})
            base = _target(scene, t["texture"], t.get("layer"), tl)
            color = None if "color" not in t else _color(t["color"], f"{tl}.color")
            items.append(Target(base.texture, base.layer, color))
    else:
        if "texture" not in args:
            raise _err("a stroke needs a texture (or targets)", f"{loc}.texture")
        items = [_target(scene, args.pop("texture"), args.pop("layer", None), loc)]
    space = args.pop("space", "surface")
    view = None
    camera = args.pop("camera", None)
    size = args.pop("view_size", None)
    if space == "view":
        if camera is None:
            raise _err("a view stroke needs a camera", f"{loc}.camera")
        if size is None:
            size = [512, 512]
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(
                isinstance(v, int) and not isinstance(v, bool) and 16 <= v <= 4096 for v in size
            )
        ):
            raise _err("view_size is [width, height], integers in [16, 4096]", f"{loc}.view_size")
        cam = _object(scene, camera, f"{loc}.camera")
        view = PaintView.render(scene, cam, size[0], size[1])
    elif camera is not None or size is not None:
        raise _err("camera and view_size are for space 'view'", loc)
    report = stroke(
        scene,
        obj,
        items,
        args.pop("brush"),
        args.pop("points"),
        space=space,
        coordinates=args.pop("coordinates", "world"),
        pressure=args.pop("pressure", None),
        view=view,
        padding=args.pop("padding", 2),
        should_stop=budget.stop,
    )
    budget.add_dabs(report.dabs, loc)
    return report.to_dict()


def _bind(scene: Scene, args: dict, loc: str) -> dict[str, Any]:
    from ..materials.material import SLOTS

    obj = _object(scene, args.pop("object"), f"{loc}.object")
    slot = args.pop("slot")
    if slot not in SLOTS:
        raise _err(f"slot must be one of {', '.join(SLOTS)}", f"{loc}.slot")
    tex = _texture(scene, args.pop("texture"), f"{loc}.texture", painted=False)
    material_ref = args.pop("material", None)
    if material_ref is not None:
        if not isinstance(material_ref, str):
            raise _err("material is a material id or name", f"{loc}.material")
        material = scene.materials.material(material_ref)
        obj.material = material
    else:
        material = obj.material
        if material is None:
            material = scene.create_material(scene.materials._unique_material_name(obj.name))
            obj.material = material
    options = {}
    if "channel" in args:
        options["channel"] = args.pop("channel")
    material.set_texture(slot, tex, **options)
    if "factor" in args:
        setattr(material, FACTOR_PARAM[slot], args.pop("factor"))
    return {"material": material.name, "slot": slot, "texture": tex.name}
