"""CLI commands for materials and textures: ``aether3d material ...`` and ``aether3d texture ...``.

Writing commands run as scene-script operations (``rendering/script.py``), so
they are atomic: on any error nothing is saved. They accept ``--dry-run`` and
``--output``, like the other project-editing commands, and ``--json`` for a
machine-readable result. Paths typed here (``--image``) are used as given;
saving still refuses texture files outside the project's directory unless
they are embedded (``--embed``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cli import (
    EXIT_GEOMETRY_ERRORS,
    EXIT_OK,
    CommandError,
    CommandOutput,
    _diagnostic_lines,
    _load_scene,
    _render_error,
    _run_scene,
    _saved,
)
from .errors import MaterialError, RenderError

MAX_GRAPH_BYTES = 1024 * 1024


def _material_error(exc: MaterialError) -> CommandError:
    details: dict[str, Any] = {"code": exc.code}
    if exc.hint:
        details["hint"] = exc.hint
    if exc.details:
        details["details"] = exc.details
    return CommandError("MaterialError", exc.message, exc.location, details)


def _value(text: str) -> Any:
    """A ``--set`` value: JSON when it parses (numbers, booleans, lists), else a string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _assignments(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise CommandError("UsageError", f"expected KEY=VALUE, got {item!r}")
        out[key.strip()] = _value(value)
    return out


def _graph(text: str) -> Any:
    """A procedural graph from JSON text, ``@file`` or ``-`` (stdin)."""
    if text == "-":
        raw = sys.stdin.read(MAX_GRAPH_BYTES + 1)
    elif text.startswith("@"):
        path = Path(text[1:])
        try:
            if path.stat().st_size > MAX_GRAPH_BYTES:
                raise CommandError(
                    "ResourceLimitError", f"graph file is larger than {MAX_GRAPH_BYTES} bytes"
                )
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CommandError(
                type(exc).__name__, f"cannot read {path}: {exc.strerror or exc}"
            ) from None
    else:
        raw = text
    if len(raw) > MAX_GRAPH_BYTES:
        raise CommandError("ResourceLimitError", f"graph is larger than {MAX_GRAPH_BYTES} bytes")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandError(
            "UsageError", f"the graph is not valid JSON: {exc}", "graph", {"code": "INVALID_JSON"}
        ) from None


def _binding_options(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("channel", "coordinates", "projection", "rotation", "blend"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    for key in ("scale", "offset"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value[0] if len(value) == 1 else list(value)
    return out


def _sampler_options(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "wrap", None):
        out["wrap"] = args.wrap[0] if len(args.wrap) == 1 else list(args.wrap)
    if getattr(args, "filter", None):
        out["filter"] = args.filter
    if getattr(args, "mipmaps", None) is not None:
        out["mipmaps"] = args.mipmaps
    if getattr(args, "color_space", None):
        out["color_space"] = args.color_space
    return out


def _run(args: argparse.Namespace, operations: list[dict[str, Any]], command: str):
    return _run_scene(args, operations, command, trusted_paths=True)


# --------------------------------------------------------------------------- #
# material
# --------------------------------------------------------------------------- #


def cmd_material_list(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    library = scene.materials
    materials = []
    for m in library.materials:
        users = library.users(m)
        materials.append(
            {
                "id": m.id,
                "name": m.name,
                "model": m.model,
                "textures": {
                    slot: library.texture(b.texture).name for slot, b in m.textures.items()
                },
                "users": [
                    {"id": o.id, "name": o.name, **({} if g is None else {"group": g})}
                    for o, g in users
                ],
            }
        )
    textures = [
        {"id": t.id, "name": t.name, "type": t.kind, "source": getattr(t, "source", "procedural")}
        for t in library.textures
    ]
    colored = [o.name for o in scene if o.mesh is not None and o._material is None]
    lines = [f'Materials in "{scene.name}" ({len(materials)}):']
    for m in materials:
        slots = ", ".join(f"{slot}={name}" for slot, name in m["textures"].items()) or "no textures"
        users = ", ".join(
            u["name"] + (f"[{u['group']}]" if "group" in u else "") for u in m["users"]
        )
        lines.append(f"  {m['name']} ({m['model']}): {slots}; used by {users or 'nothing'}")
    lines.append(
        f"Textures ({len(textures)}): "
        + (", ".join(f"{t['name']} ({t['type']})" for t in textures) or "none")
    )
    if colored:
        lines.append(f"Objects with their own color ({len(colored)}): " + ", ".join(colored[:20]))
    data = {
        "path": str(path),
        **info,
        "materials": materials,
        "textures": textures,
        "colored_objects": colored,
    }
    return data, "\n".join(lines), EXIT_OK


def cmd_material_presets(args: argparse.Namespace) -> CommandOutput:
    from .materials import preset_info

    presets = preset_info()
    lines = [
        f"{p['name']}: {p['description']} (options: {', '.join(p['options'])})" for p in presets
    ]
    return {"presets": presets}, "\n".join(lines), EXIT_OK


def cmd_material_schema(args: argparse.Namespace) -> CommandOutput:
    from .materials import material_schema, node_types

    schema = material_schema()
    nodes = node_types()
    lines = ["Models: " + ", ".join(schema["models"]), "Parameters:"]
    for p in schema["parameters"]:
        lines.append(
            f"  {p['name']} ({p['type']}, default {p['default']}; {'/'.join(p['models'])})"
        )
    lines.append("Slots: " + ", ".join(s["name"] for s in schema["slots"]))
    lines.append("Procedural node types: " + ", ".join(n["type"] for n in nodes))
    return {"material": schema, "node_types": nodes}, "\n".join(lines), EXIT_OK


def cmd_material_create(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "create_material", "name": args.name, **_assignments(args.set)}
    if args.preset:
        op["preset"] = args.preset
    if args.model:
        op["model"] = args.model
    _, data, _ = _run(args, [op], "material create")
    created = data["operations"][0]
    textures = f" with {', '.join(created['textures'])} textures" if created["textures"] else ""
    return (
        data,
        f"Created material {created['material']['name']!r}{textures}\n{_saved(data)}",
        EXIT_OK,
    )


def cmd_material_set(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "set_material", "material": args.material, **_assignments(args.set)}
    if args.rename:
        op["name"] = args.rename
    if args.model:
        op["model"] = args.model
    if len(op) == 2:
        raise CommandError(
            "UsageError", "nothing to change; give --set KEY=VALUE, --model or --rename"
        )
    _, data, _ = _run(args, [op], "material set")
    return (
        data,
        f"Updated material {data['operations'][0]['material']['name']!r}\n{_saved(data)}",
        EXIT_OK,
    )


def cmd_material_assign(args: argparse.Namespace) -> CommandOutput:
    material = None if args.material == "none" else args.material
    op: dict[str, Any] = {
        "op": "assign_material",
        "objects": list(args.objects),
        "material": material,
    }
    if args.group is not None:
        op["group"] = args.group
    _, data, _ = _run(args, [op], "material assign")
    result = data["operations"][0]
    names = ", ".join(o["name"] for o in result["objects"])
    what = (
        "Removed the material from"
        if material is None
        else f"Assigned {result['material']['name']!r} to"
    )
    group = f" (face group {args.group})" if args.group is not None else ""
    return data, f"{what} {names}{group}\n{_saved(data)}", EXIT_OK


def cmd_material_texture(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "set_texture", "material": args.material, "slot": args.slot}
    sources = [k for k in ("image", "procedural", "texture") if getattr(args, k) is not None] + (
        ["clear"] if args.clear else []
    )
    if len(sources) != 1:
        raise CommandError(
            "UsageError", "give exactly one of --image, --procedural, --texture and --clear"
        )
    if args.clear:
        op["clear"] = True
    elif args.image is not None:
        op["image"] = args.image
        op.update(_sampler_options(args))
        if args.embed:
            op["embed"] = True
    elif args.procedural is not None:
        op["procedural"] = _graph(args.procedural)
    else:
        op["texture"] = args.texture
    if args.texture_name and not args.clear:
        op["texture_name"] = args.texture_name
    op.update(_binding_options(args))
    _, data, _ = _run(args, [op], "material texture")
    result = data["operations"][0]
    if result["texture"] is None:
        text = f"Cleared the {args.slot} texture of {result['material']['name']!r}"
    else:
        text = (
            f"Bound {result['texture']['type']} texture {result['texture']['name']!r} to "
            f"{result['material']['name']}.{args.slot}"
        )
    return data, f"{text}\n{_saved(data)}", EXIT_OK


def cmd_material_inspect(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    library = scene.materials
    try:
        if args.material:
            reports = [library.inspect_material(args.material)]
        else:
            reports = [library.inspect_material(m) for m in library.materials]
    except MaterialError as exc:
        raise _material_error(exc) from None
    lines = []
    for r in reports:
        lines.append(f"{r['name']} ({r['model']}) [{r['id']}]")
        for key, value in r["parameters"].items():
            lines.append(f"  {key}: {value}")
        for slot, t in r["textures"].items():
            status = t.get("status", {})
            ok = "ok" if status.get("ok", "error" not in t) else f"{status.get('code', 'error')}"
            lines.append(
                f"  texture {slot}: {t.get('texture_name', t['texture'])} "
                f"({t.get('texture_type', '?')}, {ok})"
            )
        users = ", ".join(u["name"] for u in r["users"]) or "nothing"
        lines.append(f"  used by: {users}")
        for p in r["problems"]:
            lines.append(f"  {p['severity']} {p['code']}: {p['message']}")
    data = {"path": str(path), **info, "materials": reports}
    if args.material:
        data["material"] = reports[0]
    return data, "\n".join(lines) or "no materials", EXIT_OK


def cmd_material_validate(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    diagnostics = [d.to_dict() for d in scene.materials.validate()]
    errors = sum(d["severity"] == "error" for d in diagnostics)
    warnings = sum(d["severity"] == "warning" for d in diagnostics)
    status = "error" if errors else ("warning" if warnings else "ok")
    data = {"path": str(path), **info, "status": status, "diagnostics": diagnostics}
    lines = [
        f"materials {status}: {errors} errors, {warnings} warnings",
        *_diagnostic_lines(diagnostics),
    ]
    return data, "\n".join(lines), EXIT_GEOMETRY_ERRORS if errors else EXIT_OK


def cmd_material_remove(args: argparse.Namespace) -> CommandOutput:
    op = {"op": "remove_material", "material": args.material, "force": args.force}
    _, data, _ = _run(args, [op], "material remove")
    result = data["operations"][0]
    return data, f"Removed material {result['removed']['name']!r}\n{_saved(data)}", EXIT_OK


def cmd_material_convert_colors(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "convert_colors", "dedupe": not args.per_object}
    if args.object:
        op["objects"] = list(args.object)
    _, data, _ = _run(args, [op], "material convert-colors")
    created = data["operations"][0]["created"]
    names = ", ".join(m["name"] for m in created) or "none"
    return data, f"Created {len(created)} simple material(s): {names}\n{_saved(data)}", EXIT_OK


def cmd_material_preview(args: argparse.Namespace) -> CommandOutput:
    """Render a material on a sphere (and ground) without changing the project."""
    path = Path(args.path)
    output = Path(args.output)
    if output.suffix.lower() != ".png":
        raise CommandError("ValidationError", f"output must be a .png file, got {output}")
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    try:
        material = scene.materials.material(args.material)
    except MaterialError as exc:
        raise _material_error(exc) from None
    for obj in scene:
        obj._visible = False
    size = 2.0
    shape = {
        "sphere": lambda: scene.add_sphere("__preview__", radius=size / 2, segments=64, rings=32),
        "cube": lambda: scene.add_cube("__preview__", size=size * 0.8),
        "plane": lambda: scene.add_plane("__preview__", size=size * 1.2),
    }[args.shape]()
    shape.material = material
    scene.add_directional_light("__key__", direction=(0.5, 0.8, -0.6), intensity=2.5)
    scene.add_directional_light("__fill__", direction=(-0.7, 0.2, -0.3), intensity=0.6)
    scene.world.configure(background="#2b2f36", ambient_color="#ffffff", ambient_intensity=0.3)
    direction = (0.0, 1.0, -0.9) if args.shape == "plane" else (0.35, 1.0, -0.45)
    camera = scene.add_camera("__preview_camera__", fov=30, active=True)
    camera.frame([shape], direction=direction, margin=0.08, aspect=1.0)
    try:
        result = scene.render(
            camera=camera,
            width=args.size,
            height=args.size,
            samples=args.samples,
            backend=args.backend,
        )
        written = result.save(output)
    except RenderError as exc:
        raise _render_error(exc) from None
    data = {
        "path": str(path),
        **info,
        "material": material.name,
        "output": str(written),
        "render": result.to_dict(),
    }
    lines = [
        f"Rendered {material.name!r} -> {written} "
        f"({args.size}x{args.size}, {result.backend['name']})"
    ]
    lines += _diagnostic_lines(
        [d.to_dict() for d in result.diagnostics if d.code != "CAMERA_FRAMED"]
    )
    return data, "\n".join(lines), EXIT_OK


# --------------------------------------------------------------------------- #
# texture
# --------------------------------------------------------------------------- #


def cmd_texture_list(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    textures = [t.info() for t in scene.materials.textures]
    lines = [f'Textures in "{scene.name}" ({len(textures)}):']
    for t in textures:
        if t["type"] == "image":
            status = t["status"]
            detail = (
                f"{status['width']}x{status['height']} {status['format'].upper()}"
                if status["ok"]
                else status["code"]
            )
            where = t["path"] or "embedded"
            lines.append(
                f"  {t['name']} (image, {where}, {detail}, "
                f"wrap {'/'.join(t['wrap'])}, {t['filter']})"
            )
        else:
            lines.append(f"  {t['name']} (procedural: {', '.join(t['node_types'])})")
    return {"path": str(path), **info, "textures": textures}, "\n".join(lines), EXIT_OK


def cmd_texture_add(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {
        "op": "add_texture",
        "name": args.name,
        "image": args.file,
        **_sampler_options(args),
    }
    if args.embed:
        op["embed"] = True
    _, data, _ = _run(args, [op], "texture add")
    texture = data["operations"][0]["texture"]
    status = texture["status"]
    return (
        data,
        f"Added image texture {texture['name']!r} ({status['width']}x{status['height']})\n"
        + _saved(data),
        EXIT_OK,
    )


def cmd_texture_procedural(args: argparse.Namespace) -> CommandOutput:
    op = {"op": "add_texture", "name": args.name, "procedural": _graph(args.graph)}
    _, data, _ = _run(args, [op], "texture procedural")
    texture = data["operations"][0]["texture"]
    return (
        data,
        f"Added procedural texture {texture['name']!r} ({', '.join(texture['node_types'])})\n"
        + _saved(data),
        EXIT_OK,
    )


def cmd_texture_set(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {
        "op": "set_texture_settings",
        "texture": args.texture,
        **_sampler_options(args),
    }
    if args.rename:
        op["name"] = args.rename
    if args.embed:
        op["embed"] = True
    if args.graph is not None:
        op["graph"] = _graph(args.graph)
    if len(op) == 2:
        raise CommandError("UsageError", "nothing to change")
    _, data, _ = _run(args, [op], "texture set")
    return (
        data,
        f"Updated texture {data['operations'][0]['texture']['name']!r}\n{_saved(data)}",
        EXIT_OK,
    )


def cmd_texture_inspect(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    try:
        texture = scene.materials.texture(args.texture)
    except MaterialError as exc:
        raise _material_error(exc) from None
    report = texture.info()
    users = [
        {"material": m.name, "slot": slot}
        for m in scene.materials.materials
        for slot, b in m.textures.items()
        if b.texture == texture.id
    ]
    report["used_by"] = users
    lines = [f"{report['name']} ({report['type']}) [{report['id']}]"]
    if report["type"] == "image":
        status = report["status"]
        lines.append(f"  source: {report['path'] or 'embedded'}")
        if status["ok"]:
            lines.append(
                f"  {status['width']}x{status['height']} {status['format'].upper()}, "
                f"{status['bit_depth']}-bit, "
                f"{status['channels']} channels, {status['texture_bytes']:,} bytes with mipmaps"
            )
        else:
            lines.append(f"  {status['code']}: {status['message']}")
        lines.append(
            f"  wrap {'/'.join(report['wrap'])}, filter {report['filter']}, "
            f"mipmaps {report['mipmaps']}"
        )
    else:
        lines.append(
            f"  {report['nodes']} nodes: {', '.join(report['node_types'])}; "
            f"output {report['output']}"
        )
    lines.append(
        "  used by: " + (", ".join(f"{u['material']}.{u['slot']}" for u in users) or "nothing")
    )
    return {"path": str(path), **info, "texture": report}, "\n".join(lines), EXIT_OK


def cmd_texture_preview(args: argparse.Namespace) -> CommandOutput:
    from .rendering.png import write_png

    path = Path(args.path)
    output = Path(args.output)
    if output.suffix.lower() != ".png":
        raise CommandError("ValidationError", f"output must be a .png file, got {output}")
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    try:
        texture = scene.materials.texture(args.texture)
        if texture.kind == "procedural":
            image = texture.preview(args.size)
        else:
            image = texture.load().image.pixels
        written = write_png(output, image)
    except MaterialError as exc:
        raise _material_error(exc) from None
    except OSError as exc:
        raise CommandError(
            type(exc).__name__, f"cannot write {output}: {exc.strerror or exc}"
        ) from None
    data = {
        "path": str(path),
        **info,
        "texture": texture.name,
        "output": str(written),
        "size": list(image.shape[1::-1]),
    }
    return data, f"Wrote {texture.name!r} -> {written} ({image.shape[1]}x{image.shape[0]})", EXIT_OK


def cmd_texture_remove(args: argparse.Namespace) -> CommandOutput:
    op = {"op": "remove_texture", "texture": args.texture, "force": args.force}
    _, data, _ = _run(args, [op], "texture remove")
    return (
        data,
        f"Removed texture {data['operations'][0]['removed']['name']!r}\n{_saved(data)}",
        EXIT_OK,
    )


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def add_material_parsers(sub, common, lossy, writing) -> None:
    from .materials.material import CHANNELS, COORDINATES, MODELS, PROJECTIONS, SLOTS
    from .materials.presets import preset_names
    from .materials.sampling import FILTERS, WRAP_MODES

    def project(p, *, read_only: bool = False) -> None:
        p.add_argument(
            "path", help="the .aether project" + (" (or .glb/.gltf)" if read_only else "")
        )

    def sampler(p) -> None:
        p.add_argument(
            "--wrap",
            nargs="+",
            choices=WRAP_MODES,
            metavar="MODE",
            help="wrap mode, or u and v modes",
        )
        p.add_argument("--filter", choices=FILTERS)
        p.add_argument("--mipmaps", dest="mipmaps", action="store_const", const=True, default=None)
        p.add_argument("--no-mipmaps", dest="mipmaps", action="store_const", const=False)
        p.add_argument("--color-space", choices=["auto", "srgb", "linear"])

    p_mat = sub.add_parser("material", help="create, change, assign, inspect and preview materials")
    mat = p_mat.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")

    p = mat.add_parser(
        "list", parents=[common, lossy], help="list materials, textures and their users"
    )
    project(p, read_only=True)
    p.set_defaults(handler=cmd_material_list)

    p = mat.add_parser("presets", parents=[common], help="the material presets and their options")
    p.set_defaults(handler=cmd_material_presets)

    p = mat.add_parser(
        "schema", parents=[common], help="parameters, slots and procedural node types"
    )
    p.set_defaults(handler=cmd_material_schema)

    p = mat.add_parser("create", parents=[common, writing], help="create a material")
    project(p)
    p.add_argument("name", help="material name (unique among the scene's materials)")
    p.add_argument("--preset", choices=preset_names(), help="build it from a procedural preset")
    p.add_argument("--model", choices=MODELS, help="shading model (default: standard)")
    p.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="a parameter or preset option (repeatable; values are JSON, e.g. roughness=0.4 "
        'base_color="#c0392b" scale=2)',
    )
    p.set_defaults(handler=cmd_material_create)

    p = mat.add_parser("set", parents=[common, writing], help="change a material")
    project(p)
    p.add_argument("material", help="material id or name")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="a parameter (repeatable)")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--rename", help="new name")
    p.set_defaults(handler=cmd_material_set)

    p = mat.add_parser("assign", parents=[common, writing], help="assign a material to objects")
    project(p)
    p.add_argument("material", help="material id or name, or 'none' to remove it")
    p.add_argument("objects", nargs="+", help="object ids or names")
    p.add_argument("--group", type=int, help="only this face group of each object's mesh")
    p.set_defaults(handler=cmd_material_assign)

    p = mat.add_parser(
        "texture", parents=[common, writing], help="bind a texture to a material slot"
    )
    project(p)
    p.add_argument("material", help="material id or name")
    p.add_argument("slot", choices=list(SLOTS))
    source = p.add_argument_group("texture (exactly one)")
    source.add_argument("--image", help="a PNG or JPEG file (becomes a texture of the scene)")
    source.add_argument("--procedural", help="a node graph as JSON, @file.json or - (stdin)")
    source.add_argument("--texture", help="an existing texture (id or name)")
    source.add_argument("--clear", action="store_true", help="remove the slot's texture")
    p.add_argument("--texture-name", help="name of the new texture")
    p.add_argument("--embed", action="store_true", help="store the image in the project")
    p.add_argument("--channel", choices=CHANNELS, help="channel a number slot reads")
    p.add_argument("--coordinates", choices=COORDINATES)
    p.add_argument("--projection", choices=PROJECTIONS, help="for object/world coordinates")
    p.add_argument(
        "--scale", nargs="+", type=float, metavar="S", help="coordinate scale (1-3 numbers)"
    )
    p.add_argument(
        "--offset", nargs="+", type=float, metavar="O", help="coordinate offset (1-3 numbers)"
    )
    p.add_argument("--rotation", type=float, help="degrees (uv plane / around z)")
    p.add_argument("--blend", type=float, help="triplanar sharpness (1-64)")
    sampler(p)
    p.set_defaults(handler=cmd_material_texture)

    p = mat.add_parser(
        "inspect", parents=[common, lossy], help="parameters, textures, users and problems"
    )
    project(p, read_only=True)
    p.add_argument("material", nargs="?", help="one material (default: all)")
    p.set_defaults(handler=cmd_material_inspect)

    p = mat.add_parser(
        "validate",
        parents=[common, lossy],
        help="check textures, UVs and references (exit 3 on errors)",
    )
    project(p, read_only=True)
    p.set_defaults(handler=cmd_material_validate)

    p = mat.add_parser("remove", parents=[common, writing], help="remove a material")
    project(p)
    p.add_argument("material", help="material id or name")
    p.add_argument("--force", action="store_true", help="unassign it from objects first")
    p.set_defaults(handler=cmd_material_remove)

    p = mat.add_parser(
        "convert-colors", parents=[common, writing], help="turn object colors into simple materials"
    )
    project(p)
    p.add_argument("--object", action="append", help="only these objects (repeatable)")
    p.add_argument("--per-object", action="store_true", help="one material per object (no sharing)")
    p.set_defaults(handler=cmd_material_convert_colors)

    p = mat.add_parser(
        "preview", parents=[common, lossy], help="render a material on a sample shape"
    )
    project(p, read_only=True)
    p.add_argument("material", help="material id or name")
    p.add_argument("-o", "--output", required=True, help="PNG to write")
    p.add_argument("--size", type=int, default=256, help="image size in pixels (default 256)")
    p.add_argument("--samples", type=int, default=2, help="antialiasing samples per axis")
    p.add_argument("--shape", choices=["sphere", "cube", "plane"], default="sphere")
    p.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    p.set_defaults(handler=cmd_material_preview)

    p_tex = sub.add_parser("texture", help="add, inspect, preview and remove textures")
    tex = p_tex.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")

    p = tex.add_parser("list", parents=[common, lossy], help="list textures")
    project(p, read_only=True)
    p.set_defaults(handler=cmd_texture_list)

    p = tex.add_parser("add", parents=[common, writing], help="add a PNG/JPEG image texture")
    project(p)
    p.add_argument("name", help="texture name")
    p.add_argument("file", help="PNG or JPEG file")
    p.add_argument("--embed", action="store_true", help="store the image in the project")
    sampler(p)
    p.set_defaults(handler=cmd_texture_add)

    p = tex.add_parser("procedural", parents=[common, writing], help="add a procedural texture")
    project(p)
    p.add_argument("name", help="texture name")
    p.add_argument("graph", help="node graph as JSON, @file.json or - (stdin)")
    p.set_defaults(handler=cmd_texture_procedural)

    p = tex.add_parser("set", parents=[common, writing], help="change a texture")
    project(p)
    p.add_argument("texture", help="texture id or name")
    p.add_argument("--rename", help="new name")
    p.add_argument("--embed", action="store_true", help="store an image file in the project")
    p.add_argument("--graph", help="procedural: a new graph (JSON, @file.json or -)")
    sampler(p)
    p.set_defaults(handler=cmd_texture_set)

    p = tex.add_parser("inspect", parents=[common, lossy], help="a texture's details and users")
    project(p, read_only=True)
    p.add_argument("texture", help="texture id or name")
    p.set_defaults(handler=cmd_texture_inspect)

    p = tex.add_parser("preview", parents=[common, lossy], help="write a texture as a PNG")
    project(p, read_only=True)
    p.add_argument("texture", help="texture id or name")
    p.add_argument("-o", "--output", required=True, help="PNG to write")
    p.add_argument("--size", type=int, default=256, help="procedural: image size (default 256)")
    p.set_defaults(handler=cmd_texture_preview)

    p = tex.add_parser("remove", parents=[common, writing], help="remove a texture")
    project(p)
    p.add_argument("texture", help="texture id or name")
    p.add_argument("--force", action="store_true", help="remove its material bindings too")
    p.set_defaults(handler=cmd_texture_remove)


__all__ = ["add_material_parsers"]
