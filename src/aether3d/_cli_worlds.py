"""``aether3d world``: procedural worlds from the command line (Stage 16).

* ``world run PROJECT SCRIPT`` -- apply an ``aether3d.world-script`` (create, configure,
  features, structures, settlements, roads, block edits; queries) as one undo step, all
  or nothing, and save.
* ``world info PROJECT [--world W]`` -- every world, or one world's summary (definition
  digest, biomes, features, structures, the regional stage's estimate, caches).
* ``world sample PROJECT --world W --at X,Y ...`` -- fields at points (height, slope,
  biome, water, ...).
* ``world export PROJECT OUT.glb --world W --region X0,Y0,X1,Y1`` -- the generated geometry
  of a region as glTF.
* ``world presets`` / ``world blocks`` / ``world schema`` / ``world guide`` -- presets,
  block names, the JSON Schema of world scripts, the agent guide.

Rendering is ``aether3d render``: worlds draw through every backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cli import (
    EXIT_OK,
    CommandError,
    CommandOutput,
    _load_scene,
    _project_only,
    _read_script,
    _script_label,
    _write_or_dry_run,
)


def _engine(exc: Exception) -> CommandError:
    details = {"code": getattr(exc, "code", None)}
    if getattr(exc, "hint", None):
        details["hint"] = exc.hint  # type: ignore[attr-defined]
    message = getattr(exc, "message", None) or str(exc)
    return CommandError(type(exc).__name__, message, getattr(exc, "location", None), details)


def _world(scene: Any, ref: str | None) -> Any:
    from .errors import WorldError

    worlds = scene.worlds
    if ref is None:
        if len(worlds) != 1:
            raise CommandError(
                "ValidationError",
                "this project has " + (f"{len(worlds)} worlds" if len(worlds) else "no world"),
                None,
                {"hint": "name one with --world" if len(worlds) else "create one with world run"},
            )
        return next(iter(worlds))
    try:
        return worlds.get(ref)
    except WorldError as exc:
        raise _engine(exc) from None


def _floats(text: str, n: int, what: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != n:
        raise CommandError(
            "ValidationError", f"{what} is {n} comma-separated numbers, got {text!r}"
        )
    return values


def cmd_world_run(args: argparse.Namespace) -> CommandOutput:
    from .errors import WorldError
    from .worlds.script import run_world_script

    path = Path(args.path)
    _project_only(path, "world run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:
        result = run_world_script(scene, script, label=_script_label(script, "world run", args))
    except WorldError as exc:
        raise _engine(exc) from None
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"{len(result['operations'])} operations applied as one step"]
    lines += [f"world {w['name']} ({w['mode']})" for w in result["worlds"]]
    lines += [f"query {q['query']}: {json.dumps(q)[:160]}" for q in result["queries"]]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_world_info(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path)
    if args.world is None and len(scene.worlds) != 1:
        data = {
            "path": str(path),
            "worlds": [
                {"name": w.name, "id": w.id, "mode": w.mode, "seed": w.seed} for w in scene.worlds
            ],
        }
        lines = [f"{w['name']}: {w['mode']} world, seed {w['seed']}" for w in data["worlds"]] or [
            "no worlds"
        ]
        return data, "\n".join(lines), EXIT_OK
    w = _world(scene, args.world)
    info = w.info()
    data = {"path": str(path), "world": w.name, **info}
    lines = [f"{w.name}: {w.mode} world, seed {w.seed}, extent {info.get('extent')}"]
    lines += [
        f"{k}: {json.dumps(info[k])[:200]}"
        for k in ("biomes", "features", "structures")
        if k in info
    ]
    return data, "\n".join(lines), EXIT_OK


def cmd_world_sample(args: argparse.Namespace) -> CommandOutput:
    from .errors import WorldError

    path = Path(args.path)
    scene, _ = _load_scene(path)
    w = _world(scene, args.world)
    points = [_floats(p, 2, "--at") for p in args.at or []]
    if not points:
        raise CommandError("ValidationError", "give at least one --at X,Y")
    try:
        data = w.sample(points, fields=args.field)
    except WorldError as exc:
        raise _engine(exc) from None
    lines = []
    for k, (x, y) in enumerate(points):
        row = {f: data[f][k] for f in data if isinstance(data[f], list) and f not in ("x", "y")}
        lines.append(
            f"({x:g}, {y:g}): " + ", ".join(f"{f}={json.dumps(v)}" for f, v in row.items())
        )
    return {"path": str(path), "world": w.name, **data}, "\n".join(lines), EXIT_OK


def cmd_world_export(args: argparse.Namespace) -> CommandOutput:
    from .errors import AetherError

    path = Path(args.path)
    scene, _ = _load_scene(path)
    w = _world(scene, args.world)
    out = Path(args.output)
    if out.exists() and not args.force:
        raise CommandError("FileExistsError", f"{out} exists", str(out), {"hint": "pass --force"})
    include = tuple(args.include.split(",")) if args.include else None
    options: dict[str, Any] = {"level": args.level, "plant_lod": args.plant_lod}
    if include is not None:
        options["include"] = include
    if args.bake_textures:
        options["bake_textures"] = args.bake_textures
    try:
        data = w.export_region(out, _floats(args.region, 4, "--region"), **options)
    except AetherError as exc:
        raise _engine(exc) from None
    text = (
        f"wrote {data['path']}: {data['objects']} objects, {data['triangles']} triangles, "
        f"bounds {data['bounds']}"
    )
    return data, text, EXIT_OK


def cmd_world_presets(args: argparse.Namespace) -> CommandOutput:
    from .worlds.definition import describe_presets

    presets = describe_presets()
    return {"presets": presets}, "\n".join(f"{k}: {v}" for k, v in presets.items()), EXIT_OK


def cmd_world_blocks(args: argparse.Namespace) -> CommandOutput:
    from .worlds.blocks import BLOCK_LIST

    blocks = [{"name": b.name, "kind": b.kind} for b in BLOCK_LIST]
    return {"blocks": blocks}, ", ".join(b["name"] for b in blocks), EXIT_OK


def cmd_world_schema(args: argparse.Namespace) -> CommandOutput:
    from .worlds.script import schema

    s = schema()
    return s, json.dumps(s, indent=2), EXIT_OK


def cmd_world_guide(args: argparse.Namespace) -> CommandOutput:
    from importlib import resources

    text = resources.files("aether3d.worlds").joinpath("guide.md").read_text(encoding="utf-8")
    return {"guide": text}, text, EXIT_OK


def add_world_parsers(sub, common, writing) -> None:
    p_w = sub.add_parser("world", help="procedural worlds: scripts, queries, exports")
    ws = p_w.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = ws.add_parser(
        "run",
        parents=[common, writing],
        help="apply a JSON world script to a project (atomically)",
        description="Apply an aether3d.world-script document (docs/WORLDS.md): create and "
        "configure worlds, features, structures, settlements, roads and block edits as one "
        "step of the project's history (all or nothing), answer its queries, then save.",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("script", help="world script (.json), or - to read it from stdin")
    p.set_defaults(handler=cmd_world_run)
    p = ws.add_parser("info", parents=[common], help="the worlds of a project, or one world")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--world", help="world name or id")
    p.set_defaults(handler=cmd_world_info)
    p = ws.add_parser("sample", parents=[common], help="fields at points (read-only)")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--world", help="world name or id (default: the only one)")
    p.add_argument("--at", action="append", metavar="X,Y", help="a point (repeatable)")
    p.add_argument("--field", action="append", help="a field (repeatable; default: all)")
    p.set_defaults(handler=cmd_world_sample)
    p = ws.add_parser("export", parents=[common], help="a region's geometry as glTF")
    p.add_argument("path", help="the .aether project")
    p.add_argument("output", help="the .glb or .gltf to write")
    p.add_argument("--world", help="world name or id (default: the only one)")
    p.add_argument("--region", required=True, metavar="X0,Y0,X1,Y1", help="metres")
    p.add_argument("--level", type=int, default=0, help="terrain level of detail (0: finest)")
    p.add_argument("--plant-lod", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--include", help="parts: terrain,water,caves,vegetation,structures")
    p.add_argument("--bake-textures", type=int, metavar="PIXELS")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(handler=cmd_world_export)
    for name, handler, text in (
        ("presets", cmd_world_presets, "the world presets"),
        ("blocks", cmd_world_blocks, "the block names of block worlds"),
        ("schema", cmd_world_schema, "the JSON Schema of world scripts"),
        ("guide", cmd_world_guide, "the agent guide to worlds (Markdown)"),
    ):
        p = ws.add_parser(name, parents=[common], help=text)
        p.set_defaults(handler=handler)
