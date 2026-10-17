"""``aether3d texture``: UVs, painting and baking from the command line (Stage 14).

* ``texture run PROJECT SCRIPT`` -- apply an ``aether3d.texture-script`` (unwrap, project
  and edit UVs, painted textures and layers, strokes, fills, bakes, transfers) as one undo
  step, all or nothing, and save.
* ``texture report PROJECT`` -- UV diagnostics of each object's mesh (read-only):
  islands, overlaps, flipped triangles, stretch, texel density, padding; ``--require-clean``
  makes the exit status 3 when a mesh has flipped or overlapping UVs.
* ``texture bake PROJECT --target OBJ [--source OBJ ...] --map normal --map ao`` -- bake
  maps and write them as PNG files (``--out DIR``), and/or bind them in the project
  (``--apply``).
* ``texture schema`` -- the JSON Schema of texture scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli import (
    EXIT_GEOMETRY_ERRORS,
    EXIT_OK,
    CommandError,
    CommandOutput,
    _check_destination,
    _command_error,
    _load_scene,
    _project_only,
    _read_script,
    _script_label,
    _write_or_dry_run,
)
from .errors import AetherError, ObjectNotFoundError


def _texturing_error(exc: AetherError) -> CommandError:
    data = exc.to_dict() if hasattr(exc, "to_dict") else {}
    details = {k: v for k, v in data.items() if k not in ("message", "location")}
    return CommandError(
        "TexturingError",
        getattr(exc, "message", str(exc)),
        getattr(exc, "location", None),
        details,
    )


def _objects(scene, refs):
    out = []
    for ref in refs:
        try:
            out.append(scene.get_by_name(ref))
        except ObjectNotFoundError:
            try:
                out.append(scene.get(ref))
            except ObjectNotFoundError:
                raise CommandError("ObjectNotFoundError", f"no object {ref!r}", "object") from None
    return out


def cmd_texture_run(args: argparse.Namespace) -> CommandOutput:
    from .texturing.script import run_texture_script

    path = Path(args.path)
    _project_only(path, "texture run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:
        result = run_texture_script(scene, script, label=_script_label(script, "texture run", args))
    except AetherError as exc:
        raise _texturing_error(exc) from None
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"{len(result['operations'])} operations in {result['seconds']:.2f} s"]
    for i, r in enumerate(result["operations"]):
        extra = ""
        if r["op"] == "stroke":
            extra = f": {r['dabs']} dabs, {r['texels']} texels"
        elif r["op"] == "unwrap":
            extra = f": {r['charts']} charts, max stretch {r['stretch_max']:.3f}"
        elif r["op"] == "bake":
            extra = f": {', '.join(r['report']['maps'])}"
        lines.append(f"  [{i}] {r['op']}{extra}")
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_texture_report(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path)
    objs = (
        _objects(scene, args.object)
        if args.object
        else [o for o in scene if o.mesh is not None and o.mesh.uvs is not None]
    )
    reports = []
    dirty = []
    lines = []
    for obj in objs:
        try:
            report = obj.uv_report(resolution=args.resolution)
        except AetherError as exc:
            raise _texturing_error(exc) from None
        reports.append({"object": {"id": obj.id, "name": obj.name}, **report})
        flipped = report.get("flipped_triangles", 0)
        overlapping = report.get("overlapping_triangles", 0)
        if flipped or overlapping:
            dirty.append(obj.name)
        lines.append(
            f"{obj.name}: {report.get('islands', '?')} islands, {flipped} flipped, "
            f"{overlapping} overlapping triangles, "
            f"{report.get('stacked_pair_count', 0)} stacked pairs, "
            f"utilization {report.get('utilization', 0):.1%}"
        )
    if not objs:
        lines.append("no object has UVs")
    data = {"path": str(path), "objects": reports, "unclean": dirty}
    status = EXIT_GEOMETRY_ERRORS if args.require_clean and dirty else EXIT_OK
    return data, "\n".join(lines), status


def cmd_texture_bake(args: argparse.Namespace) -> CommandOutput:
    from .texturing import MAPS
    from .texturing.bake import apply_bake, bake

    path = Path(args.path)
    if args.apply:
        _project_only(path, "texture bake --apply")
    if not args.apply and not args.out:
        raise CommandError("ValidationError", "give --out DIR, --apply, or both")
    scene, _ = _load_scene(path, history=args.apply)
    target = _objects(scene, [args.target])[0]
    sources = _objects(scene, args.source) if args.source else None
    maps = args.map or ["normal"]
    files = {}
    if args.out:
        out_dir = Path(args.out)
        if out_dir.exists() and not out_dir.is_dir():
            raise CommandError("NotADirectoryError", f"{out_dir} is not a directory")
        for name in maps:
            if name in MAPS:
                _check_destination(out_dir / f"{target.name}_{name}.png", args.force)
    try:
        result = bake(
            scene,
            target,
            sources,
            maps,
            resolution=args.resolution,
            samples=args.samples,
            padding=args.padding,
            cage=args.cage,
            depth=args.depth,
            ao_rays=args.ao_rays,
            backend=args.backend,
        )
    except AetherError as exc:
        raise _texturing_error(exc) from None
    if args.out:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in result.maps:
            target_file = out_dir / f"{target.name}_{name}.png"
            try:
                files[name] = str(result.save_png(name, target_file, bits=args.bits))
            except OSError as exc:
                raise _command_error(exc, target_file) from None
    written = None
    if args.apply:
        try:
            apply_bake(scene, target, result)
        except AetherError as exc:
            raise _texturing_error(exc) from None
        written = _write_or_dry_run(scene, path, args)
    report = result.report
    data = {
        "path": str(path),
        "files": files,
        "output": written,
        "report": report,
        "diagnostics": [d.to_dict() for d in result.diagnostics],
    }
    lines = [
        f"baked {', '.join(report['maps'])} of {target.name} at "
        f"{report['resolution'][0]}x{report['resolution'][1]} ({report['backend']}, "
        f"{report['seconds']['total']:.2f} s), misses {report['miss_fraction']:.2%}"
    ]
    lines += [f"  {name}: {file}" for name, file in files.items()]
    lines += [f"  {d.code}: {d.message}" for d in result.diagnostics]
    if args.apply:
        lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_texture_schema(args: argparse.Namespace) -> CommandOutput:
    import json

    from .texturing.schema import texture_script_schema

    schema = texture_script_schema()
    return schema, json.dumps(schema, indent=2), EXIT_OK


def add_texture_parsers(tx, common, writing) -> None:
    """Add run, report, bake and schema to the ``texture`` command's subcommands."""
    from .texturing import MAPS

    p = tx.add_parser(
        "run",
        parents=[common, writing],
        help="apply a JSON texture script to a project (atomically)",
        description="Apply an aether3d.texture-script document (docs/TEXTURING.md): UV "
        "unwrapping and editing, painted textures and layers, strokes, fills, bakes and "
        "transfers, as one step of the project's history (all or nothing), then save.",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("script", help="texture script (.json), or - to read it from stdin")
    p.set_defaults(handler=cmd_texture_run)
    p = tx.add_parser(
        "report",
        parents=[common],
        help="UV diagnostics of meshes (read-only)",
        description="Islands, overlapping (unintended) and stacked (intended) triangles, "
        "flipped triangles, stretch, texel density, utilization and gaps of each mesh's UVs.",
    )
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument("--object", action="append", help="object id or name (repeatable)")
    p.add_argument("--resolution", type=int, default=1024, help="texture size for texel metrics")
    p.add_argument(
        "--require-clean",
        action="store_true",
        help="exit status 3 if a mesh has flipped or overlapping UV triangles",
    )
    p.set_defaults(handler=cmd_texture_report)
    p = tx.add_parser(
        "bake",
        parents=[common, writing],
        help="bake maps of an object from high-resolution sources",
        description="Bake maps (normal, ao, curvature, thickness, height, ids, material "
        "channels, ...) of --target from --source objects (default: itself) through the "
        "target's UVs; write PNG files to --out and/or bind the maps in the project "
        "(--apply, one undo step).",
    )
    p.add_argument("path", help=".aether project (or .glb/.gltf without --apply)")
    p.add_argument("--target", required=True, help="the object baked into (needs UVs)")
    p.add_argument("--source", action="append", help="a source object (repeatable)")
    p.add_argument("--map", action="append", choices=list(MAPS), help="a map (repeatable)")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--samples", type=int, default=2, help="n: n x n samples per texel")
    p.add_argument("--padding", type=int, default=8, help="texels of dilation")
    p.add_argument("--cage", type=float, help="ray start outside the target (world units)")
    p.add_argument("--depth", type=float, help="ray reach inside the target (world units)")
    p.add_argument("--ao-rays", type=int, default=64)
    p.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    p.add_argument("--out", help="directory for <target>_<map>.png files")
    p.add_argument("--bits", type=int, choices=(8, 16), default=8, help="PNG bit depth")
    p.add_argument("--apply", action="store_true", help="bind the maps in the project and save")
    p.add_argument("--force", action="store_true", help="overwrite existing PNG files")
    p.set_defaults(handler=cmd_texture_bake)
    p = tx.add_parser("schema", parents=[common], help="the JSON Schema of texture scripts")
    p.set_defaults(handler=cmd_texture_schema)
