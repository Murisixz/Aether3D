"""Command-line interface.

Commands: ``new``, ``validate``, ``inspect``, ``geometry``, ``import``,
``export``, ``topology``, ``edit`` and ``procedural`` (with the subcommands
``types``, ``inspect``, ``validate``, ``run``, ``set`` and ``bake``).

Every command accepts ``--json`` to emit a single machine-readable JSON
object on stdout, both on success (``"ok": true``) and on failure
(``"ok": false`` plus an ``"error"`` object). Exit codes:

* ``0`` -- success
* ``1`` -- the command failed (invalid file, missing file, unsupported content, ...)
* ``2`` -- invalid command-line usage
* ``3`` -- ``geometry``/``topology``/``procedural validate`` completed but found
  error-severity problems

Commands that read a scene accept native ``.aether`` projects as well as
``.glb``/``.gltf`` files (which are imported in memory).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import (
    AetherError,
    AssetFormatError,
    ProjectFormatError,
    UnsupportedFeatureError,
)
from .gltf import export_gltf, import_gltf
from .scene import Scene, SceneObject
from .serialization import FORMAT_VERSION, loads, save_scene, scene_from_dict

MAX_SCRIPT_BYTES = 16 * 1024 * 1024

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_GEOMETRY_ERRORS = 3

GLTF_SUFFIXES = (".glb", ".gltf")


class CommandError(Exception):
    """A command failed for a reason that should be reported, not traced."""

    def __init__(
        self,
        error_type: str,
        message: str,
        location: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.location = location
        self.details = details or {}


def _command_error(exc: BaseException, path: Path) -> CommandError:
    """Translate an exception raised while reading or writing ``path``."""
    if isinstance(exc, FileNotFoundError):
        return CommandError("FileNotFoundError", f"no such file: {path}")
    if isinstance(exc, AssetFormatError):
        return CommandError("AssetFormatError", exc.message, exc.location, {"code": exc.code})
    if isinstance(exc, ProjectFormatError):
        return CommandError("ProjectFormatError", exc.message, exc.location)
    if isinstance(exc, UnsupportedFeatureError):
        return CommandError(
            "UnsupportedFeatureError",
            exc.message,
            details={"diagnostics": [d.to_dict() for d in exc.diagnostics]},
        )
    from .procedural.params import ProceduralError

    if isinstance(exc, ProceduralError):
        return CommandError("ProceduralError", exc.message, exc.location, _procedural_details(exc))
    if isinstance(exc, AetherError):
        return CommandError(type(exc).__name__, str(exc))
    if isinstance(exc, OSError):
        return CommandError(type(exc).__name__, f"cannot access {path}: {exc.strerror or exc}")
    raise exc


def _procedural_details(exc: Any) -> dict[str, Any]:
    details: dict[str, Any] = {"code": exc.code}
    if exc.hint:
        details["hint"] = exc.hint
    if exc.details:
        details["details"] = exc.details
    return details


def _read_project(path: Path) -> tuple[Scene, int]:
    """Load a ``.aether`` project, returning the scene and its on-disk format version."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise CommandError("ProjectFormatError", "file is not valid UTF-8 text") from None
    except OSError as exc:
        raise _command_error(exc, path) from None
    try:
        document = loads(text)
        version = document.get("format_version") if isinstance(document, dict) else None
        return scene_from_dict(document), version
    except ProjectFormatError as exc:
        raise _command_error(exc, path) from None


def _load_scene(path: Path, allow_lossy: bool = False) -> tuple[Scene, dict[str, Any]]:
    """Load a scene from a ``.aether`` project or a ``.glb``/``.gltf`` asset.

    Returns the scene and a dict describing the source (format, version or
    import diagnostics).
    """
    if path.suffix.lower() in GLTF_SUFFIXES:
        try:
            result = import_gltf(path, allow_lossy=allow_lossy)
        except (AetherError, OSError) as exc:
            raise _command_error(exc, path) from None
        info = {
            "source_format": result.format,
            "generator": result.generator,
            "lossy": result.lossy,
            "import_diagnostics": [d.to_dict() for d in result.diagnostics],
        }
        return result.scene, info
    scene, version = _read_project(path)
    return scene, {"source_format": "aether", "format_version": version}


def _describe_source(info: dict[str, Any]) -> str:
    if info["source_format"] == "aether":
        return f"format v{info['format_version']}"
    return f"imported from {info['source_format'].upper()}"


def _check_destination(path: Path, force: bool) -> None:
    if path.is_dir():
        raise CommandError("IsADirectoryError", f"{path} is a directory")
    if path.exists() and not force:
        raise CommandError("FileExistsError", f"{path} already exists (use --force to overwrite)")


def _tree_glyphs() -> tuple[str, str, str]:
    """Box-drawing tree glyphs, or ASCII ones if stdout cannot encode them."""
    try:
        "├└│".encode(sys.stdout.encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return "|-- ", "`-- ", "|   "
    return "├── ", "└── ", "│   "


def _fmt_vec(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.6g}" for v in values) + ")"


def _mesh_count(scene: Scene) -> int:
    return len({id(obj.mesh) for obj in scene if obj.mesh is not None})


def _diagnostic_lines(diagnostics: list[dict[str, Any]]) -> list[str]:
    lines = []
    for d in diagnostics:
        where = f" at {d['location']}" if d["location"] else ""
        count = f" ({d['count']})" if d["count"] is not None else ""
        lossy = " [lossy]" if d["lossy"] else ""
        lines.append(f"  {d['severity']:<7} {d['code']}{count}{where}: {d['message']}{lossy}")
    return lines


def _object_info(obj: SceneObject, depth: int) -> dict[str, Any]:
    t = obj.transform
    mesh = None
    if obj.mesh is not None:
        lo, hi = obj.mesh.bounds()
        mesh = {
            "vertex_count": obj.mesh.vertex_count,
            "triangle_count": obj.mesh.triangle_count,
            "bounds": {"min": list(lo), "max": list(hi)},
        }
    return {
        "id": obj.id,
        "name": obj.name,
        "parent": None if obj.parent is None else obj.parent.id,
        "children": [child.id for child in obj.children],
        "depth": depth,
        "transform": {
            "position": list(t.position),
            "rotation": list(t.rotation),
            "rotation_euler": list(t.rotation_euler),
            "scale": list(t.scale),
        },
        "world_position": list(obj.world_position),
        "mesh": mesh,
        "geometry": None
        if obj.geometry is None
        else {"id": obj.geometry.id, "name": obj.geometry.name},
    }


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

CommandOutput = tuple[dict[str, Any], str, int]


def cmd_new(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _check_destination(path, args.force)
    try:
        scene = Scene(args.name if args.name is not None else (path.stem or "Untitled Scene"))
        save_scene(scene, path)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, path) from None
    result = {"path": str(path), "scene": scene.name, "format_version": FORMAT_VERSION}
    return result, f'Created empty project "{scene.name}" at {path}', EXIT_OK


def cmd_validate(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, args.allow_lossy)
    result = {
        "path": str(path),
        "valid": True,
        **info,
        "scene": scene.name,
        "object_count": len(scene),
        "mesh_count": _mesh_count(scene),
    }
    counts = f"{len(scene)} objects, {result['mesh_count']} meshes"
    if info["source_format"] == "aether":
        text = (
            f"OK: {path} is a valid Aether3D project (format v{info['format_version']}, {counts})"
        )
    else:
        diagnostics = info["import_diagnostics"]
        text = "\n".join(
            [
                f"OK: {path} can be imported by Aether3D ({info['source_format']}, {counts}, "
                f"{len(diagnostics)} diagnostics)",
                *_diagnostic_lines(diagnostics),
            ]
        )
    return result, text, EXIT_OK


def cmd_inspect(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, args.allow_lossy)

    objects: list[dict[str, Any]] = []
    lines = [f'Scene "{scene.name}" ({len(scene)} objects, {_describe_source(info)})']
    glyphs = _tree_glyphs()

    def walk(obj: SceneObject, depth: int, prefix: str, last: bool) -> None:
        objects.append(_object_info(obj, depth))
        branch = glyphs[1] if last else glyphs[0]
        details = [f"pos={_fmt_vec(obj.position)}"]
        if obj.rotation != (0.0, 0.0, 0.0, 1.0):
            details.append(f"rot_euler={_fmt_vec(obj.rotation_euler)}")
        if obj.scale != (1.0, 1.0, 1.0):
            details.append(f"scale={_fmt_vec(obj.scale)}")
        if obj.mesh is not None:
            details.append(f"mesh={obj.mesh.vertex_count}v/{obj.mesh.triangle_count}t")
        if obj.geometry is not None:
            details.append(f"procedural={obj.geometry.name}")
        lines.append(f"{prefix}{branch}{obj.name}  [{obj.id}]  " + "  ".join(details))
        kids = obj.children
        for i, child in enumerate(kids):
            walk(child, depth + 1, prefix + ("    " if last else glyphs[2]), i == len(kids) - 1)

    roots = scene.roots
    for i, root in enumerate(roots):
        walk(root, 0, "", i == len(roots) - 1)
    if not roots:
        lines.append("(empty)")

    result = {
        "path": str(path),
        **info,
        "scene": {
            "name": scene.name,
            "object_count": len(scene),
            "mesh_count": _mesh_count(scene),
            "roots": [root.id for root in roots],
        },
        "objects": objects,
    }
    return result, "\n".join(lines), EXIT_OK


def cmd_geometry(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, info = _load_scene(path, args.allow_lossy)
    report = scene.inspect_geometry().to_dict()
    summary = report["summary"]
    lines = [
        f'Geometry of "{scene.name}": {report["status"].upper()}',
        f"  objects: {summary['object_count']} ({summary['mesh_object_count']} with meshes), "
        f"unique meshes: {summary['unique_mesh_count']}",
        f"  triangles: {summary['unique_triangle_count']} unique, "
        f"{summary['instanced_triangle_count']} instanced",
    ]
    if report["world_bounds"]:
        bounds = report["world_bounds"]
        lines.append(f"  world bounds: {_fmt_vec(bounds['min'])} .. {_fmt_vec(bounds['max'])}")
    for mesh in report["meshes"]:
        topo = mesh["topology"]
        closed = "closed" if topo["closed"] else "open"
        if not topo["edge_manifold"]:
            closed += ", non-manifold edges"
        if topo["volume"] is not None:
            closed += f", volume {topo['volume']:.6g}"
        lines.append(
            f"  mesh {mesh['index']}: {mesh['vertex_count']}v/{mesh['triangle_count']}t, "
            f"area {mesh['surface_area']:.6g}, {closed} [{mesh['status']}] "
            f"used by {len(mesh['used_by'])} object(s)"
        )
    all_diagnostics = report["diagnostics"] + [
        d for m in report["meshes"] for d in m["diagnostics"]
    ]
    if all_diagnostics:
        lines.append("Diagnostics:")
        lines += _diagnostic_lines(all_diagnostics)
    code = EXIT_GEOMETRY_ERRORS if report["status"] == "error" else EXIT_OK
    return {"path": str(path), **info, "geometry": report}, "\n".join(lines), code


def cmd_import(args: argparse.Namespace) -> CommandOutput:
    source, destination = Path(args.source), Path(args.destination)
    if source.suffix.lower() not in GLTF_SUFFIXES:
        raise CommandError(
            "ValidationError", f"cannot import {source}: expected a .glb or .gltf file"
        )
    _check_destination(destination, args.force)
    try:
        result = import_gltf(source, allow_lossy=args.allow_lossy)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, source) from None
    try:
        save_scene(result.scene, destination)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, destination) from None
    data = {"destination": str(destination), **result.to_dict()}
    text = "\n".join(
        [
            f"Imported {source} -> {destination} ({len(result.scene)} objects, "
            f"{len(result.diagnostics)} diagnostics)",
            *_diagnostic_lines([d.to_dict() for d in result.diagnostics]),
        ]
    )
    return data, text, EXIT_OK


def cmd_export(args: argparse.Namespace) -> CommandOutput:
    source, destination = Path(args.source), Path(args.destination)
    if destination.suffix.lower() not in GLTF_SUFFIXES:
        raise CommandError(
            "ValidationError", f"cannot export to {destination}: expected a .glb or .gltf path"
        )
    _check_destination(destination, args.force)
    scene, info = _load_scene(source, args.allow_lossy)
    try:
        result = export_gltf(scene, destination, allow_lossy=args.allow_lossy)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, destination) from None
    data = {"source": str(source), **info, **result.to_dict()}
    text = "\n".join(
        [
            f"Exported {source} -> {destination} ({result.format}, {result.node_count} nodes, "
            f"{result.mesh_count} meshes, {result.byte_length} bytes)",
            *_diagnostic_lines([d.to_dict() for d in result.diagnostics]),
        ]
    )
    return data, text, EXIT_OK


def _find_object(scene: Scene, ref: str) -> SceneObject:
    for getter in (scene.get, scene.get_by_name):
        try:
            return getter(ref)
        except AetherError:
            continue
    raise CommandError("ObjectNotFoundError", f"no object with id or name {ref!r}")


def cmd_topology(args: argparse.Namespace) -> CommandOutput:
    from .editing import EditableMesh

    path = Path(args.path)
    scene, info = _load_scene(path, args.allow_lossy)
    obj = _find_object(scene, args.object)
    if obj.mesh is None:
        raise CommandError("ValidationError", f"object {obj.name!r} has no mesh")
    mesh = EditableMesh.from_mesh(obj.mesh, quads=not args.triangles)
    report = mesh.validate()
    data: dict[str, Any] = {
        "path": str(path),
        **info,
        "object": {"id": obj.id, "name": obj.name},
        "topology": report.to_dict(),
    }
    if args.elements:
        data["elements"] = mesh.describe(limit=args.limit)
    summary = report.summary
    lines = [
        f'Topology of "{obj.name}": {report.status.upper()}',
        f"  vertices {summary['vertex_count']}, edges {summary['edge_count']}, faces "
        f"{summary['face_count']} {summary['face_sizes']}",
        f"  closed={summary['closed']} edge_manifold={summary['edge_manifold']} "
        f"vertex_manifold={summary['vertex_manifold']} "
        f"consistently_oriented={summary['consistently_oriented']}",
    ]
    if report.diagnostics:
        lines.append("Diagnostics:")
        lines += _diagnostic_lines([d.to_dict() for d in report.diagnostics])
    code = EXIT_GEOMETRY_ERRORS if report.status == "error" else EXIT_OK
    return data, "\n".join(lines), code


def _read_script(arg: str) -> Any:
    """Read a JSON script from a file or ``-`` (stdin), with a size limit."""
    try:
        if arg == "-":
            raw = sys.stdin.buffer.read(MAX_SCRIPT_BYTES + 1)
        else:
            script_path = Path(arg)
            if script_path.stat().st_size > MAX_SCRIPT_BYTES:
                raise CommandError(
                    "ResourceLimitError", f"script is larger than {MAX_SCRIPT_BYTES} bytes"
                )
            raw = script_path.read_bytes()
    except OSError as exc:
        raise _command_error(exc, Path(arg)) from None
    if len(raw) > MAX_SCRIPT_BYTES:
        raise CommandError("ResourceLimitError", f"script is larger than {MAX_SCRIPT_BYTES} bytes")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommandError(
            "ScriptError", f"script is not valid JSON: {exc}", "script", {"code": "INVALID_JSON"}
        ) from None


def _write_or_dry_run(scene: Scene, path: Path, args: argparse.Namespace) -> str | None:
    if args.dry_run:
        return None
    output = Path(args.output) if args.output else path
    try:
        return str(save_scene(scene, output))
    except (AetherError, OSError) as exc:
        raise _command_error(exc, output) from None


def _project_only(path: Path, command: str) -> None:
    if path.suffix.lower() in GLTF_SUFFIXES:
        raise CommandError(
            "ValidationError", f"{command} works on .aether projects; import the glTF file first"
        )


def cmd_edit(args: argparse.Namespace) -> CommandOutput:
    from .editing.script import ScriptError, run_edit_script

    path = Path(args.path)
    if path.suffix.lower() in GLTF_SUFFIXES:
        raise CommandError(
            "ValidationError", "edit works on .aether projects; import the glTF file first"
        )
    scene, info = _load_scene(path)
    script = _read_script(args.script)
    try:
        result = run_edit_script(scene, script)
    except ScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("ScriptError", exc.message, exc.location, details) from None
    output = Path(args.output) if args.output else path
    written = None
    if not args.dry_run:
        try:
            written = str(save_scene(scene, output))
        except (AetherError, OSError) as exc:
            raise _command_error(exc, output) from None
    data = {
        "path": str(path),
        "output": None if args.dry_run else str(output),
        "dry_run": args.dry_run,
        **result.to_dict(),
    }
    lines = []
    for edit in result.edits:
        commit = edit["commit"]
        mesh = commit["mesh"]
        size = f"{mesh['vertex_count']}v/{mesh['triangle_count']}t" if mesh else "no mesh"
        lines.append(
            f"{edit['object']['name']}: {len(edit['operations'])} operations, "
            f"{'changed' if commit['changed'] else 'unchanged'} ({size}), "
            f"validation {commit['validation']['status']}"
        )
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


# --------------------------------------------------------------------------- #
# procedural
# --------------------------------------------------------------------------- #


def cmd_procedural_types(args: argparse.Namespace) -> CommandOutput:
    from .procedural import ProceduralSystem

    types = ProceduralSystem.types()
    lines = ["Primitives:"]
    for t in types["primitives"]:
        params = ", ".join(p["name"] for p in t["params"])
        lines.append(f"  {t['type']}({params})")
    lines.append("Modifiers:")
    for t in types["modifiers"]:
        params = ", ".join(p["name"] for p in t["params"])
        lines.append(f"  {t['type']}({params})")
    lines.append("Parameter types: " + ", ".join(types["parameter_types"]))
    return types, "\n".join(lines), EXIT_OK


def cmd_procedural_inspect(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural inspect")
    scene, _ = _load_scene(path)
    system = scene.procedural
    if args.geometry:
        info = {"geometry": [system.info(args.geometry, evaluate=args.evaluate)]}
        info["parameters"] = system.info()["parameters"]
    else:
        info = system.info(evaluate=args.evaluate)
    info.pop("cache", None)
    info["objects"] = [
        {"id": o.id, "name": o.name, "geometry": o.geometry.name} for o in scene if o.is_procedural
    ]
    lines = [f'Procedural content of "{scene.name}"']
    lines.append(f"Parameters ({len(info['parameters'])}):")
    for name, p in info["parameters"].items():
        value = p["value"]
        shown = (
            f"{value['expr']} = {p['resolved']}" if isinstance(value, dict) else f"{p['resolved']}"
        )
        lines.append(f"  {name}: {p['type']} = {shown}")
    lines.append(f"Geometry ({len(info['geometry'])}):")
    for g in info["geometry"]:
        src = g["source"]
        what = src.get("primitive") or (
            f"geometry {src.get('geometry')}" if src["type"] == "geometry" else "stored mesh"
        )
        lines.append(f"  {g['name']} <- {what}; users: {len(g['users'])}")
        for m in g["modifiers"]:
            state = "" if m["enabled"] else " (disabled)"
            lines.append(f"    - {m['name']} [{m['type']}]{state}")
        if "evaluated" in g:
            e = g["evaluated"]
            lines.append(
                f"    = {e['mesh']['vertex_count']}v/{e['mesh']['triangle_count']}t, "
                f"{e['polygons']['face_count']} faces"
            )
    return info, "\n".join(lines), EXIT_OK


def cmd_procedural_validate(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural validate")
    scene, _ = _load_scene(path)
    report = scene.procedural.validate(args.geometry)
    lines = [f"Procedural validation: {report['status']}"]
    for entry in report["geometry"]:
        name = entry["geometry"]["name"]
        if entry["status"] == "error" and "error" in entry:
            err = entry["error"]
            lines.append(f"  {name}: error [{err['code']}] {err['message']}")
            continue
        summary = entry["topology"]["summary"]
        lines.append(
            f"  {name}: {entry['status']} ({summary['face_count']} faces, "
            f"{'closed' if summary['closed'] else 'open'})"
        )
        lines += ["  " + line for line in _diagnostic_lines(entry["diagnostics"])]
    code = EXIT_GEOMETRY_ERRORS if report["status"] == "error" else EXIT_OK
    return {"path": str(path), **report}, "\n".join(lines), code


def _run_procedural(scene: Scene, script: Any) -> dict[str, Any]:
    from .editing.script import ScriptError
    from .procedural.script import run_procedural_script

    try:
        return run_procedural_script(scene, script)
    except ScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("ScriptError", exc.message, exc.location, details) from None


def cmd_procedural_run(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural run")
    scene, _ = _load_scene(path)
    script = _read_script(args.script)
    result = _run_procedural(scene, script)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"{len(result['operations'])} operations applied"]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def _parse_assignment(text: str) -> tuple[str, Any]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise CommandError("UsageError", f"expected NAME=VALUE, got {text!r}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value  # bare strings (e.g. enum values)
    return name, parsed


def cmd_procedural_set(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural set")
    scene, _ = _load_scene(path)
    operations = []
    for assignment in args.assignments:
        name, value = _parse_assignment(assignment)
        op: dict[str, Any] = {"op": "set_parameter", "name": name, "value": value}
        if args.geometry:
            op["geometry"] = args.geometry
        operations.append(op)
    script = {"format": "aether3d.procedural-script", "version": 1, "operations": operations}
    result = _run_procedural(scene, script)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"set {', '.join(a.partition('=')[0] for a in args.assignments)}"]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_procedural_bake(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural bake")
    scene, _ = _load_scene(path)
    if args.all:
        targets = [o.id for o in scene if o.is_procedural]
    elif args.object:
        targets = [_find_object(scene, ref).id for ref in args.object]
    else:
        raise CommandError("UsageError", "give --object (repeatable) or --all")
    operations = [{"op": "bake", "object": oid} for oid in targets]
    if operations:
        script = {"format": "aether3d.procedural-script", "version": 1, "operations": operations}
        _run_procedural(scene, script)
    written = _write_or_dry_run(scene, path, args)
    baked = [{"id": oid, "name": scene.get(oid).name} for oid in targets]
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, "baked": baked}
    lines = [f"baked {len(baked)} object(s): " + ", ".join(b["name"] for b in baked)]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    lossy = argparse.ArgumentParser(add_help=False)
    lossy.add_argument(
        "--allow-lossy",
        action="store_true",
        help="convert even if unsupported content must be dropped or approximated (reported)",
    )
    reading = [common, lossy]

    parser = argparse.ArgumentParser(
        prog="aether3d", description="Aether3D project command-line tools."
    )
    parser.add_argument("--version", action="version", version=f"aether3d {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_new = sub.add_parser("new", parents=[common], help="create an empty project file")
    p_new.add_argument("path", help="path of the .aether file to create")
    p_new.add_argument("--name", help="scene name (default: the file name without extension)")
    p_new.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_new.set_defaults(handler=cmd_new)

    p_validate = sub.add_parser(
        "validate", parents=reading, help="check that a project (or glTF file) can be loaded"
    )
    p_validate.add_argument("path", help="path of a .aether, .glb or .gltf file")
    p_validate.set_defaults(handler=cmd_validate)

    p_inspect = sub.add_parser(
        "inspect", parents=reading, help="show a scene's objects and hierarchy"
    )
    p_inspect.add_argument("path", help="path of a .aether, .glb or .gltf file")
    p_inspect.set_defaults(handler=cmd_inspect)

    p_geometry = sub.add_parser(
        "geometry", parents=reading, help="analyse geometry (bounds, degeneracies, topology, ...)"
    )
    p_geometry.add_argument("path", help="path of a .aether, .glb or .gltf file")
    p_geometry.set_defaults(handler=cmd_geometry)

    p_import = sub.add_parser(
        "import", parents=reading, help="convert a .glb/.gltf file into an .aether project"
    )
    p_import.add_argument("source", help="the .glb or .gltf file to import")
    p_import.add_argument("destination", help="the .aether project to write")
    p_import.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_import.set_defaults(handler=cmd_import)

    p_export = sub.add_parser("export", parents=reading, help="export a scene to .glb or .gltf")
    p_export.add_argument("source", help="a .aether project (or a .glb/.gltf file)")
    p_export.add_argument("destination", help="the .glb or .gltf file to write")
    p_export.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_export.set_defaults(handler=cmd_export)

    p_topology = sub.add_parser(
        "topology", parents=reading, help="polygon topology and validation of one object's mesh"
    )
    p_topology.add_argument("path", help="path of a .aether, .glb or .gltf file")
    p_topology.add_argument("--object", required=True, help="object id or name")
    p_topology.add_argument(
        "--elements", action="store_true", help="list vertices, edges and faces"
    )
    p_topology.add_argument("--limit", type=int, default=None, help="list at most N of each kind")
    p_topology.add_argument(
        "--triangles", action="store_true", help="do not rebuild quads from triangle pairs"
    )
    p_topology.set_defaults(handler=cmd_topology)

    p_edit = sub.add_parser(
        "edit", parents=[common], help="apply a JSON edit script to a project (atomically)"
    )
    p_edit.add_argument("path", help="the .aether project to edit")
    p_edit.add_argument("script", help="edit script (.json), or - to read it from stdin")
    p_edit.add_argument("--output", help="write the result here instead of updating PATH")
    p_edit.add_argument("--dry-run", action="store_true", help="run and report, but write nothing")
    p_edit.set_defaults(handler=cmd_edit)

    p_proc = sub.add_parser("procedural", help="procedural geometry: inspect, run scripts, bake")
    proc = p_proc.add_subparsers(dest="procedural_command", required=True, metavar="SUBCOMMAND")
    writing = argparse.ArgumentParser(add_help=False)
    writing.add_argument("--output", help="write the result here instead of updating PATH")
    writing.add_argument("--dry-run", action="store_true", help="run and report, but write nothing")

    p_types = proc.add_parser("types", parents=[common], help="primitive and modifier schemas")
    p_types.set_defaults(handler=cmd_procedural_types)

    p_pinspect = proc.add_parser("inspect", parents=[common], help="parameters, geometry, stacks")
    p_pinspect.add_argument("path", help="the .aether project")
    p_pinspect.add_argument("--geometry", help="only this geometry (id or name)")
    p_pinspect.add_argument("--evaluate", action="store_true", help="also evaluate and summarize")
    p_pinspect.set_defaults(handler=cmd_procedural_inspect)

    p_pvalidate = proc.add_parser("validate", parents=[common], help="evaluate and check geometry")
    p_pvalidate.add_argument("path", help="the .aether project")
    p_pvalidate.add_argument("--geometry", help="only this geometry (id or name)")
    p_pvalidate.set_defaults(handler=cmd_procedural_validate)

    p_run = proc.add_parser(
        "run", parents=[common, writing], help="apply a JSON procedural script (atomically)"
    )
    p_run.add_argument("path", help="the .aether project")
    p_run.add_argument("script", help="procedural script (.json), or - to read it from stdin")
    p_run.set_defaults(handler=cmd_procedural_run)

    p_set = proc.add_parser(
        "set", parents=[common, writing], help="set parameters (NAME=VALUE, values as JSON)"
    )
    p_set.add_argument("path", help="the .aether project")
    p_set.add_argument("assignments", nargs="+", metavar="NAME=VALUE")
    p_set.add_argument("--geometry", help="set this geometry's parameters (default: scene's)")
    p_set.set_defaults(handler=cmd_procedural_set)

    p_bake = proc.add_parser(
        "bake", parents=[common, writing], help="replace procedural objects by static meshes"
    )
    p_bake.add_argument("path", help="the .aether project")
    p_bake.add_argument("--object", action="append", help="object id or name (repeatable)")
    p_bake.add_argument("--all", action="store_true", help="bake every procedural object")
    p_bake.set_defaults(handler=cmd_procedural_bake)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command
    if command == "procedural":
        command = f"procedural {args.procedural_command}"
    # Never crash on non-ASCII scene/object names when stdout is a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        try:
            result, text, code = args.handler(args)
        except AetherError as exc:  # e.g. a procedural object failing to evaluate
            raise _command_error(exc, Path(getattr(args, "path", "") or "")) from None
    except CommandError as exc:
        error: dict[str, Any] = {"type": exc.error_type, "message": exc.message}
        if exc.location:
            error["location"] = exc.location
        error.update(exc.details)
        if args.json:
            print(json.dumps({"ok": False, "command": command, "error": error}, indent=2))
        else:
            where = f"{exc.location}: " if exc.location else ""
            print(f"error: {where}{exc.message}", file=sys.stderr)
            for line in _diagnostic_lines(exc.details.get("diagnostics", [])):
                print(line, file=sys.stderr)
        return EXIT_FAILURE
    if args.json:
        print(json.dumps({"ok": True, "command": command, **result}, indent=2))
    else:
        print(text)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
