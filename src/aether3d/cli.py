"""Command-line interface: ``aether3d new | validate | inspect | geometry | import | export``.

Every command accepts ``--json`` to emit a single machine-readable JSON
object on stdout, both on success (``"ok": true``) and on failure
(``"ok": false`` plus an ``"error"`` object). Exit codes:

* ``0`` -- success
* ``1`` -- the command failed (invalid file, missing file, unsupported content, ...)
* ``2`` -- invalid command-line usage
* ``3`` -- ``geometry`` completed but found error-severity problems

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
    if isinstance(exc, AetherError):
        return CommandError(type(exc).__name__, str(exc))
    if isinstance(exc, OSError):
        return CommandError(type(exc).__name__, f"cannot access {path}: {exc.strerror or exc}")
    raise exc


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Never crash on non-ASCII scene/object names when stdout is a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        result, text, code = args.handler(args)
    except CommandError as exc:
        error: dict[str, Any] = {"type": exc.error_type, "message": exc.message}
        if exc.location:
            error["location"] = exc.location
        error.update(exc.details)
        if args.json:
            print(json.dumps({"ok": False, "command": args.command, "error": error}, indent=2))
        else:
            where = f"{exc.location}: " if exc.location else ""
            print(f"error: {where}{exc.message}", file=sys.stderr)
            for line in _diagnostic_lines(exc.details.get("diagnostics", [])):
                print(line, file=sys.stderr)
        return EXIT_FAILURE
    if args.json:
        print(json.dumps({"ok": True, "command": args.command, **result}, indent=2))
    else:
        print(text)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
