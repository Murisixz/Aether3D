"""Command-line interface: ``aether3d new | validate | inspect``.

Every command accepts ``--json`` to emit a single machine-readable JSON
object on stdout, both on success (``"ok": true``) and on failure
(``"ok": false`` plus an ``"error"`` object). Exit codes:

* ``0`` -- success
* ``1`` -- the command failed (invalid project, missing file, ...)
* ``2`` -- invalid command-line usage
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import AetherError, ProjectFormatError
from .scene import Scene, SceneObject
from .serialization import FORMAT_VERSION, loads, save_scene, scene_from_dict

EXIT_OK = 0
EXIT_FAILURE = 1


class CommandError(Exception):
    """A command failed for a reason that should be reported, not traced."""

    def __init__(self, error_type: str, message: str, location: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.location = location


def _read_project(path: Path) -> tuple[Scene, int]:
    """Load a project, returning the scene and the file's on-disk format version."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CommandError("FileNotFoundError", f"no such file: {path}") from None
    except UnicodeDecodeError:
        raise CommandError("ProjectFormatError", "file is not valid UTF-8 text") from None
    except OSError as exc:
        raise CommandError(type(exc).__name__, f"cannot read {path}: {exc.strerror}") from None
    try:
        document = loads(text)
        version = document.get("format_version") if isinstance(document, dict) else None
        return scene_from_dict(document), version
    except ProjectFormatError as exc:
        raise CommandError("ProjectFormatError", exc.message, exc.location) from None


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


def cmd_new(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = Path(args.path)
    if path.exists() and not args.force:
        raise CommandError("FileExistsError", f"{path} already exists (use --force to overwrite)")
    if path.is_dir():
        raise CommandError("IsADirectoryError", f"{path} is a directory")
    try:
        scene = Scene(args.name if args.name is not None else (path.stem or "Untitled Scene"))
        save_scene(scene, path)
    except AetherError as exc:
        raise CommandError(type(exc).__name__, str(exc)) from None
    except OSError as exc:
        raise CommandError(type(exc).__name__, f"cannot write {path}: {exc.strerror}") from None
    result = {"path": str(path), "scene": scene.name, "format_version": FORMAT_VERSION}
    return result, f'Created empty project "{scene.name}" at {path}'


def cmd_validate(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = Path(args.path)
    scene, version = _read_project(path)
    result = {
        "path": str(path),
        "valid": True,
        "format_version": version,
        "scene": scene.name,
        "object_count": len(scene),
        "mesh_count": _mesh_count(scene),
    }
    text = (
        f"OK: {path} is a valid Aether3D project "
        f"(format v{version}, {len(scene)} objects, {result['mesh_count']} meshes)"
    )
    return result, text


def cmd_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = Path(args.path)
    scene, version = _read_project(path)

    objects: list[dict[str, Any]] = []
    lines = [f'Scene "{scene.name}" ({len(scene)} objects, format v{version})']
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
        "format_version": version,
        "scene": {
            "name": scene.name,
            "object_count": len(scene),
            "mesh_count": _mesh_count(scene),
            "roots": [root.id for root in roots],
        },
        "objects": objects,
    }
    return result, "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON")

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

    p_validate = sub.add_parser("validate", parents=[common], help="check that a project is valid")
    p_validate.add_argument("path", help="path of the .aether file")
    p_validate.set_defaults(handler=cmd_validate)

    p_inspect = sub.add_parser(
        "inspect", parents=[common], help="show a project's objects and hierarchy"
    )
    p_inspect.add_argument("path", help="path of the .aether file")
    p_inspect.set_defaults(handler=cmd_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Never crash on non-ASCII scene/object names when stdout is a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        result, text = args.handler(args)
    except CommandError as exc:
        error: dict[str, Any] = {"type": exc.error_type, "message": exc.message}
        if exc.location:
            error["location"] = exc.location
        if args.json:
            print(json.dumps({"ok": False, "command": args.command, "error": error}, indent=2))
        else:
            where = f"{exc.location}: " if exc.location else ""
            print(f"error: {where}{exc.message}", file=sys.stderr)
        return EXIT_FAILURE
    if args.json:
        print(json.dumps({"ok": True, "command": args.command, **result}, indent=2))
    else:
        print(text)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
