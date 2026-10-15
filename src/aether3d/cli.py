"""Command-line interface.

Commands: ``new``, ``validate``, ``inspect``, ``geometry``, ``import``,
``export``, ``topology``, ``edit``, ``procedural`` (``types``, ``inspect``,
``validate``, ``run``, ``set``, ``bake``), ``render``, ``camera`` (``add``,
``set``, ``list``, ``frame``, ``activate``), ``light`` (``add``, ``set``,
``list``), ``object`` (``set``, ``remove``), ``scene`` (``set``, ``run``),
``gpu`` (``info``), ``material`` (``list``, ``presets``, ``schema``,
``create``, ``set``, ``assign``, ``texture``, ``inspect``, ``validate``,
``remove``, ``convert-colors``, ``preview``) and ``texture`` (``list``,
``add``, ``procedural``, ``set``, ``inspect``, ``preview``, ``remove``).

Every command accepts ``--json`` to emit a single machine-readable JSON
object on stdout, both on success (``"ok": true``) and on failure
(``"ok": false`` plus an ``"error"`` object). Exit codes:

* ``0`` -- success
* ``1`` -- the command failed (invalid file, missing file, unsupported content, ...)
* ``2`` -- invalid command-line usage
* ``3`` -- ``geometry``/``topology``/``procedural validate``/``material validate``
  completed but found error-severity problems, or ``gpu info --check`` found
  that GPU rendering does not work

Commands that read a scene accept native ``.aether`` projects as well as
``.glb``/``.gltf`` files (which are imported in memory).
"""

from __future__ import annotations

import argparse
import json
import sys
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import (
    AetherError,
    AssetFormatError,
    HistoryError,
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
    from .errors import MaterialError, RenderError

    if isinstance(exc, (MaterialError, RenderError)):
        return CommandError(type(exc).__name__, exc.message, exc.location, _procedural_details(exc))
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


# Scenes of writing commands whose history is kept in a history file, and what happened
# to it (reported with the command's result).
_HISTORY: weakref.WeakKeyDictionary[Scene, dict[str, Any]] = weakref.WeakKeyDictionary()
_history_report: dict[str, Any] = {}


def _attach_history(scene: Scene, path: Path) -> None:
    """Give a project loaded to be changed its saved history, if it has one.

    Changes are then recorded on top of it and saved with it. A history that no longer
    matches the project is set aside (reported); a new one starts from this state.
    """
    from ._history_file import history_path

    file = history_path(path)
    if not file.exists():
        return
    info: dict[str, Any] = {"file": str(file), "note": None}
    try:
        scene.history.load(file, project_dir=path.absolute().parent)
    except HistoryError as exc:
        info["note"] = (
            f"the saved history was not used ([{exc.code}] {exc.message}); a new one starts"
        )
    _HISTORY[scene] = info


def _read_project(path: Path, history: bool = False) -> tuple[Scene, int]:
    """Load a ``.aether`` project, returning the scene and its on-disk format version.

    With ``history`` (commands that change the project) its saved history comes too.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise CommandError("ProjectFormatError", "file is not valid UTF-8 text") from None
    except OSError as exc:
        raise _command_error(exc, path) from None
    try:
        document = loads(text)
        version = document.get("format_version") if isinstance(document, dict) else None
        scene = scene_from_dict(document, path.absolute().parent)
    except ProjectFormatError as exc:
        raise _command_error(exc, path) from None
    if history:
        _attach_history(scene, path)
    return scene, version


def _load_scene(
    path: Path, allow_lossy: bool = False, history: bool = False
) -> tuple[Scene, dict[str, Any]]:
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
    scene, version = _read_project(path, history)
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
        "kind": obj.kind,
        "camera": None if obj.camera is None else obj.camera.to_dict(),
        "light": None if obj.light is None else obj.light.to_dict(),
        "appearance": obj.appearance.to_dict(),
        "material": None
        if obj._material is None
        else {"id": obj._material, "name": obj.material.name},
        "group_materials": {
            str(g): {"id": m.id, "name": m.name} for g, m in obj.group_materials.items()
        },
        "visible": obj.visible,
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
        if obj.camera is not None:
            active = " active" if scene.active_camera is obj else ""
            details.append(f"camera={obj.camera.projection}{active}")
        if obj.light is not None:
            details.append(f"light={obj.light.type}")
        if not obj.visible:
            details.append("hidden")
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
        result = export_gltf(
            scene, destination, allow_lossy=args.allow_lossy, bake_textures=args.bake_textures
        )
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
        written = str(save_scene(scene, output))
    except (AetherError, OSError) as exc:
        raise _command_error(exc, output) from None
    _save_history(scene, output, args)
    return written


def _save_history(scene: Scene, output: Path, args: argparse.Namespace) -> None:
    """Save the history next to the written project, if it has one or --history asks."""
    info = _HISTORY.get(scene)
    if info is None and not getattr(args, "history", False):
        return
    from ._history_file import history_path

    file = history_path(output)
    try:
        scene.history.save(file, project_dir=output.absolute().parent)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, file) from None
    _history_report.update(
        {
            "file": str(file),
            "steps": len(scene.history),
            "undone": len(scene.history.redo_entries),
            "note": None if info is None else info["note"],
        }
    )


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
    scene, info = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:
        result = run_edit_script(scene, script, label=_script_label(script, "edit", args))
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
        _save_history(scene, output, args)
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


def _script_label(script: Any, command: str, args: argparse.Namespace) -> str:
    """The history label of a script run: its own ``"label"``, else the command."""
    own = script.get("label") if isinstance(script, dict) else None
    if isinstance(own, str) and own.strip():
        return own
    name = "stdin" if args.script == "-" else Path(args.script).name
    return f"aether3d {command} {name}"


def _run_procedural(scene: Scene, script: Any, label: str | None = None) -> dict[str, Any]:
    from .editing.script import ScriptError
    from .procedural.script import run_procedural_script

    try:
        return run_procedural_script(scene, script, label=label)
    except ScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("ScriptError", exc.message, exc.location, details) from None


def cmd_procedural_run(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    result = _run_procedural(scene, script, _script_label(script, "procedural run", args))
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
    scene, _ = _load_scene(path, history=True)
    operations = []
    for assignment in args.assignments:
        name, value = _parse_assignment(assignment)
        op: dict[str, Any] = {"op": "set_parameter", "name": name, "value": value}
        if args.geometry:
            op["geometry"] = args.geometry
        operations.append(op)
    script = {"format": "aether3d.procedural-script", "version": 1, "operations": operations}
    names = ", ".join(a.partition("=")[0] for a in args.assignments)
    result = _run_procedural(scene, script, f"aether3d procedural set {names}")
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"set {names}"]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_procedural_bake(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "procedural bake")
    scene, _ = _load_scene(path, history=True)
    if args.all:
        targets = [o.id for o in scene if o.is_procedural]
    elif args.object:
        targets = [_find_object(scene, ref).id for ref in args.object]
    else:
        raise CommandError("UsageError", "give --object (repeatable) or --all")
    operations = [{"op": "bake", "object": oid} for oid in targets]
    if operations:
        script = {"format": "aether3d.procedural-script", "version": 1, "operations": operations}
        _run_procedural(scene, script, "aether3d procedural bake")
    written = _write_or_dry_run(scene, path, args)
    baked = [{"id": oid, "name": scene.get(oid).name} for oid in targets]
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, "baked": baked}
    lines = [f"baked {len(baked)} object(s): " + ", ".join(b["name"] for b in baked)]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


# --------------------------------------------------------------------------- #
# Rendering: render, camera, light, object, scene
# --------------------------------------------------------------------------- #


def _render_error(exc: BaseException) -> CommandError:
    from .errors import RenderError

    if isinstance(exc, RenderError):
        details: dict[str, Any] = {"code": exc.code}
        if exc.hint:
            details["hint"] = exc.hint
        if exc.details:
            details["details"] = exc.details
        return CommandError("RenderError", exc.message, exc.location, details)
    raise exc


_SHADOW_FLAGS = {
    "shadow_lights": ("lights", "--shadow-lights"),
    "shadow_map_size": ("map_size", "--shadow-map-size"),
    "shadow_bias": ("bias", "--shadow-bias"),
    "shadow_normal_offset": ("normal_offset", "--shadow-normal-offset"),
    "shadow_pcf_radius": ("pcf_radius", "--shadow-pcf-radius"),
}


def _shadow_options(args: argparse.Namespace) -> dict[str, Any] | None:
    given = {k: v for k, v in vars(args).items() if k in _SHADOW_FLAGS and v is not None}
    if not args.shadows:
        if given:
            flags = ", ".join(_SHADOW_FLAGS[k][1] for k in given)
            raise CommandError("UsageError", f"{flags} need(s) --shadows")
        return None
    return {_SHADOW_FLAGS[k][0]: v for k, v in given.items()}


_PATH_FLAGS = {
    "spp": "samples",
    "bounces": "bounces",
    "seed": "seed",
    "light_samples": "light_samples",
    "russian_roulette": "russian_roulette",
    "clamp": "clamp",
    "workers": "workers",
}
_AO_FLAGS = {"ao_samples": "samples", "ao_distance": "distance", "ao_strength": "strength"}


def _path_options(args: argparse.Namespace) -> tuple[dict | None, Any]:
    """``path`` and ``ao`` options from the path tracer's flags (None: not given)."""
    path = {key: getattr(args, flag) for flag, key in _PATH_FLAGS.items()}
    path = {k: v for k, v in path.items() if v is not None}
    if args.ao_only:
        path["integrator"] = "ao"
    ao_given = {key: getattr(args, flag) for flag, key in _AO_FLAGS.items()}
    ao_given = {k: v for k, v in ao_given.items() if v is not None}
    ao: Any = ao_given or (True if (args.ao or args.ao_only) else None)
    if args.backend not in ("path", "gpu_path"):
        given = [f"--{f.replace('_', '-')}" for f in _PATH_FLAGS if getattr(args, f) is not None]
        given += [f"--{f.replace('_', '-')}" for f in _AO_FLAGS if getattr(args, f) is not None]
        given += [
            flag
            for flag in ("--ao", "--ao-only", "--passes")
            if getattr(args, flag[2:].replace("-", "_"))
        ]
        if given:
            verb = "applies" if len(given) == 1 else "apply"
            raise CommandError(
                "UsageError",
                f"{', '.join(given)} {verb} to the path tracer: add --backend path (CPU) or "
                "--backend gpu_path (GPU)",
            )
        return None, None
    if args.backend == "gpu_path" and args.workers is not None:
        raise CommandError(
            "UsageError",
            "--workers sets the CPU path tracer's processes; --backend gpu_path runs on the GPU",
        )
    if args.samples is not None:
        raise CommandError(
            "UsageError",
            "--samples is raster antialiasing; give the path tracer's samples per pixel with --spp",
        )
    return path or None, ao


def _write_passes(result: Any, directory: Path, force: bool) -> list[str]:
    from ._fileio import write_atomic
    from .rendering.hdr import encode_pfm

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, values in sorted(result.passes.items()):
        target = directory / f"{name}.pfm"
        _check_destination(target, force)
        write_atomic(target, encode_pfm(values))
        written.append(str(target))
    return written


def _render_summary(result: Any, path: Path, written: Path) -> str:
    stats = result.stats
    backend = result.backend
    if backend["name"] in ("path", "gpu_path"):
        what = "AO preview" if backend["integrator"] == "ao" else f"{backend['bounces']} bounces"
        where = (
            "on the CPU"
            if backend["name"] == "path"
            else f"on the GPU ({backend['adapter']['name']})"
        )
        how = (
            f"path tracer {where}: {backend['samples_rendered']} samples per pixel, {what}, "
            f"seed {backend['seed']}, {stats['rays']:,} rays "
            f"({stats['rays_per_second'] / 1e6:.2f} M rays/s)"
        )
    elif backend["name"] == "gpu":
        aa = backend["antialiasing"]
        adapter = backend["adapter"]
        how = f"GPU: {adapter['name']} ({adapter['backend']}), {aa['msaa']}x MSAA"
        if aa["supersample"] > 1:
            how += f" on {aa['supersample']}x{aa['supersample']} supersampling"
        if backend["shadows"]:
            how += ", shadows from " + ", ".join(m["light"] for m in backend["shadows"]["maps"])
    else:
        samples = result.settings["samples"]
        how = f"{samples}x{samples} samples per pixel"
    return (
        f"Rendered {path} -> {written} ({result.width}x{result.height}, {how}, "
        f"camera {result.camera['name']!r}, {stats['triangles']} triangles, "
        f"{stats['timings']['total']:.2f} s)"
    )


def cmd_render(args: argparse.Namespace) -> CommandOutput:
    from .errors import RenderError
    from .rendering import render

    shadows = _shadow_options(args)
    path_options, ao = _path_options(args)
    path = Path(args.path)
    output = Path(args.output) if args.output else path.with_suffix(".png")
    if output.suffix.lower() not in (".png", ".hdr", ".pfm"):
        raise CommandError(
            "ValidationError", f"output must be a .png, .hdr or .pfm file, got {output}"
        )
    _check_destination(output, args.force)
    scene, info = _load_scene(path, args.allow_lossy)
    try:
        if args.background is not None:
            scene.world.background = args.background
        result = render(
            scene,
            camera=args.camera,
            width=args.width,
            height=args.height,
            samples=args.samples,
            exposure=args.exposure,
            cull=args.cull,
            default_lights=False if args.no_default_lights else None,
            frame=args.frame,
            margin=args.margin,
            backend=args.backend,
            adapter=args.adapter,
            msaa=args.msaa,
            supersample=args.supersample,
            shadows=shadows,
            path=path_options,
            ao=ao,
            tonemap=args.tonemap,
        )
        written = result.save(output)
    except RenderError as exc:
        raise _render_error(exc) from None
    data = {"path": str(path), **info, **result.to_dict()}
    if args.passes:
        data["pass_files"] = _write_passes(result, Path(args.passes), args.force)
    lines = [
        _render_summary(result, path, written),
        *_diagnostic_lines([d.to_dict() for d in result.diagnostics]),
    ]
    return data, "\n".join(lines), EXIT_OK


def cmd_gpu_info(args: argparse.Namespace) -> CommandOutput:
    from .rendering.gpu import gpu_status

    status = gpu_status(adapter=args.adapter, check=args.check)
    lines = []
    versions = status["versions"]
    if versions:
        lines.append(
            f"wgpu {versions['wgpu']} (wgpu-native {versions['wgpu_native']}), "
            f"Python {status['python']}, {status['platform']}"
        )
    for note in status["notes"]:
        lines.append(f"note: {note}")
    if status["adapters"]:
        lines.append("Adapters (best first; select with --adapter INDEX or a name part):")
        for a in status["adapters"]:
            kind = f"{a['type']}, software" if a["software"] else a["type"]
            mark = "  <- auto" if a["recommended"] else ""
            driver = f"driver {a['driver']!r}"
            lines.append(f"  [{a['index']}] {a['name']} ({a['backend']}, {kind}) {driver}{mark}")
    for attempt in status["attempts"]:
        lines.append(f"  unusable: {attempt['adapter']}: [{attempt['code']}] {attempt['message']}")
    selected = status["selected"]
    if selected is not None:
        adapter = selected["adapter"]
        limits = selected["limits"]
        lines.append(
            f"Selected: [{adapter['index']}] {adapter['name']} ({adapter['backend']}): MSAA "
            f"{', '.join(map(str, selected['msaa']))}; GPU timestamps "
            f"{'yes' if selected['timestamps'] else 'no'}; largest texture "
            f"{limits.get('max-texture-dimension-2d')}; device created in "
            f"{selected['device_init_seconds']:.2f} s"
        )
    if "check" in status:
        check = status["check"]
        if check["ok"]:
            lines.append(
                f"Check passed: {len(check['checks'])} checks on {check['adapter']} "
                f"in {check['seconds']:.2f} s"
            )
        else:
            failed = [k for k, v in check["checks"].items() if not v["ok"]]
            error = check.get("error")
            lines.append(
                "Check FAILED: "
                + (f"[{error['code']}] {error['message']}" if error else ", ".join(failed))
            )
    if not status["available"]:
        lines.append(f"GPU rendering is not available: {status['reason']}")
        if status["hint"]:
            lines.append(f"Hint: {status['hint']}")
    code = EXIT_OK
    if args.check and not status["available"]:
        code = EXIT_GEOMETRY_ERRORS
    return status, "\n".join(lines), code


def _run_scene(
    args: argparse.Namespace,
    operations: list[dict[str, Any]],
    command: str,
    *,
    trusted_paths: bool = False,
):
    from .editing.script import ScriptError
    from .rendering.script import run_scene_script

    path = Path(args.path)
    _project_only(path, command)
    scene, _ = _load_scene(path, history=True)
    script = {"format": "aether3d.scene-script", "version": 1, "operations": operations}
    try:
        result = run_scene_script(
            scene,
            script,
            base_dir=path.absolute().parent,
            trusted_paths=trusted_paths,
            label=f"aether3d {command}",
        )
    except ScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("ScriptError", exc.message, exc.location, details) from None
    written = None
    if getattr(args, "dry_run", False) is False:
        written = _write_or_dry_run(scene, path, args)
    data = {
        "path": str(path),
        "output": written,
        "dry_run": bool(getattr(args, "dry_run", False)),
        **result,
    }
    return scene, data, written


def _saved(data: dict[str, Any]) -> str:
    return "dry run: nothing written" if data["dry_run"] else f"saved {data['output']}"


def _vec(values: Sequence[float] | None) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


def _look_target(args: argparse.Namespace) -> Any:
    if getattr(args, "look_at", None) is not None and getattr(args, "look_at_object", None):
        raise CommandError("UsageError", "give --look-at or --look-at-object, not both")
    if getattr(args, "look_at", None) is not None:
        return _vec(args.look_at)
    return getattr(args, "look_at_object", None)


def _camera_params(args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "projection": args.projection,
        "fov": args.fov,
        "fov_axis": args.fov_axis,
        "ortho_scale": args.ortho_scale,
        "near": args.near,
        "far": args.far,
    }
    out = {k: v for k, v in params.items() if v is not None}
    if args.aspect_ratio is not None:
        out["aspect_ratio"] = None if args.aspect_ratio == "none" else _float(args.aspect_ratio)
    return out


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise CommandError("UsageError", f"expected a number or 'none', got {text!r}") from None


def _placement(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "position", None) is not None:
        out["position"] = _vec(args.position)
    if getattr(args, "rotation_euler", None) is not None:
        out["rotation_euler"] = _vec(args.rotation_euler)
    target = _look_target(args)
    if target is not None:
        out["look_at"] = target
    if getattr(args, "up", None) is not None:
        out["up"] = _vec(args.up)
    return out


def cmd_camera_add(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "add_camera", **_camera_params(args), **_placement(args)}
    if args.name:
        op["name"] = args.name
    if args.parent:
        op["parent"] = args.parent
    if args.active is not None:
        op["active"] = args.active
    if args.frame:
        op["frame"] = True
        op["frame_margin"] = args.margin
        if args.frame_direction is not None:
            op["frame_direction"] = _vec(args.frame_direction)
    _, data, _ = _run_scene(args, [op], "camera add")
    result = data["operations"][0]
    active = " (active)" if result["active"] else ""
    created = result["object"]
    lines = [f"Added camera {created['name']!r} [{created['id']}]{active}", _saved(data)]
    return data, "\n".join(lines), EXIT_OK


def cmd_camera_set(args: argparse.Namespace) -> CommandOutput:
    operations: list[dict[str, Any]] = []
    params = _camera_params(args)
    if params:
        operations.append({"op": "set_camera", "camera": args.camera, **params})
    placement = _placement(args)
    if placement:
        operations.append({"op": "set_object", "object": args.camera, **placement})
    if args.activate:
        operations.append({"op": "set_active_camera", "camera": args.camera})
    if not operations:
        raise CommandError("UsageError", "nothing to change; give camera parameters or placement")
    _, data, _ = _run_scene(args, operations, "camera set")
    return data, f"Updated camera {args.camera!r}\n{_saved(data)}", EXIT_OK


def cmd_camera_frame(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "frame_camera", "margin": args.margin}
    if args.camera:
        op["camera"] = args.camera
    if args.object:
        op["targets"] = list(args.object)
    if args.direction is not None:
        op["direction"] = _vec(args.direction)
    if args.no_fit_clipping:
        op["fit_clipping"] = False
    _, data, _ = _run_scene(args, [op], "camera frame")
    framing = data["operations"][0]
    pos = framing["framing"]["position"]
    text = f"Framed camera {framing['object']['name']!r} at {_fmt_vec(pos)}\n{_saved(data)}"
    return data, text, EXIT_OK


def cmd_camera_activate(args: argparse.Namespace) -> CommandOutput:
    camera = None if args.camera == "none" else args.camera
    _, data, _ = _run_scene(
        args, [{"op": "set_active_camera", "camera": camera}], "camera activate"
    )
    active = data["operations"][0]["active_camera"]
    text = f"Active camera: {active['name'] if active else 'none'}\n{_saved(data)}"
    return data, text, EXIT_OK


def cmd_camera_list(args: argparse.Namespace) -> CommandOutput:
    from .rendering.script import rendering_summary

    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    summary = rendering_summary(scene)
    lines = [f'Cameras in "{scene.name}" ({len(summary["cameras"])}):']
    for c in summary["cameras"]:
        mark = " [active]" if c["active"] else ""
        shape = (
            f"fov {c['fov']:g}"
            if c["projection"] == "perspective"
            else (f"ortho_scale {c['ortho_scale']:g}")
        )
        where = _fmt_vec(c["position"]) if "position" in c else "?"
        lines.append(f"  {c['name']}{mark}: {c['projection']}, {shape}, at {where}")
    return {"path": str(path), **info, **summary}, "\n".join(lines), EXIT_OK


def _light_params(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("color", "intensity", "falloff", "radius", "angle"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    if getattr(args, "range", None) is not None:
        out["range"] = None if args.range == "none" else _float(args.range)
    return out


def cmd_light_add(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {
        "op": "add_light",
        "type": args.type,
        **_light_params(args),
        **_placement(args),
    }
    if args.direction is not None:
        op["direction"] = _vec(args.direction)
    if args.name:
        op["name"] = args.name
    if args.parent:
        op["parent"] = args.parent
    _, data, _ = _run_scene(args, [op], "light add")
    created = data["operations"][0]["object"]
    return data, f"Added {args.type} light {created['name']!r}\n{_saved(data)}", EXIT_OK


def cmd_light_set(args: argparse.Namespace) -> CommandOutput:
    operations: list[dict[str, Any]] = []
    params = _light_params(args)
    if args.type is not None:
        params["type"] = args.type
    if params:
        operations.append({"op": "set_light", "light": args.light, **params})
    placement = _placement(args)
    if args.direction is not None:
        placement["look_along"] = _vec(args.direction)
    if placement:
        operations.append({"op": "set_object", "object": args.light, **placement})
    if not operations:
        raise CommandError("UsageError", "nothing to change; give light parameters or placement")
    _, data, _ = _run_scene(args, operations, "light set")
    return data, f"Updated light {args.light!r}\n{_saved(data)}", EXIT_OK


def cmd_light_list(args: argparse.Namespace) -> CommandOutput:
    from .rendering.script import rendering_summary

    path = Path(args.path)
    scene, info = _load_scene(path, getattr(args, "allow_lossy", False))
    summary = rendering_summary(scene)
    world = summary["world"]
    lines = [
        f'Lights in "{scene.name}" ({len(summary["lights"])}); ambient '
        f"{_fmt_vec(world['ambient_color'])} x {world['ambient_intensity']:g}:"
    ]
    for light in summary["lights"]:
        where = (
            f"from {_fmt_vec(light['position'])}"
            if light["type"] == "point"
            else f"along {_fmt_vec(light['direction'] or (0, 0, 0))}"
        )
        hidden = " (hidden)" if not light["visible"] else ""
        size = ""
        if light["type"] == "point" and light["radius"]:
            size = f", radius {light['radius']:g}"
        elif light["type"] == "directional" and light["angle"]:
            size = f", angle {light['angle']:g} deg"
        lines.append(
            f"  {light['name']}: {light['type']} {_fmt_vec(light['color'])} x "
            f"{light['intensity']:g} {where}{size}{hidden}"
        )
    env = world["environment"]
    lines.append(f"  environment (path tracer): {env['type']}")
    return {"path": str(path), **info, **summary}, "\n".join(lines), EXIT_OK


def _bool_arg(text: str) -> bool:
    lowered = text.lower()
    if lowered in ("true", "yes", "on", "1"):
        return True
    if lowered in ("false", "no", "off", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def cmd_object_set(args: argparse.Namespace) -> CommandOutput:
    op: dict[str, Any] = {"op": "set_object", "object": args.object, **_placement(args)}
    for key in ("color", "specular", "shininess", "visible"):
        value = getattr(args, key)
        if value is not None:
            op[key] = value
    if args.scale is not None:
        op["scale"] = _vec(args.scale)
    if len(op) == 2:
        raise CommandError("UsageError", "nothing to change")
    _, data, _ = _run_scene(args, [op], "object set")
    return data, f"Updated {args.object!r}\n{_saved(data)}", EXIT_OK


def cmd_object_remove(args: argparse.Namespace) -> CommandOutput:
    _, data, _ = _run_scene(args, [{"op": "remove", "object": args.object}], "object remove")
    removed = data["operations"][0]["removed"]
    return data, f"Removed {len(removed)} object(s)\n{_saved(data)}", EXIT_OK


def cmd_scene_set(args: argparse.Namespace) -> CommandOutput:
    operations: list[dict[str, Any]] = []
    world = {
        k: getattr(args, k)
        for k in ("background", "ambient_color", "ambient_intensity")
        if getattr(args, k) is not None
    }
    env_flags = {
        "environment": "type",
        "environment_color": "color",
        "environment_intensity": "intensity",
        "sky_zenith": "zenith_color",
        "sky_horizon": "horizon_color",
        "sky_ground": "ground_color",
    }
    environment = {
        key: getattr(args, flag)
        for flag, key in env_flags.items()
        if getattr(args, flag) is not None
    }
    if environment:
        world["environment"] = environment
    if world:
        operations.append({"op": "set_world", **world})
    settings = {
        k: getattr(args, k)
        for k in ("width", "height", "samples", "exposure", "cull", "default_lights")
        if getattr(args, k) is not None
    }
    if settings:
        operations.append({"op": "set_render_settings", **settings})
    if args.active_camera is not None:
        camera = None if args.active_camera == "none" else args.active_camera
        operations.append({"op": "set_active_camera", "camera": camera})
    if not operations:
        raise CommandError("UsageError", "nothing to change")
    _, data, _ = _run_scene(args, operations, "scene set")
    return data, f"Updated scene settings\n{_saved(data)}", EXIT_OK


def cmd_scene_run(args: argparse.Namespace) -> CommandOutput:
    from .editing.script import ScriptError
    from .rendering.script import run_scene_script

    path = Path(args.path)
    _project_only(path, "scene run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:  # image paths in a script stay inside the project's directory
        result = run_scene_script(
            scene,
            script,
            base_dir=path.absolute().parent,
            label=_script_label(script, "scene run", args),
        )
    except ScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("ScriptError", exc.message, exc.location, details) from None
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"{len(result['operations'])} operations applied", _saved(data)]
    return data, "\n".join(lines), EXIT_OK


def cmd_scene_merge(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "scene merge")
    scene, _ = _load_scene(path, history=True)
    source = Path(args.source)
    try:
        with scene.transaction(f"aether3d scene merge {source.name}"):
            copies = scene.merge(source, parent=args.parent, allow_lossy=args.allow_lossy)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, source) from None
    written = _write_or_dry_run(scene, path, args)
    data = {
        "path": str(path),
        "source": str(source),
        "output": written,
        "dry_run": args.dry_run,
        "objects": [{"id": o.id, "name": o.name} for o in copies],
    }
    names = ", ".join(o.name for o in copies)
    return data, f"Merged {source.name}: {names}\n{_saved(data)}", EXIT_OK


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #


def _history_scene(args: argparse.Namespace) -> tuple[Scene, Path, Path]:
    """The project of a history command and its (verified) saved history."""
    from ._history_file import history_path

    path = Path(args.path)
    _project_only(path, f"history {args.subcommand}")
    scene, _ = _read_project(path)
    file = history_path(path)
    try:
        scene.history.load(file, project_dir=path.absolute().parent)
    except HistoryError as exc:
        details = {"code": exc.code}
        if exc.code == "NO_HISTORY":
            details["hint"] = (
                f"start one with `aether3d history start {path}`, or pass --history to a "
                "command that changes the project"
            )
        elif exc.hint:
            details["hint"] = exc.hint
        raise CommandError("HistoryError", exc.message, exc.location, details) from None
    return scene, path, file


def _history_summary(scene: Scene, path: Path, file: Path) -> dict[str, Any]:
    return {"path": str(path), "file": str(file), **scene.history.to_dict()}


def _entry_lines(history_data: dict[str, Any]) -> list[str]:
    lines = []
    marks = {v: k for k, v in history_data["checkpoints"].items()}
    for e in history_data["entries"]:
        mark = f"  [checkpoint {marks[e['index']]!r}]" if e["index"] in marks else ""
        lines.append(f"  {e['index']:4d}  {e['label']}{mark}")
    lines.append("  ----  (current state)")
    for e in history_data["redo"]:
        lines.append(f"  {e['index']:4d}  {e['label']}  (undone)")
    return lines


def _write_history_command(
    scene: Scene, path: Path, file: Path, args: argparse.Namespace, project: bool
) -> None:
    if getattr(args, "dry_run", False):
        return
    try:
        if project:
            save_scene(scene, path)
        scene.history.save(file, project_dir=path.absolute().parent)
    except (AetherError, OSError) as exc:
        raise _command_error(exc, path) from None


def cmd_history_show(args: argparse.Namespace) -> CommandOutput:
    scene, path, file = _history_scene(args)
    data = _history_summary(scene, path, file)
    text = [
        f"History of {path}: {len(data['entries'])} step(s) to undo, {len(data['redo'])} to "
        f"redo, {data['bytes'] / 2**20:.1f} MiB",
        *_entry_lines(data),
    ]
    return data, "\n".join(text), EXIT_OK


def cmd_history_start(args: argparse.Namespace) -> CommandOutput:
    from ._history_file import history_path

    path = Path(args.path)
    _project_only(path, "history start")
    file = history_path(path)
    if file.exists() and not args.force:
        raise CommandError(
            "FileExistsError",
            f"{file} already exists",
            details={"hint": "use `aether3d history clear` to empty it, or --force to replace it"},
        )
    scene, _ = _read_project(path)
    _write_history_command(scene, path, file, args, project=False)
    data = {"path": str(path), "file": str(file)}
    return data, f"Started a history for {path} ({file.name})", EXIT_OK


def _history_move(args: argparse.Namespace, forward: bool) -> CommandOutput:
    scene, path, file = _history_scene(args)
    history = scene.history
    try:
        moved = history.redo(args.steps) if forward else history.undo(args.steps)
    except HistoryError as exc:
        raise CommandError("HistoryError", exc.message, exc.location, {"code": exc.code}) from None
    _write_history_command(scene, path, file, args, project=True)
    verb = "Redid" if forward else "Undid"
    data = {
        **_history_summary(scene, path, file),
        "undone" if not forward else "redone": [e.to_dict() for e in moved],
        "dry_run": args.dry_run,
    }
    text = [f"{verb} {len(moved)} step(s):", *(f"  {e.index:4d}  {e.label}" for e in moved)]
    text.append("dry run: nothing written" if args.dry_run else f"saved {path} and {file.name}")
    return data, "\n".join(text), EXIT_OK


def cmd_history_undo(args: argparse.Namespace) -> CommandOutput:
    return _history_move(args, forward=False)


def cmd_history_redo(args: argparse.Namespace) -> CommandOutput:
    return _history_move(args, forward=True)


def cmd_history_checkpoint(args: argparse.Namespace) -> CommandOutput:
    scene, path, file = _history_scene(args)
    try:
        scene.history.checkpoint(args.name)
    except HistoryError as exc:
        raise CommandError("HistoryError", exc.message, exc.location, {"code": exc.code}) from None
    _write_history_command(scene, path, file, args, project=False)
    data = _history_summary(scene, path, file)
    return data, f"Checkpoint {args.name!r} at step {scene.history.checkpoints[args.name]}", EXIT_OK


def cmd_history_goto(args: argparse.Namespace) -> CommandOutput:
    scene, path, file = _history_scene(args)
    try:
        moved = scene.history.goto(args.name)
    except HistoryError as exc:
        details = {"code": exc.code, **({"hint": exc.hint} if exc.hint else {})}
        raise CommandError("HistoryError", exc.message, exc.location, details) from None
    _write_history_command(scene, path, file, args, project=True)
    data = {**_history_summary(scene, path, file), "moved": [e.to_dict() for e in moved]}
    text = f"At checkpoint {args.name!r} ({len(moved)} step(s) undone or redone)"
    return data, text, EXIT_OK


def cmd_history_clear(args: argparse.Namespace) -> CommandOutput:
    scene, path, file = _history_scene(args)
    count = len(scene.history) + len(scene.history.redo_entries)
    scene.history.clear()
    _write_history_command(scene, path, file, args, project=False)
    return (
        {"path": str(path), "file": str(file), "cleared": count},
        (f"Cleared {count} step(s) from {file.name}"),
        EXIT_OK,
    )


def cmd_history_verify(args: argparse.Namespace) -> CommandOutput:
    scene, path, file = _history_scene(args)  # loading replays it on a copy already
    report = scene.history.verify()
    data = {"path": str(path), "file": str(file), **report}
    text = (
        f"{report['steps']} step(s) undone and redone: the project comes back exactly"
        if report["ok"]
        else "replaying the history does not restore the project"
    )
    return data, text, EXIT_OK if report["ok"] else EXIT_FAILURE


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
    p_export.add_argument(
        "--bake-textures",
        type=int,
        metavar="SIZE",
        help="bake procedural textures into SIZE x SIZE PNG images (16-8192); without it they "
        "are left out, with a warning",
    )
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
    writing.add_argument(
        "--history",
        action="store_true",
        help="record the change in the project's history file (<project>.history), starting "
        "one if needed; projects that have one always record",
    )

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

    _add_render_parsers(sub, common, lossy, writing)
    from ._cli_materials import add_material_parsers

    add_material_parsers(sub, common, lossy, writing)
    return parser


def _add_render_parsers(sub, common, lossy, writing) -> None:
    vec = {"nargs": 3, "type": float, "metavar": ("X", "Y", "Z")}

    p_render = sub.add_parser(
        "render",
        parents=[common, lossy],
        help="render an image with Aether3D's CPU rasterizer (default), GPU rasterizer or "
        "CPU path tracer",
    )
    p_render.add_argument("path", help="a .aether project (or a .glb/.gltf file, with --frame)")
    p_render.add_argument(
        "-o",
        "--output",
        help="the image to write: .png, or .hdr/.pfm for linear high dynamic range "
        "(default: PATH.png)",
    )
    p_render.add_argument("--width", type=int, help="image width in pixels")
    p_render.add_argument("--height", type=int, help="image height in pixels")
    p_render.add_argument("--camera", help="camera object id or name (default: active camera)")
    p_render.add_argument("--samples", type=int, help="antialiasing: N x N samples per pixel (1-4)")
    p_render.add_argument("--exposure", type=float, help="exposure in stops (EV)")
    p_render.add_argument("--cull", choices=["none", "back", "front"], help="face culling")
    p_render.add_argument("--background", help="background color for this render (#rrggbb)")
    p_render.add_argument(
        "--no-default-lights", action="store_true", help="no built-in light if the scene has none"
    )
    p_render.add_argument(
        "--frame",
        action="store_true",
        help="frame the camera on all visible geometry for this render (a temporary camera "
        "is used if the scene has none); the project is not modified",
    )
    p_render.add_argument("--margin", type=float, default=0.05, help="free border with --frame")
    p_render.add_argument("--force", action="store_true", help="overwrite an existing image")
    p_render.add_argument(
        "--backend",
        choices=["cpu", "gpu", "auto", "path", "gpu_path"],
        default="cpu",
        help="cpu (default), gpu (fails if no usable GPU), auto (GPU if available, else CPU), "
        "path (the path tracer: global illumination on the CPU, much slower) or gpu_path "
        "(the same path tracer on the GPU; fails if no usable GPU)",
    )
    p_render.add_argument(
        "--tonemap",
        choices=["none", "neutral"],
        default="none",
        help="map linear light to the PNG: none (clip, default) or neutral (Khronos PBR "
        "Neutral: compresses highlights)",
    )
    p_render.add_argument(
        "--adapter",
        help="GPU adapter: index, part of its name (nvidia), discrete/integrated/cpu, "
        "vulkan/metal/d3d12 (default: $AETHER3D_GPU_ADAPTER or the best one)",
    )
    p_render.add_argument("--msaa", type=int, help="GPU: MSAA samples per pixel (1, 2, 4, 8, 16)")
    p_render.add_argument(
        "--supersample", type=int, help="GPU: render N times larger per axis and average (1-4)"
    )
    p_render.add_argument(
        "--shadows", action="store_true", help="GPU: shadow maps for directional lights"
    )
    p_render.add_argument(
        "--shadow-lights", nargs="+", metavar="LIGHT", help="only these lights cast shadows"
    )
    p_render.add_argument("--shadow-map-size", type=int, help="shadow map texels per side (2048)")
    p_render.add_argument("--shadow-bias", type=float, help="shadow depth bias in texels (0.5)")
    p_render.add_argument(
        "--shadow-normal-offset", type=float, help="extra shadow normal offset in texels (0.5)"
    )
    p_render.add_argument(
        "--shadow-pcf-radius", type=int, help="shadow filter radius in texels, 0-4 (1)"
    )
    p_render.add_argument("--spp", type=int, help="path: samples (paths) per pixel (16)")
    p_render.add_argument(
        "--bounces", type=int, help="path: indirect bounces, 0 = direct light only (4)"
    )
    p_render.add_argument("--seed", type=int, help="path: random seed (0)")
    p_render.add_argument(
        "--light-samples", type=int, help="path: shadow rays per light per vertex (1)"
    )
    p_render.add_argument(
        "--russian-roulette", type=int, help="path: bounce from which dim paths may stop (3)"
    )
    p_render.add_argument(
        "--clamp", type=float, help="path: clamp indirect radiance per sample (0 = off)"
    )
    p_render.add_argument(
        "--workers",
        type=int,
        help="path: CPU processes, 0 = one per CPU for big renders (default), 1 = this process",
    )
    p_render.add_argument(
        "--ao", action="store_true", help="path: compute the ambient occlusion pass"
    )
    p_render.add_argument(
        "--ao-only",
        action="store_true",
        help="path: fast preview rendering only ambient occlusion (implies --ao)",
    )
    p_render.add_argument("--ao-samples", type=int, help="path: AO rays per camera sample (4)")
    p_render.add_argument("--ao-distance", type=float, help="path: AO reach in scene units (1)")
    p_render.add_argument("--ao-strength", type=float, help="path: AO strength in [0, 1] (1)")
    p_render.add_argument(
        "--passes",
        metavar="DIR",
        help="path: also write every render pass (beauty, direct, indirect, albedo, normal, "
        "depth, sample_count, variance, ao) to DIR as .pfm float images",
    )
    p_render.set_defaults(handler=cmd_render)

    p_gpu = sub.add_parser("gpu", help="GPU rendering: adapters, capabilities and a self-test")
    gpu = p_gpu.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = gpu.add_parser(
        "info", parents=[common], help="list graphics adapters and what the GPU backend can use"
    )
    p.add_argument("--adapter", help="adapter to select (index, name part, discrete, vulkan, ...)")
    p.add_argument(
        "--check",
        action="store_true",
        help="render a test scene on the GPU and verify it against the CPU renderer "
        "(exit code 3 if GPU rendering does not work)",
    )
    p.set_defaults(handler=cmd_gpu_info)

    def placement(p, *, target: bool = True) -> None:
        p.add_argument("--position", **vec, help="local position")
        p.add_argument("--rotation-euler", **vec, help="local XYZ Euler rotation (degrees)")
        if target:
            p.add_argument("--look-at", **vec, help="aim the front (-Y) at this world point")
            p.add_argument("--look-at-object", help="aim the front (-Y) at this object")
            p.add_argument("--up", **vec, help="up direction when aiming (default 0 0 1)")

    def camera_params(p) -> None:
        p.add_argument("--projection", choices=["perspective", "orthographic"])
        p.add_argument("--fov", type=float, help="field of view in degrees (perspective)")
        p.add_argument("--fov-axis", choices=["vertical", "horizontal", "auto"])
        p.add_argument("--ortho-scale", type=float, help="view size in units (orthographic)")
        p.add_argument("--near", type=float, help="near clipping distance")
        p.add_argument("--far", type=float, help="far clipping distance")
        p.add_argument("--aspect-ratio", help="preferred frame width/height, or 'none'")

    p_cam = sub.add_parser("camera", help="add, change, list and frame cameras")
    cam = p_cam.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = cam.add_parser("add", parents=[common, writing], help="add a camera")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--name", help="object name (default: Camera, Camera.001, ...)")
    p.add_argument("--parent", help="parent object id or name")
    camera_params(p)
    placement(p)
    p.add_argument(
        "--active",
        dest="active",
        action="store_const",
        const=True,
        default=None,
        help="make it the active camera (default: if there is none)",
    )
    p.add_argument(
        "--inactive",
        dest="active",
        action="store_const",
        const=False,
        help="do not make it the active camera",
    )
    p.add_argument("--frame", action="store_true", help="frame it on all visible geometry")
    p.add_argument("--frame-direction", **vec, help="view direction when framing")
    p.add_argument("--margin", type=float, default=0.05, help="free border when framing")
    p.set_defaults(handler=cmd_camera_add)

    p = cam.add_parser("set", parents=[common, writing], help="change a camera")
    p.add_argument("path", help="the .aether project")
    p.add_argument("camera", help="camera object id or name")
    camera_params(p)
    placement(p)
    p.add_argument("--activate", action="store_true", help="make it the active camera")
    p.set_defaults(handler=cmd_camera_set)

    p = cam.add_parser("frame", parents=[common, writing], help="frame a camera on geometry")
    p.add_argument("path", help="the .aether project")
    p.add_argument("camera", nargs="?", help="camera id or name (default: the active camera)")
    p.add_argument("--object", action="append", help="frame this object (repeatable; default all)")
    p.add_argument("--direction", **vec, help="new view direction (default: keep)")
    p.add_argument("--margin", type=float, default=0.05, help="free border fraction")
    p.add_argument("--no-fit-clipping", action="store_true", help="keep near/far unchanged")
    p.set_defaults(handler=cmd_camera_frame)

    p = cam.add_parser("activate", parents=[common, writing], help="choose the active camera")
    p.add_argument("path", help="the .aether project")
    p.add_argument("camera", help="camera id or name, or 'none'")
    p.set_defaults(handler=cmd_camera_activate)

    p = cam.add_parser("list", parents=[common, lossy], help="list cameras")
    p.add_argument("path", help="a .aether project")
    p.set_defaults(handler=cmd_camera_list)

    p_light = sub.add_parser("light", help="add, change and list lights")
    lights = p_light.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")

    def light_params(p) -> None:
        p.add_argument("--color", help="light color (#rrggbb)")
        p.add_argument("--intensity", type=float, help="unitless intensity (>= 0)")
        p.add_argument("--range", help="point lights: fade-out distance, or 'none'")
        p.add_argument("--falloff", choices=["inverse_square", "none"], help="point light falloff")
        p.add_argument("--direction", **vec, help="directional lights: travel direction")
        p.add_argument(
            "--radius", type=float, help="point lights: sphere radius (path tracer soft shadows)"
        )
        p.add_argument(
            "--angle",
            type=float,
            help="directional lights: angular diameter in degrees (path tracer soft shadows)",
        )

    p = lights.add_parser("add", parents=[common, writing], help="add a light")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--type", choices=["directional", "point"], default="directional")
    p.add_argument("--name", help="object name (default: Light, Light.001, ...)")
    p.add_argument("--parent", help="parent object id or name")
    light_params(p)
    placement(p)
    p.set_defaults(handler=cmd_light_add)

    p = lights.add_parser("set", parents=[common, writing], help="change a light")
    p.add_argument("path", help="the .aether project")
    p.add_argument("light", help="light object id or name")
    p.add_argument("--type", choices=["directional", "point"])
    light_params(p)
    placement(p)
    p.set_defaults(handler=cmd_light_set)

    p = lights.add_parser("list", parents=[common, lossy], help="list lights and ambient light")
    p.add_argument("path", help="a .aether project")
    p.set_defaults(handler=cmd_light_list)

    p_obj = sub.add_parser("object", help="change an object's placement, color and visibility")
    objs = p_obj.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = objs.add_parser("set", parents=[common, writing], help="change an object")
    p.add_argument("path", help="the .aether project")
    p.add_argument("object", help="object id or name")
    placement(p)
    p.add_argument("--scale", **vec, help="local scale")
    p.add_argument("--color", help="display color (#rrggbb)")
    p.add_argument("--specular", type=float, help="highlight strength in [0, 1]")
    p.add_argument("--shininess", type=float, help="highlight exponent in [1, 10000]")
    p.add_argument("--visible", type=_bool_arg, help="true or false")
    p.set_defaults(handler=cmd_object_set)
    p = objs.add_parser("remove", parents=[common, writing], help="remove an object and children")
    p.add_argument("path", help="the .aether project")
    p.add_argument("object", help="object id or name")
    p.set_defaults(handler=cmd_object_remove)

    p_scene = sub.add_parser("scene", help="world, render settings and scene scripts")
    scenes = p_scene.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = scenes.add_parser("set", parents=[common, writing], help="world and render settings")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--background", help="background color (#rrggbb)")
    p.add_argument("--ambient-color", help="ambient light color (#rrggbb)")
    p.add_argument("--ambient-intensity", type=float, help="ambient light strength (>= 0)")
    p.add_argument(
        "--environment",
        choices=["ambient", "uniform", "sky"],
        help="path tracer environment: ambient (the ambient light), uniform or sky",
    )
    p.add_argument("--environment-color", help="uniform environment color (#rrggbb)")
    p.add_argument(
        "--environment-intensity", type=float, help="uniform/sky environment strength (>= 0)"
    )
    p.add_argument("--sky-zenith", help="sky color straight up (#rrggbb)")
    p.add_argument("--sky-horizon", help="sky color at the horizon (#rrggbb)")
    p.add_argument("--sky-ground", help="color below the horizon (#rrggbb)")
    p.add_argument("--width", type=int, help="default image width")
    p.add_argument("--height", type=int, help="default image height")
    p.add_argument("--samples", type=int, help="default antialiasing (1-4)")
    p.add_argument("--exposure", type=float, help="default exposure (EV)")
    p.add_argument("--cull", choices=["none", "back", "front"], help="default face culling")
    p.add_argument("--default-lights", type=_bool_arg, help="true or false")
    p.add_argument("--active-camera", help="camera id or name, or 'none'")
    p.set_defaults(handler=cmd_scene_set)
    p = scenes.add_parser(
        "run", parents=[common, writing], help="apply a JSON scene-setup script (atomically)"
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("script", help="scene script (.json), or - to read it from stdin")
    p.set_defaults(handler=cmd_scene_run)
    p = scenes.add_parser(
        "merge",
        parents=[common, writing, lossy],
        help="copy the objects of another project or a glTF file into the project",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("source", help="a .aether project or a .glb/.gltf file")
    p.add_argument("--parent", help="object (id or name) to place the copies under")
    p.set_defaults(handler=cmd_scene_merge)

    p_history = sub.add_parser(
        "history", help="undo and redo changes to a project (its saved history)"
    )
    hist = p_history.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = hist.add_parser("show", parents=[common], help="list the steps and checkpoints")
    p.add_argument("path", help="the .aether project")
    p.set_defaults(handler=cmd_history_show)
    p = hist.add_parser(
        "start", parents=[common], help="start recording changes of a project (a history file)"
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("--force", action="store_true", help="replace an existing history file")
    p.set_defaults(handler=cmd_history_start, dry_run=False)
    for name, handler, what in (
        ("undo", cmd_history_undo, "undo the last change(s)"),
        ("redo", cmd_history_redo, "redo the last undone change(s)"),
    ):
        p = hist.add_parser(name, parents=[common], help=what)
        p.add_argument("path", help="the .aether project")
        p.add_argument("-n", "--steps", type=int, default=1, help="how many steps (default 1)")
        p.add_argument("--dry-run", action="store_true", help="report, but write nothing")
        p.set_defaults(handler=handler)
    p = hist.add_parser("checkpoint", parents=[common], help="name the current state")
    p.add_argument("path", help="the .aether project")
    p.add_argument("name", help="checkpoint name")
    p.set_defaults(handler=cmd_history_checkpoint, dry_run=False)
    p = hist.add_parser("goto", parents=[common], help="undo or redo to a checkpoint")
    p.add_argument("path", help="the .aether project")
    p.add_argument("name", help="checkpoint name")
    p.add_argument("--dry-run", action="store_true", help="report, but write nothing")
    p.set_defaults(handler=cmd_history_goto)
    p = hist.add_parser("clear", parents=[common], help="forget every step (keeps recording)")
    p.add_argument("path", help="the .aether project")
    p.set_defaults(handler=cmd_history_clear, dry_run=False)
    p = hist.add_parser(
        "verify", parents=[common], help="replay the history on a copy and check it is exact"
    )
    p.add_argument("path", help="the .aether project")
    p.set_defaults(handler=cmd_history_verify)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command
    subcommand = getattr(args, "procedural_command", None) or getattr(args, "subcommand", None)
    if subcommand:
        command = f"{command} {subcommand}"
    # Never crash on non-ASCII scene/object names when stdout is a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    _history_report.clear()
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
    if _history_report and "history" not in result:
        result = {**result, "history": dict(_history_report)}
        note = _history_report["note"]
        text += f"\nhistory: {_history_report['steps']} step(s) in {_history_report['file']}"
        if note:
            text += f" ({note})"
    if args.json:
        print(json.dumps({"ok": True, "command": command, **result}, indent=2))
    else:
        print(text)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
