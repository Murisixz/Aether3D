"""The tools of the Aether3D MCP server: schemas, descriptions and implementations.

Twenty tools cover the engine. Building and editing go through one tool,
``execute_scene_plan``: a whole scene (thousands of objects) is one call, one atomic step
of the history. The others open and save projects, inspect the scene compactly, run the
Stage 10 spatial queries, render previews and images, use the history, export, run the
existing JSON scripts (mesh editing, procedural, scene setup), report diagnostics and
describe the engine's capabilities; the Stage 12 tools (:mod:`.feedback`) observe the
rendered scene, diagnose it against expectations, and propose, apply and verify
corrections, alone or in a loop.

Tool implementations are plain synchronous functions ``(session, arguments, context) ->
ToolOutput``; :mod:`.server` runs them in a worker thread, one at a time, with
cancellation and progress. Arguments are checked against the tool's JSON Schema before a
tool runs, then again by the engine.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..errors import AetherError, PlanError
from .session import ServerConfig, Session, ToolError
from .workspace import (
    EXPORT_SUFFIXES,
    IMAGE_SUFFIXES,
    PROJECT_SUFFIXES,
    SCENE_SUFFIXES,
    redact_paths,
)

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


@dataclass
class ToolContext:
    """What a running tool may use: cancellation, progress and its deadline."""

    cancel: threading.Event = field(default_factory=threading.Event)
    progress: Callable[[float, float | None, str], None] | None = None
    deadline: float | None = None

    def remaining(self) -> float | None:
        return None if self.deadline is None else max(0.001, self.deadline - time.monotonic())


@dataclass
class ToolOutput:
    data: dict[str, Any]
    image: bytes | None = None  # a PNG, sent as image content
    images: list[bytes] = field(default_factory=list)  # more PNGs, in order, after it


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[Session, dict[str, Any], ToolContext], ToolOutput]
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False


# --------------------------------------------------------------------------- #
# Schema fragments
# --------------------------------------------------------------------------- #

PATH = {
    "type": "string",
    "minLength": 1,
    "maxLength": 1024,
    "description": "a path inside the workspace, relative to its root (no '..')",
}
OVERWRITE = {
    "type": "boolean",
    "default": False,
    "description": "replace an existing file (never done without it)",
}
SELECTOR = {
    "description": "objects: an id or name, a list of them, or {ids, names, tags (all), "
    "any_tags, name_glob, kind, under (an ancestor), visible, material, expect (count or "
    "{min, max})}",
    "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}},
                "names": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "any_tags": {"type": "array", "items": {"type": "string"}},
                "name_glob": {"type": "string"},
                "kind": {"enum": ["mesh", "procedural", "camera", "light", "empty"]},
                "under": {"type": "string"},
                "visible": {"type": "boolean"},
                "material": {"type": "string"},
                "expect": {},
            },
            "additionalProperties": False,
        },
    ],
}


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _section(description: str) -> dict[str, Any]:
    return {"type": "object", "description": description}


PLAN = {
    "type": "object",
    "description": "a scene plan (format aether3d.scene-plan, version 1). Full JSON Schema: "
    "resource aether3d://schema/scene-plan; guide with examples: aether3d://guide",
    "properties": {
        "format": {"const": "aether3d.scene-plan"},
        "version": {"const": 1},
        "label": {"type": "string", "description": "names the undo step"},
        "description": {"type": "string"},
        "expected_revision": {
            "type": ["integer", "null"],
            "description": "refuse unless the scene is at this revision",
        },
        "idempotency_key": {
            "type": ["string", "null"],
            "description": "resending the same plan with this key applies it once",
        },
        "options": _section(
            "on_name_conflict (error|rename), tags (added to every new object), result_ids "
            "(auto|all|none), chunk_size (opt-in steps for huge plans, needs idempotency_key), "
            "on_chunk_failure (keep|rollback), timeout (s)"
        ),
        "parameters": _section('plan constants for {"expr": ...} values: {name: value}'),
        "procedural": _section("procedural scene parameters: {parameters: {...}, set: {...}}"),
        "meshes": _section(
            'shared meshes by symbol: {"primitive": cube|box|sphere|plane|cylinder|cone|torus|'
            'grid|uv_sphere, ...size/radius/segments...} or {"data": {positions, triangles}}'
        ),
        "textures": _section('{symbol: {"procedural": graph} or {"image": "file.png"}}'),
        "materials": _section(
            '{symbol: {"preset": stone|brick|wood|metal|grass|fabric, ...} or '
            '{"base_color": "#a03020", "roughness": 0.7, "metallic": 0, ...}}'
        ),
        "geometries": _section(
            "procedural nodes: {symbol: {primitive, params, parameters, modifiers: [{type, "
            "params}]}} (modifiers: transform, mirror, array, solidify, subdivision, bevel, "
            "boolean)"
        ),
        "prefabs": _section(
            "reusable structures: {symbol: {children: {...}, mesh?, tags?}}; objects use "
            '"prefab": "@symbol"'
        ),
        "objects": _section(
            "objects by symbol: {mesh|geometry|prefab|camera|light, name, parent, position, "
            "rotation_euler (degrees), scale, material, color, visible, tags, metadata, "
            "look_at, children: {...}} plus at most one pattern: grid {count: [nx, ny, nz?], "
            "spacing, center}, array {count, offset}, circle {count, radius, orient}, line "
            "{from, to, count}, scatter {count, box|circle, seed, min_distance}, instances [...],"
            " repeat N; vary {field: {cycle|choose|uniform|linear|values|expr}}"
        ),
        "operations": {
            "type": "array",
            "description": "ordered changes after building: {op: update, select, set: {position, "
            "translate, rotation_euler, scale, scale_by, color, material, visible, tags, "
            "add_tags, remove_tags, metadata, parent, name}}, {op: delete, select, expect}, "
            "{op: set_parameters, values}, {op: frame_camera}, {op: set_active_camera}",
            "items": {"type": "object"},
        },
        "world": {"type": ["object", "null"], "description": "background, ambient light"},
        "render_settings": {"type": ["object", "null"], "description": "width, height, ..."},
        "active_camera": {"type": ["string", "object", "null"]},
        "queries": {
            "type": "array",
            "description": 'spatial queries answered after building ({"query": "overlaps", '
            '"objects": "@walls"}); "@symbol" becomes object ids',
            "items": {"type": "object"},
        },
    },
    "required": ["format", "version"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ref(obj: Any) -> dict[str, str]:
    return {"id": obj.id, "name": obj.name}


def _round(values: Any, digits: int = 6) -> Any:
    if isinstance(values, float):
        return round(values, digits)
    if isinstance(values, (list, tuple)):
        return [_round(v, digits) for v in values]
    return values


def scene_status(session: Session) -> dict[str, Any]:
    scene = session.scene
    if scene is None:
        return {"open": False}
    return {
        "open": True,
        "name": scene.name,
        "project": None if session.project is None else session.workspace.relative(session.project),
        "revision": scene.revision,
        "dirty": session.dirty,
        "objects": len(scene),
    }


def summarize(session: Session, detail: str = "summary") -> dict[str, Any]:
    """A compact description of the open scene (what an agent needs to orient itself)."""
    scene = session.require_scene()
    kinds = Counter(o.kind for o in scene)
    tags = Counter(t for o in scene for t in o.tags)
    info: dict[str, Any] = {
        **scene_status(session),
        "source": session.source,
        "counts": {
            "objects": len(scene),
            "by_kind": dict(sorted(kinds.items())),
            "meshes": len({id(o._mesh) for o in scene if o._mesh is not None}),
            "materials": len(scene.materials.materials),
            "textures": len(scene.materials.textures),
            "geometries": 0 if scene._procedural is None else len(scene.procedural.geometries),
        },
        "tags": dict(tags.most_common(30)),
        "roots": [
            {**_ref(o), "kind": o.kind, "descendants": sum(1 for _ in o.iter_descendants())}
            for o in scene.roots[:30]
        ],
        "cameras": [{**_ref(o), "active": scene.active_camera is o} for o in scene.cameras[:20]],
        "lights": [{**_ref(o), "type": o.light.type} for o in scene.lights[:20]],
        "materials": [m.name for m in scene.materials.materials[:50]],
        "history": {
            "steps": len(scene.history),
            "can_undo": scene.history.can_undo,
            "can_redo": scene.history.can_redo,
            "last": [e.label for e in scene.history.entries[-5:]],
            "checkpoints": sorted(scene.history.checkpoints),
        },
    }
    if len(scene.roots) > 30:
        info["roots_omitted"] = len(scene.roots) - 30
    bounds = None
    try:
        b = scene.bounds()
        if b is not None:
            bounds = {"min": _round(list(b.aabb.min)), "max": _round(list(b.aabb.max))}
    except AetherError as exc:
        bounds = {"error": getattr(exc, "message", str(exc))}
    info["bounds"] = bounds
    if detail == "full":
        info["world"] = scene.world.to_dict()
        info["render_settings"] = scene.render_settings.to_dict()
        if scene._procedural is not None:
            info["procedural_parameters"] = {
                name: _jsonable(p.value) for name, p in scene.procedural.parameters.items()
            }
        info["gpu"] = session.gpu()
    return info


def _jsonable(value: Any) -> Any:
    from ..procedural.params import Expr

    if isinstance(value, Expr):
        return {"expr": value.source}
    if isinstance(value, tuple):
        return list(value)
    return value


OBJECT_FIELDS = (
    "id",
    "name",
    "kind",
    "parent",
    "children",
    "position",
    "rotation_euler",
    "scale",
    "world_position",
    "bounds",
    "mesh",
    "geometry",
    "material",
    "color",
    "visible",
    "tags",
    "metadata",
    "camera",
    "light",
)
DEFAULT_FIELDS = ("id", "name", "kind", "parent", "position", "tags")


def object_fields(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields:
        if f == "id":
            out["id"] = obj.id
        elif f == "name":
            out["name"] = obj.name
        elif f == "kind":
            out["kind"] = obj.kind
        elif f == "parent":
            out["parent"] = None if obj.parent is None else obj.parent.name
        elif f == "children":
            out["children"] = len(obj.children)
        elif f in ("position", "rotation_euler", "scale"):
            out[f] = _round(list(getattr(obj, f)))
        elif f == "world_position":
            out[f] = _round(list(obj.world_position))
        elif f == "bounds":
            try:
                b = obj.bounds()
            except AetherError:
                b = None
            out[f] = (
                None
                if b is None
                else {"min": _round(list(b.aabb.min)), "max": _round(list(b.aabb.max))}
            )
        elif f == "mesh":
            mesh = obj._mesh if obj._geometry is None else None
            out[f] = (
                None
                if mesh is None
                else {"triangles": mesh.triangle_count, "vertices": mesh.vertex_count}
            )
        elif f == "geometry":
            out[f] = None if obj.geometry is None else obj.geometry.name
        elif f == "material":
            out[f] = None if obj.material is None else obj.material.name
        elif f == "color":
            out[f] = None if obj.material is not None else _round(list(obj.color), 4)
        elif f == "visible":
            out[f] = obj.visible
        elif f == "tags":
            out[f] = list(obj.tags)
        elif f == "metadata":
            out[f] = obj.metadata
        elif f == "camera":
            out[f] = None if obj.camera is None else obj.camera.to_dict()
        elif f == "light":
            out[f] = None if obj.light is None else obj.light.to_dict()
    return out


def _engine_error(exc: AetherError, session: Session) -> dict[str, Any]:
    data: dict[str, Any] = {"type": type(exc).__name__}
    if hasattr(exc, "to_dict"):
        try:
            data.update(exc.to_dict())
        except Exception:  # pragma: no cover - defensive
            data["message"] = str(exc)
    else:
        data["message"] = str(exc)
    data.setdefault("code", getattr(exc, "code", type(exc).__name__))
    data.setdefault("message", str(exc))
    return _redacted(data, session)


def _redacted(data: Any, session: Session) -> Any:
    """``data`` as JSON values, with every string redacted (``Workspace.redact``).

    Strings are redacted before they are encoded: in JSON text a Windows path's
    backslashes are doubled, and would no longer match the workspace's path.
    """
    import json

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return session.workspace.redact(value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return walk(json.loads(json.dumps(data, default=str)))


# --------------------------------------------------------------------------- #
# Tools: scenes and projects
# --------------------------------------------------------------------------- #


def scene_new(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..scene import Scene

    session.check_unsaved(args.get("discard_unsaved", False))
    project = None
    if args.get("path") is not None:
        project = session.workspace.path(
            args["path"], suffixes=PROJECT_SUFFIXES, write=True, overwrite=True
        )
        if project.exists():
            raise ToolError(
                "FILE_EXISTS",
                f"{session.workspace.relative(project)!r} exists",
                hint="open it with scene_open to change it, or choose another path",
            )
    scene = Scene(args.get("name") or (project.stem if project is not None else "Untitled Scene"))
    session.bind(scene, project, None, saved=False)
    return ToolOutput({"scene": scene_status(session)})


def scene_open(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..scene import Scene

    session.check_unsaved(args.get("discard_unsaved", False))
    path = session.workspace.path(args["path"], suffixes=SCENE_SUFFIXES)
    notes: list[str] = []
    if path.suffix.lower() in (".glb", ".gltf"):
        from ..gltf import import_gltf

        result = import_gltf(path, allow_lossy=args.get("allow_lossy", False))
        scene = result.scene
        notes += [f"{d.code}: {d.message}" for d in result.diagnostics[:20]]
        session.bind(scene, None, session.workspace.relative(path), saved=False)
        notes.append('imported: save it as a project with scene_save {"path": "name.aether"}')
    else:
        scene = Scene.load(path)
        if args.get("history", True):
            from .._history_file import history_path
            from ..errors import HistoryError

            file = history_path(path)
            if file.exists():
                try:
                    scene.history.load(file, project_dir=path.parent)
                    notes.append(f"history: {len(scene.history)} undo step(s) from {file.name}")
                except HistoryError as exc:
                    notes.append(f"the saved history was not used ([{exc.code}] {exc.message})")
        session.bind(scene, path, session.workspace.relative(path), saved=True)
    data = {"scene": scene_status(session), "notes": notes}
    data["summary"] = summarize(session)
    return ToolOutput(data)


def scene_save(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from .._history_file import history_path
    from ..serialization import save_scene

    scene = session.require_scene()
    overwrite = args.get("overwrite", False)
    if args.get("path") is not None:
        target = session.workspace.path(
            args["path"], suffixes=PROJECT_SUFFIXES, write=True, overwrite=True
        )
        same = session.project is not None and target == session.project
        if target.exists() and not overwrite and not same:
            raise ToolError(
                "FILE_EXISTS",
                f"{session.workspace.relative(target)!r} exists",
                hint='pass "overwrite": true to replace it',
            )
    elif session.project is not None:
        target = session.workspace.path(
            str(session.project), suffixes=PROJECT_SUFFIXES, write=True, overwrite=True
        )
    else:
        raise ToolError(
            "NO_PROJECT_PATH",
            "the scene has no project file yet",
            hint='give "path": "name.aether"',
        )
    known = target == session.project and session.disk is not None
    if target.exists() and not overwrite and not known:
        raise ToolError(
            "FILE_EXISTS",
            f"{session.workspace.relative(target)!r} exists and this session did not open it",
            hint='pass "overwrite": true to replace it',
        )
    if known and session.project_changed_on_disk() and not overwrite:
        raise ToolError(
            "PROJECT_CHANGED_ON_DISK",
            f"{session.workspace.relative(target)!r} changed on disk since this session read or "
            "wrote it",
            hint='reopen it (scene_open) to see the changes, or pass "overwrite": true to '
            "replace them",
        )
    history_file = history_path(target)
    keep_history = args.get("history", False) or history_file.exists()
    save_scene(scene, target)
    if keep_history:
        scene.history.save(history_file, project_dir=target.parent)
    session.mark_saved(target)
    return ToolOutput(
        {
            "scene": scene_status(session),
            "saved": session.workspace.relative(target),
            "history_file": session.workspace.relative(history_file) if keep_history else None,
        }
    )


def scene_info(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    return ToolOutput(summarize(session, args.get("detail", "summary")))


def objects_list(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..plans import select_objects

    scene = session.require_scene()
    fields = tuple(args.get("fields") or DEFAULT_FIELDS)
    unknown = sorted(set(fields) - set(OBJECT_FIELDS))
    if unknown:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"unknown field(s): {', '.join(unknown)}",
            hint=", ".join(OBJECT_FIELDS),
        )
    objects = list(scene) if args.get("select") is None else select_objects(scene, args["select"])
    limit = args.get("limit", 100)
    offset = 0
    cursor = args.get("cursor")
    if cursor is not None:
        try:
            revision, offset = (int(part) for part in cursor.split(":"))
        except ValueError:
            raise ToolError("INVALID_ARGUMENT", "invalid cursor") from None
        if revision != scene.revision:
            raise ToolError(
                "STALE_CURSOR",
                "the scene changed since this page was listed",
                hint="list again without a cursor",
            )
    page = objects[offset : offset + limit]
    end = offset + len(page)
    return ToolOutput(
        {
            "total": len(objects),
            "offset": offset,
            "objects": [object_fields(o, fields) for o in page],
            "next_cursor": f"{scene.revision}:{end}" if end < len(objects) else None,
            "revision": scene.revision,
        }
    )


# --------------------------------------------------------------------------- #
# Tools: plans
# --------------------------------------------------------------------------- #


def _plan_progress(ctx: ToolContext) -> Callable[[int, int, str], None] | None:
    if ctx.progress is None:
        return None
    report = ctx.progress

    def progress(done: int, total: int, message: str) -> None:
        report(float(done), float(total), message)

    return progress


def execute_scene_plan(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..plans import execute_plan

    scene = session.require_scene()
    mode = args.get("mode", "execute")
    timeout = ctx.remaining()
    if args.get("timeout") is not None:
        timeout = min(args["timeout"], timeout) if timeout is not None else args["timeout"]
    result = execute_plan(
        scene,
        args["plan"],
        mode=mode,
        expected_revision=args.get("expected_revision"),
        limits=session.config.plan_limits,
        base_dir=session.base_dir(),
        progress=_plan_progress(ctx),
        cancel=ctx.cancel,
        timeout=timeout,
    )
    data = _bounded_plan_result(result.to_dict(), args.get("max_results", 100))
    data["scene"] = scene_status(session)
    return ToolOutput(data)


def check_scene_plan(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..plans import execute_plan, validate_plan

    if args.get("dry_run", False):
        scene = session.require_scene()
        result = execute_plan(
            scene,
            args["plan"],
            mode="dry_run",
            limits=session.config.plan_limits,
            base_dir=session.base_dir(),
            progress=_plan_progress(ctx),
            cancel=ctx.cancel,
            timeout=ctx.remaining(),
        )
    else:
        result = validate_plan(
            args["plan"],
            session.scene,
            limits=session.config.plan_limits,
            base_dir=session.base_dir(),
        )
    return ToolOutput(_bounded_plan_result(result.to_dict(), args.get("max_results", 100)))


# --------------------------------------------------------------------------- #
# Tools: queries, rendering, history, files, scripts
# --------------------------------------------------------------------------- #


def spatial_query(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..plans import with_selections
    from ..spatial import run_query

    scene = session.require_scene()
    if ("request" in args) == ("requests" in args):
        raise ToolError("INVALID_ARGUMENT", 'give "request" (one query) or "requests" (a list)')
    requests = [args["request"]] if "request" in args else args["requests"]
    results = []
    for k, request in enumerate(requests):
        if ctx.cancel.is_set():
            raise ToolError("CANCELLED", "cancelled")
        try:
            where = "/request" if "request" in args else f"/requests/{k}"
            answer = run_query(scene, with_selections(scene, request, where))
            results.append(bounded_answer(answer, args.get("max_results", 100)))
        except AetherError as exc:
            results.append(
                {"query": request.get("query"), "index": k, "error": _engine_error(exc, session)}
            )
    data: dict[str, Any] = {"revision": scene.revision}
    if "request" in args:
        data.update(results[0])
    else:
        data["results"] = results
    return ToolOutput(data)


def _render_report(session: Session, result: Any, framed: bool) -> dict[str, Any]:
    backend = result.backend
    name = backend.get("name")
    report: dict[str, Any] = {
        "backend": name,
        "device": "gpu" if name in ("gpu", "gpu_path") else "cpu",
        "width": result.width,
        "height": result.height,
        "status": result.status,
        "framed": framed,
        "seconds": round(result.stats.get("timings", {}).get("total", 0.0), 4)
        if isinstance(result.stats.get("timings"), dict)
        else None,
        "diagnostics": [
            {"code": d.code, "severity": d.severity, "message": session.workspace.redact(d.message)}
            for d in result.diagnostics[:20]
        ],
    }
    if name in ("gpu", "gpu_path"):
        adapter = backend.get("adapter", {})
        report["adapter"] = {
            "name": adapter.get("name"),
            "backend": adapter.get("backend"),
            "software": adapter.get("software"),
        }
    if "draw_calls" in result.stats:
        report["draw_calls"] = result.stats["draw_calls"]
    fallback = [d for d in result.diagnostics if d.code == "BACKEND_FALLBACK"]
    if fallback:
        report["fallback"] = fallback[0].message
    visible = [o for o in result.objects if o.get("visible_samples", 0) > 0]
    report["visible_objects"] = len(visible)
    return report


def _camera_options(scene: Any, args: dict) -> tuple[dict[str, Any], bool]:
    options: dict[str, Any] = {}
    framed = False
    if args.get("camera") is not None:
        options["camera"] = args["camera"]
    elif args.get("frame", False) or scene.active_camera is None:
        options["frame"] = True
        framed = True
    return options, framed


def render_preview(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    scene = session.require_scene()
    width, height = args.get("width", 640), args.get("height", 360)
    if width * height > session.config.max_preview_pixels:
        raise ToolError(
            "LIMIT_EXCEEDED",
            f"previews have at most {session.config.max_preview_pixels} pixels",
            hint="use render_image for large images",
        )
    backend = args.get("backend", "auto")
    options, framed = _camera_options(scene, args)
    if args.get("shadows") and backend != "cpu":
        options["shadows"] = True
    if backend != "cpu" and session.config.gpu_adapter is not None:
        options["adapter"] = session.config.gpu_adapter
    # A preview needs the image only (on the GPU, the other outputs are not read back)
    # and favors encoding speed: zlib level 1, a few percent larger than the default.
    result = scene.render(
        width=width, height=height, backend=backend, outputs=("image",), **options
    )
    png = result.png_bytes(compression=1)
    session.last_preview = png
    report = _render_report(session, result, framed)
    if args.get("save_to") is not None:
        target = session.workspace.path(
            args["save_to"], suffixes=(".png",), write=True, overwrite=args.get("overwrite", False)
        )
        result.save(target)
        report["saved"] = session.workspace.relative(target)
    report["revision"] = scene.revision
    return ToolOutput(report, image=png)


def render_image(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    scene = session.require_scene()
    config = session.config
    target = session.workspace.path(
        args["path"], suffixes=IMAGE_SUFFIXES, write=True, overwrite=args.get("overwrite", False)
    )
    width, height = args.get("width", 1280), args.get("height", 720)
    if width * height > config.max_render_pixels:
        raise ToolError("LIMIT_EXCEEDED", f"images have at most {config.max_render_pixels} pixels")
    backend = args.get("backend", "auto")
    options, framed = _camera_options(scene, args)
    if backend in ("path", "gpu_path"):
        if not config.allow_path_tracing:
            raise ToolError("NOT_ALLOWED", "this server does not path trace (--no-path-tracing)")
        spp = args.get("spp", 16)
        if width * height * spp > config.max_path_samples:
            raise ToolError(
                "LIMIT_EXCEEDED",
                f"{width}x{height} at {spp} samples per pixel is {width * height * spp} samples; "
                f"this server allows {config.max_path_samples}",
                hint="lower the size or spp, or render with the GPU rasterizer",
            )
        options["path"] = {"samples": spp, "bounces": args.get("bounces", 4)}

        def stop(done: int, total: int, seconds: float) -> bool:
            if ctx.progress is not None:
                ctx.progress(float(done), float(total), f"{done}/{total} samples")
            if ctx.cancel.is_set():
                return False
            return ctx.deadline is None or time.monotonic() < ctx.deadline

        options["progress"] = stop
    else:
        if args.get("samples") is not None:
            options["samples"] = args["samples"]
        if args.get("shadows") and backend != "cpu":
            options["shadows"] = True
    if backend not in ("cpu", "path") and config.gpu_adapter is not None:
        options["adapter"] = config.gpu_adapter
    hdr = target.suffix.lower() in (".hdr", ".pfm")  # linear light; a PNG needs the image
    outputs = ("image", "linear") if hdr else ("image",)
    result = scene.render(width=width, height=height, backend=backend, outputs=outputs, **options)
    if ctx.cancel.is_set():
        raise ToolError("CANCELLED", "the render was cancelled; nothing was written")
    result.save(target)
    report = _render_report(session, result, framed)
    report["saved"] = session.workspace.relative(target)
    report["revision"] = scene.revision
    image = None
    if args.get("include_image") and target.suffix.lower() == ".png":
        image = result.png_bytes(compression=1)
    return ToolOutput(report, image=image)


def history(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    scene = session.require_scene()
    h = scene.history
    action = args["action"]
    moved = []
    if action == "undo":
        moved = h.undo(args.get("steps", 1))
    elif action == "redo":
        moved = h.redo(args.get("steps", 1))
    elif action == "goto":
        moved = h.goto(_name(args))
    elif action == "checkpoint":
        h.checkpoint(_name(args))
    elif action == "remove_checkpoint":
        h.remove_checkpoint(_name(args))
    data = {
        "action": action,
        "moved": [{"index": e.index, "label": e.label} for e in moved],
        "steps": [{"index": e.index, "label": e.label, "changes": e.changes} for e in h.entries],
        "undone": [{"index": e.index, "label": e.label} for e in h.redo_entries],
        "checkpoints": h.checkpoints,
        "bytes": h.nbytes,
        "scene": scene_status(session),
    }
    return ToolOutput(data)


def _name(args: dict) -> str:
    if not args.get("name"):
        raise ToolError("INVALID_ARGUMENT", f'{args["action"]} needs "name"')
    return args["name"]


def export_scene(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    scene = session.require_scene()
    target = session.workspace.path(
        args["path"], suffixes=EXPORT_SUFFIXES, write=True, overwrite=args.get("overwrite", False)
    )
    result = scene.export(
        target, allow_lossy=args.get("allow_lossy", False), bake_textures=args.get("bake_textures")
    )
    return ToolOutput(
        {
            "path": session.workspace.relative(target),
            "format": result.format,
            "bytes": result.byte_length,
            "nodes": result.node_count,
            "meshes": result.mesh_count,
            "diagnostics": [
                {"code": d.code, "severity": d.severity, "message": d.message}
                for d in result.diagnostics[:30]
            ],
        }
    )


def run_script(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    scene = session.require_scene()
    script = args["script"]
    kind = script.get("format")
    if kind == "aether3d.edit-script":
        from ..editing.script import run_edit_script

        data = run_edit_script(scene, script).to_dict()
    elif kind == "aether3d.procedural-script":
        from ..procedural.script import run_procedural_script

        data = run_procedural_script(scene, script)
        data = data if isinstance(data, dict) else {"result": data}
    elif kind == "aether3d.scene-script":
        from ..rendering.script import run_scene_script

        data = run_scene_script(scene, script, base_dir=session.base_dir())
    elif kind == "aether3d.sculpt-script":
        from ..sculpt.script import run_sculpt_script

        data = run_sculpt_script(scene, script)
    elif kind == "aether3d.texture-script":
        from ..texturing.script import run_texture_script

        data = run_texture_script(scene, script, should_stop=ctx.cancel.is_set)
    else:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"unknown script format {kind!r}",
            hint="aether3d.edit-script, aether3d.procedural-script, aether3d.scene-script, "
            "aether3d.sculpt-script or aether3d.texture-script (objects, materials and "
            "placement: use execute_scene_plan)",
        )
    data = {"format": kind, **data, "scene": scene_status(session)}
    return ToolOutput(data)


MAX_INSPECT_TRIANGLES = 2_000_000
"""Largest mesh inspect_mesh tests for self-intersections."""


def inspect_mesh(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..errors import ObjectNotFoundError
    from ..sculpt.validate import object_report

    scene = session.require_scene()
    ref = args["object"]
    try:
        obj = scene.get(ref)
    except ObjectNotFoundError:
        try:
            obj = scene.get_by_name(ref)
        except ObjectNotFoundError:
            raise ToolError("NOT_FOUND", f"no object with id or name {ref!r}") from None
    if obj.mesh is None:
        raise ToolError("INVALID_ARGUMENT", f"object {obj.name!r} has no mesh")
    tested = bool(args.get("self_intersections", False))
    if tested and obj.mesh.triangle_count > MAX_INSPECT_TRIANGLES:
        raise ToolError(
            "RESOURCE_LIMIT",
            f"the self-intersection test is limited to {MAX_INSPECT_TRIANGLES} triangles",
            hint="inspect without self_intersections, or use aether3d sculpt check",
        )
    report = object_report(obj, self_intersections=tested)
    return ToolOutput({**report, "scene": scene_status(session)})


def _scene_object(scene, ref: str):
    from ..errors import ObjectNotFoundError

    try:
        return scene.get(ref)
    except ObjectNotFoundError:
        try:
            return scene.get_by_name(ref)
        except ObjectNotFoundError:
            raise ToolError("NOT_FOUND", f"no object with id or name {ref!r}") from None


def inspect_uvs(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..rendering.png import encode_png
    from ..uv import layout_image

    scene = session.require_scene()
    obj = _scene_object(scene, args["object"])
    if obj.mesh is None:
        raise ToolError("INVALID_ARGUMENT", f"object {obj.name!r} has no mesh")
    if obj.mesh.uvs is None:
        raise ToolError(
            "NO_UVS",
            f"object {obj.name!r} has no UVs",
            hint="unwrap it with a texture script (op unwrap or project_uvs)",
        )
    report = obj.uv_report(
        resolution=args.get("resolution", 1024), islands_limit=args.get("islands_limit", 16)
    )
    mode = args.get("image", "islands")
    png = None
    if mode != "none":
        texture = None
        if mode == "texture":
            if not args.get("texture"):
                raise ToolError("INVALID_ARGUMENT", 'image "texture" needs "texture"')
            tex = scene.materials.find_texture(args["texture"])
            if tex is None:
                raise ToolError("NOT_FOUND", f"no texture {args['texture']!r}")
            texture = tex.preview(256) if tex.kind == "procedural" else tex.load().image.pixels
        png = encode_png(layout_image(obj.mesh, args.get("size", 512), mode=mode, texture=texture))
    data = {
        "object": {"id": obj.id, "name": obj.name},
        **report,
        "image": None if png is None else f"UV layout ({mode}): v up, island boundaries dark",
        "scene": scene_status(session),
    }
    return ToolOutput(data, image=png)


def inspect_texture(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    import numpy as np

    from ..rendering.png import encode_png

    scene = session.require_scene()
    tex = scene.materials.find_texture(args["texture"])
    if tex is None:
        raise ToolError("NOT_FOUND", f"no texture with id or name {args['texture']!r}")
    info = tex.info()
    users = [
        {"material": m.name, "slot": slot}
        for m in scene.materials.materials
        for slot, binding in m.textures.items()
        if binding is not None and binding.texture == tex.id
    ]
    size = args.get("preview_size", 256)
    png = None
    if size:
        if tex.kind == "procedural":
            pixels = tex.preview(size)
        else:
            pixels = tex.load().image.pixels
            h, w = pixels.shape[:2]
            step = max(1, -(-max(h, w) // size))
            pixels = pixels[::step, ::step]
        png = encode_png(np.ascontiguousarray(pixels))
    return ToolOutput({"texture": info, "users": users, "scene": scene_status(session)}, image=png)


def diagnostics(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    scene = session.require_scene()
    checks = args.get("checks") or ["geometry", "procedural", "materials"]
    out: dict[str, Any] = {}
    if "geometry" in checks:
        report = scene.inspect_geometry()
        findings = report.all_diagnostics
        out["geometry"] = {
            "status": report.status,
            "by_code": dict(Counter(d.code for d in findings)),
            "first": [d.to_dict() for d in findings[:20]],
        }
    if "procedural" in checks and scene._procedural is not None:
        report = scene.procedural.validate()
        out["procedural"] = report
    if "materials" in checks:
        found = scene.materials.validate()
        out["materials"] = {
            "by_code": dict(Counter(d.code for d in found)),
            "first": [d.to_dict() for d in found[:20]],
        }
    if "history" in checks:
        out["history"] = scene.history.verify()
    out["scene"] = scene_status(session)
    return ToolOutput(out)


def capabilities(session: Session, args: dict, ctx: ToolContext) -> ToolOutput:
    from ..plans import plan_schema
    from ..procedural import ProceduralSystem
    from ..spatial.request import QUERIES

    topic = args.get("topic", "overview")
    config: ServerConfig = session.config
    if topic == "overview":
        data = {
            "aether3d": __version__,
            "workspace": ".",
            "read_only": config.read_only,
            "scene": scene_status(session),
            "tools": {spec.name: spec.title for spec in TOOLS},
            "plan_format": {"format": "aether3d.scene-plan", "version": 1},
            "resources": [
                "aether3d://guide",
                "aether3d://schema/scene-plan",
                "aether3d://examples/{name}",
                "aether3d://sculpt/guide",
                "aether3d://schema/sculpt-script",
                "aether3d://texturing/guide",
                "aether3d://schema/texture-script",
            ],
            "conventions": "+Z up, right-handed, meters; objects face -Y; rotation_euler in "
            "degrees (XYZ); colors are sRGB '#rrggbb'",
            "tips": [
                "build and edit with execute_scene_plan: thousands of objects in one call",
                "check placement with spatial_query (overlaps, raycast, relation)",
                "render_preview shows the scene (GPU rasterizer when available)",
            ],
        }
    elif topic == "plan":
        schema = plan_schema()
        data = {
            "schema_resource": "aether3d://schema/scene-plan",
            "sections": list(schema["properties"]),
        }
    elif topic == "primitives":
        types = ProceduralSystem.types()
        data = {
            "mesh_primitives": [
                "cube",
                "box",
                "sphere",
                "plane",
                "cylinder",
                "cone",
                "torus",
                "grid",
                "uv_sphere",
            ],
            "procedural_primitives": {
                p["type"]: [q["name"] for q in p["params"]] for p in types["primitives"]
            },
        }
    elif topic == "modifiers":
        types = ProceduralSystem.types()
        data = {
            "modifiers": {m["type"]: [q["name"] for q in m["params"]] for m in types["modifiers"]},
            "expression_functions": types["expression_functions"],
        }
    elif topic == "materials":
        from ..materials import material_schema, preset_info

        schema = material_schema()
        data = {
            "models": schema["models"],
            "parameters": {p["name"]: p.get("type") for p in schema["parameters"]},
            "slots": [s["name"] for s in schema["slots"]],
            "presets": preset_info(),
        }
    elif topic == "queries":
        data = {
            "queries": {
                name: {"summary": q.summary, "arguments": list(q.arguments)}
                for name, q in QUERIES.items()
            }
        }
    elif topic == "limits":
        data = {
            "plan": config.plan_limits.to_dict(),
            "preview_pixels": config.max_preview_pixels,
            "render_pixels": config.max_render_pixels,
            "path_samples": config.max_path_samples,
            "path_tracing": config.allow_path_tracing,
            "tool_timeout": config.tool_timeout,
        }
    elif topic == "sculpt":
        from ..sculpt.brushes import AUTOMASKS, FALLOFFS, KINDS
        from ..sculpt.script import FORMAT_ID, LIMIT_KEYS, OPERATIONS
        from ..sculpt.validate import TIERS

        data = {
            "script_format": {"format": FORMAT_ID, "version": 1},
            "schema_resource": "aether3d://schema/sculpt-script",
            "guide_resource": "aether3d://sculpt/guide",
            "brushes": list(KINDS),
            "falloffs": list(FALLOFFS),
            "automask": list(AUTOMASKS),
            "operations": list(OPERATIONS),
            "validity_tiers": list(TIERS),
            "limits": list(LIMIT_KEYS),
            "unsupported": {
                "quad remeshing": "not offered: remesh (isotropic triangles) and "
                "voxel_remesh are what exist",
                "multiresolution sculpting": "not offered (no subdivision levels with "
                "displacement)",
            },
            "tips": [
                "brush radius and dyntopo detail are in world units; strokes follow the "
                "surface (project: nearest) unless told otherwise",
                "require a tier (e.g. 'closed') so a broken result is refused, not committed",
                "inspect_mesh measures validity; observe_scene shows the result",
            ],
        }
    elif topic == "texturing":
        from ..materials.painted import BLEND_MODES, ENCODINGS
        from ..texturing import BRUSH_KINDS, MAPS
        from ..texturing.paint import FALLOFFS, SPACES
        from ..texturing.script import FORMAT_ID, LIMIT_KEYS, OPERATIONS
        from ..uv import PROJECTIONS

        data = {
            "script_format": {"format": FORMAT_ID, "version": 1},
            "schema_resource": "aether3d://schema/texture-script",
            "guide_resource": "aether3d://texturing/guide",
            "operations": list(OPERATIONS),
            "projections": list(PROJECTIONS),
            "brushes": list(BRUSH_KINDS),
            "falloffs": list(FALLOFFS),
            "paint_spaces": list(SPACES),
            "blend_modes": list(BLEND_MODES),
            "encodings": list(ENCODINGS),
            "bake_maps": list(MAPS),
            "limits": list(LIMIT_KEYS),
            "tips": [
                "unwrap, then inspect_uvs (image overlaps) before painting: no flipped or "
                "overlapping triangles, a small texel density spread",
                "paint in surface space with object-space points (seams are invisible); "
                "view space paints what a camera sees, occluded by other objects",
                "bind with factor '#ffffff' so the painted color is the color you see",
                "bake from high-resolution sources onto a low-resolution target with "
                "apply: true; inspect misses in the report",
                "render_preview or observe_scene shows the result",
            ],
        }
    elif topic == "feedback":
        from ..feedback import ACTIONS, CHECKS, STOP_REASONS
        from ..feedback.checks import BASES
        from ..feedback.views import PRESETS

        data = {
            "workflow": "observe_scene -> look at the image -> diagnose_scene (expectations, "
            "your evaluations) -> propose_corrections -> apply_corrections (verified, undone "
            "if it does not help); or feedback_loop. Guide: aether3d://feedback/guide",
            "checks": {k: v["summary"] for k, v in CHECKS.items()},
            "actions": {k: v["summary"] for k, v in ACTIONS.items()},
            "views": ["camera", "virtual", "focus", *(f"preset:{p}" for p in PRESETS)],
            "bases": list(BASES),
            "safety": {
                "safe": "exact repair of a stated expectation; applied automatically",
                "verify": "kept only if verification shows improvement and no regression",
                "approval": "subjective or ambiguous; never applied without approve",
            },
            "stop_reasons": list(STOP_REASONS),
            "schemas": [
                "aether3d://schema/expectations",
                "aether3d://schema/vision-evaluation",
                "aether3d://schema/corrections",
            ],
        }
    else:  # gpu
        data = {"gpu": session.gpu()}
    return ToolOutput({"topic": topic, **data})


MAX_RESULTS = {
    "type": "integer",
    "minimum": 1,
    "maximum": 10000,
    "default": 100,
    "description": "the longest list one query answer returns; an answer cut short says "
    "so (truncated: true) and gives its full length (total)",
}


def bounded_answer(answer: dict[str, Any], limit: int) -> dict[str, Any]:
    """A query answer whose long list (the result, or ``describe``'s objects) keeps its
    first ``limit`` entries, with ``total`` and ``truncated``: a reply never floods an
    agent's context with thousands of entries it did not ask for."""
    result = answer.get("result")
    if isinstance(result, list) and len(result) > limit:
        return {**answer, "result": result[:limit], "total": len(result), "truncated": True}
    if isinstance(result, dict) and isinstance(result.get("objects"), list):
        objects = result["objects"]
        if len(objects) > limit:
            result = {**result, "objects": objects[:limit]}
            return {**answer, "result": result, "total": len(objects), "truncated": True}
    return answer


def _bounded_plan_result(data: dict[str, Any], limit: int) -> dict[str, Any]:
    if data.get("queries"):
        data["queries"] = [bounded_answer(q, limit) for q in data["queries"]]
    return data


# --------------------------------------------------------------------------- #
# The tool table
# --------------------------------------------------------------------------- #

_DISCARD = {
    "type": "boolean",
    "default": False,
    "description": "drop unsaved changes of the open scene",
}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "scene_new",
        "New scene",
        "Start a new, empty scene in this session (replaces the open one; refuses to drop "
        "unsaved changes unless discard_unsaved). Optionally give the project path that "
        "scene_save will write.",
        _schema(
            {
                "name": {"type": "string", "minLength": 1, "maxLength": 200},
                "path": PATH,
                "discard_unsaved": _DISCARD,
            }
        ),
        scene_new,
    ),
    ToolSpec(
        "scene_open",
        "Open a project",
        "Open a .aether project (with its undo history file, if any) or import a .glb/.gltf "
        "file from the workspace. Returns a compact summary of the scene.",
        _schema(
            {
                "path": PATH,
                "history": {"type": "boolean", "default": True},
                "allow_lossy": {"type": "boolean", "default": False},
                "discard_unsaved": _DISCARD,
            },
            ("path",),
        ),
        scene_open,
    ),
    ToolSpec(
        "scene_save",
        "Save the project",
        "Save the open scene as a .aether project (atomically). Without path it writes the "
        "project the scene came from; another existing file, or a project changed on disk "
        "since it was read, is replaced only with overwrite. history: also save the undo "
        "history (kept automatically once a project has a history file).",
        _schema({"path": PATH, "overwrite": OVERWRITE, "history": {"type": "boolean"}}),
        scene_save,
        destructive=True,
    ),
    ToolSpec(
        "scene_info",
        "Scene summary",
        "A compact summary of the open scene: counts by kind, tags, root objects, cameras, "
        "lights, materials, bounds, history and revision (pass revision to plans as "
        "expected_revision). detail=full adds world, settings, procedural parameters, GPU.",
        _schema({"detail": {"enum": ["summary", "full"], "default": "summary"}}),
        scene_info,
        read_only=True,
    ),
    ToolSpec(
        "objects_list",
        "List objects",
        "List objects (all, or a selection) with chosen fields, paginated: fields from "
        + ", ".join(OBJECT_FIELDS)
        + f" (default {', '.join(DEFAULT_FIELDS)}). Use next_cursor for the next page.",
        _schema(
            {
                "select": SELECTOR,
                "fields": {
                    "type": "array",
                    "items": {"enum": list(OBJECT_FIELDS)},
                    "uniqueItems": True,
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                "cursor": {"type": "string"},
            }
        ),
        objects_list,
        read_only=True,
    ),
    ToolSpec(
        "execute_scene_plan",
        "Build or change the scene (scene plan)",
        "Build or change the scene with ONE declarative scene plan: checked as a whole, "
        "then applied as one undo step (all or nothing). Use it for every creation and edit "
        "-- thousands of objects cost one call. Define meshes/materials/prefabs once and "
        'reference them as "@symbol"; place many objects with a pattern (grid, array, '
        "circle, line, scatter, instances, repeat) and vary them per instance; reach "
        'instances as "@sym[i]" and children as "@sym/child"; existing objects by id or '
        "name. operations update or delete existing objects (select by refs, ids, names, "
        "tags, name_glob; deletes by filter need expect). queries are answered afterwards "
        "(e.g. overlaps). Coordinates: +Z up, meters, rotation_euler in degrees, front is "
        "-Y. On error nothing changes and every problem is listed with a JSON pointer, a "
        'code and a hint. mode=dry_run applies and rolls back. Example: {"format": '
        '"aether3d.scene-plan", "version": 1, "meshes": {"block": {"primitive": '
        '"cube"}}, "objects": {"wall": {"mesh": "@block", "grid": {"count": '
        '[10, 1, 4]}, "position": [0, 0, 0.5]}}}. Guide: resource aether3d://guide.',
        _schema(
            {
                "plan": PLAN,
                "mode": {"enum": ["execute", "dry_run"], "default": "execute"},
                "expected_revision": {"type": "integer", "minimum": 0},
                "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 86400},
                "max_results": MAX_RESULTS,
            },
            ("plan",),
        ),
        execute_scene_plan,
    ),
    ToolSpec(
        "check_scene_plan",
        "Check a scene plan",
        "Check a scene plan without changing the scene: every problem with its JSON pointer, "
        "code and hint, and estimates (objects, meshes, triangles, memory). dry_run: true "
        "also applies it and rolls back, reporting exactly what it would create and the "
        "answers of its queries.",
        _schema(
            {
                "plan": PLAN,
                "dry_run": {"type": "boolean", "default": False},
                "max_results": MAX_RESULTS,
            },
            ("plan",),
        ),
        check_scene_plan,
        read_only=True,
    ),
    ToolSpec(
        "spatial_query",
        "Spatial query",
        "Answer spatial questions exactly, from the triangles, without changing anything: "
        "describe, bounds, find, raycast, raycast_all, closest_point, nearest, contains, "
        "distance, overlaps, region, camera_ray, pick, project, in_view, line_of_sight, "
        'visible_from, relation. request: {"query": name, ...arguments} (as aether3d '
        "query), or requests: a list of them. Objects are ids or names; objects, against "
        'and exclude also take a selector ({"tags": [...]}, {"under": id or name}, '
        '{"name_glob": "Tower */*"}, ...); points [x, y, z].',
        _schema(
            {
                "request": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "requests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "minItems": 1,
                    "maxItems": 50,
                },
                "max_results": MAX_RESULTS,
            }
        ),
        spatial_query,
        read_only=True,
        idempotent=True,
    ),
    ToolSpec(
        "render_preview",
        "Render a preview",
        "Render a quick preview image of the scene and return it. backend auto (default) "
        "uses the GPU rasterizer when one works and the CPU rasterizer otherwise (the report "
        "says which: backend, device, adapter, software, fallback). Never path traces. "
        "Without an active camera (or with frame) the view frames all visible geometry.",
        _schema(
            {
                "width": {"type": "integer", "minimum": 16, "maximum": 1920, "default": 640},
                "height": {"type": "integer", "minimum": 16, "maximum": 1920, "default": 360},
                "camera": {"type": "string", "description": "camera object id or name"},
                "frame": {"type": "boolean", "default": False},
                "backend": {"enum": ["auto", "gpu", "cpu"], "default": "auto"},
                "shadows": {"type": "boolean", "default": False},
                "save_to": PATH,
                "overwrite": OVERWRITE,
            }
        ),
        render_preview,
        read_only=True,
    ),
    ToolSpec(
        "render_image",
        "Render an image file",
        "Render to an image file in the workspace (.png, or linear .hdr/.pfm). backend: auto, "
        "gpu, cpu (rasterizers), or path / gpu_path (the path tracer: slow; spp samples per "
        "pixel, bounces; bounded by the server's sample budget). Returns the render report.",
        _schema(
            {
                "path": PATH,
                "width": {"type": "integer", "minimum": 16, "maximum": 8192, "default": 1280},
                "height": {"type": "integer", "minimum": 16, "maximum": 8192, "default": 720},
                "camera": {"type": "string"},
                "frame": {"type": "boolean", "default": False},
                "backend": {"enum": ["auto", "gpu", "cpu", "path", "gpu_path"], "default": "auto"},
                "samples": {"type": "integer", "minimum": 1, "maximum": 4},
                "spp": {"type": "integer", "minimum": 1, "maximum": 4096},
                "bounces": {"type": "integer", "minimum": 0, "maximum": 16},
                "shadows": {"type": "boolean", "default": False},
                "include_image": {"type": "boolean", "default": False},
                "overwrite": OVERWRITE,
            },
            ("path",),
        ),
        render_image,
    ),
    ToolSpec(
        "history",
        "Undo, redo, checkpoints",
        "The scene's history: list, undo (steps), redo (steps), checkpoint (name the current "
        "state), goto (a checkpoint), remove_checkpoint. A plan is one step.",
        _schema(
            {
                "action": {
                    "enum": ["list", "undo", "redo", "checkpoint", "goto", "remove_checkpoint"]
                },
                "steps": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 1},
                "name": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            ("action",),
        ),
        history,
    ),
    ToolSpec(
        "export_scene",
        "Export glTF",
        "Export the scene to a .glb or .gltf file in the workspace (materials and textures "
        "included; bake_textures=SIZE bakes procedural textures to images).",
        _schema(
            {
                "path": PATH,
                "allow_lossy": {"type": "boolean", "default": False},
                "bake_textures": {"type": "integer", "minimum": 16, "maximum": 8192},
                "overwrite": OVERWRITE,
            },
            ("path",),
        ),
        export_scene,
    ),
    ToolSpec(
        "run_script",
        "Run an edit, procedural, scene, sculpt or texture script",
        "Run one of the engine's JSON scripts atomically (one undo step): "
        "aether3d.edit-script (polygon mesh editing: extrude, bevel, bridge, knife, ... on an "
        "object), aether3d.procedural-script (procedural definitions), aether3d.scene-script "
        "(cameras, lights, materials and textures in detail), aether3d.sculpt-script (brush "
        "strokes, masks, dynamic topology, remeshing, symmetrize; schema: resource "
        "aether3d://schema/sculpt-script, guide: aether3d://sculpt/guide), "
        "aether3d.texture-script (UV unwrap/projection/island edits, painted textures and "
        "layers, brush strokes in surface/view/UV space, fills, gradients, bakes, texture "
        "transfer; schema: aether3d://schema/texture-script, guide: "
        "aether3d://texturing/guide). For creating and placing objects use "
        "execute_scene_plan.",
        _schema(
            {
                "script": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "enum": [
                                "aether3d.edit-script",
                                "aether3d.procedural-script",
                                "aether3d.scene-script",
                                "aether3d.sculpt-script",
                                "aether3d.texture-script",
                            ]
                        }
                    },
                    "required": ["format"],
                }
            },
            ("script",),
        ),
        run_script,
    ),
    ToolSpec(
        "inspect_mesh",
        "Inspect a mesh's validity",
        "Validity tier (valid < manifold < closed < solid) and quality of one object's mesh, "
        "in world units: counts, boundary and non-manifold edges and vertices, winding, "
        "degenerate triangles, components, Euler characteristic, volume, edge lengths, "
        "smallest angle; self_intersections=true adds the exact self-intersection test "
        "(needed for 'solid'). Findings have stable codes. Read-only.",
        _schema(
            {
                "object": {"type": "string", "minLength": 1, "description": "id or name"},
                "self_intersections": {"type": "boolean", "default": False},
            },
            ("object",),
        ),
        inspect_mesh,
        read_only=True,
        idempotent=True,
    ),
    ToolSpec(
        "inspect_uvs",
        "Inspect an object's UVs",
        "UV diagnostics of one object's mesh: islands, flipped, degenerate and overlapping "
        "triangles (unintended) apart from stacked islands (deliberate copies), mirrored "
        "islands, stretch, texel density and its spread, utilization, islands outside the "
        "unit square, smallest gap in texels; with a layout image (islands, density, "
        "stretch, overlaps, or a texture under the wireframe). Read-only.",
        _schema(
            {
                "object": {"type": "string", "minLength": 1, "description": "id or name"},
                "resolution": {"type": "integer", "minimum": 16, "maximum": 16384, "default": 1024},
                "islands_limit": {"type": "integer", "minimum": 0, "maximum": 256, "default": 16},
                "image": {
                    "enum": ["islands", "density", "stretch", "overlaps", "texture", "none"],
                    "default": "islands",
                },
                "texture": {"type": "string", "minLength": 1, "description": "for image texture"},
                "size": {"type": "integer", "minimum": 64, "maximum": 1024, "default": 512},
            },
            ("object",),
        ),
        inspect_uvs,
        read_only=True,
        idempotent=True,
    ),
    ToolSpec(
        "inspect_texture",
        "Inspect a texture",
        "A texture's details (size, encoding, color space; painted textures: layers with "
        "blend modes, opacity, masks, revision) and the material slots using it, with a "
        "preview image. Read-only.",
        _schema(
            {
                "texture": {"type": "string", "minLength": 1, "description": "id or name"},
                "preview_size": {"type": "integer", "minimum": 0, "maximum": 1024, "default": 256},
            },
            ("texture",),
        ),
        inspect_texture,
        read_only=True,
        idempotent=True,
    ),
    ToolSpec(
        "diagnostics",
        "Diagnostics",
        "Check the scene: geometry (degenerate or open meshes, ...), procedural (every node "
        "evaluated), materials (missing textures, ...), history (replays it on a copy). "
        "Findings have stable codes.",
        _schema(
            {
                "checks": {
                    "type": "array",
                    "items": {"enum": ["geometry", "procedural", "materials", "history"]},
                    "uniqueItems": True,
                }
            }
        ),
        diagnostics,
        read_only=True,
    ),
    ToolSpec(
        "capabilities",
        "Capabilities",
        "What this engine and server offer: overview (tools, conventions, scene), plan, "
        "primitives, modifiers, materials (models, presets), queries, limits, gpu, feedback "
        "(checks, correction actions, views, safety classes), sculpt (brushes, operations, "
        "validity tiers, script limits), texturing (texture script operations, brushes, "
        "bake maps, blend modes).",
        _schema(
            {
                "topic": {
                    "enum": [
                        "overview",
                        "plan",
                        "primitives",
                        "modifiers",
                        "materials",
                        "queries",
                        "limits",
                        "gpu",
                        "feedback",
                        "sculpt",
                        "texturing",
                    ],
                    "default": "overview",
                }
            }
        ),
        capabilities,
        read_only=True,
        idempotent=True,
    ),
)


def _with_feedback(tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
    from .feedback import specs

    return tools + specs()


TOOLS = _with_feedback(TOOLS)
TOOL_INDEX = {spec.name: spec for spec in TOOLS}


def encode_image(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def error_payload(exc: BaseException, session: Session) -> dict[str, Any]:
    """The structured error a failed tool call returns (no traceback, no absolute paths)."""
    if isinstance(exc, PlanError):
        data = exc.to_dict()
        data["type"] = "PlanError"
    elif isinstance(exc, AetherError):
        data = _engine_error(exc, session)
    else:  # a bug: say what happened, but no paths of this machine
        data = {
            "type": "InternalError",
            "code": "INTERNAL_ERROR",
            "message": f"{type(exc).__name__}: {redact_paths(str(exc))}",
        }
    return _redacted({"error": data}, session)
