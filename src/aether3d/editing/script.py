"""Declarative edit scripts: ordered JSON operations, safe to accept from untrusted input.

A script is plain data (no code is ever evaluated)::

    {
      "format": "aether3d.edit-script",
      "version": 1,
      "edits": [
        {
          "object": "Sword",                 # name, or {"id": ...} / {"name": ...}
          "create": false,                   # create an empty object if missing
          "scope": "object",                 # or "shared" (see EditSession.commit)
          "operations": [
            {"op": "select", "kind": "faces", "normal": [[0, 0, 1], 1], "expect": 1, "as": "top"},
            {"op": "extrude_faces", "faces": {"selection": "top"}, "distance": 0.5,
             "save": {"walls": "side_faces"}},
            {"op": "inset_faces", "faces": {"selection": "top"}, "thickness": 0.1},
            {"op": "validate"}
          ]
        }
      ]
    }

``op`` is ``"select"`` (or another query: ``select_all``, ``edge_loop``,
``edge_ring``, ``linked_faces``, ``region_boundary``, ``boundary_edges``,
``grow``, ``to_vertices``, ``to_edges``, ``to_faces``, ``combine``) storing a
named selection with ``"as"``; ``"validate"``; ``"summary"``/``"describe"``
(read-only queries); ``"undo"``/``"redo"``; or any editing operation of
:class:`EditableMesh` (``EditableMesh.operations()``) with its keyword
arguments. Element arguments accept ids, ``{"selection": name}`` or an inline
``{"select": {...}}`` query. ``"save": {name: key}`` stores selections from an
operation's result (``"created"`` or a key of its ``data``).

:func:`run_edit_script` applies everything or nothing: if any step fails, all
objects keep their original meshes and created objects are removed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import AetherError, EditError, ObjectNotFoundError, ValidationError
from .core import OperationResult, Selection
from .mesh import EditableMesh

if TYPE_CHECKING:
    from ..scene import Scene

FORMAT_ID = "aether3d.edit-script"
VERSION = 1
MAX_OPERATIONS = 10_000

QUERY_KEYS = {
    "kind",
    "ids",
    "box",
    "sphere",
    "plane",
    "normal",
    "nearest",
    "boundary",
    "sides",
    "group",
    "test",
    "space",
    "expect",
}
SELECTION_QUERIES = {
    "select_all": (),
    "edge_loop": ("edge",),
    "edge_ring": ("edge",),
    "linked_faces": ("faces",),
    "region_boundary": ("faces",),
    "boundary_edges": (),
    "non_manifold_edges": (),
    "grow": ("selection",),
    "to_vertices": ("selection",),
    "to_edges": ("selection",),
    "to_faces": ("selection",),
}
ELEMENT_PARAMS = {"faces", "edges", "vertices", "elements", "selection"}
SINGLE_PARAMS = {"edge": "edges", "face": "faces"}
VERTEX_ID_PARAMS = {"vertex_a", "vertex_b", "keep", "from_vertex"}


class ScriptError(AetherError, ValueError):
    """An edit script is malformed or one of its steps failed.

    ``location`` is a path such as ``edits[0].operations[3].distance``;
    ``code`` is the underlying error code (``EditError`` codes, or
    ``INVALID_SCRIPT``/``INVALID_ARGUMENT``); ``details`` carries the
    underlying error's elements and hint when available.
    """

    def __init__(
        self, message: str, location: str, code: str = "INVALID_SCRIPT", details: dict | None = None
    ):
        self.message = message
        self.location = location
        self.code = code
        self.details = details or {}
        super().__init__(f"{location}: [{code}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            **self.details,
        }


@dataclass
class ScriptResult:
    edits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"edits": self.edits}


def _object(value: Any, loc: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScriptError(f"expected an object, got {type(value).__name__}", loc)
    return value


def _check_keys(obj: dict, allowed: set[str], loc: str, required: set[str] = frozenset()) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ScriptError(f"unknown key(s): {', '.join(unknown)}", loc)
    missing = sorted(required - set(obj))
    if missing:
        raise ScriptError(f"missing key(s): {', '.join(missing)}", loc)


def _find_object(scene: Scene, ref: Any, loc: str):
    if isinstance(ref, str):
        try:
            return scene.get_by_name(ref)
        except ObjectNotFoundError:
            return None
    ref = _object(ref, loc)
    _check_keys(ref, {"id", "name"}, loc)
    if len(ref) != 1:
        raise ScriptError("give exactly one of id or name", loc)
    try:
        return scene.get(ref["id"]) if "id" in ref else scene.get_by_name(ref["name"])
    except ObjectNotFoundError:
        return None


def run_edit_script(
    scene: Scene, script: dict[str, Any], *, label: str | None = None
) -> ScriptResult:
    """Apply an edit script to ``scene`` (all edits or none); see the module docs.

    The script is one step of ``scene.history``, named ``label`` (default: the
    script's own ``"label"``, else a description).
    """
    script = _object(script, "script")
    _check_keys(
        script,
        {"format", "version", "edits", "label"},
        "script",
        {"format", "version", "edits"},
    )
    if "label" in script and (not isinstance(script["label"], str) or not script["label"].strip()):
        raise ScriptError("label must be a non-empty string", "script.label")
    if script["format"] != FORMAT_ID:
        raise ScriptError(f"format must be {FORMAT_ID!r}", "script.format")
    if script["version"] != VERSION:
        raise ScriptError(
            f"unsupported script version {script['version']!r} (supported: {VERSION})",
            "script.version",
        )
    edits = script["edits"]
    if not isinstance(edits, list) or not edits:
        raise ScriptError("edits must be a non-empty array", "script.edits")
    total = sum(len(e.get("operations", [])) for e in edits if isinstance(e, dict))
    if total > MAX_OPERATIONS:
        raise ScriptError(f"too many operations ({total} > {MAX_OPERATIONS})", "script.edits")

    created: list[str] = []
    result = ScriptResult()
    # One step of the scene's history, all or nothing: if an edit fails, the
    # transaction restores every mesh and removes the objects created before it.
    label = (
        label or script.get("label") or f"Edit script ({len(edits)} edit{'s' * (len(edits) != 1)})"
    )
    with scene.history.transaction(label):
        for i, edit in enumerate(edits):
            result.edits.append(_run_edit(scene, edit, f"edits[{i}]", created))
    return result


def _run_edit(scene: Scene, edit: Any, loc: str, created: list[str]) -> dict[str, Any]:
    edit = _object(edit, loc)
    _check_keys(
        edit,
        {"object", "create", "scope", "weld", "quads", "allow_invalid", "operations"},
        loc,
        {"object", "operations"},
    )
    obj = _find_object(scene, edit["object"], f"{loc}.object")
    if obj is None:
        if not edit.get("create", False):
            raise ScriptError(
                f'object {edit["object"]!r} not found (set "create": true to create it)',
                f"{loc}.object",
                "OBJECT_NOT_FOUND",
            )
        name = edit["object"] if isinstance(edit["object"], str) else edit["object"].get("name")
        if not isinstance(name, str):
            raise ScriptError("can only create an object given by name", f"{loc}.object")
        obj = scene.add_object(name)
        created.append(obj.id)
    operations = edit["operations"]
    if not isinstance(operations, list):
        raise ScriptError("operations must be an array", f"{loc}.operations")
    try:
        session = obj.edit_mesh(weld=edit.get("weld", "exact"), quads=edit.get("quads", True))
    except AetherError as exc:
        raise ScriptError(
            str(exc), f"{loc}.object", getattr(exc, "code", type(exc).__name__)
        ) from None
    selections: dict[str, Selection] = {}
    outputs = []
    for k, step in enumerate(operations):
        outputs.append(_run_step(session.mesh, step, f"{loc}.operations[{k}]", selections))
    try:
        commit = session.commit(
            scope=edit.get("scope", "object"), allow_invalid=bool(edit.get("allow_invalid", False))
        )
    except EditError as exc:
        raise ScriptError(exc.message, f"{loc}.commit", exc.code, {"hint": exc.hint}) from None
    except ValidationError as exc:
        raise ScriptError(str(exc), f"{loc}.scope", "INVALID_ARGUMENT") from None
    return {
        "object": {"id": obj.id, "name": obj.name},
        "operations": outputs,
        "selections": {name: sel.to_dict() for name, sel in selections.items()},
        "commit": commit.to_dict(),
    }


def _run_step(
    mesh: EditableMesh, step: Any, loc: str, selections: dict[str, Selection]
) -> dict[str, Any]:
    step = _object(step, loc)
    op = step.get("op")
    if not isinstance(op, str):
        raise ScriptError('each operation needs an "op" string', loc)
    try:
        if op == "select":
            _check_keys(step, QUERY_KEYS | {"op", "as"}, loc, {"kind", "as"})
            criteria = {k: v for k, v in step.items() if k not in ("op", "kind", "as")}
            sel = mesh.select(step["kind"], **_query_args(criteria, f"{loc}"))
            return _store(selections, step["as"], sel, op, loc)
        if op == "combine":
            _check_keys(step, {"op", "as", "union", "intersection", "difference"}, loc, {"as"})
            parts = [k for k in ("union", "intersection", "difference") if k in step]
            if len(parts) != 1:
                raise ScriptError("give exactly one of union, intersection, difference", loc)
            names = step[parts[0]]
            if not isinstance(names, list) or len(names) < 2:
                raise ScriptError(
                    f"{parts[0]} needs a list of at least 2 selection names", f"{loc}.{parts[0]}"
                )
            sels = [_named(selections, n, f"{loc}.{parts[0]}") for n in names]
            out = sels[0]
            for s in sels[1:]:
                out = (
                    out | s
                    if parts[0] == "union"
                    else out & s
                    if parts[0] == "intersection"
                    else out - s
                )
            return _store(selections, step["as"], out, op, loc)
        if op in SELECTION_QUERIES:
            params = SELECTION_QUERIES[op]
            _check_keys(step, {"op", "as", *params}, loc, {"as", *params})
            args = [_resolve(mesh, p, step[p], f"{loc}.{p}", selections) for p in params]
            return _store(selections, step["as"], getattr(mesh, op)(*args), op, loc)
        if op == "validate":
            _check_keys(step, {"op", "require_ok"}, loc)
            report = mesh.validate()
            if step.get("require_ok") and not report.ok:
                raise ScriptError(
                    "validation found errors: " + ", ".join(sorted(report.codes())),
                    loc,
                    "VALIDATION_FAILED",
                )
            return {"op": op, "report": report.to_dict()}
        if op == "summary":
            _check_keys(step, {"op"}, loc)
            return {"op": op, "summary": mesh.summary()}
        if op == "describe":
            _check_keys(step, {"op", "kinds", "limit"}, loc)
            return {
                "op": op,
                "elements": mesh.describe(
                    step.get("kinds", ("vertices", "edges", "faces")), step.get("limit")
                ),
            }
        if op in ("undo", "redo"):
            _check_keys(step, {"op"}, loc)
            out = mesh.undo() if op == "undo" else mesh.redo().to_dict()
            return {"op": op, "result": out}
        if op not in EditableMesh.operations():
            raise ScriptError(
                f"unknown operation {op!r}",
                f"{loc}.op",
                "UNKNOWN_OPERATION",
                {"hint": "see EditableMesh.operations()"},
            )
        return _run_operation(mesh, op, step, loc, selections)
    except ScriptError:
        raise
    except EditError as exc:
        raise ScriptError(
            exc.message, loc, exc.code, {"elements": exc.elements, "hint": exc.hint}
        ) from None
    except (ValidationError, TypeError, ValueError, KeyError) as exc:
        raise ScriptError(str(exc), loc, "INVALID_ARGUMENT") from None


def _run_operation(mesh, op: str, step: dict, loc: str, selections: dict) -> dict[str, Any]:
    method = getattr(mesh, op)
    signature = inspect.signature(method)
    kwargs = {}
    for key, value in step.items():
        if key in ("op", "save"):
            continue
        if key not in signature.parameters:
            raise ScriptError(
                f"{op} has no argument {key!r} (arguments: {', '.join(signature.parameters)})",
                f"{loc}.{key}",
                "INVALID_ARGUMENT",
            )
        kwargs[key] = _resolve(mesh, key, value, f"{loc}.{key}", selections)
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        raise ScriptError(str(exc), loc, "INVALID_ARGUMENT") from None
    result: OperationResult = method(**kwargs)
    saved = {}
    if "save" in step:
        save = _object(step["save"], f"{loc}.save")
        for name, key in save.items():
            try:
                sel = result.selection(key)
            except KeyError:
                raise ScriptError(
                    f"result has no selection {key!r} (data keys: {', '.join(result.data)})",
                    f"{loc}.save.{name}",
                    "INVALID_ARGUMENT",
                ) from None
            selections[name] = sel
            saved[name] = sel.to_dict()
    return {"op": op, "result": result.to_dict(), "saved": saved}


def _store(selections: dict, name: Any, sel: Selection, op: str, loc: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ScriptError('"as" must be a non-empty string', f"{loc}.as")
    selections[name] = sel
    return {"op": op, "as": name, "selection": sel.to_dict(), "count": len(sel)}


def _named(selections: dict, name: Any, loc: str) -> Selection:
    if not isinstance(name, str) or name not in selections:
        raise ScriptError(
            f"no selection named {name!r} (defined: {', '.join(sorted(selections)) or 'none'})",
            loc,
            "UNKNOWN_SELECTION",
        )
    return selections[name]


def _query_args(criteria: dict, loc: str) -> dict:
    out = dict(criteria)
    if "box" in out:
        box = out["box"]
        if isinstance(box, dict):
            _check_keys(box, {"min", "max"}, f"{loc}.box", {"min", "max"})
            out["box"] = (box["min"], box["max"])
    if "sphere" in out and isinstance(out["sphere"], dict):
        _check_keys(out["sphere"], {"center", "radius"}, f"{loc}.sphere", {"center", "radius"})
        out["sphere"] = (out["sphere"]["center"], out["sphere"]["radius"])
    if "normal" in out and isinstance(out["normal"], dict):
        _check_keys(
            out["normal"], {"direction", "max_angle"}, f"{loc}.normal", {"direction", "max_angle"}
        )
        out["normal"] = (out["normal"]["direction"], out["normal"]["max_angle"])
    return out


def _resolve(mesh, key: str, value: Any, loc: str, selections: dict) -> Any:
    """Turn JSON argument values into Python arguments (selections, ids, dicts)."""
    if key in ELEMENT_PARAMS or key in SINGLE_PARAMS:
        if isinstance(value, dict):
            _check_keys(value, {"selection", "select"}, loc)
            if len(value) != 1:
                raise ScriptError("give exactly one of selection or select", loc)
            if "selection" in value:
                sel = _named(selections, value["selection"], f"{loc}.selection")
            else:
                query = _object(value["select"], f"{loc}.select")
                _check_keys(query, QUERY_KEYS, f"{loc}.select", {"kind"})
                criteria = {k: v for k, v in query.items() if k != "kind"}
                sel = mesh.select(query["kind"], **_query_args(criteria, f"{loc}.select"))
            if key in SINGLE_PARAMS:
                ids = getattr(sel, SINGLE_PARAMS[key])
                if len(ids) != 1:
                    raise ScriptError(
                        f"{key} needs exactly one {SINGLE_PARAMS[key][:-1]}, the selection has "
                        f"{len(ids)}",
                        loc,
                        "AMBIGUOUS_SELECTION",
                    )
                return next(iter(ids))
            return sel
        if key in SINGLE_PARAMS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScriptError(
                    f"{key} must be an id or a selection reference", loc, "INVALID_ARGUMENT"
                )
            return value
        if not isinstance(value, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise ScriptError(
                f"{key} must be a list of ids or a selection reference", loc, "INVALID_ARGUMENT"
            )
        return value
    if key == "positions" and isinstance(value, dict):
        try:
            return {int(k): v for k, v in value.items()}
        except ValueError:
            raise ScriptError(
                "positions keys must be vertex ids", loc, "INVALID_ARGUMENT"
            ) from None
    if key in VERTEX_ID_PARAMS and (isinstance(value, bool) or not isinstance(value, int)):
        raise ScriptError(f"{key} must be a vertex id", loc, "INVALID_ARGUMENT")
    return value
