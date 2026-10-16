"""Declarative sculpt scripts: ordered JSON operations, safe to accept from untrusted input.

A script is plain data (no code is ever evaluated)::

    {
      "format": "aether3d.sculpt-script",
      "version": 1,
      "label": "Sculpt the head",                    # optional: the undo step's name
      "limits": {"max_dabs": 20000, "max_triangles": 2000000, "max_seconds": 60},
      "sculpts": [
        {
          "object": "Head",                          # name, or {"id": ...} / {"name": ...}
          "require": "closed",                       # validity tier the result must reach
          "check": true,                             # exact self-intersection test
          "scope": "object",                         # or "shared"
          "operations": [
            {"op": "set_mask", "value": 1, "box": [[-1, -1, -1], [1, 1, 0]]},
            {"op": "stroke", "brush": {"kind": "draw", "radius": 0.2, "strength": 0.3},
             "path": [[0, -0.9, 0.3], [0.2, -0.9, 0.4]], "symmetry": ["x"]},
            {"op": "remesh", "target_length": 0.03},
            {"op": "voxel_remesh", "voxel_size": 0.02},
            {"op": "symmetrize", "axis": "x", "direction": "+"},
            {"op": "check"}
          ]
        }
      ]
    }

``op`` is ``"stroke"`` (the rest of the step is the stroke: ``brush``, ``path``, ...;
see :class:`~aether3d.sculpt.Stroke`), ``"set_mask"``, ``"invert_mask"``,
``"clear_mask"``, ``"remesh"``, ``"voxel_remesh"``, ``"symmetrize"`` (with their keyword
arguments), ``"undo"``/``"redo"`` (the session's own steps) or ``"check"`` (a validity
report of the current state, read-only).

:func:`run_sculpt_script` applies everything or nothing, as one step of the scene's
history: if a step fails, a limit is reached or a result is below its required validity
tier, every object keeps its mesh. Errors carry a stable ``code``, the ``location`` of
the step (``sculpts[0].operations[2].brush.radius``) and a hint.
"""

from __future__ import annotations

import inspect
import math
import time
from typing import TYPE_CHECKING, Any

from ..errors import AetherError, ObjectNotFoundError, SculptError
from .validate import TIERS, check_mesh

if TYPE_CHECKING:
    from ..scene import Scene

FORMAT_ID = "aether3d.sculpt-script"
VERSION = 1
MAX_OPERATIONS = 10_000
OPERATIONS = (
    "stroke",
    "set_mask",
    "invert_mask",
    "clear_mask",
    "remesh",
    "voxel_remesh",
    "symmetrize",
    "undo",
    "redo",
    "check",
)
LIMIT_KEYS = ("max_dabs", "max_triangles", "max_seconds")


class SculptScriptError(AetherError, ValueError):
    """A sculpt script is malformed or one of its steps failed: ``code`` (the
    underlying error's code, or ``INVALID_SCRIPT``), ``location`` and ``details``."""

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


def _object(value: Any, loc: str) -> dict:
    if not isinstance(value, dict):
        raise SculptScriptError(f"expected an object, got {type(value).__name__}", loc)
    return value


def _keys(obj: dict, allowed: set[str], loc: str, required: set[str] = frozenset()) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise SculptScriptError(f"unknown key(s): {', '.join(unknown)}", loc)
    missing = sorted(required - set(obj))
    if missing:
        raise SculptScriptError(f"missing key(s): {', '.join(missing)}", loc)


def _limits(raw: Any) -> dict[str, float]:
    raw = _object(raw, "script.limits")
    _keys(raw, set(LIMIT_KEYS), "script.limits")
    out: dict[str, float] = {}
    for key, value in raw.items():
        integer = key != "max_seconds"
        ok = isinstance(value, int) if integer else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok or not math.isfinite(value) or value <= 0:
            kind = "an integer" if integer else "a number"
            raise SculptScriptError(f"{key} must be {kind} > 0", f"script.limits.{key}")
        out[key] = value
    return out


def check_script(script: Any) -> dict:
    """Check a script's structure (everything that does not need the scene); returns it."""
    script = _object(script, "script")
    _keys(
        script,
        {"format", "version", "label", "limits", "sculpts"},
        "script",
        {"format", "version", "sculpts"},
    )
    if script["format"] != FORMAT_ID:
        raise SculptScriptError(f"format must be {FORMAT_ID!r}", "script.format")
    if isinstance(script["version"], bool) or script["version"] != VERSION:
        raise SculptScriptError(
            f"unsupported version {script['version']!r} (supported: {VERSION})", "script.version"
        )
    if "label" in script and (not isinstance(script["label"], str) or not script["label"].strip()):
        raise SculptScriptError("label must be a non-empty string", "script.label")
    _limits(script.get("limits", {}))
    sculpts = script["sculpts"]
    if not isinstance(sculpts, list) or not sculpts:
        raise SculptScriptError("sculpts must be a non-empty array", "script.sculpts")
    total = 0
    for i, entry in enumerate(sculpts):
        loc = f"sculpts[{i}]"
        entry = _object(entry, loc)
        _keys(
            entry,
            {"object", "require", "check", "scope", "operations"},
            loc,
            {"object", "operations"},
        )
        if "require" in entry and entry["require"] not in TIERS:
            raise SculptScriptError(f"require must be one of {', '.join(TIERS)}", f"{loc}.require")
        if "check" in entry and not isinstance(entry["check"], bool):
            raise SculptScriptError("check must be true or false", f"{loc}.check")
        if entry.get("scope", "object") not in ("object", "shared"):
            raise SculptScriptError("scope must be 'object' or 'shared'", f"{loc}.scope")
        ops = entry["operations"]
        if not isinstance(ops, list):
            raise SculptScriptError("operations must be an array", f"{loc}.operations")
        total += len(ops)
        for k, step in enumerate(ops):
            step = _object(step, f"{loc}.operations[{k}]")
            if step.get("op") not in OPERATIONS:
                raise SculptScriptError(
                    f"op must be one of {', '.join(OPERATIONS)}", f"{loc}.operations[{k}].op"
                )
    if total > MAX_OPERATIONS:
        raise SculptScriptError(
            f"too many operations ({total} > {MAX_OPERATIONS})", "script.sculpts"
        )
    return script


def _find(scene: Scene, ref: Any, loc: str):
    try:
        if isinstance(ref, str):
            return scene.get_by_name(ref)
        ref = _object(ref, loc)
        _keys(ref, {"id", "name"}, loc)
        if len(ref) != 1 or not all(isinstance(v, str) for v in ref.values()):
            raise SculptScriptError("give exactly one of id or name (a string)", loc)
        return scene.get(ref["id"]) if "id" in ref else scene.get_by_name(ref["name"])
    except ObjectNotFoundError:
        raise SculptScriptError(f"object {ref!r} not found", loc, "OBJECT_NOT_FOUND") from None


def _fail(exc: AetherError, loc: str) -> SculptScriptError:
    code = getattr(exc, "code", None) or type(exc).__name__
    details = {}
    if getattr(exc, "hint", None):
        details["hint"] = exc.hint
    message = getattr(exc, "message", None) or str(exc)
    return SculptScriptError(message, loc, code, details)


class _Budget:
    """The script's limits, checked at the safe points of long operations and after each
    step (a stop rolls the whole script back)."""

    def __init__(self, limits: dict[str, float]) -> None:
        self.limits = limits
        self.dabs = 0
        self.deadline = (
            time.monotonic() + limits["max_seconds"] if "max_seconds" in limits else None
        )

    def stop(self) -> bool:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise SculptError(
                "BUDGET_EXCEEDED",
                f"the script ran longer than max_seconds ({self.limits['max_seconds']} s)",
                hint="raise limits.max_seconds or split the script",
            )
        return False

    def after(self, sm, data: dict) -> None:
        self.dabs += int(data.get("dabs", 0)) if isinstance(data, dict) else 0
        if "max_dabs" in self.limits and self.dabs > self.limits["max_dabs"]:
            raise SculptError(
                "BUDGET_EXCEEDED",
                f"the script made {self.dabs} dabs (max_dabs {int(self.limits['max_dabs'])})",
                hint="space the dabs further apart or raise limits.max_dabs",
            )
        if "max_triangles" in self.limits and sm.triangle_count > self.limits["max_triangles"]:
            raise SculptError(
                "BUDGET_EXCEEDED",
                f"the mesh has {sm.triangle_count} triangles (max_triangles "
                f"{int(self.limits['max_triangles'])})",
                hint="use a coarser detail size or raise limits.max_triangles",
            )


def run_sculpt_script(
    scene: Scene, script: dict[str, Any], *, label: str | None = None
) -> dict[str, Any]:
    """Apply a sculpt script to ``scene`` (everything or nothing; see the module docs).
    Returns ``{"sculpts": [...]}``: per object, each step's result and the commit."""
    script = check_script(script)
    budget = _Budget(_limits(script.get("limits", {})))
    sculpts = script["sculpts"]
    label = (
        label
        or script.get("label")
        or (f"Sculpt script ({len(sculpts)} object{'s' * (len(sculpts) != 1)})")
    )
    out = []
    sessions = []
    try:
        with scene.history.transaction(label):
            for i, entry in enumerate(sculpts):
                loc = f"sculpts[{i}]"
                obj = _find(scene, entry["object"], f"{loc}.object")
                try:
                    session = obj.sculpt()
                except AetherError as exc:
                    raise _fail(exc, f"{loc}.object") from None
                sessions.append(session)
                session.mesh.should_stop = budget.stop
                steps = []
                for k, step in enumerate(entry["operations"]):
                    steps.append(_run_step(session, step, f"{loc}.operations[{k}]", budget))
                try:
                    commit = session.commit(
                        scope=entry.get("scope", "object"),
                        check=entry.get("check", False),
                        require=entry.get("require"),
                    )
                except AetherError as exc:
                    raise _fail(exc, f"{loc}.require" if "require" in entry else loc) from None
                out.append(
                    {
                        "object": {"id": obj.id, "name": obj.name},
                        "operations": steps,
                        "commit": commit.to_dict(),
                    }
                )
    finally:
        for session in sessions:
            if not session.closed:
                session.discard()
    return {"sculpts": out}


def _run_step(session, step: dict, loc: str, budget: _Budget) -> dict[str, Any]:
    sm = session.mesh
    op = step["op"]
    args = {k: v for k, v in step.items() if k != "op"}
    try:
        if op == "check":
            _keys(args, {"self_intersections"}, loc)
            if not isinstance(args.get("self_intersections", False), bool):
                raise SculptScriptError(
                    "self_intersections must be true or false", f"{loc}.self_intersections"
                )
            mesh = sm.to_mesh()
            report = check_mesh(
                mesh.positions,
                mesh.triangles,
                self_intersections=bool(args.get("self_intersections", False)),
            )
            return {"op": op, "report": report.to_dict()}
        if op in ("undo", "redo"):
            _keys(args, set(), loc)
            getattr(sm, op)()
            return {"op": op}
        if op == "stroke":
            result = sm.stroke(args)
        else:
            method = getattr(sm, op)
            try:
                inspect.signature(method).bind(**args)
            except TypeError as exc:
                raise SculptScriptError(str(exc), loc, "INVALID_ARGUMENT") from None
            result = method(**args)
    except SculptScriptError:
        raise
    except AetherError as exc:
        raise _fail(exc, loc) from None
    try:
        budget.after(sm, result.data)
    except SculptError as exc:
        raise _fail(exc, loc) from None
    return {"op": op, **result.to_dict()}
