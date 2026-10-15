"""Declarative JSON procedural scripts (data only -- nothing is ever evaluated as code).

Format::

    {
      "format": "aether3d.procedural-script",
      "version": 1,
      "operations": [
        {"op": "add_parameter", "name": "width", "value": 20.0, "min": 4, "max": 200},
        {"op": "add_geometry", "name": "Wall", "primitive": "box",
         "params": {"width": {"expr": "width"}, "depth": 1, "height": 6}},
        {"op": "add_modifier", "geometry": "Wall", "type": "array", "params": {"count": 3}},
        {"op": "create_object", "name": "Wall", "geometry": "Wall"},
        {"op": "set_parameter", "name": "width", "value": 30.0},
        {"op": "validate", "require_ok": true}
      ]
    }

Operations: ``add_parameter``, ``set_parameter``, ``remove_parameter``,
``add_rule``, ``remove_rule`` (scene scope, or a geometry's with
``"geometry"``); ``add_geometry`` (``primitive`` +
``params`` | ``source`` + ``overrides`` [+ ``after``] | ``mesh_from_object``;
optional ``define``: a list of the node's own parameter definitions);
``remove_geometry``, ``rename_geometry``, ``set_source``,
``convert_source_to_mesh``; ``add_modifier``, ``set_modifier``,
``remove_modifier``, ``move_modifier``; ``create_object``, ``assign``,
``bake``; and the read-only ``evaluate``, ``validate``, ``info``.

:func:`run_procedural_script` applies all operations as one transaction: if
any step fails, all definitions and objects are restored and created objects
are removed. Errors are :class:`ScriptError` with a location such as
``operations[3].params.count``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..editing.script import ScriptError
from ..errors import AetherError
from .params import ProceduralError, value_from_json

if TYPE_CHECKING:
    from ..scene import Scene

FORMAT_ID = "aether3d.procedural-script"
VERSION = 1
MAX_OPERATIONS = 10_000

_PARAM_DEF = {
    "op",
    "name",
    "value",
    "type",
    "min",
    "max",
    "choices",
    "description",
    "unit",
    "geometry",
}
_SPECS: dict[str, tuple[set[str], set[str]]] = {  # op -> (required, optional)
    "add_parameter": ({"name", "value"}, _PARAM_DEF - {"op", "name", "value"}),
    "set_parameter": ({"name", "value"}, {"geometry"}),
    "remove_parameter": ({"name"}, {"geometry"}),
    "add_rule": ({"name", "expr"}, {"message", "geometry"}),
    "remove_rule": ({"name"}, {"geometry"}),
    "add_geometry": (
        {"name"},
        {"primitive", "params", "source", "overrides", "after", "mesh_from_object", "define"},
    ),
    "remove_geometry": ({"geometry"}, set()),
    "rename_geometry": ({"geometry", "name"}, set()),
    "set_source": ({"geometry", "params"}, set()),
    "convert_source_to_mesh": ({"geometry"}, set()),
    "add_modifier": ({"geometry", "type"}, {"name", "enabled", "index", "params"}),
    "set_modifier": ({"geometry", "modifier"}, {"params", "enabled", "name"}),
    "remove_modifier": ({"geometry", "modifier"}, set()),
    "move_modifier": ({"geometry", "modifier", "index"}, set()),
    "create_object": ({"name", "geometry"}, {"parent", "position", "rotation_euler", "scale"}),
    "assign": ({"object", "geometry"}, set()),
    "bake": ({"object"}, set()),
    "evaluate": ({"geometry"}, set()),
    "validate": (set(), {"geometry", "require_ok"}),
    "info": (set(), {"geometry", "evaluate"}),
}


def run_procedural_script(scene: Scene, script: Any, *, label: str | None = None) -> dict[str, Any]:
    """Apply a procedural script atomically; returns per-operation results.

    The script is one step of ``scene.history``, named ``label`` (default: the
    script's own ``"label"``, else a description).
    """
    if not isinstance(script, dict):
        raise ScriptError("the script must be a JSON object", "script")
    unknown = sorted(set(script) - {"format", "version", "operations", "label"})
    if unknown:
        raise ScriptError(f"unknown key(s): {', '.join(unknown)}", "script")
    if "label" in script and (not isinstance(script["label"], str) or not script["label"].strip()):
        raise ScriptError("label must be a non-empty string", "script.label")
    if script.get("format") != FORMAT_ID:
        raise ScriptError(f"format must be {FORMAT_ID!r}", "script.format")
    if script.get("version") != VERSION:
        raise ScriptError(
            f"unsupported version {script.get('version')!r} (supported: {VERSION})",
            "script.version",
        )
    operations = script.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ScriptError("operations must be a non-empty array", "script.operations")
    if len(operations) > MAX_OPERATIONS:
        raise ScriptError(f"too many operations (limit {MAX_OPERATIONS})", "script.operations")

    system = scene.procedural
    created: list[str] = []
    results: list[dict[str, Any]] = []
    # One step of the scene's history, all or nothing (objects created by earlier
    # operations included); validated and evaluated once, at the end.
    count = len(operations)
    label = (
        label or script.get("label") or f"Procedural script ({count} operation{'s' * (count != 1)})"
    )
    with scene.history.transaction(label), system.batch(label):
        for i, step in enumerate(operations):
            results.append(_run(scene, step, f"operations[{i}]", created))
    return {"operations": results, "revision": system.revision}


def _run(scene: Scene, step: Any, loc: str, created: list[str]) -> dict[str, Any]:
    if not isinstance(step, dict) or not isinstance(step.get("op"), str):
        raise ScriptError('each operation must be an object with an "op" string', loc)
    op = step["op"]
    if op not in _SPECS:
        raise ScriptError(
            f"unknown operation {op!r}",
            f"{loc}.op",
            "UNKNOWN_OPERATION",
            {"hint": "one of " + ", ".join(sorted(_SPECS))},
        )
    required, optional = _SPECS[op]
    keys = set(step) - {"op"}
    missing, extra = sorted(required - keys), sorted(keys - required - optional)
    if missing or extra:
        what = f"missing {', '.join(missing)}" if missing else f"unknown key(s) {', '.join(extra)}"
        raise ScriptError(f"{op}: {what}", loc)
    system = scene.procedural
    try:
        result = {"op": op, **_dispatch(scene, system, op, step, created)}
        # Check after every step (cheap: unchanged nodes hit the cache) so a failure
        # is reported at the operation that caused it.
        system._check_all()
        return result
    except ProceduralError as exc:
        location = f"{loc}: {exc.location}" if exc.location else loc
        details = {"hint": exc.hint, **({"details": exc.details} if exc.details else {})}
        raise ScriptError(exc.message, location, exc.code, details) from None
    except AetherError as exc:
        raise ScriptError(str(exc), loc, getattr(exc, "code", type(exc).__name__)) from None
    except (TypeError, ValueError) as exc:
        raise ScriptError(str(exc), loc, "INVALID_ARGUMENT") from None


def _object_ref(scene: Scene, ref: Any):
    for getter in (scene.get, scene.get_by_name):
        try:
            return getter(ref)
        except AetherError:
            continue
    raise ProceduralError("UNKNOWN_OBJECT", f"no object with id or name {ref!r}")


def _dispatch(scene: Scene, system, op: str, s: dict, created: list[str]) -> dict[str, Any]:
    geometry = s.get("geometry")
    if op == "add_parameter":
        options = {
            k: s[k] for k in ("type", "min", "max", "choices", "description", "unit") if k in s
        }
        p = system.add_parameter(
            s["name"], value_from_json(s["value"]), geometry=geometry, **options
        )
        return {"parameter": p.to_json()}
    if op == "set_parameter":
        system.set_parameter(s["name"], value_from_json(s["value"]), geometry=geometry)
        return {}
    if op == "remove_parameter":
        system.remove_parameter(s["name"], geometry=geometry)
        return {}
    if op == "add_rule":
        rule = system.add_rule(s["name"], s["expr"], s.get("message", ""), geometry=geometry)
        return {"rule": rule.to_json()}
    if op == "remove_rule":
        system.remove_rule(s["name"], geometry=geometry)
        return {}
    if op == "add_geometry":
        sources = [k for k in ("primitive", "source", "mesh_from_object") if k in s]
        if len(sources) != 1:
            raise ProceduralError(
                "INVALID_SOURCE", "give exactly one of primitive, source, mesh_from_object"
            )
        if "params" in s and "primitive" not in s:
            raise ProceduralError("INVALID_SOURCE", "params only applies to primitive sources")
        if ("overrides" in s or "after" in s) and "source" not in s:
            raise ProceduralError(
                "INVALID_SOURCE", "overrides/after only apply to geometry sources"
            )
        mesh = None
        if "mesh_from_object" in s:
            obj = _object_ref(scene, s["mesh_from_object"])
            mesh = obj.mesh
            if mesh is None:
                raise ProceduralError("INVALID_SOURCE", f"object {obj.name!r} has no mesh")
        g = system.add_geometry(
            s["name"],
            primitive=s.get("primitive"),
            params=s.get("params", {}),
            mesh=mesh,
            source=s.get("source"),
            overrides=s.get("overrides"),
            after=s.get("after"),
        )
        for k, spec in enumerate(s.get("define", [])):
            if not isinstance(spec, dict) or not {"name", "value"} <= set(spec):
                raise ProceduralError("INVALID_PARAMETER", f"define[{k}] needs name and value")
            options = {key: spec[key] for key in spec if key not in ("name", "value")}
            system.add_parameter(
                spec["name"], value_from_json(spec["value"]), geometry=g, **options
            )
        return {"geometry": {"id": g.id, "name": g.name}}
    if op == "remove_geometry":
        system.remove_geometry(geometry)
        return {}
    if op == "rename_geometry":
        system.rename_geometry(geometry, s["name"])
        return {}
    if op == "set_source":
        system.set_source(geometry, s["params"])
        return {}
    if op == "convert_source_to_mesh":
        system.convert_source_to_mesh(geometry)
        return {}
    if op == "add_modifier":
        m = system.add_modifier(
            geometry,
            s["type"],
            name=s.get("name"),
            enabled=s.get("enabled", True),
            index=s.get("index"),
            params=s.get("params", {}),
        )
        return {"modifier": m.info()}
    if op == "set_modifier":
        system.set_modifier(
            geometry,
            s["modifier"],
            enabled=s.get("enabled"),
            name=s.get("name"),
            params=s.get("params"),
        )
        return {
            "modifier": system._find_modifier(
                system.geometry(geometry), s.get("name") or s["modifier"]
            ).info()
        }
    if op == "remove_modifier":
        system.remove_modifier(geometry, s["modifier"])
        return {}
    if op == "move_modifier":
        system.move_modifier(geometry, s["modifier"], s["index"])
        return {}
    if op == "create_object":
        parent = _object_ref(scene, s["parent"]).id if s.get("parent") is not None else None
        obj = scene.add_object(
            s["name"],
            geometry=system.geometry(geometry).id,
            parent=parent,
            position=s.get("position"),
            scale=s.get("scale"),
        )
        created.append(obj.id)
        if "rotation_euler" in s:
            obj.rotation_euler = s["rotation_euler"]
        return {"object": {"id": obj.id, "name": obj.name}}
    if op == "assign":
        obj = _object_ref(scene, s["object"])
        system.assign(obj, geometry)
        return {}
    if op == "bake":
        obj = _object_ref(scene, s["object"])
        mesh = obj.bake()
        return {
            "mesh": None
            if mesh is None
            else {"vertex_count": mesh.vertex_count, "triangle_count": mesh.triangle_count}
        }
    if op == "evaluate":
        evaluation = system.evaluate(geometry)
        mesh = evaluation.mesh
        return {
            "key": evaluation.key,
            "polygons": evaluation.polygons.validate().summary,
            "mesh": {"vertex_count": mesh.vertex_count, "triangle_count": mesh.triangle_count},
            "diagnostics": [d.to_dict() for d in evaluation.diagnostics],
        }
    if op == "validate":
        report = system.validate(geometry)
        if s.get("require_ok") and report["status"] == "error":
            raise ProceduralError(
                "VALIDATION_FAILED",
                "procedural validation found errors",
                details={"report": report},
            )
        return {"report": report}
    if op == "info":
        return {"info": system.info(geometry, evaluate=bool(s.get("evaluate", False)))}
    raise AssertionError(op)  # pragma: no cover
