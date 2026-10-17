"""Declarative CAD scripts: parametric sketches, parts, assemblies, buildings and
drawings as ordered JSON operations, safe to accept from untrusted input.

A script is plain data -- no code is evaluated (expressions such as ``"=width / 2"``
use the document's arithmetic-only evaluator), no file is read or written. Every length
in a script is a number in the document's unit or a quantity (``"3/8 in"``), sketch
point positions and radii included (the stored sketch definition, the solver's state,
keeps them in metres)::

    {
      "format": "aether3d.cad-script",
      "version": 1,
      "label": "Bracket",                                  # optional: the undo step's name
      "limits": {"max_seconds": 120, "max_items": 500},
      "operations": [
        {"op": "units", "length": "mm"},
        {"op": "parameter", "name": "width", "value": "80 mm"},
        {"op": "add", "kind": "sketch", "name": "Profile",
         "definition": {"plane": "XY", "entities": [...], "constraints": [...]}},
        {"op": "add", "kind": "part", "name": "Bracket", "definition": {"features": [
          {"type": "extrude", "id": "base", "sketch": "Profile", "extent": {...}}]}},
        {"op": "append", "item": "Bracket", "list": "features",
         "values": [{"type": "hole", "id": "h1", "face": "base:end", ...}]},
        {"op": "update", "item": "Profile", "list": "constraints", "id": "d_width",
         "changes": {"value": "=width"}},
        {"op": "show", "item": "Bracket"},
        {"op": "measure", "a": {"part": "Bracket", "face": "base:end"}},
        {"op": "expect", "measure": {"a": {"part": "Bracket", "face": "base:end"},
                                      "b": {"part": "Bracket", "face": "base:start"}},
         "checks": {"parallel_distance": {"value": 10, "tolerance": 1e-9}}}
      ]
    }

Operations (``op``; every other key is an argument, checked against the operation's own
list): ``units``, ``parameter``, ``add``, ``replace`` (a whole definition), ``merge``
(top-level keys; ``null`` removes one), ``append`` / ``update`` / ``delete`` (elements of
a definition's list by ``id`` -- or ``name`` for a building's levels: a sketch's entities
and constraints, a part's features, a floor's walls, openings and rooms, an assembly's
components and mates, a drawing's items), ``rename``,
``remove``, ``show``, ``convert_to_mesh``, and the read-only ``info``, ``measure``,
``snap``, ``schedule`` and ``expect`` (checks of measurements or ``info`` values; a
failed check aborts the script, rolling everything back).

:func:`run_cad_script` applies everything or nothing as one step of the scene's history,
within ``limits.max_seconds`` (regeneration included) and ``limits.max_items``. Errors are
:class:`~aether3d.errors.CadError` with a stable ``code`` and the ``location`` of the
step (``operations[3].definition.features[1]``).
"""

from __future__ import annotations

import copy
import math
import time
from typing import TYPE_CHECKING, Any

from ..errors import AetherError, CadError

if TYPE_CHECKING:
    from ..scene import Scene

FORMAT_ID = "aether3d.cad-script"
VERSION = 1
MAX_OPERATIONS = 10_000
LIMIT_KEYS = ("max_seconds", "max_items")
DEFAULT_LIMITS = {"max_seconds": 600.0, "max_items": 5000}

# Lists of a definition whose elements are objects with an "id".
LISTS = {
    "sketch": ("entities", "constraints"),
    "part": ("features",),
    "assembly": ("components", "mates"),
    "floor": ("walls", "openings", "rooms"),
    "building": ("levels", "grids"),
    "drawing": ("items", "dimensions"),
    "beams": ("lines",),
}

OPERATIONS: dict[str, tuple[set[str], set[str]]] = {
    # op: (required keys, optional keys) -- besides "op"
    "units": (set(), {"length", "angle"}),
    "parameter": ({"name", "value"}, {"dimension"}),
    "add": ({"kind", "name", "definition"}, set()),
    "replace": ({"item", "definition"}, set()),
    "merge": ({"item", "changes"}, set()),
    "append": ({"item", "list", "values"}, set()),
    "update": ({"item", "list", "id", "changes"}, set()),
    "delete": ({"item", "list", "ids"}, set()),
    "rename": ({"item", "name"}, set()),
    "remove": ({"item"}, set()),
    "show": ({"item"}, {"name"}),
    "convert_to_mesh": ({"object"}, set()),
    "info": (set(), {"item"}),
    "measure": ({"a"}, {"b"}),
    "snap": ({"sketch", "x", "y"}, {"radius", "kinds", "grid"}),
    "schedule": ({"building"}, {"kind"}),
    "expect": ({"checks"}, {"measure", "info", "message"}),
}
READ_ONLY = ("info", "measure", "snap", "schedule", "expect")


def _err(message: str, location: str, code: str = "INVALID_SCRIPT", **kw: Any) -> CadError:
    return CadError(code, message, location, **kw)


def _dict(v: Any, loc: str) -> dict:
    if not isinstance(v, dict):
        raise _err("expected an object", loc)
    return v


def _keys(obj: dict, allowed: set[str], loc: str, required: set[str] = frozenset()) -> None:
    unknown = sorted(set(obj) - allowed - required)
    if unknown:
        raise _err(f"unknown key(s): {', '.join(unknown)}", loc)
    missing = sorted(required - set(obj))
    if missing:
        raise _err(f"missing key(s): {', '.join(missing)}", loc)


def _limits(raw: Any) -> dict[str, float]:
    raw = _dict(raw, "script.limits")
    _keys(raw, set(LIMIT_KEYS), "script.limits")
    out = dict(DEFAULT_LIMITS)
    for key, value in raw.items():
        integer = key != "max_seconds"
        ok = isinstance(value, int) if integer else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok or not math.isfinite(value) or value <= 0:
            kind = "an integer" if integer else "a number"
            raise _err(f"{key} must be {kind} > 0", f"script.limits.{key}")
        out[key] = value
    return out


def check_script(script: Any) -> dict:
    """Check a script's structure (everything that does not need the document)."""
    script = _dict(script, "script")
    _keys(script, {"label", "limits"}, "script", {"format", "version", "operations"})
    if script["format"] != FORMAT_ID:
        raise _err(f"format must be {FORMAT_ID!r}", "script.format")
    if isinstance(script["version"], bool) or script["version"] != VERSION:
        raise _err(
            f"unsupported version {script['version']!r} (supported: {VERSION})", "script.version"
        )
    if "label" in script and (not isinstance(script["label"], str) or not script["label"].strip()):
        raise _err("label must be a non-empty string", "script.label")
    _limits(script.get("limits", {}))
    ops = script["operations"]
    if not isinstance(ops, list) or not ops:
        raise _err("operations must be a non-empty array", "script.operations")
    if len(ops) > MAX_OPERATIONS:
        raise _err(f"too many operations ({len(ops)} > {MAX_OPERATIONS})", "script.operations")
    for k, step in enumerate(ops):
        loc = f"operations[{k}]"
        step = _dict(step, loc)
        op = step.get("op")
        if not isinstance(op, str) or op not in OPERATIONS:
            raise _err(f"op must be one of {', '.join(OPERATIONS)}", f"{loc}.op")
        required, optional = OPERATIONS[op]
        _keys(step, optional | {"op"}, loc, required)
    return script


def _relocate(exc: AetherError, loc: str) -> CadError:
    code = getattr(exc, "code", None) or type(exc).__name__
    message = getattr(exc, "message", None) or str(exc)
    inner = getattr(exc, "location", None)
    where = f"{loc}: {inner}" if isinstance(inner, str) and inner else loc
    details = getattr(exc, "details", None)
    return CadError(
        code,
        message,
        where,
        hint=getattr(exc, "hint", None),
        details=details if isinstance(details, dict) else None,
    )


def _list_of(doc, step, loc):
    item = doc.item(step["item"])
    name = step["list"]
    allowed = LISTS.get(item.kind, ())
    if name not in allowed:
        raise _err(
            f"a {item.kind} has no editable list {name!r}",
            f"{loc}.list",
            hint=", ".join(allowed) or f"{item.kind} definitions have no lists",
        )
    defn = copy.deepcopy(item._definition)
    lst = defn.setdefault(name, [])
    if not isinstance(lst, list):
        raise _err(f"{name} is not a list in this definition", f"{loc}.list")
    return item, defn, lst


def _element_index(lst, eid, loc):
    for k, e in enumerate(lst):
        if (isinstance(e, dict) and e.get("id", e.get("name")) == eid) or e == eid:
            return k
    ids = [e.get("id", e.get("name")) if isinstance(e, dict) else e for e in lst]
    raise _err(
        f"no element {eid!r}", loc, "UNKNOWN_ELEMENT", hint=", ".join(str(i) for i in ids[:50])
    )


def _get_path(data: Any, path: str, loc: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise _err(f"no value at {path!r}", loc, "EXPECTATION_INVALID")
    return cur


def _check_value(name: str, actual: Any, spec: Any, loc: str) -> dict:
    if not isinstance(spec, dict):
        spec = {"equals": spec}
    _keys(spec, {"value", "tolerance", "min", "max", "equals"}, loc)
    ok = True
    if "equals" in spec:
        ok = actual == spec["equals"]
    if "value" in spec:
        tol = float(spec.get("tolerance", 0.0))
        ok = (
            ok
            and isinstance(actual, (int, float))
            and abs(float(actual) - float(spec["value"])) <= tol
        )
    if "min" in spec:
        ok = ok and isinstance(actual, (int, float)) and float(actual) >= float(spec["min"])
    if "max" in spec:
        ok = ok and isinstance(actual, (int, float)) and float(actual) <= float(spec["max"])
    return {"check": name, "actual": actual, "expected": spec, "pass": bool(ok)}


def _length(doc, v, loc) -> float:
    from .units import LENGTH

    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise _err("a length is a number (document unit) or a quantity", loc)
    return doc.evaluate_value(v, LENGTH, loc)


def _sketch_entities_in(doc, entities, loc):
    """Sketch geometry in scripts is in the document's unit (like every length in a
    script); the stored definition (the solver's state) is in metres."""
    if not isinstance(entities, list):
        return entities
    out = []
    for k, e in enumerate(entities):
        if not isinstance(e, dict):
            out.append(e)
            continue
        e = dict(e)
        if "at" in e:
            at = e["at"]
            if not isinstance(at, list) or len(at) != 2:
                raise _err("at is [x, y]", f"{loc}[{k}].at")
            e["at"] = [_length(doc, v, f"{loc}[{k}].at[{i}]") for i, v in enumerate(at)]
        if "radius" in e and e.get("type") in ("circle", "arc"):
            e["radius"] = _length(doc, e["radius"], f"{loc}[{k}].radius")
        out.append(e)
    return out


def _sketch_constraints_in(doc, constraints, loc):
    if not isinstance(constraints, list):
        return constraints
    out = []
    for k, c in enumerate(constraints):
        if isinstance(c, dict) and c.get("type") == "fix" and isinstance(c.get("at"), list):
            c = dict(c)
            c["at"] = [
                [_length(doc, v, f"{loc}[{k}].at[{j}][{i}]") for i, v in enumerate(p)]
                if isinstance(p, list)
                else p
                for j, p in enumerate(c["at"])
            ]
        out.append(c)
    return out


def _sketch_in(doc, kind, defn, loc):
    if kind != "sketch" or not isinstance(defn, dict):
        return defn
    defn = dict(defn)
    if "entities" in defn:
        defn["entities"] = _sketch_entities_in(doc, defn["entities"], f"{loc}.entities")
    if "constraints" in defn:
        defn["constraints"] = _sketch_constraints_in(doc, defn["constraints"], f"{loc}.constraints")
    return defn


def _list_in(doc, item, name, values, loc):
    if item.kind != "sketch":
        return values
    if name == "entities":
        return _sketch_entities_in(doc, values, loc)
    return _sketch_constraints_in(doc, values, loc)


def _run(doc, scene, step: dict, loc: str) -> Any:
    from .measure import measure, snap

    op = step["op"]
    if op == "units":
        doc.set_units(length=step.get("length"), angle=step.get("angle"))
        return {"length": doc.length_unit}
    if op == "parameter":
        item = doc.parameter(step["name"], step["value"], dimension=step.get("dimension", "length"))
        return {"item": item.name}
    if op == "add":
        if not isinstance(step["definition"], dict):
            raise _err("definition must be an object", f"{loc}.definition")
        defn = _sketch_in(doc, step["kind"], step["definition"], f"{loc}.definition")
        item = doc.add(step["kind"], step["name"], defn)
        return {"item": item.name, "kind": item.kind}
    if op == "replace":
        item = doc.item(step["item"])
        defn = _sketch_in(
            doc, item.kind, _dict(step["definition"], f"{loc}.definition"), f"{loc}.definition"
        )
        doc.set_definition(item, defn)
        return {"item": doc.item(step["item"]).name}
    if op == "merge":
        item = doc.item(step["item"])
        changes = _sketch_in(
            doc, item.kind, _dict(step["changes"], f"{loc}.changes"), f"{loc}.changes"
        )
        doc.update(item, changes)
        return {"item": doc.item(step["item"]).name}
    if op == "append":
        item, defn, lst = _list_of(doc, step, loc)
        values = step["values"]
        if not isinstance(values, list) or not values:
            raise _err("values must be a non-empty array", f"{loc}.values")
        lst.extend(copy.deepcopy(_list_in(doc, item, step["list"], values, f"{loc}.values")))
        doc.set_definition(item, defn)
        return {"item": item.name, "count": len(lst)}
    if op == "update":
        item, defn, lst = _list_of(doc, step, loc)
        k = _element_index(lst, step["id"], f"{loc}.id")
        if not isinstance(lst[k], dict):
            raise _err("only object elements can be updated", f"{loc}.id")
        changes = _dict(step["changes"], f"{loc}.changes")
        if item.kind == "sketch":
            changes = _list_in(doc, item, step["list"], [{**lst[k], **changes}], f"{loc}.changes")[
                0
            ]
            changes = {key: changes[key] for key in step["changes"]}
        for key, v in changes.items():
            if key == "id":
                raise _err("an element's id cannot change", f"{loc}.changes.id")
            if v is None:
                lst[k].pop(key, None)
            else:
                lst[k][key] = copy.deepcopy(v)
        doc.set_definition(item, defn)
        return {"item": item.name, "id": step["id"]}
    if op == "delete":
        item, defn, lst = _list_of(doc, step, loc)
        ids = step["ids"]
        if not isinstance(ids, list) or not ids:
            raise _err("ids must be a non-empty array", f"{loc}.ids")
        for eid in ids:
            del lst[_element_index(lst, eid, f"{loc}.ids")]
        doc.set_definition(item, defn)
        return {"item": item.name, "count": len(lst)}
    if op == "rename":
        doc.rename(step["item"], step["name"])
        return {"item": step["name"]}
    if op == "remove":
        name = doc.item(step["item"]).name
        doc.remove(step["item"])
        return {"removed": name}
    if op == "show":
        obj = doc.show(step["item"], name=step.get("name"))
        return {"object": obj.name, "id": obj.id}
    if op == "convert_to_mesh":
        return doc.convert_to_mesh(step["object"])
    if op == "info":
        return doc.info(step.get("item"))
    if op == "measure":
        return measure(doc, step["a"], step.get("b"))
    if op == "snap":
        return snap(
            doc,
            step["sketch"],
            step["x"],
            step["y"],
            radius=step.get("radius"),
            kinds=step.get(
                "kinds",
                ("endpoint", "intersection", "center", "midpoint", "quadrant", "nearest", "grid"),
            ),
            grid=step.get("grid"),
        )
    if op == "schedule":
        from ..arch.api import BuildingHandle

        handle = BuildingHandle(doc, doc.item(step["building"], "building"))
        return handle.schedule(step.get("kind", "all"))
    if op == "expect":
        if ("measure" in step) == ("info" in step):
            raise _err("give exactly one of measure or info", loc)
        if "measure" in step:
            m = _dict(step["measure"], f"{loc}.measure")
            _keys(m, {"b"}, f"{loc}.measure", {"a"})
            data = measure(doc, m["a"], m.get("b"))
        else:
            data = doc.info(step["info"])
        checks = _dict(step["checks"], f"{loc}.checks")
        if not checks or len(checks) > 100:
            raise _err("checks needs 1 to 100 entries", f"{loc}.checks")
        results = [
            _check_value(k, _get_path(data, k, f"{loc}.checks.{k}"), v, f"{loc}.checks.{k}")
            for k, v in checks.items()
        ]
        failed = [r for r in results if not r["pass"]]
        if failed:
            raise CadError(
                "EXPECTATION_FAILED",
                step.get("message")
                or "; ".join(
                    f"{r['check']} is {r['actual']!r}, expected {r['expected']}" for r in failed
                ),
                loc,
                details={"checks": results},
            )
        return {"checks": results}
    raise _err(f"unknown op {op!r}", f"{loc}.op")  # pragma: no cover - checked before


def run_cad_script(
    scene: Scene, script: dict[str, Any], *, label: str | None = None, should_stop=None
) -> dict[str, Any]:
    """Apply a CAD script to ``scene``'s document (everything or nothing). ``should_stop``
    (a callable) cancels it (``CANCELLED``, rolled back). Returns each step's result,
    the items and the regeneration statistics."""
    script = check_script(script)
    limits = _limits(script.get("limits", {}))
    ops = script["operations"]
    label = label or script.get("label") or f"CAD script ({len(ops)} operations)"
    doc = scene.cad
    deadline = time.monotonic() + float(limits["max_seconds"])
    results = []
    t0 = time.perf_counter()
    before = dict(doc.stats)
    with doc.batch(label, deadline=deadline, should_stop=should_stop):
        for k, step in enumerate(ops):
            loc = f"operations[{k}]"
            if should_stop is not None and should_stop():
                raise CadError("CANCELLED", "the CAD script was cancelled; nothing changed", loc)
            if time.monotonic() > deadline:
                raise CadError(
                    "BUDGET_EXCEEDED",
                    f"the script ran longer than max_seconds ({limits['max_seconds']} s)",
                    loc,
                    hint="raise limits.max_seconds or split the script",
                )
            try:
                result = _run(doc, scene, step, loc)
                if step["op"] not in READ_ONLY:
                    # Regenerate now (incrementally), so a failure names this step.
                    doc._regenerate()
            except CadError as exc:
                if exc.location and exc.location.startswith("operations["):
                    raise
                raise _relocate(exc, loc) from None
            except AetherError as exc:
                raise _relocate(exc, loc) from None
            if len(doc.items()) > limits["max_items"]:
                raise CadError(
                    "LIMIT_EXCEEDED",
                    f"the document would hold more than max_items ({limits['max_items']}) items",
                    loc,
                )
            results.append({"op": step["op"], **({"result": result} if result is not None else {})})
    stats = {
        k: v - before.get(k, 0) if isinstance(v, (int, float)) else v for k, v in doc.stats.items()
    }
    return {
        "operations": results,
        "items": [{"name": i.name, "kind": i.kind} for i in doc.items()],
        "regeneration": stats,
        "seconds": time.perf_counter() - t0,
    }
