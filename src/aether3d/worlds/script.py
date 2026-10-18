"""World scripts: ``aether3d.world-script`` documents that build and change worlds.

A script is data, applied atomically (one undo step; on any error nothing changes)::

    {
      "format": "aether3d.world-script", "version": 1,
      "label": "Valley with a village",            (optional: the undo step's label)
      "world": "Valley",                            (optional: the default world)
      "limits": {"max_seconds": 120},               (optional)
      "operations": [
        {"op": "create", "name": "Valley", "preset": "alpine_valley", "seed": 7},
        {"op": "add_feature", "feature": {"type": "lake", "id": "tarn", ...}},
        {"op": "build_settlements", "count": 2, "houses": [6, 12]},
        ...
      ],
      "queries": [{"query": "sample", "points": [[0, 0]]}, ...]   (optional, read-only)
    }

Operations (``world`` names the world, default the script's ``world`` or the one just
created): ``create`` (``name``, ``preset``, ``seed``, ``size``, ``extent``, ``mode``,
``overrides``), ``remove``, ``rename`` (``name``), ``configure`` (``patch``),
``add_feature`` (``feature``), ``update_feature`` (``id``, ``patch``), ``remove_feature``
(``id``), ``place_structure`` (``item``), ``remove_structure`` (``id``),
``build_settlements`` (``count``, ``houses``, ``roads``, ``seed``, ``options``),
``build_road`` (``from``, ``to``, ``name``), and for block worlds ``set_blocks``
(``cells``, ``block``), ``fill_blocks`` / ``replace_blocks`` / ``revert_blocks`` (``min``,
``max``, ``block``, ``source``).

Queries run after the operations, inside the same call, and change nothing: ``info``,
``sample`` (``points``, ``fields``), ``profile`` (``points``, ``samples``), ``sites``
(``count``, ``options``), ``route`` (``from``, ``to``), ``block`` (``at``), ``stats``,
``structures``, ``features``. Errors are :class:`~aether3d.errors.WorldError` with a
stable code and a location such as ``operations[2].feature.shape``.
"""

from __future__ import annotations

import time
from typing import Any

from ..errors import AetherError, WorldError

FORMAT = "aether3d.world-script"
VERSION = 1
MAX_OPERATIONS = 512
MAX_QUERIES = 64
DEFAULT_SECONDS = 120.0
MAX_SECONDS = 3600.0

OPERATIONS: dict[str, tuple[set[str], set[str]]] = {
    # op: (required keys, optional keys) besides "op" and "world"
    "create": ({"name"}, {"preset", "seed", "size", "extent", "mode", "overrides", "materials"}),
    "remove": (set(), set()),
    "rename": ({"name"}, set()),
    "configure": ({"patch"}, {"label"}),
    "add_feature": ({"feature"}, set()),
    "update_feature": ({"id", "patch"}, set()),
    "remove_feature": ({"id"}, set()),
    "place_structure": ({"item"}, set()),
    "remove_structure": ({"id"}, set()),
    "build_settlements": (set(), {"count", "houses", "roads", "seed", "options"}),
    "build_road": ({"from", "to"}, {"name", "cell"}),
    "set_blocks": ({"cells", "block"}, set()),
    "fill_blocks": ({"min", "max", "block"}, set()),
    "replace_blocks": ({"min", "max", "source", "block"}, set()),
    "revert_blocks": ({"min", "max"}, set()),
}
QUERIES: dict[str, tuple[set[str], set[str]]] = {
    "info": (set(), set()),
    "sample": ({"points"}, {"fields"}),
    "profile": ({"points"}, {"samples"}),
    "sites": (set(), {"count", "options"}),
    "route": ({"from", "to"}, {"cell"}),
    "block": ({"at"}, set()),
    "stats": (set(), set()),
    "structures": (set(), {"kind"}),
    "features": (set(), {"kind"}),
}


def _bad(message: str, location: str, code: str = "INVALID_SCRIPT", hint: str | None = None):
    return WorldError(code, message, location, hint=hint)


def check_script(script: Any) -> dict[str, Any]:
    """Validate a script's structure (operation names, keys, types of the envelope);
    operation arguments are checked when they run, before anything changes."""
    if not isinstance(script, dict):
        raise _bad("a world script is a JSON object", "")
    allowed = {"format", "version", "label", "world", "limits", "operations", "queries"}
    for key in script:
        if key not in allowed:
            raise _bad(f"unknown key {key!r}", key, hint="keys: " + ", ".join(sorted(allowed)))
    if script.get("format") != FORMAT:
        raise _bad(f"format must be {FORMAT!r}", "format")
    if script.get("version", VERSION) != VERSION:
        raise _bad(f"unsupported version {script.get('version')!r} (this is {VERSION})", "version")
    for key in ("label", "world"):
        if key in script and (not isinstance(script[key], str) or not script[key]):
            raise _bad("expected a non-empty string", key)
    limits = script.get("limits", {})
    if not isinstance(limits, dict) or set(limits) - {"max_seconds"}:
        raise _bad("limits is {max_seconds}", "limits")
    if "max_seconds" in limits:
        v = limits["max_seconds"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= MAX_SECONDS:
            raise _bad(f"max_seconds is a number in (0, {MAX_SECONDS}]", "limits.max_seconds")
    ops = script.get("operations", [])
    if not isinstance(ops, list) or len(ops) > MAX_OPERATIONS:
        raise _bad(f"operations is a list of at most {MAX_OPERATIONS}", "operations")
    for k, step in enumerate(ops):
        _check_step(step, f"operations[{k}]", "op", OPERATIONS)
    queries = script.get("queries", [])
    if not isinstance(queries, list) or len(queries) > MAX_QUERIES:
        raise _bad(f"queries is a list of at most {MAX_QUERIES}", "queries")
    for k, step in enumerate(queries):
        _check_step(step, f"queries[{k}]", "query", QUERIES)
    if not ops and not queries:
        raise _bad("a script needs operations or queries", "operations")
    return script


def _check_step(step: Any, loc: str, key: str, table: dict) -> None:
    if not isinstance(step, dict):
        raise _bad("expected an object", loc)
    name = step.get(key)
    if not isinstance(name, str) or name not in table:
        raise _bad(f"unknown {key} {name!r}", f"{loc}.{key}", hint=", ".join(table))
    required, optional = table[name]
    keys = set(step) - {key, "world"}
    missing = sorted(required - keys)
    if missing:
        raise _bad(f"{name} needs {', '.join(missing)}", loc)
    unknown = sorted(keys - required - optional)
    if unknown:
        raise _bad(
            f"unknown key {unknown[0]!r} for {name}",
            f"{loc}.{unknown[0]}",
            hint="keys: " + ", ".join(sorted(required | optional)),
        )
    if "world" in step and (not isinstance(step["world"], str) or not step["world"]):
        raise _bad("world is a world's name or id", f"{loc}.world")


def run_world_script(
    scene: Any, script: dict[str, Any], *, label: str | None = None, should_stop: Any = None
) -> dict[str, Any]:
    """Apply a world script to ``scene`` (everything or nothing). Returns
    ``{"operations": [...], "queries": [...], "worlds": [...]}``."""
    script = check_script(script)
    ops = script.get("operations", [])
    limit = float(script.get("limits", {}).get("max_seconds", DEFAULT_SECONDS))
    deadline = time.monotonic() + limit
    worlds = scene.worlds
    current = {"name": script.get("world")}
    results: list[Any] = []

    def step_check(loc: str) -> None:
        if should_stop is not None and should_stop():
            raise _bad("the script was cancelled", loc, "CANCELLED")
        if time.monotonic() > deadline:
            raise _bad(
                f"the script ran longer than max_seconds ({limit} s)",
                loc,
                "BUDGET_EXCEEDED",
                hint="raise limits.max_seconds or split the script",
            )

    if ops:
        text = label or script.get("label") or f"World script ({len(ops)} operations)"
        with scene.history.transaction(text):
            for k, step in enumerate(ops):
                loc = f"operations[{k}]"
                step_check(loc)
                try:
                    results.append(_operation(worlds, step, current, loc))
                except WorldError as exc:
                    raise _located(exc, loc) from None
                except AetherError:
                    raise
                except (ValueError, TypeError, KeyError) as exc:
                    raise _bad(str(exc), loc, "INVALID_ARGUMENT") from None
    answers = []
    for k, q in enumerate(script.get("queries", [])):
        loc = f"queries[{k}]"
        step_check(loc)
        try:
            answers.append(_query(worlds, q, current, loc))
        except WorldError as exc:
            raise _located(exc, loc) from None
        except (ValueError, TypeError, KeyError) as exc:
            raise _bad(str(exc), loc, "INVALID_ARGUMENT") from None
    return {
        "operations": results,
        "queries": answers,
        "worlds": [{"id": w.id, "name": w.name, "mode": w.mode} for w in worlds],
    }


def _located(exc: WorldError, loc: str) -> WorldError:
    where = exc.location or ""
    if where.startswith(loc):
        return exc
    inner = where.split(".", 1)[1] if where.startswith("worlds[") and "." in where else where
    return WorldError(
        exc.code,
        exc.message,
        f"{loc}.{inner}" if inner else loc,
        hint=exc.hint,
        details=getattr(exc, "details", None),
    )


def _world(worlds: Any, step: dict, current: dict, loc: str) -> Any:
    ref = step.get("world") or current["name"]
    if ref is None:
        raise _bad("no world: give world (a name or id), or create one first", f"{loc}.world")
    try:
        return worlds.get(ref)
    except WorldError:
        raise _bad(
            f"no world {ref!r}",
            f"{loc}.world",
            "UNKNOWN_WORLD",
            hint="worlds: " + (", ".join(worlds.names()) or "none"),
        ) from None


def _operation(worlds: Any, step: dict, current: dict, loc: str) -> Any:
    op = step["op"]
    if op == "create":
        args = {
            k: step[k]
            for k in ("seed", "size", "extent", "mode", "overrides", "materials")
            if k in step
        }
        w = worlds.create(step["name"], step.get("preset"), **args)
        current["name"] = w.name
        return {"op": op, "world": w.name, "id": w.id, "mode": w.mode, "seed": w.seed}
    w = _world(worlds, step, current, loc)
    if op == "remove":
        worlds.remove(w)
        if current["name"] in (w.name, w.id):
            current["name"] = None
        return {"op": op, "world": w.name}
    if op == "rename":
        old = w.name
        w.rename(step["name"])
        if current["name"] in (old, w.id):
            current["name"] = w.name
        return {"op": op, "world": w.name, "was": old}
    if op == "configure":
        w.configure(step["patch"], label=step.get("label"))
        return {"op": op, "world": w.name}
    if op == "add_feature":
        return {"op": op, "world": w.name, "feature": w.add_feature(step["feature"])}
    if op == "update_feature":
        return {"op": op, "world": w.name, "feature": w.update_feature(step["id"], step["patch"])}
    if op == "remove_feature":
        w.remove_feature(step["id"])
        return {"op": op, "world": w.name, "id": step["id"]}
    if op == "place_structure":
        return {"op": op, "world": w.name, "item": w.place_structure(step["item"])}
    if op == "remove_structure":
        w.remove_structure(step["id"])
        return {"op": op, "world": w.name, "id": step["id"]}
    if op == "build_settlements":
        options = step.get("options", {})
        if not isinstance(options, dict):
            raise _bad("options is an object", f"{loc}.options")
        out = w.build_settlements(
            step.get("count"),
            houses=step.get("houses"),
            roads=step.get("roads"),
            seed=step.get("seed", 0),
            **options,
        )
        return {"op": op, "world": w.name, **out}
    if op == "build_road":
        out = w.build_road(
            step["from"], step["to"], name=step.get("name", "road"), cell=step.get("cell", 16.0)
        )
        return {
            "op": op,
            "world": w.name,
            "roads": len(out["roads"]),
            "bridges": len(out["items"]),
            "length": out["length"],
        }
    if op == "set_blocks":
        return {"op": op, "world": w.name, "cells": w.set_blocks(step["cells"], step["block"])}
    if op == "fill_blocks":
        return {
            "op": op,
            "world": w.name,
            "cells": w.fill_blocks(step["min"], step["max"], step["block"]),
        }
    if op == "replace_blocks":
        n = w.replace_blocks(step["min"], step["max"], step["source"], step["block"])
        return {"op": op, "world": w.name, "cells": n}
    if op == "revert_blocks":
        return {"op": op, "world": w.name, "cells": w.revert_blocks(step["min"], step["max"])}
    raise AssertionError(op)  # pragma: no cover - checked before


def _query(worlds: Any, q: dict, current: dict, loc: str) -> Any:
    kind = q["query"]
    w = _world(worlds, q, current, loc)
    if kind == "info":
        return {"query": kind, "world": w.name, **w.info()}
    if kind == "sample":
        return {"query": kind, "world": w.name, **w.sample(q["points"], fields=q.get("fields"))}
    if kind == "profile":
        return {"query": kind, "world": w.name, **w.profile(q["points"], q.get("samples", 64))}
    if kind == "sites":
        options = q.get("options", {})
        if not isinstance(options, dict):
            raise _bad("options is an object", f"{loc}.options")
        return {"query": kind, "world": w.name, "sites": w.find_sites(q.get("count", 5), **options)}
    if kind == "route":
        r = w.route(q["from"], q["to"], cell=q.get("cell", 16.0))
        return {"query": kind, "world": w.name, **r}
    if kind == "block":
        at = q["at"]
        if not isinstance(at, list) or len(at) != 3:
            raise _bad("at is [x, y, z] (metres)", f"{loc}.at")
        return {"query": kind, "world": w.name, "block": w.block_at(*at)}
    if kind == "stats":
        return {"query": kind, "world": w.name, **w.runtime.stats()}
    if kind == "structures":
        return {"query": kind, "world": w.name, "items": w.structures(q.get("kind"))}
    if kind == "features":
        return {"query": kind, "world": w.name, "features": w.features(q.get("kind"))}
    raise AssertionError(kind)  # pragma: no cover - checked before


def schema() -> dict[str, Any]:
    """The JSON Schema (2020-12) of world scripts (envelope and operation keys; values
    are validated by the engine with located errors)."""
    any_obj = {"type": "object"}
    point2 = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    point3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    value_schemas = {
        "name": {"type": "string", "minLength": 1, "maxLength": 64},
        "preset": {"type": "string"},
        "seed": {"type": "integer", "minimum": 0},
        "size": {"type": "number", "exclusiveMinimum": 0},
        "extent": {"type": ["object", "null"]},
        "mode": {"enum": ["terrain", "blocks"]},
        "overrides": any_obj,
        "materials": {"type": "boolean"},
        "patch": any_obj,
        "label": {"type": "string"},
        "feature": any_obj,
        "item": any_obj,
        "id": {"type": "string", "minLength": 1},
        "count": {"type": "integer", "minimum": 0},
        "houses": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "roads": {"type": "boolean"},
        "options": any_obj,
        "from": point2,
        "to": point2,
        "cell": {"type": "number", "exclusiveMinimum": 0},
        "cells": {"type": "array", "items": point3},
        "block": {"type": "string"},
        "source": {"type": "string"},
        "min": point3,
        "max": point3,
        "points": {"type": "array", "items": point2},
        "fields": {"type": "array", "items": {"type": "string"}},
        "samples": {"type": "integer", "minimum": 2},
        "at": point3,
        "kind": {"type": "string"},
        "world": {"type": "string", "minLength": 1},
    }

    def variants(table: dict, key: str) -> list[dict]:
        out = []
        for name, (required, optional) in table.items():
            props = {key: {"const": name}, "world": value_schemas["world"]}
            for k in sorted(required | optional):
                props[k] = value_schemas[k]
            out.append(
                {
                    "type": "object",
                    "properties": props,
                    "required": [key, *sorted(required)],
                    "additionalProperties": False,
                }
            )
        return out

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "aether3d://schema/world-script",
        "title": "aether3d.world-script version 1",
        "type": "object",
        "properties": {
            "format": {"const": FORMAT},
            "version": {"const": VERSION},
            "label": {"type": "string", "minLength": 1},
            "world": {"type": "string", "minLength": 1},
            "limits": {
                "type": "object",
                "properties": {
                    "max_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_SECONDS}
                },
                "additionalProperties": False,
            },
            "operations": {
                "type": "array",
                "maxItems": MAX_OPERATIONS,
                "items": {"oneOf": variants(OPERATIONS, "op")},
            },
            "queries": {
                "type": "array",
                "maxItems": MAX_QUERIES,
                "items": {"oneOf": variants(QUERIES, "query")},
            },
        },
        "required": ["format"],
        "additionalProperties": False,
    }
