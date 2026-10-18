"""World tools of the Aether3D MCP server.

Building and changing worlds goes through ``run_script`` with an ``aether3d.world-script``
(one undo step, all or nothing; schema ``aether3d://schema/world-script``, guide
``aether3d://worlds/guide``). ``inspect_world`` answers read-only questions: ``summary``
(every world), ``info`` (one world), ``sample``, ``profile``, ``sites``, ``route``,
``block``, ``stats``.
"""

from __future__ import annotations

from typing import Any

from .session import Session, ToolError

WHAT = ("summary", "info", "sample", "profile", "sites", "route", "block", "stats")


def inspect_world(session: Session, args: dict, ctx: Any):
    from .tools import ToolOutput, scene_status

    scene = session.require_scene()
    what = args.get("what", "summary")
    if what == "summary":
        worlds = scene.worlds
        data = {
            "worlds": [
                {
                    "name": w.name,
                    "id": w.id,
                    "mode": w.mode,
                    "seed": w.seed,
                    "extent": None if w.extent is None else list(w.extent),
                    "features": len(w.features()),
                    "structures": len(w.structures()),
                }
                for w in worlds
            ],
            "presets": sorted(worlds.presets()),
        }
        return ToolOutput({"what": what, **data, "scene": scene_status(session)})
    if not args.get("world"):
        raise ToolError("INVALID_ARGUMENT", f"{what} needs world (a name or id)")
    query = {"query": what, "world": args["world"]}
    for key in ("points", "fields", "samples", "count", "options", "from", "to", "at"):
        if key in args:
            query[key] = args[key]
    from ..worlds.script import FORMAT, run_world_script

    out = run_world_script(scene, {"format": FORMAT, "queries": [query]})
    return ToolOutput({"what": what, **out["queries"][0], "scene": scene_status(session)})


def specs() -> tuple[Any, ...]:
    from .tools import ToolSpec, _schema

    point = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3}
    return (
        ToolSpec(
            "inspect_world",
            "Inspect procedural worlds",
            "Read-only world questions: summary (all worlds, presets); info; sample (points: "
            "height, slope, biome, water...); profile (along points); sites (building sites); "
            "route (a road plan from/to); block (at, block worlds); stats. Change worlds "
            "with run_script (aether3d.world-script).",
            _schema(
                {
                    "what": {"enum": list(WHAT)},
                    "world": {"type": "string"},
                    "points": {"type": "array", "items": point},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "samples": {"type": "integer", "minimum": 2},
                    "count": {"type": "integer", "minimum": 1},
                    "options": {"type": "object"},
                    "from": point,
                    "to": point,
                    "at": point,
                }
            ),
            inspect_world,
            read_only=True,
            idempotent=True,
        ),
    )
