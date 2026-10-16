"""``aether3d sculpt``: sculpt scripts and mesh validity from the command line (Stage 13).

* ``sculpt run PROJECT SCRIPT`` -- apply an ``aether3d.sculpt-script`` (strokes, masks,
  remeshing, symmetrize) as one undo step, all or nothing, and save.
* ``sculpt check PROJECT`` -- the validity tier and quality metrics of each object's
  mesh (read-only); ``--require TIER`` makes the exit status 3 when one falls below.
* ``sculpt schema`` -- the JSON Schema of sculpt scripts (brushes, operations, limits).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli import (
    EXIT_GEOMETRY_ERRORS,
    EXIT_OK,
    CommandError,
    CommandOutput,
    _load_scene,
    _project_only,
    _read_script,
    _script_label,
    _write_or_dry_run,
)


def cmd_sculpt_run(args: argparse.Namespace) -> CommandOutput:
    from .sculpt.script import SculptScriptError, run_sculpt_script

    path = Path(args.path)
    _project_only(path, "sculpt run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:
        result = run_sculpt_script(scene, script, label=_script_label(script, "sculpt run", args))
    except SculptScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("SculptScriptError", exc.message, exc.location, details) from None
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = []
    for entry in result["sculpts"]:
        commit = entry["commit"]
        mesh = commit["mesh"]
        summary = commit["report"]["summary"]
        lines.append(
            f"{entry['object']['name']}: {len(entry['operations'])} operations, "
            f"{'changed' if commit['changed'] else 'unchanged'} "
            f"({mesh['vertex_count']}v/{mesh['triangle_count']}t), tier {summary['tier']}"
        )
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_sculpt_check(args: argparse.Namespace) -> CommandOutput:
    from .errors import ObjectNotFoundError
    from .sculpt.validate import object_report, tier_index

    path = Path(args.path)
    scene, _ = _load_scene(path)
    if args.object:
        objs = []
        for ref in args.object:
            try:
                objs.append(scene.get(ref))
            except ObjectNotFoundError:
                try:
                    objs.append(scene.get_by_name(ref))
                except ObjectNotFoundError:
                    raise CommandError(
                        "ObjectNotFoundError", f"no object {ref!r}", "object"
                    ) from None
    else:
        objs = [o for o in scene if o.mesh is not None]
    reports = []
    below = []
    for obj in objs:
        if obj.mesh is None:
            raise CommandError("ValidationError", f"object {obj.name!r} has no mesh", "object")
        report = object_report(obj, self_intersections=args.self_intersections)
        reports.append(report)
        if args.require and tier_index(report["summary"]["tier"]) < tier_index(args.require):
            below.append(obj.name)
    data = {"path": str(path), "require": args.require, "below": below, "objects": reports}
    lines = []
    for r in reports:
        s = r["summary"]
        lines.append(
            f"{r['object']['name']}: tier {s['tier']}, {s['triangle_count']} triangles, "
            f"{s['components']} component(s), closed {s['closed']}, "
            f"min angle {s['min_angle_degrees']:.1f} deg"
            + (
                f", {s['self_intersecting_pairs']} self-intersecting pairs"
                if "self_intersecting_pairs" in s
                else ""
            )
        )
    if args.require:
        lines.append(
            f"below {args.require}: {', '.join(below)}" if below else f"all reach {args.require}"
        )
    if not args.self_intersections:
        lines.append("self-intersections not checked (--self-intersections); 'solid' needs them")
    return data, "\n".join(lines), EXIT_GEOMETRY_ERRORS if below else EXIT_OK


def cmd_sculpt_schema(args: argparse.Namespace) -> CommandOutput:
    import json

    from .sculpt.schema import sculpt_script_schema

    schema = sculpt_script_schema()
    return schema, json.dumps(schema, indent=2), EXIT_OK


def add_sculpt_parsers(sub, common, writing) -> None:
    from .sculpt.validate import TIERS

    p_sc = sub.add_parser("sculpt", help="sculpt scripts and mesh validity")
    sc = p_sc.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = sc.add_parser(
        "run",
        parents=[common, writing],
        help="apply a JSON sculpt script to a project (atomically)",
        description="Apply an aether3d.sculpt-script document (docs/MODELING.md): strokes, "
        "masks, remeshing and symmetrize on objects' meshes, as one step of the project's "
        "history (all or nothing), then save.",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("script", help="sculpt script (.json), or - to read it from stdin")
    p.set_defaults(handler=cmd_sculpt_run)
    p = sc.add_parser(
        "check",
        parents=[common],
        help="validity tier and quality of meshes (read-only)",
        description="Report each mesh's validity tier (valid < manifold < closed < solid), "
        "counts, components, volume, edge lengths and smallest angle, in world units.",
    )
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument("--object", action="append", help="object id or name (repeatable)")
    p.add_argument(
        "--self-intersections",
        action="store_true",
        help="run the exact self-intersection test (needed for the 'solid' tier)",
    )
    p.add_argument("--require", choices=TIERS, help="exit status 3 if a mesh is below this tier")
    p.set_defaults(handler=cmd_sculpt_check)
    p = sc.add_parser("schema", parents=[common], help="the JSON Schema of sculpt scripts")
    p.set_defaults(handler=cmd_sculpt_schema)
