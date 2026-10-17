"""``aether3d cad``: precision modeling from the command line (Stage 15).

* ``cad run PROJECT SCRIPT`` -- apply an ``aether3d.cad-script`` (sketches, parts,
  assemblies, buildings, drawings; measurements and expect checks) as one undo step, all
  or nothing, and save. A failed ``expect`` exits with status 3.
* ``cad info PROJECT [ITEM]`` -- the document's items, or one item's report (read-only).
* ``cad export PROJECT OUT [--item NAME ...]`` -- by OUT's suffix: ``.step``/``.stp``
  (parts), ``.ifc`` (a building), ``.dxf`` (a sketch or a drawing), ``.svg``/``.png``
  (a drawing).
* ``cad import PROJECT FILE`` -- a STEP, IFC or DXF file into the document (one undo
  step), then save.
* ``cad schema`` -- the JSON Schema of CAD scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import (
    EXIT_GEOMETRY_ERRORS,
    EXIT_OK,
    CommandError,
    CommandOutput,
    _check_destination,
    _load_scene,
    _project_only,
    _read_script,
    _script_label,
    _write_or_dry_run,
)
from .errors import AetherError

MAX_IMPORT_BYTES = 256 * 2**20


def _cad_error(exc: AetherError) -> CommandError:
    data = exc.to_dict() if hasattr(exc, "to_dict") else {}
    details = {k: v for k, v in data.items() if k not in ("message", "location")}
    return CommandError(
        "CadError", getattr(exc, "message", str(exc)), getattr(exc, "location", None), details
    )


def cmd_cad_run(args: argparse.Namespace) -> CommandOutput:
    from .cad.script import run_cad_script

    path = Path(args.path)
    _project_only(path, "cad run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:
        result = run_cad_script(scene, script, label=_script_label(script, "cad run", args))
    except AetherError as exc:
        err = _cad_error(exc)
        if getattr(exc, "code", None) == "EXPECTATION_FAILED":
            data = {"path": str(path), "error": exc.to_dict(), "saved": False}
            return (
                data,
                f"expectation failed: {exc.message} ({exc.location}); nothing saved",
                (EXIT_GEOMETRY_ERRORS),
            )
        raise err from None
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    lines = [f"{len(result['operations'])} operations in {result['seconds']:.2f} s"]
    for i, r in enumerate(result["operations"]):
        res = r.get("result")
        extra = ""
        if isinstance(res, dict) and "item" in res:
            extra = f": {res['item']}"
        elif r["op"] == "expect":
            extra = f": {len(res['checks'])} checks passed"
        lines.append(f"  [{i}] {r['op']}{extra}")
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_cad_info(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path)
    doc = scene.cad
    try:
        data = doc.info(args.item) if args.item else doc.info()
    except AetherError as exc:
        raise _cad_error(exc) from None
    if args.item:
        text = json.dumps(data, indent=2, default=str)
    else:
        lines = [f"{len(data['items'])} CAD items (units {doc.length_unit})"]
        for it in data["items"]:
            status = it.get("status") or ("error: " + it["error"]["code"] if "error" in it else "")
            lines.append(f"  {it['kind']:<9} {it['name']}" + (f"  [{status}]" if status else ""))
        text = "\n".join(lines)
    return {"path": str(path), **data}, text, EXIT_OK


def cmd_cad_export(args: argparse.Namespace) -> CommandOutput:
    from ._fileio import write_atomic

    path = Path(args.path)
    out = Path(args.out)
    _check_destination(out, args.force)
    scene, _ = _load_scene(path)
    doc = scene.cad
    suffix = out.suffix.lower()
    report = {}
    try:
        if suffix in (".step", ".stp"):
            data = doc.export_step(args.item or None, units=args.units).encode()
        elif suffix == ".ifc":
            text, stats = doc.export_ifc(building=args.item[0] if args.item else None)
            data, report = text.encode(), {"entities": stats}
        elif suffix in (".dxf", ".svg", ".png"):
            if not args.item or len(args.item) != 1:
                raise CommandError("ValidationError", f"{suffix} export needs one --item")
            it = doc.item(args.item[0])
            if suffix == ".dxf" and it.kind == "sketch":
                data = doc.export_sketch_dxf(it, args.units).encode()
            elif it.kind != "drawing":
                raise CommandError("ValidationError", f"{it.name!r} is a {it.kind}, not a drawing")
            else:
                d = doc.result(it)
                if suffix == ".dxf":
                    data = d.to_dxf(args.units or "mm").encode()
                elif suffix == ".svg":
                    data = d.to_svg().encode()
                else:
                    from .rendering.png import encode_png

                    data = encode_png(d.to_png(args.dpi))
                report = {"drawing": d.summary()}
        else:
            raise CommandError(
                "ValidationError", "OUT must end with .step, .stp, .ifc, .dxf, .svg or .png"
            )
    except AetherError as exc:
        raise _cad_error(exc) from None
    write_atomic(out, data)
    return (
        {"path": str(path), "out": str(out), "bytes": len(data), **report},
        f"wrote {out} ({len(data)} bytes)",
        EXIT_OK,
    )


def cmd_cad_import(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    _project_only(path, "cad import")
    src = Path(args.file)
    suffix = src.suffix.lower()
    if suffix not in (".step", ".stp", ".ifc", ".dxf"):
        raise CommandError("ValidationError", "FILE must be .step, .stp, .ifc or .dxf")
    try:
        size = src.stat().st_size
    except OSError as exc:
        raise CommandError("FileNotFoundError", f"cannot read {src}: {exc.strerror}") from None
    if size > MAX_IMPORT_BYTES:
        raise CommandError("ValidationError", f"{src} is larger than {MAX_IMPORT_BYTES} bytes")
    raw = src.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    scene, _ = _load_scene(path, history=True)
    doc = scene.cad
    try:
        if suffix in (".step", ".stp"):
            parts = doc.import_step(text, name=args.name)
            result = {"parts": [p.name for p in parts]}
        elif suffix == ".ifc":
            elements = doc.import_ifc(text)
            result = {
                "products": len(elements),
                "parts": sum(1 for e in elements if e.body is not None),
            }
        else:
            sk = doc.import_dxf(text, args.name or src.stem, unit=args.units)
            result = {"sketch": sk.name, **sk.import_report}
    except AetherError as exc:
        raise _cad_error(exc) from None
    written = _write_or_dry_run(scene, path, args)
    return (
        {"path": str(path), "output": written, "dry_run": args.dry_run, **result},
        json.dumps(result)
        + ("\ndry run: nothing written" if args.dry_run else f"\nsaved {written}"),
        EXIT_OK,
    )


def cmd_cad_schema(args: argparse.Namespace) -> CommandOutput:
    from .cad.schema import cad_script_schema

    schema = cad_script_schema()
    return schema, json.dumps(schema, indent=2), EXIT_OK


def add_cad_parsers(sub, common, writing) -> None:
    p_cad = sub.add_parser(
        "cad",
        help="precision modeling: CAD scripts, reports, STEP/IFC/DXF exchange, drawings",
        description="Parametric sketches, parts, assemblies, buildings and drawings (docs/CAD.md).",
    )
    cad = p_cad.add_subparsers(dest="cad_command", required=True, metavar="SUBCOMMAND")
    p = cad.add_parser(
        "run",
        parents=[common, writing],
        help="apply a JSON CAD script to a project (atomically)",
        description="Apply an aether3d.cad-script document as one step of the project's "
        "history (all or nothing), then save. A failed expect check exits with status 3 "
        "and saves nothing.",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("script", help="CAD script (.json), or - to read it from stdin")
    p.set_defaults(handler=cmd_cad_run)
    p = cad.add_parser("info", parents=[common], help="CAD items, or one item's report")
    p.add_argument("path", help="the .aether project")
    p.add_argument("item", nargs="?", help="an item's name")
    p.set_defaults(handler=cmd_cad_info)
    p = cad.add_parser(
        "export",
        parents=[common],
        help="write STEP, IFC, DXF, SVG or PNG",
        description="By OUT's suffix: .step/.stp (parts: --item, repeatable; default all), "
        ".ifc (a building), .dxf (a sketch or a drawing), .svg/.png (a drawing).",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("out", help="the file to write")
    p.add_argument("--item", action="append", help="item name (repeatable for STEP)")
    p.add_argument("--units", choices=("mm", "cm", "m", "in", "ft"), help="STEP/DXF units")
    p.add_argument("--dpi", type=float, default=150.0, help="PNG resolution")
    p.add_argument("--force", action="store_true", help="overwrite OUT")
    p.set_defaults(handler=cmd_cad_export)
    p = cad.add_parser(
        "import",
        parents=[common, writing],
        help="read a STEP, IFC or DXF file into a project",
        description="STEP solids become parts, IFC4 products parts (with their property "
        "and quantity sets), DXF 2D geometry a sketch; one undo step; then save.",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("file", help="the .step/.stp, .ifc or .dxf file")
    p.add_argument("--name", help="name of the new part or sketch")
    p.add_argument(
        "--units",
        choices=("mm", "cm", "m", "in", "ft"),
        help="DXF unit when the file does not state it",
    )
    p.set_defaults(handler=cmd_cad_import)
    p = cad.add_parser("schema", parents=[common], help="the JSON Schema of CAD scripts")
    p.set_defaults(handler=cmd_cad_schema)
