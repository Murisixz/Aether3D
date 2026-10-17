"""The precision-modeling tools of the MCP server (Stage 15).

Building and editing go through ``run_script`` with an ``aether3d.cad-script`` (one
undo step, all or nothing; schema ``aether3d://schema/cad-script``, guide
``aether3d://cad/guide``). Two tools and an export complete it:

* ``inspect_cad`` (read-only) -- the document's items, or one item's report: a sketch's
  status, degrees of freedom and conflicts; a part's bodies (validity, volume, mass
  properties) and named faces; a building's quantities; a drawing's summary -- and,
  on request, a drawing of the item as an image (computed on the side: nothing is added
  to the document);
* ``export_scene`` (with a CAD suffix; :func:`export_cad`) -- STEP (parts), IFC4 (a
  building), DXF (a sketch or a drawing), SVG and PNG (a drawing) into the workspace,
  atomically;
* ``import_cad`` -- STEP solids as parts, IFC4 products as parts, DXF geometry as a
  sketch, as one undo step. Files are untrusted: strict parsers with size and entity
  limits; unsupported content is refused or skipped and reported.
"""

from __future__ import annotations

from typing import Any

from .session import Session, ToolError

READ_SUFFIXES = (".step", ".stp", ".ifc", ".dxf")
WRITE_SUFFIXES = (".step", ".stp", ".ifc", ".dxf", ".svg", ".png")
MAX_FACES_LISTED = 200


def _doc(session: Session):
    return session.require_scene().cad


def inspect_cad(session: Session, args: dict, ctx: Any):
    from .tools import ToolOutput, scene_status

    doc = _doc(session)
    name = args.get("item")
    if name is None:
        items = []
        for it in doc.items():
            entry = {"name": it.name, "kind": it.kind, "links": len(it.links)}
            if it.kind == "sketch":
                info = doc.info(it)
                entry.update({k: info.get(k) for k in ("status", "dof") if k in info})
                if "error" in info:
                    entry["error"] = info["error"]
            items.append(entry)
        return ToolOutput(
            {
                "settings": doc.settings,
                "items": items,
                "stats": dict(doc.stats),
                "scene": scene_status(session),
            }
        )
    item = doc.item(name)
    data = doc.info(item)
    if item.kind == "part" and "error" not in data:
        from ..cad.naming import describe_faces

        res = doc.result(item)
        unit = doc.evaluate_value(1.0, _length(), "unit")
        faces = {}
        for bname, body in sorted(res.bodies.items()):
            listed = describe_faces(body, unit)
            faces[bname] = listed[:MAX_FACES_LISTED]
        data["faces"] = faces
    image = None
    mode = args.get("image", "none")
    if mode == "drawing":
        image, note = _drawing_image(doc, item, args)
        data["image"] = note
    return ToolOutput({**data, "scene": scene_status(session)}, image=image)


def _length():
    from ..cad.units import LENGTH

    return LENGTH


def _drawing_image(doc, item, args):
    from ..cad.drawing.views import ViewFrame, gather, project
    from ..rendering.png import encode_png

    size = 1024
    if item.kind == "drawing":
        drawing = doc.result(item)
    else:
        if item.kind not in (
            "part",
            "assembly",
            "floor",
            "slab",
            "roof",
            "stair",
            "columns",
            "beams",
        ):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"a {item.kind} has no drawing",
                hint="parts, assemblies, building elements or drawing items",
            )
        view = args.get("view", "iso")
        solids = gather(doc, [item.name])
        drawing = project(
            solids, ViewFrame.named(view), title=item.name, view_name=view, hidden=view != "iso"
        )
    w_mm = (
        420.0 if drawing.sheet == "A3" else {"A4": 297.0, "A2": 594.0, "A1": 841.0}[drawing.sheet]
    )
    dpi = max(20.0, min(300.0, size / w_mm * 25.4))
    png = encode_png(drawing.to_png(dpi))
    return png, (
        f"{drawing.view} drawing of {item.name!r} ({drawing.scale_label()}): exact edges, "
        "hidden lines removed by ray casting; the report's numbers are the authority"
    )


def export_cad(session: Session, args: dict, ctx: Any):
    from .._fileio import write_atomic
    from .tools import ToolOutput, scene_status

    doc = _doc(session)
    target = session.workspace.path(
        args["path"], suffixes=WRITE_SUFFIXES, write=True, overwrite=args.get("overwrite", False)
    )
    suffix = target.suffix.lower()
    item = args.get("item")
    report: dict[str, Any] = {}
    if suffix in (".step", ".stp"):
        items = args.get("items") or ([item] if item else None)
        text = doc.export_step(items, units=args.get("units"))
        data = text.encode("utf-8")
    elif suffix == ".ifc":
        text, stats = doc.export_ifc(building=item)
        report["entities"] = stats
        data = text.encode("utf-8")
    else:
        if not item:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"{suffix} export needs item (a drawing"
                + (" or a sketch)" if suffix == ".dxf" else ")"),
            )
        it = doc.item(item)
        if suffix == ".dxf" and it.kind == "sketch":
            data = doc.export_sketch_dxf(it, args.get("units")).encode("utf-8")
        elif it.kind != "drawing":
            raise ToolError(
                "INVALID_ARGUMENT",
                f"{item!r} is a {it.kind}, not a drawing",
                hint="add a drawing item with run_script (kind drawing)",
            )
        else:
            drawing = doc.result(it)
            if suffix == ".dxf":
                data = drawing.to_dxf(args.get("units") or "mm").encode("utf-8")
            elif suffix == ".svg":
                data = drawing.to_svg().encode("utf-8")
            else:
                from ..rendering.png import encode_png

                data = encode_png(drawing.to_png(float(args.get("dpi", 150))))
            report["drawing"] = drawing.summary()
    write_atomic(target, data)
    return ToolOutput(
        {
            "path": session.workspace.relative(target),
            "bytes": len(data),
            **report,
            "scene": scene_status(session),
        }
    )


def import_cad(session: Session, args: dict, ctx: Any):
    from .tools import ToolOutput, scene_status

    doc = _doc(session)
    source = session.workspace.path(args["path"], suffixes=READ_SUFFIXES)
    suffix = source.suffix.lower()
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if suffix in (".step", ".stp"):
        parts = doc.import_step(text, name=args.get("name"))
        out = {"parts": [p.name for p in parts]}
    elif suffix == ".ifc":
        elements = doc.import_ifc(text)
        out = {
            "products": len(elements),
            "by_class": _count(e.ifc_class for e in elements),
            "without_geometry": [e.name for e in elements if e.body is None][:50],
            "unsupported": sorted({u for e in elements for u in e.unsupported})[:50],
        }
    else:
        name = args.get("name") or source.stem
        sk = doc.import_dxf(
            text,
            name,
            plane=args.get("plane", "XY"),
            layers=args.get("layers"),
            unit=args.get("units"),
        )
        out = {"sketch": sk.name, **sk.import_report}
    return ToolOutput(
        {"path": session.workspace.relative(source), **out, "scene": scene_status(session)}
    )


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def specs() -> tuple[Any, ...]:
    from .tools import PATH, ToolSpec, _schema

    name = {"type": "string", "minLength": 1, "maxLength": 128}
    units = {"enum": ["mm", "cm", "m", "in", "ft"]}
    return (
        ToolSpec(
            "inspect_cad",
            "Inspect the CAD document",
            "The CAD items, or one item's report (sketch status and dof; part validity, mass, "
            "named faces; quantities); image=drawing adds a drawing of it. Read-only.",
            _schema(
                {
                    "item": name,
                    "image": {"enum": ["none", "drawing"]},
                    "view": {"enum": ["iso", "top", "front", "right"]},
                }
            ),
            inspect_cad,
            read_only=True,
            idempotent=True,
        ),
        ToolSpec(
            "import_cad",
            "Import a CAD file",
            "Read a .step, .ifc (as parts) or .dxf (as a sketch: name, plane, units, layers) "
            "workspace file into the CAD document, one undo step; reports what it skips.",
            _schema(
                {
                    "path": PATH,
                    "name": name,
                    "plane": {"enum": ["XY", "XZ", "YZ"]},
                    "units": units,
                    "layers": {"type": "array", "items": {"type": "string"}, "maxItems": 256},
                },
                ("path",),
            ),
            import_cad,
        ),
    )
