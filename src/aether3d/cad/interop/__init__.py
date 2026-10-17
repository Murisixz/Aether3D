"""Exchange formats: STEP (exact B-rep solids, AP214) and IFC4 (buildings).

Document-level entry points (installed on :class:`~aether3d.cad.document.CadDocument`):

* ``doc.export_step(items=None, units=None)`` -- STEP text of parts' bodies;
* ``doc.import_step(text, name=None)`` -- one part per solid (a ``solid`` feature);
* ``doc.export_ifc(building=None, name=...)`` -- IFC4 text and entity counts;
* ``doc.import_ifc(text, create_parts=True)`` -- the products of an IFC4 file (subset:
  extruded area solids, faceted B-reps, openings), optionally as parts;
* ``doc.import_dxf(text, name, plane=..., unit=None)`` -- a sketch from 2D DXF geometry;
  ``doc.export_sketch_dxf(sketch)`` -- a sketch as DXF; drawings write DXF themselves.

Every import treats its text as untrusted: size and entity limits, a strict tokenizer,
no evaluation of anything in the file, and structured ``CadError`` on anything else.
"""

from __future__ import annotations

from typing import Any

from ...errors import CadError

MAX_IMPORTED_SOLIDS = 500


def _export_step(doc, items: Any = None, units: str | None = None, name: str = "aether3d") -> str:
    from .step import write_step

    parts = [doc.item(i, "part") for i in items] if items is not None else doc.items("part")
    bodies = []
    for it in parts:
        res = doc.result(it)
        for bname in sorted(res.bodies):
            label = it.name if bname == "main" else f"{it.name}/{bname}"
            bodies.append((label, res.bodies[bname]))
    if not bodies:
        raise CadError("NOTHING_TO_EXPORT", "there are no part bodies to export")
    doc_unit = doc._settings.get("length_unit", "mm")
    unit = units or (doc_unit if doc_unit in ("mm", "m", "in") else "mm")
    return write_step(bodies, units=unit, name=name)


def _import_step(doc, text: str, name: str | None = None) -> list:
    from ..kinds import PartHandle
    from .step import read_step

    solids = read_step(text)
    if len(solids) > MAX_IMPORTED_SOLIDS:
        raise CadError("LIMIT_EXCEEDED", f"more than {MAX_IMPORTED_SOLIDS} solids")
    handles = []
    with doc.batch(f"Import STEP ({len(solids)} solids)"):
        for k, (sname, body) in enumerate(solids):
            base = name if name else (sname or "Imported")
            label = base if len(solids) == 1 else f"{base} {k + 1}"
            label = doc._unique_name(_clean(label))
            item = doc.add(
                "part",
                label,
                {
                    "features": [{"type": "solid", "id": "import", "brep": body.to_dict()}],
                    "source": {"format": "step", "name": sname},
                },
            )
            handles.append(PartHandle(doc, item))
    return handles


def _export_ifc(doc, building: Any = None, name: str = "Aether3D model") -> tuple[str, dict]:
    from ...arch.ifc import write_ifc

    return write_ifc(doc, building=building, name=name)


def _import_ifc(doc, text: str, create_parts: bool = True) -> list:
    from ...arch.ifc import read_ifc

    elements = read_ifc(text)
    if create_parts:
        made = [e for e in elements if e.body is not None]
        if len(made) > MAX_IMPORTED_SOLIDS:
            raise CadError("LIMIT_EXCEEDED", f"more than {MAX_IMPORTED_SOLIDS} products")
        with doc.batch(f"Import IFC ({len(made)} products)"):
            for e in made:
                label = doc._unique_name(_clean(f"{e.ifc_class[3:].title()} {e.name}".strip()))
                doc.add(
                    "part",
                    label,
                    {
                        "features": [{"type": "solid", "id": "import", "brep": e.body.to_dict()}],
                        "source": {
                            "format": "ifc",
                            "class": e.ifc_class,
                            "global_id": e.global_id,
                            "storey": e.storey,
                            "properties": e.properties,
                        },
                    },
                )
    return elements


def _clean(label: str) -> str:
    out = "".join(c if c.isprintable() and c not in "\\/:|#[]" else "_" for c in label).strip()
    return out[:64] or "Imported"


def _install() -> None:
    from ..document import CadDocument

    CadDocument.export_step = _export_step
    CadDocument.import_step = _import_step
    CadDocument.export_ifc = _export_ifc
    CadDocument.import_ifc = _import_ifc
    from . import dxf  # noqa: F401  (installs import_dxf and export_sketch_dxf)


_install()
