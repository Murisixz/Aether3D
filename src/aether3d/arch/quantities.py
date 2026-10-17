"""Quantity takeoff: base quantities of building elements, and schedules.

Quantities are computed from the elements' parameters with the definitions of the IFC
base quantity sets (IFC 4.3.2 lexical pages) -- in SI units (m, m2, m3):

* walls (``Qto_WallBaseQuantities``): ``Length`` (location line), ``Width``,
  ``Height``, ``GrossFootprintArea``, ``GrossSideArea`` (length x height, the middle
  plane, openings not deducted), ``NetSideArea`` (openings deducted), ``GrossVolume``
  (footprint x height, openings not deducted) and ``NetVolume`` (openings deducted, the
  join geometry considered: the footprint already holds it);
* doors and windows: ``Width``, ``Height``, ``Area``, ``Perimeter``;
* spaces (``Qto_SpaceBaseQuantities``): ``NetFloorArea`` (inside the wall faces),
  ``NetPerimeter``, ``Height``, ``NetVolume``;
* slabs, roofs, stairs, columns and beams: see ``elements.py``.

These are exact for the supported geometry (straight walls, rectangular openings, planar
elements). The tests compare every volume with the B-rep's measured volume.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def wall_quantities(w, openings) -> dict[str, float]:
    length = w.line.length
    gross_fp = _polygon_area(w.footprint)
    ops = [openings[o] for o in w.openings]
    open_area = sum(o.width * o.height for o in ops)
    open_volume = sum(o.width * o.height * w.line.thickness for o in ops)
    floor_cuts = sum(o.width * w.line.thickness for o in ops if o.sill <= 1e-12)
    return {
        "Length": length,
        "Width": w.line.thickness,
        "Height": w.height,
        "GrossFootprintArea": gross_fp,
        "NetFootprintArea": gross_fp - floor_cuts,
        "GrossSideArea": length * w.height,
        "NetSideArea": length * w.height - open_area,
        "GrossVolume": gross_fp * w.height,
        "NetVolume": gross_fp * w.height - open_volume,
    }


def opening_quantities(o) -> dict[str, float]:
    return {
        "Width": o.width,
        "Height": o.height,
        "Area": o.width * o.height,
        "Perimeter": 2 * (o.width + o.height),
        "SillHeight": o.sill,
    }


def space_quantities(r) -> dict[str, float]:
    return {
        "NetFloorArea": r.area,
        "NetPerimeter": r.perimeter,
        "Height": r.height,
        "NetVolume": r.area * r.height,
    }


def floor_quantities(doc, r) -> dict[str, Any]:
    walls = [
        {
            "id": w.id,
            "type": w.type,
            "exterior": w.exterior,
            "joins": w.joins,
            "quantities": wall_quantities(w, r.openings),
            "openings": w.openings,
        }
        for w in r.walls.values()
    ]
    openings = [
        {
            "id": o.id,
            "wall": o.wall,
            "family": o.family,
            "kind": o.kind,
            "quantities": opening_quantities(o),
        }
        for o in r.openings.values()
    ]
    rooms = [
        {"id": x.id, "name": x.name, "number": x.number, "quantities": space_quantities(x)}
        for x in r.rooms.values()
    ]
    totals = {
        "walls": len(walls),
        "wall_length": sum(w["quantities"]["Length"] for w in walls),
        "wall_net_volume": sum(w["quantities"]["NetVolume"] for w in walls),
        "doors": sum(1 for o in openings if o["kind"] == "door"),
        "windows": sum(1 for o in openings if o["kind"] == "window"),
        "room_area": sum(x["quantities"]["NetFloorArea"] for x in rooms),
    }
    out = {
        "level": r.level.name,
        "units": {"length": "m", "area": "m2", "volume": "m3"},
        "walls": walls,
        "openings": openings,
        "rooms": rooms,
        "totals": totals,
    }
    if r.outline is not None:
        out["gross_floor_area"] = r.outline.area
    if r.diagnostics:
        out["diagnostics"] = r.diagnostics
    return out


def schedule(doc, kind: str = "all") -> list[dict[str, Any]]:
    """Rows of a schedule over the document's building elements (one row per element)."""
    rows: list[dict[str, Any]] = []
    for item in doc.items():
        if item.kind == "floor" and kind in ("all", "walls", "doors", "windows", "rooms"):
            r = doc.result(item)
            if kind in ("all", "walls"):
                for w in r.walls.values():
                    rows.append(
                        {
                            "element": "wall",
                            "floor": item.name,
                            "level": r.level.name,
                            "id": w.id,
                            "type": w.type,
                            **wall_quantities(w, r.openings),
                        }
                    )
            for o in r.openings.values():
                if kind == "all" or kind == f"{o.kind}s":
                    rows.append(
                        {
                            "element": o.kind,
                            "floor": item.name,
                            "level": r.level.name,
                            "id": o.id,
                            "type": o.family,
                            "host": o.wall,
                            **opening_quantities(o),
                        }
                    )
            if kind in ("all", "rooms"):
                for x in r.rooms.values():
                    rows.append(
                        {
                            "element": "space",
                            "floor": item.name,
                            "level": r.level.name,
                            "id": x.id,
                            "name": x.name,
                            "number": x.number,
                            **space_quantities(x),
                        }
                    )
        elif item.kind in ("slab", "roof", "stair", "columns", "beams") and kind in (
            "all",
            item.kind,
        ):
            r = doc.result(item)
            rows.append({"element": item.kind, "id": item.name, **r.quantities})
    return rows


def to_csv(rows: list[dict[str, Any]]) -> str:
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    return buf.getvalue()
