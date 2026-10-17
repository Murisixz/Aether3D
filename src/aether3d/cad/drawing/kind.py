"""The ``drawing`` item kind: a view of other items, regenerated when they change.

Definition::

    {"view": "plan" | "section" | "top" | "front" | "back" | "left" | "right" |
             "bottom" | "iso",
     "items": [names],                    # what is drawn (parts, assemblies, floors, ...)
     "level": "Ground", "cut_height": "1.2 m",          # plan: building level and cut
     "plane": {"origin": [x, y, z], "normal": [...], "right": [...]},  # section
     "hidden": true, "scale": "1:100" | null, "sheet": "A3",
     "dimensions": "auto" | "none" | [{"from": [x, y, z], "to": [x, y, z],
                                        "offset": length}],
     "title": "..."}

A plan is a horizontal section at ``cut_height`` above the level, looking down, with
door swings, room names and areas, and (``"auto"``) overall and exterior wall
dimensions. Lengths in the definition use the document's units.
"""

from __future__ import annotations

import math
import time
from fractions import Fraction

import numpy as np

from ...errors import CadError
from ..document import Context, ItemKind, register
from ..units import AREA, LENGTH, Quantity, format_quantity
from .model import SHEETS, Drawing
from .views import VIEWS, Cut, ViewFrame, gather, project

VIEW_NAMES = ("plan", "section", *VIEWS)
HATCH = {
    "wall": ("solid", 0.55),
    "slab": ("diagonal", 0.8),
    "column": ("solid", 0.35),
    "beam": ("solid", 0.35),
    "part": ("diagonal", 0.8),
}


def _parse_scale(raw) -> Fraction | None:
    if raw is None:
        return None
    if isinstance(raw, str) and ":" in raw:
        a, b = raw.split(":", 1)
        try:
            s = Fraction(int(a), int(b))
        except (ValueError, ZeroDivisionError):
            s = None
        if s is not None and 0 < s <= 100:
            return s
    raise CadError("INVALID_VALUE", "scale is 'paper:model', e.g. '1:100' or '2:1'")


@register
class DrawingKind(ItemKind):
    name = "drawing"
    displays = False

    def check(self, doc, d, location):
        view = d.get("view")
        if view not in VIEW_NAMES:
            raise CadError(
                "INVALID_VALUE", f"unknown view {view!r}", location, hint=", ".join(VIEW_NAMES)
            )
        items = d.get("items")
        if not isinstance(items, list) or not items or len(items) > 1000:
            raise CadError("INVALID_VALUE", "items is a list of 1 to 1000 item names", location)
        for name in items:
            it = doc.item(name)
            if it.kind in ("drawing", "parameter", "sketch", "frame"):
                raise CadError("INVALID_VALUE", f"{name!r} ({it.kind}) cannot be drawn", location)
        if view == "plan" and not isinstance(d.get("level"), str):
            raise CadError("INVALID_VALUE", "a plan needs a level", location)
        if view == "section":
            plane = d.get("plane")
            if not isinstance(plane, dict) or "origin" not in plane or "normal" not in plane:
                raise CadError("INVALID_VALUE", "a section needs plane {origin, normal}", location)
        _parse_scale(d.get("scale"))
        if d.get("sheet", "A3") not in SHEETS:
            raise CadError("INVALID_VALUE", f"sheet is one of {', '.join(SHEETS)}", location)
        dims = d.get("dimensions", "auto")
        if dims not in ("auto", "none") and not isinstance(dims, list):
            raise CadError("INVALID_VALUE", "dimensions is 'auto', 'none' or a list", location)

    def references(self, doc, d):
        return [doc.item(n).id for n in d.get("items", [])]

    def evaluate(self, ctx: Context) -> Drawing:
        doc, d = ctx.doc, ctx.item._definition
        view_name = d["view"]
        level = d.get("level") if view_name == "plan" else None
        solids = gather(doc, d["items"], level=level)
        cut = None
        frame = None
        if view_name == "plan":
            building = _building_of(doc, d["items"])
            b = doc.result(building)
            lv = b.level(level, f"{ctx.location}.level")
            h = ctx.length(d.get("cut_height", "1.2 m"), "cut_height")
            fr = b.frame
            origin = fr.to_world(np.array([0.0, 0.0, lv.elevation + h]))
            cut = Cut(origin, fr.z)
            frame = ViewFrame.make(-fr.z, fr.x)
        elif view_name == "section":
            p = d["plane"]
            n = np.asarray([ctx.length(v, f"plane.normal[{i}]") for i, v in enumerate(p["normal"])])
            n = n / np.linalg.norm(n)
            o = np.asarray([ctx.length(v, f"plane.origin[{i}]") for i, v in enumerate(p["origin"])])
            cut = Cut(o, n)
            right = p.get("right")
            if right is None:
                up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
                right = np.cross(-n, up)
            frame = ViewFrame.make(-n, right)
        else:
            frame = ViewFrame.named(view_name)
        deadline = ctx.deadline if ctx.deadline is not None else time.monotonic() + 600
        drawing = project(
            solids,
            frame,
            title=d.get("title") or ctx.item.name,
            view_name=view_name,
            cut=cut,
            # Hidden lines by default in orthographic views of parts (not in plans,
            # sections, isometric views or building elevations).
            hidden=bool(
                d.get(
                    "hidden",
                    view_name not in ("plan", "section", "iso")
                    and all(s.role == "part" for s in solids),
                )
            ),
            hatch=HATCH,
            deadline=deadline,
        )
        drawing.scale = _parse_scale(d.get("scale"))
        drawing.sheet = d.get("sheet", "A3")
        unit = doc._settings.get("length_unit", "m")
        drawing.unit_label = unit
        if view_name == "plan":
            _plan_annotations(doc, d, drawing, frame, level, unit, d.get("dimensions", "auto"))
        else:
            _dimensions(doc, d, drawing, frame, unit, ctx)
        return drawing

    def info(self, doc, item, result: Drawing):
        return result.summary()


def _building_of(doc, items):
    for name in items:
        it = doc.item(name)
        if it.kind == "building":
            return it
        bref = it._definition.get("building")
        if bref is not None:
            return doc.item(bref, "building")
    raise CadError("INVALID_VALUE", "a plan's items must include a building's floors")


def _fmt(value_m: float, unit: str, doc) -> str:
    precision = int(doc._settings.get("precision", 3))
    if unit == "mm":
        precision = 0
    return format_quantity(Quantity(value_m, LENGTH), unit, precision=precision)


def _plan_annotations(doc, d, drawing: Drawing, frame: ViewFrame, level, unit, dims):
    floors = [doc.item(n) for n in d["items"] if doc.item(n).kind == "floor"]
    wall_pts = []
    for it in floors:
        r = doc.result(it)
        if r.level.name != level:
            continue
        fr = r.building.frame
        z = r.level.elevation

        def w3(p2, fr=fr, z=z):
            return fr.to_world(np.array([p2[0], p2[1], z]))

        for o in r.openings.values():
            if o.kind != "door":
                continue
            wr = r.walls[o.wall]
            line = wr.line
            dvec, nrm = line.direction, line.normal
            lo_off, ro_off = line.offsets()
            side = -1.0 if o.flip else 1.0
            face = lo_off if side > 0 else ro_off
            hinge = line.start + dvec * (o.center - o.width / 2) + nrm * face
            leaf_end = hinge + nrm * side * o.width
            arc = [
                hinge + dvec * o.width * math.cos(t) + nrm * side * o.width * math.sin(t)
                for t in np.linspace(0.0, math.pi / 2, 25)
            ]
            drawing.line(frame.project(np.array([w3(hinge), w3(leaf_end)])), "thin", "door-swing")
            drawing.line(frame.project(np.array([w3(p) for p in arc])), "thin", "door-swing")
        for room in r.rooms.values():
            poly = np.array([p.curve.start for p in room.region.outer.pieces])
            c = _label_point(poly)
            p2 = frame.project(w3(c))[0]
            drawing.text(p2, room.name, 3.5, align="center", layer="rooms")
            area = format_quantity(
                Quantity(room.area, AREA),
                "m2" if unit in ("mm", "cm", "m") else "ft2",
                precision=2,
            )
            num = f"{room.number}  " if room.number else ""
            drawing.text(
                p2 + np.array([0.0, -_model_text(drawing, 5.5)]),
                f"{num}{area}",
                2.5,
                align="center",
                layer="rooms",
            )
        for wr in r.walls.values():
            if wr.exterior:
                wall_pts.append(frame.project(np.array([w3(p) for p in wr.footprint])))
    if dims == "auto" and wall_pts:
        allp = np.concatenate(wall_pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        gap = max(hi - lo) * 0.08
        drawing.dimension(
            [lo[0], lo[1]], [hi[0], lo[1]], -gap, _fmt(hi[0] - lo[0], unit, doc), layer="dimensions"
        )
        drawing.dimension(
            [lo[0], hi[1]], [lo[0], lo[1]], -gap, _fmt(hi[1] - lo[1], unit, doc), layer="dimensions"
        )
    elif isinstance(dims, list):
        _explicit_dimensions(doc, dims, drawing, frame, unit)


def _model_text(drawing: Drawing, paper_mm: float) -> float:
    """A paper length in model metres at the drawing's scale (fitted if not set)."""
    return paper_mm / 1000.0 / float(drawing.paper_scale())


def _label_point(poly: np.ndarray) -> np.ndarray:
    """A point inside a polygon near its centroid (for labels)."""
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    if abs(a) > 0:
        c = np.array(
            [((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]
        ) / (6 * a)
        if _inside(poly, c):
            return c
    # Otherwise the middle of the widest horizontal span through the vertical middle.
    ym = 0.5 * (y.min() + y.max())
    xs = []
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (ay <= ym) != (by <= ym):
            xs.append(ax + (ym - ay) * (bx - ax) / (by - ay))
    xs.sort()
    best = max(((xs[i + 1] - xs[i], i) for i in range(0, len(xs) - 1, 2)), default=(0, 0))
    i = best[1]
    return np.array([0.5 * (xs[i] + xs[i + 1]), ym]) if len(xs) >= 2 else poly.mean(axis=0)


def _inside(poly, p) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (ay <= p[1]) != (by <= p[1]) and p[0] < ax + (p[1] - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _dimensions(doc, d, drawing: Drawing, frame: ViewFrame, unit, ctx):
    dims = d.get("dimensions", "auto")
    if dims == "auto" and d["view"] != "iso":  # lengths in an isometric view are foreshortened
        lo, hi = drawing.model_bounds()
        if not np.all(np.isfinite(lo)):
            return
        gap = max(float(np.max(hi - lo)) * 0.08, 1e-6)
        drawing.dimension([lo[0], lo[1]], [hi[0], lo[1]], -gap, _fmt(hi[0] - lo[0], unit, doc))
        drawing.dimension([lo[0], hi[1]], [lo[0], lo[1]], -gap, _fmt(hi[1] - lo[1], unit, doc))
    elif isinstance(dims, list):
        _explicit_dimensions(doc, dims, drawing, frame, unit)


def _explicit_dimensions(doc, dims, drawing, frame, unit):
    if len(dims) > 500:
        raise CadError("LIMIT_EXCEEDED", "at most 500 dimensions")
    for k, dim in enumerate(dims):
        loc = f"dimensions[{k}]"
        a = np.array([doc.evaluate_value(v, LENGTH, f"{loc}.from") for v in dim["from"]])
        b = np.array([doc.evaluate_value(v, LENGTH, f"{loc}.to") for v in dim["to"]])
        off = doc.evaluate_value(dim.get("offset", 0), LENGTH, f"{loc}.offset")
        pa, pb = frame.project(a)[0], frame.project(b)[0]
        drawing.dimension(
            pa, pb, off, dim.get("text") or _fmt(float(np.linalg.norm(pb - pa)), unit, doc)
        )


def _install() -> None:
    from ..document import CadDocument

    def drawing(self: CadDocument, name: str, view: str, items=None, **options):
        """Add (or replace the definition of) a drawing item; ``items`` defaults to every
        part and assembly, or for a plan to the building's floors and elements."""
        if items is None:
            if view == "plan":
                items = [
                    i.name
                    for i in self.items()
                    if i.kind in ("floor", "slab", "roof", "stair", "columns", "beams")
                ]
            else:
                items = [i.name for i in self.items() if i.kind in ("part", "assembly")]
        definition = {"view": view, "items": list(items), **options}
        if name in self._by_name:
            item = self.item(name, "drawing")
            self.set_definition(item, definition, label=f"Edit drawing {name!r}")
            return item
        return self.add("drawing", name, definition)

    CadDocument.drawing = drawing


_install()
