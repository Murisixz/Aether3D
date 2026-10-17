"""IFC4 export (and import of that subset): buildings as Industry Foundation Classes.

The export writes an ISO 10303-21 file with schema ``IFC4`` (ISO 16739-1:2018, ADD2 TC1)
using real IFC entities and relationships:

* the spatial structure ``IfcProject`` > ``IfcSite`` > ``IfcBuilding`` >
  ``IfcBuildingStorey`` (``IfcRelAggregates``), one storey per level, with metre/m2/m3/rad
  units and a ``Body`` (and ``Axis``) representation context;
* ``IfcWall`` placed along its location line, with an ``Axis`` polyline and a
  ``SweptSolid`` body (its plan footprint, joins included, extruded to its height), its
  type (``IfcWallType`` + ``IfcRelDefinesByType``) and material layers
  (``IfcMaterialLayerSetUsage`` + ``IfcRelAssociatesMaterial``);
* every opening as an ``IfcOpeningElement`` voiding its wall (``IfcRelVoidsElement``), and
  doors and windows (``IfcDoor``/``IfcWindow``, faceted B-rep bodies) filling them
  (``IfcRelFillsElement``);
* ``IfcSpace`` per room (aggregated by its storey), ``IfcSlab``, ``IfcRoof`` and
  ``IfcStair`` (faceted B-reps of the exact planar solids), ``IfcColumn`` and ``IfcBeam``
  (extruded rectangle, circle and I profiles);
* containment (``IfcRelContainedInSpatialStructure``), property sets
  (``Pset_WallCommon``, ``Pset_StairCommon``, and ``Aether3D_Identity`` with the CAD item
  and element ids) and the base quantity sets (``Qto_WallBaseQuantities``,
  ``Qto_DoorBaseQuantities``, ``Qto_WindowBaseQuantities``, ``Qto_SpaceBaseQuantities``,
  ``Qto_SlabBaseQuantities``, ``Qto_RoofBaseQuantities``, ``Qto_ColumnBaseQuantities``,
  ``Qto_BeamBaseQuantities``) with the values of ``quantities.py``.

GlobalIds are the IFC base-64 compression of UUID 5s of the CAD item and element ids, so
re-exporting the same model gives the same GlobalIds. No model view definition is claimed
(``ViewDefinition [NotAssigned]``). The tests validate exports with IfcOpenShell (schema
and where-rules) when it is available, as an external validator.
"""

from __future__ import annotations

import datetime
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cad.brep.topology import Body
from ..cad.interop.step import _f, _str
from ..errors import CadError
from .quantities import opening_quantities, space_quantities, wall_quantities

_B64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
_NS = uuid.UUID("6c1a8e7e-6a0c-5b35-9c7e-a3d3e7f00015")


def global_id(*parts: str) -> str:
    """The 22-character IFC GlobalId of a UUID 5 of ``parts`` (deterministic)."""
    u = uuid.uuid5(_NS, "/".join(parts)).int
    chars = [_B64[(u >> 126) & 0x3]]
    for k in range(20, -1, -1):
        chars.append(_B64[(u >> (6 * k)) & 0x3F])
    return "".join(chars)


@dataclass(frozen=True)
class R:
    """A reference to an entity instance (``#n``)."""

    n: int


@dataclass(frozen=True)
class E:
    """An enumeration value (``.NAME.``)."""

    name: str


@dataclass(frozen=True)
class T:
    """A typed value (``IFCBOOLEAN(.T.)``) or any literal parameter text."""

    text: str


DERIVED = T("*")


def _arg(v: Any) -> str:
    if v is None:
        return "$"
    if isinstance(v, R):
        return f"#{v.n}"
    if isinstance(v, E):
        return f".{v.name}."
    if isinstance(v, T):
        return v.text
    if isinstance(v, bool):
        return ".T." if v else ".F."
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _f(v)
    if isinstance(v, str):
        return _str(v)
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_arg(x) for x in v) + ")"
    raise CadError("INVALID_VALUE", f"cannot write {type(v).__name__} to IFC")


class _W:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.k = 0
        self.dirs: dict[tuple, R] = {}

    def __call__(self, entity: str, *args: Any) -> R:
        self.k += 1
        self.lines.append(f"#{self.k}={entity}({','.join(_arg(a) for a in args)});")
        return R(self.k)

    def pt(self, p) -> R:
        return self("IFCCARTESIANPOINT", [float(v) for v in p])

    def dir(self, d) -> R:
        d = np.asarray(d, dtype=np.float64)
        d = d / np.linalg.norm(d)
        key = tuple(np.round(d, 14))
        if key not in self.dirs:
            self.dirs[key] = self("IFCDIRECTION", [float(v) for v in d])
        return self.dirs[key]

    def a2p3(self, origin, z=(0.0, 0.0, 1.0), x=(1.0, 0.0, 0.0)) -> R:
        return self("IFCAXIS2PLACEMENT3D", self.pt(origin), self.dir(z), self.dir(x))

    def placement(self, rel: R | None, origin, z=(0.0, 0.0, 1.0), x=(1.0, 0.0, 0.0)) -> R:
        return self("IFCLOCALPLACEMENT", rel, self.a2p3(origin, z, x))

    def polyline(self, pts2d) -> R:
        pts = [self.pt(p) for p in pts2d]
        return self("IFCPOLYLINE", [*pts, pts[0]])

    def extrusion(self, profile: R, depth: float) -> R:
        return self(
            "IFCEXTRUDEDAREASOLID", profile, self.a2p3((0.0, 0.0, 0.0)), self.dir((0, 0, 1)), depth
        )

    def shape(self, ctx: R, kind: str, items: list[R], ident: str = "Body") -> R:
        return self("IFCSHAPEREPRESENTATION", ctx, ident, kind, items)

    def product(self, reps: list[R]) -> R:
        return self("IFCPRODUCTDEFINITIONSHAPE", None, None, reps)

    def rel(self, entity: str, gid: str, *args: Any) -> R:
        return self(entity, gid, None, None, None, *args)


def _ccw(pts):
    a = 0.5 * sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    )
    return list(pts) if a >= 0 else list(pts)[::-1]


def _faceted(w: _W, body: Body, to_local) -> list[R]:
    """Faceted B-reps (one per shell) of a body with planar faces bounded by lines."""
    out = []
    for shell in body.shells:
        faces = []
        for fi in shell:
            f = body.faces[fi]
            if f.surface.kind != "plane":
                raise CadError("UNSUPPORTED_GEOMETRY", "faceted IFC bodies need planar faces")
            bounds = []
            for li, loop in enumerate(f.loops):
                pts = []
                for ei, fwd in loop:
                    e = body.edges[ei]
                    if e.curve.kind != "line":
                        raise CadError(
                            "UNSUPPORTED_GEOMETRY", "faceted IFC bodies need straight edges"
                        )
                    pts.append(body.vertices[e.v0 if fwd else e.v1])
                # Loops already run counter-clockwise seen from outside (whatever the
                # face's sense), which is what IfcPolyLoop expects.
                lp = w("IFCPOLYLOOP", [w.pt(to_local(p)) for p in pts])
                bounds.append(w("IFCFACEOUTERBOUND" if li == 0 else "IFCFACEBOUND", lp, True))
            faces.append(w("IFCFACE", bounds))
        out.append(w("IFCFACETEDBREP", w("IFCCLOSEDSHELL", faces)))
    return out


_QUANTITY_KIND = {
    **dict.fromkeys(
        ("Length", "Width", "Height", "Perimeter", "NetPerimeter", "Depth"), "IFCQUANTITYLENGTH"
    ),
    **dict.fromkeys(
        (
            "GrossFootprintArea",
            "NetFootprintArea",
            "GrossSideArea",
            "NetSideArea",
            "Area",
            "NetFloorArea",
            "GrossArea",
            "NetArea",
            "ProjectedArea",
            "CrossSectionArea",
        ),
        "IFCQUANTITYAREA",
    ),
    **dict.fromkeys(("GrossVolume", "NetVolume"), "IFCQUANTITYVOLUME"),
}


def _value(v: Any) -> T:
    if isinstance(v, bool):
        return T(f"IFCBOOLEAN({'.T.' if v else '.F.'})")
    if isinstance(v, int):
        return T(f"IFCCOUNTMEASURE({v})")
    if isinstance(v, float):
        kind = "IFCPOSITIVELENGTHMEASURE" if v > 0 else "IFCLENGTHMEASURE"
        return T(f"{kind}({_f(v)})")
    return T(f"IFCIDENTIFIER({_str(str(v)[:255])})")


class _Exporter:
    def __init__(self, doc, name: str) -> None:
        self.doc = doc
        self.w = _W()
        self.name = name
        self.stats: dict[str, int] = {}

    def count(self, kind: str) -> None:
        self.stats[kind] = self.stats.get(kind, 0) + 1

    def attach(
        self,
        item,
        key: str,
        element: R,
        qto: str | None,
        quantities: dict,
        pset: dict | None = None,
        pset_name: str | None = None,
    ) -> None:
        w = self.w
        if qto and quantities:
            items = [
                w(_QUANTITY_KIND[k], k, None, None, float(v), None)
                for k, v in quantities.items()
                if k in _QUANTITY_KIND
            ]
            q = w(
                "IFCELEMENTQUANTITY", global_id(item.id, "qto", key), None, qto, None, None, items
            )
            w.rel("IFCRELDEFINESBYPROPERTIES", global_id(item.id, "rel-qto", key), [element], q)
        ident = self._pset(
            global_id(item.id, "pset-id", key),
            "Aether3D_Identity",
            {"CadItem": item.name, "Element": key},
        )
        w.rel("IFCRELDEFINESBYPROPERTIES", global_id(item.id, "rel-id", key), [element], ident)
        if pset:
            ps = self._pset(global_id(item.id, "pset", key), pset_name, pset)
            w.rel("IFCRELDEFINESBYPROPERTIES", global_id(item.id, "rel-pset", key), [element], ps)

    def _pset(self, gid: str, name: str, props: dict) -> R:
        w = self.w
        values = [w("IFCPROPERTYSINGLEVALUE", k, None, _value(v), None) for k, v in props.items()]
        return w("IFCPROPERTYSET", gid, None, name, None, values)

    def run(self, buildings) -> str:
        w = self.w
        units = w(
            "IFCUNITASSIGNMENT",
            [
                w("IFCSIUNIT", DERIVED, E("LENGTHUNIT"), None, E("METRE")),
                w("IFCSIUNIT", DERIVED, E("AREAUNIT"), None, E("SQUARE_METRE")),
                w("IFCSIUNIT", DERIVED, E("VOLUMEUNIT"), None, E("CUBIC_METRE")),
                w("IFCSIUNIT", DERIVED, E("PLANEANGLEUNIT"), None, E("RADIAN")),
            ],
        )
        ctx = w(
            "IFCGEOMETRICREPRESENTATIONCONTEXT",
            None,
            "Model",
            3,
            1e-5,
            w.a2p3((0.0, 0.0, 0.0)),
            None,
        )
        self.body_ctx = w(
            "IFCGEOMETRICREPRESENTATIONSUBCONTEXT",
            "Body",
            "Model",
            DERIVED,
            DERIVED,
            DERIVED,
            DERIVED,
            ctx,
            None,
            E("MODEL_VIEW"),
            None,
        )
        self.axis_ctx = w(
            "IFCGEOMETRICREPRESENTATIONSUBCONTEXT",
            "Axis",
            "Model",
            DERIVED,
            DERIVED,
            DERIVED,
            DERIVED,
            ctx,
            None,
            E("GRAPH_VIEW"),
            None,
        )
        self.project = w(
            "IFCPROJECT",
            global_id("project", self.doc.scene.name, self.name),
            None,
            self.name,
            None,
            None,
            None,
            None,
            [ctx],
            units,
        )
        for b in buildings:
            self.building(b)
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")
        from .._version import __version__

        header = [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION(('ViewDefinition [NotAssigned]'),'2;1');",
            f"FILE_NAME({_str(self.name)},'{stamp}',(''),(''),"
            f"'aether3d {__version__}','aether3d {__version__}','');",
            "FILE_SCHEMA(('IFC4'));",
            "ENDSEC;",
            "DATA;",
        ]
        return "\n".join(header + w.lines + ["ENDSEC;", "END-ISO-10303-21;", ""])

    def building(self, bitem) -> None:
        w, doc = self.w, self.doc
        b = doc.result(bitem)
        fr = b.frame
        ang = math.atan2(float(fr.x[1]), float(fr.x[0]))
        site_pl = w.placement(None, fr.origin, (0, 0, 1), (math.cos(ang), math.sin(ang), 0))
        site = w(
            "IFCSITE",
            global_id(bitem.id, "site"),
            None,
            "Site",
            None,
            None,
            site_pl,
            None,
            None,
            E("ELEMENT"),
            None,
            None,
            None,
            None,
            None,
        )
        bld_pl = w.placement(site_pl, (0.0, 0.0, 0.0))
        bld = w(
            "IFCBUILDING",
            global_id(bitem.id, "building"),
            None,
            bitem.name,
            None,
            None,
            bld_pl,
            None,
            None,
            E("ELEMENT"),
            None,
            None,
            None,
        )
        w.rel("IFCRELAGGREGATES", global_id(bitem.id, "agg-site"), self.project, [site])
        w.rel("IFCRELAGGREGATES", global_id(bitem.id, "agg-bld"), site, [bld])
        self.storeys, self.storey_pl = {}, {}
        for lname in b.order:
            lv = b.levels[lname]
            pl = w.placement(bld_pl, (0.0, 0.0, lv.elevation))
            self.storeys[lname] = w(
                "IFCBUILDINGSTOREY",
                global_id(bitem.id, "storey", lname),
                None,
                lname,
                None,
                None,
                pl,
                None,
                None,
                E("ELEMENT"),
                lv.elevation,
            )
            self.storey_pl[lname] = (pl, lv.elevation)
        w.rel(
            "IFCRELAGGREGATES", global_id(bitem.id, "agg-storeys"), bld, list(self.storeys.values())
        )
        self.contained = {n: [] for n in self.storeys}
        self.spaces = {n: [] for n in self.storeys}
        self.wall_types: dict[str, R] = {}
        self.typed: dict[str, list[R]] = {}
        self.layer_sets: dict[str, R] = {}
        for item in doc.items():
            if item.kind not in ("floor", "slab", "roof", "stair", "columns", "beams"):
                continue
            if doc.item(item._definition["building"]).id != bitem.id:
                continue
            if item.kind == "floor":
                self.floor(item, b)
            else:
                self.element(item, b)
        for tname, elems in self.typed.items():
            w.rel(
                "IFCRELDEFINESBYTYPE",
                global_id(bitem.id, "type-rel", tname),
                elems,
                self.wall_types[tname],
            )
        for lname, elems in self.contained.items():
            if elems:
                w.rel(
                    "IFCRELCONTAINEDINSPATIALSTRUCTURE",
                    global_id(bitem.id, "contains", lname),
                    elems,
                    self.storeys[lname],
                )
        for lname, sp in self.spaces.items():
            if sp:
                w.rel(
                    "IFCRELAGGREGATES",
                    global_id(bitem.id, "spaces", lname),
                    self.storeys[lname],
                    sp,
                )

    # -- floors -----------------------------------------------------------------------

    def floor(self, item, b) -> None:
        w, doc = self.w, self.doc
        r = doc.result(item)
        pl_storey, elev = self.storey_pl[r.level.name]
        for wr in r.walls.values():
            line = wr.line
            d, nrm = line.direction, line.normal
            wpl = w.placement(
                pl_storey,
                (line.start[0], line.start[1], wr.base - elev),
                (0, 0, 1),
                (d[0], d[1], 0),
            )
            local = _ccw(
                [(float((p - line.start) @ d), float((p - line.start) @ nrm)) for p in wr.footprint]
            )
            prof = w("IFCARBITRARYCLOSEDPROFILEDEF", E("AREA"), None, w.polyline(local))
            body = w.shape(self.body_ctx, "SweptSolid", [w.extrusion(prof, wr.height)])
            axis = w.shape(
                self.axis_ctx,
                "Curve2D",
                [w("IFCPOLYLINE", [w.pt((0.0, 0.0)), w.pt((line.length, 0.0))])],
                "Axis",
            )
            wall = w(
                "IFCWALL",
                global_id(item.id, "wall", wr.id),
                None,
                wr.id,
                None,
                wr.type or None,
                wpl,
                w.product([axis, body]),
                None,
                E("STANDARD"),
            )
            self.contained[r.level.name].append(wall)
            self.count("IfcWall")
            if wr.type:
                if wr.type not in self.wall_types:
                    self.wall_types[wr.type] = w(
                        "IFCWALLTYPE",
                        global_id(item.id, "walltype", wr.type),
                        None,
                        wr.type,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        E("STANDARD"),
                    )
                self.typed.setdefault(wr.type, []).append(wall)
                self.material(item, wr, b, wall)
            q = wall_quantities(wr, r.openings)
            self.attach(
                item,
                f"wall/{wr.id}",
                wall,
                "Qto_WallBaseQuantities",
                q,
                {"IsExternal": bool(wr.exterior), "Reference": wr.type or "wall"},
                "Pset_WallCommon",
            )
            for oid in wr.openings:
                self.opening(item, r, wr, r.openings[oid], wpl, wall)
        for room in r.rooms.values():
            outer = _ccw([tuple(p.curve.start) for p in room.region.outer.pieces])
            if room.region.holes:
                voids = [
                    w.polyline(list(reversed(_ccw([tuple(p.curve.start) for p in h.pieces]))))
                    for h in room.region.holes
                ]
                prof = w(
                    "IFCARBITRARYPROFILEDEFWITHVOIDS", E("AREA"), None, w.polyline(outer), voids
                )
            else:
                prof = w("IFCARBITRARYCLOSEDPROFILEDEF", E("AREA"), None, w.polyline(outer))
            spl = w.placement(pl_storey, (0.0, 0.0, 0.0))
            shape = w.product(
                [w.shape(self.body_ctx, "SweptSolid", [w.extrusion(prof, room.height)])]
            )
            sp = w(
                "IFCSPACE",
                global_id(item.id, "space", room.id),
                None,
                room.number or room.id,
                None,
                None,
                spl,
                shape,
                room.name,
                E("ELEMENT"),
                E("INTERNAL"),
                None,
            )
            self.spaces[r.level.name].append(sp)
            self.attach(
                item, f"space/{room.id}", sp, "Qto_SpaceBaseQuantities", space_quantities(room)
            )
            self.count("IfcSpace")

    def material(self, item, wr, b, wall: R) -> None:
        w = self.w
        t = b.wall_types[wr.type]
        if wr.type not in self.layer_sets:
            layers = t.get("layers") or [{"name": "core", "material": t.get("material", wr.type)}]
            ids = []
            for k, ly in enumerate(layers):
                thick = (
                    wr.line.thickness
                    if len(layers) == 1
                    else self.doc.evaluate_value(ly["thickness"], _LENGTH, f"wall_types.{wr.type}")
                )
                mat = w("IFCMATERIAL", str(ly.get("material", "Unknown")), None, None)
                ids.append(
                    w(
                        "IFCMATERIALLAYER",
                        mat,
                        float(thick),
                        None,
                        str(ly.get("name", f"layer {k}")),
                        None,
                        None,
                        None,
                    )
                )
            self.layer_sets[wr.type] = w("IFCMATERIALLAYERSET", ids, wr.type, None)
        _, ro = wr.line.offsets()
        usage = w(
            "IFCMATERIALLAYERSETUSAGE",
            self.layer_sets[wr.type],
            E("AXIS2"),
            E("POSITIVE"),
            float(ro),
            None,
        )
        w.rel("IFCRELASSOCIATESMATERIAL", global_id(item.id, "material", wr.id), [wall], usage)

    def opening(self, item, r, wr, o, wpl: R, wall: R) -> None:
        w = self.w
        lo, ro = wr.line.offsets()
        m = 0.01
        opl = w.placement(wpl, (o.center, ro - m, o.sill + o.height / 2), (0, 1, 0), (1, 0, 0))
        rect = w(
            "IFCRECTANGLEPROFILEDEF",
            E("AREA"),
            None,
            w("IFCAXIS2PLACEMENT2D", w.pt((0.0, 0.0)), w.dir((1, 0))),
            o.width,
            o.height,
        )
        shape = w.product(
            [w.shape(self.body_ctx, "SweptSolid", [w.extrusion(rect, lo - ro + 2 * m)])]
        )
        opening = w(
            "IFCOPENINGELEMENT",
            global_id(item.id, "opening", o.id),
            None,
            o.id,
            None,
            None,
            opl,
            shape,
            None,
            E("OPENING"),
        )
        w.rel("IFCRELVOIDSELEMENT", global_id(item.id, "voids", o.id), wall, opening)
        self.count("IfcOpeningElement")
        if o.body is None or o.kind == "opening":
            return
        x = (-1.0, 0.0, 0.0) if o.flip else (1.0, 0.0, 0.0)
        fpl = w.placement(wpl, (o.center, 0.0, o.sill), (0, 0, 1), x)
        shape = w.product([w.shape(self.body_ctx, "Brep", _faceted(w, o.body, lambda p: p))])
        gid = global_id(item.id, o.kind, o.id)
        if o.kind == "door":
            el = w(
                "IFCDOOR",
                gid,
                None,
                o.id,
                None,
                o.family,
                fpl,
                shape,
                None,
                o.height,
                o.width,
                E("DOOR"),
                E("SINGLE_SWING_LEFT"),
                None,
            )
            qto, kind = "Qto_DoorBaseQuantities", "IfcDoor"
        else:
            el = w(
                "IFCWINDOW",
                gid,
                None,
                o.id,
                None,
                o.family,
                fpl,
                shape,
                None,
                o.height,
                o.width,
                E("WINDOW"),
                E("SINGLE_PANEL"),
                None,
            )
            qto, kind = "Qto_WindowBaseQuantities", "IfcWindow"
        w.rel("IFCRELFILLSELEMENT", global_id(item.id, "fills", o.id), opening, el)
        self.contained[r.level.name].append(el)
        q = opening_quantities(o)
        q.pop("SillHeight", None)
        self.attach(item, f"{o.kind}/{o.id}", el, qto, q)
        self.count(kind)

    # -- other elements -------------------------------------------------------------

    def element(self, item, b) -> None:
        w, doc = self.w, self.doc
        r = doc.result(item)
        d = item._definition
        level = d.get("level") or d.get("from")
        pl_storey, elev = self.storey_pl[level]
        fr = b.frame

        def to_local(p):
            q = fr.to_local(p)
            return (float(q[0]), float(q[1]), float(q[2] - elev))

        kind = item.kind
        if kind in ("slab", "roof", "stair"):
            body = r.bodies[0][1]
            if r.sweep is not None:
                # A swept solid: the profile extruded (exact, and what IFC prefers).
                sf = r.sweep["frame"]
                epl = w.placement(
                    pl_storey,
                    to_local(sf.origin),
                    tuple(float(v) for v in fr.to_local(fr.origin + sf.z)),
                    tuple(float(v) for v in fr.to_local(fr.origin + sf.x)),
                )
                loops = [_ccw([tuple(map(float, q)) for q in lp]) for lp in r.sweep["loops"]]
                if len(loops) > 1:
                    voids = [w.polyline(list(reversed(lp))) for lp in loops[1:]]
                    prof = w(
                        "IFCARBITRARYPROFILEDEFWITHVOIDS",
                        E("AREA"),
                        None,
                        w.polyline(loops[0]),
                        voids,
                    )
                else:
                    prof = w("IFCARBITRARYCLOSEDPROFILEDEF", E("AREA"), None, w.polyline(loops[0]))
                shape = w.product(
                    [w.shape(self.body_ctx, "SweptSolid", [w.extrusion(prof, r.sweep["depth"])])]
                )
            else:
                epl = w.placement(pl_storey, (0.0, 0.0, 0.0))
                shape = w.product([w.shape(self.body_ctx, "Brep", _faceted(w, body, to_local))])
            gid = global_id(item.id, kind)
            if kind == "slab":
                el = w("IFCSLAB", gid, None, item.name, None, None, epl, shape, None, E("FLOOR"))
                self.attach(item, "slab", el, "Qto_SlabBaseQuantities", r.quantities)
            elif kind == "roof":
                rtype = {
                    "flat": "FLAT_ROOF",
                    "gable": "GABLE_ROOF",
                    "hip": "HIP_ROOF",
                    "shed": "SHED_ROOF",
                }[r.info["type"]]
                el = w("IFCROOF", gid, None, item.name, None, None, epl, shape, None, E(rtype))
                q = r.quantities
                self.attach(
                    item,
                    "roof",
                    el,
                    "Qto_RoofBaseQuantities",
                    {
                        "GrossArea": q["GrossArea"],
                        "NetArea": q["GrossArea"],
                        "ProjectedArea": q["ProjectedArea"],
                    },
                )
            else:
                el = w(
                    "IFCSTAIR",
                    gid,
                    None,
                    item.name,
                    None,
                    None,
                    epl,
                    shape,
                    None,
                    E("STRAIGHT_RUN_STAIR"),
                )
                q = r.quantities
                self.attach(
                    item,
                    "stair",
                    el,
                    None,
                    {},
                    {
                        "NumberOfRiser": int(q["NumberOfRisers"]),
                        "NumberOfTreads": int(q["NumberOfTreads"]),
                        "RiserHeight": float(q["RiserHeight"]),
                        "TreadLength": float(q["TreadLength"]),
                    },
                    "Pset_StairCommon",
                )
            self.contained[level].append(el)
            self.count(f"Ifc{kind.capitalize()}")
            return
        prof = self.profile(d.get("profile", {}))
        for name, body, m in r.bodies:
            if kind == "columns":
                length = r.quantities["Length"]
                xdir = fr.to_local(fr.origin + m[:3, 0])
                epl = w.placement(
                    pl_storey, to_local(m[:3, 3]), (0, 0, 1), tuple(float(v) for v in xdir)
                )
                shape = w.product(
                    [w.shape(self.body_ctx, "SweptSolid", [w.extrusion(prof, length)])]
                )
                el = w(
                    "IFCCOLUMN",
                    global_id(item.id, "column", name),
                    None,
                    f"{item.name} {name}",
                    None,
                    None,
                    epl,
                    shape,
                    None,
                    E("COLUMN"),
                )
                qto = "Qto_ColumnBaseQuantities"
                self.count("IfcColumn")
            else:
                lo, hi = body.bounds
                length = float(hi[0] - lo[0])
                xdir, ydir, zup = m[:3, 0], m[:3, 1], m[:3, 2]
                start = m[:3, 3] + zup * float(0.5 * (lo[2] + hi[2]))
                epl = w.placement(
                    pl_storey,
                    to_local(start),
                    tuple(float(v) for v in fr.to_local(fr.origin + xdir)),
                    tuple(float(v) for v in fr.to_local(fr.origin + ydir)),
                )
                shape = w.product(
                    [w.shape(self.body_ctx, "SweptSolid", [w.extrusion(prof, length)])]
                )
                el = w(
                    "IFCBEAM",
                    global_id(item.id, "beam", name),
                    None,
                    f"{item.name} {name}",
                    None,
                    None,
                    epl,
                    shape,
                    None,
                    E("BEAM"),
                )
                qto = "Qto_BeamBaseQuantities"
                self.count("IfcBeam")
            area = r.quantities["CrossSectionArea"]
            self.attach(
                item,
                f"{kind}/{name}",
                el,
                qto,
                {"Length": length, "CrossSectionArea": area, "NetVolume": area * length},
            )
            self.contained[level].append(el)

    def profile(self, p: dict) -> R:
        w = self.w

        def length(key, default=None):
            return self.doc.evaluate_value(p.get(key, default), _LENGTH, f"profile.{key}")

        pos = w("IFCAXIS2PLACEMENT2D", w.pt((0.0, 0.0)), w.dir((1, 0)))
        shape = p.get("shape", "rect")
        if shape == "rect":
            return w(
                "IFCRECTANGLEPROFILEDEF",
                E("AREA"),
                None,
                pos,
                length("width"),
                length("depth", p.get("width")),
            )
        if shape == "circle":
            return w("IFCCIRCLEPROFILEDEF", E("AREA"), None, pos, length("diameter") / 2)
        return w(
            "IFCISHAPEPROFILEDEF",
            E("AREA"),
            None,
            pos,
            length("width"),
            length("depth"),
            length("web"),
            length("flange"),
            None,
            None,
            None,
        )


from ..cad.units import LENGTH as _LENGTH  # noqa: E402


def write_ifc(
    doc, *, building: Any = None, name: str = "Aether3D model"
) -> tuple[str, dict[str, int]]:
    """IFC4 text of the document's buildings (or one ``building``) and entity counts."""
    buildings = [doc.item(building, "building")] if building is not None else doc.items("building")
    if not buildings:
        raise CadError("NOTHING_TO_EXPORT", "the document has no building")
    ex = _Exporter(doc, name)
    text = ex.run(buildings)
    return text, ex.stats


# --------------------------------------------------------------------------- #
# Import (the subset the export writes)
# --------------------------------------------------------------------------- #

PRODUCT_CLASSES = (
    "IFCWALL",
    "IFCWALLSTANDARDCASE",
    "IFCDOOR",
    "IFCWINDOW",
    "IFCSLAB",
    "IFCROOF",
    "IFCSTAIR",
    "IFCCOLUMN",
    "IFCBEAM",
    "IFCSPACE",
    "IFCOPENINGELEMENT",
    "IFCBUILDINGELEMENTPROXY",
)


@dataclass
class IfcElement:
    ifc_class: str
    global_id: str
    name: str
    storey: str | None
    body: Body | None
    properties: dict[str, dict[str, Any]]
    unsupported: list[str]


class _IfcReader:
    def __init__(self, text: str) -> None:
        from ..cad.interop.step import _parse_data

        if len(text) > 200 * 2**20:
            raise CadError("LIMIT_EXCEEDED", "IFC files are limited to 200 MB")
        if not text.lstrip().startswith("ISO-10303-21"):
            raise CadError("INVALID_IFC", "not an ISO 10303-21 file")
        schema = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text)
        if not schema or not schema.group(1).upper().startswith("IFC4"):
            raise CadError(
                "UNSUPPORTED_IFC",
                f"schema {schema.group(1) if schema else '?'} is not supported; IFC4 is",
            )
        self.e = _parse_data(text)
        self.memo: dict = {}
        self.scale = 1.0
        for parts in self.e.values():
            t, a = parts[0]
            if t == "IFCSIUNIT" and a[1] == ("enum", "LENGTHUNIT"):
                prefix = a[2]
                self.scale = {None: 1.0, ("enum", "MILLI"): 1e-3, ("enum", "CENTI"): 1e-2}.get(
                    prefix, 1.0
                )

    def get(self, ref) -> tuple[str, list]:
        parts = self.e.get(int(ref))
        if parts is None:
            raise CadError("INVALID_IFC", f"#{ref} is referenced but not defined")
        return parts[0]

    def point(self, ref) -> np.ndarray:
        _, a = self.get(ref)
        c = [float(v) for v in a[0]]
        return np.array(c + [0.0] * (3 - len(c)))

    def dir(self, ref) -> np.ndarray:
        _, a = self.get(ref)
        d = np.array([float(v) for v in a[0]] + [0.0] * (3 - len(a[0])))
        return d / np.linalg.norm(d)

    def a2p(self, ref) -> np.ndarray:
        if ref is None:
            return np.eye(4)
        t, a = self.get(ref)
        m = np.eye(4)
        o = self.point(a[0])
        if t == "IFCAXIS2PLACEMENT2D":
            x = self.dir(a[1]) if a[1] is not None else np.array([1.0, 0, 0])
            m[:3, 0], m[:3, 1], m[:3, 3] = x, np.cross([0, 0, 1.0], x), o
            return m
        z = self.dir(a[1]) if a[1] is not None else np.array([0, 0, 1.0])
        x = self.dir(a[2]) if a[2] is not None else np.array([1.0, 0, 0])
        x = x - (x @ z) * z
        x /= np.linalg.norm(x)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, o * self.scale
        return m

    def placement(self, ref, depth: int = 0) -> np.ndarray:
        if ref is None:
            return np.eye(4)
        if depth > 64:
            raise CadError("INVALID_IFC", "placements nest too deeply (a cycle?)")
        key = ("pl", int(ref))
        if key not in self.memo:
            _, a = self.get(ref)
            parent = self.placement(a[0], depth + 1) if a[0] is not None else np.eye(4)
            self.memo[key] = parent @ self.a2p(a[1])
        return self.memo[key]

    def polyline2d(self, ref) -> np.ndarray:
        t, a = self.get(ref)
        if t != "IFCPOLYLINE":
            raise CadError("UNSUPPORTED_IFC", f"profile curve {t}")
        pts = np.array([self.point(p)[:2] for p in a[0]]) * self.scale
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        return pts

    def profile(self, ref):
        from ..cad.brep import build
        from ..cad.geom2d import Arc
        from ..cad.sketch.profile import Loop, Piece, Region
        from .elements import _poly_region

        t, a = self.get(ref)
        if t == "IFCARBITRARYCLOSEDPROFILEDEF":
            return _poly_region(self.polyline2d(a[2]))
        if t == "IFCARBITRARYPROFILEDEFWITHVOIDS":
            reg = _poly_region(self.polyline2d(a[2]))
            holes = [_poly_region(self.polyline2d(h)).outer.reversed() for h in a[3]]
            return Region(reg.outer, holes)
        pos = self.a2p(a[2]) if len(a) > 2 else np.eye(4)
        if t == "IFCRECTANGLEPROFILEDEF":
            w, h = float(a[3]) * self.scale, float(a[4]) * self.scale
            reg = build.rectangle_region(-w / 2, -h / 2, w / 2, h / 2, "side")
        elif t == "IFCCIRCLEPROFILEDEF":
            r = float(a[3]) * self.scale
            reg = Region(Loop([Piece(Arc(np.zeros(2), r, 0.0, 2 * math.pi), "side", 0)]))
        elif t == "IFCISHAPEPROFILEDEF":
            w, h, tw, tf = (float(v) * self.scale for v in a[3:7])
            hw, hh = w / 2, h / 2
            pts = [
                (-hw, -hh),
                (hw, -hh),
                (hw, -hh + tf),
                (tw / 2, -hh + tf),
                (tw / 2, hh - tf),
                (hw, hh - tf),
                (hw, hh),
                (-hw, hh),
                (-hw, hh - tf),
                (-tw / 2, hh - tf),
                (-tw / 2, -hh + tf),
                (-hw, -hh + tf),
            ]
            reg = _poly_region(np.array(pts))
        else:
            raise CadError("UNSUPPORTED_IFC", f"profile {t}")
        if not np.allclose(pos, np.eye(4)):
            raise CadError("UNSUPPORTED_IFC", "positioned parametric profiles")
        return reg

    def item(self, ref) -> Body:
        from ..cad.brep import build
        from ..cad.geom3d import Frame

        t, a = self.get(ref)
        if t == "IFCEXTRUDEDAREASOLID":
            reg = self.profile(a[0])
            pos = self.a2p(a[1])
            d = self.dir(a[2])
            if not np.allclose(d, [0, 0, 1]):
                raise CadError("UNSUPPORTED_IFC", "oblique extrusions")
            depth = float(a[3]) * self.scale
            fr = Frame(pos[:3, 3], pos[:3, 0], pos[:3, 1], pos[:3, 2])
            return build.extrude(reg, fr, 0.0, depth)
        if t == "IFCFACETEDBREP":
            return self.faceted(a[0])
        raise CadError("UNSUPPORTED_IFC", f"representation item {t}")

    def faceted(self, shell_ref) -> Body:
        _, sa = self.get(shell_ref)
        polys = []
        for fr in sa[0]:
            _, fa = self.get(fr)
            loops = []
            for br in fa[0]:
                bt, ba = self.get(br)
                _, la = self.get(ba[0])
                pts = [self.point(p) * self.scale for p in la[0]]
                if ba[1] != ("enum", "T"):
                    pts = pts[::-1]
                loops.append((bt, pts))
            loops.sort(key=lambda x: x[0] != "IFCFACEOUTERBOUND")
            polys.append([p for _, p in loops])
        return polyhedron(polys)


def polyhedron(faces: list[list[list[np.ndarray]]]) -> Body:
    """A solid from planar polygonal faces (outer loop counter-clockwise from outside,
    then holes), vertices shared by position."""
    from ..cad.brep.build import Builder
    from ..cad.geom3d import Frame, Line3, Plane

    b = Builder()
    for k, loops in enumerate(faces):
        outer = np.array(loops[0])
        n = np.zeros(3)
        for i in range(len(outer)):  # Newell's normal
            p, q = outer[i], outer[(i + 1) % len(outer)]
            n += np.cross(p, q)
        if np.linalg.norm(n) == 0:
            raise CadError("INVALID_IFC", "a degenerate face")
        frame = Frame.from_axes(outer[0], n / np.linalg.norm(n), outer[1] - outer[0])
        coloops = []
        for lp in loops:
            pts = [np.asarray(p, float) for p in lp]
            vids = [b.vertex(tuple(np.round(p, 9)), p) for p in pts]
            co = []
            for i in range(len(pts)):
                a, c = vids[i], vids[(i + 1) % len(pts)]
                if a == c:
                    continue
                key = (min(a, c), max(a, c))
                pa, pc = b.verts[a], b.verts[c]
                co.append(b.edge(key, Line3(pa, pc - pa), a, c, 0.5 * (pa + pc)))
            coloops.append(co)
        b.face(Plane(frame), True, coloops, f"face{k}")
    return b.body()


def read_ifc(text: str) -> list[IfcElement]:
    """The products of an IFC4 file with their exact bodies (extrusions and faceted
    B-reps; openings subtracted from their hosts), property and quantity sets."""
    from ..cad.brep.boolean import boolean
    from .families import merge

    r = _IfcReader(text)
    products: dict[int, tuple[str, list]] = {}
    for eid, parts in r.e.items():
        t, a = parts[0]
        if t in PRODUCT_CLASSES:
            products[eid] = (t, a)
    storey_of: dict[int, str] = {}
    props: dict[int, dict[str, dict[str, Any]]] = {}
    voids: dict[int, list[int]] = {}
    for parts in r.e.values():
        t, a = parts[0]
        if t == "IFCRELCONTAINEDINSPATIALSTRUCTURE":
            _, sa = r.get(a[5])
            for el in a[4]:
                storey_of[int(el)] = sa[2]
        elif t == "IFCRELAGGREGATES":
            pt, pa = r.get(a[4])
            if pt == "IFCBUILDINGSTOREY":
                for el in a[5]:
                    storey_of[int(el)] = pa[2]
        elif t == "IFCRELVOIDSELEMENT":
            voids.setdefault(int(a[4]), []).append(int(a[5]))
        elif t == "IFCRELDEFINESBYPROPERTIES":
            dt, da = r.get(a[5])
            values: dict[str, Any] = {}
            if dt == "IFCPROPERTYSET":
                for pr in da[4]:
                    _, pv = r.get(pr)
                    v = pv[2]
                    values[pv[0]] = v.args[0] if hasattr(v, "args") else v
            elif dt == "IFCELEMENTQUANTITY":
                for q in da[5]:
                    _, qv = r.get(q)
                    values[qv[0]] = qv[3]
            for el in a[4]:
                props.setdefault(int(el), {})[da[2]] = {
                    k: (v[1] == "T" if isinstance(v, tuple) and v[0] == "enum" else v)
                    for k, v in values.items()
                }
    out = []
    for eid, (t, a) in sorted(products.items()):
        if t == "IFCOPENINGELEMENT":
            continue
        problems = []
        body = None
        rep = a[6]
        if rep is not None:
            m = r.placement(a[5])
            _, pd = r.get(rep)
            for sr in pd[2]:
                _, sa = r.get(sr)
                if sa[1] != "Body":
                    continue
                try:
                    pieces = [r.item(i).transformed(m) for i in sa[3]]
                    body = pieces[0] if len(pieces) == 1 else merge(pieces)
                except CadError as exc:
                    problems.append(exc.message)
            for op in voids.get(eid, []):
                ot, oa = products[op]
                om = r.placement(oa[5])
                _, opd = r.get(oa[6])
                for sr in opd[2]:
                    _, sa = r.get(sr)
                    for i in sa[3]:
                        if body is not None:
                            body = boolean(body, r.item(i).transformed(om), "difference")
        out.append(
            IfcElement(t, a[0], a[2] or "", storey_of.get(eid), body, props.get(eid, {}), problems)
        )
    return out
