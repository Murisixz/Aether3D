"""Architecture items: buildings, floors (walls, openings, rooms), slabs, roofs, stairs,
columns and beams -- parametric building elements with BIM semantics.

* ``building`` -- levels (name, elevation, storey height), grids, wall types (layers
  with thicknesses and materials), door and window families, the site placement, a
  material per role and design criteria (stair proportions the *user* requires).
* ``floor`` -- one level's walls (location lines given as points or as lines of a
  sketch, a wall type, height, base offset, location of the line), the openings hosted
  by them (doors, windows, bare openings: a family, a position along the wall, overrides)
  and its rooms (a name, a number and a point inside). Walls are joined in plan exactly
  (``walls.py``); openings are validated against their wall (inside its straight part,
  under its top, not overlapping); rooms are the plan regions bounded by wall faces
  around their points (net areas).
* ``slab`` (floor or ceiling), ``roof`` (flat, shed, gable, hip), ``stair`` (straight),
  ``columns`` and ``beams`` (many members per item, sharing meshes).

Every element is an exact B-rep (planar faces) built from its parameters; its
quantities are computed from the same parameters with the formulas of the IFC base
quantity sets, and checked against the B-rep's measured volume in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cad.brep import build
from ..cad.brep.boolean import boolean
from ..cad.brep.topology import Body
from ..cad.document import (
    Context,
    DisplayPart,
    ItemKind,
    content_key,
    register,
)
from ..cad.features import _Part, parameter_refs
from ..cad.geom2d import Segment
from ..cad.geom3d import Frame
from ..cad.sketch.profile import Loop, Piece, Region, build_profile
from ..cad.tolerance import Tolerance
from ..cad.units import unit
from ..errors import CadError
from .families import FRAME, GLASS, LEAF, build_filler, merge, spec_from
from .walls import WallLine, footprints

MAX_WALLS = 20_000
MAX_OPENINGS = 50_000
ROLE_COLORS = {
    "wall_exterior": (0.86, 0.84, 0.80),
    "wall_interior": (0.93, 0.92, 0.90),
    "frame": (0.25, 0.25, 0.27),
    "glass": (0.55, 0.70, 0.80),
    "leaf": (0.55, 0.40, 0.28),
    "slab": (0.75, 0.75, 0.73),
    "roof": (0.42, 0.30, 0.26),
    "stair": (0.70, 0.68, 0.64),
    "column": (0.72, 0.72, 0.72),
    "beam": (0.66, 0.66, 0.68),
}


def _len(ctx: Context, raw, loc: str) -> float:
    return ctx.length(raw, loc)


# --------------------------------------------------------------------------- #
# Buildings
# --------------------------------------------------------------------------- #


@dataclass
class Level:
    name: str
    elevation: float
    height: float


@dataclass
class BuildingResult:
    levels: dict[str, Level]
    order: list[str]
    wall_types: dict[str, dict]
    families: dict[str, dict]
    materials: dict[str, str]
    criteria: dict[str, float]
    frame: Frame  # site placement (plan x, y; z up)
    grids: list[dict]

    def level(self, name: str, location: str) -> Level:
        if name not in self.levels:
            raise CadError(
                "UNKNOWN_LEVEL",
                f"the building has no level {name!r}",
                location,
                hint=", ".join(self.order),
            )
        return self.levels[name]

    def level_above(self, name: str) -> Level | None:
        i = self.order.index(name)
        return self.levels[self.order[i + 1]] if i + 1 < len(self.order) else None


@register
class BuildingKind(ItemKind):
    name = "building"

    def check(self, doc, d, location):
        levels = d.get("levels")
        if not isinstance(levels, list) or not levels:
            raise CadError("INVALID_BUILDING", "a building needs a list of levels", location)
        names = set()
        for k, lv in enumerate(levels):
            if (
                not isinstance(lv, dict)
                or not isinstance(lv.get("name"), str)
                or lv["name"] in names
            ):
                raise CadError(
                    "INVALID_BUILDING", "each level needs a unique name", f"{location}.levels[{k}]"
                )
            names.add(lv["name"])
        for key in ("wall_types", "families", "materials", "criteria"):
            if not isinstance(d.get(key, {}), dict):
                raise CadError("INVALID_BUILDING", f"{key} must be an object", location)
        parameter_refs(doc, d)

    def references(self, doc, d):
        return parameter_refs(doc, d)

    def evaluate(self, ctx: Context) -> BuildingResult:
        d = ctx.item._definition
        part = _Part(ctx)
        levels, order = {}, []
        for k, lv in enumerate(d["levels"]):
            el = _len(ctx, lv.get("elevation", 0), f"levels[{k}].elevation")
            h = _len(ctx, lv.get("height", "3 m"), f"levels[{k}].height")
            if h <= 0:
                raise CadError(
                    "INVALID_LEVEL",
                    "a storey height must be positive",
                    f"{ctx.location}.levels[{k}]",
                )
            levels[lv["name"]] = Level(lv["name"], el, h)
            order.append(lv["name"])
        order.sort(key=lambda n: levels[n].elevation)
        for a, b in zip(order[:-1], order[1:], strict=True):
            if levels[b].elevation <= levels[a].elevation:
                raise CadError("INVALID_LEVEL", "level elevations must differ", ctx.location)
        types = {}
        for name, t in d.get("wall_types", {}).items():
            layers = t.get("layers")
            if layers:
                thick = sum(
                    _len(ctx, ly["thickness"], f"wall_types.{name}.layers") for ly in layers
                )
            else:
                thick = _len(ctx, t.get("thickness"), f"wall_types.{name}.thickness")
            if thick <= 0:
                raise CadError(
                    "INVALID_WALL_TYPE", f"wall type {name!r} has no thickness", ctx.location
                )
            types[name] = {**t, "_thickness": thick}
        site = d.get("site", {})
        frame = Frame.world()
        if site:
            origin = part.point(site.get("origin", [0, 0, 0]), "site.origin")
            north = part.angle(site.get("rotation", 0), "site.rotation")
            frame = Frame(
                origin,
                np.array([math.cos(north), math.sin(north), 0.0]),
                np.array([-math.sin(north), math.cos(north), 0.0]),
                np.array([0.0, 0.0, 1.0]),
            )
        crit = {}
        for key, raw in d.get("criteria", {}).items():
            crit[key] = _len(ctx, raw, f"criteria.{key}")
        grids = []
        for k, g in enumerate(d.get("grids", [])):
            grids.append(
                {
                    "name": g.get("name", str(k)),
                    "start": [_len(ctx, v, f"grids[{k}].start") for v in g["start"]],
                    "end": [_len(ctx, v, f"grids[{k}].end") for v in g["end"]],
                }
            )
        return BuildingResult(
            levels,
            order,
            types,
            dict(d.get("families", {})),
            dict(d.get("materials", {})),
            crit,
            frame,
            grids,
        )

    def info(self, doc, item, result: BuildingResult):
        s = float(unit(doc.length_unit).factor)
        return {
            "levels": [
                {
                    "name": n,
                    "elevation": result.levels[n].elevation / s,
                    "height": result.levels[n].height / s,
                }
                for n in result.order
            ],
            "wall_types": {k: v["_thickness"] / s for k, v in result.wall_types.items()},
            "families": sorted(result.families),
        }


# --------------------------------------------------------------------------- #
# Floors: walls, openings, rooms
# --------------------------------------------------------------------------- #


@dataclass
class WallResult:
    id: str
    type: str
    line: WallLine
    footprint: np.ndarray  # plan polygon (metres, building plan coordinates)
    clear: tuple[float, float]
    base: float  # absolute elevation of the wall's bottom (metres)
    height: float
    body: Body  # world coordinates
    openings: list[str]
    joins: list[dict]
    exterior: bool
    key: str  # content key of the wall's geometry

    def frame(self, site: Frame) -> Frame:
        d = self.line.direction
        o = site.to_world(np.array([self.line.start[0], self.line.start[1], self.base]))
        return Frame(
            o,
            site.direction_to_world(np.array([d[0], d[1], 0.0])),
            site.direction_to_world(np.array([-d[1], d[0], 0.0])),
            site.z,
        )


@dataclass
class OpeningResult:
    id: str
    wall: str
    family: str
    kind: str
    center: float  # metres along the wall line from its start
    width: float
    height: float
    sill: float
    placement: np.ndarray  # 4x4: the filler's local frame in the world
    body: Body | None  # filler in its local frame
    key: str
    flip: bool


@dataclass
class RoomResult:
    id: str
    name: str
    number: str
    region: Region
    area: float
    perimeter: float
    height: float
    elevation: float


@dataclass
class FloorResult:
    building: BuildingResult
    level: Level
    walls: dict[str, WallResult]
    openings: dict[str, OpeningResult]
    rooms: dict[str, RoomResult]
    outline: Region | None
    diagnostics: list[dict] = field(default_factory=list)


def _wall_points(ctx: Context, w: dict, k: int, sketches: dict) -> tuple[np.ndarray, np.ndarray]:
    loc = f"walls[{k}]({w.get('id')})"
    if "line" in w:
        ref = w["line"]
        if not isinstance(ref, dict) or "sketch" not in ref or "id" not in ref:
            raise CadError(
                "INVALID_WALL",
                'line is {"sketch": name, "id": line id}',
                f"{ctx.location}.{loc}.line",
            )
        sk = sketches[ctx.doc.item(ref["sketch"], "sketch").id]
        e = sk.entity(ref["id"])
        if e is None or e["type"] != "line":
            raise CadError(
                "REFERENCE_LOST",
                f"sketch {ref['sketch']!r} has no line {ref['id']!r}",
                f"{ctx.location}.{loc}.line",
            )
        a = sk.frame.to_world(np.array([*sk.point2(e["start"]), 0.0]))
        b = sk.frame.to_world(np.array([*sk.point2(e["end"]), 0.0]))
        return a[:2], b[:2]
    s, e = w.get("start"), w.get("end")
    for name, v in (("start", s), ("end", e)):
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise CadError("INVALID_WALL", f"{name} is [x, y]", f"{ctx.location}.{loc}.{name}")
    return (
        np.array([_len(ctx, s[0], f"{loc}.start[0]"), _len(ctx, s[1], f"{loc}.start[1]")]),
        np.array([_len(ctx, e[0], f"{loc}.end[0]"), _len(ctx, e[1], f"{loc}.end[1]")]),
    )


def _prism(poly: np.ndarray, frame: Frame, z0: float, z1: float, prefix: str, group: int) -> Body:
    n = len(poly)
    pieces = [Piece(Segment(poly[i], poly[(i + 1) % n]), f"e{i}", 0) for i in range(n)]
    return build.extrude(Region(Loop(pieces)), frame, z0, z1, prefix=prefix, group=group)


@register
class FloorKind(ItemKind):
    name = "floor"
    displays = True

    def check(self, doc, d, location):
        doc.item(d.get("building"), "building")
        if not isinstance(d.get("level"), str):
            raise CadError("INVALID_FLOOR", "a floor needs its level's name", location)
        walls = d.get("walls", [])
        ops = d.get("openings", [])
        rooms = d.get("rooms", [])
        if not all(isinstance(x, list) for x in (walls, ops, rooms)):
            raise CadError("INVALID_FLOOR", "walls, openings and rooms must be lists", location)
        if len(walls) > MAX_WALLS or len(ops) > MAX_OPENINGS:
            raise CadError("LIMIT_EXCEEDED", "too many walls or openings in one floor", location)
        ids = set()
        for group, items in (("walls", walls), ("openings", ops), ("rooms", rooms)):
            for k, x in enumerate(items):
                if (
                    not isinstance(x, dict)
                    or not isinstance(x.get("id"), str)
                    or not x["id"]
                    or x["id"] in ids
                ):
                    raise CadError(
                        "INVALID_FLOOR",
                        "every wall, opening and room needs a unique id",
                        f"{location}.{group}[{k}]",
                    )
                ids.add(x["id"])
        wall_ids = {w["id"] for w in walls}
        for k, o in enumerate(ops):
            if o.get("wall") not in wall_ids:
                raise CadError(
                    "UNKNOWN_WALL",
                    f"opening {o['id']!r} names no wall of this floor",
                    f"{location}.openings[{k}].wall",
                )
        self.references(doc, d)

    def references(self, doc, d):
        refs = [doc.item(d["building"], "building").id]
        for w in d.get("walls", []):
            line = w.get("line")
            if isinstance(line, dict) and isinstance(line.get("sketch"), str):
                refs.append(doc.item(line["sketch"], "sketch").id)
        return refs + parameter_refs(doc, d)

    def evaluate(self, ctx: Context) -> FloorResult:
        d = ctx.item._definition
        doc = ctx.doc
        b: BuildingResult = ctx.inputs[doc.item(d["building"], "building").id]
        level = b.level(d["level"], f"{ctx.location}.level")
        above = b.level_above(level.name)
        sketches = {k: v for k, v in ctx.inputs.items() if hasattr(v, "regions")}
        site = b.frame
        lines, wall_info = [], {}
        for k, w in enumerate(d.get("walls", [])):
            ctx.check()
            loc = f"walls[{k}]({w['id']})"
            tname = w.get("type")
            if tname is not None and tname not in b.wall_types:
                raise CadError(
                    "UNKNOWN_WALL_TYPE",
                    f"no wall type {tname!r}",
                    f"{ctx.location}.{loc}.type",
                    hint=", ".join(b.wall_types),
                )
            thick = (
                b.wall_types[tname]["_thickness"]
                if tname
                else _len(ctx, w.get("thickness", "200 mm"), f"{loc}.thickness")
            )
            a, e = _wall_points(ctx, w, k, sketches)
            loc_line = w.get("location", "center")
            if loc_line not in ("center", "left", "right"):
                raise CadError(
                    "INVALID_WALL",
                    "location is center, left or right",
                    f"{ctx.location}.{loc}.location",
                )
            lines.append(WallLine(w["id"], a, e, thick, loc_line))
            base = level.elevation + _len(ctx, w.get("base_offset", 0), f"{loc}.base_offset")
            if "height" in w:
                height = _len(ctx, w["height"], f"{loc}.height")
            elif above is not None:
                height = above.elevation - base
            else:
                height = level.height
            if height <= 0:
                raise CadError(
                    "INVALID_WALL",
                    "the wall's height must be positive",
                    f"{ctx.location}.{loc}.height",
                )
            exterior = (
                (b.wall_types.get(tname, {}).get("function") == "exterior")
                if tname
                else bool(w.get("exterior"))
            )
            wall_info[w["id"]] = (tname or "", base, height, exterior)
        tol = (
            Tolerance.for_scale(
                max(
                    [1.0] + [float(np.max(np.abs(np.concatenate([x.start, x.end])))) for x in lines]
                )
            ).resolution
            * 1e4
        )
        try:
            fps = footprints(lines, max(tol, 1e-6))
        except CadError as exc:
            raise CadError(
                exc.code, exc.message, f"{ctx.location}.walls[{exc.location}]", hint=exc.hint
            ) from None
        # Openings per wall.
        ops_by_wall: dict[str, list] = {x.id: [] for x in lines}
        openings: dict[str, OpeningResult] = {}
        line_by_id = {x.id: x for x in lines}
        for k, o in enumerate(d.get("openings", [])):
            loc = f"openings[{k}]({o['id']})"
            fam_name = o.get("family")
            fam = b.families.get(fam_name) if fam_name else {"kind": "opening"}
            if fam is None:
                raise CadError(
                    "UNKNOWN_FAMILY",
                    f"no family {fam_name!r}",
                    f"{ctx.location}.{loc}.family",
                    hint=", ".join(b.families),
                )
            spec = spec_from(
                fam,
                o,
                lambda raw, key, _l=loc: _len(ctx, raw, f"{_l}.{key}"),
                f"{ctx.location}.{loc}",
            )
            wl = line_by_id[o["wall"]]
            if "position" in o:
                c = _len(ctx, o["position"], f"{loc}.position")
            elif "offset" in o:
                c = _len(ctx, o["offset"], f"{loc}.offset") + spec.width / 2
            else:
                c = wl.length / 2
            a, bnd = fps[wl.id].clear
            if c - spec.width / 2 < a - 1e-9 or c + spec.width / 2 > bnd + 1e-9:
                raise CadError(
                    "OPENING_OUTSIDE_WALL",
                    f"opening {o['id']!r} ({spec.width:.4g} m wide at {c:.4g} m) does not fit "
                    f"in the straight part of wall {wl.id!r} ({a:.4g}..{bnd:.4g} m)",
                    f"{ctx.location}.{loc}",
                    hint="move it along the wall (position) or make it narrower",
                    details={
                        "wall_clear_range_m": [a, bnd],
                        "opening_m": [c - spec.width / 2, c + spec.width / 2],
                    },
                )
            _, wbase, wheight, _ = wall_info[wl.id]
            if spec.sill + spec.height > wheight + 1e-9:
                raise CadError(
                    "OPENING_TOO_TALL",
                    f"opening {o['id']!r} reaches {spec.sill + spec.height:.4g} m; "
                    f"wall {wl.id!r} is {wheight:.4g} m high",
                    f"{ctx.location}.{loc}",
                )
            for other in ops_by_wall[wl.id]:
                oc, ow = other
                if abs(c - oc) < (ow + spec.width) / 2 - 1e-9:
                    raise CadError(
                        "OPENING_OVERLAP",
                        f"openings overlap in wall {wl.id!r}",
                        f"{ctx.location}.{loc}",
                        details={"wall": wl.id},
                    )
            ops_by_wall[wl.id].append((c, spec.width))
            lo, ro = wl.offsets()
            wall_frame = Frame(
                site.to_world(np.array([*wl.start, wbase])),
                site.direction_to_world(np.array([*wl.direction, 0.0])),
                site.direction_to_world(np.array([*wl.normal, 0.0])),
                site.z,
            )
            place = wall_frame.matrix @ _translate(c, 0.0, spec.sill)
            if o.get("flip"):
                place = place @ np.diag([-1.0, -1.0, 1.0, 1.0])
            key = content_key("filler", spec.__dict__, ro, lo)
            body = build_filler(
                spec, ro if not o.get("flip") else -lo, lo if not o.get("flip") else -ro
            )
            openings[o["id"]] = OpeningResult(
                o["id"],
                wl.id,
                fam_name or "",
                spec.kind,
                c,
                spec.width,
                spec.height,
                spec.sill,
                place,
                body,
                key,
                bool(o.get("flip")),
            )
        # Wall solids: footprint prism minus the openings' boxes (one multi-lump tool).
        walls: dict[str, WallResult] = {}
        level_frame = Frame(site.to_world(np.zeros(3)), site.x, site.y, site.z)
        for wl in lines:
            ctx.check()
            tname, base, height, exterior = wall_info[wl.id]
            fp = fps[wl.id]
            mine = [o for o in openings.values() if o.wall == wl.id]
            key = content_key(
                "wall",
                fp.polygon.tolist(),
                base,
                height,
                [(o.center, o.width, o.height, o.sill) for o in mine],
                site.matrix.tolist(),
            )
            cache = doc.__dict__.setdefault("_wall_bodies", {})
            body = cache.get(key)
            if body is None:
                body = _prism(
                    fp.polygon, level_frame, base, base + height, f"{ctx.item.name}/{wl.id}", 0
                )
                if mine:
                    lo, ro = wl.offsets()
                    tools = []
                    for o in mine:
                        fr = o.placement[:3, :3]
                        corner = o.placement[:3, 3]
                        margin = 0.01 + wl.thickness
                        tool_frame = Frame(
                            corner + fr @ np.array([0.0, (lo + ro) / 2, o.height / 2]),
                            fr[:, 0],
                            fr[:, 1],
                            fr[:, 2],
                        )
                        tools.append(
                            build.box(
                                (o.width, (lo - ro) + 2 * margin, o.height),
                                frame=tool_frame,
                                prefix=f"{o.id}:opening",
                            )
                        )
                    body = boolean(body, merge(tools), "difference")
                if len(cache) > 50_000:
                    cache.clear()
                cache[key] = body
            walls[wl.id] = WallResult(
                wl.id,
                tname,
                wl,
                fp.polygon,
                fp.clear,
                base,
                height,
                body,
                [o.id for o in mine],
                fp.joins,
                exterior,
                key,
            )
        # Outline and rooms from the plan arrangement of wall footprints.
        outline, rooms, diags = _rooms(ctx, d, walls, level, above)
        return FloorResult(b, level, walls, openings, rooms, outline, diags)

    def display(self, doc, item, r: FloorResult):
        from ..cad.brep.tessellate import tessellate

        mats = r.building.materials
        tol = doc.display_tolerance()
        parts = []
        for w in r.walls.values():
            role = "wall_exterior" if w.exterior else "wall_interior"
            mesh = doc.mesh_for(
                w.key,
                lambda w=w: tessellate(w.body, tol).to_mesh(
                    groups=np.zeros(len(w.body.faces), int)
                ),
            )
            mat = r.building.wall_types.get(w.type, {}).get("material") or mats.get(role)
            parts.append(DisplayPart(mesh, w.id, {-1: mat} if mat else {}, ROLE_COLORS[role]))
        for o in r.openings.values():
            if o.body is None:
                continue
            mesh = doc.mesh_for(
                o.key,
                lambda o=o: tessellate(o.body, tol).to_mesh(
                    groups=np.array([f.group for f in o.body.faces])
                ),
            )
            m = {}
            for g, role in ((FRAME, "frame"), (GLASS, "glass"), (LEAF, "leaf")):
                if mats.get(role):
                    m[g] = mats[role]
            parts.append(DisplayPart(mesh, o.id, m, ROLE_COLORS["frame"], o.placement))
        if not parts:
            return []
        return parts

    def info(self, doc, item, r: FloorResult):
        from .quantities import floor_quantities

        return floor_quantities(doc, r)


def _translate(x, y, z) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def _rooms(ctx: Context, d: dict, walls: dict, level: Level, above: Level | None):
    rooms: dict[str, RoomResult] = {}
    diags: list[dict] = []
    if not walls:
        return None, rooms, diags
    # A synthetic sketch of every footprint edge (shared edges once).
    seen: dict[tuple, int] = {}
    ents = []
    pts: dict[tuple, str] = {}

    def pid(p):
        key = (round(float(p[0]), 9), round(float(p[1]), 9))
        if key not in pts:
            pts[key] = f"p{len(pts)}"
            ents.append({"id": pts[key], "type": "point", "at": [float(p[0]), float(p[1])]})
        return pts[key]

    for w in walls.values():
        poly = w.footprint
        n = len(poly)
        for i in range(n):
            a, b = pid(poly[i]), pid(poly[(i + 1) % n])
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen[key] = len(ents)
            ents.append({"id": f"l{len(ents)}", "type": "line", "start": a, "end": b})
    defn = {"entities": ents}
    try:
        outline = build_profile(defn).regions[0]
    except CadError as exc:
        diags.append({"code": "NO_OUTLINE", "message": exc.message})
        outline = None
    height = (above.elevation - level.elevation) if above else level.height
    points = []
    for k, room in enumerate(d.get("rooms", [])):
        loc = f"{ctx.location}.rooms[{k}]({room['id']})"
        p = room.get("point")
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise CadError("INVALID_ROOM", "a room needs a point [x, y] inside it", loc)
        points.append(
            [_len(ctx, p[0], f"rooms[{k}].point[0]"), _len(ctx, p[1], f"rooms[{k}].point[1]")]
        )
    # One arrangement of the wall faces for every room (each point selects its face).
    regions = []
    if points:
        try:
            regions = build_profile(defn, select=points).regions
        except CadError as exc:
            k = _region_index(exc.location)
            room = d["rooms"][k] if k is not None else {"id": "?"}
            raise CadError(
                "ROOM_NOT_ENCLOSED",
                f"room {room.get('name', room['id'])!r}: {exc.message}",
                f"{ctx.location}.rooms[{k}]({room['id']})" if k is not None else ctx.location,
                hint="close the walls around it (rooms are bounded by wall faces)",
            ) from None
    for k, room in enumerate(d.get("rooms", [])):
        loc = f"{ctx.location}.rooms[{k}]({room['id']})"
        q, reg = points[k], regions[k]
        # The point must not be inside a wall.
        for w in walls.values():
            if _inside(q, w.footprint):
                raise CadError(
                    "ROOM_IN_WALL",
                    f"room {room.get('name', room['id'])!r} has its point inside wall {w.id!r}",
                    loc,
                )
        perim = sum(p.curve.length for lp in reg.loops for p in lp.pieces)
        rooms[room["id"]] = RoomResult(
            room["id"],
            room.get("name", room["id"]),
            str(room.get("number", "")),
            reg,
            reg.area,
            perim,
            height,
            level.elevation,
        )
    return outline, rooms, diags


def _region_index(location: str | None) -> int | None:
    """``k`` of a ``...regions[k]`` location (the failing selection point), or None."""
    import re

    m = re.search(r"regions\[(\d+)\]", location or "")
    return int(m.group(1)) if m else None


def _inside(q, poly) -> bool:
    x, y = q
    xs, ys = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool(np.count_nonzero(cond & (x < xc)) % 2)


from . import api as _api  # noqa: E402,F401  (building handles: doc.building)
from . import elements as _elements  # noqa: E402,F401  (registers slab, roof, stair, ...)
