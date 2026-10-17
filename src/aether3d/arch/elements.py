"""Slabs, roofs, stairs, columns and beams.

* ``slab`` -- a floor or ceiling plate: its boundary is a polygon or a floor's outline
  (the outside of its walls), with optional holes (stair openings); the top lies at the
  level's elevation plus ``top_offset``; ``thickness`` downwards.
* ``roof`` -- over a footprint (a polygon or a floor's outline, grown by ``overhang``):
  ``flat`` (a slab), ``shed`` (one plane rising across ``direction``), ``gable`` (two
  planes meeting at a ridge along ``ridge``: ``"x"`` or ``"y"``) or ``hip`` (every edge
  rises at the slope: the roof surface is the lower envelope of the edge planes, exact for
  convex footprints). The roof is a shell of vertical thickness ``thickness / cos(slope)``
  under that surface, built by exact planar booleans.
* ``stair`` -- a straight flight between two levels: the number of risers follows from
  the storey height and the target riser, every riser is equal; the run, the pitch and
  Blondel's 2R + T are checked against the building's ``criteria`` (the user's values --
  no building code is assumed). ``"enforce": true`` makes a failed check an error.
* ``columns`` / ``beams`` -- members with a rectangular, circular or I profile, placed at
  points (or grid intersections) or along lines; members of one size share one mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cad.brep import build
from ..cad.brep.boolean import boolean
from ..cad.brep.properties import face_area, mass_properties
from ..cad.brep.topology import Body
from ..cad.document import Context, DisplayPart, ItemKind, content_key, register
from ..cad.features import _Part, parameter_refs
from ..cad.geom2d import Arc, Segment
from ..cad.geom3d import Frame
from ..cad.offset import offset_loop
from ..cad.sketch.profile import Loop, Piece, Region
from ..cad.tolerance import Tolerance
from ..errors import CadError
from .kinds import ROLE_COLORS, BuildingResult, _prism

MAX_MEMBERS = 50_000


def _poly_region(poly: np.ndarray, name: str = "e") -> Region:
    area = 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    )
    if area < 0:
        poly = poly[::-1]
    n = len(poly)
    return Region(
        Loop([Piece(Segment(poly[i], poly[(i + 1) % n]), f"{name}{i}", 0) for i in range(n)])
    )


def _points2(ctx: Context, raw, loc: str) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) < 3:
        raise CadError(
            "INVALID_VALUE",
            "a boundary lists at least three [x, y] points",
            f"{ctx.location}.{loc}",
        )
    pts = []
    for i, p in enumerate(raw):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise CadError("INVALID_VALUE", "a point is [x, y]", f"{ctx.location}.{loc}[{i}]")
        pts.append([ctx.length(p[0], f"{loc}[{i}][0]"), ctx.length(p[1], f"{loc}[{i}][1]")])
    return np.array(pts)


def _outline(ctx: Context, d: dict, key: str, loc: str) -> Region:
    raw = d.get(key)
    if isinstance(raw, dict) and "floor" in raw:
        floor = ctx.inputs[ctx.doc.item(raw["floor"], "floor").id]
        if floor.outline is None:
            raise CadError(
                "NO_OUTLINE", "the floor's walls do not enclose an outline", f"{ctx.location}.{loc}"
            )
        return floor.outline
    return _poly_region(_points2(ctx, raw, loc))


def _building(ctx: Context, d: dict) -> BuildingResult:
    return ctx.inputs[ctx.doc.item(d["building"], "building").id]


def _refs(doc, d: dict, *keys) -> list[str]:
    refs = [doc.item(d["building"], "building").id]
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict) and isinstance(v.get("floor"), str):
            refs.append(doc.item(v["floor"], "floor").id)
    return refs + parameter_refs(doc, d)


@dataclass
class ElementResult:
    kind: str
    bodies: list[tuple[str, Body, np.ndarray]]  # (name, body in local coordinates, placement)
    quantities: dict[str, float]
    role: str
    info: dict = field(default_factory=dict)
    keys: list[str] = field(default_factory=list)
    # How the solid was swept (for exporters): {"frame": Frame, "loops": [outer, *holes]
    # (2D points in the frame), "depth": float}; None for solids built otherwise.
    sweep: dict | None = None


class _ElementKind(ItemKind):
    displays = True
    role = "slab"

    def check(self, doc, d, location):
        doc.item(d.get("building"), "building")
        self.references(doc, d)

    def display(self, doc, item, r: ElementResult):
        from ..cad.brep.tessellate import tessellate

        b = doc.result(doc.item(item._definition["building"], "building"))
        mat = item._definition.get("material") or b.materials.get(r.role)
        tol = doc.display_tolerance()
        parts = []
        for (name, body, place), key in zip(r.bodies, r.keys, strict=True):
            mesh = doc.mesh_for(key, lambda body=body: tessellate(body, tol).to_mesh())
            parts.append(
                DisplayPart(
                    mesh,
                    name,
                    {-1: mat} if mat else {},
                    ROLE_COLORS[r.role],
                    None if np.allclose(place, np.eye(4)) else place,
                )
            )
        if len(parts) == 1 and parts[0].transform is None:
            parts[0] = DisplayPart(parts[0].mesh, "", parts[0].materials, parts[0].color)
        return parts

    def info(self, doc, item, r: ElementResult):
        return {"quantities": r.quantities, **r.info}


@register
class SlabKind(_ElementKind):
    name = "slab"
    role = "slab"

    def references(self, doc, d):
        return _refs(doc, d, "outline")

    def evaluate(self, ctx: Context) -> ElementResult:
        d = ctx.item._definition
        b = _building(ctx, d)
        level = b.level(d.get("level"), f"{ctx.location}.level")
        t = ctx.length(d.get("thickness", "200 mm"), "thickness")
        if t <= 0:
            raise CadError(
                "INVALID_VALUE", "the thickness must be positive", f"{ctx.location}.thickness"
            )
        top = level.elevation + ctx.length(d.get("top_offset", 0), "top_offset")
        region = _outline(ctx, d, "outline" if "outline" in d else "boundary", "outline")
        holes = list(region.holes)
        for k, h in enumerate(d.get("holes", [])):
            holes.append(_poly_region(_points2(ctx, h, f"holes[{k}]"), f"h{k}_").outer.reversed())
        region = Region(region.outer, holes)
        frame = Frame(b.frame.origin, b.frame.x, b.frame.y, b.frame.z)
        body = build.extrude(region, frame, top - t, top, prefix=ctx.item.name)
        area = region.area
        perim = sum(p.curve.length for p in region.outer.pieces)
        q = {
            "Width": t,
            "GrossArea": region.outer.area,
            "NetArea": area,
            "Perimeter": perim,
            "GrossVolume": region.outer.area * t,
            "NetVolume": area * t,
        }
        loops2d = [np.array([p.curve.start for p in lp.pieces]) for lp in region.loops]
        key = content_key("slab", [lp.tolist() for lp in loops2d], top, t)
        sweep = {
            "frame": Frame(frame.origin + frame.z * (top - t), frame.x, frame.y, frame.z),
            "loops": loops2d,
            "depth": t,
        }
        return ElementResult(
            "slab", [(ctx.item.name, body, np.eye(4))], q, "slab", {"top": top}, [key], sweep
        )


def _halfspace_below(origin: np.ndarray, normal: np.ndarray, size: float) -> Body:
    """A big box whose top face lies on the plane (origin, normal): the half-space below."""
    n = normal / np.linalg.norm(normal)
    x = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    fr = Frame.from_axes(origin - n * size, n, x)
    return build.box((4 * size, 4 * size, 2 * size), frame=fr)


@register
class RoofKind(_ElementKind):
    name = "roof"
    role = "roof"

    def references(self, doc, d):
        return _refs(doc, d, "outline")

    def evaluate(self, ctx: Context) -> ElementResult:
        d = ctx.item._definition
        b = _building(ctx, d)
        level = b.level(d.get("level"), f"{ctx.location}.level")
        base = level.elevation + ctx.length(d.get("base_offset", 0), "base_offset")
        rtype = d.get("type", "gable")
        if rtype not in ("flat", "shed", "gable", "hip"):
            raise CadError(
                "INVALID_ROOF", "type is flat, shed, gable or hip", f"{ctx.location}.type"
            )
        t = ctx.length(d.get("thickness", "250 mm"), "thickness")
        slope = ctx.angle(d.get("slope", "30 deg"), "slope") if rtype != "flat" else 0.0
        if rtype != "flat" and not 0 < slope < math.radians(75):
            raise CadError(
                "INVALID_ROOF", "the slope must be within (0, 75) degrees", f"{ctx.location}.slope"
            )
        region = _outline(ctx, d, "outline" if "outline" in d else "boundary", "outline")
        loop = region.outer
        overhang = ctx.length(d.get("overhang", 0), "overhang")
        if overhang:
            tol = Tolerance.for_scale(100.0).resolution
            loop = offset_loop(loop, -overhang, tol)  # to the right of travel: outwards
        poly = np.array([p.curve.start for p in loop.pieces])
        frame = Frame(b.frame.origin, b.frame.x, b.frame.y, b.frame.z)
        span = float(np.max(np.ptp(poly, axis=0)))
        if rtype == "flat":
            body = build.extrude(_poly_region(poly), frame, base, base + t, prefix=ctx.item.name)
        else:
            d_vert = t / math.cos(slope)
            height = span * math.tan(slope) + 2 * d_vert + 1.0
            prism = _prism(poly, frame, base - d_vert, base + height, ctx.item.name, 0)
            # The solid taken away reaches past the footprint and below it: coplanar
            # boundaries in a difference are ill-defined (they would leave zero-thickness
            # sheets); this way every face of the shell is cut cleanly.
            margin = max(1.0, span)
            grown = offset_loop(
                _poly_region(poly).outer, -margin, Tolerance.for_scale(span).resolution
            )
            big_poly = np.array([p.curve.start for p in grown.pieces])
            big = _prism(big_poly, frame, base - d_vert - margin, base + height + margin, "cut", 0)
            planes = _roof_planes(poly, rtype, slope, base, d, ctx)
            size = 4 * (span + height + margin)
            upper = prism
            lower = big
            for o, n in planes:
                ow = frame.to_world(o)
                nw = frame.direction_to_world(n)
                upper = boolean(upper, _halfspace_below(ow, nw, size), "intersection")
                lower = boolean(
                    lower, _halfspace_below(ow - frame.z * d_vert, nw, size), "intersection"
                )
            body = boolean(upper, lower, "difference")
        sloped = 0.0
        for i, f in enumerate(body.faces):
            if f.surface.kind == "plane":
                nz = float(f.surface.frame.z[2]) * (1 if f.sense else -1)
                if nz > 1e-9:
                    sloped += face_area(body, i)
        footprint = 0.5 * abs(
            float(
                np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
            )
        )
        q = {
            "ProjectedArea": footprint,
            "GrossArea": sloped,
            "NetVolume": mass_properties(body).volume,
            "Slope_deg": math.degrees(slope),
        }
        key = content_key(
            "roof", poly.tolist(), rtype, slope, base, t, d.get("ridge"), d.get("direction")
        )
        return ElementResult(
            "roof", [(ctx.item.name, body, np.eye(4))], q, "roof", {"type": rtype}, [key]
        )


def _roof_planes(poly: np.ndarray, rtype: str, slope: float, base: float, d: dict, ctx: Context):
    """(point, upward normal) of each roof plane, in plan coordinates (z up)."""
    n = len(poly)
    area = 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    )
    if area < 0:
        poly = poly[::-1]
    tan = math.tan(slope)

    def edge_plane(a, b):
        e = (b - a) / np.linalg.norm(b - a)
        inward = np.array([-e[1], e[0]])  # left of a counter-clockwise edge
        # z = base + tan * (inward distance): normal (-tan * inward, 1).
        normal = np.array([-tan * inward[0], -tan * inward[1], 1.0])
        return np.array([a[0], a[1], base]), normal

    edges = [(poly[i], poly[(i + 1) % n]) for i in range(n)]
    if rtype == "hip":
        cross = [
            float((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0])
            for (a, b), (_, c) in zip(edges, edges[1:] + edges[:1], strict=True)
        ]
        if min(cross) < -1e-12:
            raise CadError(
                "ROOF_UNSUPPORTED", "hip roofs need a convex footprint", f"{ctx.location}.outline"
            )
        return [edge_plane(a, b) for a, b in edges]
    if rtype == "shed":
        dvec = d.get("direction", [0, 1])
        dirn = np.array([float(dvec[0]), float(dvec[1])])
        dirn /= np.linalg.norm(dirn)
        low = poly[int(np.argmin(poly @ dirn))]
        normal = np.array([-tan * dirn[0], -tan * dirn[1], 1.0])
        return [(np.array([low[0], low[1], base]), normal)]
    # Gable: the edges parallel to the ridge slope; the gable ends do not.
    ridge = d.get("ridge", "x")
    axis = np.array([1.0, 0.0]) if ridge == "x" else np.array([0.0, 1.0])
    planes = []
    for a, b in edges:
        e = (b - a) / np.linalg.norm(b - a)
        if abs(abs(float(e @ axis)) - 1.0) < 1e-9:
            planes.append(edge_plane(a, b))
    if len(planes) < 2:
        raise CadError(
            "ROOF_UNSUPPORTED",
            "a gable roof needs two eaves parallel to the ridge",
            f"{ctx.location}.ridge",
        )
    return planes


@register
class StairKind(_ElementKind):
    name = "stair"
    role = "stair"

    def check(self, doc, d, location):
        super().check(doc, d, location)
        for key in ("from", "to"):
            if not isinstance(d.get(key), str):
                raise CadError("INVALID_STAIR", f"a stair needs '{key}' (a level name)", location)

    def references(self, doc, d):
        return _refs(doc, d)

    def evaluate(self, ctx: Context) -> ElementResult:
        d = ctx.item._definition
        b = _building(ctx, d)
        lo = b.level(d["from"], f"{ctx.location}.from")
        hi = b.level(d["to"], f"{ctx.location}.to")
        rise = hi.elevation - lo.elevation
        if rise <= 0:
            raise CadError("INVALID_STAIR", "the stair must go up", ctx.location)
        part = _Part(ctx)
        width = ctx.length(d.get("width", "1000 mm"), "width")
        tread = ctx.length(d.get("tread", "280 mm"), "tread")
        target = ctx.length(d.get("riser", "175 mm"), "riser")
        n = max(2, round(rise / target))
        r = rise / n
        start = d.get("start", [0, 0])
        p0 = np.array([ctx.length(start[0], "start[0]"), ctx.length(start[1], "start[1]")])
        dvec = d.get("direction", [1, 0])
        u = np.array([part.number(dvec[0], "direction[0]"), part.number(dvec[1], "direction[1]")])
        u /= np.linalg.norm(u)
        run = (n - 1) * tread
        crit = b.criteria
        checks = []

        def check(name, value, lo_key, hi_key):
            lo_v, hi_v = crit.get(lo_key), crit.get(hi_key)
            ok = (lo_v is None or value >= lo_v - 1e-12) and (hi_v is None or value <= hi_v + 1e-12)
            checks.append({"check": name, "value": value, "min": lo_v, "max": hi_v, "pass": ok})

        check("riser", r, "riser_min", "riser_max")
        check("tread", tread, "tread_min", "tread_max")
        check("blondel_2r_plus_t", 2 * r + tread, "blondel_min", "blondel_max")
        check("width", width, "stair_width_min", "stair_width_max")
        failed = [c for c in checks if not c["pass"]]
        if failed and d.get("enforce"):
            raise CadError(
                "STAIR_CRITERIA",
                f"the stair fails the building's criteria: {', '.join(c['check'] for c in failed)}",
                ctx.location,
                details={"checks": checks},
                hint="change the target riser, the tread, the width or the criteria",
            )
        # Side profile (x along the run, z up), waist below the nosing line.
        waist = ctx.length(d.get("waist", "150 mm"), "waist")
        cos_a = tread / math.hypot(tread, r)
        prof = [(0.0, 0.0), (0.0, r)]
        for k in range(1, n):
            prof += [(k * tread, k * r), (k * tread, (k + 1) * r)]
        x_end = (n - 1) * tread
        drop = waist / cos_a
        z_end = n * r - drop
        x0 = (drop - r) * tread / r
        if 0 <= x0 < x_end and z_end > 0:
            prof += [(x_end, z_end), (x0, 0.0)]
        else:
            prof += [(x_end, 0.0)]
        pts = np.array(prof)
        # Remove collinear duplicates.
        clean = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - clean[-1]) > 1e-12:
                clean.append(p)
        region = _poly_region(np.array(clean), "s")
        side = np.array([-u[1], u[0], 0.0])
        o = b.frame.to_world(np.array([p0[0], p0[1], lo.elevation]))
        fr = Frame(
            o + b.frame.direction_to_world(side) * (width / 2),
            b.frame.direction_to_world(np.array([u[0], u[1], 0.0])),
            b.frame.z,
            -b.frame.direction_to_world(side),
        )
        body = build.extrude(region, fr, 0.0, width, prefix=ctx.item.name)
        q = {
            "NumberOfRisers": n,
            "NumberOfTreads": n - 1,
            "RiserHeight": r,
            "TreadLength": tread,
            "Run": run,
            "Width": width,
            "NetVolume": mass_properties(body).volume,
        }
        key = content_key("stair", np.array(clean).tolist(), width, o.tolist(), u.tolist())
        sweep = {
            "frame": fr,
            "loops": [np.array([p.curve.start for p in region.outer.pieces])],
            "depth": width,
        }
        return ElementResult(
            "stair",
            [(ctx.item.name, body, np.eye(4))],
            q,
            "stair",
            {"checks": checks, "criteria_met": not failed},
            [key],
            sweep,
        )


def _profile_region(ctx: Context, prof: dict, loc: str) -> Region:
    shape = prof.get("shape", "rect")
    if shape == "rect":
        w = ctx.length(prof.get("width"), f"{loc}.width")
        h = ctx.length(prof.get("depth", prof.get("width")), f"{loc}.depth")
        return build.rectangle_region(-w / 2, -h / 2, w / 2, h / 2, "side")
    if shape == "circle":
        r = ctx.length(prof.get("diameter"), f"{loc}.diameter") / 2
        return Region(Loop([Piece(Arc(np.zeros(2), r, 0.0, 2 * math.pi), "side", 0)]))
    if shape == "I":
        w = ctx.length(prof.get("width"), f"{loc}.width")
        h = ctx.length(prof.get("depth"), f"{loc}.depth")
        tw = ctx.length(prof.get("web"), f"{loc}.web")
        tf = ctx.length(prof.get("flange"), f"{loc}.flange")
        if not (0 < tw < w and 0 < 2 * tf < h):
            raise CadError(
                "INVALID_PROFILE",
                "an I profile needs web < width and 2 flange < depth",
                f"{ctx.location}.{loc}",
            )
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
        return _poly_region(np.array(pts), "flange")
    raise CadError("INVALID_PROFILE", "shape is rect, circle or I", f"{ctx.location}.{loc}.shape")


@register
class ColumnsKind(_ElementKind):
    name = "columns"
    role = "column"

    def references(self, doc, d):
        return _refs(doc, d)

    def evaluate(self, ctx: Context) -> ElementResult:
        d = ctx.item._definition
        b = _building(ctx, d)
        base_level = b.level(d.get("from"), f"{ctx.location}.from")
        if "height" in d:
            height = ctx.length(d["height"], "height")
        else:
            top = b.level(d.get("to"), f"{ctx.location}.to")
            height = top.elevation - base_level.elevation
        if height <= 0:
            raise CadError("INVALID_COLUMN", "columns need a positive height", ctx.location)
        region = _profile_region(ctx, d.get("profile", {}), "profile")
        pts = _member_points(ctx, d)
        body = build.extrude(region, Frame.world(), 0.0, height, prefix=ctx.item.name)
        key = content_key(
            "column", ctx.item._definition.get("profile"), height, ctx.doc.length_unit
        )
        bodies, keys = [], []
        for i, p in enumerate(pts):
            m = np.eye(4)
            m[:3, 3] = b.frame.to_world(np.array([p[0], p[1], base_level.elevation]))
            m[:3, :3] = b.frame.rotation
            bodies.append((f"{i}", body, m))
            keys.append(key)
        area = region.area
        q = {
            "Count": len(pts),
            "Length": height,
            "CrossSectionArea": area,
            "NetVolume": area * height * len(pts),
        }
        return ElementResult("columns", bodies, q, "column", {}, keys)


def _member_points(ctx: Context, d: dict) -> list[np.ndarray]:
    if "grid" in d:
        g = d["grid"]
        xs = [ctx.length(v, "grid.x") for v in g.get("x", [])]
        ys = [ctx.length(v, "grid.y") for v in g.get("y", [])]
        pts = [np.array([x, y]) for y in ys for x in xs]
    else:
        raw = d.get("points", [])
        pts = [np.array([ctx.length(p[0], "points"), ctx.length(p[1], "points")]) for p in raw]
    if not pts or len(pts) > MAX_MEMBERS:
        raise CadError("INVALID_VALUE", f"columns need 1..{MAX_MEMBERS} points", ctx.location)
    return pts


@register
class BeamsKind(_ElementKind):
    name = "beams"
    role = "beam"

    def references(self, doc, d):
        return _refs(doc, d)

    def evaluate(self, ctx: Context) -> ElementResult:
        d = ctx.item._definition
        b = _building(ctx, d)
        level = b.level(d.get("level"), f"{ctx.location}.level")
        top = level.elevation + ctx.length(d.get("top_offset", 0), "top_offset")
        region = _profile_region(ctx, d.get("profile", {}), "profile")
        lo, hi = region.outer.bbox()
        lines = d.get("lines", [])
        if not isinstance(lines, list) or not lines or len(lines) > MAX_MEMBERS:
            raise CadError("INVALID_VALUE", f"beams need 1..{MAX_MEMBERS} lines", ctx.location)
        bodies, keys = [], []
        total = 0.0
        for i, ln in enumerate(lines):
            a = np.array([ctx.length(ln[0][0], "lines"), ctx.length(ln[0][1], "lines")])
            c = np.array([ctx.length(ln[1][0], "lines"), ctx.length(ln[1][1], "lines")])
            length = float(np.linalg.norm(c - a))
            if length <= 0:
                raise CadError(
                    "INVALID_VALUE",
                    "a beam needs two distinct points",
                    f"{ctx.location}.lines[{i}]",
                )
            body = build.extrude(
                region,
                Frame.from_axes([0, 0, 0], [1, 0, 0], [0, 1, 0]),
                0.0,
                length,
                prefix=f"{ctx.item.name}.{i}",
            )
            u = (c - a) / length
            m = np.eye(4)
            x = b.frame.direction_to_world(np.array([u[0], u[1], 0.0]))
            z = b.frame.z
            y = np.cross(z, x)
            # The body runs along +x with the profile in its yz plane (profile x -> y,
            # profile y -> z): map +x to the beam's direction, +y to its plan normal, +z up;
            # the top of the profile lies at ``top``.
            m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
            m[:3, 3] = b.frame.to_world(np.array([a[0], a[1], top - hi[1]]))
            bodies.append((f"{i}", body, m))
            keys.append(content_key("beam", d.get("profile"), length, ctx.doc.length_unit))
            total += length
        q = {
            "Count": len(lines),
            "Length": total,
            "CrossSectionArea": region.area,
            "NetVolume": region.area * total,
        }
        return ElementResult("beams", bodies, q, "beam", {}, keys)
