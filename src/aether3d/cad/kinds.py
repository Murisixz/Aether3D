"""Item kinds of the CAD document: parameters, frames, sketches and parts, with their
Python handles (``doc.parameter``, ``doc.frame``, ``doc.sketch``, ``doc.part``)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import CadError
from . import features as feat
from .document import KINDS, CadDocument, Context, DisplayPart, Item, ItemKind, register
from .geom2d import Arc, Segment
from .geom3d import Frame
from .sketch.model import VALUE_TYPES, check_id, infer_choices, validate
from .sketch.profile import build_profile, sketch_curves
from .sketch.solve import SketchSolution, solve_sketch
from .units import ANGLE, LENGTH, Quantity, format_quantity, parse_quantity, unit

PLANES = {
    "XY": ([0, 0, 1], [1, 0, 0]),
    "XZ": ([0, -1, 0], [1, 0, 0]),
    "YZ": ([1, 0, 0], [0, 1, 0]),
}


def _unit_factor(doc: CadDocument) -> float:
    return float(unit(doc.length_unit).factor)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@register
class ParameterKind(ItemKind):
    name = "parameter"

    def check(self, doc, definition, location):
        if "value" not in definition:
            raise CadError("INVALID_PARAMETER", "a parameter needs a value", location)
        v = definition["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise CadError(
                "INVALID_PARAMETER", "a value is a number, quantity text or =expression", location
            )
        kind = definition.get("dimension", "length")
        if kind not in ("length", "angle", "ratio", "area", "volume", "mass"):
            raise CadError(
                "INVALID_PARAMETER", "dimension: length, angle, ratio, area, volume, mass", location
            )
        self.references(doc, definition)

    def references(self, doc, definition):
        return feat.parameter_refs(doc, definition.get("value"))

    def evaluate(self, ctx: Context) -> Quantity:
        from .expr import evaluate_value
        from .units import DIMENSIONS_BY_NAME

        d = ctx.item._definition
        dim = DIMENSIONS_BY_NAME[d.get("dimension", "length")]
        default = ctx.doc.length_unit if dim == LENGTH else "deg" if dim == ANGLE else None
        try:
            return evaluate_value(
                d["value"], ctx.doc.parameter_value, expected=dim, default_unit=default
            )
        except CadError as exc:
            raise CadError(exc.code, exc.message, ctx.location, hint=exc.hint) from None

    def info(self, doc, item, result):
        return {
            "value": result.value,
            "dimension": result.dimension.name,
            "display": format_quantity(
                result, doc.length_unit if result.dimension == LENGTH else None
            ),
        }


# --------------------------------------------------------------------------- #
# Frames (datum coordinate systems / planes)
# --------------------------------------------------------------------------- #


@register
class FrameKind(ItemKind):
    name = "frame"

    def check(self, doc, definition, location):
        base = definition.get("base")
        if base is not None and not isinstance(base, str):
            raise CadError("INVALID_FRAME", "base is a frame name", location)
        if "plane" in definition and definition["plane"] not in PLANES:
            raise CadError("INVALID_FRAME", "plane is XY, XZ or YZ", location)
        self.references(doc, definition)

    def references(self, doc, definition):
        refs = feat.parameter_refs(doc, definition)
        if isinstance(definition.get("base"), str):
            refs.append(doc.item(definition["base"], "frame").id)
        return refs

    def evaluate(self, ctx: Context) -> Frame:
        return frame_from_spec(ctx, ctx.item._definition, "")

    def info(self, doc, item, result):
        s = _unit_factor(doc)
        return {
            "origin": (result.origin / s).tolist(),
            "x": result.x.tolist(),
            "z": result.z.tolist(),
        }


def frame_from_spec(ctx: Context, d: dict, loc: str) -> Frame:
    part = feat._Part(ctx)
    if isinstance(d.get("base"), str):
        base = ctx.inputs[ctx.doc.item(d["base"], "frame").id]
    elif "plane" in d:
        z, x = PLANES[d["plane"]]
        base = Frame.from_axes([0, 0, 0], z, x)
    elif "z" in d or "origin" in d:
        origin = part.point(d.get("origin", [0, 0, 0]), f"{loc}origin")
        z = part.direction(d.get("z", [0, 0, 1]), f"{loc}z")
        x = d.get("x")
        base = Frame.from_axes(origin, z, None if x is None else part.direction(x, f"{loc}x"))
    else:
        base = Frame.world()
    if d.get("offset") is not None:
        base = base.offset(part.length(d["offset"], f"{loc}offset"))
    rot = d.get("rotate")
    if isinstance(rot, dict):
        axis = rot.get("axis", "x")
        local = {"x": base.x, "y": base.y, "z": base.z}.get(axis)
        if local is None:
            local = part.direction(axis, f"{loc}rotate.axis")
        m = feat._rot(local, part.angle(rot.get("angle", 0), f"{loc}rotate.angle"), base.origin)
        base = base.transformed(m)
    if d.get("translate") is not None:
        t = np.eye(4)
        t[:3, 3] = part.point(d["translate"], f"{loc}translate")
        base = base.transformed(t)
    return base


# --------------------------------------------------------------------------- #
# Sketches
# --------------------------------------------------------------------------- #


@dataclass
class SketchResult:
    frame: Frame
    solution: SketchSolution
    values: dict[str, float]
    _regions: dict = field(default_factory=dict)

    @property
    def definition(self) -> dict:
        return self.solution.definition

    def entity(self, eid: str) -> dict | None:
        for e in self.definition["entities"]:
            if e["id"] == eid:
                return e
        return None

    def point2(self, pid: str) -> list[float]:
        e = self.entity(pid)
        if e is None or e["type"] != "point":
            raise CadError("UNKNOWN_ENTITY", f"the sketch has no point {pid!r}")
        return e["at"]

    def point3(self, pid: str) -> np.ndarray:
        p = self.point2(pid)
        return self.frame.to_world(np.array([p[0], p[1], 0.0]))

    def regions(self, select):
        key = None if select is None else tuple(map(tuple, select))
        if key not in self._regions:
            self._regions[key] = build_profile(self.definition, select=select).regions
        return self._regions[key]

    def open_chain(self) -> list:
        """The path of a sweep: the non-construction lines and arcs chained end to end
        (from an end used by one curve only, if there is one)."""
        curves = [c for _, c in sketch_curves(self.definition) if isinstance(c, (Segment, Arc))]
        if not curves:
            return []
        size = max(float(np.max(np.abs(np.concatenate([c.start, c.end])))) for c in curves)
        tol = 1e-9 * max(1.0, size)

        def same(p, q) -> bool:
            return float(np.linalg.norm(p - q)) <= tol

        def uses(p) -> int:
            return sum(same(p, c.start) + same(p, c.end) for c in curves)

        first = curves[0]
        for c in curves:
            if uses(c.start) == 1:
                first = c
                break
            if uses(c.end) == 1:
                first = c.reversed()
                break
        left = [
            c
            for c in curves
            if not (same(c.start, first.start) and same(c.end, first.end))
            and not (same(c.start, first.end) and same(c.end, first.start))
        ]
        chain = [first]
        while left:
            end = chain[-1].end
            for c in left:
                if same(c.start, end):
                    chain.append(c)
                    left.remove(c)
                    break
                if same(c.end, end):
                    chain.append(c.reversed())
                    left.remove(c)
                    break
            else:
                raise CadError("INVALID_PATH", "the path's curves do not form one chain")
        for a, b in zip(chain[:-1], chain[1:], strict=True):
            ta, tb = a.tangent(1.0), b.tangent(0.0)
            cross = abs(ta[0] * tb[1] - ta[1] * tb[0])
            if cross > 1e-9 * np.linalg.norm(ta) * np.linalg.norm(tb) or ta @ tb <= 0:
                raise CadError("INVALID_PATH", "a sweep path must be tangent at its joints")
        return chain


@register
class SketchKind(ItemKind):
    name = "sketch"

    def check(self, doc, definition, location):
        validate(definition, location)
        plane = definition.get("plane", "XY")
        if isinstance(plane, str) and plane not in PLANES:
            doc.item(plane, "frame")
        elif isinstance(plane, dict):
            pass
        elif not isinstance(plane, str):
            raise CadError(
                "INVALID_SKETCH", "plane is XY, XZ, YZ, a frame name or a frame spec", location
            )
        self.references(doc, definition)

    def references(self, doc, definition):
        refs = feat.parameter_refs(doc, definition.get("constraints", []))
        plane = definition.get("plane", "XY")
        if isinstance(plane, str) and plane not in PLANES:
            refs.append(doc.item(plane, "frame").id)
        elif isinstance(plane, dict):
            refs += feat.parameter_refs(doc, plane)
            if isinstance(plane.get("base"), str):
                refs.append(doc.item(plane["base"], "frame").id)
        return refs

    def evaluate(self, ctx: Context) -> SketchResult:
        d = ctx.item._definition
        plane = d.get("plane", "XY")
        if isinstance(plane, str) and plane in PLANES:
            z, x = PLANES[plane]
            frame = Frame.from_axes([0, 0, 0], z, x)
        elif isinstance(plane, str):
            frame = ctx.inputs[ctx.doc.item(plane, "frame").id]
        else:
            frame = frame_from_spec(ctx, plane, "plane.")
        values = {}
        for k, c in enumerate(d.get("constraints", [])):
            if c["type"] in VALUE_TYPES and c.get("driving", True) is not False:
                expected = ANGLE if c["type"] == "angle" else LENGTH
                values[c["id"]] = ctx.value(
                    c["value"], expected, f"constraints[{k}]({c['id']}).value"
                )
        prev = ctx.previous
        sol = solve_sketch(
            d,
            values,
            previous_values=getattr(prev, "values", None),
            deadline=ctx.deadline,
            should_stop=ctx.should_stop,
            structure_cache=ctx.doc.__dict__.setdefault("_structures", {}),
            location=ctx.location,
        )
        if sol.status == "conflicting":
            first = sol.conflicts[0] if sol.conflicts else {}
            raise CadError(
                "SKETCH_CONFLICT",
                f"the sketch cannot satisfy its constraints: {first.get('constraint')!r} conflicts "
                f"with {first.get('conflicts_with')}",
                ctx.location,
                hint="remove or change one of the conflicting constraints (or make it a "
                "reference dimension: driving=False)",
                details={
                    "conflicts": sol.conflicts,
                    "dof": sol.dof,
                    "max_residual": sol.max_residual,
                },
            )
        return SketchResult(frame, sol, values)

    def post_evaluate(self, doc, item, result: SketchResult):
        return result.solution.definition

    def info(self, doc, item, result: SketchResult):
        s = _unit_factor(doc)
        out = result.solution.to_dict()
        out["reference"] = {k: _display(doc, item, k, v) for k, v in out["reference"].items()}
        out["frame"] = {
            "origin": (result.frame.origin / s).tolist(),
            "z": result.frame.z.tolist(),
            "x": result.frame.x.tolist(),
        }
        out["entities"] = len(result.definition["entities"])
        out["constraints"] = len(result.definition.get("constraints", []))
        try:
            regs = result.regions(None)
            out["regions"] = [{"area": r.area / (s * s), "holes": len(r.holes)} for r in regs]
        except CadError as exc:
            out["regions"] = []
            out["profile_error"] = exc.to_dict()
        return out


def _display(doc, item, cid, value):
    c = next((c for c in item._definition.get("constraints", []) if c["id"] == cid), None)
    if c is not None and c["type"] == "angle":
        return format_quantity(Quantity(value, ANGLE))
    return format_quantity(
        Quantity(value, LENGTH), doc.length_unit, precision=doc._settings["precision"]
    )


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #


@register
class PartKind(ItemKind):
    name = "part"
    displays = True

    def check(self, doc, definition, location):
        feat.check(doc, definition, location)
        mats = definition.get("materials", {})
        if not isinstance(mats, dict) or not all(isinstance(v, str) for v in mats.values()):
            raise CadError("INVALID_PART", "materials maps groups to material names", location)

    def references(self, doc, definition):
        return feat.references(doc, definition)

    def evaluate(self, ctx: Context) -> feat.PartResult:
        cache = ctx.doc.__dict__.setdefault("_feature_cache", {})
        return feat.regenerate(ctx, ctx.item._definition, cache)

    def display(self, doc, item, result: feat.PartResult):
        parts = []
        mats = {
            int(k) if k != "all" else -1: v
            for k, v in item._definition.get("materials", {}).items()
        }
        color = item._definition.get("color")
        for name in sorted(result.bodies):
            mesh = doc.body_mesh(item, name)
            parts.append(
                DisplayPart(mesh, name, mats, tuple(color) if color else (0.62, 0.65, 0.7))
            )
        return parts if len(parts) != 1 else [DisplayPart(parts[0].mesh, "", mats, parts[0].color)]

    def info(self, doc, item, result: feat.PartResult):
        from .brep.properties import mass_properties
        from .brep.topology import validate as validate_body

        s = _unit_factor(doc)
        bodies = {}
        for name, b in result.bodies.items():
            m = mass_properties(b)
            v = validate_body(b)
            bodies[name] = {
                **b.summary(),
                "valid": v["valid"],
                "volume": m.volume / s**3,
                "area": m.area / s**2,
                "centroid": (m.centroid / s).tolist(),
                "volume_estimated_error": m.error_volume / s**3,
                "notes": list(b.notes),
            }
        return {"features": result.features, "bodies": bodies, "units": doc.length_unit}


# --------------------------------------------------------------------------- #
# Handles: a Python API over item definitions
# --------------------------------------------------------------------------- #


class SketchHandle:
    """Build and change a sketch. Coordinates and lengths are numbers in the document's
    length unit or quantity text; angles numbers in degrees or text. Every call is one
    undoable step (wrap several in ``doc.batch()``)."""

    def __init__(self, doc: CadDocument, item: Item) -> None:
        self.doc = doc
        self.item = item

    @property
    def name(self) -> str:
        return self.item.name

    def _len(self, v) -> float:
        return parse_quantity(v, expected=LENGTH, default_unit=self.doc.length_unit).value

    def _defn(self) -> dict:
        return copy.deepcopy(self.item._definition)

    def _ids(self, defn) -> set[str]:
        return {e["id"] for e in defn.get("entities", [])} | {
            c["id"] for c in defn.get("constraints", [])
        }

    def _new_id(self, defn, base: str) -> str:
        used = self._ids(defn)
        k = 1
        while f"{base}{k}" in used:
            k += 1
        return f"{base}{k}"

    def _commit(self, defn, label) -> None:
        self.doc.set_definition(self.item, defn, label=f"{label} ({self.item.name})")

    def point(self, x, y, *, id: str | None = None, construction: bool = False) -> str:  # noqa: A002
        defn = self._defn()
        pid = id or self._new_id(defn, "p")
        check_id(pid, "id")
        e = {"id": pid, "type": "point", "at": [self._len(x), self._len(y)]}
        if construction:
            e["construction"] = True
        defn.setdefault("entities", []).append(e)
        self._commit(defn, "Add point")
        return pid

    def _pt(self, defn, p, made: list) -> str:
        if isinstance(p, str):
            return p
        pid = self._new_id(defn, "p")
        defn.setdefault("entities", []).append(
            {"id": pid, "type": "point", "at": [self._len(p[0]), self._len(p[1])]}
        )
        made.append(pid)
        return pid

    def line(self, a, b, *, id: str | None = None, construction: bool = False) -> str:  # noqa: A002
        defn = self._defn()
        made: list = []
        pa, pb = self._pt(defn, a, made), self._pt(defn, b, made)
        lid = id or self._new_id(defn, "l")
        e = {"id": lid, "type": "line", "start": pa, "end": pb}
        if construction:
            e["construction"] = True
        defn["entities"].append(e)
        self._commit(defn, "Add line")
        return lid

    def polyline(self, points, *, closed: bool = True, prefix: str = "l") -> list[str]:
        """Lines through the points, sharing their end points (closed by default)."""
        defn = self._defn()
        made: list = []
        pids = [self._pt(defn, p, made) for p in points]
        ids = []
        n = len(pids)
        for i in range(n if closed else n - 1):
            lid = self._new_id(defn, prefix)
            defn["entities"].append(
                {"id": lid, "type": "line", "start": pids[i], "end": pids[(i + 1) % n]}
            )
            ids.append(lid)
        self._commit(defn, "Add polyline")
        return ids

    def rectangle(self, corner, size, *, constrain: bool = True, prefix: str = "r") -> list[str]:
        """An axis-aligned rectangle (four lines); with ``constrain``, horizontal/vertical."""
        x0, y0 = self._len(corner[0]), self._len(corner[1])
        w, h = self._len(size[0]), self._len(size[1])
        u = float(unit(self.doc.length_unit).factor)
        pts = [
            (x0 / u, y0 / u),
            ((x0 + w) / u, y0 / u),
            ((x0 + w) / u, (y0 + h) / u),
            (x0 / u, (y0 + h) / u),
        ]
        with self.doc.batch(f"Add rectangle ({self.item.name})"):
            lines = self.polyline(pts, prefix=prefix)
            if constrain:
                for i, lid in enumerate(lines):
                    self.constrain("horizontal" if i % 2 == 0 else "vertical", lid)
        return lines

    def circle(self, center, radius, *, id: str | None = None, construction: bool = False) -> str:  # noqa: A002
        defn = self._defn()
        made: list = []
        c = self._pt(defn, center, made)
        cid = id or self._new_id(defn, "c")
        e = {"id": cid, "type": "circle", "center": c, "radius": self._len(radius)}
        if construction:
            e["construction"] = True
        defn["entities"].append(e)
        self._commit(defn, "Add circle")
        return cid

    def arc(self, center, start, end, *, id: str | None = None, construction: bool = False) -> str:  # noqa: A002
        """A counter-clockwise arc about ``center`` from ``start`` to ``end``."""
        defn = self._defn()
        made: list = []
        c, s, e_ = (
            self._pt(defn, center, made),
            self._pt(defn, start, made),
            self._pt(defn, end, made),
        )
        pts = {x["id"]: x["at"] for x in defn["entities"] if x["type"] == "point"}
        r = float(np.hypot(*(np.array(pts[s]) - np.array(pts[c]))))
        aid = id or self._new_id(defn, "a")
        ent = {"id": aid, "type": "arc", "center": c, "start": s, "end": e_, "radius": r}
        if construction:
            ent["construction"] = True
        defn["entities"].append(ent)
        self._commit(defn, "Add arc")
        return aid

    def spline(self, points, *, degree: int = 3, id: str | None = None) -> str:  # noqa: A002
        defn = self._defn()
        made: list = []
        pids = [self._pt(defn, p, made) for p in points]
        sid = id or self._new_id(defn, "s")
        defn["entities"].append({"id": sid, "type": "spline", "degree": degree, "points": pids})
        self._commit(defn, "Add spline")
        return sid

    def constrain(self, kind: str, *refs: str, id: str | None = None, **options: Any) -> str:  # noqa: A002
        """Add a geometric constraint (``horizontal``, ``coincident``, ``tangent``...)."""
        defn = self._defn()
        cid = id or self._new_id(defn, "k")
        c = {"id": cid, "type": kind, "refs": list(refs), **options}
        return self._add_constraint(defn, c, f"Constrain {kind}")

    def dimension(
        self,
        kind: str,
        *refs: str,
        value: Any,
        id: str | None = None,
        driving: bool = True,
        **options,
    ) -> str:  # noqa: A002
        """Add a dimension (``distance``, ``length``, ``angle``, ``radius``, ``diameter``);
        ``value`` is a number (document unit / degrees), quantity text or ``"=expr"``."""
        defn = self._defn()
        cid = id or self._new_id(defn, "d")
        c = {"id": cid, "type": kind, "refs": list(refs), "value": value, **options}
        if not driving:
            c["driving"] = False
        return self._add_constraint(defn, c, f"Dimension {kind}")

    def _add_constraint(self, defn, c, label) -> str:
        defn.setdefault("constraints", []).append(c)
        try:
            index = validate(defn, "sketch")
        except CadError:
            raise
        pts = {e["id"]: np.array(e["at"]) for e in defn["entities"] if e["type"] == "point"}
        radii = {e["id"]: e["radius"] for e in defn["entities"] if "radius" in e}
        choices = infer_choices(index, pts, radii, c)
        for k, v in choices.items():
            c.setdefault(k, v)
        if c["type"] == "fix" and "at" not in c:
            c["at"] = [list(map(float, pts[p])) for p in index.points_of(c["refs"][0])]
        self._commit(defn, label)
        return c["id"]

    def set_value(self, constraint_id: str, value: Any) -> None:
        """Change a dimension's value (the sketch and everything downstream regenerate)."""
        defn = self._defn()
        for c in defn.get("constraints", []):
            if c["id"] == constraint_id:
                if "value" not in c:
                    raise CadError("NOT_A_DIMENSION", f"{constraint_id!r} has no value")
                c["value"] = value
                self._commit(defn, f"Set {constraint_id}")
                return
        raise CadError("UNKNOWN_CONSTRAINT", f"the sketch has no constraint {constraint_id!r}")

    def remove(self, *ids: str) -> None:
        """Remove entities (and the constraints and entities that use them) or constraints."""
        defn = self._defn()
        doomed = set(ids)
        changed = True
        while changed:
            changed = False
            for e in defn["entities"]:
                if e["id"] in doomed:
                    continue
                uses = [e.get(k) for k in ("start", "end", "center")] + list(e.get("points", []))
                if any(u in doomed for u in uses if u):
                    doomed.add(e["id"])
                    changed = True
        defn["entities"] = [e for e in defn["entities"] if e["id"] not in doomed]
        defn["constraints"] = [
            c
            for c in defn.get("constraints", [])
            if c["id"] not in doomed and not (set(c["refs"]) & doomed)
        ]
        self._commit(defn, "Remove sketch elements")

    def solve(self) -> dict[str, Any]:
        return KINDS["sketch"].info(self.doc, self.item, self.doc.result(self.item))

    @property
    def result(self) -> SketchResult:
        return self.doc.result(self.item)

    def point_at(self, pid: str) -> tuple[float, float]:
        """A point's solved position in document units."""
        u = float(unit(self.doc.length_unit).factor)
        x, y = self.result.point2(pid)
        return x / u, y / u

    def measure(self, cid: str) -> float:
        """A dimension's measured value (document units or degrees)."""
        v = self.result.solution.reference[cid]
        c = next(c for c in self.item._definition["constraints"] if c["id"] == cid)
        if c["type"] == "angle":
            return math.degrees(v)
        return v / float(unit(self.doc.length_unit).factor)


class PartHandle:
    """Build and change a part's feature list. Each method appends one feature (one
    undoable step) and returns its id; ``set`` changes a feature's settings."""

    def __init__(self, doc: CadDocument, item: Item) -> None:
        self.doc = doc
        self.item = item

    @property
    def name(self) -> str:
        return self.item.name

    def _new_id(self, base: str) -> str:
        used = {f["id"] for f in self.item._definition.get("features", [])}
        k = 1
        while f"{base}{k}" in used:
            k += 1
        return f"{base}{k}"

    def add_feature(self, feature: dict) -> str:
        defn = copy.deepcopy(self.item._definition)
        f = dict(feature)
        f.setdefault("id", self._new_id(f.get("type", "f")))
        defn.setdefault("features", []).append(f)
        self.doc.set_definition(
            self.item, defn, label=f"Add {f.get('type')} {f['id']!r} to {self.item.name!r}"
        )
        return f["id"]

    def set(self, feature_id: str, **changes: Any) -> None:
        defn = copy.deepcopy(self.item._definition)
        for f in defn.get("features", []):
            if f["id"] == feature_id:
                for k, v in changes.items():
                    if v is None:
                        f.pop(k, None)
                    else:
                        f[k] = v
                self.doc.set_definition(
                    self.item, defn, label=f"Change {feature_id!r} of {self.item.name!r}"
                )
                return
        raise CadError("UNKNOWN_FEATURE", f"the part has no feature {feature_id!r}")

    def remove_feature(self, feature_id: str) -> None:
        defn = copy.deepcopy(self.item._definition)
        before = len(defn.get("features", []))
        defn["features"] = [f for f in defn.get("features", []) if f["id"] != feature_id]
        if len(defn["features"]) == before:
            raise CadError("UNKNOWN_FEATURE", f"the part has no feature {feature_id!r}")
        self.doc.set_definition(self.item, defn, label=f"Remove {feature_id!r}")

    def extrude(
        self,
        sketch,
        distance=None,
        *,
        operation=None,
        extent=None,
        draft=None,
        regions=None,
        body="main",
        reverse=False,
        id=None,
        group=0,
    ):  # noqa: A002
        f = {"type": "extrude", "sketch": _name(sketch), "body": body}
        f["extent"] = extent or {"type": "blind", "distance": distance}
        for k, v in (("operation", operation), ("draft", draft), ("regions", regions), ("id", id)):
            if v is not None:
                f[k] = v
        if reverse:
            f["reverse"] = True
        if group:
            f["group"] = group
        return self.add_feature(f)

    def revolve(self, sketch, axis, angle="360 deg", *, operation=None, body="main", id=None):  # noqa: A002
        f = {"type": "revolve", "sketch": _name(sketch), "axis": axis, "angle": angle, "body": body}
        if operation:
            f["operation"] = operation
        if id:
            f["id"] = id
        return self.add_feature(f)

    def hole(self, face, points, diameter, *, depth="through", kind="simple", id=None, **options):  # noqa: A002
        f = {
            "type": "hole",
            "face": face,
            "points": points,
            "diameter": diameter,
            "depth": depth,
            "kind": kind,
            **options,
        }
        if id:
            f["id"] = id
        return self.add_feature(f)

    def fillet(self, edges, radius, *, id=None):  # noqa: A002
        return self.add_feature(
            {"type": "fillet", "edges": list(edges), "radius": radius, **({"id": id} if id else {})}
        )

    def chamfer(self, edges, distance, *, id=None, distance2=None):  # noqa: A002
        f = {"type": "chamfer", "edges": list(edges), "distance": distance}
        if distance2 is not None:
            f["distance2"] = distance2
        if id:
            f["id"] = id
        return self.add_feature(f)

    def shell(self, feature, thickness, *, open=("end",), id=None):  # noqa: A002
        return self.add_feature(
            {
                "type": "shell",
                "feature": feature,
                "thickness": thickness,
                "open": list(open),
                **({"id": id} if id else {}),
            }
        )

    def pattern(self, features, **options):
        return self.add_feature({"type": "pattern", "features": list(features), **options})

    def mirror(self, plane, *, features=None, body=None, id=None):  # noqa: A002
        f = {"type": "mirror", "plane": plane}
        if features is not None:
            f["features"] = list(features)
        if body is not None:
            f["body"] = body
        if id:
            f["id"] = id
        return self.add_feature(f)

    @property
    def result(self) -> feat.PartResult:
        return self.doc.result(self.item)

    def body(self, name: str = "main"):
        return self.result.body(name)

    def faces(self, body: str = "main") -> list[dict]:
        from .naming import describe_faces

        return describe_faces(self.body(body), _unit_factor(self.doc))

    def measure(self, body: str = "main") -> dict[str, Any]:
        return KINDS["part"].info(self.doc, self.item, self.result)["bodies"][body]


def _name(ref) -> str:
    if isinstance(ref, (SketchHandle, PartHandle)):
        return ref.item.name
    if isinstance(ref, Item):
        return ref.name
    return ref


# --------------------------------------------------------------------------- #
# Document conveniences
# --------------------------------------------------------------------------- #


def _install() -> None:
    def parameter(self: CadDocument, name: str, value: Any, *, dimension: str = "length") -> Item:
        """Create or change a named parameter (a quantity or ``"=expression"``)."""
        defn = {"value": value, "dimension": dimension}
        if name in self._by_name:
            item = self.item(name, "parameter")
            self.set_definition(item, defn, label=f"Set parameter {name!r}")
            return item
        return self.add("parameter", name, defn)

    def frame(self, name: str, **spec: Any) -> Item:
        return self.add("frame", name, spec)

    def sketch(self, name: str | None = None, plane: Any = "XY") -> SketchHandle:
        if name is not None and name in self._by_name:
            return SketchHandle(self, self.item(name, "sketch"))
        item = self.add(
            "sketch",
            name or self._unique_name("Sketch"),
            {"plane": plane, "entities": [], "constraints": []},
        )
        return SketchHandle(self, item)

    def part(self, name: str | None = None, **options: Any) -> PartHandle:
        if name is not None and name in self._by_name:
            return PartHandle(self, self.item(name, "part"))
        item = self.add("part", name or self._unique_name("Part"), {"features": [], **options})
        return PartHandle(self, item)

    CadDocument.parameter = parameter
    CadDocument.frame = frame
    CadDocument.sketch = sketch
    CadDocument.part = part


_install()
