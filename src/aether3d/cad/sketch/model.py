"""Sketch definitions: entities, constraints, validation and compilation to equations.

A sketch definition is plain data (JSON values), owned by the CAD document and never
mutated in place::

    {
      "frame": {"origin": [0, 0, 0], "z": [0, 0, 1], "x": [1, 0, 0]},
      "entities": [
        {"id": "p1", "type": "point", "at": [0.0, 0.0]},          # metres, sketch frame
        {"id": "l1", "type": "line", "start": "p1", "end": "p2"},
        {"id": "c1", "type": "circle", "center": "p3", "radius": 0.005},
        {"id": "a1", "type": "arc", "center": "p4", "start": "p5", "end": "p6",
         "radius": 0.01},                                          # counter-clockwise
        {"id": "s1", "type": "spline", "degree": 3, "points": ["p7", "p8", "p9", "p10"]}
      ],
      "constraints": [
        {"id": "k1", "type": "horizontal", "refs": ["l1"]},
        {"id": "d1", "type": "distance", "refs": ["p1", "p2"], "value": "80 mm"},
        {"id": "d2", "type": "radius", "refs": ["c1"], "value": "=hole / 2",
         "driving": false}
      ]
    }

Any entity may be marked ``"construction": true`` (it constrains but is not part of
profiles). Point coordinates and radii are the current solution: solving starts from them
and the document stores what the solver returns. Values are quantity text (``"80 mm"``),
numbers in the document's length unit, or ``"=expression"`` over document parameters;
they are evaluated before compilation, so the solver sees SI numbers only.

Constraint types and the equations they compile to (kernels in ``equations.py``):

=================  ====================================  ===================================
type               refs                                  equations
=================  ====================================  ===================================
coincident         point, point | point, curve           2 linear | point on line/circle
horizontal         line | point, point                   y equal
vertical           line | point, point                   x equal
parallel           line, line                            sin(angle) = 0
perpendicular      line, line                            cos(angle) = 0
collinear          line, line                            both points of 2 on line 1
tangent            line|circle|arc, circle|arc           center distance = (+/-) radius;
                                                         at a shared endpoint: radius
                                                         perpendicular to the line
concentric         circle|arc, circle|arc                centers coincide
equal              line, line | circle|arc, circle|arc   equal length | equal radius
midpoint           point, line                           point = midpoint
point_on           point, curve                          on line | on circle/arc
symmetric          point, point, line|point              mirror images
fix                entity                                its points (and radius) fixed
distance           point, point | point, line            distance (``along``: x, y)
length             line                                  length
angle              line, line | line                     signed angle (or to the x axis)
radius, diameter   circle|arc                            radius
=================  ====================================  ===================================

Sign choices (which side of a line a distance is on, internal or external tangency, the
orientation of an angle) are recorded in the constraint (``"side"``) when it is added,
from the geometry at that moment, so a solve never jumps to the mirrored solution of a
single constraint.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...errors import CadError
from ..tolerance import check_coordinate

MAX_ENTITIES = 20_000
MAX_CONSTRAINTS = 20_000
MAX_SPLINE_POINTS = 64
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")

ENTITY_TYPES = ("point", "line", "circle", "arc", "spline")
CURVE_TYPES = ("line", "circle", "arc", "spline")
VALUE_TYPES = ("distance", "length", "angle", "radius", "diameter")
CONSTRAINT_TYPES = (
    "coincident",
    "horizontal",
    "vertical",
    "parallel",
    "perpendicular",
    "collinear",
    "tangent",
    "concentric",
    "equal",
    "midpoint",
    "point_on",
    "symmetric",
    "fix",
    *VALUE_TYPES,
)


def check_id(value: Any, location: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise CadError(
            "INVALID_ID",
            f"{value!r} is not a valid id (letters, digits, _ . -; at most 64)",
            location,
        )
    return value


@dataclass
class Index:
    """Entities of a definition by id, with their types (after validation)."""

    entities: dict[str, dict]
    order: list[str]
    constraints: list[dict]

    def kind(self, ref: str) -> str:
        return self.entities[ref]["type"]

    def points_of(self, ref: str) -> list[str]:
        e = self.entities[ref]
        t = e["type"]
        if t == "point":
            return [ref]
        if t == "line":
            return [e["start"], e["end"]]
        if t == "circle":
            return [e["center"]]
        if t == "arc":
            return [e["center"], e["start"], e["end"]]
        return list(e["points"])

    def endpoints(self, ref: str) -> list[str]:
        e = self.entities[ref]
        if e["type"] in ("line", "arc"):
            return [e["start"], e["end"]]
        if e["type"] == "spline":
            return [e["points"][0], e["points"][-1]]
        return []


def validate(defn: Any, location: str = "sketch") -> Index:
    """Check a definition completely (types, references, limits); raise ``CadError``."""
    if not isinstance(defn, Mapping):
        raise CadError("INVALID_SKETCH", "a sketch definition must be an object", location)
    entities = defn.get("entities", [])
    constraints = defn.get("constraints", [])
    if not isinstance(entities, list) or not isinstance(constraints, list):
        raise CadError("INVALID_SKETCH", "entities and constraints must be lists", location)
    if len(entities) > MAX_ENTITIES:
        raise CadError("LIMIT_EXCEEDED", f"more than {MAX_ENTITIES} sketch entities", location)
    if len(constraints) > MAX_CONSTRAINTS:
        raise CadError(
            "LIMIT_EXCEEDED", f"more than {MAX_CONSTRAINTS} sketch constraints", location
        )
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for i, e in enumerate(entities):
        loc = f"{location}.entities[{i}]"
        if not isinstance(e, Mapping):
            raise CadError("INVALID_SKETCH", "an entity must be an object", loc)
        eid = check_id(e.get("id"), f"{loc}.id")
        if eid in by_id:
            raise CadError("DUPLICATE_ID", f"entity id {eid!r} is used twice", loc)
        t = e.get("type")
        if t not in ENTITY_TYPES:
            raise CadError(
                "INVALID_SKETCH",
                f"unknown entity type {t!r}",
                loc,
                hint="types: " + ", ".join(ENTITY_TYPES),
            )
        if t == "point":
            at = e.get("at")
            if not isinstance(at, (list, tuple)) or len(at) != 2:
                raise CadError("INVALID_SKETCH", "a point needs 'at': [x, y]", loc)
            for k, c in enumerate(at):
                check_coordinate(c, f"{loc}.at[{k}]")
        elif t in ("circle", "arc"):
            r = e.get("radius")
            check_coordinate(r, f"{loc}.radius")
            if not r > 0:
                raise CadError("DEGENERATE_GEOMETRY", "a radius must be positive", loc)
        if t == "spline":
            pts = e.get("points")
            deg = e.get("degree", 3)
            if (
                not isinstance(pts, list)
                or not isinstance(deg, int)
                or isinstance(deg, bool)
                or not 1 <= deg <= 5
                or not deg + 1 <= len(pts) <= MAX_SPLINE_POINTS
            ):
                raise CadError(
                    "INVALID_SKETCH",
                    f"a spline needs degree 1..5 and degree+1..{MAX_SPLINE_POINTS} points",
                    loc,
                )
        by_id[eid] = dict(e)
        order.append(eid)
    for i, e in enumerate(entities):
        loc = f"{location}.entities[{i}]"
        refs = {
            "line": ("start", "end"),
            "circle": ("center",),
            "arc": ("center", "start", "end"),
        }.get(e["type"], ())
        for key in refs:
            _need_point(by_id, e.get(key), f"{loc}.{key}")
        if e["type"] == "arc" and len({e["center"], e["start"], e["end"]}) < 3:
            raise CadError("DEGENERATE_GEOMETRY", "an arc needs distinct points", loc)
        if e["type"] == "line" and e["start"] == e["end"]:
            raise CadError("DEGENERATE_GEOMETRY", "a line needs two distinct points", loc)
        if e["type"] == "spline":
            for k, p in enumerate(e["points"]):
                _need_point(by_id, p, f"{loc}.points[{k}]")
    seen: set[str] = set()
    for i, c in enumerate(constraints):
        loc = f"{location}.constraints[{i}]"
        if not isinstance(c, Mapping):
            raise CadError("INVALID_SKETCH", "a constraint must be an object", loc)
        cid = check_id(c.get("id"), f"{loc}.id")
        if cid in seen or cid in by_id:
            raise CadError("DUPLICATE_ID", f"id {cid!r} is used twice", loc)
        seen.add(cid)
        t = c.get("type")
        if t not in CONSTRAINT_TYPES:
            raise CadError(
                "INVALID_CONSTRAINT",
                f"unknown constraint type {t!r}",
                loc,
                hint="types: " + ", ".join(CONSTRAINT_TYPES),
            )
        refs = c.get("refs")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 3:
            raise CadError("INVALID_CONSTRAINT", "refs must list 1..3 entity ids", loc)
        for k, r in enumerate(refs):
            if r not in by_id:
                raise CadError(
                    "UNKNOWN_ENTITY",
                    f"constraint references unknown entity {r!r}",
                    f"{loc}.refs[{k}]",
                )
        if t in VALUE_TYPES and "value" not in c:
            raise CadError("INVALID_CONSTRAINT", f"a {t} constraint needs a value", loc)
        _check_shape(t, refs, by_id, c, loc)
    return Index(by_id, order, list(constraints))


def _need_point(by_id: dict, ref: Any, loc: str) -> None:
    if not isinstance(ref, str) or ref not in by_id:
        raise CadError("UNKNOWN_ENTITY", f"unknown point {ref!r}", loc)
    if by_id[ref]["type"] != "point":
        raise CadError("INVALID_SKETCH", f"{ref!r} is not a point", loc)


def _check_shape(t: str, refs: list[str], by_id: dict, c: Mapping, loc: str) -> None:
    kinds = [by_id[r]["type"] for r in refs]

    def bad(expected: str) -> CadError:
        return CadError(
            "INVALID_CONSTRAINT",
            f"{t} takes {expected}; got {', '.join(kinds)}",
            loc,
        )

    circ = ("circle", "arc")
    if t == "coincident":
        if len(refs) != 2 or "point" not in kinds:
            raise bad("a point and a point or curve")
        if kinds[0] != "point" and kinds[1] != "point":
            raise bad("a point and a point or curve")
        other = kinds[1] if kinds[0] == "point" else kinds[0]
        if other == "spline":
            raise bad("a point and a point, line, circle or arc")
    elif t in ("horizontal", "vertical"):
        if not (kinds == ["line"] or kinds == ["point", "point"]):
            raise bad("a line or two points")
    elif t in ("parallel", "perpendicular", "collinear"):
        if kinds != ["line", "line"]:
            raise bad("two lines")
    elif t == "tangent":
        if (
            len(refs) != 2
            or not all(k in ("line", *circ) for k in kinds)
            or kinds
            == [
                "line",
                "line",
            ]
        ):
            raise bad("a line, circle or arc and a circle or arc")
    elif t == "concentric":
        if len(refs) != 2 or not all(k in circ for k in kinds):
            raise bad("two circles or arcs")
    elif t == "equal":
        if not (kinds == ["line", "line"] or (len(kinds) == 2 and all(k in circ for k in kinds))):
            raise bad("two lines or two circles/arcs")
    elif t == "midpoint":
        if sorted(kinds) != ["line", "point"]:
            raise bad("a point and a line")
    elif t == "point_on":
        if len(refs) != 2 or kinds[0] != "point" or kinds[1] not in ("line", *circ):
            raise bad("a point and a line, circle or arc")
    elif t == "symmetric":
        if len(refs) != 3 or kinds[:2] != ["point", "point"] or kinds[2] not in ("line", "point"):
            raise bad("two points and a line or point")
    elif t == "fix":
        if len(refs) != 1:
            raise bad("one entity")
    elif t == "distance":
        if not (kinds == ["point", "point"] or sorted(kinds) == ["line", "point"]):
            raise bad("two points, or a point and a line")
        along = c.get("along")
        if along not in (None, "x", "y"):
            raise CadError("INVALID_CONSTRAINT", "along must be 'x' or 'y'", loc)
        if along is not None and kinds != ["point", "point"]:
            raise bad("two points (with along)")
    elif t == "length":
        if kinds != ["line"]:
            raise bad("a line")
    elif t == "angle":
        if kinds not in (["line", "line"], ["line"]):
            raise bad("one or two lines")
    elif t in ("radius", "diameter"):
        if len(kinds) != 1 or kinds[0] not in circ:
            raise bad("a circle or arc")


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #

KERNEL_WIDTH = {
    "linear": None,  # per equation, padded
    "dist": 4,
    "dist_var": 6,
    "dist_diff": 8,
    "point_line": 7,
    "point_line_mid": 8,
    "perp_dot": 8,
    "lines_angle": 8,
    "line_angle": 4,
}


@dataclass
class Problem:
    """Compiled equations: variables, kernels with index and constant matrices."""

    x0: np.ndarray
    var_names: list[str]
    point_vars: dict[str, tuple[int, int]]
    radius_vars: dict[str, int]
    kernels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    eq_constraint: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    eq_kernel: list[tuple[str, int]] = field(default_factory=list)
    adj: list[list[int]] = field(default_factory=list)
    scale: float = 1.0

    @property
    def n(self) -> int:
        return len(self.x0)

    @property
    def m(self) -> int:
        return len(self.eq_kernel)


class _Builder:
    def __init__(self, index: Index, values: Mapping[str, float], scale: float) -> None:
        self.ix = index
        self.values = values
        self.scale = scale
        x: list[float] = []
        names: list[str] = []
        self.pv: dict[str, tuple[int, int]] = {}
        self.rv: dict[str, int] = {}
        for eid in index.order:
            e = index.entities[eid]
            if e["type"] == "point":
                self.pv[eid] = (len(x), len(x) + 1)
                x += [float(e["at"][0]), float(e["at"][1])]
                names += [f"{eid}.x", f"{eid}.y"]
            elif e["type"] in ("circle", "arc"):
                self.rv[eid] = len(x)
                x.append(float(e["radius"]))
                names.append(f"{eid}.radius")
        self.x = np.array(x, dtype=np.float64)
        self.names = names
        self.rows: dict[str, tuple[list[list[int]], list[list[float]]]] = {}
        self.eq_constraint: list[int] = []
        self.eq_kernel: list[tuple[str, int]] = []
        self.adj: list[list[int]] = []
        self.linear_width = 0

    def P(self, ref: str) -> list[int]:  # noqa: N802 - variables of a point
        return list(self.pv[ref])

    def at(self, ref: str) -> np.ndarray:
        i, j = self.pv[ref]
        return self.x[[i, j]]

    def add(self, kernel: str, cols: list[int], consts: list[float], constraint: int) -> None:
        idx, cs = self.rows.setdefault(kernel, ([], []))
        self.eq_kernel.append((kernel, len(idx)))
        idx.append(cols)
        cs.append([float(c) for c in consts])
        self.eq_constraint.append(constraint)
        self.adj.append(sorted(set(cols)))
        if kernel == "linear":
            self.linear_width = max(self.linear_width, len(cols))

    def lin(self, terms: list[tuple[int, float]], c0: float, k: int) -> None:
        cols = [t[0] for t in terms]
        self.add("linear", cols, [c0, *(t[1] for t in terms)], k)

    def build(self) -> Problem:
        kernels = {}
        for name, (idx, cs) in self.rows.items():
            if name == "linear":
                w = self.linear_width
                idx = [row + [row[0]] * (w - len(row)) for row in idx]
                cs = [row + [0.0] * (w + 1 - len(row)) for row in cs]
            kernels[name] = (
                np.array(idx, dtype=np.int64),
                np.array(cs, dtype=np.float64),
                np.array([i for i, (kn, _) in enumerate(self.eq_kernel) if kn == name]),
            )
        return Problem(
            x0=self.x,
            var_names=self.names,
            point_vars=self.pv,
            radius_vars=self.rv,
            kernels=kernels,
            eq_constraint=np.array(self.eq_constraint, dtype=np.int64),
            eq_kernel=self.eq_kernel,
            adj=self.adj,
            scale=self.scale,
        )


def characteristic_length(index: Index) -> float:
    pts = [e["at"] for e in index.entities.values() if e["type"] == "point"]
    if not pts:
        return 1.0
    a = np.asarray(pts, dtype=np.float64)
    diag = float(np.linalg.norm(a.max(axis=0) - a.min(axis=0)))
    radii = [float(e["radius"]) for e in index.entities.values() if "radius" in e]
    return max(diag, 2 * max(radii, default=0.0), 1e-3)


def compile_problem(index: Index, values: Mapping[str, float]) -> Problem:
    """Equations of every constraint (and the arcs' built-in equal radii).

    ``values`` maps constraint ids to their evaluated SI values (driving dimensions only).
    """
    scale = characteristic_length(index)
    b = _Builder(index, values, scale)
    # Arcs: both ends on the circle of the arc's radius (row tagged to no constraint: -1).
    for eid in index.order:
        e = index.entities[eid]
        if e["type"] == "arc":
            r = b.rv[eid]
            for end in (e["start"], e["end"]):
                b.add("dist_var", b.P(end) + b.P(e["center"]) + [r, r], [0.0, 1.0, 0.0], -1)
    for k, c in enumerate(index.constraints):
        if c.get("driving", True) is False:
            continue
        _compile_constraint(b, k, c, values)
    return b.build()


def _side(c: Mapping, default: float = 1.0) -> float:
    s = c.get("side", default)
    return -1.0 if s in (-1, -1.0) else 1.0


def _compile_constraint(b: _Builder, k: int, c: Mapping, values: Mapping[str, float]) -> None:
    ix = b.ix
    t = c["type"]
    refs = c["refs"]
    kinds = [ix.kind(r) for r in refs]
    value = values.get(c["id"])
    L = b.scale  # noqa: N806
    if t == "coincident":
        p, other = (refs[0], refs[1]) if kinds[0] == "point" else (refs[1], refs[0])
        ok = ix.kind(other)
        if ok == "point":
            (px, py), (qx, qy) = b.pv[p], b.pv[other]
            b.lin([(px, 1.0), (qx, -1.0)], 0.0, k)
            b.lin([(py, 1.0), (qy, -1.0)], 0.0, k)
        else:
            _point_on(b, k, p, other)
    elif t in ("horizontal", "vertical"):
        a, e = (
            (ix.entities[refs[0]]["start"], ix.entities[refs[0]]["end"])
            if kinds == ["line"]
            else (refs[0], refs[1])
        )
        axis = 1 if t == "horizontal" else 0
        b.lin([(b.pv[a][axis], 1.0), (b.pv[e][axis], -1.0)], 0.0, k)
    elif t in ("parallel", "perpendicular"):
        theta = 0.0 if t == "parallel" else math.pi / 2
        l1, l2 = ix.entities[refs[0]], ix.entities[refs[1]]
        cols = b.P(l1["start"]) + b.P(l1["end"]) + b.P(l2["start"]) + b.P(l2["end"])
        b.add("lines_angle", cols, [L, math.cos(theta), math.sin(theta)], k)
    elif t == "collinear":
        l1, l2 = ix.entities[refs[0]], ix.entities[refs[1]]
        for p in (l2["start"], l2["end"]):
            b.add("point_line", b.P(l1["start"]) + b.P(l1["end"]) + b.P(p) + [b.P(p)[0]], [0, 0], k)
    elif t == "tangent":
        _tangent(b, k, c, refs, kinds)
    elif t == "concentric":
        c1, c2 = ix.entities[refs[0]]["center"], ix.entities[refs[1]]["center"]
        (px, py), (qx, qy) = b.pv[c1], b.pv[c2]
        b.lin([(px, 1.0), (qx, -1.0)], 0.0, k)
        b.lin([(py, 1.0), (qy, -1.0)], 0.0, k)
    elif t == "equal":
        if kinds == ["line", "line"]:
            l1, l2 = ix.entities[refs[0]], ix.entities[refs[1]]
            cols = b.P(l1["start"]) + b.P(l1["end"]) + b.P(l2["start"]) + b.P(l2["end"])
            b.add("dist_diff", cols, [0.0], k)
        else:
            b.lin([(b.rv[refs[0]], 1.0), (b.rv[refs[1]], -1.0)], 0.0, k)
    elif t == "midpoint":
        p, line = (refs[0], refs[1]) if kinds[0] == "point" else (refs[1], refs[0])
        e = ix.entities[line]
        for axis in (0, 1):
            b.lin(
                [
                    (b.pv[p][axis], 1.0),
                    (b.pv[e["start"]][axis], -0.5),
                    (b.pv[e["end"]][axis], -0.5),
                ],
                0.0,
                k,
            )
    elif t == "point_on":
        _point_on(b, k, refs[0], refs[1])
    elif t == "symmetric":
        p, q, about = refs
        if ix.kind(about) == "point":
            for axis in (0, 1):
                b.lin(
                    [(b.pv[p][axis], 0.5), (b.pv[q][axis], 0.5), (b.pv[about][axis], -1.0)], 0.0, k
                )
        else:
            e = ix.entities[about]
            line = b.P(e["start"]) + b.P(e["end"])
            b.add("point_line_mid", line + b.P(p) + b.P(q), [0.0], k)
            b.add("perp_dot", line + b.P(p) + b.P(q), [0.0], k)
    elif t == "fix":
        target = refs[0]
        pts = ix.points_of(target)
        at = c.get("at")
        for i, p in enumerate(pts):
            pos = at[i] if isinstance(at, list) and len(at) == len(pts) else ix.entities[p]["at"]
            for axis in (0, 1):
                b.lin([(b.pv[p][axis], 1.0)], float(pos[axis]), k)
        if ix.kind(target) in ("circle", "arc"):
            r = c.get("radius", ix.entities[target]["radius"])
            b.lin([(b.rv[target], 1.0)], float(r), k)
    elif t == "distance":
        _distance(b, k, c, refs, kinds, float(value))
    elif t == "length":
        e = ix.entities[refs[0]]
        b.add("dist", b.P(e["start"]) + b.P(e["end"]), [float(value)], k)
    elif t == "angle":
        theta = float(value) * _side(c)
        if kinds == ["line"]:
            e = ix.entities[refs[0]]
            b.add(
                "line_angle",
                b.P(e["start"]) + b.P(e["end"]),
                [L, math.cos(theta), math.sin(theta)],
                k,
            )
        else:
            l1, l2 = ix.entities[refs[0]], ix.entities[refs[1]]
            cols = b.P(l1["start"]) + b.P(l1["end"]) + b.P(l2["start"]) + b.P(l2["end"])
            b.add("lines_angle", cols, [L, math.cos(theta), math.sin(theta)], k)
    elif t in ("radius", "diameter"):
        r = float(value) / (2.0 if t == "diameter" else 1.0)
        b.lin([(b.rv[refs[0]], 1.0)], r, k)


def _point_on(b: _Builder, k: int, p: str, curve: str) -> None:
    e = b.ix.entities[curve]
    if e["type"] == "line":
        cols = b.P(e["start"]) + b.P(e["end"]) + b.P(p)
        b.add("point_line", cols + [cols[0]], [0.0, 0.0], k)
    else:
        r = b.rv[curve]
        b.add("dist_var", b.P(p) + b.P(e["center"]) + [r, r], [0.0, 1.0, 0.0], k)


def _tangent(b: _Builder, k: int, c: Mapping, refs: list[str], kinds: list[str]) -> None:
    ix = b.ix
    e1, e2 = ix.entities[refs[0]], ix.entities[refs[1]]
    shared = set(ix.endpoints(refs[0])) & set(ix.endpoints(refs[1]))
    if shared:
        # Tangent where the curves meet: the radius at the shared point is
        # perpendicular to the line, or both centers are collinear with it.
        s = sorted(shared)[0]
        if kinds[0] == "line" or kinds[1] == "line":
            line, circ = (e1, e2) if kinds[0] == "line" else (e2, e1)
            cols = b.P(line["start"]) + b.P(line["end"]) + b.P(circ["center"]) + b.P(s)
            b.add("perp_dot", cols, [0.0], k)
        else:
            cols = b.P(e1["center"]) + b.P(s) + b.P(e2["center"])
            b.add("point_line", cols + [cols[0]], [0.0, 0.0], k)
        return
    side = _side(c)
    if kinds[0] == "line" or kinds[1] == "line":
        line, circ, cref = (e1, e2, refs[1]) if kinds[0] == "line" else (e2, e1, refs[0])
        cols = b.P(line["start"]) + b.P(line["end"]) + b.P(circ["center"]) + [b.rv[cref]]
        b.add("point_line", cols, [0.0, side], k)
    else:
        # External (side +1): |c1 - c2| = r1 + r2; internal (-1): |c1 - c2| = |r1 - r2|,
        # with the larger circle recorded in "inner" order: r_big - r_small.
        r1, r2 = b.rv[refs[0]], b.rv[refs[1]]
        cols = b.P(e1["center"]) + b.P(e2["center"]) + [r1, r2]
        if side > 0:
            b.add("dist_var", cols, [0.0, 1.0, 1.0], k)
        else:
            sign = 1.0 if c.get("larger", 0) == 0 else -1.0
            b.add("dist_var", cols, [0.0, sign, -sign], k)


def _distance(
    b: _Builder, k: int, c: Mapping, refs: list[str], kinds: list[str], value: float
) -> None:
    ix = b.ix
    if kinds == ["point", "point"]:
        p, q = refs
        along = c.get("along")
        if along is None:
            b.add("dist", b.P(p) + b.P(q), [value], k)
        else:
            axis = 0 if along == "x" else 1
            b.lin([(b.pv[q][axis], 1.0), (b.pv[p][axis], -1.0)], _side(c) * value, k)
        return
    p, line = (refs[0], refs[1]) if kinds[0] == "point" else (refs[1], refs[0])
    e = ix.entities[line]
    cols = b.P(e["start"]) + b.P(e["end"]) + b.P(p)
    b.add("point_line", cols + [cols[0]], [_side(c) * value, 0.0], k)


# --------------------------------------------------------------------------- #
# Measurements (reference dimensions, sides at creation)
# --------------------------------------------------------------------------- #


def measure(index: Index, points: Mapping[str, np.ndarray], radii: Mapping[str, float], c: Mapping):
    """The current value of a dimensional constraint (SI), whatever its ``driving`` flag."""
    t = c["type"]
    refs = c["refs"]
    kinds = [index.kind(r) for r in refs]
    if t == "distance":
        if kinds == ["point", "point"]:
            p, q = points[refs[0]], points[refs[1]]
            along = c.get("along")
            if along is None:
                return float(np.hypot(*(q - p)))
            axis = 0 if along == "x" else 1
            return float(abs(q[axis] - p[axis]))
        p, line = (refs[0], refs[1]) if kinds[0] == "point" else (refs[1], refs[0])
        e = index.entities[line]
        return abs(_signed_distance(points[p], points[e["start"]], points[e["end"]]))
    if t == "length":
        e = index.entities[refs[0]]
        return float(np.hypot(*(points[e["end"]] - points[e["start"]])))
    if t in ("radius", "diameter"):
        r = float(radii[refs[0]])
        return 2 * r if t == "diameter" else r
    if t == "angle":
        return abs(_signed_angle(index, points, refs))
    return None


def _signed_distance(p, a, b) -> float:
    d = b - a
    n = float(np.hypot(*d))
    if n == 0:
        return 0.0
    return float((d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / n)


def _signed_angle(index: Index, points, refs) -> float:
    def direction(ref):
        e = index.entities[ref]
        return points[e["end"]] - points[e["start"]]

    d1 = direction(refs[0])
    d2 = direction(refs[1]) if len(refs) > 1 else np.array([1.0, 0.0])
    if len(refs) == 1:
        d1, d2 = np.array([1.0, 0.0]), d1
    return math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)


def infer_choices(index: Index, points, radii, c: Mapping) -> dict[str, int]:
    """The branch choices a new constraint records from the current geometry: ``side``
    (and ``larger`` for internal tangency), so solving never jumps to a mirror image."""
    out: dict[str, int] = {}
    side = _infer_side(index, points, radii, c)
    if side is not None:
        out["side"] = side
    if c["type"] == "tangent" and side == -1 and "line" not in [index.kind(r) for r in c["refs"]]:
        out["larger"] = 0 if float(radii[c["refs"][0]]) >= float(radii[c["refs"][1]]) else 1
    return out


def _infer_side(index: Index, points, radii, c: Mapping) -> int | None:
    t = c["type"]
    refs = c["refs"]
    kinds = [index.kind(r) for r in refs]
    if t == "distance":
        if kinds == ["point", "point"] and c.get("along"):
            axis = 0 if c["along"] == "x" else 1
            return 1 if points[refs[1]][axis] >= points[refs[0]][axis] else -1
        if sorted(kinds) == ["line", "point"]:
            p, line = (refs[0], refs[1]) if kinds[0] == "point" else (refs[1], refs[0])
            e = index.entities[line]
            return (
                1 if _signed_distance(points[p], points[e["start"]], points[e["end"]]) >= 0 else -1
            )
        return None
    if t == "angle":
        return 1 if _signed_angle(index, points, refs) >= 0 else -1
    if t == "tangent":
        shared = set(index.endpoints(refs[0])) & set(index.endpoints(refs[1]))
        if shared:
            return None
        if "line" in kinds:
            line, circ = (refs[0], refs[1]) if kinds[0] == "line" else (refs[1], refs[0])
            e = index.entities[line]
            center = points[index.entities[circ]["center"]]
            return 1 if _signed_distance(center, points[e["start"]], points[e["end"]]) >= 0 else -1
        c1 = points[index.entities[refs[0]]["center"]]
        c2 = points[index.entities[refs[1]]["center"]]
        r1, r2 = float(radii[refs[0]]), float(radii[refs[1]])
        d = float(np.hypot(*(c1 - c2)))
        return 1 if abs(d - (r1 + r2)) <= abs(d - abs(r1 - r2)) else -1
    return None
