"""STEP (ISO 10303-21) export and import of exact B-reps, AP214 schema.

**Export** writes every solid as a ``MANIFOLD_SOLID_BREP`` (``BREP_WITH_VOIDS`` when it
has inner shells) of ``ADVANCED_FACE``\\ s inside an ``ADVANCED_BREP_SHAPE_REPRESENTATION``,
with the product structure (``PRODUCT`` ... ``SHAPE_DEFINITION_REPRESENTATION``), the
length unit (millimetre, metre or inch) and the confusion accuracy. Geometry is written
exactly in the forms the B-rep holds: ``PLANE``, ``CYLINDRICAL_SURFACE``,
``CONICAL_SURFACE``, ``SPHERICAL_SURFACE``, ``TOROIDAL_SURFACE``,
``SURFACE_OF_LINEAR_EXTRUSION``, ``SURFACE_OF_REVOLUTION``, ``B_SPLINE_SURFACE_WITH_KNOTS``
(rational as the complex entity); edges as ``LINE``, ``CIRCLE``, ``ELLIPSE`` or
``B_SPLINE_CURVE_WITH_KNOTS``. Face names (the persistent names) are written as the
faces' names. No tessellation is ever written to STEP.

**Import** reads that subset back (and the ``SURFACE_CURVE``/``SEAM_CURVE`` wrappers and
full-circle edges other systems write) into B-reps. Files are untrusted: size and entity
limits, a bounded tokenizer, typed argument checks and acyclic references are enforced;
anything outside the subset is reported by entity type (``UNSUPPORTED_STEP``), never
approximated.

The parameterizations of ISO 10303-42 are the B-rep's own (``geom3d``), so surfaces and
curves map one to one.
"""

from __future__ import annotations

import datetime
import math
import re
from fractions import Fraction
from typing import Any

import numpy as np

from ...errors import CadError
from ..brep.topology import Body, Edge, Face
from ..geom3d import (
    Circle3,
    Cone,
    Cylinder,
    Ellipse3,
    Extrusion,
    Frame,
    Line3,
    Nurbs3,
    NurbsSurface,
    Plane,
    Revolution,
    Sphere,
    Torus,
)
from ..nurbs import BSplineCurve, BSplineSurface

SCHEMA = "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }"
UNITS = {
    "mm": (".MILLI.", Fraction(1, 1000)),
    "m": ("$", Fraction(1)),
    "in": (None, Fraction(254, 10000)),
}
MAX_FILE_BYTES = 200 * 2**20
MAX_ENTITIES = 5_000_000


def _f(x: float) -> str:
    """A STEP real: shortest round-trip repr, always with a decimal point."""
    if not math.isfinite(x):
        raise CadError("INVALID_GEOMETRY", "non-finite value in STEP export")
    s = repr(float(x))
    if "e" in s or "E" in s:
        mant, exp = s.lower().split("e")
        if "." not in mant:
            mant += "."
        return f"{mant}E{int(exp)}"
    if "." not in s:
        s += "."
    return s


def _str(s: str) -> str:
    out = []
    for ch in s:
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("\\X2\\" + f"{ord(ch):04X}" + "\\X0\\" if ord(ch) < 0x10000 else "?")
    return "'" + "".join(out) + "'"


class _Writer:
    def __init__(self, scale: float) -> None:
        self.lines: list[str] = []
        self.k = 0
        self.scale = scale  # metres -> file units
        self.cache: dict[Any, int] = {}

    def add(self, text: str) -> int:
        self.k += 1
        self.lines.append(f"#{self.k} = {text};")
        return self.k

    def point(self, p) -> int:
        q = np.asarray(p, dtype=np.float64) * self.scale
        return self.add(f"CARTESIAN_POINT('',({_f(q[0])},{_f(q[1])},{_f(q[2])}))")

    def direction(self, d) -> int:
        d = np.asarray(d, dtype=np.float64)
        d = d / np.linalg.norm(d)
        key = ("dir", tuple(np.round(d, 15)))
        if key not in self.cache:
            self.cache[key] = self.add(f"DIRECTION('',({_f(d[0])},{_f(d[1])},{_f(d[2])}))")
        return self.cache[key]

    def placement(self, frame: Frame) -> int:
        return self.add(
            f"AXIS2_PLACEMENT_3D('',#{self.point(frame.origin)},#{self.direction(frame.z)},"
            f"#{self.direction(frame.x)})"
        )

    def curve(self, c) -> int:
        s = self.scale
        if isinstance(c, Line3):
            v = self.add(f"VECTOR('',#{self.direction(c.direction)},1.)")
            return self.add(f"LINE('',#{self.point(c.origin)},#{v})")
        if isinstance(c, Circle3):
            return self.add(f"CIRCLE('',#{self.placement(c.frame)},{_f(c.radius * s)})")
        if isinstance(c, Ellipse3):
            return self.add(
                f"ELLIPSE('',#{self.placement(c.frame)},{_f(c.major * s)},{_f(c.minor * s)})"
            )
        if isinstance(c, Nurbs3):
            return self.bspline_curve(c.curve)
        raise CadError("UNSUPPORTED_GEOMETRY", f"cannot export curve {type(c).__name__}")

    def _knots(self, knots: np.ndarray) -> tuple[str, str]:
        vals, counts = np.unique(knots, return_counts=True)
        return ",".join(str(int(c)) for c in counts), ",".join(_f(v) for v in vals)

    def bspline_curve(self, c: BSplineCurve) -> int:
        pts = ",".join(f"#{self.point(p)}" for p in c.points)
        mults, knots = self._knots(c.knots)
        if c.weights is None:
            return self.add(
                f"B_SPLINE_CURVE_WITH_KNOTS('',{c.degree},({pts}),.UNSPECIFIED.,.F.,.F.,"
                f"({mults}),({knots}),.UNSPECIFIED.)"
            )
        w = ",".join(_f(x) for x in c.weights)
        return self.add(
            f"( BOUNDED_CURVE() B_SPLINE_CURVE({c.degree},({pts}),.UNSPECIFIED.,.F.,.F.) "
            f"B_SPLINE_CURVE_WITH_KNOTS(({mults}),({knots}),.UNSPECIFIED.) CURVE() "
            f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE(({w})) "
            "REPRESENTATION_ITEM('') )"
        )

    def surface(self, srf) -> int:
        s = self.scale
        if isinstance(srf, Plane):
            return self.add(f"PLANE('',#{self.placement(srf.frame)})")
        if isinstance(srf, Cylinder):
            return self.add(
                f"CYLINDRICAL_SURFACE('',#{self.placement(srf.frame)},{_f(srf.radius * s)})"
            )
        if isinstance(srf, Cone):
            fr, ang = srf.frame, srf.half_angle
            if ang < 0:
                fr, ang = Frame.from_axes(fr.origin, -fr.z, fr.x), -ang
            return self.add(
                f"CONICAL_SURFACE('',#{self.placement(fr)},{_f(srf.radius * s)},{_f(ang)})"
            )
        if isinstance(srf, Sphere):
            return self.add(
                f"SPHERICAL_SURFACE('',#{self.placement(srf.frame)},{_f(srf.radius * s)})"
            )
        if isinstance(srf, Torus):
            return self.add(
                f"TOROIDAL_SURFACE('',#{self.placement(srf.frame)},{_f(srf.major * s)},"
                f"{_f(srf.minor * s)})"
            )
        if isinstance(srf, Extrusion):
            d = srf.direction
            v = self.add(f"VECTOR('',#{self.direction(d)},{_f(float(np.linalg.norm(d)) * s)})")
            return self.add(f"SURFACE_OF_LINEAR_EXTRUSION('',#{self.curve(srf.curve)},#{v})")
        if isinstance(srf, Revolution):
            ax = self.add(
                f"AXIS1_PLACEMENT('',#{self.point(srf.frame.origin)},#{self.direction(srf.frame.z)})"
            )
            return self.add(f"SURFACE_OF_REVOLUTION('',#{self.curve(srf.curve)},#{ax})")
        if isinstance(srf, NurbsSurface):
            b = srf.surface
            rows = ",".join(
                "(" + ",".join(f"#{self.point(p)}" for p in row) + ")" for row in b.points
            )
            mu, ku = self._knots(b.knots_u)
            mv, kv = self._knots(b.knots_v)
            if b.weights is None:
                return self.add(
                    f"B_SPLINE_SURFACE_WITH_KNOTS('',{b.degree_u},{b.degree_v},({rows}),"
                    f".UNSPECIFIED.,.F.,.F.,.F.,({mu}),({mv}),({ku}),({kv}),.UNSPECIFIED.)"
                )
            w = ",".join("(" + ",".join(_f(x) for x in row) + ")" for row in b.weights)
            return self.add(
                f"( BOUNDED_SURFACE() B_SPLINE_SURFACE({b.degree_u},{b.degree_v},({rows}),"
                f".UNSPECIFIED.,.F.,.F.,.F.) B_SPLINE_SURFACE_WITH_KNOTS(({mu}),({mv}),({ku}),"
                f"({kv}),.UNSPECIFIED.) GEOMETRIC_REPRESENTATION_ITEM() "
                f"RATIONAL_B_SPLINE_SURFACE(({w})) REPRESENTATION_ITEM('') SURFACE() )"
            )
        raise CadError("UNSUPPORTED_GEOMETRY", f"cannot export surface {srf.kind}")

    def body(self, body: Body, name: str) -> list[int]:
        """Solids of a body (one per outer shell, with its voids)."""
        vids = [self.add(f"VERTEX_POINT('',#{self.point(p)})") for p in body.vertices]
        eids = []
        for e in body.edges:
            c = self.curve(e.curve)
            eids.append(
                self.add(f"EDGE_CURVE({_str(e.name)},#{vids[e.v0]},#{vids[e.v1]},#{c},.T.)")
            )
        fids = []
        for f in body.faces:
            bounds = []
            for li, loop in enumerate(f.loops):
                oes = ",".join(
                    f"#{self.add(f'ORIENTED_EDGE({chr(39)}{chr(39)},*,*,#{eids[ei]},{_b(fwd)})')}"
                    for ei, fwd in loop
                )
                lp = self.add(f"EDGE_LOOP('',({oes}))")
                kind = "FACE_OUTER_BOUND" if li == 0 else "FACE_BOUND"
                bounds.append(self.add(f"{kind}('',#{lp},.T.)"))
            s = self.surface(f.surface)
            refs = ",".join(f"#{b}" for b in bounds)
            fids.append(self.add(f"ADVANCED_FACE({_str(f.name)},({refs}),#{s},{_b(f.sense)})"))
        shells = [
            self.add(f"CLOSED_SHELL('',({','.join(f'#{fids[i]}' for i in sh)}))")
            for sh in body.shells
        ]
        vols = _shell_volumes(body)
        outers = [k for k, v in enumerate(vols) if v > 0]
        voids = [k for k, v in enumerate(vols) if v <= 0]
        solids = []
        for k in outers:
            mine = [v for v in voids if _void_in(body, v, k)]
            if mine:
                inner = ",".join(
                    f"#{self.add(f'ORIENTED_CLOSED_SHELL({chr(39)}{chr(39)},*,#{shells[v]},.F.)')}"
                    for v in mine
                )
                solids.append(self.add(f"BREP_WITH_VOIDS({_str(name)},#{shells[k]},({inner}))"))
            else:
                solids.append(self.add(f"MANIFOLD_SOLID_BREP({_str(name)},#{shells[k]})"))
        return solids


def _b(v: bool) -> str:
    return ".T." if v else ".F."


def _shell_volumes(body: Body) -> list[float]:
    from ..brep.properties import _face_integrals

    lo, hi = body.bounds
    c = 0.5 * (lo + hi)
    return [
        float(sum(_face_integrals(body, body.faces[i], c, 2)[0] for i in sh)) for sh in body.shells
    ]


def _void_in(body: Body, void: int, outer: int) -> bool:
    """Whether a void shell lies inside an outer shell (by bounding boxes, which suffices
    for the shells booleans produce: a void is enclosed by exactly one lump)."""

    def box(shell):
        pts = []
        for i in body.shells[shell]:
            for loop in body.faces[i].loops:
                for ei, _ in loop:
                    pts.append(body.vertices[body.edges[ei].v0])
        p = np.array(pts)
        return p.min(axis=0), p.max(axis=0)

    vlo, vhi = box(void)
    olo, ohi = box(outer)
    return bool(np.all(vlo >= olo - 1e-12) and np.all(vhi <= ohi + 1e-12))


def write_step(
    bodies: list[tuple[str, Body]], *, units: str = "mm", name: str = "aether3d", author: str = ""
) -> str:
    """STEP text for named bodies (metres in; ``units`` in the file: mm, m or in)."""
    if units not in UNITS:
        raise CadError("INVALID_UNITS", "STEP units are mm, m or in")
    prefix, factor = UNITS[units]
    w = _Writer(float(1 / factor))
    app = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})"
    )
    pctx = w.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    pdctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    if units == "in":
        si = w.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )")
        conv = w.add(f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.0254),#{si})")
        dim = w.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
        length = w.add(
            f"( CONVERSION_BASED_UNIT('INCH',#{conv}) LENGTH_UNIT() NAMED_UNIT(#{dim}) )"
        )
        accuracy = 1e-7 / 0.0254 * 1000 * 1e-3
    else:
        length = w.add(f"( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({prefix},.METRE.) )")
        accuracy = 1e-7 if units == "mm" else 1e-10
    angle = w.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid = w.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    unc = w.add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_f(accuracy)}),#{length},"
        "'distance_accuracy_value','confusion accuracy')"
    )
    ctx = w.add(
        f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) "
        "REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )"
    )
    for bname, body in bodies:
        if body.kind != "solid":
            raise CadError("UNSUPPORTED_GEOMETRY", f"{bname!r} is not a solid")
        prod = w.add(f"PRODUCT({_str(bname)},{_str(bname)},'',(#{pctx}))")
        pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
        pd = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})")
        pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
        origin = w.placement(Frame.world())
        solids = w.body(body, bname)
        items = ",".join(f"#{x}" for x in [origin, *solids])
        rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_str(bname)},({items}),#{ctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    from ..._version import __version__

    header = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('Aether3D precision model (exact B-rep)'),'2;1');",
        f"FILE_NAME({_str(name)},'{stamp}',({_str(author)}),(''),"
        f"'aether3d {__version__}','aether3d {__version__}','');",
        f"FILE_SCHEMA(('{SCHEMA}'));",
        "ENDSEC;",
        "DATA;",
    ]
    return "\n".join(header + w.lines + ["ENDSEC;", "END-ISO-10303-21;", ""])


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(
    r"""\s*(?:
      (?P<ref>\#[0-9]+)
    | (?P<str>'(?:[^']|'')*')
    | (?P<enum>\.[A-Z_0-9]+\.)
    | (?P<num>[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)
    | (?P<kw>[A-Z_][A-Z_0-9]*)
    | (?P<sym>[(),$*=;])
    )""",
    re.VERBOSE,
)


class _Ref(int):
    pass


class _Typed:
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: list) -> None:
        self.name, self.args = name, args


def _decode(s: str) -> str:
    body = s[1:-1].replace("''", "'")

    def x2(m):
        hexs = m.group(1)
        return "".join(chr(int(hexs[i : i + 4], 16)) for i in range(0, len(hexs), 4))

    body = re.sub(r"\\X2\\([0-9A-F]+)\\X0\\", x2, body)
    return body.replace("\\\\", "\\")


def _parse_data(text: str) -> dict[int, list[tuple[str, list]]]:
    """Entity instances of the DATA section: id -> [(type, args), ...] (several types for
    complex entities)."""
    m = re.search(r"\bDATA\s*;", text)
    if not m:
        raise CadError("INVALID_STEP", "no DATA section")
    pos = m.end()
    end = text.find("ENDSEC", pos)
    if end < 0:
        raise CadError("INVALID_STEP", "the DATA section does not end")
    data = re.sub(r"/\*.*?\*/", " ", text[pos:end], flags=re.S)
    tokens = _TOKEN.finditer(data)
    out: dict[int, list[tuple[str, list]]] = {}
    toks = []
    last = 0
    for t in tokens:
        if t.start() != last and data[last : t.start()].strip():
            raise CadError("INVALID_STEP", f"unexpected text near {data[last : last + 40]!r}")
        last = t.end()
        kind = t.lastgroup
        toks.append((kind, t.group(kind)))
        if len(toks) > 60 * MAX_ENTITIES:
            raise CadError("LIMIT_EXCEEDED", "the STEP file is too large")
    if data[last:].strip():
        raise CadError("INVALID_STEP", f"unexpected text near {data[last : last + 40]!r}")
    i = 0
    n = len(toks)

    def value(depth: int = 0):
        nonlocal i
        if depth > 64:
            raise CadError("INVALID_STEP", "parameters nest too deeply")
        kind, tok = toks[i]
        i += 1
        if kind == "ref":
            return _Ref(int(tok[1:]))
        if kind == "str":
            return _decode(tok)
        if kind == "enum":
            return ("enum", tok[1:-1])
        if kind == "num":
            if re.fullmatch(r"[+-]?[0-9]+", tok):
                return int(tok)
            v = float(tok)
            if not math.isfinite(v):
                raise CadError("INVALID_STEP", "non-finite number")
            return v
        if kind == "kw":
            if toks[i] != ("sym", "("):
                raise CadError("INVALID_STEP", f"{tok} is not followed by arguments")
            return _Typed(tok, args(depth + 1))
        if tok == "(":
            i -= 1
            return args(depth + 1)
        if tok == "$":
            return None
        if tok == "*":
            return "*"
        raise CadError("INVALID_STEP", f"unexpected {tok!r}")

    def args(depth: int) -> list:
        nonlocal i
        if toks[i] != ("sym", "("):
            raise CadError("INVALID_STEP", "expected '('")
        i += 1
        out_args = []
        if toks[i] == ("sym", ")"):
            i += 1
            return out_args
        while True:
            out_args.append(value(depth))
            kind, tok = toks[i]
            i += 1
            if tok == ")":
                return out_args
            if tok != ",":
                raise CadError("INVALID_STEP", f"expected ',' or ')', got {tok!r}")

    while i < n:
        kind, tok = toks[i]
        if kind != "ref" or toks[i + 1] != ("sym", "="):
            raise CadError("INVALID_STEP", f"expected an entity instance, got {tok!r}")
        eid = int(tok[1:])
        i += 2
        kind, tok = toks[i]
        parts = []
        if tok == "(":
            i += 1
            while toks[i] != ("sym", ")"):
                kk, name = toks[i]
                if kk != "kw":
                    raise CadError("INVALID_STEP", f"#{eid}: expected an entity type")
                i += 1
                parts.append((name, args(1)))
            i += 1
        else:
            if kind != "kw":
                raise CadError("INVALID_STEP", f"#{eid}: expected an entity type")
            i += 1
            parts.append((tok, args(1)))
        if toks[i] != ("sym", ";"):
            raise CadError("INVALID_STEP", f"#{eid}: expected ';'")
        i += 1
        if eid in out:
            raise CadError("INVALID_STEP", f"#{eid} is defined twice")
        out[eid] = parts
        if len(out) > MAX_ENTITIES:
            raise CadError("LIMIT_EXCEEDED", f"more than {MAX_ENTITIES} entities")
    return out


class _Reader:
    def __init__(self, ents: dict[int, list[tuple[str, list]]]) -> None:
        self.e = ents
        self.memo: dict[tuple[str, int], Any] = {}
        self.scale = 1e-3  # file length unit -> metres (millimetres unless stated)
        self.angle = 1.0
        self.unsupported: dict[str, int] = {}
        self._units()

    def one(self, ref: int, *types: str) -> tuple[str, list]:
        parts = self.e.get(int(ref))
        if parts is None:
            raise CadError("INVALID_STEP", f"#{ref} is referenced but not defined")
        if len(parts) == 1:
            t, a = parts[0]
            if types and t not in types:
                raise CadError(
                    "UNSUPPORTED_STEP", f"#{ref} is a {t}; expected {' or '.join(types)}"
                )
            return t, a
        names = {t for t, _ in parts}
        for t in types:
            if t in names:
                return t, dict(parts)[t]
        return parts[0]

    def complex(self, ref: int) -> dict[str, list]:
        return dict(self.e[int(ref)])

    def _units(self) -> None:
        for parts in self.e.values():
            names = {t for t, _ in parts}
            if "GLOBAL_UNIT_ASSIGNED_CONTEXT" not in names:
                continue
            units = dict(parts)["GLOBAL_UNIT_ASSIGNED_CONTEXT"][0]
            for u in units:
                comp = self.complex(u)
                if "LENGTH_UNIT" in comp:
                    self.scale = self._length_factor(comp)
                if "PLANE_ANGLE_UNIT" in comp:
                    if "SI_UNIT" in comp:
                        self.angle = 1.0
                    elif "CONVERSION_BASED_UNIT" in comp:
                        conv = comp["CONVERSION_BASED_UNIT"]
                        _, m = self.one(conv[1])
                        v = m[0].args[0] if isinstance(m[0], _Typed) else m[0]
                        self.angle = float(v)
            return

    def _length_factor(self, comp: dict) -> float:
        if "SI_UNIT" in comp:
            prefix = comp["SI_UNIT"][0]
            table = {"MILLI": 1e-3, "CENTI": 1e-2, "DECI": 1e-1, "KILO": 1e3, "MICRO": 1e-6}
            if prefix is None:
                return 1.0
            p = prefix[1] if isinstance(prefix, tuple) else prefix
            if p not in table:
                raise CadError("UNSUPPORTED_STEP", f"unsupported length prefix {p}")
            return table[p]
        if "CONVERSION_BASED_UNIT" in comp:
            conv = comp["CONVERSION_BASED_UNIT"]
            _, m = self.one(conv[1], "LENGTH_MEASURE_WITH_UNIT")
            v = m[0].args[0] if isinstance(m[0], _Typed) else m[0]
            base = self._length_factor(self.complex(m[1]))
            return float(v) * base
        raise CadError("UNSUPPORTED_STEP", "unsupported length unit")

    # -- geometry --------------------------------------------------------------------

    def point(self, ref) -> np.ndarray:
        key = ("pt", int(ref))
        if key not in self.memo:
            _, a = self.one(ref, "CARTESIAN_POINT")
            c = [float(v) for v in a[1]]
            if len(c) != 3 or not all(math.isfinite(v) for v in c):
                raise CadError("INVALID_STEP", f"#{ref}: a 3D point needs three finite numbers")
            self.memo[key] = np.array(c) * self.scale
        return self.memo[key]

    def direction(self, ref) -> np.ndarray:
        _, a = self.one(ref, "DIRECTION")
        d = np.array([float(v) for v in a[1]])
        n = float(np.linalg.norm(d))
        if len(d) != 3 or not n > 0:
            raise CadError("INVALID_STEP", f"#{ref}: invalid direction")
        return d / n

    def placement(self, ref) -> Frame:
        _, a = self.one(ref, "AXIS2_PLACEMENT_3D")
        o = self.point(a[1])
        z = self.direction(a[2]) if a[2] is not None else np.array([0.0, 0.0, 1.0])
        x = self.direction(a[3]) if a[3] is not None else None
        return Frame.from_axes(o, z, x)

    def curve(self, ref):
        key = ("curve", int(ref))
        if key in self.memo:
            return self.memo[key]
        t, a = self.one(
            ref,
            "LINE",
            "CIRCLE",
            "ELLIPSE",
            "B_SPLINE_CURVE_WITH_KNOTS",
            "SURFACE_CURVE",
            "SEAM_CURVE",
            "B_SPLINE_CURVE",
        )
        if t == "LINE":
            p = self.point(a[1])
            _, v = self.one(a[2], "VECTOR")
            c = Line3(p, self.direction(v[1]))
        elif t == "CIRCLE":
            c = Circle3(self.placement(a[1]), float(a[2]) * self.scale)
        elif t == "ELLIPSE":
            fr = self.placement(a[1])
            s1, s2 = float(a[2]) * self.scale, float(a[3]) * self.scale
            if s1 < s2:
                fr = Frame(fr.origin, fr.y, -fr.x, fr.z)
                s1, s2 = s2, s1
                self.memo[("ellipse_shift", int(ref))] = True
            c = Ellipse3(fr, s1, s2)
        elif t in ("SURFACE_CURVE", "SEAM_CURVE"):
            c = self.curve(a[1])
        else:
            c = Nurbs3(self._bspline_curve(ref))
        self.memo[key] = c
        return c

    def _bspline_curve(self, ref) -> BSplineCurve:
        parts = self.complex(ref)
        if "B_SPLINE_CURVE_WITH_KNOTS" in parts and len(parts) == 1:
            a = parts["B_SPLINE_CURVE_WITH_KNOTS"]
            deg, cps, mults, knots = a[1], a[2], a[6], a[7]
            weights = None
        else:
            b = parts["B_SPLINE_CURVE"]
            k = parts["B_SPLINE_CURVE_WITH_KNOTS"]
            deg, cps = b[0], b[1]
            mults, knots = k[0], k[1]
            weights = parts.get("RATIONAL_B_SPLINE_CURVE", [None])[0]
        kv = np.repeat(np.array(knots, dtype=np.float64), np.array(mults, dtype=np.int64))
        pts = np.array([self.point(p) for p in cps])
        return BSplineCurve(
            int(deg), kv, pts, None if weights is None else np.array(weights, float)
        )

    def surface(self, ref):
        key = ("surf", int(ref))
        if key in self.memo:
            return self.memo[key]
        t, a = self.one(
            ref,
            "PLANE",
            "CYLINDRICAL_SURFACE",
            "CONICAL_SURFACE",
            "SPHERICAL_SURFACE",
            "TOROIDAL_SURFACE",
            "B_SPLINE_SURFACE_WITH_KNOTS",
            "SURFACE_OF_LINEAR_EXTRUSION",
            "SURFACE_OF_REVOLUTION",
            "B_SPLINE_SURFACE",
        )
        s = self.scale
        if t == "PLANE":
            srf = Plane(self.placement(a[1]))
        elif t == "CYLINDRICAL_SURFACE":
            srf = Cylinder(self.placement(a[1]), float(a[2]) * s)
        elif t == "CONICAL_SURFACE":
            srf = Cone(self.placement(a[1]), float(a[2]) * s, float(a[3]) * self.angle)
        elif t == "SPHERICAL_SURFACE":
            srf = Sphere(self.placement(a[1]), float(a[2]) * s)
        elif t == "TOROIDAL_SURFACE":
            srf = Torus(self.placement(a[1]), float(a[2]) * s, float(a[3]) * s)
        elif t == "SURFACE_OF_LINEAR_EXTRUSION":
            c = self.curve(a[1])
            if not isinstance(c, Nurbs3):
                raise CadError("UNSUPPORTED_STEP", f"#{ref}: extrusions of analytic curves")
            _, v = self.one(a[2], "VECTOR")
            srf = Extrusion(c, self.direction(v[1]) * float(v[2]) * s)
        elif t == "SURFACE_OF_REVOLUTION":
            c = self.curve(a[1])
            if not isinstance(c, Nurbs3):
                raise CadError("UNSUPPORTED_STEP", f"#{ref}: revolutions of analytic curves")
            _, ax = self.one(a[2], "AXIS1_PLACEMENT")
            srf = Revolution(Frame.from_axes(self.point(ax[1]), self.direction(ax[2])), c)
        else:
            parts = self.complex(ref)
            if "B_SPLINE_SURFACE" in parts:
                b, k = parts["B_SPLINE_SURFACE"], parts["B_SPLINE_SURFACE_WITH_KNOTS"]
                du, dv, rows = b[0], b[1], b[2]
                um, vm, uk, vk = k[0], k[1], k[2], k[3]
                w = parts.get("RATIONAL_B_SPLINE_SURFACE", [None])[0]
            else:
                du, dv, rows = a[1], a[2], a[3]
                um, vm, uk, vk = a[7], a[8], a[9], a[10]
                w = None
            pts = np.array([[self.point(p) for p in row] for row in rows])
            ku = np.repeat(np.array(uk, float), np.array(um, int))
            kv = np.repeat(np.array(vk, float), np.array(vm, int))
            srf = NurbsSurface(
                BSplineSurface(
                    int(du), int(dv), ku, kv, pts, None if w is None else np.array(w, float)
                )
            )
        self.memo[key] = srf
        return srf

    # -- topology ----------------------------------------------------------------------

    def body(self, ref) -> Body:
        t, a = self.one(ref, "MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")
        shells = [a[1]]
        if t == "BREP_WITH_VOIDS":
            for v in a[2]:
                _, oa = self.one(v, "ORIENTED_CLOSED_SHELL")
                shells.append(oa[2])
        verts: dict[int, int] = {}
        vpos: list[np.ndarray] = []
        edges: list[Edge] = []
        emap: dict[int, int] = {}
        faces: list[Face] = []
        shell_faces = []

        def vertex(r) -> int:
            if int(r) not in verts:
                _, va = self.one(r, "VERTEX_POINT")
                verts[int(r)] = len(vpos)
                vpos.append(self.point(va[1]))
            return verts[int(r)]

        def edge(r) -> tuple[int, bool]:
            """Edge index and whether our edge runs from the STEP edge's start to end."""
            if int(r) not in emap:
                _, ea = self.one(r, "EDGE_CURVE")
                vs, ve = vertex(ea[1]), vertex(ea[2])
                c = self.curve(ea[3])
                same = ea[4] == ("enum", "T")
                ps, pe = vpos[vs], vpos[ve]
                t0, t1, forward = _edge_params(c, ps, pe, same)
                v0, v1 = (vs, ve) if forward else (ve, vs)
                emap[int(r)] = len(edges)
                edges.append(Edge(c, t0, t1, v0, v1, ea[0] or "", 0.0))
                self.memo[("edge_fwd", int(r))] = forward
            return emap[int(r)], self.memo[("edge_fwd", int(r))]

        for sh in shells:
            _, sa = self.one(sh, "CLOSED_SHELL", "OPEN_SHELL")
            idx = []
            for fr in sa[1]:
                ft, fa = self.one(fr, "ADVANCED_FACE", "FACE_SURFACE")
                surf = self.surface(fa[2])
                sense = fa[3] == ("enum", "T")
                loops = []
                outer_first = []
                for br in fa[1]:
                    bt, ba = self.one(br, "FACE_OUTER_BOUND", "FACE_BOUND")
                    lt, la = self.one(ba[1], "EDGE_LOOP", "VERTEX_LOOP")
                    if lt == "VERTEX_LOOP":
                        continue  # a pole: a degenerate loop of one vertex
                    co = []
                    for oe in la[1]:
                        _, oa = self.one(oe, "ORIENTED_EDGE")
                        ei, fwd_step = edge(oa[3])
                        along = oa[4] == ("enum", "T")
                        co.append((ei, along == fwd_step))
                    if ba[2] != ("enum", "T"):
                        co = [(ei, not f) for ei, f in reversed(co)]
                    (outer_first if bt == "FACE_OUTER_BOUND" else loops).append(tuple(co))
                all_loops = tuple(outer_first + loops)
                if not all_loops:
                    # A closed surface bounded by nothing but vertex loops (a whole sphere
                    # or torus): this solid is that surface alone.
                    if len(sa[1]) == 1 and isinstance(surf, (Sphere, Torus)) and sense:
                        from ..brep import build

                        whole = (
                            build.sphere(surf.radius, frame=surf.frame, prefix=fa[0] or "sphere")
                            if isinstance(surf, Sphere)
                            else build.torus(
                                surf.major, surf.minor, frame=surf.frame, prefix=fa[0] or "torus"
                            )
                        )
                        return whole
                    raise CadError(
                        "UNSUPPORTED_STEP",
                        f"#{fr}: a face without edges on a {surf.kind}",
                    )
                faces.append(Face(surf, sense, all_loops, fa[0] or "", 0, None))
                idx.append(len(faces) - 1)
            shell_faces.append(tuple(idx))
        from ..brep.topology import infer_window

        body = Body(np.array(vpos), tuple(edges), tuple(faces), tuple(shell_faces), "solid")
        fixed = []
        for f in body.faces:
            w = infer_window(body, f) if f.surface.periodic_u else None
            fixed.append(Face(f.surface, f.sense, f.loops, f.name, f.group, w))
        return Body(body.vertices, body.edges, tuple(fixed), body.shells, "solid")


def _edge_params(c, ps, pe, same: bool) -> tuple[float, float, bool]:
    """(t0, t1, forward): the parameter range and whether it runs from the STEP edge's
    start vertex to its end vertex."""
    if isinstance(c, (Circle3, Ellipse3)):
        a = float(c.inverse(ps)[()])
        b = float(c.inverse(pe)[()])
        if same:
            while b <= a + 1e-12:
                b += 2 * math.pi
            return a, b, True
        while a <= b + 1e-12:
            a += 2 * math.pi
        return b, a, False
    if isinstance(c, Line3):
        a, b = float(c.inverse(ps)[()]), float(c.inverse(pe)[()])
    else:
        a, b = float(c.inverse(ps)[0]), float(c.inverse(pe)[0])
    return (a, b, True) if a <= b else (b, a, False)


def read_step(text: str) -> list[tuple[str, Body]]:
    """Every solid of a STEP file as (name, body) in metres."""
    if len(text) > MAX_FILE_BYTES:
        raise CadError("LIMIT_EXCEEDED", f"STEP files are limited to {MAX_FILE_BYTES} bytes")
    if not text.lstrip().startswith("ISO-10303-21"):
        raise CadError("INVALID_STEP", "not an ISO 10303-21 file")
    ents = _parse_data(text)
    placed = {"NEXT_ASSEMBLY_USAGE_OCCURRENCE", "MAPPED_ITEM"}
    if any(parts[0][0] in placed for parts in ents.values()):
        raise CadError(
            "UNSUPPORTED_STEP",
            "STEP assemblies (placed product instances) are not supported",
            hint="export each part as its own STEP file",
        )
    r = _Reader(ents)
    out = []
    for eid, parts in sorted(ents.items()):
        t = parts[0][0]
        if len(parts) == 1 and t in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS"):
            name = parts[0][1][0] or f"solid {eid}"
            out.append((name, r.body(eid)))
    if not out:
        kinds = sorted({p[0][0] for p in ents.values()})
        raise CadError(
            "UNSUPPORTED_STEP",
            "the file holds no MANIFOLD_SOLID_BREP or BREP_WITH_VOIDS",
            hint="supported: exact solid B-reps (AP203/AP214/AP242); entities seen: "
            + ", ".join(kinds[:20]),
        )
    return out
