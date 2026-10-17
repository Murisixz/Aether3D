"""DXF: 2D geometry in, sketches and drawings out.

**Reading** (``read_dxf``): ASCII DXF of any version, the ``ENTITIES`` section's
``LINE``, ``ARC``, ``CIRCLE``, ``LWPOLYLINE`` and ``POLYLINE``/``VERTEX`` (straight and
bulged segments) in the drawing's XY plane (extrusion direction +Z or -Z), with the
length unit from ``$INSUNITS``. Everything else -- blocks and their ``INSERT``s, text,
splines, hatches, 3D entities -- is counted in ``skipped``, never guessed at. The text is
untrusted: a size limit, a pair limit, a strict group-code grammar and finite numbers.

**Writing**: sketches (``write_sketch_dxf``: ``LINE``, ``ARC``, ``CIRCLE`` at real size)
and drawings (:meth:`Drawing.to_dxf`), both ASCII DXF R12 (AC1009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...errors import CadError

MAX_FILE_BYTES = 64 * 2**20
MAX_PAIRS = 8_000_000
MAX_SEGMENTS = 200_000
# $INSUNITS codes -> metres (0: unitless; the caller's default unit applies).
INSUNITS = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 8: 2.54e-8, 9: 2.54e-5, 10: 0.9144}


@dataclass
class DxfGeometry:
    lines: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # (center, radius, start angle, end angle) counter-clockwise, radians
    arcs: list[tuple[np.ndarray, float, float, float]] = field(default_factory=list)
    circles: list[tuple[np.ndarray, float]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    unit_m: float | None = None  # metres per drawing unit, if the file says
    layers: set[str] = field(default_factory=set)

    def count(self) -> int:
        return len(self.lines) + len(self.arcs) + len(self.circles)


def _pairs(text: str) -> list[tuple[int, str]]:
    if len(text) > MAX_FILE_BYTES:
        raise CadError("LIMIT_EXCEEDED", f"DXF files are limited to {MAX_FILE_BYTES} bytes")
    lines = text.splitlines()
    if len(lines) % 2:
        lines = lines[:-1] if not lines[-1].strip() else lines
    if len(lines) // 2 > MAX_PAIRS:
        raise CadError("LIMIT_EXCEEDED", "the DXF file has too many group codes")
    out = []
    for i in range(0, len(lines) - 1, 2):
        raw = lines[i].strip()
        try:
            code = int(raw)
        except ValueError:
            raise CadError(
                "INVALID_DXF", f"line {i + 1}: expected a group code, got {raw[:20]!r}"
            ) from None
        out.append((code, lines[i + 1].strip()))
    return out


def _num(v: str, where: str) -> float:
    try:
        x = float(v)
    except ValueError:
        raise CadError("INVALID_DXF", f"{where}: {v[:20]!r} is not a number") from None
    if not math.isfinite(x) or abs(x) > 1e12:
        raise CadError("INVALID_DXF", f"{where}: {v[:20]!r} is out of range")
    return x


def _entities(pairs):
    """(type, [(code, value), ...]) of the ENTITIES section, and the header's $INSUNITS."""
    insunits = None
    ents = []
    section = None
    i = 0
    n = len(pairs)
    while i < n:
        code, val = pairs[i]
        if code == 0 and val == "SECTION" and i + 1 < n and pairs[i + 1][0] == 2:
            section = pairs[i + 1][1]
            i += 2
            continue
        if code == 0 and val == "ENDSEC":
            section = None
            i += 1
            continue
        if section == "HEADER" and code == 9 and val == "$INSUNITS" and i + 1 < n:
            insunits = int(_num(pairs[i + 1][1], "$INSUNITS"))
            i += 2
            continue
        if section == "ENTITIES" and code == 0:
            kind = val
            body = []
            i += 1
            while i < n and pairs[i][0] != 0:
                body.append(pairs[i])
                i += 1
            ents.append((kind, body))
            continue
        i += 1
    return ents, insunits


def _get(body, code, default=None):
    for c, v in body:
        if c == code:
            return v
    return default


def _ocs(body, where) -> float:
    """+1 for the XY plane, -1 for its mirror (extrusion -Z); anything else unsupported."""
    ex = [_num(_get(body, c, d), where) for c, d in ((210, "0"), (220, "0"), (230, "1"))]
    if abs(ex[0]) > 1e-9 or abs(ex[1]) > 1e-9:
        return 0.0
    return 1.0 if ex[2] > 0 else -1.0


def _bulge_arc(p1, p2, b):
    """The arc of a bulged polyline segment: (center, radius, a0, a1) counter-clockwise."""
    c = float(np.linalg.norm(p2 - p1))
    if c == 0:
        return None
    mid = 0.5 * (p1 + p2)
    u = (p2 - p1) / c
    n = np.array([-u[1], u[0]])
    h = (c / 2) * (1 - b * b) / (2 * b)
    center = mid + n * h
    r = math.hypot(*(p1 - center))
    a1 = math.atan2(*(p1 - center)[::-1])
    a2 = math.atan2(*(p2 - center)[::-1])
    return (center, r, a1, a2) if b > 0 else (center, r, a2, a1)


def read_dxf(text: str, *, layers: set[str] | None = None) -> DxfGeometry:
    """The 2D lines, arcs and circles of a DXF file (drawing units; see ``unit_m``)."""
    if not isinstance(text, str):
        raise CadError("INVALID_DXF", "DXF text must be a string")
    pairs = _pairs(text)
    ents, insunits = _entities(pairs)
    if not ents and not any(v == "ENTITIES" for _, v in pairs):
        raise CadError("INVALID_DXF", "no ENTITIES section")
    g = DxfGeometry(unit_m=INSUNITS.get(insunits) if insunits else None)

    def skip(kind):
        g.skipped[kind] = g.skipped.get(kind, 0) + 1

    def segments_left():
        if g.count() >= MAX_SEGMENTS:
            raise CadError("LIMIT_EXCEEDED", f"more than {MAX_SEGMENTS} curves")

    k = 0
    while k < len(ents):
        kind, body = ents[k]
        k += 1
        layer = _get(body, 8, "0")
        if layers is not None and layer not in layers and kind not in ("VERTEX", "SEQEND"):
            continue
        where = f"{kind} on layer {layer!r}"
        if kind in ("LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE"):
            m = _ocs(body, where)
            if m == 0.0:
                skip(f"{kind} (not in the XY plane)")
                continue
            g.layers.add(layer)
        segments_left()
        if kind == "LINE":
            a = np.array([_num(_get(body, 10, "0"), where), _num(_get(body, 20, "0"), where)])
            b = np.array([_num(_get(body, 11, "0"), where), _num(_get(body, 21, "0"), where)])
            if np.linalg.norm(b - a) > 0:
                g.lines.append((a, b))
        elif kind in ("ARC", "CIRCLE"):
            c = np.array([_num(_get(body, 10, "0"), where), _num(_get(body, 20, "0"), where)])
            r = _num(_get(body, 40, "0"), where)
            if r <= 0:
                raise CadError("INVALID_DXF", f"{where}: radius must be positive")
            if m < 0:
                c[0] = -c[0]
            if kind == "CIRCLE":
                g.circles.append((c, r))
            else:
                a0 = math.radians(_num(_get(body, 50, "0"), where))
                a1 = math.radians(_num(_get(body, 51, "360"), where))
                if m < 0:  # mirrored: angles reflect and the direction reverses
                    a0, a1 = math.pi - a1, math.pi - a0
                g.arcs.append((c, r, a0, a1))
        elif kind == "LWPOLYLINE":
            closed = int(_num(_get(body, 70, "0"), where)) & 1
            verts, bulges = [], []
            for code, v in body:
                if code == 10:
                    verts.append([_num(v, where), 0.0])
                    bulges.append(0.0)
                elif code == 20 and verts:
                    verts[-1][1] = _num(v, where)
                elif code == 42 and verts:
                    bulges[-1] = _num(v, where)
            _polyline(g, np.array(verts) * np.array([m, 1.0]) if verts else [], bulges, closed, m)
        elif kind == "POLYLINE":
            flags = int(_num(_get(body, 70, "0"), where))
            if flags & (8 | 16 | 64):
                skip("POLYLINE (3D or mesh)")
                while k < len(ents) and ents[k][0] != "SEQEND":
                    k += 1
                k += 1
                continue
            verts, bulges = [], []
            while k < len(ents) and ents[k][0] == "VERTEX":
                vb = ents[k][1]
                verts.append([_num(_get(vb, 10, "0"), where) * m, _num(_get(vb, 20, "0"), where)])
                bulges.append(_num(_get(vb, 42, "0"), where))
                k += 1
            if k < len(ents) and ents[k][0] == "SEQEND":
                k += 1
            _polyline(g, np.array(verts) if verts else [], bulges, flags & 1, m)
        elif kind in ("VERTEX", "SEQEND"):
            continue
        else:
            skip(kind)
    return g


def _polyline(g: DxfGeometry, verts, bulges, closed, mirror):
    n = len(verts)
    if n < 2:
        return
    for i in range(n if closed else n - 1):
        a, b = verts[i], verts[(i + 1) % n]
        if np.linalg.norm(b - a) == 0:
            continue
        bulge = bulges[i] * mirror
        if abs(bulge) < 1e-12:
            g.lines.append((a.copy(), b.copy()))
        else:
            arc = _bulge_arc(a, b, bulge)
            if arc is not None:
                g.arcs.append(arc)
        if g.count() > MAX_SEGMENTS:
            raise CadError("LIMIT_EXCEEDED", f"more than {MAX_SEGMENTS} curves")


def sketch_definition(g: DxfGeometry, scale_m: float, plane, merge_tol: float) -> dict:
    """A sketch definition (metres) with shared end points (within ``merge_tol``)."""
    ents: list[dict] = []
    keys: dict[tuple[int, int], str] = {}

    def pt(p) -> str:
        q = np.asarray(p, dtype=np.float64) * scale_m
        key = (round(q[0] / merge_tol), round(q[1] / merge_tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                hit = keys.get((key[0] + dx, key[1] + dy))
                if hit is not None:
                    return hit
        pid = f"p{len(keys) + 1}"
        keys[key] = pid
        ents.append({"id": pid, "type": "point", "at": [float(q[0]), float(q[1])]})
        return pid

    n = 0
    for a, b in g.lines:
        pa, pb = pt(a), pt(b)
        if pa != pb:
            n += 1
            ents.append({"id": f"l{n}", "type": "line", "start": pa, "end": pb})
    for k, (c, r, a0, a1) in enumerate(g.arcs, 1):
        s = c + r * np.array([math.cos(a0), math.sin(a0)])
        e = c + r * np.array([math.cos(a1), math.sin(a1)])
        ents.append(
            {
                "id": f"a{k}",
                "type": "arc",
                "center": pt(c),
                "start": pt(s),
                "end": pt(e),
                "radius": float(r * scale_m),
            }
        )
    for k, (c, r) in enumerate(g.circles, 1):
        ents.append(
            {"id": f"c{k}", "type": "circle", "center": pt(c), "radius": float(r * scale_m)}
        )
    return {"plane": plane, "entities": ents, "constraints": []}


def write_sketch_dxf(result, unit: str = "mm") -> str:
    """ASCII DXF R12 of a solved sketch's lines, arcs and circles (sketch plane
    coordinates at real size in ``unit``; construction geometry on layer CONSTRUCTION)."""
    from ..drawing.export import DXF_UNITS, _num

    factor, insunits = DXF_UNITS[unit]
    k = 1.0 / factor
    defn = result.definition
    pts = {e["id"]: np.array(e["at"]) * k for e in defn["entities"] if e["type"] == "point"}
    out = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1009",
        "9",
        "$INSUNITS",
        "70",
        str(insunits),
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "TABLES",
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        "2",
        "0",
        "LAYER",
        "2",
        "0",
        "70",
        "0",
        "62",
        "7",
        "6",
        "CONTINUOUS",
        "0",
        "LAYER",
        "2",
        "CONSTRUCTION",
        "70",
        "0",
        "62",
        "8",
        "6",
        "CONTINUOUS",
        "0",
        "ENDTAB",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    for e in defn["entities"]:
        layer = "CONSTRUCTION" if e.get("construction") else "0"
        t = e["type"]
        if t == "line":
            a, b = pts[e["start"]], pts[e["end"]]
            out += [
                "0",
                "LINE",
                "8",
                layer,
                "10",
                _num(a[0]),
                "20",
                _num(a[1]),
                "30",
                "0",
                "11",
                _num(b[0]),
                "21",
                _num(b[1]),
                "31",
                "0",
            ]
        elif t == "circle":
            c = pts[e["center"]]
            out += [
                "0",
                "CIRCLE",
                "8",
                layer,
                "10",
                _num(c[0]),
                "20",
                _num(c[1]),
                "30",
                "0",
                "40",
                _num(float(e["radius"]) * k),
            ]
        elif t == "arc":
            c, s, f = pts[e["center"]], pts[e["start"]], pts[e["end"]]
            a0 = math.degrees(math.atan2(s[1] - c[1], s[0] - c[0]))
            a1 = math.degrees(math.atan2(f[1] - c[1], f[0] - c[0]))
            out += [
                "0",
                "ARC",
                "8",
                layer,
                "10",
                _num(c[0]),
                "20",
                _num(c[1]),
                "30",
                "0",
                "40",
                _num(float(e["radius"]) * k),
                "50",
                _num(a0 % 360),
                "51",
                _num(a1 % 360),
            ]
        elif t == "spline":
            continue
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"


def _install() -> None:
    from ..document import CadDocument
    from ..kinds import SketchHandle
    from ..units import unit as unit_of

    def import_dxf(
        self: CadDocument, text: str, name: str, *, plane="XY", layers=None, unit: str | None = None
    ) -> SketchHandle:
        """A new sketch from a DXF file's 2D geometry. The file's ``$INSUNITS`` gives its
        unit; ``unit`` overrides it (and is required for unitless files)."""
        g = read_dxf(text, layers=set(layers) if layers is not None else None)
        if unit is not None:
            scale = float(unit_of(unit).factor)
        elif g.unit_m is not None:
            scale = g.unit_m
        else:
            raise CadError(
                "UNIT_REQUIRED",
                "the DXF file does not say its unit ($INSUNITS)",
                hint="pass unit='mm' (or the drawing's unit)",
            )
        if g.count() == 0:
            raise CadError(
                "NOTHING_TO_IMPORT",
                "the DXF file has no supported 2D geometry",
                details={"skipped": g.skipped},
            )
        size = (
            max(
                [float(np.max(np.abs(np.concatenate([a, b])))) for a, b in g.lines]
                + [float(np.max(np.abs(c)) + r) for c, r, *_ in g.arcs]
                + [float(np.max(np.abs(c)) + r) for c, r in g.circles]
            )
            * scale
        )
        defn = sketch_definition(g, scale, plane, max(size, 1e-9) * 1e-9)
        item = self.add("sketch", name, defn, label=f"Import DXF as sketch {name!r}")
        handle = SketchHandle(self, item)
        handle.import_report = {
            "skipped": dict(g.skipped),
            "layers": sorted(g.layers),
            "lines": len(g.lines),
            "arcs": len(g.arcs),
            "circles": len(g.circles),
        }
        return handle

    def export_sketch_dxf(self: CadDocument, sketch, unit: str | None = None) -> str:
        u = unit or self._settings.get("length_unit", "mm")
        return write_sketch_dxf(self.result(self.item(sketch, "sketch")), u)

    CadDocument.import_dxf = import_dxf
    CadDocument.export_sketch_dxf = export_sketch_dxf


_install()
