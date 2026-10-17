"""SVG and DXF output of drawings.

SVG: one sheet in paper millimetres (``viewBox`` = the sheet), strokes with their paper
weights and dash patterns, solid fills with the even-odd rule, text as ``<text>``.

DXF: ASCII DXF R12 (AC1009), the most widely readable version, in model space at real
size in the chosen unit (``$INSUNITS`` set as well). Each style is a layer with its line
type (``HIDDEN`` dashed, ``CENTER`` dash-dot); polylines are ``POLYLINE``/``VERTEX``,
two-point lines ``LINE``, solid fills ``SOLID`` triangles (R12 has no ``HATCH``) with
their boundary on the fill's layer, text ``TEXT``. Dimensions are written as plain
geometry and text (not associative ``DIMENSION`` entities).
"""

from __future__ import annotations

import math
from html import escape

import numpy as np

from ..triangulate import triangulate
from .model import Drawing

DXF_UNITS = {"mm": (0.001, 4), "cm": (0.01, 5), "m": (1.0, 6), "in": (0.0254, 1), "ft": (0.3048, 2)}


def _num(x: float) -> str:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError("non-finite coordinate")
    # The shortest decimal that reads back as the same double (no exponent).
    out = np.format_float_positional(v, unique=True, trim="-")
    return "0" if out in ("-0", "") else out


def to_svg(d: Drawing) -> str:
    paper = d.paper()
    w, h = paper["size"]

    def pts(p):
        return " ".join(f"{_num(x)},{_num(h - y)}" for x, y in p)

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(w)}mm" height="{_num(h)}mm" '
        f'viewBox="0 0 {_num(w)} {_num(h)}">',
        f"<title>{escape(d.title)}</title>",
        f'<rect x="0" y="0" width="{_num(w)}" height="{_num(h)}" fill="white"/>',
    ]
    for loops, gray, _ in paper["fills"]:
        g = int(round(gray * 255))
        path = " ".join("M " + " L ".join(pts([p]) for p in lp) + " Z" for lp in loops)
        out.append(f'<path d="{path}" fill="rgb({g},{g},{g})" fill-rule="evenodd" stroke="none"/>')
    out.append('<g fill="none" stroke="black" stroke-linecap="round" stroke-linejoin="round">')
    for p, weight, pattern in paper["strokes"]:
        dash = f' stroke-dasharray="{" ".join(_num(x) for x in pattern)}"' if pattern else ""
        out.append(f'<polyline points="{pts(p)}" stroke-width="{_num(weight)}"{dash}/>')
    out.append("</g>")
    out.append('<g font-family="DejaVu Sans, Arial, sans-serif" fill="black" stroke="none">')
    anchor = {"left": "start", "center": "middle", "right": "end"}
    for pos, height, text, angle, align in paper["texts"]:
        x, y = float(pos[0]), h - float(pos[1])
        rot = (
            f' transform="rotate({_num(-math.degrees(angle))} {_num(x)} {_num(y)})"'
            if angle
            else ""
        )
        out.append(
            f'<text x="{_num(x)}" y="{_num(y)}" font-size="{_num(height / 0.72)}" '
            f'text-anchor="{anchor.get(align, "start")}"{rot}>{escape(text)}</text>'
        )
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _dxf_text(s: str) -> str:
    out = []
    for ch in s:
        if ch in "\r\n\t\0":
            out.append(" ")
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append(f"\\U+{ord(ch):04X}")
    return "".join(out)[:250]


def _layer_name(layer: str, style: str) -> str:
    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in layer.upper())[:24] or "0"
    return f"{base}-{style.upper()}" if style else base


def to_dxf(d: Drawing, unit: str = "mm") -> str:
    factor, insunits = DXF_UNITS[unit]
    k = 1.0 / factor  # model metres -> file units
    to_paper, to_model, s = d.transform()
    text_scale = 1.0 / s * k  # paper mm -> file units
    ents: list[str] = []
    layers: dict[str, str] = {}

    def lay(name: str, ltype: str = "CONTINUOUS") -> str:
        layers.setdefault(name, ltype)
        return name

    def poly(points, layer, closed=False):
        p = np.asarray(points) * k
        if len(p) == 2 and not closed:
            ents.extend(
                [
                    "0",
                    "LINE",
                    "8",
                    layer,
                    "10",
                    _num(p[0, 0]),
                    "20",
                    _num(p[0, 1]),
                    "30",
                    "0",
                    "11",
                    _num(p[1, 0]),
                    "21",
                    _num(p[1, 1]),
                    "31",
                    "0",
                ]
            )
            return
        ents.extend(
            [
                "0",
                "POLYLINE",
                "8",
                layer,
                "66",
                "1",
                "10",
                "0",
                "20",
                "0",
                "30",
                "0",
                "70",
                "1" if closed else "0",
            ]
        )
        for x, y in p:
            ents.extend(["0", "VERTEX", "8", layer, "10", _num(x), "20", _num(y), "30", "0"])
        ents.extend(["0", "SEQEND", "8", layer])

    def text(pos, height_mm, string, angle, align, layer):
        p = np.asarray(pos) * k
        hj = {"left": 0, "center": 1, "right": 2}.get(align, 0)
        ents.extend(
            [
                "0",
                "TEXT",
                "8",
                layer,
                "10",
                _num(p[0]),
                "20",
                _num(p[1]),
                "30",
                "0",
                "40",
                _num(height_mm * text_scale),
                "1",
                _dxf_text(string),
                "50",
                _num(math.degrees(angle)),
            ]
        )
        if hj:
            ents.extend(["72", str(hj), "11", _num(p[0]), "21", _num(p[1]), "31", "0"])

    ltypes = {"hidden": "DASHED", "center": "DASHDOT"}
    for pl in d.polylines:
        poly(
            pl.points,
            lay(_layer_name(pl.layer, pl.style), ltypes.get(pl.style, "CONTINUOUS")),
            pl.closed,
        )
    for hz in d.hatches:
        layer = lay(_layer_name(hz.layer, "fill"))
        pts = np.concatenate(hz.loops)
        starts = np.cumsum([0] + [len(lp) for lp in hz.loops])
        tri, _ = triangulate(pts, starts.astype(np.int64), np.array([0, len(hz.loops)]))
        for a, b, c in tri:
            pa, pb, pc = pts[a] * k, pts[b] * k, pts[c] * k
            ents.extend(
                [
                    "0",
                    "SOLID",
                    "8",
                    layer,
                    "10",
                    _num(pa[0]),
                    "20",
                    _num(pa[1]),
                    "30",
                    "0",
                    "11",
                    _num(pb[0]),
                    "21",
                    _num(pb[1]),
                    "31",
                    "0",
                    "12",
                    _num(pc[0]),
                    "22",
                    _num(pc[1]),
                    "32",
                    "0",
                    "13",
                    _num(pc[0]),
                    "23",
                    _num(pc[1]),
                    "33",
                    "0",
                ]
            )
        for lp in hz.loops:
            poly(lp, lay(_layer_name(hz.layer, "thin")), closed=True)
    for t in d.texts:
        text(t.position, t.height, t.text, t.angle, t.align, lay(_layer_name(t.layer, "")))
    from .model import _dimension_paper

    for dim in d.dimensions:
        strokes, texts = _dimension_paper(
            to_paper(dim.p1), to_paper(dim.p2), dim.offset * s, dim.text
        )
        layer = lay(_layer_name(dim.layer, ""))
        for p, _, _ in strokes:
            poly(to_model(p), layer)
        for pos, hh, string, angle, align in texts:
            text(to_model(pos), hh, string, angle, align, layer)
    lo, hi = d.model_bounds()
    head = [
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
        "9",
        "$EXTMIN",
        "10",
        _num(lo[0] * k),
        "20",
        _num(lo[1] * k),
        "30",
        "0",
        "9",
        "$EXTMAX",
        "10",
        _num(hi[0] * k),
        "20",
        _num(hi[1] * k),
        "30",
        "0",
        "0",
        "ENDSEC",
    ]
    dash_len = 3.0 * text_scale
    tables = ["0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LTYPE", "70", "3"]
    for name, desc, pattern in (
        ("CONTINUOUS", "Solid line", ()),
        ("DASHED", "Dashed __ __ __", (dash_len, -dash_len / 2)),
        ("DASHDOT", "Dash dot __ . __", (dash_len * 2.6, -dash_len / 2, 0.0, -dash_len / 2)),
    ):
        tables += [
            "0",
            "LTYPE",
            "2",
            name,
            "70",
            "0",
            "3",
            desc,
            "72",
            "65",
            "73",
            str(len(pattern)),
            "40",
            _num(sum(abs(x) for x in pattern)),
        ]
        for x in pattern:
            tables += ["49", _num(x)]
    tables += [
        "0",
        "ENDTAB",
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        str(len(layers) + 1),
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
    ]
    colors = {"HIDDEN": 8, "SECTION": 7, "FILL": 9, "CENTER": 1, "THIN": 8}
    for name, ltype in sorted(layers.items()):
        color = colors.get(name.rsplit("-", 1)[-1], 7)
        tables += ["0", "LAYER", "2", name, "70", "0", "62", str(color), "6", ltype]
    tables += ["0", "ENDTAB", "0", "ENDSEC"]
    body = ["0", "SECTION", "2", "ENTITIES", *ents, "0", "ENDSEC", "0", "EOF"]
    return "\n".join(head + tables + body) + "\n"
