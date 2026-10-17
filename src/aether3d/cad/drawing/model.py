"""Drawings: 2D linework, fills, text and dimensions in a view plane, and their output
as SVG, DXF and PNG.

Geometry is kept in *model* units (metres, in the view plane); the drawing's ``scale``
(paper length per model length, e.g. 1/100) and its sheet put it on paper. Line weights,
dash patterns and text heights are paper millimetres, as on a printed sheet.

Styles:

=========  ============================  =========================================
style      paper weight / pattern        used for
=========  ============================  =========================================
visible    0.50 mm continuous            edges and silhouettes in view
section    0.70 mm continuous            outlines of cut material
hidden     0.25 mm dashed 3 / 1.5        edges hidden behind material
center     0.18 mm dash-dot 8 / 2 / 1    axes, grids
thin       0.18 mm continuous            hatch boundaries, symbols, door swings
dim        0.18 mm continuous            dimension and extension lines
=========  ============================  =========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from ...errors import CadError

STYLES = {
    "visible": (0.50, ()),
    "section": (0.70, ()),
    "hidden": (0.25, (3.0, 1.5)),
    "center": (0.18, (8.0, 2.0, 1.0, 2.0)),
    "thin": (0.18, ()),
    "dim": (0.18, ()),
}
SHEETS = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "A2": (594.0, 420.0), "A1": (841.0, 594.0)}
SCALES = (
    Fraction(10),
    Fraction(5),
    Fraction(2),
    Fraction(1),
    Fraction(1, 2),
    Fraction(1, 5),
    Fraction(1, 10),
    Fraction(1, 20),
    Fraction(1, 25),
    Fraction(1, 50),
    Fraction(1, 100),
    Fraction(1, 200),
    Fraction(1, 250),
    Fraction(1, 500),
    Fraction(1, 1000),
)
MAX_ENTITIES = 500_000
MARGIN = 15.0  # paper mm around the drawing area
TITLE_H = 18.0  # paper mm: the title strip at the bottom


@dataclass
class Polyline:
    points: np.ndarray  # (N, 2) model units in the view plane
    style: str = "visible"
    layer: str = "0"
    closed: bool = False


@dataclass
class Hatch:
    loops: list[np.ndarray]  # outer first, then holes (model units)
    layer: str = "section"
    pattern: str = "solid"  # solid | diagonal
    gray: float = 0.75  # fill gray level (0 black .. 1 white)


@dataclass
class Text:
    position: np.ndarray  # model units
    text: str
    height: float = 2.5  # paper mm (cap height)
    angle: float = 0.0  # radians
    align: str = "left"  # left | center | right
    layer: str = "text"


@dataclass
class Dimension:
    p1: np.ndarray
    p2: np.ndarray
    offset: float  # model units, perpendicular to p1 -> p2 (left positive)
    text: str
    layer: str = "dimensions"


@dataclass
class Drawing:
    title: str
    view: str
    scale: Fraction | None = None  # paper / model; None: chosen to fit the sheet
    sheet: str = "A3"
    unit_label: str = "m"
    polylines: list[Polyline] = field(default_factory=list)
    hatches: list[Hatch] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    # -- building ---------------------------------------------------------------------

    def _count(self) -> int:
        return len(self.polylines) + len(self.hatches) + len(self.texts) + len(self.dimensions)

    def _check(self) -> None:
        if self._count() >= MAX_ENTITIES:
            raise CadError("LIMIT_EXCEEDED", f"a drawing holds at most {MAX_ENTITIES} entities")

    def line(self, points, style: str = "visible", layer: str = "0", closed: bool = False):
        self._check()
        if style not in STYLES:
            raise CadError("INVALID_VALUE", f"unknown line style {style!r}", hint=", ".join(STYLES))
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 2 and np.all(np.isfinite(pts)):
            self.polylines.append(Polyline(pts, style, layer, closed))

    def hatch(self, loops, layer: str = "section", pattern: str = "solid", gray: float = 0.75):
        self._check()
        loops = [np.asarray(lp, dtype=np.float64).reshape(-1, 2) for lp in loops]
        loops = [lp for lp in loops if len(lp) >= 3]
        if loops:
            self.hatches.append(Hatch(loops, layer, pattern, gray))

    def text(
        self,
        position,
        text: str,
        height: float = 2.5,
        angle: float = 0.0,
        align: str = "left",
        layer: str = "text",
    ):
        self._check()
        self.texts.append(
            Text(
                np.asarray(position, dtype=np.float64), str(text)[:200], height, angle, align, layer
            )
        )

    def dimension(self, p1, p2, offset: float, text: str, layer: str = "dimensions"):
        self._check()
        a, b = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
        if np.linalg.norm(b - a) > 0:
            self.dimensions.append(Dimension(a, b, float(offset), text, layer))

    # -- extents and scale ----------------------------------------------------------------

    def model_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = [p.points for p in self.polylines]
        pts += [lp for h in self.hatches for lp in h.loops]
        pts += [np.array([d.p1, d.p2]) for d in self.dimensions]
        if not pts:
            return np.zeros(2), np.ones(2)
        allp = np.concatenate(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        # Room for dimensions placed outside the geometry.
        for d in self.dimensions:
            u = (d.p2 - d.p1) / np.linalg.norm(d.p2 - d.p1)
            n = np.array([-u[1], u[0]])
            for p in (d.p1 + n * d.offset, d.p2 + n * d.offset):
                lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        return lo, hi

    def paper_scale(self) -> Fraction:
        if self.scale is not None:
            return self.scale
        w, h = SHEETS[self.sheet]
        avail = np.array([w - 2 * MARGIN, h - 2 * MARGIN - TITLE_H]) * 0.92
        lo, hi = self.model_bounds()
        ext = np.maximum(hi - lo, 1e-9) * 1000.0  # model mm
        for s in SCALES:
            if np.all(ext * float(s) <= avail):
                return s
        return SCALES[-1]

    def scale_label(self) -> str:
        s = self.paper_scale()
        return f"{s.numerator}:{s.denominator}" if s >= 1 else f"1:{s.denominator // s.numerator}"

    # -- paper primitives -------------------------------------------------------------------

    def transform(self):
        """(model -> paper, paper -> model, paper mm per model metre)."""
        w, h = SHEETS[self.sheet]
        s = float(self.paper_scale()) * 1000.0
        lo, hi = self.model_bounds()
        cm = 0.5 * (lo + hi)
        cp = np.array([w / 2, MARGIN + TITLE_H + (h - 2 * MARGIN - TITLE_H) / 2])

        def to_paper(p):
            return (np.asarray(p, dtype=np.float64) - cm) * s + cp

        def to_model(p):
            return (np.asarray(p, dtype=np.float64) - cp) / s + cm

        return to_paper, to_model, s

    def paper(self) -> dict[str, list]:
        """Everything in paper millimetres (y up): ``strokes`` (points, weight, dashes),
        ``fills`` (loops, gray, pattern), ``texts`` (position, height, text, angle, align)."""
        from .raster import hatch_lines

        w, h = SHEETS[self.sheet]
        tp, _, s = self.transform()
        strokes, fills, texts = [], [], []
        for hz in self.hatches:
            loops = [tp(lp) for lp in hz.loops]
            if hz.pattern == "solid":
                fills.append((loops, hz.gray, hz.pattern))
            else:
                strokes += [(seg, 0.13, ()) for seg in hatch_lines(loops, 1.5)]
            for lp in loops:
                strokes.append((np.vstack([lp, lp[:1]]), *STYLES["thin"]))
        for pl in self.polylines:
            pts = tp(pl.points)
            if pl.closed:
                pts = np.vstack([pts, pts[:1]])
            strokes.append((pts, *STYLES[pl.style]))
        for t in self.texts:
            texts.append((tp(t.position), t.height, t.text, t.angle, t.align))
        for d in self.dimensions:
            st, tx = _dimension_paper(tp(d.p1), tp(d.p2), d.offset * s, d.text)
            strokes += st
            texts += tx
        # Sheet border and title strip.
        border = np.array(
            [
                [MARGIN, MARGIN],
                [w - MARGIN, MARGIN],
                [w - MARGIN, h - MARGIN],
                [MARGIN, h - MARGIN],
                [MARGIN, MARGIN],
            ]
        )
        strokes.append((border, 0.5, ()))
        strokes.append(
            (np.array([[MARGIN, MARGIN + TITLE_H], [w - MARGIN, MARGIN + TITLE_H]]), 0.35, ())
        )
        texts.append((np.array([MARGIN + 4, MARGIN + 10]), 5.0, self.title, 0.0, "left"))
        texts.append(
            (
                np.array([MARGIN + 4, MARGIN + 3.5]),
                2.5,
                f"{self.view}   scale {self.scale_label()} ({self.sheet})   "
                f"dimensions in {self.unit_label}",
                0.0,
                "left",
            )
        )
        texts.append((np.array([w - MARGIN - 4, MARGIN + 3.5]), 2.5, "Aether3D", 0.0, "right"))
        return {"strokes": strokes, "fills": fills, "texts": texts, "size": (w, h)}

    # -- outputs ----------------------------------------------------------------------------

    def to_svg(self) -> str:
        from .export import to_svg

        return to_svg(self)

    def to_dxf(self, unit: str = "mm") -> str:
        from .export import to_dxf

        return to_dxf(self, unit)

    def to_png(self, dpi: float = 150.0) -> np.ndarray:
        from .raster import rasterize

        return rasterize(self.paper(), dpi)

    def summary(self) -> dict[str, Any]:
        lo, hi = self.model_bounds()
        by_style: dict[str, int] = {}
        for p in self.polylines:
            by_style[p.style] = by_style.get(p.style, 0) + 1
        return {
            "title": self.title,
            "view": self.view,
            "scale": self.scale_label(),
            "sheet": self.sheet,
            "extent_m": (hi - lo).tolist(),
            "polylines": by_style,
            "hatches": len(self.hatches),
            "texts": len(self.texts),
            "dimensions": [
                {"text": d.text, "length_m": float(np.linalg.norm(d.p2 - d.p1))}
                for d in self.dimensions
            ],
            "notes": list(self.notes),
            **self.info,
        }


ARROW = 2.5  # paper mm
EXT_GAP = 1.0
EXT_OVER = 1.5


def _dimension_paper(p1, p2, offset, text):
    """Dimension line, extension lines, arrow heads (filled as strokes) and text."""
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    n = np.array([-u[1], u[0]])
    sgn = 1.0 if offset >= 0 else -1.0
    a, b = p1 + n * offset, p2 + n * offset
    strokes = [
        (np.array([p1 + n * sgn * EXT_GAP, a + n * sgn * EXT_OVER]), *STYLES["dim"]),
        (np.array([p2 + n * sgn * EXT_GAP, b + n * sgn * EXT_OVER]), *STYLES["dim"]),
        (np.array([a, b]), *STYLES["dim"]),
    ]
    for tip, d in ((a, u), (b, -u)):
        # A closed arrow head: drawn as a filled wedge of strokes.
        base = tip + d * ARROW
        half = n * ARROW * 0.18
        for k in np.linspace(-1, 1, 5):
            strokes.append((np.array([tip, base + half * k]), 0.18, ()))
    angle = math.atan2(u[1], u[0])
    up = n
    if angle > math.pi / 2 + 1e-9 or angle <= -math.pi / 2 + 1e-9:
        angle += math.pi  # keep text readable (left to right, or bottom to top)
        up = -n
    away = n * sgn  # the side of the dimension line away from the geometry
    lift = 1.0 if float(up @ away) > 0 else 1.0 + 2.5
    mid = 0.5 * (a + b) + away * lift
    texts = [(mid, 2.5, text, angle, "center")]
    return strokes, texts
