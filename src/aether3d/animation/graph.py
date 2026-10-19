"""Graph images of F-curves: value (with keys and handles), velocity, acceleration, jerk.

A visualization for people and for agents that can look at images -- the curves
themselves are edited through the API, scripts and MCP, never through a picture.
:func:`graph_svg` writes exact vectors; :func:`graph_png` rasterizes the same drawing
with Aether3D's own anti-aliased stroke rasterizer and stroke font (no external
library). Plots share the time axis; each curve has its colour; keys are squares
(numbered), handles thin lines ending in circles; velocity, acceleration and jerk
(0.20.0; ``plots`` chooses) are the curves' exact derivatives sampled densely (vertical
tangents are clipped and marked). With ``annotate`` (default), the exact analysis of
:mod:`~aether3d.animation.report` is drawn on top: a discontinuity of order ``k`` (value
C0, velocity C1, acceleration C2) is a dashed line on the plot of that order, labelled
``C0``/``C1``/``C2`` (grey when the step is intended: constant keys, jump pairs), and the
value's extrema are triangles labelled ``max``/``min``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import AnimationError
from .curves import FCurve

PALETTE = [
    (0.85, 0.25, 0.25),
    (0.20, 0.55, 0.90),
    (0.25, 0.70, 0.30),
    (0.90, 0.60, 0.10),
    (0.60, 0.35, 0.80),
    (0.10, 0.70, 0.70),
    (0.80, 0.30, 0.60),
    (0.45, 0.45, 0.45),
]
PLOTS = ("value", "velocity", "acceleration", "jerk")
DEFAULT_PLOTS = ("value", "velocity", "acceleration")
ORDER_PLOT = {0: "value", 1: "velocity", 2: "acceleration"}
MAX_CURVES = 32


def _nice(lo: float, hi: float, count: int = 5) -> list[float]:
    span = hi - lo
    if not span > 0:
        return [lo]
    step = 10 ** math.floor(math.log10(span / count))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= count:
            step *= m
            break
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step + 1e-9) + 1)]


def _layout(
    curves: dict[str, FCurve],
    start: float | None,
    end: float | None,
    plots: tuple[str, ...],
    samples: int,
) -> dict[str, Any]:
    if not curves or len(curves) > MAX_CURVES:
        raise AnimationError("INVALID_ARGUMENT", f"graph 1..{MAX_CURVES} curves", "curves")
    for p in plots:
        if p not in PLOTS:
            raise AnimationError(
                "INVALID_ARGUMENT", f"plots are some of {', '.join(PLOTS)}", "plots"
            )
    lo = min(c.range[0] for c in curves.values()) if start is None else float(start)
    hi = max(c.range[1] for c in curves.values()) if end is None else float(end)
    if not hi > lo:
        hi = lo + 1.0
    t = np.linspace(lo, hi, int(min(max(samples, 50), 20_000)))
    data = {}
    for name, c in curves.items():
        v, d, a, j = c.evaluate(t, order=3)
        data[name] = {"value": v, "velocity": d, "acceleration": a, "jerk": j}
    ranges = {}
    for p in plots:
        vals = np.concatenate([np.where(np.isfinite(data[n][p]), data[n][p], np.nan) for n in data])
        if p == "value":
            for c in curves.values():
                keys = c.times
                sel = (keys >= lo) & (keys <= hi)
                vals = np.concatenate(
                    [
                        vals,
                        c.values[sel],
                        (c.values + c.array("left_dv"))[sel],
                        (c.values + c.array("right_dv"))[sel],
                    ]
                )
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            a, b = -1.0, 1.0
        else:
            # Robust range for derivatives (spikes clipped and marked).
            a, b = (
                (float(finite.min()), float(finite.max()))
                if p == "value"
                else (float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5)))
            )
            if p != "value":
                a, b = min(a, 0.0), max(b, 0.0)
        if b - a < 1e-12:
            a, b = a - 1.0, b + 1.0
        pad = 0.08 * (b - a)
        ranges[p] = (a - pad, b + pad)
    return {"t": t, "data": data, "ranges": ranges, "range": (lo, hi)}


def _shapes(
    curves: dict[str, FCurve],
    layout: dict[str, Any],
    plots: tuple[str, ...],
    width: int,
    plot_h: int,
    title: str | None,
    annotate: bool = True,
) -> dict[str, Any]:
    """Device-space primitives: lines (x0, y0, x1, y1, rgb, width), boxes, texts."""
    left, right, top, gap = 64, 16, 28 if title else 12, 34
    lines: list[tuple] = []
    texts: list[tuple] = []
    marks: list[tuple] = []
    lo, hi = layout["range"]
    t = layout["t"]
    plot_w = width - left - right

    def x_of(tt):
        return left + (np.asarray(tt) - lo) / (hi - lo) * plot_w

    if title:
        texts.append((left, 18, title, 13, (0.1, 0.1, 0.1)))
    for pi, p in enumerate(plots):
        y0 = top + pi * (plot_h + gap)
        a, b = layout["ranges"][p]

        def y_of(v, y0=y0, a=a, b=b):
            return y0 + plot_h - (np.asarray(v) - a) / (b - a) * plot_h

        # frame, grid, labels
        for xx0, yy0, xx1, yy1 in (
            (left, y0, left + plot_w, y0),
            (left, y0 + plot_h, left + plot_w, y0 + plot_h),
            (left, y0, left, y0 + plot_h),
            (left + plot_w, y0, left + plot_w, y0 + plot_h),
        ):
            lines.append((xx0, yy0, xx1, yy1, (0.55, 0.55, 0.55), 0.8))
        for tick in _nice(a, b):
            yy = float(y_of(tick))
            lines.append((left, yy, left + plot_w, yy, (0.88, 0.88, 0.88), 0.6))
            texts.append((6, yy + 4, f"{tick:.4g}", 9, (0.35, 0.35, 0.35)))
        for tick in _nice(lo, hi, 8):
            xx = float(x_of(tick))
            lines.append((xx, y0, xx, y0 + plot_h, (0.92, 0.92, 0.92), 0.6))
            texts.append((xx - 8, y0 + plot_h + 14, f"{tick:.3g}", 9, (0.35, 0.35, 0.35)))
        if a < 0 < b:
            yy = float(y_of(0.0))
            lines.append((left, yy, left + plot_w, yy, (0.6, 0.6, 0.6), 0.9))
        texts.append((left + 4, y0 + 12, p, 10, (0.2, 0.2, 0.2)))
        for ci, (name, curve) in enumerate(curves.items()):
            color = PALETTE[ci % len(PALETTE)]
            vals = layout["data"][name][p]
            xs = x_of(t)
            ys = y_of(np.clip(np.where(np.isfinite(vals), vals, b), a, b))
            clipped = (vals < a) | (vals > b) | ~np.isfinite(vals)
            for i in range(len(t) - 1):
                lines.append(
                    (float(xs[i]), float(ys[i]), float(xs[i + 1]), float(ys[i + 1]), color, 1.6)
                )
            for i in np.flatnonzero(clipped)[
                :: max(1, int(np.count_nonzero(clipped) / 40) or 1)
            ].tolist():
                marks.append(("clip", float(xs[i]), float(ys[i]), color))
            if p == "value":
                kt, kv = curve.times, curve.values
                sel = (kt >= lo) & (kt <= hi)
                interp = curve.interpolation
                n = len(curve)
                for k in np.flatnonzero(sel).tolist():
                    kx, ky = float(x_of(kt[k])), float(y_of(kv[k]))
                    # Handles shape only Bezier segments: draw those that do.
                    sides = []
                    if k > 0 and interp[k - 1] == 2:
                        sides.append("left")
                    if k < n - 1 and interp[k] == 2:
                        sides.append("right")
                    for side in sides:
                        hx = float(x_of(kt[k] + curve.array(f"{side}_dt")[k]))
                        hy = float(y_of(kv[k] + curve.array(f"{side}_dv")[k]))
                        lines.append((kx, ky, hx, hy, (0.4, 0.4, 0.4), 0.8))
                        marks.append(("handle", hx, hy, color))
                    marks.append(("key", kx, ky, color))
                    if n <= 40:
                        texts.append((kx - 3, ky - 7, str(k), 8, (0.3, 0.3, 0.3)))
            if annotate:
                _annotations(curve, p, lo, hi, x_of, y_of, y0, plot_h, color, lines, texts, marks)
        if pi == 0:
            for ci, name in enumerate(curves):
                color = PALETTE[ci % len(PALETTE)]
                lx = left + plot_w - 180
                ly = y0 + 14 + 13 * ci
                lines.append((lx, ly - 3, lx + 16, ly - 3, color, 2.0))
                texts.append((lx + 20, ly, name[:28], 9, (0.15, 0.15, 0.15)))
    height = top + len(plots) * (plot_h + gap)
    return {"lines": lines, "texts": texts, "marks": marks, "width": width, "height": height}


def _annotations(curve, plot, lo, hi, x_of, y_of, y0, plot_h, color, lines, texts, marks):
    """Discontinuities (on the plot of their order) and extrema (on the value plot)."""
    from .report import continuity, extrema

    for item in continuity(curve):
        if ORDER_PLOT.get(item["order"]) != plot or not lo <= item["time"] <= hi:
            continue
        xx = float(x_of(item["time"]))
        ink = (0.6, 0.6, 0.6) if item.get("intended") else (0.85, 0.1, 0.1)
        for k in range(0, plot_h, 8):  # dashed
            lines.append((xx, y0 + k, xx, y0 + min(k + 4, plot_h), ink, 1.0))
        texts.append((xx + 3, y0 + plot_h - 6, f"C{item['order']}", 9, ink))
    if plot != "value":
        return
    for item in extrema(curve):
        if not lo <= item["time"] <= hi:
            continue
        xx, yy = float(x_of(item["time"])), float(y_of(item["value"]))
        marks.append(("max" if item["kind"] == "max" else "min", xx, yy, color))
        texts.append((xx + 4, yy + (-6 if item["kind"] == "max" else 12), item["kind"], 8,
                      (0.25, 0.25, 0.25)))  # fmt: skip


def graph_svg(
    curves: dict[str, FCurve],
    *,
    start: float | None = None,
    end: float | None = None,
    plots: tuple[str, ...] = DEFAULT_PLOTS,
    width: int = 900,
    plot_height: int = 200,
    samples: int = 600,
    title: str | None = None,
    annotate: bool = True,
) -> str:
    """An SVG document graphing ``curves`` (name -> curve)."""
    layout = _layout(curves, start, end, tuple(plots), samples)
    sh = _shapes(curves, layout, tuple(plots), int(width), int(plot_height), title, annotate)

    def rgb(c):
        return "rgb({},{},{})".format(*(int(round(255 * x)) for x in c))

    w_, h_ = sh["width"], sh["height"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_}" height="{h_}" '
        f'viewBox="0 0 {w_} {h_}">'
    ]
    out.append(f'<rect width="{sh["width"]}" height="{sh["height"]}" fill="white"/>')
    for x0, y0, x1, y1, c, w in sh["lines"]:
        out.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="{rgb(c)}" stroke-width="{w}"/>'
        )
    for kind, x, y, c in sh["marks"]:
        if kind == "key":
            out.append(
                f'<rect x="{x - 3.5:.2f}" y="{y - 3.5:.2f}" width="7" height="7" '
                f'fill="{rgb(c)}" stroke="black" stroke-width="0.6"/>'
            )
        elif kind == "handle":
            out.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="white" '
                f'stroke="{rgb(c)}" stroke-width="1"/>'
            )
        elif kind in ("max", "min"):
            dy = -4.5 if kind == "max" else 4.5
            out.append(
                f'<polygon points="{x - 4:.2f},{y - dy:.2f} {x + 4:.2f},{y - dy:.2f} '
                f'{x:.2f},{y + dy:.2f}" fill="none" stroke="{rgb(c)}" stroke-width="1.2"/>'
            )
        else:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{rgb(c)}"/>')
    for x, y, text, size, c in sh["texts"]:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
            f'font-size="{size}" fill="{rgb(c)}">{safe}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def graph_png(
    curves: dict[str, FCurve],
    *,
    start: float | None = None,
    end: float | None = None,
    plots: tuple[str, ...] = DEFAULT_PLOTS,
    width: int = 900,
    plot_height: int = 200,
    samples: int = 600,
    title: str | None = None,
    annotate: bool = True,
) -> np.ndarray:
    """The same graph as an RGB uint8 image, drawn by Aether3D (stroke rasterizer and
    stroke font)."""
    from ..cad.drawing.font import text_strokes
    from ..cad.drawing.raster import _stroke

    layout = _layout(curves, start, end, tuple(plots), samples)
    sh = _shapes(curves, layout, tuple(plots), int(width), int(plot_height), title, annotate)
    h, w = int(sh["height"]), int(sh["width"])
    image = np.ones((h, w, 3))
    groups: dict[tuple, list[tuple]] = {}
    for x0, y0, x1, y1, c, lw in sh["lines"]:
        groups.setdefault((c, lw), []).append((x0, y0, x1, y1))
    for kind, x, y, c in sh["marks"]:
        r = (
            3.0
            if kind == "key"
            else 2.4
            if kind == "handle"
            else 3.4
            if kind in ("max", "min")
            else 1.8
        )
        pts = [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)]
        for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
            groups.setdefault((c, 1.4 if kind == "key" else 0.9), []).append((*p0, *p1))
    for x, y, text, size, c in sh["texts"]:
        for s in text_strokes(text, (0.0, 0.0), size * 0.72):
            pts = s * np.array([1.0, -1.0]) + np.array([x, y])
            for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
                groups.setdefault((c, 0.55), []).append((p0[0], p0[1], p1[0], p1[1]))
    for (color, lw), segs in groups.items():
        ink = np.zeros((h, w))
        arr = np.array(segs, np.float64)
        # Split long segments (the rasterizer works on short pieces).
        pieces = []
        for x0, y0, x1, y1 in arr:
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 12.0) + 1)
            xs = np.linspace(x0, x1, n + 1)
            ys = np.linspace(y0, y1, n + 1)
            pieces.extend(np.stack([xs[:-1], ys[:-1], xs[1:], ys[1:]], axis=1))
        seg = np.array(pieces)
        _stroke(ink, seg, np.full(len(seg), lw / 2.0))
        cov = np.clip(ink, 0.0, 1.0)[..., None]
        image = image * (1.0 - cov) + np.asarray(color)[None, None, :] * cov
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
