"""Authored terrain features: shapes, distances, masks and how each type changes the world.

A feature (see :data:`aether3d.worlds.definition.FEATURE_TYPES`) has a *shape* -- a circle,
rectangle, polygon or polyline with a width -- and a *falloff*: its *mask* is 1 inside the
shape and falls smoothly (a quintic step of the distance) to 0 at ``falloff`` metres
outside. Shape-stage features change the landform before the regional stage (erosion and
rivers respond); final-stage features apply afterwards, in definition order.

Distances use only correctly rounded operations (``sqrt`` included), like the noise, so
features apply identically on every platform. Each compiled feature knows its bounding
box (shape, width and falloff), and points outside it are not visited, so a feature costs
nothing away from where it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import noise as nz
from .definition import FINAL_STAGE, SHAPE_STAGE


def segment_distance(
    px: np.ndarray, py: np.ndarray, pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance from points to a polyline ``pts`` (K, 2+), the segment of the closest point
    and the parameter ``t`` in [0, 1] along that segment."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    n = px.shape[0]
    best = np.full(n, np.inf)
    seg = np.zeros(n, dtype=np.int64)
    tt = np.zeros(n)
    a = pts[:-1, :2]
    b = pts[1:, :2]
    for k in range(a.shape[0]):
        ax, ay = a[k]
        dx, dy = b[k, 0] - ax, b[k, 1] - ay
        ll = dx * dx + dy * dy
        rx = px - ax
        ry = py - ay
        t = np.clip((rx * dx + ry * dy) / ll, 0.0, 1.0) if ll > 0.0 else np.zeros(n)
        qx = rx - t * dx
        qy = ry - t * dy
        d2 = qx * qx + qy * qy
        closer = d2 < best
        best = np.where(closer, d2, best)
        seg = np.where(closer, k, seg)
        tt = np.where(closer, t, tt)
    return np.sqrt(best), seg, tt


def polygon_inside(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Whether points lie inside a polygon (even-odd crossings of a ray toward +x)."""
    inside = np.zeros(px.shape, dtype=bool)
    m = poly.shape[0]
    for k in range(m):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % m]
        crosses = (y0 > py) != (y1 > py)
        if not np.any(crosses):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (px < xc)
    return inside


def shape_distance(shape: dict, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Signed distance to a shape (negative inside; a line's inside is its width)."""
    if "circle" in shape:
        cx, cy, r = shape["circle"]
        dx = px - cx
        dy = py - cy
        return np.sqrt(dx * dx + dy * dy) - r
    if "rect" in shape:
        x0, y0, x1, y1 = shape["rect"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        qx = np.abs(px - cx) - hx
        qy = np.abs(py - cy) - hy
        ox = np.maximum(qx, 0.0)
        oy = np.maximum(qy, 0.0)
        return np.sqrt(ox * ox + oy * oy) + np.minimum(np.maximum(qx, qy), 0.0)
    if "polygon" in shape:
        poly = np.asarray(shape["polygon"], dtype=np.float64)
        closed = np.vstack([poly, poly[:1]])
        d, _, _ = segment_distance(px, py, closed)
        return np.where(polygon_inside(px, py, poly), -d, d)
    pts = np.asarray(shape["line"], dtype=np.float64)
    d, _, _ = segment_distance(px, py, pts)
    return d - shape["width"] / 2.0


def shape_bounds(shape: dict) -> tuple[float, float, float, float]:
    if "circle" in shape:
        cx, cy, r = shape["circle"]
        return cx - r, cy - r, cx + r, cy + r
    if "rect" in shape:
        return tuple(shape["rect"])  # type: ignore[return-value]
    pts = np.asarray(shape.get("polygon") or shape["line"], dtype=np.float64)
    pad = shape.get("width", 0.0) / 2.0
    return (
        float(pts[:, 0].min()) - pad,
        float(pts[:, 1].min()) - pad,
        float(pts[:, 0].max()) + pad,
        float(pts[:, 1].max()) + pad,
    )


def falloff_mask(dist: np.ndarray, falloff: float) -> np.ndarray:
    """1 inside (``dist <= 0``), a quintic step down to 0 at ``falloff`` outside."""
    if falloff <= 0.0:
        return (dist <= 0.0).astype(np.float64)
    return 1.0 - nz.smootherstep(0.0, falloff, dist)


@dataclass
class Compiled:
    """A feature ready to evaluate: its definition, stage and padded bounding box."""

    spec: dict
    stage: str
    box: tuple[float, float, float, float]
    data: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.spec["type"]

    def touches(self, box: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = self.box
        return not (x1 < box[0] or x0 > box[2] or y1 < box[1] or y0 > box[3])


def _reach(spec: dict) -> float:
    kind = spec["type"]
    if kind in ("mountain",):
        return 0.0
    if kind == "cave":
        return spec["radius"] * 2.0 + 8.0
    return float(spec.get("falloff", 0.0) or 0.0)


def compile_features(features: list[dict]) -> list[Compiled]:
    out = []
    for spec in features:
        if not spec.get("enabled", True):
            continue
        x0, y0, x1, y1 = shape_bounds(spec["shape"])
        r = _reach(spec)
        stage = "shape" if spec["type"] in SHAPE_STAGE else "final"
        assert spec["type"] in SHAPE_STAGE or spec["type"] in FINAL_STAGE
        c = Compiled(spec, stage, (x0 - r, y0 - r, x1 + r, y1 + r))
        if "line" in spec["shape"]:
            c.data["line"] = np.asarray([p[:2] for p in spec["shape"]["line"]], dtype=np.float64)
            if all(len(p) == 3 for p in spec["shape"]["line"]):
                c.data["z"] = np.asarray([p[2] for p in spec["shape"]["line"]], dtype=np.float64)
        out.append(c)
    return out


def box_of(px: np.ndarray, py: np.ndarray) -> tuple[float, float, float, float]:
    return (float(px.min()), float(py.min()), float(px.max()), float(py.max()))


def _line_height(c: Compiled, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance to a feature's line and the height interpolated along it at the closest
    point (its points' z, or ``nan`` without them)."""
    pts = c.data["line"]
    d, seg, t = segment_distance(px, py, pts)
    z = c.data.get("z")
    if z is None:
        return d, np.full(d.shape, np.nan)
    return d, z[seg] + t * (z[seg + 1] - z[seg])


def apply_shape(
    features: list[Compiled],
    px: np.ndarray,
    py: np.ndarray,
    h: np.ndarray,
    sea_level: float,
    seed: int,
) -> np.ndarray:
    """The shape-stage features applied to heights ``h`` (a new array)."""
    if not features:
        return h
    box = box_of(px, py)
    h = h.copy()
    for c in features:
        if c.stage != "shape" or not c.touches(box):
            continue
        s = c.spec
        kind = s["type"]
        if kind == "mountain":
            cx, cy, r = s["shape"]["circle"]
            dx, dy = px - cx, py - cy
            t = np.sqrt(dx * dx + dy * dy) / r
            inside = t < 1.0
            if not np.any(inside):
                continue
            bell = 1.0 - nz.smootherstep(0.0, 1.0, t)
            sharp = s["sharpness"]
            peak = bell * bell * (3.0 - 2.0 * bell)  # steeper flanks
            rough = nz.ridged2(
                nz.derive(seed, "feature", s["id"]), px, py, octaves=6, frequency=3.0 / r
            )
            prof = (1.0 - sharp) * bell + sharp * peak * (0.65 + 0.7 * rough)
            h += s["height"] * prof
            continue
        dist = shape_distance(s["shape"], px, py)
        if kind in ("raise", "lower"):
            mask = falloff_mask(dist, s["falloff"])
            sign = 1.0 if kind == "raise" else -1.0
            h += sign * s["amount"] * mask
        elif kind == "scale":
            mask = falloff_mask(dist, s["falloff"])
            base = sea_level if s["base"] is None else s["base"]
            h = base + (h - base) * (1.0 + (s["factor"] - 1.0) * mask)
        elif kind == "ridge":
            d = np.maximum(dist, 0.0)
            prof = 1.0 - nz.smootherstep(0.0, s["falloff"], d)
            sharp = s["sharpness"]
            prof = (1.0 - sharp) * prof + sharp * prof * prof * prof
            h += s["height"] * prof
        elif kind == "valley":
            d = np.maximum(dist, 0.0)
            prof = 1.0 - nz.smootherstep(0.0, s["falloff"], d)
            h -= s["depth"] * prof
    return h


def apply_final(
    features: list[Compiled],
    px: np.ndarray,
    py: np.ndarray,
    h: np.ndarray,
    detail: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """The final-stage height features applied to ``h``: flatten, pad, road, lake, river.

    Returns the new heights and per-point extras: ``water`` (authored water level, ``nan``
    where none), ``road`` (road mask 0..1) and ``detail`` (a factor on the detail noise
    from ``smooth`` features, already folded into ``h`` when ``detail`` is given)."""
    extras: dict[str, Any] = {}
    if not features:
        return h, extras
    box = box_of(px, py)
    h = h.copy()
    for c in features:
        if c.stage != "final" or not c.touches(box):
            continue
        s = c.spec
        kind = s["type"]
        if kind in ("biome", "clear", "cave"):
            continue
        if kind == "smooth":
            dist = shape_distance(s["shape"], px, py)
            mask = falloff_mask(dist, s["falloff"])
            if detail is not None:
                h -= detail * (1.0 - s["factor"]) * mask
            f = extras.setdefault("detail", np.ones(h.shape))
            f *= 1.0 - (1.0 - s["factor"]) * mask
            continue
        if kind in ("flatten", "pad"):
            dist = shape_distance(s["shape"], px, py)
            mask = falloff_mask(dist, s["falloff"])
            target = s["height"]
            if target is None:
                continue  # (an unresolved automatic height: resolved when the feature is added)
            h += mask * (target - h)
        elif kind == "road":
            d, z = _line_height(c, px, py)
            half = s["shape"]["width"] / 2.0
            mask = falloff_mask(d - half, s["falloff"])
            target = np.where(np.isnan(z), h, z)
            h += mask * (target - h)
            road = extras.setdefault("road", np.zeros(h.shape))
            np.maximum(road, falloff_mask(d - half, 0.6), out=road)
        elif kind == "lake":
            dist = shape_distance(s["shape"], px, py)
            level = s["level"]
            inside = np.clip(-dist / max(s["falloff"], 1e-9), 0.0, 1.0)
            bowl = level - s["depth"] * (inside * inside * (3.0 - 2.0 * inside))
            shore = falloff_mask(dist, s["falloff"])
            target = np.where(dist <= 0.0, np.minimum(h, bowl), h)
            h += shore * (target - h) * (dist <= 0.0)
            water = extras.setdefault("water", np.full(h.shape, np.nan))
            water[:] = np.where(dist <= 0.0, np.fmax(water, level), water)
        elif kind == "river":
            d, z = _line_height(c, px, py)
            half = s["shape"]["width"] / 2.0
            level = z if s["level"] is None else _profile(s["level"], c, px, py)
            if np.all(np.isnan(level)):
                continue
            u = np.clip(d / half, 0.0, 1.0)
            bed = level - s["depth"] * (1.0 - u * u)
            inside = d < half
            h = np.where(inside, np.minimum(h, bed), h)
            water = extras.setdefault("water", np.full(h.shape, np.nan))
            water[:] = np.where(inside, np.fmax(water, level), water)
    return h, extras


def _profile(levels: list[float], c: Compiled, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Water levels given per line point, interpolated at the closest point."""
    pts = c.data["line"]
    d, seg, t = segment_distance(px, py, pts)
    z = np.asarray(levels, dtype=np.float64)
    if z.shape[0] != pts.shape[0]:
        return np.full(d.shape, np.nan)
    return z[seg] + t * (z[seg + 1] - z[seg])


def biome_overrides(
    features: list[Compiled], px: np.ndarray, py: np.ndarray, weights: np.ndarray, names: list[str]
) -> np.ndarray:
    """Biome weights after ``biome`` features: blended toward their biome by
    ``strength x mask`` (rows still sum to 1)."""
    box = box_of(px, py)
    for c in features:
        s = c.spec
        if s["type"] != "biome" or not c.touches(box):
            continue
        k = names.index(s["biome"])
        mask = falloff_mask(shape_distance(s["shape"], px, py), s["falloff"]) * s["strength"]
        weights = weights * (1.0 - mask)[:, None]
        weights[:, k] += mask
    return weights


def clear_mask(features: list[Compiled], px: np.ndarray, py: np.ndarray, what: str) -> np.ndarray:
    """1 where ``clear`` features remove ``what`` (``trees``, ``plants``, ``rocks``), else 0;
    pads, roads, lakes and authored rivers clear everything they cover."""
    out = np.zeros(px.shape)
    if not features:
        return out
    box = box_of(px, py)
    for c in features:
        s = c.spec
        if not c.touches(box):
            continue
        kind = s["type"]
        if kind == "clear" and (what in s["what"] or "all" in s["what"]):
            m = falloff_mask(shape_distance(s["shape"], px, py), s["falloff"])
        elif kind in ("pad", "lake"):
            m = (shape_distance(s["shape"], px, py) <= 1.0).astype(np.float64)
        elif kind in ("road", "river"):
            m = (shape_distance(s["shape"], px, py) <= 1.5).astype(np.float64)
        else:
            continue
        np.maximum(out, m, out=out)
    return out
