"""Atlas packing: charts placed into one texture, without overlaps, with padding.

1. **Orientation.** Each chart is turned to its minimum-area bounding rectangle (rotating
   calipers over its convex hull), unless ``rotate=False``.
2. **Footprints.** At a candidate texel density (texels per world unit) every chart is
   rasterized conservatively (every texel its triangles touch, separating-axis test) and
   dilated by ``ceil(padding / 2)`` texels, so islands end up at least ``padding`` texels
   apart and away from the atlas border. A footprint is summarized per column by its
   lowest and highest occupied rows.
3. **Placement** (Levy et al. 2002, "Tetris"): charts, largest footprint first, drop onto
   the *horizon* -- per atlas column, the lowest free row above everything placed. For
   every column and orientation (0, 90, 180 and 270 degrees with ``rotate``) the chart
   rests where its lowest texel first touches the horizon; the placement with the lowest
   resulting peak wins, then the least area wasted below the chart, then the leftmost
   column, then the first orientation. All of it is integer arithmetic.
4. **Density.** With ``texels_per_unit`` given, that density is used (``ATLAS_FULL`` if the
   charts do not fit). Otherwise the largest density whose packing fits is found by
   bisection.

The placement search runs in ``aether3d-native`` when available; the NumPy code makes
the same integer decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import _native
from ..errors import TexturingError

ORIENTATIONS = 4
SLACK = 1e-3
"""Footprints grow by this many texels, so float32 UVs never round outside them."""
MAX_RESOLUTION = 16384


@dataclass
class PackResult:
    uv: np.ndarray  # (N, 2) packed UVs in [0, 1]
    texels_per_unit: float
    resolution: tuple[int, int]
    utilization: float  # fraction of the atlas covered by island texels (unpadded)
    peak: int  # rows used
    rotations: np.ndarray  # (C,) quarter turns applied after the min-area rotation
    placements: np.ndarray  # (C, 2) texel offsets


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (Andrew's monotone chain), counter-clockwise, no duplicates."""
    p = np.unique(points, axis=0)
    if p.shape[0] <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def chain(pts):
        out: list = []
        for q in pts:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower = chain(p)
    upper = chain(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_angle(points: np.ndarray) -> float:
    """The rotation (radians) giving the convex hull's minimum-area bounding rectangle,
    wider than tall."""
    hull = _hull(points)
    if hull.shape[0] < 3:
        if hull.shape[0] == 2:
            d = hull[1] - hull[0]
            return -math.atan2(d[1], d[0])
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2), 12))
    best = (np.inf, 0.0, 0.0, 0.0)
    for a in angles.tolist():
        c, s = math.cos(-a), math.sin(-a)
        x = hull[:, 0] * c - hull[:, 1] * s
        y = hull[:, 0] * s + hull[:, 1] * c
        w, h = float(x.max() - x.min()), float(y.max() - y.min())
        area = w * h
        if area < best[0] - 1e-12 * max(area, 1e-300):
            best = (area, -a, w, h)
    angle, w, h = best[1], best[2], best[3]
    if h > w:
        angle += math.pi / 2
    return angle


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([points[:, 0] * c - points[:, 1] * s, points[:, 0] * s + points[:, 1] * c], 1)


# --------------------------------------------------------------------------- #
# Footprints
# --------------------------------------------------------------------------- #


def footprint(tri_uv: np.ndarray, dilation: int) -> tuple[np.ndarray, np.ndarray]:
    """The conservative texel footprint of triangles (``(F, 3, 2)`` in texel units) dilated
    by ``dilation``: ``(mask (h, w) bool, origin)`` with ``origin`` the texel coordinates of
    the mask's corner. Each triangle covers, in every texel row it meets, the texels between
    the extremes of its intersection with the row's band (exact for convex polygons)."""
    t = np.asarray(tri_uv, dtype=np.float64)
    lo = np.floor(t.reshape(-1, 2).min(axis=0) - SLACK).astype(np.int64)
    hi = np.floor(t.reshape(-1, 2).max(axis=0) + SLACK).astype(np.int64)
    w, h = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
    rel = t - lo
    y0 = np.floor(rel[:, :, 1].min(axis=1) - SLACK).astype(np.int64)
    y1 = np.floor(rel[:, :, 1].max(axis=1) + SLACK).astype(np.int64)
    rows = y1 - y0 + 1
    tri = np.repeat(np.arange(t.shape[0]), rows)
    row = y0[tri] + (np.arange(int(rows.sum())) - np.repeat(np.cumsum(rows) - rows, rows))
    xs_lo = np.full(row.shape[0], np.inf)
    xs_hi = np.full(row.shape[0], -np.inf)
    band_lo = row.astype(np.float64) - SLACK
    band_hi = row.astype(np.float64) + 1.0 + SLACK
    v = rel[tri]  # (R, 3, 2)
    for k in range(3):
        px, py = v[:, k, 0], v[:, k, 1]
        inside = (py >= band_lo) & (py <= band_hi)
        xs_lo = np.where(inside, np.minimum(xs_lo, px), xs_lo)
        xs_hi = np.where(inside, np.maximum(xs_hi, px), xs_hi)
        qx, qy = v[:, (k + 1) % 3, 0], v[:, (k + 1) % 3, 1]
        dy = qy - py
        for line in (band_lo, band_hi):
            with np.errstate(divide="ignore", invalid="ignore"):
                s_ = (line - py) / dy
                x = px + s_ * (qx - px)  # NaN where dy == 0: masked by ok
            ok = (dy != 0) & (s_ >= 0) & (s_ <= 1)
            xs_lo = np.where(ok, np.minimum(xs_lo, x), xs_lo)
            xs_hi = np.where(ok, np.maximum(xs_hi, x), xs_hi)
    valid = xs_lo <= xs_hi
    c0 = np.clip(np.floor(xs_lo[valid] - SLACK).astype(np.int64), 0, w - 1)
    c1 = np.clip(np.floor(xs_hi[valid] + SLACK).astype(np.int64), 0, w - 1)
    r = np.clip(row[valid], 0, h - 1)
    d = dilation
    diff = np.zeros((h + 2 * d, w + 2 * d + 1), dtype=np.int32)
    np.add.at(diff, (r + d, c0 + d), 1)
    np.add.at(diff, (r + d, c1 + d + 1), -1)
    mask = np.cumsum(diff, axis=1)[:, :-1] > 0
    if d:
        mask = _dilate(mask, d)
    return mask, lo - d


def oriented(mask: np.ndarray, o: int) -> np.ndarray:
    """The footprint of the chart turned ``o`` quarter turns (see :func:`orient_points`)."""
    if o == 0:
        return mask
    if o == 1:
        return mask.T[:, ::-1]
    if o == 2:
        return mask[::-1, ::-1]
    return mask.T[::-1, :]


def orient_points(p: np.ndarray, o: int, shape: tuple[int, int]) -> np.ndarray:
    """Points relative to a mask's corner, turned ``o`` quarter turns counter-clockwise
    within the mask's box (``shape``: rows, columns), consistently with :func:`oriented`."""
    h, w = shape
    x, y = p[..., 0], p[..., 1]
    if o == 0:
        return p.copy()
    if o == 1:
        return np.stack([h - y, x], axis=-1)
    if o == 2:
        return np.stack([w - x, h - y], axis=-1)
    return np.stack([y, w - x], axis=-1)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        g2 = grown.copy()
        g2[:, 1:] |= grown[:, :-1]
        g2[:, :-1] |= grown[:, 1:]
        out = g2
    return out


def profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column: the lowest occupied row and one past the highest (``-1`` for empty
    columns), rows counted upward from the mask's bottom (row 0 of ``mask``)."""
    occ = mask.any(axis=0)
    h = mask.shape[0]
    lowest = np.where(occ, np.argmax(mask, axis=0), -1)
    highest = np.where(occ, h - np.argmax(mask[::-1], axis=0), -1)
    return lowest.astype(np.int64), highest.astype(np.int64)


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


def place(
    width: int, bottoms: list[np.ndarray], tops: list[np.ndarray], order: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Horizon placement of charts in ``order``; chart ``c``'s orientation ``o`` has
    profiles ``bottoms[c * 4 + o]`` / ``tops[...]`` (empty arrays: orientation unused).
    Returns ``(orientation, (x, y))`` per chart and the peak."""
    count = len(bottoms) // ORIENTATIONS
    native = _native.module()
    if native is not None and hasattr(native, "pack_horizon"):
        lengths = np.array([b.shape[0] for b in bottoms], dtype=np.int64)
        offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        bot = np.concatenate(bottoms) if bottoms else np.zeros(0, dtype=np.int64)
        top = np.concatenate(tops) if tops else np.zeros(0, dtype=np.int64)
        orient = np.zeros(count, dtype=np.int64)
        pos = np.zeros((count, 2), dtype=np.int64)
        try:
            peak = native.pack_horizon(
                width,
                np.ascontiguousarray(bot, dtype=np.int64),
                np.ascontiguousarray(top, dtype=np.int64),
                offsets,
                np.ascontiguousarray(order, dtype=np.int64),
                orient,
                pos,
            )
        except ValueError as exc:
            if str(exc).startswith("ATLAS_FULL"):
                raise TexturingError("ATLAS_FULL", "a chart is wider than the atlas") from None
            raise
        return orient, pos, int(peak)
    return _place_numpy(width, bottoms, tops, order)


def _place_numpy(width, bottoms, tops, order):
    count = len(bottoms) // ORIENTATIONS
    horizon = np.zeros(width, dtype=np.int64)
    orient = np.zeros(count, dtype=np.int64)
    pos = np.zeros((count, 2), dtype=np.int64)
    peak = 0
    big = np.int64(1) << 60
    for c in order.tolist():
        best = None  # (peak, waste, x, o, y)
        for o in range(ORIENTATIONS):
            bot = bottoms[c * ORIENTATIONS + o]
            top = tops[c * ORIENTATIONS + o]
            w = bot.shape[0]
            if w == 0 or w > width:
                continue
            occ = bot >= 0
            win = np.lib.stride_tricks.sliding_window_view(horizon, w)  # (width - w + 1, w)
            need = np.where(occ, win - bot, -big)
            y = np.maximum(need.max(axis=1), 0)
            waste = np.where(occ, y[:, None] + bot - win, 0).sum(axis=1)
            tmax = int(top.max())
            pk = np.maximum(y + tmax, peak)
            # Lexicographic minimum: peak, waste, x.
            key_pk = pk.min()
            cand = np.flatnonzero(pk == key_pk)
            wmin = waste[cand].min()
            cand = cand[waste[cand] == wmin]
            x = int(cand[0])
            entry = (int(key_pk), int(wmin), x, o, int(y[x]))
            if best is None or entry[:4] < best[:4]:
                best = entry
        if best is None:
            raise TexturingError("ATLAS_FULL", "a chart is wider than the atlas")
        _, _, x, o, y = best
        bot = bottoms[c * ORIENTATIONS + o]
        top = tops[c * ORIENTATIONS + o]
        occ = bot >= 0
        seg = horizon[x : x + bot.shape[0]]
        seg[occ] = np.maximum(seg[occ], y + top[occ])
        peak = max(peak, y + int(top.max()))
        orient[c] = o
        pos[c] = (x, y)
    return orient, pos, peak


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def pack(
    chart_tri_uv: list[np.ndarray],
    resolution: tuple[int, int],
    *,
    padding: int = 4,
    rotate: bool = True,
    texels_per_unit: float | None = None,
    should_stop=None,
) -> tuple[list[np.ndarray], PackResult]:
    """Pack charts (each given by its triangles' UVs ``(F_c, 3, 2)`` in world units) into an
    atlas of ``resolution`` texels; returns each chart's transform as a function applied to
    its points and the pack result. See the module docstring."""
    width, height = resolution
    count = len(chart_tri_uv)
    if count == 0:
        raise TexturingError("NOTHING_TO_PACK", "there are no charts to pack")
    dil = int(math.ceil(padding / 2))
    # Base orientation and box of each chart.
    angles = np.zeros(count)
    rotated = []
    for c, t in enumerate(chart_tri_uv):
        pts = t.reshape(-1, 2)
        a = min_area_angle(pts) if rotate else 0.0
        r = _rotate(pts, a)
        angles[c] = a
        rotated.append((r - r.min(axis=0)).reshape(-1, 3, 2))
    area = np.array([_area(t) for t in rotated])
    total = float(area.sum())
    if total <= 0:
        raise TexturingError("DEGENERATE_UVS", "the charts have no UV area to pack")

    def attempt(density: float):
        bottoms, tops, masks = [], [], []
        for c in range(count):
            mask, origin = footprint(rotated[c] * density, dil)
            masks.append((mask, origin))
            for o in range(ORIENTATIONS):
                if o and not rotate:
                    bottoms.append(np.zeros(0, dtype=np.int64))
                    tops.append(np.zeros(0, dtype=np.int64))
                    continue
                b, tp = profiles(oriented(mask, o))
                bottoms.append(b)
                tops.append(tp)
        sizes = np.array([int(m[0].sum()) for m in masks])
        order = np.lexsort((np.arange(count), -sizes))
        try:
            orient, pos, peak = place(width, bottoms, tops, order)
        except TexturingError as exc:
            if exc.code != "ATLAS_FULL":
                raise
            return None, None, math.inf, masks  # a chart is wider than the atlas
        return orient, pos, peak, masks

    if texels_per_unit is not None:
        density = float(texels_per_unit)
        orient, pos, peak, masks = attempt(density)
        if peak > height:
            raise TexturingError(
                "ATLAS_FULL",
                f"the charts need {peak if math.isfinite(peak) else 'more'} rows at "
                f"{density:g} texels per unit; the atlas has {height}",
                hint="lower texels_per_unit, raise the resolution, or omit texels_per_unit",
                details={
                    "rows_needed": int(peak) if math.isfinite(peak) else None,
                    "rows": int(height),
                },
            )
    else:
        # Bisection on the density; start from an area estimate.
        hi = math.sqrt(width * height / total)  # every texel used: an upper bound
        lo = 0.0
        best = None
        density = 0.7 * hi
        for _ in range(14):
            if should_stop is not None and should_stop():
                raise TexturingError("CANCELLED", "packing was cancelled")
            result = attempt(density)
            if result[2] <= height:
                lo, best = density, (density, result)
            else:
                hi = density
            if best is not None and hi - lo <= 0.004 * hi:
                break
            density = 0.5 * (lo + hi) if best is not None else 0.5 * density
            if density < 1e-9:
                break
        if best is None:
            raise TexturingError("ATLAS_FULL", "the charts cannot be packed at any density")
        density, (orient, pos, peak, masks) = best
    # Transforms to atlas UV.
    out = []
    for c in range(count):
        mask, origin = masks[c]
        o = int(orient[c])
        rel = rotated[c] * density - origin
        p = orient_points(rel, o, mask.shape) + pos[c].astype(np.float64)
        out.append(p / np.array([width, height], dtype=np.float64))
    result = PackResult(
        np.zeros((0, 2)),
        float(density),
        (width, height),
        0.0,
        int(peak),
        orient,
        pos,
    )
    return out, result


def _area(t: np.ndarray) -> float:
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() * 0.5)
