"""Rasterizing a drawing's paper primitives into an 8-bit RGB image (white paper,
black ink, gray fills), with Aether3D's own anti-aliased line and polygon fill code:
reproducible bit for bit, no fonts or dependencies (text uses the stroke font).
"""

from __future__ import annotations

import math

import numpy as np

from ...errors import CadError
from .font import text_strokes

MAX_PIXELS = 64 * 2**20
PIECE = 12.0  # longest stroke piece rasterized at once (pixels)


def dash(points: np.ndarray, pattern: tuple) -> list[np.ndarray]:
    """A polyline cut into its dashes (``pattern``: on, off, on, off ... in its units)."""
    if not pattern:
        return [points]
    seg = np.diff(points, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    total = float(lens.sum())
    if total <= 0:
        return []
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    period = float(sum(pattern))
    out = []
    s = 0.0
    k = 0
    while s < total and len(out) < 100_000:
        length = pattern[k % len(pattern)]
        if k % 2 == 0:
            a, b = s, min(s + length, total)
            ts = np.concatenate([[a], cum[(cum > a) & (cum < b)], [b]])
            xs = np.interp(ts, cum, points[:, 0])
            ys = np.interp(ts, cum, points[:, 1])
            out.append(np.c_[xs, ys])
        s += length
        k += 1
        if period <= 0:
            break
    return out


def hatch_lines(loops: list[np.ndarray], spacing: float, angle: float = math.pi / 4):
    """Segments of parallel lines (``spacing`` apart, at ``angle``) inside the loops
    (even-odd)."""
    c, s = math.cos(angle), math.sin(angle)
    d = np.array([c, s])
    nrm = np.array([-s, c])
    allp = np.concatenate(loops)
    offs = allp @ nrm
    out = []
    starts = np.concatenate([lp for lp in loops])
    ends = np.concatenate([np.roll(lp, -1, axis=0) for lp in loops])
    oa, ob = starts @ nrm, ends @ nrm
    first = math.ceil(offs.min() / spacing)
    last = math.floor(offs.max() / spacing)
    for k in range(first, min(last, first + 5000) + 1):
        o = k * spacing
        cross = (oa <= o) != (ob <= o)
        if not np.any(cross):
            continue
        t = (o - oa[cross]) / (ob[cross] - oa[cross])
        pts = starts[cross] + t[:, None] * (ends[cross] - starts[cross])
        along = np.sort(pts @ d)
        base = nrm * o
        for i in range(0, len(along) - 1, 2):
            out.append(np.array([base + d * along[i], base + d * along[i + 1]]))
    return out


def rasterize(paper: dict, dpi: float) -> np.ndarray:
    w_mm, h_mm = paper["size"]
    ppm = float(dpi) / 25.4
    width, height = int(math.ceil(w_mm * ppm)), int(math.ceil(h_mm * ppm))
    if not (0 < dpi <= 1200) or width * height > MAX_PIXELS:
        raise CadError("LIMIT_EXCEEDED", f"the raster would be {width} x {height} pixels")

    def px(p):
        p = np.asarray(p, dtype=np.float64)
        return np.c_[p[:, 0] * ppm, height - p[:, 1] * ppm]

    img = np.ones((height, width), dtype=np.float64)
    for loops, gray, pattern in paper["fills"]:
        if pattern == "solid":
            cov = _fill_coverage([px(lp) for lp in loops], width, height)
            if cov is not None:
                (y0, x0), c = cov
                region = img[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]]
                np.minimum(region, 1.0 - c * (1.0 - gray), out=region)
    ink = np.zeros((height, width), dtype=np.float64)
    pieces, widths = [], []
    strokes = list(paper["strokes"])
    for pos, h, text, angle, align in paper["texts"]:
        for st in text_strokes(text, pos, h, angle, align):
            strokes.append((st, max(0.13, 0.1 * h), ()))
    for pts, weight, pattern in strokes:
        if pattern == "hatch":
            continue
        for part in dash(np.asarray(pts, dtype=np.float64), pattern):
            q = px(part)
            for a, b in zip(q[:-1], q[1:], strict=True):
                n = max(1, int(math.ceil(np.linalg.norm(b - a) / PIECE)))
                ts = np.linspace(0.0, 1.0, n + 1)
                seg = a[None] + ts[:, None] * (b - a)[None]
                pieces.append(np.c_[seg[:-1], seg[1:]])
                widths.append(np.full(n, max(weight * ppm / 2, 0.5)))
    if pieces:
        _stroke(ink, np.concatenate(pieces), np.concatenate(widths))
    img = img * (1.0 - ink)
    out = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


def _stroke(ink: np.ndarray, segs: np.ndarray, hw: np.ndarray) -> None:
    """Anti-aliased thick segments (``segs``: x0, y0, x1, y1 in pixels), ink = max."""
    height, width = ink.shape
    box = int(math.ceil(PIECE + 2 * float(hw.max()) + 3))
    grid = np.arange(box, dtype=np.float64)
    flat = ink.reshape(-1)
    for start in range(0, len(segs), 4096):
        s = segs[start : start + 4096]
        r = hw[start : start + 4096]
        x0 = np.floor(np.minimum(s[:, 0], s[:, 2]) - r - 1).astype(np.int64)
        y0 = np.floor(np.minimum(s[:, 1], s[:, 3]) - r - 1).astype(np.int64)
        cx = x0[:, None, None] + grid[None, None, :] + 0.5  # pixel centres
        cy = y0[:, None, None] + grid[None, :, None] + 0.5
        ax, ay = s[:, 0, None, None], s[:, 1, None, None]
        dx, dy = (s[:, 2] - s[:, 0])[:, None, None], (s[:, 3] - s[:, 1])[:, None, None]
        ll = dx * dx + dy * dy
        t = np.clip(((cx - ax) * dx + (cy - ay) * dy) / np.where(ll > 0, ll, 1.0), 0.0, 1.0)
        dist = np.hypot(cx - ax - t * dx, cy - ay - t * dy)
        cov = np.clip(r[:, None, None] + 0.5 - dist, 0.0, 1.0)
        xi = x0[:, None, None] + np.arange(box)[None, None, :]
        yi = y0[:, None, None] + np.arange(box)[None, :, None]
        xi, yi = np.broadcast_to(xi, cov.shape), np.broadcast_to(yi, cov.shape)
        ok = (cov > 0) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        np.maximum.at(flat, (yi[ok] * width + xi[ok]), cov[ok])


def _fill_coverage(loops: list[np.ndarray], width: int, height: int):
    """Even-odd coverage (2 x 2 samples per pixel) of polygon loops in pixels."""
    allp = np.concatenate(loops)
    x0 = max(int(math.floor(allp[:, 0].min())), 0)
    y0 = max(int(math.floor(allp[:, 1].min())), 0)
    x1 = min(int(math.ceil(allp[:, 0].max())) + 1, width)
    y1 = min(int(math.ceil(allp[:, 1].max())) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = np.array([0.25, 0.75])
    xs = (np.arange(x0, x1)[:, None] + sub[None]).ravel()
    ys = (np.arange(y0, y1)[:, None] + sub[None]).ravel()
    inside = np.zeros((len(ys), len(xs)), dtype=bool)
    for lp in loops:
        a = lp
        b = np.roll(lp, -1, axis=0)
        for (ax, ay), (bx, by) in zip(a, b, strict=True):
            if ay == by:
                continue
            rows = (ys >= min(ay, by)) & (ys < max(ay, by))
            if not np.any(rows):
                continue
            xc = ax + (ys[rows] - ay) * (bx - ax) / (by - ay)
            inside[rows] ^= xs[None, :] < xc[:, None]
    cov = inside.reshape(y1 - y0, 2, x1 - x0, 2).mean(axis=(1, 3))
    return (y0, x0), cov
