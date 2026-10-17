"""Rasterizing triangles in UV space: which triangle covers each texel sample, exactly.

Baking and painting need, for every texel, the surface point it stands for. The texel
grid follows Aether3D's texture convention: an image of ``W x H`` texels, ``(0, 0)`` the
bottom-left of the UV square, texel ``(x, y)`` (``y`` counted from the **top** row, as in
image files) spanning ``u in [x, x + 1] / W`` and ``v in [1 - (y + 1) / H, 1 - y / H]``.

With ``samples = n``, each texel holds ``n x n`` stratified samples at
``((x + (i + 0.5) / n), (y + (j + 0.5) / n))``. Sample coverage uses the CPU
rasterizer's rules (:mod:`aether3d.rendering.raster`): vertices snapped to 1/256 of a
sample, exact integer edge functions and the top-left fill rule, so triangles sharing an
edge cover every sample on it exactly once -- no gaps and no double coverage inside a
chart. Where UV islands overlap, a sample keeps the lowest-numbered triangle and the
overlap is counted. Barycentric coordinates come from the same integer edge functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUBPIXEL_BITS = 8
SUB = 1 << SUBPIXEL_BITS
HALF = SUB // 2
ROW_BUDGET = 1 << 17
FRAGMENT_BUDGET = 1 << 20


@dataclass
class Coverage:
    """Covered samples of a texel grid (all arrays have one row per covered sample)."""

    width: int  # texels
    height: int
    samples: int  # per axis
    sample: np.ndarray  # (k,) int64 linear index in the (H n) x (W n) sample grid, row 0 top
    triangle: np.ndarray  # (k,) int64 triangle id
    bary: np.ndarray  # (k, 3) float64
    overlaps: int  # samples covered by more than one triangle

    @property
    def texel(self) -> np.ndarray:
        """``(k,)`` linear texel index (row 0 the top row) of each sample."""
        n = self.samples
        row, col = np.divmod(self.sample, self.width * n)
        return (row // n) * self.width + col // n


def rasterize_uv(
    uv: np.ndarray,
    width: int,
    height: int,
    samples: int = 1,
    triangles: np.ndarray | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> Coverage:
    """Coverage of the triangles' UVs (``(T, 3, 2)``) on a ``width x height`` texel grid with
    ``samples x samples`` samples per texel; ``triangles`` names each row (default
    ``arange(T)``); ``region`` ``(x0, y0, x1, y1)`` (texels) limits it to a rectangle."""
    uv = np.asarray(uv, dtype=np.float64)
    T = uv.shape[0]
    ids = np.arange(T, dtype=np.int64) if triangles is None else np.asarray(triangles, np.int64)
    sw, sh = width * samples, height * samples
    # Sample-grid coordinates (row 0 at the top), snapped to fixed point.
    fx = uv[..., 0] * sw
    fy = (1.0 - uv[..., 1]) * sh
    finite = np.isfinite(fx).all(axis=1) & np.isfinite(fy).all(axis=1)
    limit = float(1 << 40)
    finite &= (np.abs(fx) < limit).all(axis=1) & (np.abs(fy) < limit).all(axis=1)
    X = np.round(fx[finite] * SUB).astype(np.int64)
    Y = np.round(fy[finite] * SUB).astype(np.int64)
    ids = ids[finite]
    A2 = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    keep = A2 != 0
    X, Y, A2, ids = X[keep], Y[keep], A2[keep], ids[keep]
    # Orient every triangle so that A2 > 0 (y grows downward on the sample grid).
    flip = A2 < 0
    if flip.any():
        X[flip, 1], X[flip, 2] = X[flip, 2].copy(), X[flip, 1].copy()
        Y[flip, 1], Y[flip, 2] = Y[flip, 2].copy(), Y[flip, 1].copy()
        A2 = np.abs(A2)
    swap = flip  # corners 1 and 2 swapped: barycentrics swap back below
    rx0, ry0, rx1, ry1 = (0, 0, width, height) if region is None else region
    r_lo = np.maximum(-((HALF - Y.min(axis=1)) // SUB), ry0 * samples)
    r_hi = np.minimum((Y.max(axis=1) - HALF) // SUB, ry1 * samples - 1)
    c_lo = np.maximum(-((HALF - X.min(axis=1)) // SUB), rx0 * samples)
    c_hi = np.minimum((X.max(axis=1) - HALF) // SUB, rx1 * samples - 1)
    rows = np.where((r_hi >= r_lo) & (c_hi >= c_lo), r_hi - r_lo + 1, 0)
    out_s, out_t, out_b = [], [], []
    cum = np.cumsum(rows)
    start = 0
    while start < X.shape[0]:
        base = cum[start - 1] if start else 0
        stop = max(int(np.searchsorted(cum, base + ROW_BUDGET, side="right")), start + 1)
        _batch(
            slice(start, stop), X, Y, A2, ids, swap, rows, r_lo, c_lo, c_hi, sw, out_s, out_t, out_b
        )
        start = stop
    if out_s:
        s = np.concatenate(out_s)
        t = np.concatenate(out_t)
        b = np.concatenate(out_b)
    else:
        s = np.zeros(0, np.int64)
        t = np.zeros(0, np.int64)
        b = np.zeros((0, 3))
    # Overlaps: keep the lowest triangle id per sample.
    order = np.lexsort((t, s))
    s, t, b = s[order], t[order], b[order]
    first = np.r_[True, s[1:] != s[:-1]]
    overlaps = int(np.unique(s[~first]).shape[0])
    return Coverage(width, height, samples, s[first], t[first], b[first], overlaps)


def _batch(sel, X, Y, A2, ids, swap, rows, r_lo, c_lo, c_hi, sw, out_s, out_t, out_b):
    n_rows = rows[sel]
    total = int(n_rows.sum())
    if total == 0:
        return
    tri = np.repeat(np.arange(sel.start, sel.stop), n_rows)
    first = np.cumsum(n_rows) - n_rows
    r = r_lo[tri] + (np.arange(total) - np.repeat(first, n_rows))
    yc = r * SUB + HALF
    j_lo = c_lo[tri].copy()
    j_hi = c_hi[tri].copy()
    Xt, Yt = X[tri], Y[tri]
    for e in range(3):
        xa, ya = Xt[:, e], Yt[:, e]
        dx = Xt[:, (e + 1) % 3] - xa
        dy = Yt[:, (e + 1) % 3] - ya
        alpha = -SUB * dy
        beta = dx * (yc - ya) - dy * (HALF - xa)
        inclusive = (dy < 0) | ((dy == 0) & (dx > 0))
        gamma = np.where(inclusive, 0, 1) - beta
        safe = np.where(alpha == 0, 1, alpha)
        lower = np.where(alpha > 0, -((-gamma) // safe), np.iinfo(np.int64).min)
        upper = np.where(alpha < 0, gamma // safe, np.iinfo(np.int64).max)
        np.maximum(j_lo, lower, out=j_lo)
        np.minimum(j_hi, upper, out=j_hi)
        j_hi[(alpha == 0) & (gamma > 0)] = -1
    count = np.maximum(j_hi - j_lo + 1, 0)
    cum = np.cumsum(count)
    s0 = 0
    while s0 < count.shape[0]:
        base = cum[s0 - 1] if s0 else 0
        s1 = max(int(np.searchsorted(cum, base + FRAGMENT_BUDGET, side="right")), s0 + 1)
        c = count[s0:s1]
        n = int(c.sum())
        if n:
            span = np.repeat(np.arange(s0, s1), c)
            offsets = np.cumsum(c) - c
            j = j_lo[span] + (np.arange(n) - np.repeat(offsets, c))
            t = tri[span]
            rr = r[span]
            xc = j * SUB + HALF
            ycc = rr * SUB + HALF
            # Edge functions opposite each vertex (exact), normalized by the area.
            e = []
            for k in range(3):
                a, b = (k + 1) % 3, (k + 2) % 3
                e.append(
                    (X[t, b] - X[t, a]) * (ycc - Y[t, a]) - (Y[t, b] - Y[t, a]) * (xc - X[t, a])
                )
            area = A2[t].astype(np.float64)
            bary = np.stack([e[0] / area, e[1] / area, e[2] / area], axis=1)
            sw_ = swap[t]
            if sw_.any():
                bary[sw_, 1], bary[sw_, 2] = bary[sw_, 2].copy(), bary[sw_, 1].copy()
            out_s.append(rr * sw + j)
            out_t.append(ids[t])
            out_b.append(bary)
        s0 = s1
