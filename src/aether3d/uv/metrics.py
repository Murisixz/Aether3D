"""Distortion and overlap measures of UV maps.

* **Stretch** (Sander et al. 2001). For a triangle whose map has singular values
  ``s1 >= s2 > 0`` (``J``: from the surface to UV), the surface-from-texture map has
  singular values ``1/s2`` and ``1/s1``; ``L2 = sqrt((1/s1^2 + 1/s2^2) / 2)`` and
  ``Linf = 1/s2``. A chart's values are area-weighted (L2, root mean square) or the
  maximum (Linf) after scaling its UVs so their total area equals the surface's: 1 means
  an isometry, larger means stretch. Inverted or flat triangles give infinity.
* **Boundary overlaps**: pairs of a chart's boundary edges that meet in UV, apart from
  neighbours sharing a vertex -- decided exactly (:mod:`aether3d.spatial.predicates`).
"""

from __future__ import annotations

import numpy as np

from ..spatial.predicates import segments_meet_2d
from . import param


def stretch(cs: param.ChartSet, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-chart scale-normalized L2 and Linf stretch (infinite for invalid maps)."""
    det = param.uv_det(uv, cs.tri)
    live = cs.area > 0
    uv_area = np.bincount(cs.face_chart, np.where(live, 0.5 * np.abs(det), 0.0), cs.charts)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(cs.chart_area / uv_area)
    j = param.jacobians(cs, np.where(np.isfinite(uv), uv, 0.0))
    _, s1, s2, _ = param.svd2(j)
    sc = scale[cs.face_chart]
    s1, s2 = s1 * sc, s2 * sc
    bad = live & ((s2 <= 0) | ~np.isfinite(s2) | ~np.isfinite(det))
    with np.errstate(divide="ignore", invalid="ignore"):
        l2_t = np.where(live & ~bad, (1.0 / (s1 * s1) + 1.0 / (s2 * s2)) / 2.0, 0.0)
        linf_t = np.where(live & ~bad, 1.0 / s2, 0.0)
    l2 = np.sqrt(
        np.bincount(cs.face_chart, cs.area * l2_t, cs.charts) / np.maximum(cs.chart_area, 1e-300)
    )
    linf = np.zeros(cs.charts)
    np.maximum.at(linf, cs.face_chart, linf_t)
    nbad = np.bincount(cs.face_chart, bad.astype(np.float64), cs.charts)
    invalid = (nbad > 0) | ~np.isfinite(scale) | (cs.chart_area <= 0)
    l2 = np.where(invalid, np.inf, l2)
    linf = np.where(invalid, np.inf, linf)
    # A chart of only degenerate triangles has nothing to stretch.
    flat = cs.chart_area <= 0
    l2 = np.where(flat, 1.0, l2)
    linf = np.where(flat, 1.0, linf)
    return l2, linf


MAX_SPAN = 64
"""Grid cells per axis beyond which a box is a "big" box (looked up, not entered)."""


def segment_pairs(lo: np.ndarray, hi: np.ndarray, group: np.ndarray, cell: np.ndarray):
    """Candidate pairs ``(i, j)``, ``i < j``, of boxes ``[lo, hi]`` (``(S, 2)``) in the same
    ``group`` whose boxes may meet, each reported once.

    Boxes spanning at most :data:`MAX_SPAN` cells per axis of a uniform grid (``cell``:
    per-group cell size) are entered in every cell they cover; two of them are a
    candidate when they share a cell, reported only in the cell at the larger of their
    lower cell corners (the one cell both always cover), so nothing needs deduplicating.
    Each larger box looks up the entered boxes of the cells it covers, one column of cells
    at a time (so a long, thin box costs its own cells, not its bounding square), and big
    boxes are paired with each other by a direct box test."""
    empty = np.zeros(0, dtype=np.int64)
    n = lo.shape[0]
    if n < 2:
        return empty, empty
    size = cell[group][:, None]
    c0 = np.floor(lo / size).astype(np.int64)
    c1 = np.floor(hi / size).astype(np.int64)
    span = c1 - c0 + 1
    big = (span > MAX_SPAN).any(axis=1)
    small = np.flatnonzero(~big)
    pa, pb = [], []
    g = cx = cy = seg = np.zeros(0, dtype=np.int64)
    if small.size:
        sp = span[small]
        counts = sp[:, 0] * sp[:, 1]
        seg = np.repeat(small, counts)
        rep = np.repeat(np.arange(small.size), counts)
        k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        cx = c0[seg, 0] + k % sp[rep, 0]
        cy = c0[seg, 1] + k // sp[rep, 0]
        order = np.lexsort((seg, cy, cx, group[seg]))
        g, cx, cy, seg = group[seg][order], cx[order], cy[order], seg[order]
        same = (g[1:] == g[:-1]) & (cx[1:] == cx[:-1]) & (cy[1:] == cy[:-1])
        starts = np.flatnonzero(np.r_[True, ~same])
        lengths = np.diff(np.r_[starts, seg.shape[0]])
        for length in np.unique(lengths[lengths > 1]).tolist():
            st = starts[lengths == length]
            ii, jj = np.triu_indices(length, 1)
            a = seg[st[:, None] + ii].reshape(-1)
            b = seg[st[:, None] + jj].reshape(-1)
            at = np.repeat(st, ii.size)
            keep = (cx[at] == np.maximum(c0[a, 0], c0[b, 0])) & (
                cy[at] == np.maximum(c0[a, 1], c0[b, 1])
            )
            pa.append(a[keep])
            pb.append(b[keep])
    bigs = np.flatnonzero(big)
    if bigs.size:
        # Big boxes against the entered ones: per group, cell keys (column, row) in one
        # sorted int64 array; each big box looks up its rows of every column it covers.
        for grp in np.unique(group[bigs]).tolist():
            lo_e = np.searchsorted(g, grp, side="left")
            hi_e = np.searchsorted(g, grp, side="right")
            mine = bigs[group[bigs] == grp]
            if hi_e > lo_e:
                ex, ey, es = cx[lo_e:hi_e], cy[lo_e:hi_e], seg[lo_e:hi_e]
                x0, y0 = int(ex.min()), int(ey.min())
                ry = int(ey.max()) - y0 + 1
                key = (ex - x0) * ry + (ey - y0)  # sorted (columns, then rows)
                xmax = int(ex.max())
                for i in mine.tolist():
                    bx0, bx1 = max(int(c0[i, 0]), x0), min(int(c1[i, 0]), xmax)
                    by0 = max(int(c0[i, 1]), y0) - y0
                    by1 = min(int(c1[i, 1]), y0 + ry - 1) - y0
                    if bx1 < bx0 or by1 < by0:
                        continue
                    cols = np.arange(bx0, bx1 + 1, dtype=np.int64) - x0
                    first = np.searchsorted(key, cols * ry + by0, side="left")
                    last = np.searchsorted(key, cols * ry + by1, side="right")
                    length = last - first
                    if not length.any():
                        continue
                    idx = np.repeat(first - np.cumsum(length) + length, length) + np.arange(
                        int(length.sum())
                    )
                    other = np.unique(es[idx])
                    pa.append(np.full(other.size, i, dtype=np.int64))
                    pb.append(other)
            # Big boxes with each other: a direct box test.
            if mine.size > 1:
                ii, jj = np.triu_indices(mine.size, 1)
                a, b = mine[ii], mine[jj]
                meet = np.all(lo[a] <= hi[b], axis=1) & np.all(lo[b] <= hi[a], axis=1)
                pa.append(a[meet])
                pb.append(b[meet])
    if not pa:
        return empty, empty
    a = np.concatenate(pa)
    b = np.concatenate(pb)
    return np.minimum(a, b), np.maximum(a, b)


def boundary_overlaps(
    cs: param.ChartSet, bd: param.Boundary, uv: np.ndarray, charts: np.ndarray
) -> np.ndarray:
    """Per chart, the number of pairs of its boundary edges that meet in UV (neighbours
    sharing a vertex excluded); only charts flagged in ``charts`` are tested."""
    out = np.zeros(cs.charts, dtype=np.int64)
    sel = charts[cs.vertex_chart[bd.start]]
    s, e = bd.start[sel], bd.end[sel]
    if s.size < 2:
        return out
    p, q = uv[s], uv[e]
    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    s, e, p, q = s[ok], e[ok], p[ok], q[ok]
    group = cs.vertex_chart[s]
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    length = np.linalg.norm(q - p, axis=1)
    count = np.bincount(group, minlength=cs.charts)
    mean = np.bincount(group, length, cs.charts) / np.maximum(count, 1)
    cell = np.where(mean > 0, 2.0 * mean, 1.0)
    a, b = segment_pairs(lo, hi, group, cell)
    if a.size == 0:
        return out
    adjacent = (s[a] == s[b]) | (s[a] == e[b]) | (e[a] == s[b]) | (e[a] == e[b])
    a, b = a[~adjacent], b[~adjacent]
    boxes = np.all(lo[a] <= hi[b], axis=1) & np.all(lo[b] <= hi[a], axis=1)
    a, b = a[boxes], b[boxes]
    if a.size == 0:
        return out
    meet = segments_meet_2d(p[a], q[a], p[b], q[b])
    np.add.at(out, group[a[meet]], 1)
    return out
