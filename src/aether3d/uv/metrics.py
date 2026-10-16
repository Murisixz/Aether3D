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


def segment_pairs(lo: np.ndarray, hi: np.ndarray, group: np.ndarray, cell: np.ndarray):
    """Candidate pairs ``(i, j)``, ``i < j``, of boxes ``[lo, hi]`` (``(S, 2)``) in the same
    ``group`` whose boxes share a grid cell (``cell``: per-group cell size)."""
    if lo.shape[0] < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    size = cell[group][:, None]
    c0 = np.floor(lo / size).astype(np.int64)
    c1 = np.floor(hi / size).astype(np.int64)
    span = (c1 - c0 + 1).clip(1, 64)
    counts = span[:, 0] * span[:, 1]
    seg = np.repeat(np.arange(lo.shape[0]), counts)
    k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    cx = c0[seg, 0] + k % span[seg, 0]
    cy = c0[seg, 1] + k // span[seg, 0]
    key = np.stack([group[seg], cx, cy], axis=1)
    order = np.lexsort((seg, key[:, 2], key[:, 1], key[:, 0]))
    key, seg = key[order], seg[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    starts = np.flatnonzero(np.r_[True, ~same])
    lengths = np.diff(np.r_[starts, seg.shape[0]])
    pa, pb = [], []
    for length in np.unique(lengths[lengths > 1]).tolist():
        s = starts[lengths == length]
        ii, jj = np.triu_indices(length, 1)
        pa.append(seg[s[:, None] + ii].reshape(-1))
        pb.append(seg[s[:, None] + jj].reshape(-1))
    if not pa:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    a = np.concatenate(pa)
    b = np.concatenate(pb)
    a, b = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(a * lo.shape[0] + b)
    return key // lo.shape[0], key % lo.shape[0]


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
