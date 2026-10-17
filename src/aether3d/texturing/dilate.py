"""Padding (edge dilation): extending island values into the empty texels around them.

Filtering reads texels beyond an island's border (bilinear: one texel; mip level ``k``:
about ``2^k``). Texels outside every island hold no surface value, so without padding a
seam shows the background color. ``dilate`` grows the covered region ring by ring: an
uncovered texel next to covered ones (8-neighbourhood) takes the average of those
neighbours' values, then counts as covered for the next ring. The result is deterministic
and only touches uncovered texels within ``steps`` rings.
"""

from __future__ import annotations

import numpy as np

SHIFTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def dilate(values: np.ndarray, covered: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """``values`` ``(H, W, C)`` grown ``steps`` rings into the texels where ``covered``
    ``(H, W)`` is false; returns the values and the grown coverage."""
    v = np.array(values, dtype=np.float64, copy=True)
    cov = np.array(covered, dtype=bool, copy=True)
    h, w = cov.shape
    for _ in range(int(steps)):
        if cov.all():
            break
        total = np.zeros_like(v)
        count = np.zeros((h, w), dtype=np.float64)
        for dy, dx in SHIFTS:
            ys = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            src_cov = cov[ys, xs]
            total[yd, xd] += np.where(src_cov[..., None], v[ys, xs], 0.0)
            count[yd, xd] += src_cov
        grow = ~cov & (count > 0)
        if not grow.any():
            break
        v[grow] = total[grow] / count[grow][:, None]
        cov = cov | grow
    return v, cov
