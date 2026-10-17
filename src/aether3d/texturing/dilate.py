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
    ``(H, W)`` is false; returns the values and the grown coverage.

    Only each ring's frontier is visited (the uncovered texels next to covered ones), its
    neighbours gathered in :data:`SHIFTS` order, so the work follows the islands' borders,
    not the image's area."""
    v = np.array(values, dtype=np.float64, copy=True)
    cov = np.array(covered, dtype=bool, copy=True)
    h, w = cov.shape
    channels = v.shape[2] if v.ndim == 3 else 1
    flat_v = v.reshape(h * w, channels)
    flat_c = cov.reshape(-1)
    steps = int(steps)
    if steps <= 0 or flat_c.all() or not flat_c.any():
        return v, cov
    # The first frontier: uncovered texels with a covered neighbour.
    near = np.zeros((h, w), dtype=bool)
    for dy, dx in SHIFTS:
        ys = slice(max(dy, 0), h + min(dy, 0))
        yd = slice(max(-dy, 0), h + min(-dy, 0))
        xs = slice(max(dx, 0), w + min(dx, 0))
        xd = slice(max(-dx, 0), w + min(-dx, 0))
        near[yd, xd] |= cov[ys, xs]
    frontier = np.flatnonzero(near.reshape(-1) & ~flat_c)
    for _ in range(steps):
        if frontier.size == 0:
            break
        fy, fx = np.divmod(frontier, w)
        total = np.zeros((frontier.size, channels))
        count = np.zeros(frontier.size)
        for dy, dx in SHIFTS:
            ny, nx = fy + dy, fx + dx
            inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            idx = np.where(inside, ny * w + nx, 0)
            ok = inside & flat_c[idx]
            total += np.where(ok[:, None], flat_v[idx], 0.0)
            count += ok
        flat_v[frontier] = total / count[:, None]
        flat_c[frontier] = True
        # The next frontier: uncovered neighbours of this one.
        nxt = []
        for dy, dx in SHIFTS:
            ny, nx = fy + dy, fx + dx
            inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            idx = (ny * w + nx)[inside]
            nxt.append(idx[~flat_c[idx]])
        frontier = np.unique(np.concatenate(nxt))
    return v, cov
