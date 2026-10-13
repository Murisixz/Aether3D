"""Triangle clipping and rasterization into a visibility buffer (vectorized NumPy).

Pipeline for a batch of triangles in homogeneous clip space:

1. **Clipping** (:func:`clip_triangles`): Sutherland-Hodgman against the six
   frustum planes (near first), in homogeneous coordinates where it is exact
   for perspective. A clipped triangle becomes one or two triangles that keep
   its winding.
2. **Setup**: perspective division, viewport transform to the sample grid,
   and snapping to fixed point with :data:`SUBPIXEL_BITS` fractional bits.
3. **Rasterization** (:func:`rasterize`): for every sample row a triangle
   covers, the covered sample span is computed *exactly* from the three
   integer edge functions with the top-left fill rule, so triangles sharing an
   edge cover every sample on it exactly once (watertight; no cracks or double
   coverage).
4. **Visibility**: each fragment's depth (NDC z in ``[0, 1]``, affine in
   screen space) is quantized and packed with its triangle index into one
   ``int64`` key; ``np.minimum.at`` keeps the nearest fragment per sample.
   This is independent of draw order; exact depth ties go to the lowest
   triangle index, so the result is deterministic.

Sample conventions: the sample grid is ``width * s`` by ``height * s`` for
``s x s`` supersampling; sample ``(j, r)`` (column, row; row 0 at the top) has
its center at ``(j + 0.5, r + 0.5)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUBPIXEL_BITS = 8
"""Fractional bits of snapped vertex coordinates (1/256 sample)."""
SUB = 1 << SUBPIXEL_BITS
HALF = SUB // 2
EMPTY = np.iinfo(np.int64).max
"""Visibility-buffer value of a sample no triangle covers."""

ROW_BUDGET = 125_000
"""Sample rows (triangle x row spans) processed per batch."""
FRAGMENT_BUDGET = 500_000
"""Fragments processed per batch (bounds temporary memory)."""

# Clip planes as (a, b, c, d) with "inside" meaning a*x + b*y + c*z + d*w >= 0.
CLIP_PLANES = (
    ("near", (0.0, 0.0, 1.0, 0.0)),
    ("far", (0.0, 0.0, -1.0, 1.0)),
    ("left", (1.0, 0.0, 0.0, 1.0)),
    ("right", (-1.0, 0.0, 0.0, 1.0)),
    ("bottom", (0.0, 1.0, 0.0, 1.0)),
    ("top", (0.0, -1.0, 0.0, 1.0)),
)


def _plane_distance(pos: np.ndarray, plane: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = plane
    return pos[..., 0] * a + pos[..., 1] * b + pos[..., 2] * c + pos[..., 3] * d


def outcodes(clip: np.ndarray) -> np.ndarray:
    """Per-vertex outcodes: bit k is set when the vertex is outside clip plane k.

    A triangle is entirely outside the frustum when its vertices' outcodes
    share a bit (bitwise AND), and needs no clipping when all are zero (OR).
    """
    code = np.zeros(clip.shape[0], dtype=np.uint8)
    for k, (_, plane) in enumerate(CLIP_PLANES):
        code |= (_plane_distance(clip, plane) < 0.0).astype(np.uint8) << k
    return code


def vertex_screen(clip: np.ndarray, width: int, height: int):
    """Perspective division, viewport transform and fixed-point snapping of vertices.

    Returns ``X, Y`` (``int64``, 1/:data:`SUB` sample units) and NDC depth
    ``z``. Only meaningful for vertices with ``w > 0``.
    """
    w = clip[..., 3]
    w = np.where(w > 0.0, w, 1.0)
    x = (clip[..., 0] / w + 1.0) * (0.5 * width)
    y = (1.0 - clip[..., 1] / w) * (0.5 * height)
    z = np.clip(clip[..., 2] / w, 0.0, 1.0)
    X = np.clip(np.rint(x * SUB), 0, width * SUB).astype(np.int64)
    Y = np.clip(np.rint(y * SUB), 0, height * SUB).astype(np.int64)
    return X, Y, z


def signed_area(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Doubled signed area of ``(T, 3)`` fixed-point triangles (exact, ``int64``)."""
    return (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (Y[:, 1] - Y[:, 0]) * (X[:, 2] - X[:, 0])


def outside_mask(pos: np.ndarray) -> np.ndarray:
    """Triangles (``(T, 3, 4)`` clip coordinates) entirely outside one frustum plane."""
    out = np.zeros(pos.shape[0], dtype=bool)
    for _, plane in CLIP_PLANES:
        out |= np.all(_plane_distance(pos, plane) < 0.0, axis=1)
    return out


def inside_mask(pos: np.ndarray) -> np.ndarray:
    """Triangles entirely inside every frustum plane (no clipping needed)."""
    inside = np.ones(pos.shape[0], dtype=bool)
    for _, plane in CLIP_PLANES:
        inside &= np.all(_plane_distance(pos, plane) >= 0.0, axis=1)
    return inside


def clip_triangles(pos: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip ``(T, 3, 4)`` clip-space triangles to the view frustum.

    Returns the resulting triangles and, for each, the id of the triangle it
    came from. Winding is preserved. Interpolation is linear in homogeneous
    clip space, which is exact for perspective projection.
    """
    for _, plane in CLIP_PLANES:
        if pos.shape[0] == 0:
            break
        d = _plane_distance(pos, plane)
        inside = d >= 0.0
        n_in = inside.sum(axis=1)
        parts_pos = [pos[n_in == 3]]
        parts_ids = [ids[n_in == 3]]
        rows = np.arange(pos.shape[0])

        one = n_in == 1
        if one.any():
            r = rows[one]
            k = np.argmax(inside[one], axis=1)  # the inside vertex comes first
            ia, ib, ic = k, (k + 1) % 3, (k + 2) % 3
            a, b, c = pos[r, ia], pos[r, ib], pos[r, ic]
            da, db, dc = d[r, ia], d[r, ib], d[r, ic]
            ab = a + (da / (da - db))[:, None] * (b - a)
            ac = a + (da / (da - dc))[:, None] * (c - a)
            parts_pos.append(np.stack([a, ab, ac], axis=1))
            parts_ids.append(ids[one])

        two = n_in == 2
        if two.any():
            r = rows[two]
            k = np.argmin(inside[two], axis=1)  # the outside vertex comes last
            ia, ib, ic = (k + 1) % 3, (k + 2) % 3, k
            a, b, c = pos[r, ia], pos[r, ib], pos[r, ic]
            da, db, dc = d[r, ia], d[r, ib], d[r, ic]
            bc = b + (db / (db - dc))[:, None] * (c - b)
            ca = c + (dc / (dc - da))[:, None] * (a - c)
            parts_pos += [np.stack([a, b, bc], axis=1), np.stack([a, bc, ca], axis=1)]
            parts_ids += [ids[two], ids[two]]

        pos = np.concatenate(parts_pos)
        ids = np.concatenate(parts_ids)
    return pos, ids


@dataclass
class RasterStats:
    triangles: int = 0  # triangles reaching the rasterizer (after clipping)
    degenerate: int = 0  # zero area after snapping to the sample grid
    fragments: int = 0  # covered samples, counting overdraw

    def to_dict(self) -> dict[str, int]:
        return {
            "rasterized": self.triangles,
            "degenerate": self.degenerate,
            "fragments": self.fragments,
        }


def screen_setup(
    pos: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perspective division and viewport transform of clipped triangles.

    Returns fixed-point sample coordinates ``X, Y`` (``(T, 3) int64``), NDC
    depth ``z`` (``(T, 3)``) and the doubled signed area ``A2`` (``int64``;
    negative for triangles that are counter-clockwise in NDC).
    """
    X, Y, z = vertex_screen(pos, width, height)
    return X, Y, z, signed_area(X, Y)


def rasterize(
    X: np.ndarray,
    Y: np.ndarray,
    z: np.ndarray,
    A2: np.ndarray,
    ids: np.ndarray,
    width: int,
    height: int,
    keys: np.ndarray,
    id_bits: int,
    stats: RasterStats,
) -> None:
    """Rasterize snapped triangles into the visibility buffer ``keys`` (updated in place).

    ``keys`` has ``width * height`` entries (row-major, row 0 at the top) and
    holds ``(depth_q << id_bits) | id`` of the nearest fragment, or
    :data:`EMPTY`.
    """
    stats.triangles += int(X.shape[0])
    nonzero = A2 != 0
    stats.degenerate += int(np.count_nonzero(~nonzero))
    X, Y, z, A2, ids = X[nonzero], Y[nonzero], z[nonzero], A2[nonzero], ids[nonzero]
    # Orient every triangle so that A2 > 0 (inside = all edge functions >= 0).
    flip = A2 < 0
    if flip.any():
        for arr in (X, Y, z):
            arr[flip, 1], arr[flip, 2] = arr[flip, 2].copy(), arr[flip, 1].copy()
        A2 = np.abs(A2)
    if X.shape[0] == 0:
        return

    # Sample rows and columns whose centers can lie inside each triangle.
    r_lo = np.maximum(-((HALF - Y.min(axis=1)) // SUB), 0)
    r_hi = np.minimum((Y.max(axis=1) - HALF) // SUB, height - 1)
    c_lo = np.maximum(-((HALF - X.min(axis=1)) // SUB), 0)
    c_hi = np.minimum((X.max(axis=1) - HALF) // SUB, width - 1)
    rows = np.where((r_hi >= r_lo) & (c_hi >= c_lo), r_hi - r_lo + 1, 0)

    # Depth plane z(x, y) = z0 + dzdx (x - x0) + dzdy (y - y0) in sample units.
    xf, yf = X / SUB, Y / SUB
    area = A2 / float(SUB * SUB)
    dz1, dz2 = z[:, 1] - z[:, 0], z[:, 2] - z[:, 0]
    dx1, dx2 = xf[:, 1] - xf[:, 0], xf[:, 2] - xf[:, 0]
    dy1, dy2 = yf[:, 1] - yf[:, 0], yf[:, 2] - yf[:, 0]
    dzdx = (dz1 * dy2 - dz2 * dy1) / area
    dzdy = (dz2 * dx1 - dz1 * dx2) / area

    qbits = 63 - id_bits
    qscale = float((1 << qbits) - 1)

    # Batches of triangles with a bounded number of row spans.
    cum = np.cumsum(rows)
    start = 0
    while start < X.shape[0]:
        base = cum[start - 1] if start else 0
        stop = int(np.searchsorted(cum, base + ROW_BUDGET, side="right"))
        stop = max(stop, start + 1)
        _rasterize_batch(
            slice(start, stop),
            X,
            Y,
            z,
            xf,
            yf,
            dzdx,
            dzdy,
            ids,
            rows,
            r_lo,
            c_lo,
            c_hi,
            width,
            keys,
            id_bits,
            qscale,
            stats,
        )
        start = stop


def _rasterize_batch(
    sel,
    X,
    Y,
    z,
    xf,
    yf,
    dzdx,
    dzdy,
    ids,
    rows,
    r_lo,
    c_lo,
    c_hi,
    width,
    keys,
    id_bits,
    qscale,
    stats,
) -> None:
    n_rows = rows[sel]
    total = int(n_rows.sum())
    if total == 0:
        return
    tri = np.repeat(np.arange(sel.start, sel.stop), n_rows)
    first = np.cumsum(n_rows) - n_rows
    r = r_lo[tri] + (np.arange(total) - np.repeat(first, n_rows))
    yc = r * SUB + HALF  # sample-center y in fixed point
    j_lo = c_lo[tri].copy()
    j_hi = c_hi[tri].copy()
    Xt, Yt = X[tri], Y[tri]
    for e in range(3):
        xa, ya = Xt[:, e], Yt[:, e]
        dx = Xt[:, (e + 1) % 3] - xa
        dy = Yt[:, (e + 1) % 3] - ya
        # Edge function at sample (j, r): alpha * j + beta. Inside: >= 0 on
        # top-left (inclusive) edges, >= 1 on the others.
        alpha = -SUB * dy
        beta = dx * (yc - ya) - dy * (HALF - xa)
        inclusive = (dy < 0) | ((dy == 0) & (dx > 0))
        gamma = np.where(inclusive, 0, 1) - beta  # need alpha * j >= gamma
        safe = np.where(alpha == 0, 1, alpha)
        lower = np.where(alpha > 0, -((-gamma) // safe), np.iinfo(np.int64).min)
        upper = np.where(alpha < 0, gamma // safe, np.iinfo(np.int64).max)
        np.maximum(j_lo, lower, out=j_lo)
        np.minimum(j_hi, upper, out=j_hi)
        j_hi[(alpha == 0) & (gamma > 0)] = -1  # a horizontal edge excludes the whole row
    count = np.maximum(j_hi - j_lo + 1, 0)

    # Expand spans into fragments, in batches of bounded size.
    cum = np.cumsum(count)
    n_spans = count.shape[0]
    s0 = 0
    while s0 < n_spans:
        base = cum[s0 - 1] if s0 else 0
        s1 = int(np.searchsorted(cum, base + FRAGMENT_BUDGET, side="right"))
        s1 = max(s1, s0 + 1)
        c = count[s0:s1]
        n = int(c.sum())
        if n:
            span = np.repeat(np.arange(s0, s1), c)
            offsets = np.cumsum(c) - c
            j = j_lo[span] + (np.arange(n) - np.repeat(offsets, c))
            rr = r[span]
            t = tri[span]
            depth = z[t, 0] + dzdx[t] * ((j + 0.5) - xf[t, 0]) + dzdy[t] * ((rr + 0.5) - yf[t, 0])
            q = (np.clip(depth, 0.0, 1.0) * qscale).astype(np.int64)
            key = (q << id_bits) | ids[t]
            np.minimum.at(keys, rr * width + j, key)
            stats.fragments += n
        s0 = s1
