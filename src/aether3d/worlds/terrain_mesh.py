"""Terrain tile meshes: heights sampled on a tile's grid, normals, colours, stitched
borders, skirts and water surfaces.

A tile is sampled on its ``(R + 1)^2`` vertices plus one ring outside them, so vertex
normals (central differences) are computed from the same samples on both sides of a tile
border: neighbouring tiles of one level share their border vertices, normals and colours
bit for bit. Positions are stored relative to the tile's corner (float32; exact binary
fractions below 2^24) and placed with a float64 matrix, so the GPU's camera-relative
transform keeps every tile precise at any distance from the origin.

* **Stitching.** A border marked as next to a coarser tile moves each odd vertex onto the
  segment between its even neighbours (their average height, normal and colour), so the
  two tiles meet along the same segments.
* **Skirts.** Each border gets a vertical strip hanging below it (depth: the tile's error
  plus a margin), which hides the float32 rounding of camera-relative coordinates and the
  remaining LOD differences without ever showing above ground.
* **Error.** The tile's geometric error estimate is half the largest deviation of a vertex
  from the line through its two neighbours (second differences along rows and columns): the
  error a mesh of twice the spacing would make here, scaled to this one.
* **Water.** Quads with water above the terrain at any corner get a water quad at the
  water level (sea, lake or river level; dry corners take the highest wet neighbour's),
  coloured by depth; the terrain hides it where the ground is higher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..mesh import Mesh
from .surface import colors as surface_colors
from .surface import water_colors
from .tiles import STITCH_BITS, Grid, TileKey


@dataclass
class TileData:
    """A generated terrain tile (immutable meshes; arrays read-only)."""

    key: TileKey
    origin: tuple[float, float]
    mesh: Mesh
    water: Mesh | None
    error: float
    zrange: tuple[float, float]
    heights: np.ndarray  # (R + 1, R + 1) final heights at the vertices (rows: y)
    seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        total = self.heights.nbytes
        for m in (self.mesh, self.water):
            if m is not None:
                for a in (m.positions, m.triangles, m.normals, m.uvs, m.colors):
                    if a is not None:
                        total += a.nbytes
        return int(total)


def _triangles(r: int) -> np.ndarray:
    """Quads of an ``(r + 1)^2`` grid (row-major, rows along y) as triangles, counter-
    clockwise seen from +z, split along alternating diagonals (a symmetric pattern)."""
    j, i = np.mgrid[0:r, 0:r]
    v00 = (j * (r + 1) + i).reshape(-1)
    v10 = v00 + 1
    v01 = v00 + (r + 1)
    v11 = v01 + 1
    flip = ((i + j) % 2 == 1).reshape(-1)
    a = np.where(flip[:, None], np.stack([v00, v10, v11], 1), np.stack([v00, v10, v01], 1))
    b = np.where(flip[:, None], np.stack([v00, v11, v01], 1), np.stack([v10, v11, v01], 1))
    return np.concatenate([a, b]).astype(np.uint32)


_TRIS: dict[int, np.ndarray] = {}


def grid_triangles(r: int) -> np.ndarray:
    tris = _TRIS.get(r)
    if tris is None:
        tris = _TRIS[r] = _triangles(r)
        tris.flags.writeable = False
    return tris


def _skirt(r: int, base: int, depth_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Border vertex indices (counter-clockwise around the tile) and the skirt triangles
    joining them to their copies ``base + k`` (facing outward)."""
    n = r + 1
    south = [j * n + i for j, i in [(0, i) for i in range(n)]]
    east = [j * n + (n - 1) for j in range(1, n)]
    north = [(n - 1) * n + i for i in range(n - 2, -1, -1)]
    west = [j * n for j in range(n - 2, 0, -1)]
    ring = np.array(south + east + north + west, dtype=np.int64)
    m = ring.shape[0]
    k = np.arange(m)
    k1 = (k + 1) % m
    top_a, top_b = ring[k], ring[k1]
    low_a, low_b = base + k, base + k1
    # Outward faces: the ring runs counter-clockwise seen from above, so (a, low_a, b)...
    t1 = np.stack([top_a, low_a, low_b], 1)
    t2 = np.stack([top_a, low_b, top_b], 1)
    return ring, np.concatenate([t1, t2]).astype(np.uint32)


def _stitch(grid2: np.ndarray, mask: int, r: int) -> None:
    """Move the odd vertices of stitched borders onto their even neighbours' segment
    (in place on a ``(r + 1, r + 1, c)`` array of per-vertex values)."""
    if mask & STITCH_BITS["-y"]:
        grid2[0, 1:r:2] = 0.5 * (grid2[0, 0 : r - 1 : 2] + grid2[0, 2 : r + 1 : 2])
    if mask & STITCH_BITS["+y"]:
        grid2[r, 1:r:2] = 0.5 * (grid2[r, 0 : r - 1 : 2] + grid2[r, 2 : r + 1 : 2])
    if mask & STITCH_BITS["-x"]:
        grid2[1:r:2, 0] = 0.5 * (grid2[0 : r - 1 : 2, 0] + grid2[2 : r + 1 : 2, 0])
    if mask & STITCH_BITS["+x"]:
        grid2[1:r:2, r] = 0.5 * (grid2[0 : r - 1 : 2, r] + grid2[2 : r + 1 : 2, r])


def build_tile(
    model: Any,
    grid: Grid,
    key: TileKey,
    stitch: int = 0,
    ctx: Any = None,
    keep_sample: bool = False,
) -> TileData:
    """Generate one terrain tile (see the module docstring)."""
    import time

    t0 = time.perf_counter()
    r = grid.resolution
    xs, ys = grid.coords(key, border=1)
    X, Y = np.meshgrid(xs, ys)
    sample = model.evaluate(X.reshape(-1), Y.reshape(-1), ctx)
    n = r + 3
    H = sample["height"].reshape(n, n)
    step = grid.step(key.level)
    # Normals from central differences on the bordered grid.
    dx = (H[1:-1, 2:] - H[1:-1, :-2]) / (2.0 * step)
    dy = (H[2:, 1:-1] - H[:-2, 1:-1]) / (2.0 * step)
    nrm = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    nrm /= np.sqrt(np.sum(nrm * nrm, axis=-1, keepdims=True))
    inner = (slice(1, -1), slice(1, -1))
    h = H[inner]
    # Error: half the largest second difference along rows and columns.
    e1 = np.abs(H[1:-1, :-2] - 2.0 * H[1:-1, 1:-1] + H[1:-1, 2:])
    e2 = np.abs(H[:-2, 1:-1] - 2.0 * H[1:-1, 1:-1] + H[2:, 1:-1])
    error = 0.25 * float(max(e1.max(), e2.max()))
    sub = {
        k: v.reshape((n, n) + v.shape[1:])[inner].reshape((-1,) + v.shape[1:])
        for k, v in sample.items()
        if isinstance(v, np.ndarray) and v.shape[:1] == (n * n,)
    }
    xi = X[inner].reshape(-1)
    yi = Y[inner].reshape(-1)
    col = surface_colors(model, sub, xi, yi, nrm.reshape(-1, 3))
    # Stitch borders toward coarser neighbours (heights, normals, colours).
    vals = np.concatenate([h[..., None], nrm, col.reshape(r + 1, r + 1, 3)], axis=-1)
    if stitch:
        _stitch(vals, stitch, r)
        n3 = vals[..., 1:4]
        vals[..., 1:4] = n3 / np.sqrt(np.sum(n3 * n3, axis=-1, keepdims=True))
    hz = vals[..., 0].reshape(-1)
    normals = vals[..., 1:4].reshape(-1, 3)
    colors = np.clip(vals[..., 4:7].reshape(-1, 3), 0.0, 1.0)
    ox, oy = float(xs[1]), float(ys[1])
    lx = (xi - ox).astype(np.float32)
    ly = (yi - oy).astype(np.float32)
    positions = np.stack([lx, ly, hz.astype(np.float32)], axis=1)
    tris = grid_triangles(r)
    # Skirts.
    ring, skirt_tris = _skirt(r, positions.shape[0], 0)
    depth = max(0.5, 2.0 * error + 0.25 * step)
    skirt_pos = positions[ring].copy()
    skirt_pos[:, 2] -= np.float32(depth)
    positions = np.concatenate([positions, skirt_pos])
    normals_all = np.concatenate([normals, normals[ring]]).astype(np.float32)
    colors_all = np.concatenate([colors, colors[ring]])
    triangles = np.concatenate([tris, skirt_tris])
    mesh = Mesh._from_arrays(
        positions, triangles, normals_all, None, None, colors_all.astype(np.float32)
    )
    water = _water_mesh(model, sub, xi, yi, hz, r, ox, oy)
    zlo = float(
        min(
            hz.min(),
            np.nanmin(np.where(np.isnan(sub["water"]), np.inf, sub["water"]))
            if np.any(~np.isnan(sub["water"]))
            else hz.min(),
        )
    )
    zhi = float(
        max(hz.max(), np.nanmax(sub["water"]) if np.any(~np.isnan(sub["water"])) else hz.max())
    )
    heights = hz.reshape(r + 1, r + 1).copy()
    heights.flags.writeable = False
    data = TileData(
        key, (ox, oy), mesh, water, error, (zlo, zhi), heights, time.perf_counter() - t0
    )
    if keep_sample:
        data.extras["sample"] = sub
        data.extras["normals"] = normals
    return data


def stitch_tile(data: TileData, grid: Grid, mask: int) -> TileData:
    """``data`` with the borders in ``mask`` stitched toward coarser neighbours (a new mesh
    sharing the triangles; the skirts follow their border vertices)."""
    if not mask:
        return data
    r = grid.resolution
    n = (r + 1) * (r + 1)
    m = data.mesh
    pos = m.positions.copy()
    nrm = m.normals.copy()
    col = m.colors.copy()
    vals = np.concatenate(
        [pos[:n, 2:3].astype(np.float64), nrm[:n].astype(np.float64), col[:n].astype(np.float64)],
        axis=1,
    ).reshape(r + 1, r + 1, 8)
    _stitch(vals, mask, r)
    n3 = vals[..., 1:4]
    vals[..., 1:4] = n3 / np.sqrt(np.sum(n3 * n3, axis=-1, keepdims=True))
    flat = vals.reshape(n, 8)
    depth = pos[n, 2] - pos[_RING_CACHE_RING(r)[0], 2]  # (the skirt depth below its border)
    pos[:n, 2] = flat[:, 0].astype(np.float32)
    nrm[:n] = flat[:, 1:4].astype(np.float32)
    col[:n] = np.clip(flat[:, 4:8], 0.0, 1.0).astype(np.float32)
    ring = _RING_CACHE_RING(r)
    pos[n:, 2] = pos[ring, 2] + depth
    nrm[n:] = nrm[ring]
    col[n:] = col[ring]
    mesh = Mesh._from_arrays(pos, m.triangles, nrm, None, None, col)
    out = TileData(
        data.key, data.origin, mesh, data.water, data.error, data.zrange, data.heights, data.seconds
    )
    out.extras = data.extras
    return out


_RINGS: dict[int, np.ndarray] = {}


def _RING_CACHE_RING(r: int) -> np.ndarray:  # noqa: N802
    ring = _RINGS.get(r)
    if ring is None:
        ring = _RINGS[r] = _skirt(r, 0, 0)[0]
    return ring


def _water_mesh(model, sample, x, y, h, r, ox, oy) -> Mesh | None:
    """The water surface of a tile (see the module docstring), or ``None``."""
    wl = sample["water"]
    wet = ~np.isnan(wl) & (wl > h - 0.05)
    if not np.any(wet):
        return None
    n = r + 1
    W = np.where(wet, wl, -np.inf).reshape(n, n)
    j, i = np.mgrid[0:r, 0:r]
    quad = np.stack([W[j, i], W[j, i + 1], W[j + 1, i], W[j + 1, i + 1]], axis=-1)
    keep = np.isfinite(quad).any(axis=-1)
    if not np.any(keep):
        return None
    # Each vertex takes its own level, or the highest wet neighbour's (dry shore vertices).
    level = W.copy()
    pad = np.pad(W, 1, constant_values=-np.inf)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            level = np.where(
                np.isfinite(W),
                level,
                np.maximum(level, pad[1 + dj : 1 + dj + n, 1 + di : 1 + di + n]),
            )
    used = np.zeros((n, n), dtype=bool)
    jj, ii = j[keep], i[keep]
    for dj, di in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used[jj + dj, ii + di] = True
    index = np.full((n, n), -1, dtype=np.int64)
    index[used] = np.arange(int(used.sum()))
    vx = (x.reshape(n, n)[used] - ox).astype(np.float32)
    vy = (y.reshape(n, n)[used] - oy).astype(np.float32)
    vz = level[used]
    vz = np.where(np.isfinite(vz), vz, h.reshape(n, n)[used])
    positions = np.stack([vx, vy, vz.astype(np.float32)], axis=1)
    a = index[jj, ii]
    b = index[jj, ii + 1]
    c = index[jj + 1, ii]
    d = index[jj + 1, ii + 1]
    tris = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)]).astype(np.uint32)
    depth = vz - h.reshape(n, n)[used]
    sub = {"weights": sample["weights"].reshape(n, n, -1)[used]}
    col = water_colors(model, sub, np.maximum(depth, 0.0))
    normals = np.zeros_like(positions)
    normals[:, 2] = 1.0
    return Mesh._from_arrays(positions, tris, normals, None, None, col.astype(np.float32))
