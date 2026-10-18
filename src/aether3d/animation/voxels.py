"""Voxel solids of character meshes: what automatic rigging and weights measure.

* :func:`voxelize` -- a solid voxel grid of any triangle soup (open, non-manifold,
  self-intersecting or in many parts): every triangle is sampled densely (spacing at most
  a third of a voxel) to mark **surface** voxels; the **exterior** is flooded from the
  grid's border through unmarked voxels (face neighbours); everything else is **solid**.
  Overlapping parts fuse; a part with holes wider than the sampling stays a shell
  (reported), which ``close`` voxels of morphological closing can seal.
* :func:`components` -- connected components of a mask (face neighbours), by minimum
  label propagation with pointer jumping (vectorized, deterministic).
* :func:`geodesic` -- shortest-path distances **inside the solid** from sets of source
  voxels (one set per bone) to query voxels, on the 26-neighbour grid with Borgefors'
  3-4-5 chamfer weights (face, edge and corner steps; a voxel's width is 3 units, the
  metric is within 8 % of Euclidean). Distances are integers, computed by Dial's bucket
  queue, so the native kernel (``aether3d-native``) and the NumPy implementation return
  identical numbers whatever the order or threading. Paths never cross empty space: two
  limbs that touch only across air are far apart.

All grids carry a one-voxel empty border, so neighbours of solid voxels are always in
the grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import AnimationError

MAX_VOXELS = 64_000_000
UNREACHED = -1
_OFFSETS = np.array(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    np.int64,
)
_WEIGHTS = np.array([2 + int(np.count_nonzero(o)) for o in _OFFSETS], np.int64)  # 3, 4, 5


def _err(code: str, message: str, location: str = "voxels") -> AnimationError:
    return AnimationError(code, message, location)


@dataclass
class VoxelGrid:
    """A solid voxel grid: voxel ``(i, j, k)`` spans ``origin + h [i, i+1) x ...``."""

    origin: np.ndarray
    h: float
    solid: np.ndarray  # (nx, ny, nz) bool
    surface: np.ndarray  # (nx, ny, nz) bool
    stats: dict[str, Any] = field(default_factory=dict)
    _sid: np.ndarray | None = None

    @property
    def dims(self) -> tuple[int, int, int]:
        return tuple(int(d) for d in self.solid.shape)  # type: ignore[return-value]

    def flat(self, ijk: np.ndarray) -> np.ndarray:
        """Flat (Fortran-free, x fastest) indices of voxel coordinates ``(n, 3)``."""
        nx, ny, _ = self.dims
        ijk = np.asarray(ijk, np.int64)
        return ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])

    def voxel_of(self, points: np.ndarray) -> np.ndarray:
        """Voxel coordinates of points (clipped into the grid)."""
        ijk = np.floor((np.asarray(points, np.float64) - self.origin) / self.h).astype(np.int64)
        return np.clip(ijk, 0, np.array(self.dims) - 1)

    def center(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(ijk, np.float64) + 0.5) * self.h

    def solid_flat(self) -> np.ndarray:
        """``(N,)`` bool over flat indices (x fastest)."""
        return self.solid.transpose(2, 1, 0).reshape(-1)

    def sid(self) -> np.ndarray:
        """Flat grid index -> index among solid voxels (``-1``: not solid)."""
        if self._sid is None:
            mask = self.solid_flat()
            sid = np.full(mask.size, -1, np.int32)
            sid[mask] = np.arange(int(mask.sum()), dtype=np.int32)
            self._sid = sid
        return self._sid


# --------------------------------------------------------------------------------------- #
# Voxelization
# --------------------------------------------------------------------------------------- #


def voxelize(
    positions: np.ndarray,
    triangles: np.ndarray,
    *,
    h: float | None = None,
    resolution: int = 96,
    close: int = 0,
    max_voxels: int = MAX_VOXELS,
) -> VoxelGrid:
    """The solid voxel grid of a triangle soup (see the module docstring). ``h`` is the
    voxel size; by default the longest extent spans ``resolution`` voxels."""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    if p.shape[0] == 0 or t.shape[0] == 0:
        raise _err("INVALID_MESH", "nothing to voxelize (no triangles)")
    if not np.all(np.isfinite(p)):
        raise _err("INVALID_MESH", "vertex positions must be finite")
    lo, hi = p.min(axis=0), p.max(axis=0)
    extent = float(np.max(hi - lo))
    if h is None:
        if not 8 <= resolution <= 1024:
            raise _err("INVALID_ARGUMENT", "resolution is 8..1024 voxels", "resolution")
        h = extent / resolution if extent > 0 else 1.0
    h = float(h)
    if not (math.isfinite(h) and h > 0):
        raise _err("INVALID_ARGUMENT", "the voxel size must be positive", "h")
    pad = 2 + int(close)
    origin = lo - pad * h
    dims = np.floor((hi - origin) / h).astype(np.int64) + 1 + pad
    if float(np.prod(dims.astype(np.float64))) > max_voxels:
        raise _err(
            "LIMIT_EXCEEDED",
            f"{int(np.prod(dims))} voxels exceed the limit of {max_voxels} (use a coarser grid)",
            "resolution",
        )
    surface = np.zeros(tuple(int(d) for d in dims), bool)
    _mark_triangles(surface, p, t, origin, h)
    blocked = surface if close <= 0 else dilate(surface, close)
    exterior = flood_exterior(blocked)
    solid = ~exterior
    if close > 0:
        solid = erode(solid, close) | surface
    interior = int(solid.sum() - surface.sum())
    stats = {
        "voxel_size": h,
        "dims": [int(d) for d in dims],
        "surface_voxels": int(surface.sum()),
        "interior_voxels": max(0, interior),
        "closing": int(close),
        "shell_only": interior <= 0.05 * max(1, int(surface.sum())),
    }
    return VoxelGrid(origin, h, solid, surface, stats)


def _mark_triangles(grid: np.ndarray, p: np.ndarray, t: np.ndarray, origin, h) -> None:
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    longest = np.max(
        np.stack(
            [
                np.linalg.norm(b - a, axis=1),
                np.linalg.norm(c - b, axis=1),
                np.linalg.norm(a - c, axis=1),
            ],
            axis=1,
        ),
        axis=1,
    )
    steps = np.maximum(1, np.ceil(longest / (h / 3.0))).astype(np.int64)
    dims = np.array(grid.shape)
    flat = grid.reshape(-1, order="F")  # x fastest
    for m in np.unique(steps).tolist():
        sel = np.flatnonzero(steps == m)
        # The barycentric lattice with m subdivisions per edge.
        i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
        keep = (i + j) <= m
        u = (i[keep] / m).astype(np.float64)
        v = (j[keep] / m).astype(np.float64)
        chunk = max(1, 2_000_000 // max(1, u.size))
        for start in range(0, sel.size, chunk):
            s = sel[start : start + chunk]
            pts = (
                a[s, None, :] * (1.0 - u - v)[None, :, None]
                + b[s, None, :] * u[None, :, None]
                + c[s, None, :] * v[None, :, None]
            ).reshape(-1, 3)
            ijk = np.floor((pts - origin) / h).astype(np.int64)
            ijk = np.clip(ijk, 0, dims - 1)
            idx = ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])
            flat[idx] = True
    grid[...] = flat.reshape(grid.shape, order="F")


def flood_exterior(blocked: np.ndarray) -> np.ndarray:
    """Voxels reachable from the grid border through unblocked voxels (face neighbours):
    runs along each axis take the exterior label in one vectorized step, repeated until
    nothing changes."""
    free = ~blocked
    ext = np.zeros_like(free)
    ext[0, :, :] = free[0, :, :]
    ext[-1, :, :] |= free[-1, :, :]
    ext[:, 0, :] |= free[:, 0, :]
    ext[:, -1, :] |= free[:, -1, :]
    ext[:, :, 0] |= free[:, :, 0]
    ext[:, :, -1] |= free[:, :, -1]
    for _ in range(10_000):
        before = int(ext.sum())
        for axis in range(3):
            ext = _spread_runs(ext, free, axis)
        if int(ext.sum()) == before:
            break
    return ext


def _spread_runs(seed: np.ndarray, free: np.ndarray, axis: int) -> np.ndarray:
    """Every run of free voxels along ``axis`` touching a seed voxel becomes seeded."""
    f = np.moveaxis(free, axis, -1)
    s = np.moveaxis(seed, axis, -1)
    shape = f.shape
    f2 = f.reshape(-1, shape[-1])
    s2 = s.reshape(-1, shape[-1])
    # Run ids: a new run starts at every free voxel whose predecessor is not free.
    start = f2 & ~np.concatenate([np.zeros((f2.shape[0], 1), bool), f2[:, :-1]], axis=1)
    run = np.cumsum(start.reshape(-1)).reshape(f2.shape)
    run = np.where(f2, run, 0)
    hit = np.zeros(int(run.max()) + 1, bool)
    hit[run[s2 & f2]] = True
    hit[0] = False
    out = hit[run] | s2
    return np.moveaxis(out.reshape(shape), -1, axis)


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """Dilation by ``r`` steps of the 26-neighbourhood (the 3x3x3 box is separable)."""
    out = mask.copy()
    for _ in range(int(r)):
        for axis in range(3):
            out = _shift_or(out, axis)
    return out


def _shift_or(m: np.ndarray, axis: int) -> np.ndarray:
    out = m.copy()
    a = np.moveaxis(out, axis, 0)
    src = np.moveaxis(m, axis, 0)
    a[1:] |= src[:-1]
    a[:-1] |= src[1:]
    return out


def erode(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return ~dilate(~mask, r)


# --------------------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------------------- #


def label_graph(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component labels (the smallest member index) of ``n`` nodes joined by edges
    ``a``-``b``: roots hook onto smaller roots, then pointer jumping, until every edge
    joins equal labels (Shiloach-Vishkin style; vectorized and deterministic)."""
    labels = np.arange(n, dtype=np.int64)
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    if a.size == 0:
        return labels
    for _ in range(10_000):
        la, lb = labels[a], labels[b]
        differ = la != lb
        if not np.any(differ):
            break
        lo = np.minimum(la[differ], lb[differ])
        hi = np.maximum(la[differ], lb[differ])
        np.minimum.at(labels, hi, lo)
        while True:  # pointer jumping: every label becomes its root
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
    return labels


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """``(labels, count)`` of the face-connected components of ``mask``: labels are
    ``0..count-1`` (ordered by each component's first voxel), ``-1`` outside."""
    flat = mask.transpose(2, 1, 0).reshape(-1)
    nx, ny, _ = mask.shape
    idx = np.flatnonzero(flat)
    sid = np.full(flat.size, -1, np.int64)
    sid[idx] = np.arange(idx.size)
    edges_a, edges_b = [], []
    for stride in (1, nx, nx * ny):
        ok = np.zeros(idx.size, bool)
        nb = idx + stride
        inside = nb < flat.size
        ok[inside] = flat[nb[inside]]
        if stride == 1:
            ok &= (idx % nx) != nx - 1
        elif stride == nx:
            ok &= ((idx // nx) % ny) != ny - 1
        edges_a.append(np.flatnonzero(ok))
        edges_b.append(sid[nb[ok]])
    roots = label_graph(idx.size, np.concatenate(edges_a), np.concatenate(edges_b))
    uniq, compact = np.unique(roots, return_inverse=True)
    labels = np.full(flat.size, -1, np.int64)
    labels[idx] = compact
    return labels.reshape(mask.shape[::-1]).transpose(2, 1, 0), int(uniq.size)


def mesh_components(vertex_count: int, triangles: np.ndarray) -> tuple[np.ndarray, int]:
    """``(labels, count)`` of a mesh's vertices joined by its triangles' edges."""
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    a = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    b = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    roots = label_graph(vertex_count, a, b)
    uniq, compact = np.unique(roots, return_inverse=True)
    return compact, int(uniq.size)


# --------------------------------------------------------------------------------------- #
# Geodesic distances inside the solid
# --------------------------------------------------------------------------------------- #


def geodesic(
    grid: VoxelGrid,
    sources: list[np.ndarray],
    queries: np.ndarray,
    *,
    cutoff: int = 1 << 30,
) -> np.ndarray:
    """``(len(queries), len(sources))`` int32 chamfer distances (units of a third of a
    voxel) from each source set (flat voxel indices; non-solid ones are ignored) to each
    query voxel, ``-1`` when farther than ``cutoff`` or unreachable."""
    from .. import _native

    sid = grid.sid()
    nx, ny, nz = grid.dims
    ptr = np.zeros(len(sources) + 1, np.int64)
    for i, s in enumerate(sources):
        ptr[i + 1] = ptr[i] + len(s)
    flat_src = (
        np.concatenate([np.asarray(s, np.int64) for s in sources])
        if sources
        else np.zeros(0, np.int64)
    )
    q = np.ascontiguousarray(queries, np.int64)
    out = np.full((q.size, len(sources)), UNREACHED, np.int32)
    if not sources or q.size == 0:
        return out
    cutoff = int(min(max(cutoff, 0), (1 << 31) - 16))
    native = _native.module()
    if native is not None and hasattr(native, "geodesic_voxels"):
        native.geodesic_voxels(
            np.ascontiguousarray(sid, np.int32),
            nx,
            ny,
            nz,
            ptr,
            flat_src,
            q,
            cutoff,
            out,
            _native.threads(),
        )
        return out
    offsets = _OFFSETS[:, 0] + nx * (_OFFSETS[:, 1] + ny * _OFFSETS[:, 2])
    n_solid = int((sid >= 0).sum())
    qs = sid[q]
    for b in range(len(sources)):
        src = sid[flat_src[ptr[b] : ptr[b + 1]]]
        src = np.unique(src[src >= 0])
        dist = _dial(sid, src, offsets, n_solid, cutoff, grid)
        ok = qs >= 0
        d = np.full(q.size, UNREACHED, np.int64)
        d[ok] = dist[qs[ok]]
        d[d > cutoff] = UNREACHED
        out[:, b] = d.astype(np.int32)
    return out


def _dial(sid, src, offsets, n_solid, cutoff, grid) -> np.ndarray:
    """Exact integer shortest paths from ``src`` (solid indices) by Dial's buckets."""
    inf = np.iinfo(np.int64).max
    dist = np.full(n_solid, inf, np.int64)
    if src.size == 0:
        return dist
    solid_flat = np.flatnonzero(sid >= 0)  # solid index -> flat index
    dist[src] = 0
    buckets: dict[int, list[np.ndarray]] = {0: [src]}
    d = 0
    pending = 1
    while pending and d <= cutoff:
        parts = buckets.pop(d, None)
        if parts is None:
            d += 1
            continue
        pending -= len(parts)
        front = np.unique(np.concatenate(parts))
        front = front[dist[front] == d]
        if front.size:
            fflat = solid_flat[front]
            for off, w in zip(offsets.tolist(), _WEIGHTS.tolist(), strict=True):
                nd = d + w
                if nd > cutoff:
                    continue
                nb = sid[fflat + off]
                nb = nb[nb >= 0]
                better = nb[dist[nb] > nd]
                if better.size:
                    dist[better] = nd
                    buckets.setdefault(nd, []).append(better)
                    pending += 1
        d += 1
    return dist


def segment_voxels(grid: VoxelGrid, a: np.ndarray, b: np.ndarray, *, search: int = 3) -> np.ndarray:
    """Flat indices of solid voxels along the segment ``a``-``b`` (sampled at a third of
    a voxel); where the segment leaves the solid, the nearest solid voxel within
    ``search`` voxels stands in. Empty when no solid voxel is near."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / (grid.h / 3.0))) + 1)
    pts = a[None] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None]
    raw = np.floor((pts - grid.origin) / grid.h).astype(np.int64)
    inside_grid = np.all((raw >= 0) & (raw < np.array(grid.dims)), axis=1)
    if not inside_grid.any():
        return np.zeros(0, np.int64)  # (the bone is outside the grid: no source)
    ijk = np.unique(raw[inside_grid], axis=0)
    flat = grid.flat(ijk)
    solid = grid.solid_flat()
    inside = solid[flat]
    out = [flat[inside]]
    if not np.all(inside):
        nx, ny, nz = grid.dims
        r = np.arange(-search, search + 1)
        dz, dy, dx = np.meshgrid(r, r, r, indexing="ij")
        offs = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        offs = offs[np.argsort(np.linalg.norm(offs, axis=1), kind="stable")]
        for v in ijk[~inside]:
            cand = v[None] + offs
            ok = np.all((cand >= 0) & (cand < np.array([nx, ny, nz])), axis=1)
            cf = grid.flat(cand[ok])
            hit = cf[solid[cf]]
            if hit.size:
                out.append(hit[:1])
    return np.unique(np.concatenate(out))
