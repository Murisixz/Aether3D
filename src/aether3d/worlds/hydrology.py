"""The regional stage: erosion, lakes and rivers on a coarse grid over an explicit domain.

Water does not stop at chunk borders: where a river runs depends on the whole catchment
upstream. This stage therefore runs once per world definition over a bounded **domain**
(the world's extent, or ``hydrology.extent`` for an unbounded world) on a grid of
``hydrology.cell`` metres, and its results are sampled by every chunk:

1. **Base surface.** The shape height (:meth:`WorldModel.shape`) at every grid node.
2. **Erosion**, ``erosion.iterations`` times:

   * *flow routing* -- Priority-Flood with an epsilon (Barnes, Lehman and Mulla 2014)
     fills depressions so every node drains; each node's receiver is its steepest
     downhill neighbour (D8) on the filled surface; nodes are ordered by filled height,
     receivers first;
   * *drainage* -- each node's catchment, weighted by rainfall, summed downstream in
     integers (exact in any order);
   * *stream power* -- the implicit ``n = 1`` solve of Braun and Willett (2013):
     ``h_i <- (h_i + U dt + F h_r) / (1 + F)``, ``F = K dt sqrt(A_i) / L_i``, receivers
     first, so it is unconditionally stable; nodes under water erode toward the water
     level;
   * *sediment* -- eroded volume travels downstream and is deposited where it exceeds a
     transport capacity ``~ sqrt(A) S`` (floodplains, fans), entirely in lakes, and at the
     coast (deltas), in the spirit of Yuan et al. (2019);
   * *thermal erosion* -- material slides from slopes steeper than the talus angle.

3. **Water.** A final routing gives lakes (filled depressions deeper than
   ``lakes.min_depth`` and larger than ``lakes.min_area``, each with one level), the river
   network (nodes whose catchment exceeds ``rivers.threshold``; polylines from sources to
   a lake, the sea, a confluence or the domain border, with widths from hydraulic geometry
   ``w ~ sqrt(A)`` and water levels that never rise downstream) and a coarse distance to
   water.

Chunks then sample the erosion delta (``eroded - base``, bicubic, tapered to zero at the
domain border), a detail attenuation (smooth valley floors and lake beds), lake levels and
river segments (channels carved at full resolution). Everything here is deterministic; the
sequential sweeps run in the native kernels when installed (the same operations, the same
order: bitwise equal) and in the Python twins below otherwise.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _native
from ..errors import WorldError
from . import noise as nz

SQRT2 = 1.4142135623730951
OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
"""Neighbour order (di, dj) everywhere: ties resolve to the earlier neighbour."""
EROSION_RATE = 2.4e-2
"""``K dt`` per iteration at ``strength = 1``: the incision speed of the stream power law."""


def THERMAL_ITERATIONS(thermal: float) -> int:  # noqa: N802 (a documented rule, like a constant)
    """Talus relaxation sweeps per erosion step for ``erosion.thermal`` (0..1)."""
    return int(round(2 + 10 * thermal))


AGGRADATION = 2.0
"""Metres a node may rise above its height before a step through deposition (a bound for
stability; deposits are also kept below every node draining into them)."""


# --------------------------------------------------------------------------------------- #
# Kernels: native when available, Python twins otherwise (the same operations, in order)
# --------------------------------------------------------------------------------------- #


def priority_flood(h: np.ndarray, outlet: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Depression-filled heights (float64): every node not an outlet gets a strictly lower
    neighbour. Outlets (``outlet``: border and sea nodes) keep their heights."""
    h = np.ascontiguousarray(h, dtype=np.float64)
    outlet = np.ascontiguousarray(outlet, dtype=np.uint8)
    out = np.empty_like(h)
    native = _native.module()
    if native is not None and hasattr(native, "priority_flood"):
        native.priority_flood(h, outlet, nx, ny, out)
        return out
    return _priority_flood_py(h, outlet, nx, ny, out)


def _priority_flood_py(h, outlet, nx, ny, out):
    out[:] = h
    closed = outlet.astype(bool).copy()
    heap = [(float(h[c]), int(c)) for c in np.flatnonzero(closed)]
    heapq.heapify(heap)
    inf = math.inf
    filled = out
    while heap:
        z, c = heapq.heappop(heap)
        i = c % nx
        j = c // nx
        for di, dj in OFFSETS:
            a = i + di
            b = j + dj
            if a < 0 or a >= nx or b < 0 or b >= ny:
                continue
            n = b * nx + a
            if closed[n]:
                continue
            closed[n] = True
            v = filled[n]
            if v <= z:
                v = math.nextafter(z, inf)
                filled[n] = v
            heapq.heappush(heap, (v, n))
    return out


def receivers(
    filled: np.ndarray, outlet: np.ndarray, nx: int, ny: int, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each node's steepest downhill neighbour on the filled surface (itself for outlets)
    and the distance to it (vectorized; ties go to the earlier neighbour)."""
    f = filled.reshape(ny, nx)
    idx = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
    best = np.zeros((ny, nx))
    recv = idx.copy()
    dist = np.full((ny, nx), cell)
    for di, dj in OFFSETS:
        d = cell * (SQRT2 if di and dj else 1.0)
        ys = slice(max(0, -dj), ny - max(0, dj))
        xs = slice(max(0, -di), nx - max(0, di))
        yn = slice(max(0, dj), ny - max(0, -dj))
        xn = slice(max(0, di), nx - max(0, -di))
        slope = (f[ys, xs] - f[yn, xn]) / d
        better = slope > best[ys, xs]
        best[ys, xs] = np.where(better, slope, best[ys, xs])
        recv[ys, xs] = np.where(better, idx[yn, xn], recv[ys, xs])
        dist[ys, xs] = np.where(better, d, dist[ys, xs])
    recv = recv.reshape(-1)
    out = outlet.astype(bool)
    recv[out] = np.flatnonzero(out)
    return recv, dist.reshape(-1)


def accumulate(order: np.ndarray, recv: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Upstream sums of integer ``weight`` (catchments): exact, so any order agrees."""
    order = np.ascontiguousarray(order, dtype=np.int64)
    recv = np.ascontiguousarray(recv, dtype=np.int64)
    area = np.ascontiguousarray(weight, dtype=np.int64).copy()
    native = _native.module()
    if native is not None and hasattr(native, "accumulate"):
        native.accumulate(order, recv, area)
        return area
    a = area.tolist()
    r = recv.tolist()
    for c in reversed(order.tolist()):
        t = r[c]
        if t != c:
            a[t] += a[c]
    return np.asarray(a, dtype=np.int64)


def stream_power(order, recv, dist, area_m2, h, filled, kdt, uplift) -> np.ndarray:
    """One implicit stream-power step (n = 1, m = 0.5), receivers first."""
    order = np.ascontiguousarray(order, dtype=np.int64)
    recv = np.ascontiguousarray(recv, dtype=np.int64)
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    area_m2 = np.ascontiguousarray(area_m2, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    filled = np.ascontiguousarray(filled, dtype=np.float64)
    uplift = np.ascontiguousarray(uplift, dtype=np.float64)
    out = h.copy()
    native = _native.module()
    if native is not None and hasattr(native, "stream_power"):
        native.stream_power(order, recv, dist, area_m2, h, filled, float(kdt), uplift, out)
        return out
    hn = out.tolist()
    hl = h.tolist()
    fl = filled.tolist()
    rl = recv.tolist()
    dl = dist.tolist()
    al = area_m2.tolist()
    ul = uplift.tolist()
    for c in order.tolist():
        r = rl[c]
        if r == c:
            continue
        hc = hl[c] + ul[c]
        if fl[c] > hl[c]:  # under water: no incision (sediment fills it)
            hn[c] = hc
            continue
        base = fl[r] if fl[r] > hl[r] else hn[r]
        f = kdt * math.sqrt(al[c]) / dl[c]
        v = (hc + f * base) / (1.0 + f)
        hn[c] = v
    return np.asarray(hn, dtype=np.float64)


def sediment(order, recv, dist, area_m2, before, after, filled, sea, cell, deposition):
    """Carry eroded volume downstream (in place on ``after``): each node keeps the depth
    ``G Qs / A`` of the flux ``Qs`` passing it -- ``G`` (``deposition``) times the mean
    depth eroded upstream in this step, ``A`` being the catchment area (after Yuan et al.
    2019),
    rising at most to its lowest donor (deposits never dam their own inflow) and by at most
    :data:`AGGRADATION` per step; lakes keep everything up to their water level; the coast
    keeps a share (deltas). Donors are visited before their receivers."""
    order = np.ascontiguousarray(order, dtype=np.int64)
    recv = np.ascontiguousarray(recv, dtype=np.int64)
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    area_m2 = np.ascontiguousarray(area_m2, dtype=np.float64)
    before = np.ascontiguousarray(before, dtype=np.float64)
    after = np.ascontiguousarray(after, dtype=np.float64)
    filled = np.ascontiguousarray(filled, dtype=np.float64)
    native = _native.module()
    if native is not None and hasattr(native, "sediment"):
        native.sediment(
            order,
            recv,
            area_m2,
            before,
            after,
            filled,
            float(sea),
            float(cell),
            float(deposition),
            float(AGGRADATION),
        )
        return after
    a2 = cell * cell
    hb = before.tolist()
    ha = after.tolist()
    fl = filled.tolist()
    rl = recv.tolist()
    al = area_m2.tolist()
    qs = [0.0] * len(ha)
    inf = math.inf
    low = [inf] * len(ha)
    for c in reversed(order.tolist()):
        e = hb[c] - ha[c]
        q = qs[c] + (e * a2 if e > 0.0 else 0.0)
        r = rl[c]
        if q <= 0.0:
            if r != c and ha[c] < low[r]:
                low[r] = ha[c]
            continue
        if r == c:
            if ha[c] < sea:  # the coast: a share builds a delta, the rest is lost offshore
                room = (sea - 0.5 - ha[c]) * a2
                d = q * deposition
                if d > room:
                    d = room
                if d > 0.0:
                    ha[c] += d / a2
            continue
        if fl[c] > hb[c]:  # a lake (under water before the step): everything settles
            room = (fl[c] - ha[c]) * a2
            d = q if q < room else room
        else:
            d = deposition * q / al[c] * a2
            top = (hb[c] if hb[c] > ha[c] else ha[c]) + AGGRADATION
            if low[c] < top:
                top = low[c]
            room = (top - ha[c]) * a2
            if d > room:
                d = room
        if d > 0.0:
            ha[c] += d / a2
            q -= d
        qs[r] += q
        if ha[c] < low[r]:
            low[r] = ha[c]
    after[:] = ha
    return after


def thermal(
    h: np.ndarray, nx: int, ny: int, cell: float, talus_tan: float, rate: float, iterations: int
) -> np.ndarray:
    """Talus relaxation: material above the talus slope moves to lower neighbours
    (Jacobi updates, mass conserving). Each iteration, every node sends
    ``rate * max_excess / 2`` downhill, shared in proportion to each neighbour's excess
    drop (the drop beyond ``talus_tan * distance``); edges are mirrored (no flow out)."""
    h = np.ascontiguousarray(h, dtype=np.float64)
    out = np.empty_like(h)
    native = _native.module()
    if native is not None and hasattr(native, "thermal"):
        native.thermal(
            h,
            nx,
            ny,
            float(cell),
            float(talus_tan),
            float(rate),
            int(iterations),
            out,
            _native.threads(),
        )
        return out
    g = h.reshape(ny, nx).copy()
    dists = [cell * (SQRT2 if di and dj else 1.0) for di, dj in OFFSETS]
    for _ in range(iterations):
        pad = np.pad(g, 1, mode="edge")
        diffs = []
        total = np.zeros_like(g)
        dmax = np.zeros_like(g)
        for (di, dj), d in zip(OFFSETS, dists, strict=True):
            nb = pad[1 + dj : 1 + dj + ny, 1 + di : 1 + di + nx]
            excess = (g - nb) - talus_tan * d
            excess = np.where(excess > 0.0, excess, 0.0)
            diffs.append(excess)
            total = total + excess
            dmax = np.maximum(dmax, excess)
        moved = rate * 0.5 * dmax
        share = np.where(total > 0.0, moved / np.where(total > 0.0, total, 1.0), 0.0)
        incoming = np.zeros((ny + 2, nx + 2))
        for (di, dj), excess in zip(OFFSETS, diffs, strict=True):
            incoming[1 + dj : 1 + dj + ny, 1 + di : 1 + di + nx] += excess * share
        g = (g - moved) + incoming[1:-1, 1:-1]
    out[:] = g.reshape(-1)
    return out


# --------------------------------------------------------------------------------------- #
# The regional result
# --------------------------------------------------------------------------------------- #


@dataclass
class Regional:
    """The regional stage's results and their sampling at full resolution."""

    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    delta_grid: np.ndarray  # (ny, nx) eroded - base
    atten_grid: np.ndarray  # (ny, nx) detail attenuation 0..1
    lake_label: np.ndarray  # (ny, nx) int32, -1 none (dilated by one node)
    lake_levels: np.ndarray  # (lakes,) float64
    distance_grid: np.ndarray  # (my, mx) distance to water, coarse
    distance_cell: float
    rivers: list[dict]  # polylines: points (K, 2), level (K,), width (K,)
    segments: np.ndarray  # (S, 8): x0 y0 x1 y1 level0 level1 width0 width1
    flow_grid: np.ndarray  # (ny, nx) catchment area in m^2 (float64)
    sea: float
    depth: float
    stats: dict = field(default_factory=dict)
    _buckets: dict | None = None
    _bucket_size: float = 256.0

    # -- grid sampling ---------------------------------------------------------------

    def _coords(self, x: np.ndarray, y: np.ndarray):
        gx = (x - self.x0) / self.cell
        gy = (y - self.y0) / self.cell
        return gx, gy

    def _taper(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """1 inside the domain, falling to 0 over its outer 8 cells (beyond: 0)."""
        edge = np.minimum(np.minimum(gx, self.nx - 1 - gx), np.minimum(gy, self.ny - 1 - gy))
        return nz.smoothstep(0.0, 8.0, edge)

    def delta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The erosion delta at points: Catmull-Rom bicubic (C1), tapered at the border."""
        gx, gy = self._coords(x, y)
        inside = (gx >= 0) & (gy >= 0) & (gx <= self.nx - 1) & (gy <= self.ny - 1)
        out = np.zeros(x.shape)
        if not np.any(inside):
            return out
        gxi, gyi = gx[inside], gy[inside]
        val = _bicubic(self.delta_grid, gxi, gyi)
        out[inside] = val * self._taper(gxi, gyi)
        return out

    def _bilinear(self, grid: np.ndarray, gx: np.ndarray, gy: np.ndarray, outside: float):
        ny, nx = grid.shape
        inside = (gx >= 0) & (gy >= 0) & (gx <= nx - 1) & (gy <= ny - 1)
        out = np.full(gx.shape, outside, dtype=np.float64)
        if np.any(inside):
            out[inside] = _bilinear(grid, gx[inside], gy[inside])
        return out

    def attenuation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx, gy = self._coords(x, y)
        a = self._bilinear(self.atten_grid, gx, gy, 1.0)
        t = self._taper(np.clip(gx, -1e9, 1e9), np.clip(gy, -1e9, 1e9))
        inside = (gx >= 0) & (gy >= 0) & (gx <= self.nx - 1) & (gy <= self.ny - 1)
        return np.where(inside, 1.0 + t * (a - 1.0), 1.0)

    def water_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = (x - self.x0) / self.distance_cell
        gy = (y - self.y0) / self.distance_cell
        return self._bilinear(self.distance_grid, gx, gy, 1e6)

    def flow(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Catchment area (m^2) at points (bilinear; 0 outside the domain)."""
        gx, gy = self._coords(x, y)
        return self._bilinear(self.flow_grid, gx, gy, 0.0)

    def lake_level(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx, gy = self._coords(x, y)
        ix = np.floor(gx + 0.5).astype(np.int64)
        iy = np.floor(gy + 0.5).astype(np.int64)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.nx) & (iy < self.ny)
        out = np.full(x.shape, np.nan)
        if not np.any(ok) or not len(self.lake_levels):
            return out
        lab = self.lake_label[iy[ok], ix[ok]]
        has = lab >= 0
        vals = np.full(lab.shape, np.nan)
        vals[has] = self.lake_levels[lab[has]]
        out[ok] = vals
        return out

    # -- rivers ----------------------------------------------------------------------

    def _index(self) -> dict:
        if self._buckets is None:
            b: dict[tuple[int, int], list[int]] = {}
            size = self._bucket_size
            seg = self.segments
            for k in range(seg.shape[0]):
                x0, y0, x1, y1 = seg[k, :4]
                w = max(seg[k, 6], seg[k, 7]) * 2.5 + 8.0
                for bx in range(
                    int(math.floor((min(x0, x1) - w) / size)),
                    int(math.floor((max(x0, x1) + w) / size)) + 1,
                ):
                    for by in range(
                        int(math.floor((min(y0, y1) - w) / size)),
                        int(math.floor((max(y0, y1) + w) / size)) + 1,
                    ):
                        b.setdefault((bx, by), []).append(k)
            self._buckets = b
        return self._buckets

    def segments_near(self, box: tuple[float, float, float, float]) -> np.ndarray:
        size = self._bucket_size
        idx: set[int] = set()
        b = self._index()
        for bx in range(int(math.floor(box[0] / size)), int(math.floor(box[2] / size)) + 1):
            for by in range(int(math.floor(box[1] / size)), int(math.floor(box[3] / size)) + 1):
                idx.update(b.get((bx, by), ()))
        return np.array(sorted(idx), dtype=np.int64)

    def carve_rivers(self, x: np.ndarray, y: np.ndarray, h: np.ndarray):
        """Channels carved into ``h``: ``(h, distance to centre in half-widths, level)``.

        Within a channel (``d < w/2``) the bed is a parabola down to ``depth`` below the
        water level; the banks (``w/2 .. w``) are pulled down to the water level plus a
        rise, so the river never runs in a slot. ``level`` is ``nan`` outside channels."""
        rel = np.full(x.shape, np.inf)
        level = np.full(x.shape, np.nan)
        if not len(x) or not self.segments.shape[0]:
            return h, rel, level
        box = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        near = self.segments_near(box)
        if not near.size:
            return h, rel, level
        seg = self.segments[near]
        best = np.full(x.shape, np.inf)
        lev = np.zeros(x.shape)
        wid = np.ones(x.shape)
        for k in range(seg.shape[0]):
            ax, ay, bx, by, l0, l1, w0, w1 = seg[k]
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            rx = x - ax
            ry = y - ay
            t = np.clip((rx * dx + ry * dy) / ll, 0.0, 1.0) if ll > 0 else np.zeros(x.shape)
            qx = rx - t * dx
            qy = ry - t * dy
            d = np.sqrt(qx * qx + qy * qy)
            w = w0 + t * (w1 - w0)
            r = d / (0.5 * w)
            closer = r < best
            best = np.where(closer, r, best)
            lev = np.where(closer, l0 + t * (l1 - l0), lev)
            wid = np.where(closer, w, wid)
        depth = self.depth * np.clip(wid / 8.0, 0.6, 3.0)
        inside = best < 1.0
        u = np.clip(best, 0.0, 1.0)
        bed = lev - 0.25 - depth * (1.0 - u * u)
        h = np.where(inside, np.minimum(h, bed), h)
        bank = (best >= 1.0) & (best < 2.5)
        if np.any(bank):
            t = (best - 1.0) / 1.5
            s = t * t * (3.0 - 2.0 * t)
            target = lev + 0.15 + s * np.maximum(h - lev - 0.15, 0.0)
            h = np.where(bank & (h > target), target, h)
        rel = best
        level = np.where(inside, lev, np.nan)
        return h, rel, level

    # -- reports ---------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "domain": [
                self.x0,
                self.y0,
                self.x0 + (self.nx - 1) * self.cell,
                self.y0 + (self.ny - 1) * self.cell,
            ],
            "cell": self.cell,
            "nodes": self.nx * self.ny,
            "rivers": len(self.rivers),
            "river_length_km": round(
                sum(float(np.sum(np.hypot(*np.diff(r["points"], axis=0).T))) for r in self.rivers)
                / 1000.0,
                3,
            ),
            "lakes": int(len(self.lake_levels)),
            **self.stats,
        }

    @property
    def nbytes(self) -> int:
        return int(
            self.delta_grid.nbytes
            + self.atten_grid.nbytes
            + self.lake_label.nbytes
            + self.distance_grid.nbytes
            + self.segments.nbytes
            + self.flow_grid.nbytes
        )


def _bilinear(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    ny, nx = grid.shape
    ix = np.clip(np.floor(gx).astype(np.int64), 0, max(nx - 2, 0))
    iy = np.clip(np.floor(gy).astype(np.int64), 0, max(ny - 2, 0))
    fx = gx - ix
    fy = gy - iy
    ix1 = np.minimum(ix + 1, nx - 1)
    iy1 = np.minimum(iy + 1, ny - 1)
    a = grid[iy, ix] + fx * (grid[iy, ix1] - grid[iy, ix])
    b = grid[iy1, ix] + fx * (grid[iy1, ix1] - grid[iy1, ix])
    return a + fy * (b - a)


def _catmull(p0, p1, p2, p3, t):
    return 0.5 * (
        2.0 * p1
        + t * (p2 - p0)
        + t * t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
        + t * t * t * (3.0 * p1 - p0 - 3.0 * p2 + p3)
    )


def _bicubic(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    ny, nx = grid.shape
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    fx = gx - ix
    fy = gy - iy
    rows = []
    for dj in (-1, 0, 1, 2):
        jj = np.clip(iy + dj, 0, ny - 1)
        cols = [grid[jj, np.clip(ix + di, 0, nx - 1)] for di in (-1, 0, 1, 2)]
        rows.append(_catmull(cols[0], cols[1], cols[2], cols[3], fx))
    return _catmull(rows[0], rows[1], rows[2], rows[3], fy)


# --------------------------------------------------------------------------------------- #
# Computing the regional stage
# --------------------------------------------------------------------------------------- #


def domain_of(model: Any) -> tuple[float, float, float, float]:
    hy = model.d["hydrology"]
    if hy["extent"] is not None:
        return tuple(hy["extent"])  # type: ignore[return-value]
    e = model.extent()
    assert e is not None
    return e


def grid_size(model: Any) -> tuple[int, int, float, float, float]:
    """``(nx, ny, x0, y0, cell)`` of the regional grid (nodes on multiples of the cell)."""
    x0, y0, x1, y1 = domain_of(model)
    cell = model.d["hydrology"]["cell"]
    gx0 = math.floor(x0 / cell) * cell
    gy0 = math.floor(y0 / cell) * cell
    nx = int(math.ceil((x1 - gx0) / cell)) + 1
    ny = int(math.ceil((y1 - gy0) / cell)) + 1
    return nx, ny, gx0, gy0, cell


def estimate(model: Any) -> dict[str, Any]:
    """What the regional stage would cost: nodes, bytes, and whether native kernels run."""
    if model.d["hydrology"]["mode"] != "drainage":
        return {"mode": model.d["hydrology"]["mode"], "nodes": 0}
    nx, ny, *_ = grid_size(model)
    n = nx * ny
    return {
        "mode": "drainage",
        "grid": [nx, ny],
        "nodes": n,
        "bytes": n * 8 * 12,
        "iterations": model.d["hydrology"]["erosion"]["iterations"],
        "native": _native.module() is not None,
    }


def compute_regional(model: Any, ctx: Any) -> Regional:
    d = model.d
    hy = d["hydrology"]
    nx, ny, x0, y0, cell = grid_size(model)
    n = nx * ny
    if n > hy["max_cells"]:
        raise WorldError(
            "LIMIT_EXCEEDED",
            f"the hydrology grid would have {n:,} nodes ({nx} x {ny}); the limit is "
            f"{hy['max_cells']:,}",
            "definition.hydrology.cell",
            hint="raise hydrology.cell, shrink the world or hydrology.extent, or raise "
            "hydrology.max_cells",
        )
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    ctx.check("regional stage")
    gx = x0 + np.arange(nx, dtype=np.float64) * cell
    gy = y0 + np.arange(ny, dtype=np.float64) * cell
    X, Y = np.meshgrid(gx, gy)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    base = np.empty(n)
    rows = max(1, 65536 // nx)
    for j0 in range(0, ny, rows):
        ctx.check("regional stage")
        sl = slice(j0 * nx, min(ny, j0 + rows) * nx)
        base[sl] = model.shape_parts(X[sl], Y[sl])[0]  # the macro surface
    timings["base"] = time.perf_counter() - t_start
    sea = model.sea
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border = border.reshape(-1)
    # Rainfall weights (integers, so drainage sums are exact): 1..1000 per node.
    rain = _rainfall(model, X, Y, base)
    er = hy["erosion"]
    iterations = er["iterations"]
    kdt = EROSION_RATE * er["strength"]
    talus_tan = _tan_deg(er["talus"])
    uplift = np.zeros(n)
    if er["uplift"] > 0.0:
        uplift = er["uplift"] * np.clip((base - sea) / max(1.0, float(np.ptp(base))), 0.0, 1.0)
    # Depressions of the noise surface are filled before erosion (sediment would fill them
    # over geological time), except the largest, which stay lakes at part of their depth.
    h, kept = _prefill(base, border | (base < sea), nx, ny, cell, hy["lakes"])
    area_unit = cell * cell / 1000.0
    ones = np.ones(n, dtype=np.int64)
    t0 = time.perf_counter()
    for it in range(iterations):
        ctx.check("regional stage (erosion)")
        outlet = border | (h < sea)
        filled = priority_flood(h, outlet, nx, ny)
        recv, dist = receivers(filled, outlet, nx, ny, cell)
        order = np.argsort(filled, kind="stable")
        area = accumulate(order, recv, rain).astype(np.float64) * area_unit
        before = h
        h = stream_power(order, recv, dist, area, h, filled, kdt, uplift)
        if er["deposition"] > 0.0:
            catchment = accumulate(order, recv, ones).astype(np.float64) * (cell * cell)
            h = sediment(
                order, recv, dist, catchment, before, h, filled, sea, cell, er["deposition"]
            )
        if er["thermal"] > 0.0:
            h = thermal(h, nx, ny, cell, talus_tan, 1.0, THERMAL_ITERATIONS(er["thermal"]))
        if ctx.progress is not None:
            ctx.progress("erosion", it + 1, iterations)
    timings["erosion"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    outlet = border | (h < sea)
    filled = priority_flood(h, outlet, nx, ny)
    recv, dist = receivers(filled, outlet, nx, ny, cell)
    order = np.argsort(filled, kind="stable")
    area = accumulate(order, recv, rain).astype(np.float64) * area_unit
    # Lakes: filled depressions, labelled, with one level each.
    lakes = hy["lakes"]
    under = (filled - h) > (lakes["min_depth"] if lakes["enabled"] else np.inf)
    label, levels = _label_lakes(under, filled, nx, ny, cell, lakes["min_area"])
    # Rivers.
    rv = hy["rivers"]
    rivers, segments = _rivers(
        h, filled, recv, order, area, label, outlet, nx, ny, x0, y0, cell, rv, sea
    )
    timings["water"] = time.perf_counter() - t0
    # Coarse distance to water, detail attenuation.
    t0 = time.perf_counter()
    water = (h < sea) | (label >= 0)
    dist_grid, dist_cell = _water_distance(water.reshape(ny, nx), segments, x0, y0, cell)
    atten = _attenuation(area.reshape(ny, nx), (filled - h).reshape(ny, nx), rv["threshold"])
    dilated = _dilate_labels(label.reshape(ny, nx))
    timings["fields"] = time.perf_counter() - t0
    delta = (h - base).reshape(ny, nx)
    for grid in (delta, atten, dilated, dist_grid):
        grid.flags.writeable = False
    stats = {
        "seconds": round(time.perf_counter() - t_start, 3),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "native": _native.module() is not None and hasattr(_native.module(), "stream_power"),
        "iterations": iterations,
        "kept_lakes": kept,
        "eroded_m3": float(np.sum(np.maximum(base - h, 0.0)) * cell * cell),
        "deposited_m3": float(np.sum(np.maximum(h - base, 0.0)) * cell * cell),
    }
    return Regional(
        x0=x0,
        y0=y0,
        cell=cell,
        nx=nx,
        ny=ny,
        delta_grid=delta,
        atten_grid=atten,
        lake_label=dilated,
        lake_levels=levels,
        distance_grid=dist_grid,
        distance_cell=dist_cell,
        rivers=rivers,
        segments=segments,
        flow_grid=area.reshape(ny, nx),
        sea=sea,
        depth=rv["depth"],
        stats=stats,
    )


def _prefill(base, outlet, nx, ny, cell, lakes):
    """``base`` with its depressions filled, except the ``lakes.max`` largest (by volume)
    of at least ``min_area`` and ``min_depth``, which are filled to 40 % of their depth."""
    filled = priority_flood(base, outlet, nx, ny)
    depth = filled - base
    if not lakes["enabled"] or lakes["max"] <= 0:
        return filled, 0
    under = depth > 1e-6
    label, _ = _label_lakes(under, filled, nx, ny, cell, 0.0)
    count = int(label.max()) + 1 if label.size else 0
    if count == 0:
        return filled, 0
    lab = label[label >= 0]
    area = np.bincount(lab, minlength=count) * (cell * cell)
    dmax = np.zeros(count)
    np.maximum.at(dmax, lab, depth[label >= 0])
    volume = np.bincount(lab, weights=depth[label >= 0], minlength=count)
    ok = (area >= lakes["min_area"]) & (dmax >= lakes["min_depth"])
    rank = np.argsort(-volume, kind="stable")
    keep = [int(k) for k in rank if ok[k]][: lakes["max"]]
    out = filled.copy()
    for k in keep:
        sel = label == k
        level = float(np.max(filled[sel]))
        floor = level - 0.4 * (level - base[sel])
        out[sel] = np.maximum(base[sel], floor)
    return out, len(keep)


def _tan_deg(deg: float) -> float:
    c, s = nz.cos_sin_deg(deg)
    return s / c


def _rainfall(model: Any, x: np.ndarray, y: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Integer rain weights (1..1000) from the climate's moisture field without its
    water-distance term (which needs this stage's results)."""
    c = model.d["climate"]
    fs = 1.0 / c["scale"]
    m = (
        c["moisture"]
        + (c["moisture_gradient"][0] * x + c["moisture_gradient"][1] * y) / 1000.0
        + c["moisture_variation"]
        * 1.6
        * nz.fbm2(model.seeds["moisture"], x, y, octaves=3, frequency=fs)
    )
    m = np.clip(m, 0.0, 1.0)
    return (1.0 + np.floor(m * 999.0)).astype(np.int64)


def _label_lakes(under, filled, nx, ny, cell, min_area):
    """Connected lake nodes (8-neighbourhood, breadth-first in index order), their levels
    (the highest filled height of each), and -1 elsewhere; lakes below ``min_area`` dropped."""
    label = np.full(nx * ny, -1, dtype=np.int32)
    levels: list[float] = []
    cand = np.flatnonzero(under)
    if not cand.size:
        return label, np.zeros(0)
    seen = np.zeros(nx * ny, dtype=bool)
    under_l = under.tolist()
    f = filled
    next_id = 0
    for start in cand.tolist():
        if seen[start]:
            continue
        comp = [start]
        seen[start] = True
        k = 0
        while k < len(comp):
            c = comp[k]
            k += 1
            i = c % nx
            j = c // nx
            for di, dj in OFFSETS:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny:
                    m = b * nx + a
                    if under_l[m] and not seen[m]:
                        seen[m] = True
                        comp.append(m)
        if len(comp) * cell * cell < min_area:
            continue
        idx = np.asarray(comp, dtype=np.int64)
        label[idx] = next_id
        levels.append(float(np.max(f[idx])))
        next_id += 1
    return label, np.asarray(levels, dtype=np.float64)


def _dilate_labels(label: np.ndarray) -> np.ndarray:
    out = label.copy()
    ny, nx = label.shape
    for di, dj in OFFSETS:
        ys = slice(max(0, -dj), ny - max(0, dj))
        xs = slice(max(0, -di), nx - max(0, di))
        yn = slice(max(0, dj), ny - max(0, -dj))
        xn = slice(max(0, di), nx - max(0, -di))
        src = label[yn, xn]
        take = (out[ys, xs] < 0) & (src >= 0)
        out[ys, xs] = np.where(take, src, out[ys, xs])
    return out


def _rivers(h, filled, recv, order, area, label, outlet, nx, ny, x0, y0, cell, rv, sea):
    """River polylines (sources to mouths/confluences) and their segment table."""
    threshold = rv["threshold"]
    is_river = (area >= threshold) & (label < 0) & ~outlet & (filled <= h + 1e-9)
    nodes = np.flatnonzero(is_river)
    if not nodes.size:
        return [], np.zeros((0, 8))
    recv_l = recv.tolist()
    river_l = is_river.tolist()
    # Sources: river nodes with no river donor.
    has_donor = np.zeros(nx * ny, dtype=bool)
    donors = recv[nodes]
    has_donor[donors[is_river[donors]]] = True
    sources = [int(c) for c in nodes if not has_donor[c]]
    # Longest catchments first, so main stems are long polylines.
    sources.sort(key=lambda c: (-float(filled[c]), c))
    visited = np.zeros(nx * ny, dtype=bool)
    rivers = []
    w_coef = rv["width"]
    w_max = rv["max_width"]
    for s in sources:
        path = [s]
        c = s
        visited[c] = True
        while True:
            r = recv_l[c]
            if r == c:
                break
            path.append(r)
            if not river_l[r] or visited[r]:
                break
            visited[r] = True
            c = r
        if len(path) < 3:
            continue
        idx = np.asarray(path, dtype=np.int64)
        px = x0 + (idx % nx) * cell
        py = y0 + (idx // nx) * cell
        lev = np.minimum.accumulate(np.asarray(filled[idx], dtype=np.float64) - 0.3)
        last = int(idx[-1])
        if label[last] >= 0:
            lev[-1] = min(lev[-1], float(filled[last]))
        if h[last] < sea:
            lev[-1] = min(lev[-1], sea)
        lev = np.minimum.accumulate(lev)
        wid = np.clip(w_coef * np.sqrt(area[idx]), 1.2, w_max)
        pts = np.stack([px, py], axis=1)
        pts, lev, wid = _chaikin(pts, lev, wid, 2)
        rivers.append({"points": pts, "level": lev, "width": wid})
    seg_rows = []
    for r in rivers:
        p, lv, w = r["points"], r["level"], r["width"]
        seg_rows.append(np.column_stack([p[:-1], p[1:], lv[:-1], lv[1:], w[:-1], w[1:]]))
    segments = np.vstack(seg_rows) if seg_rows else np.zeros((0, 8))
    return rivers, segments


def _chaikin(pts, lev, wid, rounds):
    """Corner cutting (endpoints kept); levels and widths follow linearly (levels stay
    non-increasing: convex combinations of a non-increasing sequence)."""
    for _ in range(rounds):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        ql = 0.75 * lev[:-1] + 0.25 * lev[1:]
        rl = 0.25 * lev[:-1] + 0.75 * lev[1:]
        qw = 0.75 * wid[:-1] + 0.25 * wid[1:]
        rw = 0.25 * wid[:-1] + 0.75 * wid[1:]
        mid_p = np.empty((2 * len(q), 2))
        mid_p[0::2], mid_p[1::2] = q, r
        mid_l = np.empty(2 * len(q))
        mid_l[0::2], mid_l[1::2] = ql, rl
        mid_w = np.empty(2 * len(q))
        mid_w[0::2], mid_w[1::2] = qw, rw
        pts = np.vstack([pts[:1], mid_p[1:-1], pts[-1:]])
        lev = np.concatenate([lev[:1], mid_l[1:-1], lev[-1:]])
        wid = np.concatenate([wid[:1], mid_w[1:-1], wid[-1:]])
        lev = np.minimum.accumulate(lev)
    return pts, lev, wid


def _water_distance(water, segments, x0, y0, cell):
    """Distance to water on a grid of at least 64 m (chamfer relaxation), river nodes
    marked from the river segments."""
    ny, nx = water.shape
    factor = max(1, int(math.ceil(64.0 / cell)))
    my = (ny - 1) // factor + 1
    mx = (nx - 1) // factor + 1
    coarse = water[::factor, ::factor][:my, :mx].copy()
    dcell = cell * factor
    if segments.shape[0]:
        for k in range(segments.shape[0]):
            for t in (0.0, 0.5, 1.0):
                px = segments[k, 0] + t * (segments[k, 2] - segments[k, 0])
                py = segments[k, 1] + t * (segments[k, 3] - segments[k, 1])
                i = int(round((px - x0) / dcell))
                j = int(round((py - y0) / dcell))
                if 0 <= i < mx and 0 <= j < my:
                    coarse[j, i] = True
    big = 1e9
    dist = np.where(coarse, 0.0, big)
    for _ in range(max(mx, my)):
        pad = np.pad(dist, 1, mode="constant", constant_values=big)
        best = dist
        for di, dj in OFFSETS:
            step = dcell * (SQRT2 if di and dj else 1.0)
            best = np.minimum(best, pad[1 + dj : 1 + dj + my, 1 + di : 1 + di + mx] + step)
        if np.array_equal(best, dist):
            break
        dist = best
    return dist, dcell


def _attenuation(area, depth, threshold):
    """Detail attenuation: 1 on hillsides, toward 0.4 on valley floors carrying large flow
    and 0 under lakes (smooth beds)."""
    flow = np.sqrt(np.maximum(area, 0.0) / threshold)
    a = 1.0 - 0.6 * nz.smoothstep(0.15, 1.0, flow)
    return np.where(depth > 0.5, 0.0, a)
