"""A world's generated data: compiled models, the regional stage, tiles, queries, budgets.

:class:`WorldRuntime` is a cache, never the source of truth: everything in it is
recomputed from the world's definition and edits when missing, and evicting it loses
nothing authored. Entries are keyed by **content**, so a change regenerates only what
depends on it:

* the *model* (:class:`~.fields.WorldModel`) by the definition's digest;
* the *regional stage* by the parts of the definition it reads (seed, generator, extent,
  sea level, terrain, shape-stage features, hydrology, the moisture inputs of rainfall);
* a *tile* by ``(generator, terrain-global key, tile, stitch mask, the final-stage features
  touching the tile, the regional data under the tile)``: adding a road regenerates the
  tiles it crosses, and a shape feature that changes erosion somewhere regenerates the
  tiles where the regional data changed.

Memory is bounded by ``streaming.memory_mb`` (least recently used tiles go first) and every
generation step checks the context's cancellation and deadline. :meth:`stats` separates the
logical extent, generated, cached and drawn tiles, and resident bytes.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from ..errors import WorldError
from . import definition as D
from .fields import Context, WorldModel
from .terrain_mesh import TileData, build_tile, stitch_tile
from .tiles import Grid, TileKey, View, select, stitch_masks

LOCAL_MARGIN = 128.0
"""Metres around a tile within which final-stage features (and the regional data) count as
touching it: the climate lattice spacing, normals and feature fall-offs fit inside."""


def _global_key(d: dict) -> str:
    """The definition parts every terrain tile depends on (final-stage features excluded:
    they are local)."""
    skip = {"features", "vegetation", "structures", "materials", "streaming"}
    parts = {k: v for k, v in d.items() if k not in skip}
    parts["tiles"] = [d["streaming"]["chunk"], d["streaming"]["resolution"]]
    parts["shape_features"] = [f for f in d["features"] if f["type"] in D.SHAPE_STAGE]
    return D.digest(parts)


def _regional_key(d: dict) -> str:
    c = d["climate"]
    return D.digest(
        {k: d[k] for k in ("seed", "generator", "extent", "sea_level", "terrain", "hydrology")},
        [f for f in d["features"] if f["type"] in D.SHAPE_STAGE],
        [c["scale"], c["moisture"], c["moisture_variation"], c["moisture_gradient"]],
    )


class WorldRuntime:
    """See the module docstring."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self._lock = threading.RLock()
        self._models: OrderedDict[str, WorldModel] = OrderedDict()
        self._regionals: OrderedDict[str, Any] = OrderedDict()
        self._tiles: OrderedDict[str, TileData] = OrderedDict()
        self._stitched: OrderedDict[tuple, tuple[TileData, TileData]] = OrderedDict()
        self._holes: OrderedDict[tuple, TileData] = OrderedDict()
        self._vegetation: Any = None  # vegetation.Vegetation, made on first use
        self._volumes: Any = None  # volumes.Volumes
        self._blocks: Any = None  # voxels.BlockWorld
        self._tile_bytes = 0
        self.counters = {
            "tiles_generated": 0,
            "tile_hits": 0,
            "tiles_evicted": 0,
            "generation_seconds": 0.0,
            "regional_computed": 0,
        }
        self.last_frame: dict[str, Any] = {}

    # -- compiled state ----------------------------------------------------------------

    @property
    def definition(self) -> dict:
        return self.world._definition

    def model(self) -> WorldModel:
        d = self.definition
        key = D.digest(d)
        with self._lock:
            m = self._models.get(key)
            if m is None:
                m = WorldModel(d)
                rk = _regional_key(d)
                reg = self._regionals.get(rk)
                if reg is not None:
                    m.attach_regional(reg)
                self._models[key] = m
                while len(self._models) > 2:
                    self._models.popitem(last=False)
            else:
                self._models.move_to_end(key)
            return m

    def regional(self, ctx: Context | None = None) -> Any:
        m = self.model()
        if m.d["hydrology"]["mode"] != "drainage":
            return None
        if m.regional_if_ready() is None:
            reg = m.regional(ctx or Context())
            with self._lock:
                self._regionals[_regional_key(m.d)] = reg
                while len(self._regionals) > 2:
                    self._regionals.popitem(last=False)
                self.counters["regional_computed"] += 1
        return m.regional_if_ready()

    def regional_estimate(self) -> dict[str, Any]:
        from .hydrology import estimate

        out = estimate(self.model())
        reg = self.model().regional_if_ready()
        out["computed"] = reg is not None
        if reg is not None:
            out["result"] = reg.summary()
        return out

    def grid(self) -> Grid:
        s = self.definition["streaming"]
        return Grid(float(s["chunk"]), int(s["resolution"]))

    def max_level(self) -> int:
        d = self.definition
        s = d["streaming"]
        e = self.world.extent
        span = s["view_distance"] * 2.0 if e is None else max(e[2] - e[0], e[3] - e[1])
        return max(0, math.ceil(math.log2(max(span, s["chunk"]) / s["chunk"])))

    # -- tiles -------------------------------------------------------------------------

    def tile_content_key(self, key: TileKey, stitch: int, ctx: Context | None = None) -> str:
        d = self.definition
        grid = self.grid()
        x0, y0, x1, y1 = grid.bounds(key)
        box = (x0 - LOCAL_MARGIN, y0 - LOCAL_MARGIN, x1 + LOCAL_MARGIN, y1 + LOCAL_MARGIN)
        m = self.model()
        local = [c.spec for c in m.final_features if c.touches(box)]
        h = hashlib.sha256()
        h.update(_global_key(d).encode())
        h.update(f"{key.level}/{key.i}/{key.j}/{stitch}".encode())
        h.update(D.canonical(local))
        reg = self.regional(ctx) if d["hydrology"]["mode"] == "drainage" else None
        if reg is not None:
            h.update(_regional_window(reg, box))
        return h.hexdigest()[:32]

    def tile(self, key: TileKey, stitch: int = 0, ctx: Context | None = None) -> TileData:
        """A tile (generated once; stitched variants derive from the unstitched one)."""
        base = self._tile(key, ctx)
        if not stitch:
            return base
        ck = (id(base), stitch)
        with self._lock:
            found = self._stitched.get(ck)
            if found is not None and found[0] is base:
                self._stitched.move_to_end(ck)
                return found[1]
        data = stitch_tile(base, self.grid(), stitch)
        with self._lock:
            self._stitched[ck] = (base, data)
            while len(self._stitched) > 4096:
                self._stitched.popitem(last=False)
        return data

    def _tile(self, key: TileKey, ctx: Context | None = None) -> TileData:
        ck = self.tile_content_key(key, 0, ctx)
        with self._lock:
            found = self._tiles.get(ck)
            if found is not None:
                self._tiles.move_to_end(ck)
                self.counters["tile_hits"] += 1
                return found
        if ctx is not None:
            ctx.check("tile generation")
        t0 = time.perf_counter()
        data = build_tile(self.model(), self.grid(), key, 0, ctx)
        with self._lock:
            self.counters["tiles_generated"] += 1
            self.counters["generation_seconds"] += time.perf_counter() - t0
            if ck not in self._tiles:
                self._tiles[ck] = data
                self._tile_bytes += data.nbytes
                self._evict()
        return data

    def _evict(self, keep: set[str] | None = None) -> None:
        budget = self.definition["streaming"]["memory_mb"] * 2**20
        while self._tile_bytes > budget and len(self._tiles) > 1:
            ck, data = next(iter(self._tiles.items()))
            if keep is not None and ck in keep:
                break
            self._tiles.popitem(last=False)
            self._tile_bytes -= data.nbytes
            self.counters["tiles_evicted"] += 1
            for sk in [k for k, v in self._stitched.items() if v[0] is data]:
                del self._stitched[sk]

    def select(
        self, view: View, ctx: Context | None = None, holes: set | None = None
    ) -> list[tuple[TileKey, int, TileData]]:
        """The tiles to draw from a view: LOD selection (generating tiles to measure
        their error), restriction, stitching; then each drawn tile (with its stitch mask).
        Tiles over ``holes`` (volume footprints ``(ci, cj)``, drawn by volumes) are refined
        to level 0 and drawn without those footprints. Bounded by ``streaming.max_tiles``."""
        d = self.definition
        s = d["streaming"]
        grid = self.grid()
        e = self.world.extent
        boxes = [] if not holes else [_footprint(h) for h in sorted(holes)]

        def force(key: TileKey) -> bool:
            if not boxes or key.level == 0:
                return False
            x0, y0, x1, y1 = grid.bounds(key)
            return any(b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0 for b in boxes)

        keys = select(
            grid,
            view,
            lambda k: self._tile(k, ctx).error,
            lambda k: self._tile(k, ctx).zrange,
            s["lod_error"],
            e,
            self.max_level(),
            s["max_tiles"],
            force if boxes else None,
        )
        masks = stitch_masks(keys)
        out = []
        for k in keys:
            td = self.tile(k, masks[k], ctx)
            if boxes:
                x0, y0, x1, y1 = grid.bounds(k)
                mine = [b for b in boxes if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0]
                if mine:
                    td = self._holed(k, masks[k], td, mine)
            out.append((k, masks[k], td))
        return out

    def _holed(self, key: TileKey, mask: int, td: TileData, boxes: list) -> TileData:
        """A drawn tile without the volume footprints ``boxes`` (cached)."""
        from dataclasses import replace

        from .terrain_mesh import cut_holes

        ck = (self.tile_content_key(key, mask), tuple(boxes))
        with self._lock:
            found = self._holes.get(ck)
            if found is not None:
                self._holes.move_to_end(ck)
                return found
        out = replace(td, mesh=cut_holes(td.mesh, td.origin, boxes))
        with self._lock:
            self._holes[ck] = out
            while len(self._holes) > 1024:
                self._holes.popitem(last=False)
        return out

    # -- queries -----------------------------------------------------------------------

    def heights(self, x: np.ndarray, y: np.ndarray, ctx: Context | None = None) -> np.ndarray:
        self._check_points(x, y)
        self.regional(ctx)
        return self.model().height(x, y, ctx)

    def ground(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.definition["mode"] == "blocks":
            from .voxels import column_ground

            return column_ground(self, x, y)
        return self.heights(x, y)

    def _check_points(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size > MAX_QUERY_POINTS:
            raise WorldError(
                "LIMIT_EXCEEDED", f"at most {MAX_QUERY_POINTS:,} points per query", "points"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise WorldError("INVALID_ARGUMENT", "points must be finite", "points")
        if np.any(np.abs(x) > D.MAX_COORDINATE) or np.any(np.abs(y) > D.MAX_COORDINATE):
            raise WorldError(
                "OUTSIDE_WORLD", f"coordinates are within +-{D.MAX_COORDINATE:g} m", "points"
            )

    def sample(self, points: Any, fields: Any = None) -> dict[str, Any]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise WorldError("INVALID_ARGUMENT", "points is a list of [x, y]", "points")
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        self._check_points(x, y)
        wanted = set(SAMPLE_FIELDS if fields is None else fields)
        unknown = wanted - set(SAMPLE_FIELDS)
        if unknown:
            raise WorldError(
                "INVALID_ARGUMENT",
                f"unknown fields {sorted(unknown)}",
                "fields",
                hint="fields: " + ", ".join(SAMPLE_FIELDS),
            )
        self.regional()
        m = self.model()
        s = m.evaluate(x, y)
        out: dict[str, Any] = {"x": x.tolist(), "y": y.tolist()}
        if "height" in wanted:
            out["height"] = s["height"].tolist()
        if wanted & {"normal", "slope"}:
            e = 0.5
            hx = m.height(np.concatenate([x + e, x - e]), np.concatenate([y, y]))
            hy = m.height(np.concatenate([x, x]), np.concatenate([y + e, y - e]))
            n = x.shape[0]
            gx = (hx[:n] - hx[n:]) / (2 * e)
            gy = (hy[:n] - hy[n:]) / (2 * e)
            nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=1)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            if "normal" in wanted:
                out["normal"] = np.round(nrm, 6).tolist()
            if "slope" in wanted:
                out["slope"] = np.round(
                    np.degrees(np.arccos(np.clip(nrm[:, 2], -1, 1))), 4
                ).tolist()
        names = m.biome_names
        w = s["weights"]
        if "biome" in wanted:
            out["biome"] = [names[i] for i in np.argmax(w, axis=1)]
        if "biome_weights" in wanted:
            order = np.argsort(-w, axis=1)[:, :3]
            out["biome_weights"] = [
                {names[j]: round(float(w[i, j]), 4) for j in order[i] if w[i, j] > 1e-3}
                for i in range(w.shape[0])
            ]
        if "temperature" in wanted:
            out["temperature"] = np.round(s["temperature"], 3).tolist()
        if "moisture" in wanted:
            out["moisture"] = np.round(s["moisture"], 4).tolist()
        if "water" in wanted:
            wl = s["water"]
            out["water"] = [
                None if np.isnan(v) or v <= h else float(v)
                for v, h in zip(wl.tolist(), s["height"].tolist(), strict=True)
            ]
        if "water_kind" in wanted:
            kinds = ("none", "sea", "lake", "river", "authored")
            wl = s["water"]
            out["water_kind"] = [
                kinds[int(k)] if not np.isnan(v) and v > h else "none"
                for k, v, h in zip(
                    s["water_kind"].tolist(), wl.tolist(), s["height"].tolist(), strict=True
                )
            ]
        return out

    def profile(self, points: Any, samples: int = 64) -> dict[str, Any]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 2:
            raise WorldError(
                "INVALID_ARGUMENT", "a profile needs at least two [x, y] points", "points"
            )
        if not isinstance(samples, int) or not 2 <= samples <= 10_000:
            raise WorldError("INVALID_ARGUMENT", "samples is 2..10000", "samples")
        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, cum[-1], samples)
        x = np.interp(t, cum, pts[:, 0])
        y = np.interp(t, cum, pts[:, 1])
        s = self.sample(np.stack([x, y], axis=1), ["height", "slope", "biome", "water"])
        return {"distance": np.round(t, 4).tolist(), **s}

    def block_at(self, x: float, y: float, z: float) -> str:
        from .blocks import BLOCK_LIST

        size = self.definition["blocks"]["size"]
        cell = np.floor(np.array([[x, y, z]], dtype=np.float64) / size).astype(np.int64)
        return BLOCK_LIST[int(self.blocks_at(cell)[0])].name

    def blocks_at(self, cells: np.ndarray) -> np.ndarray:
        from .voxels import blocks_at

        return blocks_at(self, cells)

    # -- reports -----------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tiles_cached": len(self._tiles),
                "tile_bytes": int(self._tile_bytes),
                "budget_bytes": int(self.definition["streaming"]["memory_mb"] * 2**20),
                **{
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in self.counters.items()
                },
                "regional_cached": len(self._regionals),
                "last_frame": dict(self.last_frame),
            }

    def clear(self) -> None:
        """Forget every cached result (the next use regenerates what it needs)."""
        with self._lock:
            self._models.clear()
            self._regionals.clear()
            self._tiles.clear()
            self._stitched.clear()
            self._holes.clear()
            self._tile_bytes = 0
        for part in ("_vegetation", "_volumes", "_blocks"):
            cache = getattr(self, part, None)
            if cache is not None:
                setattr(self, part, None)


MAX_QUERY_POINTS = 1_000_000
SAMPLE_FIELDS = (
    "height",
    "normal",
    "slope",
    "biome",
    "biome_weights",
    "temperature",
    "moisture",
    "water",
    "water_kind",
)


def _regional_window(reg: Any, box: tuple[float, float, float, float]) -> bytes:
    """The bytes of every regional result a tile inside ``box`` reads (its key's part)."""
    c = reg.cell
    i0 = max(0, int(math.floor((box[0] - reg.x0) / c)) - 3)
    j0 = max(0, int(math.floor((box[1] - reg.y0) / c)) - 3)
    i1 = min(reg.nx, int(math.ceil((box[2] - reg.x0) / c)) + 4)
    j1 = min(reg.ny, int(math.ceil((box[3] - reg.y0) / c)) + 4)
    h = hashlib.sha256()
    h.update(np.array([reg.x0, reg.y0, c, reg.nx, reg.ny], dtype=np.float64).tobytes())
    if i1 > i0 and j1 > j0:
        h.update(np.ascontiguousarray(reg.delta_grid[j0:j1, i0:i1]).tobytes())
        h.update(np.ascontiguousarray(reg.atten_grid[j0:j1, i0:i1]).tobytes())
        lab = np.ascontiguousarray(reg.lake_label[j0:j1, i0:i1])
        h.update(lab.tobytes())
        used = np.unique(lab[lab >= 0])
        h.update(reg.lake_levels[used].tobytes())
    dc = reg.distance_cell
    a0 = max(0, int(math.floor((box[0] - reg.x0) / dc)) - 2)
    b0 = max(0, int(math.floor((box[1] - reg.y0) / dc)) - 2)
    a1 = min(reg.distance_grid.shape[1], int(math.ceil((box[2] - reg.x0) / dc)) + 3)
    b1 = min(reg.distance_grid.shape[0], int(math.ceil((box[3] - reg.y0) / dc)) + 3)
    if a1 > a0 and b1 > b0:
        h.update(np.ascontiguousarray(reg.distance_grid[b0:b1, a0:a1]).tobytes())
    near = reg.segments_near(box)
    if near.size:
        h.update(np.ascontiguousarray(reg.segments[near]).tobytes())
    return h.digest()


def _footprint(cell: tuple[int, int]) -> tuple[float, float, float, float]:
    from .volumes import CHUNK

    ci, cj = cell
    return (ci * CHUNK, cj * CHUNK, (ci + 1) * CHUNK, (cj + 1) * CHUNK)
