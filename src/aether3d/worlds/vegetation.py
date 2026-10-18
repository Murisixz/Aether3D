"""Plants and rocks: biome-aware placement, stable ids, levels of detail and culling.

**Placement** is computed per placement tile (:data:`TILE` metres) and class -- ``trees``,
``shrubs``, ``rocks``, ``cover`` (grass, flowers, ferns...) -- from integer hashes, so it
never depends on which tiles were placed first or on the camera:

1. candidates sit on global grids: trees on one grid shared by every tree species (cells
   of ``sqrt(10000 / D)`` metres, ``D`` the largest total tree density per hectare over the
   biomes, times ``vegetation.density``), so two trees never share a cell; every other
   species on a grid of its own (densities times ``vegetation.density``, ``clutter`` for
   ground cover, ``rocks`` for rocks). A cell holds one candidate at a hashed position,
   owned by the tile containing it, and every candidate is decided on its own;
2. the fields a candidate needs (biome weights, temperature, moisture, slope, water,
   roads) are interpolated from exact samples on a global lattice (every :data:`LATTICE`
   metres; each sample depends only on its coordinates). A candidate grows with
   probability ``d_here / D``: ``d_here`` is the density over the biome weights there,
   modulated by clustering noise (groves and clearings) and zeroed on slopes steeper than
   the species allows, above the tree line (trees and shrubs), on snow (ground cover), in
   water, on roads and where ``clear`` features, pads, lakes or rivers say so; a tree takes
   a hashed draw among the local species densities;
3. survivors are checked against the exact fields once more (no plant in water or on a
   road) and stand on the authoritative ground, sunk :data:`SINK` metres so roots never
   float; each takes a hashed variant, rotation and size.

Every plant has a **stable id**: ``tree:<cell x>,<cell y>`` for trees, else
``<species>:<cell x>,<cell y>`` (:meth:`PlantSet.ids`).

**Drawing** keeps the plants within their class's reach (trees ``vegetation.distance``,
shrubs and rocks less, ground cover ``detail_distance``; faded out over the last fifth by
hashed thresholds) inside the view frustum widened for shadows. Beyond a class's thinning
distance ``T`` (:data:`THIN`) a plant is drawn only if its hashed ``keep`` value is below
``(T / d)^3``, slightly enlarged (:data:`GROW`); these sets are nested -- a plant drawn far
away is drawn at every shorter distance -- so nothing pops in as the camera approaches, and
far tiles place only the candidates they can draw (:data:`LEVELS`). Each plant switches
between three levels of detail at hashed distances, spreading the changes out. Beyond the
reach, forests are carried by the terrain's darker forest floor (``surface.py``).
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import noise as nz
from .assets import LAYERS, ROCKS, TREES, asset
from .definition import biome_table, digest
from .features import clear_mask

TILE = 128.0
"""Metres per side of a placement tile."""
LATTICE = 8.0
"""Metres between the exact field samples placement interpolates."""
THIN = {"trees": 400.0, "shrubs": 140.0, "rocks": 180.0, "cover": 1e9}
"""Thinning distance per class (metres)."""
REACH = {"shrubs": 600.0, "rocks": 700.0}
"""Largest drawing distance of shrubs and rocks (at most ``vegetation.distance``)."""
GROW = {"trees": 0.3, "shrubs": 0.2, "rocks": 0.15, "cover": 0.0}
"""Thinned plants are enlarged by ``(1 / fraction) ^ (GROW / 3)``, at most 2.5 times."""
LEVELS = 8
"""Thinning levels: a tile at level k places only the candidates with keep < 4^-k."""
LOD_DISTANCES = (90.0, 300.0)
"""Distances (metres, times a hashed 0.85-1.15) where plants switch to levels of detail 1
and 2; ground cover and rocks switch at a quarter of them."""
SINK = 0.15
"""Metres plants are sunk into the ground (no visible gap on slopes)."""
MAX_PLANTS = 300_000
"""Plants drawn per frame at most (the nearest kept)."""
MAX_TILES = 30_000
"""Placement tiles kept in memory (least recently used dropped first)."""
CLASSES = ("trees", "shrubs", "rocks", "cover")
SPECIES_ORDER = tuple(LAYERS)
SHRUBS = frozenset(("bush", "shrub"))


def class_of(species: str) -> str:
    if species in TREES:
        return "trees"
    if species in ROCKS:
        return "rocks"
    if species in SHRUBS:
        return "shrubs"
    return "cover"


@dataclass
class PlantSet:
    """The plants of one placement tile, class and thinning level (equal-length arrays)."""

    species: np.ndarray  # (n,) int16 index into SPECIES_ORDER
    variant: np.ndarray  # (n,) int16
    pos: np.ndarray  # (n, 3) float64
    yaw: np.ndarray  # (n,) radians
    scale: np.ndarray  # (n,)
    keep: np.ndarray  # (n,) hashed thinning values in [0, 1)
    cell: np.ndarray  # (n, 2) int64 grid cells (the ids)

    @property
    def count(self) -> int:
        return int(self.species.shape[0])

    @property
    def nbytes(self) -> int:
        return int(
            sum(
                a.nbytes
                for a in (
                    self.species,
                    self.variant,
                    self.pos,
                    self.yaw,
                    self.scale,
                    self.keep,
                    self.cell,
                )
            )
        )

    def ids(self) -> list[str]:
        out = []
        for s, (ci, cj) in zip(self.species.tolist(), self.cell.tolist(), strict=True):
            name = SPECIES_ORDER[s]
            out.append(f"{'tree' if name in TREES else name}:{ci},{cj}")
        return out


def _empty() -> PlantSet:
    return PlantSet(
        np.zeros(0, np.int16),
        np.zeros(0, np.int16),
        np.zeros((0, 3)),
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
        np.zeros((0, 2), np.int64),
    )


def placement_key(d: dict) -> str:
    """What placement depends on: everything but streaming, materials, structures and
    block settings."""
    skip = ("streaming", "materials", "structures", "blocks")
    return digest({k: d[k] for k in d if k not in skip})


@lru_cache(maxsize=8)
def _seeds(seed: int) -> dict[str, int]:
    out = {
        name: nz.derive(seed, "plants", name) for name in ("grow", "pick", "variant", "yaw", "size")
    }
    for cls in CLASSES:
        out["cluster-" + cls] = nz.derive(seed, "plants", "cluster", cls)
    for name in (*SPECIES_ORDER, "tree"):
        g = nz.derive(seed, "plants", "grid", name)
        out["grid-" + name] = g
        for part in ("x", "y", "keep"):
            out[f"grid-{name}-{part}"] = nz.derive(g, part)
    return out


class Vegetation:
    """The plants of a world (per runtime; see the module docstring)."""

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self._tiles: OrderedDict[tuple, PlantSet] = OrderedDict()
        self._lock = threading.Lock()
        self._table_key: str | None = None
        self._table: dict[str, Any] = {}
        self.counters = {
            "tiles_placed": 0,
            "plants_placed": 0,
            "candidates": 0,
            "seconds": 0.0,
            "bytes": 0,
        }

    # -- species tables ------------------------------------------------------------------

    def table(self) -> dict[str, Any]:
        """Per class: species, densities per biome (per hectare), clusters, tangents of the
        steepest slopes, scales and the candidate grids ``(name, cell, density, member
        species)``."""
        d = self.rt.definition
        key = placement_key(d)
        if key == self._table_key:
            return self._table
        biomes = biome_table(d)
        veg = d["vegetation"]
        mult = {
            "trees": veg["density"],
            "shrubs": veg["density"],
            "cover": veg["clutter"],
            "rocks": veg["rocks"],
        }
        nb = len(biomes)
        per: dict[str, dict[str, Any]] = {}
        for b, biome in enumerate(biomes.values()):
            for e in biome["vegetation"]:
                sp = e["species"]
                rec = per.setdefault(
                    sp,
                    {
                        "density": np.zeros(nb),
                        "cluster": np.zeros(nb),
                        "slope": np.zeros(nb),
                        "scale": np.ones(nb),
                    },
                )
                rec["density"][b] = e["density"] * mult[class_of(sp)]
                rec["cluster"][b] = e["cluster"]
                rec["slope"][b] = e["slope"]
                rec["scale"][b] = e["scale"]
        table: dict[str, Any] = {}
        for cls in CLASSES:
            names = [
                sp
                for sp in SPECIES_ORDER
                if sp in per and class_of(sp) == cls and per[sp]["density"].max() > 0.0
            ]
            if not names:
                continue
            dens = np.stack([per[sp]["density"] for sp in names])  # (S, B)
            if cls == "trees":
                grids = [("tree", float(dens.sum(axis=0).max()), list(range(len(names))))]
            else:
                grids = [(sp, float(dens[k].max()), [k]) for k, sp in enumerate(names)]
            table[cls] = {
                "names": names,
                "index": np.array([SPECIES_ORDER.index(sp) for sp in names], dtype=np.int16),
                "density": dens,
                "cluster": np.stack([per[sp]["cluster"] for sp in names]),
                "tan": _tan_deg(np.stack([per[sp]["slope"] for sp in names])),
                "scale": np.stack([per[sp]["scale"] for sp in names]),
                "grids": [
                    (g, max(math.sqrt(10000.0 / dmax), 0.3), dmax, members)
                    for g, dmax, members in grids
                    if dmax > 0.0
                ],
            }
        self._table_key, self._table = key, table
        return table

    # -- placement -----------------------------------------------------------------------

    def tile(self, i: int, j: int, cls: str, level: int = 0, ctx: Any = None) -> PlantSet:
        """The plants of placement tile ``(i, j)`` of a class at a thinning level (cached)."""
        return self.tiles([(cls, i, j, level)], ctx)[0]

    def tiles(self, requests: list[tuple[str, int, int, int]], ctx: Any = None) -> list[PlantSet]:
        """The plants of several ``(class, i, j, level)`` tiles; the missing ones are placed
        together (one field evaluation per class: the same results as one at a time, every
        candidate being decided on its own)."""
        pk = placement_key(self.rt.definition)
        out: list[PlantSet | None] = []
        missing: dict[str, list[int]] = {}
        with self._lock:
            for k, (cls, i, j, level) in enumerate(requests):
                found = self._tiles.get((pk, i, j, cls, level))
                if found is not None:
                    self._tiles.move_to_end((pk, i, j, cls, level))
                else:
                    missing.setdefault(cls, []).append(k)
                out.append(found)
        for cls, ks in missing.items():
            if ctx is not None:
                ctx.check("vegetation placement")
            t0 = time.perf_counter()
            placed = self._place(cls, [requests[k][1:] for k in ks])
            with self._lock:
                for k, plants in zip(ks, placed, strict=True):
                    _, i, j, level = requests[k]
                    self._tiles[(pk, i, j, cls, level)] = plants
                    self.counters["bytes"] += plants.nbytes
                    self.counters["plants_placed"] += plants.count
                    out[k] = plants
                while len(self._tiles) > MAX_TILES:
                    _, old = self._tiles.popitem(last=False)
                    self.counters["bytes"] -= old.nbytes
                self.counters["tiles_placed"] += len(ks)
                self.counters["seconds"] += time.perf_counter() - t0
        return out  # type: ignore[return-value]

    def _candidates(self, i: int, j: int, tab: dict, level: int) -> dict[str, np.ndarray] | None:
        seeds = _seeds(int(self.rt.definition["seed"]))
        x0, y0 = i * TILE, j * TILE
        x1, y1 = x0 + TILE, y0 + TILE
        ext = self.rt.model().extent()
        limit = 4.0**-level
        parts = []
        for g, (name, cell, _dmax, _members) in enumerate(tab["grids"]):
            ci0, ci1 = math.floor(x0 / cell), math.floor(x1 / cell)
            cj0, cj1 = math.floor(y0 / cell), math.floor(y1 / cell)
            ci, cj = np.meshgrid(
                np.arange(ci0, ci1 + 1, dtype=np.int64), np.arange(cj0, cj1 + 1, dtype=np.int64)
            )
            ci, cj = ci.reshape(-1), cj.reshape(-1)
            keep = nz.random2(seeds[f"grid-{name}-keep"], ci, cj)
            sel = keep < limit
            ci, cj, keep = ci[sel], cj[sel], keep[sel]
            x = (ci + 0.05 + 0.9 * nz.random2(seeds[f"grid-{name}-x"], ci, cj)) * cell
            y = (cj + 0.05 + 0.9 * nz.random2(seeds[f"grid-{name}-y"], ci, cj)) * cell
            own = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if ext is not None:
                own &= (x >= ext[0]) & (x <= ext[2]) & (y >= ext[1]) & (y <= ext[3])
            if not np.any(own):
                continue
            n = int(own.sum())
            parts.append(
                {
                    "grid": np.full(n, g, dtype=np.int64),
                    "ci": ci[own],
                    "cj": cj[own],
                    "x": x[own],
                    "y": y[own],
                    "keep": keep[own],
                }
            )
        if not parts:
            return None
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def _lattice(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Fields at points, bilinear from exact samples at the four global lattice nodes
        around each; ``slope`` (a tangent) is the gradient of the bilinear heights."""
        model = self.rt.model()
        L = LATTICE
        fx, fy = x / L, y / L
        ix0 = np.floor(fx).astype(np.int64)
        iy0 = np.floor(fy).astype(np.int64)
        tx, ty = fx - ix0, fy - iy0
        off = np.array([0, 1], dtype=np.int64)
        n = x.shape[0]
        nx = np.broadcast_to(ix0[:, None, None] + off[None, None, :], (n, 2, 2))
        ny = np.broadcast_to(iy0[:, None, None] + off[None, :, None], (n, 2, 2))
        base_x, base_y = int(nx.min()), int(ny.min())
        width = int(nx.max()) - base_x + 1
        keys = (ny - base_y) * width + (nx - base_x)
        uniq, inv = np.unique(keys.reshape(-1), return_inverse=True)
        ux = (uniq % width + base_x).astype(np.float64) * L
        uy = (uniq // width + base_y).astype(np.float64) * L
        s = model.evaluate(ux, uy)
        water = s["water"]
        wet_n = (~np.isnan(water) & (water > s["height"] - 0.3)).astype(np.float64)

        def at(values: np.ndarray) -> np.ndarray:
            # Bilinear between the corners [row y0/y1, column x0/x1].
            a = values[inv].reshape((n, 2, 2) + values.shape[1:])
            ex = (-1,) + (1,) * (a.ndim - 3)
            ax, ay = tx.reshape(ex), ty.reshape(ex)
            top = a[:, 0, 0] * (1 - ax) + a[:, 0, 1] * ax
            bot = a[:, 1, 0] * (1 - ax) + a[:, 1, 1] * ax
            return top * (1 - ay) + bot * ay

        h = s["height"][inv].reshape(n, 2, 2)
        gx = ((h[:, 0, 1] - h[:, 0, 0]) * (1 - ty) + (h[:, 1, 1] - h[:, 1, 0]) * ty) / L
        gy = ((h[:, 1, 0] - h[:, 0, 0]) * (1 - tx) + (h[:, 1, 1] - h[:, 0, 1]) * tx) / L
        return {
            "wet": at(wet_n),
            "temperature": at(s["temperature"]),
            "moisture": at(s["moisture"]),
            "weights": at(s["weights"]),
            "road": at(s["road"]),
            "slope": np.sqrt(gx * gx + gy * gy),
        }

    def _place(self, cls: str, tiles: list[tuple[int, int, int]]) -> list[PlantSet]:
        """Place the plants of a class in ``(i, j, level)`` tiles."""
        tab = self.table().get(cls)
        if tab is None:
            return [_empty() for _ in tiles]
        parts = []
        for t, (i, j, level) in enumerate(tiles):
            c = self._candidates(i, j, tab, level)
            if c is not None:
                c["tile"] = np.full(c["x"].shape[0], t, dtype=np.int64)
                parts.append(c)
        if not parts:
            return [_empty() for _ in tiles]
        cand = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.rt.regional()
        model = self.rt.model()
        d = self.rt.definition
        seeds = _seeds(int(d["seed"]))
        with self._lock:
            self.counters["candidates"] += int(cand["x"].shape[0])
        x, y, ci, cj, grid = cand["x"], cand["y"], cand["ci"], cand["cj"], cand["grid"]
        n = x.shape[0]
        rows = np.arange(n)
        f = self._lattice(x, y)
        w = f["weights"]
        dens = w @ tab["density"].T  # (n, S): every species' density here
        member = np.zeros(n, dtype=np.int64)
        dmax = np.zeros(n)
        total = np.zeros(n)
        hx = ci * 8191 + grid  # hash coordinates unique per grid
        for g, (_name, _cell, gmax, members) in enumerate(tab["grids"]):
            sel = grid == g
            if not np.any(sel):
                continue
            dmax[sel] = gmax
            local = dens[sel][:, members]
            total[sel] = local.sum(axis=1)
            if len(members) == 1:
                member[sel] = members[0]
            else:  # trees: a hashed draw among the local densities
                u = nz.random2(seeds["pick"], hx[sel], cj[sel])
                cum = np.cumsum(local, axis=1) / np.maximum(total[sel][:, None], 1e-12)
                member[sel] = np.array(members)[np.argmax(cum > u[:, None], axis=1)]
        p = total / np.maximum(dmax, 1e-12)
        # Clustering: groves and clearings at 40-80 m.
        cl_w = (w @ tab["cluster"].T)[rows, member] / np.maximum(w.sum(axis=1), 1e-12)
        cl = nz.fbm2(seeds["cluster-" + cls], x, y, octaves=2, frequency=1.0 / 55.0)
        p = p * np.clip(1.0 + cl_w * 2.2 * cl, 0.0, 2.0)
        # Slope, water, temperature, roads, clearing.
        tan_max = (w @ tab["tan"].T)[rows, member] / np.maximum(w.sum(axis=1), 1e-12)
        p = np.where(f["slope"] > tan_max, 0.0, p)
        names = np.array(tab["names"])[member]
        reeds = names == "reeds"
        wet = f["wet"] > 0.05
        p = np.where(reeds, np.where(wet | (f["moisture"] > 0.8), p, 0.0), np.where(wet, 0.0, p))
        surf = d["surface"]
        if cls in ("trees", "shrubs"):
            tl = surf["treeline_temperature"]
            p = p * nz.smoothstep(tl - 1.0, tl + 1.5, f["temperature"])
        if cls == "cover":
            st = surf["snow_temperature"]
            p = p * nz.smoothstep(st - 0.5, st + 1.5, f["temperature"])
        p = p * (1.0 - nz.smoothstep(0.1, 0.4, f["road"]))
        what = {"trees": "trees", "shrubs": "plants", "cover": "plants", "rocks": "rocks"}[cls]
        p = p * (1.0 - clear_mask(model.final_features, x, y, what))
        idx = np.flatnonzero(nz.random2(seeds["grow"], hx, cj) < p)
        if idx.size == 0:
            return [_empty() for _ in tiles]
        # Survivors against the exact fields: ground height, water, roads.
        s = model.evaluate(x[idx], y[idx])
        hz = s["height"]
        water = s["water"]
        ok = (np.isnan(water) | (water <= hz - 0.2) | reeds[idx]) & (s["road"] < 0.3)
        idx, hz = idx[ok], hz[ok]
        if idx.size == 0:
            return [_empty() for _ in tiles]
        hxi, hyi = hx[idx], cj[idx]
        variants = d["vegetation"]["variants"]
        variant = (nz.hash2(seeds["variant"], hxi, hyi) % np.uint64(variants)).astype(np.int16)
        yaw = 2.0 * math.pi * nz.random2(seeds["yaw"], hxi, hyi)
        scale_k = (w[idx] @ tab["scale"].T)[np.arange(idx.size), member[idx]]
        scale_k = scale_k / np.maximum(w[idx].sum(axis=1), 1e-12)
        size = (0.75 + 0.5 * nz.random2(seeds["size"], hxi, hyi)) * scale_k
        pos = np.stack([x[idx], y[idx], hz - SINK], axis=1)
        species = tab["index"][member[idx]]
        keep = cand["keep"][idx]
        cells = np.stack([ci[idx], cj[idx]], axis=1)
        owner = cand["tile"][idx]
        out = []
        for t in range(len(tiles)):
            sel = owner == t
            out.append(
                PlantSet(
                    species[sel], variant[sel], pos[sel], yaw[sel], size[sel], keep[sel], cells[sel]
                )
            )
        return out

    # -- drawing -------------------------------------------------------------------------

    def reach(self) -> dict[str, float]:
        veg = self.rt.definition["vegetation"]
        return {
            "trees": float(veg["distance"]),
            "shrubs": float(min(REACH["shrubs"], veg["distance"])),
            "rocks": float(min(REACH["rocks"], veg["distance"])),
            "cover": float(min(veg["detail_distance"], veg["distance"])),
        }

    def tiles_for(self, view: Any) -> list[tuple[str, int, int, int, float]]:
        """``(class, i, j, level, reach)`` of every placement tile a view needs."""
        eye = view.eye
        e = self.rt.model().extent()
        table = self.table()
        out = []
        for cls, dist in self.reach().items():
            if dist <= 0.0 or cls not in table:
                continue
            T = THIN[cls]
            i0, i1 = math.floor((eye[0] - dist) / TILE), math.floor((eye[0] + dist) / TILE)
            j0, j1 = math.floor((eye[1] - dist) / TILE), math.floor((eye[1] + dist) / TILE)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    bx0, by0 = i * TILE, j * TILE
                    if e is not None and (
                        bx0 + TILE < e[0] or bx0 > e[2] or by0 + TILE < e[1] or by0 > e[3]
                    ):
                        continue
                    cx = min(max(eye[0], bx0), bx0 + TILE)
                    cy = min(max(eye[1], by0), by0 + TILE)
                    near = math.hypot(cx - eye[0], cy - eye[1])
                    if near > dist or _box_outside(view, bx0, by0, TILE):
                        continue
                    frac = 1.0 if near <= T else (T / near) ** 3
                    level = min(LEVELS - 1, int(math.floor(math.log(1.0 / frac, 4.0) + 1e-9)))
                    out.append((cls, i, j, level, dist))
        return out

    def visible(self, view: Any, ctx: Any = None) -> dict[str, np.ndarray]:
        """The plants to draw from a view (see the module docstring)."""
        eye = np.asarray(view.eye, dtype=np.float64)
        need = self.tiles_for(view)
        placed = self.tiles([(cls, i, j, level) for cls, i, j, level, _ in need], ctx)
        sets = [(n[0], n[4], ps) for n, ps in zip(need, placed, strict=True) if ps.count]
        if not sets:
            return {}
        cat = {
            k: np.concatenate([getattr(p, k) for _, _, p in sets])
            for k in ("species", "variant", "pos", "yaw", "scale", "keep")
        }
        limit = np.concatenate([np.full(p.count, dist) for _, dist, p in sets])
        cls_i = np.concatenate([np.full(p.count, CLASSES.index(c)) for c, _, p in sets])
        thin = np.array([THIN[c] for c in CLASSES])[cls_i]
        grow = np.array([GROW[c] for c in CLASSES])[cls_i]
        rel = cat["pos"] - eye
        dist = np.sqrt(np.sum(rel * rel, axis=1))
        keep = cat["keep"]
        frac = np.where(dist <= thin, 1.0, (thin / np.maximum(dist, 1.0)) ** 3)
        # Fade out over the last fifth of the reach; thin beyond the thinning distance.
        u = keep / np.maximum(frac, 1e-300)  # uniform in [0, 1) among the kept
        show = (keep < frac) & (dist < limit * (1.0 - 0.2 * u))
        scale = cat["scale"] * np.minimum(frac ** (-grow / 3.0), 2.5)
        if view.frustum is not None:
            radius = 30.0 * scale
            for pl in view.frustum:
                show &= cat["pos"] @ pl[:3] + pl[3] > -radius
        idx = np.flatnonzero(show)
        if idx.size > MAX_PLANTS:
            idx = idx[np.argsort(dist[idx], kind="stable")[:MAX_PLANTS]]
        jit = 0.85 + 0.3 * u[idx]
        small = (cls_i[idx] == CLASSES.index("cover")) | (cls_i[idx] == CLASSES.index("rocks"))
        fd = dist[idx] / (jit * np.where(small, 0.25, 1.0))
        lod = np.where(fd < LOD_DISTANCES[0], 0, np.where(fd < LOD_DISTANCES[1], 1, 2))
        return {
            "species": cat["species"][idx],
            "variant": cat["variant"][idx],
            "pos": cat["pos"][idx],
            "yaw": cat["yaw"][idx],
            "scale": scale[idx],
            "lod": lod.astype(np.int16),
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.counters,
                "seconds": round(self.counters["seconds"], 4),
                "tiles_cached": len(self._tiles),
            }

    def clear(self) -> None:
        with self._lock:
            self._tiles.clear()
            self.counters["bytes"] = 0


def _tan_deg(deg: np.ndarray) -> np.ndarray:
    """Tangents of angles in degrees (from :func:`noise.cos_sin_deg`; 1e3 near 90)."""
    deg = np.asarray(deg, dtype=np.float64)
    out = np.full(deg.shape, 1e3)
    for v in np.unique(deg):
        if v < 89.9:
            c, s = nz.cos_sin_deg(float(v))
            out[deg == v] = s / c
    return out


def _box_outside(view: Any, x0: float, y0: float, span: float) -> bool:
    from .tiles import outside

    return outside(view, (x0, y0, x0 + span, y0 + span), -1e4, 1e4)


def vegetation_of(rt: Any) -> Vegetation:
    v = getattr(rt, "_vegetation", None)
    if v is None:
        v = rt._vegetation = Vegetation(rt)
    return v


def render_parts(rt: Any, view: Any, ctx: Any = None) -> list:
    """Instanced plant meshes for a view: ``(id, name, mesh, matrix, layers)`` per plant,
    ids ``world:<world id>:plant/<k>`` numbered per frame."""
    d = rt.definition
    if d["mode"] != "terrain":
        return []
    vis = vegetation_of(rt).visible(view, ctx)
    if not vis:
        return []
    world = rt.world
    sp, var, pos, yaw, scale, lod = (
        vis[k] for k in ("species", "variant", "pos", "yaw", "scale", "lod")
    )
    c = np.cos(yaw) * scale
    s = np.sin(yaw) * scale
    n = sp.shape[0]
    m = np.zeros((n, 4, 4))
    m[:, 0, 0], m[:, 0, 1] = c, -s
    m[:, 1, 0], m[:, 1, 1] = s, c
    m[:, 2, 2] = scale
    m[:, :3, 3] = pos
    m[:, 3, 3] = 1.0
    out = []
    found: dict[tuple[int, int, int], tuple] = {}
    for k, key in enumerate(zip(sp.tolist(), var.tolist(), lod.tolist(), strict=True)):
        a = found.get(key)
        if a is None:
            species = SPECIES_ORDER[key[0]]
            a = found[key] = (asset(species, key[1], key[2], 0), f"{world.name} {species}")
        out.append((f"world:{world.id}:plant/{k}", a[1], a[0].mesh, m[k], a[0].layers))
    return out
