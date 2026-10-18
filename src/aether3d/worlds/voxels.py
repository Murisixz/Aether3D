"""Block worlds: generation, edits, greedy meshing, distant levels of detail and queries.

A block world shares the terrain world's fields -- height, climate, biomes, rivers, lakes,
roads and features (:mod:`.fields`) -- sampled once per block column; this module turns them
into blocks (:mod:`.blocks`) and meshes:

**Generation** of a chunk (16 x 16 columns, the whole vertical range) is a pure function of
the definition and the chunk coordinates:

1. the surface block of each column is ``round(height / size) - 1``; below it lie the
   biome's top block, its filler (``depth`` blocks), stone (dark stone deep down) with
   hashed ores, and bedrock at the floor; badlands get banded terracotta; slopes of three
   blocks or more show stone; beaches and river beds get the biome's underwater block;
   cold surfaces get snow;
2. water fills empty blocks up to the column's water level (sea, lakes, rivers);
3. caves (``caves.enabled``) carve air where 3D noise forms caverns (thresholded fBm) and
   tunnels (the intersection of two noise zero-sets), never right under water;
4. trees (block versions of the biome's species) and small plants are stamped from hashed
   candidates on a grid of 5-column cells; each chunk also looks at the trees of its
   neighbours' margins, so a tree crossing a chunk border is the same on both sides;
5. **edits** (:mod:`.edits`) replace whatever generation made.

**Meshing**: a face is drawn where an opaque block meets a non-opaque one (hidden faces
removed, across chunk borders too); faces of one texture, ambient occlusion and direction
are merged into rectangles (run lengths along rows, then identical runs stacked across
rows) with UVs in block units (textures repeat once per block); vertex colours carry
per-vertex ambient occlusion (Minecraft-style, from the three neighbours of each corner)
times the biome tint (grass, foliage, water). Plants are two crossed alpha-masked quads;
water is its own surface an eighth of a block below the top, coloured by depth.

**Distant chunks** use coarse blocks (``2^k`` blocks per side) built from column heights
alone, keeping blocky silhouettes; their outer walls hang down to hide the steps between
levels.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import WorldError
from ..mesh import Mesh
from . import noise as nz
from .blocks import AIR, BLOCK_LIST, BLOCKS, WATER, texture_names
from .definition import biome_table

CHUNK = 16
AO_LEVELS = np.array([0.5, 0.66, 0.82, 1.0])
"""Vertex brightness for 0..3 open neighbours of a face corner."""
LOD0_RADIUS = 12
"""Chunks (16 blocks) around the camera meshed block by block; beyond, coarse levels."""
MAX_CHUNKS = 6000

_TEXTURES = texture_names()
_TEX_INDEX = {name: k for k, name in enumerate(_TEXTURES)}
_N = len(BLOCK_LIST)
_OPAQUE = np.array([b.kind == "solid" for b in BLOCK_LIST])
_PLANT = np.array([b.kind == "plant" for b in BLOCK_LIST])
_LIQUID = np.array([b.kind == "liquid" for b in BLOCK_LIST])
# Texture index per block and face: 0 top, 1 side, 2 bottom.
_FACE_TEX = np.zeros((_N, 3), dtype=np.int64)
_TINT = np.zeros(_N, dtype=np.int64)  # 0 none, 1 grass, 2 foliage, 3 water
for _k, _b in enumerate(BLOCK_LIST):
    for _f, _t in enumerate((_b.top, _b.side, _b.bottom)):
        _FACE_TEX[_k, _f] = _TEX_INDEX.get(_t, 0)
    _TINT[_k] = {"grass": 1, "foliage": 2, "water": 3}.get(_b.tint or "", 0)

TREE_BLOCKS = {
    # species: (log, leaves, shape)
    "oak": ("oak_log", "oak_leaves", "round"),
    "beech": ("oak_log", "oak_leaves", "round"),
    "maple": ("oak_log", "oak_leaves", "round"),
    "willow": ("oak_log", "oak_leaves", "round"),
    "birch": ("birch_log", "birch_leaves", "tall"),
    "spruce": ("spruce_log", "spruce_leaves", "cone"),
    "fir": ("spruce_log", "spruce_leaves", "cone"),
    "pine": ("spruce_log", "spruce_leaves", "cone"),
    "larch": ("spruce_log", "spruce_leaves", "cone"),
    "jungle": ("jungle_log", "jungle_leaves", "jungle"),
    "palm": ("jungle_log", "jungle_leaves", "palm"),
    "acacia": ("acacia_log", "acacia_leaves", "flat"),
    "baobab": ("acacia_log", "acacia_leaves", "flat"),
    "cactus": ("cactus", "cactus", "cactus"),
    "dead_tree": ("oak_log", "oak_log", "dead"),
}
PLANT_BLOCKS = {
    "grass": "tall_grass",
    "tall_grass": "tall_grass",
    "fern": "fern",
    "dead_bush": "dead_bush",
    "mushroom": "red_mushroom",
    "reeds": "tall_grass",
    "heather": "fern",
}
FLOWERS = ("poppy", "dandelion", "cornflower")


@dataclass
class ChunkData:
    cx: int
    cy: int
    z0: int  # block z of row 0
    blocks: np.ndarray  # (H, 16, 16) uint16: [z, y, x]
    tint: np.ndarray  # (16, 16, 3, 3) linear grass/foliage/water colours per column
    water_level: np.ndarray  # (16, 16) float: water surface z (metres) or nan
    seconds: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return int(self.blocks.nbytes + self.tint.nbytes + self.water_level.nbytes)


def _rgb(c: str) -> np.ndarray:
    from ..rendering.color import to_linear

    return np.asarray(to_linear(tuple(int(c[i : i + 2], 16) / 255.0 for i in (1, 3, 5))))


class BlockWorld:
    """The blocks of one world (per runtime; see the module docstring)."""

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self._chunks: OrderedDict[tuple, ChunkData] = OrderedDict()
        self._meshes: OrderedDict[tuple, Any] = OrderedDict()
        self._coarse: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.RLock()
        self.counters = {
            "chunks_generated": 0,
            "chunks_meshed": 0,
            "coarse_meshed": 0,
            "seconds_generate": 0.0,
            "seconds_mesh": 0.0,
        }

    # -- keys --------------------------------------------------------------------------

    def _base_key(self) -> str:
        from .definition import digest

        d = self.rt.definition
        return digest({k: d[k] for k in d if k not in ("materials", "streaming", "structures")})

    def _edit_key(self, cx: int, cy: int) -> str:
        d = self.rt.definition
        b = d["blocks"]
        s0 = b["min_height"] // 16
        s1 = (b["max_height"] - 1) // 16
        keys = [(cx, cy, sz) for sz in range(s0, s1 + 1)]
        return self.rt.world._edits.digest_of(keys)

    # -- generation --------------------------------------------------------------------

    def chunk(self, cx: int, cy: int, ctx: Any = None) -> ChunkData:
        key = (self._base_key(), cx, cy, self._edit_key(cx, cy))
        with self._lock:
            found = self._chunks.get(key)
            if found is not None:
                self._chunks.move_to_end(key)
                return found
        if ctx is not None:
            ctx.check("block generation")
        t0 = time.perf_counter()
        data = self._generate(cx, cy)
        self._apply_edits(data)
        data.seconds = time.perf_counter() - t0
        with self._lock:
            self._chunks[key] = data
            while len(self._chunks) > MAX_CHUNKS:
                self._chunks.popitem(last=False)
            self.counters["chunks_generated"] += 1
            self.counters["seconds_generate"] += data.seconds
        return data

    def _columns(self, cx: int, cy: int, margin: int) -> dict[str, np.ndarray]:
        """Fields at the column centres of a chunk plus ``margin`` columns around it."""
        d = self.rt.definition
        size = d["blocks"]["size"]
        n = CHUNK + 2 * margin
        i = np.arange(-margin, CHUNK + margin)
        bx = cx * CHUNK + i
        by = cy * CHUNK + i
        X, Y = np.meshgrid((bx + 0.5) * size, (by + 0.5) * size)
        self.rt.regional()
        s = self.rt.model().evaluate(X.reshape(-1), Y.reshape(-1))
        out = {
            k: v.reshape((n, n) + v.shape[1:])
            for k, v in s.items()
            if isinstance(v, np.ndarray) and v.shape[:1] == (n * n,)
        }
        out["bx"], out["by"] = np.meshgrid(bx, by)
        return out

    def _generate(self, cx: int, cy: int) -> ChunkData:
        d = self.rt.definition
        model = self.rt.model()
        b = d["blocks"]
        size = b["size"]
        zmin, zmax = b["min_height"], b["max_height"]
        H = zmax - zmin
        seed = int(d["seed"])
        M = 3  # margin columns: slopes and neighbouring trees
        col = self._columns(cx, cy, M)
        h = col["height"]
        top_all = np.clip(np.rint(h / size).astype(np.int64) - 1, zmin + 2, zmax - 3)
        inner = (slice(M, M + CHUNK), slice(M, M + CHUNK))
        top = top_all[inner]
        # Slope: the largest step to a neighbouring column.
        step = np.zeros_like(top)
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = top_all[M + dj : M + dj + CHUNK, M + di : M + di + CHUNK]
            step = np.maximum(step, np.abs(nb - top))
        w = col["weights"][inner]
        table = biome_table(d)
        names = list(table)
        biomes = list(table.values())
        dom = np.argmax(w, axis=-1)
        bid = {k: BLOCKS[k] for k in BLOCKS}
        top_b = np.array([bid[x["blocks"]["top"]] for x in biomes])[dom]
        fill_b = np.array([bid[x["blocks"]["filler"]] for x in biomes])[dom]
        under_b = np.array([bid[x["blocks"]["underwater"]] for x in biomes])[dom]
        depth = np.array([x["blocks"]["depth"] for x in biomes])[dom]
        temp = col["temperature"][inner]
        surf = d["surface"]
        cold = temp < surf["snow_temperature"]
        top_b = np.where(cold & (top_b == bid["grass_block"]), bid["snowy_grass_block"], top_b)
        top_b = np.where(cold & (top_b == bid["sand"]), bid["snow_block"], top_b)
        cliff = step >= 3
        top_b = np.where(cliff & ~cold, bid["stone"], top_b)
        fill_b = np.where(cliff, bid["stone"], fill_b)
        road = col["road"][inner] > 0.5
        top_b = np.where(road, bid["dirt_path"], top_b)
        water = col["water"][inner]
        wet = ~np.isnan(water) & (water > (top + 1) * size - 0.25 * size)
        water_top = np.where(
            wet, np.rint(np.where(wet, water, 0.0) / size).astype(np.int64) - 1, top
        )
        beach = (np.abs((top + 1) * size - model.sea) < 2.5 * size) & (step < 2) & ~cold
        top_b = np.where(
            wet, under_b, np.where(beach & (top_b == bid["grass_block"]), bid["sand"], top_b)
        )
        z = (zmin + np.arange(H))[:, None, None]
        T = top[None]
        blocks = np.zeros((H, CHUNK, CHUNK), dtype=np.uint16)
        stone_b = np.where(z < model.sea / size - 40, bid["dark_stone"], bid["stone"])
        blocks[:] = np.where(z <= T - depth[None], stone_b, 0).astype(np.uint16)
        filler = (z > T - depth[None]) & (z < T)
        blocks = np.where(filler, fill_b[None], blocks).astype(np.uint16)
        blocks = np.where(z == T, top_b[None], blocks).astype(np.uint16)
        # Badlands: banded terracotta in the filler.
        terra = fill_b == bid["terracotta"]
        if np.any(terra):
            bands = np.array(
                [
                    bid[k]
                    for k in (
                        "terracotta",
                        "terracotta_red",
                        "terracotta",
                        "terracotta_yellow",
                        "terracotta_white",
                        "terracotta_brown",
                        "terracotta",
                    )
                ]
            )
            band = bands[(z % 7 + 7) % 7]
            blocks = np.where(filler & terra[None], band, blocks).astype(np.uint16)
        # Ores in stone.
        BX = col["bx"][inner][None]
        BY = col["by"][inner][None]
        if b["ores"]:
            u = nz.random3(
                nz.derive(seed, "ores"),
                np.broadcast_to(BX, blocks.shape),
                np.broadcast_to(BY, blocks.shape),
                np.broadcast_to(z, blocks.shape),
            )
            is_stone = (blocks == bid["stone"]) | (blocks == bid["dark_stone"])
            deep = z < model.sea / size - 20
            blocks = np.where(is_stone & (u < 0.012), bid["coal_ore"], blocks)
            blocks = np.where(is_stone & (u >= 0.012) & (u < 0.018), bid["iron_ore"], blocks)
            blocks = np.where(
                is_stone & deep & (u >= 0.018) & (u < 0.0205), bid["gold_ore"], blocks
            ).astype(np.uint16)
        # Bedrock.
        blocks[0] = bid["bedrock"]
        u2 = nz.random2(nz.derive(seed, "bedrock"), col["bx"][inner], col["by"][inner])
        blocks[1] = np.where(u2 < 0.5, bid["bedrock"], blocks[1])
        # Caves.
        cv = d["caves"]
        if cv["enabled"]:
            blocks = self._caves(blocks, z, top, wet, water_top, col, inner, seed, size, cv)
        # Water.
        fill_water = (z > T) & (z <= water_top[None]) & (blocks == AIR)
        blocks = np.where(fill_water, WATER, blocks).astype(np.uint16)
        cold_water = cold & wet
        if np.any(cold_water):  # ice on cold water
            surface = (z == water_top[None]) & cold_water[None] & (blocks == WATER)
            blocks = np.where(surface, bid["ice"], blocks).astype(np.uint16)
        # Tints per column.
        tint = np.stack(
            [
                w @ np.stack([_hex_lin(x["colors"]["grass"]) for x in biomes]),
                w @ np.stack([_hex_lin(x["colors"]["foliage"]) for x in biomes]),
                w @ np.stack([_hex_lin(x["colors"]["water"]) for x in biomes]),
            ],
            axis=2,
        )
        data = ChunkData(cx, cy, zmin, blocks, tint, np.where(wet, water, np.nan))
        if b["trees"] or b["plants"]:
            self._stamp_plants(data, col, top_all, M, names, biomes, seed, size)
        return data

    def _caves(self, blocks, z, top, wet, water_top, col, inner, seed, size, cv):
        zmin = int(z[0, 0, 0])
        lo, hi = cv["depth"]
        x = ((col["bx"][inner] + 0.5) * size)[None]
        y = ((col["by"][inner] + 0.5) * size)[None]
        zz = (z + 0.5) * size
        zc = np.broadcast_to(zz, blocks.shape)
        band = (zc >= model_sea(self.rt) + lo) & (
            zc <= np.broadcast_to(((top[None] + 1) * size) + hi, blocks.shape)
        )
        band &= np.broadcast_to(z > zmin + 3, blocks.shape)
        solid = (blocks != AIR) & band
        if not np.any(solid):
            return blocks
        idx = np.nonzero(solid)
        px = np.broadcast_to(x, blocks.shape)[idx]
        py = np.broadcast_to(y, blocks.shape)[idx]
        pz = zc[idx]
        sc = cv["scale"]
        carve = np.zeros(px.shape, dtype=bool)
        if cv["caverns"] > 0.0:
            n = nz.fbm3(nz.derive(seed, "caverns"), px / sc, py / sc, pz / (sc * 0.55), octaves=3)
            carve |= n > 0.42 - 0.22 * cv["caverns"]
        if cv["tunnels"] > 0.0:
            t1 = nz.gradient3(
                nz.derive(seed, "tunnel-a"), px / (sc * 0.8), py / (sc * 0.8), pz / (sc * 0.5)
            )
            t2 = nz.gradient3(
                nz.derive(seed, "tunnel-b"), px / (sc * 0.8), py / (sc * 0.8), pz / (sc * 0.5)
            )
            wdt = 0.035 + 0.06 * cv["tunnels"]
            carve |= (np.abs(t1) < wdt) & (np.abs(t2) < wdt)
        # Not under water or right below it; entrances only where allowed.
        rows = idx[0]
        jj, ii = idx[1], idx[2]
        under = wet[jj, ii] & (z[rows, 0, 0] >= top[jj, ii] - 3)
        at_top = z[rows, 0, 0] >= top[jj, ii] - 1
        ent = (
            nz.random2(
                nz.derive(seed, "entrance"),
                col["bx"][inner][jj, ii] // 24,
                col["by"][inner][jj, ii] // 24,
            )
            < cv["entrances"]
        )
        carve &= ~under & (~at_top | ent)
        out = blocks.copy()
        out[idx[0][carve], idx[1][carve], idx[2][carve]] = AIR
        return out

    def _stamp_plants(self, data, col, top_all, M, names, biomes, seed, size) -> None:
        d = self.rt.definition
        blocks = data.blocks
        zmin = data.z0
        H = blocks.shape[0]
        w_all = col["weights"]
        n = top_all.shape[0]
        bx_all, by_all = col["bx"], col["by"]
        # Trees: one candidate per 5 x 5 cell (hashed), within the margin too.
        if d["blocks"]["trees"]:
            trees = {}
            for k, biome in enumerate(biomes):
                for e in biome["vegetation"]:
                    if e["species"] in TREE_BLOCKS:
                        trees.setdefault(e["species"], np.zeros(len(biomes)))[k] = e["density"]
            cell = 5
            cand = []
            x0 = int(bx_all[0, 0])
            y0 = int(by_all[0, 0])
            for ci in range(math.floor(x0 / cell), math.floor((x0 + n - 1) / cell) + 1):
                for cj in range(math.floor(y0 / cell), math.floor((y0 + n - 1) / cell) + 1):
                    cand.append((ci, cj))
            cand = np.array(cand, dtype=np.int64)
            sx = cand[:, 0] * cell + np.floor(
                nz.random2(nz.derive(seed, "tree-x"), cand[:, 0], cand[:, 1]) * cell
            ).astype(np.int64)
            sy = cand[:, 1] * cell + np.floor(
                nz.random2(nz.derive(seed, "tree-y"), cand[:, 0], cand[:, 1]) * cell
            ).astype(np.int64)
            li = sx - x0
            lj = sy - y0
            ok = (li >= 0) & (li < n) & (lj >= 0) & (lj < n)
            cand, sx, sy, li, lj = cand[ok], sx[ok], sy[ok], li[ok], lj[ok]
            wv = w_all[lj, li]
            dens = np.zeros(len(cand))
            choice = np.full(len(cand), -1)
            species = list(trees)
            if species:
                per = np.stack([wv @ trees[s] for s in species], axis=1)  # per hectare
                dens = per.sum(axis=1)
                area = (cell * size) ** 2
                p = dens * area / 10000.0 * d["vegetation"]["density"]
                u = nz.random2(nz.derive(seed, "tree-grow"), cand[:, 0], cand[:, 1])
                grow = u < p
                pick = nz.random2(nz.derive(seed, "tree-pick"), cand[:, 0], cand[:, 1])
                cum = np.cumsum(per, axis=1) / np.maximum(dens[:, None], 1e-12)
                choice = np.where(grow, np.argmax(cum > pick[:, None], axis=1), -1)
            water = col["water"]
            temp = col["temperature"]
            for k in np.flatnonzero(choice >= 0):
                j, i = int(lj[k]), int(li[k])
                if not np.isnan(water[j, i]) and water[j, i] > (top_all[j, i] + 1) * size - 0.3:
                    continue
                if temp[j, i] < d["surface"]["treeline_temperature"]:
                    continue
                sp = species[int(choice[k])]
                variant = int(
                    nz.hash2(nz.derive(seed, "tree-v"), np.array([sx[k]]), np.array([sy[k]]))[0]
                    % np.uint64(1000)
                )
                self._tree(data, sp, int(sx[k]), int(sy[k]), int(top_all[j, i]) + 1, variant)
        # Small plants on grass tops.
        if d["blocks"]["plants"]:
            inner = (slice(M, M + CHUNK), slice(M, M + CHUNK))
            top = top_all[inner]
            rows = top + 1 - zmin
            jj, ii = np.mgrid[0:CHUNK, 0:CHUNK]
            valid = (rows > 0) & (rows < H)
            r = np.clip(rows, 0, H - 1)
            below = blocks[np.clip(rows - 1, 0, H - 1), jj, ii]
            above = blocks[r, jj, ii]
            grassy = (
                np.isin(below, [BLOCKS["grass_block"], BLOCKS["dry_grass_block"], BLOCKS["sand"]])
                & (above == AIR)
                & valid
            )
            w = col["weights"][inner]
            covers: dict[str, np.ndarray] = {}
            for k, biome in enumerate(biomes):
                for e in biome["vegetation"]:
                    if e["species"] in PLANT_BLOCKS or e["species"] == "flowers":
                        covers.setdefault(e["species"], np.zeros(len(biomes)))[k] = e["density"]
            u = nz.random2(nz.derive(seed, "plant"), col["bx"][inner], col["by"][inner])
            acc = np.zeros((CHUNK, CHUNK))
            area = size * size / 10000.0 * d["vegetation"]["clutter"] * 0.35
            for sp, dens in covers.items():
                p = (w @ dens) * area
                pick = (u >= acc) & (u < acc + p) & grassy
                acc = acc + p
                if not np.any(pick):
                    continue
                if sp == "flowers":
                    fk = nz.hash2(
                        nz.derive(seed, "flower"), col["bx"][inner], col["by"][inner]
                    ) % np.uint64(3)
                    ids = np.array([BLOCKS[f] for f in FLOWERS])[fk.astype(np.int64)]
                else:
                    ids = np.full((CHUNK, CHUNK), BLOCKS[PLANT_BLOCKS[sp]])
                on_sand = below == BLOCKS["sand"]
                ok = pick & ~(on_sand & (sp != "dead_bush"))
                blocks[r[ok], jj[ok], ii[ok]] = ids[ok]

    def _tree(self, data: ChunkData, species: str, bx: int, by: int, bz: int, v: int) -> None:
        """Stamp a block tree with its trunk base at block (bx, by, bz) into the chunk."""
        log_name, leaf_name, shape = TREE_BLOCKS[species]
        log = BLOCKS[log_name]
        leaf = BLOCKS[leaf_name]
        cells: list[tuple[int, int, int, int]] = []  # (x, y, z, block)
        r = (v % 97) / 97.0
        if shape == "cactus":
            hgt = 2 + v % 3
            cells += [(0, 0, k, log) for k in range(hgt)]
        elif shape == "dead":
            hgt = 3 + v % 3
            cells += [(0, 0, k, log) for k in range(hgt)]
        elif shape == "cone":
            hgt = 6 + v % 5
            cells += [(0, 0, k, log) for k in range(hgt)]
            for k in range(2, hgt + 1):
                rad = max(0, int(round((hgt - k) * 0.38 + (0.6 if k % 2 else 0.0))))
                for x in range(-rad, rad + 1):
                    for y in range(-rad, rad + 1):
                        if abs(x) + abs(y) <= rad + (k % 2) and (x or y or k >= hgt):
                            cells.append((x, y, k, leaf))
            cells.append((0, 0, hgt, leaf))
        elif shape in ("round", "tall", "jungle"):
            hgt = {"round": 4 + v % 3, "tall": 5 + v % 3, "jungle": 9 + v % 6}[shape]
            cells += [(0, 0, k, log) for k in range(hgt)]
            rad = 2 if shape != "jungle" else 3
            for k in range(hgt - 2, hgt + 2):
                rk = rad if k < hgt else rad - 1
                for x in range(-rk, rk + 1):
                    for y in range(-rk, rk + 1):
                        corner = abs(x) == rk and abs(y) == rk
                        if corner and ((x * 31 + y * 17 + k + v) % 3 != 0):
                            continue
                        if x or y or k >= hgt:
                            cells.append((x, y, k, leaf))
        elif shape == "flat":
            hgt = 4 + v % 3
            dx = 1 if v % 2 else -1
            for k in range(hgt):
                cells.append((dx if k >= hgt - 2 else 0, 0, k, log))
            for x in range(-2, 3):
                for y in range(-2, 3):
                    if abs(x) + abs(y) <= 3:
                        cells.append((dx + x, y, hgt, leaf))
            for x in range(-1, 2):
                for y in range(-1, 2):
                    cells.append((dx + x, y, hgt + 1, leaf))
        elif shape == "palm":
            hgt = 6 + v % 4
            cells += [(0, 0, k, log) for k in range(hgt)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                cells.append((dx, dy, hgt - (1 if abs(dx) + abs(dy) == 2 else 0), leaf))
            cells.append((0, 0, hgt, leaf))
        _ = r
        blocks = data.blocks
        H = blocks.shape[0]
        x0 = data.cx * CHUNK
        y0 = data.cy * CHUNK
        for x, y, z, blk in cells:
            i = bx + x - x0
            j = by + y - y0
            k = bz + z - data.z0
            if 0 <= i < CHUNK and 0 <= j < CHUNK and 0 <= k < H:
                cur = blocks[k, j, i]
                if cur == AIR or _PLANT[cur] or (blk == log and cur == leaf):
                    blocks[k, j, i] = blk

    def _apply_edits(self, data: ChunkData) -> None:
        edits = self.rt.world._edits
        if not edits.count:
            return
        H = data.blocks.shape[0]
        for key in edits.sections():
            if key[0] != data.cx or key[1] != data.cy:
                continue
            idx, ids = edits.section(key)
            local = idx.astype(np.int64)
            lx = local % 16
            ly = (local // 16) % 16
            lz = local // 256
            z = key[2] * 16 + lz - data.z0
            ok = (z >= 0) & (z < H)
            if not data.blocks.flags.writeable:
                data.blocks = data.blocks.copy()
            data.blocks[z[ok], ly[ok], lx[ok]] = ids[ok]

    # -- queries -----------------------------------------------------------------------

    def blocks_at(self, cells: np.ndarray) -> np.ndarray:
        out = np.zeros(cells.shape[0], dtype=np.int64)
        cx = np.floor_divide(cells[:, 0], CHUNK)
        cy = np.floor_divide(cells[:, 1], CHUNK)
        for key in set(zip(cx.tolist(), cy.tolist(), strict=True)):
            sel = (cx == key[0]) & (cy == key[1])
            data = self.chunk(*key)
            lx = cells[sel, 0] - key[0] * CHUNK
            ly = cells[sel, 1] - key[1] * CHUNK
            lz = cells[sel, 2] - data.z0
            ok = (lz >= 0) & (lz < data.blocks.shape[0])
            vals = np.zeros(int(sel.sum()), dtype=np.int64)
            vals[ok] = data.blocks[lz[ok], ly[ok], lx[ok]]
            out[sel] = vals
        return out

    def ground(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Top of the highest solid block under each point (metres)."""
        size = self.rt.definition["blocks"]["size"]
        bx = np.floor(x / size).astype(np.int64)
        by = np.floor(y / size).astype(np.int64)
        out = np.zeros(x.shape)
        cx = np.floor_divide(bx, CHUNK)
        cy = np.floor_divide(by, CHUNK)
        for key in set(zip(cx.tolist(), cy.tolist(), strict=True)):
            sel = (cx == key[0]) & (cy == key[1])
            data = self.chunk(*key)
            col = data.blocks[:, by[sel] - key[1] * CHUNK, bx[sel] - key[0] * CHUNK]
            solid = _OPAQUE[col] & ~np.isin(col, _LEAF_IDS)
            idx = np.where(
                solid.any(axis=0), solid.shape[0] - 1 - np.argmax(solid[::-1], axis=0), 0
            )
            out[sel] = (data.z0 + idx + 1) * size
        return out

    # -- meshing -----------------------------------------------------------------------

    def mesh(self, cx: int, cy: int, ctx: Any = None) -> tuple[Mesh | None, Mesh | None]:
        """``(blocks mesh, water mesh)`` of a chunk (cached by the content of the chunk and
        its four neighbours' borders)."""
        key = (
            self._base_key(),
            cx,
            cy,
            self._edit_key(cx, cy),
            *(self._edit_key(cx + a, cy + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))),
        )
        with self._lock:
            found = self._meshes.get(key)
            if found is not None:
                self._meshes.move_to_end(key)
                return found
        t0 = time.perf_counter()
        data = self.chunk(cx, cy, ctx)
        nb = {k: self.chunk(cx + k[0], cy + k[1], ctx) for k in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        result = mesh_chunk(self.rt.definition, data, nb)
        with self._lock:
            self._meshes[key] = result
            while len(self._meshes) > MAX_CHUNKS:
                self._meshes.popitem(last=False)
            self.counters["chunks_meshed"] += 1
            self.counters["seconds_mesh"] += time.perf_counter() - t0
        return result

    def coarse(
        self, level: int, ci: int, cj: int, ctx: Any = None
    ) -> tuple[Mesh | None, Mesh | None]:
        """A distant chunk of ``2^level``-block cells (16 cells per side)."""
        key = (self._base_key(), level, ci, cj)
        with self._lock:
            found = self._coarse.get(key)
            if found is not None:
                self._coarse.move_to_end(key)
                return found
        if ctx is not None:
            ctx.check("block levels of detail")
        result = coarse_chunk(self.rt, level, ci, cj)
        with self._lock:
            self._coarse[key] = result
            while len(self._coarse) > MAX_CHUNKS:
                self._coarse.popitem(last=False)
            self.counters["coarse_meshed"] += 1
        return result


_LEAF_IDS = np.array([k for k, b in enumerate(BLOCK_LIST) if b.name.endswith("_leaves")])


def model_sea(rt: Any) -> float:
    return float(rt.definition["sea_level"])


def _hex_lin(c: str) -> np.ndarray:
    return _rgb(c)


# --------------------------------------------------------------------------------------- #
# Greedy meshing
# --------------------------------------------------------------------------------------- #

# Directions: (axis, sign, face kind 0 top / 1 side / 2 bottom)
_DIRS = ((2, 1, 0), (2, -1, 2), (0, 1, 1), (0, -1, 1), (1, 1, 1), (1, -1, 1))


def _padded(data: ChunkData, nb: dict) -> np.ndarray:
    """The chunk's blocks with a one-block border from its neighbours (``(H+2, 18, 18)``,
    [z, y, x]; outside the vertical range: air above, bedrock below)."""
    b = data.blocks
    H = b.shape[0]
    out = np.zeros((H + 2, CHUNK + 2, CHUNK + 2), dtype=np.uint16)
    out[1:-1, 1:-1, 1:-1] = b
    out[0] = BLOCKS["bedrock"]
    out[1:-1, 1:-1, CHUNK + 1] = nb[(1, 0)].blocks[:, :, 0]
    out[1:-1, 1:-1, 0] = nb[(-1, 0)].blocks[:, :, CHUNK - 1]
    out[1:-1, CHUNK + 1, 1:-1] = nb[(0, 1)].blocks[:, 0, :]
    out[1:-1, 0, 1:-1] = nb[(0, -1)].blocks[:, CHUNK - 1, :]
    return out


def mesh_chunk(d: dict, data: ChunkData, nb: dict) -> tuple[Mesh | None, Mesh | None]:
    size = d["blocks"]["size"]
    P = _padded(data, nb)
    opaque = _OPAQUE[P]
    H = data.blocks.shape[0]
    pos_parts, nrm_parts, uv_parts, col_parts, tri_parts, grp_parts = [], [], [], [], [], []
    count = 0
    cell = P[1:-1, 1:-1, 1:-1].astype(np.int64)
    solid = _OPAQUE[cell]
    tint = data.tint  # (16, 16, 3, 3)
    for axis, sign, kind in _DIRS:
        # Neighbour in the direction (in the padded array).
        sl = [slice(1, -1)] * 3
        arr_axis = {0: 2, 1: 1, 2: 0}[axis]  # world axis -> array axis ([z, y, x])
        sl[arr_axis] = slice(1 + sign, P.shape[arr_axis] - 1 + sign)
        nb_opaque = opaque[tuple(sl)]
        face = solid & ~nb_opaque
        if not np.any(face):
            continue
        tex = _FACE_TEX[cell, kind]
        ao = _ambient_occlusion(opaque, arr_axis, sign)  # (H, 16, 16, 4) levels 0..3
        key = np.where(
            face,
            (tex + 1) * 256 + (ao[..., 0] + 4 * ao[..., 1] + 16 * ao[..., 2] + 64 * ao[..., 3]),
            0,
        )
        quads = _merge(key, arr_axis)
        if quads is None:
            continue
        q = quads
        texi = q["key"] // 256 - 1
        aoc = q["key"] % 256
        p, nrm, uv, colr, tris = _emit(q, arr_axis, sign, size, data, aoc, texi, cell, tint)
        pos_parts.append(p)
        nrm_parts.append(nrm)
        uv_parts.append(uv)
        col_parts.append(colr)
        tri_parts.append(tris + count)
        grp_parts.append(np.repeat(texi, 2))
        count += p.shape[0]
    # Plants: two crossed quads per plant block.
    plants = _PLANT[cell]
    if np.any(plants):
        z, j, i = np.nonzero(plants)
        ids = cell[z, j, i]
        tex = _FACE_TEX[ids, 1]
        p, nrm, uv, colr, tris = _plants(z, j, i, data, size, tint, ids)
        pos_parts.append(p)
        nrm_parts.append(nrm)
        uv_parts.append(uv)
        col_parts.append(colr)
        tri_parts.append(tris + count)
        grp_parts.append(np.repeat(tex, 4))
        count += p.shape[0]
    blocks_mesh = None
    if count:
        blocks_mesh = Mesh(
            np.concatenate(pos_parts).astype(np.float32),
            np.concatenate(tri_parts),
            normals=np.concatenate(nrm_parts).astype(np.float32),
            uvs=np.concatenate(uv_parts).astype(np.float32),
            groups=np.concatenate(grp_parts),
            colors=np.clip(np.concatenate(col_parts), 0.0, 1.0).astype(np.float32),
        )
    water_mesh = _water(P, data, size)
    _ = H
    return blocks_mesh, water_mesh


def _ambient_occlusion(opaque: np.ndarray, arr_axis: int, sign: int) -> np.ndarray:
    """Per face (of the inner cells) and corner, the open-neighbour level 0..3."""
    H = opaque.shape[0] - 2
    other = [a for a in (0, 1, 2) if a != arr_axis]
    out = np.zeros((H, CHUNK, CHUNK, 4), dtype=np.int64)
    # Opaque lookups shifted to the face's neighbour layer, then along the face's axes.
    for c, (da, db) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):

        def shift(oa: int, ob: int) -> np.ndarray:
            sl = [slice(1, -1)] * 3
            sl[arr_axis] = slice(1 + sign, opaque.shape[arr_axis] - 1 + sign)
            o0, o1 = other
            sl[o0] = slice(1 + oa, opaque.shape[o0] - 1 + oa)
            sl[o1] = slice(1 + ob, opaque.shape[o1] - 1 + ob)
            return opaque[tuple(sl)]

        s1 = shift(da, 0)
        s2 = shift(0, db)
        cn = shift(da, db)
        lvl = np.where(s1 & s2, 0, 3 - (s1.astype(np.int64) + s2 + cn))
        out[..., c] = lvl
    return out


def _merge(key: np.ndarray, arr_axis: int) -> dict | None:
    """Rectangles of equal non-zero keys in every slice across ``arr_axis``: runs along the
    last in-plane axis, then identical runs stacked along the other one."""
    other = [a for a in (0, 1, 2) if a != arr_axis]
    k = np.moveaxis(key, [arr_axis, other[0], other[1]], [0, 1, 2])  # (S, R, C)
    S, R, C = k.shape
    flat = k.reshape(S * R, C)
    nonzero = flat != 0
    if not np.any(nonzero):
        return None
    start = nonzero & np.concatenate(
        [np.ones((S * R, 1), bool), flat[:, 1:] != flat[:, :-1]], axis=1
    )
    rows, c0 = np.nonzero(start)
    # Run ends: the next start or the first zero/changed key.
    vals = flat[rows, c0]
    ends = np.empty_like(c0)
    for n in range(rows.shape[0]):
        r, c = rows[n], c0[n]
        e = c + 1
        while e < C and flat[r, e] == vals[n]:
            e += 1
        ends[n] = e
    s = rows // R
    r = rows % R
    order = np.lexsort((r, vals, ends, c0, s))
    s, r, c0, ends, vals = s[order], r[order], c0[order], ends[order], vals[order]
    same = np.zeros(s.shape, dtype=bool)
    same[1:] = (
        (s[1:] == s[:-1])
        & (c0[1:] == c0[:-1])
        & (ends[1:] == ends[:-1])
        & (vals[1:] == vals[:-1])
        & (r[1:] == r[:-1] + 1)
    )
    group = np.cumsum(~same) - 1
    first = np.flatnonzero(~same)
    last = np.concatenate([first[1:] - 1, [s.shape[0] - 1]])
    return {
        "s": s[first],
        "r0": r[first],
        "r1": r[last] + 1,
        "c0": c0[first],
        "c1": ends[first],
        "key": vals[first],
        "other": other,
        "count": len(first),
        "_group": group,
    }


def _emit(q, arr_axis, sign, size, data, aoc, texi, cell, tint):
    """Quads of merged rectangles: positions (chunk-local metres), normals, UVs in blocks,
    colours (AO x tint) and triangles (counter-clockwise outward)."""
    n = q["count"]
    o0, o1 = q["other"]  # array axes of rows and columns
    s, r0, r1, c0, c1 = q["s"], q["r0"], q["r1"], q["c0"], q["c1"]
    # Array coordinates (z, y, x) of the four corners: rows along o0, columns along o1.
    plane = s + (1 if sign > 0 else 0)
    corners = []
    for rr, cc in ((r0, c0), (r0, c1), (r1, c1), (r1, c0)):
        a = np.zeros((n, 3))
        a[:, arr_axis] = plane
        a[:, o0] = rr
        a[:, o1] = cc
        corners.append(a)

    # Array (z, y, x) -> world (x, y, z) metres, z relative to the world floor.
    def world(a):
        return np.stack([a[:, 2] * size, a[:, 1] * size, (a[:, 0] + data.z0) * size], axis=1)

    pts = np.stack([world(a) for a in corners], axis=1)  # (n, 4, 3)
    normal = np.zeros(3)
    normal[{0: 2, 1: 1, 2: 0}[arr_axis]] = sign
    # Orientation: (p1 - p0) x (p3 - p0) must point along the normal.
    e1 = pts[:, 1] - pts[:, 0]
    e2 = pts[:, 3] - pts[:, 0]
    cr = np.cross(e1, e2)
    flip = (cr @ normal) < 0
    pts = np.where(flip[:, None, None], pts[:, [0, 3, 2, 1]], pts)
    # UVs: v along world z for side faces, else along the row axis.
    du = (c1 - c0).astype(np.float64)
    dv = (r1 - r0).astype(np.float64)
    uv = np.stack(
        [
            np.stack([np.zeros(n), np.zeros(n)], 1),
            np.stack([du, np.zeros(n)], 1),
            np.stack([du, dv], 1),
            np.stack([np.zeros(n), dv], 1),
        ],
        axis=1,
    )
    if arr_axis != 0:  # side face: rows run along z (array axis 0) -> v up
        if o0 == 0:
            pass
        else:
            uv = uv[:, :, ::-1]
    uv = np.where(flip[:, None, None], uv[:, [0, 3, 2, 1]], uv)
    # AO corner levels (packed: corner c at 4^c) -> brightness.
    levels = np.stack([(aoc // (4**c)) % 4 for c in range(4)], axis=1)
    # The packed corners were (-,-), (+,-), (+,+), (-,+) along (o_a, o_b) = the two other
    # axes in increasing array order -- the same order as (rows, columns) here.
    bright = AO_LEVELS[levels]
    bright = np.where(flip[:, None], bright[:, [0, 3, 2, 1]], bright)
    # Tint by block type at the rectangle's first cell (grass, foliage, water).
    z0i = np.where(arr_axis == 0, s, 0)
    cells = np.zeros((n, 3), dtype=np.int64)
    cells[:, arr_axis] = s
    cells[:, o0] = r0
    cells[:, o1] = c0
    blk = cell[cells[:, 0], cells[:, 1], cells[:, 2]]
    kind = _TINT[blk]
    _ = z0i
    jx = np.clip(cells[:, 1], 0, CHUNK - 1)
    ix = np.clip(cells[:, 2], 0, CHUNK - 1)
    col = np.ones((n, 3))
    for t in (1, 2):
        sel = kind == t
        if np.any(sel):
            col[sel] = (
                tint[jx[sel], ix[sel], t - 1]
                / np.maximum(tint[jx[sel], ix[sel], t - 1].max(axis=1, keepdims=True), 1e-6)
                * 0.92
            )
    # Side faces of grass blocks keep their own texture colours (only the top is tinted).
    side_face = arr_axis != 0
    grass_side = side_face & (kind == 1)
    col[grass_side] = 1.0
    colors = bright[:, :, None] * col[:, None, :]
    positions = pts.reshape(-1, 3)
    normals = np.broadcast_to(normal, positions.shape)
    base = np.arange(n) * 4
    tris = np.concatenate(
        [np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)]
    )
    order = np.argsort(np.concatenate([np.arange(n) * 2, np.arange(n) * 2 + 1]), kind="stable")
    tris = tris[order]
    alpha = np.ones((n * 4, 1))
    return (
        positions,
        normals,
        uv.reshape(-1, 2),
        np.concatenate([colors.reshape(-1, 3), alpha], axis=1),
        tris,
    )


def _plants(z, j, i, data, size, tint, ids):
    n = z.shape[0]
    x0 = i * size
    y0 = j * size
    zb = (z + data.z0) * size
    a = 0.15 * size
    b = 0.85 * size
    quads = []
    for (xa, ya), (xb, yb) in (((a, a), (b, b)), ((a, b), (b, a))):
        p = np.stack(
            [
                np.stack([x0 + xa, y0 + ya, zb], 1),
                np.stack([x0 + xb, y0 + yb, zb], 1),
                np.stack([x0 + xb, y0 + yb, zb + size], 1),
                np.stack([x0 + xa, y0 + ya, zb + size], 1),
            ],
            axis=1,
        )
        quads.append(p)
    pts = np.concatenate(quads)  # (2n, 4, 3)
    uv = np.broadcast_to(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (2 * n, 4, 2))
    kind = _TINT[np.concatenate([ids, ids])]
    jj = np.concatenate([j, j])
    ii = np.concatenate([i, i])
    col = np.ones((2 * n, 3))
    sel = kind == 1
    if np.any(sel):
        t = tint[jj[sel], ii[sel], 0]
        col[sel] = t / np.maximum(t.max(axis=1, keepdims=True), 1e-6) * 0.9
    colors = np.repeat(col, 4, axis=0)
    e1 = pts[:, 1] - pts[:, 0]
    e2 = pts[:, 3] - pts[:, 0]
    nrm = np.cross(e1, e2)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    normals = np.repeat(nrm, 4, axis=0)
    base = np.arange(2 * n) * 4
    tris = np.concatenate(
        [np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)]
    )
    order = np.argsort(
        np.concatenate([np.arange(2 * n) * 2, np.arange(2 * n) * 2 + 1]), kind="stable"
    )
    return (
        pts.reshape(-1, 3),
        normals,
        uv.reshape(-1, 2),
        np.concatenate([colors, np.ones((8 * n, 1))], axis=1),
        tris[order],
    )


def _water(P: np.ndarray, data: ChunkData, size: float) -> Mesh | None:
    """Top faces of water blocks open above, an eighth of a block low, coloured by depth."""
    inner = P[1:-1, 1:-1, 1:-1]
    above = P[2:, 1:-1, 1:-1]
    top = (inner == WATER) & (above != WATER) & ~_OPAQUE[above]
    if not np.any(top):
        return None
    z, j, i = np.nonzero(top)
    # Depth: water blocks below each surface cell.
    col = inner[:, j, i] == WATER
    depth = np.zeros(z.shape[0])
    for k in range(z.shape[0]):
        zz = z[k]
        d = 0
        while zz - d >= 0 and col[zz - d, k]:
            d += 1
        depth[k] = d
    n = z.shape[0]
    zt = (z + 1 + data.z0) * size - size / 8.0
    x0 = i * size
    y0 = j * size
    pts = np.stack(
        [
            np.stack([x0, y0, zt], 1),
            np.stack([x0 + size, y0, zt], 1),
            np.stack([x0 + size, y0 + size, zt], 1),
            np.stack([x0, y0 + size, zt], 1),
        ],
        axis=1,
    )
    t = data.tint[j, i, 2]
    deep = t * 0.55
    shallow = deep * 0.4 + np.array([0.18, 0.25, 0.22])
    f = nz.smoothstep(0.0, 6.0, depth)[:, None]
    c = np.clip(shallow + f * (deep - shallow), 0.0, 1.0)
    colors = np.concatenate([np.repeat(c, 4, axis=0), np.ones((4 * n, 1))], axis=1)
    base = np.arange(n) * 4
    tris = np.concatenate(
        [np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)]
    )
    normals = np.zeros((4 * n, 3))
    normals[:, 2] = 1.0
    uv = np.broadcast_to(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (n, 4, 2))
    return Mesh(
        pts.reshape(-1, 3).astype(np.float32),
        tris,
        normals=normals.astype(np.float32),
        uvs=uv.reshape(-1, 2).astype(np.float32),
        colors=colors.astype(np.float32),
    )


# --------------------------------------------------------------------------------------- #
# Distant levels of detail
# --------------------------------------------------------------------------------------- #


def coarse_chunk(rt: Any, level: int, ci: int, cj: int) -> tuple[Mesh | None, Mesh | None]:
    """A chunk of 16 x 16 coarse columns of ``s = 2^level`` blocks: tops, walls toward
    lower neighbours, outer walls hanging down; and the water surface."""
    d = rt.definition
    size = d["blocks"]["size"]
    s = 2**level
    cs = s * size
    zmin = d["blocks"]["min_height"]
    n = CHUNK + 2
    k = np.arange(-1, CHUNK + 1)
    X, Y = np.meshgrid((ci * CHUNK + k + 0.5) * cs, (cj * CHUNK + k + 0.5) * cs)
    rt.regional()
    model = rt.model()
    smp = model.evaluate(X.reshape(-1), Y.reshape(-1))
    h = smp["height"].reshape(n, n)
    top = np.rint(h / cs).astype(np.int64)  # coarse units: surface at top * cs
    water = smp["water"].reshape(n, n)
    wet = ~np.isnan(water) & (water > top * cs - 0.25 * cs)
    w = smp["weights"].reshape(n, n, -1)
    table = biome_table(d)
    biomes = list(table.values())
    dom = np.argmax(w, axis=-1)
    temp = smp["temperature"].reshape(n, n)
    top_name = np.array([x["blocks"]["top"] for x in biomes])[dom]
    cold = temp < d["surface"]["snow_temperature"]
    top_name = np.where(cold & (top_name == "grass_block"), "snowy_grass_block", top_name)
    under = np.array([x["blocks"]["underwater"] for x in biomes])[dom]
    top_name = np.where(wet, under, top_name)
    tex_top = np.vectorize(lambda b: _FACE_TEX[BLOCKS[str(b)], 0])(top_name)
    tex_side = np.vectorize(lambda b: _FACE_TEX[BLOCKS[str(b)], 1])(top_name)
    tint = w @ np.stack([_hex_lin(x["colors"]["grass"]) for x in biomes])
    tintn = tint / np.maximum(tint.max(axis=-1, keepdims=True), 1e-6) * 0.92
    kind_top = np.vectorize(lambda b: _TINT[BLOCKS[str(b)]])(top_name)
    pos, nrm, uv, col, tri, grp = [], [], [], [], [], []
    count = 0

    def quad(p4, normal, uvs, color, texture):
        nonlocal count
        pos.append(p4)
        nrm.append(np.broadcast_to(normal, (4, 3)))
        uv.append(uvs)
        col.append(np.broadcast_to(np.append(color, 1.0), (4, 4)))
        tri.append(np.array([[0, 1, 2], [0, 2, 3]]) + count)
        grp.append(np.array([texture, texture]))
        count += 4

    for j in range(1, n - 1):
        for i in range(1, n - 1):
            x0 = (i - 1) * cs
            y0 = (j - 1) * cs
            z = top[j, i] * cs
            c = tintn[j, i] if kind_top[j, i] == 1 else np.ones(3)
            quad(
                np.array([[x0, y0, z], [x0 + cs, y0, z], [x0 + cs, y0 + cs, z], [x0, y0 + cs, z]]),
                (0, 0, 1),
                np.array([[0, 0], [s, 0], [s, s], [0, s]], float),
                c,
                tex_top[j, i],
            )
            for di, dj, normal in (
                (1, 0, (1, 0, 0)),
                (-1, 0, (-1, 0, 0)),
                (0, 1, (0, 1, 0)),
                (0, -1, (0, -1, 0)),
            ):
                nbz = top[j + dj, i + di] * cs
                edge = (i + di in (0, n - 1)) or (j + dj in (0, n - 1))
                low = min(nbz, z) - (2 * cs if edge else 0.0)
                if nbz >= z and not edge:
                    continue
                if low >= z:
                    continue
                if di == 1:
                    a, b = (x0 + cs, y0), (x0 + cs, y0 + cs)
                elif di == -1:
                    a, b = (x0, y0 + cs), (x0, y0)
                elif dj == 1:
                    a, b = (x0 + cs, y0 + cs), (x0, y0 + cs)
                else:
                    a, b = (x0, y0), (x0 + cs, y0)
                hgt = (z - low) / size
                quad(
                    np.array(
                        [[a[0], a[1], low], [b[0], b[1], low], [b[0], b[1], z], [a[0], a[1], z]]
                    ),
                    normal,
                    np.array([[0, 0], [s, 0], [s, hgt], [0, hgt]], float),
                    np.ones(3) * 0.92,
                    tex_side[j, i],
                )
    mesh = None
    if count:
        mesh = Mesh(
            np.concatenate(pos).astype(np.float32),
            np.concatenate(tri),
            normals=np.concatenate(nrm).astype(np.float32),
            uvs=np.concatenate(uv).astype(np.float32),
            groups=np.concatenate(grp),
            colors=np.clip(np.concatenate(col), 0, 1).astype(np.float32),
        )
    # Water.
    wmesh = None
    inner = wet[1:-1, 1:-1]
    if np.any(inner):
        jj, ii = np.nonzero(inner)
        zt = water[1:-1, 1:-1][jj, ii] - size / 8.0
        x0 = ii * cs
        y0 = jj * cs
        m = jj.shape[0]
        pts = np.stack(
            [
                np.stack([x0, y0, zt], 1),
                np.stack([x0 + cs, y0, zt], 1),
                np.stack([x0 + cs, y0 + cs, zt], 1),
                np.stack([x0, y0 + cs, zt], 1),
            ],
            axis=1,
        ).reshape(-1, 3)
        base = np.arange(m) * 4
        tris = np.concatenate(
            [np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)]
        )
        depth = zt - top[1:-1, 1:-1][jj, ii] * cs
        t = w[1:-1, 1:-1][jj, ii] @ np.stack([_hex_lin(x["colors"]["water"]) for x in biomes])
        deep = t * 0.55
        shallow = deep * 0.4 + np.array([0.18, 0.25, 0.22])
        f = nz.smoothstep(0.0, 6.0, depth)[:, None]
        c = np.clip(shallow + f * (deep - shallow), 0.0, 1.0)
        wmesh = Mesh(
            pts.astype(np.float32),
            tris,
            normals=np.broadcast_to(np.array([0, 0, 1.0]), (4 * m, 3)).astype(np.float32),
            colors=np.concatenate([np.repeat(c, 4, axis=0), np.ones((4 * m, 1))], axis=1).astype(
                np.float32
            ),
        )
    _ = zmin
    return mesh, wmesh


# --------------------------------------------------------------------------------------- #
# Runtime entry points
# --------------------------------------------------------------------------------------- #


def _blocks(rt: Any) -> BlockWorld:
    bw = getattr(rt, "_blocks", None)
    if bw is None:
        bw = rt._blocks = BlockWorld(rt)
    return bw


def column_ground(rt: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _blocks(rt).ground(x, y)


def blocks_at(rt: Any, cells: np.ndarray) -> np.ndarray:
    if rt.definition["mode"] != "blocks":
        raise WorldError("NOT_A_BLOCK_WORLD", "this world is a terrain world", "worlds")
    return _blocks(rt).blocks_at(np.asarray(cells, dtype=np.int64))


def _layers_of(mesh: Mesh) -> dict[int, str]:
    return {g: f"block_{_TEXTURES[g]}" for g in mesh.group_ids}


def render_parts(rt: Any, view: Any, ctx: Any = None) -> list:
    """Block chunks near the camera and coarse chunks beyond, within the view distance."""
    d = rt.definition
    bw = _blocks(rt)
    size = d["blocks"]["size"]
    span = CHUNK * size
    eye = view.eye
    world = rt.world
    e = world.extent
    dist = d["streaming"]["view_distance"]
    out = []
    c0x, c0y = math.floor(eye[0] / span), math.floor(eye[1] / span)
    # Level 0: chunks within LOD0_RADIUS chunks (and the view distance).
    r0 = min(LOD0_RADIUS, int(math.ceil(dist / span)))
    near_box = (c0x - r0, c0y - r0, c0x + r0, c0y + r0)
    for cy in range(c0y - r0, c0y + r0 + 1):
        for cx in range(c0x - r0, c0x + r0 + 1):
            if e is not None and (
                (cx + 1) * span <= e[0]
                or cx * span >= e[2]
                or (cy + 1) * span <= e[1]
                or cy * span >= e[3]
            ):
                continue
            if _outside(view, cx * span, cy * span, span, d):
                continue
            mesh, water = bw.mesh(cx, cy, ctx)
            origin = np.array([cx * span, cy * span, 0.0])
            if mesh is not None:
                out.append(
                    (
                        f"world:{world.id}:chunk/{cx},{cy}",
                        f"{world.name} chunk {cx},{cy}",
                        mesh,
                        origin,
                        _layers_of(mesh),
                    )
                )
            if water is not None:
                out.append(
                    (
                        f"world:{world.id}:water/{cx},{cy}",
                        f"{world.name} water {cx},{cy}",
                        water,
                        origin,
                        "water",
                    )
                )
    # Coarse rings: level k covers chunks of 16 * 2^k blocks, outside the finer ring.
    level = 1
    inner = (
        near_box[0] * span,
        near_box[1] * span,
        (near_box[2] + 1) * span,
        (near_box[3] + 1) * span,
    )
    while True:
        cspan = span * 2**level
        reach = min(dist, (r0 + 1) * span * 2**level)
        i0, i1 = math.floor((eye[0] - reach) / cspan), math.floor((eye[0] + reach) / cspan)
        j0, j1 = math.floor((eye[1] - reach) / cspan), math.floor((eye[1] + reach) / cspan)
        for cj in range(j0, j1 + 1):
            for ci in range(i0, i1 + 1):
                bx0, by0 = ci * cspan, cj * cspan
                if (
                    bx0 >= inner[0]
                    and bx0 + cspan <= inner[2]
                    and by0 >= inner[1]
                    and by0 + cspan <= inner[3]
                ):
                    continue
                if e is not None and (
                    bx0 + cspan <= e[0] or bx0 >= e[2] or by0 + cspan <= e[1] or by0 >= e[3]
                ):
                    continue
                cxm = min(max(eye[0], bx0), bx0 + cspan)
                cym = min(max(eye[1], by0), by0 + cspan)
                if math.hypot(cxm - eye[0], cym - eye[1]) > dist:
                    continue
                if _outside(view, bx0, by0, cspan, d):
                    continue
                mesh, water = bw.coarse(level, ci, cj, ctx)
                origin = np.array([bx0, by0, 0.0])
                if mesh is not None:
                    out.append(
                        (
                            f"world:{world.id}:lod{level}/{ci},{cj}",
                            f"{world.name} lod{level} {ci},{cj}",
                            mesh,
                            origin,
                            _layers_of(mesh),
                        )
                    )
                if water is not None:
                    out.append(
                        (
                            f"world:{world.id}:lodwater{level}/{ci},{cj}",
                            f"{world.name} water lod{level} {ci},{cj}",
                            water,
                            origin,
                            "water",
                        )
                    )
        inner = (i0 * cspan, j0 * cspan, (i1 + 1) * cspan, (j1 + 1) * cspan)
        if reach >= dist or level >= 8:
            break
        level += 1
    return out


def _outside(view: Any, x0: float, y0: float, span: float, d: dict) -> bool:
    from .tiles import outside

    b = d["blocks"]
    zlo = b["min_height"] * b["size"]
    zhi = b["max_height"] * b["size"]
    return outside(view, (x0, y0, x0 + span, y0 + span), zlo, zhi)
