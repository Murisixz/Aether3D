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
from functools import lru_cache
from typing import Any

import numpy as np

from ..errors import WorldError
from ..mesh import Mesh
from . import noise as nz
from .blocks import AIR, BLOCK_LIST, BLOCKS, WATER, texture_names, texture_rgba
from .definition import biome_table
from .workers import parallel_map, thread_count

CHUNK = 16
AO_LEVELS = np.array([0.5, 0.66, 0.82, 1.0])
"""Vertex brightness for 0..3 open neighbours of a face corner."""
LOD0_RADIUS = 8
"""Chunks (16 blocks) around the camera meshed block by block; beyond, coarse levels."""
MARGIN = 3
"""Columns around a chunk its generation looks at (slopes, neighbouring trees)."""
CAVE_LATTICE = 4
"""Blocks between the samples of the cave noise (interpolated trilinearly in between)."""
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
        """The blocks of chunk ``(cx, cy)`` (generated and edited; cached)."""
        return self.chunks([(cx, cy)], ctx)[0]

    def chunks(self, coords: list[tuple[int, int]], ctx: Any = None) -> list[ChunkData]:
        """The blocks of several chunks; the missing ones are generated together (one field
        evaluation for all their columns: the same blocks as one at a time)."""
        base = self._base_key()
        keys = [(base, cx, cy, self._edit_key(cx, cy)) for cx, cy in coords]
        out: list[ChunkData | None] = []
        missing: dict[tuple, int] = {}
        with self._lock:
            for k, key in enumerate(keys):
                found = self._chunks.get(key)
                if found is not None:
                    self._chunks.move_to_end(key)
                elif key not in missing:
                    missing[key] = k
                out.append(found)
        if missing:
            if ctx is not None:
                ctx.check("block generation")
            t0 = time.perf_counter()
            todo = [coords[k] for k in missing.values()]
            cols = self._columns_many(todo, MARGIN)
            n = CHUNK + 2 * MARGIN
            regions = [
                (
                    cx * CHUNK - MARGIN,
                    cy * CHUNK - MARGIN,
                    cx * CHUNK - MARGIN + n,
                    cy * CHUNK - MARGIN + n,
                )
                for cx, cy in todo
            ]
            trees = self.trees_many(regions)

            def make(job: tuple) -> ChunkData:
                (cx, cy), col, tr = job
                data = self._generate(cx, cy, col, tr)
                self._apply_edits(data)
                return data

            jobs = list(zip(todo, cols, trees, strict=True))
            done = parallel_map(make, jobs, thread_count(self.rt.definition))
            made = dict(zip(missing, done, strict=True))
            seconds = time.perf_counter() - t0
            with self._lock:
                for key, data in made.items():
                    data.seconds = seconds / len(made)
                    self._chunks[key] = data
                while len(self._chunks) > MAX_CHUNKS:
                    self._chunks.popitem(last=False)
                self.counters["chunks_generated"] += len(made)
                self.counters["seconds_generate"] += seconds
            out = [made.get(key) if o is None else o for key, o in zip(keys, out, strict=True)]
        return out  # type: ignore[return-value]

    def _columns_many(
        self, coords: list[tuple[int, int]], margin: int
    ) -> list[dict[str, np.ndarray]]:
        """Fields at the column centres of chunks plus ``margin`` columns around each."""
        d = self.rt.definition
        size = d["blocks"]["size"]
        n = CHUNK + 2 * margin
        i = np.arange(-margin, CHUNK + margin)
        xs, ys, grids = [], [], []
        for cx, cy in coords:
            bx = cx * CHUNK + i
            by = cy * CHUNK + i
            X, Y = np.meshgrid((bx + 0.5) * size, (by + 0.5) * size)
            xs.append(X.reshape(-1))
            ys.append(Y.reshape(-1))
            grids.append(np.meshgrid(bx, by))
        self.rt.regional()
        s = self.rt.model().evaluate(np.concatenate(xs), np.concatenate(ys))
        out = []
        for k, (gx, gy) in enumerate(grids):
            sl = slice(k * n * n, (k + 1) * n * n)
            col = {
                key: v[sl].reshape((n, n) + v.shape[1:])
                for key, v in s.items()
                if isinstance(v, np.ndarray) and v.shape[:1] == (len(coords) * n * n,)
            }
            col["bx"], col["by"] = gx, gy
            out.append(col)
        return out

    def _generate(self, cx: int, cy: int, col: dict, trees: list) -> ChunkData:
        d = self.rt.definition
        model = self.rt.model()
        b = d["blocks"]
        size = b["size"]
        zmin, zmax = b["min_height"], b["max_height"]
        H = zmax - zmin
        seed = int(d["seed"])
        M = MARGIN
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
            blocks = self._caves(blocks, z, top, wet, col, inner, seed, size, cv)
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
            self._stamp_plants(data, col, top_all, M, biomes, seed, size, trees)
        return data

    def _caves(self, blocks, z, top, wet, col, inner, seed, size, cv):
        """Carve caverns (thresholded fBm) and tunnels (where two noise fields are both near
        zero) inside the cave band, from noise sampled every :data:`CAVE_LATTICE` blocks on
        a global lattice and interpolated trilinearly (seamless across chunks)."""
        H = blocks.shape[0]
        zmin = int(z[0, 0, 0])
        lo, hi = cv["depth"]
        zc = (zmin + np.arange(H) + 0.5) * size
        band_hi = float(((top + 1) * size).max()) + hi
        rows = np.flatnonzero((zc >= model_sea(self.rt) + lo) & (zc <= band_hi))
        rows = rows[rows > 3]
        if not rows.size:
            return blocks
        k0, k1 = int(rows[0]), int(rows[-1])
        L = CAVE_LATTICE
        bx = col["bx"][inner][0]  # (16,) block x of the columns
        by = col["by"][inner][:, 0]
        bz = zmin + np.arange(k0, k1 + 1)
        axes = []
        for b in (bx, by, bz):
            u = (b + 0.5) / L
            i0 = np.floor(u).astype(np.int64)
            nodes = np.arange(int(i0.min()), int(i0.max()) + 2)
            axes.append((nodes, i0 - nodes[0], u - i0))
        (nx_, ix, tx), (ny_, iy, ty), (nz_, iz, tz) = axes
        Z, Y, X = np.meshgrid(nz_ * L * size, ny_ * L * size, nx_ * L * size, indexing="ij")
        X, Y, Z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
        shape = (nz_.size, ny_.size, nx_.size)

        def sample(values: np.ndarray) -> np.ndarray:
            v = values.reshape(shape)
            v = v[:, :, ix] * (1 - tx) + v[:, :, ix + 1] * tx
            v = v[:, iy, :] * (1 - ty)[None, :, None] + v[:, iy + 1, :] * ty[None, :, None]
            return v[iz] * (1 - tz)[:, None, None] + v[iz + 1] * tz[:, None, None]

        sc = cv["scale"]
        carve = np.zeros((bz.size, CHUNK, CHUNK), dtype=bool)
        if cv["caverns"] > 0.0:
            n = nz.fbm3(nz.derive(seed, "caverns"), X / sc, Y / sc, Z / (sc * 0.55), octaves=3)
            carve |= sample(n) > 0.42 - 0.22 * cv["caverns"]
        if cv["tunnels"] > 0.0:
            fx, fy, fz = X / (sc * 0.8), Y / (sc * 0.8), Z / (sc * 0.5)
            t1 = sample(nz.gradient3(nz.derive(seed, "tunnel-a"), fx, fy, fz))
            t2 = sample(nz.gradient3(nz.derive(seed, "tunnel-b"), fx, fy, fz))
            wdt = 0.035 + 0.06 * cv["tunnels"]
            carve |= (np.abs(t1) < wdt) & (np.abs(t2) < wdt)
        # Inside the band; not under water or right below it; entrances where allowed.
        zcol = bz[:, None, None]
        carve &= ((zcol + 0.5) * size <= ((top + 1) * size + hi)[None]) & (
            blocks[k0 : k1 + 1] != AIR
        )
        under = wet[None] & (zcol >= top[None] - 3)
        ent = (
            nz.random2(nz.derive(seed, "entrance"), col["bx"][inner] // 24, col["by"][inner] // 24)
            < cv["entrances"]
        )
        at_top = zcol >= top[None] - 1
        carve &= ~under & (~at_top | ent[None])
        if not np.any(carve):
            return blocks
        out = blocks.copy()
        out[k0 : k1 + 1][carve] = AIR
        return out

    def trees(self, bx0: int, by0: int, bx1: int, by1: int) -> list[tuple[int, int, int, str, int]]:
        """``(x, y, z, species, variant)`` of the block trees whose trunk stands on a column
        of ``[bx0, bx1) x [by0, by1)`` (blocks), ``z`` the block above the ground. One hashed
        candidate per 5 x 5 column cell grows with the probability its biomes' tree
        densities give; never in water, below the tree line or on beaches."""
        return self.trees_many([(bx0, by0, bx1, by1)])[0]

    def trees_many(self, regions: list[tuple[int, int, int, int]]) -> list[list]:
        """:meth:`trees` of several regions (one field evaluation)."""
        d = self.rt.definition
        if not d["blocks"]["trees"] or not regions:
            return [[] for _ in regions]
        b = d["blocks"]
        size = b["size"]
        seed = int(d["seed"])
        biomes = list(biome_table(d).values())
        trees: dict[str, np.ndarray] = {}
        for k, biome in enumerate(biomes):
            for e in biome["vegetation"]:
                if e["species"] in TREE_BLOCKS:
                    trees.setdefault(e["species"], np.zeros(len(biomes)))[k] = e["density"]
        if not trees:
            return [[] for _ in regions]
        cell = 5
        parts = []
        for r, (bx0, by0, bx1, by1) in enumerate(regions):
            ci, cj = np.meshgrid(
                np.arange(math.floor(bx0 / cell), math.floor((bx1 - 1) / cell) + 1),
                np.arange(math.floor(by0 / cell), math.floor((by1 - 1) / cell) + 1),
            )
            ci, cj = ci.reshape(-1).astype(np.int64), cj.reshape(-1).astype(np.int64)
            sx = ci * cell + np.floor(nz.random2(nz.derive(seed, "tree-x"), ci, cj) * cell).astype(
                np.int64
            )
            sy = cj * cell + np.floor(nz.random2(nz.derive(seed, "tree-y"), ci, cj) * cell).astype(
                np.int64
            )
            ok = (sx >= bx0) & (sx < bx1) & (sy >= by0) & (sy < by1)
            parts.append((ci[ok], cj[ok], sx[ok], sy[ok], np.full(int(ok.sum()), r)))
        ci, cj, sx, sy, region = (np.concatenate([p[k] for p in parts]) for k in range(5))
        out: list[list] = [[] for _ in regions]
        if not ci.size:
            return out
        species = list(trees)
        # The same tests the chunks make, on the candidates' columns only.
        u = nz.random2(nz.derive(seed, "tree-grow"), ci, cj)
        self.rt.regional()
        smp = self.rt.model().evaluate((sx + 0.5) * size, (sy + 0.5) * size)
        per = np.stack([smp["weights"] @ trees[sp] for sp in species], axis=1)  # per hectare
        dens = per.sum(axis=1)
        grow = u < dens * (cell * size) ** 2 / 10000.0 * d["vegetation"]["density"]
        top = np.clip(
            np.rint(smp["height"] / size).astype(np.int64) - 1,
            b["min_height"] + 2,
            b["max_height"] - 3,
        )
        water = smp["water"]
        dry = np.isnan(water) | (water <= (top + 1) * size - 0.3)
        warm = smp["temperature"] >= d["surface"]["treeline_temperature"]
        beach = np.abs((top + 1) * size - float(d["sea_level"])) < 2.5 * size
        grow &= dry & warm & ~beach
        pick = nz.random2(nz.derive(seed, "tree-pick"), ci, cj)
        cum = np.cumsum(per, axis=1) / np.maximum(dens[:, None], 1e-12)
        choice = np.argmax(cum > pick[:, None], axis=1)
        variant = nz.hash2(nz.derive(seed, "tree-v"), sx, sy) % np.uint64(1000)
        for k in np.flatnonzero(grow):
            out[int(region[k])].append(
                (int(sx[k]), int(sy[k]), int(top[k]) + 1, species[int(choice[k])], int(variant[k]))
            )
        return out

    def _stamp_plants(self, data, col, top_all, M, biomes, seed, size, trees) -> None:
        d = self.rt.definition
        blocks = data.blocks
        zmin = data.z0
        H = blocks.shape[0]
        # Trees whose trunk stands in the chunk or its margin (they may reach into it).
        for bx, by, bz, sp, v in trees:
            self._tree(data, sp, bx, by, bz, v)
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
        log = BLOCKS[TREE_BLOCKS[species][0]]
        leaf = BLOCKS[TREE_BLOCKS[species][1]]
        blocks = data.blocks
        H = blocks.shape[0]
        x0 = data.cx * CHUNK
        y0 = data.cy * CHUNK
        for x, y, z, is_log in tree_cells(species, v):
            i = bx + x - x0
            j = by + y - y0
            k = bz + z - data.z0
            if 0 <= i < CHUNK and 0 <= j < CHUNK and 0 <= k < H:
                cur = blocks[k, j, i]
                blk = log if is_log else leaf
                if cur == AIR or _PLANT[cur] or (is_log and cur == leaf):
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

    def meshes(self, coords: list[tuple[int, int]], ctx: Any = None) -> list[tuple]:
        """:meth:`mesh` of several chunks: their chunks and neighbours generated together,
        the missing meshes built in parallel."""
        around = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        self.chunks(sorted({(cx + a, cy + b) for cx, cy in coords for a, b in around}), ctx)
        return parallel_map(
            lambda c: self.mesh(c[0], c[1], ctx), coords, thread_count(self.rt.definition)
        )

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
        around = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        got = self.chunks([(cx + a, cy + b) for a, b in around], ctx)
        nb = dict(zip(around[1:], got[1:], strict=True))
        result = mesh_chunk(self.rt.definition, got[0], nb)
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
        return self.coarse_many([(level, ci, cj)], ctx)[0]

    def coarse_many(self, requests: list[tuple[int, int, int]], ctx: Any = None) -> list[tuple]:
        """:meth:`coarse` of several ``(level, ci, cj)``; the missing ones are built together
        (one field evaluation for all their columns and one for their trees)."""
        base = self._base_key()
        out: list[tuple | None] = []
        missing: dict[tuple, int] = {}
        with self._lock:
            for k, (level, ci, cj) in enumerate(requests):
                found = self._coarse.get((base, level, ci, cj))
                if found is not None:
                    self._coarse.move_to_end((base, level, ci, cj))
                elif (base, level, ci, cj) not in missing:
                    missing[(base, level, ci, cj)] = k
                out.append(found)
        if missing:
            if ctx is not None:
                ctx.check("block levels of detail")
            todo = [requests[k] for k in missing.values()]
            size = self.rt.definition["blocks"]["size"]
            n = CHUNK + 2
            k1 = np.arange(-1, CHUNK + 1)
            xs, ys = [], []
            for level, ci, cj in todo:
                cs = 2**level * size
                X, Y = np.meshgrid((ci * CHUNK + k1 + 0.5) * cs, (cj * CHUNK + k1 + 0.5) * cs)
                xs.append(X.reshape(-1))
                ys.append(Y.reshape(-1))
            self.rt.regional()
            smp = self.rt.model().evaluate(np.concatenate(xs), np.concatenate(ys))
            regions = []
            for level, ci, cj in todo:
                span = CHUNK * 2**level
                regions.append((ci * span, cj * span, (ci + 1) * span, (cj + 1) * span))
            trees = self.trees_many(regions)

            def make(t: int) -> tuple:
                level, ci, cj = todo[t]
                sl = slice(t * n * n, (t + 1) * n * n)
                part = {
                    k: v[sl]
                    for k, v in smp.items()
                    if isinstance(v, np.ndarray) and v.shape[:1] == (len(todo) * n * n,)
                }
                return coarse_chunk(self.rt, level, ci, cj, part, trees[t])

            done = parallel_map(make, range(len(todo)), thread_count(self.rt.definition))
            made = dict(zip(missing, done, strict=True))
            with self._lock:
                for key, result in made.items():
                    self._coarse[key] = result
                while len(self._coarse) > MAX_CHUNKS:
                    self._coarse.popitem(last=False)
                self.counters["coarse_meshed"] += len(made)
            out = [
                made.get((base, *r)) if o is None else o for r, o in zip(requests, out, strict=True)
            ]
        return out  # type: ignore[return-value]


_LEAF_IDS = np.array([k for k, b in enumerate(BLOCK_LIST) if b.name.endswith("_leaves")])


@lru_cache(maxsize=4096)
def tree_cells(species: str, v: int) -> tuple[tuple[int, int, int, bool], ...]:
    """The cells of a block tree, ``(x, y, z, is_log)`` relative to its trunk base."""
    shape = TREE_BLOCKS[species][2]
    cells: list[tuple[int, int, int, bool]] = []
    if shape in ("cactus", "dead"):
        hgt = (2 if shape == "cactus" else 3) + v % 3
        cells += [(0, 0, k, True) for k in range(hgt)]
    elif shape == "cone":
        hgt = 6 + v % 5
        cells += [(0, 0, k, True) for k in range(hgt)]
        for k in range(2, hgt + 1):
            rad = max(0, int(round((hgt - k) * 0.38 + (0.6 if k % 2 else 0.0))))
            for x in range(-rad, rad + 1):
                for y in range(-rad, rad + 1):
                    if abs(x) + abs(y) <= rad + (k % 2) and (x or y or k >= hgt):
                        cells.append((x, y, k, False))
        cells.append((0, 0, hgt, False))
    elif shape in ("round", "tall", "jungle"):
        hgt = {"round": 4 + v % 3, "tall": 5 + v % 3, "jungle": 9 + v % 6}[shape]
        cells += [(0, 0, k, True) for k in range(hgt)]
        rad = 2 if shape != "jungle" else 3
        for k in range(hgt - 2, hgt + 2):
            rk = rad if k < hgt else rad - 1
            for x in range(-rk, rk + 1):
                for y in range(-rk, rk + 1):
                    corner = abs(x) == rk and abs(y) == rk
                    if corner and ((x * 31 + y * 17 + k + v) % 3 != 0):
                        continue
                    if x or y or k >= hgt:
                        cells.append((x, y, k, False))
    elif shape == "flat":
        hgt = 4 + v % 3
        dx = 1 if v % 2 else -1
        for k in range(hgt):
            cells.append((dx if k >= hgt - 2 else 0, 0, k, True))
        for x in range(-2, 3):
            for y in range(-2, 3):
                if abs(x) + abs(y) <= 3:
                    cells.append((dx + x, y, hgt, False))
        for x in range(-1, 2):
            for y in range(-1, 2):
                cells.append((dx + x, y, hgt + 1, False))
    elif shape == "palm":
        hgt = 6 + v % 4
        cells += [(0, 0, k, True) for k in range(hgt)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            cells.append((dx, dy, hgt - (1 if abs(dx) + abs(dy) == 2 else 0), False))
        cells.append((0, 0, hgt, False))
    return tuple(cells)


@lru_cache(maxsize=4096)
def tree_boxes(species: str, v: int) -> tuple[tuple[int, int, int, int, int, int, bool], ...]:
    """A block tree as at most two boxes for distant levels of detail: its crown's box and
    the visible part of its trunk, ``(x0, y0, z0, x1, y1, z1, is_log)`` (exclusive ends)."""
    cells = tree_cells(species, v)
    out = []
    leaves = np.array([c[:3] for c in cells if not c[3]], dtype=np.int64).reshape(-1, 3)
    logs = np.array([c[:3] for c in cells if c[3]], dtype=np.int64).reshape(-1, 3)
    crown_z = int(leaves[:, 2].min()) if leaves.size else 10**6
    if logs.size:
        below = logs[logs[:, 2] < crown_z]
        if below.size:
            lo, hi = below.min(axis=0), below.max(axis=0) + 1
            out.append((*lo.tolist(), *hi.tolist(), True))
    if leaves.size:
        lo, hi = leaves.min(axis=0), leaves.max(axis=0) + 1
        out.append((*lo.tolist(), *hi.tolist(), False))
    return tuple(out)


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
    """``(blocks mesh, water mesh)`` of a chunk, in chunk-local metres."""
    size = d["blocks"]["size"]
    P = _padded(data, nb)
    Hp = P.shape[0]
    opaque_all = _OPAQUE[P]
    # Only rows that can show something: not buried between fully opaque rows, not empty.
    full = opaque_all.reshape(Hp, -1).all(axis=1)
    filled = (P != AIR).reshape(Hp, -1).any(axis=1)
    need = filled[1:-1] & ~(full[:-2] & full[1:-1] & full[2:])
    if not np.any(need):
        return None, None
    rows = np.flatnonzero(need)
    r0, r1 = int(rows[0]), int(rows[-1])
    P = P[r0 : r1 + 3]  # inner rows r0..r1 with one row of padding on each side
    z0 = data.z0 + r0
    opaque = opaque_all[r0 : r1 + 3]
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
        face = solid & ~opaque[tuple(sl)]
        if not np.any(face):
            continue
        tex = _FACE_TEX[cell, kind]
        ao = _ambient_occlusion(opaque, arr_axis, sign)  # (H, 16, 16, 4) levels 0..3
        key = np.where(
            face,
            (tex + 1) * 256 + (ao[..., 0] + 4 * ao[..., 1] + 16 * ao[..., 2] + 64 * ao[..., 3]),
            0,
        )
        q = _merge(key, arr_axis)
        if q is None:
            continue
        texi = q["key"] // 256 - 1
        aoc = q["key"] % 256
        p, nrm, uv, colr, tris = _emit(q, arr_axis, sign, size, z0, aoc, texi, cell, tint)
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
        p, nrm, uv, colr, tris = _plants(z, j, i, z0, size, tint, ids)
        pos_parts.append(p)
        nrm_parts.append(nrm)
        uv_parts.append(uv)
        col_parts.append(colr)
        tri_parts.append(tris + count)
        grp_parts.append(np.repeat(_FACE_TEX[ids, 1], 4))
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
    return blocks_mesh, _water(P, z0, data.tint, size)


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
    vals = flat[rows, c0]
    # Run ends: the next change of key along the row (or the row's end).
    change = np.ones((S * R, C + 1), dtype=bool)
    change[:, 1:C] = flat[:, 1:] != flat[:, :-1]
    marks = np.flatnonzero(change.reshape(-1))
    at = rows * (C + 1) + c0
    ends = marks[np.searchsorted(marks, at, side="right")] - rows * (C + 1)
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
    }


def _emit(q, arr_axis, sign, size, z0, aoc, texi, cell, tint):
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
        return np.stack([a[:, 2] * size, a[:, 1] * size, (a[:, 0] + z0) * size], axis=1)

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
    cells = np.zeros((n, 3), dtype=np.int64)
    cells[:, arr_axis] = s
    cells[:, o0] = r0
    cells[:, o1] = c0
    blk = cell[cells[:, 0], cells[:, 1], cells[:, 2]]
    kind = _TINT[blk]
    jx = np.clip(cells[:, 1], 0, CHUNK - 1)
    ix = np.clip(cells[:, 2], 0, CHUNK - 1)
    col = np.ones((n, 3))
    # Tinted (grey) textures take the biome colour as their mean colour; the side faces of
    # grass blocks keep their own texture (only the tops are tinted).
    for t in (1, 2):
        sel = (kind == t) & ((arr_axis == 0) | (t != 1))
        if np.any(sel):
            col[sel] = tint[jx[sel], ix[sel], t - 1] / _tex_mean()[texi[sel]]
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


def _plants(z, j, i, z0, size, tint, ids):
    n = z.shape[0]
    x0 = i * size
    y0 = j * size
    zb = (z + z0) * size
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
        tex = _FACE_TEX[np.concatenate([ids, ids]), 1]
        col[sel] = tint[jj[sel], ii[sel], 0] / _tex_mean()[tex[sel]]
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


def _water(P: np.ndarray, z0: int, tint: np.ndarray, size: float) -> Mesh | None:
    """Top faces of water blocks open above, an eighth of a block low, coloured by depth."""
    inner = P[1:-1, 1:-1, 1:-1]
    above = P[2:, 1:-1, 1:-1]
    top = (inner == WATER) & (above != WATER) & ~_OPAQUE[above]
    if not np.any(top):
        return None
    # Depth: consecutive water blocks down from each block.
    wet = inner == WATER
    run = np.zeros(wet.shape, dtype=np.int64)
    run[0] = wet[0]
    for r in range(1, wet.shape[0]):
        run[r] = (run[r - 1] + 1) * wet[r]
    z, j, i = np.nonzero(top)
    depth = run[z, j, i].astype(np.float64)
    n = z.shape[0]
    zt = (z + 1 + z0) * size - size / 8.0
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
    c = _water_color(tint[j, i, 2], depth)
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


def _water_color(biome_water: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Linear water colours: clear shallows, the biome's water colour in the deep."""
    deep = biome_water * 0.55
    shallow = deep * 0.4 + np.array([0.18, 0.25, 0.22])
    f = nz.smoothstep(0.0, 6.0, depth)[:, None]
    return np.clip(shallow + f * (deep - shallow), 0.0, 1.0)


@lru_cache(maxsize=1)
def _tex_mean() -> np.ndarray:
    """Mean linear colour of every block texture (alpha-weighted), the reference its tint
    is relative to."""
    from ..rendering.color import srgb_to_linear

    out = np.ones((len(_TEXTURES), 3))
    for k, name in enumerate(_TEXTURES):
        rgba = texture_rgba(name).astype(np.float64) / 255.0
        a = rgba[..., 3:4]
        lin = srgb_to_linear(rgba[..., :3])
        out[k] = np.maximum((lin * a).sum(axis=(0, 1)) / max(float(a.sum()), 1e-9), 1e-3)
    return out


# --------------------------------------------------------------------------------------- #
# Distant levels of detail
# --------------------------------------------------------------------------------------- #


def coarse_chunk(
    rt: Any, level: int, ci: int, cj: int, smp: dict | None = None, trees: list | None = None
) -> tuple[Mesh | None, Mesh | None]:
    """A distant chunk of 16 x 16 columns of ``s = 2^level`` blocks: their tops, walls
    toward lower neighbours, outer walls hanging ``2 s`` blocks down (hiding the steps to the
    next level), the trees of the area as boxes, and the water surface."""
    d = rt.definition
    size = d["blocks"]["size"]
    s = 2**level
    cs = s * size
    n = CHUNK + 2
    k = np.arange(-1, CHUNK + 1)
    if smp is None:  # (fields at the column centres, when not given)
        X, Y = np.meshgrid((ci * CHUNK + k + 0.5) * cs, (cj * CHUNK + k + 0.5) * cs)
        rt.regional()
        smp = rt.model().evaluate(X.reshape(-1), Y.reshape(-1))
    h = smp["height"].reshape(n, n)
    top = np.rint(h / cs).astype(np.int64)  # surface at top * cs
    water = smp["water"].reshape(n, n)
    wet = ~np.isnan(water) & (water > top * cs - 0.25 * cs)
    w = smp["weights"].reshape(n, n, -1)
    biomes = list(biome_table(d).values())
    dom = np.argmax(w, axis=-1)
    cold = smp["temperature"].reshape(n, n) < d["surface"]["snow_temperature"]
    step = np.zeros((n, n), dtype=np.int64)
    step[1:-1, 1:-1] = np.max(
        np.abs(
            np.stack([top[1:-1, 2:], top[1:-1, :-2], top[2:, 1:-1], top[:-2, 1:-1]])
            - top[1:-1, 1:-1]
        ),
        axis=0,
    )
    top_id = np.array([BLOCKS[x["blocks"]["top"]] for x in biomes])[dom]
    fill_id = np.array([BLOCKS[x["blocks"]["filler"]] for x in biomes])[dom]
    top_id = np.where(cold & (top_id == BLOCKS["grass_block"]), BLOCKS["snowy_grass_block"], top_id)
    cliff = step * s >= 3
    top_id = np.where(cliff & ~cold, BLOCKS["stone"], top_id)
    fill_id = np.where(cliff, BLOCKS["stone"], fill_id)
    top_id = np.where(
        wet, np.array([BLOCKS[x["blocks"]["underwater"]] for x in biomes])[dom], top_id
    )
    grass_tint = w @ np.stack([_hex_lin(x["colors"]["grass"]) for x in biomes])
    foliage_tint = w @ np.stack([_hex_lin(x["colors"]["foliage"]) for x in biomes])
    quads: list[
        tuple
    ] = []  # (corners (m, 4, 3), normal, uv (m, 4, 2), colour (m, 3), texture (m,))
    jj, ii = np.mgrid[1 : n - 1, 1 : n - 1]
    jj, ii = jj.reshape(-1), ii.reshape(-1)
    x0 = (ii - 1) * cs
    y0 = (jj - 1) * cs
    z = top[jj, ii] * cs
    m = jj.size
    corners = np.stack(
        [
            np.stack([x0, y0, z], 1),
            np.stack([x0 + cs, y0, z], 1),
            np.stack([x0 + cs, y0 + cs, z], 1),
            np.stack([x0, y0 + cs, z], 1),
        ],
        1,
    )
    uv = np.broadcast_to(np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]]), (m, 4, 2))
    tex = _FACE_TEX[top_id[jj, ii], 0]
    col = np.ones((m, 3))
    tinted = _TINT[top_id[jj, ii]] == 1
    col[tinted] = grass_tint[jj, ii][tinted] / _tex_mean()[tex[tinted]]
    quads.append((corners, (0.0, 0.0, 1.0), uv, col, tex))
    # Walls toward lower neighbours; at the chunk's edges always, hanging down.
    for di, dj, normal in (
        (1, 0, (1, 0, 0)),
        (-1, 0, (-1, 0, 0)),
        (0, 1, (0, 1, 0)),
        (0, -1, (0, -1, 0)),
    ):
        nb = top[jj + dj, ii + di] * cs
        edge = ((ii + di) == 0) | ((ii + di) == n - 1) | ((jj + dj) == 0) | ((jj + dj) == n - 1)
        low = np.minimum(nb, z) - np.where(edge, 2 * cs, 0.0)
        sel = low < z
        if not np.any(sel):
            continue
        xs, ys, zt, zb = x0[sel], y0[sel], z[sel], low[sel]
        if di == 1:
            a, b = (xs + cs, ys), (xs + cs, ys + cs)
        elif di == -1:
            a, b = (xs, ys + cs), (xs, ys)
        elif dj == 1:
            a, b = (xs + cs, ys + cs), (xs, ys + cs)
        else:
            a, b = (xs, ys), (xs + cs, ys)
        c4 = np.stack(
            [
                np.stack([a[0], a[1], zb], 1),
                np.stack([b[0], b[1], zb], 1),
                np.stack([b[0], b[1], zt], 1),
                np.stack([a[0], a[1], zt], 1),
            ],
            1,
        )
        hgt = (zt - zb) / size
        k4 = hgt.size
        uv4 = np.stack(
            [
                np.zeros((k4, 2)),
                np.stack([np.full(k4, float(s)), np.zeros(k4)], 1),
                np.stack([np.full(k4, float(s)), hgt], 1),
                np.stack([np.zeros(k4), hgt], 1),
            ],
            1,
        )
        side = _FACE_TEX[fill_id[jj, ii][sel], 1]
        quads.append((c4, normal, uv4, np.full((k4, 3), 0.86), side))
    # Trees as boxes.
    span = CHUNK * s
    if trees is None:
        trees = _blocks(rt).trees(ci * span, cj * span, (ci + 1) * span, (cj + 1) * span)
    for bx, by, bz, sp, v in trees:
        log_tex = _FACE_TEX[BLOCKS[TREE_BLOCKS[sp][0]], 1]
        leaf_tex = _FACE_TEX[BLOCKS[TREE_BLOCKS[sp][1]], 1]
        for bx0, by0, bz0, bx1, by1, bz1, is_log in tree_boxes(sp, v):
            if is_log and level > 2:
                continue
            lo = np.array([bx + bx0 - ci * span, by + by0 - cj * span, bz + bz0], float) * size
            hi = np.array([bx + bx1 - ci * span, by + by1 - cj * span, bz + bz1], float) * size
            t = log_tex if is_log else leaf_tex
            if is_log:
                c = np.ones(3)
            else:
                f = foliage_tint[
                    min(max((by - cj * span) // s + 1, 0), n - 1),
                    min(max((bx - ci * span) // s + 1, 0), n - 1),
                ]
                c = f / _tex_mean()[t]
            quads.extend(_box_quads(lo, hi, c, t, size))
    pos = np.concatenate([q[0].reshape(-1, 3) for q in quads])
    nrm = np.concatenate(
        [np.broadcast_to(np.asarray(q[1], float), (q[0].shape[0] * 4, 3)) for q in quads]
    )
    uvs = np.concatenate([np.asarray(q[2], float).reshape(-1, 2) for q in quads])
    cols = np.concatenate([np.repeat(q[3], 4, axis=0) for q in quads])
    texs = np.concatenate([q[4] for q in quads])
    count = texs.size
    base = np.arange(count) * 4
    tris = np.stack(
        [np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)], 1
    )
    mesh = Mesh(
        pos.astype(np.float32),
        tris.reshape(-1, 3),
        normals=nrm.astype(np.float32),
        uvs=uvs.astype(np.float32),
        groups=np.repeat(texs, 2),
        colors=np.concatenate([np.clip(cols, 0.0, 1.0), np.ones((count * 4, 1))], 1).astype(
            np.float32
        ),
    )
    # Water.
    wmesh = None
    inner = wet[1:-1, 1:-1]
    if np.any(inner):
        wj, wi = np.nonzero(inner)
        zt = water[1:-1, 1:-1][wj, wi] - size / 8.0
        wx = wi * cs
        wy = wj * cs
        mw = wj.shape[0]
        pts = np.stack(
            [
                np.stack([wx, wy, zt], 1),
                np.stack([wx + cs, wy, zt], 1),
                np.stack([wx + cs, wy + cs, zt], 1),
                np.stack([wx, wy + cs, zt], 1),
            ],
            axis=1,
        ).reshape(-1, 3)
        base = np.arange(mw) * 4
        wtris = np.concatenate(
            [np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)]
        )
        depth = zt - top[1:-1, 1:-1][wj, wi] * cs
        biome_water = w[1:-1, 1:-1][wj, wi] @ np.stack(
            [_hex_lin(x["colors"]["water"]) for x in biomes]
        )
        c = _water_color(biome_water, depth / size)
        wmesh = Mesh(
            pts.astype(np.float32),
            wtris,
            normals=np.broadcast_to(np.array([0, 0, 1.0]), (4 * mw, 3)).astype(np.float32),
            colors=np.concatenate([np.repeat(c, 4, axis=0), np.ones((4 * mw, 1))], axis=1).astype(
                np.float32
            ),
        )
    return mesh, wmesh


def _box_quads(lo: np.ndarray, hi: np.ndarray, color: np.ndarray, tex: int, size: float) -> list:
    """The five visible faces of a box (no bottom) as quads for :func:`coarse_chunk`."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    sx, sy, sz = (x1 - x0) / size, (y1 - y0) / size, (z1 - z0) / size
    faces = [
        ([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], (0, 0, 1), sx, sy, 1.0),
        ([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], (1, 0, 0), sy, sz, 0.86),
        ([[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]], (-1, 0, 0), sy, sz, 0.86),
        ([[x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]], (0, 1, 0), sx, sz, 0.86),
        ([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], (0, -1, 0), sx, sz, 0.86),
    ]
    out = []
    for c4, normal, u, v, shade in faces:
        uv = np.array([[[0.0, 0.0], [u, 0.0], [u, v], [0.0, v]]])
        out.append((np.array([c4], float), normal, uv, np.array([color * shade]), np.array([tex])))
    return out


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
    """Block chunks near the camera and coarse chunks beyond, within the view distance
    (everything missing is generated in batches)."""
    d = rt.definition
    bw = _blocks(rt)
    size = d["blocks"]["size"]
    span = CHUNK * size
    eye = view.eye
    world = rt.world
    e = world.extent
    dist = d["streaming"]["view_distance"]

    def outside_extent(x0: float, y0: float, w: float) -> bool:
        return e is not None and (x0 + w <= e[0] or x0 >= e[2] or y0 + w <= e[1] or y0 >= e[3])

    # Level 0: chunks within LOD0_RADIUS chunks (and the view distance).
    c0x, c0y = math.floor(eye[0] / span), math.floor(eye[1] / span)
    r0 = min(LOD0_RADIUS, int(math.ceil(dist / span)))
    near = [
        (cx, cy)
        for cy in range(c0y - r0, c0y + r0 + 1)
        for cx in range(c0x - r0, c0x + r0 + 1)
        if not outside_extent(cx * span, cy * span, span)
        and not _outside(view, cx * span, cy * span, span, d)
    ]
    # Coarse rings: level k covers chunks of 16 * 2^k blocks, outside the finer ring.
    far = []
    inner = ((c0x - r0) * span, (c0y - r0) * span, (c0x + r0 + 1) * span, (c0y + r0 + 1) * span)
    level = 1
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
                ) or outside_extent(bx0, by0, cspan):
                    continue
                cxm = min(max(eye[0], bx0), bx0 + cspan)
                cym = min(max(eye[1], by0), by0 + cspan)
                if math.hypot(cxm - eye[0], cym - eye[1]) > dist:
                    continue
                if not _outside(view, bx0, by0, cspan, d):
                    far.append((level, ci, cj))
        inner = (i0 * cspan, j0 * cspan, (i1 + 1) * cspan, (j1 + 1) * cspan)
        if reach >= dist or level >= 8:
            break
        level += 1
    # Generate what is missing in batches, then mesh.
    out = []
    for (cx, cy), (mesh, water) in zip(near, bw.meshes(near, ctx), strict=True):
        origin = np.array([cx * span, cy * span, 0.0])
        tag = f"{cx},{cy}"
        if mesh is not None:
            out.append(
                (
                    f"world:{world.id}:chunk/{tag}",
                    f"{world.name} chunk {tag}",
                    mesh,
                    origin,
                    _layers_of(mesh),
                )
            )
        if water is not None:
            out.append(
                (
                    f"world:{world.id}:water/{tag}",
                    f"{world.name} water {tag}",
                    water,
                    origin,
                    "water",
                )
            )
    for (level, ci, cj), (mesh, water) in zip(far, bw.coarse_many(far, ctx), strict=True):
        cspan = span * 2**level
        origin = np.array([ci * cspan, cj * cspan, 0.0])
        tag = f"{level}/{ci},{cj}"
        if mesh is not None:
            out.append(
                (
                    f"world:{world.id}:lod{tag}",
                    f"{world.name} lod {tag}",
                    mesh,
                    origin,
                    _layers_of(mesh),
                )
            )
        if water is not None:
            out.append(
                (
                    f"world:{world.id}:lodwater{tag}",
                    f"{world.name} water lod {tag}",
                    water,
                    origin,
                    "water",
                )
            )
    return out


def _outside(view: Any, x0: float, y0: float, span: float, d: dict) -> bool:
    from .tiles import outside

    b = d["blocks"]
    zlo = b["min_height"] * b["size"]
    zhi = b["max_height"] * b["size"]
    return outside(view, (x0, y0, x0 + span, y0 + span), zlo, zhi)
