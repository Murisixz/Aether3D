"""The block registry of block worlds, and their original pixel-art textures.

Every block has a stable integer id (its index in :data:`BLOCK_LIST`; ``0`` is air), a
name, a *kind* -- ``solid`` (an opaque cube), ``liquid`` (water: rendered as its exposed
surface), ``plant`` (two crossed alpha-masked quads, not solid) -- the textures of its top,
sides and bottom, an optional biome *tint* (grass and leaves take the biome's colour through
vertex colours, as in Minecraft-style games) and an optional emission.

Textures are 16 x 16 pixel art painted here from integer hashes (no image is downloaded or
bundled; nothing here comes from Minecraft). :func:`texture_png` returns one as PNG bytes;
the world system turns each into a scene texture sampled with ``nearest`` filtering and
mipmaps (crisp pixels up close, no shimmering far away) and repeat wrapping (one texture
period per block on merged faces).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import noise as nz

TEXTURE_SIZE = 16


@dataclass(frozen=True)
class Block:
    name: str
    kind: str  # "air" | "solid" | "liquid" | "plant"
    top: str = ""
    side: str = ""
    bottom: str = ""
    tint: str | None = None  # "grass" | "foliage" | "water" (biome colours)
    emission: float = 0.0
    roughness: float = 0.9

    @property
    def opaque(self) -> bool:
        return self.kind == "solid"

    @property
    def textures(self) -> tuple[str, str, str]:
        return (self.top, self.side, self.bottom)


def _solid(name: str, top: str | None = None, side: str | None = None,
           bottom: str | None = None, **kw) -> Block:  # fmt: skip
    t = top or name
    return Block(name, "solid", t, side or t, bottom or t, **kw)


BLOCK_LIST: tuple[Block, ...] = (
    Block("air", "air"),
    _solid("stone"),
    _solid("cobblestone"),
    _solid("dark_stone"),
    _solid("bedrock"),
    _solid("dirt"),
    _solid("grass_block", "grass_top", "grass_side", "dirt", tint="grass"),
    _solid("snowy_grass_block", "snow", "snowy_grass_side", "dirt"),
    _solid("dry_grass_block", "dry_grass_top", "dry_grass_side", "dirt", tint="grass"),
    _solid("sand"),
    _solid("red_sand"),
    _solid("sandstone", "sandstone_top", "sandstone", "sandstone_top"),
    _solid("gravel"),
    _solid("clay"),
    _solid("mud"),
    _solid("snow_block", "snow"),
    _solid("ice", roughness=0.15),
    _solid("packed_ice", roughness=0.3),
    Block("water", "liquid", "water", "water", "water", tint="water", roughness=0.08),
    _solid("lava", emission=3.0, roughness=0.6),
    _solid("coal_ore"),
    _solid("iron_ore"),
    _solid("gold_ore"),
    _solid("terracotta"),
    _solid("terracotta_red"),
    _solid("terracotta_yellow"),
    _solid("terracotta_white"),
    _solid("terracotta_brown"),
    _solid("oak_log", "oak_log_top", "oak_log", "oak_log_top"),
    _solid("birch_log", "birch_log_top", "birch_log", "birch_log_top"),
    _solid("spruce_log", "spruce_log_top", "spruce_log", "spruce_log_top"),
    _solid("jungle_log", "jungle_log_top", "jungle_log", "jungle_log_top"),
    _solid("acacia_log", "acacia_log_top", "acacia_log", "acacia_log_top"),
    _solid("oak_leaves", tint="foliage", roughness=0.8),
    _solid("birch_leaves", tint="foliage", roughness=0.8),
    _solid("spruce_leaves", tint="foliage", roughness=0.8),
    _solid("jungle_leaves", tint="foliage", roughness=0.8),
    _solid("acacia_leaves", tint="foliage", roughness=0.8),
    _solid("cactus", "cactus_top", "cactus_side", "cactus_top"),
    _solid("oak_planks"),
    _solid("spruce_planks"),
    _solid("stone_bricks"),
    _solid("mossy_stone_bricks"),
    _solid("bricks"),
    _solid("glass", roughness=0.1),
    _solid("white_wool"),
    _solid("red_wool"),
    _solid("hay_block", "hay_top", "hay_side", "hay_top"),
    _solid("dirt_path", "dirt_path_top", "dirt", "dirt"),
    _solid("glowstone", emission=2.5),
    _solid("lantern_block", emission=2.0),
    Block("tall_grass", "plant", side="tall_grass", tint="grass"),
    Block("fern", "plant", side="fern", tint="grass"),
    Block("poppy", "plant", side="poppy"),
    Block("dandelion", "plant", side="dandelion"),
    Block("cornflower", "plant", side="cornflower"),
    Block("dead_bush", "plant", side="dead_bush"),
    Block("red_mushroom", "plant", side="red_mushroom"),
    Block("brown_mushroom", "plant", side="brown_mushroom"),
)
"""Every block; a block's id is its index (stable: new blocks are only ever appended)."""

BLOCKS: dict[str, int] = {b.name: i for i, b in enumerate(BLOCK_LIST)}
AIR = 0
WATER = BLOCKS["water"]


def block(name: str) -> int:
    return BLOCKS[name]


def tables() -> dict[str, np.ndarray]:
    """Per-id arrays the mesher uses: ``opaque``, ``liquid``, ``plant``, ``solid``."""
    kinds = [b.kind for b in BLOCK_LIST]
    return {
        "opaque": np.array([k == "solid" for k in kinds]),
        "liquid": np.array([k == "liquid" for k in kinds]),
        "plant": np.array([k == "plant" for k in kinds]),
        "solid": np.array([k == "solid" for k in kinds]),
    }


def texture_names() -> list[str]:
    """Every texture any block uses, in a stable order."""
    seen: dict[str, None] = {}
    for b in BLOCK_LIST[1:]:
        for t in (b.top, b.side, b.bottom):
            if t:
                seen[t] = None
    return list(seen)


# --------------------------------------------------------------------------------------- #
# Pixel art
# --------------------------------------------------------------------------------------- #


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def _rand(seed: int, salt: str) -> np.ndarray:
    """A 16 x 16 field of uniform floats from integer hashes (exactly reproducible)."""
    j, i = np.mgrid[0:TEXTURE_SIZE, 0:TEXTURE_SIZE]
    return nz.random2(nz.derive(seed, salt), i, j)


def _blobs(seed: int, salt: str, scale: float = 4.0) -> np.ndarray:
    """Tileable value-noise blobs in [0, 1] (period 16 pixels)."""
    j, i = np.mgrid[0:TEXTURE_SIZE, 0:TEXTURE_SIZE].astype(np.float64)
    n = TEXTURE_SIZE / scale
    s = nz.derive(seed, salt)
    x, y = i / scale, j / scale
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = nz.fade(x - ix), nz.fade(y - iy)
    period = int(n)

    def v(a, b):
        return nz.random2(s, a % period, b % period)

    a = v(ix, iy) + fx * (v(ix + 1, iy) - v(ix, iy))
    b = v(ix, iy + 1) + fx * (v(ix + 1, iy + 1) - v(ix, iy + 1))
    return a + fy * (b - a)


def _palette(field: np.ndarray, colors: list[str], cuts: list[float]) -> np.ndarray:
    """Pick ``colors[k]`` where ``field`` falls between cuts (``len(cuts) == len(colors) - 1``)."""
    idx = np.searchsorted(np.asarray(cuts), field)
    table = np.stack([_hex(c) for c in colors])
    return table[idx]


def _speckle(seed: int, name: str, colors: list[str], weights: list[float],
             blob: float = 0.45) -> np.ndarray:  # fmt: skip
    field = (1.0 - blob) * _rand(seed, name) + blob * _blobs(seed, name + "b")
    w = np.cumsum(weights) / np.sum(weights)
    lo, hi = field.min(), field.max()
    cuts = list(lo + (hi - lo) * w[:-1])
    return _palette(field, colors, cuts)


def _rgba(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    out = np.empty((TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out[..., 3] = 255 if alpha is None else alpha
    return out


def _cobble(seed: int, base: list[str], mortar: str) -> np.ndarray:
    pts = []
    r = _rand(seed, "cobble-pts")
    for k in range(9):
        pts.append((r[k, 0] * 16.0, r[k, 1] * 16.0, k))
    j, i = np.mgrid[0:16, 0:16].astype(np.float64) + 0.5
    d1 = np.full((16, 16), 1e9)
    d2 = np.full((16, 16), 1e9)
    cell = np.zeros((16, 16), dtype=np.int64)
    for px, py, k in pts:
        for ox in (-16.0, 0.0, 16.0):
            for oy in (-16.0, 0.0, 16.0):
                dx_, dy_ = i - (px + ox), j - (py + oy)
                d = np.sqrt(dx_ * dx_ + dy_ * dy_)
                closer = d < d1
                d2 = np.where(closer, d1, np.minimum(d2, d))
                cell = np.where(closer, k, cell)
                d1 = np.where(closer, d, d1)
    tone = _rand(seed, "cobble-tone")[0, :9]
    pal = np.stack([_hex(c) for c in base])
    rgb = pal[np.minimum((tone[cell] * len(base)).astype(int), len(base) - 1)]
    shade = 1.0 - 0.25 * np.clip((d1 - 1.0) / 5.0, 0.0, 1.0)
    rgb = rgb * shade[..., None]
    edge = (d2 - d1) < 1.1
    rgb[edge] = _hex(mortar)
    return rgb


def _bricks(seed: int, brick: list[str], mortar: str, rows: int = 4, cols: int = 2) -> np.ndarray:
    rgb = np.empty((16, 16, 3))
    h = 16 // rows
    w = 16 // cols
    tone = _rand(seed, "brick")
    pal = np.stack([_hex(c) for c in brick])
    for j in range(16):
        r = j // h
        shift = (w // 2) * (r % 2)
        for i in range(16):
            c = ((i + shift) // w) % cols
            k = int(tone[r, c] * len(brick)) % len(brick)
            rgb[j, i] = pal[k]
            if j % h == h - 1 or (i + shift) % w == w - 1:
                rgb[j, i] = _hex(mortar)
    rgb *= (0.92 + 0.12 * _rand(seed, "brick-grain"))[..., None]
    return rgb


def _planks(seed: int, colors: list[str], seam: str) -> np.ndarray:
    rgb = np.empty((16, 16, 3))
    grain = _rand(seed, "planks")
    pal = np.stack([_hex(c) for c in colors])
    for j in range(16):
        board = j // 4
        for i in range(16):
            k = int((grain[j, i] * 0.6 + 0.4 * grain[board, 7]) * len(colors)) % len(colors)
            rgb[j, i] = pal[k]
        if j % 4 == 3:
            rgb[j, :] = _hex(seam)
        cut = (board * 7 + 3) % 16
        rgb[j, cut] = _hex(seam) if j % 4 != 3 else rgb[j, cut]
    return rgb


def _log_side(seed: int, colors: list[str], dark: str) -> np.ndarray:
    r = _rand(seed, "bark")
    pal = np.stack([_hex(c) for c in colors])
    stripes = (r[0][None, :] * 0.7 + 0.3 * r) * len(colors)
    rgb = pal[np.minimum(stripes.astype(int), len(colors) - 1)]
    cracks = _rand(seed, "bark-cracks") < 0.12
    rgb[cracks] = _hex(dark)
    return rgb


def _log_top(seed: int, ring: str, core: str, bark: str) -> np.ndarray:
    j, i = np.mgrid[0:16, 0:16].astype(np.float64) + 0.5
    d = np.maximum(np.abs(i - 8.0), np.abs(j - 8.0))
    rgb = np.empty((16, 16, 3))
    rgb[:] = _hex(core)
    rgb[(d.astype(int) % 2) == 1] = _hex(ring)
    rgb[d > 7.0] = _hex(bark)
    rgb *= (0.94 + 0.1 * _rand(seed, "rings"))[..., None]
    return rgb


def _leaves(seed: int, base: list[str], hole: str) -> np.ndarray:
    """Clumps of leaves: light tufts on darker foliage, a few deep gaps (tinted later)."""
    clumps = _blobs(seed, "leaf-clumps", 4.0)
    grain = _rand(seed, "leaf-grain")
    field = 0.55 * clumps + 0.45 * grain
    rgb = _palette(field, base, [0.35, 0.5, 0.62])
    holes = (grain < 0.1) & (clumps < 0.45)
    rgb[holes] = _hex(hole)
    return rgb


def _plant(seed: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((16, 16, 3))
    alpha = np.zeros((16, 16), dtype=np.uint8)
    r = _rand(seed, "plant-" + kind)

    def stem(x: int, top: int, color: str, bend: float = 0.0) -> None:
        for j in range(top, 16):
            xi = int(round(x + bend * (15 - j) / 15.0))
            if 0 <= xi < 16:
                rgb[j, xi] = _hex(color) * (0.85 + 0.3 * r[j, xi])
                alpha[j, xi] = 255

    if kind in ("tall_grass", "fern"):
        light = "#c8c8c8" if kind == "tall_grass" else "#b8c0b8"
        for k, x in enumerate((2, 4, 6, 8, 10, 12, 14)):
            top = 2 + int(r[0, k] * 7)
            stem(x, top, light, bend=(r[1, k] - 0.5) * 4.0)
        if kind == "fern":
            for j in range(4, 16, 3):
                for x in range(3, 13):
                    if r[j, x] < 0.5:
                        rgb[j, x] = _hex("#a9b3a6")
                        alpha[j, x] = 255
    elif kind in ("poppy", "dandelion", "cornflower"):
        petal = {"poppy": "#d8322a", "dandelion": "#f2d22e", "cornflower": "#4a6fd8"}[kind]
        stem(8, 6, "#3f7a2a")
        stem(7, 11, "#4c8a32", bend=-2.0)
        for dj, di in ((3, 8), (4, 7), (4, 9), (5, 8), (4, 8), (3, 7), (5, 9)):
            rgb[dj, di] = _hex(petal) * (0.9 + 0.2 * r[dj, di])
            alpha[dj, di] = 255
        rgb[4, 8] = _hex("#f8e080" if kind != "dandelion" else "#e8a020")
    elif kind == "dead_bush":
        for k, x in enumerate((4, 7, 9, 12)):
            stem(x, 4 + int(r[2, k] * 6), "#7a5a32", bend=(r[3, k] - 0.5) * 8.0)
    else:  # mushrooms
        cap = "#c83a2a" if kind == "red_mushroom" else "#9a7552"
        stem(8, 9, "#e8dcc8")
        stem(7, 10, "#e8dcc8")
        for j in range(6, 10):
            for i in range(5, 12):
                if (i - 8) ** 2 / 9.0 + (j - 9) ** 2 / 9.0 <= 1.0 and j <= 9:
                    rgb[j, i] = _hex(cap) * (0.9 + 0.15 * r[j, i])
                    alpha[j, i] = 255
        if kind == "red_mushroom":
            for j, i in ((7, 6), (6, 9), (8, 10)):
                rgb[j, i] = _hex("#f0ece4")
    return rgb, alpha


def texture_rgba(name: str, seed: int = 1) -> np.ndarray:
    """The 16 x 16 RGBA pixels of a block texture (uint8)."""
    s = nz.derive(seed, "block-texture", name)
    gray_grass = ["#a8a8a8", "#9a9a9a", "#b6b6b6", "#8c8c8c"]
    if name == "stone":
        return _rgba(_speckle(s, name, ["#6f6f6f", "#7d7d7d", "#888888", "#959595"], [2, 4, 3, 1]))
    if name == "dark_stone":
        return _rgba(_speckle(s, name, ["#3e3e44", "#4a4a50", "#55555c", "#61616a"], [2, 4, 3, 1]))
    if name == "cobblestone":
        return _rgba(_cobble(s, ["#7a7a7a", "#8a8a8a", "#6c6c6c", "#9a9a9a"], "#4a4a4a"))
    if name == "bedrock":
        return _rgba(_speckle(s, name, ["#1e1e1e", "#3a3a3a", "#585858", "#777777"], [2, 3, 2, 1],
                              blob=0.2))  # fmt: skip
    if name == "dirt":
        return _rgba(_speckle(s, name, ["#5e4128", "#6e4c30", "#7d5a3a", "#8c6844"], [2, 4, 3, 1]))
    if name in ("grass_top", "dry_grass_top"):
        rgb = _speckle(s, name, gray_grass, [3, 3, 2, 1], blob=0.5)
        return _rgba(rgb)
    if name in ("grass_side", "snowy_grass_side", "dry_grass_side"):
        rgb = _speckle(s, "dirt", ["#5e4128", "#6e4c30", "#7d5a3a", "#8c6844"], [2, 4, 3, 1])
        top = {
            "grass_side": ["#4f8a34", "#5f9a3e", "#447a2c"],
            "dry_grass_side": ["#9a9a48", "#aaa654", "#8a8a40"],
            "snowy_grass_side": ["#f2f6f8", "#e2e8ec", "#d4dce2"],
        }[name]
        fringe = _rand(s, "fringe")
        depth = 3 + (fringe[0] * 2.2).astype(int)
        pal = np.stack([_hex(c) for c in top])
        for i in range(16):
            for j in range(depth[i]):
                rgb[j, i] = pal[int(fringe[j + 1, i] * 3) % 3]
        return _rgba(rgb)
    if name == "sand":
        return _rgba(_speckle(s, name, ["#d6c48e", "#dfcd98", "#e8d8a6", "#cbb782"], [3, 4, 2, 1]))
    if name == "red_sand":
        return _rgba(_speckle(s, name, ["#b8662e", "#c4733a", "#d08046", "#a95c28"], [3, 4, 2, 1]))
    if name in ("sandstone", "sandstone_top"):
        rgb = _speckle(s, name, ["#d4c08a", "#dccb96", "#c9b47e"], [3, 4, 2])
        if name == "sandstone":
            for j in (3, 9, 13):
                rgb[j, :] *= 0.86
        return _rgba(rgb)
    if name == "gravel":
        return _rgba(_speckle(s, name, ["#6e6866", "#837d7a", "#958f8b", "#5a5654", "#9c8c80"],
                              [2, 3, 2, 1, 1], blob=0.15))  # fmt: skip
    if name == "clay":
        return _rgba(_speckle(s, name, ["#9aa1ae", "#a4abb8", "#8f96a3"], [3, 3, 1]))
    if name == "mud":
        return _rgba(_speckle(s, name, ["#3c3128", "#463a2f", "#52443a", "#342a22"], [2, 3, 2, 1]))
    if name == "snow":
        return _rgba(_speckle(s, name, ["#f4f8fa", "#ebf1f5", "#dfe8ee", "#fbfdfe"], [4, 3, 1, 2]))
    if name in ("ice", "packed_ice"):
        base = ["#9cc2f0", "#a9cdf6", "#8db6e8"] if name == "ice" else ["#8fb0e0", "#9cbbea",
                                                                       "#84a6d6"]  # fmt: skip
        rgb = _speckle(s, name, base, [3, 3, 2])
        for k in range(3):
            j0 = int(_rand(s, "streak")[k, 0] * 16)
            for i in range(16):
                rgb[(j0 + i // 2) % 16, i] = _hex("#d8ecff")
        return _rgba(rgb)
    if name == "water":
        rgb = _speckle(s, name, ["#c9c9c9", "#d6d6d6", "#bcbcbc", "#e4e4e4"], [3, 3, 2, 1], 0.7)
        return _rgba(rgb)
    if name == "lava":
        return _rgba(_speckle(s, name, ["#d84c10", "#ee7a18", "#f8a830", "#ffd25a"], [3, 3, 2, 1],
                              blob=0.75))  # fmt: skip
    if name in ("coal_ore", "iron_ore", "gold_ore"):
        rgb = _speckle(nz.derive(seed, "block-texture", "stone"), "stone",
                       ["#6f6f6f", "#7d7d7d", "#888888", "#959595"], [2, 4, 3, 1])  # fmt: skip
        ore = {"coal_ore": "#202020", "iron_ore": "#d8a884", "gold_ore": "#f8d040"}[name]
        spots = _rand(s, "ore") < 0.16
        rgb[spots] = _hex(ore)
        return _rgba(rgb)
    if name.startswith("terracotta"):
        base = {
            "terracotta": "#9a5a3e",
            "terracotta_red": "#8c3a2c",
            "terracotta_yellow": "#b8843a",
            "terracotta_white": "#c8aa98",
            "terracotta_brown": "#5a3c2a",
        }[name]
        b = _hex(base)
        rgb = b * (0.94 + 0.1 * _rand(s, name))[..., None]
        return _rgba(rgb)
    logs = {
        "oak_log": (["#6b5232", "#5c4529", "#7a5f3b"], "#3e2e1a", "#b8935c", "#a07c4a"),
        "birch_log": (["#e4e0d4", "#d8d4c6", "#ecebe2"], "#2e2a26", "#d8c890", "#c4b07a"),
        "spruce_log": (["#3e2c1c", "#4a3522", "#33251a"], "#22180e", "#7a5a38", "#664a2c"),
        "jungle_log": (["#5a4a24", "#66552c", "#4f4020"], "#2c2412", "#b48a54", "#9a7242"),
        "acacia_log": (["#6e6862", "#7a746c", "#625c56"], "#3c3834", "#c0703a", "#a85e30"),
    }
    if name in logs:
        side, dark, _, _ = logs[name]
        rgb = _log_side(s, side, dark)
        if name == "birch_log":
            marks = _rand(s, "birch-marks")
            for j in range(16):
                if marks[j, 0] < 0.35:
                    x0 = int(marks[j, 1] * 12)
                    rgb[j, x0 : x0 + 2 + int(marks[j, 2] * 3)] = _hex("#2e2a26")
        return _rgba(rgb)
    if name.endswith("_log_top"):
        side, _, ring, core = logs[name[: -len("_top")]]
        return _rgba(_log_top(s, ring, core, side[1]))
    if name.endswith("_leaves"):
        return _rgba(_leaves(s, ["#6e6e6e", "#8a8a8a", "#a2a2a2", "#bdbdbd"], "#3c3c3c"))
    if name == "cactus_side":
        rgb = _speckle(s, name, ["#4a7a2c", "#558a34", "#3f6a26"], [3, 3, 2])
        for i in (1, 5, 10, 14):
            rgb[:, i] *= 0.75
        rgb[::3, 3] = _hex("#e8e0c0")
        return _rgba(rgb)
    if name == "cactus_top":
        return _rgba(_speckle(s, name, ["#5a8a34", "#669a3c", "#4f7a2c"], [3, 3, 2]))
    if name == "oak_planks":
        return _rgba(_planks(s, ["#a5824e", "#b18c56", "#9a7846", "#bb985e"], "#6e5430"))
    if name == "spruce_planks":
        return _rgba(_planks(s, ["#6e5034", "#7a5a3a", "#634830", "#84623f"], "#43301c"))
    if name in ("stone_bricks", "mossy_stone_bricks"):
        rgb = _bricks(s, ["#7c7c7c", "#868686", "#727272"], "#5a5a5a", rows=4, cols=2)
        if name == "mossy_stone_bricks":
            moss = _blobs(s, "moss", 4.0) > 0.6
            rgb[moss] = _hex("#4f7a34") * (0.85 + 0.3 * _rand(s, "moss-tone")[moss])[:, None]
        return _rgba(rgb)
    if name == "bricks":
        return _rgba(_bricks(s, ["#9a4a3a", "#a8543e", "#8c4234"], "#b8ab9c", rows=4, cols=2))
    if name == "glass":
        rgb = np.full((16, 16, 3), _hex("#cfe4ec"))
        rgb[0, :] = rgb[-1, :] = rgb[:, 0] = rgb[:, -1] = _hex("#e8f4f8")
        for k in range(4):
            rgb[3 + k, 4 + k] = rgb[4 + k, 4 + k] = _hex("#f4fbfd")
        return _rgba(rgb)
    if name in ("white_wool", "red_wool"):
        base = ["#e8e8e4", "#dcdcd6", "#f2f2ee"] if name == "white_wool" else [
            "#a8302a", "#982a24", "#b83a32"]  # fmt: skip
        return _rgba(_speckle(s, name, base, [3, 3, 2], blob=0.3))
    if name == "hay_side":
        rgb = _speckle(s, name, ["#c8a43a", "#d4b044", "#b89434"], [3, 3, 2])
        rgb[3, :] = rgb[12, :] = _hex("#8a6a24")
        return _rgba(rgb)
    if name == "hay_top":
        return _rgba(_speckle(s, name, ["#c8a43a", "#d4b044", "#a8842c"], [3, 3, 2]))
    if name == "dirt_path_top":
        return _rgba(_speckle(s, name, ["#94784a", "#a08450", "#86683e", "#ab9058"], [3, 3, 2, 1]))
    if name == "glowstone":
        return _rgba(_speckle(s, name, ["#f8d878", "#e8b850", "#ffeaa0", "#c89838"], [3, 3, 2, 1]))
    if name == "lantern_block":
        rgb = np.full((16, 16, 3), _hex("#ffe08a"))
        rgb[0, :] = rgb[-1, :] = rgb[:, 0] = rgb[:, -1] = _hex("#3a3a3a")
        rgb[7:9, :] = _hex("#3a3a3a")
        return _rgba(rgb)
    if name in ("tall_grass", "fern", "poppy", "dandelion", "cornflower", "dead_bush",
                "red_mushroom", "brown_mushroom"):  # fmt: skip
        rgb, alpha = _plant(s, name)
        return _rgba(rgb, alpha)
    raise KeyError(f"no block texture {name!r}")


def texture_png(name: str, seed: int = 1) -> bytes:
    from ..rendering.png import encode_png

    return encode_png(texture_rgba(name, seed))


def tinted(name: str) -> bool:
    """Whether a texture is painted in grays to be tinted by the biome (grass, leaves, water)."""
    return name in ("grass_top", "dry_grass_top", "water", "tall_grass", "fern") or name.endswith(
        "_leaves"
    )


def has_alpha(name: str) -> bool:
    return name in ("tall_grass", "fern", "poppy", "dandelion", "cornflower", "dead_bush",
                    "red_mushroom", "brown_mushroom")  # fmt: skip
