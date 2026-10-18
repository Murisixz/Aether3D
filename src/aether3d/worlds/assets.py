"""Procedural plant and rock meshes: every species in variants and levels of detail.

Each asset is built from a few primitives -- tapered cylinders (trunks, branches, stems),
cones (conifer tiers), noise-displaced icospheres (crowns, bushes, rocks), flat strips
(fronds, blades, petals) -- with sizes, counts and colours drawn from integer hashes of
``(seed, species, variant)``, so a variant is the same mesh everywhere and every time.
Meshes stand on the origin (z up) and carry two face groups -- 0 for wood, stems and stone,
1 for foliage, blades and petals -- mapped to world material layers (:data:`LAYERS`), and
vertex colours (the plant's own hue variation; the layer materials are white by default).

Three levels of detail per variant: 0 (full; a broadleaf tree is about 1,000 triangles),
1 (a few hundred) and 2 (a silhouette of tens of triangles for distant forests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..mesh import Mesh
from . import noise as nz

LAYERS: dict[str, tuple[str, str]] = {
    # species: (layer of group 0, layer of group 1)
    "oak": ("bark", "leaves"),
    "beech": ("bark", "leaves"),
    "maple": ("bark", "leaves"),
    "birch": ("birch_bark", "leaves"),
    "pine": ("bark", "needles"),
    "spruce": ("bark", "needles"),
    "fir": ("bark", "needles"),
    "larch": ("bark", "needles"),
    "palm": ("bark", "palm"),
    "acacia": ("bark", "leaves"),
    "baobab": ("bark", "leaves"),
    "jungle": ("bark", "leaves"),
    "willow": ("bark", "leaves"),
    "dead_tree": ("wood", "wood"),
    "cactus": ("cactus", "cactus"),
    "bush": ("bark", "leaves"),
    "shrub": ("bark", "leaves"),
    "fern": ("grass", "grass"),
    "grass": ("grass", "grass"),
    "tall_grass": ("grass", "grass"),
    "flowers": ("grass", "flowers"),
    "reeds": ("grass", "grass"),
    "mushroom": ("mushroom", "mushroom"),
    "dead_bush": ("wood", "wood"),
    "heather": ("grass", "flowers"),
    "rock": ("rock", "rock"),
    "boulder": ("rock", "rock"),
    "pebbles": ("rock", "rock"),
    "log": ("wood", "wood"),
}
TREES = frozenset(
    (
        "oak",
        "beech",
        "maple",
        "birch",
        "pine",
        "spruce",
        "fir",
        "larch",
        "palm",
        "acacia",
        "baobab",
        "jungle",
        "willow",
        "dead_tree",
        "cactus",
    )
)
ROCKS = frozenset(("rock", "boulder", "pebbles", "log"))
GROUND_COVER = frozenset(
    ("fern", "grass", "tall_grass", "flowers", "reeds", "mushroom", "dead_bush", "heather")
)
LODS = 3


@dataclass(frozen=True)
class Asset:
    mesh: Mesh
    height: float
    radius: float  # footprint radius (trunk spacing)
    layers: tuple[str, str]


def _rgb(hexs: str) -> np.ndarray:
    return np.array([int(hexs[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]) ** 2.2


class _Builder:
    def __init__(self) -> None:
        self.p: list[np.ndarray] = []
        self.n: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.t: list[np.ndarray] = []
        self.g: list[np.ndarray] = []
        self.count = 0

    def add(self, p, n, c, t, group: int) -> None:
        p = np.asarray(p, dtype=np.float64)
        self.p.append(p)
        self.n.append(np.asarray(n, dtype=np.float64))
        cc = np.asarray(c, dtype=np.float64)
        if cc.ndim == 1:
            cc = np.broadcast_to(cc, p.shape).copy()
        self.c.append(cc)
        t = np.asarray(t, dtype=np.int64) + self.count
        self.t.append(t)
        self.g.append(np.full(t.shape[0], group, dtype=np.int64))
        self.count += p.shape[0]

    def mesh(self) -> Mesh:
        p = np.concatenate(self.p)
        n = np.concatenate(self.n)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        c = np.clip(np.concatenate(self.c), 0.0, 1.0)
        return Mesh(
            p.astype(np.float32),
            np.concatenate(self.t),
            normals=n.astype(np.float32),
            groups=np.concatenate(self.g),
            colors=c.astype(np.float32),
        )


class _Rand:
    """Deterministic uniforms for one asset."""

    def __init__(self, seed: int, *parts: object) -> None:
        self.seed = nz.derive(seed, "asset", *parts)
        self.k = 0

    def u(self, lo: float = 0.0, hi: float = 1.0) -> float:
        self.k += 1
        v = float(nz.random2(self.seed, np.array([self.k]), np.array([0]))[0])
        return lo + (hi - lo) * v


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def tube(b: _Builder, p0, p1, r0, r1, sides, color, group, cap=True, color_top=None) -> None:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    u, v = _frame(axis)
    ang = np.arange(sides) * (2.0 * math.pi / sides)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    bottom = p0 + r0 * ring
    top = p1 + r1 * ring
    slope = (r0 - r1) / max(np.linalg.norm(axis), 1e-9)
    nrm = ring + slope * axis / max(np.linalg.norm(axis), 1e-9)
    pos = np.concatenate([bottom, top])
    nor = np.concatenate([nrm, nrm])
    c0 = np.asarray(color, dtype=np.float64)
    c1 = c0 if color_top is None else np.asarray(color_top, dtype=np.float64)
    col = np.concatenate([np.broadcast_to(c0, bottom.shape), np.broadcast_to(c1, top.shape)])
    k = np.arange(sides)
    k1 = (k + 1) % sides
    tris = np.concatenate(
        [np.stack([k, k1, sides + k1], 1), np.stack([k, sides + k1, sides + k], 1)]
    )
    b.add(pos, nor, col, tris, group)
    if cap and r1 > 1e-6:
        centre = len(pos)
        cp = np.vstack([top, p1[None]])
        cn = np.broadcast_to(axis / np.linalg.norm(axis), cp.shape)
        ct = np.stack([np.full(sides, sides), k, k1], 1)
        b.add(cp, cn, c1, ct, group)
        _ = centre


def cone(b: _Builder, base, height, radius, sides, color, group, color_top=None) -> None:
    base = np.asarray(base, dtype=np.float64)
    ang = np.arange(sides) * (2.0 * math.pi / sides)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(sides)], 1)
    rim = base + radius * ring
    apex = base + np.array([0.0, 0.0, height])
    slope = radius / max(height, 1e-9)
    nrm = ring + np.array([0.0, 0.0, slope])
    pos = np.vstack([rim, apex[None], base[None]])
    nor = np.vstack([nrm, [[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]])
    c0 = np.asarray(color, dtype=np.float64)
    c1 = c0 if color_top is None else np.asarray(color_top, dtype=np.float64)
    col = np.vstack([np.broadcast_to(c0, rim.shape), c1[None], c0[None] * 0.6])
    k = np.arange(sides)
    k1 = (k + 1) % sides
    tris = np.concatenate(
        [
            np.stack([k, k1, np.full(sides, sides)], 1),
            np.stack([k1, k, np.full(sides, sides + 1)], 1),
        ]
    )
    b.add(pos, nor, col, tris, group)


@lru_cache(maxsize=4)
def _icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    f = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    verts = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, c: int, cache: dict = cache) -> int:
            key = (min(a, c), max(a, c))
            if key not in cache:
                m = verts[a] + verts[c]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        nf = []
        for a, b_, c in faces:
            ab, bc, ca = mid(a, b_), mid(b_, c), mid(c, a)
            nf += [[a, ab, ca], [b_, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = nf
    return np.array(verts), np.array(faces, dtype=np.int64)


def blob(b: _Builder, centre, radii, level, rough, seed, color, group, dark=0.65) -> None:
    """A noise-displaced ellipsoid; the underside is shaded darker (self-shadowed foliage)."""
    v, f = _icosphere(level)
    centre = np.asarray(centre, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    disp = 1.0 + rough * nz.gradient3(seed, v[:, 0] * 1.7, v[:, 1] * 1.7, v[:, 2] * 1.7)
    p = centre + v * radii * disp[:, None]
    nrm = v / radii
    shade = dark + (1.0 - dark) * (0.5 + 0.5 * v[:, 2])
    col = np.asarray(color, dtype=np.float64)[None, :] * shade[:, None]
    b.add(p, nrm, col, f, group)


def blades(
    b: _Builder, count, height, width, spread, lean, rnd: _Rand, color, group, tip=None
) -> None:
    """Thin tapered blades (double-sided strips) fanning out from the origin."""
    pos, nor, col, tris = [], [], [], []
    c0 = np.asarray(color, dtype=np.float64)
    c1 = c0 * 1.25 if tip is None else np.asarray(tip, dtype=np.float64)
    for _ in range(count):
        a = rnd.u(0.0, 2.0 * math.pi)
        r = rnd.u(0.0, spread)
        h = height * rnd.u(0.7, 1.15)
        lean_k = lean * rnd.u(0.3, 1.0)
        base = np.array([r * math.cos(a), r * math.sin(a), 0.0])
        d = np.array([math.cos(a), math.sin(a), 0.0])
        side = np.array([-math.sin(a), math.cos(a), 0.0]) * width
        mid = base + d * lean_k * 0.4 + np.array([0.0, 0.0, h * 0.55])
        top = base + d * lean_k + np.array([0.0, 0.0, h])
        n = np.cross(side, top - base)
        n = n / max(np.linalg.norm(n), 1e-9)
        i = len(pos)
        pos += [base - side, base + side, mid - side * 0.6, mid + side * 0.6, top]
        nor += [n] * 5
        col += [c0 * 0.7, c0 * 0.7, c0, c0, c1]
        tris += [[i, i + 1, i + 3], [i, i + 3, i + 2], [i + 2, i + 3, i + 4]]
    b.add(np.array(pos), np.array(nor), np.array(col), np.array(tris), group)


def _tint(base: np.ndarray, rnd: _Rand, amount: float) -> np.ndarray:
    return np.clip(base * np.array([rnd.u(1 - amount, 1 + amount) for _ in range(3)]), 0, 1)


# --------------------------------------------------------------------------------------- #
# Species
# --------------------------------------------------------------------------------------- #

BARK = _rgb("#5a4632")
BARK_GREY = _rgb("#6e6a62")
BIRCH = _rgb("#d2cec2")
LEAF = _rgb("#4c7a2c")
NEEDLE = _rgb("#35593a")


def _conifer(b, rnd, lod, species):
    h = rnd.u(13.0, 24.0) if species != "pine" else rnd.u(14.0, 22.0)
    trunk_r = h * rnd.u(0.012, 0.018)
    sides = (8, 6, 4)[lod]
    tube(b, (0, 0, 0), (0, 0, h * 0.97), trunk_r, trunk_r * 0.25, sides, BARK, 0, cap=False)
    green = _tint(NEEDLE if species != "larch" else _rgb("#4a6e30"), rnd, 0.14)
    if species == "pine":
        crown0 = h * rnd.u(0.55, 0.68)
        n = (6, 3, 1)[lod]
        for k in range(n):
            z = crown0 + (h - crown0) * (k + 0.5) / n
            r = h * 0.11 * rnd.u(0.8, 1.2)
            if lod == 2:
                blob(
                    b,
                    (0, 0, (crown0 + h) / 2),
                    (r * 1.6, r * 1.6, (h - crown0) / 2),
                    0,
                    0.15,
                    rnd.seed + k,
                    green,
                    1,
                )
                break
            off = rnd.u(0, r * 0.8)
            a = rnd.u(0, 2 * math.pi)
            blob(
                b,
                (off * math.cos(a), off * math.sin(a), z),
                (r * 1.5, r * 1.5, r * 0.7),
                1 if lod == 0 else 0,
                0.25,
                rnd.seed + k,
                green,
                1,
            )
        return h, trunk_r * 2.5
    tiers = (9, 5, 1)[lod]
    base_r = h * rnd.u(0.16, 0.22) * (1.2 if species == "fir" else 1.0)
    z0 = h * rnd.u(0.12, 0.22)
    if lod == 2:
        cone(b, (0, 0, z0), h - z0, base_r, 6, green, 1, color_top=green * 1.15)
        return h, trunk_r * 2.5
    for k in range(tiers):
        f = k / tiers
        z = z0 + (h - z0) * f * 0.92
        r = base_r * (1.0 - f) ** 0.9 + 0.25
        th = (h - z0) / tiers * 1.9
        cone(
            b,
            (0, 0, z),
            th,
            r,
            (10, 7)[lod],
            green * (0.85 + 0.25 * f),
            1,
            color_top=green * (1.0 + 0.2 * f),
        )
    return h, trunk_r * 2.5


BROADLEAF = {
    # species: (height range, trunk radius / height, crown base / height, crown radius /
    # height, crown flattening, leaf colour)
    "oak": ((10.0, 17.0), 0.032, 0.3, 0.36, 0.8, "#3e5c24"),
    "beech": ((14.0, 24.0), 0.026, 0.34, 0.3, 0.9, "#44622a"),
    "maple": ((11.0, 18.0), 0.028, 0.32, 0.33, 0.9, "#4a6a28"),
    "birch": ((12.0, 19.0), 0.018, 0.42, 0.22, 1.1, "#55762f"),
    "jungle": ((22.0, 34.0), 0.022, 0.6, 0.27, 0.6, "#2c5a20"),
    "acacia": ((5.0, 9.0), 0.03, 0.62, 0.55, 0.35, "#5a6a2a"),
    "baobab": ((8.0, 13.0), 0.12, 0.72, 0.36, 0.55, "#4a6326"),
    "willow": ((8.0, 13.0), 0.034, 0.32, 0.38, 0.85, "#5a7a34"),
}


def _broadleaf(b, rnd, lod, species):
    """A trunk, a few limbs and a crown of overlapping leaf clumps inside an ellipsoid
    envelope; inner clumps are shaded darker (light hardly reaches them)."""
    heights, trunk_k, base_k, crown_k, flat, leaf_hex = BROADLEAF[species]
    h = rnd.u(*heights)
    trunk_r = h * trunk_k * rnd.u(0.85, 1.15)
    bark = BIRCH if species == "birch" else (BARK_GREY if species in ("beech", "acacia") else BARK)
    sides = (8, 6, 4)[lod]
    cr = h * crown_k * rnd.u(0.9, 1.1)  # crown radius
    cz = h * base_k + cr * flat  # crown centre height
    top = min(cz, h - cr * flat * 0.6)
    tube(b, (0, 0, 0), (0, 0, top), trunk_r, trunk_r * 0.55, sides, bark, 0, cap=False)
    if species == "birch" and lod == 0:  # dark bark marks
        for _ in range(6):
            z = rnd.u(0.5, h * base_k)
            tube(
                b,
                (0, 0, z),
                (0, 0, z + 0.12),
                trunk_r * 1.04,
                trunk_r * 1.02,
                6,
                _rgb("#2e2a26"),
                0,
                cap=False,
            )
    leaf = _tint(_rgb(leaf_hex), rnd, 0.12)
    if lod == 2:
        blob(b, (0, 0, cz), (cr, cr, cr * flat), 0, 0.14, rnd.seed, leaf * 0.9, 1, 0.55)
        return h, trunk_r * 2.0
    # Limbs from the crown base toward the envelope.
    limbs = (5, 3)[lod] if species != "baobab" else (6, 4)[lod]
    z_base = h * base_k
    for k in range(limbs):
        a = 2 * math.pi * (k + rnd.u(0.0, 0.7)) / limbs
        z = z_base + rnd.u(0.0, 0.25) * (cz - z_base)
        reach = cr * rnd.u(0.55, 0.8)
        tip = np.array([reach * math.cos(a), reach * math.sin(a), z + reach * flat * 0.8])
        tube(
            b,
            (0, 0, z - 0.2),
            tip,
            trunk_r * 0.5,
            trunk_r * 0.15,
            max(sides - 3, 3),
            bark,
            0,
            cap=False,
        )
    # Clumps: random points in the envelope, biased outward; radius shrinks outward.
    clumps = (14, 6)[lod]
    for k in range(clumps):
        a = rnd.u(0.0, 2 * math.pi)
        zt = rnd.u(-0.75, 1.0)
        rr = math.sqrt(max(0.0, 1.0 - zt * zt)) * rnd.u(0.35, 0.95) ** 0.5
        d = np.array([rr * math.cos(a), rr * math.sin(a), zt])
        out = float(np.linalg.norm(d))
        r = cr * rnd.u(0.38, 0.52) * (1.0 if lod == 0 else 1.25)
        centre = np.array([d[0] * cr * 0.62, d[1] * cr * 0.62, cz + d[2] * cr * flat * 0.62])
        shade = 0.72 + 0.28 * min(out, 1.0)
        blob(
            b,
            centre,
            (r, r, r * max(flat, 0.6)),
            1 if lod == 0 else 0,
            0.24,
            rnd.seed + 17 * k,
            leaf * shade * rnd.u(0.9, 1.08),
            1,
            0.55,
        )
    if species == "willow" and lod == 0:
        for _ in range(16):
            a = rnd.u(0, 2 * math.pi)
            r0 = cr * rnd.u(0.6, 1.0)
            anchor = np.array([r0 * math.cos(a), r0 * math.sin(a), cz])
            blades(b, 1, rnd.u(3.0, 5.5), 0.35, 0.0, 0.0, rnd, leaf, 1)
            b.p[-1] = np.array([anchor + p * np.array([1, 1, -1]) for p in b.p[-1]])
    return h, trunk_r * 2.0


def _palm(b, rnd, lod):
    h = rnd.u(9.0, 17.0)
    lean = rnd.u(0.5, 2.5)
    segs = (8, 4, 2)[lod]
    pts = [np.array([lean * (k / segs) ** 2, 0.0, h * k / segs]) for k in range(segs + 1)]
    for k in range(segs):
        tube(
            b,
            pts[k],
            pts[k + 1],
            0.22 - 0.06 * k / segs,
            0.22 - 0.06 * (k + 1) / segs,
            (7, 5, 4)[lod],
            _rgb("#7a6650"),
            0,
            cap=False,
        )
    top = pts[-1]
    green = _tint(_rgb("#3f7a2a"), rnd, 0.15)
    fronds = (11, 7, 4)[lod]
    for k in range(fronds):
        a = 2 * math.pi * k / fronds + rnd.u(-0.2, 0.2)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        side = np.array([-math.sin(a), math.cos(a), 0.0])
        length = rnd.u(3.0, 4.8)
        n = (5, 3, 2)[lod]
        pos, nor, col, tris = [], [], [], []
        for j in range(n + 1):
            t = j / n
            p = top + d * length * t + np.array([0, 0, 0.8 * t - 1.6 * t * t])
            w = 0.55 * math.sin(math.pi * max(t, 0.08))
            pos += [p - side * w, p + side * w]
            nor += [np.array([0, 0, 1.0])] * 2
            col += [green * (0.8 + 0.3 * t)] * 2
        for j in range(n):
            i = 2 * j
            tris += [[i, i + 2, i + 3], [i, i + 3, i + 1]]
        b.add(np.array(pos), np.array(nor), np.array(col), np.array(tris), 1)
    return h, 0.5


def _cactus(b, rnd, lod):
    h = rnd.u(2.5, 7.0)
    r = h * 0.07
    green = _tint(_rgb("#4f7a32"), rnd, 0.1)
    sides = (10, 7, 5)[lod]
    tube(b, (0, 0, 0), (0, 0, h), r, r * 0.9, sides, green, 0)
    if lod < 2:
        for _ in range(int(rnd.u(1.0, 3.99))):
            a = rnd.u(0, 2 * math.pi)
            z = h * rnd.u(0.35, 0.65)
            out = np.array([math.cos(a), math.sin(a), 0.0]) * r * 2.6
            tube(
                b,
                (0, 0, z),
                out + np.array([0, 0, z + r]),
                r * 0.6,
                r * 0.6,
                sides,
                green,
                0,
                cap=False,
            )
            tube(
                b,
                out + np.array([0, 0, z + r]),
                out + np.array([0, 0, z + h * 0.35]),
                r * 0.6,
                r * 0.55,
                sides,
                green,
                0,
            )
    return h, r * 3


def _small(b, rnd, lod, species):
    if species in ("bush", "shrub", "heather"):
        s = rnd.u(0.6, 1.8) if species != "heather" else rnd.u(0.2, 0.45)
        color = _tint(LEAF if species != "heather" else _rgb("#7a4a7c"), rnd, 0.15)
        n = (3, 2, 1)[lod]
        for k in range(n):
            off = np.array([rnd.u(-0.3, 0.3) * s, rnd.u(-0.3, 0.3) * s, s * 0.45])
            blob(
                b,
                off,
                (s * 0.6, s * 0.6, s * 0.5),
                1 if lod == 0 else 0,
                0.25,
                rnd.seed + k,
                color,
                1,
            )
        if species != "heather" and lod == 0:
            tube(b, (0, 0, 0), (0, 0, s * 0.4), 0.04, 0.03, 4, BARK, 0, cap=False)
        return s, s * 0.6
    if species == "mushroom":
        s = rnd.u(0.08, 0.3)
        cap = _rgb("#b8402a") if rnd.u() < 0.5 else _rgb("#9a7552")
        tube(b, (0, 0, 0), (0, 0, s * 0.8), s * 0.12, s * 0.1, 6, _rgb("#e8dcc8"), 0, cap=False)
        blob(b, (0, 0, s * 0.8), (s * 0.45, s * 0.45, s * 0.22), 0, 0.05, rnd.seed, cap, 1, 0.8)
        return s, s * 0.5
    if species in ("rock", "boulder", "pebbles"):
        s = {"rock": (0.3, 1.2), "boulder": (1.4, 3.6), "pebbles": (0.05, 0.18)}[species]
        s = rnd.u(*s)
        grey = _tint(_rgb("#8a857c"), rnd, 0.12)
        n = 4 if species == "pebbles" else 1
        for k in range(n):
            off = np.array([rnd.u(-1, 1), rnd.u(-1, 1), 0.0]) * (s * 2.5 if n > 1 else 0.0)
            blob(
                b,
                off + np.array([0, 0, s * 0.3]),
                (s, s * rnd.u(0.7, 1.0), s * rnd.u(0.5, 0.8)),
                (2, 1, 0)[lod] if species == "boulder" else (1, 0, 0)[lod],
                0.28,
                rnd.seed + k,
                grey,
                0,
                0.75,
            )
        return s, s
    if species == "log":
        length = rnd.u(3.0, 7.0)
        r = rnd.u(0.18, 0.38)
        tube(
            b,
            (-length / 2, 0, r * 0.8),
            (length / 2, 0, r * 0.75),
            r,
            r * 0.85,
            (8, 6, 4)[lod],
            _rgb("#5a4a38"),
            0,
        )
        return r * 2, length / 2
    if species in ("grass", "tall_grass", "reeds", "fern", "dead_bush", "flowers"):
        h = {
            "grass": (0.25, 0.55),
            "tall_grass": (0.8, 1.4),
            "reeds": (1.2, 2.2),
            "fern": (0.4, 0.9),
            "dead_bush": (0.4, 0.9),
            "flowers": (0.2, 0.45),
        }[species]
        h = rnd.u(*h)
        color = {
            "grass": _rgb("#5d8a34"),
            "tall_grass": _rgb("#8a9a48"),
            "reeds": _rgb("#6f8a3e"),
            "fern": _rgb("#3f7a2e"),
            "dead_bush": _rgb("#7a5a32"),
            "flowers": _rgb("#4f7a2e"),
        }[species]
        color = _tint(color, rnd, 0.12)
        count = {"grass": 9, "tall_grass": 8, "reeds": 7, "fern": 7, "dead_bush": 6, "flowers": 3}[
            species
        ]
        count = max(2, count // (1 + lod))
        width = {"fern": 0.07, "dead_bush": 0.015}.get(species, 0.025)
        lean = {"fern": 0.6, "tall_grass": 0.3}.get(species, 0.12) * h
        blades(b, count, h, width, 0.12 * h, lean, rnd, color, 0)
        if species == "flowers":
            petal = [
                _rgb("#d8322a"),
                _rgb("#f2d22e"),
                _rgb("#eeeef2"),
                _rgb("#8a4ad8"),
                _rgb("#4a6fd8"),
            ][int(rnd.u(0, 4.999))]
            for k in range(count):
                a = rnd.u(0, 2 * math.pi)
                p = np.array([math.cos(a) * 0.05, math.sin(a) * 0.05, h * rnd.u(0.85, 1.1)])
                blob(b, p, (0.04, 0.04, 0.025), 0, 0.0, rnd.seed + k, petal, 1, 0.9)
        return h, 0.15
    raise KeyError(species)


@lru_cache(maxsize=2048)
def asset(species: str, variant: int, lod: int, seed: int = 0) -> Asset:
    """The mesh of a species variant at a level of detail (cached)."""
    rnd = _Rand(seed, species, variant)
    b = _Builder()
    if species in ("spruce", "fir", "pine", "larch"):
        h, r = _conifer(b, rnd, lod, species)
    elif species in ("oak", "beech", "maple", "birch", "jungle", "acacia", "baobab", "willow"):
        h, r = _broadleaf(b, rnd, lod, species)
    elif species == "palm":
        h, r = _palm(b, rnd, lod)
    elif species == "cactus":
        h, r = _cactus(b, rnd, lod)
    elif species == "dead_tree":
        h = rnd.u(6.0, 13.0)
        tr = h * 0.022
        tube(b, (0, 0, 0), (0, 0, h), tr, tr * 0.3, (7, 5, 4)[lod], _rgb("#6a5a48"), 0, cap=False)
        for _ in range((6, 3, 1)[lod]):
            a = rnd.u(0, 2 * math.pi)
            z = h * rnd.u(0.35, 0.85)
            length = h * rnd.u(0.15, 0.35)
            tip = np.array([math.cos(a) * length, math.sin(a) * length, z + length * 0.6])
            tube(b, (0, 0, z), tip, tr * 0.4, tr * 0.08, 4, _rgb("#6a5a48"), 0, cap=False)
        r = tr * 3
    else:
        h, r = _small(b, rnd, lod, species)
    return Asset(b.mesh(), float(h), float(r), LAYERS[species])
