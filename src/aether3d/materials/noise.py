"""Hashes and noise shared by the CPU evaluator and the GPU shader (bit-identical hashes).

Randomness comes from integer hashing of lattice cells, never from a random
number generator, so procedural textures are reproducible everywhere: the
same cell and seed always give the same 32-bit hash on the CPU (NumPy
``uint32``, wrapping arithmetic) and on the GPU (WGSL ``u32``). The hash is
``pcg3d`` (Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT
2020). Random floats use the top 24 bits of a hash, which float32 represents
exactly.

Coordinates are clamped to ``+-COORD_LIMIT`` before their lattice cell is
taken, so the integer conversion behaves identically on both sides.
"""

from __future__ import annotations

import numpy as np

COORD_LIMIT = 1.0e6
SEED_MIX = 0x9E3779B9
INV_2_24 = 1.0 / 16777216.0

# The 12 edge directions of a cube: gradient noise picks one per lattice corner.
GRADIENTS = np.array(
    [
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, 1, -1),
        (0, -1, -1),
    ],
    dtype=np.float64,
)


def seed_bits(seed: int) -> np.uint32:
    """The value xor-ed into a cell's z coordinate for ``seed``."""
    return np.uint32((int(seed) * SEED_MIX) & 0xFFFFFFFF)


def cell_index(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(floor(p) as uint32 bit patterns, p - floor(p))`` for clamped coordinates."""
    clamped = np.clip(p, -COORD_LIMIT, COORD_LIMIT)
    whole = np.floor(clamped)
    bits = (whole.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    return bits, clamped - whole


def pcg3d(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """The ``pcg3d`` hash of three ``uint32`` arrays (returns three ``uint32`` arrays)."""
    x = x * np.uint32(1664525) + np.uint32(1013904223)
    y = y * np.uint32(1664525) + np.uint32(1013904223)
    z = z * np.uint32(1664525) + np.uint32(1013904223)
    x = x + y * z
    y = y + z * x
    z = z + x * y
    x = x ^ (x >> np.uint32(16))
    y = y ^ (y >> np.uint32(16))
    z = z ^ (z >> np.uint32(16))
    x = x + y * z
    y = y + z * x
    z = z + x * y
    return x, y, z


def unit(h: np.ndarray) -> np.ndarray:
    """A hash as a float in ``[0, 1)`` (its top 24 bits)."""
    return (h >> np.uint32(8)).astype(np.float64) * INV_2_24


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _corner_hash(ix, iy, iz, dx: int, dy: int, dz: int, seed: np.uint32) -> np.ndarray:
    hx, _, _ = pcg3d(ix + np.uint32(dx), iy + np.uint32(dy), (iz + np.uint32(dz)) ^ seed)
    return hx


def value_noise(p: np.ndarray, seed: int) -> np.ndarray:
    """Value noise in ``[0, 1]`` at ``(N, 3)`` points (random values at lattice corners)."""
    bits, f = cell_index(p)
    ix, iy, iz = bits[:, 0], bits[:, 1], bits[:, 2]
    s = seed_bits(seed)
    u = _fade(f)
    corner = {}
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                corner[dx, dy, dz] = unit(_corner_hash(ix, iy, iz, dx, dy, dz, s))
    return _trilinear(corner, u)


def gradient_noise(p: np.ndarray, seed: int) -> np.ndarray:
    """Perlin-style gradient noise mapped to ``[0, 1]`` (0.5 at every lattice point)."""
    bits, f = cell_index(p)
    ix, iy, iz = bits[:, 0], bits[:, 1], bits[:, 2]
    s = seed_bits(seed)
    u = _fade(f)
    corner = {}
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                h = _corner_hash(ix, iy, iz, dx, dy, dz, s)
                g = GRADIENTS[(h % np.uint32(12)).astype(np.int64)]
                corner[dx, dy, dz] = (
                    g[:, 0] * (f[:, 0] - dx) + g[:, 1] * (f[:, 1] - dy) + g[:, 2] * (f[:, 2] - dz)
                )
    n = _trilinear(corner, u)
    return np.clip(0.5 + 0.5 * n, 0.0, 1.0)


def _trilinear(corner: dict, u: np.ndarray) -> np.ndarray:
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    def lerp(a, b, t):
        return a + (b - a) * t

    x00 = lerp(corner[0, 0, 0], corner[1, 0, 0], ux)
    x10 = lerp(corner[0, 1, 0], corner[1, 1, 0], ux)
    x01 = lerp(corner[0, 0, 1], corner[1, 0, 1], ux)
    x11 = lerp(corner[0, 1, 1], corner[1, 1, 1], ux)
    return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz)


def noise(p: np.ndarray, seed: int, kind: str) -> np.ndarray:
    return gradient_noise(p, seed) if kind == "perlin" else value_noise(p, seed)


def fbm(
    p: np.ndarray, seed: int, kind: str, octaves: int, lacunarity: float, gain: float
) -> np.ndarray:
    """Fractal sum of ``octaves`` noise layers, normalized to ``[0, 1]``."""
    total = np.zeros(p.shape[0])
    amplitude, frequency, norm = 1.0, 1.0, 0.0
    for octave in range(octaves):
        total = total + noise(p * frequency, seed + octave, kind) * amplitude
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm


def voronoi(p: np.ndarray, seed: int, randomness: float, dimensions: int) -> np.ndarray:
    """``(N, 3)``: distance to the nearest feature point (F1), ``F2 - F1`` and a random value
    of the nearest cell, each in ``[0, 1]``."""
    q = p.copy()
    if dimensions == 2:
        q[:, 2] = 0.0
    bits, f = cell_index(q)
    s = seed_bits(seed)
    n = p.shape[0]
    f1 = np.full(n, 8.0)
    f2 = np.full(n, 8.0)
    cell_value = np.zeros(n)
    zs = (0,) if dimensions == 2 else (-1, 0, 1)
    for dz in zs:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                hx, hy, hz = pcg3d(
                    bits[:, 0] + np.uint32(dx & 0xFFFFFFFF),
                    bits[:, 1] + np.uint32(dy & 0xFFFFFFFF),
                    (bits[:, 2] + np.uint32(dz & 0xFFFFFFFF)) ^ s,
                )
                px = dx + 0.5 + (unit(hx) - 0.5) * randomness - f[:, 0]
                py = dy + 0.5 + (unit(hy) - 0.5) * randomness - f[:, 1]
                if dimensions == 2:
                    d = np.sqrt(px * px + py * py)
                else:
                    pz = dz + 0.5 + (unit(hz) - 0.5) * randomness - f[:, 2]
                    d = np.sqrt(px * px + py * py + pz * pz)
                value = unit(hx + hy + hz)
                closer = d < f1
                f2 = np.where(closer, f1, np.minimum(f2, d))
                cell_value = np.where(closer, value, cell_value)
                f1 = np.where(closer, d, f1)
    return np.stack([np.clip(f1, 0.0, 1.0), np.clip(f2 - f1, 0.0, 1.0), cell_value], axis=1)


def cell_random(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """A random value in ``[0, 1)`` per integer cell ``(ix, iy)`` (bricks, tiles)."""
    bx = (ix.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    by = (iy.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    hx, _, _ = pcg3d(bx, by, np.zeros_like(bx) ^ seed_bits(seed))
    return unit(hx)
