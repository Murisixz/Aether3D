"""Deterministic noise for world generation: integer hashes, gradient and value noise in 2D
and 3D, fractal sums (fBm, ridged multifractal, billow), domain warping and shaping curves.

Every random decision is a hash of integers -- a seed and lattice coordinates -- so a value
never depends on which other values were computed before it: chunks can be generated in any
order, on any thread, and agree exactly where they meet.

**Exactness.** The functions here use only

* wrapping uint64 arithmetic (multiply, xor, shift) for hashes, and
* correctly rounded IEEE-754 float64 operations: ``+ - * /``, ``sqrt``, ``floor``, ``abs``,
  ``min``, ``max`` and comparisons,

never a libm function such as ``exp``, ``sin`` or ``pow`` whose last bit differs between
platforms. NumPy applies each of these element by element (vectorized or not, the result of
each element is the correctly rounded one), and the gradient tables are decimal literals
that parse to the same float64 everywhere. The same inputs therefore give the same bits on
Linux, macOS and Windows, with or without the native kernels, which perform the same
operations in the same order (``tests/test_worlds_noise.py`` pins golden hashes).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

U64 = np.uint64
_K1 = U64(0x9E3779B97F4A7C15)
_K2 = U64(0xC2B2AE3D27D4EB4F)
_K3 = U64(0x165667B19E3779F9)
_M1 = U64(0xBF58476D1CE4E5B9)
_M2 = U64(0x94D049BB133111EB)
_S30, _S27, _S31, _S60, _S11 = U64(30), U64(27), U64(31), U64(60), U64(11)
_INV53 = 1.0 / 9007199254740992.0  # 2^-53 (exact)
_MASK64 = (1 << 64) - 1

# 16 unit gradients at 22.5 degree steps (cos, sin as decimal literals: exact everywhere).
_GRAD2 = np.array(
    [
        [1.0, 0.0],
        [0.9238795325112867, 0.3826834323650898],
        [0.7071067811865476, 0.7071067811865476],
        [0.3826834323650898, 0.9238795325112867],
        [0.0, 1.0],
        [-0.3826834323650898, 0.9238795325112867],
        [-0.7071067811865476, 0.7071067811865476],
        [-0.9238795325112867, 0.3826834323650898],
        [-1.0, 0.0],
        [-0.9238795325112867, -0.3826834323650898],
        [-0.7071067811865476, -0.7071067811865476],
        [-0.3826834323650898, -0.9238795325112867],
        [0.0, -1.0],
        [0.3826834323650898, -0.9238795325112867],
        [0.7071067811865476, -0.7071067811865476],
        [0.9238795325112867, -0.3826834323650898],
    ]
)
_G2X = np.ascontiguousarray(_GRAD2[:, 0])
_G2Y = np.ascontiguousarray(_GRAD2[:, 1])
# Perlin's 12 cube-edge gradients, padded to 16 (as in "Improving noise", 2002).
_GRAD3 = np.array(
    [
        [1, 1, 0],
        [-1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [-1, 0, 1],
        [1, 0, -1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, -1, 1],
        [0, 1, -1],
        [0, -1, -1],
        [1, 1, 0],
        [0, -1, 1],
        [-1, 1, 0],
        [0, -1, -1],
    ],
    dtype=np.float64,
)
_G3X = np.ascontiguousarray(_GRAD3[:, 0])
_G3Y = np.ascontiguousarray(_GRAD3[:, 1])
_G3Z = np.ascontiguousarray(_GRAD3[:, 2])

NOISE2_SCALE = 1.4142135623730951
"""2D gradient noise times this lies in about [-1, 1] (unit gradients reach sqrt(1/2))."""
NOISE3_SCALE = 0.9761904761904762
"""3D gradient noise times this lies in about [-1, 1]."""

OCTAVE_ROTATION = (0.8, 0.6)
"""cos and sin of the rotation (36.87 degrees) between octaves: no octave shares the lattice
axes of the one before, which keeps fractals free of grid-aligned artefacts."""


# --------------------------------------------------------------------------------------- #
# Seeds and hashes
# --------------------------------------------------------------------------------------- #


def check_seed(seed: object) -> int:
    """A seed as an int in ``[0, 2^64)`` (negative ints wrap, as in two's complement)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"a seed is an integer, got {type(seed).__name__}")
    return int(seed) & _MASK64


def derive(seed: int, *parts: object) -> int:
    """A seed for a named stream: BLAKE2b of the seed and the parts (strings or ints).

    Streams decorrelate the fields of one world (``derive(seed, "terrain", "hills")``)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(seed & _MASK64).to_bytes(8, "little"))
    for part in parts:
        h.update(b"\x00" + str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def _mix(h: np.ndarray) -> np.ndarray:
    """The SplitMix64 finalizer, in place on a uint64 array (returned)."""
    h ^= h >> _S30
    h *= _M1
    h ^= h >> _S27
    h *= _M2
    h ^= h >> _S31
    return h


def _as_u64(values: np.ndarray) -> np.ndarray:
    """Integers (any int dtype) as uint64 by two's complement (``-1 -> 2^64 - 1``)."""
    return np.ascontiguousarray(values, dtype=np.int64).view(np.uint64)


def hash2(seed: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """uint64 hashes of integer points ``(ix, iy)`` (broadcast) under ``seed``."""
    a = _as_u64(ix) * _K1
    b = _as_u64(iy) * _K2
    h = np.bitwise_xor(a, b)
    h ^= U64(seed & _MASK64)
    return _mix(h)


def hash3(seed: int, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """uint64 hashes of integer points ``(ix, iy, iz)`` (broadcast) under ``seed``."""
    h = np.bitwise_xor(_as_u64(ix) * _K1, _as_u64(iy) * _K2)
    h ^= _as_u64(iz) * _K3
    h ^= U64(seed & _MASK64)
    return _mix(h)


def unit_float(h: np.ndarray) -> np.ndarray:
    """uint64 hashes as floats in ``[0, 1)`` (the top 53 bits: exact)."""
    return (h >> _S11).astype(np.float64) * _INV53


def random2(seed: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """A uniform float in ``[0, 1)`` per integer point (placement jitter, choices)."""
    return unit_float(hash2(seed, ix, iy))


def random3(seed: int, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """A uniform float in ``[0, 1)`` per integer 3D point."""
    return unit_float(hash3(seed, ix, iy, iz))


# --------------------------------------------------------------------------------------- #
# Lattice noise
# --------------------------------------------------------------------------------------- #


def fade(t: np.ndarray) -> np.ndarray:
    """Perlin's quintic fade ``6t^5 - 15t^4 + 10t^3`` (C2: noise with continuous curvature)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def cos_sin_deg(degrees: float) -> tuple[float, float]:
    """``(cos, sin)`` of an angle in degrees with only correctly rounded operations (exact
    quadrant reduction, then Taylor series on [-45, 45] degrees): the same bits on every
    platform, unlike libm's ``cos``/``sin``. Accurate to a few ulp."""
    a = float(degrees) % 360.0
    q = int((a + 45.0) // 90.0)
    r = (a - 90.0 * q) * (3.141592653589793 / 180.0)
    quadrant = q % 4
    r2 = r * r
    c = 1.0
    s = r
    term_c = 1.0
    term_s = r
    for k in range(1, 12):
        term_c = -term_c * r2 / ((2 * k - 1) * (2 * k))
        term_s = -term_s * r2 / ((2 * k) * (2 * k + 1))
        c += term_c
        s += term_s
    return ((c, s), (-s, c), (-c, -s), (s, -c))[quadrant]


_ATAN = tuple((-1.0) ** k / (2 * k + 1) for k in range(13))  # (exact: decimal fractions)
_ATANH = tuple(1.0 / (2 * k + 1) for k in range(16))
_LOG10_2 = 0.30102999566398119521  # log10(2), the nearest double
_TWO_OVER_LN2 = 2.0 / 0.69314718055994530942


def atan2_deg(y: float, x: float) -> float:
    """``atan2(y, x)`` in degrees with only correctly rounded operations (two half-angle
    reductions, then a Taylor series): the same bits on every platform. Accurate to about
    1e-13 degrees; ``atan2_deg(0, 0)`` is 0."""
    y, x = float(y), float(x)
    if x == 0.0 and y == 0.0:
        return 0.0
    ax, ay = abs(x), abs(y)
    swap = ay > ax
    t = ax / ay if swap else ay / ax  # in [0, 1]
    for _ in range(2):  # atan(t) = 2 atan(t / (1 + sqrt(1 + t^2)))
        t = t / (1.0 + math.sqrt(1.0 + t * t))
    t2 = t * t  # t <= tan(pi / 16)
    s = 0.0
    for c in reversed(_ATAN):
        s = s * t2 + c
    a = 4.0 * t * s  # radians, in [0, pi / 4]
    if swap:
        a = math.pi / 2.0 - a
    if x < 0.0:
        a = math.pi - a
    if y < 0.0:
        a = -a
    return a * (180.0 / math.pi)


def log10(x: Any) -> np.ndarray:
    """``log10`` of positive values with only correctly rounded operations (an exact
    ``frexp`` split, then an atanh series of the centred mantissa): the same bits on every
    platform. Absolute error about 2e-15."""
    m, e = np.frexp(np.asarray(x, dtype=np.float64))  # x = m 2^e, m in [0.5, 1)
    m = m * 2.0
    e = e - 1
    big = m > 1.4142135623730951  # centre the mantissa on 1: [0.707, 1.414]
    m = np.where(big, m * 0.5, m)
    e = np.where(big, e + 1, e)
    z = (m - 1.0) / (m + 1.0)  # |z| <= 0.172
    z2 = z * z
    s = np.zeros_like(z)
    for c in reversed(_ATANH):
        s = s * z2 + c
    return (e + _TWO_OVER_LN2 * z * s) * _LOG10_2


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Cubic Hermite step from 0 at ``edge0`` to 1 at ``edge1`` (clamped)."""
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Quintic step (zero first and second derivatives at both ends), clamped."""
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return fade(t)


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf = np.floor(x)
    return xf.astype(np.int64), x - xf  # (the subtraction is exact)


def gradient2(seed: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D gradient (Perlin) noise at float64 points, about [-1, 1], zero on the lattice."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ix, fx = _split(x)
    iy, fy = _split(y)
    ax0 = _as_u64(ix) * _K1
    ax1 = _as_u64(ix + 1) * _K1
    by0 = _as_u64(iy) * _K2
    by1 = _as_u64(iy + 1) * _K2
    s = U64(seed & _MASK64)
    fx1 = fx - 1.0
    fy1 = fy - 1.0

    def corner(a: np.ndarray, b: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        h = np.bitwise_xor(a, b)
        h ^= s
        k = (_mix(h) >> _S60).astype(np.intp)
        return _G2X[k] * dx + _G2Y[k] * dy

    n00 = corner(ax0, by0, fx, fy)
    n10 = corner(ax1, by0, fx1, fy)
    n01 = corner(ax0, by1, fx, fy1)
    n11 = corner(ax1, by1, fx1, fy1)
    u = fade(fx)
    v = fade(fy)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)) * NOISE2_SCALE


def gradient3(seed: int, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """3D gradient (Perlin) noise at float64 points, about [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ix, fx = _split(x)
    iy, fy = _split(y)
    iz, fz = _split(z)
    ax = (_as_u64(ix) * _K1, _as_u64(ix + 1) * _K1)
    by = (_as_u64(iy) * _K2, _as_u64(iy + 1) * _K2)
    cz = (_as_u64(iz) * _K3, _as_u64(iz + 1) * _K3)
    dx = (fx, fx - 1.0)
    dy = (fy, fy - 1.0)
    dz = (fz, fz - 1.0)
    s = U64(seed & _MASK64)
    n = {}
    for i in (0, 1):
        for j in (0, 1):
            ab = np.bitwise_xor(ax[i], by[j])
            for k in (0, 1):
                h = np.bitwise_xor(ab, cz[k])
                h ^= s
                g = (_mix(h) >> _S60).astype(np.intp)
                n[i, j, k] = _G3X[g] * dx[i] + _G3Y[g] * dy[j] + _G3Z[g] * dz[k]
    u = fade(fx)
    v = fade(fy)
    w = fade(fz)
    x00 = n[0, 0, 0] + u * (n[1, 0, 0] - n[0, 0, 0])
    x10 = n[0, 1, 0] + u * (n[1, 1, 0] - n[0, 1, 0])
    x01 = n[0, 0, 1] + u * (n[1, 0, 1] - n[0, 0, 1])
    x11 = n[0, 1, 1] + u * (n[1, 1, 1] - n[0, 1, 1])
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return (y0 + w * (y1 - y0)) * NOISE3_SCALE


def gradient2_d(seed: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """2D gradient noise and its exact partial derivatives ``(n, dn/dx, dn/dy)`` (same
    value as :func:`gradient2`)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ix, fx = _split(x)
    iy, fy = _split(y)
    ax0 = _as_u64(ix) * _K1
    ax1 = _as_u64(ix + 1) * _K1
    by0 = _as_u64(iy) * _K2
    by1 = _as_u64(iy + 1) * _K2
    s = U64(seed & _MASK64)
    fx1 = fx - 1.0
    fy1 = fy - 1.0

    def grad(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.bitwise_xor(a, b)
        h ^= s
        k = (_mix(h) >> _S60).astype(np.intp)
        return _G2X[k], _G2Y[k]

    g00 = grad(ax0, by0)
    g10 = grad(ax1, by0)
    g01 = grad(ax0, by1)
    g11 = grad(ax1, by1)
    n00 = g00[0] * fx + g00[1] * fy
    n10 = g10[0] * fx1 + g10[1] * fy
    n01 = g01[0] * fx + g01[1] * fy1
    n11 = g11[0] * fx1 + g11[1] * fy1
    u = fade(fx)
    v = fade(fy)
    du = 30.0 * fx * fx * fx1 * fx1
    dv = 30.0 * fy * fy * fy1 * fy1
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    n = nx0 + v * (nx1 - nx0)
    dx0 = g00[0] + u * (g10[0] - g00[0]) + du * (n10 - n00)
    dx1 = g01[0] + u * (g11[0] - g01[0]) + du * (n11 - n01)
    dy0 = g00[1] + u * (g10[1] - g00[1])
    dy1 = g01[1] + u * (g11[1] - g01[1])
    dx = dx0 + v * (dx1 - dx0)
    dy = dy0 + v * (dy1 - dy0) + dv * (nx1 - nx0)
    return n * NOISE2_SCALE, dx * NOISE2_SCALE, dy * NOISE2_SCALE


def eroded2(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int = 8,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    strength: float = 1.0,
    split: int | None = None,
) -> Any:
    """ "Eroded" fBm (after Quilez's derivative-damped fractals): each octave is divided by
    ``1 + strength * |sum of the octave gradients so far|^2``, so detail stays on ridges and
    flat ground and fades on steep flanks -- smooth slopes, sharp crests and gully-like
    valleys without a simulation. About [-1, 1]. With ``split``, returns ``(macro, full)``:
    the sum of the octaves before ``split`` and of all of them (same normalization)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros(np.broadcast(x, y).shape)
    sx = np.zeros_like(total)
    sy = np.zeros_like(total)
    norm = 0.0
    macro = None
    px, py = x, y
    for k, (s, f, a) in enumerate(_octaves(seed, octaves, frequency, lacunarity, gain)):
        if k == split:
            macro = total.copy()
        if k:
            px, py = _rotate(px, py)
        n, dx, dy = gradient2_d(s, px * f, py * f)
        sx += a * dx
        sy += a * dy
        total += a * n / (1.0 + strength * (sx * sx + sy * sy))
        norm += a
    return _finish(total, macro, norm, split, clip=False)


def value2(seed: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D value noise in [0, 1): hashed lattice values, quintic interpolation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ix, fx = _split(x)
    iy, fy = _split(y)
    v00 = random2(seed, ix, iy)
    v10 = random2(seed, ix + 1, iy)
    v01 = random2(seed, ix, iy + 1)
    v11 = random2(seed, ix + 1, iy + 1)
    u = fade(fx)
    v = fade(fy)
    a = v00 + u * (v10 - v00)
    b = v01 + u * (v11 - v01)
    return a + v * (b - a)


# --------------------------------------------------------------------------------------- #
# Fractals
# --------------------------------------------------------------------------------------- #


def _octaves(
    seed: int, octaves: int, frequency: float, lacunarity: float, gain: float
) -> list[tuple[int, float, float]]:
    """``(seed, frequency, amplitude)`` per octave (Python floats: exact repeated products)."""
    out = []
    f, a = float(frequency), 1.0
    for i in range(int(octaves)):
        out.append((derive(seed, "octave", i), f, a))
        f *= lacunarity
        a *= gain
    return out


def _rotate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, s = OCTAVE_ROTATION
    return c * x - s * y, s * x + c * y


def fbm2(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int = 5,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    split: int | None = None,
) -> Any:
    """Fractal Brownian motion: octaves of gradient noise, normalized to about [-1, 1]
    (``split``: ``(macro, full)``, see :func:`eroded2`)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros(np.broadcast(x, y).shape)
    norm = 0.0
    macro = None
    px, py = x, y
    for k, (s, f, a) in enumerate(_octaves(seed, octaves, frequency, lacunarity, gain)):
        if k == split:
            macro = total.copy()
        if k:
            px, py = _rotate(px, py)
        total += a * gradient2(s, px * f, py * f)
        norm += a
    return _finish(total, macro, norm, split, clip=False)


def ridged2(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int = 6,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    offset: float = 1.0,
    sharpness: float = 2.0,
    split: int | None = None,
) -> Any:
    """Musgrave's ridged multifractal in [0, 1]: sharp crests where the noise crosses zero,
    each octave weighted by the one before (ridges carry detail, valleys stay smooth)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros(np.broadcast(x, y).shape)
    weight = np.ones_like(total)
    norm = 0.0
    macro = None
    px, py = x, y
    for k, (s, f, a) in enumerate(_octaves(seed, octaves, frequency, lacunarity, gain)):
        if k == split:
            macro = total.copy()
        if k:
            px, py = _rotate(px, py)
        signal = offset - np.abs(gradient2(s, px * f, py * f))
        signal = signal * signal
        signal *= weight
        weight = np.clip(signal * sharpness, 0.0, 1.0)
        total += a * signal
        norm += a * offset * offset
    return _finish(total, macro, norm, split, clip=True)


def _finish(
    total: np.ndarray, macro: np.ndarray | None, norm: float, split: int | None, clip: bool
) -> Any:
    """Normalize a fractal sum (and its macro part when split)."""

    def done(v: np.ndarray) -> np.ndarray:
        v = v / norm if norm > 0 else v
        return np.clip(v, 0.0, 1.0) if clip else v

    if split is None:
        return done(total)
    if macro is None:  # the split is past the last octave: everything is macro
        macro = total
    return done(macro), done(total)


def billow2(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int = 5,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Billowy fractal in about [0, 1]: absolute values of octaves (rounded bumps, creases)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros(np.broadcast(x, y).shape)
    norm = 0.0
    px, py = x, y
    for k, (s, f, a) in enumerate(_octaves(seed, octaves, frequency, lacunarity, gain)):
        if k:
            px, py = _rotate(px, py)
        total += a * np.abs(gradient2(s, px * f, py * f))
        norm += a
    return np.clip(total / norm, 0.0, 1.0) if norm > 0 else total


def fbm3(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    octaves: int = 3,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """3D fBm, about [-1, 1]."""
    total = np.zeros(np.broadcast(x, y, z).shape)
    norm = 0.0
    for s, f, a in _octaves(seed, octaves, frequency, lacunarity, gain):
        total += a * gradient3(s, np.multiply(x, f), np.multiply(y, f), np.multiply(z, f))
        norm += a
    return total / norm if norm > 0 else total


def warp2(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    amplitude: float,
    frequency: float,
    octaves: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Domain warping: ``p + amplitude * (fbm_a(p), fbm_b(p))`` (two independent fractals)."""
    if amplitude == 0:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx = fbm2(derive(seed, "warp-x"), x, y, octaves=octaves, frequency=frequency)
    dy = fbm2(derive(seed, "warp-y"), x, y, octaves=octaves, frequency=frequency)
    return x + amplitude * dx, y + amplitude * dy


# --------------------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------------------- #


class MonotoneCurve:
    """A monotone piecewise cubic through points ``(x_k, y_k)`` (Fritsch and Carlson 1980):
    no overshoot between knots, C1, constant beyond the ends. Height curves of terrains."""

    __slots__ = ("x", "y", "m")

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        pts = sorted((float(a), float(b)) for a, b in points)
        if len(pts) < 2:
            raise ValueError("a curve needs at least two points")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        for a, b in zip(xs, xs[1:], strict=False):
            if not b > a:
                raise ValueError("curve points need increasing x")
        n = len(xs)
        d = [(ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]) for k in range(n - 1)]
        m = [0.0] * n
        m[0], m[-1] = d[0], d[-1]
        for k in range(1, n - 1):
            m[k] = 0.0 if d[k - 1] * d[k] <= 0 else (d[k - 1] + d[k]) / 2.0
        for k in range(n - 1):  # Fritsch-Carlson limiter: alpha^2 + beta^2 <= 9
            if d[k] == 0.0:
                m[k] = m[k + 1] = 0.0
                continue
            a, b = m[k] / d[k], m[k + 1] / d[k]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / math.sqrt(s)
                m[k], m[k + 1] = t * a * d[k], t * b * d[k]
        self.x = np.array(xs)
        self.y = np.array(ys)
        self.m = np.array(m)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        xs, ys, ms = self.x, self.y, self.m
        k = np.clip(np.searchsorted(xs, v, side="right") - 1, 0, len(xs) - 2)
        x0, x1 = xs[k], xs[k + 1]
        h = x1 - x0
        t = np.clip((v - x0) / h, 0.0, 1.0)
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        out = h00 * ys[k] + h10 * h * ms[k] + h01 * ys[k + 1] + h11 * h * ms[k + 1]
        out = np.where(v <= xs[0], ys[0], out)
        return np.where(v >= xs[-1], ys[-1], out)


def terrace(h: np.ndarray, step: float, sharpness: float) -> np.ndarray:
    """Terraces of height ``step``: flat treads joined by risers whose steepness grows with
    ``sharpness`` (0: no terracing, 1: nearly vertical risers). Monotone in ``h``."""
    if step <= 0 or sharpness <= 0:
        return np.asarray(h, dtype=np.float64)
    q = np.asarray(h, dtype=np.float64) / step
    base = np.floor(q)
    f = q - base
    # Risers occupy the last (1 - s/2..) of each step: a smoothstep compressed toward 1.
    w = max(1e-6, 1.0 - 0.95 * min(sharpness, 1.0))
    t = np.clip((f - (1.0 - w)) / w, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    return (base + s) * step


def blend(weights: np.ndarray, table: Any) -> np.ndarray:
    """``weights @ table`` without BLAS: per point, the products summed in the order of
    the last axis of ``weights`` (biomes), so results never depend on how many points are
    evaluated together or on the BLAS library (``weights`` (..., B); ``table`` (B,) or
    (B, C))."""
    weights = np.asarray(weights, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)
    if table.ndim == 1:
        out = weights[..., 0] * table[0]
        for k in range(1, weights.shape[-1]):
            out = out + weights[..., k] * table[k]
        return out
    out = weights[..., 0, None] * table[0]
    for k in range(1, weights.shape[-1]):
        out = out + weights[..., k, None] * table[k]
    return out
