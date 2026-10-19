"""Random numbers and sample warps for the path tracer.

**Random numbers** come from a counter-based generator: every value is a
hash of ``(seed, pixel, sample, dimension)`` (SplitMix64's finalizer, Steele,
Lea and Flood 2014), so it does not depend on how paths are batched, on the
order they are traced in or on how many processes trace them. Dimensions are
fixed per path vertex and purpose (:func:`dimension`), which keeps the
random numbers of different decisions independent.

**Owen-scrambled Sobol'** (0.20.0, ``PathSettings.sampler="sobol"``): decisions that are
made together -- the pixel position and the lens point, a BSDF sample (its two numbers,
the lobe and Russian roulette), the three numbers of each light sample, each AO ray --
draw one point of a 4D Sobol' sequence (Joe and Kuo's generator matrices for dimensions
2-4, the first being van der Corput), indexed by the path's sample number after a nested
uniform scramble and scrambled again per coordinate (Burley 2020, "Practical Hash-based
Owen Scrambling", JCGT 9(4); Laine and Karras' permutation). The scrambling seeds are
hashes of the pixel, the path vertex and the group, so pixels and decisions stay
decorrelated while a pixel's samples are stratified. Values are 24-bit fractions (as
float32 holds them), integer arithmetic only: the GPU computes the same bits. Other
dimensions use the hash above.

**Warps** map uniform numbers to directions and points with known densities
(solid-angle densities unless noted):

* cosine-weighted hemisphere: ``pdf = cos(theta) / pi``;
* uniform cone of half-angle ``theta_max``: ``pdf = 1 / (2 pi (1 - cos theta_max))``;
* uniform triangle (area density ``1 / area``): ``b0 = 1 - sqrt(u1)``,
  ``b1 = u2 sqrt(u1)``;
* GGX visible normals (Heitz 2018, "Sampling the GGX Distribution of
  Visible Normals", JCGT 7(4)): microfacet normals ``h`` with density
  ``G1(v) max(0, v.h) D(h) / (n.v)``.

Local frames have the normal along +z; :func:`frame` builds an orthonormal
basis around any unit vector (Duff et al. 2017, "Building an Orthonormal
Basis, Revisited", JCGT 6(1)).
"""

from __future__ import annotations

import math

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = (np.uint64(k) for k in (30, 27, 31, 11))
_PIXEL = np.uint64(0xD1B54A32D192ED03)
_SAMPLE = np.uint64(0xAEF17502108EF2D9)
_DIM = np.uint64(0xDB4F0B9175AE2165)
_TO_UNIT = 1.0 / 2.0**53

DIMENSIONS_PER_VERTEX = 1 << 32
"""Random dimensions reserved for each path vertex (see :func:`dimension`)."""


def mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer of ``x + golden ratio`` (uint64 in, uint64 out; wraps)."""
    x = x + _GOLDEN
    x = (x ^ (x >> _S30)) * _M1
    x = (x ^ (x >> _S27)) * _M2
    return x ^ (x >> _S31)


def path_keys(seed: int, pixel: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """One uint64 key per path from its seed, pixel and sample index."""
    with np.errstate(over="ignore"):
        key = mix(np.full(np.shape(pixel), np.uint64(seed)))
        key = mix(key ^ (np.asarray(pixel, dtype=np.uint64) * _PIXEL))
        return mix(key ^ (np.asarray(sample, dtype=np.uint64) * _SAMPLE))


def uniform(keys: np.ndarray, dim: int) -> np.ndarray:
    """A uniform number in ``[0, 1)`` per key for random dimension ``dim``."""
    if isinstance(keys, SobolKeys):
        return keys.uniform(dim)
    with np.errstate(over="ignore"):
        x = mix(keys ^ (np.uint64(dim) * _DIM + _GOLDEN))
    return (x >> _S11).astype(np.float64) * _TO_UNIT


SAMPLERS = ("random", "sobol")


def sampler_keys(sampler: str, seed: int, pixel: np.ndarray, sample: np.ndarray):
    """The keys :func:`uniform` draws from: hash keys (``"random"``) or
    :class:`SobolKeys`."""
    keys = path_keys(seed, pixel, sample)
    if sampler == "random":
        return keys
    pixel_hash = (path_keys(seed, pixel, np.zeros_like(sample)) >> np.uint64(32)).astype(np.uint32)
    return SobolKeys(pixel_hash, np.asarray(sample, dtype=np.uint32), keys)


# Joe and Kuo's direction numbers (new-joe-kuo-6.21201) for Sobol' dimensions 2-4, as
# (s, a, m); dimension 1 is van der Corput.
_JOE_KUO = ((1, 0, (1,)), (2, 1, (1, 3)), (3, 1, (1, 3, 1)))


def _directions() -> np.ndarray:
    """``(4, 32) uint32``: the generator matrices' columns (bit ``j`` of the index)."""
    v = np.zeros((4, 32), dtype=np.uint64)
    v[0] = [1 << (31 - j) for j in range(32)]
    for d, (deg, a, m) in enumerate(_JOE_KUO, start=1):
        col = [0] * 33  # 1-based, scaled by 2^32
        for i in range(1, deg + 1):
            col[i] = m[i - 1] << (32 - i)
        for i in range(deg + 1, 33):
            x = col[i - deg] ^ (col[i - deg] >> deg)
            for k in range(1, deg):
                if (a >> (deg - 1 - k)) & 1:
                    x ^= col[i - k]
            col[i] = x
        v[d] = col[1:33]
    return v.astype(np.uint32)


SOBOL_DIRECTIONS = _directions()
"""The generator matrices of the first four Sobol' dimensions (``(4, 32) uint32``)."""
_SOBOL_TABLES = np.zeros((4, 4, 256), dtype=np.uint32)  # per lane, per index byte
for _lane in range(4):
    for _byte in range(4):
        for _value in range(256):
            _x = 0
            for _bit in range(8):
                if (_value >> _bit) & 1:
                    _x ^= int(SOBOL_DIRECTIONS[_lane, 8 * _byte + _bit])
            _SOBOL_TABLES[_lane, _byte, _value] = _x


def sobol(index: np.ndarray, lane: int) -> np.ndarray:
    """Sobol' coordinate ``lane`` (0-3) of points ``index`` (uint32)."""
    t = _SOBOL_TABLES[lane]
    i = index.astype(np.uint32)
    return (
        t[0][i & 255]
        ^ t[1][(i >> np.uint32(8)) & 255]
        ^ t[2][(i >> np.uint32(16)) & 255]
        ^ t[3][(i >> np.uint32(24)) & 255]
    )


def reverse_bits(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    x = ((x >> np.uint32(1)) & np.uint32(0x55555555)) | (
        (x & np.uint32(0x55555555)) << np.uint32(1)
    )
    x = ((x >> np.uint32(2)) & np.uint32(0x33333333)) | (
        (x & np.uint32(0x33333333)) << np.uint32(2)
    )
    x = ((x >> np.uint32(4)) & np.uint32(0x0F0F0F0F)) | (
        (x & np.uint32(0x0F0F0F0F)) << np.uint32(4)
    )
    x = ((x >> np.uint32(8)) & np.uint32(0x00FF00FF)) | (
        (x & np.uint32(0x00FF00FF)) << np.uint32(8)
    )
    return (x >> np.uint32(16)) | (x << np.uint32(16))


def hash32(x: np.ndarray) -> np.ndarray:
    """A 32-bit integer hash (Wellons' lowbias32)."""
    with np.errstate(over="ignore"):
        x = np.asarray(x, dtype=np.uint32)
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7FEB352D)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846CA68B)
        return x ^ (x >> np.uint32(16))


def nested_uniform_scramble(x: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Owen scrambling of 32-bit fractions ``x`` (Burley 2020, Laine and Karras' hash)."""
    with np.errstate(over="ignore"):
        v = reverse_bits(x)
        v = v + np.asarray(seed, dtype=np.uint32)
        v = v ^ (v * np.uint32(0x6C50B47C))
        v = v ^ (v * np.uint32(0xB82F1E52))
        v = v ^ (v * np.uint32(0xC7AFE638))
        v = v ^ (v * np.uint32(0x8D22F6E6))
        return reverse_bits(v)


def sobol_group(slot: int) -> tuple[int, int] | None:
    """``(group, lane)`` of a decision slot for the Sobol' sampler, or ``None`` (hashed)."""
    if slot in (SLOT_PIXEL, SLOT_PIXEL + 1, SLOT_LENS, SLOT_LENS + 1):
        return 0, {SLOT_PIXEL: 0, SLOT_PIXEL + 1: 1, SLOT_LENS: 2, SLOT_LENS + 1: 3}[slot]
    if slot in (SLOT_BSDF, SLOT_BSDF + 1, SLOT_LOBE, SLOT_RR):
        return 1, {SLOT_BSDF: 0, SLOT_BSDF + 1: 1, SLOT_LOBE: 2, SLOT_RR: 3}[slot]
    if SLOT_AO <= slot < SLOT_AO + 2048:
        return 2 + (slot - SLOT_AO) // 2, (slot - SLOT_AO) % 2
    if slot >= SLOT_LIGHTS:
        k, lane = divmod(slot - SLOT_LIGHTS, 3)
        return 4096 + k, lane
    return None


class SobolKeys:
    """Keys of the Sobol' sampler: each path's pixel hash and sample number (and its hash
    key, for dimensions the sampler does not group)."""

    __slots__ = ("pixel", "sample", "hashed")

    def __init__(self, pixel: np.ndarray, sample: np.ndarray, hashed: np.ndarray) -> None:
        self.pixel = pixel
        self.sample = sample
        self.hashed = hashed

    def __getitem__(self, rows) -> SobolKeys:
        return SobolKeys(self.pixel[rows], self.sample[rows], self.hashed[rows])

    def __len__(self) -> int:
        return int(self.pixel.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.pixel.shape

    def uniform(self, dim: int) -> np.ndarray:
        vertex, slot = divmod(int(dim), DIMENSIONS_PER_VERTEX)
        group = sobol_group(slot)
        if group is None:
            return uniform(self.hashed, dim)
        g, lane = group
        with np.errstate(over="ignore"):
            salt = hash32(np.uint32((vertex * 0x9E3779B9 + g * 0x85EBCA6B) & 0xFFFFFFFF))
            seed = hash32(self.pixel ^ salt)
            index = nested_uniform_scramble(self.sample, seed)
            x = nested_uniform_scramble(sobol(index, lane), hash32(seed + np.uint32(lane + 1)))
        return (x >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)


def dimension(vertex: int, slot: int) -> int:
    """The random dimension of decision ``slot`` at path vertex ``vertex``."""
    return vertex * DIMENSIONS_PER_VERTEX + slot


# Slots of the random dimensions used at every path vertex.
SLOT_PIXEL = 0  # 2: pixel position (vertex 0 only)
SLOT_LOBE = 2  # 1: BSDF lobe choice
SLOT_BSDF = 3  # 2: BSDF direction
SLOT_RR = 5  # 1: Russian roulette
SLOT_LENS = 6  # 2: the point on the camera's aperture (vertex 0 only; depth of field)
SLOT_AO = 16  # 2 per AO ray (vertex 0 only; at most 1024 rays)
SLOT_LIGHTS = 4096  # 3 per light sample: the analytic lights first, then the emitters


# --------------------------------------------------------------------------- #
# Frames and warps
# --------------------------------------------------------------------------- #


FRAME_BRANCH_TOLERANCE = 1e-12
"""Normals with ``z >= -FRAME_BRANCH_TOLERANCE`` use the ``+z`` branch of :func:`frame`."""


def frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents ``t``, ``b`` with ``(t, b, n)`` right-handed and orthonormal.

    The construction has two branches, chosen by the sign of ``n.z``. A normal
    that is horizontal up to rounding must not switch branches: a wall rotated
    by 90 degrees has ``n.z = 1 - 2 sin^2(45 deg)``, about +2e-16 or -2e-16
    depending on how the platform's math library rounds ``sin(pi/4)``, and the
    other branch turns the frame -- so the same random numbers would sample
    other directions on other platforms. Values down to
    :data:`FRAME_BRANCH_TOLERANCE` below zero take the ``+z`` branch, which
    stays accurate there (it is singular only at ``n.z = -1``).
    """
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    sign = np.where(nz >= -FRAME_BRANCH_TOLERANCE, 1.0, -1.0)
    a = -1.0 / (sign + nz)
    b = nx * ny * a
    t = np.stack([1.0 + sign * nx * nx * a, sign * b, -sign * nx], axis=1)
    bt = np.stack([b, sign + ny * ny * a, -ny], axis=1)
    return t, bt


def to_local(v: np.ndarray, t: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.stack([(v * t).sum(axis=1), (v * b).sum(axis=1), (v * n).sum(axis=1)], axis=1)


def to_world(v: np.ndarray, t: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    return t * v[:, 0:1] + b * v[:, 1:2] + n * v[:, 2:3]


def cosine_hemisphere(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Local directions with density ``z / pi`` (z >= 0)."""
    r = np.sqrt(u1)
    phi = 2.0 * math.pi * u2
    z = np.sqrt(np.maximum(0.0, 1.0 - u1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def uniform_cone(u1: np.ndarray, u2: np.ndarray, cos_max: np.ndarray) -> np.ndarray:
    """Local directions uniform in the cone ``z >= cos_max`` around +z."""
    z = 1.0 - u1 * (1.0 - cos_max)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def uniform_triangle(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Barycentrics uniformly distributed over a triangle (by area)."""
    s = np.sqrt(u1)
    b0 = 1.0 - s
    b1 = u2 * s
    return np.stack([b0, b1, 1.0 - b0 - b1], axis=1)


def ggx_visible_normal(v: np.ndarray, alpha: np.ndarray, u1: np.ndarray, u2: np.ndarray):
    """Microfacet normals sampled from the GGX distribution of normals visible from ``v``.

    ``v``: local unit view directions with ``z > 0``; ``alpha``: GGX roughness
    (``roughness^2``). Heitz 2018, listing 1.
    """
    a = alpha
    vh = np.stack([a * v[:, 0], a * v[:, 1], v[:, 2]], axis=1)
    vh /= np.linalg.norm(vh, axis=1, keepdims=True)
    lensq = vh[:, 0] * vh[:, 0] + vh[:, 1] * vh[:, 1]
    inv = np.where(lensq > 0.0, 1.0 / np.sqrt(np.where(lensq > 0.0, lensq, 1.0)), 0.0)
    t1 = np.where(
        (lensq > 0.0)[:, None],
        np.stack([-vh[:, 1] * inv, vh[:, 0] * inv, np.zeros_like(inv)], axis=1),
        np.array([1.0, 0.0, 0.0]),
    )
    t2 = np.cross(vh, t1)
    r = np.sqrt(u1)
    phi = 2.0 * math.pi * u2
    p1 = r * np.cos(phi)
    p2 = r * np.sin(phi)
    s = 0.5 * (1.0 + vh[:, 2])
    p2 = (1.0 - s) * np.sqrt(np.maximum(0.0, 1.0 - p1 * p1)) + s * p2
    nh = (
        t1 * p1[:, None]
        + t2 * p2[:, None]
        + vh * np.sqrt(np.maximum(0.0, 1.0 - p1 * p1 - p2 * p2))[:, None]
    )
    ne = np.stack([a * nh[:, 0], a * nh[:, 1], np.maximum(0.0, nh[:, 2])], axis=1)
    return ne / np.maximum(np.linalg.norm(ne, axis=1, keepdims=True), 1e-300)


def power_heuristic(pdf_a: np.ndarray, pdf_b: np.ndarray) -> np.ndarray:
    """Veach's power heuristic (beta = 2): the weight of strategy ``a``."""
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    total = a2 + b2
    return np.where(total > 0.0, a2 / np.where(total > 0.0, total, 1.0), 0.0)
