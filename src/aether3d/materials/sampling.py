"""Texture sampling shared by the CPU renderer and mirrored exactly by the GPU shader.

An image texture is sampled from a *mip chain* of 8-bit RGBA levels. The
chain is built once per image and *decoding mode*:

* ``"srgb"`` -- color data (base color, emissive): RGB are sRGB-encoded and
  are converted to linear light texel by texel (through a 256-entry table)
  *before* filtering, as GPU sRGB textures do; mip levels are averaged in
  linear light and re-encoded.
* ``"linear"`` -- data (metallic, roughness, occlusion, opacity): values are
  used as stored.
* ``"normal"`` -- tangent-space normal maps: stored like ``"linear"``; mip
  levels average the decoded unit vectors and renormalize them.

Alpha is always linear coverage.

Coordinates follow Aether3D's UV convention: ``(0, 0)`` is the bottom-left of
the image and ``v`` grows upward (glTF's top-left origin is flipped on
import). For a level of ``W x H`` texels, ``x = u W`` and ``y = (1 - v) H``
in texel units (texel centers at half-integers).

* ``filter="nearest"`` -- the texel containing the point; with mipmaps, the
  nearest level.
* ``filter="linear"`` -- bilinear between the four nearest texel centers;
  with mipmaps, linear between the two nearest levels (trilinear).
* Mip level selection uses the screen-space derivatives of the coordinates:
  ``lambda = log2(max(|d(xy)/dx|, |d(xy)/dy|))`` measured in texels of the
  first level (the standard isotropic rule).
* Wrap modes per axis: ``"repeat"``, ``"clamp"`` (to the edge texel) and
  ``"mirror"`` (mirrored repeat).

The GPU stores the same levels in one storage buffer and runs the same
arithmetic in ``wgsl/material.wgsl``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..rendering.color import linear_to_srgb, quantize, srgb_to_linear

MODES = ("srgb", "linear", "normal")
WRAP_MODES = ("repeat", "clamp", "mirror")
FILTERS = ("nearest", "linear")

SRGB_TABLE = srgb_to_linear(np.arange(256) / 255.0)
"""``SRGB_TABLE[c]``: the linear value of 8-bit sRGB code ``c`` (float64)."""
LINEAR_TABLE = np.arange(256) / 255.0


@dataclass(frozen=True)
class MipChain:
    """The 8-bit RGBA levels of one image in one decoding mode (level 0 first)."""

    levels: tuple[np.ndarray, ...]  # each (H_k, W_k, 4) uint8, read-only
    mode: str

    @property
    def width(self) -> int:
        return int(self.levels[0].shape[1])

    @property
    def height(self) -> int:
        return int(self.levels[0].shape[0])

    @property
    def nbytes(self) -> int:
        return int(sum(level.nbytes for level in self.levels))


def level_count(width: int, height: int) -> int:
    """Levels down to 1 x 1: ``floor(log2(max(width, height))) + 1``."""
    return int(max(width, height)).bit_length()


def _downsample(level: np.ndarray, mode: str) -> np.ndarray:
    """One 2x2 box-filter step (odd sizes drop their last row/column, sizes never reach 0)."""
    h, w = level.shape[:2]
    nh, nw = max(1, h // 2), max(1, w // 2)
    ys = np.minimum(np.arange(2 * nh), h - 1)
    xs = np.minimum(np.arange(2 * nw), w - 1)
    block = level[ys][:, xs]  # (2nh, 2nw, 4) uint8
    if mode == "srgb":
        rgb = SRGB_TABLE[block[..., :3]]
    elif mode == "normal":
        rgb = block[..., :3] * (2.0 / 255.0) - 1.0
    else:
        rgb = block[..., :3] / 255.0
    alpha = block[..., 3] / 255.0
    rgb = rgb.reshape(nh, 2, nw, 2, 3).mean(axis=(1, 3))
    alpha = alpha.reshape(nh, 2, nw, 2).mean(axis=(1, 3))
    out = np.empty((nh, nw, 4), dtype=np.uint8)
    if mode == "srgb":
        out[..., :3] = quantize(linear_to_srgb(rgb))
    elif mode == "normal":
        length = np.sqrt((rgb * rgb).sum(axis=-1, keepdims=True))
        unit = np.divide(rgb, length, out=np.zeros_like(rgb), where=length > 1e-12)
        unit[..., 2] = np.where(length[..., 0] > 1e-12, unit[..., 2], 1.0)  # flat if cancelled
        out[..., :3] = quantize(unit * 0.5 + 0.5)
    else:
        out[..., :3] = quantize(rgb)
    out[..., 3] = quantize(alpha)
    return out


def build_mip_chain(pixels: np.ndarray, mode: str, mipmaps: bool = True) -> MipChain:
    """The mip chain of 8-bit RGBA ``pixels`` (all levels, or only the first)."""
    if mode not in MODES:
        raise ValueError(f"unknown texture mode {mode!r}")
    base = np.ascontiguousarray(pixels, dtype=np.uint8)
    levels = [base]
    if mipmaps:
        for _ in range(level_count(base.shape[1], base.shape[0]) - 1):
            levels.append(_downsample(levels[-1], mode))
    for level in levels:
        level.flags.writeable = False
    return MipChain(tuple(levels), mode)


# --------------------------------------------------------------------------- #
# Sampling (vectorized; float64 on the CPU)
# --------------------------------------------------------------------------- #


def wrap_index(i: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Integer texel indices wrapped into ``[0, size)``."""
    if mode == "clamp":
        return np.clip(i, 0, size - 1)
    if mode == "mirror":
        m = np.mod(i, 2 * size)
        return np.where(m < size, m, 2 * size - 1 - m)
    return np.mod(i, size)


def _decode(texels: np.ndarray, mode: str) -> np.ndarray:
    """``(N, 4)`` uint8 texels -> float64 values (sRGB RGB decoded to linear in ``"srgb"`` mode)."""
    out = np.empty(texels.shape, dtype=np.float64)
    table = SRGB_TABLE if mode == "srgb" else LINEAR_TABLE
    out[:, :3] = table[texels[:, :3]]
    out[:, 3] = LINEAR_TABLE[texels[:, 3]]
    return out


def _fetch(level: np.ndarray, x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    return _decode(level[y, x], mode)


def _sample_level(
    level: np.ndarray, u: np.ndarray, v: np.ndarray, wrap: tuple[str, str], linear: bool, mode: str
) -> np.ndarray:
    h, w = level.shape[:2]
    tx = u * w
    ty = (1.0 - v) * h
    if not linear:
        x = wrap_index(np.floor(tx).astype(np.int64), w, wrap[0])
        y = wrap_index(np.floor(ty).astype(np.int64), h, wrap[1])
        return _fetch(level, x, y, mode)
    fx, fy = tx - 0.5, ty - 0.5
    x0f, y0f = np.floor(fx), np.floor(fy)
    ax, ay = (fx - x0f)[:, None], (fy - y0f)[:, None]
    x0, y0 = x0f.astype(np.int64), y0f.astype(np.int64)
    xa, xb = wrap_index(x0, w, wrap[0]), wrap_index(x0 + 1, w, wrap[0])
    ya, yb = wrap_index(y0, h, wrap[1]), wrap_index(y0 + 1, h, wrap[1])
    top = _fetch(level, xa, ya, mode) * (1.0 - ax) + _fetch(level, xb, ya, mode) * ax
    bottom = _fetch(level, xa, yb, mode) * (1.0 - ax) + _fetch(level, xb, yb, mode) * ax
    return top * (1.0 - ay) + bottom * ay


def lod(
    chain: MipChain, du_dx: np.ndarray, dv_dx: np.ndarray, du_dy: np.ndarray, dv_dy: np.ndarray
) -> np.ndarray:
    """``lambda``: log2 of the larger screen-space footprint, in first-level texels."""
    w, h = chain.width, chain.height
    ax = np.hypot(du_dx * w, dv_dx * h)
    ay = np.hypot(du_dy * w, dv_dy * h)
    rho = np.maximum(ax, ay)
    with np.errstate(divide="ignore"):
        return np.log2(np.maximum(rho, 1e-30))


def sample(
    chain: MipChain,
    u: np.ndarray,
    v: np.ndarray,
    *,
    wrap: tuple[str, str] = ("repeat", "repeat"),
    filter: str = "linear",
    lam: np.ndarray | None = None,
) -> np.ndarray:
    """Sample ``chain`` at ``(u, v)``: ``(N, 4)`` float64 (RGB linear for ``"srgb"`` chains).

    ``lam`` is the level-of-detail ``lambda`` (see :func:`lod`); ``None`` or a
    single-level chain samples the first level only.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    linear = filter == "linear"
    levels = chain.levels
    if lam is None or len(levels) == 1:
        return _sample_level(levels[0], u, v, wrap, linear, chain.mode)
    top = len(levels) - 1
    lam = np.clip(np.nan_to_num(np.asarray(lam, dtype=np.float64), nan=0.0), 0.0, float(top))
    out = np.empty((u.shape[0], 4), dtype=np.float64)
    if not linear:
        chosen = np.minimum(np.floor(lam + 0.5).astype(np.int64), top)
        for k in np.unique(chosen):
            sel = chosen == k
            out[sel] = _sample_level(levels[k], u[sel], v[sel], wrap, False, chain.mode)
        return out
    k0 = np.floor(lam).astype(np.int64)
    k1 = np.minimum(k0 + 1, top)
    t = (lam - k0)[:, None]
    for k in np.unique(k0):
        sel = k0 == k
        a = _sample_level(levels[k], u[sel], v[sel], wrap, True, chain.mode)
        upper = int(k1[sel][0])
        if upper == k:
            out[sel] = a
            continue
        b = _sample_level(levels[upper], u[sel], v[sel], wrap, True, chain.mode)
        out[sel] = a * (1.0 - t[sel]) + b * t[sel]
    return out


def chain_level_offsets(chain: MipChain) -> list[tuple[int, int, int]]:
    """``(texel offset, width, height)`` of each level when the levels are packed end to end."""
    out, offset = [], 0
    for level in chain.levels:
        h, w = level.shape[:2]
        out.append((offset, w, h))
        offset += w * h
    return out


def mip_bytes(width: int, height: int, mipmaps: bool = True) -> int:
    """Bytes of an 8-bit RGBA mip chain (all levels, or only the first)."""
    total, w, h = 0, width, height
    for _ in range(level_count(width, height) if mipmaps else 1):
        total += 4 * w * h
        w, h = max(1, w // 2), max(1, h // 2)
    return total


def log2_ceil(value: float) -> int:
    return max(0, math.ceil(math.log2(max(value, 1.0))))
