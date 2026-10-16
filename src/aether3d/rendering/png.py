"""A small, dependency-free PNG encoder (W3C PNG specification, 3rd edition).

Writes 8-bit grayscale, RGB or RGBA images, non-interlaced, with the
per-scanline adaptive filter heuristic used by libpng (all five filter types
are tried; the row with the smallest sum of absolute signed bytes wins) and
zlib compression from the Python standard library. Color images carry an
``sRGB`` chunk, because Aether3D renders are sRGB-encoded. Output contains no
timestamps, so the same pixels always produce the same bytes with a given
zlib build.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import numpy as np

from .. import _native
from .._fileio import write_atomic
from ..errors import RenderError

SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOR_TYPES = {1: 0, 3: 2, 4: 6}  # channels -> PNG color type
_ROW_BLOCK = 256


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _paeth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


_MAGNITUDE = np.minimum(np.arange(256), 256 - np.arange(256)).astype(np.uint8)
"""|signed byte| of each filtered byte value: the filter choice's cost per byte."""


def _filter_rows(rows: np.ndarray, bpp: int) -> np.ndarray:
    """Filtered scanlines (each prefixed with its filter type byte).

    Every row gets the filter with the smallest sum of absolute signed bytes (ties: the
    lower filter type). Computed in 16-bit integers; filtered bytes wrap to 8 bits.
    """
    height, stride = rows.shape
    out = np.empty((height, stride + 1), dtype=np.uint8)
    native = _native.module()
    if native is not None:  # the same choice and bytes (aether3d-native)
        native.png_filter(np.ascontiguousarray(rows), height, stride, bpp, out)
        return out
    previous = np.zeros(stride, dtype=np.int16)
    for r0 in range(0, height, _ROW_BLOCK):
        raw = rows[r0 : r0 + _ROW_BLOCK]
        x = raw.astype(np.int16)
        n = x.shape[0]
        up = np.empty_like(x)
        up[0] = previous
        up[1:] = x[:-1]
        left = np.zeros_like(x)
        left[:, bpp:] = x[:, :-bpp]
        upleft = np.zeros_like(x)
        upleft[:, bpp:] = up[:, :-bpp]
        paeth = _paeth(left, up, upleft)
        candidates = np.empty((5, n, stride), dtype=np.uint8)
        candidates[0] = raw
        candidates[1] = x - left  # wraps to 8 bits (mod 256)
        candidates[2] = x - up
        candidates[3] = x - ((left + up) >> 1)
        candidates[4] = x - paeth
        cost = _MAGNITUDE[candidates].sum(axis=2, dtype=np.int64)
        best = np.argmin(cost, axis=0)  # ties prefer the lower filter type
        out[r0 : r0 + n, 0] = best
        out[r0 : r0 + n, 1:] = candidates[best, np.arange(n)]
        previous = x[-1]
    return out


def encode_png(
    image: np.ndarray, *, text: dict[str, str] | None = None, compression: int = 6
) -> bytes:
    """Encode an ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)`` ``uint8`` array as PNG bytes."""
    if isinstance(compression, bool) or compression not in range(-1, 10):
        raise RenderError(
            "INVALID_SETTINGS",
            f"PNG compression must be an integer from 0 to 9 (or -1), got {compression!r}",
            "compression",
        )
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise RenderError("INVALID_IMAGE", f"PNG pixels must be uint8, got {arr.dtype}")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in _COLOR_TYPES or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise RenderError(
            "INVALID_IMAGE", f"expected an (H, W), (H, W, 3) or (H, W, 4) image, got {arr.shape}"
        )
    height, width, channels = arr.shape
    if width > 2**31 - 1 or height > 2**31 - 1:
        raise RenderError("INVALID_IMAGE", "PNG dimensions must be below 2^31")
    rows = np.ascontiguousarray(arr).reshape(height, width * channels)
    filtered = _filter_rows(rows, channels)
    header = struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPES[channels], 0, 0, 0)
    chunks = [_chunk(b"IHDR", header)]
    if channels >= 3:
        chunks.append(_chunk(b"sRGB", b"\x00"))  # perceptual rendering intent
    for key, value in (text or {}).items():
        keyword = key.encode("latin-1")
        if not 1 <= len(keyword) <= 79 or b"\x00" in keyword:
            raise RenderError("INVALID_IMAGE", f"invalid PNG text keyword {key!r}")
        chunks.append(_chunk(b"tEXt", keyword + b"\x00" + value.encode("latin-1", "replace")))
    chunks.append(_chunk(b"IDAT", zlib.compress(filtered.tobytes(), compression)))
    chunks.append(_chunk(b"IEND", b""))
    return SIGNATURE + b"".join(chunks)


def write_png(path: str | os.PathLike[str], image: np.ndarray, **options) -> Path:
    """Encode ``image`` and write it atomically (a failure never leaves a partial file)."""
    data = encode_png(image, **options)
    return write_atomic(path, data)
