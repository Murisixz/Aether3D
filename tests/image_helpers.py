"""A deliberately simple, independent PNG writer for decoder tests.

It writes any color type, bit depth, filter choice and Adam7 interlacing, and
can corrupt chosen parts, so the decoder in ``aether3d.materials.images`` can
be checked against known pixels (and Pillow) on files its own encoder never
produces.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"
ADAM7 = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
SAMPLES = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def chunk(kind: bytes, data: bytes, crc: int | None = None) -> bytes:
    if crc is None:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def pack_row(samples: np.ndarray, depth: int) -> bytes:
    """One scanline of sample values (already flattened per row) to bytes."""
    flat = np.asarray(samples).reshape(-1)
    if depth == 16:
        return flat.astype(">u2").tobytes()
    if depth == 8:
        return flat.astype(np.uint8).tobytes()
    bits = np.zeros(flat.size * depth, dtype=np.uint8)
    for i, v in enumerate(flat.tolist()):
        for b in range(depth):
            bits[i * depth + b] = (v >> (depth - 1 - b)) & 1
    return np.packbits(bits).tobytes()


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_rows(rows: list[bytes], bpp: int, filters: list[int]) -> bytes:
    out = bytearray()
    prev = bytes(len(rows[0])) if rows else b""
    for row, kind in zip(rows, filters, strict=True):
        out.append(kind)
        for i, x in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            pred = (0, a, b, (a + b) // 2, _paeth(a, b, c))[kind]
            out.append((x - pred) & 0xFF)
        prev = row
    return bytes(out)


def make_png(
    image: np.ndarray,
    depth: int,
    ctype: int,
    *,
    interlace: bool = False,
    palette: np.ndarray | None = None,
    trns: bytes | None = None,
    filters: str = "cycle",
    seed: int = 0,
) -> bytes:
    """Encode ``image`` (``(H, W, samples)`` integer samples) as a PNG."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    h, w, samples = image.shape
    assert samples == SAMPLES[ctype]
    bpp = max(1, depth * samples // 8)
    rng = np.random.default_rng(seed)
    passes = ADAM7 if interlace else ((0, 0, 1, 1),)
    raw = bytearray()
    for x0, y0, dx, dy in passes:
        sub = image[y0::dy, x0::dx]
        if sub.size == 0:
            continue
        rows = [pack_row(r, depth) for r in sub]
        if filters == "cycle":
            kinds = [i % 5 for i in range(len(rows))]
        elif filters == "random":
            kinds = rng.integers(0, 5, len(rows)).tolist()
        else:
            kinds = [int(filters)] * len(rows)
        raw += filter_rows(rows, bpp, kinds)
    out = SIGNATURE + chunk(
        b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, int(interlace))
    )
    if palette is not None:
        out += chunk(b"PLTE", np.asarray(palette, dtype=np.uint8).tobytes())
    if trns is not None:
        out += chunk(b"tRNS", trns)
    out += chunk(b"IDAT", zlib.compress(bytes(raw)))
    out += chunk(b"IEND", b"")
    return out


def chunks_of(data: bytes) -> list[tuple[bytes, bytes]]:
    pos, out = 8, []
    while pos < len(data):
        (n,) = struct.unpack_from(">I", data, pos)
        out.append((data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]))
        pos += 12 + n
    return out


def assemble(parts: list[tuple[bytes, bytes]]) -> bytes:
    return SIGNATURE + b"".join(chunk(k, d) for k, d in parts)
