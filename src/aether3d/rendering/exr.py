"""OpenEXR files (0.20.0): scene-linear masters of renders and their passes.

:func:`write_exr` writes a scanline OpenEXR 2 file (the format's published specification,
"OpenEXR File Layout"): any named channels (``R``, ``G``, ``B``, ``A``, ``Z``,
``normal.X`` ...), each ``half`` (16-bit float) or ``float`` (32-bit), uncompressed or
``zip`` (deflate over blocks of 16 scanlines after the specification's byte reordering
and delta predictor; a block that would not shrink is stored raw, as the format allows).
Channels are stored in name order, as readers expect. :func:`read_exr` reads back what it
writes (and other uncompressed or zip scanline files) -- for tests and tools; it refuses
tiled, deep and multi-part files and other compressions with ``EXR_UNSUPPORTED``.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import numpy as np

from .._fileio import write_atomic
from ..errors import RenderError

MAGIC = 20000630
COMPRESSIONS = {"none": 0, "zip": 3}
PIXEL_TYPES = {"half": 1, "float": 2}
BLOCK = {"none": 1, "zip": 16}


def _attr(name: str, kind: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(value)) + value


def _predict(raw: bytes) -> bytes:
    """The specification's zip preparation: interleave halves, then byte deltas."""
    a = np.frombuffer(raw, dtype=np.uint8)
    n = a.size
    t = np.empty(n, dtype=np.uint8)
    t[: (n + 1) // 2] = a[0::2]
    t[(n + 1) // 2 :] = a[1::2]
    d = t.astype(np.int16)
    d[1:] = (d[1:] - d[:-1] + 128 + 256) & 0xFF
    return d.astype(np.uint8).tobytes()


def _unpredict(data: bytes) -> bytes:
    d = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    t = (np.cumsum(np.concatenate([[d[0]], d[1:] - 128])) & 0xFF).astype(np.uint8)
    n = t.size
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = t[: (n + 1) // 2]
    out[1::2] = t[(n + 1) // 2 :]
    return out.tobytes()


def write_exr(
    path: str | os.PathLike[str],
    channels: dict[str, np.ndarray],
    *,
    pixel_type: str = "half",
    compression: str = "zip",
) -> Path:
    """Write ``channels`` (name -> ``(H, W)`` array, all the same size) atomically."""
    if compression not in COMPRESSIONS or pixel_type not in PIXEL_TYPES:
        raise RenderError("INVALID_EXR", "compression is none or zip, pixel_type half or float")
    if not channels:
        raise RenderError("INVALID_EXR", "an EXR file needs at least one channel")
    names = sorted(channels)
    planes = [np.asarray(channels[n], dtype=np.float32) for n in names]
    h, w = planes[0].shape
    if any(p.shape != (h, w) for p in planes) or h < 1 or w < 1:
        raise RenderError("INVALID_EXR", "channels must be non-empty 2D arrays of one size")
    for n in names:
        if not n or len(n) > 255 or "\0" in n:
            raise RenderError("INVALID_EXR", f"bad channel name {n!r}")
    dtype = np.dtype("<f2") if pixel_type == "half" else np.dtype("<f4")
    chlist = b"".join(
        n.encode() + b"\0" + struct.pack("<iBBBBii", PIXEL_TYPES[pixel_type], 0, 0, 0, 0, 1, 1)
        for n in names
    ) + b"\0"  # fmt: skip
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = struct.pack("<ii", MAGIC, 2) + b"".join([
        _attr("channels", "chlist", chlist),
        _attr("compression", "compression", bytes([COMPRESSIONS[compression]])),
        _attr("dataWindow", "box2i", box),
        _attr("displayWindow", "box2i", box),
        _attr("lineOrder", "lineOrder", bytes([0])),
        _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]) + b"\0"  # fmt: skip
    rows = BLOCK[compression]
    blocks = []
    data = np.stack([p.astype(dtype) for p in planes], axis=1)  # (H, C, W)
    for y0 in range(0, h, rows):
        raw = data[y0 : y0 + rows].tobytes()  # per scanline: each channel's row in turn
        if compression == "zip":
            packed = zlib.compress(_predict(raw), 6)
            payload = packed if len(packed) < len(raw) else raw
        else:
            payload = raw
        blocks.append(struct.pack("<ii", y0, len(payload)) + payload)
    offsets = []
    pos = len(header) + 8 * len(blocks)
    for b in blocks:
        offsets.append(pos)
        pos += len(b)
    body = header + struct.pack(f"<{len(offsets)}Q", *offsets) + b"".join(blocks)
    return write_atomic(path, body)


def read_exr(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """The channels of a scanline EXR file (``none`` or ``zip``), as float32 planes."""
    data = Path(path).read_bytes()
    if len(data) < 8 or struct.unpack_from("<i", data, 0)[0] != MAGIC:
        raise RenderError("EXR_INVALID", "not an OpenEXR file")
    version = struct.unpack_from("<i", data, 4)[0]
    if version & 0xFF != 2 or version & 0x1A00:  # (tiled, deep, multi-part)
        raise RenderError("EXR_UNSUPPORTED", "only single-part scanline EXR files are read")
    pos = 8
    attrs = {}
    while data[pos] != 0:
        end = data.index(b"\0", pos)
        name = data[pos:end].decode()
        end2 = data.index(b"\0", end + 1)
        kind = data[end + 1 : end2].decode()
        size = struct.unpack_from("<i", data, end2 + 1)[0]
        attrs[name] = (kind, data[end2 + 5 : end2 + 5 + size])
        pos = end2 + 5 + size
    pos += 1
    comp = attrs["compression"][1][0]
    if comp not in (0, 2, 3):
        raise RenderError("EXR_UNSUPPORTED", f"compression {comp} is not read here")
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    raw = attrs["channels"][1]
    names, types = [], []
    k = 0
    while raw[k] != 0:
        end = raw.index(b"\0", k)
        names.append(raw[k:end].decode())
        types.append(struct.unpack_from("<i", raw, end + 1)[0])
        k = end + 1 + 16
    sizes = [2 if t == 1 else 4 for t in types]
    rows = 16 if comp == 3 else 1
    n_blocks = (h + rows - 1) // rows
    offsets = struct.unpack_from(f"<{n_blocks}Q", data, pos)
    out = {n: np.zeros((h, w), np.float32) for n in names}
    line = w * sum(sizes)
    for off in offsets:
        y, size = struct.unpack_from("<ii", data, off)
        payload = data[off + 8 : off + 8 + size]
        count = min(rows, y1 + 1 - y)
        if comp != 0 and size < line * count:
            payload = _unpredict(zlib.decompress(payload))
        for r in range(count):
            p = r * line
            for n, t, sz in zip(names, types, sizes, strict=True):
                dt = np.dtype("<f2") if t == 1 else np.dtype("<f4")
                out[n][y - y0 + r] = np.frombuffer(payload, dtype=dt, count=w, offset=p)
                p += w * sz
    return out
