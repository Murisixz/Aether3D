"""High-dynamic-range image files: Radiance RGBE (``.hdr``) and Portable Float Map (``.pfm``).

Both store linear light without clipping (``RenderResult.linear``: after
exposure, before tone mapping), so bright highlights, light sources and
skies keep their values for compositing or later tone mapping.

* ``.hdr`` -- Radiance's RGBE format (Ward 1991): a shared 8-bit exponent
  per pixel and 8-bit mantissas (about 1% relative precision), scanlines
  in the "new" run-length encoding (here: literal runs only, which every
  reader accepts).
* ``.pfm`` -- three float32 values per pixel, little-endian, rows stored
  bottom to top (the PFM convention), exact.
"""

from __future__ import annotations

import numpy as np


def encode_pfm(values: np.ndarray) -> bytes:
    """``(H, W, 3)`` RGB (``PF``) or ``(H, W)`` single-channel (``Pf``) Portable Float Map."""
    h, w = values.shape[:2]
    kind = "PF" if values.ndim == 3 else "Pf"
    header = f"{kind}\n{w} {h}\n-1.0\n".encode("ascii")
    data = np.ascontiguousarray(values[::-1].astype("<f4"))
    return header + data.tobytes()


def _rgbe(linear: np.ndarray) -> np.ndarray:
    rgb = np.maximum(np.nan_to_num(linear.astype(np.float64), nan=0.0, posinf=0.0), 0.0)
    peak = rgb.max(axis=-1)
    mantissa, exponent = np.frexp(peak)
    small = peak < 1e-32
    scale = np.where(small, 0.0, mantissa * 256.0 / np.where(small, 1.0, peak))
    out = np.zeros(rgb.shape[:-1] + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(np.floor(rgb * scale[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = np.where(small, 0, np.clip(exponent + 128, 0, 255)).astype(np.uint8)
    return out


def encode_hdr(linear: np.ndarray) -> bytes:
    """``(H, W, 3)`` linear RGB as a Radiance ``.hdr`` file."""
    h, w = linear.shape[:2]
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii")
    rgbe = _rgbe(linear)
    parts = [header]
    if not 8 <= w < 32768:  # run-length encoding needs 8..32767 pixels per scanline
        parts.append(rgbe.tobytes())
        return b"".join(parts)
    chunks = np.arange(0, w, 128)
    for row in range(h):
        line = [bytes((2, 2, w >> 8, w & 255))]
        for channel in range(4):
            values = rgbe[row, :, channel]
            for start in chunks:
                piece = values[start : start + 128]
                line.append(bytes((piece.size,)))
                line.append(piece.tobytes())
        parts.append(b"".join(line))
    return b"".join(parts)
