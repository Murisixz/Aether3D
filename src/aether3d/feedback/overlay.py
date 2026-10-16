"""Images derived from an observation: annotated markers, object-id colors, depth, masks, crops.

All drawing is Aether3D's own, on the observation's 8-bit pixels (no fonts, no
dependencies): numbered markers use a fixed 5x7 bitmap of the digits, so an annotated
image is reproducible bit for bit. The annotated image is a *separate* image: the clean
render is never altered, and a vision model can be shown both.
"""

from __future__ import annotations

import numpy as np

# 5x7 bitmaps of the digits 0-9 (rows top to bottom, 5 bits each, most significant left).
_DIGITS = {
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
}
_GLYPHS = {
    ch: np.array([[(row >> (4 - b)) & 1 for b in range(5)] for row in rows], dtype=bool)
    for ch, rows in _DIGITS.items()
}

# Marker colors: saturated, distinct, readable with black or white text.
PALETTE = np.array(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
    ],
    dtype=np.uint8,
)


def _text_color(color: np.ndarray) -> np.ndarray:
    luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return np.array((0, 0, 0) if luma > 140 else (255, 255, 255), dtype=np.uint8)


def _rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, width: int = 1) -> None:
    """Outline of the pixel rectangle [x0, x1) x [y0, y1), clipped to the image."""
    h, w = img.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x0 >= x1 or y0 >= y1:
        return
    for k in range(width):
        if y0 + k < y1:
            img[y0 + k, x0:x1] = color
        if y1 - 1 - k >= y0:
            img[y1 - 1 - k, x0:x1] = color
        if x0 + k < x1:
            img[y0:y1, x0 + k] = color
        if x1 - 1 - k >= x0:
            img[y0:y1, x1 - 1 - k] = color


def _label(img: np.ndarray, text: str, x: int, y: int, color: np.ndarray, scale: int) -> None:
    """A filled tag with ``text`` (digits) whose top-left corner is near ``(x, y)``."""
    h, w = img.shape[:2]
    gw, gh = 5 * scale, 7 * scale
    pad = scale
    tw = len(text) * (gw + scale) - scale + 2 * pad
    th = gh + 2 * pad
    x = int(min(max(x, 0), max(w - tw, 0)))
    y = int(min(max(y, 0), max(h - th, 0)))
    img[y : y + th, x : x + tw] = color
    ink = _text_color(color)
    cx = x + pad
    for ch in text:
        glyph = _GLYPHS.get(ch)
        if glyph is not None:
            big = np.repeat(np.repeat(glyph, scale, axis=0), scale, axis=1)
            region = img[y + pad : y + pad + gh, cx : cx + gw]
            region[big[: region.shape[0], : region.shape[1]]] = ink
        cx += gw + scale


def annotate(
    image: np.ndarray,
    index: np.ndarray,
    markers: list[tuple[int, int, tuple[int, int, int, int]]],
) -> np.ndarray:
    """A copy of ``image`` with each marker drawn: ``(number, instance, (x0, y0, x1, y1))``.

    The instance's visible pixels get a thin outline (their boundary with other objects or
    the background) in the marker's color, its pixel bounds a rectangle, and a numbered tag
    sits at the rectangle's top-left corner.
    """
    out = image.copy()
    h, w = image.shape[:2]
    scale = 2 if min(w, h) >= 300 else 1
    for number, instance, (x0, y0, x1, y1) in markers:
        color = PALETTE[(number - 1) % len(PALETTE)]
        mask = index == instance
        edge = mask.copy()
        edge[1:-1, 1:-1] &= ~(mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
        out[edge] = color
        _rect(out, x0, y0, x1, y1, color, width=1)
        _label(out, str(number), x0, y0 - (7 * scale + 2 * scale), color, scale)
    return out


def id_colors(index: np.ndarray) -> np.ndarray:
    """The object-index buffer in false colors (background black): a segmentation image."""
    k = index.astype(np.int64)
    # A fixed integer hash per instance: distinct, deterministic colors.
    hashed = (k * 2654435761 + 0x9E3779B9) & 0xFFFFFF
    rgb = np.stack([(hashed >> 16) & 255, (hashed >> 8) & 255, hashed & 255], axis=-1)
    rgb = (rgb // 2 + 96).astype(np.uint8)  # never black: background stays distinguishable
    rgb[index < 0] = 0
    return rgb


def depth_gray(depth: np.ndarray) -> np.ndarray:
    """View depth as gray (near white, far dark, background black), normalized to the
    finite range of the image."""
    finite = np.isfinite(depth)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not finite.any():
        return out
    lo, hi = float(depth[finite].min()), float(depth[finite].max())
    span = hi - lo if hi > lo else 1.0
    level = 255.0 - 200.0 * (depth[finite] - lo) / span
    out[finite] = np.clip(np.round(level), 0, 255).astype(np.uint8)[:, None]
    return out


def mask_image(index: np.ndarray, instance: int) -> np.ndarray:
    """One instance's visible pixels: white on black."""
    out = np.zeros(index.shape + (3,), dtype=np.uint8)
    out[index == instance] = 255
    return out
