"""Color values and the linear-light / display-sRGB conversion policy.

Policy (see ``docs/RENDERING.md``):

* Every color a user specifies -- object colors, light colors, the background
  and the ambient color -- is a **display (sRGB-encoded) color**, written as a
  hex string (``"#4a7c59"``, ``"#4a7"``) or as three numbers in ``[0, 1]``.
  That is what color pickers and hex codes mean, so ``"#808080"`` is the
  familiar mid gray.
* The renderer converts those colors to **linear light** with the exact
  IEC 61966-2-1 sRGB transfer function before any lighting, so light adds up
  physically sensibly; antialiasing and exposure also operate in linear light.
* The final linear image is clamped to ``[0, 1]``, encoded with the sRGB
  transfer function and quantized to 8 bits (round to nearest, ties to even).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import RenderError

Color = tuple[float, float, float]
"""An sRGB-encoded color with components in ``[0, 1]``."""


def srgb_to_linear(values: Any) -> np.ndarray:
    """Exact sRGB electro-optical transfer function (IEC 61966-2-1), elementwise."""
    c = np.asarray(values, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def linear_to_srgb(values: Any) -> np.ndarray:
    """Exact inverse of :func:`srgb_to_linear` (inputs are clamped to ``[0, 1]`` first)."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def quantize(srgb: Any) -> np.ndarray:
    """sRGB values in ``[0, 1]`` to 8-bit codes (round to nearest, ties to even)."""
    return np.rint(np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)


def parse_color(value: Any, location: str = "color") -> Color:
    """Validate a display color: ``"#rrggbb"``, ``"#rgb"`` or three numbers in ``[0, 1]``.

    Returns the color as a tuple of floats (still sRGB-encoded). Raises
    :class:`RenderError` (``INVALID_COLOR``) otherwise.
    """
    if isinstance(value, str):
        text = value.strip()
        digits = text[1:] if text.startswith("#") else None
        if digits is not None and len(digits) in (3, 6):
            try:
                if len(digits) == 3:
                    parts = [int(ch * 2, 16) for ch in digits]
                else:
                    parts = [int(digits[i : i + 2], 16) for i in (0, 2, 4)]
            except ValueError:
                parts = None
            if parts is not None:
                r, g, b = (p / 255.0 for p in parts)
                return (r, g, b)
        raise RenderError(
            "INVALID_COLOR",
            f"expected a hex color like '#4a7c59' or '#4a7', got {value!r}",
            location,
        )
    if isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        if len(items) == 3 and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in items
        ):
            r, g, b = (float(v) for v in items)
            if all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in (r, g, b)):
                return (r + 0.0, g + 0.0, b + 0.0)
    raise RenderError(
        "INVALID_COLOR",
        f"expected three numbers in [0, 1] (sRGB) or a hex string, got {value!r}",
        location,
    )


def color_to_hex(color: Color) -> str:
    """``"#rrggbb"`` for an sRGB color (components rounded to 8 bits)."""
    r, g, b = (int(c) for c in quantize(color))
    return f"#{r:02x}{g:02x}{b:02x}"


def to_linear(color: Color) -> np.ndarray:
    """Linear-light RGB (float64 array of 3) of an sRGB color."""
    return srgb_to_linear(np.asarray(color, dtype=np.float64))
