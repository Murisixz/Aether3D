"""Cheap diagnostic drawing for the spatial-query examples (Stage 10).

Marks and outlines are drawn straight into 8-bit RGB arrays with NumPy: crosses at
image points, line segments, and world-space boxes projected through a camera with
``scene.project``. A colored id map shows which object covers which pixel.
"""

from __future__ import annotations

import numpy as np

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
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ],
    dtype=np.uint8,
)
"""Distinct colors for object ids (they repeat after 20)."""
BACKGROUND = np.array((16, 16, 16), dtype=np.uint8)


def id_image(ids: np.ndarray) -> np.ndarray:
    """``(H, W)`` object indices (``-1``: nothing) as colors."""
    out = np.empty(ids.shape + (3,), dtype=np.uint8)
    out[...] = BACKGROUND
    hit = ids >= 0
    out[hit] = PALETTE[ids[hit] % len(PALETTE)]
    return out


def put(image: np.ndarray, x: float, y: float, color, radius: int = 0) -> None:
    """Paint the pixel under image point ``(x, y)`` (and its neighbors within ``radius``)."""
    h, w = image.shape[:2]
    col, row = int(np.floor(x)), int(np.floor(y))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if 0 <= row + dy < h and 0 <= col + dx < w:
                image[row + dy, col + dx] = color


def cross(image: np.ndarray, x: float, y: float, color, size: int = 6) -> None:
    for k in range(-size, size + 1):
        put(image, x + k, y, color)
        put(image, x, y + k, color)


def segment(image: np.ndarray, a, b, color) -> None:
    """A line between image points ``a`` and ``b`` (skipped if either is undefined)."""
    if a[0] is None or b[0] is None:
        return
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    if steps > 20000:  # a point far outside the image: not worth drawing
        return
    for t in np.linspace(0.0, 1.0, steps + 1):
        put(image, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, color)


BOX_EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]


def box(scene, image: np.ndarray, corners: np.ndarray, color, camera=None) -> None:
    """The 12 edges of a box given by its 8 corners (``x`` slowest, as ``AABB.corners()``),
    projected through ``camera`` (default: the active camera)."""
    h, w = image.shape[:2]
    pts = scene.project([tuple(c) for c in corners], camera=camera, width=w, height=h)
    xy = [(p.x, p.y) for p in pts]
    for i, j in BOX_EDGES:
        segment(image, xy[i], xy[j], color)
