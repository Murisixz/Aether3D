"""UV layout images: a mesh's UV map drawn in the unit square, for people and agents.

:func:`layout_image` rasterizes the UV triangles exactly (texturing's UV rasterizer, one
sample per pixel center) and colors them by ``mode``:

* ``"islands"`` -- a distinct color per island;
* ``"density"`` -- texel density relative to the area-weighted mean of the mesh (log2
  scale: blue half or less, green equal, red double or more);
* ``"stretch"`` -- the anisotropy ``s1 / s2`` of each triangle's map (green conformal, red
  3:1 or worse);
* ``"overlaps"`` -- light gray, with unintended overlaps red, flipped triangles orange
  and stacked islands (deliberate copies) yellow;
* ``"texture"`` -- an ``(h, w, 3|4)`` image (e.g. a painted texture's ``read()``) under
  the wireframe.

UV edges are drawn on top (island boundaries darker), v up (row 0 is v = 1). Pixels
outside every triangle show a checker background.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import TexturingError
from .islands import island_labels
from .report import _check_mesh, overlapping_triangles
from .topology import Topology

MODES = ("islands", "density", "stretch", "overlaps", "texture")
MAX_SIZE = 4096
MAX_EDGE_SAMPLES = 40_000_000


def _ramp(x: np.ndarray) -> np.ndarray:
    """Blue (0) - green (0.5) - red (1)."""
    x = np.clip(x, 0.0, 1.0)[:, None]
    blue = np.array([0.20, 0.35, 0.95])
    green = np.array([0.25, 0.80, 0.30])
    red = np.array([0.95, 0.25, 0.20])
    low = blue + (green - blue) * (x / 0.5)
    high = green + (red - green) * ((x - 0.5) / 0.5)
    return np.where(x < 0.5, low, high)


def _island_colors(count: int) -> np.ndarray:
    i = np.arange(count)
    hue = (i * 0.618033988749895) % 1.0
    sat = 0.45 + 0.2 * ((i * 7) % 3) / 2
    val = 0.95 - 0.15 * ((i * 5) % 2)
    h6 = hue * 6.0
    c = val * sat
    x = c * (1 - np.abs(h6 % 2 - 1))
    m = val - c
    k = np.floor(h6).astype(int) % 6
    rgb = np.zeros((count, 3))
    table = [(0, 1, 2), (1, 0, 2), (2, 0, 1), (2, 1, 0), (1, 2, 0), (0, 2, 1)]
    for j, (a, b, z) in enumerate(table):
        sel = k == j
        rgb[sel, a] = c[sel]
        rgb[sel, b] = x[sel]
        rgb[sel, z] = 0
    return rgb + m[:, None]


def layout_image(
    mesh: Any,
    size: int = 1024,
    *,
    mode: str = "islands",
    texture: Any = None,
    edges: bool = True,
) -> np.ndarray:
    """The UV layout of ``mesh`` as a ``(size, size, 3)`` uint8 sRGB image (see the module
    docstring)."""
    from ..texturing.raster import rasterize_uv

    mesh = _check_mesh(mesh)
    if isinstance(size, bool) or not isinstance(size, int) or not 16 <= size <= MAX_SIZE:
        raise TexturingError("INVALID_ARGUMENT", f"size must be in [16, {MAX_SIZE}]", "size")
    if mode not in MODES:
        raise TexturingError("INVALID_ARGUMENT", f"mode must be one of {', '.join(MODES)}", "mode")
    if (mode == "texture") != (texture is not None):
        raise TexturingError(
            "INVALID_ARGUMENT", "texture is given with mode 'texture' (and only then)", "texture"
        )
    topo = Topology.from_mesh(mesh)
    labels, count = island_labels(topo)
    corner = topo.uvs[topo.triangles].astype(np.float64)
    T = corner.shape[0]
    e1 = corner[:, 1] - corner[:, 0]
    e2 = corner[:, 2] - corner[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    area3 = topo.face_area
    # Background: a soft checker (outside every island).
    yy, xx = np.mgrid[0:size, 0:size]
    cell = max(size // 32, 1)
    checker = ((yy // cell + xx // cell) % 2).astype(np.float64)
    image = np.repeat((0.16 + 0.04 * checker)[..., None], 3, axis=2)
    if mode == "texture":
        image = _texture_background(texture, size, image)
    tri_color = None
    if mode == "islands":
        tri_color = _island_colors(max(count, 1))[labels]
    elif mode == "density":
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.sqrt(np.abs(signed) / area3)
            ok = np.isfinite(d) & (area3 > 0)
            mean = (d[ok] * area3[ok]).sum() / max(area3[ok].sum(), 1e-300) if ok.any() else 1.0
            x = np.log2(d / mean) * 0.5 + 0.5
        tri_color = np.where(ok[:, None], _ramp(np.nan_to_num(x, nan=0.5)), 0.5)
    elif mode == "stretch":
        tri_color = _ramp(np.clip((_anisotropy(topo, corner) - 1.0) / 2.0, 0, 1) * 0.5 + 0.5)
    elif mode == "overlaps":
        tri_color = np.full((T, 3), 0.78)
        island_signed = np.bincount(labels, signed, count)
        sign = np.where(island_signed < 0, -1.0, 1.0)
        flipped = (area3 > 0) & (signed != 0) & (np.sign(signed) != sign[labels])
        ta, tb = overlapping_triangles(corner, signed)
        tri_count = np.bincount(labels, minlength=count)
        uv_area = np.bincount(labels, np.abs(signed), count)
        la, lb = labels[ta], labels[tb]
        stacked = (la != lb) & (tri_count[la] == tri_count[lb])
        stacked &= np.isclose(uv_area[la], uv_area[lb], rtol=1e-6, atol=1e-12)
        bad = np.zeros(T, bool)
        bad[ta[~stacked]] = True
        bad[tb[~stacked]] = True
        stack = np.zeros(T, bool)
        stack[ta[stacked]] = True
        stack[tb[stacked]] = True
        tri_color[stack] = [0.95, 0.85, 0.25]
        tri_color[flipped] = [0.98, 0.55, 0.15]
        tri_color[bad] = [0.92, 0.15, 0.15]
    if tri_color is not None and T:
        cov = rasterize_uv(corner, size, size, 1)
        flat = image.reshape(-1, 3)
        flat[cov.sample] = tri_color[cov.triangle]
    if edges and T:
        _draw_edges(image, topo, corner, size)
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def _anisotropy(topo: Topology, corner: np.ndarray) -> np.ndarray:
    p = topo.positions[topo.triangles].astype(np.float64)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    la = np.linalg.norm(a, axis=1)
    u = a / np.maximum(la, 1e-300)[:, None]
    q2x = (b * u).sum(axis=1)
    q2y = np.linalg.norm(b - q2x[:, None] * u, axis=1)
    d1 = corner[:, 1] - corner[:, 0]
    d2 = corner[:, 2] - corner[:, 0]
    # J [q1 q2] = [d1 d2] with q1 = (la, 0), q2 = (q2x, q2y).
    with np.errstate(divide="ignore", invalid="ignore"):
        j00 = d1[:, 0] / la
        j10 = d1[:, 1] / la
        j01 = (d2[:, 0] - j00 * q2x) / q2y
        j11 = (d2[:, 1] - j10 * q2x) / q2y
        e = (j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11) / 2
        det = np.abs(j00 * j11 - j01 * j10)
        root = np.sqrt(np.maximum(e * e - det * det, 0))
        s1 = np.sqrt(e + root)
        s2 = np.sqrt(np.maximum(e - root, 0))
        out = s1 / s2
    return np.where(np.isfinite(out), out, 3.0)


def _texture_background(texture: Any, size: int, image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(texture, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.shape[0] < 1:
        raise TexturingError("INVALID_ARGUMENT", "texture is an (h, w, 3|4) image", "texture")
    if pixels.max() > 1.0:
        pixels = pixels / 255.0
    h, w = pixels.shape[:2]
    rows = np.minimum((np.arange(size) + 0.5) * h / size, h - 1).astype(int)
    cols = np.minimum((np.arange(size) + 0.5) * w / size, w - 1).astype(int)
    tex = pixels[rows][:, cols]
    rgb = tex[..., :3]
    if pixels.shape[2] == 4:
        a = tex[..., 3:4]
        return rgb * a + image * (1 - a)
    return rgb.copy()


def _draw_edges(image: np.ndarray, topo: Topology, corner: np.ndarray, size: int) -> None:
    """Every UV edge, one sample per half pixel; island boundaries (UV seams and open
    edges) darker than interior edges."""
    T = corner.shape[0]
    a = corner.reshape(T, 3, 1, 2)
    b = corner[:, [1, 2, 0]].reshape(T, 3, 1, 2)
    seam = topo.uv_seam.reshape(T, 3) if topo.uv_seam is not None else np.zeros((T, 3), bool)
    boundary = seam | (topo.mate.reshape(T, 3) < 0)
    a = a.reshape(-1, 2)
    b = b.reshape(-1, 2)
    strong = boundary.reshape(-1)
    length = np.linalg.norm((b - a) * size, axis=1)
    n = np.clip(np.ceil(length * 2).astype(np.int64) + 1, 2, 4 * size)
    total = int(n.sum())
    if total > MAX_EDGE_SAMPLES:
        return  # too dense to draw usefully at this size
    edge = np.repeat(np.arange(a.shape[0]), n)
    k = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    t = k / (n[edge] - 1)
    p = a[edge] + (b[edge] - a[edge]) * t[:, None]
    x = np.floor(p[:, 0] * size).astype(np.int64)
    y = np.floor((1.0 - p[:, 1]) * size).astype(np.int64)
    ok = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    weak_sel = ok & ~strong[edge]
    strong_sel = ok & strong[edge]
    image[y[weak_sel], x[weak_sel]] = image[y[weak_sel], x[weak_sel]] * 0.55 + 0.03
    image[y[strong_sel], x[strong_sel]] = [0.05, 0.05, 0.07]
