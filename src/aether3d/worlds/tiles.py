"""Tiles: quadtree addressing, level-of-detail selection, neighbour restriction and stitching.

The horizontal plane is covered by square tiles at levels ``0, 1, 2, ...``: a level-``k``
tile is ``chunk * 2^k`` metres on a side and meshes ``resolution`` quads per side, so its
vertex spacing is ``chunk * 2^k / resolution``. Tile ``(k, i, j)`` covers
``[i s, (i + 1) s) x [j s, (j + 1) s)`` with ``s = chunk * 2^k`` (negative indices are
fine); its children are ``(k - 1, 2i + a, 2j + b)``. Because ``chunk`` and ``resolution`` are
powers of two, every vertex coordinate is an exact binary fraction: neighbouring tiles
compute the very same coordinates, hence the very same heights, for the vertices they share.

**Selection** (:func:`select`) is a pure function of the camera and the world: starting
from the root tiles that meet the view distance, a tile is replaced by its four children
while its geometric error (metres, measured on its own samples by :mod:`.terrain_mesh`)
projects to more than ``lod_error`` pixels at its nearest point to the camera. There is no
hysteresis -- a frame never depends on frames rendered before it -- so a change of level
pops by at most ``lod_error`` pixels. **Restriction** then splits any tile whose edge
neighbour is more than one level finer, and each remaining tile learns which of its sides
border a coarser tile: those borders are *stitched* (their odd vertices are skipped, so
both tiles share exactly the same edge segments).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SIDES = ("-x", "+x", "-y", "+y")
STITCH_BITS = {"-x": 1, "+x": 2, "-y": 4, "+y": 8}


@dataclass(frozen=True, order=True)
class TileKey:
    level: int
    i: int
    j: int

    def __str__(self) -> str:
        return f"{self.level}/{self.i}/{self.j}"

    def parent(self) -> TileKey:
        return TileKey(self.level + 1, self.i >> 1, self.j >> 1)

    def children(self) -> tuple[TileKey, TileKey, TileKey, TileKey]:
        k, i, j = self.level - 1, 2 * self.i, 2 * self.j
        return (
            TileKey(k, i, j),
            TileKey(k, i + 1, j),
            TileKey(k, i, j + 1),
            TileKey(k, i + 1, j + 1),
        )


@dataclass(frozen=True)
class Grid:
    """The tile geometry of a world: ``chunk`` metres at level 0, ``resolution`` quads."""

    chunk: float
    resolution: int

    def size(self, level: int) -> float:
        return self.chunk * float(2**level)

    def step(self, level: int) -> float:
        return self.size(level) / self.resolution

    def bounds(self, key: TileKey) -> tuple[float, float, float, float]:
        s = self.size(key.level)
        return key.i * s, key.j * s, (key.i + 1) * s, (key.j + 1) * s

    def key_at(self, x: float, y: float, level: int) -> TileKey:
        s = self.size(level)
        return TileKey(level, math.floor(x / s), math.floor(y / s))

    def coords(self, key: TileKey, border: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """World coordinates of the tile's vertex columns and rows, ``border`` extra on each
        side: ``(i R + a) * step`` -- an integer times the step, exact."""
        r = self.resolution
        step = self.step(key.level)
        a = np.arange(-border, r + 1 + border, dtype=np.int64)
        xs = (key.i * r + a).astype(np.float64) * step
        ys = (key.j * r + a).astype(np.float64) * step
        return xs, ys


@dataclass(frozen=True)
class View:
    """What selection needs from a camera: position, forward, pixels per radian."""

    eye: np.ndarray  # (3,) float64
    forward: np.ndarray  # (3,) unit
    pixels_per_radian: float  # image height / vertical field of view (perspective)
    distance: float  # view distance (m): tiles farther than this are not drawn
    frustum: Any = None  # optional (planes (6, 4)) for culling (None: no culling)


def view_from_camera(
    camera: Any, frame: Any, height: int, aspect: float, distance: float, margin: float = 0.0
) -> View:
    """A :class:`View` of a render camera (``RenderCamera.camera`` and ``.frame``): its
    frustum (side planes widened by ``margin`` metres, so tiles casting shadows into view
    keep their detail) bounds where tiles are refined."""
    hx, hy = camera.half_extents(aspect)
    eye = np.asarray(frame.position, dtype=np.float64)
    f = np.asarray(frame.forward, dtype=np.float64)
    r = np.asarray(frame.right, dtype=np.float64)
    u = np.asarray(frame.up, dtype=np.float64)
    if camera.is_perspective:
        fov = 2.0 * math.atan(hy)
        ppr = height / max(fov, 1e-6)
        # Inward plane normals of the four side planes through the eye.
        normals = [f * hx + r, f * hx - r, f * hy + u, f * hy - u]
        planes = []
        for nrm in normals:
            nrm = nrm / np.linalg.norm(nrm)
            planes.append(np.append(nrm, -float(nrm @ eye) + margin))
        planes.append(np.append(f, -float(f @ eye) + margin))  # behind the eye
        frustum = np.array(planes)
    else:
        ppr = height / max(2.0 * hy, 1e-9)  # orthographic: pixels per metre (see select)
        frustum = None
    return View(eye, f, float(ppr), float(distance), frustum)


def outside(view: View, box: tuple[float, float, float, float], zlo: float, zhi: float) -> bool:
    """Whether a box is entirely outside one of the view's frustum planes."""
    if view.frustum is None:
        return False
    xs = (box[0], box[2])
    ys = (box[1], box[3])
    zs = (zlo, zhi)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    d = corners @ view.frustum[:, :3].T + view.frustum[:, 3]
    return bool(np.any(np.all(d < 0.0, axis=0)))


def box_distance(
    eye: np.ndarray, box: tuple[float, float, float, float], zlo: float, zhi: float
) -> float:
    """Euclidean distance from ``eye`` to an axis-aligned box (0 inside)."""
    dx = max(box[0] - eye[0], 0.0, eye[0] - box[2])
    dy = max(box[1] - eye[1], 0.0, eye[1] - box[3])
    dz = max(zlo - eye[2], 0.0, eye[2] - zhi)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def roots(
    grid: Grid, view: View, extent: tuple[float, float, float, float] | None, max_level: int
) -> list[TileKey]:
    """Root tiles at ``max_level`` covering the world extent (bounded) or the view
    distance around the eye (unbounded), in key order."""
    s = grid.size(max_level)
    if extent is not None:
        x0, y0, x1, y1 = extent
    else:
        r = view.distance
        x0, y0, x1, y1 = view.eye[0] - r, view.eye[1] - r, view.eye[0] + r, view.eye[1] + r
    i0, j0 = math.floor(x0 / s), math.floor(y0 / s)
    i1, j1 = math.ceil(x1 / s) - 1, math.ceil(y1 / s) - 1
    return [TileKey(max_level, i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def overlaps_extent(
    grid: Grid, key: TileKey, extent: tuple[float, float, float, float] | None
) -> bool:
    if extent is None:
        return True
    b = grid.bounds(key)
    return not (b[2] <= extent[0] or b[0] >= extent[2] or b[3] <= extent[1] or b[1] >= extent[3])


def select(
    grid: Grid,
    view: View,
    error_of: Any,
    zrange_of: Any,
    lod_error: float,
    extent: tuple[float, float, float, float] | None,
    max_level: int,
    limit: int,
    force: Any = None,
) -> list[TileKey]:
    """The tiles to draw (see the module docstring). ``error_of(key)`` and
    ``zrange_of(key)`` give a tile's geometric error and height range (they may generate
    it); ``force(key)`` says a tile must be refined (e.g. volumetric chunks at level 0);
    ``limit`` bounds the tile count (coarser tiles are kept beyond it)."""
    out: list[TileKey] = []
    stack = sorted(roots(grid, view, extent, max_level), reverse=True)
    while stack:
        key = stack.pop()
        if not overlaps_extent(grid, key, extent):
            continue
        b = grid.bounds(key)
        zlo, zhi = zrange_of(key)
        d = box_distance(view.eye, b, zlo, zhi)
        if d > view.distance:
            continue
        refine = False
        if key.level > 0 and len(out) + len(stack) + 4 <= limit and not outside(view, b, zlo, zhi):
            err = error_of(key)
            projected = err * view.pixels_per_radian / max(d, 1e-3)
            refine = projected > lod_error or (force is not None and force(key))
        if refine:
            stack.extend(sorted(key.children(), reverse=True))
        else:
            out.append(key)
    return restrict(out)


def _neighbours(key: TileKey) -> list[tuple[str, int, int]]:
    return [
        ("-x", key.i - 1, key.j),
        ("+x", key.i + 1, key.j),
        ("-y", key.i, key.j - 1),
        ("+y", key.i, key.j + 1),
    ]


def _covering(
    leaves: dict[tuple[int, int, int], TileKey], level: int, i: int, j: int, max_level: int
) -> TileKey | None:
    """The leaf containing the level-``level`` cell ``(i, j)`` at that level or coarser."""
    k, a, b = level, i, j
    while k <= max_level:
        found = leaves.get((k, a, b))
        if found is not None:
            return found
        k, a, b = k + 1, a >> 1, b >> 1
    return None


def restrict(tiles: list[TileKey]) -> list[TileKey]:
    """Split leaves until edge neighbours differ by at most one level (a restricted
    quadtree), in key order."""
    if not tiles:
        return []
    max_level = max(t.level for t in tiles)
    leaves = {(t.level, t.i, t.j): t for t in tiles}
    changed = True
    while changed:
        changed = False
        for key in sorted(leaves.values()):
            if (key.level, key.i, key.j) not in leaves:
                continue
            for _, ni, nj in _neighbours(key):
                # Is any finer leaf along this side more than one level finer than the
                # neighbour cell's covering leaf? Check from the finer side: for each leaf,
                # its neighbour cell's covering leaf must be at most one level coarser.
                cover = _covering(leaves, key.level, ni, nj, max_level)
                if cover is not None and cover.level > key.level + 1:
                    del leaves[(cover.level, cover.i, cover.j)]
                    for c in cover.children():
                        leaves[(c.level, c.i, c.j)] = c
                    changed = True
    return sorted(leaves.values())


def stitch_masks(tiles: list[TileKey]) -> dict[TileKey, int]:
    """For each tile, the bits (:data:`STITCH_BITS`) of the sides whose neighbour is one
    level coarser (after :func:`restrict`)."""
    if not tiles:
        return {}
    max_level = max(t.level for t in tiles)
    leaves = {(t.level, t.i, t.j): t for t in tiles}
    out = {}
    for key in tiles:
        mask = 0
        for side, ni, nj in _neighbours(key):
            cover = _covering(leaves, key.level, ni, nj, max_level)
            if cover is not None and cover.level > key.level:
                mask |= STITCH_BITS[side]
        out[key] = mask
    return out
