"""Voxel remeshing: a signed distance field on a sparse narrow band, then marching cubes.

1. **Narrow band.** The grid (spacing ``voxel_size``, world axes) is split into blocks of
   8 x 8 x 8 points, and only blocks within reach of a triangle get storage (each
   triangle's box grown by the band, pruned by the distance from the block's centre to
   the triangle). Every triangle then measures the exact distance to the grid points of
   its grown box (closest points by Ericson's regions; points beyond the band from its
   plane or bounding sphere skipped), and each point keeps its nearest triangle -- the
   lowest id among equally near ones, so the result never depends on batching. Memory
   grows with the surface's area, not the grid's volume.
2. **Sign** (``mode="solid"``) from angle-weighted pseudo-normals (Baerentzen and
   Aanaes, "Signed Distance Computation using the Angle Weighted Pseudo-normal", IEEE
   TVCG 11(3), 2005): the sign of ``(p - c) . n`` where ``c`` is the closest point and
   ``n`` the pseudo-normal of the feature it lies on (the face normal, the sum of an
   edge's two face normals, or a vertex's angle-weighted normal). It is exact for each
   closed, manifold, consistently oriented connected component, so every component must
   be one (``VOXEL_NOT_WATERTIGHT`` otherwise). Components combine as solids: the result
   is the union of the outward-facing components minus the inward-facing ones (negative
   volume: cavities). Overlapping parts therefore fuse into one surface -- a point inside
   one part is inside the result however near it is to another part's buried surface.
   ``mode="shell"`` instead extracts the surface at distance ``thickness / 2`` from any
   mesh, open or not: a closed shell around it.
3. **Marching cubes** (Lorensen and Cline 1987) on every cell whose eight corners are in
   the band and change sign. The case table is generated, not copied: on each cube face
   the crossings are joined into segments, the face's ambiguous configuration (two
   diagonal corners inside) always resolved by cutting each inside corner off on its own
   (the face rule of Montani, Scateni and Scopigno, 1994). Neighbouring cells agree on
   every shared face, so the surface is closed and manifold; ``tests/test_sculpt_voxel.py``
   checks that for all 256 configurations. Each closed cycle of crossings is fanned into
   triangles facing away from the inside corners (a triangulation whose diagonals never
   lie in a cube face, which the neighbouring cell could share). A grid value of exactly
   zero counts as outside, and crossings stay at least :data:`CROSSING_MARGIN` of a voxel
   from grid points, so no triangle degenerates.

The result is a new mesh (every vertex and triangle new; see
:meth:`~aether3d.sculpt.SculptMesh.voxel_remesh` for what is kept). Details thinner than
about two voxels are lost: thin parts can vanish or break, sharp edges and corners round
off. ``max_voxels`` bounds the points visited.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np

from ..errors import SculptError, ValidationError

CORNERS = np.array(list(product((0, 1), repeat=3)))[:, ::-1]
"""Corner ``i`` of a cell: ``(i & 1, i >> 1 & 1, i >> 2 & 1)``."""

EDGES = np.array(
    [(i, i | (1 << axis)) for i in range(8) for axis in range(3) if not i & (1 << axis)]
)
"""The 12 cell edges as (lower corner, upper corner)."""

FACES = ((0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5))
"""Each cell face's corners in cyclic order."""

BLOCK = 8
"""Grid points per block side."""

BAND = 1.75
"""Narrow-band half-width in voxels. Every corner of a cell the surface crosses lies
within ``sqrt(3) = 1.732`` voxels of the crossing on its edge, so it is in the band."""

CROSSING_MARGIN = 0.01
"""Crossings stay at least this fraction of a voxel away from grid points: crossings of
one grid point's edges never meet (no zero-area triangles), at the cost of moving a
vertex by at most 1% of a voxel along its edge."""

MAX_VOXELS = 50_000_000
"""Default limit on the grid points visited (``max_voxels``)."""

_POINT_BATCH = 1 << 20  # grid points per nearest-query batch
_QUERY_BATCH = 1 << 15  # lockstep traversal width


# ------------------------------------------------------------------------------------- #
# Case table
# ------------------------------------------------------------------------------------- #


def build_table() -> list[list[tuple[int, int, int]]]:
    """Triangles (as cell-edge indices) for each of the 256 inside/outside patterns (bit
    ``i`` set: corner ``i`` inside), generated from face segments (see the module docs);
    each triangle faces away from the inside corners."""
    edge_of = {tuple(sorted(map(int, e))): k for k, e in enumerate(EDGES)}
    table: list[list[tuple[int, int, int]]] = []
    for case in range(256):
        inside = [(case >> i) & 1 for i in range(8)]
        neighbours: dict[int, list[int]] = {}
        for face in FACES:
            cut = [
                edge_of[tuple(sorted((face[j], face[(j + 1) % 4])))]
                for j in range(4)
                if inside[face[j]] != inside[face[(j + 1) % 4]]
            ]
            if len(cut) == 2:
                pairs = [(cut[0], cut[1])]
            elif len(cut) == 4:  # ambiguous: cut each inside corner off on its own
                pairs = [
                    (
                        edge_of[tuple(sorted((face[j - 1], face[j])))],
                        edge_of[tuple(sorted((face[j], face[(j + 1) % 4])))],
                    )
                    for j in range(4)
                    if inside[face[j]]
                ]
            else:
                pairs = []
            for x, y in pairs:
                neighbours.setdefault(x, []).append(y)
                neighbours.setdefault(y, []).append(x)
        triangles: list[tuple[int, int, int]] = []
        seen: set[int] = set()
        for start in sorted(neighbours):
            if start in seen:
                continue
            cycle, prev, cur = [start], -1, start
            seen.add(start)
            while True:
                nxt = next(n for n in neighbours[cur] if n != prev)
                if nxt == start:
                    break
                cycle.append(nxt)
                seen.add(nxt)
                prev, cur = cur, nxt
            mid = np.array([(CORNERS[EDGES[e][0]] + CORNERS[EDGES[e][1]]) / 2 for e in cycle])
            normal = sum(np.cross(mid[k], mid[(k + 1) % len(cycle)]) for k in range(len(cycle)))
            inner = np.mean([CORNERS[c] for e in cycle for c in EDGES[e] if inside[c]], axis=0)
            if np.dot(normal, mid.mean(axis=0) - inner) < 0:
                cycle = cycle[::-1]
            triangles += _triangulate(cycle)
        table.append(triangles)
    return table


def _faces_of(edge: int) -> set[int]:
    a, b = (int(x) for x in EDGES[edge])
    return {f for f, face in enumerate(FACES) if a in face and b in face}


def _triangulate(cycle: list[int]) -> list[tuple[int, int, int]]:
    """Triangles of a crossing cycle (kept in its order) whose diagonals never join two
    crossings on one cube face: such a diagonal would lie in the face, where the
    neighbouring cell could put the same edge. Among the valid triangulations, the one
    with the largest smallest angle on the edge midpoints is taken."""
    n = len(cycle)
    pts = [(CORNERS[EDGES[e][0]] + CORNERS[EDGES[e][1]]) / 2 for e in cycle]
    faces = [_faces_of(e) for e in cycle]

    def allowed(i: int, j: int) -> bool:
        return (j - i) % n in (1, n - 1) or not (faces[i] & faces[j])

    def quality(tri: tuple[int, int, int]) -> float:
        out = math.pi
        for k in range(3):
            u = pts[tri[(k + 1) % 3]] - pts[tri[k]]
            v = pts[tri[(k + 2) % 3]] - pts[tri[k]]
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            out = min(out, math.acos(max(-1.0, min(1.0, cos))))
        return out

    best: dict[tuple[int, int], tuple[float, list] | None] = {}

    def solve(i: int, j: int) -> tuple[float, list] | None:
        # Triangulate the polygon cycle[i..j] closed by the chord (i, j).
        if j - i < 2:
            return (math.pi, [])
        if (i, j) in best:
            return best[(i, j)]
        result = None
        for k in range(i + 1, j):
            if not (allowed(i, k) and allowed(k, j)):
                continue
            left, right = solve(i, k), solve(k, j)
            if left is None or right is None:
                continue
            score = min(left[0], right[0], quality((i, k, j)))
            if result is None or score > result[0]:
                result = (score, left[1] + right[1] + [(i, k, j)])
        best[(i, j)] = result
        return result

    found = solve(0, n - 1)
    if found is None:  # pragma: no cover - every cycle the face rule makes has one
        raise AssertionError(f"no valid triangulation of crossing cycle {cycle}")
    return [(cycle[a], cycle[b], cycle[c]) for a, b, c in found[1]]


_TABLE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _flat_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(triangles (m, 3) edge ids, start (257,), count (256,))`` of the case table."""
    global _TABLE
    if _TABLE is None:
        table = build_table()
        count = np.array([len(x) for x in table], dtype=np.int64)
        start = np.concatenate([[0], np.cumsum(count)])
        flat = np.array([tri for x in table for tri in x], dtype=np.int64).reshape(-1, 3)
        _TABLE = (flat, start, count)
    return _TABLE


# ------------------------------------------------------------------------------------- #
# Signed distances on the band
# ------------------------------------------------------------------------------------- #


def pseudonormals(p: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face normals ``(m, 3)``, edge pseudo-normals per triangle side ``(m, 3, 3)`` (side
    ``k`` joins corners ``k`` and ``k + 1``) and angle-weighted vertex normals."""
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    cross = np.cross(b - a, c - a)
    length = np.sqrt((cross * cross).sum(axis=1))
    fn = np.zeros_like(cross)
    ok = length > 0
    fn[ok] = cross[ok] / length[ok, None]
    vn = np.zeros((len(p), 3))
    for k in range(3):
        u = p[t[:, (k + 1) % 3]] - p[t[:, k]]
        v = p[t[:, (k + 2) % 3]] - p[t[:, k]]
        den = np.sqrt((u * u).sum(axis=1) * (v * v).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ang = np.arccos(np.clip((u * v).sum(axis=1) / den, -1.0, 1.0))
        ang = np.where(den > 0, ang, 0.0)
        for j in range(3):
            vn[:, j] += np.bincount(t[:, k], weights=fn[:, j] * ang, minlength=len(p))
    side_a = t.T.ravel()
    side_b = np.roll(t, -1, axis=1).T.ravel()
    key = np.minimum(side_a, side_b) * np.int64(len(p) + 1) + np.maximum(side_a, side_b)
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    face = np.tile(np.arange(len(t)), 3)
    en = np.zeros((int(inv.max()) + 1 if len(inv) else 0, 3))
    for j in range(3):
        en[:, j] = np.bincount(inv, weights=fn[face, j], minlength=len(en))
    edge = en[inv].reshape(3, len(t), 3).transpose(1, 0, 2)
    return fn, edge, vn


def _signed(q, near_t, near_p, near_b, dist, t, normals) -> np.ndarray:
    """Signed distances from the pseudo-normal of the feature holding each closest point."""
    fn, en, vn = normals
    nz = near_b > 1e-12
    kind = nz.sum(axis=1)
    normal = fn[near_t].copy()
    rows = np.flatnonzero(kind == 2)
    if len(rows):  # on an edge: the side opposite the zero coordinate
        side = (np.argmin(nz[rows], axis=1) + 1) % 3
        normal[rows] = en[near_t[rows], side]
    rows = np.flatnonzero(kind == 1)
    if len(rows):  # at a vertex
        normal[rows] = vn[t[near_t[rows], np.argmax(nz[rows], axis=1)]]
    outside = ((q - near_p) * normal).sum(axis=1) >= 0
    return np.where(outside, dist, -dist)


def _boxes(a, b, c, h: float, origin, dims, band: float):
    """Each triangle's grid-index box grown by the band: ``(lo, span)``."""
    lo = np.minimum(np.minimum(a, b), c) - band
    hi = np.maximum(np.maximum(a, b), c) + band
    glo = np.clip(np.ceil((lo - origin) / h).astype(np.int64), 0, dims - 1)
    ghi = np.clip(np.floor((hi - origin) / h).astype(np.int64), 0, dims - 1)
    return glo, np.maximum(ghi - glo + 1, 0)


def _block_pairs(rows, a, b, c, h, origin, glo, span, band):
    """``(triangle, block x, y, z)`` for the blocks within reach of triangles ``rows``:
    the blocks overlapping each grown box, pruned by the distance from the block's
    centre to the triangle."""
    from ..spatial.geometry import closest_points_on_triangles

    blo = glo[rows] // BLOCK
    bspan = (glo[rows] + span[rows] - 1) // BLOCK - blo + 1
    cnt = np.where(span[rows].min(axis=1) > 0, bspan.prod(axis=1), 0)
    owner = np.repeat(rows, cnt)
    local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    sp = np.repeat(bspan, cnt, axis=0)
    base = np.repeat(blo, cnt, axis=0)
    bx = base[:, 0] + local % sp[:, 0]
    by = base[:, 1] + (local // sp[:, 0]) % sp[:, 1]
    bz = base[:, 2] + local // (sp[:, 0] * sp[:, 1])
    centre = origin + (np.stack([bx, by, bz], axis=1) * BLOCK + (BLOCK - 1) / 2) * h
    cp, _ = closest_points_on_triangles(centre, a[owner], b[owner], c[owner])
    reach = band + (BLOCK - 1) / 2 * math.sqrt(3.0) * h * (1 + 1e-9)
    near = ((centre - cp) ** 2).sum(axis=1) <= reach * reach
    return owner[near], bx[near], by[near], bz[near]


def _chunks_of(cost: np.ndarray, budget: int):
    """Consecutive index ranges whose summed ``cost`` stays near ``budget``."""
    csum = np.cumsum(cost)
    start = 0
    while start < len(cost):
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + budget, side="right")), start + 1)
        yield start, stop
        start = stop


def _blocks(a, b, c, h: float, origin, dims, band: float, max_voxels: int):
    """Sorted linear ids of the blocks within reach of a triangle (see the module docs),
    and the triangles' grown boxes."""
    bdims = (dims + BLOCK - 1) // BLOCK
    glo, span = _boxes(a, b, c, h, origin, dims, band)
    bcount = np.where(
        span.min(axis=1) > 0,
        ((glo + span - 1) // BLOCK - glo // BLOCK + 1).prod(axis=1),
        0,
    )
    found: list[np.ndarray] = []
    seen = 0
    for start, stop in _chunks_of(bcount, 2_000_000):
        _, bx, by, bz = _block_pairs(np.arange(start, stop), a, b, c, h, origin, glo, span, band)
        found.append(np.unique((bz * bdims[1] + by) * bdims[0] + bx))
        seen += len(found[-1])
        if seen * BLOCK**3 > max_voxels:
            found = [np.unique(np.concatenate(found))]
            seen = len(found[0])
            if seen * BLOCK**3 > max_voxels:
                raise SculptError(
                    "VOXEL_GRID_TOO_LARGE",
                    f"the narrow band needs more than {max_voxels} grid points (max_voxels)",
                    hint="use a larger voxel_size or a larger max_voxels",
                )
    blocks = np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.int64)
    return blocks, glo, span


SMALL_BOX = 4096
"""Grown boxes of at most this many grid points are visited point by point; larger
triangles visit the points of their blocks within reach instead."""

_CANDIDATES = 1 << 19  # (triangle, grid point) pairs measured per batch


def band_values(
    p: np.ndarray,
    t: np.ndarray,
    h: float,
    origin: np.ndarray,
    dims: np.ndarray,
    *,
    shell: float | None,
    max_voxels: int,
    component: np.ndarray | None = None,
    outward: np.ndarray | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(keys, values, nearest)``: the band's grid points (sorted linear keys), their
    field values (signed distance, or ``distance - shell``) and nearest triangles.

    Every triangle measures the exact distance (closest points by Ericson's regions) to
    the grid points of its box grown by the band, skipping points farther than the band
    from its plane or its bounding sphere; each point keeps the nearest triangle (the
    lowest id among equally near ones), in dense per-block arrays."""
    from ..spatial.geometry import closest_points_on_triangles

    band = BAND * h + (shell or 0.0)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    blocks, glo, span = _blocks(a, b, c, h, origin, dims, band, max_voxels)
    bdims = (dims + BLOCK - 1) // BLOCK
    best_d = np.full(len(blocks) * BLOCK**3, np.inf)
    best_t = np.full(len(blocks) * BLOCK**3, -1, dtype=np.int64)
    centroid = (a + b + c) / 3.0
    radius = np.sqrt(np.maximum.reduce([((x - centroid) ** 2).sum(axis=1) for x in (a, b, c)]))
    normal = np.cross(b - a, c - a)
    nlen = np.sqrt((normal * normal).sum(axis=1))
    unit = np.divide(normal, nlen[:, None], out=np.zeros_like(normal), where=nlen[:, None] > 0)
    volume = span.prod(axis=1)
    small = np.flatnonzero(volume <= SMALL_BOX)
    large = np.flatnonzero(volume > SMALL_BOX)

    def measure(tri, gx, gy, gz):
        q = origin + np.stack([gx, gy, gz], axis=1) * h
        d2c = ((q - centroid[tri]) ** 2).sum(axis=1)
        keep = d2c <= (band + radius[tri]) ** 2
        keep &= np.abs(((q - a[tri]) * unit[tri]).sum(axis=1)) <= band
        tri, q, gx, gy, gz = tri[keep], q[keep], gx[keep], gy[keep], gz[keep]
        cp, _ = closest_points_on_triangles(q, a[tri], b[tri], c[tri])
        d = np.sqrt(((q - cp) ** 2).sum(axis=1))
        keep = d <= band
        tri, d, gx, gy, gz = tri[keep], d[keep], gx[keep], gy[keep], gz[keep]
        block = (gz // BLOCK * bdims[1] + gy // BLOCK) * bdims[0] + gx // BLOCK
        slot = np.searchsorted(blocks, block) * BLOCK**3 + (
            (gz % BLOCK * BLOCK + gy % BLOCK) * BLOCK + gx % BLOCK
        )
        # Scatter the minimum distance, then the lowest triangle id among those at it
        # (forgetting the previous winner where the distance went down).
        previous = best_d[slot]
        np.minimum.at(best_d, slot, d)
        now = best_d[slot]
        best_t[slot[now < previous]] = np.iinfo(np.int64).max
        at = d == now
        np.minimum.at(best_t, slot[at], tri[at])

    tick = checkpoint or (lambda: None)
    for start, stop in _chunks_of(volume[small], _CANDIDATES):
        tick()
        rows = small[start:stop]
        cnt = volume[rows]
        tri = np.repeat(rows, cnt)
        local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        sp = span[tri]
        base = glo[tri]
        measure(
            tri,
            base[:, 0] + local % sp[:, 0],
            base[:, 1] + (local // sp[:, 0]) % sp[:, 1],
            base[:, 2] + local // (sp[:, 0] * sp[:, 1]),
        )
    lk = np.arange(BLOCK**3)
    lx, ly, lz = lk % BLOCK, (lk // BLOCK) % BLOCK, lk // (BLOCK * BLOCK)
    for row in large:
        tick()
        owner, bx, by, bz = _block_pairs(np.array([row]), a, b, c, h, origin, glo, span, band)
        for s in range(0, len(owner), max(1, _CANDIDATES // BLOCK**3)):
            sel = slice(s, s + max(1, _CANDIDATES // BLOCK**3))
            gx = (bx[sel, None] * BLOCK + lx).ravel()
            gy = (by[sel, None] * BLOCK + ly).ravel()
            gz = (bz[sel, None] * BLOCK + lz).ravel()
            lo, hi = glo[row], glo[row] + span[row] - 1
            ok = (gx >= lo[0]) & (gx <= hi[0]) & (gy >= lo[1]) & (gy <= hi[1])
            ok &= (gz >= lo[2]) & (gz <= hi[2])
            measure(np.full(int(ok.sum()), row, dtype=np.int64), gx[ok], gy[ok], gz[ok])
    slots = np.flatnonzero(best_t >= 0)
    if not len(slots):
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64)
    block = blocks[slots // BLOCK**3]
    local = slots % BLOCK**3
    gx = block % bdims[0] * BLOCK + local % BLOCK
    gy = (block // bdims[0]) % bdims[1] * BLOCK + (local // BLOCK) % BLOCK
    gz = block // (bdims[0] * bdims[1]) * BLOCK + local // (BLOCK * BLOCK)
    keys = (gz * dims[1] + gy) * dims[0] + gx
    order = np.argsort(keys, kind="stable")
    keys, slots = keys[order], slots[order]
    nearest = best_t[slots]
    dist = best_d[slots]
    if shell is not None:
        return keys, dist - shell, nearest
    normals = pseudonormals(p, t)
    q = origin + np.stack([gx[order], gy[order], gz[order]], axis=1) * h
    values = np.empty(len(keys))
    for s in range(0, len(keys), _CANDIDATES):
        sel = slice(s, s + _CANDIDATES)
        tri = nearest[sel]
        cp, bary = closest_points_on_triangles(q[sel], a[tri], b[tri], c[tri])
        values[sel] = _signed(q[sel], tri, cp, bary, dist[sel], t, normals)
    if component is not None and len(outward) > 1:
        values = _combine_components(
            p, t, keys, values, nearest, h, origin, dims, band, component, outward, normals
        )
    return keys, values, nearest


def _combine_components(
    p, t, keys, values, nearest, h, origin, dims, band, component, outward, normals
) -> np.ndarray:
    """Union of the outward components minus the inward ones (see the module docs).

    ``values`` hold each point's signed distance to its nearest component. For every
    other component whose box (grown by the band) holds the point, its own signed
    distance is measured too (an unbounded nearest query: the sign of a point deep inside
    a part matters). Components farther than the band change neither a value within the
    band nor any sign, so they are skipped."""
    from ..spatial.surface import TriangleSurface

    tri_comp = component[t[:, 0]]
    q = (
        origin
        + np.stack(
            [keys % dims[0], (keys // dims[0]) % dims[1], keys // (dims[0] * dims[1])], axis=1
        )
        * h
    )
    own = tri_comp[nearest]
    pos_min = np.where(outward[own], values, np.inf)
    cav_max = np.where(outward[own], -np.inf, values)
    for j in range(len(outward)):
        rows_t = np.flatnonzero(tri_comp == j)
        verts = np.unique(t[rows_t])
        lo = p[verts].min(axis=0) - band
        hi = p[verts].max(axis=0) + band
        rows = np.flatnonzero((own != j) & np.all((q >= lo) & (q <= hi), axis=1))
        if not len(rows):
            continue
        near = TriangleSurface(p, t[rows_t]).nearest(q[rows], batch=_QUERY_BATCH)
        glob = rows_t[near.triangles]
        s = _signed(q[rows], glob, near.points, near.bary, near.distance, t, normals)
        if outward[j]:
            pos_min[rows] = np.minimum(pos_min[rows], s)
        else:
            cav_max[rows] = np.maximum(cav_max[rows], s)
    return np.maximum(pos_min, cav_max)


# ------------------------------------------------------------------------------------- #
# Marching cubes
# ------------------------------------------------------------------------------------- #


def marching_cubes(
    keys: np.ndarray, values: np.ndarray, dims: np.ndarray, origin: np.ndarray, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, triangles)`` of the zero level set over the band's cells: each
    crossing vertex is shared by the cells around its grid edge."""
    flat, start, count = _flat_table()
    values = np.where(values == 0.0, 1e-12 * h, values)
    nx, ny = int(dims[0]), int(dims[1])
    offsets = np.array([((c >> 2) & 1) * nx * ny + ((c >> 1) & 1) * nx + (c & 1) for c in range(8)])
    ix, iy, iz = keys % nx, (keys // nx) % ny, keys // (nx * ny)
    cells = keys[(ix < dims[0] - 1) & (iy < dims[1] - 1) & (iz < dims[2] - 1)]
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if not len(cells):
        return empty
    corner_keys = cells[:, None] + offsets[None, :]
    pos = np.minimum(np.searchsorted(keys, corner_keys), len(keys) - 1)
    complete = (keys[pos] == corner_keys).all(axis=1)
    cells, pos = cells[complete], pos[complete]
    corner_vals = values[pos]
    case = ((corner_vals < 0) * (1 << np.arange(8))).sum(axis=1)
    mixed = (case != 0) & (case != 255)
    cells, corner_vals, case = cells[mixed], corner_vals[mixed], case[mixed]
    if not len(cells):
        return empty
    per_cell = count[case]
    owner = np.repeat(np.arange(len(cells)), per_cell)
    which = np.repeat(start[case], per_cell) + (
        np.arange(int(per_cell.sum())) - np.repeat(np.cumsum(per_cell) - per_cell, per_cell)
    )
    edge_ids = flat[which].ravel()  # three cell edges per triangle
    owner3 = np.repeat(owner, 3)
    lower = EDGES[edge_ids, 0]
    upper = EDGES[edge_ids, 1]
    axis = np.log2(upper - lower).astype(np.int64)
    edge_key = (cells[owner3] + offsets[lower]) * 3 + axis
    ukey, first, inverse = np.unique(edge_key, return_index=True, return_inverse=True)
    v0 = corner_vals[owner3[first], lower[first]]
    v1 = corner_vals[owner3[first], upper[first]]
    frac = np.clip(v0 / (v0 - v1), CROSSING_MARGIN, 1.0 - CROSSING_MARGIN)
    gp = ukey // 3
    base = origin + np.stack([gp % nx, (gp // nx) % ny, gp // (nx * ny)], axis=1) * h
    step = np.zeros((len(ukey), 3))
    step[np.arange(len(ukey)), ukey % 3] = h
    positions = base + frac[:, None] * step
    return positions, inverse.reshape(-1, 3)


# ------------------------------------------------------------------------------------- #
# Driver
# ------------------------------------------------------------------------------------- #


def grid_for(p: np.ndarray, h: float, extra: float) -> tuple[np.ndarray, np.ndarray]:
    """``(origin, dims)`` of a grid of spacing ``h`` holding ``p`` with a margin."""
    margin = 4 * h + extra  # more than the band: the outermost cells are all outside
    lo, hi = p.min(axis=0) - margin, p.max(axis=0) + margin
    dims = np.ceil((hi - lo) / h).astype(np.int64) + 1
    if np.prod(dims.astype(np.float64)) > 2.0**60:
        raise SculptError(
            "VOXEL_GRID_TOO_LARGE", "the grid is too large to index", hint="use a larger voxel_size"
        )
    return lo, dims


def voxel_surface(
    p: np.ndarray,
    t: np.ndarray,
    voxel_size: float,
    *,
    mode: str = "solid",
    thickness: float | None = None,
    max_voxels: int = MAX_VOXELS,
    component: np.ndarray | None = None,
    outward: np.ndarray | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Positions and triangles of the voxel surface of a triangle mesh given in world
    space (see the module docs), and statistics."""
    h = float(voxel_size)
    shell = None if mode == "solid" else float(thickness) / 2.0
    origin, dims = grid_for(p, h, shell or 0.0)
    keys, values, _nearest = band_values(
        p,
        t,
        h,
        origin,
        dims,
        shell=shell,
        max_voxels=max_voxels,
        component=component,
        outward=outward,
        checkpoint=checkpoint,
    )
    if checkpoint is not None:
        checkpoint()
    positions, triangles = marching_cubes(keys, values, dims, origin, h)
    stats = {"grid": [int(x) for x in dims], "band_points": int(len(keys))}
    return positions, triangles, stats


def check_voxel_args(voxel_size: Any, mode: Any, thickness: Any, max_voxels: Any) -> None:
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float)):
        raise ValidationError("voxel_size must be a number")
    if not (math.isfinite(float(voxel_size)) and float(voxel_size) > 0):
        raise ValidationError("voxel_size must be a finite number > 0")
    if mode not in ("solid", "shell"):
        raise ValidationError("mode must be 'solid' or 'shell'")
    if mode == "shell":
        if isinstance(thickness, bool) or not isinstance(thickness, (int, float)):
            raise ValidationError("mode='shell' needs a thickness (a number)")
        if not (math.isfinite(float(thickness)) and float(thickness) >= 2 * float(voxel_size)):
            raise ValidationError("the shell thickness must be at least two voxels")
    elif thickness is not None:
        raise ValidationError("thickness applies to mode='shell' only")
    if isinstance(max_voxels, bool) or not isinstance(max_voxels, int) or max_voxels < 512:
        raise ValidationError("max_voxels must be an integer >= 512")
