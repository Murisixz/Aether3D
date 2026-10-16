"""Local topology changes for dynamic topology and remeshing, applied in batches.

Each pass applies a whole set of edge operations at once with NumPy, the way mesh
adaptation is parallelized: operations chosen as independent sets, so their checks and
effects do not interfere (Rokos, Gorman, Southern and Kelly, "A thread-parallel algorithm
for anisotropic mesh adaptation", arXiv:1308.2480). Lengths are world lengths.

* **split** -- every eligible edge longer than the limit at once, by the triangle
  refinement patterns: a triangle with one, two or three long edges becomes two, three
  or four triangles that share each edge's midpoint with the neighbour (no
  T-junctions); the two-edge pattern cuts its quad along the shorter diagonal;
* **collapse** -- an independent set of short edges: an edge collapses only if its
  priority (a deterministic scramble of its key, as Luby's algorithm uses random ones)
  is the smallest of the candidates touching the one-rings of its two vertices, so no
  two collapses share a triangle. Each passes the
  link condition (the two vertices share exactly the two opposite vertices), keeps the
  opposite vertices' valence above 3, creates no edge longer than the split limit and
  folds or degenerates no triangle (normals turn by less than 60 degrees);
* **flip** -- an independent set of edges (no two in one triangle) whose flip makes the
  pair Delaunay (``"delaunay"``) or brings the four valences closer to 6, or 4 on
  boundaries (``"valence"``, Botsch and Kobbelt 2004); refused when the new edge exists
  or a triangle would fold;
* **relax** -- tangential smoothing toward the area-weighted centroid of the one-ring,
  ``p += lambda (I - n n^T)(g - p)``; UVs are re-sampled from the triangle of the old
  one-ring the vertex moves into.

**Only vertices whose whole one-ring is known can change.** The caller marks *active*
vertices (inside the brush, or every vertex for a remesh); everything else is locked.
**Features are never broken:** boundary and non-manifold edges, UV seams and hard edges
(the two triangles' render vertices differ at an end), face-group borders and edges
wound inconsistently are features. Their vertices, masked vertices and vertices of
degenerate triangles are locked. Long feature edges (not non-manifold ones) are still
split -- each side keeps its own interpolated attributes -- but nothing is collapsed or
flipped across or along a feature.

**Attributes.** A split interpolates every corner attribute exactly at the midpoint (each
side of a seam on its own); a collapse moves the survivor to the edge's midpoint with the
midpoint of the two UVs (a vertex off every seam has one render vertex); a flip reuses
the four corners' render vertices (one chart: the edge is not a seam). New triangles take
the face group of the triangle they come from. Normals of changed geometry are
recomputed at commit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import SculptError

SPLIT_RATIO = 4.0 / 3.0
"""Edges longer than this times the target length are split (Botsch and Kobbelt)."""

COLLAPSE_RATIO = 4.0 / 5.0
"""Edges shorter than this times the target length are collapsed."""

FOLD_COS = math.cos(math.radians(60.0))
"""A collapse or flip may turn no triangle's normal by more than 60 degrees."""

_EMPTY_I = np.zeros(0, dtype=np.int64)


# ------------------------------------------------------------------------------------- #
# Regions
# ------------------------------------------------------------------------------------- #


@dataclass
class Region:
    """Alive triangles ``tris`` with their corners, edges and locks.

    Half-edge ``k * t + i`` runs ``tri[i, k] -> tri[i, (k + 1) % 3]``; ``he_edge`` maps it
    to its undirected edge (``edge_v``: sorted vertex pairs, ``count``: triangles here).
    ``verts`` are the region's vertices (sorted); ``locked`` and ``valence`` are per
    region vertex (``index(v)``).
    """

    tris: np.ndarray
    tri: np.ndarray
    trv: np.ndarray
    edge_v: np.ndarray
    count: np.ndarray
    he_edge: np.ndarray
    feature: np.ndarray
    verts: np.ndarray
    locked: np.ndarray
    active: np.ndarray
    valence: np.ndarray
    splittable: np.ndarray  # per region vertex: edges touching it may be split

    @property
    def size(self) -> int:
        return len(self.tris)

    def index(self, v: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.verts, v)


def build_region(
    sm, tris: np.ndarray, active_vertices: np.ndarray, split_vertices: np.ndarray | None = None
) -> Region:
    """The region of alive triangles ``tris``. ``active_vertices`` (global ids) must have
    every incident triangle in ``tris``; all other vertices are locked, and so are the
    vertices of features, masked vertices and vertices of degenerate triangles. Edges
    touching ``split_vertices`` (default: the active ones) may be split."""
    tris = np.asarray(tris, dtype=np.int64)
    tri = sm._tri[tris]
    trv = sm._trv[tris]
    t = len(tris)
    a = tri.T.ravel()
    b = np.roll(tri, -1, axis=1).T.ravel()
    nv = max(sm._nv, 1)
    key = np.minimum(a, b) * np.int64(nv) + np.maximum(a, b)
    ukey, he_edge, count = np.unique(key, return_inverse=True, return_counts=True)
    he_edge = he_edge.reshape(-1)
    edge_v = np.stack([ukey // nv, ukey % nv], axis=1)
    feature = count != 2
    order = np.argsort(he_edge, kind="stable")
    pair_edges = np.flatnonzero(count == 2)
    starts = np.searchsorted(he_edge[order], pair_edges)
    h1, h2 = order[starts], order[starts + 1]
    ra = trv.T.ravel()
    rb = np.roll(trv, -1, axis=1).T.ravel()
    inconsistent = a[h1] == a[h2]  # both run the edge the same way
    seam = (ra[h1] != rb[h2]) | (rb[h1] != ra[h2])
    groups = sm._tgroup[tris]
    border = groups[h1 % t] != groups[h2 % t]
    feature[pair_edges[inconsistent | seam | border]] = True
    verts = np.unique(tri.ravel())
    active = np.zeros(len(verts), dtype=bool)
    act = np.asarray(active_vertices, dtype=np.int64)
    pos_in = np.searchsorted(verts, act)
    ok = (pos_in < len(verts)) & (verts[np.minimum(pos_in, len(verts) - 1)] == act)
    active[pos_in[ok]] = True
    locked = ~active
    locked[np.searchsorted(verts, edge_v[feature].ravel())] = True
    locked |= sm._mask[verts] > 0
    degenerate = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    if degenerate.any():
        locked[np.searchsorted(verts, tri[degenerate].ravel())] = True
    valence = np.bincount(np.searchsorted(verts, edge_v.ravel()), minlength=len(verts))
    if split_vertices is None:
        splittable = active.copy()
    else:
        splittable = np.zeros(len(verts), dtype=bool)
        sv = np.asarray(split_vertices, dtype=np.int64)
        pos_in = np.searchsorted(verts, sv)
        hit = (pos_in < len(verts)) & (verts[np.minimum(pos_in, len(verts) - 1)] == sv)
        splittable[pos_in[hit]] = True
        splittable &= active
    return Region(
        tris, tri, trv, edge_v, count, he_edge, feature, verts, locked, active, valence, splittable
    )


def scramble(keys: np.ndarray) -> np.ndarray:
    """A deterministic pseudo-random permutation of 64-bit keys (splitmix64's finalizer):
    priorities that break the spatial correlation of edge keys and lengths, so each pass
    finds a large independent set (as Luby's randomized algorithm does)."""
    x = keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def world_lengths(sm, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (sm._pos[b] - sm._pos[a]) @ np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3].T
    return np.sqrt((d * d).sum(axis=1))


# ------------------------------------------------------------------------------------- #
# The journaled primitive
# ------------------------------------------------------------------------------------- #


def apply_topology(
    sm,
    *,
    removed_tris: np.ndarray,
    new_tri: np.ndarray,
    new_trv: np.ndarray,
    new_group: np.ndarray,
    new_chunk: np.ndarray,
    new_pos: np.ndarray | None = None,
    new_mask: np.ndarray | None = None,
    new_rgeo: np.ndarray | None = None,
    new_ruv: np.ndarray | None = None,
    new_rng: np.ndarray | None = None,
    moved: np.ndarray | None = None,
    moved_pos: np.ndarray | None = None,
    removed_vertices: np.ndarray | None = None,
    uv_ids: np.ndarray | None = None,
    uv_values: np.ndarray | None = None,
) -> np.ndarray:
    """One journaled topology change: remove triangles, add vertices, render vertices
    and triangles (new ones referenced as ``nv + k`` and ``nr + k``), move or remove
    vertices and set render-vertex UVs. Every alive triangle incident to a moved or
    removed vertex must be among ``removed_tris`` (re-added if it stays). Returns the
    new triangles' ids."""
    new_pos = np.zeros((0, 3)) if new_pos is None else new_pos
    new_mask = np.zeros(0, dtype=np.float32) if new_mask is None else new_mask
    new_rgeo = _EMPTY_I if new_rgeo is None else new_rgeo
    new_rng = _EMPTY_I if new_rng is None else new_rng
    moved = _EMPTY_I if moved is None else moved
    moved_pos = np.zeros((0, 3)) if moved_pos is None else moved_pos
    removed_vertices = _EMPTY_I if removed_vertices is None else removed_vertices
    uv_ids = _EMPTY_I if uv_ids is None else uv_ids
    nv, nt, nr = sm._nv, sm._nt, sm._nr
    k_v, k_t, k_r = len(new_pos), len(new_tri), len(new_rgeo)
    if nv + k_v > sm._limit or nt + k_t > sm._limit:
        raise SculptError(
            "TOO_MANY_ELEMENTS",
            f"the mesh would exceed {sm._limit} vertices or triangles",
            hint="use a larger detail size or a smaller brush",
        )
    if not np.all(np.isfinite(new_pos)) or not np.all(np.isfinite(moved_pos)):
        raise SculptError("NON_FINITE_RESULT", "a topology change produced non-finite positions")
    touched = np.unique(
        np.concatenate([sm._tri[removed_tris].ravel(), moved, removed_vertices]).astype(np.int64)
    )
    chunks = np.unique(np.concatenate([sm._chunk_of[removed_tris], new_chunk]).astype(np.int64))
    state = {
        "nv": nv,
        "nt": nt,
        "nr": nr,
        "removed_tris": removed_tris.copy(),
        "removed_vertices": removed_vertices.copy(),
        "moved": moved.copy(),
        "moved_pos": sm._pos[moved].copy(),
        "vsum_ids": touched,
        "vsum": sm._vsum[touched].copy(),
        "dirty_ids": touched[~sm._dirty[touched]],
        "uv_ids": uv_ids.copy(),
        "uv": None if sm._ruv is None else sm._ruv[uv_ids].copy(),
        "chunk_ids": chunks,
        "chunk_lists": [sm._chunks[i].copy() for i in chunks.tolist()],
        "chunk_lo": sm._chunk_lo[chunks].copy(),
        "chunk_hi": sm._chunk_hi[chunks].copy(),
        "chunk_count": len(sm._chunks),
        "changed": sm._topology_changed,
    }
    sm._log(("topology", state))
    sm._topology_changed = True
    sm._dirty[touched] = True
    if len(removed_tris):
        scatter_rows(sm._vsum, sm._tri[removed_tris], -sm._cross[removed_tris])
        sm._talive[removed_tris] = False
    if len(moved):
        sm._pos[moved] = moved_pos
    if len(removed_vertices):
        sm._valive[removed_vertices] = False
    sm._ensure(nv + k_v, nt + k_t, nr + k_r)
    if k_v:
        sm._pos[nv : nv + k_v] = new_pos
        sm._valive[nv : nv + k_v] = True
        sm._mask[nv : nv + k_v] = new_mask
        sm._dirty[nv : nv + k_v] = True
        sm._stamp[nv : nv + k_v] = 0
        sm._vsum[nv : nv + k_v] = 0.0
        sm._nv = nv + k_v
    if k_r:
        sm._rgeo[nr : nr + k_r] = new_rgeo
        if sm._ruv is not None:
            sm._ruv[nr : nr + k_r] = new_ruv
        if sm._rnorm is not None:
            sm._rnorm[nr : nr + k_r] = 0.0  # recomputed at commit (the vertex is changed)
        sm._rng[nr : nr + k_r] = new_rng
        sm._nr = nr + k_r
    if len(uv_ids) and sm._ruv is not None:
        sm._ruv[uv_ids] = uv_values
    new_ids = np.arange(nt, nt + k_t, dtype=np.int64)
    if k_t:
        sm._tri[nt : nt + k_t] = new_tri
        sm._trv[nt : nt + k_t] = new_trv
        sm._talive[nt : nt + k_t] = True
        sm._tgroup[nt : nt + k_t] = new_group
        sm._chunk_of[nt : nt + k_t] = new_chunk
        sm._nt = nt + k_t
        cross = sm._triangle_cross(new_ids)
        sm._cross[nt : nt + k_t] = cross
        scatter_rows(sm._vsum, new_tri, cross)
    for i in chunks.tolist():
        lst = sm._chunks[i]
        sm._chunks[i] = lst[sm._talive[lst]]
    if k_t:
        order = np.argsort(new_chunk, kind="stable")
        sorted_chunks = new_chunk[order]
        starts = np.flatnonzero(np.r_[True, sorted_chunks[1:] != sorted_chunks[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            c = int(sorted_chunks[s])
            sm._chunks[c] = np.concatenate([sm._chunks[c], new_ids[order[s:e]]])
    sm._split_big_chunks(chunks)
    refit = np.concatenate([chunks, np.arange(state["chunk_count"], len(sm._chunks))])
    sm._refit_chunks(np.unique(refit))
    return new_ids


def restore_topology(sm, state: dict) -> None:
    """Undo one :func:`apply_topology` exactly (called by the journal's rollback)."""
    sm._talive[state["nt"] : sm._nt] = False
    sm._valive[state["nv"] : sm._nv] = False
    sm._nv, sm._nt, sm._nr = state["nv"], state["nt"], state["nr"]
    sm._talive[state["removed_tris"]] = True
    sm._valive[state["removed_vertices"]] = True
    sm._pos[state["moved"]] = state["moved_pos"]
    sm._vsum[state["vsum_ids"]] = state["vsum"]
    sm._dirty[state["dirty_ids"]] = False
    if state["uv"] is not None and len(state["uv_ids"]):
        sm._ruv[state["uv_ids"]] = state["uv"]
    count = state["chunk_count"]
    del sm._chunks[count:]
    sm._chunk_lo = sm._chunk_lo[:count].copy()
    sm._chunk_hi = sm._chunk_hi[:count].copy()
    for i, lst in zip(state["chunk_ids"].tolist(), state["chunk_lists"], strict=True):
        sm._chunks[i] = lst
        sm._chunk_of[lst] = i
    sm._chunk_lo[state["chunk_ids"]] = state["chunk_lo"]
    sm._chunk_hi[state["chunk_ids"]] = state["chunk_hi"]
    sm._topology_changed = state["changed"]


def scatter_rows(target: np.ndarray, tri: np.ndarray, values: np.ndarray) -> None:
    """``target[tri[i, k]] += values[i]`` for every row and corner (repeats summed)."""
    if not len(tri):
        return
    idx = tri.ravel()
    uniq, inverse = np.unique(idx, return_inverse=True)
    vals = np.repeat(values, 3, axis=0)
    for k in range(3):
        target[uniq, k] += np.bincount(inverse, weights=vals[:, k], minlength=len(uniq))


# ------------------------------------------------------------------------------------- #
# Split
# ------------------------------------------------------------------------------------- #


def split_pass(sm, region: Region, limit: float, max_new: int) -> int:
    """Split the region's edges longer than ``limit`` (world units) with an active end
    and no masked one (feature edges too; non-manifold ones never). At most ``max_new``
    edges, longest first. Returns the number of edges split."""
    if not region.size:
        return 0
    ev = region.edge_v
    li = region.index
    # An edge with an active end has all its triangles here: splitting it is safe even
    # on a feature (each side keeps its attributes); masked ends keep their topology.
    active_end = region.splittable[li(ev[:, 0])] | region.splittable[li(ev[:, 1])]
    masked = (sm._mask[ev[:, 0]] > 0) | (sm._mask[ev[:, 1]] > 0)
    tri = region.tri
    degenerate = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    bad_edge = np.zeros(len(ev), dtype=bool)
    if degenerate.any():
        t = region.size
        bad_edge[region.he_edge.reshape(3, t)[:, degenerate].ravel()] = True
    eligible = active_end & (region.count <= 2) & ~masked & ~bad_edge
    length = world_lengths(sm, ev[:, 0], ev[:, 1])
    chosen = np.flatnonzero(eligible & (length > limit))
    if not len(chosen):
        return 0
    if len(chosen) > max_new:
        order = np.lexsort((chosen, -length[chosen]))
        chosen = np.sort(chosen[order[:max_new]])
    t = region.size
    long = np.zeros(len(ev), dtype=bool)
    long[chosen] = True
    he = region.he_edge.reshape(3, t).T  # (t, 3): edge of side k (corner k -> k + 1)
    marked = long[he]
    rows = np.flatnonzero(marked.any(axis=1))
    nv, nr = sm._nv, sm._nr
    new_v = np.full(len(ev), -1, dtype=np.int64)
    new_v[chosen] = nv + np.arange(len(chosen))
    ends = ev[chosen]
    new_pos = (sm._pos[ends[:, 0]] + sm._pos[ends[:, 1]]) / 2.0
    new_mask = ((sm._mask[ends[:, 0]] + sm._mask[ends[:, 1]]) / 2.0).astype(np.float32)
    # Render vertices at the midpoints: one per (edge, render pair) -- shared by the two
    # sides of an edge that is not a seam.
    tri_r, trv_r, mk, hk = tri[rows], region.trv[rows], marked[rows], he[rows]
    side_rows, side_k = np.nonzero(mk)
    ra = trv_r[side_rows, side_k]
    rb = trv_r[side_rows, (side_k + 1) % 3]
    keys = np.stack([hk[side_rows, side_k], np.minimum(ra, rb), np.maximum(ra, rb)], axis=1)
    ukeys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    new_rgeo = new_v[ukeys[:, 0]]
    new_ruv = None
    if sm._ruv is not None:
        new_ruv = (sm._ruv[ukeys[:, 1]] + sm._ruv[ukeys[:, 2]]) / 2.0
    gpair = np.sort(np.stack([sm._rng[ukeys[:, 1]], sm._rng[ukeys[:, 2]]], axis=1), axis=1)
    _, ginv = np.unique(
        np.concatenate([new_rgeo[:, None], gpair], axis=1), axis=0, return_inverse=True
    )
    base = int(sm._rng[:nr].max()) + 1 if nr else 0
    new_rng = base + ginv.reshape(-1)
    corner_v = np.full((len(rows), 3), -1, dtype=np.int64)
    corner_r = np.full((len(rows), 3), -1, dtype=np.int64)
    corner_v[side_rows, side_k] = new_v[hk[side_rows, side_k]]
    corner_r[side_rows, side_k] = nr + inverse

    def position(v: np.ndarray) -> np.ndarray:
        out = np.empty((len(v), 3))
        new = v >= nv
        out[new] = new_pos[v[new] - nv]
        out[~new] = sm._pos[v[~new]]
        return out

    new_tri, new_trv, parent = refine(tri_r, trv_r, mk, corner_v, corner_r, position)
    src = region.tris[rows]
    apply_topology(
        sm,
        removed_tris=src,
        new_tri=new_tri,
        new_trv=new_trv,
        new_group=sm._tgroup[src][parent],
        new_chunk=sm._chunk_of[src][parent],
        new_pos=new_pos,
        new_mask=new_mask,
        new_rgeo=new_rgeo,
        new_ruv=new_ruv,
        new_rng=new_rng,
    )
    return len(chosen)


def refine(tri, trv, marked, corner_v, corner_r, position):
    """Children of triangles with one, two or three marked sides (side ``k`` runs from
    corner ``k`` to ``k + 1``; ``corner_v``/``corner_r`` hold its midpoint's vertex and
    render vertex). Winding is kept. Returns ``(triangles, render corners, parent row)``."""
    out_t, out_r, parent = [], [], []
    n = marked.sum(axis=1)

    def emit(rows, vs, rs):
        out_t.append(np.stack(vs, axis=1))
        out_r.append(np.stack(rs, axis=1))
        parent.append(rows)

    rows = np.flatnonzero(n == 3)
    if len(rows):
        v0, v1, v2 = (tri[rows, k] for k in range(3))
        m0, m1, m2 = (corner_v[rows, k] for k in range(3))
        r0, r1, r2 = (trv[rows, k] for k in range(3))
        s0, s1, s2 = (corner_r[rows, k] for k in range(3))
        emit(rows, (v0, m0, m2), (r0, s0, s2))
        emit(rows, (m0, v1, m1), (s0, r1, s1))
        emit(rows, (m2, m1, v2), (s2, s1, r2))
        emit(rows, (m0, m1, m2), (s0, s1, s2))
    for count in (1, 2):
        rows = np.flatnonzero(n == count)
        if not len(rows):
            continue
        # Rotate so that side k is the marked side (one) or the unmarked side (two).
        k = np.argmax(marked[rows], axis=1) if count == 1 else np.argmin(marked[rows], axis=1)
        a, b, c = tri[rows, k], tri[rows, (k + 1) % 3], tri[rows, (k + 2) % 3]
        ra, rb, rc = trv[rows, k], trv[rows, (k + 1) % 3], trv[rows, (k + 2) % 3]
        if count == 1:
            m, s = corner_v[rows, k], corner_r[rows, k]
            emit(rows, (a, m, c), (ra, s, rc))
            emit(rows, (m, b, c), (s, rb, rc))
            continue
        # Sides b-c (midpoint p) and c-a (midpoint q) are marked; a-b is not.
        p, sp = corner_v[rows, (k + 1) % 3], corner_r[rows, (k + 1) % 3]
        q, sq = corner_v[rows, (k + 2) % 3], corner_r[rows, (k + 2) % 3]
        emit(rows, (q, p, c), (sq, sp, rc))
        # The quad a, b, p, q: cut along the shorter diagonal (a-p or b-q).
        use_ap = ((position(a) - position(p)) ** 2).sum(axis=1) <= (
            (position(b) - position(q)) ** 2
        ).sum(axis=1)
        x, y = rows[use_ap], rows[~use_ap]
        sel = use_ap
        emit(x, (a[sel], b[sel], p[sel]), (ra[sel], rb[sel], sp[sel]))
        emit(x, (a[sel], p[sel], q[sel]), (ra[sel], sp[sel], sq[sel]))
        sel = ~use_ap
        emit(y, (a[sel], b[sel], q[sel]), (ra[sel], rb[sel], sq[sel]))
        emit(y, (b[sel], p[sel], q[sel]), (rb[sel], sp[sel], sq[sel]))
    return (
        np.concatenate(out_t).astype(np.int64),
        np.concatenate(out_r).astype(np.int64),
        np.concatenate(parent),
    )


# ------------------------------------------------------------------------------------- #
# Collapse
# ------------------------------------------------------------------------------------- #


def _csr(rows: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compressed rows: ``values`` grouped by ``rows`` (offsets, values)."""
    order = np.argsort(rows, kind="stable")
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=off[1:])
    return off, values[order]


def _gather(off: np.ndarray, vals: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(owner, value)`` pairs listing row ``rows[i]``'s values for every ``i``."""
    deg = off[rows + 1] - off[rows]
    owner = np.repeat(np.arange(len(rows)), deg)
    start = np.repeat(off[rows], deg)
    pos = start + np.arange(int(deg.sum())) - np.repeat(np.cumsum(deg) - deg, deg)
    return owner, vals[pos]


def independent_set(
    rows_a: np.ndarray, rows_b: np.ndarray, nbr_off, nbr, nvr: int, prio: np.ndarray
) -> np.ndarray:
    """A maximal set of candidates (each touching region vertices ``rows_a``/``rows_b``)
    no two of which have an end within one edge of each other's ends: Luby's rounds --
    every candidate that is the priority minimum of its two-ring joins, its two-ring is
    blocked, repeat. Returns the indices of the chosen candidates."""
    big = np.iinfo(np.int64).max
    alive = np.ones(len(rows_a), dtype=bool)
    chosen: list[np.ndarray] = []
    blocked = np.zeros(nvr, dtype=bool)
    for _ in range(64):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        at_v = np.full(nvr, big, dtype=np.int64)
        np.minimum.at(at_v, rows_a[idx], prio[idx])
        np.minimum.at(at_v, rows_b[idx], prio[idx])
        owner, nb = _gather(nbr_off, nbr, np.arange(nvr))
        ring = at_v.copy()
        np.minimum.at(ring, owner, at_v[nb])
        win = idx[np.minimum(ring[rows_a[idx]], ring[rows_b[idx]]) == prio[idx]]
        if not len(win):
            break
        chosen.append(win)
        ends = np.concatenate([rows_a[win], rows_b[win]])
        blocked[ends] = True
        o2, nb2 = _gather(nbr_off, nbr, ends)
        blocked[nb2] = True
        alive &= ~blocked[rows_a] & ~blocked[rows_b]
    return np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)


def collapse_pass(
    sm, region: Region, low: float, high: float, max_ops: int, stats: dict | None = None
) -> int:
    """Collapse a maximal independent set of valid edges shorter than ``low`` whose two
    vertices are unlocked (see the module docs); no new edge may reach ``high``. Every
    check runs vectorized over all candidates before the set is chosen."""
    if not region.size:
        return 0
    ev = region.edge_v
    ia, ib = region.index(ev[:, 0]), region.index(ev[:, 1])
    free = ~region.locked[ia] & ~region.locked[ib] & ~region.feature & (region.count == 2)
    length = world_lengths(sm, ev[:, 0], ev[:, 1])
    cand = np.flatnonzero(free & (length < low))
    if not len(cand):
        return 0
    t = region.size
    tri, trv = region.tri, region.trv
    he = region.he_edge
    nvr = len(region.verts)
    # The two triangles of each candidate and the vertices opposite the edge.
    pick = np.zeros(len(ev), dtype=bool)
    pick[cand] = True
    hs = np.flatnonzero(pick[he])
    hs = hs[np.argsort(he[hs], kind="stable")]
    first, second = hs[0::2], hs[1::2]
    e_ids = he[first]
    r1, r2 = first % t, second % t
    opp1 = tri[r1, (first // t + 2) % 3]
    opp2 = tri[r2, (second // t + 2) % 3]
    a, b = ev[e_ids, 0], ev[e_ids, 1]
    la, lb = region.index(a), region.index(b)
    n = len(e_ids)
    ok = opp1 != opp2
    io1, io2 = region.index(opp1), region.index(opp2)
    valence_ok = (region.valence[io1] > 3) & (region.valence[io2] > 3)
    valence_ok &= region.active[io1] & region.active[io2]
    ok &= valence_ok
    # Neighbours (CSR over region vertices) and incident triangles (region rows).
    nbr_off, nbr = _csr(np.concatenate([ia, ib]), np.concatenate([ev[:, 1], ev[:, 0]]), nvr)
    nbr_local = region.index(nbr)
    tri_local = region.index(tri)
    inc_off, inc = _csr(tri_local.ravel(), np.repeat(np.arange(t), 3), nvr)
    # Link condition: a and b share exactly their two opposite vertices.
    oa, na = _gather(nbr_off, nbr, la)
    ob, nb = _gather(nbr_off, nbr, lb)
    key = np.concatenate([oa * np.int64(sm._nv + 1) + na, ob * np.int64(sm._nv + 1) + nb])
    uk, cnt = np.unique(key, return_counts=True)
    common = np.bincount(uk[cnt > 1] // np.int64(sm._nv + 1), minlength=n)
    ok &= common == 2
    # No new edge from the midpoint reaches the split threshold.
    target = (sm._pos[a] + sm._pos[b]) / 2.0
    m = np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3]
    owners = np.concatenate([oa, ob])
    others = np.concatenate([na, nb])
    d = (sm._pos[others] - target[owners]) @ m.T
    far = np.zeros(n)
    np.maximum.at(far, owners, np.sqrt((d * d).sum(axis=1)))
    ok &= far < high
    # No surrounding triangle folds (turns by 60 degrees or more) or degenerates.
    ta_o, ta_t = _gather(inc_off, inc, la)
    tb_o, tb_t = _gather(inc_off, inc, lb)
    own = np.concatenate([ta_o, tb_o])
    rows = np.concatenate([ta_t, tb_t])
    keep_pair = (rows != r1[own]) & (rows != r2[own])
    own, rows = own[keep_pair], rows[keep_pair]
    pts = sm._pos[tri[rows]].copy()
    before = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    moving = (tri[rows] == a[own][:, None]) | (tri[rows] == b[own][:, None])
    pts[moving] = np.repeat(target[own], 3, axis=0).reshape(-1, 3, 3)[moving]
    after = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    lb_ = np.sqrt((before * before).sum(axis=1))
    la_ = np.sqrt((after * after).sum(axis=1))
    good = (la_ > 1e-12 * np.maximum(lb_, 1e-300)) & (
        (before * after).sum(axis=1) > FOLD_COS * lb_ * la_
    )
    bad = np.zeros(n, dtype=bool)
    bad[own[~good]] = True
    ok &= ~bad
    if stats is not None:
        stats["candidates"] = stats.get("candidates", 0) + n
        stats["valid"] = stats.get("valid", 0) + int(ok.sum())
    valid = np.flatnonzero(ok)
    if not len(valid):
        return 0
    nv1 = np.int64(max(sm._nv, 1))
    prio = np.empty(n, dtype=np.int64)
    prio[valid] = np.argsort(np.argsort(scramble(a[valid] * nv1 + b[valid]), kind="stable"))
    sub = independent_set(la[valid], lb[valid], nbr_off, nbr_local, nvr, prio[valid])
    keep = valid[sub]
    if len(keep) > max_ops:
        keep = keep[np.argsort(prio[keep], kind="stable")[:max_ops]]
    keep = np.sort(keep)
    if not len(keep):
        return 0
    del nbr_local
    a, b, target = a[keep], b[keep], target[keep]
    gone_rows = np.concatenate([r1[keep], r2[keep]])
    owner_of_vertex = np.full(sm._nv, -1, dtype=np.int64)
    owner_of_vertex[a] = np.arange(len(keep))
    owner_of_vertex[b] = np.arange(len(keep))
    corner_owner = owner_of_vertex[tri]
    touch = (corner_owner >= 0).any(axis=1)
    touch[gone_rows] = False
    rows = np.flatnonzero(touch)
    # Each of a and b has one render vertex (no seam touches them): find it.
    render = np.full(sm._nv, -1, dtype=np.int64)
    rr, kk = np.nonzero(corner_owner >= 0)
    render[tri[rr, kk]] = trv[rr, kk]
    new_tri = tri[rows].copy()
    new_trv = trv[rows].copy()
    is_b = np.isin(new_tri, b)
    survivor = a[owner_of_vertex[new_tri[is_b]]]
    new_tri[is_b] = survivor
    new_trv[is_b] = render[survivor]
    uv_ids = uv_values = None
    if sm._ruv is not None:
        uv_ids = render[a]
        uv_values = (sm._ruv[render[a]] + sm._ruv[render[b]]) / 2.0
    src = region.tris[rows]
    apply_topology(
        sm,
        removed_tris=np.concatenate([region.tris[gone_rows], src]),
        new_tri=new_tri,
        new_trv=new_trv,
        new_group=sm._tgroup[src],
        new_chunk=sm._chunk_of[src],
        moved=a,
        moved_pos=target,
        removed_vertices=b,
        uv_ids=uv_ids,
        uv_values=uv_values,
    )
    return len(keep)


# ------------------------------------------------------------------------------------- #
# Flip
# ------------------------------------------------------------------------------------- #


def flip_pass(sm, region: Region, mode: str, max_ops: int) -> int:
    """Flip an independent set of interior, non-feature edges whose four vertices are
    active and unlocked: toward Delaunay (``"delaunay"``) or toward valence 6
    (``"valence"``)."""
    if not region.size:
        return 0
    t = region.size
    ev = region.edge_v
    he = region.he_edge
    tri, trv = region.tri, region.trv
    ok_edge = (region.count == 2) & ~region.feature
    ok_edge &= ~region.locked[region.index(ev[:, 0])] & ~region.locked[region.index(ev[:, 1])]
    cand_e = np.flatnonzero(ok_edge)
    if not len(cand_e):
        return 0
    pick = np.zeros(len(ev), dtype=bool)
    pick[cand_e] = True
    hs = np.flatnonzero(pick[he])
    hs = hs[np.argsort(he[hs], kind="stable")]
    first, second = hs[0::2], hs[1::2]
    e = he[first]
    r1, r2 = first % t, second % t
    k1, k2 = first // t, second // t
    a = tri[r1, k1]  # triangle 1 runs a -> b
    b = tri[r1, (k1 + 1) % 3]
    c = tri[r1, (k1 + 2) % 3]
    d = tri[r2, (k2 + 2) % 3]
    ic, id_ = region.index(c), region.index(d)
    usable = ~region.locked[ic] & ~region.locked[id_] & (c != d)
    pa, pb, pc, pd = sm._pos[a], sm._pos[b], sm._pos[c], sm._pos[d]
    m = np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3]

    def angle(at, u, v):
        x, y = (u - at) @ m.T, (v - at) @ m.T
        nx = np.sqrt((x * x).sum(axis=1))
        ny = np.sqrt((y * y).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.arccos(np.clip((x * y).sum(axis=1) / (nx * ny), -1.0, 1.0))

    if mode == "delaunay":
        gain = angle(pc, pa, pb) + angle(pd, pa, pb) - math.pi
    else:
        # Every vertex here is interior (unlocked, so off any boundary): target 6.
        val = region.valence
        ia, ib = region.index(a), region.index(b)
        before = (val[ia] - 6) ** 2 + (val[ib] - 6) ** 2 + (val[ic] - 6) ** 2 + (val[id_] - 6) ** 2
        after = (val[ia] - 7) ** 2 + (val[ib] - 7) ** 2 + (val[ic] - 5) ** 2 + (val[id_] - 5) ** 2
        gain = (before - after).astype(np.float64)
    good = usable & (gain > 1e-9)
    # The new triangles (a, d, c) and (d, b, c) must not fold against the old ones.
    n1 = np.cross(pb - pa, pc - pa)
    n2 = np.cross(pa - pb, pd - pb)
    m1 = np.cross(pd - pa, pc - pa)
    m2 = np.cross(pb - pd, pc - pd)
    ref = n1 + n2
    good &= ((m1 * ref).sum(axis=1) > 0) & ((m2 * ref).sum(axis=1) > 0)
    lm1 = np.sqrt((m1 * m1).sum(axis=1))
    lm2 = np.sqrt((m2 * m2).sum(axis=1))
    lr = np.sqrt((ref * ref).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        good &= ((m1 * ref).sum(axis=1) > FOLD_COS * lm1 * lr) & (
            (m2 * ref).sum(axis=1) > FOLD_COS * lm2 * lr
        )
    # The new edge must not exist already (edge keys are sorted: a binary search).
    nv = max(sm._nv, 1)
    keys = ev[:, 0] * nv + ev[:, 1]
    new_key = np.minimum(c, d) * nv + np.maximum(c, d)
    at = np.minimum(np.searchsorted(keys, new_key), len(keys) - 1)
    good &= keys[at] != new_key
    idx = np.flatnonzero(good)
    if not len(idx):
        return 0
    # Independent set: an edge flips only if it has the best gain among the candidates
    # sharing a vertex with it (valences of all four vertices change) -- for Delaunay
    # flips, sharing a triangle is enough.
    big = np.iinfo(np.int64).max
    nv1 = np.int64(max(sm._nv, 1))
    order = np.lexsort((scramble(ev[e[idx], 0] * nv1 + ev[e[idx], 1]), -gain[idx]))
    prio = np.full(len(e), big, dtype=np.int64)
    prio[idx[order]] = np.arange(len(idx))
    if mode == "valence":
        quads = [region.index(x[idx]) for x in (a, b, c, d)]
        best_v = np.full(len(region.verts), big, dtype=np.int64)
        for q in quads:
            np.minimum.at(best_v, q, prio[idx])
        winner = np.ones(len(idx), dtype=bool)
        for q in quads:
            winner &= best_v[q] == prio[idx]
        sel = idx[winner]
    else:
        best = np.full(t, big, dtype=np.int64)
        np.minimum.at(best, r1[idx], prio[idx])
        np.minimum.at(best, r2[idx], prio[idx])
        sel = idx[(best[r1[idx]] == prio[idx]) & (best[r2[idx]] == prio[idx])]
    if len(sel) > max_ops:
        sel = sel[np.argsort(prio[sel], kind="stable")[:max_ops]]
    if not len(sel):
        return 0
    ra, rb, rc = (
        trv[r1[sel], k1[sel]],
        trv[r1[sel], (k1[sel] + 1) % 3],
        trv[r1[sel], (k1[sel] + 2) % 3],
    )
    rd = trv[r2[sel], (k2[sel] + 2) % 3]
    new_tri = np.concatenate(
        [np.stack([a[sel], d[sel], c[sel]], axis=1), np.stack([d[sel], b[sel], c[sel]], axis=1)]
    )
    new_trv = np.concatenate([np.stack([ra, rd, rc], axis=1), np.stack([rd, rb, rc], axis=1)])
    src = np.concatenate([region.tris[r1[sel]], region.tris[r2[sel]]])
    apply_topology(
        sm,
        removed_tris=src,
        new_tri=new_tri,
        new_trv=new_trv,
        new_group=sm._tgroup[src],
        new_chunk=sm._chunk_of[src],
    )
    return len(sel)


# ------------------------------------------------------------------------------------- #
# Relax
# ------------------------------------------------------------------------------------- #


def relax_pass(sm, region: Region, strength: float = 0.5) -> int:
    """Tangential relaxation of the active, unlocked vertices (positions through the
    sculpt mesh's journaled moves; UVs re-sampled in the old one-ring)."""
    if not region.size:
        return 0
    free = np.flatnonzero(~region.locked)
    if not len(free):
        return 0
    verts = region.verts
    ev = region.edge_v
    ia, ib = region.index(ev[:, 0]), region.index(ev[:, 1])
    # Area weights: half the length of the vertex's area-weighted normal sum.
    area = 0.5 * np.sqrt((sm._vsum[verts] ** 2).sum(axis=1))
    rows = np.concatenate([ia, ib])
    nbrs = np.concatenate([ib, ia])
    w = area[nbrs]
    wsum = np.bincount(rows, weights=w, minlength=len(verts))
    g = np.zeros((len(verts), 3))
    for k in range(3):
        g[:, k] = np.bincount(rows, weights=w * sm._pos[verts[nbrs], k], minlength=len(verts))
    ids = verts[free]
    ok = wsum[free] > 0
    ids, fi = ids[ok], free[ok]
    if not len(ids):
        return 0
    centroid = g[fi] / wsum[fi, None]
    n = sm.vertex_normals(ids)
    delta = centroid - sm._pos[ids]
    delta -= (delta * n).sum(axis=1)[:, None] * n
    target = sm._pos[ids] + strength * delta
    if sm._ruv is not None:
        _resample_uvs(sm, region, ids, target)
    sm._move(ids, target, region.tris)
    return len(ids)


def _resample_uvs(sm, region: Region, ids: np.ndarray, target: np.ndarray) -> None:
    """New UVs for moved vertices from the old one-ring triangle the target falls in
    (each vertex has one render vertex: it touches no seam)."""
    tri, trv = region.tri, region.trv
    mark = np.full(sm._nv, -1, dtype=np.int64)
    mark[ids] = np.arange(len(ids))
    owner_rows, owner_k = np.nonzero(mark[tri] >= 0)
    p = sm._pos[tri[owner_rows]]
    q = target[mark[tri[owner_rows, owner_k]]]
    # Barycentric coordinates of q in each incident triangle (projected on its plane).
    v0, v1 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    v2 = q - p[:, 0]
    d00 = (v0 * v0).sum(axis=1)
    d01 = (v0 * v1).sum(axis=1)
    d11 = (v1 * v1).sum(axis=1)
    d20 = (v2 * v0).sum(axis=1)
    d21 = (v2 * v1).sum(axis=1)
    denom = d00 * d11 - d01 * d01
    with np.errstate(invalid="ignore", divide="ignore"):
        bv = (d11 * d20 - d01 * d21) / denom
        bw = (d00 * d21 - d01 * d20) / denom
        bu = 1.0 - bv - bw  # not finite for degenerate triangles: never used
    inside = (bu >= -1e-9) & (bv >= -1e-9) & (bw >= -1e-9) & np.isfinite(bu)
    if not inside.any():
        return
    rows, k = owner_rows[inside], owner_k[inside]
    bary = np.stack([bu, bv, bw], axis=1)[inside]
    vertex = tri[rows, k]
    first = np.unique(vertex, return_index=True)[1]
    rows, k, bary, vertex = rows[first], k[first], bary[first], vertex[first]
    uv = (sm._ruv[trv[rows]] * bary[:, :, None]).sum(axis=1)
    render = trv[rows, k]
    sm._log(("uv", render, sm._ruv[render].copy()))
    sm._ruv[render] = uv
