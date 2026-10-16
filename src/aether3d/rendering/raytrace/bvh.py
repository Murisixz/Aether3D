"""Bounding volume hierarchies: binned-SAH construction over primitive boxes, vectorized.

A :class:`BVH` is a binary tree over primitives (triangles of one mesh, or the
instances of a scene) given only by their axis-aligned bounding boxes. Each
inner node stores the boxes of its two children ("children in the parent"),
so traversal tests both children at once and never visits a child whose box
the ray misses.

Construction is the binned surface area heuristic (Wald 2007, "On fast
Construction of SAH-based Bounding Volume Hierarchies"), run level by level
for all nodes of a level at once with NumPy:

1. every node's primitives are binned into :data:`BINS` equal slots along the
   longest axis of their centroid bounds;
2. prefix and suffix sweeps over the bins give, for every candidate split,
   ``SA(left) * N(left) + SA(right) * N(right)`` (``SA``: surface area of the
   bounds, ``N``: primitive count);
3. the cheapest split partitions the node's primitive range (a stable sort,
   so construction is deterministic). Where all centroids fall in one bin,
   the range is split at its middle (object median by index), which always
   makes progress.

Nodes with at most ``leaf_size`` primitives become leaves. The tree depth is
bounded by about ``2 log2(n)`` (median fallback) and checked against
:data:`MAX_DEPTH`, which sizes traversal stacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BINS = 16
"""Bins per node and level of the SAH sweep."""
LEAF_SIZE = 4
"""Most primitives in a leaf (fewer box tests vs. more primitive tests per leaf)."""
MAX_DEPTH = 96
"""Deepest tree construction allows (the median fallback keeps trees far shallower)."""
NODE_CHUNK = 8192
"""Nodes binned together during construction (bounds temporary memory)."""


@dataclass(frozen=True)
class BVH:
    """A binary BVH over ``n`` primitives (node 0 is the root).

    * ``child`` -- ``(nodes, 2)`` int64 children of inner nodes (``-1``: leaf).
    * ``child_lo``, ``child_hi`` -- ``(nodes, 2, 3)`` boxes of the children
      of inner nodes (in the dtype of the primitive bounds).
    * ``first``, ``count`` -- ``(nodes,)`` primitive range of leaves in
      :attr:`order` (``count == 0`` for inner nodes).
    * ``order`` -- ``(n,)`` primitive indices in leaf order.
    * ``lo``, ``hi`` -- ``(3,)`` bounds of the whole tree (empty: ``+inf``/``-inf``).
    * ``depth`` -- deepest leaf (root: 0).
    """

    child: np.ndarray
    child_lo: np.ndarray
    child_hi: np.ndarray
    first: np.ndarray
    count: np.ndarray
    order: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    depth: int

    @property
    def node_count(self) -> int:
        return int(self.child.shape[0])

    @property
    def leaf_count(self) -> int:
        return int(np.count_nonzero(self.child[:, 0] < 0))

    @property
    def nbytes(self) -> int:
        return int(
            sum(
                a.nbytes for a in (self.child, self.child_lo, self.child_hi, self.first, self.count)
            )
            + self.order.nbytes
        )

    def stats(self) -> dict[str, int | float]:
        leaves = self.child[:, 0] < 0
        sizes = self.count[leaves]
        return {
            "primitives": int(self.order.shape[0]),
            "nodes": self.node_count,
            "leaves": int(np.count_nonzero(leaves)),
            "depth": self.depth,
            "max_leaf_size": int(sizes.max(initial=0)),
            "mean_leaf_size": float(sizes.mean()) if sizes.size else 0.0,
            "bytes": self.nbytes,
        }


def _surface_area(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    d = np.maximum(hi - lo, 0.0)
    return 2.0 * (d[..., 0] * d[..., 1] + d[..., 1] * d[..., 2] + d[..., 2] * d[..., 0])


def _ranges(starts: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions covered by consecutive ranges, the range of each, and range offsets."""
    offsets = np.cumsum(sizes) - sizes
    member = np.repeat(np.arange(sizes.size), sizes)
    pos = starts[member] + (np.arange(int(sizes.sum())) - offsets[member])
    return pos, member, offsets


def _split(lo, hi, centroid, order, starts, sizes) -> np.ndarray:
    """Binned-SAH split of the nodes with ranges ``(starts, sizes)`` of ``order``.

    Partitions each range in place (left side first, stable) and returns the
    number of primitives that went left in each node.
    """
    pos, member, offsets = _ranges(starts, sizes)
    total = pos.size
    k = sizes.size
    prim = order[pos]
    c = centroid[prim]
    c_lo = np.minimum.reduceat(c, offsets, axis=0)
    extent = np.maximum.reduceat(c, offsets, axis=0) - c_lo
    axis = np.argmax(extent, axis=1)
    a = axis[member]
    span = extent[member, a]
    coord = c[np.arange(total), a] - c_lo[member, a]
    del c
    scale = np.where(span > 0.0, BINS / np.where(span > 0.0, span, 1.0), 0.0)
    bins = np.minimum((coord * scale).astype(np.int64), BINS - 1)
    np.maximum(bins, 0, out=bins)
    del coord, scale, span, a
    key = member * BINS + bins
    bin_count = np.bincount(key, minlength=k * BINS).reshape(k, BINS)
    sort = np.argsort(key, kind="stable")
    unique, first = np.unique(key[sort], return_index=True)
    del key
    sorted_prim = prim[sort]
    del sort
    bin_lo = np.full((k * BINS, 3), np.inf)
    bin_hi = np.full((k * BINS, 3), -np.inf)
    bin_lo[unique] = np.minimum.reduceat(lo[sorted_prim], first, axis=0)
    bin_hi[unique] = np.maximum.reduceat(hi[sorted_prim], first, axis=0)
    del sorted_prim, unique, first
    bin_lo = bin_lo.reshape(k, BINS, 3)
    bin_hi = bin_hi.reshape(k, BINS, 3)
    left = _surface_area(
        np.minimum.accumulate(bin_lo, axis=1)[:, :-1], np.maximum.accumulate(bin_hi, axis=1)[:, :-1]
    )
    right = _surface_area(
        np.minimum.accumulate(bin_lo[:, ::-1], axis=1)[:, ::-1][:, 1:],
        np.maximum.accumulate(bin_hi[:, ::-1], axis=1)[:, ::-1][:, 1:],
    )
    del bin_lo, bin_hi
    left_n = np.cumsum(bin_count, axis=1)[:, :-1]
    right_n = np.cumsum(bin_count[:, ::-1], axis=1)[:, ::-1][:, 1:]
    cost = np.where((left_n > 0) & (right_n > 0), left * left_n + right * right_n, np.inf)
    best = np.argmin(cost, axis=1)
    binned = np.isfinite(cost[np.arange(k), best])
    del left, right, left_n, right_n, cost
    # Left side: bins up to the best split; nodes without a usable bin split take the
    # first half of their range (object median by position, deterministic).
    rank = np.arange(total) - offsets[member]
    goes_left = np.where(binned[member], bins <= best[member], rank < sizes[member] // 2)
    del rank, bins
    perm = np.argsort(member * 2 + (~goes_left).astype(np.int64), kind="stable")
    order[pos] = prim[perm]
    return np.bincount(member[goes_left], minlength=k)


def build_bvh(lo: np.ndarray, hi: np.ndarray, leaf_size: int = LEAF_SIZE) -> BVH:
    """Build a BVH over primitives with bounds ``lo``, ``hi`` (``(n, 3)`` each).

    Node boxes keep the bounds' dtype (float32 or float64; boxes are exact
    minima/maxima of the given bounds). Primitives with non-finite bounds must
    be removed by the caller. Each level is processed in chunks of at most
    :data:`NODE_CHUNK` nodes, which bounds the binning temporaries without
    changing the tree.
    """
    dtype = np.float32 if lo.dtype == np.float32 and hi.dtype == np.float32 else np.float64
    lo = np.ascontiguousarray(lo, dtype=dtype)
    hi = np.ascontiguousarray(hi, dtype=dtype)
    n = lo.shape[0]
    if n == 0:
        inf = np.full((1, 2, 3), np.inf, dtype=dtype)
        return BVH(
            np.full((1, 2), -1, dtype=np.int64),
            inf,
            -inf,
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.full(3, np.inf),
            np.full(3, -np.inf),
            0,
        )
    centroid = (lo.astype(np.float64) + hi) * 0.5
    order = np.arange(n, dtype=np.int64)
    nodes = 1
    # Per level: node ids with their bounds, leaves with their ranges, inner nodes with
    # their first child (children are allocated in pairs).
    out_id, out_lo, out_hi = [], [], []
    leaf_id, leaf_first, leaf_count = [], [], []
    inner_id, inner_kid = [], []
    # The nodes of the current level: id and primitive range [start, end) in `order`.
    act_id = np.zeros(1, dtype=np.int64)
    act_start = np.zeros(1, dtype=np.int64)
    act_end = np.array([n], dtype=np.int64)
    depth = 0
    while True:
        size = act_end - act_start
        level_lo = np.empty((act_id.size, 3), dtype=dtype)
        level_hi = np.empty((act_id.size, 3), dtype=dtype)
        for c0 in range(0, act_id.size, NODE_CHUNK):
            c1 = min(c0 + NODE_CHUNK, act_id.size)
            pos, _, offsets = _ranges(act_start[c0:c1], size[c0:c1])
            prim = order[pos]
            level_lo[c0:c1] = np.minimum.reduceat(lo[prim], offsets, axis=0)
            level_hi[c0:c1] = np.maximum.reduceat(hi[prim], offsets, axis=0)
        out_id.append(act_id)
        out_lo.append(level_lo)
        out_hi.append(level_hi)
        split = size > leaf_size
        leaves = ~split
        leaf_id.append(act_id[leaves])
        leaf_first.append(act_start[leaves])
        leaf_count.append(size[leaves])
        if not split.any():
            break
        if depth >= MAX_DEPTH:  # pragma: no cover - the median fallback bounds depth
            raise RuntimeError("BVH construction exceeded the maximum depth")
        sid = np.flatnonzero(split)
        n_left = np.empty(sid.size, dtype=np.int64)
        for c0 in range(0, sid.size, NODE_CHUNK):
            part = sid[c0 : c0 + NODE_CHUNK]
            n_left[c0 : c0 + part.size] = _split(
                lo, hi, centroid, order, act_start[part], size[part]
            )
        kids = nodes + 2 * np.arange(sid.size, dtype=np.int64)
        nodes += 2 * sid.size
        inner_id.append(act_id[sid])
        inner_kid.append(kids)
        mid = act_start[sid] + n_left
        act_id = np.concatenate([kids, kids + 1])
        act_start = np.concatenate([act_start[sid], mid])
        act_end = np.concatenate([mid, act_end[sid]])
        depth += 1
    node_lo = np.empty((nodes, 3), dtype=dtype)
    node_hi = np.empty((nodes, 3), dtype=dtype)
    ids = np.concatenate(out_id)
    node_lo[ids] = np.concatenate(out_lo)
    node_hi[ids] = np.concatenate(out_hi)
    del out_lo, out_hi
    child = np.full((nodes, 2), -1, dtype=np.int64)
    first = np.zeros(nodes, dtype=np.int64)
    count = np.zeros(nodes, dtype=np.int64)
    if inner_id:
        parents = np.concatenate(inner_id)
        kid = np.concatenate(inner_kid)
        child[parents, 0] = kid
        child[parents, 1] = kid + 1
    ids = np.concatenate(leaf_id)
    first[ids] = np.concatenate(leaf_first)
    count[ids] = np.concatenate(leaf_count)
    inner = child[:, 0] >= 0
    child_lo = np.full((nodes, 2, 3), np.inf, dtype=dtype)
    child_hi = np.full((nodes, 2, 3), -np.inf, dtype=dtype)
    child_lo[inner] = node_lo[child[inner]]
    child_hi[inner] = node_hi[child[inner]]
    return BVH(
        child,
        child_lo,
        child_hi,
        first,
        count,
        order,
        node_lo[0].astype(np.float64),
        node_hi[0].astype(np.float64),
        depth,
    )


def triangle_bounds(positions: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of indexed triangles (in the positions' dtype, one corner at a time)."""
    corner = positions[triangles[:, 0]]
    lo = corner.copy()
    hi = corner
    for k in (1, 2):
        corner = positions[triangles[:, k]]
        np.minimum(lo, corner, out=lo)
        np.maximum(hi, corner, out=hi)
    return lo, hi


def morton_codes(points: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None):
    """63-bit Morton (Z-order) codes of points quantized to 2^21 cells per axis of the
    box ``lo``..``hi`` (default: the points' bounds)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros(0, dtype=np.uint64)
    lo = p.min(axis=0) if lo is None else np.asarray(lo, dtype=np.float64)
    hi = p.max(axis=0) if hi is None else np.asarray(hi, dtype=np.float64)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    q = np.clip((p - lo) / span * (2**21 - 1), 0, 2**21 - 1).astype(np.uint64)

    def spread(x: np.ndarray) -> np.ndarray:
        x = x & np.uint64(0x1FFFFF)
        x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        return (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)

    return spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))


def build_bvh_morton(lo: np.ndarray, hi: np.ndarray, leaf_size: int = 8) -> BVH:
    """A BVH over primitive bounds built from the Morton order of their centroids, each
    node's range split at its middle (a linear BVH). Leaf boxes are reduced once and
    inner boxes merged bottom-up, so building costs one sort and O(n): several times
    faster than :func:`build_bvh`'s binned SAH, with looser boxes. Used where builds are
    frequent and queries few (sculpting, projections); the renderers keep the SAH tree."""
    dtype = np.float32 if lo.dtype == np.float32 and hi.dtype == np.float32 else np.float64
    lo = np.ascontiguousarray(lo, dtype=dtype)
    hi = np.ascontiguousarray(hi, dtype=dtype)
    n = lo.shape[0]
    if n == 0:
        return build_bvh(lo, hi, leaf_size)
    centroid = (lo.astype(np.float64) + hi) * 0.5
    order = np.argsort(morton_codes(centroid), kind="stable").astype(np.int64)
    del centroid
    # Topology: ranges split at their middle, level by level (integers only).
    levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (ids, start, end)
    act_id = np.zeros(1, dtype=np.int64)
    act_start = np.zeros(1, dtype=np.int64)
    act_end = np.array([n], dtype=np.int64)
    nodes = 1
    child_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    while True:
        levels.append((act_id, act_start, act_end))
        size = act_end - act_start
        split = size > leaf_size
        if not split.any():
            break
        sid = np.flatnonzero(split)
        kids = nodes + 2 * np.arange(sid.size, dtype=np.int64)
        nodes += 2 * sid.size
        child_pairs.append((act_id[sid], kids))
        mid = act_start[sid] + size[sid] // 2
        act_id = np.concatenate([kids, kids + 1])
        act_start = np.concatenate([act_start[sid], mid])
        act_end = np.concatenate([mid, act_end[sid]])
    child = np.full((nodes, 2), -1, dtype=np.int64)
    for parents, kids in child_pairs:
        child[parents, 0] = kids
        child[parents, 1] = kids + 1
    first = np.zeros(nodes, dtype=np.int64)
    count = np.zeros(nodes, dtype=np.int64)
    node_lo = np.empty((nodes, 3), dtype=dtype)
    node_hi = np.empty((nodes, 3), dtype=dtype)
    # Leaves: one reduction over the primitives in Morton order.
    leaf_ids, leaf_start, leaf_end = [], [], []
    for ids, start, end in levels:
        leaf = (end - start) <= leaf_size
        leaf_ids.append(ids[leaf])
        leaf_start.append(start[leaf])
        leaf_end.append(end[leaf])
    leaf_ids_a = np.concatenate(leaf_ids)
    leaf_start_a = np.concatenate(leaf_start)
    leaf_end_a = np.concatenate(leaf_end)
    by_start = np.argsort(leaf_start_a, kind="stable")
    leaf_ids_a, leaf_start_a, leaf_end_a = (
        leaf_ids_a[by_start],
        leaf_start_a[by_start],
        leaf_end_a[by_start],
    )
    first[leaf_ids_a] = leaf_start_a
    count[leaf_ids_a] = leaf_end_a - leaf_start_a
    node_lo[leaf_ids_a] = np.minimum.reduceat(lo[order], leaf_start_a, axis=0)
    node_hi[leaf_ids_a] = np.maximum.reduceat(hi[order], leaf_start_a, axis=0)
    # Inner nodes, deepest level first: the union of their children.
    for parents, kids in reversed(child_pairs):
        node_lo[parents] = np.minimum(node_lo[kids], node_lo[kids + 1])
        node_hi[parents] = np.maximum(node_hi[kids], node_hi[kids + 1])
    inner = child[:, 0] >= 0
    child_lo = np.full((nodes, 2, 3), np.inf, dtype=dtype)
    child_hi = np.full((nodes, 2, 3), -np.inf, dtype=dtype)
    child_lo[inner] = node_lo[child[inner]]
    child_hi[inner] = node_hi[child[inner]]
    return BVH(
        child,
        child_lo,
        child_hi,
        first,
        count,
        order,
        node_lo[0].astype(np.float64),
        node_hi[0].astype(np.float64),
        len(levels) - 1,
    )
