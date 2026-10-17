"""Sparse symmetric positive definite solves: nested-dissection order and LDL^T.

The parameterizations of :mod:`aether3d.uv.param` solve sparse SPD systems (LSCM once,
SLIM once per iteration with the same sparsity). This module factors them directly:

* **Ordering.** Geometric nested dissection (George 1973): the unknowns' points are split
  at the median of their longest extent, the vertices of one half adjacent to the other
  form the separator, which is numbered last; each half recursively. On mesh graphs the
  fill is O(n log n). Disconnected blocks (one per chart) are ordered block by block, so
  the factor never couples two charts.
* **Symbolic analysis** (once per sparsity): the elimination tree and the pattern of
  every row of ``L`` (ascending: an ascending order is a topological order of the tree,
  since a parent always has a larger index).
* **Numeric factorization** ``A = L D L^T`` by rows ("up-looking", Davis 2005): row ``k``
  of ``L`` is a sparse triangular solve over its pattern.
* **Solves**: forward by columns of ``L``, diagonal, backward by rows of ``L``.

The factorization and the solves run in ``aether3d-native`` when it is installed
(:mod:`aether3d._native`); the NumPy code here performs exactly the same floating-point
operations in the same order, so both give the same bits (``tests/test_native.py``).
Matrices are CSC (equivalently CSR: they are symmetric) with both triangles stored and
sorted row indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _native
from ..errors import TexturingError

LEAF = 48
"""Nested dissection stops splitting at this many points."""


@dataclass(frozen=True)
class CSC:
    """A sparse symmetric matrix: both triangles, rows sorted within each column."""

    n: int
    indptr: np.ndarray  # (n + 1,) int64
    indices: np.ndarray  # (nnz,) int64
    data: np.ndarray  # (nnz,) float64


def assemble(n: int, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> CSC:
    """The CSC matrix of summed COO entries (duplicates are added in input order).

    The sum of duplicates is computed in a fixed order (input order within each
    ``(row, col)``), so the result does not depend on the platform.
    """
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    key = cols * n + rows
    order = np.argsort(key, kind="stable")
    key = key[order]
    values = values[order]
    if key.size == 0:
        return CSC(n, np.zeros(n + 1, dtype=np.int64), key, values)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    summed = _segment_sums(values, starts)
    unique = key[starts]
    col = unique // n
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, col + 1, 1)
    return CSC(n, np.cumsum(indptr), unique % n, summed)


def _segment_sums(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sums of ``values[starts[i]:starts[i+1]]``, each added left to right."""
    out = values[starts].copy()
    lengths = np.diff(np.r_[starts, values.shape[0]])
    k = 1
    while True:
        more = lengths > k
        if not more.any():
            return out
        idx = np.flatnonzero(more)
        out[idx] = out[idx] + values[starts[idx] + k]
        k += 1


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def nested_dissection(
    points: np.ndarray, indptr: np.ndarray, indices: np.ndarray, blocks: np.ndarray | None = None
) -> np.ndarray:
    """A fill-reducing order of the graph's nodes (``perm[new] = old``).

    ``points`` (``(n, d)``) are the nodes' coordinates, ``indptr``/``indices`` the graph's
    adjacency (CSR, symmetric), ``blocks`` an optional block of every node (the graph has no
    edge between blocks); blocks are ordered one after another.
    """
    n = points.shape[0]
    points = np.asarray(points, dtype=np.float64)
    if blocks is None:
        groups = [np.arange(n, dtype=np.int64)]
    else:
        order = np.argsort(blocks, kind="stable")
        bounds = np.flatnonzero(np.r_[True, blocks[order][1:] != blocks[order][:-1], True])
        groups = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    out: list[np.ndarray] = []
    side = np.zeros(n, dtype=np.int8)  # scratch: 1 left, 2 right of the current split
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    for group in groups:
        stack: list[tuple[np.ndarray, bool]] = [(group, False)]
        while stack:
            nodes, emit = stack.pop()
            if emit or nodes.shape[0] <= LEAF:
                out.append(np.sort(nodes))
                continue
            p = points[nodes]
            axis = int(np.argmax(p.max(axis=0) - p.min(axis=0)))
            order = np.lexsort((nodes, p[:, axis]))
            half = nodes.shape[0] // 2
            left, right = nodes[order[:half]], nodes[order[half:]]
            side[left] = 1
            side[right] = 2
            # Separator: left nodes with a neighbour on the right.
            lo, hi = indptr[left], indptr[left + 1]
            counts = hi - lo
            if counts.sum():
                pos = np.repeat(lo - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
                nbr = indices[pos]
                owner = src[pos]
                cut = np.unique(owner[side[nbr] == 2])
            else:
                cut = np.zeros(0, dtype=np.int64)
            side[left] = 0
            side[right] = 0
            rest = left if cut.shape[0] == 0 else np.setdiff1d(left, cut, assume_unique=True)
            # Popped in reverse: left subtree, right subtree, then the separator.
            stack.append((cut, True))
            stack.append((right, False))
            stack.append((rest, False))
    perm = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
    return perm.astype(np.int64)


# --------------------------------------------------------------------------- #
# Symbolic analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Symbolic:
    """The structure of ``L`` for one sparsity and order (reused by every numeric factor)."""

    n: int
    perm: np.ndarray  # perm[new] = old
    pinv: np.ndarray  # pinv[old] = new
    parent: np.ndarray  # elimination tree (-1: root)
    lp: np.ndarray  # (n + 1,) column pointers of L
    li: np.ndarray  # row indices of L, ascending within each column
    rp: np.ndarray  # (n + 1,) pointers of the row patterns
    rj: np.ndarray  # row k's columns, ascending
    rpos: np.ndarray  # where each row entry lives in li/lx
    ap: np.ndarray  # the permuted matrix's upper part (rows <= k), CSC
    ai: np.ndarray
    amap: np.ndarray  # source entry (in the input matrix) of each upper entry


def _structure_numpy(n: int, ap: np.ndarray, ai: np.ndarray, ac: np.ndarray):
    """Elimination tree and column patterns of ``L`` (the same arrays as the native
    ``ldl_etree`` and ``ldl_pattern``: the tree is unique and patterns are ascending)."""
    # Elimination tree (Liu), with path compression through ``ancestor``.
    ai_list = ai.tolist()
    ap_list = ap.tolist()
    par = [-1] * n
    anc = [-1] * n
    for k in range(n):
        for p in range(ap_list[k], ap_list[k + 1]):
            i = ai_list[p]
            while i != -1 and i < k:
                nxt = anc[i]
                anc[i] = k
                if nxt == -1:
                    par[i] = k
                i = nxt
    parent = np.array(par, dtype=np.int64)
    # Column patterns: struct(L_j) = rows of A_j below j, merged with the children's.
    children: list[list[int]] = [[] for _ in range(n)]
    for j, pj in enumerate(par):
        if pj != -1:
            children[pj].append(j)
    lower = ai != ac
    lower_rows = ac[lower]
    lower_cols = ai[lower]
    lorder = np.lexsort((lower_rows, lower_cols))
    lower_rows = lower_rows[lorder]
    lower_cols = lower_cols[lorder]
    lp_a = np.zeros(n + 1, dtype=np.int64)
    np.add.at(lp_a, lower_cols + 1, 1)
    lp_a = np.cumsum(lp_a)
    cols: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * n
    for j in range(n):
        parts = [lower_rows[lp_a[j] : lp_a[j + 1]]]
        for c_ in children[j]:
            s_ = cols[c_]
            parts.append(s_[s_ > j])
        cols[j] = np.unique(np.concatenate(parts)) if len(parts) > 1 else parts[0]
    counts = np.array([c_.shape[0] for c_ in cols], dtype=np.int64)
    lp = np.zeros(n + 1, dtype=np.int64)
    lp[1:] = np.cumsum(counts)
    li = np.concatenate(cols) if n else np.zeros(0, dtype=np.int64)
    return parent, lp, li


def analyze(matrix: CSC, perm: np.ndarray) -> Symbolic:
    """The elimination tree and the patterns of ``L`` for ``P A P^T``."""
    n = matrix.n
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (n,) or not np.array_equal(np.sort(perm), np.arange(n)):
        raise TexturingError("SOLVER_FAILED", "the ordering is not a permutation", "perm")
    pinv = np.empty(n, dtype=np.int64)
    pinv[perm] = np.arange(n, dtype=np.int64)
    # Upper part of the permuted matrix, column by column, rows ascending.
    col_old = np.repeat(np.arange(n, dtype=np.int64), np.diff(matrix.indptr))
    r = pinv[matrix.indices]
    c = pinv[col_old]
    keep = r <= c
    entries = np.flatnonzero(keep)
    order = np.lexsort((r[entries], c[entries]))
    entries = entries[order]
    ai = r[entries]
    ac = c[entries]
    ap = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ap, ac + 1, 1)
    ap = np.cumsum(ap)
    native = _native.module()
    if native is not None and hasattr(native, "ldl_etree"):
        parent = np.empty(n, dtype=np.int64)
        counts = np.empty(n, dtype=np.int64)
        native.ldl_etree(ap, ai, parent, counts)
        lp = np.zeros(n + 1, dtype=np.int64)
        lp[1:] = np.cumsum(counts)
        li = np.empty(int(lp[-1]), dtype=np.int64)
        native.ldl_pattern(ap, ai, parent, lp, li)
    else:
        parent, lp, li = _structure_numpy(n, ap, ai, ac)
        counts = np.diff(lp)
    # Row patterns: transpose of the column structure.
    col_of = np.repeat(np.arange(n, dtype=np.int64), counts)
    rorder = np.lexsort((col_of, li))
    rj = col_of[rorder]
    rpos = rorder.astype(np.int64)
    rp = np.zeros(n + 1, dtype=np.int64)
    np.add.at(rp, li + 1, 1)
    rp = np.cumsum(rp)
    return Symbolic(n, perm, pinv, parent, lp, li, rp, rj, rpos, ap, ai, entries)


# --------------------------------------------------------------------------- #
# Numeric factorization and solves
# --------------------------------------------------------------------------- #


@dataclass
class Factor:
    """``P A P^T = L D L^T`` (``lx`` in the order of ``Symbolic.li``)."""

    symbolic: Symbolic
    lx: np.ndarray
    d: np.ndarray

    def solve(self, b: np.ndarray) -> np.ndarray:
        """``A^-1 b`` for ``b`` of shape ``(n,)`` or ``(n, m)``."""
        return solve(self, b)


def factor(symbolic: Symbolic, matrix: CSC) -> Factor:
    """The numeric factorization of ``matrix`` (with ``symbolic``'s sparsity and order).

    Raises ``TexturingError("SOLVER_FAILED")`` when a pivot is not positive (the matrix is
    not positive definite).
    """
    s = symbolic
    ax = np.ascontiguousarray(matrix.data[s.amap], dtype=np.float64)
    native = _native.module()
    if native is not None and hasattr(native, "ldl_factor"):
        lx = np.empty(s.li.shape[0], dtype=np.float64)
        d = np.empty(s.n, dtype=np.float64)
        bad = native.ldl_factor(s.ap, s.ai, ax, s.lp, s.li, s.rp, s.rj, s.rpos, lx, d)
    else:
        lx, d, bad = _factor_numpy(s, ax)
    if bad >= 0:
        raise TexturingError(
            "SOLVER_FAILED",
            f"the system is not positive definite (pivot {bad} of {s.n})",
            details={"pivot": int(bad), "size": s.n},
        )
    return Factor(s, lx, d)


def _factor_numpy(s: Symbolic, ax: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Up-looking LDL^T: the operations of ``native/src/ldl.rs``, in the same order."""
    n = s.n
    lx = np.zeros(s.li.shape[0], dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    filled = s.lp[:-1].copy()  # next free slot of each column
    lp, li, ap, ai, rp, rj = s.lp, s.li, s.ap, s.ai, s.rp, s.rj
    d_list = [0.0] * n
    for k in range(n):
        a0, a1 = ap[k], ap[k + 1]
        rows = ai[a0:a1]
        y[rows] += ax[a0:a1]
        dk = float(y[k])
        y[k] = 0.0
        for i in rj[rp[k] : rp[k + 1]].tolist():
            yi = float(y[i])
            y[i] = 0.0
            c0, c1 = lp[i], filled[i]
            if c1 > c0:
                y[li[c0:c1]] -= lx[c0:c1] * yi
            lki = yi / d_list[i]
            dk -= lki * yi
            lx[c1] = lki
            filled[i] = c1 + 1
        if not dk > 0.0:
            return lx, d, k
        d_list[k] = dk
    d[:] = d_list
    return lx, d, -1


def solve(f: Factor, b: np.ndarray) -> np.ndarray:
    """``A^-1 b``: permute, forward by columns, diagonal, backward by rows, permute back."""
    s = f.symbolic
    b = np.asarray(b, dtype=np.float64)
    single = b.ndim == 1
    x = np.ascontiguousarray((b[:, None] if single else b)[s.perm], dtype=np.float64)
    native = _native.module()
    if native is not None and hasattr(native, "ldl_solve"):
        native.ldl_solve(s.lp, s.li, f.lx, s.rp, s.rj, s.rpos, f.d, x)
    else:
        _solve_numpy(s, f.lx, f.d, x)
    out = np.empty_like(x)
    out[s.perm] = x
    return out[:, 0] if single else out


def _solve_numpy(s: Symbolic, lx: np.ndarray, d: np.ndarray, x: np.ndarray) -> None:
    n = s.n
    lp, li, rp, rj, rpos = s.lp, s.li, s.rp, s.rj, s.rpos
    for j in range(n):
        c0, c1 = lp[j], lp[j + 1]
        if c1 > c0:
            x[li[c0:c1]] -= lx[c0:c1, None] * x[j]
    x /= d[:, None]
    for i in range(n - 1, -1, -1):
        r0, r1 = rp[i], rp[i + 1]
        if r1 > r0:
            x[rj[r0:r1]] -= lx[rpos[r0:r1], None] * x[i]


def spd_solve(
    matrix: CSC, b: np.ndarray, points: np.ndarray, blocks: np.ndarray | None = None
) -> np.ndarray:
    """Order, analyze, factor and solve in one call (``points``: one per unknown)."""
    perm = nested_dissection(points, matrix.indptr, matrix.indices, blocks)
    return factor(analyze(matrix, perm), matrix).solve(b)
