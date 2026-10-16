"""Batched chart parameterization: LSCM, Tutte embeddings and SLIM.

All charts of an unwrap are solved together: their triangles form one block-diagonal
system (one block per chart, ordered block by block), so a thousand charts cost one
assembly and one factorization, not a thousand Python round trips.

* **Local frames.** Each triangle is laid flat isometrically (``q0 = 0``, ``q1`` on the x
  axis, counter-clockwise); ``g_i`` are the gradients of its barycentric functions, so a
  map's Jacobian is ``J = [sum u_i g_i ; sum v_i g_i]`` (rows: gradients of u and v).
* **LSCM** (Levy et al. 2002) minimizes ``sum A_t |grad u^perp - grad v|^2`` with two
  pinned vertices per chart (the farthest boundary pair), a conformal map.
* **Tutte** (Floater 1997): boundary on a circle by arc length, every interior vertex at
  the average of its neighbours; injective for 3-connected disks. Used when LSCM inverts a
  triangle.
* **SLIM** (Rabinovich et al. 2017) minimizes the symmetric Dirichlet energy
  ``sum A_t (|J|^2 + |J^-1|^2)`` (isometric distortion; infinite for inverted triangles):
  per iteration the closed-form 2x2 SVD of every Jacobian gives the closest rotation
  ``R`` and weights ``W = U diag(w) U^T`` (``w(s)^2 = (s + 1)(s^2 + 1) / s^3``, the exact
  reweighting of the energy, 2 at isometry); the proxy ``sum A_t |W (J - R)|^2`` plus a
  small proximal term is solved by LDL^T (same sparsity every iteration: one symbolic
  analysis); the step is capped at 90% of the largest flip-free step (Smith & Schaefer
  2015) and halved until the energy decreases. Maps stay free of inverted triangles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import TexturingError
from . import sparse
from .topology import union_labels

AREA_EPSILON = 1e-300
PROXIMAL = 1e-4
"""Weight of the proximal term ``sum_i a_i |u_i - u_i^prev|^2`` (``a_i``: vertex area)."""
REGULARIZATION = 1e-10
"""Diagonal added to LSCM and Tutte systems, relative to the mean triangle area."""


@dataclass
class ChartSet:
    """Triangles of several charts with chart-local vertices (ids unique across charts)."""

    tri: np.ndarray  # (F, 3) vertex ids
    points: np.ndarray  # (N, 3) float64 3D positions of the vertices
    face_chart: np.ndarray  # (F,) chart index 0..C-1
    vertex_chart: np.ndarray  # (N,)
    charts: int

    def __post_init__(self) -> None:
        p = self.points[self.tri]
        self.q, self.area, self.grad = local_frames(p)
        self.vertex_area = np.bincount(
            self.tri.reshape(-1), np.repeat(self.area / 3.0, 3), self.points.shape[0]
        )
        self.chart_area = np.bincount(self.face_chart, self.area, self.charts)


def local_frames(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isometric 2D coordinates ``(F, 3, 2)``, areas ``(F,)`` and barycentric gradients
    ``(F, 3, 2)`` of triangles ``p`` (``(F, 3, 3)``); degenerate triangles get zero area and
    zero gradients."""
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    l1 = np.linalg.norm(e1, axis=1)
    x = np.divide(e1, l1[:, None], out=np.zeros_like(e1), where=l1[:, None] > 0)
    n = np.cross(e1, e2)
    nl = np.linalg.norm(n, axis=1)
    y = np.cross(np.divide(n, nl[:, None], out=np.zeros_like(n), where=nl[:, None] > 0), x)
    q = np.zeros((p.shape[0], 3, 2))
    q[:, 1, 0] = l1
    q[:, 2, 0] = (e2 * x).sum(axis=1)
    q[:, 2, 1] = (e2 * y).sum(axis=1)
    twice = q[:, 1, 0] * q[:, 2, 1] - q[:, 2, 0] * q[:, 1, 1]
    good = twice > AREA_EPSILON
    inv = np.where(good, 1.0 / np.where(good, twice, 1.0), 0.0)
    grad = np.zeros_like(q)
    for i in range(3):
        a, b = q[:, (i + 1) % 3], q[:, (i + 2) % 3]
        edge = b - a
        grad[:, i, 0] = -edge[:, 1] * inv
        grad[:, i, 1] = edge[:, 0] * inv
    return q, np.where(good, 0.5 * twice, 0.0), grad


def jacobians(cs: ChartSet, uv: np.ndarray) -> np.ndarray:
    """``(F, 2, 2)``: rows are the gradients of u and v in each triangle's local frame."""
    t = uv[cs.tri]  # (F, 3, 2)
    return np.einsum("fic,fik->fck", t, cs.grad)


def uv_det(uv: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Twice the signed UV area of each triangle (positive: counter-clockwise)."""
    t = uv[tri]
    e1 = t[:, 1] - t[:, 0]
    e2 = t[:, 2] - t[:, 0]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #


@dataclass
class Boundary:
    """Boundary half-edges of the chart set and their loops."""

    start: np.ndarray  # (B,) vertex
    end: np.ndarray  # (B,)
    loop: np.ndarray  # (B,) loop label (smallest start vertex of the loop)
    loops_per_chart: np.ndarray  # (C,)
    is_boundary: np.ndarray  # (N,) bool
    vertex_next: np.ndarray  # (N,) next boundary vertex, -1 inside (or ambiguous)
    regular: np.ndarray  # (C,) every boundary vertex has one outgoing boundary edge


def boundary(cs: ChartSet) -> Boundary:
    n = cs.points.shape[0]
    a = cs.tri.reshape(-1)
    b = cs.tri[:, [1, 2, 0]].reshape(-1)
    key = a * n + b
    rev = b * n + a
    present = np.isin(rev, key)
    bstart, bend = a[~present], b[~present]
    # Duplicated directed edges (two faces with the same orientation) count as boundary.
    out_count = np.bincount(bstart, minlength=n)
    nxt = np.full(n, -1, dtype=np.int64)
    single = out_count[bstart] == 1
    nxt[bstart[single]] = bend[single]
    is_b = out_count > 0
    loop = union_labels(n, bstart, bend)[bstart] if bstart.size else bstart
    charts = cs.charts
    loops = np.zeros(charts, dtype=np.int64)
    if bstart.size:
        uniq = np.unique(loop)
        np.add.at(loops, cs.vertex_chart[uniq], 1)
    irregular = np.zeros(charts, dtype=bool)
    bad = np.flatnonzero(out_count > 1)
    irregular[cs.vertex_chart[bad]] = True
    return Boundary(bstart, bend, loop, loops, is_b, nxt, ~irregular)


def euler_disk(cs: ChartSet, bd: Boundary) -> np.ndarray:
    """Per chart: a topological disk (``V - E + F = 1``, one regular boundary loop)."""
    n = cs.points.shape[0]
    a = cs.tri.reshape(-1)
    b = cs.tri[:, [1, 2, 0]].reshape(-1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ekey = np.unique(lo * n + hi)
    e_chart = cs.vertex_chart[ekey // n]
    E = np.bincount(e_chart, minlength=cs.charts)
    V = np.bincount(cs.vertex_chart, minlength=cs.charts)
    F = np.bincount(cs.face_chart, minlength=cs.charts)
    return (V - E + F == 1) & (bd.loops_per_chart == 1) & bd.regular


def _loop_order(bd: Boundary, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Boundary vertices in loop order: ``(vertices, rank)``, rank 0 at each loop's smallest
    vertex (list ranking by pointer doubling over the successor of every vertex)."""
    verts = bd.start
    if verts.size == 0:
        return verts, verts
    root = np.zeros(n, dtype=bool)
    root[bd.loop] = True
    jump = np.arange(n, dtype=np.int64)
    jump[verts] = np.where(root[verts], verts, bd.vertex_next[verts])
    dist = np.zeros(n, dtype=np.int64)
    dist[verts] = np.where(root[verts], 0, 1)
    while True:
        far = jump[jump]
        if np.array_equal(far, jump):
            break
        dist = dist + dist[jump]
        jump = far
    length = np.bincount(bd.loop, minlength=n)
    steps = dist[verts]
    return verts, np.where(steps == 0, 0, length[bd.loop] - steps)


# --------------------------------------------------------------------------- #
# Systems
# --------------------------------------------------------------------------- #


def _vertex_graph(tri: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.r_[tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 1], tri[:, 2], tri[:, 0]]
    b = np.r_[tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 2]]
    key = np.unique(a * n + b)
    src, dst = key // n, key % n
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    return np.cumsum(indptr), dst


class Solver:
    """A factorization cache for the 2N x 2N systems of one chart set (u, v interleaved)."""

    def __init__(self, cs: ChartSet, free: np.ndarray | None = None) -> None:
        self.cs = cs
        n = cs.points.shape[0]
        self.free = np.ones(2 * n, dtype=bool) if free is None else free
        self.index = np.full(2 * n, -1, dtype=np.int64)
        self.index[self.free] = np.arange(int(self.free.sum()), dtype=np.int64)
        indptr, nbr = _vertex_graph(cs.tri, n)
        vperm = sparse.nested_dissection(cs.points, indptr, nbr, cs.vertex_chart)
        unknowns = np.stack([2 * vperm, 2 * vperm + 1], axis=1).reshape(-1)
        self.perm = self.index[unknowns[self.free[unknowns]]]
        self.symbolic: sparse.Symbolic | None = None

    def solve_reduced(self, matrix: sparse.CSC, rhs: np.ndarray) -> np.ndarray:
        if self.symbolic is None:
            self.symbolic = sparse.analyze(matrix, self.perm)
        return sparse.factor(self.symbolic, matrix).solve(rhs)


def _quadratic_blocks(cs: ChartSet, coeff: np.ndarray, target: np.ndarray, weight: np.ndarray):
    """COO entries of ``sum_f weight_f sum_r (coeff_fr . x_f - target_fr)^2``'s normal
    equations over the 6 unknowns of each face (u0 u1 u2 v0 v1 v2).

    ``coeff`` ``(F, R, 6)``, ``target`` ``(F, R)``. Returns rows, cols, values of the
    matrix and the per-face right-hand sides ``(F, 6)``.
    """
    local = np.einsum("f,fri,frj->fij", weight, coeff, coeff)
    rhs = np.einsum("f,fri,fr->fi", weight, coeff, target)
    unknown = np.concatenate([2 * cs.tri, 2 * cs.tri + 1], axis=1)  # (F, 6)
    rows = np.repeat(unknown, 6, axis=1).reshape(-1)
    cols = np.tile(unknown, (1, 6)).reshape(-1)
    return rows, cols, local.reshape(-1), unknown, rhs


def _reduce(
    n2: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    unknown: np.ndarray,
    face_rhs: np.ndarray,
    free: np.ndarray,
    index: np.ndarray,
    fixed: np.ndarray,
    diagonal: np.ndarray,
) -> tuple[sparse.CSC, np.ndarray]:
    """The system over free unknowns: pinned columns move to the right-hand side."""
    rhs_full = np.bincount(unknown.reshape(-1), face_rhs.reshape(-1), n2)
    keep = free[rows] & free[cols]
    pinned_col = free[rows] & ~free[cols]
    if pinned_col.any():
        rhs_full -= np.bincount(rows[pinned_col], vals[pinned_col] * fixed[cols[pinned_col]], n2)
    m = int(free.sum())
    diag_idx = np.flatnonzero(free)
    r = np.r_[index[rows[keep]], index[diag_idx]]
    c = np.r_[index[cols[keep]], index[diag_idx]]
    v = np.r_[vals[keep], diagonal[diag_idx]]
    return sparse.assemble(m, r, c, v), rhs_full[free]


# --------------------------------------------------------------------------- #
# LSCM
# --------------------------------------------------------------------------- #


def pins(cs: ChartSet, bd: Boundary) -> np.ndarray:
    """Two pinned vertices per chart ``(C, 2)``: the boundary vertex farthest from the
    boundary's centroid, then the boundary vertex farthest from it (ties: lowest id)."""
    C = cs.charts
    bv = np.unique(bd.start)
    out = np.full((C, 2), -1, dtype=np.int64)
    if bv.size == 0:
        return out
    ch = cs.vertex_chart[bv]
    p = cs.points[bv]
    count = np.bincount(ch, minlength=C)
    centroid = np.stack([np.bincount(ch, p[:, k], C) for k in range(3)], axis=1)
    centroid /= np.maximum(count, 1)[:, None]
    d0 = ((p - centroid[ch]) ** 2).sum(axis=1)
    first = _argmax_per(ch, d0, bv, C)
    out[:, 0] = first
    has = first >= 0
    d1 = ((p - cs.points[np.where(has, first, 0)][ch]) ** 2).sum(axis=1)
    out[:, 1] = _argmax_per(ch, d1, bv, C)
    return out


def _argmax_per(group: np.ndarray, value: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((ids, -value, group))
    g = group[order]
    firsts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    out = np.full(count, -1, dtype=np.int64)
    out[g[firsts]] = ids[order][firsts]
    return out


def lscm(cs: ChartSet, bd: Boundary, charts: np.ndarray) -> np.ndarray:
    """Conformal maps of the charts flagged in ``charts`` (``(C,)`` bool); other vertices
    get NaN. Every flagged chart must be a disk with at least two boundary vertices."""
    n = cs.points.shape[0]
    uv = np.full((n, 2), np.nan)
    faces = charts[cs.face_chart]
    if not faces.any():
        return uv
    sub, vmap = _subset(cs, faces)
    sbd = boundary(sub)
    pin = pins(sub, sbd)
    fixed = np.zeros(2 * sub.points.shape[0])
    free = np.ones(2 * sub.points.shape[0], dtype=bool)
    p0, p1 = pin[:, 0], pin[:, 1]
    ok = (p0 >= 0) & (p1 >= 0) & (p0 != p1)
    if not ok.all():
        raise TexturingError("CHART_NOT_A_DISK", "a chart has fewer than two boundary vertices")
    dist = np.linalg.norm(sub.points[p1] - sub.points[p0], axis=1)
    for v, u_ in ((p0, np.zeros_like(dist)), (p1, dist)):
        free[2 * v] = False
        free[2 * v + 1] = False
        fixed[2 * v] = u_
        fixed[2 * v + 1] = 0.0
    g = sub.grad
    F = sub.tri.shape[0]
    coeff = np.zeros((F, 2, 6))
    # Row 1: du/dx - dv/dy = 0; row 2: du/dy + dv/dx = 0.
    coeff[:, 0, :3] = g[:, :, 0]
    coeff[:, 0, 3:] = -g[:, :, 1]
    coeff[:, 1, :3] = g[:, :, 1]
    coeff[:, 1, 3:] = g[:, :, 0]
    rows, cols, vals, unknown, frhs = _quadratic_blocks(sub, coeff, np.zeros((F, 2)), sub.area)
    solver = Solver(sub, free)
    eps = REGULARIZATION * max(float(sub.area.mean()) if F else 1.0, AREA_EPSILON)
    matrix, rhs = _reduce(
        2 * sub.points.shape[0],
        rows,
        cols,
        vals,
        unknown,
        frhs,
        free,
        solver.index,
        fixed,
        np.full(2 * sub.points.shape[0], eps),
    )
    x = fixed.copy()
    x[free] = solver.solve_reduced(matrix, rhs)
    uv[vmap] = x.reshape(-1, 2)
    return uv


def _subset(cs: ChartSet, faces: np.ndarray) -> tuple[ChartSet, np.ndarray]:
    """The chart set restricted to ``faces``; returns it and the vertex map (new -> old)."""
    tri = cs.tri[faces]
    used = np.unique(tri)
    remap = np.full(cs.points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    chart_ids, chart_map = np.unique(cs.face_chart[faces], return_inverse=True)
    sub = ChartSet(
        remap[tri],
        cs.points[used],
        chart_map.reshape(-1).astype(np.int64),
        np.searchsorted(chart_ids, cs.vertex_chart[used]),
        int(chart_ids.shape[0]),
    )
    return sub, used


# --------------------------------------------------------------------------- #
# Tutte
# --------------------------------------------------------------------------- #


def tutte(cs: ChartSet, charts: np.ndarray) -> np.ndarray:
    """Tutte embeddings (uniform weights, boundary on a circle of the boundary's length)
    of the flagged disk charts; other vertices get NaN."""
    n = cs.points.shape[0]
    uv = np.full((n, 2), np.nan)
    faces = charts[cs.face_chart]
    if not faces.any():
        return uv
    sub, vmap = _subset(cs, faces)
    m = sub.points.shape[0]
    sbd = boundary(sub)
    verts, rank = _loop_order(sbd, m)
    seg = np.linalg.norm(sub.points[sbd.end] - sub.points[sbd.start], axis=1)
    order = np.lexsort((rank, sbd.loop))
    v_sorted = verts[order]
    loops = sbd.loop[order]
    seg_sorted = seg[order]
    starts = np.flatnonzero(np.r_[True, loops[1:] != loops[:-1]])
    csum = np.cumsum(seg_sorted)
    base = np.repeat(csum[starts] - seg_sorted[starts], np.diff(np.r_[starts, len(loops)]))
    arc = csum - seg_sorted - base  # arc length from the loop's root to each vertex
    total = np.repeat(
        np.add.reduceat(seg_sorted, starts) if len(starts) else seg_sorted,
        np.diff(np.r_[starts, len(loops)]),
    )
    angle = 2.0 * np.pi * arc / np.maximum(total, AREA_EPSILON)
    radius = total / (2.0 * np.pi)
    fixed = np.zeros((m, 2))
    fixed[v_sorted, 0] = radius * np.cos(angle)
    fixed[v_sorted, 1] = radius * np.sin(angle)
    is_b = np.zeros(m, dtype=bool)
    is_b[v_sorted] = True
    # Uniform Laplacian on interior vertices, the same for u and v.
    indptr, nbr = _vertex_graph(sub.tri, m)
    src = np.repeat(np.arange(m), np.diff(indptr))
    interior = ~is_b
    idx = np.full(m, -1, dtype=np.int64)
    idx[interior] = np.arange(int(interior.sum()))
    deg = np.diff(indptr).astype(np.float64)
    e_int = interior[src] & interior[nbr]
    rows = np.r_[idx[src[e_int]], idx[np.flatnonzero(interior)]]
    cols = np.r_[idx[nbr[e_int]], idx[np.flatnonzero(interior)]]
    vals = np.r_[-np.ones(int(e_int.sum())), deg[interior]]
    k = int(interior.sum())
    out = fixed.copy()
    if k:
        matrix = sparse.assemble(k, rows, cols, vals)
        e_b = interior[src] & is_b[nbr]
        rhs = np.zeros((k, 2))
        np.add.at(rhs, idx[src[e_b]], fixed[nbr[e_b]])
        pts = sub.points[interior]
        sub_ip = np.zeros(k + 1, dtype=np.int64)
        np.add.at(sub_ip, idx[src[e_int]] + 1, 1)
        perm = sparse.nested_dissection(
            pts,
            np.cumsum(sub_ip),
            idx[nbr[e_int]][np.argsort(idx[src[e_int]], kind="stable")],
            sub.vertex_chart[interior],
        )
        out[interior] = sparse.factor(sparse.analyze(matrix, perm), matrix).solve(rhs)
    uv[vmap] = out
    return uv


# --------------------------------------------------------------------------- #
# SLIM
# --------------------------------------------------------------------------- #


def svd2(j: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form SVD of 2x2 matrices: ``U, s1, s2, V`` with ``J = U diag(s1, s2) V^T``,
    ``s1 >= |s2|``, ``U`` and ``V`` rotations (``s2 < 0`` for inverted maps)."""
    a, b, c, d = j[:, 0, 0], j[:, 0, 1], j[:, 1, 0], j[:, 1, 1]
    e = 0.5 * (a + d)
    f = 0.5 * (a - d)
    g = 0.5 * (c + b)
    h = 0.5 * (c - b)
    q = np.hypot(e, h)
    r = np.hypot(f, g)
    s1 = q + r
    s2 = q - r
    a1 = np.arctan2(g, f)
    a2 = np.arctan2(h, e)
    theta = 0.5 * (a2 - a1)
    phi = 0.5 * (a2 + a1)
    u = _rot(phi)
    v = _rot(-theta)  # V^T = rot(theta)
    return u, s1, s2, v


def _rot(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty(angle.shape + (2, 2))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def dirichlet(cs: ChartSet, uv: np.ndarray) -> np.ndarray:
    """Per-chart symmetric Dirichlet energy, normalized by area (4 for an isometry;
    infinite if a non-degenerate triangle is inverted or flat)."""
    j = jacobians(cs, uv)
    det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
    fro = (j * j).sum(axis=(1, 2))
    live = cs.area > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(live, fro * (1.0 + 1.0 / (det * det)), 0.0)
    e = np.where(live & (det <= 0), np.inf, e)
    total = np.bincount(cs.face_chart, cs.area * np.where(live, e, 0.0), cs.charts)
    bad = np.bincount(cs.face_chart, (live & ~np.isfinite(e)).astype(np.float64), cs.charts)
    total = np.where(bad > 0, np.inf, total)
    return total / np.maximum(cs.chart_area, AREA_EPSILON)


def max_steps(cs: ChartSet, uv: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per chart, the largest ``t`` keeping every triangle of ``uv + t d`` non-inverted."""
    t = uv[cs.tri]
    s = d[cs.tri]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    d1, d2 = s[:, 1] - s[:, 0], s[:, 2] - s[:, 0]
    c = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    b = e1[:, 0] * d2[:, 1] - e1[:, 1] * d2[:, 0] + d1[:, 0] * e2[:, 1] - d1[:, 1] * e2[:, 0]
    a = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b * b - 4.0 * a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        r1 = (-b - sq) / (2.0 * a)
        r2 = (-b + sq) / (2.0 * a)
        lin = -c / b
    tmax = np.full(c.shape, np.inf)
    quad = np.abs(a) > 1e-30 * np.maximum(np.abs(b) + np.abs(c), 1e-300)
    for r in (r1, r2):
        ok = quad & (disc >= 0) & (r > 0)
        tmax = np.where(ok, np.minimum(tmax, r), tmax)
    ok = ~quad & (b < 0)
    tmax = np.where(ok, np.minimum(tmax, lin), tmax)
    tmax = np.where(cs.area > 0, tmax, np.inf)
    out = np.full(cs.charts, np.inf)
    np.minimum.at(out, cs.face_chart, tmax)
    return out


def slim(
    cs: ChartSet,
    uv: np.ndarray,
    charts: np.ndarray,
    *,
    iterations: int = 12,
    tolerance: float = 1e-4,
    should_stop=None,
) -> tuple[np.ndarray, np.ndarray]:
    """SLIM on the flagged charts, starting from flip-free ``uv``. Returns the new UVs and
    the final per-chart energies."""
    faces = charts[cs.face_chart]
    if not faces.any():
        return uv, dirichlet(cs, uv)
    sub, vmap = _subset(cs, faces)
    x = uv[vmap].copy()
    m = sub.points.shape[0]
    F = sub.tri.shape[0]
    energy = dirichlet(sub, x)
    active = np.isfinite(energy)
    solver = Solver(sub)
    prox = PROXIMAL * np.repeat(np.maximum(sub.vertex_area, AREA_EPSILON), 2)
    g = sub.grad
    for _ in range(iterations):
        if should_stop is not None and should_stop():
            break
        if not active.any():
            break
        j = jacobians(sub, x)
        u, s1, s2, _v = svd2(j)
        r = np.einsum("fij,fkj->fik", u, _v)  # U V^T
        w1 = np.sqrt((s1 + 1.0) * (s1 * s1 + 1.0) / np.maximum(s1, 1e-12) ** 3)
        w2 = np.sqrt((s2 + 1.0) * (s2 * s2 + 1.0) / np.maximum(s2, 1e-12) ** 3)
        wm = np.einsum("fik,fk,fjk->fij", u, np.stack([w1, w2], axis=1), u)
        coeff = np.zeros((F, 4, 6))
        target = np.zeros((F, 4))
        for row in range(2):
            for k in range(2):
                rr = 2 * row + k
                coeff[:, rr, :3] = wm[:, row, 0, None] * g[:, :, k]
                coeff[:, rr, 3:] = wm[:, row, 1, None] * g[:, :, k]
                target[:, rr] = wm[:, row, 0] * r[:, 0, k] + wm[:, row, 1] * r[:, 1, k]
        rows, cols, vals, unknown, frhs = _quadratic_blocks(sub, coeff, target, sub.area)
        free = np.ones(2 * m, dtype=bool)
        matrix, rhs = _reduce(
            2 * m, rows, cols, vals, unknown, frhs, free, solver.index, np.zeros(2 * m), prox
        )
        rhs = rhs + prox * x.reshape(-1)
        target_x = solver.solve_reduced(matrix, rhs).reshape(-1, 2)
        d = target_x - x
        d[~active[sub.vertex_chart]] = 0.0
        tmax = max_steps(sub, x, d)
        step = np.where(active, np.minimum(1.0, 0.9 * tmax), 0.0)
        accepted = ~active
        new_energy = energy.copy()
        for _halving in range(30):
            trial = x + step[sub.vertex_chart][:, None] * d
            e_trial = dirichlet(sub, trial)
            better = ~accepted & (e_trial < energy)
            new_energy = np.where(better, e_trial, new_energy)
            accepted |= better
            if accepted.all():
                break
            step = np.where(accepted, step, 0.5 * step)
        step = np.where(accepted & active, step, 0.0)
        x = x + step[sub.vertex_chart][:, None] * d
        improvement = (energy - new_energy) / np.maximum(energy, AREA_EPSILON)
        active &= (step > 0) & (improvement > tolerance)
        energy = np.where(step > 0, new_energy, energy)
    out = uv.copy()
    out[vmap] = x
    full = np.full(cs.charts, np.nan)
    chart_ids = np.unique(cs.face_chart[faces])
    full[chart_ids] = energy
    return out, full
