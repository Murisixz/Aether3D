"""Automatic charts: segmentation, parameterization and recursive splitting.

1. **Cuts.** Boundaries, non-manifold edges and inconsistently oriented edges always cut.
   So do the requested seams, the mesh's existing UV seams with ``keep_seams``, hard edges
   and face-group borders with ``split_by``, and every edge whose dihedral angle exceeds
   ``angle_limit`` (planar regions of hard-surface models become one chart each).
2. **Charts** are the face components across uncut edges.
3. **Every chart is parameterized** (all together, :mod:`aether3d.uv.param`): a chart that
   is not a topological disk is split; a disk gets LSCM, or a Tutte embedding if LSCM
   inverts a triangle; ``method="isometric"`` (default) refines it with SLIM.
4. **A chart is accepted** when its map has no inverted triangle, no boundary
   self-overlap and an L2 stretch (Sander et al. 2001, scale-normalized) of at most
   ``max_stretch``. Otherwise it is **split in two**: faces go to the nearer (in dual-graph
   hops; ties to the first) of two seeds, the face farthest from the chart's first face
   and the face farthest from that one (Iso-charts style). Split charts start again at 3.

Everything is deterministic: ties break by the lowest face or vertex id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import TexturingError
from . import param
from .metrics import boundary_overlaps, stretch
from .topology import Topology, dense_labels

SPLIT_BY = ("hard", "groups", "seams")
METHODS = ("isometric", "conformal")


@dataclass
class ChartOptions:
    """Options of :func:`compute_charts` (validated by :func:`check_options`)."""

    angle_limit: float = 66.0
    split_by: tuple[str, ...] = ("hard",)
    method: str = "isometric"
    max_stretch: float = 1.15
    iterations: int = 12
    max_charts: int = 100_000


@dataclass
class Charts:
    """The result of :func:`compute_charts`."""

    topology: Topology
    face_chart: np.ndarray  # (T,) chart of every triangle, 0..C-1
    corner_vertex: np.ndarray  # (T, 3) chart vertex of every corner
    uv: np.ndarray  # (N, 2) chart-vertex UVs (world units: isometric maps keep lengths)
    count: int
    stretch_l2: np.ndarray  # (C,)
    stretch_max: np.ndarray  # (C,)
    energy: np.ndarray  # (C,) normalized symmetric Dirichlet energy (4: isometric)
    splits: int = 0
    diagnostics: list = field(default_factory=list)


def compute_charts(
    topo: Topology,
    cut: np.ndarray,
    options: ChartOptions,
    *,
    faces: np.ndarray | None = None,
    atoms: np.ndarray | None = None,
    should_stop=None,
) -> Charts:
    """Charts of the faces flagged in ``faces`` (all by default) with the cut half-edges
    ``cut`` (see the module docstring). Triangles with the same ``atoms`` id (the triangles
    of one polygon) are never split apart."""
    T = topo.triangle_count
    active = np.ones(T, dtype=bool) if faces is None else faces.astype(bool)
    labels = topo.face_components(cut, active)
    labels = np.where(active, labels, -1)
    final_chart = np.full(T, -1, dtype=np.int64)
    final_uv: dict[int, np.ndarray] = {}
    pending = labels.copy()
    splits = 0
    accepted_stats: list[tuple[np.ndarray, float, float, float]] = []
    corner_vertex_final = np.full((T, 3), -1, dtype=np.int64)
    uv_parts: list[np.ndarray] = []
    vertex_base = 0
    chart_base = 0
    rounds = 0
    while (pending >= 0).any():
        rounds += 1
        if rounds > 64:
            raise TexturingError("UNWRAP_FAILED", "charts did not converge after 64 rounds")
        if should_stop is not None and should_stop():
            raise TexturingError("CANCELLED", "the unwrap was cancelled")
        live = pending >= 0
        cid, count = dense_labels(pending[live])
        chart_of = np.full(T, -1, dtype=np.int64)
        chart_of[live] = cid
        if count > options.max_charts:
            raise TexturingError(
                "TOO_MANY_CHARTS",
                f"the unwrap needs more than {options.max_charts} charts",
                hint="raise angle_limit or max_stretch, or remove seams",
            )
        cv, _ = topo.chart_vertices(np.where(live, chart_of, -1 - np.arange(T)), cut)
        faces_live = np.flatnonzero(live)
        tri_cv = cv[faces_live]
        used, inv = np.unique(tri_cv, return_inverse=True)
        tri = inv.reshape(-1, 3)
        points = topo.wpos[_corner_weld(topo, faces_live, tri_cv, used)]
        vchart = np.zeros(used.shape[0], dtype=np.int64)
        vchart[tri.reshape(-1)] = np.repeat(cid, 3)
        cs = param.ChartSet(tri, points, cid, vchart, count)
        bd = param.boundary(cs)
        disk = param.euler_disk(cs, bd)
        nfaces = np.bincount(cid, minlength=count)
        uv = np.full((used.shape[0], 2), np.nan)
        ok = disk.copy()
        if ok.any():
            uv = param.lscm(cs, bd, ok)
            inverted = np.bincount(
                cid, ((param.uv_det(uv, tri) <= 0) & (cs.area > 0)).astype(np.float64), count
            )
            nonfinite = np.bincount(
                vchart, (~np.isfinite(uv).all(axis=1)).astype(np.float64), count
            )
            bad_init = ok & ((inverted > 0) | (nonfinite > 0))
            if bad_init.any():
                tut = param.tutte(cs, bad_init)
                sel = bad_init[vchart]
                uv[sel] = tut[sel]
            if options.method == "isometric":
                uv, energy = param.slim(
                    cs, uv, ok, iterations=options.iterations, should_stop=should_stop
                )
            else:
                energy = param.dirichlet(cs, uv)
        else:
            energy = np.full(count, np.nan)
        l2, linf = stretch(cs, uv)
        inverted = np.bincount(
            cid, ((param.uv_det(uv, tri) <= 0) & (cs.area > 0)).astype(np.float64), count
        )
        overlaps = boundary_overlaps(cs, bd, uv, ok)
        good = ok & (inverted == 0) & (overlaps == 0) & np.isfinite(l2)
        good &= (l2 <= options.max_stretch) | (nfaces <= 1)
        # A chart that cannot be split further is accepted as it is (reported).
        stuck = ~good & (nfaces <= 1)
        accept = good | stuck
        # Accepted charts: record their vertices and UVs.
        acc_ids = np.flatnonzero(accept)
        if acc_ids.size:
            vsel = accept[vchart]
            new_index = np.full(used.shape[0], -1, dtype=np.int64)
            new_index[vsel] = vertex_base + np.arange(int(vsel.sum()))
            fsel = accept[cid]
            corner_vertex_final[faces_live[fsel]] = new_index[tri[fsel]]
            chart_new = np.full(count, -1, dtype=np.int64)
            chart_new[acc_ids] = chart_base + np.arange(acc_ids.size)
            final_chart[faces_live[fsel]] = chart_new[cid[fsel]]
            part = uv[vsel]
            part = np.where(np.isfinite(part), part, 0.0)
            uv_parts.append(part)
            vertex_base += int(vsel.sum())
            chart_base += acc_ids.size
            for c in acc_ids.tolist():
                accepted_stats.append((c, float(l2[c]), float(linf[c]), float(energy[c])))
        # Rejected charts: split.
        rej = ~accept
        pending = np.full(T, -1, dtype=np.int64)
        if rej.any():
            splits += int(rej.sum())
            fsel = rej[cid]
            sub_faces = faces_live[fsel]
            halves = _split(topo, cut, sub_faces, cid[fsel])
            if atoms is not None:
                # Triangles of one polygon stay together: the part of its first triangle.
                at = atoms[sub_faces]
                order = np.lexsort((sub_faces, at))
                first = np.r_[True, at[order][1:] != at[order][:-1]]
                lead = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
                halves = halves.copy()
                halves[order] = halves[order][lead]
            pending[sub_faces] = halves
    count = chart_base
    stats = np.array([s[1:] for s in accepted_stats], dtype=np.float64).reshape(-1, 3)
    uv_all = np.concatenate(uv_parts) if uv_parts else np.zeros((0, 2))
    del final_uv
    return Charts(
        topo,
        final_chart,
        corner_vertex_final,
        uv_all,
        count,
        stats[:, 0],
        stats[:, 1],
        stats[:, 2],
        splits,
    )


def _corner_weld(topo: Topology, faces: np.ndarray, tri_cv: np.ndarray, used: np.ndarray):
    """The welded vertex of each chart vertex in ``used``."""
    w = topo.corners[faces].reshape(-1)
    out = np.empty(used.shape[0], dtype=np.int64)
    pos = np.searchsorted(used, tri_cv.reshape(-1))
    out[pos] = w
    return out


def _split(topo: Topology, cut: np.ndarray, faces: np.ndarray, chart: np.ndarray) -> np.ndarray:
    """New labels splitting each chart in two (see the module docstring): labels are face
    ids (unique per new part)."""
    T = topo.triangle_count
    in_set = np.zeros(T, dtype=bool)
    in_set[faces] = True
    chart_of = np.full(T, -1, dtype=np.int64)
    chart_of[faces] = chart
    h = np.flatnonzero((topo.mate >= 0) & ~cut)
    m = topo.mate[h]
    keep = in_set[h // 3] & in_set[m // 3] & (chart_of[h // 3] == chart_of[m // 3])
    a, b = h[keep] // 3, m[keep] // 3
    # Dual adjacency in CSR (both directions are present: h and its mate).
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    indptr = np.zeros(T + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    indptr = np.cumsum(indptr)
    # First face of each chart, its farthest face, and that face's farthest face.
    first = _first_per(chart, faces)
    d0 = _bfs(T, indptr, b, first)
    seed_a = _farthest_per(chart, faces, d0)
    da = _bfs(T, indptr, b, seed_a)
    seed_b = _farthest_per(chart, faces, da)
    db = _bfs(T, indptr, b, seed_b)
    to_b = db[faces] < da[faces]
    unreachable = (da[faces] < 0) & (db[faces] < 0)
    label = np.where(to_b, seed_b[_chart_index(chart)], seed_a[_chart_index(chart)])
    # Faces not reached from either seed (disconnected pieces) become their own parts.
    label = np.where(unreachable, faces, label)
    # A chart whose seeds coincide (one face) cannot split; give it a unique label anyway.
    return label


def _chart_index(chart: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(chart, return_inverse=True)
    return inv.reshape(-1)


def _first_per(chart: np.ndarray, faces: np.ndarray) -> np.ndarray:
    order = np.lexsort((faces, chart))
    c = chart[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    return faces[order][starts]


def _farthest_per(chart: np.ndarray, faces: np.ndarray, dist: np.ndarray) -> np.ndarray:
    d = dist[faces]
    order = np.lexsort((faces, -d, chart))
    c = chart[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    return faces[order][starts]


def _bfs(n: int, indptr: np.ndarray, nbr: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Hop distances from ``sources`` (``-1`` unreached), frontier by frontier."""
    dist = np.full(n, -1, dtype=np.int64)
    frontier = np.unique(sources)
    dist[frontier] = 0
    level = 0
    while frontier.size:
        level += 1
        lo, hi = indptr[frontier], indptr[frontier + 1]
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            break
        pos = np.repeat(lo - np.cumsum(cnt) + cnt, cnt) + np.arange(total)
        cand = np.unique(nbr[pos])
        cand = cand[dist[cand] < 0]
        dist[cand] = level
        frontier = cand
    return dist


def check_options(
    angle_limit: float, split_by, method: str, max_stretch: float, iterations: int
) -> ChartOptions:
    """Validated :class:`ChartOptions` (``TexturingError("INVALID_ARGUMENT")`` otherwise)."""

    def bad(name: str, why: str) -> TexturingError:
        return TexturingError("INVALID_ARGUMENT", f"{name} {why}", name)

    if isinstance(angle_limit, bool) or not isinstance(angle_limit, (int, float)):
        raise bad("angle_limit", "must be a number of degrees")
    if not (math.isfinite(angle_limit) and 0.0 < angle_limit <= 180.0):
        raise bad("angle_limit", "must be in (0, 180] degrees")
    if isinstance(split_by, str):
        split_by = (split_by,)
    try:
        split = tuple(split_by)
    except TypeError:
        raise bad("split_by", "must be a list of 'hard', 'groups', 'seams'") from None
    for item in split:
        if item not in SPLIT_BY:
            raise bad("split_by", f"has unknown entry {item!r} (use 'hard', 'groups', 'seams')")
    if method not in METHODS:
        raise bad("method", f"must be one of {', '.join(METHODS)}")
    if isinstance(max_stretch, bool) or not isinstance(max_stretch, (int, float)):
        raise bad("max_stretch", "must be a number")
    if not (math.isfinite(max_stretch) and max_stretch >= 1.0):
        raise bad("max_stretch", "must be at least 1 (1 = no stretch)")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 0 <= iterations <= 200
    ):
        raise bad("iterations", "must be an integer in [0, 200]")
    return ChartOptions(
        float(angle_limit), tuple(dict.fromkeys(split)), method, float(max_stretch), iterations
    )
