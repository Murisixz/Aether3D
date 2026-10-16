"""Validity of triangle meshes: topology, degeneracy and (optionally) self-intersections.

:func:`check_mesh` reports, with stable codes and sample ids (docs/MODELING.md,
"Validity"):

=========================  ========  =====================================================
code                       severity  meaning
=========================  ========  =====================================================
NON_FINITE_POSITIONS       error     a vertex position is not finite
DEGENERATE_TRIANGLES       warning   a triangle repeats a vertex or has (near) zero area
                                     (``area <= 1e-12 * size^2``)
NON_MANIFOLD_EDGES         warning   an edge has more than two triangles
NON_MANIFOLD_VERTICES      warning   the triangles around a vertex form several fans
INCONSISTENT_WINDING       warning   two triangles run a shared edge the same way
BOUNDARY_EDGES             info      edges with one triangle (an open surface)
SELF_INTERSECTIONS         error     two triangles that share no vertex intersect, or two
                                     that share one vertex meet elsewhere, or two that
                                     share an edge fold onto each other
                                     (only with ``self_intersections=True``)
=========================  ========  =====================================================

The summary gives counts, ``closed``, ``edge_manifold``, ``vertex_manifold``,
``consistently_oriented``, the Euler characteristic, the signed volume of closed meshes,
the edge-length range and the smallest triangle angle (quality).

Self-intersections: candidate pairs come from a BVH over the triangles' boxes
(:func:`~aether3d.spatial.geometry.bvh_box_pairs`). Pairs without a shared vertex are
decided by :func:`~aether3d.spatial.geometry.triangles_intersect` (touching counts);
pairs sharing one vertex by exact segment-triangle tests of the edges away from it, and
pairs sharing an edge by exact coplanarity and side tests
(:mod:`aether3d.spatial.predicates`), so the coplanar neighbours of flat regions are
never mistaken for intersections. Without the option the report lists self-intersections under
``not_evaluated`` and never claims their absence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..diagnostics import ERROR, INFO, MAX_SAMPLES, WARNING, Diagnostic, worst_severity

PAIR_BATCH = 1 << 20
"""Candidate triangle pairs tested at once (bounds temporary memory)."""


@dataclass
class MeshReport:
    """What :func:`check_mesh` found."""

    summary: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    not_evaluated: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return worst_severity(self.diagnostics)

    @property
    def ok(self) -> bool:
        return self.status != "error"

    def codes(self) -> set[str]:
        return {d.code for d in self.diagnostics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": dict(self.summary),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "not_evaluated": list(self.not_evaluated),
        }


def weld(positions: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices with bit-identical positions (``-0.0 == 0.0``): the topology of a
    render mesh whose seams split vertices."""
    p = np.ascontiguousarray(np.asarray(positions, dtype=np.float64) + 0.0)
    if not len(p):
        return p, np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    keys = p.view(np.dtype((np.void, 24))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    t = rank[inverse.reshape(-1)][np.asarray(triangles, dtype=np.int64).reshape(-1, 3)]
    return p[first[order]], t


def check_mesh(
    positions: np.ndarray,
    triangles: np.ndarray,
    *,
    self_intersections: bool = False,
    welded: bool = False,
) -> MeshReport:
    """Check a triangle mesh (``(n, 3)`` positions, ``(m, 3)`` vertex ids). Vertices with
    identical positions are welded first (a render mesh's seams are not boundaries)
    unless ``welded`` says they already are."""
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not welded:
        p, t = weld(p, t)
    diags: list[Diagnostic] = []

    def add(code: str, severity: str, message: str, kind: str, ids) -> None:
        ids = np.asarray(ids).reshape(-1)
        if len(ids):
            diags.append(
                Diagnostic(
                    code,
                    severity,
                    message,
                    kind,
                    int(len(ids)),
                    tuple(int(i) for i in ids[:MAX_SAMPLES]),
                )
            )

    finite = np.isfinite(p).all(axis=1)
    add(
        "NON_FINITE_POSITIONS",
        ERROR,
        "vertex positions are not finite",
        "vertices",
        np.flatnonzero(~finite),
    )
    used = np.unique(t.ravel()) if len(t) else np.zeros(0, dtype=np.int64)
    size = float(np.ptp(p[used], axis=0).max()) if len(used) and finite.all() else 1.0
    size = max(size, 1e-300)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.sqrt((cross * cross).sum(axis=1))
    repeated = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 0] == t[:, 2])
    degenerate = repeated | (area <= 1e-12 * size * size)
    add(
        "DEGENERATE_TRIANGLES",
        WARNING,
        "triangles repeat a vertex or have zero area",
        "triangles",
        np.flatnonzero(degenerate),
    )
    # Edges: undirected keys with their triangle counts; directed duplicates.
    m = len(t)
    he_a = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    he_b = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    nv = int(t.max()) + 1 if m else 0
    key = np.minimum(he_a, he_b) * max(nv, 1) + np.maximum(he_a, he_b)
    ukeys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    boundary = np.flatnonzero(counts == 1)
    non_manifold = np.flatnonzero(counts > 2)
    add("NON_MANIFOLD_EDGES", WARNING, "edges with more than two triangles", "edges", non_manifold)
    add("BOUNDARY_EDGES", INFO, "edges with one triangle (open surface)", "edges", boundary)
    dkey = he_a * max(nv, 1) + he_b
    du, dcounts = np.unique(dkey, return_counts=True)
    twice = du[dcounts > 1]
    inconsistent = np.unique(
        np.minimum(twice // max(nv, 1), twice % max(nv, 1)) * max(nv, 1)
        + np.maximum(twice // max(nv, 1), twice % max(nv, 1))
    )
    add(
        "INCONSISTENT_WINDING",
        WARNING,
        "triangles run a shared edge the same way",
        "edges",
        np.searchsorted(ukeys, inconsistent),
    )
    nm_vertices = _non_manifold_vertices(t, inverse, len(ukeys), counts)
    add(
        "NON_MANIFOLD_VERTICES",
        WARNING,
        "the triangles around a vertex form several fans",
        "vertices",
        nm_vertices,
    )
    lengths = (
        np.sqrt(((p[ukeys // max(nv, 1)] - p[ukeys % max(nv, 1)]) ** 2).sum(axis=1))
        if len(ukeys)
        else np.zeros(0)
    )
    closed = bool(m) and not len(boundary) and not len(non_manifold)
    summary: dict[str, Any] = {
        "vertex_count": int(len(used)),
        "triangle_count": int(m),
        "edge_count": int(len(ukeys)),
        "boundary_edge_count": int(len(boundary)),
        "non_manifold_edge_count": int(len(non_manifold)),
        "non_manifold_vertex_count": int(len(nm_vertices)),
        "degenerate_triangle_count": int(degenerate.sum()),
        "closed": closed,
        "edge_manifold": not len(non_manifold),
        "vertex_manifold": not len(nm_vertices),
        "consistently_oriented": not len(inconsistent),
        "euler_characteristic": int(len(used) - len(ukeys) + m),
        "edge_length": {
            "min": float(lengths.min()) if len(lengths) else 0.0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "max": float(lengths.max()) if len(lengths) else 0.0,
        },
        "min_angle_degrees": _min_angle(a, b, c, degenerate),
    }
    if closed and not len(inconsistent):
        summary["volume"] = float((a * np.cross(b, c)).sum() / 6.0)
    report = MeshReport(summary, diags)
    if self_intersections:
        pairs = _self_intersections(p, t, degenerate)
        summary["self_intersecting_pairs"] = int(len(pairs))
        if len(pairs):
            diags.append(
                Diagnostic(
                    "SELF_INTERSECTIONS",
                    ERROR,
                    "triangles of the surface intersect each other",
                    "triangles",
                    int(len(pairs)),
                    tuple(int(x) for x in pairs[:MAX_SAMPLES].ravel()[:MAX_SAMPLES]),
                )
            )
    else:
        report.not_evaluated.append("self-intersections (self_intersections=True checks them)")
    return report


def _min_angle(a, b, c, degenerate) -> float:
    if not len(a) or degenerate.all():
        return 0.0
    ok = ~degenerate
    a, b, c = a[ok], b[ok], c[ok]
    best = math.pi
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = q - p, r - p
        lu = np.sqrt((u * u).sum(axis=1))
        lv = np.sqrt((v * v).sum(axis=1))
        cos = np.clip((u * v).sum(axis=1) / (lu * lv), -1.0, 1.0)
        best = min(best, float(np.arccos(cos).min()))
    return math.degrees(best)


def _non_manifold_vertices(
    t: np.ndarray, edge_of_half: np.ndarray, n_edges: int, counts
) -> np.ndarray:
    """Vertices whose incident triangles form more than one fan. Two triangles around a
    vertex are joined when they share a manifold edge at it; fans are counted by label
    propagation over triangle corners (corner ``k * m + i``: triangle ``i`` at ``t[i, k]``)."""
    m = len(t)
    if not m:
        return np.zeros(0, dtype=np.int64)
    k_of = np.repeat(np.arange(3), m)
    tri_of = np.tile(np.arange(m), 3)
    start_v = t[tri_of, k_of]
    end_v = t[tri_of, (k_of + 1) % 3]
    start_c = k_of * m + tri_of
    end_c = ((k_of + 1) % 3) * m + tri_of
    manifold = np.flatnonzero(counts[edge_of_half] == 2)
    order = manifold[np.argsort(edge_of_half[manifold], kind="stable")]
    h1, h2 = order[0::2], order[1::2]
    consistent = start_v[h1] == end_v[h2]
    la = np.concatenate([start_c[h1], end_c[h1]])
    lb = np.concatenate(
        [
            np.where(consistent, end_c[h2], start_c[h2]),
            np.where(consistent, start_c[h2], end_c[h2]),
        ]
    )
    label = np.arange(3 * m)
    for _ in range(4 * m + 4):
        new = label.copy()
        low = np.minimum(new[la], new[lb])
        np.minimum.at(new, la, low)
        np.minimum.at(new, lb, low)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    corner_vertex = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    pairs = np.unique(np.stack([corner_vertex, label], axis=1), axis=0)
    fans = np.bincount(pairs[:, 0], minlength=int(t.max()) + 1)
    del n_edges
    return np.flatnonzero(fans > 1)


def _self_intersections(p: np.ndarray, t: np.ndarray, degenerate: np.ndarray) -> np.ndarray:
    """``(k, 2)`` triangle pairs that intersect improperly (see the module docs)."""
    from ..rendering.raytrace.bvh import build_bvh_morton, triangle_bounds
    from ..spatial.geometry import bvh_box_pairs, triangles_intersect
    from ..spatial.predicates import (
        _drop_axis,
        _project,
        orient2d,
        orient3d,
        segment_crosses_triangle,
    )

    keep = np.flatnonzero(~degenerate)
    if len(keep) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    tk = t[keep]
    lo, hi = triangle_bounds(p, tk)
    bvh = build_bvh_morton(lo, hi)
    found = []
    for s in range(0, len(tk), 4096):
        qi, pi = bvh_box_pairs(bvh, lo, hi, lo[s : s + 4096], hi[s : s + 4096])
        qi = qi + s
        upper = qi < pi
        qi, pi = qi[upper], pi[upper]
        if not len(qi):
            continue
        ta, tb = tk[qi], tk[pi]
        shared = (ta[:, :, None] == tb[:, None, :]).sum(axis=(1, 2))
        a1, b1, c1 = p[ta[:, 0]], p[ta[:, 1]], p[ta[:, 2]]
        a2, b2, c2 = p[tb[:, 0]], p[tb[:, 1]], p[tb[:, 2]]
        hit = np.zeros(len(qi), dtype=bool)
        none = shared == 0
        if none.any():
            hit[none] = triangles_intersect(
                a1[none], b1[none], c1[none], a2[none], b2[none], c2[none]
            )
        one = np.flatnonzero(shared == 1)
        if len(one):
            # The edge of each triangle away from the shared vertex must not meet the
            # other triangle (exact predicates: coplanar neighbours are common).
            for first, second in ((ta, tb), (tb, ta)):
                f, g = first[one], second[one]
                common = (f[:, :, None] == g[:, None, :]).any(axis=2)  # row by row
                k_common = np.argmax(common, axis=1)
                rows = np.arange(len(one))
                e0 = f[rows, (k_common + 1) % 3]
                e1 = f[rows, (k_common + 2) % 3]
                hit[one] |= segment_crosses_triangle(
                    p[e0], p[e1], p[g[:, 0]], p[g[:, 1]], p[g[:, 2]]
                )
        two = np.flatnonzero(shared == 2)
        if len(two):
            # Neighbours across an edge meet improperly only when coplanar and on the
            # same side of the edge (folded onto each other).
            f, g = ta[two], tb[two]
            in_g = (f[:, :, None] == g[:, None, :]).any(axis=2)
            k_f = np.argmin(in_g, axis=1)  # f's corner not in g
            in_f = (g[:, :, None] == f[:, None, :]).any(axis=2)
            k_g = np.argmin(in_f, axis=1)
            rows = np.arange(len(two))
            c_f = f[rows, k_f]
            c_g = g[rows, k_g]
            e0 = f[rows, (k_f + 1) % 3]
            e1 = f[rows, (k_f + 2) % 3]
            flat = orient3d(p[e0], p[e1], p[c_f], p[c_g]) == 0
            if flat.any():
                fr = np.flatnonzero(flat)
                axis = _drop_axis(p[e0[fr]], p[e1[fr]], p[c_f[fr]])
                pr = [_project(p[x[fr]], axis) for x in (e0, e1, c_f, c_g)]
                same = orient2d(pr[0], pr[1], pr[2]) * orient2d(pr[0], pr[1], pr[3]) > 0
                hit[two[fr[same]]] = True
        if hit.any():
            found.append(np.stack([keep[qi[hit]], keep[pi[hit]]], axis=1))
    return np.concatenate(found) if found else np.zeros((0, 2), dtype=np.int64)
