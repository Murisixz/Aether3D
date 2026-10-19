"""Bone heat: smooth automatic skin weights by heat diffusion over the surface.

After Baran and Popovic, "Automatic Rigging and Animation of 3D Characters" (SIGGRAPH
2007), the method behind Blender's *Automatic Weights* -- re-derived and implemented here
from the paper (no code of Blender's is used). Each deforming bone is a heat source for
the vertices it is nearest to **and visible from**; heat then diffuses over the surface,
and a vertex's weight for a bone is its equilibrium temperature:

``(L + D) w_j = D p_j``

* ``L`` -- the cotangent Laplacian of the surface (``w_ab = (cot alpha + cot beta) / 2``
  per edge, each triangle's cotangent shared by the triangles of a non-manifold edge);
* ``D`` -- per vertex ``4 A_i / d_i^2``: ``A_i`` the vertex's mixed Voronoi area (Meyer
  et al. 2003), ``d_i`` its distance to the nearest bone -- lengthened when the vertex's
  normal faces away from that bone (``d / ((cos + 1.001) / 2)``), so the inside of a
  sleeve does not take the arm's other side for its own;
* ``p_j`` -- 1 (shared equally by ties) where bone ``j`` is the nearest bone and a straight
  segment from the vertex to that bone's nearest point does not enter the surface again
  (Aether3D's watertight ray traversal against the mesh), else 0.

Written this way the system is symmetric positive definite (Blender's is the same system
divided row by row by ``4 A_i``), so one sparse LDL^T factorization
(:mod:`aether3d.uv.sparse`, with its bitwise native twin) serves every bone; the weights
of a vertex sum to one before any cut (``L 1 = 0``). Small weights are then faded out
(below 0.025 nothing, up to 0.05 a ramp -- the soft cut Blender's results also show),
rows renormalized and capped to the influence limit.

What Aether3D adds around the solve: connected parts are solved together but a part
without any visible bone (a button inside a pocket, a detached prop) takes its nearest
bone in straight line, reported; degenerate triangles are left out of the Laplacian;
``diffusion`` scales the Laplacian (``1``: Blender's balance; larger: wider, softer
blends); the visibility test can be switched off for open or messy meshes.
"""

from __future__ import annotations

import math
import time
import types
from typing import Any

import numpy as np

from ..errors import AnimationError

WEIGHT_LIMIT_START = 0.05
WEIGHT_LIMIT_END = 0.025
TIE = 1e-4  # relative distance within which two bones are equally near
MAX_RAY_PASSES = 4  # exit crossings skipped per visibility ray before it gives up


def _err(code: str, message: str, location: str = "weights") -> AnimationError:
    return AnimationError(code, message, location)


# --------------------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------------------- #


def vertex_normals(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (unit; zero for unused vertices)."""
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    n = np.zeros_like(p)
    for c in range(3):
        n[:, 0] += np.bincount(t[:, c], weights=fn[:, 0], minlength=len(p))
        n[:, 1] += np.bincount(t[:, c], weights=fn[:, 1], minlength=len(p))
        n[:, 2] += np.bincount(t[:, c], weights=fn[:, 2], minlength=len(p))
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(length > 0, n / np.where(length > 0, length, 1.0), 0.0)


def closest_on_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple:
    """For every point and segment: the closest point's parameter ``(n, m)`` and the
    vector from it to the point ``(n, m, 3)``."""
    seg = b - a
    l2 = np.maximum(np.sum(seg * seg, axis=1), 1e-30)
    rel = p[:, None, :] - a[None]
    t = np.clip(np.sum(rel * seg[None], axis=2) / l2[None], 0.0, 1.0)
    return t, rel - t[..., None] * seg[None]


def laplacian(
    p: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The cotangent Laplacian's off-diagonal entries ``(rows, cols, values)`` (both
    triangles; the diagonal is minus each row's sum) and the mixed Voronoi areas."""
    n = len(p)
    v0, v1, v2 = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    double_area = np.linalg.norm(cross, axis=1)
    good = double_area > 1e-14 * max(1.0, float(np.max(double_area)) if len(t) else 1.0)
    tg = t[good]
    v0, v1, v2, da = v0[good], v1[good], v2[good], double_area[good]

    def cot(u: np.ndarray, v: np.ndarray, area2: np.ndarray) -> np.ndarray:
        return np.sum(u * v, axis=1) / area2

    c0 = cot(v1 - v0, v2 - v0, da)  # the angle at corner 0, facing edge (1, 2)
    c1 = cot(v2 - v1, v0 - v1, da)
    c2 = cot(v0 - v2, v1 - v2, da)
    # Each triangle's share of an edge: its cotangent divided by the edge's triangles.
    ea = np.concatenate([tg[:, 1], tg[:, 2], tg[:, 0]])
    eb = np.concatenate([tg[:, 2], tg[:, 0], tg[:, 1]])
    w = 0.5 * np.concatenate([c0, c1, c2])
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    key = lo * n + hi
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    w = w * (2.0 / counts[inverse])  # (manifold edges: count 2, as written above)
    rows = np.concatenate([ea, eb])
    cols = np.concatenate([eb, ea])
    vals = -np.concatenate([w, w])
    # Mixed Voronoi areas (Meyer et al. 2003): an obtuse triangle gives half its area to
    # the obtuse corner and a quarter to each other one.
    area = 0.5 * da
    e01 = np.sum((v1 - v0) ** 2, axis=1)
    e12 = np.sum((v2 - v1) ** 2, axis=1)
    e20 = np.sum((v0 - v2) ** 2, axis=1)
    vor0 = (e01 * c2 + e20 * c1) / 8.0
    vor1 = (e12 * c0 + e01 * c2) / 8.0
    vor2 = (e20 * c1 + e12 * c0) / 8.0
    obtuse = np.stack([c0 < 0, c1 < 0, c2 < 0], axis=1)
    any_obtuse = obtuse.any(axis=1)
    shares = np.where(any_obtuse[:, None], np.where(obtuse, 0.5, 0.25) * area[:, None],
                      np.stack([vor0, vor1, vor2], axis=1))  # fmt: skip
    areas = np.zeros(n)
    for c in range(3):
        areas += np.bincount(tg[:, c], weights=shares[:, c], minlength=n)
    return rows, cols, vals, areas


# --------------------------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------------------------- #


class _Rays:
    """Segments from surface points that must not enter the surface again."""

    def __init__(self, p: np.ndarray, t: np.ndarray) -> None:
        from ..rendering.raytrace.accel import BLASCache, SceneAccel
        from ..rendering.scene_data import RenderMesh

        pos = np.ascontiguousarray(p.astype(np.float32))
        tris = np.ascontiguousarray(t.astype(np.uint32))
        pos.setflags(write=False)
        tris.setflags(write=False)
        rs = types.SimpleNamespace(
            meshes=(RenderMesh(pos, tris, None),),
            instances=(types.SimpleNamespace(mesh=0, world=np.eye(4)),),
        )
        self.accel = SceneAccel.build(rs, cache=BLASCache(budget=512 * 2**20))
        self.normals = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])

    def clear(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Whether each segment ``start -> end`` reaches its end without entering the
        surface (crossing a triangle against its normal); leaving it is allowed."""
        d = end - start
        length = np.linalg.norm(d, axis=1)
        ok = length > 1e-12
        out = np.ones(len(start), bool)
        idx = np.flatnonzero(ok)
        if not idx.size:
            return out
        o = start[idx]
        u = d[idx] / length[idx, None]
        tmin = np.full(idx.size, 1e-5 * float(np.median(length[idx])))
        tmax = length[idx] * (1.0 - 2e-5)
        skip = np.full(idx.size, -1, np.int64)
        alive = np.ones(idx.size, bool)
        for _ in range(MAX_RAY_PASSES):
            a = np.flatnonzero(alive)
            if not a.size:
                break
            hits = self.accel.closest_hit(
                np.ascontiguousarray(o[a]),
                np.ascontiguousarray(u[a]),
                np.ascontiguousarray(tmin[a]),
                np.ascontiguousarray(tmax[a]),
                np.zeros(a.size, np.int64),
                np.ascontiguousarray(skip[a]),
            )
            hit = hits.inst >= 0
            done = a[~hit]
            alive[done] = False  # (nothing in the way: visible)
            h = a[hit]
            if not h.size:
                break
            facing = np.einsum("ij,ij->i", self.normals[hits.prim[hit]], u[h])
            entering = facing < 0.0
            blocked = h[entering]
            out[idx[blocked]] = False
            alive[blocked] = False
            # Leaving the surface (an exit): continue past that triangle.
            go = h[~entering]
            tmin[go] = hits.t[hit][~entering]
            skip[go] = hits.prim[hit][~entering]
        return out


# --------------------------------------------------------------------------------------- #
# The solve
# --------------------------------------------------------------------------------------- #


def heat_weights(
    positions: np.ndarray,
    triangles: np.ndarray,
    heads: np.ndarray,
    ends: np.ndarray,
    *,
    allowed: np.ndarray | None = None,
    visibility: bool = True,
    diffusion: float = 1.0,
    deadline: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``(weights (vertices, bones), info)`` by bone heat (see the module docstring).
    ``allowed`` (``(vertices, bones)`` bool): bones that may be a vertex's heat source
    (the side rule); weights still diffuse everywhere the surface connects."""
    from ..uv import sparse
    from .voxels import mesh_components

    t0 = time.perf_counter()
    p = np.asarray(positions, np.float64)
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    a = np.asarray(heads, np.float64).reshape(-1, 3)
    b = np.asarray(ends, np.float64).reshape(-1, 3)
    n, m = len(p), len(a)
    if m == 0:
        raise _err("INVALID_SKIN", "no bones to weigh")
    if not (isinstance(diffusion, (int, float)) and math.isfinite(diffusion) and diffusion > 0):
        raise _err("INVALID_ARGUMENT", "diffusion is a positive number", "diffusion")
    normals = vertex_normals(p, t)
    used = np.zeros(n, bool)
    used[t.reshape(-1)] = True
    # Distances to every bone, lengthened where the normal faces away from it.
    dist = np.empty((n, m))
    near_pt = np.empty((n, m, 3))
    for s in range(0, n, 16384):
        sl = slice(s, min(n, s + 16384))
        tt, rel = closest_on_segments(p[sl], a, b)
        d = np.linalg.norm(rel, axis=2)
        cosine = np.einsum("ijk,ik->ij", rel / np.maximum(d[..., None], 1e-30), normals[sl])
        dist[sl] = d / (0.5 * (cosine + 1.001))
        near_pt[sl] = a[None] + tt[..., None] * (b - a)[None]
    if allowed is not None:
        dist = np.where(allowed, dist, np.inf)
    mind = dist.min(axis=1)
    nearest = dist <= mind[:, None] * (1.0 + TIE)
    nearest &= np.isfinite(dist)
    t1 = time.perf_counter()
    rays = 0
    if visibility:
        tracer = _Rays(p, t)
        vi, bj = np.nonzero(nearest)
        rays = int(vi.size)
        clear = tracer.clear(p[vi], near_pt[vi, bj])
        nearest[vi[~clear], bj[~clear]] = False
    t2 = time.perf_counter()
    count = nearest.sum(axis=1)
    rows, cols, vals, areas = laplacian(p, t)
    md = np.maximum(mind, 1e-4)
    heat = np.where(count > 0, 4.0 * areas / (md * md), 0.0) * count  # (count ties x 1/count)
    comp, n_comp = mesh_components(n, t)
    # Parts with no visible bone at all take their nearest bone whole (reported).
    has = np.bincount(comp, weights=(heat > 0).astype(np.float64), minlength=n_comp) > 0
    dead = ~has[comp] | ~used
    weights = np.zeros((n, m))
    fallback = 0
    if (~has).any():
        for c in np.flatnonzero(~has).tolist():
            members = np.flatnonzero(comp == c)
            if not members.size:
                continue
            j = int(
                np.argmin(np.where(np.isfinite(dist[members]), dist[members], np.inf).min(axis=0))
            )
            weights[members, j] = 1.0
            fallback += 1
    live = np.flatnonzero(~dead)
    if live.size:
        index = np.full(n, -1, np.int64)
        index[live] = np.arange(live.size)
        keep = (index[rows] >= 0) & (index[cols] >= 0)
        r, c, v = index[rows[keep]], index[cols[keep]], diffusion * vals[keep]
        diag = -np.bincount(r, weights=v, minlength=live.size)  # (minus the row sums)
        matrix = sparse.assemble(
            live.size,
            np.concatenate([r, np.arange(live.size)]),
            np.concatenate([c, np.arange(live.size)]),
            np.concatenate([v, diag + heat[live]]),
        )
        if deadline is not None:
            deadline.check("the heat system")
        rhs = np.where(nearest[live], (heat[live] / np.maximum(count[live], 1))[:, None], 0.0)
        try:
            perm = sparse.nested_dissection(p[live], matrix.indptr, matrix.indices, comp[live])
            fac = sparse.factor(sparse.analyze(matrix, perm), matrix)
            sol = fac.solve(rhs)
        except Exception as exc:  # noqa: BLE001 - reported as the rigging error it is
            raise _err("WEIGHTS_FAILED", f"the heat system could not be solved: {exc}") from None
        weights[live] = sol
    t3 = time.perf_counter()
    # The soft cut of small weights, then rows renormalized.
    w = np.clip(weights, 0.0, None)
    cut = np.where(
        w < WEIGHT_LIMIT_END,
        0.0,
        np.where(
            w < WEIGHT_LIMIT_START,
            (w - WEIGHT_LIMIT_END) / (WEIGHT_LIMIT_START - WEIGHT_LIMIT_END) * WEIGHT_LIMIT_START,
            w,
        ),
    )
    lost = cut.sum(axis=1) <= 0
    cut[lost] = w[lost]
    total = cut.sum(axis=1, keepdims=True)
    lost = total[:, 0] <= 0
    if lost.any():  # (an isolated vertex: its nearest bone)
        j = np.argmin(np.where(np.isfinite(dist[lost]), dist[lost], np.inf), axis=1)
        cut[np.flatnonzero(lost), j] = 1.0
        total = cut.sum(axis=1, keepdims=True)
    out = cut / total
    info = {
        "technique": "bone heat: cotangent-Laplacian diffusion from the nearest visible bone "
        "(Baran and Popovic 2007), one sparse LDL^T for all bones",
        "rays": rays,
        "vertices_without_source": int(((count == 0) & used).sum()),
        "parts_without_bone": fallback,
        "diffusion": float(diffusion),
        "visibility": bool(visibility),
        "timings": {
            "distances_s": round(t1 - t0, 4),
            "visibility_s": round(t2 - t1, 4),
            "solve_s": round(t3 - t2, 4),
        },
    }
    return out, info


__all__ = ["heat_weights", "laplacian", "vertex_normals"]
