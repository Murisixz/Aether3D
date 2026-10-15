"""Ray queries on the Stage 8 two-level BVH: closest hits, every hit, rendered surfaces.

Closest hits are :meth:`SceneAccel.closest_hit <aether3d.rendering.raytrace.SceneAccel.closest_hit>`
itself. :func:`all_hits` walks the same structure without shrinking the ray at
each hit, and tests triangles with the same watertight test (Woop et al. 2013)
on the same object-space rays, computed with the same arithmetic -- so the
nearest of all hits is bitwise the closest hit.

**Shared edges.** The watertight test never lets a ray slip between adjacent
triangles, but a ray through a shared edge or vertex may hit every triangle
around it at the same distance. :func:`merge_coincident` keeps one hit per
object and distance: the one with the lowest triangle index.

**Rendered surfaces.** :func:`rendered_hits` traces with the path tracer's
:class:`~aether3d.rendering.pathtrace.scene.PathScene`, which lets rays pass where
the renderers draw nothing: texels below an alpha-masked material's cutoff,
the backs of single-sided materials and faces removed by a ``cull`` setting.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..errors import QueryError, RenderError
from ..rendering.raytrace.accel import (
    KIND_INNER,
    KIND_INSTANCES,
    KIND_TRIANGLES,
    SceneAccel,
    _box,
    _ray_setup,
    _triangles,
)

if TYPE_CHECKING:
    from .index import Entry, SpatialIndex

MERGE_TOLERANCE = 1e-9
"""Hits of one object closer than this (relative to the scale of the ray) are one hit."""


class RayHits:
    """Hits of a set of rays (one row per hit): ray, instance, triangle, distance, barycentrics."""

    __slots__ = ("ray", "inst", "prim", "t", "b")

    def __init__(self, ray, inst, prim, t, b) -> None:
        self.ray = ray
        self.inst = inst
        self.prim = prim
        self.t = t
        self.b = b

    @classmethod
    def empty(cls) -> RayHits:
        z = np.zeros(0, dtype=np.int64)
        return cls(z, z.copy(), z.copy(), np.zeros(0), np.zeros((0, 3)))

    def take(self, rows: np.ndarray) -> RayHits:
        return RayHits(self.ray[rows], self.inst[rows], self.prim[rows], self.t[rows], self.b[rows])

    def __len__(self) -> int:
        return int(self.ray.shape[0])


def object_rays(accel: SceneAccel, inst: np.ndarray, o_w: np.ndarray, d_w: np.ndarray):
    """World rays in the space of instances ``inst`` (the arithmetic of ``SceneAccel._enter``)."""
    m = accel.to_object[inst]
    o = np.empty_like(o_w)
    d = np.empty_like(d_w)
    for i in range(3):
        o[:, i] = (
            m[:, i, 0] * o_w[:, 0] + m[:, i, 1] * o_w[:, 1] + m[:, i, 2] * o_w[:, 2] + m[:, i, 3]
        )
        d[:, i] = m[:, i, 0] * d_w[:, 0] + m[:, i, 1] * d_w[:, 1] + m[:, i, 2] * d_w[:, 2]
    return o, d


def all_hits(
    accel: SceneAccel,
    o_w: np.ndarray,
    d_w: np.ndarray,
    tmin: np.ndarray,
    tmax: np.ndarray,
) -> RayHits:
    """Every intersection of every ray with a triangle in ``(tmin, tmax)`` (unsorted).

    A breadth-first walk over (ray, node) pairs: every node whose box a ray
    meets in range is entered, so no hit is missed where the closest-hit walk
    would have stopped.
    """
    o_w = np.asarray(o_w, dtype=np.float64)
    d_w = np.asarray(d_w, dtype=np.float64)
    n = o_w.shape[0]
    if n == 0 or not accel.root_lo[0] <= accel.root_hi[0]:
        return RayHits.empty()
    inv_w = _ray_setup(d_w)[0]
    finite = np.isfinite(o_w).all(axis=1) & np.isfinite(d_w).all(axis=1)
    _, ok = _box(accel.root_lo, accel.root_hi, o_w, inv_w, tmin, tmax)
    ray = np.flatnonzero(ok & finite)
    inst = np.full(ray.size, -1, dtype=np.int64)
    node = np.zeros(ray.size, dtype=np.int64)
    o, d, inv = o_w[ray], d_w[ray], inv_w[ray]
    found: list[tuple] = []
    while ray.size:
        kind = accel.kind[node]
        parts: list[tuple] = []
        sel = kind == KIND_INNER
        if sel.any():
            r, i, nd = ray[sel], inst[sel], node[sel]
            oo, dd, ii = o[sel], d[sel], inv[sel]
            _, hit = _box(
                accel.child_lo[nd],
                accel.child_hi[nd],
                oo[:, None, :],
                ii[:, None, :],
                tmin[r][:, None],
                tmax[r][:, None],
            )
            for j in (0, 1):
                m = hit[:, j]
                parts.append((r[m], i[m], accel.child[nd[m], j], oo[m], dd[m], ii[m]))
        sel = kind == KIND_INSTANCES
        if sel.any():
            r, nd = ray[sel], node[sel]
            first, count = accel.first[nd], accel.count[nd]
            for k in range(int(count.max())):
                has = count > k
                rr = r[has]
                ins = accel.slot_inst[first[has] + k]
                _, hit = _box(
                    accel.inst_lo[ins], accel.inst_hi[ins], o_w[rr], inv_w[rr], tmin[rr], tmax[rr]
                )
                rr, ins = rr[hit], ins[hit]
                if rr.size:
                    oo, dd = object_rays(accel, ins, o_w[rr], d_w[rr])
                    parts.append((rr, ins, accel.inst_root[ins], oo, dd, _ray_setup(dd)[0]))
        sel = kind == KIND_TRIANGLES
        if sel.any():
            r, i, nd = ray[sel], inst[sel], node[sel]
            first, count = accel.first[nd], accel.count[nd]
            seg = np.cumsum(count) - count
            total = int(count.sum())
            pair = np.repeat(np.arange(r.size), count)
            g = first[pair] + (np.arange(total) - seg[pair])
            oo, dd = o[sel][pair], d[sel][pair]
            _, kx, ky, kz, shear = _ray_setup(dd)
            t, b, hit = _triangles(accel.v0[g], accel.v1[g], accel.v2[g], oo, kx, ky, kz, shear)
            rr = r[pair]
            hit &= (t > tmin[rr]) & (t < tmax[rr])
            if hit.any():
                found.append((rr[hit], i[pair][hit], accel.tri_prim[g[hit]], t[hit], b[hit]))
        if parts:
            ray = np.concatenate([p[0] for p in parts])
            inst = np.concatenate([p[1] for p in parts])
            node = np.concatenate([p[2] for p in parts])
            o = np.concatenate([p[3] for p in parts])
            d = np.concatenate([p[4] for p in parts])
            inv = np.concatenate([p[5] for p in parts])
        else:
            ray = np.zeros(0, dtype=np.int64)
    if not found:
        return RayHits.empty()
    return RayHits(*(np.concatenate([f[k] for f in found]) for k in range(5)))


def merge_coincident(hits: RayHits, scale: np.ndarray) -> RayHits:
    """One hit per ray, instance and distance (see the module docstring), in no particular
    order. Hits of one instance whose distances differ by at most
    ``MERGE_TOLERANCE * (1 + t + scale[ray])`` form a run; the run's hit with the
    lowest triangle index stays."""
    if len(hits) < 2:
        return hits
    order = np.lexsort((hits.prim, hits.t, hits.inst, hits.ray))
    h = hits.take(order)
    tol = MERGE_TOLERANCE * (1.0 + np.abs(h.t) + scale[h.ray])
    same = (h.ray[1:] == h.ray[:-1]) & (h.inst[1:] == h.inst[:-1])
    starts = np.ones(len(h), dtype=bool)
    starts[1:] = ~(same & (h.t[1:] - h.t[:-1] <= tol[1:]))
    run = np.cumsum(starts) - 1
    by_triangle = np.lexsort((h.prim, run))
    first = np.ones(len(h), dtype=bool)
    first[1:] = run[by_triangle][1:] != run[by_triangle][:-1]
    return h.take(np.sort(by_triangle[first]))


# --------------------------------------------------------------------------- #
# Rendered surfaces
# --------------------------------------------------------------------------- #


def rendered_scene(index: SpatialIndex, entries: list[Entry]):
    """A PathScene of these (visible) entries, and the entry of each of its instances."""
    from ..rendering.camera import Camera, frame_from_matrix
    from ..rendering.pathtrace.scene import PathScene
    from ..rendering.scene_data import RenderCamera, extract_render_scene

    key = tuple(e.order for e in entries)
    cached = index._rendered.get(key)
    if cached is not None:
        return cached
    if index._render_scene is None:
        frame, _ = frame_from_matrix(np.eye(4))
        camera = RenderCamera(None, "spatial query", Camera(), frame)
        try:
            index._render_scene = extract_render_scene(index.scene, camera, default_lights=False)
        except RenderError as exc:  # e.g. a non-finite world matrix, which renders refuse too
            raise QueryError(exc.code, exc.message, exc.location, hint=exc.hint) from None
    rs = index._render_scene
    ids = {e.obj.id: e for e in entries}
    instances = tuple(i for i in rs.instances if i.object_id in ids)
    ps = PathScene(replace(rs, instances=instances))
    found = (ps, [ids[i.object_id] for i in instances])
    if len(index._rendered) >= 4:
        index._rendered.pop(next(iter(index._rendered)))
    index._rendered[key] = found
    return found


def rendered_filter(ps, hits: RayHits, o: np.ndarray, d: np.ndarray, cull: str = "none"):
    """The hits that rendered surfaces keep (drops pass-through ones)."""
    from ..rendering.raytrace.accel import Hits

    if not len(hits) or not (ps.pass_through or cull != "none"):
        return hits
    rows = np.arange(len(hits))
    sub = Hits(hits.t, hits.inst, hits.prim, hits.b)
    it = ps.interaction(rows, sub, o[hits.ray], d[hits.ray])
    through = ps.transparent(it, cull)
    return hits.take(np.flatnonzero(~through))
