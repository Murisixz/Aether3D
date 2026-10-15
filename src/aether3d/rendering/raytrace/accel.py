"""Two-level ray tracing acceleration for render scenes, and vectorized ray queries.

**Bottom level.** Every distinct mesh gets a :class:`MeshBLAS`: a :class:`BVH`
over its triangles in object space, plus the triangle corners in leaf order.
BLASes are cached by the identity of the mesh's read-only arrays (shared
meshes and repeated renders reuse them; an edited or re-evaluated mesh is a
new array and gets a new BLAS), with weak references and an LRU byte budget
(``AETHER3D_RAYTRACE_CACHE_MB``, default 1024).

**Top level.** A :class:`SceneAccel` builds a BVH over the world-space boxes of
the scene's instances and lays every node of the top level and of all
bottom levels out in one global table, so one traversal loop serves both:
inner nodes are tested the same way in either space, a top-level leaf pushes
its instances' bottom-level roots, and a ray is transformed into an
instance's object space (with the inverse of its world matrix -- rotations,
non-uniform and negative scales alike) when it starts popping that
instance's nodes. Object space keeps the ray parameter ``t``, so hits from
different instances compare directly.

**Traversal** is vectorized over rays ("wavefront"): every ray visits one
``(instance, node)`` per iteration and keeps a stack of entries with the
distance at which each node's box is entered. At an inner node it tests both
children's boxes, goes on to the nearer one hit and pushes the other; at a
leaf it tests the triangles (or the instances' boxes), then pops. Entries
farther than the closest hit found so far are skipped when popped.

**Triangles** are intersected with the watertight algorithm of Woop, Benthin
and Wald (JCGT 2013): the ray is made the +z axis by a permutation and a
shear, and the signs of three 2D edge functions decide the hit, so a ray
crossing a shared edge or vertex hits at least one of the adjacent triangles
(no cracks) and the test is consistent for both windings (two-sided).
Object-space corners and boxes are stored as float32 (the mesh's own
positions, exactly; top-level boxes are rounded outward); all arithmetic is
float64.

**Self-intersection.** Queries accept, per ray, an instance and triangle to
ignore: a ray leaving a triangle cannot hit that (planar) triangle again,
so excluding it is exact; callers also offset origins along the geometric
normal against neighbouring triangles.
"""

from __future__ import annotations

import os
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .bvh import BVH, LEAF_SIZE, build_bvh, triangle_bounds

if TYPE_CHECKING:
    from ..scene_data import RenderMesh, RenderScene

CACHE_ENV = "AETHER3D_RAYTRACE_CACHE_MB"
DEFAULT_CACHE_BUDGET = 1024 * 2**20
BATCH = 65_536
"""Rays per traversal batch (bounds temporary memory, keeps arrays cache-friendly)."""
MIN_DIRECTION = 1e-300
"""Direction components smaller than this are replaced (signed) before inverting."""
KIND_INNER, KIND_TRIANGLES, KIND_INSTANCES = 0, 1, 2


@dataclass(frozen=True)
class MeshBLAS:
    """The bottom level of one mesh: its BVH and triangle corners in leaf order.

    Corners and boxes are float32 in object space -- exactly the mesh's own
    float32 positions -- and every test promotes them to float64.
    """

    bvh: BVH
    v0: np.ndarray  # (k, 3) float32 object space, leaf order
    v1: np.ndarray
    v2: np.ndarray
    prim: np.ndarray  # (k,) the mesh triangle of each leaf-order triangle
    skipped: int  # triangles left out (non-finite corners)
    build_seconds: float

    @property
    def nbytes(self) -> int:
        return int(self.bvh.nbytes + 3 * self.v0.nbytes + self.prim.nbytes)


def build_mesh_blas(mesh: RenderMesh) -> MeshBLAS:
    """BLAS of a render mesh: triangles with non-finite corners are left out."""
    t0 = time.perf_counter()
    positions = np.asarray(mesh.positions, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    lo, hi = triangle_bounds(positions, tris)
    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    if finite.all():
        keep = None
        bvh = build_bvh(lo, hi)
        prim = bvh.order
    else:
        keep = np.flatnonzero(finite)
        bvh = build_bvh(lo[keep], hi[keep])
        prim = keep[bvh.order]
    del lo, hi
    corners = [np.ascontiguousarray(positions[tris[prim, k]]) for k in range(3)]
    return MeshBLAS(
        bvh,
        *corners,
        prim,
        0 if keep is None else int(tris.shape[0] - keep.size),
        time.perf_counter() - t0,
    )


def _round_out(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float64 boxes as float32 boxes that contain them."""
    lo32 = lo.astype(np.float32)
    hi32 = hi.astype(np.float32)
    up = lo32 > lo
    lo32[up] = np.nextafter(lo32[up], np.float32(-np.inf))
    down = hi32 < hi
    hi32[down] = np.nextafter(hi32[down], np.float32(np.inf))
    return lo32, hi32


class BLASCache:
    """BLASes by mesh array identity, with weak references and an LRU byte budget."""

    def __init__(self, budget: int | None = None) -> None:
        if budget is None:
            env = os.environ.get(CACHE_ENV)
            budget = int(float(env) * 2**20) if env else DEFAULT_CACHE_BUDGET
        self.budget = budget
        self._entries: OrderedDict[tuple[int, int], MeshBLAS] = OrderedDict()
        self._refs: dict[tuple[int, int], tuple] = {}
        self.bytes = 0
        self.hits = 0
        self.builds = 0
        self.evictions = 0

    def get(self, mesh: RenderMesh, keep: set[tuple[int, int]] | None = None) -> MeshBLAS:
        key = (id(mesh.positions), id(mesh.triangles))
        blas = self._entries.get(key)
        if blas is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return blas
        blas = build_mesh_blas(mesh)
        self.builds += 1
        if mesh.positions.flags.writeable or mesh.triangles.flags.writeable:
            return blas  # arrays that can still change are never cached
        self._entries[key] = blas
        self.bytes += blas.nbytes
        drop = self._drop
        self._refs[key] = (
            weakref.ref(mesh.positions, lambda _r, k=key: drop(k)),
            weakref.ref(mesh.triangles, lambda _r, k=key: drop(k)),
        )
        self._trim(keep or {key})
        return blas

    def _drop(self, key: tuple[int, int]) -> None:
        blas = self._entries.pop(key, None)
        self._refs.pop(key, None)
        if blas is not None:
            self.bytes -= blas.nbytes

    def _trim(self, keep: set[tuple[int, int]]) -> None:
        while self.bytes > self.budget:
            victim = next((k for k in self._entries if k not in keep), None)
            if victim is None:
                return
            self._drop(victim)
            self.evictions += 1

    def clear(self) -> None:
        for key in list(self._entries):
            self._drop(key)

    def stats(self) -> dict[str, int]:
        return {
            "meshes": len(self._entries),
            "bytes": self.bytes,
            "budget": self.budget,
            "hits": self.hits,
            "builds": self.builds,
            "evictions": self.evictions,
        }


BLAS_CACHE = BLASCache()
"""The process-wide BLAS cache (renders share it)."""


@dataclass
class Hits:
    """Closest hits of a set of rays (``inst == -1``: no hit)."""

    t: np.ndarray  # (n,) ray parameter (inf: no hit)
    inst: np.ndarray  # (n,) int64 instance index into RenderScene.instances
    prim: np.ndarray  # (n,) int64 triangle of the instance's mesh
    b: np.ndarray  # (n, 3) barycentrics of the triangle's corners 0, 1, 2

    @property
    def hit(self) -> np.ndarray:
        return self.inst >= 0


@dataclass
class TraversalStats:
    rays: int = 0
    shadow_rays: int = 0
    iterations: int = 0
    node_visits: int = 0
    triangle_tests: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


@dataclass
class SceneAccel:
    """The acceleration structure of one render scene (see the module docstring)."""

    child: np.ndarray  # (K, 2) global node ids
    child_lo: np.ndarray  # (K, 2, 3) float32 (top-level boxes rounded outward)
    child_hi: np.ndarray
    kind: np.ndarray  # (K,) KIND_*
    first: np.ndarray  # (K,) leaves: first triangle / first top-level slot
    count: np.ndarray  # (K,)
    v0: np.ndarray  # (T, 3) float32 all triangles, object space, leaf order
    v1: np.ndarray
    v2: np.ndarray
    tri_prim: np.ndarray  # (T,) mesh triangle of each global triangle
    slot_inst: np.ndarray  # (I',) instances in top-level leaf order
    inst_lo: np.ndarray  # (I, 3) world bounds of each instance
    inst_hi: np.ndarray
    inst_root: np.ndarray  # (I,) global id of the instance's mesh BLAS root
    to_object: np.ndarray  # (I, 3, 4) world -> object affine maps
    root_lo: np.ndarray  # (3,) world bounds of the scene
    root_hi: np.ndarray
    stack_size: int  # traversal stack entries per ray (see build)
    top_nodes: int = 0  # nodes 0 .. top_nodes - 1 are the top level (world space)
    traced: TraversalStats = field(default_factory=TraversalStats)
    info: dict = field(default_factory=dict)
    blas: tuple[MeshBLAS, ...] = ()

    # -- construction --------------------------------------------------------------------

    @classmethod
    def build(cls, rs: RenderScene, cache: BLASCache | None = None) -> SceneAccel:
        cache = BLAS_CACHE if cache is None else cache
        t0 = time.perf_counter()
        hits0, builds0 = cache.hits, cache.builds
        keys = {(id(m.positions), id(m.triangles)) for m in rs.meshes}
        mesh_blas = [cache.get(m, keys) for m in rs.meshes]
        t_blas = time.perf_counter() - t0
        # Instances: world bounds (the 8 corners of the mesh box through the world matrix),
        # inverse affine maps; singular or non-finite instances cannot be hit.
        n_inst = len(rs.instances)
        to_object = np.zeros((n_inst, 3, 4))
        inst_lo = np.full((n_inst, 3), np.inf)
        inst_hi = np.full((n_inst, 3), -np.inf)
        usable = np.zeros(n_inst, dtype=bool)
        for k, inst in enumerate(rs.instances):
            blas = mesh_blas[inst.mesh]
            m = inst.world
            if blas.bvh.order.size == 0 or not np.all(np.isfinite(m)):
                continue
            linear = m[:3, :3]
            det = float(np.linalg.det(linear))
            if not np.isfinite(det) or abs(det) < 1e-300:
                continue
            inverse = np.linalg.inv(linear)
            to_object[k, :, :3] = inverse
            to_object[k, :, 3] = -inverse @ m[:3, 3]
            lo, hi = blas.bvh.lo, blas.bvh.hi
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            world = corners @ linear.T + m[:3, 3]
            if not np.all(np.isfinite(world)):
                continue
            inst_lo[k], inst_hi[k] = world.min(axis=0), world.max(axis=0)
            usable[k] = True
        ids = np.flatnonzero(usable)
        top = build_bvh(inst_lo[ids], inst_hi[ids])
        slot_inst = ids[top.order]
        # Global node table: the top level first, then every mesh's bottom level.
        parts_child = [top.child]
        top_lo, top_hi = _round_out(top.child_lo, top.child_hi)
        parts_lo, parts_hi = [top_lo], [top_hi]
        kind = [np.where(top.child[:, 0] < 0, KIND_INSTANCES, KIND_INNER).astype(np.int8)]
        first, count = [top.first], [top.count]
        v0, v1, v2, prim = [], [], [], []
        node_base = top.node_count
        tri_base = 0
        mesh_root = np.zeros(len(mesh_blas), dtype=np.int64)
        for j, blas in enumerate(mesh_blas):
            b = blas.bvh
            mesh_root[j] = node_base
            parts_child.append(np.where(b.child >= 0, b.child + node_base, -1))
            parts_lo.append(b.child_lo)
            parts_hi.append(b.child_hi)
            kind.append(np.where(b.child[:, 0] < 0, KIND_TRIANGLES, KIND_INNER).astype(np.int8))
            first.append(b.first + tri_base)
            count.append(b.count)
            v0.append(blas.v0)
            v1.append(blas.v1)
            v2.append(blas.v2)
            prim.append(blas.prim)
            node_base += b.node_count
            tri_base += blas.prim.shape[0]
        inst_root = np.array([mesh_root[i.mesh] for i in rs.instances], dtype=np.int64)
        empty3 = np.zeros((0, 3), dtype=np.float32)
        accel = cls(
            child=np.concatenate(parts_child),
            child_lo=np.concatenate(parts_lo),
            child_hi=np.concatenate(parts_hi),
            kind=np.concatenate(kind),
            first=np.concatenate(first),
            count=np.concatenate(count),
            v0=np.concatenate(v0) if v0 else empty3,
            v1=np.concatenate(v1) if v1 else empty3,
            v2=np.concatenate(v2) if v2 else empty3,
            tri_prim=np.concatenate(prim) if prim else np.zeros(0, dtype=np.int64),
            slot_inst=slot_inst,
            inst_lo=inst_lo,
            inst_hi=inst_hi,
            inst_root=inst_root,
            to_object=to_object,
            root_lo=top.lo,
            root_hi=top.hi,
            # Descending one level pops one entry and pushes at most two; a top-level
            # leaf pushes up to LEAF_SIZE instance roots.
            stack_size=top.depth + LEAF_SIZE + max([b.bvh.depth for b in mesh_blas], default=0) + 2,
            top_nodes=top.node_count,
            blas=tuple(mesh_blas),
        )
        skipped = sum(b.skipped for b in mesh_blas)
        accel.info = {
            "instances": n_inst,
            "instances_traced": int(ids.size),
            "meshes": len(mesh_blas),
            "triangles": int(tri_base),
            "triangles_skipped": int(skipped),
            "blas_built": cache.builds - builds0,
            "blas_reused": cache.hits - hits0,
            "nodes": int(accel.child.shape[0]),
            "max_depth": int(max([top.depth] + [b.bvh.depth for b in mesh_blas])),
            "bytes": int(sum(b.nbytes for b in mesh_blas) + top.nbytes),
            "seconds_blas": t_blas,
            "seconds": time.perf_counter() - t0,
        }
        return accel

    # -- queries ------------------------------------------------------------------------

    def closest_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        tmin: np.ndarray | float = 0.0,
        tmax: np.ndarray | float = np.inf,
        skip_inst: np.ndarray | None = None,
        skip_tri: np.ndarray | None = None,
    ) -> Hits:
        """Nearest hit in ``(tmin, tmax)`` of every ray (``direction`` need not be unit)."""
        n = origin.shape[0]
        out = Hits(
            np.full(n, np.inf),
            np.full(n, -1, dtype=np.int64),
            np.full(n, -1, dtype=np.int64),
            np.zeros((n, 3)),
        )
        tmin_a = np.broadcast_to(np.asarray(tmin, dtype=np.float64), (n,))
        tmax_a = np.broadcast_to(np.asarray(tmax, dtype=np.float64), (n,))
        t0 = time.perf_counter()
        for s in range(0, n, BATCH):
            sl = slice(s, min(n, s + BATCH))
            self._trace(
                origin[sl],
                direction[sl],
                tmin_a[sl],
                tmax_a[sl],
                None if skip_inst is None else skip_inst[sl],
                None if skip_tri is None else skip_tri[sl],
                out,
                sl,
                any_hit=False,
            )
        self.traced.rays += n
        self.traced.seconds += time.perf_counter() - t0
        return out

    def occluded(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        tmin: np.ndarray | float = 0.0,
        tmax: np.ndarray | float = np.inf,
        skip_inst: np.ndarray | None = None,
        skip_tri: np.ndarray | None = None,
    ) -> np.ndarray:
        """Whether anything lies in ``(tmin, tmax)`` along each ray (any-hit query)."""
        n = origin.shape[0]
        out = Hits(
            np.full(n, np.inf),
            np.full(n, -1, dtype=np.int64),
            np.full(n, -1, dtype=np.int64),
            np.zeros((n, 3)),
        )
        tmin_a = np.broadcast_to(np.asarray(tmin, dtype=np.float64), (n,))
        tmax_a = np.broadcast_to(np.asarray(tmax, dtype=np.float64), (n,))
        t0 = time.perf_counter()
        for s in range(0, n, BATCH):
            sl = slice(s, min(n, s + BATCH))
            self._trace(
                origin[sl],
                direction[sl],
                tmin_a[sl],
                tmax_a[sl],
                None if skip_inst is None else skip_inst[sl],
                None if skip_tri is None else skip_tri[sl],
                out,
                sl,
                any_hit=True,
            )
        self.traced.shadow_rays += n
        self.traced.seconds += time.perf_counter() - t0
        return out.inst >= 0

    # -- the traversal loop ---------------------------------------------------------------

    def _trace(self, o_w, d_w, tmin, tmax, skip_inst, skip_tri, out, sl, any_hit):
        """Trace one batch of rays; results go to rows ``sl`` of ``out``.

        Every ray runs a small state machine, one node per iteration: ``node``
        (``(instance + 1) << 32 | node id``, entered at ``t_node``) is
        processed; at an inner node whose children are both hit, the farther
        child is pushed and the nearer visited next; at a leaf or a miss the
        ray pops its stack (entries already farther than the closest hit are
        discarded when visited). The stacks are stored level-major
        (``stack[level, ray]``): rays of one iteration sit at similar levels.
        """
        n = o_w.shape[0]
        o_w = np.asarray(o_w, dtype=np.float64)
        d_w = np.asarray(d_w, dtype=np.float64)
        t_best = tmax.copy()
        hit_tri = np.full(n, -1, dtype=np.int64)
        hit_inst = np.full(n, -1, dtype=np.int64)
        hit_b = np.zeros((n, 3))
        finite = np.isfinite(o_w).all(axis=1) & np.isfinite(d_w).all(axis=1)
        # Current ray space per ray (-1: world) and its derived quantities.
        cur = np.full(n, -1, dtype=np.int64)
        o = o_w.copy()
        d = d_w.copy()
        inv, kx, ky, kz, shear = _ray_setup(d)
        stack = np.empty((self.stack_size, n), dtype=np.int64)
        tstack = np.empty((self.stack_size, n))
        sp = np.zeros(n, dtype=np.int64)
        node = np.zeros(n, dtype=np.int64)  # the top-level root, in world space
        t_node = np.zeros(n)
        if self.root_lo[0] <= self.root_hi[0]:
            t_node, ok = _box(self.root_lo, self.root_hi, o, inv, tmin, t_best)
            active = np.flatnonzero(ok & finite)
        else:
            active = np.zeros(0, dtype=np.int64)
        stats = self.traced
        while active.size:
            stats.iterations += 1
            stats.node_visits += active.size
            a = active
            entry = node[a]
            live = t_node[a] < t_best[a]
            need_pop = ~live
            at = np.flatnonzero(live)  # positions in `a` of the rays visiting a node
            al = a[at]
            entry = entry[at]
            inst = (entry >> 32) - 1
            nd = entry & 0xFFFFFFFF
            moved = inst != cur[al]
            if moved.any():
                self._enter(al[moved], inst[moved], cur, o, d, o_w, d_w, inv, kx, ky, kz, shear)
            kind = self.kind[nd]
            # Inner nodes: test both children; visit the nearer hit one, push the other.
            sel = kind == KIND_INNER
            if sel.any():
                ai, ni = al[sel], nd[sel]
                tag = (inst[sel] + 1) << 32
                tn, ok = _box(
                    self.child_lo[ni],
                    self.child_hi[ni],
                    o[ai][:, None, :],
                    inv[ai][:, None, :],
                    tmin[ai][:, None],
                    t_best[ai][:, None],
                )
                kids = self.child[ni]
                ok0, ok1 = ok[:, 0], ok[:, 1]
                both = ok0 & ok1
                near0 = np.where(both, tn[:, 0] <= tn[:, 1], ok0)  # child 0 is visited next
                hit_any = ok0 | ok1
                go = ai[hit_any]
                pick = near0[hit_any]
                node[go] = tag[hit_any] | np.where(pick, kids[hit_any, 0], kids[hit_any, 1])
                t_node[go] = np.where(pick, tn[hit_any, 0], tn[hit_any, 1])
                if both.any():
                    pb = ai[both]
                    far0 = ~near0[both]
                    spb = sp[pb]
                    stack[spb, pb] = tag[both] | np.where(far0, kids[both, 0], kids[both, 1])
                    tstack[spb, pb] = np.where(far0, tn[both, 0], tn[both, 1])
                    sp[pb] = spb + 1
                need_pop[at[sel][~hit_any]] = True
            # Top-level leaves: push the bottom-level root of every instance whose box is hit.
            sel = kind == KIND_INSTANCES
            if sel.any():
                ai, ni = al[sel], nd[sel]
                f, c = self.first[ni], self.count[ni]
                for k in range(int(c.max())):
                    has = c > k
                    rr = ai[has]
                    ins = self.slot_inst[f[has] + k]
                    tn, ok = _box(
                        self.inst_lo[ins], self.inst_hi[ins], o_w[rr], inv[rr], tmin[rr], t_best[rr]
                    )
                    rr, ins = rr[ok], ins[ok]
                    if rr.size:
                        spr = sp[rr]
                        stack[spr, rr] = ((ins + 1) << 32) | self.inst_root[ins]
                        tstack[spr, rr] = tn[ok]
                        sp[rr] = spr + 1
                need_pop[at[sel]] = True
            # Bottom-level leaves: every (ray, triangle) pair at once, watertight tests in
            # object space; per ray the nearest hit wins (ties: the first in leaf order).
            sel = kind == KIND_TRIANGLES
            if sel.any():
                ai, ni, ii = al[sel], nd[sel], inst[sel]
                f, c = self.first[ni], self.count[ni]
                seg = np.cumsum(c) - c
                total = int(c.sum())
                pair_ray = np.repeat(np.arange(ai.size), c)
                g = f[pair_ray] + (np.arange(total) - seg[pair_ray])
                rr = ai[pair_ray]
                stats.triangle_tests += total
                t, b, ok = _triangles(
                    self.v0[g], self.v1[g], self.v2[g], o[rr], kx[rr], ky[rr], kz[rr], shear[rr]
                )
                ok &= (t > tmin[rr]) & (t < t_best[rr])
                if skip_tri is not None:
                    ok &= ~((ii[pair_ray] == skip_inst[rr]) & (self.tri_prim[g] == skip_tri[rr]))
                if ok.any():
                    tt = np.where(ok, t, np.inf)
                    best = np.minimum.reduceat(tt, seg)
                    pos = np.where(ok & (tt == best[pair_ray]), np.arange(total), total)
                    first_pair = np.minimum.reduceat(pos, seg)
                    win = first_pair < total
                    fp = first_pair[win]
                    rw = ai[win]
                    t_best[rw] = tt[fp]
                    hit_tri[rw] = g[fp]
                    hit_inst[rw] = ii[win]
                    hit_b[rw] = b[fp]
                    if any_hit:
                        sp[rw] = 0
                need_pop[at[sel]] = True
            # Pop the next entry of every ray that needs one; rays with empty stacks finish.
            if need_pop.any():
                where = np.flatnonzero(need_pop)
                rp = a[where]
                more = sp[rp] > 0
                rp = rp[more]
                level = sp[rp] - 1
                sp[rp] = level
                node[rp] = stack[level, rp]
                t_node[rp] = tstack[level, rp]
                if not more.all():
                    keep = np.ones(a.size, dtype=bool)
                    keep[where[~more]] = False
                    active = a[keep]
        found = hit_inst >= 0
        idx = np.arange(sl.start, sl.start + n)[found]
        out.t[idx] = t_best[found]
        out.inst[idx] = hit_inst[found]
        out.prim[idx] = self.tri_prim[hit_tri[found]]
        out.b[idx] = hit_b[found]

    def _enter(self, rays, inst, cur, o, d, o_w, d_w, inv, kx, ky, kz, shear):
        """Move ``rays`` into the space of ``inst`` (``-1``: world)."""
        world = inst < 0
        if world.any():
            rw = rays[world]
            o[rw] = o_w[rw]
            d[rw] = d_w[rw]
        if not world.all():
            ro = rays[~world]
            m = self.to_object[inst[~world]]
            src_o, src_d = o_w[ro], d_w[ro]
            for i in range(3):  # written out: a fixed summation order on every platform
                o[ro, i] = (
                    m[:, i, 0] * src_o[:, 0]
                    + m[:, i, 1] * src_o[:, 1]
                    + m[:, i, 2] * src_o[:, 2]
                    + m[:, i, 3]
                )
                d[ro, i] = (
                    m[:, i, 0] * src_d[:, 0] + m[:, i, 1] * src_d[:, 1] + m[:, i, 2] * src_d[:, 2]
                )
        inv[rays], kx[rays], ky[rays], kz[rays], shear[rays] = _ray_setup(d[rays])
        cur[rays] = inst

    def stats(self) -> dict:
        return {**self.info, "traversal": self.traced.to_dict()}


def _ray_setup(d: np.ndarray):
    """Inverse directions (slab tests) and the watertight test's permutation and shear."""
    safe = np.where(np.abs(d) < MIN_DIRECTION, np.where(d < 0, -MIN_DIRECTION, MIN_DIRECTION), d)
    inv = 1.0 / safe
    kz = np.argmax(np.abs(d), axis=1)
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    rows = np.arange(d.shape[0])
    dz = d[rows, kz]
    swap = dz < 0.0  # keep the winding of the permuted coordinates
    kx, ky = np.where(swap, ky, kx), np.where(swap, kx, ky)
    dz_safe = np.where(dz == 0.0, 1.0, dz)
    shear = np.stack([d[rows, kx] / dz_safe, d[rows, ky] / dz_safe, 1.0 / dz_safe], axis=1)
    return inv, kx, ky, kz, shear


def _box(lo, hi, o, inv, tmin, tmax):
    """Slab test: entry distance (clamped to ``tmin``) and whether the box is hit in range.

    (Per-axis minimum/maximum written out: reductions over a length-3 axis are slow.)
    """
    t1 = (lo - o) * inv
    t2 = (hi - o) * inv
    a = np.minimum(t1, t2)
    b = np.maximum(t1, t2)
    near = np.maximum(np.maximum(a[..., 0], a[..., 1]), np.maximum(a[..., 2], tmin))
    far = np.minimum(np.minimum(b[..., 0], b[..., 1]), b[..., 2]) * (1.0 + 4e-16)
    return near, (near <= far) & (near < tmax)


def _triangles(v0, v1, v2, o, kx, ky, kz, shear):
    """Watertight ray-triangle test (Woop et al. 2013) for ray/triangle pairs.

    Returns ``t``, barycentrics ``(b0, b1, b2)`` of the corners and whether the
    pair intersects (``t`` not yet range-checked).
    """
    rows = np.arange(o.shape[0])
    a = v0 - o
    b = v1 - o
    c = v2 - o
    sx, sy, sz = shear[:, 0], shear[:, 1], shear[:, 2]
    az, bz, cz = a[rows, kz], b[rows, kz], c[rows, kz]
    ax = a[rows, kx] - sx * az
    ay = a[rows, ky] - sy * az
    bx = b[rows, kx] - sx * bz
    by = b[rows, ky] - sy * bz
    cx = c[rows, kx] - sx * cz
    cy = c[rows, ky] - sy * cz
    u = cx * by - cy * bx
    v = ax * cy - ay * cx
    w = bx * ay - by * ax
    det = u + v + w
    ok = ~(((u < 0.0) | (v < 0.0) | (w < 0.0)) & ((u > 0.0) | (v > 0.0) | (w > 0.0)))
    ok &= det != 0.0
    safe = np.where(det != 0.0, det, 1.0)
    t = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / safe
    bary = np.stack([u / safe, v / safe, w / safe], axis=1)
    return t, bary, ok
