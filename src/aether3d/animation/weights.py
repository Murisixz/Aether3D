"""Skin weights: automatic generation, numeric editing, transfer and inspection.

**Generation** (:func:`generate_weights`, ``scene.animation.generate_weights``):

* ``geodesic`` (the default) -- geodesic voxel binding (after Dionne and de Lasa, SCA
  2013): the mesh (all its parts, open or not) is voxelized into a solid; each deforming
  bone's segment is rasterized into it; each vertex's distance to each bone is the
  shortest path **inside the solid** (:func:`.voxels.geodesic`). Weights fall off with that
  distance (``w ~ d^-p``) over the bones within ``ratio`` times the nearest one, are
  smoothed along the mesh (never across disconnected parts) and capped to ``influences``.
  Because paths cannot cross air, an arm resting near the torso does not pull the torso,
  and the left leg never weighs on the right.
* ``envelope`` and ``nearest`` -- Stage 17's Euclidean distances to the bone segments.
* **Policies**: ``organic`` (``p = 2``, wide smooth blends), ``stylized`` (``p = 4``,
  tighter blends), ``rigid`` (each disconnected part follows the one bone nearest to it
  as a whole -- blocky and armoured characters -- else each vertex its nearest bone).
  ``rigid_parts`` (``auto``, ``all``, ``none``) makes small separate parts (buttons,
  armour plates, a helmet) rigid in the other policies too.

Unreached vertices (a part no bone runs through and no path reaches) fall back to the
nearest bone in straight line, as rigid parts; every fallback is counted in the report,
with the technique, its assumptions, the voxel size and the timings. The work is bounded
(``max_seconds``: ``CANCELLED`` when exceeded) and deterministic.

**Editing** (:func:`edit_weights`) works on the dense ``(vertices, joints)`` weights with
selections, numeric brushes (``falloff``: center or joint, radius, curve, Euclidean or
surface distance), locks (persistent in the skin, or per call) and renormalization that
only redistributes among unlocked joints.

**Transfer** (:func:`transfer_weights`) gives clothing and armour the weights of the body
under them: each vertex takes the barycentric blend of the closest point on the source
surface, with its distance and a confidence; parts farther than ``max_distance`` are
reported and, by default, bound rigidly to their nearest joint.

**Inspection** (:func:`inspect_weights`): validity, per-joint coverage, leakage (weights far
from their bone), cross-side influences, unused and helper joints with weights.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import voxels as V

if TYPE_CHECKING:
    from .system import AnimationSystem

METHODS = ("geodesic", "envelope", "nearest")
POLICIES = {
    "organic": {"power": 2.0, "ratio": 3.0, "smooth": 6, "influences": 4},
    "stylized": {"power": 4.0, "ratio": 2.0, "smooth": 3, "influences": 4},
    "rigid": {"power": 0.0, "ratio": 1.0, "smooth": 0, "influences": 1},
}
RIGID_PARTS = ("auto", "all", "none")
EDIT_OPS = (
    "assign",
    "add",
    "subtract",
    "scale",
    "replace",
    "normalize",
    "prune",
    "cap",
    "smooth",
    "mirror",
    "rigid",
    "lock",
    "unlock",
)
CURVES = ("constant", "linear", "smooth", "sharp", "root", "sphere")
MAX_DENSE = 80_000_000  # vertices x joints held densely while editing
PRUNE = 1e-3


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


def _number(value: Any, location: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _err("INVALID_ARGUMENT", f"expected a number, got {value!r}", location)
    v = float(value)
    if not math.isfinite(v) or not lo <= v <= hi:
        raise _err("INVALID_ARGUMENT", f"a number within {lo:g}..{hi:g}", location)
    return v


def _int(value: Any, location: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise _err("INVALID_ARGUMENT", f"an integer within {lo}..{hi}", location)
    return int(value)


class _Deadline:
    def __init__(self, seconds: float | None) -> None:
        self.end = None if seconds is None else time.monotonic() + float(seconds)

    def check(self, what: str) -> None:
        if self.end is not None and time.monotonic() > self.end:
            raise _err("CANCELLED", f"stopped after the time limit ({what})", "max_seconds")


# --------------------------------------------------------------------------------------- #
# Mesh helpers
# --------------------------------------------------------------------------------------- #


def mesh_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges ``(a, b)`` with ``a < b``."""
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    a = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    b = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def smooth_rows(
    w: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    iterations: int,
    strength: float = 0.5,
    pinned_rows: np.ndarray | None = None,
    pinned_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Laplacian smoothing of the rows of ``w`` over the edges ``a``-``b``: each row moves
    ``strength`` of the way to its neighbours' mean, ``iterations`` times; pinned rows
    and columns keep their values (the rest of a row is renormalized around them)."""
    n = w.shape[0]
    if iterations <= 0 or a.size == 0:
        return w
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    has = deg > 0
    out = w.copy()
    for _ in range(int(iterations)):
        acc = np.zeros_like(out)
        np.add.at(acc, a, out[b])
        np.add.at(acc, b, out[a])
        mean = np.where(has[:, None], acc / np.maximum(deg, 1)[:, None], out)
        new = out + strength * (mean - out)
        if pinned_cols is not None and pinned_cols.any():
            new[:, pinned_cols] = out[:, pinned_cols]
            new = _renormalize(new, pinned_cols)
        if pinned_rows is not None:
            new[pinned_rows] = out[pinned_rows]
        out = new
    return out


def _renormalize(w: np.ndarray, locked_cols: np.ndarray | None = None) -> np.ndarray:
    """Rows summing to 1, scaling only the unlocked columns."""
    w = np.clip(w, 0.0, None)
    if locked_cols is None or not locked_cols.any():
        total = w.sum(axis=1, keepdims=True)
        return np.where(total > 0, w / np.where(total > 0, total, 1.0), w)
    fixed = w[:, locked_cols].sum(axis=1)
    free = w[:, ~locked_cols]
    s = free.sum(axis=1)
    room = np.clip(1.0 - fixed, 0.0, None)
    scale = np.where(s > 0, room / np.where(s > 0, s, 1.0), 0.0)
    out = w.copy()
    out[:, ~locked_cols] = free * scale[:, None]
    return out


def nearest_points(
    points: np.ndarray, queries: np.ndarray, k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest ``points`` to each query: ``(indices (q, k), distances (q, k))``.
    A uniform grid of cells holding about four points each, searched in growing shells
    until ``k`` candidates are known to be the nearest (exact)."""
    p = np.asarray(points, np.float64)
    q = np.asarray(queries, np.float64)
    n = p.shape[0]
    k = min(k, n)
    if n == 0:
        raise _err("INVALID_ARGUMENT", "no points to search", "points")
    lo = np.minimum(p.min(axis=0), q.min(axis=0) if q.size else p.min(axis=0))
    hi = np.maximum(p.max(axis=0), q.max(axis=0) if q.size else p.max(axis=0))
    extent = np.maximum(hi - lo, 1e-9)
    cell = float(np.cbrt(np.prod(extent) / max(1.0, n / 4.0)))
    cell = max(cell, float(extent.max()) / 512.0)
    dims = np.maximum(1, np.ceil(extent / cell).astype(np.int64) + 1)
    pc = np.minimum(((p - lo) / cell).astype(np.int64), dims - 1)
    key = pc[:, 0] + dims[0] * (pc[:, 1] + dims[1] * pc[:, 2])
    order = np.argsort(key, kind="stable")
    skey = key[order]
    out_i = np.zeros((q.shape[0], k), np.int64)
    out_d = np.full((q.shape[0], k), np.inf)
    qc = np.minimum(np.maximum(((q - lo) / cell).astype(np.int64), 0), dims - 1)
    todo = np.arange(q.shape[0])
    ring = 1
    while todo.size:
        r = np.arange(-ring, ring + 1)
        oz, oy, ox = np.meshgrid(r, r, r, indexing="ij")
        offs = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)
        cand_q, cand_p = [], []
        for off in offs:
            c = qc[todo] + off
            ok = np.all((c >= 0) & (c < dims), axis=1)
            ck = c[:, 0] + dims[0] * (c[:, 1] + dims[1] * c[:, 2])
            start = np.searchsorted(skey, ck, side="left")
            stop = np.searchsorted(skey, ck, side="right")
            count = np.where(ok, stop - start, 0)
            if not count.any():
                continue
            rows = np.repeat(np.arange(todo.size), count)
            first = np.repeat(start, count)
            within = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
            cand_q.append(rows)
            cand_p.append(order[first + within])
        if cand_q:
            cq = np.concatenate(cand_q)
            cp = np.concatenate(cand_p)
            d = np.linalg.norm(p[cp] - q[todo][cq], axis=1)
            srt = np.lexsort((cp, d, cq))
            cq, cp, d = cq[srt], cp[srt], d[srt]
            first_of = np.searchsorted(cq, np.arange(todo.size))
            rank = np.arange(cq.size) - first_of[cq]
            keep = rank < k
            got = np.bincount(cq[keep], minlength=todo.size)
            out_i[todo[cq[keep]], rank[keep]] = cp[keep]
            out_d[todo[cq[keep]], rank[keep]] = d[keep]
            # Exact once the k-th distance is within the searched shell's guaranteed reach.
            kth = out_d[todo, k - 1]
            done = (got >= k) & (kth <= ring * cell)
        else:
            done = np.zeros(todo.size, bool)
        if ring > int(dims.max()) + 1:
            done[:] = True
        todo = todo[~done]
        ring *= 2
    return out_i, out_d


def surface_distances(
    positions: np.ndarray, a: np.ndarray, b: np.ndarray, seeds: np.ndarray, radius: float
) -> np.ndarray:
    """Shortest distances along mesh edges from ``seeds`` (``inf`` beyond ``radius``)."""
    n = positions.shape[0]
    length = np.linalg.norm(positions[a] - positions[b], axis=1)
    order = np.argsort(np.concatenate([a, b]), kind="stable")
    src = np.concatenate([a, b])[order]
    dst = np.concatenate([b, a])[order]
    wl = np.concatenate([length, length])[order]
    ptr = np.searchsorted(src, np.arange(n + 1))
    dist = np.full(n, np.inf)
    heap = [(0.0, int(s)) for s in np.unique(seeds).tolist()]
    for _, s in heap:
        dist[s] = 0.0
    heapq.heapify(heap)
    dl = dist.tolist()
    dstl, wll, ptrl = dst.tolist(), wl.tolist(), ptr.tolist()
    while heap:
        d, v = heapq.heappop(heap)
        if d > dl[v]:
            continue
        for e in range(ptrl[v], ptrl[v + 1]):
            nd = d + wll[e]
            u = dstl[e]
            if nd < dl[u] and nd <= radius:
                dl[u] = nd
                heapq.heappush(heap, (nd, u))
    return np.array(dl)


# --------------------------------------------------------------------------------------- #
# Context: the mesh and the rig in world space at rest
# --------------------------------------------------------------------------------------- #


class _Context:
    def __init__(self, system: AnimationSystem, mesh_object: Any, armature: Any, joints=None):
        from ..rendering.scene_data import world_matrices

        self.system = system
        self.obj = system._object(mesh_object, "mesh_object")
        self.arm = system._object(armature, "armature")
        self.rig = system.rig(self.arm)
        if self.rig is None:
            raise _err("RIG_MISSING", "the armature has no rig", "armature")
        mesh = self.obj.mesh
        if mesh is None or mesh.vertex_count == 0:
            raise _err("INVALID_SKIN", f"{self.obj.name!r} has no mesh to bind", "mesh_object")
        self.mesh = mesh
        worlds = world_matrices(system.scene)
        self.m_world = worlds[self.obj._id]
        self.a_world = worlds[self.arm._id]
        pos = mesh.positions.astype(np.float64)
        self.world = pos @ self.m_world[:3, :3].T + self.m_world[:3, 3]
        rig = self.rig
        if joints is None:
            idx = list(rig.deforming) or list(range(len(rig.joints)))
        else:
            idx = []
            for i, n in enumerate(joints):
                k = rig.index(n)
                if k is None:
                    raise _err("INVALID_SKIN", f"no joint {n!r}", f"joints[{i}]")
                idx.append(k)
        self.joints = idx
        model = rig.rest_model()
        tails = rig.tails()
        heads, ends = [], []
        for k in idx:
            m = self.a_world @ model[k]
            heads.append(m[:3, 3])
            ends.append((m @ np.append(tails[k], 1.0))[:3])
        self.heads = np.array(heads)
        self.ends = np.array(ends)

    def euclidean(self, points: np.ndarray | None = None) -> np.ndarray:
        """Distances from points (default: the vertices) to every bone segment."""
        p = self.world if points is None else points
        a, b = self.heads, self.ends
        seg = b - a
        l2 = np.maximum(np.sum(seg * seg, axis=1), 1e-30)
        out = np.empty((p.shape[0], a.shape[0]))
        for start in range(0, p.shape[0], 32768):
            q = p[start : start + 32768]
            rel = q[:, None, :] - a[None]
            t = np.clip(np.sum(rel * seg[None], axis=2) / l2[None], 0.0, 1.0)
            out[start : start + q.shape[0]] = np.linalg.norm(rel - t[..., None] * seg[None], axis=2)
        return out


# --------------------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------------------- #


def generate_weights(
    system: AnimationSystem,
    mesh_object: Any,
    armature: Any,
    *,
    method: str = "geodesic",
    policy: str = "organic",
    influences: int | None = None,
    joints: list[str] | None = None,
    resolution: int = 128,
    voxel_size: float | None = None,
    smooth: int | None = None,
    hops: int = 1,
    sides: str = "auto",
    rigid_parts: str = "auto",
    skinning: str | None = None,
    dq: Any = None,
    max_seconds: float | None = None,
    bind: bool = True,
) -> dict[str, Any]:
    """Compute skin weights for ``mesh_object`` from ``armature``'s rig and (``bind``)
    bind it -- one undo step. See the module docstring; returns a report."""
    if method not in METHODS:
        raise _err("INVALID_ARGUMENT", f"method is one of {', '.join(METHODS)}", "method")
    if policy not in POLICIES:
        raise _err("INVALID_ARGUMENT", f"policy is one of {', '.join(POLICIES)}", "policy")
    if rigid_parts not in RIGID_PARTS:
        raise _err("INVALID_ARGUMENT", f"rigid_parts is one of {', '.join(RIGID_PARTS)}",
                   "rigid_parts")  # fmt: skip
    pol = dict(POLICIES[policy])
    k = pol["influences"] if influences is None else _int(influences, "influences", 1, 8)
    iterations = pol["smooth"] if smooth is None else _int(smooth, "smooth", 0, 200)
    resolution = _int(resolution, "resolution", 16, 512)
    hops = _int(hops, "hops", 0, 64)
    if sides not in ("auto", "off"):
        raise _err("INVALID_ARGUMENT", "sides is 'auto' or 'off'", "sides")
    deadline = _Deadline(
        None if max_seconds is None else _number(max_seconds, "max_seconds", 0.0, 86400.0)
    )
    t0 = time.perf_counter()
    ctx = _Context(system, mesh_object, armature, joints)
    n = ctx.mesh.vertex_count
    m = len(ctx.joints)
    timings: dict[str, float] = {}
    report: dict[str, Any] = {
        "mesh": ctx.obj.name,
        "armature": ctx.arm.name,
        "method": method,
        "policy": policy,
        "vertices": n,
        "joints": m,
    }
    a_edges, b_edges = mesh_edges(ctx.mesh.triangles)
    comp, n_comp = V.mesh_components(n, ctx.mesh.triangles)
    report["parts"] = n_comp
    if method == "geodesic":
        dist, info = _geodesic_distances(ctx, resolution, voxel_size, deadline, timings)
        report.update(info)
        unreached = ~np.isfinite(dist).any(axis=1)
    else:
        dist = ctx.euclidean()
        unreached = np.zeros(n, bool)
        report["technique"] = "euclidean distance to bone segments (Stage 17 envelope)"
    deadline.check("distances")
    t1 = time.perf_counter()
    eu = None
    if unreached.any():
        eu = ctx.euclidean(ctx.world[unreached])
    dense = np.zeros((n, m))
    if method == "nearest" or policy == "rigid":
        nearest = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        dense[np.arange(n), nearest] = 1.0
    else:
        power = pol["power"]
        d = np.where(np.isfinite(dist), dist, np.inf)
        dmin = d.min(axis=1, keepdims=True)
        span = np.maximum(dmin, 1e-6) * pol["ratio"] + 1e-9
        floor = 0.25 * _bone_floor(ctx)
        w = np.where(d <= span, 1.0 / np.maximum(d + floor, 1e-9) ** power, 0.0)
        dense = w
    # Unreached vertices: whole parts follow their nearest bone (straight line).
    rigid_info = {"unreached_vertices": int(unreached.sum()), "rigid_parts": 0}
    if eu is not None:
        dense[unreached] = 0.0
        rows = np.flatnonzero(unreached)
        for c in np.unique(comp[rows]).tolist():
            members = rows[comp[rows] == c]
            j = int(np.argmin(eu[np.isin(rows, members)].min(axis=0)))
            dense[members, j] = 1.0
            rigid_info["rigid_parts"] += 1
    # Separate parts that belong to one bone move rigidly: small parts (buttons, plates,
    # a helmet) with rigid_parts="auto", every part with "all", and with the rigid policy
    # every part most of whose vertices are nearest one bone (blocky characters).
    rigid_rows = unreached.copy()
    if n_comp > 1 or rigid_parts == "all":
        sizes = np.bincount(comp, minlength=n_comp)
        for c in range(n_comp):
            members = np.flatnonzero(comp == c)
            if unreached[members].all():
                continue  # (already bound to its nearest bone)
            share = _normalize_rows(dense[members]).sum(axis=0) / len(members)
            j = int(np.argmax(share))
            if rigid_parts == "all":
                make = True
            elif policy == "rigid":
                make = share[j] >= 0.6 and rigid_parts != "none"
            else:
                make = rigid_parts == "auto" and sizes[c] <= 0.05 * n
            if make:
                dense[members] = 0.0
                dense[members, j] = 1.0
                rigid_rows[members] = True
                rigid_info["rigid_parts"] += 1
    report.update(rigid_info)
    dense = _normalize_rows(dense)
    # Only the nearest bone and its neighbours in the skeleton (within ``hops``) may
    # weigh on a vertex: the other thigh never reaches across the pelvis.
    allowed = None
    if hops and policy != "rigid":
        allowed = _neighbourhood(ctx, np.argmax(dense, axis=1), hops)
    if sides == "auto" and policy != "rigid":
        side_ok = _sides(ctx)
        if side_ok is not None:
            allowed = side_ok if allowed is None else allowed & side_ok
            report["sides"] = "left and right joints keep to their side of the mirror plane"
    if allowed is not None:
        keep = np.where(allowed, dense, 0.0)
        lost = keep.sum(axis=1) <= 0  # (a vertex only the other side reached keeps all)
        dense = _normalize_rows(np.where(lost[:, None], dense, keep))
        allowed[lost] = True
    # Smooth along the mesh (rigid parts pinned), then keep the strongest influences.
    if iterations and policy != "rigid":
        dense = _cap(dense, k + 2)
        dense = smooth_rows(dense, a_edges, b_edges, iterations, 0.5, rigid_rows)
        if allowed is not None:
            dense = _normalize_rows(np.where(allowed, dense, 0.0))
    dense = _cap(dense, k)
    deadline.check("weights")
    timings["weights_s"] = round(time.perf_counter() - t1, 4)
    report["influences"] = k
    report["smoothing_iterations"] = iterations if policy != "rigid" else 0
    report["joints_used"] = int((dense > 0).any(axis=0).sum())
    report["timings"] = {**timings, "total_s": round(time.perf_counter() - t0, 4)}
    report["assumptions"] = [
        "the bind pose is the current rest pose",
        "bones run inside the parts they move (bones outside the solid use the nearest "
        "solid voxels)",
        "voxels thinner than the mesh's features merge nearby surfaces",
    ]
    if bind:
        names = [ctx.rig.joints[j].name for j in ctx.joints]
        idx, w = _sparse(dense, k)
        old = system._owned.get(("skin", ctx.obj._id))
        method_skin = skinning or (
            old._definition.method if old is not None and old._alive else "linear"
        )
        if method_skin == "blended" and dq is None and old is not None:
            dq = old._definition.dq
        report["skin"] = system.skin(
            ctx.obj, ctx.arm, weights={"indices": idx, "weights": w, "joints": names},
            influences=k, method=method_skin, dq=dq,
        )  # fmt: skip
    else:
        report["dense"] = dense
        report["joint_names"] = [ctx.rig.joints[j].name for j in ctx.joints]
    return report


def _neighbourhood(ctx: _Context, nearest: np.ndarray, hops: int) -> np.ndarray:
    """``(n, m)``: which of the weighted joints may weigh on each vertex -- its nearest
    joint and those within ``hops`` steps of it in the skeleton (steps through joints that
    are not weighted do not count)."""
    rig = ctx.rig
    m = len(ctx.joints)
    col = {k: i for i, k in enumerate(ctx.joints)}
    adj: list[set[int]] = [set() for _ in range(m)]
    for i, k in enumerate(ctx.joints):
        p = rig.joints[k].parent
        while p >= 0 and p not in col:  # the nearest weighted ancestor
            p = rig.joints[p].parent
        if p >= 0:
            adj[i].add(col[p])
            adj[col[p]].add(i)
    reach = np.eye(m, dtype=bool)
    frontier = reach.copy()
    for _ in range(hops):
        nxt = np.zeros_like(reach)
        for i in range(m):
            for j in adj[i]:
                nxt[:, j] |= frontier[:, i]
        frontier = nxt & ~reach
        reach |= nxt
    return reach[nearest]


SIDE_MARGIN = 0.02  # of the mesh's size: the band around the mirror plane both sides share


def _side_of(rig: Any, k: int) -> str | None:
    role = rig.joints[k].role
    if role and role[-2:] in (".L", ".R"):
        return role[-1]
    name = rig.joints[k].name
    other = rig.mirror_name(name)
    if other is None or other == name:
        return None
    for left, right in rig.mirror:
        if name.endswith(left) or name.startswith(left):
            return "L"
        if name.endswith(right) or name.startswith(right):
            return "R"
    return None


def _sides(ctx: _Context) -> np.ndarray | None:
    """``(n, m)``: left joints may not weigh on vertices clearly right of the armature's
    mirror plane (x < -margin, armature space), nor right joints left of it -- applied when
    every sided bone lies on its own side at rest (a symmetric rest pose)."""
    rig = ctx.rig
    inv = np.linalg.inv(ctx.a_world)
    x = (ctx.world @ inv[:3, :3].T + inv[:3, 3])[:, 0]
    heads = (ctx.heads @ inv[:3, :3].T + inv[:3, 3])[:, 0]
    sides = [_side_of(rig, k) for k in ctx.joints]
    if not any(sides):
        return None
    for s_, hx in zip(sides, heads, strict=True):
        if (s_ == "L" and hx < 0) or (s_ == "R" and hx > 0):
            return None  # (an asymmetric rest pose: no side rule)
    size = float(np.max(ctx.world.max(axis=0) - ctx.world.min(axis=0)))
    margin = SIDE_MARGIN * size
    out = np.ones((x.size, len(sides)), bool)
    for i, s_ in enumerate(sides):
        if s_ == "L":
            out[:, i] = x >= -margin
        elif s_ == "R":
            out[:, i] = x <= margin
    return out


def _bone_floor(ctx: _Context) -> float:
    lengths = np.linalg.norm(ctx.ends - ctx.heads, axis=1)
    return float(np.median(lengths)) * 0.05 + 1e-6


def _normalize_rows(w: np.ndarray) -> np.ndarray:
    total = w.sum(axis=1, keepdims=True)
    return np.where(total > 0, w / np.where(total > 0, total, 1.0), w)


def _cap(dense: np.ndarray, k: int) -> np.ndarray:
    """Keep each row's ``k`` largest weights (ties: lower column) and those above the
    prune threshold, renormalized."""
    if dense.shape[1] > k:
        order = np.argsort(-dense, axis=1, kind="stable")
        drop = order[:, k:]
        dense = dense.copy()
        np.put_along_axis(dense, drop, 0.0, axis=1)
    dense = np.where(dense >= PRUNE, dense, 0.0)
    empty = dense.sum(axis=1) <= 0
    return _normalize_rows(dense) if not empty.any() else _fix_empty(dense, empty)


def _fix_empty(dense: np.ndarray, empty: np.ndarray) -> np.ndarray:
    raise _err("INVALID_SKIN", f"{int(empty.sum())} vertices lost every weight", "weights")


def _sparse(dense: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, dense.shape[1])
    order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(dense, order, axis=1)
    return order.astype(np.int32), _normalize_rows(w)


def _geodesic_distances(ctx, resolution, voxel_size, deadline, timings):
    t0 = time.perf_counter()
    meshes = [(ctx.world, ctx.mesh.triangles)]
    grid = V.voxelize(meshes[0][0], meshes[0][1], resolution=resolution, h=voxel_size)
    if grid.stats["shell_only"]:  # an open mesh: close small holes and try again
        grid = V.voxelize(meshes[0][0], meshes[0][1], resolution=resolution, h=voxel_size, close=2)
    timings["voxelize_s"] = round(time.perf_counter() - t0, 4)
    deadline.check("voxelization")
    t1 = time.perf_counter()
    sources = []
    outside = []
    for i in range(len(ctx.joints)):
        src = V.segment_voxels(grid, ctx.heads[i], ctx.ends[i])
        if src.size == 0:
            outside.append(ctx.rig.joints[ctx.joints[i]].name)
        sources.append(src)
    vox = grid.flat(grid.voxel_of(ctx.world))
    uniq, inverse = np.unique(vox, return_inverse=True)
    extent = float(np.max(ctx.world.max(axis=0) - ctx.world.min(axis=0)))
    cutoff = int(3 * math.ceil(0.5 * extent / grid.h))
    out = np.empty((uniq.size, len(sources)), np.int32)
    batch = 8
    for start in range(0, len(sources), batch):
        out[:, start : start + batch] = V.geodesic(
            grid, sources[start : start + batch], uniq, cutoff=cutoff
        )
        deadline.check("geodesic distances")
    timings["geodesic_s"] = round(time.perf_counter() - t1, 4)
    d = out[inverse].astype(np.float64)
    dist = np.where(d >= 0, d * grid.h / 3.0, np.inf)
    info = {
        "technique": "geodesic voxel binding: chamfer (3-4-5) shortest paths inside the "
        "voxelized solid, inverse-distance falloff, mesh smoothing",
        "voxel_size": grid.h,
        "grid": grid.stats["dims"],
        "shell_only": bool(grid.stats["shell_only"]),
        "closing": grid.stats["closing"],
        "bones_outside_mesh": outside,
    }
    return dist, info


# --------------------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------------------- #


class _Edit:
    """A skin's weights opened for editing: dense ``(n, m)`` over the skin's palette."""

    def __init__(self, system: AnimationSystem, mesh_object: Any) -> None:
        from ..rendering.scene_data import world_matrices

        self.system = system
        self.obj = system._object(mesh_object, "mesh_object")
        item = system._owned.get(("skin", self.obj._id))
        if item is None or not item._alive:
            raise _err(
                "SKIN_MISSING", f"{self.obj.name!r} has no skin (bind it first)", "mesh_object"
            )
        self.item = item
        self.skin = item._definition
        rig_item = system._items[self.skin.rig]
        self.rig = rig_item._definition
        self.arm = system.scene._objects[self.rig.object]
        n, m = self.skin.vertex_count, len(self.skin.joints)
        if n * m > MAX_DENSE:
            raise _err("LIMIT_EXCEEDED", f"{n} x {m} weights are too many to edit at once",
                       "mesh_object")  # fmt: skip
        self.w = self.skin.dense()
        self.palette = list(self.skin.joints)  # joint ids
        self.names = [self.rig.joints[self.rig.index_of_id(j)].name for j in self.palette]
        self.locked_cols = np.array([j in self.skin.locked_joints for j in self.palette], bool)
        self.locked_rows = np.zeros(n, bool)
        if self.skin.locked_vertices is not None:
            self.locked_rows[self.skin.locked_vertices] = True
        worlds = world_matrices(system.scene)
        mesh = self.obj.mesh
        pos = mesh.positions.astype(np.float64)
        mw = worlds[self.obj._id]
        self.world = pos @ mw[:3, :3].T + mw[:3, 3]
        self.mesh = mesh
        self.a_world = worlds[self.arm._id]
        self.k = int(self.skin.indices.shape[1])
        self._edges: tuple[np.ndarray, np.ndarray] | None = None
        self._comp: np.ndarray | None = None
        self.notes: list[dict[str, Any]] = []

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        if self._edges is None:
            self._edges = mesh_edges(self.mesh.triangles)
        return self._edges

    @property
    def comp(self) -> np.ndarray:
        if self._comp is None:
            self._comp, _ = V.mesh_components(self.mesh.vertex_count, self.mesh.triangles)
        return self._comp

    def col(self, name: Any, location: str, *, add: bool = False) -> int:
        if not isinstance(name, str):
            raise _err("INVALID_ARGUMENT", "a joint name", location)
        if name in self.names:
            return self.names.index(name)
        k = self.rig.index(name)
        if k is None:
            raise _err("JOINT_MISSING", f"no joint {name!r} in the rig", location)
        if not add:
            raise _err("JOINT_MISSING", f"joint {name!r} has no weights in this skin", location)
        # A joint new to the skin: its inverse bind from the current rest.
        from ..rendering.scene_data import world_matrices

        worlds = world_matrices(self.system.scene)
        model = self.rig.rest_model()[k]
        ib = np.linalg.inv(model) @ np.linalg.inv(worlds[self.arm._id]) @ worlds[self.obj._id]
        self.palette.append(self.rig.joints[k].id)
        self.names.append(name)
        self.w = np.concatenate([self.w, np.zeros((self.w.shape[0], 1))], axis=1)
        self.locked_cols = np.append(self.locked_cols, False)
        self.new_binds = getattr(self, "new_binds", {})
        self.new_binds[self.rig.joints[k].id] = ib
        return len(self.names) - 1

    # -- selections ---------------------------------------------------------------------------

    def select(self, raw: Any, location: str) -> np.ndarray:
        n = self.w.shape[0]
        if raw is None:
            return np.ones(n, bool)
        if not isinstance(raw, dict) or len(raw) == 0:
            raise _err("INVALID_ARGUMENT", "a selection is an object", location)
        if "and" in raw or "or" in raw:
            key = "and" if "and" in raw else "or"
            parts = raw[key]
            if set(raw) != {key} or not isinstance(parts, list) or not parts:
                raise _err("INVALID_ARGUMENT", f"{key} is a list of selections", location)
            masks = [self.select(s, f"{location}.{key}[{i}]") for i, s in enumerate(parts)]
            out = masks[0].copy()
            for mk in masks[1:]:
                out = (out & mk) if key == "and" else (out | mk)
            return out
        if "not" in raw:
            if set(raw) != {"not"}:
                raise _err("INVALID_ARGUMENT", "not takes one selection", location)
            return ~self.select(raw["not"], f"{location}.not")
        if "all" in raw:
            return np.ones(n, bool)
        if "vertices" in raw:
            v = np.asarray(raw["vertices"], np.int64).reshape(-1)
            if v.size and (v.min() < 0 or v.max() >= n):
                raise _err("INVALID_ARGUMENT", f"vertices are 0..{n - 1}", f"{location}.vertices")
            out = np.zeros(n, bool)
            out[v] = True
            return out
        pts = self.world if raw.get("space", "world") == "world" else self.mesh.positions
        if "sphere" in raw:
            s = raw["sphere"]
            c = _vec3(s.get("center"), f"{location}.sphere.center")
            r = _number(s.get("radius"), f"{location}.sphere.radius", 0.0)
            return np.linalg.norm(pts - c, axis=1) <= r
        if "box" in raw:
            lo = _vec3(raw["box"].get("min"), f"{location}.box.min")
            hi = _vec3(raw["box"].get("max"), f"{location}.box.max")
            return np.all((pts >= lo) & (pts <= hi), axis=1)
        if "plane" in raw:
            pnt = _vec3(raw["plane"].get("point"), f"{location}.plane.point")
            nrm = _vec3(raw["plane"].get("normal"), f"{location}.plane.normal")
            return (pts - pnt) @ nrm >= 0.0
        if "joint" in raw:
            c = self.col(raw["joint"], f"{location}.joint")
            lo = _number(raw.get("min", 1e-9), f"{location}.min", 0.0, 1.0)
            return self.w[:, c] >= lo
        if "dominant" in raw:
            c = self.col(raw["dominant"], f"{location}.dominant")
            return np.argmax(self.w, axis=1) == c
        if "role" in raw:
            k = self.rig.role_index(raw["role"])
            if k is None:
                raise _err("JOINT_MISSING", f"no joint has the role {raw['role']!r}",
                           f"{location}.role")  # fmt: skip
            ids = {self.rig.joints[k].id}
            if raw.get("descendants", False):
                ids |= {self.rig.joints[d].id for d in self.rig.descendants(k)}
            cols = [i for i, j in enumerate(self.palette) if j in ids]
            dom = np.argmax(self.w, axis=1)
            return np.isin(dom, cols)
        if "component" in raw:
            c = _int(raw["component"], f"{location}.component", 0, 10**9)
            return self.comp == c
        if "component_of" in raw:
            v = _int(raw["component_of"], f"{location}.component_of", 0, n - 1)
            return self.comp == self.comp[v]
        if "near_joint" in raw:
            k = self.rig.index(raw["near_joint"])
            if k is None:
                raise _err("JOINT_MISSING", f"no joint {raw['near_joint']!r}",
                           f"{location}.near_joint")  # fmt: skip
            c = (self.a_world @ self.rig.rest_model()[k])[:3, 3]
            r = _number(raw.get("radius"), f"{location}.radius", 0.0)
            return self._distance(c, raw.get("distance", "euclidean"), r, location) <= r
        if "locked" in raw:
            return self.locked_rows.copy() if raw["locked"] else ~self.locked_rows
        raise _err(
            "INVALID_ARGUMENT",
            "a selection is all, vertices, sphere, box, plane, joint, dominant, role, "
            "component, component_of, near_joint, locked, and, or, not",
            location,
        )

    def _distance(self, center: np.ndarray, kind: str, radius: float, location: str):
        if kind == "euclidean":
            return np.linalg.norm(self.world - center, axis=1)
        if kind == "geodesic":
            seed = int(np.argmin(np.linalg.norm(self.world - center, axis=1)))
            a, b = self.edges
            return surface_distances(self.world, a, b, np.array([seed]), radius)
        raise _err("INVALID_ARGUMENT", "distance is 'euclidean' or 'geodesic'", location)

    def falloff(self, raw: Any, location: str) -> np.ndarray:
        n = self.w.shape[0]
        if raw is None:
            return np.ones(n)
        if not isinstance(raw, dict):
            raise _err("INVALID_ARGUMENT", "falloff is {center | joint, radius, curve, distance}",
                       location)  # fmt: skip
        if "joint" in raw:
            k = self.rig.index(raw["joint"])
            if k is None:
                raise _err("JOINT_MISSING", f"no joint {raw['joint']!r}", f"{location}.joint")
            center = (self.a_world @ self.rig.rest_model()[k])[:3, 3]
        else:
            center = _vec3(raw.get("center"), f"{location}.center")
        r = _number(raw.get("radius"), f"{location}.radius", 1e-9)
        curve = raw.get("curve", "smooth")
        if curve not in CURVES:
            raise _err("INVALID_ARGUMENT", f"curve is one of {', '.join(CURVES)}",
                       f"{location}.curve")  # fmt: skip
        d = self._distance(center, raw.get("distance", "euclidean"), r, location)
        u = np.clip(d / r, 0.0, 1.0)
        f = {
            "constant": np.where(d <= r, 1.0, 0.0),
            "linear": 1.0 - u,
            "smooth": 1.0 - u * u * (3.0 - 2.0 * u),
            "sharp": (1.0 - u) ** 2,
            "root": np.sqrt(1.0 - u),
            "sphere": np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)),
        }[curve]
        return np.where(np.isfinite(d), f, 0.0)

    # -- operations ---------------------------------------------------------------------------

    def run(self, raw: dict, location: str) -> None:
        op = raw.get("op")
        if op not in EDIT_OPS:
            raise _err("INVALID_ARGUMENT", f"op is one of {', '.join(EDIT_OPS)}", f"{location}.op")
        allowed = {
            "assign": {"joint", "value", "select", "falloff"},
            "add": {"joint", "value", "select", "falloff"},
            "subtract": {"joint", "value", "select", "falloff"},
            "scale": {"joint", "factor", "select", "falloff"},
            "replace": {"from", "to", "select", "falloff"},
            "normalize": {"select"},
            "prune": {"threshold", "select"},
            "cap": {"influences", "select"},
            "smooth": {"iterations", "strength", "select", "joints", "boundaries"},
            "mirror": {"from", "tolerance", "select"},
            "rigid": {"joint", "select", "per"},
            "lock": {"joints", "select"},
            "unlock": {"joints", "select"},
        }[op]
        unknown = set(raw) - allowed - {"op"}
        if unknown:
            raise _err("INVALID_ARGUMENT", f"unknown arguments {sorted(unknown)} of {op}",
                       location, hint=f"arguments: {', '.join(sorted(allowed))}")  # fmt: skip
        sel = self.select(raw.get("select"), f"{location}.select")
        getattr(self, f"_op_{op}")(raw, sel, location)

    def _editable(self, sel: np.ndarray) -> np.ndarray:
        return sel & ~self.locked_rows

    def _set_joint(self, c: int, target: np.ndarray, rows: np.ndarray) -> int:
        """Set column ``c`` on ``rows`` to ``target`` and rescale the other unlocked joints
        to fill the rest; returns how many rows had no other joint to give weight to."""
        w = self.w
        locked = self.locked_cols.copy()
        locked[c] = False
        fixed = w[rows][:, locked].sum(axis=1) if locked.any() else np.zeros(rows.sum())
        target = np.clip(target, 0.0, np.clip(1.0 - fixed, 0.0, 1.0))
        others = ~locked
        others[c] = False
        sub = w[rows]
        rest = sub[:, others].sum(axis=1)
        room = np.clip(1.0 - fixed - target, 0.0, None)
        stuck = (rest <= 0) & (room > 1e-12)
        target = np.where(stuck, 1.0 - fixed, target)  # nobody to receive: keep the sum
        scale = np.where(rest > 0, room / np.where(rest > 0, rest, 1.0), 0.0)
        sub[:, others] = sub[:, others] * scale[:, None]
        sub[:, c] = target
        w[rows] = sub
        return int(stuck.sum())

    def _value_op(self, raw, sel, location, fn) -> None:
        c = self.col(raw.get("joint"), f"{location}.joint", add=True)
        if self.locked_cols[c]:
            raise _err("JOINT_LOCKED", f"joint {self.names[c]!r} is locked", f"{location}.joint")
        f = self.falloff(raw.get("falloff"), f"{location}.falloff")
        rows = self._editable(sel) & (f > 0)
        if not rows.any():
            return
        current = self.w[rows, c]
        target = fn(current, f[rows])
        stuck = self._set_joint(c, target, rows)
        self.notes.append({"op": raw["op"], "joint": self.names[c], "vertices": int(rows.sum()),
                           **({"no_recipient": stuck} if stuck else {})})  # fmt: skip

    def _op_assign(self, raw, sel, loc):
        v = _number(raw.get("value", 1.0), f"{loc}.value", 0.0, 1.0)
        self._value_op(raw, sel, loc, lambda cur, f: cur + f * (v - cur))

    def _op_add(self, raw, sel, loc):
        v = _number(raw.get("value", 0.1), f"{loc}.value", 0.0, 1.0)
        self._value_op(raw, sel, loc, lambda cur, f: np.clip(cur + f * v, 0.0, 1.0))

    def _op_subtract(self, raw, sel, loc):
        v = _number(raw.get("value", 0.1), f"{loc}.value", 0.0, 1.0)
        self._value_op(raw, sel, loc, lambda cur, f: np.clip(cur - f * v, 0.0, 1.0))

    def _op_scale(self, raw, sel, loc):
        v = _number(raw.get("factor", 1.0), f"{loc}.factor", 0.0, 100.0)
        self._value_op(raw, sel, loc, lambda cur, f: np.clip(cur * (1 + f * (v - 1)), 0.0, 1.0))

    def _op_replace(self, raw, sel, loc):
        a = self.col(raw.get("from"), f"{loc}.from")
        b = self.col(raw.get("to"), f"{loc}.to", add=True)
        if self.locked_cols[a] or self.locked_cols[b]:
            raise _err("JOINT_LOCKED", "a locked joint's weights do not move", loc)
        f = self.falloff(raw.get("falloff"), f"{loc}.falloff")
        rows = self._editable(sel)
        moved = self.w[rows, a] * f[rows]
        self.w[rows, b] += moved
        self.w[rows, a] -= moved
        self.notes.append({"op": "replace", "from": self.names[a], "to": self.names[b],
                           "weight_moved": round(float(moved.sum()), 6)})  # fmt: skip

    def _op_normalize(self, raw, sel, loc):
        rows = self._editable(sel)
        self.w[rows] = _renormalize(self.w[rows], self.locked_cols)

    def _op_prune(self, raw, sel, loc):
        t = _number(raw.get("threshold", 0.01), f"{loc}.threshold", 0.0, 0.5)
        rows = self._editable(sel)
        sub = self.w[rows]
        small = (sub < t) & ~self.locked_cols[None, :]
        removed = int((small & (sub > 0)).sum())
        sub[small] = 0.0
        self.w[rows] = _renormalize(sub, self.locked_cols)
        self.notes.append({"op": "prune", "removed_influences": removed})

    def _op_cap(self, raw, sel, loc):
        k = _int(raw.get("influences", 4), f"{loc}.influences", 1, 8)
        rows = self._editable(sel)
        self.w[rows] = _cap_keep_locked(self.w[rows], k, self.locked_cols)
        self.k = min(self.k, k) if sel.all() else self.k

    def _op_smooth(self, raw, sel, loc):
        it = _int(raw.get("iterations", 3), f"{loc}.iterations", 1, 500)
        strength = _number(raw.get("strength", 0.5), f"{loc}.strength", 0.0, 1.0)
        rows = self._editable(sel)
        pinned = ~rows
        if raw.get("boundaries") is not None:
            pinned |= self.select(raw["boundaries"], f"{loc}.boundaries")
        cols = self.locked_cols.copy()
        if raw.get("joints") is not None:
            names = raw["joints"]
            if not isinstance(names, list):
                raise _err("INVALID_ARGUMENT", "joints is a list of names", f"{loc}.joints")
            keep = {self.col(n, f"{loc}.joints[{i}]") for i, n in enumerate(names)}
            cols = cols | np.array([i not in keep for i in range(len(self.names))])
        a, b = self.edges
        self.w = smooth_rows(self.w, a, b, it, strength, pinned, cols)

    def _op_mirror(self, raw, sel, loc):
        side = raw.get("from", "+x")
        if side not in ("+x", "-x"):
            raise _err("INVALID_ARGUMENT", "from is '+x' (the character's left) or '-x'",
                       f"{loc}.from")  # fmt: skip
        tol = _number(raw.get("tolerance", 0.002), f"{loc}.tolerance", 0.0, 1.0)
        # Armature space: the rig's mirror plane is x = 0.
        inv = np.linalg.inv(self.a_world)
        p = self.world @ inv[:3, :3].T + inv[:3, 3]
        src = p[:, 0] > tol if side == "+x" else p[:, 0] < -tol
        dst = (p[:, 0] < -tol if side == "+x" else p[:, 0] > tol) & self._editable(sel)
        if not src.any() or not dst.any():
            self.notes.append({"op": "mirror", "matched": 0})
            return
        src_idx = np.flatnonzero(src)
        dst_idx = np.flatnonzero(dst)
        reflected = p[dst_idx] * np.array([-1.0, 1.0, 1.0])
        near, dist = nearest_points(p[src_idx], reflected, 1)
        ok = dist[:, 0] <= max(tol, 1e-9) * 5 + 1e-6
        mapping = []
        for c, name in enumerate(self.names):
            other = self.rig.mirror_name(name) or name
            mapping.append(self.col(other, loc, add=True) if other != name else c)
        new = np.zeros((ok.sum(), len(self.names)))
        srcw = self.w[src_idx[near[ok, 0]]]
        for c in range(srcw.shape[1]):
            new[:, mapping[c]] += srcw[:, c]
        self.w[dst_idx[ok]] = new
        self.notes.append({"op": "mirror", "matched": int(ok.sum()),
                           "unmatched": int((~ok).sum())})  # fmt: skip

    def _op_rigid(self, raw, sel, loc):
        rows = self._editable(sel)
        per = raw.get("per", "selection")
        if per not in ("selection", "component", "vertex"):
            raise _err("INVALID_ARGUMENT", "per is 'selection', 'component' or 'vertex'",
                       f"{loc}.per")  # fmt: skip
        if raw.get("joint") is not None:
            c = self.col(raw["joint"], f"{loc}.joint", add=True)
            self.w[rows] = 0.0
            self.w[rows, c] = 1.0
            return
        if per == "vertex":
            dom = np.argmax(self.w[rows], axis=1)
            self.w[rows] = 0.0
            self.w[np.flatnonzero(rows), dom] = 1.0
            return
        groups = [rows] if per == "selection" else [
            rows & (self.comp == c) for c in np.unique(self.comp[rows]).tolist()]  # fmt: skip
        for g in groups:
            if not g.any():
                continue
            c = int(np.argmax(self.w[g].sum(axis=0)))
            self.w[g] = 0.0
            self.w[g, c] = 1.0

    def _op_lock(self, raw, sel, loc):
        self._lock(raw, sel, loc, True)

    def _op_unlock(self, raw, sel, loc):
        self._lock(raw, sel, loc, False)

    def _lock(self, raw, sel, loc, value: bool) -> None:
        if raw.get("joints") is not None:
            names = raw["joints"]
            if not isinstance(names, list):
                raise _err("INVALID_ARGUMENT", "joints is a list of names", f"{loc}.joints")
            for i, n in enumerate(names):
                self.locked_cols[self.col(n, f"{loc}.joints[{i}]", add=True)] = value
        if raw.get("select") is not None:
            self.locked_rows[sel] = value

    # -- result -------------------------------------------------------------------------------

    def result(self) -> Any:
        """The edited skin. Vertices no operation changed keep their weights bit for bit."""
        from .rig import make_skin

        w = np.clip(self.w, 0.0, None)
        old = self.skin
        m0 = len(old.joints)
        base = old.dense()
        touched = np.any(w[:, :m0] != base, axis=1)
        if w.shape[1] > m0:
            touched |= np.any(w[:, m0:] != 0.0, axis=1)
        empty = touched & (w.sum(axis=1) <= 0)
        if empty.any():
            raise _err("INVALID_SKIN", f"{int(empty.sum())} vertices would have no weight",
                       "operations", hint="assign them to a joint")  # fmt: skip
        k_old = int(old.indices.shape[1])
        wt = w[touched]
        if wt.size:
            limit = min(8, max(self.k, 1))
            wt = _cap_keep_locked(wt, max(limit, int((wt > 0).sum(axis=1).max())),
                                  self.locked_cols)  # fmt: skip
            if int((wt > 0).sum(axis=1).max()) > 8:
                wt = _cap_keep_locked(wt, 8, self.locked_cols)
            k_t = int((wt > 0).sum(axis=1).max())
        else:
            k_t = 1
        k = max(k_t, k_old if (~touched).any() else 1)
        n = w.shape[0]
        idx = np.zeros((n, k), np.int32)
        ww = np.zeros((n, k))
        keep = ~touched
        idx[keep, :k_old] = old.indices[keep][:, :k]
        ww[keep, :k_old] = old.weights[keep][:, :k]
        if wt.size:
            ti, tw = _sparse(wt, k)
            idx[touched], ww[touched] = ti, tw
        binds = list(old.inverse_bind)
        extra = getattr(self, "new_binds", {})
        for jid in self.palette[len(binds) :]:
            binds.append(extra[jid])
        locks = {
            "joints": [j for j, lk in zip(self.palette, self.locked_cols, strict=True) if lk],
            "vertices": np.flatnonzero(self.locked_rows).tolist(),
        }
        new = make_skin(
            old.object, old.rig, self.palette, np.array(binds), idx, ww,
            old.method, "skin", dq=old.dq, locks=locks,
        )  # fmt: skip
        exact = new.weights.copy()
        exact[keep] = ww[keep]  # (make_skin renormalizes every row; untouched rows were)
        exact.flags.writeable = False
        self.touched = touched
        return new.replace(weights=exact)


def _cap_keep_locked(w: np.ndarray, k: int, locked: np.ndarray) -> np.ndarray:
    """Each row's ``k`` largest weights (locked joints first), renormalized around the
    locked ones."""
    if w.shape[1] <= k:
        return _renormalize(w, locked)
    key = w + locked[None, :] * 2.0
    order = np.argsort(-key, axis=1, kind="stable")
    out = w.copy()
    np.put_along_axis(out, order[:, k:], 0.0, axis=1)
    return _renormalize(out, locked)


def _vec3(value: Any, location: str) -> np.ndarray:
    try:
        v = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err("INVALID_ARGUMENT", "a point is [x, y, z]", location) from None
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise _err("INVALID_ARGUMENT", "a point is [x, y, z] (finite)", location)
    return v


def edit_weights(
    system: AnimationSystem, mesh_object: Any, operations: list[Any], *, dry_run: bool = False
) -> dict[str, Any]:
    """Apply weight ``operations`` (``assign``, ``add``, ``subtract``, ``scale``,
    ``replace``, ``normalize``, ``prune``, ``cap``, ``smooth``, ``mirror``, ``rigid``,
    ``lock``, ``unlock``) to ``mesh_object``'s skin -- one undo step."""
    if not isinstance(operations, list) or not 1 <= len(operations) <= 10_000:
        raise _err("INVALID_ARGUMENT", "operations is a list of 1..10000", "operations")
    ed = _Edit(system, mesh_object)
    before = ed.w.copy()
    for i, raw in enumerate(operations):
        if not isinstance(raw, dict):
            raise _err("INVALID_ARGUMENT", "an operation is {op, ...}", f"operations[{i}]")
        ed.run(raw, f"operations[{i}]")
    new_skin = ed.result()
    del before
    report = {
        "mesh": ed.obj.name,
        "operations": ed.notes,
        "vertices_changed": int(ed.touched.sum()),
        "influences": int(new_skin.indices.shape[1]),
        "locked_joints": [ed.names[i] for i in np.flatnonzero(ed.locked_cols)],
        "locked_vertices": int(ed.locked_rows.sum()),
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        with system._op(f"Edit weights of {ed.obj.name!r}"):
            system._set(ed.item, definition=new_skin)
    return report


def normalize_weights(system: AnimationSystem, mesh_object: Any) -> dict[str, Any]:
    return edit_weights(system, mesh_object, [{"op": "normalize"}])


# --------------------------------------------------------------------------------------- #
# Transfer
# --------------------------------------------------------------------------------------- #


def transfer_weights(
    system: AnimationSystem,
    target: Any,
    source: Any,
    *,
    max_distance: float | None = None,
    influences: int = 4,
    unmatched: str = "rigid",
    smooth: int = 2,
    skinning: str | None = None,
) -> dict[str, Any]:
    """Bind ``target`` (clothing, armour, gloves...) with the weights of ``source``'s skin
    at the closest point of its surface (barycentric blend) -- one undo step. Vertices
    farther than ``max_distance`` (default: 5 % of the source's size) are reported; whole
    parts that far are bound rigidly to their nearest joint (``unmatched="rigid"``),
    keep the transferred weights (``"keep"``) or make the call fail (``"refuse"``)."""
    from ..rendering.scene_data import world_matrices
    from ..spatial.geometry import closest_points_on_triangles

    if unmatched not in ("rigid", "keep", "refuse"):
        raise _err("INVALID_ARGUMENT", "unmatched is 'rigid', 'keep' or 'refuse'", "unmatched")
    k = _int(influences, "influences", 1, 8)
    smooth = _int(smooth, "smooth", 0, 100)
    src = _Edit(system, source)
    tgt_obj = system._object(target, "target")
    if tgt_obj._id == src.obj._id:
        raise _err("INVALID_ARGUMENT", "the target is the source", "target")
    mesh = tgt_obj.mesh
    if mesh is None or mesh.vertex_count == 0:
        raise _err("INVALID_SKIN", f"{tgt_obj.name!r} has no mesh", "target")
    worlds = world_matrices(system.scene)
    tw = worlds[tgt_obj._id]
    tp = mesh.positions.astype(np.float64) @ tw[:3, :3].T + tw[:3, 3]
    size = float(np.max(src.world.max(axis=0) - src.world.min(axis=0)))
    md = 0.05 * size if max_distance is None else _number(max_distance, "max_distance", 0.0)
    # Candidates: triangles around the 8 nearest source vertices of each target vertex.
    near, _ = nearest_points(src.world, tp, 8)
    tri = np.asarray(src.mesh.triangles, np.int64)
    vt_order = np.argsort(tri.ravel(), kind="stable")
    vt_tri = vt_order // 3
    vt_ptr = np.searchsorted(tri.ravel()[vt_order], np.arange(src.w.shape[0] + 1))
    rows, cands = [], []
    for j in range(near.shape[1]):
        v = near[:, j]
        cnt = vt_ptr[v + 1] - vt_ptr[v]
        rows.append(np.repeat(np.arange(tp.shape[0]), cnt))
        first = np.repeat(vt_ptr[v], cnt)
        within = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cands.append(vt_tri[first + within])
    rows = np.concatenate(rows)
    cands = np.concatenate(cands)
    pairs = np.unique(np.stack([rows, cands], axis=1), axis=0)
    rows, cands = pairs[:, 0], pairs[:, 1]
    sw = src.world
    best_d = np.full(tp.shape[0], np.inf)
    best_t = np.zeros(tp.shape[0], np.int64)
    best_b = np.zeros((tp.shape[0], 3))
    for start in range(0, rows.size, 1_000_000):
        r = rows[start : start + 1_000_000]
        c = cands[start : start + 1_000_000]
        pts, bary = closest_points_on_triangles(tp[r], sw[tri[c, 0]], sw[tri[c, 1]], sw[tri[c, 2]])
        d = np.linalg.norm(pts - tp[r], axis=1)
        order = np.lexsort((c, d, r))
        r2 = r[order]
        first = np.ones(r2.size, bool)
        first[1:] = r2[1:] != r2[:-1]
        sel = order[first]
        better = d[sel] < best_d[r[sel]]
        idx = r[sel][better]
        best_d[idx] = d[sel][better]
        best_t[idx] = c[sel][better]
        best_b[idx] = bary[sel][better]
    corner = tri[best_t]
    dense = (
        best_b[:, 0:1] * src.w[corner[:, 0]]
        + best_b[:, 1:2] * src.w[corner[:, 1]]
        + best_b[:, 2:3] * src.w[corner[:, 2]]
    )
    # Confidence: distance and facing (a surface's inside is a poor match).
    tn = mesh.normals
    conf = np.exp(-((best_d / max(md, 1e-12)) ** 2))
    if tn is not None and src.mesh.normals is not None:
        n_t = tn.astype(np.float64) @ tw[:3, :3].T
        n_t /= np.maximum(np.linalg.norm(n_t, axis=1, keepdims=True), 1e-12)
        sn = src.mesh.normals.astype(np.float64)
        n_s = (
            best_b[:, 0:1] * sn[corner[:, 0]]
            + best_b[:, 1:2] * sn[corner[:, 1]]
            + best_b[:, 2:3] * sn[corner[:, 2]]
        )
        n_s /= np.maximum(np.linalg.norm(n_s, axis=1, keepdims=True), 1e-12)
        facing = np.sum(n_t * n_s, axis=1)
        conf *= 0.5 + 0.5 * np.clip(facing, 0.0, 1.0)
    far = best_d > md
    comp, n_comp = V.mesh_components(mesh.vertex_count, mesh.triangles)
    unmatched_parts = []
    parts = []
    for c in range(n_comp):
        members = comp == c
        parts.append(
            {"part": c, "vertices": int(members.sum()),
             "median_distance": round(float(np.median(best_d[members])), 6),
             "matched": bool(far[members].mean() <= 0.5)}
        )  # fmt: skip
        if far[members].mean() > 0.5:
            unmatched_parts.append(
                {"part": c, "vertices": int(members.sum()),
                 "distance": round(float(np.median(best_d[members])), 6)}
            )  # fmt: skip
    if unmatched_parts and unmatched == "refuse":
        raise _err(
            "UNMATCHED_PARTS",
            f"{len(unmatched_parts)} parts of {tgt_obj.name!r} are farther than "
            f"{md:g} from {src.obj.name!r}",
            "target",
        )
    if unmatched == "rigid":
        ctx = _Context(system, tgt_obj, src.arm, src.names)
        for part in unmatched_parts:
            members = comp == part["part"]
            d = ctx.euclidean(tp[members]).min(axis=0)
            j = int(np.argmin(d))
            dense[members] = 0.0
            dense[members, j] = 1.0
            part["joint"] = src.names[j]
    if smooth:
        a, b = mesh_edges(mesh.triangles)
        rigid_rows = np.zeros(mesh.vertex_count, bool)
        for part in unmatched_parts:
            rigid_rows |= comp == part["part"]
        dense = smooth_rows(dense, a, b, smooth, 0.5, rigid_rows)
    dense = _cap(_normalize_rows(np.clip(dense, 0.0, None)), k)
    idx, w = _sparse(dense, k)
    method = skinning or src.skin.method
    report = system.skin(
        tgt_obj, src.arm, weights={"indices": idx, "weights": w, "joints": src.names},
        influences=k, method=method if method != "blended" else "linear",
    )  # fmt: skip
    q = np.quantile(best_d, [0.5, 0.95, 1.0]) if best_d.size else np.zeros(3)
    return {
        "target": tgt_obj.name,
        "source": src.obj.name,
        "vertices": int(mesh.vertex_count),
        "distance": {"median": float(q[0]), "p95": float(q[1]), "max": float(q[2])},
        "max_distance": md,
        "far_vertices": int(far.sum()),
        "confidence": {
            "mean": round(float(conf.mean()), 4),
            "min": round(float(conf.min()), 4),
            "low_vertices": int((conf < 0.5).sum()),
        },  # fmt: skip
        "parts": parts[:64],
        "unmatched_parts": unmatched_parts,
        "skin": report,
    }


# --------------------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------------------- #


def inspect_weights(
    system: AnimationSystem, mesh_object: Any, *, joint: str | None = None, max_list: int = 20
) -> dict[str, Any]:
    """A skin's weights measured: validity, per-joint coverage, leakage (weights on a
    joint far beyond where that joint dominates), cross-side influences, unused joints and
    helper joints with weights. ``joint``: also that joint's per-vertex weights."""
    ed = _Edit(system, mesh_object)
    w = ed.w
    n, m = w.shape
    rig = ed.rig
    findings: list[dict[str, Any]] = []
    total = w.sum(axis=1)
    bad_sum = np.flatnonzero(np.abs(total - 1.0) > 1e-6)
    if bad_sum.size:
        findings.append(
            {"code": "WEIGHTS_NOT_NORMALIZED", "severity": "error",
             "vertices": bad_sum[:max_list].tolist(), "count": int(bad_sum.size)}
        )  # fmt: skip
    if not np.all(np.isfinite(w)) or (w < 0).any():
        findings.append({"code": "WEIGHTS_INVALID", "severity": "error"})
    # Joint positions in world space at rest (for leakage).
    model = rig.rest_model()
    joints_info = []
    leak_rows: list[int] = []
    side_rows: list[int] = []
    dominant = np.argmax(w, axis=1)
    for c, name in enumerate(ed.names):
        k = rig.index(name)
        col = w[:, c]
        used = col > 0
        head = (ed.a_world @ model[k])[:3, 3]
        dist = np.linalg.norm(ed.world - head, axis=1)
        own = dominant == c
        reach = float(np.quantile(dist[own], 0.95)) if own.any() else 0.0
        info = {
            "joint": name,
            "vertices": int(used.sum()),
            "dominant_vertices": int(own.sum()),
            "max": round(float(col.max()), 6),
            "sum": round(float(col.sum()), 4),
        }
        if own.any():
            far = used & (dist > 2.5 * max(reach, 1e-6)) & (col > 0.05)
            if far.any():
                info["far_vertices"] = int(far.sum())
                leak_rows.extend(np.flatnonzero(far)[:max_list].tolist())
        joints_info.append(info)
        # Cross-side: a sided joint weighing on the other side's vertices.
        side = name[-2:] if name[-2:] in (".L", ".R") else None
        role = rig.joints[k].role
        if role and role[-2:] in (".L", ".R"):
            side = role[-2:]
        if side:
            inv = np.linalg.inv(ed.a_world)
            x = ed.world @ inv[:3, :3].T + inv[:3, 3]
            margin = SIDE_MARGIN * float(np.max(ed.world.max(axis=0) - ed.world.min(axis=0)))
            wrong = (x[:, 0] < -margin) if side == ".L" else (x[:, 0] > margin)
            across = used & wrong & (col > 0.05)
            if across.any():
                info["other_side_vertices"] = int(across.sum())
                side_rows.extend(np.flatnonzero(across)[:max_list].tolist())
        if not rig.joints[k].deform and used.any():
            findings.append({"code": "HELPER_JOINT_WEIGHTED", "severity": "warning",
                             "joint": name, "count": int(used.sum())})  # fmt: skip
    if leak_rows:
        findings.append({"code": "WEIGHT_LEAK", "severity": "warning", "basis": "geometry",
                         "message": "weights far from where their joint dominates",
                         "vertices": sorted(set(leak_rows))[:max_list],
                         "count": len(set(leak_rows))})  # fmt: skip
    if side_rows:
        findings.append({"code": "CROSS_SIDE_WEIGHTS", "severity": "warning", "basis": "geometry",
                         "message": "a left or right joint weighs on the other side",
                         "vertices": sorted(set(side_rows))[:max_list],
                         "count": len(set(side_rows))})  # fmt: skip
    unused = []
    for k in rig.deforming:
        jid = rig.joints[k].id
        if jid not in ed.palette or not w[:, ed.palette.index(jid)].any():
            unused.append(rig.joints[k].name)
    if unused:
        findings.append({"code": "UNUSED_JOINTS", "severity": "info", "joints": unused[:max_list],
                         "count": len(unused)})  # fmt: skip
    counts = (w > 0).sum(axis=1)
    out = {
        "mesh": ed.obj.name,
        "vertices": n,
        "joints": m,
        "max_influences": int(counts.max()),
        "mean_influences": round(float(counts.mean()), 3),
        "locked_joints": [ed.names[i] for i in np.flatnonzero(ed.locked_cols)],
        "locked_vertices": int(ed.locked_rows.sum()),
        "per_joint": joints_info,
        "findings": findings,
        "valid": not any(f["severity"] == "error" for f in findings),
    }
    if joint is not None:
        c = ed.col(joint, "joint")
        out["weights"] = w[:, c].round(6).tolist()
    return out


def weight_colors(weights: np.ndarray) -> np.ndarray:
    """A heat ramp (dark blue -> cyan -> green -> yellow -> red) for weights 0..1, RGBA."""
    w = np.clip(np.asarray(weights, np.float64), 0.0, 1.0)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array(
        [
            [0.05, 0.05, 0.35],
            [0.0, 0.55, 0.9],
            [0.1, 0.8, 0.25],
            [0.95, 0.85, 0.1],
            [0.9, 0.1, 0.05],
        ]
    )
    rgb = np.stack([np.interp(w, stops, colors[:, i]) for i in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((w.size, 1))], axis=1)
