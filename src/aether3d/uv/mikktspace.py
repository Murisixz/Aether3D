"""MikkTSpace tangents: an exact port of the reference ``mikktspace.c`` for triangles.

Morten S. Mikkelsen's reference implementation (zlib license; Mikkelsen 2008) is the
tangent space glTF 2.0 specifies when a mesh carries no TANGENT attribute, and the one
Blender, xNormal, Substance and Unreal bake and render with. This module reproduces it
operation by operation, in float32, for meshes of triangles (``genTangSpaceDefault``:
no angular threshold):

1. **Welding.** Corners are the same vertex when position, normal and texture coordinate
   are exactly equal.
2. **Degenerate triangles** (two equal positions) are set aside.
3. **Per triangle** (equations 18-19 of the thesis): the first-order derivatives
   ``vOs = t31y d1 - t21y d2`` and ``vOt = -t31x d1 + t21x d2``, normalized with the sign
   of the texture area (orientation), their magnitudes, and whether the triangle is usable
   ("group with anything" otherwise: zero texture area or zero derivative).
4. **Neighbours**: triangles sharing a welded edge in opposite directions (pairs found in
   the order of the sorted edge list, as the reference does for non-manifold edges).
5. **Groups**: around every welded vertex, the triangles connected through neighbour
   links that share the vertex and the orientation; a group-with-anything triangle takes
   the orientation of the first group that reaches it (the reference's only order
   dependency, reproduced by visiting groups in the same order).
6. **Tangent of a group**: the sum of the members' derivatives projected onto the
   vertex's tangent plane and normalized, weighted by the corner's angle, accumulated in
   ascending triangle order in float32, then normalized.
7. **Degenerate triangles** copy the tangent space of the first good corner with the same
   welded vertex.

The result for each corner is the unit tangent and the sign of the bitangent
(``bitangent = sign * cross(normal, tangent)``), exactly as the reference's
``setTSpaceBasic`` callback receives them -- bit for bit (``tests/test_tangents.py``, on
fixtures computed by the reference C), with one documented exception: the reference's
edge sort leaves its *last* group unsorted, so where that group holds a non-manifold edge
(or more than one pair) it may pair other triangles than the sorted rule used here.
Manifold meshes give identical results.
"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32
FLT_MIN = np.finfo(np.float32).tiny


def _not_zero(x: np.ndarray) -> np.ndarray:
    return np.abs(x) > FLT_MIN


def _vnot_zero(v: np.ndarray) -> np.ndarray:
    return _not_zero(v[..., 0]) | _not_zero(v[..., 1]) | _not_zero(v[..., 2])


def _length(v: np.ndarray) -> np.ndarray:
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def _normalize_where(v: np.ndarray) -> np.ndarray:
    """``Normalize(v)`` where ``VNotZero(v)`` (``vscale(1 / Length(v), v)``), else ``v``."""
    ok = _vnot_zero(v)
    length = _length(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = F32(1.0) / length
    out = inv[..., None] * v
    return np.where(ok[..., None], out, v)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _project(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """``vsub(v, vscale(vdot(n, v), n))``."""
    return v - _dot(n, v)[..., None] * n


def mikktspace(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Per-corner tangents ``(T, 3, 4)`` (unit xyz, then the sign +-1) of triangles given by
    corner arrays ``positions`` ``(T, 3, 3)``, ``normals`` ``(T, 3, 3)`` (unit) and
    ``uvs`` ``(T, 3, 2)``."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _mikktspace(positions, normals, uvs)


def _mikktspace(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    P = np.ascontiguousarray(positions, dtype=F32)
    N = np.ascontiguousarray(normals, dtype=F32)
    UV = np.ascontiguousarray(uvs, dtype=F32)
    T = P.shape[0]
    out = np.zeros((T, 3, 4), dtype=F32)
    out[..., 0] = 1.0
    out[..., 3] = -1.0
    if T == 0:
        return out
    # 1. Welding by exact equality of (position, normal, uv); -0.0 equals 0.0.
    key = np.concatenate([P, N, UV], axis=2).reshape(3 * T, 8) + F32(0.0)
    rows = np.ascontiguousarray(key).view(np.dtype((np.void, 32))).reshape(-1)
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    weld = first[inverse.reshape(-1)].reshape(T, 3)  # representative: the lowest corner
    # Non-finite keys never weld in the reference (NaN != NaN): give them their own ids.
    finite = np.isfinite(key).all(axis=1).reshape(T, 3)
    weld = np.where(finite, weld, np.arange(3 * T).reshape(T, 3))
    # 2. Degenerate triangles: two equal positions.
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    degen = np.all(p0 == p1, axis=1) | np.all(p0 == p2, axis=1) | np.all(p1 == p2, axis=1)
    good = np.flatnonzero(~degen)  # in original order; "triangle numbers" below
    G = good.shape[0]
    W = weld[good]  # (G, 3)
    # 3. Per-triangle derivatives (InitTriInfo).
    v1, v2, v3 = P[good, 0], P[good, 1], P[good, 2]
    t1, t2, t3 = UV[good, 0], UV[good, 1], UV[good, 2]
    t21x = t2[:, 0] - t1[:, 0]
    t21y = t2[:, 1] - t1[:, 1]
    t31x = t3[:, 0] - t1[:, 0]
    t31y = t3[:, 1] - t1[:, 1]
    d1 = v2 - v1
    d2 = v3 - v1
    area2 = t21x * t31y - t21y * t31x
    vOs = t31y[:, None] * d1 - t21y[:, None] * d2
    vOt = (-t31x)[:, None] * d1 + t21x[:, None] * d2
    orient = area2 > 0
    gwa = np.ones(G, dtype=bool)
    tri_os = np.zeros((G, 3), dtype=F32)
    tri_ot = np.zeros((G, 3), dtype=F32)
    mag_s = np.zeros(G, dtype=F32)
    mag_t = np.zeros(G, dtype=F32)
    nz = _not_zero(area2)
    if nz.any():
        abs_area = np.abs(area2)
        len_os = _length(vOs)
        len_ot = _length(vOt)
        fs = np.where(orient, F32(1.0), F32(-1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            sc_os = fs / len_os
            sc_ot = fs / len_ot
            ms = len_os / abs_area
            mt = len_ot / abs_area
        ok_os = nz & _not_zero(len_os)
        ok_ot = nz & _not_zero(len_ot)
        tri_os = np.where(ok_os[:, None], sc_os[:, None] * vOs, tri_os)
        tri_ot = np.where(ok_ot[:, None], sc_ot[:, None] * vOt, tri_ot)
        mag_s = np.where(nz, ms, mag_s).astype(F32)
        mag_t = np.where(nz, mt, mag_t).astype(F32)
        gwa = ~(nz & _not_zero(mag_s) & _not_zero(mag_t))
    # 4. Neighbours (BuildNeighborsFast): edge e of triangle f runs W[f, e] -> W[f, e+1].
    nbr = _neighbors(W)
    # 5. Groups (Build4RuleGroups), with group-with-anything orientations resolved.
    group, group_orient, rep_corner, orient = _groups(W, nbr, orient, gwa)
    # 6. Tangent spaces (GenerateTSpaces + EvalTspace), default threshold.
    ts_os, ts_bool = _tspaces(
        P[good], N[good], W, group, group_orient, rep_corner, tri_os, tri_ot, gwa, orient
    )
    # Write the good triangles' corners.
    out_good = np.zeros((G, 3, 4), dtype=F32)
    out_good[..., 0] = 1.0
    out_good[..., 3] = -1.0
    has = group >= 0
    out_good[..., :3] = np.where(has[..., None], ts_os, out_good[..., :3])
    out_good[..., 3] = np.where(has, np.where(ts_bool, F32(1.0), F32(-1.0)), out_good[..., 3])
    out[good] = out_good
    # 7. Degenerate triangles: the first good corner with the same welded index.
    bad = np.flatnonzero(degen)
    if bad.size:
        flat_w = W.reshape(-1)
        order = np.argsort(flat_w, kind="stable")
        sw = flat_w[order]
        firsts = np.flatnonzero(np.r_[True, sw[1:] != sw[:-1]])
        keys = sw[firsts]
        src = order[firsts]  # lowest good corner position (good-triangle order)
        q = weld[bad].reshape(-1)
        pos = np.searchsorted(keys, q)
        found = (pos < keys.shape[0]) & (keys[np.minimum(pos, keys.shape[0] - 1)] == q)
        flat_good = out_good.reshape(-1, 4)
        vals = np.where(
            found[:, None],
            flat_good[src[np.minimum(pos, keys.shape[0] - 1)]],
            np.array([1.0, 0.0, 0.0, -1.0], dtype=F32),
        )
        out[bad] = vals.reshape(-1, 3, 4)
    return out


def _neighbors(W: np.ndarray) -> np.ndarray:
    """``(G, 3)`` neighbour triangle across each edge, or -1 (BuildNeighborsFast)."""
    G = W.shape[0]
    nbr = np.full((G, 3), -1, dtype=np.int64)
    a = W.reshape(-1)
    b = W[:, [1, 2, 0]].reshape(-1)
    i0, i1 = np.minimum(a, b), np.maximum(a, b)
    f = np.repeat(np.arange(G), 3)
    e = np.tile(np.arange(3), G)
    order = np.lexsort((f, i1, i0))
    i0s, i1s, fs, es = i0[order], i1[order], f[order], e[order]
    forward = a[order] == i0s  # the triangle runs i0 -> i1 along this edge
    same = np.r_[False, (i0s[1:] == i0s[:-1]) & (i1s[1:] == i1s[:-1])]
    starts = np.flatnonzero(~same)
    lengths = np.diff(np.r_[starts, i0s.shape[0]])
    two = starts[lengths == 2]
    opp = forward[two] != forward[two + 1]
    s = two[opp]
    nbr[fs[s], es[s]] = fs[s + 1]
    nbr[fs[s + 1], es[s + 1]] = fs[s]
    # Edges shared by more than two triangles: the reference's greedy pairing.
    for start in starts[lengths > 2].tolist():
        n = int(lengths[np.searchsorted(starts, start)])
        idx = list(range(start, start + n))
        for x in idx:
            fa, ea = int(fs[x]), int(es[x])
            if nbr[fa, ea] != -1:
                continue
            for y in idx:
                if y <= x:
                    continue
                fb, eb = int(fs[y]), int(es[y])
                if forward[x] != forward[y] and nbr[fb, eb] == -1:
                    nbr[fa, ea] = fb
                    nbr[fb, eb] = fa
                    break
    return nbr


def _groups(W, nbr, orient, gwa):
    """Group id of every corner (-1: none), each group's orientation and representative
    corner, and the triangles' final orientations."""
    G = W.shape[0]
    orient = orient.copy()
    group = np.full((G, 3), -1, dtype=np.int64)
    if G == 0:
        return group, np.zeros(0, bool), np.zeros(0, np.int64), orient
    # Corner links: corner (f, i) and the corner of the same welded vertex in each of the
    # two neighbours across the edges at corner i.
    links_a, links_b = [], []
    for e_off in (0, 2):  # FaceNeighbors[i] (edge i) and FaceNeighbors[i-1] (edge i-1)
        for i in range(3):
            e = (i + e_off) % 3 if e_off == 0 else (i + 2) % 3
            g = nbr[:, e]
            ok = g >= 0
            f = np.flatnonzero(ok)
            gg = g[ok]
            v = W[f, i]
            j = np.where(W[gg, 0] == v, 0, np.where(W[gg, 1] == v, 1, 2))
            links_a.append(3 * f + i)
            links_b.append(3 * gg + j)
    la = np.concatenate(links_a)
    lb = np.concatenate(links_b)
    tri_a, tri_b = la // 3, lb // 3
    dirty_vertex = np.zeros(int(W.max()) + 1, dtype=bool)
    dirty_vertex[W[gwa].reshape(-1)] = True
    corner_dirty = dirty_vertex[W.reshape(-1)]
    # Clean corners: components of links between non-GWA triangles of equal orientation.
    clean = ~corner_dirty[la] & ~corner_dirty[lb]
    use = clean & ~gwa[tri_a] & ~gwa[tri_b] & (orient[tri_a] == orient[tri_b])
    from_labels = _union(3 * G, la[use], lb[use])
    labels = from_labels.copy()
    # Exact sequential groups for corners of vertices touched by GWA triangles.
    if corner_dirty.any():
        labels, orient = _dirty_groups(W, nbr, orient, gwa, corner_dirty, labels)
    flat_tri = np.repeat(np.arange(G), 3)
    # A non-GWA corner is always in a group (its own when unlinked). A GWA corner is in a
    # group only if reached; _dirty_groups marks those by pointing them at their seed.
    reached = labels != np.arange(3 * G)
    member = (~gwa[flat_tri]) | reached
    seeds = np.unique(labels[member])
    gid = np.full(3 * G, -1, dtype=np.int64)
    gid[member] = np.searchsorted(seeds, labels[member])
    group = gid.reshape(G, 3)
    group_orient = orient[seeds // 3]
    return group, group_orient, seeds, orient


def _union(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parent = np.arange(n, dtype=np.int64)
    if a.size == 0:
        return parent
    while True:
        pa, pb = parent[a], parent[b]
        differ = pa != pb
        if not differ.any():
            return parent
        np.minimum.at(
            parent, np.maximum(pa[differ], pb[differ]), np.minimum(pa[differ], pb[differ])
        )
        while True:
            up = parent[parent]
            if np.array_equal(up, parent):
                break
            parent = up


def _dirty_groups(W, nbr, orient, gwa, corner_dirty, labels):
    """Build4RuleGroups exactly (seed order, depth-first, left before right) for the
    corners of vertices that touch a group-with-anything triangle."""
    G = W.shape[0]
    labels = labels.copy()
    orient = orient.copy()
    assigned = np.zeros((G, 3), dtype=bool)
    Wl = W.tolist()
    nb = nbr.tolist()
    gw = gwa.tolist()
    ori = orient.tolist()
    dirty = corner_dirty.reshape(G, 3).tolist()
    has_any = [False] * G  # any corner of the triangle assigned yet
    for f in range(G):
        if gw[f]:
            continue
        for i in range(3):
            if not dirty[f][i] or assigned[f, i]:
                continue
            seed = 3 * f + i
            v = Wl[f][i]
            g_orient = ori[f]
            assigned[f, i] = True
            has_any[f] = True
            labels[seed] = seed
            # Iterative DFS reproducing AssignRecur(L) then AssignRecur(R) recursion order.
            stack = []
            for g in (nb[f][(i + 2) % 3], nb[f][i]):  # pushed R then L: L pops first
                if g >= 0:
                    stack.append(g)
            while stack:
                t = stack.pop()
                row = Wl[t]
                j = 0 if row[0] == v else (1 if row[1] == v else 2)
                if assigned[t, j]:
                    continue
                if gw[t] and not has_any[t]:
                    ori[t] = g_orient
                if ori[t] != g_orient:
                    continue
                assigned[t, j] = True
                has_any[t] = True
                labels[3 * t + j] = seed
                for g in (nb[t][(j + 2) % 3], nb[t][j]):
                    if g >= 0:
                        stack.append(g)
    return labels, np.array(ori, dtype=bool)


def _tspaces(P, N, W, group, group_orient, rep_corner, tri_os, tri_ot, gwa, orient):
    """Tangent of every grouped corner: the group's (single, default-threshold) subgroup
    tangent, evaluated as EvalTspace does."""
    G = W.shape[0]
    ts = np.zeros((G, 3, 3), dtype=F32)
    ts[..., 0] = 1.0
    flags = np.zeros((G, 3), dtype=bool)
    sel = group >= 0
    if not sel.any():
        return ts, flags
    f_idx, c_idx = np.nonzero(sel)
    g = group[f_idx, c_idx]
    ngroups = int(group.max()) + 1
    # Members of each group: (group, triangle) pairs. Subgroups with the default threshold:
    # members whose projected derivatives are not exactly opposite (checked below).
    n = N[f_idx, c_idx]  # every corner of a group has the group's normal (welded)
    pos_os = _normalize_where(_project(tri_os[f_idx], n))
    pos_ot = _normalize_where(_project(tri_ot[f_idx], n))
    # The default threshold cos(180 deg) = -1 separates only exactly opposite directions.
    order = np.lexsort((f_idx, g))
    g_s, f_s, c_s = g[order], f_idx[order], c_idx[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])
    lengths = np.diff(np.r_[starts, g_s.shape[0]])
    split_groups = np.zeros(ngroups, dtype=bool)
    if (lengths > 1).any():
        # Pairs within each group.
        gl = lengths[lengths > 1]
        gs = starts[lengths > 1]
        for L in np.unique(gl).tolist():
            st = gs[gl == L]
            ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
            a_ = (st[:, None] + ii.reshape(-1)).reshape(-1)
            b_ = (st[:, None] + jj.reshape(-1)).reshape(-1)
            oa, ob = order[a_], order[b_]
            cos_s = _dot(pos_os[oa], pos_os[ob])
            cos_t = _dot(pos_ot[oa], pos_ot[ob])
            anyg = gwa[f_s[a_]] | gwa[f_s[b_]]
            same = f_s[a_] == f_s[b_]
            member = anyg | same | ((cos_s > F32(-1.0)) & (cos_t > F32(-1.0)))
            bad = ~member
            if bad.any():
                split_groups[g_s[a_[bad]]] = True
    # Evaluate each group's tangent (EvalTspace) over its non-GWA members in ascending
    # triangle order, accumulating in float32.
    contrib = ~gwa[f_s]
    fs_c = f_s[contrib]
    cs_c = c_s[contrib]
    gs_c = g_s[contrib]
    n_c = N[fs_c, cs_c]
    vos = _normalize_where(_project(tri_os[fs_c], n_c))
    vot = _normalize_where(_project(tri_ot[fs_c], n_c))
    i1 = cs_c
    i0 = (cs_c + 2) % 3
    i2 = (cs_c + 1) % 3
    p0 = P[fs_c, i0]
    p1 = P[fs_c, i1]
    p2 = P[fs_c, i2]
    e1 = _normalize_where(_project(p0 - p1, n_c))
    e2 = _normalize_where(_project(p2 - p1, n_c))
    fcos = _dot(e1, e2)
    fcos = np.where(fcos > F32(1.0), F32(1.0), np.where(fcos < F32(-1.0), F32(-1.0), fcos))
    ang = np.arccos(fcos.astype(np.float64)).astype(F32)
    acc_os = np.zeros((ngroups, 3), dtype=F32)
    acc_ot = np.zeros((ngroups, 3), dtype=F32)
    # Ascending triangle order within each group (order is by (group, triangle)).
    np.add.at(acc_os, gs_c, ang[:, None] * vos)
    np.add.at(acc_ot, gs_c, ang[:, None] * vot)
    res_os = _normalize_where(acc_os)
    ts[f_idx, c_idx] = res_os[g]
    flags[f_idx, c_idx] = group_orient[g]
    if split_groups.any():
        # Rare: members with exactly opposite derivatives form subgroups (GenerateTSpaces).
        for grp in np.flatnonzero(split_groups).tolist():
            rows = np.flatnonzero(g_s == grp)
            tris = f_s[rows].tolist()
            corners = c_s[rows].tolist()
            k = len(rows)
            ro = order[rows]
            for a in range(k):
                members = []
                for b in range(k):
                    anyg = gwa[tris[a]] or gwa[tris[b]]
                    cs_ = float(_dot(pos_os[ro[a]], pos_os[ro[b]]))
                    ct_ = float(_dot(pos_ot[ro[a]], pos_ot[ro[b]]))
                    if anyg or tris[a] == tris[b] or (cs_ > -1.0 and ct_ > -1.0):
                        members.append(b)
                acc = np.zeros(3, dtype=F32)
                for b in members:  # ascending triangle order (rows are sorted)
                    if gwa[tris[b]]:
                        continue
                    t, c = tris[b], corners[b]
                    nn = N[t, c]
                    vo = _normalize_where(_project(tri_os[t], nn))
                    q0, q1, q2 = P[t, (c + 2) % 3], P[t, c], P[t, (c + 1) % 3]
                    w1 = _normalize_where(_project(q0 - q1, nn))
                    w2 = _normalize_where(_project(q2 - q1, nn))
                    fc = _dot(w1, w2)
                    fc = min(max(float(fc), -1.0), 1.0)
                    acc = acc + F32(math.acos(fc)) * vo
                ts[tris[a], corners[a]] = _normalize_where(acc)
    return ts, flags


def face_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Unit geometric normals ``(T, 3)`` float32 (zero for degenerate triangles): the corner
    normals of a mesh without normals, as the renderers shade it (flat)."""
    p = positions.astype(np.float64)[triangles]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 0).astype(F32)


def mesh_tangents(mesh) -> np.ndarray | None:
    """MikkTSpace tangents of every corner of a :class:`~aether3d.Mesh` (``(T, 3, 4)``
    float32, object space), from its positions, normals (flat face normals when it has
    none) and UVs; ``None`` without UVs. Cached on the mesh (meshes are immutable)."""
    if mesh.uvs is None:
        return None
    cached = mesh._derived_get("mikktspace")
    if cached is not None:
        return cached
    t = mesh.triangles.astype(np.int64)
    p = mesh.positions[t]
    if mesh.normals is not None:
        n = mesh.normals[t]
    else:
        n = np.repeat(face_normals(mesh.positions, t)[:, None, :], 3, axis=1)
    out = mikktspace(p, n, mesh.uvs[t])
    out.flags.writeable = False
    mesh._derived_set("mikktspace", out)
    return out
