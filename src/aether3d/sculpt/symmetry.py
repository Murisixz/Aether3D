"""Symmetrize by mirroring: cut the mesh at the plane, keep one half, mirror it.

:meth:`~aether3d.sculpt.SculptMesh.symmetrize` uses this when the two sides' topologies
differ (after dynamic topology or one-sided remeshing), where copying vertex positions
cannot reproduce the source side's shape:

1. Every vertex within the tolerance of the plane is moved onto it exactly.
2. Triangles with vertices on both sides are cut at the plane: each crossed edge gets one
   vertex on the plane (its position and mask interpolated along the edge, its UVs per
   render edge, so UV seams stay seams), and the source-side piece is kept (a triangle,
   or a quad split along its shorter diagonal).
3. Triangles with no vertex on the source side are removed, including triangles lying
   in the plane (they would be inside the mirrored surface).
4. The source half is mirrored -- vertices, render vertices (same UVs, their own normal
   groups) and triangles with reversed winding -- sharing the vertices on the plane, so
   the two halves join without a seam in the geometry.

Face groups, UVs and the mask are carried over; normals are recomputed at commit (hard
edges stay hard: render vertices keep their normal groups). The result is exactly
symmetric: mirrored vertices are exact negations of the coordinate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mirror_half(sm, axis: int, sign: float, tol: float) -> dict[str, Any]:
    """Replace the mesh by its ``sign`` side across ``axis = 0``, mirrored (see the
    module docs). Returns counts for the operation's ``data``."""
    tris = sm.triangle_ids()
    tri = sm._tri[tris]
    trv = sm._trv[tris]
    groups = sm._tgroup[tris]
    pos = sm._pos.copy()
    d = sign * pos[:, axis]
    on = np.abs(d) <= tol
    pos[on, axis] = 0.0
    d = np.where(on, 0.0, d)
    src_v = d > 0
    tgt_v = d < 0
    n_src = src_v[tri].sum(axis=1)
    n_tgt = tgt_v[tri].sum(axis=1)
    keep = (n_src > 0) & (n_tgt == 0)
    cut = (n_src > 0) & (n_tgt > 0)

    # Cut vertices: one per crossed welded edge, one render vertex per render edge.
    rows = np.flatnonzero(cut)
    ct, cr = tri[rows], trv[rows]
    k0 = np.repeat(np.arange(3)[None, :], len(rows), axis=0)
    a_w, b_w = ct, np.roll(ct, -1, axis=1)
    a_r, b_r = cr, np.roll(cr, -1, axis=1)
    crosses = (d[a_w] * d[b_w]) < 0
    e_rows, e_k = np.nonzero(crosses)
    wa, wb = a_w[e_rows, e_k], b_w[e_rows, e_k]
    ra, rb = a_r[e_rows, e_k], b_r[e_rows, e_k]
    flip = wa > wb  # orient every edge from its lower welded id
    wa, wb = np.where(flip, wb, wa), np.where(flip, wa, wb)
    ra, rb = np.where(flip, rb, ra), np.where(flip, ra, rb)
    nv = sm._nv
    wkey = wa * np.int64(nv) + wb
    uw, w_first, w_inv = np.unique(wkey, return_index=True, return_inverse=True)
    w_inv = w_inv.reshape(-1)
    ea, eb = wa[w_first], wb[w_first]
    t = d[ea] / (d[ea] - d[eb])
    cut_pos = pos[ea] + t[:, None] * (pos[eb] - pos[ea])
    cut_pos[:, axis] = 0.0
    cut_mask = sm._mask[ea] + t * (sm._mask[eb] - sm._mask[ea])
    cut_ids = nv + np.arange(len(uw))  # new welded ids
    nr = sm._nr
    rkey = np.stack([w_inv, ra, rb], axis=1)
    ur, r_first, r_inv = np.unique(rkey, axis=0, return_index=True, return_inverse=True)
    r_inv = r_inv.reshape(-1)
    cut_rgeo = cut_ids[ur[:, 0]]
    cut_rids = nr + np.arange(len(ur))
    tr = t[ur[:, 0]]
    cut_uv = None
    if sm._ruv is not None:
        uva, uvb = sm._ruv[ur[:, 1]], sm._ruv[ur[:, 2]]
        cut_uv = uva + tr[:, None] * (uvb - uva)
    # Normal groups of cut render vertices: one per cut vertex and pair of groups.
    pair = np.stack([ur[:, 0], sm._rng[ur[:, 1]], sm._rng[ur[:, 2]]], axis=1)
    _, cut_ng = np.unique(pair, axis=0, return_inverse=True)
    base_ng = int(sm._rng[:nr].max()) + 1 if nr else 0
    cut_rng = base_ng + cut_ng.reshape(-1)

    # The source-side piece of each cut triangle, walked around its corners.
    edge_w = np.full((len(rows), 3), -1, dtype=np.int64)
    edge_r = np.full((len(rows), 3), -1, dtype=np.int64)
    edge_w[e_rows, e_k] = cut_ids[w_inv]
    edge_r[e_rows, e_k] = cut_rids[r_inv]
    del k0
    pieces_w: list[list[int]] = []
    pieces_r: list[list[int]] = []
    pieces_g: list[int] = []
    for i in range(len(rows)):
        poly_w: list[int] = []
        poly_r: list[int] = []
        for k in range(3):
            w = int(ct[i, k])
            if d[w] >= 0:
                poly_w.append(w)
                poly_r.append(int(cr[i, k]))
            if edge_w[i, k] >= 0:
                poly_w.append(int(edge_w[i, k]))
                poly_r.append(int(edge_r[i, k]))
        g = int(groups[rows[i]])
        if len(poly_w) == 3:
            pieces_w.append(poly_w)
            pieces_r.append(poly_r)
            pieces_g.append(g)
        else:  # a quad: split along the shorter diagonal
            p4 = np.array([pos[w] if w < nv else cut_pos[w - nv] for w in poly_w], dtype=np.float64)
            if np.linalg.norm(p4[0] - p4[2]) <= np.linalg.norm(p4[1] - p4[3]):
                split = ((0, 1, 2), (0, 2, 3))
            else:
                split = ((0, 1, 3), (1, 2, 3))
            for s in split:
                pieces_w.append([poly_w[j] for j in s])
                pieces_r.append([poly_r[j] for j in s])
                pieces_g.append(g)
    half_w = np.concatenate([tri[keep], np.array(pieces_w, dtype=np.int64).reshape(-1, 3)])
    half_r = np.concatenate([trv[keep], np.array(pieces_r, dtype=np.int64).reshape(-1, 3)])
    half_g = np.concatenate([groups[keep], np.array(pieces_g, dtype=np.int64)])

    # All vertices (old and cut) in one table, then the mirror map.
    all_pos = np.concatenate([pos[:nv], cut_pos])
    all_mask = np.concatenate([sm._mask[:nv].astype(np.float64), cut_mask])
    all_rgeo = np.concatenate([sm._rgeo[:nr], cut_rgeo])
    all_rng = np.concatenate([sm._rng[:nr], cut_rng])
    all_uv = None if cut_uv is None else np.concatenate([sm._ruv[:nr], cut_uv])
    on_all = np.concatenate([on[:nv], np.ones(len(cut_ids), dtype=bool)])
    used_w = np.unique(half_w)
    used_r = np.unique(half_r)
    mirror_w = np.arange(len(all_pos))
    off_w = used_w[~on_all[used_w]]
    mirror_w[off_w] = len(all_pos) + np.arange(len(off_w))
    mirror_r = np.arange(len(all_rgeo))
    off_r = used_r[~on_all[all_rgeo[used_r]]]
    mirror_r[off_r] = len(all_rgeo) + np.arange(len(off_r))
    m_pos = all_pos[off_w].copy()
    m_pos[:, axis] *= -1.0
    ng_base = int(all_rng.max()) + 1 if len(all_rng) else 0
    tri_out = np.concatenate([half_w, mirror_w[half_w][:, ::-1]])
    trv_out = np.concatenate([half_r, mirror_r[half_r][:, ::-1]])
    groups_out = np.concatenate([half_g, half_g])
    pos_out = np.concatenate([all_pos, m_pos])
    mask_out = np.concatenate([all_mask, all_mask[off_w]])
    rgeo_out = np.concatenate([all_rgeo, mirror_w[all_rgeo[off_r]]])
    rng_out = np.concatenate([all_rng, ng_base + all_rng[off_r]])
    uv_out = None if all_uv is None else np.concatenate([all_uv, all_uv[off_r]])

    # Compact: only what the triangles use, in order.
    keep_w = np.unique(tri_out)
    remap_w = np.full(len(pos_out), -1, dtype=np.int64)
    remap_w[keep_w] = np.arange(len(keep_w))
    keep_r = np.unique(trv_out)
    remap_r = np.full(len(rgeo_out), -1, dtype=np.int64)
    remap_r[keep_r] = np.arange(len(keep_r))
    _, rng_compact = np.unique(rng_out[keep_r], return_inverse=True)
    sm._replace(
        pos_out[keep_w],
        remap_w[tri_out],
        groups_out,
        mask_out[keep_w],
        normals=sm._rnorm is not None,
        trv=remap_r[trv_out],
        rgeo=remap_w[rgeo_out[keep_r]],
        ruv=None if uv_out is None else uv_out[keep_r],
        rng=rng_compact.reshape(-1),
    )
    return {
        "mode": "mirror",
        "cut_triangles": int(len(rows)),
        "removed_triangles": int((~keep & ~cut).sum()),
        "mirrored_triangles": int(len(half_w)),
        "triangles": int(len(tri_out)),
        "vertices": int(len(keep_w)),
    }
