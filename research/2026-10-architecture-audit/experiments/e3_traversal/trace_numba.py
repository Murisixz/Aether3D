"""Scalar port of SceneAccel._trace (closest hit) in Numba (fastmath off)."""
import math
import numba
import numpy as np
from numba import njit, prange

MIN_DIR = 1e-300

@njit(inline="always")
def _setup(d, inv, shear):
    for i in range(3):
        v = d[i]
        safe = (-MIN_DIR if v < 0 else MIN_DIR) if abs(v) < MIN_DIR else v
        inv[i] = 1.0 / safe
    z = 0; m = abs(d[0])
    if abs(d[1]) > m:
        z = 1; m = abs(d[1])
    if abs(d[2]) > m:
        z = 2
    x = (z + 1) % 3; y = (x + 1) % 3
    dz = d[z]
    if dz < 0.0:
        x, y = y, x
    dzs = 1.0 if dz == 0.0 else dz
    shear[0] = d[x] / dzs; shear[1] = d[y] / dzs; shear[2] = 1.0 / dzs
    return x, y, z

@njit(inline="always")
def _slab(l0, l1, l2, h0, h1, h2, o, inv, tmin, tmax):
    a0 = (l0 - o[0]) * inv[0]; b0 = (h0 - o[0]) * inv[0]
    a1 = (l1 - o[1]) * inv[1]; b1 = (h1 - o[1]) * inv[1]
    a2 = (l2 - o[2]) * inv[2]; b2 = (h2 - o[2]) * inv[2]
    n0 = a0 if a0 < b0 else b0; f0 = a0 if a0 > b0 else b0
    n1 = a1 if a1 < b1 else b1; f1 = a1 if a1 > b1 else b1
    n2 = a2 if a2 < b2 else b2; f2 = a2 if a2 > b2 else b2
    x = n0 if n0 > n1 else n1; y = n2 if n2 > tmin else tmin
    near = x if x > y else y
    fa = f0 if f0 < f1 else f1
    far = (fa if fa < f2 else f2) * (1.0 + 4e-16)
    return (near <= far) and (near < tmax), near

@njit
def _trace_range(begin, end, o_all, d_all, tmin_all, tmax_all, skip_inst, skip_tri, has_skip,
                 child, child_lo, child_hi, kind, first, count, v0, v1, v2, tri_prim, slot_inst,
                 inst_lo, inst_hi, inst_root, to_object, root_lo, root_hi, out_t, out_inst, out_prim, out_b):
    stack = np.empty(256, np.int64); tstack = np.empty(256)
    o = np.empty(3); d = np.empty(3); inv = np.empty(3); shear = np.empty(3)
    a = np.empty(3); b = np.empty(3); c = np.empty(3)
    for r in range(begin, end):
        tmin = tmin_all[r]; t_best = tmax_all[r]
        hit_tri = -1; hit_inst = -1; h0 = 0.0; h1 = 0.0; h2 = 0.0
        finite = True
        for i in range(3):
            if not (math.isfinite(o_all[r, i]) and math.isfinite(d_all[r, i])):
                finite = False
            o[i] = o_all[r, i]; d[i] = d_all[r, i]
        kx, ky, kz = _setup(d, inv, shear)
        cur = -1; sp = 0; node = 0
        ok, t_node = _slab(root_lo[0], root_lo[1], root_lo[2], root_hi[0], root_hi[1], root_hi[2], o, inv, tmin, t_best)
        active = finite and root_lo[0] <= root_hi[0] and ok
        sk_i = skip_inst[r] if has_skip else -2
        sk_t = skip_tri[r] if has_skip else -2
        while active:
            need_pop = True
            if t_node < t_best:
                inst = (node >> 32) - 1; nd = node & 0xFFFFFFFF
                if inst != cur:
                    if inst < 0:
                        for i in range(3):
                            o[i] = o_all[r, i]; d[i] = d_all[r, i]
                    else:
                        for i in range(3):
                            o[i] = to_object[inst, i, 0] * o_all[r, 0] + to_object[inst, i, 1] * o_all[r, 1] + to_object[inst, i, 2] * o_all[r, 2] + to_object[inst, i, 3]
                            d[i] = to_object[inst, i, 0] * d_all[r, 0] + to_object[inst, i, 1] * d_all[r, 1] + to_object[inst, i, 2] * d_all[r, 2]
                    kx, ky, kz = _setup(d, inv, shear)
                    cur = inst
                k = kind[nd]
                if k == 0:
                    tag = (inst + 1) << 32
                    ok0, tn0 = _slab(np.float64(child_lo[nd, 0, 0]), np.float64(child_lo[nd, 0, 1]), np.float64(child_lo[nd, 0, 2]),
                                     np.float64(child_hi[nd, 0, 0]), np.float64(child_hi[nd, 0, 1]), np.float64(child_hi[nd, 0, 2]), o, inv, tmin, t_best)
                    ok1, tn1 = _slab(np.float64(child_lo[nd, 1, 0]), np.float64(child_lo[nd, 1, 1]), np.float64(child_lo[nd, 1, 2]),
                                     np.float64(child_hi[nd, 1, 0]), np.float64(child_hi[nd, 1, 1]), np.float64(child_hi[nd, 1, 2]), o, inv, tmin, t_best)
                    if ok0 or ok1:
                        near0 = (tn0 <= tn1) if (ok0 and ok1) else ok0
                        if near0:
                            node = tag | child[nd, 0]; t_node = tn0
                        else:
                            node = tag | child[nd, 1]; t_node = tn1
                        if ok0 and ok1:
                            if near0:
                                stack[sp] = tag | child[nd, 1]; tstack[sp] = tn1
                            else:
                                stack[sp] = tag | child[nd, 0]; tstack[sp] = tn0
                            sp += 1
                        need_pop = False
                elif k == 2:
                    f = first[nd]; cnt = count[nd]
                    for j in range(cnt):
                        ins = slot_inst[f + j]
                        ok, tn = _slab(inst_lo[ins, 0], inst_lo[ins, 1], inst_lo[ins, 2], inst_hi[ins, 0], inst_hi[ins, 1], inst_hi[ins, 2], o, inv, tmin, t_best)
                        if ok:
                            stack[sp] = ((ins + 1) << 32) | inst_root[ins]; tstack[sp] = tn; sp += 1
                else:
                    f = first[nd]; cnt = count[nd]
                    sx = shear[0]; sy = shear[1]; sz = shear[2]
                    for g in range(f, f + cnt):
                        for i in range(3):
                            a[i] = np.float64(v0[g, i]) - o[i]; b[i] = np.float64(v1[g, i]) - o[i]; c[i] = np.float64(v2[g, i]) - o[i]
                        az = a[kz]; bz = b[kz]; cz = c[kz]
                        ax = a[kx] - sx * az; ay = a[ky] - sy * az
                        bx = b[kx] - sx * bz; by = b[ky] - sy * bz
                        cx = c[kx] - sx * cz; cy = c[ky] - sy * cz
                        u = cx * by - cy * bx; v = ax * cy - ay * cx; w = bx * ay - by * ax
                        det = u + v + w
                        if ((u < 0.0 or v < 0.0 or w < 0.0) and (u > 0.0 or v > 0.0 or w > 0.0)) or det == 0.0:
                            continue
                        t = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / det
                        if not (t > tmin and t < t_best):
                            continue
                        if inst == sk_i and tri_prim[g] == sk_t:
                            continue
                        t_best = t; hit_tri = g; hit_inst = inst; h0 = u / det; h1 = v / det; h2 = w / det
            if need_pop:
                if sp == 0:
                    break
                sp -= 1; node = stack[sp]; t_node = tstack[sp]
        if hit_inst >= 0:
            out_t[r] = t_best; out_inst[r] = hit_inst; out_prim[r] = tri_prim[hit_tri]
            out_b[r, 0] = h0; out_b[r, 1] = h1; out_b[r, 2] = h2
        else:
            out_t[r] = np.inf; out_inst[r] = -1; out_prim[r] = -1; out_b[r, 0] = 0.0; out_b[r, 1] = 0.0; out_b[r, 2] = 0.0

@njit(parallel=True)
def _trace_parallel(n, *args):
    chunks = (n + 511) // 512
    for ch in prange(chunks):
        _trace_range(ch * 512, min(n, (ch + 1) * 512), *args)

def closest_hit(arrs, rb, outs, parallel=False):
    has = rb.skip_inst is not None
    dummy = np.zeros(1, np.int64)
    args = (rb.o, rb.d, rb.tmin, rb.tmax, rb.skip_inst if has else dummy, rb.skip_tri if has else dummy, has,
            arrs["child"], arrs["child_lo"], arrs["child_hi"], arrs["kind"], arrs["first"], arrs["count"],
            arrs["v0"], arrs["v1"], arrs["v2"], arrs["tri_prim"], arrs["slot_inst"], arrs["inst_lo"], arrs["inst_hi"],
            arrs["inst_root"], arrs["to_object"], arrs["root_lo"], arrs["root_hi"], *outs)
    if parallel:
        _trace_parallel(rb.n, *args)
    else:
        _trace_range(0, rb.n, *args)
