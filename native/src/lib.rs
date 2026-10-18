//! `aether3d_native`: native kernels for Aether3D (see `native/README.md`).
//!
//! Each function computes exactly what an Aether3D NumPy function computes -- the same
//! floating-point operations in the same order, never fused into multiply-adds -- so the
//! results are bitwise identical; Aether3D's tests compare them. Each function checks its
//! arrays (types, contiguity, sizes, indices) and raises `ValueError` instead of reading
//! out of bounds, and releases the GIL while it computes. Aether3D calls them through
//! `aether3d._native`, which falls back to NumPy when this module is missing or switched
//! off (`AETHER3D_NATIVE=0`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod buffers;
mod dqs;
mod earcut;
mod hierarchy;
mod hydro;
mod ldl;
mod pack;
mod png;
mod skin;
mod traversal;

use buffers::Arrays;

/// The calling convention between `aether3d._native` and this module: a module whose
/// `ABI` differs from what Aether3D expects is not used.
const ABI: u32 = 6;

fn check(ok: bool, message: impl FnOnce() -> String) -> PyResult<()> {
    if ok {
        Ok(())
    } else {
        Err(PyValueError::new_err(message()))
    }
}

/// Trace rays through a SceneAccel's arrays; hits are written to `t`, `inst`, `prim`, `b`
/// (only for rays that hit something). Returns `(iterations, node_visits, triangle_tests)`.
#[pyfunction]
#[pyo3(signature = (
    child, child_lo, child_hi, kind, first, count, v0, v1, v2, tri_prim, slot_inst,
    inst_lo, inst_hi, inst_root, to_object, root_lo, root_hi, stack_size,
    o, d, tmin, tmax, skip_inst, skip_tri, t, inst, prim, b, any_hit, batch, threads
))]
#[allow(clippy::too_many_arguments)]
fn trace(
    py: Python<'_>,
    child: &Bound<'_, PyAny>,
    child_lo: &Bound<'_, PyAny>,
    child_hi: &Bound<'_, PyAny>,
    kind: &Bound<'_, PyAny>,
    first: &Bound<'_, PyAny>,
    count: &Bound<'_, PyAny>,
    v0: &Bound<'_, PyAny>,
    v1: &Bound<'_, PyAny>,
    v2: &Bound<'_, PyAny>,
    tri_prim: &Bound<'_, PyAny>,
    slot_inst: &Bound<'_, PyAny>,
    inst_lo: &Bound<'_, PyAny>,
    inst_hi: &Bound<'_, PyAny>,
    inst_root: &Bound<'_, PyAny>,
    to_object: &Bound<'_, PyAny>,
    root_lo: &Bound<'_, PyAny>,
    root_hi: &Bound<'_, PyAny>,
    stack_size: usize,
    o: &Bound<'_, PyAny>,
    d: &Bound<'_, PyAny>,
    tmin: &Bound<'_, PyAny>,
    tmax: &Bound<'_, PyAny>,
    skip_inst: Option<&Bound<'_, PyAny>>,
    skip_tri: Option<&Bound<'_, PyAny>>,
    t: &Bound<'_, PyAny>,
    inst: &Bound<'_, PyAny>,
    prim: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    any_hit: bool,
    batch: usize,
    threads: usize,
) -> PyResult<(u64, u64, u64)> {
    let mut arrays = Arrays::new();
    let kind_a = arrays.input::<i8>(kind, "kind", None)?;
    let nodes = unsafe { kind_a.slice() }.len();
    let child_a = arrays.input::<i64>(child, "child", Some(2 * nodes))?;
    let child_lo_a = arrays.input::<f32>(child_lo, "child_lo", Some(6 * nodes))?;
    let child_hi_a = arrays.input::<f32>(child_hi, "child_hi", Some(6 * nodes))?;
    let first_a = arrays.input::<i64>(first, "first", Some(nodes))?;
    let count_a = arrays.input::<i64>(count, "count", Some(nodes))?;
    let v0_a = arrays.input::<f32>(v0, "v0", None)?;
    let tris = unsafe { v0_a.slice() }.len() / 3;
    let v1_a = arrays.input::<f32>(v1, "v1", Some(3 * tris))?;
    let v2_a = arrays.input::<f32>(v2, "v2", Some(3 * tris))?;
    check(unsafe { v0_a.slice() }.len() == 3 * tris, || {
        "v0: not (T, 3)".into()
    })?;
    let tri_prim_a = arrays.input::<i64>(tri_prim, "tri_prim", Some(tris))?;
    let slot_inst_a = arrays.input::<i64>(slot_inst, "slot_inst", None)?;
    let inst_root_a = arrays.input::<i64>(inst_root, "inst_root", None)?;
    let instances = unsafe { inst_root_a.slice() }.len();
    let inst_lo_a = arrays.input::<f64>(inst_lo, "inst_lo", Some(3 * instances))?;
    let inst_hi_a = arrays.input::<f64>(inst_hi, "inst_hi", Some(3 * instances))?;
    let to_object_a = arrays.input::<f64>(to_object, "to_object", Some(12 * instances))?;
    let root_lo_a = arrays.input::<f64>(root_lo, "root_lo", Some(3))?;
    let root_hi_a = arrays.input::<f64>(root_hi, "root_hi", Some(3))?;
    let tmin_a = arrays.input::<f64>(tmin, "tmin", None)?;
    let n = unsafe { tmin_a.slice() }.len();
    let tmax_a = arrays.input::<f64>(tmax, "tmax", Some(n))?;
    let o_a = arrays.input::<f64>(o, "o", Some(3 * n))?;
    let d_a = arrays.input::<f64>(d, "d", Some(3 * n))?;
    let skip = match (skip_inst, skip_tri) {
        (Some(si), Some(st)) => Some((
            arrays.input::<i64>(si, "skip_inst", Some(n))?,
            arrays.input::<i64>(st, "skip_tri", Some(n))?,
        )),
        (None, None) => None,
        _ => return Err(PyValueError::new_err("skip_inst and skip_tri go together")),
    };
    let t_a = arrays.output::<f64>(t, "t", Some(n))?;
    let inst_a = arrays.output::<i64>(inst, "inst", Some(n))?;
    let prim_a = arrays.output::<i64>(prim, "prim", Some(n))?;
    let b_a = arrays.output::<f64>(b, "b", Some(3 * n))?;
    arrays.check_overlaps()?;
    check(batch > 0, || "batch must be positive".into())?;

    // Every index the traversal follows must be in range (it would raise otherwise, but
    // checking first keeps the threads' work simple and the error clear).
    let sc = unsafe {
        traversal::Accel {
            child: child_a.slice(),
            child_lo: child_lo_a.slice(),
            child_hi: child_hi_a.slice(),
            kind: kind_a.slice(),
            first: first_a.slice(),
            count: count_a.slice(),
            v0: v0_a.slice(),
            v1: v1_a.slice(),
            v2: v2_a.slice(),
            tri_prim: tri_prim_a.slice(),
            slot_inst: slot_inst_a.slice(),
            inst_lo: inst_lo_a.slice(),
            inst_hi: inst_hi_a.slice(),
            inst_root: inst_root_a.slice(),
            to_object: to_object_a.slice(),
            root_lo: [
                root_lo_a.slice()[0],
                root_lo_a.slice()[1],
                root_lo_a.slice()[2],
            ],
            root_hi: [
                root_hi_a.slice()[0],
                root_hi_a.slice()[1],
                root_hi_a.slice()[2],
            ],
            stack_size,
        }
    };
    validate(&sc)?;
    let rays = unsafe {
        traversal::Rays {
            o: o_a.slice(),
            d: d_a.slice(),
            tmin: tmin_a.slice(),
            tmax: tmax_a.slice(),
            skip: skip.as_ref().map(|(si, st)| (si.slice(), st.slice())),
        }
    };
    let hits = traversal::Hits {
        t: &t_a,
        inst: &inst_a,
        prim: &prim_a,
        b: &b_a,
    };
    let stats = py.detach(|| traversal::trace(&sc, &rays, &hits, any_hit, batch, threads));
    drop(arrays);
    Ok((stats.iterations, stats.node_visits, stats.triangle_tests))
}

/// The indices of an accel are in range: node ids, leaf ranges, slots, instance roots.
fn validate(sc: &traversal::Accel) -> PyResult<()> {
    let nodes = sc.kind.len();
    let tris = sc.tri_prim.len();
    let slots = sc.slot_inst.len();
    let instances = sc.inst_root.len();
    for (k, &kind) in sc.kind.iter().enumerate() {
        match kind {
            traversal::KIND_INNER => {
                for c in 0..2 {
                    let id = sc.child[2 * k + c];
                    check(id >= 0 && (id as usize) < nodes, || {
                        format!("node {k}: child {id} out of range")
                    })?;
                }
            }
            traversal::KIND_TRIANGLES | traversal::KIND_INSTANCES => {
                let (f, c) = (sc.first[k], sc.count[k]);
                let limit = if kind == traversal::KIND_TRIANGLES {
                    tris
                } else {
                    slots
                };
                check(f >= 0 && c >= 0 && (f + c) as usize <= limit, || {
                    format!("node {k}: leaf range out of range")
                })?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "node {k}: unknown kind {kind}"
                )))
            }
        }
    }
    for &s in sc.slot_inst {
        check(s >= 0 && (s as usize) < instances, || {
            format!("slot instance {s} out of range")
        })?;
    }
    for &root in sc.inst_root {
        check(root >= 0 && (root as usize) < nodes, || {
            format!("instance root {root} out of range")
        })?;
    }
    check(instances < (1 << 31) && nodes < (1 << 32), || {
        "too many nodes or instances".into()
    })?;
    Ok(())
}

/// `_local_matrices`: 4x4 matrices `out` (n, 4, 4) from `position` (n, 3), `rotation`
/// (n, 4) and `scale` (n, 3).
#[pyfunction]
fn local_matrices(
    py: Python<'_>,
    position: &Bound<'_, PyAny>,
    rotation: &Bound<'_, PyAny>,
    scale: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let p = arrays.input::<f64>(position, "position", None)?;
    let n = unsafe { p.slice() }.len() / 3;
    check(unsafe { p.slice() }.len() == 3 * n, || {
        "position: not (n, 3)".into()
    })?;
    let r = arrays.input::<f64>(rotation, "rotation", Some(4 * n))?;
    let s = arrays.input::<f64>(scale, "scale", Some(3 * n))?;
    let m = arrays.output::<f64>(out, "out", Some(16 * n))?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe { hierarchy::local_matrices(p.slice(), r.slice(), s.slice(), m.slice()) });
    drop(arrays);
    Ok(())
}

/// The world matrices `out` (n, 4, 4) of objects in depth-first order, from their local
/// matrices (n, 4, 4) and parent rows (n,; -1 for roots, otherwise an earlier row).
#[pyfunction]
fn world_matrices(
    py: Python<'_>,
    local: &Bound<'_, PyAny>,
    parent: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let p = arrays.input::<i64>(parent, "parent", None)?;
    let n = unsafe { p.slice() }.len();
    let l = arrays.input::<f64>(local, "local", Some(16 * n))?;
    let w = arrays.output::<f64>(out, "out", Some(16 * n))?;
    arrays.check_overlaps()?;
    for (k, &q) in unsafe { p.slice() }.iter().enumerate() {
        check(q < k as i64 && q >= -1, || {
            format!("parent of row {k} is {q}: not an earlier row")
        })?;
    }
    py.detach(|| unsafe { hierarchy::world_matrices(l.slice(), p.slice(), w.slice()) });
    drop(arrays);
    Ok(())
}

/// `_filter_rows`: the filtered scanlines `out` (height, 1 + stride) of `rows`
/// (height, stride) with `bpp` bytes per pixel.
#[pyfunction]
fn png_filter(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    height: usize,
    stride: usize,
    bpp: usize,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    check(bpp >= 1, || "bpp must be positive".into())?;
    let mut arrays = Arrays::new();
    let r = arrays.input::<u8>(rows, "rows", Some(height * stride))?;
    let o = arrays.output::<u8>(out, "out", Some(height * (stride + 1)))?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe { png::filter_rows(r.slice(), height, stride, bpp, o.slice()) });
    drop(arrays);
    Ok(())
}

/// `_unfilter`: the scanlines `out` (height, stride) reconstructed from `block`
/// (height, 1 + stride: filter type, then the filtered bytes).
#[pyfunction]
fn png_unfilter(
    py: Python<'_>,
    block: &Bound<'_, PyAny>,
    height: usize,
    stride: usize,
    bpp: usize,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    check(bpp >= 1, || "bpp must be positive".into())?;
    let mut arrays = Arrays::new();
    let blk = arrays.input::<u8>(block, "block", Some(height * (stride + 1)))?;
    let o = arrays.output::<u8>(out, "out", Some(height * stride))?;
    arrays.check_overlaps()?;
    for r in 0..height {
        let kind = unsafe { blk.slice() }[r * (stride + 1)];
        check(kind <= 4, || format!("invalid PNG filter type {kind}"))?;
    }
    py.detach(|| unsafe { png::unfilter(blk.slice(), height, stride, bpp, o.slice()) });
    drop(arrays);
    Ok(())
}

fn to_err(r: Result<(), String>) -> PyResult<()> {
    r.map_err(PyValueError::new_err)
}

/// `sparse.analyze`, first half: the elimination tree `parent` (n,) and the column counts
/// `lnz` (n,) of `L` for the upper triangle `ap`, `ai` (CSC, rows <= k in column k).
#[pyfunction]
fn ldl_etree(
    py: Python<'_>,
    ap: &Bound<'_, PyAny>,
    ai: &Bound<'_, PyAny>,
    parent: &Bound<'_, PyAny>,
    lnz: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let ap_a = arrays.input::<i64>(ap, "ap", None)?;
    let n = unsafe { ap_a.slice() }.len().saturating_sub(1);
    let ai_a = arrays.input::<i64>(ai, "ai", None)?;
    let parent_a = arrays.output::<i64>(parent, "parent", Some(n))?;
    let lnz_a = arrays.output::<i64>(lnz, "lnz", Some(n))?;
    arrays.check_overlaps()?;
    to_err(ldl::check_upper(n, unsafe { ap_a.slice() }, unsafe {
        ai_a.slice()
    }))?;
    py.detach(|| unsafe {
        ldl::etree(
            n,
            ap_a.slice(),
            ai_a.slice(),
            parent_a.slice(),
            lnz_a.slice(),
        )
    });
    drop(arrays);
    Ok(())
}

/// `sparse.analyze`, second half: the ascending row indices `li` of every column of `L`.
#[pyfunction]
fn ldl_pattern(
    py: Python<'_>,
    ap: &Bound<'_, PyAny>,
    ai: &Bound<'_, PyAny>,
    parent: &Bound<'_, PyAny>,
    lp: &Bound<'_, PyAny>,
    li: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let ap_a = arrays.input::<i64>(ap, "ap", None)?;
    let n = unsafe { ap_a.slice() }.len().saturating_sub(1);
    let ai_a = arrays.input::<i64>(ai, "ai", None)?;
    let parent_a = arrays.input::<i64>(parent, "parent", Some(n))?;
    let lp_a = arrays.input::<i64>(lp, "lp", Some(n + 1))?;
    let li_a = arrays.output::<i64>(li, "li", None)?;
    arrays.check_overlaps()?;
    to_err(ldl::check_upper(n, unsafe { ap_a.slice() }, unsafe {
        ai_a.slice()
    }))?;
    // The counts must be those of the tree: recompute and compare.
    let mut par = vec![0i64; n];
    let mut cnt = vec![0i64; n];
    ldl::etree(
        n,
        unsafe { ap_a.slice() },
        unsafe { ai_a.slice() },
        &mut par,
        &mut cnt,
    );
    check(par.as_slice() == unsafe { parent_a.slice() }, || {
        "parent: not the elimination tree of the matrix".into()
    })?;
    let lp_s = unsafe { lp_a.slice() };
    check(lp_s[0] == 0, || "lp: must start at 0".into())?;
    for k in 0..n {
        check(lp_s[k + 1] - lp_s[k] == cnt[k], || {
            format!("lp: column {k} does not have its count")
        })?;
    }
    check(lp_s[n] as usize == unsafe { li_a.slice() }.len(), || {
        "li: length is not lp[n]".into()
    })?;
    py.detach(|| unsafe {
        ldl::pattern(
            n,
            ap_a.slice(),
            ai_a.slice(),
            parent_a.slice(),
            lp_s,
            li_a.slice(),
        )
    });
    drop(arrays);
    Ok(())
}

/// Checks that `lp/li` (columns of L) and `rp/rj/rpos` (its rows) describe the same
/// strictly lower triangular structure.
fn check_factor_structure(
    n: usize,
    lp: &[i64],
    li: &[i64],
    rp: &[i64],
    rj: &[i64],
    rpos: &[i64],
) -> PyResult<()> {
    to_err(ldl::check_segments("li", n, lp, li, |k| {
        (k as i64 + 1, n as i64)
    }))?;
    to_err(ldl::check_segments("rj", n, rp, rj, |k| (0, k as i64)))?;
    check(rpos.len() == rj.len() && rj.len() == li.len(), || {
        "rpos, rj and li must have the same length".into()
    })?;
    for k in 0..n {
        for q in rp[k] as usize..rp[k + 1] as usize {
            let col = rj[q] as usize;
            let pos = rpos[q];
            check(
                pos >= lp[col] && pos < lp[col + 1] && li[pos as usize] == k as i64,
                || format!("rpos: entry {q} does not point at row {k} of column {col}"),
            )?;
            if q > rp[k] as usize {
                check(rj[q] > rj[q - 1], || {
                    format!("rj: row {k} is not ascending")
                })?;
            }
        }
        for p in lp[k] as usize..lp[k + 1] as usize {
            if p > lp[k] as usize {
                check(li[p] > li[p - 1], || {
                    format!("li: column {k} is not ascending")
                })?;
            }
        }
    }
    Ok(())
}

/// `sparse._factor_numpy`: `lx` and `d` of `P A P^T = L D L^T`. Returns the first row with
/// a pivot that is not positive, or -1.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ldl_factor(
    py: Python<'_>,
    ap: &Bound<'_, PyAny>,
    ai: &Bound<'_, PyAny>,
    ax: &Bound<'_, PyAny>,
    lp: &Bound<'_, PyAny>,
    li: &Bound<'_, PyAny>,
    rp: &Bound<'_, PyAny>,
    rj: &Bound<'_, PyAny>,
    rpos: &Bound<'_, PyAny>,
    lx: &Bound<'_, PyAny>,
    d: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let mut arrays = Arrays::new();
    let ap_a = arrays.input::<i64>(ap, "ap", None)?;
    let n = unsafe { ap_a.slice() }.len().saturating_sub(1);
    let ai_a = arrays.input::<i64>(ai, "ai", None)?;
    let ax_a = arrays.input::<f64>(ax, "ax", Some(unsafe { ai_a.slice() }.len()))?;
    let lp_a = arrays.input::<i64>(lp, "lp", Some(n + 1))?;
    let li_a = arrays.input::<i64>(li, "li", None)?;
    let rp_a = arrays.input::<i64>(rp, "rp", Some(n + 1))?;
    let rj_a = arrays.input::<i64>(rj, "rj", None)?;
    let rpos_a = arrays.input::<i64>(rpos, "rpos", None)?;
    let lx_a = arrays.output::<f64>(lx, "lx", Some(unsafe { li_a.slice() }.len()))?;
    let d_a = arrays.output::<f64>(d, "d", Some(n))?;
    arrays.check_overlaps()?;
    to_err(ldl::check_upper(n, unsafe { ap_a.slice() }, unsafe {
        ai_a.slice()
    }))?;
    unsafe {
        check_factor_structure(
            n,
            lp_a.slice(),
            li_a.slice(),
            rp_a.slice(),
            rj_a.slice(),
            rpos_a.slice(),
        )?;
    }
    let bad = py.detach(|| unsafe {
        ldl::factor(
            n,
            ap_a.slice(),
            ai_a.slice(),
            ax_a.slice(),
            lp_a.slice(),
            li_a.slice(),
            rp_a.slice(),
            rj_a.slice(),
            lx_a.slice(),
            d_a.slice(),
        )
    });
    drop(arrays);
    Ok(bad)
}

/// `sparse._solve_numpy`: solves `L D L^T x = b` in place (`x`: n rows of m columns).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ldl_solve(
    py: Python<'_>,
    lp: &Bound<'_, PyAny>,
    li: &Bound<'_, PyAny>,
    lx: &Bound<'_, PyAny>,
    rp: &Bound<'_, PyAny>,
    rj: &Bound<'_, PyAny>,
    rpos: &Bound<'_, PyAny>,
    d: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let lp_a = arrays.input::<i64>(lp, "lp", None)?;
    let n = unsafe { lp_a.slice() }.len().saturating_sub(1);
    let li_a = arrays.input::<i64>(li, "li", None)?;
    let lx_a = arrays.input::<f64>(lx, "lx", Some(unsafe { li_a.slice() }.len()))?;
    let rp_a = arrays.input::<i64>(rp, "rp", Some(n + 1))?;
    let rj_a = arrays.input::<i64>(rj, "rj", None)?;
    let rpos_a = arrays.input::<i64>(rpos, "rpos", None)?;
    let d_a = arrays.input::<f64>(d, "d", Some(n))?;
    let x_a = arrays.output::<f64>(x, "x", None)?;
    arrays.check_overlaps()?;
    let total = unsafe { x_a.slice() }.len();
    check(n == 0 || total % n == 0, || "x: not n rows".into())?;
    let m = total.checked_div(n).unwrap_or(0);
    unsafe {
        check_factor_structure(
            n,
            lp_a.slice(),
            li_a.slice(),
            rp_a.slice(),
            rj_a.slice(),
            rpos_a.slice(),
        )?;
    }
    py.detach(|| unsafe {
        ldl::solve(
            n,
            m,
            lp_a.slice(),
            li_a.slice(),
            lx_a.slice(),
            rp_a.slice(),
            rj_a.slice(),
            rpos_a.slice(),
            d_a.slice(),
            x_a.slice(),
        )
    });
    drop(arrays);
    Ok(())
}

/// `pack._place_numpy`: horizon placement. `orient` (count,) and `pos` (count, 2) receive
/// each chart's orientation and texel offset; returns the peak.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pack_horizon(
    py: Python<'_>,
    width: usize,
    bot: &Bound<'_, PyAny>,
    top: &Bound<'_, PyAny>,
    offsets: &Bound<'_, PyAny>,
    order: &Bound<'_, PyAny>,
    orient: &Bound<'_, PyAny>,
    pos: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let mut arrays = Arrays::new();
    let bot_a = arrays.input::<i64>(bot, "bot", None)?;
    let top_a = arrays.input::<i64>(top, "top", None)?;
    let off_a = arrays.input::<i64>(offsets, "offsets", None)?;
    let order_a = arrays.input::<i64>(order, "order", None)?;
    let count = to_err_count(pack::check(
        unsafe { bot_a.slice() },
        unsafe { top_a.slice() },
        unsafe { off_a.slice() },
        unsafe { order_a.slice() },
    ))?;
    let orient_a = arrays.output::<i64>(orient, "orient", Some(count))?;
    let pos_a = arrays.output::<i64>(pos, "pos", Some(2 * count))?;
    arrays.check_overlaps()?;
    let peak = py.detach(|| unsafe {
        pack::place(
            width,
            bot_a.slice(),
            top_a.slice(),
            off_a.slice(),
            order_a.slice(),
            orient_a.slice(),
            pos_a.slice(),
        )
    });
    drop(arrays);
    peak.ok_or_else(|| PyValueError::new_err("ATLAS_FULL: a chart is wider than the atlas"))
}

/// `cad.triangulate.triangulate_numpy`: ear clipping of many polygons with holes.
/// `points` (n, 2) float64, `ring_starts` and `polygon_starts` int64 tables; `tris`
/// (at least n + 2 rings, 3) and `owner` (as many rows) receive the triangles and their
/// polygons. Returns the triangle count.
#[pyfunction]
fn triangulate(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
    ring_starts: &Bound<'_, PyAny>,
    polygon_starts: &Bound<'_, PyAny>,
    tris: &Bound<'_, PyAny>,
    owner: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let mut arrays = Arrays::new();
    let p = arrays.input::<f64>(points, "points", None)?;
    let n = unsafe { p.slice() }.len() / 2;
    check(unsafe { p.slice() }.len() == 2 * n, || {
        "points: not (n, 2)".into()
    })?;
    check(unsafe { p.slice() }.iter().all(|v| v.is_finite()), || {
        "points must be finite".into()
    })?;
    let r = arrays.input::<i64>(ring_starts, "ring_starts", None)?;
    let q = arrays.input::<i64>(polygon_starts, "polygon_starts", None)?;
    let cap = to_err_count(earcut::check(n, unsafe { r.slice() }, unsafe { q.slice() }))?;
    let t = arrays.output::<i64>(tris, "tris", None)?;
    let o = arrays.output::<i64>(owner, "owner", None)?;
    check(
        unsafe { t.slice() }.len() >= 3 * cap && unsafe { o.slice() }.len() >= cap,
        || format!("tris and owner need room for {cap} triangles"),
    )?;
    arrays.check_overlaps()?;
    let count = py.detach(|| unsafe {
        earcut::triangulate(p.slice(), r.slice(), q.slice(), t.slice(), o.slice())
    });
    drop(arrays);
    count.ok_or_else(|| PyValueError::new_err("the triangle buffers were too small"))
}

/// Linear blend skinning (`animation.rig.skin_linear`): `out_p` (n, 3) float32 from
/// `pos` (n, 3) float32, `idx` (n, k) int32 joint indices, `w` (n, k) float64 weights and
/// `pal` (m, 3, 4) float64 skinning matrices; with `nrm`, also `out_n` (n, 3).
#[pyfunction]
#[pyo3(signature = (pos, nrm, idx, w, pal, out_p, out_n, threads=1))]
#[allow(clippy::too_many_arguments)]
fn skin_linear(
    py: Python<'_>,
    pos: &Bound<'_, PyAny>,
    nrm: Option<&Bound<'_, PyAny>>,
    idx: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    pal: &Bound<'_, PyAny>,
    out_p: &Bound<'_, PyAny>,
    out_n: Option<&Bound<'_, PyAny>>,
    threads: usize,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let p = arrays.input::<f32>(pos, "pos", None)?;
    let n = unsafe { p.slice() }.len() / 3;
    check(unsafe { p.slice() }.len() == 3 * n, || {
        "pos: not (n, 3)".into()
    })?;
    let ix = arrays.input::<i32>(idx, "idx", None)?;
    let len = unsafe { ix.slice() }.len();
    let k = len.checked_div(n).unwrap_or(1);
    check(n * k == len && (n > 0 || len == 0), || {
        "idx: not (n, k)".into()
    })?;
    check((1..=8).contains(&k), || {
        format!("idx: {k} influences per vertex (1..8)")
    })?;
    let wt = arrays.input::<f64>(w, "w", Some(n * k))?;
    let pl = arrays.input::<f64>(pal, "pal", None)?;
    let m = unsafe { pl.slice() }.len() / 12;
    check(unsafe { pl.slice() }.len() == 12 * m, || {
        "pal: not (m, 3, 4)".into()
    })?;
    for (e, &j) in unsafe { ix.slice() }.iter().enumerate() {
        check(j >= 0 && (j as usize) < m, || {
            format!("idx[{e}] = {j}: not a joint of the palette ({m})")
        })?;
    }
    let op = arrays.output::<f32>(out_p, "out_p", Some(3 * n))?;
    let nr = match nrm {
        Some(obj) => Some(arrays.input::<f32>(obj, "nrm", Some(3 * n))?),
        None => None,
    };
    let on = match out_n {
        Some(obj) => Some(arrays.output::<f32>(obj, "out_n", Some(3 * n))?),
        None => None,
    };
    check(nr.is_some() == on.is_some(), || {
        "nrm and out_n go together".into()
    })?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe {
        skin::skin_linear(
            p.slice(),
            nr.as_ref().map(|a| a.slice() as &[f32]),
            ix.slice(),
            wt.slice(),
            pl.slice(),
            k,
            op.slice(),
            on.as_ref().map(|a| a.slice()),
            threads,
        )
    });
    drop(arrays);
    Ok(())
}

/// Dual quaternion skinning (`animation.rig.skin_dual_quaternion`): `out_p` (n, 3)
/// float32 from `pos` (n, 3) float32, `idx` (n, k) int32, `w` (n, k) float64 and per joint
/// `q`, `d` (m, 4) and `kmat` (m, 9) float64; with `pal` (m, 12) and `blend` (n), mixed
/// with linear blend skinning per vertex; with `nrm`, also `out_n` (n, 3).
#[pyfunction]
#[pyo3(signature = (pos, nrm, idx, w, q, d, kmat, pal, blend, out_p, out_n, threads=1))]
#[allow(clippy::too_many_arguments)]
fn skin_dual_quaternion(
    py: Python<'_>,
    pos: &Bound<'_, PyAny>,
    nrm: Option<&Bound<'_, PyAny>>,
    idx: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    d: &Bound<'_, PyAny>,
    kmat: &Bound<'_, PyAny>,
    pal: Option<&Bound<'_, PyAny>>,
    blend: Option<&Bound<'_, PyAny>>,
    out_p: &Bound<'_, PyAny>,
    out_n: Option<&Bound<'_, PyAny>>,
    threads: usize,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let p = arrays.input::<f32>(pos, "pos", None)?;
    let n = unsafe { p.slice() }.len() / 3;
    check(unsafe { p.slice() }.len() == 3 * n, || {
        "pos: not (n, 3)".into()
    })?;
    let ix = arrays.input::<i32>(idx, "idx", None)?;
    let len = unsafe { ix.slice() }.len();
    let k = len.checked_div(n).unwrap_or(1);
    check(n * k == len && (n > 0 || len == 0), || {
        "idx: not (n, k)".into()
    })?;
    check((1..=8).contains(&k), || {
        format!("idx: {k} influences per vertex (1..8)")
    })?;
    let wt = arrays.input::<f64>(w, "w", Some(n * k))?;
    let qq = arrays.input::<f64>(q, "q", None)?;
    let m = unsafe { qq.slice() }.len() / 4;
    check(unsafe { qq.slice() }.len() == 4 * m, || {
        "q: not (m, 4)".into()
    })?;
    let dd = arrays.input::<f64>(d, "d", Some(4 * m))?;
    let km = arrays.input::<f64>(kmat, "kmat", Some(9 * m))?;
    for (e, &j) in unsafe { ix.slice() }.iter().enumerate() {
        check(j >= 0 && (j as usize) < m, || {
            format!("idx[{e}] = {j}: not a joint of the palette ({m})")
        })?;
    }
    let pl = match pal {
        Some(obj) => Some(arrays.input::<f64>(obj, "pal", Some(12 * m))?),
        None => None,
    };
    let bl = match blend {
        Some(obj) => Some(arrays.input::<f64>(obj, "blend", Some(n))?),
        None => None,
    };
    check(pl.is_some() == bl.is_some(), || {
        "pal and blend go together".into()
    })?;
    let op = arrays.output::<f32>(out_p, "out_p", Some(3 * n))?;
    let nr = match nrm {
        Some(obj) => Some(arrays.input::<f32>(obj, "nrm", Some(3 * n))?),
        None => None,
    };
    let on = match out_n {
        Some(obj) => Some(arrays.output::<f32>(obj, "out_n", Some(3 * n))?),
        None => None,
    };
    check(nr.is_some() == on.is_some(), || {
        "nrm and out_n go together".into()
    })?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe {
        dqs::skin_dual_quaternion(
            p.slice(),
            nr.as_ref().map(|a| a.slice() as &[f32]),
            ix.slice(),
            wt.slice(),
            qq.slice(),
            dd.slice(),
            km.slice(),
            pl.as_ref().map(|a| a.slice() as &[f64]),
            bl.as_ref().map(|a| a.slice() as &[f64]),
            k,
            op.slice(),
            on.as_ref().map(|a| a.slice()),
            threads,
        )
    });
    drop(arrays);
    Ok(())
}

/// Priority-Flood with an epsilon (`worlds.hydrology.priority_flood`): `out` (n) float64
/// from `h` (n) float64 and `outlet` (n) uint8 on an `nx` x `ny` grid.
#[pyfunction]
fn priority_flood(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    outlet: &Bound<'_, PyAny>,
    nx: usize,
    ny: usize,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let n = nx
        .checked_mul(ny)
        .ok_or_else(|| PyValueError::new_err("grid too large"))?;
    let mut arrays = Arrays::new();
    let hh = arrays.input::<f64>(h, "h", Some(n))?;
    let ol = arrays.input::<u8>(outlet, "outlet", Some(n))?;
    let o = arrays.output::<f64>(out, "out", Some(n))?;
    check(unsafe { hh.slice() }.iter().all(|v| v.is_finite()), || {
        "h must be finite".into()
    })?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe { hydro::priority_flood(hh.slice(), ol.slice(), nx, ny, o.slice()) });
    drop(arrays);
    Ok(())
}

fn check_indices(name: &str, values: &[i64], n: usize) -> PyResult<()> {
    for (e, &v) in values.iter().enumerate() {
        check(v >= 0 && (v as usize) < n, || {
            format!("{name}[{e}] = {v}: not a node index (< {n})")
        })?;
    }
    Ok(())
}

/// Drainage accumulation (`worlds.hydrology.accumulate`): in place on `area` (n) int64.
#[pyfunction]
fn accumulate(
    py: Python<'_>,
    order: &Bound<'_, PyAny>,
    recv: &Bound<'_, PyAny>,
    area: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let a = arrays.output::<i64>(area, "area", None)?;
    let n = unsafe { a.slice() }.len();
    let od = arrays.input::<i64>(order, "order", Some(n))?;
    let rv = arrays.input::<i64>(recv, "recv", Some(n))?;
    check_indices("order", unsafe { od.slice() }, n)?;
    check_indices("recv", unsafe { rv.slice() }, n)?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe { hydro::accumulate(od.slice(), rv.slice(), a.slice()) });
    drop(arrays);
    Ok(())
}

/// One implicit stream-power step (`worlds.hydrology.stream_power`): `out` (n) float64,
/// which the caller fills with `h` first.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn stream_power(
    py: Python<'_>,
    order: &Bound<'_, PyAny>,
    recv: &Bound<'_, PyAny>,
    dist: &Bound<'_, PyAny>,
    area: &Bound<'_, PyAny>,
    h: &Bound<'_, PyAny>,
    filled: &Bound<'_, PyAny>,
    kdt: f64,
    uplift: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let o = arrays.output::<f64>(out, "out", None)?;
    let n = unsafe { o.slice() }.len();
    let od = arrays.input::<i64>(order, "order", Some(n))?;
    let rv = arrays.input::<i64>(recv, "recv", Some(n))?;
    let ds = arrays.input::<f64>(dist, "dist", Some(n))?;
    let ar = arrays.input::<f64>(area, "area", Some(n))?;
    let hh = arrays.input::<f64>(h, "h", Some(n))?;
    let fl = arrays.input::<f64>(filled, "filled", Some(n))?;
    let up = arrays.input::<f64>(uplift, "uplift", Some(n))?;
    check_indices("order", unsafe { od.slice() }, n)?;
    check_indices("recv", unsafe { rv.slice() }, n)?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe {
        hydro::stream_power(
            od.slice(),
            rv.slice(),
            ds.slice(),
            ar.slice(),
            hh.slice(),
            fl.slice(),
            kdt,
            up.slice(),
            o.slice(),
        )
    });
    drop(arrays);
    Ok(())
}

/// Sediment transport (`worlds.hydrology.sediment`): in place on `after` (n) float64.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sediment(
    py: Python<'_>,
    order: &Bound<'_, PyAny>,
    recv: &Bound<'_, PyAny>,
    area: &Bound<'_, PyAny>,
    before: &Bound<'_, PyAny>,
    after: &Bound<'_, PyAny>,
    filled: &Bound<'_, PyAny>,
    sea: f64,
    cell: f64,
    deposition: f64,
    aggradation: f64,
) -> PyResult<()> {
    let mut arrays = Arrays::new();
    let af = arrays.output::<f64>(after, "after", None)?;
    let n = unsafe { af.slice() }.len();
    let od = arrays.input::<i64>(order, "order", Some(n))?;
    let rv = arrays.input::<i64>(recv, "recv", Some(n))?;
    let ar = arrays.input::<f64>(area, "area", Some(n))?;
    let bf = arrays.input::<f64>(before, "before", Some(n))?;
    let fl = arrays.input::<f64>(filled, "filled", Some(n))?;
    check_indices("order", unsafe { od.slice() }, n)?;
    check_indices("recv", unsafe { rv.slice() }, n)?;
    check(unsafe { ar.slice() }.iter().all(|&v| v > 0.0), || {
        "area must be positive".into()
    })?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe {
        hydro::sediment(
            od.slice(),
            rv.slice(),
            ar.slice(),
            bf.slice(),
            af.slice(),
            fl.slice(),
            sea,
            cell,
            deposition,
            aggradation,
        )
    });
    drop(arrays);
    Ok(())
}

/// Talus relaxation (`worlds.hydrology.thermal`): `out` (n) float64 from `h` (n) float64.
#[pyfunction]
#[pyo3(signature = (h, nx, ny, cell, talus_tan, rate, iterations, out, threads=1))]
#[allow(clippy::too_many_arguments)]
fn thermal(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    nx: usize,
    ny: usize,
    cell: f64,
    talus_tan: f64,
    rate: f64,
    iterations: usize,
    out: &Bound<'_, PyAny>,
    threads: usize,
) -> PyResult<()> {
    let n = nx
        .checked_mul(ny)
        .ok_or_else(|| PyValueError::new_err("grid too large"))?;
    let mut arrays = Arrays::new();
    let hh = arrays.input::<f64>(h, "h", Some(n))?;
    let o = arrays.output::<f64>(out, "out", Some(n))?;
    check(iterations <= 100_000, || {
        "iterations: at most 100000".into()
    })?;
    arrays.check_overlaps()?;
    py.detach(|| unsafe {
        hydro::thermal(
            hh.slice(),
            nx,
            ny,
            cell,
            talus_tan,
            rate,
            iterations,
            o.slice(),
            threads,
        )
    });
    drop(arrays);
    Ok(())
}

fn to_err_count(r: Result<usize, String>) -> PyResult<usize> {
    r.map_err(PyValueError::new_err)
}

#[pymodule]
fn aether3d_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI", ABI)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(local_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(world_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(png_filter, m)?)?;
    m.add_function(wrap_pyfunction!(png_unfilter, m)?)?;
    m.add_function(wrap_pyfunction!(ldl_etree, m)?)?;
    m.add_function(wrap_pyfunction!(ldl_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(ldl_factor, m)?)?;
    m.add_function(wrap_pyfunction!(ldl_solve, m)?)?;
    m.add_function(wrap_pyfunction!(pack_horizon, m)?)?;
    m.add_function(wrap_pyfunction!(triangulate, m)?)?;
    m.add_function(wrap_pyfunction!(skin_linear, m)?)?;
    m.add_function(wrap_pyfunction!(skin_dual_quaternion, m)?)?;
    m.add_function(wrap_pyfunction!(priority_flood, m)?)?;
    m.add_function(wrap_pyfunction!(accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(stream_power, m)?)?;
    m.add_function(wrap_pyfunction!(sediment, m)?)?;
    m.add_function(wrap_pyfunction!(thermal, m)?)?;
    Ok(())
}
