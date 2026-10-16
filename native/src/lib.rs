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
mod hierarchy;
mod png;
mod traversal;

use buffers::Arrays;

/// The calling convention between `aether3d._native` and this module: a module whose
/// `ABI` differs from what Aether3D expects is not used.
const ABI: u32 = 1;

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

#[pymodule]
fn aether3d_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI", ABI)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(local_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(world_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(png_filter, m)?)?;
    m.add_function(wrap_pyfunction!(png_unfilter, m)?)?;
    Ok(())
}
