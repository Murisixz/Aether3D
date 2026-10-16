//! Ray traversal of a `SceneAccel` (`aether3d/rendering/raytrace/accel.py`): closest hit
//! and any hit.
//!
//! This is `SceneAccel._trace`, one ray at a time instead of a NumPy batch at a time,
//! with the same arithmetic in the same order: the same slab tests (float32 boxes
//! promoted to float64), the same moves into an instance's object space, the same
//! watertight triangle test, the same visiting order (the nearer child first, the farther
//! pushed; a leaf's instances pushed in slot order) and the same choice among a leaf's
//! triangles (the nearest, the first on ties). The hits are therefore bitwise those of
//! the NumPy traversal, and so are its statistics: per batch of `batch` rays, the loop
//! iterations of the slowest ray (`iterations`), and over all rays the node visits and the
//! triangle tests.

use crate::buffers::Out;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const KIND_INNER: i8 = 0;
pub const KIND_TRIANGLES: i8 = 1;
pub const KIND_INSTANCES: i8 = 2;
const MIN_DIRECTION: f64 = 1e-300;

/// The arrays of a `SceneAccel`.
pub struct Accel<'a> {
    pub child: &'a [i64],    // (K, 2) global node ids
    pub child_lo: &'a [f32], // (K, 2, 3)
    pub child_hi: &'a [f32], // (K, 2, 3)
    pub kind: &'a [i8],      // (K,)
    pub first: &'a [i64],    // (K,)
    pub count: &'a [i64],    // (K,)
    pub v0: &'a [f32],       // (T, 3) object space, leaf order
    pub v1: &'a [f32],
    pub v2: &'a [f32],
    pub tri_prim: &'a [i64],  // (T,)
    pub slot_inst: &'a [i64], // (S,)
    pub inst_lo: &'a [f64],   // (I, 3)
    pub inst_hi: &'a [f64],   // (I, 3)
    pub inst_root: &'a [i64], // (I,)
    pub to_object: &'a [f64], // (I, 3, 4)
    pub root_lo: [f64; 3],
    pub root_hi: [f64; 3],
    pub stack_size: usize,
}

/// The rays and where their hits go.
pub struct Rays<'a> {
    pub o: &'a [f64],                         // (n, 3) world space
    pub d: &'a [f64],                         // (n, 3)
    pub tmin: &'a [f64],                      // (n,)
    pub tmax: &'a [f64],                      // (n,)
    pub skip: Option<(&'a [i64], &'a [i64])>, // (instance, mesh triangle) never hit
}

pub struct Hits<'a> {
    pub t: &'a Out<f64>,    // (n,)
    pub inst: &'a Out<i64>, // (n,)
    pub prim: &'a Out<i64>, // (n,) the mesh triangle
    pub b: &'a Out<f64>,    // (n, 3) barycentrics
}

#[derive(Default)]
pub struct Stats {
    pub iterations: u64,
    pub node_visits: u64,
    pub triangle_tests: u64,
}

// numpy.minimum / numpy.maximum: the second operand on ties, NaN propagated.
#[inline(always)]
fn mn(a: f64, b: f64) -> f64 {
    if a < b || a.is_nan() {
        a
    } else {
        b
    }
}
#[inline(always)]
fn mx(a: f64, b: f64) -> f64 {
    if a > b || a.is_nan() {
        a
    } else {
        b
    }
}

#[inline(always)]
fn f3(s: &[f32], k: usize) -> [f64; 3] {
    [s[k] as f64, s[k + 1] as f64, s[k + 2] as f64]
}
#[inline(always)]
fn d3(s: &[f64], k: usize) -> [f64; 3] {
    [s[k], s[k + 1], s[k + 2]]
}

/// A ray in one space (world or an instance's object space): `_ray_setup`.
#[derive(Clone, Copy)]
struct Space {
    o: [f64; 3],
    d: [f64; 3],
    inv: [f64; 3],
    shear: [f64; 3],
    kx: usize,
    ky: usize,
    kz: usize,
}

impl Space {
    fn new(o: [f64; 3], d: [f64; 3]) -> Space {
        let mut s = Space {
            o,
            d,
            inv: [0.0; 3],
            shear: [0.0; 3],
            kx: 0,
            ky: 0,
            kz: 0,
        };
        s.setup();
        s
    }

    fn setup(&mut self) {
        for i in 0..3 {
            let v = self.d[i];
            let safe = if v.abs() < MIN_DIRECTION {
                if v < 0.0 {
                    -MIN_DIRECTION
                } else {
                    MIN_DIRECTION
                }
            } else {
                v
            };
            self.inv[i] = 1.0 / safe;
        }
        // argmax of |d| (the first of equal ones)
        let mut kz = 0;
        let mut m = self.d[0].abs();
        if self.d[1].abs() > m {
            kz = 1;
            m = self.d[1].abs();
        }
        if self.d[2].abs() > m {
            kz = 2;
        }
        let mut kx = (kz + 1) % 3;
        let mut ky = (kx + 1) % 3;
        let dz = self.d[kz];
        if dz < 0.0 {
            std::mem::swap(&mut kx, &mut ky);
        }
        let dz_safe = if dz == 0.0 { 1.0 } else { dz };
        self.shear = [self.d[kx] / dz_safe, self.d[ky] / dz_safe, 1.0 / dz_safe];
        self.kx = kx;
        self.ky = ky;
        self.kz = kz;
    }

    /// `_box`: the entry distance (clamped to `tmin`) and whether the box is hit in range.
    #[inline(always)]
    fn slab(&self, lo: [f64; 3], hi: [f64; 3], tmin: f64, tmax: f64) -> (bool, f64) {
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        for i in 0..3 {
            let t1 = (lo[i] - self.o[i]) * self.inv[i];
            let t2 = (hi[i] - self.o[i]) * self.inv[i];
            a[i] = mn(t1, t2);
            b[i] = mx(t1, t2);
        }
        let near = mx(mx(a[0], a[1]), mx(a[2], tmin));
        let far = mn(mn(b[0], b[1]), b[2]) * (1.0 + 4e-16);
        (near <= far && near < tmax, near)
    }
}

/// `_triangles` for one pair: whether it intersects, `t` (not range-checked), barycentrics.
#[inline(always)]
fn triangle(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], s: &Space) -> (bool, f64, [f64; 3]) {
    let a = [p0[0] - s.o[0], p0[1] - s.o[1], p0[2] - s.o[2]];
    let b = [p1[0] - s.o[0], p1[1] - s.o[1], p1[2] - s.o[2]];
    let c = [p2[0] - s.o[0], p2[1] - s.o[1], p2[2] - s.o[2]];
    let (sx, sy, sz) = (s.shear[0], s.shear[1], s.shear[2]);
    let (az, bz, cz) = (a[s.kz], b[s.kz], c[s.kz]);
    let ax = a[s.kx] - sx * az;
    let ay = a[s.ky] - sy * az;
    let bx = b[s.kx] - sx * bz;
    let by = b[s.ky] - sy * bz;
    let cx = c[s.kx] - sx * cz;
    let cy = c[s.ky] - sy * cz;
    let u = cx * by - cy * bx;
    let v = ax * cy - ay * cx;
    let w = bx * ay - by * ax;
    let det = u + v + w;
    let mixed = (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0);
    if mixed || det == 0.0 {
        return (false, 0.0, [0.0; 3]);
    }
    let t = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / det;
    (true, t, [u / det, v / det, w / det])
}

/// The result of one ray: its hit (instance -1: none) and its statistics.
struct One {
    t: f64,
    inst: i64,
    tri: i64,
    b: [f64; 3],
    iterations: u64,
    triangle_tests: u64,
}

#[allow(clippy::too_many_arguments)]
fn trace_one(
    sc: &Accel,
    ow: [f64; 3],
    dw: [f64; 3],
    tmin: f64,
    tmax: f64,
    skip: Option<(i64, i64)>,
    any_hit: bool,
    stack: &mut [i64],
    tstack: &mut [f64],
) -> One {
    let mut out = One {
        t: tmax,
        inst: -1,
        tri: -1,
        b: [0.0; 3],
        iterations: 0,
        triangle_tests: 0,
    };
    let finite = ow.iter().chain(dw.iter()).all(|v| v.is_finite());
    let world = Space::new(ow, dw);
    let mut s = world;
    let mut cur: i64 = -1;
    let mut sp = 0usize;
    let mut node: i64 = 0; // the top-level root, in world space
    let mut t_node = 0.0;
    let mut active = false;
    if sc.root_lo[0] <= sc.root_hi[0] {
        let (ok, t) = s.slab(sc.root_lo, sc.root_hi, tmin, out.t);
        t_node = t;
        active = ok && finite;
    }
    if !active {
        return out;
    }
    loop {
        out.iterations += 1;
        let mut need_pop = true;
        if t_node < out.t {
            let inst = (node >> 32) - 1;
            let nd = (node & 0xFFFF_FFFF) as usize;
            if inst != cur {
                if inst < 0 {
                    s = world;
                } else {
                    let k = 12 * inst as usize;
                    let m = &sc.to_object[k..k + 12];
                    let mut o = [0.0; 3];
                    let mut d = [0.0; 3];
                    for i in 0..3 {
                        o[i] = m[4 * i] * ow[0]
                            + m[4 * i + 1] * ow[1]
                            + m[4 * i + 2] * ow[2]
                            + m[4 * i + 3];
                        d[i] = m[4 * i] * dw[0] + m[4 * i + 1] * dw[1] + m[4 * i + 2] * dw[2];
                    }
                    s = Space::new(o, d);
                }
                cur = inst;
            }
            let kind = sc.kind[nd];
            if kind == KIND_INNER {
                let tag = (inst + 1) << 32;
                let (ok0, tn0) = s.slab(
                    f3(sc.child_lo, 6 * nd),
                    f3(sc.child_hi, 6 * nd),
                    tmin,
                    out.t,
                );
                let (ok1, tn1) = s.slab(
                    f3(sc.child_lo, 6 * nd + 3),
                    f3(sc.child_hi, 6 * nd + 3),
                    tmin,
                    out.t,
                );
                if ok0 || ok1 {
                    let near0 = if ok0 && ok1 { tn0 <= tn1 } else { ok0 };
                    node = tag | sc.child[2 * nd + if near0 { 0 } else { 1 }];
                    t_node = if near0 { tn0 } else { tn1 };
                    if ok0 && ok1 {
                        stack[sp] = tag | sc.child[2 * nd + if near0 { 1 } else { 0 }];
                        tstack[sp] = if near0 { tn1 } else { tn0 };
                        sp += 1;
                    }
                    need_pop = false;
                }
            } else if kind == KIND_INSTANCES {
                let f = sc.first[nd] as usize;
                let c = sc.count[nd] as usize;
                for j in 0..c {
                    let ins = sc.slot_inst[f + j] as usize;
                    let (ok, tn) = s.slab(
                        d3(sc.inst_lo, 3 * ins),
                        d3(sc.inst_hi, 3 * ins),
                        tmin,
                        out.t,
                    );
                    if ok {
                        stack[sp] = ((ins as i64 + 1) << 32) | sc.inst_root[ins];
                        tstack[sp] = tn;
                        sp += 1;
                    }
                }
            } else {
                let f = sc.first[nd] as usize;
                let c = sc.count[nd] as usize;
                out.triangle_tests += c as u64;
                let mut found = false;
                for g in f..f + c {
                    let (ok, t, b) =
                        triangle(f3(sc.v0, 3 * g), f3(sc.v1, 3 * g), f3(sc.v2, 3 * g), &s);
                    if !ok || !(t > tmin && t < out.t) {
                        continue;
                    }
                    if let Some((si, st)) = skip {
                        if inst == si && sc.tri_prim[g] == st {
                            continue;
                        }
                    }
                    out.t = t;
                    out.tri = g as i64;
                    out.inst = inst;
                    out.b = b;
                    found = true;
                }
                if found && any_hit {
                    sp = 0;
                }
            }
        }
        if need_pop {
            if sp == 0 {
                break;
            }
            sp -= 1;
            node = stack[sp];
            t_node = tstack[sp];
        }
    }
    out
}

/// Trace every ray; hits go to `hits`. Rays are taken in chunks by `threads` threads (each
/// ray's result does not depend on which thread traced it, nor on the order).
pub fn trace(
    sc: &Accel,
    rays: &Rays,
    hits: &Hits,
    any_hit: bool,
    batch: usize,
    threads: usize,
) -> Stats {
    let n = rays.tmin.len();
    let batches = n.div_ceil(batch.max(1));
    let slowest: Vec<AtomicU64> = (0..batches).map(|_| AtomicU64::new(0)).collect();
    let visits = AtomicU64::new(0);
    let tests = AtomicU64::new(0);
    let next = AtomicUsize::new(0);
    const CHUNK: usize = 256;
    let work = || {
        let mut stack = vec![0i64; sc.stack_size.max(1)];
        let mut tstack = vec![0f64; sc.stack_size.max(1)];
        let (mut my_visits, mut my_tests) = (0u64, 0u64);
        loop {
            let start = next.fetch_add(CHUNK, Ordering::Relaxed);
            if start >= n {
                break;
            }
            for i in start..(start + CHUNK).min(n) {
                let skip = rays.skip.map(|(si, st)| (si[i], st[i]));
                let r = trace_one(
                    sc,
                    d3(rays.o, 3 * i),
                    d3(rays.d, 3 * i),
                    rays.tmin[i],
                    rays.tmax[i],
                    skip,
                    any_hit,
                    &mut stack,
                    &mut tstack,
                );
                // SAFETY: each ray's elements are written by exactly one thread.
                unsafe {
                    if r.inst >= 0 {
                        hits.t.set(i, r.t);
                        hits.inst.set(i, r.inst);
                        hits.prim.set(i, sc.tri_prim[r.tri as usize]);
                        for k in 0..3 {
                            hits.b.set(3 * i + k, r.b[k]);
                        }
                    }
                }
                my_visits += r.iterations;
                my_tests += r.triangle_tests;
                slowest[i / batch.max(1)].fetch_max(r.iterations, Ordering::Relaxed);
            }
        }
        visits.fetch_add(my_visits, Ordering::Relaxed);
        tests.fetch_add(my_tests, Ordering::Relaxed);
    };
    let threads = threads.max(1).min(n.div_ceil(CHUNK).max(1));
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(work); // (the closure only borrows: a copy per thread)
            }
        });
    }
    Stats {
        iterations: slowest.iter().map(|a| a.load(Ordering::Relaxed)).sum(),
        node_visits: visits.load(Ordering::Relaxed),
        triangle_tests: tests.load(Ordering::Relaxed),
    }
}
