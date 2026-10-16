//! Scalar port of SceneAccel._trace (closest hit), idiomatic slices (bounds-checked).
use std::sync::atomic::{AtomicI64, Ordering};

#[repr(C)]
pub struct Accel {
    child: *const i64, child_lo: *const f32, child_hi: *const f32, kind: *const i8,
    first: *const i64, count: *const i64, v0: *const f32, v1: *const f32, v2: *const f32,
    tri_prim: *const i64, slot_inst: *const i64, inst_lo: *const f64, inst_hi: *const f64,
    inst_root: *const i64, to_object: *const f64, root_lo: *const f64, root_hi: *const f64,
    stack_size: i64,
}
#[repr(C)]
pub struct Rays {
    n: i64, o: *const f64, d: *const f64, tmin: *const f64, tmax: *const f64,
    skip_inst: *const i64, skip_tri: *const i64, t: *mut f64, inst: *mut i64, prim: *mut i64, b: *mut f64,
}
const MIN_DIR: f64 = 1e-300;
const STACK_MAX: usize = 256;

struct Scene<'a> {
    child: &'a [i64], child_lo: &'a [f32], child_hi: &'a [f32], kind: &'a [i8], first: &'a [i64],
    count: &'a [i64], v0: &'a [f32], v1: &'a [f32], v2: &'a [f32], tri_prim: &'a [i64],
    slot_inst: &'a [i64], inst_lo: &'a [f64], inst_hi: &'a [f64], inst_root: &'a [i64],
    to_object: &'a [f64], root_lo: &'a [f64], root_hi: &'a [f64],
}
#[derive(Clone, Copy)]
struct Space { o: [f64; 3], d: [f64; 3], inv: [f64; 3], shear: [f64; 3], kx: usize, ky: usize, kz: usize }

#[inline(always)] fn mn(a: f64, b: f64) -> f64 { if a < b { a } else { b } }
#[inline(always)] fn mx(a: f64, b: f64) -> f64 { if a > b { a } else { b } }

impl Space {
    fn setup(&mut self) {
        for i in 0..3 {
            let v = self.d[i];
            let safe = if v.abs() < MIN_DIR { if v < 0.0 { -MIN_DIR } else { MIN_DIR } } else { v };
            self.inv[i] = 1.0 / safe;
        }
        let mut z = 0; let mut m = self.d[0].abs();
        if self.d[1].abs() > m { z = 1; m = self.d[1].abs(); }
        if self.d[2].abs() > m { z = 2; }
        let mut x = (z + 1) % 3; let mut y = (x + 1) % 3;
        let dz = self.d[z];
        if dz < 0.0 { std::mem::swap(&mut x, &mut y); }
        let dzs = if dz == 0.0 { 1.0 } else { dz };
        self.shear = [self.d[x] / dzs, self.d[y] / dzs, 1.0 / dzs];
        self.kx = x; self.ky = y; self.kz = z;
    }
    #[inline(always)]
    fn slab(&self, lo: [f64; 3], hi: [f64; 3], tmin: f64, tmax: f64) -> (bool, f64) {
        let mut a = [0.0; 3]; let mut b = [0.0; 3];
        for i in 0..3 {
            let t1 = (lo[i] - self.o[i]) * self.inv[i]; let t2 = (hi[i] - self.o[i]) * self.inv[i];
            a[i] = mn(t1, t2); b[i] = mx(t1, t2);
        }
        let near = mx(mx(a[0], a[1]), mx(a[2], tmin));
        let far = mn(mn(b[0], b[1]), b[2]) * (1.0 + 4e-16);
        (near <= far && near < tmax, near)
    }
}
#[inline(always)]
fn f3(s: &[f32], k: usize) -> [f64; 3] { [s[k] as f64, s[k + 1] as f64, s[k + 2] as f64] }
#[inline(always)]
fn d3(s: &[f64], k: usize) -> [f64; 3] { [s[k], s[k + 1], s[k + 2]] }

#[inline(always)]
fn triangle(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], s: &Space) -> (bool, f64, [f64; 3]) {
    let a = [p0[0] - s.o[0], p0[1] - s.o[1], p0[2] - s.o[2]];
    let b = [p1[0] - s.o[0], p1[1] - s.o[1], p1[2] - s.o[2]];
    let c = [p2[0] - s.o[0], p2[1] - s.o[1], p2[2] - s.o[2]];
    let (sx, sy, sz) = (s.shear[0], s.shear[1], s.shear[2]);
    let (az, bz, cz) = (a[s.kz], b[s.kz], c[s.kz]);
    let ax = a[s.kx] - sx * az; let ay = a[s.ky] - sy * az;
    let bx = b[s.kx] - sx * bz; let by = b[s.ky] - sy * bz;
    let cx = c[s.kx] - sx * cz; let cy = c[s.ky] - sy * cz;
    let u = cx * by - cy * bx; let v = ax * cy - ay * cx; let w = bx * ay - by * ax;
    let det = u + v + w;
    if ((u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0)) || det == 0.0 {
        return (false, 0.0, [0.0; 3]);
    }
    let t = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / det;
    (true, t, [u / det, v / det, w / det])
}

fn trace_one(sc: &Scene, ow: [f64; 3], dw: [f64; 3], tmin: f64, tmax: f64, sk_i: i64, sk_t: i64,
             stack: &mut [i64; STACK_MAX], tstack: &mut [f64; STACK_MAX]) -> (f64, i64, i64, [f64; 3]) {
    let mut t_best = tmax; let mut hit_tri: i64 = -1; let mut hit_inst: i64 = -1; let mut hb = [0.0; 3];
    let finite = ow.iter().chain(dw.iter()).all(|v| v.is_finite());
    let mut s = Space { o: ow, d: dw, inv: [0.0; 3], shear: [0.0; 3], kx: 0, ky: 0, kz: 0 };
    s.setup();
    let mut cur: i64 = -1; let mut sp = 0usize; let mut node: i64 = 0;
    let (ok, mut t_node) = s.slab(d3(sc.root_lo, 0), d3(sc.root_hi, 0), tmin, t_best);
    let mut active = finite && sc.root_lo[0] <= sc.root_hi[0] && ok;
    while active {
        let mut need_pop = true;
        if t_node < t_best {
            let inst = (node >> 32) - 1; let nd = (node & 0xFFFF_FFFF) as usize;
            if inst != cur {
                if inst < 0 { s.o = ow; s.d = dw; } else {
                    let m = &sc.to_object[12 * inst as usize..12 * inst as usize + 12];
                    for i in 0..3 {
                        s.o[i] = m[4 * i] * ow[0] + m[4 * i + 1] * ow[1] + m[4 * i + 2] * ow[2] + m[4 * i + 3];
                        s.d[i] = m[4 * i] * dw[0] + m[4 * i + 1] * dw[1] + m[4 * i + 2] * dw[2];
                    }
                }
                s.setup(); cur = inst;
            }
            match sc.kind[nd] {
                0 => {
                    let tag = (inst + 1) << 32;
                    let (ok0, tn0) = s.slab(f3(sc.child_lo, 6 * nd), f3(sc.child_hi, 6 * nd), tmin, t_best);
                    let (ok1, tn1) = s.slab(f3(sc.child_lo, 6 * nd + 3), f3(sc.child_hi, 6 * nd + 3), tmin, t_best);
                    if ok0 || ok1 {
                        let near0 = if ok0 && ok1 { tn0 <= tn1 } else { ok0 };
                        node = tag | sc.child[2 * nd + if near0 { 0 } else { 1 }];
                        t_node = if near0 { tn0 } else { tn1 };
                        if ok0 && ok1 {
                            stack[sp] = tag | sc.child[2 * nd + if near0 { 1 } else { 0 }];
                            tstack[sp] = if near0 { tn1 } else { tn0 }; sp += 1;
                        }
                        need_pop = false;
                    }
                }
                2 => {
                    let f = sc.first[nd] as usize; let c = sc.count[nd] as usize;
                    for j in 0..c {
                        let ins = sc.slot_inst[f + j] as usize;
                        let (ok, tn) = s.slab(d3(sc.inst_lo, 3 * ins), d3(sc.inst_hi, 3 * ins), tmin, t_best);
                        if ok { stack[sp] = ((ins as i64 + 1) << 32) | sc.inst_root[ins]; tstack[sp] = tn; sp += 1; }
                    }
                }
                _ => {
                    let f = sc.first[nd] as usize; let c = sc.count[nd] as usize;
                    for g in f..f + c {
                        let (ok, t, b) = triangle(f3(sc.v0, 3 * g), f3(sc.v1, 3 * g), f3(sc.v2, 3 * g), &s);
                        if !ok || !(t > tmin && t < t_best) { continue; }
                        if inst == sk_i && sc.tri_prim[g] == sk_t { continue; }
                        t_best = t; hit_tri = g as i64; hit_inst = inst; hb = b;
                    }
                }
            }
        }
        if need_pop {
            if sp == 0 { break; }
            sp -= 1; node = stack[sp]; t_node = tstack[sp];
        }

    }
    if hit_inst >= 0 { (t_best, hit_inst, sc.tri_prim[hit_tri as usize], hb) } else { (f64::INFINITY, -1, -1, [0.0; 3]) }
}

struct Out(*mut f64, *mut i64, *mut i64, *mut f64);
unsafe impl Sync for Out {}

/// # Safety
/// `a` and `r` describe valid arrays of the sizes documented in trace.h.
#[no_mangle]
pub unsafe extern "C" fn closest_hit(a: *const Accel, r: *const Rays, nthreads: i64) -> i32 {
    let a = &*a; let r = &*r;
    if a.stack_size as usize > STACK_MAX { return -1; }
    let n = r.n as usize;
    let sc = Scene {
        child: std::slice::from_raw_parts(a.child, LEN.with(|l| l.get()[0])),
        child_lo: std::slice::from_raw_parts(a.child_lo, LEN.with(|l| l.get()[1])),
        child_hi: std::slice::from_raw_parts(a.child_hi, LEN.with(|l| l.get()[1])),
        kind: std::slice::from_raw_parts(a.kind, LEN.with(|l| l.get()[2])),
        first: std::slice::from_raw_parts(a.first, LEN.with(|l| l.get()[2])),
        count: std::slice::from_raw_parts(a.count, LEN.with(|l| l.get()[2])),
        v0: std::slice::from_raw_parts(a.v0, LEN.with(|l| l.get()[3])),
        v1: std::slice::from_raw_parts(a.v1, LEN.with(|l| l.get()[3])),
        v2: std::slice::from_raw_parts(a.v2, LEN.with(|l| l.get()[3])),
        tri_prim: std::slice::from_raw_parts(a.tri_prim, LEN.with(|l| l.get()[4])),
        slot_inst: std::slice::from_raw_parts(a.slot_inst, LEN.with(|l| l.get()[5])),
        inst_lo: std::slice::from_raw_parts(a.inst_lo, LEN.with(|l| l.get()[6])),
        inst_hi: std::slice::from_raw_parts(a.inst_hi, LEN.with(|l| l.get()[6])),
        inst_root: std::slice::from_raw_parts(a.inst_root, LEN.with(|l| l.get()[7])),
        to_object: std::slice::from_raw_parts(a.to_object, LEN.with(|l| l.get()[8])),
        root_lo: std::slice::from_raw_parts(a.root_lo, 3),
        root_hi: std::slice::from_raw_parts(a.root_hi, 3),
    };
    let o = std::slice::from_raw_parts(r.o, 3 * n); let d = std::slice::from_raw_parts(r.d, 3 * n);
    let tmin = std::slice::from_raw_parts(r.tmin, n); let tmax = std::slice::from_raw_parts(r.tmax, n);
    let ski = if r.skip_inst.is_null() { None } else { Some(std::slice::from_raw_parts(r.skip_inst, n)) };
    let skt = if r.skip_tri.is_null() { None } else { Some(std::slice::from_raw_parts(r.skip_tri, n)) };
    let out = Out(r.t, r.inst, r.prim, r.b);
    let next = AtomicI64::new(0);
    const CHUNK: i64 = 512;
    let work = || {
        let mut stack = [0i64; STACK_MAX]; let mut tstack = [0f64; STACK_MAX];
        loop {
            let s0 = next.fetch_add(CHUNK, Ordering::Relaxed);
            if s0 >= n as i64 { break; }
            for i in s0 as usize..((s0 + CHUNK) as usize).min(n) {
                let (t, inst, prim, b) = trace_one(&sc, d3(o, 3 * i), d3(d, 3 * i), tmin[i], tmax[i],
                    ski.map_or(-2, |s| s[i]), skt.map_or(-2, |s| s[i]), &mut stack, &mut tstack);
                let out = &out;
                *out.0.add(i) = t; *out.1.add(i) = inst; *out.2.add(i) = prim;
                *out.3.add(3 * i) = b[0]; *out.3.add(3 * i + 1) = b[1]; *out.3.add(3 * i + 2) = b[2];
            }
        }
    };
    let nt = nthreads.max(1) as usize;
    if nt == 1 { work(); } else { std::thread::scope(|s| { for _ in 0..nt { s.spawn(&work); } }); }
    0
}

thread_local! { static LEN: std::cell::Cell<[usize; 9]> = const { std::cell::Cell::new([0; 9]) }; }
/// Lengths of the accel arrays (elements): child, child_lo, kind/first/count, v*, tri_prim, slot_inst, inst_lo, inst_root, to_object.
#[no_mangle]
pub extern "C" fn set_lengths(l: *const i64) {
    let l = unsafe { std::slice::from_raw_parts(l, 9) };
    let mut a = [0usize; 9];
    for i in 0..9 { a[i] = l[i] as usize; }
    LEN.with(|c| c.set(a));
}
#[no_mangle]
pub extern "C" fn noop() {}
