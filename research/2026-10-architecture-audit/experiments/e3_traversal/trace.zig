//! Scalar port of SceneAccel._trace (closest hit), many-item pointers (no bounds checks in ReleaseFast).
const std = @import("std");
const Accel = extern struct {
    child: [*]const i64, child_lo: [*]const f32, child_hi: [*]const f32, kind: [*]const i8,
    first: [*]const i64, count: [*]const i64, v0: [*]const f32, v1: [*]const f32, v2: [*]const f32,
    tri_prim: [*]const i64, slot_inst: [*]const i64, inst_lo: [*]const f64, inst_hi: [*]const f64,
    inst_root: [*]const i64, to_object: [*]const f64, root_lo: [*]const f64, root_hi: [*]const f64,
    stack_size: i64,
};
const Rays = extern struct {
    n: i64, o: [*]const f64, d: [*]const f64, tmin: [*]const f64, tmax: [*]const f64,
    skip_inst: ?[*]const i64, skip_tri: ?[*]const i64, t: [*]f64, inst: [*]i64, prim: [*]i64, b: [*]f64,
};
const MIN_DIR: f64 = 1e-300;
const STACK_MAX = 256;
const Space = struct {
    o: [3]f64, d: [3]f64, inv: [3]f64 = .{ 0, 0, 0 }, shear: [3]f64 = .{ 0, 0, 0 }, kx: usize = 0, ky: usize = 0, kz: usize = 0,
    fn setup(s: *Space) void {
        for (0..3) |i| {
            const v = s.d[i];
            const safe = if (@abs(v) < MIN_DIR) (if (v < 0) -MIN_DIR else MIN_DIR) else v;
            s.inv[i] = 1.0 / safe;
        }
        var z: usize = 0;
        var m = @abs(s.d[0]);
        if (@abs(s.d[1]) > m) { z = 1; m = @abs(s.d[1]); }
        if (@abs(s.d[2]) > m) z = 2;
        var x = (z + 1) % 3;
        var y = (x + 1) % 3;
        const dz = s.d[z];
        if (dz < 0.0) { const t = x; x = y; y = t; }
        const dzs = if (dz == 0.0) 1.0 else dz;
        s.shear = .{ s.d[x] / dzs, s.d[y] / dzs, 1.0 / dzs };
        s.kx = x; s.ky = y; s.kz = z;
    }
    inline fn slab(s: *const Space, lo: [3]f64, hi: [3]f64, tmin: f64, tmax: f64, tn: *f64) bool {
        var a: [3]f64 = undefined;
        var b: [3]f64 = undefined;
        for (0..3) |i| {
            const t1 = (lo[i] - s.o[i]) * s.inv[i];
            const t2 = (hi[i] - s.o[i]) * s.inv[i];
            a[i] = if (t1 < t2) t1 else t2;
            b[i] = if (t1 > t2) t1 else t2;
        }
        const n01 = if (a[0] > a[1]) a[0] else a[1];
        const n2 = if (a[2] > tmin) a[2] else tmin;
        const near = if (n01 > n2) n01 else n2;
        const f01 = if (b[0] < b[1]) b[0] else b[1];
        const far = (if (f01 < b[2]) f01 else b[2]) * (1.0 + 4e-16);
        tn.* = near;
        return near <= far and near < tmax;
    }
};
inline fn f3(p: [*]const f32, k: usize) [3]f64 { return .{ @floatCast(p[k]), @floatCast(p[k + 1]), @floatCast(p[k + 2]) }; }
inline fn d3(p: [*]const f64, k: usize) [3]f64 { return .{ p[k], p[k + 1], p[k + 2] }; }
inline fn triangle(p0: [3]f64, p1: [3]f64, p2: [3]f64, s: *const Space, t: *f64, bary: *[3]f64) bool {
    const a = [3]f64{ p0[0] - s.o[0], p0[1] - s.o[1], p0[2] - s.o[2] };
    const b = [3]f64{ p1[0] - s.o[0], p1[1] - s.o[1], p1[2] - s.o[2] };
    const c = [3]f64{ p2[0] - s.o[0], p2[1] - s.o[1], p2[2] - s.o[2] };
    const sx = s.shear[0]; const sy = s.shear[1]; const sz = s.shear[2];
    const az = a[s.kz]; const bz = b[s.kz]; const cz = c[s.kz];
    const ax = a[s.kx] - sx * az; const ay = a[s.ky] - sy * az;
    const bx = b[s.kx] - sx * bz; const by = b[s.ky] - sy * bz;
    const cx = c[s.kx] - sx * cz; const cy = c[s.ky] - sy * cz;
    const u = cx * by - cy * bx; const v = ax * cy - ay * cx; const w = bx * ay - by * ax;
    const det = u + v + w;
    if (((u < 0.0 or v < 0.0 or w < 0.0) and (u > 0.0 or v > 0.0 or w > 0.0)) or det == 0.0) return false;
    t.* = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / det;
    bary.* = .{ u / det, v / det, w / det };
    return true;
}
fn traceOne(A: *const Accel, R: *const Rays, r: usize, stack: *[STACK_MAX]i64, tstack: *[STACK_MAX]f64) void {
    const ow = d3(R.o, 3 * r); const dw = d3(R.d, 3 * r);
    const tmin = R.tmin[r];
    var t_best = R.tmax[r];
    var hit_tri: i64 = -1; var hit_inst: i64 = -1; var hb = [3]f64{ 0, 0, 0 };
    const finite = std.math.isFinite(ow[0]) and std.math.isFinite(ow[1]) and std.math.isFinite(ow[2]) and
        std.math.isFinite(dw[0]) and std.math.isFinite(dw[1]) and std.math.isFinite(dw[2]);
    var s = Space{ .o = ow, .d = dw };
    s.setup();
    var cur: i64 = -1; var sp: usize = 0; var node: i64 = 0; var t_node: f64 = 0;
    const sk_i: i64 = if (R.skip_inst) |p| p[r] else -2;
    const sk_t: i64 = if (R.skip_tri) |p| p[r] else -2;
    var active = finite and A.root_lo[0] <= A.root_hi[0] and s.slab(d3(A.root_lo, 0), d3(A.root_hi, 0), tmin, t_best, &t_node);
    while (active) {
        var need_pop = true;
        if (t_node < t_best) {
            const inst = (node >> 32) - 1;
            const nd: usize = @intCast(node & 0xFFFFFFFF);
            if (inst != cur) {
                if (inst < 0) { s.o = ow; s.d = dw; } else {
                    const m = A.to_object + 12 * @as(usize, @intCast(inst));
                    for (0..3) |i| {
                        s.o[i] = m[4 * i] * ow[0] + m[4 * i + 1] * ow[1] + m[4 * i + 2] * ow[2] + m[4 * i + 3];
                        s.d[i] = m[4 * i] * dw[0] + m[4 * i + 1] * dw[1] + m[4 * i + 2] * dw[2];
                    }
                }
                s.setup(); cur = inst;
            }
            const k = A.kind[nd];
            if (k == 0) {
                const tag = (inst + 1) << 32;
                var tn0: f64 = 0; var tn1: f64 = 0;
                const ok0 = s.slab(f3(A.child_lo, 6 * nd), f3(A.child_hi, 6 * nd), tmin, t_best, &tn0);
                const ok1 = s.slab(f3(A.child_lo, 6 * nd + 3), f3(A.child_hi, 6 * nd + 3), tmin, t_best, &tn1);
                if (ok0 or ok1) {
                    const near0 = if (ok0 and ok1) tn0 <= tn1 else ok0;
                    node = tag | A.child[2 * nd + @as(usize, if (near0) 0 else 1)];
                    t_node = if (near0) tn0 else tn1;
                    if (ok0 and ok1) {
                        stack[sp] = tag | A.child[2 * nd + @as(usize, if (near0) 1 else 0)];
                        tstack[sp] = if (near0) tn1 else tn0;
                        sp += 1;
                    }
                    need_pop = false;
                }
            } else if (k == 2) {
                const f: usize = @intCast(A.first[nd]); const c: usize = @intCast(A.count[nd]);
                for (0..c) |j| {
                    const ins: usize = @intCast(A.slot_inst[f + j]);
                    var tn: f64 = 0;
                    if (s.slab(d3(A.inst_lo, 3 * ins), d3(A.inst_hi, 3 * ins), tmin, t_best, &tn)) {
                        stack[sp] = ((@as(i64, @intCast(ins)) + 1) << 32) | A.inst_root[ins];
                        tstack[sp] = tn;
                        sp += 1;
                    }
                }
            } else {
                const f: usize = @intCast(A.first[nd]); const c: usize = @intCast(A.count[nd]);
                var g = f;
                while (g < f + c) : (g += 1) {
                    var t: f64 = 0; var b: [3]f64 = undefined;
                    if (!triangle(f3(A.v0, 3 * g), f3(A.v1, 3 * g), f3(A.v2, 3 * g), &s, &t, &b)) continue;
                    if (!(t > tmin and t < t_best)) continue;
                    if (inst == sk_i and A.tri_prim[g] == sk_t) continue;
                    t_best = t; hit_tri = @intCast(g); hit_inst = inst; hb = b;
                }
            }
        }
        if (need_pop) {
            if (sp == 0) { active = false; continue; }
            sp -= 1; node = stack[sp]; t_node = tstack[sp];
        }
    }
    if (hit_inst >= 0) {
        R.t[r] = t_best; R.inst[r] = hit_inst; R.prim[r] = A.tri_prim[@intCast(hit_tri)];
        R.b[3 * r] = hb[0]; R.b[3 * r + 1] = hb[1]; R.b[3 * r + 2] = hb[2];
    } else {
        R.t[r] = std.math.inf(f64); R.inst[r] = -1; R.prim[r] = -1;
        R.b[3 * r] = 0; R.b[3 * r + 1] = 0; R.b[3 * r + 2] = 0;
    }
}
fn worker(A: *const Accel, R: *const Rays, next: *std.atomic.Value(usize)) void {
    var stack: [STACK_MAX]i64 = undefined;
    var tstack: [STACK_MAX]f64 = undefined;
    const n: usize = @intCast(R.n);
    while (true) {
        const s0 = next.fetchAdd(512, .monotonic);
        if (s0 >= n) break;
        const end = @min(s0 + 512, n);
        var i = s0;
        while (i < end) : (i += 1) traceOne(A, R, i, &stack, &tstack);
    }
}
export fn closest_hit(A: *const Accel, R: *const Rays, nthreads: i64) c_int {
    if (A.stack_size > STACK_MAX) return -1;
    var next = std.atomic.Value(usize).init(0);
    const nt: usize = @intCast(@max(nthreads, 1));
    if (nt == 1) { worker(A, R, &next); return 0; }
    var threads: [64]std.Thread = undefined;
    const count = @min(nt, 64);
    for (0..count) |i| threads[i] = std.Thread.spawn(.{}, worker, .{ A, R, &next }) catch return -2;
    for (0..count) |i| threads[i].join();
    return 0;
}
export fn noop() void {}
