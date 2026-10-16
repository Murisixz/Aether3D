// Scalar port of SceneAccel._trace (closest hit); unsafe pointers, Native AOT.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
[module: SkipLocalsInit]

[StructLayout(LayoutKind.Sequential)]
public unsafe struct Accel {
    public long* child; public float* child_lo; public float* child_hi; public sbyte* kind;
    public long* first; public long* count; public float* v0; public float* v1; public float* v2;
    public long* tri_prim; public long* slot_inst; public double* inst_lo; public double* inst_hi;
    public long* inst_root; public double* to_object; public double* root_lo; public double* root_hi;
    public long stack_size;
}
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Rays {
    public long n; public double* o; public double* d; public double* tmin; public double* tmax;
    public long* skip_inst; public long* skip_tri; public double* t; public long* inst; public long* prim; public double* b;
}
public unsafe struct Space {
    public double o0, o1, o2, d0, d1, d2, i0, i1, i2, sx, sy, sz; public int kx, ky, kz;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double O(int i) => i == 0 ? o0 : i == 1 ? o1 : o2;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double D(int i) => i == 0 ? d0 : i == 1 ? d1 : d2;
}
public static unsafe class Trace {
    const double MinDir = 1e-300;
    const int StackMax = 256;
    static double Safe(double v) => Math.Abs(v) < MinDir ? (v < 0 ? -MinDir : MinDir) : v;
    static void Setup(ref Space s) {
        s.i0 = 1.0 / Safe(s.d0); s.i1 = 1.0 / Safe(s.d1); s.i2 = 1.0 / Safe(s.d2);
        int z = 0; double m = Math.Abs(s.d0);
        if (Math.Abs(s.d1) > m) { z = 1; m = Math.Abs(s.d1); }
        if (Math.Abs(s.d2) > m) z = 2;
        int x = (z + 1) % 3, y = (x + 1) % 3;
        double dz = s.D(z);
        if (dz < 0.0) { int t = x; x = y; y = t; }
        double dzs = dz == 0.0 ? 1.0 : dz;
        s.sx = s.D(x) / dzs; s.sy = s.D(y) / dzs; s.sz = 1.0 / dzs;
        s.kx = x; s.ky = y; s.kz = z;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] static double Mn(double a, double b) => a < b ? a : b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] static double Mx(double a, double b) => a > b ? a : b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static bool Slab(in Space s, double l0, double l1, double l2, double h0, double h1, double h2, double tmin, double tmax, out double near) {
        double a0 = (l0 - s.o0) * s.i0, b0 = (h0 - s.o0) * s.i0;
        double a1 = (l1 - s.o1) * s.i1, b1 = (h1 - s.o1) * s.i1;
        double a2 = (l2 - s.o2) * s.i2, b2 = (h2 - s.o2) * s.i2;
        double n0 = Mn(a0, b0), f0 = Mx(a0, b0), n1 = Mn(a1, b1), f1 = Mx(a1, b1), n2 = Mn(a2, b2), f2 = Mx(a2, b2);
        near = Mx(Mx(n0, n1), Mx(n2, tmin));
        double far = Mn(Mn(f0, f1), f2) * (1.0 + 4e-16);
        return near <= far && near < tmax;
    }
    static bool Tri(float* p0, float* p1, float* p2, in Space s, out double t, out double u0, out double u1, out double u2) {
        double* a = stackalloc double[3]; double* b = stackalloc double[3]; double* c = stackalloc double[3];
        a[0] = p0[0] - s.o0; a[1] = p0[1] - s.o1; a[2] = p0[2] - s.o2;
        b[0] = p1[0] - s.o0; b[1] = p1[1] - s.o1; b[2] = p1[2] - s.o2;
        c[0] = p2[0] - s.o0; c[1] = p2[1] - s.o1; c[2] = p2[2] - s.o2;
        double az = a[s.kz], bz = b[s.kz], cz = c[s.kz];
        double ax = a[s.kx] - s.sx * az, ay = a[s.ky] - s.sy * az;
        double bx = b[s.kx] - s.sx * bz, by = b[s.ky] - s.sy * bz;
        double cx = c[s.kx] - s.sx * cz, cy = c[s.ky] - s.sy * cz;
        double u = cx * by - cy * bx, v = ax * cy - ay * cx, w = bx * ay - by * ax;
        double det = u + v + w;
        if (((u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0)) || det == 0.0) { t = 0; u0 = u1 = u2 = 0; return false; }
        t = (u * (s.sz * az) + v * (s.sz * bz) + w * (s.sz * cz)) / det;
        u0 = u / det; u1 = v / det; u2 = w / det;
        return true;
    }
    static void One(Accel* A, Rays* R, long r, long[] stack, double[] tstack) {
        double* ow = R->o + 3 * r; double* dw = R->d + 3 * r;
        double tmin = R->tmin[r], tBest = R->tmax[r];
        long hitTri = -1, hitInst = -1; double h0 = 0, h1 = 0, h2 = 0;
        bool finite = double.IsFinite(ow[0]) && double.IsFinite(ow[1]) && double.IsFinite(ow[2]) && double.IsFinite(dw[0]) && double.IsFinite(dw[1]) && double.IsFinite(dw[2]);
        Space s = default; s.o0 = ow[0]; s.o1 = ow[1]; s.o2 = ow[2]; s.d0 = dw[0]; s.d1 = dw[1]; s.d2 = dw[2];
        Setup(ref s);
        long cur = -1; int sp = 0; long node = 0; double tNode;
        long skI = R->skip_inst != null ? R->skip_inst[r] : -2, skT = R->skip_tri != null ? R->skip_tri[r] : -2;
        bool active = Slab(s, A->root_lo[0], A->root_lo[1], A->root_lo[2], A->root_hi[0], A->root_hi[1], A->root_hi[2], tmin, tBest, out tNode);
        active = finite && A->root_lo[0] <= A->root_hi[0] && active;
        while (active) {
            bool needPop = true;
            if (tNode < tBest) {
                long inst = (node >> 32) - 1; long nd = node & 0xFFFFFFFF;
                if (inst != cur) {
                    if (inst < 0) { s.o0 = ow[0]; s.o1 = ow[1]; s.o2 = ow[2]; s.d0 = dw[0]; s.d1 = dw[1]; s.d2 = dw[2]; }
                    else {
                        double* m = A->to_object + 12 * inst;
                        s.o0 = m[0] * ow[0] + m[1] * ow[1] + m[2] * ow[2] + m[3];
                        s.o1 = m[4] * ow[0] + m[5] * ow[1] + m[6] * ow[2] + m[7];
                        s.o2 = m[8] * ow[0] + m[9] * ow[1] + m[10] * ow[2] + m[11];
                        s.d0 = m[0] * dw[0] + m[1] * dw[1] + m[2] * dw[2];
                        s.d1 = m[4] * dw[0] + m[5] * dw[1] + m[6] * dw[2];
                        s.d2 = m[8] * dw[0] + m[9] * dw[1] + m[10] * dw[2];
                    }
                    Setup(ref s); cur = inst;
                }
                int k = A->kind[nd];
                if (k == 0) {
                    long tag = (inst + 1) << 32;
                    float* lo = A->child_lo + 6 * nd; float* hi = A->child_hi + 6 * nd;
                    bool ok0 = Slab(s, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], tmin, tBest, out double tn0);
                    bool ok1 = Slab(s, lo[3], lo[4], lo[5], hi[3], hi[4], hi[5], tmin, tBest, out double tn1);
                    if (ok0 || ok1) {
                        bool near0 = (ok0 && ok1) ? tn0 <= tn1 : ok0;
                        node = tag | A->child[2 * nd + (near0 ? 0 : 1)]; tNode = near0 ? tn0 : tn1;
                        if (ok0 && ok1) { stack[sp] = tag | A->child[2 * nd + (near0 ? 1 : 0)]; tstack[sp] = near0 ? tn1 : tn0; sp++; }
                        needPop = false;
                    }
                } else if (k == 2) {
                    long f = A->first[nd], c = A->count[nd];
                    for (long j = 0; j < c; j++) {
                        long ins = A->slot_inst[f + j];
                        double* lo = A->inst_lo + 3 * ins; double* hi = A->inst_hi + 3 * ins;
                        if (Slab(s, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], tmin, tBest, out double tn)) {
                            stack[sp] = ((ins + 1) << 32) | A->inst_root[ins]; tstack[sp] = tn; sp++;
                        }
                    }
                } else {
                    long f = A->first[nd], c = A->count[nd];
                    for (long g = f; g < f + c; g++) {
                        if (!Tri(A->v0 + 3 * g, A->v1 + 3 * g, A->v2 + 3 * g, s, out double t, out double u0, out double u1, out double u2)) continue;
                        if (!(t > tmin && t < tBest)) continue;
                        if (inst == skI && A->tri_prim[g] == skT) continue;
                        tBest = t; hitTri = g; hitInst = inst; h0 = u0; h1 = u1; h2 = u2;
                    }
                }
            }
            if (needPop) {
                if (sp == 0) break;
                sp--; node = stack[sp]; tNode = tstack[sp];
            }
        }
        if (hitInst >= 0) { R->t[r] = tBest; R->inst[r] = hitInst; R->prim[r] = A->tri_prim[hitTri]; R->b[3 * r] = h0; R->b[3 * r + 1] = h1; R->b[3 * r + 2] = h2; }
        else { R->t[r] = double.PositiveInfinity; R->inst[r] = -1; R->prim[r] = -1; R->b[3 * r] = 0; R->b[3 * r + 1] = 0; R->b[3 * r + 2] = 0; }
    }
    [UnmanagedCallersOnly(EntryPoint = "closest_hit")]
    public static int ClosestHit(Accel* A, Rays* R, long nthreads) {
        if (A->stack_size > StackMax) return -1;
        long n = R->n;
        long chunks = (n + 511) / 512;
        if (nthreads <= 1) {
            var st = new long[StackMax]; var ts = new double[StackMax];
            for (long r = 0; r < n; r++) One(A, R, r, st, ts);
            return 0;
        }
        var opts = new ParallelOptions { MaxDegreeOfParallelism = (int)nthreads };
        Parallel.For(0L, chunks, opts, () => (new long[StackMax], new double[StackMax]), (c, _, buf) => {
            long end = Math.Min(n, (c + 1) * 512);
            for (long r = c * 512; r < end; r++) One(A, R, r, buf.Item1, buf.Item2);
            return buf;
        }, _ => { });
        return 0;
    }
    [UnmanagedCallersOnly(EntryPoint = "noop")]
    public static void Noop() {}
}
