/* Scalar port of aether3d.rendering.raytrace.accel.SceneAccel._trace (closest hit):
   same state machine, slab test, watertight triangle test (float64) and tie rules. */
#include <math.h>
#include "trace.h"
#define MIN_DIR 1e-300
#define STACK_MAX 256
enum { KIND_INNER = 0, KIND_TRIANGLES = 1, KIND_INSTANCES = 2 };

typedef struct { double o[3], d[3], inv[3], shear[3]; int kx, ky, kz; } Space;

static inline void setup(Space *s) {
    for (int i = 0; i < 3; i++) {
        double v = s->d[i];
        double safe = fabs(v) < MIN_DIR ? (v < 0 ? -MIN_DIR : MIN_DIR) : v;
        s->inv[i] = 1.0 / safe;
    }
    int z = 0; double m = fabs(s->d[0]);
    if (fabs(s->d[1]) > m) { z = 1; m = fabs(s->d[1]); }
    if (fabs(s->d[2]) > m) z = 2;
    int x = (z + 1) % 3, y = (x + 1) % 3;
    double dz = s->d[z];
    if (dz < 0.0) { int t = x; x = y; y = t; }
    double dzs = dz == 0.0 ? 1.0 : dz;
    s->shear[0] = s->d[x] / dzs; s->shear[1] = s->d[y] / dzs; s->shear[2] = 1.0 / dzs;
    s->kx = x; s->ky = y; s->kz = z;
}
static inline double mn(double a, double b) { return a < b ? a : b; }
static inline double mx(double a, double b) { return a > b ? a : b; }
static inline int box_f(const float *lo, const float *hi, const Space *s, double tmin, double tmax, double *tn) {
    double a[3], b[3];
    for (int i = 0; i < 3; i++) {
        double t1 = ((double)lo[i] - s->o[i]) * s->inv[i], t2 = ((double)hi[i] - s->o[i]) * s->inv[i];
        a[i] = mn(t1, t2); b[i] = mx(t1, t2);
    }
    double near = mx(mx(a[0], a[1]), mx(a[2], tmin));
    double far = mn(mn(b[0], b[1]), b[2]) * (1.0 + 4e-16);
    *tn = near; return near <= far && near < tmax;
}
static inline int box_d(const double *lo, const double *hi, const Space *s, double tmin, double tmax, double *tn) {
    double a[3], b[3];
    for (int i = 0; i < 3; i++) {
        double t1 = (lo[i] - s->o[i]) * s->inv[i], t2 = (hi[i] - s->o[i]) * s->inv[i];
        a[i] = mn(t1, t2); b[i] = mx(t1, t2);
    }
    double near = mx(mx(a[0], a[1]), mx(a[2], tmin));
    double far = mn(mn(b[0], b[1]), b[2]) * (1.0 + 4e-16);
    *tn = near; return near <= far && near < tmax;
}
static inline int triangle(const float *p0, const float *p1, const float *p2, const Space *s, double *t, double bary[3]) {
    double a[3], b[3], c[3];
    for (int i = 0; i < 3; i++) { a[i] = (double)p0[i] - s->o[i]; b[i] = (double)p1[i] - s->o[i]; c[i] = (double)p2[i] - s->o[i]; }
    double sx = s->shear[0], sy = s->shear[1], sz = s->shear[2];
    double az = a[s->kz], bz = b[s->kz], cz = c[s->kz];
    double ax = a[s->kx] - sx * az, ay = a[s->ky] - sy * az;
    double bx = b[s->kx] - sx * bz, by = b[s->ky] - sy * bz;
    double cx = c[s->kx] - sx * cz, cy = c[s->ky] - sy * cz;
    double u = cx * by - cy * bx, v = ax * cy - ay * cx, w = bx * ay - by * ax;
    double det = u + v + w;
    if (((u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0)) || det == 0.0) return 0;
    *t = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / det;
    bary[0] = u / det; bary[1] = v / det; bary[2] = w / det;
    return 1;
}
static void enter(const Accel *A, int64_t inst, Space *s, const double *ow, const double *dw) {
    if (inst < 0) { for (int i = 0; i < 3; i++) { s->o[i] = ow[i]; s->d[i] = dw[i]; } }
    else {
        const double *m = A->to_object + 12 * inst;
        for (int i = 0; i < 3; i++) {
            s->o[i] = m[4 * i] * ow[0] + m[4 * i + 1] * ow[1] + m[4 * i + 2] * ow[2] + m[4 * i + 3];
            s->d[i] = m[4 * i] * dw[0] + m[4 * i + 1] * dw[1] + m[4 * i + 2] * dw[2];
        }
    }
    setup(s);
}
static void trace_one(const Accel *A, const Rays *R, int64_t r) {
    const double *ow = R->o + 3 * r, *dw = R->d + 3 * r;
    double tmin = R->tmin[r], t_best = R->tmax[r];
    int64_t hit_tri = -1, hit_inst = -1; double hb[3] = {0, 0, 0};
    int64_t stack[STACK_MAX]; double tstack[STACK_MAX]; int sp = 0;
    int finite = isfinite(ow[0]) && isfinite(ow[1]) && isfinite(ow[2]) && isfinite(dw[0]) && isfinite(dw[1]) && isfinite(dw[2]);
    Space s; int64_t cur = -1;
    for (int i = 0; i < 3; i++) { s.o[i] = ow[i]; s.d[i] = dw[i]; }
    setup(&s);
    double t_node; int64_t node = 0;
    int active = finite && A->root_lo[0] <= A->root_hi[0] && box_d(A->root_lo, A->root_hi, &s, tmin, t_best, &t_node);
    int64_t sk_i = R->skip_inst ? R->skip_inst[r] : -2, sk_t = R->skip_tri ? R->skip_tri[r] : -2;
    while (active) {
        int need_pop = 1;
        if (t_node < t_best) {
            int64_t inst = (node >> 32) - 1, nd = node & 0xFFFFFFFF;
            if (inst != cur) { enter(A, inst, &s, ow, dw); cur = inst; }
            int k = A->kind[nd];
            if (k == KIND_INNER) {
                int64_t tag = (inst + 1) << 32;
                double tn0, tn1;
                int ok0 = box_f(A->child_lo + 6 * nd, A->child_hi + 6 * nd, &s, tmin, t_best, &tn0);
                int ok1 = box_f(A->child_lo + 6 * nd + 3, A->child_hi + 6 * nd + 3, &s, tmin, t_best, &tn1);
                if (ok0 || ok1) {
                    int near0 = (ok0 && ok1) ? (tn0 <= tn1) : ok0;
                    node = tag | A->child[2 * nd + (near0 ? 0 : 1)];
                    t_node = near0 ? tn0 : tn1;
                    if (ok0 && ok1) { stack[sp] = tag | A->child[2 * nd + (near0 ? 1 : 0)]; tstack[sp] = near0 ? tn1 : tn0; sp++; }
                    need_pop = 0;
                }
            } else if (k == KIND_INSTANCES) {
                int64_t f = A->first[nd], c = A->count[nd];
                for (int64_t j = 0; j < c; j++) {
                    int64_t ins = A->slot_inst[f + j]; double tn;
                    /* world-space ray here (cur == -1) */
                    if (box_d(A->inst_lo + 3 * ins, A->inst_hi + 3 * ins, &s, tmin, t_best, &tn)) {
                        stack[sp] = ((ins + 1) << 32) | A->inst_root[ins]; tstack[sp] = tn; sp++;
                    }
                }
            } else {
                int64_t f = A->first[nd], c = A->count[nd];
                for (int64_t g = f; g < f + c; g++) {
                    double t, b[3];
                    if (!triangle(A->v0 + 3 * g, A->v1 + 3 * g, A->v2 + 3 * g, &s, &t, b)) continue;
                    if (!(t > tmin && t < t_best)) continue;
                    if (inst == sk_i && A->tri_prim[g] == sk_t) continue;
                    t_best = t; hit_tri = g; hit_inst = inst; hb[0] = b[0]; hb[1] = b[1]; hb[2] = b[2];
                }
            }
        }
        if (need_pop) {
            if (sp == 0) break;
            sp--; node = stack[sp]; t_node = tstack[sp];
        }
    }
    if (hit_inst >= 0) {
        R->t[r] = t_best; R->inst[r] = hit_inst; R->prim[r] = A->tri_prim[hit_tri];
        R->b[3 * r] = hb[0]; R->b[3 * r + 1] = hb[1]; R->b[3 * r + 2] = hb[2];
    } else {
        R->t[r] = INFINITY; R->inst[r] = -1; R->prim[r] = -1; R->b[3 * r] = R->b[3 * r + 1] = R->b[3 * r + 2] = 0.0;
    }
}
int closest_hit(const Accel *A, const Rays *R, int64_t nthreads) {
    if (A->stack_size > STACK_MAX) return -1;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 512) num_threads(nthreads)
#endif
    for (int64_t r = 0; r < R->n; r++) trace_one(A, R, r);
    return 0;
}
void noop(void) {}
