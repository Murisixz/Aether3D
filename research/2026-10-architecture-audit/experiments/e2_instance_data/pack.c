/* Pack `struct Instance` records (32 floats) like Aether3D's pack_instances, from flat arrays. */
#include <stdint.h>
#include <string.h>
typedef struct { float f[32]; } Rec;
static inline void pack_one(Rec *r, const double *w, const double *nm, const float *col, float spec, float shin,
                            uint32_t flags, uint32_t index, const double *eye) {
    memset(r, 0, sizeof *r);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double v = w[4 * i + j];
            if (j == 3) v -= eye[i];            /* camera-relative in float64, then rounded */
            r->f[4 * i + j] = (float)v;
        }
        for (int j = 0; j < 3; j++) r->f[12 + 4 * i + j] = (float)nm[3 * i + j];
    }
    r->f[24] = col[0]; r->f[25] = col[1]; r->f[26] = col[2]; r->f[27] = spec; r->f[28] = shin;
    memcpy(&r->f[29], &flags, 4); memcpy(&r->f[30], &index, 4);
}
void pack_all(int64_t n, const int64_t *order, const double *world, const double *normal, const float *color,
              const float *spec, const float *shin, const uint32_t *flags, const double *eye, Rec *out) {
    for (int64_t k = 0; k < n; k++) {
        int64_t s = order[k];
        pack_one(out + k, world + 16 * s, normal + 9 * s, color + 3 * s, spec[s], shin[s], flags[s], (uint32_t)s, eye);
    }
}
/* Retained records: rewrite only the dirty instances (slot[k] is where instance idx[k] lives). */
void pack_dirty(int64_t m, const int64_t *idx, const int64_t *slot, const double *world, const double *normal,
                const float *color, const float *spec, const float *shin, const uint32_t *flags, const double *eye, Rec *out) {
    for (int64_t k = 0; k < m; k++) {
        int64_t s = idx[k];
        pack_one(out + slot[k], world + 16 * s, normal + 9 * s, color + 3 * s, spec[s], shin[s], flags[s], (uint32_t)s, eye);
    }
}
