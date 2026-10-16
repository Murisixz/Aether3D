#include <stdint.h>
typedef struct {
    const int64_t *child; const float *child_lo, *child_hi; const int8_t *kind;
    const int64_t *first, *count; const float *v0, *v1, *v2; const int64_t *tri_prim;
    const int64_t *slot_inst; const double *inst_lo, *inst_hi; const int64_t *inst_root;
    const double *to_object, *root_lo, *root_hi; int64_t stack_size;
} Accel;
typedef struct {
    int64_t n; const double *o, *d, *tmin, *tmax; const int64_t *skip_inst, *skip_tri;
    double *t; int64_t *inst, *prim; double *b;
} Rays;
