/* World matrices of a hierarchy: world[k] = world[parent[k]] @ local[k] (parents first).
   Same arithmetic order as Aether3D's NumPy code (_local_matrices, _mat4_mul_many). */
#include <stdint.h>
static void local_matrix(const double *p, const double *q, const double *s, double *m) {
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - z * w), r02 = 2 * (x * z + y * w);
    double r10 = 2 * (x * y + z * w), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - x * w);
    double r20 = 2 * (x * z - y * w), r21 = 2 * (y * z + x * w), r22 = 1 - 2 * (x * x + y * y);
    m[0] = r00 * s[0]; m[1] = r01 * s[1]; m[2] = r02 * s[2]; m[3] = p[0];
    m[4] = r10 * s[0]; m[5] = r11 * s[1]; m[6] = r12 * s[2]; m[7] = p[1];
    m[8] = r20 * s[0]; m[9] = r21 * s[1]; m[10] = r22 * s[2]; m[11] = p[2];
    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
}
void world_matrices(int64_t n, const int64_t *parent, const double *pos, const double *quat,
                    const double *scale, double *out) {
    double local[16];
    for (int64_t k = 0; k < n; k++) {
        double *o = out + 16 * k;
        if (parent[k] < 0) { local_matrix(pos + 3 * k, quat + 4 * k, scale + 3 * k, o); continue; }
        local_matrix(pos + 3 * k, quat + 4 * k, scale + 3 * k, local);
        const double *a = out + 16 * parent[k];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                o[4 * i + j] = a[4 * i + 0] * local[0 + j] + a[4 * i + 1] * local[4 + j]
                             + a[4 * i + 2] * local[8 + j] + a[4 * i + 3] * local[12 + j];
    }
}
void noop(void) {}
