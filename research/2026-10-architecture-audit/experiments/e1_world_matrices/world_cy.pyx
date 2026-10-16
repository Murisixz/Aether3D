# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
cimport cython
cdef inline void local_matrix(const double[:, ::1] pos, const double[:, ::1] quat, const double[:, ::1] scale, Py_ssize_t k, double* m) noexcept nogil:
    cdef double x = quat[k, 0], y = quat[k, 1], z = quat[k, 2], w = quat[k, 3]
    cdef double r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - z * w), r02 = 2 * (x * z + y * w)
    cdef double r10 = 2 * (x * y + z * w), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - x * w)
    cdef double r20 = 2 * (x * z - y * w), r21 = 2 * (y * z + x * w), r22 = 1 - 2 * (x * x + y * y)
    m[0] = r00 * scale[k, 0]; m[1] = r01 * scale[k, 1]; m[2] = r02 * scale[k, 2]; m[3] = pos[k, 0]
    m[4] = r10 * scale[k, 0]; m[5] = r11 * scale[k, 1]; m[6] = r12 * scale[k, 2]; m[7] = pos[k, 1]
    m[8] = r20 * scale[k, 0]; m[9] = r21 * scale[k, 1]; m[10] = r22 * scale[k, 2]; m[11] = pos[k, 2]
    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1
def world_matrices(const long[::1] parent, const double[:, ::1] pos, const double[:, ::1] quat, const double[:, ::1] scale, double[:, :, ::1] out):
    cdef Py_ssize_t n = parent.shape[0], k, i, j, p
    cdef double local[16]
    cdef double* o
    with nogil:
        for k in range(n):
            o = &out[k, 0, 0]
            if parent[k] < 0:
                local_matrix(pos, quat, scale, k, o)
                continue
            local_matrix(pos, quat, scale, k, local)
            p = parent[k]
            for i in range(4):
                for j in range(4):
                    o[4 * i + j] = (out[p, i, 0] * local[0 + j] + out[p, i, 1] * local[4 + j]
                                    + out[p, i, 2] * local[8 + j] + out[p, i, 3] * local[12 + j])
