//! Transform hierarchies (`aether3d/rendering/scene_data.py`).
//!
//! `local_matrices` is `_local_matrices` and `world_matrices` the top-down world stack of
//! `_world_stack` (`(root @ parent) @ local`, each product `_mat4_mul`'s fixed sum order),
//! with the same operations in the same order: bitwise the NumPy results.

/// 4x4 local matrices from positions (n, 3), unit quaternions (n, 4, xyzw) and scales (n, 3).
pub fn local_matrices(position: &[f64], rotation: &[f64], scale: &[f64], out: &mut [f64]) {
    let n = position.len() / 3;
    for k in 0..n {
        let (x, y, z, w) = (
            rotation[4 * k],
            rotation[4 * k + 1],
            rotation[4 * k + 2],
            rotation[4 * k + 3],
        );
        let r = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        let m = &mut out[16 * k..16 * k + 16];
        for i in 0..3 {
            for j in 0..3 {
                m[4 * i + j] = r[i][j] * scale[3 * k + j];
            }
            m[4 * i + 3] = position[3 * k + i];
        }
        m[12] = 0.0;
        m[13] = 0.0;
        m[14] = 0.0;
        m[15] = 1.0;
    }
}

/// World matrices (n, 4, 4) of objects in depth-first order: a root's is its local
/// matrix, every other's its parent's world matrix times its local one. `parent[k]` is -1
/// or a row before `k` (checked by the caller).
pub fn world_matrices(local: &[f64], parent: &[i64], out: &mut [f64]) {
    let n = parent.len();
    for k in 0..n {
        let p = parent[k];
        if p < 0 {
            out[16 * k..16 * k + 16].copy_from_slice(&local[16 * k..16 * k + 16]);
            continue;
        }
        let p = p as usize;
        let mut m = [0.0f64; 16];
        {
            let a = &out[16 * p..16 * p + 16];
            let b = &local[16 * k..16 * k + 16];
            for i in 0..4 {
                for j in 0..4 {
                    m[4 * i + j] = a[4 * i] * b[j]
                        + a[4 * i + 1] * b[4 + j]
                        + a[4 * i + 2] * b[8 + j]
                        + a[4 * i + 3] * b[12 + j];
                }
            }
        }
        out[16 * k..16 * k + 16].copy_from_slice(&m);
    }
}
