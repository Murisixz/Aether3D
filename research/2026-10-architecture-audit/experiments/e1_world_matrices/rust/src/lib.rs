#[inline(always)]
fn local_matrix(p: &[f64], q: &[f64], s: &[f64], m: &mut [f64; 16]) {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let r00 = 1.0 - 2.0 * (y * y + z * z); let r01 = 2.0 * (x * y - z * w); let r02 = 2.0 * (x * z + y * w);
    let r10 = 2.0 * (x * y + z * w); let r11 = 1.0 - 2.0 * (x * x + z * z); let r12 = 2.0 * (y * z - x * w);
    let r20 = 2.0 * (x * z - y * w); let r21 = 2.0 * (y * z + x * w); let r22 = 1.0 - 2.0 * (x * x + y * y);
    *m = [r00 * s[0], r01 * s[1], r02 * s[2], p[0], r10 * s[0], r11 * s[1], r12 * s[2], p[1],
          r20 * s[0], r21 * s[1], r22 * s[2], p[2], 0.0, 0.0, 0.0, 1.0];
}
/// # Safety: pointers to n (parent), 3n (pos, scale), 4n (quat), 16n (out) elements.
#[no_mangle]
pub unsafe extern "C" fn world_matrices(n: i64, parent: *const i64, pos: *const f64, quat: *const f64,
                                         scale: *const f64, out: *mut f64) {
    let n = n as usize;
    let parent = std::slice::from_raw_parts(parent, n);
    let pos = std::slice::from_raw_parts(pos, 3 * n);
    let quat = std::slice::from_raw_parts(quat, 4 * n);
    let scale = std::slice::from_raw_parts(scale, 3 * n);
    let out = std::slice::from_raw_parts_mut(out, 16 * n);
    let mut local = [0.0f64; 16];
    for k in 0..n {
        local_matrix(&pos[3 * k..3 * k + 3], &quat[4 * k..4 * k + 4], &scale[3 * k..3 * k + 3], &mut local);
        let p = parent[k];
        if p < 0 { out[16 * k..16 * k + 16].copy_from_slice(&local); continue; }
        let p = p as usize;
        let (head, tail) = out.split_at_mut(16 * k);
        let a = &head[16 * p..16 * p + 16];
        let o = &mut tail[..16];
        for i in 0..4 { for j in 0..4 {
            o[4 * i + j] = a[4 * i] * local[j] + a[4 * i + 1] * local[4 + j]
                         + a[4 * i + 2] * local[8 + j] + a[4 * i + 3] * local[12 + j];
        } }
    }
}
#[no_mangle]
pub extern "C" fn noop() {}
