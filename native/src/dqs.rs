//! Dual quaternion skinning (`aether3d.animation.rig.skin_dual_quaternion`): the NumPy
//! reference's arithmetic per vertex, in the same order (no fused multiply-adds). Per
//! joint, the caller passes the rotation `q` (x, y, z, w), the dual part `d` and the
//! scale/shear `kmat` (row-major 3x3) of each skinning matrix. Per vertex: the rotations
//! and dual parts are summed with weights signed against the first influence (both
//! starting from +0), `K` is summed with the plain weights, both quaternions are divided
//! by the real part's norm, the position is `K p` rotated plus the translation
//! `2 vec(dual conj(real))`; normals are `sign(det K) cof(K) n` normalized, rotated, and
//! normalized. With `blend`, linear blend skinning (`skin_linear`'s arithmetic) is mixed
//! in: `a + b (q - a)`, exactly LBS at `b = 0` and DQS at `b = 1`. Vertices are split
//! between threads in fixed ranges: the result never depends on the number of threads.

const CHUNK: usize = 4096;

#[inline]
fn rotate(u: [f64; 3], w: f64, v: [f64; 3]) -> [f64; 3] {
    let tx = 2.0 * (u[1] * v[2] - u[2] * v[1]);
    let ty = 2.0 * (u[2] * v[0] - u[0] * v[2]);
    let tz = 2.0 * (u[0] * v[1] - u[1] * v[0]);
    [
        (v[0] + w * tx) + (u[1] * tz - u[2] * ty),
        (v[1] + w * ty) + (u[2] * tx - u[0] * tz),
        (v[2] + w * tz) + (u[0] * ty - u[1] * tx),
    ]
}

/// `sign(det M) cof(M) n` normalized (`rig._cofactor_apply`), `m` row-major 3x3.
#[inline]
fn cofactor_apply(m: [f64; 9], n: [f64; 3]) -> [f64; 3] {
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[3], m[4], m[5]);
    let (g, h, i) = (m[6], m[7], m[8]);
    let c00 = e * i - f * h;
    let c01 = f * g - d * i;
    let c02 = d * h - e * g;
    let c10 = c * h - b * i;
    let c11 = a * i - c * g;
    let c12 = b * g - a * h;
    let c20 = b * f - c * e;
    let c21 = c * d - a * f;
    let c22 = a * e - b * d;
    let nx = (c00 * n[0] + c01 * n[1]) + c02 * n[2];
    let ny = (c10 * n[0] + c11 * n[1]) + c12 * n[2];
    let nz = (c20 * n[0] + c21 * n[1]) + c22 * n[2];
    let det = (a * c00 + b * c01) + c * c02;
    let sign = if det < 0.0 { -1.0 } else { 1.0 };
    let length = ((nx * nx + ny * ny) + nz * nz).sqrt();
    let inv = if length > 0.0 { sign / length } else { 0.0 };
    [nx * inv, ny * inv, nz * inv]
}

#[inline]
fn unit(v: [f64; 3]) -> [f64; 3] {
    let length = ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]).sqrt();
    let inv = if length > 0.0 { 1.0 / length } else { 0.0 };
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

#[inline]
fn mix(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    if t == 0.0 {
        a
    } else if t == 1.0 {
        b
    } else {
        [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ]
    }
}

/// Skins `n` vertices. `idx` and `w` are (n, k); `q`, `d` (m, 4); `kmat` (m, 9); `pal`
/// (m, 12) and `blend` (n) go together. Joint indices must be checked by the caller.
#[allow(clippy::too_many_arguments)]
pub fn skin_dual_quaternion(
    pos: &[f32],
    nrm: Option<&[f32]>,
    idx: &[i32],
    w: &[f64],
    q: &[f64],
    d: &[f64],
    kmat: &[f64],
    pal: Option<&[f64]>,
    blend: Option<&[f64]>,
    k: usize,
    out_p: &mut [f32],
    out_n: Option<&mut [f32]>,
    threads: usize,
) {
    let n = pos.len() / 3;
    let threads = threads.max(1).min(n.div_ceil(CHUNK).max(1));
    let work = |start: usize, out_p: &mut [f32], out_n: Option<&mut [f32]>| {
        let count = out_p.len() / 3;
        let mut out_n = out_n;
        for local in 0..count {
            let v = start + local;
            let i0 = idx[v * k] as usize;
            let (q0x, q0y, q0z, q0w) = (q[4 * i0], q[4 * i0 + 1], q[4 * i0 + 2], q[4 * i0 + 3]);
            let (mut rx, mut ry, mut rz, mut rw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let (mut dx, mut dy, mut dz, mut dw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut kb = [0.0f64; 9];
            for t in 0..k {
                let j = idx[v * k + t] as usize;
                let wj = w[v * k + t];
                let (qx, qy, qz, qw) = (q[4 * j], q[4 * j + 1], q[4 * j + 2], q[4 * j + 3]);
                let dot = ((qx * q0x + qy * q0y) + qz * q0z) + qw * q0w;
                let s = (if dot < 0.0 { -1.0 } else { 1.0 }) * wj;
                rx += s * qx;
                ry += s * qy;
                rz += s * qz;
                rw += s * qw;
                dx += s * d[4 * j];
                dy += s * d[4 * j + 1];
                dz += s * d[4 * j + 2];
                dw += s * d[4 * j + 3];
                for (e, slot) in kb.iter_mut().enumerate() {
                    *slot += wj * kmat[9 * j + e];
                }
            }
            let mut norm = (((rx * rx + ry * ry) + rz * rz) + rw * rw).sqrt();
            if norm.is_nan() || norm <= 0.0 {
                norm = 1.0;
            }
            rx /= norm;
            ry /= norm;
            rz /= norm;
            rw /= norm;
            dx /= norm;
            dy /= norm;
            dz /= norm;
            dw /= norm;
            let x = pos[3 * v] as f64;
            let y = pos[3 * v + 1] as f64;
            let z = pos[3 * v + 2] as f64;
            let px = (kb[0] * x + kb[1] * y) + kb[2] * z;
            let py = (kb[3] * x + kb[4] * y) + kb[5] * z;
            let pz = (kb[6] * x + kb[7] * y) + kb[8] * z;
            let o = rotate([rx, ry, rz], rw, [px, py, pz]);
            let (cx, cy, cz, cw) = (-rx, -ry, -rz, rw);
            let tx = 2.0 * (((dw * cx + dx * cw) + dy * cz) - dz * cy);
            let ty = 2.0 * (((dw * cy - dx * cz) + dy * cw) + dz * cx);
            let tz = 2.0 * (((dw * cz + dx * cy) - dy * cx) + dz * cw);
            let mut p = [o[0] + tx, o[1] + ty, o[2] + tz];
            let mut nn: Option<[f64; 3]> = None;
            if let Some(src) = nrm {
                let n0 = [
                    src[3 * v] as f64,
                    src[3 * v + 1] as f64,
                    src[3 * v + 2] as f64,
                ];
                let nk = cofactor_apply(kb, n0);
                nn = Some(unit(rotate([rx, ry, rz], rw, nk)));
            }
            if let (Some(pl), Some(bl)) = (pal, blend) {
                let b = bl[v];
                let j0 = idx[v * k] as usize * 12;
                let w0 = w[v * k];
                let mut m = [0.0f64; 12];
                for e in 0..12 {
                    m[e] = w0 * pl[j0 + e];
                }
                for t in 1..k {
                    let j = idx[v * k + t] as usize * 12;
                    let wt = w[v * k + t];
                    for e in 0..12 {
                        m[e] += wt * pl[j + e];
                    }
                }
                let lp = [
                    ((m[0] * x + m[1] * y) + m[2] * z) + m[3],
                    ((m[4] * x + m[5] * y) + m[6] * z) + m[7],
                    ((m[8] * x + m[9] * y) + m[10] * z) + m[11],
                ];
                p = mix(lp, p, b);
                if let (Some(src), Some(dn)) = (nrm, nn) {
                    let n0 = [
                        src[3 * v] as f64,
                        src[3 * v + 1] as f64,
                        src[3 * v + 2] as f64,
                    ];
                    let ln =
                        cofactor_apply([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]], n0);
                    nn = Some(unit(mix(ln, dn, b)));
                }
            }
            out_p[3 * local] = p[0] as f32;
            out_p[3 * local + 1] = p[1] as f32;
            out_p[3 * local + 2] = p[2] as f32;
            if let (Some(dst), Some(v3)) = (out_n.as_deref_mut(), nn) {
                dst[3 * local] = v3[0] as f32;
                dst[3 * local + 1] = v3[1] as f32;
                dst[3 * local + 2] = v3[2] as f32;
            }
        }
    };
    if threads == 1 {
        work(0, out_p, out_n);
        return;
    }
    let per = n.div_ceil(threads).div_ceil(CHUNK) * CHUNK;
    std::thread::scope(|scope| {
        let mut rest_p = out_p;
        let mut rest_n = out_n;
        let mut start = 0;
        while start < n {
            let take = per.min(n - start);
            let (head_p, tail_p) = rest_p.split_at_mut(3 * take);
            rest_p = tail_p;
            let head_n = match rest_n {
                Some(buf) => {
                    let (h, t) = buf.split_at_mut(3 * take);
                    rest_n = Some(t);
                    Some(h)
                }
                None => None,
            };
            let s = start;
            scope.spawn(move || work(s, head_p, head_n));
            start += take;
        }
    });
}
