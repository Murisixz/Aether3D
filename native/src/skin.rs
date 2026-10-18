//! Linear blend skinning: `aether3d.animation.rig.skin_linear`'s NumPy arithmetic, per
//! vertex, in the same order (no fused multiply-adds): the blended matrix accumulates the
//! influences in order, positions are `((m0 x + m1 y) + m2 z) + m3` per row, normals are the
//! cofactor of the blended linear part applied to the normal, then normalized (with the
//! determinant's sign). Vertices are split between threads in fixed ranges: the result
//! never depends on the number of threads.

const CHUNK: usize = 4096;

/// Skins `n` vertices. `idx` and `w` are (n, k); `pal` is (m, 3, 4); joint indices must be
/// checked by the caller (0 <= j < m).
#[allow(clippy::too_many_arguments)]
pub fn skin_linear(
    pos: &[f32],
    nrm: Option<&[f32]>,
    idx: &[i32],
    w: &[f64],
    pal: &[f64],
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
            let mut m = [0.0f64; 12];
            let j0 = idx[v * k] as usize * 12;
            let w0 = w[v * k];
            for e in 0..12 {
                m[e] = w0 * pal[j0 + e];
            }
            for t in 1..k {
                let j = idx[v * k + t] as usize * 12;
                let wt = w[v * k + t];
                for e in 0..12 {
                    m[e] += wt * pal[j + e];
                }
            }
            let x = pos[3 * v] as f64;
            let y = pos[3 * v + 1] as f64;
            let z = pos[3 * v + 2] as f64;
            out_p[3 * local] = (((m[0] * x + m[1] * y) + m[2] * z) + m[3]) as f32;
            out_p[3 * local + 1] = (((m[4] * x + m[5] * y) + m[6] * z) + m[7]) as f32;
            out_p[3 * local + 2] = (((m[8] * x + m[9] * y) + m[10] * z) + m[11]) as f32;
            if let (Some(src), Some(dst)) = (nrm, out_n.as_deref_mut()) {
                let (a, b, c) = (m[0], m[1], m[2]);
                let (d, e, f) = (m[4], m[5], m[6]);
                let (g, h, i) = (m[8], m[9], m[10]);
                let c00 = e * i - f * h;
                let c01 = f * g - d * i;
                let c02 = d * h - e * g;
                let c10 = c * h - b * i;
                let c11 = a * i - c * g;
                let c12 = b * g - a * h;
                let c20 = b * f - c * e;
                let c21 = c * d - a * f;
                let c22 = a * e - b * d;
                let nx0 = src[3 * v] as f64;
                let ny0 = src[3 * v + 1] as f64;
                let nz0 = src[3 * v + 2] as f64;
                let nx = (c00 * nx0 + c01 * ny0) + c02 * nz0;
                let ny = (c10 * nx0 + c11 * ny0) + c12 * nz0;
                let nz = (c20 * nx0 + c21 * ny0) + c22 * nz0;
                let det = (a * c00 + b * c01) + c * c02;
                let sign = if det < 0.0 { -1.0 } else { 1.0 };
                let length = ((nx * nx + ny * ny) + nz * nz).sqrt();
                let inv = if length > 0.0 { sign / length } else { 0.0 };
                dst[3 * local] = (nx * inv) as f32;
                dst[3 * local + 1] = (ny * inv) as f32;
                dst[3 * local + 2] = (nz * inv) as f32;
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
