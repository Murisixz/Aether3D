//! Delta mush: `aether3d.animation.rig.delta_mush`'s NumPy arithmetic, in the same order
//! (no fused multiply-adds). The smoothing sums each vertex's weighted neighbours in the
//! order of the sorted edge list -- the order `np.bincount` accumulates them in -- and the
//! detail is averaged over the triangle corners in triangle order (corner `c` of triangle
//! `f` is the `3 f + c`-th), as `np.bincount` over the flattened triangles does, so every
//! sum is the NumPy one.

/// `((x * x + y * y) + z * z)`, then the unit vector (zero for a zero vector) and the length.
#[inline]
fn unit3(x: f64, y: f64, z: f64) -> (f64, f64, f64, f64) {
    let length = ((x * x + y * y) + z * z).sqrt();
    let inv = if length > 0.0 { 1.0 / length } else { 0.0 };
    (x * inv, y * inv, z * inv, length)
}

#[inline]
fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

/// Smooths `x` (n, 3) in place `iterations` times: `x += lam (weighted mean of neighbours -
/// x)`, the weights `w` of the edges `start[v]..start[v + 1]` of `src` (sorted by
/// destination) over their sum `wsum[v]` (vertices with no weight stay).
fn smooth(
    x: &mut [f64],
    src: &[i64],
    w: &[f64],
    start: &[usize],
    wsum: &[f64],
    iterations: usize,
    lam: f64,
) {
    let n = x.len() / 3;
    let mut nxt = x.to_vec();
    for _ in 0..iterations {
        for v in 0..n {
            let has = wsum[v] > 0.0;
            if !has {
                nxt[3 * v] = x[3 * v];
                nxt[3 * v + 1] = x[3 * v + 1];
                nxt[3 * v + 2] = x[3 * v + 2];
                continue;
            }
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            for e in start[v]..start[v + 1] {
                let u = src[e] as usize;
                sx += w[e] * x[3 * u];
                sy += w[e] * x[3 * u + 1];
                sz += w[e] * x[3 * u + 2];
            }
            let (mx, my, mz) = (sx / wsum[v], sy / wsum[v], sz / wsum[v]);
            nxt[3 * v] = x[3 * v] + lam * (mx - x[3 * v]);
            nxt[3 * v + 1] = x[3 * v + 1] + lam * (my - x[3 * v + 1]);
            nxt[3 * v + 2] = x[3 * v + 2] + lam * (mz - x[3 * v + 2]);
        }
        x.copy_from_slice(&nxt);
    }
}

/// Delta mush of `pos` (n, 3) into `out` (n, 3) float32. The caller checks the indices:
/// `dst` sorted and `dst`, `src`, `tris` within 0..n; `delta` is (3 t, 3), one row per
/// triangle corner.
#[allow(clippy::too_many_arguments)]
pub fn delta_mush(
    pos: &[f64],
    dst: &[i64],
    src: &[i64],
    w: &[f64],
    wsum: &[f64],
    tris: &[i64],
    delta: &[f64],
    mask: &[f64],
    iterations: usize,
    lam: f64,
    out: &mut [f32],
) {
    let n = pos.len() / 3;
    let mut start = vec![0usize; n + 1];
    for &d in dst {
        start[d as usize + 1] += 1;
    }
    for v in 0..n {
        start[v + 1] += start[v];
    }
    let mut sm = pos.to_vec();
    smooth(&mut sm, src, w, &start, wsum, iterations, lam);
    let mut acc = vec![0.0f64; 3 * n];
    let mut weight = vec![0.0f64; n];
    let t = tris.len() / 3;
    for f in 0..t {
        for c in 0..3 {
            let v = tris[3 * f + c] as usize;
            let a = tris[3 * f + (c + 1) % 3] as usize;
            let b = tris[3 * f + (c + 2) % 3] as usize;
            let (ux, uy, uz, _) = unit3(
                sm[3 * a] - sm[3 * v],
                sm[3 * a + 1] - sm[3 * v + 1],
                sm[3 * a + 2] - sm[3 * v + 2],
            );
            let (wx, wy, wz, _) = unit3(
                sm[3 * b] - sm[3 * v],
                sm[3 * b + 1] - sm[3 * v + 1],
                sm[3 * b + 2] - sm[3 * v + 2],
            );
            let cr = cross((ux, uy, uz), (wx, wy, wz));
            let (nx, ny, nz, sine) = unit3(cr.0, cr.1, cr.2);
            let (xx, xy, xz, _) = unit3(ux + wx, uy + wy, uz + wz);
            let (yx, yy, yz) = cross((nx, ny, nz), (xx, xy, xz));
            let i = 3 * f + c;
            let (d0, d1, d2) = (delta[3 * i], delta[3 * i + 1], delta[3 * i + 2]);
            acc[3 * v] += sine * ((xx * d0 + yx * d1) + nx * d2);
            acc[3 * v + 1] += sine * ((xy * d0 + yy * d1) + ny * d2);
            acc[3 * v + 2] += sine * ((xz * d0 + yz * d1) + nz * d2);
            weight[v] += sine;
        }
    }
    for v in 0..n {
        let m = mask[v];
        for k in 0..3 {
            let p = pos[3 * v + k];
            let o = if weight[v] > 0.0 {
                sm[3 * v + k] + acc[3 * v + k] / weight[v]
            } else {
                p
            };
            out[3 * v + k] = (p + m * (o - p)) as f32;
        }
    }
}
