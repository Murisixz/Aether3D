//! Delta mush: `aether3d.animation.rig.delta_mush`'s NumPy arithmetic, in the same order
//! (no fused multiply-adds). The smoothing gathers each vertex's neighbours in the order of
//! the sorted edge list -- the order `np.bincount` accumulates them in -- so every sum is the
//! NumPy one; the area-weighted normals add each triangle's cross product corner by corner
//! in triangle order (three accumulators, summed as `(c0 + c1) + c2`); frames and the result
//! are per vertex.

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

/// Smooths `x` (n, 3) in place `iterations` times: `x += lam (mean of neighbours - x)`.
/// `start` (n + 1) are each vertex's first edge in `src` (edges sorted by destination).
fn smooth(x: &mut [f64], src: &[i64], start: &[usize], deg: &[i64], iterations: usize, lam: f64) {
    let n = x.len() / 3;
    let mut nxt = x.to_vec();
    for _ in 0..iterations {
        for v in 0..n {
            if deg[v] <= 0 {
                nxt[3 * v] = x[3 * v];
                nxt[3 * v + 1] = x[3 * v + 1];
                nxt[3 * v + 2] = x[3 * v + 2];
                continue;
            }
            let inv = 1.0 / (deg[v] as f64);
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            for &u in &src[start[v]..start[v + 1]] {
                let u = u as usize;
                sx += x[3 * u];
                sy += x[3 * u + 1];
                sz += x[3 * u + 2];
            }
            let (mx, my, mz) = (sx * inv, sy * inv, sz * inv);
            nxt[3 * v] = x[3 * v] + lam * (mx - x[3 * v]);
            nxt[3 * v + 1] = x[3 * v + 1] + lam * (my - x[3 * v + 1]);
            nxt[3 * v + 2] = x[3 * v + 2] + lam * (mz - x[3 * v + 2]);
        }
        x.copy_from_slice(&nxt);
    }
}

/// Per vertex, the sum of its triangles' cross products, corner by corner.
fn face_normal_sums(p: &[f64], tris: &[i64]) -> Vec<f64> {
    let n = p.len() / 3;
    let t = tris.len() / 3;
    let mut acc = [
        vec![0.0f64; 3 * n],
        vec![0.0f64; 3 * n],
        vec![0.0f64; 3 * n],
    ];
    for f in 0..t {
        let a = tris[3 * f] as usize;
        let b = tris[3 * f + 1] as usize;
        let c = tris[3 * f + 2] as usize;
        let u = (
            p[3 * b] - p[3 * a],
            p[3 * b + 1] - p[3 * a + 1],
            p[3 * b + 2] - p[3 * a + 2],
        );
        let w = (
            p[3 * c] - p[3 * a],
            p[3 * c + 1] - p[3 * a + 1],
            p[3 * c + 2] - p[3 * a + 2],
        );
        let fx = u.1 * w.2 - u.2 * w.1;
        let fy = u.2 * w.0 - u.0 * w.2;
        let fz = u.0 * w.1 - u.1 * w.0;
        for (corner, &v) in [a, b, c].iter().enumerate() {
            acc[corner][3 * v] += fx;
            acc[corner][3 * v + 1] += fy;
            acc[corner][3 * v + 2] += fz;
        }
    }
    let mut out = vec![0.0f64; 3 * n];
    for e in 0..3 * n {
        out[e] = (acc[0][e] + acc[1][e]) + acc[2][e];
    }
    out
}

/// Delta mush of `pos` (n, 3) into `out` (n, 3) float32. Indices must be checked by the
/// caller: `src` within 0..n and sorted by destination with `deg` edges per vertex,
/// `first` within -1..n, `tris` within 0..n.
#[allow(clippy::too_many_arguments)]
pub fn delta_mush(
    pos: &[f64],
    src: &[i64],
    deg: &[i64],
    first: &[i64],
    tris: &[i64],
    delta: &[f64],
    mask: &[f64],
    iterations: usize,
    lam: f64,
    out: &mut [f32],
) {
    let n = pos.len() / 3;
    let mut start = vec![0usize; n + 1];
    for v in 0..n {
        start[v + 1] = start[v] + deg[v].max(0) as usize;
    }
    let mut sm = pos.to_vec();
    smooth(&mut sm, src, &start, deg, iterations, lam);
    let f = face_normal_sums(&sm, tris);
    for v in 0..n {
        let (nx, ny, mut nz, length) = unit3(f[3 * v], f[3 * v + 1], f[3 * v + 2]);
        if length <= 0.0 {
            nz = 1.0;
        }
        let (rx, ry, rz) = if first[v] >= 0 {
            let j = first[v] as usize;
            (
                sm[3 * j] - sm[3 * v],
                sm[3 * j + 1] - sm[3 * v + 1],
                sm[3 * j + 2] - sm[3 * v + 2],
            )
        } else {
            (1.0, 0.0, 0.0)
        };
        let d = (rx * nx + ry * ny) + rz * nz;
        let (tx, ty, tz, tl) = unit3(rx - d * nx, ry - d * ny, rz - d * nz);
        let a = cross((nx, ny, nz), (1.0, 0.0, 0.0));
        let (_, _, _, al) = unit3(a.0, a.1, a.2);
        let b = cross((nx, ny, nz), (0.0, 1.0, 0.0));
        let a = if al > 1e-6 { a } else { b };
        let (ax, ay, az, _) = unit3(a.0, a.1, a.2);
        let (tx, ty, tz) = if tl > 1e-12 {
            (tx, ty, tz)
        } else {
            (ax, ay, az)
        };
        let (bx, by, bz) = cross((nx, ny, nz), (tx, ty, tz));
        let (d0, d1, d2) = (delta[3 * v], delta[3 * v + 1], delta[3 * v + 2]);
        let ox = sm[3 * v] + ((tx * d0 + bx * d1) + nx * d2);
        let oy = sm[3 * v + 1] + ((ty * d0 + by * d1) + ny * d2);
        let oz = sm[3 * v + 2] + ((tz * d0 + bz * d1) + nz * d2);
        let m = mask[v];
        let (px, py, pz) = (pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]);
        out[3 * v] = (px + m * (ox - px)) as f32;
        out[3 * v + 1] = (py + m * (oy - py)) as f32;
        out[3 * v + 2] = (pz + m * (oz - pz)) as f32;
    }
}
