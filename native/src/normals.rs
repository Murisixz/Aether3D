//! Surface-following normals: `aether3d.animation.rig.surface_normals`'s NumPy arithmetic,
//! in the same order (no fused multiply-adds). The per-vertex sums of the triangles' cross
//! products are accumulated corner by corner in triangle order -- each corner's sum as
//! `np.bincount` makes it, from zero -- and added as `face_normal_sums` adds them,
//! `(corner 0 + corner 1) + corner 2`; every other sum is written out in the reference.

/// Per vertex, the sum of its triangles' `(b - a) x (c - a)` (`face_normal_sums`).
fn face_normal_sums(p: &[f64], tris: &[i64], out: &mut [f64]) {
    let n = p.len() / 3;
    let mut s = vec![0.0f64; 9 * n]; // (corner, vertex, axis)
    for f in tris.chunks_exact(3) {
        let (ia, ib, ic) = (3 * f[0] as usize, 3 * f[1] as usize, 3 * f[2] as usize);
        let (ux, uy, uz) = (p[ib] - p[ia], p[ib + 1] - p[ia + 1], p[ib + 2] - p[ia + 2]);
        let (vx, vy, vz) = (p[ic] - p[ia], p[ic + 1] - p[ia + 1], p[ic + 2] - p[ia + 2]);
        let fx = uy * vz - uz * vy;
        let fy = uz * vx - ux * vz;
        let fz = ux * vy - uy * vx;
        for (corner, &v) in f.iter().enumerate() {
            let o = 3 * (corner * n + v as usize);
            s[o] += fx;
            s[o + 1] += fy;
            s[o + 2] += fz;
        }
    }
    for k in 0..3 * n {
        out[k] = (s[k] + s[3 * n + k]) + s[6 * n + k];
    }
}

/// Each rest normal turned by the smallest rotation taking the vertex's rest area-weighted
/// normal (`frame`, unit) to its posed one; where the surface turned over (`c <= -0.95`),
/// the posed area-weighted normal itself. Unit length, as float32.
pub fn surface_normals(
    frame: &[f64],
    posed: &[f64],
    tris: &[i64],
    normals: &[f32],
    out: &mut [f32],
) {
    let n = posed.len() / 3;
    let mut b = vec![0.0f64; 3 * n];
    face_normal_sums(posed, tris, &mut b);
    for i in 0..n {
        let (ax, ay, az) = (frame[3 * i], frame[3 * i + 1], frame[3 * i + 2]);
        let (mut bx, mut by, mut bz) = (b[3 * i], b[3 * i + 1], b[3 * i + 2]);
        let length = ((bx * bx + by * by) + bz * bz).sqrt();
        if length > 0.0 {
            bx /= length;
            by /= length;
            bz /= length;
        } else {
            (bx, by, bz) = (ax, ay, az);
        }
        let xx = normals[3 * i] as f64;
        let xy = normals[3 * i + 1] as f64;
        let xz = normals[3 * i + 2] as f64;
        let c = (ax * bx + ay * by) + az * bz;
        let (vx, vy, vz) = (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
        let d = 1.0 + c;
        let d = if d.is_nan() || d > 1e-12 { d } else { 1e-12 }; // (np.maximum keeps a NaN)
        let s = ((vx * xx + vy * xy) + vz * xz) / d;
        let (ox, oy, oz) = if c > -0.95 {
            (
                (c * xx + (vy * xz - vz * xy)) + vx * s,
                (c * xy + (vz * xx - vx * xz)) + vy * s,
                (c * xz + (vx * xy - vy * xx)) + vz * s,
            )
        } else {
            (bx, by, bz)
        };
        let norm = ((ox * ox + oy * oy) + oz * oz).sqrt();
        let div = if norm > 0.0 { norm } else { 1.0 };
        out[3 * i] = (ox / div) as f32;
        out[3 * i + 1] = (oy / div) as f32;
        out[3 * i + 2] = (oz / div) as f32;
    }
}
