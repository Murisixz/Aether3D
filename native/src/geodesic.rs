//! Geodesic voxel distances (`aether3d.animation.voxels.geodesic`): exact integer
//! shortest paths inside a solid voxel grid on the 26-neighbourhood with the 3-4-5
//! chamfer weights, from one set of source voxels per bone, by Dial's bucket queue (six
//! cyclic buckets: the largest step is 5). Integer distances are unique, so the result
//! is the NumPy implementation's whatever the visiting order; bones are split between
//! threads in fixed ranges.

/// `sid`: flat grid index (x fastest) -> solid index or -1; the grid has an empty border.
/// `ptr` (b + 1) and `src` give each bone's source voxels (flat indices); `queries` are
/// flat indices; `out` is (queries, bones) row-major, -1 beyond `cutoff` or unreachable.
#[allow(clippy::too_many_arguments)]
pub fn geodesic_voxels(
    sid: &[i32],
    nx: usize,
    ny: usize,
    ptr: &[i64],
    src: &[i64],
    queries: &[i64],
    cutoff: i64,
    out: &mut [i32],
    threads: usize,
) {
    let bones = ptr.len() - 1;
    let n_solid = sid.iter().filter(|&&s| s >= 0).count();
    let mut solid_flat = vec![0usize; n_solid];
    for (f, &s) in sid.iter().enumerate() {
        if s >= 0 {
            solid_flat[s as usize] = f;
        }
    }
    let mut offsets: Vec<(i64, i64)> = Vec::with_capacity(26);
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nonzero = (dx != 0) as i64 + (dy != 0) as i64 + (dz != 0) as i64;
                let off = dx + nx as i64 * (dy + ny as i64 * dz);
                offsets.push((off, 2 + nonzero));
            }
        }
    }
    let q = queries.len();
    let run = |bone: usize, dist: &mut Vec<i64>, column: &mut [i32]| {
        for d in dist.iter_mut() {
            *d = i64::MAX;
        }
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 6];
        let mut pending = 0usize;
        for &f in &src[ptr[bone] as usize..ptr[bone + 1] as usize] {
            let s = sid[f as usize];
            if s >= 0 && dist[s as usize] != 0 {
                dist[s as usize] = 0;
                buckets[0].push(s as u32);
                pending += 1;
            }
        }
        let mut d: i64 = 0;
        while pending > 0 && d <= cutoff {
            let slot = (d % 6) as usize;
            let front = std::mem::take(&mut buckets[slot]);
            pending -= front.len();
            for &s in &front {
                if dist[s as usize] != d {
                    continue;
                }
                let f = solid_flat[s as usize] as i64;
                for &(off, w) in &offsets {
                    let nd = d + w;
                    if nd > cutoff {
                        continue;
                    }
                    let nb = sid[(f + off) as usize];
                    if nb >= 0 && dist[nb as usize] > nd {
                        dist[nb as usize] = nd;
                        buckets[(nd % 6) as usize].push(nb as u32);
                        pending += 1;
                    }
                }
            }
            d += 1;
        }
        for (k, &qf) in queries.iter().enumerate() {
            let s = sid[qf as usize];
            column[k] = if s >= 0 && dist[s as usize] <= cutoff {
                dist[s as usize] as i32
            } else {
                -1
            };
        }
    };
    let threads = threads.max(1).min(bones.max(1));
    let mut columns = vec![0i32; bones * q];
    if threads == 1 {
        let mut dist = vec![i64::MAX; n_solid];
        for (b, col) in columns.chunks_mut(q.max(1)).enumerate().take(bones) {
            run(b, &mut dist, col);
        }
    } else {
        let per = bones.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [i32] = &mut columns;
            let mut start = 0;
            while start < bones {
                let take = per.min(bones - start);
                let (head, tail) = rest.split_at_mut(take * q);
                rest = tail;
                let s = start;
                let run = &run;
                scope.spawn(move || {
                    let mut dist = vec![i64::MAX; n_solid];
                    for (i, col) in head.chunks_mut(q.max(1)).enumerate().take(take) {
                        run(s + i, &mut dist, col);
                    }
                });
                start += take;
            }
        });
    }
    for b in 0..bones {
        for k in 0..q {
            out[k * bones + b] = columns[b * q + k];
        }
    }
}
