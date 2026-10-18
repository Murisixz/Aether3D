//! The sequential sweeps of the regional hydrology stage (`aether3d.worlds.hydrology`):
//! Priority-Flood with an epsilon, drainage accumulation, the implicit stream-power step
//! and sediment transport. Each performs its Python twin's operations in the same order --
//! the same heap order (height, then index), the same visiting order of nodes, the same
//! floating-point expressions without fused multiply-adds -- so the results are bitwise
//! identical (`tests/test_worlds_hydrology.py`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

const OFFSETS: [(i64, i64); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// `math.nextafter(x, inf)` for finite `x`.
fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// A heap entry ordered like Python's `(height, index)` tuples, smallest first.
#[derive(Clone, Copy)]
struct Entry(f64, usize);

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap pops its maximum, Python's heapq its minimum.
        match other.0.partial_cmp(&self.0) {
            Some(Ordering::Equal) | None => other.1.cmp(&self.1),
            Some(o) => o,
        }
    }
}

/// Depression filling (Barnes, Lehman and Mulla 2014, with an epsilon): `out` starts as a
/// copy of `h`; outlets seed the queue in index order.
pub fn priority_flood(h: &[f64], outlet: &[u8], nx: usize, ny: usize, out: &mut [f64]) {
    out.copy_from_slice(h);
    let n = nx * ny;
    let mut closed = vec![false; n];
    let mut heap = BinaryHeap::new();
    for c in 0..n {
        if outlet[c] != 0 {
            closed[c] = true;
            heap.push(Entry(h[c], c));
        }
    }
    while let Some(Entry(z, c)) = heap.pop() {
        let i = (c % nx) as i64;
        let j = (c / nx) as i64;
        for (di, dj) in OFFSETS {
            let a = i + di;
            let b = j + dj;
            if a < 0 || a >= nx as i64 || b < 0 || b >= ny as i64 {
                continue;
            }
            let m = (b as usize) * nx + a as usize;
            if closed[m] {
                continue;
            }
            closed[m] = true;
            let mut v = out[m];
            if v <= z {
                v = next_up(z);
                out[m] = v;
            }
            heap.push(Entry(v, m));
        }
    }
}

/// Upstream sums: `area[recv[c]] += area[c]` for `c` in reversed `order`.
pub fn accumulate(order: &[i64], recv: &[i64], area: &mut [i64]) {
    for &c in order.iter().rev() {
        let c = c as usize;
        let r = recv[c] as usize;
        if r != c {
            area[r] += area[c];
        }
    }
}

/// One implicit stream-power step (n = 1, m = 0.5); `out` starts as a copy of `h`.
#[allow(clippy::too_many_arguments)]
pub fn stream_power(
    order: &[i64],
    recv: &[i64],
    dist: &[f64],
    area: &[f64],
    h: &[f64],
    filled: &[f64],
    kdt: f64,
    uplift: &[f64],
    out: &mut [f64],
) {
    for &c in order {
        let c = c as usize;
        let r = recv[c] as usize;
        if r == c {
            continue;
        }
        let hc = h[c] + uplift[c];
        if filled[c] > h[c] {
            out[c] = hc;
            continue;
        }
        let base = if filled[r] > h[r] { filled[r] } else { out[r] };
        let f = kdt * area[c].sqrt() / dist[c];
        out[c] = (hc + f * base) / (1.0 + f);
    }
}

/// Sediment transport and deposition (in place on `after`): the flux `q` passing a node
/// leaves the depth `G q / A` there (at most back to its height before the step plus
/// `aggradation`); lakes keep everything up to their level; the coast keeps a share.
#[allow(clippy::too_many_arguments)]
pub fn sediment(
    order: &[i64],
    recv: &[i64],
    area: &[f64],
    before: &[f64],
    after: &mut [f64],
    filled: &[f64],
    sea: f64,
    cell: f64,
    deposition: f64,
    aggradation: f64,
) {
    let a2 = cell * cell;
    let mut qs = vec![0.0f64; after.len()];
    let mut low = vec![f64::INFINITY; after.len()];
    for &c in order.iter().rev() {
        let c = c as usize;
        let e = before[c] - after[c];
        let mut q = qs[c] + if e > 0.0 { e * a2 } else { 0.0 };
        let r = recv[c] as usize;
        if q <= 0.0 {
            if r != c && after[c] < low[r] {
                low[r] = after[c];
            }
            continue;
        }
        if r == c {
            if after[c] < sea {
                let room = ((sea - 0.5) - after[c]) * a2;
                let mut d = q * deposition;
                if d > room {
                    d = room;
                }
                if d > 0.0 {
                    after[c] += d / a2;
                }
            }
            continue;
        }
        let d = if filled[c] > before[c] {
            let room = (filled[c] - after[c]) * a2;
            if q < room {
                q
            } else {
                room
            }
        } else {
            let mut d = deposition * q / area[c] * a2;
            let mut top = (if before[c] > after[c] {
                before[c]
            } else {
                after[c]
            }) + aggradation;
            if low[c] < top {
                top = low[c];
            }
            let room = (top - after[c]) * a2;
            if d > room {
                d = room;
            }
            d
        };
        if d > 0.0 {
            after[c] += d / a2;
            q -= d;
        }
        qs[r] += q;
        if after[c] < low[r] {
            low[r] = after[c];
        }
    }
}

/// Talus relaxation (`worlds.hydrology.thermal`): `iterations` Jacobi sweeps from `h` into
/// `out`, rows split between `threads` threads (each node's update reads only the previous
/// sweep, so results do not depend on the thread count).
#[allow(clippy::too_many_arguments)]
pub fn thermal(
    h: &[f64],
    nx: usize,
    ny: usize,
    cell: f64,
    talus_tan: f64,
    rate: f64,
    iterations: usize,
    out: &mut [f64],
    threads: usize,
) {
    let n = nx * ny;
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut dist = [0.0f64; 8];
    for (k, (di, dj)) in OFFSETS.iter().enumerate() {
        dist[k] = cell * (if *di != 0 && *dj != 0 { sqrt2 } else { 1.0 });
    }
    let mut thr = [0.0f64; 8];
    for k in 0..8 {
        thr[k] = talus_tan * dist[k];
    }
    let half = rate * 0.5;
    let mut g = h.to_vec();
    let mut flows = vec![0.0f64; 8 * n];
    let mut moved = vec![0.0f64; n];
    let threads = threads.max(1).min(ny.max(1));
    let rows_per = ny.div_ceil(threads).max(1);
    for _ in 0..iterations {
        // Pass 1: each node's excess drops, the amount it moves and its outgoing flows.
        {
            let g = &g;
            let thr = &thr;
            std::thread::scope(|scope| {
                let mut fl_rest: &mut [f64] = &mut flows;
                let mut mv_rest: &mut [f64] = &mut moved;
                let mut j0 = 0usize;
                while j0 < ny {
                    let j1 = (j0 + rows_per).min(ny);
                    let (fl, fr) = fl_rest.split_at_mut(8 * (j1 - j0) * nx);
                    let (mv, mr) = mv_rest.split_at_mut((j1 - j0) * nx);
                    fl_rest = fr;
                    mv_rest = mr;
                    scope.spawn(move || {
                        for j in j0..j1 {
                            for i in 0..nx {
                                let c = j * nx + i;
                                let local = (j - j0) * nx + i;
                                let gc = g[c];
                                let mut ex = [0.0f64; 8];
                                let mut total = 0.0f64;
                                let mut dmax = 0.0f64;
                                for (k, (di, dj)) in OFFSETS.iter().enumerate() {
                                    let a = (i as i64 + di).clamp(0, nx as i64 - 1) as usize;
                                    let b = (j as i64 + dj).clamp(0, ny as i64 - 1) as usize;
                                    let e = (gc - g[b * nx + a]) - thr[k];
                                    let e = if e > 0.0 { e } else { 0.0 };
                                    ex[k] = e;
                                    total += e;
                                    if e > dmax {
                                        dmax = e;
                                    }
                                }
                                let m = half * dmax;
                                mv[local] = m;
                                let share = if total > 0.0 { m / total } else { 0.0 };
                                for k in 0..8 {
                                    fl[8 * local + k] = ex[k] * share;
                                }
                            }
                        }
                    });
                    j0 = j1;
                }
            });
        }
        // Pass 2: new heights = (old - moved) + incoming, summed in neighbour order.
        let mut next = vec![0.0f64; n];
        {
            let g = &g;
            let flows = &flows;
            let moved = &moved;
            std::thread::scope(|scope| {
                let mut rest: &mut [f64] = &mut next;
                let mut j0 = 0usize;
                while j0 < ny {
                    let j1 = (j0 + rows_per).min(ny);
                    let (part, r) = rest.split_at_mut((j1 - j0) * nx);
                    rest = r;
                    scope.spawn(move || {
                        for j in j0..j1 {
                            for i in 0..nx {
                                let c = j * nx + i;
                                let mut inc = 0.0f64;
                                for (k, (di, dj)) in OFFSETS.iter().enumerate() {
                                    let si = i as i64 - di;
                                    let sj = j as i64 - dj;
                                    if si < 0 || si >= nx as i64 || sj < 0 || sj >= ny as i64 {
                                        continue;
                                    }
                                    let s = sj as usize * nx + si as usize;
                                    inc += flows[8 * s + k];
                                }
                                part[(j - j0) * nx + i] = (g[c] - moved[c]) + inc;
                            }
                        }
                    });
                    j0 = j1;
                }
            });
        }
        g = next;
    }
    out.copy_from_slice(&g);
}
