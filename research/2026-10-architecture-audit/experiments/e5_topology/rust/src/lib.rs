//! Edge topology like EditableMesh._add_edge_use; std HashMap with the default SipHash or a fast multiplicative hasher.
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct Mix(u64);
impl Hasher for Mix {
    fn finish(&self) -> u64 { let mut k = self.0; k ^= k >> 33; k = k.wrapping_mul(0xff51afd7ed558ccd); k ^ (k >> 33) }
    fn write(&mut self, _: &[u8]) { unreachable!() }
    fn write_u64(&mut self, v: u64) { self.0 = v; }
}

fn build<S: std::hash::BuildHasher + Default>(loops: &[i64], f: usize, edge_of_use: &mut [i64], lo_out: &mut [i64], hi_out: &mut [i64],
                                                start: &mut [i64], faces: &mut [i64]) -> i64 {
    let mut map: HashMap<u64, i64, S> = HashMap::with_capacity_and_hasher(3 * f / 2 + 16, S::default());
    let mut e = 0i64;
    for t in 0..f {
        for k in 0..3 {
            let a = loops[3 * t + k]; let b = loops[3 * t + (k + 1) % 3];
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let key = ((lo as u64) << 32) | hi as u64;
            let id = *map.entry(key).or_insert_with(|| { lo_out[e as usize] = lo; hi_out[e as usize] = hi; e += 1; e - 1 });
            edge_of_use[3 * t + k] = id;
        }
    }
    let ne = e as usize;
    start[..=ne].fill(0);
    for u in 0..3 * f { start[edge_of_use[u] as usize + 1] += 1; }
    for i in 0..ne { start[i + 1] += start[i]; }
    let mut fill: Vec<i64> = start[..ne].to_vec();
    for u in 0..3 * f { let x = edge_of_use[u] as usize; faces[fill[x] as usize] = (u / 3) as i64; fill[x] += 1; }
    e
}

/// # Safety: pointers to arrays of the documented sizes (3f uses, up to 3f edges, 3f+1 starts).
#[no_mangle]
pub unsafe extern "C" fn build_edges(f: i64, loops: *const i64, edge_of_use: *mut i64, lo: *mut i64, hi: *mut i64,
                                     start: *mut i64, faces: *mut i64, fast_hash: i32) -> i64 {
    let f = f as usize; let m = 3 * f;
    let loops = std::slice::from_raw_parts(loops, m);
    let (u, l, h, s, fa) = (std::slice::from_raw_parts_mut(edge_of_use, m), std::slice::from_raw_parts_mut(lo, m),
                            std::slice::from_raw_parts_mut(hi, m), std::slice::from_raw_parts_mut(start, m + 1),
                            std::slice::from_raw_parts_mut(faces, m));
    if fast_hash != 0 { build::<BuildHasherDefault<Mix>>(loops, f, u, l, h, s, fa) }
    else { build::<std::collections::hash_map::RandomState>(loops, f, u, l, h, s, fa) }
}
