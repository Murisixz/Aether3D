//! PNG row filters: the encoder's choice of filter (`aether3d/rendering/png.py`,
//! `_filter_rows`) and the decoder's reconstruction (`aether3d/materials/images.py`,
//! `_unfilter`). Integer arithmetic, the same rules: bytes identical to the NumPy code's.

#[inline(always)]
fn paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

#[inline(always)]
fn predict(kind: u8, a: i32, b: i32, c: i32) -> i32 {
    match kind {
        1 => a,
        2 => b,
        3 => (a + b) >> 1,
        4 => paeth(a, b, c),
        _ => 0,
    }
}

/// |signed byte|: the cost of a filtered byte when choosing a row's filter.
#[inline(always)]
fn magnitude(v: u8) -> u64 {
    let v = v as u64;
    if v < 128 {
        v
    } else {
        256 - v
    }
}

/// Filtered scanlines of `rows` (height x stride bytes): each output row is the filter
/// type with the smallest sum of |signed byte| (ties: the lower type), then the row
/// filtered with it.
pub fn filter_rows(rows: &[u8], height: usize, stride: usize, bpp: usize, out: &mut [u8]) {
    let zeros = vec![0u8; stride];
    for r in 0..height {
        let row = &rows[r * stride..(r + 1) * stride];
        let up: &[u8] = if r == 0 {
            &zeros
        } else {
            &rows[(r - 1) * stride..r * stride]
        };
        let mut cost = [0u64; 5];
        for i in 0..stride {
            let x = row[i] as i32;
            let a = if i >= bpp { row[i - bpp] as i32 } else { 0 };
            let b = up[i] as i32;
            let c = if i >= bpp { up[i - bpp] as i32 } else { 0 };
            for kind in 0..5u8 {
                cost[kind as usize] += magnitude((x - predict(kind, a, b, c)) as u8);
            }
        }
        let mut best = 0usize;
        for kind in 1..5 {
            if cost[kind] < cost[best] {
                best = kind;
            }
        }
        let dst = &mut out[r * (stride + 1)..(r + 1) * (stride + 1)];
        dst[0] = best as u8;
        for i in 0..stride {
            let x = row[i] as i32;
            let a = if i >= bpp { row[i - bpp] as i32 } else { 0 };
            let b = up[i] as i32;
            let c = if i >= bpp { up[i - bpp] as i32 } else { 0 };
            dst[1 + i] = (x - predict(best as u8, a, b, c)) as u8;
        }
    }
}

/// The scanlines of `block` (height rows of 1 + stride bytes: the filter type, then the
/// filtered bytes; types are 0-4, checked by the caller) reconstructed into `out`.
pub fn unfilter(block: &[u8], height: usize, stride: usize, bpp: usize, out: &mut [u8]) {
    for r in 0..height {
        let kind = block[r * (stride + 1)];
        let src = &block[r * (stride + 1) + 1..(r + 1) * (stride + 1)];
        let (done, rest) = out.split_at_mut(r * stride);
        let up: Option<&[u8]> = if r == 0 {
            None
        } else {
            Some(&done[(r - 1) * stride..])
        };
        let dst = &mut rest[..stride];
        for i in 0..stride {
            let a = if i >= bpp { dst[i - bpp] as i32 } else { 0 };
            let b = up.map_or(0, |u| u[i] as i32);
            let c = if i >= bpp {
                up.map_or(0, |u| u[i - bpp] as i32)
            } else {
                0
            };
            dst[i] = (src[i] as i32 + predict(kind, a, b, c)) as u8;
        }
    }
}
