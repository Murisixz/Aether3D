//! Sparse LDL^T (`aether3d/uv/sparse.py`): symbolic analysis, up-looking numeric
//! factorization and solves.
//!
//! The numeric factorization and the solves perform the NumPy code's floating-point
//! operations in the same order (`_factor_numpy`, `_solve_numpy`): row `k` scatters the
//! upper part of column `k` into `y`, then for every column `i` of row `k`'s pattern in
//! ascending order subtracts `lx * y_i` from `y` at the rows already in column `i`, and
//! accumulates `d_k -= l_ki * y_i`. The solves go forward by columns of `L` and backward
//! by its rows. The results are therefore bitwise NumPy's.
//!
//! The symbolic analysis (elimination tree, column counts and patterns) is integer work:
//! the patterns are sets, listed in ascending order, so any algorithm gives the same
//! arrays; this one walks row subtrees (Davis, Algorithm 849).

/// Elimination tree `parent` and the column counts `lnz` of `L` for the upper triangle
/// (`ap`, `ai`: CSC, rows `<= k` in column `k`). Returns `Err` with a message on bad input.
pub fn etree(n: usize, ap: &[i64], ai: &[i64], parent: &mut [i64], lnz: &mut [i64]) {
    let mut flag = vec![usize::MAX; n];
    for k in 0..n {
        parent[k] = -1;
        flag[k] = k;
        lnz[k] = 0;
        for &a in &ai[ap[k] as usize..ap[k + 1] as usize] {
            let mut i = a as usize;
            while i < k && flag[i] != k {
                if parent[i] == -1 {
                    parent[i] = k as i64;
                }
                lnz[i] += 1;
                flag[i] = k;
                i = parent[i] as usize;
            }
        }
    }
}

/// The row indices `li` of every column of `L` (ascending), given `lp` from the counts.
pub fn pattern(n: usize, ap: &[i64], ai: &[i64], parent: &[i64], lp: &[i64], li: &mut [i64]) {
    let mut flag = vec![usize::MAX; n];
    let mut next: Vec<usize> = lp[..n].iter().map(|&v| v as usize).collect();
    for k in 0..n {
        flag[k] = k;
        for &a in &ai[ap[k] as usize..ap[k + 1] as usize] {
            let mut i = a as usize;
            while i < k && flag[i] != k {
                li[next[i]] = k as i64;
                next[i] += 1;
                flag[i] = k;
                i = parent[i] as usize;
            }
        }
    }
}

/// `_factor_numpy`. Returns the first row whose pivot is not positive, or -1.
#[allow(clippy::too_many_arguments)]
pub fn factor(
    n: usize,
    ap: &[i64],
    ai: &[i64],
    ax: &[f64],
    lp: &[i64],
    li: &[i64],
    rp: &[i64],
    rj: &[i64],
    lx: &mut [f64],
    d: &mut [f64],
) -> i64 {
    let mut y = vec![0.0f64; n];
    let mut filled: Vec<usize> = lp[..n].iter().map(|&v| v as usize).collect();
    for x in lx.iter_mut() {
        *x = 0.0;
    }
    let mut dl = vec![0.0f64; n];
    for k in 0..n {
        for p in ap[k] as usize..ap[k + 1] as usize {
            let r = ai[p] as usize;
            y[r] += ax[p];
        }
        let mut dk = y[k];
        y[k] = 0.0;
        for &col in &rj[rp[k] as usize..rp[k + 1] as usize] {
            let i = col as usize;
            let yi = y[i];
            y[i] = 0.0;
            let c0 = lp[i] as usize;
            let c1 = filled[i];
            for p in c0..c1 {
                let r = li[p] as usize;
                y[r] -= lx[p] * yi;
            }
            let lki = yi / dl[i];
            dk -= lki * yi;
            lx[c1] = lki;
            filled[i] = c1 + 1;
        }
        // `not dk > 0.0` also catches NaN, as in Python.
        if dk.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return k as i64;
        }
        dl[k] = dk;
    }
    d.copy_from_slice(&dl);
    -1
}

/// `_solve_numpy` on `x` (n rows of m right-hand sides, row-major), in place.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    n: usize,
    m: usize,
    lp: &[i64],
    li: &[i64],
    lx: &[f64],
    rp: &[i64],
    rj: &[i64],
    rpos: &[i64],
    d: &[f64],
    x: &mut [f64],
) {
    let mut xj = vec![0.0f64; m];
    for j in 0..n {
        xj.copy_from_slice(&x[j * m..(j + 1) * m]);
        for p in lp[j] as usize..lp[j + 1] as usize {
            let r = li[p] as usize;
            let l = lx[p];
            for c in 0..m {
                x[r * m + c] -= l * xj[c];
            }
        }
    }
    for j in 0..n {
        for c in 0..m {
            x[j * m + c] /= d[j];
        }
    }
    for i in (0..n).rev() {
        xj.copy_from_slice(&x[i * m..(i + 1) * m]);
        for q in rp[i] as usize..rp[i + 1] as usize {
            let r = rj[q] as usize;
            let l = lx[rpos[q] as usize];
            for c in 0..m {
                x[r * m + c] -= l * xj[c];
            }
        }
    }
}

/// Checks the CSC structure of an upper triangle: pointers monotone from 0 to `nnz`,
/// every row index in `[0, k]` within column `k`.
pub fn check_upper(n: usize, ap: &[i64], ai: &[i64]) -> Result<(), String> {
    if ap.len() != n + 1 || ap[0] != 0 || ap[n] as usize != ai.len() {
        return Err("ap: not the column pointers of ai".into());
    }
    for k in 0..n {
        if ap[k + 1] < ap[k] {
            return Err(format!("ap: decreasing at column {k}"));
        }
        for &row in &ai[ap[k] as usize..ap[k + 1] as usize] {
            if row < 0 || row as usize > k {
                return Err(format!("ai: row {row} out of range in column {k}"));
            }
        }
    }
    Ok(())
}

/// Checks a pointer array `ptr` (n + 1 entries, monotone, ending at `len`) and that the
/// indices of each segment lie in `[lo(k), hi(k))`.
pub fn check_segments(
    name: &str,
    n: usize,
    ptr: &[i64],
    idx: &[i64],
    bounds: impl Fn(usize) -> (i64, i64),
) -> Result<(), String> {
    if ptr.len() != n + 1 || ptr[0] != 0 || ptr[n] as usize != idx.len() {
        return Err(format!("{name}: bad pointers"));
    }
    for k in 0..n {
        if ptr[k + 1] < ptr[k] {
            return Err(format!("{name}: decreasing pointers at {k}"));
        }
        let (lo, hi) = bounds(k);
        for &v in &idx[ptr[k] as usize..ptr[k + 1] as usize] {
            if v < lo || v >= hi {
                return Err(format!("{name}: index {v} out of range at {k}"));
            }
        }
    }
    Ok(())
}
