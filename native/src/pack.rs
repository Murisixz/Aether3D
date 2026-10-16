//! Atlas placement by the horizon rule (`aether3d/uv/pack.py`, `_place_numpy`).
//!
//! Charts are dropped, in the given order, onto the horizon (per atlas column, the first
//! free row). For every orientation and column, a chart rests where one of its columns
//! first touches the horizon; the placement with the lowest peak wins, then the least
//! area wasted below the chart, then the leftmost column, then the first orientation.
//! Integer arithmetic only, so the choices are exactly NumPy's.

pub const ORIENTATIONS: usize = 4;

/// Places the charts; `orient` and `pos` (x, y) receive each chart's placement. Returns the
/// peak, or `None` when a chart fits in no orientation.
pub fn place(
    width: usize,
    bot: &[i64],
    top: &[i64],
    offsets: &[i64],
    order: &[i64],
    orient: &mut [i64],
    pos: &mut [i64],
) -> Option<i64> {
    let mut horizon = vec![0i64; width];
    let mut peak = 0i64;
    for &c in order {
        let c = c as usize;
        // (peak, waste, x, o, y)
        let mut best: Option<(i64, i64, usize, usize, i64)> = None;
        for o in 0..ORIENTATIONS {
            let k = c * ORIENTATIONS + o;
            let (s, e) = (offsets[k] as usize, offsets[k + 1] as usize);
            let w = e - s;
            if w == 0 || w > width {
                continue;
            }
            let b = &bot[s..e];
            let t = &top[s..e];
            let tmax = t.iter().copied().max().unwrap_or(-1);
            let mut local: Option<(i64, i64, usize, i64)> = None;
            for x in 0..=(width - w) {
                let mut need = i64::MIN;
                for i in 0..w {
                    if b[i] >= 0 {
                        need = need.max(horizon[x + i] - b[i]);
                    }
                }
                let y = need.max(0);
                let mut waste = 0i64;
                for i in 0..w {
                    if b[i] >= 0 {
                        waste += y + b[i] - horizon[x + i];
                    }
                }
                let pk = (y + tmax).max(peak);
                let better = match local {
                    None => true,
                    Some((lp, lw, _, _)) => pk < lp || (pk == lp && waste < lw),
                };
                if better {
                    local = Some((pk, waste, x, y));
                }
            }
            if let Some((pk, waste, x, y)) = local {
                let entry = (pk, waste, x, o, y);
                let better = match best {
                    None => true,
                    Some(bst) => {
                        (entry.0, entry.1, entry.2, entry.3) < (bst.0, bst.1, bst.2, bst.3)
                    }
                };
                if better {
                    best = Some(entry);
                }
            }
        }
        let (_, _, x, o, y) = best?;
        let k = c * ORIENTATIONS + o;
        let (s, e) = (offsets[k] as usize, offsets[k + 1] as usize);
        let mut tmax = -1i64;
        for i in 0..(e - s) {
            if bot[s + i] >= 0 {
                let h = &mut horizon[x + i];
                *h = (*h).max(y + top[s + i]);
            }
            tmax = tmax.max(top[s + i]);
        }
        peak = peak.max(y + tmax);
        orient[c] = o as i64;
        pos[2 * c] = x as i64;
        pos[2 * c + 1] = y;
    }
    Some(peak)
}

/// Checks the profiles: `offsets` monotone over `bot`/`top`, `order` a permutation, each
/// column empty (`-1`, `-1`) or `0 <= bot < top`.
pub fn check(bot: &[i64], top: &[i64], offsets: &[i64], order: &[i64]) -> Result<usize, String> {
    if bot.len() != top.len() {
        return Err("bot and top must have the same length".into());
    }
    if offsets.is_empty() || (offsets.len() - 1) % ORIENTATIONS != 0 {
        return Err("offsets: need 4 orientations per chart, plus one".into());
    }
    if offsets[0] != 0 || *offsets.last().unwrap() as usize != bot.len() {
        return Err("offsets: must run from 0 to the profiles' length".into());
    }
    if offsets.windows(2).any(|w| w[1] < w[0]) {
        return Err("offsets: decreasing".into());
    }
    let count = (offsets.len() - 1) / ORIENTATIONS;
    if order.len() != count {
        return Err("order: one entry per chart".into());
    }
    let mut seen = vec![false; count];
    for &c in order {
        if c < 0 || c as usize >= count || seen[c as usize] {
            return Err("order: not a permutation of the charts".into());
        }
        seen[c as usize] = true;
    }
    for (&b, &t) in bot.iter().zip(top) {
        let empty = b == -1 && t == -1;
        if !(empty || (b >= 0 && t > b)) {
            return Err(format!("profile column ({b}, {t}) is invalid"));
        }
    }
    Ok(count)
}
