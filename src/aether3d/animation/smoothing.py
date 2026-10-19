"""Automatic handles in the two Blender-compatible smoothing modes of an F-curve.

A curve's ``smoothing`` decides how automatic handles (``auto``, ``auto_clamped``) are
computed. ``"local"`` is Aether3D's own rule (:func:`.curves._recompute_handles`: the slope
of the parabola through a key and its neighbours, third-of-interval extents, Fritsch-Carlson
clamping). This module implements the other two, which reproduce what Blender 5.2 computes
for the same keys (measured with headless ``bpy`` on uneven keys, extrema, plateaus,
cycles, vector and free neighbours; the reference data is ``tests/data/blender_curves``).
They were written from those measurements and the mathematics of cubic splines, not from
Blender's code.

``"blender"`` -- Blender's automatic handles without smoothing (*Auto Handle Smoothing:
None*). A key's slope is the mean of the secants to its two neighbours; each handle's
time extent is the interval to its neighbour divided by 2.5614, every interval first
limited to five times the other one (the direction keeps the unlimited secants). Clamped
keys keep each handle's height within the value difference to the neighbour on its side
(no overshoot) and of the same sign (no reversal), with one slope for both handles: flat at
extrema and plateaus. Without a neighbour a key mirrors the one it has (the secant); with
``constant`` (or ``mirror``) extrapolation the end keys are flat instead.

``"continuous"`` -- Blender's *Continuous Acceleration*, its default for new curves: the
keys of every run of consecutive automatic keys get third-of-interval extents (cubic
Hermite segments in time) and the slopes that make the curve's acceleration continuous at
each of them -- one tridiagonal linear system per run (a C2 cubic spline). Runs end at keys
with other handles: a free or aligned handle enters the equations with its own, exact
Bezier geometry (whatever its extent), and a vector handle next to a run becomes a free end
whose acceleration is zero (natural), as Blender does. The curve's ends are flat with
``constant`` or ``mirror`` extrapolation and natural with ``linear``; a cyclic curve (both
sides ``cycle`` or ``cycle_offset``) is solved as one periodic system across the seam, its
value change ignored (only derivatives must match). Clamped keys turn into bounds on their
slopes (the same box as above); a solution outside its bounds is clamped there and the
key locked, and the rest solved again, until every bound holds. Keys whose handles are only
partly automatic take the ``"blender"`` handles. Systems are solved by parallel cyclic
reduction: vectorized, ``+ - * /`` only, deterministic.
"""

from __future__ import annotations

import numpy as np

HANDLE_LENGTH = 2.5614
"""Blender's divisor of a neighbour interval for an automatic handle's time extent."""
LENGTH_RATIO = 5.0
"""Largest ratio of the two intervals a handle's extent is computed from."""

# Handle types (curves.HANDLE_TYPES order) and extrapolation codes (curves.EXTRAPOLATIONS).
_FREE, _ALIGNED, _VECTOR, _AUTO, _AUTO_CLAMPED = range(5)
_CONSTANT, _LINEAR, _CYCLE, _CYCLE_OFFSET, _MIRROR = range(5)


def is_cyclic(before: int, after: int) -> bool:
    """Blender's rule: handles wrap around only when both sides repeat."""
    return before in (_CYCLE, _CYCLE_OFFSET) and after in (_CYCLE, _CYCLE_OFFSET)


def _neighbours(t: np.ndarray, v: np.ndarray, cyclic: bool):
    """Previous and next neighbour (time, value) of every key, NaN where there is none.

    A jump partner (same time) is no neighbour. A cyclic curve's end keys take the key next
    to the other end, one period away and shifted by the value change across the keys."""
    n = t.shape[0]
    pt = np.full(n, np.nan)
    pv = np.full(n, np.nan)
    nt = np.full(n, np.nan)
    nv = np.full(n, np.nan)
    if n > 1:
        gap = np.diff(t) > 0
        pt[1:] = np.where(gap, t[:-1], np.nan)
        pv[1:] = np.where(gap, v[:-1], np.nan)
        nt[:-1] = np.where(gap, t[1:], np.nan)
        nv[:-1] = np.where(gap, v[1:], np.nan)
        span = float(t[-1] - t[0])
        if cyclic and span > 0.0:
            delta = float(v[-1] - v[0])
            if t[-2] < t[-1]:
                pt[0], pv[0] = t[-2] - span, v[-2] - delta
            if t[1] > t[0]:
                nt[-1], nv[-1] = t[1] + span, v[1] + delta
    return pt, pv, nt, nv


def blender_auto(
    t: np.ndarray, v: np.ndarray, before: int, after: int, clamped: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slopes and left/right time extents (positive) of Blender's automatic handles
    without smoothing; ``clamped`` marks the keys clamped against overshoot."""
    n = t.shape[0]
    cyclic = is_cyclic(before, after)
    pt, pv, nt, nv = _neighbours(t, v, cyclic)
    has_p = np.isfinite(pt)
    has_n = np.isfinite(nt)
    with np.errstate(invalid="ignore", divide="ignore"):
        dp = t - pt
        dn = nt - t
        sp = (v - pv) / dp
        sn = (nv - v) / dn
    # A missing neighbour mirrors the other one (a lone key: flat, 1/3 s extents).
    lone = ~has_p & ~has_n
    dp = np.where(has_p, dp, np.where(has_n, dn, 1.0))
    dn = np.where(has_n, dn, np.where(has_p, dp, 1.0))
    sp = np.where(has_p, sp, np.where(has_n, sn, 0.0))
    sn = np.where(has_n, sn, np.where(has_p, sp, 0.0))
    slope = 0.5 * (sp + sn)
    # Ends of a curve that does not repeat: flat unless extrapolated linearly.
    if not cyclic and n > 0:
        if before != _LINEAR and not has_p[0]:
            slope[0] = 0.0
        if after != _LINEAR and not has_n[-1]:
            slope[-1] = 0.0
    slope[lone] = 0.0
    lp = np.minimum(dp, LENGTH_RATIO * dn)
    ln = np.minimum(dn, LENGTH_RATIO * dp)
    left = np.where(lone, 1.0 / 3.0, lp / HANDLE_LENGTH)
    right = np.where(lone, 1.0 / 3.0, ln / HANDLE_LENGTH)
    if np.any(clamped):
        lo, hi = _bounds(v, pv, nv, has_p, has_n, left, right)
        slope = np.where(clamped, np.minimum(np.maximum(slope, lo), hi), slope)
    return slope, left, right


def _bounds(v, pv, nv, has_p, has_n, left, right) -> tuple[np.ndarray, np.ndarray]:
    """Slope bounds of clamped keys: each handle's height within the value difference to
    its neighbour and of its sign (zero difference: flat)."""
    n = v.shape[0]
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    for has, dy, ext in ((has_p, v - pv, left), (has_n, nv - v, right)):
        dy = np.where(has, dy, np.nan)
        up = has & (dy > 0)
        down = has & (dy < 0)
        flat = has & (dy == 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lim = dy / ext
        lo = np.where(up, np.maximum(lo, 0.0), lo)
        hi = np.where(up, np.minimum(hi, lim), hi)
        lo = np.where(down, np.maximum(lo, lim), lo)
        hi = np.where(down, np.minimum(hi, 0.0), hi)
        lo = np.where(flat, 0.0, lo)
        hi = np.where(flat, 0.0, hi)
    return lo, hi


# --------------------------------------------------------------------------------------- #
# Continuous acceleration
# --------------------------------------------------------------------------------------- #


def _accel_terms(dy, dur, x1, h1, x2, h2):
    """Second time derivatives at the start (``A``) and end (``B``) of cubic Bezier segments
    with control points (0, 0), (x1, h1), (dur + x2, dy + h2), (dur, dy)."""
    xs1 = 3.0 * x1
    ys1 = 3.0 * h1
    xss0 = 6.0 * (dur + x2 - 2.0 * x1)
    yss0 = 6.0 * (dy - 2.0 * h1 + h2)
    xs2 = -3.0 * x2
    ys2 = -3.0 * h2
    xss1 = 6.0 * (x1 - dur - 2.0 * x2)
    yss1 = 6.0 * (h1 - dy - 2.0 * h2)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        a = (yss0 * xs1 - ys1 * xss0) / (xs1 * xs1 * xs1)
        b = (yss1 * xs2 - ys2 * xss1) / (xs2 * xs2 * xs2)
    return a, b


def _solve_tridiagonal(a, b, c, d) -> np.ndarray:
    """Parallel cyclic reduction of ``a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i``
    (``a_0 = c_{n-1} = 0``; independent systems may be concatenated with zero couplings)."""
    a = a.astype(np.float64).copy()
    b = b.astype(np.float64).copy()
    c = c.astype(np.float64).copy()
    d = d.astype(np.float64).copy()
    n = b.shape[0]
    s = 1
    while s < n:
        # Neighbour equations s away (outside: the identity row 0 = 0).
        bp = np.ones(n)
        ap = np.zeros(n)
        cp = np.zeros(n)
        dp = np.zeros(n)
        bn = np.ones(n)
        an = np.zeros(n)
        cn = np.zeros(n)
        dn = np.zeros(n)
        bp[s:], ap[s:], cp[s:], dp[s:] = b[:-s], a[:-s], c[:-s], d[:-s]
        bn[:-s], an[:-s], cn[:-s], dn[:-s] = b[s:], a[s:], c[s:], d[s:]
        alpha = -a / bp
        gamma = -c / bn
        a, c = alpha * ap, gamma * cn
        b = b + alpha * cp + gamma * an
        d = d + alpha * dp + gamma * dn
        s *= 2
    return d / b


def _solve_periodic(a, b, c, d) -> np.ndarray:
    """A periodic tridiagonal system (``a_0`` couples ``x_{n-1}``, ``c_{n-1}`` couples
    ``x_0``) by the Sherman-Morrison formula over two reductions."""
    n = b.shape[0]
    if n == 1:
        return d / (a + b + c)
    if n == 2:
        m = np.array([[b[0], a[0] + c[0]], [a[1] + c[1], b[1]]])
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        return np.array(
            [(d[0] * m[1, 1] - m[0, 1] * d[1]) / det, (m[0, 0] * d[1] - d[0] * m[1, 0]) / det]
        )
    alpha, beta = float(a[0]), float(c[-1])
    gamma = -float(b[0])
    bb = b.astype(np.float64).copy()
    bb[0] -= gamma
    bb[-1] -= alpha * beta / gamma
    aa = a.astype(np.float64).copy()
    cc = c.astype(np.float64).copy()
    aa[0] = 0.0
    cc[-1] = 0.0
    x = _solve_tridiagonal(aa, bb, cc, d)
    u = np.zeros(n)
    u[0] = gamma
    u[-1] = alpha
    z = _solve_tridiagonal(aa, bb, cc, u)
    factor = (x[0] + beta * x[-1] / gamma) / (1.0 + z[0] + beta * z[-1] / gamma)
    return x - factor * z


def continuous(
    t: np.ndarray,
    v: np.ndarray,
    lt: np.ndarray,
    rt: np.ndarray,
    ldt: np.ndarray,
    ldv: np.ndarray,
    rdt: np.ndarray,
    rdv: np.ndarray,
    before: int,
    after: int,
) -> None:
    """Replace the handles of every run of fully automatic keys (and of the vector handles
    next to one) with the continuous-acceleration solution, in place (see the module
    docstring). The arrays hold the curve's handles in the ``"blender"`` mode."""
    n = t.shape[0]
    auto_l = (lt == _AUTO) | (lt == _AUTO_CLAMPED)
    auto_r = (rt == _AUTO) | (rt == _AUTO_CLAMPED)
    run_key = auto_l & auto_r
    if n < 2 or not np.any(run_key):
        return
    cyclic = is_cyclic(before, after) and t[-1] > t[0]
    clamped = run_key & ((lt == _AUTO_CLAMPED) | (rt == _AUTO_CLAMPED))
    # The circle (cyclic) or line of keys: m positions, segments k -> k + 1.
    m = n - 1 if cyclic else n
    if cyclic and not (run_key[0] and run_key[-1]):
        run_key = run_key.copy()
        run_key[0] = run_key[-1] = False  # the seam key is automatic on both copies or not
    seg_count = m if cyclic else n - 1
    k0 = np.arange(seg_count)
    k1 = (k0 + 1) % m if cyclic else k0 + 1
    # Physical keys of each segment (the seam segment ends at the last key).
    pa = k0
    pb = np.where(k1 == 0, n - 1, k1) if cyclic else k1
    dur = t[pb] - t[pa]
    dy = v[pb] - v[pa]
    live = dur > 0.0
    pt, pv, nt, nv = _neighbours(t, v, cyclic)
    left3, right3 = _thirds(t, cyclic)
    lo_b, hi_b = _bounds(v, pv, nv, np.isfinite(pt), np.isfinite(nt), left3, right3)
    fixed = ~run_key[:m].copy()
    fixed_slope = np.zeros(m)
    # Clamped keys with no room (extrema, plateaus) are flat from the start.
    pinned = clamped[:m] & run_key[:m] & (lo_b[:m] == hi_b[:m])
    fixed |= pinned
    fixed_slope = np.where(pinned, lo_b[:m], fixed_slope)
    for _ in range(4 * m + 4):
        slopes, vec_left, vec_right = _solve_runs(
            t, m, cyclic, k0, k1, pa, pb, dur, dy, live, fixed, fixed_slope,
            lt, rt, ldt, ldv, rdt, rdv, before, after,
        )  # fmt: skip
        free = ~fixed & clamped[:m]
        bad = free & ((slopes < lo_b[:m]) | (slopes > hi_b[:m]))
        if not np.any(bad):
            break
        fixed = fixed | bad
        fixed_slope = np.where(bad, np.minimum(np.maximum(slopes, lo_b[:m]), hi_b[:m]), fixed_slope)
    # Write the run keys' handles (third-of-interval extents, one slope).
    keys = np.flatnonzero(run_key[:m])
    s = np.where(fixed[keys], fixed_slope[keys], slopes[keys])
    ldt[keys] = -left3[keys]
    ldv[keys] = -s * left3[keys]
    rdt[keys] = right3[keys]
    rdv[keys] = s * right3[keys]
    if cyclic and run_key[0]:  # the last key is the first one, a period later
        s0 = fixed_slope[0] if fixed[0] else slopes[0]
        ldt[n - 1], ldv[n - 1] = -left3[n - 1], -s0 * left3[n - 1]
        rdt[n - 1], rdv[n - 1] = right3[n - 1], s0 * right3[n - 1]
    for k, sv in vec_left.items():  # vector handles at run ends: natural free ends
        rdt[k], rdv[k] = right3[k], sv * right3[k]
    for k, sv in vec_right.items():
        ldt[k], ldv[k] = -left3[k], -sv * left3[k]


def _thirds(t: np.ndarray, cyclic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Third-of-interval extents (left, right) of every key (ends mirror; cyclic wraps)."""
    n = t.shape[0]
    d = np.diff(t)
    left = np.empty(n)
    right = np.empty(n)
    left[1:] = d / 3.0
    right[:-1] = d / 3.0
    if cyclic:
        left[0] = d[-1] / 3.0
        right[-1] = d[0] / 3.0
    else:
        left[0] = d[0] / 3.0
        right[-1] = d[-1] / 3.0
    return left, right


def _solve_runs(
    t, m, cyclic, k0, k1, pa, pb, dur, dy, live, fixed, fixed_slope,
    lt, rt, ldt, ldv, rdt, rdv, before, after,
):  # fmt: skip
    """One solve of every run with the current locks: slopes of the run keys (per circle
    position) and of the vector handles that end runs (``{key: slope}``, the handles that
    start a run, then those that end one)."""
    n = t.shape[0]
    third = dur / 3.0
    run_a = _run_mask(lt, rt, pa)
    run_b = _run_mask(lt, rt, pb)
    var_key = ~fixed  # per position: an automatic run key that is not locked
    # Which side of each segment is a variable: its start (key k0's right handle) or its
    # end (key k1's left handle). Vector handles next to a run become free (natural) ends.
    start_vec = live & ~var_key[k0] & (rt[pa] == _VECTOR) & run_b & ~run_a
    end_vec = live & ~var_key[k1] & (lt[pb] == _VECTOR) & run_a & ~run_b
    start_var = (live & var_key[k0]) | start_vec
    end_var = (live & var_key[k1]) | end_vec
    # Fixed geometry of each segment's handles (corrected to the segment's span, as the
    # evaluation does), or the variable's third-of-interval extent.
    span = np.where(live, dur, 1.0)
    x1 = np.where(start_var, third, np.minimum(rdt[pa], span))
    f1 = np.where(start_var | (rdt[pa] <= dur), 1.0, dur / np.where(rdt[pa] > 0, rdt[pa], 1.0))
    h1_fixed = rdv[pa] * f1
    x2 = np.where(end_var, -third, np.maximum(ldt[pb], -span))
    f2 = np.where(end_var | (-ldt[pb] <= dur), 1.0, dur / np.where(ldt[pb] < 0, -ldt[pb], 1.0))
    h2_fixed = ldv[pb] * f2
    # Locked run keys: their slope on a third-of-interval extent.
    lock_a = ~start_var & fixed[k0] & run_a
    lock_b = ~end_var & fixed[k1] & run_b
    h1_fixed = np.where(lock_a, fixed_slope[k0] * third, h1_fixed)
    x1 = np.where(lock_a, third, x1)
    h2_fixed = np.where(lock_b, -fixed_slope[k1] * third, h2_fixed)
    x2 = np.where(lock_b, -third, x2)
    # A and B are affine in the variables' slopes (h1 = slope third, h2 = -slope third).
    h1_0 = np.where(start_var, 0.0, h1_fixed)
    h2_0 = np.where(end_var, 0.0, h2_fixed)
    a00, b00 = _accel_terms(dy, dur, x1, h1_0, x2, h2_0)
    a10, b10 = _accel_terms(dy, dur, x1, h1_0 + np.where(start_var, third, 0.0), x2, h2_0)
    a01, b01 = _accel_terms(dy, dur, x1, h1_0, x2, h2_0 - np.where(end_var, third, 0.0))
    ca_s, ca_e = a10 - a00, a01 - a00  # dA / d(start slope), dA / d(end slope)
    cb_s, cb_e = b10 - b00, b01 - b00
    # The segment ending / starting at every position (-1: none).
    seg_ids = np.arange(k0.shape[0])
    ends_at = np.full(m, -1)
    starts_at = np.full(m, -1)
    ends_at[k1[live]] = seg_ids[live]
    starts_at[k0[live]] = seg_ids[live]
    # Variables in circle order; at a vector key between two runs its left handle (which
    # ends the run before it) comes first, then its right handle (which starts the next).
    has_ve = np.zeros(m, bool)
    has_vs = np.zeros(m, bool)
    has_ve[k1[end_vec]] = True
    has_vs[k0[start_vec]] = True
    count = has_ve.astype(int) + var_key + has_vs
    base = np.cumsum(count) - count
    idx_ve = np.where(has_ve, base, -1)
    idx_k = np.where(var_key, base + has_ve, -1)
    idx_vs = np.where(has_vs, base + has_ve + var_key, -1)
    nv = int(count.sum())
    if nv == 0:
        return np.zeros(m), {}, {}
    seg_start_var = np.where(start_vec, idx_vs[k0], np.where(start_var, idx_k[k0], -1))
    seg_end_var = np.where(end_vec, idx_ve[k1], np.where(end_var, idx_k[k1], -1))
    # Rows: (row index, variable index, coefficient) triples and right-hand sides.
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs = np.zeros(nv)

    def put(row, col, coef):
        keep = col >= 0
        rows.append(row[keep])
        cols.append(col[keep])
        vals.append(coef[keep])

    # Vector handles ending a run: B(segment) = 0; starting one: A(segment) = 0.
    p = np.flatnonzero(has_ve)
    if p.size:
        sg = ends_at[p]
        put(idx_ve[p], seg_start_var[sg], cb_s[sg])
        put(idx_ve[p], seg_end_var[sg], cb_e[sg])
        rhs[idx_ve[p]] -= b00[sg]
    p = np.flatnonzero(has_vs)
    if p.size:
        sg = starts_at[p]
        put(idx_vs[p], seg_start_var[sg], ca_s[sg])
        put(idx_vs[p], seg_end_var[sg], ca_e[sg])
        rhs[idx_vs[p]] -= a00[sg]
    # Run keys: continuous acceleration B(left) - A(right) = 0, a natural end where one
    # segment is missing (linear extrapolation, a jump), flat ends otherwise.
    p = np.flatnonzero(var_key)
    if p.size:
        sl = ends_at[p]
        sr = starts_at[p]
        row = idx_k[p]
        first = (sl < 0) & (p == 0) & (not cyclic)
        last = (sr < 0) & (p == n - 1) & (not cyclic)
        flat = (first & (before != _LINEAR)) | (last & (after != _LINEAR)) | ((sl < 0) & (sr < 0))
        both = ~flat & (sl >= 0) & (sr >= 0)
        only_r = ~flat & (sl < 0) & (sr >= 0)
        only_l = ~flat & (sl >= 0) & (sr < 0)
        slc = np.where(sl >= 0, sl, 0)
        src = np.where(sr >= 0, sr, 0)
        zero = np.zeros(p.size)
        put(row, np.where(flat, row, -1), np.ones(p.size))
        use_l = both | only_l
        use_r = both | only_r
        put(row, np.where(use_l, seg_start_var[slc], -1), np.where(use_l, cb_s[slc], zero))
        put(row, np.where(use_l, seg_end_var[slc], -1), np.where(use_l, cb_e[slc], zero))
        put(row, np.where(use_r, seg_start_var[src], -1), np.where(use_r, -ca_s[src], zero))
        put(row, np.where(use_r, seg_end_var[src], -1), np.where(use_r, -ca_e[src], zero))
        rhs[row] -= np.where(use_l, b00[slc], 0.0) - np.where(use_r, a00[src], 0.0)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    A = np.zeros(nv)
    B = np.zeros(nv)
    C = np.zeros(nv)
    corner = np.zeros(2)
    off = c - r
    np.add.at(B, r[off == 0], v[off == 0])
    np.add.at(A, r[off == -1], v[off == -1])
    np.add.at(C, r[off == 1], v[off == 1])
    wrap_lo = (r == 0) & (c == nv - 1) & (nv > 2)
    wrap_hi = (r == nv - 1) & (c == 0) & (nv > 2)
    corner[0] = v[wrap_lo].sum() if cyclic else 0.0
    corner[1] = v[wrap_hi].sum() if cyclic else 0.0
    other = (np.abs(off) > 1) & ~(cyclic & (wrap_lo | wrap_hi))
    if np.any(other):  # pragma: no cover - the equations only couple neighbours
        raise AssertionError("non-tridiagonal smoothing system")
    D = rhs
    bad = ~np.isfinite(A) | ~np.isfinite(B) | ~np.isfinite(C) | ~np.isfinite(D) | (B == 0.0)
    if np.any(bad):  # a vertical neighbour handle: keep those variables flat
        A[bad], B[bad], C[bad], D[bad] = 0.0, 1.0, 0.0, 0.0
    if cyclic and (corner[0] != 0.0 or corner[1] != 0.0):
        A[0] += corner[0]
        C[-1] += corner[1]
        x = _solve_periodic(A, B, C, D)
    else:
        x = _solve_tridiagonal(A, B, C, D)
    slopes = np.zeros(m)
    slopes[var_key] = x[idx_k[var_key]]
    vec_left = {int(pa[s]): float(x[seg_start_var[s]]) for s in np.flatnonzero(start_vec).tolist()}
    vec_right = {int(pb[s]): float(x[seg_end_var[s]]) for s in np.flatnonzero(end_vec).tolist()}
    return slopes, vec_left, vec_right


def _is_run(lt, rt, k) -> bool:
    return bool(lt[k] in (_AUTO, _AUTO_CLAMPED) and rt[k] in (_AUTO, _AUTO_CLAMPED))


def _run_mask(lt, rt, keys) -> np.ndarray:
    auto = (lt[keys] == _AUTO) | (lt[keys] == _AUTO_CLAMPED)
    return auto & ((rt[keys] == _AUTO) | (rt[keys] == _AUTO_CLAMPED))
