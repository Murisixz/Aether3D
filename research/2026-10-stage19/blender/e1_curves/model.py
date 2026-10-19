"""An independent model of Blender's automatic F-curve handles, written from the measured
behaviour (E1, E1b) and the mathematics of cubic Hermite splines -- not from Blender's code.

none:        slope = mean of the two secants; extents = interval / 2.5614 (the end key
             mirrors its neighbour: secant slope; constant extrapolation: flat ends);
             clamped: each handle's height bounded by its neighbour key's value difference
             (no reverse, no overshoot), extrema flat.
continuous:  extents = interval / 3 (Hermite in time); slopes solve C2 continuity at every
             unlocked key of each run (tridiagonal); ends: constant extrapolation -> slope 0,
             linear -> natural (zero acceleration); clamped keys: box constraints on the
             handle heights, enforced by locking violators at their bound and solving again.
"""

import numpy as np

K = 2.5614


def secants(t, v):
    d = np.diff(t)
    return np.diff(v) / d, d


def model_none(t, v, clamped, linear_ends):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = len(t)
    s, d = secants(t, v)
    slope = np.zeros(n)
    left = np.zeros(n)
    right = np.zeros(n)
    for k in range(n):
        has_p, has_n = k > 0, k < n - 1
        sp = s[k - 1] if has_p else None
        sn = s[k] if has_n else None
        lp = d[k - 1] if has_p else d[k]
        ln = d[k] if has_n else d[k - 1]
        # handle lengths: each neighbour distance at most five times the other one
        lp, ln = min(lp, 5.0 * ln), min(ln, 5.0 * lp)
        left[k], right[k] = lp / K, ln / K
        if has_p and has_n:
            m = 0.5 * (sp + sn)
        else:
            m = (sn if has_n else sp) if linear_ends else 0.0
        if clamped:
            if has_p and has_n and sp * sn <= 0:
                m = 0.0
            else:
                # handle heights within the neighbour's value difference, same sign
                lo, hi = -np.inf, np.inf
                for dy, ext in (
                    ((v[k] - v[k - 1]) if has_p else None, left[k]),
                    ((v[k + 1] - v[k]) if has_n else None, right[k]),
                ):
                    if dy is None:
                        continue
                    if dy > 0:
                        lo, hi = max(lo, 0.0), min(hi, dy / ext)
                    elif dy < 0:
                        lo, hi = max(lo, dy / ext), min(hi, 0.0)
                    else:
                        lo = hi = 0.0
                m = min(max(m, lo), hi)
        slope[k] = m
    return slope, left, right


def solve_c2(t, v, fixed, linear_ends):
    """Slopes of the C2 cubic Hermite spline (1/3 extents) with slopes ``fixed`` (NaN:
    unknown) imposed; ends: clamped (slope 0) or natural."""
    n = len(t)
    s, d = secants(t, v)
    A = np.zeros((n, n))
    r = np.zeros(n)
    for k in range(n):
        if np.isfinite(fixed[k]):
            A[k, k] = 1.0
            r[k] = fixed[k]
            continue
        if k == 0:
            if linear_ends:
                A[0, 0], A[0, 1], r[0] = 2.0, 1.0, 3.0 * s[0]
            else:
                A[0, 0], r[0] = 1.0, 0.0
            continue
        if k == n - 1:
            if linear_ends:
                A[k, k - 1], A[k, k], r[k] = 1.0, 2.0, 3.0 * s[k - 1]
            else:
                A[k, k], r[k] = 1.0, 0.0
            continue
        a, b = d[k - 1], d[k]
        A[k, k - 1] = 1.0 / a
        A[k, k] = 2.0 * (1.0 / a + 1.0 / b)
        A[k, k + 1] = 1.0 / b
        r[k] = 3.0 * (s[k - 1] / a + s[k] / b)
    return np.linalg.solve(A, r)


def model_cont(t, v, clamped, linear_ends, policy="all"):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = len(t)
    s, d = secants(t, v)
    left = np.r_[d[:1], d] / 3.0
    right = np.r_[d, d[-1:]] / 3.0
    fixed = np.full(n, np.nan)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if clamped:
        for k in range(n):
            for dy, ext in (
                ((v[k] - v[k - 1]) if k > 0 else None, left[k]),
                ((v[k + 1] - v[k]) if k < n - 1 else None, right[k]),
            ):
                if dy is None:
                    continue
                if dy > 0:
                    lo[k], hi[k] = max(lo[k], 0.0), min(hi[k], dy / ext)
                elif dy < 0:
                    lo[k], hi[k] = max(lo[k], dy / ext), min(hi[k], 0.0)
                else:
                    lo[k] = hi[k] = 0.0
            if lo[k] == hi[k]:
                fixed[k] = lo[k]
    for _ in range(4 * n):
        m = solve_c2(t, v, fixed, linear_ends)
        bad = [
            k
            for k in range(n)
            if not np.isfinite(fixed[k]) and (m[k] < lo[k] - 1e-12 or m[k] > hi[k] + 1e-12)
        ]
        if not bad:
            return m, left, right
        if policy == "worst":
            k = max(bad, key=lambda k: max(lo[k] - m[k], m[k] - hi[k]))
            bad = [k]
        for k in bad:
            fixed[k] = min(max(m[k], lo[k]), hi[k])
    return m, left, right


def model_cont_cyclic(t, v, clamped, offset=False):
    """Periodic C2 spline: key n-1 is key 0 one period later (its slope equals key 0's)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = len(t)
    P = t[-1] - t[0]
    dv = v[-1] - v[0]
    # unknowns m_0..m_{n-2}; neighbours wrap: prev(0) = key n-2 - P, next(n-2) = key n-1
    m_idx = n - 1
    A = np.zeros((m_idx, m_idx))
    r = np.zeros(m_idx)
    for k in range(m_idx):
        kp = (k - 1) % m_idx
        kn = (k + 1) % m_idx
        tp = t[k - 1] if k > 0 else t[n - 2] - P
        vp = v[k - 1] if k > 0 else v[n - 2] - (dv if offset else 0.0)
        tn = t[k + 1]
        vn = v[k + 1] if k + 1 < n - 1 else v[n - 1] - (0.0 if offset else dv)
        a, b = t[k] - tp, tn - t[k]
        sa, sb = (v[k] - vp) / a, (vn - v[k]) / b
        A[k, kp] += 1.0 / a
        A[k, k] += 2.0 * (1.0 / a + 1.0 / b)
        A[k, kn] += 1.0 / b
        r[k] = 3.0 * (sa / a + sb / b)
    m = np.linalg.solve(A, r)
    m = np.r_[m, m[0]]
    d = np.diff(t)
    left = np.r_[d[-1:], d] / 3.0
    right = np.r_[d, d[:1]] / 3.0
    return m, left, right
