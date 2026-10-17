"""Batched evaluation of many F-curves at once (one vectorized pass per frame).

A :class:`CurveBatch` concatenates the compiled segments of a list of curves into one
structure of arrays. ``evaluate(curve_index, times)`` evaluates any number of
(curve, time) queries together: extrapolation is mapped per query, the segment is found by
a vectorized bisection within each curve's keys (only comparisons: exact), and the
segment kernel of :mod:`.curves` computes values and derivatives. Each query's result is
bit for bit what ``curve.evaluate(time)`` gives (the same arithmetic per element), whatever
else is in the batch. Curves with modifiers are evaluated one by one through their own
chain (modifiers are rare and may need several base evaluations per time).
"""

from __future__ import annotations

import numpy as np

from .curves import _SEG_JUMP, FCurve, _Compiled, _segment_values


class CurveBatch:
    """The compiled segments of ``curves``, concatenated."""

    def __init__(self, curves: list[FCurve]) -> None:
        self.curves = list(curves)
        n = len(self.curves)
        comps = [c.compiled() for c in self.curves]
        nkeys = np.array([c.t.shape[0] for c in comps], np.int64)
        self.key_start = np.zeros(n, np.int64)
        if n:
            self.key_start[1:] = np.cumsum(nkeys)[:-1]
        self.nkeys = nkeys
        # Segment numbering: curve c's segments start at key_start[c] - c.
        self.seg_start = self.key_start - np.arange(n, dtype=np.int64)
        self.first = np.array([c.t[0] for c in comps]) if n else np.zeros(0)
        self.last = np.array([c.t[-1] for c in comps]) if n else np.zeros(0)
        self.v_first = np.array([c.v[0] for c in comps]) if n else np.zeros(0)
        self.v_last = np.array([c.v[-1] for c in comps]) if n else np.zeros(0)
        self.before = np.array([c._before for c in self.curves], np.int8)
        self.after = np.array([c._after for c in self.curves], np.int8)
        self.slope_before = np.array([c.slope_before for c in comps]) if n else np.zeros(0)
        self.slope_after = np.array([c.slope_after for c in comps]) if n else np.zeros(0)
        self.modified = np.array([bool(c.modifiers) for c in self.curves], bool)
        merged = _Compiled()
        if n:
            merged.t = np.concatenate([c.t for c in comps])
            merged.v = np.concatenate([c.v for c in comps])
            for name in (
                "kind",
                "span",
                "easing",
                "mode",
                "back",
                "amplitude",
                "period",
                "corrected",
            ):
                parts = [getattr(c, name) for c in comps]
                setattr(merged, name, np.concatenate(parts) if parts else np.zeros(0))
            merged.x = np.concatenate([c.x.reshape(-1, 4) for c in comps])
            merged.y = np.concatenate([c.y.reshape(-1, 4) for c in comps])
        self.merged = merged
        self.max_keys = int(nkeys.max()) if n else 0

    def __len__(self) -> int:
        return len(self.curves)

    def evaluate(self, curve: np.ndarray, times: np.ndarray, order: int = 0) -> list[np.ndarray]:
        """Values (and derivatives up to ``order``) of ``curves[curve[i]]`` at ``times[i]``."""
        curve = np.asarray(curve, np.int64)
        t = np.asarray(times, np.float64)
        out = [np.zeros_like(t) for _ in range(order + 1)]
        if t.size == 0:
            return out
        plain = ~self.modified[curve]
        if np.any(~plain):
            for c in np.unique(curve[~plain]).tolist():
                sel = curve == c
                vals = self.curves[c].evaluate(t[sel], order=order)
                vals = [vals] if order == 0 else list(vals)
                for k in range(order + 1):
                    out[k][sel] = vals[k]
        if not np.any(plain):
            return out
        res = self._base(curve[plain], t[plain], order)
        for k in range(order + 1):
            out[k][plain] = res[k]
        return out

    def _base(self, c: np.ndarray, t: np.ndarray, order: int) -> list[np.ndarray]:
        """``curves.FCurve.evaluate_base`` for many curves (the same arithmetic)."""
        m = self.merged
        first = self.first[c]
        last = self.last[c]
        out = [np.empty_like(t) for _ in range(order + 1)]
        for o in out[1:]:
            o.fill(0.0)
        tt = t.copy()
        offset = np.zeros_like(t)
        sign = np.ones_like(t)
        period = last - first
        for mask_all, modes in ((t < first, self.before[c]), (t > last, self.after[c])):
            mask = mask_all & (modes >= 2)
            if not np.any(mask):
                continue
            p = period[mask]
            f = first[mask]
            rel = (t[mask] - f) / p
            cycle = np.floor(rel)
            local = np.clip(t[mask] - f - cycle * p, 0.0, p)
            md = modes[mask]
            mirror = md == 4
            odd = (np.abs(cycle) % 2.0) == 1.0
            flip = mirror & odd
            tt[mask] = np.where(flip, last[mask] - local, f + local)
            sign[mask] = np.where(flip, -1.0, 1.0)
            delta = self.v_last[c][mask] - self.v_first[c][mask]
            offset[mask] = np.where(md == 3, cycle * delta, 0.0)
        inside = (tt >= first) & (tt <= last)
        single = self.nkeys[c] == 1
        out[0][single & inside] = self.v_first[c][single & inside]
        q = inside & ~single
        if np.any(q):
            cq = c[q]
            tq = tt[q]
            lo = self.key_start[cq].copy()
            hi = lo + self.nkeys[cq] - 1
            # Largest key index i in [lo, hi] with key_time[i] <= tq (right-continuous).
            keys = m.t
            for _ in range(int(np.ceil(np.log2(max(self.max_keys, 2)))) + 1):
                live = lo < hi
                if not np.any(live):
                    break
                mid = (lo + hi + 1) // 2
                go = live & (keys[np.minimum(mid, len(keys) - 1)] <= tq)
                lo = np.where(go, mid, lo)
                hi = np.where(live & ~go, mid - 1, hi)
            seg = lo - cq  # global segment index of key lo
            last_seg = self.seg_start[cq] + self.nkeys[cq] - 2
            seg = np.minimum(seg, last_seg)
            vals = _segment_values(m, seg, tq, order, seg + cq)
            at_last = tq == last[q]
            if np.any(at_last):
                vals[0] = np.where(at_last, self.v_last[cq], vals[0])
                if order >= 1:
                    ls = last_seg[at_last]
                    lim = _segment_values(m, ls, tq[at_last], order, ls + cq[at_last])
                    for o in range(1, order + 1):
                        vals[o][at_last] = lim[o]
            for o in range(order + 1):
                out[o][q] = vals[o]
        for mask_all, modes, end in ((t < first, self.before[c], 0), (t > last, self.after[c], 1)):
            mask = mask_all & (modes < 2)
            if not np.any(mask):
                continue
            key_v = (self.v_first if end == 0 else self.v_last)[c][mask]
            key_t = first[mask] if end == 0 else last[mask]
            lin = modes[mask] == 1
            slope = (self.slope_before if end == 0 else self.slope_after)[c][mask]
            out[0][mask] = np.where(lin, key_v + slope * (t[mask] - key_t), key_v)
            if order >= 1:
                out[1][mask] = np.where(lin, slope, 0.0)
            for o in range(2, order + 1):
                out[o][mask] = 0.0
        if np.any(offset):
            out[0] = out[0] + offset
        if np.any(sign < 0):
            for o in range(1, order + 1, 2):
                out[o] = out[o] * sign
        return out


__all__ = ["CurveBatch", "_SEG_JUMP"]


class RotationBatch:
    """Many :class:`~.tracks.RotationTrack`s concatenated: ``evaluate(track, times)``
    gives each query's orientation with the arithmetic of ``RotationTrack.evaluate``
    (which evaluates through a batch of one, so both agree bit for bit)."""

    def __init__(self, tracks: list) -> None:
        from . import rotation as R

        self.tracks = list(tracks)
        n = len(self.tracks)
        nkeys = np.array([len(tr) for tr in self.tracks], np.int64)
        self.nkeys = nkeys
        self.key_start = np.zeros(n, np.int64)
        if n:
            self.key_start[1:] = np.cumsum(nkeys)[:-1]
        self.max_keys = int(nkeys.max()) if n else 0
        cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)  # noqa: E731
        self.t = cat([tr._times for tr in self.tracks], (0,))
        self.q = cat([tr._quats for tr in self.tracks], (0, 4))
        self.aligned = cat([tr._aligned() for tr in self.tracks], (0, 4))
        self.controls = cat([tr._controls() for tr in self.tracks], (0, 4))
        self.interp = cat([tr._interp for tr in self.tracks], (0,))
        self.timing = cat([tr._timing for tr in self.tracks], (0,))
        self.timing_params = cat([tr._timing_params for tr in self.tracks], (0, 5))
        self.tin = cat([tr._in for tr in self.tracks], (0, 4))
        self.tout = cat([tr._out for tr in self.tracks], (0, 4))
        self.before = np.array([tr._before for tr in self.tracks], np.int8)
        self.after = np.array([tr._after for tr in self.tracks], np.int8)
        firsts = [tr._aligned()[0] for tr in self.tracks]
        lasts = [tr._aligned()[-1] for tr in self.tracks]
        # Rotation from the first to the last key (turning per cycle, cycle_offset).
        self.delta_log = (
            R.log(R.multiply(np.array(lasts), R.conjugate(np.array(firsts))))
            if n
            else np.zeros((0, 3))
        )

    def __len__(self) -> int:
        return len(self.tracks)

    def segment(self, c: np.ndarray, t: np.ndarray):
        """For queries: (mapped time, pre-rotation log * cycle, sign, key index k, at-end)."""
        first = self.t[self.key_start[c]]
        last = self.t[self.key_start[c] + self.nkeys[c] - 1]
        period = last - first
        tt = t.copy()
        cyc_log = np.zeros((t.shape[0], 3))
        sign = np.ones_like(t)
        for mask_all, modes in ((t < first, self.before[c]), (t > last, self.after[c])):
            mask = mask_all & (modes >= 1)
            if not np.any(mask):
                continue
            p = period[mask]
            f = first[mask]
            rel = (t[mask] - f) / p
            cyc = np.floor(rel)
            local = np.clip(t[mask] - f - cyc * p, 0.0, p)
            md = modes[mask]
            odd = (np.abs(cyc) % 2.0) == 1.0
            flip = (md == 3) & odd
            tt[mask] = np.where(flip, last[mask] - local, f + local)
            sign[mask] = np.where(flip, -1.0, 1.0)
            turn = md == 2
            cyc_log[mask] = np.where(turn[:, None], self.delta_log[c[mask]] * cyc[:, None], 0.0)
        tt = np.clip(tt, first, last)
        lo = self.key_start[c].copy()
        hi = lo + self.nkeys[c] - 1
        for _ in range(int(np.ceil(np.log2(max(self.max_keys, 2)))) + 1):
            live = lo < hi
            if not np.any(live):
                break
            mid = (lo + hi + 1) // 2
            go = live & (self.t[np.minimum(mid, len(self.t) - 1)] <= tt)
            lo = np.where(go, mid, lo)
            hi = np.where(live & ~go, mid - 1, hi)
        end = self.key_start[c] + self.nkeys[c] - 1
        at_end = lo >= end
        k = np.where(at_end, np.maximum(end - 1, self.key_start[c]), lo)
        return tt, cyc_log, sign, k, at_end

    def evaluate(self, c: np.ndarray, t: np.ndarray) -> np.ndarray:
        from . import rotation as R
        from .tracks import CUBIC, SLERP, SQUAD, timing_values

        c = np.asarray(c, np.int64)
        t = np.asarray(t, np.float64)
        out = np.empty((t.shape[0], 4))
        if t.size == 0:
            return out
        tt, cyc_log, _, k, at_end = self.segment(c, t)
        single = self.nkeys[c] == 1
        out[single] = self.q[self.key_start[c][single]]
        multi = ~single
        if np.any(multi):
            km = k[multi]
            tm = tt[multi]
            span = self.t[km + 1] - self.t[km]
            s = np.clip((tm - self.t[km]) / span, 0.0, 1.0)
            u = timing_values(self.timing[km], self.timing_params[km], s, 0)[0]
            kind = self.interp[km]
            q0 = self.aligned[km]
            q1 = self.aligned[km + 1]
            res = q0.copy()
            sel = kind == SLERP
            if np.any(sel):
                res[sel] = R.slerp(q0[sel], q1[sel], u[sel], shortest=False)
            sel = kind == SQUAD
            if np.any(sel):
                res[sel] = R.squad(
                    q0[sel], q1[sel], self.controls[km[sel]], self.controls[km[sel] + 1], u[sel]
                )
            sel = kind == CUBIC
            if np.any(sel):
                uu = u[sel]
                u2, u3 = uu * uu, uu * uu * uu
                sp = span[sel]
                r = (
                    (2 * u3 - 3 * u2 + 1)[:, None] * self.q[km[sel]]
                    + (sp * (u3 - 2 * u2 + uu))[:, None] * self.tout[km[sel]]
                    + (-2 * u3 + 3 * u2)[:, None] * self.q[km[sel] + 1]
                    + (sp * (u3 - u2))[:, None] * self.tin[km[sel] + 1]
                )
                res[sel] = R.normalize(r)
            # Exactly at a key: the key (continuous sign sequence); at the end: the last.
            at = tm == self.t[km]
            res[at] = q0[at]
            ae = at_end[multi]
            res[ae] = self.aligned[km[ae] + 1]
            out[multi] = res
        turned = np.any(cyc_log != 0.0, axis=1)
        if np.any(turned):
            out[turned] = R.multiply(R.exp(cyc_log[turned]), out[turned])
        return out
