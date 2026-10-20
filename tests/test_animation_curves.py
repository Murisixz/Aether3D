"""F-curves: keys, interpolation, handles, easing, extrapolation, modifiers, derivatives.

Expected values come from closed forms, independent references written here (cubic
Hermite polynomials, an exact-rational Bezier inversion) and properties (no overshoot,
continuity, monotone time), never from the implementation's own output.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pytest

from aether3d.animation import FCurve
from aether3d.animation.curves import solve_bezier_time
from aether3d.animation.modifiers import gradient_noise
from aether3d.animation.time import FrameRate, frame_range
from aether3d.errors import AnimationError

RNG = np.random.default_rng(17)


def fd(curve: FCurve, t: np.ndarray, k: int, h: float = 1e-6) -> np.ndarray:
    """Central difference of derivative ``k`` (0 = value) at ``t``."""
    lo = curve.evaluate(t - h, order=3)[k]
    hi = curve.evaluate(t + h, order=3)[k]
    return (hi - lo) / (2 * h)


# --------------------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("rate", [24, 25, "30000/1001", "24000/1001", 29.97, [60000, 1001], 120])
def test_frames_and_times_round_trip_exactly_at_every_rate(rate):
    fr = FrameRate(rate)
    frames = np.concatenate([np.arange(-5000, 5000), np.arange(10**7, 10**7 + 500)])
    times = fr.frame_times(frames)
    for n, t in zip(frames[::97].tolist(), times[::97].tolist(), strict=True):
        assert fr.frame_time(n) == t  # the exact rational, rounded once
        assert float(Fraction(n * fr.denominator, fr.numerator)) == t
        for rounding in ("nearest", "floor", "ceil"):
            assert fr.frame_at(t, rounding) == n
    # Between frames: floor gives the frame on screen, ceil the next one.
    mid = fr.frame_time(10) + 0.25 * fr.frame_duration
    assert fr.frame_at(mid, "floor") == 10 and fr.frame_at(mid, "ceil") == 11
    assert fr.frame_at(mid) == 10


def test_ntsc_rates_are_exact_rationals_and_timecode_is_non_drop():
    fr = FrameRate(29.97)
    assert (fr.numerator, fr.denominator) == (30000, 1001)
    # One hour of NTSC frames is 3600 * 30000 / 1001 = 107892.1 frames: the non-drop
    # timecode (counted at 30 fps) runs behind the clock.
    assert fr.timecode(3600.0) == "00:59:56:12"
    assert FrameRate(25).timecode(3723.5) == "01:02:03:12"
    assert FrameRate(24).timecode(-1.0) == "-00:00:01:00"
    assert frame_range(FrameRate(24), 0.0, 1.0).tolist() == list(range(25))


@pytest.mark.parametrize("bad", [0, -24, "abc", 29.5, [1, 0], True, 1e9, float("nan")])
def test_invalid_frame_rates_are_refused(bad):
    with pytest.raises(AnimationError) as err:
        FrameRate(bad)
    assert err.value.code == "INVALID_FRAME_RATE"


# --------------------------------------------------------------------------------------- #
# Keys and interpolation
# --------------------------------------------------------------------------------------- #

KEYS = [(0.0, 0.0), (1.0, 2.0), (2.5, -1.0), (4.0, 3.0)]


@pytest.mark.parametrize(
    "interp",
    [
        "constant",
        "linear",
        "bezier",
        "sine",
        "quad",
        "cubic",
        "quart",
        "quint",
        "expo",
        "circ",
        "back",
        "elastic",
        "bounce",
    ],
)
@pytest.mark.parametrize("ease", ["in", "out", "in_out"])
def test_every_interpolation_passes_through_its_keys_exactly(interp, ease):
    c = FCurve.from_keys(
        [{"time": t, "value": v, "interpolation": interp, "ease": ease} for t, v in KEYS]
    )
    assert np.array_equal(c.evaluate(c.times), c.values)


def test_constant_and_linear_are_the_closed_forms():
    c = FCurve.from_keys(KEYS, interpolation="constant")
    t = np.array([0.0, 0.999, 1.0, 2.49, 2.5, 3.99, 4.0, 9.0])
    assert c.evaluate(t).tolist() == [0.0, 0.0, 2.0, 2.0, -1.0, -1.0, 3.0, 3.0]
    lin = FCurve.from_keys(KEYS, interpolation="linear")
    ts = np.linspace(1.0, 2.5, 7)
    expect = 2.0 + (-1.0 - 2.0) * (ts - 1.0) / 1.5
    assert np.allclose(lin.evaluate(ts), expect, rtol=0, atol=1e-15)
    v = lin.evaluate(ts[1:-1], order=2)
    assert np.allclose(v[1], -2.0) and np.all(v[2] == 0.0)


def hermite(t0, t1, v0, v1, m0, m1, t):
    """Independent cubic Hermite evaluation (value, velocity, acceleration)."""
    h = t1 - t0
    s = (t - t0) / h
    h00, h10, h01, h11 = (
        2 * s**3 - 3 * s**2 + 1,
        s**3 - 2 * s**2 + s,
        -2 * s**3 + 3 * s**2,
        s**3 - s**2,
    )
    v = h00 * v0 + h10 * h * m0 + h01 * v1 + h11 * h * m1
    d = (
        (6 * s**2 - 6 * s) * v0
        + (3 * s**2 - 4 * s + 1) * h * m0
        + (-6 * s**2 + 6 * s) * v1
        + (3 * s**2 - 2 * s) * h * m1
    ) / h
    a = (
        (12 * s - 6) * v0 + (6 * s - 4) * h * m0 + (-12 * s + 6) * v1 + (6 * s - 2) * h * m1
    ) / h**2
    return v, d, a


def test_auto_handles_are_the_parabola_slopes_and_give_a_c1_hermite_spline():
    t = np.array([0.0, 1.0, 3.0, 4.0, 7.0])
    v = np.array([0.0, 2.0, 1.0, 5.0, 0.0])
    c = FCurve.from_arrays(t, v, handles="auto", smoothing="local")
    sec = np.diff(v) / np.diff(t)
    slopes = np.empty(5)
    slopes[0], slopes[-1] = sec[0], sec[-1]
    for i in range(1, 4):
        dp, dn = t[i] - t[i - 1], t[i + 1] - t[i]
        slopes[i] = (sec[i - 1] * dn + sec[i] * dp) / (dp + dn)  # the parabola's slope
    for i in range(5):
        assert c.slope(i, "right") == pytest.approx(slopes[i], abs=1e-14)
        assert c.slope(i, "left") == pytest.approx(slopes[i], abs=1e-14)
    for k in range(4):
        ts = np.linspace(t[k], t[k + 1], 9)[1:-1]
        val, vel, acc = c.evaluate(ts, order=2)
        ev, ed, ea = hermite(t[k], t[k + 1], v[k], v[k + 1], slopes[k], slopes[k + 1], ts)
        assert np.allclose(val, ev, atol=1e-13) and np.allclose(vel, ed, atol=1e-12)
        assert np.allclose(acc, ea, atol=1e-10)


def test_auto_clamped_never_overshoots_and_is_flat_at_extrema():
    worst = 0.0
    for trial in range(400):
        n = int(RNG.integers(3, 25))
        t = np.cumsum(RNG.uniform(1e-3, 3.0, n))
        v = RNG.normal(0, 5, n)
        if trial % 2:
            v = np.cumsum(np.abs(v))  # monotone data must stay monotone
        c = FCurve.from_arrays(t, v, handles="auto_clamped")
        for k in range(n - 1):
            ys = c.evaluate(np.linspace(t[k], t[k + 1], 64))
            lo, hi = min(v[k], v[k + 1]), max(v[k], v[k + 1])
            worst = max(worst, float(np.max(ys - hi)), float(np.max(lo - ys)))
            if trial % 2:
                assert np.all(np.diff(ys) >= -1e-12)
    assert worst <= 0.0
    c = FCurve.from_arrays([0, 1, 2], [0, 5, 1], handles="auto_clamped")
    assert c.slope(1, "left") == 0.0 == c.slope(1, "right")  # a peak is flat
    assert c.slope(0, "right") == 0.0 and c.slope(2, "left") == 0.0  # ends are eased


def test_vector_aligned_and_free_handles():
    c = FCurve.from_keys(
        [
            {"time": 0, "value": 0, "right": "vector"},
            {
                "time": 3,
                "value": 6,
                "left": "vector",
                "right": {"type": "free", "dt": 1.0, "dv": 5.0},
            },
            {"time": 6, "value": 0, "left": {"type": "aligned", "dt": -1.0, "dv": 9.0}},
        ]
    )
    assert c.key(0)["right"]["dv"] == pytest.approx(2.0)  # 1/3 of the way to the next key
    assert c.slope(1, "left") == pytest.approx(2.0)
    # Unweighted: the free handle's extent becomes a third of the interval, slope kept.
    assert c.key(1)["right"]["dt"] == pytest.approx(1.0) and c.slope(1, "right") == pytest.approx(
        5.0
    )
    # An aligned handle takes the other side's slope (its own was 9 / -1 = -9).
    assert c.slope(2, "left") == pytest.approx(c.slope(2, "right"))


def test_weighted_handles_follow_an_exact_bezier_with_monotone_time():
    """Against an independent exact-rational reference: de Casteljau in Fractions,
    bisection on the parameter."""

    def bezier(p0, p1, p2, p3, u):
        a = [p0 + (p1 - p0) * u, p1 + (p2 - p1) * u, p2 + (p3 - p2) * u]
        b = [a[0] + (a[1] - a[0]) * u, a[1] + (a[2] - a[1]) * u]
        return b[0] + (b[1] - b[0]) * u

    keys = [
        {
            "time": 0.0,
            "value": 0.0,
            "weighted": True,
            "right": {"type": "free", "dt": 1.5, "dv": 3.0},
        },
        {
            "time": 2.0,
            "value": 1.0,
            "weighted": True,
            "left": {"type": "free", "dt": -0.25, "dv": 0.5},
        },
    ]
    c = FCurve.from_keys(keys)
    xs = [Fraction(0), Fraction(3, 2), Fraction(2) - Fraction(1, 4), Fraction(2)]
    ys = [Fraction(0), Fraction(3), Fraction(1) + Fraction(1, 2), Fraction(1)]
    for target in (0.1, 0.5, 0.9, 1.3, 1.7, 1.95):
        lo, hi = Fraction(0), Fraction(1)
        for _ in range(80):
            mid = (lo + hi) / 2
            if bezier(*xs, mid) < Fraction(target):
                lo = mid
            else:
                hi = mid
        expect = float(bezier(*ys, (lo + hi) / 2))
        assert c(target) == pytest.approx(expect, abs=1e-13)


def test_handles_reaching_past_the_segment_are_scaled_back_and_time_stays_monotone():
    for _ in range(300):
        n = int(RNG.integers(2, 7))
        t = np.cumsum(RNG.uniform(0.01, 2.0, n))
        keys = [
            {
                "time": t[i],
                "value": RNG.normal(0, 3),
                "weighted": True,
                "left": {"type": "free", "dt": -RNG.uniform(0, 5.0), "dv": RNG.normal(0, 3)},
                "right": {"type": "free", "dt": RNG.uniform(0, 5.0), "dv": RNG.normal(0, 3)},
            }
            for i in range(n)
        ]
        c = FCurve.from_keys(keys)
        comp = c.compiled()
        u = np.linspace(0, 1, 257)
        for k in range(n - 1):
            x = ((comp.x[k, 0] * u + comp.x[k, 1]) * u + comp.x[k, 2]) * u + comp.x[k, 3]
            assert np.all(np.diff(x) >= -1e-12 * comp.span[k])
            assert x[0] >= 0.0 and x[-1] == pytest.approx(comp.span[k])
        long_right = keys[0]["right"]["dt"] > t[1] - t[0]
        long_left = n > 1 and -keys[1]["left"]["dt"] > t[1] - t[0]
        assert bool(comp.corrected[0]) == (long_right or long_left)


def test_newton_bisection_inverts_monotone_cubics_to_the_ulp():
    for _ in range(200):
        a, b = RNG.uniform(0, 1, 2)
        span = float(RNG.uniform(1e-6, 1e6))
        x = np.array([[span * (3 * a - 3 * b + 1), span * (-6 * a + 3 * b), span * 3 * a, 0.0]])
        u_true = RNG.uniform(0, 1, 50)
        xa = ((x[0, 0] * u_true + x[0, 1]) * u_true + x[0, 2]) * u_true
        deriv = (3 * x[0, 0] * u_true + 2 * x[0, 1]) * u_true + x[0, 2]
        u = solve_bezier_time(np.repeat(x, 50, 0), xa, np.full(50, span))
        well = deriv > 1e-3 * span
        assert np.all(np.abs(u - u_true)[well] <= 64 * np.finfo(float).eps)


def test_jumps_are_right_continuous_and_three_equal_times_are_refused():
    c = FCurve.from_keys([(0, 0), (1, 1), (1, 5), (2, 6)], interpolation="linear")
    assert c(0.999) == pytest.approx(0.999) and c(1.0) == 5.0 and c(1.5) == 5.5
    with pytest.raises(AnimationError) as err:
        FCurve.from_keys([(0, 0), (1, 1), (1, 2), (1, 3)])
    assert err.value.code == "INVALID_KEYS"


# --------------------------------------------------------------------------------------- #
# Easing
# --------------------------------------------------------------------------------------- #


def test_easing_closed_forms_and_back_overshoot():
    c = FCurve.from_keys([{"time": 0, "value": 1, "interpolation": "quad", "ease": "in"}, (2, 5)])
    ts = np.linspace(0, 2, 11)
    u = ts / 2
    v, d, a = c.evaluate(ts, order=2)
    assert (
        np.allclose(v, 1 + 4 * u**2) and np.allclose(d, 4 * 2 * u / 2) and np.allclose(a, 4 * 2 / 4)
    )
    s = FCurve.from_keys([{"time": 0, "value": 0, "interpolation": "sine", "ease": "out"}, (1, 1)])
    assert np.allclose(s.evaluate(ts / 2), np.sin(math.pi * ts / 4), atol=1e-15)
    # Back easing (u^2 ((s + 1) u - s)) overshoots by 4 s^3 / (27 (s + 1)^2): 10 % for
    # Penner's s = 1.70158.
    sb = 1.70158
    b = FCurve.from_keys([{"time": 0, "value": 0, "interpolation": "back", "ease": "out"}, (1, 1)])
    peak = b.evaluate(np.linspace(0, 1, 100001)).max()
    assert peak == pytest.approx(1 + 4 * sb**3 / (27 * (sb + 1) ** 2), abs=1e-9)
    assert peak == pytest.approx(1.1, abs=1e-5)


@pytest.mark.parametrize(
    "interp", ["sine", "quad", "cubic", "quart", "quint", "expo", "circ", "back", "elastic"]
)
@pytest.mark.parametrize("ease", ["in", "out", "in_out"])
def test_easing_derivatives_are_exact(interp, ease):
    c = FCurve.from_keys(
        [{"time": 0.0, "value": -2.0, "interpolation": interp, "ease": ease}, (1.6, 3.0)]
    )
    u = np.linspace(0.02, 0.98, 61)
    keep = np.abs(u - 0.5) > 0.02  # in_out: the halves meet at u = 1/2
    if interp == "circ":
        keep &= (u > 0.05) & (u < 0.95)  # vertical tangents at the steep ends
    t = 1.6 * u[keep]
    v = c.evaluate(t, order=3)
    for k in range(3):
        num = fd(c, t, k, h=1e-6 if interp != "elastic" else 1e-7)
        assert np.allclose(num, v[k + 1], rtol=2e-5, atol=2e-5 * (1 + np.abs(v[k + 1]).max())), k


def test_bounce_and_elastic_end_exactly_at_their_keys():
    for interp in ("bounce", "elastic", "expo"):
        for ease in ("in", "out", "in_out"):
            c = FCurve.from_keys(
                [{"time": 0, "value": 0, "interpolation": interp, "ease": ease}, (1, 1)]
            )
            assert c(0.0) == 0.0 and c(1.0) == 1.0
            assert abs(c(1e-12)) < 1e-6 and abs(c(1 - 1e-12) - 1) < 1e-6


# --------------------------------------------------------------------------------------- #
# Extrapolation
# --------------------------------------------------------------------------------------- #


def test_extrapolation_modes():
    base = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]
    hold = FCurve.from_keys(base)
    assert hold(-5.0) == 0.0 and hold(9.0) == 0.5
    lin = FCurve.from_keys(base, interpolation="linear", extrapolation="linear")
    assert lin(-1.0) == pytest.approx(-1.0) and lin(3.0) == pytest.approx(0.0)
    cyc = FCurve.from_keys(base, handles="auto", extrapolation="cycle")
    off = FCurve.from_keys(base, handles="auto", extrapolation="cycle_offset")
    mir = FCurve.from_keys(base, handles="auto", extrapolation="mirror")
    for t in (0.25, 0.8, 1.6):
        assert cyc(t + 4.0) == pytest.approx(cyc(t), abs=1e-14)
        assert off(t + 4.0) == pytest.approx(off(t) + 2 * 0.5, abs=1e-14)
        assert off(t - 2.0) == pytest.approx(off(t) - 0.5, abs=1e-14)
        assert mir(4.0 - t) == pytest.approx(mir(t), abs=1e-14)  # cycle 1 runs backwards
        assert mir.evaluate(2.0 + t, order=1)[1] == pytest.approx(
            -mir.evaluate(2.0 - t, order=1)[1]
        )
    with pytest.raises(AnimationError):
        FCurve.from_keys([(0, 1)], extrapolation="cycle")


# --------------------------------------------------------------------------------------- #
# Modifiers
# --------------------------------------------------------------------------------------- #


def test_value_modifiers_closed_forms():
    base = FCurve.from_keys([(0, 0), (2, 4)], interpolation="linear")  # v = 2t
    t = np.linspace(0, 2, 9)
    off = base.replace(modifiers=[{"type": "offset", "value": 1.5}])
    assert np.allclose(off.evaluate(t), 2 * t + 1.5)
    sc = base.replace(modifiers=[{"type": "scale", "factor": 3, "pivot": 1}])
    assert np.allclose(sc.evaluate(t), 1 + (2 * t - 1) * 3)
    lim = base.replace(modifiers=[{"type": "limits", "min": 0.5, "max": 3}])
    tl = np.array([0.1, 0.2, 0.3, 1.0, 1.4, 1.6, 1.9])  # (the kinks at 0.25 and 1.5 aside)
    v, d = lim.evaluate(tl, order=1)
    assert np.allclose(v, np.clip(2 * tl, 0.5, 3)) and np.allclose(
        d, np.where((2 * tl > 0.5) & (2 * tl < 3), 2, 0)
    )
    wave = base.replace(modifiers=[{"type": "wave", "amplitude": 0.3, "period": 0.5}])
    assert np.allclose(wave.evaluate(t), 2 * t + 0.3 * np.sin(2 * math.pi * t / 0.5))
    env = base.replace(
        modifiers=[{"type": "envelope", "reference": 0, "points": [[0, 1, 0], [2, 0.5, 1]]}]
    )
    g, o = 1 - 0.25 * t, 0.5 * t
    assert np.allclose(env.evaluate(t), 2 * t * g + o)


def test_time_modifiers_and_chain_order():
    base = FCurve.from_keys([(0, 0), (2, 4)], interpolation="linear")
    step = base.replace(modifiers=[{"type": "stepped", "step": 0.5}])
    t = np.array([0.1, 0.49, 0.5, 1.26, 1.99])
    v, d = step.evaluate(t, order=1)
    assert np.allclose(v, 2 * np.floor(t / 0.5) * 0.5) and np.all(d == 0)
    shift = base.replace(modifiers=[{"type": "time_offset", "offset": 0.5}])
    assert np.allclose(shift.evaluate(t), 2 * np.clip(t - 0.5, 0, 2))
    scale = base.replace(modifiers=[{"type": "time_scale", "factor": 2.0}])
    v, d = scale.evaluate(np.array([0.25, 0.5]), order=1)
    assert np.allclose(v, [1.0, 2.0]) and np.allclose(d, [4.0, 4.0])
    # [noise, stepped] holds the noise too; [stepped, noise] does not.
    noise = {"type": "noise", "strength": 0.5, "scale": 0.1, "seed": 7}
    held = base.replace(modifiers=[noise, {"type": "stepped", "step": 0.5}])
    live = base.replace(modifiers=[{"type": "stepped", "step": 0.5}, noise])
    assert held(0.3) == held(0.1) and live(0.3) != live(0.1)


def test_modifier_influence_and_ranges_blend_with_exact_derivatives():
    base = FCurve.from_keys([(0, 0), (4, 4)], interpolation="linear")
    mod = base.replace(
        modifiers=[
            {
                "type": "offset",
                "value": 2.0,
                "influence": 0.5,
                "range": [1, 3],
                "blend_in": 0.5,
                "blend_out": 1.0,
            }
        ]
    )
    assert mod(0.5) == pytest.approx(0.5)  # outside the range
    assert mod(1.25) == pytest.approx(1.25 + 0.5 * 0.5 * 2.0)  # half way up the ramp
    assert mod(1.75) == pytest.approx(1.75 + 0.5 * 2.0)
    assert mod(2.5) == pytest.approx(2.5 + 0.5 * 0.5 * 2.0)  # half way down
    t = np.array([1.2, 1.7, 2.3, 2.8])
    assert np.allclose(fd(mod, t, 0), mod.evaluate(t, order=1)[1], atol=1e-8)


def test_noise_is_deterministic_smooth_and_has_exact_derivatives():
    x = np.linspace(-3.3, 7.7, 2001)
    a = gradient_noise(x, 5, 3)
    b = gradient_noise(x, 5, 3)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    assert not np.array_equal(a[0], gradient_noise(x, 6, 0)[0])
    assert np.abs(a[0]).max() <= 1.0
    h = 1e-6
    for k in range(3):
        num = (gradient_noise(x + h, 5, 3)[k] - gradient_noise(x - h, 5, 3)[k]) / (2 * h)
        # Quintic fades make the noise C2: its third derivative jumps at lattice points.
        away = np.abs(x - np.round(x)) > 1e-4 if k == 2 else np.ones(x.shape, bool)
        assert np.allclose(num[away], a[k + 1][away], atol=1e-5)
    c = FCurve.from_keys([(0, 0), (1, 0)], interpolation="linear").replace(
        modifiers=[{"type": "noise", "strength": 0.2, "scale": 0.3, "octaves": 3, "seed": 1}]
    )
    t = np.linspace(0.05, 0.95, 40)
    for k in range(2):
        assert np.allclose(fd(c, t, k), c.evaluate(t, order=3)[k + 1], atol=1e-4)


def test_smoothing_is_a_seekable_average_with_matching_derivatives():
    c = FCurve.from_keys([(0, 0), (1, 1), (1, 0), (2, 0)], interpolation="linear")  # a sawtooth
    s = c.replace(modifiers=[{"type": "smooth", "sigma": 0.1}])
    # The documented rule: 15 nodes over +-3 sigma, Gaussian weights normalized.
    x = np.arange(-7, 8) * (3.0 / 7.0)
    w = np.exp(-0.5 * x * x)
    w /= w.sum()
    for t0 in (0.95, 1.0, 1.05, 1.3):
        assert s(t0) == pytest.approx(float(np.sum(w * c.evaluate(t0 + 0.1 * x))), abs=1e-14)
    assert 0.0 < s(1.0) < 1.0  # the jump is softened
    t = np.linspace(0.6, 1.4, 17)
    assert np.array_equal(s.evaluate(t[::-1])[::-1], s.evaluate(t))  # no state: any order
    # Derivatives (a smooth input: the discrete average of a jump jumps where a node
    # crosses it).
    smooth = FCurve.from_arrays([0, 0.7, 1.5, 2], [0, 1, -0.5, 0.2], handles="auto")
    ss = smooth.replace(modifiers=[{"type": "smooth", "sigma": 0.1}])
    tt = t + 0.0123  # (no node on a key, where the C1 input's acceleration jumps)
    for k in range(2):
        assert np.allclose(fd(ss, tt, k, h=1e-6), ss.evaluate(tt, order=3)[k + 1], atol=1e-5)


# --------------------------------------------------------------------------------------- #
# Persistence and validation
# --------------------------------------------------------------------------------------- #


def random_curve(n: int) -> FCurve:
    t = np.cumsum(RNG.uniform(0.01, 2.0, n))
    keys = []
    for i in range(n):
        keys.append(
            {
                "time": float(t[i]),
                "value": float(RNG.normal(0, 3)),
                "weighted": bool(RNG.integers(0, 2)),
                "interpolation": str(
                    RNG.choice(["bezier", "linear", "constant", "back", "elastic", "sine"])
                ),
                "ease": str(RNG.choice(["in", "out", "in_out"])),
                "left": {
                    "type": str(RNG.choice(["free", "aligned", "vector", "auto", "auto_clamped"])),
                    "dt": -float(RNG.uniform(0, 1)),
                    "dv": float(RNG.normal(0, 3)),
                },
                "right": {
                    "type": str(RNG.choice(["free", "aligned", "vector", "auto", "auto_clamped"])),
                    "dt": float(RNG.uniform(0, 1)),
                    "dv": float(RNG.normal(0, 3)),
                },
            }
        )
    return FCurve.from_keys(
        keys,
        extrapolation=["linear", "cycle"] if n > 1 else "linear",
        modifiers=[{"type": "noise", "strength": 0.1, "scale": 0.5, "seed": 3}],
    )


def test_json_round_trips_bit_for_bit_and_handle_recomputation_is_idempotent():
    for _ in range(60):
        c = random_curve(int(RNG.integers(1, 15)))
        c2 = FCurve.from_json(json.loads(json.dumps(c.to_json())))
        assert c.same(c2)
        t = RNG.uniform(-5, 40, 200)
        assert np.array_equal(c.evaluate(t), c2.evaluate(t))
        assert c.same(c.replace({}))


def test_curves_are_immutable():
    c = FCurve.from_keys(KEYS)
    with pytest.raises(ValueError):
        c.times[0] = 5.0
    edited = c.replace({"values": np.array([1.0, 2.0, 3.0, 4.0])})
    assert c.values.tolist() == [0.0, 2.0, -1.0, 3.0] and edited.values.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


@pytest.mark.parametrize(
    ("keys", "code"),
    [
        ([], "INVALID_KEYS"),
        ([(0, float("nan"))], "INVALID_KEYS"),
        ([(float("inf"), 1)], "INVALID_KEYS"),
        ([{"time": 0, "value": 0, "right": {"type": "free", "dt": -1, "dv": 0}}], "INVALID_HANDLE"),
        ([{"time": 0, "value": 0, "left": {"type": "free", "dt": 1, "dv": 0}}], "INVALID_HANDLE"),
        ([{"time": 0, "value": 0, "interpolation": "wobbly"}], "INVALID_INTERPOLATION"),
        ([{"time": 0, "value": 0, "colour": 3}], "INVALID_KEYS"),
        ([{"time": 0, "value": 0, "interpolation": "back", "back": -1}], "INVALID_EASING"),
        ([{"time": "0", "value": 0}], "INVALID_KEYS"),
    ],
)
def test_invalid_keys_are_refused_with_codes(keys, code):
    with pytest.raises(AnimationError) as err:
        FCurve.from_keys(keys)
    assert err.value.code == code


@pytest.mark.parametrize(
    "mod",
    [
        {"type": "noise", "strength": 1},
        {"type": "noise", "strength": 1, "scale": 0},
        {"type": "limits"},
        {"type": "stepped", "step": -1},
        {"type": "offset", "value": 1, "range": [2, 1]},
        {"type": "offset", "value": 1, "blend_in": 1},
        {"type": "teleport"},
        {"type": "smooth", "sigma": 1, "extra": 2},
    ],
)
def test_invalid_modifiers_are_refused(mod):
    with pytest.raises(AnimationError) as err:
        FCurve.from_keys(KEYS, modifiers=[mod])
    assert err.value.code == "INVALID_MODIFIER"
