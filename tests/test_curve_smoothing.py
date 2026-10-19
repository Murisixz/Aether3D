"""Automatic handles in the Blender-compatible smoothing modes (``"blender"``,
``"continuous"``) against Blender 5.2's own handles and values, and their mathematics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether3d.animation.curves import FCurve
from aether3d.animation.report import _limits

DATA = json.loads((Path(__file__).parent / "data" / "blender_curves" / "curves.json").read_text())
CASES = DATA["cases"]
RNG = np.random.default_rng(19)


def build(case: dict) -> FCurve:
    rows = []
    for i, (time, value) in enumerate(case["keys"]):
        row = {"time": time, "value": value, "interpolation": case["interpolation"][i]}
        for side in ("left", "right"):
            kind = case[f"{side}_type"][i]
            if kind in ("free", "aligned"):
                hx, hy = case[side][i]
                row[side] = {"type": kind, "dt": hx - time, "dv": hy - value}
                row["weighted"] = True  # Blender keeps a free handle's own extent
            else:
                row[side] = kind
        rows.append(row)
    return FCurve.from_keys(rows, extrapolation=case["extrapolation"], smoothing=case["smoothing"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_handles_and_values_are_blenders(name):
    case = CASES[name]
    curve = build(case)
    a = curve.arrays()
    ours = np.stack(
        [
            a["times"] + a["left_dt"],
            a["values"] + a["left_dv"],
            a["times"] + a["right_dt"],
            a["values"] + a["right_dv"],
        ],
        1,
    )
    ref = np.hstack([np.array(case["left"]), np.array(case["right"])])
    if case["extrapolation"] == "cycle":  # the ends' outer extents never shape the curve
        ours[0, :2] = ref[0, :2]
        ours[-1, 2:] = ref[-1, 2:]
    scale = max(1.0, float(np.abs(ref[:, [1, 3]]).max()))
    # Blender stores float32: agreement to its rounding.
    assert np.abs(ours - ref).max() <= 2e-6 * scale
    values = curve.evaluate(np.array(case["frames"]))
    assert np.abs(values - np.array(case["values"])).max() <= 2e-6 * scale


def acceleration_jumps(curve: FCurve, keys: np.ndarray) -> np.ndarray:
    """Exact one-sided acceleration limits at the keys at ``keys`` (their times)."""
    index = np.searchsorted(curve.times, keys)
    jumps = []
    for k in index.tolist():
        lim = _limits(curve, int(k))
        jumps.append(abs(lim["right"][2] - lim["left"][2]))
    return np.array(jumps)


def test_continuous_acceleration_at_every_automatic_key():
    jumps_local = []
    for _ in range(50):
        n = int(RNG.integers(3, 30))
        t = np.cumsum(RNG.uniform(0.05, 2.0, n))
        v = RNG.normal(0.0, 3.0, n)
        continuous = FCurve.from_arrays(t, v, handles="auto", smoothing="continuous")
        local = FCurve.from_arrays(t, v, handles="auto", smoothing="local")
        scale = float(np.abs(local.evaluate(t, order=2)[2]).max()) + 1.0
        assert acceleration_jumps(continuous, t[1:-1]).max() <= 1e-4 * scale
        jumps_local.append(acceleration_jumps(local, t[1:-1]).max() / scale)
        # The values still pass through the keys, and the slopes are continuous.
        assert np.allclose(continuous.evaluate(t), v, atol=1e-12)
        _, d_before = continuous.evaluate(t[1:-1] - 1e-9, order=1)
        _, d_after = continuous.evaluate(t[1:-1] + 1e-9, order=1)
        assert np.allclose(d_before, d_after, atol=1e-5 * scale)
    # What it improves: the local rule's cubic Hermite splines are only C1.
    assert np.median(jumps_local) > 0.1


def test_continuous_clamped_curves_never_overshoot():
    for trial in range(300):
        n = int(RNG.integers(3, 25))
        t = np.cumsum(RNG.uniform(1e-3, 3.0, n))
        v = RNG.normal(0, 5, n)
        if trial % 2:
            v = np.cumsum(np.abs(v))
        c = FCurve.from_arrays(t, v, handles="auto_clamped", smoothing="continuous")
        for k in range(n - 1):
            ys = c.evaluate(np.linspace(t[k], t[k + 1], 48))
            lo, hi = min(v[k], v[k + 1]), max(v[k], v[k + 1])
            span = hi - lo
            assert ys.max() <= hi + 1e-12 * (1 + span) and ys.min() >= lo - 1e-12 * (1 + span)
            if trial % 2:
                assert np.all(np.diff(ys) >= -1e-12 * (1 + span))
    peak = FCurve.from_arrays([0, 1, 2], [0, 5, 1], handles="auto_clamped", smoothing="continuous")
    assert peak.slope(1, "left") == 0.0 == peak.slope(1, "right")


def test_a_cyclic_curve_is_c2_across_its_seam():
    keys = [(0.0, 0.0), (0.3, 1.0), (0.5, 0.2), (0.8, -0.7), (1.0, 0.5)]
    for mode in ("cycle", "cycle_offset"):
        c = FCurve.from_keys(keys, handles="auto", extrapolation=mode, smoothing="continuous")
        v, d, a = c.evaluate(np.array([1.0 - 1e-8, 1.0 + 1e-8]), order=2)
        assert d[0] == pytest.approx(d[1], abs=1e-5)
        assert a[0] == pytest.approx(a[1], abs=1e-4)
        assert acceleration_jumps(c, np.array([0.3, 0.5, 0.8])).max() < 1e-4


def test_blender_mode_slopes_and_extents():
    t = np.array([0.0, 2.0, 15.0, 16.0])  # 13 > 5 x 2: the long interval is limited
    v = np.array([0.0, 1.0, 1.0, 3.0])
    c = FCurve.from_arrays(t, v, handles="auto", smoothing="blender")
    k = c.key(1)
    assert c.slope(1, "right") == pytest.approx(0.5 * (0.5 + 0.0))  # the secants' mean
    assert k["left"]["dt"] == pytest.approx(-2.0 / 2.5614)
    assert k["right"]["dt"] == pytest.approx(10.0 / 2.5614)  # min(13, 5 x 2)
    assert c.slope(0, "right") == 0.0  # constant extrapolation: flat end
    linear = FCurve.from_arrays(t, v, handles="auto", smoothing="blender", extrapolation="linear")
    assert linear.slope(0, "right") == pytest.approx(0.5)  # the secant


def test_smoothing_is_kept_by_edits_json_and_defaults():
    c = FCurve.from_keys([(0, 0), (1, 1), (3, 2), (4, 4)])
    assert c.smoothing == "continuous"  # new curves: Blender's default
    data = c.to_json()
    assert data["smoothing"] == "continuous"
    back = FCurve.from_json(json.loads(json.dumps(data)))
    assert back.same(c)
    moved = c.replace({"values": [0.0, 1.5, 2.0, 4.0]})
    assert moved.smoothing == "continuous"
    assert moved.slope(1, "right") != c.slope(1, "right")
    # A document written before 0.20.0 has no smoothing: Aether3D's local rule.
    old = {"keys": {"time": [0, 1, 3, 4], "value": [0, 1, 2, 4]}}
    legacy = FCurve.from_json(old)
    assert legacy.smoothing == "local" and "smoothing" not in legacy.to_json()
    local = FCurve.from_keys([(0, 0), (1, 1), (3, 2), (4, 4)], smoothing="local")
    assert legacy.same(local)
    with pytest.raises(Exception) as err:
        FCurve.from_keys([(0, 0), (1, 1)], smoothing="cubic")
    assert getattr(err.value, "code", "") == "INVALID_SMOOTHING"


def test_large_continuous_curves():
    n = 20_000
    t = np.cumsum(RNG.uniform(0.01, 0.1, n))
    v = np.cumsum(RNG.normal(0.0, 0.1, n))
    c = FCurve.from_arrays(t, v, handles="auto", smoothing="continuous")
    probe = t[1:-1:997]
    scale = float(np.abs(c.evaluate(t, order=2)[2]).max()) + 1.0
    assert acceleration_jumps(c, probe).max() <= 1e-4 * scale


MODIFIERS = DATA["modifiers"]


@pytest.mark.parametrize("name", sorted(MODIFIERS))
def test_modifiers_are_blenders(name):
    case = MODIFIERS[name]
    t, v = np.array(case["keys"]).T
    c = FCurve.from_arrays(
        t, v, handles="auto_clamped", smoothing="continuous", modifiers=[case["modifier"]]
    )
    ours = c.evaluate(np.array(case["frames"]))
    assert np.abs(ours - np.array(case["values"])).max() <= 4e-6


@pytest.mark.parametrize(
    "modifier",
    [
        {"type": "generator", "coefficients": [0.2, -0.3, 0.05, 0.01]},
        {"type": "generator", "coefficients": [1.0, 2.0], "additive": False},
        {"type": "function", "function": "sin", "amplitude": 0.4, "phase_multiplier": 3.0},
        {"type": "function", "function": "cos", "phase_offset": 0.3},
        {"type": "function", "function": "sinc", "phase_multiplier": 4.0, "phase_offset": -6.0},
        {"type": "function", "function": "sqrt", "phase_offset": 0.5},
        {"type": "function", "function": "ln", "phase_offset": 0.5, "amplitude": 0.3},
        {"type": "cycles", "before": "mirror", "after": "repeat_offset"},
        {"type": "cycles", "after": "mirror", "count_after": 3},
    ],
)
def test_new_modifiers_have_exact_derivatives(modifier):
    c = FCurve.from_keys([(0.0, 0.0), (0.4, 1.0), (1.1, 0.3), (1.5, 0.8)], modifiers=[modifier])
    times = np.array([-2.3, -0.7, 0.1, 0.55, 0.9, 1.3, 2.2, 3.7, 5.1])
    eps = 1e-5
    v, d, a = c.evaluate(times, order=2)
    vp, dp = c.evaluate(times + eps, order=1)
    vm, dm = c.evaluate(times - eps, order=1)
    assert np.allclose(d, (vp - vm) / (2 * eps), rtol=1e-5, atol=1e-5)
    assert np.allclose(a, (dp - dm) / (2 * eps), rtol=1e-4, atol=1e-3)
    back = FCurve.from_json(json.loads(json.dumps(c.to_json())))
    assert back.same(c)


def test_an_endless_cycles_modifier_makes_handles_cycle_aware():
    keys = [(0.0, 0.0), (0.3, 1.0), (0.5, 0.2), (0.8, -0.7), (1.0, 0.0)]
    by_modifier = FCurve.from_keys(keys, modifiers=[{"type": "cycles"}])
    by_extrapolation = FCurve.from_keys(keys, extrapolation="cycle")
    for i in (0, len(keys) - 1):
        assert by_modifier.slope(i, "right") == pytest.approx(by_extrapolation.slope(i, "right"))
    _, d, a = by_modifier.evaluate(np.array([1.0 - 1e-8, 1.0 + 1e-8]), order=2)
    assert d[0] == pytest.approx(d[1], abs=1e-5) and a[0] == pytest.approx(a[1], abs=1e-4)
    counted = FCurve.from_keys(keys, modifiers=[{"type": "cycles", "count_after": 2}])
    assert counted.slope(0, "right") == 0.0  # counted cycles: ordinary (flat) ends
    with pytest.raises(Exception) as err:
        FCurve.from_keys(keys, modifiers=[{"type": "function", "function": "tan"}])
    assert getattr(err.value, "code", "") == "INVALID_MODIFIER"
