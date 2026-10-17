"""Quaternion animation: slerp, squad, cubic (glTF), timing, paths, Euler, determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d import math3d
from aether3d.animation import FCurve
from aether3d.animation import rotation as R
from aether3d.animation.tracks import RotationTrack
from aether3d.errors import AnimationError

RNG = np.random.default_rng(23)


def euler_z(q) -> float:
    return float(R.to_euler(q)[0][2])


def test_euler_conversions_follow_math3d_conventions():
    for _ in range(300):
        e = RNG.uniform(-180, 180, 3)
        expected = np.array(math3d.quat_from_euler(e))
        got = R.from_euler(e)[0]
        got = got if got[3] >= 0 else -got
        assert np.allclose(got, expected, atol=1e-14)
        assert np.allclose(R.to_euler(expected)[0], math3d.quat_to_euler(expected), atol=1e-9)
        m = R.matrices(expected[None])[0]
        assert np.allclose(m, math3d.quat_to_matrix3(expected), atol=1e-15)
        back = R.from_matrices(m[None])[0]
        assert np.allclose(back, expected, atol=1e-14)


def test_slerp_keeps_unit_norm_and_constant_angular_speed():
    a = R.normalize(RNG.normal(size=(400, 4)))
    b = R.normalize(RNG.normal(size=(400, 4)))
    full = R.angle(a, b)
    for u in (0.0, 0.1, 0.5, 0.9, 1.0):
        s = R.slerp(a, b, np.full(400, u))
        assert np.allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-14)
        assert np.allclose(R.angle(a, s), u * full, atol=1e-12)
    # Nearly identical orientations: no division by a vanishing sine.
    c = R.normalize(a + 1e-9 * RNG.normal(size=a.shape))
    assert np.all(np.isfinite(R.slerp(a, c, np.full(400, 0.3))))


def test_shortest_and_stored_paths():
    q270 = [0.0, 0.0, math.sin(math.radians(135)), math.cos(math.radians(135))]
    short = RotationTrack.from_keys([(0, [0, 0, 0, 1]), (1, q270)])
    long_way = RotationTrack.from_keys(
        [{"time": 0, "rotation": [0, 0, 0, 1], "path": "stored"}, (1, q270)]
    )
    assert euler_z(short.evaluate(0.5)) == pytest.approx(-45.0)  # -90 the short way
    assert euler_z(long_way.evaluate(0.5)) == pytest.approx(135.0)  # +270 as stored
    # A stored path keeps the arc between the stored quaternions even after an earlier
    # key flipped the sign sequence.
    # (-1 * identity to -q270 is the long 270 degree arc; the first segment flips the
    # sign sequence, and the stored arc must survive that.)
    neg_identity = [0.0, 0.0, 0.0, -1.0]
    neg_q270 = [-c for c in q270]
    tr = RotationTrack.from_keys(
        [(0, [0, 0, 0, 1]), {"time": 1, "rotation": neg_identity, "path": "stored"}, (2, neg_q270)]
    )
    assert euler_z(tr.evaluate(1.5)) == pytest.approx(135.0)
    short_second = RotationTrack.from_keys(
        [(0, [0, 0, 0, 1]), {"time": 1, "rotation": neg_identity, "path": "stored"}, (2, q270)]
    )
    assert euler_z(short_second.evaluate(1.5)) == pytest.approx(-45.0)  # that arc is short


def test_slerp_tracks_have_exact_angular_velocity_and_acceleration_with_timing():
    for timing in (
        "linear",
        {"easing": "sine", "ease": "in_out"},
        {"bezier": [0.3, 0.0, 0.2, 1.0]},
    ):
        tr = RotationTrack.from_keys(
            [
                {"time": 0.0, "euler": [10, -20, 5], "timing": timing},
                {"time": 1.7, "euler": [40, 30, 100]},
            ]
        )
        for t in (0.2, 0.85, 1.4):
            h = 1e-6
            q0, q1 = tr.evaluate(t - h), tr.evaluate(t + h)
            omega_num = 2 * R.log(R.multiply(q1, R.conjugate(q0))) / (2 * h)
            _, omega, alpha = tr.evaluate(t, order=2)
            assert np.allclose(omega, omega_num, atol=1e-6)
            w0, w1 = tr.evaluate(t - h, order=1)[1], tr.evaluate(t + h, order=1)[1]
            assert np.allclose(alpha, (w1 - w0) / (2 * h), atol=1e-4)
    # Peak angular speed of a 90 degree sine in-out turn over 1 s is (pi/2)(pi/2).
    tr = RotationTrack.from_keys(
        [{"time": 0, "euler": [0, 0, 0], "timing": "sine"}, {"time": 1, "euler": [0, 0, 90]}]
    )
    assert tr.evaluate(0.5, order=1)[1][2] == pytest.approx(math.pi**2 / 4)


def test_squad_has_continuous_angular_velocity_where_slerp_jumps():
    qs = R.normalize(RNG.normal(size=(6, 4)))
    keys = [(float(i), qs[i].tolist()) for i in range(6)]
    squad = RotationTrack.from_keys(keys, interpolation="squad")
    slerp = RotationTrack.from_keys(keys, interpolation="slerp")
    for k in (1, 2, 3, 4):
        assert np.allclose(squad.evaluate(float(k)), slerp.evaluate(float(k)))  # through keys
        jump_squad = np.linalg.norm(
            squad.evaluate(k - 1e-4, order=1)[1] - squad.evaluate(k + 1e-4, order=1)[1]
        )
        jump_slerp = np.linalg.norm(
            slerp.evaluate(k - 1e-4, order=1)[1] - slerp.evaluate(k + 1e-4, order=1)[1]
        )
        assert jump_squad < 5e-3 < 0.1 < jump_slerp


def test_gltf_cubic_matches_the_specification_formula():
    v = R.normalize(RNG.normal(size=(3, 4)))
    v[1] = R.align(v[0], v[1])
    v[2] = R.align(v[1], v[2])
    a = RNG.normal(size=(3, 4)) * 0.3
    b = RNG.normal(size=(3, 4)) * 0.3
    times = [0.0, 0.8, 2.0]
    tr = RotationTrack.from_keys(
        [
            {
                "time": times[i],
                "rotation": v[i].tolist(),
                "interpolation": "cubic",
                "in": a[i].tolist(),
                "out": b[i].tolist(),
            }
            for i in range(3)
        ]
    )
    t = 1.3
    k, td = 1, times[2] - times[1]
    s = (t - times[1]) / td
    p = (
        (2 * s**3 - 3 * s**2 + 1) * v[k]
        + td * (s**3 - 2 * s**2 + s) * b[k]
        + (-2 * s**3 + 3 * s**2) * v[k + 1]
        + td * (s**3 - s**2) * a[k + 1]
    )
    assert np.allclose(tr.evaluate(t), p / np.linalg.norm(p), atol=1e-15)


def test_rotation_extrapolation_cycles_and_turns():
    q30 = R.from_euler([0, 0, 30])[0].tolist()
    turn = RotationTrack.from_keys([(0, [0, 0, 0, 1]), (1, q30)], extrapolation="cycle_offset")
    assert euler_z(turn.evaluate(2.5)) == pytest.approx(75.0)
    assert euler_z(turn.evaluate(-0.5)) == pytest.approx(-15.0)
    cyc = RotationTrack.from_keys([(0, [0, 0, 0, 1]), (1, q30)], extrapolation="cycle")
    assert euler_z(cyc.evaluate(3.25)) == pytest.approx(7.5)
    mir = RotationTrack.from_keys([(0, [0, 0, 0, 1]), (1, q30)], extrapolation="mirror")
    assert euler_z(mir.evaluate(1.25)) == pytest.approx(22.5)


def test_euler_filter_and_unwrap_make_sequences_continuous():
    angles = np.array([170.0, 179.0, -178.0, -170.0, 175.0])
    assert np.all(np.abs(np.diff(R.unwrap_degrees(angles))) <= 180.0)
    seq = np.array([[0, 0, 170], [0, 0, -175], [180, 180, 0], [0, 0, -150]], float)
    filtered = R.euler_filter(seq)
    assert np.all(np.abs(np.diff(filtered, axis=0)).sum(axis=1) <= 60.0)
    a = R.from_euler(seq)
    b = R.from_euler(filtered)
    assert np.allclose(np.abs(np.sum(a * b, axis=1)), 1.0, atol=1e-12)  # same rotations


def test_additive_and_blend_composition():
    base = R.from_euler([[0, 0, 10]])
    ref = R.from_euler([[0, 0, 0]])
    value = R.from_euler([[0, 0, 40]])
    full = R.additive(base, value, ref, np.array([1.0]))
    half = R.additive(base, value, ref, np.array([0.5]))
    assert euler_z(full) == pytest.approx(50.0) and euler_z(half) == pytest.approx(30.0)
    mid = R.blend(base, value, np.array([0.5]))
    assert euler_z(mid) == pytest.approx(25.0)


def test_evaluation_does_not_depend_on_the_other_times_in_a_batch():
    """Re-evaluating time t gives the same bits whether alone, in any batch, in any order."""
    keys = []
    t = np.cumsum(RNG.uniform(0.05, 1.0, 12))
    for i in range(12):
        keys.append(
            {
                "time": float(t[i]),
                "value": float(RNG.normal()),
                "weighted": bool(i % 2),
                "interpolation": ["bezier", "back", "elastic", "linear"][i % 4],
                "left": {"type": "free", "dt": -0.3, "dv": float(RNG.normal())},
                "right": {"type": "free", "dt": 0.4, "dv": float(RNG.normal())},
            }
        )
    curve = FCurve.from_keys(
        keys,
        extrapolation="cycle_offset",
        modifiers=[
            {"type": "noise", "strength": 0.2, "scale": 0.3, "seed": 2},
            {"type": "smooth", "sigma": 0.05},
        ],
    )
    qs = R.normalize(RNG.normal(size=(8, 4)))
    track = RotationTrack.from_keys(
        [
            {
                "time": float(i) * 0.7,
                "rotation": qs[i].tolist(),
                "interpolation": ["slerp", "squad", "constant"][i % 3],
            }
            for i in range(8)
        ],
        extrapolation="mirror",
    )
    batch = RNG.uniform(-3, 20, 500)
    probe = batch[::37]
    full = curve.evaluate(batch, order=3)
    qfull = track.evaluate(batch, order=2)
    for t0, idx in zip(probe, range(0, 500, 37), strict=True):
        alone = curve.evaluate(np.array([t0]), order=3)
        for k in range(4):
            assert alone[k][0].tobytes() == full[k][idx].tobytes()
        qa = track.evaluate(np.array([t0]), order=2)
        for k in range(3):
            assert qa[k][0].tobytes() == qfull[k][idx].tobytes()
    rev = curve.evaluate(batch[::-1], order=3)
    assert all(np.array_equal(a[::-1], b) for a, b in zip(rev, full, strict=True))


@pytest.mark.parametrize(
    "keys",
    [
        [],
        [(0, [0, 0, 0, 0])],
        [(0, [0, 0, 1])],
        [(0, [0, 0, 0, 1]), (0, [0, 0, 0, 1])],
        [{"time": 0, "rotation": [0, 0, 0, 1], "euler": [0, 0, 0]}],
        [{"time": 0, "rotation": [0, 0, 0, 1], "timing": {"bezier": [2, 0, 0, 1]}}],
        [{"time": float("nan"), "rotation": [0, 0, 0, 1]}],
    ],
)
def test_invalid_rotation_keys_are_refused(keys):
    with pytest.raises(AnimationError):
        RotationTrack.from_keys(keys)
