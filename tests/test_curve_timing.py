"""Timing operations on keys (0.20.0): shape-keeping insertion, eases, anticipation,
overshoot, holds, value smoothing and the smoothing mode -- through the curve functions,
the animation system and scripts."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation.clip import edit_curve
from aether3d.animation.curves import FCurve
from aether3d.animation.script import run_animation_script
from aether3d.errors import AnimationError

RNG = np.random.default_rng(23)


def random_curve(smoothing: str, handles: str = "auto_clamped") -> FCurve:
    n = int(RNG.integers(3, 12))
    t = np.cumsum(RNG.uniform(0.1, 1.5, n))
    v = RNG.normal(0.0, 2.0, n)
    return FCurve.from_arrays(t, v, handles=handles, smoothing=smoothing)


@pytest.mark.parametrize("smoothing", ["local", "blender", "continuous"])
@pytest.mark.parametrize("handles", ["auto", "auto_clamped"])
def test_inserting_a_key_that_keeps_the_shape(smoothing, handles):
    for _ in range(20):
        c = random_curve(smoothing, handles)
        t0, t1 = c.range
        time = float(RNG.uniform(t0, t1))
        new = edit_curve(c, "insert", {"time": time, "keep_shape": True})
        assert len(new) == len(c) + 1
        grid = np.linspace(t0 - 0.5, t1 + 0.5, 801)
        v0, d0, a0 = c.evaluate(grid, order=2)
        v1, d1, a1 = new.evaluate(grid, order=2)
        scale = 1.0 + float(np.abs(v0).max())
        assert np.abs(v1 - v0).max() <= 1e-10 * scale
        assert np.abs(d1 - d0).max() <= 1e-8 * (1.0 + float(np.abs(d0).max()))
        k = int(np.flatnonzero(new.times == time)[0])
        key = new.key(k)
        assert key["left"]["type"] == key["right"]["type"] == "aligned"
        assert new.slope(k, "left") == pytest.approx(new.slope(k, "right"), rel=1e-9, abs=1e-12)


def test_shape_keeping_insertion_on_other_segments():
    c = FCurve.from_keys(
        [
            {"time": 0, "value": 0, "interpolation": "linear"},
            {"time": 1, "value": 2, "interpolation": "constant"},
            {"time": 2, "value": 1, "interpolation": "sine", "ease": "in_out"},
            {"time": 3, "value": 3},
        ]
    )
    lin = edit_curve(c, "insert", {"time": 0.25, "keep_shape": True})
    assert lin.interpolation[1] == lin.interpolation[0]
    grid = np.linspace(0, 3, 301)
    assert np.allclose(lin.evaluate(grid), c.evaluate(grid), atol=1e-12)
    const = edit_curve(c, "insert", {"time": 1.5, "keep_shape": True})
    assert np.allclose(const.evaluate(grid), c.evaluate(grid), atol=1e-12)
    with pytest.raises(AnimationError) as err:
        edit_curve(c, "insert", {"time": 2.5, "keep_shape": True})
    assert err.value.code == "CANNOT_KEEP_SHAPE" and err.value.hint
    plain = edit_curve(c, "insert", {"time": 2.5})  # without keep_shape: the value there
    assert plain(2.5) == pytest.approx(c(2.5))


def test_ease_sets_flat_handles_of_the_given_influence():
    c = FCurve.from_keys([(0, 0), (1, 1), (3, 1.5), (4, 3)])
    e = edit_curve(c, "ease", {"keys": [1], "ease_in": 0.5, "ease_out": 0.25})
    k = e.key(1)
    assert k["left"]["dt"] == pytest.approx(-0.5) and k["right"]["dt"] == pytest.approx(0.5)
    assert e.slope(1, "left") == 0.0 == e.slope(1, "right") and k["weighted"]
    moving = edit_curve(c, "ease", {"keys": [1], "ease_out": 0.4, "slope": 0.2})
    assert moving.slope(1, "right") == pytest.approx(0.2)
    assert moving.slope(1, "left") == pytest.approx(0.2)  # aligned
    # A longer ease lingers longer near the key.
    short = edit_curve(c, "ease", {"keys": [1], "ease_out": 0.1})
    long = edit_curve(c, "ease", {"keys": [1], "ease_out": 0.9})
    assert abs(long(1.4) - 1.0) < abs(short(1.4) - 1.0)
    with pytest.raises(AnimationError):
        edit_curve(c, "ease", {"keys": [1]})
    with pytest.raises(AnimationError):
        edit_curve(c, "ease", {"keys": [1], "ease_in": 1.5})


def test_anticipation_and_overshoot_follow_the_move():
    c = FCurve.from_keys([(0, 0), (1, 10), (3, 10)])
    a = edit_curve(c, "anticipate", {"keys": [0], "amount": 0.2, "at": 0.3})
    grid = np.linspace(0, 1, 1001)
    assert a.evaluate(grid).min() == pytest.approx(-2.0, abs=1e-9)  # 20 % the other way
    assert a.times[1] == pytest.approx(0.3)
    o = edit_curve(c, "overshoot", {"keys": [1], "amount": 0.1, "settle": 0.6})
    after = o.evaluate(np.linspace(1, 3, 2001))
    assert after.max() == pytest.approx(11.0, abs=1e-6)  # 10 % beyond the end
    assert o(1.6) == pytest.approx(10.0) and o(3.0) == pytest.approx(10.0)
    _, d = o.evaluate(np.array([1.0 - 1e-9, 1.0 + 1e-9]), order=1)
    assert d[0] > 0 and d[1] > 0  # passes through the target, still moving
    swings = edit_curve(
        c, "overshoot", {"keys": [1], "amount": 0.2, "settle": 0.9, "oscillations": 3, "decay": 0.5}
    )
    assert len(swings) == len(c) + 4
    assert swings.values[2:5].tolist() == pytest.approx([12.0, 9.0, 10.5])
    with pytest.raises(AnimationError):
        edit_curve(c, "overshoot", {"keys": [1], "settle": 5.0})  # past the next key
    with pytest.raises(AnimationError):
        edit_curve(c, "anticipate", {"keys": [2]})  # no move starts there


def test_holds_insert_time_and_drift():
    c = FCurve.from_keys([(0, 0), (1, 1), (2, 3)])
    h = edit_curve(c, "hold", {"keys": [1], "duration": 0.5, "drift": 0.1})
    assert h.times.tolist() == pytest.approx([0, 1, 1.5, 2.5])
    assert h.values.tolist() == pytest.approx([0, 1, 1.2, 3])
    inside = edit_curve(c, "hold", {"keys": [1], "duration": 0.4, "shift": False})
    assert inside.times.tolist() == pytest.approx([0, 1, 1.4, 2])
    with pytest.raises(AnimationError):
        edit_curve(c, "hold", {"keys": [1], "duration": 2.0, "shift": False})


def test_smoothing_key_values_and_the_smoothing_mode():
    t = np.linspace(0, 2, 21)
    v = np.sin(t) + RNG.normal(0, 0.05, t.size)
    c = FCurve.from_arrays(t, v)
    s = edit_curve(c, "smooth", {"sigma": 0.1})
    assert s.values[0] == c.values[0] and s.values[-1] == c.values[-1]
    rough = np.abs(np.diff(c.values, 2)).mean()
    assert np.abs(np.diff(s.values, 2)).mean() < 0.6 * rough
    assert c.smoothing == "continuous"
    local = edit_curve(c, "smoothing", {"smoothing": "local"})
    assert local.smoothing == "local" and np.array_equal(local.values, c.values)


def test_timing_operations_through_the_system_and_scripts():
    scene = Scene()
    box = scene.add_cube("Box", size=0.5)
    anim = scene.animation
    anim.set_channel(box, "transform.position.x", [(0, 0), (1, 2), (2, 2)])
    anim.set_channel(box, "transform.position.y", [(0, 0), (1, 1), (2, 1)])
    anim.edit_keys(box, "transform.position", "anticipate", keys=[0], amount=0.1)
    x = anim.channel(box, "transform.position.x")
    y = anim.channel(box, "transform.position.y")
    assert len(x) == len(y) == 4  # both components, the same timing
    assert x.values[1] == pytest.approx(-0.2) and y.values[1] == pytest.approx(-0.1)
    scene.history.undo()
    assert len(anim.channel(box, "transform.position.x")) == 3
    run_animation_script(
        scene,
        {
            "format": "aether3d.animation-script",
            "version": 1,
            "operations": [
                {
                    "op": "set_channel",
                    "target": "Box",
                    "path": "transform.position.z",
                    "keys": [[0, 0], [1, 1], [2, 0.5]],
                    "smoothing": "blender",
                },
                {
                    "op": "edit_keys",
                    "target": "Box",
                    "path": "transform.position.z",
                    "edit": "insert",
                    "time": 0.5,
                    "keep_shape": True,
                },
                {
                    "op": "edit_keys",
                    "target": "Box",
                    "path": "transform.position.z",
                    "edit": "overshoot",
                    "keys": [2],
                    "amount": 0.2,
                    "settle": 0.4,
                },
            ],
        },  # fmt: skip
    )
    z = anim.channel(box, "transform.position.z")
    assert z.smoothing == "blender" and len(z) == 6


def test_graphs_draw_jerk_and_the_exact_analysis():
    from aether3d import Scene

    scene = Scene()
    box = scene.add_cube("B")
    anim = scene.animation
    anim.set_channel(box, "transform.position.x", [
        {"time": 0, "value": 0, "interpolation": "linear"},
        {"time": 1, "value": 2, "interpolation": "back", "ease": "out", "back": 2.5},
        {"time": 2, "value": 1},
    ])  # fmt: skip
    plots = ("value", "velocity", "acceleration", "jerk")
    svg = anim.graph([(box, "transform.position.x")], plots=plots)
    assert ">jerk<" in svg and ">C1<" in svg  # the linear segment's corner
    assert "<polygon" in svg and (">max<" in svg or ">min<" in svg)  # the back overshoot
    assert ">0<" in svg and ">2<" in svg  # keys are numbered
    plain = anim.graph([(box, "transform.position.x")], plots=plots, annotate=False)
    assert ">C1<" not in plain and "<polygon" not in plain
    image = anim.graph([(box, "transform.position.x")], plots=("value", "jerk"), format="png")
    assert image.shape[1] == 900 and image.dtype.name == "uint8"


def test_graphs_draw_reference_samples_to_compare_with():
    from aether3d.animation.graph import graph_png, graph_svg
    from aether3d.errors import AnimationError

    curve = FCurve.from_keys([(0.0, 0.0), (1.0, 1.0)], interpolation="linear")
    t = np.linspace(0.0, 1.0, 11)
    refs = {"Other program": (t, t + 5.0)}  # (far above: it widens the value range)
    svg = graph_svg({"Ours": curve}, references=refs, plots=("value",))
    # Eleven crosses (two strokes each) and one in the legend; the range covers them.
    ink = 'stroke="rgb(26,26,26)" stroke-width="0.9"/>'  # (the first reference's)
    assert svg.count(ink) == 2 * 12 and ">Other program<" in svg
    assert ">6<" in svg
    plain = graph_png({"Ours": curve}, plots=("value",))
    marked = graph_png({"Ours": curve}, references=refs, plots=("value",))
    assert plain.shape == marked.shape and (plain != marked).any()
    for bad in ({"x": (t, t[:-1])}, {"x": (t, np.full(11, np.nan))}, {"x": 3},
                {"x": (np.zeros(20_001), np.zeros(20_001))}):  # fmt: skip
        with pytest.raises(AnimationError):
            graph_svg({"Ours": curve}, references=bad)
