"""Painting: texel maps, brushes, strokes across UV seams, projection painting with
occlusion, fills, gradients, history and errors."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Mesh, Scene, TexturingError
from aether3d.primitives import create_sphere
from aether3d.texturing.paint import (
    Brush,
    PaintView,
    falloff,
    fill,
    gradient,
    stroke,
    texel_map,
)


def painted_ball(res=256, segments=32):
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, segments, segments // 2))
    ball.unwrap(resolution=res, padding=4)
    doc = scene.add_painted_texture("Paint", res, res)
    doc.add_layer("Ink")
    return scene, ball, doc


def sample_at(doc, obj, points):
    """The document's composite at the texels whose surface points are nearest ``points``
    (exactly: through the texel map)."""
    tm = texel_map(obj.mesh, doc.width, doc.height)
    allpos, allval = [], []
    full = doc.read()
    for t in range(tm.tiles_x * tm.tiles_y):
        tx = tm.texels(t)
        if tx.local.size == 0:
            continue
        x0, y0, _, _ = tm.tile_rect(t)
        ly, lx = np.divmod(tx.local, 64)
        allpos.append(tx.position)
        allval.append(full[y0 + ly, x0 + lx])
    pos = np.concatenate(allpos)
    val = np.concatenate(allval)
    out = []
    for p in np.asarray(points, dtype=float):
        out.append(val[np.argmin(np.linalg.norm(pos - p, axis=1))])
    return np.array(out)


def test_texel_map_points_lie_on_the_surface():
    scene, ball, doc = painted_ball(128)
    tm = texel_map(ball.mesh, 128, 128)
    tx = tm.texels(1)
    assert tx.local.size > 0
    r = np.linalg.norm(tx.position, axis=1)
    assert np.all(r > 0.98) and np.all(r <= 1.0 + 1e-9)  # on the polygonal sphere
    assert np.allclose(np.linalg.norm(tx.normal, axis=1), 1.0)


def test_falloff_curves():
    d = np.linspace(0, 1.2, 13)
    for kind in ("smooth", "linear", "constant", "gaussian"):
        w = falloff(Brush(falloff=kind, hardness=0.3), d)
        assert w[0] == pytest.approx(1.0) and np.all(w[d > 1] == 0)
        assert np.all(np.diff(w[d <= 1]) <= 1e-12)  # never increases outward


def test_a_surface_stroke_paints_both_sides_of_a_uv_seam_alike():
    scene, ball, doc = painted_ball(256)
    # The unwrap cuts the sphere into two hemispheres along the equator; paint across it.
    stroke(
        scene, ball, doc, {"radius": 0.25, "color": "#ff0000", "hardness": 0.9}, [[1.0, 0.0, 0.0]]
    )
    above = sample_at(doc, ball, [[0.995, 0.0, 0.08]])[0]
    below = sample_at(doc, ball, [[0.995, 0.0, -0.08]])[0]
    assert np.allclose(above[:3], [1, 0, 0], atol=0.02)
    assert np.allclose(below[:3], [1, 0, 0], atol=0.02)
    far = sample_at(doc, ball, [[-1.0, 0.0, 0.0]])[0]
    assert np.allclose(far, 1.0)  # untouched (white background)


def test_opacity_caps_a_stroke_and_strength_builds_up():
    scene, ball, doc = painted_ball(128)
    brush = {
        "radius": 0.3,
        "color": "#000000",
        "strength": 0.3,
        "opacity": 0.5,
        "falloff": "constant",
        "spacing": 0.05,
    }
    stroke(scene, ball, doc, brush, [[1, 0, 0], [0.9, 0.3, 0.1]])
    value = sample_at(doc, ball, [[1, 0, 0]])[0][0]
    # Black at most at half opacity over white, whatever the number of dabs: the coverage
    # builds up (1 - 0.7^n after n dabs) toward the 0.5 ceiling and never passes it.
    assert 0.5 - 2 / 65535 <= value < 0.5 + 0.7**15


def test_erase_lowers_the_layers_alpha():
    scene, ball, doc = painted_ball(128)
    stroke(
        scene, ball, doc, {"radius": 0.3, "color": "#00ff00", "falloff": "constant"}, [[1, 0, 0]]
    )
    stroke(scene, ball, doc, {"kind": "erase", "radius": 0.3, "falloff": "constant"}, [[1, 0, 0]])
    assert np.allclose(sample_at(doc, ball, [[1, 0, 0]])[0], 1.0)


def test_symmetric_strokes_mirror_in_object_space():
    scene, ball, doc = painted_ball(128)
    stroke(
        scene, ball, doc, {"radius": 0.2, "color": "#0000ff", "symmetry": "x"}, [[0.8, 0.6, 0.0]]
    )
    a, b = sample_at(doc, ball, [[0.8, 0.6, 0.0], [-0.8, 0.6, 0.0]])
    assert np.allclose(a[:3], [0, 0, 1], atol=0.05) and np.allclose(b[:3], [0, 0, 1], atol=0.05)


def test_one_stroke_is_one_history_step_and_undoes_exactly():
    scene, ball, doc = painted_ball(128)
    before = doc.pixels16().copy()
    n = len(scene.history.entries)
    report = stroke(scene, ball, doc, {"radius": 0.3, "color": "#ff00ff"}, [[1, 0, 0], [0, 1, 0]])
    assert report.dabs > 1 and len(scene.history.entries) == n + 1
    scene.history.undo()
    assert np.array_equal(doc.pixels16(), before)


def test_material_painting_lays_several_textures_in_one_stroke():
    scene, ball, base = painted_ball(128)
    rough = scene.add_painted_texture("Rough", 128, 128, encoding="data")
    rough.add_layer("Ink")
    n = len(scene.history.entries)
    stroke(
        scene,
        ball,
        [{"texture": base, "color": "#ffaa00"}, {"texture": rough, "color": [0.2]}],
        {"radius": 0.3, "falloff": "constant"},
        [[1, 0, 0]],
    )
    assert len(scene.history.entries) == n + 1
    assert sample_at(rough, ball, [[1, 0, 0]])[0][0] == pytest.approx(0.2, abs=1e-4)


def test_view_strokes_respect_occlusion():
    scene, ball, doc = painted_ball(256)
    m = scene.create_material("M")
    m.set_texture("base_color", doc)
    ball.material = m
    # A wall between the camera and the left half of the ball.
    wall = create_sphere(0.3, 8, 4)
    scene.add_object(
        "Blocker", mesh=Mesh(wall.positions, wall.triangles), position=(-0.45, -1.8, 0)
    )
    scene.add_camera("Cam", position=(0, -4, 0), look_at=(0, 0, 0), fov=35)
    view = PaintView.render(scene, "Cam", 256, 256)
    # Paint a horizontal line across the whole ball in screen space.
    stroke(
        scene,
        ball,
        doc,
        {"radius": 12, "color": "#ff0000"},
        [[60, 128], [196, 128]],
        space="view",
        view=view,
    )
    front_right = sample_at(doc, ball, [[0.45, -0.89, 0.0]])[0]
    behind_blocker = sample_at(doc, ball, [[-0.25, -0.97, 0.0]])[0]
    back = sample_at(doc, ball, [[0.0, 1.0, 0.0]])[0]
    assert np.allclose(front_right[:3], [1, 0, 0], atol=0.05)
    assert np.allclose(behind_blocker, 1.0)  # the blocker took the paint
    assert np.allclose(back, 1.0)  # the hidden side was not painted


def test_uv_strokes_paint_texels_directly():
    scene, ball, doc = painted_ball(128)
    stroke(
        scene,
        ball,
        doc,
        {"radius": 4, "color": "#00ff00", "falloff": "constant"},
        [[0.5, 0.5]],
        space="uv",
    )
    assert np.allclose(doc.read(63, 63, 65, 65)[..., 1], 1.0)
    assert np.allclose(doc.read(63, 63, 65, 65)[..., 0], 0.0)


def test_blur_and_smudge_change_only_the_footprint():
    scene, ball, doc = painted_ball(128)
    stroke(
        scene, ball, doc, {"radius": 0.4, "color": "#000000", "falloff": "constant"}, [[1, 0, 0]]
    )
    before = doc.pixels16().copy()
    stroke(scene, ball, doc, {"kind": "blur", "radius": 0.3}, [[0.92, 0.39, 0]])  # on the edge
    after = doc.pixels16()
    changed = np.any(after != before, axis=2)
    assert changed.any()
    far = sample_at(doc, ball, [[-1, 0, 0]])[0]
    assert np.allclose(far, 1.0)
    stroke(scene, ball, doc, {"kind": "smudge", "radius": 0.2}, [[1, 0, 0], [0.7, -0.7, 0]])


def test_fill_islands_and_gradients():
    scene, ball, doc = painted_ball(128)
    fill(scene, ball, doc, "#ff0000", where={"islands": [0]})
    colors = sample_at(doc, ball, [[0, 0, 0.99], [0, 0, -0.99]])
    assert np.allclose(colors[0][:3], [1, 0, 0]) != np.allclose(colors[1][:3], [1, 0, 0])
    gradient(scene, ball, doc, [0, 0, -1], [0, 0, 1], ["#000000", "#ffffff"])
    low, mid, high = sample_at(doc, ball, [[0, 0, -1], [1, 0, 0], [0, 0, 1]])
    assert low[0] < 0.05 and high[0] > 0.95 and 0.4 < mid[0] < 0.6


@pytest.mark.parametrize(
    "brush",
    [
        {"kind": "spray"},
        {"radius": -1},
        {"strength": 2},
        {"falloff": "box"},
        {"symmetry": "xx"},
        {"blend": "dodge"},
        {"color": "nope"},
        {"radius": float("nan")},
        {"wobble": 1},
    ],
)
def test_invalid_brushes_are_refused(brush):
    scene, ball, doc = painted_ball(64)
    with pytest.raises(TexturingError):
        stroke(scene, ball, doc, brush, [[1, 0, 0]])


def test_invalid_strokes_are_refused_and_change_nothing():
    scene, ball, doc = painted_ball(64)
    n = len(scene.history.entries)
    for kwargs in (
        {"points": [[np.nan, 0, 0]]},
        {"points": [[1, 0]]},
        {"pressure": [2.0]},
        {"space": "screen"},
        {"space": "view"},
    ):
        args = {"points": [[1, 0, 0]], **kwargs}
        with pytest.raises(TexturingError):
            stroke(scene, ball, doc, {"radius": 0.1}, **args)
    assert len(scene.history.entries) == n
    with pytest.raises(TexturingError) as exc:
        stroke(scene, ball, doc, {"radius": 0.1}, [[1, 0, 0]], should_stop=lambda: True)
    assert exc.value.code == "CANCELLED"
    assert len(scene.history.entries) == n
