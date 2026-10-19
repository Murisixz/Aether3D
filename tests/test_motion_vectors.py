"""Motion vectors (0.20.0, ``render(motion=dt)``): object, camera and deformation motion
in pixels, on any backend, and as EXR frame channels."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation.frames import render_frames
from aether3d.errors import RenderError
from aether3d.rendering.exr import read_exr

LINEAR = "linear"


def keys(a, b):
    return [{"time": 0, "value": a, "interpolation": LINEAR},
            {"time": 1, "value": b, "interpolation": LINEAR}]  # fmt: skip


def moving_box(camera_moves: bool = False) -> Scene:
    scene = Scene()
    box = scene.add_cube("Box", size=0.5)
    cam = scene.add_camera("Cam", projection="orthographic", ortho_scale=4.0,
                           position=(0, -5, 0), look_at=(0, 0, 0))  # fmt: skip
    if camera_moves:
        scene.animation.set_channel(cam, "transform.position.z", keys(0.0, 2.0))
    else:
        scene.animation.set_channel(box, "transform.position.x", keys(0.0, 1.0))
    return scene


@pytest.mark.parametrize("backend", ["cpu", "path"])
def test_moving_objects_and_cameras_have_their_pixel_motion(backend):
    opts = {"path": {"samples": 1, "bounces": 0}} if backend == "path" else {}
    r = moving_box().render(width=40, height=40, backend=backend, time=0.5, motion=0.1, **opts)
    m = r.passes["motion"]
    box = r.object_index >= 0
    # 0.1 m in 0.1 s at 10 pixels per metre: one pixel left before, one right after.
    assert np.allclose(m[box, 0], -1.0, atol=1e-6) and np.allclose(m[box, 2], 1.0, atol=1e-6)
    assert np.allclose(m[box][:, [1, 3]], 0.0, atol=1e-6)
    assert not m[:10].any() and not m[-10:].any()  # the background has no motion
    r = moving_box(camera_moves=True).render(width=40, height=40, backend=backend, time=0.5,
                                            motion=0.1, **opts)  # fmt: skip
    m = r.passes["motion"]
    box = r.object_index >= 0
    # The camera rises 0.2 m in 0.1 s: the box moves 2 pixels down, then up (rows go down).
    assert np.allclose(m[box, 1], -2.0, atol=1e-6) and np.allclose(m[box, 3], 2.0, atol=1e-6)


def test_deforming_meshes_move_with_their_vertices():
    scene = Scene()
    ball = scene.add_sphere("Ball", radius=0.5, segments=24, rings=12)
    scene.add_camera("Cam", position=(0, -4, 0), look_at=(0, 0, 0))
    scene.animation.set_channel(ball, "transform.scale.z", keys(1.0, 2.0))  # stretches upward
    r = scene.render(width=48, height=48, backend="cpu", time=0.5, motion=0.25)
    m = r.passes["motion"]
    ball_px = r.object_index >= 0
    upper = ball_px & (np.arange(48)[:, None] < 20)
    lower = ball_px & (np.arange(48)[:, None] > 28)
    # Stretching about the centre: the upper half moves up (negative y) toward the next
    # time, the lower half down.
    assert m[upper, 3].mean() < -0.3 and m[lower, 3].mean() > 0.3


def test_motion_frames_and_checks(tmp_path):
    scene = moving_box()
    report = render_frames(scene, tmp_path, start=0.2, end=0.3, fps=10, width=40, height=40,
                           backend="cpu", format="exr", passes=["motion"])  # fmt: skip
    channels = read_exr(report["frames"][0]["path"])
    assert {"motion.X", "motion.Y", "motion.Z", "motion.W"} <= set(channels)
    assert channels["motion.Z"].max() == pytest.approx(1.0, abs=1e-2)  # half floats
    with pytest.raises(RenderError):
        scene.render(width=8, height=8, backend="cpu", motion=-1.0)
    with pytest.raises(RenderError):
        Scene().render(width=8, height=8, backend="cpu", frame=True, motion=0.1)
