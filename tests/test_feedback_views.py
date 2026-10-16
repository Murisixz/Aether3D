"""Views for observations: virtual cameras, presets and framed close-ups (Stage 12).

A virtual camera renders and projects exactly like a scene camera with the same lens and
frame, and observing through any view leaves the scene untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import FeedbackError, RenderError, Scene
from aether3d.feedback import camera_view, check_view_spec, observe, resolve_view, virtual_camera
from aether3d.rendering.camera import CameraFrame
from aether3d.rendering.scene_data import RenderCamera
from aether3d.serialization import scene_to_dict


def _scene() -> Scene:
    scene = Scene("Views")
    scene.add_plane("Ground", size=10)
    scene.add_cube("Crate", size=1, position=(1, 0, 0.5))
    scene.add_sphere("Ball", radius=0.4, position=(-1.5, 0.5, 0.4))
    cam = scene.add_camera("Cam", position=(3, -6, 3), look_at=(0, 0, 0.5))
    cam.camera.fov = 45
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=1.5)
    return scene


def test_a_virtual_twin_of_a_camera_renders_bit_for_bit_like_it():
    scene = _scene()
    twin = camera_view(scene, "Cam")
    assert twin.object_id is None
    a = scene.render(width=96, height=54)
    b = scene.render(width=96, height=54, camera=twin)
    for name in ("image", "linear", "depth", "object_index", "coverage"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert b.camera["id"] is None and b.camera["name"] == "Cam"


def test_a_virtual_twin_renders_bit_for_bit_on_the_gpu(gpu):
    scene = _scene()
    a = gpu.render(scene, width=96, height=54)
    b = gpu.render(scene, width=96, height=54, camera=camera_view(scene, "Cam"))
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.object_index, b.object_index)


def test_spatial_queries_take_virtual_cameras():
    scene = _scene()
    twin = camera_view(scene, "Cam")
    points = [(1, 0, 1), (-1.5, 0.5, 0.4), (0, 3, 0)]
    for p, q in zip(
        scene.project(points, width=200, height=100),
        scene.project(points, camera=twin, width=200, height=100),
        strict=True,
    ):
        assert (p.x, p.y, p.depth, p.in_view) == (q.x, q.y, q.depth, q.in_view)
    hit = scene.pick(100, 60, width=200, height=100)
    hit2 = scene.pick(100, 60, camera=twin, width=200, height=100)
    assert hit.object.name == hit2.object.name and hit.position == hit2.position


def test_render_refuses_malformed_virtual_cameras():
    scene = _scene()
    twin = camera_view(scene, "Cam")
    claimed = RenderCamera(scene.get_by_name("Cam").id, "x", twin.camera, twin.frame)
    with pytest.raises(RenderError) as exc:
        scene.render(width=32, height=18, camera=claimed)
    assert exc.value.code == "INVALID_CAMERA"
    f = twin.frame
    skewed = CameraFrame(f.position, f.forward, f.up + 0.1 * f.forward, f.right)
    with pytest.raises(RenderError) as exc:
        scene.render(width=32, height=18, camera=RenderCamera(None, "x", twin.camera, skewed))
    assert exc.value.code == "INVALID_CAMERA"
    bad = CameraFrame(np.array([np.nan, 0, 0]), f.forward, f.up, f.right)
    with pytest.raises(RenderError):
        scene.render(width=32, height=18, camera=RenderCamera(None, "x", twin.camera, bad))


def test_virtual_camera_placement_and_errors():
    cam = virtual_camera((0, -5, 0), look_at=(0, 0, 0), fov=30)
    assert np.allclose(cam.frame.forward, (0, 1, 0)) and np.allclose(cam.frame.up, (0, 0, 1))
    assert cam.camera.fov == 30
    down = virtual_camera((0, 0, 10), direction=(0, 0, -1))
    assert np.allclose(down.frame.up, (0, 1, 0))  # a top view has +Y at the top
    for kwargs in (
        {"look_at": (0, 0, 0), "direction": (1, 0, 0)},
        {},
        {"look_at": (0, -5, 0)},
        {"look_at": (0, 0, 0), "fov": 500},
    ):
        with pytest.raises(FeedbackError) as exc:
            virtual_camera((0, -5, 0), **kwargs)
        assert exc.value.code == "INVALID_VIEW"


def test_view_specifications_are_checked_with_pointers():
    with pytest.raises(FeedbackError) as exc:
        check_view_spec({"camera": "Cam", "preset": "top", "zoom": 2})
    codes = {i["code"] for i in exc.value.issues}
    assert {"UNKNOWN_KEY", "INVALID_VALUE"} <= codes
    with pytest.raises(FeedbackError) as exc:
        check_view_spec({"virtual": {"position": [0, 0], "look_at": [0, 0, 0]}})
    assert exc.value.issues[0]["pointer"] == "/virtual/position"
    with pytest.raises(FeedbackError) as exc:
        check_view_spec({"preset": "topp"})
    assert exc.value.hint == "did you mean 'top'?"
    with pytest.raises(FeedbackError):
        check_view_spec({"focus": {"direction": [0, 1, 0]}})  # objects are required
    with pytest.raises(FeedbackError):
        check_view_spec(
            {"virtual": {"position": [0, 0, 1], "look_at": [0, 0, 0], "direction": [0, 0, -1]}}
        )
    assert check_view_spec(None) == {}
    assert check_view_spec({"preset": "iso", "margin": 0.2}) == {"preset": "iso", "margin": 0.2}


def test_presets_look_along_their_axes_and_frame_the_scene():
    scene = _scene()
    top = resolve_view(scene, {"preset": "top"}, 200, 200)
    rc = top.render_camera(scene)
    assert np.allclose(rc.frame.forward, (0, 0, -1)) and np.allclose(rc.frame.up, (0, 1, 0))
    obs = observe(scene, {"preset": "top"}, width=200, height=200)
    ball = obs.record("Ball")
    crate = obs.record("Crate")
    # Seen from above, +Y is up in the image and +X to the right.
    assert ball["center"][1] < crate["center"][1]  # the ball is at y = 0.5, the crate at 0
    assert ball["center"][0] < crate["center"][0]  # the ball is at x = -1.5
    front = resolve_view(scene, {"preset": "front"}, 200, 100).render_camera(scene)
    assert np.allclose(front.frame.forward, (0, 1, 0))
    right = resolve_view(scene, {"preset": "right"}, 200, 100).render_camera(scene)
    assert np.allclose(right.frame.forward, (-1, 0, 0))


def test_a_focus_view_frames_its_objects_exactly():
    scene = _scene()
    view = resolve_view(scene, {"focus": {"objects": "Crate", "margin": 0.1}}, 160, 90)
    obs = observe(scene, {"focus": {"objects": "Crate", "margin": 0.1}}, width=160, height=90)
    geometry = obs.geometry("Crate")
    x0, y0, x1, y1 = geometry["projected_bounds"]
    assert geometry["in_frame_fraction"] == 1.0
    assert x0 >= 0 and y0 >= 0 and x1 <= 160 and y1 <= 90
    # One axis is filled up to the margin: 10 % of the half extent on each side.
    assert min(x0, 160 - x1) <= 160 * 0.05 * 1.01 or min(y0, 90 - y1) <= 90 * 0.05 * 1.01
    assert view.kind == "focus" and view.camera is None


def test_views_by_scene_camera_need_a_camera():
    scene = Scene("Empty")
    scene.add_cube("Box")
    with pytest.raises(FeedbackError) as exc:
        resolve_view(scene, {}, 64, 36)
    assert exc.value.code == "NO_CAMERA"
    with pytest.raises(FeedbackError) as exc:
        resolve_view(scene, {"focus": {"objects": {"tags": ["nothing"]}}}, 64, 36)
    assert exc.value.code == "NOTHING_TO_FRAME"
    with pytest.raises(FeedbackError) as exc:
        resolve_view(scene, {"camera": "Nope"}, 64, 36)
    assert exc.value.code == "OBJECT_NOT_FOUND"


def test_observing_changes_nothing():
    scene = _scene()
    before = scene_to_dict(scene)
    revision, steps = scene.revision, len(scene.history)
    for view in (
        None,
        {"preset": "top"},
        {"focus": {"objects": "Ball"}},
        {"virtual": {"position": [0, -4, 2], "look_at": [0, 0, 0]}},
    ):
        obs = observe(scene, view, width=64, height=36, focus=["Ball", "Crate"], occlusion=True)
        obs.to_dict()
        obs.png("annotated")
    assert scene.revision == revision and len(scene.history) == steps
    assert scene_to_dict(scene) == before
    assert len(scene) == 5  # no camera was added
