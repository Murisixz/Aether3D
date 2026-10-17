"""Cinematics: paths by arc length, path following, camera moves and orbits, shot
sequences with cuts and blends, frame sequences, motion blur, contact sheets, video."""

from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation.cinematics import parse_path
from aether3d.animation.frames import contact_sheet, render_frames, video_command
from aether3d.errors import AnimationError
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import world_matrices


def worlds_at(scene: Scene, t: float) -> dict[str, np.ndarray]:
    return world_matrices(scene, animation_overlay(scene, t))


# --------------------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["catmull_rom", "linear"])
@pytest.mark.parametrize("closed", [False, True])
def test_paths_pass_through_their_points_and_move_at_constant_speed(kind, closed):
    pts = [[0, 0, 0], [2, 0, 0], [3, 2, 1], [1, 4, 0], [-1, 2, 0.5]]
    path = parse_path({"points": pts, "kind": kind, "closed": closed}, "path")
    if kind == "linear":  # straight segments: their lengths exactly
        ring = np.array(pts + ([pts[0]] if closed else []), float)
        assert path.length == pytest.approx(
            float(np.sum(np.linalg.norm(np.diff(ring, axis=0), axis=1))), rel=1e-14
        )
    else:  # a dense polyline of the smooth curve
        dense, _ = path.point(np.linspace(0, 1, 40001))
        polyline = float(np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))
        assert path.length == pytest.approx(polyline, rel=1e-7)
    # Equal offsets cover equal arc lengths (measured on a dense polyline of each piece).
    marks = np.linspace(0, 1, 11)
    pieces = []
    for a, b in zip(marks[:-1], marks[1:], strict=True):
        sub, _ = path.point(np.linspace(a, b, 2001))
        pieces.append(np.sum(np.linalg.norm(np.diff(sub, axis=0), axis=1)))
    # (a polyline of a piece across a corner of a linear path cuts the corner)
    assert np.allclose(pieces, path.length / 10, rtol=1e-6 if kind != "linear" else 2e-3)
    # Through every point (the segment boundaries).
    seg = path.lengths()[1]
    at, _ = path.point(seg / seg[-1])
    expect = np.array(pts + ([pts[0]] if closed else []), float)
    assert np.allclose(at, expect, atol=1e-9)


def test_circle_paths_are_exact():
    path = parse_path({"kind": "circle", "center": [1, 2, 3], "radius": 2.5, "start": 30}, "path")
    f = np.linspace(0, 1, 97)
    p, tan = path.point(f)
    assert np.allclose(np.linalg.norm(p - [1, 2, 3], axis=1), 2.5, atol=1e-12)
    assert np.allclose(p[:, 2], 3.0)
    theta = np.radians(30) + 2 * math.pi * f
    assert np.allclose(
        p[:, :2], np.stack([1 + 2.5 * np.cos(theta), 2 + 2.5 * np.sin(theta)], 1), atol=1e-12
    )
    assert np.allclose(np.sum(tan * (p - [1, 2, 3]), axis=1), 0.0, atol=1e-12)  # tangent
    assert path.length == pytest.approx(2 * math.pi * 2.5)


@pytest.mark.parametrize(
    "raw",
    [
        {"points": [[0, 0, 0]]},
        {"points": [[0, 0, 0], [0, 0, 0]]},
        {"points": [[0, 0, 0], [1, 0, float("nan")]]},
        {"points": [[0, 0, 0], [1, 0, 0]], "kind": "spiral"},
        {"points": [[0, 0, 0], [1, 0, 0]], "kind": "bezier"},
        {"kind": "circle", "center": [0, 0, 0], "radius": -1},
    ],
)
def test_invalid_paths_are_refused(raw):
    with pytest.raises(AnimationError):
        parse_path(raw, "path")


def test_follow_path_moves_objects_along_their_path_with_easing_and_orientation():
    s = Scene("Follow")
    cart = s.add_cube(name="Cart")
    a = s.animation
    pid = a.add_path("Track", [[0, 0, 0], [4, 0, 0], [8, 3, 0], [8, 8, 0]])
    a.follow_path(cart, "Track", start=1.0, end=3.0, easing="sine", ease="in_out")
    path = a.path(pid)
    for t in (0.5, 1.0, 1.7, 2.0, 3.0, 4.0):
        u = min(max((t - 1.0) / 2.0, 0.0), 1.0)
        frac = 0.5 - 0.5 * math.cos(math.pi * u)  # sine in-out
        pos, tan = path.point(frac)
        w = worlds_at(s, t)[cart.id]
        assert np.allclose(w[:3, 3], pos[0], atol=1e-9)
        assert np.allclose(w[:3, 1] / np.linalg.norm(w[:3, 1]), tan[0], atol=1e-9)  # +y along
    assert cart.position == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------------------- #
# Camera moves, orbits, shots
# --------------------------------------------------------------------------------------- #


def camera_scene() -> Scene:
    s = Scene("Shots")
    s.add_plane(name="Ground", size=30)
    s.add_cube(name="Hero", position=(0, 0, 0.5))
    s.add_camera(name="Wide", position=(0, -12, 4), look_at=(0, 0, 0.5))
    s.add_camera(name="Close", position=(4, -4, 1.5), look_at=(0, 0, 0.5))
    s.active_camera = s.get_by_name("Wide")
    return s


def test_dolly_pan_tilt_and_zoom_are_exact_editable_keys():
    s = camera_scene()
    a = s.animation
    cam = s.get_by_name("Wide")
    w0 = worlds_at(s, 0.0)[cam.id]
    a.camera_move(cam, "dolly", 3.0, 0.0, 2.0)
    w1 = worlds_at(s, 2.0)[cam.id]
    forward = -w0[:3, 1] / np.linalg.norm(w0[:3, 1])
    assert np.allclose(w1[:3, 3], w0[:3, 3] + 3.0 * forward, atol=1e-12)
    assert len(a.channel(cam, "transform.position.y")) == 2  # ordinary keys
    a.camera_move(cam, "pan", 30.0, 2.0, 3.0)
    w2 = worlds_at(s, 3.0)[cam.id]
    f1 = -w1[:3, 1] / np.linalg.norm(w1[:3, 1])
    f2 = -w2[:3, 1] / np.linalg.norm(w2[:3, 1])
    yaw = math.degrees(math.atan2(f2[1], f2[0]) - math.atan2(f1[1], f1[0]))
    assert yaw == pytest.approx(30.0, abs=1e-9)
    assert f2[2] == pytest.approx(f1[2], abs=1e-12)  # a pan keeps the pitch
    a.camera_move(cam, "zoom", 2.0, 3.0, 4.0)
    fov0 = cam.camera.fov
    fov1 = a.evaluate(4.0).fields[(cam.id, "camera")]["fov"]
    assert math.tan(math.radians(fov1) / 2) == pytest.approx(math.tan(math.radians(fov0) / 2) / 2)


def test_orbits_keep_distance_and_height_and_look_at_the_target():
    s = camera_scene()
    a = s.animation
    cam = s.get_by_name("Close")
    a.camera_orbit(cam, "Hero", 0.0, 4.0, degrees=180)
    center = np.array([0.0, 0.0])
    r0 = math.hypot(4, -4)
    for t in np.linspace(0, 4, 9):
        w = worlds_at(s, float(t))[cam.id]
        assert math.hypot(*(w[:2, 3] - center)) == pytest.approx(r0, abs=1e-12)
        assert w[2, 3] == pytest.approx(1.5, abs=1e-12)
        forward = -w[:3, 1] / np.linalg.norm(w[:3, 1])
        to = np.array([0, 0, 0.5]) - w[:3, 3]
        assert np.allclose(forward, to / np.linalg.norm(to), atol=1e-9)
    end = worlds_at(s, 4.0)[cam.id][:3, 3]
    assert np.allclose(end[:2], [-4, 4], atol=1e-9)  # half way round


def test_shots_cut_and_blend_between_cameras():
    s = camera_scene()
    a = s.animation
    a.set_shots(
        [
            {"camera": "Wide", "start": 0, "end": 2, "name": "wide"},
            {
                "camera": "Close",
                "start": 1.5,
                "end": 4,
                "name": "close",
                "transition": {"type": "blend", "duration": 1.0},
            },
            {"camera": "Wide", "start": 3.0, "end": 4, "name": "back"},
        ]
    )
    wide, close = s.get_by_name("Wide"), s.get_by_name("Close")
    assert animation_overlay(s, 1.0).camera == wide.id
    assert animation_overlay(s, 3.5).camera == wide.id  # a cut back
    blend = animation_overlay(s, 2.0).camera  # half way through the blend (smoothstep 0.5)
    assert blend.object_id is None
    expect = 0.5 * (np.array(wide.world_position) + np.array(close.world_position))
    assert np.allclose(blend.frame.position, expect, atol=1e-12)
    assert animation_overlay(s, 2.6).camera == close.id
    result = s.render(time=1.0, width=48, height=32)
    assert result.camera["name"] == "Wide"
    # Persisted, and a camera that is gone leaves its shot without a camera.
    loaded = Scene.from_dict(json.loads(json.dumps(s.to_dict())))
    assert loaded.animation.shots() == a.shots()


# --------------------------------------------------------------------------------------- #
# Frame sequences
# --------------------------------------------------------------------------------------- #


def animated_scene() -> Scene:
    s = camera_scene()
    a = s.animation
    a.set_timeline(fps="24000/1001", start=0.0, end=2.0)
    hero = s.get_by_name("Hero")
    a.keyframe(hero, "transform.position.x", 0.0, -2.0, interpolation="linear")
    a.keyframe(hero, "transform.position.x", 2.0, 2.0, interpolation="linear")
    return s


def test_render_frames_writes_the_frames_of_the_timeline_at_exact_times(tmp_path):
    s = animated_scene()
    rep = render_frames(s, tmp_path / "out", start=0.0, end=1.0, step=4, width=40, height=30)
    frames = [f["frame"] for f in rep["frames"]]
    assert frames == list(range(0, 24, 4))  # 23.976 fps: frame 24 is after 1 s
    fps = s.animation.fps
    for f in rep["frames"]:
        assert f["time"] == fps.frame_time(f["frame"])
        assert (tmp_path / "out" / f"frame_{f['frame']:05d}.png").exists()
    single = s.render(time=fps.frame_time(8), width=40, height=30).image
    from aether3d.materials.images import decode_png

    written = decode_png((tmp_path / "out" / "frame_00008.png").read_bytes()).pixels[..., :3]
    assert np.array_equal(written, single)
    assert s.get_by_name("Hero").position == (0.0, 0.0, 0.5)


def test_motion_blur_is_the_average_of_sub_frame_renders(tmp_path):
    from aether3d.rendering.tonemap import display

    s = animated_scene()
    rep = render_frames(
        s, tmp_path, frames=[12], width=40, height=30, motion_blur={"shutter": 0.5, "samples": 4}
    )
    fps = s.animation.fps
    t = fps.frame_time(12)
    dt = fps.frame_duration * 0.5
    acc = None
    for j in range(4):
        lin = s.render(time=t + ((j + 0.5) / 4 - 0.5) * dt, width=40, height=30).linear.astype(
            np.float64
        )
        acc = lin if acc is None else acc + lin
    expect = display((acc / 4).astype(np.float32))
    from aether3d.materials.images import decode_png

    got = decode_png((tmp_path / "frame_00012.png").read_bytes()).pixels[..., :3]
    assert np.array_equal(got, expect)
    assert rep["motion_blur"] == {"shutter": 0.5, "samples": 4}


def test_frame_rendering_stops_on_request_and_checks_its_arguments(tmp_path):
    s = animated_scene()
    calls = []
    rep = render_frames(
        s,
        tmp_path,
        start=0,
        end=1,
        width=24,
        height=16,
        progress=lambda done, total, sec: calls.append(done) or done < 3,
    )
    assert rep["stopped"] and len(rep["frames"]) == 3
    rep = render_frames(s, tmp_path, start=0, end=1, width=24, height=16, should_stop=lambda: True)
    assert rep["stopped"] and rep["frames"] == []
    for bad in ("../x_{frame}.png", "frame.png", "x_{frame}.jpg", "{other}_{frame}.png"):
        with pytest.raises(AnimationError):
            render_frames(s, tmp_path, frames=[0], pattern=bad, width=24, height=16)
    with pytest.raises(AnimationError):
        render_frames(s, tmp_path, frames=[0], motion_blur={"shutter": 2.0}, width=24, height=16)


def test_contact_sheets_and_video_commands(tmp_path):
    images = [np.full((20, 30, 3), v, np.uint8) for v in (40, 90, 140, 190, 240)]
    sheet = contact_sheet(images, columns=2, labels=["a1", "b2", "c3", "d4", "e5"], gap=4)
    assert sheet.shape == (3 * (20 + 22) + 4 * 4, 2 * 30 + 3 * 4, 3)
    label_band = sheet[4 + 20 : 4 + 20 + 22, 4 : 4 + 30]
    assert label_band.max() > 200  # the label was drawn
    cmd = video_command(tmp_path, "frame_{frame:05d}.png", "30000/1001", tmp_path / "a.mp4")
    assert cmd[:4] == ["ffmpeg", "-y", "-framerate", "30000/1001"] and cmd[5].endswith(
        "frame_%05d.png"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_assembled_videos_are_written_when_ffmpeg_exists(tmp_path):
    from aether3d.animation.frames import assemble_video

    s = animated_scene()
    render_frames(s, tmp_path, start=0, end=0.5, width=32, height=24)
    out = assemble_video(
        tmp_path, "frame_{frame:05d}.png", s.animation.fps.to_json(), tmp_path / "clip.mp4"
    )
    assert out["bytes"] > 0 and (tmp_path / "clip.mp4").exists()
