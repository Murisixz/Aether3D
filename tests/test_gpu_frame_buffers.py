"""Per-frame GPU data kept between renders: identical images, only changes written."""

from __future__ import annotations

import numpy as np
from parity import assert_bitwise

from aether3d import Scene, create_cube, create_sphere


def scene_with_variety() -> Scene:
    s = Scene("Frames")
    rng = np.random.default_rng(3)
    cube, ball = create_cube(0.8), create_sphere(0.4, 12, 6)
    stone = s.create_material("Stone", base_color="#998877", roughness=0.6)
    for k in range(120):
        obj = s.add_object(
            f"o{k}",
            mesh=cube if k % 2 else ball,
            position=(float(k % 12) - 6, float(k // 12) - 5, 0.4),
            scale=(1.0, -1.0 if k % 9 == 0 else 1.0, 1.0),
        )
        if k % 4 == 0:
            obj.material = stone
        else:
            obj.color = tuple(rng.random(3).tolist())
    s.add_directional_light("Sun", direction=(-0.3, 0.4, -1.0))
    s.add_camera("Cam", position=(0, -14, 9), look_at=(0, 0, 0))
    return s


def outputs(result):
    return (result.image, result.linear, result.depth, result.coverage, result.object_index)


def edits(s: Scene):
    objs = [o for o in s if o.mesh is not None]
    yield lambda: None
    yield lambda: setattr(objs[5], "position", (0.0, 0.0, 2.0))
    yield lambda: setattr(s.materials["Stone"], "roughness", 0.2)
    yield lambda: s.remove(objs[7])
    yield lambda: s.add_object("New", mesh=create_cube(0.5), position=(1.0, 1.0, 1.0))
    yield lambda: setattr(s.get_by_name("Cam"), "position", (1.0, -14.0, 9.0))
    yield lambda: s.history.undo(2)


def test_kept_buffers_render_exactly_what_fresh_ones_render(gpu, monkeypatch):
    s = scene_with_variety()
    for edit in edits(s):
        edit()
        kept = gpu.render(s, width=96, height=64, shadows=True)
        assert kept.backend["resources"]["frame_buffers_kept"]
        with monkeypatch.context() as m:
            m.setenv("AETHER3D_GPU_PERSISTENT_BUFFERS", "0")
            fresh = gpu.render(s, width=96, height=64, shadows=True)
        assert not fresh.backend["resources"]["frame_buffers_kept"]
        for a, b in zip(outputs(kept), outputs(fresh), strict=True):
            assert_bitwise(a, b)


def test_an_unchanged_frame_writes_nothing_and_one_move_writes_little(gpu):
    s = scene_with_variety()
    gpu.render(s, width=64, height=48)
    same = gpu.render(s, width=64, height=48).backend["resources"]
    assert same["frame_buffer_writes"] == 0
    obj = next(o for o in s if o.mesh is not None)
    x, y, z = obj.position
    obj.position = (x, y, z + 0.25)
    moved = gpu.render(s, width=64, height=48).backend["resources"]
    assert 0 < moved["frame_buffer_bytes_written"] <= 512  # one 128-byte instance record
