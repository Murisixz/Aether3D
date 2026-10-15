"""Caches kept between renders give exactly what rendering without them gives.

Each test extracts the same scene with the caches on and with them switched off
(``AETHER3D_MATERIAL_CACHE=0``) and compares the render scenes byte for byte, after the
kinds of change a cache could miss: image files changed on disk, failing textures (their
diagnostics), undo and redo, colors that differ only in the sign of a zero.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from parity import assert_render_scenes_equal

from aether3d import Scene
from aether3d.rendering.png import write_png
from aether3d.rendering.scene_data import extract_render_scene


def save_png(path, value: int, size: int = 4):
    arr = np.full((size, size, 4), 255, np.uint8)
    arr[..., :3] = value
    write_png(path, arr)
    return path


def extract(scene: Scene, camera):
    return extract_render_scene(scene, camera, aspect=1.0)


def uncached(scene: Scene, camera, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("AETHER3D_MATERIAL_CACHE", "0")
        return extract(scene, camera)


def textured_scene(tmp_path):
    scene = Scene("Textured")
    plane = scene.add_plane("P", size=2)
    ball = scene.add_sphere("B", radius=0.5, position=(0, 0, 1))
    m = scene.create_material("M", base_color="#ffffff")
    m.set_texture("base_color", save_png(tmp_path / "t.png", 200))
    plane.material = m
    ball.material = m
    camera = scene.add_camera("Cam", position=(0, -5, 3), look_at=(0, 0, 0))
    return scene, camera


def digests(rs):
    return [t.image.digest for m in rs.materials for t in m.textures.values() if t.image]


def test_an_image_changed_on_disk_is_used_by_the_next_render(tmp_path, monkeypatch):
    scene, camera = textured_scene(tmp_path)
    first = extract(scene, camera)
    assert_render_scenes_equal(first, uncached(scene, camera, monkeypatch))
    before = digests(first)
    save_png(tmp_path / "t.png", 30, size=8)  # another size: a new file signature
    stamp = os.stat(tmp_path / "t.png").st_mtime_ns + 10**9
    os.utime(tmp_path / "t.png", ns=(stamp, stamp))
    second = extract(scene, camera)
    assert digests(second) != before
    assert_render_scenes_equal(second, uncached(scene, camera, monkeypatch))


def test_failing_textures_are_reported_in_every_render(tmp_path, monkeypatch):
    scene, camera = textured_scene(tmp_path)
    (tmp_path / "t.png").unlink()
    first = extract(scene, camera)
    codes = [d.code for d in first.diagnostics]
    assert any(code.startswith("TEXTURE") for code in codes), codes
    second = extract(scene, camera)  # from the cache: the same diagnostics again
    assert [d.code for d in second.diagnostics] == codes
    assert_render_scenes_equal(second, uncached(scene, camera, monkeypatch))
    save_png(tmp_path / "t.png", 90)  # the file is back: used again, no diagnostic
    third = extract(scene, camera)
    assert not any(d.code.startswith("TEXTURE") for d in third.diagnostics)
    assert_render_scenes_equal(third, uncached(scene, camera, monkeypatch))


def test_material_edits_undo_and_redo_are_never_served_stale(tmp_path, monkeypatch):
    scene, camera = textured_scene(tmp_path)
    m = scene.materials["M"]
    states = []
    for value in (0.1, 0.7):
        extract(scene, camera)
        m.roughness = value
        states.append(extract(scene, camera))
    scene.history.undo()
    assert_render_scenes_equal(extract(scene, camera), states[0])
    scene.history.redo()
    assert_render_scenes_equal(extract(scene, camera), states[1])
    scene.materials.remove_material("M", force=True)
    assert_render_scenes_equal(extract(scene, camera), uncached(scene, camera, monkeypatch))


@pytest.mark.parametrize("value", [0.0, -0.0])
def test_colors_keep_their_own_materials(value, monkeypatch):
    scene = Scene("Colors")
    a = scene.add_cube("A", position=(-1, 0, 0))
    b = scene.add_cube("B", position=(1, 0, 0))
    a.color = (0.0, 0.5, 0.5)
    b.color = (value, 0.5, 0.5)
    camera = scene.add_camera("Cam", position=(0, -6, 2), look_at=(0, 0, 0))
    for _ in range(2):
        assert_render_scenes_equal(extract(scene, camera), uncached(scene, camera, monkeypatch))
