"""GPU frames that reuse and update what earlier frames computed.

A retained render world gives its render scenes a shared cache; the GPU backend keeps there
what it computes from the instances, meshes and materials (draw order, instance records,
material tables, the float32 range check) and, when only some instances' transforms or
colors changed, updates it: their records again, every record's camera-relative translation
when the camera moved. ``VERIFY_KEPT`` (on in every test) recomputes everything each frame
and requires the same bits; these tests also compare each frame with one rendered without
the retained world, after every kind of change.
"""

from __future__ import annotations

import pytest
from parity import assert_bitwise

from aether3d import Scene, create_cube, create_sphere

OUTPUTS = ("image", "linear", "depth", "object_index", "coverage")


def scene_for_frames() -> Scene:
    s = Scene("Frames")
    cube, ball = create_cube(0.6), create_sphere(0.35, 10, 5)
    for k in range(24):
        s.add_object(
            f"o{k}", mesh=cube if k % 3 else ball, position=(k % 6 - 2.5, k // 6 - 1.5, 0.4)
        )
    s.create_material("Red", base_color="#cc3322")
    s.assign_material("o4", "Red")
    s.add_plane(name="Floor", size=12)
    s.add_directional_light(name="Sun", direction=(-0.3, 0.5, -1.0))
    s.add_camera(name="Cam", position=(0, -9, 6), look_at=(0, 0, 0))
    return s


def same_frame(gpu, s: Scene, monkeypatch, **options) -> None:
    kept = gpu.render(s, width=64, height=48, **options)
    with monkeypatch.context() as m:
        m.setenv("AETHER3D_RETAINED_RENDER_WORLD", "0")
        fresh = gpu.render(s, width=64, height=48, **options)
    for name in OUTPUTS:
        assert_bitwise(getattr(kept, name), getattr(fresh, name), name)
    assert kept.objects == fresh.objects
    assert [d.to_dict() for d in kept.diagnostics] == [d.to_dict() for d in fresh.diagnostics]
    for key in ("pixels_covered", "samples_covered", "draw_calls", "triangles", "instances"):
        assert kept.stats.get(key) == fresh.stats.get(key), key


def edits(s: Scene) -> list:
    o = [s.get_by_name(f"o{k}") for k in range(24)]
    cam = s.get_by_name("Cam")

    def move(obj, dz):
        x, y, z = obj.position
        obj.position = (x, y, z + dz)

    return [
        ("unchanged", lambda: None),
        ("one moved", lambda: move(o[5], 0.3)),
        ("camera moved", lambda: setattr(cam, "position", (0.5, -9.0, 6.0))),
        ("both", lambda: (move(o[7], -0.2), setattr(cam, "position", (-0.5, -8.5, 6.2)))),
        ("recolored", lambda: setattr(o[2], "color", (0.1, 0.8, 0.2))),
        ("highlight", lambda: setattr(o[3].appearance, "specular", 0.7)),
        ("mirrored", lambda: setattr(o[8], "scale", (-1.0, 1.0, 1.0))),
        ("unmirrored", lambda: setattr(o[8], "scale", (1.0, 1.0, 1.0))),
        ("flattened", lambda: setattr(o[9], "scale", (1.0, 0.0, 1.0))),
        ("restored", lambda: setattr(o[9], "scale", (1.0, 1.0, 1.0))),
        ("renamed", lambda: setattr(o[10], "name", "renamed")),
        ("hidden", lambda: setattr(o[11], "visible", False)),
        ("material", lambda: s.assign_material(o[12], "Red")),
        ("many moved", lambda: [move(x, 0.05) for x in o[:12]]),
        ("undo", lambda: s.history.undo()),
        ("added", lambda: s.add_cube(name="late", position=(3.0, 3.0, 0.5))),
    ]


@pytest.mark.parametrize("cull", ["back", "none"])
def test_every_frame_equals_one_computed_from_scratch(gpu, monkeypatch, cull):
    s = scene_for_frames()
    same_frame(gpu, s, monkeypatch, cull=cull)
    for _name, edit in edits(s):
        edit()
        same_frame(gpu, s, monkeypatch, cull=cull)
        same_frame(gpu, s, monkeypatch, cull=cull)  # and the frame after, unchanged


def test_settings_that_change_between_frames(gpu, monkeypatch):
    s = scene_for_frames()
    same_frame(gpu, s, monkeypatch, cull="back")
    same_frame(gpu, s, monkeypatch, cull="none")
    same_frame(gpu, s, monkeypatch, shadows=True)
    s.get_by_name("o1").position = (0.0, 0.0, 1.5)
    same_frame(gpu, s, monkeypatch, shadows=True, outputs=("image",))
    same_frame(gpu, s, monkeypatch, cull="back", tonemap="neutral")


def test_objects_beyond_float32_range_of_a_moving_camera(gpu, monkeypatch):
    s = scene_for_frames()
    far = s.add_cube(name="Far", position=(4e38, 0.0, 0.0))
    same_frame(gpu, s, monkeypatch)  # NONFINITE_GEOMETRY: the far cube is skipped
    s.get_by_name("Cam").position = (1.0, -9.0, 6.0)
    same_frame(gpu, s, monkeypatch)
    far.position = (5.0, 0.0, 0.0)
    same_frame(gpu, s, monkeypatch)
    s.get_by_name("o0").position = (4e38, 1.0, 0.0)
    same_frame(gpu, s, monkeypatch)


def test_an_unchanged_frame_uploads_nothing_and_a_move_one_record(gpu):
    s = scene_for_frames()
    gpu.render(s, width=64, height=48)
    res = gpu.render(s, width=64, height=48).backend["resources"]
    assert res["frame_buffer_writes"] == 0
    s.get_by_name("o5").position = (0.0, 0.0, 2.0)
    res = gpu.render(s, width=64, height=48).backend["resources"]
    assert 0 < res["frame_buffer_bytes_written"] <= 128


def test_moves_update_the_kept_records_instead_of_packing_them_all(gpu, monkeypatch):
    from aether3d.rendering.gpu import renderer

    packed: list[int] = []
    original = renderer.pack_instances

    def counting(rs, order, eye):
        packed.append(len(order))
        return original(rs, order, eye)

    s = scene_for_frames()
    gpu.render(s, width=64, height=48)
    monkeypatch.setattr(renderer, "VERIFY_KEPT", False)  # (it packs everything to compare)
    monkeypatch.setattr(renderer, "pack_instances", counting)
    gpu.render(s, width=64, height=48)
    assert packed == []  # unchanged: everything kept
    s.get_by_name("o5").position = (0.0, 0.0, 2.0)
    gpu.render(s, width=64, height=48)
    assert packed == [1]  # one record packed again
    s.get_by_name("Cam").position = (0.3, -9.0, 6.0)
    gpu.render(s, width=64, height=48)
    assert packed == [1]  # translations updated in place
    s.get_by_name("o6").position = (1.0, 0.0, 2.0)
    gpu.render(s, width=64, height=48)
    assert packed == [1, 1]
    s.get_by_name("o6").color = (0.9, 0.9, 0.1)  # a color of its own: a material change
    gpu.render(s, width=64, height=48)
    assert packed[-1] == 25  # (instance materials changed: packed again)
    monkeypatch.setattr(renderer, "VERIFY_KEPT", True)
    s.get_by_name("o7").position = (1.0, 1.0, 1.0)
    s.get_by_name("Cam").position = (0.0, -9.5, 6.0)
    gpu.render(s, width=64, height=48)  # (checked against a full computation)
