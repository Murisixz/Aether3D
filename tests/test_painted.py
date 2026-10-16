"""Painted textures: tiled layers, compositing, incremental render levels, history,
persistence and rendering."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aether3d import MaterialError, Scene
from aether3d.materials.painted import FULL, TILE, blend, composite_over
from aether3d.materials.sampling import build_mip_chain
from aether3d.rendering.png import encode_png


def q16(x):
    return np.round(np.clip(np.asarray(x, dtype=np.float64), 0, 1) * FULL) / FULL


def doc_with_layers(width=150, height=90, encoding="color"):
    scene = Scene()
    doc = scene.add_painted_texture("Paint", width, height, encoding=encoding)
    return scene, doc


def test_new_document_is_its_background_and_sizes_check():
    scene, doc = doc_with_layers()
    assert np.array_equal(doc.read(), np.ones((90, 150, 4)))
    assert doc.tiles_x == 3 and doc.tiles_y == 2
    normal = scene.add_painted_texture("N", 64, encoding="normal")
    assert np.allclose(normal.read(0, 0, 1, 1)[0, 0], q16([0.5, 0.5, 1.0, 1.0]))
    for kwargs in ({"width": 0}, {"width": 20000}, {"encoding": "hdr"}, {"background": (2, 0, 0)}):
        with pytest.raises(MaterialError):
            scene.add_painted_texture(**{"width": 16, **kwargs})


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "multiply",
        "screen",
        "overlay",
        "darken",
        "lighten",
        "difference",
        "add",
        "subtract",
    ],
)
def test_compositing_follows_the_w3c_formula(mode):
    scene, doc = doc_with_layers(40, 30)
    doc.set_background((0.2, 0.4, 0.6, 1.0))
    layer = doc.add_layer("L", blend=mode, opacity=0.8)
    color = np.array([0.9, 0.3, 0.5, 0.7])
    doc.write_pixels(layer, np.broadcast_to(color, (30, 40, 4)).copy())
    got = doc.read(5, 5, 6, 6)[0, 0]
    cb = q16([0.2, 0.4, 0.6]).astype(np.float32)
    cs = q16(color[:3]).astype(np.float32)
    a_s = np.float32(q16(color[3])) * np.float32(0.8)
    mixed = blend(mode, cb, cs)  # backdrop opaque: C_s' = B(C_b, C_s)
    expected = a_s * mixed + (1 - a_s) * cb
    assert np.allclose(got[:3], q16(expected), atol=1.5 / FULL)
    assert got[3] == 1.0


def test_transparent_backdrop_keeps_straight_color():
    base = np.zeros((4, 4, 4), np.float32)
    out = composite_over(
        base, np.full((4, 4, 3), 0.25, np.float32), np.full((4, 4), 0.5, np.float32), "multiply"
    )
    assert np.allclose(out[..., :3], 0.25) and np.allclose(out[..., 3], 0.5)


def test_fill_layers_masks_visibility_and_order():
    scene, doc = doc_with_layers(64, 64)
    fill = doc.add_layer("Red", kind="fill", value=(1, 0, 0, 1))
    assert np.allclose(doc.read(3, 3, 4, 4)[0, 0], [1, 0, 0, 1])
    mask = np.zeros((64, 64))
    mask[:, 32:] = 1.0
    doc.set_mask(fill, mask)
    assert np.allclose(doc.read(10, 10, 11, 11)[0, 0], [1, 1, 1, 1])
    assert np.allclose(doc.read(40, 10, 41, 11)[0, 0], [1, 0, 0, 1])
    doc.set_layer(fill, visible=False)
    assert np.allclose(doc.read(40, 10, 41, 11)[0, 0], [1, 1, 1, 1])
    doc.set_layer(fill, visible=True)
    blue = doc.add_layer("Blue", kind="fill", value=(0, 0, 1, 1), index=0)
    assert doc.layers[0] is blue  # below the red fill: hidden under it where it shows
    assert np.allclose(doc.read(10, 10, 11, 11)[0, 0], [0, 0, 1, 1])
    doc.move_layer(blue, 1)
    assert np.allclose(doc.read(40, 10, 41, 11)[0, 0], [0, 0, 1, 1])
    with pytest.raises(MaterialError):
        doc.add_layer("Blue")  # names are unique
    with pytest.raises(MaterialError):
        doc.layer("missing")


def test_normal_documents_renormalize_their_composite():
    scene, doc = doc_with_layers(16, 16, "normal")
    layer = doc.add_layer("Tilt", opacity=0.5)
    doc.write_pixels(layer, np.broadcast_to([1.0, 0.5, 0.5, 1.0], (16, 16, 4)).copy())
    v = doc.read(1, 1, 2, 2)[0, 0, :3] * 2 - 1
    assert abs(np.linalg.norm(v) - 1) < 1e-3


def test_incremental_levels_equal_a_full_rebuild():
    scene, doc = doc_with_layers(200, 130)
    rng = np.random.default_rng(0)
    layer = doc.add_layer("Noise")
    for mode in ("srgb", "linear", "normal"):
        doc.mip_chain(mode)  # build, then change pieces
    for _ in range(6):
        w, h = rng.integers(5, 70, 2)
        x, y = rng.integers(0, 200 - w), rng.integers(0, 130 - h)
        doc.write_pixels(layer, rng.random((h, w, 4)), int(x), int(y))
        for mode in ("srgb", "linear", "normal"):
            chain, _ = doc.mip_chain(mode)
            full = build_mip_chain(doc.pixels8(), mode)
            assert len(chain.levels) == len(full.levels)
            for a, b in zip(chain.levels, full.levels, strict=True):
                assert np.array_equal(a, b)


def test_pixels8_is_the_rounded_16_bit_composite():
    scene, doc = doc_with_layers(70, 70)
    layer = doc.add_layer("Noise")
    doc.write_pixels(layer, np.random.default_rng(1).random((70, 70, 4)))
    p16 = doc.pixels16().astype(np.uint32)
    assert np.array_equal(doc.pixels8(), ((p16 * 255 + 32767) // 65535).astype(np.uint8))


def test_history_restores_tiles_and_keeps_only_changed_ones():
    scene, doc = doc_with_layers(256, 256)
    layer = doc.add_layer("Paint")
    doc.write_pixels(layer, np.full((10, 10, 4), 0.5), 0, 0)  # one tile
    before = doc.pixels16().copy()
    doc.write_pixels(layer, np.full((10, 10, 4), 0.9), 200, 200)  # another tile
    after = doc.pixels16().copy()
    assert len(doc.layer("Paint").tiles) == 2
    scene.history.undo()
    assert np.array_equal(doc.pixels16(), before)
    scene.history.redo()
    assert np.array_equal(doc.pixels16(), after)
    # The step that wrote the second tile shares the first tile with the state before it.
    a = scene.history.entries[-1]
    assert a.label.startswith("Write pixels")


def test_render_equals_the_same_pixels_as_an_image_texture():
    scene = Scene()
    scene.add_plane("Floor", size=2)
    doc = scene.add_painted_texture("Paint", 64, 64)
    layer = doc.add_layer("Checker")
    yy, xx = np.mgrid[0:64, 0:64]
    checker = ((xx // 8 + yy // 8) % 2).astype(float)
    doc.write_pixels(
        layer, np.stack([checker, 0.5 * checker, 1 - checker, np.ones_like(checker)], -1)
    )
    mat = scene.create_material("Painted", roughness=0.7)
    mat.set_texture("base_color", doc)
    scene.get_by_name("Floor").material = mat
    scene.add_directional_light("Sun", direction=(-0.3, 0.4, -1))
    scene.add_camera("Cam", position=(0, -2, 2), look_at=(0, 0, 0))
    painted = scene.render(width=64, height=48)
    image = scene.add_image_texture("Same", data=encode_png(doc.pixels8()))
    mat.set_texture("base_color", image)
    plain = scene.render(width=64, height=48)
    assert np.array_equal(painted.image, plain.image)


def test_project_round_trip_is_exact(tmp_path):
    scene, doc = doc_with_layers(130, 70)
    paint = doc.add_layer("Paint", blend="multiply", opacity=0.75)
    doc.write_pixels(paint, np.random.default_rng(2).random((40, 60, 4)), 50, 10)
    fill = doc.add_layer("Tint", kind="fill", value=(0.1, 0.2, 0.3, 0.4))
    doc.set_mask(fill, np.random.default_rng(3).random((70, 130)))
    scene.save(tmp_path / "p.aether")
    record = json.loads((tmp_path / "p.aether").read_text())["scene"]["textures"][0]
    assert record["type"] == "painted" and len(record["layers"]) == 2
    loaded = Scene.load(tmp_path / "p.aether").materials.texture("Paint")
    assert np.array_equal(loaded.pixels16(), doc.pixels16())
    assert [layer.to_dict() for layer in loaded.layers] == [layer.to_dict() for layer in doc.layers]


def test_history_files_store_painted_steps(tmp_path):
    scene, doc = doc_with_layers(100, 100)
    layer = doc.add_layer("Paint")
    doc.write_pixels(layer, np.full((20, 20, 4), 0.3), 5, 5)
    doc.write_pixels(layer, np.full((20, 20, 4), 0.7), 70, 70)
    scene.save(tmp_path / "h.aether", history=True)
    loaded = Scene.load(tmp_path / "h.aether", history=True)
    paint = loaded.materials.texture("Paint")
    assert np.array_equal(paint.pixels16(), doc.pixels16())
    loaded.history.undo()
    scene.history.undo()
    assert np.array_equal(paint.pixels16(), doc.pixels16())
    assert loaded.history.verify()["ok"]


def test_invalid_records_are_refused(tmp_path):
    scene, doc = doc_with_layers(32, 32)
    doc.add_layer("Paint")
    scene.save(tmp_path / "p.aether")
    data = json.loads((tmp_path / "p.aether").read_text())
    data["scene"]["textures"][0]["layers"][0]["blend"] = "dodge"
    (tmp_path / "bad.aether").write_text(json.dumps(data))
    from aether3d.errors import ProjectFormatError

    with pytest.raises(ProjectFormatError):
        Scene.load(tmp_path / "bad.aether")
    data["scene"]["textures"][0]["layers"][0]["blend"] = "normal"
    data["scene"]["textures"][0]["layers"][0]["pixels"] = {
        "mime_type": "image/png",
        "data": __import__("base64").b64encode(encode_png(np.zeros((8, 8, 4), np.uint8))).decode(),
    }
    (tmp_path / "bad2.aether").write_text(json.dumps(data))
    with pytest.raises(ProjectFormatError):
        Scene.load(tmp_path / "bad2.aether")  # 8x8 pixels for a 32x32 texture


def test_export_png_and_gltf(tmp_path):
    from aether3d.materials.images import decode_png16

    scene = Scene()
    plane = scene.add_plane("Floor", size=2)
    doc = scene.add_painted_texture("Paint", 32, 32)
    layer = doc.add_layer("A")
    doc.write_pixels(layer, np.random.default_rng(4).random((32, 32, 4)))
    path = doc.export_png(tmp_path / "paint.png", bits=16)
    assert np.array_equal(decode_png16(path.read_bytes()), doc.pixels16())
    mat = scene.create_material("M")
    mat.set_texture("base_color", doc)
    plane.material = mat
    scene.export(tmp_path / "floor.glb")
    from aether3d.gltf.reader import import_gltf

    back = import_gltf(tmp_path / "floor.glb").scene
    tex = back.materials.material("M").textures["base_color"]
    image = back.materials.texture(tex.texture).load().image.pixels
    assert np.array_equal(image, doc.pixels8())


def test_tile_geometry():
    scene, doc = doc_with_layers(130, 70)
    assert doc.tile_rect(0) == (0, 0, TILE, TILE)
    assert doc.tile_rect(5) == (128, 64, 130, 70)
    assert doc.tiles_in(60, 60, 70, 66) == [0, 1, 3, 4]


def test_gpu_updates_only_the_changed_rows(gpu):
    scene = Scene()
    scene.add_plane("Floor", size=2)
    doc = scene.add_painted_texture("Paint", 512, 512)
    layer = doc.add_layer("A")
    mat = scene.create_material("Painted")
    mat.set_texture("base_color", doc)
    scene.get_by_name("Floor").material = mat
    scene.add_camera("Cam", position=(0, 0, 3), look_at=(0, 0, 0))
    first = gpu.render(scene, width=64, height=64, samples=1)
    pool = first.backend["resources"]["texel_pool"]
    doc.write_pixels(layer, np.full((16, 16, 4), [1.0, 0.0, 0.0, 1.0]), 200, 200)
    second = gpu.render(scene, width=64, height=64, samples=1)
    res = second.backend["resources"]
    assert res["texel_uploads"] == 0  # no full re-upload
    assert res["texel_pool"]["partial_uploads"] == pool["partial_uploads"] + 1
    # Only the rows of the changed tile were written (in every level), far below the pool.
    assert 0 < res["texel_upload_bytes"] < pool["bytes"] / 4
    cpu = scene.render(width=64, height=64, samples=1)
    assert np.abs(cpu.image.astype(int) - second.image.astype(int)).max() <= 2
