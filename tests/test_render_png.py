"""The PNG encoder, checked against an independent decoder (Pillow) and the specification."""

import io
import struct
import zlib

import numpy as np
import pytest

from aether3d import RenderError
from aether3d.rendering.png import SIGNATURE, encode_png, write_png

PIL = pytest.importorskip("PIL.Image")


def chunks(data: bytes):
    assert data[:8] == SIGNATURE
    pos, out = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        out.append((kind, body))
        pos += 12 + length
    return out


@pytest.mark.parametrize(
    "shape", [(1, 1, 3), (7, 1, 3), (1, 9, 3), (37, 53, 3), (64, 33, 4), (20, 30), (300, 257, 3)]
)
@pytest.mark.parametrize("content", ["noise", "gradient", "flat"])
def test_pillow_decodes_exactly(shape, content):
    rng = np.random.default_rng(sum(shape))
    if content == "noise":
        image = rng.integers(0, 256, shape, dtype=np.uint8)
    elif content == "gradient":
        yy, xx = np.indices(shape[:2])
        g = (xx * 7 + yy * 3) % 256
        layers = [g, (g * 3) % 256, (g + 50) % 256, g // 2]
        image = (np.stack(layers[: shape[2]], -1) if len(shape) == 3 else g).astype(np.uint8)
    else:
        image = np.full(shape, 77, np.uint8)
    data = encode_png(image)
    PIL.open(io.BytesIO(data)).verify()
    decoded = np.asarray(PIL.open(io.BytesIO(data)))
    assert decoded.shape == image.shape and np.array_equal(decoded, image)


def test_chunk_structure_and_metadata():
    data = encode_png(np.zeros((4, 5, 3), np.uint8), text={"Software": "Aether3D test"})
    kinds = [k for k, _ in chunks(data)]
    assert kinds == [b"IHDR", b"sRGB", b"tEXt", b"IDAT", b"IEND"]
    ihdr = chunks(data)[0][1]
    assert struct.unpack(">IIBBBBB", ihdr) == (5, 4, 8, 2, 0, 0, 0)
    assert PIL.open(io.BytesIO(data)).info.get("Software") == "Aether3D test"
    gray = [k for k, _ in chunks(encode_png(np.zeros((2, 2), np.uint8)))]
    assert b"sRGB" not in gray  # only color images declare sRGB


def test_output_is_deterministic():
    image = np.random.default_rng(1).integers(0, 256, (50, 60, 3), dtype=np.uint8)
    assert encode_png(image) == encode_png(image.copy())


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4, 3), np.float32),
        np.zeros((4, 4, 2), np.uint8),
        np.zeros((0, 4, 3), np.uint8),
        np.zeros((4,), np.uint8),
    ],
)
def test_invalid_images_are_rejected(bad):
    with pytest.raises(RenderError) as info:
        encode_png(bad)
    assert info.value.code == "INVALID_IMAGE"


def test_writes_are_atomic(tmp_path, monkeypatch):
    target = tmp_path / "image.png"
    target.write_bytes(b"previous")
    import os

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        write_png(target, np.zeros((2, 2, 3), np.uint8))
    assert target.read_bytes() == b"previous"  # untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["image.png"]  # no temporary left


def test_rendered_images_decode_with_the_reported_size(tmp_path):
    from aether3d import Scene

    scene = Scene()
    scene.add_cube("Cube").color = "#2266aa"
    scene.add_camera("Cam", position=(3, -4, 2), look_at=(0, 0, 0))
    result = scene.render(width=77, height=41)
    path = result.save(tmp_path / "cube.png")
    decoded = np.asarray(PIL.open(path))
    assert decoded.shape == (41, 77, 3) and np.array_equal(decoded, result.image)
