"""Image decoding for textures: every PNG variant, JPEG via Pillow, and hostile input.

PNGs are written by ``image_helpers.make_png`` (an independent encoder that can
produce every color type, bit depth, filter and interlacing, and corrupt files on
purpose) and, for cross-checks, by Pillow.
"""

import io
import struct
import sys
import zlib

import numpy as np
import pytest
from image_helpers import SIGNATURE, assemble, chunk, chunks_of, make_png

from aether3d.errors import MaterialError
from aether3d.materials.images import ImageLimits, decode_image, decode_png

PIL = pytest.importorskip("PIL.Image")


def expected_rgba(samples: np.ndarray, depth: int, ctype: int, palette=None, trns=None):
    """What the decoder must return, computed independently from the sample values."""
    s = samples.astype(np.int64)
    h, w = s.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.int64)

    def to8(v):
        if depth == 16:
            return (v * 255 + 32767) // 65535
        return v * (255 // (2**depth - 1))

    if ctype == 3:
        out[..., :3] = np.asarray(palette)[s[..., 0]]
        alpha = np.full(256, 255)
        if trns is not None:
            alpha[: len(trns)] = list(trns)
        out[..., 3] = alpha[s[..., 0]]
        return out.astype(np.uint8)
    if ctype in (0, 4):
        out[..., 0] = out[..., 1] = out[..., 2] = to8(s[..., 0])
        out[..., 3] = to8(s[..., 1]) if ctype == 4 else 255
        if trns is not None:
            (key,) = struct.unpack(">H", trns)
            out[..., 3] = np.where(s[..., 0] == key, 0, 255)
    else:
        out[..., :3] = to8(s[..., :3])
        out[..., 3] = to8(s[..., 3]) if ctype == 6 else 255
        if trns is not None:
            key = np.array(struct.unpack(">HHH", trns))
            out[..., 3] = np.where(np.all(s[..., :3] == key, axis=-1), 0, 255)
    return out.astype(np.uint8)


CASES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (0, 8),
    (0, 16),
    (2, 8),
    (2, 16),
    (3, 1),
    (3, 2),
    (3, 4),
    (3, 8),
    (4, 8),
    (4, 16),
    (6, 8),
    (6, 16),
]


@pytest.mark.parametrize("ctype,depth", CASES)
@pytest.mark.parametrize("interlace", [False, True])
@pytest.mark.parametrize("size", [(1, 1), (5, 3), (13, 9), (33, 20)])
def test_every_color_type_bit_depth_and_interlacing_decodes_exactly(ctype, depth, interlace, size):
    w, h = size
    rng = np.random.default_rng(ctype * 100 + depth + w)
    samples = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    top = 2**depth - 1
    palette = None
    if ctype == 3:
        entries = min(2**depth, 40)
        palette = rng.integers(0, 256, (entries, 3))
        top = entries - 1
    image = rng.integers(0, top + 1, (h, w, samples))
    data = make_png(
        image, depth, ctype, interlace=interlace, palette=palette, filters="random", seed=w
    )
    decoded = decode_png(data)
    assert decoded.pixels.shape == (h, w, 4) and decoded.pixels.dtype == np.uint8
    assert np.array_equal(decoded.pixels, expected_rgba(image, depth, ctype, palette))
    assert decoded.bit_depth == depth and not decoded.pixels.flags.writeable


@pytest.mark.parametrize(
    "ctype,depth,trns",
    [
        (0, 8, struct.pack(">H", 7)),
        (0, 16, struct.pack(">H", 300)),
        (2, 8, struct.pack(">HHH", 1, 2, 3)),
        (0, 2, struct.pack(">H", 1)),
    ],
)
def test_trns_color_keys_make_matching_pixels_transparent(ctype, depth, trns):
    rng = np.random.default_rng(3)
    samples = 1 if ctype == 0 else 3
    image = rng.integers(0, min(2**depth, 9), (6, 7, samples))
    image[0, 0] = struct.unpack(">" + "H" * samples, trns)  # at least one keyed pixel
    data = make_png(image, depth, ctype, trns=trns)
    decoded = decode_png(data)
    assert decoded.has_alpha
    assert np.array_equal(decoded.pixels, expected_rgba(image, depth, ctype, trns=trns))
    assert decoded.pixels[0, 0, 3] == 0


def test_palette_transparency():
    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    image = np.array([[[0], [1], [2], [1]]])
    data = make_png(image, 8, 3, palette=palette, trns=bytes([0, 128]))
    pixels = decode_png(data).pixels
    assert pixels[0, :, 3].tolist() == [0, 128, 255, 128]  # entries beyond tRNS are opaque


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "P", "1"])
def test_pillow_written_pngs_decode_like_pillow(mode):
    rng = np.random.default_rng(len(mode))
    arr = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    img = PIL.fromarray(arr, "RGBA").convert(mode)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    ours = decode_image(buffer.getvalue()).pixels
    theirs = np.array(PIL.open(io.BytesIO(buffer.getvalue())).convert("RGBA"))
    assert np.array_equal(ours, theirs)


def test_interlaced_files_decode_like_pillow():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (29, 31, 4))
    data = make_png(image, 8, 6, interlace=True, filters="random")
    theirs = np.array(PIL.open(io.BytesIO(data)).convert("RGBA"))
    assert np.array_equal(decode_png(data).pixels, theirs)


def test_large_paeth_filtered_images_decode_quickly():
    import time

    from aether3d.rendering.png import encode_png

    rng = np.random.default_rng(0)
    image = np.cumsum(rng.integers(0, 4, (512, 768, 3), dtype=np.uint8), axis=1, dtype=np.uint8)
    data = encode_png(image)  # adaptive filters: every row type occurs
    t0 = time.perf_counter()
    decoded = decode_png(data)
    assert np.array_equal(decoded.pixels[..., :3], image)
    assert time.perf_counter() - t0 < 5.0  # ~0.1 s on a laptop


def test_multiple_idat_chunks_and_ancillary_chunks_are_fine():
    image = np.arange(24).reshape(2, 4, 3) * 10
    data = make_png(image, 8, 2)
    parts = chunks_of(data)
    idat = next(d for k, d in parts if k == b"IDAT")
    rebuilt = [parts[0], (b"tEXt", b"Comment\x00hello"), (b"gAMA", struct.pack(">I", 45455))]
    rebuilt += [(b"IDAT", idat[:5]), (b"IDAT", idat[5:]), (b"zzZz", b"ignored"), (b"IEND", b"")]
    assert np.array_equal(decode_png(assemble(rebuilt)).pixels[..., :3], image)


def valid_png() -> bytes:
    return make_png(np.arange(48).reshape(4, 4, 3), 8, 2)


def mutate(kind: bytes, payload: bytes) -> bytes:
    return assemble([(k, payload if k == kind else d) for k, d in chunks_of(valid_png())])


BROKEN = {
    "bad signature": b"\x89PNX\r\n\x1a\n" + valid_png()[8:],
    "truncated": valid_png()[:-20],
    "no IEND": valid_png()[:-12],
    "bad crc": valid_png()[:-4] + b"\x00\x00\x00\x01",
    "IHDR not first": SIGNATURE + chunk(b"tEXt", b"a\x00b") + valid_png()[8:],
    "two IHDR": assemble([chunks_of(valid_png())[0], *chunks_of(valid_png())]),
    "bad bit depth": mutate(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 3, 2, 0, 0, 0)),
    "zero width": mutate(b"IHDR", struct.pack(">IIBBBBB", 0, 4, 8, 2, 0, 0, 0)),
    "bad interlace": mutate(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 2)),
    "short IHDR": mutate(b"IHDR", b"\x00" * 12),
    "data too long": mutate(b"IDAT", zlib.compress(bytes(13 * 5))),
    "data too short": mutate(b"IDAT", zlib.compress(bytes(13 * 3))),
    "bad filter": mutate(b"IDAT", zlib.compress(bytes([7] + [0] * 12) * 4)),
    "corrupt zlib": mutate(b"IDAT", b"\x78\x9c\xff\xff\xff\xff"),
    "incomplete zlib": mutate(b"IDAT", zlib.compress(bytes(13 * 4))[:-6]),
    "no IDAT": assemble([c for c in chunks_of(valid_png()) if c[0] != b"IDAT"]),
    "unknown critical chunk": assemble(
        [chunks_of(valid_png())[0], (b"ABCD", b""), *chunks_of(valid_png())[1:]]
    ),
    "PLTE in grayscale": assemble(
        [
            chunks_of(make_png(np.zeros((2, 2, 1)), 8, 0))[0],
            (b"PLTE", bytes(6)),
            *chunks_of(make_png(np.zeros((2, 2, 1)), 8, 0))[1:],
        ]
    ),
    "indexed without PLTE": assemble(
        [
            c
            for c in chunks_of(make_png(np.zeros((2, 2, 1)), 8, 3, palette=[[0, 0, 0]]))
            if c[0] != b"PLTE"
        ]
    ),
    "palette index out of range": make_png(
        np.array([[[0], [5]]]), 8, 3, palette=[[0, 0, 0], [1, 1, 1]]
    ),
    "tRNS longer than palette": make_png(
        np.array([[[0]]]), 8, 3, palette=[[0, 0, 0]], trns=bytes(2)
    ),
    "tRNS with alpha": assemble(
        [
            *chunks_of(make_png(np.zeros((1, 1, 4)), 8, 6))[:1],
            (b"tRNS", bytes(2)),
            *chunks_of(make_png(np.zeros((1, 1, 4)), 8, 6))[1:],
        ]
    ),
    "IDAT not consecutive": None,
}


def _split_idat() -> bytes:
    parts = chunks_of(valid_png())
    idat = next(d for k, d in parts if k == b"IDAT")
    return assemble(
        [parts[0], (b"IDAT", idat[:4]), (b"tEXt", b"a\x00b"), (b"IDAT", idat[4:]), (b"IEND", b"")]
    )


BROKEN["IDAT not consecutive"] = _split_idat()


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_malformed_pngs_are_rejected_with_a_code(name):
    with pytest.raises(MaterialError) as info:
        decode_image(BROKEN[name], location="textures[T]")
    assert info.value.code in ("INVALID_IMAGE", "UNSUPPORTED_IMAGE")
    assert info.value.location == "textures[T]"


def test_size_limits_are_checked_before_decompressing():
    huge = mutate(b"IHDR", struct.pack(">IIBBBBB", 20000, 20000, 8, 6, 0, 0, 0))
    with pytest.raises(MaterialError) as info:
        decode_png(huge)
    assert info.value.code == "IMAGE_TOO_LARGE"
    with pytest.raises(MaterialError) as info:
        decode_png(valid_png(), limits=ImageLimits(max_dimension=3))
    assert info.value.code == "IMAGE_TOO_LARGE"
    with pytest.raises(MaterialError) as info:
        decode_png(valid_png(), limits=ImageLimits(max_pixels=15))
    assert info.value.code == "IMAGE_TOO_LARGE"
    with pytest.raises(MaterialError) as info:
        decode_image(valid_png(), limits=ImageLimits(max_bytes=50))
    assert info.value.code == "IMAGE_TOO_LARGE"


def test_a_decompression_bomb_stops_at_the_declared_size():
    import time

    bomb = zlib.compress(bytes(200 * 2**20), 9)  # 200 MiB of zeros in ~200 KiB
    data = assemble(
        [(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0)), (b"IDAT", bomb), (b"IEND", b"")]
    )
    t0 = time.perf_counter()
    with pytest.raises(MaterialError, match="larger than its header"):
        decode_png(data)
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.parametrize(
    "data",
    [b"GIF89a\x01\x00", b"RIFF\x00\x00\x00\x00WEBPVP8 ", b"BM" + bytes(40), b"", b"not an image"],
)
def test_other_formats_are_unsupported(data):
    with pytest.raises(MaterialError) as info:
        decode_image(data)
    assert info.value.code == "UNSUPPORTED_IMAGE"


def test_the_format_comes_from_the_signature_not_the_name():
    rgb = np.full((3, 3, 3), 77, np.uint8)
    buffer = io.BytesIO()
    PIL.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    decoded = decode_image(buffer.getvalue())
    assert decoded.format == "jpeg" and abs(int(decoded.pixels[1, 1, 0]) - 77) <= 2


@pytest.mark.parametrize("mode", ["RGB", "L", "CMYK"])
def test_jpegs_decode_through_pillow(mode):
    rng = np.random.default_rng(1)
    arr = np.repeat(np.repeat(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8), 8, 0), 8, 1)
    img = PIL.fromarray(arr).convert(mode)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=98)
    decoded = decode_image(buffer.getvalue())
    reference = np.array(PIL.open(io.BytesIO(buffer.getvalue())).convert("RGBA"))
    assert decoded.pixels.shape == (48, 64, 4)
    assert np.array_equal(decoded.pixels, reference)
    assert decoded.mime_type == "image/jpeg" and decoded.channels == (1 if mode == "L" else 3)


def test_broken_and_oversized_jpegs_are_rejected():
    buffer = io.BytesIO()
    PIL.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buffer, format="JPEG")
    data = buffer.getvalue()
    with pytest.raises(MaterialError) as info:
        decode_image(data[: len(data) // 3])
    assert info.value.code == "INVALID_IMAGE"
    with pytest.raises(MaterialError) as info:
        decode_image(data, limits=ImageLimits(max_dimension=32))
    assert info.value.code == "IMAGE_TOO_LARGE"


def test_jpeg_without_pillow_gives_an_actionable_error(monkeypatch):
    buffer = io.BytesIO()
    PIL.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buffer, format="JPEG")
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(MaterialError) as info:
        decode_image(buffer.getvalue())
    assert info.value.code == "IMAGE_DECODER_MISSING" and "aether3d[images]" in info.value.hint
    # PNG never needs Pillow.
    assert decode_image(valid_png()).width == 4
