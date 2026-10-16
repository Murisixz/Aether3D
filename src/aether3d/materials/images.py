"""Safe image decoding for textures: PNG (built in) and JPEG (optional Pillow).

Every image is treated as untrusted input:

* the format is recognised from the file's first bytes (its signature), never
  from a file name or a declared MIME type;
* sizes are checked against :class:`ImageLimits` *before* pixels are
  allocated or decompressed, and compressed data may not expand beyond the
  size its header declares (no decompression bombs);
* any malformation -- bad signature, CRC or chunk order, truncated or
  surplus data, invalid filter types, out-of-range palette indices -- raises
  :class:`~aether3d.errors.MaterialError` with a stable ``code``
  (``INVALID_IMAGE``, ``IMAGE_TOO_LARGE``, ``UNSUPPORTED_IMAGE``,
  ``IMAGE_DECODER_MISSING``).

**PNG** is decoded by Aether3D itself (W3C PNG specification, 3rd edition):
every color type (gray, RGB, palette, gray + alpha, RGBA), bit depths 1-16,
``tRNS`` transparency and Adam7 interlacing. Color-management chunks
(``gAMA``, ``cHRM``, ``iCCP``, ``sRGB``, ``cICP``) are ignored: texture data is
interpreted by its material role (sRGB for colors, linear for data), as in
glTF. **JPEG** needs Pillow (``pip install 'aether3d[images]'``); EXIF
orientation is not applied (textures map by UV, not by camera orientation).

Decoded images are 8-bit RGBA (:class:`DecodedImage`); 16-bit samples are
rounded to 8 bits (``bit_depth`` keeps the original depth).
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .. import _native
from ..errors import MaterialError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
MIME_TYPES = {"png": "image/png", "jpeg": "image/jpeg"}

# Valid (color type -> allowed bit depths, samples per pixel).
_COLOR_TYPES = {
    0: ((1, 2, 4, 8, 16), 1),  # grayscale
    2: ((8, 16), 3),  # truecolor
    3: ((1, 2, 4, 8), 1),  # indexed
    4: ((8, 16), 2),  # grayscale + alpha
    6: ((8, 16), 4),  # truecolor + alpha
}
# Adam7 passes: (x start, y start, x step, y step).
_ADAM7 = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
_MAX_CHUNKS = 1_000_000


@dataclass(frozen=True)
class ImageLimits:
    """Safety limits for decoding one image (checked before allocating pixels).

    * ``max_bytes`` -- largest encoded file (default 256 MiB).
    * ``max_dimension`` -- largest width or height in pixels (default 16384).
    * ``max_pixels`` -- largest ``width x height`` (default 8192 x 8192).
    """

    max_bytes: int = 256 * 2**20
    max_dimension: int = 16384
    max_pixels: int = 8192 * 8192


@dataclass(frozen=True)
class DecodedImage:
    """An image decoded to 8-bit RGBA (row 0 is the top row of the picture)."""

    pixels: np.ndarray  # (height, width, 4) uint8, read-only
    format: str  # "png" or "jpeg"
    bit_depth: int  # bits per sample in the file (1-16)
    channels: int  # channels in the file: 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA
    has_alpha: bool  # the file carries transparency (alpha channel or tRNS)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def mime_type(self) -> str:
        return MIME_TYPES[self.format]

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "has_alpha": self.has_alpha,
        }


def sniff_format(data: bytes) -> str | None:
    """``"png"``, ``"jpeg"`` or ``None`` from the first bytes of ``data``."""
    if data[:8] == PNG_SIGNATURE:
        return "png"
    if data[:3] == JPEG_SIGNATURE:
        return "jpeg"
    return None


def _invalid(
    message: str, location: str | None, code: str = "INVALID_IMAGE", **kw
) -> MaterialError:
    return MaterialError(code, message, location, **kw)


def _check_size(width: int, height: int, limits: ImageLimits, location: str | None) -> None:
    if width < 1 or height < 1:
        raise _invalid(f"image size {width}x{height} is empty", location)
    if width > limits.max_dimension or height > limits.max_dimension:
        raise _invalid(
            f"image size {width}x{height} exceeds the limit of {limits.max_dimension} pixels "
            "per side",
            location,
            "IMAGE_TOO_LARGE",
            hint="downscale the image, or raise ImageLimits.max_dimension",
        )
    if width * height > limits.max_pixels:
        raise _invalid(
            f"image size {width}x{height} ({width * height:,} pixels) exceeds the limit of "
            f"{limits.max_pixels:,} pixels",
            location,
            "IMAGE_TOO_LARGE",
            hint="downscale the image, or raise ImageLimits.max_pixels",
        )


def decode_image(
    data: bytes, *, limits: ImageLimits | None = None, location: str | None = None
) -> DecodedImage:
    """Decode PNG or JPEG bytes (recognised by their signature) to 8-bit RGBA."""
    limits = limits or ImageLimits()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise _invalid(f"image data must be bytes, got {type(data).__name__}", location)
    data = bytes(data)
    if len(data) > limits.max_bytes:
        raise _invalid(
            f"image file has {len(data):,} bytes (limit {limits.max_bytes:,})",
            location,
            "IMAGE_TOO_LARGE",
        )
    kind = sniff_format(data)
    if kind == "png":
        return decode_png(data, limits=limits, location=location)
    if kind == "jpeg":
        return decode_jpeg(data, limits=limits, location=location)
    raise _invalid(
        "unsupported image format (only PNG and JPEG images can be used as textures)",
        location,
        "UNSUPPORTED_IMAGE",
        hint="convert the image to PNG or JPEG",
        details={"signature": data[:8].hex()},
    )


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #


def _chunks(data: bytes, location: str | None):
    """Yield ``(type, payload)`` for each chunk, checking lengths and CRCs."""
    pos = len(PNG_SIGNATURE)
    count = 0
    while True:
        if pos + 8 > len(data):
            raise _invalid("truncated PNG: the IEND chunk is missing", location)
        (length,) = struct.unpack_from(">I", data, pos)
        kind = data[pos + 4 : pos + 8]
        if length > 2**31 - 1:
            raise _invalid(f"invalid PNG chunk length {length}", location)
        if not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind):
            raise _invalid(f"invalid PNG chunk type {kind!r}", location)
        end = pos + 8 + length
        if end + 4 > len(data):
            raise _invalid(f"truncated PNG: chunk {kind.decode('ascii')} ends early", location)
        payload = data[pos + 8 : end]
        (crc,) = struct.unpack_from(">I", data, end)
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise _invalid(f"PNG chunk {kind.decode('ascii')} has a bad CRC", location)
        count += 1
        if count > _MAX_CHUNKS:
            raise _invalid("PNG has too many chunks", location)
        yield kind.decode("ascii"), payload
        pos = end + 4
        if kind == b"IEND":
            return


def _row_bytes(width: int, bit_depth: int, samples: int) -> int:
    return (width * bit_depth * samples + 7) // 8


def decode_png(
    data: bytes, *, limits: ImageLimits | None = None, location: str | None = None
) -> DecodedImage:
    """Decode a PNG file (see the module docstring for what is supported and checked)."""
    limits = limits or ImageLimits()
    if data[:8] != PNG_SIGNATURE:
        raise _invalid("not a PNG file (bad signature)", location)
    header = None
    palette = None
    transparency = None
    idat: list[bytes] = []
    previous = None
    for index, (kind, payload) in enumerate(_chunks(data, location)):
        if index == 0 and kind != "IHDR":
            raise _invalid("the first PNG chunk must be IHDR", location)
        if kind == "IHDR":
            if index != 0:
                raise _invalid("PNG has more than one IHDR chunk", location)
            if len(payload) != 13:
                raise _invalid("PNG IHDR chunk must be 13 bytes", location)
            width, height, depth, ctype, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if ctype not in _COLOR_TYPES or depth not in _COLOR_TYPES[ctype][0]:
                raise _invalid(f"invalid PNG bit depth {depth} for color type {ctype}", location)
            if compression != 0 or filtering != 0 or interlace not in (0, 1):
                raise _invalid("invalid PNG compression, filter or interlace method", location)
            _check_size(width, height, limits, location)
            header = (width, height, depth, ctype, interlace)
        elif kind == "PLTE":
            if idat or palette is not None:
                raise _invalid("misplaced PNG PLTE chunk", location)
            if header[3] in (0, 4):
                raise _invalid("PNG PLTE chunk is not allowed for grayscale images", location)
            if len(payload) % 3 or not 1 <= len(payload) // 3 <= 256:
                raise _invalid("invalid PNG palette length", location)
            if header[3] == 3 and len(payload) // 3 > 2 ** header[2]:
                raise _invalid("PNG palette has more entries than the bit depth allows", location)
            palette = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        elif kind == "tRNS":
            if idat or transparency is not None:
                raise _invalid("misplaced PNG tRNS chunk", location)
            ctype = header[3]
            if ctype in (4, 6):
                raise _invalid("PNG tRNS chunk is not allowed with an alpha channel", location)
            if ctype == 3 and (palette is None or len(payload) > len(palette)):
                raise _invalid("invalid PNG tRNS chunk for an indexed image", location)
            if (ctype == 0 and len(payload) != 2) or (ctype == 2 and len(payload) != 6):
                raise _invalid("invalid PNG tRNS chunk length", location)
            transparency = payload
        elif kind == "IDAT":
            if idat and previous != "IDAT":
                raise _invalid("PNG IDAT chunks must be consecutive", location)
            if header[3] == 3 and palette is None:
                raise _invalid("indexed PNG has no PLTE chunk before its image data", location)
            idat.append(payload)
        elif kind == "IEND":
            if payload:
                raise _invalid("PNG IEND chunk must be empty", location)
        elif kind[0].isupper():  # a critical chunk this decoder does not know
            raise _invalid(
                f"PNG uses the unknown critical chunk {kind}", location, "UNSUPPORTED_IMAGE"
            )
        # Ancillary chunks (text, time, color management, ...) are ignored.
        previous = kind
    if header is None or not idat:
        raise _invalid("PNG has no image data", location)
    width, height, depth, ctype, interlace = header
    samples = _COLOR_TYPES[ctype][1]
    bpp = max(1, depth * samples // 8)  # bytes per complete pixel, for filtering

    passes = []
    if interlace:
        for x0, y0, dx, dy in _ADAM7:
            pw = (width - x0 + dx - 1) // dx if width > x0 else 0
            ph = (height - y0 + dy - 1) // dy if height > y0 else 0
            passes.append((x0, y0, dx, dy, pw, ph))
    else:
        passes.append((0, 0, 1, 1, width, height))
    expected = sum(ph * (1 + _row_bytes(pw, depth, samples)) for *_, pw, ph in passes if pw and ph)
    raw = _inflate(b"".join(idat), expected, location)

    sample_dtype = np.uint16 if depth == 16 else np.uint8
    image = np.zeros((height, width, samples), dtype=sample_dtype)
    offset = 0
    for x0, y0, dx, dy, pw, ph in passes:
        if not pw or not ph:
            continue
        stride = _row_bytes(pw, depth, samples)
        block = np.frombuffer(raw, dtype=np.uint8, count=ph * (stride + 1), offset=offset)
        offset += ph * (stride + 1)
        rows = _unfilter(block.reshape(ph, stride + 1), bpp, location)
        image[y0::dy, x0::dx] = _unpack(rows, pw, depth, samples)
    return _to_rgba(image, depth, ctype, palette, transparency, location)


def _inflate(compressed: bytes, expected: int, location: str | None) -> bytes:
    """Decompress exactly ``expected`` bytes; more or fewer is an error (no zip bombs)."""
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(compressed, expected)
        tail = inflater.unconsumed_tail
        while tail:  # the rest may only be the stream's end (checksum, empty blocks)
            if inflater.decompress(tail, 1):
                raise _invalid("PNG image data is larger than its header declares", location)
            if inflater.unconsumed_tail == tail:  # pragma: no cover - no progress
                break
            tail = inflater.unconsumed_tail
        if not inflater.eof:
            raise _invalid("PNG image data is truncated (incomplete zlib stream)", location)
    except zlib.error as exc:
        raise _invalid(f"PNG image data is corrupt ({exc})", location) from None
    if len(raw) != expected:
        raise _invalid(f"PNG image data has {len(raw):,} bytes, expected {expected:,}", location)
    return raw


def _unfilter(block: np.ndarray, bpp: int, location: str | None) -> np.ndarray:
    """Undo the row filters of one (sub-)image: ``(rows, 1 + stride)`` -> ``(rows, stride)``."""
    types = block[:, 0]
    if types.max(initial=0) > 4:
        raise _invalid(f"invalid PNG filter type {int(types.max())}", location)
    filtered = block[:, 1:]
    height, stride = filtered.shape
    native = _native.module()
    if native is not None:  # the same bytes (aether3d-native)
        out = np.empty((height, stride), dtype=np.uint8)
        native.png_unfilter(np.ascontiguousarray(block), height, stride, bpp, out)
        return out
    if not np.any(types >= 3):
        # None, Sub and Up only: rows are independent apart from Up's previous row.
        out = np.empty_like(filtered)
        previous = np.zeros(stride, dtype=np.uint8)
        for r in range(height):
            row = filtered[r]
            kind = types[r]
            if kind == 0:
                out[r] = row
            elif kind == 1:
                lanes = np.zeros(((stride + bpp - 1) // bpp) * bpp, dtype=np.uint8)
                lanes[:stride] = row
                out[r] = np.cumsum(lanes.reshape(-1, bpp), axis=0, dtype=np.uint8).reshape(-1)[
                    :stride
                ]
            else:
                out[r] = row + previous
            previous = out[r]
        return out
    return _unfilter_wavefront(filtered, types, bpp)


def _unfilter_wavefront(filtered: np.ndarray, types: np.ndarray, bpp: int) -> np.ndarray:
    """All five filters at once, one anti-diagonal of pixels at a time.

    Every predictor of pixel ``(r, x)`` uses only ``(r, x-1)``, ``(r-1, x)`` and
    ``(r-1, x-1)``, which lie on earlier anti-diagonals ``r + x``. Each step
    reconstructs one anti-diagonal for all rows at once (vectorized), so the
    Python loop runs ``width + height`` times instead of once per byte.
    """
    height, stride = filtered.shape
    width = (stride + bpp - 1) // bpp
    padded = np.zeros((height, width * bpp), dtype=np.uint8)
    padded[:, :stride] = filtered
    f = padded.reshape(height, width, bpp).astype(np.int16)
    # recon has an extra zero row on top and an extra zero column on the left.
    recon = np.zeros((height + 1, width + 1, bpp), dtype=np.int16)
    kinds = types.astype(np.int16)
    for d in range(height + width - 1):
        r0, r1 = max(0, d - width + 1), min(height - 1, d)
        r = np.arange(r0, r1 + 1)
        x = d - r
        a = recon[r + 1, x]  # left
        b = recon[r, x + 1]  # up
        c = recon[r, x]  # upper left
        kind = kinds[r][:, None]
        p = a + b - c
        pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
        paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
        predictor = np.select(
            [kind == 1, kind == 2, kind == 3, kind == 4],
            [a, b, (a + b) >> 1, paeth],
            default=0,
        )
        recon[r + 1, x + 1] = (f[r, x] + predictor) & 0xFF
    return recon[1:, 1:].reshape(height, width * bpp)[:, :stride].astype(np.uint8)


def _unpack(rows: np.ndarray, width: int, depth: int, samples: int) -> np.ndarray:
    """Scanline bytes -> ``(rows, width, samples)`` samples (uint8, or uint16 for depth 16)."""
    n = rows.shape[0]
    if depth == 16:
        values = rows.reshape(n, -1).view(">u2").astype(np.uint16)
        return values.reshape(n, width, samples)
    if depth == 8:
        return rows.reshape(n, width, samples)
    bits = np.unpackbits(rows, axis=1)  # MSB first, as PNG packs sub-byte samples
    per_row = width * samples * depth
    bits = bits[:, :per_row].reshape(n, width * samples, depth)
    weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint8)
    values = (bits * weights).sum(axis=2, dtype=np.uint16).astype(np.uint8)
    return values.reshape(n, width, samples)


def _to_8bit(values: np.ndarray, depth: int) -> np.ndarray:
    if depth == 16:
        return ((values.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
    if depth == 8:
        return values.astype(np.uint8)
    scale = 255 // (2**depth - 1)  # 1 -> 255, 2 -> 85, 4 -> 17 (exact)
    return (values.astype(np.uint16) * scale).astype(np.uint8)


def _to_rgba(
    image: np.ndarray,
    depth: int,
    ctype: int,
    palette: np.ndarray | None,
    transparency: bytes | None,
    location: str | None,
) -> DecodedImage:
    height, width, _ = image.shape
    out = np.empty((height, width, 4), dtype=np.uint8)
    has_alpha = ctype in (4, 6) or transparency is not None
    channels = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}[ctype]
    if ctype == 3:
        indices = image[..., 0]
        assert palette is not None
        if int(indices.max(initial=0)) >= len(palette):
            raise _invalid("PNG pixel uses a palette index beyond the palette", location)
        out[..., :3] = palette[indices]
        alpha = np.full(256, 255, dtype=np.uint8)
        if transparency is not None:
            alpha[: len(transparency)] = np.frombuffer(transparency, dtype=np.uint8)
            channels = 4
        out[..., 3] = alpha[indices]
        return DecodedImage(_readonly(out), "png", depth, channels, has_alpha)
    if ctype in (0, 4):
        gray = _to_8bit(image[..., 0], depth)
        out[..., 0] = out[..., 1] = out[..., 2] = gray
        if ctype == 4:
            out[..., 3] = _to_8bit(image[..., 1], depth)
        elif transparency is not None:
            (key,) = struct.unpack(">H", transparency)
            out[..., 3] = np.where(image[..., 0] == key, 0, 255)
        else:
            out[..., 3] = 255
    else:
        out[..., :3] = _to_8bit(image[..., :3], depth)
        if ctype == 6:
            out[..., 3] = _to_8bit(image[..., 3], depth)
        elif transparency is not None:
            key = np.array(struct.unpack(">HHH", transparency), dtype=np.uint16)
            out[..., 3] = np.where(np.all(image[..., :3] == key, axis=-1), 0, 255)
        else:
            out[..., 3] = 255
    if transparency is not None:
        channels += 1
    return DecodedImage(_readonly(out), "png", depth, channels, has_alpha)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# --------------------------------------------------------------------------- #
# JPEG (optional Pillow)
# --------------------------------------------------------------------------- #


def jpeg_available() -> bool:
    """Whether JPEG images can be decoded (Pillow is installed)."""
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


def decode_jpeg(
    data: bytes, *, limits: ImageLimits | None = None, location: str | None = None
) -> DecodedImage:
    """Decode a baseline or progressive JPEG with Pillow (an optional dependency)."""
    limits = limits or ImageLimits()
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        raise _invalid(
            "JPEG textures need the optional Pillow package",
            location,
            "IMAGE_DECODER_MISSING",
            hint="pip install 'aether3d[images]' (or convert the texture to PNG)",
        ) from None
    if data[:3] != JPEG_SIGNATURE:
        raise _invalid("not a JPEG file (bad signature)", location)
    try:
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as img:
            width, height = img.size  # read from the header; pixels are not decoded yet
            _check_size(width, height, limits, location)
            mode = img.mode
            if mode not in ("L", "RGB", "CMYK"):
                raise _invalid(
                    f"unsupported JPEG color mode {mode!r}", location, "UNSUPPORTED_IMAGE"
                )
            img.load()
            rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    except MaterialError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise _invalid(f"JPEG data is corrupt or truncated ({exc})", location) from None
    except Image.DecompressionBombError as exc:  # pragma: no cover - our limit is stricter
        raise _invalid(str(exc), location, "IMAGE_TOO_LARGE") from None
    if rgba.shape != (height, width, 4):
        raise _invalid("JPEG decoded to an unexpected size", location)
    channels = 1 if mode == "L" else 3
    return DecodedImage(_readonly(rgba), "jpeg", 8, channels, False)
