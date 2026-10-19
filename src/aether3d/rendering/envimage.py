"""Image environments: equirectangular HDR images that light a scene (0.20.0).

``scene.world.environment`` with ``type="image"`` takes ``image``: a Radiance RGBE file
(``.hdr``, ``.pic``; flat or run-length scanlines) or a portable float map (``.pfm``)
inside the project's directory, or a built-in environment drawn by Aether3D
(``"builtin:studio"``, ``"builtin:overcast"``, ``"builtin:sunset"``, ``"builtin:night"``).
Files are untrusted: sizes are checked before anything is allocated and problems raise
:class:`~aether3d.errors.RenderError` with a code (``ENVIRONMENT_*``).

Mapping (Blender's, measured with Blender 5.2: research/2026-10-stage19/blender/
e6_environment): a world direction ``d`` (unit, +Z up) is first turned by ``rotation``
degrees about +Z (``e = R_z(rotation) d``, as Blender's mapping node turns the lookup
vector), then read at ``u = 0.5 - atan2(e_y, e_x) / (2 pi)`` (mod 1) and ``v = 0.5 +
asin(e_z) / pi``: +X is the middle of the image, +Y a quarter from its left edge, +Z its
top row. Pixels are interpolated bilinearly, wrapping across the left and right edges.

Images are used at up to :data:`MAX_WIDTH` x :data:`MAX_HEIGHT` pixels (larger ones are
halved by 2 x 2 box filtering until they fit), so both path tracers hold the same data.

* :meth:`EnvironmentImage.sample` -- importance sampling: a piecewise-constant
  distribution over cells (the pixels, or 2 x 2 blocks of them beyond 1024 columns), each
  weighted by the largest luminance of it and its neighbours (bilinear lookups never see
  light a zero-probability cell would miss) times the cell's solid angle; densities are
  per steradian, ``pdf_uv / (2 pi^2 cos(elevation))``.
* :meth:`EnvironmentImage.raster_lighting` -- what the rasterizers light with (an
  approximation; the path tracers integrate the image itself): a compact bright source (a
  sun: the pixels above :data:`SUN_CLAMP` times the image's mean luminance, near the
  brightest one, holding at least :data:`SUN_SHARE` of the light) becomes a directional
  light along its energy-weighted direction with its irradiance, and the rest -- the image
  clamped there -- is projected on the first nine real spherical harmonics, convolved with
  the clamped cosine (Ramamoorthi and Hanrahan 2001) and divided by pi: the ambient radiance
  a white Lambertian surface of normal ``n`` reflects.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import RenderError

MAX_WIDTH = 2048
MAX_HEIGHT = 1024
MAX_FILE_BYTES = 512 * 2**20
MAX_DIMENSION = 16384
MAX_PIXELS = 16384 * 8192
BUILTINS = ("studio", "overcast", "sunset", "night")
SUFFIXES = (".hdr", ".pic", ".pfm")
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
SUN_CLAMP = 50.0
"""Pixels brighter than this times the mean luminance can make up an environment sun."""
SUN_SHARE = 0.2
"""The least share of the image's light a compact bright source needs to be a sun."""
SUN_RADIUS = 10.0
"""Degrees around the brightest pixel the environment sun is gathered from."""


def _err(code: str, message: str, location: str | None, **kw: Any) -> RenderError:
    return RenderError(code, message, location, **kw)


# --------------------------------------------------------------------------- decoders --


def _check_size(width: int, height: int, location: str | None) -> None:
    if width < 1 or height < 1:
        raise _err("ENVIRONMENT_INVALID", f"image size {width}x{height} is empty", location)
    if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise _err(
            "ENVIRONMENT_TOO_LARGE",
            f"image size {width}x{height} exceeds the limits ({MAX_DIMENSION} per side, "
            f"{MAX_PIXELS:,} pixels)",
            location,
        )


def decode_hdr(data: bytes, location: str | None = None) -> np.ndarray:
    """Radiance RGBE bytes as ``(height, width, 3) float32`` (row 0 at the top)."""
    if not (data.startswith(b"#?RADIANCE") or data.startswith(b"#?RGBE")):
        raise _err("ENVIRONMENT_INVALID", "not a Radiance HDR file (no #? header)", location)
    end = data.find(b"\n\n")
    if end < 0 or end > 65536:
        raise _err("ENVIRONMENT_INVALID", "the HDR header does not end", location)
    header = data[:end].decode("latin-1")
    fmt = re.search(r"^FORMAT=(\S+)", header, re.M)
    if fmt is not None and fmt.group(1) != "32-bit_rle_rgbe":
        raise _err(
            "ENVIRONMENT_UNSUPPORTED",
            f"HDR pixel format {fmt.group(1)!r} is not supported (only 32-bit_rle_rgbe)",
            location,
        )
    line_end = data.find(b"\n", end + 2)
    if line_end < 0:
        raise _err("ENVIRONMENT_INVALID", "the HDR resolution line is missing", location)
    res = data[end + 2 : line_end].decode("latin-1").split()
    if len(res) != 4 or res[0] != "-Y" or res[2] != "+X":
        raise _err(
            "ENVIRONMENT_UNSUPPORTED",
            f"HDR orientation {' '.join(res)!r} is not supported (only -Y H +X W)",
            location,
        )
    try:
        height, width = int(res[1]), int(res[3])
    except ValueError:
        raise _err("ENVIRONMENT_INVALID", "the HDR resolution is not numeric", location) from None
    _check_size(width, height, location)
    body = np.frombuffer(data, dtype=np.uint8, offset=line_end + 1)
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    pos = 0
    n = body.shape[0]
    for row in range(height):
        rle = (
            8 <= width < 32768
            and pos + 4 <= n
            and body[pos] == 2
            and body[pos + 1] == 2
            and (int(body[pos + 2]) << 8 | int(body[pos + 3])) == width
        )
        if not rle:  # flat scanline
            need = 4 * width
            if pos + need > n:
                raise _err("ENVIRONMENT_INVALID", f"HDR scanline {row} is truncated", location)
            rgbe[row] = body[pos : pos + need].reshape(width, 4)
            pos += need
            continue
        pos += 4
        for c in range(4):
            x = 0
            while x < width:
                if pos >= n:
                    raise _err("ENVIRONMENT_INVALID", f"HDR scanline {row} is truncated", location)
                count = int(body[pos])
                pos += 1
                if count > 128:  # a run
                    count -= 128
                    if count > width - x or pos >= n:
                        raise _err("ENVIRONMENT_INVALID", f"bad HDR run in row {row}", location)
                    rgbe[row, x : x + count, c] = body[pos]
                    pos += 1
                else:  # literal bytes
                    if count == 0 or count > width - x or pos + count > n:
                        raise _err("ENVIRONMENT_INVALID", f"bad HDR data in row {row}", location)
                    rgbe[row, x : x + count, c] = body[pos : pos + count]
                    pos += count
                x += count
    e = rgbe[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)  # 2^(e - 128) / 256
    return (rgbe[..., :3].astype(np.float64) * scale[..., None]).astype(np.float32)


def decode_pfm(data: bytes, location: str | None = None) -> np.ndarray:
    """Portable float map bytes as ``(height, width, 3) float32`` (row 0 at the top)."""
    parts = []
    pos = 0
    for _ in range(3):  # "PF" / "Pf", "W H", "scale" separated by whitespace
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        parts.append(data[start:pos].decode("latin-1"))
        if len(parts) == 2:  # width and height are two tokens
            while pos < len(data) and data[pos : pos + 1].isspace():
                pos += 1
            start = pos
            while pos < len(data) and not data[pos : pos + 1].isspace():
                pos += 1
            parts.append(data[start:pos].decode("latin-1"))
    pos += 1  # the single whitespace before the pixels
    if len(parts) != 4 or parts[0] not in ("PF", "Pf"):
        raise _err("ENVIRONMENT_INVALID", "not a portable float map (PF/Pf header)", location)
    try:
        width, height, scale = int(parts[1]), int(parts[2]), float(parts[3])
    except ValueError:
        raise _err("ENVIRONMENT_INVALID", "the PFM header is not numeric", location) from None
    _check_size(width, height, location)
    channels = 3 if parts[0] == "PF" else 1
    need = width * height * channels * 4
    if len(data) - pos < need:
        raise _err("ENVIRONMENT_INVALID", "the PFM pixels are truncated", location)
    dtype = "<f4" if scale < 0 else ">f4"
    pix = np.frombuffer(data, dtype=dtype, count=width * height * channels, offset=pos)
    pix = pix.reshape(height, width, channels)[::-1].astype(np.float32)  # bottom-up rows
    if channels == 1:
        pix = np.repeat(pix, 3, axis=2)
    if not np.all(np.isfinite(pix)):
        raise _err("ENVIRONMENT_INVALID", "the PFM has non-finite pixels", location)
    return np.maximum(pix, 0.0)


def decode(data: bytes, suffix: str, location: str | None = None) -> np.ndarray:
    if len(data) > MAX_FILE_BYTES:
        raise _err("ENVIRONMENT_TOO_LARGE", f"the file has {len(data):,} bytes", location)
    if suffix in (".hdr", ".pic"):
        return decode_hdr(data, location)
    if suffix == ".pfm":
        return decode_pfm(data, location)
    raise _err(
        "ENVIRONMENT_UNSUPPORTED",
        f"environment images are {', '.join(SUFFIXES)} files, not {suffix or 'no suffix'!r}",
        location,
    )


# --------------------------------------------------------------------------- built-ins --


def _directions(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth ``atan2(y, x)`` and elevation (radians) of every pixel centre."""
    u = (np.arange(width) + 0.5) / width
    v = 1.0 - (np.arange(height) + 0.5) / height
    return (0.5 - u)[None, :] * 2.0 * math.pi, (v - 0.5)[:, None] * math.pi


@lru_cache(maxsize=8)
def builtin(name: str, width: int = 1024) -> np.ndarray:
    """A built-in environment (``(width / 2, width, 3) float32``, read-only), drawn here."""
    height = width // 2
    az, el = _directions(width, height)
    az = np.broadcast_to(az, (height, width))
    el = np.broadcast_to(el, (height, width))
    s = np.sin(el)
    img = np.zeros((height, width, 3))

    def box(az0, az1, el0, el1, value):
        a = (az + math.pi - math.radians(az0)) % (2 * math.pi) - math.pi  # relative azimuth
        inside = (a >= 0) & (a <= math.radians(az1 - az0))
        inside &= (el >= math.radians(el0)) & (el <= math.radians(el1))
        img[inside] = value

    def disk(az0, el0, radius, value):
        d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), s], axis=-1)
        a, e = math.radians(az0), math.radians(el0)
        c = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        img[(d @ c) >= math.cos(math.radians(radius))] = value

    if name == "studio":  # a dim grey room, two soft boxes, an overhead strip, a dark floor
        img[:] = np.where((el >= 0)[..., None], 0.05, 0.02)
        box(-80, -40, 15, 50, (6.0, 6.0, 6.0))
        box(40, 80, 15, 50, (4.0, 4.0, 4.0))
        box(160, 200, 55, 80, (3.0, 3.0, 3.0))
    elif name == "overcast":  # CIE standard overcast sky, a grey ground
        sky = (1.0 + 2.0 * np.maximum(s, 0.0)) / 3.0
        img[:] = np.where((el >= 0)[..., None], sky[..., None] * (0.92, 0.95, 1.0), 0.15)
    elif name == "sunset":
        up = np.maximum(s, 0.0)[..., None]
        sky = (1.0, 0.45, 0.18) + (np.array([0.10, 0.18, 0.45]) - (1.0, 0.45, 0.18)) * up**0.5
        img[:] = np.where((el >= 0)[..., None], sky * 1.5, 0.05)
        disk(0.0, 4.0, 1.0, (4000.0, 2000.0, 700.0))  # a low sun, toward +X
    elif name == "night":
        up = np.maximum(s, 0.0)[..., None]
        sky = (0.010, 0.014, 0.030) + (np.array([0.002, 0.003, 0.010]) - 0.01) * up
        img[:] = np.where((el >= 0)[..., None], np.maximum(sky, 0.0), 0.002)
        disk(120.0, 35.0, 0.5, (40.0, 40.0, 44.0))  # the moon
        # Stars: a deterministic hash of each pixel, about one in 400 above 10 degrees.
        idx = np.arange(height * width, dtype=np.uint64).reshape(height, width)
        h = (idx * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFF)
        h = (h ^ (h >> np.uint64(15))) * np.uint64(0x2C1B3C6D) & np.uint64(0xFFFFFFFF)
        star = ((h % np.uint64(400)) == 0) & (el > math.radians(10))
        img[star] = 3.0
    else:
        raise _err(
            "ENVIRONMENT_UNSUPPORTED",
            f"no built-in environment {name!r} (built-ins: {', '.join(BUILTINS)})",
            "world.environment.image",
        )
    out = img.astype(np.float32)
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- the image --


def _fit(pixels: np.ndarray) -> np.ndarray:
    """Halve by 2 x 2 box filtering until it fits MAX_WIDTH x MAX_HEIGHT."""
    p = np.asarray(pixels, dtype=np.float64)
    while p.shape[1] > MAX_WIDTH or p.shape[0] > MAX_HEIGHT:
        h, w = p.shape[0] // 2 * 2, p.shape[1] // 2 * 2
        p = p[:h, :w]
        p = 0.25 * (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2])
    return p.astype(np.float32)


@dataclass(frozen=True, eq=False)
class EnvironmentImage:
    """An environment image ready to render: pixels (``(H, W, 3) float32``, read-only, row
    0 at the top), its rotation about +Z (degrees) and its intensity."""

    pixels: np.ndarray
    rotation: float = 0.0
    intensity: float = 1.0
    source: str = ""
    _tables: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, pixels: np.ndarray, rotation: float, intensity: float, source: str = ""):
        fits = pixels.shape[1] <= MAX_WIDTH and pixels.shape[0] <= MAX_HEIGHT
        fitted = pixels if fits and pixels.dtype == np.float32 else _fit(pixels)
        if fitted.flags.writeable:
            fitted = fitted.copy()
            fitted.flags.writeable = False
        return cls(fitted, float(rotation), float(intensity), source)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def cos_sin(self) -> tuple[float, float]:
        a = math.radians(self.rotation)
        return math.cos(a), math.sin(a)

    def to_image(self, d: np.ndarray) -> np.ndarray:
        """World directions to the image's frame (``R_z(rotation) d``)."""
        c, s = self.cos_sin
        return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)

    def uv(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = self.to_image(d)
        u = (0.5 - np.arctan2(e[:, 1], e[:, 0]) / (2.0 * math.pi)) % 1.0
        v = 0.5 + np.arcsin(np.clip(e[:, 2], -1.0, 1.0)) / math.pi
        return u, v

    def radiance(self, d: np.ndarray) -> np.ndarray:
        """``(n, 3)`` radiance toward unit world directions ``d`` (bilinear, times the
        intensity)."""
        u, v = self.uv(d)
        return self.lookup(u, v) * self.intensity

    def lookup(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        w, h = self.width, self.height
        x = u * w - 0.5
        y = (1.0 - v) * h - 0.5
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        x0 = x0.astype(np.int64) % w
        x1 = (x0 + 1) % w
        y1 = np.clip(y0.astype(np.int64) + 1, 0, h - 1)
        y0 = np.clip(y0.astype(np.int64), 0, h - 1)
        p = self.pixels
        top = p[y0, x0] * (1.0 - fx) + p[y0, x1] * fx
        bottom = p[y1, x0] * (1.0 - fx) + p[y1, x1] * fx
        return top * (1.0 - fy) + bottom * fy

    # -- importance sampling ------------------------------------------------------------

    def tables(self) -> dict[str, Any]:
        """The sampling distribution: ``cells`` ``(nv, nu)``, the marginal CDF over rows
        (``nv + 1``) and the conditional CDFs (``(nv, nu + 1)``), float32."""
        t = self._tables.get("sampling")
        if t is not None:
            return t
        p = np.asarray(self.pixels, np.float64)
        if p.shape[1] > 1024:
            h, w = p.shape[0] // 2 * 2, p.shape[1] // 2 * 2
            p = p[:h, :w]
            p = 0.25 * (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2])
        lum = np.maximum(p @ LUMINANCE, 0.0)
        # The largest luminance of each cell and its neighbours (wrapping across u).
        big = lum.copy()
        for dy in (-1, 0, 1):
            rolled = np.roll(lum, dy, axis=0)
            if dy == 1:
                rolled[0] = lum[0]
            elif dy == -1:
                rolled[-1] = lum[-1]
            for dx in (-1, 0, 1):
                big = np.maximum(big, np.roll(rolled, dx, axis=1))
        nv, nu = lum.shape
        el = (0.5 - (np.arange(nv) + 0.5) / nv) * math.pi
        weight = big * np.cos(el)[:, None]
        rows = weight.sum(axis=1)
        total = float(rows.sum())
        if not total > 0.0:
            t = {"cells": (nv, nu), "black": True}
            self._tables["sampling"] = t
            return t
        cond = np.zeros((nv, nu + 1))
        safe = np.where(rows > 0.0, rows, 1.0)
        cond[:, 1:] = np.cumsum(weight, axis=1) / safe[:, None]
        cond[:, -1] = 1.0
        cond[rows <= 0.0, 1:] = np.arange(1, nu + 1) / nu  # (never picked)
        marg = np.zeros(nv + 1)
        marg[1:] = np.cumsum(rows) / total
        marg[-1] = 1.0
        t = {
            "cells": (nv, nu),
            "black": False,
            "marginal": marg.astype(np.float32),
            "conditional": cond.astype(np.float32),
        }
        self._tables["sampling"] = t
        return t

    def sample(self, u0: np.ndarray, u1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Directions (world, unit) drawn by the distribution and their densities (per
        steradian); ``u0`` picks the row, ``u1`` the column, their remainders the point."""
        t = self.tables()
        nv, nu = t["cells"]
        marg = t["marginal"].astype(np.float64)
        cond = t["conditional"].astype(np.float64)
        r = np.clip(np.searchsorted(marg, u0, side="right") - 1, 0, nv - 1)
        pr = marg[r + 1] - marg[r]
        fr = np.clip((u0 - marg[r]) / np.where(pr > 0, pr, 1.0), 0.0, 1.0)
        # Every row's CDF shifted by its index: one increasing array for all the rows.
        flat = t.get("flat")
        if flat is None:
            flat = t["flat"] = (cond + np.arange(nv)[:, None]).reshape(-1)
        k = np.searchsorted(flat, u1 + r, side="right") - 1
        c = np.clip(k - r * (nu + 1), 0, nu - 1)
        lo = cond[r, c]
        pc = cond[r, c + 1] - lo
        fc = np.clip((u1 - lo) / np.where(pc > 0, pc, 1.0), 0.0, 1.0)
        u = (c + fc) / nu
        v = 1.0 - (r + fr) / nv
        el = (v - 0.5) * math.pi
        az = (0.5 - u) * 2.0 * math.pi
        ce = np.cos(el)
        e = np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=1)
        cos_r, sin_r = self.cos_sin
        d = np.stack(  # back to the world: R_z(-rotation) e
            [cos_r * e[:, 0] + sin_r * e[:, 1], -sin_r * e[:, 0] + cos_r * e[:, 1], e[:, 2]],
            axis=1,
        )
        pdf_uv = pr * pc * nu * nv
        pdf = pdf_uv / (2.0 * math.pi * math.pi * np.maximum(ce, 1e-12))
        return d, np.where(ce > 1e-9, pdf, 0.0)

    def pdf(self, d: np.ndarray) -> np.ndarray:
        """The density :meth:`sample` draws unit world directions ``d`` with."""
        t = self.tables()
        if t["black"]:
            return np.zeros(d.shape[0])
        nv, nu = t["cells"]
        marg = t["marginal"].astype(np.float64)
        cond = t["conditional"].astype(np.float64)
        u, v = self.uv(d)
        r = np.clip(((1.0 - v) * nv).astype(np.int64), 0, nv - 1)
        c = np.clip((u * nu).astype(np.int64), 0, nu - 1)
        p = (marg[r + 1] - marg[r]) * (cond[r, c + 1] - cond[r, c]) * nu * nv
        ce = np.cos((v - 0.5) * math.pi)
        return np.where(ce > 1e-9, p / (2.0 * math.pi * math.pi * np.maximum(ce, 1e-12)), 0.0)

    # -- the rasterizers' ambient light --------------------------------------------------

    def raster_lighting(self) -> dict[str, Any]:
        """``{"sh": (9, 3), "sun": None | {"direction", "irradiance", "solid_angle"}}``
        (world frame, with the intensity; see the module docstring)."""
        sun = self._sun()
        out = {"sh": self.ambient_sh(), "sun": None}
        if sun is not None:
            c, s = self.cos_sin
            e = sun["direction"]  # image frame -> world: R_z(-rotation)
            d = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1], e[2]])
            out["sun"] = {
                "direction": d,
                "irradiance": sun["irradiance"] * self.intensity,
                "solid_angle": sun["solid_angle"],
            }
        return out

    def _sun(self) -> dict[str, Any] | None:
        """A compact bright source of the image (its own frame), or ``None``; cached."""
        if "sun" in self._tables:
            return self._tables["sun"]
        p = np.asarray(self.pixels, np.float64)
        h, w = p.shape[:2]
        az, el = _directions(w, h)
        domega = np.broadcast_to(np.cos(el) * (2.0 * math.pi / w) * (math.pi / h), (h, w))
        lum = np.maximum(p @ LUMINANCE, 0.0)
        total = float((lum * domega).sum())
        mean = total / (4.0 * math.pi)
        sun = None
        if total > 0.0:
            limit = SUN_CLAMP * mean
            i = int(np.argmax(lum))
            r0, c0 = divmod(i, w)
            ce = np.cos(el)
            dirs = np.stack(np.broadcast_arrays(ce * np.cos(az), ce * np.sin(az), np.sin(el)), -1)
            near = dirs @ dirs[r0, c0] >= math.cos(math.radians(SUN_RADIUS))
            hot = near & (lum > limit)
            excess = np.where(
                hot[..., None], p - p * (limit / np.maximum(lum, 1e-300))[..., None], 0
            )
            energy = float(((excess @ LUMINANCE) * domega).sum())
            if energy >= SUN_SHARE * total:
                weight = (excess @ LUMINANCE) * domega
                centre = (dirs * weight[..., None]).sum(axis=(0, 1))
                centre /= np.linalg.norm(centre)
                sun = {
                    "direction": centre,  # toward the sun (image frame)
                    "irradiance": (excess * domega[..., None]).sum(axis=(0, 1)),  # facing it
                    "solid_angle": float(domega[hot].sum()),
                    "excess": excess.astype(np.float32),
                }
        self._tables["sun"] = sun
        return sun

    def ambient_sh(self) -> np.ndarray:
        """``(9, 3)``: coefficients ``c`` with ``ambient(n) = sum_k c_k Y_k(n)`` (see
        :func:`sh_basis`): irradiance over pi, with the intensity, of the image without its
        environment sun (:meth:`raster_lighting`)."""
        key = ("sh", self.rotation)
        t = self._tables.get(key)
        if t is not None:
            return t * self.intensity
        h, w = self.height, self.width
        az, el = _directions(w, h)
        cos_r, sin_r = self.cos_sin
        coeff = np.zeros((9, 3))
        sun = self._sun()
        for start in range(0, h, 64):  # row blocks: bounded temporaries
            ce = np.cos(el[start : start + 64])
            ex = (ce * np.cos(az)).reshape(-1)
            ey = (ce * np.sin(az)).reshape(-1)
            ez = np.broadcast_to(np.sin(el[start : start + 64]), (ce.shape[0], w)).reshape(-1)
            d = np.stack([cos_r * ex + sin_r * ey, -sin_r * ex + cos_r * ey, ez], axis=1)
            domega = (2.0 * math.pi / w) * (math.pi / h) * np.broadcast_to(ce, (ce.shape[0], w))
            radiance = np.asarray(self.pixels[start : start + 64], np.float64)
            if sun is not None:  # the residual: the image clamped where the sun was taken
                radiance = radiance - sun["excess"][start : start + 64]
            radiance = radiance.reshape(-1, 3)
            coeff += (sh_basis(d) * domega.reshape(-1, 1)).T @ radiance  # projections L_lm
        band = np.array([1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25])
        t = coeff * band[:, None]  # (A_l / pi) L_lm, without the intensity
        rotations = [k for k in self._tables if isinstance(k, tuple) and k[0] == "sh"]
        for old in rotations[: max(0, len(rotations) - 7)]:  # a few rotations at a time
            del self._tables[old]
        self._tables[key] = t
        return t * self.intensity


def sh_basis(d: np.ndarray) -> np.ndarray:
    """``(n, 9)`` real spherical harmonics up to band 2 at unit directions ``d``."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    return np.stack(
        [
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=1,
    )


def sh_ambient(coeff: np.ndarray, n: np.ndarray) -> np.ndarray:
    """``(n, 3)`` ambient radiance at normals ``n`` (clamped at 0: band-2 ringing)."""
    return np.maximum(sh_basis(n) @ coeff, 0.0)


# --------------------------------------------------------------------------- loading --


def check_reference(value: Any, location: str) -> str:
    """An ``image`` reference: ``builtin:NAME`` or a relative path with a known suffix
    that stays inside the project's directory (checked textually here)."""
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        raise _err("INVALID_WORLD", "image must be a file path or builtin:NAME", location)
    if value.startswith("builtin:"):
        if value[8:] not in BUILTINS:
            raise _err(
                "INVALID_WORLD",
                f"no built-in environment {value[8:]!r} (built-ins: {', '.join(BUILTINS)})",
                location,
            )
        return value
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or value.startswith(("/", "\\")) or ":" in value:
        raise _err(
            "INVALID_WORLD",
            "image paths are relative to the project's directory and stay inside it",
            location,
        )
    if p.suffix.lower() not in SUFFIXES:
        raise _err("INVALID_WORLD", f"environment images are {', '.join(SUFFIXES)} files", location)
    return value


def resolve_file(base_dir: Path | None, value: str, location: str) -> Path:
    """The file of a relative ``image`` path, inside ``base_dir`` after resolving links."""
    if base_dir is None:
        raise _err(
            "ENVIRONMENT_MISSING",
            f"the environment image {value!r} is relative to the project's directory, and "
            "the scene has none (save or load the project first)",
            location,
        )
    root = Path(base_dir).resolve()
    path = (root / value).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise _err("ENVIRONMENT_OUTSIDE", "the environment image is outside the project", location)
    if not path.is_file():
        raise _err("ENVIRONMENT_MISSING", f"no environment image at {value!r}", location)
    return path


def load(scene: Any, env: Any) -> EnvironmentImage:
    """The scene's environment image (``env`` its ``world.environment``), cached with the
    scene's derived data by reference, file size and modification time."""
    ref = env.image
    location = "world.environment.image"
    if ref.startswith("builtin:"):
        key = (ref,)
        stat = None
    else:
        path = resolve_file(scene.materials.base_dir, ref, location)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
    derived = scene._derived()
    cache = derived.setdefault("environment_images", {})
    pixels = cache.get(key)
    if pixels is None:
        if stat is None:
            pixels = builtin(ref[8:])
        else:
            if stat.st_size > MAX_FILE_BYTES:
                raise _err("ENVIRONMENT_TOO_LARGE", f"{ref!r} has {stat.st_size:,} bytes", location)
            pixels = _fit(decode(path.read_bytes(), path.suffix.lower(), location))
            pixels.flags.writeable = False
        cache.clear()  # one image at a time
        cache[key] = pixels
        cache[(key, "tables")] = {}
    # Images share their sampling tables (in the image's own frame) and harmonics.
    built = EnvironmentImage.build(pixels, env.rotation, env.intensity, ref)
    object.__setattr__(built, "_tables", cache[(key, "tables")])
    return built
