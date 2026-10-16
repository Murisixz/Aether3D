"""Painted textures: images authored in Aether3D, as tiled layers.

A :class:`PaintedTexture` is an image texture whose pixels come from **layers** instead of
a file. It binds to material slots, samples, renders and exports exactly like an image
texture -- the same sampler settings and the same slot color spaces -- so every renderer
uses it unchanged. What differs is how its pixels are stored and changed.

**Storage.** ``width x height`` texels in 4 channels (RGBA), cut into 64 x 64 **tiles**
of ``uint16`` (0..65535 for 0..1: 8- and 16-bit PNG values are exact, and low-opacity
strokes accumulate without banding). A paint layer keeps only the tiles it has -- a
missing tile is transparent -- so a decal on a 4K texture costs a few tiles. Tiles are
immutable: an edit replaces the tiles it changes, so a document's state is a tuple of
layers that refer to tiles, and the scene history keeps only the tiles a step changed.
Row 0 of a tile grid is the top of the image (as in PNG files), so ``v = 1`` is row 0.

**Layers** (bottom to top, over the ``background``): ``"paint"`` layers (tiles) and
``"fill"`` layers (one RGBA value everywhere), each with an ``opacity``, ``visible`` flag,
a ``blend`` mode and an optional **mask** (single-channel tiles; where the mask has no
tile the layer is fully shown). Compositing follows the W3C *Compositing and Blending*
formula: Porter-Duff source-over with a separable blend function ``B``::

    a_s = alpha * opacity * mask
    C_s' = (1 - a_b) C_s + a_b B(C_b, C_s)
    a_o = a_s + a_b (1 - a_s),   C_o = (a_s C_s' + a_b (1 - a_s) C_b) / a_o

with ``B`` one of ``normal``, ``multiply``, ``screen``, ``overlay``, ``darken``,
``lighten``, ``difference``, ``add`` (clamped) and ``subtract`` (clamped), applied to the
values as stored, in float32.

**Encodings.** ``"color"`` documents store display (sRGB-encoded) colors, like 8-bit color
images; ``"data"`` documents store linear values (roughness, metallic, masks, AO);
``"normal"`` documents store tangent-space unit vectors (``n * 0.5 + 0.5``) and
renormalize the composite's vectors. A render still decodes the texture by the material
slot it is bound to (``color_space="auto"``), as for any image.

**Rendering.** The composite is quantized to 8-bit RGBA (``round(v * 255)``, the rounding
of a 16-bit PNG read as 8 bits), the levels every renderer samples. After an edit only the
changed tiles are recomposited and requantized, and only the matching regions of every
mip level are rebuilt (with the same arithmetic as a full rebuild, so the levels are
always exactly those of :func:`~aether3d.materials.sampling.build_mip_chain`). Each change
gets a new revision; renders see the texture as an image whose digest names the document
and the revision, and the GPU's texel pool receives only the rows that changed.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..errors import MaterialError
from .images import DecodedImage, ImageLimits
from .sampling import MODES, MipChain, _downsample, level_count, mip_bytes
from .textures import ImageTexture, LoadedImage

TILE = 64
MAX_LAYERS = 64
FULL = 65535
BLEND_MODES = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "difference",
    "add",
    "subtract",
)
LAYER_KINDS = ("paint", "fill")
ENCODINGS = ("color", "data", "normal")
LOG_REVISIONS = 64
"""How many revisions back the record of changed tiles reaches (for incremental uploads)."""


def _err(code: str, message: str, location: str | None = None, **kw) -> MaterialError:
    return MaterialError(code, message, location, **kw)


@dataclass(frozen=True, eq=False)
class Layer:
    """One layer of a painted texture (immutable; edits make new layers)."""

    id: str
    name: str
    kind: str = "paint"
    blend: str = "normal"
    opacity: float = 1.0
    visible: bool = True
    value: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # fill layers
    tiles: Any = None  # dict[int, (TILE, TILE, 4) uint16] for paint layers
    mask: Any = None  # dict[int, (TILE, TILE) uint16], or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "blend": self.blend,
            "opacity": self.opacity,
            "visible": self.visible,
            **({"value": list(self.value)} if self.kind == "fill" else {}),
            "tiles": 0 if self.tiles is None else len(self.tiles),
            "masked": self.mask is not None,
        }


def frozen_tile(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.uint16)
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- #
# Compositing (float32, per tile)
# --------------------------------------------------------------------------- #


def blend(mode: str, cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """The separable blend function ``B(C_b, C_s)`` (W3C Compositing and Blending)."""
    if mode == "normal":
        return cs
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "overlay":  # hard light with the layers swapped
        return np.where(cb <= 0.5, 2.0 * cb * cs, 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)).astype(
            np.float32
        )
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "add":
        return np.minimum(cb + cs, 1.0)
    if mode == "subtract":
        return np.maximum(cb - cs, 0.0)
    raise ValueError(mode)


def composite_over(
    base: np.ndarray, src_rgb: np.ndarray, src_alpha: np.ndarray, mode: str
) -> np.ndarray:
    """``src`` (premultiplied by nothing: straight color, coverage ``src_alpha``) over
    ``base`` (straight RGBA float32), in place on a copy; see the module docstring."""
    cb = base[..., :3]
    ab = base[..., 3:4]
    as_ = src_alpha[..., None]
    mixed = (1.0 - ab) * src_rgb + ab * blend(mode, cb, src_rgb)
    ao = as_ + ab * (1.0 - as_)
    num = as_ * mixed + ab * (1.0 - as_) * cb
    out = np.empty_like(base)
    safe = np.where(ao > 0.0, ao, 1.0)
    out[..., :3] = np.where(ao > 0.0, num / safe, 0.0)
    out[..., 3:4] = ao
    return out


class PaintedTexture(ImageTexture):
    """An image texture made of tiled layers (see the module docstring)."""

    kind = "image"  # binds, samples and exports as an image texture
    painted = True
    __slots__ = (
        "_width",
        "_height",
        "_encoding",
        "_background",
        "_layers",
        "_rev",
        "_log",
        "_cache",
    )

    def __init__(
        self,
        library: Any,
        texture_id: str,
        name: str,
        *,
        width: int,
        height: int,
        encoding: str = "color",
        background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        layers: tuple[Layer, ...] = (),
        wrap: tuple[str, str] = ("repeat", "repeat"),
        filter: str = "linear",
        mipmaps: bool = True,
        color_space: str = "auto",
    ) -> None:
        super().__init__(
            library,
            texture_id,
            name,
            wrap=wrap,
            filter=filter,
            mipmaps=mipmaps,
            color_space=color_space,
        )
        self._width = int(width)
        self._height = int(height)
        self._encoding = encoding
        self._background = tuple(float(v) for v in background)
        self._layers = tuple(layers)
        self._rev = 0
        self._log: deque[tuple[int, frozenset[int] | None]] = deque(maxlen=LOG_REVISIONS)
        self._cache = _Cache()

    # -- geometry of the tile grid --------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def tiles_x(self) -> int:
        return -(-self._width // TILE)

    @property
    def tiles_y(self) -> int:
        return -(-self._height // TILE)

    @property
    def tile_count(self) -> int:
        return self.tiles_x * self.tiles_y

    def tile_rect(self, index: int) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` texel bounds of a tile (row 0: the top of the image)."""
        ty, tx = divmod(index, self.tiles_x)
        x0, y0 = tx * TILE, ty * TILE
        return x0, y0, min(x0 + TILE, self._width), min(y0 + TILE, self._height)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def background(self) -> tuple[float, float, float, float]:
        return self._background

    @property
    def layers(self) -> tuple[Layer, ...]:
        return self._layers

    @property
    def revision(self) -> int:
        """Increases with every change of the pixels (undo and redo included)."""
        return self._rev

    @property
    def source(self) -> str:
        return "painted"

    @property
    def file(self):  # noqa: D102 - painted textures have no file
        return None

    @property
    def path(self):  # noqa: D102
        return None

    @property
    def mime_type(self) -> str | None:
        return None

    def _read_bytes(self) -> bytes:
        """The composite as an 8-bit PNG file (what glTF export embeds)."""
        from ..rendering.png import encode_png

        cache = self._cache
        if cache.png is None or cache.png[0] != self._rev:
            cache.png = (self._rev, encode_png(self.pixels8(), srgb=self._encoding == "color"))
        return cache.png[1]

    def export_png(self, path: Any, *, bits: int = 8) -> Any:
        """Write the composite as an 8- or 16-bit PNG file (atomically); returns the path."""
        from pathlib import Path

        from .._fileio import write_atomic
        from ..rendering.png import encode_png

        if bits not in (8, 16):
            raise _err("INVALID_ARGUMENT", "bits must be 8 or 16", self.location)
        pixels = self.pixels8() if bits == 8 else self.pixels16()
        target = Path(path).absolute()
        if target.suffix.lower() != ".png":
            raise _err("INVALID_OUTPUT", "painted textures are written as .png files", "path")
        write_atomic(target, encode_png(pixels, srgb=self._encoding == "color"))
        return target

    def embed(self) -> None:
        raise _err(
            "INVALID_TEXTURE",
            f"texture {self._name!r} is painted: its layers are already part of the project",
            self.location,
        )

    def extract(self, path):  # noqa: D102
        raise _err(
            "INVALID_TEXTURE",
            f"texture {self._name!r} is painted; use export_png() to write its image",
            self.location,
        )

    # -- layer lookup ----------------------------------------------------------------

    def layer(self, ref: Any) -> Layer:
        """A layer by id, name or index."""
        if isinstance(ref, Layer):
            ref = ref.id
        if isinstance(ref, int) and not isinstance(ref, bool):
            if not -len(self._layers) <= ref < len(self._layers):
                raise _err("LAYER_NOT_FOUND", f"no layer {ref}", self.location)
            return self._layers[ref]
        for layer in self._layers:
            if layer.id == ref or layer.name == ref:
                return layer
        raise _err("LAYER_NOT_FOUND", f"no layer {ref!r}", self.location)

    def _layer_index(self, ref: Any) -> int:
        target = self.layer(ref)
        return next(i for i, layer in enumerate(self._layers) if layer is target)

    # -- changes (each one a step of the scene's history) -------------------------------

    def _set_layers(self, layers: tuple[Layer, ...], label: str, changed: Any) -> None:
        """Replace the layers (one history step); ``changed`` is a set of tile indices or
        ``None`` (everything)."""
        with self._edit(label):
            self._layers = layers
            self._mark(changed)
            self._touch()

    def _mark(self, changed: Any) -> None:
        self._rev += 1
        self._log.append((self._rev, None if changed is None else frozenset(changed)))

    def add_layer(
        self,
        name: str | None = None,
        *,
        kind: str = "paint",
        blend: str = "normal",
        opacity: float = 1.0,
        visible: bool = True,
        value: Any = None,
        index: int | None = None,
    ) -> Layer:
        """Add a layer (on top, or at ``index``) and return it."""
        if len(self._layers) >= MAX_LAYERS:
            raise _err("RESOURCE_LIMIT", f"a painted texture has at most {MAX_LAYERS} layers")
        if name is None:
            taken = {layer.name for layer in self._layers}
            n = len(self._layers) + 1
            while f"Layer {n}" in taken:
                n += 1
            name = f"Layer {n}"
        layer = check_layer(
            Layer(
                uuid.uuid4().hex,
                name,
                kind,
                blend,
                opacity,
                visible,
                _rgba(value if value is not None else (0, 0, 0, 0), self.location),
                {} if kind == "paint" else None,
            ),
            self.location,
        )
        if any(other.name == layer.name for other in self._layers):
            raise _err("NAME_CONFLICT", f"a layer named {layer.name!r} exists", self.location)
        layers = list(self._layers)
        at = len(layers) if index is None else _index(index, len(layers) + 1, self.location)
        layers.insert(at, layer)
        changed = None if kind == "fill" or layer.tiles else set()
        self._set_layers(tuple(layers), f"Add layer {layer.name!r} to {self._name!r}", changed)
        return layer

    def remove_layer(self, ref: Any) -> None:
        i = self._layer_index(ref)
        layer = self._layers[i]
        changed = self._layer_footprint(layer)
        self._set_layers(
            self._layers[:i] + self._layers[i + 1 :],
            f"Remove layer {layer.name!r} from {self._name!r}",
            changed,
        )

    def move_layer(self, ref: Any, index: int) -> None:
        i = self._layer_index(ref)
        layers = list(self._layers)
        layer = layers.pop(i)
        layers.insert(_index(index, len(self._layers), self.location), layer)
        self._set_layers(
            tuple(layers),
            f"Move layer {layer.name!r} of {self._name!r}",
            self._layer_footprint(layer),
        )

    def set_layer(self, ref: Any, **changes: Any) -> Layer:
        """Change a layer's ``name``, ``blend``, ``opacity``, ``visible`` or ``value``."""
        i = self._layer_index(ref)
        old = self._layers[i]
        allowed = {"name", "blend", "opacity", "visible", "value"}
        unknown = set(changes) - allowed
        if unknown:
            raise _err(
                "INVALID_ARGUMENT",
                f"unknown layer setting(s): {', '.join(sorted(unknown))}",
                self.location,
            )
        if "value" in changes:
            changes["value"] = _rgba(changes["value"], self.location)
        new = check_layer(replace(old, **changes), self.location)
        if new.name != old.name and any(o.name == new.name for o in self._layers):
            raise _err("NAME_CONFLICT", f"a layer named {new.name!r} exists", self.location)
        pixels = any(k in changes for k in ("blend", "opacity", "visible", "value"))
        layers = self._layers[:i] + (new,) + self._layers[i + 1 :]
        self._set_layers(
            layers,
            f"Set layer {old.name!r} of {self._name!r}",
            self._layer_footprint(old) if pixels else set(),
        )
        return new

    def clear_layer(self, ref: Any) -> None:
        i = self._layer_index(ref)
        old = self._layers[i]
        if old.kind != "paint":
            raise _err("INVALID_ARGUMENT", "only paint layers hold pixels", self.location)
        changed = set(old.tiles)
        layers = self._layers[:i] + (replace(old, tiles={}),) + self._layers[i + 1 :]
        self._set_layers(layers, f"Clear layer {old.name!r} of {self._name!r}", changed)

    def set_mask(self, ref: Any, mask: Any) -> None:
        """Set a layer's mask from an ``(height, width)`` array of 0..1 (``None`` removes
        it)."""
        i = self._layer_index(ref)
        old = self._layers[i]
        if mask is None:
            new_mask = None
        else:
            values = np.asarray(mask, dtype=np.float64)
            if values.shape != (self._height, self._width) or not np.all(np.isfinite(values)):
                raise _err(
                    "INVALID_ARGUMENT",
                    f"a mask must be a finite ({self._height}, {self._width}) array",
                    self.location,
                )
            new_mask = {}
            for t in range(self.tile_count):
                x0, y0, x1, y1 = self.tile_rect(t)
                tile = np.full((TILE, TILE), FULL, dtype=np.uint16)
                tile[: y1 - y0, : x1 - x0] = np.round(np.clip(values[y0:y1, x0:x1], 0, 1) * FULL)
                if not np.all(tile == FULL):
                    new_mask[t] = frozen_tile(tile)
        layers = self._layers[:i] + (replace(old, mask=new_mask),) + self._layers[i + 1 :]
        self._set_layers(layers, f"Set mask of layer {old.name!r} of {self._name!r}", None)

    def write_pixels(self, ref: Any, pixels: Any, x: int = 0, y: int = 0) -> None:
        """Write ``(h, w, 4)`` (or ``(h, w, 3)``: opaque) values in 0..1 into a paint layer
        at texel ``(x, y)`` (``y`` counted from the top row), replacing what was there."""
        values = np.asarray(pixels, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] not in (3, 4) or not np.all(np.isfinite(values)):
            raise _err(
                "INVALID_ARGUMENT", "pixels must be a finite (h, w, 3|4) array", self.location
            )
        if values.shape[2] == 3:
            values = np.concatenate([values, np.ones(values.shape[:2] + (1,))], axis=2)
        h, w = values.shape[:2]
        if x < 0 or y < 0 or x + w > self._width or y + h > self._height:
            raise _err(
                "INVALID_ARGUMENT", "the pixels do not fit inside the texture", self.location
            )
        q = np.round(np.clip(values, 0.0, 1.0) * FULL).astype(np.uint16)
        i = self._layer_index(ref)
        old = self._layers[i]
        if old.kind != "paint":
            raise _err("INVALID_ARGUMENT", "only paint layers hold pixels", self.location)
        tiles = dict(old.tiles)
        changed = set()
        for t in self.tiles_in(x, y, x + w, y + h):
            tx0, ty0, tx1, ty1 = self.tile_rect(t)
            ax0, ay0 = max(tx0, x), max(ty0, y)
            ax1, ay1 = min(tx1, x + w), min(ty1, y + h)
            tile = tiles.get(t)
            tile = np.zeros((TILE, TILE, 4), np.uint16) if tile is None else tile.copy()
            tile[ay0 - ty0 : ay1 - ty0, ax0 - tx0 : ax1 - tx0] = q[
                ay0 - y : ay1 - y, ax0 - x : ax1 - x
            ]
            tiles[t] = frozen_tile(tile)
            changed.add(t)
        layers = self._layers[:i] + (replace(old, tiles=tiles),) + self._layers[i + 1 :]
        self._set_layers(layers, f"Write pixels of layer {old.name!r} of {self._name!r}", changed)

    def replace_tiles(self, edits: dict[str, dict[int, np.ndarray | None]], label: str) -> None:
        """Replace tiles of several layers at once (one history step): ``{layer id:
        {tile index: (TILE, TILE, 4) uint16 or None (remove)}}``; used by painting."""
        layers = list(self._layers)
        changed: set[int] = set()
        for ref, tiles in edits.items():
            i = self._layer_index(ref)
            old = layers[i]
            if old.kind != "paint":
                raise _err("INVALID_ARGUMENT", "only paint layers hold pixels", self.location)
            new = dict(old.tiles)
            for t, tile in tiles.items():
                if not 0 <= t < self.tile_count:
                    raise _err(
                        "INVALID_ARGUMENT", f"tile {t} is outside the texture", self.location
                    )
                if tile is None:
                    new.pop(t, None)
                else:
                    new[t] = frozen_tile(tile)
                changed.add(t)
            layers[i] = replace(old, tiles=new)
        self._set_layers(tuple(layers), label, changed)

    def replace_masks(self, edits: dict[str, dict[int, np.ndarray | None]], label: str) -> None:
        """Replace mask tiles (``(TILE, TILE)`` uint16, ``None``: fully shown) of layers."""
        layers = list(self._layers)
        changed: set[int] = set()
        for ref, tiles in edits.items():
            i = self._layer_index(ref)
            old = layers[i]
            new = dict(old.mask or {})
            for t, tile in tiles.items():
                if tile is None:
                    new.pop(t, None)
                else:
                    new[t] = frozen_tile(tile)
                changed.add(t)
            layers[i] = replace(old, mask=new)
        self._set_layers(tuple(layers), label, changed)

    def flatten(self) -> None:
        """Merge every visible layer into one paint layer (the composite)."""
        tiles = {}
        for t in range(self.tile_count):
            tiles[t] = frozen_tile(self.composite_tile(t))
        layer = Layer(uuid.uuid4().hex, "Flattened", "paint", tiles=tiles)
        self._set_layers((layer,), f"Flatten {self._name!r}", None)
        self._background_set((0.0, 0.0, 0.0, 0.0))

    def _background_set(self, value: tuple) -> None:
        with self._edit(f"Set background of {self._name!r}"):
            self._background = tuple(float(v) for v in value)
            self._mark(None)
            self._touch()

    def set_background(self, value: Any) -> None:
        """The RGBA (0..1, stored encoding) under every layer."""
        self._background_set(_rgba(value, self.location))

    def _layer_footprint(self, layer: Layer) -> set[int] | None:
        if layer.kind == "fill" or layer.mask is not None and layer.kind == "fill":
            return None
        return set(layer.tiles or ())

    def tiles_in(self, x0: int, y0: int, x1: int, y1: int) -> list[int]:
        """Indices of the tiles meeting the texel rectangle ``[x0, x1) x [y0, y1)``."""
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self._width, x1), min(self._height, y1)
        if x1 <= x0 or y1 <= y0:
            return []
        tx = range(x0 // TILE, (x1 - 1) // TILE + 1)
        ty = range(y0 // TILE, (y1 - 1) // TILE + 1)
        return [j * self.tiles_x + i for j in ty for i in tx]

    # -- reading ---------------------------------------------------------------------

    def composite_tile(self, index: int) -> np.ndarray:
        """The composite of one tile as ``(TILE, TILE, 4)`` uint16 (cached per revision)."""
        cache = self._cache
        if cache.rev_composite != self._rev:
            self._sync_cache()
        found = cache.composite.get(index)
        if found is not None:
            return found
        out = self._composite(index)
        cache.composite[index] = out
        return out

    def _composite(self, index: int) -> np.ndarray:
        acc = np.empty((TILE, TILE, 4), dtype=np.float32)
        acc[:] = np.asarray(self._background, dtype=np.float32)
        for layer in self._layers:
            if not layer.visible or layer.opacity <= 0.0:
                continue
            if layer.kind == "paint":
                tile = layer.tiles.get(index)
                if tile is None:
                    continue
                src = tile.astype(np.float32) * np.float32(1.0 / FULL)
                rgb, alpha = src[..., :3], src[..., 3]
            else:
                rgb = np.broadcast_to(np.asarray(layer.value[:3], np.float32), (TILE, TILE, 3))
                alpha = np.full((TILE, TILE), np.float32(layer.value[3]))
            alpha = alpha * np.float32(layer.opacity)
            if layer.mask is not None:
                m = layer.mask.get(index)
                if m is not None:
                    alpha = alpha * (m.astype(np.float32) * np.float32(1.0 / FULL))
            acc = composite_over(acc, rgb, alpha, layer.blend)
        if self._encoding == "normal":
            v = acc[..., :3] * 2.0 - 1.0
            length = np.sqrt((v * v).sum(axis=2, keepdims=True))
            unit = np.where(
                length > 1e-12, v / np.maximum(length, 1e-30), np.array([0, 0, 1], np.float32)
            )
            acc[..., :3] = unit * 0.5 + 0.5
        out = np.round(np.clip(acc, 0.0, 1.0) * FULL).astype(np.uint16)
        out.flags.writeable = False
        return out

    def read(
        self, x0: int = 0, y0: int = 0, x1: int | None = None, y1: int | None = None
    ) -> np.ndarray:
        """The composite of a texel rectangle as ``(h, w, 4)`` float64 in 0..1."""
        x1 = self._width if x1 is None else x1
        y1 = self._height if y1 is None else y1
        out = np.zeros((y1 - y0, x1 - x0, 4))
        for t in self.tiles_in(x0, y0, x1, y1):
            tx0, ty0, tx1, ty1 = self.tile_rect(t)
            ax0, ay0, ax1, ay1 = max(tx0, x0), max(ty0, y0), min(tx1, x1), min(ty1, y1)
            tile = self.composite_tile(t)
            out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = (
                tile[ay0 - ty0 : ay1 - ty0, ax0 - tx0 : ax1 - tx0] / FULL
            )
        return out

    def pixels16(self) -> np.ndarray:
        """The whole composite as ``(height, width, 4)`` uint16 (row 0: the top)."""
        out = np.empty((self._height, self._width, 4), dtype=np.uint16)
        for t in range(self.tile_count):
            x0, y0, x1, y1 = self.tile_rect(t)
            out[y0:y1, x0:x1] = self.composite_tile(t)[: y1 - y0, : x1 - x0]
        return out

    def pixels8(self) -> np.ndarray:
        """The 8-bit composite renders sample (``(height, width, 4)`` uint8, shared, read-only)."""
        self._sync_cache()
        return _readonly_view(self._cache.level0)

    # -- the render cache ------------------------------------------------------------

    def _changed_since(self, rev: int) -> set[int] | None:
        """Tiles changed after revision ``rev`` (``None``: unknown, everything)."""
        if rev == self._rev:
            return set()
        if not self._log or self._log[0][0] > rev + 1:
            return None
        out: set[int] = set()
        for r, tiles in self._log:
            if r <= rev:
                continue
            if tiles is None:
                return None
            out |= tiles
        return out

    def _sync_cache(self) -> None:
        cache = self._cache
        if cache.rev_composite == self._rev and cache.level0 is not None:
            return
        changed = None if cache.level0 is None else self._changed_since(cache.rev_composite)
        if (
            cache.level0 is None
            or changed is None
            or cache.level0.shape[:2] != (self._height, self._width)
        ):
            cache.level0 = np.empty((self._height, self._width, 4), dtype=np.uint8)
            changed = set(range(self.tile_count))
            cache.chains.clear()
            cache.composite.clear()
        for t in changed:
            cache.composite.pop(t, None)
        cache.rev_composite = self._rev
        for t in sorted(changed):
            tile = self.composite_tile(t)
            x0, y0, x1, y1 = self.tile_rect(t)
            q = ((tile[: y1 - y0, : x1 - x0].astype(np.uint32) * 255 + 32767) // 65535).astype(
                np.uint8
            )
            cache.level0[y0:y1, x0:x1] = q
        for chain in cache.chains.values():
            chain.pending.append(changed)

    def mip_chain(self, mode: str) -> tuple[MipChain, LoadedImage]:
        """The 8-bit levels for ``mode`` (updated in place where tiles changed)."""
        if mode not in MODES:
            raise ValueError(f"unknown texture mode {mode!r}")
        self._sync_cache()
        cache = self._cache
        key = (mode, self._mipmaps)
        state = cache.chains.get(key)
        if state is None:
            state = cache.chains[key] = _ChainState(_full_levels(cache.level0, mode, self._mipmaps))
        elif state.pending:
            rects = _merge_pending(self, state.pending)
            state.pending.clear()
            if rects is None:
                state.levels = _full_levels(cache.level0, mode, self._mipmaps)
            else:
                _update_levels(state.levels, rects, mode)
            state.rev = self._rev
        loaded = self.load()
        return MipChain(tuple(_readonly_view(level) for level in state.levels), mode), loaded

    def load(self) -> LoadedImage:
        self._sync_cache()
        image = DecodedImage(_readonly_view(self._cache.level0), "png", 16, 4, True)
        return LoadedImage(image, self.digest, "painted", 0)

    @property
    def digest(self) -> str:
        """A key naming this document and revision (unique within the process)."""
        return f"painted:{self._id}:{id(self._cache)}:{self._rev}"

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "format": "painted",
            "width": self._width,
            "height": self._height,
            "layers": len(self._layers),
            "tiles": sum(len(layer.tiles or ()) for layer in self._layers),
            "revision": self._rev,
            "digest": self.digest,
            "texture_bytes": mip_bytes(self._width, self._height, self._mipmaps),
        }

    def info(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "type": "painted",
            "width": self._width,
            "height": self._height,
            "encoding": self._encoding,
            "background": list(self._background),
            "layers": [layer.to_dict() for layer in self._layers],
            "wrap": list(self._wrap),
            "filter": self._filter,
            "mipmaps": self._mipmaps,
            "color_space": self._color_space,
            "status": self.status(),
        }

    # -- records and history ------------------------------------------------------------

    def to_dict(self, project_dir: Any = None) -> dict[str, Any]:
        """The project record: layers stored as embedded 16-bit PNG images."""
        import base64

        def encode(png: bytes) -> dict[str, str]:
            return {"mime_type": "image/png", "data": base64.b64encode(png).decode("ascii")}

        return {
            "id": self._id,
            "name": self._name,
            "type": "painted",
            "width": self._width,
            "height": self._height,
            "encoding": self._encoding,
            "background": list(self._background),
            "layers": [layer_record(self, layer, encode) for layer in self._layers],
            "wrap": list(self._wrap),
            "filter": self._filter,
            "mipmaps": self._mipmaps,
            "color_space": self._color_space,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        """Apply a history state (undo/redo): the layers, background and sampler; the tiles
        that differ are marked changed."""
        old_layers, old_bg = self._layers, self._background
        for key, value in state.items():
            object.__setattr__(self, key, value)
        if old_bg != self._background or len(old_layers) != len(self._layers):
            self._mark(None)
            return
        changed: set[int] = set()
        for a, b in zip(old_layers, self._layers, strict=True):
            if a is b:
                continue
            same_look = (a.id, a.kind, a.blend, a.opacity, a.visible, a.value) == (
                b.id,
                b.kind,
                b.blend,
                b.opacity,
                b.visible,
                b.value,
            )
            if not same_look or (a.mask is None) != (b.mask is None):
                if a.kind == "fill" or b.kind == "fill":
                    self._mark(None)
                    return
                changed |= set(a.tiles or ()) | set(b.tiles or ())
            for x, y in ((a.tiles or {}, b.tiles or {}), (a.mask or {}, b.mask or {})):
                if x is y:
                    continue
                for t in set(x) | set(y):
                    if x.get(t) is not y.get(t):
                        changed.add(t)
        self._mark(changed)


class _Cache:
    __slots__ = ("composite", "rev_composite", "level0", "chains", "png")

    def __init__(self) -> None:
        self.png: tuple[int, bytes] | None = None
        self.composite: dict[int, np.ndarray] = {}
        self.rev_composite = -1
        self.level0: np.ndarray | None = None
        self.chains: dict[tuple, _ChainState] = {}


class _ChainState:
    __slots__ = ("levels", "pending", "rev")

    def __init__(self, levels: list[np.ndarray]) -> None:
        self.levels = levels
        self.pending: list[set[int] | None] = []
        self.rev = 0


def _readonly_view(array: np.ndarray) -> np.ndarray:
    """A read-only view of a level the document keeps updating (callers cannot change it,
    nor freeze the document's own array)."""
    view = array.view()
    view.flags.writeable = False
    return view


def _full_levels(level0: np.ndarray, mode: str, mipmaps: bool) -> list[np.ndarray]:
    levels = [level0]
    if mipmaps:
        for _ in range(level_count(level0.shape[1], level0.shape[0]) - 1):
            levels.append(_downsample(levels[-1], mode))
    return levels


def _merge_pending(doc: PaintedTexture, pending: list) -> list[tuple[int, int, int, int]] | None:
    tiles: set[int] = set()
    for p in pending:
        if p is None:
            return None
        tiles |= p
    return [doc.tile_rect(t) for t in sorted(tiles)]


def _update_levels(levels: list[np.ndarray], rects, mode: str) -> None:
    """Rebuild the regions of levels 1.. above the level-0 rectangles ``rects`` with
    ``_downsample``'s arithmetic (exactly what a full rebuild gives)."""
    for k in range(1, len(levels)):
        src = levels[k - 1]
        dst = levels[k]
        h, w = src.shape[:2]
        nh, nw = dst.shape[:2]
        new_rects = []
        for x0, y0, x1, y1 in rects:
            r0, r1 = y0 // 2, min(nh, -(-y1 // 2))
            c0, c1 = x0 // 2, min(nw, -(-x1 // 2))
            if r1 <= r0 or c1 <= c0:
                continue
            ys = np.minimum(np.arange(2 * r0, 2 * r1), h - 1)
            xs = np.minimum(np.arange(2 * c0, 2 * c1), w - 1)
            block = src[ys][:, xs]
            dst[r0:r1, c0:c1] = _downsample_block(block, mode)
            new_rects.append((c0, r0, c1, r1))
        rects = new_rects


def _downsample_block(block: np.ndarray, mode: str) -> np.ndarray:
    """``_downsample`` of an even-sized block (the same arithmetic)."""
    from ..rendering.color import linear_to_srgb, quantize
    from .sampling import SRGB_TABLE

    nh, nw = block.shape[0] // 2, block.shape[1] // 2
    if mode == "srgb":
        rgb = SRGB_TABLE[block[..., :3]]
    elif mode == "normal":
        rgb = block[..., :3] * (2.0 / 255.0) - 1.0
    else:
        rgb = block[..., :3] / 255.0
    alpha = block[..., 3] / 255.0
    rgb = rgb.reshape(nh, 2, nw, 2, 3).mean(axis=(1, 3))
    alpha = alpha.reshape(nh, 2, nw, 2).mean(axis=(1, 3))
    out = np.empty((nh, nw, 4), dtype=np.uint8)
    if mode == "srgb":
        out[..., :3] = quantize(linear_to_srgb(rgb))
    elif mode == "normal":
        length = np.sqrt((rgb * rgb).sum(axis=-1, keepdims=True))
        unit = np.divide(rgb, length, out=np.zeros_like(rgb), where=length > 1e-12)
        unit[..., 2] = np.where(length[..., 0] > 1e-12, unit[..., 2], 1.0)
        out[..., :3] = quantize(unit * 0.5 + 0.5)
    else:
        out[..., :3] = quantize(rgb)
    out[..., 3] = quantize(alpha)
    return out


def _rgba(value: Any, location: str) -> tuple[float, float, float, float]:
    try:
        v = [float(x) for x in value]
    except (TypeError, ValueError):
        raise _err("INVALID_ARGUMENT", "a value must be 1-4 numbers in [0, 1]", location) from None
    if not 1 <= len(v) <= 4 or not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in v):
        raise _err("INVALID_ARGUMENT", "a value must be 1-4 numbers in [0, 1]", location)
    if len(v) == 1:
        v = [v[0], v[0], v[0], 1.0]
    elif len(v) == 2:
        v = [v[0], v[0], v[0], v[1]]
    elif len(v) == 3:
        v = v + [1.0]
    return (v[0], v[1], v[2], v[3])


def _index(value: Any, size: int, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < max(size, 1):
        raise _err("INVALID_ARGUMENT", f"index must be in [0, {size - 1}]", location)
    return value


def check_layer(layer: Layer, location: str) -> Layer:
    if not isinstance(layer.name, str) or not 1 <= len(layer.name) <= 128:
        raise _err("INVALID_ARGUMENT", "a layer name must have 1-128 characters", location)
    if layer.kind not in LAYER_KINDS:
        raise _err(
            "INVALID_ARGUMENT", f"layer kind must be one of {', '.join(LAYER_KINDS)}", location
        )
    if layer.blend not in BLEND_MODES:
        raise _err("INVALID_ARGUMENT", f"blend must be one of {', '.join(BLEND_MODES)}", location)
    if (
        isinstance(layer.opacity, bool)
        or not isinstance(layer.opacity, (int, float))
        or not (math.isfinite(layer.opacity) and 0.0 <= layer.opacity <= 1.0)
    ):
        raise _err("INVALID_ARGUMENT", "opacity must be in [0, 1]", location)
    if not isinstance(layer.visible, bool):
        raise _err("INVALID_ARGUMENT", "visible must be true or false", location)
    return replace(layer, opacity=float(layer.opacity))


def tile_digest(tile: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tile).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Records (project files and history files)
# --------------------------------------------------------------------------- #


def layer_pixels(doc: PaintedTexture, layer: Layer) -> np.ndarray:
    """A paint layer's pixels as ``(height, width, 4)`` uint16 (zeros where it has no tile)."""
    out = np.zeros((doc.height, doc.width, 4), dtype=np.uint16)
    for t, tile in (layer.tiles or {}).items():
        x0, y0, x1, y1 = doc.tile_rect(t)
        out[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
    return out


def mask_pixels(doc: PaintedTexture, layer: Layer) -> np.ndarray:
    out = np.full((doc.height, doc.width), FULL, dtype=np.uint16)
    for t, tile in (layer.mask or {}).items():
        x0, y0, x1, y1 = doc.tile_rect(t)
        out[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
    return out


def tiles_from_pixels(doc_size: tuple[int, int], pixels: np.ndarray, empty: Any) -> dict:
    """Tiles of an ``(h, w, ...)`` uint16 image, without the tiles equal to ``empty``."""
    w, h = doc_size
    tx = -(-w // TILE)
    ty = -(-h // TILE)
    tiles = {}
    shape = (TILE, TILE) + pixels.shape[2:]
    for j in range(ty):
        for i in range(tx):
            x0, y0 = i * TILE, j * TILE
            x1, y1 = min(x0 + TILE, w), min(y0 + TILE, h)
            block = pixels[y0:y1, x0:x1]
            if np.all(block == empty):
                continue
            tile = np.empty(shape, dtype=np.uint16)
            tile[...] = empty
            tile[: y1 - y0, : x1 - x0] = block
            tiles[j * tx + i] = frozen_tile(tile)
    return tiles


def layer_record(doc: PaintedTexture, layer: Layer, encode) -> dict[str, Any]:
    """A layer's record; ``encode(png bytes) -> JSON value`` stores pixel images."""
    from ..rendering.png import encode_png

    record: dict[str, Any] = {
        "id": layer.id,
        "name": layer.name,
        "kind": layer.kind,
        "blend": layer.blend,
        "opacity": layer.opacity,
        "visible": layer.visible,
    }
    if layer.kind == "fill":
        record["value"] = list(layer.value)
    else:
        record["pixels"] = (
            encode(encode_png(layer_pixels(doc, layer), compression=6, srgb=False))
            if layer.tiles
            else None
        )
    record["mask"] = (
        encode(encode_png(mask_pixels(doc, layer), compression=6, srgb=False))
        if layer.mask is not None
        else None
    )
    return record


def layer_from_record(size: tuple[int, int], raw: Any, decode, location: str) -> Layer:
    """The layer of a record (strict); ``decode(JSON value) -> png bytes``."""
    from .images import decode_png16

    if not isinstance(raw, dict):
        raise _err("INVALID_TEXTURE", "a layer must be an object", location)
    allowed = {"id", "name", "kind", "blend", "opacity", "visible", "value", "pixels", "mask"}
    unknown = sorted(set(raw) - allowed)
    missing = sorted({"id", "name", "kind"} - set(raw))
    if unknown or missing:
        what = f"unknown {', '.join(unknown)}" if unknown else f"missing {', '.join(missing)}"
        raise _err("INVALID_TEXTURE", f"layer keys: {what}", location)
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise _err("INVALID_TEXTURE", "a layer id must be a non-empty string", location)
    kind = raw["kind"]
    if kind not in LAYER_KINDS:
        raise _err(
            "INVALID_TEXTURE", f"layer kind must be one of {', '.join(LAYER_KINDS)}", location
        )
    w, h = size
    limits = ImageLimits(max_dimension=16384, max_pixels=max(w * h, 1))

    def image(value: Any, what: str) -> np.ndarray | None:
        if value is None:
            return None
        pixels = decode_png16(decode(value), limits=limits, location=f"{location}.{what}")
        if pixels.shape[:2] != (h, w):
            raise _err(
                "INVALID_TEXTURE",
                f"the {what} image is {pixels.shape[1]}x{pixels.shape[0]}, the texture {w}x{h}",
                location,
            )
        return pixels

    tiles = None
    value = (0.0, 0.0, 0.0, 0.0)
    if kind == "paint":
        if "value" in raw:
            raise _err("INVALID_TEXTURE", "a paint layer has pixels, not a value", location)
        pixels = image(raw.get("pixels"), "pixels")
        tiles = {} if pixels is None else tiles_from_pixels(size, pixels, 0)
    else:
        if raw.get("pixels") is not None:
            raise _err("INVALID_TEXTURE", "a fill layer has a value, not pixels", location)
        value = _rgba(raw.get("value", (0, 0, 0, 0)), location)
    mask_image = image(raw.get("mask"), "mask")
    mask = None if mask_image is None else tiles_from_pixels(size, mask_image[..., 0], FULL)
    layer = Layer(
        raw["id"],
        raw["name"],
        kind,
        raw.get("blend", "normal"),
        raw.get("opacity", 1.0),
        raw.get("visible", True),
        value,
        tiles,
        mask,
    )
    return check_layer(layer, location)


def check_document(width: Any, height: Any, encoding: Any, background: Any, location: str):
    """Validated ``(width, height, encoding, background)`` of a painted texture."""
    limits = ImageLimits()
    for value, what in ((width, "width"), (height, "height")):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= limits.max_dimension
        ):
            raise _err(
                "INVALID_TEXTURE",
                f"{what} must be an integer in [1, {limits.max_dimension}]",
                location,
            )
    if width * height > limits.max_pixels:
        raise _err(
            "IMAGE_TOO_LARGE",
            f"{width}x{height} exceeds the limit of {limits.max_pixels:,} texels",
            location,
        )
    if encoding not in ENCODINGS:
        raise _err("INVALID_TEXTURE", f"encoding must be one of {', '.join(ENCODINGS)}", location)
    return int(width), int(height), encoding, _rgba(background, location)


__all__ = ["BLEND_MODES", "ENCODINGS", "Layer", "PaintedTexture", "TILE"]
