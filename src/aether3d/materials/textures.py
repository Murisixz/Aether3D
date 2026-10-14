"""Texture resources: image textures (PNG/JPEG files or embedded bytes) and procedural textures.

Textures live in a scene's material library (``scene.materials``) and are
shared by any number of materials. Each has a stable ``id`` and a ``name``
unique among the scene's textures.

:class:`ImageTexture`
    Pixels come from a PNG or JPEG file referenced by path, or from image
    bytes embedded in the project. Sampler settings belong to the texture (as
    in glTF): ``wrap`` per axis (``"repeat"``, ``"clamp"``, ``"mirror"``),
    ``filter`` (``"linear"`` or ``"nearest"``) and ``mipmaps``. How the
    channels are interpreted -- sRGB colors or linear data -- follows the
    material slot the texture is bound to (``color_space="auto"``), unless
    ``color_space`` forces ``"srgb"`` or ``"linear"``.

:class:`ProceduralTexture`
    A node graph evaluated per sample on the CPU and per fragment on the GPU
    (:mod:`aether3d.materials.graph`); never stored as a bitmap. ``preview``
    and ``bake`` render it to an image on request.

Files are read through :class:`ImageCache`: a file is re-read when its size,
modification time or inode changes (so edited textures are picked up by the
next render), identical content is decoded once (keyed by its SHA-256
digest), and decoded images and mip chains are kept within a memory budget
(``AETHER3D_TEXTURE_CACHE_MB``, default 1024).
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import MaterialError
from .graph import EvalContext, TextureGraph, evaluate
from .images import DecodedImage, ImageLimits, decode_image, sniff_format
from .sampling import FILTERS, WRAP_MODES, MipChain, build_mip_chain, mip_bytes

if TYPE_CHECKING:
    from .library import MaterialLibrary

COLOR_SPACES = ("auto", "srgb", "linear")
CACHE_ENV = "AETHER3D_TEXTURE_CACHE_MB"
DEFAULT_CACHE_BUDGET = 1024 * 2**20
MAX_EMBEDDED_BYTES = 64 * 2**20
"""Largest image that can be embedded in a project (encoded bytes)."""


def _err(code: str, message: str, location: str | None = None, **kw) -> MaterialError:
    return MaterialError(code, message, location, **kw)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def check_relative_path(text: Any, location: str) -> str:
    """Validate a project-relative texture path (POSIX separators, inside the project).

    Absolute paths, drive letters, backslashes, ``..`` components and empty
    components are refused: a project can only reference files in its own
    directory or below.
    """
    if not isinstance(text, str) or not text or len(text) > 4096 or "\x00" in text:
        raise _err("UNSAFE_PATH", "a texture path must be a non-empty relative path", location)
    if "\\" in text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise _err(
            "UNSAFE_PATH",
            f"texture path {text!r} must be relative to the project, with '/' separators",
            location,
        )
    parts = PurePosixPath(text).parts
    if any(part in ("..", "") for part in parts) or text.startswith("./") and len(parts) == 0:
        raise _err(
            "UNSAFE_PATH",
            f"texture path {text!r} leaves the project directory",
            location,
            hint="keep textures in the project's directory (or below), or embed them",
        )
    if text.endswith("/"):
        raise _err("UNSAFE_PATH", f"texture path {text!r} names a directory", location)
    return text


def relative_to_project(file: Path, project_dir: Path, location: str) -> str:
    """The POSIX path of ``file`` relative to ``project_dir``; it must be inside it."""
    try:
        rel = Path(os.path.relpath(os.path.abspath(file), os.path.abspath(project_dir)))
    except ValueError:  # different drives on Windows
        rel = None
    if rel is None or rel.is_absolute() or rel.parts[:1] == ("..",):
        raise _err(
            "TEXTURE_OUTSIDE_PROJECT",
            f"texture file {str(file)!r} is outside the project directory {str(project_dir)!r}",
            location,
            hint="copy the image into the project's directory (or below) and use that file, or "
            "embed it (texture.embed())",
        )
    return rel.as_posix()


# --------------------------------------------------------------------------- #
# The image cache
# --------------------------------------------------------------------------- #


def _cache_budget() -> int:
    raw = os.environ.get(CACHE_ENV, "").strip()
    if not raw:
        return DEFAULT_CACHE_BUDGET
    try:
        return max(0, int(float(raw) * 2**20))
    except ValueError:
        raise _err(
            "INVALID_SETTINGS", f"{CACHE_ENV} must be a number of megabytes, got {raw!r}"
        ) from None


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image and where it came from."""

    image: DecodedImage
    digest: str  # SHA-256 of the encoded bytes
    source: str  # "file" or "embedded"
    nbytes: int  # encoded size


class ImageCache:
    """Decoded images and mip chains, shared by every scene (see the module docstring)."""

    def __init__(self) -> None:
        self.budget = _cache_budget()
        self._files: dict[str, tuple[tuple[int, int, int, int], str]] = {}
        self._decoded: OrderedDict[str, DecodedImage] = OrderedDict()
        self._chains: OrderedDict[tuple[str, str, bool], MipChain] = OrderedDict()
        self.bytes = 0
        self.reads = 0
        self.decodes = 0
        self.hits = 0
        self.evictions = 0

    def clear(self) -> None:
        self._files.clear()
        self._decoded.clear()
        self._chains.clear()
        self.bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "images": len(self._decoded),
            "mip_chains": len(self._chains),
            "bytes": self.bytes,
            "budget": self.budget,
            "file_reads": self.reads,
            "decodes": self.decodes,
            "hits": self.hits,
            "evictions": self.evictions,
        }

    def read_file(self, file: Path, limits: ImageLimits, location: str) -> tuple[bytes | None, str]:
        """``(bytes or None if unchanged and cached, digest)`` of a texture file."""
        try:
            info = os.stat(file)
        except FileNotFoundError:
            raise _err(
                "TEXTURE_MISSING",
                f"texture file {str(file)!r} does not exist",
                location,
                hint="restore the file, point the texture at another file, or remove it",
            ) from None
        except OSError as exc:
            raise _err(
                "TEXTURE_UNREADABLE", f"cannot read {str(file)!r}: {exc.strerror}", location
            ) from None
        if not stat.S_ISREG(info.st_mode):
            raise _err("TEXTURE_UNREADABLE", f"{str(file)!r} is not a regular file", location)
        if info.st_size > limits.max_bytes:
            raise _err(
                "IMAGE_TOO_LARGE",
                f"texture file {str(file)!r} has {info.st_size:,} bytes "
                f"(limit {limits.max_bytes:,})",
                location,
            )
        key = str(file)
        signature = (info.st_mtime_ns, info.st_size, info.st_ino, info.st_dev)
        known = self._files.get(key)
        if known is not None and known[0] == signature and known[1] in self._decoded:
            self.hits += 1
            return None, known[1]
        try:
            with open(file, "rb") as handle:
                data = handle.read(limits.max_bytes + 1)
        except OSError as exc:
            raise _err(
                "TEXTURE_UNREADABLE", f"cannot read {str(file)!r}: {exc.strerror}", location
            ) from None
        if len(data) > limits.max_bytes:
            raise _err(
                "IMAGE_TOO_LARGE", f"texture file {str(file)!r} grew beyond the limit", location
            )
        self.reads += 1
        digest = hashlib.sha256(data).hexdigest()
        self._files[key] = (signature, digest)
        return data, digest

    def decode(
        self, digest: str, data: bytes | None, limits: ImageLimits, location: str
    ) -> DecodedImage:
        image = self._decoded.get(digest)
        if image is not None:
            self._decoded.move_to_end(digest)
            return image
        if data is None:  # pragma: no cover - read_file only skips data for cached digests
            raise _err("TEXTURE_INVALID", "image data is no longer cached", location)
        image = decode_image(data, limits=limits, location=location)
        self.decodes += 1
        self._decoded[digest] = image
        self.bytes += image.pixels.nbytes
        self._trim(keep={digest})
        return image

    def chain(self, digest: str, image: DecodedImage, mode: str, mipmaps: bool) -> MipChain:
        key = (digest, mode, bool(mipmaps))
        chain = self._chains.get(key)
        if chain is not None:
            self._chains.move_to_end(key)
            return chain
        chain = build_mip_chain(image.pixels, mode, mipmaps)
        self._chains[key] = chain
        self.bytes += chain.nbytes - (
            chain.levels[0].nbytes if chain.levels[0] is image.pixels else 0
        )
        self._trim(keep={digest})
        return chain

    def _trim(self, keep: set[str]) -> None:
        while self.bytes > self.budget:
            victim = next((k for k in self._chains if k[0] not in keep), None)
            if victim is not None:
                chain = self._chains.pop(victim)
                base = self._decoded.get(victim[0])
                shared = base is not None and chain.levels[0] is base.pixels
                self.bytes -= chain.nbytes - (chain.levels[0].nbytes if shared else 0)
                self.evictions += 1
                continue
            image_key = next((k for k in self._decoded if k not in keep), None)
            if image_key is None:
                return
            image = self._decoded.pop(image_key)
            self.bytes -= image.pixels.nbytes
            self.evictions += 1


IMAGE_CACHE = ImageCache()
"""The process-wide image cache."""


# --------------------------------------------------------------------------- #
# Textures
# --------------------------------------------------------------------------- #


class Texture:
    """Base class of :class:`ImageTexture` and :class:`ProceduralTexture`."""

    kind = ""
    __slots__ = ("_library", "_id", "_name")

    def __init__(self, library: MaterialLibrary, texture_id: str, name: str) -> None:
        self._library: MaterialLibrary | None = library
        self._id = texture_id
        self._name = name

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._require_library()._rename_texture(self, value)

    @property
    def location(self) -> str:
        return f"textures[{self._name}]"

    def _require_library(self) -> MaterialLibrary:
        if self._library is None:
            raise _err("TEXTURE_REMOVED", f"texture {self._name!r} was removed from its scene")
        return self._library

    def _touch(self) -> None:
        if self._library is not None:
            self._library._changed()

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"


class ImageTexture(Texture):
    """A PNG or JPEG image (file or embedded bytes) with its sampler settings."""

    kind = "image"
    __slots__ = (
        "_file",
        "_root",
        "_data",
        "_wrap",
        "_filter",
        "_mipmaps",
        "_color_space",
        "_limits",
    )

    def __init__(
        self,
        library: MaterialLibrary,
        texture_id: str,
        name: str,
        *,
        file: Path | None = None,
        root: Path | None = None,
        data: bytes | None = None,
        wrap: tuple[str, str] = ("repeat", "repeat"),
        filter: str = "linear",
        mipmaps: bool = True,
        color_space: str = "auto",
    ) -> None:
        super().__init__(library, texture_id, name)
        self._file = file
        self._root = root
        self._data = data
        self._wrap = wrap
        self._filter = filter
        self._mipmaps = mipmaps
        self._color_space = color_space
        self._limits = ImageLimits()

    # -- source ---------------------------------------------------------------------

    @property
    def source(self) -> str:
        """``"file"`` or ``"embedded"``."""
        return "embedded" if self._data is not None else "file"

    @property
    def file(self) -> Path | None:
        """The image file (absolute), or ``None`` for an embedded image."""
        return self._file if self._data is None else None

    @property
    def path(self) -> str | None:
        """The file as given or as stored in the project (``None`` when embedded)."""
        if self._data is not None or self._file is None:
            return None
        if self._root is not None:
            try:
                return Path(os.path.relpath(self._file, self._root)).as_posix()
            except ValueError:  # pragma: no cover - different drives
                return str(self._file)
        return str(self._file)

    @property
    def embedded_bytes(self) -> bytes | None:
        return self._data

    @property
    def mime_type(self) -> str | None:
        """``image/png`` or ``image/jpeg`` for embedded images (files: from their content)."""
        data = self._data
        kind = sniff_format(data[:16]) if data is not None else None
        return {"png": "image/png", "jpeg": "image/jpeg"}.get(kind) if kind else None

    def embed(self) -> None:
        """Read the image file into the project (it no longer depends on the file)."""
        if self._data is not None:
            return
        loaded = self.load()
        data = self._read_bytes()
        if len(data) > MAX_EMBEDDED_BYTES:
            raise _err(
                "IMAGE_TOO_LARGE",
                f"{len(data):,} bytes is too large to embed (limit {MAX_EMBEDDED_BYTES:,})",
                self.location,
            )
        assert loaded.image is not None
        self._data = data
        self._touch()

    def extract(self, path: str | os.PathLike[str]) -> Path:
        """Write an embedded image to ``path`` (atomically) and reference that file instead."""
        from .._fileio import write_atomic

        if self._data is None:
            raise _err("NOT_EMBEDDED", f"texture {self._name!r} is not embedded", self.location)
        target = Path(path).absolute()
        kind = sniff_format(self._data[:16])
        if target.suffix.lower() not in ({".png"} if kind == "png" else {".jpg", ".jpeg"}):
            raise _err(
                "INVALID_OUTPUT",
                f"a {kind.upper() if kind else 'image'} must be written to a "
                f"{'.png' if kind == 'png' else '.jpg/.jpeg'} file",
                "path",
            )
        write_atomic(target, self._data)
        self._data = None
        self._file = target
        self._root = None
        self._touch()
        return target

    def _read_bytes(self) -> bytes:
        if self._data is not None:
            return self._data
        file = self._checked_file()
        data, _ = IMAGE_CACHE.read_file(file, self._limits, self.location)
        if data is None:
            with open(file, "rb") as handle:
                data = handle.read(self._limits.max_bytes + 1)
        return data

    def _checked_file(self) -> Path:
        file = self._file
        if file is None:
            raise _err("TEXTURE_MISSING", f"texture {self._name!r} has no image", self.location)
        if self._root is not None:
            root = self._root.resolve()
            try:
                resolved = file.resolve()
            except OSError:  # pragma: no cover - resolution loops
                resolved = file
            if not resolved.is_relative_to(root):
                raise _err(
                    "UNSAFE_PATH",
                    f"texture file {str(file)!r} resolves outside the project directory",
                    self.location,
                    hint="replace the symbolic link by the file itself, or embed the image",
                )
        return file

    def load(self) -> LoadedImage:
        """Read and decode the image (cached); raises :class:`MaterialError` if it cannot."""
        if self._data is not None:
            digest = hashlib.sha256(self._data).hexdigest()
            image = IMAGE_CACHE.decode(digest, self._data, self._limits, self.location)
            return LoadedImage(image, digest, "embedded", len(self._data))
        file = self._checked_file()
        data, digest = IMAGE_CACHE.read_file(file, self._limits, self.location)
        image = IMAGE_CACHE.decode(digest, data, self._limits, self.location)
        size = len(data) if data is not None else os.stat(file).st_size
        return LoadedImage(image, digest, "file", size)

    def mip_chain(self, mode: str) -> tuple[MipChain, LoadedImage]:
        """The image's levels decoded for ``mode`` (``"srgb"``, ``"linear"``, ``"normal"``)."""
        loaded = self.load()
        return IMAGE_CACHE.chain(loaded.digest, loaded.image, mode, self._mipmaps), loaded

    def status(self) -> dict[str, Any]:
        """``{"ok": True, ...image info}`` or ``{"ok": False, "code": ..., "message": ...}``."""
        try:
            loaded = self.load()
        except MaterialError as exc:
            return {"ok": False, "code": exc.code, "message": exc.message, "hint": exc.hint}
        image = loaded.image
        return {
            "ok": True,
            **image.to_dict(),
            "encoded_bytes": loaded.nbytes,
            "digest": loaded.digest,
            "texture_bytes": mip_bytes(image.width, image.height, self._mipmaps),
        }

    # -- sampler --------------------------------------------------------------------

    @property
    def wrap(self) -> tuple[str, str]:
        return self._wrap

    @property
    def filter(self) -> str:
        return self._filter

    @property
    def mipmaps(self) -> bool:
        return self._mipmaps

    @property
    def color_space(self) -> str:
        return self._color_space

    def configure(self, **changes: Any) -> ImageTexture:
        """Change ``wrap`` (one mode or ``[u, v]``), ``wrap_u``, ``wrap_v``, ``filter``,
        ``mipmaps`` or ``color_space`` (validated together)."""
        values = check_sampler(
            {
                "wrap": changes.pop("wrap", self._wrap),
                "filter": changes.pop("filter", self._filter),
                "mipmaps": changes.pop("mipmaps", self._mipmaps),
                "color_space": changes.pop("color_space", self._color_space),
                **{k: changes.pop(k) for k in ("wrap_u", "wrap_v") if k in changes},
            },
            self.location,
        )
        if changes:
            raise _err(
                "INVALID_TEXTURE",
                f"unknown image texture setting(s): {', '.join(sorted(changes))}",
                self.location,
                hint="settings: wrap, wrap_u, wrap_v, filter, mipmaps, color_space",
            )
        self._wrap, self._filter = values["wrap"], values["filter"]
        self._mipmaps, self._color_space = values["mipmaps"], values["color_space"]
        self._touch()
        return self

    def mode_for(self, slot_mode: str) -> str:
        """The decoding mode for a slot: the slot's, unless ``color_space`` forces one."""
        if self._color_space == "srgb":
            return "srgb"
        if self._color_space == "linear" and slot_mode == "srgb":
            return "linear"
        return slot_mode

    def to_dict(self, project_dir: Path | None = None) -> dict[str, Any]:
        """The project record (file paths relative to ``project_dir`` when given)."""
        data: dict[str, Any] = {"id": self._id, "name": self._name, "type": "image"}
        if self._data is not None:
            data["embedded"] = {
                "mime_type": self.mime_type or "application/octet-stream",
                "data": base64.b64encode(self._data).decode("ascii"),
            }
        elif project_dir is not None and self._file is not None:
            data["path"] = relative_to_project(self._file, project_dir, self.location)
        else:
            data["path"] = self.path
        data["wrap"] = list(self._wrap)
        data["filter"] = self._filter
        data["mipmaps"] = self._mipmaps
        data["color_space"] = self._color_space
        return data

    def info(self) -> dict[str, Any]:
        """A JSON-compatible description, including the image's status."""
        return {
            "id": self._id,
            "name": self._name,
            "type": "image",
            "source": self.source,
            "path": self.path,
            "wrap": list(self._wrap),
            "filter": self._filter,
            "mipmaps": self._mipmaps,
            "color_space": self._color_space,
            "status": self.status(),
        }


def check_sampler(values: dict[str, Any], location: str) -> dict[str, Any]:
    """Validate image sampler settings; returns ``wrap`` (a pair), ``filter``, ``mipmaps``,
    ``color_space``."""
    wrap = values.get("wrap", ("repeat", "repeat"))
    if isinstance(wrap, str):
        wrap = (wrap, wrap)
    if not isinstance(wrap, (list, tuple)) or len(wrap) != 2:
        raise _err(
            "INVALID_TEXTURE",
            f"wrap must be a mode or [u, v] modes, got {wrap!r}",
            f"{location}.wrap",
        )
    wrap = (values.get("wrap_u", wrap[0]), values.get("wrap_v", wrap[1]))
    for axis, mode in zip("uv", wrap, strict=True):
        if mode not in WRAP_MODES:
            raise _err(
                "INVALID_TEXTURE",
                f"wrap {axis} must be one of {', '.join(WRAP_MODES)}, got {mode!r}",
                f"{location}.wrap",
            )
    filter_ = values.get("filter", "linear")
    if filter_ not in FILTERS:
        raise _err(
            "INVALID_TEXTURE",
            f"filter must be one of {', '.join(FILTERS)}, got {filter_!r}",
            f"{location}.filter",
        )
    mipmaps = values.get("mipmaps", True)
    if not isinstance(mipmaps, bool):
        raise _err(
            "INVALID_TEXTURE",
            f"mipmaps must be true or false, got {mipmaps!r}",
            f"{location}.mipmaps",
        )
    color_space = values.get("color_space", "auto")
    if color_space not in COLOR_SPACES:
        raise _err(
            "INVALID_TEXTURE",
            f"color_space must be one of {', '.join(COLOR_SPACES)}, got {color_space!r}",
            f"{location}.color_space",
        )
    return {"wrap": tuple(wrap), "filter": filter_, "mipmaps": mipmaps, "color_space": color_space}


class ProceduralTexture(Texture):
    """A procedural texture: a node graph evaluated where it is sampled (never a bitmap)."""

    kind = "procedural"
    __slots__ = ("_graph",)

    def __init__(self, library: MaterialLibrary, texture_id: str, name: str, graph: TextureGraph):
        super().__init__(library, texture_id, name)
        self._graph = graph

    @property
    def graph(self) -> TextureGraph:
        return self._graph

    def set_graph(self, graph: Any) -> None:
        """Replace the graph (a dict, a single node or a :class:`TextureGraph`)."""
        library = self._require_library()
        new = TextureGraph.from_dict(graph, self.location)
        library._check_graph(new, self.location)
        self._graph = new
        self._touch()

    def set_params(self, node: str | None = None, **params: Any) -> None:
        """Change parameters of one node (the output node by default)."""
        data = self._graph.to_dict()
        node_id = node or self._graph.output.node
        if node_id not in data["nodes"]:
            raise _err("INVALID_TEXTURE_GRAPH", f"no node {node_id!r} in this graph", self.location)
        data["nodes"][node_id].update(params)
        self.set_graph(data)

    def evaluate(self, points: Any) -> np.ndarray:
        """``(N, 4)`` linear RGBA at ``(N, 3)`` (or ``(N, 2)`` uv) texture coordinates."""
        p = np.asarray(points, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] not in (2, 3):
            raise _err("INVALID_ARGUMENT", "points must have shape (N, 2) or (N, 3)", self.location)
        if p.shape[1] == 2:
            p = np.concatenate([p, np.zeros((p.shape[0], 1))], axis=1)
        zeros = np.zeros_like(p)
        library = self._require_library()
        return evaluate(self._graph, EvalContext(p, zeros, zeros, library._image_sampler()))

    def preview(
        self,
        width: int = 256,
        height: int | None = None,
        *,
        region: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (1.0, 1.0)),
    ) -> np.ndarray:
        """An 8-bit sRGB ``(height, width, 3)`` image of the texture over ``region`` (in uv).

        Pixel centers are sampled once each; row 0 is the top (``v`` = region max).
        """
        from ..rendering.color import linear_to_srgb, quantize

        height = width if height is None else height
        for value, what in ((width, "width"), (height, "height")):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8192:
                raise _err(
                    "INVALID_ARGUMENT", f"{what} must be an integer in [1, 8192]", self.location
                )
        (u0, v0), (u1, v1) = region
        u = u0 + (np.arange(width) + 0.5) / width * (u1 - u0)
        v = v1 - (np.arange(height) + 0.5) / height * (v1 - v0)
        uu, vv = np.meshgrid(u, v)
        p = np.stack([uu.ravel(), vv.ravel(), np.zeros(uu.size)], axis=1)
        du = np.zeros_like(p)
        du[:, 0] = (u1 - u0) / width
        dv = np.zeros_like(p)
        dv[:, 1] = -(v1 - v0) / height
        library = self._require_library()
        rgba = evaluate(self._graph, EvalContext(p, du, dv, library._image_sampler()))
        return quantize(linear_to_srgb(rgba[:, :3])).reshape(height, width, 3)

    def bake(
        self, path: str | os.PathLike[str], width: int = 1024, height: int | None = None
    ) -> Path:
        """Write :meth:`preview` of the unit uv square as a PNG file (an explicit, optional output;
        the texture itself stays procedural)."""
        from ..rendering.png import write_png

        image = self.preview(width, height)
        return write_png(path, image)

    def to_dict(self, project_dir: Path | None = None) -> dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "type": "procedural",
            "graph": self._graph.to_dict(),
        }

    def info(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "type": "procedural",
            "output": self._graph.output_kind(),
            "nodes": len(self._graph.nodes),
            "node_types": sorted({n.type for n in self._graph.nodes.values()}),
            "graph": self._graph.to_dict(),
            "status": {"ok": True},
        }
