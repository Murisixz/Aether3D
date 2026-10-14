"""glTF 2.0 import (``.gltf`` and ``.glb``) into Aether3D scenes.

The reader treats every file as untrusted input:

* the whole document is validated as it is read, and every problem is raised
  as :class:`AssetFormatError` with a stable ``code`` and a JSON-pointer
  ``location`` (e.g. ``/accessors/4``);
* sizes are checked against :class:`ImportLimits` *before* data is allocated;
* external files are only read from the asset's own directory (or below it);
  absolute paths, ``..`` escapes, symlinks leading outside, and non-``data:``
  URI schemes (``http:``, ``file:``, ...) are refused;
* only buffers needed for imported geometry and materials are read; images
  are decoded with the same safeguards as texture files (signature, size
  limits, no decompression bombs);
* a new :class:`Scene` is built in isolation and returned only on success, so
  a failed import has no side effects.

Content that Aether3D cannot represent is reported, never silently changed.
Materials and their textures are imported (see :mod:`aether3d.gltf.materials`);
primitives with different materials become face groups of one mesh. Other
non-geometric data (cameras, animations, extensions that are not required)
is skipped and listed in ``ImportResult.diagnostics``.
Content whose omission would change the imported geometry (skins, morph
targets, point/line primitives, sheared node matrices) raises
:class:`UnsupportedFeatureError` unless ``allow_lossy=True`` is passed.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import (
    AetherError,
    AssetFormatError,
    ResourceLimitError,
    UnsupportedFeatureError,
    ValidationError,
)
from ..math3d import compose_matrix, decompose_matrix, quat
from ..mesh import Mesh
from ..scene import Scene, SceneObject
from ..transform import Transform
from . import axes
from .materials import MaterialImporter, assign
from .writer import CHUNK_BIN, CHUNK_JSON, EXTRAS_KEY, GLB_MAGIC

COMPONENT_TYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
INDEX_TYPES = (5121, 5123, 5125)

MODE_NAMES = {
    0: "POINTS",
    1: "LINES",
    2: "LINE_LOOP",
    3: "LINE_STRIP",
    4: "TRIANGLES",
    5: "TRIANGLE_STRIP",
    6: "TRIANGLE_FAN",
}
SUPPORTED_ATTRIBUTES = ("POSITION", "NORMAL", "TEXCOORD_0")


@dataclass(frozen=True)
class ImportLimits:
    """Safety limits applied while importing untrusted files.

    Exceeding any limit raises :class:`ResourceLimitError` before the
    offending data is allocated.
    """

    max_file_bytes: int = 512 * 1024 * 1024
    """Largest ``.gltf``/``.glb`` file, and largest external buffer file."""
    max_buffer_bytes: int = 1024 * 1024 * 1024
    """Total bytes of all buffers loaded for one import."""
    max_nodes: int = 100_000
    max_vertices: int = 50_000_000
    """Total vertices across all imported (unique) meshes."""
    max_triangles: int = 50_000_000
    """Total triangles across all imported (unique) meshes."""


@dataclass
class ImportResult:
    """Outcome of an import: the new scene plus a report of what happened.

    ``objects`` maps glTF node indices to the created objects
    (``{"node", "id", "name", "id_source"}``, where ``id_source`` is
    ``"extras"`` when the id was restored from ``extras.aether3d.id`` and
    ``"generated"`` otherwise).
    """

    scene: Scene
    source: Path
    format: str
    generator: str | None
    objects: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def lossy(self) -> bool:
        """``True`` if content was dropped or approximated (only possible with ``allow_lossy``)."""
        return any(d.lossy for d in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "format": self.format,
            "generator": self.generator,
            "scene": self.scene.name,
            "object_count": len(self.scene),
            "lossy": self.lossy,
            "objects": [dict(o) for o in self.objects],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    return {dict: "object", list: "array", str: "string"}.get(type(value), "number")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssetFormatError(f"duplicate JSON key {key!r}", None, "INVALID_JSON")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise AssetFormatError(f"{name} is not valid JSON", None, "INVALID_JSON")


class _Reader:
    def __init__(self, path: Path, limits: ImportLimits, allow_lossy: bool) -> None:
        self.path = path
        self.base_dir = path.parent.resolve()
        self.limits = limits
        self.allow_lossy = allow_lossy
        self.diagnostics: list[Diagnostic] = []
        self.bin_chunk: bytes | None = None
        self.buffers: dict[int, bytes] = {}
        self.buffer_bytes = 0
        self.vertex_total = 0
        self.triangle_total = 0
        self.meshes: dict[int, Mesh | None] = {}
        self.mesh_materials: dict[int, tuple[dict[int, Any], bool]] = {}
        self.importer: MaterialImporter | None = None
        self.doc: dict[str, Any] = {}
        self.format = "gltf"

    # -- diagnostics -------------------------------------------------------

    def note(
        self,
        code: str,
        severity: str,
        message: str,
        location: str | None = None,
        count: int | None = None,
        lossy: bool = False,
    ) -> None:
        self.diagnostics.append(Diagnostic(code, severity, message, location, count, lossy=lossy))

    def lossy(self, code: str, message: str, location: str | None = None, count: int | None = None):
        self.note(code, WARNING, message, location, count, lossy=True)

    # -- typed access to the JSON document ---------------------------------

    def array(self, key: str) -> list[Any]:
        value = self.doc.get(key, [])
        if not isinstance(value, list):
            raise AssetFormatError(f"expected an array, got {_type_name(value)}", f"/{key}")
        return value

    def item(self, key: str, index: Any, loc: str) -> tuple[dict[str, Any], str]:
        items = self.array(key)
        if not _is_int(index) or not 0 <= index < len(items):
            raise AssetFormatError(
                f"invalid reference {index!r} to {key} (have {len(items)})",
                loc,
                "INVALID_REFERENCE",
            )
        value = items[index]
        where = f"/{key}/{index}"
        if not isinstance(value, dict):
            raise AssetFormatError(f"expected an object, got {_type_name(value)}", where)
        return value, where

    @staticmethod
    def int_field(
        obj: dict[str, Any], key: str, loc: str, default: int | None = None, minimum: int = 0
    ) -> int:
        value = obj.get(key, default)
        if not _is_int(value) or value < minimum:
            raise AssetFormatError(f"{key} must be an integer >= {minimum}, got {value!r}", loc)
        return value

    @staticmethod
    def numbers(obj: dict[str, Any], key: str, count: int, loc: str) -> list[float] | None:
        if key not in obj:
            return None
        value = obj[key]
        if not isinstance(value, list) or len(value) != count or not all(map(_is_number, value)):
            raise AssetFormatError(f"{key} must be an array of {count} numbers", f"{loc}/{key}")
        return value

    # -- container ---------------------------------------------------------

    def load(self) -> None:
        size = self.path.stat().st_size
        if size > self.limits.max_file_bytes:
            raise ResourceLimitError(
                f"{self.path.name} is {size} bytes; the limit is {self.limits.max_file_bytes}"
            )
        data = self.path.read_bytes()
        if data[:4] == b"glTF":
            self.format = "glb"
            json_bytes = self.parse_glb(data)
        else:
            json_bytes = data[3:] if data.startswith(b"\xef\xbb\xbf") else data
        try:
            text = json_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AssetFormatError(
                f"JSON is not valid UTF-8: {exc.reason}", None, "INVALID_JSON"
            ) from None
        try:
            doc = json.loads(
                text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
            )
        except json.JSONDecodeError as exc:
            raise AssetFormatError(
                f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
                None,
                "INVALID_JSON",
            ) from None
        except RecursionError:
            raise AssetFormatError("JSON is nested too deeply", None, "INVALID_JSON") from None
        if not isinstance(doc, dict):
            raise AssetFormatError("top level must be a JSON object", None, "INVALID_JSON")
        self.doc = doc

    def parse_glb(self, data: bytes) -> bytes:
        if len(data) < 12:
            raise AssetFormatError("file is too short to be a GLB", None, "INVALID_GLB")
        magic, version, length = struct.unpack_from("<III", data, 0)
        if magic != GLB_MAGIC:
            raise AssetFormatError("bad GLB magic", None, "INVALID_GLB")
        if version != 2:
            raise AssetFormatError(
                f"unsupported GLB container version {version}", None, "INVALID_GLB"
            )
        if length != len(data):
            raise AssetFormatError(
                f"GLB header declares {length} bytes but the file has {len(data)}",
                None,
                "INVALID_GLB",
            )
        offset, json_chunk, index = 12, None, 0
        while offset < length:
            if offset + 8 > length:
                raise AssetFormatError("truncated GLB chunk header", None, "INVALID_GLB")
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            start, end = offset + 8, offset + 8 + chunk_length
            if end > length:
                raise AssetFormatError(
                    f"GLB chunk {index} extends past the end of the file", None, "INVALID_GLB"
                )
            if index == 0:
                if chunk_type != CHUNK_JSON:
                    raise AssetFormatError("the first GLB chunk must be JSON", None, "INVALID_GLB")
                json_chunk = data[start:end]
            elif index == 1 and chunk_type == CHUNK_BIN:
                self.bin_chunk = data[start:end]
            elif chunk_type in (CHUNK_JSON, CHUNK_BIN):
                raise AssetFormatError(
                    f"unexpected {('JSON', 'BIN')[chunk_type == CHUNK_BIN]} chunk "
                    f"at position {index}",
                    None,
                    "INVALID_GLB",
                )
            offset, index = end, index + 1
        if json_chunk is None:
            raise AssetFormatError("GLB has no JSON chunk", None, "INVALID_GLB")
        # The spec pads with spaces, but some exporters pad with NUL bytes.
        return json_chunk.rstrip(b"\x00")

    # -- buffers -----------------------------------------------------------

    def charge(self, nbytes: int, loc: str) -> None:
        self.buffer_bytes += nbytes
        if self.buffer_bytes > self.limits.max_buffer_bytes:
            raise ResourceLimitError(
                f"{loc}: loading buffers would exceed {self.limits.max_buffer_bytes} bytes"
            )

    def buffer(self, index: int, loc: str) -> bytes:
        if index in self.buffers:
            return self.buffers[index]
        buf, where = self.item("buffers", index, loc)
        byte_length = self.int_field(buf, "byteLength", where, minimum=1)
        self.charge(byte_length, where)
        uri = buf.get("uri")
        if uri is None:
            if self.format != "glb" or index != 0 or self.bin_chunk is None:
                raise AssetFormatError(
                    "buffer has no uri and there is no GLB BIN chunk", where, "MISSING_RESOURCE"
                )
            data = self.bin_chunk
        elif not isinstance(uri, str):
            raise AssetFormatError("uri must be a string", f"{where}/uri")
        elif uri.startswith("data:"):
            data = self.data_uri(uri, byte_length, f"{where}/uri")
        else:
            data = self.external_file(uri, f"{where}/uri")
        if len(data) < byte_length:
            raise AssetFormatError(
                f"buffer declares {byte_length} bytes but only {len(data)} are available",
                where,
                "BUFFER_TOO_SHORT",
            )
        self.buffers[index] = data[:byte_length]
        return self.buffers[index]

    def data_uri(self, uri: str, byte_length: int, loc: str) -> bytes:
        header, sep, payload = uri.partition(",")
        if not sep or not header.endswith(";base64"):
            raise AssetFormatError("only base64 data URIs are supported", loc, "INVALID_URI")
        mime = header[5:-7]
        if mime not in ("application/octet-stream", "application/gltf-buffer"):
            raise AssetFormatError(f"unexpected buffer media type {mime!r}", loc, "INVALID_URI")
        if len(payload) > (byte_length + 3) * 4 // 3 + 4:
            raise AssetFormatError(
                "data URI is longer than the declared byteLength", loc, "BUFFER_LENGTH_MISMATCH"
            )
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise AssetFormatError("data URI is not valid base64", loc, "INVALID_URI") from None

    def external_file(self, uri: str, loc: str) -> bytes:
        decoded = urllib.parse.unquote(uri)
        parsed = urllib.parse.urlsplit(uri)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or "\\" in decoded
            or "\0" in decoded
            or PurePosixPath(decoded).is_absolute()
            or PureWindowsPath(decoded).is_absolute()
            or PureWindowsPath(decoded).drive
        ):
            raise AssetFormatError(
                f"refusing to load {uri!r}: only relative paths inside the asset's directory "
                "and base64 data URIs are allowed",
                loc,
                "UNSAFE_URI",
            )
        candidate = (self.base_dir / decoded).resolve()
        if not candidate.is_relative_to(self.base_dir):
            raise AssetFormatError(
                f"refusing to load {uri!r}: it resolves outside the asset's directory",
                loc,
                "UNSAFE_URI",
            )
        if not candidate.is_file():
            raise AssetFormatError(
                f"referenced file {decoded!r} was not found", loc, "MISSING_RESOURCE"
            )
        size = candidate.stat().st_size
        if size > self.limits.max_file_bytes:
            raise ResourceLimitError(
                f"{loc}: {decoded!r} is {size} bytes; the limit is {self.limits.max_file_bytes}"
            )
        return candidate.read_bytes()

    # -- accessors ---------------------------------------------------------

    def accessor(self, index: Any, loc: str, purpose: str) -> np.ndarray:
        acc, where = self.item("accessors", index, loc)
        component = acc.get("componentType")
        kind = acc.get("type")
        if component not in COMPONENT_TYPES or kind not in TYPE_SIZES:
            raise AssetFormatError(f"invalid componentType/type {component!r}/{kind!r}", where)
        count = self.int_field(acc, "count", where, minimum=1)
        normalized = acc.get("normalized", False)
        if not isinstance(normalized, bool):
            raise AssetFormatError("normalized must be a boolean", f"{where}/normalized")
        self.check_accessor_kind(purpose, component, kind, normalized, where)
        width = TYPE_SIZES[kind]
        limit = 3 * self.limits.max_triangles if purpose == "indices" else self.limits.max_vertices
        if count > limit:
            raise ResourceLimitError(
                f"{where}: accessor has {count} elements; the limit is {limit}"
            )
        dtype = COMPONENT_TYPES[component]

        if "bufferView" in acc:
            view, view_loc = self.item("bufferViews", acc["bufferView"], f"{where}/bufferView")
            data = self.buffer(self.int_field(view, "buffer", view_loc), f"{view_loc}/buffer")
            view_offset = self.int_field(view, "byteOffset", view_loc, default=0)
            view_length = self.int_field(view, "byteLength", view_loc, minimum=1)
            if view_offset + view_length > len(data):
                raise AssetFormatError(
                    "bufferView extends past the end of its buffer",
                    view_loc,
                    "BUFFER_VIEW_OUT_OF_BOUNDS",
                )
            element = dtype.itemsize * width
            stride = view.get("byteStride", element)
            if not _is_int(stride) or stride < element:
                raise AssetFormatError(f"byteStride must be an integer >= {element}", view_loc)
            offset = self.int_field(acc, "byteOffset", where, default=0)
            needed = offset + stride * (count - 1) + element
            if needed > view_length:
                raise AssetFormatError(
                    f"accessor needs {needed} bytes but its bufferView has {view_length}",
                    where,
                    "ACCESSOR_OUT_OF_BOUNDS",
                )
            start = view_offset + offset
            if stride == element:
                raw = np.frombuffer(data, dtype=dtype, count=count * width, offset=start)
            else:
                rows = np.frombuffer(data, dtype=np.uint8, count=needed - offset, offset=start)
                rows = np.concatenate([rows, np.zeros(stride - element, np.uint8)])
                raw = rows.reshape(count, stride)[:, :element].copy().view(dtype)
            values = raw.reshape(count, width).copy()
        else:
            values = np.zeros((count, width), dtype=dtype)

        if "sparse" in acc:
            self.apply_sparse(acc["sparse"], values, dtype, width, f"{where}/sparse")
        if normalized:
            if dtype.kind == "u":
                values = values.astype(np.float64) / np.iinfo(dtype).max
            else:
                values = np.maximum(values.astype(np.float64) / np.iinfo(dtype).max, -1.0)
        return values

    def apply_sparse(
        self, sparse: Any, values: np.ndarray, dtype: np.dtype, width: int, loc: str
    ) -> None:
        if not isinstance(sparse, dict):
            raise AssetFormatError("sparse must be an object", loc)
        count = self.int_field(sparse, "count", loc, minimum=1)
        if count > len(values):
            raise AssetFormatError("sparse count exceeds the accessor count", loc)

        def read(part: str, part_dtype: np.dtype, part_width: int) -> np.ndarray:
            spec = sparse.get(part)
            if not isinstance(spec, dict):
                raise AssetFormatError(f"sparse.{part} must be an object", f"{loc}/{part}")
            view, view_loc = self.item(
                "bufferViews", spec.get("bufferView"), f"{loc}/{part}/bufferView"
            )
            data = self.buffer(self.int_field(view, "buffer", view_loc), f"{view_loc}/buffer")
            start = self.int_field(view, "byteOffset", view_loc, default=0) + self.int_field(
                spec, "byteOffset", f"{loc}/{part}", default=0
            )
            nbytes = count * part_width * part_dtype.itemsize
            view_end = self.int_field(view, "byteOffset", view_loc, default=0) + self.int_field(
                view, "byteLength", view_loc, minimum=1
            )
            if start + nbytes > min(view_end, len(data)):
                raise AssetFormatError(
                    f"sparse {part} extend past their bufferView",
                    f"{loc}/{part}",
                    "ACCESSOR_OUT_OF_BOUNDS",
                )
            return np.frombuffer(data, part_dtype, count * part_width, start).reshape(
                count, part_width
            )

        index_spec = sparse.get("indices")
        index_type = index_spec.get("componentType") if isinstance(index_spec, dict) else None
        if index_type not in INDEX_TYPES:
            raise AssetFormatError(
                "sparse.indices.componentType must be an unsigned integer type", f"{loc}/indices"
            )
        indices = read("indices", COMPONENT_TYPES[index_type], 1).ravel().astype(np.int64)
        if np.any(indices >= len(values)) or np.any(np.diff(indices) <= 0):
            raise AssetFormatError(
                "sparse indices must be strictly increasing and within the accessor",
                f"{loc}/indices",
                "INVALID_SPARSE",
            )
        values[indices] = read("values", dtype, width)

    def check_accessor_kind(
        self, purpose: str, component: int, kind: str, normalized: bool, loc: str
    ) -> None:
        if purpose == "indices":
            ok = kind == "SCALAR" and component in INDEX_TYPES and not normalized
            expected = "SCALAR unsigned byte/short/int"
        elif purpose == "TEXCOORD_0":
            ok = kind == "VEC2" and (
                component == 5126 or (component in (5121, 5123) and normalized)
            )
            expected = "VEC2 float or normalized unsigned byte/short"
        else:
            ok = kind == "VEC3" and component == 5126
            expected = "VEC3 float"
        if not ok:
            raise AssetFormatError(
                f"{purpose} accessor must be {expected} (got {kind} componentType {component}"
                f"{' normalized' if normalized else ''}); quantized data needs "
                "KHR_mesh_quantization, "
                "which is not supported",
                loc,
                "INVALID_ACCESSOR_TYPE",
            )

    # -- meshes ------------------------------------------------------------

    def mesh(self, index: Any, loc: str) -> Mesh | None:
        gmesh, where = self.item("meshes", index, loc)
        if index in self.meshes:
            return self.meshes[index]
        primitives = gmesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise AssetFormatError("primitives must be a non-empty array", f"{where}/primitives")
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        for p_index, prim in enumerate(primitives):
            part = self.primitive(prim, f"{where}/primitives/{p_index}")
            if part is not None:
                parts.append(part)
        mesh = self.merge(parts, where, index) if parts else None
        if mesh is None:
            self.note(
                "MESH_WITHOUT_TRIANGLES",
                WARNING,
                "mesh has no importable triangle primitives; nodes using it get no geometry",
                where,
            )
        self.meshes[index] = mesh
        return mesh

    def primitive(self, prim: Any, loc: str):
        if not isinstance(prim, dict):
            raise AssetFormatError("primitive must be an object", loc)
        mode = prim.get("mode", 4)
        if mode not in MODE_NAMES:
            raise AssetFormatError(f"invalid primitive mode {mode!r}", f"{loc}/mode")
        attributes = prim.get("attributes")
        if not isinstance(attributes, dict):
            raise AssetFormatError("attributes must be an object", f"{loc}/attributes")
        if "targets" in prim:
            self.lossy(
                "MORPH_TARGETS_IGNORED",
                "morph targets are not supported; only the base geometry is imported",
                f"{loc}/targets",
            )
        if mode < 4:
            self.lossy(
                "PRIMITIVE_MODE_UNSUPPORTED",
                f"{MODE_NAMES[mode]} primitives cannot be represented as triangles; skipped",
                f"{loc}/mode",
            )
            return None
        if "POSITION" not in attributes:
            self.note(
                "PRIMITIVE_WITHOUT_POSITIONS",
                WARNING,
                "primitive has no POSITION attribute; skipped",
                f"{loc}/attributes",
            )
            return None
        ignored = sorted(set(attributes) - set(SUPPORTED_ATTRIBUTES))
        if ignored:
            self.note(
                "ATTRIBUTES_IGNORED",
                INFO,
                f"vertex attributes not imported: {', '.join(ignored)}",
                f"{loc}/attributes",
            )
        positions = self.accessor(attributes["POSITION"], f"{loc}/attributes/POSITION", "POSITION")
        n = len(positions)
        self.vertex_total += n
        if self.vertex_total > self.limits.max_vertices:
            raise ResourceLimitError(
                f"{loc}: more than {self.limits.max_vertices} vertices in total"
            )
        extra = {}
        for name in ("NORMAL", "TEXCOORD_0"):
            if name in attributes:
                values = self.accessor(attributes[name], f"{loc}/attributes/{name}", name)
                if len(values) != n:
                    raise AssetFormatError(
                        f"{name} has {len(values)} elements but POSITION has {n}",
                        f"{loc}/attributes/{name}",
                        "ATTRIBUTE_COUNT_MISMATCH",
                    )
                extra[name] = values
        if "indices" in prim:
            indices = (
                self.accessor(prim["indices"], f"{loc}/indices", "indices").ravel().astype(np.int64)
            )
            if indices.size and indices.max() >= n:
                raise AssetFormatError(
                    f"index {int(indices.max())} is out of range for {n} vertices",
                    f"{loc}/indices",
                    "INDEX_OUT_OF_RANGE",
                )
        else:
            indices = np.arange(n, dtype=np.int64)
        triangles = self.triangulate(indices, mode, loc)
        self.triangle_total += len(triangles)
        if self.triangle_total > self.limits.max_triangles:
            raise ResourceLimitError(
                f"{loc}: more than {self.limits.max_triangles} triangles in total"
            )
        material = prim.get("material")
        if material is not None:
            self.item("materials", material, f"{loc}/material")
        group = None  # a face group stored by Aether3D's exporter
        extras = prim.get("extras")
        stored = extras.get(EXTRAS_KEY) if isinstance(extras, dict) else None
        if isinstance(stored, dict) and "group" in stored:
            value = stored["group"]
            if _is_int(value) and 0 <= value <= 65535:
                group = value
        return positions, triangles, extra.get("NORMAL"), extra.get("TEXCOORD_0"), material, group

    @staticmethod
    def triangulate(indices: np.ndarray, mode: int, loc: str) -> np.ndarray:
        n = len(indices)
        if mode == 4:
            if n % 3:
                raise AssetFormatError(
                    f"TRIANGLES primitive has {n} indices, not a multiple of 3",
                    loc,
                    "INVALID_INDEX_COUNT",
                )
            return indices.reshape(-1, 3)
        if n < 3:
            return np.zeros((0, 3), dtype=np.int64)
        i = np.arange(n - 2)
        if mode == 5:  # glTF spec: p_i = {v_i, v_{i+(1+i%2)}, v_{i+(2-i%2)}}
            return np.stack([indices[i], indices[i + 1 + i % 2], indices[i + 2 - i % 2]], axis=1)
        return np.stack([indices[i + 1], indices[i + 2], np.full(n - 2, indices[0])], axis=1)

    def merge(self, parts, loc: str, mesh_index: int) -> Mesh:
        materials = [p[4] for p in parts]
        stored_groups = [p[5] for p in parts]
        grouped = len(set(materials)) > 1 or any(g is not None for g in stored_groups)
        group_of = [
            (g if g is not None else k) if grouped else 0 for k, g in enumerate(stored_groups)
        ]
        self.mesh_materials[mesh_index] = (
            {group_of[k]: m for k, m in enumerate(materials) if m is not None},
            grouped,
        )
        if len(parts) > 1 and not grouped:
            self.note(
                "PRIMITIVES_MERGED",
                INFO,
                f"{len(parts)} primitives with the same material merged into one mesh",
                loc,
            )
        elif len(parts) > 1:
            self.note(
                "PRIMITIVES_AS_GROUPS",
                INFO,
                f"{len(parts)} primitives became face groups of one mesh (each keeps its material)",
                loc,
            )
        positions, triangles, normals, uvs, offset = [], [], [], [], 0
        groups = []
        has_normals = all(p[2] is not None for p in parts)
        has_uvs = all(p[3] is not None for p in parts)
        for name, has, index in (("NORMAL", has_normals, 2), ("TEXCOORD_0", has_uvs, 3)):
            if not has and any(p[index] is not None for p in parts):
                self.note(
                    "ATTRIBUTE_DROPPED",
                    WARNING,
                    f"{name} is present on only some primitives; dropped for the whole mesh",
                    loc,
                )
        for k, (pos, tri, nor, uv, _, _) in enumerate(parts):
            positions.append(pos)
            triangles.append(tri + offset)
            groups.append(np.full(len(tri), group_of[k], dtype=np.int64))
            offset += len(pos)
            if has_normals:
                normals.append(nor)
            if has_uvs:
                uvs.append(uv)
        try:
            return Mesh(
                axes.vectors_from_gltf(np.concatenate(positions).astype(np.float32)),
                np.concatenate(triangles),
                normals=axes.vectors_from_gltf(np.concatenate(normals).astype(np.float32))
                if has_normals
                else None,
                uvs=axes.flip_uvs(np.concatenate(uvs)) if has_uvs else None,
                groups=np.concatenate(groups) if grouped else None,
            )
        except ValidationError as exc:
            raise AssetFormatError(str(exc), loc, "INVALID_MESH_DATA") from None

    # -- nodes and scene ---------------------------------------------------

    def transform(self, node: dict[str, Any], loc: str) -> Transform:
        matrix = self.numbers(node, "matrix", 16, loc)
        trs = [k for k in ("translation", "rotation", "scale") if k in node]
        try:
            if matrix is None:
                t = self.numbers(node, "translation", 3, loc) or (0.0, 0.0, 0.0)
                r = self.numbers(node, "rotation", 4, loc) or (0.0, 0.0, 0.0, 1.0)
                s = self.numbers(node, "scale", 3, loc) or (1.0, 1.0, 1.0)
                return Transform(
                    axes.vector_from_gltf(tuple(t)),
                    axes.rotation_from_gltf(quat(r, "rotation")),
                    axes.scale_from_gltf(tuple(s)),
                )
            if trs:
                raise AssetFormatError("node has both matrix and " + "/".join(trs), loc)
            m = np.array(matrix, dtype=np.float64).reshape(4, 4).T  # glTF matrices are column-major
            if not np.all(np.isfinite(m)):
                raise AssetFormatError("matrix contains non-finite values", f"{loc}/matrix")
            if not np.allclose(m[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
                raise AssetFormatError(
                    "matrix is not affine", f"{loc}/matrix", "NODE_MATRIX_INVALID"
                )
            try:
                t, r, s = decompose_matrix(m)
            except ValidationError:
                t, r = tuple(m[:3, 3].tolist()), (0.0, 0.0, 0.0, 1.0)
                s = tuple(np.linalg.norm(m[:3, :3], axis=0).tolist())
            if not np.allclose(
                compose_matrix(t, r, s), m, rtol=1e-6, atol=1e-6 * max(1.0, np.abs(m).max())
            ):
                self.lossy(
                    "NODE_MATRIX_NOT_TRS",
                    "node matrix contains shear (or is singular) and cannot be represented as "
                    "translation/rotation/scale; the closest TRS transform is used",
                    f"{loc}/matrix",
                )
            return Transform(
                axes.vector_from_gltf(t), axes.rotation_from_gltf(r), axes.scale_from_gltf(s)
            )
        except ValidationError as exc:
            raise AssetFormatError(str(exc), loc, "INVALID_TRANSFORM") from None

    def report_ignored(self) -> None:
        for key, code, what in (("cameras", "CAMERAS_IGNORED", "cameras"),):
            if self.array(key):
                self.note(
                    code,
                    INFO,
                    f"{len(self.array(key))} {what} not imported",
                    f"/{key}",
                    len(self.array(key)),
                )
        if self.array("animations"):
            self.note(
                "ANIMATIONS_IGNORED",
                WARNING,
                "animations are not imported; nodes keep their static (rest) transforms",
                "/animations",
                len(self.array("animations")),
            )
        used = self.doc.get("extensionsUsed", [])
        from .materials import SUPPORTED_MATERIAL_EXTENSIONS

        for name in used if isinstance(used, list) else []:
            if name in SUPPORTED_MATERIAL_EXTENSIONS:
                continue
            self.note(
                "EXTENSION_IGNORED",
                INFO,
                f"optional extension {name!r} is not supported; ignored",
                "/extensionsUsed",
            )

    def build(self) -> ImportResult:
        asset = self.doc.get("asset")
        if not isinstance(asset, dict) or not isinstance(asset.get("version"), str):
            raise AssetFormatError("missing asset.version", "/asset", "INVALID_GLTF")
        major = asset["version"].split(".")[0]
        if major != "2":
            raise UnsupportedFeatureError(
                f"glTF version {asset['version']} is not supported (2.x only)"
            )
        min_version = asset.get("minVersion")
        if min_version is not None and min_version != "2.0":
            raise UnsupportedFeatureError(
                f"asset requires glTF {min_version}; only 2.0 is supported"
            )
        required = self.doc.get("extensionsRequired", [])
        if not isinstance(required, list):
            raise AssetFormatError("extensionsRequired must be an array", "/extensionsRequired")
        if required:
            diags = [
                Diagnostic(
                    "REQUIRED_EXTENSION_UNSUPPORTED",
                    "error",
                    f"required extension {name!r} is not supported",
                    "/extensionsRequired",
                )
                for name in required
            ]
            raise UnsupportedFeatureError(
                "asset requires unsupported extensions: " + ", ".join(map(str, required)), diags
            )
        self.report_ignored()

        nodes = self.array("nodes")
        if len(nodes) > self.limits.max_nodes:
            raise ResourceLimitError(
                f"asset has {len(nodes)} nodes; the limit is {self.limits.max_nodes}"
            )
        parent: dict[int, int] = {}
        for index in range(len(nodes)):
            node, where = self.item("nodes", index, "/nodes")
            children = node.get("children", [])
            if not isinstance(children, list):
                raise AssetFormatError("children must be an array", f"{where}/children")
            for child in children:
                self.item("nodes", child, f"{where}/children")
                if child == index:
                    raise AssetFormatError(
                        "node lists itself as a child", f"{where}/children", "INVALID_HIERARCHY"
                    )
                if child in parent:
                    raise AssetFormatError(
                        f"node {child} has more than one parent",
                        f"{where}/children",
                        "INVALID_HIERARCHY",
                    )
                parent[child] = index
        for index in range(len(nodes)):  # every chain must reach a root
            seen, node = set(), index
            while node in parent:
                if node in seen:
                    raise AssetFormatError(
                        "node hierarchy contains a cycle", f"/nodes/{index}", "INVALID_HIERARCHY"
                    )
                seen.add(node)
                node = parent[node]

        scenes = self.array("scenes")
        name = None
        if scenes:
            scene_index = self.doc.get("scene", 0)
            gscene, where = self.item("scenes", scene_index, "/scene")
            roots = gscene.get("nodes", [])
            if not isinstance(roots, list):
                raise AssetFormatError("nodes must be an array", f"{where}/nodes")
            for root in roots:
                self.item("nodes", root, f"{where}/nodes")
                if root in parent:
                    raise AssetFormatError(
                        f"scene root {root} is the child of another node",
                        f"{where}/nodes",
                        "INVALID_HIERARCHY",
                    )
            if len(set(roots)) != len(roots):
                raise AssetFormatError(
                    "scene lists a root node twice", f"{where}/nodes", "INVALID_HIERARCHY"
                )
            if len(scenes) > 1:
                self.note(
                    "OTHER_SCENES_IGNORED",
                    WARNING,
                    f"asset has {len(scenes)} scenes; only scene {scene_index} is imported",
                    "/scenes",
                    len(scenes) - 1,
                )
            if isinstance(gscene.get("name"), str) and gscene["name"].strip():
                name = gscene["name"]
        else:
            roots = [i for i in range(len(nodes)) if i not in parent]
            if roots:
                self.note("NO_SCENE", INFO, "asset defines no scene; all root nodes are imported")

        scene = Scene(name or self.path.stem.strip() or "Imported Scene")
        self.importer = MaterialImporter(self, scene)
        objects: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        stack: list[tuple[int, SceneObject | None]] = [(r, None) for r in reversed(roots)]
        while stack:
            index, parent_obj = stack.pop()
            node, where = self.item("nodes", index, "/nodes")
            obj = self.create_object(scene, node, index, where, parent_obj, used_ids, objects)
            for child in reversed(node.get("children", [])):
                stack.append((child, obj))
        unused = self.importer.unused()
        if unused:
            self.note(
                "MATERIALS_UNUSED",
                INFO,
                f"{unused} materials are not used by imported meshes and were not imported",
                "/materials",
                unused,
            )
        skipped = len(nodes) - len(objects)
        if skipped:
            self.note(
                "NODES_NOT_IN_SCENE",
                WARNING,
                f"{skipped} nodes are not part of the imported scene and were skipped",
                "/nodes",
                skipped,
            )
        return ImportResult(
            scene,
            self.path,
            self.format,
            asset.get("generator") if isinstance(asset.get("generator"), str) else None,
            objects,
            self.diagnostics,
        )

    def create_object(
        self,
        scene: Scene,
        node: dict[str, Any],
        index: int,
        where: str,
        parent_obj: SceneObject | None,
        used_ids: set[str],
        objects: list[dict[str, Any]],
    ) -> SceneObject:
        transform = self.transform(node, where)
        mesh = self.mesh(node["mesh"], f"{where}/mesh") if "mesh" in node else None
        if "skin" in node:
            self.lossy(
                "SKIN_IGNORED",
                "skinning is not supported; the mesh is imported unskinned in its bind pose",
                f"{where}/skin",
            )
        if "camera" in node:
            self.note("NODE_CAMERA_IGNORED", INFO, "camera on node not imported", f"{where}/camera")

        raw_name = node.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            base = raw_name.strip()
        else:  # unnamed node: fall back to its mesh's name, as Blender does
            gmesh = self.array("meshes")[node["mesh"]] if "mesh" in node else None
            mesh_name = gmesh.get("name") if isinstance(gmesh, dict) else None
            base = mesh_name.strip() if isinstance(mesh_name, str) and mesh_name.strip() else "Node"
        name = scene.unique_name(base)
        if isinstance(raw_name, str) and raw_name.strip() and name != raw_name:
            self.note(
                "NAME_CHANGED",
                INFO,
                f"node name {raw_name!r} imported as {name!r} "
                "(object names must be unique and non-blank)",
                f"{where}/name",
            )

        object_id, source = None, "generated"
        extras = node.get("extras")
        stored = extras.get(EXTRAS_KEY) if isinstance(extras, dict) else None
        if stored is not None:
            candidate = stored.get("id") if isinstance(stored, dict) else None
            if isinstance(candidate, str) and candidate.strip() and candidate not in used_ids:
                object_id, source = candidate, "extras"
            else:
                self.note(
                    "STORED_ID_REJECTED",
                    WARNING,
                    "extras.aether3d.id is missing, invalid or duplicated; a new id was generated",
                    f"{where}/extras",
                )
        obj = scene.add_object(name, mesh=mesh, parent=parent_obj, object_id=object_id)
        obj.transform = transform
        if mesh is not None and node["mesh"] in self.mesh_materials:
            materials, grouped = self.mesh_materials[node["mesh"]]
            assign(
                obj,
                {g: self.importer.material(m, f"{where}/mesh") for g, m in materials.items()},
                grouped,
            )
        used_ids.add(obj.id)
        objects.append({"node": index, "id": obj.id, "name": obj.name, "id_source": source})
        return obj


def input_format(path: str | os.PathLike[str]) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in (".glb", ".gltf"):
        raise ValidationError(
            f"unsupported import format {suffix or '(no extension)'!r}; supported: .glb, .gltf "
            "(use Scene.load() for .aether projects)"
        )
    return suffix[1:]


def import_gltf(
    path: str | os.PathLike[str],
    *,
    allow_lossy: bool = False,
    limits: ImportLimits | None = None,
) -> ImportResult:
    """Import a ``.gltf`` or ``.glb`` file as a new :class:`Scene`.

    Returns an :class:`ImportResult` with the scene, a node-to-object map and
    diagnostics. Raises :class:`FileNotFoundError` if ``path`` does not exist,
    :class:`AssetFormatError` for malformed files, :class:`ResourceLimitError`
    when ``limits`` are exceeded, and :class:`UnsupportedFeatureError` when
    the asset needs unsupported extensions or -- unless ``allow_lossy=True`` --
    contains content that cannot be imported faithfully.
    """
    input_format(path)
    reader = _Reader(Path(path), limits or ImportLimits(), allow_lossy)
    reader.load()
    try:
        result = reader.build()
    except RecursionError:
        raise AssetFormatError(
            "asset structure is nested too deeply", None, "INVALID_GLTF"
        ) from None
    except AetherError:
        raise
    except (TypeError, KeyError, AttributeError, ValueError, IndexError, OverflowError) as exc:
        # Malformed values in places the structural checks above do not cover.
        raise AssetFormatError(f"malformed glTF: {exc}", None, "INVALID_GLTF") from None
    lossy = [d for d in reader.diagnostics if d.lossy]
    if lossy and not allow_lossy:
        raise UnsupportedFeatureError(
            "the asset contains content that cannot be imported faithfully: "
            + ", ".join(sorted({d.code for d in lossy}))
            + "; pass allow_lossy=True to import the rest and drop/approximate it",
            lossy,
        )
    return result
