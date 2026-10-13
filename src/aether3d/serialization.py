"""Reading and writing the native ``.aether`` project format.

An ``.aether`` file is a UTF-8 JSON document. The format is versioned with a
single integer, ``format_version``; see ``docs/FILE_FORMAT.md`` for the full
specification. Readers are strict: unknown keys, wrong types and dangling
references are rejected with a :class:`ProjectFormatError` whose ``location``
points at the offending element (e.g. ``scene.objects[1].transform.scale``).

Evolving the format: bump :data:`FORMAT_VERSION`, update the writer, and
register a function in :data:`MIGRATIONS` that upgrades a document from the
previous version. Older files are then upgraded step by step on load.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .errors import AetherError, ProjectFormatError, ValidationError
from .mesh import Mesh
from .scene import Scene
from .transform import Transform

FORMAT_ID = "aether3d.project"
"""Value of the top-level ``format`` field identifying an Aether3D project."""

FORMAT_VERSION = 1
"""Version written by this release. Files with a higher version are rejected."""

FILE_EXTENSION = ".aether"

MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""``MIGRATIONS[n]`` upgrades a version-``n`` document to version ``n + 1``."""


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _float32_list(array: np.ndarray) -> list[list[float]]:
    # str() of a float32 gives the shortest text that round-trips to the same
    # float32, so files stay readable ("0.1" rather than "0.10000000149011612").
    return [[float(str(v)) for v in row] for row in array.astype(np.float32)]


def _mesh_to_dict(mesh: Mesh) -> dict[str, Any]:
    data: dict[str, Any] = {"positions": _float32_list(mesh.positions)}
    if mesh.normals is not None:
        data["normals"] = _float32_list(mesh.normals)
    if mesh.uvs is not None:
        data["uvs"] = _float32_list(mesh.uvs)
    data["triangles"] = mesh.triangles.tolist()
    return data


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    """Serialize ``scene`` into a JSON-compatible project document."""
    mesh_index: dict[int, int] = {}  # id(mesh) -> position in "meshes"
    meshes: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for obj in scene:  # depth-first: parents always precede their children
        mesh_ref = None
        if obj.mesh is not None:
            key = id(obj.mesh)
            if key not in mesh_index:
                mesh_index[key] = len(meshes)
                meshes.append(_mesh_to_dict(obj.mesh))
            mesh_ref = mesh_index[key]
        t = obj.transform
        objects.append(
            {
                "id": obj.id,
                "name": obj.name,
                "parent": None if obj.parent is None else obj.parent.id,
                "transform": {
                    "position": list(t.position),
                    "rotation": list(t.rotation),
                    "scale": list(t.scale),
                },
                "mesh": mesh_ref,
            }
        )
    return {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "generator": f"aether3d {__version__}",
        "scene": {"name": scene.name, "objects": objects},
        "meshes": meshes,
    }


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def dumps(document: dict[str, Any]) -> str:
    """Format a project document as JSON text.

    Output is deterministic and diff-friendly: two-space indentation, with
    short numeric tuples (vectors, triangles) kept on a single line.
    """

    def emit(value: Any, depth: int) -> str:
        pad, inner = "  " * depth, "  " * (depth + 1)
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = [f"{inner}{json.dumps(k)}: {emit(v, depth + 1)}" for k, v in value.items()]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        if isinstance(value, list):
            if all(_is_scalar(v) for v in value):
                return "[" + ", ".join(emit(v, depth) for v in value) + "]"
            return "[\n" + ",\n".join(inner + emit(v, depth + 1) for v in value) + "\n" + pad + "]"
        return json.dumps(value, allow_nan=False, ensure_ascii=False)

    return emit(document, 0) + "\n"


def save_scene(scene: Scene, path: str | os.PathLike[str]) -> Path:
    """Write ``scene`` to ``path`` atomically and return the resolved path.

    The file is written to a temporary sibling first and then moved into
    place, so an interrupted save never leaves a half-written project.
    """
    target = Path(path)
    text = dumps(scene_to_dict(scene))
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent if target.parent.parts else "."
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target.resolve()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _object(
    value: Any, loc: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectFormatError(f"expected an object, got {_type_name(value)}", loc)
    missing = [key for key in required if key not in value]
    if missing:
        raise ProjectFormatError(f"missing required key(s): {', '.join(missing)}", loc)
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise ProjectFormatError(f"unknown key(s): {', '.join(unknown)}", loc)
    return value


def _list(value: Any, loc: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProjectFormatError(f"expected an array, got {_type_name(value)}", loc)
    return value


def _string(value: Any, loc: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectFormatError(f"expected a non-empty string, got {_type_name(value)}", loc)
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers(value: Any, count: int, loc: str) -> list[float]:
    items = _list(value, loc)
    if len(items) != count or not all(_is_number(v) for v in items):
        raise ProjectFormatError(f"expected an array of {count} numbers", loc)
    return items


def _rows(value: Any, width: int, loc: str, integer: bool = False) -> list[list[Any]]:
    rows = _list(value, loc)
    kind = "integers" if integer else "numbers"
    for i, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != width
            or not all(isinstance(v, int) if integer else _is_number(v) for v in row)
            or any(isinstance(v, bool) for v in row)
        ):
            raise ProjectFormatError(f"expected an array of {width} {kind}", f"{loc}[{i}]")
    return rows


def _type_name(value: Any) -> str:
    return {dict: "object", list: "array", str: "string", bool: "boolean"}.get(
        type(value), "null" if value is None else "number"
    )


def _mesh_from_dict(value: Any, loc: str) -> Mesh:
    data = _object(value, loc, ("positions", "triangles"), ("normals", "uvs"))
    positions = _rows(data["positions"], 3, f"{loc}.positions")
    triangles = _rows(data["triangles"], 3, f"{loc}.triangles", integer=True)
    normals = _rows(data["normals"], 3, f"{loc}.normals") if "normals" in data else None
    uvs = _rows(data["uvs"], 2, f"{loc}.uvs") if "uvs" in data else None
    try:
        return Mesh(
            np.array(positions, dtype=np.float64).reshape(-1, 3),
            np.array(triangles, dtype=np.int64).reshape(-1, 3),
            normals=None if normals is None else np.array(normals, dtype=np.float64).reshape(-1, 3),
            uvs=None if uvs is None else np.array(uvs, dtype=np.float64).reshape(-1, 2),
        )
    except OverflowError:
        raise ProjectFormatError("number out of range", loc) from None
    except AetherError as exc:
        raise ProjectFormatError(str(exc), loc) from None


def migrate(document: dict[str, Any]) -> dict[str, Any]:
    """Check the header of a raw document and upgrade it to :data:`FORMAT_VERSION`."""
    if not isinstance(document, dict):
        raise ProjectFormatError(
            f"expected a JSON object at the top level, got {_type_name(document)}"
        )
    if document.get("format") != FORMAT_ID:
        raise ProjectFormatError(f'not an Aether3D project (expected "format": "{FORMAT_ID}")')
    version = document.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProjectFormatError("format_version must be a positive integer", "format_version")
    if version > FORMAT_VERSION:
        raise ProjectFormatError(
            f"project uses format version {version}, but this release of Aether3D only "
            f"supports up to version {FORMAT_VERSION}; upgrade Aether3D to open it",
            "format_version",
        )
    if version < FORMAT_VERSION:
        document = dict(document)  # never modify the caller's document in place
    while version < FORMAT_VERSION:
        document = MIGRATIONS[version](document)
        version += 1
        document["format_version"] = version
    return document


def scene_from_dict(document: dict[str, Any]) -> Scene:
    """Build a :class:`Scene` from a project document, validating it fully."""
    doc = _object(
        migrate(document), "", ("format", "format_version", "scene", "meshes"), ("generator",)
    )
    meshes = [
        _mesh_from_dict(m, f"meshes[{i}]") for i, m in enumerate(_list(doc["meshes"], "meshes"))
    ]
    scene_data = _object(doc["scene"], "scene", ("name", "objects"))
    scene = Scene(_string(scene_data["name"], "scene.name"))

    # Pass 1: parse and check every object record on its own.
    records: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    for i, raw in enumerate(_list(scene_data["objects"], "scene.objects")):
        loc = f"scene.objects[{i}]"
        data = _object(raw, loc, ("id", "name", "parent", "transform", "mesh"))
        object_id = _string(data["id"], f"{loc}.id")
        name = _string(data["name"], f"{loc}.name")
        if object_id in seen_ids:
            raise ProjectFormatError(
                f"duplicate object id {object_id!r} (also at {seen_ids[object_id]})", f"{loc}.id"
            )
        if name in seen_names:
            raise ProjectFormatError(
                f"duplicate object name {name!r} (also at {seen_names[name]})", f"{loc}.name"
            )
        seen_ids[object_id], seen_names[name] = loc, loc
        parent = data["parent"]
        if parent is not None:
            _string(parent, f"{loc}.parent")
        mesh_ref = data["mesh"]
        if mesh_ref is not None and (
            not isinstance(mesh_ref, int)
            or isinstance(mesh_ref, bool)
            or not 0 <= mesh_ref < len(meshes)
        ):
            raise ProjectFormatError(
                f"must be null or an index into meshes (0..{len(meshes) - 1}), got {mesh_ref!r}",
                f"{loc}.mesh",
            )
        t = _object(data["transform"], f"{loc}.transform", ("position", "rotation", "scale"))
        position = _numbers(t["position"], 3, f"{loc}.transform.position")
        rotation = _numbers(t["rotation"], 4, f"{loc}.transform.rotation")
        scale = _numbers(t["scale"], 3, f"{loc}.transform.scale")
        try:
            transform = Transform(position, rotation, scale)
        except ValidationError as exc:
            raise ProjectFormatError(str(exc), f"{loc}.transform") from None
        records.append(
            {
                "loc": loc,
                "id": object_id,
                "name": name,
                "parent": parent,
                "transform": transform,
                "mesh": None if mesh_ref is None else meshes[mesh_ref],
            }
        )

    # Pass 2: link the hierarchy. Children keep their relative file order.
    children: dict[str | None, list[dict[str, Any]]] = {}
    for record in records:
        if record["parent"] is not None and record["parent"] not in seen_ids:
            raise ProjectFormatError(
                f"parent {record['parent']!r} does not exist", f"{record['loc']}.parent"
            )
        children.setdefault(record["parent"], []).append(record)

    stack = list(reversed(children.get(None, [])))
    while stack:
        record = stack.pop()
        obj = scene.add_object(
            record["name"], mesh=record["mesh"], parent=record["parent"], object_id=record["id"]
        )
        obj.transform = record["transform"]
        stack.extend(reversed(children.get(record["id"], [])))

    if len(scene) != len(records):
        orphan = next(r for r in records if r["id"] not in scene)
        raise ProjectFormatError("object is part of a parent cycle", f"{orphan['loc']}.parent")
    return scene


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectFormatError(f"duplicate key {key!r} in JSON object")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ProjectFormatError(f"{name} is not allowed; numbers must be finite")


def loads(text: str) -> dict[str, Any]:
    """Parse project JSON text (strict JSON: no NaN/Infinity, no duplicate keys)."""
    try:
        return json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ProjectFormatError(
            f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from None


def load_scene(path: str | os.PathLike[str]) -> Scene:
    """Read and validate a ``.aether`` file. Raises :class:`ProjectFormatError` if invalid."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectFormatError(f"file is not valid UTF-8 text: {exc.reason}") from None
    return scene_from_dict(loads(text))
