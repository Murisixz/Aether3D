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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import history
from ._fileio import write_atomic
from ._version import __version__
from .errors import AetherError, ProjectFormatError, RenderError, ValidationError
from .mesh import MAX_GROUP, Mesh
from .procedural.params import value_from_json, value_to_json
from .rendering.camera import Camera
from .rendering.light import Light
from .rendering.settings import Appearance, RenderSettings, World
from .scene import Scene, _check_metadata, _check_tags, _owned
from .transform import Transform

FORMAT_ID = "aether3d.project"
"""Value of the top-level ``format`` field identifying an Aether3D project."""

FORMAT_VERSION = 13
"""Version written by this release. Files with a higher version are rejected.

* Version 2 (Aether3D 0.4.0) adds procedural geometry: ``scene.parameters``,
  ``scene.rules``, ``scene.geometry`` and a ``geometry`` reference on every object.
* Version 3 (Aether3D 0.5.0) adds rendering: ``scene.active_camera``,
  ``scene.world``, ``scene.render`` and, on every object, ``camera``,
  ``light``, ``appearance`` and ``visible``.
* Version 4 (Aether3D 0.7.0) adds materials: ``scene.materials``,
  ``scene.textures``, per-triangle ``groups`` on meshes and, on every object,
  ``material`` and ``group_materials``.
* Version 5 (Aether3D 0.8.0) adds path-tracing data: ``scene.world.environment``
  and the light sizes ``radius`` and ``angle``.
* Version 6 (Aether3D 0.10.0) adds ``tags`` and ``metadata`` on every object.
* Version 7 (Aether3D 0.14.0) adds painted textures (``"type": "painted"``).
* Version 8 (Aether3D 0.15.0) adds the CAD document: ``scene.cad`` with its settings
  and items (parameters, frames, sketches, parts, assemblies, building elements,
  drawings) and their links to the objects that show them.
* Version 9 (Aether3D 0.17.0) adds the animation document: ``scene.animation`` with the
  timeline settings and its items (clips, bindings).
* Version 10 (Aether3D 0.18.0) adds procedural worlds (``scene.worlds``: definitions and
  block edits), the outdoor atmosphere (``scene.world.atmosphere``) and vertex colours on
  meshes (``colors``).
* Version 11 (Aether3D 0.19.0) extends rigs (joint roles, helper joints, swing-twist and
  hinge limits, controls, correctives, landmarks, info) and skins (blended LBS/DQS
  skinning, weight locks).
* Version 12 (Aether3D 0.19.1, the Stage 18 revision) adds a joint's weighting ``span``
  (the part of its bone its skin weights cover) and skins' surface-following normals and
  corrective smoothing.
* Version 13 (Aether3D 0.20.0) adds the camera's lens (sensor, shift, f-stop, focus,
  aperture), curves' ``smoothing`` of automatic handles, the curve modifiers ``cycles``,
  ``generator`` and ``function``, strips' ``blend_curve``, and Penner's in-and-out
  overshoot for ``back`` easing (v12 keys are rescaled: their curves keep their shapes).

Older files are upgraded on load by the functions in :data:`MIGRATIONS`.
"""

FILE_EXTENSION = ".aether"


def _migrate_v1_to_v2(document: dict[str, Any]) -> dict[str, Any]:
    """v1 -> v2: no procedural data; every object gets ``"geometry": null``."""
    doc = dict(document)
    scene = doc.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        scene.setdefault("parameters", [])
        scene.setdefault("rules", [])
        scene.setdefault("geometry", [])
        objects = scene.get("objects")
        if isinstance(objects, list):
            scene["objects"] = [
                {**o, "geometry": None} if isinstance(o, dict) and "geometry" not in o else o
                for o in objects
            ]
        doc["scene"] = scene
    return doc


def _migrate_v2_to_v3(document: dict[str, Any]) -> dict[str, Any]:
    """v2 -> v3: no cameras, lights or colors; default world and render settings."""
    from .rendering.settings import RenderSettings, World

    doc = dict(document)
    scene = doc.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        scene.setdefault("active_camera", None)
        scene.setdefault("world", World().to_dict())
        scene.setdefault("render", RenderSettings().to_dict())
        objects = scene.get("objects")
        if isinstance(objects, list):
            defaults = {"camera": None, "light": None, "appearance": None, "visible": True}
            scene["objects"] = [
                {**o, **{k: v for k, v in defaults.items() if k not in o}}
                if isinstance(o, dict)
                else o
                for o in objects
            ]
        doc["scene"] = scene
    return doc


def _migrate_v3_to_v4(document: dict[str, Any]) -> dict[str, Any]:
    """v3 -> v4: no materials or textures; objects keep their colors (no material)."""
    doc = dict(document)
    scene = doc.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        scene.setdefault("materials", [])
        scene.setdefault("textures", [])
        objects = scene.get("objects")
        if isinstance(objects, list):
            defaults = {"material": None, "group_materials": {}}
            scene["objects"] = [
                {**o, **{k: v for k, v in defaults.items() if k not in o}}
                if isinstance(o, dict)
                else o
                for o in objects
            ]
        doc["scene"] = scene
    return doc


def _migrate_v4_to_v5(document: dict[str, Any]) -> dict[str, Any]:
    """v4 -> v5: the default ("ambient") environment; lights without size."""
    from .rendering.settings import Environment

    doc = dict(document)
    scene = doc.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        world = scene.get("world")
        if isinstance(world, dict) and "environment" not in world:
            scene["world"] = {**world, "environment": Environment().to_dict()}
        objects = scene.get("objects")
        if isinstance(objects, list):
            upgraded = []
            for o in objects:
                light = o.get("light") if isinstance(o, dict) else None
                if isinstance(light, dict):
                    sizes = {k: 0.0 for k in ("radius", "angle") if k not in light}
                    o = {**o, "light": {**light, **sizes}}
                upgraded.append(o)
            scene["objects"] = upgraded
        doc["scene"] = scene
    return doc


def _migrate_v5_to_v6(document: dict[str, Any]) -> dict[str, Any]:
    """v5 -> v6: objects have no tags and no metadata (the reader's defaults)."""
    return dict(document)


def _migrate_v6_to_v7(document: dict[str, Any]) -> dict[str, Any]:
    """v6 -> v7: painted textures (``"type": "painted"``) exist; a v6 document has none."""
    return dict(document)


def _migrate_v7_to_v8(document: dict[str, Any]) -> dict[str, Any]:
    """v7 -> v8: an empty CAD document (default units and display tolerance)."""
    from .cad.document import DEFAULT_SETTINGS

    out = dict(document)
    scene = out.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        scene.setdefault("cad", {"settings": dict(DEFAULT_SETTINGS), "items": []})
        out["scene"] = scene
    return out


def _migrate_v8_to_v9(document: dict[str, Any]) -> dict[str, Any]:
    """v8 -> v9: an empty animation document (default timeline)."""
    from .animation.system import DEFAULT_SETTINGS

    out = dict(document)
    scene = out.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        scene.setdefault(
            "animation", {"settings": json.loads(json.dumps(DEFAULT_SETTINGS)), "items": []}
        )
        out["scene"] = scene
    return out


def _migrate_v9_to_v10(document: dict[str, Any]) -> dict[str, Any]:
    """v9 -> v10: no worlds; the default (disabled) atmosphere; meshes have no colours."""
    from .rendering.settings import Atmosphere

    out = dict(document)
    scene = out.get("scene")
    if isinstance(scene, dict):
        scene = dict(scene)
        scene.setdefault("worlds", [])
        world = scene.get("world")
        if isinstance(world, dict):
            world = dict(world)
            world.setdefault("atmosphere", Atmosphere().to_dict())
            scene["world"] = world
        out["scene"] = scene
    return out


def _migrate_v10_to_v11(document: dict[str, Any]) -> dict[str, Any]:
    """v10 -> v11: rigs may have roles, helper joints, swing-twist and hinge limits,
    controls, correctives, landmarks and info; skins may blend LBS and DQS and keep weight
    locks. A v10 document has none of them: nothing changes."""
    return dict(document)


def _migrate_v11_to_v12(document: dict[str, Any]) -> dict[str, Any]:
    """v11 -> v12: joints may give a weighting span, skins surface-following normals and
    corrective smoothing. A v11 document has none of them (whole bones, rest normals
    carried by the joints, no smoothing): nothing changes."""
    return dict(document)


def _migrate_v12_to_v13(document: dict[str, Any]) -> dict[str, Any]:
    """v12 -> v13 (0.20.0): cameras get the lens (a pinhole: every render as before); back
    easing in and out uses Penner's 1.525 x overshoot, so keys of v12 documents get their
    ``back`` divided by 1.525 (their curves keep their shapes); curves without a
    ``smoothing`` keep Aether3D's local automatic handles (nothing to write)."""
    import copy

    from .animation.easing import BACK_IN_OUT, DEFAULT_BACK
    from .rendering.camera import LENS_DEFAULTS

    doc = copy.deepcopy(document)
    scene = doc.get("scene")
    if not isinstance(scene, dict):
        return doc
    for obj in scene.get("objects", []) if isinstance(scene.get("objects"), list) else []:
        cam = obj.get("camera") if isinstance(obj, dict) else None
        if isinstance(cam, dict):
            for key, value in LENS_DEFAULTS.items():
                cam.setdefault(key, value)

    def column(keys: dict[str, Any], name: str, n: int, default: Any) -> list[Any]:
        value = keys.get(name, default)
        return list(value) if isinstance(value, list) else [value] * n

    def fix_curve(curve: dict[str, Any]) -> None:
        keys = curve.get("keys")
        if not isinstance(keys, dict) or not isinstance(keys.get("time"), list):
            return
        n = len(keys["time"])
        interp = column(keys, "interpolation", n, "bezier")
        ease = column(keys, "ease", n, "in_out")
        back = column(keys, "back", n, DEFAULT_BACK)
        if len(interp) != n or len(ease) != n or len(back) != n:
            return  # (malformed: the reader reports it)
        hit = False
        for i in range(n):
            if interp[i] == "back" and ease[i] == "in_out" and isinstance(back[i], (int, float)):
                back[i] = back[i] / BACK_IN_OUT
                hit = True
        if hit:
            keys["back"] = back[0] if all(b == back[0] for b in back) else back

    def fix_timing(track: dict[str, Any]) -> None:
        keys = track.get("keys")
        timing = keys.get("timing") if isinstance(keys, dict) else None
        if not isinstance(timing, list):
            return
        for t in timing:
            if (
                isinstance(t, dict)
                and t.get("easing") == "back"
                and t.get("ease", "in_out") == "in_out"
            ):
                t["back"] = t.get("back", DEFAULT_BACK) / BACK_IN_OUT

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            keys = value.get("keys")
            if isinstance(keys, dict) and isinstance(keys.get("time"), list):
                if "rotation" in keys:
                    fix_timing(value)
                else:
                    fix_curve(value)
                return
            for k, v in value.items():
                if k != "events":  # event data is the user's own JSON
                    walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    animation = scene.get("animation")
    if isinstance(animation, dict):
        for item in animation.get("items", []) if isinstance(animation.get("items"), list) else []:
            if isinstance(item, dict):
                walk(item.get("definition"))
    return doc


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
    5: _migrate_v5_to_v6,
    6: _migrate_v6_to_v7,
    7: _migrate_v7_to_v8,
    8: _migrate_v8_to_v9,
    9: _migrate_v9_to_v10,
    10: _migrate_v10_to_v11,
    11: _migrate_v11_to_v12,
    12: _migrate_v12_to_v13,
}
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
    if mesh.groups is not None:
        data["groups"] = mesh.groups.tolist()
    if mesh.colors is not None:
        data["colors"] = _float32_list(mesh.colors)
    return data


def scene_to_dict(
    scene: Scene, project_dir: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Serialize ``scene`` into a JSON-compatible project document.

    Image texture files are written as paths relative to ``project_dir`` (the
    directory the project is saved in; default: the scene's own project
    directory, else the current directory); a file outside it raises
    ``MaterialError("TEXTURE_OUTSIDE_PROJECT")`` -- embed it or copy it there.
    """
    base = Path(project_dir) if project_dir is not None else scene.materials.base_dir
    library = scene.materials.to_dict(Path(base if base is not None else Path.cwd()).absolute())
    mesh_index: dict[int, int] = {}  # id(mesh) -> position in "meshes"
    meshes: list[dict[str, Any]] = []

    def index_of(mesh: Mesh) -> int:
        key = id(mesh)
        if key not in mesh_index:
            mesh_index[key] = len(meshes)
            meshes.append(_mesh_to_dict(mesh))
        return mesh_index[key]

    objects: list[dict[str, Any]] = []
    for obj in scene:  # depth-first: parents always precede their children
        # Procedural objects store their definition, never an evaluated mesh.
        mesh_ref = None if obj._mesh is None or obj._geometry is not None else index_of(obj._mesh)
        t = obj.transform
        objects.append(
            {
                "id": obj.id,
                "name": obj.name,
                "tags": list(obj._tags),
                "metadata": json.loads(obj._metadata),
                "parent": None if obj.parent is None else obj.parent.id,
                "transform": {
                    "position": list(t.position),
                    "rotation": list(t.rotation),
                    "scale": list(t.scale),
                },
                "mesh": mesh_ref,
                "geometry": obj._geometry,
                "camera": None if obj._camera is None else obj._camera.to_dict(),
                "light": None if obj._light is None else obj._light.to_dict(),
                "appearance": None if obj._appearance.is_default else obj._appearance.to_dict(),
                "visible": obj._visible,
                "material": obj._material,
                "group_materials": {str(g): m for g, m in sorted(obj._group_materials.items())},
            }
        )
    system = scene._procedural
    parameters = [p.to_json() for p in system.parameters.values()] if system else []
    rules = [r.to_json() for r in system.rules.values()] if system else []
    geometry = []
    for g in system.geometries if system else []:
        geometry.append(
            {
                "id": g.id,
                "name": g.name,
                "parameters": [p.to_json() for p in g.parameters.values()],
                "rules": [r.to_json() for r in g.rules.values()],
                "source": g.source.to_json(index_of),
                "modifiers": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "type": m.type,
                        "enabled": m.enabled,
                        "params": value_to_json(m.params),
                    }
                    for m in g.modifiers
                ],
            }
        )
    return {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "generator": f"aether3d {__version__}",
        "scene": {
            "name": scene.name,
            "active_camera": scene._active_camera,
            "world": scene.world.to_dict(),
            "render": scene.render_settings.to_dict(),
            "parameters": parameters,
            "rules": rules,
            "geometry": geometry,
            "materials": library["materials"],
            "textures": library["textures"],
            "objects": objects,
            "cad": _cad_to_dict(scene),
            "animation": _animation_to_dict(scene),
            "worlds": [] if scene._worlds is None else scene._worlds.to_json(),
        },
        "meshes": meshes,
    }


def _cad_to_dict(scene: Scene) -> dict[str, Any]:
    if scene._cad is None:
        from .cad.document import DEFAULT_SETTINGS

        return {"settings": json.loads(json.dumps(DEFAULT_SETTINGS)), "items": []}
    return scene._cad.to_json()


def _animation_to_dict(scene: Scene) -> dict[str, Any]:
    if scene._animation is None:
        from .animation.system import DEFAULT_SETTINGS

        return {"settings": json.loads(json.dumps(DEFAULT_SETTINGS)), "items": []}
    return scene._animation.to_json()


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
    Texture files are stored relative to the project's directory (see
    :func:`scene_to_dict`); nothing is written if one cannot be.
    """
    project_dir = Path(path).absolute().parent
    data = dumps(scene_to_dict(scene, project_dir)).encode("utf-8")
    written = write_atomic(path, data)
    scene.materials.base_dir = project_dir
    return written


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
    data = _object(value, loc, ("positions", "triangles"), ("normals", "uvs", "groups", "colors"))
    positions = _rows(data["positions"], 3, f"{loc}.positions")
    triangles = _rows(data["triangles"], 3, f"{loc}.triangles", integer=True)
    normals = _rows(data["normals"], 3, f"{loc}.normals") if "normals" in data else None
    uvs = _rows(data["uvs"], 2, f"{loc}.uvs") if "uvs" in data else None
    colors = _rows(data["colors"], 4, f"{loc}.colors") if "colors" in data else None
    groups = None
    if "groups" in data:
        groups = _list(data["groups"], f"{loc}.groups")
        if not all(isinstance(g, int) and not isinstance(g, bool) for g in groups):
            raise ProjectFormatError("expected an array of integers", f"{loc}.groups")
    try:
        return Mesh(
            np.array(positions, dtype=np.float64).reshape(-1, 3),
            np.array(triangles, dtype=np.int64).reshape(-1, 3),
            normals=None if normals is None else np.array(normals, dtype=np.float64).reshape(-1, 3),
            uvs=None if uvs is None else np.array(uvs, dtype=np.float64).reshape(-1, 2),
            groups=None if groups is None else np.array(groups, dtype=np.int64),
            colors=None if colors is None else np.array(colors, dtype=np.float64).reshape(-1, 4),
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


def scene_from_dict(
    document: dict[str, Any], project_dir: str | os.PathLike[str] | None = None
) -> Scene:
    """Build a :class:`Scene` from a project document, validating it fully.

    Texture paths are resolved inside ``project_dir`` (the project file's
    directory; default: the current directory) and may not leave it.
    """
    doc = _object(
        migrate(document), "", ("format", "format_version", "scene", "meshes"), ("generator",)
    )
    meshes = [
        _mesh_from_dict(m, f"meshes[{i}]") for i, m in enumerate(_list(doc["meshes"], "meshes"))
    ]
    scene_data = _object(
        doc["scene"],
        "scene",
        (
            "name",
            "active_camera",
            "world",
            "render",
            "parameters",
            "rules",
            "geometry",
            "materials",
            "textures",
            "objects",
            "cad",
            "animation",
            "worlds",
        ),
    )
    scene = Scene(_string(scene_data["name"], "scene.name"))
    # Built through the API, but loading is not an edit: the history starts empty.
    with scene.history._suspended():
        _read_scene(scene, scene_data, meshes, project_dir)
    return scene


def _read_scene(
    scene: Scene,
    scene_data: dict[str, Any],
    meshes: list[Mesh],
    project_dir: str | os.PathLike[str] | None,
) -> None:
    scene._world = _owned(
        _render_value(World, scene_data["world"], "scene.world"), scene, history.WORLD
    )
    scene._render_settings = _owned(
        _render_value(RenderSettings, scene_data["render"], "scene.render"), scene, history.RENDER
    )

    # Pass 1: parse and check every object record on its own.
    records: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    for i, raw in enumerate(_list(scene_data["objects"], "scene.objects")):
        loc = f"scene.objects[{i}]"
        data = _object(
            raw,
            loc,
            ("id", "name", "parent", "transform", "mesh", "geometry"),
            # optional: null/null/null/true/null/{}
            (
                "camera",
                "light",
                "appearance",
                "visible",
                "material",
                "group_materials",
                "tags",
                "metadata",
            ),
        )
        data = {
            "camera": None,
            "light": None,
            "appearance": None,
            "visible": True,
            "material": None,
            "group_materials": {},
            "tags": [],
            "metadata": {},
            **data,
        }
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
        geometry_ref = data["geometry"]
        if geometry_ref is not None:
            _string(geometry_ref, f"{loc}.geometry")
            if mesh_ref is not None:
                raise ProjectFormatError("an object cannot have both a mesh and a geometry", loc)
        camera = (
            None
            if data["camera"] is None
            else _render_value(Camera, data["camera"], f"{loc}.camera")
        )
        light = (
            None if data["light"] is None else _render_value(Light, data["light"], f"{loc}.light")
        )
        kinds = [
            k
            for k, v in (
                ("mesh", mesh_ref),
                ("geometry", geometry_ref),
                ("camera", camera),
                ("light", light),
            )
            if v is not None
        ]
        if len(kinds) > 1:
            raise ProjectFormatError(
                f"an object can have only one of mesh, geometry, camera and light "
                f"(got {', '.join(kinds)})",
                loc,
            )
        appearance = (
            Appearance()
            if data["appearance"] is None
            else _render_value(Appearance, data["appearance"], f"{loc}.appearance")
        )
        if not isinstance(data["visible"], bool):
            raise ProjectFormatError("must be true or false", f"{loc}.visible")
        material_ref = data["material"]
        if material_ref is not None:
            _string(material_ref, f"{loc}.material")
        group_materials = _group_materials(data["group_materials"], f"{loc}.group_materials")
        if (material_ref is not None or group_materials) and (
            camera is not None or light is not None
        ):
            raise ProjectFormatError("cameras and lights cannot have materials", loc)
        records.append(
            {
                "material": material_ref,
                "group_materials": group_materials,
                "camera": camera,
                "light": light,
                "appearance": appearance,
                "visible": data["visible"],
                "tags": _annotation(_check_tags, data["tags"], f"{loc}.tags"),
                "metadata": _annotation(_check_metadata, data["metadata"], f"{loc}.metadata"),
                "geometry": geometry_ref,
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
        obj._geometry = record["geometry"]
        obj._camera = _owned(record["camera"], obj, history.CAMERA)
        obj._light = _owned(record["light"], obj, history.LIGHT)
        obj._appearance = _owned(record["appearance"], obj, history.APPEARANCE)
        obj._visible = record["visible"]
        obj._material = record["material"]
        obj._group_materials = record["group_materials"]
        obj._tags, obj._metadata = record["tags"], record["metadata"]
        stack.extend(reversed(children.get(record["id"], [])))

    if len(scene) != len(records):
        orphan = next(r for r in records if r["id"] not in scene)
        raise ProjectFormatError("object is part of a parent cycle", f"{orphan['loc']}.parent")
    active = scene_data["active_camera"]
    if active is not None:
        _string(active, "scene.active_camera")
        if active not in seen_ids or scene.get(active).camera is None:
            raise ProjectFormatError(
                f"must be null or the id of a camera object, got {active!r}", "scene.active_camera"
            )
        scene._active_camera = active
    _read_procedural(scene, scene_data, meshes)
    base = Path(project_dir) if project_dir is not None else Path.cwd()
    scene.materials.load_records(
        _list(scene_data["materials"], "scene.materials"),
        _list(scene_data["textures"], "scene.textures"),
        base.absolute(),
    )
    library = scene.materials
    for record in records:
        refs = ([record["material"]] if record["material"] is not None else []) + list(
            record["group_materials"].values()
        )
        for ref in refs:
            if ref not in library._materials:
                raise ProjectFormatError(
                    f"material {ref!r} does not exist (materials are referenced by id)",
                    f"{record['loc']}.material",
                )
    if project_dir is not None:
        library.base_dir = base.absolute()
    _read_cad(scene, scene_data["cad"])
    _read_animation(scene, scene_data["animation"])
    _read_worlds(scene, scene_data["worlds"])


def _read_worlds(scene: Scene, data: Any) -> None:
    if not isinstance(data, list):
        raise ProjectFormatError(f"expected an array, got {_type_name(data)}", "scene.worlds")
    if not data:
        return
    try:
        scene.worlds._load(data, "scene.worlds")
    except AetherError as exc:
        raise ProjectFormatError(
            getattr(exc, "message", str(exc)), getattr(exc, "location", None) or "scene.worlds"
        ) from None


def _read_animation(scene: Scene, data: Any) -> None:
    from .animation.system import DEFAULT_SETTINGS

    if not isinstance(data, dict):
        raise ProjectFormatError(f"expected an object, got {_type_name(data)}", "scene.animation")
    if not data.get("items") and data.get("settings") in (None, DEFAULT_SETTINGS):
        return
    try:
        scene.animation._load(data, "scene.animation")
    except AetherError as exc:
        raise ProjectFormatError(
            getattr(exc, "message", str(exc)), getattr(exc, "location", None) or "scene.animation"
        ) from None


def _read_cad(scene: Scene, data: Any) -> None:
    if not isinstance(data, dict):
        raise ProjectFormatError(f"expected an object, got {_type_name(data)}", "scene.cad")
    if not data.get("items") and data.get("settings") in (None, {}):
        return
    from .cad.document import DEFAULT_SETTINGS

    if not data.get("items") and data.get("settings") == DEFAULT_SETTINGS:
        return
    doc = scene.cad
    doc._load(data, "scene.cad")
    for item in doc.items():
        for oid in item._links:
            if oid not in scene._objects:
                raise ProjectFormatError(
                    f"CAD item {item.name!r} links to a missing object {oid!r}", "scene.cad"
                )


def _annotation(check: Callable[[Any], Any], value: Any, loc: str) -> Any:
    if loc.endswith(".tags") and isinstance(value, str):
        raise ProjectFormatError("expected an array of strings", loc)
    try:
        return check(value)
    except ValidationError as exc:
        raise ProjectFormatError(str(exc), loc) from None


def _group_materials(value: Any, loc: str) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ProjectFormatError(f"expected an object, got {_type_name(value)}", loc)
    out: dict[int, str] = {}
    for key, ref in value.items():
        if not (isinstance(key, str) and key.isdigit() and len(key) <= 5) or int(key) > MAX_GROUP:
            raise ProjectFormatError(
                f"group keys must be integers in [0, {MAX_GROUP}] written as strings, got {key!r}",
                loc,
            )
        out[int(key)] = _string(ref, f"{loc}.{key}")
    return out


def _read_procedural(scene: Scene, scene_data: dict[str, Any], meshes: list[Mesh]) -> None:
    """Rebuild parameters and geometry nodes; validate structure (evaluation stays lazy)."""
    from .procedural.graph import Modifier, ProceduralGeometry, Source
    from .procedural.modifiers import MODIFIER_TYPES
    from .procedural.params import ProceduralError, make_parameter, make_rule
    from .procedural.primitives import PRIMITIVES

    parameter_keys = ("name", "type", "value")
    parameter_optional = ("min", "max", "choices", "description", "unit")

    def parameter(raw: Any, loc: str):
        data = _object(raw, loc, parameter_keys, parameter_optional)
        try:
            return make_parameter(
                _string(data["name"], f"{loc}.name"),
                value_from_json(data["value"]),
                type=data["type"],
                min=data.get("min"),
                max=data.get("max"),
                choices=data.get("choices"),
                description=data.get("description", ""),
                unit=data.get("unit", ""),
            )
        except ProceduralError as exc:
            raise ProjectFormatError(f"[{exc.code}] {exc.message}", loc) from None

    def rules(raw_list: Any, loc: str, target: dict) -> None:
        for k, raw in enumerate(_list(raw_list, loc)):
            rloc = f"{loc}[{k}]"
            data = _object(raw, rloc, ("name", "expr"), ("message",))
            try:
                rule = make_rule(
                    _string(data["name"], f"{rloc}.name"),
                    _string(data["expr"], f"{rloc}.expr"),
                    data.get("message", ""),
                )
            except ProceduralError as exc:
                raise ProjectFormatError(f"[{exc.code}] {exc.message}", rloc) from None
            if rule.name in target:
                raise ProjectFormatError(f"duplicate rule {rule.name!r}", rloc)
            target[rule.name] = rule

    system = scene.procedural
    rules(scene_data["rules"], "scene.rules", system.rules)
    for i, raw in enumerate(_list(scene_data["parameters"], "scene.parameters")):
        p = parameter(raw, f"scene.parameters[{i}]")
        if p.name in system.parameters:
            raise ProjectFormatError(
                f"duplicate parameter {p.name!r}", f"scene.parameters[{i}].name"
            )
        system.parameters[p.name] = p
    names: set[str] = set()
    for i, raw in enumerate(_list(scene_data["geometry"], "scene.geometry")):
        loc = f"scene.geometry[{i}]"
        data = _object(raw, loc, ("id", "name", "parameters", "rules", "source", "modifiers"))
        gid, name = _string(data["id"], f"{loc}.id"), _string(data["name"], f"{loc}.name")
        if gid in system._geometries or name in names:
            raise ProjectFormatError(f"duplicate geometry id or name ({gid!r}, {name!r})", loc)
        names.add(name)
        src = _object(
            data["source"],
            f"{loc}.source",
            ("type",),
            ("primitive", "params", "mesh", "geometry", "parameters", "after"),
        )
        kind = src["type"]
        sloc = f"{loc}.source"
        if kind == "primitive":
            _object(src, sloc, ("type", "primitive", "params"))
            if src["primitive"] not in PRIMITIVES:
                raise ProjectFormatError(
                    f"unknown primitive {src['primitive']!r}", f"{sloc}.primitive"
                )
            source = Source(
                "primitive",
                primitive=src["primitive"],
                params=_json_values(src["params"], f"{sloc}.params"),
            )
        elif kind == "mesh":
            _object(src, sloc, ("type", "mesh"))
            index = src["mesh"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(meshes)
            ):
                raise ProjectFormatError("must be an index into meshes", f"{sloc}.mesh")
            source = Source("mesh", mesh=meshes[index])
        elif kind == "geometry":
            _object(src, sloc, ("type", "geometry", "parameters"), ("after",))
            source = Source(
                "geometry",
                geometry=_string(src["geometry"], f"{sloc}.geometry"),
                overrides=_json_values(src["parameters"], f"{sloc}.parameters"),
                after=src.get("after"),
            )
        else:
            raise ProjectFormatError(f"unknown source type {kind!r}", f"{sloc}.type")
        g = ProceduralGeometry(system, gid, name, source)
        for k, rawp in enumerate(_list(data["parameters"], f"{loc}.parameters")):
            p = parameter(rawp, f"{loc}.parameters[{k}]")
            if p.name in g.parameters:
                raise ProjectFormatError(
                    f"duplicate parameter {p.name!r}", f"{loc}.parameters[{k}]"
                )
            g.parameters[p.name] = p
        rules(data["rules"], f"{loc}.rules", g.rules)
        seen_mods: set[str] = set()
        for k, rawm in enumerate(_list(data["modifiers"], f"{loc}.modifiers")):
            mloc = f"{loc}.modifiers[{k}]"
            m = _object(rawm, mloc, ("id", "name", "type", "enabled", "params"))
            if m["type"] not in MODIFIER_TYPES:
                raise ProjectFormatError(f"unknown modifier type {m['type']!r}", f"{mloc}.type")
            if not isinstance(m["enabled"], bool):
                raise ProjectFormatError("enabled must be true or false", f"{mloc}.enabled")
            mid, mname = _string(m["id"], f"{mloc}.id"), _string(m["name"], f"{mloc}.name")
            if mid in seen_mods or any(x.name == mname for x in g.modifiers):
                raise ProjectFormatError("duplicate modifier id or name", mloc)
            seen_mods.add(mid)
            g.modifiers.append(
                Modifier(
                    g,
                    mid,
                    m["type"],
                    mname,
                    m["enabled"],
                    _json_values(m["params"], f"{mloc}.params"),
                )
            )
        system._geometries[gid] = g
    for g in system.geometries:
        if g.source.kind == "geometry":
            other = system._geometries.get(g.source.geometry)
            if other is None:
                raise ProjectFormatError(
                    f"source geometry {g.source.geometry!r} is missing",
                    f"scene.geometry[{g.name}].source.geometry",
                )
            if g.source.after is not None and not any(
                m.id == g.source.after for m in other.modifiers
            ):
                raise ProjectFormatError(
                    "after must name a modifier id of the source geometry",
                    f"scene.geometry[{g.name}].source.after",
                )
    try:
        system._check_all(evaluate=False)
    except ProceduralError as exc:
        where = f" at {exc.location}" if exc.location else ""
        raise ProjectFormatError(f"[{exc.code}] {exc.message}{where}", "scene.geometry") from None


def _render_value(kind: Any, raw: Any, loc: str) -> Any:
    """Build a camera/light/appearance/world/settings value; errors become format errors."""
    try:
        return kind.from_dict(raw, loc)
    except RenderError as exc:
        raise ProjectFormatError(f"[{exc.code}] {exc.message}", exc.location or loc) from None


def _json_values(raw: Any, loc: str) -> dict[str, Any]:
    from .procedural.params import ProceduralError

    if not isinstance(raw, dict):
        raise ProjectFormatError(f"expected an object, got {_type_name(raw)}", loc)
    try:
        return {k: value_from_json(v) for k, v in raw.items()}
    except ProceduralError as exc:
        raise ProjectFormatError(f"[{exc.code}] {exc.message}", loc) from None


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
    return scene_from_dict(loads(text), Path(path).absolute().parent)
