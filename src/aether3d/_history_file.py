"""A scene's history saved next to its project (``<project>.aether.history``).

The file lets a later process -- the command line above all -- undo and redo the
changes made in earlier ones. It is a JSON document::

    {
      "format": "aether3d.history", "format_version": 1, "generator": "aether3d 0.9.0",
      "project": "<SHA-256 of the project document it belongs to>",
      "position": 7, "serial": 9, "checkpoints": {"walls": 3},
      "max_entries": 256, "max_bytes": 268435456, "dropped": 0,
      "undo": [step, ...], "redo": [step, ...],
      "objects": [...], "materials": [...], "textures": [...],
      "meshes": [...], "blobs": [...]
    }

Each step is ``{"index", "parent", "label", "changes": [{"aspect", "target", "before",
"after"}]}``: the before and after states of every part of the scene it changed, in
the encodings below. ``objects``, ``materials`` and ``textures`` hold the full records
of entities the history refers to that are not in the project (removed ones, which
undo may bring back); ``meshes`` and ``blobs`` hold the meshes (project format) and
embedded image bytes (base64) the states refer to by index.

A history is applied to a project only if ``project`` matches the project's own
document (its fingerprint): a project changed by another program, or saved without
its history, keeps working, but its old history is not used. Loading checks a history
by replaying it -- every step undone and redone on a copy of the scene, with the
scene's invariants checked after each one -- and refuses it if the copy does not come
back to the same document.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import history as h
from ._fileio import write_atomic
from ._version import __version__
from .errors import AetherError, HistoryError

if TYPE_CHECKING:
    from .scene import Scene, SceneObject

FORMAT_ID = "aether3d.history"
FORMAT_VERSION = 1
SUFFIX = ".history"
"""``castle.aether`` keeps its history in ``castle.aether.history``."""

MAX_FILE_BYTES = 1 << 30

ASPECTS: dict[str, h.Aspect] = {
    "object.name": h.NAME,
    "object.transform": h.TRANSFORM,
    "object.visible": h.VISIBLE,
    "object.appearance": h.APPEARANCE,
    "object.camera": h.CAMERA,
    "object.light": h.LIGHT,
    "object.geometry": h.DATA,
    "object.material": h.MATERIALS,
    "object.parent": h.PARENT,
    "object.children": h.CHILDREN,
    "object.alive": h.ALIVE,
    "object.annotations": h.ANNOTATIONS,
    "scene.name": h.SCENE_NAME,
    "scene.roots": h.ROOTS,
    "scene.active_camera": h.ACTIVE_CAMERA,
    "scene.world": h.WORLD,
    "scene.render": h.RENDER,
    "library.materials": h.LIBRARY_MATERIALS,
    "library.textures": h.LIBRARY_TEXTURES,
    "material": h.MATERIAL,
    "texture": h.TEXTURE,
    "procedural": h.PROCEDURAL,
    "cad.item": h.CAD_ITEM,
    "cad.settings": h.CAD_SETTINGS,
    "animation.item": h.ANIMATION_ITEM,
    "animation.settings": h.ANIMATION_SETTINGS,
}
KEYS = {aspect: key for key, aspect in ASPECTS.items()}


def history_path(project: str | os.PathLike[str]) -> Path:
    return Path(str(project) + SUFFIX)


def fingerprint(scene: Scene, project_dir: Path) -> str:
    """SHA-256 of the scene's project document (its ``generator`` line excluded)."""
    from .serialization import scene_to_dict

    document = scene_to_dict(scene, project_dir)
    document.pop("generator", None)
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bad(message: str, location: str | None = None) -> HistoryError:
    return HistoryError("INVALID_HISTORY", message, location)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


class _Writer:
    def __init__(self, scene: Scene, project_dir: Path) -> None:
        self.scene = scene
        self.project_dir = project_dir
        self.meshes: list[dict[str, Any]] = []
        self.mesh_index: dict[int, int] = {}
        self.blobs: list[str] = []
        self.blob_index: dict[int, int] = {}
        self.detached_objects: dict[str, Any] = {}
        self.detached_materials: dict[str, Any] = {}
        self.detached_textures: dict[str, Any] = {}
        self.cad_items: dict[str, Any] = {}
        self.animation_items: dict[str, Any] = {}

    # -- references -------------------------------------------------------------------

    def mesh(self, mesh: Any) -> int | None:
        if mesh is None:
            return None
        from .serialization import _mesh_to_dict

        key = id(mesh)
        if key not in self.mesh_index:
            self.mesh_index[key] = len(self.meshes)
            self.meshes.append(_mesh_to_dict(mesh))
        return self.mesh_index[key]

    def blob(self, data: bytes | None) -> int | None:
        if data is None:
            return None
        key = id(data)
        if key not in self.blob_index:
            self.blob_index[key] = len(self.blobs)
            self.blobs.append(base64.b64encode(data).decode("ascii"))
        return self.blob_index[key]

    def obj(self, obj: SceneObject | None) -> str | None:
        if obj is None:
            return None
        if self.scene._objects.get(obj._id) is not obj:
            self.detached_objects.setdefault(obj._id, obj)
        return obj._id

    def material(self, m: Any) -> str:
        if self.scene._materials._materials.get(m._id) is not m:
            self.detached_materials.setdefault(m._id, m)
        return m._id

    def texture(self, t: Any) -> str:
        if self.scene._materials._textures.get(t._id) is not t:
            self.detached_textures.setdefault(t._id, t)
        return t._id

    def target(self, key: str, target: Any) -> Any:
        if key == "cad.item":
            self.cad_items.setdefault(target.id, target)
            return target.id
        if key == "animation.item":
            self.animation_items.setdefault(target.id, target)
            return target.id
        if key.startswith("object."):
            return self.obj(target)
        if key == "material":
            return self.material(target)
        if key == "texture":
            return self.texture(target)
        return None

    # -- states -------------------------------------------------------------------------

    def state(self, key: str, s: Any) -> Any:
        if key in ("object.name", "object.visible", "scene.name", "scene.active_camera"):
            return s
        if key == "object.transform":
            return [list(s[0]), list(s[1]), list(s[2])]
        if key in ("object.appearance", "object.camera", "object.light", "scene.render"):
            return None if s is None else _fields_to_dict(ASPECTS[key], s)
        if key == "object.geometry":
            return {"mesh": self.mesh(s[0]), "geometry": s[1]}
        if key == "object.material":
            return {"material": s[0], "groups": {str(g): m for g, m in s[1]}}
        if key == "object.parent":
            return self.obj(s)
        if key in ("object.children", "scene.roots"):
            return [self.obj(o) for o in s]
        if key == "object.alive":
            return s
        if key == "object.annotations":
            return {"tags": list(s[0]), "metadata": json.loads(s[1])}
        if key == "scene.world":
            return _world_to_dict(s)
        if key == "library.materials":
            return [self.material(m) for m in s]
        if key == "library.textures":
            return [self.texture(t) for t in s]
        if key == "material":
            return _material_state_to_dict(s)
        if key == "texture":
            return self.texture_state(s)
        if key == "procedural":
            return self.procedural_state(s[1])
        if key == "cad.item":
            alive, name, definition, links = s
            return {"alive": alive, "name": name, "definition": definition, "links": list(links)}
        if key == "cad.settings":
            return s
        if key == "animation.item":
            alive, name, definition = s
            return {"alive": alive, "name": name, "definition": definition.to_json()}
        if key == "animation.settings":
            return s
        raise AssertionError(key)  # pragma: no cover

    def tile_blob(self, tile: Any) -> int:
        """A tile (uint16 array) stored once as a zlib-compressed little-endian blob."""
        key = ("tile", id(tile))
        if key not in self.blob_index:
            raw = zlib.compress(np.ascontiguousarray(tile, dtype="<u2").tobytes(), 6)
            self.blob_index[key] = len(self.blobs)
            self.blobs.append(base64.b64encode(raw).decode("ascii"))
        return self.blob_index[key]

    def texture_state(self, s: tuple) -> dict[str, Any]:
        name, fields = s
        if "_graph" in fields:
            return {"name": name, "type": "procedural", "graph": fields["_graph"].to_dict()}
        if "_layers" in fields:
            layers = []
            for layer in fields["_layers"]:
                record: dict[str, Any] = {
                    "id": layer.id,
                    "name": layer.name,
                    "kind": layer.kind,
                    "blend": layer.blend,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                    "value": list(layer.value),
                    "tiles": None
                    if layer.tiles is None
                    else [[t, self.tile_blob(a)] for t, a in sorted(layer.tiles.items())],
                    "mask": None
                    if layer.mask is None
                    else [[t, self.tile_blob(a)] for t, a in sorted(layer.mask.items())],
                }
                layers.append(record)
            return {
                "name": name,
                "type": "painted",
                "width": fields["_width"],
                "height": fields["_height"],
                "encoding": fields["_encoding"],
                "background": list(fields["_background"]),
                "layers": layers,
                "wrap": list(fields["_wrap"]),
                "filter": fields["_filter"],
                "mipmaps": fields["_mipmaps"],
                "color_space": fields["_color_space"],
            }
        file = fields["_file"]
        path, relative = None, False
        if file is not None:
            try:
                path, relative = Path(file).relative_to(self.project_dir).as_posix(), True
            except ValueError:
                path = str(file)
        return {
            "name": name,
            "type": "image",
            "file": path,
            "relative": relative,
            "rooted": fields["_root"] is not None,
            "data": self.blob(fields["_data"]),
            "wrap": list(fields["_wrap"]),
            "filter": fields["_filter"],
            "mipmaps": fields["_mipmaps"],
            "color_space": fields["_color_space"],
        }

    def procedural_state(self, defs: dict) -> dict[str, Any]:
        from .procedural.params import value_to_json

        geometry = []
        for g in defs["geometries"]:
            geometry.append(
                {
                    "id": g["id"],
                    "name": g["name"],
                    "parameters": [p.to_json() for p in g["parameters"].values()],
                    "rules": [r.to_json() for r in g["rules"].values()],
                    "source": g["source"].to_json(self.mesh),
                    "modifiers": [
                        {
                            "id": mid,
                            "name": name,
                            "type": mtype,
                            "enabled": enabled,
                            "params": value_to_json(params),
                        }
                        for mid, mtype, name, enabled, params in g["modifiers"]
                    ],
                }
            )
        return {
            "parameters": [p.to_json() for p in defs["parameters"].values()],
            "rules": [r.to_json() for r in defs["rules"].values()],
            "geometry": geometry,
        }

    def step(self, step: h._Step) -> dict[str, Any]:
        changes = []
        for c in step.changes:
            key = KEYS[c.aspect]
            changes.append(
                {
                    "aspect": key,
                    "target": self.target(key, c.target),
                    "before": self.state(key, c.before),
                    "after": self.state(key, c.after),
                }
            )
        return {
            "index": step.serial,
            "parent": step.parent,
            "label": step.label,
            "changes": changes,
        }

    # -- detached entities ----------------------------------------------------------

    def object_record(self, obj: SceneObject) -> dict[str, Any]:
        record = {"id": obj._id}
        for key in (
            "object.name",
            "object.transform",
            "object.visible",
            "object.appearance",
            "object.camera",
            "object.light",
            "object.geometry",
            "object.material",
            "object.parent",
            "object.children",
            "object.annotations",
        ):
            aspect = ASPECTS[key]
            record[key.split(".", 1)[1]] = self.state(key, aspect.capture(obj))
        return record

    def entities(self) -> tuple[list, list, list]:
        objects: list[dict[str, Any]] = []
        done: set[str] = set()
        while len(done) < len(self.detached_objects):  # records may add more objects
            for oid, obj in list(self.detached_objects.items()):
                if oid not in done:
                    done.add(oid)
                    objects.append(self.object_record(obj))
        materials = [
            {"id": m._id, **self.state("material", h.MATERIAL.capture(m))}
            for m in self.detached_materials.values()
        ]
        textures = [
            {"id": t._id, **self.state("texture", h.TEXTURE.capture(t))}
            for t in self.detached_textures.values()
        ]
        return objects, materials, textures


def history_document(scene: Scene, project_dir: Path) -> dict[str, Any]:
    """The history of ``scene`` as a JSON-compatible document (see the module docstring)."""
    history = scene.history
    history._settle()
    if history._frames:
        raise HistoryError("TRANSACTION_OPEN", "cannot save a history while a transaction is open")
    writer = _Writer(scene, project_dir)
    undo = [writer.step(s) for s in history._undo]
    redo = [writer.step(s) for s in history._redo]
    objects, materials, textures = writer.entities()
    return {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "generator": f"aether3d {__version__}",
        "project": fingerprint(scene, project_dir),
        "position": history._current,
        "serial": history._serial,
        "checkpoints": dict(history._checkpoints),
        "max_entries": history._max_entries,
        "max_bytes": history._max_bytes,
        "dropped": history.dropped,
        "undo": undo,
        "redo": redo,
        "objects": objects,
        "materials": materials,
        "textures": textures,
        "meshes": writer.meshes,
        "blobs": writer.blobs,
        "cad_items": [
            {"id": i.id, "kind": i.kind}
            for i in sorted(writer.cad_items.values(), key=lambda i: i._serial)
        ],
        "animation_items": [
            {"id": i.id, "kind": i.kind}
            for i in sorted(writer.animation_items.values(), key=lambda i: i._serial)
        ],
    }


def save_history(scene: Scene, path: str | os.PathLike[str], project_dir: Path) -> Path:
    """Write the history of ``scene`` to ``path`` (atomically)."""
    from .serialization import dumps

    document = history_document(scene, project_dir)
    return write_atomic(path, dumps(document).encode("utf-8"))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _fields_to_dict(aspect: h.Aspect, state: tuple) -> dict[str, Any]:
    from .materials.material import _jsonable

    names = [f[1:] for f in aspect.fields]
    return {n: _jsonable(v) for n, v in zip(names, state, strict=True)}


def _fields_from_dict(key: str, data: Any, loc: str) -> tuple | None:
    if data is None:
        return None
    from .rendering.camera import Camera
    from .rendering.light import Light
    from .rendering.settings import Appearance, RenderSettings

    if key == "object.camera":
        value = Camera.from_dict(data, loc)
    elif key == "object.light":
        value = Light.from_dict(data, loc)
    elif key == "object.appearance":
        value = Appearance.from_dict(data, loc)
    else:
        value = RenderSettings.from_dict(data, loc)
    return tuple(getattr(value, f) for f in ASPECTS[key].fields)


def _world_to_dict(state: tuple) -> dict[str, Any]:
    from .materials.material import _jsonable
    from .rendering.settings import Environment

    env = dict(zip(Environment._FIELDS, state[3], strict=True))
    return {
        "background": _jsonable(state[0]),
        "ambient_color": _jsonable(state[1]),
        "ambient_intensity": state[2],
        "environment": {k: _jsonable(v) for k, v in env.items()},
    }


def _world_from_dict(data: Any, loc: str) -> tuple:
    from .rendering.settings import World

    world = World.from_dict(data, loc)
    env = world._environment
    return (
        world._background,
        world._ambient_color,
        world._ambient_intensity,
        tuple(getattr(env, f"_{f}") for f in env._FIELDS),
    )


def _material_state_to_dict(state: tuple) -> dict[str, Any]:
    from .materials.material import _jsonable

    name, model, values, textures = state
    return {
        "name": name,
        "model": model,
        "params": {k: _jsonable(v) for k, v in values.items()},
        "textures": {slot: b.to_dict() for slot, b in textures.items()},
    }


def _material_state_from_dict(data: Any, loc: str) -> tuple:
    from .materials.material import MODELS, binding_from_dict, check_param

    data = _record(data, loc, ("name", "model", "params", "textures"), ("id",))
    if data["model"] not in MODELS:
        raise _bad(f"unknown material model {data['model']!r}", f"{loc}.model")
    if not isinstance(data["params"], dict) or not isinstance(data["textures"], dict):
        raise _bad("params and textures must be objects", loc)
    values = {k: check_param(k, v, f"{loc}.params.{k}") for k, v in data["params"].items()}
    textures = {
        slot: binding_from_dict(slot, raw, f"{loc}.textures.{slot}")
        for slot, raw in data["textures"].items()
    }
    return (_string(data["name"], f"{loc}.name"), data["model"], values, textures)


def _record(
    value: Any, loc: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _bad("expected an object", loc)
    missing = [k for k in required if k not in value]
    unknown = sorted(set(value) - set(required) - set(optional))
    if missing or unknown:
        what = f"missing {', '.join(missing)}" if missing else f"unknown {', '.join(unknown)}"
        raise _bad(f"keys: {what}", loc)
    return value


def _string(value: Any, loc: str) -> str:
    if not isinstance(value, str) or not value:
        raise _bad("expected a non-empty string", loc)
    return value


def _list(value: Any, loc: str) -> list:
    if not isinstance(value, list):
        raise _bad("expected an array", loc)
    return value


def _index(value: Any, items: list, loc: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(items):
        raise _bad(f"expected an index below {len(items)}", loc)
    return items[value]


class _Reader:
    def __init__(self, scene: Scene, doc: dict[str, Any], project_dir: Path) -> None:
        from .serialization import _mesh_from_dict

        self.scene = scene
        self.project_dir = project_dir
        self.meshes = []
        for i, raw in enumerate(_list(doc["meshes"], "meshes")):
            try:
                self.meshes.append(_mesh_from_dict(raw, f"meshes[{i}]"))
            except AetherError as exc:
                raise _bad(str(exc), f"meshes[{i}]") from None
        self.blobs = []
        for i, raw in enumerate(_list(doc["blobs"], "blobs")):
            try:
                self.blobs.append(base64.b64decode(_string(raw, f"blobs[{i}]"), validate=True))
            except (binascii.Error, ValueError):
                raise _bad("not valid base64", f"blobs[{i}]") from None
        library = scene.materials
        self.objects: dict[str, Any] = dict(scene._objects)
        self.materials: dict[str, Any] = dict(library._materials)
        self.textures: dict[str, Any] = dict(library._textures)
        self._detached(doc)
        self.cad_items: dict[str, Any] = {}
        self._cad_items(doc.get("cad_items", []))
        self.animation_items: dict[str, Any] = {}
        self._animation_items(doc.get("animation_items", []))

    def _animation_items(self, records: Any) -> None:
        """Animation items the history refers to (removed ones come back through undo)."""
        import uuid

        from .animation.system import KINDS, AnimItem

        records = _list(records, "animation_items")
        if not records:
            return
        system = self.scene.animation
        for i, raw in enumerate(records):
            loc = f"animation_items[{i}]"
            raw = _record(raw, loc, ("id", "kind"))
            try:
                iid = str(uuid.UUID(_string(raw["id"], f"{loc}.id")))
            except ValueError:
                raise _bad("not a UUID", f"{loc}.id") from None
            kind = _string(raw["kind"], f"{loc}.kind")
            if kind not in KINDS:
                raise _bad(f"unknown animation item kind {kind!r}", f"{loc}.kind")
            item = system._items.get(iid)
            if item is None:
                item = AnimItem(system, iid, kind, f"(removed {iid[:8]})", None)
                item._alive = False
                system._items[iid] = item
            elif item.kind != kind:
                raise _bad("the item's kind differs from the project's", loc)
            self.animation_items[iid] = item

    def _cad_items(self, records: Any) -> None:
        """CAD items the history refers to: those of the project, or recreated (removed
        items come back through undo)."""
        import uuid

        from .cad.document import KINDS, Item

        records = _list(records, "cad_items")
        if not records:
            return
        doc = self.scene.cad
        for i, raw in enumerate(records):
            loc = f"cad_items[{i}]"
            raw = _record(raw, loc, ("id", "kind"))
            try:
                iid = str(uuid.UUID(_string(raw["id"], f"{loc}.id")))
            except ValueError:
                raise _bad("not a UUID", f"{loc}.id") from None
            kind = _string(raw["kind"], f"{loc}.kind")
            if kind not in KINDS:
                raise _bad(f"unknown CAD item kind {kind!r}", f"{loc}.kind")
            item = doc._items.get(iid)
            if item is None:
                item = Item(doc, iid, kind, f"(removed {iid[:8]})", {})
                item._alive = False
                doc._items[iid] = item
            elif item.kind != kind:
                raise _bad("the item's kind differs from the project's", loc)
            self.cad_items[iid] = item

    # -- detached entities ----------------------------------------------------------

    def _detached(self, doc: dict[str, Any]) -> None:
        from .materials.material import Material
        from .materials.textures import ImageTexture, ProceduralTexture
        from .scene import SceneObject
        from .transform import Transform

        scene = self.scene
        library = scene.materials
        for i, raw in enumerate(_list(doc["textures"], "textures")):
            loc = f"textures[{i}]"
            tid = _string(_record(raw, loc, ("id",), _TEXTURE_KEYS)["id"], f"{loc}.id")
            if tid in self.textures:
                raise _bad(f"texture {tid!r} is both in the project and detached", loc)
            name, fields = self.texture_state(raw, loc)
            if "_graph" in fields:
                texture = ProceduralTexture(library, tid, name, fields["_graph"])
            elif "_layers" in fields:
                from .materials.painted import PaintedTexture

                texture = PaintedTexture(
                    library,
                    tid,
                    name,
                    width=fields["_width"],
                    height=fields["_height"],
                    encoding=fields["_encoding"],
                    background=fields["_background"],
                    layers=fields["_layers"],
                    wrap=fields["_wrap"],
                    filter=fields["_filter"],
                    mipmaps=fields["_mipmaps"],
                    color_space=fields["_color_space"],
                )
            else:
                texture = ImageTexture(library, tid, name)
                for k, v in fields.items():
                    object.__setattr__(texture, k, v)
            texture._library = None
            self.textures[tid] = texture
        for i, raw in enumerate(_list(doc["materials"], "materials")):
            loc = f"materials[{i}]"
            mid = _string(_record(raw, loc, ("id",), _MATERIAL_KEYS)["id"], f"{loc}.id")
            if mid in self.materials:
                raise _bad(f"material {mid!r} is both in the project and detached", loc)
            name, model, values, textures = _material_state_from_dict(raw, loc)
            material = Material(library, mid, name, model, values, textures)
            material._library = None
            self.materials[mid] = material
        records = []
        for i, raw in enumerate(_list(doc["objects"], "objects")):
            loc = f"objects[{i}]"
            # "annotations" (tags and metadata, 0.10.0) is optional: 0.9.0 files have none.
            raw = _record(raw, loc, ("id", *_OBJECT_KEYS), ("annotations",))
            raw = {"annotations": {"tags": [], "metadata": {}}, **raw}
            oid = _string(raw["id"], f"{loc}.id")
            if oid in self.objects:
                raise _bad(f"object {oid!r} is both in the project and detached", loc)
            obj = SceneObject.__new__(SceneObject)
            obj._scene, obj._home, obj._id = None, scene, oid
            obj._transform = Transform()
            obj._transform._owner = obj
            obj._appearance = h._appearance()
            obj._camera = obj._light = None
            obj._parent, obj._children = None, []
            obj._tags, obj._metadata = (), "{}"
            self.objects[oid] = obj
            records.append((obj, raw, loc))
        for obj, raw, loc in records:  # every object exists now: link them
            for key in (*_OBJECT_KEYS, "annotations"):
                full = f"object.{key}"
                ASPECTS[full].apply(obj, self.state(full, raw[key], f"{loc}.{key}"))
            for aspect in (h.APPEARANCE, h.CAMERA, h.LIGHT):
                value = getattr(obj, aspect.attr)
                if value is not None:
                    value._owner = (obj, aspect)

    def texture_state(self, data: Any, loc: str) -> tuple:
        from .materials.graph import TextureGraph
        from .materials.textures import check_sampler

        kind = data.get("type") if isinstance(data, dict) else None
        if kind == "procedural":
            data = _record(data, loc, ("name", "type", "graph"), ("id",))
            graph = TextureGraph.from_dict(data["graph"], f"{loc}.graph")
            return (_string(data["name"], f"{loc}.name"), {"_graph": graph})
        if kind == "painted":
            return self.painted_state(data, loc)
        data = _record(data, loc, ("name", "type", *_IMAGE_KEYS), ("id",))
        if kind != "image":
            raise _bad(f"unknown texture type {kind!r}", f"{loc}.type")
        sampler = check_sampler(
            {k: data[k] for k in ("wrap", "filter", "mipmaps", "color_space")}, loc
        )
        file = data["file"]
        if file is not None:
            file = _string(file, f"{loc}.file")
            file = (self.project_dir / file).absolute() if data["relative"] else Path(file)
        return (
            _string(data["name"], f"{loc}.name"),
            {
                "_file": file,
                "_root": self.project_dir if data["rooted"] else None,
                "_data": _index(data["data"], self.blobs, f"{loc}.data"),
                "_wrap": sampler["wrap"],
                "_filter": sampler["filter"],
                "_mipmaps": sampler["mipmaps"],
                "_color_space": sampler["color_space"],
            },
        )

    def painted_state(self, data: Any, loc: str) -> tuple:
        from .materials.painted import TILE, Layer, check_document, check_layer
        from .materials.textures import check_sampler

        data = _record(data, loc, ("name", "type", *_PAINTED_KEYS), ("id",))
        width, height, encoding, background = check_document(
            data["width"], data["height"], data["encoding"], data["background"], loc
        )
        sampler = check_sampler(
            {k: data[k] for k in ("wrap", "filter", "mipmaps", "color_space")}, loc
        )
        tiles_total = -(-width // TILE) * -(-height // TILE)

        def tiles(raw: Any, channels: int, where: str) -> dict | None:
            if raw is None:
                return None
            out = {}
            for k, item in enumerate(_list(raw, where)):
                if not isinstance(item, list) or len(item) != 2:
                    raise _bad("a tile is [index, blob]", f"{where}[{k}]")
                index, blob = item
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < tiles_total
                ):
                    raise _bad("tile index out of range", f"{where}[{k}]")
                packed = _index(blob, self.blobs, f"{where}[{k}]")
                size = TILE * TILE * channels * 2
                try:
                    raw_bytes = zlib.decompressobj().decompress(packed, size + 1)
                except zlib.error:
                    raise _bad("tile data is corrupt", f"{where}[{k}]") from None
                if len(raw_bytes) != size:
                    raise _bad("tile data has the wrong size", f"{where}[{k}]")
                shape = (TILE, TILE, channels) if channels > 1 else (TILE, TILE)
                tile = np.frombuffer(raw_bytes, dtype="<u2").astype(np.uint16).reshape(shape)
                tile.flags.writeable = False
                out[index] = tile
            return out

        layers = []
        for i, raw in enumerate(_list(data["layers"], f"{loc}.layers")):
            where = f"{loc}.layers[{i}]"
            raw = _record(
                raw,
                where,
                ("id", "name", "kind", "blend", "opacity", "visible", "value", "tiles", "mask"),
                (),
            )
            layer = Layer(
                _string(raw["id"], f"{where}.id"),
                _string(raw["name"], f"{where}.name"),
                raw["kind"],
                raw["blend"],
                raw["opacity"],
                raw["visible"],
                tuple(float(v) for v in _list(raw["value"], f"{where}.value")),
                tiles(raw["tiles"], 4, f"{where}.tiles"),
                tiles(raw["mask"], 1, f"{where}.mask"),
            )
            try:
                layers.append(check_layer(layer, where))
            except Exception as exc:  # MaterialError: report as a history error
                raise _bad(str(exc), where) from None
        return (
            _string(data["name"], f"{loc}.name"),
            {
                "_width": width,
                "_height": height,
                "_encoding": encoding,
                "_layers": tuple(layers),
                "_background": background,
                "_wrap": sampler["wrap"],
                "_filter": sampler["filter"],
                "_mipmaps": sampler["mipmaps"],
                "_color_space": sampler["color_space"],
            },
        )

    # -- references and states ------------------------------------------------------

    def obj(self, ref: Any, loc: str) -> Any:
        if ref is None:
            return None
        try:
            return self.objects[ref]
        except (KeyError, TypeError):
            raise _bad(f"unknown object {ref!r}", loc) from None

    def target(self, key: str, ref: Any, loc: str) -> Any:
        if key == "cad.item":
            if ref not in self.cad_items:
                raise _bad(f"unknown CAD item {ref!r}", loc)
            return self.cad_items[ref]
        if key == "cad.settings":
            if ref is not None:
                raise _bad("a change of cad.settings has no target", loc)
            return self.scene.cad
        if key == "animation.item":
            if ref not in self.animation_items:
                raise _bad(f"unknown animation item {ref!r}", loc)
            return self.animation_items[ref]
        if key == "animation.settings":
            if ref is not None:
                raise _bad("a change of animation.settings has no target", loc)
            return self.scene.animation
        if key.startswith("object."):
            return self.obj(ref, loc)
        if key == "material":
            if ref not in self.materials:
                raise _bad(f"unknown material {ref!r}", loc)
            return self.materials[ref]
        if key == "texture":
            if ref not in self.textures:
                raise _bad(f"unknown texture {ref!r}", loc)
            return self.textures[ref]
        if ref is not None:
            raise _bad(f"a change of {key} has no target", loc)
        if key == "procedural":
            return self.scene.procedural
        if key.startswith("library."):
            return self.scene.materials
        return self.scene

    def state(self, key: str, data: Any, loc: str) -> Any:
        if key in ("object.name", "scene.name"):
            if not isinstance(data, str) or not data.strip():
                raise _bad("expected a name (a non-empty string)", loc)
            return data
        if key == "object.annotations":
            from .scene import _check_metadata, _check_tags

            data = _record(data, loc, ("tags", "metadata"))
            try:
                return (_check_tags(data["tags"]), _check_metadata(data["metadata"]))
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        if key in ("object.visible", "object.alive"):
            if not isinstance(data, bool):
                raise _bad("expected true or false", loc)
            return data
        if key == "scene.active_camera":
            return None if data is None else _string(data, loc)
        if key == "object.transform":
            from .transform import Transform

            if not (isinstance(data, list) and len(data) == 3):
                raise _bad("expected [position, rotation, scale]", loc)
            try:
                t = Transform(*data)
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
            rotation = tuple(float(v) for v in data[1])
            if abs(sum(v * v for v in rotation) - 1.0) > 1e-6:
                raise _bad("the rotation is not a unit quaternion", loc)
            # The stored quaternion is the unit quaternion it was: keep it exactly.
            return (t._position, rotation, t._scale)
        if key in ("object.appearance", "object.camera", "object.light", "scene.render"):
            try:
                return _fields_from_dict(key, data, loc)
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        if key == "object.geometry":
            data = _record(data, loc, ("mesh", "geometry"))
            geometry = data["geometry"]
            if geometry is not None:
                geometry = _string(geometry, f"{loc}.geometry")
            return (_index(data["mesh"], self.meshes, f"{loc}.mesh"), geometry)
        if key == "object.material":
            data = _record(data, loc, ("material", "groups"))
            material = data["material"]
            if material is not None:
                material = _string(material, f"{loc}.material")
            if not isinstance(data["groups"], dict):
                raise _bad("groups must be an object", f"{loc}.groups")
            groups = []
            for g, m in data["groups"].items():
                if not g.isdigit():
                    raise _bad(f"face group {g!r} is not an integer", f"{loc}.groups")
                groups.append((int(g), _string(m, f"{loc}.groups.{g}")))
            return (material, tuple(sorted(groups)))
        if key == "object.parent":
            return self.obj(data, loc)
        if key in ("object.children", "scene.roots"):
            return tuple(self.obj(r, f"{loc}[{i}]") for i, r in enumerate(_list(data, loc)))
        if key == "scene.world":
            try:
                return _world_from_dict(data, loc)
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        if key == "library.materials":
            return tuple(
                self.target("material", r, f"{loc}[{i}]") for i, r in enumerate(_list(data, loc))
            )
        if key == "library.textures":
            return tuple(
                self.target("texture", r, f"{loc}[{i}]") for i, r in enumerate(_list(data, loc))
            )
        if key == "material":
            try:
                return _material_state_from_dict(data, loc)
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        if key == "texture":
            try:
                return self.texture_state(data, loc)
            except HistoryError:
                raise
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        if key == "procedural":
            return (-1, self.procedural_state(data, loc))
        if key == "cad.item":
            from .cad.document import check_json

            data = _record(data, loc, ("alive", "name", "definition", "links"))
            if not isinstance(data["alive"], bool):
                raise _bad("expected true or false", f"{loc}.alive")
            try:
                definition = check_json(data["definition"], f"{loc}.definition")
            except AetherError as exc:
                raise _bad(str(exc), f"{loc}.definition") from None
            links = tuple(
                _string(x, f"{loc}.links[{i}]")
                for i, x in enumerate(_list(data["links"], f"{loc}.links"))
            )
            return (data["alive"], _string(data["name"], f"{loc}.name"), definition, links)
        if key == "cad.settings":
            from .cad.document import check_json

            try:
                return check_json(data, loc)
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        if key == "animation.settings":
            from .animation.system import _check_settings

            try:
                return _check_settings(data, loc)
            except AetherError as exc:
                raise _bad(str(exc), loc) from None
        raise _bad(f"unknown aspect {key!r}", loc)  # pragma: no cover - checked before

    def animation_state(self, item: Any, data: Any, loc: str) -> tuple:
        """``(alive, name, definition)`` of an animation item (its kind decides how the
        definition reads; clips are referenced by id, alive or not)."""
        from .animation.system import definition_from_json

        data = _record(data, loc, ("alive", "name", "definition"))
        if not isinstance(data["alive"], bool):
            raise _bad("expected true or false", f"{loc}.alive")
        name = _string(data["name"], f"{loc}.name")
        system = self.scene.animation

        def any_clip(ref: str, where: str) -> str:
            found = system._items.get(ref)
            if found is None or found.kind != "clip":
                raise _bad(f"unknown clip {ref!r}", where)
            return ref

        try:
            definition = definition_from_json(
                item.kind, data["definition"], system, f"{loc}.definition", any_clip
            )
        except AetherError as exc:
            raise _bad(str(exc), f"{loc}.definition") from None
        return (data["alive"], name, definition)

    def procedural_state(self, data: Any, loc: str) -> dict:
        from .scene import Scene
        from .serialization import _read_procedural

        data = _record(data, loc, ("parameters", "rules", "geometry"))
        scratch = Scene("history")
        try:
            with scratch.history._suspended():
                _read_procedural(scratch, data, self.meshes)
        except AetherError as exc:
            raise _bad(str(exc), loc) from None
        return scratch.procedural._copy_defs()

    def step(self, data: Any, loc: str) -> h._Step:
        data = _record(data, loc, ("index", "parent", "label", "changes"))
        for k in ("index", "parent"):
            if isinstance(data[k], bool) or not isinstance(data[k], int) or data[k] < 0:
                raise _bad("expected a non-negative integer", f"{loc}.{k}")
        changes = []
        for i, raw in enumerate(_list(data["changes"], f"{loc}.changes")):
            cloc = f"{loc}.changes[{i}]"
            raw = _record(raw, cloc, ("aspect", "target", "before", "after"))
            key = raw["aspect"]
            if key not in ASPECTS:
                raise _bad(f"unknown aspect {key!r}", f"{cloc}.aspect")
            target = self.target(key, raw["target"], f"{cloc}.target")
            if key == "animation.item":
                before = self.animation_state(target, raw["before"], f"{cloc}.before")
                after = self.animation_state(target, raw["after"], f"{cloc}.after")
            else:
                before = self.state(key, raw["before"], f"{cloc}.before")
                after = self.state(key, raw["after"], f"{cloc}.after")
            changes.append(h._Change(ASPECTS[key], target, before, after))
        if not changes:
            raise _bad("a step changes something", f"{loc}.changes")
        return h._Step(
            data["index"], data["parent"], _string(data["label"], f"{loc}.label"), changes
        )


_OBJECT_KEYS = (
    "name",
    "transform",
    "visible",
    "appearance",
    "camera",
    "light",
    "geometry",
    "material",
    "parent",
    "children",
)
_MATERIAL_KEYS = ("name", "model", "params", "textures")
_IMAGE_KEYS = (
    "file",
    "relative",
    "rooted",
    "data",
    "wrap",
    "filter",
    "mipmaps",
    "color_space",
)
_PAINTED_KEYS = (
    "width",
    "height",
    "encoding",
    "background",
    "layers",
    "wrap",
    "filter",
    "mipmaps",
    "color_space",
)
_TEXTURE_KEYS = tuple(dict.fromkeys(("name", "type", "graph", *_IMAGE_KEYS, *_PAINTED_KEYS)))
_TOP_KEYS = (
    "format",
    "format_version",
    "generator",
    "project",
    "position",
    "serial",
    "checkpoints",
    "max_entries",
    "max_bytes",
    "dropped",
    "undo",
    "redo",
    "objects",
    "materials",
    "textures",
    "meshes",
    "blobs",
)


def _check_header(doc: Any) -> dict[str, Any]:
    doc = _record(doc, "", _TOP_KEYS, ("cad_items", "animation_items"))
    if doc["format"] != FORMAT_ID:
        raise _bad(f"format must be {FORMAT_ID!r}", "format")
    if doc["format_version"] != FORMAT_VERSION:
        raise HistoryError(
            "UNSUPPORTED_HISTORY",
            f"history format version {doc['format_version']!r} is not supported "
            f"(this is version {FORMAT_VERSION})",
            "format_version",
        )
    for k in ("position", "serial", "max_entries", "max_bytes", "dropped"):
        if isinstance(doc[k], bool) or not isinstance(doc[k], int) or doc[k] < 0:
            raise _bad("expected a non-negative integer", k)
    if not isinstance(doc["checkpoints"], dict) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in doc["checkpoints"].values()
    ):
        raise _bad("checkpoints must map names to step indexes", "checkpoints")
    return doc


def install(scene: Scene, doc: dict[str, Any], project_dir: Path) -> None:
    """Decode ``doc`` (checked against the scene's fingerprint) into ``scene.history``."""
    doc = _check_header(doc)
    if doc["project"] != fingerprint(scene, project_dir):
        raise HistoryError(
            "HISTORY_MISMATCH",
            "the history belongs to another version of this project",
            hint="the project was changed without its history (by another program, or "
            "saved without history=True); its old history cannot be applied",
        )
    reader = _Reader(scene, doc, project_dir)
    undo = [reader.step(s, f"undo[{i}]") for i, s in enumerate(_list(doc["undo"], "undo"))]
    redo = [reader.step(s, f"redo[{i}]") for i, s in enumerate(_list(doc["redo"], "redo"))]
    history = scene.history
    history.clear()
    history._undo, history._redo = undo, redo
    history._current, history._serial = doc["position"], doc["serial"]
    history._max_entries, history._max_bytes = doc["max_entries"], doc["max_bytes"]
    history.dropped = doc["dropped"]
    for step in (*undo, *redo):
        history._account(step, 1)
    history._checkpoints = dict(doc["checkpoints"])
    history._prune_checkpoints()


def replay(scene: Scene, doc: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    """Install ``doc`` into ``scene`` and undo and redo every step, checking the scene's
    invariants after each; returns a report. ``scene`` should be a scratch copy."""
    install(scene, doc, project_dir)
    history = scene.history
    before = fingerprint(scene, project_dir)
    undone = len(history._undo)
    redone = len(history._redo)
    for _ in range(undone):
        history.undo()
        h.check_scene(scene)
    base = fingerprint(scene, project_dir)
    for _ in range(undone + redone):
        history.redo()
        h.check_scene(scene)
    history.undo(redone) if redone else None
    after = fingerprint(scene, project_dir)
    return {
        "steps": undone + redone,
        "undone_and_redone": undone,
        "redone_and_undone": redone,
        "base": base,
        "ok": after == before,
    }


def read_document(path: str | os.PathLike[str]) -> dict[str, Any]:
    from .serialization import loads

    file = Path(path)
    try:
        size = file.stat().st_size
    except FileNotFoundError:
        raise HistoryError("NO_HISTORY", f"no history file {str(file)!r}") from None
    if size > MAX_FILE_BYTES:
        raise HistoryError("INVALID_HISTORY", f"the history file is too large ({size:,} bytes)")
    try:
        return loads(file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, AetherError) as exc:
        raise HistoryError(
            "INVALID_HISTORY", f"the history file is not valid JSON: {exc}"
        ) from None


def load_history(
    scene: Scene, path: str | os.PathLike[str], project_dir: Path, *, verify: bool = True
) -> None:
    """Give ``scene`` the history saved in ``path``.

    With ``verify`` (default) the history is first replayed on a copy of the scene and
    refused (``INVALID_HISTORY``) unless every step applies and the copy comes back
    to the same document. Nothing changes if loading fails.
    """
    doc = read_document(path)
    if verify:
        from .serialization import scene_from_dict, scene_to_dict

        scratch = scene_from_dict(scene_to_dict(scene, project_dir), project_dir)
        try:
            report = replay(scratch, doc, project_dir)
        except HistoryError:
            raise
        except (AetherError, AssertionError) as exc:
            raise HistoryError(
                "INVALID_HISTORY", f"the history does not replay on this project: {exc}"
            ) from None
        if not report["ok"]:
            raise HistoryError(
                "INVALID_HISTORY", "replaying the history does not restore the project"
            )
    install(scene, doc, project_dir)
