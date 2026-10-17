"""A scene's material library: materials, textures, assignments, validation and persistence.

Every :class:`~aether3d.Scene` has one (``scene.materials``). It keeps
material definitions (:class:`Material`), texture resources
(:class:`ImageTexture`, :class:`ProceduralTexture`) and procedural graphs
separate from the objects that use them: an object refers to a material by
id (``obj.material``), possibly per face group (``obj.group_materials``), and
a material refers to textures by id through its texture bindings. Removing a
material that objects use, or a texture that materials use, is refused
unless ``force=True`` (which then unassigns or unbinds it explicitly).

Changes made through :meth:`MaterialLibrary.transaction` are all-or-nothing:
if the block raises, materials, textures and assignments return to their
previous state (the same Python objects, with their previous values).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import ERROR, INFO, WARNING, Diagnostic
from ..errors import AetherError, MaterialError
from .graph import TextureGraph, compile_graph
from .material import (
    MODELS,
    SLOTS,
    Material,
    TextureBinding,
    binding_from_dict,
    check_param,
    default_params,
)
from .sampling import sample
from .textures import (
    ImageTexture,
    ProceduralTexture,
    Texture,
    check_relative_path,
    check_sampler,
)

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

MAX_NAME_LENGTH = 256
MAX_MATERIALS = 10_000
MAX_TEXTURES = 10_000
MAX_GROUP = 2**16 - 1
"""Largest face group index (groups are stored per triangle)."""


def _history_aspects():
    from .. import history

    return history


def _err(code: str, message: str, location: str | None = None, **kw) -> MaterialError:
    return MaterialError(code, message, location, **kw)


def _check_name(name: Any, what: str) -> str:
    if not isinstance(name, str) or not name.strip() or len(name) > MAX_NAME_LENGTH:
        raise _err(
            "INVALID_NAME",
            f"{what} name must be a non-empty string of at most {MAX_NAME_LENGTH} characters, "
            f"got {name!r}",
        )
    return name


def new_id() -> str:
    return str(uuid.uuid4())


class MaterialLibrary:
    """Materials and textures of one scene (``scene.materials``); see the module docstring."""

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._materials: dict[str, Material] = {}
        self._material_names: dict[str, Material] = {}
        self._textures: dict[str, Texture] = {}
        self._texture_names: dict[str, Texture] = {}
        self.base_dir: Path | None = None
        """Directory of the project file the library was loaded from or saved to."""
        self.version = 0
        """Incremented by every change (renderers use it to refresh cached data)."""

    # -- access ---------------------------------------------------------------------

    @property
    def materials(self) -> tuple[Material, ...]:
        return tuple(self._materials.values())

    @property
    def textures(self) -> tuple[Texture, ...]:
        return tuple(self._textures.values())

    def __len__(self) -> int:
        return len(self._materials)

    def __iter__(self) -> Iterator[Material]:
        return iter(tuple(self._materials.values()))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Material):
            return item._library is self
        return isinstance(item, str) and (item in self._materials or item in self._material_names)

    def __getitem__(self, ref: Any) -> Material:
        return self.material(ref)

    def material(self, ref: Any) -> Material:
        """A material of this scene by object, id or name."""
        found = self.find_material(ref)
        if found is None:
            names = ", ".join(sorted(self._material_names)[:20]) or "none"
            raise _err(
                "MATERIAL_NOT_FOUND",
                f"no material with id or name {ref!r} (materials: {names})",
                hint="create one with scene.create_material(...)",
            )
        return found

    def find_material(self, ref: Any) -> Material | None:
        if isinstance(ref, Material):
            return ref if ref._library is self else None
        if isinstance(ref, str):
            return self._materials.get(ref) or self._material_names.get(ref)
        return None

    def texture(self, ref: Any) -> Texture:
        """A texture of this scene by object, id or name."""
        found = self.find_texture(ref)
        if found is None:
            names = ", ".join(sorted(self._texture_names)[:20]) or "none"
            raise _err(
                "TEXTURE_NOT_FOUND", f"no texture with id or name {ref!r} (textures: {names})"
            )
        return found

    def find_texture(self, ref: Any) -> Texture | None:
        if isinstance(ref, Texture):
            return ref if ref._library is self else None
        if isinstance(ref, str):
            return self._textures.get(ref) or self._texture_names.get(ref)
        return None

    # -- creating -------------------------------------------------------------------

    def create_material(
        self,
        name: str | None = None,
        *,
        model: str = "standard",
        preset: str | None = None,
        material_id: str | None = None,
        **params: Any,
    ) -> Material:
        """Create a material (all parameters validated before anything is added).

        ``preset`` builds one of :func:`aether3d.materials.presets.preset_names`
        (stone, wood, metal, grass, brick, fabric) with its procedural
        textures; ``params`` then override the preset's parameters (and
        preset options such as ``scale``, ``seed`` or ``coordinates``).
        """
        if preset is not None:
            from .presets import build_preset

            label = "Create material" if name is None else f"Create material {name!r}"
            with self.transaction(label):
                return build_preset(self, preset, name, material_id=material_id, **params)
        if model not in MODELS:
            raise _err(
                "INVALID_MATERIAL",
                f"model must be one of {', '.join(MODELS)}, got {model!r}",
                "model",
            )
        values = default_params(model)
        location = f"materials[{name}]" if name else "material"
        for key, value in params.items():
            normalized = check_param(key, value, location)
            if key not in values:
                raise _err(
                    "PARAMETER_NOT_USED",
                    f"the {model} model has no {key!r} parameter",
                    f"{location}.{key}",
                )
            values[key] = normalized
        name = (
            self._unique_material_name("Material")
            if name is None
            else _check_name(name, "material")
        )
        if name in self._material_names:
            raise _err(
                "NAME_CONFLICT", f"a material named {name!r} already exists", f"materials[{name}]"
            )
        material_id = self._check_new_id(material_id, self._materials, "material")
        if len(self._materials) >= MAX_MATERIALS:
            raise _err("RESOURCE_LIMIT", f"a scene has at most {MAX_MATERIALS} materials")
        material = Material(self, material_id, name, model, values)
        h = _history_aspects()
        with self._op(f"Create material {name!r}", h.LIBRARY_MATERIALS, self):
            self._materials[material_id] = material
            self._material_names[name] = material
            self._changed()
        return material

    def add_image_texture(
        self,
        name: str | None = None,
        path: str | Path | None = None,
        *,
        data: bytes | None = None,
        embed: bool = False,
        texture_id: str | None = None,
        **sampler: Any,
    ) -> ImageTexture:
        """Add an image texture from a PNG/JPEG ``path`` or encoded ``data`` bytes.

        The image is read and decoded now, so a missing, malformed or
        oversized image is reported immediately (nothing is added).
        ``embed=True`` stores the file's bytes in the project. ``sampler``:
        ``wrap`` (``"repeat"``/``"clamp"``/``"mirror"`` or a pair), ``filter``
        (``"linear"``/``"nearest"``), ``mipmaps`` (bool), ``color_space``
        (``"auto"``/``"srgb"``/``"linear"``).
        """
        if (path is None) == (data is None):
            raise _err("INVALID_TEXTURE", "give a file path or image data (not both)")
        settings = check_sampler(sampler, "texture")
        unknown = set(sampler) - {"wrap", "wrap_u", "wrap_v", "filter", "mipmaps", "color_space"}
        if unknown:
            raise _err(
                "INVALID_TEXTURE", f"unknown image texture setting(s): {', '.join(sorted(unknown))}"
            )
        if name is None:
            base = Path(str(path)).stem if path is not None else "Image"
            name = self._unique_texture_name(base or "Image")
        _check_name(name, "texture")
        if name in self._texture_names:
            raise _err(
                "NAME_CONFLICT", f"a texture named {name!r} already exists", f"textures[{name}]"
            )
        texture_id = self._check_new_id(texture_id, self._textures, "texture")
        if len(self._textures) >= MAX_TEXTURES:
            raise _err("RESOURCE_LIMIT", f"a scene has at most {MAX_TEXTURES} textures")
        file = None
        if path is not None:
            raw = Path(path)
            file = raw if raw.is_absolute() else (self.base_dir or Path.cwd()) / raw
            file = Path(file).absolute()
        texture = ImageTexture(
            self,
            texture_id,
            name,
            file=file,
            data=bytes(data) if data is not None else None,
            **settings,
        )
        texture.load()  # validate now: missing or broken images are not added
        if embed:
            texture.embed()
        h = _history_aspects()
        with self._op(f"Add texture {name!r}", h.LIBRARY_TEXTURES, self):
            self._textures[texture_id] = texture
            self._texture_names[name] = texture
            self._changed()
        return texture

    def add_painted_texture(
        self,
        name: str | None = None,
        width: int = 1024,
        height: int | None = None,
        *,
        encoding: str = "color",
        background: Any = None,
        image: Any = None,
        texture_id: str | None = None,
        **sampler: Any,
    ) -> Any:
        """Add a painted texture (``docs/TEXTURING.md``): ``width x height`` texels of tiled
        layers over a ``background`` RGBA (default: opaque white, or a flat normal
        ``(0.5, 0.5, 1, 1)`` for ``encoding="normal"``). ``image`` (an image texture, a
        PNG/JPEG path or an ``(h, w, 3|4)`` array of 0..1) becomes a first paint layer;
        the size then defaults to the image's. ``encoding``: ``"color"`` (display sRGB
        colors), ``"data"`` (linear values) or ``"normal"`` (tangent-space normals).
        ``sampler``: as :meth:`add_image_texture`."""
        from .images import decode_image
        from .painted import Layer, PaintedTexture, check_document, tiles_from_pixels

        settings = check_sampler(sampler, "texture")
        unknown = set(sampler) - {"wrap", "wrap_u", "wrap_v", "filter", "mipmaps", "color_space"}
        if unknown:
            raise _err(
                "INVALID_TEXTURE", f"unknown texture setting(s): {', '.join(sorted(unknown))}"
            )
        pixels16 = None
        if image is not None:
            if isinstance(image, ImageTexture):
                rgba8 = image.load().image.pixels
                pixels16 = rgba8.astype(np.uint16) * 257
            elif isinstance(image, (str, Path)):
                raw = Path(image)
                file = raw if raw.is_absolute() else (self.base_dir or Path.cwd()) / raw
                data = Path(file).read_bytes()
                from .images import decode_png16, sniff_format

                if sniff_format(data[:16]) == "png":
                    pixels16 = decode_png16(data, location="image")
                else:
                    pixels16 = decode_image(data, location="image").pixels.astype(np.uint16) * 257
            else:
                arr = np.asarray(image, dtype=np.float64)
                if arr.ndim != 3 or arr.shape[2] not in (3, 4) or not np.all(np.isfinite(arr)):
                    raise _err(
                        "INVALID_TEXTURE", "image must be a finite (h, w, 3|4) array", "image"
                    )
                if arr.shape[2] == 3:
                    arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,))], axis=2)
                pixels16 = np.round(np.clip(arr, 0.0, 1.0) * 65535).astype(np.uint16)
            if height is None and width == 1024:
                height, width = pixels16.shape[:2]
        height = width if height is None else height
        if background is None:
            background = (0.5, 0.5, 1.0, 1.0) if encoding == "normal" else (1.0, 1.0, 1.0, 1.0)
        width, height, encoding, background = check_document(
            width, height, encoding, background, "texture"
        )
        if pixels16 is not None and pixels16.shape[:2] != (height, width):
            raise _err(
                "INVALID_TEXTURE",
                f"the image is {pixels16.shape[1]}x{pixels16.shape[0]}, not {width}x{height}",
                "image",
            )
        if name is None:
            name = self._unique_texture_name("Painted")
        _check_name(name, "texture")
        if name in self._texture_names:
            raise _err(
                "NAME_CONFLICT", f"a texture named {name!r} already exists", f"textures[{name}]"
            )
        texture_id = self._check_new_id(texture_id, self._textures, "texture")
        if len(self._textures) >= MAX_TEXTURES:
            raise _err("RESOURCE_LIMIT", f"a scene has at most {MAX_TEXTURES} textures")
        layers = ()
        if pixels16 is not None:
            import uuid

            tiles = tiles_from_pixels((width, height), pixels16, 0)
            layers = (Layer(uuid.uuid4().hex, "Image", "paint", tiles=tiles),)
        texture = PaintedTexture(
            self,
            texture_id,
            name,
            width=width,
            height=height,
            encoding=encoding,
            background=background,
            layers=layers,
            **settings,
        )
        h = _history_aspects()
        with self._op(f"Add texture {name!r}", h.LIBRARY_TEXTURES, self):
            self._textures[texture_id] = texture
            self._texture_names[name] = texture
            self._changed()
        return texture

    def add_procedural_texture(
        self, name: str | None, graph: Any, *, texture_id: str | None = None
    ) -> ProceduralTexture:
        """Add a procedural texture from a graph (dict), a single node dict or a node type name."""
        if isinstance(graph, str):
            graph = {"type": graph}
        name = (
            self._unique_texture_name("Procedural")
            if name is None
            else _check_name(name, "texture")
        )
        if name in self._texture_names:
            raise _err(
                "NAME_CONFLICT", f"a texture named {name!r} already exists", f"textures[{name}]"
            )
        location = f"textures[{name}]"
        parsed = TextureGraph.from_dict(graph, location)
        self._check_graph(parsed, location)
        texture_id = self._check_new_id(texture_id, self._textures, "texture")
        if len(self._textures) >= MAX_TEXTURES:
            raise _err("RESOURCE_LIMIT", f"a scene has at most {MAX_TEXTURES} textures")
        texture = ProceduralTexture(self, texture_id, name, parsed)
        h = _history_aspects()
        with self._op(f"Add texture {name!r}", h.LIBRARY_TEXTURES, self):
            self._textures[texture_id] = texture
            self._texture_names[name] = texture
            self._changed()
        return texture

    # -- removing -------------------------------------------------------------------

    def remove_material(self, ref: Any, *, force: bool = False) -> list[str]:
        """Remove a material. Objects using it keep their own colors again (only with
        ``force=True``; otherwise a used material is not removed). Returns the ids of the
        objects it was unassigned from."""
        material = self.material(ref)
        users = self.users(material)
        if users and not force:
            names = sorted({obj.name for obj, _ in users})
            raise _err(
                "MATERIAL_IN_USE",
                f"material {material.name!r} is used by {len(names)} object(s): "
                + ", ".join(names[:10]),
                material.location,
                hint="assign other materials first, or pass force=True to unassign it",
            )
        touched = []
        h = _history_aspects()
        with self._op(f"Remove material {material.name!r}", h.LIBRARY_MATERIALS, self):
            for obj, group in users:
                self._touch(h.MATERIALS, obj)
                if group is None:
                    obj._material = None
                else:
                    del obj._group_materials[group]
                touched.append(obj.id)
            del self._materials[material.id]
            del self._material_names[material.name]
            material._library = None
            animation = self._scene._animation
            if animation is not None:  # (its binding goes with it, in this step)
                animation._material_removed(material.id)
            self._changed()
        return sorted(set(touched))

    def remove_texture(self, ref: Any, *, force: bool = False) -> list[str]:
        """Remove a texture. If materials or graphs use it, this is refused unless
        ``force=True``, which removes those bindings (graphs that sample it are an error).
        Returns the ids of the materials whose bindings were removed."""
        texture = self.texture(ref)
        bound = [
            (m, slot)
            for m in self._materials.values()
            for slot, b in m._textures.items()
            if b.texture == texture.id
        ]
        graphs = [
            t
            for t in self._textures.values()
            if isinstance(t, ProceduralTexture)
            and any(self.find_texture(r) is texture for r in t.graph.image_references())
        ]
        if graphs:
            raise _err(
                "TEXTURE_IN_USE",
                f"texture {texture.name!r} is sampled by procedural texture(s) "
                + ", ".join(t.name for t in graphs),
                texture.location,
                hint="change or remove those procedural textures first",
            )
        if bound and not force:
            raise _err(
                "TEXTURE_IN_USE",
                f"texture {texture.name!r} is bound to "
                + ", ".join(f"{m.name}.{slot}" for m, slot in bound[:10]),
                texture.location,
                hint="clear those slots first, or pass force=True to remove the bindings",
            )
        h = _history_aspects()
        with self._op(f"Remove texture {texture.name!r}", h.LIBRARY_TEXTURES, self):
            for material, slot in bound:
                self._touch(h.MATERIAL, material)
                del material._textures[slot]
            del self._textures[texture.id]
            del self._texture_names[texture.name]
            texture._library = None
            self._changed()
        return sorted({m.id for m, _ in bound})

    # -- copying from another scene ------------------------------------------------

    def _copy_from(self, other: MaterialLibrary) -> dict[str | None, str | None]:
        """Copy every material and texture of ``other`` (new ids, free names); returns the
        new id of each copied material by its old id (``None`` maps to ``None``)."""
        import dataclasses

        if len(self._materials) + len(other._materials) > MAX_MATERIALS:
            raise _err("RESOURCE_LIMIT", f"a scene has at most {MAX_MATERIALS} materials")
        if len(self._textures) + len(other._textures) > MAX_TEXTURES:
            raise _err("RESOURCE_LIMIT", f"a scene has at most {MAX_TEXTURES} textures")
        h = _history_aspects()
        renamed: dict[str, str] = {}  # old id or name -> new id or name (graph references)
        textures: dict[str, Texture] = {}
        taken = set(self._texture_names)
        for t in other._textures.values():
            name = _unique(t._name, dict.fromkeys(taken))
            taken.add(name)
            if isinstance(t, ImageTexture):
                copy: Texture = ImageTexture(
                    self,
                    new_id(),
                    name,
                    file=t._file,
                    root=t._root,
                    data=t._data,
                    wrap=t._wrap,
                    filter=t._filter,
                    mipmaps=t._mipmaps,
                    color_space=t._color_space,
                )
            else:
                copy = ProceduralTexture(self, new_id(), name, t._graph)
            textures[t._id] = copy
            renamed[t._id], renamed[t._name] = copy._id, name
        for copy in textures.values():  # image nodes sample textures by id or name
            if isinstance(copy, ProceduralTexture) and copy._graph.image_references():
                data = copy._graph.to_dict()
                for node in data["nodes"].values():
                    if node.get("type") == "image":
                        node["texture"] = renamed.get(node["texture"], node["texture"])
                copy._graph = TextureGraph.from_dict(data, copy.location)
        materials: dict[str | None, str | None] = {None: None}
        copies = []
        taken = set(self._material_names)
        for m in other._materials.values():
            name = _unique(m._name, dict.fromkeys(taken))
            taken.add(name)
            bindings = {
                slot: dataclasses.replace(b, texture=textures[b.texture]._id)
                for slot, b in m._textures.items()
            }
            copy = Material(self, new_id(), name, m._model, dict(m._values), bindings)
            materials[m._id] = copy._id
            copies.append(copy)
        if textures:
            self._touch(h.LIBRARY_TEXTURES, self)
            for t in textures.values():
                self._textures[t._id] = t
                self._texture_names[t._name] = t
        if copies:
            self._touch(h.LIBRARY_MATERIALS, self)
            for m in copies:
                self._materials[m._id] = m
                self._material_names[m._name] = m
        if textures or copies:
            self._changed()
        return materials

    # -- assignment -----------------------------------------------------------------

    def assign(self, objects: Any, material: Any, *, group: int | None = None) -> list[SceneObject]:
        """Assign ``material`` (object, id, name; ``None`` to unassign) to mesh objects.

        ``group`` assigns it to one face group of each object's mesh instead
        of the whole object (see :attr:`aether3d.Mesh.groups`). All objects
        are checked before any is changed.
        """
        scene = self._scene
        if isinstance(objects, (str,)) or not isinstance(objects, Iterable):
            objects = [objects]
        targets = [scene._resolve_object(o) for o in objects]
        if not targets:
            raise _err("INVALID_ARGUMENT", "no objects given")
        mat = None if material is None else self.material(material)
        if group is not None and (
            isinstance(group, bool)
            or not isinstance(group, (int, np.integer))
            or not 0 <= group <= MAX_GROUP
        ):
            raise _err(
                "INVALID_ARGUMENT", f"group must be an integer in [0, {MAX_GROUP}], got {group!r}"
            )
        for obj in targets:
            if obj.camera is not None or obj.light is not None:
                raise _err(
                    "NOT_A_MESH_OBJECT",
                    f"object {obj.name!r} is a {obj.kind}; only mesh objects have materials",
                    f"objects[{obj.name}]",
                )
        h = _history_aspects()
        what = "Unassign material" if mat is None else f"Assign material {mat.name!r}"
        where = targets[0].name if len(targets) == 1 else f"{len(targets)} objects"
        with self._op(f"{what} ({where!r})" if len(targets) == 1 else f"{what} ({where})"):
            for obj in targets:
                self._touch(h.MATERIALS, obj)
                if group is None:
                    obj._material = None if mat is None else mat.id
                elif mat is None:
                    obj._group_materials.pop(int(group), None)
                else:
                    obj._group_materials[int(group)] = mat.id
            self._changed()
        return targets

    def users(self, material: Any) -> list[tuple[SceneObject, int | None]]:
        """``(object, group)`` pairs using ``material`` (group ``None``: the whole object)."""
        mat = self.material(material)
        out: list[tuple[SceneObject, int | None]] = []
        for obj in self._scene:
            if obj._material == mat.id:
                out.append((obj, None))
            for group, mid in sorted(obj._group_materials.items()):
                if mid == mat.id:
                    out.append((obj, group))
        return out

    def convert_appearances(
        self, objects: Any = None, *, dedupe: bool = True, prefix: str = "Color"
    ) -> list[Material]:
        """Turn object colors into equivalent ``"simple"`` materials and assign them.

        Each object's appearance (color, specular, shininess) becomes a simple
        material with the same values, which renders identically. With
        ``dedupe`` (default) objects with equal appearances share one
        material. Objects that already have a material are left unchanged.
        Returns the created materials.
        """
        scene = self._scene
        if objects is None:
            targets = [o for o in scene if (o._mesh is not None or o._geometry is not None)]
        else:
            if isinstance(objects, str) or not isinstance(objects, Iterable):
                objects = [objects]
            targets = [scene._resolve_object(o) for o in objects]
        targets = [
            o for o in targets if o._material is None and o.camera is None and o.light is None
        ]
        created: list[Material] = []
        h = _history_aspects()
        with self._op("Convert colors to materials"):
            shared: dict[tuple, Material] = {}
            for obj in targets:
                look = obj.appearance.to_dict()
                key = (tuple(look["color"]), look["specular"], look["shininess"])
                material = shared.get(key) if dedupe else None
                if material is None:
                    from ..rendering.color import color_to_hex

                    base = (
                        f"{prefix} {color_to_hex(look['color'])}"
                        if dedupe
                        else f"{obj.name} {prefix}"
                    )
                    if dedupe and (look["specular"] > 0.0 or look["shininess"] != 32.0):
                        base += f" spec {look['specular']:g}/{look['shininess']:g}"
                    material = self.create_material(
                        self._unique_material_name(base),
                        model="simple",
                        base_color=look["color"],
                        specular=look["specular"],
                        shininess=look["shininess"],
                    )
                    created.append(material)
                    shared[key] = material
                self._touch(h.MATERIALS, obj)
                obj._material = material.id
            self._changed()
        return created

    # -- validation and inspection --------------------------------------------------

    def validate(self) -> list[Diagnostic]:
        """Problems with materials, textures and assignments (nothing is changed).

        Errors: unreadable, missing or invalid images, unsafe paths.
        Warnings: objects whose mesh lacks the UVs a material needs, or the face groups
        its group materials need; objects whose procedural geometry cannot be evaluated
        when those checks need it (``GEOMETRY_EVALUATION_FAILED``; nothing is evaluated
        otherwise).
        Infos: textures and materials that nothing uses.
        """
        out: list[Diagnostic] = []
        for texture in self._textures.values():
            if isinstance(texture, ImageTexture):
                status = texture.status()
                if not status["ok"]:
                    out.append(
                        Diagnostic(status["code"], ERROR, status["message"], texture.location)
                    )
        used_textures = {b.texture for m in self._materials.values() for b in m._textures.values()}
        for t in self._textures.values():
            if isinstance(t, ProceduralTexture):
                used_textures.update(self.texture(r).id for r in t.graph.image_references())
        for texture in self._textures.values():
            if texture.id not in used_textures:
                out.append(
                    Diagnostic(
                        "UNUSED_TEXTURE",
                        INFO,
                        f"texture {texture.name!r} is not used",
                        texture.location,
                    )
                )
        used_materials: set[str] = set()
        for obj in self._scene:
            ids = ([obj._material] if obj._material else []) + list(obj._group_materials.values())
            used_materials.update(ids)
            # In id order: a missing material's error, or a material mapping textures by UV.
            checks: list[Diagnostic | tuple[Material, list[str]]] = []
            for mid in ids:
                material = self._materials.get(mid)
                if material is None:
                    checks.append(
                        Diagnostic(
                            "MATERIAL_NOT_FOUND",
                            ERROR,
                            f"material id {mid!r} does not exist",
                            f"objects[{obj.name}]",
                        )
                    )
                    continue
                needs = [s for s, b in material._textures.items() if b.coordinates == "uv"]
                if needs:
                    checks.append((material, needs))
            # The mesh is needed only for UV and face-group checks; procedural geometry is
            # evaluated only then, and a geometry that cannot be evaluated is reported.
            mesh = None
            wanted = obj._group_materials or any(isinstance(c, tuple) for c in checks)
            if wanted and (obj._mesh is not None or obj._geometry is not None):
                try:
                    mesh = obj.mesh
                except AetherError as exc:
                    out.append(
                        Diagnostic(
                            "GEOMETRY_EVALUATION_FAILED",
                            WARNING,
                            f"object {obj.name!r}: its geometry could not be evaluated "
                            f"({getattr(exc, 'code', type(exc).__name__)}), so its UVs and face "
                            "groups were not checked",
                            f"objects[{obj.name}]",
                        )
                    )
            for check in checks:
                if isinstance(check, Diagnostic):
                    out.append(check)
                    continue
                material, needs = check
                if mesh is not None and mesh.uvs is None:
                    out.append(
                        Diagnostic(
                            "MISSING_UVS",
                            WARNING,
                            f"object {obj.name!r} has no texture coordinates (UVs) but material "
                            f"{material.name!r} maps {', '.join(needs)} by UV; those textures are "
                            "ignored when rendering",
                            f"objects[{obj.name}]",
                        )
                    )
            if obj._group_materials and mesh is not None and mesh.groups is None:
                out.append(
                    Diagnostic(
                        "NO_FACE_GROUPS",
                        WARNING,
                        f"object {obj.name!r} has face-group materials but its mesh has no "
                        "face groups; "
                        "only group 0 (every face) uses them",
                        f"objects[{obj.name}]",
                    )
                )
        for material in self._materials.values():
            if material.id not in used_materials:
                out.append(
                    Diagnostic(
                        "UNUSED_MATERIAL",
                        INFO,
                        f"material {material.name!r} is not used",
                        material.location,
                    )
                )
        return out

    def inspect_material(self, material: Any) -> dict[str, Any]:
        mat = self.material(material)
        textures = {}
        for slot, binding in mat._textures.items():
            texture = self._textures.get(binding.texture)
            entry: dict[str, Any] = {**binding.to_dict()}
            if texture is None:
                entry["error"] = {
                    "code": "TEXTURE_NOT_FOUND",
                    "message": "the texture does not exist",
                }
            else:
                entry["texture_name"] = texture.name
                entry["texture_type"] = texture.kind
                entry["channel"] = binding.channel_for(slot, texture.kind)
                entry["status"] = (
                    texture.status() if isinstance(texture, ImageTexture) else {"ok": True}
                )
                if isinstance(texture, ImageTexture):
                    entry["decoded_as"] = texture.mode_for(SLOTS[slot].mode)
            textures[slot] = entry
        users = [
            {"id": obj.id, "name": obj.name, **({} if group is None else {"group": group})}
            for obj, group in self.users(mat)
        ]
        problems = [
            d.to_dict()
            for d in self.validate()
            if d.severity != INFO
            and (
                d.location == mat.location
                or (d.location or "").startswith("objects[")
                and mat.name in d.message
            )
        ]
        return {
            "id": mat.id,
            "name": mat.name,
            "model": mat.model,
            "parameters": mat.params,
            "textures": textures,
            "users": users,
            "problems": problems,
        }

    def inspect(self) -> dict[str, Any]:
        """Every material (with bindings and users) and texture (with status)."""
        return {
            "materials": [self.inspect_material(m) for m in self._materials.values()],
            "textures": [t.info() for t in self._textures.values()],
        }

    # -- persistence ----------------------------------------------------------------

    def to_dict(self, project_dir: Path | None = None) -> dict[str, list[dict[str, Any]]]:
        """Project records: ``{"materials": [...], "textures": [...]}``."""
        return {
            "materials": [m.to_dict() for m in self._materials.values()],
            "textures": [t.to_dict(project_dir) for t in self._textures.values()],
        }

    def load_records(
        self,
        materials: list[Any],
        textures: list[Any],
        project_dir: Path | None,
        location: str = "scene",
    ) -> None:
        """Rebuild the library from project records (strict; see docs/FILE_FORMAT.md).

        Image files are not read here: a missing file is reported when it is
        rendered or validated, never a reason to refuse the project.
        """
        from ..errors import ProjectFormatError

        def fail(exc: MaterialError, loc: str) -> ProjectFormatError:
            where = exc.location or loc
            return ProjectFormatError(f"[{exc.code}] {exc.message}", where)

        self.base_dir = project_dir
        for i, raw in enumerate(textures):
            loc = f"{location}.textures[{i}]"
            if not isinstance(raw, dict):
                raise ProjectFormatError("expected an object", loc)
            try:
                self._texture_from_record(raw, loc, project_dir)
            except MaterialError as exc:
                raise fail(exc, loc) from None
        for t in self._textures.values():
            if isinstance(t, ProceduralTexture):
                try:
                    self._check_graph(t.graph, t.location)
                except MaterialError as exc:
                    raise fail(exc, t.location) from None
        for i, raw in enumerate(materials):
            loc = f"{location}.materials[{i}]"
            if not isinstance(raw, dict):
                raise ProjectFormatError("expected an object", loc)
            try:
                self._material_from_record(raw, loc)
            except MaterialError as exc:
                raise fail(exc, loc) from None

    def _texture_from_record(self, raw: dict[str, Any], loc: str, project_dir: Path | None) -> None:
        kind = raw.get("type")
        common = {"id", "name", "type"}
        if kind == "image":
            allowed = common | {"path", "embedded", "wrap", "filter", "mipmaps", "color_space"}
        elif kind == "procedural":
            allowed = common | {"graph"}
        elif kind == "painted":
            allowed = common | {
                "width",
                "height",
                "encoding",
                "background",
                "layers",
                "wrap",
                "filter",
                "mipmaps",
                "color_space",
            }
        else:
            raise _err(
                "INVALID_TEXTURE",
                f"type must be 'image', 'painted' or 'procedural', got {kind!r}",
                f"{loc}.type",
            )
        missing = sorted({"id", "name"} - set(raw))
        unknown = sorted(set(raw) - allowed)
        if missing or unknown:
            what = f"missing {', '.join(missing)}" if missing else f"unknown {', '.join(unknown)}"
            raise _err("INVALID_TEXTURE", f"keys: {what}", loc)
        name = _check_name(raw["name"], "texture")
        texture_id = raw["id"]
        if not isinstance(texture_id, str) or not texture_id:
            raise _err("INVALID_TEXTURE", "id must be a non-empty string", f"{loc}.id")
        if texture_id in self._textures or name in self._texture_names:
            raise _err(
                "INVALID_TEXTURE", f"duplicate texture id or name ({texture_id!r}, {name!r})", loc
            )
        if kind == "painted":
            texture = _painted_from_record(self, raw, texture_id, name, loc)
        elif kind == "procedural":
            if "graph" not in raw:
                raise _err("INVALID_TEXTURE", "keys: missing graph", loc)
            graph = TextureGraph.from_dict(raw["graph"], f"{loc}.graph")
            compile_graph(graph, f"{loc}.graph")  # complexity limits hold for stored graphs too
            texture: Texture = ProceduralTexture(self, texture_id, name, graph)
        else:
            settings = check_sampler(
                {k: raw[k] for k in ("wrap", "filter", "mipmaps", "color_space") if k in raw}, loc
            )
            has_path, has_data = "path" in raw, "embedded" in raw
            if has_path == has_data:
                raise _err(
                    "INVALID_TEXTURE",
                    "an image texture needs exactly one of path and embedded",
                    loc,
                )
            if has_path:
                rel = check_relative_path(raw["path"], f"{loc}.path")
                base = project_dir if project_dir is not None else Path.cwd()
                texture = ImageTexture(
                    self,
                    texture_id,
                    name,
                    file=(base / rel).absolute(),
                    root=base.absolute(),
                    **settings,
                )
            else:
                texture = ImageTexture(
                    self,
                    texture_id,
                    name,
                    data=_embedded(raw["embedded"], f"{loc}.embedded"),
                    **settings,
                )
        self._textures[texture_id] = texture
        self._texture_names[name] = texture

    def _material_from_record(self, raw: dict[str, Any], loc: str) -> None:
        for key in ("id", "name", "model"):
            if key not in raw:
                raise _err("INVALID_MATERIAL", f"keys: missing {key}", loc)
        model = raw["model"]
        if model not in MODELS:
            raise _err(
                "INVALID_MATERIAL",
                f"model must be one of {', '.join(MODELS)}, got {model!r}",
                f"{loc}.model",
            )
        values = default_params(model)
        allowed = set(values) | {"id", "name", "model", "textures"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise _err(
                "INVALID_MATERIAL",
                f"unknown key(s) for a {model} material: {', '.join(unknown)}",
                loc,
            )
        for key in values:
            if key in raw:
                values[key] = check_param(key, raw[key], loc)
        name = _check_name(raw["name"], "material")
        material_id = raw["id"]
        if not isinstance(material_id, str) or not material_id:
            raise _err("INVALID_MATERIAL", "id must be a non-empty string", f"{loc}.id")
        if material_id in self._materials or name in self._material_names:
            raise _err(
                "INVALID_MATERIAL",
                f"duplicate material id or name ({material_id!r}, {name!r})",
                loc,
            )
        bindings: dict[str, TextureBinding] = {}
        raw_textures = raw.get("textures", {})
        if not isinstance(raw_textures, dict):
            raise _err(
                "INVALID_MATERIAL",
                "textures must be an object of {slot: binding}",
                f"{loc}.textures",
            )
        material = Material(self, material_id, name, model, values)
        for slot, binding_raw in raw_textures.items():
            sloc = f"{loc}.textures.{slot}"
            if slot not in SLOTS or model not in SLOTS[slot].models:
                raise _err("INVALID_MATERIAL", f"the {model} model has no {slot!r} slot", sloc)
            binding = binding_from_dict(slot, binding_raw, sloc)
            texture = self._textures.get(binding.texture)
            if texture is None:
                raise _err(
                    "TEXTURE_NOT_FOUND",
                    f"texture id {binding.texture!r} does not exist",
                    f"{sloc}.texture",
                )
            self._check_binding(material, slot, binding, texture)
            bindings[slot] = binding
        material._textures = bindings
        self._materials[material_id] = material
        self._material_names[name] = material

    # -- transactions ---------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything a rollback needs (materials, textures, names, object assignments)."""
        return {
            "materials": [
                (m, m._name, m._model, dict(m._values), dict(m._textures))
                for m in self._materials.values()
            ],
            "textures": [(t, t._name, _texture_state(t)) for t in self._textures.values()],
            "objects": [(o, o._material, dict(o._group_materials)) for o in self._scene],
            "base_dir": self.base_dir,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Put back the state of :meth:`snapshot` (recorded as one history step)."""
        h = _history_aspects()
        with self._op("Restore materials"):
            self._touch(h.LIBRARY_MATERIALS, self)
            self._touch(h.LIBRARY_TEXTURES, self)
            for m in (*self._materials.values(), *(m for m, *_ in snap["materials"])):
                self._touch(h.MATERIAL, m)
            for t in (*self._textures.values(), *(t for t, *_ in snap["textures"])):
                self._touch(h.TEXTURE, t)
            for obj, *_ in snap["objects"]:
                self._touch(h.MATERIALS, obj)
            self._restore(snap)

    def _restore(self, snap: dict[str, Any]) -> None:
        kept_materials = {id(m) for m, *_ in snap["materials"]}
        for m in self._materials.values():
            if id(m) not in kept_materials:
                m._library = None
        kept_textures = {id(t) for t, *_ in snap["textures"]}
        for t in self._textures.values():
            if id(t) not in kept_textures:
                t._library = None
        self._materials, self._material_names = {}, {}
        for m, name, model, values, textures in snap["materials"]:
            m._library, m._name, m._model, m._values, m._textures = (
                self,
                name,
                model,
                values,
                textures,
            )
            self._materials[m.id] = m
            self._material_names[name] = m
        self._textures, self._texture_names = {}, {}
        for t, name, state in snap["textures"]:
            t._library, t._name = self, name
            _set_texture_state(t, state)
            self._textures[t.id] = t
            self._texture_names[name] = t
        for obj, material, groups in snap["objects"]:
            obj._material = material
            obj._group_materials = groups
        self.base_dir = snap["base_dir"]
        self._changed()

    @contextlib.contextmanager
    def transaction(self, label: str = "Change materials") -> Iterator[MaterialLibrary]:
        """All-or-nothing changes: if the block raises, everything is restored.

        This is a transaction of the scene's history (one undo step): it covers
        every change made in the block, not only those of materials.
        """
        with self._scene._history.transaction(label):
            yield self

    _transaction = transaction

    def _op(self, label: str | None, aspect: Any = None, target: Any = None) -> Any:
        return self._scene._history._op(label, aspect, target)

    def _touch(self, aspect: Any, target: Any) -> None:
        self._scene._history._touch(aspect, target)

    # -- internals ------------------------------------------------------------------

    def _changed(self) -> None:
        self.version += 1

    def _check_new_id(self, value: Any, existing: dict, what: str) -> str:
        if value is None:
            return new_id()
        if not isinstance(value, str) or not value:
            raise _err("INVALID_ARGUMENT", f"{what} id must be a non-empty string, got {value!r}")
        if value in existing:
            raise _err("INVALID_ARGUMENT", f"a {what} with id {value!r} already exists")
        return value

    def _unique_material_name(self, base: str) -> str:
        return _unique(base, self._material_names)

    def _unique_texture_name(self, base: str) -> str:
        return _unique(base, self._texture_names)

    def _rename_material(self, material: Material, name: str) -> None:
        _check_name(name, "material")
        if name == material._name:
            return
        if name in self._material_names:
            raise _err(
                "NAME_CONFLICT", f"a material named {name!r} already exists", f"materials[{name}]"
            )
        h = _history_aspects()
        with self._op(f"Rename material {material._name!r} to {name!r}", h.MATERIAL, material):
            del self._material_names[material._name]
            material._name = name
            self._material_names[name] = material
            self._changed()

    def _rename_texture(self, texture: Texture, name: str) -> None:
        _check_name(name, "texture")
        if name == texture._name:
            return
        if name in self._texture_names:
            raise _err(
                "NAME_CONFLICT", f"a texture named {name!r} already exists", f"textures[{name}]"
            )
        old = texture._name
        h = _history_aspects()
        with self._op(f"Rename texture {old!r} to {name!r}", h.TEXTURE, texture):
            del self._texture_names[old]
            texture._name = name
            self._texture_names[name] = texture
            # Graphs that sample this texture by name follow the rename.
            for t in self._textures.values():
                if isinstance(t, ProceduralTexture) and old in t.graph.image_references():
                    data = t.graph.to_dict()
                    for node in data["nodes"].values():
                        if node.get("type") == "image" and node.get("texture") == old:
                            node["texture"] = name
                    self._touch(h.TEXTURE, t)
                    t._graph = TextureGraph.from_dict(data, t.location)
            self._changed()

    def _texture_or_file(self, value: Any, location: str) -> Texture:
        """A texture by object/id/name, or an image texture for a file path (reused if loaded)."""
        found = self.find_texture(value)
        if found is not None:
            return found
        if isinstance(value, Texture):
            raise _err("TEXTURE_NOT_FOUND", f"{value!r} belongs to another scene", location)
        if isinstance(value, (str, Path)):
            text = str(value)
            if not text.lower().endswith((".png", ".jpg", ".jpeg")):
                raise _err(
                    "TEXTURE_NOT_FOUND",
                    f"no texture named {text!r}, and it is not a .png/.jpg/.jpeg file path",
                    location,
                )
            raw = Path(text)
            file = (raw if raw.is_absolute() else (self.base_dir or Path.cwd()) / raw).absolute()
            for t in self._textures.values():
                if isinstance(t, ImageTexture) and t.file is not None and t.file == file:
                    return t
            return self.add_image_texture(None, file)
        raise _err(
            "INVALID_ARGUMENT",
            f"expected a texture, its id or name, or an image path, got {value!r}",
            location,
        )

    def _check_binding(
        self, material: Material, slot: str, binding: TextureBinding, texture: Texture
    ) -> None:
        spec = SLOTS[slot]
        location = f"{material.location}.textures.{slot}"
        if texture._library is not self:
            raise _err(
                "TEXTURE_NOT_FOUND", f"texture {texture.name!r} is not in this scene", location
            )
        if (
            spec.kind == "normal"
            and isinstance(texture, ProceduralTexture)
            and texture.graph.output_kind() != "color"
        ):
            raise _err(
                "INVALID_MATERIAL",
                f"the normal slot needs a texture that outputs a normal map (a color); "
                f"{texture.name!r} outputs a number",
                location,
                hint="use a bump node: {'type': 'bump', 'height': {'node': ...}}",
            )

    def _check_graph(self, graph: TextureGraph, location: str) -> None:
        for ref in graph.image_references():
            target = self.find_texture(ref)
            if target is None:
                raise _err(
                    "TEXTURE_NOT_FOUND", f"image node references unknown texture {ref!r}", location
                )
            if not isinstance(target, ImageTexture):
                raise _err(
                    "INVALID_TEXTURE_GRAPH",
                    f"image nodes sample image textures; {target.name!r} is procedural",
                    location,
                )
        compile_graph(graph, location)

    def _image_sampler(self):
        """The image sampler used by graph ``image`` nodes on the CPU."""

        def sample_image(ref: str, color_space: str, p, dpdx, dpdy):
            texture = self.texture(ref)
            assert isinstance(texture, ImageTexture)
            mode = "srgb" if color_space == "srgb" else "linear"
            chain, _ = texture.mip_chain(mode)
            from .sampling import lod

            lam = (
                lod(chain, dpdx[:, 0], dpdx[:, 1], dpdy[:, 0], dpdy[:, 1])
                if texture.mipmaps
                else None
            )
            return sample(
                chain, p[:, 0], p[:, 1], wrap=texture.wrap, filter=texture.filter, lam=lam
            )

        return sample_image


def _unique(base: str, taken: dict[str, Any]) -> str:
    if base not in taken:
        return base
    k = 1
    while f"{base}.{k:03d}" in taken:
        k += 1
    return f"{base}.{k:03d}"


def _embedded(raw: Any, loc: str, limit: int | None = None) -> bytes:
    import base64
    import binascii

    from .textures import MAX_EMBEDDED_BYTES

    if limit is not None:
        MAX_EMBEDDED_BYTES = limit  # noqa: N806 - the limit for this record

    if not isinstance(raw, dict) or set(raw) != {"mime_type", "data"}:
        raise _err("INVALID_TEXTURE", 'embedded must be {"mime_type": ..., "data": base64}', loc)
    if raw["mime_type"] not in ("image/png", "image/jpeg"):
        raise _err(
            "INVALID_TEXTURE",
            f"mime_type must be image/png or image/jpeg, got {raw['mime_type']!r}",
            loc,
        )
    text = raw["data"]
    if not isinstance(text, str) or len(text) > (MAX_EMBEDDED_BYTES * 4) // 3 + 4:
        raise _err(
            "INVALID_TEXTURE", "embedded data must be a base64 string within the size limit", loc
        )
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise _err("INVALID_TEXTURE", "embedded data is not valid base64", loc) from None


def _painted_from_record(library: Any, raw: dict, texture_id: str, name: str, loc: str):
    from .painted import PaintedTexture, check_document, layer_from_record

    for key in ("width", "height", "layers"):
        if key not in raw:
            raise _err("INVALID_TEXTURE", f"keys: missing {key}", loc)
    width, height, encoding, background = check_document(
        raw["width"],
        raw["height"],
        raw.get("encoding", "color"),
        raw.get("background", (1.0, 1.0, 1.0, 1.0)),
        loc,
    )
    settings = check_sampler(
        {k: raw[k] for k in ("wrap", "filter", "mipmaps", "color_space") if k in raw}, loc
    )
    if not isinstance(raw["layers"], list):
        raise _err("INVALID_TEXTURE", "layers must be a list", f"{loc}.layers")
    from .painted import MAX_LAYERS

    if len(raw["layers"]) > MAX_LAYERS:
        raise _err("RESOURCE_LIMIT", f"a painted texture has at most {MAX_LAYERS} layers", loc)

    def decode(value: Any) -> bytes:
        return _embedded(value, f"{loc}.layers", 256 * 2**20)

    layers = tuple(
        layer_from_record((width, height), item, decode, f"{loc}.layers[{i}]")
        for i, item in enumerate(raw["layers"])
    )
    ids = [layer.id for layer in layers]
    names = [layer.name for layer in layers]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise _err("INVALID_TEXTURE", "layer ids and names must be unique", f"{loc}.layers")
    return PaintedTexture(
        library,
        texture_id,
        name,
        width=width,
        height=height,
        encoding=encoding,
        background=background,
        layers=layers,
        **settings,
    )


def _texture_state(t: Texture) -> dict[str, Any]:
    from .painted import PaintedTexture

    if isinstance(t, PaintedTexture):
        return {
            "_width": t._width,
            "_height": t._height,
            "_encoding": t._encoding,
            "_layers": t._layers,
            "_background": t._background,
            "_wrap": t._wrap,
            "_filter": t._filter,
            "_mipmaps": t._mipmaps,
            "_color_space": t._color_space,
        }
    if isinstance(t, ImageTexture):
        return {
            "_file": t._file,
            "_root": t._root,
            "_data": t._data,
            "_wrap": t._wrap,
            "_filter": t._filter,
            "_mipmaps": t._mipmaps,
            "_color_space": t._color_space,
        }
    assert isinstance(t, ProceduralTexture)
    return {"_graph": t._graph}


def _set_texture_state(t: Texture, state: dict[str, Any]) -> None:
    from .painted import PaintedTexture

    if isinstance(t, PaintedTexture):
        t._restore(state)
        return
    for key, value in state.items():
        object.__setattr__(t, key, value)
