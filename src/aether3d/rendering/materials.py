"""Materials as the renderers see them: resolved once per render, shared by both backends.

:func:`resolve_materials` turns the materials used by visible objects into
immutable :class:`RenderMaterial` records -- linear-light factors, texture
bindings with their image levels already decoded (:class:`RenderImage`) or
their procedural graphs -- so backends never touch the scene. An object
without a material gets the ``"simple"`` material equivalent to its own
color (one record per distinct appearance).

Problems never abort a render; they become diagnostics and a documented
fallback:

* ``TEXTURE_MISSING`` / ``TEXTURE_INVALID`` / ``UNSAFE_PATH`` / ... -- an
  image that cannot be loaded: the binding is ignored (the slot uses its
  factor alone).
* ``MISSING_UVS`` -- an object's mesh has no UVs but its material maps a
  texture by UV: that binding is ignored for that object.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from ..errors import MaterialError
from .color import to_linear
from .settings import Appearance

if TYPE_CHECKING:
    from ..materials.graph import TextureGraph
    from ..materials.material import Material, TextureBinding
    from ..materials.sampling import MipChain
    from ..scene import Scene, SceneObject


@dataclass(frozen=True)
class RenderImage:
    """One image's levels, decoded for one mode, with its sampler settings."""

    name: str
    digest: str
    chain: MipChain
    wrap: tuple[str, str]
    filter: str
    mipmaps: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.digest, self.chain.mode)


@dataclass(frozen=True)
class RenderTexture:
    """A material slot's texture binding, resolved for rendering."""

    slot: str
    kind: str  # "image" | "procedural"
    texture_id: str
    name: str
    channel: str  # "rgb" | "r" | "g" | "b" | "a" | "luminance"
    coordinates: str  # "uv" | "object" | "world"
    projection: str  # "triplanar" | "xy" | "xz" | "yz" | "volume" (object/world)
    transform: np.ndarray  # (3, 4) affine map applied to the coordinates
    blend: float  # triplanar sharpness
    image: RenderImage | None = None  # kind "image"
    graph: TextureGraph | None = None  # kind "procedural"
    images: dict[tuple[str, str], RenderImage] = field(default_factory=dict)  # graph image nodes

    @property
    def needs_uvs(self) -> bool:
        return self.coordinates == "uv"

    @property
    def mikktspace(self) -> bool:
        """A UV normal map read in the MikkTSpace basis (see ``rendering/tangents.py``)."""
        from .tangents import uv_aligned

        return self.coordinates == "uv" and uv_aligned(self.transform)

    @property
    def key(self) -> tuple:
        """Bindings with equal keys evaluate to equal values (shared work)."""
        return (
            self.texture_id,
            self.image.chain.mode if self.image is not None else "",
            self.coordinates,
            self.projection,
            self.transform.tobytes(),
            self.blend,
        )


@dataclass(frozen=True)
class RenderMaterial:
    """A material in linear light, with resolved textures (see the module docstring)."""

    key: str
    name: str
    material_id: str | None  # None: an object's own color (appearance)
    model: str  # "standard" | "simple"
    base_color: np.ndarray  # (3,) linear
    opacity: float
    alpha_mode: str
    alpha_cutoff: float
    metallic: float
    roughness: float
    specular: float
    shininess: float
    emissive: np.ndarray  # (3,) linear, x emissive_strength
    normal_strength: float
    occlusion_strength: float
    double_sided: bool
    textures: dict[str, RenderTexture] = field(default_factory=dict)

    @property
    def masked(self) -> bool:
        return self.alpha_mode == "mask"

    @property
    def plain(self) -> bool:
        """No textures and no emission: shading needs only the factors."""
        return not self.textures and not np.any(self.emissive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "id": self.material_id,
            "model": self.model,
            "textures": {
                slot: {"texture": t.name, "type": t.kind, "coordinates": t.coordinates}
                for slot, t in self.textures.items()
            },
        }


def appearance_material(appearance: Appearance, key: str) -> RenderMaterial:
    """The simple material equivalent to an object's own color."""
    return RenderMaterial(
        key=key,
        name="(object color)",
        material_id=None,
        model="simple",
        base_color=to_linear(appearance.color),
        opacity=1.0,
        alpha_mode="opaque",
        alpha_cutoff=0.5,
        metallic=0.0,
        roughness=1.0,
        specular=appearance.specular,
        shininess=appearance.shininess,
        emissive=np.zeros(3),
        normal_strength=1.0,
        occlusion_strength=1.0,
        double_sided=True,
    )


@dataclass
class MaterialTable:
    """The materials of one render and, per mesh instance, which one each face group uses."""

    materials: list[RenderMaterial] = field(default_factory=list)
    diagnostics: list[tuple[int, Diagnostic]] = field(default_factory=list)  # (object position, d)
    texture_seconds: float = 0.0
    _index: dict[str, int] = field(default_factory=dict)
    _failed: dict[str, MaterialError] = field(default_factory=dict)
    _reported: set[tuple[str, str]] = field(default_factory=set)

    def add(self, material: RenderMaterial) -> int:
        index = self._index.get(material.key)
        if index is None:
            index = self._index[material.key] = len(self.materials)
            self.materials.append(material)
        return index


@dataclass
class _Resolved:
    """A library material resolved for rendering, with what resolving it reported and read."""

    material: RenderMaterial
    reports: list[tuple[str, Diagnostic]]  # (slot, diagnostic): replayed at its first use
    images: list[tuple[Any, str, str]]  # (texture, mode, digest or "error:CODE") it read


class MaterialCache:
    """Materials resolved for one scene, reused between renders (``scene._derived()``).

    A library material's resolution is valid for one ``MaterialLibrary.version`` --
    every change of a material, a texture or the library increments it, undo and redo
    included -- and while the images it read are unchanged: every render reads them
    again (as resolving does, so files changed on disk are picked up) and resolves the
    material afresh if one changed. Its diagnostics are reported again, at its first
    use in each render. An object's own color depends on nothing else. The results are
    exactly those of resolving everything again (``tests/test_invariants.py`` compares
    them with a fresh copy of the scene along random edits).
    """

    def __init__(self) -> None:
        self.library: Any = None
        self.version = -1
        self.resolved: dict[str, _Resolved] = {}
        self.appearances: dict[bytes, RenderMaterial] = {}


_APPEARANCE = struct.Struct("<5d")


def _appearance_key(look: Appearance) -> bytes:
    """An appearance's exact values as bytes: equal keys give equal material keys
    (``repr``), including the sign of zeros (``-0.0 == 0.0``, but their reprs differ)."""
    return _APPEARANCE.pack(*look.color, look.specular, look.shininess)


def resolve_materials(
    scene: Scene,
    candidates: list[tuple[SceneObject, Any, int]],
    cache: MaterialCache | None = None,
) -> tuple[MaterialTable, list[tuple[int, dict[int, int]]]]:
    """Materials of ``candidates`` (``(object, mesh, position in the scene)``).

    Returns the table and, per candidate, ``(default material index,
    {group: material index})``. With a ``cache`` (see :class:`MaterialCache`), work
    from earlier renders of the scene is reused; the results are the same.
    """
    table = MaterialTable()
    library = scene.materials
    cache = MaterialCache() if cache is None else cache
    _prepare(cache, library)
    this_render: dict[str, RenderMaterial] = {}  # library materials resolved, by id
    per_object: list[tuple[int, dict[int, int]]] = []
    t0 = time.perf_counter()
    for obj, mesh, position in candidates:
        if obj._material is not None:
            default = table.add(
                _material(library.material(obj._material), cache, this_render, table, position)
            )
        else:
            look = obj.appearance
            akey = _appearance_key(look)
            material = cache.appearances.get(akey)
            if material is None:
                key = "appearance:" + repr((look.color, look.specular, look.shininess))
                material = cache.appearances[akey] = appearance_material(look, key)
            default = table._index.get(material.key)  # objects of one color share it
            if default is None:
                default = table.add(material)
        groups: dict[int, int] = {}
        for group, ref in obj._group_materials.items():
            groups[group] = table.add(
                _material(library.material(ref), cache, this_render, table, position)
            )
        if mesh.uvs is None:
            used = [table.materials[default], *(table.materials[i] for i in groups.values())]
            table.diagnostics += _missing_uvs(obj, used, position)
        per_object.append((default, groups))
    table.texture_seconds = time.perf_counter() - t0
    return table, per_object


def _prepare(cache: MaterialCache, library: Any) -> None:
    """Forget library resolutions made for another library or version of it."""
    if cache.library is not library or cache.version != library.version:
        cache.library, cache.version = library, library.version
        cache.resolved.clear()


def _missing_uvs(
    obj: SceneObject, used: list[RenderMaterial], position: int
) -> list[tuple[int, Diagnostic]]:
    """``MISSING_UVS`` for each material in ``used`` (the object's, in order: its default
    then its face groups') that maps a texture by UV, ``obj``'s mesh having no UVs."""
    out = []
    for material in used:
        slots = sorted(s for s, t in material.textures.items() if t.needs_uvs)
        if slots:
            out.append(
                (
                    position,
                    Diagnostic(
                        "MISSING_UVS",
                        WARNING,
                        f"object {obj.name!r} has no texture coordinates (UVs), so the "
                        f"{', '.join(slots)} texture(s) of material {material.name!r} "
                        "were ignored on it",
                        f"objects[{obj.name}]",
                    ),
                )
            )
    return out


def _material(
    material: Material,
    cache: MaterialCache,
    this_render: dict[str, RenderMaterial],
    table: MaterialTable,
    position: int,
) -> RenderMaterial:
    """``material`` resolved for this render: first use reports its diagnostics."""
    found = this_render.get(material.id)
    if found is not None:
        return found
    entry = cache.resolved.get(material.id)
    if entry is not None and all(_image_state(t, mode) == state for t, mode, state in entry.images):
        for slot, diagnostic in entry.reports:
            if (material.id, slot) not in table._reported:
                table._reported.add((material.id, slot))
                table.diagnostics.append((position, diagnostic))
    else:
        start = len(table.diagnostics)
        images: list[tuple[Any, str, str]] = []
        resolved = _resolve(material, table, position, images)
        reports = [
            (slot, diagnostic)
            for (_, diagnostic), slot in zip(
                table.diagnostics[start:], _report_slots(table, material, start), strict=True
            )
        ]
        entry = cache.resolved[material.id] = _Resolved(resolved, reports, images)
    this_render[material.id] = entry.material
    return entry.material


def _report_slots(table: MaterialTable, material: Material, start: int) -> list[str]:
    """The slots of the diagnostics ``material`` reported from ``start`` (their locations)."""
    prefix = f"materials[{material.name}].textures."
    slots = []
    for _, diagnostic in table.diagnostics[start:]:
        location = diagnostic.location or ""
        slots.append(location[len(prefix) :] if location.startswith(prefix) else "")
    return slots


def _image_state(texture: Any, mode: str) -> str:
    """The digest of the image ``texture`` gives in ``mode`` now (reads the file as
    resolving does), or ``"error:CODE"``."""
    try:
        _, loaded = texture.mip_chain(mode)
    except MaterialError as exc:
        return f"error:{exc.code}"
    return loaded.digest


def _resolve(
    material: Material,
    table: MaterialTable,
    position: int,
    images: list[tuple[Any, str, str]] | None = None,
) -> RenderMaterial:
    values = material._values
    textures: dict[str, RenderTexture] = {}
    for slot, binding in material._textures.items():
        texture = _resolve_texture(material, slot, binding, table, position, images)
        if texture is not None:
            textures[slot] = texture
    model = material.model
    out = RenderMaterial(
        key=f"material:{material.id}",
        name=material.name,
        material_id=material.id,
        model=model,
        base_color=to_linear(values["base_color"]),
        opacity=float(values["opacity"]),
        alpha_mode=values["alpha_mode"],
        alpha_cutoff=float(values["alpha_cutoff"]),
        metallic=float(values.get("metallic", 0.0)),
        roughness=float(values.get("roughness", 1.0)),
        specular=float(values["specular"]),
        shininess=float(values.get("shininess", 32.0)),
        emissive=to_linear(values["emissive"]) * float(values["emissive_strength"]),
        normal_strength=float(values["normal_strength"]),
        occlusion_strength=float(values["occlusion_strength"]),
        double_sided=bool(values["double_sided"]),
        textures=textures,
    )
    return out


def _resolve_texture(
    material: Material,
    slot: str,
    binding: TextureBinding,
    table: MaterialTable,
    position: int,
    images: list[tuple[Any, str, str]] | None = None,
) -> RenderTexture | None:
    from ..materials.material import SLOTS
    from ..materials.textures import ImageTexture, ProceduralTexture

    library = material._library
    texture = library.find_texture(binding.texture) if library is not None else None
    common: dict[str, Any] = {
        "slot": slot,
        "texture_id": binding.texture,
        "coordinates": binding.coordinates,
        "projection": binding.projection if binding.coordinates != "uv" else "triplanar",
        "transform": binding.matrix,
        "blend": float(binding.blend),
    }
    if texture is None:
        _report(
            table,
            position,
            material,
            slot,
            binding.texture,
            "TEXTURE_NOT_FOUND",
            "the texture does not exist",
        )
        return None
    common["name"] = texture.name
    common["channel"] = binding.channel_for(slot, texture.kind)
    try:
        if isinstance(texture, ImageTexture):
            mode = texture.mode_for(SLOTS[slot].mode)
            return RenderTexture(kind="image", image=_image(texture, mode, table, images), **common)
        assert isinstance(texture, ProceduralTexture)
        node_images: dict[tuple[str, str], RenderImage] = {}
        for ref in texture.graph.image_references():
            for node in texture.graph.nodes.values():
                if node.type == "image" and node.params["texture"] == ref:
                    space = node.params["color_space"]
                    target = library.texture(ref)
                    assert isinstance(target, ImageTexture)
                    node_images[(ref, space)] = _image(
                        target, "srgb" if space == "srgb" else "linear", table, images
                    )
        return RenderTexture(kind="procedural", graph=texture.graph, images=node_images, **common)
    except MaterialError as exc:
        _report(table, position, material, slot, texture.name, exc.code, exc.message)
        return None


def _image(
    texture: Any, mode: str, table: MaterialTable, images: list | None = None
) -> RenderImage:
    key = f"{texture.id}:{mode}"
    if key in table._failed:
        if images is not None:
            images.append((texture, mode, f"error:{table._failed[key].code}"))
        raise table._failed[key]
    try:
        chain, loaded = texture.mip_chain(mode)
    except MaterialError as exc:
        table._failed[key] = exc
        if images is not None:
            images.append((texture, mode, f"error:{exc.code}"))
        raise
    if images is not None:
        images.append((texture, mode, loaded.digest))
    return RenderImage(
        texture.name, loaded.digest, chain, texture.wrap, texture.filter, texture.mipmaps
    )


def _report(
    table: MaterialTable,
    position: int,
    material: Material,
    slot: str,
    texture: str,
    code: str,
    message: str,
) -> None:
    if (material.id, slot) in table._reported:
        return
    table._reported.add((material.id, slot))
    table.diagnostics.append(
        (
            position,
            Diagnostic(
                code,
                WARNING,
                f"material {material.name!r}: the {slot} texture {texture!r} cannot be used "
                f"({message}); the slot renders with its factor alone",
                f"materials[{material.name}].textures.{slot}",
            ),
        )
    )
