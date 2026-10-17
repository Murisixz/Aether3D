"""glTF materials: the pbrMetallicRoughness subset, in and out.

**Import** (:class:`MaterialImporter`): every glTF material used by imported
nodes becomes a ``"standard"`` Aether3D material -- base color and alpha,
metallic, roughness, emission (with ``KHR_materials_emissive_strength``),
alpha mode (``OPAQUE``, ``MASK``; ``BLEND`` is imported opaque with a
warning, as Aether3D draws no blended transparency), double-sidedness and
dielectric reflectance (``KHR_materials_specular``) -- with its textures:
base color, metallic-roughness (blue = metallic, green = roughness), normal
(``scale``), occlusion (``strength``, red) and emissive, including samplers
and ``KHR_texture_transform``. Images (PNG, JPEG) are read from buffer views,
data URIs or files next to the asset (never outside its directory), decoded
and embedded in the scene. Anything else -- other material extensions,
second texture coordinates, other image formats, missing or broken images
-- is reported; the rest of the material is kept. A material exported by
Aether3D carries its exact parameters in ``extras.aether3d``, which restores
``"simple"`` materials and object colors exactly.

**Export** (:class:`MaterialExporter`): each object's material (per face
group) becomes a glTF material; an object without one exports its color
(the default light gray needs no material).
Image textures keep their original PNG/JPEG bytes. What glTF cannot express
-- textures mapped by object/world coordinates, separate opacity textures,
unusual channel packings, procedural textures -- is left out with a
warning, except that procedural textures can be *baked* to PNG images when
explicitly requested (``export_gltf(..., bake_textures=1024)``).
"""

from __future__ import annotations

import base64
import binascii
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import AssetFormatError, MaterialError
from ..rendering.color import linear_to_srgb, srgb_to_linear

if TYPE_CHECKING:
    from ..materials import Material, MaterialLibrary, Texture
    from ..scene import Scene, SceneObject
    from .reader import _Reader

EXTRAS_KEY = "aether3d"
WRAP_FROM_GLTF = {10497: "repeat", 33071: "clamp", 33648: "mirror"}
WRAP_TO_GLTF = {v: k for k, v in WRAP_FROM_GLTF.items()}
NEAREST, LINEAR = 9728, 9729
MIPMAP_FILTERS = (9984, 9985, 9986, 9987)
SUPPORTED_MATERIAL_EXTENSIONS = (
    "KHR_texture_transform",
    "KHR_materials_emissive_strength",
    "KHR_materials_specular",
)
DEFAULT_SPECULAR = 0.5


def _srgb(linear: Any) -> tuple[float, float, float]:
    c = np.clip(np.asarray(linear, dtype=np.float64)[:3], 0.0, 1.0)
    out = linear_to_srgb(c)
    return (float(out[0]), float(out[1]), float(out[2]))


def _linear(srgb: Any) -> list[float]:
    return [float(v) for v in srgb_to_linear(np.asarray(srgb, dtype=np.float64))]


def roughness_from_shininess(specular: float, shininess: float) -> float:
    """A GGX roughness close to a Blinn-Phong highlight (``alpha^2 = 2 / (n + 2)``)."""
    if specular <= 0.0:
        return 1.0
    return float((2.0 / (shininess + 2.0)) ** 0.25)


def transform_from_gltf(ext: dict[str, Any]) -> dict[str, Any]:
    """A ``KHR_texture_transform`` (glTF uv, origin top-left) as Aether3D binding options.

    glTF maps ``uv' = T(o) R(t) S(s) uv`` with ``v`` pointing down; in
    Aether3D's flipped uv space the same map is ``R(-t) S(s)`` followed by the
    offset ``(o.x - sin(t) s.y, 1 - o.y - cos(t) s.y)``.
    """
    ox, oy = ext.get("offset", [0.0, 0.0])
    t = float(ext.get("rotation", 0.0))
    sx, sy = ext.get("scale", [1.0, 1.0])
    return {
        "offset": [ox - math.sin(t) * sy, 1.0 - oy - math.cos(t) * sy, 0.0],
        "rotation": -math.degrees(t),
        "scale": [sx, sy, 1.0],
    }


def transform_to_gltf(offset: Any, rotation: float, scale: Any) -> dict[str, Any] | None:
    """The inverse of :func:`transform_from_gltf` (``None`` for the identity)."""
    t = -math.radians(rotation)
    sx, sy = float(scale[0]), float(scale[1])
    ox = float(offset[0]) + math.sin(t) * sy
    oy = 1.0 - float(offset[1]) - math.cos(t) * sy
    out: dict[str, Any] = {}
    if abs(ox) > 1e-12 or abs(oy) > 1e-12:
        out["offset"] = [ox, oy]
    if abs(t) > 1e-12:
        out["rotation"] = t
    if sx != 1.0 or sy != 1.0:
        out["scale"] = [sx, sy]
    return out or None


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


class MaterialImporter:
    """Turns the glTF materials that imported nodes use into scene materials."""

    def __init__(self, reader: _Reader, scene: Scene) -> None:
        self.r = reader
        self.scene = scene
        self.library: MaterialLibrary = scene.materials
        self.results: dict[int, tuple[str, Any] | None] = {}
        self.textures: dict[tuple[int, int | None], Texture | None] = {}
        self.images: dict[int, bytes | None] = {}

    # -- materials -----------------------------------------------------------------

    def material(self, index: int, loc: str) -> tuple[str, Any] | None:
        """``("material", Material)``, ``("appearance", dict)`` (an Aether3D object color) or
        ``None`` for glTF material ``index``."""
        if index not in self.results:
            self.results[index] = self._convert(index, loc)
        return self.results[index]

    def unused(self) -> int:
        return len(self.r.array("materials")) - len(self.results)

    def _convert(self, index: int, loc: str) -> tuple[str, Any] | None:
        gm, where = self.r.item("materials", index, loc)
        extras = gm.get("extras")
        stored = extras.get(EXTRAS_KEY) if isinstance(extras, dict) else None
        if isinstance(stored, dict) and "appearance" in stored:
            appearance = self._stored_appearance(stored["appearance"], where)
            if appearance is not None:
                return ("appearance", appearance)
        extensions = gm.get("extensions", {})
        if not isinstance(extensions, dict):
            raise AssetFormatError("extensions must be an object", f"{where}/extensions")
        for name in sorted(set(extensions) - set(SUPPORTED_MATERIAL_EXTENSIONS)):
            self.r.note(
                "MATERIAL_EXTENSION_IGNORED",
                INFO,
                f"material extension {name!r} is not supported; the material is imported "
                "without it",
                f"{where}/extensions/{name}",
            )
        pbr = gm.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict):
            raise AssetFormatError("pbrMetallicRoughness must be an object", where)
        base = self._factor(
            pbr, "baseColorFactor", 4, [1.0, 1.0, 1.0, 1.0], f"{where}/pbrMetallicRoughness"
        )
        emissive = self._factor(gm, "emissiveFactor", 3, [0.0, 0.0, 0.0], where)
        params: dict[str, Any] = {
            "base_color": _srgb(base[:3]),
            "opacity": float(base[3]),
            "metallic": self._unit(pbr, "metallicFactor", 1.0, f"{where}/pbrMetallicRoughness"),
            "roughness": self._unit(pbr, "roughnessFactor", 1.0, f"{where}/pbrMetallicRoughness"),
            "emissive": _srgb(emissive),
            "double_sided": bool(gm.get("doubleSided", False)),
        }
        strength = extensions.get("KHR_materials_emissive_strength", {})
        if isinstance(strength, dict) and "emissiveStrength" in strength:
            value = strength["emissiveStrength"]
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 <= value <= 1e6
            ):
                params["emissive_strength"] = float(value)
        mode = gm.get("alphaMode", "OPAQUE")
        if mode == "MASK":
            params["alpha_mode"] = "mask"
            params["alpha_cutoff"] = self._unit(gm, "alphaCutoff", 0.5, where)
        elif mode == "BLEND":
            self.r.note(
                "ALPHA_BLEND_UNSUPPORTED",
                WARNING,
                "alphaMode BLEND (transparency) is not rendered by Aether3D; the material is "
                "imported opaque (its alpha is kept as opacity)",
                f"{where}/alphaMode",
            )
        elif mode != "OPAQUE":
            raise AssetFormatError(f"invalid alphaMode {mode!r}", f"{where}/alphaMode")
        specular = extensions.get("KHR_materials_specular")
        if isinstance(specular, dict):
            factor = specular.get("specularFactor", 1.0)
            color = specular.get("specularColorFactor", [1.0, 1.0, 1.0])
            if isinstance(color, list) and len(color) == 3 and len(set(color)) > 1:
                self.r.note(
                    "SPECULAR_COLOR_APPROXIMATED",
                    INFO,
                    "a colored specularColorFactor is imported as its average",
                    f"{where}/extensions/KHR_materials_specular",
                )
            try:
                level = float(factor) * float(np.mean(color)) * DEFAULT_SPECULAR
                params["specular"] = float(min(max(level, 0.0), 1.0))
            except (TypeError, ValueError):
                raise AssetFormatError(
                    "invalid KHR_materials_specular values", f"{where}/extensions"
                ) from None
            if "specularTexture" in specular or "specularColorTexture" in specular:
                self.r.note(
                    "SPECULAR_TEXTURE_IGNORED",
                    INFO,
                    "specular textures are not supported; the factors are imported",
                    f"{where}/extensions/KHR_materials_specular",
                )
        model = "standard"
        if isinstance(stored, dict) and "material" in stored:
            restored = self._stored_material(stored["material"], where)
            if restored is not None:
                model, params = restored
        name = gm.get("name")
        base_name = name.strip() if isinstance(name, str) and name.strip() else f"Material {index}"
        material_id = None  # keep an Aether3D material's id when the file stores it
        stored_id = stored.get("id") if isinstance(stored, dict) else None
        if isinstance(stored_id, str) and stored_id and stored_id not in self.library._materials:
            material_id = stored_id
        try:
            material = self.library.create_material(
                self.library._unique_material_name(base_name),
                model=model,
                material_id=material_id,
                **params,
            )
        except MaterialError as exc:
            raise AssetFormatError(
                f"invalid material: {exc.message}", where, "INVALID_MATERIAL"
            ) from None
        self._bind(material, pbr, gm, where)
        return ("material", material)

    def _factor(
        self, obj: dict, key: str, count: int, default: list[float], loc: str
    ) -> list[float]:
        values = self.r.numbers(obj, key, count, loc)
        if values is None:
            return default
        if not all(math.isfinite(v) for v in values):
            raise AssetFormatError(f"{key} must be finite", f"{loc}/{key}")
        if any(v < 0.0 or v > 1.0 for v in values):
            self.r.note(
                "FACTOR_CLAMPED",
                WARNING,
                f"{key} values must lie in [0, 1]; they were clamped",
                f"{loc}/{key}",
            )
        return [min(max(float(v), 0.0), 1.0) for v in values]

    def _unit(self, obj: dict, key: str, default: float, loc: str) -> float:
        value = obj.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise AssetFormatError(f"{key} must be a number", f"{loc}/{key}")
        if not 0.0 <= value <= 1.0:
            self.r.note(
                "FACTOR_CLAMPED", WARNING, f"{key} must lie in [0, 1]; clamped", f"{loc}/{key}"
            )
        return float(min(max(value, 0.0), 1.0))

    def _stored_appearance(self, raw: Any, where: str) -> dict | None:
        from ..rendering.settings import Appearance

        try:
            return Appearance.from_dict(raw, "appearance").to_dict()
        except Exception:
            self.r.note(
                "STORED_MATERIAL_REJECTED",
                WARNING,
                "extras.aether3d.appearance is invalid; the glTF material is used instead",
                f"{where}/extras",
            )
            return None

    def _stored_material(self, raw: Any, where: str) -> tuple[str, dict] | None:
        from ..materials.material import MODELS, check_param, default_params

        try:
            if not isinstance(raw, dict) or raw.get("model") not in MODELS:
                raise ValueError("model")
            model = raw["model"]
            values = default_params(model)
            for key, value in raw.items():
                if key == "model":
                    continue
                if key not in values:
                    raise ValueError(key)
                values[key] = check_param(key, value, "material")
            return model, values
        except (ValueError, MaterialError):
            self.r.note(
                "STORED_MATERIAL_REJECTED",
                WARNING,
                "extras.aether3d.material is invalid; the glTF material parameters are used",
                f"{where}/extras",
            )
            return None

    # -- textures ------------------------------------------------------------------

    def _bind(self, material: Material, pbr: dict, gm: dict, where: str) -> None:
        slots = [
            (pbr, "baseColorTexture", ["base_color"], f"{where}/pbrMetallicRoughness"),
            (
                pbr,
                "metallicRoughnessTexture",
                ["metallic", "roughness"],
                f"{where}/pbrMetallicRoughness",
            ),
            (gm, "normalTexture", ["normal"], where),
            (gm, "occlusionTexture", ["occlusion"], where),
            (gm, "emissiveTexture", ["emissive"], where),
        ]
        for owner, key, targets, loc in slots:
            info = owner.get(key)
            if info is None:
                continue
            bound = self._texture_info(info, f"{loc}/{key}")
            if bound is None:
                continue
            texture, options = bound
            for slot in targets:
                if (
                    slot not in ("base_color", "normal", "occlusion", "emissive")
                    and material.model != "standard"
                ):
                    continue
                channel = {"metallic": "b", "roughness": "g", "occlusion": "r"}.get(slot)
                if channel is not None:
                    options = {**options, "channel": channel}
                else:
                    options = {k: v for k, v in options.items() if k != "channel"}
                material.set_texture(slot, texture, **options)
            if key == "normalTexture" and "scale" in info:
                material.configure(
                    normal_strength=self._number(info, "scale", -10.0, 10.0, f"{loc}/{key}")
                )
            if key == "occlusionTexture" and "strength" in info:
                material.configure(
                    occlusion_strength=self._number(info, "strength", 0.0, 1.0, f"{loc}/{key}")
                )

    def _number(self, obj: dict, key: str, lo: float, hi: float, loc: str) -> float:
        value = obj[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise AssetFormatError(f"{key} must be a number", f"{loc}/{key}")
        if not lo <= value <= hi:
            self.r.note(
                "FACTOR_CLAMPED",
                WARNING,
                f"{key} must lie in [{lo:g}, {hi:g}]; clamped",
                f"{loc}/{key}",
            )
        return float(min(max(value, lo), hi))

    def _texture_info(self, info: Any, loc: str) -> tuple[Texture, dict[str, Any]] | None:
        if not isinstance(info, dict):
            raise AssetFormatError("textureInfo must be an object", loc)
        tex_coord = info.get("texCoord", 0)
        options: dict[str, Any] = {}
        extensions = info.get("extensions", {})
        if not isinstance(extensions, dict):
            raise AssetFormatError("extensions must be an object", f"{loc}/extensions")
        transform = extensions.get("KHR_texture_transform")
        if transform is not None:
            if not isinstance(transform, dict):
                raise AssetFormatError(
                    "KHR_texture_transform must be an object", f"{loc}/extensions"
                )
            for key, count in (("offset", 2), ("scale", 2)):
                if key in transform:
                    self.r.numbers(transform, key, count, f"{loc}/extensions/KHR_texture_transform")
            rotation = transform.get("rotation", 0.0)
            if isinstance(rotation, bool) or not isinstance(rotation, (int, float)):
                raise AssetFormatError(
                    "rotation must be a number", f"{loc}/extensions/KHR_texture_transform"
                )
            tex_coord = transform.get("texCoord", tex_coord)
            options.update(transform_from_gltf(transform))
        if tex_coord != 0:
            self.r.note(
                "TEXCOORD_UNSUPPORTED",
                WARNING,
                f"the texture uses TEXCOORD_{tex_coord}; only TEXCOORD_0 is imported, so this "
                "texture is left out",
                loc,
            )
            return None
        index = info.get("index")
        tex, tex_where = self.r.item("textures", index, f"{loc}/index")
        sampler = tex.get("sampler")
        key = (index, sampler)
        if key not in self.textures:
            self.textures[key] = self._texture(tex, tex_where, sampler)
        texture = self.textures[key]
        return None if texture is None else (texture, options)

    def _texture(self, tex: dict, where: str, sampler_index: Any) -> Texture | None:
        if "source" not in tex:
            names = (
                sorted(tex.get("extensions", {})) if isinstance(tex.get("extensions"), dict) else []
            )
            self.r.note(
                "TEXTURE_FORMAT_UNSUPPORTED",
                WARNING,
                "the texture has no PNG/JPEG source"
                + (f" (only {', '.join(names)})" if names else "")
                + "; it is left out",
                where,
            )
            return None
        image_index = tex["source"]
        image, image_where = self.r.item("images", image_index, f"{where}/source")
        data = self._image_bytes(image_index, image, image_where)
        if data is None:
            return None
        settings = self._sampler(sampler_index, f"{where}/sampler")
        name = image.get("name")
        if not (isinstance(name, str) and name.strip()):
            uri = image.get("uri")
            name = (
                uri.rsplit("/", 1)[-1]
                if isinstance(uri, str) and not uri.startswith("data:")
                else f"Image {image_index}"
            )
        try:
            return self.library.add_image_texture(
                self.library._unique_texture_name(name.strip()), data=data, **settings
            )
        except MaterialError as exc:
            self.r.note(
                exc.code if exc.code != "INVALID_IMAGE" else "TEXTURE_INVALID",
                WARNING,
                f"image cannot be used ({exc.message}); materials use their factors instead",
                image_where,
            )
            self.images[image_index] = None
            return None

    def _image_bytes(self, index: int, image: dict, where: str) -> bytes | None:
        if index in self.images:
            return self.images[index]
        data: bytes | None = None
        if "bufferView" in image:
            mime = image.get("mimeType")
            if mime not in ("image/png", "image/jpeg"):
                self.r.note(
                    "IMAGE_FORMAT_UNSUPPORTED",
                    WARNING,
                    f"image mimeType {mime!r} is not supported (PNG and JPEG are); left out",
                    where,
                )
                self.images[index] = None
                return None
            view, view_loc = self.r.item("bufferViews", image["bufferView"], f"{where}/bufferView")
            buffer = self.r.buffer(self.r.int_field(view, "buffer", view_loc), f"{view_loc}/buffer")
            offset = self.r.int_field(view, "byteOffset", view_loc, default=0)
            length = self.r.int_field(view, "byteLength", view_loc, minimum=1)
            if offset + length > len(buffer):
                raise AssetFormatError(
                    "bufferView extends past the end of its buffer",
                    view_loc,
                    "BUFFER_VIEW_OUT_OF_BOUNDS",
                )
            data = bytes(buffer[offset : offset + length])
        elif isinstance(image.get("uri"), str):
            uri = image["uri"]
            if uri.startswith("data:"):
                header, sep, payload = uri.partition(",")
                mime = header[5:].split(";")[0]
                if (
                    not sep
                    or not header.endswith(";base64")
                    or mime not in ("image/png", "image/jpeg")
                ):
                    self.r.note(
                        "IMAGE_FORMAT_UNSUPPORTED",
                        WARNING,
                        "only base64 PNG/JPEG data URIs are supported; image left out",
                        f"{where}/uri",
                    )
                    self.images[index] = None
                    return None
                try:
                    data = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    raise AssetFormatError(
                        "image data URI is not valid base64", f"{where}/uri", "INVALID_URI"
                    ) from None
            else:
                try:
                    data = self.r.external_file(uri, f"{where}/uri")
                except AssetFormatError as exc:
                    if exc.code != "MISSING_RESOURCE":
                        raise
                    self.r.note(
                        "TEXTURE_MISSING",
                        WARNING,
                        f"image file {uri!r} was not found; materials use their factors instead",
                        f"{where}/uri",
                    )
                    self.images[index] = None
                    return None
        else:
            raise AssetFormatError("image needs a uri or a bufferView", where)
        self.r.charge(len(data), where)
        self.images[index] = data
        return data

    def _sampler(self, index: Any, loc: str) -> dict[str, Any]:
        if index is None:
            return {"wrap": ("repeat", "repeat"), "filter": "linear", "mipmaps": True}
        sampler, where = self.r.item("samplers", index, loc)
        wrap = []
        for key in ("wrapS", "wrapT"):
            value = sampler.get(key, 10497)
            if value not in WRAP_FROM_GLTF:
                raise AssetFormatError(f"invalid {key} {value!r}", f"{where}/{key}")
            wrap.append(WRAP_FROM_GLTF[value])
        mag = sampler.get("magFilter")
        minf = sampler.get("minFilter")
        if mag not in (None, NEAREST, LINEAR) or minf not in (
            None,
            NEAREST,
            LINEAR,
            *MIPMAP_FILTERS,
        ):
            raise AssetFormatError("invalid magFilter/minFilter", where)
        nearest = mag == NEAREST
        mipmaps = minf is None or minf in MIPMAP_FILTERS
        min_nearest = minf in (NEAREST, 9984, 9986)
        if minf is not None and min_nearest != nearest:
            self.r.note(
                "SAMPLER_APPROXIMATED",
                INFO,
                "different minification and magnification filters are imported as one filter "
                f"({'nearest' if nearest else 'linear'})",
                where,
            )
        return {
            "wrap": tuple(wrap),
            "filter": "nearest" if nearest else "linear",
            "mipmaps": mipmaps,
        }


def assign(obj: SceneObject, materials: dict[int, tuple[str, Any] | None], grouped: bool) -> None:
    """Give an imported object its materials: ``{group: result}`` from
    :meth:`MaterialImporter.material`, for the whole mesh (not ``grouped``) or per face group."""
    for group, result in sorted(materials.items()):
        if result is None:
            continue
        kind, value = result
        if kind == "appearance":  # an Aether3D object color (the faces without a material)
            obj.appearance.configure(**value)
        elif grouped:
            obj._group_materials[group] = value.id
        else:
            obj._material = value.id


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


class MaterialExporter:
    """Collects glTF materials, textures, images and samplers while a scene is written."""

    def __init__(self, buffers: Any, diagnostics: list[Diagnostic], bake: int | None) -> None:
        self.buffers = buffers
        self.diagnostics = diagnostics
        self.bake = bake
        self.materials: list[dict[str, Any]] = []
        self.textures: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []
        self.samplers: list[dict[str, Any]] = []
        self.extensions: set[str] = set()
        self._materials: dict[Any, int] = {}
        self._images: dict[str, int] = {}
        self._samplers: dict[tuple, int] = {}
        self._textures: dict[tuple[int, int], int] = {}
        self._approximated: set[str] = set()

    def note(self, code: str, severity: str, message: str, location: str | None) -> None:
        self.diagnostics.append(Diagnostic(code, severity, message, location))

    def for_object(self, obj: SceneObject, group: int) -> int | None:
        """The glTF material index for the faces of ``obj`` in face group ``group``."""
        material = obj.material_for_group(group)
        if material is None:
            look = obj.appearance
            if look.is_default:
                return None  # glTF's default material; importing it restores this color
            key = ("appearance", look.color, look.specular, look.shininess)
            if key not in self._materials:
                self._materials[key] = self._appearance(look)
            return self._materials[key]
        key = ("material", material.id)
        if key not in self._materials:
            self._materials[key] = self._material(material)
        return self._materials[key]

    def _appearance(self, look: Any) -> int:
        record = {
            "name": "Object color",
            "pbrMetallicRoughness": {
                "baseColorFactor": [*_linear(look.color), 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": roughness_from_shininess(look.specular, look.shininess),
            },
            "extras": {EXTRAS_KEY: {"appearance": look.to_dict()}},
        }
        if look.specular > 0.0 and "appearance" not in self._approximated:
            self._approximated.add("appearance")
            self.note(
                "SIMPLE_MATERIAL_APPROXIMATED",
                INFO,
                "object colors with a Blinn-Phong highlight are written as metallic-roughness "
                "materials (roughness from shininess); Aether3D restores them exactly from extras",
                None,
            )
        self.materials.append(record)
        return len(self.materials) - 1

    def _material(self, m: Material) -> int:
        values = m._values
        loc = m.location
        pbr: dict[str, Any] = {
            "baseColorFactor": [*_linear(values["base_color"]), values["opacity"]]
        }
        if m.model == "standard":
            pbr["metallicFactor"] = values["metallic"]
            pbr["roughnessFactor"] = values["roughness"]
        else:
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = roughness_from_shininess(
                values["specular"], values["shininess"]
            )
            if m.id not in self._approximated:
                self._approximated.add(m.id)
                self.note(
                    "SIMPLE_MATERIAL_APPROXIMATED",
                    INFO,
                    f"simple material {m.name!r} is written as metallic-roughness (roughness from "
                    "shininess); Aether3D restores it exactly from extras",
                    loc,
                )
        record: dict[str, Any] = {"name": m.name, "pbrMetallicRoughness": pbr}
        emissive = np.asarray(_linear(values["emissive"])) * values["emissive_strength"]
        if np.any(emissive > 0):
            peak = float(emissive.max())
            if peak > 1.0:
                record["emissiveFactor"] = (emissive / peak).tolist()
                record.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": peak
                }
                self.extensions.add("KHR_materials_emissive_strength")
            else:
                record["emissiveFactor"] = emissive.tolist()
        if values["alpha_mode"] == "mask":
            record["alphaMode"] = "MASK"
            record["alphaCutoff"] = values["alpha_cutoff"]
        if values["double_sided"]:
            record["doubleSided"] = True
        if m.model == "standard" and values["specular"] != DEFAULT_SPECULAR:
            level = values["specular"] / DEFAULT_SPECULAR
            ext = (
                {"specularFactor": level} if level <= 1.0 else {"specularColorFactor": [level] * 3}
            )
            record.setdefault("extensions", {})["KHR_materials_specular"] = ext
            self.extensions.add("KHR_materials_specular")
        record["extras"] = {
            EXTRAS_KEY: {"id": m.id, "material": {"model": m.model, **_json(values)}}
        }
        self._textures_of(m, record, pbr)
        self.materials.append(record)
        return len(self.materials) - 1

    def _textures_of(self, m: Material, record: dict, pbr: dict) -> None:
        library = m._library
        bindings = m._textures
        loc = m.location

        def usable(slot: str) -> Any:
            binding = bindings.get(slot)
            if binding is None:
                return None
            texture = library.find_texture(binding.texture)
            if texture is None:
                return None
            if texture.kind == "procedural":
                if self.bake is None:
                    self.note(
                        "PROCEDURAL_TEXTURE_NOT_EXPORTED",
                        WARNING,
                        f"the {slot} texture {texture.name!r} is procedural; glTF has no "
                        "procedural textures, so the slot keeps only its factor",
                        f"{loc}.textures.{slot}",
                    )
                    return None
                if binding.coordinates != "uv":
                    self.note(
                        "TEXTURE_MAPPING_NOT_EXPORTED",
                        WARNING,
                        f"the {slot} texture maps by {binding.coordinates} coordinates, which glTF "
                        "cannot express; left out",
                        f"{loc}.textures.{slot}",
                    )
                    return None
                return binding, texture
            if binding.coordinates != "uv":
                self.note(
                    "TEXTURE_MAPPING_NOT_EXPORTED",
                    WARNING,
                    f"the {slot} texture maps by {binding.coordinates} coordinates, which glTF "
                    "cannot express; left out",
                    f"{loc}.textures.{slot}",
                )
                return None
            return binding, texture

        for slot, key, holder in (
            ("base_color", "baseColorTexture", pbr),
            ("normal", "normalTexture", record),
            ("emissive", "emissiveTexture", record),
            ("occlusion", "occlusionTexture", record),
        ):
            found = usable(slot)
            if found is None:
                continue
            binding, texture = found
            channel = binding.channel_for(slot, texture.kind)
            if slot == "occlusion" and texture.kind == "image" and channel != "r":
                info = self._packed_occlusion(m, binding, texture, channel)
            else:
                info = self._texture_info(m, slot, binding, texture)
            if info is None:
                continue
            if slot == "normal" and m._values["normal_strength"] != 1.0:
                info["scale"] = m._values["normal_strength"]
            if slot == "occlusion" and m._values["occlusion_strength"] != 1.0:
                info["strength"] = m._values["occlusion_strength"]
            holder[key] = info
        if m.model == "standard":
            self._metallic_roughness(m, pbr, usable)
        if "opacity" in bindings:
            self.note(
                "OPACITY_TEXTURE_NOT_EXPORTED",
                WARNING,
                "glTF reads opacity only from the base color texture's alpha; the separate "
                "opacity texture is left out",
                f"{loc}.textures.opacity",
            )

    def _metallic_roughness(self, m: Material, pbr: dict, usable) -> None:
        metal, rough = usable("metallic"), usable("roughness")
        if metal is None and rough is None:
            return
        procedural = [b for b in (metal, rough) if b is not None and b[1].kind == "procedural"]
        if procedural:
            info = self._bake_metallic_roughness(m, metal, rough)
            if info is not None:
                pbr["metallicRoughnessTexture"] = info
            return
        # Images: glTF packs metallic in blue and roughness in green of one texture.
        if metal is not None and rough is not None:
            same = (
                metal[1].id == rough[1].id
                and metal[0].channel_for("metallic", "image") == "b"
                and rough[0].channel_for("roughness", "image") == "g"
                and metal[0].matrix.tolist() == rough[0].matrix.tolist()
            )
            info = (
                self._texture_info(m, "metallic", *metal)
                if same
                else self._packed_metallic_roughness(m, metal, rough)
            )
        else:
            # One image for one of the two: exporting it as the packed texture would also drive
            # the other value, which is harmless only for roughness alone with metallic 0.
            binding, texture = metal if metal is not None else rough
            slot = "metallic" if metal is not None else "roughness"
            exportable = (
                slot == "roughness"
                and binding.channel_for(slot, "image") == "g"
                and m._values["metallic"] == 0.0
            )
            info = (
                self._texture_info(m, slot, binding, texture)
                if exportable
                else self._packed_metallic_roughness(m, metal, rough)
            )
        if info is not None:
            pbr["metallicRoughnessTexture"] = info

    def _packed_occlusion(self, m: Material, binding: Any, texture: Any, channel: str):
        """glTF reads occlusion from red: an image whose occlusion is in another channel is
        copied there (a new PNG)."""
        from ..rendering.png import encode_png

        try:
            pixels = np.asarray(texture.load().image.pixels)
        except MaterialError as exc:
            message = f"texture {texture.name!r} cannot be read ({exc.message}); left out"
            self.note(exc.code, WARNING, message, f"{m.location}.textures.occlusion")
            return None
        if channel == "luminance":
            plane = pixels[..., :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
        else:
            plane = pixels[..., "rgba".index(channel)].astype(np.float64)
        out = np.full(pixels.shape[:2] + (4,), 255, dtype=np.uint8)
        out[..., 0] = np.round(plane).astype(np.uint8)
        out[..., 1] = out[..., 2] = out[..., 0]
        self.note(
            "TEXTURE_CHANNELS_PACKED",
            INFO,
            f"occlusion (channel {channel!r} of {texture.name!r}) copied to the red channel of "
            "a PNG, as glTF reads it",
            f"{m.location}.textures.occlusion",
        )
        image = self._image(encode_png(out), "image/png", f"{texture.name} occlusion")
        sampler = self._sampler(texture.wrap, texture.filter, texture.mipmaps)
        return self._info(image, sampler, binding)

    def _packed_metallic_roughness(self, m: Material, metal: Any, rough: Any) -> dict | None:
        """glTF's metallic-roughness texture built from image or painted textures whose
        channels glTF cannot read as they are: metallic to blue, roughness to green (a value
        without a texture is 1 there, so its factor alone gives it), at the larger size
        (nearest texel). The bindings must share one transform."""
        from ..rendering.png import encode_png

        loc = m.location
        found = [(b, t, slot) for b, t, slot in (
            (*metal, "metallic") if metal is not None else (None, None, None),
            (*rough, "roughness") if rough is not None else (None, None, None),
        ) if b is not None]  # fmt: skip
        if len(found) == 2 and found[0][0].matrix.tolist() != found[1][0].matrix.tolist():
            self.note(
                "TEXTURE_CHANNEL_NOT_EXPORTED",
                WARNING,
                "glTF needs metallic and roughness in one texture; these bindings use "
                "different transforms, so they are left out",
                f"{loc}.textures",
            )
            return None
        planes = {}
        for binding, texture, slot in found:
            try:
                pixels = np.asarray(texture.load().image.pixels)
            except MaterialError as exc:
                self.note(exc.code, WARNING, f"texture {texture.name!r} cannot be read "
                          f"({exc.message}); the {slot} slot keeps only its factor",
                          f"{loc}.textures.{slot}")  # fmt: skip
                return None
            channel = binding.channel_for(slot, "image")
            if channel == "luminance":
                plane = pixels[..., :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
            else:
                plane = pixels[..., "rgba".index(channel)].astype(np.float64)
            planes[slot] = plane
        h = max(p.shape[0] for p in planes.values())
        w = max(p.shape[1] for p in planes.values())
        packed = np.full((h, w, 4), 255, dtype=np.uint8)
        for slot, plane in planes.items():
            rows = np.minimum((np.arange(h) + 0.5) * plane.shape[0] / h, plane.shape[0] - 1)
            cols = np.minimum((np.arange(w) + 0.5) * plane.shape[1] / w, plane.shape[1] - 1)
            resized = plane[rows.astype(int)][:, cols.astype(int)]
            packed[..., 2 if slot == "metallic" else 1] = np.round(resized).astype(np.uint8)
        binding, texture, _ = found[0]
        self.note(
            "TEXTURE_CHANNELS_PACKED",
            INFO,
            f"{' and '.join(slot for *_, slot in found)} packed into one {w}x{h} "
            "metallic-roughness PNG (metallic in blue, roughness in green), as glTF reads them",
            f"{loc}.textures",
        )
        image = self._image(encode_png(packed), "image/png", f"{m.name} metallic-roughness")
        sampler = self._sampler(texture.wrap, texture.filter, texture.mipmaps)
        return self._info(image, sampler, binding)

    # -- images, samplers and texture infos -----------------------------------------

    def _texture_info(self, m: Material, slot: str, binding: Any, texture: Any) -> dict | None:
        if texture.kind == "procedural":
            return self._baked_info(m, slot, binding, texture)
        from ..materials.material import SLOTS

        mode = texture.mode_for(SLOTS[slot].mode)
        if (mode == "srgb") != (SLOTS[slot].mode == "srgb"):
            self.note(
                "TEXTURE_COLOR_SPACE_NOT_EXPORTED",
                WARNING,
                f"texture {texture.name!r} overrides its color space; glTF decodes each slot in "
                "a fixed color space",
                f"{m.location}.textures.{slot}",
            )
        try:
            data = texture._read_bytes()
        except MaterialError as exc:
            self.note(
                exc.code,
                WARNING,
                f"texture {texture.name!r} cannot be read ({exc.message}); the {slot} slot keeps "
                "only its factor",
                f"{m.location}.textures.{slot}",
            )
            return None
        from ..materials.images import sniff_format

        kind = sniff_format(data[:16])
        image = self._image(data, "image/png" if kind == "png" else "image/jpeg", texture.name)
        sampler = self._sampler(texture.wrap, texture.filter, texture.mipmaps)
        return self._info(image, sampler, binding)

    def _info(self, image: int, sampler: int, binding: Any) -> dict:
        key = (image, sampler)
        if key not in self._textures:
            self.textures.append({"sampler": sampler, "source": image})
            self._textures[key] = len(self.textures) - 1
        info: dict[str, Any] = {"index": self._textures[key]}
        transform = transform_to_gltf(binding.offset, binding.rotation, binding.scale)
        if transform is not None:
            info["extensions"] = {"KHR_texture_transform": transform}
            self.extensions.add("KHR_texture_transform")
        return info

    def _image(self, data: bytes, mime: str, name: str) -> int:
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        if digest not in self._images:
            view = self.buffers.add_bytes(data)
            self.images.append({"name": name, "bufferView": view, "mimeType": mime})
            self._images[digest] = len(self.images) - 1
        return self._images[digest]

    def _sampler(self, wrap: tuple[str, str], filter_: str, mipmaps: bool) -> int:
        key = (wrap, filter_, mipmaps)
        if key not in self._samplers:
            linear = filter_ == "linear"
            minf = (9987 if linear else 9984) if mipmaps else (LINEAR if linear else NEAREST)
            self.samplers.append(
                {
                    "magFilter": LINEAR if linear else NEAREST,
                    "minFilter": minf,
                    "wrapS": WRAP_TO_GLTF[wrap[0]],
                    "wrapT": WRAP_TO_GLTF[wrap[1]],
                }
            )
            self._samplers[key] = len(self.samplers) - 1
        return self._samplers[key]

    # -- baking (explicitly requested) -------------------------------------------------

    def _bake_values(self, binding: Any, texture: Any) -> np.ndarray:
        """``(size, size, 4)`` linear RGBA of a procedural texture over the uv square, through
        the binding's transform (row 0 = v near 1)."""
        from ..materials.graph import EvalContext, evaluate

        size = self.bake
        u = (np.arange(size) + 0.5) / size
        v = 1.0 - (np.arange(size) + 0.5) / size
        uu, vv = np.meshgrid(u, v)
        p = np.stack([uu.ravel(), vv.ravel(), np.zeros(uu.size)], axis=1)
        m = binding.matrix
        q = p @ m[:, :3].T + m[:, 3]
        du = np.tile(m[:, 0] / size, (q.shape[0], 1))
        dv = np.tile(-m[:, 1] / size, (q.shape[0], 1))
        rgba = evaluate(texture.graph, EvalContext(q, du, dv, texture._library._image_sampler()))
        return rgba.reshape(size, size, 4)

    def _baked_png(self, rgba: np.ndarray, srgb: bool) -> bytes:
        from ..rendering.color import quantize
        from ..rendering.png import encode_png

        out = np.empty(rgba.shape, dtype=np.uint8)
        out[..., :3] = quantize(linear_to_srgb(rgba[..., :3]) if srgb else rgba[..., :3])
        out[..., 3] = quantize(rgba[..., 3])
        return encode_png(out)

    def _baked_info(self, m: Material, slot: str, binding: Any, texture: Any) -> dict:
        rgba = self._bake_values(binding, texture)
        if slot == "occlusion":  # glTF reads occlusion from red: bake the bound channel there
            channel = binding.channel_for(slot, "procedural")
            if channel == "luminance":
                value = rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
            else:
                value = rgba[..., "rgba".index(channel)]
            rgba = np.stack([value, value, value, np.ones_like(value)], axis=-1)
        if slot == "base_color":
            rgba[..., 3] = 1.0  # procedural alpha is not opacity
        data = self._baked_png(rgba, slot in ("base_color", "emissive"))
        self.note(
            "TEXTURE_BAKED",
            INFO,
            f"procedural texture {texture.name!r} was baked to a {self.bake}x{self.bake} PNG for "
            f"the {slot} slot (exact for UVs in [0, 1]; beyond, the image repeats)",
            f"{m.location}.textures.{slot}",
        )
        image = self._image(data, "image/png", f"{texture.name} (baked)")
        sampler = self._sampler(("repeat", "repeat"), "linear", True)
        return self._info(image, sampler, _Identity())

    def _bake_metallic_roughness(self, m: Material, metal: Any, rough: Any) -> dict | None:
        size = self.bake
        if size is None:
            return None
        if any(b is not None and b[1].kind == "image" for b in (metal, rough)):
            self.note(
                "TEXTURE_CHANNEL_NOT_EXPORTED",
                WARNING,
                "an image and a procedural texture for metallic/roughness cannot share glTF's "
                "packed texture; they are left out",
                f"{m.location}.textures",
            )
            return None
        packed = np.ones((size, size, 4))
        for found, slot, target in ((metal, "metallic", 2), (rough, "roughness", 1)):
            if found is None:
                continue
            binding, texture = found
            rgba = self._bake_values(binding, texture)
            channel = binding.channel_for(slot, "procedural")
            if channel == "luminance":
                packed[..., target] = rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
            else:
                packed[..., target] = rgba[..., "rgba".index(channel)]
            self.note(
                "TEXTURE_BAKED",
                INFO,
                f"procedural texture {texture.name!r} was baked into a {size}x{size} "
                f"metallic-roughness PNG ({slot})",
                f"{m.location}.textures.{slot}",
            )
        packed[..., 0] = 1.0
        image = self._image(
            self._baked_png(packed, False), "image/png", f"{m.name} metallic-roughness (baked)"
        )
        sampler = self._sampler(("repeat", "repeat"), "linear", True)
        return self._info(image, sampler, _Identity())

    def document_parts(self) -> dict[str, Any]:
        parts: dict[str, Any] = {}
        for key, items in (
            ("materials", self.materials),
            ("textures", self.textures),
            ("images", self.images),
            ("samplers", self.samplers),
        ):
            if items:
                parts[key] = items
        return parts


class _Identity:
    offset = (0.0, 0.0, 0.0)
    rotation = 0.0
    scale = (1.0, 1.0, 1.0)


def _json(values: dict[str, Any]) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}


def split_for_tangents(mesh: Any) -> tuple[Any, np.ndarray]:
    """``mesh`` with a vertex per distinct (vertex, MikkTSpace tangent) and those per-vertex
    tangents ``(V, 4)``: glTF's TANGENT is per vertex, while MikkTSpace gives every corner its
    own (they differ where a vertex joins mirrored or separately mapped UV islands)."""
    split, tangents, _ = split_for_tangents_mapped(mesh)
    return split, tangents


def split_for_tangents_mapped(mesh: Any) -> tuple[Any, np.ndarray, np.ndarray | None]:
    """:func:`split_for_tangents` and, when vertices were split, each new vertex's source
    vertex (to carry per-vertex skin weights and morph offsets along)."""
    from ..mesh import Mesh
    from ..uv.mikktspace import mesh_tangents

    corner = mesh_tangents(mesh)  # (T, 3, 4) float32
    tri = mesh.triangles.astype(np.int64).reshape(-1)
    bits = np.ascontiguousarray(corner.reshape(-1, 4)).view(np.uint32)
    key = np.concatenate([tri[:, None].astype(np.uint64), bits.astype(np.uint64)], axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if first.shape[0] == mesh.vertex_count and np.array_equal(
        np.sort(tri[first]), np.arange(mesh.vertex_count)
    ):
        tangents = np.zeros((mesh.vertex_count, 4), dtype=np.float32)
        tangents[tri[first]] = corner.reshape(-1, 4)[first]
        return mesh, tangents, None
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0])
    new_index = rank[inverse]
    src_corner = first[order]
    src_vertex = tri[src_corner]
    split = Mesh(
        mesh.positions[src_vertex],
        new_index.reshape(-1, 3),
        normals=mesh.normals[src_vertex],
        uvs=mesh.uvs[src_vertex],
        groups=mesh.groups,
    )
    return split, corner.reshape(-1, 4)[src_corner].astype(np.float32), src_vertex
