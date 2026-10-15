"""Materials: typed parameters, texture bindings and their schemas.

A :class:`Material` describes how a surface reflects and emits light. It has a
stable ``id``, a ``name`` unique among the scene's materials, a shading
``model`` and parameters:

``"standard"`` (default) -- metallic-roughness, the model of glTF 2.0's
``pbrMetallicRoughness``: a GGX microfacet specular lobe (Smith
height-correlated visibility, Schlick Fresnel) over a Lambertian diffuse
lobe; see ``docs/MATERIALS.md`` for the exact formulas and their limits.

``"simple"`` -- Aether3D 0.5's look: Lambertian diffuse plus an optional
Blinn-Phong highlight. An object's own color (its :class:`~aether3d.Appearance`)
is exactly a simple material, so converting colors to simple materials
changes no pixel.

Parameters (``configure`` validates them together, all or nothing):

* ``base_color`` -- display (sRGB) color; ``opacity`` -- 0..1 (alpha).
* ``alpha_mode`` -- ``"opaque"`` (opacity ignored) or ``"mask"`` (cut out
  where opacity < ``alpha_cutoff``). There is no blended transparency.
* standard: ``metallic`` (0..1), ``roughness`` (0..1), ``specular`` (0..1,
  default 0.5: the reflectance of non-metals, F0 = 0.08 x specular = 4%).
* simple: ``specular`` (highlight strength, default 0) and ``shininess``
  (Blinn-Phong exponent, default 32).
* ``emissive`` (display color, default black) x ``emissive_strength``:
  light the surface emits itself (it does not illuminate other objects).
* ``normal_strength`` -- scale of the normal map's tilt;
  ``occlusion_strength`` -- how much the occlusion map darkens ambient light.
* ``double_sided`` -- ``True`` (default): both sides render; ``False``:
  back faces are not drawn (as glTF's ``doubleSided: false``).

Texture slots (:data:`SLOTS`) bind a texture to a parameter; the texture
multiplies the factor (``base_color x texture``, ``roughness x texture``,
...), except ``normal`` (a tangent-space normal map) and ``occlusion``.
A :class:`TextureBinding` chooses the texture, the channel read by number
slots, the coordinates (mesh ``uv``, ``object`` or ``world`` space with a
projection) and a transform (``offset``, ``scale``, ``rotation``).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import MaterialError
from ..rendering.color import parse_color, srgb_to_linear

if TYPE_CHECKING:
    from .library import MaterialLibrary
    from .textures import Texture

MODELS = ("standard", "simple")
ALPHA_MODES = ("opaque", "mask")
COORDINATES = ("uv", "object", "world")
PROJECTIONS = ("triplanar", "xy", "xz", "yz", "volume")
CHANNELS = ("rgb", "r", "g", "b", "a", "luminance")
DEFAULT_BASE_COLOR = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "color" | "float" | "enum" | "bool"
    default: Any
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    models: tuple[str, ...] = MODELS
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.kind,
            "default": _jsonable(self.default),
        }
        if self.lo is not None:
            data["min"], data["max"] = self.lo, self.hi
        if self.choices:
            data["choices"] = list(self.choices)
        data["models"] = list(self.models)
        data["description"] = self.description
        return data


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("base_color", "color", DEFAULT_BASE_COLOR, description="display (sRGB) color"),
    ParamSpec("opacity", "float", 1.0, 0.0, 1.0, description="alpha; used with alpha_mode mask"),
    ParamSpec("alpha_mode", "enum", "opaque", choices=ALPHA_MODES),
    ParamSpec("alpha_cutoff", "float", 0.5, 0.0, 1.0, description="mask: cut out below this"),
    ParamSpec("metallic", "float", 0.0, 0.0, 1.0, models=("standard",)),
    ParamSpec("roughness", "float", 0.5, 0.0, 1.0, models=("standard",)),
    ParamSpec(
        "specular",
        "float",
        0.5,
        0.0,
        1.0,
        description="standard: dielectric reflectance (F0 = 0.08 x specular); simple: "
        "highlight strength (default 0)",
    ),
    ParamSpec("shininess", "float", 32.0, 1.0, 10000.0, models=("simple",)),
    ParamSpec("emissive", "color", (0.0, 0.0, 0.0), description="display color of emitted light"),
    ParamSpec("emissive_strength", "float", 1.0, 0.0, 1e6),
    ParamSpec("normal_strength", "float", 1.0, -10.0, 10.0, description="normal map tilt scale"),
    ParamSpec("occlusion_strength", "float", 1.0, 0.0, 1.0),
    ParamSpec("double_sided", "bool", True, description="render back faces too"),
)
PARAM_SPECS = {p.name: p for p in PARAMS}
MODEL_DEFAULTS = {"standard": {"specular": 0.5}, "simple": {"specular": 0.0}}


@dataclass(frozen=True)
class SlotSpec:
    name: str
    kind: str  # "color" | "value" | "normal"
    mode: str  # image decoding: "srgb" | "linear" | "normal"
    image_channel: str  # default channel of image textures (glTF packing)
    models: tuple[str, ...] = MODELS
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "color_space": "srgb" if self.mode == "srgb" else "linear",
            "default_image_channel": self.image_channel,
            "models": list(self.models),
            "description": self.description,
        }


SLOTS: dict[str, SlotSpec] = {
    s.name: s
    for s in (
        SlotSpec("base_color", "color", "srgb", "rgb", description="multiplies base_color"),
        SlotSpec("opacity", "value", "linear", "a", description="multiplies opacity"),
        SlotSpec("metallic", "value", "linear", "b", ("standard",), "multiplies metallic"),
        SlotSpec("roughness", "value", "linear", "g", ("standard",), "multiplies roughness"),
        SlotSpec("normal", "normal", "normal", "rgb", description="tangent-space normal map"),
        SlotSpec("emissive", "color", "srgb", "rgb", description="multiplies emissive"),
        SlotSpec("occlusion", "value", "linear", "r", description="ambient occlusion"),
    )
}


def _err(code: str, message: str, location: str | None = None, **kw) -> MaterialError:
    return MaterialError(code, message, location, **kw)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _float(value: Any, lo: float, hi: float, name: str, loc: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)) or not lo <= float(value) <= hi:
        raise _err(
            "INVALID_MATERIAL",
            f"{name} must be a finite number in [{lo:g}, {hi:g}], got {value!r}",
            loc,
        )
    return float(value)


def _vec3(value: Any, name: str, loc: str, lo: float = -1e6, hi: float = 1e6) -> tuple[float, ...]:
    if _is_number(value):
        v = _float(value, lo, hi, name, loc)
        return (v, v, v)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) in (2, 3):
        items = [_float(v, lo, hi, name, loc) for v in value]
        if len(items) == 2:
            items.append(1.0 if name == "scale" else 0.0)
        return tuple(items)
    raise _err("INVALID_MATERIAL", f"{name} must be a number or 2-3 numbers, got {value!r}", loc)


def check_param(name: str, value: Any, location: str) -> Any:
    """Validate one material parameter; returns its normalized value."""
    spec = PARAM_SPECS.get(name)
    if spec is None:
        raise _err(
            "INVALID_MATERIAL",
            f"unknown material parameter {name!r}",
            location,
            hint="parameters: " + ", ".join(PARAM_SPECS),
        )
    loc = f"{location}.{name}"
    if spec.kind == "color":
        try:
            return parse_color(value, loc)
        except Exception as exc:
            raise _err("INVALID_MATERIAL", str(getattr(exc, "message", exc)), loc) from None
    if spec.kind == "float":
        return _float(value, spec.lo, spec.hi, name, loc)
    if spec.kind == "enum":
        if value not in spec.choices:
            raise _err(
                "INVALID_MATERIAL",
                f"{name} must be one of {', '.join(spec.choices)}, got {value!r}",
                loc,
            )
        return value
    if not isinstance(value, bool):
        raise _err("INVALID_MATERIAL", f"{name} must be true or false, got {value!r}", loc)
    return value


def model_params(model: str) -> tuple[str, ...]:
    """The parameters a model uses, in schema order."""
    return tuple(p.name for p in PARAMS if model in p.models)


def default_params(model: str) -> dict[str, Any]:
    values = {p.name: p.default for p in PARAMS if model in p.models}
    values.update(MODEL_DEFAULTS[model])
    return values


@dataclass(frozen=True)
class TextureBinding:
    """How a material slot reads a texture.

    * ``texture`` -- the texture's id.
    * ``channel`` -- for number slots: ``"r"``, ``"g"``, ``"b"``, ``"a"`` or
      ``"luminance"``; ``None`` picks the slot's default (glTF packing for
      images: metallic ``b``, roughness ``g``, occlusion ``r``, opacity ``a``;
      ``r`` -- the value -- for procedural textures). Color and normal slots
      read ``"rgb"``.
    * ``coordinates`` -- ``"uv"`` (the mesh's texture coordinates),
      ``"object"`` or ``"world"`` (positions in object or world space).
    * ``projection`` -- for object/world coordinates: ``"triplanar"`` (three
      planar projections blended by the surface normal, sharpness ``blend``),
      ``"xy"``, ``"xz"``, ``"yz"`` (one plane) or ``"volume"`` (the 3D point,
      for 3D procedural patterns such as noise, Voronoi and wood).
    * ``offset``, ``scale``, ``rotation`` -- the transform applied to the
      coordinates before lookup: ``p' = R(rotation) (scale * p) + offset``
      (rotation in degrees, counterclockwise around z / in the uv plane).
    """

    texture: str
    channel: str | None = None
    coordinates: str = "uv"
    projection: str = "triplanar"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: float = 0.0
    blend: float = 4.0

    FIELDS = ("channel", "coordinates", "projection", "offset", "scale", "rotation", "blend")

    @classmethod
    def build(
        cls, slot: str, texture: str, options: dict[str, Any], location: str
    ) -> TextureBinding:
        unknown = sorted(set(options) - set(cls.FIELDS))
        if unknown:
            raise _err(
                "INVALID_MATERIAL",
                f"unknown texture binding option(s): {', '.join(unknown)}",
                location,
                hint="options: " + ", ".join(cls.FIELDS),
            )
        spec = SLOTS[slot]
        channel = options.get("channel")
        if channel is not None:
            if channel not in CHANNELS:
                raise _err(
                    "INVALID_MATERIAL",
                    f"channel must be one of {', '.join(CHANNELS)}, got {channel!r}",
                    f"{location}.channel",
                )
            if spec.kind == "value" and channel == "rgb":
                raise _err(
                    "INVALID_MATERIAL",
                    f"the {slot} slot reads one channel, not rgb",
                    f"{location}.channel",
                )
            if spec.kind != "value" and channel != "rgb":
                raise _err(
                    "INVALID_MATERIAL",
                    f"the {slot} slot reads rgb, not {channel!r}",
                    f"{location}.channel",
                )
        coordinates = options.get("coordinates", "uv")
        if coordinates not in COORDINATES:
            raise _err(
                "INVALID_MATERIAL",
                f"coordinates must be one of {', '.join(COORDINATES)}, got {coordinates!r}",
                f"{location}.coordinates",
            )
        projection = options.get("projection", "triplanar")
        if projection not in PROJECTIONS:
            raise _err(
                "INVALID_MATERIAL",
                f"projection must be one of {', '.join(PROJECTIONS)}, got {projection!r}",
                f"{location}.projection",
            )
        if coordinates == "uv" and "projection" in options and projection != "triplanar":
            raise _err(
                "INVALID_MATERIAL",
                "projection applies to object or world coordinates, not uv",
                f"{location}.projection",
            )
        if spec.kind == "normal" and coordinates != "uv" and projection == "volume":
            raise _err(
                "INVALID_MATERIAL",
                "a normal map needs a surface frame: use uv coordinates or a planar/triplanar "
                "projection, not volume",
                f"{location}.projection",
            )
        offset = _vec3(options.get("offset", (0.0, 0.0, 0.0)), "offset", f"{location}.offset")
        scale = _vec3(options.get("scale", (1.0, 1.0, 1.0)), "scale", f"{location}.scale")
        rotation = _float(
            options.get("rotation", 0.0), -1e6, 1e6, "rotation", f"{location}.rotation"
        )
        blend = _float(options.get("blend", 4.0), 1.0, 64.0, "blend", f"{location}.blend")
        return cls(texture, channel, coordinates, projection, offset, scale, rotation, blend)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"texture": self.texture}
        default = TextureBinding(self.texture)
        for name in self.FIELDS:
            value = getattr(self, name)
            if value != getattr(default, name):
                data[name] = list(value) if isinstance(value, tuple) else value
        if self.coordinates == "uv":
            data.pop("projection", None)
            data.pop("blend", None)
        return data

    def channel_for(self, slot: str, texture_kind: str) -> str:
        """The channel actually read: the explicit one, or the slot's default."""
        spec = SLOTS[slot]
        if spec.kind != "value":
            return "rgb"
        if self.channel is not None:
            return self.channel
        return spec.image_channel if texture_kind == "image" else "r"

    @property
    def matrix(self) -> np.ndarray:
        """``3 x 4`` affine map of the binding transform (``p' = M [p, 1]``)."""
        c, s = math.cos(math.radians(self.rotation)), math.sin(math.radians(self.rotation))
        sx, sy, sz = self.scale
        m = np.zeros((3, 4))
        m[0, 0], m[0, 1] = c * sx, -s * sy
        m[1, 0], m[1, 1] = s * sx, c * sy
        m[2, 2] = sz
        m[:, 3] = self.offset
        return m


class Material:
    """A material of a scene's library (``scene.materials``); see the module docstring.

    Create materials with :meth:`aether3d.Scene.create_material`; assign them
    with ``obj.material = material`` or :meth:`aether3d.Scene.assign_material`.
    """

    __slots__ = ("_library", "_home", "_id", "_name", "_model", "_values", "_textures")

    def __init__(
        self,
        library: MaterialLibrary,
        material_id: str,
        name: str,
        model: str,
        values: dict[str, Any],
        textures: dict[str, TextureBinding] | None = None,
    ) -> None:
        self._library: MaterialLibrary | None = library
        self._home = library  # the library whose scene history records this material
        self._id = material_id
        self._name = name
        self._model = model
        self._values = values
        self._textures: dict[str, TextureBinding] = dict(textures or {})

    # -- identity -------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._require_library()._rename_material(self, value)

    @property
    def library(self) -> MaterialLibrary | None:
        return self._library

    @property
    def location(self) -> str:
        return f"materials[{self._name}]"

    def _require_library(self) -> MaterialLibrary:
        if self._library is None:
            raise _err("MATERIAL_REMOVED", f"material {self._name!r} was removed from its scene")
        return self._library

    # -- parameters -----------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self.configure(model=value)

    @property
    def params(self) -> dict[str, Any]:
        """The parameters of the current model (a copy)."""
        return {k: _jsonable(v) for k, v in self._values.items()}

    def __getattr__(self, name: str) -> Any:
        if name in PARAM_SPECS:
            values = object.__getattribute__(self, "_values")
            if name not in values:
                raise _err(
                    "PARAMETER_NOT_USED",
                    f"the {self._model} model has no {name!r} parameter",
                    f"{self.location}.{name}",
                )
            return values[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in PARAM_SPECS:
            self.configure(**{name: value})
            return
        object.__setattr__(self, name, value)

    def configure(self, **changes: Any) -> Material:
        """Change parameters (and ``model``) together: validated all or nothing.

        Changing ``model`` keeps the shared parameters (color, opacity,
        emission, ...) and resets the model's own ones to their defaults;
        texture slots the new model does not use are removed.
        """
        self._require_library()
        loc = self.location
        model = changes.pop("model", self._model)
        if model not in MODELS:
            raise _err(
                "INVALID_MATERIAL",
                f"model must be one of {', '.join(MODELS)}, got {model!r}",
                f"{loc}.model",
            )
        values = dict(self._values)
        if model != self._model:
            # Shared parameters carry over; model-specific ones (specular means something
            # else in each model) start from the new model's defaults.
            fresh = default_params(model)
            shared = {
                k: v for k, v in values.items() if k in fresh and k not in MODEL_DEFAULTS[model]
            }
            values = {**fresh, **shared}
        for name, value in changes.items():
            normalized = check_param(name, value, loc)
            if name not in values:
                raise _err(
                    "PARAMETER_NOT_USED",
                    f"the {model} model has no {name!r} parameter",
                    f"{loc}.{name}",
                    hint=f"{model} parameters: {', '.join(model_params(model))}",
                )
            values[name] = normalized
        textures = {s: b for s, b in self._textures.items() if model in SLOTS[s].models}
        with self._edit(f"Set {', '.join(changes) or 'model'} of material {self._name!r}"):
            self._model = model
            self._values = values
            self._textures = textures
            self._touch()
        return self

    def _touch(self) -> None:
        if self._library is not None:
            self._library._changed()

    def _edit(self, label: str) -> Any:
        """An operation changing this material, recorded in its scene's history."""
        from ..history import MATERIAL

        return self._home._scene._history._op(label, MATERIAL, self)

    # -- textures -------------------------------------------------------------------

    @property
    def textures(self) -> dict[str, TextureBinding]:
        """Texture bindings by slot (a copy)."""
        return dict(self._textures)

    def texture(self, slot: str) -> Texture | None:
        """The texture bound to ``slot``, or ``None``."""
        binding = self._textures.get(_check_slot(slot, self._model, self.location))
        return None if binding is None else self._require_library().texture(binding.texture)

    def set_texture(self, slot: str, texture: Any, **options: Any) -> TextureBinding:
        """Bind a texture (object, id or name; or an image file path) to ``slot``.

        A path to a PNG or JPEG file creates an image texture in the scene
        (or reuses one already loaded from that file). ``options`` are the
        :class:`TextureBinding` fields (``channel``, ``coordinates``,
        ``projection``, ``offset``, ``scale``, ``rotation``, ``blend``).
        """
        library = self._require_library()
        slot = _check_slot(slot, self._model, self.location)
        with self._edit(f"Set {slot} texture of material {self._name!r}"):
            tex = library._texture_or_file(texture, f"{self.location}.textures.{slot}")
            binding = TextureBinding.build(
                slot, tex.id, options, f"{self.location}.textures.{slot}"
            )
            library._check_binding(self, slot, binding, tex)
            self._textures[slot] = binding
            self._touch()
        return binding

    def set_procedural_texture(
        self, slot: str, graph: Any, *, name: str | None = None, **params: Any
    ) -> TextureBinding:
        """Create a procedural texture and bind it to ``slot``.

        ``graph`` is a node type (``"checker"``, ``"bricks"``, ... with
        ``params`` as its parameters) or a whole graph / node as a dict.
        Binding options (``coordinates``, ``scale``, ...) can be passed in
        ``params`` too: names that are not node parameters are binding options.
        """
        from .graph import NODE_TYPES

        library = self._require_library()
        slot = _check_slot(slot, self._model, self.location)
        binding_options = {k: params.pop(k) for k in list(params) if k in TextureBinding.FIELDS}
        if isinstance(graph, str):
            spec = NODE_TYPES.get(graph)
            if spec is not None and "scale" in binding_options and spec.pattern:
                params["scale"] = binding_options.pop("scale")  # a pattern's own scale
            data = {"type": graph, **params}
        else:
            if params:
                raise _err(
                    "INVALID_MATERIAL",
                    "pass node parameters inside the graph, or give a node type name",
                    f"{self.location}.textures.{slot}",
                )
            data = graph
        with library._transaction(f"Set {slot} texture of material {self._name!r}"):
            tex = library.add_procedural_texture(
                name or library._unique_texture_name(f"{self._name} {slot}"), data
            )
            return self.set_texture(slot, tex, **binding_options)

    def clear_texture(self, slot: str) -> None:
        """Remove the texture of ``slot`` (the texture itself stays in the scene)."""
        slot = _check_slot(slot, self._model, self.location)
        if slot in self._textures:
            with self._edit(f"Clear {slot} texture of material {self._name!r}"):
                del self._textures[slot]
                self._touch()

    def set_texture_options(self, slot: str, **options: Any) -> TextureBinding:
        """Change binding options of ``slot`` (``scale``, ``coordinates``, ...)."""
        slot = _check_slot(slot, self._model, self.location)
        binding = self._textures.get(slot)
        if binding is None:
            raise _err(
                "NO_TEXTURE", f"slot {slot!r} has no texture", f"{self.location}.textures.{slot}"
            )
        current = {k: v for k, v in binding.to_dict().items() if k != "texture"}
        new = TextureBinding.build(
            slot, binding.texture, {**current, **options}, f"{self.location}.textures.{slot}"
        )
        library = self._require_library()
        library._check_binding(self, slot, new, library.texture(new.texture))
        with self._edit(f"Set {slot} texture options of material {self._name!r}"):
            self._textures[slot] = new
            self._touch()
        return new

    # -- inspection -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self._id, "name": self._name, "model": self._model}
        for name, value in self._values.items():
            data[name] = _jsonable(value)
        data["textures"] = {slot: b.to_dict() for slot, b in self._textures.items()}
        return data

    def inspect(self) -> dict[str, Any]:
        """Parameters, texture bindings (with each texture's status), users and problems."""
        return self._require_library().inspect_material(self)

    def linear(self, name: str) -> np.ndarray:
        """A color parameter in linear light."""
        return srgb_to_linear(np.asarray(self._values[name], dtype=np.float64))

    def copy_values(self) -> tuple[str, dict[str, Any], dict[str, TextureBinding]]:
        return self._model, dict(self._values), dict(self._textures)

    def __repr__(self) -> str:
        return f"Material(name={self._name!r}, model={self._model!r})"


def _check_slot(slot: Any, model: str, location: str) -> str:
    if slot not in SLOTS:
        raise _err(
            "INVALID_MATERIAL",
            f"unknown texture slot {slot!r}",
            f"{location}.textures",
            hint="slots: " + ", ".join(SLOTS),
        )
    if model not in SLOTS[slot].models:
        raise _err(
            "PARAMETER_NOT_USED",
            f"the {model} model has no {slot!r} slot",
            f"{location}.textures.{slot}",
            hint=f"{model} slots: " + ", ".join(s for s, v in SLOTS.items() if model in v.models),
        )
    return slot


def binding_from_dict(slot: str, raw: Any, location: str) -> TextureBinding:
    if not isinstance(raw, dict) or not isinstance(raw.get("texture"), str):
        raise _err(
            "INVALID_MATERIAL", 'a texture binding must be an object with a "texture" id', location
        )
    options = {k: v for k, v in raw.items() if k != "texture"}
    return TextureBinding.build(slot, raw["texture"], options, location)


def material_schema() -> dict[str, Any]:
    """JSON-compatible schema of material parameters, slots and binding options."""
    return {
        "models": list(MODELS),
        "parameters": [p.to_dict() for p in PARAMS],
        "model_defaults": {m: dict(d) for m, d in MODEL_DEFAULTS.items()},
        "slots": [s.to_dict() for s in SLOTS.values()],
        "binding": {
            "channel": list(CHANNELS),
            "coordinates": list(COORDINATES),
            "projection": list(PROJECTIONS),
            "offset": "number or [x, y(, z)]",
            "scale": "number or [x, y(, z)]",
            "rotation": "degrees",
            "blend": "triplanar sharpness, 1..64 (default 4)",
        },
    }


def new_material_id() -> str:
    return str(uuid.uuid4())


__all__ = [
    "ALPHA_MODES",
    "COORDINATES",
    "MODELS",
    "PARAMS",
    "PROJECTIONS",
    "SLOTS",
    "Material",
    "TextureBinding",
    "binding_from_dict",
    "material_schema",
]
