"""Scene graph: scenes, scene objects, and their hierarchy."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from . import primitives
from .errors import (
    DetachedObjectError,
    HierarchyError,
    NameConflictError,
    ObjectNotFoundError,
    ValidationError,
)
from .history import (
    ACTIVE_CAMERA,
    ALIVE,
    ANNOTATIONS,
    APPEARANCE,
    CAMERA,
    CHILDREN,
    DATA,
    LIGHT,
    NAME,
    PARENT,
    RENDER,
    ROOTS,
    SCENE_NAME,
    TRANSFORM,
    VISIBLE,
    WORLD,
    Aspect,
    History,
)
from .math3d import Quat, Vec3, decompose_matrix, look_rotation_matrix, matrix3_to_quat, vec3
from .mesh import Mesh
from .rendering.camera import Camera
from .rendering.light import Light
from .rendering.settings import Appearance, RenderSettings, World
from .transform import Transform

if TYPE_CHECKING:
    from .editing import EditSession
    from .gltf import ExportResult, ImportLimits
    from .inspection import SceneGeometryReport
    from .materials import Material, MaterialLibrary
    from .plans import PlanResult
    from .procedural import ProceduralGeometry, ProceduralSystem
    from .rendering import RenderResult
    from .rendering.framing import Framing
    from .sculpt.session import SculptSession
    from .spatial import (
        CameraRay,
        Containment,
        LineOfSight,
        ObjectBounds,
        ObjectDistance,
        Overlap,
        RegionMatch,
        Relation,
        ScreenPoint,
        SurfaceHit,
        ViewMatch,
        Visibility,
    )

ObjectRef = Union["SceneObject", str]
"""A scene object, or its identifier."""


def new_object_id() -> str:
    """Return a new globally unique object identifier (a UUID4 string)."""
    return str(uuid.uuid4())


def new_object_ids(count: int) -> list[str]:
    """``count`` new object identifiers, as :func:`new_object_id` makes them -- random
    version-4 UUIDs, ``str(uuid.uuid4())``'s format -- from one read of the system's random
    source (instead of one per identifier)."""
    if count <= 0:
        return []
    raw = np.frombuffer(os.urandom(16 * count), dtype=np.uint8).reshape(count, 16).copy()
    raw[:, 6] = (raw[:, 6] & 0x0F) | 0x40  # version 4
    raw[:, 8] = (raw[:, 8] & 0x3F) | 0x80  # the RFC 4122 variant
    h = raw.tobytes().hex()
    return [
        f"{h[i : i + 8]}-{h[i + 8 : i + 12]}-{h[i + 12 : i + 16]}-{h[i + 16 : i + 20]}-"
        f"{h[i + 20 : i + 32]}"
        for i in range(0, 32 * count, 32)
    ]


def _check_name(name: Any, what: str = "name") -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{what} must be a non-empty string, got {name!r}")
    return name


def _owned(value: Any, holder: Any, aspect: Aspect) -> Any:
    """``value`` (a camera, light, appearance, ...) with its owner set, so that in-place
    changes are recorded in the holder's history."""
    if value is not None and getattr(value, "_owner", None) is None:
        value._owner = (holder, aspect)
    return value


def _merge_source(source: Any, allow_lossy: bool) -> Scene:
    if isinstance(source, Scene):
        return source
    if not isinstance(source, (str, os.PathLike)):
        raise ValidationError(
            f"merge needs a Scene or the path of a .aether, .glb or .gltf file, got {source!r}"
        )
    path = os.fspath(source)
    if path.lower().endswith((".glb", ".gltf")):
        from .gltf import import_gltf

        return import_gltf(path, allow_lossy=allow_lossy).scene
    return Scene.load(path)


MAX_TAGS = 64
MAX_TAG_LENGTH = 64
MAX_METADATA_BYTES = 16 * 1024


def _check_tags(value: Any) -> tuple[str, ...]:
    """Validated tags as a sorted tuple of unique strings."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValidationError(f"tags must be a list of strings, got {value!r}")
    tags = set()
    for tag in value:
        if not isinstance(tag, str) or not tag or tag != tag.strip() or len(tag) > MAX_TAG_LENGTH:
            raise ValidationError(
                f"a tag must be a string of 1 to {MAX_TAG_LENGTH} characters without leading "
                f"or trailing whitespace, got {tag!r}"
            )
        tags.add(tag)
    if len(tags) > MAX_TAGS:
        raise ValidationError(f"an object has at most {MAX_TAGS} tags, got {len(tags)}")
    return tuple(sorted(tags))


def _check_json(value: Any, location: str, depth: int = 0) -> None:
    if depth > 16:
        raise ValidationError(f"{location}: metadata is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValidationError(f"{location}: metadata numbers must be finite")
        return
    if isinstance(value, (list, tuple)):
        for k, item in enumerate(value):
            _check_json(item, f"{location}[{k}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValidationError(f"{location}: metadata keys must be non-empty strings")
            _check_json(item, f"{location}.{key}", depth + 1)
        return
    raise ValidationError(f"{location}: {type(value).__name__} is not a JSON value")


def _check_metadata(value: Any) -> str:
    """Validated metadata as canonical JSON text (sorted keys, no spaces)."""
    if not isinstance(value, dict):
        raise ValidationError(f"metadata must be a dict, got {type(value).__name__}")
    _check_json(value, "metadata")
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(text.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValidationError(f"metadata is limited to {MAX_METADATA_BYTES} bytes of JSON")
    return text


def _base_name(name: str) -> str:
    """``"Cube"`` for ``"Cube.003"`` (the name duplicates are numbered from)."""
    stem, dot, digits = name.rpartition(".")
    return stem if dot and stem and len(digits) == 3 and digits.isdigit() else name


_DEFAULT_APPEARANCE = Appearance()
"""Every object starts with a copy of this appearance (cloned: it is valid already)."""


class NewObject:
    """One object for :meth:`Scene._create_many`, the bulk creation path (scene plans).

    ``parent`` is a scene object, the index of an earlier record of the same batch, or
    ``None``. ``position`` and ``scale`` are 3-tuples of finite floats and ``rotation`` a
    unit quaternion (``math3d.quat``): values the caller has validated. Everything else
    is checked by ``_create_many`` before the scene changes. An object has at most one of
    ``mesh``, ``camera`` and ``light`` (value objects, copied for each object).
    """

    __slots__ = (
        "name",
        "parent",
        "mesh",
        "position",
        "rotation",
        "scale",
        "object_id",
        "visible",
        "appearance",
        "material",
        "group_materials",
        "tags",
        "metadata",
        "camera",
        "light",
    )

    def __init__(
        self,
        name: str,
        *,
        parent: SceneObject | int | None = None,
        mesh: Mesh | None = None,
        camera: Camera | None = None,
        light: Light | None = None,
        position: Vec3 = (0.0, 0.0, 0.0),
        rotation: Quat = (0.0, 0.0, 0.0, 1.0),
        scale: Vec3 = (1.0, 1.0, 1.0),
        object_id: str | None = None,
        visible: bool = True,
        appearance: Appearance | None = None,
        material: str | None = None,
        group_materials: dict[int, str] | None = None,
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self.mesh = mesh
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.object_id = object_id
        self.visible = visible
        self.appearance = appearance
        self.material = material
        self.group_materials = group_materials
        self.tags = tags
        self.metadata = metadata
        self.camera = camera
        self.light = light


def _finite3(value: Any, what: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(type(v) is float and math.isfinite(v) for v in value)
    ):
        raise ValidationError(f"{what} must be a tuple of 3 finite floats, got {value!r}")


def _valid_transforms(records: list[NewObject]) -> bool:
    """Whether every record's position, scale and rotation pass ``_create_many``'s checks,
    tested for all records at once with the same tests (a tuple of Python floats, finite;
    a unit quaternion with its squares summed in the same order). When this says no, each
    record is checked on its own, so the first error is reported as it always was."""
    positions = [r.position for r in records]
    scales = [r.scale for r in records]
    rotations = [r.rotation for r in records]
    for items, size in ((positions, 3), (scales, 3), (rotations, 4)):
        if not all(type(t) is tuple and len(t) == size for t in items):
            return False
    flat = list(chain.from_iterable(positions + scales + rotations))
    if not all(type(v) is float for v in flat):
        return False
    values = np.array(flat, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return False
    q = values[6 * len(records) :].reshape(-1, 4)
    norm = ((q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1]) + q[:, 2] * q[:, 2]) + q[:, 3] * q[:, 3]
    return bool(np.all(np.abs(norm - 1.0) <= 1e-9))


class SceneObject:
    """A node in a scene's hierarchy.

    Every object has a stable unique :attr:`id`, a :attr:`name` that is unique
    within its scene, a local :attr:`transform` relative to its parent, and at
    most one kind of data: a :attr:`mesh` (static or procedural), a
    :attr:`camera` or a :attr:`light`. Objects without data are useful as
    groups/pivots. Mesh objects have a :attr:`material` (from the scene's
    material library), or else render with their own :attr:`appearance`
    (color); faces can use other materials per face group
    (:attr:`group_materials`). Every object has a :attr:`visible` flag.

    Objects are created through a :class:`Scene` (``scene.add_object()``,
    ``scene.add_cube()``, ...), never directly.
    """

    __slots__ = (
        "_scene",
        "_home",
        "_id",
        "_name",
        "_transform",
        "_mesh",
        "_parent",
        "_children",
        "_geometry",
        "_camera",
        "_light",
        "_appearance",
        "_visible",
        "_material",
        "_group_materials",
        "_tags",
        "_metadata",
    )

    def __init__(
        self, scene: Scene, object_id: str, name: str, transform: Transform, mesh: Mesh | None
    ) -> None:
        self._scene: Scene | None = scene
        self._home = scene  # the scene whose history records this object, even once removed
        self._id = object_id
        self._name = name
        transform._owner = self
        self._transform = transform
        self._mesh = mesh
        self._geometry: str | None = None  # id of a procedural geometry node, if any
        self._camera: Camera | None = None
        self._light: Light | None = None
        self._appearance = _DEFAULT_APPEARANCE._clone()
        self._appearance._owner = (self, APPEARANCE)
        self._visible = True
        self._material: str | None = None  # id of a material in scene.materials
        self._group_materials: dict[int, str] = {}  # face group -> material id
        self._tags: tuple[str, ...] = ()  # sorted, unique
        self._metadata = "{}"  # canonical JSON text (immutable; see `metadata`)
        self._parent: SceneObject | None = None
        self._children: list[SceneObject] = []

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> str:
        """Stable unique identifier; preserved across save/load."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable name, unique within the scene."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._require_scene()._rename(self, value)

    @property
    def scene(self) -> Scene | None:
        """The owning scene, or ``None`` once the object has been removed."""
        return self._scene

    # -- components --------------------------------------------------------

    @property
    def mesh(self) -> Mesh | None:
        """The object's geometry as a triangle :class:`Mesh`.

        For procedural objects this is the *evaluated* result of their geometry
        node (cached; regenerated automatically after any change), so exports
        and inspection never see an outdated or unevaluated base mesh.
        """
        if self._geometry is not None:
            return self._require_scene().procedural.mesh(self._geometry)
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh | None) -> None:
        if self._geometry is not None:
            from .procedural.params import ProceduralError

            raise ProceduralError(
                "PROCEDURAL_OBJECT",
                f"object {self._name!r} is procedural; its mesh is generated by its geometry",
                hint="call obj.bake() to turn it into a static mesh, or obj.set_geometry(None)",
            )
        if value is not None and not isinstance(value, Mesh):
            raise ValidationError(f"mesh must be a Mesh or None, got {type(value).__name__}")
        if value is not None and (self._camera is not None or self._light is not None):
            raise ValidationError(
                f"object {self._name!r} is a {self.kind}; it cannot also have a mesh"
            )
        with self._edit(DATA, f"Set mesh of {self._name!r}"):
            self._mesh = value

    # -- cameras, lights and appearance -------------------------------------

    @property
    def kind(self) -> str:
        """``"mesh"``, ``"procedural"``, ``"camera"``, ``"light"`` or ``"empty"``."""
        if self._camera is not None:
            return "camera"
        if self._light is not None:
            return "light"
        if self._geometry is not None:
            return "procedural"
        return "mesh" if self._mesh is not None else "empty"

    @property
    def camera(self) -> Camera | None:
        """The :class:`~aether3d.rendering.Camera` of a camera object (``None`` otherwise).

        Edit it in place (``obj.camera.fov = 35``); create cameras with
        :meth:`Scene.add_camera`.
        """
        return _owned(self._camera, self, CAMERA)

    @property
    def light(self) -> Light | None:
        """The :class:`~aether3d.rendering.Light` of a light object (``None`` otherwise)."""
        return _owned(self._light, self, LIGHT)

    @property
    def appearance(self) -> Appearance:
        """How the object's mesh renders (color, highlight); edit it in place."""
        return _owned(self._appearance, self, APPEARANCE)

    @appearance.setter
    def appearance(self, value: Appearance) -> None:
        if not isinstance(value, Appearance):
            raise ValidationError(f"appearance must be an Appearance, got {type(value).__name__}")
        new = value.copy()
        with self._edit(APPEARANCE, f"Set appearance of {self._name!r}"):
            self._appearance._owner = None
            new._owner = (self, APPEARANCE)
            self._appearance = new

    @property
    def color(self) -> tuple[float, float, float]:
        """Display (sRGB) base color; accepts ``"#rrggbb"`` or three numbers in ``[0, 1]``.

        This is the object's own color (its :attr:`appearance`). While a
        :attr:`material` is assigned the material decides the look, so
        setting the color then raises ``MaterialError("OBJECT_HAS_MATERIAL")``.
        """
        return self._appearance.color

    @color.setter
    def color(self, value: Any) -> None:
        self._check_no_material("color")
        _owned(self._appearance, self, APPEARANCE).color = value

    def _check_no_material(self, what: str) -> None:
        if self._material is not None:
            from .errors import MaterialError

            material = self.material
            raise MaterialError(
                "OBJECT_HAS_MATERIAL",
                f"object {self._name!r} uses material {material.name!r}; its {what} comes from "
                "the material",
                f"objects[{self._name}].{what}",
                hint=f"change the material (scene.materials[{material.name!r}]) or remove it "
                "(obj.material = None) to use the object's own color",
            )

    # -- materials ----------------------------------------------------------

    @property
    def material(self) -> Material | None:
        """The object's material (``None``: it renders with its own color).

        Assign a :class:`~aether3d.materials.Material` of the same scene, or its
        id or name; ``None`` removes it.
        """
        if self._material is None:
            return None
        return self._require_scene().materials.material(self._material)

    @material.setter
    def material(self, value: Any) -> None:
        self._require_scene().materials.assign(self, value)

    @property
    def group_materials(self) -> dict[int, Material]:
        """Materials of individual face groups (``{group: material}``, a copy).

        Faces whose group has no entry use :attr:`material` (or the object's
        color). Change with ``scene.assign_material(obj, material, group=k)``.
        """
        library = self._require_scene().materials
        return {g: library.material(m) for g, m in sorted(self._group_materials.items())}

    def material_for_group(self, group: int) -> Material | None:
        """The material faces of ``group`` render with (``None``: the object's color)."""
        ref = self._group_materials.get(group, self._material)
        return None if ref is None else self._require_scene().materials.material(ref)

    @property
    def visible(self) -> bool:
        """Whether the object renders (lights: whether they shine). Not inherited by children."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValidationError(f"visible must be True or False, got {value!r}")
        with self._edit(VISIBLE, f"Set visible of {self._name!r}"):
            self._visible = value

    # -- tags and metadata --------------------------------------------------

    @property
    def tags(self) -> tuple[str, ...]:
        """Labels for finding objects (``scene.find_objects(tags=...)``), sorted and unique.

        Assign any iterable of tag strings; :meth:`add_tags` and :meth:`remove_tags`
        change some. Tags are case-sensitive; each is 1 to 64 characters without
        leading or trailing whitespace, and an object has at most 64.
        """
        return self._tags

    @tags.setter
    def tags(self, value: Iterable[str]) -> None:
        if isinstance(value, str):
            raise ValidationError("tags must be an iterable of strings, not a single string")
        tags = _check_tags(value)
        with self._edit(ANNOTATIONS, f"Set tags of {self._name!r}"):
            self._tags = tags

    def add_tags(self, *tags: str) -> None:
        """Add tags (one undo step); tags it already has are kept once."""
        new = _check_tags((*self._tags, *tags))
        with self._edit(ANNOTATIONS, f"Tag {self._name!r} {', '.join(map(repr, tags))}"):
            self._tags = new

    def remove_tags(self, *tags: str) -> None:
        """Remove tags (one undo step); tags it does not have are ignored."""
        drop = set(_check_tags(tags))
        with self._edit(ANNOTATIONS, f"Untag {self._name!r} {', '.join(map(repr, tags))}"):
            self._tags = tuple(t for t in self._tags if t not in drop)

    def has_tag(self, tag: str) -> bool:
        return tag in self._tags

    @property
    def metadata(self) -> dict[str, Any]:
        """Free-form JSON data about the object (a copy: change it with :meth:`set_metadata`
        or by assigning a whole dict).

        Keys are non-empty strings; values are JSON values (``None``, booleans,
        finite numbers, strings, lists and objects), at most 16 KiB in all.
        """
        return json.loads(self._metadata)

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        text = _check_metadata(value)
        with self._edit(ANNOTATIONS, f"Set metadata of {self._name!r}"):
            self._metadata = text

    def set_metadata(self, key: str, value: Any) -> None:
        """Set one metadata entry (one undo step)."""
        text = _check_metadata({**json.loads(self._metadata), key: value})
        with self._edit(ANNOTATIONS, f"Set metadata {key!r} of {self._name!r}"):
            self._metadata = text

    def remove_metadata(self, key: str) -> None:
        """Remove one metadata entry if present (one undo step)."""
        data = json.loads(self._metadata)
        if key in data:
            del data[key]
            text = _check_metadata(data)
            with self._edit(ANNOTATIONS, f"Remove metadata {key!r} of {self._name!r}"):
                self._metadata = text

    # -- procedural geometry ------------------------------------------------

    @property
    def geometry(self) -> ProceduralGeometry | None:
        """The procedural geometry node this object uses, or ``None`` for a static mesh."""
        if self._geometry is None:
            return None
        return self._require_scene().procedural.geometry(self._geometry)

    @property
    def is_procedural(self) -> bool:
        return self._geometry is not None

    def set_geometry(self, geometry: ProceduralGeometry | str | None) -> None:
        """Use a procedural geometry node (by object, id or name); ``None`` detaches.

        Detaching leaves the object without a mesh; use :meth:`bake` to keep the
        evaluated geometry as a static mesh instead.
        """
        self._require_scene().procedural.assign(self, geometry)

    def bake(self) -> Mesh | None:
        """Replace the procedural geometry by its evaluated mesh (a static copy).

        The geometry node itself is kept (other objects may use it). Returns the
        baked mesh. Static objects are left unchanged.
        """
        scene = self._require_scene()
        if self._geometry is None:
            return self._mesh
        mesh = scene.procedural.mesh(self._geometry)  # evaluate first: failures change nothing
        with scene._history._op(f"Bake {self._name!r}"), scene.procedural.batch():
            scene.procedural.assign(self, None)
            self._mesh = mesh
        return mesh

    @property
    def transform(self) -> Transform:
        """Local transform, relative to the parent (or the scene root)."""
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        if not isinstance(value, Transform):
            raise ValidationError(f"transform must be a Transform, got {type(value).__name__}")
        new = value.copy()
        with self._edit(TRANSFORM, f"Set transform of {self._name!r}"):
            self._transform._owner = None
            new._owner = self
            self._transform = new

    def _place(self, transform: Transform) -> None:
        """Copy ``transform``'s values into this object's own transform (same instance)."""
        t = self._transform
        t._position, t._rotation, t._scale = (
            transform._position,
            transform._rotation,
            transform._scale,
        )

    # Local-space shortcuts, so ``obj.position = (0, 0, 1)`` just works.

    @property
    def position(self) -> Vec3:
        return self._transform.position

    @position.setter
    def position(self, value: Iterable[float]) -> None:
        self._transform.position = value

    @property
    def rotation(self) -> Quat:
        return self._transform.rotation

    @rotation.setter
    def rotation(self, value: Iterable[float]) -> None:
        self._transform.rotation = value

    @property
    def rotation_euler(self) -> Vec3:
        return self._transform.rotation_euler

    @rotation_euler.setter
    def rotation_euler(self, value: Iterable[float]) -> None:
        self._transform.rotation_euler = value

    @property
    def scale(self) -> Vec3:
        return self._transform.scale

    @scale.setter
    def scale(self, value: float | Iterable[float]) -> None:
        self._transform.scale = value

    # -- hierarchy ---------------------------------------------------------

    @property
    def parent(self) -> SceneObject | None:
        return self._parent

    @property
    def children(self) -> tuple[SceneObject, ...]:
        """Direct children, in order."""
        return tuple(self._children)

    def iter_descendants(self) -> Iterator[SceneObject]:
        """All descendants in depth-first pre-order (excluding this object)."""
        for child in self._children:
            yield child
            yield from child.iter_descendants()

    def iter_ancestors(self) -> Iterator[SceneObject]:
        """Parent, grandparent, ... up to the root."""
        node = self._parent
        while node is not None:
            yield node
            node = node._parent

    def set_parent(self, parent: ObjectRef | None, *, keep_world_transform: bool = False) -> None:
        """Re-parent this object (``None`` makes it a root object).

        By default the local transform is kept, so the object moves with its
        new parent. With ``keep_world_transform=True`` the local transform is
        recomputed so the object stays where it is in world space.
        """
        self._require_scene()._reparent(self, parent, keep_world_transform)

    # -- spatial queries ---------------------------------------------------

    @property
    def local_matrix(self) -> np.ndarray:
        """4x4 local-to-parent matrix."""
        return self._transform.matrix

    @property
    def world_matrix(self) -> np.ndarray:
        """4x4 local-to-world matrix (``parent.world_matrix @ local_matrix``)."""
        matrix = self._transform.matrix
        for ancestor in self.iter_ancestors():
            matrix = ancestor._transform.matrix @ matrix
        return matrix

    @property
    def world_position(self) -> Vec3:
        x, y, z = self.world_matrix[:3, 3].tolist()
        return (x, y, z)

    @property
    def world_rotation(self) -> Quat:
        """World-space rotation (see ``math3d.decompose_matrix`` for caveats)."""
        return decompose_matrix(self.world_matrix)[1]

    @property
    def world_scale(self) -> Vec3:
        """World-space scale (see ``math3d.decompose_matrix`` for caveats)."""
        return decompose_matrix(self.world_matrix)[2]

    def bounds(
        self, *, space: str = "world", children: bool = False, include_hidden: bool = False
    ) -> ObjectBounds | None:
        """Bounding box, oriented box and sphere of the object's geometry -- with
        ``children``, also its descendants' -- in ``"world"`` or ``"local"`` space (``None``
        without geometry; see :func:`aether3d.spatial.bounds`)."""
        from .spatial import queries

        return queries.bounds(
            self._require_scene(),
            self,
            space=space,
            children=children,
            include_hidden=include_hidden,
        )

    @property
    def world_forward(self) -> Vec3:
        """Unit world direction of the object's front (local -Y): where cameras look
        and directional lights shine."""
        direction = -self.world_matrix[:3, 1]
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            raise ValidationError(f"object {self._name!r} has zero scale along its forward axis")
        x, y, z = (direction / length).tolist()
        return (x, y, z)

    # -- orientation -------------------------------------------------------

    def look_at(
        self, target: ObjectRef | Iterable[float], *, up: Iterable[float] = (0.0, 0.0, 1.0)
    ) -> None:
        """Turn the object so its front (local -Y) faces ``target`` and local +Z is up.

        ``target`` is a world-space point or another object (its world
        position). Parents, including scaled and rotated ones, are taken into
        account exactly. Raises ``ValidationError`` if the target coincides
        with the object's position. See :func:`aether3d.math3d.look_rotation_matrix`
        for what happens when looking straight along ``up``.
        """
        if isinstance(target, (SceneObject, str)):
            point = np.asarray(self._require_scene()._resolve_object(target).world_position)
        else:
            point = np.asarray(vec3(target, "target"))
        direction = point - np.asarray(self.world_position)
        if float(np.linalg.norm(direction)) < 1e-12:
            raise ValidationError(
                f"cannot aim {self._name!r} at its own position; move it or the target first"
            )
        self.look_along(direction, up=up)

    def look_along(
        self, direction: Iterable[float], *, up: Iterable[float] = (0.0, 0.0, 1.0)
    ) -> None:
        """Turn the object so its front (local -Y) points along a world ``direction``."""
        rotation = self._aim_rotation(direction, up)
        with self._edit(TRANSFORM, f"Aim {self._name!r}"):
            self._transform.rotation = rotation

    def _aim_rotation(self, direction: Iterable[float], up: Iterable[float]) -> Quat:
        """Local rotation that makes the world front/up axes match ``direction``/``up``.

        Exact under any invertible parent transform and the object's own
        (possibly negative) scale.
        """
        world_rot = look_rotation_matrix(direction, up)
        forward, upv = -world_rot[:, 1], world_rot[:, 2]
        parent = np.eye(3) if self._parent is None else self._parent.world_matrix[:3, :3]
        try:
            inv = np.linalg.inv(parent)
        except np.linalg.LinAlgError:
            raise ValidationError(
                f"cannot aim {self._name!r}: its parent has a zero world scale"
            ) from None
        sx, sy, sz = (np.sign(c) or 1.0 for c in self._transform.scale)
        y_local = inv @ (-sy * forward)
        y_local /= np.linalg.norm(y_local)
        w = inv @ (sz * upv)
        z_local = w - float(w @ y_local) * y_local
        z_len = float(np.linalg.norm(z_local))
        if z_len < 1e-9 * float(np.linalg.norm(w)):
            raise ValidationError(
                f"cannot aim {self._name!r}: its parent's transform collapses the up axis"
            )
        z_local /= z_len
        x_local = np.cross(y_local, z_local)
        return matrix3_to_quat(np.column_stack([x_local, y_local, z_local]))

    def set_world_position(self, position: Iterable[float]) -> None:
        """Move the object so its world position is ``position`` (parents respected)."""
        p = np.append(np.asarray(vec3(position, "position")), 1.0)
        if self._parent is not None:
            try:
                p = np.linalg.solve(self._parent.world_matrix, p)
            except np.linalg.LinAlgError:
                raise ValidationError(
                    f"cannot place {self._name!r}: its parent has a zero world scale"
                ) from None
        with self._edit(TRANSFORM, f"Move {self._name!r}"):
            self._transform.position = p[:3]

    def frame(
        self,
        targets: Any = None,
        *,
        direction: Iterable[float] | None = None,
        up: Iterable[float] = (0.0, 0.0, 1.0),
        margin: float = 0.05,
        fit_clipping: bool = True,
        aspect: float | None = None,
    ) -> Framing:
        """Camera objects: move the camera so ``targets`` exactly fill its frame.

        ``targets`` are objects (or ids/names); by default every visible mesh
        object. The camera keeps its orientation unless a view ``direction``
        is given. ``margin`` is the fraction of the frame left free around the
        geometry. With ``fit_clipping`` the near/far planes (and, for
        orthographic cameras, ``ortho_scale``) are adjusted to the geometry.
        ``aspect`` defaults to the camera's ``aspect_ratio`` or the scene's
        render settings. See :mod:`aether3d.rendering.framing`.
        """
        from .rendering.framing import frame_object

        with self._home._history._op(f"Frame {self._name!r}"):
            return frame_object(
                self,
                targets,
                direction=direction,
                up=up,
                margin=margin,
                fit_clipping=fit_clipping,
                aspect=aspect,
            )

    # -- editing -----------------------------------------------------------

    def edit_mesh(self, *, weld: str = "exact", quads: bool = True) -> EditSession:
        """Start an :class:`~aether3d.editing.EditSession` on this object's geometry.

        Objects without a mesh start from an empty editable mesh. The current
        mesh is never modified; ``commit()`` assigns a new one (to this object
        only, unless ``scope="shared"``). See ``docs/EDITING.md``.
        """
        from .editing import EditSession
        from .errors import EditError

        self._require_scene()
        if self._geometry is not None:
            raise EditError(
                "PROCEDURAL_OBJECT",
                f"object {self._name!r} is procedural; its mesh is regenerated from its geometry",
                hint="edit its geometry (modifiers/parameters), edit the base mesh with "
                "obj.geometry.edit_source(), or obj.bake() it first",
            )
        return EditSession(self, weld=weld, quads=quads)

    def sculpt(self) -> SculptSession:
        """Start a :class:`~aether3d.sculpt.SculptSession` on this object's mesh.

        Brush strokes, masks and remeshing work on a sculpt mesh; ``commit()`` assigns
        the result as one step of the scene's history (to this object only, unless
        ``scope="shared"``). See ``docs/MODELING.md``.
        """
        from .errors import SculptError
        from .sculpt.session import SculptSession

        self._require_scene()
        if self._geometry is not None:
            raise SculptError(
                "PROCEDURAL_OBJECT",
                f"object {self._name!r} is procedural; its mesh is regenerated from its geometry",
                hint="sculpt the base mesh with obj.geometry.sculpt_source(), or obj.bake() it "
                "first",
            )
        return SculptSession(self)

    # -- UV maps (docs/TEXTURING.md) ------------------------------------------

    def _uv_mesh(self) -> Mesh:
        from .errors import TexturingError

        self._require_scene()
        if self._geometry is not None:
            raise TexturingError(
                "PROCEDURAL_OBJECT",
                f"object {self._name!r} is procedural; its mesh is regenerated from its geometry",
                f"objects[{self._name}]",
                hint="bake it first (obj.bake())",
            )
        if self._mesh is None:
            raise TexturingError(
                "NOT_A_MESH_OBJECT", f"object {self._name!r} has no mesh", f"objects[{self._name}]"
            )
        return self._mesh

    def _set_uv_mesh(self, mesh: Mesh, label: str) -> None:
        with self._edit(DATA, f"{label} of {self._name!r}"):
            self._mesh = mesh

    def unwrap(self, **options: Any) -> dict[str, Any]:
        """New UVs for this object's mesh (:func:`aether3d.uv.unwrap`; options as there), as
        one step of the scene's history. Returns the report, with ``diagnostics``."""
        from . import uv

        result = uv.unwrap(self._uv_mesh(), **options)
        self._set_uv_mesh(result.mesh, "Unwrap UVs")
        return {**result.report, "diagnostics": [d.to_dict() for d in result.diagnostics]}

    def project_uvs(self, kind: str, **options: Any) -> dict[str, Any]:
        """UVs by projection (:func:`aether3d.uv.project`: ``"planar"``, ``"box"``,
        ``"cylindrical"``, ``"spherical"``), as one step of the scene's history."""
        from . import uv

        result = uv.project(self._uv_mesh(), kind, **options)
        self._set_uv_mesh(result.mesh, f"Project UVs ({kind})")
        return {**result.report, "diagnostics": [d.to_dict() for d in result.diagnostics]}

    def edit_uvs(self, operation: str, **options: Any) -> dict[str, Any]:
        """An island operation on this object's UVs, as one step of the scene's history:
        ``"transform"``, ``"align"``, ``"normalize"``, ``"texel_density"``, ``"pack"`` or
        ``"stitch"`` (the functions of :mod:`aether3d.uv`, with their options)."""
        from . import uv
        from .errors import TexturingError

        ops = {
            "transform": uv.transform_islands,
            "align": uv.align_islands,
            "normalize": uv.normalize_islands,
            "texel_density": uv.set_texel_density,
            "pack": uv.pack_islands,
            "stitch": uv.stitch,
        }
        fn = ops.get(operation)
        if fn is None:
            raise TexturingError(
                "INVALID_ARGUMENT",
                f"unknown UV operation {operation!r} (use {', '.join(ops)})",
                "operation",
            )
        result = fn(self._uv_mesh(), **options)
        self._set_uv_mesh(result.mesh, f"Edit UVs ({operation})")
        return {**result.report, "diagnostics": [d.to_dict() for d in result.diagnostics]}

    def uv_report(self, **options: Any) -> dict[str, Any]:
        """UV diagnostics of this object's mesh (:func:`aether3d.uv.uv_report`)."""
        from . import uv

        return uv.uv_report(self._uv_mesh(), **options)

    # -- misc --------------------------------------------------------------

    def _edit(self, aspect: Aspect, label: str) -> Any:
        """An operation changing ``aspect`` of this object, recorded in its scene's history."""
        return self._home._history._op(label, aspect, self)

    def _edit_value(self, aspect: Aspect, fields: Iterable[str]) -> Any:
        """The operation of an in-place change of this object's camera, light or appearance."""
        return self._edit(aspect, f"Set {aspect.name} {', '.join(fields)} of {self._name!r}")

    def _require_scene(self) -> Scene:
        if self._scene is None:
            raise DetachedObjectError(
                f"object {self._name!r} ({self._id}) has been removed from its scene"
            )
        return self._scene

    def __repr__(self) -> str:
        return f"SceneObject(name={self._name!r}, id={self._id!r})"


class Scene:
    """A named collection of :class:`SceneObject` instances arranged in a hierarchy.

    Example::

        scene = Scene("First Scene")
        cube = scene.add_cube(name="Cube", size=2)
        cube.position = (0, 0, 1)
        scene.save("example.aether")
        loaded = Scene.load("example.aether")
    """

    def __init__(self, name: str = "Untitled Scene") -> None:
        self._history = History(self)
        self._procedural: ProceduralSystem | None = None
        self._name = _check_name(name, "scene name")
        self._objects: dict[str, SceneObject] = {}
        self._by_name: dict[str, SceneObject] = {}
        self._name_hints: dict[str, int] = {}  # see _unique_name
        self._roots: list[SceneObject] = []
        self._active_camera: str | None = None
        self._world = _owned(World(), self, WORLD)
        self._render_settings = _owned(RenderSettings(), self, RENDER)
        from .materials import MaterialLibrary

        self._materials = MaterialLibrary(self)
        # Data that other modules derive from the scene (see _derived): never scene state.
        self._derived_data: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        value = _check_name(value, "scene name")
        with self._history._op(f"Rename scene to {value!r}", SCENE_NAME, self):
            self._name = value

    # -- history -----------------------------------------------------------

    @property
    def history(self) -> History:
        """The scene's undo/redo history (see :class:`aether3d.history.History`)."""
        return self._history

    @property
    def revision(self) -> int:
        """Increases with every change of the scene (undo and redo included); viewers and
        tools can compare it to know whether to refresh."""
        return self._history._revision

    def transaction(self, label: str = "Transaction") -> Any:
        """``with scene.transaction("Build tower"):`` -- every change in the block is one
        undo step, and none survives if the block raises (see
        :meth:`aether3d.history.History.transaction`)."""
        return self._history.transaction(label)

    def _derived(self) -> dict[str, Any]:
        """Caches that other modules compute from this scene, by consumer name.

        Not scene state: never saved, recorded or compared. Each consumer keys its entry by
        ``scene.revision`` (or its own bookkeeping) and rebuilds when that changes. Keeping
        the caches on the scene ties their lifetime to it: an entry may reference the scene
        (the spatial index holds its objects), which is an ordinary reference cycle, freed
        with the scene. A module-level ``WeakKeyDictionary`` keyed by the scene would keep
        such an entry -- and the scene -- alive forever, because its values are strong.
        """
        return self._derived_data

    def _edit_value(self, aspect: Aspect, fields: Iterable[str]) -> Any:
        """The operation of an in-place change of the world or the render settings."""
        return self._history._op(f"Set {aspect.name} {', '.join(fields)}", aspect, self)

    def _touch_siblings(self, parent: SceneObject | None) -> None:
        """Record the child list ``parent`` (or the root list) is about to change."""
        if parent is None:
            self._history._touch(ROOTS, self)
        else:
            self._history._touch(CHILDREN, parent)

    # -- creating objects --------------------------------------------------

    def add_object(
        self,
        name: str | None = None,
        *,
        mesh: Mesh | None = None,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        scale: float | Iterable[float] | None = None,
        object_id: str | None = None,
        geometry: ProceduralGeometry | str | None = None,
    ) -> SceneObject:
        """Add an object (optionally with a mesh) and return it.

        If ``name`` is omitted a unique one is generated (``"Object"``,
        ``"Object.001"``, ...). An explicit name that is already taken raises
        :class:`NameConflictError`. ``object_id`` is normally left out; it
        exists so that loaders can restore saved identifiers.
        """
        if name is None:
            name = self._unique_name("Object" if mesh is None else "Mesh")
        if geometry is not None:
            if mesh is not None:
                raise ValidationError("give either mesh= or geometry=, not both")
            node = self.procedural.geometry(geometry)  # validates the reference first
            with self._history._op(f"Add object {name!r}"):
                obj = self._create(name, None, parent, position, rotation, scale, object_id)
                self.procedural.assign(obj, node)  # an undoable procedural change
            return obj
        return self._create(name, mesh, parent, position, rotation, scale, object_id)

    def add_cube(
        self,
        name: str | None = None,
        *,
        size: float = 1.0,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        scale: float | Iterable[float] | None = None,
    ) -> SceneObject:
        """Add a cube with edge length ``size`` (see :func:`primitives.create_cube`)."""
        mesh = primitives.create_cube(size)
        name = self._unique_name("Cube") if name is None else name
        return self._create(name, mesh, parent, position, rotation, scale, None)

    def add_sphere(
        self,
        name: str | None = None,
        *,
        radius: float = 1.0,
        segments: int = 32,
        rings: int = 16,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        scale: float | Iterable[float] | None = None,
    ) -> SceneObject:
        """Add a UV sphere (see :func:`primitives.create_sphere`)."""
        mesh = primitives.create_sphere(radius, segments, rings)
        name = self._unique_name("Sphere") if name is None else name
        return self._create(name, mesh, parent, position, rotation, scale, None)

    def add_plane(
        self,
        name: str | None = None,
        *,
        size: float = 1.0,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        scale: float | Iterable[float] | None = None,
    ) -> SceneObject:
        """Add a square plane facing +Z (see :func:`primitives.create_plane`)."""
        mesh = primitives.create_plane(size)
        name = self._unique_name("Plane") if name is None else name
        return self._create(name, mesh, parent, position, rotation, scale, None)

    def _create(
        self,
        name: str,
        mesh: Mesh | None,
        parent: ObjectRef | None,
        position: Iterable[float] | None,
        rotation: Iterable[float] | None,
        scale: float | Iterable[float] | None,
        object_id: str | None,
    ) -> SceneObject:
        # Validate everything before mutating the scene.
        _check_name(name)
        if name in self._by_name:
            raise NameConflictError(f"an object named {name!r} already exists in this scene")
        if object_id is None:
            object_id = new_object_id()
        elif not isinstance(object_id, str) or not object_id:
            raise ValidationError(f"object_id must be a non-empty string, got {object_id!r}")
        if object_id in self._objects:
            raise ValidationError(f"an object with id {object_id!r} already exists in this scene")
        parent_obj = None if parent is None else self._resolve(parent)
        if mesh is not None and not isinstance(mesh, Mesh):
            raise ValidationError(f"mesh must be a Mesh or None, got {type(mesh).__name__}")
        transform = Transform()
        if position is not None:
            transform.position = position
        if rotation is not None:
            transform.rotation = rotation
        if scale is not None:
            transform.scale = scale

        obj = SceneObject(self, object_id, name, transform, mesh)
        history = self._history
        with history._op(f"Add object {name!r}", ALIVE, obj):
            history._touch(PARENT, obj)
            self._touch_siblings(parent_obj)
            self._objects[object_id] = obj
            self._by_name[name] = obj
            self._attach(obj, parent_obj)
        return obj

    def _create_many(self, records: list[NewObject], label: str | None = None) -> list[SceneObject]:
        """Create many objects in one operation (the bulk path of scene plans).

        Equivalent to one ``_create`` per record, followed by the component assignments
        that ``duplicate`` and ``merge`` make on new objects, but checked as a whole first
        and recorded once: each parent's child list is captured once, not once per object,
        so creating ``n`` objects costs O(n). Everything is validated before the scene
        changes; nothing is created if anything is wrong.
        """
        count = len(records)
        objects, by_name, materials = self._objects, self._by_name, self._materials._materials
        names: set[str] = set()
        ids: set[str] = set()
        tag_cache: dict[tuple, tuple[str, ...]] = {}
        meta_cache: dict[int, tuple[Any, str]] = {}
        resolved: list[tuple[str, str, SceneObject | int | None, tuple, str]] = []
        fresh_ids = iter(new_object_ids(sum(1 for rec in records if rec.object_id is None)))
        transforms_valid = _valid_transforms(records)
        for k, rec in enumerate(records):
            where = f"objects[{k}]"
            name = rec.name
            if not isinstance(name, str) or not name.strip():
                raise ValidationError(f"{where}: name must be a non-empty string, got {name!r}")
            if name in by_name:
                raise NameConflictError(
                    f"{where}: an object named {name!r} already exists in this scene"
                )
            if name in names:
                raise NameConflictError(f"{where}: two new objects would be named {name!r}")
            names.add(name)
            object_id = rec.object_id
            if object_id is None:
                object_id = next(fresh_ids)
            elif not isinstance(object_id, str) or not object_id:
                raise ValidationError(f"{where}: object_id must be a non-empty string")
            if object_id in objects or object_id in ids:
                raise ValidationError(f"{where}: an object with id {object_id!r} already exists")
            ids.add(object_id)
            parent = rec.parent
            if isinstance(parent, int) and not isinstance(parent, bool):
                if not 0 <= parent < k:
                    raise ValidationError(
                        f"{where}: parent index {parent} is not an earlier record"
                    )
            elif parent is not None:
                parent = self._resolve(parent)
            if rec.mesh is not None and not isinstance(rec.mesh, Mesh):
                raise ValidationError(f"{where}: mesh must be a Mesh or None")
            if rec.camera is not None and not isinstance(rec.camera, Camera):
                raise ValidationError(f"{where}: camera must be a Camera or None")
            if rec.light is not None and not isinstance(rec.light, Light):
                raise ValidationError(f"{where}: light must be a Light or None")
            if sum(v is not None for v in (rec.mesh, rec.camera, rec.light)) > 1:
                raise ValidationError(f"{where}: an object has at most one of mesh, camera, light")
            if (rec.camera is not None or rec.light is not None) and (
                rec.material is not None or rec.group_materials or rec.appearance is not None
            ):
                raise ValidationError(f"{where}: cameras and lights have no material or color")
            if not transforms_valid:
                _finite3(rec.position, f"{where}: position")
                _finite3(rec.scale, f"{where}: scale")
                q = rec.rotation
                if (
                    not isinstance(q, tuple)
                    or len(q) != 4
                    or not all(type(v) is float and math.isfinite(v) for v in q)
                    or abs(sum(v * v for v in q) - 1.0) > 1e-9
                ):
                    raise ValidationError(f"{where}: rotation must be a unit quaternion tuple")
            if rec.appearance is not None and not isinstance(rec.appearance, Appearance):
                raise ValidationError(f"{where}: appearance must be an Appearance")
            if not isinstance(rec.visible, bool):
                raise ValidationError(f"{where}: visible must be True or False")
            for mid in (rec.material, *(rec.group_materials or {}).values()):
                if mid is not None and mid not in materials:
                    raise ValidationError(f"{where}: no material with id {mid!r} in this scene")
            for group in rec.group_materials or {}:
                if isinstance(group, bool) or not isinstance(group, int) or not 0 <= group <= 65535:
                    raise ValidationError(f"{where}: face group {group!r} is not in [0, 65535]")
            raw_tags = rec.tags if isinstance(rec.tags, tuple) else tuple(rec.tags)
            tags = tag_cache.get(raw_tags)
            if tags is None:
                tags = tag_cache[raw_tags] = _check_tags(raw_tags)
            meta = rec.metadata
            if meta is None:
                text = "{}"
            else:
                cached = meta_cache.get(id(meta))
                if cached is None or cached[0] is not meta:
                    cached = meta_cache[id(meta)] = (meta, _check_metadata(meta))
                text = cached[1]
            resolved.append((name, object_id, parent, tags, text))

        history = self._history
        created: list[SceneObject] = []
        touched: set[int] = set()
        with history._op(label or f"Add {count} object{'s' * (count != 1)}"):
            for rec, (name, object_id, parent, tags, text) in zip(records, resolved, strict=True):
                parent_obj = created[parent] if isinstance(parent, int) else parent
                transform = Transform.__new__(Transform)
                transform._position, transform._rotation, transform._scale = (
                    rec.position,
                    rec.rotation,
                    rec.scale,
                )
                obj = SceneObject(self, object_id, name, transform, rec.mesh)
                # New in this operation: undo removes it whole, so its components need no
                # recording of their own (as in duplicate and merge).
                if rec.appearance is not None:
                    obj._appearance = _owned(rec.appearance._clone(), obj, APPEARANCE)
                obj._visible = rec.visible
                if rec.camera is not None:
                    obj._camera = _owned(rec.camera.copy(), obj, CAMERA)
                if rec.light is not None:
                    obj._light = _owned(rec.light.copy(), obj, LIGHT)
                obj._material = rec.material
                if rec.group_materials:
                    obj._group_materials = dict(rec.group_materials)
                obj._tags, obj._metadata = tags, text
                key = id(parent_obj)
                if key not in touched:
                    touched.add(key)
                    self._touch_siblings(parent_obj)
                history._touch(ALIVE, obj)
                history._touch(PARENT, obj)
                objects[object_id] = obj
                by_name[name] = obj
                self._attach(obj, parent_obj)
                created.append(obj)
        return created

    # -- lookup ------------------------------------------------------------

    def get(self, object_id: str) -> SceneObject:
        """Return the object with identifier ``object_id``."""
        try:
            return self._objects[object_id]
        except (KeyError, TypeError):
            raise ObjectNotFoundError(f"no object with id {object_id!r} in this scene") from None

    def get_by_name(self, name: str) -> SceneObject:
        """Return the object named ``name``."""
        try:
            return self._by_name[name]
        except (KeyError, TypeError):
            raise ObjectNotFoundError(f"no object named {name!r} in this scene") from None

    @property
    def roots(self) -> tuple[SceneObject, ...]:
        """Objects without a parent, in order."""
        return tuple(self._roots)

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        """All objects in depth-first hierarchy order (parents before children)."""
        return tuple(self)

    def __iter__(self) -> Iterator[SceneObject]:
        for root in self._roots:
            yield root
            yield from root.iter_descendants()

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, SceneObject):
            return item._scene is self
        return isinstance(item, str) and item in self._objects

    # -- mutation ----------------------------------------------------------

    def remove(self, obj: ObjectRef, *, recursive: bool = True) -> None:
        """Remove an object from the scene.

        With ``recursive=True`` (default) all descendants are removed too.
        With ``recursive=False`` the children are moved up to the removed
        object's parent, keeping their world-space placement.
        """
        target = self._resolve(obj)
        history = self._history
        doomed = [target, *target.iter_descendants()] if recursive else [target]
        more = f" and {len(doomed) - 1} descendant(s)" if len(doomed) > 1 else ""
        with history._op(f"Remove {target._name!r}{more}"):
            self._touch_siblings(target._parent)
            history._touch(PARENT, target)
            for item in doomed:
                history._touch(ALIVE, item)
            if any(item._id == self._active_camera for item in doomed):
                history._touch(ACTIVE_CAMERA, self)
            if not recursive:
                # Compute every new transform first so a failure leaves the scene intact.
                new_parent = target._parent
                moves = [
                    (child, self._local_transform_for(child, new_parent))
                    for child in target._children
                ]
                history._touch(CHILDREN, target)
                siblings = self._siblings(target)
                index = siblings.index(target)
                for offset, (child, transform) in enumerate(moves):
                    history._touch(TRANSFORM, child)
                    history._touch(PARENT, child)
                    child._place(transform)
                    child._parent = new_parent
                    siblings.insert(index + 1 + offset, child)
                target._children = []

            self._detach(target)
            for item in doomed:
                del self._objects[item._id]
                del self._by_name[item._name]
                self._name_freed(item._name)
                item._scene = None
                if item._id == self._active_camera:
                    self._active_camera = None

    def duplicate(
        self,
        obj: ObjectRef,
        name: str | None = None,
        *,
        parent: Any = "same",
        children: bool = True,
    ) -> SceneObject:
        """Copy an object -- and, with ``children``, its descendants -- in one undoable step.

        Copies get new ids and names (``name`` for the top copy, by default the first
        free ``"<name>.001"``, ``"<name>.002"``, ...), and the originals' transforms,
        visibility, colors, camera or light settings and materials. Meshes are
        immutable and shared with the originals; a procedural object's copy uses the
        same geometry node (``bake()`` either one to separate them). The copy has the
        original's parent unless ``parent`` (an object, or ``None`` for the scene
        root) says otherwise. A copied camera does not become the active camera.
        """
        source = self._resolve_object(obj)
        if name is not None:
            _check_name(name)
            if name in self._by_name:
                raise NameConflictError(f"an object named {name!r} already exists in this scene")
        if not isinstance(children, bool):
            raise ValidationError(f"children must be True or False, got {children!r}")
        if isinstance(parent, str) and parent == "same":
            parent_obj = source._parent
        else:
            parent_obj = None if parent is None else self._resolve_object(parent)
        # The subtree as it is now: a copy placed inside it is not copied again.
        nodes = [source, *source.iter_descendants()] if children else [source]
        kids = {id(n): list(n._children) for n in nodes} if children else {id(source): []}
        with self._history._op(f"Duplicate {source._name!r}"):
            top = self._copy_object(source, name, parent_obj)
            pending = [(source, top)]
            while pending:
                original, copy = pending.pop(0)
                for child in kids[id(original)]:
                    pending.append((child, self._copy_object(child, None, copy)))
        return top

    def _copy_object(
        self, source: SceneObject, name: str | None, parent: SceneObject | None
    ) -> SceneObject:
        name = self._unique_name(_base_name(source._name)) if name is None else name
        t = source._transform
        mesh = source._mesh if source._geometry is None else None
        obj = self._create(name, mesh, parent, t._position, t._rotation, t._scale, None)
        # The copy is new in this operation: undo removes it whole, so these writes need
        # no recording of their own.
        obj._visible = source._visible
        obj._appearance = _owned(source._appearance.copy(), obj, APPEARANCE)
        if source._camera is not None:
            obj._camera = _owned(source._camera.copy(), obj, CAMERA)
        if source._light is not None:
            obj._light = _owned(source._light.copy(), obj, LIGHT)
        obj._material = source._material
        obj._group_materials = dict(source._group_materials)
        obj._tags, obj._metadata = source._tags, source._metadata
        if source._geometry is not None:
            self.procedural.assign(obj, source._geometry)
        return obj

    def merge(
        self,
        source: Scene | str | os.PathLike[str],
        *,
        parent: ObjectRef | None = None,
        allow_lossy: bool = False,
    ) -> list[SceneObject]:
        """Copy the objects of another scene into this one, in one undoable step.

        ``source`` is a :class:`Scene`, a ``.aether`` project or a ``.glb``/``.gltf``
        file (imported with ``allow_lossy``). Its objects keep their hierarchy and are
        placed under ``parent`` (default: the scene root); they get new ids, and names
        free in this scene (``"Cube"`` becomes ``"Cube.001"`` if taken). Their materials
        and textures are copied too, renamed the same way, with every reference
        followed. Procedural objects are copied as static meshes of their current
        evaluation (their geometry nodes stay in the source); cameras and lights are
        copied, but the active camera, the world and the render settings stay this
        scene's. Returns the copies of the source's root objects.
        """
        other = _merge_source(source, allow_lossy)
        parent_obj = None if parent is None else self._resolve_object(parent)
        roots = list(other._roots)
        # The hierarchy as it is now (merging a scene into itself must not copy the copies).
        children = {id(o): list(o._children) for o in other}
        label = f"Merge {other._name!r} ({len(other)} object{'s' * (len(other) != 1)})"
        with self._history._op(label):
            materials = self._materials._copy_from(other._materials)
            copies: list[SceneObject] = []
            pending = [(obj, parent_obj) for obj in roots]
            while pending:
                original, target_parent = pending.pop(0)
                t = original._transform
                mesh = original.mesh  # procedural objects: the evaluation, as a static mesh
                name = self._unique_name(original._name)
                obj = self._create(
                    name, mesh, target_parent, t._position, t._rotation, t._scale, None
                )
                # New in this operation: undo removes it whole (no recording needed).
                obj._visible = original._visible
                obj._appearance = _owned(original._appearance.copy(), obj, APPEARANCE)
                if original._camera is not None:
                    obj._camera = _owned(original._camera.copy(), obj, CAMERA)
                if original._light is not None:
                    obj._light = _owned(original._light.copy(), obj, LIGHT)
                obj._material = materials.get(original._material)
                obj._group_materials = {
                    g: materials[m] for g, m in original._group_materials.items()
                }
                obj._tags, obj._metadata = original._tags, original._metadata
                if target_parent is parent_obj:
                    copies.append(obj)
                pending.extend((child, obj) for child in children[id(original)])
        return copies

    # -- cameras, lights and rendering -------------------------------------

    def add_camera(
        self,
        name: str | None = None,
        *,
        projection: str = "perspective",
        fov: float = 40.0,
        fov_axis: str = "vertical",
        ortho_scale: float = 10.0,
        near: float = 0.1,
        far: float = 1000.0,
        aspect_ratio: float | None = None,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        look_at: ObjectRef | Iterable[float] | None = None,
        up: Iterable[float] = (0.0, 0.0, 1.0),
        active: bool | None = None,
    ) -> SceneObject:
        """Add a camera object (see :class:`~aether3d.rendering.Camera`).

        A camera looks along its local -Y axis with +Z up; ``look_at`` aims it
        at a point or object after placing it. The new camera becomes the
        scene's active camera if ``active`` is true, or if it is ``None`` and
        there is no active camera yet. Everything is validated before the
        scene changes.
        """
        camera = Camera(projection, fov, fov_axis, ortho_scale, near, far, aspect_ratio)
        if active is not None and not isinstance(active, bool):
            raise ValidationError(f"active must be True, False or None, got {active!r}")
        label = "Add camera" if name is None else f"Add camera {name!r}"
        with self._history._op(label):
            obj = self._create_with(name, "Camera", parent, position, rotation, look_at, up)
            obj._camera = _owned(camera, obj, CAMERA)
            if active or (active is None and self._active_camera is None):
                self._history._touch(ACTIVE_CAMERA, self)
                self._active_camera = obj.id
        return obj

    def add_light(
        self,
        type: str = "directional",
        name: str | None = None,
        *,
        color: Any = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        range: float | None = None,
        falloff: str = "inverse_square",
        radius: float = 0.0,
        angle: float = 0.0,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        direction: Iterable[float] | None = None,
        look_at: ObjectRef | Iterable[float] | None = None,
        up: Iterable[float] = (0.0, 0.0, 1.0),
    ) -> SceneObject:
        """Add a light object (see :class:`~aether3d.rendering.Light`).

        Directional lights shine along their front (local -Y): give a world
        ``direction`` (the way the light travels) or a ``look_at`` target.
        Point lights shine from their ``position``. ``radius`` (point) and
        ``angle`` (directional, degrees) give lights a size for the path
        tracer's soft shadows.
        """
        light = Light(type, color, intensity, range, falloff, radius, angle)
        if direction is not None and look_at is not None:
            raise ValidationError("give either direction= or look_at=, not both")
        if direction is not None:
            look_rotation_matrix(direction, up)  # validate before creating anything
        label = "Add light" if name is None else f"Add light {name!r}"
        with self._history._op(label):  # all or nothing
            obj = self._create_with(name, "Light", parent, position, rotation, look_at, up)
            if direction is not None:
                obj.look_along(direction, up=up)
            obj._light = _owned(light, obj, LIGHT)
        return obj

    def add_directional_light(
        self,
        name: str | None = None,
        *,
        direction: Iterable[float] = (0.3, 0.5, -1.0),
        **options: Any,
    ) -> SceneObject:
        """Add a directional ("sun") light travelling along ``direction``."""
        return self.add_light("directional", name, direction=direction, **options)

    def add_point_light(
        self,
        name: str | None = None,
        *,
        position: Iterable[float] = (0.0, 0.0, 5.0),
        **options: Any,
    ) -> SceneObject:
        """Add a point light at ``position``."""
        return self.add_light("point", name, position=position, **options)

    def _create_with(
        self,
        name: str | None,
        base: str,
        parent: ObjectRef | None,
        position: Iterable[float] | None,
        rotation: Iterable[float] | None,
        look_at: ObjectRef | Iterable[float] | None,
        up: Iterable[float],
    ) -> SceneObject:
        """Create an empty object and aim it; nothing is added if aiming fails."""
        name = self._unique_name(base) if name is None else name
        with self._history._op(f"Add object {name!r}"):  # all or nothing
            obj = self._create(name, None, parent, position, rotation, None, None)
            if look_at is not None:
                obj.look_at(look_at, up=up)
        return obj

    @property
    def cameras(self) -> tuple[SceneObject, ...]:
        """Camera objects, in hierarchy order."""
        return tuple(o for o in self if o._camera is not None)

    @property
    def lights(self) -> tuple[SceneObject, ...]:
        """Light objects, in hierarchy order."""
        return tuple(o for o in self if o._light is not None)

    @property
    def active_camera(self) -> SceneObject | None:
        """The camera used by :meth:`render` when none is given (``None`` if unset)."""
        return None if self._active_camera is None else self._objects[self._active_camera]

    @active_camera.setter
    def active_camera(self, value: ObjectRef | None) -> None:
        camera_id = None
        if value is not None:
            obj = self._resolve_object(value)
            if obj._camera is None:
                raise ValidationError(f"object {obj.name!r} is not a camera")
            camera_id = obj.id
        label = "Clear active camera" if camera_id is None else f"Activate camera {obj._name!r}"
        with self._history._op(label, ACTIVE_CAMERA, self):
            self._active_camera = camera_id

    @property
    def world(self) -> World:
        """Background color and ambient light (edit in place: ``scene.world.background = ...``)."""
        return _owned(self._world, self, WORLD)

    @world.setter
    def world(self, value: World) -> None:
        if not isinstance(value, World):
            raise ValidationError(f"world must be a World, got {type(value).__name__}")
        new = value.copy()
        with self._history._op("Set world", WORLD, self):
            self._world._owner = None
            self._world = _owned(new, self, WORLD)

    @property
    def render_settings(self) -> RenderSettings:
        """Saved output settings (size, antialiasing, exposure, culling, default lights)."""
        return _owned(self._render_settings, self, RENDER)

    @render_settings.setter
    def render_settings(self, value: RenderSettings) -> None:
        if not isinstance(value, RenderSettings):
            raise ValidationError(
                f"render_settings must be a RenderSettings, got {type(value).__name__}"
            )
        new = value.copy()
        with self._history._op("Set render settings", RENDER, self):
            self._render_settings._owner = None
            self._render_settings = _owned(new, self, RENDER)

    def render(self, **options: Any) -> RenderResult:
        """Render an image (see :func:`aether3d.rendering.render`).

        Options override the saved :attr:`render_settings` for this render
        only: ``camera``, ``width``, ``height``, ``samples``, ``exposure``,
        ``cull``, ``default_lights``, ``frame``, ``limits``; and choose the
        renderer: ``backend`` (``"cpu"`` by default, ``"gpu"`` or ``"auto"``),
        ``adapter``, ``msaa``, ``supersample``, ``shadows``. The scene is not
        modified. Save the image with ``result.save("image.png")``.
        """
        from .rendering import render

        return render(self, **options)

    def _resolve_object(self, ref: ObjectRef) -> SceneObject:
        """An object of this scene by object, id or name."""
        if isinstance(ref, SceneObject):
            return self._resolve(ref)
        if isinstance(ref, str):
            if ref in self._objects:
                return self._objects[ref]
            if ref in self._by_name:
                return self._by_name[ref]
            raise ObjectNotFoundError(f"no object with id or name {ref!r} in this scene")
        raise ValidationError(f"expected an object, id or name, got {type(ref).__name__}")

    # -- materials -----------------------------------------------------------

    @property
    def materials(self) -> MaterialLibrary:
        """The scene's materials and textures (see docs/MATERIALS.md)."""
        return self._materials

    def create_material(
        self,
        name: str | None = None,
        *,
        model: str = "standard",
        preset: str | None = None,
        **params: Any,
    ) -> Material:
        """Create a material in :attr:`materials` (see
        :meth:`~aether3d.materials.MaterialLibrary.create_material`).

        ``preset`` (``"stone"``, ``"brick"``, ``"wood"``, ``"metal"``, ``"grass"``,
        ``"fabric"``) builds a material with procedural textures.
        """
        return self._materials.create_material(name, model=model, preset=preset, **params)

    def assign_material(self, objects: Any, material: Any, *, group: int | None = None) -> None:
        """Assign ``material`` (object, id, name or ``None``) to one or more objects,
        or only to face group ``group`` of their meshes. Checked fully before any change."""
        self._materials.assign(objects, material, group=group)

    def add_image_texture(self, name: str | None = None, path: Any = None, **options: Any):
        """Add a PNG/JPEG texture (see ``MaterialLibrary.add_image_texture``)."""
        return self._materials.add_image_texture(name, path, **options)

    def add_procedural_texture(self, name: str | None, graph: Any):
        """Add a procedural texture from a node graph (see docs/MATERIALS.md)."""
        return self._materials.add_procedural_texture(name, graph)

    # -- procedural geometry -----------------------------------------------

    @property
    def procedural(self) -> ProceduralSystem:
        """Procedural parameters, geometry nodes and their evaluation (see docs/PROCEDURAL.md)."""
        if self._procedural is None:
            from .procedural import ProceduralSystem

            self._procedural = ProceduralSystem(self)
        return self._procedural

    def add_geometry(
        self,
        name: str,
        primitive: str | None = None,
        *,
        mesh: Mesh | None = None,
        source: ProceduralGeometry | str | None = None,
        overrides: dict[str, Any] | None = None,
        after: str | int | None = None,
        parameters: dict[str, Any] | None = None,
        **params: Any,
    ) -> ProceduralGeometry:
        """Create a procedural geometry node (not yet used by any object).

        Exactly one source: ``primitive`` (a type from ``scene.procedural.types()``
        with ``**params``), ``mesh`` (a stored base mesh), or ``source`` (another
        geometry node, with optional parameter ``overrides`` and ``after`` = a
        modifier of it to stop at). ``parameters`` declares the node's own
        parameters as ``{name: value}`` or ``{name: {"value": ..., "min": ...}}``.
        """
        system = self.procedural
        with system.batch():
            node = system.add_geometry(
                name,
                primitive=primitive,
                params=params if primitive is not None else None,
                mesh=mesh,
                source=source,
                overrides=overrides,
                after=after,
            )
            if params and primitive is None:
                from .procedural.params import ProceduralError

                raise ProceduralError(
                    "INVALID_SOURCE", "keyword parameters are only for primitive sources"
                )
            for pname, spec in (parameters or {}).items():
                if isinstance(spec, dict) and "value" in spec:
                    options = {k: v for k, v in spec.items() if k != "value"}
                    system.add_parameter(pname, spec["value"], geometry=node, **options)
                else:
                    system.add_parameter(pname, spec, geometry=node)
            return node

    def add_procedural(
        self,
        name: str | None = None,
        primitive: str = "box",
        *,
        parent: ObjectRef | None = None,
        position: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None,
        scale: float | Iterable[float] | None = None,
        parameters: dict[str, Any] | None = None,
        **params: Any,
    ) -> SceneObject:
        """Create a primitive geometry node and an object that uses it (one atomic step).

        The geometry node gets the object's name (suffixed if that geometry
        name is taken); reach it as ``obj.geometry``.
        """
        name = self._unique_name(primitive.capitalize()) if name is None else name
        system = self.procedural
        taken = {g.name for g in system.geometries}
        gname, k = name, 1
        while gname in taken:
            gname, k = f"{name}.{k:03d}", k + 1
        with self._history._op(f"Add procedural {name!r}"), system.batch():
            node = self.add_geometry(gname, primitive, parameters=parameters, **params)
            obj = self.add_object(
                name,
                geometry=node,
                parent=parent,
                position=position,
                rotation=rotation,
                scale=scale,
            )
        return obj

    # -- persistence -------------------------------------------------------

    def save(self, path: str | os.PathLike[str], *, history: bool = False) -> None:
        """Save the scene to a ``.aether`` project file (written atomically).

        With ``history=True`` its undo history is saved too, next to it
        (``<path>.history``), so that a later session can undo and redo the changes
        made in this one (see :meth:`load`).
        """
        from .serialization import save_scene

        save_scene(self, path)
        if history:
            from pathlib import Path

            from ._history_file import history_path

            self._history.save(history_path(path), project_dir=Path(path).absolute().parent)

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, history: bool = False) -> Scene:
        """Load a scene from a ``.aether`` project file.

        Its history starts empty, unless ``history=True``: then the history saved with
        it (``scene.save(path, history=True)``) is restored and checked, and
        ``HistoryError`` is raised if there is none or if it no longer matches the
        project (``NO_HISTORY``, ``HISTORY_MISMATCH``, ``INVALID_HISTORY``).
        """
        from .serialization import load_scene

        scene = load_scene(path)
        if history:
            from pathlib import Path

            from ._history_file import history_path

            scene._history.load(history_path(path), project_dir=Path(path).absolute().parent)
        return scene

    def to_dict(self) -> dict[str, Any]:
        """The scene as a JSON-compatible document (the ``.aether`` file content)."""
        from .serialization import scene_to_dict

        return scene_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        """Build a scene from a document produced by :meth:`to_dict`."""
        from .serialization import scene_from_dict

        return scene_from_dict(data)

    # -- interchange and inspection ----------------------------------------

    def export(
        self,
        path: str | os.PathLike[str],
        *,
        allow_lossy: bool = False,
        bake_textures: int | None = None,
    ) -> ExportResult:
        """Export to an interchange format chosen by extension (``.glb`` or ``.gltf``).

        Returns an :class:`aether3d.gltf.ExportResult` describing what was
        written. See :func:`aether3d.gltf.export_gltf` for details
        (``bake_textures`` bakes procedural textures to PNG images of that size).
        """
        from .gltf import export_gltf

        return export_gltf(self, path, allow_lossy=allow_lossy, bake_textures=bake_textures)

    @classmethod
    def import_file(
        cls,
        path: str | os.PathLike[str],
        *,
        allow_lossy: bool = False,
        limits: ImportLimits | None = None,
    ) -> Scene:
        """Import a ``.glb`` or ``.gltf`` file as a new scene.

        Use :func:`aether3d.gltf.import_gltf` instead to also receive the
        import report (diagnostics and the node-to-object map).
        """
        from .gltf import import_gltf

        return import_gltf(path, allow_lossy=allow_lossy, limits=limits).scene

    def inspect_geometry(self) -> SceneGeometryReport:
        """Analyse all geometry in the scene (see :func:`aether3d.inspect_scene`)."""
        from .inspection import inspect_scene

        return inspect_scene(self)

    # -- spatial queries (aether3d.spatial; read-only: no history step) --------

    def bounds(
        self,
        obj: ObjectRef | None = None,
        *,
        space: str = "world",
        children: bool = False,
        include_hidden: bool = False,
    ) -> ObjectBounds | None:
        """Bounding box, oriented box and sphere of an object's geometry (with ``children``,
        also its descendants'), or of the whole scene (see :func:`aether3d.spatial.bounds`)."""
        from .spatial import queries

        return queries.bounds(
            self, obj, space=space, children=children, include_hidden=include_hidden
        )

    def find_objects(self, **filters: Any) -> list[SceneObject]:
        """Objects matching filters -- ``name`` (glob), ``kind``, ``tags``, ``any_tags``,
        ``metadata``, ``metadata_keys``, ``parent``, ``ancestor``, ``visible``,
        ``material``, ``region`` (see :func:`aether3d.spatial.find_objects`)."""
        from .spatial import queries

        return queries.find_objects(self, **filters)

    def raycast(
        self,
        origin: Iterable[float],
        direction: Iterable[float],
        *,
        max_distance: float = math.inf,
        min_distance: float = 0.0,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
        surfaces: str = "geometry",
    ) -> SurfaceHit | None:
        """The first surface along a ray, or ``None`` (see :func:`aether3d.spatial.raycast`)."""
        from .spatial import queries

        return queries.raycast(
            self,
            origin,
            direction,
            max_distance=max_distance,
            min_distance=min_distance,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
            surfaces=surfaces,
        )

    def raycast_all(
        self,
        origin: Iterable[float],
        direction: Iterable[float],
        *,
        max_distance: float = math.inf,
        min_distance: float = 0.0,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
        surfaces: str = "geometry",
        limit: int | None = None,
    ) -> list[SurfaceHit]:
        """Every surface along a ray, nearest first (see :func:`aether3d.spatial.raycast_all`)."""
        from .spatial import queries

        return queries.raycast_all(
            self,
            origin,
            direction,
            max_distance=max_distance,
            min_distance=min_distance,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
            surfaces=surfaces,
            limit=limit,
        )

    def closest_point(
        self,
        point: Iterable[float],
        *,
        max_distance: float = math.inf,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
    ) -> SurfaceHit | None:
        """The surface point nearest to ``point`` and its distance, or ``None`` (see
        :func:`aether3d.spatial.closest_point`)."""
        from .spatial import queries

        return queries.closest_point(
            self,
            point,
            max_distance=max_distance,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
        )

    def nearest_objects(
        self,
        point: Iterable[float],
        *,
        count: int | None = None,
        max_distance: float = math.inf,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
    ) -> list[SurfaceHit]:
        """Objects by the distance of their surface from ``point``, nearest first (see
        :func:`aether3d.spatial.nearest_objects`)."""
        from .spatial import queries

        return queries.nearest_objects(
            self,
            point,
            count=count,
            max_distance=max_distance,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
        )

    def contains_point(
        self, obj: ObjectRef, point: Iterable[float], *, tolerance: float = 1e-6
    ) -> Containment:
        """Whether ``point`` is inside ``obj``'s closed solid: inside, outside, surface or
        undefined (see :func:`aether3d.spatial.contains_point`)."""
        from .spatial import queries

        return queries.contains_point(self, obj, point, tolerance=tolerance)

    def distance(self, a: ObjectRef, b: ObjectRef) -> ObjectDistance:
        """The shortest distance between two objects' surfaces (see
        :func:`aether3d.spatial.distance`)."""
        from .spatial import queries

        return queries.distance(self, a, b)

    def overlaps(
        self,
        objects: Any = None,
        *,
        against: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
        exact: bool = True,
        tolerance: float = 0.0,
    ) -> list[Overlap]:
        """Pairs of objects whose geometry intersects or nests (``exact=False``: whose
        bounds overlap) (see :func:`aether3d.spatial.overlaps`)."""
        from .spatial import queries

        return queries.overlaps(
            self,
            objects,
            against=against,
            exclude=exclude,
            include_hidden=include_hidden,
            exact=exact,
            tolerance=tolerance,
        )

    def query_region(
        self,
        region: Any,
        *,
        mode: str = "intersects",
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
    ) -> list[RegionMatch]:
        """Objects in a :class:`~aether3d.spatial.Box` or :class:`~aether3d.spatial.Sphere`
        (see :func:`aether3d.spatial.query_region`)."""
        from .spatial import queries

        return queries.query_region(
            self,
            region,
            mode=mode,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
        )

    def camera_ray(
        self,
        x: float,
        y: float,
        *,
        camera: ObjectRef | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> CameraRay:
        """The ray a camera sees through image point ``(x, y)`` in pixels (see
        :func:`aether3d.spatial.camera_ray`)."""
        from .spatial import queries

        return queries.camera_ray(self, x, y, camera=camera, width=width, height=height)

    def pick(
        self,
        x: float,
        y: float,
        *,
        camera: ObjectRef | None = None,
        width: int | None = None,
        height: int | None = None,
        objects: Any = None,
        exclude: Any = None,
        surfaces: str = "rendered",
        cull: str | None = None,
    ) -> SurfaceHit | None:
        """The surface a camera shows at image point ``(x, y)`` (see
        :func:`aether3d.spatial.pick`)."""
        from .spatial import queries

        return queries.pick(
            self,
            x,
            y,
            camera=camera,
            width=width,
            height=height,
            objects=objects,
            exclude=exclude,
            surfaces=surfaces,
            cull=cull,
        )

    def project(
        self,
        points: Any,
        *,
        camera: ObjectRef | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> ScreenPoint | list[ScreenPoint]:
        """Where world points appear in a camera's image (see
        :func:`aether3d.spatial.project`)."""
        from .spatial import queries

        return queries.project(self, points, camera=camera, width=width, height=height)

    def objects_in_view(
        self,
        *,
        camera: ObjectRef | None = None,
        width: int | None = None,
        height: int | None = None,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
        exact: bool = True,
    ) -> list[ViewMatch]:
        """Objects in a camera's view volume, ignoring occlusion (see
        :func:`aether3d.spatial.objects_in_view`)."""
        from .spatial import queries

        return queries.objects_in_view(
            self,
            camera=camera,
            width=width,
            height=height,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
            exact=exact,
        )

    def line_of_sight(
        self,
        start: Any,
        end: Any,
        *,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
        surfaces: str = "geometry",
        margin: float = 1e-6,
    ) -> LineOfSight:
        """Whether nothing lies between two points (or objects' positions) (see
        :func:`aether3d.spatial.line_of_sight`)."""
        from .spatial import queries

        return queries.line_of_sight(
            self,
            start,
            end,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
            surfaces=surfaces,
            margin=margin,
        )

    def visible_from(
        self,
        target: ObjectRef,
        observer: Any,
        *,
        samples: int = 64,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
        surfaces: str = "geometry",
        margin: float = 1e-6,
        width: int | None = None,
        height: int | None = None,
    ) -> Visibility:
        """How much of ``target``'s surface an observer (a point, an object or a camera)
        sees (see :func:`aether3d.spatial.visible_from`)."""
        from .spatial import queries

        return queries.visible_from(
            self,
            target,
            observer,
            samples=samples,
            objects=objects,
            exclude=exclude,
            include_hidden=include_hidden,
            surfaces=surfaces,
            margin=margin,
            width=width,
            height=height,
        )

    def relation(
        self,
        a: ObjectRef,
        b: ObjectRef,
        *,
        near_distance: float | None = None,
        tolerance: float = 1e-6,
        viewer: Any = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Relation:
        """Spatial relations of ``a`` to ``b``: distance, intersecting, near, above, below,
        inside, contains, in front of, behind (see :func:`aether3d.spatial.relation`)."""
        from .spatial import queries

        return queries.relation(
            self,
            a,
            b,
            near_distance=near_distance,
            tolerance=tolerance,
            viewer=viewer,
            width=width,
            height=height,
        )

    # -- scene plans (aether3d.plans) ---------------------------------------------

    def execute_plan(self, plan: Any, **options: Any) -> PlanResult:
        """Build or change the scene with a declarative scene plan, in one checked step.

        ``plan`` is a dict (or JSON text) in the ``aether3d.scene-plan`` format; the whole
        plan is checked before anything changes, then applied as one undo step (all or
        nothing). ``options``: ``mode`` (``"execute"``, ``"dry_run"``, ``"validate"``),
        ``expected_revision``, ``limits``, ``base_dir``, ``progress``, ``cancel``,
        ``timeout`` (see :func:`aether3d.plans.execute_plan` and docs/SCENE_PLANS.md).
        """
        from .plans import execute_plan

        return execute_plan(self, plan, **options)

    def observe(self, view: Any = None, **options: Any) -> Any:
        """Render a view and measure what it shows: an :class:`aether3d.feedback.Observation`
        (per-object visible pixels, bounds, depth, occlusion; image statistics; images).

        ``view`` is a view specification (``None``: the active camera; ``{"preset": "top"}``,
        ``{"focus": {"objects": "Lamp"}}``, ...); ``options`` as
        :func:`aether3d.feedback.observe` takes them (``width``, ``height``, ``backend``,
        ``focus``, ``occlusion``, ...). Never changes the scene. See docs/FEEDBACK.md.
        """
        from .feedback import observe

        return observe(self, view, **options)

    def describe(self, *, include_hidden: bool = True) -> dict[str, Any]:
        """A compact JSON summary of the scene's objects and their placement (see
        :func:`aether3d.spatial.describe`)."""
        from .spatial import queries

        return queries.describe(self, include_hidden=include_hidden)

    # -- internals ---------------------------------------------------------

    def _resolve(self, ref: ObjectRef) -> SceneObject:
        if isinstance(ref, SceneObject):
            if ref._scene is not self:
                raise ObjectNotFoundError(f"{ref!r} does not belong to this scene")
            return ref
        if isinstance(ref, str):
            return self.get(ref)
        raise ValidationError(
            f"expected a SceneObject or object id string, got {type(ref).__name__}"
        )

    def unique_name(self, base: str) -> str:
        """Return ``base`` if unused, else the first free ``base.001``, ``base.002``, ..."""
        return self._unique_name(_check_name(base, "base name"))

    def _unique_name(self, base: str) -> str:
        by_name = self._by_name
        if base not in by_name:
            return base
        # ``_name_hints[base]`` is a counter below which every ``base.NNN`` is taken, so
        # naming many objects after one base costs O(1) each instead of a scan from .001.
        counter = self._name_hints.get(base, 1)
        while f"{base}.{counter:03d}" in by_name:
            counter += 1
        self._name_hints[base] = counter
        return f"{base}.{counter:03d}"

    def _name_freed(self, name: str) -> None:
        """Forget the naming hint that ``name`` (no longer used) may invalidate."""
        stem, dot, digits = name.rpartition(".")
        if dot and digits.isdigit():
            self._name_hints.pop(stem, None)

    def _set_name_index(self, by_name: dict[str, SceneObject]) -> None:
        """Replace the name index (the history rebuilds it after undo and redo)."""
        self._by_name = by_name
        self._name_hints.clear()

    def _rename(self, obj: SceneObject, new_name: str) -> None:
        _check_name(new_name)
        if new_name == obj._name:
            return
        if new_name in self._by_name:
            raise NameConflictError(f"an object named {new_name!r} already exists in this scene")
        with self._history._op(f"Rename {obj._name!r} to {new_name!r}", NAME, obj):
            del self._by_name[obj._name]
            self._name_freed(obj._name)
            obj._name = new_name
            self._by_name[new_name] = obj

    def _siblings(self, obj: SceneObject) -> list[SceneObject]:
        return self._roots if obj._parent is None else obj._parent._children

    def _attach(self, obj: SceneObject, parent: SceneObject | None) -> None:
        obj._parent = parent
        (self._roots if parent is None else parent._children).append(obj)

    def _detach(self, obj: SceneObject) -> None:
        self._siblings(obj).remove(obj)
        obj._parent = None

    @staticmethod
    def _local_transform_for(obj: SceneObject, new_parent: SceneObject | None) -> Transform:
        """Local transform that keeps ``obj``'s world placement under ``new_parent``."""
        world = obj.world_matrix
        if new_parent is not None:
            try:
                world = np.linalg.inv(new_parent.world_matrix) @ world
            except np.linalg.LinAlgError:
                raise ValidationError(
                    f"cannot keep world transform: {new_parent!r} has a zero world scale"
                ) from None
        return Transform.from_matrix(world)

    def _reparent(
        self, obj: SceneObject, parent: ObjectRef | None, keep_world_transform: bool
    ) -> None:
        new_parent = None if parent is None else self._resolve(parent)
        if new_parent is obj:
            raise HierarchyError(f"{obj!r} cannot be its own parent")
        if new_parent is not None and obj in new_parent.iter_ancestors():
            raise HierarchyError(f"cannot parent {obj!r} to its own descendant {new_parent!r}")
        transform = self._local_transform_for(obj, new_parent) if keep_world_transform else None
        label = (
            f"Unparent {obj._name!r}"
            if new_parent is None
            else f"Parent {obj._name!r} to {new_parent._name!r}"
        )
        history = self._history
        with history._op(label, PARENT, obj):
            self._touch_siblings(obj._parent)
            self._touch_siblings(new_parent)
            if transform is not None:
                history._touch(TRANSFORM, obj)
            self._detach(obj)
            self._attach(obj, new_parent)
            if transform is not None:
                obj._place(transform)

    def __repr__(self) -> str:
        return f"Scene(name={self._name!r}, objects={len(self._objects)})"
