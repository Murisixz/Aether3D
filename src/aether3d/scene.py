"""Scene graph: scenes, scene objects, and their hierarchy."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator
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
from .math3d import Quat, Vec3, decompose_matrix
from .mesh import Mesh
from .transform import Transform

if TYPE_CHECKING:
    from .gltf import ExportResult, ImportLimits
    from .inspection import SceneGeometryReport

ObjectRef = Union["SceneObject", str]
"""A scene object, or its identifier."""


def new_object_id() -> str:
    """Return a new globally unique object identifier (a UUID4 string)."""
    return str(uuid.uuid4())


def _check_name(name: Any, what: str = "name") -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{what} must be a non-empty string, got {name!r}")
    return name


class SceneObject:
    """A node in a scene's hierarchy.

    Every object has a stable unique :attr:`id`, a :attr:`name` that is unique
    within its scene, a local :attr:`transform` relative to its parent, and an
    optional :attr:`mesh`. Objects without a mesh are useful as groups/pivots.

    Objects are created through a :class:`Scene` (``scene.add_object()``,
    ``scene.add_cube()``, ...), never directly.
    """

    __slots__ = ("_scene", "_id", "_name", "_transform", "_mesh", "_parent", "_children")

    def __init__(
        self, scene: Scene, object_id: str, name: str, transform: Transform, mesh: Mesh | None
    ) -> None:
        self._scene: Scene | None = scene
        self._id = object_id
        self._name = name
        self._transform = transform
        self._mesh = mesh
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
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh | None) -> None:
        if value is not None and not isinstance(value, Mesh):
            raise ValidationError(f"mesh must be a Mesh or None, got {type(value).__name__}")
        self._mesh = value

    @property
    def transform(self) -> Transform:
        """Local transform, relative to the parent (or the scene root)."""
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        if not isinstance(value, Transform):
            raise ValidationError(f"transform must be a Transform, got {type(value).__name__}")
        self._transform = value.copy()

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

    # -- misc --------------------------------------------------------------

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
        self._name = _check_name(name, "scene name")
        self._objects: dict[str, SceneObject] = {}
        self._by_name: dict[str, SceneObject] = {}
        self._roots: list[SceneObject] = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = _check_name(value, "scene name")

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
    ) -> SceneObject:
        """Add an object (optionally with a mesh) and return it.

        If ``name`` is omitted a unique one is generated (``"Object"``,
        ``"Object.001"``, ...). An explicit name that is already taken raises
        :class:`NameConflictError`. ``object_id`` is normally left out; it
        exists so that loaders can restore saved identifiers.
        """
        if name is None:
            name = self._unique_name("Object" if mesh is None else "Mesh")
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
        self._objects[object_id] = obj
        self._by_name[name] = obj
        self._attach(obj, parent_obj)
        return obj

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
        if recursive:
            doomed = [target, *target.iter_descendants()]
        else:
            # Compute every new transform first so a failure leaves the scene intact.
            new_parent = target._parent
            moves = [
                (child, self._local_transform_for(child, new_parent)) for child in target._children
            ]
            siblings = self._siblings(target)
            index = siblings.index(target)
            for offset, (child, transform) in enumerate(moves):
                child._transform = transform
                child._parent = new_parent
                siblings.insert(index + 1 + offset, child)
            target._children = []
            doomed = [target]

        self._detach(target)
        for item in doomed:
            del self._objects[item._id]
            del self._by_name[item._name]
            item._scene = None

    # -- persistence -------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the scene to a ``.aether`` project file (written atomically)."""
        from .serialization import save_scene

        save_scene(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Scene:
        """Load a scene from a ``.aether`` project file."""
        from .serialization import load_scene

        return load_scene(path)

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

    def export(self, path: str | os.PathLike[str], *, allow_lossy: bool = False) -> ExportResult:
        """Export to an interchange format chosen by extension (``.glb`` or ``.gltf``).

        Returns an :class:`aether3d.gltf.ExportResult` describing what was
        written. See :func:`aether3d.gltf.export_gltf` for details.
        """
        from .gltf import export_gltf

        return export_gltf(self, path, allow_lossy=allow_lossy)

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
        if base not in self._by_name:
            return base
        counter = 1
        while f"{base}.{counter:03d}" in self._by_name:
            counter += 1
        return f"{base}.{counter:03d}"

    def _rename(self, obj: SceneObject, new_name: str) -> None:
        _check_name(new_name)
        if new_name == obj._name:
            return
        if new_name in self._by_name:
            raise NameConflictError(f"an object named {new_name!r} already exists in this scene")
        del self._by_name[obj._name]
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
        self._detach(obj)
        self._attach(obj, new_parent)
        if transform is not None:
            obj._transform = transform

    def __repr__(self) -> str:
        return f"Scene(name={self._name!r}, objects={len(self._objects)})"
