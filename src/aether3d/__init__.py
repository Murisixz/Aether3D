"""Aether3D: an open-source, AI-native, code-first 3D engine.

Quick start::

    from aether3d import Scene

    scene = Scene("First Scene")
    cube = scene.add_cube(name="Cube", size=2)
    cube.position = (0, 0, 1)
    scene.save("example.aether")
    loaded = Scene.load("example.aether")

Everything listed in ``__all__`` is the supported public API.
"""

from ._version import __version__
from .errors import (
    AetherError,
    DetachedObjectError,
    HierarchyError,
    NameConflictError,
    ObjectNotFoundError,
    ProjectFormatError,
    ValidationError,
)
from .mesh import Mesh
from .primitives import create_cube, create_plane, create_sphere
from .scene import Scene, SceneObject
from .serialization import FORMAT_VERSION, load_scene, save_scene
from .transform import Transform

__all__ = [
    "FORMAT_VERSION",
    "AetherError",
    "DetachedObjectError",
    "HierarchyError",
    "Mesh",
    "NameConflictError",
    "ObjectNotFoundError",
    "ProjectFormatError",
    "Scene",
    "SceneObject",
    "Transform",
    "ValidationError",
    "__version__",
    "create_cube",
    "create_plane",
    "create_sphere",
    "load_scene",
    "save_scene",
]
