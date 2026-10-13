"""Aether3D: an open-source, AI-native, code-first 3D engine.

Quick start::

    from aether3d import Scene

    scene = Scene("First Scene")
    cube = scene.add_cube(name="Cube", size=2)
    cube.position = (0, 0, 1)
    scene.save("example.aether")
    loaded = Scene.load("example.aether")

    scene.export("example.glb")              # glTF 2.0 binary
    imported = Scene.import_file("example.glb")
    report = imported.inspect_geometry().to_dict()

Everything listed in ``__all__`` is the supported public API.
"""

from ._version import __version__
from .diagnostics import Diagnostic
from .editing import (
    CommitResult,
    EditableMesh,
    EditSession,
    OperationResult,
    Selection,
    TopologyReport,
)
from .editing.script import ScriptError, run_edit_script
from .errors import (
    AetherError,
    AssetFormatError,
    DetachedObjectError,
    EditError,
    HierarchyError,
    NameConflictError,
    ObjectNotFoundError,
    ProjectFormatError,
    ResourceLimitError,
    UnsupportedFeatureError,
    ValidationError,
)
from .gltf import ExportResult, ImportLimits, ImportResult, export_gltf, import_gltf
from .inspection import MeshGeometryReport, SceneGeometryReport, inspect_mesh, inspect_scene
from .mesh import Mesh
from .primitives import create_cube, create_plane, create_sphere
from .scene import Scene, SceneObject
from .serialization import FORMAT_VERSION, load_scene, save_scene
from .transform import Transform

__all__ = [
    "FORMAT_VERSION",
    "AetherError",
    "AssetFormatError",
    "CommitResult",
    "DetachedObjectError",
    "Diagnostic",
    "EditError",
    "EditSession",
    "EditableMesh",
    "ExportResult",
    "HierarchyError",
    "ImportLimits",
    "ImportResult",
    "Mesh",
    "MeshGeometryReport",
    "NameConflictError",
    "ObjectNotFoundError",
    "OperationResult",
    "ProjectFormatError",
    "ResourceLimitError",
    "Scene",
    "SceneGeometryReport",
    "SceneObject",
    "ScriptError",
    "Selection",
    "TopologyReport",
    "Transform",
    "UnsupportedFeatureError",
    "ValidationError",
    "__version__",
    "create_cube",
    "create_plane",
    "create_sphere",
    "export_gltf",
    "import_gltf",
    "inspect_mesh",
    "inspect_scene",
    "load_scene",
    "run_edit_script",
    "save_scene",
]
