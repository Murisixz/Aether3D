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

    # Non-destructive procedural geometry (docs/PROCEDURAL.md)
    tower = scene.add_procedural("Tower", "cylinder", radius=1, depth=4, segments=8)
    tower.geometry.add_modifier("bevel", width=0.05, limit="angle", angle=40)

    # Materials and textures (docs/MATERIALS.md)
    tower.material = scene.create_material("Stone", preset="stone")

    # Spatial queries (docs/SPATIAL.md): read-only, exact, in world space
    hit = scene.raycast((0, 0, 10), (0, 0, -1))  # -> hit.object.name, hit.position
    near = scene.nearest_objects((3, 0, 0), count=2)

    # Scene plans (docs/SCENE_PLANS.md): a whole scene in one checked, atomic step
    result = scene.execute_plan({
        "format": "aether3d.scene-plan", "version": 1,
        "meshes": {"block": {"primitive": "cube", "size": 1}},
        "objects": {"wall": {"mesh": "@block", "grid": {"count": [20, 1, 5], "spacing": 1}}},
    })

    # Visual feedback (docs/FEEDBACK.md): look at the render, check it, correct it
    obs = scene.observe(width=640, height=360)        # measured facts per object
    feedback = FeedbackSession(scene, expectations)   # an aether3d.expectations document
    report = feedback.run()                           # observe, diagnose, correct, verify

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
    FeedbackError,
    HierarchyError,
    HistoryError,
    MaterialError,
    NameConflictError,
    ObjectNotFoundError,
    PlanError,
    ProjectFormatError,
    QueryError,
    RenderError,
    ResourceLimitError,
    SculptError,
    UnsupportedFeatureError,
    ValidationError,
)
from .feedback import FeedbackSession
from .gltf import ExportResult, ImportLimits, ImportResult, export_gltf, import_gltf
from .history import History, HistoryEntry, Transaction
from .inspection import MeshGeometryReport, SceneGeometryReport, inspect_mesh, inspect_scene
from .materials import (
    ImageTexture,
    Material,
    MaterialLibrary,
    ProceduralTexture,
    TextureBinding,
    TextureGraph,
)
from .mesh import Mesh
from .plans import PlanLimits, PlanResult, execute_plan, plan_schema, validate_plan
from .primitives import create_cube, create_plane, create_sphere
from .procedural import (
    Expr,
    Modifier,
    PolyMesh,
    ProceduralError,
    ProceduralGeometry,
    ProceduralSystem,
)
from .procedural.script import run_procedural_script
from .rendering import (
    AOSettings,
    Appearance,
    Camera,
    Environment,
    Light,
    PathSettings,
    RenderLimits,
    RenderResult,
    RenderSettings,
    ShadowSettings,
    World,
    gpu_status,
    list_adapters,
    release_gpu,
    render,
)
from .rendering.script import run_scene_script
from .scene import Scene, SceneObject
from .serialization import FORMAT_VERSION, load_scene, save_scene
from .spatial import run_query
from .transform import Transform

__all__ = [
    "FORMAT_VERSION",
    "AOSettings",
    "AetherError",
    "Appearance",
    "AssetFormatError",
    "Camera",
    "CommitResult",
    "DetachedObjectError",
    "Diagnostic",
    "EditError",
    "History",
    "HistoryEntry",
    "HistoryError",
    "EditSession",
    "EditableMesh",
    "Environment",
    "ExportResult",
    "Expr",
    "FeedbackError",
    "FeedbackSession",
    "HierarchyError",
    "ImageTexture",
    "ImportLimits",
    "ImportResult",
    "Light",
    "Material",
    "MaterialError",
    "MaterialLibrary",
    "Mesh",
    "MeshGeometryReport",
    "Modifier",
    "NameConflictError",
    "ObjectNotFoundError",
    "OperationResult",
    "PathSettings",
    "PlanError",
    "PlanLimits",
    "PlanResult",
    "PolyMesh",
    "ProceduralError",
    "ProceduralGeometry",
    "ProceduralSystem",
    "ProceduralTexture",
    "ProjectFormatError",
    "QueryError",
    "RenderError",
    "RenderLimits",
    "RenderResult",
    "RenderSettings",
    "ResourceLimitError",
    "Scene",
    "SceneGeometryReport",
    "SceneObject",
    "ScriptError",
    "SculptError",
    "Transaction",
    "ShadowSettings",
    "Selection",
    "TextureBinding",
    "TextureGraph",
    "TopologyReport",
    "Transform",
    "UnsupportedFeatureError",
    "ValidationError",
    "World",
    "__version__",
    "create_cube",
    "create_plane",
    "create_sphere",
    "execute_plan",
    "export_gltf",
    "gpu_status",
    "import_gltf",
    "inspect_mesh",
    "inspect_scene",
    "list_adapters",
    "load_scene",
    "plan_schema",
    "release_gpu",
    "render",
    "run_edit_script",
    "run_procedural_script",
    "run_query",
    "run_scene_script",
    "save_scene",
    "validate_plan",
]
