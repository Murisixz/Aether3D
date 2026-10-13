"""Render-scene extraction: everything a renderer backend needs, and nothing else.

:func:`extract_render_scene` walks a :class:`~aether3d.Scene` once and returns
an immutable :class:`RenderScene`:

* unique meshes (``RenderMesh``) -- the scene's own read-only arrays, shared
  by every instance that uses them (procedural objects contribute their
  *current evaluated* mesh);
* instances (``RenderInstance``) -- one per visible mesh object, with its
  world matrix, normal matrix and linear-light appearance;
* lights in world space (``RenderLight``), the camera (``RenderCamera``) and
  the linear background and ambient colors.

Extraction never modifies the scene. Backends (today the CPU rasterizer,
later GPU backends) consume a ``RenderScene`` without touching scene objects,
so they all render exactly the same content.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import AetherError, RenderError
from ..mesh import Mesh
from .camera import Camera, CameraFrame, frame_from_matrix
from .color import to_linear
from .settings import Appearance

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject


@dataclass(frozen=True)
class RenderMesh:
    positions: np.ndarray  # (N, 3) float32, read-only (shared with the scene's Mesh)
    triangles: np.ndarray  # (M, 3) uint32
    normals: np.ndarray | None  # (N, 3) float32 or None (flat shading from geometry)


@dataclass(frozen=True)
class RenderInstance:
    object_id: str
    name: str
    mesh: int  # index into RenderScene.meshes
    world: np.ndarray  # (4, 4) float64
    normal_matrix: np.ndarray  # (3, 3) float64: inverse transpose of the linear part
    mirrored: bool  # world determinant < 0 (winding appears reversed)
    base_color: np.ndarray  # (3,) linear light
    specular: float
    shininess: float


@dataclass(frozen=True)
class RenderLight:
    object_id: str | None  # None for the built-in default light
    name: str
    type: str  # "directional" | "point"
    radiance: np.ndarray  # (3,) linear color x intensity
    position: np.ndarray | None  # point lights
    direction: np.ndarray | None  # directional lights: unit direction the light travels
    range: float | None
    falloff: str


@dataclass(frozen=True)
class RenderCamera:
    object_id: str | None  # None for a temporary (auto-framed) camera
    name: str
    camera: Camera
    frame: CameraFrame

    def to_dict(self, aspect: float | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.object_id,
            "name": self.name,
            **self.camera.to_dict(),
            **self.frame.to_dict(),
        }
        if aspect is not None:
            hx, hy = self.camera.half_extents(aspect)
            if self.camera.is_perspective:
                data["horizontal_fov"] = math.degrees(2 * math.atan(hx))
                data["vertical_fov"] = math.degrees(2 * math.atan(hy))
            else:
                data["view_width"], data["view_height"] = 2 * hx, 2 * hy
        return data


@dataclass(frozen=True)
class RenderScene:
    meshes: tuple[RenderMesh, ...]
    instances: tuple[RenderInstance, ...]
    lights: tuple[RenderLight, ...]
    camera: RenderCamera
    background: np.ndarray  # (3,) linear
    ambient: np.ndarray  # (3,) linear ambient color x intensity
    diagnostics: tuple[Diagnostic, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def triangle_count(self) -> int:
        """Triangles to render, counting every instance."""
        return sum(int(self.meshes[i.mesh].triangles.shape[0]) for i in self.instances)


def _mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4x4 product with an explicit, fixed summation order (bitwise reproducible)."""
    out = np.empty((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = (
                a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j] + a[i, 3] * b[3, j]
            )
    return out


def world_matrices(scene: Scene) -> dict[str, np.ndarray]:
    """World matrix of every object, computed once per object in hierarchy order."""
    result: dict[str, np.ndarray] = {}
    for obj in scene:  # depth-first: parents first
        local = obj.local_matrix
        parent = obj.parent
        result[obj.id] = local if parent is None else _mat4_mul(result[parent.id], local)
    return result


def normal_matrix(linear: np.ndarray) -> np.ndarray | None:
    """Inverse transpose of a 3x3 matrix (``None`` if singular).

    Normals transform with ``M^-T`` -- including under mirroring, where it keeps
    outward normals outward (the cofactor matrix alone would flip them).
    """
    det = float(np.linalg.det(linear))
    if not math.isfinite(det) or abs(det) < 1e-300:
        return None
    try:
        inv = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        return None
    result = inv.T
    return result if np.all(np.isfinite(result)) else None


def resolve_camera(scene: Scene, camera: Any) -> SceneObject:
    """The camera object to render with: ``camera`` (object, id or name) or the active camera."""
    if camera is None:
        active = scene.active_camera
        if active is None:
            cameras = [o.name for o in scene if o.camera is not None]
            raise RenderError(
                "NO_CAMERA",
                "the scene has no active camera"
                + (f" (cameras: {', '.join(cameras)})" if cameras else " and no cameras"),
                hint="add one with scene.add_camera(...) (it becomes active), choose one with "
                "scene.active_camera = ..., or render with frame=True for a temporary camera",
            )
        return active
    obj = _find_object(scene, camera)
    if obj.camera is None:
        raise RenderError(
            "NOT_A_CAMERA", f"object {obj.name!r} is not a camera", f"objects[{obj.name}]"
        )
    return obj


def _find_object(scene: Scene, ref: Any) -> SceneObject:
    from ..scene import SceneObject

    if isinstance(ref, SceneObject):
        if ref.scene is not scene:
            raise RenderError("OBJECT_NOT_FOUND", f"{ref!r} does not belong to this scene")
        return ref
    if isinstance(ref, str):
        for getter in (scene.get, scene.get_by_name):
            try:
                return getter(ref)
            except AetherError:
                continue
    raise RenderError("OBJECT_NOT_FOUND", f"no object with id or name {ref!r}")


def extract_render_scene(
    scene: Scene,
    camera: SceneObject | RenderCamera,
    *,
    default_lights: bool = True,
    aspect: float = 16 / 9,
) -> RenderScene:
    """Collect everything needed to render ``scene`` from ``camera`` (read-only)."""
    start = time.perf_counter()
    diagnostics: list[Diagnostic] = []
    worlds = world_matrices(scene)
    if isinstance(camera, RenderCamera):
        render_camera = camera
    else:
        frame, frame_diags = frame_from_matrix(worlds[camera.id], f"objects[{camera.name}]")
        diagnostics += frame_diags
        render_camera = RenderCamera(camera.id, camera.name, camera.camera, frame)
        if camera.camera.aspect_ratio is not None and not math.isclose(
            camera.camera.aspect_ratio, aspect, rel_tol=1e-3
        ):
            diagnostics.append(
                Diagnostic(
                    "ASPECT_MISMATCH",
                    INFO,
                    f"the image aspect {aspect:.4g} differs from the camera's aspect_ratio "
                    f"{camera.camera.aspect_ratio:.4g}; the {camera.camera.fov_axis} field of "
                    "view is kept and the other extent adapts (pixels stay square)",
                    f"objects[{camera.name}].camera.aspect_ratio",
                )
            )

    meshes: list[RenderMesh] = []
    mesh_index: dict[int, int] = {}
    instances: list[RenderInstance] = []
    lights: list[RenderLight] = []
    evaluate_time = 0.0
    for obj in scene:
        loc = f"objects[{obj.name}]"
        world = worlds[obj.id]
        if obj.light is not None:
            if not obj.visible:
                continue
            light = obj.light
            radiance = to_linear(light.color) * light.intensity
            if not np.all(np.isfinite(world)):
                raise RenderError("NONFINITE_TRANSFORM", "world matrix is not finite", loc)
            if light.type == "point":
                lights.append(
                    RenderLight(
                        obj.id,
                        obj.name,
                        "point",
                        radiance,
                        world[:3, 3].copy(),
                        None,
                        light.range,
                        light.falloff,
                    )
                )
            else:
                direction = -world[:3, 1]
                length = float(np.linalg.norm(direction))
                if not length > 1e-12:
                    diagnostics.append(
                        Diagnostic(
                            "LIGHT_DIRECTION_UNDEFINED",
                            WARNING,
                            f"light {obj.name!r} has zero scale along its forward axis; "
                            "it is ignored",
                            loc,
                        )
                    )
                    continue
                lights.append(
                    RenderLight(
                        obj.id,
                        obj.name,
                        "directional",
                        radiance,
                        None,
                        direction / length,
                        None,
                        "none",
                    )
                )
            continue
        if obj.camera is not None or not obj.visible:
            continue
        if obj._mesh is None and obj._geometry is None:  # (no evaluation here)
            continue
        t0 = time.perf_counter()
        try:
            mesh = obj.mesh  # procedural objects: the current evaluation (cached)
        except AetherError as exc:
            code = getattr(exc, "code", type(exc).__name__)
            raise RenderError(
                "GEOMETRY_EVALUATION_FAILED",
                f"the geometry of {obj.name!r} could not be evaluated: [{code}] "
                + getattr(exc, "message", str(exc)),
                loc,
                hint="fix or disable the failing procedural definition, or hide the object "
                "(obj.visible = False)",
                details={
                    "error": exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}
                },
            ) from None
        evaluate_time += time.perf_counter() - t0
        if mesh is None or mesh.triangle_count == 0:
            continue
        if not np.all(np.isfinite(world)):
            raise RenderError(
                "NONFINITE_TRANSFORM",
                f"the world matrix of {obj.name!r} is not finite",
                loc,
                hint="check for extreme scales in the object's hierarchy",
            )
        linear = world[:3, :3]
        nmat = normal_matrix(linear)
        if nmat is None:
            diagnostics.append(
                Diagnostic(
                    "ZERO_WORLD_SCALE",
                    WARNING,
                    f"object {obj.name!r} has a singular world transform (zero scale); "
                    "it collapses and is not rendered",
                    loc,
                )
            )
            continue
        key = id(mesh)
        if key not in mesh_index:
            mesh_index[key] = len(meshes)
            meshes.append(_render_mesh(mesh))
        appearance: Appearance = obj.appearance
        instances.append(
            RenderInstance(
                obj.id,
                obj.name,
                mesh_index[key],
                world,
                nmat,
                bool(np.linalg.det(linear) < 0.0),
                to_linear(appearance.color),
                appearance.specular,
                appearance.shininess,
            )
        )

    if not lights and default_lights:
        lights.append(default_light(render_camera.frame))
        diagnostics.append(
            Diagnostic(
                "DEFAULT_LIGHTING",
                INFO,
                "the scene has no lights; a built-in key light from the camera's upper left "
                "was used (set default_lights=False to disable)",
            )
        )
    elif not lights:
        diagnostics.append(
            Diagnostic(
                "NO_LIGHTS",
                WARNING,
                "the scene has no lights and default lights are off; only ambient light is used",
            )
        )
    if not instances:
        diagnostics.append(
            Diagnostic(
                "NOTHING_TO_RENDER",
                WARNING,
                "no visible mesh objects; the image shows only the background",
            )
        )
    world_settings = scene.world
    total = time.perf_counter() - start
    return RenderScene(
        meshes=tuple(meshes),
        instances=tuple(instances),
        lights=tuple(lights),
        camera=render_camera,
        background=to_linear(world_settings.background),
        ambient=to_linear(world_settings.ambient_color) * world_settings.ambient_intensity,
        diagnostics=tuple(diagnostics),
        timings={"evaluate": evaluate_time, "extract": total - evaluate_time},
    )


def default_light(frame: CameraFrame) -> RenderLight:
    """The built-in key light: white, intensity 1, from the camera's upper left and behind it.

    Its direction (the way the light travels) is
    ``normalize(0.5 * right - 0.6 * up + 0.6 * forward)`` in the camera frame.
    """
    direction = 0.5 * frame.right - 0.6 * frame.up + 0.6 * frame.forward
    direction = direction / np.linalg.norm(direction)
    return RenderLight(
        None, "default light", "directional", np.ones(3), None, direction, None, "none"
    )


def _render_mesh(mesh: Mesh) -> RenderMesh:
    return RenderMesh(mesh.positions, mesh.triangles, mesh.normals)
