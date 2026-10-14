"""Render-scene extraction: everything a renderer backend needs, and nothing else.

:func:`extract_render_scene` walks a :class:`~aether3d.Scene` once and returns
an immutable :class:`RenderScene`:

* unique meshes (``RenderMesh``) -- the scene's own read-only arrays, shared
  by every instance that uses them (procedural objects contribute their
  *current evaluated* mesh);
* instances (``RenderInstance``) -- one per visible mesh object, with its
  world matrix, normal matrix, linear-light appearance and the materials of
  its face groups;
* materials (``RenderMaterial``, :mod:`aether3d.rendering.materials`) --
  linear factors and resolved textures (images already decoded);
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
from .materials import RenderMaterial, resolve_materials
from .settings import Appearance, Environment

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject


@dataclass(frozen=True)
class RenderMesh:
    positions: np.ndarray  # (N, 3) float32, read-only (shared with the scene's Mesh)
    triangles: np.ndarray  # (M, 3) uint32
    normals: np.ndarray | None  # (N, 3) float32 or None (flat shading from geometry)
    uvs: np.ndarray | None = None  # (N, 2) float32 or None
    groups: np.ndarray | None = None  # (M,) uint16 face groups or None (all group 0)


@dataclass(frozen=True)
class RenderInstance:
    object_id: str
    name: str
    mesh: int  # index into RenderScene.meshes
    world: np.ndarray  # (4, 4) float64
    normal_matrix: np.ndarray  # (3, 3) float64: inverse transpose of the linear part
    mirrored: bool  # world determinant < 0 (winding appears reversed)
    base_color: np.ndarray  # (3,) linear light (the object's own color)
    specular: float
    shininess: float
    material: int = 0  # index into RenderScene.materials for faces without a group material
    group_materials: tuple[tuple[int, int], ...] = ()  # (face group, material index), sorted

    def material_of_groups(self, groups: np.ndarray | None, count: int) -> np.ndarray:
        """Material index of each of ``count`` triangles with face groups ``groups``."""
        if not self.group_materials or groups is None:
            return np.full(count, self.material, dtype=np.int32)
        table = np.full(int(groups.max(initial=0)) + 1, self.material, dtype=np.int32)
        for group, index in self.group_materials:
            if group < table.shape[0]:
                table[group] = index
        return table[groups]


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
    radius: float = 0.0  # point lights: emitting sphere (path tracer)
    angle: float = 0.0  # directional lights: angular diameter in degrees (path tracer)


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
    materials: tuple[RenderMaterial, ...] = ()
    environment: Environment | None = None  # scene.world.environment (path tracer)

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


def _mat4_mul_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:func:`_mat4_mul` for stacks of matrices: the same products, summed in the same order."""
    out = np.empty(a.shape, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = (
                a[:, i, 0] * b[:, 0, j]
                + a[:, i, 1] * b[:, 1, j]
                + a[:, i, 2] * b[:, 2, j]
                + a[:, i, 3] * b[:, 3, j]
            )
    return out


def _local_matrices(objects: list[SceneObject]) -> np.ndarray:
    """``(n, 4, 4)`` local matrices, bitwise equal to ``obj.local_matrix`` for each object.

    A transform stores its rotation already normalized by ``math3d.quat`` (every
    assignment goes through it, and normalizing again changes nothing), so this
    is :func:`~aether3d.math3d.compose_matrix` with the same arithmetic,
    evaluated for all objects at once.
    """
    n = len(objects)
    transforms = [obj._transform for obj in objects]
    position = np.array([t._position for t in transforms], dtype=np.float64).reshape(n, 3)
    rotation = np.array([t._rotation for t in transforms], dtype=np.float64).reshape(n, 4)
    scale = np.array([t._scale for t in transforms], dtype=np.float64).reshape(n, 3)
    x, y, z, w = rotation.T
    r = np.empty((n, 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    m = np.zeros((n, 4, 4), dtype=np.float64)
    m[:, 3, 3] = 1.0
    m[:, :3, :3] = r * scale[:, None, :]
    m[:, :3, 3] = position
    return m


def world_matrices(scene: Scene) -> dict[str, np.ndarray]:
    """World matrix of every object (``parent world @ local``, in hierarchy order).

    Computed for all objects at once, one hierarchy level at a time, with the
    same arithmetic as ``obj.local_matrix`` and :func:`_mat4_mul`.
    """
    objects = list(scene)  # depth-first: parents first
    if not objects:
        return {}
    world = _local_matrices(objects)
    index = {obj.id: k for k, obj in enumerate(objects)}
    parent = np.full(len(objects), -1, dtype=np.int64)
    depth = np.zeros(len(objects), dtype=np.int64)
    for k, obj in enumerate(objects):
        if obj.parent is not None:
            parent[k] = index[obj.parent.id]
            depth[k] = depth[parent[k]] + 1
    for level in range(1, int(depth.max()) + 1):
        rows = np.flatnonzero(depth == level)
        world[rows] = _mat4_mul_many(world[parent[rows]], world[rows])
    return {obj.id: world[k] for k, obj in enumerate(objects)}


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


def normal_matrices(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`normal_matrix` for a stack of 3x3 matrices, plus their determinants.

    Returns ``(normal, valid, det)``: ``normal[k]`` is meaningful where
    ``valid[k]``. Each matrix goes through the same LAPACK routines as with
    :func:`normal_matrix`, so the results are identical.
    """
    n = linear.shape[0]
    det = np.linalg.det(linear) if n else np.zeros(0)
    valid = np.isfinite(det) & (np.abs(det) >= 1e-300)
    normal = np.zeros((n, 3, 3), dtype=np.float64)
    if valid.any():
        try:
            inverse = np.linalg.inv(linear[valid])
        except np.linalg.LinAlgError:  # pragma: no cover - singular despite its determinant
            inverse = np.stack(
                [
                    m if (m := normal_matrix(lin)) is not None else np.full((3, 3), np.nan)
                    for lin in linear[valid]
                ]
            ).transpose(0, 2, 1)
        normal[valid] = inverse.transpose(0, 2, 1)
    valid &= np.all(np.isfinite(normal), axis=(1, 2))
    return normal, valid, det


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
    lights: list[RenderLight] = []
    candidates: list[tuple[SceneObject, Mesh, np.ndarray, int]] = []  # mesh objects, in order
    ordered: list[tuple[int, Diagnostic]] = []  # loop diagnostics, by object position
    evaluate_time = 0.0
    for position, obj in enumerate(scene):
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
                        radius=light.radius,
                    )
                )
            else:
                direction = -world[:3, 1]
                length = float(np.linalg.norm(direction))
                if not length > 1e-12:
                    ordered.append(
                        (
                            position,
                            Diagnostic(
                                "LIGHT_DIRECTION_UNDEFINED",
                                WARNING,
                                f"light {obj.name!r} has zero scale along its forward axis; "
                                "it is ignored",
                                loc,
                            ),
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
                        angle=light.angle,
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
        candidates.append((obj, mesh, world, position))

    # Normal matrices, determinants and colors of all mesh objects at once.
    linear = np.array([c[2][:3, :3] for c in candidates], dtype=np.float64).reshape(-1, 3, 3)
    normal, valid, det = normal_matrices(linear)
    material_table, object_materials = resolve_materials(
        scene,
        [
            (obj, mesh, position)
            for k, (obj, mesh, _, position) in enumerate(candidates)
            if valid[k]
        ],
    )
    ordered += material_table.diagnostics
    material_iter = iter(object_materials)
    colors: dict[tuple, np.ndarray] = {}
    instances: list[RenderInstance] = []
    for k, (obj, mesh, world, position) in enumerate(candidates):
        if not valid[k]:
            ordered.append(
                (
                    position,
                    Diagnostic(
                        "ZERO_WORLD_SCALE",
                        WARNING,
                        f"object {obj.name!r} has a singular world transform (zero scale); "
                        "it collapses and is not rendered",
                        f"objects[{obj.name}]",
                    ),
                )
            )
            continue
        key = id(mesh)
        if key not in mesh_index:
            mesh_index[key] = len(meshes)
            meshes.append(_render_mesh(mesh))
        appearance: Appearance = obj.appearance
        base = colors.get(appearance.color)
        if base is None:
            base = colors[appearance.color] = to_linear(appearance.color)
        default_material, group_materials = next(material_iter)
        instances.append(
            RenderInstance(
                obj.id,
                obj.name,
                mesh_index[key],
                world,
                normal[k],
                bool(det[k] < 0.0),
                base.copy(),
                appearance.specular,
                appearance.shininess,
                default_material,
                tuple(sorted(group_materials.items())),
            )
        )
    diagnostics += [d for _, d in sorted(ordered, key=lambda item: item[0])]

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
        timings={
            "evaluate": evaluate_time,
            "textures": material_table.texture_seconds,
            "extract": total - evaluate_time - material_table.texture_seconds,
        },
        materials=tuple(material_table.materials),
        environment=world_settings.environment.copy(),
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
    return RenderMesh(mesh.positions, mesh.triangles, mesh.normals, mesh.uvs, mesh.groups)
