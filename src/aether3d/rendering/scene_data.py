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

import itertools
import math
import operator
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import _features, _native
from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import AetherError, RenderError
from ..mesh import Mesh
from .camera import Camera, CameraFrame, frame_from_matrix
from .color import to_linear
from .materials import MaterialCache, RenderMaterial, resolve_materials
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
    colors: np.ndarray | None = None  # (N, 4) float32 linear vertex colours or None


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


MAX_LINKED_LIGHTS = 64
"""Light linking on the GPU backends covers the first 64 lights (two words per instance)."""


@dataclass(frozen=True)
class RenderLight:
    object_id: str | None  # None for the built-in default light
    name: str
    type: str  # "directional" | "point" | "spot" | "area"
    radiance: np.ndarray  # (3,) linear color x intensity (Aether3D units, temperature, exposure)
    position: np.ndarray | None  # point, spot and area lights (an area light's centre)
    direction: np.ndarray | None  # unit direction the light travels (directional, spot, area)
    range: float | None
    falloff: str
    radius: float = 0.0  # point and spot lights: emitting sphere (path tracer)
    angle: float = 0.0  # directional lights: angular diameter in degrees (path tracer)
    cos_outer: float = -1.0  # spot: cosine of the cone's half angle
    spot_smooth: float = 0.0  # spot: (1 - cos_outer) * blend, the smoothstep's width in cos
    axis_x: np.ndarray | None = None  # area: half extent along the light's local X (world)
    axis_y: np.ndarray | None = None  # area: half extent along its local Z (world)
    shape: str = "square"  # area: square | rectangle | disk | ellipse
    area: float = 0.0  # area: emitting area (m^2, world scale)
    shadow: bool = True  # occluded by geometry (False: shines through)
    spread_tan: float = 0.0  # area: tan(spread / 2), 0 for 180 degrees (Lambertian)
    spread_norm: float = 1.0  # area: 1 / (1 - h / tan h), which keeps the power
    # Light linking: per instance of the render scene, whether this light lights it
    # (None: every instance). Set by the extraction from the light's include / exclude.
    receivers: np.ndarray | None = field(default=None, compare=False)
    links: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())  # (include, exclude) ids

    @property
    def area_radiance(self) -> np.ndarray:
        """An area light's emitted radiance ``pi c I / A`` (``(3,)``)."""
        return math.pi * self.radiance / max(self.area, 1e-300)

    def spot_mask(self, cos_t: np.ndarray) -> np.ndarray:
        """A spot light's mask at the cosines between its axis and the directions from it:
        1 inside, a smoothstep of the cosine over ``spot_smooth`` at the edge, 0 outside."""
        inside = cos_t > self.cos_outer
        if self.spot_smooth > 0.0:
            x = np.clip((cos_t - self.cos_outer) / self.spot_smooth, 0.0, 1.0)
            return np.where(inside, x * x * (3.0 - 2.0 * x), 0.0)
        return np.where(inside, 1.0, 0.0)

    def spread_factor(self, cos_a: np.ndarray) -> np.ndarray:
        """A spread area light's radiance factor toward directions at ``cos_a`` from its
        normal (1 everywhere for a spread of 180 degrees)."""
        if self.spread_tan <= 0.0:
            return np.where(cos_a > 0.0, 1.0, 0.0)
        c = np.maximum(cos_a, 1e-300)
        tan_a = np.sqrt(np.maximum(1.0 - cos_a * cos_a, 0.0)) / c
        g = np.maximum(1.0 - tan_a / self.spread_tan, 0.0) * self.spread_norm
        return np.where(cos_a > 0.0, g, 0.0)

    def spread_window(self, p: np.ndarray):
        """The part of a spread area light that reaches points ``p`` (``(m, 3)``): the
        rectangle ``[u0, u1] x [w0, w1]`` of its local coordinates (half extents: the
        shape spans ``[-1, 1]^2``) around the points' feet, ``t tan(spread / 2)`` wide (``t``
        their height in front of it), its area in square metres, and where it is not
        empty."""
        rel = p - self.position
        t = rel @ self.direction
        lx = float(np.sqrt(self.axis_x @ self.axis_x))
        ly = float(np.sqrt(self.axis_y @ self.axis_y))
        qu = (rel @ self.axis_x) / (lx * lx)
        qw = (rel @ self.axis_y) / (ly * ly)
        reach = np.maximum(t, 0.0) * self.spread_tan
        u0 = np.maximum(qu - reach / lx, -1.0)
        u1 = np.minimum(qu + reach / lx, 1.0)
        w0 = np.maximum(qw - reach / ly, -1.0)
        w1 = np.minimum(qw + reach / ly, 1.0)
        ok = (t > 0.0) & (u1 > u0) & (w1 > w0)
        area = np.where(ok, (u1 - u0) * lx * (w1 - w0) * ly, 0.0)
        return u0, u1, w0, w1, area, ok


@dataclass(frozen=True)
class RenderCamera:
    object_id: str | None  # None for a temporary (auto-framed) camera
    name: str
    camera: Camera
    frame: CameraFrame
    focus: float | None = None  # the focus distance of this render (its focus_object's)

    @property
    def focus_distance(self) -> float:
        """Metres along the view axis to the plane in focus for this render."""
        return self.camera.focus_distance if self.focus is None else self.focus

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
class InstanceArrays:
    """The fields of every instance as arrays: row ``k`` holds ``RenderScene.instances[k]``'s
    values, exactly. Backends pack these instead of visiting instances one by one."""

    world: np.ndarray  # (n, 4, 4) float64
    normal_matrix: np.ndarray  # (n, 3, 3) float64
    mirrored: np.ndarray  # (n,) bool
    mesh: np.ndarray  # (n,) int64
    base_color: np.ndarray  # (n, 3) float64
    specular: np.ndarray  # (n,) float64
    shininess: np.ndarray  # (n,) float64
    material: np.ndarray  # (n,) int64
    has_groups: np.ndarray  # (n,) bool: the instance has face-group materials

    @classmethod
    def gather(cls, instances: tuple[RenderInstance, ...]) -> InstanceArrays:
        """The arrays of ``instances`` (render scenes built without them)."""
        n = len(instances)
        return cls(
            world=np.array([i.world for i in instances], dtype=np.float64).reshape(n, 4, 4),
            normal_matrix=np.array([i.normal_matrix for i in instances], dtype=np.float64).reshape(
                n, 3, 3
            ),
            mirrored=np.fromiter((i.mirrored for i in instances), dtype=bool, count=n),
            mesh=np.fromiter((i.mesh for i in instances), dtype=np.int64, count=n),
            base_color=np.array([i.base_color for i in instances], dtype=np.float64).reshape(n, 3),
            specular=np.fromiter((i.specular for i in instances), dtype=np.float64, count=n),
            shininess=np.fromiter((i.shininess for i in instances), dtype=np.float64, count=n),
            material=np.fromiter((i.material for i in instances), dtype=np.int64, count=n),
            has_groups=np.fromiter((bool(i.group_materials) for i in instances), bool, count=n),
        )


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
    environment_image: Any = None  # its EnvironmentImage (type "image"), loaded
    ambient_sh: np.ndarray | None = None  # rasterizers: ambient harmonics of an image
    atmosphere: dict | None = None  # {"settings": scene.world.atmosphere, "ambient_intensity"}
    arrays: InstanceArrays | None = field(default=None, repr=False, compare=False)
    cache: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def shared_cache(self) -> dict[str, Any] | None:
        """A dict where backends may keep work between frames, or ``None``.

        The retained render world gives the render scenes it returns one dict, shared by
        later frames while their instances, meshes and materials are the same objects. Keep
        in it only results that depend on those alone, or key them by the other inputs
        (the camera, settings). Other render scenes -- from the full extraction, or derived
        with :func:`dataclasses.replace` -- get ``None``.
        """
        cache = self.cache
        if (
            cache is None
            or cache.get("instances") is not self.instances
            or cache.get("meshes") is not self.meshes
            or cache.get("materials") is not self.materials
        ):
            return None
        return cache

    @property
    def instance_arrays(self) -> InstanceArrays:
        """The instances' fields as arrays (:class:`InstanceArrays`)."""
        if self.arrays is not None:
            return self.arrays
        arrays = InstanceArrays.gather(self.instances)
        object.__setattr__(self, "arrays", arrays)
        return arrays

    @property
    def triangle_count(self) -> int:
        """Triangles to render, counting every instance."""
        if not self.instances:
            return 0
        per_mesh = np.array([m.triangles.shape[0] for m in self.meshes], dtype=np.int64)
        return int(per_mesh[self.instance_arrays.mesh].sum())


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


_TRANSFORM = operator.attrgetter("_transform")
_POSITION = operator.attrgetter("_position")
_ROTATION = operator.attrgetter("_rotation")
_SCALE = operator.attrgetter("_scale")
_chain = itertools.chain.from_iterable


def _local_matrices(objects: list[SceneObject]) -> np.ndarray:
    """``(n, 4, 4)`` local matrices, bitwise equal to ``obj.local_matrix`` for each object.

    A transform stores its rotation already normalized by ``math3d.quat`` (every
    assignment goes through it, and normalizing again changes nothing), so this
    is :func:`~aether3d.math3d.compose_matrix` with the same arithmetic,
    evaluated for all objects at once.
    """
    n = len(objects)
    transforms = list(map(_TRANSFORM, objects))
    position = np.fromiter(_chain(map(_POSITION, transforms)), np.float64, 3 * n).reshape(n, 3)
    rotation = np.fromiter(_chain(map(_ROTATION, transforms)), np.float64, 4 * n).reshape(n, 4)
    scale = np.fromiter(_chain(map(_SCALE, transforms)), np.float64, 3 * n).reshape(n, 3)
    return _compose_locals(position, rotation, scale)


def _compose_locals(position: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(n, 4, 4)`` ``T @ R @ S`` matrices of TRS arrays with the arithmetic of
    ``obj.local_matrix`` (rotations already unit: as transforms store them)."""
    n = position.shape[0]
    position = np.ascontiguousarray(position, dtype=np.float64)
    rotation = np.ascontiguousarray(rotation, dtype=np.float64)
    scale = np.ascontiguousarray(scale, dtype=np.float64)
    native = _native.module()
    if native is not None:  # the same arithmetic (aether3d-native)
        m = np.empty((n, 4, 4), dtype=np.float64)
        native.local_matrices(position, rotation, scale, m)
        return m
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


def world_matrices(scene: Scene, overlay: Any = None) -> dict[str, np.ndarray]:
    """World matrix of every object (``parent world @ local``, in hierarchy order).

    Computed for all objects at once, one hierarchy level at a time, with the
    same arithmetic as ``obj.local_matrix`` and :func:`_mat4_mul`. With an ``overlay``
    (:class:`~aether3d.rendering.overlay.RenderOverlay`), its local matrices replace
    the authored ones of the objects it names (an animated pose); without, these are the
    authored world matrices.
    """
    if overlay is not None and overlay.ids:
        objects, world = _world_stack(scene, overlay)
        return {obj.id: world[k] for k, obj in enumerate(objects)}
    if _features.enabled("AETHER3D_RETAINED_RENDER_WORLD"):  # a rendered scene keeps them
        from .retained import retained_world_matrices

        found = retained_world_matrices(scene)
        if found is not None:
            return found
    objects, world = _world_stack(scene)
    return {obj.id: world[k] for k, obj in enumerate(objects)}


def _world_stack(scene: Scene, overlay: Any = None) -> tuple[list[SceneObject], np.ndarray]:
    """The objects in depth-first order and their world matrices, ``(n, 4, 4)`` (an
    ``overlay``'s local matrices replace the authored ones of the objects it names)."""
    objects = list(scene)  # depth-first: parents first
    if not objects:
        return objects, np.zeros((0, 4, 4))
    world = _local_matrices(objects)
    if overlay is not None and overlay.ids:
        index = overlay.index
        rows = [k for k, obj in enumerate(objects) if obj._id in index]
        if rows:
            world[rows] = overlay.local[[index[objects[k]._id] for k in rows]]
    index = {obj.id: k for k, obj in enumerate(objects)}
    parent = np.full(len(objects), -1, dtype=np.int64)
    depth = np.zeros(len(objects), dtype=np.int64)
    for k, obj in enumerate(objects):
        if obj.parent is not None:
            parent[k] = index[obj.parent.id]
            depth[k] = depth[parent[k]] + 1
    return objects, _world_from_local(world, parent, depth)


def _world_from_local(local: np.ndarray, parent: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """World matrices of objects in depth-first order (parents first) from their local
    matrices: ``world[parent] @ local`` with :func:`_mat4_mul_many`'s sums, one hierarchy
    level at a time (or in one native pass, with the same arithmetic). ``local`` may be
    returned, updated in place."""
    native = _native.module()
    if native is not None and len(local):
        world = np.empty_like(local)
        native.world_matrices(np.ascontiguousarray(local), np.ascontiguousarray(parent), world)
        return world
    world = local
    for level in range(1, int(depth.max(initial=0)) + 1):
        rows = np.flatnonzero(depth == level)
        world[rows] = _mat4_mul_many(world[parent[rows]], world[rows])
    return world


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
    overlay: Any = None,
) -> RenderScene:
    """Collect everything needed to render ``scene`` from ``camera`` (read-only).

    Since 0.11.1 the scene's retained render world (:mod:`.retained`) brings what an
    earlier extraction computed up to date with the changes made since, and returns
    exactly the render scene the full walk below returns; the full walk runs when it
    cannot (inside an open operation, with ``AETHER3D_RETAINED_RENDER_WORLD=0``, ...).
    """
    if overlay is not None and overlay.empty:
        overlay = None
    if _features.enabled("AETHER3D_RETAINED_RENDER_WORLD"):
        from .retained import retained_render_scene

        rs = retained_render_scene(
            scene, camera, default_lights=default_lights, aspect=aspect, overlay=overlay
        )
        if rs is not None:
            return rs
    return _extract_full(
        scene, camera, default_lights=default_lights, aspect=aspect, overlay=overlay
    )


def _render_camera(
    camera: SceneObject,
    world: np.ndarray,
    aspect: float,
    overlay: Any = None,
    world_of: Any = None,
) -> tuple[RenderCamera, list[Diagnostic]]:
    """The render camera of the camera object ``camera`` with world matrix ``world`` (an
    ``overlay``'s camera fields replace its own: an animated lens). ``world_of`` gives an
    object's world matrix by id (the focus object's)."""
    frame, frame_diags = frame_from_matrix(world, f"objects[{camera.name}]")
    diagnostics = list(frame_diags)
    lens = camera.camera
    values = None if overlay is None else overlay.field_values(camera.id, "camera")
    if values:
        lens = _with_fields(lens, values)
    focus = None
    if lens.has_depth_of_field and lens.focus_object is not None:
        m = None if world_of is None else world_of(lens.focus_object)
        if m is not None:
            d = float(np.dot(np.asarray(m)[:3, 3] - frame.position, frame.forward))
            if d > 0.0:
                focus = d
        if focus is None:
            diagnostics.append(
                Diagnostic(
                    "FOCUS_OBJECT_UNUSABLE",
                    INFO,
                    f"the focus object of {camera.name!r} is missing or behind the camera; "
                    f"its focus_distance ({lens.focus_distance:g} m) is used",
                    f"objects[{camera.name}].camera.focus_object",
                )
            )
    render_camera = RenderCamera(camera.id, camera.name, lens, frame, focus)
    if lens.aspect_ratio is not None and not math.isclose(lens.aspect_ratio, aspect, rel_tol=1e-3):
        diagnostics.append(
            Diagnostic(
                "ASPECT_MISMATCH",
                INFO,
                f"the image aspect {aspect:.4g} differs from the camera's aspect_ratio "
                f"{lens.aspect_ratio:.4g}; the {lens.fov_axis} field of "
                "view is kept and the other extent adapts (pixels stay square)",
                f"objects[{camera.name}].camera.aspect_ratio",
            )
        )
    return render_camera, diagnostics


def _light(
    obj: SceneObject,
    world: np.ndarray,
    finite: bool,
    position: int,
    ordered: list[tuple[int, Diagnostic]],
    overlay: Any = None,
) -> RenderLight | None:
    """The render light of the visible light object ``obj`` (``None``: it is ignored, and
    ``ordered`` gets the diagnostic saying why); an ``overlay`` may animate its fields."""
    light = obj.light
    values = None if overlay is None else overlay.field_values(obj._id, "light")
    if values:
        light = _with_fields(light, values)
    if (
        light.type in ("directional", "point")
        and light.units == "aether"
        and not (light.use_temperature or light.exposure)
    ):
        radiance = to_linear(light.color) * light.intensity  # (as always: the same bits)
    else:
        radiance = np.array(light.linear_color()) * light.intensity_value()
    if not finite:
        raise RenderError(
            "NONFINITE_TRANSFORM", "world matrix is not finite", f"objects[{obj.name}]"
        )
    if light.type == "point":
        return RenderLight(
            obj.id,
            obj.name,
            "point",
            radiance,
            world[:3, 3].copy(),
            None,
            light.range,
            light.falloff,
            radius=light.radius,
            shadow=light.shadow,
            links=(light.include, light.exclude),
        )
    direction = -world[:3, 1]
    length = float(np.linalg.norm(direction))
    if not length > 1e-12:
        ordered.append(
            (
                position,
                Diagnostic(
                    "LIGHT_DIRECTION_UNDEFINED",
                    WARNING,
                    f"light {obj.name!r} has zero scale along its forward axis; it is ignored",
                    f"objects[{obj.name}]",
                ),
            )
        )
        return None
    unit = direction / length
    if light.type == "spot":
        cos_outer = math.cos(math.radians(light.spot_size) / 2.0)
        return RenderLight(
            obj.id,
            obj.name,
            "spot",
            radiance,
            world[:3, 3].copy(),
            unit,
            light.range,
            light.falloff,
            radius=light.radius,
            cos_outer=cos_outer,
            spot_smooth=(1.0 - cos_outer) * light.spot_blend,
            shadow=light.shadow,
            links=(light.include, light.exclude),
        )
    if light.type == "area":
        width = light.size
        height = light.size if light.shape in ("square", "disk") else light.size_y
        ax = world[:3, 0] * (width / 2.0)
        ay = world[:3, 2] * (height / 2.0)
        cross = float(np.linalg.norm(np.cross(ax, ay)))
        area = (math.pi if light.shape in ("disk", "ellipse") else 4.0) * cross
        if not area > 1e-18:
            ordered.append(
                (
                    position,
                    Diagnostic(
                        "LIGHT_SIZE_UNDEFINED",
                        WARNING,
                        f"area light {obj.name!r} has no area (zero scale); it is ignored",
                        f"objects[{obj.name}]",
                    ),
                )
            )
            return None
        # One-sided: it emits toward its front (local -Y), the side its normal faces.
        normal = np.cross(ay, ax)
        normal = normal / np.linalg.norm(normal)
        if float(normal @ unit) < 0.0:
            normal = -normal
        spread = {}
        if light.spread < 180.0:
            half = math.radians(light.spread) / 2.0
            spread = {
                "spread_tan": math.tan(half),
                "spread_norm": 1.0 / (1.0 - half / math.tan(half)),
            }
        return RenderLight(
            obj.id,
            obj.name,
            "area",
            radiance,
            world[:3, 3].copy(),
            normal,
            None,
            "inverse_square",
            axis_x=ax,
            axis_y=ay,
            shape=light.shape,
            area=area,
            shadow=light.shadow,
            links=(light.include, light.exclude),
            **spread,
        )
    return RenderLight(
        obj.id,
        obj.name,
        "directional",
        radiance,
        None,
        unit,
        None,
        "none",
        angle=light.angle,
        shadow=light.shadow,
        links=(light.include, light.exclude),
    )


def _overlay_looks(overlay: Any, candidates: list) -> dict[str, Appearance] | None:
    """Animated appearances of the candidates (object id -> detached Appearance)."""
    out = {}
    for obj, *_ in candidates:
        values = overlay.field_values(obj._id, "appearance")
        if values:
            out[obj._id] = _with_fields(obj.appearance, values)
    return out or None


def _with_fields(value: Any, values: Any) -> Any:
    """A detached copy of a camera, light or appearance with animated ``values`` (checked
    by its own setters; the scene's own value is untouched)."""
    copy = value.copy()
    copy._owner = None
    copy.configure(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()})
    return copy


def _evaluated_mesh(obj: SceneObject, overlay: Any = None) -> Mesh | None:
    """``obj.mesh`` -- for procedural objects, their current evaluation (cached); an
    ``overlay``'s deformed mesh when it has one."""
    if overlay is not None:
        found = overlay.meshes.get(obj._id)
        if found is not None:
            return found
    try:
        return obj.mesh
    except AetherError as exc:
        code = getattr(exc, "code", type(exc).__name__)
        raise RenderError(
            "GEOMETRY_EVALUATION_FAILED",
            f"the geometry of {obj.name!r} could not be evaluated: [{code}] "
            + getattr(exc, "message", str(exc)),
            f"objects[{obj.name}]",
            hint="fix or disable the failing procedural definition, or hide the object "
            "(obj.visible = False)",
            details={"error": exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}},
        ) from None


def _nonfinite_mesh_object(obj: SceneObject) -> RenderError:
    return RenderError(
        "NONFINITE_TRANSFORM",
        f"the world matrix of {obj.name!r} is not finite",
        f"objects[{obj.name}]",
        hint="check for extreme scales in the object's hierarchy",
    )


def _zero_scale(obj: SceneObject) -> Diagnostic:
    return Diagnostic(
        "ZERO_WORLD_SCALE",
        WARNING,
        f"object {obj.name!r} has a singular world transform (zero scale); "
        "it collapses and is not rendered",
        f"objects[{obj.name}]",
    )


def _extract_full(
    scene: Scene,
    camera: SceneObject | RenderCamera,
    *,
    default_lights: bool = True,
    aspect: float = 16 / 9,
    overlay: Any = None,
) -> RenderScene:
    """The full extraction: walks every object of the scene (the reference the retained
    render world is compared with, and what runs when it cannot). An ``overlay``
    (animation) replaces local transforms, visibility, fields, materials and meshes."""
    start = time.perf_counter()
    diagnostics: list[Diagnostic] = []
    objects, stack = _world_stack(scene, overlay)
    finite_world = np.all(np.isfinite(stack), axis=(1, 2)).tolist()
    if isinstance(camera, RenderCamera):
        render_camera = camera
    else:
        rows = {o._id: i for i, o in enumerate(objects)}
        render_camera, camera_diags = _render_camera(
            camera,
            stack[objects.index(camera)],
            aspect,
            overlay,
            lambda oid: stack[rows[oid]] if oid in rows else None,
        )
        diagnostics += camera_diags

    meshes: list[RenderMesh] = []
    mesh_index: dict[int, int] = {}
    lights: list[RenderLight] = []
    candidates: list[tuple[SceneObject, Mesh, np.ndarray, int]] = []  # mesh objects, in order
    ordered: list[tuple[int, Diagnostic]] = []  # loop diagnostics, by object position
    evaluate_time = 0.0
    for position, obj in enumerate(objects):
        world = stack[position]
        visible = obj.visible if overlay is None else overlay.is_visible(obj)
        if obj.light is not None:
            if not visible:
                continue
            light = _light(obj, world, finite_world[position], position, ordered, overlay)
            if light is not None:
                lights.append(light)
            continue
        if obj.camera is not None or not visible:
            continue
        if obj._mesh is None and obj._geometry is None:  # (no evaluation here)
            continue
        t0 = time.perf_counter()
        mesh = _evaluated_mesh(obj, overlay)  # procedural: the current evaluation (cached)
        evaluate_time += time.perf_counter() - t0
        if mesh is None or mesh.triangle_count == 0:
            continue
        if not finite_world[position]:
            raise _nonfinite_mesh_object(obj)
        candidates.append((obj, mesh, world, position))

    # Normal matrices, determinants and colors of all mesh objects at once.
    candidate_worlds = stack[[c[3] for c in candidates]].reshape(-1, 4, 4)
    linear = np.ascontiguousarray(candidate_worlds[:, :3, :3])
    normal, valid, det = normal_matrices(linear)
    material_table, object_materials = resolve_materials(
        scene,
        [
            (obj, mesh, position)
            for k, (obj, mesh, _, position) in enumerate(candidates)
            if valid[k]
        ],
        _material_cache(scene) if overlay is None or not overlay.materials else None,
        None if overlay is None else overlay.materials,
        None if overlay is None else _overlay_looks(overlay, candidates),
    )
    ordered += material_table.diagnostics
    material_iter = iter(object_materials)
    colors: dict[tuple, np.ndarray] = {}
    instances: list[RenderInstance] = []
    columns: tuple[list, list, list, list, list, list] = ([], [], [], [], [], [])
    for k, (obj, mesh, world, position) in enumerate(candidates):
        if not valid[k]:
            ordered.append((position, _zero_scale(obj)))
            continue
        key = id(mesh)
        if key not in mesh_index:
            mesh_index[key] = len(meshes)
            meshes.append(_render_mesh(mesh))
        appearance: Appearance = obj.appearance
        if overlay is not None:
            values = overlay.field_values(obj._id, "appearance")
            if values:
                appearance = _with_fields(appearance, values)
        base = colors.get(appearance.color)
        if base is None:
            base = colors[appearance.color] = to_linear(appearance.color)
        default_material, group_materials = next(material_iter)
        instance = RenderInstance(
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
        instances.append(instance)
        for column, value in zip(
            columns,
            (
                k,
                instance.mesh,
                instance.base_color,
                instance.specular,
                instance.shininess,
                instance.material,
            ),
            strict=True,
        ):
            column.append(value)
    rows, mesh_of, base_colors, speculars, shininesses, defaults = columns
    count = len(rows)
    arrays = InstanceArrays(
        world=candidate_worlds[rows].reshape(count, 4, 4),
        normal_matrix=normal[rows].reshape(count, 3, 3),
        mirrored=(det[rows] < 0.0).reshape(count),
        mesh=np.array(mesh_of, dtype=np.int64),
        base_color=np.array(base_colors, dtype=np.float64).reshape(count, 3),
        specular=np.array(speculars, dtype=np.float64),
        shininess=np.array(shininesses, dtype=np.float64),
        material=np.array(defaults, dtype=np.int64),
        has_groups=np.array([bool(i.group_materials) for i in instances], dtype=bool),
    )
    return _finish(
        scene,
        meshes=tuple(meshes),
        instances=tuple(instances),
        lights=lights,
        camera=render_camera,
        diagnostics=diagnostics,
        ordered=ordered,
        materials=tuple(material_table.materials),
        arrays=arrays,
        default_lights=default_lights,
        evaluate_time=evaluate_time,
        texture_time=material_table.texture_seconds,
        start=start,
    )


def _finish(
    scene: Scene,
    *,
    meshes: tuple[RenderMesh, ...],
    instances: tuple[RenderInstance, ...],
    lights: list[RenderLight],
    camera: RenderCamera,
    diagnostics: list[Diagnostic],
    ordered: list[tuple[int, Diagnostic]],
    materials: tuple[RenderMaterial, ...],
    arrays: InstanceArrays,
    default_lights: bool,
    evaluate_time: float,
    texture_time: float,
    start: float,
    cache: dict[str, Any] | None = None,
) -> RenderScene:
    """The render scene: diagnostics in object order, then the scene-wide ones."""
    diagnostics += [d for _, d in sorted(ordered, key=lambda item: item[0])]
    if any(light.links != ((), ()) for light in lights):
        lights = _link_lights(scene, lights, instances, diagnostics)
    lit_by_image = scene.world.environment.type == "image"  # (no built-in key light then)
    if not lights and default_lights and not lit_by_image:
        lights.append(default_light(camera.frame))
        diagnostics.append(
            Diagnostic(
                "DEFAULT_LIGHTING",
                INFO,
                "the scene has no lights; a built-in key light from the camera's upper left "
                "was used (set default_lights=False to disable)",
            )
        )
    elif not lights and not lit_by_image:
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
    environment = world_settings.environment.copy()
    environment_image = None
    if environment.type == "image":
        from .envimage import load

        try:
            environment_image = load(scene, environment)
        except RenderError as exc:  # a missing or broken image: render without it, and say so
            diagnostics.append(
                Diagnostic(
                    exc.code,
                    WARNING,
                    f"{exc.message}; the environment is the ambient light instead",
                    "world.environment.image",
                )
            )
            environment = environment._clone()
            object.__setattr__(environment, "_type", "ambient")
    total = time.perf_counter() - start
    return RenderScene(
        meshes=meshes,
        instances=instances,
        lights=tuple(lights),
        camera=camera,
        background=to_linear(world_settings.background),
        ambient=to_linear(world_settings.ambient_color) * world_settings.ambient_intensity,
        diagnostics=tuple(diagnostics),
        timings={
            "evaluate": evaluate_time,
            "textures": texture_time,
            "extract": total - evaluate_time - texture_time,
        },
        materials=materials,
        environment=environment,
        environment_image=environment_image,
        atmosphere={
            "settings": world_settings.atmosphere.copy(),
            "ambient_intensity": world_settings.ambient_intensity,
        },
        arrays=arrays,
        cache=cache,
    )


def _link_lights(
    scene: Scene,
    lights: list[RenderLight],
    instances: tuple[RenderInstance, ...],
    diagnostics: list[Diagnostic],
) -> list[RenderLight]:
    """Lights with their receivers: which instances each linked light lights (an object
    named in ``include`` or ``exclude`` stands for itself and its descendants)."""
    ancestry: list[tuple[str, ...]] = []
    for inst in instances:
        chain = []
        try:
            obj = scene.get(inst.object_id)
        except AetherError:  # (an instance of a temporary object: only its own id)
            obj = None
            chain.append(inst.object_id)
        while obj is not None:
            chain.append(obj.id)
            obj = obj.parent
        ancestry.append(tuple(chain))
    out = []
    for light in lights:
        include, exclude = light.links
        if not include and not exclude:
            out.append(light)
            continue
        missing = [i for i in (*include, *exclude) if not _exists(scene, i)]
        if missing:
            diagnostics.append(
                Diagnostic(
                    "LIGHT_LINK_MISSING",
                    WARNING,
                    f"light {light.name!r} links {len(missing)} object id(s) the scene does "
                    f"not have: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""),
                    f"objects[{light.name}].light",
                    count=len(missing),
                )
            )
        inc, exc = set(include), set(exclude)
        lit = np.array(
            [
                (not inc or any(i in inc for i in chain)) and not any(i in exc for i in chain)
                for chain in ancestry
            ],
            dtype=bool,
        )
        out.append(replace(light, receivers=lit))
    return out


def _exists(scene: Scene, object_id: str) -> bool:
    try:
        scene.get(object_id)
    except AetherError:
        return False
    return True


def raster_environment(rs: RenderScene) -> RenderScene:
    """The render scene as the rasterizers light it with an image environment: its sun (if
    it has one) as a directional light and its ambient harmonics
    (:meth:`~aether3d.rendering.envimage.EnvironmentImage.raster_lighting`)."""
    image = rs.environment_image
    if image is None or rs.ambient_sh is not None:
        return rs
    lighting = image.raster_lighting()
    lights = rs.lights
    sun = lighting["sun"]
    if sun is not None:
        omega = min(sun["solid_angle"], 2.0 * math.pi)
        lights = (
            *lights,
            RenderLight(
                None,
                "environment sun",
                "directional",
                np.asarray(sun["irradiance"], np.float64) / math.pi,  # white facing it: E / pi
                None,
                -np.asarray(sun["direction"], np.float64),
                None,
                "none",
                angle=math.degrees(2.0 * math.acos(1.0 - omega / (2.0 * math.pi))),
            ),
        )
    return replace(rs, lights=tuple(lights), ambient_sh=lighting["sh"])


def _material_cache(scene: Scene) -> MaterialCache | None:
    """The scene's material cache (kept with its derived caches), unless switched off."""
    if not _features.enabled("AETHER3D_MATERIAL_CACHE"):
        return None
    derived = scene._derived()
    cache = derived.get("render_materials")
    if cache is None:
        cache = derived["render_materials"] = MaterialCache()
    return cache


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
    return RenderMesh(
        mesh.positions, mesh.triangles, mesh.normals, mesh.uvs, mesh.groups, mesh.colors
    )
