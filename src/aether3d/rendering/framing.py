"""Camera framing: place a camera so that given geometry exactly fills the frame.

For a fixed view direction (camera forward ``f``, up ``u``, right ``r``) and
the world-space vertices ``p`` of the targets, write each vertex relative to
a reference point ``c`` as ``x = (p - c).r``, ``y = (p - c).u`` and
``z = (p - c).f``. Moving the camera to ``c + a r + b u - t f`` keeps the
direction and gives the vertex view coordinates ``(x - a, y - b, z + t)``.

**Perspective** (half-angle tangents ``tx``, ``ty`` reduced by the margin):
every vertex is inside the frame when ``|x - a| <= (z + t) tx`` and
``|y - b| <= (z + t) ty``. For the horizontal axis this holds for some ``a``
exactly when ``t >= (A - B) / (2 tx)`` with ``A = max(x - z tx)`` and
``B = min(x + z tx)``, and ``a = (A + B) / 2`` is then always valid; likewise
vertically. The smallest ``t`` satisfying both axes gives the closest camera
position from which everything is visible, centered on each axis
independently. This is exact (not a bounding-sphere approximation).

**Orthographic**: the view size is ``max`` of the extents (plus margin) along
the frame's axes, centered; the camera is moved back so the nearest vertex is
in front of the near plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, RenderError
from .camera import Camera, CameraFrame

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

AUTO_VIEW_DIRECTION = (-0.6, 1.0, -0.55)
"""View direction of temporary cameras: from the front (-Y), right (+X) and above."""
MIN_EXTENT = 1e-3


@dataclass(frozen=True)
class Framing:
    """The result of :func:`compute_framing` (world-space)."""

    position: np.ndarray
    camera_changes: dict[str, float]  # new near/far (and ortho_scale) if requested
    distance: float  # from the camera to the nearest framed vertex, along the view
    target_center: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "camera_changes": dict(self.camera_changes),
            "distance": self.distance,
            "target_center": self.target_center.tolist(),
        }


def target_points(scene: Scene, targets: Any = None) -> np.ndarray:
    """World-space vertices of ``targets`` (objects, ids or names), or of every visible mesh."""
    from .scene_data import _find_object, world_matrices

    worlds = world_matrices(scene)
    if targets is None:
        objects = [o for o in scene if o.visible and o.camera is None and o.light is None]
    else:
        if isinstance(targets, (str,)) or not isinstance(targets, (list, tuple)):
            targets = [targets]
        objects = [_find_object(scene, t) for t in targets]
    chunks = []
    for obj in objects:
        try:
            mesh = obj.mesh
        except AetherError as exc:
            raise RenderError(
                "GEOMETRY_EVALUATION_FAILED",
                f"cannot frame {obj.name!r}: its geometry could not be evaluated ({exc})",
                f"objects[{obj.name}]",
            ) from None
        if mesh is None or mesh.vertex_count == 0:
            continue
        m = worlds[obj.id]
        p = mesh.positions.astype(np.float64)
        chunks.append(p @ m[:3, :3].T + m[:3, 3])
    if not chunks:
        raise RenderError(
            "NOTHING_TO_FRAME",
            "there is no geometry to frame" + ("" if targets is None else " in the given objects"),
            hint="add mesh objects, or make them visible",
        )
    points = np.concatenate(chunks)
    if not np.all(np.isfinite(points)):
        raise RenderError("NONFINITE_TRANSFORM", "the geometry to frame has non-finite positions")
    return points


def orientation_frame(direction: Any, up: Any = (0.0, 0.0, 1.0)) -> CameraFrame:
    """A camera frame looking along ``direction`` (position at the origin)."""
    from ..math3d import look_rotation_matrix

    rot = look_rotation_matrix(direction, up)
    forward = -rot[:, 1]
    upv = rot[:, 2]
    return CameraFrame(np.zeros(3), forward, upv, np.cross(forward, upv))


def compute_framing(
    camera: Camera,
    frame: CameraFrame,
    points: np.ndarray,
    aspect: float,
    *,
    margin: float = 0.05,
    fit_clipping: bool = True,
) -> Framing:
    """Where to put the camera (keeping its orientation) so ``points`` fill the frame."""
    if not (isinstance(margin, (int, float)) and not isinstance(margin, bool)) or not (
        0.0 <= margin < 0.95
    ):
        raise RenderError("INVALID_FRAMING", f"margin must be in [0, 0.95), got {margin!r}")
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2.0
    rel = points - center
    x, y, z = rel @ frame.right, rel @ frame.up, rel @ frame.forward
    size = float(np.linalg.norm(hi - lo))
    extent = max(size, MIN_EXTENT)
    hx, hy = camera.half_extents(aspect)
    keep = 1.0 - margin
    changes: dict[str, float] = {}
    if camera.is_perspective:
        tx, ty = hx * keep, hy * keep
        ax, bx = np.max(x - z * tx), np.min(x + z * tx)
        ay, by = np.max(y - z * ty), np.min(y + z * ty)
        t = max((ax - bx) / (2 * tx), (ay - by) / (2 * ty))
        a, b = (ax + bx) / 2.0, (ay + by) / 2.0
        # Never put a vertex closer than a sliver of the target size (degenerate targets).
        t = max(t, float(np.max(-z)) + 0.05 * extent)
        position = center + a * frame.right + b * frame.up - t * frame.forward
        depths = z + t
    else:
        a = (x.max() + x.min()) / 2.0
        b = (y.max() + y.min()) / 2.0
        half_w = max((x.max() - x.min()) / 2.0, MIN_EXTENT) / keep
        half_h = max((y.max() - y.min()) / 2.0, MIN_EXTENT) / keep
        # Scale so that both extents fit: hx/hy of a unit ortho_scale give the shape.
        unit_hx, unit_hy = Camera(
            projection="orthographic", ortho_scale=1.0, fov_axis=camera.fov_axis
        ).half_extents(aspect)
        changes["ortho_scale"] = max(half_w / unit_hx, half_h / unit_hy)
        t = float(np.max(-z)) + max(camera.near, 0.0) + 0.05 * extent
        position = center + a * frame.right + b * frame.up - t * frame.forward
        depths = z + t
    nearest, farthest = float(depths.min()), float(depths.max())
    if fit_clipping:
        if camera.is_perspective:
            near = max(nearest * 0.5, farthest * 1e-6, 1e-6)
        else:
            near = max(0.0, nearest * 0.5)
        far = farthest * 1.5 + 0.05 * extent
        if camera.is_perspective and far / near > 1e8:
            near = far / 1e8
        changes["near"], changes["far"] = near, far
    if not all(math.isfinite(v) for v in (*position.tolist(), *changes.values())):
        raise RenderError("INVALID_FRAMING", "framing produced non-finite values")
    return Framing(position, changes, nearest, center)


def frame_object(
    obj: SceneObject,
    targets: Any = None,
    *,
    direction: Any = None,
    up: Any = (0.0, 0.0, 1.0),
    margin: float = 0.05,
    fit_clipping: bool = True,
    aspect: float | None = None,
) -> Framing:
    """Move (and optionally turn) a camera object so ``targets`` fill its frame.

    Everything is computed first; the object changes only if all of it succeeds.
    """
    from ..math3d import compose_matrix
    from .camera import frame_from_matrix
    from .scene_data import _mat4_mul

    scene = obj._require_scene()
    if obj.camera is None:
        raise RenderError(
            "NOT_A_CAMERA", f"object {obj.name!r} is not a camera", f"objects[{obj.name}]"
        )
    points = target_points(scene, targets)
    transform = obj.transform
    rotation = transform.rotation if direction is None else obj._aim_rotation(direction, up)
    local = compose_matrix(transform.position, rotation, transform.scale)
    parent = None if obj.parent is None else obj.parent.world_matrix
    world = local if parent is None else _mat4_mul(parent, local)
    frame, _ = frame_from_matrix(world, f"objects[{obj.name}]")
    if aspect is None:
        settings = scene.render_settings
        aspect = obj.camera.aspect_ratio or settings.width / settings.height
    framing = compute_framing(
        obj.camera, frame, points, aspect, margin=margin, fit_clipping=fit_clipping
    )
    target = np.append(framing.position, 1.0)
    if parent is not None:
        try:
            target = np.linalg.solve(parent, target)
        except np.linalg.LinAlgError:
            raise RenderError(
                "INVALID_CAMERA_TRANSFORM", f"the parent of {obj.name!r} has a zero world scale"
            ) from None
    new_camera = obj.camera.copy().configure(**framing.camera_changes)  # validates first
    obj.camera.configure(**new_camera.to_dict())
    transform.rotation = rotation
    transform.position = target[:3]
    return framing
