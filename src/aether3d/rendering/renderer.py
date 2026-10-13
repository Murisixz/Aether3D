"""The public rendering entry point: :func:`render` and :class:`RenderResult`."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._version import __version__
from ..diagnostics import INFO, Diagnostic, worst_severity
from ..errors import RenderError
from . import cpu
from .camera import Camera, CameraFrame, frame_from_matrix
from .color import linear_to_srgb, quantize
from .framing import AUTO_VIEW_DIRECTION, compute_framing, orientation_frame, target_points
from .png import encode_png
from .scene_data import RenderCamera, extract_render_scene, resolve_camera, world_matrices
from .settings import RenderLimits, RenderSettings

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject


@dataclass
class RenderResult:
    """An image rendered by Aether3D's CPU renderer, with everything known about it.

    * ``image`` -- ``(height, width, 3) uint8`` sRGB pixels (row 0 at the top).
    * ``linear`` -- ``(height, width, 3) float32`` linear-light values after
      exposure and before clamping/encoding (values above 1 are kept).
    * ``depth`` -- ``(height, width) float32`` distance along the camera's view
      direction to the surface seen at each pixel (``inf`` for background).
    * ``coverage`` -- ``(height, width) float32`` fraction of each pixel's
      samples covered by geometry.
    * ``object_index`` -- ``(height, width) int32`` index into :attr:`objects`
      of the object seen at each pixel (``-1`` for background).
    * ``objects`` -- rendered objects: id, name, triangle count and how many
      samples of each are visible.
    * ``camera``, ``settings``, ``stats`` (counts and timings in seconds),
      ``diagnostics`` and ``output`` (set by :meth:`save`).

    ``depth`` and ``object_index`` describe one representative sample per pixel
    (the center sample for odd ``samples``).
    """

    image: np.ndarray
    linear: np.ndarray
    depth: np.ndarray
    coverage: np.ndarray
    object_index: np.ndarray
    objects: list[dict[str, Any]]
    camera: dict[str, Any]
    settings: dict[str, Any]
    stats: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    output: Path | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def status(self) -> str:
        """``"ok"`` or ``"warning"`` (renders that fail raise instead)."""
        return worst_severity(self.diagnostics)

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        """The 8-bit sRGB color at column ``x``, row ``y`` (row 0 at the top)."""
        r, g, b = (int(v) for v in self.image[y, x])
        return (r, g, b)

    def png_bytes(self) -> bytes:
        """The image encoded as PNG."""
        return encode_png(self.image, text={"Software": f"Aether3D {__version__}"})

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the image as a PNG file, atomically; returns the resolved path.

        The file is written to a temporary sibling and renamed into place, so
        a failure never leaves a partial or corrupted image.
        """
        from .._fileio import write_atomic

        target = Path(path)
        if target.suffix.lower() != ".png":
            raise RenderError(
                "INVALID_OUTPUT", f"output must be a .png file, got {str(target)!r}", "output"
            )
        t0 = time.perf_counter()
        data = self.png_bytes()
        t1 = time.perf_counter()
        try:
            written = write_atomic(target, data)
        except OSError as exc:
            raise RenderError(
                "OUTPUT_ERROR",
                f"cannot write {target}: {exc.strerror or exc}",
                "output",
                hint="check that the directory exists and is writable",
            ) from None
        self.stats["timings"]["encode"] = t1 - t0
        self.stats["timings"]["write"] = time.perf_counter() - t1
        self.stats["png_bytes"] = len(data)
        self.output = written
        return written

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible report (without pixel arrays)."""
        return {
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "output": None if self.output is None else str(self.output),
            "camera": self.camera,
            "settings": self.settings,
            "stats": self.stats,
            "objects": self.objects,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _resolve_size(
    settings: RenderSettings, width: Any, height: Any, camera: Camera
) -> tuple[Any, Any]:
    if (width is None) == (height is None):
        return width, height
    aspect = camera.aspect_ratio or settings.width / settings.height
    if width is not None and isinstance(width, int) and not isinstance(width, bool):
        return width, max(1, round(width / aspect))
    if height is not None and isinstance(height, int) and not isinstance(height, bool):
        return max(1, round(height * aspect)), height
    return width, height  # invalid values: let validation report them


def render(
    scene: Scene,
    *,
    camera: SceneObject | str | None = None,
    width: int | None = None,
    height: int | None = None,
    samples: int | None = None,
    exposure: float | None = None,
    cull: str | None = None,
    default_lights: bool | None = None,
    frame: bool = False,
    margin: float = 0.05,
    limits: RenderLimits | None = None,
) -> RenderResult:
    """Render ``scene`` with the CPU rasterizer and return a :class:`RenderResult`.

    * ``camera`` -- camera object, id or name; default: ``scene.active_camera``.
    * ``width``, ``height``, ``samples``, ``exposure``, ``cull``,
      ``default_lights`` -- override ``scene.render_settings`` for this render.
      Giving only one of ``width``/``height`` derives the other from the
      camera's ``aspect_ratio`` (or the saved settings' shape).
    * ``frame`` -- frame the camera on all visible geometry for this render
      only (the scene keeps its camera placement). With no camera at all, a
      temporary perspective camera looking from the front-right and above is
      used. ``margin`` is the free border fraction.
    * ``limits`` -- :class:`RenderLimits` (image size, samples, triangles).

    The scene is never modified. Procedural objects render their current
    evaluated geometry. Errors raise :class:`~aether3d.errors.RenderError`
    with a stable ``code``.
    """
    t_start = time.perf_counter()
    limits = limits or RenderLimits()
    if not isinstance(limits, RenderLimits):
        raise RenderError("INVALID_SETTINGS", "limits must be a RenderLimits instance")
    if not isinstance(frame, bool):
        raise RenderError("INVALID_SETTINGS", f"frame must be True or False, got {frame!r}")
    diagnostics: list[Diagnostic] = []

    camera_obj: SceneObject | None
    try:
        camera_obj = resolve_camera(scene, camera)
    except RenderError as exc:
        if not (frame and exc.code == "NO_CAMERA"):
            raise
        camera_obj = None
    base_camera = camera_obj.camera if camera_obj is not None else Camera()

    settings = scene.render_settings.copy()
    width, height = _resolve_size(settings, width, height, base_camera)
    overrides = {
        k: v
        for k, v in (
            ("width", width),
            ("height", height),
            ("samples", samples),
            ("exposure", exposure),
            ("cull", cull),
            ("default_lights", default_lights),
        )
        if v is not None
    }
    settings.configure(**overrides)
    aspect = settings.width / settings.height
    if settings.width > limits.max_dimension or settings.height > limits.max_dimension:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"image size {settings.width}x{settings.height} exceeds the limit of "
            f"{limits.max_dimension} pixels per side",
        )

    render_camera: SceneObject | RenderCamera
    if frame:
        if camera_obj is not None:
            cam = camera_obj.camera.copy()
            world = world_matrices(scene)[camera_obj.id]
            base_frame, cam_diags = frame_from_matrix(world, f"objects[{camera_obj.name}]")
            diagnostics += cam_diags
            cam_id, cam_name = camera_obj.id, camera_obj.name
        else:
            cam = Camera()
            base_frame = orientation_frame(AUTO_VIEW_DIRECTION)
            cam_id, cam_name = None, "temporary camera"
        framing = compute_framing(
            cam, base_frame, target_points(scene), aspect, margin=margin, fit_clipping=True
        )
        cam.configure(**framing.camera_changes)
        framed = CameraFrame(framing.position, base_frame.forward, base_frame.up, base_frame.right)
        render_camera = RenderCamera(cam_id, cam_name, cam, framed)
        diagnostics.append(
            Diagnostic(
                "CAMERA_FRAMED",
                INFO,
                f"{cam_name!r} was framed on the visible geometry for this render only "
                "(the scene was not modified)",
            )
        )
    else:
        assert camera_obj is not None
        render_camera = camera_obj

    rs = extract_render_scene(
        scene, render_camera, default_lights=settings.default_lights, aspect=aspect
    )
    diagnostics += list(rs.diagnostics)
    frame_data = cpu.render_scene(
        rs, settings.width, settings.height, settings.samples, settings.cull, limits
    )
    diagnostics += frame_data.diagnostics

    t0 = time.perf_counter()
    linear = frame_data.linear
    if settings.exposure != 0.0:
        linear *= 2.0**settings.exposure
    image = np.empty(linear.shape, dtype=np.uint8)
    for r0 in range(0, linear.shape[0], 128):  # bounded temporaries
        image[r0 : r0 + 128] = quantize(linear_to_srgb(linear[r0 : r0 + 128]))
    t_output = time.perf_counter() - t0

    objects = [
        {
            "index": k,
            "id": inst.object_id,
            "name": inst.name,
            "triangles": int(rs.meshes[inst.mesh].triangles.shape[0]),
            "visible_samples": int(frame_data.visible_samples[k]),
        }
        for k, inst in enumerate(rs.instances)
    ]
    counts = frame_data.counts
    timings = {
        "evaluate": rs.timings.get("evaluate", 0.0),
        "extract": rs.timings.get("extract", 0.0),
        **frame_data.timings,
        "output": t_output,
    }
    timings["total"] = time.perf_counter() - t_start
    stats = {
        **counts,
        "pixels_covered": int(np.count_nonzero(frame_data.coverage > 0)),
        "timings": timings,
    }
    cam_info = rs.camera.to_dict(aspect)
    cam_info["temporary"] = frame
    if not math.isfinite(float(np.max(linear, initial=0.0))):
        raise RenderError("RENDER_FAILED", "the rendered image contains non-finite values")
    return RenderResult(
        image=image,
        linear=linear.astype(np.float32),
        depth=frame_data.depth,
        coverage=frame_data.coverage,
        object_index=frame_data.instance,
        objects=objects,
        camera=cam_info,
        settings=settings.to_dict(),
        stats=stats,
        diagnostics=diagnostics,
    )
