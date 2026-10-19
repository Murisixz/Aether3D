"""The public rendering entry point: :func:`render` and :class:`RenderResult`."""

from __future__ import annotations

import math
import os
import time
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._version import __version__
from ..diagnostics import INFO, WARNING, Diagnostic, worst_severity
from ..errors import RenderError
from . import cpu
from .backends import PATH_BACKENDS, Antialiasing, antialiasing_plan, check_backend
from .camera import Camera, CameraFrame, frame_from_matrix
from .framing import AUTO_VIEW_DIRECTION, compute_framing, orientation_frame, target_points
from .overlay import animation_overlay
from .png import encode_png
from .scene_data import (
    MAX_LINKED_LIGHTS,
    RenderCamera,
    RenderScene,
    _with_fields,
    extract_render_scene,
    raster_environment,
    resolve_camera,
    world_matrices,
)
from .settings import AOSettings, PathSettings, RenderLimits, RenderSettings, ShadowSettings
from .shadows import ShadowPlan, plan_shadows
from .tonemap import check_tonemap, color_info, display

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject


@dataclass
class RenderResult:
    """An image rendered by Aether3D (CPU or GPU backend), with everything known about it.

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
    * ``backend`` -- which backend rendered it: ``{"name": "cpu"}``, for
      the GPU the adapter, graphics API, antialiasing, shadow maps, GPU pass
      timings and resource statistics, for the path tracer
      (``{"name": "path", "device": "cpu", ...}``) its settings, BVH and
      light statistics.
    * ``passes`` -- path tracer only: named linear buffers computed by the
      renderer (``beauty``, ``direct``, ``indirect``, ``albedo``, ``normal``,
      ``depth``, ``sample_count``, ``variance`` and ``ao`` when enabled;
      before exposure). Empty for the rasterizers.
    * ``color`` -- the output transform: working space, exposure, tone
      mapper and encoding.

    ``linear``, ``depth``, ``coverage`` and ``object_index`` are ``None`` when a
    GPU render was asked for fewer ``outputs``.

    ``depth`` and ``object_index`` describe one representative sample per pixel
    (the center sample for odd ``samples``; on the GPU with MSAA, the first
    MSAA sample of the center of the supersampled block; for the path tracer
    a ray through the pixel center).
    """

    image: np.ndarray
    linear: np.ndarray | None  # None: not requested (render(outputs=...))
    depth: np.ndarray | None
    coverage: np.ndarray | None
    object_index: np.ndarray | None
    objects: list[dict[str, Any]]
    camera: dict[str, Any]
    settings: dict[str, Any]
    stats: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    output: Path | None = None
    backend: dict[str, Any] = field(default_factory=lambda: {"name": "cpu"})
    passes: dict[str, np.ndarray] = field(default_factory=dict)
    color: dict[str, Any] = field(default_factory=lambda: color_info(0.0, "none"))

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

    def png_bytes(self, compression: int = 6) -> bytes:
        """The image encoded as PNG (zlib ``compression`` 0-9: 1 is several times faster
        to encode than the default 6 and a few percent larger; the pixels are the same).

        A ``Software`` text chunk names Aether3D; GPU renders also carry a
        ``Source`` chunk naming the graphics API and adapter that drew them.
        """
        text = {"Software": f"Aether3D {__version__}"}
        if self.backend.get("name") == "gpu":
            adapter = self.backend.get("adapter", {})
            text["Source"] = (
                f"Aether3D GPU renderer (wgpu, {adapter.get('backend')}) on {adapter.get('name')}"
            )
        elif self.backend.get("name") == "path":
            b = self.backend
            text["Source"] = (
                f"Aether3D path tracer on the CPU (NumPy), {b.get('samples')} samples per pixel, "
                f"{b.get('bounces')} bounces, seed {b.get('seed')}"
            )
        elif self.backend.get("name") == "gpu_path":
            b = self.backend
            adapter = b.get("adapter", {})
            text["Source"] = (
                f"Aether3D path tracer on the GPU (wgpu compute, {adapter.get('backend')}) on "
                f"{adapter.get('name')}, {b.get('samples')} samples per pixel, "
                f"{b.get('bounces')} bounces, seed {b.get('seed')}"
            )
        return encode_png(self.image, text=text, compression=compression)

    def save(self, path: str | os.PathLike[str], *, bits: int = 8) -> Path:
        """Write the image to a file, atomically; returns the resolved path.

        ``.png`` writes the display image -- 8-bit, or with ``bits=16`` (0.20.0) a 16-bit
        master of the same view transform from :attr:`linear`; ``.hdr`` (Radiance RGBE)
        and ``.pfm`` (float32) write :attr:`linear` -- high dynamic range, after exposure
        and before tone mapping; ``.exr`` (0.20.0) writes it as scene-linear OpenEXR (half
        ``R``, ``G``, ``B``) with the path tracers' passes (``Z`` depth, ``normal.X/Y/Z``,
        ``albedo``, ``direct``, ``indirect``, ``variance`` as ``.R/G/B``, before exposure;
        ``motion.X/Y/Z/W``). The file is written to a temporary sibling and renamed into
        place, so a failure never leaves a partial or corrupted image.
        """
        from .._fileio import write_atomic
        from .hdr import encode_hdr, encode_pfm

        target = Path(path)
        suffix = target.suffix.lower()
        if suffix not in (".png", ".hdr", ".pfm", ".exr"):
            raise RenderError(
                "INVALID_OUTPUT",
                f"output must be a .png, .hdr, .pfm or .exr file, got {str(target)!r}",
                "output",
            )
        if bits not in (8, 16) or (bits == 16 and suffix != ".png"):
            raise RenderError("INVALID_OUTPUT", "bits is 8 or 16 (16: a .png master)", "bits")
        if (suffix != ".png" or bits == 16) and self.linear is None:
            raise RenderError(
                "OUTPUT_NOT_RENDERED",
                f"{suffix} files hold linear light, and this render did not return it",
                "output",
                hint="render with outputs=None (all) or outputs including 'linear'",
            )
        t0 = time.perf_counter()
        if suffix == ".exr":
            return self._save_exr(target, t0)
        if suffix == ".png" and bits == 16:
            from .png import encode_png
            from .tonemap import display

            image = display(self.linear, self.color.get("tonemap", "none"), bits=16)
            data = encode_png(image, text={"Software": f"Aether3D {__version__}"})
        elif suffix == ".png":
            data = self.png_bytes()
        elif suffix == ".hdr":
            data = encode_hdr(self.linear)
        else:
            data = encode_pfm(self.linear)
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
        self.stats["png_bytes" if suffix == ".png" else "file_bytes"] = len(data)
        self.output = written
        return written

    def _save_exr(self, target: Path, t0: float) -> Path:
        from ..animation.frames import PASS_CHANNELS
        from .exr import write_exr

        lin = self.linear
        channels = {"R": lin[..., 0], "G": lin[..., 1], "B": lin[..., 2]}
        for name, plane in (self.passes or {}).items():
            names = PASS_CHANNELS.get(name)
            if names is None:
                continue
            if len(names) == 1:
                channels[names[0]] = plane
            else:
                for k, c in enumerate(names):
                    channels[c] = plane[..., k]
        try:
            written = write_exr(target, channels)
        except OSError as exc:
            raise RenderError(
                "OUTPUT_ERROR",
                f"cannot write {target}: {exc.strerror or exc}",
                "output",
                hint="check that the directory exists and is writable",
            ) from None
        self.stats["timings"]["encode"] = time.perf_counter() - t0
        self.stats["file_bytes"] = written.stat().st_size
        self.output = written
        return written

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible report (without pixel arrays)."""
        return {
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "output": None if self.output is None else str(self.output),
            "backend": self.backend,
            "camera": self.camera,
            "settings": self.settings,
            "color": self.color,
            "passes": sorted(self.passes),
            "stats": self.stats,
            "objects": self.objects,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


OUTPUTS = frozenset({"image", "linear", "depth", "object_index", "coverage"})


def check_outputs(outputs: Any) -> frozenset[str]:
    """``render(outputs=...)``: ``None`` (all) or names from :data:`OUTPUTS`."""
    if outputs is None:
        return OUTPUTS
    names = [outputs] if isinstance(outputs, str) else outputs
    try:
        chosen = frozenset(names)
    except TypeError:
        raise RenderError(
            "INVALID_SETTINGS",
            f"outputs must be None or a list of output names, got {outputs!r}",
            "outputs",
        ) from None
    unknown = sorted(str(n) for n in chosen - OUTPUTS)
    if unknown:
        raise RenderError(
            "INVALID_SETTINGS",
            f"unknown render outputs: {', '.join(unknown)}",
            "outputs",
            hint=f"outputs: {', '.join(sorted(OUTPUTS))}",
        )
    return chosen | {"image"}


def check_virtual_camera(camera: RenderCamera) -> RenderCamera:
    """A virtual camera passed to :func:`render`, checked: no object id (it is not in the
    scene), a :class:`Camera`, and a finite, orthonormal, right-handed frame."""
    if camera.object_id is not None:
        raise RenderError(
            "INVALID_CAMERA",
            "a RenderCamera passed as camera= is a virtual camera and has no object id",
            "camera",
            hint="pass the camera object (or its id or name) to render from a scene camera",
        )
    if not isinstance(camera.camera, Camera) or not isinstance(camera.frame, CameraFrame):
        raise RenderError("INVALID_CAMERA", "a virtual camera needs a Camera and a CameraFrame")
    f = camera.frame
    vectors = [np.asarray(v, dtype=np.float64) for v in (f.position, f.forward, f.up, f.right)]
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in vectors):
        raise RenderError("INVALID_CAMERA", "the virtual camera's frame has non-finite values")
    _, forward, up, right = vectors
    tolerance = 1e-9
    unit = all(abs(float(v @ v) - 1.0) <= tolerance for v in (forward, up, right))
    orthogonal = abs(float(forward @ up)) <= tolerance and abs(float(forward @ right)) <= tolerance
    handed = np.allclose(np.cross(forward, up), right, rtol=0.0, atol=1e-9)
    if not (unit and orthogonal and handed):
        raise RenderError(
            "INVALID_CAMERA",
            "the virtual camera's frame is not orthonormal (right = forward x up, unit vectors)",
            "camera",
            hint="build it with aether3d.feedback.virtual_camera(...)",
        )
    return camera


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
    backend: str = "cpu",
    adapter: int | str | None = None,
    msaa: int | None = None,
    supersample: int | None = None,
    shadows: bool | dict[str, Any] | ShadowSettings | None = None,
    path: dict[str, Any] | PathSettings | None = None,
    ao: bool | dict[str, Any] | AOSettings | None = None,
    tonemap: str = "none",
    progress: Any = None,
    outputs: Any = None,
    time: float | None = None,
    dof_samples: int | None = None,
    denoise: bool | dict[str, Any] | None = None,
    motion: float | None = None,
) -> RenderResult:
    """Render ``scene`` and return a :class:`RenderResult`.

    * ``motion`` -- seconds (a frame's interval): adds ``passes["motion"]``, the pixel
      motion to ``time - motion`` and ``time + motion`` of what each pixel shows
      (:mod:`~aether3d.rendering.motion`; any backend, a scene camera).

    * ``denoise`` -- path tracers only: ``True`` (or settings, see
      :class:`~aether3d.rendering.denoise.DenoiseSettings`) filters the image with the
      edge-avoiding, variance-guided denoiser (:mod:`~aether3d.rendering.denoise`; biased).
      ``linear`` and ``image`` are then denoised, ``passes["beauty"]`` stays the raw
      estimate and ``backend["denoiser"]`` describes the filter.

    * ``dof_samples`` -- rasterizers only: how many views from points of the camera's
      aperture are averaged when it has depth of field (``fstop > 0``; default 16, ``1``:
      a pinhole view). Each view is a real render from a lens point, its frustum sheared
      so the plane in focus stays fixed (an accumulation buffer, Haeberli and Akeley
      1990); the path tracers sample the lens with every path instead.

    * ``time`` -- for an animated scene, the time in seconds to render (default: the
      animation's current time, ``scene.animation.time``); the animation's evaluation
      there poses the objects for this render only. Scenes without animation ignore it.

    * ``camera`` -- camera object, id or name; default: ``scene.active_camera``. A
      :class:`~aether3d.rendering.scene_data.RenderCamera` without an object id (a
      *virtual* camera: a :class:`Camera` and a :class:`CameraFrame` that are not part of
      the scene; :func:`aether3d.feedback.virtual_camera` makes one) renders from that view
      without adding anything to the scene.
    * ``width``, ``height``, ``samples``, ``exposure``, ``cull``,
      ``default_lights`` -- override ``scene.render_settings`` for this render.
      Giving only one of ``width``/``height`` derives the other from the
      camera's ``aspect_ratio`` (or the saved settings' shape).
    * ``frame`` -- frame the camera on all visible geometry for this render
      only (the scene keeps its camera placement). With no camera at all, a
      temporary perspective camera looking from the front-right and above is
      used. ``margin`` is the free border fraction.
    * ``limits`` -- :class:`RenderLimits` (image size, samples, triangles,
      GPU memory).
    * ``backend`` -- ``"cpu"`` (default: Aether3D's NumPy rasterizer),
      ``"gpu"`` (wgpu: Vulkan, Metal or Direct3D 12; fails with a ``GPU_*``
      error if no usable GPU) or ``"auto"`` (the GPU when available, else the
      CPU, with a ``BACKEND_FALLBACK`` diagnostic). See ``docs/GPU.md``.
    * ``adapter`` -- GPU adapter selection (index, ``"discrete"``, ``"nvidia"``,
      ``"vulkan"``, ...); default ``AETHER3D_GPU_ADAPTER`` or the best one.
    * ``msaa``, ``supersample`` -- GPU antialiasing: MSAA samples per pixel and
      supersampling factor per axis. By default the GPU uses ``samples^2``
      samples per pixel like the CPU (``samples=2``: 4x MSAA).
      ``supersample=samples, msaa=1`` reproduces the CPU's sample positions.
    * ``shadows`` -- GPU shadow maps for directional lights: ``True``, a
      :class:`ShadowSettings` or a dict of its fields; default off.
    * ``path`` -- path tracer settings (``backend="path"``): a
      :class:`PathSettings` or a dict of its fields (``samples`` per pixel,
      ``bounces``, ``seed``, ``light_samples``, ...). ``samples`` above
      means raster antialiasing and does not apply to the path tracer.
    * ``ao`` -- ambient occlusion pass of the path tracer: ``True``, an
      :class:`AOSettings` or a dict of its fields (``samples``,
      ``distance``, ``strength``). ``path={"integrator": "ao"}`` renders
      only AO (a fast preview).
    * ``tonemap`` -- ``"none"`` (default: clip, as always) or ``"neutral"``
      (Khronos PBR Neutral): how linear light is mapped to the PNG (any
      backend); ``result.color`` reports it.
    * ``progress`` -- path tracer only: called after every completed sample
      pass as ``progress(samples_done, samples_total, seconds)``; returning
      ``False`` stops the render there (the image averages the samples done
      and a ``RENDER_STOPPED`` diagnostic says so).
    * ``outputs`` -- which per-pixel outputs to return: ``None`` (default:
      all) or some of ``"image"``, ``"linear"``, ``"depth"``,
      ``"object_index"``, ``"coverage"`` (the image always comes). The GPU
      backend then reads back only those -- most of a warm GPU frame is reading
      outputs back, 28 bytes per pixel for all of them, 4 for the image alone --
      and the others are ``None`` in the result; ``objects``, statistics and
      diagnostics are complete either way. The CPU renderer and the path tracers
      compute every output anyway and return them all.

    The scene is never modified. Procedural objects render their current
    evaluated geometry. Errors raise :class:`~aether3d.errors.RenderError`
    with a stable ``code``.
    """
    t_start = _time.perf_counter()
    overlay = animation_overlay(scene, time)
    if camera is None and overlay is not None and overlay.camera is not None:
        camera = overlay.camera  # the shot camera of the animation's sequence
    limits = limits or RenderLimits()
    if not isinstance(limits, RenderLimits):
        raise RenderError("INVALID_SETTINGS", "limits must be a RenderLimits instance")
    if not isinstance(frame, bool):
        raise RenderError("INVALID_SETTINGS", f"frame must be True or False, got {frame!r}")
    requested = check_backend(backend)
    tonemap = check_tonemap(tonemap)
    wanted = check_outputs(outputs)
    if tonemap != "none":
        wanted = wanted | {"linear"}  # the display image is tone mapped from linear light
    from .denoise import DenoiseSettings

    denoiser = DenoiseSettings.coerce(denoise)
    if denoiser is not None and backend not in PATH_BACKENDS:
        raise RenderError(
            "INVALID_DENOISE",
            "denoise filters the path tracers' images (their variance and feature passes)",
            "denoise",
            hint="use backend='path' or 'gpu_path', or render without denoise",
        )
    shadow_settings = ShadowSettings.coerce(shadows)
    if requested not in PATH_BACKENDS:
        for name, value in (("path", path), ("ao", ao), ("progress", progress)):
            if value is not None:
                raise RenderError(
                    "INVALID_SETTINGS",
                    f"{name} applies to the path tracer (backend='path' or 'gpu_path'), not to "
                    f"backend={requested!r}",
                    name,
                    hint="use backend='path' or 'gpu_path', or remove the option",
                )
    path_settings = ao_settings = None
    if dof_samples is not None and (
        isinstance(dof_samples, bool)
        or not isinstance(dof_samples, int)
        or not 1 <= dof_samples <= MAX_DOF_SAMPLES
    ):
        raise RenderError(
            "INVALID_SETTINGS",
            f"dof_samples is an integer 1..{MAX_DOF_SAMPLES}, got {dof_samples!r}",
            "dof_samples",
        )
    if requested in PATH_BACKENDS and dof_samples is not None:
        raise RenderError(
            "INVALID_SETTINGS",
            "dof_samples applies to the rasterizers; the path tracer samples the lens with "
            "every path (its samples)",
            "dof_samples",
            hint="remove dof_samples, or raise path={'samples': N}",
        )
    if requested in PATH_BACKENDS:
        ignored = [("msaa", msaa), ("supersample", supersample), ("samples", samples)]
        if requested == "path":
            ignored.append(("adapter", adapter))
        for name, value in ignored:
            if value is not None:
                hint = (
                    "set the path tracer's samples per pixel with path={'samples': N}"
                    if name == "samples"
                    else "remove the option (the path tracer runs on the CPU; "
                    "backend='gpu_path' runs it on a GPU adapter)"
                    if name == "adapter"
                    else "remove the option (the path tracer antialiases with its samples)"
                )
                raise RenderError(
                    "INVALID_SETTINGS",
                    f"{name} does not apply to the path tracer (backend={requested!r})",
                    name,
                    hint=hint,
                )
        if shadow_settings is not None:
            raise RenderError(
                "INVALID_SETTINGS",
                "the path tracer always traces shadow rays; shadow maps (shadows=) are a GPU "
                "raster feature",
                "shadows",
                hint="remove shadows=, or use backend='gpu'",
            )
        path_settings = PathSettings.coerce(path)
        if requested == "gpu_path" and path_settings.workers != 0:
            raise RenderError(
                "INVALID_SETTINGS",
                "workers sets the CPU path tracer's processes; backend='gpu_path' runs on the GPU",
                "path.workers",
                hint="remove workers, or use backend='path'",
            )
        ao_settings = AOSettings.coerce(ao)
        if path_settings.integrator == "ao" and ao_settings is None:
            ao_settings = AOSettings()  # the AO preview needs AO settings: the defaults
        if progress is not None and not callable(progress):
            raise RenderError("INVALID_SETTINGS", "progress must be callable", "progress")
    if requested == "cpu":
        for name, value in (("msaa", msaa), ("supersample", supersample)):
            if value is not None:
                raise RenderError(
                    "INVALID_SETTINGS",
                    f"{name} applies to the GPU backend; the CPU renderer antialiases with "
                    "samples (samples x samples per pixel)",
                    name,
                    hint="use backend='gpu' or 'auto', or remove the option",
                )
        if shadow_settings is not None:
            raise RenderError(
                "BACKEND_UNSUPPORTED",
                "the CPU renderer does not render shadows",
                "shadows",
                hint="use backend='gpu' (or 'auto'), or render without shadows",
            )
    diagnostics: list[Diagnostic] = []

    camera_obj: SceneObject | None
    virtual = check_virtual_camera(camera) if isinstance(camera, RenderCamera) else None
    if virtual is not None:
        camera_obj = None
        base_camera = virtual.camera
    else:
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
    antialiasing = antialiasing_plan(settings.samples, msaa, supersample)
    aspect = settings.width / settings.height
    if settings.width > limits.max_dimension or settings.height > limits.max_dimension:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"image size {settings.width}x{settings.height} exceeds the limit of "
            f"{limits.max_dimension} pixels per side",
        )

    render_camera: SceneObject | RenderCamera
    if frame:
        if virtual is not None:
            cam = virtual.camera.copy()
            base_frame = virtual.frame
            cam_id, cam_name = None, virtual.name
        elif camera_obj is not None:
            cam = camera_obj.camera.copy()
            values = None if overlay is None else overlay.field_values(camera_obj.id, "camera")
            if values:
                cam = _with_fields(cam, values)
            world = world_matrices(scene, overlay)[camera_obj.id]
            base_frame, cam_diags = frame_from_matrix(world, f"objects[{camera_obj.name}]")
            diagnostics += cam_diags
            cam_id, cam_name = camera_obj.id, camera_obj.name
        else:
            cam = Camera()
            base_frame = orientation_frame(AUTO_VIEW_DIRECTION)
            cam_id, cam_name = None, "temporary camera"
        framing = compute_framing(
            cam,
            base_frame,
            target_points(scene, overlay=overlay),
            aspect,
            margin=margin,
            fit_clipping=True,
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
    elif virtual is not None:
        render_camera = virtual
    else:
        assert camera_obj is not None
        render_camera = camera_obj

    rs = extract_render_scene(
        scene, render_camera, default_lights=settings.default_lights, aspect=aspect, overlay=overlay
    )
    if scene._worlds is not None and len(scene._worlds):
        from ..worlds.render import contribute

        rs = contribute(scene, rs, settings.height, aspect)  # procedural worlds for this camera
    diagnostics += list(rs.diagnostics)
    if requested not in PATH_BACKENDS:  # an image environment's sun and ambient harmonics
        rs = raster_environment(rs)
    plan = None if shadow_settings is None else plan_shadows(rs, shadow_settings, aspect)
    if requested in ("gpu", "gpu_path", "auto") and any(
        light.receivers is not None for light in rs.lights[MAX_LINKED_LIGHTS:]
    ):
        late = [
            light.name for light in rs.lights[MAX_LINKED_LIGHTS:] if light.receivers is not None
        ]
        diagnostics.append(
            Diagnostic(
                "LIGHT_LINK_LIMIT",
                WARNING,
                f"the GPU links only the first {MAX_LINKED_LIGHTS} lights (put linked lights "
                f"first, or render on the CPU); these light every object there: "
                f"{', '.join(late[:5])}" + (" ..." if len(late) > 5 else ""),
                "lights",
                count=len(late),
            )
        )

    if motion is not None and (
        isinstance(motion, bool) or not isinstance(motion, (int, float)) or not 0 < motion <= 3600
    ):
        raise RenderError("INVALID_SETTINGS", "motion is a time interval in seconds (> 0)",
                          "motion")  # fmt: skip
    if requested in PATH_BACKENDS:
        result = _finish_path(
            rs,
            settings,
            path_settings,
            ao_settings,
            limits,
            diagnostics,
            frame,
            aspect,
            t_start,
            tonemap,
            progress,
            requested,
            adapter,
            denoiser,
        )
    else:
        lens_views = 1
        if rs.camera.camera.has_depth_of_field:
            lens_views = DOF_SAMPLES if dof_samples is None else dof_samples
        if lens_views > 1:
            result = _lens_views(
                rs, lens_views, aspect, settings, tonemap, wanted, diagnostics,
                lambda rs_k, wanted_k, diags: _finish_raster(
                    rs_k, settings, antialiasing, plan, adapter, limits, requested, diags,
                    frame, aspect, t_start, tonemap, wanted_k, msaa, supersample,
                ),
            )  # fmt: skip
        else:
            result = _finish_raster(
                rs, settings, antialiasing, plan, adapter, limits, requested, diagnostics, frame,
                aspect, t_start, tonemap, wanted, msaa, supersample,
            )  # fmt: skip
    if motion is not None:
        from .motion import motion_vectors

        now = scene.animation.time if time is None else float(time)
        result.passes["motion"] = motion_vectors(
            scene, rs, settings.width, settings.height, now, float(motion),
            None if frame else camera, settings.default_lights,
        )  # fmt: skip
    return result


DOF_SAMPLES = 16
"""Aperture views a rasterized render with depth of field averages by default."""
MAX_DOF_SAMPLES = 1024


def _lens_points(n: int) -> np.ndarray:
    """``n`` well-spread uniform points of the unit square (Hammersley: (k + 1/2) / n and
    the base-2 radical inverse of k), the same for every render."""
    k = np.arange(n)
    inv = np.zeros(n)
    bit = 0.5
    rest = k.copy()
    while np.any(rest):
        inv += bit * (rest & 1)
        rest >>= 1
        bit *= 0.5
    return np.stack([(k + 0.5) / n, inv], 1)


def _lens_views(rs, n, aspect, settings, tonemap, wanted, diagnostics, finish) -> RenderResult:
    """A rasterized render with depth of field: the pinhole view (depth, objects and the
    statistics) and ``n`` views from points of the aperture, averaged in linear light."""
    from dataclasses import replace

    from .camera import aperture_points
    from .tonemap import display

    base = finish(rs, wanted, list(diagnostics))
    cam = rs.camera.camera
    frame = rs.camera.frame
    radius = cam.aperture_radius()
    focus = rs.camera.focus_distance
    hx, hy = cam.half_extents(aspect)
    pts = _lens_points(n)
    ax, ay = aperture_points(pts[:, 0], pts[:, 1], cam.aperture_blades,
                             cam.aperture_rotation, cam.aperture_ratio)  # fmt: skip
    total = None
    for lx, ly in zip((ax * radius).tolist(), (ay * radius).tolist(), strict=True):
        moved = CameraFrame(
            frame.position + lx * frame.right + ly * frame.up, frame.forward, frame.up, frame.right
        )
        # The plane in focus stays put: shear the frustum by the lens offset over the focus.
        sheared = cam.copy()
        sheared._owner = None
        sheared.configure(
            shift_x=cam.shift_x - lx / (focus * hx) / (2.0 * max(1.0, 1.0 / aspect)),
            shift_y=cam.shift_y - ly / (focus * hy) / (2.0 * max(1.0, aspect)),
        )
        view = replace(rs, camera=replace(rs.camera, camera=sheared, frame=moved))
        out = finish(view, frozenset({"image", "linear"}), [])
        lin = out.linear.astype(np.float64)
        total = lin if total is None else total + lin
    linear = (total / n).astype(np.float32)
    base.linear = linear if base.linear is not None or "linear" in wanted else None
    base.image = display(linear, tonemap)
    base.stats = {**base.stats, "dof_samples": n}
    base.diagnostics = [
        *base.diagnostics,
        Diagnostic(
            "DEPTH_OF_FIELD",
            INFO,
            f"{n} views from points of the aperture (radius {radius * 1000:.3g} mm, focus "
            f"{focus:.4g} m) were averaged; depth and objects are the pinhole view's",
            "dof_samples",
        ),
    ]
    return base


def _finish_raster(
    rs, settings, antialiasing, plan, adapter, limits, requested, diagnostics, frame,
    aspect, t_start, tonemap, wanted, msaa, supersample,
) -> RenderResult:  # fmt: skip
    """The rasterizers' finish: the GPU (``gpu``, ``auto``) or the CPU renderer."""
    if requested in ("gpu", "auto"):
        try:
            return _finish_gpu(
                rs,
                settings,
                antialiasing,
                plan,
                adapter,
                limits,
                requested,
                diagnostics,
                frame,
                aspect,
                t_start,
                tonemap,
                wanted,
            )
        except RenderError as exc:
            if requested == "gpu" or not exc.code.startswith("GPU_"):
                raise
            diagnostics.append(
                Diagnostic(
                    "BACKEND_FALLBACK",
                    INFO,
                    f"backend 'auto' used the CPU renderer: [{exc.code}] {exc.message}",
                    "backend",
                )
            )
            if msaa is not None or supersample is not None:
                diagnostics.append(
                    Diagnostic(
                        "GPU_OPTIONS_IGNORED",
                        WARNING,
                        "msaa/supersample apply to the GPU backend and were ignored; the CPU "
                        f"renderer used {settings.samples}x{settings.samples} samples per pixel",
                        "backend",
                    )
                )
            if plan is not None and plan.maps:
                diagnostics.append(
                    Diagnostic(
                        "SHADOWS_NOT_RENDERED",
                        WARNING,
                        "shadows need the GPU backend; the CPU renderer rendered without them",
                        "shadows",
                    )
                )
    return _finish_cpu(
        rs, settings, limits, requested, diagnostics, frame, aspect, t_start, tonemap
    )


def _objects(rs: RenderScene, visible: np.ndarray) -> list[dict[str, Any]]:
    shared = rs.shared_cache()  # the rows depend on the instances and meshes only
    rows = None if shared is None else shared.get("result.objects")
    if rows is None and shared is not None and "changed" in shared:
        # The frame before's instances, some with another transform or color (the same ids,
        # names and meshes): the same rows.
        rows = shared["previous"].get("result.objects")
        if rows is not None:
            shared["result.objects"] = rows
    if rows is None:
        rows = [
            (k, inst.object_id, inst.name, int(rs.meshes[inst.mesh].triangles.shape[0]))
            for k, inst in enumerate(rs.instances)
        ]
        if shared is not None:
            shared["result.objects"] = rows
    return [
        {"index": k, "id": object_id, "name": name, "triangles": triangles, "visible_samples": v}
        for (k, object_id, name, triangles), v in zip(
            rows, np.asarray(visible[: len(rows)], dtype=np.int64).tolist(), strict=True
        )
    ]


def _finish_cpu(
    rs: RenderScene,
    settings: RenderSettings,
    limits: RenderLimits,
    requested: str,
    diagnostics: list[Diagnostic],
    frame: bool,
    aspect: float,
    t_start: float,
    tonemap: str = "none",
) -> RenderResult:
    frame_data = cpu.render_scene(
        rs, settings.width, settings.height, settings.samples, settings.cull, limits
    )
    diagnostics += frame_data.diagnostics

    t0 = time.perf_counter()
    linear = frame_data.linear
    behind = cpu.environment_background(rs, settings.width, settings.height, aspect)
    if behind is not None:  # an image environment behind the geometry
        linear += ((1.0 - frame_data.coverage)[..., None] * behind).astype(linear.dtype)
    if settings.exposure != 0.0:
        linear *= 2.0**settings.exposure
    image = display(linear, tonemap)  # in row blocks: bounded temporaries
    t_output = time.perf_counter() - t0

    counts = frame_data.counts
    timings = {
        "evaluate": rs.timings.get("evaluate", 0.0),
        "textures": rs.timings.get("textures", 0.0),
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
        objects=_objects(rs, frame_data.visible_samples),
        camera=cam_info,
        settings=settings.to_dict(),
        stats=stats,
        diagnostics=diagnostics,
        backend={"name": "cpu", "requested": requested},
        color=color_info(settings.exposure, tonemap),
    )


def _finish_gpu(
    rs: RenderScene,
    settings: RenderSettings,
    antialiasing: Antialiasing,
    plan: ShadowPlan | None,
    adapter: Any,
    limits: RenderLimits,
    requested: str,
    diagnostics: list[Diagnostic],
    frame: bool,
    aspect: float,
    t_start: float,
    tonemap: str = "none",
    outputs: frozenset[str] = OUTPUTS,
) -> RenderResult:
    from .gpu import device as gpu_device
    from .gpu import renderer as gpu_renderer

    t0 = time.perf_counter()
    context, failures = gpu_device.get_context(adapter)
    t_init = time.perf_counter() - t0
    fresh = context.renders == 0
    gpu_frame = gpu_renderer.render_scene(
        context,
        rs,
        width=settings.width,
        height=settings.height,
        antialiasing=antialiasing,
        cull=settings.cull,
        exposure=settings.exposure,
        shadows=plan,
        limits=limits,
        outputs=outputs,
    )
    notes: list[Diagnostic] = []
    info = context.info
    for failure in failures:
        notes.append(
            Diagnostic(
                "ADAPTER_FALLBACK",
                WARNING,
                f"{failure['adapter']} could not be used ([{failure['code']}] "
                f"{failure['message']}); rendered on {info.label} instead",
                "adapter",
            )
        )
    if requested == "auto":
        notes.append(
            Diagnostic(
                "BACKEND_SELECTED",
                INFO,
                f"backend 'auto' rendered on the GPU: {info.label}",
                "backend",
            )
        )
    if info.software:
        notes.append(
            Diagnostic(
                "SOFTWARE_ADAPTER",
                INFO,
                f"{info.name} is a software implementation of {info.backend}: the GPU "
                "pipeline ran on the CPU, not on graphics hardware",
                "adapter",
            )
        )
    if plan is not None:
        notes += list(plan.diagnostics)
    diagnostics += notes + gpu_frame.diagnostics
    if gpu_frame.nonfinite_pixels:  # counted on the GPU: a NaN or +inf in linear light
        raise RenderError("RENDER_FAILED", "the rendered image contains non-finite values")
    timings = {
        "evaluate": rs.timings.get("evaluate", 0.0),
        "textures": rs.timings.get("textures", 0.0),
        "extract": rs.timings.get("extract", 0.0),
        "init": t_init + gpu_frame.timings["compile"],
        "upload": gpu_frame.timings["upload"],
        "prepare": gpu_frame.timings["prepare"],
        "gpu": gpu_frame.timings["gpu"],
        "readback": gpu_frame.timings["readback"],
    }
    timings["total"] = time.perf_counter() - t_start
    stats = {
        **gpu_frame.counts,
        "pixels_covered": gpu_frame.covered_pixels,
        "timings": timings,
        "gpu_timings": gpu_frame.gpu_timings,
    }
    cam_info = rs.camera.to_dict(aspect)
    cam_info["temporary"] = frame
    backend_info = {
        "name": "gpu",
        "requested": requested,
        "api": "wgpu",
        "versions": gpu_device.wgpu_versions(),
        "adapter": info.to_dict(features=False),
        "antialiasing": antialiasing.to_dict(),
        "shadows": None if plan is None or not plan.maps else plan.to_dict(),
        "device_created": fresh,
        "device_init_seconds": context.init_seconds if fresh else 0.0,
        "resources": gpu_frame.resources,
    }
    behind = cpu.environment_background(rs, settings.width, settings.height, aspect)
    if behind is not None:  # an image environment behind the geometry (exposure applied)
        gain = 2.0**settings.exposure
        gpu_frame.linear += ((1.0 - gpu_frame.coverage)[..., None] * behind * gain).astype(
            gpu_frame.linear.dtype
        )
    image = (
        gpu_frame.image
        if tonemap == "none" and behind is None
        else display(gpu_frame.linear, tonemap)
    )
    return RenderResult(
        image=image,
        linear=gpu_frame.linear,
        depth=gpu_frame.depth,
        coverage=gpu_frame.coverage,
        object_index=gpu_frame.instance,
        objects=_objects(rs, gpu_frame.visible_samples),
        camera=cam_info,
        settings=settings.to_dict(),
        stats=stats,
        diagnostics=diagnostics,
        backend=backend_info,
        color=color_info(settings.exposure, tonemap),
    )


def _finish_path(
    rs: RenderScene,
    settings: RenderSettings,
    path: PathSettings,
    ao: AOSettings | None,
    limits: RenderLimits,
    diagnostics: list[Diagnostic],
    frame: bool,
    aspect: float,
    t_start: float,
    tonemap: str,
    progress: Any,
    backend: str = "path",
    adapter: Any = None,
    denoiser: Any = None,
) -> RenderResult:
    gpu_info: dict[str, Any] = {}
    if backend == "gpu_path":
        from .gpu import device as gpu_device
        from .gpu import pathtrace as gpu_pathtrace

        t0 = time.perf_counter()
        context, failures = gpu_device.get_context(adapter, path_tracer=True)
        t_init = time.perf_counter() - t0
        fresh = context.renders == 0
        data = gpu_pathtrace.render_scene(
            context,
            rs,
            settings.width,
            settings.height,
            path,
            ao,
            cull=settings.cull,
            limits=limits,
            progress=progress,
        )
        context.renders += 1
        info = context.info
        for failure in failures:
            diagnostics.append(
                Diagnostic(
                    "ADAPTER_FALLBACK",
                    WARNING,
                    f"{failure['adapter']} could not be used ([{failure['code']}] "
                    f"{failure['message']}); rendered on {info.label} instead",
                    "adapter",
                )
            )
        if info.software:
            diagnostics.append(
                Diagnostic(
                    "SOFTWARE_ADAPTER",
                    INFO,
                    f"{info.name} is a software implementation of {info.backend}: the GPU "
                    "path tracer ran on the CPU, not on graphics hardware",
                    "adapter",
                )
            )
        data.timings["init"] = t_init
        gpu_info = {
            "api": "wgpu",
            "versions": gpu_device.wgpu_versions(),
            "adapter": info.to_dict(features=False),
            "hardware_ray_tracing": False,
            "device_created": fresh,
            "device_init_seconds": context.init_seconds if fresh else 0.0,
            "dispatch": data.info["dispatch"],
            "gpu_bytes": data.info["gpu_bytes"],
        }
    else:
        from . import pathtrace

        data = pathtrace.render_scene(
            rs,
            settings.width,
            settings.height,
            path,
            ao,
            cull=settings.cull,
            limits=limits,
            progress=progress,
        )
    diagnostics += data.diagnostics
    t0 = time.perf_counter()
    linear = data.linear
    t_denoise = 0.0
    if denoiser is not None and path.integrator == "path":
        from .denoise import denoise as run_denoiser

        t1 = time.perf_counter()
        linear = run_denoiser(data.passes, denoiser)
        t_denoise = time.perf_counter() - t1
        diagnostics.append(
            Diagnostic(
                "DENOISED",
                INFO,
                "the image was denoised (an edge-avoiding filter: less noise, some bias -- fine "
                "shadows, caustics and reflections may blur); passes['beauty'] is the raw image",
                "denoise",
            )
        )
    if settings.exposure != 0.0:
        linear = linear * 2.0**settings.exposure
    if not math.isfinite(float(np.max(linear, initial=0.0))):
        raise RenderError("RENDER_FAILED", "the rendered image contains non-finite values")
    image = display(linear, tonemap)
    t_output = time.perf_counter() - t0
    timings = {
        "evaluate": rs.timings.get("evaluate", 0.0),
        "textures": rs.timings.get("textures", 0.0),
        "extract": rs.timings.get("extract", 0.0),
        **data.timings,
        "output": t_output,
    }
    if denoiser is not None:
        timings["denoise"] = t_denoise
    timings["total"] = time.perf_counter() - t_start
    counts = data.counts
    rays = sum(counts.get(k, 0) for k in ("camera_rays", "bounce_rays", "shadow_rays", "ao_rays"))
    stats = {
        **counts,
        "rays": rays + counts.get("center_rays", 0),
        "rays_per_second": rays / max(timings.get("trace", 0.0), 1e-9),
        "pixels_covered": int(np.count_nonzero(data.coverage > 0)),
        "noise": data.info["noise"],
        "timings": timings,
    }
    cam_info = rs.camera.to_dict(aspect)
    cam_info["temporary"] = frame
    if backend == "gpu_path":
        path_info = path.to_dict()
        path_info.pop("workers")
        backend_info = {
            "name": "gpu_path",
            "requested": "gpu_path",
            "device": "gpu",
            "implementation": "Aether3D GPU path tracer (wgpu compute; software BVH traversal)",
            "gpu": True,
            **gpu_info,
            **path_info,
        }
    else:
        stats["cpu_seconds"] = data.info["cpu_seconds"]
        backend_info = {
            "name": "path",
            "requested": "path",
            "device": "cpu",
            "implementation": "Aether3D path tracer (NumPy)",
            "gpu": False,
            "processes": data.info["workers"],
            "parallel": data.info["parallel"],
            **path.to_dict(),
        }
    backend_info.update(
        {
            "samples_rendered": counts["samples_per_pixel"],
            "ao": None if ao is None else ao.to_dict(),
            "environment": data.info["environment"],
            "lights": data.info["lights"],
            "bvh": data.info["bvh"],
            "work_estimate": data.info["work_estimate"],
            "denoiser": None if denoiser is None else denoiser.to_dict(),
        }
    )
    if settings.cull != "none":
        diagnostics.append(
            Diagnostic(
                "CULL_CAMERA_RAYS",
                INFO,
                f"cull={settings.cull!r} hides those faces from camera rays only; light still "
                "bounces off both sides of every double-sided surface",
                "cull",
            )
        )
    return RenderResult(
        image=image,
        linear=linear.astype(np.float32),
        depth=data.depth,
        coverage=data.coverage,
        object_index=data.instance,
        objects=_objects(rs, data.visible_samples),
        camera=cam_info,
        settings=settings.to_dict(),
        stats=stats,
        diagnostics=diagnostics,
        backend=backend_info,
        passes={k: v.astype(np.float32) for k, v in data.passes.items()},
        color=color_info(settings.exposure, tonemap),
    )
