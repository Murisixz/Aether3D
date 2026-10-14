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
from ..diagnostics import INFO, WARNING, Diagnostic, worst_severity
from ..errors import RenderError
from . import cpu
from .backends import Antialiasing, antialiasing_plan, check_backend
from .camera import Camera, CameraFrame, frame_from_matrix
from .framing import AUTO_VIEW_DIRECTION, compute_framing, orientation_frame, target_points
from .png import encode_png
from .scene_data import (
    RenderCamera,
    RenderScene,
    extract_render_scene,
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

    ``depth`` and ``object_index`` describe one representative sample per pixel
    (the center sample for odd ``samples``; on the GPU with MSAA, the first
    MSAA sample of the center of the supersampled block; for the path tracer
    a ray through the pixel center).
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

    def png_bytes(self) -> bytes:
        """The image encoded as PNG.

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
        return encode_png(self.image, text=text)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the image to a file, atomically; returns the resolved path.

        ``.png`` writes the 8-bit display image; ``.hdr`` (Radiance RGBE) and
        ``.pfm`` (float32) write :attr:`linear` -- high dynamic range, after
        exposure and before tone mapping. The file is written to a temporary
        sibling and renamed into place, so a failure never leaves a partial
        or corrupted image.
        """
        from .._fileio import write_atomic
        from .hdr import encode_hdr, encode_pfm

        target = Path(path)
        suffix = target.suffix.lower()
        if suffix not in (".png", ".hdr", ".pfm"):
            raise RenderError(
                "INVALID_OUTPUT",
                f"output must be a .png, .hdr or .pfm file, got {str(target)!r}",
                "output",
            )
        t0 = time.perf_counter()
        if suffix == ".png":
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
) -> RenderResult:
    """Render ``scene`` and return a :class:`RenderResult`.

    * ``camera`` -- camera object, id or name; default: ``scene.active_camera``.
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
    requested = check_backend(backend)
    tonemap = check_tonemap(tonemap)
    shadow_settings = ShadowSettings.coerce(shadows)
    if requested != "path":
        for name, value in (("path", path), ("ao", ao), ("progress", progress)):
            if value is not None:
                raise RenderError(
                    "INVALID_SETTINGS",
                    f"{name} applies to the path tracer (backend='path'), not to "
                    f"backend={requested!r}",
                    name,
                    hint="use backend='path', or remove the option",
                )
    path_settings = ao_settings = None
    if requested == "path":
        for name, value in (
            ("msaa", msaa),
            ("supersample", supersample),
            ("adapter", adapter),
            ("samples", samples),
        ):
            if value is not None:
                hint = (
                    "set the path tracer's samples per pixel with path={'samples': N}"
                    if name == "samples"
                    else "remove the option (the path tracer runs on the CPU)"
                )
                raise RenderError(
                    "INVALID_SETTINGS",
                    f"{name} does not apply to the path tracer (backend='path')",
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
    plan = None if shadow_settings is None else plan_shadows(rs, shadow_settings, aspect)

    if requested == "path":
        return _finish_path(
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
        )
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
    return [
        {
            "index": k,
            "id": inst.object_id,
            "name": inst.name,
            "triangles": int(rs.meshes[inst.mesh].triangles.shape[0]),
            "visible_samples": int(visible[k]),
        }
        for k, inst in enumerate(rs.instances)
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
    if not math.isfinite(float(np.max(gpu_frame.linear, initial=0.0))):
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
        "pixels_covered": int(np.count_nonzero(gpu_frame.coverage > 0)),
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
    image = gpu_frame.image if tonemap == "none" else display(gpu_frame.linear, tonemap)
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
) -> RenderResult:
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
    backend_info = {
        "name": "path",
        "requested": "path",
        "device": "cpu",
        "implementation": "Aether3D path tracer (NumPy, single process)",
        "gpu": False,
        **path.to_dict(),
        "samples_rendered": counts["samples_per_pixel"],
        "ao": None if ao is None else ao.to_dict(),
        "environment": data.info["environment"],
        "lights": data.info["lights"],
        "bvh": data.info["bvh"],
        "work_estimate": data.info["work_estimate"],
    }
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
