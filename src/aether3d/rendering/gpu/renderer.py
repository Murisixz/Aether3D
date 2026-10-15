"""The GPU backend: a RenderScene in, the image and its auxiliary buffers out.

One command buffer per render, with three kinds of passes:

1. **shadow** (per shadowed directional light) -- the depth of every instance
   seen from the light, orthographic, into one layer of a ``depth32float``
   shadow-map array (depth only, no fragment shader);
2. **scene** -- every instance, multisampled (``msaa`` samples per pixel of an
   image ``supersample`` times larger per axis): linear radiance, object
   index and reversed depth, with native depth testing (``greater``) and
   hardware back/front-face culling;
3. **finalize** (compute) -- one invocation per output pixel: averages its
   samples in linear light (the resolve), applies exposure, encodes 8-bit
   sRGB and writes view depth, object index and coverage, and counts the
   visible samples of every object with atomics.

The outputs are copied to a mappable buffer. Mapping it waits until the GPU
has really finished, so the ``gpu`` timing is the synchronized time from
submission to completion (never just the submission). When the device
supports timestamp queries, the GPU time of each pass is reported too.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ... import _features
from ...diagnostics import INFO, WARNING, Diagnostic
from ...errors import RenderError
from ..backends import Antialiasing
from ..scene_data import RenderScene
from ..settings import RenderLimits
from ..shadows import ShadowPlan, instance_corners
from . import resources
from .device import GPUContext, log_mark, translate_errors, validation_messages_since
from .materials import HEADER_IMAGES, IMAGE_WORDS, image_key, pack_material_data
from .resources import (
    MAX_GPU_INSTANCES,
    draw_order,
    instance_masks,
    pack_finalize,
    pack_frame,
    pack_instances,
    pack_lights,
    pack_shadow_maps,
    shadow_draws,
    target_bytes,
)

WORKGROUP = 8
"""Finalize-pass workgroup size per axis (8 x 8 pixels)."""
MAX_COORDINATE = 1e30
"""Camera-relative coordinates beyond this would overflow float32 math; such objects are skipped."""


@dataclass
class GPUFrame:
    """Output of :func:`render_scene` at the final resolution."""

    image: np.ndarray  # (H, W, 3) uint8 sRGB, exposure applied
    linear: np.ndarray | None  # (H, W, 3) float32 linear light, exposure applied
    coverage: np.ndarray | None  # (H, W) float32
    depth: np.ndarray | None  # (H, W) float32 view depth at the representative sample
    instance: np.ndarray | None  # (H, W) int32 (-1: none)
    visible_samples: np.ndarray  # (instances,) int64
    nonfinite_pixels: int = 0  # pixels whose linear value has a NaN or +infinity
    covered_pixels: int = 0  # pixels with coverage > 0
    counts: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    gpu_timings: dict[str, float] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def check_request(
    context: GPUContext,
    rs: RenderScene,
    width: int,
    height: int,
    aa: Antialiasing,
    plan: ShadowPlan | None,
    limits: RenderLimits,
) -> None:
    """Raise before allocating anything if the device or the limits cannot take this render."""
    name = context.info.name
    supported = context.msaa_counts()
    if aa.msaa not in supported:
        raise RenderError(
            "GPU_FEATURE_UNSUPPORTED",
            f"{name} does not support {aa.msaa}x MSAA for Aether3D's render targets "
            f"(supported: {', '.join(map(str, supported))})",
            "msaa",
            hint="choose a supported msaa, or supersample instead",
            details={"supported_msaa": list(supported)},
        )
    max_side = context.limits.get("max-texture-dimension-2d", 8192)
    big = max(width, height) * aa.supersample
    if big > max_side:
        raise RenderError(
            "GPU_LIMIT_EXCEEDED",
            f"a {width}x{height} image with {aa.supersample}x supersampling needs "
            f"{big}-pixel textures; {name} allows at most {max_side}",
            "supersample" if aa.supersample > 1 else "width",
            hint="lower the resolution or supersample (MSAA does not enlarge the image)",
        )
    if plan is not None and plan.maps and plan.settings.map_size > max_side:
        raise RenderError(
            "GPU_LIMIT_EXCEEDED",
            f"shadow maps of {plan.settings.map_size} texels exceed {name}'s limit of {max_side}",
            "shadows.map_size",
        )
    samples = width * height * aa.supersample**2 * aa.msaa
    if samples > limits.max_samples:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"{width}x{height} pixels with {aa.samples_per_pixel} samples each need "
            f"{samples:,} samples (limit {limits.max_samples:,})",
            hint="lower the resolution, msaa or supersample",
        )
    if rs.triangle_count > limits.max_triangles:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"the scene has {rs.triangle_count:,} triangles to render "
            f"(limit {limits.max_triangles:,})",
            hint="hide objects or reduce subdivision levels",
        )
    if len(rs.instances) > MAX_GPU_INSTANCES:
        raise RenderError(
            "GPU_LIMIT_EXCEEDED",
            f"{len(rs.instances):,} objects exceed the GPU backend's limit of "
            f"{MAX_GPU_INSTANCES:,}",
        )
    color_bytes = 16 if context.color_format(aa.msaa) == "rgba32float" else 8
    layers = 0 if plan is None else len(plan.maps)
    size = 0 if plan is None else plan.settings.map_size
    estimate = target_bytes(width, height, aa.supersample, aa.msaa, color_bytes, (size, layers))
    if estimate > limits.max_gpu_memory:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"this render needs about {estimate / 2**20:,.0f} MiB of GPU memory for its "
            f"images (limit {limits.max_gpu_memory / 2**20:,.0f} MiB)",
            hint="lower the resolution, msaa, supersample or shadow map size, or raise "
            "RenderLimits.max_gpu_memory",
            details={"estimated_bytes": estimate},
        )


def _finite_instances(
    rs: RenderScene, eye: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, list[Diagnostic]]:
    """Which instances stay within float32 range relative to the camera."""
    if not rs.instances:
        return np.zeros(0, dtype=bool), []
    # Every corner is within |translation - eye| + |linear| @ max(|lo|, |hi|) of the camera.
    # When that bound (with a margin far above rounding) is in range for every instance,
    # so is every corner the exact test below would compute: it is skipped.
    arrays = rs.instance_arrays
    reach = np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1]))
    with np.errstate(over="ignore", invalid="ignore"):
        bound = np.abs(arrays.world[:, :3, 3] - eye) + np.sum(
            np.abs(arrays.world[:, :3, :3]) * reach[arrays.mesh][:, None, :], axis=2
        )
        if np.all(bound * (1.0 + 1e-9) < MAX_COORDINATE):  # NaN and inf fail: exact test
            return np.ones(len(rs.instances), dtype=bool), []
    corners = instance_corners(rs, bounds) - eye
    keep = np.all(np.isfinite(corners) & (np.abs(corners) < MAX_COORDINATE), axis=(1, 2))
    diagnostics = []
    if not keep.all():
        names = [rs.instances[i].name for i in np.flatnonzero(~keep)]
        diagnostics.append(
            Diagnostic(
                "NONFINITE_GEOMETRY",
                WARNING,
                f"{len(names)} object(s) extend beyond float32 range relative to the camera "
                f"and were skipped: {', '.join(names[:10])}",
                count=len(names),
            )
        )
    return keep, diagnostics


def render_scene(
    context: GPUContext,
    rs: RenderScene,
    *,
    width: int,
    height: int,
    antialiasing: Antialiasing,
    cull: str,
    exposure: float,
    shadows: ShadowPlan | None,
    limits: RenderLimits,
    outputs: frozenset[str] = resources.OUTPUTS,
) -> GPUFrame:
    """Render ``rs`` on ``context``'s device (see the module docstring). Only ``outputs``
    are read back (the image always); the others are ``None``."""
    with context.lock, translate_errors(context, "GPU rendering"):
        return _render(
            context, rs, width, height, antialiasing, cull, exposure, shadows, limits, outputs
        )


def _render(
    context: GPUContext,
    rs: RenderScene,
    width: int,
    height: int,
    aa: Antialiasing,
    cull: str,
    exposure: float,
    plan: ShadowPlan | None,
    limits: RenderLimits,
    outputs: frozenset[str] = resources.OUTPUTS,
) -> GPUFrame:
    wgpu = context.wgpu
    device = context.device
    t_start = time.perf_counter()
    mark = log_mark()
    compile_before = context.shaders.compile_seconds
    check_request(context, rs, width, height, aa, plan, limits)
    aspect = width / height
    eye = rs.camera.frame.position
    diagnostics: list[Diagnostic] = []
    if plan is not None and not plan.maps:
        plan = None

    # ---- uploads: meshes (cached) and per-frame data -----------------------------------
    t0 = time.perf_counter()
    uploads_before = (context.meshes.uploads, context.meshes.upload_bytes)
    first_user: dict[int, str] = {}
    for inst in rs.instances:
        first_user.setdefault(inst.mesh, inst.name)
    meshes = []
    for index, mesh in enumerate(rs.meshes):
        location = f"objects[{first_user[index]}]" if index in first_user else "meshes"
        gpu_mesh, _ = context.meshes.get(mesh, location)
        meshes.append(gpu_mesh)
    bounds = np.stack([m.bounds for m in meshes]) if meshes else np.zeros((0, 2, 3))
    keep, finite_diags = _finite_instances(rs, eye, bounds)
    diagnostics += finite_diags
    masked = instance_masks(rs)
    order, draws = draw_order(rs, cull, keep, masked)
    storage = wgpu.BufferUsage.STORAGE
    uniform = wgpu.BufferUsage.UNIFORM
    frame_buffers = []
    persistent = _features.enabled("AETHER3D_GPU_PERSISTENT_BUFFERS")
    writes_before = context.frame_buffers.writes
    written_before = context.frame_buffers.bytes_written
    bound: dict[int, int] = {}  # id(buffer): bytes of this render's data

    def buffer(data: np.ndarray, usage: int, label: str) -> Any:
        if persistent:
            kept, size = context.frame_buffers.upload(label, data, usage)
            bound[id(kept)] = size
            return kept
        created = device.create_buffer_with_data(label=f"aether3d-{label}", data=data, usage=usage)
        frame_buffers.append(created)
        bound[id(created)] = created.size
        return created

    def binding(buf: Any) -> dict[str, Any]:
        return {"buffer": buf, "offset": 0, "size": bound[id(buf)]}

    instance_buffer = buffer(pack_instances(rs, order, eye), storage, "instances")
    materials = pack_material_data(rs, order)
    textures_before = (context.textures.uploads, context.textures.upload_bytes)
    t_textures = time.perf_counter()
    texel_buffer, offsets, _ = context.textures.prepare(materials.images)
    t_textures = time.perf_counter() - t_textures
    words = materials.words
    for i, image in enumerate(materials.images):  # where each image's levels start in the pool
        words[words[HEADER_IMAGES] + i * IMAGE_WORDS] = offsets[image_key(image)]
    material_buffer = buffer(words, storage, "materials")
    light_buffer = buffer(pack_lights(rs, eye, plan), storage, "lights")
    shadow_buffer = buffer(pack_shadow_maps(plan), storage, "shadow-maps")
    frame_buffer = buffer(pack_frame(rs, aspect, plan), uniform, "frame")
    params_buffer = buffer(
        pack_finalize(width, height, aa.supersample, aa.msaa, rs, exposure),
        uniform,
        "finalize",
    )
    shadow_pass_buffers = [
        buffer(
            np.concatenate([m.matrix.T.reshape(-1), eye, [0.0]]).astype(np.float32),
            uniform,
            f"shadow-pass-{layer}",
        )
        for layer, m in enumerate(plan.maps if plan is not None else ())
    ]
    t_upload = time.perf_counter() - t0

    # ---- targets, bind groups, command encoding ---------------------------------------
    t0 = time.perf_counter()
    color_format = context.color_format(aa.msaa)
    layers = 0 if plan is None else len(plan.maps)
    map_size = 0 if plan is None else plan.settings.map_size
    targets, allocated = context.targets.get(
        width, height, aa.supersample, aa.msaa, color_format, len(rs.instances), map_size, layers
    )
    layout = targets.layout
    sizes = layout.sizes()
    utility = context.utility
    frame_group = device.create_bind_group(
        layout=context.shaders.layout("frame"),
        entries=[
            {"binding": 0, "resource": binding(frame_buffer)},
            {"binding": 1, "resource": binding(instance_buffer)},
            {"binding": 2, "resource": binding(light_buffer)},
            {"binding": 3, "resource": binding(shadow_buffer)},
            {"binding": 4, "resource": targets.shadow_view},
            {"binding": 5, "resource": utility.shadow_sampler},
            {"binding": 6, "resource": binding(material_buffer)},
            {"binding": 7, "resource": {"buffer": texel_buffer}},
        ],
    )
    multisampled = aa.msaa > 1
    finalize_name = "finalize-ms" if multisampled else "finalize"
    output_entries = [
        {"binding": b, "resource": {"buffer": targets.output, "offset": off, "size": size}}
        for b, off, size in (
            (5, layout.image, sizes["image"]),
            (6, layout.linear, sizes["linear"]),
            (7, layout.aux, sizes["aux"]),
            (8, layout.counts, sizes["counts"]),
        )
    ]
    finalize_group = device.create_bind_group(
        layout=context.shaders.layout(finalize_name),
        entries=[
            {"binding": 0, "resource": targets.color_view},
            {"binding": 1, "resource": targets.objects_view},
            {"binding": 2, "resource": targets.depth_view},
            {"binding": 3, "resource": binding(params_buffer)},
            {"binding": 4, "resource": {"buffer": utility.thresholds}},
            *output_entries,
        ],
    )
    pipelines = {
        False: context.shaders.scene_pipeline(aa.msaa, color_format, cull, False),
        True: context.shaders.scene_pipeline(aa.msaa, color_format, cull, True),
    }
    finalize_pipeline = context.shaders.finalize_pipeline(multisampled)
    query_set = targets.query_set
    first_query = 0 if layers else 2

    encoder = device.create_command_encoder(label="aether3d-render")
    encoder.clear_buffer(targets.output, layout.counts, resources._align(sizes["counts"], 4))
    draw_calls = 0
    if layers:
        shadow_runs = shadow_draws(order, draws, masked)
        shadow_pipelines = {False: context.shaders.shadow_pipeline()}
        if any(run[1] for run in shadow_runs):
            shadow_pipelines[True] = context.shaders.shadow_pipeline(masked=True)
        for layer, view in enumerate(targets.shadow_layers):
            group = device.create_bind_group(
                layout=context.shaders.layout("shadow"),
                entries=[
                    {"binding": 0, "resource": binding(shadow_pass_buffers[layer])},
                    {"binding": 1, "resource": binding(instance_buffer)},
                    {"binding": 6, "resource": binding(material_buffer)},
                    {"binding": 7, "resource": {"buffer": texel_buffer}},
                ],
            )
            writes = {}
            if query_set is not None:
                writes = {"query_set": query_set}
                if layer == 0:
                    writes["beginning_of_pass_write_index"] = 0
                if layer == layers - 1:
                    writes["end_of_pass_write_index"] = 1
            shadow_pass = encoder.begin_render_pass(
                label="aether3d-shadow-pass",
                color_attachments=[],
                depth_stencil_attachment={
                    "view": view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                },
                timestamp_writes=writes or None,
            )
            shadow_pass.set_bind_group(0, group)
            current_shadow = None
            for mesh, alpha_tested, first, count in shadow_runs:
                if current_shadow is not alpha_tested:
                    shadow_pass.set_pipeline(shadow_pipelines[alpha_tested])
                    current_shadow = alpha_tested
                shadow_pass.set_bind_group(1, meshes[mesh].bind_group)
                shadow_pass.draw(3 * meshes[mesh].triangles, count, 0, first)
                draw_calls += 1
            shadow_pass.end()

    background = rs.background
    scene_pass = encoder.begin_render_pass(
        label="aether3d-scene-pass",
        color_attachments=[
            {
                "view": targets.color_view,
                "clear_value": (
                    float(background[0]),
                    float(background[1]),
                    float(background[2]),
                    1.0,
                ),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": targets.objects_view,
                "clear_value": (0.0, 0.0, 0.0, 0.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            },
        ],
        depth_stencil_attachment={
            "view": targets.depth_view,
            "depth_clear_value": 0.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
        timestamp_writes=None
        if query_set is None
        else {
            "query_set": query_set,
            "beginning_of_pass_write_index": 2,
            "end_of_pass_write_index": 3,
        },
    )
    scene_pass.set_bind_group(0, frame_group)
    current = None
    for mesh, mirrored, first, count in draws:
        if current is not mirrored:
            scene_pass.set_pipeline(pipelines[mirrored])
            current = mirrored
        scene_pass.set_bind_group(1, meshes[mesh].bind_group)
        scene_pass.draw(3 * meshes[mesh].triangles, count, 0, first)
        draw_calls += 1
    scene_pass.end()

    finalize_pass = encoder.begin_compute_pass(
        label="aether3d-finalize",
        timestamp_writes=None
        if query_set is None
        else {
            "query_set": query_set,
            "beginning_of_pass_write_index": 4,
            "end_of_pass_write_index": 5,
        },
    )
    finalize_pass.set_pipeline(finalize_pipeline)
    finalize_pass.set_bind_group(0, finalize_group)
    finalize_pass.dispatch_workgroups(math.ceil(width / WORKGROUP), math.ceil(height / WORKGROUP))
    finalize_pass.end()
    if query_set is not None:
        count = 6 - first_query
        encoder.resolve_query_set(query_set, first_query, count, targets.query_buffer, 0)
        encoder.copy_buffer_to_buffer(
            targets.query_buffer, 0, targets.output, layout.queries, 8 * count
        )
    if outputs >= resources.OUTPUTS:
        encoder.copy_buffer_to_buffer(targets.output, 0, targets.staging, 0, layout.total)
    else:  # only what is read back crosses to the readable buffer
        for region in layout.regions(outputs):
            offset = getattr(layout, region)
            encoder.copy_buffer_to_buffer(
                targets.output, offset, targets.staging, offset, sizes[region]
            )
    commands = encoder.finish()
    # Pipelines compiled on first use are reported as "compile", not as preparation.
    t_prepare = time.perf_counter() - t0 - (context.shaders.compile_seconds - compile_before)

    # ---- execute and read back --------------------------------------------------------
    t0 = time.perf_counter()
    context.queue.submit([commands])
    targets.staging.map_sync(wgpu.MapMode.READ)  # returns once the GPU work is complete
    t_gpu = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        raw = np.frombuffer(targets.staging.read_mapped(copy=False), dtype=np.uint8)
        decoded = layout.decode(raw, outputs)
        del raw
    finally:
        targets.staging.unmap()
    t_readback = time.perf_counter() - t0
    for created in frame_buffers:
        created.destroy()
    context.meshes.trim({id(m) for m in meshes})
    context.renders += 1

    timings = {
        "upload": t_upload,
        "prepare": t_prepare,
        "gpu": t_gpu,
        "readback": t_readback,
        "compile": context.shaders.compile_seconds - compile_before,
    }
    gpu_timings: dict[str, float] = {}
    stamps = decoded["timestamps"][: 6 - first_query]  # only the resolved queries
    if query_set is not None and stamps.size == 6 - first_query:
        named = dict(zip(range(first_query, 6), stamps.tolist(), strict=True))

        def span(a: int, b: int) -> float | None:
            if a in named and b in named and named[b] >= named[a] > 0:
                return (named[b] - named[a]) * 1e-9
            return None

        for name, (a, b) in (("shadow", (0, 1)), ("scene", (2, 3)), ("finalize", (4, 5))):
            value = span(a, b)
            if value is not None:
                gpu_timings[name] = value
    for level, text in validation_messages_since(mark):
        diagnostics.append(
            Diagnostic(
                "GPU_VALIDATION_MESSAGE",
                WARNING if level in ("WARNING", "ERROR", "CRITICAL") else INFO,
                " ".join(text.split())[:500],
                "gpu",
            )
        )
    uploads, upload_bytes = (
        context.meshes.uploads - uploads_before[0],
        context.meshes.upload_bytes - uploads_before[1],
    )
    counts = {
        "instances": len(rs.instances),
        "meshes": len(rs.meshes),
        "lights": len(rs.lights),
        "triangles": int(rs.triangle_count),
        "samples": width * height * aa.samples_per_pixel,
        "samples_covered": int(decoded["counts"].sum()),
        "draw_calls": draw_calls,
        "shadow_maps": layers,
    }
    resources_info = {
        "mesh_uploads": uploads,
        "mesh_upload_bytes": upload_bytes,
        "mesh_cache": context.meshes.stats(),
        "materials": len(rs.materials),
        "textured_materials": materials.textured,
        "texture_bindings": materials.bindings,
        "texture_program_instructions": materials.instructions,
        "material_data_bytes": int(materials.words.nbytes),
        "images": len(materials.images),
        "texel_uploads": context.textures.uploads - textures_before[0],
        "texel_upload_bytes": context.textures.upload_bytes - textures_before[1],
        "texel_upload_seconds": t_textures,
        "texel_pool": context.textures.stats(),
        "target_bytes": targets.nbytes,
        "targets_allocated": allocated,
        "frame_buffer_bytes": int(sum(bound.values())),
        "frame_buffers_kept": persistent,
        "frame_buffer_writes": context.frame_buffers.writes - writes_before,
        "frame_buffer_bytes_written": context.frame_buffers.bytes_written - written_before,
        "color_format": color_format,
        "pipelines": context.shaders.pipeline_count,
    }
    timings["total_gpu_backend"] = time.perf_counter() - t_start
    return GPUFrame(
        image=decoded["image"],
        linear=decoded["linear"],
        coverage=decoded["coverage"],
        depth=decoded["depth"],
        instance=decoded["object_index"],
        visible_samples=decoded["counts"],
        nonfinite_pixels=decoded["nonfinite_pixels"],
        covered_pixels=decoded["covered_pixels"],
        counts=counts,
        timings=timings,
        gpu_timings=gpu_timings,
        resources=resources_info,
        diagnostics=diagnostics,
    )
