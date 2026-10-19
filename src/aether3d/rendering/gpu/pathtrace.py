"""The GPU path tracer (``backend="gpu_path"``): the CPU path tracer's estimator on the GPU.

The paths are traced by one compute kernel (``wgsl/pathtrace.wgsl``) on the
selected adapter -- with the RTX 4060 here, on its shader cores through
Vulkan. Everything else comes from the CPU path tracer
(:mod:`aether3d.rendering.pathtrace`), so both backends render the same
scene description:

* the scene preparation (:class:`~aether3d.rendering.pathtrace.scene.PathScene`):
  the two-level BVH (with its BLAS cache), the emitter table and the
  environment model;
* the materials: the GPU rasterizer's packed material data and texel pool,
  evaluated by the same WGSL as on the rasterizer;
* the estimator, its constants and its random numbers (see the kernel's
  header): with the same settings, the GPU traces the CPU's paths.

Traversal is done in software by the kernel (the BVH in storage buffers); it
does not use hardware ray tracing units.

The image is rendered in sample passes, each one dispatch per tile of
pixels; every invocation adds its pixel's samples to running sums on the GPU
(float32), in sample order. Ray counts and per-object visibility come back
after every dispatch (the counters are 32-bit). At the end, a pixel-center
pass gives depth and object index, and the sums are read back and turned into
the same :class:`~aether3d.rendering.pathtrace.integrator.PathFrame` the CPU
path tracer returns.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ...diagnostics import INFO, WARNING, Diagnostic
from ...errors import RenderError
from ..pathtrace.integrator import PathFrame, estimate_work
from ..pathtrace.lights import LightSet
from ..pathtrace.scene import MAX_LAYERS, PathScene
from ..raytrace.accel import KIND_INNER, _round_out
from ..scene_data import RenderScene
from ..settings import AOSettings, PathSettings, RenderLimits
from ..surface import LUMINANCE
from ..tangents import corner_tangents, meshes_needing_tangents
from .device import GPUContext, translate_errors
from .materials import HEADER_IMAGES, IMAGE_WORDS, image_key, pack_material_data
from .resources import link_masks, pack_instances

WORKGROUP = 8
STACK_SIZE = 64
"""Traversal stack entries per ray in the kernel (``PT_STACK``)."""
ACC_PLANES = 22
"""Per-pixel float32 sums (see the kernel's ``ACC_*``)."""
COUNTER_SLOTS = 8
HEADER_WORDS = 16
INST_WORDS = 24
LIGHT_WORDS = 44
EMITTER_WORDS = 12
PARAMS_WORDS = 128  # 68 + struct Atmo (48) + lens, aperture, film (4 each)
TILE = 512
"""Largest tile side of one dispatch (bounds the work and the 32-bit counters per dispatch)."""
MAX_PIXEL_SAMPLES_PER_DISPATCH = 1 << 20
"""Pixel samples one dispatch may trace when no progress callback asks for every pass."""
MODE_PATHS, MODE_CENTER, MODE_RNG = 0, 1, 2
FLAG_PERSPECTIVE, FLAG_PASS_THROUGH, FLAG_AO, FLAG_SIZED_LIGHTS = 1, 2, 4, 8
ENV_KINDS = {"ambient": 0, "uniform": 1, "sky": 2, "image": 3}
CULLS = {"none": 0, "back": 1, "front": 2}
RAY_COUNTERS = (
    "camera_rays",
    "bounce_rays",
    "shadow_rays",
    "ao_rays",
    "center_rays",
    "pass_through_rays",
    "layer_limited",
)


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


def _u32(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint32).reshape(-1)


def _f32_words(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).reshape(-1).view(np.uint32)


def _nonempty(words: np.ndarray, size: int = 4) -> np.ndarray:
    """Storage bindings must not be empty."""
    return words if words.size else np.zeros(size, dtype=words.dtype)


def pack_meshes(rs: RenderScene) -> tuple[list[np.ndarray], np.ndarray]:
    """The meshes concatenated (common.wgsl's mesh buffers, global vertex indices) and the
    first triangle of each mesh."""
    positions, normals, indices, uvs, groups = [], [], [], [], []
    tri_base = np.zeros(len(rs.meshes) + 1, dtype=np.int64)
    vertex_base = 0
    for j, mesh in enumerate(rs.meshes):
        p = np.asarray(mesh.positions, dtype=np.float32).reshape(-1, 3)
        t = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        positions.append(p.reshape(-1))
        n = mesh.normals
        normals.append((np.zeros_like(p) if n is None else np.asarray(n, np.float32)).reshape(-1))
        indices.append((t + vertex_base).astype(np.uint32).reshape(-1))
        uv = mesh.uvs
        uvs.append(
            (
                np.zeros((p.shape[0], 2), np.float32) if uv is None else np.asarray(uv, np.float32)
            ).reshape(-1)
        )
        g = mesh.groups
        groups.append(np.zeros(t.shape[0], np.uint32) if g is None else _u32(g))
        tri_base[j + 1] = tri_base[j] + t.shape[0]
        vertex_base += p.shape[0]
    if vertex_base >= 2**32:
        raise RenderError(
            "GPU_LIMIT_EXCEEDED", "the scene has more vertices than the GPU path tracer indexes"
        )

    if any(m.colors is not None for m in rs.meshes):
        # Vertex colours follow every UV, in the same buffer (common.wgsl's mesh_color):
        # 4 floats per vertex of every mesh, white for meshes without colours.
        uvs = uvs + [
            (
                np.ones((m.positions.shape[0], 4), np.float32)
                if m.colors is None
                else np.asarray(m.colors, np.float32)
            ).reshape(-1)
            for m in rs.meshes
        ]

    def join(parts: list[np.ndarray], dtype) -> np.ndarray:
        return _nonempty(np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype))

    # MikkTSpace corner tangents, only when some mesh is drawn with a UV normal map.
    needed = meshes_needing_tangents(rs)
    tangents = []
    if needed:
        for j, mesh in enumerate(rs.meshes):
            count = int(tri_base[j + 1] - tri_base[j])
            t = corner_tangents(mesh) if j in needed else None
            tangents.append(
                np.zeros(count * 12, np.float32)
                if t is None
                else np.asarray(t, np.float32).reshape(-1)
            )
    buffers = [
        join(positions, np.float32),
        join(normals, np.float32),
        join(indices, np.uint32),
        join(uvs, np.float32),
        join(groups, np.uint32),
        join(tangents, np.float32),
    ]
    return buffers, tri_base


def _light_records(rs: RenderScene, lights: LightSet, eye: np.ndarray) -> np.ndarray:
    """Analytic lights, with the cone quantities computed in float64 (see the kernel)."""
    data = np.zeros((max(lights.count, 1), LIGHT_WORDS), dtype=np.uint32)
    f = data.view(np.float32)
    for k, light in enumerate(rs.lights):
        f[k, 4:7] = lights.radiance[k]
        if lights.kind[k] == 0:
            half = math.radians(light.angle) * 0.5
            sized = half > 0.0 and lights.cos_max[k] < 1.0
            data[k, 0] = 0
            data[k, 1] = int(sized)
            f[k, 12:15] = lights.to_light[k]
            f[k, 15] = lights.cos_max[k]
            omc = 2.0 * math.sin(half * 0.5) ** 2  # 1 - cos(half), without cancellation
            f[k, 16] = omc
            if sized:
                f[k, 17] = 1.0 / (2.0 * math.pi * omc)
                f[k, 18:21] = lights.radiance[k] / math.sin(half) ** 2
            f[k, 21] = 2.0 / (1.0 + lights.cos_max[k])
        elif lights.kind[k] == 3:
            data[k, 0] = 3
            data[k, 1] = 1  # BSDF rays find it
            f[k, 8:11] = light.position - eye
            f[k, 24:27] = light.direction
            data[k, 29] = int(light.shape in ("disk", "ellipse"))
            f[k, 30] = light.area
            f[k, 31:34] = light.axis_x
            f[k, 34:37] = light.axis_y
            f[k, 37:40] = light.area_radiance
            f[k, 40] = light.spread_tan
            f[k, 41] = light.spread_norm
        else:
            data[k, 0] = lights.kind[k]
            data[k, 1] = int(lights.radius[k] > 0.0)
            data[k, 2] = int(lights.inverse_square[k])
            data[k, 3] = int(np.isfinite(lights.range[k]))
            f[k, 7] = lights.range[k] if np.isfinite(lights.range[k]) else 0.0
            f[k, 8:11] = lights.position[k] - eye
            f[k, 11] = lights.radius[k]
            if lights.kind[k] == 2:
                f[k, 24:27] = light.direction
                f[k, 27] = light.cos_outer
                f[k, 28] = light.spot_smooth
        data[k, 22] = 0 if lights.shadow is None or lights.shadow[k] else 1
    return data


def pack_scene_words(
    ps: PathScene, lights: LightSet, eye: np.ndarray, mesh_tri_base: np.ndarray
) -> np.ndarray:
    """The kernel's ``pt`` words: header, BVH, instances, lights and emitters."""
    accel = ps.accel
    rs = ps.rs
    n_nodes = accel.child.shape[0]
    nodes = np.zeros((max(n_nodes, 1), 4), dtype=np.uint32)
    inner = accel.kind == KIND_INNER
    nodes[:n_nodes, 0] = np.where(inner, accel.child[:, 0], accel.first)
    nodes[:n_nodes, 1] = np.where(inner, accel.child[:, 1], accel.count)
    nodes[:n_nodes, 2] = accel.kind
    lo = accel.child_lo.astype(np.float64).copy()
    hi = accel.child_hi.astype(np.float64).copy()
    top = accel.top_nodes
    # Camera-relative top-level boxes, rounded outward again (bottom levels are in object space).
    lo32, hi32 = accel.child_lo.copy(), accel.child_hi.copy()
    lo32[:top], hi32[:top] = _round_out(lo[:top] - eye, hi[:top] - eye)
    boxes = np.concatenate([lo32[:, 0], hi32[:, 0], lo32[:, 1], hi32[:, 1]], axis=1)
    tris = np.concatenate([accel.v0, accel.v1, accel.v2], axis=1)
    n_inst = len(rs.instances)
    inst = np.zeros((max(n_inst, 1), INST_WORDS), dtype=np.uint32)
    fi = inst.view(np.float32)
    if n_inst:
        linear = accel.to_object[:, :, :3]
        to_object = accel.to_object.copy()
        to_object[:, :, 3] += np.einsum("kij,j->ki", linear, eye)  # camera-relative input points
        fi[:n_inst, 0:12] = to_object.reshape(n_inst, 12)
        box_lo, box_hi = _round_out(accel.inst_lo - eye, accel.inst_hi - eye)
        fi[:n_inst, 12:15] = box_lo
        fi[:n_inst, 15:18] = box_hi
        inst[:n_inst, 18] = accel.inst_root
        inst[:n_inst, 19] = mesh_tri_base[[i.mesh for i in rs.instances]]
        inst[:n_inst, 20] = ps.tri_offset[:n_inst]
        inst[:n_inst, 21], inst[:n_inst, 22] = link_masks(rs, np.arange(n_inst))
    em = lights.emitters
    if em is not None:
        emitters = np.zeros((em.count, EMITTER_WORDS), dtype=np.uint32)
        fe = emitters.view(np.float32)
        emitters[:, 0] = em.inst
        emitters[:, 1] = em.prim
        emitters[:, 2] = em.material
        emitters[:, 3] = em.double_sided
        fe[:, 4:7] = em.normal
        fe[:, 7] = em.area
        fe[:, 8] = em.pmf
        cdf = em.cdf.astype(np.float32)
        lookup = np.where(em.lookup >= 0, em.lookup, 0xFFFFFFFF).astype(np.uint32)
    else:
        emitters = np.zeros(EMITTER_WORDS, dtype=np.uint32)
        cdf = np.zeros(1, dtype=np.float32)
        lookup = np.full(1, 0xFFFFFFFF, dtype=np.uint32)
    regions = [
        nodes.reshape(-1),
        _f32_words(boxes),
        _nonempty(_f32_words(tris)),
        _nonempty(_u32(accel.tri_prim)),
        _nonempty(_u32(accel.slot_inst)),
        inst.reshape(-1),
        _light_records(rs, lights, eye).reshape(-1),
        emitters.reshape(-1),
        cdf.view(np.uint32),
        lookup,
        _environment_words(ps.environment),
    ]
    header = np.zeros(HEADER_WORDS, dtype=np.uint32)
    offset = HEADER_WORDS
    for k, region in enumerate(regions):
        header[k] = offset
        offset += region.shape[0]
    return np.concatenate([header, *regions]).astype(np.uint32)


def _environment_words(env) -> np.ndarray:
    """An image environment's words: width, height, cells (nu, nv), its pixels (RGB
    float32, row 0 at the top), the marginal CDF and the conditional CDFs (float32); a
    placeholder for the other environments."""
    if env.kind != "image":
        return np.zeros(4, dtype=np.uint32)
    img = env.image
    t = img.tables()
    nv, nu = t["cells"]
    head = np.array([img.width, img.height, nu, nv], dtype=np.uint32)
    if t["black"]:
        marg = np.zeros(nv + 1, np.float32)
        cond = np.zeros((nv, nu + 1), np.float32)
    else:
        marg, cond = t["marginal"], t["conditional"]
    parts = [
        head,
        np.ascontiguousarray(img.pixels, dtype=np.float32).reshape(-1).view(np.uint32),
        np.ascontiguousarray(marg, dtype=np.float32).view(np.uint32),
        np.ascontiguousarray(cond, dtype=np.float32).reshape(-1).view(np.uint32),
    ]
    return np.concatenate(parts)


class Params:
    """The kernel's uniform ``Params`` (116 words; see pathtrace.wgsl)."""

    def __init__(self) -> None:
        self.words = np.zeros(PARAMS_WORDS, dtype=np.uint32)
        self.floats = self.words.view(np.float32)

    def vec(self, word: int, value) -> None:
        self.floats[word : word + 3] = value

    def f(self, word: int, value: float) -> None:
        self.floats[word] = value

    def u(self, word: int, value: int) -> None:
        self.words[word] = value


def base_params(
    ps: PathScene,
    lights: LightSet,
    width: int,
    height: int,
    path: PathSettings,
    ao: AOSettings | None,
    cull: str,
) -> Params:
    rs = ps.rs
    cam = rs.camera.camera
    frame = rs.camera.frame
    hx, hy = cam.half_extents(width / height)
    env = ps.environment
    pr = Params()
    pr.vec(0, frame.position)
    pr.f(3, cam.near)
    pr.vec(4, frame.forward)
    pr.f(7, cam.far)
    pr.vec(8, frame.right)
    pr.f(11, hx)
    pr.vec(12, frame.up)
    pr.f(15, hy)
    pr.vec(16, env.uniform)
    pr.u(19, ENV_KINDS[env.kind])
    pr.vec(20, env.background)
    pr.u(23, lights.count)
    pr.vec(24, env.zenith)
    pr.u(27, 0 if lights.emitters is None else lights.emitters.count)
    pr.vec(28, env.horizon)
    pr.u(31, len(rs.instances))
    pr.vec(32, env.ground)
    sized = lights.sized  # suns, light spheres and area lights BSDF rays can find
    flags = (
        (FLAG_PERSPECTIVE if cam.is_perspective else 0)
        | (FLAG_PASS_THROUGH if ps.pass_through else 0)
        | (FLAG_AO if ao is not None else 0)
        | (FLAG_SIZED_LIGHTS if sized else 0)
    )
    pr.u(35, flags)
    pr.u(36, path.seed & 0xFFFFFFFF)
    pr.u(37, path.seed >> 32)
    pr.u(38, width)
    pr.u(39, height)
    pr.u(42, path.bounces)
    pr.u(43, path.light_samples)
    pr.u(44, path.russian_roulette)
    pr.u(45, 1 if path.integrator == "ao" else 0)
    pr.u(46, CULLS[cull])
    pr.f(48, path.clamp)
    pr.f(49, 1.0 / math.sqrt(path.samples))
    pr.u(50, 0 if ao is None else ao.samples)
    pr.f(51, 0.0 if ao is None else ao.distance)
    pr.f(52, 0.0 if ao is None else ao.strength)
    pr.u(53, ps.accel.stack_size)
    pr.u(58, width * height)
    root_lo, root_hi = _round_out(
        ps.accel.root_lo - frame.position, ps.accel.root_hi - frame.position
    )
    pr.vec(60, root_lo)
    pr.vec(64, root_hi)
    from .resources import pack_atmosphere

    pr.words[68:116] = pack_atmosphere(rs).view(np.uint32)
    ox, oy = cam.ndc_offset(width / height)
    pr.floats[116:120] = (ox, oy, cam.aperture_radius(), rs.camera.focus_distance)
    rot = math.radians(cam.aperture_rotation)
    pr.floats[120:124] = (
        float(cam.aperture_blades),
        math.cos(rot),
        math.sin(rot),
        1.0 / cam.aperture_ratio,
    )
    if env.kind == "image":
        cos_r, sin_r = env.image.cos_sin
        pr.floats[124:127] = (cos_r, sin_r, env.image.intensity)
        pr.words[127] = (1 if env.visible else 0) | (2 if env.sampled else 0)
    return pr


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, min(TILE, width - x), min(TILE, height - y))
        for y in range(0, height, TILE)
        for x in range(0, width, TILE)
    ]


def check_request(
    context: GPUContext,
    ps: PathScene,
    width: int,
    height: int,
    limits: RenderLimits,
    byte_sizes: dict[str, int],
) -> None:
    """Raise before allocating GPU memory if the device or the limits cannot take this render."""
    name = context.info.name
    if ps.accel.stack_size > STACK_SIZE:
        raise RenderError(
            "GPU_LIMIT_EXCEEDED",
            f"the scene's BVH needs a traversal stack of {ps.accel.stack_size} entries; the GPU "
            f"path tracer has {STACK_SIZE}",
            hint="use backend='path' (the CPU path tracer has no such limit)",
        )
    binding = context.limits.get("max-storage-buffer-binding-size", 128 * 2**20)
    largest = context.limits.get("max-buffer-size", 256 * 2**20)
    for what, size in byte_sizes.items():
        if size > min(binding, largest):
            raise RenderError(
                "GPU_LIMIT_EXCEEDED",
                f"the GPU path tracer's {what} buffer needs {size / 2**20:,.1f} MiB; {name} "
                f"binds at most {min(binding, largest) / 2**20:,.1f} MiB",
                "width" if what == "accumulation" else None,
                hint="lower the resolution (the accumulation buffer holds 22 floats per pixel), "
                "use a GPU with larger buffers, or use backend='path'",
                details={"buffer": what, "bytes": size, "limit": min(binding, largest)},
            )
    total = sum(byte_sizes.values()) + byte_sizes.get("accumulation", 0)  # + its readback copy
    if total > limits.max_gpu_memory:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"this path-traced render needs about {total / 2**20:,.0f} MiB of GPU memory "
            f"(limit {limits.max_gpu_memory / 2**20:,.0f} MiB)",
            hint="lower the resolution, or raise RenderLimits.max_gpu_memory",
            details={"estimated_bytes": total},
        )


def render_scene(
    context: GPUContext,
    rs: RenderScene,
    width: int,
    height: int,
    path: PathSettings,
    ao: AOSettings | None,
    cull: str,
    limits: RenderLimits,
    progress: Any = None,
) -> PathFrame:
    """Path trace ``rs`` on ``context``'s GPU (see the module docstring)."""
    work = estimate_work(rs, width, height, path, ao)
    if work["rays"] > limits.max_path_rays:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"this path-traced render would trace up to {work['rays']:,} rays "
            f"(limit {limits.max_path_rays:,})",
            "path",
            hint="lower the resolution, samples, bounces or light_samples",
        )
    if rs.triangle_count > limits.max_triangles:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"the scene has {rs.triangle_count:,} triangles to render "
            f"(limit {limits.max_triangles:,})",
            hint="hide objects or reduce subdivision levels",
        )
    t_start = time.perf_counter()
    ps = PathScene(rs)
    lights = LightSet.build(ps)
    eye = np.asarray(rs.camera.frame.position, dtype=np.float64)
    meshes, mesh_tri_base = pack_meshes(rs)
    order = np.arange(len(rs.instances), dtype=np.int64)
    instance_words = pack_instances(rs, order, eye)
    # Global vertex indices here: every mesh's colours start after all UVs (pack_meshes).
    total_vertices = sum(int(m.positions.shape[0]) for m in rs.meshes)
    instance_words.view(np.uint32)[: len(rs.instances), 31] = np.where(
        instance_words.view(np.uint32)[: len(rs.instances), 29] & 16, 2 * total_vertices, 0
    )
    materials = pack_material_data(rs, order)
    scene_words = pack_scene_words(ps, lights, eye, mesh_tri_base)
    n_pix = width * height
    n_inst = len(rs.instances)
    byte_sizes = {
        "accumulation": ACC_PLANES * n_pix * 4,
        "scene": scene_words.nbytes,
        "materials": materials.words.nbytes,
        "instances": instance_words.nbytes,
        **{name: a.nbytes for name, a in zip(MESH_BUFFERS, meshes, strict=True)},
    }
    check_request(context, ps, width, height, limits, byte_sizes)
    t_setup = time.perf_counter() - t_start
    wgpu = context.wgpu
    device = context.device
    with context.lock, translate_errors(context, "GPU path tracing"):
        t0 = time.perf_counter()
        compile_before = context.shaders.compile_seconds
        pipeline = context.shaders.pathtrace_pipeline()
        storage = wgpu.BufferUsage.STORAGE
        copy = wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        owned: list[Any] = []

        def buffer(data: np.ndarray, label: str, usage: int = storage) -> Any:
            created = device.create_buffer_with_data(label=label, data=data, usage=usage)
            owned.append(created)
            return created

        texel_buffer, offsets, _ = context.textures.prepare(materials.images)
        words = materials.words
        for i, image in enumerate(materials.images):
            words[words[HEADER_IMAGES] + i * IMAGE_WORDS] = offsets[image_key(image)]
        group0_buffers = {
            1: buffer(instance_words, "aether3d-pt-instances"),
            6: buffer(words, "aether3d-pt-materials"),
            7: texel_buffer,
            8: buffer(scene_words, "aether3d-pt-scene"),
        }
        params_buffer = buffer(
            np.zeros(PARAMS_WORDS, np.uint32),
            "aether3d-pt-params",
            wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        accum = device.create_buffer(
            label="aether3d-pt-accumulation",
            size=ACC_PLANES * n_pix * 4,
            usage=storage | copy,
        )
        owned.append(accum)
        counters = device.create_buffer(
            label="aether3d-pt-counters",
            size=4 * (COUNTER_SLOTS + max(n_inst, 1)),
            usage=storage | copy,
        )
        owned.append(counters)
        mesh_buffers = [
            buffer(a, f"aether3d-pt-{name}") for name, a in zip(MESH_BUFFERS, meshes, strict=True)
        ]
        group0 = device.create_bind_group(
            layout=context.shaders.layout("pathtrace"),
            entries=[
                *({"binding": b, "resource": {"buffer": buf}} for b, buf in group0_buffers.items()),
                {"binding": 9, "resource": {"buffer": params_buffer}},
                {"binding": 10, "resource": {"buffer": accum}},
                {"binding": 11, "resource": {"buffer": counters}},
            ],
        )
        group1 = device.create_bind_group(
            layout=context.shaders.layout("pathtrace-mesh"),
            entries=[
                {"binding": b, "resource": {"buffer": buf}} for b, buf in enumerate(mesh_buffers)
            ],
        )
        t_upload = time.perf_counter() - t0
        params = base_params(ps, lights, width, height, path, ao, cull)
        tiles = _tiles(width, height)
        ray_counts = dict.fromkeys(RAY_COUNTERS, 0)
        visible = np.zeros(n_inst, dtype=np.int64)

        def dispatch(mode: int, tile, first: int, count: int) -> None:
            x, y, w, h = tile
            params.u(40, first)
            params.u(41, count)
            params.u(47, mode)
            params.u(54, x)
            params.u(55, y)
            params.u(56, w)
            params.u(57, h)
            context.queue.write_buffer(params_buffer, 0, params.words)
            encoder = device.create_command_encoder(label="aether3d-pathtrace")
            encoder.clear_buffer(counters)
            compute = encoder.begin_compute_pass(label="aether3d-pathtrace")
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group0)
            compute.set_bind_group(1, group1)
            compute.dispatch_workgroups(math.ceil(w / WORKGROUP), math.ceil(h / WORKGROUP))
            compute.end()
            context.queue.submit([encoder.finish()])
            raw = np.frombuffer(context.queue.read_buffer(counters), dtype=np.uint32)
            for k, name in enumerate(RAY_COUNTERS):
                ray_counts[name] += int(raw[k])
            visible[:] += raw[COUNTER_SLOTS : COUNTER_SLOTS + n_inst].astype(np.int64)

        # ---- sample passes ------------------------------------------------------------------
        t_trace = time.perf_counter()
        if progress is None:
            per = max(1, min(path.samples, MAX_PIXEL_SAMPLES_PER_DISPATCH // max(n_pix, 1)))
        else:
            per = 1
        done = 0
        stopped = False
        while done < path.samples:
            count = min(per, path.samples - done)
            for tile in tiles:
                dispatch(MODE_PATHS, tile, done, count)
            done += count
            if (
                progress is not None
                and progress(done, path.samples, time.perf_counter() - t_start) is False
                and done < path.samples
            ):
                stopped = True
                break
        t_trace = time.perf_counter() - t_trace

        # ---- pixel centers, read back -------------------------------------------------------
        t0 = time.perf_counter()
        for tile in tiles:
            dispatch(MODE_CENTER, tile, 0, 1)
        t_center = time.perf_counter() - t0
        t0 = time.perf_counter()
        planes = np.frombuffer(context.queue.read_buffer(accum), dtype=np.float32).reshape(
            ACC_PLANES, n_pix
        )
        t_readback = time.perf_counter() - t0
        for created in owned:
            created.destroy()
        compile_seconds = context.shaders.compile_seconds - compile_before

    frame = _frame(planes, done, width, height, path, ao, ray_counts, visible, ps, lights, stopped)
    frame.timings = {
        "setup": t_setup,
        "bvh": ps.timings["bvh"],
        "upload": t_upload,
        "trace": t_trace,
        "center": t_center,
        "readback": t_readback,
        "compile": compile_seconds,
    }
    slow = compile_diagnostic(compile_seconds)
    if slow is not None:
        frame.diagnostics.append(slow)
    frame.info.update(
        {
            "dispatch": {"tiles": len(tiles), "tile_size": TILE, "samples_per_dispatch": per},
            "work_estimate": work,
            "bvh": ps.accel.stats(),
            "lights": lights.to_dict(),
            "environment": ps.environment.to_dict(),
            "gpu_bytes": byte_sizes,
        }
    )
    return frame


MESH_BUFFERS = ("positions", "normals", "indices", "uvs", "groups", "tangents")

SLOW_COMPILE_SECONDS = 60.0


def compile_diagnostic(seconds: float) -> Diagnostic | None:
    """``SLOW_SHADER_COMPILE`` when compiling the kernel took over a minute.

    It is compiled once per process and device: about a second on Vulkan and Metal,
    12 seconds with DXC on GitHub's Windows runners.
    """
    if seconds < SLOW_COMPILE_SECONDS:
        return None
    return Diagnostic(
        "SLOW_SHADER_COMPILE",
        INFO,
        f"compiling the GPU path tracer took {seconds:.0f} s; later renders in this process "
        "reuse it",
        "path",
    )


def _frame(
    planes: np.ndarray,
    spp: int,
    width: int,
    height: int,
    path: PathSettings,
    ao: AOSettings | None,
    ray_counts: dict[str, int],
    visible: np.ndarray,
    ps: PathScene,
    lights: LightSet,
    stopped: bool,
) -> PathFrame:
    """The CPU path tracer's outputs (passes, coverage, noise, counts) from the GPU's sums."""
    n_pix = width * height
    shape = (height, width)
    inv = 1.0 / spp
    p = planes.astype(np.float64)

    def rgb(k: int) -> np.ndarray:
        return p[k : k + 3].T

    mean = rgb(14)
    if spp > 1:
        variance = np.maximum(rgb(17), 0.0) / (spp * (spp - 1))
    else:
        variance = np.full((n_pix, 3), np.inf)
    passes = {
        "beauty": mean.reshape(*shape, 3),
        "direct": (rgb(0) * inv).reshape(*shape, 3),
        "indirect": (rgb(3) * inv).reshape(*shape, 3),
        "albedo": (rgb(6) * inv).reshape(*shape, 3),
        "normal": (rgb(9) * inv).reshape(*shape, 3),
        "depth": planes[20].reshape(shape).copy(),
        "sample_count": np.full(shape, spp, dtype=np.int32),
        "variance": variance.reshape(*shape, 3),
    }
    passes["depth"][passes["depth"] >= 3.0e38] = np.inf
    if ao is not None:
        passes["ao"] = (p[12] * inv).reshape(shape)
    if path.integrator == "ao":
        passes["beauty"] = passes["direct"]
    instance = planes[21].view(np.uint32).astype(np.int64)
    instance[instance == 0xFFFFFFFF] = -1
    coverage = (p[13] * inv).astype(np.float32).reshape(shape)
    diagnostics = []
    if stopped:
        diagnostics.append(
            Diagnostic(
                "RENDER_STOPPED",
                INFO,
                f"the render stopped after {spp} of {path.samples} samples per pixel "
                "(progress callback)",
                "path",
            )
        )
    limited = ray_counts.pop("layer_limited")
    if limited:
        diagnostics.append(
            Diagnostic(
                "PASS_THROUGH_LIMIT",
                WARNING,
                f"{limited} rays crossed more than {MAX_LAYERS} cut-out or single-sided "
                "surfaces; they were treated as leaving the scene",
                "path",
            )
        )
    lum = (mean @ LUMINANCE).reshape(-1)
    err = np.sqrt(np.maximum((variance @ (LUMINANCE * LUMINANCE)).reshape(-1), 0.0))
    lit = lum > 1e-6
    measured = bool(lit.any()) and spp > 1
    noise = {
        "mean_standard_error": float(err[lit].mean()) if measured else None,
        "mean_relative_error": float((err[lit] / lum[lit]).mean()) if measured else None,
    }
    counts = {
        **{k: v for k, v in ray_counts.items() if k != "pass_through_rays"},
        "triangles": ps.triangle_count,
        "instances": len(ps.rs.instances),
        "lights": lights.count,
        "emissive_triangles": 0 if lights.emitters is None else lights.emitters.count,
        "paths": int(n_pix * spp),
        "samples_per_pixel": int(spp),
        "pass_through_rays": ray_counts["pass_through_rays"],
    }
    linear = passes["beauty"]
    passes = {k: v for k, v in passes.items()}
    return PathFrame(
        linear=linear.copy(),
        passes=passes,
        coverage=coverage,
        depth=passes["depth"].astype(np.float32),
        instance=instance.reshape(shape).astype(np.int32),
        visible_samples=visible,
        counts=counts,
        diagnostics=diagnostics,
        info={"noise": noise},
    )


def random_numbers(
    context: GPUContext, seed: int, width: int, height: int, sample: int
) -> np.ndarray:
    """Test hook: the kernel's random numbers for a few fixed decisions, ``(4, height, width)``.

    Planes 0-3 hold the numbers of (vertex 0, pixel u), (vertex 0, pixel v), (vertex 3,
    light slot 5) and (vertex 2, Russian roulette) of every pixel's path ``sample``.
    """
    wgpu = context.wgpu
    device = context.device
    n_pix = width * height
    with context.lock, translate_errors(context, "GPU random numbers"):
        pipeline = context.shaders.pathtrace_pipeline()
        storage = wgpu.BufferUsage.STORAGE
        empty = np.zeros(64, np.uint32)  # at least one element of every binding's type
        bufs = [device.create_buffer_with_data(data=empty, usage=storage) for _ in range(4)]
        scene = np.zeros(HEADER_WORDS + 4, np.uint32)
        bufs.append(device.create_buffer_with_data(data=scene, usage=storage))
        pr = Params()
        pr.u(36, seed & 0xFFFFFFFF)
        pr.u(37, seed >> 32)
        pr.u(38, width)
        pr.u(39, height)
        pr.u(40, sample)
        pr.u(41, 1)
        pr.u(47, MODE_RNG)
        pr.u(56, width)
        pr.u(57, height)
        pr.u(58, n_pix)
        params = device.create_buffer_with_data(data=pr.words, usage=wgpu.BufferUsage.UNIFORM)
        accum = device.create_buffer(
            size=ACC_PLANES * n_pix * 4, usage=storage | wgpu.BufferUsage.COPY_SRC
        )
        counters = device.create_buffer(size=4 * (COUNTER_SLOTS + 1), usage=storage)
        group0 = device.create_bind_group(
            layout=context.shaders.layout("pathtrace"),
            entries=[
                {"binding": 1, "resource": {"buffer": bufs[0]}},
                {"binding": 6, "resource": {"buffer": bufs[1]}},
                {"binding": 7, "resource": {"buffer": bufs[2]}},
                {"binding": 8, "resource": {"buffer": bufs[4]}},
                {"binding": 9, "resource": {"buffer": params}},
                {"binding": 10, "resource": {"buffer": accum}},
                {"binding": 11, "resource": {"buffer": counters}},
            ],
        )
        mesh = [device.create_buffer_with_data(data=empty, usage=storage) for _ in range(6)]
        group1 = device.create_bind_group(
            layout=context.shaders.layout("pathtrace-mesh"),
            entries=[{"binding": b, "resource": {"buffer": m}} for b, m in enumerate(mesh)],
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group0)
        compute.set_bind_group(1, group1)
        compute.dispatch_workgroups(math.ceil(width / WORKGROUP), math.ceil(height / WORKGROUP))
        compute.end()
        context.queue.submit([encoder.finish()])
        out = np.frombuffer(context.queue.read_buffer(accum), dtype=np.float32)
        return out.reshape(ACC_PLANES, height, width)[:4].copy()
