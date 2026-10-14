"""GPU resources: cached mesh buffers, render targets, and the packed per-frame data.

**Meshes** are uploaded once per device and shared by every instance that
uses them (:class:`MeshCache`). A cache entry is keyed by the identity of the
mesh's NumPy arrays: :class:`~aether3d.Mesh` arrays are read-only, so the
same arrays always hold the same data, and editing or re-evaluating geometry
(a procedural parameter change, a committed edit session) produces a *new*
mesh. Entries hold only weak references; when a mesh is garbage collected its
buffers are freed at the next render. Arrays that were made writable again are
never cached. Least-recently-used entries are evicted beyond
:data:`DEFAULT_MESH_BUDGET` bytes (``AETHER3D_GPU_CACHE_MB``), but never meshes
of the render in progress.

**Targets** (multisampled color, object-index and depth images, shadow maps,
output and staging buffers) are reused while consecutive renders keep the
same size and settings (:class:`TargetCache`).

The ``pack_*`` functions build the exact byte layouts of the WGSL structs in
``wgsl/`` with NumPy; they need no GPU and are unit-tested on their own.
"""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...errors import RenderError
from ..color import srgb_to_linear
from ..scene_data import RenderMesh, RenderScene

if TYPE_CHECKING:
    from ..shadows import ShadowPlan
    from .device import GPUContext

DEFAULT_MESH_BUDGET = 1024 * 2**20
CACHE_ENV = "AETHER3D_GPU_CACHE_MB"

INSTANCE_FLOATS = 32  # 128 bytes, see `struct Instance` in common.wgsl
LIGHT_FLOATS = 12  # 48 bytes, `struct Light` in scene.wgsl
SHADOW_FLOATS = 20  # 80 bytes, `struct ShadowMap` in scene.wgsl
FRAME_FLOATS = 28  # 112 bytes, `struct Frame` in scene.wgsl
MIRRORED = 1
HAS_NORMALS = 2
MAX_GPU_INSTANCES = 2**24 - 1
"""Object indices travel through a float32 target, exact up to 2**24."""

ALIGN = 256  # buffer offsets (storage binding and query resolve alignment)


def _align(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def mesh_budget() -> int:
    raw = os.environ.get(CACHE_ENV, "").strip()
    if not raw:
        return DEFAULT_MESH_BUDGET
    try:
        megabytes = float(raw)
    except ValueError:
        raise RenderError(
            "INVALID_SETTINGS", f"{CACHE_ENV} must be a number of megabytes, got {raw!r}", CACHE_ENV
        ) from None
    return max(0, int(megabytes * 2**20))


# --------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------- #


@dataclass
class GPUMesh:
    """The storage buffers and bind group of one uploaded mesh."""

    positions: Any
    normals: Any | None  # None: the shared dummy buffer is bound instead
    indices: Any
    bind_group: Any
    vertices: int
    triangles: int
    nbytes: int
    bounds: np.ndarray  # (2, 3) float64: min and max corner

    def destroy(self) -> None:
        for buffer in (self.positions, self.normals, self.indices):
            if buffer is not None:
                buffer.destroy()


def _mesh_key(mesh: RenderMesh) -> tuple[int, int, int] | None:
    arrays = [mesh.positions, mesh.triangles] + ([mesh.normals] if mesh.normals is not None else [])
    if any(a.flags.writeable for a in arrays):
        return None  # mutable data: identity says nothing about content
    return (
        id(mesh.positions),
        id(mesh.triangles),
        id(mesh.normals) if mesh.normals is not None else 0,
    )


class MeshCache:
    """Uploaded meshes of one device, keyed by array identity (see the module docstring)."""

    def __init__(self, context: GPUContext) -> None:
        self.context = context
        self.budget = mesh_budget()
        self._entries: OrderedDict[tuple, tuple[list[weakref.ref], GPUMesh, int]] = OrderedDict()
        self._dead: list[tuple[tuple, int]] = []
        self._serial = 0
        self.resident_bytes = 0
        self.uploads = 0
        self.upload_bytes = 0
        self.hits = 0
        self.evictions = 0

    def stats(self) -> dict[str, int]:
        self._purge()
        return {
            "meshes": len(self._entries),
            "bytes": self.resident_bytes,
            "budget": self.budget,
            "uploads": self.uploads,
            "upload_bytes": self.upload_bytes,
            "hits": self.hits,
            "evictions": self.evictions,
        }

    def clear(self) -> None:
        for _, mesh, _ in self._entries.values():
            mesh.destroy()
        self._entries.clear()
        self._dead.clear()
        self.resident_bytes = 0

    def _forget(self, key: tuple, serial: int) -> None:
        self._dead.append((key, serial))  # called by the GC; buffers are freed by the next render

    def _purge(self) -> None:
        while self._dead:
            key, serial = self._dead.pop()
            entry = self._entries.get(key)
            if entry is not None and entry[2] == serial:  # not a newer entry reusing the ids
                del self._entries[key]
                entry[1].destroy()
                self.resident_bytes -= entry[1].nbytes

    def _check_limits(self, mesh: RenderMesh, location: str) -> None:
        limit = self.context.limits.get("max-storage-buffer-binding-size", 128 * 2**20)
        largest = max(mesh.positions.nbytes, mesh.triangles.nbytes)
        if largest > limit:
            raise RenderError(
                "GPU_LIMIT_EXCEEDED",
                f"a mesh with {mesh.positions.shape[0]:,} vertices and "
                f"{mesh.triangles.shape[0]:,} triangles needs a {largest:,}-byte buffer; "
                f"{self.context.info.name} binds at most {limit:,} bytes",
                location,
                hint="split the mesh, or render it with backend='cpu'",
            )

    def get(self, mesh: RenderMesh, location: str = "meshes") -> tuple[GPUMesh, bool]:
        """The uploaded mesh (uploading it if needed) and whether it was uploaded now."""
        self._purge()
        key = _mesh_key(mesh)
        if key is not None:
            entry = self._entries.get(key)
            if entry is not None:
                refs, gpu_mesh, _ = entry
                arrays = [mesh.positions, mesh.triangles, mesh.normals]
                if all(r() is a for r, a in zip(refs, arrays, strict=False)):
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return gpu_mesh, False
                self._entries.pop(key)  # an id was reused by different arrays
                gpu_mesh.destroy()
                self.resident_bytes -= gpu_mesh.nbytes
        self._check_limits(mesh, location)
        gpu_mesh = self._upload(mesh)
        self.uploads += 1
        self.upload_bytes += gpu_mesh.nbytes
        if key is not None:
            arrays = [mesh.positions, mesh.triangles] + (
                [mesh.normals] if mesh.normals is not None else []
            )
            self._serial += 1
            serial = self._serial
            refs = [weakref.ref(a, lambda _, k=key, n=serial: self._forget(k, n)) for a in arrays]
            self._entries[key] = (refs, gpu_mesh, serial)
            self.resident_bytes += gpu_mesh.nbytes
        return gpu_mesh, True

    def trim(self, keep: set[int]) -> None:
        """Evict least-recently-used meshes beyond the budget (never those with ids in ``keep``)."""
        self._purge()
        for key in list(self._entries):
            if self.resident_bytes <= self.budget:
                break
            if id(self._entries[key][1]) in keep:
                continue
            _, gpu_mesh, _ = self._entries.pop(key)
            gpu_mesh.destroy()
            self.resident_bytes -= gpu_mesh.nbytes
            self.evictions += 1

    def _upload(self, mesh: RenderMesh) -> GPUMesh:
        ctx = self.context
        wgpu = ctx.wgpu
        usage = wgpu.BufferUsage.STORAGE
        device = ctx.device
        positions = device.create_buffer_with_data(
            label="aether3d-positions",
            data=np.ascontiguousarray(mesh.positions, dtype=np.float32),
            usage=usage,
        )
        indices = device.create_buffer_with_data(
            label="aether3d-indices",
            data=np.ascontiguousarray(mesh.triangles, dtype=np.uint32),
            usage=usage,
        )
        normals = None
        if mesh.normals is not None:
            normals = device.create_buffer_with_data(
                label="aether3d-normals",
                data=np.ascontiguousarray(mesh.normals, dtype=np.float32),
                usage=usage,
            )
        bind_group = device.create_bind_group(
            layout=ctx.shaders.layout("mesh"),
            entries=[
                {"binding": 0, "resource": {"buffer": positions}},
                {"binding": 1, "resource": {"buffer": normals or ctx.utility.dummy_storage}},
                {"binding": 2, "resource": {"buffer": indices}},
            ],
        )
        nbytes = mesh.positions.nbytes + mesh.triangles.nbytes
        nbytes += 0 if mesh.normals is None else mesh.normals.nbytes
        bounds = np.stack([mesh.positions.min(axis=0), mesh.positions.max(axis=0)])
        return GPUMesh(
            positions,
            normals,
            indices,
            bind_group,
            int(mesh.positions.shape[0]),
            int(mesh.triangles.shape[0]),
            nbytes,
            bounds.astype(np.float64),
        )


# --------------------------------------------------------------------------- #
# Per-device utility buffers and reusable targets
# --------------------------------------------------------------------------- #


def srgb_thresholds() -> np.ndarray:
    """``t[c]``: the smallest float32 whose sRGB value rounds to code ``c`` or more.

    Each boundary ``srgb_to_linear((c - 0.5) / 255)`` is computed in float64
    and rounded *up* to float32 (``t[0] = -inf``), so for every float32 ``v`` the GPU's count
    of thresholds ``<= v`` equals the CPU's ``round(255 * srgb(v))``.
    """
    exact = np.empty(256, dtype=np.float64)
    exact[0] = -np.inf
    exact[1:] = srgb_to_linear((np.arange(1, 256) - 0.5) / 255.0)
    t = exact.astype(np.float32)
    below = t.astype(np.float64) < exact
    t[below] = np.nextafter(t[below], np.float32(np.inf))
    return t


class UtilityBuffers:
    """Small constant resources every render binds."""

    def __init__(self, context: GPUContext) -> None:
        self.context = context
        self._items: dict[str, Any] = {}

    def clear(self) -> None:
        for item in self._items.values():
            if hasattr(item, "destroy"):
                item.destroy()
        self._items.clear()

    def _get(self, name: str, create) -> Any:
        item = self._items.get(name)
        if item is None:
            item = self._items[name] = create()
        return item

    @property
    def dummy_storage(self) -> Any:
        wgpu = self.context.wgpu
        return self._get(
            "dummy",
            lambda: self.context.device.create_buffer(
                label="aether3d-dummy", size=256, usage=wgpu.BufferUsage.STORAGE
            ),
        )

    @property
    def thresholds(self) -> Any:
        wgpu = self.context.wgpu
        return self._get(
            "thresholds",
            lambda: self.context.device.create_buffer_with_data(
                label="aether3d-srgb", data=srgb_thresholds(), usage=wgpu.BufferUsage.STORAGE
            ),
        )

    @property
    def shadow_sampler(self) -> Any:
        wgpu = self.context.wgpu
        return self._get(
            "sampler",
            lambda: self.context.device.create_sampler(
                label="aether3d-shadow",
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                compare=wgpu.CompareFunction.less_equal,
            ),
        )


@dataclass(frozen=True)
class OutputLayout:
    """Byte layout of the finalize pass outputs in one storage (and staging) buffer."""

    width: int
    height: int
    instances: int
    timestamps: int
    image: int = 0
    linear: int = 0
    aux: int = 0
    counts: int = 0
    queries: int = 0
    total: int = 0

    @classmethod
    def build(cls, width: int, height: int, instances: int, timestamps: int) -> OutputLayout:
        n = width * height
        image = 0
        linear = _align(image + 4 * n)
        aux = _align(linear + 12 * n)
        counts = _align(aux + 12 * n)
        queries = _align(counts + 4 * max(1, instances))
        total = _align(queries + 8 * max(1, timestamps))
        return cls(width, height, instances, timestamps, image, linear, aux, counts, queries, total)

    def sizes(self) -> dict[str, int]:
        n = self.width * self.height
        return {
            "image": 4 * n,
            "linear": 12 * n,
            "aux": 12 * n,
            "counts": 4 * max(1, self.instances),
            "queries": 8 * max(1, self.timestamps),
        }

    def decode(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        """Copy the outputs out of the mapped bytes ``raw`` (uint8) into NumPy arrays."""
        h, w = self.height, self.width
        sizes = self.sizes()

        def part(name: str) -> np.ndarray:
            start = getattr(self, name)
            return raw[start : start + sizes[name]]

        image = part("image").reshape(h, w, 4)[..., :3].copy()
        linear = part("linear").view(np.float32).reshape(h, w, 3).copy()
        aux = part("aux").view(np.float32).reshape(h, w, 3)
        return {
            "image": image,
            "linear": linear,
            "depth": aux[..., 0].copy(),
            "object_index": aux[..., 1].copy().view(np.int32),
            "coverage": aux[..., 2].copy(),
            "counts": part("counts").view(np.uint32)[: self.instances].astype(np.int64),
            "timestamps": part("queries").view(np.uint64)[: self.timestamps].copy(),
        }


@dataclass
class FrameTargets:
    """Render targets and buffers for one image size and configuration."""

    key: tuple
    color: Any
    objects: Any
    depth: Any
    color_view: Any
    objects_view: Any
    depth_view: Any
    shadow: Any
    shadow_view: Any  # 2d-array view for sampling
    shadow_layers: list[Any]  # one 2d view per layer, for rendering
    output: Any
    staging: Any
    layout: OutputLayout
    query_set: Any | None
    query_buffer: Any | None
    nbytes: int

    def destroy(self) -> None:
        for item in (self.color, self.objects, self.depth, self.shadow, self.output, self.staging):
            item.destroy()
        if self.query_buffer is not None:
            self.query_buffer.destroy()
        if self.query_set is not None:
            self.query_set.destroy()


def target_bytes(
    width: int, height: int, supersample: int, msaa: int, color_bytes: int, shadow: tuple
) -> int:
    """Estimated GPU memory of one :class:`FrameTargets` (images, shadow maps, outputs)."""
    samples = width * supersample * height * supersample * msaa
    shadow_size, shadow_layers = shadow
    outputs = OutputLayout.build(width, height, 0, 0).total
    return (
        samples * (color_bytes + 4 + 4)
        + shadow_size * shadow_size * 4 * shadow_layers
        + 2 * outputs
    )


class TargetCache:
    """Keeps the targets of the last configuration for the next render."""

    def __init__(self, context: GPUContext) -> None:
        self.context = context
        self.current: FrameTargets | None = None
        self.allocations = 0

    def clear(self) -> None:
        if self.current is not None:
            self.current.destroy()
            self.current = None

    def get(
        self,
        width: int,
        height: int,
        supersample: int,
        msaa: int,
        color_format: str,
        instances: int,
        shadow_size: int,
        shadow_layers: int,
    ) -> tuple[FrameTargets, bool]:
        """Targets for this configuration and whether they were (re)allocated now."""
        timestamps = 6 if self.context.has_timestamps else 0
        key = (width, height, supersample, msaa, color_format, shadow_size, shadow_layers)
        current = self.current
        if current is not None and current.key == key and current.layout.instances >= instances:
            return current, False
        self.clear()
        self.current = self._allocate(key, max(instances, 1), timestamps)
        self.allocations += 1
        return self.current, True

    def _allocate(self, key: tuple, instances: int, timestamps: int) -> FrameTargets:
        from .shaders import DEPTH_FORMAT, OBJECT_TARGET

        width, height, k, msaa, color_format, shadow_size, shadow_layers = key
        ctx = self.context
        wgpu = ctx.wgpu
        device = ctx.device
        size = (width * k, height * k, 1)
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING

        def image(fmt: str, label: str) -> Any:
            return device.create_texture(
                label=label, size=size, format=fmt, usage=usage, sample_count=msaa
            )

        color = image(color_format, "aether3d-color")
        objects = image(OBJECT_TARGET, "aether3d-objects")
        depth = image(DEPTH_FORMAT, "aether3d-depth")
        layers = max(shadow_layers, 1)
        map_size = shadow_size if shadow_layers else 1
        shadow = device.create_texture(
            label="aether3d-shadow-maps",
            size=(map_size, map_size, layers),
            format=DEPTH_FORMAT,
            usage=usage,
        )
        shadow_view = shadow.create_view(
            dimension="2d-array", base_array_layer=0, array_layer_count=layers
        )
        shadow_layer_views = [
            shadow.create_view(dimension="2d", base_array_layer=i, array_layer_count=1)
            for i in range(shadow_layers)
        ]
        layout = OutputLayout.build(width, height, instances, timestamps)
        output = device.create_buffer(
            label="aether3d-output",
            size=layout.total,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        staging = device.create_buffer(
            label="aether3d-staging",
            size=layout.total,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        query_set = query_buffer = None
        if timestamps:
            query_set = device.create_query_set(type="timestamp", count=timestamps)
            query_buffer = device.create_buffer(
                label="aether3d-timestamps",
                size=_align(8 * timestamps),
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
            )
        color_bytes = 16 if color_format == "rgba32float" else 8
        nbytes = target_bytes(width, height, k, msaa, color_bytes, (shadow_size, shadow_layers))
        return FrameTargets(
            key=key,
            color=color,
            objects=objects,
            depth=depth,
            color_view=color.create_view(),
            objects_view=objects.create_view(),
            depth_view=depth.create_view(),
            shadow=shadow,
            shadow_view=shadow_view,
            shadow_layers=shadow_layer_views,
            output=output,
            staging=staging,
            layout=layout,
            query_set=query_set,
            query_buffer=query_buffer,
            nbytes=nbytes,
        )


# --------------------------------------------------------------------------- #
# Packed per-frame data (exact WGSL layouts)
# --------------------------------------------------------------------------- #


def draw_order(
    rs: RenderScene, cull: str, keep: np.ndarray | None = None
) -> tuple[np.ndarray, list[tuple[int, bool, int, int]]]:
    """Instance order for drawing, and draws ``(mesh, mirrored, first, count)``.

    Instances sharing a mesh (and, when culling, the same winding) become one
    instanced draw call; ``first`` indexes the instance records in draw order.
    Instances where ``keep`` is ``False`` are left out.
    """

    def winding(i: int) -> bool:
        return rs.instances[i].mirrored if cull != "none" else False

    indices = [i for i in range(len(rs.instances)) if keep is None or keep[i]]
    keyed = sorted(indices, key=lambda i: (rs.instances[i].mesh, winding(i), i))
    draws: list[tuple[int, bool, int, int]] = []
    for position, index in enumerate(keyed):
        mesh, mirrored = rs.instances[index].mesh, winding(index)
        if draws and draws[-1][0] == mesh and draws[-1][1] == mirrored:
            draws[-1] = (mesh, mirrored, draws[-1][2], draws[-1][3] + 1)
        else:
            draws.append((mesh, mirrored, position, 1))
    return np.array(keyed, dtype=np.int64), draws


def pack_instances(rs: RenderScene, order: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """``struct Instance`` records in draw order (camera-relative model matrices)."""
    n = len(order)
    data = np.zeros((max(n, 1), INSTANCE_FLOATS), dtype=np.float32)
    if n == 0:
        return data
    words = data.view(np.uint32)
    instances = [rs.instances[i] for i in order]
    world = np.stack([inst.world for inst in instances])
    model = world[:, :3, :4].copy()
    model[:, :, 3] -= eye  # camera-relative, in float64 before rounding to float32
    data[:, 0:12] = model.reshape(n, 12)
    normal = np.stack([inst.normal_matrix for inst in instances])
    for row in range(3):
        data[:, 4 * (3 + row) : 4 * (3 + row) + 3] = normal[:, row]
    data[:, 24:27] = np.stack([inst.base_color for inst in instances])
    data[:, 27] = [inst.specular for inst in instances]
    data[:, 28] = [inst.shininess for inst in instances]
    has_normals = np.array([m.normals is not None for m in rs.meshes], dtype=bool)
    mesh = np.fromiter((inst.mesh for inst in instances), dtype=np.int64, count=n)
    mirrored = np.fromiter((inst.mirrored for inst in instances), dtype=bool, count=n)
    words[:n, 29] = np.where(mirrored, MIRRORED, 0) | np.where(has_normals[mesh], HAS_NORMALS, 0)
    words[:n, 30] = order
    return data


def pack_lights(rs: RenderScene, eye: np.ndarray, plan: ShadowPlan | None) -> np.ndarray:
    """``struct Light`` records (point light positions camera-relative)."""
    data = np.zeros((max(len(rs.lights), 1), LIGHT_FLOATS), dtype=np.float32)
    words = data.view(np.uint32)
    signed = data.view(np.int32)
    for i, light in enumerate(rs.lights):
        data[i, 0:3] = light.radiance
        if light.type == "directional":
            words[i, 3] = 0
            data[i, 4:7] = light.direction
        else:
            words[i, 3] = 1
            data[i, 4:7] = light.position - eye
            data[i, 7] = 0.0 if light.range is None else light.range
            words[i, 8] = 1 if light.falloff == "inverse_square" else 0
        signed[i, 9] = -1 if plan is None else plan.layer_of(i)
    return data


def pack_shadow_maps(plan: ShadowPlan | None) -> np.ndarray:
    """``struct ShadowMap`` records, one per layer."""
    maps = () if plan is None else plan.maps
    data = np.zeros((max(len(maps), 1), SHADOW_FLOATS), dtype=np.float32)
    for i, shadow in enumerate(maps):
        data[i, 0:16] = shadow.matrix.T.reshape(-1)  # WGSL matrices are column-major
        data[i, 16] = shadow.bias
        data[i, 17] = shadow.normal_offset
    return data


def view_projection(rs: RenderScene, aspect: float) -> np.ndarray:
    """Camera-relative world -> clip, with reversed depth (near plane 1, far plane 0).

    Reversed depth keeps float32 depth precise over the whole ``near..far``
    range (also for ``far / near`` up to 10^9). It is the camera's projection
    matrix with its depth row replaced by ``w - z``, after the view rotation.
    """
    projection = rs.camera.camera.projection_matrix(aspect).copy()
    projection[2] = projection[3] - projection[2]
    frame = rs.camera.frame
    rotation = np.eye(4)
    rotation[0, :3], rotation[1, :3], rotation[2, :3] = frame.right, frame.up, -frame.forward
    return projection @ rotation


def pack_frame(rs: RenderScene, aspect: float, plan: ShadowPlan | None) -> np.ndarray:
    """``struct Frame``: the uniforms of the scene pass."""
    data = np.zeros(FRAME_FLOATS, dtype=np.float32)
    words = data.view(np.uint32)
    signed = data.view(np.int32)
    data[0:16] = view_projection(rs, aspect).T.reshape(-1)
    data[16:19] = rs.camera.frame.forward
    data[19] = 1.0 if rs.camera.camera.is_perspective else 0.0
    data[20:23] = rs.ambient
    words[24] = len(rs.lights)
    if plan is not None and plan.maps:
        signed[25] = plan.settings.pcf_radius
        data[26] = 1.0 / plan.settings.map_size
    return data


def pack_finalize(
    width: int, height: int, supersample: int, msaa: int, rs: RenderScene, exposure: float
) -> np.ndarray:
    """``struct Params`` of the finalize pass."""
    data = np.zeros(8, dtype=np.float32)
    data[:4].view(np.uint32)[:] = (width, height, supersample, msaa)
    camera = rs.camera.camera
    data[4] = camera.near
    data[5] = camera.far
    data[6] = 1.0 if camera.is_perspective else 0.0
    data[7] = 2.0**exposure
    return data
