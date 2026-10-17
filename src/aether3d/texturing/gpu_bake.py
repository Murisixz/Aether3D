"""Bake rays on the GPU (``bake(..., backend="gpu")``).

The rays of a bake -- projection rays, and the AO and thickness hemispheres, which the
kernel generates itself from each point's normal and random shift -- are traced by
``wgsl/bake.wgsl`` against the CPU path tracer's BVH, uploaded as the GPU path tracer's
scene words (:func:`aether3d.rendering.gpu.pathtrace.pack_scene_words`) and traversed by
the same code (``wgsl/traverse.wgsl``). Everything else -- texel samples, shading of the
hits, accumulation, padding -- is the CPU code of :mod:`.bake`, so a GPU bake differs from
a CPU bake only by float32 rounding in the traversal (and, for AO and thickness, in the
directions).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..errors import RenderError, TexturingError
from ..rendering.raytrace.accel import Hits

ITEM_FLOATS = 12
RESULT_FLOATS = 6
WORKGROUP = 64
ROW = 65535 * WORKGROUP  # items per dispatch row
MAX_ITEMS = 1 << 20  # items per dispatch (bounds buffers and keeps the device responsive)


class GPUTracer:
    """Bake rays on a GPU adapter; ``TexturingError("GPU_UNAVAILABLE")`` without one."""

    name = "gpu"

    def __init__(self, ps: Any, adapter: Any = None) -> None:
        try:
            from ..rendering.gpu.device import get_context
            from ..rendering.gpu.pathtrace import pack_meshes, pack_scene_words
            from ..rendering.pathtrace.lights import LightSet
            from ..rendering.raytrace.accel import _round_out

            context, _ = get_context(adapter)
        except (RenderError, ImportError) as exc:
            raise TexturingError(
                "GPU_UNAVAILABLE", f"no GPU for baking ({exc})", "backend"
            ) from None
        self.context = context
        self.ps = ps
        self.rays = 0
        self.seconds = 0.0
        eye = np.zeros(3)
        _, mesh_tri_base = pack_meshes(ps.rs)
        words = pack_scene_words(ps, LightSet.build(ps), eye, mesh_tri_base)
        lo, hi = _round_out(ps.accel.root_lo[None, :], ps.accel.root_hi[None, :])
        self.root = (np.asarray(lo, np.float32).reshape(3), np.asarray(hi, np.float32).reshape(3))
        self.instance_count = len(ps.rs.instances)
        wgpu = context.wgpu
        with context.lock:
            self.scene = context.device.create_buffer_with_data(
                label="aether3d-bake-scene", data=words, usage=wgpu.BufferUsage.STORAGE
            )
            self.pipeline = context.shaders.bake_pipeline()

    def _run(
        self, mode: int, items: np.ndarray, rays: int = 1, distance: float = 0.0
    ) -> np.ndarray:
        context = self.context
        wgpu = context.wgpu
        device = context.device
        n = items.shape[0]
        out = np.empty((n, RESULT_FLOATS), dtype=np.float32)
        t0 = time.perf_counter()
        with context.lock:
            for start in range(0, n, MAX_ITEMS):
                chunk = np.ascontiguousarray(items[start : start + MAX_ITEMS], dtype=np.float32)
                count = chunk.shape[0]
                params = np.zeros(12, dtype=np.float32)
                params[0:3] = self.root[0]
                params.view(np.uint32)[3] = self.instance_count
                params[4:7] = self.root[1]
                params.view(np.uint32)[7] = mode
                params.view(np.uint32)[8] = count
                params.view(np.uint32)[9] = rays
                params[10] = distance
                row = min(count, ROW)
                params.view(np.uint32)[11] = row
                storage = wgpu.BufferUsage.STORAGE
                inp = device.create_buffer_with_data(data=chunk, usage=storage)
                res = device.create_buffer(
                    size=max(count * RESULT_FLOATS * 4, 16),
                    usage=storage | wgpu.BufferUsage.COPY_SRC,
                )
                uni = device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM)
                group = device.create_bind_group(
                    layout=context.shaders.layout("bake"),
                    entries=[
                        {"binding": 0, "resource": {"buffer": self.scene}},
                        {"binding": 1, "resource": {"buffer": uni}},
                        {"binding": 2, "resource": {"buffer": inp}},
                        {"binding": 3, "resource": {"buffer": res}},
                    ],
                )
                encoder = device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(self.pipeline)
                compute.set_bind_group(0, group)
                compute.dispatch_workgroups(math.ceil(row / WORKGROUP), math.ceil(count / row))
                compute.end()
                context.queue.submit([encoder.finish()])
                data = np.frombuffer(context.queue.read_buffer(res), dtype=np.float32)
                out[start : start + count] = data[: count * RESULT_FLOATS].reshape(
                    count, RESULT_FLOATS
                )
                for buffer in (inp, res, uni):
                    buffer.destroy()
        self.seconds += time.perf_counter() - t0
        return out

    def closest(self, origin, direction, tmax, skip_inst=None, skip_tri=None) -> Hits:
        n = origin.shape[0]
        self.rays += n
        items = np.zeros((n, ITEM_FLOATS), dtype=np.float32)
        items[:, 0:3] = origin
        items[:, 3] = np.broadcast_to(tmax, (n,))
        items[:, 4:7] = direction
        _skip(items, skip_inst, skip_tri)
        out = self._run(0, items)
        inst = out[:, 1].view(np.uint32).astype(np.int64)
        prim = out[:, 2].view(np.uint32).astype(np.int64)
        miss = inst == 0xFFFFFFFF
        inst[miss] = -1
        prim[miss] = -1
        t = np.where(miss, np.inf, out[:, 0].astype(np.float64))
        return Hits(t, inst, prim, out[:, 3:6].astype(np.float64))

    def hemisphere(self, mode: int, origin, normal, shifts, count, distance, skip_inst, skip_tri):
        n = origin.shape[0]
        self.rays += n * count
        items = np.zeros((n, ITEM_FLOATS), dtype=np.float32)
        items[:, 0:3] = origin
        items[:, 4:7] = normal
        items[:, 7:9] = shifts
        _skip(items, skip_inst, skip_tri)
        return self._run(mode, items, count, distance)[:, 0].astype(np.float64)


def _skip(items: np.ndarray, skip_inst, skip_tri) -> None:
    words = items.view(np.uint32)
    if skip_inst is None:
        words[:, 10] = 0xFFFFFFFF
        words[:, 11] = 0xFFFFFFFF
    else:
        words[:, 10] = np.asarray(skip_inst, dtype=np.int64).astype(np.uint32)
        words[:, 11] = np.asarray(skip_tri, dtype=np.int64).astype(np.uint32)
