"""WGSL shader sources, shader modules, bind group layouts and pipelines (cached per device).

The shaders live in ``wgsl/*.wgsl`` as package data:

* ``common.wgsl`` -- instance and mesh declarations, vertex pulling helpers;
* ``scene.wgsl`` -- the scene pass (vertex + fragment: shading, shadows);
* ``shadow.wgsl`` -- the depth-only shadow pass;
* ``finalize.wgsl`` -- the compute pass that resolves samples and encodes sRGB.

Modules are assembled from these fixed files only (``common`` is prepended to
the scene and shadow shaders, and ``finalize`` has two variants, single- and
multi-sampled); nothing from a scene, project or command line is ever
compiled. Compilation errors become ``RenderError("GPU_SHADER_ERROR")`` with
the compiler's message in ``details``.
"""

from __future__ import annotations

import time
from importlib import resources
from typing import TYPE_CHECKING, Any

from ...errors import RenderError

if TYPE_CHECKING:
    from .device import GPUContext

COLOR_TARGET = "rgba32float"
COLOR_TARGET_HALF = "rgba16float"
OBJECT_TARGET = "r32float"
DEPTH_FORMAT = "depth32float"

_FINALIZE_VARIANTS = {
    True: {
        "{{COLOR_TEXTURE}}": "texture_multisampled_2d<f32>",
        "{{DEPTH_TEXTURE}}": "texture_depth_multisampled_2d",
        "{{SAMPLE}}": "i32(s)",
    },
    False: {
        "{{COLOR_TEXTURE}}": "texture_2d<f32>",
        "{{DEPTH_TEXTURE}}": "texture_depth_2d",
        "{{SAMPLE}}": "0",
    },
}


def shader_source(name: str) -> str:
    """The text of a bundled WGSL file: ``"common"``, ``"scene"``, ``"shadow"``, ``"finalize"``."""
    return resources.files(__package__).joinpath("wgsl", f"{name}.wgsl").read_text("utf-8")


def module_source(kind: str, multisampled: bool = False) -> str:
    """The complete WGSL of a shader module: ``"scene"``, ``"shadow"`` or ``"finalize"``."""
    if kind in ("scene", "shadow"):
        return shader_source("common") + "\n" + shader_source(kind)
    if kind == "finalize":
        text = shader_source("finalize")
        for marker, value in _FINALIZE_VARIANTS[bool(multisampled)].items():
            text = text.replace(marker, value)
        return text
    raise ValueError(f"unknown shader module {kind!r}")


class ShaderLibrary:
    """Compiled shader modules, bind group layouts and pipelines of one device."""

    def __init__(self, context: GPUContext) -> None:
        self.context = context
        self._modules: dict[tuple[str, bool], Any] = {}
        self._pipelines: dict[tuple, Any] = {}
        self.compile_seconds = 0.0
        self._layouts: dict[str, Any] = {}

    def clear(self) -> None:
        self._modules.clear()
        self._pipelines.clear()
        self._layouts.clear()

    @property
    def pipeline_count(self) -> int:
        return len(self._pipelines)

    # -- modules -----------------------------------------------------------------------

    def module(self, kind: str, multisampled: bool = False) -> Any:
        key = (kind, bool(multisampled) if kind == "finalize" else False)
        module = self._modules.get(key)
        if module is None:
            module = self.compile(module_source(kind, multisampled), f"aether3d-{kind}")
            self._modules[key] = module
        return module

    def compile(self, code: str, label: str) -> Any:
        """Compile WGSL; errors become ``GPU_SHADER_ERROR`` with the compiler output."""
        wgpu = self.context.wgpu
        t0 = time.perf_counter()
        try:
            module = self.context.device.create_shader_module(label=label, code=code)
        except wgpu.GPUError as exc:
            message = str(exc)
            if "lost" in message.lower():
                self.context.mark_lost()
                raise RenderError(
                    "GPU_DEVICE_LOST",
                    f"the GPU device was lost while compiling {label}",
                    hint="render again: the next render creates a new device",
                ) from None
            raise RenderError(
                "GPU_SHADER_ERROR",
                f"shader {label} failed to compile on {self.context.info.name}",
                f"shader[{label}]",
                hint="this is a bug in Aether3D or the GPU driver; please report it with "
                "`aether3d gpu info --json`",
                details={"compiler": message, "adapter": self.context.info.name},
            ) from None
        self.compile_seconds += time.perf_counter() - t0
        return module

    # -- layouts -----------------------------------------------------------------------

    def layout(self, name: str) -> Any:
        layout = self._layouts.get(name)
        if layout is None:
            layout = self._create_layout(name)
            self._layouts[name] = layout
        return layout

    def _create_layout(self, name: str) -> Any:
        wgpu = self.context.wgpu
        device = self.context.device
        vertex, fragment = wgpu.ShaderStage.VERTEX, wgpu.ShaderStage.FRAGMENT
        storage = {"type": wgpu.BufferBindingType.read_only_storage}
        uniform = {"type": wgpu.BufferBindingType.uniform}
        if name == "mesh":
            entries = [{"binding": b, "visibility": vertex, "buffer": storage} for b in range(3)]
        elif name == "frame":
            entries = [
                {"binding": 0, "visibility": vertex | fragment, "buffer": uniform},
                {"binding": 1, "visibility": vertex | fragment, "buffer": storage},
                {"binding": 2, "visibility": fragment, "buffer": storage},
                {"binding": 3, "visibility": fragment, "buffer": storage},
                {
                    "binding": 4,
                    "visibility": fragment,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.depth,
                        "view_dimension": wgpu.TextureViewDimension.d2_array,
                    },
                },
                {
                    "binding": 5,
                    "visibility": fragment,
                    "sampler": {"type": wgpu.SamplerBindingType.comparison},
                },
            ]
        elif name == "shadow":
            entries = [
                {"binding": 0, "visibility": vertex, "buffer": uniform},
                {"binding": 1, "visibility": vertex, "buffer": storage},
            ]
        elif name in ("finalize", "finalize-ms"):
            multisampled = name == "finalize-ms"
            compute = wgpu.ShaderStage.COMPUTE
            texture = {
                "sample_type": wgpu.TextureSampleType.unfilterable_float,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "multisampled": multisampled,
            }
            depth = {
                "sample_type": wgpu.TextureSampleType.depth,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "multisampled": multisampled,
            }
            writable = {"type": wgpu.BufferBindingType.storage}
            entries = [
                {"binding": 0, "visibility": compute, "texture": texture},
                {"binding": 1, "visibility": compute, "texture": texture},
                {"binding": 2, "visibility": compute, "texture": depth},
                {"binding": 3, "visibility": compute, "buffer": uniform},
                {"binding": 4, "visibility": compute, "buffer": storage},
                *[{"binding": b, "visibility": compute, "buffer": writable} for b in (5, 6, 7, 8)],
            ]
        else:  # pragma: no cover - internal misuse
            raise ValueError(name)
        return device.create_bind_group_layout(label=f"aether3d-{name}", entries=entries)

    # -- pipelines ---------------------------------------------------------------------

    def _cached(self, key: tuple, create) -> Any:
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            t0 = time.perf_counter()
            pipeline = create()
            self.compile_seconds += time.perf_counter() - t0
            self._pipelines[key] = pipeline
        return pipeline

    def scene_pipeline(self, msaa: int, color_format: str, cull: str, mirrored: bool) -> Any:
        """The scene pass for an MSAA count, color format, cull mode and winding."""
        wgpu = self.context.wgpu
        # Mirrored instances render with clockwise front faces so culling keeps outward faces.
        front = "ccw" if (not mirrored or cull == "none") else "cw"

        def create() -> Any:
            module = self.module("scene")
            layout = self.context.device.create_pipeline_layout(
                bind_group_layouts=[self.layout("frame"), self.layout("mesh")]
            )
            return self.context.device.create_render_pipeline(
                label=f"aether3d-scene-{msaa}x-{cull}-{front}",
                layout=layout,
                vertex={"module": module, "entry_point": "vs_main"},
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": front,
                    "cull_mode": cull,
                },
                depth_stencil={
                    "format": DEPTH_FORMAT,
                    "depth_write_enabled": True,
                    "depth_compare": wgpu.CompareFunction.greater,
                },
                multisample={"count": msaa},
                fragment={
                    "module": module,
                    "entry_point": "fs_main",
                    "targets": [{"format": color_format}, {"format": OBJECT_TARGET}],
                },
            )

        return self._cached(("scene", msaa, color_format, cull, front), create)

    def shadow_pipeline(self) -> Any:
        wgpu = self.context.wgpu

        def create() -> Any:
            layout = self.context.device.create_pipeline_layout(
                bind_group_layouts=[self.layout("shadow"), self.layout("mesh")]
            )
            return self.context.device.create_render_pipeline(
                label="aether3d-shadow",
                layout=layout,
                vertex={"module": self.module("shadow"), "entry_point": "vs_shadow"},
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "cull_mode": wgpu.CullMode.none,
                },
                depth_stencil={
                    "format": DEPTH_FORMAT,
                    "depth_write_enabled": True,
                    "depth_compare": wgpu.CompareFunction.less,
                },
            )

        return self._cached(("shadow",), create)

    def finalize_pipeline(self, multisampled: bool) -> Any:
        name = "finalize-ms" if multisampled else "finalize"

        def create() -> Any:
            layout = self.context.device.create_pipeline_layout(
                bind_group_layouts=[self.layout(name)]
            )
            return self.context.device.create_compute_pipeline(
                label=f"aether3d-{name}",
                layout=layout,
                compute={
                    "module": self.module("finalize", multisampled),
                    "entry_point": "finalize",
                },
            )

        return self._cached((name,), create)
