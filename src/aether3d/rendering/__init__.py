"""Cameras, lights and Aether3D's own renderers.

See ``docs/RENDERING.md`` (CPU rasterizer), ``docs/GPU.md`` (GPU rasterizer)
and ``docs/RENDER_QUALITY.md`` (path tracer).

Quick start::

    from aether3d import Scene

    scene = Scene()
    scene.add_cube("Crate", size=2).color = "#c0392b"
    camera = scene.add_camera(position=(4, -6, 3), look_at=(0, 0, 0))
    scene.add_directional_light(direction=(-0.4, 0.6, -1.0))
    result = scene.render(width=640, height=360)  # CPU renderer
    result.save("crate.png")
    gpu = scene.render(width=1920, height=1080, backend="gpu", shadows=True)  # needs wgpu
    path = scene.render(width=640, height=360, backend="path", path={"samples": 64})  # CPU
"""

from .camera import Camera, CameraFrame
from .color import color_to_hex, linear_to_srgb, parse_color, srgb_to_linear
from .framing import Framing
from .gpu import gpu_status, list_adapters, release_gpu
from .light import Light
from .png import encode_png, write_png
from .renderer import RenderResult, render
from .scene_data import RenderScene, extract_render_scene
from .settings import (
    AOSettings,
    Appearance,
    Atmosphere,
    Environment,
    PathSettings,
    RenderLimits,
    RenderSettings,
    ShadowSettings,
    World,
)

__all__ = [
    "AOSettings",
    "Appearance",
    "Atmosphere",
    "Camera",
    "CameraFrame",
    "Environment",
    "Framing",
    "Light",
    "PathSettings",
    "RenderLimits",
    "RenderResult",
    "RenderScene",
    "RenderSettings",
    "ShadowSettings",
    "World",
    "color_to_hex",
    "encode_png",
    "extract_render_scene",
    "gpu_status",
    "linear_to_srgb",
    "list_adapters",
    "parse_color",
    "release_gpu",
    "render",
    "srgb_to_linear",
    "write_png",
]
