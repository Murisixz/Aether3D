"""Cameras, lights and Aether3D's own CPU renderer (see ``docs/RENDERING.md``).

Quick start::

    from aether3d import Scene

    scene = Scene()
    scene.add_cube("Crate", size=2).color = "#c0392b"
    camera = scene.add_camera(position=(4, -6, 3), look_at=(0, 0, 0))
    scene.add_directional_light(direction=(-0.4, 0.6, -1.0))
    result = scene.render(width=640, height=360)
    result.save("crate.png")
"""

from .camera import Camera, CameraFrame
from .color import color_to_hex, linear_to_srgb, parse_color, srgb_to_linear
from .framing import Framing
from .light import Light
from .png import encode_png, write_png
from .renderer import RenderResult, render
from .scene_data import RenderScene, extract_render_scene
from .settings import Appearance, RenderLimits, RenderSettings, World

__all__ = [
    "Appearance",
    "Camera",
    "CameraFrame",
    "Framing",
    "Light",
    "RenderLimits",
    "RenderResult",
    "RenderScene",
    "RenderSettings",
    "World",
    "color_to_hex",
    "encode_png",
    "extract_render_scene",
    "linear_to_srgb",
    "parse_color",
    "render",
    "srgb_to_linear",
    "write_png",
]
