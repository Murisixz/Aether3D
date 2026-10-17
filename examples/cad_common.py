"""Shared helpers of the precision-modeling examples (``cad_*.py``, ``mcp_cad_agent.py``).

Everything drawn here is Aether3D's own: renders come from its renderers (the GPU path
tracer when available, else the CPU path tracer or rasterizer), drawings from
``aether3d.cad.drawing``, captions from the drawing module's stroke font. No external
renderer, font or image library is used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aether3d import RenderError, __version__
from aether3d.cad.drawing.raster import rasterize
from aether3d.rendering.png import encode_png

WHITE = 255


def render(scene, camera, width, height, backend="auto", samples=64):
    """``(image (h, w, 3) uint8, backend name, seconds)``: the GPU path tracer (auto, gpu),
    else the CPU path tracer (auto) -- or the CPU rasterizer (cpu, for quick runs). Path
    tracing caps indirect radiance per sample (``clamp``, a documented, biased option)
    against fireflies on glass and metal."""
    options = {"width": width, "height": height, "camera": camera}
    t0 = time.perf_counter()
    if backend in ("auto", "gpu"):
        try:
            r = scene.render(**options, backend="gpu_path", path={"samples": samples, "clamp": 4.0})
            return _rgb(r.image), "gpu_path", time.perf_counter() - t0
        except RenderError:
            if backend == "gpu":
                raise
    if backend == "auto":
        r = scene.render(**options, backend="path",
                         path={"samples": max(4, samples // 4), "clamp": 4.0})  # fmt: skip
        return _rgb(r.image), "path", time.perf_counter() - t0
    r = scene.render(**options, backend="cpu")
    return _rgb(r.image), "cpu", time.perf_counter() - t0


def _rgb(image) -> np.ndarray:
    px = image.pixels if hasattr(image, "pixels") else np.asarray(image)
    return np.ascontiguousarray(px[..., :3]).astype(np.uint8)


def drawing_image(drawing, width: int) -> np.ndarray:
    """A drawing sheet rasterized ``width`` pixels wide."""
    w_mm = drawing.paper()["size"][0]
    return drawing.to_png(width / w_mm * 25.4)[:, :width]


def caption(image: np.ndarray, text: str, height: int = 36) -> np.ndarray:
    """``image`` with a white band above it holding ``text`` (stroke font)."""
    h, w = image.shape[:2]
    mm = 25.4 / 96.0  # rasterize the band at 96 dpi: 1 px = mm
    paper = {
        "size": (w * mm, height * mm),
        "strokes": [],
        "fills": [],
        "texts": [(np.array([4 * mm * 2, height * mm * 0.3]), height * mm * 0.45, text, 0.0,
                   "left")],
    }  # fmt: skip
    band = rasterize(paper, 96.0)[:height, :w]
    return np.concatenate([band, image], axis=0)


def row(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """Images side by side on white, top-aligned."""
    h = max(i.shape[0] for i in images)
    parts = []
    for k, img in enumerate(images):
        pad = np.full((h - img.shape[0], img.shape[1], 3), WHITE, np.uint8)
        parts.append(np.concatenate([img, pad], axis=0))
        if k + 1 < len(images):
            parts.append(np.full((h, gap, 3), WHITE, np.uint8))
    return np.concatenate(parts, axis=1)


def column(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    w = max(i.shape[1] for i in images)
    parts = []
    for k, img in enumerate(images):
        pad = np.full((img.shape[0], w - img.shape[1], 3), WHITE, np.uint8)
        parts.append(np.concatenate([img, pad], axis=1))
        if k + 1 < len(images):
            parts.append(np.full((gap, w, 3), WHITE, np.uint8))
    return np.concatenate(parts, axis=0)


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    """Box-filtered downscale (or nearest upscale) to ``width`` pixels wide."""
    h, w = image.shape[:2]
    height = max(1, round(h * width / w))
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(int), h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(int), w - 1)
    if width >= w:
        return image[rows][:, cols]
    # Average the source pixels of each target pixel (simple area filter).
    ys = np.linspace(0, h, height + 1).astype(int)
    xs = np.linspace(0, w, width + 1).astype(int)
    acc = np.add.reduceat(np.add.reduceat(image.astype(np.float64), ys[:-1], axis=0), xs[:-1],
                          axis=1)  # fmt: skip
    counts = np.outer(np.diff(ys), np.diff(xs))[..., None]
    return np.round(acc / counts).astype(np.uint8)


def save_png(path: Path, image: np.ndarray) -> None:
    path.write_bytes(encode_png(image, text={"Software": f"Aether3D {__version__}"}))


def save_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, default=str))


def ground(scene, size=40.0, z=0.0, color="#b9b4aa"):
    """A large ground plane under the model."""
    from aether3d import create_plane

    g = scene.add_object("Ground", mesh=create_plane(size), position=(0, 0, z))
    g.color = color
    return g


def daylight(scene, sun=(0.45, 0.65, -0.62)):
    """A sun with the real sun's angular size and a sky (path tracers), plus ambient light
    (rasterizers)."""
    s = scene.add_directional_light("Sun", direction=sun, color="#fff3dc", intensity=2.3)
    s.light.angle = 0.53
    scene.world.configure(background="#cfdcea", ambient_color="#dfe8f3", ambient_intensity=0.35)
    scene.world.environment = {
        "type": "sky",
        "zenith_color": "#3f72c4",
        "horizon_color": "#d8e6f3",
        "ground_color": "#6d6556",
        "intensity": 0.8,
    }


def meshes_of(objects) -> list:
    """The objects and their descendants that carry meshes (CAD items with several display
    parts are shown as a parent object with one child per part)."""
    out, stack = [], list(objects)
    while stack:
        o = stack.pop()
        if o.mesh is not None:
            out.append(o)
        stack.extend(o.children)
    return out
