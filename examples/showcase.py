"""Shared helpers of the Stage 8 rendering showcases (path tracing examples).

* :func:`caption_strip` puts rendered images side by side, each under a short
  caption band (drawn with Pillow's built-in font when Pillow is installed;
  without it the images are joined uncaptioned). Captions are drawn on a
  separate band, never over rendered pixels.
* :func:`record` keeps what every render was, how long it took and how it
  was made, and :func:`save_report` writes that list as JSON next to the
  images, so every number quoted about them can be checked.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import aether3d
from aether3d.rendering.png import write_png

CAPTION_HEIGHT = 34


def _caption(text: str, width: int) -> np.ndarray:
    band = np.full((CAPTION_HEIGHT, width, 3), 24, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # captions are optional
        return band
    image = Image.fromarray(band)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1: a fixed bitmap font
        font = ImageFont.load_default()
    draw.text((10, 7), text, fill=(235, 235, 235), font=font)
    return np.asarray(image)


def half(image: np.ndarray) -> np.ndarray:
    """2x2 box downsampling in linear light."""
    from aether3d.rendering.color import linear_to_srgb, quantize, srgb_to_linear

    h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    lin = srgb_to_linear(image[:h, :w].astype(np.float64) / 255.0)
    lin = lin.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    return quantize(linear_to_srgb(lin))


def caption_strip(panels: list[tuple[str, np.ndarray]], gap: int = 8) -> np.ndarray:
    """Images side by side, each with its caption above it."""
    height = max(p.shape[0] for _, p in panels)
    columns = []
    for k, (text, image) in enumerate(panels):
        pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
        column = np.concatenate([_caption(text, image.shape[1]), image, pad], axis=0)
        columns.append(column)
        if k < len(panels) - 1:
            columns.append(np.full((column.shape[0], gap, 3), 24, dtype=np.uint8))
    return np.concatenate(columns, axis=1)


def write(path: Path, image: np.ndarray) -> Path:
    rgba = np.concatenate([image, np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    return write_png(path, rgba)


REPORT: list[dict[str, Any]] = []


def record(name: str, result, seconds: float, output: Path | None = None) -> dict:
    """Remember one render for the JSON report and print a one-line summary."""
    backend = result.backend
    entry: dict[str, Any] = {
        "image": name,
        "file": None if output is None else output.name,
        "width": result.width,
        "height": result.height,
        "backend": backend["name"],
        "seconds": round(seconds, 2),
    }
    if backend["name"] == "path":
        entry.update(
            device=backend["device"],
            processes=backend["processes"],
            samples=backend["samples_rendered"],
            bounces=backend["bounces"] if backend["integrator"] == "path" else None,
            seed=backend["seed"],
            clamp=backend["clamp"],
            integrator=backend["integrator"],
            rays=result.stats["rays"],
            mrays_per_second=round(result.stats["rays_per_second"] / 1e6, 3),
            relative_noise=result.stats["noise"]["mean_relative_error"],
        )
    elif backend["name"] == "gpu":
        entry.update(adapter=backend["adapter"]["name"], api=backend["adapter"]["backend"])
    entry["tonemap"] = result.color["tonemap"]
    entry["exposure"] = result.color["exposure"]
    REPORT.append(entry)
    extra = ""
    if backend["name"] == "path":
        what = f"{entry['bounces']} bounces" if entry["bounces"] is not None else "AO only"
        extra = (
            f", {entry['samples']} spp, {what}, {entry['processes']} "
            f"processes, {entry['rays']:,} rays ({entry['mrays_per_second']} M rays/s)"
        )
    elif backend["name"] == "gpu":
        extra = f", {entry['adapter']}"
    print(f"  {name}: {result.width}x{result.height} {backend['name']}{extra}, {seconds:.1f} s")
    return entry


def timed_render(scene, **options):
    t0 = time.perf_counter()
    result = scene.render(**options)
    return result, time.perf_counter() - t0


def save_report(path: Path, title: str) -> Path:
    from aether3d.rendering.pathtrace.parallel import cpu_count

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    data = {
        "title": title,
        "aether3d": aether3d.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "cpus": cpu_count(),
        "renders": REPORT,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
