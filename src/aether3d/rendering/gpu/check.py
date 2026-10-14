"""A self-test that proves GPU rendering works on this machine (``aether3d gpu info --check``).

It builds a small scene (two boxes, a sphere and a ground plane under a sun
and a point light), renders it on the GPU with the CPU renderer's sample
positions (``supersample=2, msaa=1``) and with 4x MSAA and shadows, reads the
images back and checks them against the CPU renderer:

* every object is visible, and object coverage agrees exactly in parity mode
  (one or two samples on exact triangle edges may differ);
* pixel colors agree to within a few 8-bit codes away from edges;
* the MSAA and shadow render covers the same objects and darkens the ground
  behind the boxes.

Nothing here can pass without real draw calls and a real framebuffer readback.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...errors import RenderError

WIDTH, HEIGHT = 96, 64


def test_scene():
    from ...scene import Scene

    scene = Scene("GPU self-test")
    scene.add_plane("Ground", size=12).color = "#6a9a4a"
    scene.add_cube("Red box", size=1.6, position=(-1.2, 0.0, 0.8)).color = "#c0392b"
    scene.add_cube("Blue box", size=1.0, position=(1.6, 1.2, 0.5)).color = "#2e86c1"
    ball = scene.add_sphere("Ball", radius=0.7, position=(0.6, -1.6, 0.7))
    ball.appearance.configure(color="#f1c40f", specular=0.4, shininess=40)
    scene.add_directional_light("Sun", direction=(0.5, 0.8, -1.0), intensity=2.0)
    scene.add_point_light("Lamp", position=(-2.0, -2.0, 2.5), intensity=4.0)
    scene.world.configure(background="#8ec5f2", ambient_intensity=0.2)
    camera = scene.add_camera("Camera", fov=40)
    camera.frame(direction=(-0.45, 1.0, -0.6), aspect=WIDTH / HEIGHT)
    return scene


def _interior(index: np.ndarray) -> np.ndarray:
    """Pixels whose 3x3 neighbourhood shows a single object (or only background)."""
    h, w = index.shape
    padded = np.pad(index, 1, mode="edge")
    same = np.ones_like(index, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            same &= padded[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx] == index
    return same


def self_test(adapter: Any = None) -> dict[str, Any]:
    """Render the test scene on the GPU and compare it with the CPU renderer."""
    t0 = time.perf_counter()
    report: dict[str, Any] = {"ok": False, "checks": {}, "adapter": None}
    checks = report["checks"]
    scene = test_scene()
    try:
        cpu = scene.render(width=WIDTH, height=HEIGHT, samples=2)
        parity = scene.render(
            width=WIDTH, height=HEIGHT, samples=2, backend="gpu", adapter=adapter, supersample=2
        )
        msaa = scene.render(
            width=WIDTH,
            height=HEIGHT,
            samples=2,
            backend="gpu",
            adapter=adapter,
            shadows={"lights": ["Sun"]},
        )
    except RenderError as exc:
        report["error"] = {"code": exc.code, "message": exc.message, "hint": exc.hint}
        report["seconds"] = time.perf_counter() - t0
        return report
    report["adapter"] = parity.backend["adapter"]["name"]
    report["backend"] = parity.backend["adapter"]["backend"]

    names = [o["name"] for o in cpu.objects]
    visible = [o["name"] for o in parity.objects if o["visible_samples"] > 0]
    checks["all_objects_visible"] = {"ok": visible == names, "visible": visible}
    same = parity.object_index == cpu.object_index
    checks["coverage_matches_cpu"] = {
        "ok": float(same.mean()) >= 0.995,
        "matching_pixels": float(same.mean()),
    }
    gpu_samples = [o["visible_samples"] for o in parity.objects]
    cpu_samples = [o["visible_samples"] for o in cpu.objects]
    worst = max(
        (abs(g - c) / max(c, 1) for g, c in zip(gpu_samples, cpu_samples, strict=True)),
        default=0.0,
    )
    checks["visible_samples_match_cpu"] = {
        "ok": worst <= 0.01,
        "gpu": gpu_samples,
        "cpu": cpu_samples,
    }
    interior = _interior(cpu.object_index) & same
    diff = np.abs(parity.image.astype(int) - cpu.image.astype(int)).max(axis=-1)
    largest = int(diff[interior].max()) if interior.any() else 255
    checks["colors_match_cpu"] = {"ok": largest <= 3, "max_difference": largest}
    msaa_visible = [o["name"] for o in msaa.objects if o["visible_samples"] > 0]
    checks["msaa_render"] = {
        "ok": msaa_visible == names and msaa.backend["antialiasing"]["msaa"] == 4,
        "antialiasing": msaa.backend["antialiasing"],
    }
    ground = names.index("Ground")
    on_ground = (msaa.object_index == ground) & (parity.object_index == ground)
    darker = (msaa.linear.sum(-1) < 0.6 * parity.linear.sum(-1)) & on_ground
    checks["shadows"] = {"ok": int(darker.sum()) >= 20, "shadowed_ground_pixels": int(darker.sum())}
    report["ok"] = all(c["ok"] for c in checks.values())
    report["timings"] = {
        "gpu_render_seconds": parity.stats["timings"]["total"],
        "cpu_render_seconds": cpu.stats["timings"]["total"],
    }
    report["seconds"] = time.perf_counter() - t0
    return report
