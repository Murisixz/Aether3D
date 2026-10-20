"""The scene of showcase G, rendered by whichever Aether3D is imported (``PYTHONPATH``).

A Cornell-box room lit only by an emissive panel (as ``examples/cornell_box.py``), built
with nothing newer than 0.19.1's API, so the same file runs on v0.19.1 and on 0.20.0.
It path-traces the room on the GPU (or the CPU with ``--cpu``) and saves the linear image
and what the render reported to ``--out`` (``.npz``). Path options newer than 0.19.1
(``--sampler``, ``--filter``, ``--adaptive``, ``--denoise``) are passed only when given.

    python examples/film_g_scene.py --out x.npz --spp 64 [--width 960 --height 540]
        [--bounces 8] [--seed 0] [--sampler sobol] [--filter blackman_harris]
        [--adaptive 0.02] [--denoise] [--glass]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import aether3d
from aether3d import Scene
from aether3d.rendering.color import linear_to_srgb

WHITE = (0.725, 0.71, 0.68)
RED = (0.63, 0.065, 0.05)
GREEN = (0.14, 0.45, 0.091)


def display(linear) -> tuple[float, float, float]:
    return tuple(float(v) for v in linear_to_srgb(np.asarray(linear, dtype=np.float64)))


def build(glass: bool = False) -> Scene:
    """The room (2 m cube open at the front), two blocks, the panel; ``glass`` (0.20.0
    only) adds a glass sphere and a disk area light and turns on depth of field."""
    scene = Scene("Cornell Box")
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    scene.render_settings.default_lights = False
    white = scene.create_material("White", model="simple", base_color=display(WHITE))
    red = scene.create_material("Red", model="simple", base_color=display(RED))
    green = scene.create_material("Green", model="simple", base_color=display(GREEN))
    walls = {
        "Floor": ((0, 0, 0), (0, 0, 0), white),
        "Ceiling": ((0, 0, 2), (180, 0, 0), white),
        "Back Wall": ((0, 1, 1), (90, 0, 0), white),
        "Left Wall": ((-1, 0, 1), (0, 90, 0), red),
        "Right Wall": ((1, 0, 1), (0, -90, 0), green),
    }
    for name, (position, rotation, material) in walls.items():
        wall = scene.add_plane(name, size=2, position=position)
        wall.rotation_euler = rotation
        wall.material = material
    tall = scene.add_cube("Tall Block", size=1.0, position=(-0.33, 0.3, 0.6),
                          scale=(0.55, 0.55, 1.2))  # fmt: skip
    tall.rotation_euler = (0, 0, 17)
    tall.material = white
    short = scene.add_cube("Short Block", size=0.55, position=(0.35, -0.3, 0.275))
    short.rotation_euler = (0, 0, -17)
    short.material = white
    panel = scene.add_plane("Light Panel", size=0.5, position=(0, 0, 1.98))
    panel.rotation_euler = (180, 0, 0)
    panel.material = scene.create_material(
        "Panel", model="simple", base_color=display((0.78, 0.78, 0.78)),
        emissive=display((1.0, 0.86, 0.64)), emissive_strength=16.0,
    )  # fmt: skip
    cam = scene.add_camera("Camera", position=(0, -3.35, 1.0), look_at=(0, 0, 1.0), fov=40)
    if glass:
        ball = scene.add_sphere("Glass", radius=0.24, position=(0.33, -0.32, 0.79),
                                segments=96, rings=48)  # fmt: skip
        ball.material = scene.create_material("Glass", preset="glass")
        scene.add_area_light("Disk", position=(-0.5, -0.6, 1.9), look_at=(-0.1, 0.0, 0.6),
                             shape="disk", size=0.35, intensity=2.0)  # fmt: skip
        cam.camera.configure(fstop=4.0, focus_distance=3.35 - 0.32)
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler")
    parser.add_argument("--filter")
    parser.add_argument("--adaptive", type=float)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--glass", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    scene = build(args.glass)
    path = {"samples": args.spp, "bounces": args.bounces, "seed": args.seed}
    if args.sampler:
        path["sampler"] = args.sampler
    if args.filter:
        path["filter"] = args.filter
        path["filter_width"] = 1.5
    if args.adaptive:
        path["adaptive_threshold"] = args.adaptive
    options = {"backend": "path" if args.cpu else "gpu_path", "width": args.width,
               "height": args.height, "path": path, "tonemap": "neutral"}  # fmt: skip
    if args.denoise:
        options["denoise"] = True
    scene.render(**{**options, "width": 64, "height": 36, "path": {**path, "samples": 1}})
    t0 = time.perf_counter()  # (timed after a warm-up render: shaders compiled, caches made)
    result = scene.render(**options)
    seconds = time.perf_counter() - t0
    stats = result.stats or {}
    report = {
        "aether3d": aether3d.__version__,
        "seconds": seconds,
        "options": options,
        "backend": result.backend,
        "samples_mean": stats.get("samples_mean"),
        "rays": stats.get("rays"),
        "codes": sorted({d.code for d in result.diagnostics}),
    }
    np.savez_compressed(
        args.out,
        linear=np.asarray(result.linear, np.float32),
        image=np.asarray(result.image),
        report=json.dumps(report, default=str),
    )
    print(json.dumps({"seconds": round(seconds, 3), "aether3d": aether3d.__version__}))


if __name__ == "__main__":
    main()
