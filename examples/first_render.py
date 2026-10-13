"""Aether3D's first render: a small scene drawn by Aether3D's own CPU renderer.

Run from the repository root:

    python examples/first_render.py [OUTPUT_DIR] [--width W] [--height H] [--samples N]

Builds a scene from static and procedural primitives, adds a camera, a sun, a
fill light and a point light, renders it with ``scene.render()`` and writes
``OUTPUT_DIR/aether3d_first_render.png`` (default ``examples/output``) plus the
project ``first_render.aether``. No external renderer is involved: the image is
produced by Aether3D's rasterizer (projection, clipping, depth testing,
lighting) and Aether3D's own PNG encoder.

Then it checks the result: every object is visible, and reloading the saved
project re-renders the identical image. Reproduce the image from the terminal
with::

    aether3d render OUTPUT_DIR/first_render.aether -o first_render.png

Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aether3d import Scene

HERE = Path(__file__).resolve().parent


def build_scene(width: int = 1280, height: int = 720, samples: int = 3) -> Scene:
    scene = Scene("First Render")
    ground = scene.add_plane("Ground", size=60)
    ground.color = "#6f9b4f"

    crate = scene.add_cube("Crate", size=1.6, position=(-2.2, 0.3, 0.8))
    crate.rotation_euler = (0, 0, 20)
    crate.color = "#c0392b"

    ball = scene.add_sphere("Ball", radius=0.9, segments=48, rings=24, position=(0.4, -0.6, 0.9))
    ball.appearance.configure(color="#2e86c1", specular=0.5, shininess=60)

    ring = scene.add_procedural(
        "Ring", "torus", major_radius=0.9, minor_radius=0.28, position=(2.6, 0.9, 0.28)
    )
    ring.color = "#f1c40f"

    spire = scene.add_procedural(
        "Spire",
        "cone",
        radius_bottom=0.7,
        radius_top=0.0,
        depth=2.2,
        segments=6,
        smooth=False,
        position=(0.6, 2.6, 1.1),
    )
    spire.color = "#27ae60"

    scene.world.configure(background="#9cc6e8", ambient_color="#dbe8ff", ambient_intensity=0.25)
    scene.add_directional_light("Sun", direction=(-0.5, 0.7, -0.9), color="#fff1d6", intensity=2.2)
    scene.add_directional_light(
        "Fill", direction=(0.7, -0.2, -0.3), color="#c8d8ff", intensity=0.35
    )
    scene.add_point_light(
        "Lamp", position=(-0.6, -2.2, 1.6), color="#ffbb77", intensity=2.5, range=8
    )
    camera = scene.add_camera("Camera", fov=36)
    # Frame the four shapes (the ground is much larger and fills the background).
    camera.frame(
        ["Crate", "Ball", "Ring", "Spire"],
        direction=(-0.45, 1.0, -0.48),
        margin=0.12,
        aspect=width / height,
    )
    scene.render_settings.configure(width=width, height=height, samples=samples)
    return scene


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print("1. Building the scene")
    scene = build_scene(args.width, args.height, args.samples)
    print(f"   {len(scene)} objects: {', '.join(o.name for o in scene)}")

    print("2. Rendering with Aether3D's CPU renderer")
    result = scene.render()
    stats = result.stats
    print(
        f"   {result.width}x{result.height}, {args.samples}x{args.samples} samples per pixel, "
        f"{stats['triangles']} triangles, {stats['timings']['total']:.2f} s"
    )
    for obj in result.objects:
        print(f"   {obj['name']:8s} {obj['visible_samples']:9d} visible samples")
        if obj["visible_samples"] == 0:
            failures.append(f"{obj['name']} is not visible")
    for d in result.diagnostics:
        print(f"   {d.severity}: {d.code} {d.message}")
    image_path = result.save(out_dir / "aether3d_first_render.png")
    print(f"   wrote {image_path} ({stats['png_bytes']} bytes)")

    print("3. Saving, reloading and re-rendering the project")
    project = out_dir / "first_render.aether"
    scene.save(project)
    again = Scene.load(project).render()
    if not np.array_equal(again.image, result.image):
        failures.append("the reloaded project renders differently")
    else:
        print(f"   {project}: identical image after reload")
    print(f"   reproduce: aether3d render {project} -o first_render.png")

    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
