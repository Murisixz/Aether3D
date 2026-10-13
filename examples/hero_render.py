"""Render the blocky hero and sword (modeled with the mesh editing engine) with Aether3D.

Run from the repository root:

    python examples/hero_render.py [OUTPUT_DIR] [--width W] [--height H] [--samples N]

Builds the posed character and sword of ``examples/minecraft_hero.py`` (every
mesh made by editing operations: extrusions, insets, bevels, loop cuts...),
gives each part a solid color, stands it on a grass block, frames a camera on
it, lights it, and renders ``OUTPUT_DIR/blocky_hero_render.png`` (default
``examples/output``) with Aether3D's CPU renderer. Checks that every body part
and the sword are visible, and that the saved project re-renders identically.
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from minecraft_hero import build_hero, build_sword  # noqa: E402

from aether3d import Scene  # noqa: E402

PARTS = {
    "Head": "#c69c72",
    "Torso": "#1f9fa3",
    "Right Arm": "#c69c72",
    "Left Arm": "#c69c72",
    "Right Leg": "#3b44a8",
    "Left Leg": "#3b44a8",
    "Sword": "#62d6e0",
}


def build_scene(width: int = 1280, height: int = 720, samples: int = 3) -> Scene:
    scene = Scene("Blocky Hero")
    with contextlib.redirect_stdout(io.StringIO()):  # the modeling steps log every operation
        right_arm = build_hero(scene)
        build_sword(scene, right_arm)
    for name, color in PARTS.items():
        scene.get_by_name(name).color = color
    scene.get_by_name("Sword").appearance.configure(specular=0.7, shininess=90)
    grass = scene.add_cube("Grass Block", size=1.0, position=(0, 0, -0.5), scale=(8, 8, 1))
    grass.color = "#5b8f3a"
    scene.world.configure(background="#a9d0f5", ambient_color="#dbe8ff", ambient_intensity=0.3)
    # A raking key light from the upper left reveals the relief of the face and sword.
    scene.add_directional_light("Sun", direction=(0.8, 0.45, -0.55), color="#fff1d6", intensity=2.3)
    scene.add_directional_light("Rim", direction=(-0.5, -0.6, -0.3), color="#c8d8ff", intensity=0.5)
    camera = scene.add_camera("Camera", fov=30)
    camera.frame(list(PARTS), direction=(-0.3, 1.0, -0.18), margin=0.1, aspect=width / height)
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

    print("1. Modeling the hero and sword (mesh editing), coloring and lighting them")
    scene = build_scene(args.width, args.height, args.samples)
    print(f"   {len(scene)} objects; the arms and legs share meshes where they are identical")

    print("2. Rendering with Aether3D's CPU renderer")
    result = scene.render()
    stats = result.stats
    print(
        f"   {result.width}x{result.height}, {stats['triangles']} triangles, "
        f"{stats['timings']['total']:.2f} s"
    )
    seen = {o["name"]: o["visible_samples"] for o in result.objects}
    for name in PARTS:
        print(f"   {name:10s} {seen.get(name, 0):8d} visible samples")
        if seen.get(name, 0) == 0:
            failures.append(f"{name} is not visible")
    path = result.save(out_dir / "blocky_hero_render.png")
    print(f"   wrote {path} ({stats['png_bytes']:,} bytes)")

    print("3. Saving, reloading and re-rendering")
    project = out_dir / "blocky_hero_render.aether"
    scene.save(project)
    if not np.array_equal(Scene.load(project).render().image, result.image):
        failures.append("the reloaded project renders differently")
    else:
        print(f"   identical after reload; reproduce: aether3d render {project} -o hero.png")

    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
