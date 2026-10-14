"""Render the blocky hero and sword (modeled with the mesh editing engine) with Aether3D.

Run from the repository root:

    python examples/hero_render.py [OUTPUT_DIR] [--width W] [--height H] [--samples N]

Builds the posed character and sword of ``examples/minecraft_hero.py`` (every
mesh made by editing operations: extrusions, insets, bevels, loop cuts...),
stands it on a grass block, frames a camera on it, lights it, and gives it
materials (Aether3D 0.7): skin, woven fabric clothes, a sword with a diamond
blade on a steel hilt, and a grass block -- the hilt and the block's top are
face groups with their own materials (grass on top, earth with a grass
fringe on the sides). Renders ``OUTPUT_DIR/blocky_hero_render.png`` (default
``examples/output``) with Aether3D's CPU renderer, and the same scene in
Aether3D 0.6's flat colors side by side with it, at half size
(``hero_flat_vs_materials.png``). Checks that every body part and the sword
are visible, that the materials add detail, and that the saved project
re-renders identically. Exits with status 1 if a check fails.
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


def build_scene(
    width: int = 1280, height: int = 720, samples: int = 3, *, materials: bool = False
) -> Scene:
    """The hero scene: flat colors (as in Aether3D 0.6) or ``materials``."""
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
    if materials:
        dress_with_materials(scene)
    return scene


EARTH_SIDES = {
    "output": "side",
    "nodes": {
        "where": {"type": "coordinates"},
        "clods": {"type": "voronoi", "scale": 9.0, "randomness": 1.0, "seed": 31},
        "speckle": {"type": "fbm", "scale": 30.0, "octaves": 3, "seed": 32},
        "earth": {
            "type": "ramp",
            "factor": {"node": "clods", "output": "cell"},
            "stops": [[0.0, "#5b3c22"], [0.6, "#7a5232"], [1.0, "#8c6440"]],
        },
        "dirt": {
            "type": "mix",
            "a": {"node": "earth"},
            "b": {"node": "speckle"},
            "factor": 0.35,
            "mode": "multiply",
        },
        # The top 10 cm of each side (world height near 0) is grass, with a ragged edge.
        "edge": {
            "type": "math",
            "op": "add",
            "a": {"node": "where", "output": "y"},
            "b": {"node": "ragged"},
        },
        "ragged": {"type": "math", "op": "multiply", "a": {"node": "speckle"}, "b": 0.08},
        "fringe": {
            "type": "math",
            "op": "smoothstep",
            "a": {"node": "edge"},
            "b": -0.08,
            "c": -0.04,
        },
        "side": {
            "type": "mix",
            "a": {"node": "dirt"},
            "b": "#4e8f2f",
            "factor": {"node": "fringe"},
        },
    },
}


def dress_with_materials(scene: Scene) -> None:
    """Skin, fabric, a gem blade on a metal hilt and a grass block (face groups)."""
    skin = scene.create_material("Skin", base_color="#c69c72", roughness=0.62, specular=0.45)
    skin.set_procedural_texture(
        "base_color",
        {
            "nodes": {
                "mottle": {"type": "fbm", "scale": 14.0, "octaves": 3, "seed": 41},
                "tone": {
                    "type": "ramp",
                    "factor": {"node": "mottle"},
                    "stops": [[0.2, 0.9], [0.8, 1.06]],
                },
            },
            "output": "tone",
        },
        coordinates="object",
        projection="volume",
    )
    # A coarse weave (threads about 3 cm wide) reads as cloth at the hero's scale.
    shirt = scene.create_material("Shirt", preset="fabric", color="#1f9fa3", scale=3.0)
    trousers = scene.create_material("Trousers", preset="fabric", color="#3b44a8", scale=3.0)
    for name, material in (
        ("Head", skin),
        ("Right Arm", skin),
        ("Left Arm", skin),
        ("Torso", shirt),
        ("Right Leg", trousers),
        ("Left Leg", trousers),
    ):
        scene.get_by_name(name).material = material
    # The sword: a diamond blade on a steel hilt -- the hilt's faces (below the blade root,
    # selected with mesh editing) are face group 1, which has its own material.
    sword = scene.get_by_name("Sword")
    with sword.edit_mesh() as m:
        m.set_group(m.select_faces(box=((-1, -1, -1), (1, 1, 0.645))), 1)
    diamond = scene.create_material("Diamond", base_color="#55dde6", roughness=0.12, specular=1.0)
    steel = scene.create_material("Steel", preset="metal", color="#c3c7cf")
    sword.material = diamond
    scene.assign_material(sword, steel, group=1)
    # The grass block: its top face is face group 1, the other faces group 0.
    block = scene.get_by_name("Grass Block")
    cube = block.mesh
    top = cube.normals[cube.triangles[:, 0], 2] > 0.5
    block.mesh = cube.with_groups(np.where(top, 1, 0))
    earth = scene.create_material("Earth", roughness=0.95, specular=0.3)
    earth.set_procedural_texture(
        "base_color", EARTH_SIDES, coordinates="world", projection="triplanar"
    )
    earth.set_procedural_texture(
        "normal",
        {
            "nodes": {
                "clods": {"type": "voronoi", "scale": 9.0, "randomness": 1.0, "seed": 31},
                "bump": {
                    "type": "bump",
                    "height": {"node": "clods", "output": "edge"},
                    "depth": 0.004,
                },
            },
            "output": "bump",
        },
        coordinates="world",
        projection="triplanar",
    )
    lawn = scene.create_material(
        "Lawn", preset="grass", coordinates="world", projection="xy", scale=0.4
    )
    block.material = earth
    scene.assign_material(block, lawn, group=1)


def main(argv: list[str]) -> int:
    from minecraft_castle_render import side_by_side, texture_detail

    from aether3d import __version__
    from aether3d.rendering.png import write_png

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print("1. Modeling the hero and sword (mesh editing), lighting them, giving them materials")
    scene = build_scene(args.width, args.height, args.samples, materials=True)
    print(
        f"   {len(scene)} objects, {len(scene.materials.materials)} materials; the arms and "
        "legs share meshes where they are identical"
    )

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
    flat = build_scene(args.width, args.height, args.samples).render()
    pair = write_png(
        out_dir / "hero_flat_vs_materials.png",
        side_by_side(flat, result),
        text={"Software": f"Aether3D {__version__}"},
    )
    print(f"   wrote {pair} (half size: Aether3D 0.6's flat colors | materials)")
    for name in ("Torso", "Grass Block"):
        plain, textured = texture_detail(flat, result, name)
        print(f"   {name}: neighbouring pixels differ by {plain:.2f} (flat) -> {textured:.2f}")
        if not textured > plain + 0.5:
            failures.append(f"{name}: the material adds no visible detail")
    if [d for d in result.diagnostics if d.severity != "info"]:
        failures.append(f"unexpected diagnostics: {[d.code for d in result.diagnostics]}")

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
