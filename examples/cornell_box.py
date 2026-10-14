"""A Cornell-box-inspired room, path traced: color bleeding from bounced light.

Run from the repository root:

    python examples/cornell_box.py [OUTPUT_DIR] [--size N] [--samples N] [--quick]

1. Builds the room (2 x 2 x 2 meters, open at the front): a white floor,
   ceiling and back wall, a red left and a green right wall (reflectances
   close to the classic Cornell box measurements), a tall and a short white
   block, and a square emissive panel under the ceiling -- the room's only
   light, facing down.
2. Renders it from one camera, every image by Aether3D:

   * ``cornell_raster.png`` -- the rasterizer (the GPU when available, else
     the CPU): emission is unlit there, so the panel glows but lights nothing;
   * ``cornell_direct.png`` -- the path tracer with ``bounces=0``: light
     straight from the panel, with soft shadows, and no bounce light (the
     ceiling stays black);
   * ``cornell_path.png`` -- the path tracer with 8 bounces: global
     illumination, the walls' colors bleeding onto the white surfaces;
   * ``cornell_comparison.png`` -- the three side by side, half size.

3. Checks what the images show: near the red wall the bounced light on white
   surfaces is red, near the green wall green; the ceiling gets no direct
   light but is lit by bounces; bounces add light. Writes the settings,
   times and ray counts of every render to ``cornell_box.json``.

Exits with status 1 if a check fails. ``--quick`` renders small and fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from showcase import caption_strip, half, record, save_report, timed_render, write  # noqa: E402

from aether3d import RenderError, Scene  # noqa: E402
from aether3d.rendering.color import linear_to_srgb  # noqa: E402

WHITE = (0.725, 0.71, 0.68)
RED = (0.63, 0.065, 0.05)
GREEN = (0.14, 0.45, 0.091)


def display(linear) -> tuple[float, float, float]:
    """The display color whose linear value is ``linear`` (materials take display colors)."""
    return tuple(float(v) for v in linear_to_srgb(np.asarray(linear, dtype=np.float64)))


def build_scene() -> Scene:
    scene = Scene("Cornell Box")
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    scene.render_settings.default_lights = False  # the panel is the only light
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
    tall = scene.add_cube(
        "Tall Block", size=1.0, position=(-0.33, 0.3, 0.6), scale=(0.55, 0.55, 1.2)
    )
    tall.rotation_euler = (0, 0, 17)
    tall.material = white
    short = scene.add_cube("Short Block", size=0.55, position=(0.35, -0.3, 0.275))
    short.rotation_euler = (0, 0, -18)
    short.material = white
    lamp = scene.add_plane("Lamp", size=0.5, position=(0, 0, 1.995))
    lamp.rotation_euler = (180, 0, 0)  # facing down
    lamp.material = scene.create_material(
        "Lamp",
        model="simple",
        base_color="#000000",
        emissive=display((1.0, 0.86, 0.64)),
        emissive_strength=16.0,
        double_sided=False,
    )
    scene.add_camera("Camera", position=(0, -3.35, 1.0), look_at=(0, 0, 1.0), fov=40)
    return scene


def interior(mask: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` whose 8 neighbours are in it too (away from object outlines)."""
    pad = np.pad(mask, 1)
    h, w = mask.shape
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
    return out


def check(direct, full) -> list[str]:
    problems = []
    names = [o["name"] for o in full.objects]
    ids = full.object_index
    white_floor_ceiling = np.isin(ids, [names.index("Floor"), names.index("Ceiling")])
    cols = np.arange(ids.shape[1])[None, :].repeat(ids.shape[0], axis=0)
    third = ids.shape[1] // 3
    near_red = white_floor_ceiling & (cols < third)
    near_green = white_floor_ceiling & (cols >= 2 * third)
    indirect = full.passes["indirect"]
    red_side = indirect[near_red].mean(axis=0)
    green_side = indirect[near_green].mean(axis=0)
    print(f"  bounce light near the red wall (RGB): {np.round(red_side, 4).tolist()}")
    print(f"  bounce light near the green wall (RGB): {np.round(green_side, 4).tolist()}")
    # The lamp is warm, so bounce light is reddish everywhere; the walls tint it relative to
    # each other: compare the green/red balance near each wall.
    tint = (green_side[1] / green_side[0]) / (red_side[1] / red_side[0])
    print(f"  green/red balance of bounce light: {tint:.2f}x higher near the green wall")
    if not tint > 1.5:
        problems.append("bounce light near the walls is not tinted by their colors")
    ceiling = interior(ids == names.index("Ceiling"))  # away from the lamp's outline
    if direct.passes["direct"][ceiling].max() > 0.0:
        problems.append("the ceiling receives direct light")
    if not full.linear[ceiling].mean() > 0.02:
        problems.append("the ceiling is not lit by bounced light")
    walls = np.isin(
        ids, [names.index(n) for n in ("Floor", "Back Wall", "Left Wall", "Right Wall")]
    )
    gain = full.linear[walls].mean() / direct.linear[walls].mean()
    print(f"  bounces add {100 * (gain - 1):.0f}% light on the walls and floor")
    if gain < 1.2:
        problems.append("bounce light adds less than 20% on the walls")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--size", type=int, default=640, help="image width and height")
    parser.add_argument("--samples", type=int, default=256, help="path tracing samples per pixel")
    parser.add_argument("--quick", action="store_true", help="160x160, 16 samples")
    args = parser.parse_args(argv)
    if args.quick:
        args.size, args.samples = 160, 16
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    size = {"width": args.size, "height": args.size}
    print("1. Rasterizer (emission is unlit: the panel glows, nothing else is lit)")
    try:
        raster, seconds = timed_render(scene, backend="gpu", **size)
    except RenderError as exc:
        print(f"  GPU unavailable ([{exc.code}]); using the CPU rasterizer")
        raster, seconds = timed_render(scene, backend="cpu", **size)
    raster.save(out / "cornell_raster.png")
    record("cornell_raster", raster, seconds, out / "cornell_raster.png")
    print("2. Path tracer, direct light only (bounces=0)")
    direct, seconds = timed_render(
        scene, backend="path", path={"samples": args.samples, "bounces": 0}, **size
    )
    direct.save(out / "cornell_direct.png")
    record("cornell_direct", direct, seconds, out / "cornell_direct.png")
    print("3. Path tracer, 8 bounces")
    full, seconds = timed_render(
        scene, backend="path", path={"samples": args.samples, "bounces": 8}, **size
    )
    full.save(out / "cornell_path.png")
    record("cornell_path", full, seconds, out / "cornell_path.png")
    strip = caption_strip(
        [
            (f"Rasterizer ({raster.backend['name'].upper()})", half(raster.image)),
            ("Path tracer: direct light only", half(direct.image)),
            ("Path tracer: 8 bounces", half(full.image)),
        ]
    )
    write(out / "cornell_comparison.png", strip)
    print("4. Checks")
    problems = check(direct, full)
    save_report(out / "cornell_box.json", "Cornell-box-inspired room")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
