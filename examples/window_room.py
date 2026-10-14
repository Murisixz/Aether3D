"""A room lit through its window: light that reaches the interior only by bouncing.

Run from the repository root:

    python examples/window_room.py [OUTPUT_DIR] [--width W] [--height H]
        [--samples N] [--quick]

1. Builds a closed room (4 x 4 x 2.6 m) with a window in its front wall, a
   table, a bench and a shelf, under a low sun and a sky outside.
2. Renders it from inside, every image by Aether3D:

   * ``window_direct.png`` -- the path tracer with ``bounces=0``: only the
     patch of floor the sun reaches through the window and the surfaces
     facing the window are lit; the rest of the room stays dark;
   * ``window_path.png`` -- the path tracer with 6 bounces: the sunlit patch
     lights the whole room;
   * ``window_comparison.png`` -- both side by side, half size.

   Both use one stop of exposure, so the window clips to white.

3. Checks that the walls the sun cannot reach get light only from bounces,
   and writes settings, times and ray counts to ``window_room.json``.

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

from aether3d import Scene  # noqa: E402

W, D, H = 4.0, 4.0, 2.6  # room width (x), depth (y), height (z)
WINDOW = (1.4, 1.2, 0.9)  # window width, height, sill height


def build_scene() -> Scene:
    scene = Scene("Window Room")
    scene.render_settings.default_lights = False
    scene.world.configure(
        background="#9ec4e8",
        ambient_intensity=0.0,
        environment={
            "type": "sky",
            "zenith_color": "#3b67b0",
            "horizon_color": "#dfe8f0",
            "ground_color": "#6a6050",
            "intensity": 0.6,
        },
    )
    plaster = scene.create_material("Plaster", model="simple", base_color="#e3ddd2")
    floor_mat = scene.create_material("Oak Floor", preset="wood", scale=1.6, roughness=0.6)
    for name, position, rotation, size, material in (
        ("Floor", (0, 0, 0), (0, 0, 0), (W, D), floor_mat),
        ("Ceiling", (0, 0, H), (180, 0, 0), (W, D), plaster),
        ("Back Wall", (0, D / 2, H / 2), (90, 0, 0), (W, H), plaster),
        ("Left Wall", (-W / 2, 0, H / 2), (0, 90, 0), (H, D), plaster),
        ("Right Wall", (W / 2, 0, H / 2), (0, -90, 0), (H, D), plaster),
    ):
        obj = scene.add_plane(name, size=1, position=position, scale=(*size, 1))
        obj.rotation_euler = rotation
        obj.material = material
    # The front wall (y = -D/2) as four panels around the window.
    ww, wh, sill = WINDOW
    top = sill + wh
    for name, (x0, x1, z0, z1) in {
        "Wall Below Window": (-W / 2, W / 2, 0, sill),
        "Wall Above Window": (-W / 2, W / 2, top, H),
        "Wall Left of Window": (-W / 2, -ww / 2, sill, top),
        "Wall Right of Window": (ww / 2, W / 2, sill, top),
    }.items():
        panel = scene.add_plane(
            name,
            size=1,
            position=((x0 + x1) / 2, -D / 2, (z0 + z1) / 2),
            scale=(x1 - x0, z1 - z0, 1),
        )
        panel.rotation_euler = (-90, 0, 0)
        panel.material = plaster
    wood = scene.create_material("Walnut", base_color="#6b4630", roughness=0.5)
    red = scene.create_material("Red Paint", base_color="#b0352a", roughness=0.45)
    for name, position, scale, material in (
        ("Table Top", (0.3, 0.4, 0.74), (1.4, 0.8, 0.06), wood),
        ("Table Leg 1", (-0.32, 0.08, 0.36), (0.06, 0.06, 0.72), wood),
        ("Table Leg 2", (0.92, 0.08, 0.36), (0.06, 0.06, 0.72), wood),
        ("Table Leg 3", (-0.32, 0.72, 0.36), (0.06, 0.06, 0.72), wood),
        ("Table Leg 4", (0.92, 0.72, 0.36), (0.06, 0.06, 0.72), wood),
        ("Bench", (-1.3, 0.2, 0.22), (0.4, 1.4, 0.44), red),
        ("Shelf", (1.75, 1.2, 0.9), (0.4, 1.0, 1.8), wood),
    ):
        scene.add_cube(name, size=1, position=position, scale=scale).material = material
    # Low afternoon sun from the front, through the window.
    scene.add_directional_light(
        "Sun", direction=(0.25, 0.85, -0.47), color="#ffe2b8", intensity=4.0, angle=0.53
    )
    scene.add_camera("Camera", position=(-1.6, 1.75, 1.55), look_at=(0.3, -0.6, 0.9), fov=62)
    return scene


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=128, help="path tracing samples per pixel")
    parser.add_argument("--quick", action="store_true", help="240x160, 16 samples")
    args = parser.parse_args(argv)
    if args.quick:
        args.width, args.height, args.samples = 240, 160, 16
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    # Exposed for the interior, as a photographer would: the window is allowed to blow out.
    size = {"width": args.width, "height": args.height, "exposure": 1.0}
    print("1. Path tracer, direct light only (bounces=0)")
    direct, seconds = timed_render(
        scene, backend="path", path={"samples": args.samples, "bounces": 0}, **size
    )
    direct.save(out / "window_direct.png")
    record("window_direct", direct, seconds, out / "window_direct.png")
    print("2. Path tracer, 6 bounces")
    full, seconds = timed_render(
        scene, backend="path", path={"samples": args.samples, "bounces": 6}, **size
    )
    full.save(out / "window_path.png")
    record("window_path", full, seconds, out / "window_path.png")
    write(
        out / "window_comparison.png",
        caption_strip(
            [
                ("Path tracer: direct light only", half(direct.image)),
                ("Path tracer: 6 bounces", half(full.image)),
            ]
        ),
    )
    print("3. Checks")
    problems = []
    names = [o["name"] for o in full.objects]
    # The front wall around the window faces away from the sun and cannot see the sky:
    # only bounced light reaches it. The ceiling sees a little sky through the window.
    front = np.isin(
        full.object_index,
        [
            names.index(n)
            for n in ("Wall Above Window", "Wall Left of Window", "Wall Right of Window")
        ],
    )
    ceiling = full.object_index == names.index("Ceiling")
    for name, mask, ratio in (("front wall", front, 10.0), ("ceiling", ceiling, 4.0)):
        before = float(direct.linear[mask].mean())
        after = float(full.linear[mask].mean())
        print(f"  {name}: {before:.4f} with direct light, {after:.4f} with bounces")
        if not (after > 0.01 and after > ratio * max(before, 1e-6)):
            problems.append(f"the {name} is not lit by bounced light")
    if not np.any(direct.linear.max(axis=2) > 0.5):
        problems.append("no sunlit patch in the direct-light image")
    save_report(out / "window_room.json", "Room lit through its window")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
