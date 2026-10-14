"""Ambient occlusion contact test: the fast AO preview next to a path traced image.

Run from the repository root:

    python examples/ao_contact.py [OUTPUT_DIR] [--width W] [--height H]
        [--samples N] [--quick]

1. Builds a corner of two walls with stairs, touching boxes, a sphere resting
   on the floor and a box hovering 1.5 m above it -- higher than the AO reach.
2. Renders, every image by Aether3D:

   * ``ao_preview.png`` -- the AO preview (``path={"integrator": "ao"}``):
     only camera rays and AO rays, much faster than path tracing. Each pixel
     is ``1 - occlusion``: the cosine-weighted fraction of the hemisphere
     above the surface left open within the AO reach (0.6 m here);
   * ``ao_overcast.png`` -- the path tracer under a uniform overcast sky
     (4 bounces): the same darkening in corners and contacts arises
     physically, from light transport, with no AO applied;
   * ``ao_comparison.png`` -- both side by side, half size.

   AO is an artistic approximation of what the path tracer computes: the
   path tracer never multiplies AO into its image (that would darken
   corners twice); it is a separate pass and preview.
3. Checks that contacts and corners are darker than the open floor, that the
   floor under the hovering box is exactly unoccluded (beyond the reach), and
   that the preview traced no bounce or shadow rays. Writes settings, times
   and ray counts to ``ao_contact.json``.

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

REACH = 0.6


def build_scene() -> Scene:
    scene = Scene("AO Contact")
    scene.render_settings.default_lights = False  # the overcast sky is the only light
    scene.world.configure(
        background="#d8dde3",
        ambient_intensity=0.0,
        environment={"type": "uniform", "color": "#ffffff", "intensity": 1.0},
    )
    plaster = scene.create_material("Plaster", model="simple", base_color="#d9d6d0")
    scene.add_plane("Floor", size=12).material = plaster
    back = scene.add_plane("Back Wall", size=12, position=(0, 2.0, 6))
    back.rotation_euler = (90, 0, 0)
    back.material = plaster
    side = scene.add_plane("Side Wall", size=12, position=(-2.5, 0, 6))
    side.rotation_euler = (0, 90, 0)
    side.material = plaster
    for step in range(4):  # stairs against the side wall
        stair = scene.add_cube(
            f"Stair {step}",
            size=1.0,
            position=(-2.2, 1.2 - 0.45 * step, 0.2 * (step + 1) / 2 * 1.0),
            scale=(0.6, 0.45, 0.2 * (step + 1)),
        )
        stair.material = plaster
    for k, (x, y) in enumerate(((0.0, 0.8), (0.62, 0.8), (0.31, 0.8))):
        box = scene.add_cube(f"Box {k}", size=0.6, position=(x, y, 0.3 + 0.6 * (k == 2)))
        box.material = plaster
    ball = scene.add_sphere("Ball", radius=0.45, segments=64, rings=32, position=(1.6, -0.3, 0.45))
    ball.material = plaster
    floating = scene.add_cube("Hovering Box", size=0.6, position=(-0.6, -0.9, 1.8))
    floating.material = plaster
    scene.add_camera("Camera", position=(2.8, -5.2, 3.0), look_at=(-0.3, 0.3, 0.5), fov=40)
    return scene


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=64, help="samples per pixel")
    parser.add_argument("--quick", action="store_true", help="320x180, 8 samples")
    args = parser.parse_args(argv)
    if args.quick:
        args.width, args.height, args.samples = 320, 180, 8
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    size = {"width": args.width, "height": args.height}
    print("1. AO preview")
    ao, seconds = timed_render(
        scene,
        backend="path",
        path={"samples": args.samples, "integrator": "ao"},
        ao={"samples": 4, "distance": REACH},
        **size,
    )
    ao.save(out / "ao_preview.png")
    record("ao_preview", ao, seconds, out / "ao_preview.png")
    print("2. Path tracer under a uniform overcast sky (no AO applied)")
    overcast, seconds = timed_render(
        scene, backend="path", path={"samples": args.samples, "bounces": 4}, **size
    )
    overcast.save(out / "ao_overcast.png")
    record("ao_overcast", overcast, seconds, out / "ao_overcast.png")
    write(
        out / "ao_comparison.png",
        caption_strip(
            [
                (f"AO preview (reach {REACH} m)", half(ao.image)),
                ("Path tracer, overcast sky (no AO)", half(overcast.image)),
            ]
        ),
    )
    print("3. Checks")
    problems = []
    names = [o["name"] for o in ao.objects]
    values = ao.passes["ao"]
    floor = ao.object_index == names.index("Floor")
    open_floor = floor & (values >= 0.999)
    contact = floor & (values < 0.8)
    print(
        f"  floor: {100 * open_floor.mean() / max(floor.mean(), 1e-9):.0f}% fully open, "
        f"{100 * contact.mean() / max(floor.mean(), 1e-9):.0f}% darker than 0.8 near contacts"
    )
    if not open_floor.any() or not contact.any():
        problems.append("the AO preview does not separate open floor from contacts")
    if ao.stats["bounce_rays"] or ao.stats["shadow_rays"]:
        problems.append("the AO preview traced bounce or shadow rays")
    # The floor straight below the hovering box (1.5 m above it): beyond the reach.
    cam = scene.add_camera(
        "Under Hovering Box",
        projection="orthographic",
        ortho_scale=0.2,
        position=(-0.6, -0.9, 0.5),
        look_at=(-0.6, -0.9, 0.0),
        up=(0, 1, 0),
    )
    below, _ = timed_render(
        scene,
        camera=cam,
        width=4,
        height=4,
        backend="path",
        path={"samples": 16, "integrator": "ao"},
        ao={"samples": 8, "distance": REACH},
    )
    if not np.all(below.passes["ao"] == 1.0):
        problems.append("geometry beyond the AO reach occludes")
    save_report(out / "ao_contact.json", "Ambient occlusion contact test")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
