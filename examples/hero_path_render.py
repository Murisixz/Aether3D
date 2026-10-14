"""The blocky hero with its Stage 7 materials, path traced.

Run from the repository root:

    python examples/hero_path_render.py [OUTPUT_DIR] [--width W] [--height H]
        [--samples N] [--quick]

1. Builds the hero of ``examples/hero_render.py`` with its materials
   (procedural skin and fabric, a diamond blade on a metal hilt, a grass
   block with a face-group top) and lights it for the path tracer: the same
   sun, now with the angular size of the real sun, and a procedural sky
   instead of the flat ambient light. The rim light stays.
2. Renders, every image by Aether3D from the same camera and materials:

   * ``hero_path.png`` -- the path tracer, 4 bounces;
   * ``hero_path_comparison.png`` -- half size: the GPU rasterizer with sun
     shadow maps (the CPU rasterizer without shadows if there is no GPU), and
     the path tracer with 4 bounces.

3. Checks that every part of the hero is visible and nothing is reported,
   and writes settings, times and ray counts to ``hero_path.json``.

Exits with status 1 if a check fails. ``--quick`` renders small and fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hero_render import PARTS  # noqa: E402
from hero_render import build_scene as build_hero  # noqa: E402
from showcase import caption_strip, half, record, save_report, timed_render, write  # noqa: E402

from aether3d import RenderError  # noqa: E402


def build_scene(width: int, height: int):
    scene = build_hero(width, height, 2, materials=True)
    scene.get_by_name("Sun").light.angle = 0.53
    scene.world.environment = {
        "type": "sky",
        "zenith_color": "#3f72c4",
        "horizon_color": "#d8e6f3",
        "ground_color": "#5d5446",
        "intensity": 0.7,
    }
    return scene


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=64, help="path tracing samples per pixel")
    parser.add_argument("--quick", action="store_true", help="320x180, 8 samples")
    args = parser.parse_args(argv)
    if args.quick:
        args.width, args.height, args.samples = 320, 180, 8
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args.width, args.height)
    problems = []
    print(f"1. Path tracer, 4 bounces, {args.width}x{args.height}")
    full, seconds = timed_render(
        scene,
        width=args.width,
        height=args.height,
        backend="path",
        path={"samples": args.samples, "bounces": 4},
    )
    full.save(out / "hero_path.png")
    record("hero_path", full, seconds, out / "hero_path.png")
    problems += [f"{d.code}: {d.message}" for d in full.diagnostics if d.severity != "info"]
    missing = set(PARTS) - {o["name"] for o in full.objects if o["visible_samples"]}
    if missing:
        problems.append(f"parts not visible: {sorted(missing)}")
    print("2. Comparison at half size")
    size = {"width": args.width // 2, "height": args.height // 2}
    try:
        raster, seconds = timed_render(scene, backend="gpu", shadows=True, **size)
    except RenderError as exc:
        print(f"  GPU unavailable ([{exc.code}]); using the CPU rasterizer (no shadows)")
        raster, seconds = timed_render(scene, backend="cpu", **size)
    record("hero_raster_half", raster, seconds)
    write(
        out / "hero_path_comparison.png",
        caption_strip(
            [
                (f"Rasterizer ({raster.backend['name'].upper()}, Stage 7)", half(raster.image)),
                ("Path tracer: 4 bounces", half(full.image)),
            ]
        ),
    )
    save_report(out / "hero_path.json", "The blocky hero (path traced)")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
