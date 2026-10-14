"""The textured procedural castle at golden hour, path traced.

Run from the repository root (booleans need ``pip install 'aether3d[boolean]'``):

    python examples/castle_path_render.py [OUTPUT_DIR] [--width W] [--height H]
        [--samples N] [--quick]

1. Builds the castle of ``examples/minecraft_castle_render.py`` with its
   Stage 7 materials (stone-brick image textures with a normal map, plank
   texture, procedural shingles, cobbles, grass, bark and leaves, glowing
   torch flames), sets it on a meadow and lights it for the path tracer: a
   low warm sun with the angular size of the real sun (0.53 degrees:
   soft-edged shadows), a procedural sky with a warm horizon, and the two
   torches. The old "fill" light is removed -- bounced light now does its
   job.
2. Renders, every image by Aether3D from the same camera and materials:

   * ``castle_path.png`` -- the path tracer, 4 bounces (default 1920x1080,
     64 samples per pixel);
   * ``castle_path_comparison.png`` -- half size: the GPU rasterizer with
     sun shadow maps (Stage 7; the CPU rasterizer without shadows if there is
     no GPU), the path tracer with direct light only, and with 4 bounces.

   The rasterizer cannot use the sky (it keeps the world's ambient light) or
   the sun's size.
3. Checks that bounced light brightens the image where the direct light
   leaves it dark (its darkest quarter), that nothing is reported, and
   writes settings, times and ray counts to ``castle_path.json``.

Exits with status 1 if a check fails. ``--quick`` renders small and fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from minecraft_castle_render import build_scene as build_castle  # noqa: E402
from showcase import caption_strip, half, record, save_report, timed_render, write  # noqa: E402

from aether3d import RenderError  # noqa: E402


def build_scene(width: int, height: int):
    scene = build_castle(width, height, 2, materials=True)
    scene.remove(scene.get_by_name("Fill"))
    # A meadow all around, level with the top of the castle's ground slab (z = 0), replaces
    # the slab: in low sunlight the slab's edges would show as lines.
    scene.get_by_name("Ground").visible = False
    meadow = scene.add_plane("Meadow", size=600)
    meadow.material = "Meadow"
    # The camera was framed on the castle; let it see the meadow up to the top of the image.
    scene.active_camera.camera.far = 2000.0
    sun = scene.get_by_name("Sun")
    sun.look_along((0.82, 0.38, -0.27), up=(0, 0, 1))  # about 15 degrees above the horizon
    sun.light.configure(color="#ffc988", intensity=4.0, angle=0.53)
    scene.world.configure(
        background="#c9a77f",
        ambient_color="#ffe0bd",
        ambient_intensity=0.3,
        environment={
            "type": "sky",
            "zenith_color": "#36599a",
            "horizon_color": "#f0c389",
            "ground_color": "#4b4136",
            "intensity": 0.9,
        },
    )
    return scene


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=64, help="path tracing samples per pixel")
    parser.add_argument("--quick", action="store_true", help="480x270, 8 samples")
    args = parser.parse_args(argv)
    if args.quick:
        args.width, args.height, args.samples = 480, 270, 8
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
    full.save(out / "castle_path.png")
    record("castle_path", full, seconds, out / "castle_path.png")
    problems += [f"{d.code}: {d.message}" for d in full.diagnostics if d.severity != "info"]
    print("2. Comparison at half size")
    size = {"width": args.width // 2, "height": args.height // 2}
    try:
        raster, seconds = timed_render(scene, backend="gpu", shadows=True, **size)
    except RenderError as exc:
        print(f"  GPU unavailable ([{exc.code}]); using the CPU rasterizer (no shadows)")
        raster, seconds = timed_render(scene, backend="cpu", **size)
    record("castle_raster_half", raster, seconds)
    samples = max(args.samples // 2, 4)
    direct, seconds = timed_render(
        scene, backend="path", path={"samples": samples, "bounces": 0}, **size
    )
    record("castle_direct_half", direct, seconds)
    bounced, seconds = timed_render(
        scene, backend="path", path={"samples": samples, "bounces": 4}, **size
    )
    record("castle_path_half", bounced, seconds)
    write(
        out / "castle_path_comparison.png",
        caption_strip(
            [
                (f"Rasterizer ({raster.backend['name'].upper()}, Stage 7)", half(raster.image)),
                ("Path tracer: direct light only", half(direct.image)),
                ("Path tracer: 4 bounces", half(bounced.image)),
            ]
        ),
    )
    print("3. Checks")
    covered = (direct.coverage == 1.0) & (bounced.coverage == 1.0)
    gain = bounced.linear[covered].mean() / direct.linear[covered].mean()
    dark = covered & (direct.linear.mean(axis=2) < np.percentile(direct.linear[covered], 25))
    dark_gain = bounced.linear[dark].mean() / max(direct.linear[dark].mean(), 1e-9)
    print(
        f"  bounced light: +{100 * (gain - 1):.0f}% over the image, "
        f"+{100 * (dark_gain - 1):.0f}% on its darkest quarter"
    )
    if gain < 1.0 or dark_gain < 1.15:
        problems.append("bounced light does not brighten the image's darker parts")
    save_report(out / "castle_path.json", "Castle at golden hour (path traced)")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
