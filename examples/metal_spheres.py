"""Polished and rough metal spheres, path traced: reflections of the scene around them.

Run from the repository root:

    python examples/metal_spheres.py [OUTPUT_DIR] [--width W] [--height H]
        [--samples N] [--quick]

1. Builds a row of spheres on a checkered floor under a procedural sky and a
   sun with a soft edge (``angle=2`` degrees): mirror-polished chrome,
   satin gold (roughness 0.3), rough copper (0.6) and glossy red plastic
   (a dielectric, roughness 0.12). An orange and a green block stand close
   enough to show up in the reflections.
2. Renders it from one camera, every image by Aether3D:

   * ``metals_raster.png`` -- the GPU rasterizer with sun shadows (the CPU
     rasterizer without shadows when there is no GPU): Stage 7's GGX shading
     of the sun and the ambient light, no reflected geometry;
   * ``metals_direct.png`` -- the path tracer with ``bounces=0``: the sun
     and the sky, seen directly and in reflections, but not the lit scene;
   * ``metals_path.png`` -- the path tracer with 4 bounces: the floor, the
     blocks and the other spheres appear in the metals, sharp in the chrome
     and blurred by roughness in the gold and copper. The chrome sphere also
     reflects the sun onto the floor (a caustic), which a path tracer finds
     only rarely: such samples are very bright and show as speckles. This
     render clamps indirect light at 3 per sample (``path={"clamp": 3}``,
     a documented, biased option) to remove them;
   * ``metals_comparison.png`` -- the three side by side, half size.

3. Checks that the green block appears in the chrome sphere only once light
   bounces, and writes settings, times and ray counts to ``metals.json``.

Exits with status 1 if a check fails. ``--quick`` renders small and fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from showcase import caption_strip, half, record, save_report, timed_render, write  # noqa: E402

from aether3d import RenderError, Scene, create_sphere  # noqa: E402

CLAMP = 3.0
"""Indirect radiance cap per sample for the 4-bounce render (removes caustic speckles)."""
SPHERES = {
    # name: (x, base color, metallic, roughness)
    "Chrome": (-2.1, "#f4f4f4", 1.0, 0.0),
    "Gold": (-0.7, "#ffc35a", 1.0, 0.3),
    "Copper": (0.7, "#e8957a", 1.0, 0.6),
    "Red Plastic": (2.1, "#c01818", 0.0, 0.12),
}


def build_scene() -> Scene:
    scene = Scene("Metal Spheres")
    scene.world.configure(
        background="#9ec4e8",
        ambient_color="#cfe0ff",
        ambient_intensity=0.35,
        environment={
            "type": "sky",
            "zenith_color": "#2f63b8",
            "horizon_color": "#d6e2ee",
            "ground_color": "#5a5248",
            "intensity": 0.9,
        },
    )
    floor = scene.add_plane("Floor", size=40)
    tiles = scene.create_material("Tiles", roughness=0.55, specular=0.5)
    tiles.set_procedural_texture(
        "base_color",
        "checker",
        color1="#3a3a3a",
        color2="#c9c9c9",
        coordinates="world",
        projection="xy",
        scale=1.25,
    )
    floor.material = tiles
    ball = create_sphere(0.6, 96, 48)
    for name, (x, color, metallic, roughness) in SPHERES.items():
        obj = scene.add_object(name, mesh=ball, position=(x, 0, 0.6))
        obj.material = scene.create_material(
            name, base_color=color, metallic=metallic, roughness=roughness
        )
    orange = scene.add_cube("Orange Block", size=0.8, position=(-3.1, 1.3, 0.4))
    orange.rotation_euler = (0, 0, 25)
    orange.material = scene.create_material("Orange", base_color="#ff7a1a", roughness=0.7)
    green = scene.add_cube("Green Block", size=0.7, position=(-1.45, -1.25, 0.35))
    green.rotation_euler = (0, 0, -15)
    green.material = scene.create_material("Green", base_color="#2fbf3f", roughness=0.7)
    scene.add_directional_light(
        "Sun", direction=(0.55, 0.65, -0.7), color="#fff1dc", intensity=2.6, angle=2.0
    )
    scene.add_camera("Camera", position=(0.35, -6.8, 2.1), look_at=(0, 0, 0.45), fov=34)
    return scene


def green_in_chrome(result) -> float:
    """Fraction of the chrome sphere's pixels (away from its outline) that look green."""
    names = [o["name"] for o in result.objects]
    sphere = result.object_index == names.index("Chrome")
    pad = np.pad(sphere, 2)
    h, w = sphere.shape
    chrome = sphere.copy()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            chrome &= pad[2 + dy : h + 2 + dy, 2 + dx : w + 2 + dx]
    rgb = result.linear[chrome]
    green = (rgb[:, 1] > 1.6 * rgb[:, 0]) & (rgb[:, 1] > 1.6 * rgb[:, 2]) & (rgb[:, 1] > 0.02)
    return float(green.mean())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=128, help="path tracing samples per pixel")
    parser.add_argument("--quick", action="store_true", help="320x180, 16 samples")
    args = parser.parse_args(argv)
    if args.quick:
        args.width, args.height, args.samples = 320, 180, 16
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    size = {"width": args.width, "height": args.height}
    print("1. Rasterizer")
    try:
        raster, seconds = timed_render(scene, backend="gpu", shadows=True, **size)
    except RenderError as exc:
        print(f"  GPU unavailable ([{exc.code}]); using the CPU rasterizer (no shadows)")
        raster, seconds = timed_render(scene, backend="cpu", **size)
    raster.save(out / "metals_raster.png")
    record("metals_raster", raster, seconds, out / "metals_raster.png")
    print("2. Path tracer, direct light only (bounces=0)")
    direct, seconds = timed_render(
        scene, backend="path", path={"samples": args.samples, "bounces": 0}, **size
    )
    direct.save(out / "metals_direct.png")
    record("metals_direct", direct, seconds, out / "metals_direct.png")
    print("3. Path tracer, 4 bounces")
    full, seconds = timed_render(
        scene,
        backend="path",
        path={"samples": args.samples, "bounces": 4, "clamp": CLAMP},
        **size,
    )
    full.save(out / "metals_path.png")
    record("metals_path", full, seconds, out / "metals_path.png")
    write(
        out / "metals_comparison.png",
        caption_strip(
            [
                (f"Rasterizer ({raster.backend['name'].upper()})", half(raster.image)),
                ("Path tracer: direct light only", half(direct.image)),
                (f"Path tracer: 4 bounces (clamp {CLAMP:g})", half(full.image)),
            ]
        ),
    )
    print("4. Checks")
    problems = []
    before, after = green_in_chrome(direct), green_in_chrome(full)
    print(
        f"  green block in the chrome sphere: {100 * before:.2f}% of its pixels direct, "
        f"{100 * after:.2f}% with bounces"
    )
    if not (after > 0.002 and after > 5 * before):
        problems.append("the green block is not reflected in the chrome sphere")
    save_report(out / "metals.json", "Metal spheres")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
