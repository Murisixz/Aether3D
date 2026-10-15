"""The path tracing showcases rendered by the GPU path tracer (``backend="gpu_path"``).

Run from the repository root (needs ``pip install 'aether3d[gpu]'`` and a working GPU
adapter; the castle also needs ``aether3d[boolean]``):

    python examples/gpu_path_showcase.py [OUTPUT_DIR] [--adapter SPEC] [--quick]

Every image is rendered by Aether3D's GPU path tracer on the selected adapter (with an
RTX 4060, on its shader cores through Vulkan; traversal is software, not the RT cores).
The scenes are the Stage 8 showcases (``cornell_box.py``, ``metal_spheres.py``,
``window_room.py``, ``castle_path_render.py``, ``hero_path_render.py``), with many more
samples than the CPU renders could afford:

* ``gpu_path_cornell.png`` -- the Cornell-box-inspired room: light bounced 8 times tints
  the white walls red and green;
* ``gpu_path_metals.png`` -- chrome, satin gold, rough copper and glossy plastic reflecting
  the floor, the blocks and each other (4 bounces; indirect light clamped at 3 per sample,
  as on the CPU, against the chrome sphere's sun caustic);
* ``gpu_path_window.png`` -- a room lit by the sun through its window, mostly by bounces;
* ``gpu_path_castle.png`` -- the textured procedural castle at golden hour, 1920x1080;
* ``gpu_path_hero.png`` -- the blocky hero under a sun and a sky;
* ``gpu_path_vs_cpu.png`` -- the check behind all of them: the Cornell box at 160x160,
  16 samples, 8 bounces, path traced by the CPU and by the GPU, and their difference
  (times 50). The GPU draws the CPU's random numbers, so both trace the same paths and the
  images agree to float32 rounding.

Checks that the GPU traced the CPU's paths in that comparison, that every render ran on
the GPU and reported nothing, and writes settings, times and ray counts to
``gpu_path_showcase.json``. Exits with status 1 if a check fails, 2 without a GPU.
``--quick`` renders small and fast.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from showcase import caption_strip, record, save_report, timed_render, write  # noqa: E402

from aether3d import RenderError  # noqa: E402
from aether3d.rendering.color import linear_to_srgb, quantize  # noqa: E402

# name, scene builder, full size, samples (full, quick), path options, render options
SHOWCASES = [
    ("cornell", "cornell_box", (640, 640), (4096, 32), {"bounces": 8}, {}),
    ("metals", "metal_spheres", (960, 540), (2048, 16), {"bounces": 4, "clamp": 3.0}, {}),
    ("window", "window_room", (960, 640), (4096, 32), {"bounces": 6}, {"exposure": 1.0}),
    ("castle", "castle_path_render", (1920, 1080), (512, 8), {"bounces": 4}, {}),
    ("hero", "hero_path_render", (1280, 720), (1024, 16), {"bounces": 4, "clamp": 3.0}, {}),
]


def build(module: str, width: int, height: int):
    builder = __import__(module).build_scene
    if module in ("castle_path_render", "hero_path_render"):
        return builder(width, height)
    return builder()


def compare(adapter, size: int, problems: list[str]) -> np.ndarray:
    """The Cornell box on the CPU and the GPU, and their difference."""
    scene = build("cornell_box", size, size)
    options = {"width": size, "height": size, "default_lights": False}
    path = {"samples": 16, "bounces": 8, "seed": 1}
    cpu, seconds = timed_render(scene, backend="path", path=path, **options)
    record("cornell_cpu", cpu, seconds)
    gpu, seconds = timed_render(scene, backend="gpu_path", adapter=adapter, path=path, **options)
    record("cornell_gpu", gpu, seconds)
    a = cpu.passes["beauty"].astype(np.float64)
    b = gpu.passes["beauty"].astype(np.float64)
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-3)
    median, share = float(np.median(rel)), float((rel > 0.01).mean())
    print(f"  CPU vs GPU: median relative difference {median:.1e}, {100 * share:.2f}% beyond 1%")
    if median > 1e-4 or share > 0.03:
        problems.append(f"the GPU did not trace the CPU's paths (median {median:.1e})")
    difference = quantize(linear_to_srgb(np.clip(np.abs(a - b) * 50.0, 0.0, 1.0)))
    return caption_strip(
        [
            ("CPU path tracer", cpu.image),
            ("GPU path tracer", gpu.image),
            ("|GPU - CPU| x 50", difference),
        ]
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--adapter", help="GPU adapter (index, part of a name, discrete, ...)")
    parser.add_argument("--quick", action="store_true", help="small images, few samples")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    print("1. CPU and GPU path tracers on the same paths")
    try:
        strip = compare(args.adapter, 48 if args.quick else 160, problems)
    except RenderError as exc:
        if exc.code.startswith("GPU_"):
            print(f"  no usable GPU ([{exc.code}] {exc.message})")
            return 2
        raise
    write(out / "gpu_path_vs_cpu.png", strip)
    print("2. Showcases on the GPU")
    for name, module, (width, height), (full, quick), path, options in SHOWCASES:
        if module == "castle_path_render" and importlib.util.find_spec("manifold3d") is None:
            print("  castle skipped: it needs aether3d[boolean]")
            continue
        if args.quick:
            width, height = width // 8, height // 8
        scene = build(module, width, height)
        result, seconds = timed_render(
            scene,
            width=width,
            height=height,
            backend="gpu_path",
            adapter=args.adapter,
            path={"samples": quick if args.quick else full, **path},
            **options,
        )
        target = out / f"gpu_path_{name}.png"
        result.save(target)
        record(f"gpu_path_{name}", result, seconds, target)
        if result.backend["device"] != "gpu":
            problems.append(f"{name}: did not run on the GPU")
        # NO_LIGHTS: the Cornell box is lit by its emissive panel, not by light objects.
        problems += [
            f"{name}: {d.code}: {d.message}"
            for d in result.diagnostics
            if d.severity != "info" and d.code != "NO_LIGHTS"
        ]
    save_report(out / "gpu_path_showcase.json", "GPU path tracing showcases")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
