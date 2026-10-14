"""Render the procedural castle, the blocky hero and the primitives on the GPU.

Run from the repository root (needs ``pip install 'aether3d[gpu,boolean]'``):

    python examples/gpu_render.py [OUTPUT_DIR] [--width W] [--height H]
        [--adapter SPEC] [--require-gpu]

1. Reports the GPU: the adapters wgpu finds and the one the renders use.
2. Renders the castle of ``examples/minecraft_castle_render.py`` with its
   materials (image textures with a normal map, procedural textures and
   glowing torches) three ways, from the same scene and camera, at 1920x1080
   by default:

   * ``castle_cpu.png`` -- the CPU renderer, 2x2 samples per pixel;
   * ``castle_gpu.png`` -- the GPU, 4x MSAA (the same 4 samples per pixel);
   * ``castle_gpu_shadows.png`` -- the GPU, 4x MSAA and sun shadows;
   * ``castle_comparison.png`` -- the three side by side at half size
     (CPU | GPU | GPU with shadows).

3. Checks the GPU against the CPU renderer. In parity mode (the CPU's own
   sample positions) the GPU shows the same object at every pixel except
   along edges, and the same colors, textures included (a few texels may
   round differently). With 4x MSAA it samples other positions and shades
   once per pixel -- where the CPU shades every sample -- so it differs along
   silhouettes and in the finest texture detail; it is checked to show the
   same objects elsewhere, with the same average colors. Also checks that
   shadows darken the ground behind the walls and trees, that GPU renders
   are deterministic, and that ``aether3d render --backend gpu`` reproduces
   the pixels exactly from the saved project.
4. Renders the Stage 3 hero (with its materials) and the primitives scene
   on the GPU with shadows (``hero_gpu.png``, ``primitives_gpu.png``).
5. Prints CPU and GPU timings, the first GPU render (which creates the
   device and compiles the shaders) separately from the next ones.

Every image is drawn by Aether3D: GPU images record the adapter in their
``Source`` PNG text chunk. Without a usable GPU the script says so, writes
only the CPU image as ``castle_cpu_only.png`` and exits with status 0 (or 1
with ``--require-gpu``). Any failed check exits with status 1.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import first_render  # noqa: E402
import hero_render  # noqa: E402
import minecraft_castle_render  # noqa: E402

from aether3d import gpu_status  # noqa: E402
from aether3d.rendering.color import linear_to_srgb, quantize  # noqa: E402
from aether3d.rendering.png import write_png  # noqa: E402

SHADOWS = {"lights": ["Sun"], "map_size": 4096}


def edges(ids: np.ndarray) -> np.ndarray:
    """Pixels whose 3x3 neighbourhood shows more than one object."""
    h, w = ids.shape
    pad = np.pad(ids, 1, mode="edge")
    out = np.zeros(ids.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] != ids
    return out


def half_size(result) -> np.ndarray:
    """A 2x2 box-filtered half-size copy, averaged in linear light like the renderers."""
    h, w = result.height // 2 * 2, result.width // 2 * 2
    lin = result.linear[:h, :w].astype(np.float64)
    small = lin.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    return quantize(linear_to_srgb(small))


def report_gpu(adapter) -> dict | None:
    status = gpu_status(adapter=adapter)
    if status["versions"]:
        v = status["versions"]
        print(f"   wgpu {v['wgpu']} (wgpu-native {v['wgpu_native']})")
    for a in status["adapters"]:
        kind = f"{a['type']}, software" if a["software"] else a["type"]
        print(f"   [{a['index']}] {a['name']} ({a['backend']}, {kind})")
    if not status["available"]:
        print(f"   GPU rendering is NOT available: {status['reason']}")
        return None
    selected = status["selected"]
    a = selected["adapter"]
    print(
        f"   using [{a['index']}] {a['name']} ({a['backend']}), MSAA "
        f"{', '.join(map(str, selected['msaa']))}"
    )
    if a["software"]:
        print(
            "   note: a software adapter -- the GPU pipeline runs on the CPU (not hardware speed)"
        )
    return selected


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--adapter", help="GPU adapter (default: the best one)")
    parser.add_argument("--require-gpu", action="store_true", help="fail if no GPU works")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # The saved project and its textures live in a scratch directory, removed at the end.
    work = Path(tempfile.mkdtemp(prefix="aether3d-gpu-example-"))
    try:
        return run(args, out, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run(args: argparse.Namespace, out: Path, work: Path) -> int:
    w, h = args.width, args.height
    failures: list[str] = []

    print("1. The GPU")
    selected = report_gpu(args.adapter)

    print("2. Rendering the procedural castle with its materials")
    textures = minecraft_castle_render.copy_textures(work)
    scene = minecraft_castle_render.build_scene(w, h, 2, materials=True, textures=textures)
    t0 = time.perf_counter()
    cpu = scene.render(width=w, height=h, samples=2)
    t_cpu = time.perf_counter() - t0
    if selected is None:
        path = cpu.save(out / "castle_cpu_only.png")
        print(f"   CPU renderer only (no GPU): wrote {path} in {t_cpu:.2f} s")
        return 1 if args.require_gpu else 0
    gpu_options = {"backend": "gpu", "adapter": args.adapter, "width": w, "height": h, "samples": 2}
    t0 = time.perf_counter()
    gpu = scene.render(**gpu_options)
    t_cold = time.perf_counter() - t0
    t_warm = float("inf")
    for _ in range(3):  # the best of three: a single render is at the mercy of GPU clocks
        t0 = time.perf_counter()
        again = scene.render(**gpu_options)
        t_warm = min(t_warm, time.perf_counter() - t0)
    shadowed = scene.render(**gpu_options, shadows=SHADOWS)
    for name, result in (("cpu", cpu), ("gpu", gpu), ("gpu_shadows", shadowed)):
        path = result.save(out / f"castle_{name}.png")
        print(f"   wrote {path}")
    gutter = np.full((h // 2, 8, 3), 24, dtype=np.uint8)
    panels = [half_size(cpu), gutter, half_size(gpu), gutter, half_size(shadowed)]
    path = write_png(out / "castle_comparison.png", np.concatenate(panels, axis=1))
    print(f"   wrote {path} (CPU | GPU | GPU with shadows)")
    aa = gpu.backend["antialiasing"]
    print(
        f"   CPU {cpu.settings['samples']}x{cpu.settings['samples']} samples per pixel: "
        f"{t_cpu:.2f} s; GPU {aa['msaa']}x MSAA: first render {t_cold:.2f} s (device and "
        f"shaders {gpu.stats['timings']['init']:.2f} s), next (best of 3) {t_warm:.3f} s "
        f"({t_cpu / t_warm:.0f}x faster), with shadows {shadowed.stats['timings']['total']:.3f} s"
    )

    print("3. Checking the GPU images against the CPU renderer")
    # Parity mode samples the CPU's own positions: the objects agree but along sub-pixel
    # edges, and so do the colors, textures included -- but where a sample sits exactly on a
    # crease between two faces (either is correct) or a texel boundary rounds the other way.
    parity = scene.render(**{**gpu_options, "msaa": 1, "supersample": 2})
    same = parity.object_index == cpu.object_index
    cpu_edges = edges(cpu.object_index)
    away = ~cpu_edges & ~edges(parity.object_index)
    diff = np.abs(parity.image.astype(int) - cpu.image.astype(int)).max(axis=-1)
    close = (diff[away] <= 2).mean()
    if not np.all(cpu_edges[~same]) or close < 0.995 or diff[away].mean() > 0.25:
        failures.append("the GPU in parity mode differs from the CPU away from edges")
    print(
        f"   parity mode (the CPU's sample positions): the same object at {same.mean():.3%} "
        f"of pixels (all others on edges); away from edges {close:.2%} of the pixels within "
        f"2 codes (mean difference {diff[away].mean():.3f})"
    )
    # 4x MSAA samples other positions and shades once per pixel (the CPU shades every
    # sample): objects agree except along silhouettes, and their average colors agree.
    same = gpu.object_index == cpu.object_index
    near = cpu_edges | edges(gpu.object_index)
    if same.mean() < 0.97 or not np.all(near[~same]):
        failures.append("the MSAA image shows other objects than the CPU image away from edges")
    worst, worst_name = 0.0, ""
    for o in cpu.objects:
        both = (cpu.object_index == o["index"]) & (gpu.object_index == o["index"]) & ~near
        if both.sum() >= 100:
            delta = float(np.abs(gpu.image[both].mean(0) - cpu.image[both].mean(0)).max())
            if delta > worst:
                worst, worst_name = delta, o["name"]
    if worst > 5:
        failures.append(f"{worst_name}: the MSAA average color differs by {worst:.1f} codes")
    print(
        f"   4x MSAA: the same object at {same.mean():.2%} of pixels (all others on edges); "
        f"average colors per object within {worst:.1f} codes (largest: {worst_name})"
    )
    reported = [d.code for r in (cpu, gpu) for d in r.diagnostics if d.severity != "info"]
    if reported:  # (a software adapter is reported as information)
        failures.append(f"unexpected diagnostics: {reported}")
    names = {o["name"]: o["visible_samples"] for o in gpu.objects}
    missing = [
        n
        for n in ("Ground", "Curtain Wall", "Towers", "Keep", "Watchtower Roof")
        if names.get(n, 0) == 0
    ]
    if missing:
        failures.append(f"not visible on the GPU: {missing}")
    ground = [o["index"] for o in gpu.objects if o["name"] == "Ground"][0]
    on_ground = (gpu.object_index == ground) & (shadowed.object_index == ground)
    dark = on_ground & (shadowed.linear.sum(-1) < 0.7 * gpu.linear.sum(-1))
    fraction = dark.sum() / max(on_ground.sum(), 1)
    if not 0.03 < fraction < 0.7:
        failures.append(f"implausible shadowed ground fraction {fraction:.2f}")
    print(f"   sun shadows cover {fraction:.0%} of the visible ground")
    if not np.array_equal(again.image, gpu.image):
        failures.append("two GPU renders differ")
    project = work / "gpu_castle.aether"
    scene.save(project)  # its textures are in work/textures: saved as relative paths
    with tempfile.TemporaryDirectory() as tmp:
        cli_png = Path(tmp) / "cli.png"
        command = [sys.executable, "-m", "aether3d", "render", str(project), "-o", str(cli_png)]
        command += ["--backend", "gpu", "--width", str(w), "--height", str(h), "--samples", "2"]
        command += ["--shadows", "--shadow-lights", "Sun", "--shadow-map-size", "4096"]
        if args.adapter:
            command += ["--adapter", args.adapter]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            failures.append(f"aether3d render failed: {run.stdout[-300:]} {run.stderr[-300:]}")
        elif cli_png.read_bytes() != shadowed.png_bytes():
            failures.append("aether3d render --backend gpu produced different pixels")
        else:
            print("   GPU renders are deterministic, and `aether3d render` reproduces them exactly")
    print(f"   reproduce: aether3d render PROJECT --backend gpu --shadows --width {w} --height {h}")

    print("4. The blocky hero (with its materials) and the primitives on the GPU")
    for name, other in (
        ("hero", hero_render.build_scene(w, h, 2, materials=True)),
        ("primitives", first_render.build_scene(w, h, 2)),
    ):
        result = other.render(backend="gpu", adapter=args.adapter, width=w, height=h, shadows=True)
        path = result.save(out / f"{name}_gpu.png")
        seen = sum(1 for o in result.objects if o["visible_samples"] > 0)
        print(f"   wrote {path}: {seen} of {len(result.objects)} objects visible")
        if seen < len(result.objects) - 1:
            failures.append(f"{name}: objects missing")

    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
