"""Benchmark the CPU renderer: scenes of increasing size at three resolutions.

Run from the repository root (the castle needs ``aether3d[boolean]``):

    python benchmarks/bench_render.py [--json results.json] [--quick]

For each scene and resolution it reports the best wall-clock time of several
renders (``--quick``: one) split into stages -- procedural evaluation,
extraction, geometry (transform/cull/clip/setup), rasterization (visibility
buffer), shading and output (resolve/encode) -- plus PNG encoding, the
fragments rasterized, and the peak memory of one render measured with
``tracemalloc`` (NumPy allocations included). All renders use 2x2
antialiasing (4 samples per pixel) unless noted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

import aether3d
from aether3d import Scene, create_sphere

ROOT = Path(__file__).resolve().parent.parent
RESOLUTIONS = [(640, 360), (1280, 720), (1920, 1080)]


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aether3d": aether3d.__version__,
        "threads": "single-threaded NumPy (no parallel rendering)",
    }


def castle_scene() -> Scene:
    spec = importlib.util.spec_from_file_location(
        "minecraft_castle_render", ROOT / "examples" / "minecraft_castle_render.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "examples"))
    spec.loader.exec_module(module)
    return module.build_scene()


def sphere_scene(segments: int) -> Scene:
    scene = Scene(f"Sphere {segments}")
    scene.add_plane("Ground", size=20).color = "#557755"
    ball = scene.add_object(
        "Ball", mesh=create_sphere(1.0, segments, segments // 2), position=(0, 0, 1)
    )
    ball.appearance.configure(color="#3080c0", specular=0.4, shininess=40)
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2)
    cam = scene.add_camera("Camera", fov=35)
    cam.frame(direction=(-0.5, 1, -0.4), aspect=16 / 9)
    return scene


def instances_scene(n: int = 20) -> Scene:
    scene = Scene("Instances")
    cube = aether3d.create_cube(0.8)
    for i in range(n):
        for j in range(n):
            obj = scene.add_object(f"Cube {i} {j}", mesh=cube, position=(i - n / 2, j - n / 2, 0))
            obj.color = (i / n, j / n, 0.5)
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2)
    cam = scene.add_camera("Camera", fov=40)
    cam.frame(direction=(-0.5, 1, -0.8), aspect=16 / 9)
    return scene


def measure(scene: Scene, width: int, height: int, repeat: int, samples: int = 2) -> dict:
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = scene.render(width=width, height=height, samples=samples)
        png = result.png_bytes()
        total = time.perf_counter() - t0
        if best is None or total < best[0]:
            best = (total, result, len(png))
    total, result, png_size = best
    t0 = time.perf_counter()
    result.png_bytes()
    encode = time.perf_counter() - t0
    tracemalloc.start()
    scene.render(width=width, height=height, samples=samples)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings = {k: v for k, v in result.stats["timings"].items()}
    return {
        "width": width,
        "height": height,
        "samples": samples,
        "triangles": result.stats["triangles"],
        "fragments": result.stats["fragments"],
        "samples_covered": result.stats["samples_covered"],
        "seconds": total,
        "stages": timings,
        "png_encode_seconds": encode,
        "png_bytes": png_size,
        "peak_mib": peak / 2**20,
    }


def run(quick: bool = False) -> dict:
    repeat = 1 if quick else 3
    scenes = {}
    try:
        import manifold3d  # noqa: F401

        scenes["castle (procedural)"] = castle_scene
    except ImportError:
        pass
    scenes["sphere 65k triangles"] = lambda: sphere_scene(256)
    scenes["sphere 1M triangles"] = lambda: sphere_scene(1024)
    scenes["400 instanced cubes"] = instances_scene
    results = {"hardware": hardware(), "scenes": {}}
    for name, build in scenes.items():
        scene = build()
        rows = []
        resolutions = RESOLUTIONS[:2] if quick else RESOLUTIONS
        for width, height in resolutions:
            rows.append(measure(scene, width, height, repeat))
        results["scenes"][name] = rows
    return results


def print_table(data: dict) -> None:
    hw = data["hardware"]
    print(f"{hw['cpu']} ({hw['logical_cpus']} logical CPUs), {hw['platform']}")
    print(f"Python {hw['python']}, NumPy {hw['numpy']}, Aether3D {hw['aether3d']}; {hw['threads']}")
    for name, rows in data["scenes"].items():
        print(f"\n{name}:")
        print(
            f"  {'size':>10s} {'tris':>9s} {'fragments':>11s} {'total':>8s} {'geometry':>9s} "
            f"{'raster':>8s} {'shade':>8s} {'output':>8s} {'png':>7s} {'peak':>8s}"
        )
        for r in rows:
            s = r["stages"]
            print(
                f"  {r['width']:>4d}x{r['height']:<5d} {r['triangles']:9d} {r['fragments']:11,d} "
                f"{r['seconds']:7.2f}s {s['geometry']:8.3f}s {s['raster']:7.3f}s "
                f"{s['shade']:7.3f}s {s['output']:7.3f}s {r['png_encode_seconds']:6.2f}s "
                f"{r['peak_mib']:6.0f}MiB"
            )


if __name__ == "__main__":
    data = run(quick="--quick" in sys.argv)
    print_table(data)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
