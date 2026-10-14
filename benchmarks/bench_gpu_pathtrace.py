"""Benchmark the GPU path tracer (``backend="gpu_path"``) on the Stage 8 showcase scenes.

Run from the repository root (needs ``aether3d[gpu]``, a GPU, and ``aether3d[boolean]``
for the castle):

    python benchmarks/bench_gpu_pathtrace.py [--adapter SPEC] [--json results.json] [--quick]

Renders each showcase scene with the settings of its CPU render (size, samples, bounces,
clamp) on the GPU and reports the time, the rays and the rays per second. The CPU time
of the same render is read from the showcase's JSON record in ``examples/output`` (the
CPU path tracer with 8 processes on the development machine) -- nothing is re-rendered on
the CPU. A separate AO-only render measures traversal alone (camera and AO rays, almost
no shading). Every number is one measured run with the device already created.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import aether3d  # noqa: E402

# name, example module, CPU record (file, image), path options, render options
CASES = [
    ("cornell", "cornell_box", ("cornell_box.json", "cornell_path"), {"bounces": 8}, {}),
    (
        "metals",
        "metal_spheres",
        ("metals.json", "metals_path"),
        {"bounces": 4, "clamp": 3.0},
        {},
    ),
    (
        "window room",
        "window_room",
        ("window_room.json", "window_path"),
        {"bounces": 6},
        {"exposure": 1.0},
    ),
    ("castle", "castle_path_render", ("castle_path.json", "castle_path"), {"bounces": 4}, {}),
    (
        "hero",
        "hero_path_render",
        ("hero_path.json", "hero_path"),
        {"bounces": 4, "clamp": 3.0},
        {},
    ),
]


def cpu_record(file: str, image: str) -> dict | None:
    path = ROOT / "examples" / "output" / file
    if not path.exists():
        return None
    for entry in json.loads(path.read_text())["renders"]:
        if entry["image"] == image:
            return entry
    return None


def build(module: str, width: int, height: int):
    builder = __import__(module).build_scene
    if module in ("castle_path_render", "hero_path_render"):
        return builder(width, height)
    return builder()


def run(adapter, quick: bool) -> dict:
    results = {"aether3d": aether3d.__version__, "cases": []}
    for name, module, (file, image), path, options in CASES:
        cpu = cpu_record(file, image)
        if cpu is None:
            continue
        width, height, samples = cpu["width"], cpu["height"], cpu["samples"]
        if quick:
            width, height, samples = width // 4, height // 4, max(1, samples // 8)
        scene = build(module, width, height)
        warm = scene.render(
            width=32, height=18, backend="gpu_path", adapter=adapter, path={"samples": 1}
        )
        t0 = time.perf_counter()
        r = scene.render(
            width=width,
            height=height,
            backend="gpu_path",
            adapter=adapter,
            path={"samples": samples, **path},
            **options,
        )
        seconds = time.perf_counter() - t0
        row = {
            "scene": name,
            "width": width,
            "height": height,
            "samples": samples,
            **path,
            "gpu_seconds": round(seconds, 2),
            "rays": r.stats["rays"],
            "mrays_per_second": round(r.stats["rays_per_second"] / 1e6, 1),
            "adapter": warm.backend["adapter"]["name"],
        }
        if not quick:
            row["cpu_seconds"] = cpu["seconds"]
            row["cpu_processes"] = cpu["processes"]
            row["speedup"] = round(cpu["seconds"] / seconds, 1)
        results["cases"].append(row)
        if name == "castle":
            t0 = time.perf_counter()
            ao = scene.render(
                width=width,
                height=height,
                backend="gpu_path",
                adapter=adapter,
                path={"samples": max(1, samples // 8), "integrator": "ao"},
                ao={"samples": 4, "distance": 1.0},
            )
            results["traversal"] = {
                "scene": "castle, AO only (camera and AO rays)",
                "seconds": round(time.perf_counter() - t0, 2),
                "rays": ao.stats["rays"],
                "mrays_per_second": round(ao.stats["rays_per_second"] / 1e6, 1),
            }
    return results


def print_table(data: dict) -> None:
    for row in data["cases"]:
        cpu = ""
        if "cpu_seconds" in row:
            cpu = (
                f"  CPU ({row['cpu_processes']} processes) {row['cpu_seconds']:8.1f} s"
                f"  speed-up {row['speedup']:6.1f}x"
            )
        print(
            f"{row['scene']:12s} {row['width']}x{row['height']} {row['samples']:5d} spp "
            f"{row['bounces']} bounces: GPU {row['gpu_seconds']:7.2f} s, "
            f"{row['rays'] / 1e6:7.0f} M rays, {row['mrays_per_second']:6.1f} M rays/s{cpu}"
        )
    t = data.get("traversal")
    if t:
        print(f"{t['scene']}: {t['seconds']} s, {t['mrays_per_second']} M rays/s")


if __name__ == "__main__":
    argv = sys.argv[1:]
    adapter = argv[argv.index("--adapter") + 1] if "--adapter" in argv else None
    data = run(adapter, quick="--quick" in argv)
    print_table(data)
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(data, indent=2) + "\n")
