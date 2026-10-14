"""Benchmark the GPU renderer against the CPU renderer on identical scenes.

Run from the repository root (needs ``aether3d[gpu]``; the castle also needs
``aether3d[boolean]``):

    python benchmarks/bench_gpu.py [--quick] [--adapter SPEC] [--no-4k] [--json results.json]

For each scene and resolution, every backend renders the same scene from the
same camera with the same number of samples per pixel (4):

* ``cpu`` -- the CPU renderer, 2x2 supersampling (``samples=2``);
* ``gpu`` -- the GPU, 4x MSAA (the default for ``samples=2``);
* ``gpu-ss`` -- the GPU with the CPU's exact sample positions (``supersample=2``);
* ``gpu+shadows`` -- the GPU, 4x MSAA, with a 2048x2048 sun shadow map
  (the CPU renderer has no shadows: no CPU equivalent).

Times are wall-clock seconds of a complete ``scene.render()`` -- extraction,
uploads, command preparation, GPU execution *until the results are read back*
(mapping the output buffer waits for the GPU), and conversion to NumPy -- best
of several runs after warm-up renders (the GPU's at full size, which also lets a
laptop GPU leave its idle clocks; ``--quick``: fewer runs). The first GPU render
of the process, which creates the device and compiles the shaders, is reported
separately ("cold start"), as is PNG encoding (the same code for both
backends). GPU pass times come from GPU timestamp queries when the device has
them. Peak host memory is measured with ``tracemalloc`` (NumPy included); GPU
memory is the renderer's own estimate of its images, shadow maps, output
buffers and cached meshes.
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
from aether3d import Scene, create_cube, create_sphere, gpu_status, release_gpu

ROOT = Path(__file__).resolve().parent.parent
RESOLUTIONS = [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)]
MODES = {
    "cpu": {},
    "gpu": {"backend": "gpu"},
    "gpu-ss": {"backend": "gpu", "msaa": 1, "supersample": 2},
    "gpu+shadows": {"backend": "gpu", "shadows": {"lights": ["Sun"]}},
}


def hardware(adapter) -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    status = gpu_status(adapter=adapter)
    selected = status["selected"] or {}
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aether3d": aether3d.__version__,
        "wgpu": status["versions"],
        "gpu": selected.get("adapter"),
        "gpu_msaa": selected.get("msaa"),
        "cpu_renderer": "single-threaded NumPy",
    }


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "examples"))
    spec.loader.exec_module(module)
    return module


def sphere_scene(segments: int, rings: int) -> Scene:
    scene = Scene(f"Sphere {segments}x{rings}")
    scene.add_plane("Ground", size=20).color = "#557755"
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, segments, rings), position=(0, 0, 1))
    ball.appearance.configure(color="#3080c0", specular=0.4, shininess=40)
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2)
    cam = scene.add_camera("Camera", fov=35)
    cam.frame(direction=(-0.5, 1, -0.4), aspect=16 / 9)
    return scene


def instances_scene(n: int = 100) -> Scene:
    scene = Scene(f"{n * n} instances")
    cube = create_cube(0.7)
    for i in range(n):
        for j in range(n):
            obj = scene.add_object(f"Cube {i} {j}", mesh=cube, position=(i - n / 2, j - n / 2, 0))
            obj.color = (i / n, j / n, 0.5)
    scene.add_plane("Ground", size=n * 1.2, position=(-0.5, -0.5, -0.36)).color = "#444444"
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2)
    cam = scene.add_camera("Camera", fov=40)
    cam.frame(direction=(-0.5, 1, -0.8), aspect=16 / 9)
    return scene


def scenes(quick: bool) -> dict:
    found = {}
    if importlib.util.find_spec("manifold3d") is not None:
        found["castle (procedural)"] = lambda: _example("minecraft_castle_render").build_scene()
    found["hero (edited meshes)"] = lambda: _example("hero_render").build_scene()
    found["10,000 instanced cubes"] = instances_scene
    found["sphere ~10k triangles"] = lambda: sphere_scene(100, 50)
    found["sphere ~100k triangles"] = lambda: sphere_scene(320, 160)
    found["sphere ~1M triangles"] = lambda: sphere_scene(1000, 500)
    if quick:
        keep = ("castle (procedural)", "10,000 instanced cubes", "sphere ~1M triangles")
        found = {k: v for k, v in found.items() if k in keep}
    return found


def timed(scene: Scene, repeat: int, **options):
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = scene.render(**options)
        seconds = time.perf_counter() - t0
        if best is None or seconds < best[0]:
            best = (seconds, result)
    return best


def peak_memory(scene: Scene, **options) -> float:
    tracemalloc.start()
    scene.render(**options)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def measure(scene: Scene, width: int, height: int, mode: str, adapter, repeat: int) -> dict:
    options = {"width": width, "height": height, "samples": 2, **MODES[mode]}
    if mode != "cpu":
        options["adapter"] = adapter
        # Warm up at full size: pipelines, uploads, targets, and the GPU's clocks.
        for _ in range(3):
            scene.render(**options)
        repeat += 2
    else:
        scene.render(**{**options, "width": 64, "height": 36})  # warm up (caches)
    seconds, result = timed(scene, repeat, **options)
    t0 = time.perf_counter()
    png = result.png_bytes()
    encode = time.perf_counter() - t0
    row = {
        "mode": mode,
        "width": width,
        "height": height,
        "samples_per_pixel": result.stats["samples"] // (width * height),
        "triangles": result.stats["triangles"],
        "seconds": seconds,
        "timings": result.stats["timings"],
        "png_seconds": encode,
        "png_bytes": len(png),
        "peak_host_mib": peak_memory(scene, **options),
    }
    if mode != "cpu":
        resources = result.backend["resources"]
        row["gpu_timings"] = result.stats["gpu_timings"]
        row["gpu_mib"] = (resources["target_bytes"] + resources["mesh_cache"]["bytes"]) / 2**20
        row["draw_calls"] = result.stats["draw_calls"]
    return row


FRESH_PROCESS = """
import json, sys, time
t0 = time.perf_counter()
import aether3d
from aether3d import Scene
t1 = time.perf_counter()
scene = Scene()
scene.add_cube("Crate")
scene.add_camera("Cam", position=(3, -4, 2), look_at=(0, 0, 0))
adapter = sys.argv[1] if len(sys.argv) > 1 else None
t2 = time.perf_counter()
first = scene.render(backend="gpu", adapter=adapter, width=640, height=360)
t3 = time.perf_counter()
scene.render(backend="gpu", adapter=adapter, width=640, height=360)
t4 = time.perf_counter()
report = {"import_seconds": t1 - t0, "first_render_seconds": t3 - t2}
report["init_seconds"] = first.stats["timings"]["init"]
report["second_render_seconds"] = t4 - t3
print(json.dumps(report))
"""


def fresh_process(adapter) -> dict:
    """What a new process (e.g. one ``aether3d render --backend gpu``) pays before drawing."""
    import subprocess

    command = [sys.executable, "-c", FRESH_PROCESS] + ([adapter] if adapter else [])
    best = None
    for _ in range(3):
        done = subprocess.run(command, capture_output=True, text=True, check=True)
        run = json.loads(done.stdout.strip().splitlines()[-1])
        if best is None or run["first_render_seconds"] < best["first_render_seconds"]:
            best = run
    return best


def cold_start(adapter) -> dict:
    """The first GPU render after releasing the devices: device creation and pipelines."""
    scene = sphere_scene(32, 16)
    release_gpu()
    t0 = time.perf_counter()
    result = scene.render(backend="gpu", adapter=adapter, width=640, height=360)
    total = time.perf_counter() - t0
    t0 = time.perf_counter()
    scene.render(backend="gpu", adapter=adapter, width=640, height=360)
    warm = time.perf_counter() - t0
    return {
        "first_render_seconds": total,
        "device_and_shaders_seconds": result.stats["timings"]["init"],
        "device_seconds": result.backend["device_init_seconds"],
        "second_render_seconds": warm,
    }


def run(quick: bool, adapter, with_4k: bool) -> dict:
    repeat = 2 if quick else 3
    data = {
        "hardware": hardware(adapter),
        "fresh_process": fresh_process(adapter),
        "cold_start": cold_start(adapter),
        "scenes": {},
    }
    resolutions = RESOLUTIONS if with_4k else RESOLUTIONS[:3]
    if quick:
        resolutions = [(640, 360), (1920, 1080)]
    for name, build in scenes(quick).items():
        scene = build()
        rows = []
        for width, height in resolutions:
            for mode in MODES:
                runs = 1 if (mode == "cpu" and width > 1920) else repeat
                rows.append(measure(scene, width, height, mode, adapter, runs))
                print(
                    f"  {name:24s} {width}x{height} {mode:12s} {rows[-1]['seconds']:.3f}s",
                    flush=True,
                )
        data["scenes"][name] = rows
    return data


def print_table(data: dict) -> None:
    hw = data["hardware"]
    gpu = hw["gpu"] or {}
    print(f"\nCPU: {hw['cpu']} ({hw['logical_cpus']} logical CPUs), {hw['platform']}")
    print(
        f"GPU: {gpu.get('name')} ({gpu.get('backend')}, {gpu.get('type')}, driver "
        f"{gpu.get('driver')}); wgpu {hw['wgpu']['wgpu']}, wgpu-native {hw['wgpu']['wgpu_native']}"
    )
    print(f"Python {hw['python']}, NumPy {hw['numpy']}, Aether3D {hw['aether3d']}")
    fresh = data["fresh_process"]
    print(
        f"New process: import {fresh['import_seconds']:.2f} s, first GPU render "
        f"{fresh['first_render_seconds']:.2f} s (adapters, device and pipelines "
        f"{fresh['init_seconds']:.2f} s), second render {fresh['second_render_seconds']:.3f} s"
    )
    cold = data["cold_start"]
    print(
        f"GPU cold start: first render {cold['first_render_seconds']:.3f} s (device "
        f"{cold['device_seconds']:.3f} s, device + shaders "
        f"{cold['device_and_shaders_seconds']:.3f} s); second render "
        f"{cold['second_render_seconds']:.3f} s"
    )
    for name, rows in data["scenes"].items():
        print(f"\n{name} ({rows[0]['triangles']:,} triangles), 4 samples per pixel:")
        print(
            f"  {'size':>10s} {'cpu':>8s} {'gpu':>8s} {'speedup':>8s} {'gpu-ss':>8s} "
            f"{'+shadows':>9s} {'gpu pass':>9s} {'readback':>9s} {'png':>6s} {'host MiB':>9s} "
            f"{'gpu MiB':>8s}"
        )
        by_size: dict[tuple, dict] = {}
        for r in rows:
            by_size.setdefault((r["width"], r["height"]), {})[r["mode"]] = r
        for (w, h), modes in by_size.items():
            cpu, gpu = modes["cpu"], modes["gpu"]
            passes = sum(gpu.get("gpu_timings", {}).values())
            print(
                f"  {w:>4d}x{h:<5d} {cpu['seconds']:7.3f}s {gpu['seconds']:7.3f}s "
                f"{cpu['seconds'] / gpu['seconds']:7.1f}x {modes['gpu-ss']['seconds']:7.3f}s "
                f"{modes['gpu+shadows']['seconds']:8.3f}s {passes * 1e3:7.2f}ms "
                f"{gpu['timings']['readback'] * 1e3:7.1f}ms {gpu['png_seconds']:5.2f}s "
                f"{cpu['peak_host_mib']:4.0f}/{gpu['peak_host_mib']:<4.0f} {gpu['gpu_mib']:7.0f}"
            )


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    adapter = argv[argv.index("--adapter") + 1] if "--adapter" in argv else None
    status = gpu_status(adapter=adapter)
    if not status["available"]:
        print(f"GPU rendering is not available: {status['reason']}")
        return 1
    data = run(quick, adapter, with_4k="--no-4k" not in argv)
    print_table(data)
    if "--json" in argv:
        path = argv[argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
