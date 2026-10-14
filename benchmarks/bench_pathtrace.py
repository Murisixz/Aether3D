"""Benchmark the path tracer (``backend="path"``) and compare it with the rasterizers.

Run from the repository root (the castle needs ``aether3d[boolean]``; the GPU
column needs ``aether3d[gpu]`` and a working adapter):

    python benchmarks/bench_pathtrace.py [--json results.json] [--quick]

For a Cornell-box-like room, the textured procedural castle and a
100k-triangle sphere it reports, per resolution and setting:

* wall-clock time of one render, split into scene extraction, BVH build,
  tracing (camera rays, material evaluation, direct-light sampling, BSDF
  bounces -- CPU seconds summed over worker processes), the pixel-center
  pass and output encoding; PNG encoding separately;
* rays traced and rays per second, samples per pixel per second;
* the peak Python memory of a one-process render (``tracemalloc``);
* the same image size on the CPU rasterizer and, when available, the GPU
  rasterizer -- to make plain how much slower path tracing is.

Micro-benchmarks measure BSDF evaluation and sampling, light sampling and
material (texture) evaluation throughput in isolation. Every number is a
single measured run on this machine; nothing is extrapolated.
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
sys.path.insert(0, str(ROOT / "examples"))


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    from aether3d.rendering.pathtrace.parallel import cpu_count, fork_available

    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "usable_cpus": cpu_count(),
        "parallel_tracing": fork_available(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aether3d": aether3d.__version__,
    }


def room() -> Scene:
    scene = Scene("Room")
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    white = scene.create_material("White", model="simple", base_color="#bfbfbf")
    red = scene.create_material("Red", model="simple", base_color="#bf1a1a")
    green = scene.create_material("Green", model="simple", base_color="#1abf1a")
    walls = {
        "Floor": ((0, 0, 0), (0, 0, 0), white),
        "Ceiling": ((0, 0, 2), (180, 0, 0), white),
        "Back": ((0, 1, 1), (90, 0, 0), white),
        "Left": ((-1, 0, 1), (0, 90, 0), red),
        "Right": ((1, 0, 1), (0, -90, 0), green),
    }
    for name, (position, rotation, material) in walls.items():
        wall = scene.add_plane(name, size=2, position=position)
        wall.rotation_euler = rotation
        wall.material = material
    lamp = scene.add_plane("Lamp", size=0.5, position=(0, 0, 1.99))
    lamp.rotation_euler = (180, 0, 0)
    lamp.material = scene.create_material(
        "Lamp", model="simple", emissive="#ffffff", emissive_strength=12, double_sided=False
    )
    box = scene.add_cube("Box", size=0.6, position=(0.3, 0.2, 0.3))
    box.rotation_euler = (0, 0, 20)
    box.material = white
    scene.add_camera("Camera", position=(0, -3.2, 1), look_at=(0, 0, 1), fov=40)
    return scene


def castle() -> Scene:
    from minecraft_castle_render import build_scene

    return build_scene(1280, 720, 2, materials=True)


def sphere() -> Scene:
    scene = Scene("Sphere")
    scene.add_plane("Ground", size=20).color = "#557755"
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 316, 158), position=(0, 0, 1))
    ball.material = scene.create_material("Chrome", base_color="#e0e0e0", metallic=1, roughness=0.2)
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2)
    cam = scene.add_camera("Camera", fov=35)
    cam.frame(direction=(-0.5, 1, -0.4), aspect=16 / 9)
    return scene


def measure(scene: Scene, width: int, height: int, path: dict, memory: bool) -> dict:
    t0 = time.perf_counter()
    result = scene.render(width=width, height=height, backend="path", path=path)
    total = time.perf_counter() - t0
    t0 = time.perf_counter()
    png = result.png_bytes()
    encode = time.perf_counter() - t0
    row = {
        "width": width,
        "height": height,
        **{k: path.get(k) for k in ("samples", "bounces", "workers")},
        "processes": result.backend["processes"],
        "seconds": total,
        "timings": result.stats["timings"],
        "rays": result.stats["rays"],
        "rays_per_second": result.stats["rays_per_second"],
        "samples_per_second": width * height * path["samples"] / result.stats["timings"]["trace"],
        "png_encode_seconds": encode,
        "png_bytes": len(png),
        "noise": result.stats["noise"]["mean_relative_error"],
    }
    if memory:
        tracemalloc.start()
        scene.render(width=width, height=height, backend="path", path={**path, "workers": 1})
        row["peak_mib_one_process"] = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
    return row


def rasterizers(scene: Scene, width: int, height: int) -> dict:
    out = {}
    t0 = time.perf_counter()
    scene.render(width=width, height=height, samples=2)
    out["cpu_raster_seconds"] = time.perf_counter() - t0
    if importlib.util.find_spec("wgpu") is not None:
        try:
            scene.render(width=width, height=height, backend="gpu")  # warm-up: device, shaders
            t0 = time.perf_counter()
            r = scene.render(width=width, height=height, backend="gpu")
            out["gpu_raster_seconds"] = time.perf_counter() - t0
            out["gpu_adapter"] = r.backend["adapter"]["name"]
        except aether3d.RenderError as exc:
            out["gpu_raster_seconds"] = None
            out["gpu_error"] = exc.code
    return out


def micro() -> dict:
    """Throughput of the path tracer's building blocks, in millions per second."""
    from aether3d.rendering.pathtrace.bsdf import BSDF
    from aether3d.rendering.pathtrace.lights import LightSet
    from aether3d.rendering.pathtrace.sampling import path_keys
    from aether3d.rendering.pathtrace.scene import PathScene
    from aether3d.rendering.scene_data import extract_render_scene

    rng = np.random.default_rng(0)
    n = 200_000
    normal = np.tile([0.0, 0.0, 1.0], (n, 1))
    view = rng.normal(size=(n, 3))
    view[:, 2] = np.abs(view[:, 2]) + 0.1
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    t0 = time.perf_counter()
    bsdf = BSDF.build(
        normal,
        view,
        rng.random((n, 3)),
        rng.random(n),
        rng.random(n),
        np.full(n, 0.5),
        np.ones(n, bool),
    )
    build = time.perf_counter() - t0
    wi = rng.normal(size=(n, 3))
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    t0 = time.perf_counter()
    bsdf.eval(wi)
    evaluate = time.perf_counter() - t0
    t0 = time.perf_counter()
    bsdf.sample(rng.random(n), rng.random(n), rng.random(n))
    sample = time.perf_counter() - t0
    out = {
        "bsdf_build_m_per_s": n / build / 1e6,
        "bsdf_eval_m_per_s": n / evaluate / 1e6,
        "bsdf_sample_m_per_s": n / sample / 1e6,
    }
    scene = room()
    rs = extract_render_scene(scene, scene.active_camera, default_lights=False)
    ps = PathScene(rs)
    lights = LightSet.build(ps)
    points = rng.uniform(-0.9, 0.9, (n, 3)) + [0, 0, 1]
    keys = path_keys(0, np.arange(n), np.zeros(n))
    t0 = time.perf_counter()
    lights.sample_emitter(ps, points, keys, 0, 0)
    out["emitter_sample_m_per_s"] = n / (time.perf_counter() - t0) / 1e6
    if importlib.util.find_spec("manifold3d") is not None:
        # Material evaluation (image textures, normal maps, procedural graphs) at the
        # castle's camera hits.
        cs = castle()
        rs = extract_render_scene(cs, cs.active_camera)
        ps = PathScene(rs)
        o, d = camera_rays(rs, 640, 360)
        hits, it = ps.trace(o, d)
        t0 = time.perf_counter()
        ps.shade(it)
        out["castle_material_eval_m_per_s"] = it.rows.size / (time.perf_counter() - t0) / 1e6
    return out


def camera_rays(rs, width, height):
    frame = rs.camera.frame
    hx, hy = rs.camera.camera.half_extents(width / height)
    xs, ys = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 - 1, 1 - (np.arange(height) + 0.5) / height * 2
    )
    d = frame.forward + xs.ravel()[:, None] * hx * frame.right + ys.ravel()[:, None] * hy * frame.up
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.broadcast_to(frame.position, d.shape).copy(), d


def run(quick: bool) -> dict:
    scenes = {"room (Cornell-like, 14 triangles)": room, "sphere (100k triangles, chrome)": sphere}
    if importlib.util.find_spec("manifold3d") is not None:
        scenes["castle (textured, procedural)"] = castle
    sizes = [(320, 180)] if quick else [(320, 180), (640, 360)]
    settings = [{"samples": 16, "bounces": 0}, {"samples": 16, "bounces": 4}]
    results = {"hardware": hardware(), "scenes": {}, "micro": micro()}
    for name, build in scenes.items():
        scene = build()
        rows = []
        for width, height in sizes:
            for path in settings:
                for workers in (1, 0):
                    row = measure(
                        scene,
                        width,
                        height,
                        {**path, "workers": workers},
                        memory=workers == 1 and width == sizes[0][0] and path["bounces"] == 4,
                    )
                    rows.append(row)
            rows[-1].update(rasterizers(scene, width, height))
        results["scenes"][name] = rows
    return results


def print_table(data: dict) -> None:
    hw = data["hardware"]
    print(f"{hw['cpu']} ({hw['logical_cpus']} logical CPUs), {hw['platform']}")
    print(f"Python {hw['python']}, NumPy {hw['numpy']}, Aether3D {hw['aether3d']}")
    m = data["micro"]
    print(
        "micro: BSDF eval {bsdf_eval_m_per_s:.1f} M/s, sample {bsdf_sample_m_per_s:.1f} M/s, "
        "emitter samples {emitter_sample_m_per_s:.1f} M/s".format(**m)
    )
    if "castle_material_eval_m_per_s" in m:
        print(f"micro: castle material evaluation {m['castle_material_eval_m_per_s']:.2f} M/s")
    for name, rows in data["scenes"].items():
        print(f"\n{name}:")
        print(
            f"  {'size':>9s} {'spp':>4s} {'b':>2s} {'proc':>4s} {'total':>8s} {'bvh':>6s} "
            f"{'trace':>8s} {'Mrays/s':>8s} {'spp/s(M)':>8s} {'peak':>7s} {'raster':>8s} "
            f"{'gpu':>7s}"
        )
        for r in rows:
            t = r["timings"]
            peak = f"{r['peak_mib_one_process']:5.0f}MB" if "peak_mib_one_process" in r else ""
            raster = f"{r['cpu_raster_seconds']:7.2f}s" if "cpu_raster_seconds" in r else ""
            gpu = r.get("gpu_raster_seconds")
            gpu = f"{gpu:6.3f}s" if isinstance(gpu, float) else ""
            print(
                f"  {r['width']:>4d}x{r['height']:<4d} {r['samples']:4d} {r['bounces']:2d} "
                f"{r['processes']:4d} {r['seconds']:7.2f}s {t['bvh']:5.2f}s {t['trace']:7.2f}s "
                f"{r['rays_per_second'] / 1e6:8.2f} {r['samples_per_second'] / 1e6:8.3f} "
                f"{peak:>7s} {raster:>8s} {gpu:>7s}"
            )


if __name__ == "__main__":
    data = run(quick="--quick" in sys.argv)
    print_table(data)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
