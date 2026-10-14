"""Benchmark the ray tracing core: BVH construction and ray traversal (CPU, NumPy).

Run from the repository root:

    python benchmarks/bench_raytrace.py [--json results.json] [--quick]

For meshes of about 1k, 100k and 1M triangles (``--quick``: the first two)
and a scene of 400 instances of one mesh it reports:

* BLAS build time (binned SAH over the mesh's triangles) and the rebuild
  time when the BLAS comes from the cache (top level only);
* tree statistics (nodes, depth) and the structure's memory;
* closest-hit and any-hit throughput for coherent camera rays (320x240,
  pinhole) and for incoherent rays (random directions from points on the
  surfaces, like diffuse bounces), in millions of rays per second;
* the peak memory of the build, measured with ``tracemalloc``.

Everything runs single-threaded in NumPy.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

import aether3d
from aether3d import Scene, create_cube, create_sphere
from aether3d.rendering.raytrace import BLASCache, SceneAccel
from aether3d.rendering.scene_data import extract_render_scene


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
        "threads": "single-threaded NumPy",
    }


def sphere_scene(segments: int) -> Scene:
    scene = Scene()
    scene.add_object("Ball", mesh=create_sphere(1.0, segments, segments // 2))
    scene.add_camera("Camera", position=(0, -4, 0))
    return scene


def instances_scene(n: int = 20) -> Scene:
    scene = Scene()
    cube = create_cube(0.8)
    for i in range(n):
        for j in range(n):
            scene.add_object(f"Cube {i} {j}", mesh=cube, position=(i - n / 2, 0, j - n / 2))
    scene.add_camera("Camera", position=(0, -25, 0))
    return scene


def camera_rays(eye, width=320, height=240, spread=0.35):
    xs, ys = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 - 1, 1 - (np.arange(height) + 0.5) / height * 2
    )
    d = np.stack([xs.ravel() * spread, np.ones(width * height), ys.ravel() * spread], axis=1)
    return np.tile(np.asarray(eye, dtype=np.float64), (width * height, 1)), d


def surface_rays(accel: SceneAccel, o: np.ndarray, d: np.ndarray, rng):
    """Random directions leaving the surface points the camera rays hit."""
    hits = accel.closest_hit(o, d)
    k = np.flatnonzero(hits.hit)
    p = o[k] + d[k] * hits.t[k, None]
    dirs = rng.normal(size=p.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return p + dirs * 1e-6, dirs, hits.inst[k], hits.prim[k]


def throughput(fn, rays: int) -> float:
    t0 = time.perf_counter()
    fn()
    return rays / (time.perf_counter() - t0) / 1e6


def measure(scene: Scene, eye) -> dict:
    rs = extract_render_scene(scene, scene.active_camera)
    cache = BLASCache()
    tracemalloc.start()
    t0 = time.perf_counter()
    accel = SceneAccel.build(rs, cache)
    build = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t0 = time.perf_counter()
    SceneAccel.build(rs, cache)
    rebuild = time.perf_counter() - t0
    o, d = camera_rays(eye)
    rng = np.random.default_rng(0)
    so, sd, si, sp = surface_rays(accel, o, d, rng)
    accel.traced.__init__()
    rates = {
        "camera_closest": throughput(lambda: accel.closest_hit(o, d), o.shape[0]),
        "camera_any": throughput(lambda: accel.occluded(o, d), o.shape[0]),
        "bounce_closest": throughput(
            lambda: accel.closest_hit(so, sd, skip_inst=si, skip_tri=sp), so.shape[0]
        ),
        "bounce_any": throughput(
            lambda: accel.occluded(so, sd, skip_inst=si, skip_tri=sp), so.shape[0]
        ),
    }
    info = accel.info
    return {
        "triangles": info["triangles"],
        "instances": info["instances"],
        "build_seconds": build,
        "rebuild_cached_seconds": rebuild,
        "nodes": info["nodes"],
        "max_depth": info["max_depth"],
        "structure_mib": info["bytes"] / 2**20,
        "build_peak_mib": peak / 2**20,
        "mrays_per_second": rates,
        "node_visits_per_ray": accel.traced.node_visits
        / max(1, accel.traced.rays + accel.traced.shadow_rays),
    }


def run(quick: bool = False) -> dict:
    scenes = {
        "sphere ~1k triangles": (lambda: sphere_scene(32), (0, -4, 0)),
        "sphere ~100k triangles": (lambda: sphere_scene(316), (0, -4, 0)),
    }
    if not quick:
        scenes["sphere ~1M triangles"] = (lambda: sphere_scene(1000), (0, -4, 0))
    scenes["400 instances of a cube"] = (instances_scene, (0, -25, 0))
    results = {"hardware": hardware(), "scenes": {}}
    for name, (build, eye) in scenes.items():
        results["scenes"][name] = measure(build(), eye)
    return results


def print_table(data: dict) -> None:
    hw = data["hardware"]
    print(f"{hw['cpu']} ({hw['logical_cpus']} logical CPUs), {hw['platform']}")
    print(f"Python {hw['python']}, NumPy {hw['numpy']}, Aether3D {hw['aether3d']}; {hw['threads']}")
    print(
        f"\n{'scene':26s} {'tris':>9s} {'build':>8s} {'cached':>8s} {'depth':>5s} {'MiB':>6s} "
        f"{'peak':>6s} | Mrays/s: {'cam':>6s} {'cam-any':>7s} {'bounce':>6s} {'b-any':>6s}"
    )
    for name, r in data["scenes"].items():
        m = r["mrays_per_second"]
        print(
            f"{name:26s} {r['triangles']:9d} {r['build_seconds']:7.3f}s "
            f"{r['rebuild_cached_seconds']:7.4f}s {r['max_depth']:5d} {r['structure_mib']:6.1f} "
            f"{r['build_peak_mib']:6.0f} |          {m['camera_closest']:6.3f} "
            f"{m['camera_any']:7.3f} {m['bounce_closest']:6.3f} {m['bounce_any']:6.3f}"
        )


if __name__ == "__main__":
    data = run(quick="--quick" in sys.argv)
    print_table(data)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
