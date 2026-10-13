"""Benchmark procedural geometry: every modifier at three input sizes, and caching.

Run from the repository root:

    python benchmarks/bench_procedural.py [--json results.json] [--quick]

For each modifier and input size it reports the best wall-clock time of several
runs and the peak memory allocated during one run (``tracemalloc``: Python and
NumPy allocations; memory used inside the Manifold C++ library by the boolean
modifier is not included). The cache section times the Minecraft castle demo:
a cold build, a warm re-evaluation (every result cached) and the regeneration
after changing one parameter. ``--quick`` uses one run and skips the largest
inputs.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

import aether3d
from aether3d import Scene
from aether3d.procedural.modifiers import MODIFIER_TYPES, ModifierContext
from aether3d.procedural.params import validate_values
from aether3d.procedural.primitives import PRIMITIVES

ROOT = Path(__file__).resolve().parent.parent

SIZES = {  # uv_sphere segments x rings -> about 2 * segments * rings / 2 quads
    "small": (32, 16),
    "medium": (128, 64),
    "large": (320, 160),
}


def sphere(segments: int, rings: int, radius: float = 1.0):
    t = PRIMITIVES["uv_sphere"]
    values = {"radius": radius, "segments": segments, "rings": rings}
    return t.generate(validate_values(t.params, values, "s"), "s")


def grid(n: int):
    t = PRIMITIVES["grid"]
    values = {"x_size": 2.0, "y_size": 2.0, "x_segments": n, "y_segments": n}
    return t.generate(validate_values(t.params, values, "g"), "g")


def modifier(name: str, mesh, operand_mesh=None, **params):
    t = MODIFIER_TYPES[name]
    values = validate_values(t.params, params, name)
    t.check(values, name)
    ctx = ModifierContext(name, lambda spec, loc: operand_mesh)
    return lambda: t.apply(mesh, values, ctx)


def measure(fn, repeat: int) -> tuple[float, float]:
    """(best seconds, peak MiB allocated during one run)."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return min(times), peak / 2**20


def cases(size: str):
    segments, rings = SIZES[size]
    s = sphere(segments, rings)
    flat = grid(max(4, int(round(np.sqrt(s.face_count)))))
    out = {
        "transform": (s, modifier("transform", s, rotation=(10, 20, 30), scale=(1, -2, 1))),
        "mirror (merge)": (s, modifier("mirror", s, axes="x", origin=(1, 0, 0))),
        "array x10": (s, modifier("array", s, count=10, relative_offset=(1.1, 0, 0))),
        "array x10 merge": (flat, modifier("array", flat, count=10, merge=True)),
        "solidify": (flat, modifier("solidify", flat, thickness=0.05, even_thickness=True)),
        "subdivision L1": (s, modifier("subdivision", s, levels=1)),
        "bevel (every edge)": (s, modifier("bevel", s, width=0.002, limit="angle", angle=1)),
        "to_mesh (triangulate)": (s, s.to_mesh),
    }
    if size != "large":
        out["subdivision L2"] = (s, modifier("subdivision", s, levels=2))
    try:
        import manifold3d  # noqa: F401

        other = sphere(segments, rings, radius=0.8)
        shifted = other.with_(positions=other.positions + (0.6, 0.2, 0.1))
        out["boolean difference"] = (
            s,
            modifier("boolean", s, shifted, operation="difference", operand={"geometry": "x"}),
        )
    except ImportError:
        pass
    return out


def castle_cache(repeat: int) -> dict:
    spec = importlib.util.spec_from_file_location(
        "minecraft_castle", ROOT / "examples" / "minecraft_castle.py"
    )
    castle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(castle)

    def cold():
        scene = Scene()
        castle.build_castle(scene)
        for obj in scene:
            _ = obj.mesh
        return scene

    cold_time, cold_peak = measure(cold, repeat)
    scene = cold()

    def warm():
        for obj in scene:
            _ = obj.mesh

    warm_time, _ = measure(warm, repeat)
    proc = scene.procedural
    widths = iter(np.linspace(24.5, 40, 1000))

    def one_parameter():
        proc.set_parameter("castle_width", float(next(widths)))
        for obj in scene:
            _ = obj.mesh

    change_time, change_peak = measure(one_parameter, repeat)
    detail = iter([1, 2] * 1000)

    def detail_only():
        proc.set_parameter("detail_density", next(detail))
        for obj in scene:
            _ = obj.mesh

    detail_only()  # evaluate both values once, so the timed runs are cache hits
    detail_only()
    detail_time, _ = measure(detail_only, repeat)
    triangles = sum(o.mesh.triangle_count for o in scene)
    return {
        "triangles": triangles,
        "cold_build_s": cold_time,
        "cold_peak_mib": cold_peak,
        "warm_reevaluate_s": warm_time,
        "change_castle_width_s": change_time,
        "change_castle_width_peak_mib": change_peak,
        "toggle_detail_density_s": detail_time,
        "cache": proc.cache_info,
    }


def run(quick: bool = False) -> dict:
    repeat = 1 if quick else 3
    sizes = ["small", "medium"] if quick else list(SIZES)
    results: dict = {
        "aether3d": aether3d.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "modifiers": {},
    }
    for size in sizes:
        rows = {}
        for name, (mesh, fn) in cases(size).items():
            seconds, peak = measure(fn, repeat)
            rows[name] = {"input_faces": mesh.face_count, "seconds": seconds, "peak_mib": peak}
        results["modifiers"][size] = rows
    results["castle"] = castle_cache(repeat)
    return results


def print_table(data: dict) -> None:
    print(f"aether3d {data['aether3d']}, Python {data['python']}, NumPy {data['numpy']}")
    for size, rows in data["modifiers"].items():
        print(f"\n{size}:")
        print(f"  {'operation':24s} {'faces in':>9s} {'time':>10s} {'peak mem':>10s}")
        for name, row in rows.items():
            print(
                f"  {name:24s} {row['input_faces']:9d} {row['seconds'] * 1000:8.2f}ms "
                f"{row['peak_mib']:8.1f}MiB"
            )
    c = data["castle"]
    print(f"\ncastle demo ({c['triangles']} triangles):")
    print(
        f"  cold build + evaluate      {c['cold_build_s'] * 1000:8.1f}ms  "
        f"(peak {c['cold_peak_mib']:.1f}MiB)"
    )
    print(f"  warm re-evaluation         {c['warm_reevaluate_s'] * 1000:8.3f}ms (all cached)")
    print(f"  change castle_width        {c['change_castle_width_s'] * 1000:8.1f}ms")
    print(f"  detail_density back/forth  {c['toggle_detail_density_s'] * 1000:8.1f}ms (cache hits)")
    cache = c["cache"]
    print(
        f"  cache: {cache['entries']} polygon + {cache['meshes']} mesh entries, "
        f"{cache['bytes'] / 2**20:.1f}MiB, {cache['hits']} hits / {cache['misses']} misses"
    )


if __name__ == "__main__":
    data = run(quick="--quick" in sys.argv)
    print_table(data)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
