"""Procedural worlds: generation, streaming and rendering, measured.

Run from the repository root:

    python benchmarks/bench_worlds.py [--json OUT.json] [--label NAME] [--quick]
        [--only regional,terrain,vegetation,blocks,volumes,queries,render]
        [--compare BASELINE.json]

Every group runs in its own Python process, so caches and memory peaks of one group never
influence another; each reports its process's peak RSS. Timings are wall clock (first,
"cold", calls generate everything they need; "warm" calls reuse the caches).

Groups:

* ``regional`` -- the regional hydrology stage of ``temperate`` (4 km, 16 m cells) and
  ``alpine_valley`` (8 km), with the native kernels and with ``AETHER3D_NATIVE=0``;
* ``terrain`` -- what a terrain world adds to frames: the first frame (cold), the same
  frame again (warm), and a camera flying 1.5 km (30 frames);
* ``vegetation`` -- placement and culling for a 2.5 km view (cold and warm);
* ``blocks`` -- block chunks (generation and meshing per chunk) and a first and moving
  view of ``blocky_adventure``;
* ``volumes`` -- cave cubes (classification per column, meshing per cube);
* ``queries`` -- ``sample`` throughput, ``ground_path``, ``find_sites``;
* ``render`` -- complete GPU frames at 1920x1080 (wall clock, world contribution
  included) of a terrain and a block world along camera paths.

Results depend on the machine: compare runs on the same machine only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
GROUPS = ("regional", "terrain", "vegetation", "blocks", "volumes", "queries", "render")


def _peak() -> float | None:
    from bench_engine import peak_rss_mib

    return peak_rss_mib()


def _view(scene, world, eye, target, height=1080, aspect=16 / 9):
    from aether3d.rendering.scene_data import extract_render_scene, resolve_camera
    from aether3d.worlds.render import SHADOW_MARGIN
    from aether3d.worlds.tiles import view_from_camera

    cam = scene.get_by_name("BenchCam") if any(o.name == "BenchCam" for o in scene) else None
    if cam is None:
        cam = scene.add_camera("BenchCam", position=eye, look_at=target, fov=55, far=30000)
    else:
        cam.transform.position = eye
        cam.look_at(target)
    rs = extract_render_scene(scene, resolve_camera(scene, cam), aspect=aspect)
    d = world.definition["streaming"]["view_distance"]
    return rs, view_from_camera(rs.camera.camera, rs.camera.frame, height, aspect, d, SHADOW_MARGIN)


def _contribute(scene, rs, height=1080, aspect=16 / 9):
    from aether3d.worlds.render import contribute

    t0 = time.perf_counter()
    out = contribute(scene, rs, height, aspect)
    return time.perf_counter() - t0, out


def group_regional(quick: bool) -> dict[str, Any]:
    from aether3d import Scene, _native

    out = {"native": _native.module() is not None}
    cases = [("temperate", 4096.0, 16.0)] + ([] if quick else [("alpine_valley", None, None)])
    for preset, size, cell in cases:
        scene = Scene("R")
        over = {"hydrology": {"cell": cell}} if cell else {}
        kw = {"size": size} if size else {}
        w = scene.worlds.create("W", preset, seed=7, overrides=over, **kw)
        t0 = time.perf_counter()
        reg = w.runtime.regional()
        s = reg.summary()
        out[preset] = {
            "seconds": round(time.perf_counter() - t0, 3),
            "nodes": s["nodes"],
            "rivers": s["rivers"],
            "lakes": s["lakes"],
            "iterations": s["iterations"],
            "timings": s["timings"],
        }
    return out


def group_terrain(quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    scene = Scene("T")
    w = scene.worlds.create("W", "alpine_valley", seed=7, size=4096.0 if quick else 8192.0)
    t0 = time.perf_counter()
    w.runtime.regional()
    regional = time.perf_counter() - t0
    g = w.height(-1500.0, -1500.0)
    rs, _ = _view(scene, w, (-1500.0, -1500.0, g + 120.0), (0.0, 0.0, g))
    cold, _ = _contribute(scene, rs)
    frame = dict(w.runtime.last_frame)
    warm, _ = _contribute(scene, rs)
    frames = 8 if quick else 30
    times = []
    for k in range(frames):
        t = k / max(frames - 1, 1)
        x, y = -1500.0 + 1060.0 * t, -1500.0 + 1060.0 * t
        gz = w.height(x, y)
        rs, _ = _view(scene, w, (x, y, gz + 120.0), (x + 900.0, y + 900.0, gz))
        dt, _ = _contribute(scene, rs)
        times.append(dt)
    stats = w.runtime.stats()
    return {
        "regional_seconds": round(regional, 3),
        "first_frame_seconds": round(cold, 3),
        "warm_frame_seconds": round(warm, 4),
        "first_frame": frame,
        "path_frames": frames,
        "path_seconds_median": round(sorted(times)[len(times) // 2], 4),
        "path_seconds_max": round(max(times), 4),
        "tiles_generated": stats["tiles_generated"],
        "cached_bytes": stats["memory"]["cached_bytes"],
        "process_rss_mib": round((stats["memory"]["process_rss_bytes"] or 0) / 2**20, 1),
    }


def group_vegetation(quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import Scene
    from aether3d.worlds.tiles import View
    from aether3d.worlds.vegetation import vegetation_of

    scene = Scene("V")
    w = scene.worlds.create("W", "temperate", seed=11, size=4096.0)
    w.runtime.regional()
    veg = vegetation_of(w.runtime)
    eye = np.array([960.0, 360.0, w.height(960.0, 360.0) + 25.0])
    view = View(eye, np.array([0.8, 0.6, 0.0]), 1080 / math.radians(55), 12000.0, None)
    t0 = time.perf_counter()
    vis = veg.visible(view)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    veg.visible(view)
    warm = time.perf_counter() - t0
    s = veg.stats()
    return {
        "cold_seconds": round(cold, 3),
        "warm_seconds": round(warm, 4),
        "tiles_placed": s["tiles_placed"],
        "plants_placed": s["plants_placed"],
        "candidates": s["candidates"],
        "drawn": int(len(vis["species"])),
    }


def group_blocks(quick: bool) -> dict[str, Any]:
    from aether3d import Scene
    from aether3d.worlds.voxels import _blocks

    scene = Scene("B")
    w = scene.worlds.create("B", "blocky_adventure", seed=5)
    w.runtime.regional()
    bw = _blocks(w.runtime)
    coords = [(i, j) for i in range(8) for j in range(8)]
    t0 = time.perf_counter()
    bw.chunks(coords)
    gen = (time.perf_counter() - t0) / len(coords)
    inner = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    t0 = time.perf_counter()
    bw.meshes(inner)
    mesh = (time.perf_counter() - t0) / len(inner)
    g = w.ground(0.0, 0.0)
    rs, _ = _view(scene, w, (-60.0, -80.0, g + 40.0), (0.0, 0.0, g))
    cold, _ = _contribute(scene, rs)
    frame = dict(w.runtime.last_frame)
    warm, _ = _contribute(scene, rs)
    times = []
    for k in range(6 if quick else 20):
        x = -60.0 + 8.0 * k
        rs, _ = _view(scene, w, (x, -80.0, g + 40.0), (x + 60.0, 0.0, g))
        dt, _ = _contribute(scene, rs)
        times.append(dt)
    s = w.runtime.stats()
    return {
        "chunk_generate_ms": round(1000 * gen, 2),
        "chunk_mesh_ms": round(1000 * mesh, 2),
        "first_frame_seconds": round(cold, 3),
        "warm_frame_seconds": round(warm, 4),
        "first_frame": frame,
        "moving_seconds_median": round(sorted(times)[len(times) // 2], 4),
        "cached_bytes": s["memory"]["cached_bytes"],
    }


def group_volumes(quick: bool) -> dict[str, Any]:
    from aether3d import Scene
    from aether3d.worlds.volumes import volumes_of

    scene = Scene("C")
    w = scene.worlds.create("C", "cavernous_highlands", seed=3)
    w.runtime.regional()
    vol = volumes_of(w.runtime)
    n = 6 if quick else 12
    t0 = time.perf_counter()
    cols = {
        (i, j): vol.column(i, j) for i in range(-n // 2, n // 2) for j in range(-n // 2, n // 2)
    }
    classify = (time.perf_counter() - t0) / len(cols)
    cubes = [(i, j, k) for (i, j), c in cols.items() for k, f in c][: 12 if quick else 40]
    t0 = time.perf_counter()
    tris = 0
    for i, j, k in cubes:
        m = vol.cube(i, j, k, 0)
        tris += 0 if m is None else m.triangle_count
    mesh = (time.perf_counter() - t0) / max(len(cubes), 1)
    return {
        "column_ms": round(1000 * classify, 2),
        "cube_ms": round(1000 * mesh, 2),
        "cubes": len(cubes),
        "triangles": tris,
    }


def group_queries(quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import Scene

    scene = Scene("Q")
    w = scene.worlds.create("W", "temperate", seed=3, size=4096.0)
    w.runtime.regional()
    rng = np.random.default_rng(1)
    n = 20_000 if quick else 100_000
    pts = rng.uniform(-2000, 2000, (n, 2))
    w.sample(pts[:100])
    t0 = time.perf_counter()
    w.sample(pts, fields=["height"])
    heights = time.perf_counter() - t0
    t0 = time.perf_counter()
    w.sample(pts[: n // 10])
    full = time.perf_counter() - t0
    t0 = time.perf_counter()
    path = w.ground_path([[-1500, -1500], [1500, 1200]], spacing=1.0)
    ground_path = time.perf_counter() - t0
    t0 = time.perf_counter()
    sites = w.find_sites(5)
    find = time.perf_counter() - t0
    return {
        "height_points_per_second": round(n / heights),
        "all_fields_points_per_second": round(n / 10 / full),
        "ground_path_seconds": round(ground_path, 3),
        "ground_path_points": len(path),
        "find_sites_seconds": round(find, 3),
        "sites": len(sites),
    }


def group_render(quick: bool) -> dict[str, Any]:
    from aether3d import Scene
    from aether3d.rendering.gpu import gpu_status

    status = gpu_status()
    if not status["available"]:
        return {"skipped": status["reason"]}
    out: dict[str, Any] = {"adapter": status["selected"]["adapter"]["name"]}
    width, height = (1280, 720) if quick else (1920, 1080)
    for preset, kw in (("alpine_valley", {"size": 8192.0}), ("blocky_adventure", {})):
        scene = Scene("R")
        w = scene.worlds.create("W", preset, seed=7, **kw)
        x0, y0 = (-1500.0, -1500.0) if preset != "blocky_adventure" else (-60.0, -80.0)
        g = w.ground(x0, y0)
        cam = scene.add_camera(
            "Cam", position=(x0, y0, g + 80.0), look_at=(0.0, 0.0, g), fov=55, far=30000
        )
        scene.add_directional_light("Sun", direction=(0.5, 0.6, -0.4), intensity=1.8)
        scene.world.atmosphere.configure(
            enabled=True,
            sun="Sun",
            fog_density=0.0002,
            fog_height=g,
            fog_falloff=0.001,
            ambient="hemisphere",
        )
        scene.world.environment.configure(type="sky")
        opts = {
            "backend": "gpu",
            "width": width,
            "height": height,
            "shadows": {"cascades": 4, "distance": 3000.0},
            "tonemap": "neutral",
        }
        t0 = time.perf_counter()
        first = scene.render(**opts)
        cold = time.perf_counter() - t0
        times = []
        n = 6 if quick else 24
        step = 40.0 if preset != "blocky_adventure" else 4.0
        for k in range(n):
            cam.transform.position = (x0 + step * k, y0 + step * k, g + 80.0)
            cam.look_at((x0 + step * k + 1000.0, y0 + step * k + 1000.0, g))
            t0 = time.perf_counter()
            r = scene.render(**opts)
            times.append(time.perf_counter() - t0)
        res = r.backend.get("resources", {}).get("mesh_cache", {})
        med = sorted(times)[len(times) // 2]
        out[preset] = {
            "first_frame_seconds": round(cold, 3),
            "frame_seconds_median": round(med, 4),
            "fps_median": round(1.0 / med, 2),
            "frames": n,
            "size": [width, height],
            "world_parts": w.runtime.last_frame.get("parts"),
            "triangles": w.runtime.last_frame.get("triangles"),
            "gpu_mesh_cache_bytes": res.get("bytes"),
            "timings": {
                k: round(v, 4) for k, v in first.stats["timings"].items() if isinstance(v, float)
            },
        }
    return out


def run_group(name: str, quick: bool) -> dict[str, Any]:
    result = globals()[f"group_{name}"](quick)
    result["peak_rss_mib"] = _peak()
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join(GROUPS))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--numpy", action="store_true", help="also run regional without native")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print("BENCH-RESULT " + json.dumps(run_group(args.worker, args.quick)))
        return 0
    from bench_engine import compare, machine

    groups = [g for g in args.only.split(",") if g]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        parser.error(f"unknown groups: {', '.join(unknown)}")
    results: dict[str, Any] = {}
    jobs = [(g, {}) for g in groups]
    if args.numpy and "regional" in groups:
        jobs.append(("regional", {"AETHER3D_NATIVE": "0"}))
    for group, env in jobs:
        key = group if not env else f"{group}_numpy"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", group]
        if args.quick:
            cmd.append("--quick")
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env={**os.environ, **env}, cwd=ROOT
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("BENCH-RESULT ")]
        if proc.returncode != 0 or not lines:
            results[key] = {"error": proc.stderr[-2000:]}
        else:
            results[key] = json.loads(lines[-1][len("BENCH-RESULT ") :])
        results[key]["wall_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"{key}: {json.dumps(results[key])[:300]}", flush=True)
    report = {"label": args.label, "quick": args.quick, "machine": machine(), "groups": results}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        compare(report, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
