"""UV mapping, texture painting and baking benchmarks (Stage 14).

Run from the repository root:

    python benchmarks/bench_texturing.py [--json OUT.json] [--label NAME] [--quick]
        [--only unwrap,...] [--compare SAVED.json]

Every group (and every size of a sized group) runs in its own Python process, so memory
peaks and caches of one never influence another; each reports its peak RSS (``VmHWM`` on
Linux). Timings are medians of repeated runs after a warm-up where a run is short, single
runs where it is long, with Python's cyclic garbage collector paused
(``bench_engine.median_time``). Meshes are generated (UV spheres of about 10k, 100k and
1M triangles), so every run measures the same inputs. Results depend on the machine:
compare runs on the same machine only.

Groups:

* ``unwrap:SIZE`` -- ``uv.unwrap`` at 2048 (charts, stretch, seconds), the same with the
  NumPy kernels (``AETHER3D_NATIVE=0`` in a child process; 10k and 100k), ``uv_report``,
  and ``layout_image`` at 1024.
* ``mikktspace:SIZE`` -- MikkTSpace tangents of every corner.
* ``pack`` -- repacking the unwrapped 100k sphere at 1K, 2K, 4K and 8K.
* ``paint:SIZE`` -- textures of 1K, 2K, 4K and 8K (``--quick``: 1K, 2K) on the 100k
  sphere: the first stroke (texel maps built), 100-dab strokes of radius 0.05 and 0.2
  (per dab), a fill of the whole surface, compositing the whole texture, the 8-bit level
  0 and mipmaps, and undo.
* ``bake:SIZE`` -- maps of a 10k sphere from a 1M (``--quick``: 100k) bumpy sphere at
  1K, 2K and 4K (``--quick``: 1K): normal on the CPU and the GPU; AO with 64 rays on the
  CPU and the GPU at 1K.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from bench_engine import compare, machine, median_time, ms, peak_rss_mib  # noqa: E402

SIZES = {"10k": (64, 80), "100k": (224, 224), "1m": (707, 707)}
"""Sphere ``(rings, segments)``: about 2 x rings x segments triangles."""


def sphere(size: str):
    from aether3d import create_sphere

    rings, segments = SIZES[size]
    return create_sphere(1.0, segments, rings)


def bumpy(size: str):
    """A sphere with bumps (the high-resolution source of the bake group)."""
    import numpy as np

    from aether3d import Mesh

    m = sphere(size)
    p = m.positions.astype(np.float64)
    r = 1.0 + 0.02 * np.sin(12 * np.arctan2(p[:, 1], p[:, 0])) * np.sin(10 * np.arccos(
        np.clip(p[:, 2], -1, 1)))  # fmt: skip
    return Mesh((p * r[:, None]).astype(np.float32), m.triangles)


def native_off_seconds(size: str) -> float | None:
    """Unwrap time with the NumPy kernels, measured in a child process."""
    code = (
        f"import sys, time; sys.path.insert(0, {str(HERE)!r}); "
        "from bench_texturing import sphere; from aether3d import uv; "
        f"m = sphere({size!r}); t0 = time.perf_counter(); "
        "uv.unwrap(m, resolution=2048); print(time.perf_counter() - t0)"
    )
    env = {**os.environ, "AETHER3D_NATIVE": "0"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    try:
        return round(float(proc.stdout.strip().splitlines()[-1]), 3)
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------------------
# Groups


def group_unwrap(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import _native, uv

    m = sphere(size)
    out: dict[str, Any] = {"triangles": m.triangle_count, "native": _native.status()["enabled"]}
    t0 = time.perf_counter()
    r = uv.unwrap(m, resolution=2048)
    out["unwrap_s"] = round(time.perf_counter() - t0, 3)
    out["charts"] = r.report["charts"]
    out["stretch_l2_max"] = round(r.report["stretch_l2"]["max"], 4)
    if size != "1m":
        out["unwrap_numpy_s"] = native_off_seconds(size)
    t0 = time.perf_counter()
    rep = uv.uv_report(r.mesh, resolution=2048)
    out["uv_report_s"] = round(time.perf_counter() - t0, 3)
    out["utilization"] = round(rep["utilization"], 4)
    t0 = time.perf_counter()
    uv.layout_image(r.mesh, 1024)
    out["layout_image_1024_s"] = round(time.perf_counter() - t0, 3)
    return out


def group_mikktspace(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.uv.mikktspace import mikktspace

    m = sphere(size)
    t = m.triangles
    p, n, u = m.positions[t], m.normals[t], m.uvs[t]
    repeat = 3 if size != "1m" else 1
    return {
        "triangles": m.triangle_count,
        "mikktspace_s": round(median_time(lambda: mikktspace(p, n, u), repeat, warmup=0), 3),
    }


def group_pack(quick: bool) -> dict[str, Any]:
    from aether3d import uv

    mesh = uv.unwrap(sphere("100k"), resolution=2048).mesh
    out: dict[str, Any] = {}
    for res in (1024, 2048) if quick else (1024, 2048, 4096, 8192):
        t0 = time.perf_counter()
        r = uv.pack_islands(mesh, resolution=res, padding=8)
        out[f"pack_{res}_s"] = round(time.perf_counter() - t0, 3)
        out[f"pack_{res}_utilization"] = round(
            uv.uv_report(r.mesh, resolution=res)["utilization"], 4
        )
    return out


def group_paint(size: str, quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import Scene, uv
    from aether3d.texturing import fill, stroke

    res = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192}[size]
    scene = Scene()
    ball = scene.add_object("Ball", mesh=uv.unwrap(sphere("100k"), resolution=res).mesh)
    doc = scene.add_painted_texture("Paint", res, res)
    doc.add_layer("Ink")
    out: dict[str, Any] = {"texels": res * res}
    t0 = time.perf_counter()
    stroke(scene, ball, doc, {"radius": 0.05, "color": "#ff0000"}, [[1, 0, 0]])
    out["first_dab_ms"] = ms(time.perf_counter() - t0)

    for radius in (0.05, 0.2):
        # 100 dabs: a chord of 99 x spacing (0.15) x radius.
        length = 99 * 0.15 * radius
        pts = np.array([[1.0, 0.0, 0.0], [np.cos(length), np.sin(length), 0.0]])
        t0 = time.perf_counter()
        rep = stroke(scene, ball, doc, {"radius": radius, "color": "#00ff00"}, pts)
        seconds = time.perf_counter() - t0
        out[f"stroke_r{radius}_dabs"] = rep.dabs
        out[f"stroke_r{radius}_ms"] = ms(seconds)
        out[f"stroke_r{radius}_per_dab_ms"] = ms(seconds / max(rep.dabs, 1))
        out[f"stroke_r{radius}_tiles"] = rep.tiles
    t0 = time.perf_counter()
    fill(scene, ball, doc, "#0000ff", opacity=0.5)
    out["fill_s"] = round(time.perf_counter() - t0, 3)
    out["composite_s"] = round(median_time(doc.read, 1, warmup=0), 3)
    t0 = time.perf_counter()
    doc.mip_chain()
    out["level0_and_mips_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    scene.history.undo()
    out["undo_fill_ms"] = ms(time.perf_counter() - t0)
    return out


def group_bake(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene, uv
    from aether3d.texturing import bake

    res = {"1k": 1024, "2k": 2048, "4k": 4096}[size]
    scene = Scene()
    high = scene.add_object("High", mesh=bumpy("100k" if quick else "1m"))
    low = scene.add_object("Low", mesh=uv.unwrap(sphere("10k"), resolution=res).mesh)
    out: dict[str, Any] = {"source_triangles": high.mesh.triangle_count}
    for backend in ("cpu", "gpu"):
        try:
            t0 = time.perf_counter()
            r = bake(scene, low, [high], ["normal"], resolution=res, backend=backend)
            out[f"normal_{backend}_s"] = round(time.perf_counter() - t0, 3)
            out[f"normal_{backend}_stages"] = {
                k: round(v, 3) for k, v in r.report["seconds"].items()
            }
            out["miss_fraction"] = r.report["miss_fraction"]
        except Exception as exc:  # noqa: BLE001 - a missing GPU is a result, not a crash
            out[f"normal_{backend}"] = f"unavailable: {type(exc).__name__}"
        if res == 1024:
            try:
                t0 = time.perf_counter()
                r = bake(scene, low, [high], ["ao"], resolution=res, ao_rays=64, backend=backend)
                seconds = time.perf_counter() - t0
                out[f"ao64_{backend}_s"] = round(seconds, 3)
                out[f"ao64_{backend}_mrays_per_s"] = round(r.report["rays"] / seconds / 1e6, 2)
            except Exception as exc:  # noqa: BLE001
                out[f"ao64_{backend}"] = f"unavailable: {type(exc).__name__}"
    return out


GROUPS = {"pack": group_pack}
SIZED = {
    "unwrap": (group_unwrap, ("10k", "100k", "1m")),
    "mikktspace": (group_mikktspace, ("10k", "100k", "1m")),
    "paint": (group_paint, ("1k", "2k", "4k", "8k")),
    "bake": (group_bake, ("1k", "2k", "4k")),
}
QUICK_SIZES = {"unwrap": ("10k", "100k"), "mikktspace": ("10k", "100k"),
               "paint": ("1k", "2k"), "bake": ("1k",)}  # fmt: skip


# ---------------------------------------------------------------------------------------
# Driver


def run_group(name: str, quick: bool) -> dict[str, Any]:
    group, _, size = name.partition(":")
    result = SIZED[group][0](size, quick) if size else GROUPS[group](quick)
    result["peak_rss_mib"] = peak_rss_mib()
    return result


def jobs(only: list[str], quick: bool) -> list[str]:
    out = []
    for name in only:
        group, _, size = name.partition(":")
        if group in GROUPS and not size:
            out.append(group)
        elif group in SIZED:
            sizes = (size,) if size else (QUICK_SIZES[group] if quick else SIZED[group][1])
            out += [f"{group}:{s}" for s in sizes]
        else:
            raise SystemExit(f"unknown group {name!r}")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join([*SIZED, *GROUPS]))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print("BENCH-RESULT " + json.dumps(run_group(args.worker, args.quick)))
        return 0
    groups = jobs([g for g in args.only.split(",") if g], args.quick)
    data: dict[str, Any] = {"label": args.label, "machine": machine(), "groups": {}}
    for group in groups:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", group]
        if args.quick:
            command.append("--quick")
        t0 = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BENCH-RESULT ")]
        if proc.returncode != 0 or not lines:
            data["groups"][group] = {"error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"{group}: FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        data["groups"][group] = json.loads(lines[-1][len("BENCH-RESULT ") :])
        print(f"{group} ({time.perf_counter() - t0:.0f} s): {json.dumps(data['groups'][group])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        compare(data, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
