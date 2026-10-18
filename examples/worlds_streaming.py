"""Showcase C (Stage 16): streaming an unbounded world -- a camera flies kilometres over
``infinite_highlands`` a thousand kilometres from the origin while every frame generates
only what its view needs; memory, tiles and frame times are measured per frame.

Run from the repository root:

    python examples/worlds_streaming.py [OUTPUT_DIR] [--quick] [--no-video]

Writes to ``OUTPUT_DIR`` (default ``examples/output/worlds/c_streaming``): ``frames/`` (not
committed), ``flight_sheet.png``, ``start.png``, ``middle.png``, ``end.png`` (and
previews), ``memory.png`` (cached bytes, resident memory and frame seconds per frame,
drawn with NumPy), ``flight_1080p.mp4`` (libx264 slow, CRF 14), ``flight_master.mkv``,
``flight_preview.mp4`` and ``report.json`` (every frame's measurements).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from worlds_common import (  # noqa: E402
    OUT,
    encode,
    exposure,
    light,
    machine,
    preview,
    sheet,
    software,
    write_report,
)

from aether3d import Scene  # noqa: E402
from aether3d.rendering.png import encode_png  # noqa: E402

ORIGIN = (1_000_000.0, 2_000_000.0)  # a thousand kilometres east, two thousand north
DISTANCE = 5000.0  # metres flown


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(OUT / "c_streaming"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (480, 270) if args.quick else (1920, 1080)
    seconds, fps = (4.0, 15) if args.quick else (20.0, 30)
    scene = Scene("Streaming")
    world = scene.worlds.create(
        "Highlands", "infinite_highlands", seed=21, overrides={"streaming": {"memory_mb": 768.0}}
    )
    x0, y0 = ORIGIN
    g0 = world.height(x0, y0)
    light(scene, "morning", fog=0.00009, fog_height=g0, falloff=0.0008)
    cam = scene.add_camera(
        "Flight",
        position=(x0, y0, g0 + 140.0),
        look_at=(x0 + 1000.0, y0 + 300.0, g0),
        fov=60,
        far=40000,
    )
    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    count = int(seconds * fps)
    rows = []
    t_all = time.perf_counter()
    rt = world.runtime
    previous = dict(rt.counters)
    for k in range(count):
        t = k / max(count - 1, 1)
        x = x0 + DISTANCE * t
        y = y0 + 0.3 * DISTANCE * t + 180.0 * np.sin(t * 3.0)
        ground = max(world.height(x, y), world.height(x + 200.0, y + 60.0))
        cam.transform.position = (x, y, ground + 140.0)
        cam.look_at((x + 1000.0, y + 300.0, ground - 20.0))
        t0 = time.perf_counter()
        r = scene.render(
            backend="gpu",
            width=w,
            height=h,
            tonemap="neutral",
            exposure=exposure("morning"),
            shadows={"cascades": 4, "distance": 4000.0, "map_size": 4096},
        )
        dt = time.perf_counter() - t0
        r.save(frames_dir / f"frame_{k:05d}.png")
        stats = rt.stats()
        now = dict(rt.counters)
        cache = (r.backend.get("resources") or {}).get("mesh_cache") or {}
        rows.append(
            {
                "frame": k,
                "x": round(x, 2),
                "y": round(y, 2),
                "seconds": round(dt, 3),
                "tiles_new": now["tiles_generated"] - previous["tiles_generated"],
                "tiles_cached": stats["tiles_cached"],
                "tile_bytes": stats["tile_bytes"],
                "cached_total": stats["memory"]["cached_total"],
                "rss_bytes": stats["memory"]["process_rss_bytes"],
                "gpu_mesh_cache_bytes": cache.get("bytes"),
                "parts": rt.last_frame.get("parts"),
                "triangles": rt.last_frame.get("triangles"),
            }
        )
        previous = now
        if k % 30 == 0 or k == count - 1:
            print(
                f"  frame {k + 1}/{count}: {dt:.2f} s, {rows[-1]['tiles_new']} new tiles, "
                f"{rows[-1]['cached_total'] / 2**20:.0f} MiB cached",
                flush=True,
            )
    total = time.perf_counter() - t_all
    paths = sorted(frames_dir.glob("frame_*.png"))
    for name, k in (("start", 0), ("middle", len(paths) // 2), ("end", len(paths) - 1)):
        shutil.copy(paths[k], out / f"{name}.png")
        preview(out / f"{name}.png", out / f"{name}_preview.png")
    step = max(1, len(paths) // 12)
    sheet(
        paths[::step][:12],
        out / "flight_sheet.png",
        labels=[f"{k / fps:.1f} s" for k in range(0, len(paths), step)][:12],
    )
    (out / "memory.png").write_bytes(encode_png(chart(rows), text=software()))
    secs = sorted(r["seconds"] for r in rows)
    report = {
        "machine": machine(),
        "world": {
            "preset": "infinite_highlands",
            "origin": ORIGIN,
            "flown_m": DISTANCE,
            "memory_mb": 768.0,
        },
        "frames": len(rows),
        "fps": fps,
        "size": [w, h],
        "seconds": round(total, 1),
        "seconds_per_frame_median": secs[len(secs) // 2],
        "seconds_per_frame_p95": secs[int(0.95 * (len(secs) - 1))],
        "tiles_generated": rt.counters["tiles_generated"],
        "peak_cached_mib": round(max(r["cached_total"] for r in rows) / 2**20, 1),
        "peak_rss_mib": round(max(r["rss_bytes"] or 0 for r in rows) / 2**20, 1),
        "peak_gpu_mesh_cache_mib": round(
            max((r["gpu_mesh_cache_bytes"] or 0) for r in rows) / 2**20, 1
        ),
        "stats": rt.stats(),
        "rows": rows,
    }
    if not args.no_video and not args.quick:
        report["encode"] = encode(frames_dir, out, "flight", fps)
    write_report(out / "report.json", report)
    print(f"done in {total:.0f} s: {out}")
    return 0


def chart(rows: list[dict]) -> np.ndarray:
    """Three stacked line charts (cached MiB, resident MiB, frame seconds), white on dark."""
    w, h, pad = 960, 540, 30
    img = np.full((h, w, 3), 22, np.uint8)
    series = [
        ([r["cached_total"] / 2**20 for r in rows], (120, 200, 255)),
        ([(r["rss_bytes"] or 0) / 2**20 for r in rows], (255, 190, 90)),
        ([r["seconds"] for r in rows], (140, 230, 140)),
    ]
    band = (h - 4 * pad) // 3
    for k, (values, color) in enumerate(series):
        top = pad + k * (band + pad)
        v = np.asarray(values, dtype=np.float64)
        lo, hi = 0.0, max(float(v.max()), 1e-9)
        img[top + band, pad : w - pad] = 70
        xs = np.linspace(pad, w - pad - 1, len(v))
        ys = top + band - (v - lo) / (hi - lo) * band
        for i in range(len(v) - 1):
            n = int(max(abs(xs[i + 1] - xs[i]), abs(ys[i + 1] - ys[i]))) + 1
            for s in np.linspace(0.0, 1.0, n):
                x = int(round(xs[i] + s * (xs[i + 1] - xs[i])))
                y = int(round(ys[i] + s * (ys[i + 1] - ys[i])))
                img[max(y - 1, 0) : y + 1, x] = color
    return img


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
