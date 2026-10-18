"""Showcase A (Stage 16): a cinematic alpine valley -- eroded ranges, a river, a lake,
forests, a village, golden-hour light -- generated and rendered by Aether3D.

Run from the repository root:

    python examples/worlds_alpine_valley.py [OUTPUT_DIR] [--quick] [--no-video] [--spp N]

Writes to ``OUTPUT_DIR`` (default ``examples/output/worlds/a_alpine_valley``):

* ``valley.png``, ``village.png``, ``lake.png`` -- stills (GPU path tracer, 1920x1080) and
  their previews ``*_preview.png`` (960x540);
* ``frames/`` -- every frame of the flight (GPU rasterizer with cascaded shadows; not
  committed), ``flight_sheet.png`` -- a contact sheet;
* ``flight_1080p.mp4`` (libx264 slow, CRF 14), ``flight_master.mkv`` (FFV1, lossless) and
  ``flight_preview.mp4`` (960x540) -- with PSNR and SSIM against the frames;
* ``valley.aether`` (the scene: a few KB) and ``report.json`` (timings, worlds' statistics).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from worlds_common import (  # noqa: E402
    OUT,
    encode,
    flight,
    frame_size,
    light,
    machine,
    preview,
    progress_printer,
    render_still,
    render_video_frames,
    set_backend,
    sheet,
    write_report,
)

from aether3d import Scene  # noqa: E402

SEED = 7
LAKE = (-1300.0, -700.0, 170.0)
VILLAGE_REGION = [-2700.0, -3500.0, -1100.0, -1500.0]


def build() -> tuple[Scene, object, dict]:
    scene = Scene("Alpine valley")
    valley = scene.worlds.create("Valley", "alpine_valley", seed=SEED)
    t0 = time.perf_counter()
    valley.runtime.regional()
    info = {
        "regional_seconds": round(time.perf_counter() - t0, 2),
        "regional": valley.runtime.regional().summary(),
    }
    x, y, r = LAKE
    model = valley.runtime.model()
    import numpy as np

    level = float(
        np.median(model.natural(np.array([x, x + r, x - r, x]), np.array([y, y, y, y + r])))
    )
    valley.add_feature(
        {
            "type": "lake",
            "id": "tarn",
            "shape": {"circle": [x, y, r]},
            "level": round(level + 1.5, 2),
            "depth": 7.0,
            "falloff": 40.0,
        }
    )
    valley.add_feature(
        {
            "type": "clear",
            "id": "shore",
            "shape": {"circle": [x, y, r + 90.0]},
            "what": ["trees"],
            "falloff": 50.0,
        }
    )
    info["settlements"] = valley.build_settlements(
        1, houses=[9, 13], roads=False, region=VILLAGE_REGION, max_slope=14.0, spacing=300.0
    )
    light(scene, "golden", fog=0.00011, fog_height=level, falloff=0.0009)
    return scene, valley, info


def shots(valley, info) -> dict:
    """Camera placements computed from the village and lake positions."""
    vx, vy = info["settlements"]["settlements"][0]["at"]
    lx, ly, lr = LAKE

    def at(x, y, up):
        return (x, y, valley.height(x, y) + up)

    massif = (1100.0, -1100.0, valley.height(1100.0, -1100.0))
    return {
        "valley": (at(-1900.0, -3200.0, 260.0), massif),
        "village": (
            at(vx - 62.0, vy - 88.0, 34.0),
            (vx + 8.0, vy + 12.0, valley.height(vx, vy) + 3.0),
        ),
        "lake": (
            at(lx - 40.0, ly - lr - 60.0, 30.0),
            (lx + 700.0, ly + 1700.0, valley.height(lx, ly) + 380.0),
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(OUT / "a_alpine_valley"))
    parser.add_argument("--quick", action="store_true", help="small, few samples, short")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                        help="cpu: the CPU rasterizer for every image (tests)")  # fmt: skip
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--spp", type=int, default=256)
    args = parser.parse_args(argv)
    set_backend(args.backend)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = frame_size(args.quick)
    spp = 8 if args.quick else args.spp
    report: dict = {"machine": machine(), "backend": args.backend, "size": [w, h]}
    t0 = time.perf_counter()
    scene, valley, info = build()
    report["build"] = {**info, "seconds": round(time.perf_counter() - t0, 2)}
    print("built:", report["build"]["regional"], flush=True)
    stills = {}
    for name, (eye, target) in shots(valley, info).items():
        cam = scene.add_camera(f"Still {name}", position=eye, look_at=target, fov=52, far=40000)
        stills[name] = render_still(
            scene,
            out / f"{name}.png",
            camera=cam,
            width=w,
            height=h,
            path_tracing=spp,
            preset="golden",
        )
        stills[name]["world"] = dict(valley.runtime.last_frame)
        preview(out / f"{name}.png", out / f"{name}_preview.png")
        print(f"still {name}: {stills[name]['seconds']} s", flush=True)
    report["stills"] = stills
    if not args.no_video:
        seconds = 4.0 if args.quick else 12.0
        fps = 30
        scene.animation.set_timeline(fps=fps, start=0.0, end=seconds)
        vx, vy = info["settlements"]["settlements"][0]["at"]
        lx, ly, _ = LAKE
        eyes = [
            (vx - 700.0, vy - 900.0, 0),
            (vx - 200.0, vy - 350.0, 0),
            (vx + 250.0, vy + 300.0, 0),
            (lx - 250.0, ly - 700.0, 0),
            (lx - 50.0, ly - 150.0, 0),
        ]
        eyes = [
            (x, y, valley.height(x, y) + up)
            for (x, y, _), up in zip(eyes, (240, 170, 140, 160, 200), strict=True)
        ]
        aims = [
            (vx, vy, valley.height(vx, vy)),
            (vx + 250.0, vy + 300.0, valley.height(vx, vy)),
            (lx, ly, valley.height(lx, ly)),
            (lx + 400.0, ly + 900.0, valley.height(lx, ly) + 200.0),
            (1100.0, -1100.0, valley.height(1100.0, -1100.0)),
        ]
        cam = flight(scene, "Flight", eyes, aims, start=0.0, end=seconds, fov=52)
        frames = out / "frames"
        video = render_video_frames(
            scene,
            frames,
            fps=fps,
            start=0.0,
            end=seconds,
            camera=cam,
            width=w,
            height=h,
            preset="golden",
            progress=progress_printer("flight"),
        )
        paths = sorted(frames.glob("frame_*.png"))
        pick = paths[:: max(1, len(paths) // 12)][:12]
        sheet(
            pick,
            out / "flight_sheet.png",
            labels=[f"{i / fps:.1f} s" for i in range(0, len(paths), max(1, len(paths) // 12))][
                :12
            ],
        )
        video_summary = {k: v for k, v in video.items() if k != "rows"}
        video_summary["frames_report"] = video["rows"][:: max(1, len(video["rows"]) // 24)]
        report["video"] = video_summary
        if not args.quick:
            report["encode"] = encode(frames, out, "flight", fps)
    report["stats"] = valley.runtime.stats()
    scene.save(out / "valley.aether")
    report["project_bytes"] = (out / "valley.aether").stat().st_size
    write_report(out / "report.json", report)
    print(f"{out}: {time.perf_counter() - t0:.0f} s, done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
