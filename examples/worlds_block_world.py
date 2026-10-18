"""Showcase B (Stage 16): a Minecraft-style block world -- original pixel-art textures,
block trees, a castle on a hill, a village and a road -- generated and rendered by Aether3D.

Run from the repository root:

    python examples/worlds_block_world.py [OUTPUT_DIR] [--quick] [--no-video] [--spp N]

Writes to ``OUTPUT_DIR`` (default ``examples/output/worlds/b_block_world``): the stills
``castle.png``, ``village.png``, ``panorama.png`` (1920x1080; the castle path-traced on
the GPU, the others rasterized with cascaded shadows) and their previews; ``frames/`` (not
committed) and ``orbit_sheet.png``; ``orbit_1080p.mp4`` (libx264 slow, CRF 14),
``orbit_master.mkv`` (FFV1) and ``orbit_preview.mp4``, checked with PSNR and SSIM;
``blocks.aether`` and ``report.json``.

The block textures are original 16x16 pixel art drawn by ``aether3d/worlds/blocks.py``;
no game's assets are used.
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
    light,
    machine,
    preview,
    progress_printer,
    render_still,
    render_video_frames,
    sheet,
    write_report,
)

from aether3d import Scene  # noqa: E402

SEED = 5


def build() -> tuple[Scene, object, dict]:
    scene = Scene("Block world")
    world = scene.worlds.create("Blocks", "blocky_adventure", seed=SEED, size=2048.0)
    world.runtime.regional()
    castle = world.find_sites(1, kind="castle", radius=22.0, max_slope=25.0, spacing=150.0)[0]
    world.place_structure(
        {"id": "keep", "kind": "castle", "at": castle["at"], "size": [34.0, 30.0], "rotation": 0.0}
    )
    world.add_feature(
        {
            "type": "clear",
            "id": "keep-clearing",
            "what": ["trees"],
            "falloff": 8.0,
            "shape": {"circle": [castle["at"][0], castle["at"][1], 34.0]},
        }
    )
    village = world.build_settlements(
        1,
        houses=[7, 10],
        roads=False,
        spacing=150.0,
        region=[
            castle["at"][0] - 260,
            castle["at"][1] - 260,
            castle["at"][0] + 260,
            castle["at"][1] + 260,
        ],
    )
    vx, vy = village["settlements"][0]["at"]
    road = world.build_road([vx, vy], castle["at"], name="castle-road", cell=4.0)
    light(scene, "morning", fog=0.0006, fog_height=world.ground(*castle["at"]), falloff=0.01)
    return scene, world, {"castle": castle, "village": village, "road": road["length"]}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(OUT / "b_block_world"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--spp", type=int, default=192)
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (480, 270) if args.quick else (1920, 1080)
    report: dict = {"machine": machine(), "size": [w, h]}
    t0 = time.perf_counter()
    scene, world, info = build()
    report["build"] = {**info, "seconds": round(time.perf_counter() - t0, 2)}
    cx, cy = info["castle"]["at"]
    cz = world.ground(cx, cy)
    vx, vy = info["village"]["settlements"][0]["at"]
    vz = world.ground(vx, vy)
    shots = {
        "castle": (
            (cx - 48.0, cy - 62.0, cz + 26.0),
            (cx, cy, cz + 7.0),
            8 if args.quick else args.spp,
        ),
        "village": ((vx - 30.0, vy - 38.0, vz + 16.0), (vx + 6.0, vy + 6.0, vz + 2.0), 0),
        "panorama": ((cx - 150.0, cy - 210.0, cz + 70.0), (cx + 200.0, cy + 260.0, cz), 0),
    }
    stills = {}
    for name, (eye, target, spp) in shots.items():
        cam = scene.add_camera(f"Still {name}", position=eye, look_at=target, fov=55, far=6000)
        stills[name] = render_still(
            scene,
            out / f"{name}.png",
            camera=cam,
            width=w,
            height=h,
            path_tracing=spp,
            preset="morning",
            distance=900.0,
        )
        stills[name]["world"] = dict(world.runtime.last_frame)
        preview(out / f"{name}.png", out / f"{name}_preview.png")
        print(f"still {name}: {stills[name]['seconds']} s", flush=True)
    report["stills"] = stills
    if not args.no_video:
        seconds, fps = (3.0 if args.quick else 10.0), 30
        a = scene.animation
        a.set_timeline(fps=fps, start=0.0, end=seconds)
        orbit = scene.add_camera(
            "Orbit",
            position=(cx - 60.0, cy, cz + 30.0),
            look_at=(cx, cy, cz + 6.0),
            fov=55,
            far=6000,
        )
        a.add_path(
            "orbit",
            kind="circle",
            center=[cx, cy, cz + 30.0],
            radius=62.0,
            normal=[0, 0, 1],
            start=200.0,
        )
        a.follow_path(
            orbit, "orbit", start=0.0, end=seconds, offsets=(0.0, 0.25), easing=None, follow=False
        )
        a.add_constraint(orbit, {"type": "look_at", "target": {"point": [cx, cy, cz + 6.0]}})
        frames = out / "frames"
        video = render_video_frames(
            scene,
            frames,
            fps=fps,
            start=0.0,
            end=seconds,
            camera=orbit,
            width=w,
            height=h,
            preset="morning",
            distance=900.0,
            progress=progress_printer("orbit"),
        )
        paths = sorted(frames.glob("frame_*.png"))
        step = max(1, len(paths) // 12)
        sheet(
            paths[::step][:12],
            out / "orbit_sheet.png",
            labels=[f"{k / fps:.1f} s" for k in range(0, len(paths), step)][:12],
        )
        summary = {k: v for k, v in video.items() if k != "rows"}
        summary["frames_report"] = video["rows"][:: max(1, len(video["rows"]) // 24)]
        report["video"] = summary
        if not args.quick:
            report["encode"] = encode(frames, out, "orbit", fps)
    report["stats"] = world.runtime.stats()
    scene.save(out / "blocks.aether")
    report["project_bytes"] = (out / "blocks.aether").stat().st_size
    write_report(out / "report.json", report)
    print(f"done in {time.perf_counter() - t0:.0f} s: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
