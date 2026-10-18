"""Showcase F (Stage 16 + Stage 17): Nova, Stage 17's blocky character (unchanged), runs
through the block world along a village road toward the castle, followed by a tracking
camera -- the world streams around the moving camera; every frame is rendered by Aether3D.

Run from the repository root:

    python examples/worlds_character.py [OUTPUT_DIR] [--quick] [--no-video]

Writes to ``OUTPUT_DIR`` (default ``examples/output/worlds/f_character``): ``frames/`` (not
committed), ``run_sheet.png``, ``hero.png`` (and preview), ``run_1080p.mp4`` (libx264
slow, CRF 14), ``run_master.mkv``, ``run_preview.mp4``, ``run.aether`` and ``report.json``
(the run clip, measured foot contacts on the path, frame timings).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from animation_common import blocky_character, pixel_skin  # noqa: E402
from worlds_common import (  # noqa: E402
    OUT,
    encode,
    light,
    machine,
    preview,
    progress_printer,
    render_video_frames,
    sheet,
    write_report,
)

from aether3d import Scene  # noqa: E402
from aether3d.rendering.png import encode_png  # noqa: E402

FEET = ["Leg.L", "Leg.R"]


def run_line(world, village, castle, length: float) -> tuple[float, float, float]:
    """The start and heading of a straight run toward the castle whose corridor (the
    camera's too) misses every structure: from the village outward, turning up to 40
    degrees off the castle's bearing if the houses are in the way."""
    from aether3d.worlds.structures import footprint

    polys = [footprint(it, grow=1.0) for it in world.structures() if it["kind"] != "bridge"]
    bearing = math.atan2(castle[1] - village[1], castle[0] - village[0])
    for turn in (0, 10, -10, 20, -20, 30, -30, 40, -40):
        ang = bearing + math.radians(turn)
        dx, dy = math.cos(ang), math.sin(ang)
        for start in range(0, 120, 2):
            x0, y0 = village[0] + start * dx, village[1] + start * dy
            if all(
                not _crosses(
                    p, (x0 - 10 * dx, y0 - 10 * dy), (x0 + length * dx, y0 + length * dy), 4.5
                )
                for p in polys
            ):
                return x0, y0, ang
    raise RuntimeError("no clear run toward the castle")


def _crosses(poly, a, b, half: float) -> bool:
    """Whether the corridor ``a``-``b`` (half-width ``half``) meets the convex ``poly``
    (separating axes of the corridor rectangle and the polygon)."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    rect = [
        (a[0] - uy * half, a[1] + ux * half),
        (b[0] - uy * half, b[1] + ux * half),
        (b[0] + uy * half, b[1] - ux * half),
        (a[0] + uy * half, a[1] - ux * half),
    ]
    for shape in (rect, poly):
        for i in range(len(shape)):
            p, q = shape[i], shape[(i + 1) % len(shape)]
            nx, ny = q[1] - p[1], p[0] - q[0]
            r = [nx * v[0] + ny * v[1] for v in rect]
            s = [nx * v[0] + ny * v[1] for v in poly]
            if max(r) < min(s) or max(s) < min(r):
                return False
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(OUT / "f_character"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (480, 270) if args.quick else (1920, 1080)
    fps = 30
    seconds = 3.0 if args.quick else 10.0
    t0 = time.perf_counter()
    scene = Scene("Run")
    world = scene.worlds.create("Blocks", "blocky_adventure", seed=5, size=2048.0)
    world.runtime.regional()
    castle = world.find_sites(1, kind="castle", radius=22.0, max_slope=25.0, spacing=150.0)[0]
    world.place_structure(
        {"id": "keep", "kind": "castle", "at": castle["at"], "size": [34.0, 30.0]}
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
    cx, cy = castle["at"]
    a = scene.animation
    speed = 3.15  # the generated run's speed (m/s, reported below)
    length = speed * seconds + 8.0
    sx, sy, ang = run_line(world, (vx, vy), (cx, cy), length)
    dx, dy = math.cos(ang), math.sin(ang)
    level = round(world.ground(sx, sy))
    ex, ey = sx + length * dx, sy + length * dy
    back = (sx - 10 * dx, sy - 10 * dy)
    world.add_feature(
        {
            "type": "flatten",
            "id": "run-bed",
            "height": float(level),
            "falloff": 4.0,
            "shape": {"line": [list(back), [ex, ey]], "width": 6.0},
        }
    )
    world.add_feature(
        {
            "type": "road",
            "id": "run-road",
            "falloff": 1.0,
            "material": "dirt",
            "shape": {"line": [list(back), [ex, ey]], "width": 3.0},
        }
    )
    world.add_feature(
        {
            "type": "clear",
            "id": "run-clear",
            "what": ["trees", "plants"],
            "falloff": 3.0,
            "shape": {"line": [list(back), [ex, ey]], "width": 12.0},
        }
    )
    ground = world.ground(sx, sy)
    light(scene, "morning", fog=0.0005, fog_height=ground, falloff=0.01)
    # Nova (Stage 17, unchanged) and its parts under an empty that places and turns them:
    # attached parts follow their joints relative to where they were authored, so they
    # move with the armature.
    yaw = math.degrees(ang) + 90.0  # its forward is -Y
    runner = scene.add_object("Runner", position=(sx, sy, ground))
    runner.rotation_euler = (0.0, 0.0, yaw)
    nova = blocky_character(scene, "Nova", encode_png(pixel_skin()))
    for obj in [nova, *(o for o in scene.objects if o.name.startswith("Nova "))]:
        obj.set_parent(runner)
    a.set_timeline(fps=fps, start=0.0, end=seconds)
    clip = a.generate(nova, "run", name="Run", forward="-y", bind=False)
    a.set_action(nova, "Run")
    assert abs(clip["speed"] - speed) < 0.01, clip["speed"]
    # The tracking camera: behind and to the side, slightly above, with a little lag.
    cam = scene.add_camera(
        "Follow",
        position=(sx - 6 * dx + 2.0 * dy, sy - 6 * dy - 2.0 * dx, ground + 2.6),
        look_at=(sx, sy, ground + 1.0),
        fov=48,
        far=6000,
    )
    a.follow_camera(
        cam,
        nova,
        offset=(-6.0 * dx + 2.2 * dy, -6.0 * dy - 2.2 * dx, 2.6),
        lag=0.3,
        aim=(0.0, 0.0, 1.0),
        start=0.0,
        end=seconds,
    )
    contacts = a.foot_contacts(nova, FEET, start=0.0, end=seconds, rate=120, ground=ground)
    report: dict = {
        "machine": machine(),
        "size": [w, h],
        "clip": {
            k: clip[k] for k in ("keys", "duration", "travel", "speed", "period") if k in clip
        },
        "road": {
            "start": [round(sx, 2), round(sy, 2)],
            "end": [round(ex, 2), round(ey, 2)],
            "level": level,
            "length_m": round(length, 2),
        },
        "contacts": {
            f: {
                "sliding_m": round(d["sliding_max"], 4),
                "penetration_m": round(d["penetration"], 4),
                "contacts": len(d["contacts"]),
            }
            for f, d in contacts["feet"].items()
        },
    }
    hero = scene.render(
        camera=cam,
        time=seconds * 0.55,
        width=w,
        height=h,
        backend="gpu",
        shadows={"cascades": 4, "distance": 600.0, "map_size": 4096},
        tonemap="neutral",
    )
    hero.save(out / "hero.png")
    preview(out / "hero.png", out / "hero_preview.png")
    if not args.no_video:
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
            preset="morning",
            distance=600.0,
            progress=progress_printer("run"),
        )
        paths = sorted(frames.glob("frame_*.png"))
        step = max(1, len(paths) // 12)
        sheet(
            paths[::step][:12],
            out / "run_sheet.png",
            labels=[f"{k / fps:.1f} s" for k in range(0, len(paths), step)][:12],
        )
        summary = {k: v for k, v in video.items() if k != "rows"}
        summary["frames_report"] = video["rows"][:: max(1, len(video["rows"]) // 24)]
        report["video"] = summary
        if not args.quick:
            report["encode"] = encode(frames, out, "run", fps)
    report["stats"] = world.runtime.stats()
    report["seconds"] = round(time.perf_counter() - t0, 1)
    scene.save(out / "run.aether")
    write_report(out / "report.json", report)
    print(f"done in {report['seconds']:.0f} s: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
