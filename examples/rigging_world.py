"""Showcase F (Stage 18 + Stage 16): characters of both workflows in a procedural world --
a smoothly skinned, auto-rigged walker and Kit, the rigid-part Minecraft-inspired
character (no bones), moving along a road through rolling plains, filmed in three shots.

The world is Stage 16's ``rolling_plains`` preset (generated, streamed, rendered by
Aether3D). A straight stretch is chosen where the ground stays level, flattened (a
``flatten`` feature) and paved (a ``road``). The walker plays an in-place walk while an
empty carries it along the road (``follow_path`` at exactly the walk's speed, so the feet
do not slide); Kit runs an in-place rigid run on a parallel lane at its own speed and
overtakes. Ground contact is measured against the road: the walker's feet by
``foot_contacts``, Kit's leg blocks by their soles' heights and slip while planted.

Shots: a rising crane over the road, a tracking shot beside the pair, a low shot ahead
as they pass. Lighting: the low golden sun (key, with sky, glows and height fog), a cool
fill from the opposite side and a rim from behind the characters.

Run from the repository root:

    python examples/rigging_world.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--no-video] [--fps 60]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/f_world``): ``world_sheet.png``,
``hero_*.png`` (and previews), ``frames/`` (not committed), ``world_1080p.mp4``/
``world_master.mkv``/``world_preview.mp4``, ``world.aether`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rigging_common as C  # noqa: E402
import worlds_common as W  # noqa: E402
from animation_common import pixel_skin, skin_material  # noqa: E402
from rigging_minecraft_rigid import KIT, kit  # noqa: E402

from aether3d import Scene  # noqa: E402
from aether3d.rendering.png import encode_png  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402

SECONDS = 9.0
SHOTS = ((0.0, 3.0, "crane"), (3.0, 6.0, "tracking"), (6.0, 9.0, "ahead"))


def level_line(world, length: float, seed_xy=(0.0, 0.0)) -> tuple[float, float, float]:
    """A start and heading whose ground stays within 2 m along ``length`` metres."""
    best = None
    for ring in range(0, 400, 40):
        for k in range(12):
            ang = 2 * math.pi * k / 12
            x0 = seed_xy[0] + ring * math.cos(1.7 * k)
            y0 = seed_xy[1] + ring * math.sin(1.7 * k)
            t = np.arange(-12.0, length + 12.0, 2.0)
            g = np.asarray(world.ground(x0 + t * math.cos(ang), y0 + t * math.sin(ang)), float)
            spread = float(g.max() - g.min())
            if best is None or spread < best[0]:
                best = (spread, x0, y0, ang)
            if spread < 2.0:
                return x0, y0, ang
    return best[1], best[2], best[3]


def rigid_contacts(scene: Scene, legs, sole_local, times, ground: float) -> dict:
    """Kit's soles: how high above the road, and how far they slip while planted (a sole
    within 2 cm of the road and moving under 0.4 m/s)."""
    out = {}
    pts = {leg.name: [] for leg in legs}
    a = scene.animation
    for t in times:
        w = world_matrices(scene, a.overlay(float(t)))
        for leg in legs:
            pts[leg.name].append((w[leg.id] @ np.append(sole_local, 1.0))[:3])
    for name, p in pts.items():
        p = np.asarray(p)
        h = p[:, 2] - ground
        v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / np.diff(times)
        planted = (h[1:] < 0.02) & (v < 0.4)
        slip = float((v[planted] * np.diff(times)[planted]).max()) if planted.any() else 0.0
        out[name] = {
            "lowest_m": round(float(h.min()), 4),
            "planted_frames": int(planted.sum()),
            "max_slip_per_frame_m": round(slip, 5),
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "f_world"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = C.set_backend(args.backend)
    W.set_backend(backend)
    size = (320, 180) if args.quick else (1920, 1080)
    fps = 8 if args.quick else args.fps
    seconds = 3.0 if args.quick else SECONDS
    t_all = time.perf_counter()
    scene = Scene("Showcase F")
    world = scene.worlds.create("Plains", "rolling_plains", seed=7, size=4096.0)
    a = scene.animation
    print("1. characters")
    body = scene.add_object("Walker", mesh=C.character_mesh(args.quick))
    edges = C.tailor(scene, body)
    a.autorig(body, name="Walker Rig")
    walker = scene.get_by_name("Walker Rig")
    C.dress(scene, body, walker, edges)
    walk = a.generate(walker, "walk", name="Walk", forward="-y", travel=False)
    skin_img = pixel_skin(seed=11, **KIT)
    parts = kit(scene, skin_material(scene, "Kit", encode_png(skin_img)))
    run = a.generate_rigid(parts["root"], "run", name="Run", travel=False)
    report: dict = {
        "backend": backend,
        "machine": C.machine(),
        "walker": {k: walk[k] for k in ("speed", "period", "keys")},
        "kit": {
            "speed": run["speed"],
            "keys": run["keys"],
            "verification": run["verification"],
            "bones": sum(1 for o in parts.values() if a.rig(o) is not None),
        },
    }
    print("2. the road")
    v_walk, v_run = walk["speed"], run["speed"]
    length = v_run * seconds + 14.0
    x0, y0, ang = level_line(world, length)
    dx, dy = math.cos(ang), math.sin(ang)
    level = float(
        round(
            float(
                np.median(
                    world.ground(
                        x0 + np.arange(0, length, 4.0) * dx, y0 + np.arange(0, length, 4.0) * dy
                    )
                )
            ),
            2,
        )
    )
    nx, ny = -dy, dx  # (to the road's left)
    back = (x0 - 12 * dx, y0 - 12 * dy)
    end = (x0 + length * dx, y0 + length * dy)
    world.add_feature(
        {
            "type": "flatten",
            "id": "road-bed",
            "height": level,
            "falloff": 6.0,
            "shape": {"line": [list(back), list(end)], "width": 9.0},
        }
    )
    world.add_feature(
        {
            "type": "road",
            "id": "road",
            "falloff": 1.0,
            "material": "dirt",
            "shape": {"line": [list(back), list(end)], "width": 4.0},
        }
    )
    world.add_feature(
        {
            "type": "clear",
            "id": "road-clear",
            "what": ["trees", "plants"],
            "falloff": 3.0,
            "shape": {"line": [list(back), list(end)], "width": 16.0},
        }
    )
    # (and a band beside the road's start for the crane: it rises 9-14 m to the side)
    crane_a = (x0 - 8 * dx - 11 * nx, y0 - 8 * dy - 11 * ny)
    crane_b = (x0 + 16 * dx - 11 * nx, y0 + 16 * dy - 11 * ny)
    world.add_feature(
        {
            "type": "clear",
            "id": "crane-clear",
            "what": ["trees", "plants"],
            "falloff": 3.0,
            "shape": {"line": [list(crane_a), list(crane_b)], "width": 12.0},
        }
    )
    report["road"] = {
        "start": [round(x0, 2), round(y0, 2)],
        "heading_deg": round(math.degrees(ang), 1),
        "length_m": round(length, 1),
        "level_m": level,
    }
    # Movers: empties on paths at the motions' own speeds (in-place cycles: no sliding).
    lanes = {"walker": (walker, -1.0, v_walk, 6.0), "kit": (parts["root"], 1.0, v_run, 0.0)}
    for name, (char, side, speed, ahead) in lanes.items():
        mover = scene.add_object(f"{name} mover")
        char.set_parent(mover)
        char.position = (0.0, 0.0, 0.0)
        char.rotation_euler = (0.0, 0.0, 0.0)
        s0 = ahead
        s1 = ahead + speed * seconds
        pts = [[x0 + s * dx + side * nx, y0 + s * dy + side * ny, level] for s in (s0, s1)]
        a.add_path(f"{name} lane", pts, kind="linear")
        a.follow_path(mover, f"{name} lane", start=0.0, end=seconds, easing=None, axis="-y")
        report[f"{name}_lane"] = {"speed_m_s": round(speed, 4), "from_m": s0, "to_m": round(s1, 2)}
    body.set_parent(scene.get_by_name("walker mover"))  # (the skinned mesh with its rig)
    body.position = (0.0, 0.0, 0.0)
    a.set_timeline(start=0.0, end=seconds)
    print("3. contacts")
    contacts = a.foot_contacts(
        walker, ["Foot.L", "Foot.R"], start=0.0, end=seconds, rate=120, ground=level
    )
    report["walker_contacts"] = {
        f: {
            "sliding_m": round(d["sliding_max"], 5),
            "penetration_m": round(d["penetration"], 5),
            "contacts": len(d["contacts"]),
        }
        for f, d in contacts["feet"].items()
    }
    times = np.arange(0.0, seconds, 1.0 / 120.0)
    from animation_common import PX

    report["kit_contacts"] = rigid_contacts(
        scene, [parts["leg.L"], parts["leg.R"]], np.array([0.0, 0.0, -12 * PX]), times, level
    )
    print(f"   walker {report['walker_contacts']}")
    print(f"   kit {report['kit_contacts']}")
    print("4. lights and cameras")
    W.light(scene, "golden", fog=0.0003, fog_height=level, falloff=0.01)
    fill = scene.add_directional_light(
        "Fill", direction=(-dx * 0.3 - nx * 0.8, -dy * 0.3 - ny * 0.8, -0.4), intensity=0.35
    )
    fill.light.configure(color=(0.7, 0.8, 1.0), angle=8.0)
    rim = scene.add_directional_light(
        "Rim", direction=(-dx * 0.75, -dy * 0.75, -0.55), intensity=1.1
    )
    rim.light.configure(color=(1.0, 0.93, 0.85), angle=2.0)
    report["lighting"] = {"key": "golden sun (worlds_common.light)", "fill": 0.35, "rim": 1.1}

    def at(s: float, side: float, h: float) -> list[float]:
        return [x0 + s * dx + side * nx, y0 + s * dy + side * ny, level + h]

    crane = W.flight(
        scene,
        "Crane",
        eyes=[at(-6, -9, 1.5), at(2, -12, 6.0), at(10, -14, 11.0)],
        aims=[at(4, 0, 1.0), at(8, 0, 1.0), at(13, 0, 1.0)],
        start=0.0,
        end=3.0 if not args.quick else 1.0,
        fov=42,
        far=6000.0,
    )
    # The crane's clearance above the ground along its whole move (measured, not assumed).
    eye = []
    for t in np.linspace(0.0, 3.0 if not args.quick else 1.0, 61):
        eye.append(world_matrices(scene, a.overlay(float(t)))[crane.id][:3, 3])
    eye = np.array(eye)
    report["crane_clearance_m"] = round(
        float(np.min(eye[:, 2] - world.ground(eye[:, 0], eye[:, 1]))), 3
    )
    track = scene.add_camera(
        "Tracking", position=tuple(at(0, -6, 1.6)), look_at=tuple(at(0, 0, 1)), fov=36, far=6000.0
    )
    walker_mover = scene.get_by_name("walker mover")
    a.follow_camera(
        track,
        walker_mover,
        offset=(nx * -6.0 + dx * 1.5, ny * -6.0 + dy * 1.5, 1.1),
        lag=0.25,
        aim=(0.0, 0.0, 1.0),
        start=0.0,
        end=seconds,
    )
    ahead_s = v_walk * 7.5 + 6.0
    low = scene.add_camera(
        "Ahead",
        position=tuple(at(ahead_s + 5.0, 2.5, 0.35)),
        look_at=tuple(at(ahead_s - 2.0, 0.0, 1.2)),
        fov=40,
        far=6000.0,
    )
    cams = {"crane": crane, "tracking": track, "ahead": low}
    print("5. frames")
    frames_dir = out / "frames"
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
    shots = (
        SHOTS
        if not args.quick
        else ((0.0, 1.0, "crane"), (1.0, 2.0, "tracking"), (2.0, 3.0, "ahead"))
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    first, rows, sheet, labels, stills = 0, [], [], [], []
    opts = W.render_options("golden", distance=900.0)
    for t0, t1, name in shots:
        cam = cams[name]
        count = int(round((t1 - t0) * fps))
        secs = []
        for k in range(count):
            t = t0 + k / fps
            s0 = time.perf_counter()
            r = scene.render(camera=cam, time=t, width=size[0], height=size[1], **opts)
            secs.append(time.perf_counter() - s0)
            r.save(frames_dir / f"frame_{first + k:05d}.png")
        rows.append(
            {
                "shot": name,
                "start": t0,
                "end": t1,
                "frames": count,
                "seconds": round(sum(secs), 1),
                "seconds_per_frame_median": round(sorted(secs)[len(secs) // 2], 3),
            }
        )
        for k in np.linspace(0, count - 1, 3).round().astype(int):
            sheet.append(frames_dir / f"frame_{first + k:05d}.png")
            labels.append(f"{name} {t0 + k / fps:.2f}s")
        first += count
        mid = (t0 + t1) / 2
        if name == "tracking":  # (as the runner passes the walker: both in the shot)
            mid = min(max(6.0 / (v_run - v_walk) + 0.5, t0), t1)
        stills.append(
            W.render_still(
                scene,
                out / f"hero_{name}.png",
                camera=cam,
                time=mid,
                width=size[0],
                height=size[1],
                preset="golden",
                distance=900.0,
            )
        )
        W.preview(out / f"hero_{name}.png", out / f"hero_{name}_preview.png")
        print(f"   {name}: {count} frames, {rows[-1]['seconds']} s")
    report["frames"] = {"shots": rows, "total": first, "fps": fps, "size": list(size)}
    report["stills"] = stills
    C.sheet(sheet, out / "world_sheet.png", labels=labels, columns=3, width=480)
    if not args.no_video and not args.quick:
        report["video"] = C.encode(frames_dir, out, "world", fps)
    report["world_stats"] = world.runtime.stats()
    scene.save(out / "world.aether")
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
