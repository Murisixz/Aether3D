"""Showcase 5 (Stage 17): a large animated scene -- a wave pool of 10,000 instanced blocks
and a crowd of 48 walking characters -- with every stage of a frame measured.

* **Wave pool**: 10,000 blocks (the grid points 0.5 m apart nearest the centre: a round
  pool of 28 m radius) play ONE shared two-second clip (a cyclic Bezier stretch: each
  block rises to six times its height and settles, its base fixed) through NLA strips
  whose start is offset by the block's distance from the centre: a stadium wave rolling
  outward from one clip and 10,000 strips, no per-block keys.
* **Crowd**: 48 blocky characters (four skin variants; six part meshes shared by all) on
  four concentric circles around the pool. Each plays the generated walk cycle in place,
  phase-offset by its strip, while a looping follow-path constraint carries it along its
  circle at the cycle's own speed (the feet stay planted).
* **Camera**: a crane that starts among the walkers and rises to reveal the pool (a
  path followed at constant speed, aimed at a focus point that drifts to the centre).

Measured on this machine: the build, then for sampled frames the animation evaluation,
the render overlay (with the rigging pass of 48 rigs), the posed world matrices, and the
whole frame (overlay, extraction, render, PNG) on the GPU -- with a few frames on the CPU
for comparison -- plus counts, peak memory and the GPU adapter. Nothing is estimated.

Run from the repository root:

    python examples/animation_large_scene.py [OUTPUT_DIR] [--quick] [--no-video]

(``--quick``: a smoke test -- 2,500 blocks and 12 walkers, every 8th frame, small images.)

Writes to ``OUTPUT_DIR`` (default ``examples/output/animation/large``):
``sequence_sheet.png``, ``hero_{crowd,pool}.png`` (1280x720), ``large.aether``,
``large.mp4`` (FFmpeg, optional) and ``report.json``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from animation_common import (  # noqa: E402
    backend,
    blocky_character,
    pixel_skin,
    save_png,
    shadows,
    skin_material,
    use_backend,
    write_json,
)

from aether3d import Scene  # noqa: E402
from aether3d.animation import FCurve  # noqa: E402
from aether3d.animation.clip import Clip  # noqa: E402
from aether3d.animation.frames import assemble_video, contact_sheet, render_frames  # noqa: E402
from aether3d.materials.images import decode_png  # noqa: E402
from aether3d.rendering.png import encode_png  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402

FPS = 24
DURATION = 8.0
BLOCKS = 10_000
PITCH = 0.5  # metres between blocks
RINGS = (31.0, 33.5, 36.0, 38.5)  # crowd circles (the pool's radius is 28.2 m)
PER_RING = 12
VARIANTS = [  # (jacket, jacket shade, pants, hair)
    ((34, 136, 132), (24, 104, 102), (44, 54, 92), (74, 46, 34)),
    ((186, 64, 60), (148, 46, 44), (60, 60, 64), (30, 24, 20)),
    ((230, 180, 60), (196, 146, 40), (52, 84, 60), (150, 98, 50)),
    ((110, 80, 170), (84, 58, 136), (40, 40, 48), (220, 200, 150)),
]
WATER = [(0.16, 0.42, 0.72), (0.2, 0.5, 0.78), (0.26, 0.58, 0.82), (0.36, 0.68, 0.88)]


def build(blocks: int = BLOCKS, per_ring: int = PER_RING) -> tuple[Scene, dict]:
    s = Scene("Large")
    s.world.background = (0.62, 0.78, 0.95)
    s.world.ambient_intensity = 0.3
    ground = s.add_plane("Ground", size=120)
    ground.color = (0.46, 0.62, 0.36)
    s.add_directional_light("Sun", direction=(-0.45, 0.35, -0.8), intensity=2.6)
    a = s.animation
    a.set_timeline(fps=FPS, start=0.0, end=DURATION)
    info: dict = {}

    # The pool: one clip, 10,000 offset strips.
    t0 = time.perf_counter()
    # A block stretches up to six times its height and settles, its base fixed: the
    # centre rises by half the stretch (auto-clamped handles are affine-invariant, so
    # the two curves stay in proportion between the keys too).
    stretch = [(0.0, 1.0), (0.35, 6.0), (0.8, 0.7), (1.1, 1.0), (2.0, 1.0)]
    lift = [(t, 0.22 * (v - 1.0)) for t, v in stretch]
    ripple = a.add_clip("Ripple", Clip()
                        .with_channel("transform.scale.z",
                                      FCurve.from_keys(stretch, extrapolation="cycle"))
                        .with_channel("transform.position.z",
                                      FCurve.from_keys(lift, extrapolation="cycle")))  # fmt: skip
    materials = [s.create_material(f"Water {i}", roughness=0.35) for i in range(len(WATER))]
    for m, color in zip(materials, WATER, strict=True):
        m.base_color = color
    n = 2 * math.ceil(math.sqrt(blocks / math.pi)) + 4
    grid = (np.indices((n, n)).reshape(2, -1).T - (n - 1) / 2) * PITCH
    grid = grid[np.argsort(np.hypot(grid[:, 0], grid[:, 1]), kind="stable")[:blocks]]
    info["pool_radius_m"] = round(float(np.hypot(*grid[-1])), 3)
    tones = np.random.default_rng(9).integers(0, len(materials), blocks).tolist()
    with s.history._suspended():  # (building: no undo steps for 10,000 objects)
        mesh = s.add_cube(name="Block", size=0.44, position=(0, 0, -50)).mesh
        for i, (x, y) in enumerate(grid.tolist()):
            r = math.hypot(x, y)
            block = s.add_object(f"B{i}", mesh=mesh, position=(x, y, 0.0))
            block.material = materials[tones[i]]  # (random tones: the wave is the shape)
            strip = {"clip": ripple.name, "start": -((-r * 0.12) % 2.0), "repeat": 2 + DURATION / 2}
            a.set_binding(block, {"layers": [{"name": "ripple", "strips": [strip]}]})
    info["pool_build_s"] = round(time.perf_counter() - t0, 3)

    # The crowd: a walk cycle generated once (in place), played by every character.
    t0 = time.perf_counter()
    skins = [skin_material(s, f"Walker {k}", encode_png(pixel_skin(
        seed=11 + k, jacket=j, jacket_dark=jd, pants=p, hair=h)))
        for k, (j, jd, p, h) in enumerate(VARIANTS)]  # fmt: skip
    first = blocky_character(s, "Walker 0", position=(RINGS[0], 0, 0), material=skins[0])
    walk = a.generate(first, "walk", forward="-y", name="Walk", travel=False, bind=False)
    period, speed = walk["period"], walk["speed"]
    walkers = [first]
    for k in range(1, len(RINGS) * per_ring):
        walkers.append(blocky_character(s, f"Walker {k}", position=(RINGS[0], 0, 0),
                                        material=skins[k % len(skins)]))  # fmt: skip
    rng = np.random.default_rng(4)
    for k, w in enumerate(walkers):
        ring, slot = divmod(k, per_ring)
        radius = RINGS[ring]
        direction = 1 if ring % 2 == 0 else -1  # (neighbouring rings walk opposite ways)
        a.add_path(f"Ring {k}", kind="circle", center=[0, 0, 0], radius=radius,
                   normal=[0, 0, direction], start=0.0)  # fmt: skip
        phase = slot / per_ring + float(rng.uniform(-0.02, 0.02))
        laps = speed * DURATION / (2 * math.pi * radius)
        # (the binding first: follow_path keys the path offset into the action beside it)
        a.set_binding(w, {"layers": [{"name": "walk", "strips": [{
            "clip": "Walk", "start": -float(rng.uniform(0, period)),
            "repeat": 2 + DURATION / period}]}]})  # fmt: skip
        a.follow_path(w, f"Ring {k}", start=0.0, end=DURATION, offsets=(phase, phase + laps),
                      easing=None, follow=True, axis="-y", loop=True)  # fmt: skip
    info["crowd_build_s"] = round(time.perf_counter() - t0, 3)
    info["walk"] = {"period_s": round(period, 4), "speed_m_s": round(speed, 4),
                    "keys": walk["keys"]}  # fmt: skip

    # The crane: from among the walkers up to a high view of the pool.
    focus = s.add_object("Focus", position=(RINGS[1], -2.0, 1.0))
    a.set_channel(focus, "transform.position.x", [(0.0, RINGS[1]), (2.0, RINGS[1] - 2),
                                                  (6.5, 0.0)], interpolation="sine")  # fmt: skip
    a.set_channel(focus, "transform.position.y", [(0.0, -2.0), (6.5, 0.0)],
                  interpolation="sine")  # fmt: skip
    a.set_channel(focus, "transform.position.z", [(0.0, 1.0), (6.5, 0.0)], interpolation="sine")
    cam = s.add_camera("Crane", position=(RINGS[1] + 4.5, -9.0, 1.6), look_at=(RINGS[1], 0, 1),
                       fov=42)  # fmt: skip
    a.add_path("Crane path", [[RINGS[1] + 4.5, -9.0, 1.6], [RINGS[1] + 10, -16, 4.5],
                              [RINGS[1] + 20, -32, 13.0], [RINGS[1] + 28, -50, 22.0]])  # fmt: skip
    a.follow_path(cam, "Crane path", start=0.0, end=DURATION, follow=False, name="crane")
    a.add_constraint(cam, {"name": "aim", "type": "look_at", "target": {"object": focus.id}})
    s.active_camera = cam
    info["objects"] = len(list(s.objects))
    info["triangles"] = int(sum(o.mesh.triangle_count for o in s.objects if o.mesh is not None))
    info["strips"] = blocks + len(walkers)
    info["walkers"] = len(walkers)
    return s, info


def stage_timings(s: Scene, times) -> dict:
    """Median per-frame cost of each stage at fresh times (nothing cached between)."""
    a = s.animation
    ev, ov, wm = [], [], []
    for t in times:
        t0 = time.perf_counter()
        a.evaluate(t)
        ev.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        overlay = a.overlay(t + 1e-4)
        ov.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        world_matrices(s, overlay)
        wm.append(time.perf_counter() - t0)
    med = lambda xs: round(1000 * statistics.median(xs), 2)  # noqa: E731
    return {
        "evaluate_ms": med(ev),
        "overlay_with_rigging_ms": med(ov),
        "world_matrices_ms": med(wm),
    }


def feet(s: Scene, walkers=(0, 13, 26, 39), end: float = 5.0) -> dict:
    """Planted contacts and foot sliding of a few walkers from 1 s to ``end`` (measured;
    a walker with no contacts would be sliding or floating, and shows as such)."""
    a = s.animation
    out = {}
    for k in walkers:
        name = f"Walker {k}"
        c = a.foot_contacts(name, ["Leg.L", "Leg.R"], start=1.0, end=end, rate=120,
                            height=0.03, speed=0.3)  # fmt: skip
        out[name] = {
            "contacts": sum(len(d["contacts"]) for d in c["feet"].values()),
            "sliding_max_m": round(max(d["sliding_max"] for d in c["feet"].values()), 5),
            "penetration_m": round(max(d["penetration"] for d in c["feet"].values()), 5),
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output/animation/large"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args(argv)
    use_backend(args.backend)
    out = Path(args.output)
    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    out.mkdir(parents=True, exist_ok=True)
    # --quick: a small version of the scene (2,500 blocks, 12 walkers) for smoke tests
    w, h = (320, 180) if args.quick else (960, 540)
    t0 = time.perf_counter()
    s, info = build(2_500, 3) if args.quick else build()
    report: dict = {"backend": backend(), "fps": FPS, "size": [w, h], **info}
    report["build_s"] = round(time.perf_counter() - t0, 3)
    print("built:", {k: report[k] for k in ("objects", "triangles", "strips", "build_s")})
    sample = np.linspace(0.3, DURATION - 0.3, 9).tolist()
    report["per_frame"] = stage_timings(s, sample)
    print("per frame:", report["per_frame"])
    report["foot_sliding_m"] = feet(s, (0,), 3.0) if args.quick else feet(s)
    t0 = time.perf_counter()
    s.save(out / "large.aether")
    report["save_s"] = round(time.perf_counter() - t0, 3)
    report["file_bytes"] = (out / "large.aether").stat().st_size

    # The sequence on the GPU (when available), then a few frames on the CPU.
    step = 8 if args.quick else 1
    opts = {"width": w, "height": h, "shadows": shadows()}
    render_frames(s, out / "frames" / "warm", frames=[0], backend=backend(), **opts)
    t0 = time.perf_counter()
    rep = render_frames(s, out / "frames", start=0.0, end=DURATION, fps=FPS, step=step,
                        backend=backend(), **opts)  # fmt: skip
    seconds = time.perf_counter() - t0
    frames = rep["frames"]
    per = [f["seconds"] for f in frames]
    report["sequence"] = {"backend": backend(), "frames": len(frames),
                          "seconds": round(seconds, 2),
                          "frame_ms_median": round(1000 * statistics.median(per), 1),
                          "frame_ms_max": round(1000 * max(per), 1)}  # fmt: skip
    print("sequence:", report["sequence"])
    if backend() == "gpu" and not args.quick:
        # the same three frames on both renderers (no shadows: the CPU renderer draws
        # none), after a warm-up frame on each
        same = {"frames": [24, 96, 168], "width": w, "height": h, "shadows": False}
        for name in ("gpu", "cpu"):
            render_frames(s, out / "frames" / f"warm_{name}", frames=[0], width=w, height=h,
                          backend=name)  # fmt: skip
            rep2 = render_frames(s, out / "frames" / name, backend=name, **same)
            report[f"{name}_frame_ms_no_shadows"] = [round(1000 * f["seconds"], 1)
                                                     for f in rep2["frames"]]  # fmt: skip
        print("gpu vs cpu (no shadows):", report["gpu_frame_ms_no_shadows"],
              report["cpu_frame_ms_no_shadows"])  # fmt: skip
    pick = frames[:: max(1, len(frames) // 12)][:12]
    images = [decode_png(Path(f["path"]).read_bytes()).pixels[..., :3] for f in pick]
    fps = s.animation.fps
    labels = [fps.timecode(f["time"]) for f in pick]
    save_png(
        out / "sequence_sheet.png",
        contact_sheet(images, columns=4, labels=labels),
        "a contact sheet of Aether3D renders",
    )
    hw, hh = (640, 360) if args.quick else (1280, 720)
    for name, t in (("crowd", 0.9), ("pool", DURATION - 0.4)):
        s.render(time=t, width=hw, height=hh, backend=backend(), shadows=shadows()).save(
            out / f"hero_{name}.png"
        )
    try:
        from aether3d.rendering.gpu import gpu_status

        status = gpu_status()
        report["gpu"] = {k: status[k] for k in ("available", "selected", "reason") if k in status}
    except Exception as exc:  # (no GPU stack)
        report["gpu"] = {"available": False, "reason": str(exc)}
    if not args.no_video:
        try:
            report["video"] = assemble_video(out / "frames", "frame_{frame:05d}.png",
                                             FPS / step, out / "large.mp4", crf=26)  # fmt: skip
        except Exception as exc:  # (FFmpeg is optional)
            report["video"] = {"skipped": str(exc)}
    from bench_engine import peak_rss_mib

    report["peak_rss_mib"] = peak_rss_mib()
    write_json(out / "report.json", report)
    print("outputs:", out)
    print("done")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "benchmarks"))
    sys.exit(main(sys.argv[1:]))
