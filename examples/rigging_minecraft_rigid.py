"""Showcase E (Stage 18): a Minecraft-inspired segmented character animated by **rigid
object transforms only** -- no armature, no skin, no weights -- proving Stage 18 imposes
no bones on that workflow.

"Kit, the cartographer" is an original character (a 64x64 pixel skin drawn by
``animation_common.pixel_skin`` with its own palette; boxes UV-mapped to the standard
layout): six solid parts in a plain object hierarchy -- a root, the torso pivoting at the
waist, the head at the neck, the arms at the shoulders, the legs at the hips.
``generate_rigid`` writes each motion -- idle, walk, run, jump, wave -- as ordinary
``transform`` clips on those objects (travel on the root); the scene never holds a rig or
a skin, and the report proves it. Each segment's clips play in turn through a pixel world
(grass, trees), followed by a tracking camera under a warm key sun, a cool fill, a rim
from behind and a sky.

Measured: the rigid motions' verification (the parts in the scene against the generated
motion), the feet's sliding, and the per-frame cost of animating the rigid character
against the same character as bone-attached blocks (Stage 17's Nova workflow).

Run from the repository root:

    python examples/rigging_minecraft_rigid.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--no-video] [--fps 60] [--samples 192]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/e_rigid``): ``kit_skin.png``
(8x), ``kit_sheet.png``, ``hero_*.png`` (and previews), ``frames/`` (not committed),
``kit_1080p.mp4``/``kit_master.mkv``/``kit_preview.mp4``, ``kit.aether``, ``report.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rigging_common as C  # noqa: E402
from animation_common import (  # noqa: E402
    PX,
    _fix_winding,
    blocky_character,
    part_box,
    pixel_skin,
    pixel_world,
    skin_material,
)

from aether3d import Scene  # noqa: E402
from aether3d.rendering.png import encode_png  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402

KIT = {
    "jacket": (186, 120, 52),
    "jacket_dark": (150, 92, 38),
    "pants": (58, 74, 54),
    "pants_dark": (44, 58, 42),
    "hair": (196, 150, 82),
    "hair_light": (222, 182, 110),
    "iris": (64, 132, 74),
    "badge": (60, 170, 200),
    "cuff": (60, 170, 200),
    "bag": (120, 84, 54),
}
# part: (parent, pivot (px, in the root's space), box centre relative to the pivot (px))
PARTS = {
    "torso": (None, (0, 0, 12), (0, 0, 6), "body"),
    "head": ("torso", (0, 0, 24), (0, 0, 4), "head"),
    "arm.R": ("torso", (-6, 0, 22), (0, 0, -4), "arm_r"),
    "arm.L": ("torso", (6, 0, 22), (0, 0, -4), "arm_l"),
    "leg.R": (None, (-2, 0, 12), (0, 0, -6), "leg_r"),
    "leg.L": (None, (2, 0, 12), (0, 0, -6), "leg_l"),
}
SEGMENTS = (
    ("idle", 2.0, {}),
    ("walk", 3.0, {}),
    ("run", 2.5, {}),
    ("jump", 2.0, {"height": 0.55}),
    ("wave", 2.5, {}),
)


def kit(scene: Scene, material) -> dict:
    """The rigid hierarchy: objects only, origins at the joints."""
    root = scene.add_object("Kit")
    objs = {}
    for part, (parent, pivot, centre, net) in PARTS.items():
        p = np.asarray(pivot, float) * PX
        above = objs[parent] if parent else None
        local = p - (np.asarray(PARTS[parent][1], float) * PX if parent else 0.0)
        obj = scene.add_object(
            f"Kit {part}",
            parent=above or root,
            position=tuple(local),
            mesh=_fix_winding(part_box(net, tuple(np.asarray(centre) * PX))),
        )
        obj.material = material
        objs[part] = obj
    return {"root": root, **objs}


def frame_cost(scene: Scene, n: int = 120) -> float:
    """Milliseconds per frame to evaluate the animation and the posed world matrices."""
    a = scene.animation
    t0 = time.perf_counter()
    for k in range(n):
        world_matrices(scene, a.overlay(0.0137 * k + 0.001))
    return (time.perf_counter() - t0) / n * 1000.0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "e_rigid"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=192)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = C.set_backend(args.backend)
    size = (320, 180) if args.quick else (1920, 1080)
    fps = 10 if args.quick else args.fps
    t_all = time.perf_counter()
    scene = Scene("Showcase E")
    pixel_world(scene, size=48.0)
    # Cinematic daylight: a warm key sun, a cool sky fill, a rim from behind, a sky.
    scene.world.background = (0.56, 0.74, 0.95)
    scene.world.ambient_color = (0.74, 0.84, 1.0)
    scene.world.ambient_intensity = 0.28
    scene.world.environment.configure(
        type="sky",
        intensity=0.9,
        zenith_color=(0.36, 0.56, 0.9),
        horizon_color=(0.78, 0.86, 0.95),
        ground_color=(0.3, 0.36, 0.22),
    )
    lights = {
        "key": ((0.42, 0.55, -0.72), (1.0, 0.93, 0.8), 2.5, 1.5),
        "fill": ((-0.8, 0.3, -0.35), (0.7, 0.8, 1.0), 0.45, 10.0),
        "rim": ((-0.25, -0.75, -0.6), (1.0, 0.95, 0.88), 1.2, 3.0),
    }
    for name, (d, color, inten, ang) in lights.items():
        light = scene.add_directional_light(name.title(), direction=d, intensity=inten)
        light.light.configure(color=color, angle=ang)
    report: dict = {
        "backend": backend,
        "machine": C.machine(),
        "lighting": {
            k: {"direction": v[0], "color": v[1], "intensity": v[2], "angle_deg": v[3]}
            for k, v in lights.items()
        },
    }
    skin_img = pixel_skin(seed=11, **KIT)
    C.save(out / "kit_skin.png", np.kron(skin_img[..., :3], np.ones((8, 8, 1), np.uint8)))
    mat = skin_material(scene, "Kit", encode_png(skin_img))
    parts = kit(scene, mat)
    root = parts["root"]
    a = scene.animation
    print("1. parts (no rig)")
    report["parts"] = a.rigid_parts(root)
    print(
        f"   {[(r, p['object'], p['pivot_source']) for r, p in report['parts']['parts'].items()]}"
    )
    print("2. motions (object transforms)")
    motions = {}
    for kind, _, params in SEGMENTS:
        travel = kind in ("walk", "run")
        # (bound as generated: the report verifies the scene against the motion)
        motions[kind] = a.generate_rigid(root, kind, name=kind.title(), travel=travel, **params)
    a.set_timeline(start=0.0, end=max(d for _, d, _ in SEGMENTS))
    report["motions"] = {
        k: {
            "clips": len(v["clips"]),
            "keys": v["keys"],
            "max_error": v["max_error"],
            "travel_m": v["travel"],
            "duration": v["duration"],
            "feet": v.get("feet"),
            "verification": v["verification"],
            "cleared": v["cleared"],
        }
        for k, v in motions.items()
    }
    rigs = sum(1 for o in scene.objects if a.rig(o) is not None)
    skins = sum(1 for k, _ in a._owned if k == "skin")
    report["bones"] = {"rigs": rigs, "skins": skins}
    assert rigs == 0 and skins == 0, "the rigid workflow must not create bones"
    print(f"   rigs {rigs}, skins {skins}")
    # Cost per frame, the character alone: rigid parts vs the same blocks on bones.
    alone = Scene("Rigid alone")
    kit_alone = kit(alone, skin_material(alone, "Kit", encode_png(skin_img)))
    alone.animation.generate_rigid(kit_alone["root"], "walk")
    rigid_ms = frame_cost(alone)
    ref = Scene("Attached alone")
    nova = blocky_character(ref, "Nova", encode_png(skin_img))
    ref.animation.generate(nova, "walk", forward="-y")
    attached_ms = frame_cost(ref)
    report["frame_cost_ms"] = {
        "rigid_parts": round(rigid_ms, 4),
        "bone_attached_blocks": round(attached_ms, 4),
    }
    print(f"   cost per frame: rigid {rigid_ms:.3f} ms, attached {attached_ms:.3f} ms")
    print("3. renders")
    cam = scene.add_camera("Camera", position=(0.0, -6.0, 1.6), look_at=(0, 0, 1.0), fov=34)
    frames_dir = out / "frames"
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
    first, rows, sheet, labels, stills = 0, [], [], [], []
    for kind, duration, _ in SEGMENTS:
        for obj in [root, *(parts[p] for p in PARTS)]:
            a.set_action(obj, None)
        for obj_name, clip in motions[kind]["clips"].items():
            a.set_action(scene.get_by_name(obj_name), clip)
        travel = kind in ("walk", "run")
        root.rotation_euler = (0.0, 0.0, 90.0 if travel else 20.0)
        root.position = (-4.0 if kind == "walk" else -5.0, 0.0, 0.0) if travel else (0, 0, 0)
        a.set_action(cam, None)
        if travel:
            a.follow_camera(
                cam,
                root,
                offset=(0.8, -6.0, 0.7),
                lag=0.25,
                aim=(0.0, 0.0, 1.0),
                start=0.0,
                end=duration,
            )
        else:
            cam.position = (1.6, -5.4, 1.5)
            cam.look_at((0.0, 0.0, 1.05))
        dur = duration if not args.quick else min(duration, 0.8)
        info = C.render_frames(
            scene,
            frames_dir,
            camera=cam,
            fps=fps,
            start=0.0,
            end=dur,
            width=size[0],
            height=size[1],
            first=first,
            progress=C.progress_printer(kind),
        )
        info["segment"] = kind
        rows.append(info)
        for k in np.linspace(0, info["frames"] - 1, 3).round().astype(int):
            sheet.append(frames_dir / f"frame_{first + k:05d}.png")
            labels.append(f"{kind} {k / fps:.2f}s")
        first += info["frames"]
        hero_t = {"idle": 1.0, "walk": 0.4, "run": 0.25, "jump": 0.7, "wave": 1.2}[kind]
        stills.append(
            C.still(
                scene,
                out / f"hero_{kind}.png",
                camera=cam,
                time=min(hero_t, dur - 1 / fps),
                width=size[0],
                height=size[1],
                path_samples=16 if args.quick else args.samples,
            )
        )
        C.preview(out / f"hero_{kind}.png", out / f"hero_{kind}_preview.png")
    report["frames"] = {"segments": rows, "total": first, "fps": fps, "size": list(size)}
    report["stills"] = stills
    C.sheet(sheet, out / "kit_sheet.png", labels=labels, columns=5, width=384)
    if not args.no_video and not args.quick:
        report["video"] = C.encode(frames_dir, out, "kit", fps)
    scene.save(out / "kit.aether")
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
