"""Showcase 1 (Stage 17): a Minecraft-style character with an original pixel skin --
idle, walk, run, jump and a superhero landing, with a three-quarter cinematic.

Everything is generated and rendered by Aether3D: the 64x64 skin is pixel art drawn by
``animation_common.pixel_skin`` (an original character, "Nova the courier", not any
game's), the boxes are UV-mapped to the standard skin layout, and the motions are
written by the motion generators as ordinary, editable keys (their parameters are in each
clip's metadata). Feet are measured, not assumed: the report gives each motion's foot
sliding and penetration from :meth:`AnimationSystem.foot_contacts`.

Run from the repository root:

    python examples/animation_minecraft.py [OUTPUT_DIR] [--quick] [--no-video]
        [--width W] [--height H]

Writes to ``OUTPUT_DIR`` (default ``examples/output/animation/minecraft``):

* ``nova_skin.png`` (the skin, 8x) and ``nova_views.png`` (front, three-quarter, back);
* ``frames/<motion>/frame_NNNNN.png`` -- every frame of every sequence (not committed);
* ``<motion>_sheet.png`` contact sheets, ``hero_<motion>.png`` key frames (1280x720);
* ``cinematic_sheet.png`` -- the landing through three cameras (cuts and a blend);
* ``nova.aether`` (the scene with every clip), ``nova.glb`` (rig, parts and the clips as
  glTF animations, checked by the Khronos validator when installed);
* ``nova.mp4`` (all sequences, when FFmpeg is installed);
* ``report.json`` -- clip sizes and fitting errors, foot measurements, render timings.
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

from animation_common import (  # noqa: E402
    backend,
    blocky_character,
    lights,
    pixel_skin,
    pixel_world,
    shadows,
    validate_glb,
    write_json,
)

from aether3d import Scene  # noqa: E402
from aether3d.animation.frames import assemble_video, contact_sheet, render_frames  # noqa: E402
from aether3d.materials.images import decode_png  # noqa: E402
from aether3d.rendering.png import encode_png, write_png  # noqa: E402

FPS = 24
FEET = ["Leg.L", "Leg.R"]


def build() -> tuple[Scene, object]:
    scene = Scene("Nova")
    pixel_world(scene)
    lights(scene)
    nova = blocky_character(scene, "Nova", encode_png(pixel_skin()))
    scene.animation.set_timeline(fps=FPS, start=0.0, end=4.0)
    return scene, nova


def generate(scene: Scene, nova) -> dict[str, dict]:
    """The motions, as clips (the last one bound; each sequence binds its own)."""
    a = scene.animation
    out = {}
    for name, kind, options in (
        ("Idle", "idle", {}),
        ("Walk", "walk", {}),
        ("Run", "run", {}),
        ("Jump", "jump", {"height": 0.55}),
        ("Land", "land", {"style": "superhero", "height": 5.0}),
    ):
        t0 = time.perf_counter()
        rep = a.generate(nova, kind, name=name, forward="-y", bind=False, **options)
        rep["seconds"] = round(time.perf_counter() - t0, 3)
        out[name] = {k: rep[k] for k in CLIP_FIELDS}
    return out


CLIP_FIELDS = ("clip", "keys", "channels", "max_error", "duration", "cyclic", "travel",
               "seconds", "quaternion_joints")  # fmt: skip


def feet_summary(report: dict) -> dict:
    """Measured sliding and penetration of each foot (from foot_contacts)."""
    return {
        foot: {
            "sliding_m": round(d["sliding_max"], 5),
            "penetration_m": round(d["penetration"], 5),
            "contacts": len(d["contacts"]),
        }
        for foot, d in report["feet"].items()
    }


def camera(scene: Scene, name: str, position, look_at, fov: float = 38.0):
    cam = scene.add_camera(name, position=position, look_at=look_at, fov=fov)
    return cam


def render_sequence(scene, name, out, *, start, end, width, height, camera_obj=None, blur=None):
    t0 = time.perf_counter()
    options = {"width": width, "height": height, "backend": backend(), "shadows": shadows()}
    if camera_obj is not None:
        options["camera"] = camera_obj
    report = render_frames(scene, out / "frames" / name, start=start, end=end, fps=FPS,
                           motion_blur=blur, **options)  # fmt: skip
    seconds = time.perf_counter() - t0
    frames = report["frames"]
    pick = frames[:: max(1, len(frames) // 12)][:12]
    images = [decode_png(Path(f["path"]).read_bytes()).pixels[..., :3] for f in pick]
    sheet = contact_sheet(images, columns=4, labels=[f"{f['time']:.2f} s" for f in pick])
    write_png(out / f"{name}_sheet.png", sheet)
    return {"frames": len(frames), "seconds": round(seconds, 2),
            "ms_per_frame": round(1000 * seconds / max(1, len(frames)), 1),
            "backend": options["backend"], "paths": [f["path"] for f in frames]}  # fmt: skip


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output/animation/minecraft"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    out.mkdir(parents=True, exist_ok=True)
    w, h = (320, 180) if args.quick else (args.width, args.height)
    report: dict = {"backend": backend(), "fps": FPS, "size": [w, h]}

    scene, nova = build()
    a = scene.animation
    skin = pixel_skin()
    write_png(out / "nova_skin.png", np.kron(skin[..., :3], np.ones((8, 8, 1), np.uint8)))
    views = []
    for pos in ((0, -4.6, 1.4), (3.0, -3.6, 1.9), (0, 4.6, 1.4)):
        cam = camera(scene, f"View {len(views)}", pos, (0, 0, 1.0))
        views.append(scene.render(camera=cam, width=360, height=360, backend=backend(),
                                  shadows=shadows()).image)  # fmt: skip
        scene.remove(cam)
    labels = ["front", "three-quarter", "back"]
    write_png(out / "nova_views.png", contact_sheet(views, columns=3, labels=labels))
    report["clips"] = generate(scene, nova)
    print(
        "clips:",
        {k: (v["keys"], round(v["max_error"]["degrees"], 3)) for k, v in report["clips"].items()},
    )

    # Three-quarter cameras: one that follows Nova (a critically damped follow, for the
    # walk and the run) and one that stands still (the jump's height stays visible).
    follow = camera(scene, "Follow", (3.2, -3.4, 1.9), (0, 0, 1.0))
    still = camera(scene, "Three-quarter", (3.6, -4.2, 2.0), (0, 0, 1.15))
    sequences = {}
    contacts = {}
    shorten = 0.5 if args.quick else 1.0
    for clip, end in (("Idle", 4.0), ("Walk", 2 * 1.111), ("Run", 2 * 0.714), ("Jump", None)):
        a.set_action(nova, clip)
        end = report["clips"][clip]["duration"] if end is None else end
        end *= shorten
        a.set_timeline(start=0.0, end=end)
        cam = still
        if clip in ("Walk", "Run"):
            a.follow_camera(follow, nova, offset=(3.0, -3.4, 1.9), lag=0.25, aim=(0, 0, 1.0),
                            start=0.0, end=end)  # fmt: skip
            cam = follow
        sequences[clip] = render_sequence(scene, clip.lower(), out, start=0.0, end=end,
                                          width=w, height=h, camera_obj=cam)  # fmt: skip
        c = a.foot_contacts(nova, FEET, start=0.0, end=end, rate=120, height=0.03, speed=0.3)
        contacts[clip] = feet_summary(c)
        mid = end * (0.3 if clip != "Jump" else 0.55)
        hero = scene.render(camera=cam, time=mid, width=1280, height=720, backend=backend(),
                            shadows=shadows())  # fmt: skip
        hero.save(out / f"hero_{clip.lower()}.png")
        print(clip, sequences[clip]["frames"], "frames", contacts[clip])

    # The cinematic: a superhero landing through three cameras (cut, blend, cut).
    a.set_action(nova, "Land")
    land = report["clips"]["Land"]
    t_hit = float(np.sqrt(2 * 5.0 / 9.81))  # (the fall: 5 m under gravity, to the knees' reach)
    end = land["duration"]  # (the whole landing, even with --quick)
    a.set_timeline(start=0.0, end=end)
    wide = camera(scene, "Wide", (7.5, -9.0, 1.2), (0, 0, 1.6), fov=42)
    a.camera_move(wide, "dolly", 2.0, 0.0, t_hit)
    camera(scene, "Close", (-2.7, -2.9, 0.75), (-0.15, -0.25, 0.6), fov=40)
    camera(scene, "Low", (0.0, -3.4, 0.25), (0, 0, 1.15), fov=46)
    a.set_shots([
        {"camera": "Wide", "start": 0.0, "end": t_hit + 0.1, "name": "fall"},
        {"camera": "Close", "start": t_hit - 0.2, "end": t_hit + 1.5, "name": "impact",
         "transition": {"type": "blend", "duration": 0.3}},
        {"camera": "Low", "start": t_hit + 1.5, "end": end, "name": "rise"},
    ])  # fmt: skip
    sequences["Land"] = render_sequence(scene, "cinematic", out, start=0.0, end=end, width=w,
                                        height=h, blur={"shutter": 0.5, "samples": 3})  # fmt: skip
    c = a.foot_contacts(nova, FEET, start=0.0, end=end, rate=120, height=0.03, speed=0.3)
    contacts["Land"] = feet_summary(c)
    for label, t in (("impact", t_hit + 0.45), ("rise", t_hit + 1.9)):
        frame = scene.render(time=min(t, end), width=1280, height=720, backend=backend(),
                             shadows=shadows())  # fmt: skip
        frame.save(out / f"hero_land_{label}.png")
    report["shots"] = a.shots()
    report["impact_s"] = round(t_hit, 4)
    report["contacts"] = contacts
    report["sequences"] = {k: {kk: vv for kk, vv in v.items() if kk != "paths"}
                           for k, v in sequences.items()}  # fmt: skip

    # Keep the whole scene (clips included) and export it.
    a.set_timeline(start=0.0, end=4.0)
    a.set_shots(None)
    a.set_action(nova, "Idle")
    scene.save(out / "nova.aether")
    exp = scene.export(out / "nova.glb", animation="clips")
    report["gltf"] = {"bytes": exp.byte_length, "animations": exp.animations,
                      "validator": validate_glb(out / "nova.glb")}  # fmt: skip
    print("glTF:", report["gltf"]["validator"])
    if not args.no_video:
        combined = out / "frames" / "all"
        combined.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in ("idle", "walk", "run", "jump", "cinematic"):
            for src in sorted((out / "frames" / name).glob("frame_*.png")):
                shutil.copyfile(src, combined / f"frame_{n:05d}.png")
                n += 1
        try:
            report["video"] = assemble_video(combined, "frame_{frame:05d}.png", FPS,
                                             out / "nova.mp4")  # fmt: skip
        except Exception as exc:  # (FFmpeg is optional)
            report["video"] = {"skipped": str(exc)}
    write_json(out / "report.json", report)
    print("done:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
