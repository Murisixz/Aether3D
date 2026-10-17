"""Showcase 4 (Stage 17): cinematic cameras -- a path fly-through, an orbit, a dolly zoom
and a damped follow, cut and blended into one timecoded sequence.

Four cameras, every move ordinary keys or constraints (editable afterwards):

* **Crane** flies along a centripetal Catmull-Rom path at constant speed (arc-length
  parameterized, eased in and out) and aims at Nova;
* **Orbit** circles Nova 120 degrees on an exact circle, looking at her;
* **Vertigo** dollies in while widening the lens (a dolly zoom: the field of view is
  solved at every instant to keep Nova's height on screen, fitted into keys; the
  background stretches away behind her);
* **Follow** trails Nova as she runs along a curved path (a critically damped spring,
  0.35 s lag, fitted into keys), aimed at her chest.

A shot sequence cuts and blends between them (a blend interpolates the camera's position,
orientation -- slerp -- and lens). Nova idles, waves (an upper-body layer), then runs: the
run cycle plays in place while a follow-path constraint carries the armature along the
path at exactly the cycle's speed. Motion blur is real temporal sampling (sub-frame
renders averaged in linear light). **Depth of field is not simulated** by Aether3D's
renderers: there is none in these images.

Run from the repository root:

    python examples/animation_cinematics.py [OUTPUT_DIR] [--quick] [--no-video]

Writes to ``OUTPUT_DIR`` (default ``examples/output/animation/cinematics``):
``sequence_sheet.png`` (timecoded), ``shot_<name>.png`` (1280x720, one per shot),
``motion_blur.png`` (a fast frame without and with blur), ``cinematics.aether``,
``cinematics.mp4`` (FFmpeg, optional) and ``report.json`` (shots, path length and speed,
measured orbit radius, dolly-zoom subject size, render timings).
"""

from __future__ import annotations

import argparse
import math
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
    write_json,
)

from aether3d import Scene  # noqa: E402
from aether3d.animation.fitting import fit_samples  # noqa: E402
from aether3d.animation.frames import assemble_video, contact_sheet, render_frames  # noqa: E402
from aether3d.materials.images import decode_png  # noqa: E402
from aether3d.rendering.overlay import animation_overlay  # noqa: E402
from aether3d.rendering.png import encode_png, write_png  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402

FPS = 24
SHOTS = [  # (camera, start, end, transition into it)
    ("Crane", 0.0, 3.0, None),
    ("Orbit", 2.6, 5.5, {"type": "blend", "duration": 0.4}),
    ("Vertigo", 5.5, 8.0, None),
    ("Follow", 7.7, 12.0, {"type": "blend", "duration": 0.3}),
]


def build():
    s = Scene("Cinematics")
    pixel_world(s, size=40.0)
    lights(s)
    nova = blocky_character(s, "Nova", encode_png(pixel_skin()))
    a = s.animation
    a.set_timeline(fps=FPS, start=0.0, end=12.0)
    a.generate(nova, "idle", forward="-y", name="Idle", bind=False)
    wave = a.generate(nova, "wave", forward="-y", name="Wave", bind=False)
    run = a.generate(nova, "run", forward="-y", name="Run", travel=False, bind=False)
    a.set_action(nova, "Idle")
    a.add_layer(nova, "gestures", mask=[{"joint": "Body", "descendants": True}])
    a.add_strip(nova, "gestures", "Wave", start=3.0, blend_in=0.3, blend_out=0.3)
    a.add_layer(nova, "locomotion")
    a.add_strip(nova, "locomotion", "Run", start=8.0, repeat=6.0, blend_in=0.25)
    # The run's path: Nova runs it at exactly the cycle's speed (no sliding).
    pid = a.add_path("Track", [[0, 0, 0], [0, -4, 0], [3, -8, 0], [8, -9, 0], [12, -7, 0]])
    length = a.path(pid).length
    speed = run["speed"]
    reach = min(1.0, 4.0 * speed / length)
    a.follow_path(nova, "Track", start=8.0, end=12.0, offsets=(0.0, reach), easing=None,
                  follow=True, axis="-y", up="+z", name="run")  # fmt: skip
    a.set_channel(nova, 'constraints["run"].influence', [(7.9, 0.0), (8.0, 1.0)],
                  interpolation="linear")  # fmt: skip
    info = {"path_length_m": round(length, 4), "run_speed_m_s": round(speed, 4),
            "path_fraction_run": round(reach, 4), "wave_s": wave["duration"]}  # fmt: skip

    # Cameras.
    crane = s.add_camera("Crane", position=(-14, 10, 12), look_at=(0, 0, 1), fov=40)
    a.add_path("Crane path", [[-14, 10, 12], [-9, 0, 8.5], [-3, -7.5, 4.5], [2.5, -6.5, 2.2]])
    a.follow_path(crane, "Crane path", start=0.0, end=3.2, follow=False, name="fly")
    a.add_constraint(crane, {"name": "aim", "type": "look_at", "target": {"object": nova.id},
                             "target_offset": [0, 0, 1.0]})  # fmt: skip
    orbit = s.add_camera("Orbit", position=(0.0, -5.5, 1.9), look_at=(0, 0, 1.1), fov=38)
    a.camera_orbit(orbit, nova, 2.6, 5.5, degrees=120)
    vertigo = s.add_camera("Vertigo", position=(0.0, -9.0, 1.2), look_at=(0, 0, 1.1), fov=18)
    a.camera_move(vertigo, "dolly", 5.5, 5.5, 8.0)  # 9 m -> 3.5 m from Nova
    # The dolly zoom: the field of view that keeps Nova's height on screen at every
    # instant (fov = 2 atan(H / 2d) for her distance d), fitted into keys.
    times = np.linspace(5.5, 8.0, 121)
    fovs = []
    frame_height = 2 * 9.0 * math.tan(math.radians(18) / 2)  # (the view's height at 9 m)
    for t in times.tolist():
        w = world_matrices(s, animation_overlay(s, t))
        d = float(np.linalg.norm(w[vertigo.id][:3, 3] - w[nova.id][:3, 3] - [0, 0, 1.1]))
        fovs.append(math.degrees(2 * math.atan(frame_height / (2 * d))))
    curve, fit = fit_samples(times, fovs, 0.01)
    a.set_channel(vertigo, "camera.fov", curve)
    info["vertigo_fov_keys"] = fit["keys"]
    follow = s.add_camera("Follow", position=(2.5, 4.0, 2.2), look_at=(0, 0, 1.0), fov=42)
    a.follow_camera(follow, nova, start=7.5, end=12.0, offset=(3.0, 4.0, 2.2), lag=0.35,
                    aim=(0, 0, 1.0))  # fmt: skip
    a.set_shots([
        {"camera": c, "start": t0, "end": t1, "name": c.lower(),
         **({"transition": tr} if tr else {})}
        for c, t0, t1, tr in SHOTS
    ])  # fmt: skip
    return s, nova, info


def measure(s, nova) -> dict:
    """The orbit's radius, the vertigo subject's screen height, from evaluated worlds."""
    a = s.animation
    orbit = s.get_by_name("Orbit")
    radii = []
    for t in np.linspace(2.6, 5.5, 7):
        w = world_matrices(s, animation_overlay(s, float(t)))
        radii.append(float(np.linalg.norm(w[orbit.id][:2, 3] - w[nova.id][:2, 3])))
    vertigo = s.get_by_name("Vertigo")
    heights = []
    for t in (5.5, 6.1, 6.75, 7.4, 8.0 - 1e-6):
        w = world_matrices(s, animation_overlay(s, t))
        dist = float(np.linalg.norm(w[vertigo.id][:3, 3] - w[nova.id][:3, 3] - [0, 0, 1.1]))
        fov = a.evaluate(t).fields.get((vertigo.id, "camera"), {}).get("fov", vertigo.camera.fov)
        heights.append(2.0 / (2 * dist * math.tan(math.radians(fov) / 2)))  # Nova (2 m) / view
    return {"orbit_radius_m": [round(r, 6) for r in radii],
            "vertigo_subject_height_fraction": [round(h, 4) for h in heights]}  # fmt: skip


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output/animation/cinematics"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    out.mkdir(parents=True, exist_ok=True)
    w, h = (320, 180) if args.quick else (640, 360)
    s, nova, info = build()
    report = {"backend": backend(), "fps": FPS, "size": [w, h], **info, **measure(s, nova)}
    report["shots"] = s.animation.shots()
    s.save(out / "cinematics.aether")
    t0 = time.perf_counter()
    rep = render_frames(s, out / "frames", start=0.0, end=12.0, fps=FPS,
                        step=2 if args.quick else 1, width=w, height=h, backend=backend(),
                        shadows=shadows())  # fmt: skip
    report["render"] = {"frames": len(rep["frames"]), "seconds": round(time.perf_counter() - t0, 2)}
    frames = rep["frames"]
    pick = frames[:: max(1, len(frames) // 16)][:16]
    fps = s.animation.fps
    images = [decode_png(Path(f["path"]).read_bytes()).pixels[..., :3] for f in pick]
    labels = [
        f"{fps.timecode(f['time'])} {f['camera'] if isinstance(f['camera'], str) else 'blend'}"
        for f in pick
    ]
    write_png(out / "sequence_sheet.png", contact_sheet(images, columns=4, labels=labels))
    for cam, t0_, t1, _ in SHOTS:
        mid = 0.5 * (t0_ + t1)
        s.render(time=mid, width=1280, height=720, backend=backend(), shadows=shadows()).save(
            out / f"shot_{cam.lower()}.png"
        )
    # Motion blur: the crane's fastest frame, without and with temporal sampling.
    fast = 1.4
    still = s.render(time=fast, width=w, height=h, backend=backend(), shadows=shadows()).image
    blur_rep = render_frames(s, out / "frames" / "blur", frames=[fps.frame_at(fast)], width=w,
                             height=h, backend=backend(), shadows=shadows(),
                             motion_blur={"shutter": 0.5, "samples": 6})  # fmt: skip
    blurred = decode_png(Path(blur_rep["frames"][0]["path"]).read_bytes()).pixels[..., :3]
    write_png(out / "motion_blur.png",
              contact_sheet([still, blurred], columns=2,
                            labels=["no blur", "shutter 0.5, 6 samples"]))  # fmt: skip
    report["motion_blur_ms"] = round(1000 * blur_rep["frames"][0]["seconds"], 1)
    if not args.no_video:
        try:
            report["video"] = assemble_video(out / "frames", "frame_{frame:05d}.png",
                                             FPS if not args.quick else FPS / 2,
                                             out / "cinematics.mp4", crf=26)  # fmt: skip
        except Exception as exc:  # (FFmpeg is optional)
            report["video"] = {"skipped": str(exc)}
    write_json(out / "report.json", report)
    print({k: report[k] for k in ("path_length_m", "run_speed_m_s", "orbit_radius_m",
                                  "vertigo_subject_height_fraction", "render")})  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
