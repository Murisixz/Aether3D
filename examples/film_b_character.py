"""Showcase B -- the Stage 18 skinned figure greets the camera, lit and filmed with Stage 19.

The anatomical figure (``aether3d.sculpt.figure``: an original procedural body) is
rigged automatically (``autorig``: about 60 joints, bone heat weights) and dressed along
cut edges, as in the Stage 18 showcases. Its performance is layered, not replaced: an
idle breathes throughout (the action), the head turns to the camera and the right arm
waves on layers above it -- generated clips fading in and out with C2 ramps
(``blend_curve: "smoother"``; rotations blend by slerp) -- and a nod on an additive layer
is keyed by hand: two rotation keys of the head whose segments are timed with easings
(``sine`` in-out down; ``back`` in-out on the way up: it dips a touch further first,
rises a little past the rest pose and settles, at rest at both ends). The gestures'
transitions are measured in world space (``motion_report`` on the joints) with hard cuts
and crossfaded, and the planted feet are checked for sliding.

Lighting is a ``loop`` light rig (key, fill and rim area lights solved from the
figure's bounds) under a dim studio environment, on a seamless backdrop; the camera frames
it with the ``medium`` shot preset at f/2.8 (real depth of field, focused on the figure)
and drifts in. It writes:

* ``hero.png`` (1920x1080, GPU path tracer: Sobol', Blackman-Harris, adaptive, denoised,
  film transform) and ``hero_preview.png``; ``closeup.png``/``closeup_preview.png`` (the
  ``close_up`` preset at f/2: the face sharp, the shoulders soft);
* a 3.6-second 60 fps sequence, path traced: ``greeting_1080p.mp4``,
  ``greeting_master.mkv``, the committed ``greeting_preview.mp4``;
* ``report.json``: the rig, the measurements and the renders.

    .venv/bin/python examples/film_b_character.py [--quick] [--no-video]
        [--part all|hero|closeup|video]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rigging_common as C  # noqa: E402
from film_common import (  # noqa: E402
    OUT,
    arguments,
    film_options,
    preview,
    sequence,
    still,
    write_report,
)

from aether3d import Scene  # noqa: E402
from aether3d.animation import rotation as R  # noqa: E402

DURATION = 3.6
LOOK = 0.3  # the head turns to the camera
WAVE = 0.85  # the right arm waves
NOD = 2.75  # the nod
BLEND = 0.4


def build(quick: bool) -> tuple[Scene, object, dict]:
    scene = Scene("Greeting")
    C.cyclorama(scene, width=16.0, depth=8.0, height=6.0, radius=1.8, back=2.6)
    body = scene.add_object("Body", mesh=C.character_mesh(quick))
    a = scene.animation
    edges = C.tailor(scene, body)
    rig = a.autorig(body, name="Hero Rig")
    arm = scene.get_by_name("Hero Rig")
    report = {
        "rig": {"joints": len(rig["joints"]), "confidence": rig["confidence"],
                "diagnostics": [d["code"] for d in rig["diagnostics"]]},
        "clothes_faces": C.dress(scene, body, arm, edges),
        "triangles": body.mesh.triangle_count,
    }  # fmt: skip
    arm.rotation_euler = (0.0, 0.0, 18.0)  # (a three-quarter view of the figure)
    scene.world.environment.configure(type="image", image="builtin:studio", intensity=0.3,
                                      visible=False)  # fmt: skip
    scene.world.background = "#16171a"
    scene.world.ambient_intensity = 0.0
    rig_report = scene.light_rig("Body", "loop", temperature=5200.0, rim=1.2, level=1.2)
    report["light_rig"] = {"preset": "loop", "ratio": rig_report.get("ratio"),
                           "lights": [x["name"] for x in rig_report["lights"]]}  # fmt: skip
    cam = scene.add_camera("Camera", position=(0.0, -4.0, 1.4), look_at=(0.0, 0.0, 1.2))
    scene.active_camera = cam
    return scene, arm, report


def nod_clip(scene: Scene, arm) -> str:
    """The head's nod as a clip of rotation keys relative to the rest pose: down 11 degrees
    (``sine`` in-out: no jump in angular velocity where it starts), up with ``back``
    in-out (a little further down first, a little past the rest pose, at rest at both
    ends)."""
    a = scene.animation
    rig = a.rig(arm)
    head = rig.joints[rig.role_index("head")]
    rest = np.asarray(head.rotation, np.float64)
    down = R.multiply(rest, R.axis_angle(np.array([1.0, 0.0, 0.0]), np.array(math.radians(11.0))))
    clip = a.add_clip("Nod", target="object")
    path = f'joints["{head.name}"].rotation'
    clip.set_channel(path, [
        {"time": 0.0, "rotation": rest.tolist(), "timing": {"easing": "sine", "ease": "in_out"}},
        {"time": 0.3, "rotation": [float(v) for v in np.ravel(down)],
         "timing": {"easing": "back", "ease": "in_out"}},
        {"time": 0.75, "rotation": rest.tolist()},
    ])  # fmt: skip
    return clip.name


def perform(scene: Scene, arm) -> dict:
    a = scene.animation
    idle = a.generate(arm, "idle", name="Idle", forward="-y", bind=True)
    look = a.generate(arm, "head_turn", name="Look", forward="-y", bind=False, yaw=-28.0,
                      hold=1.6)  # fmt: skip
    wave = a.generate(arm, "wave", name="Wave", forward="-y", bind=False, side="R", waves=3)
    a.set_timeline(fps=60, start=0.0, end=DURATION)
    # Masks keep each gesture to its own joints (the generated clips pose the whole body:
    # unmasked, their legs would blend with the idle's and the planted feet would slide).
    masks = {"Look": [{"joint": "Neck", "descendants": True}],
             "Wave": [{"joint": "Shoulder.R", "descendants": True}]}  # fmt: skip
    for layer, clip, start in (("Look", "Look", LOOK), ("Wave", "Wave", WAVE)):
        a.add_layer(arm, layer, mask=masks[layer])
        a.add_strip(arm, layer, clip, start=start, blend_in=BLEND, blend_out=BLEND,
                    blend_curve="smoother")  # fmt: skip
    a.add_layer(arm, "Nod", mode="additive")
    a.add_strip(arm, "Nod", nod_clip(scene, arm), start=NOD)
    return {name: {k: m[k] for k in ("keys", "duration", "cyclic") if k in m}
            for name, m in (("idle", idle), ("look", look), ("wave", wave))}  # fmt: skip


def set_blends(scene: Scene, arm, seconds: float) -> None:
    a = scene.animation
    for layer in ("Look", "Wave"):
        a.update_strip(arm, layer, 0, blend_in=seconds, blend_out=seconds)


JUMPS = ("POSITION_JUMP", "VELOCITY_JUMP", "ROTATION_SNAP", "ANGULAR_VELOCITY_JUMP")


def joint_quality(scene: Scene, arm) -> dict:
    """World-space findings of the joints that the gestures move; ``jumps``: those of
    position, velocity or rotation (what a cut between poses makes)."""
    a = scene.animation
    joints = ["Head", "Neck", "UpperArm.R", "ForeArm.R", "Hand.R", "Spine2"]
    rig = a.rig(arm)
    joints = [j for j in joints if any(x.name == j for x in rig.joints)]
    rep = a.motion_report([{"object": arm.id, "joints": joints}], rate=240)
    codes: dict[str, int] = {}
    times = set()
    for f in rep.get("findings", []):
        codes[f["code"]] = codes.get(f["code"], 0) + 1
        if f.get("time") is not None:
            times.add(round(float(f["time"]), 3))
    jumps = sum(n for c, n in codes.items() if c in JUMPS)
    return {"joints": joints, "findings": codes, "jumps": jumps, "times": sorted(times)}


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "b_character"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    scene, arm, report = build(args.quick)
    report["showcase"] = "B skinned character"
    report["quick"] = args.quick
    report["motions"] = perform(scene, arm)
    a = scene.animation
    set_blends(scene, arm, 0.0)  # (the gestures with hard cuts, measured, then crossfaded)
    report["gestures_cuts"] = joint_quality(scene, arm)
    set_blends(scene, arm, BLEND)
    report["gestures_crossfades"] = joint_quality(scene, arm)
    report["motion_note"] = (
        "inside the wave, the generated clip's fitted keys (C1 Hermite) leave acceleration "
        "steps at the hand, with cuts or crossfades; the transitions add none when crossfaded"
    )
    feet = a.foot_contacts(arm, ["Foot.L", "Foot.R"], start=0.0, end=DURATION, rate=120.0)
    report["feet"] = {f: {"sliding_max_m": round(d["sliding_max"], 6),
                          "penetration_m": round(d["penetration"], 6)}
                      for f, d in feet["feet"].items()}  # fmt: skip
    body = scene.get_by_name("Body")
    # (shots are measured on the rest pose: a raised hand would grow the framed part)
    shot = a.frame_shot("Camera", body, "medium", azimuth=12.0, screen=[-0.15, -0.05],
                        fill=0.7, fstop=2.8, time=0.0)  # fmt: skip
    a.camera_move("Camera", "dolly", 0.25, 0.0, DURATION)
    report["shot"] = {k: shot[k] for k in ("preset", "distance", "focal_length") if k in shot}
    report["camera_quality"] = [f["code"] for f in
                                a.motion_report(["Camera"], subject="Body",
                                                rate=120).get("findings", [])]  # fmt: skip
    if args.part in ("all", "hero"):
        hero = still(scene, out / "hero.png", film_options(args.quick, samples=384,
                                                           adaptive=0.015), time=1.9)  # fmt: skip
        report["hero"] = {k: v for k, v in hero.items() if k != "image"}
        report["hero_preview"] = preview(hero["image"], out / "hero_preview.png")
    if args.part in ("all", "closeup"):
        cam2 = scene.add_camera("Close-up", position=(0.0, -2.0, 1.6), look_at=(0, 0, 1.6))
        a.frame_shot(cam2, body, "close_up", azimuth=20.0, screen=[0.12, 0.0], fstop=2.0,
                     time=0.0)  # fmt: skip
        close = still(scene, out / "closeup.png",
                      film_options(args.quick, samples=384, adaptive=0.015),
                      time=2.0, camera=cam2)  # fmt: skip
        report["closeup"] = {k: v for k, v in close.items() if k != "image"}
        report["closeup_preview"] = preview(close["image"], out / "closeup_preview.png")
        scene.remove(cam2)
    if not args.no_video and args.part in ("all", "video"):
        opts = film_options(args.quick, samples=32, adaptive=0.04)
        end = 2 / 60 if args.quick else DURATION
        report["greeting"] = sequence(scene, out, "greeting", start=0.0, end=end, options=opts,
                                      quick=args.quick)  # fmt: skip
    report["seconds"] = round(time.perf_counter() - t0, 1)
    write_report(out / "report.json", report)
    return report


if __name__ == "__main__":
    data = main()
    print(
        f"B character: {data['seconds']} s, gestures cut {data['gestures_cuts']['findings']} "
        f"-> crossfaded {data['gestures_crossfades']['findings']}"
    )
    print(json.dumps(data["feet"]))
