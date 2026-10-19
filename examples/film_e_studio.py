"""Showcase E -- studio lighting: a product still life lit by Aether3D's light rigs.

Glass, chrome, gold, frosted glass and wood on a plinth in front of a seamless backdrop,
lit by ``scene.light_rig`` (area lights placed and solved from the subject's measured
bounds) and a soft glow on the backdrop, under the built-in studio environment, seen
through a 70 mm lens at f/4 (real depth of field: the path tracers sample the aperture).
It writes:

* ``hero.png`` (1920x1080, GPU path tracer, Sobol', Blackman-Harris, adaptive sampling,
  denoised, film view transform) and its committed ``hero_preview.png``;
* ``presets_sheet.png``: the same set under six presets (three_point, rembrandt,
  butterfly, high_key, low_key, silhouette) with each rig's solved lighting ratio, lit by
  the rig alone (the environment dimmed to 5% and the backdrop glow off, so each setup's
  shadows show);
* a 2.5-second 60 fps orbit with a rack focus (``orbit_1080p.mp4``, ``orbit_master.mkv``,
  the committed ``orbit_preview.mp4``), its camera checked by ``motion_report``;
* ``report.json``: everything measured.

    .venv/bin/python examples/film_e_studio.py [--quick] [--no-video]
        [--part all|hero|presets|orbit]
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

from film_common import (  # noqa: E402
    OUT,
    arguments,
    film_options,
    preview,
    sequence,
    sheet,
    still,
    write_report,
)
from rigging_common import cyclorama  # noqa: E402

from aether3d import Scene  # noqa: E402

PRESETS = ("three_point", "rembrandt", "butterfly", "high_key", "low_key", "silhouette")


def build(quick: bool) -> Scene:
    scene = Scene("Studio")
    backdrop = cyclorama(scene, width=14.0, depth=8.0, height=6.0, radius=1.6, back=2.2)
    backdrop.material.configure(base_color="#8c9096", roughness=0.9)
    seg = 24 if quick else 96
    plinth = scene.add_procedural("Plinth", "box", width=1.4, depth=0.8, height=0.5,
                                  position=(0, 0, 0.25))  # fmt: skip
    plinth.geometry.add_modifier("bevel", width=0.012, segments=3, limit="all")
    plinth.material = scene.create_material("Plinth", base_color="#e8e4dc", roughness=0.6)
    products = scene.add_object("Products", position=(0, 0, 0.5))
    vase = scene.add_procedural("Vase", "cylinder", radius=0.12, depth=0.36, segments=seg,
                                caps=True, smooth=True, parent=products,
                                position=(-0.32, 0.08, 0.18))  # fmt: skip
    hollow = scene.add_geometry("Vase inside", "cylinder", radius=0.112, depth=0.36,
                                segments=seg, caps=True, smooth=True)  # fmt: skip
    vase.geometry.add_modifier("boolean", operation="difference",
                               operand={"geometry": hollow.name},
                               operand_translation=[0.0, 0.0, 0.012])  # fmt: skip
    vase.material = scene.create_material("Glass", preset="glass", color="#f2fbf8")
    ball = scene.add_procedural("Chrome", "uv_sphere", radius=0.1, segments=seg,
                                rings=seg // 2, parent=products,
                                position=(0.08, -0.06, 0.16))  # fmt: skip
    ball.material = scene.create_material("Chrome", model="standard", base_color="#e9ecef",
                                          metallic=1.0, roughness=0.08)  # fmt: skip
    riser = scene.add_procedural("Riser", "box", width=0.22, depth=0.22, height=0.06,
                                 parent=products, position=(0.08, -0.06, 0.03))  # fmt: skip
    riser.geometry.add_modifier("bevel", width=0.006, segments=2, limit="all")
    riser.material = scene.create_material("Wood", preset="wood", color="#9b6a3c", scale=0.2)
    ring = scene.add_procedural("Gold", "torus", major_radius=0.08, minor_radius=0.026,
                                major_segments=seg, minor_segments=seg // 3, parent=products,
                                position=(0.26, -0.24, 0.026))  # fmt: skip
    ring.material = scene.create_material("Gold", model="standard", base_color="#e2b04a",
                                          metallic=1.0, roughness=0.22)  # fmt: skip
    cube = scene.add_procedural("Frost", "box", width=0.13, depth=0.13, height=0.13,
                                parent=products, position=(0.46, 0.06, 0.065))  # fmt: skip
    cube.rotation_euler = (0.0, 0.0, 25.0)
    cube.geometry.add_modifier("bevel", width=0.008, segments=3, limit="all")
    cube.material = scene.create_material("Frosted", preset="frosted_glass", color="#dff0ff")
    cam = scene.add_camera("Camera", position=(0.12, -1.75, 1.02), look_at=(0.04, 0.0, 0.66),
                           focal_length=70.0, fstop=4.0, focus_object=vase.id)  # fmt: skip
    scene.active_camera = cam
    scene.world.environment.configure(type="image", image="builtin:studio", intensity=0.5,
                                      visible=False)  # fmt: skip
    scene.world.background = "#1a1b1e"
    scene.world.ambient_intensity = 0.0
    # A soft light on the backdrop behind the set: the product shot's background glow.
    scene.add_area_light("Backdrop glow", position=(0.0, 0.6, 0.35), look_at=(0.0, 2.2, 1.4),
                         shape="ellipse", size=1.6, size_y=0.9, intensity=6.0,
                         temperature=6200.0, use_temperature=True)  # fmt: skip
    scene.light_rig("Products", "three_point", rim=1.4, level=1.35, temperature=5600.0)
    return scene


def orbit(scene: Scene) -> None:
    """A slow 30-degree orbit with a rack focus from the vase to the frosted cube."""
    cam = scene.get_by_name("Camera")
    anim = scene.animation
    keys_x, keys_y, keys_z = [], [], []
    for t, a in ((0.0, -18.0), (1.25, -3.0), (2.5, 12.0)):
        r = math.radians(a)
        keys_x.append((t, 1.75 * math.sin(r)))
        keys_y.append((t, -1.75 * math.cos(r)))
        keys_z.append((t, 1.02 - 0.08 * (t / 2.5)))
    for path, keys in (("transform.position.x", keys_x), ("transform.position.y", keys_y),
                       ("transform.position.z", keys_z)):  # fmt: skip
        anim.set_channel(cam, path, keys)
    anim.set_constraints(cam, [{"name": "Aim", "type": "look_at",
                                "target": {"object": scene.get_by_name("Products").id},
                                "target_offset": [0.02, 0.0, 0.12]}])  # fmt: skip
    cam.camera.focus_object = None
    aim = np.array([0.02, 0.0, 0.62])  # Products' origin + the look-at offset

    def depth(name: str, t: float) -> float:
        """The object's centre along the camera's view axis at ``t`` (the focus plane)."""
        eye = np.array([anim.sample(cam, f"transform.position.{a}", [t])[0] for a in "xyz"])
        forward = (aim - eye) / np.linalg.norm(aim - eye)
        return float(np.dot(np.asarray(scene.get_by_name(name).world_matrix[:3, 3]) - eye,
                            forward))  # fmt: skip

    holds = [(t, depth("Vase", t)) for t in np.linspace(0.0, 0.9, 7)]
    holds += [(t, depth("Frost", t)) for t in np.linspace(1.9, 2.5, 5)]
    anim.set_channel(cam, "camera.focus_distance", [(float(t), d) for t, d in holds])
    anim.set_timeline(fps=60, start=0.0, end=2.5)


def presets(scene: Scene, out: Path, quick: bool) -> dict:
    """The six presets, each lit by its rig alone: the environment dimmed, the glow off."""
    env = scene.world.environment
    glow = scene.get_by_name("Backdrop glow").light
    saved = (env.intensity, glow.intensity)
    env.intensity, glow.intensity = 0.05 * saved[0], 0.0
    images, labels, rigs = [], [], {}
    try:
        for name in PRESETS:
            rig = scene.light_rig("Products", name, rim=1.4, level=1.35, temperature=5600.0)
            rigs[name] = {
                "ratio": rig.get("ratio"),
                "lights": [{k: light[k] for k in ("name", "role", "azimuth", "elevation",
                                                   "intensity")} for light in rig["lights"]],
                "removed": rig["removed"],
            }  # fmt: skip
            opts = film_options(quick, width=640, height=360, samples=128, adaptive=0.03)
            shot = still(scene, out / "presets" / f"{name}.png", opts)
            images.append(shot["image"])
            ratio = f" {rig['ratio']:.0f}:1" if rig.get("ratio") else ""
            labels.append(f"{name}{ratio}")
    finally:
        env.intensity, glow.intensity = saved
        scene.light_rig("Products", "three_point", rim=1.4, level=1.35, temperature=5600.0)
    return {
        "presets": rigs,
        "presets_lighting": "each rig alone: the environment at 5% and the backdrop glow off",
        "presets_sheet": sheet(images, labels, out / "presets_sheet.png"),
    }


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "e_studio"
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    scene = build(args.quick)
    report_path = out / "report.json"
    report: dict = {"showcase": "E studio lighting", "quick": args.quick}
    if args.part != "all" and report_path.exists():  # (one part again: keep the others)
        report.update(json.loads(report_path.read_text()))
        report.pop("machine", None)
    if args.part in ("all", "hero"):
        hero_opts = film_options(args.quick, samples=512, adaptive=0.015)
        hero = still(scene, out / "hero.png", hero_opts)
        report["hero"] = {k: v for k, v in hero.items() if k != "image"}
        report["hero_preview"] = preview(hero["image"], out / "hero_preview.png")
    if args.part in ("all", "presets"):
        report.update(presets(scene, out, args.quick))
    if not args.no_video and args.part in ("all", "orbit"):
        orbit(scene)
        # The camera, measured as it moves: framing and steadiness on the whole set, and
        # focus on what each hold of the rack focus is meant to show.
        quality = {}
        for label, subject, start, end in (("set", "Products", 0.0, 2.5),
                                           ("vase_hold", "Vase", 0.0, 0.9),
                                           ("cube_hold", "Frost", 1.9, 2.5)):  # fmt: skip
            got = scene.animation.motion_report(subject=subject, start=start, end=end, rate=240)
            codes = [f["code"] for f in got.get("findings", [])]
            if label == "set":  # (the set's centre is between the two focus planes)
                codes = [c for c in codes if c != "FOCUS_ERROR"]
            quality[label] = codes
        report["camera_quality"] = quality
        frame_opts = film_options(args.quick, samples=64, adaptive=0.03)
        if args.quick:
            report["orbit"] = sequence(scene, out, "orbit", start=0.0, end=2 / 60,
                                       options=frame_opts, quick=True)  # fmt: skip
        else:
            report["orbit"] = sequence(scene, out, "orbit", start=0.0, end=2.5,
                                       options=frame_opts, quick=False)  # fmt: skip
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    write_report(report_path, report)
    return report


if __name__ == "__main__":
    data = main()
    print(f"E studio: {data['seconds']} s -> {OUT / 'e_studio'}")
