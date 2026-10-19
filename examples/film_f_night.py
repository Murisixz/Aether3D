"""Showcase F -- a night street: light that comes from the scene's own sources.

A brick facade with lit windows, three street lamps, a wet patch of cobbles, a bench and
a cool moon under a dark sky. Everything that glows also lights: the windows are emissive
meshes the path tracers sample as lights (their light reaches the cobbles and the bench),
each lamp is a spot light at 2400 K inside an emissive lamp head, the moon a small cool
sun at 7500 K. One lamp flickers -- its intensity curve carries a noise modifier -- and
the camera trucks slowly past the bench with a 35 mm lens at f/2.8. It writes:

* ``hero.png`` (1920x1080, GPU path tracer, Sobol', Blackman-Harris, adaptive sampling,
  denoised, film view transform) and its committed ``hero_preview.png``;
* ``emissive_proof.png``: the same view with the lamps' spot lights and the moon switched
  off -- only the emissive windows and lamp heads light the street, measured as the
  cobbles' mean luminance with and without them (``report.json`` ``emissive_proof``);
* a 2-second 60 fps truck with the flicker (48 samples per pixel at most, adaptive):
  ``truck_1080p.mp4``, ``truck_master.mkv``, the committed ``truck_preview.mp4``;
* ``report.json``: everything measured.

    .venv/bin/python examples/film_f_night.py [--quick] [--no-video]
        [--part all|hero|proof|truck]
"""

from __future__ import annotations

import json
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
    still,
    write_report,
)

from aether3d import Scene  # noqa: E402

LAMPS = ((-3.6, -1.6), (1.2, -1.6), (6.0, -1.6))
WARM = "#ff9a46"
GLOWS = (0.55, 0.8, 1.05)  # the lit windows' emissive strengths (each room differs)


def lamp(scene: Scene, name: str, x: float, y: float, seg: int, metal, head_glow) -> None:
    """A pole, an arm and a lamp head: an emissive glass disc with a spot light just
    below it (aimed down, soft-edged, small radius for soft shadows)."""
    root = scene.add_object(name, position=(x, y, 0.0))
    pole = scene.add_procedural(f"{name} pole", "cylinder", radius=0.055, depth=3.3,
                                segments=seg, parent=root, position=(0, 0, 1.65))  # fmt: skip
    arm = scene.add_procedural(f"{name} arm", "box", width=0.06, depth=0.55, height=0.06,
                               parent=root, position=(0, 0.25, 3.25))  # fmt: skip
    head = scene.add_procedural(f"{name} head", "cone", radius_bottom=0.24, radius_top=0.08,
                                depth=0.22, segments=seg, parent=root,
                                position=(0, 0.5, 3.18))  # fmt: skip
    for part in (pole, arm, head):
        part.material = metal
    glow = scene.add_procedural(f"{name} glass", "cylinder", radius=0.2, depth=0.02,
                                segments=seg, parent=root, position=(0, 0.5, 3.06))  # fmt: skip
    glow.material = head_glow
    spot = scene.add_spot_light(f"{name} light", position=(x, y + 0.5, 3.0),
                                direction=(0.0, 0.0, -1.0), intensity=9.0, spot_size=110.0,
                                spot_blend=0.6, radius=0.06, temperature=2400.0,
                                use_temperature=True)  # fmt: skip
    spot.set_parent(root, keep_world_transform=True)


def build(quick: bool) -> Scene:
    scene = Scene("Night street")
    seg = 16 if quick else 48
    ground = scene.add_plane("Cobbles", size=30.0)
    ground.material = scene.create_material("Cobbles", preset="stone", color="#6d6a66",
                                            scale=0.35)  # fmt: skip
    wet = scene.add_procedural("Wet patch", "cylinder", radius=1.6, depth=0.004, segments=seg,
                               position=(0.9, -2.6, 0.002))  # fmt: skip
    wet.scale = (1.5, 0.7, 1.0)
    wet.material = scene.create_material("Wet stone", base_color="#34312d", roughness=0.07,
                                         specular=1.0)  # fmt: skip
    facade = scene.add_procedural("Facade", "box", width=16.0, depth=0.6, height=8.0,
                                  position=(1.0, 2.0, 4.0))  # fmt: skip
    facade.material = scene.create_material("Brick", preset="brick", color="#7a4636")
    window_lit = [scene.create_material(f"Window lit {k}", base_color="#1a120a", emissive=WARM,
                                        emissive_strength=g)
                  for k, g in enumerate(GLOWS)]  # fmt: skip
    window_dark = scene.create_material("Window dark", base_color="#0b0d10", roughness=0.08,
                                        specular=1.0)  # fmt: skip
    frame = scene.create_material("Window frame", base_color="#2b2622", roughness=0.6)
    lit = {(0, 1), (0, 3), (0, 4), (1, 0), (1, 2), (1, 5), (0, 6)}  # (row, column)
    for row, z in enumerate((2.0, 5.0)):
        for col in range(7):
            x = -6.2 + 2.2 * col
            pane = scene.add_procedural(f"Window {row}{col}", "box", width=1.1, depth=0.04,
                                        height=1.5, position=(x, 1.68, z))  # fmt: skip
            glow = window_lit[(3 * row + 5 * col) % len(GLOWS)]
            pane.material = glow if (row, col) in lit else window_dark
            sill = scene.add_procedural(f"Sill {row}{col}", "box", width=1.3, depth=0.16,
                                        height=0.08, position=(x, 1.64, z - 0.8))  # fmt: skip
            sill.material = frame
            for k, (w, h, dz) in enumerate(((0.05, 1.5, 0.0), (1.1, 0.05, 0.25))):  # mullions
                bar = scene.add_procedural(f"Mullion {row}{col}{k}", "box", width=w, depth=0.05,
                                           height=h, position=(x, 1.645, z + dz))  # fmt: skip
                bar.material = frame
    door = scene.add_procedural("Door", "box", width=1.4, depth=0.06, height=2.4,
                                position=(9.0, 1.67, 1.2))  # fmt: skip
    door.material = scene.create_material("Door", preset="wood", color="#3b2416", scale=0.25)
    metal = scene.create_material("Lamp metal", preset="metal", color="#2e3034")
    head_glow = scene.create_material("Lamp glass", base_color="#2a2118", emissive="#ffc890",
                                      emissive_strength=6.0)  # fmt: skip
    for k, (x, y) in enumerate(LAMPS):
        lamp(scene, f"Lamp {k + 1}", x, y, seg, metal, head_glow)
    wood = scene.create_material("Bench wood", preset="wood", color="#7b5132", scale=0.2)
    bench = scene.add_object("Bench", position=(1.9, -0.6, 0.0))
    for k in range(3):  # slats
        slat = scene.add_procedural(f"Slat {k}", "box", width=1.8, depth=0.12, height=0.04,
                                    parent=bench, position=(0, -0.16 + 0.16 * k, 0.46))  # fmt: skip
        slat.material = wood
    back = scene.add_procedural("Backrest", "box", width=1.8, depth=0.04, height=0.32,
                                parent=bench, position=(0, 0.2, 0.75))  # fmt: skip
    back.material = wood
    for sx in (-0.8, 0.8):
        leg = scene.add_procedural(f"Leg {sx:+.0f}", "box", width=0.06, depth=0.44, height=0.46,
                                   parent=bench, position=(sx, 0.0, 0.23))  # fmt: skip
        leg.material = metal
    moon = scene.add_directional_light("Moon", direction=(0.5, 0.75, -0.45), intensity=0.18,
                                       angle=0.6, temperature=7500.0,
                                       use_temperature=True)  # fmt: skip
    moon.tags = ("moon",)
    scene.world.environment.configure(type="sky", intensity=0.35, zenith_color="#04070f",
                                      horizon_color="#141b2c",
                                      ground_color="#050505")  # fmt: skip
    scene.world.background = "#04070f"
    scene.world.ambient_intensity = 0.0
    cam = scene.add_camera("Camera", position=(-2.0, -8.5, 1.55), look_at=(1.6, 0.4, 1.4),
                           focal_length=35.0, fstop=2.8,
                           focus_object=bench.id)  # fmt: skip
    scene.active_camera = cam
    return scene


def animate(scene: Scene) -> None:
    """The camera trucks right past the bench; lamp 2 flickers (noise on its intensity)."""
    anim = scene.animation
    cam = scene.get_by_name("Camera")
    anim.set_channel(cam, "transform.position.x", [(0.0, -2.0), (2.0, -0.8)],
                     handles="auto_clamped")  # fmt: skip
    anim.set_channel(cam, "transform.position.y", [(0.0, -8.5), (2.0, -8.2)])
    anim.set_constraints(cam, [{"name": "Aim", "type": "look_at",
                                "target": {"object": scene.get_by_name("Bench").id},
                                "target_offset": [-0.3, 1.0, 1.4]}])  # fmt: skip
    flicker = scene.get_by_name("Lamp 2 light")
    anim.set_channel(flicker, "light.intensity", [(0.0, 9.0), (2.0, 9.0)])
    anim.set_modifiers(flicker, "light.intensity", [
        {"type": "noise", "strength": 3.0, "scale": 0.09, "octaves": 2, "seed": 7},
    ])  # fmt: skip
    anim.set_timeline(fps=60, start=0.0, end=2.0)


def proof(scene: Scene, out: Path, quick: bool) -> dict:
    """Only the emissive meshes light the street: the spots and the moon off, the sky dark.
    The cobbles' mean luminance with the glowing meshes against with them black."""
    lights = [o for o in scene if o.light is not None]  # (hidden: one has animated intensity)
    env = scene.world.environment
    env_saved = env.intensity
    mats = [m for m in scene.materials.materials if m.name.startswith(("Window lit", "Lamp glass"))]
    glow_saved = [m.emissive_strength for m in mats]
    results = {}
    try:
        for o in lights:
            o.visible = False
        env.intensity = 0.0
        for label, factor in (("emissive_on", 1.0), ("emissive_off", 0.0)):
            for m, s in zip(mats, glow_saved, strict=True):
                m.emissive_strength = s * factor
            opts = film_options(quick, width=960, height=540, samples=256, adaptive=0.03,
                                denoise=False)  # fmt: skip
            shot = scene.render(**{**opts, "outputs": ("image", "linear", "object_index")},
                                default_lights=False)  # fmt: skip
            ground = scene.get_by_name("Cobbles")
            rows = [o["index"] for o in shot.objects if o["id"] == ground.id]
            mask = np.isin(shot.object_index, rows)
            lin = shot.linear.astype(np.float64)
            lum = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
            results[label] = {"cobbles_mean_luminance": float(lum[mask].mean()),
                              "cobbles_pixels": int(mask.sum())}  # fmt: skip
            if factor:
                from aether3d.rendering.png import write_png

                write_png(out / "emissive_proof.png", shot.image)
    finally:
        for o in lights:
            o.visible = True
        env.intensity = env_saved
        for m, s in zip(mats, glow_saved, strict=True):
            m.emissive_strength = s
    on = results["emissive_on"]["cobbles_mean_luminance"]
    off = results["emissive_off"]["cobbles_mean_luminance"]
    results["ratio"] = on / off if off > 0 else None
    results["lights_by_emission"] = on > 0.0 and off == 0.0
    results["note"] = ("every light and the sky off; the windows and lamp glass emit: the "
                       "cobbles are lit only by them (path-traced emissive sampling)")  # fmt: skip
    return results


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "f_night"
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    scene = build(args.quick)
    animate(scene)
    report_path = out / "report.json"
    report: dict = {"showcase": "F night street", "quick": args.quick}
    if args.part != "all" and report_path.exists():
        report.update(json.loads(report_path.read_text()))
        report.pop("machine", None)
    if args.part in ("all", "hero"):
        hero = still(scene, out / "hero.png",
                     film_options(args.quick, samples=384, adaptive=0.015),
                     time=1.0)  # fmt: skip
        report["hero"] = {k: v for k, v in hero.items() if k != "image"}
        report["hero_preview"] = preview(hero["image"], out / "hero_preview.png")
    if args.part in ("all", "proof"):
        report["emissive_proof"] = proof(scene, out, args.quick)
    if not args.no_video and args.part in ("all", "truck"):
        quality = scene.animation.motion_report(subject="Bench", rate=240)
        report["camera_quality"] = [f["code"] for f in quality.get("findings", [])]
        flick = scene.animation.sample(scene.get_by_name("Lamp 2 light"), "light.intensity",
                                       np.linspace(0, 2, 9))  # fmt: skip
        report["flicker_samples"] = [round(float(v), 3) for v in np.ravel(flick)]
        opts = film_options(args.quick, samples=48, adaptive=0.04)
        end = 2 / 60 if args.quick else 2.0
        report["truck"] = sequence(scene, out, "truck", start=0.0, end=end, options=opts,
                                   quick=args.quick)  # fmt: skip
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    write_report(report_path, report)
    return report


if __name__ == "__main__":
    data = main()
    print(f"F night: {data['seconds']} s -> {OUT / 'f_night'}")
