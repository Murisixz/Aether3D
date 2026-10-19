"""Showcase A -- a rigid-part character's performance, crossfaded and measured.

Kit (the original Minecraft-style cartographer of the Stage 18 showcases: six solid parts
in a plain object hierarchy, no armature, no skin) idles, waves, turns to the camera and
jumps -- motions generated as ordinary transform clips (``generate_rigid``) and layered as
strips: a base layer idling throughout, a gesture layer (the wave, the head turn) and an
action layer (the jump), each strip fading in and out with a C2 ramp
(``blend_curve: "smoother"``). Beside Kit, Blip -- a pixel cube -- hops three times,
keyed by hand with the timing tools: eased apexes (``ease``), a squash before each
take-off (``anticipate``), and a wobble after each landing (``overshoot`` with decaying
oscillations), its base staying on the ground (the cube scales about its bottom).

The camera frames Kit with the ``full`` shot preset on the left third and pushes in
(``camera_move`` dolly). Everything is measured in world space by ``motion_report``: the
performance with hard cuts between the strips (the blends set to 0) against the same
strips crossfaded. It writes:

* ``hero.png`` (1920x1080, GPU rasterizer with sun shadows) and ``hero_preview.png``;
* ``blip_graph.png`` -- Blip's height and squash curves (value, velocity, acceleration,
  jerk) with the exact analysis drawn on: velocity corners at the impacts (C1), maxima
  and minima;
* ``sheet.png`` -- six moments of the performance;
* a 5-second 60 fps sequence: ``performance_1080p.mp4``, ``performance_master.mkv`` and
  the committed ``performance_preview.mp4``;
* ``report.json``: the motion reports (cuts against crossfades), the clips, the camera.

    .venv/bin/python examples/film_a_rigid.py [--quick] [--no-video] [--part all|hero|video]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from animation_common import pixel_skin, pixel_world, skin_material  # noqa: E402
from film_common import OUT, arguments, preview, rel, sequence, sheet, write_report  # noqa: E402
from rigging_minecraft_rigid import KIT, kit  # noqa: E402

from aether3d import Scene  # noqa: E402
from aether3d.animation.graph import graph_png  # noqa: E402
from aether3d.mesh import Mesh  # noqa: E402
from aether3d.rendering.png import encode_png, write_png  # noqa: E402

DURATION = 5.0
STRIPS = (  # layer, motion, start, generator parameters
    ("Gesture", "wave", 0.6, {"waves": 3}),
    ("Gesture", "head_turn", 2.2, {"yaw": 35.0}),
    ("Action", "jump", 3.3, {"height": 0.45}),
)
BLEND = 0.35  # seconds of each fade
HOPS = ((0.4, 1.7), (1.55, 2.5), (2.7, 1.9))  # Blip's take-off times and landing x


def blip_skin() -> bytes:
    """A 16x16 slime face: green, two eyes, a mouth (an original pixel design)."""
    rng = np.random.default_rng(5)
    img = np.zeros((16, 16, 4), np.uint8)
    img[..., 3] = 255
    base = np.array([96, 186, 88]) + rng.integers(-10, 11, size=(16, 16, 1))
    img[..., :3] = np.clip(base, 0, 255)
    img[[0, 15], :, :3] = (62, 132, 58)
    img[:, [0, 15], :3] = (62, 132, 58)
    for x in (4, 11):  # eyes
        img[5:8, x - 1 : x + 1, :3] = (24, 40, 24)
        img[5, x - 1, :3] = (230, 255, 230)
    img[11, 6:10, :3] = (36, 70, 34)  # mouth
    return encode_png(img)


def blip(scene: Scene) -> object:
    """A 0.5 m cube whose origin is at the centre of its base (it squashes about it)."""
    h = 0.5
    s = h / 2
    corners = np.array([[x, y, z] for z in (0.0, h) for y in (-s, s) for x in (-s, s)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    pos, uv, nrm, tris = [], [], [], []
    middle = np.array([0.0, 0.0, s])
    for f in faces:
        base = len(pos)
        out = corners[list(f)].mean(axis=0) - middle  # (an axis: the face's outward normal)
        normal = np.round(out / np.abs(out).max())
        for k, (u, v) in zip(f, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
            pos.append(corners[k])
            uv.append((u, v))
            nrm.append(normal)
        tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    mesh = Mesh(np.array(pos, np.float32), np.array(tris, np.int64), np.array(nrm, np.float32),
                uvs=np.array(uv, np.float32))  # fmt: skip
    from animation_common import _fix_winding

    obj = scene.add_object("Blip", mesh=_fix_winding(mesh), position=(1.0, -0.4, 0.0))
    tex = scene.materials.add_image_texture("Blip skin", data=blip_skin(), embed=True,
                                            filter="nearest", mipmaps=False)  # fmt: skip
    mat = scene.create_material("Blip", roughness=0.5)
    mat.set_texture("base_color", tex)
    mat.base_color = (1.0, 1.0, 1.0)
    obj.material = mat
    return obj


def animate_blip(scene: Scene) -> dict:
    """Three hops keyed by hand, then shaped with the timing tools."""
    a = scene.animation
    cube = scene.get_by_name("Blip")
    z_keys, x_keys, s_keys = [(0.0, 0.0)], [(0.0, 1.0)], [(0.0, 1.0)]
    x_prev = 1.0
    for take_off, x_land in HOPS:
        apex, land = take_off + 0.32, take_off + 0.64
        z_keys += [(take_off, 0.0), (apex, 0.55), (land, 0.0)]
        x_keys += [(take_off, x_prev), (land, x_land)]
        s_keys += [(take_off, 1.0), (take_off + 0.12, 1.18), (apex, 1.0), (land - 0.1, 1.12),
                   (land, 1.0)]  # fmt: skip
        x_prev = x_land
    z_keys.append((DURATION, 0.0))
    x_keys.append((DURATION, x_prev))
    s_keys.append((DURATION, 1.0))
    # Contacts are corners (vector handles): the cube's base stops dead on the ground.
    corner = {True: "vector", False: "auto_clamped"}
    a.set_channel(cube, "transform.position.z",
                  [{"time": t, "value": v, "left": corner[v == 0], "right": corner[v == 0]}
                   for t, v in z_keys])  # fmt: skip
    a.set_channel(cube, "transform.position.x", x_keys, handles="auto_clamped")
    a.set_channel(cube, "transform.scale.z", s_keys, handles="auto_clamped")
    z = a.channel(cube, "transform.position.z")
    apexes = [int(k) for k in np.flatnonzero(z.values > 0.1)]
    a.edit_keys(cube, "transform.position.z", "ease", keys=apexes, ease_in=0.12, ease_out=0.12)
    s = a.channel(cube, "transform.scale.z")
    takeoffs = [int(np.flatnonzero(np.isclose(s.times, t))[0]) for t, _ in HOPS]
    for k in reversed(takeoffs):  # (later keys first: earlier indices stay put)
        a.edit_keys(cube, "transform.scale.z", "anticipate", keys=[k], amount=1.6, at=0.14)
    s = a.channel(cube, "transform.scale.z")
    landings = [int(np.flatnonzero(np.isclose(s.times, t + 0.64))[0]) for t, _ in HOPS]
    for k in reversed(landings):
        a.edit_keys(cube, "transform.scale.z", "overshoot", keys=[k], amount=2.4, settle=0.42,
                    oscillations=3, decay=0.45)  # fmt: skip
    # Width and depth keep the volume roughly: 1 / sqrt(scale z).
    s = a.channel(cube, "transform.scale.z")
    t = np.linspace(0.0, DURATION, 301)
    side = 1.0 / np.sqrt(np.maximum(s.evaluate(t), 0.2))
    for axis in ("x", "y"):
        a.set_channel(cube, f"transform.scale.{axis}", list(zip(t.tolist(), side.tolist(),
                                                               strict=True)))  # fmt: skip
    return {
        "keys": {p: len(a.channel(cube, p)) for p in
                 ("transform.position.z", "transform.position.x", "transform.scale.z")},
        "squash_min": float(s.evaluate(t).min()),
        "stretch_max": float(s.evaluate(t).max()),
    }  # fmt: skip


def perform(scene: Scene, root) -> dict:
    """Kit's motions as strips over an idle base, every part on the same layers."""
    a = scene.animation
    parts = [root, *root.iter_descendants()]
    motions = {"idle": a.generate_rigid(root, "idle", name="Idle", bind=False, travel=False)}
    for _layer, kind, _start, params in STRIPS:
        motions[kind] = a.generate_rigid(root, kind, name=kind.replace("_", " ").title(),
                                         bind=False, travel=False, **params)  # fmt: skip
    for obj in parts:
        for layer in ("Base", "Gesture", "Action"):
            a.add_layer(obj, layer)
    idle = motions["idle"]
    for obj_name, clip in idle["clips"].items():
        a.add_strip(scene.get_by_name(obj_name), "Base", clip, start=0.0,
                    repeat=DURATION / idle["duration"] + 1.0)  # fmt: skip
    for layer, kind, start, _params in STRIPS:
        for obj_name, clip in motions[kind]["clips"].items():
            a.add_strip(scene.get_by_name(obj_name), layer, clip, start=start,
                        blend_in=BLEND, blend_out=BLEND, blend_curve="smoother")  # fmt: skip
    a.set_timeline(fps=60, start=0.0, end=DURATION)
    clips = {k: {"clips": len(m["clips"]), "keys": m["keys"], "duration": m["duration"],
                 "max_error": m["max_error"]} for k, m in motions.items()}  # fmt: skip
    for k in ("takeoff", "landing"):
        clips["jump"][k] = motions["jump"][k]
    edges = []
    for _layer, kind, start, _params in STRIPS:  # where strips begin and end
        edges += [start, start + motions[kind]["duration"]]
    return clips, edges


def set_blends(scene: Scene, root, seconds: float) -> None:
    a = scene.animation
    for obj in [root, *root.iter_descendants()]:
        binding = a.binding(obj)
        for layer in [] if binding is None else binding.layers:
            if layer.name in ("Gesture", "Action"):
                for k in range(len(layer.strips)):
                    a.update_strip(obj, layer.name, k, blend_in=seconds, blend_out=seconds)


def mute_action(scene: Scene, root, mute: bool) -> None:
    a = scene.animation
    for obj in [root, *root.iter_descendants()]:
        a.update_layer(obj, "Action", mute=mute)


def quality(scene: Scene, root) -> dict:
    """World-space motion findings of Kit's parts (``motion_report`` at 240 samples per
    second), by code."""
    a = scene.animation
    parts = [o for o in root.iter_descendants()] + [root]
    rep = a.motion_report(parts, rate=240)
    codes: dict[str, int] = {}
    times = []
    for f in rep.get("findings", []):
        codes[f["code"]] = codes.get(f["code"], 0) + 1
        if f.get("time") is not None:
            times.append(round(float(f["time"]), 3))
    return {"findings": codes, "count": sum(codes.values()), "times": sorted(set(times))}


def build(quick: bool) -> tuple[Scene, object]:
    scene = Scene("Kit and Blip")
    pixel_world(scene, size=48.0)
    scene.world.environment.configure(type="sky", intensity=0.85, zenith_color="#3a6dc4",
                                      horizon_color="#cfe1f7", ground_color="#4b5a33")  # fmt: skip
    scene.world.background = "#9fb8dc"
    scene.world.ambient_color = "#c9d6ff"
    scene.world.ambient_intensity = 0.22
    scene.add_directional_light("Sun", direction=(-0.55, 0.45, -0.5), intensity=2.6, angle=1.2,
                                temperature=4300.0, use_temperature=True)  # fmt: skip
    scene.add_directional_light("Sky fill", direction=(0.6, 0.3, -0.6), intensity=0.35,
                                color="#a9c4ff")  # fmt: skip
    # The Stage 16 atmosphere: the sky drawn behind the geometry with the sun's glow and a
    # warm horizon, a thin aerial fog, hemisphere ambient light (sky above, grass below).
    scene.world.atmosphere.configure(enabled=True, sky=True, sun="Sun", sun_glow=0.6,
                                     horizon_glow=0.3, fog_density=0.004, fog_falloff=0.08,
                                     fog_color="#d4e2f4", ambient="hemisphere")  # fmt: skip
    skin = pixel_skin(seed=11, **KIT)
    parts = kit(scene, skin_material(scene, "Kit", encode_png(skin)))
    blip(scene)
    cam = scene.add_camera("Camera", position=(0.0, -7.0, 1.4), look_at=(0.0, 0.0, 1.0))
    scene.active_camera = cam
    return scene, parts["root"]


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "a_rigid"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    scene, root = build(args.quick)
    report: dict = {"showcase": "A rigid-part performance", "quick": args.quick}
    report["clips"], edges = perform(scene, root)
    report["strip_edges"] = edges
    report["blip"] = animate_blip(scene)
    a = scene.animation
    shot = a.frame_shot("Camera", root, "full", azimuth=-22.0, screen=[-0.3, -0.1], fill=0.6,
                        time=0.0)  # fmt: skip
    a.camera_move("Camera", "dolly", 0.6, 0.0, DURATION)
    report["shot"] = {k: shot[k] for k in ("preset", "distance", "focal_length") if k in shot}
    camera = a.motion_report(["Camera"], subject=root.name, rate=120)
    report["camera_quality"] = [f["code"] for f in camera.get("findings", [])]
    # The gestures' transitions (the jump's layer muted): the same strips with hard cuts,
    # then crossfaded; then the whole performance, whose jump changes velocity at its own
    # take-off and landing (its ballistic model).
    mute_action(scene, root, True)
    set_blends(scene, root, 0.0)
    report["gestures_cuts"] = quality(scene, root)
    set_blends(scene, root, BLEND)
    report["gestures_crossfades"] = quality(scene, root)
    mute_action(scene, root, False)
    report["performance_motion"] = quality(scene, root)
    jump = report["clips"]["jump"]
    start = next(s for _l, k, s, _p in STRIPS if k == "jump")
    report["jump_events"] = {"takeoff": start + jump["takeoff"], "landing": start + jump["landing"]}
    z = a.channel("Blip", "transform.position.z")
    sz = a.channel("Blip", "transform.scale.z")
    write_png(out / "blip_graph.png", graph_png(
        {"Blip height": z, "Blip squash (scale z)": sz}, start=0.0, end=3.8,
        plots=("value", "velocity", "acceleration", "jerk"), width=1200, plot_height=150,
        title="Blip: eased apexes, anticipation squashes, overshoot wobbles"))  # fmt: skip
    report["blip_graph"] = rel(out / "blip_graph.png")
    size = (320, 180) if args.quick else (1920, 1080)
    options = {"backend": "cpu" if args.quick else "gpu", "width": size[0], "height": size[1],
               "tonemap": "film"}  # fmt: skip
    if not args.quick:
        options.update(shadows=True, samples=4)
    if args.part in ("all", "hero"):
        r = scene.render(**options, time=2.6)
        write_png(out / "hero.png", r.image)
        report["hero"] = {"backend": r.backend.get("name"), "size": list(size),
                          "codes": sorted({d.code for d in r.diagnostics})}  # fmt: skip
        report["hero_preview"] = preview(r.image, out / "hero_preview.png")
        moments = (0.3, 1.0, 2.0, 2.6, 3.6, 4.5)
        small = {**options, "width": size[0] // 3, "height": size[1] // 3}
        images = [scene.render(**small, time=t).image for t in moments]
        report["sheet"] = sheet(images, [f"t = {t:.1f} s" for t in moments], out / "sheet.png")
    if not args.no_video and args.part in ("all", "video"):
        end = 2 / 60 if args.quick else DURATION
        report["performance"] = sequence(scene, out, "performance", start=0.0, end=end,
                                         options=options, quick=args.quick)  # fmt: skip
    report["bones"] = sum(1 for o in scene if a.rig(o) is not None)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    write_report(out / "report.json", report)
    return report


if __name__ == "__main__":
    data = main()
    print(
        f"A rigid: {data['seconds']} s, gestures cut {data['gestures_cuts']['findings']} -> "
        f"crossfaded {data['gestures_crossfades']['findings']}"
    )
    print(json.dumps(data.get("blip")))
