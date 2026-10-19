"""Showcase D (Stage 18): a character of several independently modeled meshes -- a body,
a jacket, trousers, boots and gloves bound by weight transfer, a helmet, shoulder plates
and a backpack attached rigidly -- walking, running and turning.

The body is the anatomical figure (``aether3d.sculpt.figure``); the clothes are its parts'
shapes inflated (``figure_garment``: jacket over the torso and arms, trousers over the
hips and legs, boots, gloves), each meshed separately with its own vertices and triangles.
``autorig`` rigs the body and binds each part as asked (``parts``): the body by bone heat
weights, the clothes by **transfer** (each vertex takes the weights of the nearest point
of the body's surface, then a light smoothing; per-part distances and confidence are
reported), the hard pieces by **attachment** (rigid, to the joint they sit on: no skin,
no deformation cost). The clothing's clearance over the body is measured at rest and
over the motion (nearest-vertex distances: a numeric proxy for poke-through, not a cloth
simulation); the whole outfit is exported to GLB and checked by the Khronos validator.

Run from the repository root:

    python examples/rigging_outfit.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--no-video] [--fps 60] [--samples 192]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/d_outfit``): ``parts.png``
(the separate meshes), ``weights_jacket.png``, ``hero_*.png``, ``outfit_sheet.png``,
``frames/`` (not committed), ``outfit_1080p.mp4``/``outfit_master.mkv``/
``outfit_preview.mp4``, ``outfit.glb``, ``outfit.aether`` and ``report.json``.
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

from aether3d import Scene  # noqa: E402
from aether3d.animation.weights import nearest_points  # noqa: E402
from aether3d.rendering.overlay import animation_overlay  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402

# The figure's parts each garment covers, how much it adds (metres) and where it is cut:
# the jacket ends 5 cm below the waist, over the trousers, which start 2 cm above it.
GARMENTS = {
    "Jacket": (("torso", "upper_arm", "forearm"), 0.014, (0.42, 0.13, 0.12), 0.55,
               {"bottom": -0.05}),
    "Trousers": (("torso", "hips", "thigh", "knee", "shin"), 0.012, (0.17, 0.2, 0.27), 0.7,
                 {"top": 0.02}),
    "Boots": (("foot",), 0.010, (0.24, 0.16, 0.1), 0.4, {}),
    "Gloves": (("hand",), 0.005, (0.12, 0.12, 0.13), 0.5, {}),
}  # fmt: skip


def outfit(scene: Scene, quick: bool) -> dict[str, object]:
    body = scene.add_object("Body", mesh=C.character_mesh(quick))
    body.material = scene.create_material("Skin", base_color=(0.8, 0.62, 0.5), roughness=0.55)
    parts = {"Body": body}
    for name, (covers, by, color, rough, cut) in GARMENTS.items():
        cut = {k: C.waist() + v for k, v in cut.items()}  # (relative to the waist)
        obj = scene.add_object(name, mesh=C.garment(covers, by, quick, **cut))
        obj.material = scene.create_material(name, base_color=color, roughness=rough)
        parts[name] = obj
    metal = scene.create_material(
        "Steel", base_color=(0.72, 0.74, 0.78), metallic=0.9, roughness=0.28
    )
    canvas = scene.create_material("Canvas", base_color=(0.35, 0.33, 0.22), roughness=0.85)
    for name, mesh in C.armour(body.mesh, quick).items():
        obj = scene.add_object(name, mesh=mesh)
        obj.material = canvas if name == "Backpack" else metal
        parts[name] = obj
    return parts


def clearance(scene: Scene, body, garments, times) -> dict[str, dict]:
    """Per garment: nearest body-vertex distance of its vertices (percentiles, metres) at
    rest and over ``times`` -- how close the cloth comes to the skin."""
    out = {}
    for g in garments:
        rest_d = nearest_points(body.mesh.positions, g.mesh.positions)[1][:, 0]
        worst = []
        for t in times:
            ov = animation_overlay(scene, float(t))
            w = world_matrices(scene, ov)
            pb = (
                ov.meshes.get(body.id, body.mesh).positions @ w[body.id][:3, :3].T
                + w[body.id][:3, 3]
            )
            pg = ov.meshes.get(g.id, g.mesh).positions @ w[g.id][:3, :3].T + w[g.id][:3, 3]
            worst.append(nearest_points(pb, pg)[1][:, 0])
        d = np.concatenate(worst)
        out[g.name] = {
            "rest_p5_m": round(float(np.percentile(rest_d, 5)), 5),
            "rest_median_m": round(float(np.median(rest_d)), 5),
            "motion_p5_m": round(float(np.percentile(d, 5)), 5),
            "motion_min_m": round(float(d.min()), 5),
            "motion_median_m": round(float(np.median(d)), 5),
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "d_outfit"))
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
    scene = Scene("Showcase D")
    report: dict = {"backend": backend, "machine": C.machine(), "lighting": C.studio(scene)}
    parts = outfit(scene, args.quick)
    report["meshes"] = {
        n: {"vertices": o.mesh.vertex_count, "triangles": o.mesh.triangle_count}
        for n, o in parts.items()
    }
    cam = scene.add_camera("Camera", position=(1.6, -3.6, 1.35), look_at=(0.0, 0.0, 0.95), fov=32)
    C.still(
        scene,
        out / "parts.png",
        camera=cam,
        width=size[0] // 2,
        height=size[1] // 2,
        path_samples=16 if args.quick else 96,
    )
    print("1. rig and bind the outfit")
    a = scene.animation
    modes = {
        "Jacket": "transfer",
        "Trousers": "transfer",
        "Boots": "transfer",
        "Gloves": "transfer",
        "Helmet": "attach",
        "Shoulder plate.L": "attach",
        "Shoulder plate.R": "attach",
        "Backpack": "attach",
    }
    t0 = time.perf_counter()
    rep = a.autorig(list(parts.values()), name="Outfit Rig", parts=modes)
    arm = scene.get_by_name("Outfit Rig")
    report["autorig"] = {
        "seconds": round(time.perf_counter() - t0, 2),
        "joints": len(rep["joints"]),
        "parts": rep["weights"],
        "confidence": rep["confidence"],
    }
    rig = a.rig(arm)
    report["attachments"] = {
        scene._objects[o].name: rig.joints[rig.index_of_id(j)].name for o, j in rig.attachments
    }
    print(f"   {report['attachments']}")
    w = a.rig_preview(
        arm,
        mode="weights",
        joint="UpperArm.L",
        view="three_quarter",
        width=size[0] // 2,
        height=size[1],
    )
    C.save(out / "weights_jacket.png", w["image"])
    print("2. motion")
    a.set_timeline(start=0.0, end=7.0)
    arm.rotation_euler = (0.0, 0.0, 90.0)
    arm.position = (-2.6, 0.0, 0.0)
    walk = a.generate(arm, "walk", name="Walk", forward="-y", bind=False)
    run = a.generate(arm, "run", name="Run", forward="-y", bind=False)
    a.set_action(arm, "Walk")
    report["motions"] = {
        k: {m: v[m] for m in ("speed", "period", "keys", "ik_residual")}
        for k, v in (("walk", walk), ("run", run))
    }
    times = np.linspace(0.0, walk["period"], 13 if args.quick else 49)
    report["clearance"] = clearance(scene, parts["Body"], [parts[n] for n in GARMENTS], times)
    contacts = a.foot_contacts(arm, ["Foot.L", "Foot.R"], start=0.0, end=2 * walk["period"])
    report["contacts"] = {
        f: {"sliding_m": round(d["sliding_max"], 5), "penetration_m": round(d["penetration"], 5)}
        for f, d in contacts["feet"].items()
    }
    print(f"   clearance {report['clearance']}")
    print("3. glTF")
    glb = out / "outfit.glb"
    exp = scene.export(glb, animation="bake", allow_lossy=False).to_dict()
    report["gltf"] = {
        "bytes": exp["byte_length"],
        "skins": exp.get("skins"),
        "diagnostics": sorted({d["code"] for d in exp["diagnostics"]}),
        "validator": __import__("animation_common").validate_glb(glb),
    }
    print(f"   {report['gltf']}")
    print("4. renders")
    segments = [
        ("walk", "Walk", 3.0 if not args.quick else 0.8),
        ("run", "Run", 2.4 if not args.quick else 0.6),
    ]
    frames_dir = out / "frames"
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
    first, rows, sheet = 0, [], []
    for name, clip, duration in segments:
        a.set_action(arm, clip)
        arm.position = (-2.6 if name == "walk" else -4.0, 0.0, 0.0)
        a.set_action(cam, None)
        a.follow_camera(
            cam,
            arm,
            offset=(0.6, -4.2, 0.5),
            lag=0.25,
            aim=(0.0, 0.0, 0.95),
            start=0.0,
            end=duration,
        )
        info = C.render_frames(
            scene,
            frames_dir,
            camera=cam,
            fps=fps,
            start=0.0,
            end=duration,
            width=size[0],
            height=size[1],
            first=first,
            progress=C.progress_printer(name),
        )
        info["segment"] = name
        rows.append(info)
        for k in np.linspace(0, info["frames"] - 1, 4).round().astype(int):
            sheet.append(frames_dir / f"frame_{first + k:05d}.png")
        first += info["frames"]
        C.still(
            scene,
            out / f"hero_{name}.png",
            camera=cam,
            time=0.4,
            width=size[0],
            height=size[1],
            path_samples=16 if args.quick else args.samples,
        )
        C.preview(out / f"hero_{name}.png", out / f"hero_{name}_preview.png")
    report["frames"] = {"segments": rows, "total": first, "fps": fps, "size": list(size)}
    C.sheet(sheet, out / "outfit_sheet.png", columns=4, width=480)
    if not args.no_video and not args.quick:
        report["video"] = C.encode(frames_dir, out, "outfit", fps)
    scene.save(out / "outfit.aether")
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
