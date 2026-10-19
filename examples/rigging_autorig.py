"""Showcase A (Stage 18, revised): an unrigged figure, rigged automatically, checked, and
animated -- a walk, a run, a crouch, a reach and a rise onto tiptoe, with elbows, knees,
shoulders, feet and fingers bending visibly.

The body begins as a plain mesh: the anatomical figure of ``aether3d.sculpt.figure`` (an
original procedural body with muscles, knees, toes, hands with fingers and a face; no
third-party asset), meshed by marching cubes and remeshed to even triangles -- no joints,
no weights. ``detect_landmarks`` measures it (fingers included), ``autorig`` builds the
skeleton (about 60 joints: a four-joint torso, clavicles, split twist bones, pelvis bones,
three joints per finger; swing-twist limits, IK targets, poles and a gaze) and generates
bone heat weights; ``diagnose_rig`` checks it over the diagnostic pose suite. The
clothes' edges (waist, cuffs, collar, shoe tops, hairline) are cut into the bare mesh from
its landmarks before rigging (``tailor``), so the clothes meet along clean lines
(``dress``). Then:

* **walk** and **run** -- the motion generators (planted feet: their sliding is measured);
* **crouch** -- ``pose_character`` keys a crouch with the feet planted by IK and back;
* **reach** -- the right hand reaches a lamp by IK (``reach``), the head looks at it, and
  ``ik_fk`` switches the arm back to FK with a matched pose before it lowers;
* **rise** -- up on tiptoe (``roll_foot``: the feet roll on their balls, then their toes),
  both arms raised high in a V (``raise_arm``: the clavicles lift with them) and the hands
  closing into fists (``curl_fingers``), then down again.

Run from the repository root:

    python examples/rigging_autorig.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--no-video] [--fps 60] [--samples 256]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/a_autorig``): ``landmarks.png``
(the measured landmarks on the bare mesh), ``skeleton.png``/``skeleton.svg``,
``weights_*.png`` (weight heatmaps), ``hero_*.png`` (path-traced stills, 1920x1080; their
``*_preview.png``), ``motion_sheet.png``, ``frames/`` (every frame, not committed),
``autorig_1080p.mp4`` (libx264 slow CRF 10) with ``autorig_master.mkv`` (lossless) and
``autorig_preview.mp4``, ``autorig.aether`` and ``report.json``.
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

FEET = ["Foot.L", "Foot.R"]


def build(quick: bool) -> tuple[Scene, dict]:
    scene = Scene("Showcase A")
    lighting = C.studio(scene)
    body = scene.add_object("Body", mesh=C.character_mesh(quick))
    a = scene.animation
    before = {
        "rigs": sum(1 for o in scene.objects if a.rig(o) is not None),
        "skins": sum(1 for k, _ in a._owned if k == "skin"),
        "vertices": body.mesh.vertex_count,
        "triangles": body.mesh.triangle_count,
    }
    return scene, {"lighting": lighting, "unrigged": before}


def landmark_image(scene: Scene, body, out: Path, size, report: dict) -> None:
    """The landmarks the detector measured, drawn over the bare body (projected by the
    render's camera; colour by confidence: green measured, amber weaker, red a prior)."""
    a = scene.animation
    t0 = time.perf_counter()
    det = a.detect_landmarks(body)
    report["landmarks"] = {
        "seconds": round(time.perf_counter() - t0, 3),
        "confidence": det["confidence"],
        "pose": det.get("pose"),
        "frame": det["frame"],
        "points": {
            n: {"position": lm["position"], "confidence": lm["confidence"], "method": lm["method"]}
            for n, lm in det["landmarks"].items()
        },
        "diagnostics": [d["code"] for d in det["diagnostics"]],
    }
    cam = scene.add_camera("Landmarks", position=(0.0, -3.4, 1.0), look_at=(0.0, 0.0, 0.92), fov=36)
    w, h = size[0] // 2, size[1]
    r = scene.render(camera=cam, **C.render_options(width=w, height=h))
    img = np.array(r.image, copy=True)
    names = list(det["landmarks"])
    pts = scene.project(
        [tuple(det["landmarks"][n]["position"]) for n in names], camera=cam, width=w, height=h
    )
    radius = max(3, w // 160)
    for n, sp in zip(names, pts, strict=True):
        if sp.x is None:
            continue
        conf = det["landmarks"][n]["confidence"]
        color = (60, 220, 90) if conf >= 0.7 else (250, 180, 40) if conf >= 0.5 else (240, 60, 50)
        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - sp.x) ** 2 + (yy - sp.y) ** 2
        img[d2 <= (radius + 2) ** 2] = (20, 20, 24)
        img[d2 <= radius**2] = color
    C.save(out / "landmarks.png", img)
    scene.remove(cam)


def rig_it(scene: Scene, body, out: Path, size, report: dict):
    a = scene.animation
    edges = C.tailor(scene, body)  # (the clothes' edges, cut before rigging)
    rep = a.autorig(body, name="Hero Rig")
    arm = scene.get_by_name("Hero Rig")
    report["autorig"] = {
        "joints": rep["joints"],
        "confidence": rep["confidence"],
        "controls": rep.get("controls"),
        "constraints": rep.get("constraints"),
        "timings": rep["timings"],
        "weights": rep["weights"],
        "diagnostics": [d["code"] for d in rep["diagnostics"]],
    }
    report["clothes_faces"] = C.dress(scene, body, arm, edges)
    report["clothes_cut_vertices"] = edges["vertices_added"]
    diag = a.diagnose_rig(arm, animation=False)
    report["checks"] = {
        "summary": diag["summary"],
        "codes": sorted({f["code"] for f in diag["findings"]}),
        "errors": [f["message"] for f in diag["findings"] if f["severity"] == "error"],
        "joint_ring": {
            pose: {m: round(r.get("joint_ring", 1.0), 3) for m, r in meshes.items()}
            for pose, meshes in diag["poses"].items()
        },
    }
    prev = a.rig_preview(arm, mode="skeleton", view="front", width=size[0] // 2, height=size[1])
    C.save(out / "skeleton.png", prev["image"])
    (out / "skeleton.svg").write_text(prev["svg"], encoding="utf-8")
    for joint in ("UpperArm.L", "Thigh.R", "ForeArmTwist.L"):
        w = a.rig_preview(
            arm,
            mode="weights",
            joint=joint,
            view="three_quarter",
            width=size[0] // 2,
            height=size[1],
        )
        C.save(out / f"weights_{joint.replace('.', '_')}.png", w["image"])
    return arm


def choreograph(scene: Scene, arm, report: dict) -> list[dict]:
    """Four segments, each its own clip (the frames of each start at 0)."""
    a = scene.animation
    segments = []
    # Walk and run cross the studio from left to right (the armature turned 90 degrees:
    # its travel follows; the camera tracks it from the front).
    arm.rotation_euler = (0.0, 0.0, 90.0)
    arm.position = (-3.2, 0.0, 0.0)
    walk = a.generate(arm, "walk", name="Walk", forward="-y", bind=False)
    run = a.generate(arm, "run", name="Run", forward="-y", bind=False)
    segments.append({"name": "walk", "clip": "Walk", "duration": 3.0, "facts": walk})
    segments.append({"name": "run", "clip": "Run", "duration": 2.4, "facts": run})
    # Crouch: an idle base; the feet planted by IK, the hips keyed down and up.
    idle = a.generate(arm, "idle", name="Crouch", forward="-y", bind=False)
    segments.append({"name": "crouch", "clip": "Crouch", "duration": 2.8, "facts": idle})
    # Reach: an idle base; the right hand reaches the lamp, then back to FK and down.
    a.generate(arm, "idle", name="Reach", forward="-y", bind=False)
    segments.append({"name": "reach", "clip": "Reach", "duration": 3.4, "facts": None})
    # Rise: an idle base; tiptoe, arms up, fists, and back.
    a.generate(arm, "idle", name="Rise", forward="-y", bind=False)
    segments.append({"name": "rise", "clip": "Rise", "duration": 3.4, "facts": None})
    report["motions"] = {
        s["name"]: {
            k: s["facts"][k]
            for k in ("keys", "duration", "speed", "period", "travel", "ik_residual")
            if s["facts"] and k in s["facts"]
        }
        for s in segments
    }
    return segments


def play(scene: Scene, arm, seg: dict, report: dict) -> None:
    """Bind a segment's clip (and key what it adds), placed for the camera."""
    a = scene.animation
    a.set_action(arm, seg["clip"])
    for prop in ("Lamp", "Lamp stand"):  # (only the reach needs the lamp)
        scene.get_by_name(prop).visible = seg["name"] == "reach"
    if seg["name"] in ("walk", "run"):
        arm.rotation_euler = (0.0, 0.0, 90.0)
        arm.position = (-3.2 if seg["name"] == "walk" else -4.2, 0.0, 0.0)
        return
    # The crouch in three-quarter profile (the knees read), the reach facing the camera.
    arm.rotation_euler = (0.0, 0.0, {"crouch": 65.0, "rise": 20.0}.get(seg["name"], 30.0))
    arm.position = (0.0, 0.0, 0.0)
    if seg["name"] == "rise" and not seg.get("keyed"):
        # (the idle breathes in the torso; the arms, hands and feet are the keys' alone)
        limbs = ("Shoulder.", "UpperArm", "ForeArm", "Hand.", "Thumb", "Index", "Middle",
                 "Ring", "Pinky", "Foot.", "Toe.")  # fmt: skip
        for path, _ in list(a.clip("Rise").data.channels):
            if any(f'joints["{n}' in path for n in limbs):
                a.set_channel(arm, path, None)
        # (time, arms, feet, fists)
        keys = [(0.0, -25.0, 0.0, 0.0), (0.5, -25.0, 0.0, 0.0), (1.5, 140.0, 50.0, 1.0),
                (2.3, 140.0, 50.0, 1.0), (3.3, -25.0, 0.0, 0.0)]  # fmt: skip
        out = []
        fingers = a.rig(arm).role_index("index.1.L") is not None  # (quick: no fingers)
        for t, arms, feet, fist in keys:
            cmds = []
            for side in ("L", "R"):
                cmds += [
                    {"do": "raise_arm", "side": side, "angle": arms, "direction": "side"},
                    {"do": "roll_foot", "side": side, "angle": feet},
                ]
                if fingers:
                    cmds.append({"do": "curl_fingers", "side": side, "shape": "fist",
                                 "amount": fist})  # fmt: skip
            out.append(a.pose_character(arm, cmds, time=t))
        top = out[2]["commands"]  # (the peak)
        report["rise"] = {
            "keys": [
                {"time": t, "arms_deg": w, "feet_deg": f, "fists": bool(k)} for t, w, f, k in keys
            ],  # fmt: skip
            "clavicle_deg": top[0].get("clavicle_angle"),
            "foot_pivot": top[1].get("pivot"),
            "finger_joints": sum(
                len(c.get("joints", [])) for c in top if c["do"] == "curl_fingers"
            ),
            "leg_residual_m": max(out[2]["residuals"].get(f"leg_ik.{s}", 0.0) for s in "LR"),
        }
        seg["keyed"] = True
    if seg["name"] == "crouch" and not seg.get("keyed"):
        poses = [(0.5, 0.0, 0.0), (1.2, 0.4, 18.0), (1.9, 0.4, 18.0), (2.6, 0.0, 0.0)]
        out = []
        for t, depth, lean in poses:
            out.append(
                a.pose_character(
                    arm, [{"do": "crouch", "depth": depth}, {"do": "lean", "angle": lean}], time=t
                )
            )
        report["crouch"] = {
            "keys": [{"time": t, "depth": d, "lean_deg": w} for t, d, w in poses],
            # (the legs' IK: the feet stay where they were planted)
            "foot_residual_m": max(
                (v for r in out for v in r["commands"][0]["residuals"].values()), default=0.0
            ),
        }
        seg["keyed"] = True
    if seg["name"] == "reach" and not seg.get("keyed"):
        lamp = scene.get_by_name("Lamp")
        goal = list(lamp.world_position)
        # FK to IK with a matched pose, the reach, hold; IK back to FK (matched), lower.
        to_ik = a.ik_fk(arm, "arm_ik.R", "ik", 0.4)
        r1 = a.pose_character(
            arm,
            [{"do": "reach", "hand": "R", "target": goal}, {"do": "look_at", "target": goal}],
            time=1.2,
        )
        r2 = a.pose_character(arm, [{"do": "reach", "hand": "R", "target": goal}], time=2.0)
        to_fk = a.ik_fk(arm, "arm_ik.R", "fk", 2.3)
        rest = [
            {"do": "rotate", "joint": j, "euler": [0, 0, 0]}
            for j in ("Shoulder.R", "UpperArm.R", "ForeArm.R", "ForeArmTwist.R", "Hand.R")
        ]
        a.pose_character(arm, rest, time=3.3)
        report["reach"] = {
            "goal": [round(v, 4) for v in goal],
            "distance_m": [r1["commands"][0]["distance"], r2["commands"][0]["distance"]],
            "reachable": r1["commands"][0]["reachable"],
            "residual_m": [r1["residuals"].get("arm_ik.R"), r2["residuals"].get("arm_ik.R")],
            "fk_to_ik": {k: to_ik[k] for k in ("mismatch", "rotation_mismatch", "time")},
            "ik_to_fk": {k: to_fk[k] for k in ("mismatch", "rotation_mismatch", "time")},
        }
        seg["keyed"] = True


def track(scene: Scene, arm, seg: dict, cam) -> None:
    """The camera: tracking the walk and run from the front, framing the crouch and reach."""
    a = scene.animation
    a.set_action(cam, None)
    if seg["name"] in ("walk", "run"):
        a.follow_camera(
            cam,
            arm,
            offset=(0.4, -4.6, 0.55),
            lag=0.25,
            aim=(0.0, 0.0, 0.95),
            start=0.0,
            end=seg["duration"],
        )
    else:
        placed = {"reach": ((1.2, -3.5, 1.3), (-0.1, -0.2, 1.05)),
                  "crouch": ((2.0, -3.9, 1.0), (0.0, 0.0, 0.62)),
                  "rise": ((0.9, -4.4, 1.25), (0.0, 0.0, 1.05))}[seg["name"]]  # fmt: skip
        cam.position = placed[0]
        cam.look_at(placed[1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "a_autorig"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = C.set_backend(args.backend)
    size = (320, 180) if args.quick else (1920, 1080)
    fps = 12 if args.quick else args.fps
    t_all = time.perf_counter()
    scene, report = build(args.quick)
    report["backend"] = backend
    report["machine"] = C.machine()
    body = scene.get_by_name("Body")
    print("1. landmarks on the unrigged body")
    landmark_image(scene, body, out, size, report)
    print("2. auto-rig, weights, clothes, checks")
    arm = rig_it(scene, body, out, size, report)
    print(f"   {len(report['autorig']['joints'])} joints, checks {report['checks']['summary']}")
    a = scene.animation
    # The lamp stands at the character's right and in front (the character turns 30 degrees
    # for the crouch and the reach): within the right arm's reach.
    yaw = np.radians(30.0)
    right = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
    ahead = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    spot = 0.42 * right + 0.36 * ahead
    lamp = scene.add_sphere(name="Lamp", radius=0.05, position=(spot[0], spot[1], 1.55))
    lamp.material = scene.create_material(
        "Lamp glass", base_color=(1.0, 0.85, 0.5), emissive=(1.0, 0.8, 0.45), emissive_strength=2.0
    )
    stand = scene.add_cube(name="Lamp stand", size=1.0, position=(spot[0], spot[1], 0.75))
    stand.scale = (0.03, 0.03, 1.5)
    stand.material = scene.create_material(
        "Stand", base_color=(0.15, 0.15, 0.16), metallic=0.8, roughness=0.35
    )
    print("3. motions")
    segments = choreograph(scene, arm, report)
    cam = scene.add_camera("Camera", position=(0.0, -4.6, 1.5), look_at=(0, 0, 0.95), fov=32)
    frames_dir = out / "frames"
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
    rows = []
    first = 0
    stills = []
    sheet_images, sheet_labels = [], []
    contacts = {}
    for seg in segments:
        play(scene, arm, seg, report)
        track(scene, arm, seg, cam)
        if seg["name"] in ("walk", "run"):
            rep = a.foot_contacts(arm, FEET, start=0.0, end=seg["duration"], rate=120)
            contacts[seg["name"]] = {
                f: {
                    "sliding_m": round(d["sliding_max"], 5),
                    "penetration_m": round(d["penetration"], 5),
                    "contacts": len(d["contacts"]),
                }
                for f, d in rep["feet"].items()
            }
        duration = seg["duration"] if not args.quick else min(seg["duration"], 1.0)
        progress = C.progress_printer(seg["name"])
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
            progress=progress,
        )
        info["segment"] = seg["name"]
        rows.append(info)
        for k in np.linspace(0, info["frames"] - 1, 3).round().astype(int):
            sheet_images.append(frames_dir / f"frame_{first + k:05d}.png")
            sheet_labels.append(f"{seg['name']} {k / fps:.2f}s")
        first += info["frames"]
        hero_t = {"walk": 0.55, "run": 0.3, "crouch": 1.5, "reach": 1.5, "rise": 1.9}[seg["name"]]
        hero_t = min(hero_t, duration - 1.0 / fps)
        samples = 16 if args.quick else args.samples
        stills.append(
            C.still(
                scene,
                out / f"hero_{seg['name']}.png",
                camera=cam,
                time=hero_t,
                width=size[0],
                height=size[1],
                path_samples=samples,
            )
        )
        C.preview(out / f"hero_{seg['name']}.png", out / f"hero_{seg['name']}_preview.png")
    report["contacts"] = contacts
    report["frames"] = {
        "segments": rows,
        "total": first,
        "fps": fps,
        "size": list(size),
        "seconds": round(sum(r["seconds"] for r in rows), 1),
    }
    report["stills"] = stills
    C.sheet(sheet_images, out / "motion_sheet.png", labels=sheet_labels, columns=4, width=480)
    if not args.no_video and not args.quick:
        print("4. encoding")
        report["video"] = C.encode(frames_dir, out, "autorig", fps)
    scene.save(out / "autorig.aether")
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print(
        f"   {first} frames at {fps} fps ({size[0]}x{size[1]}), "
        f"{report['frames']['seconds']} s of rendering"
    )
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
