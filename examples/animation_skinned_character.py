"""Showcase 3 (Stage 17): a smooth-skinned character -- IK foot placement, an FK-to-IK
transition, pose blending, mirroring, layers, and linear vs dual-quaternion skinning.

One mannequin mesh (tubes merged into one surface, 6,330 vertices) is skinned to a
17-joint humanoid rig with weight profiles along each limb (``animation_common``). Four
demonstrations, each rendered by Aether3D and measured:

* **walk** -- a generated walk cycle (planted feet: two-bone IK on foot trajectories,
  heel and toe roll) with the sliding and penetration of each foot measured;
* **step** -- an IK constraint places the right foot on a 0.24 m step: its influence is
  keyed 0 -> 1 (the FK pose blends into the IK solution) while the hips lower; the foot's
  height above the step and the solver's residual are measured;
* **layers** -- an idle action, an upper-body layer (masked to the chest and everything
  below it in the rig) that crossfades a right-hand wave into its mirror image
  (``mirror_clip``: left and right swapped and reflected), and an additive look-around
  layer masked to the neck and head;
* **twist** -- the hand twists 160 degrees about the forearm: linear blend skinning
  collapses the wrist's blend zone (the "candy wrapper"), dual-quaternion skinning keeps
  its thickness -- measured as the zone's mean radius around the bone, rest vs twisted.

Run from the repository root:

    python examples/animation_skinned_character.py [OUTPUT_DIR] [--quick]

Writes to ``OUTPUT_DIR`` (default ``examples/output/animation/skinned``):
``{walk,step,layers}_sheet.png``, ``twist_lbs_vs_dqs.png``, ``hero_{walk,step,layers}.png``,
``mannequin.aether``, ``mannequin.glb`` (skin, rig and clips; validated when the Khronos
validator is installed) and ``report.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from animation_common import (  # noqa: E402
    backend,
    humanoid_rig,
    mannequin,
    mannequin_mesh,
    shadows,
    validate_glb,
    write_json,
)

from aether3d import Scene  # noqa: E402
from aether3d.animation import rotation as R  # noqa: E402
from aether3d.animation.frames import contact_sheet  # noqa: E402
from aether3d.rendering.overlay import animation_overlay  # noqa: E402
from aether3d.rendering.png import write_png  # noqa: E402

SKIN = (0.84, 0.71, 0.58)
FEET = ["Foot.L", "Foot.R"]


def stage(name: str) -> Scene:
    s = Scene(name)
    s.world.background = (0.78, 0.82, 0.88)
    s.world.ambient_intensity = 0.25
    floor = s.add_plane("Floor", size=14)
    floor.color = (0.58, 0.6, 0.62)
    s.add_directional_light("Key", direction=(-0.45, 0.6, -0.75), intensity=2.3)
    s.add_directional_light("Rim", direction=(0.5, -0.4, -0.3), intensity=0.6)
    return s


def sheet(scene, camera, times, out, name, size=(320, 320)):
    images = [
        scene.render(camera=camera, time=t, width=size[0], height=size[1], backend=backend(),
                     shadows=shadows()).image
        for t in times
    ]  # fmt: skip
    write_png(out / f"{name}_sheet.png",
              contact_sheet(images, columns=min(6, len(images)),
                            labels=[f"{t:.2f} s" for t in times]))  # fmt: skip


def walk(out: Path, quick: bool) -> dict:
    s = stage("Walk")
    hero = humanoid_rig(s, "Hero")
    body = mannequin(s, hero)
    body.color = SKIN
    a = s.animation
    rep = a.generate(hero, "walk", forward="-y")
    period = rep["period"]
    a.set_timeline(fps=24, start=0.0, end=2 * period)
    cam = s.add_camera("Cam", position=(3.2, -2.8, 1.4), look_at=(0, -0.6, 0.9), fov=40)
    a.follow_camera(cam, hero, offset=(3.2, -2.6, 1.4), lag=0.0, aim=(0, 0, 0.9))
    times = np.linspace(0, period, 7 if quick else 12, endpoint=False)
    sheet(s, cam, times, out, "walk")
    s.render(camera=cam, time=0.3 * period, width=1280, height=720, backend=backend(),
             shadows=shadows()).save(out / "hero_walk.png")  # fmt: skip
    contacts = a.foot_contacts(hero, FEET, start=0, end=2 * period, rate=120, height=0.03)
    s.save(out / "mannequin.aether")
    exp = s.export(out / "mannequin.glb", animation="clips")
    return {
        "clip": {k: rep[k] for k in ("keys", "channels", "max_error", "period", "speed")},
        "feet": {
            f: {
                "sliding_m": round(d["sliding_max"], 5),
                "penetration_m": round(d["penetration"], 5),
            }
            for f, d in contacts["feet"].items()
        },  # fmt: skip
        "vertices": body.mesh.vertex_count,
        "gltf": {"bytes": exp.byte_length, "validator": validate_glb(out / "mannequin.glb")},
    }


def step(out: Path, quick: bool) -> dict:
    s = stage("Step")
    hero = humanoid_rig(s, "Hero")
    body = mannequin(s, hero)
    body.color = SKIN
    box = s.add_cube("Step", size=1.0, position=(-0.1, -0.35, 0.12), scale=(0.5, 0.4, 0.24))
    box.color = (0.42, 0.45, 0.55)
    a = s.animation
    a.generate(hero, "idle", forward="-y", name="Idle")
    a.set_timeline(fps=24, start=0.0, end=2.0)
    # The foot's IK target sits on the step, the knee points forward (an armature point).
    rest = a.rig(hero).rest_model()
    ankle_h = float(rest[a.rig(hero).index("Foot.R")][2, 3])
    goal = s.add_object("Foot goal", position=(-0.1, -0.33, 0.24 + ankle_h))
    a.add_constraint(hero, {"name": "step", "type": "ik", "chain": ["Thigh.R", "Shin.R", "Foot.R"],
                            "target": {"object": goal.id},
                            "pole": {"point": [-0.1, -1.0, 0.6], "space": "armature"}})  # fmt: skip
    a.set_channel(hero, 'constraints["step"].influence', [(0.0, 0.0), (0.5, 0.0), (1.3, 1.0)],
                  interpolation="sine", ease="in_out")  # fmt: skip
    # (the idle action owns the hips: an additive layer lowers them a little)
    lower = a.add_clip("Lower hips")
    lower.set_channel('joints["Hips"].position.z', [(0.0, 0.95), (0.5, 0.95), (1.3, 0.91)],
                      interpolation="sine", ease="in_out")  # fmt: skip
    a.add_layer(hero, "weight shift", mode="additive")
    a.add_strip(hero, "weight shift", "Lower hips")
    cam = s.add_camera("Cam", position=(2.9, -2.9, 1.2), look_at=(0, -0.2, 0.85), fov=40)
    times = np.linspace(0, 2.0, 6 if quick else 12)
    sheet(s, cam, times, out, "step")
    s.render(camera=cam, time=1.8, width=1280, height=720, backend=backend(),
             shadows=shadows()).save(out / "hero_step.png")  # fmt: skip
    measured = []
    for t in (0.0, 0.9, 1.6, 2.0):
        foot = a.pose_joints(hero, t, "world")["Foot.R"][:3, 3]
        residual = max(a.rigging(t).residuals.values(), default=0.0)
        measured.append({"time": t, "ankle_above_floor_m": round(float(foot[2]), 5),
                         "residual_m": round(residual, 6)})  # fmt: skip
    return {"step_top_m": 0.24, "ankle_height_m": round(ankle_h, 4), "samples": measured}


def layers(out: Path, quick: bool) -> dict:
    s = stage("Layers")
    hero = humanoid_rig(s, "Hero")
    body = mannequin(s, hero)
    body.color = SKIN
    a = s.animation
    a.generate(hero, "idle", forward="-y", name="Idle", bind=False)
    wave = a.generate(hero, "wave", forward="-y", name="Wave", bind=False, side="R")
    a.generate(hero, "head_turn", forward="-y", name="Look", bind=False, yaw=40, back=True)
    mirrored = a.mirror_clip("Wave", hero, name="Wave mirrored")
    a.set_action(hero, "Idle")
    a.add_layer(hero, "upper", mask=[{"joint": "Chest", "descendants": True}])
    a.add_strip(hero, "upper", "Wave", start=0.0, blend_out=0.5)
    a.add_strip(hero, "upper", mirrored.name, start=wave["duration"] - 0.5, blend_in=0.5)
    a.add_layer(hero, "look", mode="additive",
                mask=[{"joint": "Neck"}, {"joint": "Head"}])  # fmt: skip
    a.add_strip(hero, "look", "Look", start=0.4)
    end = 2 * wave["duration"] - 0.5
    a.set_timeline(fps=24, start=0.0, end=end)
    cam = s.add_camera("Cam", position=(1.6, -3.6, 1.45), look_at=(0, 0, 1.15), fov=40)
    times = np.linspace(0, end, 8 if quick else 12)
    sheet(s, cam, times, out, "layers")
    s.render(camera=cam, time=1.2, width=1280, height=720, backend=backend(),
             shadows=shadows()).save(out / "hero_layers.png")  # fmt: skip
    hands = []
    for t in (1.2, end - 1.0):
        pose = a.pose_joints(hero, t, "world")
        hands.append({"time": round(float(t), 3),
                      "right_hand_z": round(float(pose["Hand.R"][2, 3]), 4),
                      "left_hand_z": round(float(pose["Hand.L"][2, 3]), 4)})  # fmt: skip
    return {"strips": 3, "layers": ["upper (masked: Chest and below)", "look (additive)"],
            "hands": hands, "mirrored_clip": mirrored.name}  # fmt: skip


def twist(out: Path) -> dict:
    """The hand twists 160 degrees about the forearm: LBS and DQS side by side, and the
    wrist's blend zone (where the forearm's flesh blends into the hand's) measured."""
    s = stage("Twist")
    results = {}
    for x, method in ((-0.35, "linear"), (0.35, "dual_quaternion")):
        hero = humanoid_rig(s, f"Hero {method}", position=(x, 0, 0))
        body = mannequin(s, hero, name=f"Body {method}", method=method)
        body.color = SKIN
        a = s.animation
        clip = a.add_clip(f"Twist {method}")
        clip.keyframe('joints["Hand.R"].rotation', 0.0, [0.0, 0.0, 0.0, 1.0])
        clip.keyframe('joints["Hand.R"].rotation', 1.0,
                      R.axis_angle([0, 0, 1], np.radians([160.0]))[0].tolist())  # fmt: skip
        a.set_action(hero, clip.name)
        # The blend zone: the right forearm's lower part, blending into the hand.
        _, profiles = mannequin_mesh(a.rig(hero))
        start, stop, ta, tb, _ = next(
            part for part in profiles if part[4][:2] == [(0.0, "UpperArm.R"), (0.15, "ForeArm.R")]
        )  # (the right forearm's tube)
        rest = body.mesh.positions.astype(np.float64)[start:stop]
        axis = (tb - ta) / np.linalg.norm(tb - ta)
        f = ((rest - ta) @ axis) / np.linalg.norm(tb - ta)
        zone = (f > 0.6) & (f < 0.95)  # (inside the forearm-to-hand blend: 0.55..1)

        def radius(points, ta=ta, axis=axis):
            rel = points - ta
            return np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)

        twisted = animation_overlay(s, 1.0).meshes[body.id].positions.astype(np.float64)
        r0 = radius(rest[zone])
        r1 = radius(twisted[start:stop][zone])
        results[method] = {
            "zone_radius_rest_m": round(float(r0.mean()), 5),
            "zone_radius_twisted_m": round(float(r1.mean()), 5),
            "kept": round(float(r1.mean() / r0.mean()), 4),
            "min_kept": round(float((r1 / r0).min()), 4),
            "vertices": int(zone.sum()),
        }
    cam = s.add_camera("Cam", position=(0.15, -1.25, 0.98), look_at=(-0.08, 0, 0.92), fov=40)
    options = {"width": 640, "height": 420, "backend": backend(), "shadows": shadows()}
    images = [s.render(camera=cam, time=t, **options).image for t in (0.0, 1.0)]
    labels = ["rest: LBS (left), DQS (right)", "hand twisted 160 degrees"]
    write_png(out / "twist_lbs_vs_dqs.png", contact_sheet(images, columns=2, labels=labels))
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output/animation/skinned"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {"backend": backend()}
    report["walk"] = walk(out, args.quick)
    print("walk:", report["walk"]["feet"], report["walk"]["gltf"]["validator"])
    report["step"] = step(out, args.quick)
    print("step:", report["step"]["samples"])
    report["layers"] = layers(out, args.quick)
    print("layers:", report["layers"]["hands"])
    report["twist"] = twist(out)
    print("twist:", report["twist"])
    write_json(out / "report.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
