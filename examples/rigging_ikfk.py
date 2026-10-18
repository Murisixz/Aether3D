"""Showcase C (Stage 18): IK/FK character controls -- both hands and both feet reach
explicit targets, and switching between IK and FK keeps the visible pose, checked at every
switch and at every keyframe.

The auto-rigged humanoid (``autorig``: two-bone IK limbs with poles, influence 0 so the
rig starts in FK) is driven by four target objects:

* 0.0 s -- the rest pose, FK;
* 0.6 s -- all four limbs switch to IK (``ik_fk``: targets and poles keyed where the FK
  pose puts the hands and feet; influence keyed to 1) -- nothing moves;
* 1.6 s -- the right hand reaches a high target, the left hand a target at its side, the
  left foot steps up onto a block while the right foot stays planted (hand targets lie
  along chosen directions at 85-90 % of the arm's measured length: an auto-rig's arm
  length follows the mesh);
* 2.6 s -- the hand targets move (keyed) while the hips lower 0.18 m; the feet stay;
* 3.4 s -- all four limbs switch back to FK (``ik_fk``: joints keyed with the IK
  solution's rotations; influence keyed to 0) -- nothing moves;
* 4.4 s -- FK keys only: the arms come down (rotations relative to rest); the legs hold
  the switched pose (the idle breathes in the upper body only).

Measured: each limb's distance to its target at every IK keyframe (the solver's residual
after joint limits), the largest joint displacement and the largest skin-vertex
displacement across each switch (the visible pose), and each limb's IK residual sampled
over the IK interval.

Run from the repository root:

    python examples/rigging_ikfk.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--no-video] [--fps 60]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/c_ikfk``): ``ikfk_sheet.png``
(the keyframes), ``switch_<time>.png`` (each switch: just before and just after),
``frames/`` (not committed), ``ikfk_1080p.mp4``/``ikfk_master.mkv``/``ikfk_preview.mp4``,
``ikfk.aether`` and ``report.json``.
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
from aether3d.rendering.overlay import animation_overlay  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402

LIMBS = {"arm_ik.L": "Hand.L", "arm_ik.R": "Hand.R", "leg_ik.L": "Foot.L", "leg_ik.R": "Foot.R"}
FEET = ("Foot.L", "Foot.R")
LEGS = tuple(f'joints["{j}' for j in ("Hips", "Thigh", "Shin", "Foot", "Toe"))
SWITCH_IK, REACH, MOVE, SWITCH_FK, FK_END = 0.6, 1.6, 2.6, 3.4, 4.4


def skin_world(scene: Scene, body, t: float) -> np.ndarray:
    ov = animation_overlay(scene, t)
    mesh = ov.meshes.get(body.id, body.mesh) if ov is not None else body.mesh
    w = world_matrices(scene, ov)[body.id]
    return mesh.positions.astype(np.float64) @ w[:3, :3].T + w[:3, 3]


def within_reach(arm, rig, side: str, toward, fraction: float, drop: float = 0.0):
    """A hand's goal: from the shoulder (``drop`` metres lower when the hips are lowered)
    toward the point ``toward``, at ``fraction`` of the arm's measured length -- an
    auto-rig's arm length follows the mesh, so fixed points may lie beyond it."""
    model = rig.rest_model()
    world = arm.world_matrix

    def at(name: str) -> np.ndarray:
        return (world @ model[rig.index(name)])[:3, 3]

    upper, fore, hand = at(f"UpperArm.{side}"), at(f"ForeArm.{side}"), at(f"Hand.{side}")
    length = float(np.linalg.norm(fore - upper) + np.linalg.norm(hand - fore))
    shoulder = upper - np.array([0.0, 0.0, drop])
    d = np.asarray(toward, float) - shoulder
    return tuple(float(v) for v in shoulder + d / np.linalg.norm(d) * fraction * length), length


def target_object(scene: Scene, rig, limb: str):
    c = next(c for c in rig.constraints if c.name == limb)
    return scene._objects[c.p["target"]["object"]]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "c_ikfk"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = C.set_backend(args.backend)
    size = (320, 180) if args.quick else (1920, 1080)
    fps = 10 if args.quick else args.fps
    t_all = time.perf_counter()
    scene = Scene("Showcase C")
    report: dict = {"backend": backend, "machine": C.machine(), "lighting": C.studio(scene)}
    body = scene.add_object("Body", mesh=C.humanoid_mesh(0.02 if args.quick else 0.012))
    a = scene.animation
    edges = C.tailor(scene, body)
    a.autorig(body, name="Rig")
    arm = scene.get_by_name("Rig")
    C.dress(scene, body, arm, edges)
    a.set_timeline(start=0.0, end=FK_END + 0.6)
    a.generate(arm, "idle", name="Controls", forward="-y")
    # The idle moves the upper body only: its leg and hip channels are removed, so the
    # legs carry exactly what IK and the switches key (a planted foot stays planted, and
    # after the switch to FK the legs hold the pose).
    removed = [path for path, _ in a.clip("Controls").data.channels if path.startswith(LEGS)]
    for path in removed:
        a.set_channel(arm, path, None)
    removed = len(removed)
    rig = a.rig(arm)
    # Markers on the targets (children: they move with them), a block for the step.
    colors = {
        "arm_ik.L": (0.95, 0.35, 0.2),
        "arm_ik.R": (0.2, 0.75, 0.95),
        "leg_ik.L": (0.95, 0.8, 0.2),
        "leg_ik.R": (0.55, 0.9, 0.35),
    }
    for limb, color in colors.items():
        tgt = target_object(scene, rig, limb)
        mat = scene.create_material(
            f"{limb} marker", base_color=color, emissive=color, emissive_strength=0.8, roughness=0.3
        )
        dot = scene.add_sphere(name=f"{limb} marker", radius=0.035, parent=tgt)
        dot.material = mat
    block = scene.add_cube(name="Block", size=1.0, position=(0.16, -0.18, 0.1))
    block.scale = (0.32, 0.36, 0.2)
    block.material = scene.create_material("Block", base_color=(0.62, 0.55, 0.45), roughness=0.7)
    print("1. FK -> IK at", SWITCH_IK)
    switches = {}
    joints_before = a.pose_joints(arm, SWITCH_IK - 1e-3, space="world")
    skin_before = skin_world(scene, body, SWITCH_IK - 1e-3)
    reps = {limb: a.ik_fk(arm, limb, "ik", SWITCH_IK) for limb in LIMBS}
    joints_after = a.pose_joints(arm, SWITCH_IK, space="world")
    skin_after = skin_world(scene, body, SWITCH_IK)
    switches["fk_to_ik"] = {
        "time": SWITCH_IK,
        "joint_mismatch_m": max(r["mismatch"] for r in reps.values()),
        "skin_displacement_m": float(np.abs(skin_after - skin_before).max()),
        "joints_moved_m": float(
            max(
                np.linalg.norm(joints_after[k][:3, 3] - joints_before[k][:3, 3])
                for k in joints_after
            )
        ),
    }
    print("2. targets keyed")
    hand_goals = {  # (side, toward, fraction of the arm's length, shoulder drop)
        REACH: {"arm_ik.R": ("R", (-0.32, -0.42, 1.72), 0.9, 0.0),
                "arm_ik.L": ("L", (0.62, -0.12, 1.12), 0.85, 0.0)},
        MOVE: {"arm_ik.R": ("R", (-0.12, -0.55, 1.3), 0.9, 0.18),
               "arm_ik.L": ("L", (0.48, -0.42, 1.25), 0.85, 0.18)},
    }  # fmt: skip
    goals = {t: {"leg_ik.L": (0.12, -0.2, 0.27), "leg_ik.R": None} for t in hand_goals}
    lengths = {}
    for t, hands in hand_goals.items():
        for limb, (side, toward, fraction, drop) in hands.items():
            goals[t][limb], lengths[side] = within_reach(arm, rig, side, toward, fraction, drop)
    report["goals"] = {
        "arm_length_m": {k: round(v, 4) for k, v in lengths.items()},
        "hands": {str(t): {k: [round(x, 4) for x in g[k]] for k in g if g[k] and "arm" in k}
                  for t, g in goals.items()},
        "rule": "hand goals along fixed directions from the shoulder at 85-90 % of the arm's "
        "measured length (the shoulder 0.18 m lower with the hips at 2.6 s)",
    }  # fmt: skip
    worlds0 = world_matrices(scene)
    for t, targets in goals.items():
        for limb, goal in targets.items():
            tgt = target_object(scene, rig, limb)
            pos = tgt.world_position if goal is None else goal
            parent = tgt.parent
            local = np.linalg.solve(worlds0[parent.id], np.append(pos, 1.0))[:3] if parent else pos
            a.keyframe(tgt, "transform.position", t, list(map(float, local)))
    hips = rig.joints[rig.role_index("hips")]
    a.keyframe(arm, 'joints["Hips"].position', REACH, list(hips.position))
    model = rig.rest_model()
    parent = rig.parents[rig.role_index("hips")]
    down = np.array([0.0, 0.0, -0.18])
    if parent >= 0:
        down = np.linalg.solve(model[parent][:3, :3], down)
    a.keyframe(arm, 'joints["Hips"].position', MOVE, list(np.asarray(hips.position) + down))
    print("3. IK -> FK at", SWITCH_FK)
    joints_before = a.pose_joints(arm, SWITCH_FK - 1e-3, space="world")
    skin_before = skin_world(scene, body, SWITCH_FK - 1e-3)
    reps = {limb: a.ik_fk(arm, limb, "fk", SWITCH_FK) for limb in LIMBS}
    joints_after = a.pose_joints(arm, SWITCH_FK, space="world")
    skin_after = skin_world(scene, body, SWITCH_FK)
    switches["ik_to_fk"] = {
        "time": SWITCH_FK,
        "joint_mismatch_m": max(r["mismatch"] for r in reps.values()),
        "skin_displacement_m": float(np.abs(skin_after - skin_before).max()),
        "joints_moved_m": float(
            max(
                np.linalg.norm(joints_after[k][:3, 3] - joints_before[k][:3, 3])
                for k in joints_after
            )
        ),
    }
    # Until the switch to IK the legs stand at rest (a channel holds its first key before
    # it: the switch's keys at 3.4 s would otherwise pose the legs from the start).
    legs = [
        {"do": "rotate", "joint": f"{j}.{side}", "euler": [0, 0, 0]}
        for side in ("L", "R")
        for j in ("Thigh", "Shin", "Foot")
    ]
    for t in (0.0, SWITCH_IK):
        a.pose_character(arm, legs, time=t)
    feet_switch = a.pose_joints(arm, SWITCH_FK, space="world")
    rest = [
        {"do": "rotate", "joint": j, "euler": [0, 0, 0]}
        for side in ("L", "R")
        for j in (f"UpperArm.{side}", f"ForeArm.{side}", f"Hand.{side}")
    ]
    a.pose_character(arm, rest, time=FK_END)
    feet_end = a.pose_joints(arm, FK_END + 0.5, space="world")
    switches["legs_held"] = {
        "idle_channels_removed": removed,
        "feet_moved_m": float(
            max(np.linalg.norm(feet_end[f][:3, 3] - feet_switch[f][:3, 3]) for f in FEET)
        ),
    }
    report["switches"] = switches
    # Reaching: each IK limb's end at its target, at the keyframes and sampled in between.
    reach = {}
    for t in (REACH, MOVE):
        rr = a.rigging(t, deform=False)
        reach[str(t)] = {
            limb: round(rr.residuals.get(f"Rig/{limb}", float("nan")), 6) for limb in LIMBS
        }
    samples = np.linspace(SWITCH_IK, SWITCH_FK - 1e-3, 41)
    worst = {limb: 0.0 for limb in LIMBS}
    for t in samples:
        rr = a.rigging(float(t), deform=False)
        for limb in LIMBS:
            worst[limb] = max(worst[limb], rr.residuals.get(f"Rig/{limb}", 0.0))
    report["reach"] = {
        "keyframes": reach,
        "worst_over_ik_interval_m": worst,
        "samples": int(samples.size),
    }
    print(f"   switches {switches}")
    print(f"   residuals {reach}")
    # Pictures.
    cam = scene.add_camera("Camera", position=(1.9, -3.5, 1.35), look_at=(0.05, 0.0, 0.9), fov=34)
    keys = [0.0, SWITCH_IK, REACH, MOVE, SWITCH_FK, FK_END]
    labels = [
        "rest (FK)",
        "switch FK->IK",
        "reach + step",
        "targets move",
        "switch IK->FK",
        "FK keys",
    ]
    shots = []
    for t in keys:
        shots.append(
            C.still(
                scene,
                out / "keys" / f"key_{t:.1f}.png",
                camera=cam,
                time=t,
                width=size[0] // 2,
                height=size[1] // 2,
                path_samples=16 if args.quick else 128,
            )
        )
    C.sheet(
        [s["path"] for s in shots],
        out / "ikfk_sheet.png",
        labels=labels,
        columns=3,
        width=size[0] // 2,
    )
    for t in (SWITCH_IK, SWITCH_FK):
        pair = [
            C.still(
                scene,
                out / "keys" / f"switch_{t:.1f}_{k}.png",
                camera=cam,
                time=tt,
                width=size[0] // 2,
                height=size[1] // 2,
            )["path"]
            for k, tt in (("before", t - 1e-3), ("after", t))
        ]
        C.sheet(
            pair,
            out / f"switch_{t:.1f}.png",
            columns=2,
            width=size[0] // 2,
            labels=[f"{t - 1e-3:.3f} s (before)", f"{t:.3f} s (after)"],
        )
    frames_dir = out / "frames"
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
    report["frames"] = C.render_frames(
        scene,
        frames_dir,
        camera=cam,
        fps=fps,
        start=0.0,
        end=FK_END + 0.6,
        width=size[0],
        height=size[1],
        progress=C.progress_printer("ik/fk"),
    )
    if not args.no_video and not args.quick:
        report["video"] = C.encode(frames_dir, out, "ikfk", fps)
    scene.save(out / "ikfk.aether")
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
