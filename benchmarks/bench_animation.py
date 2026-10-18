"""Animation benchmarks (Stage 17): curves, many animated objects, rigs, skinning (NumPy
and native), IK, NLA blending, motion generation, frame rendering and glTF interchange.

Run from the repository root:

    python benchmarks/bench_animation.py [--json OUT.json] [--label NAME] [--quick]
        [--only curves:1000,...] [--compare SAVED.json]

Every group (and every size of a sized group) runs in its own Python process, so memory
peaks and caches of one never influence another; each reports its peak RSS (``VmHWM`` on
Linux). Timings are medians of repeated runs after a warm-up (``bench_engine.median_time``).
Inputs are generated from fixed seeds, so every run measures the same data. Results
depend on the machine: compare runs on one machine only.

Groups:

* ``curves:K`` -- one Bezier F-curve of K auto-handle keys evaluated at 100,000 times
  (values; values and three derivatives), a weighted-handle curve (time inversion by
  safeguarded Newton), and a batch of 1,000 such curves at one time each.
* ``objects:N`` -- N objects, each animated by its own clip (position Bezier curves and a
  slerp rotation track): one evaluation (a new time each run, so no cache), the render
  overlay and the world matrices of the posed scene.
* ``instances:N`` -- N objects playing one shared clip through NLA strips, each with its
  own time offset (crowds, fields, instanced props): evaluation, overlay, posed worlds.
* ``crowd`` -- the 10,000 instances plus 48 walking blocky characters (rigs with six
  attached parts each, one shared walk cycle in offset strips, looping circle paths): the
  overlay with its rigging pass (rigs, attachments, path constraints) per frame.
* ``rig:J`` -- a branching rig of J joints, every joint with a rotation track: the pose,
  forward kinematics and the rigging pass per frame.
* ``skin:V`` -- linear blend skinning of V vertices (4 influences, 64 joints): NumPy,
  the native kernel on one thread and on all of them (bitwise equal), dual quaternions.
* ``ik`` -- the analytic two-bone solver, FABRIK on a 12-joint chain, and an IK
  constraint inside the rigging pass.
* ``nla`` -- a binding with 4 layers of 4 strips each (replace and additive, weights,
  blends, masks) evaluated per frame.
* ``generate`` -- the walk and the superhero landing generated on the humanoid test rig.
* ``frames`` -- 48 frames of an animated scene at 320x180 rendered to PNG on the CPU (and
  on the GPU when one works), and one motion-blurred frame (4 sub-frames).
* ``gltf`` -- export of a rigged, skinned, morphed scene with its clips (exact and fitted
  channels) and baked; import of the file back.
* ``fit`` -- keyframe reduction of a 10,000-key curve and fitting 20,000 samples.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests"))

from bench_engine import compare, machine, median_time, ms, peak_rss_mib  # noqa: E402

SIZED = {
    "curves": ("100", "10000"),
    "objects": ("100", "1000", "10000"),
    "instances": ("1000", "10000"),
    "rig": ("50", "200"),
    "skin": ("10000", "100000", "1000000"),
}
GROUPS = ("crowd", "ik", "nla", "generate", "frames", "gltf", "fit")


def _repeat(quick: bool, n: int) -> int:
    return max(1, n // 3) if quick else n


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #


def group_curves(size: str, quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d.animation import FCurve
    from aether3d.animation.batch import CurveBatch

    k = int(size)
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.05, 0.5, k))
    v = rng.normal(size=k)
    curve = FCurve.from_arrays(t, v, handles="auto")
    times = np.linspace(t[0], t[-1], 100_000)
    curve.evaluate(times)
    out: dict[str, Any] = {"keys": k, "times": times.size}
    out["evaluate_ms"] = ms(median_time(lambda: curve.evaluate(times), _repeat(quick, 9)))
    out["evaluate_order3_ms"] = ms(
        median_time(lambda: curve.evaluate(times, order=3), _repeat(quick, 5))
    )
    a = curve.arrays()
    a["weighted"][:] = True
    a["right_dt"] = a["right_dt"] * 1.4
    a["left_dt"] = a["left_dt"] * 0.7
    weighted = curve.replace(a)
    weighted.evaluate(times[:10])
    out["weighted_evaluate_ms"] = ms(
        median_time(lambda: weighted.evaluate(times), _repeat(quick, 5))
    )
    curves = [FCurve.from_arrays(t[:64], rng.normal(size=64), handles="auto") for _ in range(1000)]
    batch = CurveBatch(curves)
    idx = np.arange(1000)
    at = np.full(1000, float(t[30]))
    batch.evaluate(idx, at)
    out["batch_1000_ms"] = ms(median_time(lambda: batch.evaluate(idx, at), _repeat(quick, 21)))
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def _animated_scene(n: int):
    import numpy as np

    from aether3d import Scene
    from aether3d.animation import FCurve
    from aether3d.animation.clip import Clip
    from aether3d.animation.tracks import RotationTrack

    rng = np.random.default_rng(2)
    s = Scene("Objects")
    a = s.animation
    with s.history._suspended():
        mesh_obj = s.add_cube(name="Proto", size=0.5)
        mesh = mesh_obj.mesh
        for i in range(n):
            obj = s.add_object(f"O{i}", mesh=mesh, position=(i % 100, i // 100, 0))
            keys = np.array([0.0, 1.0, 2.0, 3.0])
            clip = Clip()
            for c, axis in enumerate("xyz"):
                base = (i % 100, i // 100, 0.0)[c]
                clip = clip.with_channel(
                    f"transform.position.{axis}",
                    FCurve.from_arrays(keys, base + rng.normal(scale=0.3, size=4), handles="auto"),
                )
            q = rng.normal(size=(4, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            clip = clip.with_channel("transform.rotation", RotationTrack.from_arrays(keys, q))
            handle = a.add_clip(f"C{i}", clip)
            a.set_action(obj, handle.name)
    return s


def group_objects(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.rendering.scene_data import world_matrices

    n = int(size)
    t0 = time.perf_counter()
    s = _animated_scene(n)
    build = time.perf_counter() - t0
    a = s.animation
    counter = [0]

    def fresh() -> float:
        counter[0] += 1
        return 0.5 + 1e-4 * counter[0]  # a new time each run: nothing cached

    a.evaluate(0.1)
    out: dict[str, Any] = {"objects": n, "build_s": round(build, 3)}
    out["evaluate_ms"] = ms(median_time(lambda: a.evaluate(fresh()), _repeat(quick, 7)))
    out["overlay_ms"] = ms(median_time(lambda: a.overlay(fresh()), _repeat(quick, 7)))
    out["posed_worlds_ms"] = ms(
        median_time(lambda: world_matrices(s, a.overlay(fresh())), _repeat(quick, 5))
    )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def _instanced_scene(n: int):
    import math

    from aether3d import Scene
    from aether3d.animation import FCurve
    from aether3d.animation.clip import Clip

    s = Scene("Instances")
    a = s.animation
    keys = [(0.0, 0.0), (0.5, 0.6), (1.0, 0.0), (2.0, 0.0)]
    clip = Clip().with_channel(
        "transform.position.z", FCurve.from_keys(keys, extrapolation="cycle")
    )
    ripple = a.add_clip("Ripple", clip)
    side = math.isqrt(n)
    with s.history._suspended():
        mesh = s.add_cube(name="Proto", size=0.9).mesh
        for i in range(n):
            x, y = i % side - side / 2, i // side - side / 2
            obj = s.add_object(f"B{i}", mesh=mesh, position=(x, y, 0))
            strip = {"clip": ripple.name, "start": -(math.hypot(x, y) * 0.15 % 2.0),
                     "repeat": 12.0}  # fmt: skip
            a.set_binding(obj, {"layers": [{"name": "base", "strips": [strip]}]})
    return s


def group_instances(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.rendering.scene_data import world_matrices

    n = int(size)
    t0 = time.perf_counter()
    s = _instanced_scene(n)
    build = time.perf_counter() - t0
    a = s.animation
    counter = [0]

    def fresh() -> float:
        counter[0] += 1
        return 0.5 + 1e-4 * counter[0]  # a new time each run: nothing cached

    a.evaluate(0.1)
    out: dict[str, Any] = {"objects": n, "strips": n, "build_s": round(build, 3)}
    out["evaluate_ms"] = ms(median_time(lambda: a.evaluate(fresh()), _repeat(quick, 7)))
    out["overlay_ms"] = ms(median_time(lambda: a.overlay(fresh()), _repeat(quick, 7)))
    out["posed_worlds_ms"] = ms(
        median_time(lambda: world_matrices(s, a.overlay(fresh())), _repeat(quick, 5))
    )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_crowd(quick: bool) -> dict[str, Any]:
    from test_animation_motion import blocky

    from aether3d.rendering.scene_data import world_matrices

    s = _instanced_scene(10_000)
    a = s.animation
    t0 = time.perf_counter()
    walkers = []
    for k in range(48):
        arm = blocky(s, f"Walker {k}")
        for joint in ("Body", "Head", "Arm.L", "Arm.R", "Leg.L", "Leg.R"):
            part = s.add_cube(f"Walker {k} {joint}", size=0.25)
            a.attach(part, arm, joint)
        walkers.append(arm)
    walk = a.generate(walkers[0], "walk", name="Walk", travel=False, bind=False)
    for k, arm in enumerate(walkers):
        radius = 28.0 + 2.5 * (k // 12)
        a.add_path(f"Ring {k}", kind="circle", center=[0, 0, 0], radius=radius, start=0.0)
        laps = walk["speed"] * 8.0 / (2 * 3.141592653589793 * radius)
        strip = {"clip": "Walk", "start": -(k % 7) * 0.1, "repeat": 12.0}
        a.set_binding(arm, {"layers": [{"name": "walk", "strips": [strip]}]})
        a.follow_path(arm, f"Ring {k}", start=0.0, end=8.0, offsets=(k / 48, k / 48 + laps),
                      easing=None, loop=True)  # fmt: skip
    build = time.perf_counter() - t0
    counter = [0]

    def fresh() -> float:
        counter[0] += 1
        return 0.5 + 1e-4 * counter[0]  # a new time each run: nothing cached

    a.overlay(0.1)
    out: dict[str, Any] = {"objects": len(list(s.objects)), "rigs": 48,
                           "crowd_build_s": round(build, 3)}  # fmt: skip
    out["evaluate_ms"] = ms(median_time(lambda: a.evaluate(fresh()), _repeat(quick, 7)))
    out["overlay_ms"] = ms(median_time(lambda: a.overlay(fresh()), _repeat(quick, 7)))
    out["posed_worlds_ms"] = ms(
        median_time(lambda: world_matrices(s, a.overlay(fresh())), _repeat(quick, 5))
    )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def _rig_scene(joints: int):
    import numpy as np

    from aether3d import Scene
    from aether3d.animation.clip import Clip
    from aether3d.animation.tracks import RotationTrack

    rng = np.random.default_rng(3)
    s = Scene("Rig")
    arm = s.add_object("Armature")
    rows = []
    for k in range(joints):
        parent = None if k == 0 else f"J{(k - 1) // 2}"
        rows.append({"name": f"J{k}", **({"parent": parent} if parent else {}),
                     "position": [0, 0.3, 0] if k else [0, 0, 1]})  # fmt: skip
    a = s.animation
    a.add_rig(arm, rows)
    clip = Clip()
    keys = np.array([0.0, 1.0, 2.0])
    for k in range(joints):
        q = rng.normal(size=(3, 4)) * [0.2, 0.2, 0.2, 1.0]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        clip = clip.with_channel(f'joints["J{k}"].rotation', RotationTrack.from_arrays(keys, q))
    handle = a.add_clip("Pose", clip)
    a.set_action(arm, handle.name)
    return s, arm


def group_rig(size: str, quick: bool) -> dict[str, Any]:
    j = int(size)
    s, arm = _rig_scene(j)
    a = s.animation
    counter = [0]

    def fresh() -> float:
        counter[0] += 1
        return 0.3 + 1e-4 * counter[0]

    a.rigging(0.1)
    out: dict[str, Any] = {"joints": j}
    out["evaluate_ms"] = ms(median_time(lambda: a.evaluate(fresh()), _repeat(quick, 9)))
    out["rigging_ms"] = ms(median_time(lambda: a.rigging(fresh()), _repeat(quick, 9)))
    out["pose_joints_ms"] = ms(
        median_time(lambda: a.pose_joints(arm, fresh(), "armature"), _repeat(quick, 9))
    )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_skin(size: str, quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import _native
    from aether3d.animation import rotation as R
    from aether3d.animation.rig import skin_dual_quaternion, skin_linear

    v = int(size)
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(v, 3)).astype(np.float32)
    nrm = rng.normal(size=(v, 3)).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    idx = rng.integers(0, 64, size=(v, 4)).astype(np.int32)
    w = rng.uniform(size=(v, 4))
    w /= w.sum(axis=1, keepdims=True)
    q = R.normalize(rng.normal(size=(64, 4)))
    pal = np.tile(np.eye(4), (64, 1, 1))
    pal[:, :3, :3] = R.matrices(q)
    pal[:, :3, 3] = rng.normal(scale=0.1, size=(64, 3))
    out: dict[str, Any] = {"vertices": v, "influences": 4, "joints": 64}
    repeat = _repeat(quick, 7 if v <= 100_000 else 3)
    import os

    native = _native.status()["available"]
    out["native_available"] = bool(native)
    os.environ["AETHER3D_NATIVE"] = "0"
    ref = skin_linear(pos, nrm, idx, w, pal)
    out["numpy_ms"] = ms(median_time(lambda: skin_linear(pos, nrm, idx, w, pal), repeat))
    os.environ.pop("AETHER3D_NATIVE")
    if native:
        os.environ["AETHER3D_NATIVE_THREADS"] = "1"
        got1 = skin_linear(pos, nrm, idx, w, pal)
        out["native_1_thread_ms"] = ms(
            median_time(lambda: skin_linear(pos, nrm, idx, w, pal), repeat)
        )
        os.environ["AETHER3D_NATIVE_THREADS"] = "0"
        got = skin_linear(pos, nrm, idx, w, pal)
        out["native_threads"] = _native.threads()
        out["native_all_threads_ms"] = ms(
            median_time(lambda: skin_linear(pos, nrm, idx, w, pal), repeat)
        )
        os.environ.pop("AETHER3D_NATIVE_THREADS")
        out["bitwise_equal"] = bool(
            np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
            and np.array_equal(got1[0], got[0]) and np.array_equal(got1[1], got[1])
        )  # fmt: skip
    out["dual_quaternion_ms"] = ms(
        median_time(lambda: skin_dual_quaternion(pos, nrm, idx, w, pal), max(1, repeat // 2))
    )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_ik(quick: bool) -> dict[str, Any]:
    import numpy as np
    from test_animation_motion import humanoid

    from aether3d import Scene
    from aether3d.animation.ik import fabrik, two_bone

    root, mid, end = np.zeros(3), np.array([0, 0.45, 0.0]), np.array([0, 0.87, 0.0])
    target = np.array([0.2, 0.6, 0.1])
    pole = np.array([1.0, 0.4, 0.0])
    out: dict[str, Any] = {}
    out["two_bone_us"] = round(
        1e6 * median_time(lambda: [two_bone(root, mid, end, target, pole) for _ in range(1000)],
                          _repeat(quick, 5)) / 1000, 3)  # fmt: skip
    chain = np.array([[0.0, 0.1 * k, 0.0] for k in range(12)])
    out["fabrik_12_us"] = round(
        1e6 * median_time(lambda: [fabrik(chain, np.array([0.4, 0.6, 0.3])) for _ in range(100)],
                          _repeat(quick, 5)) / 100, 3)  # fmt: skip
    s = Scene("IK")
    goal = s.add_cube(name="Goal", position=(0.3, 0.6, 0.4))
    arm = humanoid(s)
    a = s.animation
    a.set_channel(goal, "transform.position.y", [(0, 0.6), (2, -0.2)])
    a.add_constraint(arm, {"name": "reach", "type": "ik", "chain": ["Thigh.R", "Shin.R", "Foot.R"],
                           "target": {"object": goal.id}})  # fmt: skip
    counter = [0]

    def fresh() -> float:
        counter[0] += 1
        return 0.2 + 1e-4 * counter[0]

    a.rigging(0.1)
    out["rigging_with_ik_ms"] = ms(median_time(lambda: a.rigging(fresh()), _repeat(quick, 15)))
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_nla(quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import Scene

    s = Scene("NLA")
    obj = s.add_cube(name="Cube")
    a = s.animation
    rng = np.random.default_rng(5)
    for c in range(8):
        clip = a.add_clip(f"C{c}")
        for axis in "xyz":
            clip.set_channel(f"transform.position.{axis}",
                             [(t, float(rng.normal())) for t in range(5)])  # fmt: skip
        clip.set_channel("transform.rotation_euler.z", [(0, 0.0), (4, float(rng.normal() * 90))])
    for layer in range(4):
        a.add_layer(obj, f"L{layer}", mode="additive" if layer % 2 else "replace", weight=0.7)
        for k in range(4):
            a.add_strip(obj, f"L{layer}", f"C{(layer + k) % 8}", start=k * 1.0, blend_in=0.3,
                        blend_out=0.3)  # fmt: skip
    counter = [0]

    def fresh() -> float:
        counter[0] += 1
        return 1.5 + 1e-4 * counter[0]

    a.evaluate(0.1)
    return {
        "layers": 4,
        "strips": 16,
        "evaluate_ms": ms(median_time(lambda: a.evaluate(fresh()), _repeat(quick, 15))),
        "peak_rss_mib": peak_rss_mib(),
    }


def group_generate(quick: bool) -> dict[str, Any]:
    from test_animation_motion import blocky, humanoid

    from aether3d import Scene

    out: dict[str, Any] = {}
    for kind in ("walk", "land"):

        def run(kind=kind):
            s = Scene("Gen")
            arm = humanoid(s)
            return s.animation.generate(arm, kind)

        rep = run()
        out[f"{kind}_ms"] = ms(median_time(run, _repeat(quick, 3), warmup=0))
        out[f"{kind}_keys"] = rep["keys"]
    s = Scene("Blocky")
    arm = blocky(s)
    out["blocky_walk_ms"] = ms(median_time(lambda: s.animation.generate(arm, "walk", bind=False),
                                           _repeat(quick, 3), warmup=0))  # fmt: skip
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def _frames_scene():
    from aether3d import Scene

    s = Scene("Frames")
    s.add_plane("Floor", size=20)
    for i in range(30):
        s.add_cube(f"Box{i}", size=0.6, position=((i % 6) * 1.5 - 4, (i // 6) * 1.5 - 3, 0.3))
    ball = s.add_sphere("Ball", position=(0, 0, 1))
    cam = s.add_camera("Cam", position=(6, -9, 5), look_at=(0, 0, 0.5))
    s.active_camera = cam
    s.add_directional_light("Sun", direction=(-0.4, 0.5, -1))
    a = s.animation
    a.set_timeline(fps=24, start=0, end=2)
    a.set_channel(ball, "transform.position.x", [(0, -4.0), (2, 4.0)])
    a.set_channel(
        ball, "transform.position.z", [(0, 1.0), (0.5, 2.5), (1, 1.0), (1.5, 2.5), (2, 1.0)]
    )
    a.camera_orbit(cam, "Ball", 0, 2, degrees=60)
    return s


def group_frames(quick: bool) -> dict[str, Any]:
    import tempfile

    from aether3d.animation.frames import render_frames
    from aether3d.rendering.gpu import gpu_status

    s = _frames_scene()
    out: dict[str, Any] = {"frames": 48, "size": "320x180"}
    with tempfile.TemporaryDirectory() as tmp:
        backends = ["cpu"]
        try:
            if gpu_status().get("available"):
                backends.append("gpu")
        except Exception:  # (no GPU stack)
            pass
        for backend in backends:
            render_frames(s, Path(tmp) / f"warm_{backend}", frames=[0], width=320, height=180,
                          backend=backend)  # fmt: skip
            t0 = time.perf_counter()
            rep = render_frames(s, Path(tmp) / backend, end=2.0, width=320, height=180,
                                backend=backend)  # fmt: skip
            sec = time.perf_counter() - t0
            out[f"{backend}_frames"] = len(rep["frames"])
            out[f"{backend}_seconds"] = round(sec, 3)
            out[f"{backend}_ms_per_frame"] = ms(sec / len(rep["frames"]))
        t0 = time.perf_counter()
        render_frames(s, Path(tmp) / "blur", frames=[12], width=320, height=180,
                      motion_blur={"shutter": 0.5, "samples": 4}, backend="cpu")  # fmt: skip
        out["cpu_motion_blur_4_ms"] = ms(time.perf_counter() - t0)
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_gltf(quick: bool) -> dict[str, Any]:
    import tempfile

    from test_animation_gltf import rigged_scene

    from aether3d import import_gltf

    s = rigged_scene()
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        clips = Path(tmp) / "clips.glb"
        baked = Path(tmp) / "baked.glb"
        out["export_clips_ms"] = ms(median_time(lambda: s.export(clips, animation="clips"),
                                                _repeat(quick, 3), warmup=0))  # fmt: skip
        out["export_bake_ms"] = ms(median_time(lambda: s.export(baked, animation="bake"),
                                               _repeat(quick, 3), warmup=0))  # fmt: skip
        out["clips_bytes"] = clips.stat().st_size
        out["baked_bytes"] = baked.stat().st_size
        out["import_ms"] = ms(median_time(lambda: import_gltf(clips), _repeat(quick, 3), warmup=0))
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_fit(quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d.animation import FCurve
    from aether3d.animation.fitting import fit_samples, simplify

    t = np.linspace(0, 100, 10_000)
    v = np.sin(t) + 0.3 * np.sin(3.1 * t)
    curve = FCurve.from_arrays(t, v, handles="auto")
    rep = simplify(curve, 1e-3)[1]
    out = {
        "simplify_10000_ms": ms(
            median_time(lambda: simplify(curve, 1e-3), _repeat(quick, 3), warmup=0)
        ),
        "simplify_keys_after": rep["keys_after"],
    }
    ts = np.linspace(0, 40, 20_000)
    vs = np.sin(ts) + 0.05 * np.sin(17 * ts)
    rep = fit_samples(ts, vs, 1e-3)[1]
    out["fit_20000_ms"] = ms(
        median_time(lambda: fit_samples(ts, vs, 1e-3), _repeat(quick, 3), warmup=0)
    )
    out["fit_keys"] = rep["keys"]
    out["peak_rss_mib"] = peak_rss_mib()
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_group(name: str, quick: bool) -> dict[str, Any]:
    if ":" in name:
        group, size = name.split(":", 1)
        return globals()[f"group_{group}"](size, quick)
    return globals()[f"group_{name}"](quick)


def jobs(only: list[str], quick: bool) -> list[str]:
    out = []
    for name in only:
        if ":" in name or name in GROUPS:
            out.append(name)
        elif name in SIZED:
            sizes = SIZED[name][:2] if quick else SIZED[name]
            out.extend(f"{name}:{s}" for s in sizes)
        else:
            raise SystemExit(f"unknown group {name!r}")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join([*SIZED, *GROUPS]))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print("BENCH-RESULT " + json.dumps(run_group(args.worker, args.quick)))
        return 0
    groups = jobs([g for g in args.only.split(",") if g], args.quick)
    data: dict[str, Any] = {"label": args.label, "machine": machine(), "groups": {}}
    for group in groups:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", group]
        if args.quick:
            command.append("--quick")
        t0 = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BENCH-RESULT ")]
        if proc.returncode != 0 or not lines:
            data["groups"][group] = {"error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"{group}: FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        data["groups"][group] = json.loads(lines[-1][len("BENCH-RESULT ") :])
        print(f"{group} ({time.perf_counter() - t0:.0f} s): {json.dumps(data['groups'][group])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        compare(data, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
