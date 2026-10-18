"""Rigging benchmarks (Stage 18): auto-rigging, landmarks, skin weights, dual quaternion
skinning, per-frame evaluation of many rigged characters, the three character workflows
side by side, diagnostics, controls, rendering a changed pose and glTF.

Run from the repository root:

    python benchmarks/bench_rigging.py [--json OUT.json] [--label NAME] [--quick]
        [--only autorig,eval:50,...] [--compare SAVED.json]

Every group (and every size of a sized group) runs in its own Python process, so memory
peaks and caches of one never influence another; each reports its peak RSS (``VmHWM`` on
Linux). Timings are medians of repeated runs after a warm-up (``bench_engine.median_time``).
The character is the implicit humanoid of ``aether3d.sculpt.implicit`` (fixed recipe), so
every run measures the same data. Results depend on the machine: compare runs on one
machine only.

Groups:

* ``autorig:VOXEL`` -- landmarks, skeleton, controls and geodesic weights for the humanoid
  meshed at VOXEL millimetres (20: 6,480 vertices; 12: about 18,000): total, detection
  and binding, on a fresh scene each run.
* ``landmarks`` -- landmark detection alone at resolutions 96 and 128.
* ``weights`` -- geodesic weights (organic policy) for the 12 mm body on an existing rig,
  four smoothing passes over every vertex, a jacket's weight transfer, the inspector.
* ``dqs:V`` -- dual quaternion and blended (half DQS) skinning of V vertices, 4 influences,
  64 joints: NumPy reference, the native kernel on one thread and on all of them
  (bitwise equal to the reference).
* ``eval:N`` -- N auto-rigged walking characters (6,480 vertices each, IK constraints at
  influence 0, twist helpers, a gaze): the overlay and world matrices per frame (a new
  time each run, so no cache) and the memory each character adds.
* ``workflows`` -- 50 Minecraft-style characters animated three ways: rigid parts moved by
  their own transforms (no bones), blocks attached to an armature (Stage 17), one mesh
  skinned rigidly block by block (the blocky auto-rig of a merged mesh): per frame.
* ``diagnose`` -- the diagnostic pose suite and timeline checks of one character, the
  proposals, and applying the safe ones (verified).
* ``controls`` -- a matched IK/FK switch and a batch of pose commands.
* ``render`` -- a changed pose rendered at 320x180 on the CPU (evaluation, skinning,
  rasterization) against the same frame unposed.
* ``gltf`` -- export (baked) of a body, a jacket (transferred weights) and a helmet, and
  import back.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "examples"))

from bench_engine import compare, machine, median_time, ms, peak_rss_mib  # noqa: E402

SIZED = {"autorig": ("20", "12"), "dqs": ("100000", "1000000"), "eval": ("1", "10", "50")}
GROUPS = ("landmarks", "weights", "workflows", "diagnose", "controls", "render", "gltf")


def _repeat(quick: bool, n: int) -> int:
    return max(1, n // 3) if quick else n


def _body(voxel_mm: float):
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    return implicit_surface(humanoid_shapes(), voxel_mm / 1000.0)


def _rigged(mesh, name: str = "Body", scene=None, position=(0.0, 0.0, 0.0)):
    from aether3d import Scene

    s = scene or Scene("Bench")
    body = s.add_object(name, mesh=mesh, position=position)
    s.animation.autorig(body, name=f"{name} Rig")
    return s, body


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #


def group_autorig(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    mesh = _body(float(size))
    reports = []

    def run() -> None:
        s = Scene("Bench")
        body = s.add_object("Body", mesh=mesh)
        reports.append(s.animation.autorig(body, name="Rig"))

    total = median_time(run, _repeat(quick, 5))
    timings = reports[-1]["timings"]
    return {
        "voxel_mm": float(size),
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "total_ms": ms(total),
        "detect_ms": round(timings["detect_s"] * 1000, 3),
        "bind_ms": round(timings["bind_s"] * 1000, 3),
        "joints": len(reports[-1]["joints"]),
        "confidence": reports[-1]["confidence"],
        "peak_rss_mib": peak_rss_mib(),
    }


def group_landmarks(size: str, quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    mesh = _body(12.0)
    s = Scene("Bench")
    body = s.add_object("Body", mesh=mesh)
    out: dict[str, Any] = {"vertices": mesh.vertex_count}
    for res in (96, 128):
        out[f"detect_{res}_ms"] = ms(
            median_time(
                lambda r=res: s.animation.detect_landmarks(body, resolution=r), _repeat(quick, 5)
            )  # fmt: skip
        )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def _jacket(voxel: float):
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    shapes = humanoid_shapes()
    jacket = []
    for i in (0, 2, 3, 4, 9, 14):
        sh = copy.deepcopy(shapes[i])
        for key in ("radius", "radius_b"):
            if key in sh:
                sh[key] += 0.015
        if "radii" in sh:
            sh["radii"] = [r + 0.015 for r in sh["radii"]]
        jacket.append(sh)
    return implicit_surface(jacket, voxel)


def group_weights(size: str, quick: bool) -> dict[str, Any]:
    mesh = _body(12.0)
    s, body = _rigged(mesh)
    a = s.animation
    arm = s.get_by_name("Body Rig")
    jacket = s.add_object("Jacket", mesh=_jacket(0.012))
    out: dict[str, Any] = {
        "vertices": mesh.vertex_count,
        "jacket_vertices": jacket.mesh.vertex_count,
    }
    reps = _repeat(quick, 5)
    out["generate_ms"] = ms(median_time(lambda: a.generate_weights(body, arm), reps))
    out["smooth_4_ms"] = ms(median_time(
        lambda: a.edit_weights(body, [{"op": "smooth", "iterations": 4}]), reps))  # fmt: skip
    out["transfer_ms"] = ms(median_time(lambda: a.transfer_weights(jacket, body), reps))
    out["inspect_ms"] = ms(median_time(lambda: a.inspect_weights(body), reps))
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_dqs(size: str, quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import _native
    from aether3d.animation import rotation as R
    from aether3d.animation.rig import skin_dual_quaternion

    v = int(size)
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(v, 3)).astype(np.float32)
    nrm = rng.normal(size=(v, 3)).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    idx = rng.integers(0, 64, size=(v, 4)).astype(np.int32)
    w = rng.uniform(size=(v, 4))
    w /= w.sum(axis=1, keepdims=True)
    pal = np.tile(np.eye(4), (64, 1, 1))
    pal[:, :3, :3] = R.matrices(R.normalize(rng.normal(size=(64, 4))))
    pal[:, :3, 3] = rng.normal(scale=0.1, size=(64, 3))
    blend = np.full(v, 0.5)
    out: dict[str, Any] = {"vertices": v, "influences": 4, "joints": 64}
    repeat = _repeat(quick, 5 if v <= 100_000 else 3)
    native = bool(_native.status()["available"])
    out["native_available"] = native
    os.environ["AETHER3D_NATIVE"] = "0"
    ref = skin_dual_quaternion(pos, nrm, idx, w, pal)
    ref_b = skin_dual_quaternion(pos, nrm, idx, w, pal, blend=blend)
    out["numpy_ms"] = ms(median_time(lambda: skin_dual_quaternion(pos, nrm, idx, w, pal), repeat))
    out["numpy_blended_ms"] = ms(
        median_time(lambda: skin_dual_quaternion(pos, nrm, idx, w, pal, blend=blend), repeat)
    )
    os.environ.pop("AETHER3D_NATIVE")
    if native:
        os.environ["AETHER3D_NATIVE_THREADS"] = "1"
        got1 = skin_dual_quaternion(pos, nrm, idx, w, pal)
        out["native_1_thread_ms"] = ms(
            median_time(lambda: skin_dual_quaternion(pos, nrm, idx, w, pal), repeat)
        )
        os.environ["AETHER3D_NATIVE_THREADS"] = "0"
        got = skin_dual_quaternion(pos, nrm, idx, w, pal)
        got_b = skin_dual_quaternion(pos, nrm, idx, w, pal, blend=blend)
        out["native_threads"] = _native.threads()
        out["native_all_threads_ms"] = ms(
            median_time(lambda: skin_dual_quaternion(pos, nrm, idx, w, pal), repeat)
        )
        out["native_blended_all_threads_ms"] = ms(
            median_time(lambda: skin_dual_quaternion(pos, nrm, idx, w, pal, blend=blend), repeat)
        )
        os.environ.pop("AETHER3D_NATIVE_THREADS")
        out["bitwise_equal"] = bool(
            all(np.array_equal(x, y) for x, y in zip(ref, got, strict=True))
            and all(np.array_equal(x, y) for x, y in zip(got1, got, strict=True))
            and all(np.array_equal(x, y) for x, y in zip(ref_b, got_b, strict=True))
        )
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def _walkers(n: int):
    from aether3d import Scene

    mesh = _body(20.0)
    s = Scene("Crowd")
    a = s.animation
    rss = []
    for i in range(n):
        body = s.add_object(f"Body {i}", mesh=mesh, position=(1.2 * (i % 10), 1.5 * (i // 10), 0))
        a.autorig(body, name=f"Rig {i}")
        a.generate(f"Rig {i}", "walk", forward="-y")
        rss.append(peak_rss_mib())
    return s, mesh, rss


def _per_frame(scene, quick: bool, frames: int = 25) -> float:
    import numpy as np

    from aether3d.rendering.scene_data import world_matrices

    a = scene.animation
    shift = [0.0]
    times = np.linspace(0.0, 2.0, frames)

    def run() -> None:
        shift[0] += 0.0007  # (new times every run: nothing cached)
        for t in times:
            world_matrices(scene, a.overlay(float(t) + shift[0]))

    return median_time(run, _repeat(quick, 5)) / frames


def group_eval(size: str, quick: bool) -> dict[str, Any]:
    n = int(size)
    s, mesh, rss = _walkers(n)
    per = _per_frame(s, quick)
    out = {
        "characters": n,
        "vertices_each": mesh.vertex_count,
        "frame_ms": ms(per),
        "per_character_ms": round(ms(per) / n, 4),
        "fps": round(1.0 / per, 1),
    }
    if n >= 10:
        out["memory_per_character_mib"] = round((rss[-1] - rss[n // 2]) / (n - 1 - n // 2), 2)
    rig = s.animation.rig(s.get_by_name("Rig 0"))
    out["rig_json_bytes"] = len(json.dumps(rig.to_json()))
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_workflows(size: str, quick: bool) -> dict[str, Any]:
    import numpy as np
    from animation_common import blocky_character, pixel_skin, skin_material

    from aether3d import Scene
    from aether3d.mesh import Mesh
    from aether3d.rendering.png import encode_png

    n = 25 if quick else 50
    out: dict[str, Any] = {"characters": n}
    # 1. Rigid parts, no bones (object transforms).
    s1 = Scene("Rigid")
    for i in range(n):
        root = s1.add_object(f"Hero {i}", position=(1.5 * (i % 10), 2.0 * (i // 10), 0))
        torso = s1.add_object(f"Torso {i}", parent=root, position=(0, 0, 1.125))
        for name, size_, center, parent, pos in (
            ("Torso", None, None, None, None),
            ("Head", 0.5, (0, 0, 0), torso, (0, 0, 0.625)),
            ("Left Arm", (0.25, 0.25, 0.75), (0, 0, -0.3), torso, (0.375, 0, 0.3)),
            ("Right Arm", (0.25, 0.25, 0.75), (0, 0, -0.3), torso, (-0.375, 0, 0.3)),
            ("Left Leg", (0.25, 0.25, 0.75), (0, 0, -0.375), root, (0.125, 0, 0.75)),
            ("Right Leg", (0.25, 0.25, 0.75), (0, 0, -0.375), root, (-0.125, 0, 0.75)),
        ):
            o = torso if name == "Torso" else s1.add_object(f"{name} {i}", parent=parent,
                                                            position=pos)  # fmt: skip
            with o.edit_mesh() as m:
                m.add_box(size_ or (0.5, 0.25, 0.75), center=center or (0, 0, 0))
        s1.animation.generate_rigid(root, "walk", travel=False)
    out["rigid_frame_ms"] = ms(_per_frame(s1, quick))
    # 2. Blocks attached to joints (Stage 17's Nova).
    s2 = Scene("Attached")
    skin = encode_png(pixel_skin())
    material = skin_material(s2, "Nova", skin)
    for i in range(n):
        arm = blocky_character(s2, f"Nova {i}", skin, position=(1.5 * (i % 10), 2.0 * (i // 10), 0),
                               material=material)  # fmt: skip
        s2.animation.generate(arm, "walk", forward="-y", travel=False)
    out["attached_frame_ms"] = ms(_per_frame(s2, quick))
    # 3. One mesh, skinned rigidly block by block (the blocky auto-rig of a merged mesh).
    pos, tris, base = [], [], 0
    for o in s2.objects:
        if o.name.startswith("Nova 0 ") and o.mesh is not None:
            w = o.world_matrix
            pos.append(o.mesh.positions.astype(np.float64) @ w[:3, :3].T + w[:3, 3])
            tris.append(np.asarray(o.mesh.triangles, np.int64) + base)
            base += o.mesh.vertex_count
    merged = Mesh(np.concatenate(pos).astype(np.float32), np.concatenate(tris))
    s3 = Scene("Skinned")
    for i in range(n):
        body = s3.add_object(f"Block {i}", mesh=merged,
                             position=(1.5 * (i % 10), 2.0 * (i // 10), 0))  # fmt: skip
        s3.animation.autorig(body, name=f"Rig {i}", template="blocky")
        s3.animation.generate(f"Rig {i}", "walk", forward="-y", travel=False)
    out["skinned_frame_ms"] = ms(_per_frame(s3, quick))
    out["skinned_vertices_each"] = merged.vertex_count
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_diagnose(size: str, quick: bool) -> dict[str, Any]:
    s, body = _rigged(_body(20.0))
    a = s.animation
    a.generate("Body Rig", "walk", forward="-y")
    reps = _repeat(quick, 3)
    out: dict[str, Any] = {"vertices": body.mesh.vertex_count}
    out["diagnose_ms"] = ms(median_time(lambda: a.diagnose_rig("Body Rig"), reps))
    out["poses"] = len(a.diagnose_rig("Body Rig")["poses"])
    out["propose_ms"] = ms(median_time(lambda: a.propose_rig_fix("Body Rig"), reps))
    t0 = time.perf_counter()
    rep = a.apply_rig_fix("Body Rig", "verify")
    out["apply_verify_ms"] = ms(time.perf_counter() - t0)
    out["applied"], out["reverted"] = len(rep["applied"]), len(rep["reverted"])
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_controls(size: str, quick: bool) -> dict[str, Any]:
    s, _ = _rigged(_body(20.0))
    a = s.animation
    a.generate("Body Rig", "walk", forward="-y")
    reps = _repeat(quick, 5)
    out: dict[str, Any] = {}
    flip = [0]

    def switch() -> None:
        flip[0] ^= 1
        a.ik_fk("Body Rig", "leg_ik.L", "ik" if flip[0] else "fk", 0.4)

    out["ik_fk_switch_ms"] = ms(median_time(switch, reps))
    commands = [
        {"do": "look_at", "target": [0.5, -2.0, 1.6]},
        {"do": "reach", "hand": "R", "target": [0.35, -0.4, 1.2]},
        {"do": "crouch", "depth": 0.2},
        {"do": "plant_feet"},
    ]
    out["pose_4_commands_ms"] = ms(median_time(
        lambda: a.pose_character("Body Rig", commands, time=1.0), reps))  # fmt: skip
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_render(size: str, quick: bool) -> dict[str, Any]:
    s, body = _rigged(_body(20.0))
    a = s.animation
    a.generate("Body Rig", "walk", forward="-y", travel=False)
    s.add_light("directional", "Sun", direction=(-0.4, 0.6, -0.8), intensity=3.0)
    cam = s.add_camera("Cam", position=(2.4, -3.0, 1.4), look_at=(0, 0, 0.9), fov=40)
    reps = _repeat(quick, 5)
    shift = [0.0]

    def posed() -> None:
        shift[0] += 0.013
        a.time = 0.3 + shift[0]
        s.render(camera=cam, width=320, height=180, backend="cpu")

    out: dict[str, Any] = {"vertices": body.mesh.vertex_count}
    out["posed_frame_ms"] = ms(median_time(posed, reps))
    s2, body2 = _rigged(_body(20.0))
    s2.add_light("directional", "Sun", direction=(-0.4, 0.6, -0.8), intensity=3.0)
    cam2 = s2.add_camera("Cam", position=(2.4, -3.0, 1.4), look_at=(0, 0, 0.9), fov=40)
    out["rest_frame_ms"] = ms(median_time(
        lambda: s2.render(camera=cam2, width=320, height=180, backend="cpu"), reps))  # fmt: skip
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def group_gltf(size: str, quick: bool) -> dict[str, Any]:
    import tempfile

    from aether3d import Scene, import_gltf

    s = Scene("Outfit")
    body = s.add_object("Body", mesh=_body(20.0))
    s.add_object("Jacket", mesh=_jacket(0.02))
    from aether3d.sculpt.implicit import implicit_surface

    s.add_object("Helmet", mesh=implicit_surface(
        [{"type": "sphere", "center": [0.0, 0.0, 1.64], "radius": 0.125}], 0.02))  # fmt: skip
    a = s.animation
    a.autorig([body, "Jacket", "Helmet"], name="Rig",
              parts={"Jacket": "transfer", "Helmet": "attach"})  # fmt: skip
    a.generate("Rig", "walk", forward="-y")
    reps = _repeat(quick, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "outfit.glb"
        out: dict[str, Any] = {}
        out["export_bake_ms"] = ms(median_time(lambda: s.export(path, animation="bake"), reps))
        out["bytes"] = path.stat().st_size
        out["import_ms"] = ms(median_time(lambda: import_gltf(path), reps))
    out["peak_rss_mib"] = peak_rss_mib()
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_group(group: str, quick: bool) -> dict[str, Any]:
    name, _, size = group.partition(":")
    return globals()[f"group_{name}"](size, quick)


def jobs(selected: list[str], quick: bool) -> list[str]:
    out = []
    for g in selected:
        if g in SIZED:
            sizes = SIZED[g][:-1] if quick and len(SIZED[g]) > 1 else SIZED[g]
            out.extend(f"{g}:{s}" for s in sizes)
        else:
            out.append(g)
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
