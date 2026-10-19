"""Showcase B (the Stage 18 revision): deformation quality before and after, against Blender
-- a shoulder raised 150 degrees, a hip flexed 100, a wrist twisted 90, a knee and an elbow
bent 140, the torso twisted 60 -- the stress suite's hardest poses.

Four copies of the anatomical figure (``aether3d.sculpt.figure``), each bound differently,
all deformed and rendered by Aether3D:

* **v0.19.0** -- the rig and weights v0.19.0 made for this exact mesh (its 23-joint
  auto-rig, geodesic voxel weights, four influences; computed by the v0.19.0 sources and
  committed as data, ``examples/data/rigging/figure_reference_*.npz``);
* **heat (default)** -- the revision's auto-rig: bone heat weights, eight influences, linear
  blend skinning, normals that follow the surface;
* **heat + delta mush** -- the same weights with corrective smoothing (opt-in);
* **Blender** -- Blender 5.2's Automatic Weights for the revision's skeleton on the same
  mesh (computed by Blender in the revision's experiment and committed as data), skinned by
  Aether3D (equal to Blender's own LBS within 3e-6 m).

Each pose is the stress suite's (``diagnose_rig(poses="stress")``), resolved on each rig's
own joints; every measure comes from the same geometry-only code: the share of its size the
skin around the bent joint keeps, new creases (edges smooth at rest bent past 60 degrees),
folds and the worst edge stretch. The bodies are clay (one material), so nothing hides the
deformation; the GPU path tracer renders each close up (key, fill and rim lights, a sky).

Run from the repository root:

    python examples/rigging_deformation.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--samples 96]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/b_deformation``):
``compare_<pose>.png`` (the four bindings side by side, measured), ``deformation_grid.png``
(every pose and binding), ``stretch_<pose>.png`` (stretch maps: red stretched, blue
compressed), ``deformation.aether`` and ``report.json`` (every measurement).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rigging_common as C  # noqa: E402

from aether3d import Scene  # noqa: E402
from aether3d.animation import rotation as R  # noqa: E402

DATA = HERE / "data" / "rigging"
METHODS = (
    ("v019", "v0.19.0"),
    ("heat", "heat"),
    ("mush", "heat + mush"),
    ("blender", "Blender"),
)
# The poses shown (the stress suite's), the joint looked at, the camera's offset from it.
POSES = {
    "shoulder_raise_150.L": ("UpperArm.L", (0.55, -0.75, 0.05)),
    "hip_flex_100.L": ("Thigh.L", (0.95, -0.35, 0.1)),
    "wrist_twist_90.L": ("ForeArm.L", (0.3, -0.62, 0.12)),
    "knee_bend_140.L": ("Shin.L", (0.95, -0.45, 0.15)),
    "elbow_bend_140.L": ("ForeArm.L", (0.55, -0.55, 0.1)),
    "torso_twist_60": ("Chest", (0.6, -1.25, 0.15)),
}
STRETCH_VIEW = {
    "shoulder_raise_150.L": "front",
    "hip_flex_100.L": "left",
    "wrist_twist_90.L": "front",
    "knee_bend_140.L": "left",
    "elbow_bend_140.L": "left",
    "torso_twist_60": "front",
}
MEASURES = ("joint_ring", "new_creases", "sharpest_crease", "folded_triangles", "worst_stretch")


def _weights(data, prefix: str) -> dict:
    """Stored weights (the strongest per vertex, 16-bit) as ``skin``'s explicit weights."""
    w = data[f"{prefix}_weights"].astype(np.float64)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return {"indices": data[f"{prefix}_indices"].astype(np.int32), "weights": w,
            "joints": [str(n) for n in data[f"{prefix}_joints"]]}  # fmt: skip


def build(quick: bool) -> tuple[Scene, dict]:
    scene = Scene("Showcase B")
    lighting = C.studio(scene)
    mesh = C.character_mesh(quick)
    tag = "20mm" if quick else "7mm"
    data = np.load(DATA / f"figure_reference_{tag}.npz")
    digest = hashlib.sha256(np.ascontiguousarray(mesh.positions, np.float32).tobytes())
    if str(data["mesh_sha256"]) != digest.hexdigest():
        raise SystemExit(
            f"the figure's mesh changed: regenerate {DATA}/figure_reference_{tag}.npz "
            "(research/2026-10-stage18-revision/blender/README.md)"
        )
    a = scene.animation
    clay = scene.create_material("Clay", base_color=(0.78, 0.66, 0.58), roughness=0.62)
    bodies = {key: scene.add_object(f"Body ({label})", mesh=mesh) for key, label in METHODS}
    for body in bodies.values():
        body.material = clay
    t0 = time.perf_counter()
    rep = a.autorig(bodies["heat"], name="Rig")
    seconds = time.perf_counter() - t0
    arm = scene.get_by_name("Rig")
    sk = a._owned[("skin", bodies["heat"]._id)]._definition
    rig = a.rig(arm)
    heat = {"indices": np.asarray(sk.indices), "weights": np.asarray(sk.weights),
            "joints": [rig.joints[rig.index_of_id(j)].name for j in sk.joints]}  # fmt: skip
    a.skin(bodies["mush"], arm, weights=heat, normals="surface", smooth={"iterations": 8})
    a.skin(bodies["blender"], arm, weights=_weights(data, "blender"), normals="surface")
    old = scene.add_object("Rig v0.19")
    m = np.asarray(data["v019_armature"])
    old.position = tuple(m[:3, 3].tolist())
    q = R.from_matrices((m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0))[None])[0]
    old.rotation = tuple(q.tolist())
    # (its IK targets, poles and gaze were objects of v0.19's scene: the suite's poses turn
    # joints with IK off, so only constraints between joints -- the forearm twist -- stay)
    raw = json.loads(str(data["v019_rig"]))

    def on_objects(c: dict) -> bool:
        return any(isinstance(v, dict) and "object" in v for v in c.values())

    raw["constraints"] = [c for c in raw.get("constraints", []) if not on_objects(c)]
    raw.pop("controls", None)
    a.set_rig(old, raw)
    a.skin(bodies["v019"], old, weights=_weights(data, "v019"))  # (v0.19's joint normals)
    for key, body in bodies.items():
        body.visible = key == "heat"
    report = {
        "lighting": lighting,
        "vertices": mesh.vertex_count,
        "autorig": {"joints": len(rep["joints"]), "seconds": round(seconds, 2)},
        "v019": {"joints": len(a.rig(old).joints)},
        "blender": str(data["blender"]),
        "methods": dict(METHODS),
    }
    return scene, report


def measure(scene: Scene, report: dict) -> dict:
    """Every binding under the stress suite, by ``diagnose_rig`` (the same code for all)."""
    a = scene.animation
    t0 = time.perf_counter()
    names = {f"Body ({label})": key for key, label in METHODS}
    table: dict[str, dict] = {}
    totals: dict[str, dict] = {}
    for rig_name in ("Rig", "Rig v0.19"):
        diag = a.diagnose_rig(rig_name, poses="stress", animation=False, max_findings=500)
        for pose, meshes in diag["poses"].items():
            for mesh, rec in meshes.items():
                key = names[mesh]
                table.setdefault(pose, {})[key] = {k: rec.get(k) for k in MEASURES}
                t = totals.setdefault(key, {"new_creases": 0, "folded_triangles": 0,
                                            "worst_stretch": 0.0})  # fmt: skip
                t["new_creases"] += rec["new_creases"]
                t["folded_triangles"] += rec["folded_triangles"]
                t["worst_stretch"] = max(t["worst_stretch"], rec["worst_stretch"])
    report["measurements"] = table
    report["totals"] = totals
    report["measure_seconds"] = round(time.perf_counter() - t0, 2)
    return table


def key_pose(scene: Scene, armature: str, pose: str) -> None:
    """The stress pose as keys at time 0 (its turns: armature-space axes, from rest)."""
    from aether3d.animation.rig_report import resolve_pose

    a = scene.animation
    rig = a.rig(armature)
    a.set_action(armature, None)
    model = rig.rest_model()
    for k, (axis, deg) in resolve_pose(rig, pose)["turns"].items():
        local = np.linalg.solve(model[k][:3, :3], np.asarray(axis, float))
        qd = R.axis_angle(local / np.linalg.norm(local), math.radians(deg))[0]
        q = R.multiply(np.asarray(rig.joints[k].rotation, float)[None], qd[None])[0]
        a.keyframe(armature, f'joints["{rig.joints[k].name}"].rotation', 0.0,
                   list(q / np.linalg.norm(q)))  # fmt: skip


def label(name: str, m: dict) -> str:
    """A tile's measures in at most about 50 characters (the sheet's font is wide)."""
    folds = m["folded_triangles"]
    return (f"{name}: {m['joint_ring']:.2f} kept, {m['new_creases']} creases, "
            f"{folds} fold{'' if folds == 1 else 's'}, x{m['worst_stretch']:.1f}")  # fmt: skip


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "b_deformation"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = C.set_backend(args.backend)
    tile = 240 if args.quick else 720
    samples = 8 if args.quick else args.samples
    t_all = time.perf_counter()
    scene, report = build(args.quick)
    report["backend"] = backend
    report["machine"] = C.machine()
    a = scene.animation
    bodies = {key: scene.get_by_name(f"Body ({lab})") for key, lab in METHODS}
    print("1. measurements (the stress suite)")
    table = measure(scene, report)
    for key, t in report["totals"].items():
        print(f"   {key}: {t}")
    print("2. renders")
    cam = scene.add_camera("Close", position=(1, -1, 1), look_at=(0, 0, 1), fov=30)
    grid, grid_labels, stills = [], [], []
    for pose, (joint, offset) in POSES.items():
        key_pose(scene, "Rig", pose)
        key_pose(scene, "Rig v0.19", pose)
        row = []
        for key, name in METHODS:
            for k2, body in bodies.items():
                body.visible = k2 == key
            # (each binding framed on its own rig's joint: v0.19.0's joints sit elsewhere)
            rig_name = "Rig v0.19" if key == "v019" else "Rig"
            p = a.pose_joints(rig_name, 0.0, space="world")[joint][:3, 3]
            cam.position = tuple(p + np.asarray(offset))
            cam.look_at(tuple(p))
            path = out / "tiles" / f"{pose}_{key}.png"
            stills.append(C.still(scene, path, camera=cam, time=0.0, width=tile, height=tile,
                                  path_samples=samples))  # fmt: skip
            row.append(path)
            m = table[pose][key]  # (the grid's tiles are narrow: the kept size only)
            grid_labels.append(f"{name}: {m['joint_ring']:.2f} kept")
        grid += row
        C.sheet(row, out / f"compare_{pose}.png", columns=4, width=tile,
                labels=[label(n, table[pose][k]) for k, n in METHODS])  # fmt: skip
    C.sheet(grid, out / "deformation_grid.png", columns=4, width=min(tile, 360), labels=grid_labels)
    report["renders"] = {
        "tiles": len(stills),
        "seconds": round(sum(s["seconds"] for s in stills), 1),
        "backend": stills[0]["backend"] if stills else None,
        "samples": samples,
        "size": [tile, tile],
    }
    print("3. stretch maps")
    for pose in POSES:
        maps = []
        for key, armature in (("v019", "Rig v0.19"), ("heat", "Rig"), ("blender", "Rig")):
            for k2, body in bodies.items():
                body.visible = k2 == key
            prev = a.rig_preview(armature, mode="stretch", pose=pose, view=STRETCH_VIEW[pose],
                                 width=tile // 2, height=tile * 2 // 3)  # fmt: skip
            maps.append(prev["image"])
        C.sheet(maps, out / f"stretch_{pose}.png", columns=3, width=tile // 2,
                labels=["v0.19.0", "heat (the default)", "Blender's weights"])  # fmt: skip
    for k2, body in bodies.items():
        body.visible = k2 == "heat"
    scene.save(out / "deformation.aether")
    report["notes"] = [
        "keeps: joint_ring, the mean distance of the skin around the bent joint to its "
        "pivot, posed over rest (1.00: no collapse; below 0.75 diagnose_rig reports "
        "JOINT_COLLAPSE); creases: edges smooth at rest bent past 60 degrees; folds: "
        "triangles turned against their posed neighbourhood; stretch: the worst edge "
        "length ratio",
        "v0.19.0 and Blender are data computed for this exact mesh (its hash is checked); "
        "every deformation and image here is Aether3D's",
    ]
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
