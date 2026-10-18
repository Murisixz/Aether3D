"""Showcase B (Stage 18): deformation quality under demanding poses -- a shoulder raised
overhead, a hip flexed past 90 degrees, a wrist twisted 90 degrees, knees and elbows bent
140 degrees, the torso twisted 60 degrees -- compared across weights and skinning methods.

One auto-rigged skeleton drives four copies of the same body, each bound differently:

* **envelope, linear** -- inverse-distance weights over the nearest bones (``skin``'s
  automatic weights, Stage 17's), linear blend skinning;
* **geodesic, linear** -- Stage 18's weights (geodesic distances inside the body, body-part
  regions, the forearm twist helper), linear blend skinning;
* **geodesic, dual quaternion** -- the same weights, dual quaternion skinning;
* **geodesic, verified blend** -- the same weights, with the fixes ``apply_rig_fix``
  proposes and verifies for this body (dual quaternion blended in only where a joint
  collapses, the smallest share that resolves it; kept only when measured better).

Each pose is a custom diagnostic pose (data: ``{name, turns: [{joint, axis, angle}]}``),
measured by ``diagnose_rig`` -- how much of its size the skin around the joint keeps
(``joint_ring``), triangles turned over, the worst edge stretch -- and rendered close up by
the GPU path tracer (key, fill and rim lights, a sky environment) and as stretch maps.

Run from the repository root:

    python examples/rigging_deformation.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]
        [--samples 96]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/b_deformation``):
``compare_<pose>.png`` (the four bindings side by side), ``deformation_grid.png`` (every
pose and binding), ``stretch_<pose>.png`` (stretch maps: red stretched, blue compressed),
``deformation.aether`` and ``report.json`` (every measurement).
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

METHODS = (
    ("envelope", "envelope LBS"),
    ("geodesic", "geodesic LBS"),
    ("dqs", "geodesic DQS"),
    ("blended", "verified blend"),
)

# name: (turns, the joint looked at, camera offset from it (metres, world))
POSES = {
    "shoulder_raise_150.L": (
        [{"joint": "UpperArm.L", "axis": [0.0, -1.0, 0.0], "angle": 150.0}],
        "UpperArm.L",
        (0.55, -0.75, 0.05),
    ),
    "hip_flex_100.L": (
        [{"joint": "Thigh.L", "axis": "+x", "angle": -100.0}],
        "Thigh.L",
        (0.95, -0.35, 0.1),
    ),
    "wrist_twist_90.L": (
        [{"joint": "Hand.L", "axis": "+y", "angle": 90.0}],
        "ForeArm.L",
        (0.3, -0.62, 0.12),
    ),
    "knee_bend_140.L": (
        [
            {"joint": "Thigh.L", "axis": "+x", "angle": -60.0},
            {"joint": "Shin.L", "axis": "+x", "angle": 140.0},
        ],
        "Shin.L",
        (0.95, -0.45, 0.15),
    ),
    "elbow_bend_140.L": (
        [
            {"joint": "UpperArm.L", "axis": "+x", "angle": -60.0},
            {"joint": "ForeArm.L", "axis": "+x", "angle": 140.0},
        ],
        "ForeArm.L",
        (0.55, -0.55, 0.1),
    ),
    "torso_twist_60": (
        [
            {"joint": "Spine", "axis": "+y", "angle": 30.0},
            {"joint": "Chest", "axis": "+y", "angle": 30.0},
        ],
        "Chest",
        (0.6, -1.25, 0.15),
    ),
}


# Where each pose's stretch map is seen from (the posed limbs are the left ones).
STRETCH_VIEW = {
    "shoulder_raise_150.L": "front",
    "hip_flex_100.L": "left",
    "wrist_twist_90.L": "front",
    "knee_bend_140.L": "left",
    "elbow_bend_140.L": "left",
    "torso_twist_60": "front",
}

# The joint whose surrounding skin each pose measures (``joint_ring``).
MEASURED = {
    "shoulder_raise_150.L": "UpperArm.L",
    "hip_flex_100.L": "Thigh.L",
    "wrist_twist_90.L": "Hand.L",
    "knee_bend_140.L": "Shin.L",
    "elbow_bend_140.L": "ForeArm.L",
    "torso_twist_60": "Chest",
}


def build(voxel: float) -> tuple[Scene, dict]:
    scene = Scene("Showcase B")
    lighting = C.studio(scene)
    mesh = C.humanoid_mesh(voxel)
    a = scene.animation
    bodies = {key: scene.add_object(f"Body ({label})", mesh=mesh) for key, label in METHODS}
    edges = C.tailor(scene, bodies["geodesic"])
    for body in bodies.values():
        body.mesh = bodies["geodesic"].mesh
    rep = a.autorig(bodies["geodesic"], name="Rig")
    arm = scene.get_by_name("Rig")
    rig = a.rig(arm)
    sk = a._owned[("skin", bodies["geodesic"]._id)]._definition
    explicit = {
        "indices": np.asarray(sk.indices),
        "weights": np.asarray(sk.weights),
        "joints": [rig.joints[rig.index_of_id(j)].name for j in sk.joints],
    }
    a.skin(bodies["envelope"], arm, weights="envelope")
    a.skin(bodies["dqs"], arm, weights=explicit, method="dual_quaternion")
    a.skin(bodies["blended"], arm, weights=explicit)
    for key, body in bodies.items():
        C.dress(scene, body, arm, edges)
        body.visible = key == "geodesic"
    report = {
        "lighting": lighting,
        "vertices": bodies["geodesic"].mesh.vertex_count,
        "autorig": {"joints": len(rep["joints"]), "seconds": rep["timings"]["total_s"]},
        "methods": dict(METHODS),
    }
    return scene, report


def verified_blend(scene: Scene, arm, body, report: dict) -> None:
    """The fixes Stage 18 proposes for this body, verified one by one."""
    a = scene.animation
    t0 = time.perf_counter()
    # Measured in the showcase's own poses (custom diagnostic poses are data).
    specs = [{"name": n, "turns": t, "joint": MEASURED[n]} for n, (t, _, _) in POSES.items()]
    props = a.propose_rig_fix(arm, poses=specs, animation=False)["proposals"]
    mine = [
        p
        for p in props
        if p.get("target", {}).get("mesh") == body.name
        and p["operations"][0]["call"] == "skin_method"
    ]
    rep = a.apply_rig_fix(arm, mine, poses=specs, animation=False)
    sk = a._owned[("skin", body._id)]._definition
    report["verified_blend"] = {
        "proposed": [p["title"] for p in mine],
        "applied": len(rep["applied"]),
        "reverted": len(rep["reverted"]),
        "why": rep.get("why"),
        "method": sk.method,
        "dq_vertices": int((np.asarray(sk.dq) > 0).sum()) if sk.dq is not None else 0,
        "dq_share_max": float(np.max(sk.dq)) if sk.dq is not None else 0.0,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def measure(scene: Scene, arm, report: dict) -> dict:
    a = scene.animation
    specs = [
        {"name": n, "turns": turns, "joint": MEASURED[n]} for n, (turns, _, _) in POSES.items()
    ]
    t0 = time.perf_counter()
    diag = a.diagnose_rig(arm, poses=specs, animation=False, max_findings=500)
    names = {f"Body ({label})": key for key, label in METHODS}
    table = {}
    for pose, meshes in diag["poses"].items():
        row = {}
        for mesh, rec in meshes.items():
            row[names[mesh]] = {
                k: rec[k]
                for k in (
                    "joint_ring",
                    "flipped_triangles",
                    "stretched_edges",
                    "worst_stretch",
                    "volume_change",
                    "interpenetrating",
                )
                if k in rec
            }
        table[pose] = row
    report["measurements"] = table
    report["measure_seconds"] = round(time.perf_counter() - t0, 2)
    report["findings"] = sorted(
        {
            (f["code"], f.get("mesh"), f.get("pose"))
            for f in diag["findings"]
            if f["severity"] != "info"
        }
    )
    return table


def pose_clip(scene: Scene, arm, name: str) -> None:
    """The pose as keys at time 0 of its own clip (rotations from rest, joint frames)."""
    a = scene.animation
    turns = POSES[name][0]
    a.set_action(arm, None)
    commands = []
    for t in turns:
        if isinstance(t["axis"], str):
            euler = [0.0, 0.0, 0.0]
            euler["xyz".index(t["axis"][1])] = t["angle"] * (1 if t["axis"][0] == "+" else -1)
            commands.append({"do": "rotate", "joint": t["joint"], "euler": euler})
    if commands:
        a.pose_character(arm, commands, time=0.0)
    for t in turns:  # (an armature-space axis: through the joint's frame)
        if not isinstance(t["axis"], str):
            rig = a.rig(arm)
            k = rig.index(t["joint"])
            local = np.linalg.solve(rig.rest_model()[k][:3, :3], np.asarray(t["axis"], float))
            local /= np.linalg.norm(local)
            from aether3d.animation import rotation as R

            qd = R.axis_angle(local, np.radians(t["angle"]))[0]
            rest = np.array(rig.joints[k].rotation, float)
            q = R.multiply(rest[None], qd[None])[0]
            a.keyframe(arm, f'joints["{t["joint"]}"].rotation', 0.0, list(q / np.linalg.norm(q)))


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
    scene, report = build(0.02 if args.quick else 0.012)
    report["backend"] = backend
    report["machine"] = C.machine()
    a = scene.animation
    arm = scene.get_by_name("Rig")
    bodies = {key: scene.get_by_name(f"Body ({label})") for key, label in METHODS}
    print("1. the verified blend")
    verified_blend(scene, arm, bodies["blended"], report)
    print(f"   {report['verified_blend']}")
    print("2. measurements")
    table = measure(scene, arm, report)
    for pose, row in table.items():
        print(
            f"   {pose}: "
            + ", ".join(f"{m} keeps {r.get('joint_ring', 1.0):.2f}" for m, r in row.items())
        )
    print("3. renders")
    cam = scene.add_camera("Close", position=(1, -1, 1), look_at=(0, 0, 1), fov=30)
    grid, grid_labels, stills = [], [], []
    for pose, (_, joint, offset) in POSES.items():
        pose_clip(scene, arm, pose)
        p = a.pose_joints(arm, 0.0, space="world")[joint][:3, 3]
        cam.position = tuple(p + np.asarray(offset))
        cam.look_at(tuple(p))
        row = []
        for key, label in METHODS:
            for k2, body in bodies.items():
                body.visible = k2 == key
            path = out / "tiles" / f"{pose}_{key}.png"
            stills.append(
                C.still(
                    scene, path, camera=cam, time=0.0, width=tile, height=tile, path_samples=samples
                )
            )
            row.append(path)
            m = table[pose][key]
            grid_labels.append(f"{label} {m.get('joint_ring', 1.0):.2f}")
        grid += row
        C.sheet(
            row,
            out / f"compare_{pose}.png",
            columns=4,
            width=tile,
            labels=[
                f"{lab} keeps {table[pose][k].get('joint_ring', 1.0):.2f}" for k, lab in METHODS
            ],
        )
        for k2, body in bodies.items():
            body.visible = k2 == "geodesic"
    C.sheet(grid, out / "deformation_grid.png", columns=4, width=min(tile, 360), labels=grid_labels)
    report["renders"] = {
        "tiles": len(stills),
        "seconds": round(sum(s["seconds"] for s in stills), 1),
        "backend": stills[0]["backend"] if stills else None,
        "samples": samples,
        "size": [tile, tile],
    }
    print("4. stretch maps")
    for pose, (turns, _, _) in POSES.items():
        spec = {"name": pose, "turns": turns}
        maps = []
        for key in ("envelope", "geodesic", "blended"):
            for k2, body in bodies.items():
                body.visible = k2 == key
            prev = a.rig_preview(
                arm,
                mode="stretch",
                pose=spec,
                view=STRETCH_VIEW[pose],
                width=tile // 2,
                height=tile * 2 // 3,
            )
            maps.append(prev["image"])
        C.sheet(
            maps,
            out / f"stretch_{pose}.png",
            columns=3,
            width=tile // 2,
            labels=["envelope LBS", "geodesic LBS", "verified blend"],
        )
    for k2, body in bodies.items():
        body.visible = k2 == "geodesic"
    scene.save(out / "deformation.aether")
    report["notes"] = [
        "keeps: joint_ring, the mean distance of the skin around the measured joint to its "
        "pivot, posed over rest (1.00: no collapse; below 0.75 diagnose_rig reports "
        "JOINT_COLLAPSE)",
        "the verified blend adds dual quaternion skinning only where a measured collapse "
        "asks for it (here the elbow; the knee keeps more than 0.75)",
        "jagged dark patches on the twisted torso, in all four methods: the path tracer "
        "gives no direct light to a facet facing away from the light, and skinned normals "
        "(the rest normals transformed by the blended joint matrices) turn less than the "
        "facets where weights change quickly; the rasterizer draws the surface smooth",
    ]
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
