"""Export an Aether3D auto-rigged case (mesh, skeleton frames, stress poses, weights) for
the Blender reference experiment. Usage: python export_case.py OUT_DIR [--voxel 0.012]"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import rigging_common as C
from aether3d import Scene
from aether3d.animation import rotation as R
from aether3d.animation.rig_report import resolve_pose, pose_suite

POSES = {
    "shoulder_raise_150.L": [{"joint": "UpperArm.L", "axis": [0.0, -1.0, 0.0], "angle": 150.0}],
    "shoulder_raise_90.L": [{"joint": "UpperArm.L", "axis": [0.0, -1.0, 0.0], "angle": 60.0}],
    "hip_flex_100.L": [{"joint": "Thigh.L", "axis": "+x", "angle": -100.0}],
    "wrist_twist_90.L": [{"joint": "Hand.L", "axis": "+y", "angle": 90.0}],
    "knee_bend_140.L": [{"joint": "Thigh.L", "axis": "+x", "angle": -60.0},
                        {"joint": "Shin.L", "axis": "+x", "angle": 140.0}],
    "elbow_bend_140.L": [{"joint": "UpperArm.L", "axis": "+x", "angle": -60.0},
                         {"joint": "ForeArm.L", "axis": "+x", "angle": 140.0}],
    "torso_twist_60": [{"joint": "Spine", "axis": "+y", "angle": 30.0},
                       {"joint": "Chest", "axis": "+y", "angle": 30.0}],
    "spine_bend_45": [{"joint": "Spine", "axis": "+x", "angle": 22.5},
                      {"joint": "Chest", "axis": "+x", "angle": 22.5}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.012)
    ap.add_argument("--rig", default=None, help="a JSON rig to use instead of autorig")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene("export")
    body = scene.add_object("Body", mesh=C.humanoid_mesh(a.voxel))
    anim = scene.animation
    rep = anim.autorig(body, name="Rig")
    arm = scene.get_by_name("Rig")
    rig = anim.rig(arm)
    model = rig.rest_model()
    tails = rig.tails()
    joints = []
    for k, j in enumerate(rig.joints):
        head = model[k][:3, 3]
        tail = (model[k] @ np.append(tails[k], 1.0))[:3]
        frame = model[k][:3, :3] / np.linalg.norm(model[k][:3, :3], axis=0)
        joints.append({"name": j.name, "parent": None if j.parent < 0 else rig.joints[j.parent].name,
                       "head": head.tolist(), "tail": tail.tolist(), "frame": frame.tolist(),
                       "deform": bool(j.deform)})
    poses = []
    for name, turns in POSES.items():
        p = resolve_pose(rig, {"name": name, "turns": turns})
        poses.append(p)
    for p in pose_suite(rig):
        if p["name"] not in POSES:
            poses.append(p)
    pose_out = []
    for p in poses:
        rots = {}
        for k, (axis, deg) in p["turns"].items():
            local_axis = np.linalg.solve(model[k][:3, :3], axis)
            local_axis /= np.linalg.norm(local_axis)
            qd = R.axis_angle(local_axis, math.radians(deg))[0]  # x y z w
            rots[rig.joints[k].name] = qd.tolist()
        pose_out.append({"name": p["name"], "joint": rig.joints[p["joint"]].name, "rotations": rots})
    mesh = body.mesh
    np.save(out / "positions.npy", mesh.positions.astype(np.float64))
    np.save(out / "triangles.npy", np.asarray(mesh.triangles, np.int64))
    np.save(out / "normals.npy", mesh.normals.astype(np.float64))
    sk = anim._owned[("skin", body._id)]._definition
    dense = sk.dense()
    names = [rig.joints[rig.index_of_id(j)].name for j in sk.joints]
    np.save(out / "weights_aether_v019.npy", dense)
    (out / "case.json").write_text(json.dumps({
        "joints": joints, "poses": pose_out, "weight_joints": names,
        "landmarks": {k: v["position"] for k, v in rep["landmarks"].items()},
    }, indent=1))
    print("vertices", mesh.vertex_count, "triangles", mesh.triangle_count, "joints", len(joints),
          "poses", len(pose_out))


if __name__ == "__main__":
    main()
