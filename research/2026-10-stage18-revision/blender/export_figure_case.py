"""Export Aether3D's anatomical figure, auto-rigged by Aether3D (the revision), as a case
for the Blender reference experiment: the mesh (world space), every joint as a Blender bone
(its weighting segment: head, tail; its rest frame; deform flag), and for every stress pose
each joint's posed rigid transform (armature-space world, posed @ rest^-1). Also Aether3D's
heat weights (plain LBS, no smoothing) for comparison.
Usage: python export_figure_case.py OUT_DIR [VOXEL [REMESH | none]]"""
import json, sys
from pathlib import Path
import numpy as np
from aether3d import Scene
from aether3d.sculpt.figure import figure_mesh
from aether3d.animation import rig_report as RR
from aether3d.rendering.scene_data import world_matrices

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
voxel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.007
remesh = sys.argv[3] if len(sys.argv) > 3 else "0.007"
remesh = None if remesh == "none" else float(remesh)
mesh = figure_mesh(voxel, remesh=remesh)
s = Scene("Figure case")
body = s.add_object("Body", mesh=mesh)
a = s.animation
a.autorig(body, name="Rig", weights={"corrective": None, "normals": "joints"})
rig = a.rig("Rig"); arm = s.get_by_name("Rig")
worlds = world_matrices(s)
A = worlds[arm.id]
rest = rig.rest_model()
starts, ends = rig.deform_segments()
joints = []
for k, j in enumerate(rig.joints):
    frame = rest[k][:3, :3] / np.linalg.norm(rest[k][:3, :3], axis=0)
    head = (A @ np.append(starts[k], 1.0))[:3]
    tail = (A @ np.append(ends[k], 1.0))[:3]
    if np.linalg.norm(tail - head) < 1e-6:  # (a zero-length segment: along its own +Y)
        tail = head + A[:3, :3] @ frame[:, 1] * 0.01
    joints.append({"name": j.name, "parent": None if j.parent < 0 else rig.joints[j.parent].name,
                   "head": head.tolist(), "tail": tail.tolist(),
                   "frame": (A[:3, :3] @ frame).tolist(), "deform": bool(j.deform)})
skin = a._owned[("skin", body.id)]._definition
names = [rig.joints[rig.index_of_id(jid)].name for jid in skin.joints]
poses = []
for p in RR.pose_suite(rig, "stress"):
    res = RR._posed(a, arm, rig, p["turns"], 0.0)
    _, model = res.joints[arm.id]
    mats = {}
    for k, j in enumerate(rig.joints):
        T = A @ model[k] @ np.linalg.inv(rest[k]) @ np.linalg.inv(A)
        mats[j.name] = T.tolist()
    poses.append({"name": p["name"], "matrices": mats})
P = (mesh.positions.astype(np.float64) @ worlds[body.id][:3, :3].T) + worlds[body.id][:3, 3]
np.save(out / "positions.npy", P)
np.save(out / "triangles.npy", np.asarray(mesh.triangles, np.int64))
np.save(out / "weights_aether_heat.npy", skin.dense())
(out / "case.json").write_text(json.dumps({"joints": joints, "weight_joints": names, "poses": poses,
                                           "mesh": f"figure_mesh({voxel}, remesh={remesh})"}))
print("vertices", len(P), "joints", len(joints), "weighted", len(names), "poses", len(poses))
