"""Showcase B's reference data from a figure case (see README.md): v0.19.0's rig and
weights on the case's mesh (``v019.aether``, made with the v0.19.0 sources) and Blender's
Automatic Weights on the revision's skeleton (``weights_blender_heat.npy``), with the
mesh's hash -- written to ``examples/data/rigging/figure_reference_<TAG>.npz``.
Weights are kept as the strongest k per vertex, quantized to 16 bits.
Usage: python make_showcase_data.py CASE_DIR TAG"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from aether3d import load_scene

case_dir, tag = Path(sys.argv[1]), sys.argv[2]
root = Path(__file__).resolve().parents[3]
case = json.loads((case_dir / "case.json").read_text())
p = np.load(case_dir / "positions.npy").astype(np.float32)


def top(W, k):
    idx = np.argsort(-W, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(W, idx, axis=1)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return idx.astype(np.int16), np.round(w * 65535).astype(np.uint16)


v = load_scene(case_dir / "v019.aether")
a = v.animation
arm = v.get_by_name("Rig")
rig = a.rig(arm)
skin = a._owned[("skin", v.get_by_name("Body")._id)]._definition
v_names = [rig.joints[rig.index_of_id(j)].name for j in skin.joints]
v_idx, v_w = top(skin.dense(), int(skin.indices.shape[1]))
B = np.load(case_dir / "weights_blender_heat.npy")
b_idx, b_w = top(B, 8)
out = root / "examples" / "data" / "rigging" / f"figure_reference_{tag}.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out,
    mesh_sha256=np.array(hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()),
    v019_rig=np.array(json.dumps(rig.to_json())),
    v019_armature=np.asarray(arm.world_matrix, np.float64),
    v019_joints=np.array(v_names), v019_indices=v_idx, v019_weights=v_w,
    blender_joints=np.array(case["weight_joints"]), blender_indices=b_idx, blender_weights=b_w,
    blender=np.array(json.loads((case_dir / "blender_run.json").read_text())["blender"]),
)
print(out, out.stat().st_size, "bytes")
