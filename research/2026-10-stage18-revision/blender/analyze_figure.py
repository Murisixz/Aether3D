"""Measure every variant of the figure case under the stress poses with Aether3D's
geometry-only measures (kept size of the bent joint's skin, new creases, sharpest crease,
folds, worst edge stretch): v0.19.0's own rig and weights; the revision's heat weights
(LBS: the default skin's geometry) and with delta mush (8 iterations); Blender 5.2's Automatic
Weights posed by Blender (LBS, DQS, LBS + Corrective Smooth at 5 and 12 iterations).
Usage: python analyze_figure.py CASE_DIR"""
import json, sys
from pathlib import Path
import numpy as np
from aether3d import Scene, Mesh, load_scene
from aether3d.animation import rig_report as RR
from aether3d.rendering.scene_data import world_matrices

case_dir = Path(sys.argv[1])
case = json.loads((case_dir / "case.json").read_text())
p0 = np.load(case_dir / "positions.npy"); tris = np.load(case_dir / "triangles.npy")
POSES = [p["name"] for p in case["poses"]]

def scene_with(opts):
    s = Scene("F"); body = s.add_object("Body", mesh=Mesh(p0.astype(np.float32), tris))
    s.animation.autorig(body, name="Rig", weights=opts)
    return s, body

def posed(s, body):
    a = s.animation; rig = a.rig("Rig"); arm = s.get_by_name("Rig")
    sk = RR._Skinned(a, body.id, a._owned[("skin", body.id)]._definition, rig, world_matrices(s))
    out = {}
    for p in RR.pose_suite(rig, "stress"):
        res = RR._posed(a, arm, rig, p["turns"], 0.0)
        out[p["name"]] = (sk.posed(res)[0], res, p)
    return sk, out, rig, arm

def measure(sk, rig, arm, p1, res, pose, worlds):
    m = RR._tri_measures(sk.p0, p1, sk.tris, None, None, sk.pairs, sk.rest_dihedral)
    rec = {"new_creases": int(m["creases"].size), "sharpest_crease": round(m["sharpest_crease"], 1),
           "folded_triangles": int(m["folded"].size), "worst_stretch": round(m["worst_stretch"], 3)}
    k = pose["joint"]
    ring = RR._joint_ring(sk, rig, k, worlds[arm.id])
    _, model = res.joints[arm.id]
    piv0 = (worlds[arm.id] @ rig.rest_model()[k])[:3, 3]; piv1 = (worlds[arm.id] @ model[k])[:3, 3]
    rec["joint_ring"] = round(float(np.linalg.norm(p1[ring] - piv1, axis=1).mean()
                                    / np.linalg.norm(sk.p0[ring] - piv0, axis=1).mean()), 4)
    return rec

table = {}
s_heat, b_heat = scene_with({"corrective": None, "normals": "joints"})
sk, heat_posed, rig, arm = posed(s_heat, b_heat)
worlds = world_matrices(s_heat)
table["aether_heat_lbs"] = {n: measure(sk, rig, arm, *heat_posed[n], worlds) for n in POSES}
s_def, b_def = scene_with({"corrective": {"iterations": 8}, "normals": "joints"})
sk_d, def_posed, rig_d, arm_d = posed(s_def, b_def)
table["aether_heat_mush8"] = {n: measure(sk_d, rig_d, arm_d, *def_posed[n], world_matrices(s_def)) for n in POSES}
for v in ("lbs", "dqs", "lbs_cs", "lbs_cs12"):
    table[f"blender_{v}"] = {}
    for n in POSES:
        P = np.load(case_dir / "posed_blender" / f"{n}__{v}.npy")
        _, res, pose = heat_posed[n]
        table[f"blender_{v}"][n] = measure(sk, rig, arm, P, res, pose, worlds)
# Blender's weights skinned by Aether3D (LBS) against Blender's own LBS: the same skinning.
B = np.load(case_dir / "weights_blender_heat.npy"); B /= np.maximum(B.sum(1, keepdims=True), 1e-12)
H = np.load(case_dir / "weights_aether_heat.npy")
agree = float((B.argmax(1) == H.argmax(1)).mean()); mad = float(np.abs(B - H).sum(1).mean() / 2)
s_b, b_b = scene_with({"corrective": None, "normals": "joints"})
a = s_b.animation
names = case["weight_joints"]
k = 8
idx = np.argsort(-B, axis=1)[:, :k]
w = np.take_along_axis(B, idx, 1); w /= w.sum(1, keepdims=True)
a.skin(b_b, "Rig", weights={"indices": idx.astype(np.int32), "weights": w, "joints": names})
skb, bl_posed, _, _ = posed(s_b, b_b)
diff = max(float(np.abs(bl_posed[n][0] - np.load(case_dir / "posed_blender" / f"{n}__lbs.npy")).max()) for n in POSES)
# v0.19.0: its own rig and weights, measured by the same code (its own suite under stress).
v = load_scene(case_dir / "v019.aether")
d = v.animation.diagnose_rig("Rig", poses="stress", animation=False, max_findings=500)
table["v0.19.0"] = {n: {k2: e["Body"].get(k2) for k2 in ("joint_ring", "new_creases", "sharpest_crease",
                    "folded_triangles", "worst_stretch")} for n, e in d["poses"].items()}
summary = {"weights": {"aether_vs_blender_dominant_agreement": round(agree, 4),
                       "aether_vs_blender_mean_abs_difference": round(mad, 4),
                       "blender_weights_lbs_in_aether3d_vs_blender_max_m": diff},
           "poses": table}
(case_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
cols = ["v0.19.0", "aether_heat_lbs", "aether_heat_mush8", "blender_lbs", "blender_dqs", "blender_lbs_cs", "blender_lbs_cs12"]
print(json.dumps(summary["weights"]))
print("pose".ljust(22) + "".join(c[:15].rjust(26) for c in cols))
for n in POSES:
    row = n.ljust(22)
    for c in cols:
        r = table[c].get(n)
        row += (f"{r['joint_ring']:.3f}/{r['new_creases']}/{r['folded_triangles']}/{r['worst_stretch']:.2f}" if r else "-").rjust(26)
    print(row)
tot = {c: (sum(table[c][n]["new_creases"] for n in POSES if n in table[c]), sum(table[c][n]["folded_triangles"] for n in POSES if n in table[c]),
           max(table[c][n]["worst_stretch"] for n in POSES if n in table[c])) for c in cols}
print("totals (creases, folds, worst stretch):", json.dumps(tot))
