"""Blender reference experiment (run headless inside Blender):
    blender -b --factory-startup --python blender_ref.py -- CASE_DIR [--weights FILE]
Builds the case's mesh and skeleton exactly (bone frames = the Aether3D joint frames),
parents with Automatic Weights (bone heat), saves the weights, poses every stress pose and
saves the deformed positions for: LBS, DQS (Preserve Volume), LBS + Corrective Smooth,
DQS + Corrective Smooth. With --weights, uses those weights (n x bones .npy, columns in
case['weight_joints'] order) instead of computing bone heat."""
import json, sys, time
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
case_dir = Path(argv[0])
given = None
if "--weights" in argv:
    given = argv[argv.index("--weights") + 1]
case = json.loads((case_dir / "case.json").read_text())
pos = np.load(case_dir / "positions.npy")
tris = np.load(case_dir / "triangles.npy")

bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata(pos.tolist(), [], tris.tolist())
mesh.update()
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
assert len(mesh.vertices) == len(pos), (len(mesh.vertices), len(pos))

arm_data = bpy.data.armatures.new("Rig")
arm = bpy.data.objects.new("Rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for j in case["joints"]:
    b = arm_data.edit_bones.new(j["name"])
    b.head = Vector(j["head"])
    b.tail = Vector(j["tail"])
    frame = np.array(j["frame"])  # columns x y z
    b.align_roll(Vector(frame[:, 2].tolist()))
    b.use_deform = j["deform"]
    eb[j["name"]] = b
for j in case["joints"]:
    if j["parent"]:
        eb[j["name"]].parent = eb[j["parent"]]
        eb[j["name"]].use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")
# The bone frames must equal the Aether3D frames.
worst = 0.0
for j in case["joints"]:
    m = np.array(arm_data.bones[j["name"]].matrix_local)[:3, :3]
    worst = max(worst, float(np.abs(m - np.array(j["frame"])).max()))
print("FRAME_DEVIATION", worst)

timings = {}
if given is None:
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    t0 = time.perf_counter()
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    timings["automatic_weights_s"] = time.perf_counter() - t0
else:
    W = np.load(given)
    names = case["weight_joints"]
    for c, n in enumerate(names):
        g = body.vertex_groups.new(name=n)
        nz = np.flatnonzero(W[:, c] > 0)
        for v in nz.tolist():
            g.add([v], float(W[v, c]), "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    body.parent = arm

# Weights out (dense, columns in case['weight_joints'] order).
names = case["weight_joints"]
col = {n: i for i, n in enumerate(names)}
W = np.zeros((len(pos), len(names)))
gname = {g.index: g.name for g in body.vertex_groups}
for v in mesh.vertices:
    for g in v.groups:
        n = gname[g.group]
        if n in col:
            W[v.index, col[n]] = g.weight
tag = "blender_heat" if given is None else "given"
np.save(case_dir / f"weights_{tag}.npy", W)
print("WEIGHTS", W.shape, "unweighted vertices", int((W.sum(1) <= 0).sum()))

armmod = [m for m in body.modifiers if m.type == "ARMATURE"][0]
cs = body.modifiers.new("CorrectiveSmooth", "CORRECTIVE_SMOOTH")
print("CS defaults", cs.factor, cs.iterations, cs.scale, cs.smooth_type, cs.rest_source,
      cs.use_only_smooth, cs.use_pin_boundary)
variants = {
    "lbs": (False, False),
    "dqs": (True, False),
    "lbs_cs": (False, True),
    "dqs_cs": (True, True),
}
outdir = case_dir / f"posed_{tag}"
outdir.mkdir(exist_ok=True)
depsgraph = bpy.context.evaluated_depsgraph_get()
for pose in [{"name": "rest", "rotations": {}}] + case["poses"]:
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    for jn, q in pose["rotations"].items():
        x, y, z, w = q
        arm.pose.bones[jn].rotation_quaternion = Quaternion((w, x, y, z))
    for vname, (dq, smooth) in variants.items():
        armmod.use_deform_preserve_volume = dq
        cs.show_viewport = smooth
        cs.show_render = smooth
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(depsgraph)
        me = ev.to_mesh()
        P = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", P)
        ev.to_mesh_clear()
        np.save(outdir / f"{pose['name']}__{vname}.npy", P.reshape(-1, 3))
print("TIMINGS", json.dumps(timings))
print("DONE", tag)
