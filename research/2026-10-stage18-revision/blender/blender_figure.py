"""Blender reference experiment for Aether3D's figure case (run headless inside Blender):
    blender -b --factory-startup --python blender_figure.py -- CASE_DIR
Builds the case's mesh and bones (bone frames = the Aether3D joint frames), parents with
Automatic Weights (bone heat), saves the weights, and for every stress pose sets each pose
bone to the posed transform Aether3D computed (T @ rest), then saves the deformed positions
for LBS, DQS (Preserve Volume), LBS + Corrective Smooth (Blender's defaults) and LBS +
Corrective Smooth with 12 iterations."""
import json, sys, time
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
case_dir = Path(argv[0])
case = json.loads((case_dir / "case.json").read_text())
pos = np.load(case_dir / "positions.npy")
tris = np.load(case_dir / "triangles.npy")

bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata(pos.tolist(), [], tris.tolist())
mesh.update()
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
assert len(mesh.vertices) == len(pos)

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
    frame = np.array(j["frame"])
    b.align_roll(Vector(frame[:, 2].tolist()))
    b.use_deform = j["deform"]
    eb[j["name"]] = b
for j in case["joints"]:
    if j["parent"]:
        eb[j["name"]].parent = eb[j["parent"]]
        eb[j["name"]].use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")
worst = max(float(np.abs(np.array(arm_data.bones[j["name"]].matrix_local)[:3, :3]
                         - np.array(j["frame"])).max()) for j in case["joints"])
print("FRAME_DEVIATION", worst)

bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
t0 = time.perf_counter()
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
seconds = time.perf_counter() - t0
names = case["weight_joints"]
col = {n: i for i, n in enumerate(names)}
W = np.zeros((len(pos), len(names)))
gname = {g.index: g.name for g in body.vertex_groups}
for v in mesh.vertices:
    for g in v.groups:
        if gname[g.group] in col:
            W[v.index, col[gname[g.group]]] = g.weight
np.save(case_dir / "weights_blender_heat.npy", W)
print("WEIGHTS", W.shape, "unweighted", int((W.sum(1) <= 0).sum()), "seconds", round(seconds, 2))

armmod = [m for m in body.modifiers if m.type == "ARMATURE"][0]
cs = body.modifiers.new("CorrectiveSmooth", "CORRECTIVE_SMOOTH")
print("CS defaults", cs.factor, cs.iterations, cs.scale, cs.smooth_type, cs.rest_source)
variants = {"lbs": (False, None), "dqs": (True, None), "lbs_cs": (False, 5), "lbs_cs12": (False, 12)}
outdir = case_dir / "posed_blender"
outdir.mkdir(exist_ok=True)
order = []  # parents first
depth = {}
for j in case["joints"]:
    depth[j["name"]] = 0 if not j["parent"] else depth[j["parent"]] + 1
    order.append(j["name"])
levels = sorted(set(depth.values()))
for pose in [{"name": "rest", "matrices": {}}] + case["poses"]:
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for lv in levels:
        for name in order:
            if depth[name] != lv or name not in pose["matrices"]:
                continue
            T = Matrix(pose["matrices"][name])
            arm.pose.bones[name].matrix = T @ arm_data.bones[name].matrix_local
        bpy.context.view_layer.update()
    for vname, (dq, it) in variants.items():
        armmod.use_deform_preserve_volume = dq
        cs.show_viewport = cs.show_render = it is not None
        if it is not None:
            cs.iterations = it
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg)
        me = ev.to_mesh()
        P = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", P)
        ev.to_mesh_clear()
        np.save(outdir / f"{pose['name']}__{vname}.npy", P.reshape(-1, 3))
json.dump({"blender": bpy.app.version_string, "automatic_weights_s": seconds,
           "frame_deviation": worst}, open(case_dir / "blender_run.json", "w"))
print("DONE")
