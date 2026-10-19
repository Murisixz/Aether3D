"""Rigify's generated human rig, summarized (run inside Blender: blender -b
--factory-startup --python rigify_dump.py): its bones, deforming bones and their B-bone
segments, constraint types, custom properties and controls -- the structure the revision's
humanoid template was compared with."""
import bpy, addon_utils, json, collections
addon_utils.enable("rigify", default_set=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("rigify", default_set=True)
bpy.ops.object.armature_human_metarig_add()
meta = bpy.context.object
meta_bones = {b.name: (tuple(round(v, 3) for v in b.head_local), tuple(round(v, 3) for v in b.tail_local), b.parent.name if b.parent else None) for b in meta.data.bones}
bpy.ops.pose.rigify_generate()
rig = bpy.data.objects.get("rig") or [o for o in bpy.data.objects if o.type == "ARMATURE" and o != meta][0]
bones = rig.data.bones
defs = [b for b in bones if b.use_deform]
print("TOTAL_BONES", len(bones), "DEFORM", len(defs))
print("DEF", [ (b.name, b.bbone_segments) for b in defs])
kinds = collections.Counter(b.name.split("-")[0] if "-" in b.name else "CTRL" for b in bones)
print("KINDS", dict(kinds))
cons = collections.Counter()
for pb in rig.pose.bones:
    for c in pb.constraints:
        cons[c.type] += 1
print("CONSTRAINTS", dict(cons))
props = {}
for pb in rig.pose.bones:
    keys = [k for k in pb.keys() if not k.startswith("_")]
    if keys:
        props[pb.name] = keys
print("PROPS", json.dumps(props)[:3000])
ctrl = [b.name for b in bones if not b.name.startswith(("DEF-", "ORG-", "MCH-"))]
print("CONTROLS", ctrl)
print("META", json.dumps({k: v for k, v in meta_bones.items() if any(s in k for s in ("spine", "thigh", "shin", "foot", "toe", "heel", "upper_arm", "forearm", "hand", "shoulder", "breast", "pelvis"))}))
