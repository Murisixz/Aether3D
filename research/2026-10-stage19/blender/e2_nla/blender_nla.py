"""E2: Blender NLA blending: strip crossfade ramp shape, blend types, quaternion blending."""

import json
import math
import sys

import bpy
from mathutils import Quaternion

out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene


def action_with(name, path, index_values):
    act = bpy.data.actions.new(name)
    slot = act.slots.new(id_type="OBJECT", name=name)
    bag = act.layers.new("L").strips.new(type="KEYFRAME").channelbag(slot, ensure=True)
    for index, keys in index_values.items():
        fc = bag.fcurves.new(path, index=index)
        fc.keyframe_points.add(len(keys))
        for kp, (f, v) in zip(fc.keyframe_points, keys):
            kp.co = (f, v)
            kp.interpolation = "LINEAR"
        fc.update()
    return act, slot


obj = bpy.data.objects.new("O", None)
scene.collection.objects.link(obj)
obj.rotation_mode = "QUATERNION"
ad = obj.animation_data_create()
qa = Quaternion((0, 0, 1), 0.0)
qb = Quaternion((0, 0, 1), math.radians(170))
A, sa = action_with("A", "location", {0: [(1, 0.0), (60, 0.0)]})
B, sb = action_with("B", "location", {0: [(1, 1.0), (60, 1.0)]})
QA, sqa = action_with("QA", "rotation_quaternion", {i: [(1, qa[i]), (60, qa[i])] for i in range(4)})
QB, sqb = action_with("QB", "rotation_quaternion", {i: [(1, qb[i]), (60, qb[i])] for i in range(4)})
results = {"blender": bpy.app.version_string}
for blend in ("REPLACE", "COMBINE", "ADD"):
    ad.action = None
    for t in list(ad.nla_tracks):
        ad.nla_tracks.remove(t)
    t1 = ad.nla_tracks.new()
    s1 = t1.strips.new("A", 1, A)
    s1.action_slot = sa
    s1q = t1.strips.new("QA", 70, QA)
    s1q.action_slot = sqa
    t2 = ad.nla_tracks.new()
    s2 = t2.strips.new("B", 21, B)
    s2.action_slot = sb
    s2.blend_type = blend
    s2.blend_in = 20.0
    s2.blend_out = 0.0
    s2q = t2.strips.new("QB", 90, QB)
    s2q.action_slot = sqb
    s2q.blend_type = blend
    s2q.blend_in = 20.0
    rows = []
    for f10 in range(10, 1300, 5):
        f = f10 / 10.0
        scene.frame_set(int(f), subframe=f - int(f))
        rows.append([f, obj.location.x, *obj.rotation_quaternion])
    results[blend] = rows
json.dump(results, open(out, "w"), indent=0)
print("WROTE", out)
