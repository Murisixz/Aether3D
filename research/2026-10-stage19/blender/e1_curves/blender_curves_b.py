"""E1b: clamping under Continuous Acceleration, cyclic smoothing, mixed handle types,
elastic defaults, Blender's default smoothing preference.  Same output format as E1."""

import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
GRID = [f / 4.0 for f in range(-40, 4 * 62 + 1)]


def new_curve(name):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    act = bpy.data.actions.new(name)
    obj.animation_data_create()
    obj.animation_data.action = act
    slot = act.slots.new(id_type="OBJECT", name=name)
    obj.animation_data.action_slot = slot
    bag = act.layers.new("L").strips.new(type="KEYFRAME").channelbag(slot, ensure=True)
    return bag.fcurves.new("location", index=0)


def build(frames, values, types, smoothing, extrap="CONSTANT", interp="BEZIER", cyclic=False):
    fc = new_curve("c")
    fc.auto_smoothing = smoothing
    fc.extrapolation = extrap
    if cyclic:
        fc.modifiers.new("CYCLES")
    fc.keyframe_points.add(len(frames))
    for kp, f, v, t in zip(fc.keyframe_points, frames, values, types):
        kp.co = (f, v)
        kp.interpolation = interp
        kp.handle_left_type = t
        kp.handle_right_type = t
    fc.update()
    return fc


def dump(fc):
    return {
        "keys": [
            {
                "co": list(k.co),
                "left": list(k.handle_left),
                "right": list(k.handle_right),
                "left_type": k.handle_left_type,
                "right_type": k.handle_right_type,
                "interpolation": k.interpolation,
                "easing": k.easing,
                "back": k.back,
                "amplitude": k.amplitude,
                "period": k.period,
            }
            for k in fc.keyframe_points
        ],
        "extrapolation": fc.extrapolation,
        "auto_smoothing": fc.auto_smoothing,
        "cyclic": any(m.type == "CYCLES" for m in fc.modifiers),
        "grid": GRID,
        "values": [fc.evaluate(f) for f in GRID],
    }


cases = {
    "prefs": {
        "fcurve_new_auto_smoothing": bpy.context.preferences.edit.fcurve_new_auto_smoothing,
        "keyframe_new_handle_type": bpy.context.preferences.edit.keyframe_new_handle_type,
    }
}
MONO_F = [1.0, 5.0, 9.0, 20.0, 22.0, 30.0, 34.0, 41.0, 50.0]
MONO_V = [0.0, 0.2, 1.8, 2.0, 2.1, 5.0, 5.0, 4.0, 0.0]
UNIF_F = [1.0, 7.0, 13.0, 19.0, 25.0, 31.0, 37.0]
UNIF_V = [0.0, 1.0, 1.2, 3.0, 2.0, 2.2, 0.0]
for label, frames, values in (("mono", MONO_F, MONO_V), ("uniform", UNIF_F, UNIF_V)):
    for t in ("AUTO", "AUTO_CLAMPED"):
        for s in ("NONE", "CONT_ACCEL"):
            for e in ("CONSTANT", "LINEAR"):
                cases[f"{label}_{t}_{s}_{e}"] = dump(build(frames, values, [t] * len(frames), s, e))
# Cyclic: end keys equal (a loop) and unequal; cycles modifier makes handles cycle-aware.
LOOP_F = [1.0, 6.0, 14.0, 17.0, 25.0]
for label, values in (
    ("loop_equal", [0.0, 1.0, 0.3, 0.9, 0.0]),
    ("loop_unequal", [0.0, 1.0, 0.3, 0.9, 0.6]),
):
    for t in ("AUTO", "AUTO_CLAMPED"):
        for s in ("NONE", "CONT_ACCEL"):
            cases[f"{label}_{t}_{s}"] = dump(build(LOOP_F, values, [t] * 5, s, cyclic=True))
# Mixed handle types inside an automatic run: a VECTOR and a FREE key lock the solve.
MIX_F = [1.0, 5.0, 12.0, 16.0, 24.0, 30.0]
MIX_V = [0.0, 1.0, 1.5, 3.0, 3.5, 6.0]
for s in ("NONE", "CONT_ACCEL"):
    fc = build(
        MIX_F,
        MIX_V,
        ["AUTO_CLAMPED", "AUTO_CLAMPED", "VECTOR", "AUTO_CLAMPED", "AUTO_CLAMPED", "AUTO_CLAMPED"],
        s,
    )
    cases[f"mixed_vector_{s}"] = dump(fc)
    fc = build(MIX_F, MIX_V, ["AUTO", "AUTO", "AUTO", "FREE", "AUTO", "AUTO"], s)
    kp = fc.keyframe_points[3]
    kp.handle_left = (14.0, 3.6)
    kp.handle_right = (19.0, 2.4)
    fc.update()
    cases[f"mixed_free_{s}"] = dump(fc)
# A constant and a linear segment inside a Bezier run (neighbouring interpolation).
for s in ("NONE", "CONT_ACCEL"):
    fc = build(MIX_F, MIX_V, ["AUTO_CLAMPED"] * 6, s)
    fc.keyframe_points[2].interpolation = "LINEAR"
    fc.keyframe_points[4].interpolation = "CONSTANT"
    fc.update()
    cases[f"mixed_interp_{s}"] = dump(fc)
# Elastic with Blender's defaults (amplitude 0, period 0) and with amplitude > change.
for label, amp, per in (
    ("defaults", 0.0, 0.0),
    ("amp15_per6", 15.0, 6.0),
    ("amp15_per0", 15.0, 0.0),
):
    for ease in ("EASE_IN", "EASE_OUT", "EASE_IN_OUT"):
        fc = build(
            [1.0, 25.0, 31.0], [0.0, 10.0, 4.0], ["AUTO_CLAMPED"] * 3, "NONE", interp="ELASTIC"
        )
        for k in fc.keyframe_points:
            k.easing = ease
            k.amplitude = amp
            k.period = per
        fc.update()
        cases[f"elastic_{label}_{ease}"] = dump(fc)
for back in (0.0, 1.70158, 2.5):
    for ease in ("EASE_IN", "EASE_OUT", "EASE_IN_OUT"):
        fc = build([1.0, 25.0, 31.0], [0.0, 10.0, 4.0], ["AUTO_CLAMPED"] * 3, "NONE", interp="BACK")
        for k in fc.keyframe_points:
            k.easing = ease
            k.back = back
        fc.update()
        cases[f"back_{back}_{ease}"] = dump(fc)
json.dump({"blender": bpy.app.version_string, "cases": cases}, open(out, "w"), indent=1)
print("WROTE", out, len(cases))
