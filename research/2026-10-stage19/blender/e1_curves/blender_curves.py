"""E1: Blender F-curve reference data (handles, smoothing, easing, extrapolation, modifiers).

Run: blender -b --factory-startup --python blender_curves.py -- OUT.json
Times are Blender frames; values are the F-curve's. Every case samples fcurve.evaluate on
a dense frame grid and records the keys' handles after Blender recalculated them.
"""

import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
scene.render.fps = 24

FRAMES = [1.0, 4.0, 13.0, 15.0, 30.0, 31.0, 50.0]
VALUES = [0.0, 2.0, -1.0, 3.0, 3.0, 0.5, 1.0]
GRID = [f / 4.0 for f in range(-40, 4 * 62 + 1)]  # frames -10 .. 62, step 0.25


def new_curve(name):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    act = bpy.data.actions.new(name)
    obj.animation_data_create()
    obj.animation_data.action = act
    # Blender 5.x layered actions: create the F-curve through the channelbag API.
    slot = act.slots.new(id_type="OBJECT", name=name)
    obj.animation_data.action_slot = slot
    layer = act.layers.new("Layer")
    strip = layer.strips.new(type="KEYFRAME")
    bag = strip.channelbag(slot, ensure=True)
    fc = bag.fcurves.new("location", index=0)
    return obj, fc


def keys(fc, frames=FRAMES, values=VALUES, left="AUTO_CLAMPED", right=None, interp="BEZIER"):
    fc.keyframe_points.add(len(frames))
    for kp, f, v in zip(fc.keyframe_points, frames, values):
        kp.co = (f, v)
        kp.interpolation = interp
        kp.handle_left_type = left
        kp.handle_right_type = right or left
    fc.update()


def dump(fc, extra=None):
    data = {
        "keys": [
            {
                "co": list(kp.co),
                "left": list(kp.handle_left),
                "right": list(kp.handle_right),
                "left_type": kp.handle_left_type,
                "right_type": kp.handle_right_type,
                "interpolation": kp.interpolation,
                "easing": kp.easing,
                "back": kp.back,
                "amplitude": kp.amplitude,
                "period": kp.period,
            }
            for kp in fc.keyframe_points
        ],
        "extrapolation": fc.extrapolation,
        "auto_smoothing": fc.auto_smoothing,
        "grid": GRID,
        "values": [fc.evaluate(f) for f in GRID],
    }
    if extra:
        data.update(extra)
    return data


cases = {}
# Defaults of a key inserted the ordinary way.
obj, fc = new_curve("defaults")
obj.location.x = 0.0
obj.keyframe_insert("location", index=0, frame=1)
cases["insert_defaults"] = {
    "handle_type": fc.keyframe_points[0].handle_left_type,
    "interpolation": fc.keyframe_points[0].interpolation,
    "auto_smoothing": fc.auto_smoothing,
    "extrapolation": fc.extrapolation,
    "pref_keyframe_new_handle_type": bpy.context.preferences.edit.keyframe_new_handle_type,
    "pref_keyframe_new_interpolation_type": bpy.context.preferences.edit.keyframe_new_interpolation_type,
}

for htype in ("AUTO", "AUTO_CLAMPED", "VECTOR"):
    for smoothing in ("NONE", "CONT_ACCEL"):
        for extrap in ("CONSTANT", "LINEAR"):
            name = f"{htype}_{smoothing}_{extrap}"
            obj, fc = new_curve(name)
            fc.auto_smoothing = smoothing
            fc.extrapolation = extrap
            keys(fc, left=htype)
            cases[name] = dump(fc)

# Value-scale invariance: the same keys with values x 10 (do handle time extents change?).
for htype in ("AUTO", "AUTO_CLAMPED"):
    obj, fc = new_curve(f"scaled_{htype}")
    keys(fc, values=[10 * v for v in VALUES], left=htype)
    cases[f"scaled10_{htype}"] = dump(fc)
    obj, fc = new_curve(f"tscaled_{htype}")
    keys(fc, frames=[2 * f for f in FRAMES], left=htype)
    cases[f"timescaled2_{htype}"] = dump(fc)

# Free handles reaching past the next key (correct_bezpart at evaluation).
obj, fc = new_curve("long_free")
keys(fc, frames=[1.0, 11.0, 21.0], values=[0.0, 1.0, 0.0], left="FREE")
kp = fc.keyframe_points
kp[0].handle_right = (18.0, 0.0)  # reaches past key 1 (frame 11)
kp[1].handle_left = (2.0, 1.0)
kp[1].handle_right = (14.0, 1.0)
kp[2].handle_left = (12.0, 0.0)
cases["long_free"] = dump(fc)
cases["long_free"]["stored_after"] = [[list(k.handle_left), list(k.handle_right)] for k in kp]

# Aligned and asymmetric free handles (broken tangents, long/short).
obj, fc = new_curve("free_asym")
keys(fc, frames=[1.0, 9.0, 25.0], values=[0.0, 4.0, 1.0], left="FREE")
kp = fc.keyframe_points
kp[0].handle_right = (6.0, 3.0)
kp[1].handle_left = (8.0, 2.0)  # short, steep
kp[1].handle_right = (20.0, 6.0)  # long, broken from the left
kp[2].handle_left = (24.0, -2.0)
cases["free_asym"] = dump(fc)

# Easing equations.
for interp in (
    "SINE",
    "QUAD",
    "CUBIC",
    "QUART",
    "QUINT",
    "EXPO",
    "CIRC",
    "BACK",
    "BOUNCE",
    "ELASTIC",
):
    for ease in ("EASE_IN", "EASE_OUT", "EASE_IN_OUT"):
        obj, fc = new_curve(f"{interp}_{ease}")
        keys(fc, frames=[1.0, 25.0, 31.0], values=[0.0, 10.0, 4.0], interp=interp)
        for kp in fc.keyframe_points:
            kp.easing = ease
        if interp == "BACK":
            fc.keyframe_points[0].back = 2.5
        if interp == "ELASTIC":
            fc.keyframe_points[0].amplitude = 2.0
            fc.keyframe_points[0].period = 4.0
        fc.update()
        cases[f"ease_{interp}_{ease}"] = dump(fc)
# AUTO easing (Blender picks per interpolation type).
obj, fc = new_curve("ease_auto")
keys(fc, frames=[1.0, 25.0], values=[0.0, 10.0], interp="BACK")
fc.keyframe_points[0].easing = "AUTO"
cases["ease_BACK_AUTO"] = dump(fc)


# Modifiers.
def with_mod(name, kind, setup, frames=(1.0, 9.0, 17.0), values=(0.0, 2.0, 0.5)):
    obj, fc = new_curve(name)
    keys(fc, frames=list(frames), values=list(values))
    m = fc.modifiers.new(kind)
    setup(m)
    fc.update()
    cases[name] = dump(fc, {"modifier": kind})


for mb in ("REPEAT", "REPEAT_OFFSET", "MIRROR", "NONE"):
    for ma in ("REPEAT", "REPEAT_OFFSET", "MIRROR"):

        def setup(m, mb=mb, ma=ma):
            m.mode_before = mb
            m.mode_after = ma
            m.cycles_before = 0
            m.cycles_after = 2

        with_mod(f"cycles_{mb}_{ma}", "CYCLES", setup)


def stepped(m):
    m.frame_step = 3.0
    m.frame_offset = 1.0


with_mod("stepped", "STEPPED", stepped)


def generator(m):
    m.mode = "POLYNOMIAL"
    m.poly_order = 2
    m.coefficients = (0.5, 0.1, -0.01)
    m.use_additive = True


with_mod("generator_additive", "GENERATOR", generator)


def fngen(m):
    m.function_type = "SIN"
    m.amplitude = 0.5
    m.phase_multiplier = 0.4
    m.phase_offset = 0.25
    m.value_offset = 0.1
    m.use_additive = True


with_mod("fn_sin_additive", "FNGENERATOR", fngen)


def limits(m):
    m.use_min_y = True
    m.min_y = 0.2
    m.use_max_y = True
    m.max_y = 1.5


with_mod("limits", "LIMITS", limits)


def noise(m):
    m.scale = 4.0
    m.strength = 0.5
    m.phase = 1.0
    m.offset = 0.0
    m.depth = 0


with_mod("noise", "NOISE", noise)


# A modifier restricted to a frame range with blend in/out.
def ranged(m):
    m.mode = "POLYNOMIAL"
    m.poly_order = 1
    m.coefficients = (1.0, 0.0)
    m.use_additive = True
    m.use_restricted_range = True
    m.frame_start = 5.0
    m.frame_end = 13.0
    m.blend_in = 2.0
    m.blend_out = 3.0


with_mod("generator_ranged", "GENERATOR", ranged)

json.dump(
    {"blender": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode(), "cases": cases},
    open(out, "w"),
    indent=1,
)
print("WROTE", out, len(cases))
