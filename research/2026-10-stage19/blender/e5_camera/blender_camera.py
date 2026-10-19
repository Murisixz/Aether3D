"""E5: Blender camera lens shift and depth of field, measured in pixels (Cycles, CPU).

A tiny emissive sphere is rendered against black: its image centroid gives the projection
(fov from focal length and sensor, the effect of shift_x/shift_y), and with depth of field
its blur disk gives the circle of confusion (diameter, aperture blades, ratio)."""
import json
import math
import os
import sys

import bpy

out, work = sys.argv[sys.argv.index("--") + 1 :][:2]
os.makedirs(work, exist_ok=True)


def scene(res=(320, 180)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 512
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0
    sc.cycles.filter_width = 0.01
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    w = bpy.data.worlds.new("W"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sc.world = w
    cam = bpy.data.cameras.new("C")
    co = bpy.data.objects.new("C", cam)
    co.location = (0.0, -4.0, 0.0)
    co.rotation_euler = (math.radians(90), 0.0, 0.0)  # looks along +Y
    sc.collection.objects.link(co)
    sc.camera = co
    return sc, cam


def emitter(loc, radius=0.004):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc, segments=16, ring_count=8)
    o = bpy.context.object
    m = bpy.data.materials.new("E"); m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Strength"].default_value = 2000.0
    o2 = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(e.outputs[0], o2.inputs[0])
    o.data.materials.append(m)


def measure(name):
    sc = bpy.context.scene
    path = os.path.join(work, name + ".exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    w, h = img.size
    px = list(img.pixels)
    tot = sx = sy = 0.0
    lit = 0
    peak = max(px[(i * 4)] for i in range(w * h))
    xs = []
    for y in range(h):
        for x in range(w):
            v = px[(y * w + x) * 4]
            if v > 0:
                tot += v; sx += v * (x + 0.5); sy += v * (y + 0.5)
            if v > 0.02 * peak:
                lit += 1
    bpy.data.images.remove(img)
    # centroid in pixels from the left / from the top (Blender's image rows go bottom-up)
    return {"cx": sx / tot, "cy": h - sy / tot, "lit_pixels": lit, "total": tot, "w": w, "h": h}


R = {"blender": bpy.app.version_string, "cases": {}}
for shift in ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.2, -0.05)):
    sc, cam = scene()
    cam.lens, cam.sensor_width, cam.sensor_fit = 50.0, 36.0, "AUTO"
    cam.shift_x, cam.shift_y = shift
    emitter((0.3, 0.0, 0.2))
    R["cases"][f"shift_{shift[0]}_{shift[1]}"] = measure(f"shift{shift}")
# portrait frame: the larger dimension is the height
sc, cam = scene(res=(180, 320))
cam.lens, cam.sensor_width, cam.sensor_fit = 50.0, 36.0, "AUTO"
cam.shift_x = 0.1
emitter((0.3, 0.0, 0.2))
R["cases"]["portrait_shift_0.1_0"] = measure("portrait")
# depth of field: focus at 2 m, emitter at 4 m (on axis), 50 mm f/2, f/4; 6 blades; ratio 2
for name, fstop, blades, ratio in (("dof_f2", 2.0, 0, 1.0), ("dof_f4", 4.0, 0, 1.0),
                                   ("dof_f2_6blades", 2.0, 6, 1.0), ("dof_f2_ratio2", 2.0, 0, 2.0)):
    sc, cam = scene(res=(640, 640))
    cam.lens, cam.sensor_width, cam.sensor_fit = 50.0, 36.0, "AUTO"
    cam.dof.use_dof = True
    cam.dof.focus_distance = 0.5
    cam.dof.aperture_fstop = fstop
    cam.dof.aperture_blades = blades
    cam.dof.aperture_ratio = ratio
    sc.cycles.samples = 2048
    emitter((0.0, 0.0, 0.0))  # 4 m in front
    r = measure(name)
    # the blur disk's extent along x and y
    path = os.path.join(work, name + ".exr")
    img = bpy.data.images.load(path)
    w, h = img.size
    px = list(img.pixels)
    peak = max(px[(i * 4)] for i in range(w * h))
    cols = [x for y in range(h) for x in range(w) if px[(y * w + x) * 4] > 0.1 * peak]
    rows = [y for y in range(h) for x in range(w) if px[(y * w + x) * 4] > 0.1 * peak]
    r["width_px"] = max(cols) - min(cols) + 1
    r["height_px"] = max(rows) - min(rows) + 1
    bpy.data.images.remove(img)
    R["cases"][name] = r
json.dump(R, open(out, "w"), indent=1)
print("WROTE", out)
