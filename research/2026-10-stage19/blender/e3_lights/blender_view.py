"""E3b: display transforms on exposure ramps (scene-linear emission patches -> 16-bit PNG)
and the angular profile of area-light spread."""

import json
import math
import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1 :]
out, work = args[0], args[1]
os.makedirs(work, exist_ok=True)
STOPS = [s / 2.0 for s in range(-24, 25)]  # -12 .. +12 EV around middle grey 0.18
HUES = {
    "grey": (1, 1, 1),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "orange": (1.0, 0.4, 0.05),
    "cyan": (0.0, 0.8, 1.0),
    "skin": (0.8, 0.5, 0.35),
    "purple": (0.5, 0.1, 1.0),
}


def base_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 4
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sc.world = w
    return sc


def ramp_scene():
    sc = base_scene()
    nx, ny = len(STOPS), len(HUES)
    sc.render.resolution_x = nx * 4
    sc.render.resolution_y = ny * 4
    cam = bpy.data.cameras.new("C")
    cam.type = "ORTHO"
    cam.ortho_scale = float(nx)
    co = bpy.data.objects.new("C", cam)
    co.location = (nx / 2.0, ny / 2.0, 10.0)
    sc.collection.objects.link(co)
    sc.camera = co
    for j, (hue, rgb) in enumerate(HUES.items()):
        for i, ev in enumerate(STOPS):
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(i + 0.5, ny - j - 0.5, 0))
            p = bpy.context.object
            m = bpy.data.materials.new(f"{hue}{i}")
            m.use_nodes = True
            nt = m.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            e = nt.nodes.new("ShaderNodeEmission")
            e.inputs["Color"].default_value = (*rgb, 1)
            e.inputs["Strength"].default_value = 0.18 * 2.0**ev
            o = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(e.outputs[0], o.inputs[0])
            p.data.materials.append(m)
    return sc


def read_patches(path, nx, ny):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = list(img.pixels)
    vals = {}
    for j, hue in enumerate(HUES):
        row = []
        for i in range(nx):
            x = i * 4 + 2
            y = (ny - 1 - j) * 4 + 2
            k = (y * w + x) * 4
            row.append(px[k : k + 3])
        vals[hue] = row
    bpy.data.images.remove(img)
    return vals


results = {"blender": bpy.app.version_string, "stops": STOPS, "hues": HUES, "transforms": {}}
sc = ramp_scene()
views = ["Standard", "AgX", "Filmic", "Khronos PBR Neutral", "ACES 1.3", "ACES 2.0", "Raw"]
results["views"] = views
for view in views:
    sc.view_settings.view_transform = view
    looks = ["None"] + (
        [
            "AgX - Punchy",
            "AgX - Base Contrast",
            "AgX - High Contrast",
            "AgX - Medium High Contrast",
            "AgX - Medium Low Contrast",
            "AgX - Low Contrast",
            "AgX - Very High Contrast",
            "AgX - Very Low Contrast",
            "AgX - Greyscale",
        ]
        if view == "AgX"
        else []
    )
    for look in looks:
        if view not in ("AgX",) and look != "None":
            continue
        try:
            sc.view_settings.look = look
        except TypeError as exc:
            results["transforms"][f"{view}|{look}"] = {"error": str(exc)}
            continue
        sc.view_settings.exposure = 0.0
        sc.view_settings.gamma = 1.0
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_depth = "16"
        sc.render.image_settings.color_mode = "RGB"
        path = os.path.join(work, f"view_{view}_{look}.png".replace(" ", "_"))
        sc.render.filepath = path
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            results["transforms"][f"{view}|{look}"] = {"error": str(exc)}
            continue
        results["transforms"][f"{view}|{look}"] = read_patches(path, len(STOPS), len(HUES))
results["display_device"] = sc.display_settings.display_device

# Area light spread profile: a tiny square light (2 cm), 1 m above a white plane.
sc2 = {}
for spread in (180, 120, 90, 60, 30, 10):
    sc = base_scene()
    sc.cycles.samples = 4096
    sc.cycles.max_bounces = 0
    sc.render.resolution_x = 65
    sc.render.resolution_y = 1
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.view_settings.view_transform = "Standard"
    bpy.ops.mesh.primitive_plane_add(size=200.0)
    p = bpy.context.object
    m = bpy.data.materials.new("W")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (1, 1, 1, 1)
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(d.outputs[0], o.inputs[0])
    p.data.materials.append(m)
    cam = bpy.data.cameras.new("C")
    cam.type = "ORTHO"
    cam.ortho_scale = 8.0
    co = bpy.data.objects.new("C", cam)
    co.location = (4.0, 0.0, 20.0)
    sc.collection.objects.link(co)
    sc.camera = co
    ld = bpy.data.lights.new("A", "AREA")
    ld.shape = "SQUARE"
    ld.size = 0.02
    ld.energy = 1.0
    ld.spread = math.radians(spread)
    lo = bpy.data.objects.new("A", ld)
    lo.location = (0, 0, 1.0)
    sc.collection.objects.link(lo)
    path = os.path.join(work, f"spread{spread}.exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = list(img.pixels)
    sc2[spread] = [px[4 * i] for i in range(65)]
    bpy.data.images.remove(img)
results["spread_profile"] = {
    "x": [(i + 0.5) * 8.0 / 65 for i in range(65)],
    "values": sc2,
    "light": {"size": 0.02, "height": 1.0, "power": 1.0},
}
json.dump(results, open(out, "w"))
print("WROTE", out)
