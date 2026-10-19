"""E3: Cycles radiometry of Blender's lights, measured on a white Lambertian plane.

Each case renders a small orthographic top view of a large white diffuse plane (albedo 1)
with one light, Cycles on the CPU, many samples, no denoising, a tiny pixel filter, and
reads the linear (scene-referred) radiance back from an OpenEXR file through bpy.
"""

import json
import math
import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1 :]
out, work = args[0], args[1]
os.makedirs(work, exist_ok=True)
RES = 33


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 2048
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0  # direct light only
    sc.cycles.seed = 0
    sc.cycles.filter_width = 0.01
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sc.world = world
    return sc


def plane(size=200.0, z=0.0, albedo=1.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z))
    p = bpy.context.object
    mat = bpy.data.materials.new("White")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (albedo, albedo, albedo, 1)
    d.inputs["Roughness"].default_value = 0.0
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(d.outputs[0], o.inputs[0])
    p.data.materials.append(mat)
    return p


def camera(height=50.0, scale=1.0, at=(0.0, 0.0)):
    cam = bpy.data.cameras.new("C")
    cam.type = "ORTHO"
    cam.ortho_scale = scale
    cam.clip_end = 1000.0
    co = bpy.data.objects.new("C", cam)
    co.location = (at[0], at[1], height)
    bpy.context.scene.collection.objects.link(co)
    bpy.context.scene.camera = co
    return co


def light(kind, **kw):
    ld = bpy.data.lights.new("L", kind)
    for k, v in kw.items():
        if k not in ("location", "rotation"):
            setattr(ld, k, v)
    lo = bpy.data.objects.new("L", ld)
    lo.location = kw.get("location", (0, 0, 0))
    lo.rotation_euler = kw.get("rotation", (0, 0, 0))
    bpy.context.scene.collection.objects.link(lo)
    return lo


def render(name):
    sc = bpy.context.scene
    path = os.path.join(work, name + ".exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = list(img.pixels)
    w, h = img.size
    rows = [[px[(y * w + x) * 4 : (y * w + x) * 4 + 3] for x in range(w)] for y in range(h)]
    bpy.data.images.remove(img)
    return rows


def center(rows, r=2):
    c = len(rows) // 2
    vals = [rows[y][x] for y in range(c - r, c + r + 1) for x in range(c - r, c + r + 1)]
    return [sum(v[i] for v in vals) / len(vals) for i in range(3)]


results = {
    "blender": bpy.app.version_string,
    "light_props": sorted(bpy.types.Light.bl_rna.properties.keys()),
    "area_props": sorted(bpy.types.AreaLight.bl_rna.properties.keys()),
    "spot_props": sorted(bpy.types.SpotLight.bl_rna.properties.keys()),
    "cases": {},
}
C = results["cases"]

# Sun, straight down, strength 1 W/m^2 (and angle 0 / 30 degrees).
for angle in (0.0, math.radians(30)):
    reset()
    plane()
    camera()
    light("SUN", energy=1.0, angle=angle)
    C[f"sun_E1_angle{round(math.degrees(angle))}"] = center(render(f"sun{angle}"))
# Point light, power 1000 W, 2 m above (radius 0 and 0.5 m).
for radius in (0.0, 0.5):
    reset()
    plane()
    camera()
    light("POINT", energy=1000.0, shadow_soft_size=radius, location=(0, 0, 2.0))
    C[f"point_P1000_h2_r{radius}"] = center(render(f"point{radius}"))
# Point light radial profile (a wide view): irradiance vs horizontal distance.
reset()
plane()
camera(scale=8.0)
light("POINT", energy=1000.0, shadow_soft_size=0.0, location=(0, 0, 2.0))
C["point_profile_scale8"] = render("point_profile")
# Spot light: power 1000 W, 2 m up, size 60 degrees, blend 0 / 0.3 (radial profiles).
for blend in (0.0, 0.3):
    reset()
    plane()
    camera(scale=4.0)
    light(
        "SPOT",
        energy=1000.0,
        spot_size=math.radians(60),
        spot_blend=blend,
        shadow_soft_size=0.0,
        location=(0, 0, 2.0),
    )
    C[f"spot_P1000_h2_size60_blend{blend}"] = render(f"spot{blend}")
# Area lights: power 100 W, 1 m above, facing down (-Z): square 0.5, rectangle 0.5x1,
# disk 0.5, ellipse 0.5x1; spread 180 / 30 degrees.
for shape, size, size_y in (
    ("SQUARE", 0.5, 0.5),
    ("RECTANGLE", 0.5, 1.0),
    ("DISK", 0.5, 0.5),
    ("ELLIPSE", 0.5, 1.0),
):
    for spread in (180.0, 30.0):
        reset()
        plane()
        camera(scale=4.0)
        light(
            "AREA",
            energy=100.0,
            shape=shape,
            size=size,
            size_y=size_y,
            spread=math.radians(spread),
            location=(0, 0, 1.0),
        )
        C[f"area_{shape}_P100_h1_spread{int(spread)}"] = render(f"area{shape}{spread}")
# Emission: a plane with an Emission shader of strength 1 seen directly (radiance check).
reset()
camera()
bpy.ops.mesh.primitive_plane_add(size=10.0)
p = bpy.context.object
mat = bpy.data.materials.new("E")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
e = nt.nodes.new("ShaderNodeEmission")
e.inputs["Strength"].default_value = 1.0
o = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(e.outputs[0], o.inputs[0])
p.data.materials.append(mat)
C["emission_strength1"] = center(render("emission"))
# Blackbody colors (emission shader with a Blackbody color, Standard view, linear EXR).
temps = [
    800,
    1000,
    1500,
    1850,
    2000,
    2700,
    3000,
    3200,
    4000,
    4500,
    5000,
    5500,
    6000,
    6500,
    7000,
    7500,
    8000,
    10000,
    12000,
    15000,
    20000,
    40000,
]
bb = {}
for T in temps:
    reset()
    camera()
    bpy.ops.mesh.primitive_plane_add(size=10.0)
    p = bpy.context.object
    mat = bpy.data.materials.new("BB")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    b = nt.nodes.new("ShaderNodeBlackbody")
    b.inputs["Temperature"].default_value = float(T)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Strength"].default_value = 1.0
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(b.outputs[0], e.inputs["Color"])
    nt.links.new(e.outputs[0], o.inputs[0])
    p.data.materials.append(mat)
    bpy.context.scene.cycles.samples = 16
    bb[T] = center(render(f"bb{T}"))
C["blackbody"] = bb
# A light's own temperature property, if Blender 5.2 has one.
if "temperature" in bpy.types.Light.bl_rna.properties.keys():
    tl = {}
    for T in (2700, 6500):
        reset()
        plane()
        camera()
        lo = light("SUN", energy=1.0)
        lo.data.use_temperature = True
        lo.data.temperature = float(T)
        tl[T] = center(render(f"suntemp{T}"))
    C["sun_temperature"] = tl
json.dump(results, open(out, "w"))
print("WROTE", out)
