"""E6: Blender 5.2's equirectangular world orientation and strength.

An HDR "UV image" (R = u, G = v of each pixel centre, B = 0.25) lights nothing; tiny-FOV
cameras look along known world directions and read back the background radiance, which
names the (u, v) Blender samples for that direction. A second case measures a white plane's
radiance under a uniform image of 1 with strength 2 (expected 2: E = pi L, L_out = E / pi).
"""
import json, math, os, sys
import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
out, work = args[0], args[1]
os.makedirs(work, exist_ok=True)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01
    sc.render.resolution_x = 4
    sc.render.resolution_y = 4
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    return sc


def uv_image(w=256, h=128):
    img = bpy.data.images.new("UV", w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"  # (before the pixels: it regenerates them)
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    px = np.zeros((h, w, 4), np.float32)
    px[..., 0] = u[None, :]
    px[..., 1] = v[:, None]
    px[..., 2] = 0.25
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.reshape(-1))  # row 0 = bottom of the image in Blender
    img.pack()
    return img


def world_with(img, strength=1.0, rotation_z=0.0):
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    env.interpolation = "Closest"
    if rotation_z:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, math.radians(rotation_z))
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    return world


def render_dir(sc, d, name):
    cam_data = bpy.data.cameras.new(name)
    cam_data.angle = math.radians(0.05)
    cam = bpy.data.objects.new(name, cam_data)
    sc.collection.objects.link(cam)
    d = np.asarray(d, float) / np.linalg.norm(d)
    # aim the camera's -Z along d (explicit frame: columns right, up, back)
    from mathutils import Matrix
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    back = -d
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    upv = np.cross(back, right)
    m = Matrix.Identity(4)
    for r in range(3):
        m[r][0], m[r][1], m[r][2] = right[r], upv[r], back[r]
    cam.matrix_world = m
    sc.camera = cam
    bpy.context.view_layer.update()
    path = os.path.join(work, name + ".exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path)
    px = np.array(im.pixels[:], np.float32).reshape(-1, 4)
    return px[:, :3].mean(axis=0).tolist()


results = {"blender": bpy.app.version_string, "directions": {}, "rotated": {}}
sc = reset()
img = uv_image()
sc.world = world_with(img)
dirs = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
        "+Z": (0, 0, 1), "-Z": (0, 0, -1), "XY45": (1, 1, 0), "X+Z30": (math.cos(math.radians(30)), 0, math.sin(math.radians(30)))}
for k, d in dirs.items():
    results["directions"][k] = render_dir(sc, d, "d_" + k.replace("+", "p").replace("-", "m"))
sc.world = world_with(img, rotation_z=90.0)
for k in ("+X", "+Y"):
    results["rotated"][k] = render_dir(sc, dirs[k], "r_" + k.replace("+", "p"))
# strength: a white diffuse plane under a uniform image of 1, strength 2
sc = reset()
sc.cycles.samples = 1024
sc.cycles.max_bounces = 0
white = bpy.data.images.new("White", 8, 4, float_buffer=True)
white.pixels.foreach_set(np.ones(8 * 4 * 4, np.float32))
sc.world = world_with(white, strength=2.0)
bpy.ops.mesh.primitive_plane_add(size=0.2)
plane = bpy.context.object
mat = bpy.data.materials.new("D")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
dn = nt.nodes.new("ShaderNodeBsdfDiffuse")
dn.inputs["Color"].default_value = (1, 1, 1, 1)
o = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(dn.outputs[0], o.inputs[0])
plane.data.materials.append(mat)
cam_data = bpy.data.cameras.new("Top")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 0.05
cam = bpy.data.objects.new("Top", cam_data)
cam.location = (0, 0, 1)
sc.collection.objects.link(cam)
sc.camera = cam
path = os.path.join(work, "strength.exr")
sc.render.filepath = path
bpy.ops.render.render(write_still=True)
im = bpy.data.images.load(path)
results["plane_strength2"] = np.array(im.pixels[:], np.float32).reshape(-1, 4)[:, :3].mean(axis=0).tolist()
json.dump(results, open(out, "w"), indent=1)
print("done", out)
