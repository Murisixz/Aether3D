"""Render the Minecraft-inspired procedural castle with Aether3D's own renderer.

Run from the repository root (booleans need ``pip install 'aether3d[boolean]'``; the
JPEG texture needs ``aether3d[images]``, without it the keep gets procedural planks):

    python examples/minecraft_castle_render.py [OUTPUT_DIR] [--width W] [--height H]
        [--samples N] [--blender]

1. Builds the procedural castle of ``examples/minecraft_castle.py`` and dresses
   it non-destructively: the watchtower roof becomes its own object (by
   disabling one boolean modifier), and a path, blocky trees (shared meshes),
   a sky, sunlight, fill light and two torches are added.
2. Gives it materials (Aether3D 0.7): stone-brick walls and towers and a
   plank keep from the example image textures (``examples/textures``, a PNG
   color map with a normal map and a JPEG), projected in object space
   (triplanar, so the procedural geometry needs no UVs); procedural shingles,
   cobblestones, grass, bark and leaves; and glowing torch flames.
3. Frames a camera on the castle and renders it with Aether3D's CPU renderer
   (default 1920x1080, 3x3 samples per pixel) to
   ``OUTPUT_DIR/minecraft_castle_render.png`` (default ``examples/output``),
   and saves the scene as ``minecraft_castle_render.aether`` (its textures are
   copied next to it, into ``textures/``). The same castle in Aether3D 0.6's
   flat colors is rendered too, and both are put side by side at half size
   (``castle_flat_vs_materials.png``).
4. Checks the image: every part of the castle is visible, the parts keep
   their distinct colors, the textures add detail, nothing is reported, the
   saved project re-renders identically, and the ``aether3d render`` command
   reproduces the same pixels.
5. With ``--blender`` (optional external verification): exports the scene to
   GLB, renders the same camera in Blender (headless, Workbench) and compares
   the object silhouettes. Blender is never used to make the PNG.

Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from minecraft_castle import build_castle  # noqa: E402

from aether3d import Expr, Scene  # noqa: E402
from aether3d.materials import jpeg_available  # noqa: E402

COLORS = {
    "Ground": "#4f8a33",
    "Curtain Wall": "#8f8f88",
    "Towers": "#a3a299",
    "Keep": "#8c7356",
    "Watchtower": "#7c7b75",
    "Watchtower Roof": "#b83a2a",
    "Path": "#8f7c62",
}
TREES = [(-17, -14, 1.0), (16.5, -13, 1.15), (-17.5, 6, 0.9), (17, 9, 1.05), (-9, -17, 0.8)]


TEXTURES = HERE / "textures"
"""The example textures (``examples/make_textures.py``)."""


def build_scene(
    width: int = 1920,
    height: int = 1080,
    samples: int = 3,
    *,
    materials: bool = False,
    textures: Path = TEXTURES,
) -> Scene:
    """The dressed castle: flat colors (as in Aether3D 0.6), or ``materials`` (textures from
    ``textures``)."""
    scene = Scene("Minecraft Castle")
    build_castle(scene)
    proc = scene.procedural
    width_m = proc.info()["parameters"]["castle_width"]["resolved"]

    # The watchtower roof is unioned into the watchtower mesh by a boolean; disable that
    # modifier (non-destructive) and show the same roof geometry as its own red object.
    proc.geometry("Watchtower").modifier("Roof").set(enabled=False)
    roof = scene.add_geometry("Watchtower Roof", source="Roof")
    roof.add_modifier(
        "transform", translation=Expr("(0, 0, wall_height * 1.1 + tower_radius * 0.9 + 0.5)")
    )
    scene.add_object("Watchtower Roof", geometry=roof, parent=scene.get_by_name("Watchtower"))

    path = scene.add_geometry(
        "Path", "box", width=Expr("gate_width"), depth=Expr("castle_width * 0.3"), height=0.06
    )
    path.add_modifier(
        "transform", translation=Expr("(0, -castle_width / 2 - castle_width * 0.15, 0.03)")
    )
    scene.add_object("Path", geometry=path)

    # Trees: two shared meshes, instanced five times with their own transforms.
    trunk = scene.add_geometry("Trunk", "box", width=0.7, depth=0.7, height=3.0)
    trunk.add_modifier("transform", translation=(0, 0, 1.5))
    leaves = scene.add_geometry("Leaves", "box", width=3.2, depth=3.2, height=2.6)
    leaves.add_modifier("bevel", width=0.25, limit="all")
    leaves.add_modifier("transform", translation=(0, 0, 3.9))
    for i, (x, y, size) in enumerate(TREES, start=1):
        scene.add_object(f"Trunk {i}", geometry=trunk, position=(x, y, 0), scale=size)
        scene.add_object(f"Leaves {i}", geometry=leaves, position=(x, y, 0), scale=size)

    for obj in scene:
        if obj.name in COLORS:
            obj.color = COLORS[obj.name]
        elif obj.name.startswith("Bush") or obj.name.startswith("Leaves"):
            obj.color = "#2e7a2c" if obj.name.startswith("Bush") else "#3d8a33"
        elif obj.name.startswith("Trunk"):
            obj.color = "#6b4a2b"

    scene.world.configure(background="#8ec5f2", ambient_color="#d5e6ff", ambient_intensity=0.32)
    scene.add_directional_light("Sun", direction=(0.55, 0.6, -0.8), color="#fff1d6", intensity=2.4)
    scene.add_directional_light(
        "Fill", direction=(-0.6, -0.3, -0.4), color="#bcd2ff", intensity=0.35
    )
    for k, x in enumerate((-2.6, 2.6), start=1):
        scene.add_point_light(
            f"Torch {k}",
            position=(x, -width_m / 2 - 2.0, 3.0),
            color="#ff9a3c",
            intensity=6.0,
            range=10.0,
        )

    if materials:
        dress_with_materials(scene, textures, width_m)
    camera = scene.add_camera("Camera", fov=30)
    camera.frame(direction=(-0.55, 1.0, -0.62), margin=0.02, aspect=width / height)
    scene.render_settings.configure(width=width, height=height, samples=samples)
    return scene


LEAVES = {
    "output": "color",
    "nodes": {
        "clusters": {"type": "voronoi", "scale": 3.5, "randomness": 1.0, "seed": 21},
        "light": {"type": "fbm", "scale": 1.2, "octaves": 3, "seed": 22},
        "tone": {
            "type": "ramp",
            "factor": {"node": "clusters", "output": "cell"},
            "stops": [[0.0, "#2c6a24"], [0.5, "#3d8a33"], [1.0, "#5ea43c"]],
        },
        "shade": {
            "type": "ramp",
            "factor": {"node": "clusters", "output": "distance"},
            "stops": [[0.0, 1.08], [0.9, 0.62]],
        },
        "lit": {
            "type": "mix",
            "a": {"node": "tone"},
            "b": {"node": "shade"},
            "factor": 1.0,
            "mode": "multiply",
        },
        "color": {
            "type": "mix",
            "a": {"node": "lit"},
            "b": "#9cc95a",
            "factor": {"node": "light"},
            "mode": "overlay",
        },
    },
}
LEAF_BUMPS = {
    "output": "bump",
    "nodes": {
        "clusters": {"type": "voronoi", "scale": 3.5, "randomness": 1.0, "seed": 21},
        "height": {"type": "math", "op": "invert", "a": {"node": "clusters", "output": "distance"}},
        "bump": {"type": "bump", "height": {"node": "height"}, "depth": 0.06, "step": 0.003},
    },
}


def dress_with_materials(scene: Scene, textures: Path, width_m: float) -> None:
    """Materials for every part: image textures, procedural presets and graphs, emission."""
    # Stone bricks from the example images: 4 bricks per texture tile, about 0.55 m long.
    stone = scene.create_material("Stone Bricks", roughness=0.88)
    projected = {"coordinates": "object", "projection": "triplanar", "scale": 1 / 2.2}
    stone.set_texture("base_color", textures / "stone_bricks_color.png", **projected)
    stone.set_texture("normal", textures / "stone_bricks_normal.png", **projected)
    light_stone = scene.create_material("Pale Stone Bricks", roughness=0.85, base_color="#e6e2da")
    light_stone.set_texture("base_color", "stone_bricks_color", **projected)
    light_stone.set_texture("normal", "stone_bricks_normal", **projected)
    if jpeg_available():
        planks = scene.create_material("Oak Planks", roughness=0.72, base_color="#e8dccc")
        planks.set_texture(
            "base_color", textures / "wood_planks.jpg", coordinates="object", scale=1 / 3.0
        )
    else:  # JPEG needs Pillow (the `images` extra); main() says so
        planks = scene.create_material(
            "Oak Planks", preset="wood", color="#c9965e", color2="#7b4f2a", axis="z"
        )
    shingles = scene.create_material(
        "Shingles",
        preset="brick",
        color="#9c3a27",
        color2="#c4553a",
        mortar_color="#4a1d14",
        brick_width=0.42,
        brick_height=0.21,
        mortar=0.03,
        roughness=0.8,
    )
    cobbles = scene.create_material("Cobblestones", preset="stone", color="#9a8a72", scale=0.7)
    grass = scene.create_material("Meadow", preset="grass", scale=1.5)
    bark = scene.create_material(
        "Bark", preset="wood", color="#7a5232", color2="#3e2716", axis="z", scale=0.5
    )
    leaves = scene.create_material("Leaves", roughness=0.85, specular=0.4)
    leaves.set_procedural_texture("base_color", LEAVES, coordinates="object", projection="volume")
    leaves.set_procedural_texture(
        "normal", LEAF_BUMPS, coordinates="object", projection="triplanar"
    )
    flame = scene.create_material(
        "Torch Flame", base_color="#ffcf6b", emissive="#ff9a3c", emissive_strength=6.0
    )
    for name, material in (
        ("Curtain Wall", stone),
        ("Watchtower", stone),
        ("Towers", light_stone),
        ("Keep", planks),
        ("Watchtower Roof", shingles),
        ("Path", cobbles),
        ("Ground", grass),
    ):
        scene.get_by_name(name).material = material
    for obj in scene:
        if obj.name.startswith("Trunk"):
            obj.material = bark
        elif obj.name.startswith(("Leaves", "Bush")):
            obj.material = leaves
    # Torch posts and their glowing flames, under the two torch lights.
    post = scene.add_geometry("Torch Post", "box", width=0.18, depth=0.18, height=2.4)
    post.add_modifier("transform", translation=(0, 0, 1.2))
    for k, x in enumerate((-2.6, 2.6), start=1):
        y = -width_m / 2 - 2.0
        scene.add_object(f"Torch Post {k}", geometry=post, position=(x, y, 0)).material = bark
        flame_obj = scene.add_cube(f"Torch Flame {k}", size=0.32, position=(x, y, 2.58))
        flame_obj.material = flame


def mean_colors(result) -> dict[str, np.ndarray]:
    out = {}
    for obj in result.objects:
        mask = result.object_index == obj["index"]
        if mask.any():
            out[obj["name"]] = result.image[mask].mean(axis=0)
    return out


BLENDER_SCRIPT = r"""
import bpy, json, sys, mathutils
glb, out, spec = sys.argv[-3], sys.argv[-2], json.loads(sys.argv[-1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "SINGLE"
scene.display.shading.single_color = (1, 1, 1)
scene.display.render_aa = "OFF"
scene.render.resolution_x, scene.render.resolution_y = spec["width"], spec["height"]
scene.render.resolution_percentage = 100
world = bpy.data.worlds.new("w"); world.color = (0, 0, 0); scene.world = world
scene.view_settings.view_transform = "Standard"
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "VERTICAL"
cam_data.angle_y = spec["fov_radians"]
cam_data.clip_start, cam_data.clip_end = spec["near"], spec["far"]
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
# Blender cameras look along local -Z with +Y up: columns (right, up, -forward).
r, u, f = (mathutils.Vector(v) for v in (spec["right"], spec["up"], spec["forward"]))
m = mathutils.Matrix((r, u, -f)).transposed().to_4x4()
m.translation = mathutils.Vector(spec["position"])
cam.matrix_world = m
scene.camera = cam
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
"""


def blender_command() -> list[str] | None:
    if shutil.which("blender"):
        return ["blender"]
    if shutil.which("flatpak"):
        listed = subprocess.run(
            ["flatpak", "list", "--app", "--columns=application"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "org.blender.Blender" in listed.stdout.split():
            return ["flatpak", "run", "org.blender.Blender"]
    return None


def compare_with_blender(scene: Scene, result) -> list[str]:
    """External verification: compare silhouettes with Blender rendering the same camera."""
    command = blender_command()
    if command is None:
        return ["Blender not found"]
    from aether3d.rendering.png import encode_png  # noqa: F401 (our encoder made the PNG)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if command[0] == "flatpak":
            command = [*command[:2], f"--filesystem={tmp_path}", *command[2:]]
        glb = tmp_path / "castle.glb"
        scene.export(glb)
        cam = result.camera
        spec = {
            "width": result.width,
            "height": result.height,
            "near": cam["near"],
            "far": cam["far"],
            "fov_radians": float(np.radians(cam["vertical_fov"])),
            "position": cam["position"],
            "forward": cam["forward"],
            "up": cam["up"],
            "right": cam["right"],
        }
        script = tmp_path / "render.py"
        script.write_text(BLENDER_SCRIPT)
        out = tmp_path / "blender.png"
        run = subprocess.run(
            [
                *command,
                "--background",
                "--factory-startup",
                "--python",
                str(script),
                "--",
                str(glb),
                str(out),
                json.dumps(spec),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if not out.exists():
            return ["Blender did not render: " + run.stderr[-400:]]
        try:
            from PIL import Image
        except ImportError:
            return ["Pillow is needed to read Blender's image"]
        blender_mask = np.asarray(Image.open(out).convert("L")) > 127
    ours = result.object_index >= 0
    union = np.count_nonzero(ours | blender_mask)
    iou = np.count_nonzero(ours & blender_mask) / union
    differ = np.count_nonzero(ours != blender_mask)
    print(f"   silhouettes match Blender's: IoU {iou:.5f}, {differ} of {ours.size} pixels differ")
    return [] if iou > 0.995 else [f"silhouettes differ from Blender's (IoU {iou:.4f})"]


def texture_detail(flat, result, name: str) -> tuple[float, float]:
    """Mean 8-bit change between horizontally adjacent pixels of object ``name`` where its
    flat-colored render is smooth (no silhouettes, creases or shading steps): what the
    materials add there. Returns ``(flat, materials)``."""

    def changes(render):
        index = next(o["index"] for o in render.objects if o["name"] == name)
        ids, image = render.object_index, render.image.astype(int)
        inside = (ids[:, 1:] == index) & (ids[:, :-1] == index)
        return inside, np.abs(image[:, 1:] - image[:, :-1]).max(axis=-1)

    flat_inside, flat_change = changes(flat)
    inside, change = changes(result)
    smooth = flat_inside & inside & (flat_change <= 2)
    if not smooth.any():
        return 0.0, 0.0
    return float(flat_change[smooth].mean()), float(change[smooth].mean())


def side_by_side(left, right) -> np.ndarray:
    """Two renders at half size with a gutter (2x2 averages in linear light)."""
    from aether3d.rendering.color import linear_to_srgb, quantize

    def half(result) -> np.ndarray:
        h, w = result.height // 2 * 2, result.width // 2 * 2
        lin = result.linear[:h, :w].astype(np.float64)
        return quantize(linear_to_srgb(lin.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))))

    a, b = half(left), half(right)
    return np.concatenate([a, np.full((a.shape[0], 8, 3), 24, np.uint8), b], axis=1)


def copy_textures(out_dir: Path) -> Path:
    """The example textures next to the project, so it refers to them by relative paths."""
    target = out_dir / "textures"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("stone_bricks_color.png", "stone_bricks_normal.png", "wood_planks.jpg"):
        shutil.copyfile(TEXTURES / name, target / name)
    return target


def main(argv: list[str]) -> int:
    from aether3d import __version__
    from aether3d.rendering.png import write_png

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--blender", action="store_true", help="compare silhouettes with Blender")
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print("1. Building the procedural castle and dressing it with materials")
    textures = copy_textures(out_dir)
    scene = build_scene(args.width, args.height, args.samples, materials=True, textures=textures)
    if not jpeg_available():
        print(
            "   note: Pillow is not installed, so the keep uses procedural planks instead of "
            "wood_planks.jpg (pip install 'aether3d[images]' for JPEG textures)"
        )
    procedural = sum(1 for o in scene if o.is_procedural)
    print(
        f"   {len(scene)} objects ({procedural} procedural), {len(scene.lights)} lights, "
        f"{len(scene.materials.materials)} materials, {len(scene.materials.textures)} textures, "
        f"camera {scene.active_camera.name!r}"
    )

    print("2. Rendering with Aether3D's CPU renderer")
    result = scene.render()
    stats = result.stats
    t = stats["timings"]
    print(
        f"   {result.width}x{result.height}, {args.samples}x{args.samples} samples per pixel, "
        f"{stats['triangles']} triangles, {stats['fragments']:,} fragments"
    )
    print(
        f"   {t['total']:.2f} s (evaluate {t['evaluate']:.2f}, textures {t['textures']:.2f}, "
        f"geometry {t['geometry']:.2f}, raster {t['raster']:.2f}, shade {t['shade']:.2f}, "
        f"output {t['output']:.2f})"
    )
    image_path = result.save(out_dir / "minecraft_castle_render.png")
    print(f"   wrote {image_path} ({stats['png_bytes']:,} bytes)")
    flat = build_scene(args.width, args.height, args.samples).render()
    pair = write_png(
        out_dir / "castle_flat_vs_materials.png",
        side_by_side(flat, result),
        text={"Software": f"Aether3D {__version__}"},
    )
    print(f"   wrote {pair} (half size: Aether3D 0.6's flat colors | materials)")

    print("3. Checking the image")
    must_see = ["Ground", "Curtain Wall", "Towers", "Keep", "Watchtower", "Watchtower Roof", "Path"]
    seen = {o["name"]: o["visible_samples"] for o in result.objects}
    for name in must_see + [f"Leaves {i}" for i in (1, 2, 4)] + ["Torch Flame 1"]:
        if seen.get(name, 0) == 0:
            failures.append(f"{name} is not visible")
    colors = mean_colors(result)
    roof, walls, ground = colors["Watchtower Roof"], colors["Curtain Wall"], colors["Ground"]
    if not (roof[0] > 1.5 * roof[1] and ground[1] > 1.2 * ground[0] and np.ptp(walls) < 30):
        failures.append("parts do not show their colors (red roof, green ground, gray stone)")
    print(
        f"   all {len(must_see)} castle parts, the trees and the torches are visible; red roof "
        f"{roof.round().tolist()}, stone {walls.round().tolist()}, grass "
        f"{ground.round().tolist()}"
    )
    detail = {}
    for name in ("Curtain Wall", "Keep", "Ground", "Watchtower Roof"):
        detail[name] = plain, textured = texture_detail(flat, result, name)
        if not textured > plain + 0.5:
            failures.append(f"{name}: the material adds no visible detail")
    print(
        "   texture detail (mean change between neighbouring pixels where the flat colors are "
        "smooth, flat -> materials): "
        + ", ".join(f"{k} {a:.2f} -> {b:.2f}" for k, (a, b) in detail.items())
    )
    if result.diagnostics:
        failures.append(f"unexpected diagnostics: {[d.code for d in result.diagnostics]}")

    print("4. Saving, reloading and re-rendering (API and CLI)")
    project = out_dir / "minecraft_castle_render.aether"
    scene.save(project)
    small = {"width": 480, "height": 270, "samples": 2}
    reference = scene.render(**small)
    reloaded = Scene.load(project).render(**small)
    if not np.array_equal(reloaded.image, reference.image):
        failures.append("the reloaded project renders differently")
    with tempfile.TemporaryDirectory() as tmp:
        cli_png = Path(tmp) / "cli.png"
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "aether3d",
                "render",
                str(project),
                "-o",
                str(cli_png),
                "--width",
                "480",
                "--height",
                "270",
                "--samples",
                "2",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            failures.append(f"aether3d render failed: {run.stdout[-300:]} {run.stderr[-300:]}")
        elif cli_png.read_bytes() != reference.png_bytes():
            failures.append("aether3d render produced different pixels")
        else:
            print("   reloaded project and `aether3d render` reproduce the image exactly")
    print(f"   reproduce: aether3d render {project} -o castle.png")

    if args.blender:
        print("5. External verification: Blender renders the same camera (silhouettes only)")
        failures += compare_with_blender(scene, result)

    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
