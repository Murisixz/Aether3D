"""Stage 7 materials in the path tracer: the same surfaces as on the rasterizer.

The path tracer evaluates materials with the rasterizer's own code
(``surface.evaluate_material``), so under lighting both can reproduce --
uniform ambient light on open surfaces, or direct light without shadows --
the images must agree. Where an exact expectation exists (texel lookups,
emission) it is computed here independently.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d import Mesh, Scene, import_gltf
from aether3d.rendering.png import write_png

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
KHRONOS = Path(__file__).parent / "data" / "khronos"
HAS_MANIFOLD = importlib.util.find_spec("manifold3d") is not None


def top_camera(scene: Scene, scale: float = 2.0):
    return scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=scale,
        position=(0, 0, 5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )


def ambient_only(scene: Scene) -> None:
    """Uniform white light of radiance 1 from every direction, no lights."""
    scene.world.configure(background="#000000", ambient_color="#ffffff", ambient_intensity=1.0)


def save_png(path: Path, pixels) -> Path:
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    write_png(path, arr)
    return path


def path_render(scene, width, height, **path):
    path.setdefault("bounces", 0)
    return scene.render(width=width, height=height, backend="path", path=path, default_lights=False)


def test_texels_land_exactly_where_uvs_put_them(tmp_path):
    """A Lambertian plane open to a uniform sky reflects exactly its texel colors."""
    rng = np.random.default_rng(0)
    texels = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    m.set_texture(
        "base_color",
        scene.add_image_texture(
            "T", save_png(tmp_path / "t.png", texels), filter="nearest", mipmaps=False
        ),
    )
    plane.material = m
    ambient_only(scene)
    top_camera(scene)
    r = path_render(scene, 64, 64, samples=4)
    expected = np.repeat(np.repeat(texels, 16, axis=0), 16, axis=1)
    assert np.array_equal(r.image, expected)
    raster = scene.render(width=64, height=64, samples=1, default_lights=False)
    assert np.array_equal(r.image, raster.image)


def lit_material_scene(material_setup) -> Scene:
    """A plane under a directional light, seen from straight above (no shadows possible)."""
    scene = Scene()
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    plane = scene.add_plane("Plane", size=2)
    plane.material = material_setup(scene)
    scene.add_directional_light("Sun", direction=(0.3, -0.5, -1.0), intensity=1.5)
    top_camera(scene)
    return scene


def compare_with_raster(scene, size=48, block_tolerance=0.04):
    """Path (16 random samples, footprint 1/4 pixel) vs raster (4x4 grid, footprint 1/4 pixel).

    The same footprint selects the same mip levels and filtering, so the mean must
    match closely; per pixel, 16 random positions vs a regular grid differ by sampling
    noise on detailed, glossy textures, so 4x4-pixel blocks are compared too.
    """
    raster = scene.render(width=size, height=size, samples=4, default_lights=False)
    path = path_render(scene, size, size, samples=16)
    assert float(path.linear.mean() / raster.linear.mean()) == pytest.approx(1.0, abs=0.01)
    k = size // 4

    def blocks(a):
        return a.astype(np.float64).reshape(k, 4, k, 4, 3).mean(axis=(1, 3))

    diff = np.abs(blocks(path.linear) - blocks(raster.linear)).mean()
    assert float(diff) < block_tolerance * float(raster.linear.mean())
    return raster, path


def test_image_textures_with_normal_maps_shade_like_the_rasterizer():
    textures = EXAMPLES / "textures"

    def setup(scene):
        m = scene.create_material("Stone", roughness=0.8)
        m.set_texture("base_color", textures / "stone_bricks_color.png", scale=2.0)
        m.set_texture("normal", textures / "stone_bricks_normal.png", scale=2.0)
        return m

    raster, path = compare_with_raster(lit_material_scene(setup))
    assert raster.linear.std() > 0.02  # the texture and normal map add real detail


def test_procedural_textures_in_world_and_object_space_shade_like_the_rasterizer():
    def setup(scene):
        m = scene.create_material("Planks", preset="wood", scale=0.5)
        m.set_procedural_texture(
            "roughness",
            {"type": "fbm", "scale": 6.0, "octaves": 3, "seed": 2},
            coordinates="world",
            projection="xy",
        )
        return m

    raster, _ = compare_with_raster(lit_material_scene(setup))
    assert raster.linear.std() > 0.01


def test_triplanar_bump_mapped_materials_shade_like_the_rasterizer():
    def setup(scene):
        return scene.create_material("Wall", preset="brick", scale=0.6)

    compare_with_raster(lit_material_scene(setup))


def test_face_group_materials_paint_their_own_faces():
    cube = Mesh(
        *(lambda c: (c.positions, c.triangles, c.normals, c.uvs))(
            __import__("aether3d").create_cube(1.0)
        ),
        groups=np.repeat([0, 0, 0, 0, 1, 1], 2),
    )
    scene = Scene()
    ambient_only(scene)
    obj = scene.add_object("Cube", mesh=cube)
    red = scene.create_material("Red", model="simple", base_color="#ff0000")
    blue = scene.create_material("Blue", model="simple", base_color="#0000ff")
    obj.material = red
    scene.assign_material(obj, blue, group=1)
    scene.add_camera("Cam", position=(2.5, -3.0, 2.2), look_at=(0, 0, 0))
    raster = scene.render(width=40, height=30, samples=1, default_lights=False)
    path = path_render(scene, 40, 30, samples=8, bounces=0)
    reddish = raster.linear[..., 0] > raster.linear[..., 2]
    blueish = raster.linear[..., 2] > raster.linear[..., 0]
    assert reddish.sum() > 30 and blueish.sum() > 30

    def inner(m):
        return m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)

    assert np.all(path.linear[inner(reddish)][:, 0] > path.linear[inner(reddish)][:, 2])
    assert np.all(path.linear[inner(blueish)][:, 2] > path.linear[inner(blueish)][:, 0])


def test_alpha_cutouts_are_holes_for_camera_and_shadow_rays(tmp_path):
    checker = np.zeros((8, 8, 4), dtype=np.uint8)
    checker[..., :3] = 200
    checker[..., 3] = np.where((np.indices((8, 8)).sum(axis=0) % 2) == 0, 255, 0)
    write_png(tmp_path / "mask.png", checker)
    scene = Scene()
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    floor = scene.add_plane("Floor", size=8)
    floor.material = scene.create_material("Floor", model="simple", base_color="#ffffff")
    screen = scene.add_plane("Screen", size=2, position=(0, 0, 1.0))
    mesh_material = scene.create_material(
        "Lace", model="simple", base_color="#ffffff", alpha_mode="mask", alpha_cutoff=0.5
    )
    mesh_material.set_texture(
        "base_color",
        scene.add_image_texture("Mask", tmp_path / "mask.png", filter="nearest", mipmaps=False),
    )
    screen.material = mesh_material
    scene.add_directional_light("Sun", direction=(0.0, 0.0, -1.0), intensity=1.0)
    # Seen from above, the floor shows through the holes of the screen.
    top_camera(scene, scale=2.0)
    r = path_render(scene, 32, 32, samples=4)
    sees_floor = r.object_index == 0
    assert 0.35 < sees_floor.mean() < 0.65  # half the screen is holes
    # The floor in the screen's shadow is lit exactly through the holes.
    cam = scene.add_camera(
        "Floor Camera",
        projection="orthographic",
        ortho_scale=2.0,
        position=(0, 0, 0.5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )
    shadow = scene.render(
        camera=cam,
        width=32,
        height=32,
        backend="path",
        path={"samples": 4, "bounces": 0},
        default_lights=False,
    )
    lit = shadow.linear[..., 0] > 0.5
    assert 0.35 < lit.mean() < 0.65


def test_textured_emission_is_seen_and_lights_the_scene(tmp_path):
    glow = np.zeros((2, 2, 3), dtype=np.uint8)
    glow[0, 0] = (255, 0, 0)
    glow[1, 1] = (0, 0, 255)
    scene = Scene()
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    panel = scene.add_plane("Panel", size=2)
    m = scene.create_material("Neon", model="simple", base_color="#000000", emissive="#ffffff")
    m.set_texture(
        "emissive",
        scene.add_image_texture(
            "Glow", save_png(tmp_path / "glow.png", glow), filter="nearest", mipmaps=False
        ),
    )
    panel.material = m
    top_camera(scene)
    path = path_render(scene, 16, 16, samples=2)
    raster = scene.render(width=16, height=16, samples=1, default_lights=False)
    assert np.array_equal(path.image, raster.image)  # emission seen directly, exactly
    # The panel lights a floor below it (facing up at it).
    floor = scene.add_plane("Floor", size=6, position=(0, 0, -1.0))
    floor.material = scene.create_material("White", model="simple", base_color="#ffffff")
    panel.rotation_euler = (180, 0, 0)  # face the floor
    scene.add_camera("Under", position=(0, -3, 0.5), look_at=(0, 0, -1))
    lit = scene.render(
        camera="Under",
        width=24,
        height=16,
        backend="path",
        path={"samples": 16, "bounces": 1},
        default_lights=False,
    )
    on_floor = lit.object_index == 1
    assert on_floor.any() and lit.linear[on_floor].max() > 0.05


def test_single_sided_back_faces_are_invisible():
    scene = Scene()
    ambient_only(scene)
    plane = scene.add_plane("Plane", size=2)
    plane.material = scene.create_material(
        "Front", model="simple", base_color="#ffffff", double_sided=False
    )
    scene.add_camera("Below", position=(0, 0, -3), look_at=(0, 0, 0), up=(0, 1, 0))
    scene.add_camera("Above", position=(0, 0, 3), look_at=(0, 0, 0), up=(0, 1, 0))
    below = scene.render(camera="Below", width=8, height=8, backend="path", path={"samples": 1})
    above = scene.render(camera="Above", width=8, height=8, backend="path", path={"samples": 1})
    assert below.coverage.max() == 0.0 and above.coverage[2:6, 2:6].min() == 1.0


def test_glossy_surfaces_reflect_lights_that_diffuse_ones_scatter():
    """Equal-albedo spheres: the polished one shows a small, bright image of the lamp."""
    peaks = {}
    for roughness in (0.1, 1.0):
        scene = Scene()
        scene.world.configure(background="#000000", ambient_intensity=0.0)
        ball = scene.add_sphere("Ball", radius=1.0, segments=48, rings=24)
        ball.material = scene.create_material(
            "Paint", base_color="#808080", roughness=roughness, metallic=1.0
        )
        lamp = scene.add_plane("Lamp", size=0.6, position=(0, -3, 1.5))
        lamp.look_at((0, 0, 0))
        lamp.rotation_euler = (lamp.rotation_euler[0] + 180, *lamp.rotation_euler[1:])
        lamp.material = scene.create_material(
            "Lamp", model="simple", base_color="#000000", emissive="#ffffff", emissive_strength=20
        )
        scene.add_camera("Cam", position=(0, -4, 0.6), look_at=(0, 0, 0))
        r = scene.render(
            width=48,
            height=48,
            backend="path",
            path={"samples": 32, "bounces": 1},
            default_lights=False,
        )
        on_ball = r.object_index == 0
        peaks[roughness] = (float(r.linear[on_ball].max()), float(r.linear[on_ball].mean()))
    assert peaks[0.1][0] > 4 * peaks[1.0][0]  # a sharp highlight
    assert peaks[0.1][1] < 2.0 * peaks[1.0][1]  # but no energy appears from nowhere


@pytest.mark.skipif(not HAS_MANIFOLD, reason="the castle needs aether3d[boolean]")
def test_the_textured_castle_path_traces_with_indirect_light_in_the_courtyard():
    sys.path.insert(0, str(EXAMPLES))
    from minecraft_castle_render import build_scene

    scene = build_scene(160, 90, 2, materials=True)
    direct = scene.render(width=160, height=90, backend="path", path={"samples": 4, "bounces": 0})
    full = scene.render(width=160, height=90, backend="path", path={"samples": 4, "bounces": 3})
    raster = scene.render(width=160, height=90)
    assert [d.code for d in full.diagnostics if d.severity != "info"] == []
    assert full.backend["lights"]["emissive_triangles"] > 0  # glowing torches are lights
    # Every object the rasterizer shows is in the path-traced image too.
    seen_raster = {o["name"] for o in raster.objects if o["visible_samples"]}
    seen_path = {o["name"] for o in full.objects if o["visible_samples"]}
    assert seen_raster <= seen_path
    # Bounced light adds to the direct light (sun, fill, torches and the sky) on the castle.
    geometry = (direct.coverage == 1.0) & (full.coverage == 1.0)
    assert full.linear[geometry].mean() > 1.02 * direct.linear[geometry].mean()
    assert full.passes["indirect"][geometry].mean() > 0.02 * full.linear[geometry].mean()
    np.testing.assert_allclose(
        full.passes["direct"] + full.passes["indirect"], full.passes["beauty"], rtol=1e-4, atol=1e-6
    )


def test_the_blocky_hero_with_materials_path_traces():
    sys.path.insert(0, str(EXAMPLES))
    from hero_render import PARTS, build_scene

    scene = build_scene(96, 54, 2, materials=True)
    r = scene.render(width=96, height=54, backend="path", path={"samples": 2, "bounces": 2})
    names = {o["name"] for o in r.objects if o["visible_samples"]}
    assert set(PARTS) <= names
    assert [d.code for d in r.diagnostics if d.severity != "info"] == []


def test_imported_gltf_textures_render_in_the_path_tracer():
    result = import_gltf(KHRONOS / "TextureCoordinateTest/glTF-Binary/TextureCoordinateTest.glb")
    scene = result.scene
    raster = scene.render(width=48, height=48, frame=True)
    path = scene.render(
        width=48, height=48, frame=True, backend="path", path={"samples": 8, "bounces": 1}
    )
    covered = (raster.coverage == 1.0) & (path.coverage == 1.0)
    assert covered.sum() > 200
    # The same textures, the same places: per-pixel colors correlate strongly.
    a = raster.linear[covered].ravel()
    b = path.linear[covered].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.9
