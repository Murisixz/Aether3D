"""Materials on the CPU renderer: exact texel lookups, filtering, alpha, culling, lighting models.

Expected values are computed here from first principles (texel indices, the
glTF BRDF formulas) rather than by calling the renderer's own helpers.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from aether3d import Mesh, Scene
from aether3d.rendering.color import linear_to_srgb, quantize, srgb_to_linear
from aether3d.rendering.png import write_png


def flat_lit(scene: Scene, ambient: float = 1.0) -> None:
    scene.world.configure(background="#000000", ambient_color="#ffffff", ambient_intensity=ambient)


def top_camera(scene: Scene, scale: float = 2.0, axis: str = "vertical"):
    """Looks straight down at the origin; ``scale`` world units span the image along ``axis``."""
    return scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=scale,
        fov_axis=axis,
        position=(0, 0, 5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )


def save_png(path: Path, pixels) -> Path:
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    write_png(path, arr)
    return path


def test_texels_land_exactly_where_uvs_put_them(tmp_path):
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
    flat_lit(scene)
    top_camera(scene)
    result = scene.render(width=64, height=64, samples=1, default_lights=False)
    # Pixel (col, row) sees texel (col // 16, row // 16): row 0 of the image is the top (v = 1).
    expected = np.repeat(np.repeat(texels, 16, axis=0), 16, axis=1)
    assert np.array_equal(result.image, expected)


def test_bilinear_filtering_and_wrapping_between_texel_centers(tmp_path):
    texels = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)  # 2 x 1: black | white
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    tex = scene.add_image_texture(
        "T", save_png(tmp_path / "t.png", texels), mipmaps=False, wrap="clamp", color_space="linear"
    )
    m.set_texture("base_color", tex)
    plane.material = m
    flat_lit(scene)
    top_camera(scene, axis="horizontal")
    result = scene.render(width=8, height=1, samples=1, default_lights=False, exposure=0.0)
    # u at pixel centers: (i + 0.5) / 8; texel coordinate x = 2u - 0.5, clamped at the edges.
    u = (np.arange(8) + 0.5) / 8
    expected = np.clip(2 * u - 0.5, 0.0, 1.0)
    assert result.linear[0, :, 0] == pytest.approx(expected, abs=1e-6)
    tex.configure(wrap="repeat")
    wrapped = scene.render(width=8, height=1, samples=1, default_lights=False).linear[0, :, 0]
    x = 2 * u - 0.5
    blend = x - np.floor(x)
    left = np.floor(x).astype(int) % 2
    expected = np.where(left == 0, blend, 1.0 - blend)  # texel 0 is black, texel 1 white
    assert wrapped == pytest.approx(expected, abs=1e-6)


def test_mipmaps_remove_aliasing_when_textures_shrink(tmp_path):
    stripes = np.zeros((64, 64, 3), np.uint8)
    stripes[:, ::2] = 255  # one-texel stripes
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    tex = scene.add_image_texture("T", save_png(tmp_path / "s.png", stripes))
    m.set_texture("base_color", tex, scale=8)  # 512 texels across a 32-pixel image
    plane.material = m
    flat_lit(scene)
    top_camera(scene)
    filtered = scene.render(width=32, height=32, samples=1, default_lights=False).linear[..., 0]
    tex.configure(mipmaps=False, filter="nearest")
    aliased = scene.render(width=32, height=32, samples=1, default_lights=False).linear[..., 0]
    assert filtered.std() < 0.02 and filtered.mean() == pytest.approx(0.5, abs=0.02)
    # Without them every pixel center happens to land on a white stripe: aliasing.
    assert aliased.min() == 1.0


def test_alpha_masks_cut_holes_through_which_other_objects_show(tmp_path):
    mask = np.zeros((2, 2, 4), np.uint8)
    mask[..., :3] = 255
    mask[:, 1, 3] = 255  # right half opaque, left half transparent
    scene = Scene()
    back = scene.add_plane("Back", size=4, position=(0, 0, -1))
    back.color = "#ff0000"
    front = scene.add_plane("Front", size=2)
    m = scene.create_material(
        "Cut", model="simple", base_color="#00ff00", alpha_mode="mask", alpha_cutoff=0.5
    )
    m.set_texture(
        "base_color",
        scene.add_image_texture("Mask", save_png(tmp_path / "m.png", mask), filter="nearest"),
    )
    front.material = m
    flat_lit(scene)
    top_camera(scene)
    result = scene.render(width=16, height=16, samples=1, default_lights=False)
    names = [o["name"] for o in result.objects]
    assert result.object_index[8, 2] == names.index("Back") and result.object_index[
        8, 13
    ] == names.index("Front")
    assert result.pixel(2, 8) == (255, 0, 0) and result.pixel(13, 8) == (0, 255, 0)
    assert result.stats["alpha_discarded"] == 128
    m.alpha_mode = "opaque"
    opaque = scene.render(width=16, height=16, samples=1, default_lights=False)
    assert opaque.pixel(2, 8) == (0, 255, 0)  # opaque ignores opacity


def test_procedural_opacity_with_supersampling_gives_soft_cutout_edges():
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", alpha_mode="mask", base_color="#ffffff")
    m.set_procedural_texture("opacity", "gradient")  # opacity = u
    plane.material = m
    flat_lit(scene)
    top_camera(scene, axis="horizontal")
    result = scene.render(width=20, height=4, samples=4, default_lights=False)
    # The cutoff is u = 0.5: pixels left of the middle are background, right are covered.
    assert result.coverage[1, :9].max() == 0.0 and result.coverage[1, 11:].min() == 1.0


def test_single_sided_materials_hide_back_faces():
    scene = Scene()
    plane = scene.add_plane("P", size=2, rotation=(0, 1, 0, 0))  # turned over: faces -z
    plane.material = scene.create_material("M", model="simple", base_color="#ffffff")
    flat_lit(scene)
    top_camera(scene)
    assert scene.render(width=8, height=8, default_lights=False).stats["pixels_covered"] == 64
    plane.material.double_sided = False
    hidden = scene.render(width=8, height=8, default_lights=False)
    assert hidden.stats["pixels_covered"] == 0 and hidden.stats["culled_backface"] == 2


def test_emission_adds_unlit_light():
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    plane.material = scene.create_material(
        "Glow", base_color="#000000", emissive="#336699", emissive_strength=1.0
    )
    flat_lit(scene, ambient=0.0)
    top_camera(scene)
    result = scene.render(width=4, height=4, default_lights=False)
    assert result.pixel(1, 1) == (0x33, 0x66, 0x99)
    plane.material.emissive_strength = 2.0
    doubled = scene.render(width=4, height=4, default_lights=False).linear[1, 1]
    assert doubled == pytest.approx(
        2 * srgb_to_linear(np.array([0x33, 0x66, 0x99]) / 255), rel=1e-6
    )


def standard_reference(base, metallic, roughness, specular, n, v, to_light, radiance):
    """glTF 2.0 metallic-roughness BRDF (spec appendix B) times pi, for one light."""
    base = np.asarray(base, dtype=np.float64)
    rough = max(roughness, 0.045)
    alpha = rough * rough
    h = (to_light + v) / np.linalg.norm(to_light + v)
    pairs = ((n, to_light), (n, v), (n, h), (v, h))
    ndl, ndv, ndh, vdh = (max(float(np.dot(a, b)), 0.0) for a, b in pairs)
    f0 = (1 - metallic) * 0.08 * specular + metallic * base
    fresnel = f0 + (1 - f0) * (1 - vdh) ** 5
    d = alpha**2 / (math.pi * (ndh**2 * (alpha**2 - 1) + 1) ** 2)
    vis = 0.5 / (
        ndl * math.sqrt(ndv**2 * (1 - alpha**2) + alpha**2)
        + ndv * math.sqrt(ndl**2 * (1 - alpha**2) + alpha**2)
    )
    c_diff = base * (1 - metallic)
    return radiance * ndl * ((1 - fresnel) * c_diff + math.pi * fresnel * d * vis)


@pytest.mark.parametrize("metallic,roughness", [(0.0, 1.0), (0.0, 0.3), (1.0, 0.5), (0.5, 0.7)])
def test_the_standard_model_follows_the_gltf_brdf(metallic, roughness):
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    color = "#b07040"
    plane.material = scene.create_material(
        "M", base_color=color, metallic=metallic, roughness=roughness
    )
    flat_lit(scene, ambient=0.0)
    direction = np.array([0.3, 0.2, -1.0])
    scene.add_directional_light("Sun", direction=tuple(direction), intensity=2.0)
    camera = scene.add_camera("Cam", position=(0.4, -1.2, 3.0), look_at=(0, 0, 0))
    result = scene.render(width=33, height=33, samples=1)
    # The center pixel sees the origin; recompute its radiance independently.
    eye = np.array(camera.world_position)
    v = eye / np.linalg.norm(eye)
    to_light = -direction / np.linalg.norm(direction)
    base = srgb_to_linear(np.array([0xB0, 0x70, 0x40]) / 255)
    expected = standard_reference(
        base, metallic, roughness, 0.5, np.array([0, 0, 1.0]), v, to_light, 2.0
    )
    assert result.linear[16, 16] == pytest.approx(expected, rel=2e-3)


def test_ambient_light_on_metals_uses_the_environment_brdf():
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    plane.material = scene.create_material(
        "Gold", base_color="#e6c37a", metallic=1.0, roughness=1.0
    )
    flat_lit(scene, ambient=1.0)
    top_camera(scene)
    got = scene.render(width=4, height=4, default_lights=False).linear[1, 1]
    # Karis' fit at roughness 1, n.v = 1: F0 * 0.4524 - 0.0024 (nonzero: metals are not black).
    f0 = srgb_to_linear(np.array([0xE6, 0xC3, 0x7A]) / 255)
    a004 = min(0.0, 2 ** (-9.28)) * 0.0 + 0.015
    expected = f0 * (a004 * -1.04 + 0.468) + (a004 * 1.04 - 0.018)
    assert got == pytest.approx(expected, rel=1e-6)


def test_a_normal_map_tilts_the_shading_normal_in_the_uv_frame(tmp_path):
    tilt = np.array([-0.6, 0.0, 0.8])
    code = np.round((tilt * 0.5 + 0.5) * 255).astype(np.uint8)
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    m.set_texture(
        "normal",
        scene.add_image_texture(
            "N", save_png(tmp_path / "n.png", np.tile(code, (2, 2, 1))), mipmaps=False
        ),
    )
    plane.material = m
    flat_lit(scene, ambient=0.0)
    light = np.array([1.0, 0.0, -1.0]) / math.sqrt(2)  # travels toward +x: comes from -x
    scene.add_directional_light("Sun", direction=tuple(light), intensity=1.0)
    top_camera(scene)
    got = scene.render(width=4, height=4).linear[1, 1, 0]
    n = code / 255.0 * 2 - 1  # the stored (quantized) normal; u = +x, v = +y on the plane
    n = n / np.linalg.norm(n)
    assert got == pytest.approx(max(float(np.dot(n, -light)), 0.0), rel=1e-6)
    m.normal_strength = 0.0  # flattens the map: back to n = +z
    flat = scene.render(width=4, height=4).linear[1, 1, 0]
    assert flat == pytest.approx(1 / math.sqrt(2), rel=1e-6) and got > flat


def test_face_groups_render_with_their_own_materials():
    mesh = Mesh(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]], groups=[0, 1]
    )
    scene = Scene()
    obj = scene.add_object("Q", mesh=mesh)
    obj.material = scene.create_material("Blue", model="simple", base_color="#0000ff")
    scene.assign_material(
        obj, scene.create_material("Red", model="simple", base_color="#ff0000"), group=1
    )
    flat_lit(scene)
    top_camera(scene)
    result = scene.render(width=16, height=16, samples=1, default_lights=False)
    assert result.pixel(12, 12) == (0, 0, 255)  # lower right: triangle 0 (group 0)
    assert result.pixel(3, 3) == (255, 0, 0)  # upper left: triangle 1 (group 1)


def test_uv_textures_on_meshes_without_uvs_are_reported_and_skipped():
    mesh = Mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]])
    scene = Scene()
    obj = scene.add_object("NoUV", mesh=mesh)
    m = scene.create_material("M", model="simple", base_color="#808080")
    m.set_procedural_texture("base_color", "checker", color1="#000000", color2="#000000")
    obj.material = m
    flat_lit(scene)
    top_camera(scene)
    result = scene.render(width=8, height=8, default_lights=False)
    assert [d.code for d in result.diagnostics if d.code == "MISSING_UVS"] == ["MISSING_UVS"]
    assert result.pixel(4, 4) == (128, 128, 128)  # the factor alone
    m.set_texture_options("base_color", coordinates="object", projection="xy")
    assert scene.render(width=8, height=8, default_lights=False).pixel(4, 4) == (0, 0, 0)


def test_world_coordinates_stay_put_while_object_coordinates_move_with_the_object():
    def render(coordinates, x):
        scene = Scene()
        plane = scene.add_plane("P", size=8, position=(x, 0, 0))
        m = scene.create_material("M", model="simple", base_color="#ffffff")
        m.set_procedural_texture(
            "base_color",
            "checker",
            color1="#000000",
            color2="#ffffff",
            coordinates=coordinates,
            projection="xy",
        )
        plane.material = m
        flat_lit(scene)
        top_camera(scene, scale=4.0)
        return scene.render(width=32, height=32, samples=1, default_lights=False).image

    assert np.array_equal(render("world", 0.0), render("world", 1.0))
    assert np.array_equal(render("object", 0.0), 255 - render("object", 1.0))


def test_triplanar_projection_textures_every_face_of_a_cube():
    scene = Scene()
    cube = scene.add_cube("C", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    m.set_procedural_texture(
        "base_color", "checker", scale=2, color1="#000000", color2="#ffffff", coordinates="object"
    )
    cube.material = m
    flat_lit(scene)
    scene.add_camera("Cam", position=(4, -5, 4), look_at=(0, 0, 0))
    result = scene.render(width=96, height=96, samples=1, default_lights=False)
    covered = result.object_index >= 0
    gray = result.image[..., 0][covered]
    # Three faces are visible; each shows both checker colors (no smearing along an axis).
    assert (gray < 30).mean() > 0.3 and (gray > 225).mean() > 0.3


def test_converting_colors_to_simple_materials_changes_no_pixel():
    scene = Scene()
    scene.add_plane("Ground", size=8).color = "#4f8a33"
    scene.add_cube("Red", position=(0, 0, 0.5)).color = "#c0392b"
    ball = scene.add_sphere("Ball", radius=0.6, position=(1.5, 0.5, 0.6))
    ball.appearance.configure(color="#f1c40f", specular=0.5, shininess=40)
    scene.add_directional_light("Sun", direction=(0.4, 0.6, -1.0), intensity=2.0)
    scene.add_point_light("Lamp", position=(-2, -2, 3), intensity=5, range=8)
    scene.add_camera("Cam", position=(3, -6, 4), look_at=(0, 0, 0.5))
    before = scene.render(width=120, height=80, samples=2)
    scene.materials.convert_appearances()
    after = scene.render(width=120, height=80, samples=2)
    assert all(o.material is not None for o in scene if o.mesh is not None)
    assert np.array_equal(before.linear, after.linear)


def test_presets_render_without_warnings():
    scene = Scene()
    names = ["stone", "brick", "wood", "metal", "grass", "fabric"]
    for i, name in enumerate(names):
        scene.add_cube(
            name, size=0.9, position=(i * 1.1, 0, 0.45)
        ).material = scene.create_material(preset=name)
    scene.add_directional_light("Sun", direction=(0.3, 0.6, -0.8), intensity=2.0)
    camera = scene.add_camera("Cam", fov=35)
    camera.frame(direction=(0.2, 1.0, -0.5), aspect=2.0)
    result = scene.render(width=160, height=80, samples=1)
    assert result.status == "ok"
    seen = [o["name"] for o in result.objects if o["visible_samples"] > 0]
    assert seen == names
    # Procedural textures add detail: each cube shows real variation, not a flat color.
    for k, name in enumerate(names):
        pixels = result.linear[result.object_index == k]
        assert pixels.std(axis=0).max() > 0.004, name


def test_srgb_round_trip_assumption():
    codes = np.arange(256) / 255.0
    assert np.array_equal(quantize(linear_to_srgb(srgb_to_linear(codes))), np.arange(256))
