"""Shading and color: analytic lighting values, exposure, background, antialiasing, symmetry."""

import math

import numpy as np
import pytest

from aether3d import Scene, create_plane, create_sphere
from aether3d.rendering.color import linear_to_srgb, parse_color, quantize, srgb_to_linear

W = H = 32


def top_down_scene(color="#808080", ambient=0.0, size=10.0, ortho=10.0):
    """A plane facing +Z seen straight from above by an orthographic camera."""
    scene = Scene()
    scene.add_object("Plane", mesh=create_plane(size)).color = color
    cam = scene.add_camera(
        "Top", projection="orthographic", ortho_scale=ortho, position=(0, 0, 5), near=0.1, far=20
    )
    cam.look_along((0, 0, -1))
    scene.world.configure(ambient_intensity=ambient, background="#000000")
    return scene


def pixel_world_xy(width, height, ortho):
    """World (x, y) of each pixel center for the top-down camera (image up = +Y, right = +X)."""
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    x = (cols / width * 2 - 1) * ortho / 2 * (width / height)
    y = (1 - rows / height * 2) * ortho / 2
    return x, y


def test_srgb_transfer_functions():
    assert srgb_to_linear(0.5) == pytest.approx(0.21404114048223255, abs=1e-15)
    assert srgb_to_linear(0.04045) == pytest.approx(0.04045 / 12.92)
    values = np.linspace(0, 1, 1001)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-12)
    assert np.all(np.diff(srgb_to_linear(values)) > 0)
    assert list(quantize([0.0, 0.5, 1.0, 1.5, -1])) == [0, 128, 255, 255, 0]
    assert parse_color("#336699") == (0x33 / 255, 0x66 / 255, 0x99 / 255)
    assert parse_color("#fff") == (1.0, 1.0, 1.0)
    assert parse_color([0.1, 0.2, 0.3]) == (0.1, 0.2, 0.3)
    for bad in ("336699", "#12345", "#ggg", [1, 2], [0.1, 0.2, 1.5], [0.1, float("nan"), 0], 7):
        with pytest.raises(Exception):  # noqa: B017 - RenderError subclass of ValueError
            parse_color(bad)


@pytest.mark.parametrize("angle", [0.0, 30.0, 60.0, 89.0])
def test_lambert_directional_light_is_exact(angle):
    scene = top_down_scene(color="#a06040", ambient=0.25)
    theta = math.radians(angle)
    direction = (math.sin(theta), 0.0, -math.cos(theta))  # light travels down at `angle`
    scene.add_directional_light("Sun", direction=direction, intensity=1.7, color="#ffffff")
    r = scene.render(width=W, height=H, samples=1)
    albedo = srgb_to_linear(parse_color("#a06040"))
    expected = albedo * (0.25 + 1.7 * math.cos(theta))
    assert np.allclose(r.linear, expected, rtol=1e-6, atol=1e-7)
    assert np.array_equal(
        r.image, np.broadcast_to(quantize(linear_to_srgb(expected)), r.image.shape)
    )


def test_light_from_behind_contributes_nothing_and_ambient_remains():
    scene = top_down_scene(color="#ffffff", ambient=0.3)
    scene.get_by_name("Plane").scale = (1, 1, 1)
    scene.add_directional_light("Under", direction=(0, 0, 1), intensity=5)  # shines upward
    r = scene.render(width=W, height=H, samples=1, cull="back")
    assert np.allclose(r.linear, 0.3, rtol=1e-6)


@pytest.mark.parametrize(
    ("falloff", "light_range"), [("inverse_square", None), ("inverse_square", 9.0), ("none", None)]
)
def test_point_light_attenuation_is_exact(falloff, light_range):
    scene = top_down_scene(color="#ffffff", ambient=0.0, size=20.0, ortho=12.0)
    h = 2.0
    scene.add_point_light(
        "Bulb", position=(0.7, -0.4, h), intensity=6.0, falloff=falloff, range=light_range
    )
    r = scene.render(width=48, height=32, samples=1)
    x, y = pixel_world_xy(48, 32, 12.0)
    dx, dy = 0.7 - x, -0.4 - y
    d = np.sqrt(dx * dx + dy * dy + h * h)
    cos = h / d
    att = 1 / d**2 if falloff == "inverse_square" else np.ones_like(d)
    if light_range is not None:
        att = att * np.clip(1 - (d / light_range) ** 4, 0, 1) ** 2
    expected = 6.0 * cos * att
    assert np.allclose(r.linear[..., 0], expected, rtol=1e-6, atol=1e-7)
    assert np.allclose(r.linear[..., 0], r.linear[..., 1]) and np.allclose(
        r.linear[..., 2], r.linear[..., 1]
    )


def test_specular_highlight_peaks_at_the_half_vector():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 96, 48))
    ball.color = "#000000"  # highlight only
    ball.appearance.configure(specular=1.0, shininess=200)
    cam = scene.add_camera(
        "Cam", projection="orthographic", ortho_scale=2.4, position=(0, -5, 0), near=0.1, far=10
    )
    cam.look_at((0, 0, 0))
    light_dir = np.array([0.6, 0.5, -0.62])  # travel direction
    scene.add_directional_light("Sun", direction=light_dir, intensity=1.0)
    scene.world.configure(ambient_intensity=0.0, background="#000000")
    r = scene.render(width=121, height=121, samples=1)
    L = -light_dir / np.linalg.norm(light_dir)
    V = np.array([0.0, -1.0, 0.0])
    half = (L + V) / np.linalg.norm(L + V)  # the sphere's normal at the peak is h
    # Orthographic image coordinates of the peak point: right = +X... the camera looks
    # along +Y, so image right is world +X and image up is world +Z.
    col = (half[0] / 1.2 + 1) / 2 * 121
    row = (1 - half[2] / 1.2) / 2 * 121
    peak = np.unravel_index(np.argmax(r.linear[..., 0]), r.linear.shape[:2])
    assert abs(peak[1] + 0.5 - col) <= 1.0 and abs(peak[0] + 0.5 - row) <= 1.0
    # (n.h)^s is 1 exactly at the peak; the pixel center nearest it, on a tessellated
    # sphere, has n.h = 0.9999 (0.989 after the power of 200).
    assert 0.95 <= r.linear[..., 0].max() <= 1.0


def test_exposure_scales_linear_light_exactly():
    scene = top_down_scene(color="#404040", ambient=0.2)
    scene.add_directional_light("Sun", direction=(0, 0, -1), intensity=0.5)
    base = scene.render(width=8, height=8, samples=1)
    brighter = scene.render(width=8, height=8, samples=1, exposure=1.5)
    assert np.allclose(brighter.linear, base.linear * 2**1.5, rtol=1e-6)


@pytest.mark.parametrize("color", ["#336699", "#000000", "#ffffff", "#7f8081"])
def test_background_color_is_exact(color):
    scene = Scene()
    scene.add_camera("Cam")
    scene.world.background = color
    r = scene.render(width=7, height=5)
    code = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    assert np.all(r.image == code)
    assert [d.code for d in r.diagnostics].count("NOTHING_TO_RENDER") == 1
    assert np.all(r.object_index == -1) and np.all(np.isinf(r.depth))


@pytest.mark.parametrize("samples", [2, 4])
def test_antialiasing_averages_samples_in_linear_light(samples):
    # A plane covering exactly the left half of the middle pixel column's samples.
    scene = top_down_scene(color="#ffffff", ambient=1.0, size=10.0, ortho=10.0)
    plane = scene.get_by_name("Plane")
    width = 9  # the view spans x in [-5, 5]: column 4 covers [-5/9, 5/9]
    plane.position = (-5, 0, 0)  # the plane's right edge is at x = 0, mid-column 4
    scene.world.background = "#000000"
    r = scene.render(width=width, height=9, samples=samples, default_lights=False)
    assert np.allclose(r.coverage[:, 4], 0.5)
    assert np.allclose(r.linear[:, 4], 0.5, rtol=1e-6)  # half white, half black, in linear
    assert np.all(r.coverage[:, :4] == 1) and np.all(r.coverage[:, 5:] == 0)
    # Displayed: the sRGB encoding of 0.5 linear (188), not 128.
    assert np.all(r.image[:, 4] == 188)


def test_two_sided_lighting_and_culling():
    scene = Scene()
    scene.add_object("Plane", mesh=create_plane(10)).color = "#ffffff"
    cam = scene.add_camera(
        "Below", projection="orthographic", ortho_scale=4, position=(0, 0, -3), near=0.1, far=10
    )
    cam.look_along((0, 0, 1))  # looking up at the plane's back side
    scene.add_directional_light("Up", direction=(0, 0, 1), intensity=1.0)  # light from below
    scene.world.configure(ambient_intensity=0.0, background="#000000")
    r = scene.render(width=8, height=8, samples=1)
    assert np.allclose(r.linear, 1.0)  # back faces are lit with the flipped normal
    culled = scene.render(width=8, height=8, samples=1, cull="back")
    assert np.all(culled.image == 0) and culled.stats["culled_backface"] == 2
    front_only = scene.render(width=8, height=8, samples=1, cull="front")
    assert np.allclose(front_only.linear, 1.0) and front_only.stats["culled_backface"] == 0


def test_mirrored_instances_render_as_exact_mirror_images():
    def build(mirror):
        scene = Scene()
        sign = -1 if mirror else 1
        ball = scene.add_object("Ball", mesh=create_sphere(1, 24, 12), position=(sign * 0.8, 0, 0))
        ball.scale = (sign * 1.5, 1.0, 0.8)  # mirrored and non-uniform
        ball.rotation_euler = (0, 0, sign * 25)
        ball.color = "#3080c0"
        box = scene.add_cube("Box", size=0.8, position=(sign * -1.2, 0.5, 0.3))
        box.rotation_euler = (0, 0, sign * 30)
        cam = scene.add_camera("Cam", position=(0, -6, 0))
        cam.look_at((0, 0, 0))
        scene.add_directional_light("Sun", direction=(sign * 0.5, 0.7, -0.6), intensity=1.5)
        scene.add_point_light("Bulb", position=(sign * 1.5, -2, 1.0), intensity=3)
        return scene.render(width=64, height=48, samples=1)

    a, b = build(False), build(True)
    mirrored = b.image[:, ::-1]
    differs = np.any(np.abs(a.image.astype(int) - mirrored.astype(int)) > 1, axis=-1)
    # Identical except possibly a few silhouette pixels (the fill rule is not symmetric).
    assert differs.sum() <= 0.01 * differs.size
    assert np.allclose(
        a.linear[~differs], mirrored_linear := b.linear[:, ::-1][~differs], atol=2e-3
    )
    assert np.isfinite(mirrored_linear).all()


def test_default_lighting_and_no_lights():
    scene = top_down_scene(color="#ffffff", ambient=0.0)
    r = scene.render(width=8, height=8)
    assert "DEFAULT_LIGHTING" in [d.code for d in r.diagnostics]
    assert r.linear.max() > 0.1  # the built-in key light lights the plane
    dark = scene.render(width=8, height=8, default_lights=False)
    assert "NO_LIGHTS" in [d.code for d in dark.diagnostics] and np.all(dark.linear == 0)


def test_hidden_lights_do_not_shine_and_colors_stay_in_range():
    scene = top_down_scene(color="#ffffff", ambient=0.0)
    sun = scene.add_directional_light("Sun", direction=(0, 0, -1), intensity=50)
    r = scene.render(width=8, height=8)
    assert (
        r.image.dtype == np.uint8 and r.image.max() == 255 and r.linear.max() == pytest.approx(50)
    )
    sun.visible = False
    r = scene.render(width=8, height=8, default_lights=False)
    assert np.all(r.linear == 0)
