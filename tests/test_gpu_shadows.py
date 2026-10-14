"""GPU shadow maps, checked against analytic shadows and an independent ray tracer.

Shadow maps are discrete: a boundary can only be placed to within about a
texel, and filtering (PCF) blurs it over ``pcf_radius + 1`` texels on each side.
So every comparison here excludes a band around the true boundary of that
width (plus one pixel for the pixel footprint) and requires exact agreement
everywhere else: shadowed where the reference is shadowed, and lit -- to the
8-bit code of the unshadowed render -- where it is lit (no acne).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from render_reference import shadow_mask  # noqa: E402
from test_render_scene import boundary_mask  # noqa: E402

from aether3d import RenderError, Scene, create_sphere  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HAS_MANIFOLD = importlib.util.find_spec("manifold3d") is not None


def dilate(mask, radius):
    h, w = mask.shape
    out = mask.copy()
    pad = np.pad(mask, radius)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= pad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
    return out


def floating_box(a=0.6):
    """A box hovering over the ground, lit from the side, seen from straight above."""
    scene = Scene("Shadow")
    scene.add_plane("Ground", size=20).color = "#d0d0d0"
    scene.add_cube("Box", size=2, position=(-1, 0, 2))  # x -2..0, y -1..1, z 1..3
    scene.add_directional_light("Sun", direction=(a, 0.0, -1.0), intensity=1.5)
    scene.world.configure(background="#000000", ambient_intensity=0.25)
    cam = scene.add_camera("Top", projection="orthographic", ortho_scale=12, position=(0, 0, 20))
    cam.look_along((0, 0, -1), up=(0, 1, 0))
    return scene


@pytest.mark.parametrize("pcf_radius", [0, 1, 2])
def test_a_floating_box_casts_its_analytic_shadow(gpu, pcf_radius):
    a, size = 0.6, 192
    scene = floating_box(a)
    lit = gpu.render(scene, width=size, height=size, samples=1)
    shaded = gpu.render(
        scene, width=size, height=size, samples=1, shadows={"pcf_radius": pcf_radius}
    )
    (shadow,) = shaded.backend["shadows"]["maps"]
    # Pixel centers in world units: x to the right, y up, 12 units across.
    px = (np.arange(size) + 0.5) / size * 12 - 6
    x, y = np.meshgrid(px, -px)
    # Points (x, y, z) of the box reach the ground at (x + a z, y): the shadow spans
    # x in [-2 + a, 0 + 3a] and y in [-1, 1]; the box itself hides x in [-2, 0].
    inside = (x >= -2 + a) & (x <= 3 * a) & (np.abs(y) <= 1)
    ground = shaded.object_index == 0
    texel = shadow["texel_size"]
    band = (pcf_radius + 2.5) * texel + 12 / size  # filter blur + snapping + one pixel
    near_edge = (
        (np.abs(x - (-2 + a)) < band)
        | (np.abs(x - 3 * a) < band)
        | ((np.abs(np.abs(y) - 1) < band) & (x > -2 + a - band) & (x < 3 * a + band))
    )
    check = ground & ~near_edge & ~boundary_mask(shaded.object_index)
    darkened = shaded.linear.sum(-1) < 0.99 * lit.linear.sum(-1)
    assert np.array_equal(darkened[check], inside[check])
    # No acne: lit ground is exactly as bright as without shadows.
    sunny = check & ~inside
    assert np.abs(shaded.image[sunny].astype(int) - lit.image[sunny].astype(int)).max() <= 1
    # Fully shadowed ground receives only ambient light: 0.25 x its color.
    dark = check & inside
    assert np.allclose(
        shaded.linear[dark], lit.linear[dark] * 0.25 / (0.25 + 1.5 / np.sqrt(1 + a * a)), rtol=1e-3
    )


def test_filter_radius_and_map_size_shape_the_penumbra(gpu):
    scene = floating_box()

    def soft_pixels(**shadows):
        r = gpu.render(scene, width=192, height=192, samples=1, shadows=shadows)
        lit = gpu.render(scene, width=192, height=192, samples=1)
        ratio = r.linear.sum(-1) / np.maximum(lit.linear.sum(-1), 1e-9)
        return int(((ratio > 0.3) & (ratio < 0.99) & (r.object_index == 0)).sum()), r

    hard, r0 = soft_pixels(pcf_radius=0)
    soft, r2 = soft_pixels(pcf_radius=3)
    assert soft > hard
    small, r_small = soft_pixels(map_size=512, pcf_radius=2)
    large, r_large = soft_pixels(map_size=4096, pcf_radius=2)
    assert large < small
    # The same fitted area, padded by pcf_radius + 2 texels per side, divided into the map.
    covered = [
        r.backend["shadows"]["maps"][0]["texel_size"] * (n - 2 * (2 + 2))
        for r, n in ((r_small, 512), (r_large, 4096))
    ]
    assert covered[0] == pytest.approx(covered[1], rel=1e-9)


def shadow_reference_check(gpu, scene, light_name, width, height, *, compared=0.2):
    """GPU shadows of one light vs ray-traced shadow rays, away from boundaries."""
    light = scene.get_by_name(light_name)
    for other in scene.lights:
        other.visible = other is light
    scene.world.ambient_intensity = 0.0
    camera = scene.active_camera
    ref = shadow_mask(scene, camera, width, height, light)
    options = {"width": width, "height": height, "samples": 1, "msaa": 1}
    lit = gpu.render(scene, **options)
    shaded = gpu.render(scene, shadows={"lights": [light_name]}, **options)
    receives = lit.linear.sum(-1) > 1e-4  # surfaces facing the light
    visibility = shaded.linear.sum(-1) / np.maximum(lit.linear.sum(-1), 1e-12)
    edges = dilate(ref != np.roll(ref, 1, 0), 1) | dilate(ref != np.roll(ref, 1, 1), 1)
    edges |= boundary_mask(lit.object_index)
    check = receives & ~edges & (lit.object_index >= 0)
    wrong = check & ((visibility < 0.5) != ref)
    assert check.sum() > compared * width * height
    assert wrong.sum() <= 0.002 * check.sum(), f"{wrong.sum()} of {check.sum()} pixels disagree"
    assert ref[check].any() and (~ref[check]).any()  # both lit and shadowed areas compared
    return shaded


def test_shadows_under_rotated_scaled_and_mirrored_parents_match_ray_tracing(gpu):
    scene = Scene()
    scene.add_plane("Ground", size=16).color = "#cccccc"
    pivot = scene.add_object("Pivot", position=(0.5, 0.5, 1.5), scale=(1.5, -0.8, 1.2))
    pivot.rotation_euler = (20, -15, 35)
    scene.add_cube("Block", size=1, parent=pivot, position=(0.5, 0.2, 0.4), scale=(1, 2, 0.5))
    child = scene.add_object("Ball", mesh=create_sphere(0.6, 24, 12), parent=pivot)
    child.position = (-1.0, 0.5, 0.8)
    scene.add_directional_light("Sun", direction=(-0.5, 0.7, -1.0))
    scene.add_camera("Cam", position=(6, -8, 7), look_at=(0, 0, 0.5), fov=45)
    shaded = shadow_reference_check(gpu, scene, "Sun", 128, 96)
    assert shaded.backend["shadows"]["maps"][0]["light"] == "Sun"


@pytest.mark.skipif(not HAS_MANIFOLD, reason="the castle uses booleans (manifold3d)")
def test_castle_shadows_match_ray_tracing(gpu):
    spec = importlib.util.spec_from_file_location(
        "castle", ROOT / "examples" / "minecraft_castle_render.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "examples"))
    spec.loader.exec_module(module)
    scene = module.build_scene(144, 81, 1)
    # Castle walls are full of edges: about a tenth of the pixels is far enough from any.
    shadow_reference_check(gpu, scene, "Sun", 144, 81, compared=0.08)


def test_two_shadowed_lights_follow_the_shading_equation(gpu):
    """With shadows, each light contributes only where it is unblocked (checked per light)."""
    scene = Scene()
    scene.add_plane("Ground", size=14).color = "#bbbbbb"
    scene.add_cube("Pillar", size=1, position=(0, 0, 1.5), scale=(1, 1, 3))
    scene.add_directional_light("Key", direction=(0.6, 0.5, -1), intensity=1.2)
    scene.add_directional_light("Rim", direction=(-0.8, 0.2, -0.7), intensity=0.6, color="#a0c0ff")
    scene.world.ambient_intensity = 0.1
    scene.add_camera("Cam", position=(5, -9, 8), look_at=(0, 0, 0), fov=50)
    w, h = 120, 90
    options = {"width": w, "height": h, "samples": 1, "msaa": 1}
    both = gpu.render(scene, shadows=True, **options)
    assert [m["light"] for m in both.backend["shadows"]["maps"]] == ["Key", "Rim"]
    masks, parts = {}, {}
    for name in ("Key", "Rim"):
        light = scene.get_by_name(name)
        masks[name] = shadow_mask(scene, scene.active_camera, w, h, light)
        for other in scene.lights:
            other.visible = other is light
        scene.world.ambient_intensity = 0.0
        parts[name] = gpu.render(scene, **options).linear
        for other in scene.lights:
            other.visible = True
        scene.world.ambient_intensity = 0.1
    for other in scene.lights:
        other.visible = False
    ambient = gpu.render(scene, default_lights=False, **options).linear
    for other in scene.lights:
        other.visible = True
    expected = ambient + sum(parts[n] * ~masks[n][..., None] for n in parts)
    edges = boundary_mask(both.object_index)
    for m in masks.values():
        edges |= dilate(m != np.roll(m, 1, 0), 2) | dilate(m != np.roll(m, 1, 1), 2)
    check = ~edges & (both.object_index >= 0)
    assert check.sum() > 0.3 * w * h
    error = np.abs(both.linear - expected).max(axis=-1)[check]
    assert np.percentile(error, 99.5) < 1e-3


def test_shadow_requests_are_validated_and_reported(gpu):
    scene = floating_box()
    scene.add_point_light("Bulb", position=(3, 3, 3), intensity=2)
    result = gpu.render(scene, width=32, height=32, shadows=True)
    codes = [d.code for d in result.diagnostics]
    assert "POINT_LIGHT_SHADOWS_UNSUPPORTED" in codes
    report = result.backend["shadows"]
    assert report["map_size"] == 2048 and report["pcf_radius"] == 1 and report["lights"] is None
    for shadows, code in (
        ({"lights": ["Bulb"]}, "INVALID_SHADOWS"),
        ({"map_size": 99999}, "INVALID_SHADOWS"),
    ):
        with pytest.raises(RenderError) as info:
            gpu.render(scene, width=32, height=32, shadows=shadows)
        assert info.value.code == code
    huge = gpu.info["limits"]["max-texture-dimension-2d"] * 2
    if huge <= 16384:
        with pytest.raises(RenderError) as info:
            gpu.render(scene, width=32, height=32, shadows={"map_size": huge})
        assert info.value.code == "GPU_LIMIT_EXCEEDED"
    plain = gpu.render(scene, width=32, height=32)
    assert plain.backend["shadows"] is None and plain.stats["shadow_maps"] == 0


def torus_scene(major, minor, sun=(-0.4, 0.56, -0.72)):
    scene = Scene("Torus")
    scene.add_plane("Ground", size=6).color = "#88aa66"
    ring = scene.add_procedural(
        "Ring",
        "torus",
        major_radius=0.9,
        minor_radius=0.28,
        major_segments=major,
        minor_segments=minor,
        position=(0, 0, 0.28),
    )
    ring.color = "#f1c40f"
    scene.add_directional_light("Sun", direction=sun, intensity=2.2)
    scene.world.ambient_intensity = 0.2
    scene.add_camera("Cam", position=(1.5, -3.2, 2.4), look_at=(0, 0, 0.1), fov=40)
    return scene


def test_coarse_smooth_meshes_do_not_shadow_their_own_facets(gpu):
    """A 48x12 torus is shaded smooth: its facets must not cast jagged shadows on itself."""
    scene = torus_scene(48, 12)
    options = {"width": 160, "height": 120, "samples": 1, "msaa": 1}
    plain = gpu.render(scene, **options)
    shaded = gpu.render(scene, shadows=True, **options)
    ring = (plain.object_index == 1) & ~boundary_mask(plain.object_index)
    diff = np.abs(shaded.image.astype(int) - plain.image.astype(int)).max(axis=-1)
    assert (diff[ring] <= 2).mean() > 0.995  # nothing on the ring may cast onto it here
    ground = plain.object_index == 0
    assert (diff[ground] > 20).sum() > 80  # while it does shadow the ground


def test_smooth_self_shadowing_matches_ray_tracing(gpu):
    """On a finer torus under a low sun, its self-shadows agree with exact shadow rays."""
    scene = torus_scene(96, 48, sun=(0.0, 1.0, -0.2))  # low sun: the ring shades itself
    w, h = 128, 96
    sun = scene.get_by_name("Sun")
    ref = shadow_mask(scene, scene.active_camera, w, h, sun)
    options = {"width": w, "height": h, "samples": 1, "msaa": 1}
    lit = gpu.render(scene, **options)
    shaded = gpu.render(scene, shadows=True, **options)
    sun.visible = False
    ambient = gpu.render(scene, **options)
    sunlit = lit.linear.sum(-1) > ambient.linear.sum(-1) + 1e-4  # the sun reaches it if unblocked
    visibility = (shaded.linear.sum(-1) - ambient.linear.sum(-1)) / np.maximum(
        lit.linear.sum(-1) - ambient.linear.sum(-1), 1e-12
    )
    edges = dilate(ref != np.roll(ref, 1, 0), 1) | dilate(ref != np.roll(ref, 1, 1), 1)
    edges |= boundary_mask(lit.object_index)
    check = sunlit & ~edges
    wrong = check & ((visibility < 0.5) != ref)
    assert ref[check & (lit.object_index == 1)].sum() > 30 and check.sum() > 1000
    assert wrong.sum() <= 0.002 * check.sum(), f"{wrong.sum()} of {check.sum()} pixels disagree"
