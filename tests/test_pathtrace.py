"""The path tracer against analytic results and independent references.

Every expected value here is derived independently of the renderer:

* closed-form radiometry (furnace scenes, the inverse-square law, the form
  factor of a rectangle, the geometric series of interreflection in a
  closed glowing room, the cosine-weighted visibility next to a wall);
* the brute-force shadow-ray reference of ``tests/render_reference.py``;
* the scene seen from a mirrored camera (what a perfect mirror must show).

Stochastic estimates are compared with tolerances of several standard
errors, taken from the renderer's own variance pass where noted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from render_reference import shadow_mask  # noqa: E402

import aether3d as a3  # noqa: E402
from aether3d.rendering.color import linear_to_srgb  # noqa: E402
from aether3d.rendering.pathtrace import integrator  # noqa: E402


def dark(scene: a3.Scene) -> a3.Scene:
    """No ambient light, black background: only the scene's own lights."""
    scene.world.configure(background="#000000", ambient_intensity=0.0)
    return scene


def display(linear) -> tuple[float, float, float]:
    """A linear color as the display color that converts back to it exactly."""
    return tuple(float(v) for v in linear_to_srgb(np.asarray(linear, dtype=np.float64)))


def lambert(scene, name, albedo, **extra):
    return scene.create_material(
        name, model="simple", base_color=display(albedo), specular=0.0, **extra
    )


def interior(mask: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` whose 8 neighbours are in it too (away from edges)."""
    pad = np.pad(mask, 1, mode="constant")
    h, w = mask.shape
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
    return out


def render(scene, width, height, *, ao=None, **path):
    return scene.render(
        width=width, height=height, backend="path", path=path, ao=ao, default_lights=False
    )


# --------------------------------------------------------------------------- #
# Closed-form radiometry
# --------------------------------------------------------------------------- #


def test_an_open_lambertian_surface_under_a_uniform_sky_reflects_albedo_times_radiance():
    """White furnace: cosine-sampled sky light on a Lambertian plane has zero variance."""
    scene = a3.Scene()
    plane = scene.add_plane("Plane", size=4)
    plane.material = lambert(scene, "Paint", (0.5, 0.25, 0.75))
    scene.world.configure(environment={"type": "uniform", "color": "#ffffff", "intensity": 2.0})
    scene.add_camera("Cam", position=(0, 0, 3), look_at=(0, 0, 0))
    for bounces in (0, 3):
        r = render(scene, 16, 16, samples=4, bounces=bounces)
        covered = r.coverage == 1.0
        assert covered.sum() > 100
        expected = np.broadcast_to([1.0, 0.5, 1.5], r.linear[covered].shape)
        np.testing.assert_allclose(r.linear[covered], expected, rtol=1e-6)


@pytest.mark.parametrize("bounces", [0, 1, 3])
def test_a_closed_glowing_room_follows_the_geometric_series_of_its_bounces(bounces):
    """Inside a closed room whose walls emit Le and reflect rho, every path vertex adds
    rho times the light of the previous one: L = Le (1 + rho + ... + rho^(bounces+1))."""
    rho, le = 0.6, 0.7
    scene = a3.Scene()
    room = scene.add_cube("Room", size=4)
    room.material = lambert(scene, "Glow", (rho,) * 3, emissive=display((le,) * 3))
    scene.add_camera("Cam", position=(0.3, -0.5, 0.2), look_at=(1, 1, 0.5))
    r = render(scene, 20, 20, samples=48, bounces=bounces, russian_roulette=64)
    expected = le * sum(rho**k for k in range(bounces + 2))
    mean = float(r.linear.mean())
    # Standard error of the image mean from the renderer's own per-pixel variance.
    error = math.sqrt(float(r.passes["variance"].mean()) / r.linear[..., 0].size)
    assert abs(mean - expected) < 5 * error + 1e-9
    assert error < 0.004 * expected
    # Direct + indirect is the image; with no bounces everything is direct.
    np.testing.assert_allclose(
        r.passes["direct"] + r.passes["indirect"], r.passes["beauty"], rtol=1e-6
    )
    if bounces == 0:
        assert float(np.abs(r.passes["indirect"]).max()) == 0.0


def test_unshadowed_direct_light_matches_the_rasterizer_exactly():
    """bounces=0 under a directional light, seen orthographically from straight above."""
    scene = dark(a3.Scene())
    ground = scene.add_plane("Ground", size=20)
    ground.material = scene.create_material(
        "Gloss", base_color="#b07040", metallic=0.3, roughness=0.4
    )
    scene.add_directional_light("Sun", direction=(0.3, -0.4, -1.0), intensity=1.7)
    cam = scene.add_camera(
        "Top", projection="orthographic", ortho_scale=4, position=(0, 0, 5), look_at=(0, 0, 0)
    )
    cam.camera.near = 0.1
    raster = scene.render(width=12, height=12, samples=1, default_lights=False)
    path = render(scene, 12, 12, samples=2, bounces=0)
    np.testing.assert_allclose(path.linear, raster.linear, rtol=1e-6)


def test_a_point_light_follows_the_inverse_square_law():
    """Radiance of a Lambertian floor under a point light: rho I cos(theta) / d^2."""
    scene = dark(a3.Scene())
    floor = scene.add_plane("Floor", size=8)
    floor.material = lambert(scene, "White", (0.8, 0.8, 0.8))
    height, intensity = 1.5, 3.0
    scene.add_point_light("Bulb", position=(0, 0, height), intensity=intensity)
    scene.add_camera(
        "Top", projection="orthographic", ortho_scale=4, position=(0, 0, 5), look_at=(0, 0, 0)
    )
    n = 32
    r = render(scene, n, n, samples=256, bounces=0)
    # The expected pixel value: the formula averaged over each pixel's area (8x8 points).
    sub = (np.arange(8) + 0.5) / 8
    xs = ((np.arange(n)[:, None] + sub[None, :]) / n * 4 - 2).ravel()
    ys = (2 - (np.arange(n)[:, None] + sub[None, :]) / n * 4).ravel()
    gx, gy = np.meshgrid(xs, ys)
    d2 = gx * gx + gy * gy + height * height
    value = 0.8 * intensity * (height / np.sqrt(d2)) / d2
    expected = value.reshape(n, 8, n, 8).mean(axis=(1, 3))
    # Random positions inside each pixel vary by up to ~5% across it: about 0.3% noise.
    np.testing.assert_allclose(r.linear[..., 0], expected, rtol=0.015)
    assert abs(float(r.linear[..., 0].mean() / expected.mean()) - 1.0) < 0.002


def test_a_square_emitter_lights_the_floor_below_by_its_form_factor():
    """A Lambertian floor under an emissive square: L = rho Le F, with F the form factor
    of the square seen from the point (sum of four corner rectangles)."""
    scene = dark(a3.Scene())
    floor = scene.add_plane("Floor", size=10)
    floor.material = lambert(scene, "Floor", (0.5, 0.5, 0.5))
    side, height, le = 1.0, 0.8, 2.0
    lamp = scene.add_plane("Lamp", size=side, position=(0, 0, height))
    lamp.rotation_euler = (180, 0, 0)  # facing down
    lamp.material = scene.create_material(
        "Lamp",
        model="simple",
        base_color="#000000",
        emissive="#ffffff",
        emissive_strength=le,
        double_sided=False,
    )
    scene.add_camera(
        "Top", projection="orthographic", ortho_scale=0.1, position=(0, 0, 0.5), look_at=(0, 0, 0)
    )
    r = render(scene, 4, 4, samples=256, bounces=0)

    def corner(x, y):  # point below a corner of an x-by-y rectangle (units of height)
        a, b = x / math.sqrt(1 + x * x), y / math.sqrt(1 + y * y)
        return (
            a * math.atan(y / math.sqrt(1 + x * x)) + b * math.atan(x / math.sqrt(1 + y * y))
        ) / (2 * math.pi)

    q = side / 2 / height
    expected = 0.5 * le * 4 * corner(q, q)
    mean = float(r.linear[..., 0].mean())
    error = math.sqrt(float(r.passes["variance"][..., 0].mean()) / 16)
    assert abs(mean - expected) < 5 * error
    assert error < 0.01 * expected


def test_emissive_surfaces_are_seen_at_their_radiance():
    scene = dark(a3.Scene())
    glow = scene.add_plane("Glow", size=4)
    glow.material = scene.create_material(
        "Neon", base_color="#000000", emissive=display((0.2, 0.9, 0.4)), emissive_strength=3.0
    )
    scene.add_camera("Cam", position=(0, 0, 2), look_at=(0, 0, 0))
    r = render(scene, 8, 8, samples=2, bounces=2)
    np.testing.assert_allclose(r.linear[2:6, 2:6], np.full((4, 4, 3), [0.6, 2.7, 1.2]), rtol=1e-6)


def test_the_sky_is_seen_and_lights_the_scene():
    scene = a3.Scene()
    scene.world.configure(
        environment={
            "type": "sky",
            "zenith_color": display((0.1, 0.2, 0.8)),
            "horizon_color": display((0.8, 0.8, 0.8)),
            "ground_color": display((0.3, 0.2, 0.1)),
            "intensity": 1.5,
        }
    )
    scene.add_camera("Up", position=(0, 0, 0), look_at=(0, 0, 10), up=(0, 1, 0), fov=0.2)
    r = render(scene, 3, 3, samples=1)
    np.testing.assert_allclose(r.linear[1, 1], [0.15, 0.3, 1.2], rtol=1e-4)
    assert r.coverage.max() == 0.0
    # A Lambertian plane facing up sees the upper hemisphere of the sky only.
    plane = scene.add_plane("Plane", size=4)
    plane.material = lambert(scene, "Grey", (0.5, 0.5, 0.5))
    scene.add_camera("Down", position=(0, 0, 2), look_at=(0, 0, 0))
    lit = scene.render(
        camera="Down",
        width=8,
        height=8,
        backend="path",
        path={"samples": 1024, "bounces": 0},
        default_lights=False,
    )
    # Irradiance of the sky over the upper hemisphere: E = 2 pi int_0^1 L(z) z dz with
    # L = horizon + (zenith - horizon) sqrt(z): int z dz = 1/2, int z^1.5 dz = 2/5.
    zen, hor = np.array([0.1, 0.2, 0.8]) * 1.5, np.array([0.8, 0.8, 0.8]) * 1.5
    irradiance = 2 * math.pi * (hor / 2 + (zen - hor) * 2 / 5)
    expected = 0.5 / math.pi * irradiance
    # 65536 cosine-weighted samples of a sky varying 8x from horizon to zenith: ~0.4% noise.
    covered = lit.coverage == 1.0
    np.testing.assert_allclose(lit.linear[covered].mean(axis=0), expected, rtol=0.015)


# --------------------------------------------------------------------------- #
# Shadows
# --------------------------------------------------------------------------- #


def shadow_scene(angle: float = 0.0) -> a3.Scene:
    scene = dark(a3.Scene())
    floor = scene.add_plane("Floor", size=12)
    floor.material = lambert(scene, "Floor", (0.7, 0.7, 0.7))
    box = scene.add_cube("Box", size=1.2, position=(0.2, 0.4, 1.0))
    box.rotation_euler = (0, 0, 30)
    box.material = lambert(scene, "Box", (0.7, 0.3, 0.2))
    scene.add_directional_light("Sun", direction=(1.2, 0.5, -1.0), intensity=1.0, angle=angle)
    scene.add_camera(
        "Top", projection="orthographic", ortho_scale=6, position=(0, 0, 8), look_at=(0, 0, 0)
    )
    return scene


def test_hard_shadows_match_an_independent_shadow_ray_reference():
    scene = shadow_scene()
    n = 40
    r = render(scene, n, n, samples=16, bounces=0)
    reference = shadow_mask(scene, scene.active_camera, n, n, scene.get_by_name("Sun"))
    floor = interior(r.object_index == 0)
    # Pixels whose neighbourhood is entirely shadowed (or lit) in the reference.
    pad = np.pad(reference, 1, mode="edge")
    around = [
        pad[1 + dy : n + 1 + dy, 1 + dx : n + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ]
    inside = np.logical_and.reduce(around)
    outside = ~np.logical_or.reduce(around)
    lit_value = 0.7 * (1.0 / np.linalg.norm([1.2, 0.5, -1.0]))
    assert (floor & inside).sum() > 20 and (floor & outside).sum() > 200
    assert np.all(r.linear[floor & inside] == 0.0)
    np.testing.assert_allclose(r.linear[floor & outside][:, 0], lit_value, rtol=1e-6)


def test_light_size_softens_shadows_but_keeps_the_light_strength():
    n = 48
    hard = render(shadow_scene(0.0), n, n, samples=32, bounces=0)
    soft = render(shadow_scene(12.0), n, n, samples=32, bounces=0)
    floor = hard.object_index == 0
    lit = hard.linear[..., 0][floor].max()
    # Open floor far from the box: the same irradiance from the point and the disk sun
    # (per sample, the disk's cosine varies by about 13% here: compare 512 samples).
    np.testing.assert_allclose(
        soft.linear[:4, :4].mean(axis=(0, 1)), hard.linear[:4, :4].mean(axis=(0, 1)), rtol=0.01
    )
    partial_hard = np.count_nonzero(
        (hard.linear[..., 0] > 0.05 * lit) & (hard.linear[..., 0] < 0.95 * lit) & floor
    )
    partial_soft = np.count_nonzero(
        (soft.linear[..., 0] > 0.05 * lit) & (soft.linear[..., 0] < 0.95 * lit) & floor
    )
    assert partial_soft > 2 * partial_hard  # a penumbra appears


# --------------------------------------------------------------------------- #
# Reflections and global illumination
# --------------------------------------------------------------------------- #


def mirror_scenes(roughness: float):
    """A mirror floor below a glowing box, and the box with its mirror image instead."""
    scenes = []
    for mirrored in (False, True):
        scene = dark(a3.Scene())
        neon = scene.create_material(
            "Neon", base_color="#000000", emissive=display((0.9, 0.2, 0.1)), emissive_strength=1.0
        )
        for z in (1.0, -1.0) if mirrored else (1.0,):
            box = scene.add_cube(f"Box {z}", size=0.8, position=(0.3, 0.5, z))
            box.rotation_euler = (0, 0, 25)
            box.material = neon
        if not mirrored:
            floor = scene.add_plane("Mirror", size=40)
            floor.material = scene.create_material(
                "Chrome", base_color="#ffffff", metallic=1.0, roughness=roughness
            )
        scene.add_camera("Cam", position=(0, -5, 2.0), look_at=(0, 0, 0.3))
        scenes.append(scene)
    return scenes


def test_a_mirror_shows_the_scene_as_seen_from_the_mirrored_camera():
    with_mirror, mirror_image = mirror_scenes(roughness=0.0)
    a = render(with_mirror, 48, 32, samples=16, bounces=1)
    b = render(mirror_image, 48, 32, samples=16, bounces=1)
    red_a = a.linear[..., 0] > 0.3
    red_b = b.linear[..., 0] > 0.3
    below = np.zeros_like(red_a)
    below[18:] = True  # the reflection is below the horizon line
    assert (red_b & below).sum() > 40  # the mirror image is in view
    overlap = (red_a & red_b).sum() / max((red_a | red_b).sum(), 1)
    assert overlap > 0.95
    # Away from the outlines the reflected radiance is the box's own (F = 1 for white
    # metal; a roughness-0 surface still has the GGX floor 0.045, a ~0.1 degree blur).
    both = interior(interior(red_a & red_b))
    assert both.sum() > 20
    np.testing.assert_allclose(a.linear[both], b.linear[both], rtol=0.02, atol=0.01)
    assert float(a.linear[both].sum() / b.linear[both].sum()) == pytest.approx(1.0, abs=0.01)


def test_rough_metal_blurs_the_reflection():
    sharp = render(mirror_scenes(0.0)[0], 48, 32, samples=32, bounces=1)
    rough = render(mirror_scenes(0.45)[0], 48, 32, samples=32, bounces=1)
    reflection = (slice(18, 32), slice(None))
    gradient = lambda img: np.abs(np.diff(img[reflection][..., 0], axis=1)).max()  # noqa: E731
    assert gradient(rough.linear) < 0.6 * gradient(sharp.linear)
    assert rough.linear[reflection][..., 0].sum() > 0.5 * sharp.linear[reflection][..., 0].sum()


def cornell_box() -> a3.Scene:
    scene = dark(a3.Scene())
    white = lambert(scene, "White", (0.75, 0.75, 0.75))
    red = lambert(scene, "Red", (0.75, 0.1, 0.1))
    green = lambert(scene, "Green", (0.1, 0.75, 0.1))
    walls = {
        "Floor": ((0, 0, 0), (0, 0, 0), white),
        "Ceiling": ((0, 0, 2), (180, 0, 0), white),
        "Back": ((0, 1, 1), (90, 0, 0), white),
        "Left": ((-1, 0, 1), (0, 90, 0), red),
        "Right": ((1, 0, 1), (0, -90, 0), green),
    }
    for name, (position, rotation, material) in walls.items():
        wall = scene.add_plane(name, size=2, position=position)
        wall.rotation_euler = rotation
        wall.material = material
    lamp = scene.add_plane("Lamp", size=0.5, position=(0, 0, 1.99))
    lamp.rotation_euler = (180, 0, 0)
    lamp.material = scene.create_material(
        "Lamp",
        model="simple",
        base_color="#000000",
        emissive="#ffffff",
        emissive_strength=12.0,
        double_sided=False,  # emits downward only
    )
    scene.add_camera("Cam", position=(0, -3.2, 1), look_at=(0, 0, 1), fov=40)
    return scene


def test_a_cornell_box_bleeds_the_wall_colors_onto_white_surfaces():
    scene = cornell_box()
    r = render(scene, 32, 32, samples=32, bounces=3)
    floor = r.object_index == 0
    ceiling = r.object_index == 1
    cols = np.arange(32)[None, :].repeat(32, axis=0)
    left = (floor | ceiling) & (cols < 10)
    right = (floor | ceiling) & (cols > 21)
    indirect = r.passes["indirect"]
    # Near the red wall the bounced light is red, near the green wall green.
    assert indirect[left][:, 0].mean() > 1.5 * indirect[left][:, 1].mean()
    assert indirect[right][:, 1].mean() > 1.5 * indirect[right][:, 0].mean()
    # The light itself is white: direct light on white surfaces has no such tint.
    direct = r.passes["direct"]
    lit = interior(floor | ceiling) & left & (direct[..., 0] > 0)
    assert lit.sum() > 10
    np.testing.assert_allclose(direct[lit][:, 0], direct[lit][:, 1], rtol=1e-9)
    # The ceiling gets no direct light (the lamp faces down) but is lit by bounces
    # (pixels next to the lamp also see the lamp itself).
    inner = interior(ceiling)
    assert inner.sum() > 10
    assert float(direct[inner].max()) == 0.0 and float(indirect[inner].mean()) > 0.01
    # Bounced light adds a lot on the walls (away from the lamp itself).
    direct_only = render(scene, 32, 32, samples=32, bounces=0)
    walls = interior(r.object_index <= 4)
    assert r.linear[walls].mean() > 1.3 * direct_only.linear[walls].mean()


# --------------------------------------------------------------------------- #
# Ambient occlusion
# --------------------------------------------------------------------------- #


def test_ao_next_to_a_wall_is_the_blocked_part_of_the_cosine_lobe():
    """Cosine-weighted directions project to a uniform disk (Malley), so the occluded
    fraction at distance x from a tall wall with AO reach D is the disk segment beyond
    c = x / D: (acos c - c sqrt(1 - c^2)) / pi."""
    scene = dark(a3.Scene())
    scene.add_plane("Floor", size=20)
    wall = scene.add_plane("Wall", size=20, position=(0, 0, 10))
    wall.rotation_euler = (0, 90, 0)  # the plane x = 0
    distance = 1.0
    xs = [0.0, 0.25, 0.5, 0.75, 1.2]
    # One orthographic pixel per sample point, looking straight down.
    values = []
    for x in xs:
        cam = scene.add_camera(
            f"At {x}",
            projection="orthographic",
            ortho_scale=1e-3,
            position=(x + 1e-6, 0, 3),
            look_at=(x + 1e-6, 0, 0),
        )
        r = scene.render(
            camera=cam,
            width=1,
            height=1,
            backend="path",
            path={"samples": 1024, "integrator": "ao"},
            ao={"samples": 8, "distance": distance},
        )
        values.append(float(r.passes["ao"][0, 0]))
    for x, value in zip(xs, values, strict=True):
        c = min(x / distance, 1.0)
        expected = 1.0 - (math.acos(c) - c * math.sqrt(1 - c * c)) / math.pi
        assert value == pytest.approx(expected, abs=0.02)
    assert values[-1] == 1.0  # beyond the reach: exactly unoccluded


def test_ao_is_one_on_open_surfaces_and_darkens_contacts():
    scene = dark(a3.Scene())
    scene.add_plane("Floor", size=20)
    scene.add_cube("A", size=1, position=(0, 0, 0.5))
    scene.add_cube("B", size=1, position=(1.0, 0, 0.5))  # touching A
    scene.add_cube("Far", size=1, position=(-6, 0, 3.0))  # 2.5 above the floor
    scene.add_camera(
        "Top", projection="orthographic", ortho_scale=10, position=(0, 0, 9), look_at=(0, 0, 0)
    )
    r = scene.render(
        width=40,
        height=40,
        backend="path",
        path={"samples": 32, "integrator": "ao"},
        ao={"samples": 4, "distance": 1.0},
    )
    ao = r.passes["ao"]
    floor = r.object_index == 0
    # Floor far from every box (more than the AO reach away): exactly 1.
    assert np.all(ao[:6, 30:][floor[:6, 30:]] == 1.0)
    # Under the box hovering 2.5 above the floor (beyond the 1.0 reach): exactly 1.
    under = (slice(18, 22), slice(3, 6))
    assert np.all(ao[under] == 1.0)
    # The floor right next to the boxes is darkened.
    beside = ao[floor & (np.abs(np.arange(40)[:, None] - 20) <= 3) & (np.arange(40)[None, :] == 18)]
    assert beside.size and beside.mean() < 0.85
    # strength scales occlusion; the AO preview image is the AO pass.
    weak = scene.render(
        width=40,
        height=40,
        backend="path",
        path={"samples": 32, "integrator": "ao"},
        ao={"samples": 4, "distance": 1.0, "strength": 0.5},
    )
    np.testing.assert_allclose(1 - weak.passes["ao"], 0.5 * (1 - ao), atol=1e-12)
    covered = r.coverage == 1.0
    np.testing.assert_allclose(r.passes["beauty"][covered][:, 0], ao[covered], atol=1e-12)


# --------------------------------------------------------------------------- #
# Sampling, convergence and reproducibility
# --------------------------------------------------------------------------- #


def test_error_falls_with_more_samples():
    rho, le = 0.5, 1.0
    scene = a3.Scene()
    room = scene.add_cube("Room", size=4)
    room.material = lambert(scene, "Glow", (rho,) * 3, emissive=display((le,) * 3))
    scene.add_camera("Cam", position=(0.3, -0.5, 0.2), look_at=(1, 1, 0.5))
    expected = le * (1 + rho + rho**2)
    errors = []
    for samples in (2, 8, 32):
        r = render(scene, 16, 16, samples=samples, bounces=1, russian_roulette=64)
        errors.append(float(np.sqrt(((r.linear[..., 0] - expected) ** 2).mean())))
    assert errors[1] < 0.7 * errors[0] and errors[2] < 0.7 * errors[1]
    # Close to 1 / sqrt(samples): a factor 2 per four times the samples.
    assert errors[0] / errors[2] == pytest.approx(4.0, rel=0.35)


def test_renders_repeat_bit_for_bit_and_do_not_depend_on_batching(monkeypatch):
    scene = cornell_box()
    a = render(scene, 12, 12, samples=6, bounces=2, seed=5)
    b = render(scene, 12, 12, samples=6, bounces=2, seed=5)
    assert np.array_equal(a.linear, b.linear) and np.array_equal(a.image, b.image)
    other = render(scene, 12, 12, samples=6, bounces=2, seed=6)
    assert not np.array_equal(a.linear, other.linear)
    monkeypatch.setattr(integrator, "BATCH", 50)
    small = render(scene, 12, 12, samples=6, bounces=2, seed=5)
    assert np.array_equal(a.linear, small.linear)
    for name in a.passes:
        assert np.array_equal(a.passes[name], small.passes[name])
