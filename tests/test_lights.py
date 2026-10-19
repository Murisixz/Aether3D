"""Spot and area lights, power units, colour temperature, exposure and per-light shadows
(0.20.0) on every backend, measured against Blender 5.2 Cycles: a white Lambertian plane
under one light (tests/data/blender_lights; research/2026-10-stage19/blender/e3_lights)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene
from aether3d.rendering.blackbody import blackbody
from aether3d.rendering.gpu import gpu_status
from aether3d.rendering.scene_data import extract_render_scene, resolve_camera
from aether3d.rendering.settings import ShadowSettings
from aether3d.rendering.shadows import select_shadow_lights
from aether3d.serialization import scene_from_dict, scene_to_dict

GPU = gpu_status().get("available", False)
DATA = json.loads((Path(__file__).parent / "data" / "blender_lights" / "lights.json").read_text())
RASTERS = ["cpu", *(["gpu"] if GPU else [])]
TRACERS = ["path", *(["gpu_path"] if GPU else [])]
AREAS = [("square", 0.5, 0.5), ("rectangle", 0.5, 1.0), ("disk", 0.5, 0.5), ("ellipse", 0.5, 1.0)]


def floor_scene(scale: float) -> Scene:
    """Blender's setup: a large white plane at z = 0 seen from straight above."""
    scene = Scene()
    floor = scene.add_cube("Floor", size=1.0, position=(0, 0, -0.5))
    floor.scale = (200, 200, 1)
    floor.color = "#ffffff"
    scene.add_camera("Top", projection="orthographic", ortho_scale=scale, position=(0, 0, 50),
                     look_at=(0, 0, 0), near=1, far=100)  # fmt: skip
    scene.world.ambient_intensity = 0.0
    return scene


def render(
    scene: Scene, backend: str, samples: int = 256, size: int = 33, centres: bool = False
) -> np.ndarray:
    """Red channel. Rasterizers sample each pixel's centre, as Blender's references do; the
    path tracers average its area -- with ``centres``, over a fifth of it (the image at
    five times the size, sampled at the centres of the requested pixels)."""
    if "path" not in backend:
        r = scene.render(width=size, height=size, backend=backend, default_lights=False,
                         samples=1)  # fmt: skip
        return r.linear[..., 0].astype(np.float64)
    k = 5 if centres else 1
    r = scene.render(width=k * size, height=k * size, backend=backend, default_lights=False,
                     path={"samples": samples, "bounces": 0})  # fmt: skip
    return r.linear[k // 2 :: k, k // 2 :: k, 0].astype(np.float64)


def settled(mask: np.ndarray) -> np.ndarray:
    """Pixels whose 3 x 3 neighbourhood is all in ``mask``: away from a cone's edge, where
    Blender's pixel centres and the path tracers' pixel areas see different coverage."""
    keep = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            keep &= np.roll(np.roll(mask, dy, 0), dx, 1)
    return keep


def area_scene(shape: str, sx: float, sy: float) -> Scene:
    scene = floor_scene(4.0)
    scene.add_area_light("Panel", position=(0, 0, 1), shape=shape, size=sx, size_y=sy,
                         intensity=100.0, units="watts")  # fmt: skip
    return scene


def spot_scene(blend: float) -> Scene:
    scene = floor_scene(4.0)
    scene.add_spot_light("Spot", position=(0, 0, 2), spot_size=60.0, spot_blend=blend,
                         intensity=1000.0, units="watts")  # fmt: skip
    return scene


@pytest.mark.parametrize("backend", RASTERS)
@pytest.mark.parametrize(("shape", "sx", "sy"), AREAS)
def test_the_rasterizers_integrate_area_lights_as_blender_measures_them(backend, shape, sx, sy):
    # Lambert's polygon formula: exact for squares and rectangles, disks and ellipses as
    # 16-gons of the same area. Measured: 2e-4 at most, Blender's own noise.
    ref = np.array(DATA["images"][f"area_{shape.upper()}_P100_h1_spread180"])
    ours = render(area_scene(shape, sx, sy), backend)
    assert np.allclose(ours, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("backend", TRACERS)
def test_the_path_tracers_converge_to_blenders_area_lights(backend):
    for shape, sx, sy in AREAS:
        ref = np.array(DATA["images"][f"area_{shape.upper()}_P100_h1_spread180"])
        ours = render(area_scene(shape, sx, sy), backend)
        # 256 samples: about 1.5 % noise per pixel, 0.05 % over the image.
        assert ours.sum() == pytest.approx(ref.sum(), rel=0.004), shape
        assert np.median(np.abs(ours - ref) / ref) < 0.025, shape


def spread_scene(shape: str, sx: float, sy: float) -> Scene:
    scene = floor_scene(4.0)
    scene.add_area_light("Panel", position=(0, 0, 1), shape=shape, size=sx, size_y=sy,
                         intensity=100.0, units="watts", spread=30.0)  # fmt: skip
    return scene


@pytest.mark.parametrize("backend", RASTERS)
def test_the_rasterizers_integrate_spread_lights_within_the_measured_error(backend):
    # A slat grid's emission, max(1 - tan a / tan 15 deg, 0) normalized to keep the power,
    # is what Blender's spread measures as; the rasterizers integrate it with a 12 x 12
    # midpoint rule (measured: median 0.13-0.44 %, 4.9 % at most at the footprint's rim).
    for shape, sx, sy in AREAS:
        ref = np.array(DATA["images"][f"area_{shape.upper()}_P100_h1_spread30"])
        ours = render(spread_scene(shape, sx, sy), backend)
        lit = ref > 1e-3 * ref.max()
        assert ((ours > 1e-3 * ref.max()) == lit).all(), shape  # the same footprint
        rel = np.abs(ours - ref)[lit] / ref[lit]
        assert np.median(rel) < 0.006 and rel.max() < 0.06, shape
        assert ours.sum() == pytest.approx(ref.sum(), rel=0.004), shape


@pytest.mark.parametrize("backend", TRACERS)
def test_the_path_tracers_converge_to_blenders_spread_lights(backend):
    # Only the central 15 x 15 of Blender's pixels see the light: render them (camera
    # narrowed) at five times the size and keep their centres.
    samples = 1024 if backend == "gpu_path" else 256
    for shape, sx, sy in AREAS if backend == "gpu_path" else AREAS[::3]:
        ref = np.array(DATA["images"][f"area_{shape.upper()}_P100_h1_spread30"])
        lit = ref > 1e-3 * ref.max()
        assert lit[9:24, 9:24].sum() == lit.sum()
        scene = spread_scene(shape, sx, sy)
        scene.get_by_name("Top").camera.ortho_scale = 4.0 * 15 / 33
        ours = render(scene, backend, samples=samples, size=15, centres=True)
        assert not ours[settled(~lit[9:24, 9:24])].any(), shape  # nothing outside the footprint
        # Uniform samples of the part reaching each point: relative noise about 1.35 per
        # sample, so 0.6 % (1024 samples) and 1.1 % (256) over the footprint's 70-105 pixels.
        tol = 0.025 if backend == "gpu_path" else 0.045
        assert ours.sum() == pytest.approx(ref[9:24, 9:24].sum(), rel=tol), shape


@pytest.mark.parametrize("backend", RASTERS + TRACERS)
@pytest.mark.parametrize("blend", [0.0, 0.3])
def test_spot_cones_and_their_soft_edges_match_blender(backend, blend):
    ref = np.array(DATA["images"][f"spot_P1000_h2_size60_blend{blend}"])
    ours = render(spot_scene(blend), backend, samples=64, centres=True)
    inside, outside = settled(ref > 1e-3), settled(ref <= 1e-3)
    # Path tracers: the pixel jitter's noise on the soft edge's steep ramp (measured with
    # 64 samples: 0.16 % at most with a hard edge, 1.15 % on the ramp of blend 0.3).
    tol = 1e-3 if backend in RASTERS else 0.005 if blend == 0.0 else 0.025
    assert np.allclose(ours[inside], ref[inside], rtol=tol)
    assert not ours[outside].any()  # no light outside the cone


@pytest.mark.parametrize("backend", RASTERS + TRACERS)
def test_watts_suns_and_point_lights_match_blender(backend):
    scene = floor_scene(1.0)
    scene.add_directional_light("Sun", direction=(0, 0, -1), intensity=1.0, units="watts")
    sun = render(scene, backend, samples=4, size=5)
    assert sun[2, 2] == pytest.approx(DATA["scalars"]["sun_E1_angle0"], rel=2e-4)  # 1 / pi
    scene = floor_scene(8.0)
    scene.add_point_light("Bulb", position=(0, 0, 2), intensity=1000.0, units="watts")
    ref = np.array(DATA["images"]["point_profile_scale8"])
    ours = render(scene, backend, samples=64, centres=True)
    # Path tracers: the jitter's noise where the light falls off fastest (0.1 % per pixel).
    tol = 2e-4 if backend in RASTERS else 0.006
    assert np.allclose(ours, ref, rtol=tol, atol=1e-5)


def test_sized_lights_keep_the_light_on_facing_surfaces_blenders_spheres_do_not():
    # A sun's size keeps its irradiance in both engines. A point light's radius r keeps
    # Aether3D's (a uniform sphere's: P / (4 pi d^2)); Blender 5.2's gives d^2 / (d^2 + r^2)
    # of it, as a disk facing the point would (measured: 0.94125 for r = 0.5, d = 2).
    s = DATA["scalars"]
    assert s["sun_E1_angle30"] == pytest.approx(s["sun_E1_angle0"], rel=1e-4)
    assert s["point_P1000_h2_r0.5"] / s["point_P1000_h2_r0.0"] == pytest.approx(4 / 4.25, abs=1e-3)
    values = []
    for radius in (0.0, 0.5):
        scene = floor_scene(0.2)
        scene.add_point_light("Bulb", position=(0, 0, 2), radius=radius, intensity=1000.0,
                              units="watts")  # fmt: skip
        values.append(render(scene, "path", samples=256, size=3)[1, 1])
    assert values[0] == pytest.approx(s["point_P1000_h2_r0.0"], rel=2e-3)
    assert values[1] == pytest.approx(values[0], rel=0.01)


def test_black_bodies_match_blenders_temperatures():
    lum = np.array([0.2126, 0.7152, 0.0722])
    for kelvin, theirs in DATA["blackbody"].items():
        t = float(kelvin)
        if not 800 <= t <= 12000:
            continue
        ours = np.array(blackbody(t))
        theirs = np.array(theirs) / (lum @ np.array(theirs))  # Blender's node: luminance 1 too?
        assert lum @ ours == pytest.approx(1.0, abs=1e-9)
        # Measured: 1.1 % at most from 2,700 K, 2 % at 2,000 K, 2.6 % at 1,850 K and 6.2 %
        # at 1,000-1,500 K (where the CIE fit is least accurate); 800 K: the same hue.
        tol = 0.012 if t >= 2700 else 0.021 if t >= 2000 else 0.027 if t >= 1850 else 0.065
        big = theirs > 0.05
        assert np.allclose(ours[big], theirs[big], rtol=tol), kelvin
    # A sun tinted by its temperature lights the plane with luminance 1 / pi.
    for kelvin, theirs in DATA["sun_temperature"].items():
        scene = floor_scene(1.0)
        scene.add_directional_light("Sun", direction=(0, 0, -1), intensity=1.0, units="watts",
                                    temperature=float(kelvin), use_temperature=True)  # fmt: skip
        r = scene.render(width=3, height=3, backend="cpu", default_lights=False).linear[1, 1]
        assert np.allclose(r, theirs, rtol=0.012), kelvin


@pytest.mark.parametrize("backend", RASTERS + TRACERS)
def test_area_lights_are_one_sided(backend):
    scene = floor_scene(4.0)
    scene.add_area_light("Panel", position=(0, 0, 1), direction=(0, 0, 1), size=0.5,
                         intensity=100.0, units="watts")  # fmt: skip
    assert not render(scene, backend, samples=16, size=9).any()


def test_exposure_and_units_scale_the_intensity():
    base = floor_scene(1.0)
    base.add_point_light("Bulb", position=(0, 0, 1), intensity=2.0)
    doubled = floor_scene(1.0)
    doubled.add_point_light("Bulb", position=(0, 0, 1), intensity=2.0, exposure=1.0)
    a, b = (render(s, "cpu", size=3)[1, 1] for s in (base, doubled))
    assert b == pytest.approx(2 * a, rel=1e-12)
    light = base.get_by_name("Bulb").light
    light.units = "watts"
    assert light.intensity_value() == pytest.approx(2.0 / (4 * math.pi**2))
    for kind, factor in (("directional", 1 / math.pi), ("area", 1 / math.pi**2)):
        light.type = kind
        assert light.intensity_value() == pytest.approx(2.0 * factor)


def test_lights_without_shadows_are_never_blocked():
    def scene_with(shadow: bool) -> Scene:
        scene = floor_scene(1.0)
        scene.add_spot_light("Spot", position=(-2, 0, 2), look_at=(0, 0, 0), spot_size=60.0,
                             intensity=50.0, shadow=shadow)  # fmt: skip
        scene.add_cube("Blocker", size=0.3, position=(-1, 0, 1))
        return scene

    lit, blocked = (render(scene_with(s), "path", samples=16, size=5)[2, 2] for s in (False, True))
    assert blocked == 0.0 < lit
    if GPU:
        assert render(scene_with(False), "gpu_path", samples=16, size=5)[2, 2] == pytest.approx(
            lit, rel=1e-4
        )
    scene = floor_scene(1.0)
    scene.add_directional_light("Sun", direction=(0, 0, -1), shadow=False)
    rs = extract_render_scene(scene, resolve_camera(scene, None), aspect=1.0)
    chosen, _ = select_shadow_lights(rs, ShadowSettings())
    assert chosen == []


@pytest.mark.skipif(not GPU, reason="needs a GPU")
def test_the_gpu_path_tracer_traces_the_cpu_paths_for_spots_and_areas():
    scene = floor_scene(4.0)
    scene.add_spot_light("Spot", position=(0.5, 0, 2), spot_size=50.0, radius=0.2,
                         intensity=600.0, units="watts")  # fmt: skip
    scene.add_area_light("Panel", position=(-0.5, 0, 1), shape="disk", size=0.6,
                         intensity=80.0, units="watts")  # fmt: skip
    cpu, gpu = (render(scene, b, samples=32, size=48) for b in ("path", "gpu_path"))
    diff = np.abs(cpu - gpu) / max(float(cpu.max()), 1e-12)
    assert np.median(diff) < 1e-5 and np.percentile(diff, 99) < 2e-3


def test_new_light_fields_save_undo_and_animate():
    scene = floor_scene(4.0)
    obj = scene.add_spot_light("Spot", position=(0, 0, 2), spot_size=40.0, temperature=3000.0,
                               use_temperature=True, intensity=500.0, units="watts")  # fmt: skip
    obj.light.spot_size = 80.0
    scene.history.undo()
    assert obj.light.spot_size == 40.0
    copy = scene_from_dict(scene_to_dict(scene))
    assert copy.get_by_name("Spot").light == obj.light
    old = scene_to_dict(scene)
    for o in old["scene"]["objects"]:
        if o.get("light"):
            for key in ("spot_size", "spot_blend", "shape", "size", "size_y", "units",
                        "temperature", "use_temperature", "exposure", "shadow"):  # fmt: skip
                o["light"].pop(key)
            o["light"]["type"] = "point"
    legacy = scene_from_dict(old).get_by_name("Spot").light
    assert legacy.units == "aether" and not legacy.use_temperature and legacy.shadow
    scene.animation.set_channel(obj, "light.spot_size", [(0, 20.0), (1, 90.0)])
    narrow, wide = (
        float((scene.render(width=33, height=33, backend="cpu", default_lights=False, time=t)
               .linear[..., 0] > 0).sum())
        for t in (0.0, 1.0)
    )  # fmt: skip
    assert narrow < wide


def linked_scene(**links) -> tuple[Scene, dict]:
    scene = floor_scene(4.0)
    a = scene.add_cube("A", size=0.6, position=(-0.8, 0, 0.3))
    b = scene.add_cube("B", size=0.6, position=(0.8, 0, 0.3))
    hat = scene.add_cube("Hat", size=0.3, position=(0.8, 0, 0.75))
    hat.set_parent(b, keep_world_transform=True)
    ids = {"A": a.id, "B": b.id, "Hat": hat.id, "Floor": scene.get_by_name("Floor").id}
    scene.add_spot_light("Key", position=(0, 0, 3), spot_size=120.0, intensity=20.0,
                         **{k: [ids[n] for n in v] for k, v in links.items()})  # fmt: skip
    return scene, ids


def lit_objects(scene: Scene, backend: str) -> dict[str, float]:
    """Mean red radiance inside each object's pixels (away from its outline, where the
    path tracers' pixels also see what is next to it)."""
    opts = {"path": {"samples": 8, "bounces": 0}} if "path" in backend else {"samples": 1}
    r = scene.render(width=96, height=96, backend=backend, default_lights=False, **opts)
    index = {o["name"]: k for k, o in enumerate(r.objects)}
    return {
        name: float(r.linear[..., 0][settled(r.object_index == index[name])].mean())
        for name in ("A", "B", "Hat", "Floor")
    }


@pytest.mark.parametrize("backend", RASTERS + TRACERS)
def test_light_linking_lights_only_the_linked_objects_everywhere(backend):
    plain = lit_objects(linked_scene()[0], backend)
    assert all(v > 0.05 for v in plain.values())
    excluded = lit_objects(linked_scene(exclude=["A"])[0], backend)
    assert excluded["A"] == 0.0 and excluded["B"] > 0.05 and excluded["Floor"] > 0.05
    only = lit_objects(linked_scene(include=["B"])[0], backend)  # B and its descendant
    assert only["A"] == 0.0 and only["Floor"] == 0.0
    assert only["B"] == pytest.approx(plain["B"], rel=1e-6)
    assert only["Hat"] == pytest.approx(plain["Hat"], rel=1e-6)


def test_light_links_are_validated_saved_and_reported():
    scene, ids = linked_scene(exclude=["A"])
    key = scene.get_by_name("Key").light
    assert key.exclude == (ids["A"],)
    with pytest.raises(Exception, match="object ids"):
        key.include = "not-a-list"
    copy = scene_from_dict(scene_to_dict(scene))
    assert copy.get_by_name("Key").light.exclude == (ids["A"],)
    key.include = ["no-such-object"]
    r = scene.render(width=8, height=8, backend="cpu", default_lights=False)
    assert "LIGHT_LINK_MISSING" in {d.code for d in r.diagnostics}
    scene.history.undo()
    assert key.include == ()
