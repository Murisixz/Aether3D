"""The atmosphere (sky, sun glow, height fog, hemisphere ambient) and cascaded shadows:
settings, the fog's exact optical depth against numerical integration, the renderers
against each other, history and files."""

import math

import numpy as np
import pytest

from aether3d import RenderError, Scene, create_plane
from aether3d.rendering import atmosphere as atmo
from aether3d.rendering.scene_data import extract_render_scene, resolve_camera
from aether3d.rendering.settings import ShadowSettings
from aether3d.rendering.shadows import cascade_splits


def outdoor(fog=0.02, falloff=0.15, ambient="hemisphere", sun=(0.3, 0.6, -0.5)):
    scene = Scene("Outdoor")
    ground = scene.add_object("Ground", mesh=create_plane(400.0))
    ground.appearance.configure(color="#7a8a5a")
    for k, y in enumerate((5.0, 25.0, 60.0, 120.0)):
        scene.add_cube(f"Box{k}", size=3.0, position=(0.0, y, 1.5))
    scene.add_camera("Cam", position=(0.0, -12.0, 3.0), look_at=(0.0, 40.0, 1.0), fov=60, far=1000)
    scene.add_directional_light("Sun", direction=sun, intensity=1.5)
    scene.world.environment.configure(type="sky")
    scene.world.atmosphere.configure(
        enabled=True,
        sun="Sun",
        fog_density=fog,
        fog_height=0.0,
        fog_falloff=falloff,
        ambient=ambient,
        horizon_glow=0.4,
        fog_sun=0.5,
    )
    return scene


def model_of(scene):
    rs = extract_render_scene(scene, resolve_camera(scene, None), aspect=16 / 9)
    return atmo.resolve(rs)


def test_settings_validate_record_history_and_save(tmp_path):
    scene = outdoor()
    a = scene.world.atmosphere
    assert a.enabled and a.fog_density == 0.02 and a.ambient == "hemisphere"
    with pytest.raises(RenderError):
        a.configure(fog_density=-1.0)
    with pytest.raises(RenderError):
        a.configure(ambient="upside-down")
    with pytest.raises(RenderError):
        a.configure(fog_color="#nothex")
    before = a.to_dict()
    a.configure(fog_density=0.05)
    scene.history.undo()
    assert scene.world.atmosphere.to_dict() == before
    path = tmp_path / "a.aether"
    scene.save(path)
    assert Scene.load(path).world.atmosphere.to_dict() == before


def test_fog_transmittance_is_the_exact_optical_depth():
    model = model_of(outdoor(fog=0.03, falloff=0.2))
    rng = np.random.default_rng(3)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = rng.uniform(0.0, 80.0, 64)
    got = atmo.transmittance(model, d, t)
    # Independent reference: midpoint-rule integration of the density along each ray.
    n = 20000
    s = (np.arange(n) + 0.5) / n
    z = model.fog_rel + d[:, 2:3] * (t[:, None] * s[None, :])
    tau = (model.fog_density * np.exp(-model.fog_falloff * z)).sum(axis=1) * t / n
    np.testing.assert_allclose(got, np.exp(-tau), rtol=1e-6, atol=1e-9)
    # Looking up into an infinite sky: the closed form; down: no light gets through.
    up = atmo.transmittance(model, np.array([[0.0, 0.6, 0.8]]), np.array([np.inf]))
    rho = model.fog_density * math.exp(-model.fog_falloff * model.fog_rel)
    assert up[0] == pytest.approx(math.exp(-rho / (model.fog_falloff * 0.8)))
    down = atmo.transmittance(model, np.array([[0.0, 0.6, -0.8]]), np.array([np.inf]))
    assert down[0] < 1e-300


def test_farther_ground_fades_into_the_fog():
    scene = outdoor(fog=0.03, falloff=0.0)
    img = scene.render(width=160, height=90).image.astype(float)
    fogless = outdoor(fog=0.0).render(width=160, height=90).image.astype(float)
    rs = extract_render_scene(scene, resolve_camera(scene, None), aspect=16 / 9)
    _, v_near = _project(rs, np.array([0.0, -6.0, 0.0]), 160, 90)
    _, v_far = _project(rs, np.array([0.0, 150.0, 0.0]), 160, 90)
    near = np.abs(img[v_near] - fogless[v_near]).sum(axis=1).mean()
    far = np.abs(img[v_far] - fogless[v_far]).sum(axis=1).mean()
    assert far > near + 30  # farther ground: more fog
    fog_colour = 255 * np.array(scene.world.atmosphere.fog_color)  # (a display colour)
    with_fog = np.abs(img[v_far].mean(axis=0) - fog_colour).sum()
    assert with_fog < np.abs(fogless[v_far].mean(axis=0) - fog_colour).sum()


def _project(rs, p, w, h):
    f = rs.camera.frame
    rel = p - f.position
    x, y, z = rel @ f.right, rel @ f.up, rel @ f.forward
    hx, hy = rs.camera.camera.half_extents(w / h)
    u = int((x / z / hx * 0.5 + 0.5) * w)
    v = int((0.5 - y / z / hy * 0.5) * h)
    return min(max(u, 0), w - 1), min(max(v, 0), h - 1)


def test_hemisphere_ambient_lights_from_above():
    model = model_of(outdoor())
    amb = np.array([0.3, 0.3, 0.3])
    a = atmo.ambient_at(model, amb, np.array([1.0, 0.0, -1.0]))
    np.testing.assert_allclose(a[0], amb)
    np.testing.assert_allclose(a[2], model.ground_ambient)
    np.testing.assert_allclose(a[1], (amb + model.ground_ambient) / 2)


def test_the_sky_glows_toward_the_sun():
    model = model_of(outdoor())
    sun = model.sun
    away = np.array([-sun[0], -sun[1], abs(sun[2])])
    away /= np.linalg.norm(away)
    near, far = atmo.sky(model, np.stack([sun, away]))
    assert near.sum() > far.sum()


def test_the_gpu_atmosphere_matches_the_cpu(gpu):
    scene = outdoor()
    cpu = scene.render(width=160, height=90, samples=2)
    on_gpu = gpu.render(scene, width=160, height=90, samples=2, supersample=2, msaa=1)
    diff = np.abs(cpu.image.astype(int) - on_gpu.image.astype(int))
    assert diff.max() <= 2  # 8-bit codes: float32 on the GPU, float64 on the CPU


def test_the_gpu_path_tracer_matches_the_cpu_one_with_fog(gpu):
    """The same paths (the same random numbers): only float32 margins differ."""
    scene = outdoor(fog=0.01)
    opts = {"width": 40, "height": 24, "path": {"samples": 16, "bounces": 1}}
    cpu = scene.render(backend="path", **opts).image.astype(int)
    on_gpu = scene.render(backend="gpu_path", adapter=gpu.adapter, **opts).image.astype(int)
    diff = np.abs(cpu - on_gpu)
    assert np.mean(diff) < 0.5 and np.percentile(diff, 99) <= 3


def test_cascade_settings_and_splits():
    s = ShadowSettings(cascades=3, cascade_split=0.75, distance=500.0)
    assert s.cascades == 3
    with pytest.raises(RenderError):
        ShadowSettings(cascades=5)
    with pytest.raises(RenderError):
        ShadowSettings(cascade_split=1.5)
    splits = cascade_splits(0.1, 500.0, 3, 0.75)  # the boundaries, near and far included
    assert len(splits) == 4 and splits[0] == 0.1 and splits[-1] == pytest.approx(500.0)
    assert all(a < b for a, b in zip(splits[:-1], splits[1:], strict=True))
    equal = cascade_splits(0.1, 500.0, 4, 0.0)
    np.testing.assert_allclose(np.diff(equal), np.diff(equal)[0], rtol=1e-9)
    logarithmic = cascade_splits(0.1, 500.0, 4, 1.0)
    np.testing.assert_allclose(
        np.diff(np.log(logarithmic)), np.diff(np.log(logarithmic))[0], rtol=1e-9
    )


def test_cascaded_shadows_render_and_are_reported(gpu):
    scene = outdoor(fog=0.0, sun=(1.0, 0.25, -0.45))  # shadows fall sideways, in view
    r = gpu.render(scene, width=160, height=90, shadows={"cascades": 3, "distance": 300.0})
    maps = r.backend["shadows"]["maps"]
    assert [m["cascade"] for m in maps] == [0, 1, 2]
    assert maps[0]["texel_size"] < maps[1]["texel_size"] < maps[2]["texel_size"]
    lit = gpu.render(scene, width=160, height=90, shadows=False).image.astype(int)
    # The boxes cast shadows on the ground (darker pixels than without shadows).
    assert np.sum(lit.sum(axis=2) - r.image.astype(int).sum(axis=2) > 30) > 50
