"""The Owen-scrambled Sobol' sampler (0.20.0, ``path={"sampler": "sobol"}``): stratified
per pixel, unbiased, the same numbers on the GPU, less noise for the same samples."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from gpu_support import GPU_PATH

from aether3d.errors import RenderError
from aether3d.rendering.pathtrace import integrator
from aether3d.rendering.pathtrace.sampling import (
    SLOT_BSDF,
    SLOT_LIGHTS,
    SLOT_PIXEL,
    SLOT_RR,
    dimension,
    sampler_keys,
    sobol,
    uniform,
)
from aether3d.rendering.settings import PathSettings

sys.path.insert(0, str(Path(__file__).parent))
from test_lights import DATA, area_scene  # noqa: E402


def test_the_sequence_is_sobols_and_stays_stratified_after_scrambling():
    idx = np.arange(16, dtype=np.uint32)
    x, y = sobol(idx, 0) / 2**32, sobol(idx, 1) / 2**32
    assert list(y[:4]) == [0.0, 0.5, 0.75, 0.25]  # Sobol's second dimension
    assert (
        len(set(zip((x * 4).astype(int), (y * 4).astype(int), strict=True))) == 16
    )  # a (0, 4, 2)-net
    for dims in ((dimension(0, SLOT_PIXEL), dimension(0, SLOT_PIXEL + 1)),
                 (dimension(2, SLOT_BSDF), dimension(2, SLOT_BSDF + 1)),
                 (dimension(1, SLOT_LIGHTS + 4), dimension(1, SLOT_LIGHTS + 5))):  # fmt: skip
        keys = sampler_keys("sobol", 9, np.full(64, 1234), np.arange(64))
        u, v = uniform(keys, dims[0]), uniform(keys, dims[1])
        assert len(set(zip((u * 8).astype(int), (v * 8).astype(int), strict=True))) == 64


def test_pixels_and_decisions_are_decorrelated_and_uniform():
    keys = sampler_keys("sobol", 3, np.arange(200_000), np.zeros(200_000, np.int64))
    a = uniform(keys, dimension(2, SLOT_BSDF))
    b = uniform(keys, dimension(3, SLOT_BSDF))
    c = uniform(keys, dimension(2, SLOT_RR))
    for u in (a, b, c):
        assert abs(u.mean() - 0.5) < 0.003 and abs(u.var() - 1 / 12) < 0.001
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.01 and abs(np.corrcoef(a, c)[0, 1]) < 0.01


@pytest.mark.skipif(not GPU_PATH, reason="needs a GPU that runs the path tracer")
def test_the_gpu_draws_the_cpus_sobol_numbers():
    from aether3d.rendering.gpu import device
    from aether3d.rendering.gpu.pathtrace import random_numbers

    context, _ = device.get_context(None, path_tracer=True)
    w, h, sample = 37, 23, 5
    pixels = np.arange(w * h)
    dims = [dimension(0, SLOT_PIXEL), dimension(0, SLOT_PIXEL + 1),
            dimension(3, SLOT_LIGHTS + 5), dimension(2, SLOT_RR)]  # fmt: skip
    for seed in (0, 11, 2**40 + 12345):
        values = random_numbers(context, seed, w, h, sample, "sobol")
        keys = sampler_keys("sobol", seed, pixels, np.full(pixels.size, sample))
        for k, dim in enumerate(dims):
            expected = np.floor(uniform(keys, dim) * 2**24) / 2**24
            assert np.array_equal(values[k].reshape(-1).astype(np.float64), expected)


def rms_error(sampler: str, samples: int, seed: int, backend: str = "path") -> float:
    ref = np.array(DATA["images"]["area_SQUARE_P100_h1_spread180"])
    scene = area_scene("square", 0.5, 0.5)
    path = {"samples": samples, "bounces": 0, "sampler": sampler, "seed": seed}
    r = scene.render(width=33, height=33, backend=backend, default_lights=False, path=path)
    return float(np.sqrt(np.mean(((r.linear[..., 0] - ref) / ref) ** 2)))


def test_sobol_has_less_noise_for_the_same_samples_and_no_bias():
    # An area-lit plane, 16 samples: RMS error 6.4 % with hashed numbers, 0.74 % with
    # Sobol' points (measured over seeds) -- as many as ~75 times the samples would give.
    random = np.mean([rms_error("random", 16, s) for s in range(2)])
    sobol_ = np.mean([rms_error("sobol", 16, s) for s in range(2)])
    assert sobol_ < random / 5
    assert rms_error("sobol", 256, 3) < 0.003  # converges to Blender's image (no bias)


@pytest.mark.skipif(not GPU_PATH, reason="needs a GPU that runs the path tracer")
def test_the_gpu_traces_the_same_sobol_paths():
    a = rms_error("sobol", 16, 1, "path")
    b = rms_error("sobol", 16, 1, "gpu_path")
    assert b == pytest.approx(a, rel=0.02)


def test_sobol_images_do_not_depend_on_batching(monkeypatch):
    scene = area_scene("disk", 0.5, 0.5)
    opts = {"width": 12, "height": 12, "backend": "path", "default_lights": False,
            "path": {"samples": 6, "bounces": 2, "sampler": "sobol"}}  # fmt: skip
    a = scene.render(**opts)
    monkeypatch.setattr(integrator, "BATCH", 50)
    b = scene.render(**opts)
    assert np.array_equal(a.linear, b.linear)


def test_samplers_are_checked():
    assert PathSettings(sampler="sobol").sampler == "sobol"
    with pytest.raises(RenderError, match="sampler"):
        PathSettings(sampler="halton")


@pytest.mark.parametrize("name", ["gaussian", "blackman_harris"])
def test_pixel_filters_spread_an_edge_as_their_profiles_say(name):
    from aether3d import Scene
    from aether3d.rendering.pathtrace.sampling import filter_offset, filter_table

    scene = Scene()
    wall = scene.add_cube("Wall", size=1, position=(0.5, 0, 0))
    wall.scale = (1, 1, 10)
    wall.material = scene.create_material("Lamp", model="standard", base_color="#000000",
                                          emissive="#ffffff")  # fmt: skip
    scene.add_camera("Cam", projection="orthographic", ortho_scale=0.1, position=(0, -5, 0),
                     look_at=(0, 0, 0))  # fmt: skip
    scene.world.environment.configure(type="uniform", color="#000000")
    path = {"samples": 2048, "bounces": 0, "filter": name, "filter_width": 3.0,
            "sampler": "sobol"}  # fmt: skip
    u = (np.arange(100_000) + 0.5) / 100_000
    offsets = filter_offset(filter_table(name, 3.0), u)
    expect = [np.mean(offsets > 5 - k) for k in range(3, 8)]  # the edge: pixel 5's centre
    for backend in ["path", *(["gpu_path"] if GPU_PATH else [])]:
        r = scene.render(width=11, height=3, backend=backend, default_lights=False, path=path)
        assert np.allclose(r.linear[1, 3:8, 0], expect, atol=2e-3), backend
    with pytest.raises(RenderError):
        PathSettings(filter="lanczos")
    with pytest.raises(RenderError):
        PathSettings(filter="gaussian", filter_width=20.0)


def adaptive_scene():
    from test_denoise import two_walls

    return two_walls()


@pytest.mark.parametrize("backend", ["path", *(["gpu_path"] if GPU_PATH else [])])
def test_adaptive_sampling_spends_samples_where_the_noise_is(backend):
    scene = adaptive_scene()
    path = {"samples": 128, "bounces": 3, "sampler": "sobol", "adaptive_threshold": 0.03}
    r = scene.render(width=48, height=36, backend=backend, path=path)
    counts = r.passes["sample_count"]
    assert counts.min() >= 8 and counts.max() == 128  # rounds: 8, 16, 32, 64, 128
    assert counts.mean() < 0.6 * 128 and r.stats["paths"] == int(counts.sum())
    sky = r.object_index < 0
    assert counts[sky].mean() < counts[~sky].mean()  # the flat sky converges first
    # Pixels that stopped early meet the threshold (their estimated relative error).
    lum = np.maximum(r.passes["beauty"] @ [0.2126, 0.7152, 0.0722], 0.05)
    err = np.sqrt(r.passes["variance"] @ np.array([0.2126, 0.7152, 0.0722]) ** 2) / lum
    early = counts < 128
    assert np.percentile(err[early], 95) < 0.03


def test_adaptive_images_do_not_depend_on_batching_or_workers(monkeypatch):
    scene = adaptive_scene()
    opts = {"width": 24, "height": 18, "backend": "path",
            "path": {"samples": 32, "bounces": 2, "adaptive_threshold": 0.05}}  # fmt: skip
    a = scene.render(**opts)
    monkeypatch.setattr(integrator, "BATCH", 37)
    b = scene.render(**{**opts, "path": {**opts["path"], "workers": 3}})
    assert np.array_equal(a.linear, b.linear)
    assert np.array_equal(a.passes["sample_count"], b.passes["sample_count"])


@pytest.mark.skipif(not GPU_PATH, reason="needs a GPU that runs the path tracer")
def test_the_gpu_stops_the_pixels_the_cpu_stops():
    scene = adaptive_scene()
    path = {"samples": 64, "bounces": 2, "sampler": "sobol", "adaptive_threshold": 0.03}
    rc = scene.render(width=40, height=30, backend="path", path=path)
    rg = scene.render(width=40, height=30, backend="gpu_path", path=path)
    c, g = rc.passes["sample_count"], rg.passes["sample_count"]
    assert np.mean(c == g) > 0.99  # (float32 sums may flip a pixel at the threshold)
    # Both report what they spent the same way.
    for r, counts in ((rc, c), (rg, g)):
        assert r.stats["samples_mean"] == pytest.approx(counts.mean())
        assert r.stats["pixels_converged_early"] == int((counts < 64).sum())


def test_every_new_feature_together_does_not_depend_on_batching_or_workers(monkeypatch):
    # Spot and area lights (spread), glass, an image environment, depth of field, Sobol',
    # a Blackman-Harris filter and adaptive sampling at once: the same bits for any batch
    # size and number of processes.
    from aether3d import Scene

    def scene():
        s = Scene()
        s.add_plane("Floor", size=6)
        s.add_sphere("Ball", radius=0.5, position=(0, 0, 0.5))
        glass = s.add_sphere("Glass", radius=0.3, position=(0.7, -0.4, 0.3))
        glass.material = s.create_material("Glass", preset="glass")
        s.add_spot_light("Spot", position=(1.5, -1.5, 2.5), look_at=(0, 0, 0), spot_size=50,
                         radius=0.1, intensity=30)  # fmt: skip
        s.add_area_light("Area", position=(-1.5, -1, 2), look_at=(0, 0, 0.3), shape="ellipse",
                         size=0.8, size_y=0.4, spread=60, intensity=8)  # fmt: skip
        s.world.environment.configure(type="image", image="builtin:studio", intensity=0.5)
        cam = s.add_camera("Cam", position=(0, -4, 1.5), look_at=(0, 0, 0.4))
        cam.camera.configure(fstop=2.8, focus_distance=4.0)
        return s

    path = {"samples": 8, "bounces": 4, "sampler": "sobol", "filter": "blackman_harris",
            "adaptive_threshold": 0.05, "workers": 1}  # fmt: skip
    opts = {"width": 40, "height": 30, "backend": "path", "path": path}
    a = scene().render(**opts)
    monkeypatch.setattr(integrator, "BATCH", 37)
    b = scene().render(**{**opts, "path": {**path, "workers": 3}})
    assert np.array_equal(a.linear, b.linear)
    assert np.array_equal(a.passes["sample_count"], b.passes["sample_count"])
