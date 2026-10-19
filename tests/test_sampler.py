"""The Owen-scrambled Sobol' sampler (0.20.0, ``path={"sampler": "sobol"}``): stratified
per pixel, unbiased, the same numbers on the GPU, less noise for the same samples."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d.errors import RenderError
from aether3d.rendering.gpu import gpu_status
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

GPU = gpu_status().get("available", False)


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


@pytest.mark.skipif(not GPU, reason="needs a GPU")
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


@pytest.mark.skipif(not GPU, reason="needs a GPU")
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
    for backend in ["path", *(["gpu_path"] if GPU else [])]:
        r = scene.render(width=11, height=3, backend=backend, default_lights=False, path=path)
        assert np.allclose(r.linear[1, 3:8, 0], expect, atol=2e-3), backend
    with pytest.raises(RenderError):
        PathSettings(filter="lanczos")
    with pytest.raises(RenderError):
        PathSettings(filter="gaussian", filter_width=20.0)
