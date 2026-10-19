"""The path tracers' denoiser (0.20.0, ``render(denoise=True)``): measured against a
converged image, edges kept, the raw estimate kept, deterministic, and labelled biased."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.errors import RenderError
from aether3d.rendering.color import linear_to_srgb
from aether3d.rendering.denoise import DenoiseSettings, denoise
from aether3d.rendering.gpu import gpu_status

GPU = gpu_status().get("available", False)
BACKEND = "gpu_path" if GPU else "path"


def two_walls() -> Scene:
    scene = Scene()
    floor = scene.add_cube("Floor", size=1.0, position=(0, 0, -0.5))
    floor.scale = (6, 6, 1)
    floor.color = "#c8c0b0"
    red = scene.add_cube("Red", size=1.0, position=(-0.75, 1.0, 0.75))
    red.scale = (1.5, 0.2, 1.5)
    red.color = "#c03020"
    blue = scene.add_cube("Blue", size=1.0, position=(0.75, 1.0, 0.75))
    blue.scale = (1.5, 0.2, 1.5)
    blue.color = "#2040c0"
    scene.add_area_light("Key", position=(-1.5, -1.5, 2.5), look_at=(0, 0.5, 0.5), size=0.8,
                         intensity=12.0)  # fmt: skip
    scene.add_camera("Cam", position=(0, -3.5, 1.0), look_at=(0, 0.5, 0.6), fov=50)
    scene.world.environment.configure(type="image", image="builtin:overcast", intensity=0.4)
    return scene


def display(x):
    return linear_to_srgb(np.clip(x, 0.0, 1.0))


@pytest.fixture(scope="module")
def reference():
    scene = two_walls()
    samples = 2048 if GPU else 512
    r = scene.render(width=64, height=48, backend=BACKEND, path={"samples": samples,
                     "bounces": 4, "sampler": "sobol"})  # fmt: skip
    return scene, r.linear.astype(np.float64)


def test_denoising_brings_a_noisy_image_closer_to_the_converged_one(reference):
    # Measured (two scenes, 2-64 samples, hashed and Sobol' numbers): 20-35 % less
    # display-space error than the raw image -- what 1.5-2 times the samples would give.
    scene, ref = reference
    r = scene.render(width=64, height=48, backend=BACKEND, denoise=True,
                     path={"samples": 16, "bounces": 4, "sampler": "sobol", "seed": 7})  # fmt: skip
    raw = np.sqrt(np.mean((display(r.passes["beauty"]) - display(ref)) ** 2))
    den = np.sqrt(np.mean((display(r.linear) - display(ref)) ** 2))
    assert den < 0.88 * raw  # (measured: 0.78-0.82 here)
    assert r.backend["denoiser"]["biased"] and "DENOISED" in {d.code for d in r.diagnostics}
    assert "denoise" in r.stats["timings"]


def test_edges_between_objects_do_not_bleed(reference):
    scene, ref = reference
    r = scene.render(width=64, height=48, backend=BACKEND, denoise=True,
                     path={"samples": 8, "bounces": 4, "sampler": "sobol", "seed": 2})  # fmt: skip
    index = {o["name"]: k for k, o in enumerate(r.objects)}
    red = r.object_index == index["Red"]
    blue = r.object_index == index["Blue"]
    # The red wall stays red and the blue one blue (no colour carried across their seam).
    assert (r.linear[red, 0] > r.linear[red, 2]).mean() > 0.97
    assert (r.linear[blue, 2] > r.linear[blue, 0]).mean() > 0.97


def test_denoising_is_deterministic_and_checked():
    scene = two_walls()
    opts = {"width": 32, "height": 24, "backend": "path", "denoise": {"iterations": 2},
            "path": {"samples": 4, "bounces": 1}}  # fmt: skip
    a, b = scene.render(**opts), scene.render(**opts)
    assert np.array_equal(a.linear, b.linear)
    assert not np.array_equal(a.linear, a.passes["beauty"])
    with pytest.raises(RenderError) as info:
        scene.render(width=8, height=8, backend="cpu", denoise=True)
    assert info.value.code == "INVALID_DENOISE"
    for bad in ({"iterations": 0}, {"strength": 2}, "yes"):
        with pytest.raises(RenderError):
            DenoiseSettings.coerce(bad)


def test_a_flat_noise_free_image_is_left_alone():
    h, w = 16, 16
    passes = {
        "beauty": np.full((h, w, 3), 0.4, np.float32),
        "albedo": np.full((h, w, 3), 0.8, np.float32),
        "normal": np.tile(np.array([0, 0, 1], np.float32), (h, w, 1)),
        "depth": np.full((h, w), 3.0, np.float32),
        "variance": np.zeros((h, w, 3), np.float32),
    }
    assert np.allclose(denoise(passes), 0.4, atol=1e-6)
