"""Performance on graphics hardware: the GPU must be much faster than the CPU renderer.

Skipped on software adapters (Mesa llvmpipe, WARP, SwiftShader): they run the
GPU pipeline on the CPU, which validates correctness but says nothing about
hardware speed. Skipped on CI too (``CI`` is set), whose shared runners offer
only software or virtualized GPUs, unless ``AETHER3D_PERFORMANCE_TESTS=1``.
The threshold is deliberately loose (the RTX 4060 Laptop GPU measures about
35x on this scene, see docs/GPU.md), so a busy machine does not fail the test.
"""

import os
import time

import pytest

from aether3d import Scene, create_sphere


def test_the_gpu_is_much_faster_than_the_cpu_on_a_large_scene(gpu):
    if os.environ.get("CI") and os.environ.get("AETHER3D_PERFORMANCE_TESTS") != "1":
        pytest.skip("performance tests need dedicated GPU hardware, not a shared CI runner")
    if gpu.info["adapter"]["software"]:
        pytest.skip(f"{gpu.name} is a software adapter: no hardware performance to measure")
    scene = Scene("Heavy")
    scene.add_plane("Ground", size=20).color = "#557755"
    scene.add_object("Ball", mesh=create_sphere(1.0, 320, 160), position=(0, 0, 1))
    for i in range(400):
        scene.add_cube(f"Cube {i}", size=0.3, position=(i % 20 - 10, i // 20 - 10, 0.15))
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1))
    scene.add_camera("Cam", fov=40).frame(direction=(-0.5, 1, -0.6), aspect=16 / 9)
    options = {"width": 1280, "height": 720, "samples": 2}
    gpu.render(scene, **options)  # warm up: device, shaders, mesh uploads
    t0 = time.perf_counter()
    cpu_result = scene.render(**options)
    cpu = time.perf_counter() - t0
    best = min(_timed(lambda: gpu.render(scene, **options)) for _ in range(3))
    assert cpu_result.stats["triangles"] > 100_000
    assert best < cpu / 3, f"GPU {best:.3f} s vs CPU {cpu:.3f} s"


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
