"""GPU fault handling: limits, unsupported requests, shader errors, lost devices, validation.

Every failure must surface as a ``RenderError`` with a stable ``GPU_*`` (or
``RESOURCE_LIMIT``) code and an actionable message, before anything large is
allocated, without changing the scene or writing a file -- and the next render
must work again.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_render_scene import hostile_scene  # noqa: E402

from aether3d import RenderError, RenderLimits, Scene, release_gpu  # noqa: E402
from aether3d.rendering.gpu import device as gpu_device  # noqa: E402


def small_scene():
    scene = Scene()
    scene.add_cube("Crate").color = "#c0392b"
    scene.add_camera("Cam", position=(3, -4, 2), look_at=(0, 0, 0))
    return scene


def test_unknown_adapters_are_reported_with_the_alternatives(gpu):
    with pytest.raises(RenderError) as info:
        small_scene().render(backend="gpu", adapter="no such adapter 3dfx", width=8, height=8)
    error = info.value
    assert error.code == "GPU_ADAPTER_NOT_FOUND" and error.location == "adapter"
    assert gpu.name in error.message and "gpu info" in error.hint
    assert any(a["name"] == gpu.name for a in error.details["adapters"])


def test_unsupported_msaa_counts_are_refused_not_invented(gpu):
    for count in (2, 8, 16):
        if count in gpu.msaa:
            assert gpu.render(small_scene(), width=16, height=16, msaa=count)
            continue
        with pytest.raises(RenderError) as info:
            gpu.render(small_scene(), width=16, height=16, msaa=count)
        assert info.value.code == "GPU_FEATURE_UNSUPPORTED" and info.value.location == "msaa"
        assert info.value.details["supported_msaa"] == list(gpu.msaa)


def test_device_limits_and_memory_limits_are_checked_before_allocating(gpu):
    largest = gpu.info["limits"]["max-texture-dimension-2d"]
    side = min(16384, largest)
    factor = largest // side + 1
    if factor <= 4:
        with pytest.raises(RenderError) as info:
            gpu.render(small_scene(), width=side, height=16, supersample=factor)
        assert info.value.code == "GPU_LIMIT_EXCEEDED" and str(largest) in info.value.message
    tight = RenderLimits(max_gpu_memory=1_000_000)
    with pytest.raises(RenderError) as info:
        gpu.render(small_scene(), width=640, height=480, limits=tight)
    assert info.value.code == "RESOURCE_LIMIT" and "GPU memory" in info.value.message
    with pytest.raises(RenderError) as info:
        gpu.render(
            small_scene(), width=640, height=480, msaa=4, limits=RenderLimits(max_samples=10)
        )
    assert info.value.code == "RESOURCE_LIMIT"
    with pytest.raises(RenderError) as info:
        gpu.render(hostile_scene(), width=32, height=32, limits=RenderLimits(max_triangles=10))
    assert info.value.code == "RESOURCE_LIMIT"


def test_shader_compilation_errors_are_structured(gpu):
    context, _ = gpu_device.get_context(gpu.adapter)
    with pytest.raises(RenderError) as info:
        context.shaders.compile("@vertex fn broken( -> vec4<f32> {}", "broken-test")
    error = info.value
    assert error.code == "GPU_SHADER_ERROR" and error.location == "shader[broken-test]"
    assert "broken-test" in error.message and error.details["compiler"]
    assert gpu.render(small_scene(), width=8, height=8)  # the device is still fine


def test_a_lost_device_is_reported_and_the_next_render_recovers(gpu):
    scene = small_scene()
    before = gpu.render(scene, width=32, height=24)
    context, _ = gpu_device.get_context(gpu.adapter)
    context.device.destroy()  # what a driver reset or GPU removal looks like
    with pytest.raises(RenderError) as info:
        gpu.render(scene, width=32, height=24)
    assert info.value.code == "GPU_DEVICE_LOST" and "render again" in info.value.hint
    after = gpu.render(scene, width=32, height=24)
    assert after.backend["device_created"] is True
    assert np.array_equal(after.image, before.image)


def test_release_frees_devices_and_rendering_creates_them_again(gpu):
    gpu.render(small_scene(), width=8, height=8)
    assert release_gpu() >= 1
    again = gpu.render(small_scene(), width=8, height=8)
    assert again.backend["device_created"] is True


def test_rendering_under_wgpu_validation_reports_no_problems(gpu, tmp_path):
    """A fresh process with AETHER3D_GPU_DEBUG=1 (wgpu and backend validation layers on)."""
    script = tmp_path / "validate.py"
    script.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_render_scene import hostile_scene\n"
        "from aether3d import gpu_status\n"
        "scene = hostile_scene()\n"
        "codes = []\n"
        "for options in ({}, {'msaa': 1, 'supersample': 2}, {'shadows': True}, {'cull': 'back'}):\n"
        f"    r = scene.render(width=96, height=64, backend='gpu', adapter={gpu.adapter!r},"
        " **options)\n"
        "    codes += [d.to_dict() for d in r.diagnostics if d.code == 'GPU_VALIDATION_MESSAGE']\n"
        "print(json.dumps({'notes': gpu_status()['notes'], 'messages': codes}))\n"
    )
    env = {**os.environ, "AETHER3D_GPU_DEBUG": "1"}
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300
    )
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert any("validation enabled" in note for note in report["notes"])
    assert report["messages"] == []
