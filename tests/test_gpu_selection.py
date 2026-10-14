"""Backend choice, adapter selection and GPU options -- tested without needing a GPU.

Adapter ranking and selection run on fabricated adapter descriptions; the
unavailable-GPU paths are simulated by hiding the wgpu package, so these tests
pass identically on machines with and without graphics hardware.
"""

import sys

import numpy as np
import pytest

from aether3d import RenderError, Scene, gpu_status, list_adapters
from aether3d.rendering import backends
from aether3d.rendering.backends import antialiasing_plan
from aether3d.rendering.gpu import device
from aether3d.rendering.gpu.device import AdapterInfo, AdapterQuery, parse_adapter_query, ranked


def fake(index, name, kind, backend, software=False):
    return (
        AdapterInfo(
            index=index,
            name=name,
            vendor="",
            vendor_id=0,
            device_id=index,
            type=kind,
            backend=backend,
            driver="test",
            software=software,
            features=(),
            limits=(),
        ),
        object(),
    )


# A hybrid laptop as wgpu reports it on Linux with both Vulkan drivers installed.
LAPTOP = [
    fake(0, "Intel(R) Graphics (ADL GT2)", "integrated", "vulkan"),
    fake(1, "NVIDIA GeForce RTX 4060 Laptop GPU", "discrete", "vulkan"),
    fake(2, "llvmpipe (LLVM 19.1.7, 256 bits)", "cpu", "vulkan", software=True),
    fake(3, "NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2", "unknown", "opengl"),
    fake(4, "Mesa Intel(R) Graphics (ADL GT2)", "integrated", "opengl"),
]


def names(spec):
    return [info.name for info, _ in ranked(LAPTOP, parse_adapter_query(spec))]


def test_auto_prefers_the_discrete_gpu_then_integrated_then_software(monkeypatch):
    monkeypatch.setattr(device.sys, "platform", "linux")
    assert names("auto") == [
        "NVIDIA GeForce RTX 4060 Laptop GPU",
        "Intel(R) Graphics (ADL GT2)",
        "Mesa Intel(R) Graphics (ADL GT2)",
        "NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2",
        "llvmpipe (LLVM 19.1.7, 256 bits)",
    ]
    assert device.recommended(LAPTOP).name == "NVIDIA GeForce RTX 4060 Laptop GPU"


def test_the_platforms_native_api_comes_first_for_one_device(monkeypatch):
    windows = [
        fake(0, "NVIDIA GeForce RTX 4060 Laptop GPU", "discrete", "vulkan"),
        fake(1, "NVIDIA GeForce RTX 4060 Laptop GPU", "discrete", "d3d12"),
        fake(2, "Microsoft Basic Render Driver", "cpu", "d3d12", software=True),
    ]
    monkeypatch.setattr(device.sys, "platform", "win32")
    order = [i.backend for i, _ in ranked(windows, AdapterQuery())]
    assert order == ["d3d12", "vulkan", "d3d12"]
    monkeypatch.setattr(device.sys, "platform", "linux")
    assert [i.backend for i, _ in ranked(windows, AdapterQuery())][:2] == ["vulkan", "d3d12"]


@pytest.mark.parametrize(
    "spec, expected",
    [
        (
            "nvidia",
            ["NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2"],
        ),
        (
            "RTX 4060",
            ["NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2"],
        ),
        ("integrated", ["Intel(R) Graphics (ADL GT2)", "Mesa Intel(R) Graphics (ADL GT2)"]),
        ("intel", ["Intel(R) Graphics (ADL GT2)", "Mesa Intel(R) Graphics (ADL GT2)"]),
        ("software", ["llvmpipe (LLVM 19.1.7, 256 bits)"]),
        ("cpu", ["llvmpipe (LLVM 19.1.7, 256 bits)"]),
        ("llvmpipe", ["llvmpipe (LLVM 19.1.7, 256 bits)"]),
        (
            "gl",
            ["Mesa Intel(R) Graphics (ADL GT2)", "NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2"],
        ),
        (4, ["Mesa Intel(R) Graphics (ADL GT2)"]),
        ("2", ["llvmpipe (LLVM 19.1.7, 256 bits)"]),
        ("name=nvidia,backend=vulkan", ["NVIDIA GeForce RTX 4060 Laptop GPU"]),
        ("type=integrated, backend=opengl", ["Mesa Intel(R) Graphics (ADL GT2)"]),
        ("hardware,vulkan", ["NVIDIA GeForce RTX 4060 Laptop GPU", "Intel(R) Graphics (ADL GT2)"]),
        ("amd", []),
        ("llvmpipe (LLVM 19.1.7, 256 bits)", ["llvmpipe (LLVM 19.1.7, 256 bits)"]),
    ],
)
def test_adapter_selection_by_index_kind_api_and_name(monkeypatch, spec, expected):
    monkeypatch.setattr(device.sys, "platform", "linux")
    assert names(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        True,
        1.5,
        -1,
        "index=x",
        "type=quantum",
        "backend=glide",
        "colour=red",
        "name=",
        "index=1,,type=cpu",
        "index=1,index=2",
        "software=maybe",
    ],
)
def test_invalid_adapter_selections_are_reported(spec):
    with pytest.raises(RenderError) as info:
        parse_adapter_query(spec)
    assert info.value.code == "INVALID_SETTINGS" and info.value.location == "adapter"


def test_the_environment_variable_sets_the_default_adapter(monkeypatch):
    monkeypatch.setenv("AETHER3D_GPU_ADAPTER", "llvmpipe")
    assert parse_adapter_query(None).name == "llvmpipe"
    assert parse_adapter_query("nvidia").name == "nvidia"  # explicit wins
    monkeypatch.delenv("AETHER3D_GPU_ADAPTER")
    assert parse_adapter_query(None).is_auto


@pytest.mark.parametrize(
    "samples, msaa, supersample, expected",
    [
        (1, None, None, (1, 1, 1)),
        (2, None, None, (4, 1, 4)),
        (3, None, None, (1, 3, 9)),
        (4, None, None, (4, 2, 16)),
        (2, 8, None, (8, 1, 8)),
        (2, None, 2, (1, 2, 4)),
        (3, 4, 3, (4, 3, 36)),
    ],
)
def test_gpu_antialiasing_keeps_samples_per_pixel(samples, msaa, supersample, expected):
    aa = antialiasing_plan(samples, msaa, supersample)
    assert (aa.msaa, aa.supersample, aa.samples_per_pixel) == expected


@pytest.mark.parametrize(
    "msaa, supersample", [(3, None), (32, None), (True, None), (None, 0), (None, 5), (None, 2.0)]
)
def test_invalid_antialiasing_options(msaa, supersample):
    with pytest.raises(RenderError) as info:
        antialiasing_plan(2, msaa, supersample)
    assert info.value.code == "INVALID_SETTINGS"


def simple_scene():
    scene = Scene("Fallback")
    scene.add_cube("Crate", size=2).color = "#c0392b"
    scene.add_plane("Ground", size=10, position=(0, 0, -1)).color = "#6a9a4a"
    scene.add_directional_light("Sun", direction=(0.4, 0.8, -1))
    scene.add_camera("Cam", position=(4, -6, 3), look_at=(0, 0, 0))
    return scene


def test_backend_options_are_validated_before_rendering():
    scene = simple_scene()
    for options, code, location in (
        ({"backend": "vulkan"}, "INVALID_SETTINGS", "backend"),
        ({"msaa": 4}, "INVALID_SETTINGS", "msaa"),
        ({"supersample": 2}, "INVALID_SETTINGS", "supersample"),
        ({"shadows": True}, "BACKEND_UNSUPPORTED", "shadows"),
        ({"backend": "auto", "shadows": "yes"}, "INVALID_SHADOWS", "shadows"),
        ({"backend": "auto", "shadows": {"blur": 1}}, "INVALID_SHADOWS", "shadows"),
        ({"backend": "auto", "shadows": {"map_size": 100}}, "INVALID_SHADOWS", "shadows.map_size"),
        ({"backend": "auto", "msaa": 6}, "INVALID_SETTINGS", "msaa"),
    ):
        with pytest.raises(RenderError) as info:
            scene.render(width=32, height=24, **options)
        assert (info.value.code, info.value.location) == (code, location), options


@pytest.fixture
def no_wgpu(monkeypatch):
    """Pretend the optional wgpu package is not installed."""
    monkeypatch.setattr(device, "_wgpu", None)
    monkeypatch.setitem(sys.modules, "wgpu", None)


def test_without_wgpu_the_cpu_renderer_still_works_and_gpu_is_explained(no_wgpu):
    scene = simple_scene()
    cpu = scene.render(width=48, height=32)
    assert cpu.backend == {"name": "cpu", "requested": "cpu"}
    with pytest.raises(RenderError) as info:
        scene.render(width=48, height=32, backend="gpu")
    assert info.value.code == "GPU_UNAVAILABLE" and "aether3d[gpu]" in info.value.hint
    status = gpu_status()
    assert status["available"] is False and "wgpu" in status["reason"]
    assert status["adapters"] == [] and status["selected"] is None
    with pytest.raises(RenderError):
        list_adapters()


def test_auto_falls_back_to_the_cpu_and_says_why(no_wgpu):
    scene = simple_scene()
    cpu = scene.render(width=48, height=32)
    auto = scene.render(width=48, height=32, backend="auto", msaa=4, shadows=True)
    assert auto.backend == {"name": "cpu", "requested": "auto"}
    assert np.array_equal(auto.image, cpu.image)  # the same CPU image, unchanged
    codes = {d.code: d for d in auto.diagnostics}
    assert "GPU_UNAVAILABLE" in codes["BACKEND_FALLBACK"].message
    assert codes["GPU_OPTIONS_IGNORED"].severity == "warning"
    assert codes["SHADOWS_NOT_RENDERED"].severity == "warning"
    assert auto.status == "warning"


def test_auto_only_falls_back_for_gpu_problems(monkeypatch):
    """Errors in the request itself are reported, never hidden by a fallback."""

    def broken(adapter=None):
        raise RenderError("INVALID_SETTINGS", "invalid adapter selection", "adapter")

    monkeypatch.setattr(device, "get_context", broken)
    with pytest.raises(RenderError) as info:
        simple_scene().render(width=16, height=16, backend="auto")
    assert info.value.code == "INVALID_SETTINGS"


def test_backend_names():
    assert backends.BACKENDS == ("cpu", "gpu", "auto")
    assert backends.check_backend("auto") == "auto"
