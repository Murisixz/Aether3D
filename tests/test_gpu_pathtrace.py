"""The GPU path tracer (``backend="gpu_path"``) against the CPU path tracer and references.

The kernel draws exactly the CPU path tracer's random numbers (the first test checks
every bit a float32 keeps), so with the same scene and settings both backends trace the
same paths. Their images then differ only by float32 rounding (about 1e-7 per operation,
so ~1e-5 along a path) -- except where rounding flips a discrete decision: a ray grazing
an edge, a Russian-roulette draw within 2^-24 of its threshold, a texel at the alpha
cutoff. That changes one path, and its pixel by up to a few tens of percent. The
comparisons therefore bound, over the beauty values of each image:

* the median relative difference (rounding), at float32 level;
* the share of values that differ by more than 1% (flipped paths);
* the image means (flips are as likely either way).

Procedural textures are evaluated in float32 on the GPU (float64 on the CPU): their values
differ in the last digits everywhere and their hard edges flip more paths, so the castle
gets wider bounds. Analytic references (a white furnace, the inverse-square law) are
checked on the GPU directly.

GPU tests take the ``gpu`` fixture (see conftest.py). Option checks that fail before a GPU
is needed run everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import aether3d as a3
from aether3d.errors import RenderError
from aether3d.rendering.pathtrace.sampling import (
    SLOT_LIGHTS,
    SLOT_PIXEL,
    SLOT_RR,
    dimension,
    path_keys,
    uniform,
)

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXAMPLES))

from test_pathtrace import cornell_box, dark, lambert  # noqa: E402

HAS_MANIFOLD = importlib.util.find_spec("manifold3d") is not None


@pytest.fixture
def gpu(gpu):
    """The GPU of the tests (conftest.py), if it can run the GPU path tracer.

    Microsoft WARP, and Direct3D 12 without DXC, refuse the GPU path tracer (see
    ``device.pathtrace_support``): these tests are skipped there, with the reason, and
    ``test_adapters_that_cannot_run_the_path_tracer_refuse_it`` checks the refusal.
    """
    from aether3d.rendering.gpu import device

    raster, _ = device.get_context(gpu.adapter)
    why = device.pathtrace_support(raster.info)
    if why is not None:
        pytest.skip(f"{raster.info.label} cannot run the GPU path tracer: {why}")
    return gpu


def gpu_path(gpu, scene, width, height, *, ao=None, **path):
    path.setdefault("samples", 4)
    return scene.render(
        width=width,
        height=height,
        backend="gpu_path",
        adapter=gpu.adapter,
        path=path,
        ao=ao,
        default_lights=False,
    )


def cpu_path(scene, width, height, *, ao=None, **path):
    path.setdefault("samples", 4)
    return scene.render(
        width=width,
        height=height,
        backend="path",
        path={**path, "workers": 1},
        ao=ao,
        default_lights=False,
    )


def assert_same_paths(g, c, *, median=1e-4, flipped=0.03, mean=2e-3, centers=0.99):
    """``g`` traced the same paths as ``c``, to float32 rounding (see the module docstring)."""
    gi = g.passes["beauty"].astype(np.float64)
    ci = c.passes["beauty"].astype(np.float64)
    assert gi.shape == ci.shape
    rel = np.abs(gi - ci) / np.maximum(np.abs(ci), 1e-3)
    assert np.median(rel) < median, f"median relative difference {np.median(rel):.2e}"
    assert (rel > 0.01).mean() < flipped, f"{100 * (rel > 0.01).mean():.1f}% of values differ"
    ratio = gi.mean() / max(ci.mean(), 1e-12)
    assert abs(ratio - 1.0) < mean, f"mean ratio {ratio:.6f}"
    # The same surfaces at the pixel centers (rays exactly on seams may resolve differently).
    assert (g.object_index == c.object_index).mean() > centers


# --------------------------------------------------------------------------- #
# Random numbers and the same paths as the CPU
# --------------------------------------------------------------------------- #


def test_the_gpu_draws_the_cpu_path_tracers_random_numbers(gpu):
    from aether3d.rendering.gpu import device
    from aether3d.rendering.gpu.pathtrace import random_numbers

    context, _ = device.get_context(gpu.adapter, path_tracer=True)
    width, height, sample = 37, 23, 5
    pixels = np.arange(width * height)
    dims = [
        dimension(0, SLOT_PIXEL),
        dimension(0, SLOT_PIXEL + 1),
        dimension(3, SLOT_LIGHTS + 5),
        dimension(2, SLOT_RR),
    ]
    for seed in (0, 11, 2**40 + 12345):
        values = random_numbers(context, seed, width, height, sample)
        keys = path_keys(seed, pixels, np.full(pixels.size, sample))
        for k, dim in enumerate(dims):
            expected = np.floor(uniform(keys, dim) * 2**24) / 2**24  # the 24 bits of a float32
            assert np.array_equal(values[k].reshape(-1).astype(np.float64), expected)


def test_a_cornell_box_traces_the_cpu_path_tracers_paths(gpu):
    scene = cornell_box()
    g = gpu_path(gpu, scene, 24, 24, samples=16, bounces=3, seed=11)
    c = cpu_path(scene, 24, 24, samples=16, bounces=3, seed=11)
    assert g.stats["camera_rays"] == c.stats["camera_rays"]
    assert g.stats["bounce_rays"] == c.stats["bounce_rays"]  # every path took the same turns
    assert abs(g.stats["shadow_rays"] - c.stats["shadow_rays"]) <= 3
    # The box is symmetric: the centers of pixels on the image's diagonals lie exactly on the
    # seams between walls, where float32 and float64 rounding pick different walls.
    assert_same_paths(g, c, median=2e-5, flipped=0.01, mean=1e-4, centers=0.97)
    for name in ("direct", "indirect", "albedo", "normal"):
        np.testing.assert_allclose(g.passes[name], c.passes[name], rtol=0.02, atol=1e-3)


def metal_spheres():
    from metal_spheres import build_scene

    return build_scene()


def window_room():
    from window_room import build_scene

    return build_scene()


def ao_scene():
    from ao_contact import build_scene

    return build_scene()


def mirror_with_sized_lights():
    scene = dark(a3.Scene())
    mirror = scene.add_plane("Mirror", size=40)
    mirror.material = scene.create_material(
        "Chrome", base_color="#ffffff", metallic=1.0, roughness=0.0
    )
    scene.add_directional_light("Sun", direction=(0, 0.5, -1), intensity=1.0, angle=10.0)
    scene.add_point_light("Bulb", position=(0.3, -0.8, 1.6), intensity=2.0, radius=0.3)
    scene.add_camera("Cam", position=(0, 2.2, 4.4), look_at=(0, 0, 0), fov=60)
    return scene


def orthographic_metals():
    scene = metal_spheres()
    camera = scene.active_camera.camera
    camera.projection = "orthographic"
    camera.ortho_scale = 6
    return scene


@pytest.mark.parametrize(
    "build, size, options",
    [
        (metal_spheres, (48, 27), {"bounces": 4}),
        (metal_spheres, (48, 27), {"bounces": 2, "light_samples": 2, "clamp": 1.0}),
        (window_room, (48, 32), {"bounces": 6, "samples": 8}),
        (mirror_with_sized_lights, (32, 32), {"bounces": 2, "samples": 8}),
        (orthographic_metals, (48, 27), {"bounces": 2}),
    ],
    ids=["metals", "light-samples-clamp", "window-room", "sized-lights", "orthographic"],
)
def test_lights_metals_and_the_sky_trace_the_cpus_paths(gpu, build, size, options):
    scene = build()
    assert_same_paths(gpu_path(gpu, scene, *size, **options), cpu_path(scene, *size, **options))


def test_ambient_occlusion_and_the_ao_preview_match_the_cpu(gpu):
    scene = ao_scene()
    ao = {"samples": 4, "distance": 0.6}
    g = gpu_path(gpu, scene, 48, 27, ao=ao, bounces=2)
    c = cpu_path(scene, 48, 27, ao=ao, bounces=2)
    assert_same_paths(g, c)
    np.testing.assert_allclose(g.passes["ao"], c.passes["ao"], atol=1e-6)
    preview = {"samples": 8, "distance": 0.6}
    g = gpu_path(gpu, scene, 48, 27, ao=preview, integrator="ao")
    c = cpu_path(scene, 48, 27, ao=preview, integrator="ao")
    assert g.stats["ao_rays"] == c.stats["ao_rays"] and g.stats["bounce_rays"] == 0
    np.testing.assert_allclose(g.passes["beauty"], c.passes["beauty"], atol=1e-6)


def test_cutouts_single_sided_surfaces_and_textured_emission_match_the_cpu(gpu, tmp_path):
    from aether3d.rendering.png import write_png

    checker = np.zeros((8, 8, 4), dtype=np.uint8)
    checker[..., :3] = 200
    checker[..., 3] = np.where((np.indices((8, 8)).sum(axis=0) % 2) == 0, 255, 0)
    write_png(tmp_path / "mask.png", checker)
    glow = np.zeros((2, 2, 4), dtype=np.uint8)
    glow[..., 3] = 255
    glow[0, 0, :3] = (255, 40, 0)
    glow[1, 1, :3] = (0, 60, 255)
    write_png(tmp_path / "glow.png", glow)
    scene = dark(a3.Scene())
    floor = scene.add_plane("Floor", size=8)
    floor.material = lambert(scene, "Floor", (0.8, 0.8, 0.8))
    screen = scene.add_plane("Screen", size=2, position=(0, 0, 1.0))
    lace = scene.create_material(
        "Lace", model="simple", base_color="#ffffff", alpha_mode="mask", alpha_cutoff=0.5
    )
    lace.set_texture(
        "base_color",
        scene.add_image_texture("Mask", tmp_path / "mask.png", filter="nearest", mipmaps=False),
    )
    screen.material = lace
    back = scene.add_plane("One Sided", size=1.5, position=(1.5, 0.5, 0.6))
    back.rotation_euler = (180, 0, 0)  # its front faces down: from above, it is invisible
    back.material = scene.create_material(
        "Front Only", model="simple", base_color="#ffffff", double_sided=False
    )
    panel = scene.add_plane("Neon", size=1.0, position=(-1.5, -1.0, 1.6))
    panel.rotation_euler = (180, 0, 0)  # glowing downward onto the floor
    neon = scene.create_material(
        "Neon", model="simple", base_color="#000000", emissive="#ffffff", emissive_strength=3.0
    )
    neon.set_texture(
        "emissive",
        scene.add_image_texture("Glow", tmp_path / "glow.png", filter="nearest", mipmaps=False),
    )
    panel.material = neon
    scene.add_directional_light("Sun", direction=(0.2, 0.1, -1.0), intensity=1.0, angle=2.0)
    scene.add_camera("Cam", position=(0, -4, 4), look_at=(0, 0, 0.3), fov=50)
    g = gpu_path(gpu, scene, 48, 36, samples=8, bounces=2)
    c = cpu_path(scene, 48, 36, samples=8, bounces=2)
    assert g.stats["pass_through_rays"] > 0
    assert abs(g.stats["pass_through_rays"] - c.stats["pass_through_rays"]) <= 0.02 * max(
        c.stats["pass_through_rays"], 1
    )
    assert_same_paths(g, c)


@pytest.mark.skipif(not HAS_MANIFOLD, reason="the castle needs aether3d[boolean]")
def test_the_textured_procedural_castle_matches_the_cpu(gpu):
    from castle_path_render import build_scene

    scene = build_scene(96, 54)
    g = gpu_path(gpu, scene, 96, 54, bounces=4)
    c = cpu_path(scene, 96, 54, bounces=4)
    # Procedural textures in float32: wider bounds (see the module docstring).
    assert_same_paths(g, c, median=5e-3, flipped=0.2, mean=5e-3)


# --------------------------------------------------------------------------- #
# Analytic references on the GPU
# --------------------------------------------------------------------------- #


def test_a_white_furnace_is_exact_on_the_gpu(gpu):
    """A Lambertian plane open to a uniform sky reflects albedo x radiance, with no noise."""
    scene = a3.Scene()
    plane = scene.add_plane("Plane", size=4)
    plane.material = lambert(scene, "Paint", (0.5, 0.25, 0.75))
    scene.world.configure(environment={"type": "uniform", "color": "#ffffff", "intensity": 2.0})
    scene.add_camera("Cam", position=(0, 0, 3), look_at=(0, 0, 0))
    for bounces in (0, 3):
        r = gpu_path(gpu, scene, 16, 16, samples=4, bounces=bounces)
        covered = r.coverage == 1.0
        assert covered.sum() > 100
        expected = np.broadcast_to([1.0, 0.5, 1.5], r.linear[covered].shape)
        np.testing.assert_allclose(r.linear[covered], expected, rtol=2e-6)


def test_a_point_light_follows_the_inverse_square_law_on_the_gpu(gpu):
    scene = dark(a3.Scene())
    floor = scene.add_plane("Floor", size=8)
    floor.material = lambert(scene, "White", (0.8, 0.8, 0.8))
    height, intensity = 1.5, 3.0
    scene.add_point_light("Bulb", position=(0, 0, height), intensity=intensity)
    scene.add_camera(
        "Top", projection="orthographic", ortho_scale=4, position=(0, 0, 5), look_at=(0, 0, 0)
    )
    n = 32
    r = gpu_path(gpu, scene, n, n, samples=256, bounces=0)
    sub = (np.arange(8) + 0.5) / 8
    xs = ((np.arange(n)[:, None] + sub[None, :]) / n * 4 - 2).ravel()
    ys = (2 - (np.arange(n)[:, None] + sub[None, :]) / n * 4).ravel()
    gx, gy = np.meshgrid(xs, ys)
    d2 = gx * gx + gy * gy + height * height
    value = 0.8 * intensity * (height / np.sqrt(d2)) / d2
    expected = value.reshape(n, 8, n, 8).mean(axis=(1, 3))
    np.testing.assert_allclose(r.linear[..., 0], expected, rtol=0.015)
    assert abs(float(r.linear[..., 0].mean() / expected.mean()) - 1.0) < 0.002


# --------------------------------------------------------------------------- #
# Determinism, progress, outputs
# --------------------------------------------------------------------------- #


def test_renders_repeat_bit_for_bit_however_they_are_dispatched(gpu, monkeypatch):
    from aether3d.rendering.gpu import pathtrace as gpu_pathtrace

    scene = cornell_box()
    first = gpu_path(gpu, scene, 40, 30, samples=6, bounces=3, seed=3)
    again = gpu_path(gpu, scene, 40, 30, samples=6, bounces=3, seed=3)
    assert np.array_equal(first.passes["beauty"], again.passes["beauty"])
    # Small tiles, one sample per dispatch (a progress callback asks for every pass).
    monkeypatch.setattr(gpu_pathtrace, "TILE", 16)
    seen = []
    tiled = scene.render(
        width=40,
        height=30,
        backend="gpu_path",
        adapter=gpu.adapter,
        path={"samples": 6, "bounces": 3, "seed": 3},
        default_lights=False,
        progress=lambda done, total, seconds: seen.append(done),
    )
    assert seen == [1, 2, 3, 4, 5, 6]
    assert tiled.backend["dispatch"]["tiles"] == 6
    for name in ("beauty", "variance", "albedo", "depth"):
        assert np.array_equal(first.passes[name], tiled.passes[name]), name
    other = gpu_path(gpu, scene, 40, 30, samples=6, bounces=3, seed=4)
    assert not np.array_equal(first.passes["beauty"], other.passes["beauty"])


def test_progress_can_stop_a_gpu_render(gpu):
    scene = cornell_box()
    r = scene.render(
        width=16,
        height=16,
        backend="gpu_path",
        adapter=gpu.adapter,
        path={"samples": 8, "bounces": 1},
        default_lights=False,
        progress=lambda done, total, seconds: done < 3,
    )
    assert r.backend["samples_rendered"] == 3
    assert int(r.passes["sample_count"].max()) == 3
    assert any(d.code == "RENDER_STOPPED" for d in r.diagnostics)


def test_results_report_the_gpu_honestly(gpu, tmp_path):
    from PIL import Image

    scene = cornell_box()
    r = gpu_path(gpu, scene, 16, 16, samples=2, bounces=1)
    b = r.backend
    assert b["name"] == "gpu_path" and b["device"] == "gpu" and b["gpu"] is True
    assert b["adapter"]["name"] == gpu.name
    assert b["hardware_ray_tracing"] is False
    assert "workers" not in b and b["samples_rendered"] == 2
    cpu = cpu_path(scene, 16, 16, samples=2, bounces=1)
    assert set(r.passes) == set(cpu.passes)
    for name, values in r.passes.items():
        assert values.shape == cpu.passes[name].shape and values.dtype == cpu.passes[name].dtype
    assert r.stats["rays"] > 0 and r.stats["rays_per_second"] > 0
    assert {o["name"]: o["visible_samples"] for o in r.objects} == {
        o["name"]: o["visible_samples"] for o in cpu.objects
    }
    path = r.save(tmp_path / "gpu.png")
    source = Image.open(path).text["Source"]
    assert "path tracer on the GPU" in source and gpu.name in source


def test_scene_edits_reach_the_next_gpu_render(gpu):
    scene = cornell_box()
    before = gpu_path(gpu, scene, 16, 16, samples=2, bounces=1)
    scene.get_by_name("Lamp").position = (0.4, 0.3, 1.99)
    scene.materials["Red"].base_color = "#2040c0"
    after = gpu_path(gpu, scene, 16, 16, samples=2, bounces=1)
    assert not np.array_equal(before.passes["beauty"], after.passes["beauty"])
    reference = cpu_path(scene, 16, 16, samples=2, bounces=1)
    assert_same_paths(after, reference, centers=0.9)  # diagonal pixel centers lie on seams


def test_undo_restores_the_gpu_render_exactly(gpu):
    scene = cornell_box()
    before = gpu_path(gpu, scene, 16, 16, samples=2, bounces=2)  # caches are warm now
    scene.get_by_name("Lamp").position = (0.4, 0.3, 1.99)
    scene.materials["Red"].base_color = "#2040c0"
    scene.remove(next(o for o in scene if o.mesh is not None and o.name != "Lamp"))
    changed = gpu_path(gpu, scene, 16, 16, samples=2, bounces=2)
    assert not np.array_equal(before.passes["beauty"], changed.passes["beauty"])
    scene.history.undo(3)
    again = gpu_path(gpu, scene, 16, 16, samples=2, bounces=2)
    assert np.array_equal(again.passes["beauty"], before.passes["beauty"])  # bit for bit
    scene.history.redo(3)
    redone = gpu_path(gpu, scene, 16, 16, samples=2, bounces=2)
    assert np.array_equal(redone.passes["beauty"], changed.passes["beauty"])


# --------------------------------------------------------------------------- #
# Limits and errors
# --------------------------------------------------------------------------- #


def test_limits_are_checked_before_the_gpu_is_used(gpu, monkeypatch):
    from aether3d.rendering.gpu import pathtrace as gpu_pathtrace

    scene = cornell_box()
    with pytest.raises(RenderError) as exc:
        scene.render(
            width=64,
            height=64,
            backend="gpu_path",
            adapter=gpu.adapter,
            limits=a3.RenderLimits(max_gpu_memory=2**16),
            default_lights=False,
        )
    assert exc.value.code == "RESOURCE_LIMIT" and "GPU memory" in exc.value.message
    with pytest.raises(RenderError) as exc:
        scene.render(
            width=64,
            height=64,
            backend="gpu_path",
            adapter=gpu.adapter,
            path={"samples": 1000},
            limits=a3.RenderLimits(max_path_rays=10**6),
            default_lights=False,
        )
    assert exc.value.code == "RESOURCE_LIMIT" and "rays" in exc.value.message
    monkeypatch.setattr(gpu_pathtrace, "STACK_SIZE", 2)
    with pytest.raises(RenderError) as exc:
        gpu_path(gpu, scene, 8, 8)
    assert exc.value.code == "GPU_LIMIT_EXCEEDED" and "stack" in exc.value.message


@pytest.mark.parametrize(
    "options, code",
    [
        ({"path": {"workers": 2}}, "INVALID_SETTINGS"),
        ({"msaa": 4}, "INVALID_SETTINGS"),
        ({"samples": 2}, "INVALID_SETTINGS"),
        ({"shadows": True}, "INVALID_SETTINGS"),
        ({"path": {"bounces": -1}}, "INVALID_PATH_SETTINGS"),
    ],
)
def test_options_that_do_not_apply_are_refused(options, code):
    with pytest.raises(RenderError) as exc:
        cornell_box().render(width=8, height=8, backend="gpu_path", **options)
    assert exc.value.code == code


def test_without_wgpu_gpu_path_fails_and_never_falls_back(monkeypatch):
    from aether3d.rendering.gpu import device

    monkeypatch.setattr(device, "_wgpu", None)
    monkeypatch.setitem(sys.modules, "wgpu", None)
    with pytest.raises(RenderError) as exc:
        cornell_box().render(width=8, height=8, backend="gpu_path", default_lights=False)
    assert exc.value.code == "GPU_UNAVAILABLE"
    # The CPU path tracer still works: it is the fallback the user chooses.
    assert cornell_box().render(width=8, height=8, backend="path", default_lights=False)


def test_a_slow_kernel_compile_is_reported():
    from aether3d.rendering.gpu.pathtrace import compile_diagnostic

    assert compile_diagnostic(12.0) is None  # DXC on Windows; Vulkan and Metal: ~1 s
    slow = compile_diagnostic(70.0)
    assert slow.code == "SLOW_SHADER_COMPILE" and slow.severity == "info"
    assert "70 s" in slow.message


def _adapter(index, backend, *, software=False, kind="discrete"):
    from aether3d.rendering.gpu.device import AdapterInfo

    return AdapterInfo(
        index, f"GPU {index}", "", 0, 0, "cpu" if software else kind, backend, "", software, (), ()
    )


def test_direct3d_12_needs_dxc_and_warp_is_refused(monkeypatch):
    from aether3d.rendering.gpu import device

    monkeypatch.setattr(device, "_dxc_path", None)
    assert device.pathtrace_support(_adapter(0, "vulkan")) is None
    assert device.pathtrace_support(_adapter(0, "metal")) is None
    assert device.pathtrace_support(_adapter(0, "vulkan", software=True)) is None  # lavapipe
    assert "DXC" in device.pathtrace_support(_adapter(0, "d3d12"))
    warp = device.pathtrace_support(_adapter(0, "d3d12", software=True))
    assert "WARP" in warp and "36 minutes" in warp
    monkeypatch.setattr(device, "_dxc_path", "C:/sdk/dxcompiler.dll")
    assert device.pathtrace_support(_adapter(0, "d3d12")) is None
    assert device.pathtrace_support(_adapter(0, "d3d12", software=True)) is not None


def test_the_path_tracer_picks_an_adapter_that_can_run_it(monkeypatch):
    """A Windows GPU seen through Direct3D 12 and Vulkan: rasterizer renders use Direct3D
    12 (the native API), path tracing the Vulkan adapter; with only WARP it is refused."""
    from aether3d.rendering.gpu import device

    def create(wgpu, info, raw):
        return type("Context", (), {"usable": lambda self: True, "info": info})()

    d3d12, vulkan = _adapter(0, "d3d12"), _adapter(1, "vulkan")
    both = [(d3d12, None), (vulkan, None)]
    monkeypatch.setattr(device, "_dxc_path", None)
    monkeypatch.setattr(device, "load_wgpu", lambda: None)
    monkeypatch.setattr(device, "_create_context", create)
    monkeypatch.setattr(device, "_contexts", {})
    monkeypatch.setattr(device, "_NATIVE_API", {sys.platform: "d3d12"})
    monkeypatch.setattr(device, "enumerate_adapters", lambda refresh=False: both)
    assert device.get_context()[0].info is d3d12
    assert device.get_context(path_tracer=True)[0].info is vulkan
    with pytest.raises(RenderError) as exc:
        device.get_context("d3d12", path_tracer=True)  # explicitly Direct3D 12, no DXC
    assert exc.value.code == "GPU_FEATURE_UNSUPPORTED" and "AETHER3D_GPU_DXC" in exc.value.hint
    warp = _adapter(0, "d3d12", software=True)
    monkeypatch.setattr(device, "enumerate_adapters", lambda refresh=False: [(warp, None)])
    with pytest.raises(RenderError) as exc:
        device.get_context(path_tracer=True)
    assert exc.value.code == "GPU_FEATURE_UNSUPPORTED" and "WARP" in exc.value.message
    assert exc.value.details["adapters"][0]["reason"]


def test_adapters_that_cannot_run_the_path_tracer_refuse_it():
    """On a real adapter that cannot run the path tracer (WARP in the Windows CI job), a
    gpu_path render fails at once with the reason, and never reaches the compiler."""
    import os

    if os.environ.get("AETHER3D_GPU_TESTS") == "skip":
        pytest.skip("GPU tests are switched off here (AETHER3D_GPU_TESTS=skip)")
    pytest.importorskip("wgpu")
    from aether3d.rendering.gpu import device

    try:
        adapters = device.enumerate_adapters()
    except RenderError as exc:
        pytest.skip(f"no GPU adapter here: {exc.message}")
    refused = [info for info, _ in adapters if device.pathtrace_support(info)]
    if not refused:
        pytest.skip("every adapter here can run the GPU path tracer")
    with pytest.raises(RenderError) as exc:
        cornell_box().render(width=8, height=8, backend="gpu_path", adapter=str(refused[0].index))
    assert exc.value.code == "GPU_FEATURE_UNSUPPORTED"


def test_a_dxc_path_that_does_not_exist_is_an_error(tmp_path):
    pytest.importorskip("wgpu")
    import subprocess

    code = (
        "from aether3d.rendering.gpu import device\n"
        "try:\n"
        "    device.load_wgpu()\n"
        "except Exception as exc:\n"
        "    print(exc.code, exc.location)\n"
    )
    env = {**__import__("os").environ, "AETHER3D_GPU_DXC": str(tmp_path / "dxcompiler.dll")}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.stdout.split() == ["GPU_UNAVAILABLE", "AETHER3D_GPU_DXC"], out.stdout + out.stderr


def test_the_cli_renders_with_the_gpu_path_tracer(gpu, tmp_path, capsys):
    from aether3d.cli import main

    project = tmp_path / "box.aether"
    cornell_box().save(project)
    out = tmp_path / "box.png"
    argv = [
        "render", str(project), "-o", str(out), "--backend", "gpu_path", "--spp", "4",
        "--bounces", "2", "--seed", "7", "--no-default-lights", "--width", "24", "--height",
        "24", "--passes", str(tmp_path / "passes"), "--json",
    ]  # fmt: skip
    if gpu.adapter:
        argv += ["--adapter", gpu.adapter]
    code = main(argv)
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["backend"]["name"] == "gpu_path"
    assert data["backend"]["samples_rendered"] == 4 and out.exists()
    assert sorted(p.name for p in (tmp_path / "passes").iterdir())[:2] == [
        "albedo.pfm",
        "beauty.pfm",
    ]
    code = main(["render", str(project), "-o", str(out), "--backend", "gpu_path",
                 "--workers", "2", "--force", "--json"])  # fmt: skip
    data = json.loads(capsys.readouterr().out)
    assert code == 1 and "--workers" in data["error"]["message"]
