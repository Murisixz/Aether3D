"""The path tracer's public interface: options, results, files, persistence and limits."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

import aether3d as a3
from aether3d import (
    FORMAT_VERSION,
    AOSettings,
    Environment,
    PathSettings,
    RenderError,
    RenderLimits,
    Scene,
    run_scene_script,
)
from aether3d.rendering.tonemap import neutral

DATA = Path(__file__).parent / "data"
V070 = DATA / "v0.7.0" / "studio.aether"  # written by Aether3D 0.7.0 (format version 4)


def small_scene() -> Scene:
    scene = Scene()
    scene.add_plane("Ground", size=10).color = "#b0b0b0"
    ball = scene.add_sphere("Ball", radius=1.0, segments=24, rings=12, position=(0, 0, 1))
    ball.color = "#c04030"
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2.0)
    scene.add_camera("Cam", position=(0, -6, 3), look_at=(0, 0, 1))
    return scene


def render(scene, **options):
    options.setdefault("width", 32)
    options.setdefault("height", 18)
    options.setdefault("path", {"samples": 4, "bounces": 1})
    return scene.render(backend="path", **options)


def png_text(data: bytes) -> dict[str, str]:
    out, pos = {}, 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"tEXt":
            key, _, value = body.partition(b"\0")
            out[key.decode("latin-1")] = value.decode("latin-1")
        pos += 12 + length
    return out


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def test_path_renders_report_the_cpu_path_tracer_and_its_passes():
    r = render(small_scene(), ao={"samples": 2})
    assert r.backend["name"] == "path" and r.backend["device"] == "cpu"
    assert r.backend["gpu"] is False
    assert r.backend["samples"] == 4 and r.backend["bounces"] == 1 and r.backend["seed"] == 0
    assert r.backend["ao"] == {"samples": 2, "distance": 1.0, "strength": 1.0}
    assert r.backend["bvh"]["triangles"] == r.stats["triangles"] > 0
    assert set(r.passes) == {
        "beauty",
        "direct",
        "indirect",
        "albedo",
        "normal",
        "depth",
        "sample_count",
        "variance",
        "ao",
    }
    assert r.passes["beauty"].shape == (18, 32, 3) and r.passes["ao"].shape == (18, 32)
    np.testing.assert_allclose(r.passes["beauty"], r.linear, rtol=1e-6)
    np.testing.assert_allclose(
        r.passes["direct"] + r.passes["indirect"], r.passes["beauty"], rtol=1e-5, atol=1e-7
    )
    assert np.all(r.passes["sample_count"] == 4)
    assert np.all(r.passes["variance"] >= 0)
    assert r.stats["rays"] > 32 * 18 * 4 and r.stats["camera_rays"] == 32 * 18 * 4
    assert r.stats["noise"]["mean_relative_error"] > 0
    for key in ("bvh", "trace", "output", "total"):
        assert r.stats["timings"][key] >= 0
    report = json.loads(json.dumps(r.to_dict()))
    assert report["backend"]["name"] == "path" and "ao" in report["passes"]
    assert report["color"]["tonemap"] == "none"
    source = png_text(r.png_bytes())["Source"]
    assert "path tracer on the CPU" in source and "4 samples per pixel" in source
    # Depth and object ids come from rays through the pixel centers, like the raster's.
    raster = small_scene().render(width=32, height=18, samples=1)
    assert np.array_equal(r.object_index, raster.object_index)
    hit = np.isfinite(raster.depth)
    np.testing.assert_allclose(r.depth[hit], raster.depth[hit], rtol=1e-4)


def test_exposure_scales_the_image_but_not_the_passes():
    scene = small_scene()
    plain = render(scene)
    bright = render(scene, exposure=1.0)
    np.testing.assert_allclose(bright.linear, 2 * plain.linear, rtol=1e-6)
    np.testing.assert_allclose(bright.passes["beauty"], plain.passes["beauty"], rtol=1e-6)
    assert bright.color["exposure"] == 1.0 and bright.color["exposure_scale"] == 2.0


def test_ao_preview_traces_only_camera_and_ao_rays():
    r = render(small_scene(), path={"samples": 4, "integrator": "ao"}, ao={"samples": 3})
    assert r.stats["bounce_rays"] == 0 and r.stats["shadow_rays"] == 0
    assert r.stats["ao_rays"] > 0
    covered = r.coverage == 1.0
    np.testing.assert_allclose(r.linear[covered][:, 0], r.passes["ao"][covered], atol=1e-6)
    default = render(small_scene(), path={"samples": 2, "integrator": "ao"})
    assert default.backend["ao"] == AOSettings().to_dict()


def test_progress_reports_every_sample_pass_and_can_stop_the_render():
    calls = []

    def progress(done, total, seconds):
        calls.append((done, total))
        return done < 3

    r = render(small_scene(), path={"samples": 8, "bounces": 1}, progress=progress)
    assert calls == [(1, 8), (2, 8), (3, 8)]
    assert np.all(r.passes["sample_count"] == 3) and r.backend["samples_rendered"] == 3
    assert "RENDER_STOPPED" in [d.code for d in r.diagnostics]
    # The first three samples of the full render are the same samples.
    full = render(small_scene(), path={"samples": 3, "bounces": 1})
    np.testing.assert_allclose(r.linear, full.linear, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Options and limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "options,location",
    [
        ({"backend": "cpu", "path": {"samples": 4}}, "path"),
        ({"backend": "gpu", "ao": True}, "ao"),
        ({"backend": "cpu", "progress": print}, "progress"),
        ({"backend": "path", "samples": 2}, "samples"),
        ({"backend": "path", "msaa": 4}, "msaa"),
        ({"backend": "path", "shadows": True}, "shadows"),
        ({"backend": "path", "path": {"spp": 4}}, "path"),
        ({"backend": "path", "path": {"bounces": -1}}, "path.bounces"),
        ({"backend": "path", "ao": {"distance": 0}}, "ao.distance"),
        ({"backend": "path", "tonemap": "filmic"}, "tonemap"),
    ],
)
def test_options_are_validated_before_rendering(options, location):
    with pytest.raises(RenderError) as info:
        small_scene().render(width=8, height=8, **options)
    assert info.value.location == location


def test_work_and_memory_limits_stop_oversized_renders():
    scene = small_scene()
    with pytest.raises(RenderError) as info:
        render(scene, limits=RenderLimits(max_path_rays=10_000))
    assert info.value.code == "RESOURCE_LIMIT" and "rays" in info.value.message
    with pytest.raises(RenderError) as info:
        render(scene, limits=RenderLimits(max_path_memory=1_000_000))
    assert info.value.code == "RESOURCE_LIMIT" and "MiB" in info.value.message


def test_settings_objects_and_dicts_are_equivalent():
    scene = small_scene()
    a = render(scene, path=PathSettings(samples=3, bounces=2, seed=9), ao=AOSettings(samples=1))
    b = render(scene, path={"samples": 3, "bounces": 2, "seed": 9}, ao={"samples": 1})
    assert np.array_equal(a.image, b.image)


# --------------------------------------------------------------------------- #
# Display and HDR output
# --------------------------------------------------------------------------- #


def test_the_neutral_tone_mapper_offsets_colors_and_compresses_highlights():
    """Khronos PBR Neutral: a 0.04 offset (less near black), then highlight compression."""
    rng = np.random.default_rng(0)
    normal = rng.uniform(0.08, 0.7, (1000, 3))
    np.testing.assert_allclose(neutral(normal), normal - 0.04, atol=1e-12)
    dark = np.array([[0.02, 0.05, 0.3]])  # smallest channel below 0.08
    offset = 0.02 - 6.25 * 0.02**2
    np.testing.assert_allclose(neutral(dark), dark - offset, atol=1e-12)
    bright = np.linspace(0.0, 100.0, 2001)[:, None].repeat(3, axis=1)
    mapped = neutral(bright)[:, 0]
    assert np.all(np.diff(mapped) >= 0) and mapped.max() < 1.0
    assert neutral(np.zeros((1, 3))).max() == 0.0


def test_tonemap_none_keeps_legacy_raster_images_and_neutral_changes_only_highlights():
    scene = small_scene()
    legacy = scene.render(width=32, height=18)
    again = scene.render(width=32, height=18, tonemap="none")
    assert np.array_equal(legacy.image, again.image)
    mapped = scene.render(width=32, height=18, tonemap="neutral", exposure=2.0)
    assert mapped.color["tonemap"] == "neutral"
    clipped = scene.render(width=32, height=18, exposure=2.0)
    assert np.array_equal(mapped.linear, clipped.linear)  # linear is before tone mapping
    bright = clipped.linear.max(axis=2) > 1.0
    assert (
        bright.any() and (mapped.image[bright] < 255).any() and clipped.image[bright].max() == 255
    )


def read_hdr(data: bytes) -> np.ndarray:
    """An independent Radiance .hdr decoder (flat and new-style run-length scanlines)."""
    header, _, rest = data.partition(b"\n\n")
    assert header.startswith(b"#?RADIANCE") and b"FORMAT=32-bit_rle_rgbe" in header
    size, _, body = rest.partition(b"\n")
    _, h, _, w = size.split()
    h, w = int(h), int(w)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    pos = 0
    for row in range(h):
        if body[pos] == 2 and body[pos + 1] == 2:
            pos += 4
            for c in range(4):
                x = 0
                while x < w:
                    count = body[pos]
                    pos += 1
                    if count > 128:
                        out[row, x : x + count - 128, c] = body[pos]
                        pos += 1
                        x += count - 128
                    else:
                        out[row, x : x + count, c] = list(body[pos : pos + count])
                        pos += count
                        x += count
        else:
            out[row] = np.frombuffer(body[pos : pos + 4 * w], dtype=np.uint8).reshape(w, 4)
            pos += 4 * w
    e = out[..., 3].astype(np.int64)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return out[..., :3] * scale[..., None]


def test_hdr_and_pfm_files_hold_the_unclipped_linear_image(tmp_path):
    r = render(small_scene(), exposure=3.0)
    assert r.linear.max() > 1.0  # values a PNG would clip
    hdr = r.save(tmp_path / "image.hdr")
    decoded = read_hdr(hdr.read_bytes())
    # RGBE shares one exponent per pixel: each channel is exact to 1/128 of the pixel's
    # largest channel (8-bit mantissas, truncated).
    peak = r.linear.max(axis=2, keepdims=True)
    assert np.all(np.abs(decoded - r.linear) <= peak / 128 + 1e-12)
    pfm = r.save(tmp_path / "image.pfm").read_bytes()
    header, rest = pfm.split(b"\n", 3)[:3], pfm.split(b"\n", 3)[3]
    assert header == [b"PF", b"32 18", b"-1.0"]
    pixels = np.frombuffer(rest, dtype="<f4").reshape(18, 32, 3)[::-1]
    assert np.array_equal(pixels, r.linear)
    with pytest.raises(RenderError):
        r.save(tmp_path / "image.tif")  # (.exr is a master since 0.20.0)
    narrow = render(small_scene(), width=4, height=3)  # too narrow for run-length scanlines
    flat = read_hdr(narrow.save(tmp_path / "narrow.hdr").read_bytes())
    assert np.all(np.abs(flat - narrow.linear) <= narrow.linear.max(axis=2, keepdims=True) / 128)
    assert zlib.crc32(r.png_bytes())  # the PNG still encodes


# --------------------------------------------------------------------------- #
# Scene data: light sizes and environments
# --------------------------------------------------------------------------- #


def test_light_sizes_and_environments_are_validated_and_saved(tmp_path):
    scene = small_scene()
    sun = scene.get_by_name("Sun")
    sun.light.angle = 0.53
    bulb = scene.add_point_light("Bulb", position=(1, 1, 3), radius=0.25, range=9.0)
    scene.world.environment = Environment(type="sky", intensity=0.8)
    scene.world.configure(environment={"zenith_color": "#2050a0"})
    for bad in ({"radius": -1}, {"angle": 200}, {"angle": float("nan")}):
        with pytest.raises(RenderError):
            bulb.light.configure(**bad)
    with pytest.raises(RenderError):
        scene.world.configure(environment={"type": "hdri"})
    path = tmp_path / "scene.aether"
    scene.save(path)
    doc = json.loads(path.read_text())
    assert doc["format_version"] == FORMAT_VERSION == 13
    assert doc["scene"]["world"]["environment"]["type"] == "sky"
    loaded = Scene.load(path)
    assert loaded.world == scene.world
    assert loaded.get_by_name("Bulb").light.radius == 0.25
    assert loaded.get_by_name("Sun").light.angle == 0.53
    a = render(scene, path={"samples": 2, "bounces": 1})
    b = render(loaded, path={"samples": 2, "bounces": 1})
    assert np.array_equal(a.image, b.image)
    assert a.backend["environment"]["type"] == "sky"
    assert a.backend["lights"] == {
        "lights": 2,
        "sized_lights": 2,
        "spot_lights": 0,  # (0.20.0)
        "area_lights": 0,
        "emissive_triangles": 0,
    }


def test_version_4_projects_load_with_the_ambient_environment_and_point_sized_lights():
    old = Scene.load(V070)
    assert json.loads(V070.read_text())["format_version"] == 4
    assert old.world.environment == Environment()
    assert [(o.light.radius, o.light.angle) for o in old.lights] == [(0.0, 0.0), (0.0, 0.0)]
    # Raster renders are unaffected by the new fields; the path tracer renders the project.
    assert old.render(width=32, height=18).stats["pixels_covered"] > 0
    r = old.render(width=32, height=18, backend="path", path={"samples": 2, "bounces": 1})
    assert r.backend["environment"]["type"] == "ambient"
    assert r.stats["pixels_covered"] > 0


def test_scene_scripts_set_light_sizes_and_the_environment():
    scene = small_scene()
    run_scene_script(
        scene,
        {
            "format": "aether3d.scene-script",
            "version": 1,
            "operations": [
                {
                    "op": "add_light",
                    "type": "point",
                    "name": "Orb",
                    "position": [0, 0, 4],
                    "radius": 0.3,
                },
                {"op": "set_light", "light": "Sun", "angle": 2.0},
                {"op": "set_world", "environment": {"type": "uniform", "intensity": 0.5}},
            ],
        },
    )
    assert scene.get_by_name("Orb").light.radius == 0.3
    assert scene.get_by_name("Sun").light.angle == 2.0
    assert scene.world.environment.type == "uniform" and scene.world.environment.intensity == 0.5


def test_the_environment_does_not_change_raster_renders():
    scene = small_scene()
    before = scene.render(width=32, height=18)
    scene.world.environment = {"type": "sky", "intensity": 3.0}
    after = scene.render(width=32, height=18)
    assert np.array_equal(before.image, after.image)


def test_top_level_names_are_exported():
    for name in ("PathSettings", "AOSettings", "Environment"):
        assert hasattr(a3, name) and name in a3.__all__


# --------------------------------------------------------------------------- #
# Parallel tracing
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not __import__("aether3d.rendering.pathtrace.parallel", fromlist=["x"]).fork_available(),
    reason="parallel path tracing forks worker processes (Linux only)",
)
def test_worker_processes_give_the_same_image_bit_for_bit():
    scene = small_scene()
    options = {"samples": 3, "bounces": 2, "seed": 4}
    one = render(scene, width=40, height=24, path={**options, "workers": 1}, ao=True)
    three = render(scene, width=40, height=24, path={**options, "workers": 3}, ao=True)
    assert one.backend["processes"] == 1 and three.backend["processes"] == 3
    assert np.array_equal(one.linear, three.linear) and np.array_equal(one.image, three.image)
    for name in one.passes:
        assert np.array_equal(one.passes[name], three.passes[name]), name
    for key in ("camera_rays", "bounce_rays", "shadow_rays", "ao_rays", "center_rays", "paths"):
        assert one.stats[key] == three.stats[key], key
    assert [o["visible_samples"] for o in one.objects] == [
        o["visible_samples"] for o in three.objects
    ]
    calls = []
    stopped = render(
        scene,
        width=40,
        height=24,
        path={**options, "workers": 2},
        progress=lambda done, total, seconds: calls.append(done) or done < 2,
    )
    assert calls == [1, 2] and np.all(stopped.passes["sample_count"] == 2)


def test_worker_plans():
    from aether3d.rendering.pathtrace import parallel

    assert parallel.plan_workers(1, 10**12)[0] == 1
    if parallel.fork_available():
        assert parallel.plan_workers(0, 10)[0] == 1  # too small to be worth processes
        assert parallel.plan_workers(0, 10**9)[0] == min(
            parallel.cpu_count(), parallel.MAX_AUTO_WORKERS
        )
        assert parallel.plan_workers(5, 10)[0] == 5
    assert parallel.tiles(10, 4) == [(0, 2), (2, 5), (5, 8), (8, 10)]
    assert parallel.tiles(3, 8) == [(0, 1), (1, 2), (2, 3)]


def test_without_fork_renders_use_one_process(monkeypatch):
    from aether3d.rendering.pathtrace import parallel

    monkeypatch.setattr(parallel, "fork_available", lambda: False)
    r = render(small_scene(), path={"samples": 2, "bounces": 1, "workers": 4})
    assert r.backend["processes"] == 1 and "Linux only" in r.backend["parallel"]
