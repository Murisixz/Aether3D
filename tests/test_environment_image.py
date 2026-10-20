"""Image environments (0.20.0): HDR files and built-ins, Blender's mapping (measured:
tests/data/blender_env), importance sampling, every backend, files and migration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gpu_support import GPU, GPU_PATH

from aether3d import Scene
from aether3d.errors import RenderError
from aether3d.rendering.color import to_linear
from aether3d.rendering.envimage import (
    BUILTINS,
    EnvironmentImage,
    builtin,
    decode_hdr,
    decode_pfm,
    sh_ambient,
)
from aether3d.serialization import scene_from_dict, scene_to_dict

BLENDER = json.loads((Path(__file__).parent / "data" / "blender_env" / "env.json").read_text())
DIRECTIONS = {
    "+X": (1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "-Z": (0, 0, -1), "XY45": (1, 1, 0),
    "X+Z30": (math.cos(math.radians(30)), 0, math.sin(math.radians(30))),
}  # fmt: skip


# -- files ------------------------------------------------------------------------------


def rgbe(pixels: np.ndarray) -> np.ndarray:
    """Radiance RGBE bytes of float pixels (the reference encoding)."""
    m = pixels.max(axis=-1).astype(np.float64)
    mant, exp = np.frexp(m)
    scale = np.where(m > 1e-32, mant * 256.0 / np.maximum(m, 1e-300), 0.0)
    out = np.zeros(pixels.shape[:-1] + (4,), np.uint8)
    out[..., :3] = np.clip(pixels * scale[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = np.where(m > 1e-32, exp + 128, 0)
    return out


def hdr_bytes(pixels: np.ndarray, rle: bool) -> bytes:
    h, w = pixels.shape[:2]
    head = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode()
    data = rgbe(pixels)
    if not rle:
        return head + data.tobytes()
    body = bytearray()
    for row in data:
        body += bytes([2, 2, w >> 8, w & 255])
        for c in range(4):
            x, ch = 0, row[:, c]
            while x < w:
                run = 1
                while x + run < w and run < 127 and ch[x + run] == ch[x]:
                    run += 1
                if run >= 3:
                    body += bytes([128 + run, ch[x]])
                    x += run
                else:
                    n = min(128, w - x)
                    body += bytes([n]) + bytes(ch[x : x + n])
                    x += n
    return head + bytes(body)


def pfm_bytes(pixels: np.ndarray, little: bool = True) -> bytes:
    h, w = pixels.shape[:2]
    data = pixels[::-1].astype("<f4" if little else ">f4").tobytes()
    return f"PF\n{w} {h}\n{-1.0 if little else 1.0}\n".encode() + data


def test_hdr_and_pfm_files_decode_exactly():
    rng = np.random.default_rng(3)
    pix = (rng.random((6, 40, 3)) * 50).astype(np.float32)
    pix[:, 10:30] = (2.5, 1.0, 0.25)  # runs for the run-length coder
    expect = decode_hdr(hdr_bytes(pix, rle=False))
    assert np.array_equal(decode_hdr(hdr_bytes(pix, rle=True)), expect)
    # RGBE keeps 8 bits relative to each pixel's brightest channel.
    assert (np.abs(expect - pix) <= pix.max(axis=-1, keepdims=True) / 128).all()
    for little in (True, False):
        assert np.array_equal(decode_pfm(pfm_bytes(pix, little)), pix)


@pytest.mark.parametrize(
    "data",
    [
        b"P6\n1 1\n255\n\x00\x00\x00",
        b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n\x00\x00\x00\x00",
        b"#?RADIANCE\n\n-Y 99999 +X 99999\n",
        b"#?RADIANCE\n\n-Y 2 +X 2\n\x00",
        b"#?RADIANCE\n\n+X 2 -Y 2\n" + bytes(16),
        b"#?RADIANCE\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 8, 200, 1]),
    ],
)
def test_malformed_hdr_files_are_refused_with_codes(data):
    with pytest.raises(RenderError) as info:
        decode_hdr(data)
    assert info.value.code.startswith("ENVIRONMENT_")


def test_malformed_pfm_files_are_refused():
    for data in (b"PF\n2 2\n-1.0\n" + bytes(8), b"PX\n1 1\n-1\n" + bytes(12), b"PF\na b\n-1\n"):
        with pytest.raises(RenderError) as info:
            decode_pfm(data)
        assert info.value.code.startswith("ENVIRONMENT_")


# -- mapping and sampling ---------------------------------------------------------------


def uv_image(w: int = 256, h: int = 128) -> np.ndarray:
    pix = np.zeros((h, w, 3), np.float32)
    pix[..., 0] = ((np.arange(w) + 0.5) / w)[None, :]
    pix[..., 1] = (1.0 - (np.arange(h) + 0.5) / h)[:, None]  # row 0 at the top
    return pix


def test_directions_land_where_blender_looks_them_up():
    for rotation, table in ((0.0, BLENDER["directions"]), (90.0, BLENDER["rotated"])):
        img = EnvironmentImage.build(uv_image(), rotation, 1.0)
        for name, uv in table.items():
            d = np.array([DIRECTIONS[name]], float)
            d /= np.linalg.norm(d)
            u, v = img.uv(d)
            assert (u[0], v[0]) == pytest.approx(uv, abs=1 / 128 + 1e-3), (rotation, name)


@pytest.mark.parametrize("name", BUILTINS)
def test_importance_sampling_is_unbiased_and_its_density_consistent(name):
    img = EnvironmentImage.build(builtin(name), 25.0, 2.0)
    u = np.random.default_rng(7).random((300_000, 2))
    d, pdf = img.sample(u[:, 0], u[:, 1])
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(img.pdf(d), pdf, rtol=1e-6)
    est = (img.radiance(d) / pdf[:, None]).mean(axis=0)  # E[L / pdf] = the integral of L
    px = img.pixels.astype(np.float64)
    h, w = px.shape[:2]
    el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    domega = np.cos(el)[:, None, None] * (2 * np.pi / w) * (np.pi / h)
    exact = 2.0 * (px * domega).sum(axis=(0, 1))
    assert np.allclose(est, exact, rtol=0.01)


def test_black_images_are_not_sampled():
    img = EnvironmentImage.build(np.zeros((8, 16, 3), np.float32), 0.0, 1.0)
    assert img.tables()["black"] and not img.pdf(np.array([[0.0, 0.0, 1.0]])).any()


# -- rendering --------------------------------------------------------------------------


def floor(image: str, **env) -> Scene:
    scene = Scene()
    plane = scene.add_cube("Floor", size=1.0, position=(0, 0, -0.5))
    plane.scale = (200, 200, 1)
    plane.color = "#ffffff"
    scene.add_camera("Top", projection="orthographic", ortho_scale=1.0, position=(0, 0, 10),
                     look_at=(0, 0, 0))  # fmt: skip
    scene.world.environment.configure(type="image", image=image, **env)
    return scene


def up_plane_exact(image: str, rotation: float = 0.0) -> np.ndarray:
    """What a white Lambertian plane facing +Z reflects under the image: E / pi."""
    px = builtin(image[8:]).astype(np.float64)
    h, w = px.shape[:2]
    el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    weight = np.cos(el) * np.maximum(np.sin(el), 0.0) * (2 * np.pi / w) * (np.pi / h)
    return (px * weight[:, None, None]).sum(axis=(0, 1)) / np.pi


@pytest.mark.parametrize("backend", ["path", *(["gpu_path"] if GPU_PATH else [])])
def test_the_path_tracers_converge_to_the_image_integral(backend):
    for image in ("builtin:overcast", "builtin:sunset"):
        r = floor(image).render(width=8, height=8, backend=backend, default_lights=False,
                                path={"samples": 512, "bounces": 0})  # fmt: skip
        mean = r.linear.reshape(-1, 3).astype(np.float64).mean(axis=0)
        assert np.allclose(mean, up_plane_exact(image), rtol=0.01), image


@pytest.mark.parametrize("backend", ["cpu", *(["gpu"] if GPU else [])])
def test_the_rasterizers_light_with_the_images_harmonics_and_sun(backend):
    # Smooth skies: within half a percent; a low sun: taken out as a directional light, the
    # rest in the harmonics (measured: +2.0 % for an upward plane).
    for image, tol in (("builtin:overcast", 0.005), ("builtin:sunset", 0.025)):
        r = floor(image).render(width=4, height=4, backend=backend, samples=1,
                                default_lights=False)  # fmt: skip
        mean = r.linear.reshape(-1, 3).astype(np.float64).mean(axis=0)
        assert np.allclose(mean, up_plane_exact(image), rtol=tol), image


@pytest.mark.skipif(not GPU, reason="needs a GPU")
def test_the_gpu_rasterizer_matches_the_cpu_under_images():
    scene = floor("builtin:studio", rotation=40.0)
    scene.add_sphere("Ball", radius=0.2, position=(0, 0, 0.2))
    a = scene.render(width=48, height=48, backend="cpu", samples=2).linear
    b = scene.render(width=48, height=48, backend="gpu", supersample=2, msaa=1).linear
    assert np.abs(a - b).max() <= 1e-3 * a.max()


@pytest.mark.parametrize(
    "backend", ["cpu", "path", *(["gpu"] if GPU else []), *(["gpu_path"] if GPU_PATH else [])]
)
def test_camera_rays_see_the_image_unless_it_is_hidden(backend):
    scene = Scene()
    toward = np.array([math.cos(math.radians(20)) * math.cos(math.radians(30)),
                       math.cos(math.radians(20)) * math.sin(math.radians(30)),
                       math.sin(math.radians(20))])  # fmt: skip
    scene.add_camera("Cam", position=(0, 0, 0), look_at=tuple(toward), fov=1.0)
    scene.world.environment.configure(type="image", image="builtin:sunset", rotation=15.0)
    opts = {"path": {"samples": 4}} if "path" in backend else {}
    r = scene.render(width=8, height=8, backend=backend, default_lights=False, **opts)
    img = EnvironmentImage.build(builtin("sunset"), 15.0, 1.0)
    want = img.radiance(toward[None])[0]  # smooth sky, 20 degrees up
    assert np.allclose(r.linear[4, 4], want, rtol=0.02)
    scene.world.environment.visible = False
    r = scene.render(width=8, height=8, backend=backend, default_lights=False, **opts)
    assert np.allclose(r.linear[4, 4], to_linear(scene.world.background), rtol=1e-5)


def test_environment_files_load_from_the_project_and_report_problems(tmp_path):
    pix = np.zeros((32, 64, 3), np.float32)
    pix[:16] = (0.5, 0.6, 0.9)  # a blue sky over a black ground
    (tmp_path / "sky.hdr").write_bytes(hdr_bytes(pix, rle=True))
    scene = floor("builtin:overcast")
    scene.save(tmp_path / "p.aether")
    scene.world.environment.image = "sky.hdr"
    r = scene.render(width=4, height=4, backend="path", default_lights=False,
                     path={"samples": 256, "bounces": 0})  # fmt: skip
    expect = np.array([0.5, 0.6, 0.9]) * 1.0  # E / pi of a uniform upper hemisphere: L
    assert np.allclose(r.linear.reshape(-1, 3).mean(axis=0), expect, rtol=0.03)
    with pytest.raises(RenderError):
        scene.world.environment.image = "../outside.hdr"
    with pytest.raises(RenderError):
        scene.world.environment.image = "sky.png"
    (tmp_path / "sky.hdr").unlink()
    r = scene.render(width=4, height=4, backend="cpu", default_lights=False)
    assert "ENVIRONMENT_MISSING" in {d.code for d in r.diagnostics}


def test_environment_fields_save_migrate_and_undo():
    scene = floor("builtin:studio", rotation=30.0, visible=False)
    copy = scene_from_dict(scene_to_dict(scene))
    env = copy.world.environment
    assert (env.type, env.image, env.rotation, env.visible) == ("image", "builtin:studio", 30.0,
                                                                False)  # fmt: skip
    doc = scene_to_dict(Scene())
    doc["format_version"] = 12
    for key in ("image", "rotation", "visible"):
        del doc["scene"]["world"]["environment"][key]
    assert scene_from_dict(doc).world.environment.image is None
    scene.world.environment.rotation = 90.0
    scene.history.undo()
    assert scene.world.environment.rotation == 30.0
    with pytest.raises(RenderError, match="needs an image"):
        Scene().world.environment.type = "image"


def test_harmonics_are_exact_for_low_frequencies():
    # A sky of radiance 1 + z: its band-1 projection is exact, and E(up) / pi = 1 + 2 / 3.
    h, w = 256, 512
    el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    pix = np.repeat((1.0 + np.sin(el))[:, None, None], w, axis=1).repeat(3, axis=2)
    img = EnvironmentImage.build(pix.astype(np.float32), 0.0, 1.0)
    up = sh_ambient(img.ambient_sh(), np.array([[0.0, 0.0, 1.0]]))[0]
    assert up == pytest.approx([1.0 + 2.0 / 3.0] * 3, rel=1e-3)
