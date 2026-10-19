"""Film output (0.20.0): the film view transform against Blender's AgX (measured), 16-bit
and OpenEXR masters, resumable frame sequences with a manifest, and the FFmpeg commands."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation.frames import assemble_video, render_frames, video_command
from aether3d.errors import AetherError
from aether3d.materials.images import decode_png16
from aether3d.rendering.color import linear_to_srgb
from aether3d.rendering.exr import read_exr, write_exr
from aether3d.rendering.tonemap import display, film, film_curve

ROOT = Path(__file__).resolve().parent.parent
VIEW = ROOT / "research" / "2026-10-stage19" / "blender" / "e3_lights" / "blender_view_520.json.gz"
FFMPEG = shutil.which("ffmpeg")


def test_the_film_transform_lands_where_blenders_agx_does():
    with gzip.open(VIEW) as f:
        d = json.load(f)
    stops = np.array(d["stops"])
    agx = d["transforms"]["AgX|None"]
    sel = (stops >= -8) & (stops <= 10)
    for hue, rgb in d["hues"].items():
        target = linear_to_srgb(np.clip(np.array(agx[hue]), 0, 1))[sel]
        ours = film(np.array(rgb, float)[None] * 0.18 * 2.0 ** stops[sel, None])
        err = np.abs(ours - target)
        assert np.sqrt(np.mean(err**2)) < 0.011 and err.max() < 0.04, hue  # (fit: 0.008, 0.038)
    assert film(np.array([[0.18, 0.18, 0.18]]))[0] == pytest.approx([0.4613] * 3, abs=1e-4)
    curve = film_curve(np.linspace(-14, 14, 4001))
    assert np.all(np.diff(curve) >= 0) and curve[0] == 0.0 and curve[-1] == 1.0
    # A path to white: a saturated red 10 stops up is (nearly) white, not clipped red.
    assert film(np.array([[0.18 * 2**10, 0.0, 0.0]]))[0].min() > 0.98


def test_sixteen_bit_display_matches_eight_bit_rounded():
    lin = np.random.default_rng(2).random((7, 9, 3)).astype(np.float32) * 3
    for tonemap in ("none", "neutral", "film"):
        d8, d16 = display(lin, tonemap), display(lin, tonemap, bits=16)
        assert d16.dtype == np.uint16
        assert np.abs(np.rint(d16 / 65535.0 * 255.0) - d8).max() <= 1


@pytest.mark.parametrize("compression", ["none", "zip"])
@pytest.mark.parametrize("pixel_type", ["half", "float"])
def test_exr_files_round_trip(tmp_path, compression, pixel_type):
    rng = np.random.default_rng(3)
    img = (rng.random((21, 34, 3)) * 20).astype(np.float32)
    img[3:12, 5:30] = (0.25, 4.0, 1e-3)
    depth = np.where(rng.random((21, 34)) > 0.2, rng.random((21, 34)) * 50, np.inf)
    path = write_exr(tmp_path / "x.exr", {"R": img[..., 0], "G": img[..., 1], "B": img[..., 2],
                                          "Z": depth}, pixel_type=pixel_type,
                     compression=compression)  # fmt: skip
    back = read_exr(path)
    cast = np.float16 if pixel_type == "half" else np.float32
    for k, name in enumerate("RGB"):
        assert np.array_equal(back[name], img[..., k].astype(cast).astype(np.float32))
    assert np.array_equal(back["Z"], depth.astype(cast).astype(np.float32))


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg not installed")
def test_ffmpegs_exr_decoder_reads_our_files(tmp_path):
    img = np.zeros((4, 6, 3), np.float32)
    img[..., 0] = np.linspace(0, 4, 6)[None]
    img[..., 1], img[..., 2] = 0.5, 2.0
    for compression in ("none", "zip"):
        path = write_exr(tmp_path / f"{compression}.exr",
                         {"R": img[..., 0], "G": img[..., 1], "B": img[..., 2]},
                         compression=compression)  # fmt: skip
        cmd = [FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrpf16le",
               "-"]  # fmt: skip
        out = subprocess.run(cmd, capture_output=True, check=True)
        g, b, r = np.frombuffer(out.stdout, np.float16).astype(np.float32).reshape(3, 4, 6)
        half = img.astype(np.float16).astype(np.float32)
        assert np.array_equal(np.stack([r, g, b], -1), half)


def animated() -> Scene:
    scene = Scene()
    box = scene.add_cube("Box", size=0.6)
    scene.add_camera("Cam", position=(0, -4, 1.5), look_at=(0, 0, 0))
    scene.add_directional_light("Sun", direction=(0.3, 1, -1))
    scene.animation.set_channel(box, "transform.rotation_euler.z", [(0, 0), (1, 90)])
    return scene


def test_masters_resume_and_manifest(tmp_path):
    scene = animated()
    opts = {"start": 0, "end": 0.3, "fps": 10, "width": 32, "height": 24, "backend": "cpu"}
    report = render_frames(scene, tmp_path / "p16", format="png16", tonemap="film", **opts)
    first = decode_png16(Path(report["frames"][0]["path"]).read_bytes())
    assert first.dtype == np.uint16 and len(report["frames"]) == 4
    manifest = json.loads((tmp_path / "p16" / "frames.json").read_text())
    assert manifest["frame_format"] == "png16" and len(manifest["frames"]) == 4
    Path(report["frames"][2]["path"]).unlink()
    again = render_frames(scene, tmp_path / "p16", format="png16", tonemap="film", resume=True,
                          **opts)  # fmt: skip
    assert [f.get("resumed", False) for f in again["frames"]] == [True, True, False, True]
    exr = render_frames(scene, tmp_path / "exr", format="exr", passes=["depth", "albedo"],
                        **{**opts, "backend": "path", "path": {"samples": 2}})  # fmt: skip
    channels = read_exr(exr["frames"][0]["path"])
    assert {"R", "G", "B", "Z", "albedo.R"} <= set(channels)
    with pytest.raises(AetherError):
        render_frames(scene, tmp_path / "bad", format="png", passes=["depth"], **opts)
    with pytest.raises(AetherError):
        render_frames(scene, tmp_path / "bad", format="exr", passes=["depth"], **opts)  # (cpu)


def test_video_commands_are_high_quality_and_tagged():
    cmd = video_command("frames", "frame_{frame:05d}.png", 60, "out.mp4")
    assert cmd[cmd.index("-preset") + 1] == "slow" and cmd[cmd.index("-crf") + 1] == "10"
    vf = cmd[cmd.index("-vf") + 1]
    assert "out_color_matrix=bt709" in vf and "color_trc=bt709" in vf
    assert cmd[cmd.index("-colorspace") + 1] == "bt709"
    lossless = video_command("frames", "frame_{frame:05d}.png", 60, "m.mkv", lossless=True,
                             deep=True)  # fmt: skip
    assert "ffv1" in lossless and "rgb48le" in lossless
    with pytest.raises(AetherError):
        video_command("frames", "frame_{frame:05d}.png", 60, "o.mp4", crf=99)
    with pytest.raises(AetherError):
        video_command("frames", "frame_{frame:05d}.exr", 60, "o.mp4")


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg not installed")
def test_videos_assemble_with_bt709_tags(tmp_path):
    scene = animated()
    render_frames(scene, tmp_path / "f", start=0, end=0.5, fps=10, width=32, height=24,
                  backend="cpu")  # fmt: skip
    video = assemble_video(tmp_path / "f", "frame_{frame:05d}.png", 10, tmp_path / "c.mp4")
    probe = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_name,pix_fmt,color_space,color_transfer,color_primaries",
                            "-of", "json", video["output"]], capture_output=True, text=True,
                           check=True)  # fmt: skip
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert stream["color_space"] == stream["color_transfer"] == stream["color_primaries"] == "bt709"
