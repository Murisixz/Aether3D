"""Shared pieces of the Stage 19 showcases (``examples/film_*.py``): animation, lighting,
cameras and film output, everything modeled, animated, lit and rendered by Aether3D (FFmpeg
only encodes the frames Aether3D rendered).

Each script runs in two modes: ``--quick`` (the tests' mode: small images on the CPU in
seconds) and the full mode (the GPU path tracer at 1920x1080 with Owen-scrambled Sobol'
samples, a Blackman-Harris pixel filter, adaptive sampling, the denoiser and the film view
transform; sequences at a true 60 frames per second). Outputs go to
``examples/output/film/<letter>_<name>/``; frames, masters and 1080p encodes are
regenerated and stay out of Git, the previews, sheets, preview videos and reports are kept.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from worlds_common import machine, verify  # noqa: E402

from aether3d.animation.frames import assemble_video, contact_sheet, render_frames  # noqa: E402
from aether3d.rendering.png import write_png  # noqa: E402

OUT = HERE / "output" / "film"
FPS = 60


def rel(path: Path | str) -> str:
    """A path relative to ``examples/`` when it is inside it (reports), else absolute."""
    path = Path(path)
    return str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)


def arguments(description: str, extra: tuple[tuple[str, dict], ...] = ()) -> argparse.Namespace:
    """The showcases' options (``extra``: more, as ``(flag, add_argument options)``)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="small CPU renders (tests)")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--no-video", action="store_true", help="skip the sequences")
    parser.add_argument("--part", default="all", help="render only this part (see each script)")
    for flag, options in extra:
        parser.add_argument(flag, **options)
    return parser.parse_args()


def gpu_available() -> bool:
    try:
        from aether3d.rendering.gpu import gpu_status

        return bool(gpu_status().get("available"))
    except Exception:  # noqa: BLE001 - no GPU stack: the CPU renders
        return False


def film_options(
    quick: bool,
    *,
    width: int = 1920,
    height: int = 1080,
    samples: int = 256,
    bounces: int = 6,
    adaptive: float = 0.02,
    denoise: bool = True,
    exposure: float = 0.0,
    clamp: float = 8.0,
) -> dict[str, Any]:
    """Render options of a still or a frame: the full film path, or a quick CPU render.
    ``clamp`` limits the radiance one sample carries in indirect light (the path tracer's
    opt-in bias against caustic fireflies; reports say so)."""
    if quick:
        return {
            "backend": "path",
            "width": max(16, width // 12),
            "height": max(16, height // 12),
            "path": {"samples": 4, "bounces": 2, "sampler": "sobol"},
            "tonemap": "film",
            "exposure": exposure,
        }
    backend = "gpu_path" if gpu_available() else "path"
    path = {
        "samples": samples,
        "bounces": bounces,
        "sampler": "sobol",
        "filter": "blackman_harris",
        "filter_width": 1.5,
        "clamp": clamp,
    }
    if adaptive:
        path["adaptive_threshold"] = adaptive
    options = {
        "backend": backend,
        "width": width,
        "height": height,
        "path": path,
        "tonemap": "film",
        "exposure": exposure,
    }
    if denoise:
        options["denoise"] = True
    return options


def still(scene: Any, path: Path, options: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Render and save a still; returns what the report needs."""
    t0 = time.perf_counter()
    result = scene.render(**options, **extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, result.image)
    stats = result.stats or {}
    return {
        "path": rel(path),
        "seconds": round(time.perf_counter() - t0, 2),
        "size": [options["width"], options["height"]],
        "backend": result.backend.get("name"),
        "device": result.backend.get("device", "gpu" if "gpu" in str(result.backend) else "cpu"),
        "samples_per_pixel": stats.get("samples_per_pixel"),
        "samples_mean": stats.get("samples_mean"),
        "noise": (stats.get("noise") or {}).get("mean_relative_error"),
        "codes": sorted({d.code for d in result.diagnostics}),
        "image": result.image,
    }


def preview(image: np.ndarray, path: Path, width: int = 960) -> str:
    """A committed preview: the image scaled down by box averaging (integer factors)."""
    h, w = image.shape[:2]
    k = max(1, w // width)
    if k > 1:
        cut = image[: h // k * k, : w // k * k].astype(np.float64)
        image = cut.reshape(h // k, k, w // k, k, 3).mean(axis=(1, 3)).round().astype(np.uint8)
    write_png(path, image)
    return rel(path)


def sheet(images: list[np.ndarray], labels: list[str], path: Path, columns: int = 3) -> str:
    write_png(path, contact_sheet(images, columns=columns, labels=labels))
    return rel(path)


def sequence(
    scene: Any,
    out_dir: Path,
    name: str,
    *,
    start: float,
    end: float,
    options: dict[str, Any],
    quick: bool,
) -> dict[str, Any]:
    """Frames at 60 fps (resumable), then :func:`encode` them."""
    frames_dir = out_dir / "frames" / name
    t0 = time.perf_counter()
    report = render_frames(scene, frames_dir, start=start, end=end, fps=FPS, resume=True, **options)
    out: dict[str, Any] = {
        "frames": len(report["frames"]),
        "fps": FPS,
        "seconds": round(time.perf_counter() - t0, 1),
        "seconds_per_frame": round((time.perf_counter() - t0) / max(1, len(report["frames"])), 2),
        "resumed": sum(1 for f in report["frames"] if f.get("resumed")),
        "manifest": rel(report["manifest"]),
        "video": encode(frames_dir, out_dir, name, quick=quick),
    }
    return out


def encode(frames_dir: Path, out_dir: Path, name: str, *, quick: bool = False) -> dict[str, Any]:
    """``frame_%05d.png`` at 60 fps encoded by FFmpeg: ``<name>_1080p.mp4`` (H.264 preset
    slow, CRF 10, BT.709), ``<name>_master.mkv`` (FFV1, lossless) and ``<name>_preview.mp4``
    (960 wide, small enough to commit), verified by PSNR/SSIM against the frames."""
    if shutil.which("ffmpeg") is None:
        return {"skipped": "ffmpeg is not installed"}
    import subprocess

    pattern = "frame_{frame:05d}.png"
    mp4 = out_dir / f"{name}_1080p.mp4"
    master = out_dir / f"{name}_master.mkv"
    videos = {
        "mp4": assemble_video(frames_dir, pattern, FPS, mp4, crf=10, preset="slow"),
        "master": assemble_video(frames_dir, pattern, FPS, master, lossless=True),
    }
    small = out_dir / f"{name}_preview.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "scale=960:-2:flags=lanczos",
         "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-movflags", "+faststart",
         str(small)],
        check=True, timeout=3600,
    )  # fmt: skip
    rate = f"{FPS}"
    out = {
        "mp4": {"path": rel(mp4), "bytes": videos["mp4"]["bytes"],
                "command": " ".join(videos["mp4"]["command"][1:])},
        "master": {"path": rel(master), "bytes": videos["master"]["bytes"]},
        "preview": {"path": rel(small), "bytes": small.stat().st_size},
        "verify": verify(mp4, str(frames_dir / "frame_%05d.png"), rate),
        "verify_master": verify(master, str(frames_dir / "frame_%05d.png"), rate),
    }  # fmt: skip
    if quick:
        out["quick"] = True
    return out


def write_report(path: Path, data: dict[str, Any]) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value if not isinstance(v, np.ndarray)]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    data = {**data, "machine": machine()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=1) + "\n")
