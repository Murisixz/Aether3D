"""Shared helpers of the Stage 16 world showcases (``examples/worlds_*.py``).

Everything visible is built and rendered by Aether3D: worlds, structures, plants, the
character, the cameras and every frame. FFmpeg only encodes the rendered PNG frames
(H.264 with ``libx264 -preset slow -crf 14`` at 1920x1080, a lossless FFV1 master and a
small preview), and the encodes are checked against the frames (PSNR and SSIM).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "worlds"

LIGHTING = {
    # name: (sun direction (of travel), intensity, sun glow colour, fog colour, horizon glow,
    #        ambient intensity, exposure)
    "midday": ((0.35, 0.55, -0.75), 2.2, (1.0, 0.95, 0.85), (0.74, 0.82, 0.92), 0.15, 0.45, 0.0),
    "morning": ((0.75, 0.45, -0.38), 2.0, (1.0, 0.86, 0.66), (0.78, 0.82, 0.88), 0.35, 0.4, 0.0),
    "golden": ((0.82, 0.42, -0.2), 2.4, (1.0, 0.72, 0.42), (0.9, 0.78, 0.66), 0.7, 0.32, 0.1),
    "sunset": ((0.9, 0.3, -0.09), 2.6, (1.0, 0.55, 0.28), (0.92, 0.66, 0.52), 1.2, 0.26, 0.2),
}


def light(scene, preset: str = "golden", *, fog: float = 0.00012, fog_height: float = 0.0,
          falloff: float = 0.0012) -> Any:  # fmt: skip
    """A sun, a sky with glows, height fog and hemisphere ambient light."""
    direction, intensity, glow, fog_color, horizon, ambient, _ = LIGHTING[preset]
    sun = scene.add_directional_light("Sun", direction=direction, intensity=intensity)
    sun.light.configure(color=tuple(min(1.0, 0.75 + 0.25 * c) for c in glow), angle=0.6)
    scene.world.environment.configure(type="sky", intensity=1.0)
    scene.world.configure(ambient_intensity=ambient)
    scene.world.atmosphere.configure(
        enabled=True,
        sun="Sun",
        sun_glow=1.2 if preset in ("golden", "sunset") else 0.7,
        sun_glow_color=glow,
        horizon_glow=horizon,
        fog_density=fog,
        fog_height=fog_height,
        fog_falloff=falloff,
        fog_color=fog_color,
        fog_sun=0.6 if preset in ("golden", "sunset") else 0.25,
        ambient="hemisphere",
    )
    return sun


def exposure(preset: str) -> float:
    return LIGHTING[preset][6]


def flight(scene, name: str, eyes: list, aims: list, *, start: float, end: float, fov: float = 50.0,
           far: float = 40000.0) -> Any:  # fmt: skip
    """A camera flying through ``eyes`` (smooth path, constant speed) while looking at a
    target moving through ``aims``."""
    a = scene.animation
    cam = scene.add_camera(name, position=tuple(eyes[0]), look_at=tuple(aims[0]), fov=fov, far=far)
    aim = scene.add_object(f"{name} aim", position=tuple(aims[0]))
    a.add_path(f"{name} path", [list(map(float, p)) for p in eyes])
    a.add_path(f"{name} aim path", [list(map(float, p)) for p in aims])
    a.follow_path(cam, f"{name} path", start=start, end=end, easing=None, follow=False)
    a.follow_path(aim, f"{name} aim path", start=start, end=end, easing=None, follow=False)
    a.add_constraint(cam, {"type": "look_at", "target": aim.id})
    return cam


def render_still(scene, path: Path, *, camera=None, time: float | None = None, width=1920,
                 height=1080, path_tracing: int = 0, preset: str = "golden",
                 cascades: int = 4, distance: float = 6000.0) -> dict:  # fmt: skip
    """One still (GPU rasterizer with cascaded shadows, or the GPU path tracer with
    ``path_tracing`` samples per pixel); returns timings and the backend report."""
    options: dict[str, Any] = {"width": width, "height": height, "tonemap": "neutral",
                               "exposure": exposure(preset)}  # fmt: skip
    if camera is not None:
        options["camera"] = camera
    if time is not None:
        options["time"] = time
    t0 = _now()
    if path_tracing:
        r = scene.render(backend="gpu_path", path={"samples": path_tracing, "bounces": 3}, **options)
    else:
        r = scene.render(backend="gpu", shadows={"cascades": cascades, "distance": distance,
                                                 "map_size": 4096}, **options)  # fmt: skip
    seconds = _now() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    r.save(path)
    return {"path": str(path), "seconds": round(seconds, 2), "backend": r.backend.get("name"),
            "adapter": (r.backend.get("adapter") or {}).get("name"),
            "diagnostics": sorted({d.code for d in r.diagnostics})}  # fmt: skip


def render_video_frames(scene, frames_dir: Path, *, fps: int, start: float, end: float, camera=None,
                        width=1920, height=1080, preset: str = "golden", cascades: int = 4,
                        distance: float = 6000.0, progress=None, step: int = 1) -> dict:  # fmt: skip
    """Every frame of ``[start, end)`` at ``fps`` as PNG (GPU rasterizer, cascaded shadows),
    with per-frame seconds and the worlds' statistics."""
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    count = int(round((end - start) * fps))
    options: dict[str, Any] = {"width": width, "height": height, "tonemap": "neutral",
                               "exposure": exposure(preset), "backend": "gpu",
                               "shadows": {"cascades": cascades, "distance": distance,
                                           "map_size": 4096}}  # fmt: skip
    if camera is not None:
        options["camera"] = camera
    rows = []
    t_all = _now()
    for k in range(0, count, step):
        t = start + k / fps
        t0 = _now()
        r = scene.render(time=t, **options)
        seconds = _now() - t0
        r.save(frames_dir / f"frame_{k // step:05d}.png")
        worlds = {w.name: dict(w.runtime.last_frame) for w in scene.worlds}
        cache = (r.backend.get("resources") or {}).get("mesh_cache") or {}
        rows.append({"frame": k // step, "time": round(t, 4), "seconds": round(seconds, 3),
                     "worlds": worlds, "gpu_mesh_cache_bytes": cache.get("bytes")})  # fmt: skip
        if progress is not None:
            progress(k // step + 1, count // step)
    total = _now() - t_all
    secs = sorted(r["seconds"] for r in rows)
    return {"frames": len(rows), "fps": fps / step, "seconds": round(total, 1),
            "seconds_per_frame_median": secs[len(secs) // 2], "seconds_per_frame_max": secs[-1],
            "size": [width, height], "rows": rows}  # fmt: skip


def encode(frames_dir: Path, out_dir: Path, name: str, fps: float, *, master: bool = True) -> dict:
    """``<name>_1080p.mp4`` (libx264, preset slow, CRF 14, yuv420p, BT.709, faststart),
    ``<name>_master.mkv`` (FFV1, lossless RGB) and ``<name>_preview.mp4`` (960x540,
    CRF 23); then PSNR and SSIM of the 1080p encode against the frames."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return {"skipped": "ffmpeg is not installed"}
    pattern = str(frames_dir / "frame_%05d.png")
    rate = f"{fps:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{name}_1080p.mp4"
    commands = {
        "mp4": [exe, "-y", "-loglevel", "error", "-framerate", rate, "-i", pattern, "-c:v", "libx264",
                "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-vf",
                "scale=1920:1080:flags=lanczos:out_color_matrix=bt709:out_range=tv",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-movflags", "+faststart", str(mp4)],
        "preview": [exe, "-y", "-loglevel", "error", "-framerate", rate, "-i", pattern, "-c:v", "libx264",
                    "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-vf",
                    "scale=960:540:flags=lanczos:out_color_matrix=bt709:out_range=tv",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-movflags", "+faststart", str(out_dir / f"{name}_preview.mp4")],
    }  # fmt: skip
    if master:
        commands["master"] = [exe, "-y", "-loglevel", "error", "-framerate", rate, "-i", pattern,
                              "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
                              str(out_dir / f"{name}_master.mkv")]  # fmt: skip
    out: dict[str, Any] = {"commands": {k: " ".join(v[1:]) for k, v in commands.items()}}
    for key, cmd in commands.items():
        t0 = _now()
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, check=False)
        if run.returncode != 0:
            raise RuntimeError(f"ffmpeg {key} failed: {run.stderr[-600:]}")
        target = Path(cmd[-1])
        out[key] = {"path": str(target), "bytes": target.stat().st_size, "seconds": round(_now() - t0, 1)}
    out["verify"] = verify(mp4, pattern, rate)
    if master:
        out["verify_master"] = verify(out_dir / f"{name}_master.mkv", pattern, rate)
    return out


def verify(video: Path, pattern: str, rate: str) -> dict:
    """PSNR (dB) and SSIM of a video's decoded frames against the PNG frames (both in
    RGB, at the frames' size)."""
    exe = shutil.which("ffmpeg")
    graph = ("[0:v]scale=flags=lanczos:in_color_matrix=bt709:in_range=tv,format=rgb24[a];"
             "[1:v]format=rgb24[b];[a][b]{}")  # fmt: skip
    result: dict[str, Any] = {}
    for metric in ("psnr", "ssim"):
        cmd = [exe, "-hide_banner", "-i", str(video), "-framerate", rate, "-i", pattern,
               "-lavfi", graph.format(metric), "-f", "null", "-"]  # fmt: skip
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, check=False)
        line = [x for x in run.stderr.splitlines() if "Parsed_" in x and "All:" in x or "average:" in x]
        result[metric] = line[-1].split("] ", 1)[-1].strip() if line else run.stderr[-300:]
    return result


def sheet(paths: list, out: Path, *, labels: list[str] | None = None, columns: int = 4,
          width: int = 480) -> str:  # fmt: skip
    """A contact sheet of PNG frames (downscaled to ``width``), drawn by Aether3D."""
    from aether3d.animation.frames import contact_sheet
    from aether3d.materials.images import decode_png
    from aether3d.rendering.png import encode_png

    images = []
    for p in paths:
        pixels = decode_png(Path(p).read_bytes()).pixels[..., :3]
        images.append(downscale(pixels, width))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_png(contact_sheet(images, columns=columns, labels=labels)))
    return str(out)


def downscale(pixels: np.ndarray, width: int) -> np.ndarray:
    """Area-average downscaling by an integer factor (for previews and sheets)."""
    f = max(1, int(round(pixels.shape[1] / width)))
    h, w = (pixels.shape[0] // f) * f, (pixels.shape[1] // f) * f
    p = pixels[:h, :w].astype(np.float64).reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))
    return np.clip(np.round(p), 0, 255).astype(np.uint8)


def preview(png: Path, out: Path, width: int = 960) -> str:
    from aether3d.materials.images import decode_png
    from aether3d.rendering.png import encode_png

    pixels = decode_png(png.read_bytes()).pixels[..., :3]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_png(downscale(pixels, width)))
    return str(out)


def write_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_portable(data), indent=2) + "\n", encoding="utf-8")


def _portable(value: Any) -> Any:
    root = str(HERE.parent)
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, str) and value.startswith(root):
        return value[len(root) + 1 :]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def machine() -> dict:
    import platform

    import aether3d
    from aether3d import _native

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "aether3d": aether3d.__version__, "native": _native.status().get("enabled"),
            "python": platform.python_version()}  # fmt: skip


def progress_printer(label: str):
    t0 = _now()

    def report(done: int, total: int) -> None:
        if done == total or done % 30 == 0:
            print(f"  {label}: {done}/{total} frames, {(_now() - t0):.0f} s", flush=True)

    return report


def _now() -> float:
    return time.perf_counter()
