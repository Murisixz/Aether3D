"""Rendering animation: frame sequences, motion blur, contact sheets, video assembly.

:func:`render_frames` renders a scene at the times of a range of frames (the timeline's
frame rate by default) with any of Aether3D's own renderers, one frame at a time (memory
stays bounded), and writes each PNG atomically. ``motion_blur={"shutter": s,
"samples": n}`` averages ``n`` renders spread over the fraction ``s`` of the frame
interval centred on the frame time, in linear light, before the same display encoding
the renderer uses: real temporal sampling, not a filter (the frame's sample budget is
shared across the shutter: the path tracers split their samples over the time samples,
the rasterizers' depth of field its aperture views). The shutter stays inside the
rendered range (``start``..``end``, or the timeline for a list of ``frames``): the first
and last frames never sample beyond it (where shots or keys may not exist).
``should_stop`` cancels between frames (the frames written so far are listed);
``progress`` is called after each frame.

Film output (0.20.0): ``format`` is ``"png"`` (8-bit display frames, the default),
``"png16"`` (16-bit display masters) or ``"exr"`` (scene-linear OpenEXR masters: half
``R``, ``G``, ``B`` after exposure, plus any of the path tracers' ``passes`` -- ``depth``
as ``Z``, ``normal``, ``albedo``, ``direct``, ``indirect``, ``variance`` as
``name.X/Y/Z`` or ``name.R/G/B``; and, from any backend, ``motion`` as ``motion.X/Y``
(pixels to the previous frame) and ``motion.Z/W`` (to the next)). ``resume=True`` keeps
the frames already written (a
valid file at a frame's path) and renders the others, so an interrupted long render
continues where it stopped. The report is also written next to the frames
(``frames.json``, after every frame, atomically): settings, per-frame seconds, camera,
diagnostic codes, sample counts and noise.

:func:`contact_sheet` lays frames out in a grid with labels drawn by Aether3D's stroke
font. :func:`video_command` gives the FFmpeg command line that turns a numbered PNG
sequence into an MP4 -- H.264 at ``preset`` slow and ``crf`` 10 by default, converted with
the BT.709 matrix to limited-range yuv420p and tagged BT.709 (FFmpeg's default conversion
of RGB frames would be BT.601; the frames' sRGB encoding is shown as players show BT.709
video), even dimensions, ``faststart`` -- or, ``lossless=True``, a
lossless FFV1 RGB master in Matroska (16-bit for ``png16`` frames); :func:`assemble_video`
runs it when FFmpeg is installed -- an encoder of frames Aether3D rendered, never a
renderer, and never run from the MCP server.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AnimationError

MAX_FRAMES = 100_000
MAX_BLUR_SAMPLES = 64
DEFAULT_PATTERN = "frame_{frame:05d}.png"
FORMATS = {"png": ".png", "png16": ".png", "exr": ".exr"}
PASS_CHANNELS = {
    "depth": ("Z",),
    "normal": ("normal.X", "normal.Y", "normal.Z"),
    "albedo": ("albedo.R", "albedo.G", "albedo.B"),
    "direct": ("direct.R", "direct.G", "direct.B"),
    "indirect": ("indirect.R", "indirect.G", "indirect.B"),
    "variance": ("variance.R", "variance.G", "variance.B"),
    "motion": ("motion.X", "motion.Y", "motion.Z", "motion.W"),  # to the previous, next frame
}
MANIFEST = "frames.json"


def _valid_frame(path: Path, fmt: str) -> bool:
    """Whether ``path`` holds a complete frame of ``fmt`` (for ``resume``)."""
    try:
        with path.open("rb") as f:
            head = f.read(8)
            f.seek(0, 2)
            size = f.tell()
    except OSError:
        return False
    if fmt == "exr":
        return size > 8 and head[:4] == b"\x76\x2f\x31\x01"
    return size > 33 and head == b"\x89PNG\r\n\x1a\n"


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def frame_list(
    scene: Any,
    start: Any = None,
    end: Any = None,
    *,
    fps: Any = None,
    step: int = 1,
    frames: Any = None,
) -> tuple[Any, list[int]]:
    """``(frame rate, frames)`` of a request (default: the timeline's range and rate)."""
    from .time import FrameRate, check_time, frame_range

    anim = scene.animation
    rate = anim.fps if fps is None else FrameRate(fps, "fps")
    if frames is not None:
        if not isinstance(frames, (list, tuple)) or not frames or len(frames) > MAX_FRAMES:
            raise _err(
                "INVALID_ARGUMENT", f"frames is a list of 1..{MAX_FRAMES} frame numbers", "frames"
            )
        out = []
        for i, f in enumerate(frames):
            if isinstance(f, bool) or not isinstance(f, (int, np.integer)):
                raise _err("INVALID_ARGUMENT", "frames are integers", f"frames[{i}]")
            out.append(int(f))
        return rate, out
    s = anim.start if start is None else check_time(start, "start")
    e = anim.end if end is None else check_time(end, "end")
    if e < s:
        raise _err("INVALID_RANGE", "end must not be before start", "end")
    found = frame_range(rate, s, e, step).tolist()
    if len(found) > MAX_FRAMES:
        raise _err("LIMIT_EXCEEDED", f"more than {MAX_FRAMES} frames", "frames")
    return rate, found


def render_frames(
    scene: Any,
    out_dir: Any,
    *,
    start: Any = None,
    end: Any = None,
    fps: Any = None,
    step: int = 1,
    frames: Any = None,
    pattern: str | None = None,
    motion_blur: Any = None,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int, float], Any] | None = None,
    format: str = "png",
    passes: Any = None,
    resume: bool = False,
    **render_options: Any,
) -> dict[str, Any]:
    """Render frames to ``out_dir`` (see the module docstring); returns the report:
    ``frames`` (``frame``, ``time``, ``path``, ``seconds``, ``camera``, diagnostic
    ``codes``; ``resumed`` for frames kept), ``fps``, ``seconds``, ``stopped``,
    ``format``."""
    from ..rendering.exr import write_exr
    from ..rendering.png import write_png
    from ..rendering.tonemap import display

    rate, todo = frame_list(scene, start, end, fps=fps, step=step, frames=frames)
    if format not in FORMATS:
        raise _err("INVALID_ARGUMENT", f"format is one of {', '.join(FORMATS)}", "format")
    suffix = FORMATS[format]
    if pattern is None:
        pattern = DEFAULT_PATTERN.replace(".png", suffix)
    if (
        not isinstance(pattern, str)
        or "{frame" not in pattern
        or "/" in pattern
        or "\\" in pattern
        or not pattern.endswith(suffix)
    ):
        raise _err(
            "INVALID_ARGUMENT",
            f"pattern is a file name with {{frame}} ending in {suffix} (no folders)",
            "pattern",
        )
    wanted_passes: list[str] = []
    if passes is not None:
        if format != "exr" or not isinstance(passes, (list, tuple)):
            raise _err("INVALID_ARGUMENT", "passes is a list of pass names, for exr frames",
                       "passes")  # fmt: skip
        for name in passes:
            if name not in PASS_CHANNELS:
                raise _err("INVALID_ARGUMENT", f"passes are {', '.join(PASS_CHANNELS)}",
                           "passes")  # fmt: skip
        traced = [p for p in passes if p != "motion"]
        if traced and render_options.get("backend") not in ("path", "gpu_path"):
            raise _err("INVALID_ARGUMENT", "passes other than motion come from the path tracers "
                       "(backend path or gpu_path)", "passes")  # fmt: skip
        wanted_passes = list(dict.fromkeys(passes))
    if not isinstance(resume, bool):
        raise _err("INVALID_ARGUMENT", "resume is true or false", "resume")
    try:
        pattern.format(frame=0)
    except (KeyError, IndexError, ValueError):
        raise _err(
            "INVALID_ARGUMENT", "pattern may only use {frame} (with a format spec)", "pattern"
        ) from None
    blur = None
    if motion_blur is not None:
        if not isinstance(motion_blur, dict) or set(motion_blur) - {"shutter", "samples"}:
            raise _err("INVALID_ARGUMENT", "motion_blur is {shutter, samples}", "motion_blur")
        shutter = motion_blur.get("shutter", 0.5)
        samples = motion_blur.get("samples", 8)
        if (
            isinstance(shutter, bool)
            or not isinstance(shutter, (int, float))
            or not 0 < float(shutter) <= 1
        ):
            raise _err(
                "INVALID_ARGUMENT",
                "shutter is a fraction of the frame interval, 0..1",
                "motion_blur.shutter",
            )
        if (
            isinstance(samples, bool)
            or not isinstance(samples, int)
            or not 2 <= samples <= MAX_BLUR_SAMPLES
        ):
            raise _err(
                "INVALID_ARGUMENT", f"samples is 2..{MAX_BLUR_SAMPLES}", "motion_blur.samples"
            )
        blur = (float(shutter), int(samples))
        if wanted_passes:
            raise _err("INVALID_ARGUMENT", "passes are not averaged over the shutter: render "
                       "them without motion blur", "passes")  # fmt: skip
    if "time" in render_options:
        raise _err("INVALID_ARGUMENT", "frame times come from the frames", "time")
    anim = scene.animation
    lo = anim.start if frames is not None or start is None else float(start)
    hi = anim.end if frames is not None or end is None else float(end)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tonemap = render_options.get("tonemap", "none")
    report: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    stopped = False
    previous = _read_manifest(out / MANIFEST) if resume else {}
    settings = {
        k: v for k, v in render_options.items()
        if isinstance(v, (str, int, float, bool, type(None), dict, list))
    }  # fmt: skip

    def summary() -> dict[str, Any]:
        return {
            "format": "aether3d.frames",
            "version": 1,
            "fps": rate.to_json(),
            "pattern": pattern,
            "frame_format": format,
            "passes": wanted_passes,
            "render": settings,
            "motion_blur": None if blur is None else {"shutter": blur[0], "samples": blur[1]},
            "frames": report,
            "stopped": stopped,
        }

    for n, f in enumerate(todo):
        if should_stop is not None and should_stop():
            stopped = True
            break
        path = out / pattern.format(frame=f)
        if resume and _valid_frame(path, format):
            kept = dict(previous.get(f, {"frame": f, "time": rate.frame_time(f)}))
            kept.update(path=str(path), resumed=True)
            report.append(kept)
            continue
        t0 = time.perf_counter()
        t = rate.frame_time(f)
        need_linear = format != "png"
        if blur is None:
            options = dict(render_options)
            if "motion" in wanted_passes:
                options["motion"] = rate.frame_duration
            if need_linear:
                options["outputs"] = sorted({*(options.get("outputs") or ("image",)), "linear"})
            result = scene.render(time=t, **options)
            linear = result.linear
            image = result.image
        else:
            shutter, samples = blur
            dt = rate.frame_duration * shutter
            acc = None
            result = None
            for j in range(samples):
                sub = min(max(t + ((j + 0.5) / samples - 0.5) * dt, lo), hi)
                options = _shutter_share(render_options, samples, j)
                outputs = options.pop("outputs", None)
                wanted = {"image", "linear"} if outputs is None else {*outputs, "linear"}
                result = scene.render(time=sub, outputs=sorted(wanted), **options)
                lin = result.linear.astype(np.float64)
                acc = lin if acc is None else acc + lin
            linear = (acc / samples).astype(np.float32)
            image = display(linear, tonemap)
        if format == "png":
            write_png(path, image)
        elif format == "png16":
            write_png(path, display(linear, tonemap, bits=16))
        else:
            channels = {"R": linear[..., 0], "G": linear[..., 1], "B": linear[..., 2]}
            for name in wanted_passes:
                plane = result.passes.get(name) if result.passes else None
                if plane is None:
                    continue
                names = PASS_CHANNELS[name]
                if len(names) == 1:
                    channels[names[0]] = plane  # (background depth: inf, as EXR holds it)
                else:
                    for k, c in enumerate(names):
                        channels[c] = plane[..., k]
            write_exr(path, channels)
        entry = {
            "frame": f,
            "time": t,
            "path": str(path),
            "seconds": time.perf_counter() - t0,
            "camera": result.camera.get("name"),
            "codes": sorted({d.code for d in result.diagnostics}),
        }
        stats = result.stats or {}
        if "samples_per_pixel" in stats:
            entry["samples_per_pixel"] = stats["samples_per_pixel"]
        noise = stats.get("noise") or {}
        if noise.get("mean_relative_error") is not None:
            entry["noise"] = noise["mean_relative_error"]
        report.append(entry)
        _write_manifest(out / MANIFEST, summary())
        if (
            progress is not None
            and progress(n + 1, len(todo), time.perf_counter() - t_all) is False
        ):
            stopped = n + 1 < len(todo)
            break
    final = summary()
    final["seconds"] = time.perf_counter() - t_all
    _write_manifest(out / MANIFEST, final)
    return {
        "frames": report,
        "fps": rate.to_json(),
        "seconds": final["seconds"],
        "stopped": stopped,
        "motion_blur": final["motion_blur"],
        "format": format,
        "manifest": str(out / MANIFEST),
    }


def _read_manifest(path: Path) -> dict[int, dict[str, Any]]:
    """The frames of an earlier run's manifest, by frame number ({} without one)."""
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(e["frame"]): e for e in data.get("frames", []) if isinstance(e, dict)}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    import json

    from .._fileio import write_atomic

    write_atomic(path, (json.dumps(data, indent=1, default=str) + "\n").encode("utf-8"))


def _shutter_share(options: dict[str, Any], samples: int, j: int) -> dict[str, Any]:
    """Render options of time sample ``j`` of ``samples`` across the shutter: the frame's
    budget is shared, not multiplied -- a path tracer traces its samples split over the
    time samples (a seed of their own each, so their noise is independent), and the
    rasterizers' depth of field averages its aperture views over them (at least 2 per
    time sample)."""
    from ..rendering.renderer import DOF_SAMPLES
    from ..rendering.settings import PathSettings

    out = dict(options)
    backend = out.get("backend", "cpu")
    if backend in ("path", "gpu_path"):
        path = PathSettings.coerce(out.get("path")).to_dict()
        path["samples"] = max(1, -(-path["samples"] // samples))
        path["seed"] = (path["seed"] + j * 0x9E3779B1) % (2**63)
        out["path"] = path
    else:
        views = out.get("dof_samples") or DOF_SAMPLES
        out["dof_samples"] = max(2, -(-views // samples))
    return out


def contact_sheet(
    images: list[Any],
    *,
    columns: int = 4,
    labels: list[str] | None = None,
    gap: int = 6,
    background: tuple[int, int, int] = (24, 24, 28),
    label_height: int = 14,
) -> np.ndarray:
    """``images`` (uint8 arrays or PNG paths, one size) in a grid, each with its label
    under it (Aether3D's stroke font); returns a uint8 RGB image."""
    from ..cad.drawing.font import text_strokes
    from ..cad.drawing.raster import _stroke

    arrays = []
    for i, im in enumerate(images):
        if isinstance(im, (str, Path)):
            from ..materials.images import decode_png

            decoded = decode_png(Path(im).read_bytes(), location=str(im))
            pixels = decoded.pixels
            if pixels.ndim == 2:
                pixels = np.repeat(pixels[..., None], 3, axis=2)
            if pixels.shape[2] < 3:
                pixels = np.repeat(pixels[..., :1], 3, axis=2)
            im = pixels[..., :3]
        im = np.asarray(im)
        if im.ndim != 3 or im.shape[2] < 3 or im.dtype != np.uint8:
            raise _err(
                "INVALID_ARGUMENT", "images are (h, w, 3) uint8 arrays or PNG paths", f"images[{i}]"
            )
        arrays.append(im[..., :3])
    if not arrays:
        raise _err("INVALID_ARGUMENT", "no images", "images")
    h, w = arrays[0].shape[:2]
    if any(a.shape[:2] != (h, w) for a in arrays):
        raise _err("INVALID_ARGUMENT", "every image must have the same size", "images")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise _err("INVALID_ARGUMENT", "columns is a positive integer", "columns")
    cols = min(columns, len(arrays))
    rows = math.ceil(len(arrays) / cols)
    label_space = (label_height + 8) if labels else 0
    cell_h = h + label_space
    sheet = np.empty((rows * cell_h + (rows + 1) * gap, cols * w + (cols + 1) * gap, 3), np.uint8)
    sheet[:] = background
    ink = np.zeros(sheet.shape[:2])
    for i, a in enumerate(arrays):
        r, c = divmod(i, cols)
        y = gap + r * (cell_h + gap)
        x = gap + c * (w + gap)
        sheet[y : y + h, x : x + w] = a
        if labels and i < len(labels):
            base = (x + 2.0, y + h + 4.0 + label_height)
            strokes = text_strokes(str(labels[i])[:80], (0.0, 0.0), float(label_height))
            segs = []
            for s in strokes:
                pts = s * np.array([1.0, -1.0]) + np.array(base)  # y down in images
                for a0, a1 in zip(pts[:-1], pts[1:], strict=True):
                    segs.append([a0[0], a0[1], a1[0], a1[1]])
            if segs:
                seg_arr = np.array(segs)
                _stroke(ink, seg_arr, np.full(len(segs), 0.7))
    cov = np.clip(ink, 0.0, 1.0)[..., None]
    sheet = (sheet * (1.0 - cov) + 235.0 * cov).astype(np.uint8)
    return sheet


PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow",
           "slower", "veryslow", "placebo")  # fmt: skip


def video_command(
    frames_dir: Any,
    pattern: str,
    fps: Any,
    output: Any,
    *,
    crf: int = 10,
    preset: str = "slow",
    lossless: bool = False,
    deep: bool = False,
) -> list[str]:
    """The FFmpeg arguments that encode a numbered PNG sequence (see the module docstring):
    H.264 MP4 (``crf``, ``preset``, BT.709), or a lossless FFV1 master (``lossless``;
    ``deep``: 16-bit frames)."""
    from .time import FrameRate

    rate = FrameRate(fps, "fps")
    if not pattern.endswith(".png"):
        raise _err("INVALID_ARGUMENT", "videos are made of PNG frames", "pattern")
    ff_pattern = pattern.replace("{frame:05d}", "%05d").replace("{frame}", "%d")
    if "{" in ff_pattern:
        raise _err(
            "INVALID_ARGUMENT", "use {frame} or {frame:05d} in the pattern for video", "pattern"
        )
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise _err("INVALID_ARGUMENT", "crf is an integer 0..51 (lower: better)", "crf")
    if preset not in PRESETS:
        raise _err("INVALID_ARGUMENT", f"preset is one of {', '.join(PRESETS)}", "preset")
    head = ["ffmpeg", "-y", "-framerate", f"{rate.numerator}/{rate.denominator}",
            "-i", str(Path(frames_dir) / ff_pattern)]  # fmt: skip
    if lossless:
        return [*head, "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
                "-pix_fmt", "rgb48le" if deep else "bgr0", str(output)]  # fmt: skip
    return [
        *head,
        "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv,"
        "format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:"
        "range=tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv", "-movflags", "+faststart",
        str(output),
    ]  # fmt: skip


def assemble_video(
    frames_dir: Any,
    pattern: str,
    fps: Any,
    output: Any,
    *,
    crf: int = 10,
    preset: str = "slow",
    lossless: bool = False,
    deep: bool = False,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    """Run :func:`video_command` (FFmpeg must be installed); returns the command, the
    output path and its size. Raises ``FFMPEG_UNAVAILABLE`` or ``VIDEO_FAILED``."""
    exe = shutil.which("ffmpeg")
    cmd = video_command(
        frames_dir, pattern, fps, output, crf=crf, preset=preset, lossless=lossless, deep=deep
    )
    if exe is None:
        raise _err(
            "FFMPEG_UNAVAILABLE", "ffmpeg is not installed; run the command elsewhere", "ffmpeg"
        )
    cmd[0] = exe
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    out = Path(output)
    if run.returncode != 0 or not out.exists():
        raise _err(
            "VIDEO_FAILED", f"ffmpeg failed ({run.returncode}): {run.stderr[-400:]}", "ffmpeg"
        )
    return {"command": cmd, "output": str(out), "bytes": out.stat().st_size}
