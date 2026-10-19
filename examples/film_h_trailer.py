"""Showcase H -- the 1080p60 trailer: the showcases' own frames, cut together by Aether3D.

Every frame of the trailer is a frame Aether3D rendered for showcases A, B, D, E and F (at
1920x1080, 60 frames per second, never repeated) or a title card it drew itself (its
stroke font and anti-aliased line rasterizer); dissolves between shots mix the frames in
linear light (sRGB decoded, mixed, encoded again). FFmpeg only encodes the finished frames:
``trailer_1080p.mp4`` (H.264, preset slow, CRF 10, BT.709 tags), ``trailer_master.mkv``
(FFV1, lossless) and the committed ``trailer_preview.mp4`` (960 wide), checked by PSNR and
SSIM against the frames. Shots whose frames are missing are left out and listed.

    .venv/bin/python examples/film_h_trailer.py [--quick]

``--quick`` (the tests) cuts tiny synthetic shots instead of the showcases' frames.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from film_common import FPS, OUT, arguments, encode, rel, write_report  # noqa: E402

from aether3d.cad.drawing.font import text_strokes, text_width  # noqa: E402
from aether3d.cad.drawing.raster import _stroke  # noqa: E402  (the drawings' line rasterizer)
from aether3d.materials.images import decode_image  # noqa: E402
from aether3d.rendering.color import linear_to_srgb, srgb_to_linear  # noqa: E402
from aether3d.rendering.png import write_png  # noqa: E402

DISSOLVE = 0.4  # seconds of each dissolve
SHOTS = (  # showcase folder, sequence name, first and last second used, caption
    ("d_world", "flight", 0.0, 5.0, "a camera through a generated world"),
    ("a_rigid", "performance", 0.4, 4.6, "rigid parts, crossfaded motion"),
    ("e_studio", "orbit", 0.0, 2.5, "studio light rigs, glass, depth of field"),
    ("b_character", "greeting", 0.0, 3.6, "a skinned character, path traced"),
    ("f_night", "truck", 0.0, 2.0, "emissive light at night"),
)
INK = (0.94, 0.93, 0.9)
PAPER = (0.035, 0.04, 0.05)


def title_card(width: int, height: int, lines: list[tuple[str, float]]) -> np.ndarray:
    """Centred lines of text (``(text, height as a fraction of the image's)``), linear RGB."""
    scale = height / 1080.0
    ink = np.zeros((height, width))
    total = sum(h for _, h in lines) * height * 1.9
    y = 0.5 * (height - total)
    for text, h in lines:
        size = h * height
        y += 1.4 * size
        x = 0.5 * (width - text_width(text, size))
        segs = []
        for s in text_strokes(text, (0.0, 0.0), size):
            pts = s * np.array([1.0, -1.0]) + np.array([x, y])
            for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
                n = max(1, int(math.hypot(*(p1 - p0)) / 12.0) + 1)  # (short pieces)
                xs, ys = np.linspace(p0[0], p1[0], n + 1), np.linspace(p0[1], p1[1], n + 1)
                segs.extend(np.stack([xs[:-1], ys[:-1], xs[1:], ys[1:]], axis=1))
        if segs:
            arr = np.array(segs)
            _stroke(ink, arr, np.full(len(arr), max(1.0, size / 14.0) * (0.5 + 0.5 * scale)))
        y += 0.5 * size
    cov = np.clip(ink, 0.0, 1.0)[..., None]
    paper = srgb_to_linear(np.asarray(PAPER))[None, None, :]
    inked = srgb_to_linear(np.asarray(INK))[None, None, :]
    return paper * (1.0 - cov) + inked * cov


def read(path: Path) -> np.ndarray:
    pixels = np.asarray(decode_image(path.read_bytes()).pixels)[..., :3]
    return srgb_to_linear(pixels.astype(np.float64) / 255.0)


def to_png(linear: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(linear_to_srgb(np.clip(linear, 0.0, 1.0)) * 255.0), 0, 255).astype(
        np.uint8
    )


class Cut:
    """Writes the trailer's frames in order; a dissolve mixes the last ``n`` written frames'
    sources with the next shot's first ones."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("frame_*.png"):
            old.unlink()
        self.count = 0
        self.tail: list[np.ndarray] = []  # the previous shot's last frames (linear)

    def shot(self, frames, fade_in: int = 0, fade_out: int = 0, dissolve: int = 0) -> int:
        """``frames``: linear images. Its first frames dissolve out of the previous shot's
        held-back tail (a smoothstep mix); its last ``dissolve`` frames are held back for the
        next shot to dissolve out of. Returns the frames written."""
        frames = list(frames)
        written = 0
        for k, img in enumerate(frames):
            if k < len(self.tail):  # the dissolve from the previous shot
                w = (k + 1) / (len(self.tail) + 1)
                w = w * w * (3.0 - 2.0 * w)
                img = (1.0 - w) * self.tail[k] + w * img
            elif k < fade_in:
                img = img * ((k + 1) / (fade_in + 1))
            if k >= len(frames) - fade_out:
                img = img * ((len(frames) - k) / (fade_out + 1))
            if k >= len(frames) - dissolve:
                continue  # (held back: the next shot dissolves out of these)
            write_png(self.folder / f"frame_{self.count:05d}.png", to_png(img))
            self.count += 1
            written += 1
        self.tail = frames[len(frames) - dissolve :] if dissolve else []
        return written


def synthetic_shots(folder: Path, size) -> None:
    """Tiny stand-ins for the showcases' frames (the tests' --quick run)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    for k, (name, seq, _first, last, _caption) in enumerate(SHOTS):
        d = folder / name / "frames" / seq
        d.mkdir(parents=True, exist_ok=True)
        for f in range(int(round(last * FPS)) + 1):
            hue = (k / len(SHOTS) + f / 600.0) % 1.0
            rgb = [0.5 + 0.5 * np.cos(2 * np.pi * (hue + c / 3) + xx / w) for c in range(3)]
            img = np.stack(rgb, axis=-1) * (0.6 + 0.4 * yy / h)[..., None]
            write_png(d / f"frame_{f:05d}.png", to_png(img))


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "h_trailer"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    size = (160, 90) if args.quick else (1920, 1080)
    source = OUT
    if args.quick:
        source = out / "sources"
        synthetic_shots(source, size)
    cut = Cut(out / "frames")
    n_dis = int(round(DISSOLVE * FPS))
    title = title_card(*size, [("AETHER3D 0.20", 0.075),
                               ("ANIMATION - LIGHTING - PATH TRACING - FILM", 0.026)])  # fmt: skip
    report: dict = {"showcase": "H trailer", "quick": args.quick, "fps": FPS,
                    "size": list(size), "dissolve_s": DISSOLVE, "shots": []}  # fmt: skip
    cut.shot([title] * int(1.8 * FPS), fade_in=int(0.5 * FPS), dissolve=n_dis)
    report["shots"].append({"shot": "title", "frames": int(1.8 * FPS)})
    missing = []
    for folder, seq, t_first, t_last, caption in SHOTS:
        d = source / folder / "frames" / seq
        first, last = int(round(t_first * FPS)), int(round(t_last * FPS))
        paths = [d / f"frame_{f:05d}.png" for f in range(first, last + 1)]
        if not all(p.exists() for p in paths):
            missing.append(f"{folder}/{seq}")
            continue
        written = cut.shot((read(p) for p in paths), dissolve=n_dis)
        report["shots"].append({"shot": f"{folder}/{seq}", "caption": caption,
                                "source_frames": [first, last], "frames": len(paths),
                                "written": written})  # fmt: skip
    end_card = title_card(*size, [("EVERY FRAME RENDERED BY AETHER3D", 0.04),
                                  ("FFMPEG ONLY ENCODED THEM", 0.026)])  # fmt: skip
    cut.shot([end_card] * int(2.2 * FPS), fade_out=int(0.6 * FPS))
    report["shots"].append({"shot": "end", "frames": int(2.2 * FPS)})
    report["missing"] = missing
    report["frames"] = cut.count
    report["duration_s"] = round(cut.count / FPS, 3)
    report["video"] = encode(out / "frames", out, "trailer", quick=args.quick)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    if args.quick:
        report["sources"] = rel(source)
    write_report(out / "report.json", report)
    return report


if __name__ == "__main__":
    data = main()
    print(f"H trailer: {data['frames']} frames ({data['duration_s']} s), missing {data['missing']}")
