"""Showcase G -- the path tracer before and after: v0.19.1 against 0.20.0, measured.

The same Cornell-box room (``film_g_scene.py``: lit only by an emissive panel, eight
bounces, 960x540, the GPU path tracer) is rendered by v0.19.1 -- the real release, run
from a temporary git worktree of ``--before-ref`` (default ``main``) -- and by this
version, each in its own process (timed after a warm-up render), and every image is
measured against a reference of 4096 samples per pixel made with the same pixel filter:

* ``before`` -- v0.19.1, 64 samples (independent random numbers, a box filter);
* ``after_default`` -- 0.20.0's defaults, 64 samples: the same estimator (its error must
  equal v0.19.1's; the CPU path tracer gives the same bits);
* ``sobol`` -- Owen-scrambled Sobol' samples, 64;
* ``sobol_bh`` -- and a Blackman-Harris pixel filter (1.5 pixels; its own reference);
* ``full`` -- and adaptive sampling (threshold 0.02, at most 64) and the denoiser
  (opt-in, biased: its error includes its bias);
* ``full_equal_time`` -- the full path given v0.19.1's time.

Errors are the relative RMSE of linear luminance (``rmse / mean``) and the PSNR of the
displayed 8-bit images (neutral tonemap). It writes ``sheet.png`` (crops of every
variant and the references), ``new_in_020.png`` (the same room with a glass sphere, a
disk area light and depth of field -- none of which v0.19.1 can render) and
``report.json``. Nothing is excluded or changed between the runs but what the variant
names.

    .venv/bin/python examples/film_g_renderer.py [--quick] [--before-ref main]
        [--before-src PATH]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from film_common import OUT, arguments, rel, sheet, write_report  # noqa: E402

from aether3d.rendering.png import write_png  # noqa: E402

SCENE = HERE / "film_g_scene.py"
ROOT = HERE.parent
SPP = 64
REFERENCE_SPP = 4096
SOBOL = ["--sampler", "sobol"]
BH = ["--filter", "blackman_harris"]
VARIANTS = [  # name, version, arguments, label, reference
    ("before", "before", [], "v0.19.1: random, box", "box"),
    ("after_default", "after", [], "0.20.0 defaults", "box"),
    ("sobol", "after", SOBOL, "Sobol', box", "box"),
    ("sobol_bh", "after", SOBOL + BH, "Sobol', Blackman-Harris", "bh"),
    ("full", "after", SOBOL + BH + ["--adaptive", "0.02", "--denoise"],
     "+ adaptive + denoised", "bh"),
]  # fmt: skip
LUMA = np.array([0.2126, 0.7152, 0.0722])


def render(name: str, args: list[str], spp: int, size, out: Path, src: Path | None, cpu: bool):
    """One render in its own process (``src``: the Aether3D to import instead of this one)."""
    target = out / f"{name}.npz"
    command = [sys.executable, str(SCENE), "--out", str(target), "--spp", str(spp),
               "--width", str(size[0]), "--height", str(size[1]), *args]  # fmt: skip
    if cpu:
        command.append("--cpu")
    env = dict(os.environ)
    if src is not None:
        env["PYTHONPATH"] = str(src)
    done = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{name}: {done.stderr[-2000:]}")
    data = np.load(target)
    report = json.loads(str(data["report"]))
    return np.asarray(data["linear"], np.float64), np.asarray(data["image"]), report


def errors(linear: np.ndarray, image: np.ndarray, ref_linear: np.ndarray, ref_image) -> dict:
    y, yr = linear @ LUMA, ref_linear @ LUMA
    rmse = float(np.sqrt(np.mean((y - yr) ** 2)))
    mse8 = float(np.mean((image.astype(np.float64) - ref_image.astype(np.float64)) ** 2))
    psnr = float("inf") if mse8 == 0 else 10.0 * np.log10(255.0**2 / mse8)
    return {"relative_rmse": rmse / float(yr.mean()), "psnr_db": psnr}


def worktree(ref: str) -> tuple[Path, Path]:
    """A detached git worktree of ``ref`` in a temporary folder: (its src, the folder)."""
    folder = Path(tempfile.mkdtemp(prefix="aether3d-before-"))
    path = folder / "tree"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(path), ref],
                   check=True, capture_output=True, text=True)  # fmt: skip
    return path / "src", folder


def crop(image: np.ndarray, quick: bool) -> np.ndarray:
    """The shadow and the bounce light between the blocks and the red wall, enlarged."""
    h, w = image.shape[:2]
    y0, y1, x0, x1 = int(0.45 * h), int(0.85 * h), int(0.18 * w), int(0.58 * w)
    part = image[y0:y1, x0:x1, :3]
    k = 1 if quick else 2
    return np.kron(part, np.ones((k, k, 1), np.uint8))


def main() -> dict:
    args = arguments(__doc__, (("--before-ref", {"default": "main"}),
                               ("--before-src", {"type": Path})))  # fmt: skip
    before_ref, before_src = args.before_ref, args.before_src
    out = args.out or OUT / "g_renderer"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    quick = args.quick
    size = (96, 54) if quick else (960, 540)
    spp = 4 if quick else SPP
    ref_spp = 64 if quick else REFERENCE_SPP
    work = Path(tempfile.mkdtemp(prefix="aether3d-g-"))
    report: dict = {"showcase": "G path tracer before and after", "quick": quick,
                    "size": list(size), "spp": spp, "reference_spp": ref_spp,
                    "scene": "Cornell box, emissive panel only, 8 bounces"}  # fmt: skip
    folder = None
    try:
        src = before_src
        if src is None and not quick:
            src, folder = worktree(before_ref)
        report["before"] = {"ref": before_ref, "ran": src is not None}
        refs = {}
        for key, extra in (("box", []), ("bh", SOBOL + BH)):
            lin, img, rep = render(f"reference_{key}", [*extra, "--seed", "1"], ref_spp, size,
                                   work, None, quick)  # fmt: skip
            refs[key] = (lin, img)
            report[f"reference_{key}"] = {"seconds": rep["seconds"], "args": extra}
        results, crops, labels = {}, [], []
        for name, version, extra, label, ref in VARIANTS:
            if version == "before" and src is None:
                continue
            lin, img, rep = render(name, extra, spp, size, work,
                                   src if version == "before" else None, quick)  # fmt: skip
            row = {"label": label, "aether3d": rep["aether3d"], "seconds": rep["seconds"],
                   "spp": spp, "samples_mean": rep.get("samples_mean"), "codes": rep["codes"],
                   **errors(lin, img, *refs[ref]), "reference": ref}  # fmt: skip
            results[name] = row
            crops.append(crop(img, quick))
            labels.append(f"{label}: {row['relative_rmse']:.3f}, {row['seconds']:.1f} s")
            if name == "before":
                before_linear = lin
            if name == "after_default" and "before" in results:
                row["same_as_before"] = bool(np.array_equal(lin, before_linear))
        # Equal time: the full path with as many samples as fit in v0.19.1's time.
        base = results.get("before", results["after_default"])
        full = results["full"]
        eq_spp = max(8, int(round(spp * base["seconds"] / full["seconds"] / 8.0)) * 8)
        lin, img, rep = render("full_equal_time", VARIANTS[-1][2], eq_spp, size, work, None,
                               quick)  # fmt: skip
        row = {"label": "full path, v0.19.1's time", "aether3d": rep["aether3d"],
               "seconds": rep["seconds"], "spp": eq_spp,
               "samples_mean": rep.get("samples_mean"), "codes": rep["codes"],
               **errors(lin, img, *refs["bh"]), "reference": "bh"}  # fmt: skip
        results["full_equal_time"] = row
        crops.append(crop(img, quick))
        labels.append(f"{row['label']}: {row['relative_rmse']:.3f}, {row['seconds']:.1f} s")
        for key in ("box", "bh"):
            crops.append(crop(refs[key][1], quick))
            labels.append(f"reference ({key}, {ref_spp} spp)")
        report["variants"] = results
        b = results.get("before")
        if b is not None:
            report["improvement"] = {
                "equal_samples_error_ratio": b["relative_rmse"] / results["full"]["relative_rmse"],
                "equal_time_error_ratio": b["relative_rmse"] / row["relative_rmse"],
                "sobol_error_ratio": b["relative_rmse"] / results["sobol"]["relative_rmse"],
            }  # fmt: skip
        report["sheet"] = sheet(crops, labels, out / "sheet.png", columns=3)
        lin, img, rep = render("new", SOBOL + BH + ["--adaptive", "0.02", "--denoise", "--glass"],
                               4 * spp, size, work, None, quick)  # fmt: skip
        write_png(out / "new_in_020.png", img)
        report["new_in_020"] = {"path": rel(out / "new_in_020.png"), "seconds": rep["seconds"],
                                "spp": 4 * spp, "codes": rep["codes"]}  # fmt: skip
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if folder is not None:
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                            str(folder / "tree")], check=False, capture_output=True)  # fmt: skip
            shutil.rmtree(folder, ignore_errors=True)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    write_report(out / "report.json", report)
    return report


if __name__ == "__main__":
    data = main()
    print(f"G renderer: {data['seconds']} s, {data.get('improvement')}")
