"""Phase 5 prototype, step 3: agreement and timings of one exported scene.

Reads OUT/SCENE (export.py's capture, the host's planes, agree.bin and runs.jsonl) and
prints one JSON summary:

- ``host_vs_python``: the Rust host's software render against render_scene's planes;
- ``image``: the hardware render against the software one -- the bias of the image mean,
  the per-pixel difference measured against the sample noise (the software render of
  the next ``samples`` samples: independent paths), pixels bitwise equal, and the pixel
  centers' instance and depth (the center pass);
- ``rays``: the agreement kernel's rays -- the same triangle for camera rays and for
  bounce rays leaving the camera hits, the same verdict for short any-hit rays;
- ``timings``: median stage times of the timed host renders (the first of each variant
  is a warm-up), render_scene's own stages (Python, warm), and the end to end
  comparison described in README.md.

    python compare.py OUT/SCENE
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ACC_PLANES = 22
ACC_DIRECT, ACC_INDIRECT, ACC_DEPTH, ACC_INSTANCE = 0, 3, 20, 21
PT_NONE = 0xFFFFFFFF


def load(path: Path, n_pix: int) -> np.ndarray:
    return np.fromfile(path, np.float32).reshape(ACC_PLANES, n_pix)


def beauty(p: np.ndarray, samples: int) -> np.ndarray:
    """The mean beauty of each pixel, (direct + indirect) / samples, RGB rows."""
    return (p[ACC_DIRECT : ACC_DIRECT + 3] + p[ACC_INDIRECT : ACC_INDIRECT + 3]) / samples


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def image(sw: np.ndarray, hw: np.ndarray, noise: np.ndarray, samples: int) -> dict:
    a, b, c = beauty(sw, samples), beauty(hw, samples), beauty(noise, samples)
    la, lb, lc = luminance(a), luminance(b), luminance(c)
    diff = lb - la
    rms = lambda x: float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))  # noqa: E731
    scale = np.maximum(np.abs(la), 1e-3)
    inst_sw = sw[ACC_INSTANCE].view(np.uint32)
    inst_hw = hw[ACC_INSTANCE].view(np.uint32)
    both = (inst_sw != PT_NONE) & (inst_hw != PT_NONE) & (inst_sw == inst_hw)
    depth = np.abs(hw[ACC_DEPTH][both] - sw[ACC_DEPTH][both]) / np.maximum(
        np.abs(sw[ACC_DEPTH][both]), 1e-30
    )
    return {
        "mean_luminance_sw": float(la.mean()),
        "mean_luminance_hw": float(lb.mean()),
        "mean_bias": float((lb.mean() - la.mean()) / max(abs(la.mean()), 1e-30)),
        "rms_difference": rms(diff),
        "rms_sample_noise": rms(lc - la),
        "difference_over_noise": rms(diff) / max(rms(lc - la), 1e-30),
        "pixels_bitwise_equal": float(np.mean(np.all(a.view(np.uint32) == b.view(np.uint32), axis=0))),
        "pixels_within_0.1%": float(np.mean(np.abs(diff) <= 1e-3 * scale)),
        "pixels_within_1%": float(np.mean(np.abs(diff) <= 1e-2 * scale)),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "center_same_instance": float(np.mean(inst_sw == inst_hw)),
        "center_depth_rel_diff_median": float(np.median(depth)) if depth.size else 0.0,
        "center_depth_rel_diff_max": float(np.max(depth)) if depth.size else 0.0,
    }


def rays(agree: np.ndarray) -> dict:
    u = agree.view(np.uint32)

    def same(k_sw: int, k_hw: int, where: np.ndarray) -> dict:
        si, sp, st = u[k_sw][where], u[k_sw + 1][where], agree[k_sw + 2][where]
        hi, hp, ht = u[k_hw][where], u[k_hw + 1][where], agree[k_hw + 2][where]
        hit_sw, hit_hw = si != PT_NONE, hi != PT_NONE
        same_tri = (si == hi) & (sp == hp)
        both = same_tri & hit_sw
        rel = np.abs(ht[both] - st[both]) / np.maximum(np.abs(st[both]), 1e-30)
        # The rays that differ: the same point on two triangles (|dt| <= 1e-4 t: a shared
        # edge or coincident faces), a hit for one traversal only, or two distances.
        other = ~same_tri & hit_sw & hit_hw
        close = np.abs(ht - st) <= 1e-4 * np.maximum(np.abs(st), 1e-30)
        return {
            "rays": int(where.sum()),
            "hits_sw": int(hit_sw.sum()),
            "hits_hw": int(hit_hw.sum()),
            "same_triangle": float(np.mean(same_tri)) if where.any() else 1.0,
            "differ": int((~same_tri).sum()),
            "differ_same_point": int((other & close).sum()),
            "differ_hit_sw_only": int((hit_sw & ~hit_hw).sum()),
            "differ_hit_hw_only": int((hit_hw & ~hit_sw).sum()),
            "differ_elsewhere": int((other & ~close).sum()),
            "same_triangle_t_rel_diff_median": float(np.median(rel)) if rel.size else 0.0,
            "same_triangle_t_rel_diff_p99": float(np.quantile(rel, 0.99)) if rel.size else 0.0,
            "same_triangle_t_rel_diff_max": float(np.max(rel)) if rel.size else 0.0,
        }

    everyone = np.ones(agree.shape[1], dtype=bool)
    secondary = agree[14] == 1.0
    any_sw, any_hw = agree[12][secondary], agree[13][secondary]
    return {
        "camera": same(0, 3, everyone),
        "bounce": same(6, 9, secondary),
        "any_hit": {
            "rays": int(secondary.sum()),
            "occluded_sw": int((any_sw == 1.0).sum()),
            "occluded_hw": int((any_hw == 1.0).sum()),
            "same_verdict": float(np.mean(any_sw == any_hw)) if secondary.any() else 1.0,
        },
    }


def timings(runs: list[dict], meta: dict) -> dict:
    out: dict = {}
    for variant in ("sw", "hw"):
        timed = [r for r in runs if r["variant"] == variant and r["sample_offset"] == 0][1:]
        if not timed:
            continue
        keys = ("upload_ms", "as_build_ms", "trace_ms", "trace_gpu_ms", "center_ms",
                "readback_ms", "total_ms")  # fmt: skip
        out[variant] = {k: round(statistics.median(r[k] for r in timed), 2) for k in keys}
        out[variant]["runs"] = len(timed)
        out[variant]["trace_ms_all"] = [r["trace_ms"] for r in timed]
        out[variant]["compile_ms"] = timed[0]["compile_ms"]
        out[variant]["fallbacks"] = timed[0]["fallbacks"]
        out[variant]["rays"] = timed[0]["rays"]
    py = meta["python_warm"]
    st = py["render_scene_s"]
    out["python_warm"] = py
    if "sw" in out and "hw" in out:
        sw, hw = out["sw"], out["hw"]
        gpu_side_py = 1e3 * (st["upload"] + st["trace"] + st["center"] + st["readback"])
        rest = 1e3 * py["wall_s"] - gpu_side_py  # extraction, setup (BVH included), finishing
        bvh = 1e3 * st["bvh"]
        sw_e2e = rest + sw["total_ms"]
        hw_e2e = rest - bvh + hw["total_ms"]
        out["end_to_end_ms"] = {
            "python_measured": round(1e3 * py["wall_s"], 1),
            "sw_composed": round(sw_e2e, 1),
            "hw_composed": round(hw_e2e, 1),
            "cpu_side_ms": round(rest, 1),
            "of_which_bvh_ms": round(bvh, 1),
        }
        out["speedup"] = {
            "trace_wall": round(sw["trace_ms"] / hw["trace_ms"], 3),
            "trace_gpu": round(sw["trace_gpu_ms"] / hw["trace_gpu_ms"], 3),
            "end_to_end": round(sw_e2e / hw_e2e, 3),
        }
    return out


def main() -> None:
    d = Path(sys.argv[1])
    meta = json.loads((d / "meta.json").read_text())
    plan = {
        line.split()[0]: [int(v) for v in line.split()[1:]]
        for line in (d / "plan.txt").read_text().splitlines()
    }
    n_pix = plan["width"][0] * plan["height"][0]
    samples = plan["samples"][0]
    python = load(d / "planes_python.bin", n_pix)
    sw = load(d / "planes_sw.bin", n_pix)
    hw = load(d / "planes_hw.bin", n_pix)
    noise = load(d / "planes_sw_offset.bin", n_pix)
    eq = python.view(np.uint32) == sw.view(np.uint32)
    runs = [json.loads(line) for line in (d / "runs.jsonl").read_text().splitlines() if line]
    summary = {
        "scene": meta["scene"],
        "size": [plan["width"][0], plan["height"][0]],
        "samples": samples,
        "settings": meta["settings"],
        "triangles": meta["triangles"],
        "instances": meta["instances"],
        "meshes": meta["meshes"],
        "pass_through": meta["pass_through"],
        "cull": meta["cull"],
        "adapter": runs[0]["adapter"] if runs else meta["adapter"],
        "host_vs_python": {
            "values_bitwise_equal": float(eq.mean()),
            "planes_bitwise_equal": int(np.all(eq, axis=1).sum()),
            # (the instance plane holds ids as float bits: compared bitwise only)
            "max_abs_difference": float(
                np.max(np.abs(np.delete(python, ACC_INSTANCE, 0) - np.delete(sw, ACC_INSTANCE, 0)))
            ),
        },
        "image": image(sw, hw, noise, samples),
        "timings": timings(runs, meta),
    }
    if (d / "agree.bin").exists():
        summary["rays"] = rays(load(d / "agree.bin", n_pix))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
