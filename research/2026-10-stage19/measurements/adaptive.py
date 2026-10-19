"""Adaptive sampling against uniform sampling on the test scene, at equal mean samples
per pixel and at equal time (the CPU path tracer; tests/test_denoise.two_walls).

    .venv/bin/python research/2026-10-stage19/measurements/adaptive.py OUT.json [W H REF]

Errors are the relative RMSE of linear luminance against a Sobol' reference of ``REF``
samples (another seed). Each adaptive render is compared with a uniform render of as many
samples as its mean, and with one given its wall time.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
from test_denoise import two_walls  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def main() -> None:
    out_path = sys.argv[1]
    w, h, ref_spp = (int(a) for a in sys.argv[2:5]) if len(sys.argv) > 4 else (96, 72, 4096)
    scene = two_walls()

    def run(path):
        t0 = time.perf_counter()
        r = scene.render(width=w, height=h, backend="path", path={"bounces": 3, **path})
        return r, time.perf_counter() - t0

    run({"samples": 1})  # (warm-up: worker processes, caches)
    ref, tref = run({"samples": ref_spp, "sampler": "sobol", "seed": 1})
    yr = ref.linear @ LUMA

    def error(r):
        return float(np.sqrt(np.mean((r.linear @ LUMA - yr) ** 2)) / yr.mean())

    out = {"size": [w, h], "reference_spp": ref_spp, "reference_seconds": round(tref, 1),
           "rows": []}  # fmt: skip
    for threshold in (0.02, 0.04):
        for cap in (64, 128):
            a, ta = run({"samples": cap, "sampler": "sobol", "adaptive_threshold": threshold})
            mean = float(a.passes["sample_count"].mean())
            b, tb = run({"samples": max(1, round(mean)), "sampler": "sobol"})
            # Equal time: uniform samples scaled by the measured time ratio.
            n_t = max(1, round(round(mean) * ta / tb))
            c, tc = run({"samples": n_t, "sampler": "sobol"})
            row = {"threshold": threshold, "cap": cap, "samples_mean": round(mean, 2),
                   "adaptive": {"rel_rmse": error(a), "seconds": round(ta, 2)},
                   "equal_samples": {"spp": round(mean), "rel_rmse": error(b),
                                     "seconds": round(tb, 2)},
                   "equal_time": {"spp": n_t, "rel_rmse": error(c), "seconds": round(tc, 2)}}  # fmt: skip
            row["less_error_equal_samples"] = 1 - row["adaptive"]["rel_rmse"] / error(b)
            row["less_error_equal_time"] = 1 - row["adaptive"]["rel_rmse"] / error(c)
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
