"""E1: Aether3D (v0.19.1 baseline or later) against Blender's F-curve data.

Run with the repository's Python:  python compare_aether.py blender_curves_520.json OUT.json
Frames are used as Aether3D seconds directly (both systems are affine in time).
"""

import json
import sys

import numpy as np

from aether3d.animation.curves import FCurve

data = json.load(open(sys.argv[1]))
cases = data["cases"]
EASE = {"EASE_IN": "in", "EASE_OUT": "out", "EASE_IN_OUT": "in_out"}
results = {}


def handles_of(curve):
    a = curve.arrays()
    t, v = a["times"], a["values"]
    return [
        [
            [float(t[k] + a["left_dt"][k]), float(v[k] + a["left_dv"][k])],
            [float(t[k] + a["right_dt"][k]), float(v[k] + a["right_dv"][k])],
        ]
        for k in range(len(t))
    ]


def record(name, curve, case, note=""):
    grid = np.array(case["grid"])
    ours = curve.evaluate(grid)
    ref = np.array(case["values"])
    err = np.abs(ours - ref)
    out = {
        "max_abs": float(err.max()),
        "rms": float(np.sqrt((err**2).mean())),
        "range": float(ref.max() - ref.min()),
        "note": note,
    }
    if "keys" in case and curve is not None:
        bh = [[k["left"], k["right"]] for k in case["keys"]]
        ah = handles_of(curve)
        out["handle_max_abs"] = float(
            max(abs(x - y) for p, q in zip(bh, ah) for s, r in zip(p, q) for x, y in zip(s, r))
        )
    results[name] = out


for name, case in cases.items():
    if name == "insert_defaults":
        continue
    keys = case["keys"]
    frames = [k["co"][0] for k in keys]
    values = [k["co"][1] for k in keys]
    extrap = case["extrapolation"].lower()
    try:
        if name.split("_")[0] in ("AUTO", "VECTOR") or name.startswith(("scaled10", "timescaled2")):
            htype = keys[0]["left_type"].lower()
            c = FCurve.from_arrays(frames, values, handles=htype, extrapolation=extrap)
            record(name, c, case, f"aether {htype} vs blender {case['auto_smoothing']}")
        elif name in ("long_free", "free_asym"):
            rows = []
            for k in keys:
                t0, v0 = k["co"]
                rows.append(
                    {
                        "time": t0,
                        "value": v0,
                        "weighted": True,
                        "left": [k["left"][0] - t0, k["left"][1] - v0],
                        "right": [k["right"][0] - t0, k["right"][1] - v0],
                    }
                )
            c = FCurve.from_keys(rows, extrapolation=extrap)
            record(name, c, case, "free handles, weighted")
        elif name.startswith("ease_") and not name.endswith("AUTO"):
            interp = keys[0]["interpolation"].lower()
            rows = []
            for k in keys:
                rows.append(
                    {
                        "time": k["co"][0],
                        "value": k["co"][1],
                        "interpolation": interp,
                        "ease": EASE[k["easing"]],
                        "back": k["back"],
                        "amplitude": max(1.0, k["amplitude"]),
                        "period": min(10.0, max(1e-3, k["period"])),
                    }
                )
            c = FCurve.from_keys(rows, extrapolation=extrap)
            record(name, c, case, f"amp {keys[0]['amplitude']} period {keys[0]['period']}")
        elif name.startswith("cycles_"):
            _, mb, ma = name.split("_", 2)
            # Aether3D: extrapolation modes (no counts); compare inside the 2 cycles after.
            emap = {
                "REPEAT": "cycle",
                "REPEAT_OFFSET": "cycle_offset",
                "MIRROR": "mirror",
                "NONE": "constant",
            }
            if mb.startswith("REPEAT_OFFSET") or ma.startswith("REPEAT_OFFSET"):
                parts = name[len("cycles_") :]
                mb, ma = (
                    ("REPEAT_OFFSET", parts[len("REPEAT_OFFSET_") :])
                    if parts.startswith("REPEAT_OFFSET")
                    else (parts.split("_", 1)[0], parts.split("_", 1)[1])
                )
            c = FCurve.from_arrays(
                frames, values, handles="auto_clamped", extrapolation=[emap[mb], emap[ma]]
            )
            g = np.array(case["grid"])
            span = frames[-1] - frames[0]
            inside = (g >= frames[0]) & (g <= frames[-1] + 2 * span)
            ours = c.evaluate(g)
            ref = np.array(case["values"])
            results[name] = {
                "max_abs_within_counts": float(np.abs(ours - ref)[inside].max()),
                "note": f"extrapolation {emap[mb]}/{emap[ma]} (no counts)",
            }
        else:
            results[name] = {"note": "no direct Aether3D equivalent (baseline)"}
    except Exception as exc:  # noqa: BLE001
        results[name] = {"error": f"{type(exc).__name__}: {exc}"}

json.dump(results, open(sys.argv[2], "w"), indent=1)
for k, v in results.items():
    print(f"{k:42s} {json.dumps(v)}")
