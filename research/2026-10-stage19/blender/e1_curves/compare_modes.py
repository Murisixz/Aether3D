"""E1: the Blender-compatible smoothing modes against Blender's handles and values."""

import json
import sys

import numpy as np

from aether3d.animation.curves import FCurve

EASE = {"EASE_IN": "in", "EASE_OUT": "out", "EASE_IN_OUT": "in_out"}
worst = {}
for path in sys.argv[1:]:
    cases = json.load(open(path))["cases"]
    for name, c in cases.items():
        if "keys" not in c:
            continue
        ks = c["keys"]
        if any(k["interpolation"] not in ("BEZIER", "LINEAR", "CONSTANT") for k in ks):
            continue
        if c.get("modifier") and not name.startswith("cycles_"):
            continue
        mode = "continuous" if c["auto_smoothing"] == "CONT_ACCEL" else "blender"
        extrap = c["extrapolation"].lower()
        if c.get("cyclic"):
            extrap = "cycle"
        rows = []
        for k in ks:
            t0, v0 = k["co"]
            row = {"time": t0, "value": v0, "interpolation": k["interpolation"].lower()}
            for side in ("left", "right"):
                typ = k[f"{side}_type"].lower()
                if typ in ("free", "aligned"):
                    row[side] = {"type": typ, "dt": k[side][0] - t0, "dv": k[side][1] - v0}
                    row["weighted"] = True
                else:
                    row[side] = typ
            rows.append(row)
        cur = FCurve.from_keys(rows, extrapolation=extrap, smoothing=mode)
        a = cur.arrays()
        ours = np.stack(
            [
                a["times"] + a["left_dt"],
                a["values"] + a["left_dv"],
                a["times"] + a["right_dt"],
                a["values"] + a["right_dv"],
            ],
            1,
        )
        ref = np.array([[k["left"][0], k["left"][1], k["right"][0], k["right"][1]] for k in ks])
        scale = max(1.0, float(np.abs(ref[:, [1, 3]]).max()))
        # Handles that matter: interior ones, and end handles unless the curve is cyclic.
        herr = np.abs(ours - ref)
        if c.get("cyclic"):
            herr[0, :2] = 0.0
            herr[-1, 2:] = 0.0
        g = np.array(c["grid"])
        inside = (g >= ks[0]["co"][0]) & (g <= ks[-1]["co"][0])
        verr = np.abs(cur.evaluate(g[inside]) - np.array(c["values"])[inside]).max()
        worst[name] = (herr.max() / scale, verr / scale)
        print(f"{name:42s} {mode:10s} handles {herr.max() / scale:.1e}  values {verr / scale:.1e}")
bad = {k: v for k, v in worst.items() if max(v) > 1e-5}
print("cases:", len(worst), "over 1e-5 (relative):", bad)
