import json, sys
import numpy as np
from model import model_none, model_cont

for path in ("blender_curves_520.json", "blender_curves_b_520.json"):
    cases = json.load(open(path))["cases"]
    for name, c in cases.items():
        if "keys" not in c or c.get("cyclic"):
            continue
        ks = c["keys"]
        types = {k["left_type"] for k in ks}
        if (
            not types <= {"AUTO", "AUTO_CLAMPED"}
            or len(types) != 1
            or any(k["interpolation"] != "BEZIER" for k in ks)
        ):
            continue
        t = [k["co"][0] for k in ks]
        v = [k["co"][1] for k in ks]
        clamped = types == {"AUTO_CLAMPED"}
        linear = c["extrapolation"] == "LINEAR"
        bl = np.array([[k["left"][0], k["left"][1], k["right"][0], k["right"][1]] for k in ks])
        res = []
        for label, fn in (
            ("none", lambda: model_none(t, v, clamped, linear)),
            ("cont_all", lambda: model_cont(t, v, clamped, linear, "all")),
            ("cont_worst", lambda: model_cont(t, v, clamped, linear, "worst")),
        ):
            m, le, ri = fn()
            ours = np.stack(
                [np.array(t) - le, np.array(v) - m * le, np.array(t) + ri, np.array(v) + m * ri], 1
            )
            res.append((label, float(np.abs(ours - bl).max())))
        want = "none" if c["auto_smoothing"] == "NONE" else "cont"
        best = {l: e for l, e in res}
        if want == "none":
            print(f"{name:40s} NONE  model err {best['none']:.2e}")
        else:
            print(
                f"{name:40s} CONT  all {best['cont_all']:.2e}  worst-first {best['cont_worst']:.2e}"
            )
