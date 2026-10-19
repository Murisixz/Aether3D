"""E1: Aether3D's cycles, generator, function, stepped, limits and ranged modifiers against
Blender's (the E1 modifier cases: keys at frames 1, 9, 17; Blender's default smoothing)."""

import json
import sys

import numpy as np

from aether3d.animation.curves import FCurve

cases = json.load(open(sys.argv[1]))["cases"]
MODE = {"REPEAT": "repeat", "REPEAT_OFFSET": "repeat_offset", "MIRROR": "mirror", "NONE": "none"}
for name, c in cases.items():
    if not c.get("modifier"):
        continue
    ks = c["keys"]
    t = [k["co"][0] for k in ks]
    v = [k["co"][1] for k in ks]
    if name.startswith("cycles_"):
        rest = name[len("cycles_") :]
        for mb in ("REPEAT_OFFSET", "REPEAT", "MIRROR", "NONE"):
            if rest.startswith(mb + "_"):
                ma = rest[len(mb) + 1 :]
                break
        mod = {"type": "cycles", "before": MODE[mb], "after": MODE[ma], "count_after": 2}
    elif name == "stepped":
        mod = {"type": "stepped", "step": 3.0, "offset": 1.0}
    elif name == "generator_additive":
        mod = {"type": "generator", "coefficients": [0.5, 0.1, -0.01]}
    elif name == "fn_sin_additive":
        mod = {
            "type": "function",
            "function": "sin",
            "amplitude": 0.5,
            "phase_multiplier": 0.4,
            "phase_offset": 0.25,
            "value_offset": 0.1,
        }
    elif name == "limits":
        mod = {"type": "limits", "min": 0.2, "max": 1.5}
    elif name == "generator_ranged":
        mod = {
            "type": "generator",
            "coefficients": [1.0, 0.0],
            "range": [5.0, 13.0],
            "blend_in": 2.0,
            "blend_out": 3.0,
        }
    else:
        print(f"{name:34s} (no equivalent: Blender's noise is its own algorithm)")
        continue
    cur = FCurve.from_arrays(t, v, handles="auto_clamped", smoothing="continuous", modifiers=[mod])
    g = np.array(c["grid"])
    ref = np.array(c["values"])
    err = np.abs(cur.evaluate(g) - ref)
    print(f"{name:34s} max {err.max():.2e} at frame {g[err.argmax()]:.2f}")
