"""Showcase C -- graph curves against Blender's: the same keys, measured both ways.

Blender 5.2 evaluated 49 F-curve cases and 17 modifier cases headless (experiment E1,
``research/2026-10-stage19/blender/e1_curves``; the fixture is
``tests/data/blender_curves/curves.json``) and four NLA blends of a 170-degree turn (E2).
This script rebuilds every case with Aether3D's own curves and writes:

* ``cases.png`` -- four cases (Blender 5.2's default new curve: auto-clamped handles with
  continuous acceleration; Blender's older auto smoothing; free handles mixed in; a cyclic
  curve with uneven key spacing) graphed by Aether3D -- value, velocity, acceleration and
  jerk -- with Blender's own evaluation drawn as crosses on the value plot, and under each
  the difference Aether3D minus Blender in millionths;
* ``smoothing.png`` -- one set of keys with automatic handles under Aether3D's three
  smoothing modes (``local``, ``blender``, ``continuous``): the acceleration plot shows
  the steps (``C2`` marks) the first two leave at every key and the third removes;
* ``quaternions.png`` -- the turn's angle through a crossfade: Blender's REPLACE (it adds
  quaternion components, so the angle moves unevenly), Blender's COMBINE and Aether3D's
  replace layer (successive slerp: even);
* ``report.json`` -- the largest handle and value difference of every case, and the
  angles.

    .venv/bin/python examples/film_c_graphs.py [--quick]
"""

from __future__ import annotations

import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from film_common import OUT, arguments, rel, write_report  # noqa: E402

from aether3d import Scene  # noqa: E402
from aether3d.animation.curves import FCurve  # noqa: E402
from aether3d.animation.graph import graph_png, graph_svg  # noqa: E402
from aether3d.animation.report import continuity  # noqa: E402
from aether3d.rendering.png import write_png  # noqa: E402

ROOT = HERE.parent
FIXTURE = ROOT / "tests" / "data" / "blender_curves" / "curves.json"
NLA = ROOT / "research" / "2026-10-stage19" / "blender" / "e2_nla" / "blender_nla_520.json.gz"
PANELS = (
    ("AUTO_CLAMPED_CONT_ACCEL_CONSTANT",
     "auto clamped, continuous acceleration (Blender's new-curve default)"),
    ("AUTO_NONE_CONSTANT", "auto, Blender's older smoothing (None)"),
    ("mixed_free_CONT_ACCEL", "free handles among automatic ones"),
    ("loop_unequal_AUTO_CLAMPED_CONT_ACCEL", "cyclic, uneven key spacing"),
)  # fmt: skip
FPS = 24.0  # Blender's frames (the NLA experiment ran at its default rate)


def build(case: dict) -> FCurve:
    """A fixture case as an Aether3D F-curve: its keys, interpolations, handle types (a
    free handle keeps its own extent: weighted) and extrapolation."""
    rows = []
    for i, (t, value) in enumerate(case["keys"]):
        row = {"time": t, "value": value, "interpolation": case["interpolation"][i]}
        for side in ("left", "right"):
            kind = case[f"{side}_type"][i]
            if kind in ("free", "aligned"):
                hx, hy = case[side][i]
                row[side] = {"type": kind, "dt": hx - t, "dv": hy - value}
                row["weighted"] = True
            else:
                row[side] = kind
        rows.append(row)
    return FCurve.from_keys(rows, extrapolation=case["extrapolation"], smoothing=case["smoothing"])


def differences(case: dict, curve: FCurve) -> dict:
    a = curve.arrays()
    ours = np.stack([a["times"] + a["left_dt"], a["values"] + a["left_dv"],
                     a["times"] + a["right_dt"], a["values"] + a["right_dv"]], 1)  # fmt: skip
    ref = np.hstack([np.array(case["left"]), np.array(case["right"])])
    if case["extrapolation"] == "cycle":  # (the ends' outer extents never shape the curve)
        ours[0, :2] = ref[0, :2]
        ours[-1, 2:] = ref[-1, 2:]
    frames, values = np.array(case["frames"]), np.array(case["values"])
    diff = curve.evaluate(frames) - values
    return {
        "handles": float(np.abs(ours - ref).max()),
        "values": float(np.abs(diff).max()),
        "scale": max(1.0, float(np.abs(ref[:, [1, 3]]).max())),
        "diff": diff,
    }


def sampled(times: np.ndarray, values: np.ndarray) -> FCurve:
    """Samples as a curve (linear between them) -- for graphing measured series."""
    return FCurve.from_keys([{"time": float(t), "value": float(v), "interpolation": "linear"}
                             for t, v in zip(times, values, strict=True)])  # fmt: skip


def stack(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    w = max(im.shape[1] for im in images)
    rows = []
    for im in images:
        pad = np.full((im.shape[0], w - im.shape[1], 3), 255, np.uint8)
        rows += [np.concatenate([im, pad], axis=1), np.full((gap, w, 3), 255, np.uint8)]
    return np.concatenate(rows[:-1], axis=0)


def angle_deg(q: np.ndarray) -> np.ndarray:
    """The rotation angle of (possibly unnormalized) quaternions ``(x, y, z, w)``."""
    v = np.linalg.norm(q[:, :3], axis=1)
    return np.degrees(2.0 * np.arctan2(v, np.abs(q[:, 3])))


def quaternion_blend() -> dict:
    """Blender's E2 setup in Aether3D: a strip holding 0 degrees, a strip above it holding
    170 degrees about Z from frame 90 with a 20-frame linear blend in."""
    scene = Scene()
    obj = scene.add_object("O")
    a = scene.animation
    q0 = [0.0, 0.0, 0.0, 1.0]
    h = math.radians(170.0) / 2.0
    q1 = [0.0, 0.0, math.sin(h), math.cos(h)]
    for name, q in (("QA", q0), ("QB", q1)):
        clip = a.add_clip(name)
        clip.set_channel("transform.rotation", [(0.0, q), (59.0 / FPS, q)])
    a.add_layer(obj, "Lower")
    a.add_strip(obj, "Lower", "QA", start=70.0 / FPS, after="hold")
    a.add_layer(obj, "Upper")
    a.add_strip(obj, "Upper", "QB", start=90.0 / FPS, blend_in=20.0 / FPS, after="hold")
    frames = np.arange(85.0, 116.0, 0.5)
    ours = np.array([a.evaluate(float(f) / FPS).rotation[0] for f in frames])
    with gzip.open(NLA) as f:
        blender = json.load(f)
    out = {"frames": frames, "Aether3D": angle_deg(ours)}
    for mode in ("REPLACE", "COMBINE"):
        rows = np.array(blender[mode])
        keep = (rows[:, 0] >= frames[0]) & (rows[:, 0] <= frames[-1])
        q = rows[keep][:, [3, 4, 5, 2]]  # Blender stores (w, x, y, z)
        out[f"Blender {mode}"] = (rows[keep][:, 0], angle_deg(q))
        out[f"Blender {mode} norm"] = np.linalg.norm(q, axis=1)
    return out


def main() -> dict:
    args = arguments(__doc__)
    out = args.out or OUT / "c_graphs"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data = json.loads(FIXTURE.read_text())
    report: dict = {"showcase": "C graph curves against Blender", "blender": data["blender"],
                    "fixture": rel(FIXTURE), "quick": args.quick}  # fmt: skip
    worst = {"handles": 0.0, "values": 0.0}
    cases = {}
    for name, case in data["cases"].items():
        d = differences(case, build(case))
        cases[name] = {k: d[k] for k in ("handles", "values", "scale")}
        for k in worst:
            worst[k] = max(worst[k], d[k] / d["scale"])
    report["cases"] = cases
    report["largest_relative_difference"] = worst
    report["agreement"] = ("Blender stores float32: every handle and value agrees within "
                           "2e-6 of the case's scale") if max(worst.values()) <= 2e-6 else (
                           "differences above float32 rounding: see cases")  # fmt: skip
    panels = []
    for name, label in PANELS:
        case = data["cases"][name]
        curve = build(case)
        d = differences(case, curve)
        frames = np.array(case["frames"])
        refs = {"Blender 5.2": (frames, np.array(case["values"]))}
        g = graph_png({"Aether3D": curve}, references=refs, title=f"{name}: {label}",
                      plots=("value", "velocity", "acceleration", "jerk"), width=1100,
                      plot_height=150)  # fmt: skip
        diff = graph_png({"Aether3D - Blender (x 1e6)": sampled(frames, d["diff"] * 1e6)},
                         plots=("value",), width=1100, plot_height=90,
                         annotate=False)  # fmt: skip
        panels += [g, diff]
        (out / f"{name}.svg").write_text(graph_svg({"Aether3D": curve}, references=refs,
                                                   title=name, plots=("value", "velocity",
                                                   "acceleration", "jerk")))  # fmt: skip
        if args.quick:
            break
    write_png(out / "cases.png", stack(panels))
    report["cases_png"] = rel(out / "cases.png")
    # The three smoothing modes on one set of keys (automatic handles: clamped ones are
    # flat at extrema, which no smoothing mode makes continuous in acceleration).
    base = data["cases"]["AUTO_CONT_ACCEL_CONSTANT"]
    modes = {}
    steps = {}
    for mode in ("local", "blender", "continuous"):
        curve = build({**base, "smoothing": mode})
        modes[mode] = curve
        steps[mode] = sum(1 for c in continuity(curve)
                          if c["order"] == 2 and not c.get("intended"))  # fmt: skip
    title = "automatic handles, three smoothing modes (C2: a step in acceleration)"
    image = graph_png(modes, plots=("value", "acceleration", "jerk"), width=1100,
                      plot_height=170, title=title)  # fmt: skip
    write_png(out / "smoothing.png", image)
    report["smoothing"] = {"acceleration_steps": steps, "png": rel(out / "smoothing.png")}
    # Quaternion crossfades.
    q = quaternion_blend()
    frames = q["frames"]
    refs = {k: v for k, v in q.items() if k.startswith("Blender") and not k.endswith("norm")}
    title = "a 170-degree turn crossfaded in over 20 frames: angle (degrees) by frame"
    write_png(out / "quaternions.png", graph_png(
        {"Aether3D (slerp)": sampled(frames, q["Aether3D"])}, references=refs,
        plots=("value", "velocity"), width=1100, plot_height=200, title=title))  # fmt: skip
    mid = len(frames) // 2
    report["quaternions"] = {
        "png": rel(out / "quaternions.png"),
        "angles_at_quarter_half_three_quarters": {
            "Aether3D": [round(float(np.interp(f, frames, q["Aether3D"])), 2)
                         for f in (95.0, 100.0, 105.0)],
            **{k: [round(float(np.interp(f, v[0], v[1])), 2) for f in (95.0, 100.0, 105.0)]
               for k, v in refs.items()},
        },
        "blender_replace_norm_midway": round(float(np.interp(100.0, refs["Blender REPLACE"][0],
                                                             q["Blender REPLACE norm"])), 3),
        "note": "Blender's REPLACE adds quaternion components (unnormalized midway); "
        "Aether3D slerps, so a linear ramp turns at an even rate",
        "samples": int(frames.size),
        "middle_frame": float(frames[mid]),
    }  # fmt: skip
    report["seconds"] = round(time.perf_counter() - t0, 2)
    write_report(out / "report.json", report)
    return report


if __name__ == "__main__":
    data = main()
    print(
        f"C graphs: {data['seconds']} s, largest difference {data['largest_relative_difference']}"
    )
