"""A poorly composed, underexposed view, corrected through the camera and the exposure
(Stage 12).

Run from the repository root:

    python examples/feedback_composition.py [OUTPUT_DIR] [--backend auto|gpu|cpu] [--quick]

A small garden with a monument. The camera of the first draft is aimed past the monument
(it sits near the image's left border) and stands too far away (it covers under 1 % of the
image), and the exposure is set three stops too low: a dark image.

The expectations state the composition and the light as measurable rules -- the monument
fully in frame, its visible center within the middle region, covering 6-35 % of the image,
the mean luminance of the objects within 0.12-0.45 in linear light, at most 3 % clipped --
and the feedback loop corrects them with the camera's own operations (aim, dolly, frame)
and the exposure, each step verified on a new render. The measurements before and after
come from the renders themselves, not from the corrections' predictions.

Writes ``feedback_composition.png`` (before and after) and ``feedback_composition.json``
to OUTPUT_DIR (default ``examples/output``). Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import FeedbackSession, Scene, __version__
from aether3d.feedback import overlay, run_loop
from aether3d.rendering.png import encode_png

HERE = Path(__file__).resolve().parent

PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "label": "Garden (first draft)",
    "materials": {
        "grass": {"preset": "grass"},
        "stone": {"preset": "stone", "color": "#a39e93"},
        "bronze": {"preset": "metal", "color": "#8c6a3f"},
        "hedge": {"base_color": "#2f5d32", "roughness": 0.95},
        "gravel": {"base_color": "#c8b89a", "roughness": 0.9},
    },
    "objects": {
        "ground": {
            "name": "Ground",
            "mesh": {"primitive": "plane", "size": 40},
            "material": "@grass",
        },
        "path": {
            "name": "Path",
            "mesh": {"primitive": "box", "size": [2, 16, 0.02]},
            "material": "@gravel",
            "position": [0, -6, 0.01],
        },
        "monument": {
            "name": "Monument",
            "children": {
                "plinth": {
                    "mesh": {"primitive": "box", "size": [1.6, 1.6, 1.2]},
                    "material": "@stone",
                    "position": [0, 0, 0.6],
                },
                "column": {
                    "mesh": {"primitive": "cylinder", "radius": 0.35, "depth": 1.8, "segments": 24},
                    "material": "@stone",
                    "position": [0, 0, 2.1],
                },
                "orb": {
                    "mesh": {"primitive": "sphere", "radius": 0.55},
                    "material": "@bronze",
                    "position": [0, 0, 3.55],
                },
            },
        },
        "hedges": {
            "name": "Hedge {i}",
            "mesh": {"primitive": "box", "size": [1, 1, 1]},
            "material": "@hedge",
            "scale": [1.2, 1.2, 1.0],
            "circle": {"count": 8, "radius": 6, "z": 0.5},
        },
        "sun": {
            "name": "Sun",
            "light": {"type": "directional", "intensity": 1.4},
            "direction": [-0.3, 0.7, -0.6],
        },
        "camera": {
            "name": "Camera",
            "camera": {"fov": 40},
            "position": [9, -30, 6],
            "look_at": [9.5, 0, 2],
            "active": True,
        },
    },
    "world": {"background": "#a9c8e6", "ambient_intensity": 0.05},
    "render_settings": {"exposure": -3.0},
}

EXPECTATIONS = {
    "format": "aether3d.expectations",
    "version": 1,
    "label": "Garden view",
    "views": {"main": {"camera": "Camera"}},
    "checks": [
        {"id": "monument-seen", "check": "visible", "objects": "Monument", "in_frame": True},
        {
            "id": "composition",
            "check": "framing",
            "objects": "Monument",
            "region": [0.3, 0.2, 0.7, 0.8],
            "fill": {"min": 0.06, "max": 0.35},
        },
        {
            "id": "light",
            "check": "exposure",
            "mean_luminance": {"min": 0.12, "max": 0.45},
            "max_clipped_fraction": 0.03,
        },
    ],
}


def measure(feedback: FeedbackSession) -> dict:
    """The composition and light, measured on a fresh render (not predicted)."""
    obs = feedback.observe("main", linear=True, focus="Monument")
    record = obs.record("Monument", geometry=False)
    w, h = obs.width, obs.height
    return {
        "observation": obs.id,
        "monument_center": [record["center"][0] / w, record["center"][1] / h]
        if "center" in record
        else None,
        "monument_fill": record["visible_pixels"] / (w * h),
        "touches_border": record.get("touches_border"),
        "mean_luminance": obs.image_stats["luminance_covered"]["mean"],
        "clipped_fraction": obs.image_stats["clipped_covered_fraction"],
        "image": overlay.annotate(obs.image, obs.index, obs._marker_boxes()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="smaller images (CI)")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (480, 270) if args.quick else (960, 540)

    scene = Scene("Garden")
    scene.execute_plan(PLAN)
    # The first draft is far off: the camera must move over 20 m, beyond the default limit
    # (a quarter of the scene's size), which would leave that correction for approval.
    # This scene's policy allows larger moves and exposure changes.
    feedback = FeedbackSession(
        scene,
        EXPECTATIONS,
        backend=args.backend,
        width=width,
        height=height,
        policy={"auto_apply": ["safe", "verify"], "max_translation": 40, "max_exposure_change": 6},
    )
    before = measure(feedback)
    diagnosis = feedback.diagnose()
    found = [
        (f.code, f.basis, f.magnitude, f.unit) for f in diagnosis.findings if f.severity == "error"
    ]
    t0 = time.perf_counter()
    loop = run_loop(feedback, budget={"max_iterations": 8})
    seconds = time.perf_counter() - t0
    after = measure(feedback)

    failures = []
    if loop.stop_reason != "accepted":
        failures.append(f"the loop stopped with {loop.stop_reason}")
    cx, cy = after["monument_center"]
    if not (0.3 <= cx <= 0.7 and 0.2 <= cy <= 0.8):
        failures.append(f"the monument's center is at {cx:.3f}, {cy:.3f}")
    if not 0.06 <= after["monument_fill"] <= 0.35:
        failures.append(f"the monument covers {after['monument_fill']:.3f} of the image")
    if not 0.12 <= after["mean_luminance"] <= 0.45:
        failures.append(f"the mean luminance is {after['mean_luminance']:.4f}")
    if any(after["touches_border"].values()):
        failures.append("the monument touches the image border")
    codes = {code for code, *_ in found}
    for code in ("OFF_CENTER", "TOO_SMALL_IN_FRAME", "UNDEREXPOSED"):
        if code not in codes:
            failures.append(f"the first draft should show {code}")

    image = np.full((height, 2 * width + 8, 3), 255, dtype=np.uint8)
    image[:, :width] = before.pop("image")
    image[:, width + 8 :] = after.pop("image")
    (out / "feedback_composition.png").write_bytes(
        encode_png(image, text={"Software": f"Aether3D {__version__}"})
    )
    report = {
        "aether3d": __version__,
        "renderer": feedback.observe("main").renderer,
        "before": before,
        "findings": [{"code": c, "basis": b, "magnitude": m, "unit": u} for c, b, m, u in found],
        "loop": {
            "stop_reason": loop.stop_reason,
            "seconds": seconds,
            "iterations": [
                {
                    "iteration": it["iteration"],
                    "errors": it["diagnosis"]["errors"],
                    "applied": [
                        p["title"]
                        for p in it.get("proposals", [])
                        if p["id"] in ((it.get("applied") or {}).get("applied") or [])
                    ],
                    "decision": ((it.get("applied") or {}).get("verification") or {}).get(
                        "decision"
                    ),
                }
                for it in loop.iterations
            ],
            "budget_used": loop.budget["used"],
        },
        "after": after,
        "camera": {
            "position": list(scene.get_by_name("Camera").world_position),
            "exposure": scene.render_settings.exposure,
        },
        "failures": failures,
    }
    (out / "feedback_composition.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        f"before: center {before['monument_center'][0]:.3f}, {before['monument_center'][1]:.3f}"
        f"; fill {before['monument_fill']:.4f}; mean luminance {before['mean_luminance']:.4f}"
    )
    for it in report["loop"]["iterations"]:
        print(
            f"  iteration {it['iteration']}: {it['errors']} error(s); {it['applied']} "
            f"({it['decision']})"
        )
    print(
        f"after: center {cx:.3f}, {cy:.3f}; fill {after['monument_fill']:.4f}; mean luminance "
        f"{after['mean_luminance']:.4f} ({loop.stop_reason}, {seconds:.2f} s)"
    )
    for failure in failures:
        print(f"CHECK FAILED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
