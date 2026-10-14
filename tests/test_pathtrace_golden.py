"""A fixed path-traced image, reproduced on every CI platform.

``tests/data/renders/path_cornell.npy`` holds the linear image (float32) of a
small Cornell-box-like room rendered by Aether3D's path tracer with a fixed
seed. It is a regression reference (has anything changed?); correctness is
established by the analytic tests in ``test_pathtrace.py``.

On the platform that wrote it the render is bit-identical. Elsewhere, math
libraries may round transcendental functions differently in the last bit;
that can flip a rare path's decision (a ray grazing an edge), after which
that path's contribution is a different random sample. The comparison allows
for a few such pixels, never for a systematic change. Regenerate deliberately
after an intended change with::

    python tests/test_pathtrace_golden.py --update
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

from aether3d.rendering.pathtrace.sampling import path_keys, uniform

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "renders" / "path_cornell.npy"
OPTIONS = {"samples": 16, "bounces": 3, "seed": 11, "workers": 1}
REFERENCE_PLATFORM = {
    # Measured where the reference image was written (Linux x86_64, NumPy 2.5). A failure
    # reports this platform's values beside them, to show where a render starts to differ:
    # random numbers, the camera and geometry (pixel-center rays use no random numbers),
    # path decisions (ray counts), or only the light carried.
    "rng": [0.1470932985001353, 0.9794047947348427, 0.42258281119235785],
    "depth_sum": 2116.27490234375,
    "objects": [76, 70, 256, 80, 80, 10],
    "rays": {"camera_rays": 9216, "bounce_rays": 24115, "shadow_rays": 22952},
    "albedo": [0.6924957036972046, 0.6919260621070862, 0.6164466142654419],
}


def render():
    sys.path.insert(0, str(HERE))
    try:
        from test_pathtrace import cornell_box
    finally:
        sys.path.remove(str(HERE))
    scene = cornell_box()
    return scene.render(width=24, height=24, backend="path", path=OPTIONS, default_lights=False)


def fingerprint(result, image: np.ndarray, reference: np.ndarray) -> str:
    """This platform's fingerprint of the render beside the reference platform's."""
    depth = result.depth[np.isfinite(result.depth)]
    found = {
        "rng": uniform(path_keys(11, np.arange(3), np.zeros(3, np.int64)), 0).tolist(),
        "depth_sum": float(depth.sum()),
        "objects": np.bincount(result.object_index[result.object_index >= 0].ravel()).tolist(),
        "rays": {k: result.stats[k] for k in REFERENCE_PLATFORM["rays"]},
        "albedo": result.passes["albedo"].mean(axis=(0, 1)).tolist(),
    }
    lines = [f"{k}: {found[k]} (reference platform: {v})" for k, v in REFERENCE_PLATFORM.items()]
    mean, expected = image.mean(axis=(0, 1)).tolist(), reference.mean(axis=(0, 1)).tolist()
    lines.append(f"mean: {mean} (reference: {expected})")
    lines.append(f"NumPy {np.__version__}, {platform.platform()}, {platform.machine()}")
    return "\n".join(lines)


def test_the_fixed_path_traced_image_is_reproduced():
    result = render()
    reference = np.load(GOLDEN, allow_pickle=False)
    image = result.passes["beauty"]
    assert image.shape == reference.shape == (24, 24, 3)
    # Real rays met real geometry: camera, bounce and shadow rays, the room's objects.
    assert result.stats["bounce_rays"] > 0 and result.stats["shadow_rays"] > 0
    assert {o["name"] for o in result.objects if o["visible_samples"]} >= {
        "Floor",
        "Back",
        "Left",
        "Right",
        "Lamp",
    }
    diff = np.abs(image.astype(np.float64) - reference)
    close = diff <= 1e-5 * np.maximum(np.abs(reference), 1e-3)
    assert close.mean() > 0.97, (
        f"{100 * (1 - close.mean()):.1f}% of values changed\n"
        + fingerprint(result, image, reference)
    )
    assert diff.mean() < 0.005 * reference.mean(), fingerprint(result, image, reference)


if __name__ == "__main__":
    if "--update" not in sys.argv:
        print(__doc__)
        sys.exit(2)
    image = render().passes["beauty"]
    np.save(GOLDEN, image.astype(np.float32), allow_pickle=False)
    print(f"wrote {GOLDEN} {image.shape}, mean {image.mean():.5f}")
