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

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "renders" / "path_cornell.npy"
OPTIONS = {"samples": 16, "bounces": 3, "seed": 11, "workers": 1}


def render():
    sys.path.insert(0, str(HERE))
    try:
        from test_pathtrace import cornell_box
    finally:
        sys.path.remove(str(HERE))
    scene = cornell_box()
    return scene.render(width=24, height=24, backend="path", path=OPTIONS, default_lights=False)


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
    assert close.mean() > 0.97, f"{100 * (1 - close.mean()):.1f}% of values changed"
    assert diff.mean() < 0.005 * reference.mean()


if __name__ == "__main__":
    if "--update" not in sys.argv:
        print(__doc__)
        sys.exit(2)
    image = render().passes["beauty"]
    np.save(GOLDEN, image.astype(np.float32), allow_pickle=False)
    print(f"wrote {GOLDEN} {image.shape}, mean {image.mean():.5f}")
