"""Rendering regression tests: today's renders against stored reference PNGs.

The references were rendered by Aether3D (see ``render_goldens.py``); these
tests catch unintended changes, while correctness is established analytically
and against the independent ray caster. On the platform that produced them the
match is exact. Across platforms, floating-point details (e.g. the last bit of
``tan``/``pow`` in a C library) may flip a sample on a silhouette, so up to
0.5% of pixels may differ by more than 2 codes, and the mean difference must
stay below 0.25 codes.
"""

import io

import numpy as np
import pytest
from render_goldens import GOLDEN_DIR, GOLDENS, render_golden

PIL = pytest.importorskip("PIL.Image")

try:
    import manifold3d  # noqa: F401

    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_render_matches_reference(name):
    if GOLDENS[name][2] and not HAS_MANIFOLD:
        pytest.skip("needs manifold3d")
    expected = np.asarray(PIL.open(GOLDEN_DIR / f"{name}.png").convert("RGB")).astype(int)
    result = render_golden(name)
    actual = np.asarray(PIL.open(io.BytesIO(result.png_bytes()))).astype(int)
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected).max(axis=-1)
    assert np.mean(diff > 2) <= 0.005, f"{np.count_nonzero(diff > 2)} pixels differ"
    assert diff.mean() < 0.25
