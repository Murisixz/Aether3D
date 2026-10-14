"""Render backends: names, the GPU antialiasing plan and option validation.

* ``"cpu"`` (default) -- Aether3D's NumPy rasterizer (``rendering/cpu.py``):
  always available, deterministic across platforms, no shadows.
* ``"gpu"`` -- the wgpu backend (``rendering/gpu/``): real GPU rasterization
  through Vulkan, Metal or Direct3D 12, with MSAA and shadow maps. Needs the
  optional ``wgpu`` package and a working adapter; if either is missing, the
  render fails with an actionable ``GPU_*`` error (it never silently runs on
  the CPU instead).
* ``"auto"`` -- the GPU backend when it is available and supports the
  request, otherwise the CPU backend, with a ``BACKEND_FALLBACK``
  diagnostic explaining why.
* ``"path"`` -- the path tracer (``rendering/pathtrace/``): physically based
  light transport with global illumination, ray-traced reflections and
  shadows, computed on the CPU with NumPy (it does not use the GPU). Much
  slower than the rasterizers; see ``docs/RENDER_QUALITY.md``.
* ``"gpu_path"`` -- the same path tracer on the GPU (``rendering/gpu/pathtrace.py``):
  a wgpu compute kernel that traces the CPU path tracer's paths (same
  estimator, same random numbers) with software BVH traversal. Needs a
  working adapter like ``"gpu"`` (``GPU_*`` errors otherwise; it never falls
  back to the CPU silently). ``"path"`` remains the reference and the
  fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import RenderError
from .settings import MAX_SAMPLES_PER_AXIS

BACKENDS = ("cpu", "gpu", "auto", "path", "gpu_path")
PATH_BACKENDS = ("path", "gpu_path")
MSAA_COUNTS = (1, 2, 4, 8, 16)
"""MSAA sample counts that can be requested (devices support a subset; 1 and 4 always)."""
MAX_SUPERSAMPLE = 4
"""Largest GPU supersampling factor per axis."""

AUTO_ANTIALIASING = {1: (1, 1), 2: (4, 1), 3: (1, 3), 4: (4, 2)}
"""``samples`` -> ``(msaa, supersample)`` when neither is given: always ``samples^2`` per pixel."""


def check_backend(value: Any) -> str:
    if value not in BACKENDS:
        raise RenderError(
            "INVALID_SETTINGS",
            f"backend must be one of {', '.join(BACKENDS)}, got {value!r}",
            "backend",
        )
    return value


@dataclass(frozen=True)
class Antialiasing:
    """How the GPU backend antialiases one render.

    The image is rendered at ``supersample`` times its size along each axis
    with ``msaa`` samples per (supersampled) pixel; every pixel averages
    ``msaa * supersample**2`` samples in linear light.
    """

    msaa: int
    supersample: int

    @property
    def samples_per_pixel(self) -> int:
        return self.msaa * self.supersample * self.supersample

    def to_dict(self) -> dict[str, int]:
        return {
            "msaa": self.msaa,
            "supersample": self.supersample,
            "samples_per_pixel": self.samples_per_pixel,
        }


def antialiasing_plan(samples: int, msaa: Any = None, supersample: Any = None) -> Antialiasing:
    """The GPU antialiasing for ``samples`` and optional explicit ``msaa``/``supersample``.

    With neither given, the GPU keeps the meaning of ``samples`` (``samples^2``
    samples per pixel): 1 -> no antialiasing, 2 -> 4x MSAA, 3 -> 3x3
    supersampling, 4 -> 4x MSAA on a 2x2-supersampled image. Giving only one
    of them sets the other to 1. ``supersample=samples, msaa=1`` uses exactly
    the CPU renderer's sample positions.
    """
    if msaa is not None and (
        isinstance(msaa, bool) or not isinstance(msaa, int) or msaa not in MSAA_COUNTS
    ):
        raise RenderError(
            "INVALID_SETTINGS",
            f"msaa must be one of {', '.join(map(str, MSAA_COUNTS))}, got {msaa!r}",
            "msaa",
        )
    if supersample is not None and (
        isinstance(supersample, bool)
        or not isinstance(supersample, int)
        or not 1 <= supersample <= MAX_SUPERSAMPLE
    ):
        raise RenderError(
            "INVALID_SETTINGS",
            f"supersample must be an integer in [1, {MAX_SUPERSAMPLE}], got {supersample!r}",
            "supersample",
        )
    if msaa is None and supersample is None:
        m, k = AUTO_ANTIALIASING[min(max(int(samples), 1), MAX_SAMPLES_PER_AXIS)]
        return Antialiasing(m, k)
    return Antialiasing(1 if msaa is None else msaa, 1 if supersample is None else supersample)
