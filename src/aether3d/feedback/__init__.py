"""Visual feedback and automatic corrections (Stage 12): observe, diagnose, correct, verify.

* :func:`observe` renders a view (a scene camera, or a virtual inspection camera) with an
  existing backend and measures what it shows: per object (pixels, bounds, depth,
  occlusion, frustum, clipping) and for the image (coverage, luminance, clipping).

See ``docs/FEEDBACK.md``.
"""

from .observe import Observation, ObservationLimits, ObservationStore, observe
from .views import ResolvedView, camera_view, check_view_spec, resolve_view, virtual_camera

__all__ = [
    "Observation",
    "ObservationLimits",
    "ObservationStore",
    "ResolvedView",
    "camera_view",
    "check_view_spec",
    "observe",
    "resolve_view",
    "virtual_camera",
]
