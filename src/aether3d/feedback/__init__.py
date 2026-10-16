"""Visual feedback and automatic corrections (Stage 12): observe, diagnose, correct, verify.

* :func:`observe` renders a view (a scene camera, or a virtual inspection camera) with an
  existing backend and measures what it shows: per object (pixels, bounds, depth,
  occlusion, frustum, clipping) and for the image (coverage, luminance, clipping).
* An **expectations** document (:func:`parse_expectations`) states what the scene should
  satisfy; :class:`FeedbackSession` checks it (:meth:`FeedbackSession.diagnose`), with
  findings labeled by their basis (geometry, render, heuristic, vision, agent).
* **Vision evaluations** (:func:`check_evaluation`) bring an image-aware model's judgment
  in as checked, grounded data -- never as facts.
* **Corrections** (:meth:`FeedbackSession.propose`, :meth:`FeedbackSession.apply`) are
  typed actions from an allowlist, compiled to scene plans and scripts, applied as one
  undo step, verified, and undone when they do not help.
* :func:`run_loop` repeats it all within budgets, and says why it stopped.

See ``docs/FEEDBACK.md``.
"""

from .checks import CheckResult, Diagnosis, Finding
from .corrections import (
    ACTIONS,
    CorrectionPolicy,
    CorrectionSet,
    Proposal,
    check_actions,
)
from .expectations import CHECKS, Check, Expectations, parse_expectations
from .loop import STOP_REASONS, FeedbackBudget, LoopReport, run_loop
from .observe import Observation, ObservationLimits, ObservationStore, observe
from .session import ApplyResult, FeedbackSession, compare
from .views import ResolvedView, camera_view, check_view_spec, resolve_view, virtual_camera
from .vision import (
    CritiqueRequest,
    VisionCritic,
    VisionEvaluation,
    check_evaluation,
    critique_request,
)

__all__ = [
    "ACTIONS",
    "CHECKS",
    "STOP_REASONS",
    "ApplyResult",
    "Check",
    "CheckResult",
    "CorrectionPolicy",
    "CorrectionSet",
    "CritiqueRequest",
    "Diagnosis",
    "Expectations",
    "FeedbackBudget",
    "FeedbackSession",
    "Finding",
    "LoopReport",
    "Observation",
    "ObservationLimits",
    "ObservationStore",
    "Proposal",
    "ResolvedView",
    "VisionCritic",
    "VisionEvaluation",
    "camera_view",
    "check_actions",
    "check_evaluation",
    "check_view_spec",
    "compare",
    "critique_request",
    "observe",
    "parse_expectations",
    "resolve_view",
    "run_loop",
    "virtual_camera",
]
