"""The automatic feedback loop: observe, diagnose, critique, propose, apply, verify, repeat.

:func:`run_loop` drives a :class:`~.session.FeedbackSession` until the expectations are
met or a reason to stop appears, and says which (:data:`STOP_REASONS`):

* ``accepted`` -- every check passes (and the vision verdict, if required);
* ``awaiting_approval`` -- what remains has only proposals that need approval (subjective
  or ambiguous); the report lists them, and the loop can be resumed after the agent or
  user approved some (``FeedbackSession.apply(..., approve=[...])``);
* ``no_safe_progress`` -- nothing remaining can be corrected automatically;
* ``stagnated`` -- ``patience`` iterations without improvement, or every remaining
  proposal was already tried and undone;
* ``cycle_detected`` -- a finding resolved earlier came back after a later correction;
* ``max_iterations``, ``budget_renders``, ``budget_pixels``, ``budget_time``,
  ``budget_memory``, ``budget_cost`` -- a budget is exhausted;
* ``cancelled`` -- the cancel event was set (between steps; a step in progress finishes
  or rolls back first).

Every iteration is recorded: the diagnosis summary, the proposals, what was applied (the
history step), the verification (resolved, persisting with magnitudes, regressions) and
the decision (kept or undone), with timings. Before the first change the loop sets a
history checkpoint (``feedback:LOOP_ID:before``): ``scene.history.goto(...)`` returns to
the starting point in one call.

A critic (:class:`~.vision.VisionCritic`) is optional; when given, each iteration's main
observation is sent to it and its evaluation joins the diagnosis as vision findings,
which can only produce approval proposals. Deterministic checks need no network and no
model.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import AetherError, FeedbackError
from ._doc import BAD, Doc, join

if TYPE_CHECKING:
    from .session import FeedbackSession

STOP_REASONS = (
    "accepted",
    "awaiting_approval",
    "no_safe_progress",
    "stagnated",
    "cycle_detected",
    "max_iterations",
    "budget_renders",
    "budget_pixels",
    "budget_time",
    "budget_memory",
    "budget_cost",
    "cancelled",
)


@dataclass(frozen=True)
class FeedbackBudget:
    """Bounds of one feedback loop (every one is checked before each step that spends)."""

    max_iterations: int = 5
    max_renders: int = 60
    max_pixels: int = 60_000_000
    """Rendered pixels in total (width x height of every observation)."""
    max_seconds: float = 120.0
    max_memory: int = 512 * 2**20
    """Bytes of observations kept (images and buffers)."""
    max_critic_calls: int = 3
    max_cost: float = float("inf")
    """Critic cost units (``critic.cost`` per call)."""
    patience: int = 2
    """Iterations without improvement before stopping (``stagnated``)."""

    FIELDS = (
        "max_iterations",
        "max_renders",
        "max_pixels",
        "max_seconds",
        "max_memory",
        "max_critic_calls",
        "max_cost",
        "patience",
    )

    @classmethod
    def coerce(cls, value: Any, ptr: str = "/budget") -> FeedbackBudget:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        doc = Doc()
        obj = doc.object(value, ptr, "a budget")
        out: dict[str, Any] = {}
        if obj is not None:
            doc.keys(obj, ptr, cls.FIELDS)
            limits = {
                "max_iterations": (1, 100),
                "max_renders": (1, 10_000),
                "max_pixels": (1, 10**11),
                "max_critic_calls": (0, 100),
                "patience": (1, 100),
                "max_memory": (1 << 20, 1 << 40),
            }
            for key, v in obj.items():
                p = join(ptr, key)
                if key in limits:
                    lo, hi = limits[key]
                    r = doc.integer(v, p, lo=lo, hi=hi)
                elif key == "max_seconds":
                    r = doc.number(v, p, lo=0.0, lo_open=True, hi=86400.0)
                else:  # max_cost
                    r = doc.number(v, p, lo=0.0)
                if r is not BAD:
                    out[key] = r
        doc.issues.raise_if_any("invalid budget")
        return cls(**out)

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.FIELDS}
        if out["max_cost"] == float("inf"):
            out["max_cost"] = None
        return out


@dataclass
class LoopState:
    """Everything a loop carries between iterations (and across a resume)."""

    id: str
    budget: FeedbackBudget
    started: float
    renders_at_start: int
    pixels_at_start: int
    iterations: list[dict[str, Any]] = field(default_factory=list)
    tried: dict[str, str] = field(default_factory=dict)  # proposal signature -> outcome
    resolved: set[str] = field(default_factory=set)
    failing: set[str] = field(default_factory=set)  # the last diagnosis's failing keys
    best: tuple[int, int, float] | None = None
    since_best: int = 0
    critic_calls: int = 0
    cost: float = 0.0
    checkpoint: str | None = None
    status: str = "running"
    stop_reason: str | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)
    revision: int | None = None
    elapsed: float = 0.0


@dataclass
class LoopReport:
    id: str
    status: str
    stop_reason: str
    iterations: list[dict[str, Any]]
    final: dict[str, Any]
    budget: dict[str, Any]
    pending: list[dict[str, Any]]
    checkpoint: str | None
    timings: dict[str, float]
    diagnosis: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "aether3d.feedback-loop",
            "version": 1,
            "id": self.id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "final": self.final,
            "budget": self.budget,
            "pending_approval": self.pending,
            "checkpoint": self.checkpoint,
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
        }


def run_loop(
    session: FeedbackSession,
    *,
    budget: Any = None,
    policy: Any = None,
    critic: Callable[..., Any] | None = None,
    intent: str | None = None,
    cancel: threading.Event | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    resume: str | None = None,
) -> LoopReport:
    """Run (or ``resume``) a feedback loop on ``session`` (see the module docstring)."""
    from .corrections import CorrectionPolicy

    policy = CorrectionPolicy.coerce(policy) if policy is not None else session.policy
    if resume is not None:
        state = session.loops.get(resume)
        if state is None:
            raise FeedbackError("UNKNOWN_LOOP", f"no loop {resume!r} to resume", "resume")
        if budget is not None:
            state.budget = FeedbackBudget.coerce(budget)
        state.status, state.stop_reason, state.pending = "running", None, []
        state.started = time.perf_counter() - state.elapsed
    else:
        budget = FeedbackBudget.coerce(budget)
        seed = json.dumps(
            [session.scene.revision, session.expectations.fingerprint, time.time_ns()]
        )
        loop_id = "loop-" + hashlib.sha256(seed.encode()).hexdigest()[:10]
        state = LoopState(
            loop_id, budget, time.perf_counter(), session.renders, session.rendered_pixels
        )
        session.loops[loop_id] = state
    t0 = time.perf_counter()
    diagnosis = None
    ready = None  # the verification's diagnosis of the scene as it is now, when kept
    while True:
        reason = _budget_reason(session, state, cancel)
        if reason is not None:
            state.stop_reason = reason
            break
        n = len(state.iterations) + 1
        if progress is not None:
            progress(n - 1, state.budget.max_iterations, f"iteration {n}: diagnosing")
        record: dict[str, Any] = {"iteration": n, "revision_before": session.scene.revision}
        t_it = time.perf_counter()
        if ready is not None and not ready.stale:
            diagnosis, ready = ready, None
        else:
            diagnosis = session.diagnose()
        evaluation = None
        if critic is not None and state.critic_calls < state.budget.max_critic_calls:
            evaluation = _critique(session, state, diagnosis, critic, intent, record)
            if evaluation is not None:
                diagnosis = session.diagnose(evaluations=[evaluation])
        record["diagnosis"] = _summary(diagnosis)
        key = (diagnosis.errors, diagnosis.warnings, diagnosis.score)
        if state.best is None or key < state.best:
            state.best, state.since_best = key, 0
        else:
            state.since_best += 1
        failing = set(diagnosis.failing())
        state.resolved |= state.failing - failing  # failed before, not any more
        state.failing = failing
        returned = sorted(failing & state.resolved)
        if returned:
            record["returned"] = returned
            state.iterations.append(_close(record, t_it))
            state.stop_reason = "cycle_detected"
            break
        if diagnosis.accepted:
            state.iterations.append(_close(record, t_it))
            state.stop_reason = "accepted"
            break
        if state.since_best >= state.budget.patience:
            state.iterations.append(_close(record, t_it))
            state.stop_reason = "stagnated"
            break
        corrections = session.propose(diagnosis, policy=policy)
        record["proposals"] = [
            {
                "id": p.id,
                "title": p.title,
                "safety": p.safety,
                "auto": p.policy.get("auto"),
                "signature": p.signature,
            }
            for p in corrections.proposals
        ]
        record["unresolved"] = [
            {"finding": u["finding"], "code": u["code"], "reason": u["reason"]}
            for u in corrections.unresolved
        ]
        eligible = [
            p
            for p in corrections.proposals
            if p.policy.get("auto") and p.signature not in state.tried
        ]
        skipped = [
            p.id
            for p in corrections.proposals
            if p.policy.get("auto") and p.signature in state.tried
        ]
        if skipped:
            record["skipped_tried"] = skipped
        if not eligible:
            approvals = [
                p
                for p in corrections.proposals
                if p.safety == "approval" or not p.policy.get("auto")
            ]
            state.pending = [p.to_dict(compiled=False) for p in approvals]
            state.iterations.append(_close(record, t_it))
            if skipped and not approvals:
                state.stop_reason = "stagnated"
            else:
                state.stop_reason = "awaiting_approval" if approvals else "no_safe_progress"
            break
        if state.checkpoint is None:
            name = f"feedback:{state.id}:before"
            try:
                session.scene.history.checkpoint(name)
                state.checkpoint = name
            except AetherError:
                state.checkpoint = None
        if progress is not None:
            progress(n - 1, state.budget.max_iterations, f"iteration {n}: applying")
        if cancel is not None and cancel.is_set():
            state.iterations.append(_close(record, t_it))
            state.stop_reason = "cancelled"
            break
        result = session.apply(
            corrections, [p.id for p in eligible], policy=policy, verify=policy.verify
        )
        record["applied"] = result.to_dict()
        for p in eligible:
            if p.id in result.applied:
                state.tried[p.signature] = result.status
        if result.status == "applied" and result.verification is not None:
            ready = result.after
        state.iterations.append(_close(record, t_it))
        if result.after is not None and result.after.accepted:
            diagnosis = result.after
            state.stop_reason = "accepted"
            break
    state.elapsed = time.perf_counter() - state.started
    state.status = "accepted" if state.stop_reason == "accepted" else "stopped"
    state.revision = session.scene.revision
    final = _summary(diagnosis) if diagnosis is not None else {}
    if diagnosis is not None and diagnosis.stale:
        final["note"] = "measured before the last kept correction; diagnose again for the scene now"
    if state.stop_reason == "accepted" and diagnosis is not None and not diagnosis.stale:
        final["accepted"] = True
    report = LoopReport(
        id=state.id,
        status=state.status,
        stop_reason=state.stop_reason or "max_iterations",
        iterations=state.iterations,
        final=final,
        budget={
            "limits": state.budget.to_dict(),
            "used": {
                "iterations": len(state.iterations),
                "renders": session.renders - state.renders_at_start,
                "pixels": session.rendered_pixels - state.pixels_at_start,
                "seconds": round(state.elapsed, 6),
                "memory": session.store.nbytes,
                "critic_calls": state.critic_calls,
                "cost": state.cost,
            },
        },
        pending=state.pending,
        checkpoint=state.checkpoint,
        timings={"total": time.perf_counter() - t0},
        diagnosis=diagnosis,
    )
    return report


def _budget_reason(session: FeedbackSession, state: LoopState, cancel: Any) -> str | None:
    b = state.budget
    if cancel is not None and cancel.is_set():
        return "cancelled"
    if len(state.iterations) >= b.max_iterations:
        return "max_iterations"
    if session.renders - state.renders_at_start >= b.max_renders:
        return "budget_renders"
    if session.rendered_pixels - state.pixels_at_start >= b.max_pixels:
        return "budget_pixels"
    if time.perf_counter() - state.started >= b.max_seconds:
        return "budget_time"
    if session.store.nbytes > b.max_memory:
        return "budget_memory"
    if state.cost >= b.max_cost:
        return "budget_cost"
    return None


def _critique(session, state, diagnosis, critic, intent, record) -> dict[str, Any] | None:
    from .vision import critique_request

    default = diagnosis.expectations.default_view or "camera"
    obs = diagnosis.observed.get(default) or next(iter(diagnosis.observed.values()), None)
    if obs is None:
        record["critic"] = {"skipped": "no observation"}
        return None
    cost = float(getattr(critic, "cost", 0.0) or 0.0)
    if state.cost + cost > state.budget.max_cost:
        record["critic"] = {"skipped": "budget_cost"}
        return None
    state.critic_calls += 1
    state.cost += cost
    try:
        answer = critic(critique_request(obs, intent))
    except Exception as exc:  # an external provider failed: recorded, never fatal
        record["critic"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        return None
    if not isinstance(answer, dict):
        record["critic"] = {"error": "the critic returned no JSON object"}
        return None
    record["critic"] = {"name": getattr(critic, "name", None), "observation": obs.id}
    return answer


def _summary(diagnosis: Any) -> dict[str, Any]:
    return {
        "id": diagnosis.id,
        "revision": diagnosis.revision,
        "accepted": diagnosis.accepted,
        "errors": diagnosis.errors,
        "warnings": diagnosis.warnings,
        "score": diagnosis.score,
        "failing": [
            {
                "id": f.id,
                "code": f.code,
                "severity": f.severity,
                "basis": f.basis,
                "message": f.message,
                "magnitude": f.magnitude,
                "unit": f.unit,
            }
            for f in diagnosis.findings
            if f.severity in ("error", "warning")
        ][:30],
        "observations": diagnosis.observations,
    }


def _close(record: dict[str, Any], t0: float) -> dict[str, Any]:
    record["seconds"] = round(time.perf_counter() - t0, 6)
    return record
