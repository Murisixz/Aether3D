"""The feedback session: observe, diagnose, propose, apply and verify, on one scene.

:class:`FeedbackSession` holds what the steps share -- the scene, the expectations, the
render settings of observations, an :class:`~.observe.ObservationStore` (observations are
reused while the scene's revision is unchanged) and the default correction policy -- so
each step reuses the work of the last: a diagnosis renders each view once, a correction
re-renders only after the scene changed (and the retained render world then updates only
what changed), spatial queries share one index per revision.

Applying corrections is the only step that changes the scene. It runs the compiled plan
and script as one history transaction, refuses a scene that changed since the proposals
were made (``REVISION_MISMATCH``), and -- unless asked not to -- verifies: it diagnoses
again and undoes the step (``history.undo()``, exact) when a check that passed now fails,
or when the findings it addressed did not improve. The result says what changed, what
was resolved, what persists and what regressed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import AetherError, FeedbackError
from .checks import Diagnoser, Diagnosis, annotate_regions, number_findings, tally
from .corrections import (
    CorrectionPolicy,
    CorrectionSet,
    Proposal,
    bind,
    check_actions,
    compile_actions,
    propose,
    run_compiled,
)
from .expectations import EMPTY_DOCUMENT, parse_expectations
from .observe import Observation, ObservationLimits, ObservationStore, observe

if TYPE_CHECKING:
    from ..scene import Scene


@dataclass
class ApplyResult:
    """What applying corrections did (see the module docstring)."""

    status: str  # "applied" | "rolled_back" | "rejected" | "nothing_to_apply"
    applied: list[str]
    rejected: list[dict[str, Any]]
    revision: dict[str, int]
    step: dict[str, Any] | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    compiled: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    reason: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    after: Diagnosis | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        from .checks import _clean

        out: dict[str, Any] = {
            "status": self.status,
            "applied": self.applied,
            "rejected": self.rejected,
            "revision": self.revision,
            "step": self.step,
            "changes": _clean(self.changes),
        }
        if self.reason:
            out["reason"] = self.reason
        if self.compiled is not None:
            out["compiled"] = self.compiled
        if self.verification is not None:
            out["verification"] = self.verification
        out["timings"] = {k: round(v, 6) for k, v in self.timings.items()}
        return out


class FeedbackSession:
    """Observe, diagnose, propose, apply and verify corrections on ``scene``.

    * ``expectations`` -- an expectations document (dict, JSON text or
      :class:`Expectations`); default: none (only the default checks).
    * ``backend``, ``adapter``, ``width``, ``height``, ``samples``, ``shadows`` -- how
      observations render (default: the CPU rasterizer at 640 x 360; ``"auto"`` uses the
      GPU when it works).
    * ``policy`` -- the default :class:`CorrectionPolicy`.
    * ``store`` -- where observations are kept (default: a new store of 256 MiB).
    """

    def __init__(
        self,
        scene: Scene,
        expectations: Any = None,
        *,
        backend: str = "cpu",
        adapter: Any = None,
        width: int = 640,
        height: int = 360,
        samples: int | None = None,
        shadows: bool = False,
        policy: Any = None,
        store: ObservationStore | None = None,
        limits: ObservationLimits | None = None,
        base_dir: Any = None,
    ) -> None:
        self.scene = scene
        self.expectations = parse_expectations(
            EMPTY_DOCUMENT if expectations is None else expectations
        )
        self.backend = backend
        self.adapter = adapter
        self.width, self.height = width, height
        self.samples = samples
        self.shadows = shadows
        self.policy = CorrectionPolicy.coerce(policy)
        self.store = store if store is not None else ObservationStore()
        self.limits = limits or ObservationLimits()
        self.base_dir = base_dir
        self.renders = 0
        self.rendered_pixels = 0
        self.last_diagnosis: Diagnosis | None = None
        self._diagnoses: dict[str, Diagnosis] = {}
        self.last_corrections: CorrectionSet | None = None
        self.loops: dict[str, Any] = {}

    # -- observing -------------------------------------------------------------

    def view_spec(self, view: Any) -> Any:
        """A view name of the expectations, or a view specification, as a specification."""
        if isinstance(view, str):
            if view in self.expectations.views:
                return self.expectations.views[view]
            raise FeedbackError(
                "UNKNOWN_VIEW",
                f"no view {view!r}",
                "view",
                hint=f"views: {', '.join(self.expectations.views) or 'none'}; or pass a "
                'specification such as {"camera": "Cam"}',
            )
        if view is None and self.expectations.default_view is not None:
            return self.expectations.views[self.expectations.default_view]
        if view is None and self.scene.active_camera is None:
            # No camera to look through: an iso view framed on everything visible (as
            # render(frame=True) does), never added to the scene.
            return {"preset": "iso", "name": "iso (the scene has no camera)"}
        return view

    def observe(
        self,
        view: Any = None,
        *,
        focus: Any = None,
        occlusion: bool = False,
        linear: bool = False,
        width: int | None = None,
        height: int | None = None,
        backend: str | None = None,
        path: dict[str, Any] | None = None,
    ) -> Observation:
        """Observe a view (a name of the expectations' views, or a specification)."""
        before = len(self.store)
        known = set(self.store._items)
        obs = observe(
            self.scene,
            self.view_spec(view),
            width=width or self.width,
            height=height or self.height,
            backend=backend or self.backend,
            adapter=self.adapter,
            samples=self.samples,
            shadows=self.shadows,
            path=path,
            linear=linear,
            focus=focus,
            occlusion=occlusion,
            limits=self.limits,
            store=self.store,
        )
        if obs.id not in known or len(self.store) != before:
            self.renders += 1
            self.rendered_pixels += obs.width * obs.height
        if isinstance(view, str):
            obs.options.setdefault("view_name", view)
        return obs

    def observation(self, oid: str) -> Observation:
        return self.store.get(oid)

    # -- diagnosing ----------------------------------------------------------------

    def _needed_views(self) -> dict[str | None, bool]:
        """The views the checks need, and whether each needs linear light."""
        views: dict[str | None, bool] = {}
        exp = self.expectations
        for check in exp.checks:
            if check.kind in ("visible", "framing", "exposure", "camera_clipping"):
                linear = check.kind == "exposure"
                views[check.view] = views.get(check.view, False) or linear
        if exp.defaults:
            views.setdefault(exp.default_view, False)
        return views

    def diagnose(self, *, evaluations: Any = (), defaults: bool | None = None) -> Diagnosis:
        """Check the scene against the expectations (and default checks); observe the views
        the checks need (reusing observations at this revision). ``evaluations`` are vision
        evaluation documents of observations in the store; they add vision findings."""
        from .vision import check_evaluation

        t0 = time.perf_counter()
        exp = self.expectations
        scene = self.scene
        revision = scene.revision
        key = None
        if not scene.history.in_progress:
            key = json.dumps(  # the expectations' normalized content, not their source
                [revision, exp.to_dict(), defaults, list(evaluations or ())],
                sort_keys=True,
                default=str,
            )
            cached = self._diagnoses.get(key)
            if cached is not None and not cached.stale:
                return cached
        needed = self._needed_views()
        observed: dict[str, Observation] = {}
        notes: list[str] = []
        failures: dict[str | None, FeedbackError] = {}

        def observation(name: str | None) -> Observation:
            key = name if name is not None else "camera"
            if key in observed:
                return observed[key]
            if name in failures:
                raise failures[name]
            try:
                obs = self.observe(name, linear=needed.get(name, False))
            except FeedbackError as exc:
                failures[name] = exc
                raise
            except AetherError as exc:
                failure = FeedbackError(
                    getattr(exc, "code", "OBSERVATION_FAILED"),
                    f"view {key!r} could not be observed: {getattr(exc, 'message', exc)}",
                    "view",
                    hint=getattr(exc, "hint", None),
                )
                failures[name] = failure
                raise failure from None
            observed[key] = obs
            return obs

        t_observe = time.perf_counter()
        for name in needed:
            try:
                observation(name)
            except FeedbackError as exc:
                notes.append(f"view {name or 'camera'!r}: [{exc.code}] {exc.message}")
        t_observe = time.perf_counter() - t_observe
        t_checks = time.perf_counter()
        diagnoser = Diagnoser(scene, exp, observation)
        diagnoser.run()
        use_defaults = exp.defaults if defaults is None else defaults
        if use_defaults:
            diagnoser.defaults(observed)
        t_checks = time.perf_counter() - t_checks
        default_key = exp.default_view if exp.default_view is not None else "camera"
        findings = annotate_regions(diagnoser.findings, observed.get(default_key), default_key)
        # Vision evaluations: checked, grounded, labeled.
        vision_info = None
        checked = []
        if evaluations:
            items = evaluations if isinstance(evaluations, (list, tuple)) else [evaluations]
            vision_info = {"evaluations": [], "rejected": []}
            for k, doc in enumerate(items):
                try:
                    oid = doc.get("observation") if isinstance(doc, dict) else None
                    obs = self.store.get(oid)
                    ev = check_evaluation(doc, obs)
                except FeedbackError as exc:
                    vision_info["rejected"].append({"index": k, "error": exc.to_dict()})
                    continue
                checked.append(ev)
                findings.extend(ev.findings)
                vision_info["evaluations"].append(
                    {
                        k2: v
                        for k2, v in ev.to_dict().items()
                        if k2
                        in (
                            "digest",
                            "observation",
                            "basis",
                            "provenance",
                            "overall",
                            "stale",
                            "warnings",
                        )
                    }
                )
        findings = number_findings(findings)
        key_to_id = {f.key: f.id for f in findings}
        for result in diagnoser.results:
            result.findings = [key_to_id.get(k, k) for k in result.findings]
        deterministic = [f for f in findings if f.basis not in ("vision", "agent")]
        errors, warnings, score = tally(deterministic)
        unknown_errors = [
            r.id for r in diagnoser.results if r.status == "unknown" and r.severity == "error"
        ]
        accepted = errors <= exp.max_errors and not unknown_errors
        if exp.max_warnings is not None and warnings > exp.max_warnings:
            accepted = False
        acceptance: dict[str, Any] = {
            "max_errors": exp.max_errors,
            "max_warnings": exp.max_warnings,
            "unknown_checks": unknown_errors,
            "basis": "deterministic checks",
        }
        if exp.require_vision:
            fresh = [e for e in checked if not e.stale and e.basis == "vision"]
            verdict = fresh[-1].overall.get("matches_request") if fresh else None
            vision_errors = [f for f in findings if f.basis == "vision" and f.severity == "error"]
            acceptance["vision"] = {
                "required": True,
                "evaluations": len(fresh),
                "matches_request": verdict,
                "vision_errors": len(vision_errors),
                "note": "a declared judgment of an image-aware evaluator, not a fact",
            }
            accepted = accepted and verdict is True and not vision_errors
        digest = hashlib.sha256(
            json.dumps(
                [
                    revision,
                    exp.fingerprint,
                    [f.key for f in findings],
                    sorted(o.id for o in observed.values()),
                ]
            ).encode()
        ).hexdigest()[:12]
        diagnosis = Diagnosis(
            id=f"diag-{digest}",
            scene=scene,
            revision=revision,
            expectations=exp,
            checks=diagnoser.results,
            findings=findings,
            observations={k: o.id for k, o in observed.items()},
            errors=errors,
            warnings=warnings,
            score=score,
            accepted=accepted,
            acceptance=acceptance,
            vision=vision_info,
            notes=notes + diagnoser.notes,
            observed=observed,
            evaluations=checked,
            transient=scene.history.in_progress,
        )
        diagnosis.timings = {
            "observe": t_observe,
            "checks": t_checks,
            "total": time.perf_counter() - t0,
        }
        if not diagnosis.transient:
            self.last_diagnosis = diagnosis
            if key is not None:  # the same question at the same revision: the same answer
                self._diagnoses = {k: d for k, d in self._diagnoses.items() if not d.stale}
                self._diagnoses[key] = diagnosis
        return diagnosis

    def preview(self, corrections: CorrectionSet, ids: Any = None) -> dict[str, Any]:
        """What applying proposals would do, without doing it: they are applied inside a
        transaction, the scene is diagnosed and observed, and the transaction is rolled
        back -- the scene, its revision and its history are unchanged afterwards.

        Returns the diagnosis after (summary), the comparison with before, and the main
        view's observation after (transient: never reused).
        """
        scene = self.scene
        if corrections.revision != scene.revision:
            raise FeedbackError(
                "STALE_PROPOSALS", "the scene changed since these proposals", "proposals"
            )
        wanted = (
            [p for p in corrections.proposals if p.policy.get("auto")]
            if ids is None
            else [corrections.get(pid) for pid in ids]
        )
        wanted = [p for p in wanted if p.compiled is not None]
        if not wanted:
            return {"proposals": [], "note": "nothing to preview"}
        bound = [bind(scene, a) for p in wanted for a in p.actions]
        compiled = compile_actions(bound, "Feedback preview", scene.revision)
        before = corrections.diagnosis
        revision, steps = scene.revision, len(scene.history)
        with scene.transaction("Feedback preview") as tx:
            try:
                run_compiled(scene, compiled, "Feedback preview", self.base_dir)
                after = self.diagnose()
                if not after.observed:  # a view of the result, even when no check needed one
                    with contextlib.suppress(AetherError):
                        after.observed["camera"] = self.observe(None)
            finally:
                tx.rollback()
        assert scene.revision == revision and len(scene.history) == steps
        comparison = compare(before, after, [k for p in wanted for k in p.findings])
        default = after.expectations.default_view or "camera"
        main = after.observed.get(default) or next(iter(after.observed.values()), None)
        return {
            "proposals": [p.id for p in wanted],
            "after": {
                "errors": after.errors,
                "warnings": after.warnings,
                "accepted": after.accepted,
                "findings": [f.to_dict() for f in after.findings[:30]],
            },
            "comparison": comparison,
            "observation": main,
            "note": "applied in a transaction and rolled back: nothing changed",
        }

    # -- proposing ------------------------------------------------------------------

    def propose(self, diagnosis: Diagnosis | None = None, *, policy: Any = None) -> CorrectionSet:
        """Proposals for a diagnosis (default: a fresh one)."""
        diagnosis = diagnosis if diagnosis is not None else self.diagnose()
        corrections = propose(diagnosis, CorrectionPolicy.coerce(policy) if policy else self.policy)
        self.last_corrections = corrections
        return corrections

    # -- the loop -------------------------------------------------------------------------

    def run(self, **options: Any) -> Any:
        """Run the automatic feedback loop (:func:`aether3d.feedback.run_loop`: ``budget``,
        ``policy``, ``critic``, ``intent``, ``cancel``, ``progress``, ``resume``)."""
        from .loop import run_loop

        return run_loop(self, **options)

    # -- applying -----------------------------------------------------------------------

    def apply(
        self,
        corrections: CorrectionSet | None = None,
        ids: Any = None,
        *,
        actions: Any = None,
        approve: Any = (),
        policy: Any = None,
        verify: bool | None = None,
        expected_revision: int | None = None,
        label: str | None = None,
    ) -> ApplyResult:
        """Apply proposals (``ids``; default: those the policy applies automatically) of
        ``corrections`` (default: the last proposals), or explicit ``actions`` (an agent's
        own corrections: the call is the approval), as ONE history step.

        ``approve`` lists proposal ids approved explicitly (needed for ``"approval"``
        proposals and those beyond ``auto_apply``). Raises ``FeedbackError``
        (``REVISION_MISMATCH``, ``STALE_PROPOSALS``, ``UNKNOWN_PROPOSAL``, invalid actions).
        """
        t0 = time.perf_counter()
        policy = CorrectionPolicy.coerce(policy) if policy is not None else self.policy
        verify = policy.verify if verify is None else verify
        scene = self.scene
        revision = scene.revision
        if expected_revision is not None and expected_revision != revision:
            raise FeedbackError(
                "REVISION_MISMATCH",
                f"the scene is at revision {revision}, not {expected_revision}",
                "expected_revision",
                hint="observe and diagnose again",
            )
        approved = set(approve or ())
        chosen: list[Proposal] = []
        rejected: list[dict[str, Any]] = []
        explicit: list[dict[str, Any]] = []
        before = None
        if actions is not None:
            explicit = check_actions(actions)
            if corrections is not None or ids is not None:
                raise FeedbackError(
                    "INVALID_ARGUMENT", "give actions or proposals, not both", "actions"
                )
        else:
            corrections = corrections if corrections is not None else self.last_corrections
            if corrections is None:
                raise FeedbackError(
                    "NO_PROPOSALS",
                    "there are no proposals to apply",
                    "proposals",
                    hint="propose corrections first",
                )
            if corrections.revision != revision:
                raise FeedbackError(
                    "STALE_PROPOSALS",
                    f"proposals {corrections.id} were made at revision {corrections.revision}; "
                    f"the scene is at {revision}",
                    "proposals",
                    hint="diagnose and propose again",
                )
            before = corrections.diagnosis
            wanted = list(ids) if ids is not None else None
            for p in corrections.proposals:
                if wanted is not None and p.id not in wanted:
                    continue
                reasons = []
                if p.compiled is None:
                    reasons.append("it cannot be applied as proposed")
                if p.safety == "approval" and p.id not in approved:
                    reasons.append("it needs explicit approval (approve)")
                elif p.safety not in policy.auto_apply and p.id not in approved:
                    if wanted is None:
                        reasons.append(f"safety {p.safety!r} is not in auto_apply")
                    elif p.safety != "approval":
                        pass  # named explicitly: that is the approval of a "verify" proposal
                if reasons:
                    rejected.append({"id": p.id, "reasons": reasons})
                    continue
                chosen.append(p)
            if wanted is not None:
                known = {p.id for p in corrections.proposals}
                for pid in wanted:
                    if pid not in known:
                        corrections.get(pid)  # raises UNKNOWN_PROPOSAL
            chosen = _one_per_target(chosen, rejected)
        action_list = [a for p in chosen for a in p.actions] + explicit
        if not action_list:
            return ApplyResult(
                "nothing_to_apply" if not rejected else "rejected",
                [],
                rejected,
                {"before": revision, "after": revision},
                reason="no proposal is applicable" if rejected else "nothing was asked",
                timings={"total": time.perf_counter() - t0},
            )
        bound = [bind(scene, a, f"/actions/{k}") for k, a in enumerate(action_list)]
        violations = policy.violations(scene, bound)
        if violations:
            raise FeedbackError(
                "NOT_ALLOWED",
                violations[0],
                "policy",
                hint="change the policy or the corrections",
                details={"violations": violations},
            )
        titles = [p.title for p in chosen] or [f"{len(explicit)} action(s)"]
        step_label = label or ("Feedback: " + "; ".join(titles))[:200]
        compiled = compile_actions(bound, step_label, revision)
        if before is None and verify:
            before = self.diagnose()
        t_apply = time.perf_counter()
        try:
            entry = run_compiled(scene, compiled, step_label, self.base_dir)
        except AetherError as exc:
            raise FeedbackError(
                "APPLY_FAILED",
                f"the corrections failed and were rolled back: {getattr(exc, 'message', exc)}",
                "proposals",
                details={"error": exc.to_dict() if hasattr(exc, "to_dict") else str(exc)},
            ) from None
        t_apply = time.perf_counter() - t_apply
        step = None
        if entry is not None:
            step = {"index": entry.index, "label": entry.label, "changes": entry.changes}
        result = ApplyResult(
            "applied",
            [p.id for p in chosen] + (["actions"] if explicit else []),
            rejected,
            {"before": revision, "after": scene.revision},
            step=step,
            changes=[c for b in bound for c in b.changes],
            compiled=compiled,
        )
        t_verify = 0.0
        if entry is None:
            result.status = "applied"
            result.reason = "the corrections changed nothing (no history step)"
        elif verify:
            t_verify = time.perf_counter()
            after = self.diagnose()
            targeted = [k for p in chosen for k in p.findings]
            verification = compare(before, after, targeted)
            decision, why = decide(verification, policy, bool(explicit) and not chosen)
            verification["decision"] = decision
            verification["reason"] = why
            if decision == "rolled_back":
                last = scene.history.entries[-1] if len(scene.history) else None
                if last is not None and last.index == entry.index:
                    scene.history.undo()
                    result.status = "rolled_back"
                    result.reason = why
                    result.revision["after"] = scene.revision
                else:  # pragma: no cover - nothing else runs between apply and verify
                    verification["decision"] = "kept"
                    verification["reason"] = "the step is no longer the last one"
            result.verification = verification
            result.after = after if result.status == "applied" else None
            t_verify = time.perf_counter() - t_verify
        result.timings = {"apply": t_apply, "verify": t_verify, "total": time.perf_counter() - t0}
        return result


def _one_per_target(chosen: list[Proposal], rejected: list[dict[str, Any]]) -> list[Proposal]:
    """At most one proposal per changed object (and one exposure change) per step, and none
    whose findings were measured against an object another proposal moves in the same
    step (a vase's gap to a table that moves too): those wait for the next diagnosis, which
    sees the first one's effect."""
    moved = {t for p in chosen for t in p.targets}
    waiting = []
    for p in chosen:
        if set(p.related) & (moved - set(p.targets)):
            waiting.append(p)
            rejected.append(
                {
                    "id": p.id,
                    "reasons": [
                        "an object it is measured against changes in this "
                        "step: diagnose again after it"
                    ],
                }
            )
    chosen = [p for p in chosen if p not in waiting]
    out, taken = [], set()
    for p in chosen:
        keys = set(p.targets) or {f"global:{a['action']}" for a in p.actions}
        if any(a["action"] in ("set_exposure", "set_ambient") for a in p.actions):
            keys.add("global:exposure")
        if keys & taken:
            rejected.append(
                {
                    "id": p.id,
                    "reasons": [
                        "another proposal changes the same target in "
                        "this step: diagnose again after it"
                    ],
                }
            )
            continue
        taken |= keys
        out.append(p)
    return out


def compare(before: Diagnosis, after: Diagnosis, targeted: list[str] = ()) -> dict[str, Any]:
    """What changed between two diagnoses of the same expectations."""
    b, a = before.failing(), after.failing()
    resolved = sorted(k for k in b if k not in a)
    new = sorted(k for k in a if k not in b)
    persisting = []
    for k in sorted(set(b) & set(a)):
        persisting.append(
            {
                "key": k,
                "code": a[k].code,
                "before": b[k].magnitude,
                "after": a[k].magnitude,
                "improved": _improved(b[k].magnitude, a[k].magnitude),
            }
        )
    targeted = [k for k in dict.fromkeys(targeted) if k in b]
    targeted_resolved = [k for k in targeted if k in resolved]
    targeted_improved = [p["key"] for p in persisting if p["key"] in targeted and p["improved"]]
    return {
        "before": {
            "errors": before.errors,
            "warnings": before.warnings,
            "score": before.score,
            "diagnosis": before.id,
        },
        "after": {
            "errors": after.errors,
            "warnings": after.warnings,
            "score": after.score,
            "diagnosis": after.id,
            "accepted": after.accepted,
        },
        "resolved": resolved,
        "new_failures": [{"key": k, "code": a[k].code, "message": a[k].message} for k in new],
        "persisting": persisting,
        "targeted": targeted,
        "targeted_resolved": targeted_resolved,
        "targeted_improved": targeted_improved,
        "improved": (after.errors, after.warnings, after.score)
        < (before.errors, before.warnings, before.score),
    }


def _improved(before: float | None, after: float | None) -> bool:
    """Whether a finding's magnitude dropped measurably: by 1 % of it, at least (float
    noise between two measurements of an unchanged configuration is not progress)."""
    if before is None or after is None:
        return False
    return after < before - max(1e-9, 0.01 * abs(before))


def decide(verification: dict[str, Any], policy: CorrectionPolicy, agent: bool) -> tuple[str, str]:
    """Keep or roll back a verified step."""
    if verification["new_failures"] and policy.rollback_on_regression:
        codes = ", ".join(sorted({n["code"] for n in verification["new_failures"]}))
        return "rolled_back", f"regression: checks that passed now fail ({codes})"
    progress = verification["targeted_resolved"] or verification["targeted_improved"]
    if agent:
        progress = progress or verification["improved"] or not verification["persisting"]
    if not progress and policy.rollback_if_ineffective and (verification["targeted"] or not agent):
        return "rolled_back", "ineffective: the findings it addressed did not improve"
    return "kept", "the findings it addressed improved and nothing regressed"
