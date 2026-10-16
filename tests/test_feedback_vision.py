"""Vision evaluations: checking, grounding, provenance and their limits (Stage 12).

FIXTURES, NOT VISION. The evaluation documents in this file are written by the test, not
by a model that looked at an image: they isolate the provider contract (what the engine
accepts, how it grounds regions in the object index, how it labels provenance, what it
refuses). Nothing here validates any model's image understanding. The renders they refer
to are real Aether3D renders of known scenes, so the grounding is checked against
geometry the test knows.
"""

from __future__ import annotations

import json

import pytest

from aether3d import FeedbackError, FeedbackSession, Scene
from aether3d.feedback import check_evaluation, critique_request, run_loop
from aether3d.feedback.vision import CritiqueRequest

FORMAT = "aether3d.vision-evaluation"


def scene_and_observation():
    scene = Scene("Studio")
    scene.add_plane("Floor", size=10)
    scene.add_cube("Crate", size=1, position=(-1, 0, 0.5))
    scene.add_sphere("Ball", radius=0.5, position=(1.2, 0, 0.5))
    scene.add_camera("Cam", position=(0, -6, 2), look_at=(0, 0, 0.5))
    scene.add_directional_light("Sun", direction=(-0.3, 0.6, -1))
    session = FeedbackSession(scene, width=200, height=100)
    obs = session.observe()
    return scene, session, obs


def evaluation(obs, *findings, image_input=True, **extra) -> dict:
    """A FIXTURE evaluation (written by the test, not by a model)."""
    return {
        "format": FORMAT,
        "version": 1,
        "observation": obs.id,
        "image_sha256": obs.sha256,
        "provenance": {
            "provider": "test-fixture",
            "model": "none (a fixture written by the test)",
            "image_input": image_input,
        },
        "findings": list(findings),
        **extra,
    }


def bounds(obs, name):
    return obs.record(name, geometry=False)["screen_bounds"]


def test_a_grounded_finding_names_what_its_region_shows():
    scene, session, obs = scene_and_observation()
    crate = scene.get_by_name("Crate")
    ev = check_evaluation(
        evaluation(
            obs,
            {
                "category": "material",
                "severity": "warning",
                "confidence": 0.9,
                "summary": "the crate looks like plastic",
                "region": bounds(obs, "Crate"),
                "objects": [crate.id],
            },
        ),
        obs,
    )
    (f,) = ev.findings
    assert f.basis == "vision" and not f.exact
    assert f.evidence["grounding"] == "grounded"
    assert f.subjects[0]["id"] == crate.id
    assert f.evidence["region_objects"][0]["name"] == "Crate"
    assert f.code == "VISION_MATERIAL" and f.region is not None
    assert ev.to_dict()["provenance"]["image_input"] == "declared"


def test_a_finding_naming_an_object_outside_its_region_is_ungrounded():
    scene, session, obs = scene_and_observation()
    ball = scene.get_by_name("Ball")
    ev = check_evaluation(
        evaluation(
            obs,
            {
                "category": "placement",
                "confidence": 0.99,
                "summary": "the ball floats",
                "region": bounds(obs, "Crate"),
                "objects": [ball.id],
            },
        ),
        obs,
    )
    (f,) = ev.findings
    assert f.evidence["grounding"] == "ungrounded"
    assert f.subjects == ()  # its target is not used
    assert any("ungrounded" in w or "not visible in their region" in w for w in ev.warnings)


def test_region_only_findings_report_the_dominant_object():
    scene, session, obs = scene_and_observation()
    x0, y0, x1, y1 = bounds(obs, "Ball")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ev = check_evaluation(
        evaluation(
            obs,
            {
                "category": "color",
                "confidence": 0.5,
                "summary": "too saturated",
                "region": [cx - 3, cy - 3, cx + 3, cy + 3],
            },
            {
                "category": "composition",
                "confidence": 0.5,
                "summary": "empty right side",
                "region_normalized": [0.8, 0.0, 1.0, 1.0],
            },
        ),
        obs,
    )
    first, second = ev.findings
    assert first.evidence["grounding"] == "region_only"
    assert first.evidence["dominant_object"]["name"] == "Ball"
    assert second.region[0] == 160 and second.region[2] == 200


def test_provenance_without_an_image_input_is_an_agent_judgment():
    scene, session, obs = scene_and_observation()
    ev = check_evaluation(
        evaluation(
            obs, {"category": "style", "confidence": 0.7, "summary": "dull"}, image_input=False
        ),
        obs,
    )
    assert ev.basis == "agent" and ev.findings[0].basis == "agent"
    assert ev.to_dict()["provenance"]["image_input"] is False
    assert ev.to_dict()["provenance"]["image_delivered_by_aether3d"] is False
    critique_request(obs)  # handing the image to a critic is recorded
    assert check_evaluation(evaluation(obs), obs).delivered


def test_evaluations_are_tied_to_their_observation_and_its_revision():
    scene, session, obs = scene_and_observation()
    other = session.observe({"preset": "top"})
    with pytest.raises(FeedbackError) as exc:
        check_evaluation(evaluation(other), obs)
    assert exc.value.code == "OBSERVATION_MISMATCH"
    bad = evaluation(obs)
    bad["image_sha256"] = "0" * 64
    with pytest.raises(FeedbackError) as exc:
        check_evaluation(bad, obs)
    assert exc.value.code == "OBSERVATION_MISMATCH"
    scene.get_by_name("Ball").position = (1.5, 0, 0.5)
    ev = check_evaluation(
        evaluation(obs, {"category": "other", "confidence": 1, "summary": "x"}), obs
    )
    assert ev.stale and ev.findings[0].evidence.get("stale")


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"format": "aether3d.vision"}, "INVALID_FORMAT"),
        ({"provenance": {"model": "m"}}, "MISSING_KEY"),
        ({"provenance": {"provider": "p", "image_input": "yes"}}, "INVALID_VALUE"),
        ({"findings": [{"category": "magic", "confidence": 1, "summary": "x"}]}, "INVALID_VALUE"),
        ({"findings": [{"category": "other", "confidence": 2, "summary": "x"}]}, "OUT_OF_RANGE"),
        (
            {"findings": [{"category": "other", "confidence": 1, "summary": "x" * 501}]},
            "LIMIT_EXCEEDED",
        ),
        (
            {
                "findings": [
                    {
                        "category": "other",
                        "confidence": 1,
                        "summary": "x",
                        "region": [500, 500, 600, 600],
                    }
                ]
            },
            "OUT_OF_RANGE",
        ),
        (
            {
                "findings": [
                    {
                        "category": "other",
                        "confidence": 1,
                        "summary": "x",
                        "suggestion": {"action": "delete", "object": "Ball"},
                    }
                ]
            },
            "INVALID_VALUE",
        ),
        (
            {
                "findings": [
                    {
                        "category": "other",
                        "confidence": 1,
                        "summary": "x",
                        "suggestion": {
                            "action": "translate",
                            "object": "Ball",
                            "offset": [0, 0, 1],
                            "code": "import os",
                        },
                    }
                ]
            },
            "UNKNOWN_KEY",
        ),
        (
            {
                "findings": [
                    {"category": "other", "confidence": 1, "summary": "x", "objects": ["a"] * 11}
                ]
            },
            "LIMIT_EXCEEDED",
        ),
        (
            {"findings": [{"category": "other", "confidence": 1, "summary": "x\x07"}]},
            "INVALID_VALUE",
        ),
        ({"exec": "rm -rf /"}, "UNKNOWN_KEY"),
    ],
)
def test_malformed_and_malicious_evaluations_are_refused(change, code):
    scene, session, obs = scene_and_observation()
    document = {**evaluation(obs), **change}
    with pytest.raises(FeedbackError) as exc:
        check_evaluation(document, obs)
    assert code in {i["code"] for i in exc.value.issues}
    with pytest.raises(FeedbackError):
        check_evaluation(json.dumps(document), obs)


def test_oversized_and_non_json_documents():
    scene, session, obs = scene_and_observation()
    with pytest.raises(FeedbackError) as exc:
        check_evaluation("x" * (2**20 + 1), obs)
    assert exc.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(FeedbackError) as exc:
        check_evaluation("{", obs)
    assert exc.value.code == "INVALID_JSON"


def test_vision_findings_join_a_diagnosis_but_not_its_acceptance():
    scene, session, obs = scene_and_observation()
    fixture = evaluation(
        obs,
        {
            "category": "lighting",
            "severity": "error",
            "confidence": 0.95,
            "summary": "too flat",
            "region": [0, 0, 200, 100],
        },
        overall={"matches_request": False, "summary": "not yet"},
    )
    d = session.diagnose(evaluations=[fixture])
    vision = [f for f in d.findings if f.basis == "vision"]
    assert len(vision) == 1 and d.accepted  # judgments do not count unless asked
    assert d.vision["evaluations"][0]["overall"]["matches_request"] is False
    session.expectations = session.expectations.__class__(
        **{**session.expectations.__dict__, "require_vision": True}
    )
    d = session.diagnose(evaluations=[fixture])
    assert not d.accepted and d.acceptance["vision"]["matches_request"] is False
    fixture["overall"] = {"matches_request": True}
    fixture["findings"][0]["severity"] = "info"
    d = session.diagnose(evaluations=[fixture])
    assert d.accepted


def test_rejected_evaluations_are_reported_not_raised():
    scene, session, obs = scene_and_observation()
    d = session.diagnose(
        evaluations=[
            {
                "format": FORMAT,
                "version": 1,
                "observation": "obs-x",
                "provenance": {"provider": "p"},
            }
        ]
    )
    assert d.vision["rejected"][0]["error"]["code"] == "OBSERVATION_NOT_FOUND"


def test_vision_suggestions_only_ever_make_approval_proposals():
    scene, session, obs = scene_and_observation()
    ball = scene.get_by_name("Ball")
    fixture = evaluation(
        obs,
        {
            "category": "placement",
            "severity": "error",
            "confidence": 0.95,
            "summary": "the ball should touch the crate",
            "region": bounds(obs, "Ball"),
            "objects": [ball.id],
            "suggestion": {"action": "translate", "object": ball.id, "offset": [-0.7, 0, 0]},
        },
        {
            "category": "placement",
            "severity": "error",
            "confidence": 0.2,
            "summary": "unsure",
            "region": bounds(obs, "Ball"),
            "objects": [ball.id],
            "suggestion": {"action": "translate", "object": ball.id, "offset": [1, 0, 0]},
        },
    )
    d = session.diagnose(evaluations=[fixture])
    corrections = session.propose(d, policy={"auto_apply": ["safe", "verify"]})
    (p,) = corrections.proposals
    assert p.safety == "approval" and p.basis == "vision" and not p.policy["auto"]
    assert p.confidence == 0.95
    assert any("min_vision_confidence" in u["reason"] for u in corrections.unresolved)
    result = session.apply(corrections)
    assert result.status == "rejected" and ball.position == (1.2, 0.0, 0.5)
    result = session.apply(corrections, approve=[p.id], verify=False)
    assert result.status == "applied" and ball.position[0] == pytest.approx(0.5)


class FixtureCritic:
    """A STUB critic for the provider contract: it returns a fixed document (no model)."""

    name = "fixture-critic (stub, not a model)"
    cost = 1.0

    def __init__(self, answer=None, fail=False):
        self.calls: list[CritiqueRequest] = []
        self.answer = answer
        self.fail = fail

    def __call__(self, request: CritiqueRequest):
        self.calls.append(request)
        if self.fail:
            raise ConnectionError("provider unreachable")
        if self.answer is not None:
            return self.answer
        return {
            "format": FORMAT,
            "version": 1,
            "observation": request.observation,
            "image_sha256": request.image_sha256,
            "provenance": {"provider": "test-fixture", "image_input": False},
            "findings": [],
        }


def test_a_critic_gets_the_images_and_its_failures_are_contained():
    scene, session, obs = scene_and_observation()
    critic = FixtureCritic()
    report = run_loop(session, critic=critic, budget={"max_critic_calls": 1})
    assert report.stop_reason == "accepted"
    (request,) = critic.calls
    assert request.image_png.startswith(b"\x89PNG") and request.annotated_png.startswith(b"\x89PNG")
    assert request.markers and request.width == 200
    assert report.budget["used"]["critic_calls"] == 1 and report.budget["used"]["cost"] == 1.0
    failing = FixtureCritic(fail=True)
    report = run_loop(session, critic=failing)
    assert report.iterations[0]["critic"]["error"].startswith("ConnectionError")
    garbage = FixtureCritic(answer=["not", "an", "object"])
    report = run_loop(session, critic=garbage)
    assert "no JSON object" in report.iterations[0]["critic"]["error"]
    costly = FixtureCritic()
    report = run_loop(session, critic=costly, budget={"max_cost": 0.5})
    assert not costly.calls and report.iterations[0]["critic"]["skipped"] == "budget_cost"
