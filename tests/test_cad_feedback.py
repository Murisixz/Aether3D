"""Stage 12 feedback meets Stage 15 CAD: the ``cad_valid`` and ``cad_dimension``
expectation checks (exact, from the document)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether3d import FeedbackError, FeedbackSession, Scene
from aether3d.cad.script import run_cad_script

pytest.importorskip("manifold3d")

SCRIPT = json.loads((Path(__file__).parent / "data" / "cad" / "bracket_script.json").read_text())


def diagnose(scene: Scene, *checks):
    doc = {"format": "aether3d.expectations", "version": 1, "checks": list(checks),
           "defaults": False}  # fmt: skip
    return FeedbackSession(scene, doc, width=96, height=54).diagnose()


def bracket():
    scene = Scene("feedback")
    run_cad_script(scene, SCRIPT)
    return scene


def test_cad_valid_reports_unconstrained_sketches():
    scene = bracket()
    d = diagnose(scene, {"check": "cad_valid", "fully_constrained": True})
    assert d.accepted and d.findings == []
    run_cad_script(scene, {"format": "aether3d.cad-script", "version": 1, "operations": [
        {"op": "delete", "item": "Profile", "list": "constraints", "ids": ["d"]}]})  # fmt: skip
    d = diagnose(scene, {"check": "cad_valid", "items": ["Profile"], "fully_constrained": True})
    [f] = d.findings
    assert f.code == "CAD_UNDERCONSTRAINED" and f.magnitude == 1.0
    assert diagnose(scene, {"check": "cad_valid"}).accepted  # free, but valid


def test_cad_dimension_is_exact():
    scene = bracket()
    ref = {"part": "Bracket", "face": "base:end"}
    other = {"part": "Bracket", "face": "base:start"}
    ok = diagnose(
        scene,
        {
            "check": "cad_dimension",
            "reference": ref,
            "other": other,
            "quantity": "parallel_distance",
            "value": 10,
            "tolerance": 1e-9,
        },
    )
    assert ok.accepted
    bad = diagnose(
        scene,
        {
            "check": "cad_dimension",
            "reference": ref,
            "other": other,
            "quantity": "parallel_distance",
            "min": 12,
        },
    )
    [f] = bad.findings
    assert f.code == "CAD_DIMENSION" and f.magnitude == pytest.approx(2.0, rel=1e-12)
    area = diagnose(
        scene, {"check": "cad_dimension", "reference": ref, "quantity": "area", "max": 1}
    )
    assert area.findings[0].code == "CAD_DIMENSION"
    lost = diagnose(
        scene,
        {
            "check": "cad_dimension",
            "reference": {"part": "Bracket", "face": "gone"},
            "quantity": "area",
            "max": 1,
        },
    )
    assert lost.findings[0].code == "CAD_REFERENCE"


@pytest.mark.parametrize(
    "bad",
    [
        {"quantity": "volume", "max": 1},
        {"quantity": "area"},
        {"quantity": "area", "max": 1, "items": ["x"]},
    ],
)
def test_cad_check_arguments_are_checked(bad):
    scene = bracket()
    with pytest.raises(FeedbackError):
        diagnose(scene, {"check": "cad_dimension", "reference": {"part": "Bracket"}, **bad})
