"""``aether3d feedback``: observe, check, correct, loop and schema from the command line."""

from __future__ import annotations

import json

import pytest

from aether3d import Scene
from aether3d.cli import main

EXPECT = {
    "format": "aether3d.expectations",
    "version": 1,
    "views": {"main": {"camera": "Cam"}},
    "checks": [
        {"id": "vase", "check": "rests_on", "objects": "Vase", "support": "Table"},
        {"id": "seen", "check": "visible", "objects": ["Table", "Vase"]},
    ],
}


@pytest.fixture
def project(tmp_path):
    scene = Scene("Room")
    scene.add_plane("Floor", size=10)
    scene.add_cube("Table", size=1, position=(0, 0, 0.5))
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, 1.35))
    scene.add_camera("Cam", position=(3, -5, 3), look_at=(0, 0, 0.6))
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1))
    path = tmp_path / "room.aether"
    scene.save(path)
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps(EXPECT))
    return path, expect


def run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr().out
    return code, json.loads(out) if "--json" in args else out


def test_observe_writes_the_observation_and_its_images(project, capsys, tmp_path):
    path, _ = project
    out = tmp_path / "obs"
    code, data = run(
        capsys,
        "feedback",
        "observe",
        str(path),
        "--width",
        "160",
        "--height",
        "90",
        "--out",
        str(out),
        "--images",
        "image,annotated,ids,depth",
        "--json",
    )
    assert code == 0 and data["ok"] and data["renderer"]["device"] == "cpu"
    files = data["files"]
    assert set(files) == {"json", "image", "annotated", "ids", "depth"}
    for key, file in files.items():
        with open(file, "rb") as handle:
            blob = handle.read()
        assert (
            blob.startswith(b"\x89PNG") if key != "json" else json.loads(blob)["id"] == data["id"]
        )
    code, text = run(
        capsys,
        "feedback",
        "observe",
        str(path),
        "--view",
        '{"preset": "top"}',
        "--width",
        "64",
        "--height",
        "64",
    )
    assert code == 0 and "view 'top on everything visible'" in text


def test_check_exits_3_when_not_accepted_then_0(project, capsys):
    path, expect = project
    code, data = run(capsys, "feedback", "check", str(path), "--expect", str(expect), "--json")
    assert code == 3 and not data["accepted"]
    assert [f["code"] for f in data["findings"] if f["severity"] == "error"] == ["FLOATING"]
    code, text = run(capsys, "feedback", "check", str(path), "--expect", str(expect))
    assert code == 3 and "FLOATING" in text and "NOT accepted" in text


def test_correct_proposes_previews_and_applies_with_history(project, capsys):
    path, expect = project
    code, data = run(
        capsys, "feedback", "correct", str(path), "--expect", str(expect), "--preview", "--json"
    )
    assert code == 0 and data["corrections"]["proposals"][0]["safety"] == "safe"
    assert data["preview"]["comparison"]["resolved"]
    assert Scene.load(path).get_by_name("Vase").position[2] == pytest.approx(1.35)  # unchanged
    code, data = run(
        capsys,
        "feedback",
        "correct",
        str(path),
        "--expect",
        str(expect),
        "--apply",
        "--history",
        "--json",
    )
    assert code == 0 and data["applied"]["status"] == "applied"
    assert Scene.load(path).get_by_name("Vase").position[2] == pytest.approx(1.1, abs=1e-6)
    code, data = run(capsys, "history", "show", str(path), "--json")
    assert data["entries"][-1]["label"].startswith("Feedback:")
    code, _ = run(capsys, "history", "undo", str(path), "--json")
    assert Scene.load(path).get_by_name("Vase").position[2] == pytest.approx(1.35)
    code, data = run(capsys, "feedback", "check", str(path), "--expect", str(expect), "--json")
    assert code == 3


def test_loop_saves_the_corrected_project(project, capsys, tmp_path):
    path, expect = project
    output = tmp_path / "fixed.aether"
    code, data = run(
        capsys,
        "feedback",
        "loop",
        str(path),
        "--expect",
        str(expect),
        "--output",
        str(output),
        "--out",
        str(tmp_path / "final"),
        "--json",
    )
    assert code == 0 and data["stop_reason"] == "accepted" and data["output"] == str(output)
    assert Scene.load(output).get_by_name("Vase").position[2] == pytest.approx(1.1, abs=1e-6)
    assert Scene.load(path).get_by_name("Vase").position[2] == pytest.approx(1.35)
    assert data["files"]["main"]["annotated"].endswith("_annotated.png")


def test_errors_are_structured(project, capsys, tmp_path):
    path, _ = project
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "format": "aether3d.expectations",
                "version": 1,
                "checks": [{"check": "rests_onn", "objects": "Vase"}],
            }
        )
    )
    code, data = run(capsys, "feedback", "check", str(path), "--expect", str(bad), "--json")
    assert code == 1 and data["error"]["type"] == "FeedbackError"
    assert (
        data["error"]["code"] == "INVALID_VALUE" and data["error"]["location"] == "/checks/0/check"
    )
    assert data["error"]["issues"][0]["hint"] == "did you mean 'rests_on'?"
    code, data = run(capsys, "feedback", "observe", str(path), "--view", "nope", "--json")
    assert code == 1 and data["error"]["code"] == "UNKNOWN_VIEW"
    code, data = run(
        capsys, "feedback", "correct", str(path), "--expect", '{"format": 1}', "--json"
    )
    assert code == 1 and data["error"]["code"] == "INVALID_FORMAT"


@pytest.mark.parametrize("kind", ["expectations", "vision-evaluation", "corrections"])
def test_schema(capsys, kind):
    code, data = run(capsys, "feedback", "schema", kind, "--json")
    assert code == 0 and data["schema"]["$schema"].endswith("2020-12/schema")
