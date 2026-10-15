"""Scene plans on the command line: `execute-plan`, `plan validate` and `plan schema`."""

from __future__ import annotations

import io
import json

import pytest

from aether3d import Scene
from aether3d.cli import main


def plan(**sections) -> dict:
    return {"format": "aether3d.scene-plan", "version": 1, **sections}


YARD = plan(
    label="Yard",
    meshes={"crate": {"primitive": "cube", "size": 1}},
    objects={
        "ground": {"mesh": {"primitive": "plane", "size": 20}},
        "crates": {
            "mesh": "@crate",
            "grid": {"count": [4, 3], "spacing": 2},
            "position": [0, 0, 0.5],
        },
    },
    queries=[{"query": "overlaps", "objects": "@crates"}],
)


def run(capsys, *argv: str) -> tuple[int, dict]:
    code = main([*argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "yard.aether"
    Scene("Yard").save(path)
    return path


def _write(tmp_path, doc: dict, name: str = "plan.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_execute_plan_builds_and_saves_the_project(capsys, tmp_path, project):
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, YARD)))
    assert code == 0 and out["ok"] and out["status"] == "applied"
    assert out["output"] == str(project) and out["counts"]["objects"] == 13
    assert out["queries"][0]["result"] == []
    loaded = Scene.load(project)
    assert len(loaded) == 13 and loaded.get_by_name("crates 11").position == (6.0, 4.0, 0.5)


def test_dry_runs_and_validation_write_nothing(capsys, tmp_path, project):
    before = project.read_bytes()
    plan_path = str(_write(tmp_path, YARD))
    code, out = run(capsys, "execute-plan", str(project), plan_path, "--dry-run")
    assert code == 0 and out["status"] == "dry_run" and out["output"] is None
    code, out = run(capsys, "execute-plan", str(project), plan_path, "--validate")
    assert code == 0 and out["status"] == "validated" and out["estimate"]["objects"] == 13
    assert project.read_bytes() == before


def test_history_records_the_plan_as_one_step(capsys, tmp_path, project):
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, YARD)), "--history")
    assert code == 0 and out["history"]["steps"] == 1 and out["step"]["label"] == "Yard"
    code, out = run(capsys, "history", "show", str(project))
    assert [e["label"] for e in out["entries"]] == ["Yard"]
    code, out = run(capsys, "history", "undo", str(project))
    assert code == 0 and len(Scene.load(project)) == 0


def test_plan_errors_are_structured(capsys, tmp_path, project):
    bad = plan(objects={"a": {"mesh": "@nothing", "positon": [0, 0, 0]}})
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, bad)))
    assert code == 1 and not out["ok"]
    error = out["error"]
    assert error["type"] == "PlanError" and error["code"] in ("UNKNOWN_KEY", "UNKNOWN_REFERENCE")
    assert error["location"].startswith("/objects/a")
    assert {i["code"] for i in error["issues"]} == {"UNKNOWN_KEY", "UNKNOWN_REFERENCE"}
    assert len(Scene.load(project)) == 0
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, YARD)))
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, YARD)))
    assert code == 1 and out["error"]["code"] == "NAME_CONFLICT"
    assert out["error"]["details"]["name"] == "ground"


def test_resending_a_keyed_plan_writes_nothing(capsys, tmp_path, project):
    keyed = {**YARD, "idempotency_key": "yard-1"}
    path = str(_write(tmp_path, keyed))
    code, first = run(capsys, "execute-plan", str(project), path)
    stamp = project.stat().st_mtime_ns
    code, again = run(capsys, "execute-plan", str(project), path)
    assert code == 0 and again["status"] == "already_applied" and again["output"] is None
    assert project.stat().st_mtime_ns == stamp
    assert again["symbols"]["crates"]["count"] == 12


def test_plans_from_stdin_limits_and_text_output(capsys, tmp_path, project, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(YARD).encode())))
    code = main(["execute-plan", str(project), "-", "--max-objects", "5"])
    captured = capsys.readouterr()
    assert code == 1 and "LIMIT_EXCEEDED" not in captured.out
    assert "more than 5 objects" in captured.err
    code = main(["execute-plan", str(project), str(_write(tmp_path, YARD)), "--progress"])
    captured = capsys.readouterr()
    assert code == 0 and "Applied plan 'Yard': 13 objects" in captured.out
    assert "saved" in captured.out and "/" in captured.err  # progress lines


def test_texture_images_resolve_in_the_project_directory(capsys, tmp_path, project):
    import numpy as np

    from aether3d.rendering.png import write_png

    write_png(tmp_path / "wood.png", np.full((2, 2, 3), 100, np.uint8))
    good = plan(textures={"wood": {"image": "wood.png"}})
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, good)))
    assert code == 0
    evil = plan(textures={"x": {"image": "../../etc/passwd"}})
    code, out = run(capsys, "execute-plan", str(project), str(_write(tmp_path, evil)))
    assert code == 1 and out["error"]["code"] == "UNSAFE_PATH"


def test_plan_validate(capsys, tmp_path, project):
    good = str(_write(tmp_path, YARD))
    code, out = run(capsys, "plan", "validate", good)
    assert code == 0 and out["valid"] and out["estimate"]["objects"] == 13
    assert out["warnings"][0]["code"] == "NOT_BOUND"
    code, out = run(capsys, "plan", "validate", good, "--project", str(project))
    assert code == 0 and out["valid"] and out["estimate"]["triangles"] == 12 * 12 + 2
    bad = str(_write(tmp_path, plan(objects={"a": {"repeat": 0, "colour": "#fff"}}), "bad.json"))
    code, out = run(capsys, "plan", "validate", bad)
    assert code == 3 and not out["valid"]
    assert [i["pointer"] for i in out["issues"]] == ["/objects/a/colour", "/objects/a/repeat"]
    assert out["issues"][0]["hint"] == "did you mean 'color'?"
    code = main(["plan", "validate", bad])
    text = capsys.readouterr().out
    assert code == 3 and "UNKNOWN_KEY at /objects/a/colour" in text


def test_plan_schema(capsys):
    code, out = run(capsys, "plan", "schema")
    assert code == 0 and out["schema"]["$id"] == "urn:aether3d:schema:scene-plan:1"
    code = main(["plan", "schema"])
    assert json.loads(capsys.readouterr().out)["title"] == "Aether3D scene plan"


def test_importing_aether3d_does_not_load_the_mcp_sdk():
    import subprocess
    import sys

    code = "import aether3d, sys; print(sorted(m for m in sys.modules if m.split('.')[0] == 'mcp'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("mcp") is not None,
    reason="the MCP SDK is installed here (the job without optional dependencies runs this)",
)
def test_the_mcp_server_explains_how_to_install_its_sdk(capsys, tmp_path):
    code = main(["mcp", "serve", "--workspace", str(tmp_path)])
    assert code == 1
    assert 'pip install "aether3d[mcp]"' in capsys.readouterr().err
