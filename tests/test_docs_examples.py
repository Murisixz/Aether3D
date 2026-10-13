"""The code examples in the documentation actually run and give the stated results."""

import json
import re
from pathlib import Path

import pytest

from aether3d import Scene, inspect_mesh, run_edit_script

ROOT = Path(__file__).resolve().parent.parent
# Inset 0.2 then sink 0.1 removes a 1.6 x 1.6 x 0.1 recess; the 0.05 chamfer of one
# 2-unit edge removes 0.05^2 / 2 * 2.
CRATE_VOLUME = 8 - 1.6 * 1.6 * 0.1 - 0.05**2 / 2 * 2


def python_block(path: Path, starts_with: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```python\n(" + re.escape(starts_with) + r".*?)```", text, re.S)
    assert match, f"no example starting with {starts_with!r} in {path.name}"
    return match.group(1)


@pytest.mark.parametrize("doc", ["docs/EDITING.md", "README.md"])
def test_crate_example(doc):
    source = python_block(ROOT / doc, "from aether3d import Scene\n\nscene = Scene()\ncrate")
    namespace: dict = {}
    exec(compile(source, doc, "exec"), namespace)
    volume = inspect_mesh(namespace["crate"].mesh).topology["volume"]
    assert volume == pytest.approx(CRATE_VOLUME, rel=1e-6)


def test_editing_json_script_example():
    text = (ROOT / "docs/EDITING.md").read_text(encoding="utf-8")
    script = json.loads(re.search(r"```json\n(\{.*?\})\n```", text, re.S).group(1))
    scene = Scene()
    scene.add_cube("Crate", size=2)
    result = run_edit_script(scene, script)
    assert result.edits[0]["commit"]["validation"]["status"] == "ok"
    volume = inspect_mesh(scene.get_by_name("Crate").mesh).topology["volume"]
    assert volume == pytest.approx(CRATE_VOLUME, rel=1e-6)


def test_procedural_quick_start(tmp_path, monkeypatch):
    pytest.importorskip("manifold3d")
    monkeypatch.chdir(tmp_path)
    source = python_block(ROOT / "docs/PROCEDURAL.md", "from aether3d import Expr, Scene\n")
    namespace: dict = {}
    exec(compile(source, "PROCEDURAL.md", "exec"), namespace)
    scene, tower = namespace["scene"], namespace["tower"]
    left = scene.get_by_name("Left Tower")
    topo = inspect_mesh(left.mesh).topology
    assert topo["closed"] and topo["volume"] > 0
    # Six slits cut into the 6 m prism (radius 1.5, octagon area 2 * sqrt(2) * 1.5^2).
    prism = 2 * 2**0.5 * 1.5**2 * 6
    assert topo["volume"] < prism - 6 * 0.3 * 0.05
    reloaded = Scene.load(tmp_path / "towers.aether")
    assert reloaded.procedural.parameters["windows"].value == 6
    assert reloaded.get(left.id).mesh == left.mesh
    assert (tmp_path / "towers.glb").is_file()
    assert tower.users == [left.id, scene.get_by_name("Right Tower").id]


def test_procedural_json_script_example():
    from aether3d import run_procedural_script

    text = (ROOT / "docs/PROCEDURAL.md").read_text(encoding="utf-8")
    block = re.search(
        r'```json\n(\{\n  "format": "aether3d.procedural-script".*?)\n```', text, re.S
    )
    scene = Scene()
    result = run_procedural_script(scene, json.loads(block.group(1)))
    assert result["operations"][-1]["report"]["status"] == "ok"
    fence = scene.get_by_name("Fence")
    report = fence.geometry.evaluate_polygons().validate()
    assert report.ok and report.summary["face_count"] == 7 * 26  # 7 bevelled posts


def test_readme_procedural_example():
    source = python_block(
        ROOT / "README.md", "from aether3d import Expr, Scene\n\nscene = Scene()\n"
    )
    namespace: dict = {}
    exec(compile(source, "README.md", "exec"), namespace)
    row = namespace["row"]
    # A bevelled cube: 6 quads + 12 chamfer quads + 8 corner triangles = 44 triangles.
    assert row.is_procedural and row.mesh.triangle_count == 5 * 44
    topo = inspect_mesh(row.mesh).topology
    single = 1 - 12 * 0.08**2 / 2 * (1 - 2 * 0.08) - 8 * (5 / 6) * 0.08**3
    assert topo["volume"] == pytest.approx(5 * single, rel=1e-5)


def test_rendering_quick_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = python_block(
        ROOT / "docs/RENDERING.md", 'from aether3d import Scene\n\nscene = Scene("Studio")'
    )
    namespace: dict = {}
    exec(compile(source, "RENDERING.md", "exec"), namespace)
    result = namespace["result"]
    assert (result.width, result.height) == (640, 360) and (tmp_path / "studio.png").is_file()
    assert all(o["visible_samples"] > 0 for o in result.objects)
    assert result.diagnostics == []


def test_rendering_scene_script_example():
    from aether3d import run_scene_script

    text = (ROOT / "docs/RENDERING.md").read_text(encoding="utf-8")
    block = re.search(r'```json\n(\{\n  "format": "aether3d.scene-script".*?)\n```', text, re.S)
    scene = Scene()
    scene.add_cube("Crate", size=2, position=(0, 0, 1))
    scene.add_plane("Ground", size=20)
    run_scene_script(scene, json.loads(block.group(1)))
    result = scene.render(width=128, height=72)
    assert result.camera["name"] == "Main" and result.stats["lights"] == 2
    assert all(o["visible_samples"] > 0 for o in result.objects)


def test_readme_render_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = python_block(
        ROOT / "README.md", "from aether3d import Scene\n\nscene = Scene()\nscene.add_plane"
    )
    namespace: dict = {}
    exec(compile(source, "README.md", "exec"), namespace)
    result = namespace["result"]
    assert (tmp_path / "crate.png").is_file() and result.width == 640
    assert {o["name"]: o["visible_samples"] > 0 for o in result.objects} == {
        "Ground": True,
        "Crate": True,
    }
