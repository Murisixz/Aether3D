"""The code examples in the documentation actually run and give the stated results."""

import json
import os
import re
import shutil
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


def test_gpu_quick_start(tmp_path, monkeypatch, gpu):
    monkeypatch.chdir(tmp_path)
    if gpu.adapter:
        monkeypatch.setenv("AETHER3D_GPU_ADAPTER", gpu.adapter)
    source = python_block(ROOT / "docs/GPU.md", "from aether3d import Scene, gpu_status\n")
    namespace: dict = {}
    exec(compile(source, "GPU.md", "exec"), namespace)
    result = namespace["result"]
    assert (tmp_path / "studio.png").is_file() and result.width == 1920
    assert result.backend["name"] == "gpu" and result.backend["adapter"]["name"] == gpu.name
    assert [m["light"] for m in result.backend["shadows"]["maps"]] == ["Sun"]


def test_materials_quick_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the example runs from the repository root: same layout here
    if os.environ.get("AETHER3D_GPU_TESTS") == "skip":  # this job keeps off the GPU, so the
        monkeypatch.setenv("AETHER3D_GPU_ADAPTER", "no adapter has this name")  # "auto": CPU
    shutil.copytree(ROOT / "examples" / "textures", tmp_path / "examples" / "textures")
    source = python_block(
        ROOT / "docs/MATERIALS.md", 'from aether3d import Scene\n\nscene = Scene("Courtyard")'
    )
    namespace: dict = {}
    exec(compile(source, "MATERIALS.md", "exec"), namespace)
    assert (tmp_path / "courtyard.png").is_file() and (tmp_path / "courtyard_gpu.png").is_file()
    project = Scene.load(tmp_path / "courtyard.aether")
    assert {m.name for m in project.materials.materials} == {"Lawn", "Bricks", "Wood"}
    paths = sorted(t.path for t in project.materials.textures if getattr(t, "path", None))
    assert paths == [f"examples/textures/stone_bricks_{n}.png" for n in ("color", "normal")]
    result = project.render(width=96, height=54)
    assert result.diagnostics == [] and all(o["visible_samples"] for o in result.objects)


def test_materials_scene_script_example(tmp_path):
    pytest.importorskip("PIL.Image")  # the example's JPEG texture
    from aether3d import run_scene_script

    text = (ROOT / "docs/MATERIALS.md").read_text(encoding="utf-8")
    block = re.search(r'```json\n(\{\n  "format": "aether3d.scene-script".*?)\n```', text, re.S)
    (tmp_path / "textures").mkdir()
    shutil.copyfile(ROOT / "examples/textures/wood_planks.jpg", tmp_path / "textures/planks.jpg")
    scene = Scene()
    for name in ("Wall 1", "Wall 2", "Floor"):
        scene.add_cube(name)
    run_scene_script(scene, json.loads(block.group(1)), base_dir=tmp_path)
    walls = [scene.get_by_name(n).material.name for n in ("Wall 1", "Wall 2")]
    floor = scene.get_by_name("Floor").material
    assert walls == ["Wall", "Wall"] and floor.texture("base_color").status()["format"] == "jpeg"


def test_path_tracing_quick_start(tmp_path, monkeypatch):
    """The path tracer's quick start runs as written (at a smaller size and sample count)."""
    monkeypatch.chdir(tmp_path)
    source = python_block(ROOT / "docs/RENDER_QUALITY.md", "from aether3d import Scene\n")
    small = source.replace("width=960", "width=48").replace("height=540", "height=27")
    small = small.replace('"samples": 64', '"samples": 2')
    assert small != source
    namespace: dict = {}
    exec(compile(small, "RENDER_QUALITY.md", "exec"), namespace)
    result = namespace["result"]
    assert result.backend["name"] == "path" and result.backend["device"] == "cpu"
    assert "ao" in result.passes and (tmp_path / "studio.hdr").is_file()
    assert (tmp_path / "studio.png").read_bytes().startswith(b"\x89PNG")


def test_history_quick_start(capsys):
    """The history guide's first example runs as written and prints what it says."""
    source = python_block(ROOT / "docs/HISTORY.md", "from aether3d import Scene\n")
    namespace: dict = {}
    exec(compile(source, "HISTORY.md", "exec"), namespace)
    printed = capsys.readouterr().out.strip()
    assert printed == str(
        [
            "Add object 'Crate'",
            "Set position of 'Crate'",
            "Set appearance color of 'Crate'",
            "Build the gate",
        ]
    )
    scene = namespace["scene"]
    assert scene.get_by_name("Crate").position == (2.0, 0.0, 0.5)
    assert len(scene) == 3 and not scene.history.can_redo


def test_spatial_quick_start(capsys):
    start = "from aether3d import Scene\nfrom aether3d.spatial"
    source = python_block(ROOT / "docs/SPATIAL.md", start)
    exec(compile(source, "docs/SPATIAL.md", "exec"), {})
    printed = capsys.readouterr().out.splitlines()
    # Every printed value is stated in a comment of the example.
    comments = [line.split("# ")[1] for line in source.splitlines() if "print(" in line]
    assert len(printed) == len(comments) == 7
    for value, comment in zip(printed, comments, strict=True):
        assert comment.startswith(value), (value, comment)


def test_readme_spatial_example(capsys):
    source = python_block(ROOT / "README.md", "from aether3d import Scene\nfrom aether3d.spatial")
    namespace: dict = {}
    exec(compile(source, "README.md", "exec"), namespace)
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("Crate (2.0, 0.0, 1.0) (0.0, 0.0, 1.0)")
    assert printed[1:] == ["4.0", "inside", "True"]
    assert [(o.a.name, o.b.name) for o in namespace["touching"]] == [("Ground", "Crate")]
    assert [m.object.name for m in namespace["in_yard"]] == ["Ground", "Crate"]  # Cam: y = -8
    assert [o.name for o in namespace["props"]] == ["Crate"]


def test_readme_scene_plan_example():
    source = python_block(
        ROOT / "README.md", 'from aether3d import Scene\n\nscene = Scene("Courtyard")'
    )
    namespace: dict = {}
    exec(compile(source, "README.md", "exec"), namespace)
    result, scene = namespace["result"], namespace["scene"]
    assert result.counts["objects"] == 37 and result.queries[0]["result"] == []
    assert len(scene) == 0  # the example ends with an undo


def test_scene_plan_docs_first_plan():
    text = (ROOT / "docs/SCENE_PLANS.md").read_text(encoding="utf-8")
    plan = json.loads(re.search(r"## A first plan\n\n```json\n(.*?)```", text, re.S).group(1))
    scene = Scene("Courtyard")
    result = scene.execute_plan(plan)
    assert result.counts["objects"] == 38 and result.queries[0]["result"] == []
    assert sum(1 for o in scene if o.name.startswith("wall ")) == 27


def test_every_complete_plan_in_the_docs_is_valid():
    from aether3d import validate_plan

    found = 0
    for doc in ("docs/SCENE_PLANS.md", "docs/MCP.md", "README.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for block in re.findall(r"```json\n(.*?)```", text, re.S):
            if '"aether3d.scene-plan"' not in block:
                continue
            result = validate_plan(json.loads(block))
            assert result.ok, (doc, result.issues)
            found += 1
    assert found >= 1
