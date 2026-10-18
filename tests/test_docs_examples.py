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


@pytest.mark.parametrize(
    ("doc", "starts_with"),
    [
        (
            "docs/FEEDBACK.md",
            'from aether3d import FeedbackSession, Scene\n\nscene = Scene("Reading',
        ),
        ("README.md", 'from aether3d import FeedbackSession, Scene\n\nscene = Scene("Corner")'),
    ],
)
def test_feedback_quick_starts(doc, starts_with, capsys):
    """The visual feedback quick starts print what their comments say (CPU renders)."""
    source = python_block(ROOT / doc, starts_with)
    namespace: dict = {}
    exec(compile(source, doc, "exec"), namespace)
    printed = capsys.readouterr().out.splitlines()
    stated = [line.split("# ", 1)[1] for line in source.splitlines() if "print(" in line]
    assert printed == stated
    assert namespace["report"].stop_reason == "accepted"
    if "corrections" in namespace:  # "Lower 'Vase' 0.3 m down onto 'Table'" (safe)
        (proposal,) = namespace["corrections"].proposals
        assert proposal.title == "Lower 'Vase' 0.3 m down onto 'Table'"
        assert proposal.safety == "safe"
    vase, table = namespace["scene"].get_by_name("Vase"), namespace["scene"].get_by_name("Table")
    assert vase.bounds().aabb.min[2] == pytest.approx(table.bounds().aabb.max[2], abs=1e-6)


def test_feedback_documents_in_the_docs_are_valid():
    """The JSON examples of docs/FEEDBACK.md and the agent guide pass schemas and checkers."""
    jsonschema = pytest.importorskip("jsonschema")
    from aether3d.feedback import parse_expectations
    from aether3d.feedback.schema import evaluation_schema, expectations_schema

    found = 0
    for name in ("docs/FEEDBACK.md", "src/aether3d/feedback/guide.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for block in re.findall(r"```json\n(.*?)```", text, re.S):
            data = json.loads(block)
            if data.get("format") == "aether3d.expectations":
                jsonschema.validate(data, expectations_schema())
                parse_expectations(data)
                found += 1
            elif data.get("format") == "aether3d.vision-evaluation":
                jsonschema.validate(data, evaluation_schema())
                found += 1
    assert found >= 3


# --------------------------------------------------------------------------- #
# Advanced modeling and sculpting (0.13.0)
# --------------------------------------------------------------------------- #


def assert_printed(source: str, out: str) -> None:
    """Each ``print(...)  # expected`` line of an example printed what its comment says
    (the comment may go on after the value: ``# solid: closed, ...``)."""
    expected = [
        line.split("  # ", 1)[1].strip()
        for line in source.splitlines()
        if line.lstrip().startswith("print(") and "  # " in line
    ]
    printed = out.strip().splitlines()
    assert len(printed) == len(expected), (printed, expected)
    for got, want in zip(printed, expected, strict=True):
        assert want == got or want.startswith((got + ":", got + " (")), (got, want)


def test_modeling_bevel_example(capsys):
    doc = ROOT / "docs/MODELING.md"
    source = python_block(doc, "from aether3d.editing import EditableMesh\n\nm = EditableMesh()")
    exec(compile(source, "MODELING.md", "exec"), {})
    assert_printed(source, capsys.readouterr().out)


@pytest.mark.parametrize("doc", ["docs/MODELING.md", "README.md"])
def test_sculpting_quick_start(doc, capsys):
    source = python_block(
        ROOT / doc, "from aether3d import Scene, create_sphere\n\nscene = Scene()\nhead"
    )
    namespace: dict = {}
    exec(compile(source, doc, "exec"), namespace)
    assert_printed(source, capsys.readouterr().out)
    assert namespace["s"].result.changed


def test_sculpt_script_example():
    from aether3d import create_sphere, run_sculpt_script

    text = (ROOT / "docs/MODELING.md").read_text(encoding="utf-8")
    block = re.search(r'```json\n(\{\n  "format": "aether3d.sculpt-script".*?)\n```', text, re.S)
    scene = Scene()
    scene.add_object("Head", mesh=create_sphere(1.0, 64, 32))
    out = run_sculpt_script(scene, json.loads(block.group(1)))
    entry = out["sculpts"][0]
    assert entry["commit"]["changed"]
    assert entry["operations"][-1]["report"]["summary"]["tier"] == "solid"
    assert scene.history.entries[-1].label == "Shape the head"


# --------------------------------------------------------------------------- #
# UV texture painting and baking (0.14.0)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("doc", ["docs/TEXTURING.md", "README.md"])
def test_texturing_quick_start(doc, capsys):
    source = python_block(
        ROOT / doc, "from aether3d import Scene, create_sphere\n\nscene = Scene()\nball"
    )
    namespace: dict = {}
    exec(compile(source, doc, "exec"), namespace)
    assert_printed(source, capsys.readouterr().out)


def test_texture_script_example():
    from aether3d.texturing import run_texture_script
    from aether3d.texturing.schema import texture_script_schema

    jsonschema = pytest.importorskip("jsonschema")
    text = (ROOT / "docs/TEXTURING.md").read_text(encoding="utf-8")
    block = re.search(r'```json\n(\{\n  "format": "aether3d.texture-script".*?)\n```', text, re.S)
    script = json.loads(block.group(1))
    jsonschema.validate(script, texture_script_schema())
    scene = Scene()
    scene.add_cube("Crate", size=2)
    out = run_texture_script(scene, script)
    assert out["operations"][4]["dabs"] > 1
    assert out["operations"][-1]["overlapping_triangles"] == 0
    assert scene.history.entries[-1].label == "Paint the crate"


def test_texturing_guide_script_is_valid():
    """The agent guide's example script matches the schema and runs."""
    from aether3d.texturing import run_texture_script
    from aether3d.texturing.schema import texture_script_schema

    jsonschema = pytest.importorskip("jsonschema")
    text = (ROOT / "src/aether3d/texturing/guide.md").read_text(encoding="utf-8")
    script = json.loads(re.search(r"```json\n(\{.*?\})\n```", text, re.S).group(1))
    jsonschema.validate(script, texture_script_schema())
    scene = Scene()
    scene.add_cube("Crate", size=1)
    high = scene.add_cube("Crate high", size=1)
    high.visible = False
    out = run_texture_script(scene, script)
    assert [r["op"] for r in out["operations"]][-1] == "bake"


# --------------------------------------------------------------------------- #
# Precision modeling, CAD and architecture (0.15.0)
# --------------------------------------------------------------------------- #

CAD_START = (
    'from aether3d import Scene\n\nscene = Scene()\ndoc = scene.cad\ndoc.set_units(length="mm")'
)


@pytest.mark.parametrize("doc", ["docs/CAD.md", "README.md"])
def test_cad_quick_start(doc, capsys):
    pytest.importorskip("manifold3d")
    source = python_block(ROOT / doc, CAD_START)
    namespace: dict = {}
    exec(compile(source, doc, "exec"), namespace)
    assert_printed(source, capsys.readouterr().out)


def test_cad_doc_script_and_assembly_examples(capsys):
    pytest.importorskip("manifold3d")
    from aether3d.cad.schema import cad_script_schema
    from aether3d.cad.script import run_cad_script

    jsonschema = pytest.importorskip("jsonschema")
    text = (ROOT / "docs/CAD.md").read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(python_block(ROOT / "docs/CAD.md", CAD_START), "CAD.md", "exec"), namespace)
    capsys.readouterr()
    scene, doc = namespace["scene"], namespace["doc"]
    script = json.loads(re.search(r'```json\n(\{"format": "aether3d.cad-script".*?)\n```', text,
                                  re.S).group(1))  # fmt: skip
    jsonschema.validate(script, cad_script_schema())
    out = run_cad_script(scene, script)
    assert out["operations"][-1]["result"]["checks"][0]["pass"]
    assert scene.history.entries[-1].label == "Wider plate"
    # The assembly snippet, with the pin it refers to.
    pin = doc.part("Pin")
    pin.add_feature({"type": "cylinder", "id": "shaft", "radius": 4, "height": 30,
                     "frame": {"origin": [0, 0, 15]}})  # fmt: skip
    pin.add_feature({"type": "cylinder", "id": "head", "radius": 7, "height": 4,
                     "frame": {"origin": [0, 0, 32]}, "operation": "add"})  # fmt: skip
    block = re.search(r"```python\n(doc.add\(\s*\"assembly\".*?)```", text, re.S).group(1)
    exec(compile(block, "CAD.md", "exec"), {"doc": doc})
    info = doc.info("Hinge")
    assert info["status"] in ("under_constrained", "fully_constrained") and not info["conflicts"]


def test_buildings_quick_start(capsys):
    pytest.importorskip("manifold3d")
    source = python_block(ROOT / "docs/BUILDINGS.md", CAD_START.replace('"mm"', '"m"'))
    exec(compile(source, "BUILDINGS.md", "exec"), {})
    assert_printed(source, capsys.readouterr().out)


# --------------------------------------------------------------------------- #
# Animation, rigging and cinematics (0.17.0)
# --------------------------------------------------------------------------- #

ANIMATION_START = (
    'from aether3d import Scene\n\nscene = Scene()\nball = scene.add_sphere("Ball", radius=0.25'
)


@pytest.mark.parametrize("doc", ["docs/ANIMATION.md", "README.md"])
def test_animation_quick_start(doc, capsys):
    source = python_block(ROOT / doc, ANIMATION_START)
    exec(compile(source, doc, "exec"), {})
    assert_printed(source, capsys.readouterr().out)


def test_animation_doc_script_example():
    from aether3d.animation.schema import animation_script_schema
    from aether3d.animation.script import run_animation_script

    jsonschema = pytest.importorskip("jsonschema")
    text = (ROOT / "docs/ANIMATION.md").read_text(encoding="utf-8")
    script = json.loads(re.search(r'```json\n(\{"format": "aether3d.animation-script".*?)\n```',
                                  text, re.S).group(1))  # fmt: skip
    jsonschema.validate(script, animation_script_schema())
    scene = Scene()
    scene.add_sphere("Ball")
    out = run_animation_script(scene, script)
    report = out["operations"][-1]["result"]
    assert report["keys"] == 3 and not report["overshoot"]
    assert scene.history.entries[-1].label == "Bounce"


# --------------------------------------------------------------------------- #
# Procedural worlds (0.18.0)
# --------------------------------------------------------------------------- #

WORLDS_START = 'from aether3d import Scene\n\nscene = Scene("Hills")\nhills = scene.worlds.create('


@pytest.mark.parametrize("doc", ["docs/WORLDS.md", "README.md"])
def test_worlds_quick_start(doc, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = python_block(ROOT / doc, WORLDS_START)
    exec(compile(source, doc, "exec"), {})
    assert_printed(source, capsys.readouterr().out)
    assert (tmp_path / "hills.aether").stat().st_size < 64_000  # no generated geometry


def test_worlds_doc_script_example():
    from aether3d.worlds.script import run_world_script, schema

    jsonschema = pytest.importorskip("jsonschema")
    text = (ROOT / "docs/WORLDS.md").read_text(encoding="utf-8")
    script = json.loads(re.search(r'```json\n(\{"format": "aether3d.world-script".*?)\n```',
                                  text, re.S).group(1))  # fmt: skip
    jsonschema.validate(script, schema())
    scene = Scene()
    out = run_world_script(scene, script)
    assert [o["op"] for o in out["operations"]] == ["create", "add_feature", "build_settlements"]
    assert out["operations"][2]["settlements"]  # a village was built
    assert any(i["kind"] == "house" for i in out["queries"][0]["items"])
    assert scene.history.entries[-1].label == "Valley with a village"


# --------------------------------------------------------------------------- #
# Character rigging (0.19.0)
# --------------------------------------------------------------------------- #

RIGGING_START = "from aether3d import Scene\nfrom aether3d.sculpt.implicit import humanoid_shapes"
RIGID_START = 'from aether3d import Scene\n\nscene = Scene("Blocks")'


def test_rigging_quick_start(capsys):
    source = python_block(ROOT / "docs/CHARACTER_RIGGING.md", RIGGING_START)
    exec(compile(source, "CHARACTER_RIGGING.md", "exec"), {})
    assert_printed(source, capsys.readouterr().out)


def test_rigid_parts_example(capsys):
    source = python_block(ROOT / "docs/CHARACTER_RIGGING.md", RIGID_START)
    exec(compile(source, "CHARACTER_RIGGING.md", "exec"), {})
    assert_printed(source, capsys.readouterr().out)


def test_rigging_doc_script_and_pose_examples():
    from aether3d.animation.schema import animation_script_schema
    from aether3d.animation.script import run_animation_script
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    jsonschema = pytest.importorskip("jsonschema")
    text = (ROOT / "docs/CHARACTER_RIGGING.md").read_text(encoding="utf-8")
    script = json.loads(re.search(r'```json\n(\{"format": "aether3d.animation-script".*?)\n```',
                                  text, re.S).group(1))  # fmt: skip
    jsonschema.validate(script, animation_script_schema())
    scene = Scene()
    scene.add_object("Body", mesh=implicit_surface(humanoid_shapes(), 0.02))
    out = run_animation_script(scene, script)
    assert [o["op"] for o in out["operations"]][0] == "autorig"
    assert "knee_bend_120.L" in out["operations"][-1]["result"]["poses"]
    assert scene.history.entries[-1].label == "Rig the hero"
    pose = json.loads(re.search(r'```json\n(\{"name": "deep_knee".*?)\n```', text,
                                re.S).group(1))  # fmt: skip
    rep = scene.animation.diagnose_rig("Hero", poses=[pose], animation=False)
    assert set(rep["poses"]) == {"deep_knee"}
    limits = re.search(r"```json\n(\{\"swing\".*?)\n```", text, re.S).group(1).splitlines()
    from aether3d.animation.limits import check_limits

    for line in limits:
        check_limits(json.loads(line), "limits")
