import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene
from aether3d.materials import ImageTexture

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CASTLE_TEXTURES = ("stone_bricks_color.png", "stone_bricks_normal.png", "wood_planks.jpg")


def test_first_scene_example_runs(tmp_path):
    output = tmp_path / "first_scene.aether"
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "first_scene.py"), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Saved 5 objects" in result.stdout
    scene = Scene.load(output)
    moon = scene.get_by_name("Moon")
    # Pivot at (3, 0, 1) rotated 90 degrees about Z puts the moon's (2, 0, 0) offset at +Y.
    x, y, z = moon.world_position
    assert abs(x - 3) < 1e-9 and abs(y - 2) < 1e-9 and abs(z - 1) < 1e-9


def test_committed_example_project_is_valid():
    scene = Scene.load(EXAMPLES / "first_scene.aether")
    assert [o.name for o in scene] == ["Cube", "Sphere", "Ground", "Pivot", "Moon"]


def test_gltf_roundtrip_example_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "gltf_roundtrip.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks passed" in result.stdout
    assert (tmp_path / "town.glb").is_file()


def test_gltf_roundtrip_verifier_detects_corruption(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gltf_roundtrip", EXAMPLES / "gltf_roundtrip.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    scene = example.build_scene()
    scene.export(tmp_path / "t.glb")
    assert example.verify(scene, Scene.import_file(tmp_path / "t.glb")) == []
    tampered = Scene.import_file(tmp_path / "t.glb")
    tampered.get_by_name("Beacon").position = (0, 0, 0.56)
    failures = example.verify(scene, tampered)
    assert any("Beacon: local transform changed" in f for f in failures)
    assert any("Antenna: world matrix changed" in f for f in failures)


V010 = Path(__file__).resolve().parent / "data" / "v0.1.0" / "first_scene.aether"


def test_v010_project_still_loads_and_exports(tmp_path):
    """A project written by Aether3D 0.1.0 loads unchanged and exports to glTF."""
    scene = Scene.load(V010)
    assert [o.name for o in scene] == ["Cube", "Sphere", "Ground", "Pivot", "Moon"]
    scene.save(tmp_path / "resaved.aether")
    assert Scene.load(tmp_path / "resaved.aether").to_dict()["scene"] == scene.to_dict()["scene"]
    scene.export(tmp_path / "v010.glb")
    back = Scene.import_file(tmp_path / "v010.glb")
    assert [o.id for o in back] == [o.id for o in scene]
    # 0.1.0 spheres have a ~1e-16 seam gap that exact welding reports as open.
    sphere = scene.inspect_geometry().meshes[1]
    assert not sphere.topology["closed"] and sphere.status == "ok"


def test_minecraft_castle_example_runs(tmp_path):
    pytest.importorskip("manifold3d")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "minecraft_castle.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed." in result.stdout
    assert "towers: 6 (tower_count = 6)" in result.stdout
    assert "towers: 8 (tower_count = 8)" in result.stdout
    scene = Scene.load(tmp_path / "castle.aether")
    assert scene.procedural.parameters["tower_count"].value == 8
    assert all(o.is_procedural for o in scene)
    assert (tmp_path / "castle.glb").is_file()


def _decode(path):
    image = pytest.importorskip("PIL.Image")
    return np.asarray(image.open(path))


def test_first_render_example_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "first_render.py"),
            str(tmp_path),
            "--width",
            "160",
            "--height",
            "90",
            "--samples",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed." in result.stdout
    assert _decode(tmp_path / "aether3d_first_render.png").shape == (90, 160, 3)
    assert Scene.load(tmp_path / "first_render.aether").active_camera.name == "Camera"


def test_minecraft_castle_render_example_runs(tmp_path):
    pytest.importorskip("manifold3d")
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "minecraft_castle_render.py"),
            str(tmp_path),
            "--width",
            "320",
            "--height",
            "180",
            "--samples",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed." in result.stdout
    assert "reproduce the image exactly" in result.stdout
    assert _decode(tmp_path / "minecraft_castle_render.png").shape == (180, 320, 3)
    assert _decode(tmp_path / "castle_flat_vs_materials.png").shape == (90, 2 * 160 + 8, 3)
    # The project refers to the textures copied next to it by relative paths.
    project = Scene.load(tmp_path / "minecraft_castle_render.aether")
    paths = sorted(t.path for t in project.materials.textures if isinstance(t, ImageTexture))
    assert paths == [f"textures/{n}" for n in CASTLE_TEXTURES]
    assert [d.code for d in project.materials.validate() if d.severity != "info"] == []


def test_hero_render_example_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "hero_render.py"),
            str(tmp_path),
            "--width",
            "160",
            "--height",
            "90",
            "--samples",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed." in result.stdout
    assert _decode(tmp_path / "hero_flat_vs_materials.png").shape == (45, 2 * 80 + 8, 3)


def test_history_example_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "history_undo.py"), str(tmp_path), "--quick"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAILED" not in result.stdout and result.stdout.rstrip().endswith("done")
    assert "the render equals the first one, pixel for pixel" in result.stdout
    report = json.loads((tmp_path / "history_undo.json").read_text())
    assert len(report["steps"]) == 13 and report["problems"] == []
    assert (tmp_path / "history_undo.png").read_bytes().startswith(b"\x89PNG")


def test_castle_spatial_example_runs(tmp_path):
    pytest.importorskip("manifold3d")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "castle_spatial.py"), str(tmp_path), "--quick"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout and result.stdout.count("ok    ") == 15
    report = json.loads((tmp_path / "castle_spatial.json").read_text())
    assert report["problems"] == [] and report["describe"]["format"] == "aether3d.spatial-scene"
    assert report["pick"]["agreeing_with_the_rasterizer"] == report["pick"]["interior_samples"]
    for name in ("castle_spatial_view.png", "castle_spatial_heights.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_hero_picking_example_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "hero_picking.py"), str(tmp_path), "--quick"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout and result.stdout.count("ok    ") == 15
    report = json.loads((tmp_path / "hero_picking.json").read_text())
    assert report["problems"] == [] and report["clicks"]["sky"] is None
    assert report["id_map"]["agreeing"] == report["id_map"]["interior_samples"] > 500
    assert _decode(tmp_path / "hero_picking.png").shape[1] == 3 * 320 + 2 * 8


def test_example_textures_are_reproducible(tmp_path):
    """``examples/make_textures.py`` regenerates the committed textures exactly."""
    pytest.importorskip("PIL.Image")  # the JPEG
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "make_textures.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in (*CASTLE_TEXTURES, "uv_grid.png"):
        assert (tmp_path / name).read_bytes() == (EXAMPLES / "textures" / name).read_bytes(), name


def test_committed_renders_are_real_aether3d_pngs():
    """The PNGs in examples/output are Aether3D renders (they carry its Software tag)."""
    image = pytest.importorskip("PIL.Image")
    for name, size in (
        ("minecraft_castle_render.png", (1920, 1080)),
        ("aether3d_first_render.png", (1280, 720)),
        ("blocky_hero_render.png", (1280, 720)),
        ("castle_flat_vs_materials.png", (2 * 960 + 8, 540)),
        ("hero_flat_vs_materials.png", (2 * 640 + 8, 360)),
        ("plan_basics.png", (960, 540)),
        ("village.png", (1280, 720)),
        ("mcp_castle_agent.png", (960, 540)),
    ):
        with image.open(EXAMPLES / "output" / name) as png:
            assert png.size == size and png.info["Software"].startswith("Aether3D ")
    project = Scene.load(EXAMPLES / "output" / "minecraft_castle_render.aether")
    assert project.active_camera.name == "Camera" and project.render_settings.width == 1920
    # Its image textures are committed next to it and load.
    assert [d.code for d in project.materials.validate() if d.severity != "info"] == []


def test_gpu_render_example_runs(tmp_path, gpu):
    command = [sys.executable, str(EXAMPLES / "gpu_render.py"), str(tmp_path)]
    command += ["--width", "320", "--height", "180", "--require-gpu"]
    if gpu.adapter:
        command += ["--adapter", gpu.adapter]
    if importlib.util.find_spec("manifold3d") is None:
        pytest.skip("the castle uses booleans (manifold3d)")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed." in result.stdout
    assert "`aether3d render` reproduces them exactly" in result.stdout
    for name in ("castle_cpu", "castle_gpu", "castle_gpu_shadows", "hero_gpu", "primitives_gpu"):
        assert _decode(tmp_path / f"{name}.png").shape == (180, 320, 3)
    assert _decode(tmp_path / "castle_comparison.png").shape == (90, 3 * 160 + 16, 3)


def test_gpu_render_example_without_a_gpu_says_so(tmp_path):
    if importlib.util.find_spec("manifold3d") is None:
        pytest.skip("the castle uses booleans (manifold3d)")
    env = {**os.environ, "AETHER3D_GPU_ADAPTER": "no adapter has this name"}
    command = [sys.executable, str(EXAMPLES / "gpu_render.py"), str(tmp_path)]
    command += ["--width", "160", "--height", "90"]
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU rendering is NOT available" in result.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["castle_cpu_only.png"]
    required = subprocess.run(
        [*command, "--require-gpu"], capture_output=True, text=True, check=False, env=env
    )
    assert required.returncode == 1


def test_committed_gpu_renders_record_their_adapter():
    """The GPU images in examples/output were drawn by Aether3D's GPU backend (Source chunk)."""
    image = pytest.importorskip("PIL.Image")
    for name, size in (
        ("castle_gpu.png", (1920, 1080)),
        ("castle_gpu_shadows.png", (1920, 1080)),
        ("hero_gpu.png", (1920, 1080)),
        ("primitives_gpu.png", (1920, 1080)),
    ):
        with image.open(EXAMPLES / "output" / name) as png:
            assert png.size == size and png.info["Software"].startswith("Aether3D ")
            assert png.info["Source"].startswith("Aether3D GPU renderer (wgpu, ")
    with image.open(EXAMPLES / "output" / "castle_cpu.png") as png:
        assert "Source" not in png.info  # the CPU reference says so by omission
    with image.open(EXAMPLES / "output" / "castle_comparison.png") as png:
        assert png.size == (3 * 960 + 16, 540)


# --------------------------------------------------------------------------- #
# Stage 11: scene plans and the MCP server
# --------------------------------------------------------------------------- #


def _run_example(name: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    command = [sys.executable, str(EXAMPLES / name), *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAILED" not in result.stdout and result.stdout.rstrip().endswith("done")
    return result


def test_plan_basics_example_runs(tmp_path):
    _run_example("plan_basics.py", str(tmp_path), "--quick")
    report = json.loads((tmp_path / "plan_basics.json").read_text())
    assert report["problems"] == [] and report["built"]["objects"] == 161
    assert [e["code"] for e in report["errors"]] == [
        "INVALID_VALUE", "OUT_OF_RANGE", "INVALID_VALUE", "UNKNOWN_REFERENCE"
    ]  # fmt: skip
    assert report["history"] == [
        "Market square", "Close two stalls, paint the lamps, fell two trees", "Benches"
    ]  # fmt: skip
    assert _decode(tmp_path / "plan_basics.png").shape == (180, 320, 3)


def test_village_example_runs(tmp_path):
    _run_example("village.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "village.json").read_text())
    assert report["problems"] == [] and report["build"]["objects"] == 11270
    assert report["edit"]["changed"] == ["Block 3-0/houses 4/door"]
    assert report["render"]["backend"] == "cpu" and report["render"]["draw_calls"] is None
    assert _decode(tmp_path / "village.png").shape == (180, 320, 3)


def test_village_example_on_the_gpu(tmp_path, gpu):
    env = {**os.environ, **({"AETHER3D_GPU_ADAPTER": gpu.adapter} if gpu.adapter else {})}
    _run_example("village.py", str(tmp_path), "--quick", "--backend", "gpu", env=env)
    report = json.loads((tmp_path / "village.json").read_text())
    assert report["render"]["backend"] == "gpu"
    assert report["render"]["draw_calls"] <= 20  # one instanced draw call per mesh, at most


def test_mcp_client_example_runs(tmp_path):
    pytest.importorskip("mcp")
    result = _run_example("mcp_client.py", str(tmp_path), "--backend", "cpu")
    assert "26 tools" in result.stdout and "on the CPU: preview.png" in result.stdout
    assert (tmp_path / "preview.png").read_bytes().startswith(b"\x89PNG")
    assert len(Scene.load(tmp_path / "courtyard.aether")) == 44


def test_mcp_castle_agent_example_runs(tmp_path):
    pytest.importorskip("mcp")
    _run_example("mcp_castle_agent.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "mcp_castle_agent.json").read_text())
    assert report["problems"] == [] and report["tool_calls"] == 14
    found = {c["tool"]: c.get("found") for c in report["calls"] if c.get("found")}
    assert found["spatial_query"]["trees_in_the_keep"] == ["Tree 1"]
    assert found["execute_scene_plan"] == {"trees_in_the_keep": [], "floating_towers": {}}
    assert found["render_preview"]["device"] == "cpu"
    workspace = tmp_path / "mcp_castle_workspace"
    assert (workspace / "castle.glb").read_bytes()[:4] == b"glTF"
    assert len(Scene.load(workspace / "castle.aether")) == 2190
    assert _decode(tmp_path / "mcp_castle_agent.png").shape == (180, 320, 3)


def test_feedback_room_example_runs(tmp_path):
    _run_example("feedback_room.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "feedback_room.json").read_text())
    assert report["failures"] == [] and report["loop"]["stop_reason"] == "accepted"
    before = {f["code"] for f in report["before"]["findings"]}
    assert {"FLOATING", "SUNKEN", "INTERSECTION"} <= before
    assert report["after"]["accepted"] and report["renderer"]["device"] == "cpu"
    assert _decode(tmp_path / "feedback_room.png").shape == (270, 968, 3)


def test_feedback_composition_example_runs(tmp_path):
    _run_example("feedback_composition.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "feedback_composition.json").read_text())
    assert report["failures"] == [] and report["loop"]["stop_reason"] == "accepted"
    assert {f["code"] for f in report["findings"]} >= {"OFF_CENTER", "UNDEREXPOSED"}
    assert report["before"]["mean_luminance"] < 0.12 <= report["after"]["mean_luminance"]
    assert report["before"]["monument_fill"] < 0.06 <= report["after"]["monument_fill"]


def test_mcp_feedback_agent_example_runs(tmp_path):
    pytest.importorskip("mcp")
    _run_example("mcp_feedback_agent.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "mcp_feedback_agent.json").read_text())
    assert report["failures"] == []
    codes = {(code, basis) for code, basis, _ in report["first_diagnosis"]["findings"]}
    assert ("VISION_FLOATING", "agent") in codes  # scripted, labeled as no visual observation
    assert report["final"]["accepted"]
    assert len(Scene.load(tmp_path / "mcp_feedback_workspace" / "castle.aether")) == 2190


def test_feedback_room_example_on_the_gpu(tmp_path, gpu):
    env = {**os.environ, **({"AETHER3D_GPU_ADAPTER": gpu.adapter} if gpu.adapter else {})}
    _run_example("feedback_room.py", str(tmp_path), "--quick", "--backend", "gpu", env=env)
    report = json.loads((tmp_path / "feedback_room.json").read_text())
    assert report["failures"] == [] and report["renderer"]["device"] == "gpu"


def test_committed_feedback_outputs():
    """The visual feedback examples' committed results: Aether3D renders and reports."""
    image = pytest.importorskip("PIL.Image")
    for name, size in (
        ("feedback_room.png", (2 * 960 + 8, 540)),
        ("feedback_composition.png", (2 * 960 + 8, 540)),
        ("mcp_feedback_agent.png", (2 * 960 + 8, 540)),
    ):
        with image.open(EXAMPLES / "output" / name) as png:
            assert png.size == size and png.info["Software"].startswith("Aether3D ")
    for name in ("feedback_room", "feedback_composition", "mcp_feedback_agent"):
        report = json.loads((EXAMPLES / "output" / f"{name}.json").read_text())
        assert report["failures"] == [] and re.fullmatch(r"\d+\.\d+\.\d+", report["aether3d"])


# --------------------------------------------------------------------------- #
# Stage 13: modeling and sculpting
# --------------------------------------------------------------------------- #


def test_hard_surface_example_runs(tmp_path):
    _run_example("hard_surface.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "hard_surface.json").read_text())
    assert report["failures"] == []
    assert {p["tier"] for p in report["parts"].values()} == {"solid"}
    used = {op["operation"] for op in report["operations"]}
    assert {"bevel_edges", "knife_project", "bridge_edge_loops", "bisect", "spin"} <= used
    assert _decode(tmp_path / "hard_surface.png").shape == (300, 800, 3)


def test_organic_sculpt_example_runs(tmp_path):
    _run_example("organic_sculpt.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "organic_sculpt.json").read_text())
    assert report["failures"] == [] and report["sculpt"]["tier"] == "solid"
    assert report["symmetry_error"]["max"] == 0.0  # mirrored, so exactly symmetric
    assert _decode(tmp_path / "organic_sculpt.png").shape == (320, 960, 3)


def test_remesh_validity_example_runs(tmp_path):
    _run_example("remesh_validity.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "remesh_validity.json").read_text())
    stages = report["stages"]
    assert report["failures"] == []
    assert stages["input"]["self_intersecting_pairs"] > 0 and stages["input"]["components"] == 3
    assert stages["voxel"]["tier"] == stages["isotropic"]["tier"] == "solid"
    assert stages["isotropic"]["min_angle_degrees"] > 15
    assert _decode(tmp_path / "remesh_validity.png").shape == (300, 1080, 3)


def test_mcp_sculpt_agent_example_runs(tmp_path):
    pytest.importorskip("mcp")
    _run_example("mcp_sculpt_agent.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "mcp_sculpt_agent.json").read_text())
    assert report["failures"] == [] and report["final"]["accepted"]
    assert report["refused"]["code"] == "VOXEL_NOT_WATERTIGHT"
    assert report["history"][1] == "Sculpt the bust"


def test_committed_sculpting_outputs():
    """The Stage 13 examples' committed results: Aether3D renders and reports."""
    image = pytest.importorskip("PIL.Image")
    for name, size in (
        ("hard_surface.png", (1600, 600)),
        ("organic_sculpt.png", (1920, 640)),
        ("remesh_validity.png", (2160, 600)),
        ("mcp_sculpt_agent.png", (720 + 8 + 720, 720)),
    ):
        with image.open(EXAMPLES / "output" / name) as png:
            assert png.size == size and png.info["Software"].startswith("Aether3D ")
    for name in ("hard_surface", "organic_sculpt", "remesh_validity", "mcp_sculpt_agent"):
        report = json.loads((EXAMPLES / "output" / f"{name}.json").read_text())
        assert report["failures"] == [] and re.fullmatch(r"\d+\.\d+\.\d+", report["aether3d"])


def test_texturing_hard_surface_example_runs(tmp_path):
    _run_example("texturing_hard_surface.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "texturing_hard_surface.json").read_text())
    assert report["failures"] == []
    densities = {round(v["texel_density"]["mean"], 3) for v in report["uv"].values()}
    assert len(densities) == 1  # one texel density on every part
    assert all(v["overlapping_triangles"] == 0 for v in report["uv"].values())
    assert _decode(tmp_path / "texturing_hard_surface.png").shape == (600, 900, 3)


def test_texturing_organic_paint_example_runs(tmp_path):
    _run_example("texturing_organic_paint.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "texturing_organic_paint.json").read_text())
    assert report["failures"] == [] and report["spot_symmetry_alpha_difference"] < 0.1
    assert report["uv"]["flipped_triangles"] == 0
    assert _decode(tmp_path / "texturing_organic_paint.png").shape == (320, 1280, 3)


def test_texturing_bake_example_runs(tmp_path):
    _run_example("texturing_bake.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "texturing_bake.json").read_text())
    assert report["failures"] == []
    assert report["rms_to_high"]["low_with_maps"] < report["rms_to_high"]["low"]
    assert _decode(tmp_path / "texturing_bake.png").shape == (640, 960, 3)


def test_texturing_pbr_example_runs(tmp_path):
    _run_example("texturing_pbr.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "texturing_pbr.json").read_text())
    assert report["failures"] == [] and set(report["renders"]) >= {"cpu", "path"}
    assert _decode(tmp_path / "texturing_pbr.png").shape == (480, 960, 3)


def test_mcp_texture_agent_example_runs(tmp_path):
    pytest.importorskip("mcp")
    _run_example("mcp_texture_agent.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "mcp_texture_agent.json").read_text())
    assert report["failures"] == [] and report["final"]["accepted"]
    assert report["refused"]["code"] == "TEXTURE_NOT_FOUND"
    assert ("UV_OVERLAP", "geometry") in {tuple(f[:2]) for f in report["diagnosis"]["findings"]}


def test_committed_texturing_outputs():
    """The Stage 14 examples' committed results: Aether3D renders and reports."""
    image = pytest.importorskip("PIL.Image")
    for name, size in (
        ("texturing_hard_surface.png", (1800, 1200)),
        ("texturing_organic_paint.png", (2560, 640)),
        ("texturing_bake.png", (1920, 1280)),
        ("texturing_pbr.png", (1920, 960)),
        ("mcp_texture_agent.png", (2160, 720)),
    ):
        with image.open(EXAMPLES / "output" / name) as png:
            assert png.size == size and png.info["Software"].startswith("Aether3D ")
    for name in ("texturing_hard_surface", "texturing_organic_paint", "texturing_bake",
                 "texturing_pbr", "mcp_texture_agent"):  # fmt: skip
        report = json.loads((EXAMPLES / "output" / f"{name}.json").read_text())
        assert report["failures"] == [] and re.fullmatch(r"\d+\.\d+\.\d+", report["aether3d"])


# --------------------------------------------------------------------------- #
# Precision modeling, CAD and architecture (0.15.0)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "checks"),
    [
        ("cad_bracket.py", lambda r: r["edited"]["valid"] and r["edit"]["evaluations"] > 0),
        ("cad_floor_plan.py", lambda r: r["wall_volume_max_difference_m3"] <= 1e-9),
        ("cad_modern_house.py", lambda r: r["headroom_m"] >= 2.0),
        ("cad_grid_building.py", lambda r: r["edit_door"]["evaluations"] == 1),
    ],
)
def test_cad_examples_run(tmp_path, name, checks):
    pytest.importorskip("manifold3d")
    _run_example(name, str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / name.replace(".py", ".json")).read_text())
    assert report["failures"] == [] and checks(report)


def test_mcp_cad_agent_example_runs(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("manifold3d")
    _run_example("mcp_cad_agent.py", str(tmp_path), "--quick", "--backend", "cpu")
    report = json.loads((tmp_path / "mcp_cad_agent.json").read_text())
    assert report["failures"] == [] and report["refused"]["code"] == "SKETCH_CONFLICT"
    assert report["diagnosis"]["accepted"]


def test_committed_cad_outputs():
    """The Stage 15 examples' committed results: Aether3D renders, drawings and reports."""
    image = pytest.importorskip("PIL.Image")
    for name in ("cad_bracket", "cad_floor_plan", "cad_modern_house", "cad_grid_building",
                 "mcp_cad_agent"):  # fmt: skip
        with image.open(EXAMPLES / "output" / f"{name}.png") as png:
            assert png.info["Software"].startswith("Aether3D ")
        report = json.loads((EXAMPLES / "output" / f"{name}.json").read_text())
        assert report["failures"] == [] and re.fullmatch(r"\d+\.\d+\.\d+", report["aether3d"])
