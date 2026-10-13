import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


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


def test_committed_renders_are_real_aether3d_pngs():
    """The PNGs in examples/output are Aether3D renders (they carry its Software tag)."""
    image = pytest.importorskip("PIL.Image")
    for name, size in (
        ("minecraft_castle_render.png", (1920, 1080)),
        ("aether3d_first_render.png", (1280, 720)),
        ("blocky_hero_render.png", (1280, 720)),
    ):
        with image.open(EXAMPLES / "output" / name) as png:
            assert png.size == size and png.info["Software"].startswith("Aether3D ")
    project = Scene.load(EXAMPLES / "output" / "minecraft_castle_render.aether")
    assert project.active_camera.name == "Camera" and project.render_settings.width == 1920
