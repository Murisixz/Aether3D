"""Procedural scenes export evaluated geometry that the official Khronos validator accepts.

Uses the pinned validator in ``tools/gltf-validator`` (see test_khronos_validator.py
for how it is installed and when it is required).
"""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aether3d import Expr, Scene, import_gltf

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "gltf-validator"

if not (shutil.which("node") and (TOOL / "node_modules" / "gltf-validator").is_dir()):
    if os.environ.get("AETHER3D_REQUIRE_GLTF_VALIDATOR") == "1":
        pytest.fail("Khronos glTF Validator required but not installed", pytrace=False)
    pytest.skip("Khronos glTF Validator not installed", allow_module_level=True)

try:
    import manifold3d  # noqa: F401

    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


def validate(*paths):
    out = subprocess.run(
        ["node", str(TOOL / "validate.mjs"), *map(str, paths)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    reports = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    assert len(reports) == len(paths), out.stderr
    return reports


def modifier_scene() -> Scene:
    """Every modifier type (booleans when available), shared and mirrored instances."""
    scene = Scene("Modifiers")
    scene.procedural.add_parameter("n", 3, min=1, max=10)
    rows = scene.add_geometry("Rows", "box", width=0.8, depth=0.8, height=0.8)
    rows.add_modifier("array", count=Expr("n"), relative_offset=(1.5, 0, 0))
    rows.add_modifier("mirror", axes="y", origin=(0, 1, 0))
    slab = scene.add_geometry("Slab", "grid", x_size=4, y_size=4, x_segments=4, y_segments=4)
    slab.add_modifier("solidify", thickness=0.2, even_thickness=True)
    blob = scene.add_geometry("Blob", "torus", major_segments=12, minor_segments=8)
    blob.add_modifier("subdivision", levels=2)
    chamfer = scene.add_geometry("Chamfer", "cylinder", segments=12, smooth=False)
    chamfer.add_modifier("bevel", width=0.05, limit="angle", angle=60)
    chamfer.add_modifier("transform", rotation=(30, 0, 45), scale=(1, -1, 1))
    if HAS_MANIFOLD:
        hole = scene.add_geometry("Hole", "uv_sphere", radius=0.65, segments=16, rings=8)
        carved = scene.add_geometry("Carved", "box")
        carved.add_modifier("boolean", operation="difference", operand={"geometry": hole})
        scene.add_object("Carved", geometry=carved, position=(0, -4, 0))
    scene.add_object("Rows", geometry=rows)
    scene.add_object("Slab", geometry=slab, position=(0, 6, 0))
    scene.add_object("Blob", geometry=blob, position=(6, 0, 0))
    scene.add_object("Blob Mirror", geometry=blob, position=(-6, 0, 0), scale=(-1, 1, 1))
    scene.add_object("Chamfer", geometry=chamfer, position=(0, 0, 4))
    return scene


def test_procedural_exports_pass_khronos_validator(tmp_path):
    scene = modifier_scene()
    paths = []
    for fmt in ("glb", "gltf"):
        result = scene.export(tmp_path / f"procedural.{fmt}")
        assert result.diagnostics == []
        paths.append(result.path)
    for report in validate(*paths):
        assert report["ok"] and report["errors"] == 0 and report["warnings"] == 0, report
    back = import_gltf(paths[0]).scene
    for obj in scene:
        assert np.array_equal(back.get(obj.id).mesh.positions, obj.mesh.positions), obj.name
    # Shared geometry is written once.
    assert back.get_by_name("Blob").mesh is back.get_by_name("Blob Mirror").mesh


@pytest.mark.skipif(not HAS_MANIFOLD, reason="the castle uses booleans (manifold3d)")
def test_castle_passes_khronos_validator(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "minecraft_castle", ROOT / "examples" / "minecraft_castle.py"
    )
    castle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(castle)
    scene = Scene("Castle")
    castle.build_castle(scene)
    paths = [scene.export(tmp_path / "castle.glb").path]
    scene.procedural.set_parameters({"tower_count": 10, "castle_width": 40.0, "detail_density": 3})
    castle.place_watchtower(scene)
    paths.append(scene.export(tmp_path / "castle_big.gltf").path)
    for report in validate(*paths):
        assert report["ok"] and report["errors"] == 0 and report["warnings"] == 0, report
