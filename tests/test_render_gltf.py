"""Scenes with cameras, lights and colors still export valid glTF (Khronos validator).

Cameras and lights are exported as plain nodes that keep their transforms (glTF
cameras and KHR_lights_punctual are not written yet; see docs/RENDERING.md).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene, import_gltf

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "gltf-validator"

if not (shutil.which("node") and (TOOL / "node_modules" / "gltf-validator").is_dir()):
    if os.environ.get("AETHER3D_REQUIRE_GLTF_VALIDATOR") == "1":
        pytest.fail("Khronos glTF Validator required but not installed", pytrace=False)
    pytest.skip("Khronos glTF Validator not installed", allow_module_level=True)


def render_scene() -> Scene:
    scene = Scene("Lit")
    scene.add_cube("Crate", size=2).color = "#c0392b"
    rig = scene.add_object("Rig", position=(1, -1, 0), scale=(2, 1, 1))
    scene.add_camera("Main", parent=rig, position=(4, -6, 3), look_at=(0, 0, 0))
    scene.add_directional_light("Sun", direction=(-0.3, 0.6, -1))
    scene.add_point_light("Bulb", position=(2, -2, 3), range=10)
    return scene


def test_scenes_with_cameras_and_lights_pass_the_khronos_validator(tmp_path):
    scene = render_scene()
    paths = [scene.export(tmp_path / f"lit.{fmt}").path for fmt in ("glb", "gltf")]
    out = subprocess.run(
        ["node", str(TOOL / "validate.mjs"), *map(str, paths)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    reports = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    assert len(reports) == 2, out.stderr
    for report in reports:
        assert report["ok"] and report["errors"] == 0 and report["warnings"] == 0, report
    back = import_gltf(paths[0]).scene
    for name in ("Main", "Sun", "Bulb"):  # nodes keep their (world) placement
        assert np.allclose(
            back.get_by_name(name).world_matrix, scene.get_by_name(name).world_matrix
        )
