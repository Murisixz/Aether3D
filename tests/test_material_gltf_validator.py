"""glTF files with materials and textures pass the official Khronos glTF Validator.

Skipped without Node.js and the validator package (``tools/gltf-validator``)
unless ``AETHER3D_REQUIRE_GLTF_VALIDATOR=1`` (CI), like
``test_khronos_validator.py``.
"""

import os
from pathlib import Path

import pytest
from test_khronos_validator import validate, validator_available
from test_material_gltf import textured_scene

from aether3d import Scene, import_gltf

KHRONOS = Path(__file__).parent / "data" / "khronos"

if not validator_available():
    if os.environ.get("AETHER3D_REQUIRE_GLTF_VALIDATOR") == "1":
        pytest.fail("Khronos glTF Validator required but not installed", pytrace=False)
    pytest.skip("Khronos glTF Validator not installed", allow_module_level=True)


def test_material_exports_are_valid(tmp_path):
    paths = []
    scene = textured_scene(tmp_path)
    for fmt in ("glb", "gltf"):
        scene.export(tmp_path / f"materials.{fmt}")
        paths.append(tmp_path / f"materials.{fmt}")
    baked = Scene("Baked")
    plane = baked.add_plane("P", size=2)
    m = baked.create_material("Procedural", preset="brick", coordinates="uv")
    plane.material = m
    baked.export(tmp_path / "baked.glb", bake_textures=64)
    paths.append(tmp_path / "baked.glb")
    for sample in ("TextureCoordinateTest", "TextureSettingsTest"):
        source = import_gltf(KHRONOS / sample / "glTF-Binary" / f"{sample}.glb").scene
        source.export(tmp_path / f"{sample}.glb")
        paths.append(tmp_path / f"{sample}.glb")
    for path, report in zip(paths, validate(*paths), strict=True):
        assert report["ok"] and report["errors"] == 0 and report["warnings"] == 0, (
            path.name,
            report,
        )
