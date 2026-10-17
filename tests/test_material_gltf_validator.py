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


def test_painted_and_baked_exports_are_valid(tmp_path):
    """MikkTSpace TANGENT (vertices split at UV seams), painted textures and baked maps
    bound through channels glTF repacks (0.14.0) pass the validator with no warning."""
    from aether3d import create_sphere

    scene = Scene("Painted")
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 32, 16))
    ball.unwrap(resolution=128, padding=4)
    scene.add_object("Detail", mesh=create_sphere(1.0, 96, 48), position=(0, 0, 0))
    scene.get_by_name("Detail").visible = False
    color = scene.add_painted_texture("Color", 128, 128, background="#3d6fb5")
    color.add_layer("Ink")
    rough = scene.add_painted_texture("Rough", 128, 128, encoding="data",
                                      background=[0.5, 0.5, 0.5, 1])  # fmt: skip
    rough.add_layer("Ink")
    m = scene.create_material("M", base_color="#ffffff", roughness=1.0)
    m.set_texture("base_color", color)
    m.set_texture("roughness", rough, channel="r")
    ball.material = m
    ball.paint([{"texture": color, "color": "#ff8800"}, {"texture": rough, "color": [0.9]}],
               {"radius": 0.4}, [[1, 0, 0], [0, 1, 0]])  # fmt: skip
    scene.bake(ball, ["Detail"], ["normal", "ao", "metallic"], resolution=128, apply=True,
               backend="cpu")  # fmt: skip
    paths = []
    for fmt in ("glb", "gltf"):
        result = scene.export(tmp_path / f"painted.{fmt}")
        assert "TEXTURE_CHANNELS_PACKED" in {d.code for d in result.diagnostics}
        paths.append(tmp_path / f"painted.{fmt}")
    for path, report in zip(paths, validate(*paths), strict=True):
        assert report["ok"] and report["errors"] == 0 and report["warnings"] == 0, (
            path.name,
            report,
        )
