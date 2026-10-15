"""Exported files pass the official Khronos glTF Validator.

The validator is a Node.js package pinned in ``tools/gltf-validator``. These
tests are skipped when Node.js or the package is not installed, unless the
environment variable ``AETHER3D_REQUIRE_GLTF_VALIDATOR=1`` is set (CI sets it),
in which case a missing validator is a failure.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from scenes import challenging_scene

from aether3d import Mesh, Scene, import_gltf

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "gltf-validator"
KHRONOS = Path(__file__).parent / "data" / "khronos"


def validator_available() -> bool:
    return shutil.which("node") is not None and (TOOL / "node_modules" / "gltf-validator").is_dir()


if not validator_available():
    if os.environ.get("AETHER3D_REQUIRE_GLTF_VALIDATOR") == "1":
        pytest.fail(
            "Khronos glTF Validator required but not installed "
            "(run `npm ci --prefix tools/gltf-validator`)",
            pytrace=False,
        )
    pytest.skip("Khronos glTF Validator not installed", allow_module_level=True)


def validate(*paths: Path) -> list[dict]:
    result = subprocess.run(
        ["node", str(TOOL / "validate.mjs"), *map(str, paths)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    reports = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(reports) == len(paths), result.stderr
    return reports


def corpus(directory: Path) -> list[Path]:
    """A set of exports covering the exporter's features and edge cases."""
    scenes = {"challenging": challenging_scene(), "empty": Scene("Empty")}
    single = Scene("Single")
    single.add_object("Lonely Empty")
    scenes["empty_node"] = single
    tri = Scene("Custom")
    tri.add_object(
        "Tri", mesh=Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]), scale=(-1, 2, 0.5)
    )
    scenes["no_attributes_mirrored"] = tri
    tagged = Scene("Tagged")  # tags and metadata in node extras (0.10.0)
    keep = tagged.add_cube("Keep")
    keep.tags = ["castle", "stone"]
    keep.metadata = {"floors": 3, "doors": [{"side": "south", "open": True}], "note": "naïve ✓"}
    tagged.add_sphere("Ball", parent=keep).set_metadata("weight", None)
    scenes["tags_and_metadata"] = tagged
    paths = []
    for name, scene in scenes.items():
        for fmt in ("glb", "gltf"):
            scene.export(directory / f"{name}.{fmt}")
            paths.append(directory / f"{name}.{fmt}")
    # Khronos samples imported and re-exported by Aether3D.
    for sample in KHRONOS.rglob("*.gl*"):
        out = directory / f"reexport_{sample.parent.parent.name}_{sample.parent.name}.glb"
        import_gltf(sample, allow_lossy=True).scene.export(out, allow_lossy=True)
        paths.append(out)
    return paths


def test_exports_have_no_errors_or_warnings(tmp_path):
    paths = corpus(tmp_path)
    reports = validate(*paths)
    problems = [r for r in reports if not r["ok"]]
    assert problems == [], json.dumps(problems, indent=2)
    for report in reports:
        assert report["errors"] == 0 and report["warnings"] == 0
        assert report["validatorVersion"] == "2.0.0-dev.3.10"


def test_validator_detects_broken_files(tmp_path):
    """Sanity check that the integration really reports errors."""
    path = tmp_path / "broken.glb"
    challenging_scene().export(path)
    data = bytearray(path.read_bytes())
    data[-40:] = b"\xff" * 40  # corrupt the tail of the index/vertex data
    path.write_bytes(bytes(data))
    (report,) = validate(path)
    assert not report["ok"] and report["errors"] > 0
