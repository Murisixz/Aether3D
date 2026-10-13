"""Edited meshes export to glTF that the official Khronos validator accepts.

Uses the pinned validator in ``tools/gltf-validator`` (see test_khronos_validator.py
for how it is installed and when it is required).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aether3d import Scene, create_sphere, import_gltf, inspect_mesh

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "gltf-validator"

if not (shutil.which("node") and (TOOL / "node_modules" / "gltf-validator").is_dir()):
    if os.environ.get("AETHER3D_REQUIRE_GLTF_VALIDATOR") == "1":
        pytest.fail("Khronos glTF Validator required but not installed", pytrace=False)
    pytest.skip("Khronos glTF Validator not installed", allow_module_level=True)


def edited_scene() -> Scene:
    scene = Scene("Edited")
    a = scene.add_cube("Bevelled", size=2)
    with a.edit_mesh() as edit:
        edit.bevel_edges(edit.select_all(), 0.2)
    b = scene.add_cube("Inset Extrude", size=2, position=(4, 0, 0))
    with b.edit_mesh() as edit:
        top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
        edit.inset_faces(top, 0.3)
        edit.extrude_faces(top, 1.0)
        edit.loop_cut(next(iter(edit.select_edges(nearest=(1, 1, 0)).edges)), cuts=2)
    c = scene.add_object("Mirrored Copy", mesh=b.mesh, position=(-4, 0, 0), scale=(-1, 1, 1))
    with c.edit_mesh() as edit:
        edit.scale(edit.select_all(), (-1, 1, 1))  # mirror the geometry back
    d = scene.add_sphere("Subdivided", radius=1, segments=12, rings=6, position=(0, 4, 0))
    with d.edit_mesh() as edit:
        edit.subdivide_faces(edit.select_faces(sides=4), 1)
        edit.set_smooth(edit.select_all())
    e = scene.add_object("From Scratch", position=(0, -4, 0))
    with e.edit_mesh() as edit:
        edit.add_grid(3, 3, 2)
        edit.extrude_faces(edit.select_all(), 0.5)
        edit.triangulate_faces(edit.select_faces(sides=4, box=((-2, -2, 0.4), (2, 2, 0.6))))
    return scene


def test_edited_meshes_pass_khronos_validator(tmp_path):
    scene = edited_scene()
    for obj in scene:
        if obj.mesh is not None and obj.name != "From Scratch":
            assert inspect_mesh(obj.mesh).topology["volume"] > 0, obj.name
    paths = []
    for fmt in ("glb", "gltf"):
        result = scene.export(tmp_path / f"edited.{fmt}")
        assert result.diagnostics == []
        paths.append(result.path)
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
    back = import_gltf(paths[0])
    assert back.diagnostics == [] and len(back.scene) == len(scene)


def test_smooth_sphere_keeps_analytic_normals_in_export(tmp_path):
    scene = Scene()
    s = scene.add_sphere("S", radius=1, segments=16, rings=8)
    with s.edit_mesh() as edit:
        edit.subdivide_faces(edit.select_faces(sides=4), 1)
    original = {
        tuple(p): tuple(n)
        for p, n in zip(
            create_sphere(1, 16, 8).positions, create_sphere(1, 16, 8).normals, strict=True
        )
    }
    kept = sum(
        1
        for p, n in zip(s.mesh.positions, s.mesh.normals, strict=True)
        if original.get(tuple(p)) == tuple(n)
    )
    assert kept >= len(original) - 2  # every original vertex (except unused pole copies)
