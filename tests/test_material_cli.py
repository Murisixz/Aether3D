"""``aether3d material`` and ``aether3d texture``: atomic edits, JSON output, codes, safety."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_cli import run, run_json
from test_materials import unevaluable_project

from aether3d import Scene
from aether3d.cli import EXIT_FAILURE, EXIT_GEOMETRY_ERRORS, EXIT_OK
from aether3d.rendering.png import write_png


@pytest.fixture
def project(tmp_path):
    scene = Scene("Materials CLI")
    scene.add_cube("Box", size=2, position=(0, 0, 1))
    scene.add_plane("Ground", size=8)
    scene.add_camera("Cam", position=(5, -7, 5), look_at=(0, 0, 0.5))
    scene.add_directional_light("Sun", direction=(0.3, 0.5, -1), intensity=2)
    path = tmp_path / "project.aether"
    scene.save(path)
    (tmp_path / "textures").mkdir()
    pixels = np.zeros((8, 8, 4), np.uint8)
    pixels[..., 0] = 220
    pixels[::2, ..., 1] = 180
    pixels[..., 3] = 255
    write_png(tmp_path / "textures" / "stripes.png", pixels)
    return path


def test_create_assign_texture_and_list(project, capsys):
    code, data = run_json(
        capsys,
        "material",
        "create",
        str(project),
        "Bricks",
        "--preset",
        "brick",
        "--set",
        "scale=2",
    )
    assert code == EXIT_OK and data["operations"][0]["textures"] == [
        "base_color",
        "normal",
        "roughness",
    ]
    code, data = run_json(
        capsys,
        "material",
        "create",
        str(project),
        "Paint",
        "--set",
        "roughness=0.4",
        "--set",
        'base_color="#336699"',
    )
    assert code == EXIT_OK
    image = str(project.parent / "textures" / "stripes.png")
    code, data = run_json(
        capsys,
        "material",
        "texture",
        str(project),
        "Paint",
        "base_color",
        "--image",
        image,
        "--scale",
        "4",
        "2",
        "--wrap",
        "clamp",
        "mirror",
    )
    assert code == EXIT_OK
    result = data["operations"][0]
    assert result["texture"]["type"] == "image" and result["binding"]["scale"] == [4.0, 2.0, 1.0]
    assert run_json(capsys, "material", "assign", str(project), "Bricks", "Box")[0] == EXIT_OK
    assert (
        run_json(capsys, "material", "assign", str(project), "Paint", "Ground", "--group", "0")[0]
        == EXIT_OK
    )
    code, listing = run_json(capsys, "material", "list", str(project))
    users = {m["name"]: m["users"] for m in listing["materials"]}
    assert users == {
        "Bricks": [{"id": users["Bricks"][0]["id"], "name": "Box"}],
        "Paint": [{"id": users["Paint"][0]["id"], "name": "Ground", "group": 0}],
    }
    scene = Scene.load(project)
    texture = scene.materials.texture("stripes")
    assert texture.wrap == ("clamp", "mirror") and texture.path == "textures/stripes.png"
    code, out, _ = run(capsys, "material", "list", str(project))
    assert "Bricks (standard)" in out and "stripes (image)" in out


def test_set_inspect_validate_and_remove(project, capsys):
    run_json(capsys, "material", "create", str(project), "M")
    code, data = run_json(
        capsys, "material", "set", str(project), "M", "--set", "metallic=1", "--rename", "Metal"
    )
    assert code == EXIT_OK and data["operations"][0]["parameters"]["metallic"] == 1.0
    code, data = run_json(capsys, "material", "inspect", str(project), "Metal")
    assert data["material"]["name"] == "Metal" and data["material"]["users"] == []
    code, data = run_json(capsys, "material", "validate", str(project))
    assert code == EXIT_OK and [d["code"] for d in data["diagnostics"]] == ["UNUSED_MATERIAL"]
    run_json(capsys, "material", "assign", str(project), "Metal", "Box")
    code, data = run_json(capsys, "material", "remove", str(project), "Metal")
    assert code == EXIT_FAILURE and data["error"]["code"] == "MATERIAL_IN_USE"
    code, data = run_json(capsys, "material", "remove", str(project), "Metal", "--force")
    assert code == EXIT_OK and Scene.load(project).materials.materials == ()
    assert Scene.load(project).get_by_name("Box").material is None


def test_missing_textures_make_validation_fail_with_exit_3(project, capsys):
    image = project.parent / "textures" / "stripes.png"
    run_json(capsys, "texture", "add", str(project), "Stripes", str(image))
    run_json(capsys, "material", "create", str(project), "M")
    run_json(capsys, "material", "texture", str(project), "M", "base_color", "--texture", "Stripes")
    run_json(capsys, "material", "assign", str(project), "M", "Box")
    image.unlink()
    code, data = run_json(capsys, "material", "validate", str(project))
    assert code == EXIT_GEOMETRY_ERRORS and data["status"] == "error"
    assert [d["code"] for d in data["diagnostics"] if d["severity"] == "error"] == [
        "TEXTURE_MISSING"
    ]
    code, data = run_json(
        capsys,
        "render",
        str(project),
        "-o",
        str(project.parent / "r.png"),
        "--width",
        "40",
        "--height",
        "30",
    )
    assert code == EXIT_OK and data["status"] == "warning"
    assert "TEXTURE_MISSING" in [d["code"] for d in data["diagnostics"]]


def test_failed_commands_change_nothing(project, capsys):
    before = project.read_bytes()
    code, data = run_json(capsys, "material", "create", str(project), "Bad", "--set", "roughness=3")
    assert code == EXIT_FAILURE and data["error"]["code"] == "INVALID_MATERIAL"
    assert data["error"]["location"].endswith("roughness")
    code, data = run_json(
        capsys, "material", "texture", str(project), "Nope", "base_color", "--clear"
    )
    assert code == EXIT_FAILURE and data["error"]["code"] == "MATERIAL_NOT_FOUND"
    code, data = run_json(capsys, "material", "assign", str(project), "Nope", "Box")
    assert code == EXIT_FAILURE
    code, data = run_json(capsys, "texture", "add", str(project), "T", str(project))  # not an image
    assert code == EXIT_FAILURE and data["error"]["code"] == "UNSUPPORTED_IMAGE"
    code, data = run_json(capsys, "material", "create", str(project), "X", "--dry-run")
    assert code == EXIT_OK and data["dry_run"]
    assert project.read_bytes() == before


def test_textures_outside_the_project_need_embedding(project, tmp_path, capsys):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    write_png(outside, np.full((2, 2, 4), 255, np.uint8))
    run_json(capsys, "material", "create", str(project), "M")
    before = project.read_bytes()
    code, data = run_json(
        capsys, "material", "texture", str(project), "M", "base_color", "--image", str(outside)
    )
    assert code == EXIT_FAILURE and data["error"]["code"] == "TEXTURE_OUTSIDE_PROJECT"
    assert project.read_bytes() == before
    code, data = run_json(
        capsys,
        "material",
        "texture",
        str(project),
        "M",
        "base_color",
        "--image",
        str(outside),
        "--embed",
    )
    assert code == EXIT_OK
    assert (
        Scene.load(project).materials.texture(data["operations"][0]["texture"]["id"]).source
        == "embedded"
    )


def test_procedural_textures_from_json_and_files(project, capsys):
    graph = {"type": "checker", "scale": 4, "color1": "#000000", "color2": "#ffffff"}
    code, data = run_json(
        capsys, "texture", "procedural", str(project), "Checks", json.dumps(graph)
    )
    assert code == EXIT_OK and data["operations"][0]["texture"]["node_types"] == ["checker"]
    graph_file = project.parent / "noise.json"
    graph_file.write_text(
        json.dumps({"nodes": {"n": {"type": "fbm", "octaves": 3}}, "output": "n"})
    )
    run_json(capsys, "material", "create", str(project), "M")
    code, data = run_json(
        capsys,
        "material",
        "texture",
        str(project),
        "M",
        "roughness",
        "--procedural",
        f"@{graph_file}",
        "--coordinates",
        "object",
        "--projection",
        "triplanar",
    )
    assert code == EXIT_OK and data["operations"][0]["binding"]["coordinates"] == "object"
    code, data = run_json(
        capsys, "material", "texture", str(project), "M", "base_color", "--procedural", "{not json"
    )
    assert code == EXIT_FAILURE and data["error"]["code"] == "INVALID_JSON"
    code, data = run_json(
        capsys,
        "texture",
        "procedural",
        str(project),
        "Loop",
        json.dumps({"type": "mix", "a": {"node": "out"}}),
    )
    assert code == EXIT_FAILURE and data["error"]["code"] == "GRAPH_CYCLE"
    out = project.parent / "checks.png"
    code, data = run_json(
        capsys, "texture", "preview", str(project), "Checks", "-o", str(out), "--size", "16"
    )
    assert code == EXIT_OK and out.read_bytes()[:4] == b"\x89PNG" and data["size"] == [16, 16]
    code, data = run_json(capsys, "texture", "inspect", str(project), "Checks")
    assert data["texture"]["type"] == "procedural" and data["texture"]["used_by"] == []
    code, data = run_json(
        capsys, "texture", "set", str(project), "Checks", "--graph", json.dumps({"type": "stripes"})
    )
    assert (
        code == EXIT_OK
        and Scene.load(project).materials.texture("Checks").graph.nodes["out"].type == "stripes"
    )


def test_scene_scripts_keep_image_paths_inside_the_project(project, tmp_path, capsys):
    script = tmp_path / "s.json"
    for image, expected in (
        ("textures/stripes.png", EXIT_OK),
        ("../escape.png", EXIT_FAILURE),
        ("/etc/hosts", EXIT_FAILURE),
    ):
        script.write_text(
            json.dumps(
                {
                    "format": "aether3d.scene-script",
                    "version": 1,
                    "operations": [
                        {"op": "add_texture", "name": f"T {len(image)}", "image": image}
                    ],
                }
            )
        )
        code, data = run_json(capsys, "scene", "run", str(project), str(script))
        assert code == expected, data
        if expected == EXIT_FAILURE:
            assert data["error"]["code"] == "UNSAFE_PATH"


def test_scene_scripts_roll_back_material_changes(project, tmp_path, capsys):
    script = tmp_path / "s.json"
    script.write_text(
        json.dumps(
            {
                "format": "aether3d.scene-script",
                "version": 1,
                "operations": [
                    {"op": "create_material", "name": "Wood", "preset": "wood"},
                    {"op": "assign_material", "objects": ["Box"], "material": "Wood"},
                    {"op": "set_object", "object": "Box", "color": "#ff0000"},
                ],
            }
        )
    )
    before = project.read_bytes()
    code, data = run_json(capsys, "scene", "run", str(project), str(script))
    assert code == EXIT_FAILURE and data["error"]["code"] == "OBJECT_HAS_MATERIAL"
    assert project.read_bytes() == before


def test_convert_colors_preview_presets_and_schema(project, capsys):
    scene = Scene.load(project)
    scene.get_by_name("Box").color = "#c0392b"
    scene.save(project)
    code, data = run_json(capsys, "material", "convert-colors", str(project))
    assert code == EXIT_OK and sorted(m["name"] for m in data["operations"][0]["created"]) == [
        "Color #c0392b",
        "Color #cccccc",
    ]
    out = project.parent / "preview.png"
    code, data = run_json(
        capsys, "material", "preview", str(project), "Color #c0392b", "-o", str(out), "--size", "32"
    )
    assert code == EXIT_OK and out.exists() and data["render"]["width"] == 32
    assert "__preview__" not in [o.name for o in Scene.load(project)]  # the project is unchanged
    code, data = run_json(capsys, "material", "presets")
    assert [p["name"] for p in data["presets"]] == [
        "stone",
        "brick",
        "wood",
        "metal",
        "grass",
        "fabric",
    ]
    code, data = run_json(capsys, "material", "schema")
    assert data["material"]["models"] == ["standard", "simple"] and len(data["node_types"]) == 16


def test_export_bakes_procedural_textures_only_on_request(project, capsys):
    assert run_json(capsys, "material", "create", str(project), "Checks")[0] == EXIT_OK
    checker = '{"type": "checker", "scale": 4}'  # mapped by UV: glTF can carry a baked image
    code, _ = run_json(
        capsys, "material", "texture", str(project), "Checks", "base_color", "--procedural", checker
    )
    assert code == EXIT_OK
    assert run_json(capsys, "material", "assign", str(project), "Checks", "Box")[0] == EXIT_OK
    plain, baked = project.with_name("plain.glb"), project.with_name("baked.glb")
    code, data = run_json(capsys, "export", str(project), str(plain))
    codes = [d["code"] for d in data["diagnostics"]]
    assert code == EXIT_OK and "PROCEDURAL_TEXTURE_NOT_EXPORTED" in codes
    assert "TEXTURE_BAKED" not in codes
    code, data = run_json(capsys, "export", str(project), str(baked), "--bake-textures", "32")
    codes = [d["code"] for d in data["diagnostics"]]
    assert code == EXIT_OK and "TEXTURE_BAKED" in codes
    assert "PROCEDURAL_TEXTURE_NOT_EXPORTED" not in codes
    assert baked.stat().st_size > plain.stat().st_size
    bad = project.with_name("bad.glb")
    code, data = run_json(capsys, "export", str(project), str(bad), "--bake-textures", "3")
    assert code == EXIT_FAILURE and "bake_textures" in data["error"]["message"]
    assert not bad.exists()


def test_listing_and_validating_never_need_failing_geometry(tmp_path, capsys):
    path = tmp_path / "unevaluable.aether"
    path.write_text(json.dumps(unevaluable_project()), encoding="utf-8")
    code, data = run_json(capsys, "material", "list", str(path))
    assert code == EXIT_OK and data["colored_objects"] == ["Thing"]
    code, _ = run_json(capsys, "material", "create", str(path), "Stone", "--preset", "stone")
    assert code == EXIT_OK
    assert run_json(capsys, "material", "assign", str(path), "Stone", "Thing")[0] == EXIT_OK
    code, data = run_json(capsys, "material", "validate", str(path))
    assert code == EXIT_OK and data["ok"]
    assert run_json(capsys, "material", "inspect", str(path), "Stone")[0] == EXIT_OK


def test_inspect_shows_object_materials(project, capsys):
    run_json(capsys, "material", "create", str(project), "M")
    run_json(capsys, "material", "assign", str(project), "M", "Box")
    code, data = run_json(capsys, "inspect", str(project))
    box = next(o for o in data["objects"] if o["name"] == "Box")
    assert box["material"]["name"] == "M" and box["group_materials"] == {}


def test_texture_list_remove_and_glb_inputs(project, capsys, tmp_path):
    image = str(project.parent / "textures" / "stripes.png")
    run_json(
        capsys,
        "texture",
        "add",
        str(project),
        "Stripes",
        image,
        "--filter",
        "nearest",
        "--no-mipmaps",
    )
    code, data = run_json(capsys, "texture", "list", str(project))
    (texture,) = data["textures"]
    assert (
        texture["filter"] == "nearest" and texture["mipmaps"] is False and texture["status"]["ok"]
    )
    code, data = run_json(capsys, "texture", "remove", str(project), "Stripes")
    assert code == EXIT_OK and Scene.load(project).materials.textures == ()
    glb = tmp_path / "scene.glb"
    Scene.load(project).export(glb)
    code, data = run_json(capsys, "material", "create", str(glb), "M")
    assert code == EXIT_FAILURE and "import the glTF file first" in data["error"]["message"]
    code, data = run_json(capsys, "material", "list", str(glb))
    assert code == EXIT_OK
    assert Path(project).exists()
