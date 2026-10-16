"""Materials in .aether projects (format version 4): round trips, safety, strictness, migration."""

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from aether3d import FORMAT_VERSION, Mesh, Scene
from aether3d.errors import MaterialError, ProjectFormatError
from aether3d.rendering.png import write_png
from aether3d.serialization import dumps, loads, migrate, scene_from_dict, scene_to_dict

V040 = Path(__file__).parent / "data" / "v0.4.0" / "castle.aether"


def png(path: Path, color=(200, 100, 50)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 4), np.uint8)
    image[..., :3] = color
    image[..., 3] = 255
    image[:4, :4, :3] = 255
    write_png(path, image)
    return path


def grouped_quad() -> Mesh:
    return Mesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [0, 2, 3]],
        normals=[[0, 0, 1]] * 4,
        uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
        groups=[0, 1],
    )


def build(project: Path) -> Scene:
    scene = Scene("Materials")
    quad = scene.add_object("Quad", mesh=grouped_quad())
    cube = scene.add_cube("Cube", position=(2, 0, 0.5))
    scene.add_cube("Plain").color = "#336699"
    floor = scene.create_material("Floor", roughness=0.7, metallic=0.1, base_color="#a0a0a0")
    floor.set_texture("base_color", png(project / "textures" / "floor.png"), scale=[4, 4])
    floor.set_procedural_texture("roughness", "fbm", scale=6, octaves=3, seed=4)
    stamp = scene.add_image_texture(
        "Stamp",
        png(project.parent / "elsewhere" / "stamp.png", (9, 9, 9)),
        embed=True,
        filter="nearest",
    )
    glow = scene.create_material(
        "Glow", model="simple", emissive="#ff8800", emissive_strength=3.0, alpha_mode="mask"
    )
    glow.set_texture("opacity", stamp, channel="r", coordinates="object", projection="xy")
    scene.create_material("Stone", preset="stone", seed=3)
    quad.material = floor
    scene.assign_material(quad, glow, group=1)
    cube.material = "Stone"
    scene.add_directional_light("Sun", direction=(0.3, 0.5, -1))
    scene.add_camera("Cam", position=(1, -5, 4), look_at=(1, 0, 0))
    return scene


def test_projects_round_trip_materials_textures_and_groups(tmp_path):
    project = tmp_path / "project"
    scene = build(project)
    path = project / "scene.aether"
    scene.save(path)
    doc = json.loads(path.read_text())
    assert doc["format_version"] == FORMAT_VERSION == 7
    textures = {t["name"]: t for t in doc["scene"]["textures"]}
    assert textures["floor"]["path"] == "textures/floor.png"  # relative, POSIX
    assert (
        textures["Stamp"]["embedded"]["mime_type"] == "image/png"
        and "path" not in textures["Stamp"]
    )
    assert textures["Floor roughness"]["graph"]["nodes"]["out"]["type"] == "fbm"
    objects = {o["name"]: o for o in doc["scene"]["objects"]}
    floor_id = next(m["id"] for m in doc["scene"]["materials"] if m["name"] == "Floor")
    glow_id = next(m["id"] for m in doc["scene"]["materials"] if m["name"] == "Glow")
    assert objects["Quad"]["material"] == floor_id and objects["Quad"]["group_materials"] == {
        "1": glow_id
    }
    assert objects["Plain"]["material"] is None and objects["Plain"]["appearance"]["color"]
    assert doc["meshes"][objects["Quad"]["mesh"]]["groups"] == [0, 1]
    loaded = Scene.load(path)
    assert scene_to_dict(loaded, project) == scene_to_dict(scene, project)
    a = scene.render(width=96, height=64, samples=2)
    b = loaded.render(width=96, height=64, samples=2)
    assert np.array_equal(a.image, b.image) and not [
        d for d in b.diagnostics if d.severity != "info"
    ]
    # The project folder can move: paths are relative to it.
    moved = tmp_path / "moved"
    shutil.copytree(project, moved)
    shutil.rmtree(project)
    again = Scene.load(moved / "scene.aether")
    assert np.array_equal(again.render(width=96, height=64, samples=2).image, a.image)


def test_saving_refuses_textures_outside_the_project_and_writes_nothing(tmp_path):
    scene = Scene()
    m = scene.create_material("M")
    m.set_texture("base_color", png(tmp_path / "outside" / "t.png"))
    target = tmp_path / "project" / "scene.aether"
    target.parent.mkdir()
    target.write_text("previous content")
    with pytest.raises(MaterialError) as info:
        scene.save(target)
    assert info.value.code == "TEXTURE_OUTSIDE_PROJECT" and "embed" in info.value.hint
    assert target.read_text() == "previous content"
    assert [p.name for p in target.parent.iterdir()] == ["scene.aether"]  # no temporary left
    scene.materials.texture("t").embed()
    scene.save(target)
    assert Scene.load(target).materials.texture("t").source == "embedded"


@pytest.mark.parametrize(
    "path", ["../secret.png", "/etc/passwd", "a/../../b.png", "C:/x.png", "x\\y.png"]
)
def test_loading_refuses_paths_that_leave_the_project(tmp_path, path):
    doc = scene_to_dict(Scene())
    doc["scene"]["textures"] = [{"id": "t", "name": "T", "type": "image", "path": path}]
    with pytest.raises(ProjectFormatError) as info:
        scene_from_dict(doc, tmp_path)
    assert info.value.location.startswith("scene.textures[0]") and "UNSAFE_PATH" in str(info.value)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_symlinks_out_of_the_project_are_not_followed(tmp_path):
    secret = png(tmp_path / "secret" / "s.png")
    project = tmp_path / "project"
    project.mkdir()
    try:
        os.symlink(secret, project / "link.png")
    except OSError:
        pytest.skip("cannot create symbolic links here")
    scene = Scene()
    plane = scene.add_plane("P")
    scene.add_camera("C", position=(0, -2, 2), look_at=(0, 0, 0))
    m = scene.create_material("M")
    m.set_texture("base_color", project / "link.png")  # the API trusts the caller
    plane.material = m
    scene.save(project / "scene.aether")
    loaded = Scene.load(project / "scene.aether")  # a project file is not trusted
    result = loaded.render(width=32, height=32)
    codes = [d.code for d in result.diagnostics]
    assert "UNSAFE_PATH" in codes and result.status == "warning"


def test_missing_texture_files_load_and_fall_back_with_diagnostics(tmp_path):
    project = tmp_path / "p"
    scene = build(project)
    scene.save(project / "scene.aether")
    (project / "textures" / "floor.png").unlink()
    loaded = Scene.load(project / "scene.aether")
    result = loaded.render(width=64, height=48)
    missing = [d for d in result.diagnostics if d.code == "TEXTURE_MISSING"]
    assert len(missing) == 1 and missing[0].location == "materials[Floor].textures.base_color"
    assert result.status == "warning"
    errors = [d.code for d in loaded.materials.validate() if d.severity == "error"]
    assert errors == ["TEXTURE_MISSING"]


def test_version_3_projects_migrate_without_materials(tmp_path):
    scene = Scene("Old")
    scene.add_cube("Crate").color = "#c0392b"
    doc = scene_to_dict(scene)
    v3 = copy.deepcopy(doc)
    v3["format_version"] = 3
    del v3["scene"]["materials"], v3["scene"]["textures"]
    for o in v3["scene"]["objects"]:
        del o["material"], o["group_materials"]
    migrated = migrate(copy.deepcopy(v3))
    assert migrated["format_version"] == FORMAT_VERSION
    assert migrated["scene"]["materials"] == [] and migrated["scene"]["textures"] == []
    assert all(
        o["material"] is None and o["group_materials"] == {} for o in migrated["scene"]["objects"]
    )
    old = scene_from_dict(v3)
    assert old.get_by_name("Crate").color == pytest.approx((0xC0 / 255, 0x39 / 255, 0x2B / 255))
    assert np.array_equal(
        old.render(width=40, height=30, frame=True).image,
        scene.render(width=40, height=30, frame=True).image,
    )


def test_the_v040_castle_still_loads_and_converts():
    castle = Scene.load(V040)
    assert castle.materials.materials == ()
    doc = scene_to_dict(castle)
    assert doc["format_version"] == FORMAT_VERSION and doc["scene"]["materials"] == []


def corrupt(tmp_path, mutate) -> ProjectFormatError:
    project = tmp_path / "p"
    scene = build(project)
    doc = scene_to_dict(scene, project)
    mutate(doc)
    with pytest.raises(ProjectFormatError) as info:
        scene_from_dict(loads(dumps(doc)), project)
    return info.value


@pytest.mark.parametrize(
    "mutate,where",
    [
        (lambda d: d["scene"]["materials"][0].update(roughness=2), "scene.materials[0].roughness"),
        (lambda d: d["scene"]["materials"][0].update(model="phong"), "scene.materials[0].model"),
        (lambda d: d["scene"]["materials"][0].update(shininess=3), "scene.materials[0]"),
        (lambda d: d["scene"]["materials"][0].update(sparkle=1), "scene.materials[0]"),
        (
            lambda d: d["scene"]["materials"][1].update(id=d["scene"]["materials"][0]["id"]),
            "scene.materials[1]",
        ),
        (
            lambda d: d["scene"]["materials"][0]["textures"]["base_color"].update(texture="nope"),
            "scene.materials[0].textures.base_color.texture",
        ),
        (
            lambda d: d["scene"]["materials"][0]["textures"]["base_color"].update(channel="q"),
            "scene.materials[0].textures.base_color.channel",
        ),
        (
            lambda d: d["scene"]["materials"][1]["textures"].update(metallic={"texture": "x"}),
            "scene.materials[1].textures.metallic",
        ),
        (lambda d: d["scene"]["textures"][0].update(type="video"), "scene.textures[0].type"),
        (lambda d: d["scene"]["textures"][0].update(wrap="spin"), "scene.textures[0].wrap"),
        (
            lambda d: d["scene"]["textures"][0].update(
                embedded={"mime_type": "image/png", "data": "@@"}
            ),
            "scene.textures[0]",
        ),
        (
            lambda d: d["scene"]["textures"][2]["embedded"].update(data="not base64!"),
            "scene.textures[2].embedded",
        ),
        (
            lambda d: d["scene"]["textures"][2]["embedded"].update(mime_type="image/gif"),
            "scene.textures[2].embedded",
        ),
        (
            lambda d: d["scene"]["textures"][1].update(graph={"type": "bogus"}),
            "scene.textures[1].graph.nodes.out.type",
        ),
        (lambda d: d["scene"]["objects"][0].update(material="nope"), "scene.objects[0].material"),
        (
            lambda d: d["scene"]["objects"][0].update(group_materials={"x": "y"}),
            "scene.objects[0].group_materials",
        ),
        (
            lambda d: d["scene"]["objects"][0].update(group_materials={"99999": "y"}),
            "scene.objects[0].group_materials",
        ),
        (lambda d: d["meshes"][0].update(groups=[0, 1, 2]), "meshes[0]"),
        (lambda d: d["meshes"][0].update(groups=[0, "a"]), "meshes[0].groups"),
    ],
)
def test_invalid_material_data_is_rejected_with_a_location(tmp_path, mutate, where):
    error = corrupt(tmp_path, mutate)
    assert error.location == where, str(error)


def test_cameras_cannot_have_materials(tmp_path):
    def mutate(d):
        cam = next(o for o in d["scene"]["objects"] if o["camera"] is not None)
        cam["material"] = d["scene"]["materials"][0]["id"]

    assert "cannot have materials" in corrupt(tmp_path, mutate).message
