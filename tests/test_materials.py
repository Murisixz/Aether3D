"""The material API: materials, textures, bindings, the library, presets and assignment."""

import os
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene
from aether3d.errors import MaterialError
from aether3d.materials import IMAGE_CACHE, ImageTexture, ProceduralTexture, material_schema
from aether3d.materials.presets import preset_info, preset_names
from aether3d.materials.textures import check_relative_path, relative_to_project
from aether3d.rendering.png import write_png


def png(path: Path, color=(200, 100, 50), size=(4, 4)) -> Path:
    image = np.zeros((size[1], size[0], 4), np.uint8)
    image[..., :3] = color
    image[..., 3] = 255
    write_png(path, image)
    return path


def test_materials_have_stable_ids_unique_names_and_typed_defaults():
    scene = Scene()
    a = scene.create_material()
    b = scene.create_material()
    assert (a.name, b.name) == ("Material", "Material.001") and a.id != b.id
    assert a.model == "standard" and a.roughness == 0.5 and a.metallic == 0.0
    assert a.specular == 0.5 and a.base_color == (0.8, 0.8, 0.8) and a.double_sided is True
    simple = scene.create_material("Plain", model="simple")
    assert simple.specular == 0.0 and simple.shininess == 32.0
    with pytest.raises(MaterialError) as info:
        simple.roughness  # noqa: B018 - the simple model has no roughness
    assert info.value.code == "PARAMETER_NOT_USED"
    with pytest.raises(MaterialError) as info:
        scene.create_material("Material")
    assert info.value.code == "NAME_CONFLICT"
    assert scene.materials["Plain"] is simple and scene.materials[simple.id] is simple


@pytest.mark.parametrize(
    "params,where",
    [
        ({"roughness": 1.5}, "roughness"),
        ({"metallic": -0.1}, "metallic"),
        ({"base_color": "#12"}, "base_color"),
        ({"alpha_mode": "blend"}, "alpha_mode"),
        ({"double_sided": 1}, "double_sided"),
        ({"emissive_strength": float("inf")}, "emissive_strength"),
        ({"shininess": 5.0}, "shininess"),  # not a standard parameter
        ({"glossiness": 1.0}, None),
    ],
)
def test_invalid_parameters_change_nothing(params, where):
    scene = Scene()
    m = scene.create_material("M")
    before = m.to_dict()
    with pytest.raises(MaterialError) as info:
        m.configure(**params, opacity=0.25)
    assert m.to_dict() == before
    if where:
        assert info.value.location.endswith(where)
    with pytest.raises(MaterialError):
        scene.create_material("N", **params)
    assert "N" not in scene.materials


def test_switching_models_keeps_shared_values_and_resets_the_rest():
    scene = Scene()
    m = scene.create_material(
        "M", base_color="#336699", roughness=0.2, specular=0.9, emissive="#ff0000"
    )
    m.set_procedural_texture("roughness", "noise")
    m.set_procedural_texture("base_color", "checker")
    m.model = "simple"
    assert m.base_color == pytest.approx((0x33 / 255, 0x66 / 255, 0x99 / 255))
    assert m.emissive == (1.0, 0.0, 0.0) and m.specular == 0.0 and m.shininess == 32.0
    assert set(m.textures) == {"base_color"}  # the simple model has no roughness slot
    m.configure(model="standard", metallic=1.0)
    assert m.specular == 0.5 and m.metallic == 1.0 and m.roughness == 0.5


def test_texture_bindings_validate_slots_channels_and_coordinates(tmp_path):
    scene = Scene()
    m = scene.create_material("M")
    image = png(tmp_path / "a.png")
    b = m.set_texture("roughness", image)
    assert b.channel is None and b.channel_for("roughness", "image") == "g"
    assert m.texture("roughness").name == "a"
    with pytest.raises(MaterialError, match="unknown texture slot"):
        m.set_texture("specular_color", image)
    with pytest.raises(MaterialError, match="reads one channel"):
        m.set_texture("metallic", image, channel="rgb")
    with pytest.raises(MaterialError, match="reads rgb"):
        m.set_texture("base_color", image, channel="r")
    with pytest.raises(MaterialError, match="projection applies"):
        m.set_texture("base_color", image, projection="xy")
    with pytest.raises(MaterialError, match="normal map needs a surface frame"):
        m.set_texture("normal", image, coordinates="object", projection="volume")
    with pytest.raises(MaterialError, match="unknown texture binding"):
        m.set_texture("base_color", image, tiling=3)
    binding = m.set_texture(
        "base_color",
        "a",
        coordinates="world",
        projection="xz",
        scale=[2, 3],
        rotation=90,
        offset=0.5,
    )
    assert binding.scale == (2.0, 3.0, 1.0) and binding.offset == (0.5, 0.5, 0.5)
    assert np.allclose(binding.matrix[:2, :2], [[0.0, -3.0], [2.0, 0.0]], atol=1e-12)
    assert len(scene.materials.textures) == 1  # the same file is reused, by path or name
    m.set_texture_options("base_color", scale=4)
    assert (
        m.textures["base_color"].scale == (4.0, 4.0, 4.0)
        and m.textures["base_color"].projection == "xz"
    )
    m.clear_texture("base_color")
    assert "base_color" not in m.textures


def test_image_textures_are_checked_when_added(tmp_path):
    scene = Scene()
    with pytest.raises(MaterialError) as info:
        scene.add_image_texture("Missing", tmp_path / "nope.png")
    assert info.value.code == "TEXTURE_MISSING"
    (tmp_path / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\nnot really")
    with pytest.raises(MaterialError) as info:
        scene.add_image_texture("Bad", tmp_path / "bad.png")
    assert info.value.code == "INVALID_IMAGE"
    os.mkfifo(tmp_path / "fifo.png") if hasattr(os, "mkfifo") else None
    if (tmp_path / "fifo.png").exists():
        with pytest.raises(MaterialError) as info:
            scene.add_image_texture("Fifo", tmp_path / "fifo.png")
        assert info.value.code == "TEXTURE_UNREADABLE"
    assert scene.materials.textures == ()
    with pytest.raises(MaterialError, match="wrap u"):
        scene.add_image_texture("W", png(tmp_path / "w.png"), wrap="tile")
    with pytest.raises(MaterialError, match="filter"):
        scene.add_image_texture("F", png(tmp_path / "f.png"), filter="cubic")


def test_sampler_settings_and_color_space(tmp_path):
    scene = Scene()
    t = scene.add_image_texture(
        "T", png(tmp_path / "t.png"), wrap=("clamp", "mirror"), filter="nearest", mipmaps=False
    )
    assert (t.wrap, t.filter, t.mipmaps, t.color_space) == (
        ("clamp", "mirror"),
        "nearest",
        False,
        "auto",
    )
    t.configure(wrap_u="repeat", color_space="linear")
    assert (
        t.wrap == ("repeat", "mirror")
        and t.mode_for("srgb") == "linear"
        and t.mode_for("normal") == "normal"
    )
    with pytest.raises(MaterialError):
        t.configure(wrap="spin")
    assert t.wrap == ("repeat", "mirror")


def test_identical_images_decode_once_and_edited_files_reload(tmp_path):
    IMAGE_CACHE.clear()
    scene = Scene()
    a = scene.add_image_texture("A", png(tmp_path / "a.png", (10, 20, 30)))
    b = scene.add_image_texture("B", png(tmp_path / "b.png", (10, 20, 30)))
    decodes = IMAGE_CACHE.decodes
    assert a.load().digest == b.load().digest and IMAGE_CACHE.decodes == decodes  # dedup
    first = a.load().image.pixels[0, 0, :3].tolist()
    png(tmp_path / "a.png", (90, 80, 70), size=(5, 4))  # edited on disk: size and content change
    second = a.load()
    assert first == [10, 20, 30] and second.image.pixels[0, 0, :3].tolist() == [90, 80, 70]
    assert second.image.width == 5
    stats = IMAGE_CACHE.stats()
    assert stats["file_reads"] >= 3 and stats["bytes"] > 0


def test_embedding_and_extracting(tmp_path):
    scene = Scene()
    t = scene.add_image_texture("T", png(tmp_path / "t.png", (1, 2, 3)), embed=True)
    assert t.source == "embedded" and t.file is None and t.mime_type == "image/png"
    (tmp_path / "t.png").unlink()
    assert t.load().image.pixels[0, 0, :3].tolist() == [1, 2, 3]  # no longer needs the file
    with pytest.raises(MaterialError) as info:
        t.extract(tmp_path / "out.jpg")
    assert info.value.code == "INVALID_OUTPUT"
    written = t.extract(tmp_path / "out.png")
    assert t.source == "file" and written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = scene.add_image_texture("D", data=written.read_bytes())
    assert data.source == "embedded"


@pytest.mark.parametrize(
    "path",
    ["../up.png", "/etc/passwd", "C:/x.png", "a\\b.png", "a/../../b.png", "", "dir/", "a/\x00.png"],
)
def test_unsafe_relative_paths_are_refused(path):
    with pytest.raises(MaterialError) as info:
        check_relative_path(path, "textures[0].path")
    assert info.value.code == "UNSAFE_PATH"


def test_safe_relative_paths(tmp_path):
    assert check_relative_path("textures/a.png", "x") == "textures/a.png"
    assert check_relative_path("./a.png", "x") == "./a.png"
    assert relative_to_project(tmp_path / "t" / "a.png", tmp_path, "x") == "t/a.png"
    with pytest.raises(MaterialError) as info:
        relative_to_project(tmp_path.parent / "a.png", tmp_path, "x")
    assert info.value.code == "TEXTURE_OUTSIDE_PROJECT"


def test_procedural_textures_preview_and_bake(tmp_path):
    scene = Scene()
    t = scene.add_procedural_texture(
        "Checks", {"type": "checker", "scale": 2, "color1": "#000000", "color2": "#ffffff"}
    )
    assert isinstance(t, ProceduralTexture)
    image = t.preview(4, 4)
    assert image.shape == (4, 4, 3)
    # Row 0 is the top (v near 1): cells alternate black/white in 2 x 2 blocks.
    assert image[:, :, 0].tolist() == [
        [255, 255, 0, 0],
        [255, 255, 0, 0],
        [0, 0, 255, 255],
        [0, 0, 255, 255],
    ]
    out = t.bake(tmp_path / "checks.png", 16)
    assert out.read_bytes()[:4] == b"\x89PNG" and isinstance(
        scene.materials.texture("Checks"), ProceduralTexture
    )
    values = t.evaluate([[0.25, 0.25], [0.75, 0.25]])
    assert values[:, 0].tolist() == [0.0, 1.0]
    t.set_params(scale=4)
    assert t.graph.nodes["out"].params["scale"] == 4
    with pytest.raises(MaterialError):
        t.set_params(scale="big")
    assert t.graph.nodes["out"].params["scale"] == 4


def test_graph_image_nodes_reference_scene_images_and_follow_renames(tmp_path):
    scene = Scene()
    scene.add_image_texture("Photo", png(tmp_path / "p.png", (255, 0, 0)))
    tinted = scene.add_procedural_texture(
        "Tinted",
        {
            "nodes": {
                "img": {"type": "image", "texture": "Photo"},
                "out": {"type": "mix", "a": {"node": "img"}, "b": "#0000ff", "factor": 0.5},
            },
            "output": "out",
        },
    )
    assert tinted.evaluate([[0.5, 0.5]])[0, :3] == pytest.approx([0.5, 0.0, 0.5])
    scene.materials.texture("Photo").name = "Picture"
    assert tinted.graph.image_references() == ["Picture"]
    with pytest.raises(MaterialError) as info:
        scene.materials.remove_texture("Picture")
    assert info.value.code == "TEXTURE_IN_USE"
    with pytest.raises(MaterialError) as info:
        scene.add_procedural_texture("Broken", {"type": "image", "texture": "Nope"})
    assert info.value.code == "TEXTURE_NOT_FOUND"
    with pytest.raises(MaterialError):
        scene.add_procedural_texture("Inception", {"type": "image", "texture": "Tinted"})


def test_assignment_users_and_removal():
    scene = Scene()
    a, b = scene.add_cube("A"), scene.add_cube("B")
    cam = scene.add_camera("Cam")
    m = scene.create_material("M")
    other = scene.create_material("Other")
    scene.assign_material([a, "B"], m)
    scene.assign_material(a, other, group=2)
    assert a.material is m and b.material is m and a.group_materials == {2: other}
    assert a.material_for_group(2) is other and a.material_for_group(0) is m
    assert [(o.name, g) for o, g in scene.materials.users(m)] == [("A", None), ("B", None)]
    with pytest.raises(MaterialError) as info:
        scene.assign_material([a, cam], other)  # all checked first: nothing changes
    assert info.value.code == "NOT_A_MESH_OBJECT" and a.material is m
    with pytest.raises(MaterialError) as info:
        scene.materials.remove_material(m)
    assert info.value.code == "MATERIAL_IN_USE"
    assert scene.materials.remove_material(m, force=True) == sorted([a.id, b.id])
    assert a.material is None and b.material is None and m.library is None
    with pytest.raises(MaterialError):
        m.roughness = 0.3
    a.material = None
    scene.assign_material(a, None, group=2)
    assert a.group_materials == {}


def test_object_colors_cannot_be_set_while_a_material_decides():
    scene = Scene()
    cube = scene.add_cube("C")
    cube.material = scene.create_material("M")
    with pytest.raises(MaterialError) as info:
        cube.color = "#ff0000"
    assert info.value.code == "OBJECT_HAS_MATERIAL" and "obj.material = None" in info.value.hint
    cube.material = None
    cube.color = "#ff0000"
    assert cube.color == (1.0, 0.0, 0.0)


def test_convert_appearances_shares_equal_colors():
    scene = Scene()
    for i in range(3):
        scene.add_cube(f"Red {i}").color = "#c0392b"
    blue = scene.add_sphere("Blue")
    blue.appearance.configure(color="#2e86c1", specular=0.4, shininess=50)
    scene.add_cube("Default")
    scene.add_camera("Cam")
    created = scene.materials.convert_appearances()
    names = sorted(m.name for m in created)
    assert names == ["Color #2e86c1 spec 0.4/50", "Color #c0392b", "Color #cccccc"]
    red = scene.materials["Color #c0392b"]
    assert red.model == "simple" and [o.name for o, _ in scene.materials.users(red)] == [
        "Red 0",
        "Red 1",
        "Red 2",
    ]
    assert scene.materials["Color #2e86c1 spec 0.4/50"].shininess == 50
    assert scene.materials.convert_appearances() == []  # nothing left to convert
    per_object = Scene()
    per_object.add_cube("X").color = "#123456"
    per_object.add_cube("Y").color = "#123456"
    assert [m.name for m in per_object.materials.convert_appearances(dedupe=False)] == [
        "X Color",
        "Y Color",
    ]


def test_transactions_restore_everything_on_failure():
    scene = Scene()
    cube = scene.add_cube("C")
    m = scene.create_material("M", roughness=0.3)
    cube.material = m
    library = scene.materials
    with pytest.raises(RuntimeError), library.transaction():
        m.roughness = 0.9
        m.name = "Renamed"
        library.create_material("Temp")
        library.add_procedural_texture("Noise", "noise")
        cube.material = None
        raise RuntimeError("boom")
    assert m.roughness == 0.3 and m.name == "M" and cube.material is m
    assert [x.name for x in library.materials] == ["M"] and library.textures == ()


def test_set_procedural_texture_is_atomic():
    scene = Scene()
    m = scene.create_material("M")
    with pytest.raises(MaterialError):
        m.set_procedural_texture("base_color", "checker", coordinates="sideways")
    assert scene.materials.textures == () and m.textures == {}
    binding = m.set_procedural_texture(
        "normal", {"type": "bump", "height": 0.5}, coordinates="object", projection="xy"
    )
    assert binding.coordinates == "object" and m.texture("normal").name == "M normal"
    with pytest.raises(MaterialError, match="outputs a number"):
        m.set_procedural_texture("normal", "noise")


@pytest.mark.parametrize("name", preset_names())
def test_presets_build_procedural_materials(name):
    scene = Scene()
    m = scene.create_material(preset=name)
    assert m.name == name.capitalize() and m.model == "standard" and m.textures
    for slot, binding in m.textures.items():
        texture = scene.materials.texture(binding.texture)
        assert isinstance(texture, ProceduralTexture) and texture.name == f"{m.name} {slot}"
    big = scene.create_material(f"Big {name}", preset=name, scale=2.0, seed=5, roughness=0.4)
    assert all(b.scale == (0.5, 0.5, 0.5) for b in big.textures.values())
    assert big.roughness == 0.4
    with pytest.raises(MaterialError):
        scene.create_material("X", preset=name, sparkle=True)
    assert "X" not in scene.materials


def test_preset_and_schema_listings():
    info = {p["name"]: p for p in preset_info()}
    assert set(info) == {"stone", "brick", "wood", "metal", "grass", "fabric"}
    assert "mortar_color" in info["brick"]["options"]
    schema = material_schema()
    assert schema["models"] == ["standard", "simple"]
    assert {s["name"] for s in schema["slots"]} == {
        "base_color",
        "opacity",
        "metallic",
        "roughness",
        "normal",
        "emissive",
        "occlusion",
    }
    with pytest.raises(MaterialError) as info:
        Scene().create_material(preset="lava")
    assert info.value.code == "UNKNOWN_PRESET"


def test_inspect_and_validate(tmp_path):
    scene = Scene()
    flat = scene.add_object(
        "Flat", mesh=__import__("aether3d").Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    )
    m = scene.create_material("M")
    image = scene.add_image_texture("Img", png(tmp_path / "i.png"))
    m.set_texture("base_color", image)
    flat.material = m
    scene.create_material("Unused")
    report = m.inspect()
    assert report["textures"]["base_color"]["status"]["ok"] and report["users"] == [
        {"id": flat.id, "name": "Flat"}
    ]
    assert report["textures"]["base_color"]["decoded_as"] == "srgb"
    codes = [d.code for d in scene.materials.validate()]
    assert "MISSING_UVS" in codes and "UNUSED_MATERIAL" in codes
    (tmp_path / "i.png").unlink()
    image_status = image.status()
    assert not image_status["ok"] and image_status["code"] == "TEXTURE_MISSING"
    errors = [d for d in scene.materials.validate() if d.severity == "error"]
    assert [d.code for d in errors] == ["TEXTURE_MISSING"]
    full = scene.materials.inspect()
    assert [x["name"] for x in full["materials"]] == ["M", "Unused"] and full["textures"][0][
        "name"
    ] == "Img"
    assert isinstance(image, ImageTexture)
