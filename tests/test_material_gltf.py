"""Materials through glTF: export, import, round trips, conversions and what is reported."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_gltf_import import embedded, write_gltf

from aether3d import Mesh, Scene, create_cube, import_gltf
from aether3d.errors import AssetFormatError
from aether3d.gltf.materials import roughness_from_shininess, transform_from_gltf, transform_to_gltf
from aether3d.gltf.writer import build_gltf
from aether3d.rendering.png import encode_png, write_png

KHRONOS = Path(__file__).parent / "data" / "khronos"


def png(path: Path, pixels) -> Path:
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
    write_png(path, arr)
    return path


def textured_scene(tmp_path: Path) -> Scene:
    rng = np.random.default_rng(3)
    scene = Scene("Materials")
    floor = scene.add_plane("Floor", size=6)
    stone = scene.create_material(
        "Stone", base_color="#d0c8b8", roughness=0.9, metallic=0.2, specular=0.3, double_sided=True
    )
    stone.set_texture(
        "base_color",
        png(tmp_path / "color.png", rng.integers(0, 256, (8, 8, 3))),
        scale=[3, 2],
        rotation=30,
        offset=[0.25, 0.5],
    )
    packed = rng.integers(0, 256, (8, 8, 3))
    stone.set_texture("metallic", png(tmp_path / "mr.png", packed))
    stone.set_texture("roughness", "mr")  # the same image: green = roughness, blue = metallic
    normal = np.zeros((4, 4, 3), np.uint8) + (128, 140, 250)
    stone.set_texture("normal", png(tmp_path / "n.png", normal))
    stone.configure(normal_strength=0.7, occlusion_strength=0.6)
    stone.set_texture("occlusion", png(tmp_path / "ao.png", rng.integers(0, 256, (4, 4, 3))))
    scene.materials.texture("color").configure(
        wrap=("clamp", "mirror"), filter="nearest", mipmaps=False
    )
    floor.material = stone
    glow = scene.create_material(
        "Glow",
        model="simple",
        emissive="#ff8800",
        emissive_strength=4.0,
        alpha_mode="mask",
        alpha_cutoff=0.3,
        specular=0.6,
        shininess=80,
    )
    lamp = scene.add_sphere("Lamp", radius=0.4, position=(1, 1, 1))
    lamp.material = glow
    crate = scene.add_cube("Crate", size=1.0, position=(-1.5, 0, 0.5))
    crate.appearance.configure(color="#c0392b", specular=0.4, shininess=40)
    cube = create_cube(1.0)
    grouped = Mesh(
        cube.positions,
        cube.triangles,
        cube.normals,
        cube.uvs,
        groups=np.repeat([0, 0, 1, 1, 2, 2], 2),
    )
    box = scene.add_object("Grouped", mesh=grouped, position=(2, -1, 0.5))
    box.material = scene.create_material(
        "Shiny", metallic=1.0, roughness=0.2, base_color="#e6c37a", specular=0.8
    )
    scene.assign_material(box, glow, group=2)
    scene.add_cube("Plain", size=0.5, position=(0, -2, 0.25))  # default color: no glTF material
    scene.add_directional_light("Sun", direction=(0.3, 0.5, -1), intensity=2.0)
    scene.add_camera("Cam", position=(4, -7, 5), look_at=(0, 0, 0.5))
    return scene


def test_texture_transforms_convert_between_uv_conventions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ext = {
            "offset": rng.uniform(-2, 2, 2).tolist(),
            "rotation": float(rng.uniform(-3, 3)),
            "scale": rng.uniform(0.2, 4, 2).tolist(),
        }
        ours = transform_from_gltf(ext)
        # glTF maps uv_g (v down); Aether3D maps uv_a = (u, 1 - v_g): both must agree.
        uv_g = rng.uniform(-1, 2, 2)
        c, s = math.cos(ext["rotation"]), math.sin(ext["rotation"])
        scaled = np.array(ext["scale"]) * uv_g
        mapped_g = (
            np.array([c * scaled[0] - s * scaled[1], s * scaled[0] + c * scaled[1]]) + ext["offset"]
        )
        uv_a = np.array([uv_g[0], 1 - uv_g[1]])
        t = math.radians(ours["rotation"])
        scaled_a = np.array(ours["scale"][:2]) * uv_a
        mapped_a = (
            np.array(
                [
                    math.cos(t) * scaled_a[0] - math.sin(t) * scaled_a[1],
                    math.sin(t) * scaled_a[0] + math.cos(t) * scaled_a[1],
                ]
            )
            + ours["offset"][:2]
        )
        assert mapped_a == pytest.approx([mapped_g[0], 1 - mapped_g[1]], abs=1e-9)
        back = transform_to_gltf(ours["offset"], ours["rotation"], ours["scale"])
        assert back["offset"] == pytest.approx(ext["offset"]) and back["rotation"] == pytest.approx(
            ext["rotation"]
        )
    assert transform_to_gltf((0, 0, 0), 0.0, (1, 1, 1)) is None


@pytest.mark.parametrize("suffix", ["glb", "gltf"])
def test_materials_round_trip_through_gltf(tmp_path, suffix):
    scene = textured_scene(tmp_path)
    result = scene.export(tmp_path / f"m.{suffix}")
    codes = [d.code for d in result.diagnostics]
    assert codes.count("SIMPLE_MATERIAL_APPROXIMATED") == 2 and not [
        d for d in result.diagnostics if d.severity != "info"
    ]
    doc = build_gltf(scene)[0]
    assert sorted(doc["extensionsUsed"]) == [
        "KHR_materials_emissive_strength",
        "KHR_materials_specular",
        "KHR_texture_transform",
    ]
    assert len(doc["images"]) == 4 and len(doc["samplers"]) == 2
    back = import_gltf(tmp_path / f"m.{suffix}")
    # Face groups come back from primitives; exported tangents are recomputed, not imported.
    assert sorted(d.code for d in back.diagnostics) == [
        "ATTRIBUTES_IGNORED",
        "PRIMITIVES_AS_GROUPS",
    ]
    assert "TANGENT" in next(d.message for d in back.diagnostics if d.code == "ATTRIBUTES_IGNORED")
    restored = back.scene
    for name in ("Stone", "Glow", "Shiny"):
        original, copy = scene.materials[name], restored.materials[name]
        assert copy.id == original.id and copy.model == original.model
        for key, value in original.params.items():
            assert copy.params[key] == pytest.approx(value, abs=1e-6), (name, key)
        assert set(copy.textures) == set(original.textures)
        for slot, binding in original.textures.items():
            other = copy.textures[slot]
            assert other.channel_for(slot, "image") == binding.channel_for(slot, "image")
            assert np.allclose(other.matrix, binding.matrix, atol=1e-9), slot
            a, b = (
                scene.materials.texture(binding.texture),
                restored.materials.texture(other.texture),
            )
            assert a.load().digest == b.load().digest  # the original PNG bytes
            assert (a.wrap, a.filter, a.mipmaps) == (b.wrap, b.filter, b.mipmaps)
    crate = restored.get_by_name("Crate")
    assert crate.material is None and crate.appearance == scene.get_by_name("Crate").appearance
    grouped = restored.get_by_name("Grouped")
    assert grouped.mesh.groups is not None and grouped.material is None
    assert {g: m.name for g, m in grouped.group_materials.items()} == {
        0: "Shiny",
        1: "Shiny",
        2: "Glow",
    }
    assert restored.get_by_name("Plain").material is None
    # glTF keeps cameras and lights as plain nodes: give the copy the same view and light.
    restored.add_camera("View", position=(4, -7, 5), look_at=(0, 0, 0.5))
    restored.add_directional_light("Sun light", direction=(0.3, 0.5, -1), intensity=2.0)
    before = scene.render(width=96, height=64, samples=2)
    after = restored.render(width=96, height=64, samples=2, camera="View")
    same = before.object_index == after.object_index
    assert same.mean() > 0.99
    diff = np.abs(before.image.astype(int) - after.image.astype(int)).max(-1)[same]
    assert diff.mean() < 0.5 and np.percentile(diff, 99) <= 3


def test_procedural_textures_are_reported_or_baked_on_request(tmp_path):
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("Checks", roughness=0.7)
    m.set_procedural_texture("base_color", "checker", scale=2, color1="#000000", color2="#ffffff")
    m.set_procedural_texture("roughness", "gradient")
    m.set_procedural_texture("occlusion", "noise", coordinates="object", projection="xy")
    plane.material = m
    plain = scene.export(tmp_path / "plain.glb")
    warnings = sorted(d.code for d in plain.diagnostics if d.severity == "warning")
    assert warnings == ["PROCEDURAL_TEXTURE_NOT_EXPORTED"] * 3
    baked = scene.export(tmp_path / "baked.glb", bake_textures=32)
    codes = sorted(d.code for d in baked.diagnostics)
    assert codes.count("TEXTURE_BAKED") == 2 and "TEXTURE_MAPPING_NOT_EXPORTED" in codes
    back = import_gltf(tmp_path / "baked.glb").scene
    material = back.materials["Checks"]
    colors = material.texture("base_color").load().image.pixels
    # 2 x 2 checker cells over the uv square; row 0 is the top (v = 1).
    assert colors[0, 0, 0] == 255 and colors[0, 31, 0] == 0 and colors[31, 0, 0] == 0
    packed = material.texture("roughness").load().image.pixels
    assert packed[16, 0, 1] < 20 and packed[16, 31, 1] > 235  # roughness = u, in green
    assert material.textures["roughness"].channel_for("roughness", "image") == "g"
    with pytest.raises(Exception, match="bake_textures"):
        scene.export(tmp_path / "x.glb", bake_textures=3)


def test_what_gltf_cannot_express_is_reported(tmp_path):
    scene = Scene()
    plane = scene.add_plane("P")
    m = scene.create_material("M")
    image = png(tmp_path / "a.png", np.zeros((2, 2, 3)))
    m.set_texture("base_color", image, coordinates="world", projection="xy")
    m.set_texture("opacity", image)
    m.set_texture("roughness", image, channel="r")
    m.set_texture("occlusion", image, channel="g")
    plane.material = m
    codes = sorted(d.code for d in scene.export(tmp_path / "x.glb").diagnostics)
    assert codes == [
        "OPACITY_TEXTURE_NOT_EXPORTED",
        "TEXTURE_CHANNEL_NOT_EXPORTED",
        "TEXTURE_CHANNEL_NOT_EXPORTED",
        "TEXTURE_MAPPING_NOT_EXPORTED",
    ]


def test_appearance_roughness_follows_blinn_phong():
    assert roughness_from_shininess(0.0, 32.0) == 1.0
    assert roughness_from_shininess(0.5, 2.0) == pytest.approx(0.5**0.25)  # (2 / (n + 2))^(1/4)
    assert roughness_from_shininess(0.5, 1000.0) < 0.25


def test_khronos_texture_settings_map_to_samplers():
    result = import_gltf(KHRONOS / "TextureSettingsTest/glTF-Binary/TextureSettingsTest.glb")
    wraps = {t.wrap for t in result.scene.materials.textures}
    assert {
        ("clamp", "repeat"),
        ("repeat", "clamp"),
        ("mirror", "repeat"),
        ("repeat", "mirror"),
    } <= wraps
    doubles = {m.name: m.double_sided for m in result.scene.materials.materials}
    assert True in doubles.values() and False in doubles.values()
    coords = import_gltf(KHRONOS / "TextureCoordinateTest/glTF-Binary/TextureCoordinateTest.glb")
    assert [d.code for d in coords.diagnostics] == []
    assert all(t.status()["ok"] for t in coords.scene.materials.textures)


def image_document(mutate_material=None, image=None):
    """A triangle with UVs and a textured material, as an embedded .gltf document."""
    import base64

    pixels = encode_png(np.full((2, 2, 4), 200, np.uint8))

    def mutate(d):
        binary = d["buffers"][0]["uri"].split(",", 1)[1]
        raw = base64.b64decode(binary)
        uvs = np.float32([[0, 0], [1, 0], [0, 1]]).tobytes()
        raw = raw + uvs
        d["buffers"][0] = {
            "byteLength": len(raw),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(raw).decode(),
        }
        d["bufferViews"].append({"buffer": 0, "byteOffset": len(raw) - 24, "byteLength": 24})
        d["accessors"].append(
            {
                "bufferView": len(d["bufferViews"]) - 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
            }
        )
        d["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = len(d["accessors"]) - 1
        d["meshes"][0]["primitives"][0]["material"] = 0
        d["images"] = [
            image or {"uri": "data:image/png;base64," + base64.b64encode(pixels).decode()}
        ]
        d["textures"] = [{"source": 0}]
        d["materials"] = [
            {"name": "Mat", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ]
        if mutate_material:
            mutate_material(d)

    return embedded(mutate)


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda d: d["materials"][0].update(alphaMode="BLEND"), "ALPHA_BLEND_UNSUPPORTED"),
        (
            lambda d: d["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"].update(
                texCoord=1
            ),
            "TEXCOORD_UNSUPPORTED",
        ),
        (
            lambda d: d["materials"][0].update(extensions={"KHR_materials_clearcoat": {}}),
            "MATERIAL_EXTENSION_IGNORED",
        ),
        (
            lambda d: d["materials"][0]["pbrMetallicRoughness"].update(
                baseColorFactor=[2, 0, 0, 1]
            ),
            "FACTOR_CLAMPED",
        ),
        (
            lambda d: d["textures"].__setitem__(
                0, {"extensions": {"KHR_texture_basisu": {"source": 0}}}
            ),
            "TEXTURE_FORMAT_UNSUPPORTED",
        ),
        (lambda d: d["images"].__setitem__(0, {"uri": "missing.png"}), "TEXTURE_MISSING"),
        (
            lambda d: d["images"].__setitem__(0, {"uri": "data:image/png;base64,iVBORw0KGgo="}),
            "TEXTURE_INVALID",
        ),
        (
            lambda d: d["images"].__setitem__(0, {"uri": "data:image/webp;base64,AAAA"}),
            "IMAGE_FORMAT_UNSUPPORTED",
        ),
    ],
)
def test_import_reports_what_it_cannot_use(tmp_path, mutate, code):
    result = import_gltf(write_gltf(tmp_path / "m.gltf", image_document(mutate)))
    assert code in [d.code for d in result.diagnostics]
    assert result.scene.objects[0].material.name == "Mat"  # the rest of the material is kept


def test_material_extensions_and_factors_import(tmp_path):
    def mutate(d):
        d["materials"][0].update(
            emissiveFactor=[1, 0.5, 0],
            alphaMode="MASK",
            alphaCutoff=0.25,
            doubleSided=True,
            extensions={
                "KHR_materials_emissive_strength": {"emissiveStrength": 5.0},
                "KHR_materials_specular": {"specularFactor": 0.5},
            },
        )
        d["materials"][0]["pbrMetallicRoughness"].update(metallicFactor=0.25, roughnessFactor=0.75)

    material = import_gltf(write_gltf(tmp_path / "m.gltf", image_document(mutate))).scene.materials[
        "Mat"
    ]
    assert material.emissive_strength == 5.0 and material.linear("emissive") == pytest.approx(
        [1, 0.5, 0]
    )
    assert (material.alpha_mode, material.alpha_cutoff, material.double_sided) == (
        "mask",
        0.25,
        True,
    )
    assert (material.metallic, material.roughness, material.specular) == (0.25, 0.75, 0.25)


def test_unsafe_image_uris_are_refused(tmp_path):
    for uri in ("../outside.png", "/etc/passwd", "http://example.com/a.png"):
        with pytest.raises(AssetFormatError) as info:
            import_gltf(
                write_gltf(
                    tmp_path / "m.gltf",
                    image_document(lambda d, u=uri: d["images"].__setitem__(0, {"uri": u})),
                )
            )
        assert info.value.code == "UNSAFE_URI"


def test_exports_are_deterministic(tmp_path):
    scene = textured_scene(tmp_path)
    a = build_gltf(scene)
    b = build_gltf(scene)
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True) and a[1] == b[1]
