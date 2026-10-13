import base64
import json
import struct

import numpy as np
import pytest
from scenes import challenging_scene

from aether3d import Mesh, Scene, UnsupportedFeatureError, ValidationError, create_cube, export_gltf
from aether3d.gltf.writer import build_gltf


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A and json_len % 4 == 0
    doc = json.loads(data[20 : 20 + json_len])
    binary = b""
    if 20 + json_len < len(data):
        bin_len, bin_type = struct.unpack_from("<II", data, 20 + json_len)
        assert bin_type == 0x004E4942 and bin_len % 4 == 0
        binary = data[28 + json_len : 28 + json_len + bin_len]
    return doc, binary


def test_glb_container_and_document_structure(tmp_path):
    scene = challenging_scene()
    result = scene.export(tmp_path / "s.glb")
    doc, binary = read_glb(tmp_path / "s.glb")
    assert result.format == "glb" and result.byte_length == (tmp_path / "s.glb").stat().st_size
    assert doc["asset"]["version"] == "2.0"
    assert doc["asset"]["generator"].startswith("Aether3D ")
    assert doc["scene"] == 0 and doc["scenes"][0]["name"] == "Challenging"
    assert len(doc["buffers"]) == 1 and "uri" not in doc["buffers"][0]
    assert doc["buffers"][0]["byteLength"] == len(binary)
    # Nodes are in depth-first order with the object ids stored in extras.
    assert [n["name"] for n in doc["nodes"]] == [o.name for o in scene]
    assert [n["extras"]["aether3d"]["id"] for n in doc["nodes"]] == [o.id for o in scene]
    assert [doc["nodes"][i]["name"] for i in doc["scenes"][0]["nodes"]] == [
        r.name for r in scene.roots
    ]
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(binary)
        assert view["target"] in (34962, 34963)
    for mesh in doc["meshes"]:
        (prim,) = mesh["primitives"]
        assert prim["mode"] == 4
        assert set(prim["attributes"]) == {"POSITION", "NORMAL", "TEXCOORD_0"}
        pos = doc["accessors"][prim["attributes"]["POSITION"]]
        assert pos["componentType"] == 5126 and pos["type"] == "VEC3"
        assert "min" in pos and "max" in pos
        assert doc["accessors"][prim["indices"]]["componentType"] == 5123  # small -> uint16


def test_position_min_max_are_exact(tmp_path):
    scene = Scene()
    scene.add_object(mesh=Mesh([(0.1, -0.2, 0.3), (1.5, 2.5, -3.5), (0, 0, 0)], [(0, 1, 2)]))
    scene.export(tmp_path / "m.glb")
    doc, binary = read_glb(tmp_path / "m.glb")
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = doc["bufferViews"][acc["bufferView"]]
    data = np.frombuffer(binary, "<f4", 9, view["byteOffset"]).reshape(3, 3)
    assert acc["min"] == data.min(axis=0).astype(float).tolist()
    assert acc["max"] == data.max(axis=0).astype(float).tolist()
    # Z-up -> Y-up: glTF (x, y, z) = (x, z, -y)
    np.testing.assert_array_equal(data[1], np.float32([1.5, -3.5, -2.5]))


def test_shared_meshes_are_instanced(tmp_path):
    scene = Scene()
    mesh = create_cube()
    for i in range(3):
        scene.add_object(f"Box {i}", mesh=mesh, position=(i, 0, 0))
    scene.add_cube("Other")
    result = scene.export(tmp_path / "i.glb")
    doc, _ = read_glb(tmp_path / "i.glb")
    assert result.mesh_count == 2
    assert [n.get("mesh") for n in doc["nodes"]] == [0, 0, 0, 1]


def test_large_meshes_use_uint32_indices(tmp_path):
    n = 70_000
    positions = np.zeros((n, 3), np.float32)
    positions[:, 0] = np.arange(n)
    positions[1::2, 1] = 1
    triangles = np.stack([np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)], axis=1)
    scene = Scene()
    scene.add_object(mesh=Mesh(positions, triangles))
    scene.export(tmp_path / "big.glb")
    doc, _ = read_glb(tmp_path / "big.glb")
    assert doc["accessors"][doc["meshes"][0]["primitives"][0]["indices"]]["componentType"] == 5125
    back = Scene.import_file(tmp_path / "big.glb")
    assert np.array_equal(back.objects[0].mesh.triangles, triangles)


def test_default_transforms_are_omitted(tmp_path):
    scene = Scene()
    scene.add_object("Plain")
    scene.add_object("Moved", position=(1, 2, 3))
    doc, _ = read_glb(scene.export(tmp_path / "t.glb").path)
    plain, moved = doc["nodes"]
    assert not {"translation", "rotation", "scale", "matrix"} & set(plain)
    assert moved["translation"] == [1.0, 3.0, -2.0]


def test_empty_scene(tmp_path):
    result = Scene("Nothing").export(tmp_path / "e.glb")
    doc, binary = read_glb(tmp_path / "e.glb")
    assert binary == b""
    assert doc["scenes"] == [{"name": "Nothing"}]
    assert "nodes" not in doc and "buffers" not in doc
    assert result.node_count == 0
    assert len(Scene.import_file(tmp_path / "e.glb")) == 0


def test_gltf_embeds_buffer_as_data_uri(tmp_path):
    scene = challenging_scene()
    scene.export(tmp_path / "s.gltf")
    doc = json.loads((tmp_path / "s.gltf").read_text(encoding="utf-8"))
    uri = doc["buffers"][0]["uri"]
    assert uri.startswith("data:application/octet-stream;base64,")
    assert len(base64.b64decode(uri.split(",", 1)[1])) == doc["buffers"][0]["byteLength"]
    glb_doc, _ = read_glb(scene.export(tmp_path / "s.glb").path)
    del doc["buffers"][0]["uri"]
    assert doc == glb_doc  # same document, different container
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.glb", "s.gltf"]


def test_export_is_deterministic(tmp_path):
    scene = challenging_scene()
    a = scene.export(tmp_path / "a.glb").path.read_bytes()
    b = scene.export(tmp_path / "b.glb").path.read_bytes()
    assert a == b


@pytest.mark.parametrize("name", ["s.obj", "s", "s.aether", "s.GLB.txt"])
def test_unsupported_export_formats(tmp_path, name):
    with pytest.raises(ValidationError, match="supported: .glb, .gltf"):
        Scene().export(tmp_path / name)
    assert list(tmp_path.iterdir()) == []


def test_uppercase_extension_is_accepted(tmp_path):
    assert Scene().export(tmp_path / "UPPER.GLB").format == "glb"


def test_empty_mesh_is_lossy(tmp_path):
    scene = Scene()
    scene.add_object("Points", mesh=Mesh([(0, 0, 0)], []))
    with pytest.raises(UnsupportedFeatureError) as info:
        scene.export(tmp_path / "p.glb")
    assert [d.code for d in info.value.diagnostics] == ["EMPTY_MESH_NOT_EXPORTED"]
    assert list(tmp_path.iterdir()) == []  # nothing written
    result = scene.export(tmp_path / "p.glb", allow_lossy=True)
    doc, _ = read_glb(result.path)
    assert "mesh" not in doc["nodes"][0] and "meshes" not in doc
    assert result.diagnostics[0].lossy


def test_zero_normals_are_lossy_and_non_unit_normals_renormalized(tmp_path):
    tri = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    scene = Scene()
    scene.add_object("Zero", mesh=Mesh(tri, [(0, 1, 2)], normals=[(0, 0, 1), (0, 0, 0), (0, 0, 1)]))
    with pytest.raises(UnsupportedFeatureError, match="ZERO_NORMALS_NOT_EXPORTED"):
        scene.export(tmp_path / "z.glb")
    doc, _ = read_glb(scene.export(tmp_path / "z.glb", allow_lossy=True).path)
    assert "NORMAL" not in doc["meshes"][0]["primitives"][0]["attributes"]

    scene = Scene()
    scene.add_object("Long", mesh=Mesh(tri, [(0, 1, 2)], normals=[(0, 0, 2)] * 3))
    result = scene.export(tmp_path / "l.glb")
    assert [d.code for d in result.diagnostics] == ["NORMALS_RENORMALIZED"]
    back = Scene.import_file(tmp_path / "l.glb")
    np.testing.assert_array_equal(back.objects[0].mesh.normals, np.float32([(0, 0, 1)] * 3))


def test_failed_export_keeps_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "keep.glb"
    challenging_scene().export(path)
    before = path.read_bytes()
    import aether3d._fileio as fileio

    monkeypatch.setattr(fileio.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError):
        Scene("Other").export(path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["keep.glb"]


def test_build_gltf_has_no_side_effects():
    scene = challenging_scene()
    before = scene.to_dict()
    build_gltf(scene)
    assert scene.to_dict() == before


def test_export_gltf_function_matches_method(tmp_path):
    scene = challenging_scene()
    assert (
        export_gltf(scene, tmp_path / "f.glb").path.read_bytes()
        == scene.export(tmp_path / "m.glb").path.read_bytes()
    )
