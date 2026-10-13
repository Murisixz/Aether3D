import base64
import copy
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d import (
    AssetFormatError,
    ImportLimits,
    ResourceLimitError,
    Scene,
    UnsupportedFeatureError,
    ValidationError,
    import_gltf,
)
from aether3d.gltf.writer import encode_glb

KHRONOS = Path(__file__).parent / "data" / "khronos"


# --------------------------------------------------------------------------- #
# Khronos sample assets
# --------------------------------------------------------------------------- #


def codes(result):
    return sorted({d.code for d in result.diagnostics})


@pytest.mark.parametrize(
    "variant", ["glTF/Box.gltf", "glTF-Binary/Box.glb", "glTF-Embedded/Box.gltf"]
)
def test_box_variants_are_identical(variant):
    reference = import_gltf(KHRONOS / "Box/glTF/Box.gltf").scene
    result = import_gltf(KHRONOS / "Box" / variant)
    assert result.format == ("glb" if variant.endswith(".glb") else "gltf")
    assert result.generator == "COLLADA2GLTF"
    assert codes(result) == ["MATERIALS_IGNORED"]
    scene = result.scene
    assert [o.name for o in scene] == ["Node", "Mesh"]
    mesh_obj = scene.objects[1]
    assert mesh_obj.parent is scene.objects[0]
    assert mesh_obj.mesh.vertex_count == 24 and mesh_obj.mesh.triangle_count == 12
    np.testing.assert_array_equal(mesh_obj.mesh.positions, reference.objects[1].mesh.positions)
    # Box's root matrix rotates its Z-up content into glTF's Y-up space; converting
    # back to Aether3D's Z-up cancels that, leaving an axis-aligned unit cube.
    lo, hi = (
        scene.inspect_geometry().world_bounds["min"],
        scene.inspect_geometry().world_bounds["max"],
    )
    np.testing.assert_allclose(lo, [-0.5, -0.5, -0.5], atol=1e-6)
    np.testing.assert_allclose(hi, [0.5, 0.5, 0.5], atol=1e-6)
    report = scene.inspect_geometry().meshes[0]
    assert report.topology["closed"] and report.topology["volume"] == pytest.approx(1.0)
    assert report.status == "ok"


def test_interleaved_box_matches_box():
    a = Scene.import_file(KHRONOS / "Box/glTF-Binary/Box.glb")
    b = Scene.import_file(KHRONOS / "BoxInterleaved/glTF-Binary/BoxInterleaved.glb")
    ma, mb = a.objects[1].mesh, b.objects[1].mesh
    np.testing.assert_array_equal(np.sort(ma.positions, axis=0), np.sort(mb.positions, axis=0))
    np.testing.assert_array_equal(np.sort(ma.normals, axis=0), np.sort(mb.normals, axis=0))


def test_triangle_with_and_without_indices():
    a = Scene.import_file(KHRONOS / "Triangle/glTF/Triangle.gltf").objects[0].mesh
    b = Scene.import_file(KHRONOS / "TriangleWithoutIndices/glTF/TriangleWithoutIndices.gltf")
    b = b.objects[0].mesh
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.triangles, [[0, 1, 2]])
    np.testing.assert_array_equal(b.triangles, [[0, 1, 2]])
    # glTF (1, 0, 0), (0, 1, 0) -> Aether3D (1, 0, 0), (0, 0, 1)
    np.testing.assert_array_equal(a.positions, np.float32([[0, 0, 0], [1, 0, 0], [0, 0, 1]]))
    assert a.normals is None and a.uvs is None


def test_simple_meshes_share_one_mesh():
    result = import_gltf(KHRONOS / "SimpleMeshes/glTF/SimpleMeshes.gltf")
    a, b = result.scene.objects
    assert a.mesh is b.mesh
    assert a.position == (0.0, 0.0, 0.0) and b.position == (1.0, 0.0, 0.0)
    assert [o["id_source"] for o in result.objects] == ["generated", "generated"]


def test_sparse_accessor_is_applied():
    mesh = Scene.import_file(KHRONOS / "SimpleSparseAccessor/glTF/SimpleSparseAccessor.gltf")
    mesh = mesh.objects[0].mesh
    # Expected values decoded independently with struct: the base accessor is a flat
    # row; sparse indices 8, 10, 12 replace glTF y with 2, 3, 4.
    gltf_y = [0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 1, 4, 1]
    expected = [(x, 0, y) for x, y in zip(list(range(7)) * 2, gltf_y, strict=True)]
    np.testing.assert_array_equal(mesh.positions, np.float32(expected))


def test_primitive_modes():
    path = KHRONOS / "MeshPrimitiveModes/glTF/MeshPrimitiveModes.gltf"
    with pytest.raises(UnsupportedFeatureError) as info:
        import_gltf(path)
    assert {d.code for d in info.value.diagnostics} == {"PRIMITIVE_MODE_UNSUPPORTED"}
    assert len(info.value.diagnostics) == 4  # POINTS, LINES, LINE_LOOP, LINE_STRIP

    result = import_gltf(path, allow_lossy=True)
    assert result.lossy
    objects = {o.name: o for o in result.scene}
    assert sum(o.mesh is None for o in objects.values()) == 4
    areas = []
    for name, expected_triangles in [
        ("mesh with TRIANGLES", 6),
        ("mesh with GL_TRIANGLE_STRIP", 4),
        ("mesh with GL_TRIANGLE_FAN", 6),
    ]:
        mesh = objects[name].mesh
        assert mesh.triangle_count == expected_triangles
        p = mesh.positions.astype(float)
        t = mesh.triangles
        cross = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        areas.append(np.linalg.norm(cross, axis=1).sum() / 2)
        # Every triangle faces glTF +Z (the asset front), i.e. Aether3D -Y.
        unit = cross / np.linalg.norm(cross, axis=1)[:, None]
        np.testing.assert_allclose(unit, [[0, -1, 0]] * len(unit), atol=1e-6)
    # All three encodings describe the same hexagon of radius 1 (the sample's
    # coordinates are rounded to 3 decimals, hence the tolerance).
    assert areas[0] == areas[1] == areas[2]
    np.testing.assert_allclose(areas, [3 * np.sqrt(3) / 2] * 3, rtol=1e-4)


@pytest.mark.parametrize(
    "sample, code",
    [
        ("SimpleSkin/glTF-Embedded/SimpleSkin.gltf", "SKIN_IGNORED"),
        ("AnimatedMorphCube/glTF-Binary/AnimatedMorphCube.glb", "MORPH_TARGETS_IGNORED"),
    ],
)
def test_skins_and_morph_targets_are_reported_not_hidden(sample, code):
    with pytest.raises(UnsupportedFeatureError) as info:
        import_gltf(KHRONOS / sample)
    assert code in {d.code for d in info.value.diagnostics}
    assert all(d.lossy for d in info.value.diagnostics)
    result = import_gltf(KHRONOS / sample, allow_lossy=True)
    assert result.lossy and code in codes(result) and "ANIMATIONS_IGNORED" in codes(result)


def test_multiple_scenes_imports_default_scene_only():
    result = import_gltf(KHRONOS / "MultipleScenes/glTF/MultipleScenes.gltf")
    assert len(result.scene) == 1
    assert codes(result) == ["NODES_NOT_IN_SCENE", "OTHER_SCENES_IGNORED"]
    assert not result.lossy


def test_cameras_and_vertex_colors_are_reported():
    assert codes(import_gltf(KHRONOS / "Cameras/glTF-Embedded/Cameras.gltf")) == [
        "CAMERAS_IGNORED",
        "NODE_CAMERA_IGNORED",
    ]
    result = import_gltf(KHRONOS / "BoxVertexColors/glTF-Binary/BoxVertexColors.glb")
    (diag,) = result.diagnostics
    assert diag.code == "ATTRIBUTES_IGNORED" and "COLOR_0" in diag.message


def test_unicode_percent_encoded_uris_and_images_never_loaded(tmp_path):
    source = KHRONOS / "Unicode❤♻Test/glTF"
    result = import_gltf(source / "Unicode❤♻Test.gltf")
    assert result.scene.objects[0].name == "Unicode❤♻Mesh"  # unnamed node -> mesh name
    assert "IMAGES_IGNORED" in codes(result)
    # The texture is never read: the import still works without it.
    for name in ("Unicode❤♻Test.gltf", "Unicode❤♻Binary.bin"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    assert len(import_gltf(tmp_path / "Unicode❤♻Test.gltf").scene) == 1


def test_orientation_test_matches_independent_parser():
    trimesh = pytest.importorskip("trimesh")
    path = KHRONOS / "OrientationTest/glTF-Binary/OrientationTest.glb"
    ours = Scene.import_file(path)
    reference = trimesh.load(path, force="scene", process=False)
    compare_with_trimesh(ours, reference)


@pytest.mark.parametrize(
    "sample",
    [
        "Box/glTF/Box.gltf",
        "BoxInterleaved/glTF-Binary/BoxInterleaved.glb",
        "SimpleMeshes/glTF/SimpleMeshes.gltf",
        # SimpleSparseAccessor is excluded: trimesh ignores sparse substitution (verified
        # against an independent struct decode in test_sparse_accessor_is_applied).
        "TriangleWithoutIndices/glTF/TriangleWithoutIndices.gltf",
        "Cameras/glTF-Embedded/Cameras.gltf",
    ],
)
def test_samples_match_independent_parser(sample):
    trimesh = pytest.importorskip("trimesh")
    ours = Scene.import_file(KHRONOS / sample)
    compare_with_trimesh(ours, trimesh.load(KHRONOS / sample, force="scene", process=False))


def compare_with_trimesh(ours, reference):
    """World-space triangles from trimesh (Y-up) equal ours converted to Y-up."""
    from aether3d.gltf.axes import AXIS_MATRIX

    def triangle_set(points, faces):
        return np.sort(np.round(points[faces].reshape(-1, 9), 5), axis=0)

    expected = []
    for node in reference.graph.nodes_geometry:
        matrix, name = reference.graph[node]
        geometry = reference.geometry[name]
        points = geometry.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        expected.append(triangle_set(points, geometry.faces))
    actual = []
    for obj in ours:
        if obj.mesh is not None:
            m = obj.world_matrix
            points = (obj.mesh.positions.astype(float) @ m[:3, :3].T + m[:3, 3]) @ AXIS_MATRIX.T
            actual.append(triangle_set(points, obj.mesh.triangles.astype(int)))
    assert len(actual) == len(expected)
    np.testing.assert_allclose(
        np.sort(np.concatenate(actual), axis=0),
        np.sort(np.concatenate(expected), axis=0),
        atol=2e-5,
    )


# --------------------------------------------------------------------------- #
# Hand-built documents: malformed input, security, limits
# --------------------------------------------------------------------------- #


def triangle_binary():
    positions = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    indices = np.uint16([0, 1, 2, 0])  # padded to 4-byte alignment
    return positions.tobytes() + indices.tobytes()


def minimal_document(uri=None):
    binary = triangle_binary()
    buffer = {"byteLength": len(binary)}
    if uri is not None:
        buffer["uri"] = uri
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Tri", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],  # fmt: skip
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "buffers": [buffer],
    }


def data_uri(binary):
    return "data:application/octet-stream;base64," + base64.b64encode(binary).decode()


def write_gltf(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def embedded(mutate=None):
    doc = minimal_document(data_uri(triangle_binary()))
    if mutate:
        mutate(doc)
    return doc


def test_minimal_documents_import(tmp_path):
    scene = import_gltf(write_gltf(tmp_path / "t.gltf", embedded())).scene
    assert scene.objects[0].mesh.triangle_count == 1
    (tmp_path / "t.bin").write_bytes(triangle_binary())
    assert len(import_gltf(write_gltf(tmp_path / "x.gltf", minimal_document("t.bin"))).scene) == 1
    (tmp_path / "t.glb").write_bytes(encode_glb(minimal_document(), triangle_binary()))
    assert len(import_gltf(tmp_path / "t.glb").scene) == 1


@pytest.mark.parametrize(
    "mutate, code, location",
    [
        (lambda d: d["accessors"][1].update(count=4), "ACCESSOR_OUT_OF_BOUNDS", "/accessors/1"),
        (lambda d: d["bufferViews"][0].update(byteLength=999), "BUFFER_VIEW_OUT_OF_BOUNDS", None),
        (lambda d: d["accessors"][0].update(componentType=5123), "INVALID_ACCESSOR_TYPE", None),
        (lambda d: d["accessors"][0].update(type="VEC2"), "INVALID_ACCESSOR_TYPE", None),
        (lambda d: d["accessors"][0].update(count=0), "INVALID_ASSET", "/accessors/0"),
        (lambda d: d["meshes"][0]["primitives"][0].update(indices=7), "INVALID_REFERENCE", None),
        (lambda d: d["nodes"][0].update(mesh=3), "INVALID_REFERENCE", "/nodes/0/mesh"),
        (lambda d: d["meshes"][0]["primitives"][0].update(mode=9), "INVALID_ASSET", None),
        (lambda d: d["meshes"][0].update(primitives=[]), "INVALID_ASSET", None),
        (lambda d: d["nodes"][0].update(rotation=[0, 0, 0, 0]), "INVALID_TRANSFORM", "/nodes/0"),
        (lambda d: d["nodes"][0].update(scale=[1, 1]), "INVALID_ASSET", "/nodes/0/scale"),
        (lambda d: d["nodes"][0].update(matrix=[1] * 16), "NODE_MATRIX_INVALID", None),
        (
            lambda d: d["nodes"][0].update(
                matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], translation=[1, 2, 3]
            ),
            "INVALID_ASSET",
            "/nodes/0",
        ),  # fmt: skip
        (lambda d: d["nodes"][0].update(children=[0]), "INVALID_HIERARCHY", None),
        (
            lambda d: (d["nodes"].append({"children": [0]}), d["nodes"].append({"children": [0]})),
            "INVALID_HIERARCHY",
            None,
        ),
        (
            lambda d: (d["nodes"].append({"children": [2]}), d["nodes"].append({"children": [1]})),
            "INVALID_HIERARCHY",
            None,
        ),
        (lambda d: d["nodes"].append({"children": [0]}), "INVALID_HIERARCHY", "/scenes/0/nodes"),
        (lambda d: d["scenes"][0].update(nodes=[0, 0]), "INVALID_HIERARCHY", None),
        (lambda d: d.update(scene=5), "INVALID_REFERENCE", "/scene"),
        (lambda d: d.pop("asset"), "INVALID_GLTF", "/asset"),
        (lambda d: d["buffers"][0].update(byteLength=1000), "BUFFER_TOO_SHORT", None),
        (lambda d: d["buffers"][0].update(uri="data:text/plain;base64,AAAA"), "INVALID_URI", None),
        (
            lambda d: d["buffers"][0].update(uri="data:application/octet-stream,AAAA"),
            "INVALID_URI",
            None,
        ),  # fmt: skip
        (
            lambda d: d["buffers"][0].update(uri="data:application/octet-stream;base64,@@@@"),
            "INVALID_URI",
            None,
        ),  # fmt: skip
        (
            lambda d: d["accessors"].__setitem__(1, {**d["accessors"][1], "componentType": 5126}),
            "INVALID_ACCESSOR_TYPE",
            None,
        ),
        (
            lambda d: d["meshes"][0]["primitives"][0]["attributes"].update(NORMAL=1),
            "INVALID_ACCESSOR_TYPE",
            None,
        ),
    ],
)
def test_malformed_documents(tmp_path, mutate, code, location):
    path = write_gltf(tmp_path / "bad.gltf", embedded(mutate))
    with pytest.raises(AssetFormatError) as info:
        import_gltf(path)
    assert info.value.code == code, info.value
    if location is not None:
        assert info.value.location == location


def test_numeric_overflow_in_transform(tmp_path):
    text = json.dumps(embedded()).replace(
        '"name": "Tri"', '"name": "Tri", "translation": [0, 0, 1e400]'
    )
    (tmp_path / "o.gltf").write_text(text)
    with pytest.raises(AssetFormatError) as info:
        import_gltf(tmp_path / "o.gltf")
    assert info.value.code == "INVALID_TRANSFORM" and info.value.location == "/nodes/0"


def test_index_out_of_range(tmp_path):
    binary = bytearray(triangle_binary())
    binary[38:40] = struct.pack("<H", 9)  # second index -> 9
    doc = minimal_document(data_uri(bytes(binary)))
    with pytest.raises(AssetFormatError) as info:
        import_gltf(write_gltf(tmp_path / "i.gltf", doc))
    assert info.value.code == "INDEX_OUT_OF_RANGE"
    assert info.value.location == "/meshes/0/primitives/0/indices"


def test_non_finite_vertex_data_is_rejected(tmp_path):
    binary = bytearray(triangle_binary())
    binary[0:4] = struct.pack("<f", float("nan"))
    with pytest.raises(AssetFormatError) as info:
        import_gltf(write_gltf(tmp_path / "n.gltf", minimal_document(data_uri(bytes(binary)))))
    assert info.value.code == "INVALID_MESH_DATA" and "finite" in info.value.message


def test_triangle_count_must_be_multiple_of_three(tmp_path):
    def mutate(d):
        d["accessors"][1]["count"] = 2

    with pytest.raises(AssetFormatError) as info:
        import_gltf(write_gltf(tmp_path / "c.gltf", embedded(mutate)))
    assert info.value.code == "INVALID_INDEX_COUNT"


@pytest.mark.parametrize(
    "text, code",
    [
        ("{nope", "INVALID_JSON"),
        ("[]", "INVALID_JSON"),
        ('{"asset": {"version": "2.0"}, "asset": {}}', "INVALID_JSON"),
        ('{"asset": {"version": NaN}}', "INVALID_JSON"),
        ("[" * 100_000 + "]" * 100_000, "INVALID_JSON"),
    ],
)
def test_invalid_json(tmp_path, text, code):
    (tmp_path / "j.gltf").write_text(text)
    with pytest.raises(AssetFormatError) as info:
        import_gltf(tmp_path / "j.gltf")
    assert info.value.code == code


def test_non_utf8_json(tmp_path):
    (tmp_path / "u.gltf").write_bytes(b'{"asset": "\xff"}')
    with pytest.raises(AssetFormatError, match="UTF-8"):
        import_gltf(tmp_path / "u.gltf")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data[:10], "too short"),
        (lambda data: b"glTF" + struct.pack("<I", 1) + data[8:], "container version"),
        (lambda data: data[:8] + struct.pack("<I", len(data) + 4) + data[12:], "declares"),
        (lambda data: data + b"\0\0\0\0", "declares"),
        (
            lambda data: data[:12] + struct.pack("<I", 10_000) + data[16:],
            "extends past",
        ),
        (lambda data: data[:16] + b"BIN\0" + data[20:], "first GLB chunk must be JSON"),
    ],
)
def test_malformed_glb(tmp_path, mutate, message):
    data = encode_glb(minimal_document(), triangle_binary())
    (tmp_path / "b.glb").write_bytes(mutate(data))
    with pytest.raises(AssetFormatError, match=message) as info:
        import_gltf(tmp_path / "b.glb")
    assert info.value.code == "INVALID_GLB"


def test_glb_without_bin_chunk(tmp_path):
    (tmp_path / "b.glb").write_bytes(encode_glb(minimal_document(), b""))
    with pytest.raises(AssetFormatError) as info:
        import_gltf(tmp_path / "b.glb")
    assert info.value.code == "MISSING_RESOURCE"


def test_glb_json_padded_with_nul_bytes_is_accepted(tmp_path):
    document = json.dumps(minimal_document(), separators=(",", ":")).encode()
    document += b"\0" * (-len(document) % 4 or 4)
    binary = triangle_binary()
    chunks = struct.pack("<II", len(document), 0x4E4F534A) + document
    chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    (tmp_path / "nul.glb").write_bytes(
        struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks
    )
    assert len(import_gltf(tmp_path / "nul.glb").scene) == 1


def test_glb_unknown_chunks_are_ignored(tmp_path):
    data = encode_glb(minimal_document(), triangle_binary())
    extra = struct.pack("<II", 4, 0x12345678) + b"abcd"
    data = data[:8] + struct.pack("<I", len(data) + len(extra)) + data[12:] + extra
    (tmp_path / "x.glb").write_bytes(data)
    assert len(import_gltf(tmp_path / "x.glb").scene) == 1


@pytest.mark.parametrize(
    "uri",
    [
        "../secret.bin",
        "sub/../../secret.bin",
        "%2E%2E/secret.bin",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "C:secret.bin",
        "\\\\server\\share\\x.bin",
        "file:///etc/passwd",
        "https://example.com/x.bin",
        "//example.com/x.bin",
        "t.bin?x=1",
        "t.bin#frag",
        "dir\\t.bin",
        "t.bin%00.png",
    ],
)
def test_unsafe_uris_are_refused(tmp_path, uri):
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    (tmp_path / "secret.bin").write_bytes(triangle_binary())
    (asset_dir / "t.bin").write_bytes(triangle_binary())
    path = write_gltf(asset_dir / "a.gltf", minimal_document(uri))
    with pytest.raises(AssetFormatError) as info:
        import_gltf(path)
    assert info.value.code == "UNSAFE_URI"
    assert info.value.location == "/buffers/0/uri"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlink_escaping_asset_directory_is_refused(tmp_path):
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    (tmp_path / "secret.bin").write_bytes(triangle_binary())
    os.symlink(tmp_path / "secret.bin", asset_dir / "link.bin")
    with pytest.raises(AssetFormatError) as info:
        import_gltf(write_gltf(asset_dir / "a.gltf", minimal_document("link.bin")))
    assert info.value.code == "UNSAFE_URI"


def test_relative_subdirectory_and_percent_encoding_are_allowed(tmp_path):
    (tmp_path / "data dir").mkdir()
    (tmp_path / "data dir" / "t.bin").write_bytes(triangle_binary())
    path = write_gltf(tmp_path / "a.gltf", minimal_document("data%20dir/t.bin"))
    assert len(import_gltf(path).scene) == 1


def test_missing_external_buffer(tmp_path):
    path = write_gltf(tmp_path / "a.gltf", minimal_document("missing.bin"))
    with pytest.raises(AssetFormatError) as info:
        import_gltf(path)
    assert info.value.code == "MISSING_RESOURCE" and info.value.location == "/buffers/0/uri"


def test_missing_file_and_bad_extension(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_gltf(tmp_path / "nope.glb")
    (tmp_path / "x.obj").write_text("v 0 0 0")
    with pytest.raises(ValidationError, match="supported: .glb, .gltf"):
        Scene.import_file(tmp_path / "x.obj")


@pytest.mark.parametrize(
    "limits",
    [
        ImportLimits(max_file_bytes=100),
        ImportLimits(max_buffer_bytes=10),
        ImportLimits(max_nodes=0),
        ImportLimits(max_vertices=2),
        ImportLimits(max_triangles=0),
    ],
)
def test_resource_limits(tmp_path, limits):
    path = write_gltf(tmp_path / "t.gltf", embedded())
    with pytest.raises(ResourceLimitError):
        import_gltf(path, limits=limits)


def test_huge_accessor_without_buffer_view_is_refused_before_allocating(tmp_path):
    def mutate(d):
        d["accessors"][0] = {"componentType": 5126, "count": 10**12, "type": "VEC3"}

    with pytest.raises(ResourceLimitError):
        import_gltf(write_gltf(tmp_path / "h.gltf", embedded(mutate)))


def test_accessor_without_buffer_view_is_zero_filled(tmp_path):
    def mutate(d):
        d["accessors"][0] = {"componentType": 5126, "count": 3, "type": "VEC3"}

    mesh = import_gltf(write_gltf(tmp_path / "z.gltf", embedded(mutate))).scene.objects[0].mesh
    np.testing.assert_array_equal(mesh.positions, np.zeros((3, 3), np.float32))


def test_required_extensions_are_rejected(tmp_path):
    def mutate(d):
        d["extensionsUsed"] = d["extensionsRequired"] = ["KHR_draco_mesh_compression"]

    with pytest.raises(UnsupportedFeatureError) as info:
        import_gltf(write_gltf(tmp_path / "e.gltf", embedded(mutate)), allow_lossy=True)
    assert info.value.diagnostics[0].code == "REQUIRED_EXTENSION_UNSUPPORTED"


def test_optional_extensions_and_unsupported_versions(tmp_path):
    def optional(d):
        d["extensionsUsed"] = ["KHR_materials_unlit"]

    result = import_gltf(write_gltf(tmp_path / "o.gltf", embedded(optional)))
    assert codes(result) == ["EXTENSION_IGNORED"]
    for version in ("1.0", "3.0"):
        doc = embedded(lambda d, v=version: d["asset"].update(version=v))
        with pytest.raises(UnsupportedFeatureError):
            import_gltf(write_gltf(tmp_path / "v.gltf", doc))


def test_sheared_matrix_is_lossy_and_trs_matrix_is_exact(tmp_path):
    shear = [1, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    doc = embedded(lambda d: d["nodes"][0].update(matrix=shear))
    path = write_gltf(tmp_path / "s.gltf", doc)
    with pytest.raises(UnsupportedFeatureError, match="NODE_MATRIX_NOT_TRS"):
        import_gltf(path)
    assert import_gltf(path, allow_lossy=True).lossy

    # A TRS-decomposable matrix (rotate 90 deg about glTF Y, scale 2, translate) is exact.
    trs = [0, 0, -2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 2, 3, 1]
    obj = import_gltf(write_gltf(tmp_path / "m.gltf", embedded(
        lambda d: d["nodes"][0].update(matrix=trs)))).scene.objects[0]  # fmt: skip
    assert obj.position == (1.0, -3.0, 2.0)
    assert obj.scale == pytest.approx((2, 2, 2))
    np.testing.assert_allclose(obj.rotation_euler, (0, 0, 90), atol=1e-9)


def test_names_are_deduplicated_and_ids_restored(tmp_path):
    def mutate(d):
        d["nodes"] = [
            {"name": "Twin", "children": [1, 2, 3]},
            {"name": "Twin", "extras": {"aether3d": {"id": "kept-id"}}},
            {"name": "  ", "extras": {"aether3d": {"id": "kept-id"}}},
            {"extras": {"aether3d": "garbage"}},
        ]

    result = import_gltf(write_gltf(tmp_path / "n.gltf", embedded(mutate)))
    names = [o["name"] for o in result.objects]
    assert names == ["Twin", "Twin.001", "Node", "Node.001"]
    sources = [o["id_source"] for o in result.objects]
    assert sources == ["generated", "extras", "generated", "generated"]
    assert result.scene.get("kept-id").name == "Twin.001"
    assert [(d.code, d.location) for d in result.diagnostics] == [
        ("NAME_CHANGED", "/nodes/1/name"),
        ("STORED_ID_REJECTED", "/nodes/2/extras"),  # duplicate of node 1's id
        ("STORED_ID_REJECTED", "/nodes/3/extras"),  # not an object
    ]


def test_import_has_no_side_effects_on_failure(tmp_path):
    doc = embedded(lambda d: d["accessors"][1].update(count=4))
    original = copy.deepcopy(doc)
    path = write_gltf(tmp_path / "bad.gltf", doc)
    before = sorted(tmp_path.iterdir())
    with pytest.raises(AssetFormatError):
        import_gltf(path)
    assert sorted(tmp_path.iterdir()) == before
    assert doc == original


def test_no_scene_imports_all_roots(tmp_path):
    def mutate(d):
        del d["scene"], d["scenes"]
        d["nodes"].append({"name": "Second"})

    result = import_gltf(write_gltf(tmp_path / "ns.gltf", embedded(mutate)))
    assert [o.name for o in result.scene] == ["Tri", "Second"]
    assert codes(result) == ["NO_SCENE"]


def test_normalized_texcoords_are_decoded(tmp_path):
    binary = triangle_binary() + np.uint8([0, 255, 255, 0, 128, 128, 0, 0]).tobytes()

    def mutate(d):
        d["buffers"][0] = {"byteLength": len(binary), "uri": data_uri(binary)}
        d["bufferViews"].append({"buffer": 0, "byteOffset": 44, "byteLength": 6})
        d["accessors"].append(
            {"bufferView": 2, "componentType": 5121, "normalized": True, "count": 3, "type": "VEC2"}
        )
        d["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 2

    uvs = import_gltf(write_gltf(tmp_path / "uv.gltf", embedded(mutate))).scene.objects[0].mesh.uvs
    # glTF (0, 1), (1, 0), (128/255, 128/255) with v flipped to a bottom-left origin.
    np.testing.assert_allclose(uvs, [[0, 0], [1, 1], [128 / 255, 1 - 128 / 255]], atol=1e-7)


def test_primitives_are_merged_and_mismatched_attributes_dropped(tmp_path):
    def mutate(d):
        prim = d["meshes"][0]["primitives"][0]
        d["meshes"][0]["primitives"].append(copy.deepcopy(prim))

    result = import_gltf(write_gltf(tmp_path / "p.gltf", embedded(mutate)))
    mesh = result.scene.objects[0].mesh
    assert mesh.vertex_count == 6
    np.testing.assert_array_equal(mesh.triangles, [[0, 1, 2], [3, 4, 5]])
    assert codes(result) == ["PRIMITIVES_MERGED"]
