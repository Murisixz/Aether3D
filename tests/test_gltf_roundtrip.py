"""Aether3D -> glTF -> Aether3D preserves everything glTF can carry."""

from pathlib import Path

import numpy as np
import pytest
from scenes import challenging_scene, world_vertices

from aether3d import Scene, import_gltf

KHRONOS = Path(__file__).parent / "data" / "khronos"
UV_TOLERANCE = 2.5e-7  # two float32 roundings of v -> 1 - v -> v


@pytest.fixture(params=["glb", "gltf"])
def fmt(request):
    return request.param


def assert_equivalent(original: Scene, loaded: Scene):
    assert loaded.name == original.name
    assert [o.id for o in loaded] == [o.id for o in original]
    for a in original:
        b = loaded.get(a.id)
        assert b.name == a.name
        assert (b.parent and b.parent.id) == (a.parent and a.parent.id)
        assert [c.id for c in b.children] == [c.id for c in a.children]
        assert b.transform == a.transform, a.name  # bit-exact local transforms
        np.testing.assert_array_equal(b.world_matrix, a.world_matrix)
        assert (a.mesh is None) == (b.mesh is None)
        if a.mesh is not None:
            np.testing.assert_array_equal(b.mesh.positions, a.mesh.positions)
            np.testing.assert_array_equal(b.mesh.normals, a.mesh.normals)
            np.testing.assert_array_equal(b.mesh.triangles, a.mesh.triangles)
            assert np.abs(b.mesh.uvs - a.mesh.uvs).max() <= UV_TOLERANCE


def test_challenging_scene_round_trip(tmp_path, fmt):
    scene = challenging_scene()
    scene.export(tmp_path / f"s.{fmt}")
    result = import_gltf(tmp_path / f"s.{fmt}")
    assert_equivalent(scene, result.scene)
    assert result.diagnostics == []
    assert all(o["id_source"] == "extras" for o in result.objects)
    assert [o["node"] for o in result.objects] == list(range(len(scene)))


def test_instancing_survives_round_trip(tmp_path, fmt):
    scene = challenging_scene()
    scene.export(tmp_path / f"s.{fmt}")
    loaded = Scene.import_file(tmp_path / f"s.{fmt}")
    assert loaded.get_by_name("Moon").mesh is loaded.get_by_name("Mirrored Moon").mesh
    assert loaded.get_by_name("Hand").mesh is loaded.get_by_name("Hand Twin").mesh
    assert loaded.get_by_name("Cube").mesh is not loaded.get_by_name("Hand").mesh


def test_world_space_geometry_is_preserved(tmp_path, fmt):
    scene = challenging_scene()
    scene.export(tmp_path / f"s.{fmt}")
    loaded = Scene.import_file(tmp_path / f"s.{fmt}")
    for obj in scene:
        if obj.mesh is not None:
            np.testing.assert_array_equal(world_vertices(loaded.get(obj.id)), world_vertices(obj))


def test_geometry_report_is_unchanged_by_round_trip(tmp_path, fmt):
    scene = challenging_scene()
    scene.export(tmp_path / f"s.{fmt}")
    before = scene.inspect_geometry().to_dict()
    after = Scene.import_file(tmp_path / f"s.{fmt}").inspect_geometry().to_dict()
    for key in ("status", "summary", "world_bounds", "objects", "diagnostics"):
        assert after[key] == before[key], key
    for m_before, m_after in zip(before["meshes"], after["meshes"], strict=True):
        m_after["uv_bounds"] = m_before["uv_bounds"]  # UVs are within UV_TOLERANCE only
        assert m_after == m_before


def test_second_round_trip_is_byte_identical(tmp_path):
    challenging_scene().export(tmp_path / "a.glb")
    Scene.import_file(tmp_path / "a.glb").export(tmp_path / "b.glb")
    Scene.import_file(tmp_path / "b.glb").export(tmp_path / "c.glb")
    # UV flips may move a value by one ulp once; after that the data is stable.
    assert (tmp_path / "b.glb").read_bytes() == (tmp_path / "c.glb").read_bytes()


def test_aether_project_round_trip_through_gltf(tmp_path):
    scene = challenging_scene()
    scene.export(tmp_path / "s.glb")
    imported = Scene.import_file(tmp_path / "s.glb")
    imported.save(tmp_path / "s.aether")
    assert_equivalent(scene, Scene.load(tmp_path / "s.aether"))


def test_zero_scale_round_trip(tmp_path):
    scene = Scene()
    flat = scene.add_object("Flat", scale=(1, 0, 1))
    scene.add_cube("Child", parent=flat, position=(0, 1, 0))
    scene.export(tmp_path / "z.glb")
    assert_equivalent(scene, Scene.import_file(tmp_path / "z.glb"))


KHRONOS_STATIC = [
    "Box/glTF/Box.gltf",
    "Box/glTF-Binary/Box.glb",
    "Box/glTF-Embedded/Box.gltf",
    "BoxInterleaved/glTF-Binary/BoxInterleaved.glb",
    "BoxVertexColors/glTF-Binary/BoxVertexColors.glb",
    "Cameras/glTF-Embedded/Cameras.gltf",
    "MultipleScenes/glTF/MultipleScenes.gltf",
    "OrientationTest/glTF-Binary/OrientationTest.glb",
    "SimpleMeshes/glTF/SimpleMeshes.gltf",
    "SimpleSparseAccessor/glTF/SimpleSparseAccessor.gltf",
    "Triangle/glTF/Triangle.gltf",
    "TriangleWithoutIndices/glTF/TriangleWithoutIndices.gltf",
    "Unicode❤♻Test/glTF/Unicode❤♻Test.gltf",
]


@pytest.mark.parametrize("sample", KHRONOS_STATIC)
def test_khronos_samples_reexport_without_geometric_change(tmp_path, sample):
    first = Scene.import_file(KHRONOS / sample)
    first.export(tmp_path / "re.glb")
    second = import_gltf(tmp_path / "re.glb")
    assert second.diagnostics == []
    assert [o.name for o in second.scene] == [o.name for o in first]
    for a in first:
        b = second.scene.get(a.id)  # generated ids are preserved by the re-export
        assert b.transform == a.transform
        np.testing.assert_array_equal(b.world_matrix, a.world_matrix)
        if a.mesh is not None:
            np.testing.assert_array_equal(world_vertices(b), world_vertices(a))
            np.testing.assert_array_equal(b.mesh.triangles, a.mesh.triangles)
            if a.mesh.normals is not None:
                np.testing.assert_allclose(b.mesh.normals, a.mesh.normals, atol=1e-6)
            if a.mesh.uvs is not None:
                assert np.abs(b.mesh.uvs - a.mesh.uvs).max() <= UV_TOLERANCE
        else:
            assert b.mesh is None
