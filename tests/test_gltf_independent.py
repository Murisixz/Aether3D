"""Exported files are read back by an independent glTF implementation (trimesh)."""

import numpy as np
import pytest
from scenes import challenging_scene, world_vertices

from aether3d.gltf.axes import AXIS_MATRIX

trimesh = pytest.importorskip("trimesh")


@pytest.mark.parametrize("fmt", ["glb", "gltf"])
def test_trimesh_sees_the_same_world_geometry(tmp_path, fmt):
    scene = challenging_scene()
    path = tmp_path / f"scene.{fmt}"
    scene.export(path)
    loaded = trimesh.load(path, force="scene", process=False)

    by_name = {}
    for node in loaded.graph.nodes_geometry:
        matrix, geometry_name = loaded.graph[node]
        by_name[node] = (matrix, loaded.geometry[geometry_name], geometry_name)

    mesh_objects = [o for o in scene if o.mesh is not None]
    assert sorted(by_name) == sorted(o.name for o in mesh_objects)
    for obj in mesh_objects:
        matrix, geometry, _ = by_name[obj.name]
        # trimesh works in glTF's Y-up space: compare against our world vertices in Y-up.
        theirs = geometry.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        ours = world_vertices(obj) @ AXIS_MATRIX.T
        np.testing.assert_allclose(theirs, ours, atol=1e-5)
        np.testing.assert_array_equal(geometry.faces, obj.mesh.triangles)  # same winding


def test_trimesh_sees_instancing(tmp_path):
    scene = challenging_scene()
    scene.export(tmp_path / "scene.glb")
    loaded = trimesh.load(tmp_path / "scene.glb", force="scene", process=False)
    geometry_of = {node: loaded.graph[node][1] for node in loaded.graph.nodes_geometry}
    assert geometry_of["Moon"] == geometry_of["Mirrored Moon"]
    assert geometry_of["Hand"] == geometry_of["Hand Twin"]
    assert len(loaded.geometry) == 5


def test_trimesh_sees_closed_volumes(tmp_path):
    """An independent volume computation agrees (the conversion preserves orientation)."""
    from aether3d import Scene

    scene = Scene()
    scene.add_cube("Cube", size=2, position=(5, -3, 1))
    scene.add_sphere("Ball", radius=1.5, segments=48, rings=24)
    scene.export(tmp_path / "v.glb")
    loaded = trimesh.load(tmp_path / "v.glb", force="scene", process=False)
    ours = {r["used_by"][0]: r for r in scene.inspect_geometry().to_dict()["meshes"]}
    for obj in scene:
        geometry = loaded.geometry[loaded.graph[obj.name][1]].copy()
        geometry.merge_vertices()  # weld UV seams the same way we do
        assert geometry.is_watertight and geometry.is_winding_consistent
        assert geometry.volume == pytest.approx(ours[obj.id]["topology"]["volume"], rel=1e-6)
        assert geometry.volume > 0  # outward-facing after conversion
