"""Editing sessions on scene objects: commit, rollback, instancing, persistence, export."""

import numpy as np
import pytest

from aether3d import (
    DetachedObjectError,
    EditError,
    Scene,
    ValidationError,
    import_gltf,
    inspect_mesh,
)


def top_extrude(edit, distance=1.0):
    top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
    return edit.extrude_faces(top, distance)


def test_context_manager_commits_and_closes():
    scene = Scene()
    cube = scene.add_cube("Cube", size=2)
    original = cube.mesh
    with cube.edit_mesh() as edit:
        top_extrude(edit)
    assert cube.mesh is not original and edit.closed
    assert inspect_mesh(cube.mesh).topology["volume"] == pytest.approx(12)
    result = edit.result
    assert result.changed and result.objects == [cube.id] and result.operations == ["extrude_faces"]
    assert result.validation.ok and result.inspection.status == "ok"
    with pytest.raises(EditError, match="SESSION_CLOSED"):
        edit.translate(edit.select_all(), (1, 0, 0))
    with pytest.raises(EditError, match="SESSION_CLOSED"):
        edit.commit()


def test_exception_discards_everything():
    scene = Scene()
    cube = scene.add_cube("Cube")
    original = cube.mesh
    with pytest.raises(RuntimeError), cube.edit_mesh() as edit:
        top_extrude(edit)
        raise RuntimeError("agent changed its mind")
    assert cube.mesh is original and edit.closed


def test_failed_operation_inside_session_keeps_session_usable():
    scene = Scene()
    cube = scene.add_cube("Cube", size=2)
    with cube.edit_mesh() as edit:
        with pytest.raises(EditError):
            edit.inset_faces(edit.select_faces(normal=((0, 0, 1), 1)), 10)
        top_extrude(edit, 0.5)
    assert inspect_mesh(cube.mesh).topology["volume"] == pytest.approx(10)


def test_instancing_is_preserved_unless_shared_scope():
    scene = Scene()
    a = scene.add_cube("A", size=2)
    b = scene.add_object("B", mesh=a.mesh, position=(3, 0, 0))
    c = scene.add_object("C", mesh=a.mesh, position=(6, 0, 0))
    shared = a.mesh
    with a.edit_mesh() as edit:
        top_extrude(edit)
    assert b.mesh is shared and c.mesh is shared and a.mesh is not shared
    edit = b.edit_mesh()
    top_extrude(edit, 2)
    result = edit.commit(scope="shared")
    assert sorted(result.objects) == sorted([b.id, c.id])
    assert b.mesh is c.mesh and b.mesh is not shared and a.mesh is not b.mesh
    with pytest.raises(ValidationError):
        c.edit_mesh().commit(scope="everyone")


def test_no_op_session_keeps_the_same_mesh_object():
    scene = Scene()
    cube = scene.add_cube("Cube")
    original = cube.mesh
    with cube.edit_mesh() as edit:
        edit.select_faces(normal=((0, 0, 1), 1))
    assert cube.mesh is original and not edit.result.changed


def test_stale_session_is_refused():
    scene = Scene()
    cube = scene.add_cube("Cube")
    edit = cube.edit_mesh()
    top_extrude(edit)
    cube.mesh = scene.add_sphere("S").mesh
    with pytest.raises(EditError, match="STALE_SESSION"):
        edit.commit()


def test_validation_failure_blocks_commit_but_keeps_session_open():
    scene = Scene()
    obj = scene.add_object("Bad")
    edit = obj.edit_mesh()
    ids = [
        edit.create_vertex(p).data["vertex"] for p in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    ]
    f = edit.create_face(ids).data["face"]
    # Force a self-intersecting polygon: edge 0-1 now crosses edge 2-3.
    edit.set_positions({ids[1]: (1.5, 3, 0)})  # non-zero signed area, so not "degenerate"
    assert "SELF_INTERSECTING_FACES" in edit.validate().codes()
    with pytest.raises(EditError, match="VALIDATION_FAILED"):
        edit.commit()
    assert not edit.closed and obj.mesh is None
    edit.undo()
    result = edit.commit()
    assert result.mesh.triangle_count == 2 and obj.mesh is result.mesh
    assert f is not None


def test_building_from_scratch_and_deleting_everything():
    scene = Scene()
    obj = scene.add_object("Empty")
    with obj.edit_mesh() as edit:
        edit.add_box((1, 2, 3))
    assert inspect_mesh(obj.mesh).topology["volume"] == pytest.approx(6)
    with obj.edit_mesh() as edit:
        edit.delete_faces(edit.select_all())
    assert obj.mesh is None
    assert [d.code for d in edit.result.diagnostics] == ["MESH_EMPTY"]


def test_world_matrix_is_bound_and_detached_objects_refused():
    scene = Scene()
    parent = scene.add_object("P", position=(0, 0, 5))
    obj = scene.add_cube("C", parent=parent)
    edit = obj.edit_mesh()
    assert np.allclose(edit.mesh.world_matrix, obj.world_matrix)
    edit.discard()
    scene.remove(obj)
    with pytest.raises(DetachedObjectError):
        obj.edit_mesh()


def test_repeated_sessions_and_persistence(tmp_path):
    scene = Scene("Workshop")
    obj = scene.add_cube("Block", size=2)
    for distance in (1.0, 0.5, 0.25):
        with obj.edit_mesh() as edit:
            top_extrude(edit, distance)
    assert inspect_mesh(obj.mesh).topology["volume"] == pytest.approx(4 * 3.75)
    with obj.edit_mesh() as edit:
        assert edit.summary()["face_sizes"] == {"4": 18}  # quads rebuilt every time
    scene.save(tmp_path / "w.aether")
    loaded = Scene.load(tmp_path / "w.aether")
    assert loaded.get(obj.id).mesh == obj.mesh
    with loaded.get(obj.id).edit_mesh() as edit:
        edit.bevel_edges(edit.select_edges(nearest=(1, 1, 4.75)), 0.1)
    loaded.save(tmp_path / "w2.aether")
    assert Scene.load(tmp_path / "w2.aether").get(obj.id).mesh == loaded.get(obj.id).mesh


def test_edited_mesh_exports_and_reimports(tmp_path):
    scene = Scene()
    obj = scene.add_cube("Block", size=2)
    with obj.edit_mesh() as edit:
        top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
        edit.inset_faces(top, 0.3)
        edit.extrude_faces(top, 0.8)
    scene.export(tmp_path / "e.glb")
    back = import_gltf(tmp_path / "e.glb")
    assert back.diagnostics == []
    mesh = back.scene.get(obj.id).mesh
    assert np.array_equal(mesh.positions, obj.mesh.positions)
    assert np.array_equal(mesh.normals, obj.mesh.normals)
    assert np.array_equal(mesh.triangles, obj.mesh.triangles)
    assert np.abs(mesh.uvs - obj.mesh.uvs).max() <= 2.5e-7  # the documented glTF UV flip
