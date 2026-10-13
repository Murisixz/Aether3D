"""Element transforms in local and world space, orientation and normals."""

import math

import numpy as np
import pytest
from editing_helpers import box, outward_closed, snapshot, volume

from aether3d import EditError, Scene, ValidationError, create_sphere, math3d
from aether3d.editing import EditableMesh


def test_translate_vertices_edges_faces():
    m = box(2)
    top = m.select_faces(normal=((0, 0, 1), 1))
    m.translate(top, (0, 0, 1))
    assert m.bounds() == ((-1.0, -1.0, -1.0), (1.0, 1.0, 2.0)) and volume(m) == pytest.approx(12)
    edge = m.select_edges(nearest=(0, 1, 2))
    m.translate(edge, (0, 1, 0))
    assert outward_closed(m)
    (v,) = m.select_vertices(nearest=(-1, -1, -1)).vertices
    m.translate([v], (-1, 0, 0))
    assert m.position(v) == (-2.0, -1.0, -1.0)


def test_rotate_about_center_and_pivots():
    m = box(2)
    top = m.select_faces(normal=((0, 0, 1), 1))
    m.rotate(top, 90, (0, 0, 1))  # twist the top face about its own center
    assert outward_closed(m)  # a twisted prism: still closed and outward-facing
    corners = sorted(tuple(np.round(m.position(v), 9)) for v in m.to_vertices(top).vertices)
    assert corners == sorted(
        tuple(np.round(p, 9)) for p in [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)]
    )
    n = box(2)
    n.rotate(n.select_all(), 90, (1, 0, 0), pivot="origin")
    assert np.allclose(n.bounds(), ((-1, -1, -1), (1, 1, 1)))
    n.rotate(n.select_all(), 180, (0, 0, 1), pivot=(1, 0, 0))
    assert np.allclose(n.bounds()[0], (1, -1, -1))
    with pytest.raises(ValidationError):
        n.rotate(n.select_all(), 10, pivot="somewhere")


def test_scale_and_flatten_is_degenerate():
    m = box(2)
    m.scale(m.select_all(), (2, 1, 0.5))
    assert volume(m) == pytest.approx(8 * 2 * 0.5)
    before = snapshot(m)
    with pytest.raises(EditError, match="DEGENERATE_RESULT"):
        m.scale(m.select_all(), (1, 1, 0))
    assert snapshot(m) == before
    m.scale(m.select_all(), (1, 1, 0), allow_degenerate=True)
    assert "ZERO_AREA_FACES" in m.validate().codes()


def test_mirror_keeps_faces_outward():
    m = box(2)
    m.translate(m.select_all(), (2, 0, 0))
    r = m.scale(m.select_all(), (-1, 1, 1), pivot="origin")
    assert r.data["determinant"] == -1 and len(r.data["reversed_faces"]) == 6
    assert [d.code for d in r.diagnostics] == ["WINDING_REVERSED"]
    assert outward_closed(m) and volume(m) == pytest.approx(8)
    assert m.bounds() == ((-3.0, -1.0, -1.0), (-1.0, 1.0, 1.0))
    for f in m.face_ids():  # stored normals were mirrored too and still agree with the faces
        assert np.dot(m.face_normal(f), m.corner_values(f, "normal")[0]) > 0.999


def test_mirroring_an_attached_part_keeps_consistent_winding():
    m = box(2)
    top_face = m.select_faces(normal=((0, 0, 1), 1))
    (f,) = top_face.faces
    # Mirror the top vertices across the plane z = 2: the top face moves to z = 3 but is
    # still attached to the (stretched) sides, so its inside stays below it.
    r = m.scale(m.to_vertices(top_face), (1, 1, -1), pivot=(0, 0, 2))
    assert r.data["reversed_faces"] == []
    assert m.face_normal(f) == (0.0, 0.0, 1.0)
    assert np.allclose(m.corner_values(f, "normal"), [(0, 0, 1)] * 4)
    assert outward_closed(m) and volume(m) == pytest.approx(2 * 2 * 4)


def test_mirroring_a_detached_piece_rewinds_it():
    m = box(2)
    m.add_box(1, center=(5, 0, 0))
    piece = m.linked_faces(m.select_faces(nearest=(5.5, 0, 0)))
    r = m.scale(piece, (-1, 1, 1), pivot=(0, 0, 0))
    assert sorted(r.data["reversed_faces"]) == sorted(piece.faces)
    assert outward_closed(m) and volume(m) == pytest.approx(8 + 1)


def test_partial_mirror_that_folds_faces_is_refused():
    m = box(2)
    top = m.select_vertices(box=((-2, -2, 0.5), (2, 2, 2)))
    before = snapshot(m)
    with pytest.raises(EditError, match="DEGENERATE_RESULT"):
        m.scale(top, (-1, 1, 1))  # swaps the top corners: east/west faces become bowties
    assert snapshot(m) == before


def test_general_matrix_transform():
    m = box(2)
    shear = np.eye(4)
    shear[0, 2] = 0.5  # x += 0.5 z
    m.transform(m.select_all(), shear)
    assert volume(m) == pytest.approx(8)  # shear preserves volume
    with pytest.raises(ValidationError):
        m.transform(m.select_all(), np.ones((3, 3)))
    bad = np.eye(4)
    bad[3, 0] = 1
    with pytest.raises(ValidationError):
        m.transform(m.select_all(), bad)


def test_set_positions_and_move_along_normals():
    m = box(2)
    (v,) = m.select_vertices(nearest=(1, 1, 1)).vertices
    m.set_positions({v: (2, 2, 2)})
    assert m.position(v) == (2.0, 2.0, 2.0)
    with pytest.raises(ValidationError):
        m.set_positions({})
    n = box(2)
    n.move_along_normals(n.select_all(), 0.5)
    lo, hi = n.bounds()
    expected = 1 + 0.5 / math.sqrt(3)  # corner normal is the diagonal
    assert hi == pytest.approx((expected,) * 3)


def test_rigid_transform_transforms_stored_normals_exactly():
    s = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    s.rotate(s.select_all(), 37, (1, 2, 3))
    rotation = math3d.quat_to_matrix3(math3d.quat_from_axis_angle((1, 2, 3), 37))
    original = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    for f in s.face_ids():
        expected = np.array(original.corner_values(f, "normal")) @ rotation.T
        assert np.allclose(s.corner_values(f, "normal"), expected, atol=1e-12)
    assert not s._dirty_faces  # nothing needs recomputing: analytic normals survive


def world_vertices(obj):
    m = obj.world_matrix
    return obj.mesh.positions.astype(float) @ m[:3, :3].T + m[:3, 3]


@pytest.mark.parametrize(
    "parent_scale, parent_euler",
    [((1, 1, 1), (0, 0, 0)), ((2, 0.5, 3), (10, 20, 30)), ((-1, 2, 1), (0, 45, 0))],
)
def test_world_space_translation_moves_by_world_offset(parent_scale, parent_euler):
    scene = Scene()
    parent = scene.add_object("P", position=(1, 2, 3), scale=parent_scale)
    parent.rotation_euler = parent_euler
    obj = scene.add_cube("C", size=1, parent=parent, position=(0.5, 0, 0))
    obj.rotation_euler = (5, -15, 70)
    before = world_vertices(obj)
    with obj.edit_mesh() as edit:
        edit.translate(edit.select_all(), (0, 0, 1), space="world")
    after = world_vertices(obj)
    assert np.allclose(np.sort(after - [0, 0, 1], axis=0), np.sort(before, axis=0), atol=1e-6)


def test_world_space_rotation_about_world_pivot():
    scene = Scene()
    obj = scene.add_cube("C", size=1, position=(3, 0, 0), scale=(1, 2, 1))
    obj.rotation_euler = (0, 0, 30)
    with obj.edit_mesh() as edit:
        edit.rotate(edit.select_all(), 90, (0, 0, 1), pivot=(0, 0, 0), space="world")
    pts = world_vertices(obj)
    center = pts.mean(axis=0)
    assert center == pytest.approx((0, 3, 0), abs=1e-6)  # (3, 0, 0) rotated 90 deg about world Z
    assert outward_closed_world(obj)


def outward_closed_world(obj) -> bool:
    from aether3d import inspect_mesh

    return inspect_mesh(obj.mesh).topology["volume"] > 0


def test_world_space_on_singular_transform_is_an_error():
    scene = Scene()
    obj = scene.add_cube("Flat", scale=(1, 0, 1))
    edit = obj.edit_mesh()
    with pytest.raises(EditError, match="SINGULAR_WORLD_TRANSFORM"):
        edit.translate(edit.select_all(), (1, 0, 0), space="world")
    edit.translate(edit.select_all(), (1, 0, 0))  # local space still works
    edit.discard()
