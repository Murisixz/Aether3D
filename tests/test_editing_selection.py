"""Selection by ids, predicates, normals, adjacency; ambiguity and count guards."""

import numpy as np
import pytest
from editing_helpers import box, grid

from aether3d import EditError, Scene, ValidationError, create_sphere
from aether3d.editing import EditableMesh, Selection


def test_select_by_ids_and_missing_ids():
    m = box()
    assert m.select_faces([0, 1]).faces == {0, 1}
    with pytest.raises(EditError) as info:
        m.select_vertices([0, 99])
    assert info.value.code == "INVALID_ELEMENT" and info.value.elements == {"vertices": [99]}


def test_select_by_box_sphere_and_plane():
    m = box(2)
    top_verts = m.select_vertices(box=((-2, -2, 0.5), (2, 2, 2)))
    assert len(top_verts.vertices) == 4
    assert len(m.select_vertices(sphere=((1, 1, 1), 0.1)).vertices) == 1
    front = m.select_faces(plane={"point": (0, 0, 0), "normal": (1, 0, 0), "side": "front"})
    assert len(front.faces) == 1
    on = m.select_vertices(plane={"point": (1, 0, 0), "normal": (1, 0, 0), "side": "on"})
    assert len(on.vertices) == 4
    behind = m.select_edges(plane={"normal": (0, 0, 1), "side": "back"})
    assert len(behind.edges) == 4  # bottom edges have midpoints below z = 0
    with pytest.raises(ValidationError):
        m.select_faces(plane={"normal": (0, 0, 0)})


def test_test_modes_for_edges_and_faces():
    g = grid(4, 4, 4)
    region = ((-1.01, -1.01, -1), (1.01, 1.01, 1))
    assert len(g.select_faces(box=region).faces) == 4  # centers inside
    assert len(g.select_faces(box=region, test="all").faces) == 4
    # The box holds the central 3x3 vertices; every one of the 16 faces touches one.
    assert len(g.select_faces(box=region, test="any").faces) == 16
    assert len(g.select_edges(box=region, test="all").edges) == 12  # edges among those 9


def test_select_by_normal_and_sides():
    m = box(2)
    assert len(m.select_faces(normal=((0, 0, 1), 1)).faces) == 1
    assert len(m.select_faces(normal=((0, 0, 1), 90.0001)).faces) == 5
    s = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    assert len(s.select_faces(sides=3).faces) == 32
    north = s.select_vertices(normal=((0, 0, 1), 1))
    assert len(north.vertices) == 1
    with pytest.raises(ValidationError):
        s.select_vertices(sides=3)


def test_select_boundary():
    g = grid(2, 2, 2)
    assert len(g.select_edges(boundary=True).edges) == 8
    assert len(g.select_vertices(boundary=False).vertices) == 1
    assert len(g.select_faces(boundary=True).faces) == 4
    assert box().select_edges(boundary=True).is_empty


def test_nearest_is_unique_or_an_error():
    m = box(2)
    (v,) = m.select_vertices(nearest=(0.9, 0.9, 0.9)).vertices
    assert m.position(v) == (1.0, 1.0, 1.0)
    with pytest.raises(EditError) as info:
        m.select_faces(nearest=(0, 0, 0))  # all 6 face centers are 1 away
    assert info.value.code == "AMBIGUOUS_SELECTION" and len(info.value.elements["faces"]) == 6
    with pytest.raises(EditError, match="AMBIGUOUS_SELECTION"):
        m.select_edges(nearest=(1, 1, 1))  # three edges share that corner


def test_expect_guards_against_wrong_counts():
    m = box()
    assert len(m.select_faces(normal=((1, 0, 0), 1), expect=1).faces) == 1
    with pytest.raises(EditError) as info:
        m.select_faces(normal=((1, 0, 0), 100), expect=1)
    assert info.value.code == "SELECTION_COUNT_MISMATCH"


def test_predicate_and_combination():
    m = box(2)
    high = m.select_vertices(predicate=lambda vid, p: p[2] > 0)
    east = m.select_vertices(predicate=lambda vid, p: p[0] > 0)
    assert len(high & east) == 2 and len(high | east) == 6 and len(high - east) == 2
    assert (high | Selection.of(faces=[0])).faces == {0}


def test_world_space_selection_uses_object_transform():
    scene = Scene()
    obj = scene.add_cube("C", size=2, position=(10, 0, 0))
    obj.rotation_euler = (0, 0, 90)
    with obj.edit_mesh() as edit:
        # Local +X faces point along world +Y after the rotation.
        local = edit.select_faces(normal=((1, 0, 0), 1), expect=1)
        world = edit.select_faces(normal=((0, 1, 0), 1), space="world", expect=1)
        assert local == world
        near = edit.select_vertices(nearest=(11, 1, 1), space="world")
        assert len(near.vertices) == 1
        assert edit.select_faces(box=((9, -2, -2), (11, 2, 2)), space="world").faces == set(
            edit.face_ids()
        )
        edit.discard()


def test_world_space_normals_under_nonuniform_and_mirrored_scale():
    scene = Scene()
    obj = scene.add_object("Skewed", scale=(-1, 3, 1))
    with obj.edit_mesh() as edit:
        edit.add_box(2)
        # A mirrored X axis: the local +X face points to world -X.
        (f,) = edit.select_faces(normal=((-1, 0, 0), 1), space="world", expect=1).faces
        assert edit.face_normal(f) == (1.0, 0.0, 0.0)
        edit.discard()


def test_conversions():
    m = box(2)
    top = m.select_faces(normal=((0, 0, 1), 1))
    assert len(m.to_vertices(top).vertices) == 4
    assert len(m.to_edges(top).edges) == 4
    assert m.to_faces(m.to_vertices(top)) == top
    assert len(m.to_faces(m.to_edges(top)).faces) == 1
    assert len(m.grow(top).faces) == 5
    assert len(m.linked_faces(top).faces) == 6
    assert len(m.region_boundary(top).edges) == 4
    assert len(m.region_boundary(m.select_all()).edges) == 0


def test_edge_loops_on_sphere_and_boundaries():
    s = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    (equator_edge,) = s.select_edges(nearest=(np.cos(np.pi / 16), np.sin(np.pi / 16), 0)).edges
    loop = s.edge_loop(equator_edge)
    assert len(loop.edges) == 16  # a closed ring around the equator
    assert all(abs(s.edge_midpoint(e)[2]) < 1e-6 for e in loop.edges)
    ring = s.edge_ring(equator_edge)
    assert len(ring.edges) >= 6
    g = grid(3, 3, 3)
    boundary = next(iter(g.boundary_edges().edges))
    assert len(g.edge_loop(boundary).edges) == 12  # follows the whole open boundary


def test_kind_and_mode_validation():
    m = box()
    for bad in (
        dict(kind="corners"),
        dict(kind="faces", test="some"),
        dict(kind="faces", space="x"),
    ):
        kind = bad.pop("kind")
        with pytest.raises(ValidationError):
            m.select(kind, **bad)
