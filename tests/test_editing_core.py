"""EditableMesh storage, conversion, ids, transactions and history."""

import numpy as np
import pytest
from editing_helpers import assert_healthy, box, corner_map, snapshot

from aether3d import (
    EditError,
    Mesh,
    ValidationError,
    create_cube,
    create_plane,
    create_sphere,
    inspect_mesh,
)
from aether3d.editing import EditableMesh, Selection


def corners_of(mesh: Mesh) -> list[tuple]:
    rows = []
    for tri in mesh.triangles:
        for i in tri:
            parts = [mesh.positions[i]]
            if mesh.normals is not None:
                parts.append(mesh.normals[i])
            if mesh.uvs is not None:
                parts.append(mesh.uvs[i])
            rows.append(tuple(np.round(np.concatenate(parts), 6)))
    return sorted(rows)


@pytest.mark.parametrize("source", [create_cube(2), create_sphere(1, 12, 6), create_plane(3)])
def test_from_mesh_round_trip_preserves_every_corner(source):
    # One face per triangle: the very same triangles come back.
    em = EditableMesh.from_mesh(source, quads=False)
    assert_healthy(em)
    assert corners_of(em.to_mesh()) == corners_of(source)
    # Rebuilt quads may be re-triangulated along the other diagonal: the distinct
    # corners, the area and the volume are unchanged.
    em = EditableMesh.from_mesh(source)
    assert_healthy(em)
    out = em.to_mesh()
    assert sorted(set(corners_of(out))) == sorted(set(corners_of(source)))
    before, after = inspect_mesh(source), inspect_mesh(out)
    assert after.surface_area == pytest.approx(before.surface_area, rel=1e-6)
    assert after.topology["volume"] == pytest.approx(before.topology["volume"], rel=1e-6, abs=1e-9)
    assert after.bounds == before.bounds


def test_cube_welds_into_closed_quad_box():
    em = EditableMesh.from_mesh(create_cube(2))
    assert (em.vertex_count, em.edge_count, em.face_count) == (8, 12, 6)
    assert em.summary()["face_sizes"] == {"4": 6}
    assert em.validate().summary["closed"]
    raw = EditableMesh.from_mesh(create_cube(2), quads=False)
    assert raw.face_count == 12 and raw.summary()["face_sizes"] == {"3": 12}
    split = EditableMesh.from_mesh(create_cube(2), weld="none", quads=False)
    assert split.vertex_count == 24 and split.validate().summary["boundary_edge_count"] == 24


def test_sphere_converts_to_quads_and_pole_triangles():
    em = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    assert em.summary()["face_sizes"] == {"3": 32, "4": 96}
    assert em.validate().status == "ok"


def test_ids_are_deterministic_across_conversions():
    a, b = EditableMesh.from_mesh(create_sphere()), EditableMesh.from_mesh(create_sphere())
    assert snapshot(a) == snapshot(b)


def test_degenerate_triangles_are_dropped_and_reported():
    mesh = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)], [(0, 1, 2), (0, 1, 3)])
    em = EditableMesh.from_mesh(mesh)
    assert em.face_count == 1
    (diag,) = em.source_diagnostics
    assert diag.code == "DEGENERATE_TRIANGLES_DROPPED" and diag.samples == (1,)


def test_shading_is_inferred():
    cube = EditableMesh.from_mesh(create_cube())
    assert not any(cube.is_smooth(f) for f in cube.face_ids())
    assert all(cube.is_sharp(e) for e in cube.edge_ids())
    sphere = EditableMesh.from_mesh(create_sphere())
    assert all(sphere.is_smooth(f) for f in sphere.face_ids())
    assert not any(sphere.is_sharp(e) for e in sphere.edge_ids())


def test_queries():
    m = box(2)
    f = m.select_faces(normal=((0, 0, 1), 1), expect=1).faces
    (top,) = f
    assert len(m.face_vertices(top)) == 4 and len(m.face_edges(top)) == 4
    assert m.face_normal(top) == (0.0, 0.0, 1.0)
    assert m.face_center(top) == (0.0, 0.0, 1.0)
    assert m.face_area(top) == pytest.approx(4.0)
    e = m.face_edges(top)[0]
    assert m.edge_length(e) == pytest.approx(2.0)
    assert len(m.edge_faces(e)) == 2
    a, b = m.edge_vertices(e)
    assert m.edge_between(b, a) == e and a < b
    assert len(m.vertex_edges(a)) == 3 and len(m.vertex_faces(a)) == 3
    assert m.bounds() == ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    listing = m.describe(limit=2)
    assert len(listing["vertices"]) == 2 and set(listing["faces"][0]) >= {
        "id",
        "vertices",
        "normal",
    }


def test_invalid_ids_are_rejected_with_details():
    m = box()
    with pytest.raises(EditError) as info:
        m.face_vertices(99)
    assert info.value.code == "INVALID_ELEMENT" and info.value.elements == {"faces": [99]}
    with pytest.raises(EditError, match="EMPTY_SELECTION"):
        m.extrude_faces([], 1.0)
    with pytest.raises(ValidationError):
        m.extrude_faces(["a"], 1.0)
    with pytest.raises(EditError, match="WRONG_ELEMENT_KIND"):
        m.extrude_faces(Selection.of(vertices=[0]), 1.0)


def test_untouched_ids_are_stable_and_results_are_complete():
    m = box()
    before_faces = {f: m.face_vertices(f) for f in m.face_ids()}
    top = m.select_faces(normal=((0, 0, 1), 1)).faces
    result = m.extrude_faces(top, 1.0)
    (t,) = top
    assert m.has_face(t)  # the extruded face keeps its id
    for f, loop in before_faces.items():
        if f != t:
            assert m.face_vertices(f) == loop  # untouched faces unchanged
    assert result.created["faces"] == sorted(result.data["side_faces"].faces)
    assert len(result.created["vertices"]) == 4 and result.removed == {
        "vertices": [],
        "edges": [],
        "faces": [],
    }
    assert result.modified_faces == [t]
    assert set(result.created["edges"]) == set(m.edge_ids()) - set(range(12)) | (
        set(result.created["edges"]) & set(range(12))
    )
    assert len(result.created["edges"]) == 8
    assert all(m.has_edge(e) for e in result.created["edges"])


def test_failed_operation_leaves_mesh_bit_identical():
    m = box()
    m.extrude_faces(m.select_faces(normal=((0, 0, 1), 1)), 1.0)
    before = snapshot(m)
    top = m.select_faces(normal=((0, 0, 1), 1)).faces
    with pytest.raises(EditError, match="INSET_TOO_LARGE"):
        m.inset_faces(top, 5.0)
    assert snapshot(m) == before
    with pytest.raises(EditError):
        m.bevel_edges(m.select_all(), 50.0)
    assert snapshot(m) == before
    assert [h["operation"] for h in m.history] == ["add_box", "extrude_faces"]


def test_unexpected_exceptions_also_roll_back(monkeypatch):
    m = box()
    before = snapshot(m)

    def explode(*_a, **_k):
        raise RuntimeError("bug")

    monkeypatch.setattr(EditableMesh, "_require_non_degenerate", explode)
    with pytest.raises(RuntimeError):
        m.extrude_faces(m.select_faces(normal=((0, 0, 1), 1)), 1.0)
    assert snapshot(m) == before


def test_undo_redo_restore_exact_state_and_ids():
    m = box()
    states = [snapshot(m)]
    top = m.select_faces(normal=((0, 0, 1), 1)).faces
    r1 = m.extrude_faces(top, 1.0)
    states.append(snapshot(m))
    r2 = m.inset_faces(top, 0.2)
    states.append(snapshot(m))
    r3 = m.bevel_edges(m.select_edges(nearest=(1, 1, -0.5)), 0.1)
    states.append(snapshot(m))
    assert [h["operation"] for h in m.history] == [
        "add_box",
        "extrude_faces",
        "inset_faces",
        "bevel_edges",
    ]
    for i in (3, 2, 1):
        m.undo()
        assert snapshot(m) == states[i - 1]
    for i, original in ((1, r1), (2, r2), (3, r3)):
        redone = m.redo()
        assert snapshot(m) == states[i]
        assert redone.created == original.created  # same ids again
    with pytest.raises(EditError, match="NOTHING_TO_REDO"):
        m.redo()
    m.undo()
    m.translate(m.select_all(), (1, 0, 0))  # a new operation clears the redo stack
    assert not m.can_redo


def test_history_is_bounded(monkeypatch):
    m = box()
    monkeypatch.setattr(EditableMesh, "max_history", 3)
    for _ in range(5):
        m.translate(m.select_all(), (0.1, 0, 0))
    assert len(m.history) == 3


def test_copy_is_independent():
    m = box()
    c = m.copy()
    c.translate(c.select_all(), (1, 0, 0))
    assert m.position(0) != c.position(0)
    assert_healthy(c)


def test_to_mesh_dedupes_corners_and_omits_loose_vertices():
    m = box()
    m.create_vertex((5, 5, 5))
    mesh, diags = m.build_mesh()
    assert mesh.vertex_count == 24 and mesh.triangle_count == 12  # flat box: 4 corners x 6 faces
    assert [d.code for d in diags] == ["LOOSE_VERTICES_OMITTED"]
    assert inspect_mesh(mesh).topology["volume"] == pytest.approx(8.0)


def test_layers_without_uvs_or_normals():
    m = box(uvs=False, normals=False)
    mesh = m.to_mesh()
    assert mesh.uvs is None and mesh.normals is None and mesh.vertex_count == 8
    with_normals, _ = m.build_mesh(compute_normals=True)
    assert with_normals.normals is not None and with_normals.vertex_count == 24


def test_untouched_corners_are_preserved_exactly():
    sphere = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    before = corner_map(sphere, "normal")
    far = sphere.select_vertices(nearest=(0, 0, 1))  # the north pole
    sphere.translate(far, (0, 0, 0.1))
    after = corner_map(sphere, "normal")
    from aether3d.editing.normals import corner_normals

    recomputed = corner_normals(sphere)
    pole_faces = set(sphere.vertex_faces(next(iter(far.vertices))))
    near = {v for f in pole_faces for v in sphere.face_vertices(f)}
    for (f, v), n in before.items():
        k = sphere.face_vertices(f).index(v)
        if v not in near:
            assert recomputed[f][k] == n  # untouched corners keep the original normals
    assert after == before  # stored values are only replaced when building the mesh
