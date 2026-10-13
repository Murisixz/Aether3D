"""UVs and normals through editing: preserved, interpolated, recomputed, or reported."""

import numpy as np
import pytest
from editing_helpers import box, corner_map, face_with_normal, grid

from aether3d import create_sphere
from aether3d.editing import EditableMesh
from aether3d.editing.normals import corner_normals


def affine_ok(m: EditableMesh, f: int) -> bool:
    """True if the face's UVs are an affine function of its positions."""
    pts = np.array([m.position(v) for v in m.face_vertices(f)])
    uvs = np.array(m.corner_values(f, "uv"))
    a = np.c_[pts, np.ones(len(pts))]
    coef, *_ = np.linalg.lstsq(a, uvs, rcond=None)
    return bool(np.allclose(a @ coef, uvs, atol=1e-9))


def test_subdivide_interpolates_uvs_bilinearly():
    g = grid(1, 1, 2)
    (f,) = g.face_ids()
    g.subdivide_faces([f], 3)
    for face in g.face_ids():
        for v, uv in zip(g.face_vertices(face), g.corner_values(face, "uv"), strict=True):
            x, y, _ = g.position(v)
            assert uv == pytest.approx(((x + 1) / 2, (y + 1) / 2))


def test_loop_cut_and_split_edges_interpolate_uvs():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(1, 1, 0)).edges))
    m.loop_cut(edge, cuts=2)
    m.split_edges(m.select_edges(nearest=(0, -1, 1)), cuts=2)
    assert all(affine_ok(m, f) for f in m.face_ids())


def test_inset_and_bevel_keep_uvs_affine():
    m = box(2)
    top = face_with_normal(m, (0, 0, 1))
    r = m.inset_faces([top], 0.3)
    assert all(affine_ok(m, f) for f in [top, *r.data["side_faces"].faces])
    n = box(2)
    n.bevel_edges(n.select_edges(nearest=(1, 1, 0)), 0.2)
    assert all(affine_ok(n, f) for f in n.face_ids())


def test_extrusion_reports_degenerate_side_uvs():
    m = box(2)
    r = m.extrude_faces([face_with_normal(m, (0, 0, 1))], 1)
    (diag,) = [d for d in r.diagnostics if d.code == "UV_DEGENERATE_FACES"]
    assert diag.count == 4 and diag.severity == "warning"
    for f in r.data["side_faces"].faces:
        uvs = np.array(m.corner_values(f, "uv"))
        assert np.linalg.matrix_rank(uvs - uvs[0], tol=1e-12) <= 1  # zero UV area, as reported
    no_uv = box(2, uvs=False)
    r = no_uv.extrude_faces([face_with_normal(no_uv, (0, 0, 1))], 1)
    assert r.diagnostics == []


def test_flat_faces_get_exact_face_normals_after_edits():
    m = box(2)
    m.extrude_faces([face_with_normal(m, (0, 0, 1))], 1)
    m.bevel_edges(m.select_edges(nearest=(1, 1, 2)), 0.2)
    mesh = m.to_mesh()
    for tri in mesh.triangles:
        p = mesh.positions[tri].astype(float)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        assert np.allclose(mesh.normals[tri], n, atol=1e-6)


def test_smooth_normals_survive_shape_preserving_edits():
    s = EditableMesh.from_mesh(create_sphere(1, 16, 8))
    before = s.to_mesh()
    s.subdivide_faces(s.select_faces(sides=4), 1)
    after = s.to_mesh()
    # Original vertices keep their analytic normals; new ones are interpolated.
    for i, p in enumerate(after.positions):
        n = after.normals[i]
        assert np.linalg.norm(n) == pytest.approx(1, abs=1e-6)
        assert np.dot(n, p / np.linalg.norm(p)) > 0.98
    original = {tuple(p): tuple(n) for p, n in zip(before.positions, before.normals, strict=True)}
    for p, n in zip(after.positions, after.normals, strict=True):
        if tuple(p) in original:
            assert tuple(n) == original[tuple(p)]


def test_moved_smooth_region_is_recomputed_and_rest_preserved():
    s = EditableMesh.from_mesh(create_sphere(1, 24, 12))
    stored = corner_map(s, "normal")
    pole = s.select_vertices(nearest=(0, 0, 1))
    s.translate(pole, (0, 0, 0.5))
    fresh = corner_normals(s)
    changed = {
        (f, v)
        for f in s.face_ids()
        for k, v in enumerate(s.face_vertices(f))
        if fresh[f][k] != stored[(f, v)]
    }
    pole_faces = set(s.vertex_faces(next(iter(pole.vertices))))
    allowed = {
        (f, v)
        for f in s.face_ids()
        for v in s.face_vertices(f)
        if v in {u for g in pole_faces for u in s.face_vertices(g)}
    }
    assert changed and changed <= allowed  # only corners next to the edit changed
    north = fresh[next(iter(pole_faces))][
        s.face_vertices(next(iter(pole_faces))).index(next(iter(pole.vertices)))
    ]
    assert np.allclose(north, (0, 0, 1), atol=1e-9)  # symmetric fan: still straight up


def test_sharp_edges_split_smooth_normals():
    m = box(2, uvs=False)
    m.set_smooth(m.select_all())
    m.set_sharp(m.select_edges(ids=m.edge_ids()), True)
    mesh = m.to_mesh()
    assert mesh.vertex_count == 24  # every edge sharp: flat-looking even though smooth
    m.set_sharp(m.select_edges(ids=m.edge_ids()), False)
    assert m.to_mesh().vertex_count == 8


def test_merge_averages_merged_corner_uvs():
    g = grid(2, 1, (2, 1))
    a = next(iter(g.select_vertices(nearest=(0, -0.5, 0)).vertices))
    b = next(iter(g.select_vertices(nearest=(0, 0.5, 0)).vertices))
    g.merge_vertices([a, b])
    for f in g.face_ids():
        loop = g.face_vertices(f)
        if a in loop:
            uv = g.corner_values(f, "uv")[loop.index(a)]
            assert uv == pytest.approx((0.5, 0.5))  # mean of (0.5, 0) and (0.5, 1)


def test_custom_corner_layer_is_carried_and_interpolated():
    m = box(2)
    m._layers["color"] = type(m._layers["uv"])("color", 3)  # a future per-corner layer
    for f in m.face_ids():
        m._layers["color"].data[f] = [
            tuple(np.array(m.position(v)) / 2 + 0.5) for v in m.face_vertices(f)
        ]
    m.subdivide_faces(m.select_all(), 1)
    m.inset_faces(m.select_faces(normal=((0, 0, 1), 1)), 0.1)
    for f in m.face_ids():
        for v, c in zip(m.face_vertices(f), m.corner_values(f, "color"), strict=True):
            assert np.allclose(c, np.array(m.position(v)) / 2 + 0.5, atol=1e-9)
