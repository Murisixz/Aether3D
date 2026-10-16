"""Modeling operations, verified with independent geometric checks."""

import math

import numpy as np
import pytest
from editing_helpers import (
    assert_healthy,
    box,
    euler,
    face_with_normal,
    grid,
    outward_closed,
    snapshot,
    total_area,
    trimesh_volume,
    volume,
)

from aether3d import EditError, ValidationError, create_cube
from aether3d.editing import EditableMesh, Selection

UP, DOWN, EAST = (0, 0, 1), (0, 0, -1), (1, 0, 0)


# --------------------------------------------------------------------------- #
# Primitives and creation
# --------------------------------------------------------------------------- #


def test_add_box_and_grid():
    m = box(2)
    assert (m.vertex_count, m.edge_count, m.face_count) == (8, 12, 6)
    assert outward_closed(m) and volume(m) == pytest.approx(8)
    g = grid(3, 2, (3, 2))
    assert g.face_count == 6 and g.vertex_count == 12
    assert total_area(g) == pytest.approx(6)
    assert all(g.face_normal(f) == (0.0, 0.0, 1.0) for f in g.face_ids())
    uvs = {uv for f in g.face_ids() for uv in g.corner_values(f, "uv")}
    assert min(uvs) == (0.0, 0.0) and max(uvs) == (1.0, 1.0)


def test_box_with_size_vector_and_center():
    m = EditableMesh()
    m.add_box((2, 4, 6), center=(1, 1, 1))
    assert m.bounds() == ((0.0, -1.0, -2.0), (2.0, 3.0, 4.0))
    assert volume(m) == pytest.approx(48)


def test_create_face_and_vertex():
    m = EditableMesh()
    ids = [m.create_vertex(p).data["vertex"] for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]]
    r = m.create_face(ids)
    assert m.face_normal(r.data["face"]) == (0.0, 0.0, 1.0)
    # Loose vertices have no UVs to inherit: reported, not silently invented.
    assert [d.code for d in r.diagnostics] == ["UVS_UNDEFINED"]
    apex = m.create_vertex((0.5, 0.5, 1)).data["vertex"]
    side = m.create_face([ids[1], ids[0], apex])  # shares edge 0-1 in the opposite direction
    assert side.data["face"] in m.face_ids()
    with pytest.raises(EditError, match="INCONSISTENT_WINDING"):
        m.create_face([ids[1], ids[2], apex])  # the quad already runs 1 -> 2
    with pytest.raises(EditError, match="DUPLICATE_FACE"):
        m.create_face(list(reversed(ids)))
    with pytest.raises(EditError, match="INVALID_POLYGON"):
        m.create_face([ids[0], ids[0], ids[1]])
    with pytest.raises(EditError, match="DEGENERATE_RESULT"):
        m.create_face([ids[0], ids[2], ids[1], ids[3]])  # symmetric bowtie: zero signed area
    star = [
        m.create_vertex((10 + math.cos(a), math.sin(a), 0)).data["vertex"]
        for a in np.linspace(0, 2 * math.pi, 6)[:-1]
    ]
    with pytest.raises(EditError, match="SELF_INTERSECTING_FACE"):
        m.create_face([star[k] for k in (0, 2, 4, 1, 3)])  # pentagram
    collinear = [m.create_vertex((x, 5, 0)).data["vertex"] for x in (0, 1, 2)]
    with pytest.raises(EditError, match="DEGENERATE_RESULT"):
        m.create_face(collinear)
    assert_healthy(m)


def test_create_face_refuses_third_face_on_edge():
    m = box()
    (top,) = m.select_faces(normal=(UP, 1)).faces
    a, b = m.face_vertices(top)[:2]
    extra = m.create_vertex((0, 0, 3)).data["vertex"]
    with pytest.raises(EditError, match="NON_MANIFOLD_RESULT"):
        m.create_face([b, a, extra])
    r = m.create_face([b, a, extra], allow_non_manifold=True)
    assert len(m.edge_faces(m.edge_between(a, b))) == 3
    assert "NON_MANIFOLD_EDGES" in m.validate().codes()
    assert r.data["face"] in m.face_ids()


def test_create_face_reports_undefined_uvs():
    m = box()
    (top,) = m.select_faces(normal=(UP, 1)).faces
    loop = m.face_vertices(top)
    m.delete_faces([top], keep_loose_vertices=True)
    r = m.create_face(loop)
    # Each corner vertex already has different UVs in its three faces -> undefined.
    assert [d.code for d in r.diagnostics] == ["UVS_UNDEFINED"]
    assert outward_closed(m)


# --------------------------------------------------------------------------- #
# Extrusion and inset
# --------------------------------------------------------------------------- #


def test_extrude_region_volume_topology_and_ids():
    m = box(2)
    top = face_with_normal(m, UP)
    r = m.extrude_faces([top], 1.5)
    assert outward_closed(m) and volume(m) == pytest.approx(4 * 3.5)
    assert euler(m) == 2
    assert m.face_center(top) == pytest.approx((0, 0, 2.5))
    assert len(r.data["side_faces"].faces) == 4
    assert all(m.face_normal(f)[2] == pytest.approx(0) for f in r.data["side_faces"].faces)
    assert len(r.data["vertices"]) == 4
    assert set(r.data["vertices"].values()) == set(m.face_vertices(top))
    assert trimesh_volume(m) == pytest.approx(14)


def test_extrude_inward_and_by_offset():
    m = box(2)
    m.extrude_faces([face_with_normal(m, UP)], -0.5)
    assert volume(m) == pytest.approx(8 - 2)  # the cap moved 0.5 down: height 1.5
    n = box(2)
    n.extrude_faces([face_with_normal(n, UP)], offset=(1, 0, 1))  # sheared extrusion
    assert volume(n) == pytest.approx(8 + 4)  # parallelepiped: base 4 x height 1
    assert outward_closed(n)


def test_extrude_connected_region_keeps_inner_edges():
    g = grid(4, 4, 4)
    inner = g.select_faces(box=((-1, -1, -1), (1, 1, 1)))
    assert len(inner.faces) == 4
    r = g.extrude_faces(inner, 1.0)
    assert len(r.data["side_faces"].faces) == 8  # outline of a 2x2 block
    assert len(r.data["vertices"]) == 8  # interior vertex is not duplicated
    assert_healthy(g)
    s = g.validate().summary
    assert s["boundary_edge_count"] == 16 and s["consistently_oriented"]


def test_extrude_disconnected_regions_and_individual_mode():
    m = box(2)
    sides = m.select_faces(normal=(EAST, 1)) | m.select_faces(normal=((-1, 0, 0), 1))
    m.extrude_faces(sides, 1.0)  # two separate regions, each along its own normal
    assert volume(m) == pytest.approx(16) and outward_closed(m)
    n = box(2)
    r = n.extrude_faces(n.select_faces(boundary=False), 0.5, mode="individual")
    assert len(r.data["side_faces"].faces) == 24
    assert volume(n) == pytest.approx(8 + 6 * 4 * 0.5)


def test_extrude_open_surface_creates_walls():
    g = grid(1, 1, 2)
    r = g.extrude_faces(g.select_all(), 1.0)
    assert len(r.data["side_faces"].faces) == 4
    s = g.validate().summary
    assert s["boundary_edge_count"] == 4 and s["face_count"] == 5


def test_extrude_errors():
    m = box()
    before = snapshot(m)
    with pytest.raises(EditError, match="ZERO_EXTRUSION"):
        m.extrude_faces([0], 0.0)
    with pytest.raises(ValidationError):
        m.extrude_faces([0])
    with pytest.raises(EditError, match="DEGENERATE_RESULT"):
        top = face_with_normal(m, UP)
        m.extrude_faces([top], offset=(1, 0, 0))  # sideways: walls along x have zero area
    assert snapshot(m) == before


def test_extrude_closed_shell_is_a_move():
    m = box(2)
    r = m.extrude_faces(m.select_all(), offset=(0, 0, 1))
    assert [d.code for d in r.diagnostics] == ["EXTRUDE_WITHOUT_BOUNDARY"]
    assert m.bounds() == ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))


def test_extrude_inconsistent_region_is_rejected():
    m = box()
    top, east = face_with_normal(m, UP), face_with_normal(m, EAST)
    m.flip_faces([east])
    with pytest.raises(EditError, match="INCONSISTENT_WINDING"):
        m.extrude_faces([top, east], 1.0)


def test_inset_individual_geometry_and_uvs():
    m = box(2)
    top = face_with_normal(m, UP)
    uv_before = dict(zip(m.face_vertices(top), m.corner_values(top, "uv"), strict=True))
    r = m.inset_faces([top], 0.25)
    assert volume(m) == pytest.approx(8) and outward_closed(m)
    assert m.face_area(top) == pytest.approx(1.5**2)
    pts = np.array([m.position(v) for v in m.face_vertices(top)])
    assert np.allclose(np.abs(pts[:, :2]), 0.75) and np.allclose(pts[:, 2], 1)
    sides = r.data["side_faces"].faces
    assert len(sides) == 4 and all(
        m.face_area(f) == pytest.approx((2 + 1.5) / 2 * 0.25) for f in sides
    )
    # UVs of the inset face follow the same affine map as the positions: (x, y) -> uv.
    for v, uv in zip(m.face_vertices(top), m.corner_values(top, "uv"), strict=True):
        x, y, _ = m.position(v)
        assert uv == pytest.approx(((x + 1) / 2, (y + 1) / 2))
    assert set(uv_before.values()) == {(0, 0), (1, 0), (1, 1), (0, 1)}


def test_inset_with_depth_and_mitred_corners():
    m = EditableMesh()
    a, b, c = (m.create_vertex(p).data["vertex"] for p in [(0, 0, 0), (4, 0, 0), (0, 3, 0)])
    f = m.create_face([a, b, c]).data["face"]
    m.inset_faces([f], 0.5)
    # The inset triangle's edges are all exactly 0.5 inside the original ones.
    pts = [np.array(m.position(v)) for v in m.face_vertices(f)]
    assert min(p[1] for p in pts) == pytest.approx(0.5)  # edge along y=0
    assert min(p[0] for p in pts) == pytest.approx(0.5)  # edge along x=0
    hyp = [abs(3 * p[0] + 4 * p[1] - 12) / 5 for p in pts]  # distance to the hypotenuse
    assert min(hyp) == pytest.approx(0.5)
    n = box(2)
    top = face_with_normal(n, UP)
    n.inset_faces([top], 0.5, depth=-0.5)
    assert volume(n) == pytest.approx(8 - 0.5 / 3 * (4 + 1 + 2))  # square frustum


def test_inset_region_insets_only_the_outline():
    g = grid(2, 2, 2)
    r = g.inset_faces(g.select_all(), 0.1, mode="region")
    assert len(r.data["side_faces"].faces) == 8
    assert g.bounds() == ((-1.0, -1.0, 0.0), (1.0, 1.0, 0.0))
    inner = r.data["faces"].faces
    pts = np.array([g.position(v) for f in inner for v in g.face_vertices(f)])
    assert np.max(np.abs(pts[:, :2])) == pytest.approx(0.9)
    assert total_area(g) == pytest.approx(4)


def test_inset_too_large_and_bad_regions():
    m = box(2)
    before = snapshot(m)
    for t in (1.0, 5.0):
        with pytest.raises(EditError, match="INSET_TOO_LARGE"):
            m.inset_faces([face_with_normal(m, UP)], t)
        assert snapshot(m) == before
    with pytest.raises(EditError, match="UNSUPPORTED_TOPOLOGY"):
        m.inset_faces(m.select_all(), 0.1, mode="region")  # closed shell has no outline
    with pytest.raises(ValidationError):
        m.inset_faces([0], 0)


# --------------------------------------------------------------------------- #
# Bevel
# --------------------------------------------------------------------------- #


def chamfered_cube_volume(size: float, w: float) -> float:
    """Exact volume of a cube with all edges chamfered by w (each corner loses 5/6 w^3)."""
    return size**3 - 12 * (w * w / 2) * (size - 2 * w) - 8 * (5 / 6) * w**3


def test_bevel_single_edge():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(1, 1, 0)).edges))
    r = m.bevel_edges([edge], 0.25)
    assert outward_closed(m) and volume(m) == pytest.approx(8 - 0.25**2 / 2 * 2)
    (chamfer,) = r.data["faces"].faces
    n = np.array(m.face_normal(chamfer))
    assert np.allclose(n, [math.sqrt(0.5), math.sqrt(0.5), 0])
    assert m.face_area(chamfer) == pytest.approx(2 * 0.25 * math.sqrt(2))
    assert euler(m) == 2 and not m.has_edge(edge)
    assert m.summary()["face_sizes"] == {"4": 5, "5+": 2}  # the two end faces become pentagons


@pytest.mark.parametrize("w", [0.1, 0.3, 0.49])
def test_bevel_all_edges_matches_analytic_volume(w):
    m = box(2)
    r = m.bevel_edges(m.select_all(), w)
    assert len(r.data["faces"].faces) == 12 and len(r.data["corner_faces"].faces) == 8
    assert outward_closed(m) and euler(m) == 2
    assert volume(m) == pytest.approx(chamfered_cube_volume(2, w), rel=1e-6)
    assert trimesh_volume(m) == pytest.approx(chamfered_cube_volume(2, w), rel=1e-6)


def test_bevel_edges_meeting_at_corner():
    m = box(2)
    corner = np.array([1, 1, 1])
    edges = [
        e
        for e in m.edge_ids()
        if any(np.allclose(m.position(v), corner) for v in m.edge_vertices(e))
    ]
    assert len(edges) == 3
    r = m.bevel_edges(edges[:2], 0.2)  # two of the three edges at a corner
    assert outward_closed(m) and len(r.data["corner_faces"].faces) == 0
    m2 = box(2)
    r2 = m2.bevel_edges(edges, 0.2)
    assert len(r2.data["corner_faces"].faces) == 1 and outward_closed(m2)


def test_bevel_on_subdivided_surface_merges_corner_into_chamfer():
    m = box(2)
    m.subdivide_faces(m.select_all(), 1)  # valence-4 vertices in the middle of faces
    e = next(iter(m.select_edges(nearest=(0.5, 0, 1)).edges))
    m.bevel_edges([e], 0.1)
    assert outward_closed(m) and volume(m) == pytest.approx(8)  # flat region: volume unchanged
    assert m.validate().status == "ok"


def test_bevel_errors_leave_mesh_untouched():
    g = grid(2, 2, 2)
    before = snapshot(g)
    boundary = next(iter(g.boundary_edges().edges))
    with pytest.raises(EditError, match="NOT_MANIFOLD"):
        g.bevel_edges([boundary], 0.1)
    assert snapshot(g) == before
    # Since 0.13 an end vertex on an open boundary is supported if its faces form one
    # fan; two fans meeting at a vertex (a bowtie) are still refused.
    b = EditableMesh()
    ids = [
        b.create_vertex(p).data["vertex"]
        for p in [(0, 0, 0), (1, -1, 0), (1, 0, 0), (1, 1, 0), (-1, 1, 0), (-1, 0, 0), (-1, -1, 0)]
    ]
    v, a1, a2, a3, b1, b2, b3 = ids
    for loop in ([v, a1, a2], [v, a2, a3], [v, b1, b2], [v, b2, b3]):
        b.create_face(loop)
    before = snapshot(b)
    with pytest.raises(EditError, match="BEVEL_UNSUPPORTED_VERTEX"):
        b.bevel_edges([b.edge_between(v, a2)], 0.1)
    assert snapshot(b) == before
    m = box(2)
    before = snapshot(m)
    for w in (1.0, 1.5, 5.0):  # faces are 2 wide: they collapse at w = 1
        with pytest.raises(EditError, match="BEVEL_WIDTH_TOO_LARGE"):
            m.bevel_edges(m.select_all(), w)
    with pytest.raises(ValidationError):  # (segments > 1 is supported since 0.13)
        m.bevel_edges([0], 0.1, segments=0)
    assert snapshot(m) == before


def test_bevel_interpolates_uvs_within_faces():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(1, 1, 0)).edges))
    r = m.bevel_edges([edge], 0.5)
    (chamfer,) = r.data["faces"].faces
    for f in m.face_ids():
        if f == chamfer:
            continue
        # The original faces keep an affine position -> UV map after the bevel.
        pts = np.array([m.position(v) for v in m.face_vertices(f)])
        uvs = np.array(m.corner_values(f, "uv"))
        A = np.c_[pts, np.ones(len(pts))]
        coef, res, *_ = np.linalg.lstsq(A, uvs, rcond=None)
        assert np.allclose(A @ coef, uvs, atol=1e-9)


# --------------------------------------------------------------------------- #
# Subdivision and loop cuts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cuts", [1, 2, 3])
def test_subdivide_box(cuts):
    m = box(2)
    r = m.subdivide_faces(m.select_all(), cuts)
    k = cuts + 1
    assert m.face_count == 6 * k * k and euler(m) == 2
    assert volume(m) == pytest.approx(8) and total_area(m) == pytest.approx(24)
    assert outward_closed(m)
    assert all(len(pieces) == k * k for pieces in r.data["faces"].values())


def test_subdivide_triangles_and_ngons_and_neighbours():
    m = EditableMesh()
    ids = [m.create_vertex(p).data["vertex"] for p in [(0, 0, 0), (2, 0, 0), (0, 2, 0)]]
    tri = m.create_face(ids).data["face"]
    m.subdivide_faces([tri], 2)
    assert m.face_count == 9 and total_area(m) == pytest.approx(2)
    pent = EditableMesh()
    pts = [(math.cos(a), math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 6)[:-1]]
    f = pent.create_face([pent.create_vertex(p).data["vertex"] for p in pts]).data["face"]
    area = pent.face_area(f)
    pent.subdivide_faces([f])
    assert pent.face_count == 5 and pent.summary()["face_sizes"] == {"4": 5}
    assert total_area(pent) == pytest.approx(area)
    g = EditableMesh()
    f5 = g.create_face([g.create_vertex(p).data["vertex"] for p in pts]).data["face"]
    with pytest.raises(EditError, match="UNSUPPORTED_TOPOLOGY"):
        g.subdivide_faces([f5], 2)
    m = box(2)
    m.subdivide_faces([face_with_normal(m, UP)], 1)  # neighbours gain the edge midpoints
    assert outward_closed(m) and m.summary()["face_sizes"] == {"4": 5, "5+": 4}


def test_loop_cut_closed_ring_on_box():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(1, 1, 0)).edges))  # vertical edge
    r = m.loop_cut(edge, cuts=3)
    assert r.data["closed"] and len(r.data["edges"].edges) == 12
    assert m.face_count == 6 - 4 + 16 and outward_closed(m) and volume(m) == pytest.approx(8)
    heights = sorted({round(m.position(v)[2], 9) for v in r.data["vertices"].vertices})
    assert heights == pytest.approx([-0.5, 0.0, 0.5])
    loops = [m.edge_loop(e) for e in r.data["edges"].edges]
    assert all(len(lp.edges) == 4 for lp in loops)


def test_loop_cut_factor_is_measured_from_lower_vertex():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(1, 1, 0)).edges))
    a, _ = m.edge_vertices(edge)
    r = m.loop_cut(edge, factor=0.25)
    z0 = m.position(a)[2]
    new_z = {round(m.position(v)[2], 9) for v in r.data["vertices"].vertices}
    assert new_z == {round(z0 + 0.25 * (2 if z0 < 0 else -2), 9)}


def test_loop_cut_open_ring_on_grid_and_errors():
    g = grid(3, 2, (3, 2))
    edge = next(iter(g.select_edges(nearest=(-1.5, -0.5, 0)).edges))  # vertical boundary edge
    r = g.loop_cut(edge)
    assert not r.data["closed"] and g.face_count == 9  # the 3 quads of the row are halved
    assert total_area(g) == pytest.approx(6)
    tri = EditableMesh.from_mesh(create_cube(), quads=False)
    with pytest.raises(EditError, match="NO_QUAD_RING"):
        tri.loop_cut(tri.edge_ids()[0])
    with pytest.raises(ValidationError):
        g.loop_cut(g.edge_ids()[0], cuts=2, factor=0.3)


def test_edge_loop_and_ring_selection():
    g = grid(4, 3, 4)
    e = next(iter(g.select_edges(nearest=(0, 0, 0)).edges))  # the central vertical edge
    ring = g.edge_ring(e)
    assert len(ring.edges) == 5
    assert all(abs(g.edge_midpoint(x)[1]) < 1e-9 for x in ring.edges)
    loop = g.edge_loop(e)
    xs = {round(g.edge_midpoint(x)[0], 9) for x in loop.edges}
    assert len(xs) == 1  # a straight column of edges
    assert len(loop.edges) == 3
    m = box(2)
    top_edge = next(iter(m.select_edges(nearest=(0, 1, 1)).edges))
    assert len(m.edge_loop(top_edge).edges) == 1  # box corners are poles (valence 3)
    assert len(m.edge_ring(top_edge).edges) == 4


# --------------------------------------------------------------------------- #
# Merging, collapsing, dissolving, joining
# --------------------------------------------------------------------------- #


def test_merge_vertices_to_pyramid():
    m = box(2)
    top = face_with_normal(m, UP)
    r = m.merge_vertices(list(m.face_vertices(top)), at="center")
    assert top in r.removed["faces"] and m.face_count == 5
    # Base 2x2 at z = -1, apex at z = 1: a pyramid of volume 4 * 2 / 3.
    assert outward_closed(m) and volume(m) == pytest.approx(8 / 3)
    apex = r.data["vertex"]
    assert m.position(apex) == (0.0, 0.0, 1.0)
    assert m.summary()["face_sizes"] == {"3": 4, "4": 1}


def test_merge_keep_and_point_and_errors():
    m = box(2)
    top = face_with_normal(m, UP)
    loop = m.face_vertices(top)
    keep = loop[2]
    kept_position = m.position(keep)
    r = m.merge_vertices([loop[1], loop[2]], at="keep", keep=keep)
    assert r.data["vertex"] == keep and m.position(keep) == kept_position
    assert not m.has_vertex(loop[1]) and outward_closed(m)
    n = box(2)
    top = face_with_normal(n, UP)
    loop = n.face_vertices(top)
    before = snapshot(n)
    with pytest.raises(EditError, match="MERGE_INVALID_FACE"):
        n.merge_vertices([loop[0], loop[2]])  # opposite corners of a quad
    assert snapshot(n) == before
    with pytest.raises(EditError, match="TOO_FEW_ELEMENTS"):
        n.merge_vertices([loop[0]])
    n.merge_vertices([loop[0], loop[1]], at=(5, 5, 5))
    assert any(n.position(v) == (5.0, 5.0, 5.0) for v in n.vertex_ids())


def test_merge_by_distance_welds_split_mesh():
    m = EditableMesh.from_mesh(create_cube(2), weld="none", quads=False)
    assert m.vertex_count == 24 and not m.validate().summary["closed"]
    r = m.merge_by_distance(1e-6)
    assert m.vertex_count == 8 and r.data["merged_count"] == 16
    assert m.validate().summary["closed"] and outward_closed(m)
    n = box(2)
    near = n.create_vertex((1.0005, 1, 1)).data["vertex"]
    n.merge_by_distance(0.001)
    assert not n.has_vertex(near) and n.vertex_count == 8


def test_collapse_edges():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(0, 1, 1)).edges))
    r = m.collapse_edges([edge])
    assert m.vertex_count == 7 and m.position(r.data["vertices"][0]) == (0.0, 1.0, 1.0)
    assert euler(m) == 2 and m.validate().summary["closed"]
    assert 0 < volume(m) < 8 and volume(m) == pytest.approx(trimesh_volume(m))


def test_dissolve_and_join_triangles():
    m = EditableMesh.from_mesh(create_cube(2), quads=False)
    assert m.face_count == 12
    r = m.join_triangles()
    assert r.data["joined_count"] == 6 and m.summary()["face_sizes"] == {"4": 6}
    assert outward_closed(m)
    m2 = box(2)
    top, east = face_with_normal(m2, UP), face_with_normal(m2, EAST)
    shared = set(m2.face_edges(top)) & set(m2.face_edges(east))
    r = m2.dissolve_edges(shared)
    (merged,) = r.data["faces"].faces
    assert len(m2.face_vertices(merged)) == 6 and m2.face_count == 5
    assert "NONPLANAR_FACES" in m2.validate().codes()
    g = grid(1, 1)
    with pytest.raises(EditError, match="EDGE_NOT_INTERIOR"):
        g.dissolve_edges(g.edge_ids()[:1])


def test_join_triangles_respects_angle_and_seams():
    m = EditableMesh.from_mesh(create_cube(2), quads=False)
    assert m.join_triangles(max_angle=0.0).data["joined_count"] == 6
    folded = EditableMesh()
    ids = [
        folded.create_vertex(p).data["vertex"]
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (0, 1, 0)]
    ]
    folded.create_face([ids[0], ids[1], ids[2]])
    folded.create_face([ids[0], ids[2], ids[3]])
    assert folded.join_triangles(max_angle=1).data["joined_count"] == 0
    assert folded.join_triangles(max_angle=60).data["joined_count"] == 1


# --------------------------------------------------------------------------- #
# Splitting, separating, triangulating, deleting, flipping
# --------------------------------------------------------------------------- #


def test_split_edges_inserts_into_both_faces():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(0, 1, 1)).edges))
    faces = m.edge_faces(edge)
    r = m.split_edges([edge], cuts=3)
    new = r.data["vertices"][edge]
    assert len(new) == 3 and not m.has_edge(edge)
    assert all(len(m.face_vertices(f)) == 7 for f in faces)
    xs = [m.position(v)[0] for v in new]
    assert xs == pytest.approx(sorted(xs)) or xs == pytest.approx(sorted(xs, reverse=True))
    assert outward_closed(m) and volume(m) == pytest.approx(8)


def test_split_face_and_concave_chord():
    m = box(2)
    top = face_with_normal(m, UP)
    loop = m.face_vertices(top)
    r = m.split_face(top, loop[0], loop[2])
    assert m.face_count == 7 and m.has_face(r.data["face"]) and outward_closed(m)
    assert m.face_vertices(top) == loop[:3]  # the part from loop[0] to loop[2] keeps the id
    with pytest.raises(EditError, match="VERTICES_ADJACENT"):
        m.split_face(top, loop[0], loop[1])
    c = EditableMesh()  # an L-shaped (concave) hexagon
    pts = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)]
    ids = [c.create_vertex(p).data["vertex"] for p in pts]
    f = c.create_face(ids).data["face"]
    with pytest.raises(EditError, match="CHORD_OUTSIDE_FACE"):
        c.split_face(f, ids[2], ids[4])
    c.split_face(f, ids[0], ids[3])
    assert total_area(c) == pytest.approx(3)


def test_triangulate_concave_polygon_preserves_area_and_orientation():
    c = EditableMesh()
    pts = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (2, 1, 0), (0, 4, 0)]  # an arrow / dart shape
    f = c.create_face([c.create_vertex(p).data["vertex"] for p in pts]).data["face"]
    area = c.face_area(f)
    r = c.triangulate_faces([f])
    assert len(r.data["faces"][f]) == 3
    assert total_area(c) == pytest.approx(area)
    assert all(c.face_normal(t) == pytest.approx((0, 0, 1)) for t in c.face_ids())


def test_rip_and_separate():
    m = box(2)
    corner = next(iter(m.select_vertices(nearest=(1, 1, 1)).vertices))
    r = m.rip_vertices([corner])
    assert len(r.data["vertices"][corner]) == 2 and m.vertex_count == 10
    assert m.validate().summary["boundary_edge_count"] == 6
    n = box(2)
    top = face_with_normal(n, UP)
    r = n.separate_faces([top])
    assert len(r.data["vertices"]) == 4 and len(n.linked_faces([top]).faces) == 1


def test_delete_and_flip():
    m = box(2)
    top = face_with_normal(m, UP)
    m.delete_faces([top])
    assert m.validate().summary["boundary_edge_count"] == 4 and m.vertex_count == 8
    m.delete_vertices(m.select_vertices(nearest=(1, 1, -1)))
    # Its three faces go; the west and south faces remain, using 6 vertices.
    assert m.face_count == 2 and m.vertex_count == 6
    n = box(2)
    n.flip_faces(n.select_all())
    assert volume(n) == pytest.approx(-8)
    assert all(
        np.dot(n.face_normal(f), n.corner_values(f, "normal")[0]) > 0.99 for f in n.face_ids()
    )
    n.delete_edges(n.edge_ids()[:1])
    assert n.face_count == 4


def test_split_edge_single_and_validation():
    m = box(2)
    edge = next(iter(m.select_edges(nearest=(0, 1, 1)).edges))
    a, b = m.edge_vertices(edge)
    v = m.split_edge(edge, 0.25, from_vertex=b).data["vertex"]
    expected = np.array(m.position(b)) + 0.25 * (np.array(m.position(a)) - np.array(m.position(b)))
    assert np.allclose(m.position(v), expected)
    with pytest.raises(ValidationError):
        m.split_edge(m.edge_ids()[0], 1.0)


def test_set_smooth_and_sharp_change_normals():
    m = box(2, uvs=False)
    m.set_smooth(m.select_all(), True)
    m.set_sharp(Selection.of(edges=m.edge_ids()), False)
    mesh = m.to_mesh()
    assert mesh.vertex_count == 8  # fully smooth and no UV seams: one normal per vertex
    n = mesh.normals[0]
    assert np.allclose(np.abs(n), 1 / math.sqrt(3), atol=1e-6)
