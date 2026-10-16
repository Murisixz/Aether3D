"""Stage 13 modeling operations, checked with independent geometry: exact volumes and
positions where they are known, topology invariants everywhere, untouched meshes after
every refusal, and exact undo/redo."""

import math

import numpy as np
import pytest
from editing_helpers import box, euler, grid, outward_closed, snapshot

from aether3d import EditError, Mesh, Scene, ValidationError, create_cube, create_sphere
from aether3d.editing import EditableMesh
from aether3d.editing.script import run_edit_script


def volume(m: EditableMesh) -> float:
    total = 0.0
    for f in m.face_ids():
        p = np.array([m.position(v) for v in m.face_vertices(f)])
        for i in range(1, len(p) - 1):
            total += float(np.dot(p[0], np.cross(p[i], p[i + 1])))
    return total / 6.0


def sphere_mesh(segments=32, rings=16) -> EditableMesh:
    return EditableMesh.from_mesh(create_sphere(1.0, segments, rings))


def healthy(m: EditableMesh) -> None:
    assert m._consistency_errors() == []
    assert m.validate().ok


# --------------------------------------------------------------------------- #
# Bisect
# --------------------------------------------------------------------------- #


def test_bisect_cuts_every_face_and_clears_one_side():
    m = box(2)
    r = m.bisect((0, 0, 0.3), (0, 0, 1))
    healthy(m)
    assert outward_closed(m) and volume(m) == pytest.approx(8)
    on = r.data["cut_vertices"].vertices
    assert len(on) == 4 and all(abs(m.position(v)[2] - 0.3) < 1e-15 for v in on)
    assert len(r.data["cut_edges"].edges) == 4
    assert len(r.data["front"].faces) == 5 and len(r.data["back"].faces) == 5
    c = box(2)
    r = c.bisect((0, 0, 0.3), (0, 0, 1), clear="front", fill=True)
    healthy(c)
    assert outward_closed(c) and volume(c) == pytest.approx(2 * 2 * 1.3)
    assert len(r.data["fill_faces"].faces) == 1 and not r.data["front"].faces


def test_bisect_an_oblique_plane_through_a_sphere_and_fill():
    m = sphere_mesh()
    n = np.array([0.3, 0.1, 1.0])
    r = m.bisect((0, 0, 0.05), n, clear="back", fill=True)
    healthy(m)
    assert outward_closed(m)
    unit = n / np.linalg.norm(n)
    signed = (m.positions() - np.array([0, 0, 0.05])) @ unit
    assert signed.min() > -1e-12  # nothing behind the plane remains
    (cap,) = r.data["fill_faces"].faces
    assert np.allclose(np.array(m.face_normal(cap)), -unit)  # the cap faces backward


def test_bisect_snaps_vertices_within_the_threshold_and_spares_other_faces():
    m = grid(2, 2, 2)
    count = m.vertex_count
    r = m.bisect((0, 1e-9, 0), (0, 1, 0))  # the grid's middle line, a hair off
    assert m.vertex_count == count and r.data["front"].faces
    g = grid(4, 1, (4, 1))
    left = g.select_faces(box=((-2, -1, -1), (0, 1, 1)))
    before = g.face_count
    r = g.bisect((-0.5, 0, 0), (1, 0.2, 0), faces=left)
    healthy(g)
    assert g.face_count > before  # the selected faces were split ...
    assert all(f in g._faces for f in g.select_faces(box=((0.5, -1, -1), (2, 1, 1))).faces)


def test_bisect_triangulates_concave_faces_crossed_more_than_twice():
    m = EditableMesh()
    pts = [(0, 0, 0), (4, 0, 0), (4, 3, 0), (3, 3, 0), (3, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)]
    ids = [m.create_vertex(p).data["vertex"] for p in pts]
    m.create_face(ids)  # a U shape: the line y = 2 crosses it four times
    r = m.bisect((0, 2, 0), (0, 1, 0))
    assert "BISECT_TRIANGULATED" in [d.code for d in r.diagnostics]
    healthy(m)
    area = sum(m.face_area(f) for f in m.face_ids())
    assert area == pytest.approx(4 * 3 - 2 * 2)


def test_bisect_refusals_change_nothing():
    m = box(2)
    before = snapshot(m)
    with pytest.raises(ValidationError):
        m.bisect((0, 0, 0), (0, 0, 0))
    with pytest.raises(ValidationError):
        m.bisect((0, 0, 0), (0, 0, 1), fill=True)
    with pytest.raises(ValidationError):
        m.bisect((0, 0, 0), (0, 0, 1), clear="up")
    assert snapshot(m) == before


def test_bisect_in_world_space_under_a_rotated_scaled_parent():
    scene = Scene()
    obj = scene.add_object("Block", mesh=create_cube(2), position=(5, 0, 0), scale=(2, 1, 1))
    with obj.edit_mesh() as edit:
        edit.bisect((6, 0, 0), (1, 0, 0), space="world", clear="front", fill=True)
    lo, hi = obj.mesh.bounds()
    assert hi[0] == pytest.approx(0.5) and lo[0] == pytest.approx(-1)  # world x = 6 -> 0.5


# --------------------------------------------------------------------------- #
# Fill and bridge
# --------------------------------------------------------------------------- #


def test_fill_holes_closes_boxes_and_sphere_caps():
    m = box(2)
    m.delete_faces(m.select_faces(normal=((0, 0, 1), 1), expect=1))
    r = m.fill_holes()
    healthy(m)
    assert outward_closed(m) and volume(m) == pytest.approx(8)
    assert len(r.data["faces"].faces) == 1
    s = sphere_mesh(16, 8)
    s.delete_faces(s.select_faces(normal=((0, 0, 1), 40)))
    r = s.fill_holes(mode="triangulate")
    healthy(s)
    assert outward_closed(s)
    assert all(len(s.face_vertices(f)) == 3 for f in r.data["faces"].faces)
    f = sphere_mesh(16, 8)
    f.delete_faces(f.select_faces(normal=((0, 0, -1), 40)))
    f.fill_holes(mode="fan")
    assert outward_closed(f)


def test_fill_holes_modes_and_limits():
    m = box(2)
    m.delete_faces(m.select_faces(normal=((0, 0, 1), 1), expect=1))
    r = m.fill_holes(max_sides=3)
    assert not r.data["faces"].faces and "HOLES_NOT_FILLED" in [d.code for d in r.diagnostics]
    with pytest.raises(ValidationError):
        m.fill_holes(mode="magic")
    inner = [e for e in m.edge_ids() if len(m.edge_faces(e)) == 2][:1]
    before = snapshot(m)
    with pytest.raises(EditError, match="NOT_BOUNDARY"):
        m.fill_holes(inner)
    assert snapshot(m) == before


def tube(z0: float, z1: float) -> EditableMesh:
    m = EditableMesh()
    m.add_box((2, 2, z1 - z0), center=(0, 0, (z0 + z1) / 2))
    m.delete_faces(m.select_faces(normal=((0, 0, 1), 1)) | m.select_faces(normal=((0, 0, -1), 1)))
    return m


def two_tubes() -> EditableMesh:
    m = tube(-1, 1)
    other = tube(3, 5)
    offset = m._nv
    ids = {}
    for v in other.vertex_ids():
        ids[v] = m.create_vertex(other.position(v)).data["vertex"]
    for f in other.face_ids():
        m.create_face([ids[v] for v in other.face_vertices(f)])
    del offset
    return m


def ring(m: EditableMesh, z: float) -> list[int]:
    return [
        e
        for e in m.boundary_edges().edges
        if all(abs(m.position(v)[2] - z) < 1e-9 for v in m.edge_vertices(e))
    ]


@pytest.mark.parametrize("segments", [1, 3])
def test_bridge_joins_two_tubes_into_one(segments):
    m = two_tubes()
    r = m.bridge_edge_loops(ring(m, 1) + ring(m, 3), segments=segments)
    healthy(m)
    assert len(r.data["faces"].faces) == 4 * segments
    assert len(m.boundary_edges().edges) == 8  # only the two far rims remain open
    # The bridge's rows are straight: intermediate vertices interpolate linearly.
    zs = sorted({round(m.position(v)[2], 12) for v in r.data["vertices"].vertices})
    assert zs == [round(1 + 2 * k / segments, 12) for k in range(1, segments)]
    m.fill_holes()
    assert outward_closed(m) and volume(m) == pytest.approx(2 * 2 * 6)


def test_bridge_with_different_counts_makes_triangles():
    m = two_tubes()
    m.split_edges(ring(m, 3), cuts=1)  # the upper loop now has 8 vertices
    r = m.bridge_edge_loops(ring(m, 1) + ring(m, 3))
    healthy(m)
    assert all(len(m.face_vertices(f)) == 3 for f in r.data["faces"].faces)
    assert len(r.data["faces"].faces) == 12
    m.fill_holes()
    assert outward_closed(m)
    with pytest.raises(EditError, match="BRIDGE_SEGMENTS_NEED_EQUAL_COUNTS"):
        n = two_tubes()
        n.split_edges(ring(n, 3), cuts=1)
        n.bridge_edge_loops(ring(n, 1) + ring(n, 3), segments=2)


def test_bridge_refusals():
    m = two_tubes()
    before = snapshot(m)
    with pytest.raises(EditError, match="BRIDGE_NEEDS_TWO_LOOPS"):
        m.bridge_edge_loops(ring(m, 1))
    inner = [e for e in m.edge_ids() if len(m.edge_faces(e)) == 2][:2]
    with pytest.raises(EditError, match="NOT_BOUNDARY"):
        m.bridge_edge_loops(inner)
    assert snapshot(m) == before


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #


def test_edge_slide_moves_a_loop_along_its_rails_and_keeps_uvs_affine():
    g = grid(4, 4, 4)
    (e,) = g.select_edges(nearest=(-1.5, -1, 0)).edges  # on the row y = -1
    loop = g.edge_loop(e)
    verts = {v for e2 in loop.edges for v in g.edge_vertices(e2)}
    r = g.slide_edges(loop, 0.5)
    healthy(g)
    ys = {round(g.position(v)[1], 12) for v in verts}
    assert len(ys) == 1 and ys.pop() in (-1.5, -0.5)
    # UVs still follow the grid's affine layout u = (x + 2) / 4, v = (y + 2) / 4.
    for f in g.face_ids():
        for v, uv in zip(g.face_vertices(f), g.corner_values(f, "uv"), strict=True):
            x, y, _ = g.position(v)
            assert uv == pytest.approx(((x + 2) / 4, (y + 2) / 4), abs=1e-12)
    assert set(r.data["vertices"]) == verts


def test_even_slide_moves_every_vertex_the_same_distance():
    g = grid(4, 4, 4)
    # Rails toward the bottom row: length 1 on the left half, 2 on the right half.
    g.translate(
        [v for v in g.vertex_ids() if g.position(v)[1] == -2 and g.position(v)[0] > 0],
        (0, -1, 0),
    )
    (e,) = g.select_edges(nearest=(-1.5, -1, 0)).edges
    loop = g.edge_loop(e)
    verts = sorted({v for e2 in loop.edges for v in g.edge_vertices(e2)})
    before = g.positions(verts)
    g.slide_edges(loop, -0.5, even=True)
    moved = np.linalg.norm(g.positions(verts) - before, axis=1)
    assert np.allclose(moved, moved[0]) and moved[0] == pytest.approx(0.5)


def test_slides_refuse_folds_and_bad_factors():
    g = grid(4, 4, 4)
    (e,) = g.select_edges(nearest=(-1.5, -1, 0)).edges
    loop = g.edge_loop(e)
    before = snapshot(g)
    with pytest.raises(EditError, match="SLIDE_FOLDS_FACE"):
        g.slide_edges(loop, 1.0)
    with pytest.raises(ValidationError):
        g.slide_edges(loop, 1.5)
    assert snapshot(g) == before
    v = next(iter(g.select_vertices(nearest=(0, 0, 0)).vertices))
    u = next(iter(g.select_vertices(nearest=(1, 0, 0)).vertices))
    g.slide_vertex(v, u, 0.25)
    assert g.position(v) == pytest.approx((0.25, 0, 0))
    with pytest.raises(EditError, match="NOT_ADJACENT"):
        g.slide_vertex(v, next(iter(g.select_vertices(nearest=(2, 2, 0)).vertices)), 0.5)


# --------------------------------------------------------------------------- #
# Dissolve, connect, cleanup
# --------------------------------------------------------------------------- #


def test_dissolve_vertices_and_faces():
    g = grid(3, 3, 3)
    v = next(iter(g.select_vertices(nearest=(0.5, 0.5, 0)).vertices))
    r = g.dissolve_vertices([v])
    healthy(g)
    (merged,) = r.data["faces"].faces
    assert g.face_count == 6 and len(g.face_vertices(merged)) == 8
    assert g.face_area(merged) == pytest.approx(4)
    h = grid(3, 3, 3)
    mid = next(iter(h.select_vertices(nearest=(0.5, -1.5, 0)).vertices))  # on the boundary
    h.dissolve_vertices([mid])  # two edges: it leaves the loops
    healthy(h)
    d = grid(3, 3, 3)
    r = d.dissolve_faces(d.select_all())
    healthy(d)
    (f,) = r.data["faces"].faces
    assert d.face_count == 1 and d.face_area(f) == pytest.approx(9)
    hole = grid(3, 3, 3)
    ring_faces = hole.select_faces(box=((-2, -2, -1), (2, 2, 1))) - hole.select_faces(
        box=((-0.6, -0.6, -1), (0.6, 0.6, 1))
    )
    before = snapshot(hole)
    with pytest.raises(EditError, match="DISSOLVE_INVALID_FACE"):
        hole.dissolve_faces(ring_faces)
    assert snapshot(hole) == before


def test_limited_dissolve_rebuilds_planar_faces_but_respects_groups():
    m = EditableMesh.from_mesh(create_cube(2), quads=False)
    assert m.face_count == 12
    m.limited_dissolve(1.0)
    healthy(m)
    assert m.face_count == 6 and outward_closed(m)
    g = EditableMesh.from_mesh(create_cube(2), quads=False)
    g.set_group([g.face_ids()[0]], 4)
    g.limited_dissolve(1.0)
    assert g.face_count == 7


def test_connect_vertices_and_paths():
    g = grid(2, 2, 2)
    a = next(iter(g.select_vertices(nearest=(-1, -1, 0)).vertices))
    b = next(iter(g.select_vertices(nearest=(0, 0, 0)).vertices))
    c = next(iter(g.select_vertices(nearest=(1, 1, 0)).vertices))
    r = g.connect_vertex_path([a, b, c])
    healthy(g)
    assert len(r.data["edges"].edges) == 2 and g.face_count == 6
    with pytest.raises(EditError, match="CONNECT_NO_SHARED_FACE"):
        g.connect_vertex_path([a, c])
    with pytest.raises(ValidationError):
        g.connect_vertex_path(g.select_vertices(ids=[a, b]))


def test_cleanup_reports_what_it_fixed_and_what_remains():
    m = box(2)
    m.add_box(2)  # coincident duplicate
    m.merge_by_distance(0.0)
    m.create_vertex((5, 5, 5))
    r = m.cleanup()
    steps = {s["step"]: s for s in r.data["report"]}
    assert steps["duplicate_faces"]["fixed"] == 6
    assert steps["loose_vertices"]["fixed"] == 1
    assert steps["zero_area_faces"]["remaining"] == 0
    healthy(m)
    assert outward_closed(m)


# --------------------------------------------------------------------------- #
# Shading
# --------------------------------------------------------------------------- #


def test_recalculate_normals_restores_outward_winding():
    m = sphere_mesh()
    rng = np.random.default_rng(3)
    flips = sorted(int(f) for f in rng.choice(m.face_ids(), size=40, replace=False))
    m.flip_faces(flips)
    r = m.recalculate_normals()
    assert sorted(r.data["flipped"]) == flips
    assert outward_closed(m)
    m.recalculate_normals(inside=True)
    assert volume(m) < 0


def test_recalculate_normals_on_open_and_non_orientable_pieces():
    g = grid(2, 2, 2)
    g.flip_faces(g.select_all())
    r = g.recalculate_normals()
    assert "ORIENTATION_HEURISTIC" in [d.code for d in r.diagnostics]
    # A Moebius strip of 8 quads cannot be oriented.
    strip = EditableMesh()
    n = 8
    top, bottom = [], []
    for i in range(n):
        t = 2 * math.pi * i / n
        w = 0.3 * math.cos(t / 2)
        h = 0.3 * math.sin(t / 2)
        center = np.array([math.cos(t), math.sin(t), 0])
        radial = np.array([math.cos(t), math.sin(t), 0])
        top.append(strip.create_vertex(center + w * radial + (0, 0, h)).data["vertex"])
        bottom.append(strip.create_vertex(center - w * radial - (0, 0, h)).data["vertex"])
    for i in range(n - 1):
        strip.create_face([bottom[i], bottom[i + 1], top[i + 1], top[i]])
    strip.create_face([bottom[n - 1], top[0], bottom[0], top[n - 1]], allow_non_manifold=True)
    before = snapshot(strip)
    r = strip.recalculate_normals()
    assert "NON_ORIENTABLE" in [d.code for d in r.diagnostics]
    assert snapshot(strip) == before


def test_auto_smooth_marks_creases_only():
    m = EditableMesh.from_mesh(create_sphere(1.0, 16, 8))
    m.set_sharp(m.select_all(), True)
    r = m.set_auto_smooth(30)
    assert not r.data["sharp_edges"].edges and not any(m.is_sharp(e) for e in m.edge_ids())
    b = box(2)
    r = b.set_auto_smooth(30)
    assert len(r.data["sharp_edges"].edges) == 12 and all(b.is_smooth(f) for f in b.face_ids())


def test_weighted_normals_favour_large_faces_and_survive_commit():
    scene = Scene()
    obj = scene.add_object("Bevelled", mesh=create_cube(2))
    with obj.edit_mesh() as edit:
        edit.bevel_edges(edit.select_all(), 0.1, segments=2)
        edit.set_auto_smooth(80)
        edit.set_weighted_normals(weight=100)
        top = next(iter(edit.select_faces(normal=((0, 0, 1), 1), expect=1).faces))
        written = edit.corner_values(top, "normal")
    # The big top face's corners point (almost) straight up despite the rounded edges.
    assert all(n[2] > 0.99 for n in written)
    normals = obj.mesh.normals
    ups = normals[obj.mesh.positions[:, 2] > 0.999]
    assert ups[:, 2].min() > 0.99


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #


def test_extrude_edges_builds_walls_wound_like_their_faces():
    g = grid(2, 2, 2)
    r = g.extrude_edges(g.boundary_edges(), (0, 0, 1))
    healthy(g)
    assert len(r.data["faces"].faces) == 8 and len(r.data["vertices"]) == 8
    assert "INCONSISTENT_WINDING" not in g.validate().codes()
    with pytest.raises(EditError, match="NOT_BOUNDARY"):
        g.extrude_edges([e for e in g.edge_ids() if len(g.edge_faces(e)) == 2][:1], (0, 0, 1))


@pytest.mark.parametrize("steps", [8, 16, 32])
def test_spin_makes_a_closed_torus_with_the_polygonal_volume(steps):
    m = EditableMesh()
    ids = [
        m.create_vertex(p).data["vertex"]
        for p in [(1.5, 0, -0.5), (2.5, 0, -0.5), (2.5, 0, 0.5), (1.5, 0, 0.5)]
    ]
    m.create_face([ids[0], ids[3], ids[2], ids[1]])
    m.spin(m.select_all(), 360, steps)
    healthy(m)
    assert outward_closed(m) and euler(m) == 0
    # A polygon of `steps` sides rotated: V = (n / 2) sin(2 pi / n) * integral of r^2 dz.
    expected = steps / 2 * math.sin(2 * math.pi / steps) * (2.5**2 - 1.5**2)
    assert volume(m) == pytest.approx(expected, rel=1e-12)


def test_partial_spin_keeps_the_end_region():
    m = EditableMesh()
    ids = [m.create_vertex(p).data["vertex"] for p in [(1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)]]
    m.create_face([ids[0], ids[3], ids[2], ids[1]])
    r = m.spin(m.select_all(), 90, 4)
    healthy(m)
    (end,) = r.data["faces"].faces
    assert np.allclose(np.array(m.face_center(end))[:2], (0, 1.5), atol=1e-12)


def test_lathe_builds_closed_solids_with_uvs():
    m = EditableMesh()
    r = m.add_lathe([(0, 0), (1, 0), (1, 2), (0, 2)], segments=24)
    healthy(m)
    assert outward_closed(m)
    assert volume(m) == pytest.approx(12 * math.sin(2 * math.pi / 24) * 2, rel=1e-12)
    uvs = np.array([uv for f in r.data["faces"].faces for uv in m.corner_values(f, "uv")])
    assert uvs.min() == 0 and uvs.max() == 1
    with pytest.raises(ValidationError):
        m.add_lathe([(1, 0)])
    with pytest.raises(ValidationError):
        m.add_lathe([(-1, 0), (1, 1)])


# --------------------------------------------------------------------------- #
# Shrinkwrap and knife
# --------------------------------------------------------------------------- #


def test_shrinkwrap_nearest_and_project():
    m = EditableMesh()
    m.add_box(3)
    m.subdivide_faces(m.select_all(), 3)
    target = create_sphere(1.0, 64, 32)
    r = m.shrinkwrap(m.select_all(), target)
    assert len(r.data["moved"].vertices) == m.vertex_count
    radius = np.linalg.norm(m.positions(), axis=1)
    assert radius.max() <= 1 + 1e-6 and radius.min() > math.cos(math.pi / 32) - 1e-6
    g = grid(4, 4, 4)
    g.translate(g.select_all(), (0, 0, 3))
    floor = Mesh([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], [(0, 1, 2), (0, 2, 3)])
    g.shrinkwrap(g.select_all(), floor, mode="project", direction=(0, 0, -1), offset=0.25)
    assert np.allclose(g.positions()[:, 2], 0.25)
    far = grid(1, 1, 1)
    far.translate(far.select_all(), (100, 0, 0))
    r = far.shrinkwrap(far.select_all(), floor, mode="project", direction=(0, 0, 1))
    assert "SHRINKWRAP_MISSED" in [d.code for d in r.diagnostics]


def test_shrinkwrap_onto_another_scene_object():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 64, 32), position=(0, 0, 5), scale=2)
    sheet = scene.add_object("Sheet", mesh=create_cube(1), position=(0, 0, 9))
    with sheet.edit_mesh() as edit:
        edit.shrinkwrap(edit.select_all(), ball)
    world = sheet.mesh.positions @ np.asarray(sheet.world_matrix)[:3, :3].T + (0, 0, 9)
    assert np.allclose(np.linalg.norm(world - (0, 0, 5), axis=1), 2, atol=0.01)


def test_knife_cuts_along_a_projected_line():
    m = box(2)
    r = m.knife_project([(-2, -0.3, 3), (2, 0.3, 3)], (0, 0, -1), through=False)
    healthy(m)
    assert outward_closed(m) and volume(m) == pytest.approx(8)
    assert len(r.data["vertices"].vertices) == 2 and len(r.data["faces"].faces) == 2
    t = box(2)
    # Through: the cut's vertical ribbon also crosses the bottom and the two side faces.
    t.knife_project([(-2, -0.3, 3), (2, 0.3, 3)], (0, 0, -1))
    healthy(t)
    assert t.face_count == 10 and outward_closed(t)


def test_knife_corners_inside_a_face_become_vertices():
    m = box(2)
    path = [(-2, -0.5, 3), (0, 0.5, 3), (2, -0.5, 3)]  # a V whose corner is inside the top
    r = m.knife_project(path, (0, 0, -1), through=False)
    healthy(m)
    corner = [v for v in r.data["vertices"].vertices if np.allclose(m.position(v), (0, 0.5, 1))]
    assert len(corner) == 1
    assert outward_closed(m)
    # A cut ending inside a face is not applied, and says so.
    p = box(2)
    before = p.face_count
    r = p.knife_project([(-2, 0, 3), (0, 0, 3)], (0, 0, -1), through=False)
    assert p.face_count == before and "KNIFE_PARTIAL_CUT" in [d.code for d in r.diagnostics]


# --------------------------------------------------------------------------- #
# Undo, redo, scripts
# --------------------------------------------------------------------------- #


CASES = [
    ("bisect", lambda m: m.bisect((0, 0, 0.2), (0.2, 0.1, 1), clear="front", fill=True)),
    ("fill", lambda m: (m.delete_faces([m.face_ids()[0]]), m.fill_holes())),
    ("dissolve", lambda m: m.dissolve_faces([m.face_ids()[0], m.face_ids()[1]])),
    ("normals", lambda m: (m.flip_faces([m.face_ids()[3]]), m.recalculate_normals())),
    ("weighted", lambda m: m.set_weighted_normals()),
    ("knife", lambda m: m.knife_project([(-3, 0.1, 3), (3, -0.1, 3)], (0, 0, -1))),
    ("spin", lambda m: m.spin([m.face_ids()[0]], 120, 4, center=(3, 0, 0))),
]


@pytest.mark.parametrize("name,ops", CASES, ids=[c[0] for c in CASES])
def test_new_operations_undo_and_redo_exactly(name, ops):
    m = sphere_mesh(16, 8)
    m.subdivide_faces(m.select_all(), 1)
    start = snapshot(m)
    applied = len(m.history)
    ops(m)
    done = snapshot(m)
    count = len(m.history) - applied
    for _ in range(count):
        m.undo()
    assert snapshot(m) == start
    for _ in range(count):
        m.redo()
    assert snapshot(m) == done


def test_edit_scripts_reach_the_new_operations():
    scene = Scene()
    scene.add_object("Crate", mesh=create_cube(2))
    scene.add_object("Ball", mesh=create_sphere(1.0, 32, 16), position=(0, 0, 4))
    script = {
        "format": "aether3d.edit-script",
        "version": 1,
        "edits": [
            {
                "object": "Crate",
                "operations": [
                    {
                        "op": "bevel_edges",
                        "edges": {"select": {"kind": "edges"}},
                        "width": 0.1,
                        "segments": 3,
                    },
                    {
                        "op": "bisect",
                        "point": [0, 0, 0.5],
                        "normal": [0, 0, 1],
                        "clear": "front",
                        "fill": True,
                    },
                    {"op": "validate", "require_ok": True},
                ],
            },
            {
                "object": "Ball",
                "operations": [
                    {
                        "op": "shrinkwrap",
                        "vertices": {
                            "select": {"kind": "vertices", "box": [[-2, -2, -2], [2, 2, -0.5]]}
                        },
                        "target": "Crate",
                        "mode": "project",
                        "direction": [0, 0, -1],
                    },
                ],
            },
        ],
    }
    out = run_edit_script(scene, script)
    assert out.edits[0]["commit"]["validation"]["status"] in ("ok", "warning")
    crate = scene.get_by_name("Crate").mesh
    assert crate.bounds()[1][2] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Procedural: creases and mirror bisect
# --------------------------------------------------------------------------- #


def test_subdivision_creases_keep_sharp_rims():
    scene = Scene()
    radii = {}
    for crease in (0.0, 0.5, 4.0):
        node = scene.add_geometry(f"Can{crease}", "cylinder", radius=1, depth=2, segments=16)
        node.add_modifier("subdivision", levels=3, crease=crease)
        p = node.evaluate().positions.astype(np.float64)
        radii[crease] = (p[:, 2].max(), np.linalg.norm(p[:, :2], axis=1).max())
    assert radii[0.0][0] < radii[0.5][0] < radii[4.0][0] == pytest.approx(1.0)
    plain = scene.add_geometry("Plain", "cylinder", radius=1, depth=2, segments=16)
    plain.add_modifier("subdivision", levels=3)
    assert plain.evaluate() == scene.procedural.geometry("Can0.0").evaluate()


def test_a_fully_creased_cube_stays_a_cube():
    scene = Scene()
    cube = scene.add_geometry("Sharp", "box", width=2, depth=2, height=2)
    cube.add_modifier("subdivision", levels=2, crease=2)
    p = cube.evaluate().positions.astype(np.float64)
    # No edge of a box primitive is marked sharp: the crease has nothing to keep.
    assert np.abs(p).max() < 1
    from aether3d.procedural.modifiers import catmull_clark
    from aether3d.procedural.polymesh import PolyMesh

    box_pm = PolyMesh.from_mesh(create_cube(2))
    edges = box_pm.edges[0]
    sharp = box_pm.with_(sharp_edges=edges)
    out = catmull_clark(catmull_clark(sharp, crease=2.0), crease=1.0)
    q = out.positions
    assert np.allclose(np.abs(q).max(axis=1), 1.0)  # every vertex on the cube's surface


def test_mirror_bisect_cuts_before_mirroring():
    scene = Scene()
    node = scene.add_geometry("Half", "box", width=2, depth=2, height=2)
    node.add_modifier("transform", translation=(0.3, 0, 0))
    node.add_modifier("mirror", axes="x", bisect=True)
    mesh = node.evaluate()
    lo, hi = mesh.bounds()
    assert lo[0] == pytest.approx(-1.3) and hi[0] == pytest.approx(1.3)
    codes = {d.code for d in scene.procedural.evaluate(node).diagnostics}
    assert "MIRROR_OVERLAP" not in codes
    flipped = scene.add_geometry("Other", "box", width=2, depth=2, height=2)
    flipped.add_modifier("transform", translation=(0.3, 0, 0))
    flipped.add_modifier("mirror", axes="x", bisect=True, bisect_flip=True)
    lo, hi = flipped.evaluate().bounds()
    assert hi[0] == pytest.approx(0.7)
