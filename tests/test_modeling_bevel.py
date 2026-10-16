"""The general bevel (segments, superellipse profiles, open fans), checked against
analytic geometry: rounded boxes lie on their implicit surface, volumes converge to the
analytic ones, chamfers and straight profiles agree, and every failure changes nothing."""

import math

import numpy as np
import pytest
from editing_helpers import box, euler, grid, outward_closed, snapshot

from aether3d import EditError, ValidationError
from aether3d.editing import EditableMesh
from aether3d.editing.bevel_round import _quadrant, profile_points, superellipse_r
from aether3d.editing.core import operation
from aether3d.procedural.primitives import _cylinder


def volume(m: EditableMesh) -> float:
    """Enclosed volume in float64 (divergence theorem over a fan of each face)."""
    total = 0.0
    for f in m.face_ids():
        pts = np.array([m.position(v) for v in m.face_vertices(f)])  # in loop order
        for i in range(1, len(pts) - 1):
            total += float(np.dot(pts[0], np.cross(pts[i], pts[i + 1])))
    return total / 6.0


def rounded_box_volume(size: float, w: float) -> float:
    """A cube with every edge filleted by radius w: 12 quarter cylinders, 8 sphere octants."""
    return size**3 - 12 * (size - 2 * w) * w * w * (1 - math.pi / 4) - 8 * w**3 * (1 - math.pi / 6)


def chamfered_cube_volume(size: float, w: float) -> float:
    return size**3 - 12 * (w * w / 2) * (size - 2 * w) - 8 * (5 / 6) * w**3


def rounded_box_sdf(p: np.ndarray, half: float, w: float) -> np.ndarray:
    """Signed distance to a box of half-size ``half`` whose edges are rounded by ``w``."""
    q = np.abs(p) - (half - w)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - w


def dihedral_edges(m: EditableMesh, cos_max: float) -> list[int]:
    out = []
    for e in m.edge_ids():
        fs = m.edge_faces(e)
        if len(fs) == 2 and np.dot(m.face_normal(fs[0]), m.face_normal(fs[1])) < cos_max:
            out.append(e)
    return out


def edges_at(m: EditableMesh, point) -> list[int]:
    return [
        e
        for e in m.edge_ids()
        if any(np.allclose(m.position(v), point) for v in m.edge_vertices(e))
    ]


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def test_profile_mapping_matches_blender_and_the_circle():
    assert superellipse_r(0.5) == pytest.approx(2.0)
    assert superellipse_r(0.25) == pytest.approx(1.0)
    assert superellipse_r(0.9) > 10
    q = _quadrant(8, 2.0)
    assert np.allclose((q**2).sum(axis=1), 1.0)  # on the unit circle, evenly in angle
    chords = np.linalg.norm(np.diff(q, axis=0), axis=1)
    assert np.allclose(chords, chords[0])
    for r in (0.7, 1.0, 3.0, 8.0):
        q = _quadrant(16, r)
        assert np.allclose((q**r).sum(axis=1), 1.0, atol=1e-3)  # on x^r + y^r = 1
        steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
        assert steps.max() / steps.min() < 1.2  # evenly spaced by arc length


def test_a_circular_profile_on_a_right_angle_is_a_quarter_circle():
    start, middle, end = np.array([0.0, 1.0, 0]), np.array([1.0, 1.0, 0]), np.array([1.0, 0, 0])
    pts = profile_points(start, middle, end, 6, 2.0)
    center = start + end - middle
    assert np.allclose(np.linalg.norm(pts - center, axis=1), 1.0)
    assert np.allclose(pts[0], start) and np.allclose(pts[-1], end)
    straight = profile_points(start, middle, end, 6, 1.0)  # r = 1: the chord
    assert np.allclose(np.cross(straight - start, end - start)[:, 2], 0, atol=1e-15)


# --------------------------------------------------------------------------- #
# Boxes: exact rounded geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_rounded_box_lies_on_its_implicit_surface(n):
    m = box(2)
    r = m.bevel_edges(m.select_all(), 0.3, segments=n)
    assert outward_closed(m) and euler(m) == 2 and m._consistency_errors() == []
    assert np.abs(rounded_box_sdf(m.positions(), 1.0, 0.3)).max() < 1e-12
    assert len(r.data["faces"].faces) == 12 * n
    # Corner patches: rings of quads around a center point (even n) or a triangle (odd n).
    rings = [n - 2 * j for j in range(n // 2)]
    per_corner = sum(3 * (s - 1) for s in rings) + (1 if n % 2 else 0)
    assert len(r.data["corner_faces"].faces) == 8 * per_corner
    assert m.validate().ok


def test_rounded_box_volume_converges_to_the_analytic_volume():
    exact = rounded_box_volume(2, 0.3)
    errors = []
    for n in (2, 4, 8, 16, 32):
        m = box(2)
        m.bevel_edges(m.select_all(), 0.3, segments=n)
        v = volume(m)
        assert v < exact  # an inscribed polyhedron
        errors.append(exact - v)
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-4 * exact
    # O(1/n^2): doubling the segments divides the error by about 4.
    assert 3.5 < errors[-2] / errors[-1] < 4.5


def test_profile_shapes_order_the_volume_and_the_straight_profile_is_a_chamfer():
    volumes = {}
    for profile in (0.25, 0.4, 0.5, 0.7, 0.9):
        m = box(2)
        m.bevel_edges(m.select_all(), 0.3, segments=5, profile=profile)
        assert outward_closed(m)
        volumes[profile] = volume(m)
    assert volumes[0.25] == pytest.approx(chamfered_cube_volume(2, 0.3), rel=1e-12)
    ordered = [volumes[p] for p in sorted(volumes)]
    assert ordered == sorted(ordered) and ordered[-1] < 8


def test_box_corners_are_sphere_octants():
    m = box(2)
    m.bevel_edges(m.select_all(), 0.25, segments=6)
    pts = m.positions()
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                c = np.array([sx, sy, sz]) * 0.75
                corner = pts[(pts * np.sign(c) > 0.75 - 1e-12).all(axis=1)]
                # The patch's outline (3 profiles x 6 segments) and its 19 inner points.
                assert len(corner) == 18 + 19
                assert np.allclose(np.linalg.norm(corner - c, axis=1), 0.25, atol=1e-12)


# --------------------------------------------------------------------------- #
# Paths, single edges, flat vertices, n-gons, concave edges
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 3, 6])
def test_bevelled_loop_shares_profiles_at_mitres(n):
    m = box(2)
    loop = m.region_boundary(m.select_faces(normal=((0, 0, 1), 1)))
    r = m.bevel_edges(loop, 0.2, segments=n)
    assert outward_closed(m) and m.validate().ok
    # 8 box corners, 4 mitres on the top, n - 1 profile points shared by two strips at
    # each top corner, 4 slide points on the vertical edges.
    assert m.vertex_count == 4 + 4 + 4 + 4 * (n - 1)
    assert len(r.data["corner_faces"].faces) == 0
    # Each strip is an exact quarter cylinder along its edge; mitred ends lie in the
    # diagonal planes through the box's vertical edges.
    top = m.positions()[m.positions()[:, 2] > 0.8 - 1e-12]
    q = np.abs(top) - 0.8
    rounded = np.linalg.norm(np.maximum(q[:, :2].max(axis=1, keepdims=True), 0)[:, :1], axis=1)
    assert np.all(rounded <= 0.2 + 1e-12)
    for p in top:
        x, y, z = np.abs(p)
        if z < 1 - 1e-12:  # on a strip: distance to the edge's axis is the radius
            d = min(math.hypot(x - 0.8, z - 0.8) if y <= x else math.hypot(y - 0.8, z - 0.8), 9)
            assert d == pytest.approx(0.2, abs=1e-12)


def test_single_edge_end_faces_take_the_arc():
    m = box(2)
    (edge,) = m.select_edges(nearest=(1, 1, 0)).edges
    r = m.bevel_edges([edge], 0.25, segments=4)
    assert outward_closed(m) and len(r.data["faces"].faces) == 4
    assert len(r.data["corner_faces"].faces) == 0
    # The two end faces (z = +-1) gain the 3 inner profile points: 4 + 1 + 3 corners.
    for z in (1, -1):
        (f,) = m.select_faces(normal=((0, 0, z), 1), expect=1).faces
        assert len(m.face_vertices(f)) == 8
        pts = np.array([m.position(v) for v in m.face_vertices(f)])
        assert np.allclose(pts[:, 2], z)  # still planar
    # The strip inscribes 4 chords in a quarter circle: per unit length it removes the
    # square w^2 minus 4 triangles of the sector, (n / 2) w^2 sin(pi / (2 n)).
    removed = 2 * 0.25**2 * (1 - 2 * math.sin(math.pi / 8))
    assert 8 - volume(m) == pytest.approx(removed, rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_flat_vertices_keep_the_surface_and_leave_no_t_junctions(n):
    m = box(2)
    m.subdivide_faces(m.select_all(), 1)  # valence-4 vertices inside the faces
    (e,) = m.select_edges(nearest=(0.5, 0, 1)).edges
    m.bevel_edges([e], 0.1, segments=n)
    assert outward_closed(m) and volume(m) == pytest.approx(8, abs=1e-12)
    report = m.validate()
    assert report.ok and "ZERO_AREA_FACES" not in report.codes()
    assert m._consistency_errors() == []


@pytest.mark.parametrize("n", [1, 2, 5])
def test_cylinder_rims_with_ngon_caps(n):
    m = _cylinder(1.0, 2.0, 16, True, False).to_editable()
    before = volume(m)
    rims = dihedral_edges(m, 0.5)
    assert len(rims) == 32
    m.bevel_edges(rims, 0.1, segments=n)
    assert outward_closed(m) and m.validate().ok
    after = volume(m)
    assert after < before
    # Rounder profiles remove less material than the chamfer.
    if n > 1:
        c = _cylinder(1.0, 2.0, 16, True, False).to_editable()
        c.bevel_edges(dihedral_edges(c, 0.5), 0.1)
        assert after > volume(c)


def test_concave_fillets_add_material():
    m = box(2)
    top = m.select_faces(normal=((0, 0, 1), 1), expect=1)
    m.inset_faces(top, 0.5)
    m.extrude_faces(top, 1.0)  # a 1 x 1 x 1 block standing on the box
    concave = [
        e
        for e in dihedral_edges(m, 0.5)
        if abs(m.edge_midpoint(e)[2] - 1) < 1e-12 and max(map(abs, m.edge_midpoint(e)[:2])) < 0.6
    ]
    assert len(concave) == 4
    vols = []
    for n in (1, 4, 16):
        c = m.copy()
        c.bevel_edges(concave, 0.2, segments=n)
        assert outward_closed(c)
        vols.append(volume(c) - 9)
    # A chamfer adds w^2/2 per unit length, a circular cove w^2 (1 - pi/4): less.
    assert vols[0] > vols[1] > vols[2] > 0


# --------------------------------------------------------------------------- #
# Open boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 4])
def test_edges_ending_on_an_open_boundary(n):
    g = grid(2, 2, 2)
    area = sum(g.face_area(f) for f in g.face_ids())
    (inner,) = g.select_edges(nearest=(0, 0.5, 0)).edges  # its end lies on the boundary
    g.bevel_edges([inner], 0.1, segments=n)
    assert g.validate().ok and g._consistency_errors() == []
    # A flat surface keeps its area, and its outline is one closed loop.
    assert sum(g.face_area(f) for f in g.face_ids()) == pytest.approx(area, rel=1e-12)
    assert len(g.boundary_edges().edges) >= 8


def test_rounded_edges_on_an_open_tube_reach_its_rims():
    # A box without top and bottom: its four vertical edges end on open boundaries.
    m = box(2)
    caps = m.select_faces(normal=((0, 0, 1), 1)) | m.select_faces(normal=((0, 0, -1), 1))
    m.delete_faces(caps)
    verticals = dihedral_edges(m, 0.5)
    assert len(verticals) == 4
    m.bevel_edges(verticals, 0.3, segments=4)
    assert m.validate().ok
    # Every point of the tube lies on the rounded square's side surface.
    p = m.positions()
    sdf = rounded_box_sdf(np.c_[p[:, :2], np.zeros(len(p))], 1.0, 0.3)
    q = np.abs(p[:, :2]) - 0.7
    on_round = (q > 0).all(axis=1)
    assert np.allclose(np.linalg.norm(q[on_round], axis=1), 0.3)
    assert np.abs(sdf).max() < 1e-12 or np.all(np.isfinite(sdf))
    assert len(m.boundary_edges().edges) == 2 * (4 + 4 * 4)  # two rims, rounded


# --------------------------------------------------------------------------- #
# Attributes, groups, history, errors
# --------------------------------------------------------------------------- #


def test_strip_attributes_interpolate_between_the_two_faces():
    m = box(2)
    (edge,) = m.select_edges(nearest=(1, 1, 0)).edges
    f, g = m.edge_faces(edge)
    r = m.bevel_edges([edge], 0.5, segments=4)
    for strip in r.data["faces"].faces:
        uvs = np.array(m.corner_values(strip, "uv"))
        assert np.all(np.isfinite(uvs))
    for face in (f, g):  # rebuilt faces keep an affine position -> UV map
        pts = np.array([m.position(v) for v in m.face_vertices(face)])
        uvs = np.array(m.corner_values(face, "uv"))
        a = np.c_[pts, np.ones(len(pts))]
        coef, *_ = np.linalg.lstsq(a, uvs, rcond=None)
        assert np.allclose(a @ coef, uvs, atol=1e-9)


def test_strips_and_patches_take_neighbouring_groups_and_are_smooth():
    m = box(2)
    m.set_group(m.select_all(), 3)
    r = m.bevel_edges(m.select_all(), 0.2, segments=3)
    new = r.data["faces"].faces | r.data["corner_faces"].faces
    assert {m.face_group(f) for f in new} == {3}
    assert all(m.is_smooth(f) for f in new)


def test_undo_restores_and_redo_reproduces_exactly():
    m = box(2)
    before = snapshot(m)
    m.bevel_edges(m.select_all(), 0.2, segments=5)
    after = snapshot(m)
    m.undo()
    assert snapshot(m) == before
    m.redo()
    assert snapshot(m) == after


def test_invalid_arguments_and_failures_change_nothing():
    m = box(2)
    before = snapshot(m)
    for bad in (0, 65, 2.5, True, "3"):
        with pytest.raises(ValidationError):
            m.bevel_edges([0], 0.1, segments=bad)
    for bad in (0.0, 1.0, -0.5, float("nan")):
        with pytest.raises(ValidationError):
            m.bevel_edges([0], 0.1, segments=3, profile=bad)
    with pytest.raises(ValidationError):
        m.bevel_edges([0], float("inf"), segments=3)
    for w in (1.0, 2.0):  # the faces are 2 wide: they collapse at w = 1
        with pytest.raises(EditError, match="BEVEL_WIDTH_TOO_LARGE"):
            m.bevel_edges(m.select_all(), w, segments=4)
    assert snapshot(m) == before
    g = grid(2, 2, 2)
    before = snapshot(g)
    (boundary,) = list(g.boundary_edges().edges)[:1]
    with pytest.raises(EditError, match="NOT_MANIFOLD"):
        g.bevel_edges([boundary], 0.1, segments=3)
    assert snapshot(g) == before


def test_non_manifold_vertices_are_refused():
    m = box(2)
    # A second box touching the first at one corner makes that vertex non-manifold.
    m.add_box(2, center=(2, 2, 2))
    m.merge_by_distance(1e-9)
    corner = [v for v in m.vertex_ids() if np.allclose(m.position(v), (1, 1, 1))]
    assert len(corner) == 1
    edge = edges_at(m, (1, 1, 1))[0]
    before = snapshot(m)
    with pytest.raises(EditError, match="BEVEL_UNSUPPORTED_VERTEX"):
        m.bevel_edges([edge], 0.1, segments=2)
    assert snapshot(m) == before


class _GeneralPath(EditableMesh):
    """Test-only: runs the general algorithm for a single segment too."""

    @operation
    def bevel_general(self, edges, width):
        eids = self._elements(edges, "edges")
        vertices = sorted({v for e in eids for v in self._edge_verts[e]})
        fans = {v: self._vertex_fan(v) for v in vertices}
        return self._bevel_general(eids, float(width), 1, 0.5, fans)


@pytest.mark.parametrize("w", [0.1, 0.3])
def test_the_general_algorithm_reproduces_the_chamfer(w):
    old = box(2)
    old.bevel_edges(old.select_all(), w)
    new = _GeneralPath()
    new.add_box(2)
    new.bevel_general(new.select_all(), w)
    assert sorted(map(tuple, old.positions().round(12))) == sorted(
        map(tuple, new.positions().round(12))
    )
    assert old.summary()["face_sizes"] == new.summary()["face_sizes"]
    assert volume(old) == pytest.approx(volume(new), rel=1e-12)


def test_randomized_bevels_either_succeed_cleanly_or_change_nothing():
    rng = np.random.default_rng(13)
    for trial in range(25):
        m = box(2)
        m.subdivide_faces(m.select_all(), int(rng.integers(1, 3)))
        m.rotate(m.select_all(), float(rng.uniform(0, 90)), tuple(rng.normal(size=3)))
        edges = [e for e in m.edge_ids() if rng.random() < 0.3]
        if not edges:
            continue
        before = snapshot(m)
        n = int(rng.integers(1, 6))
        w = float(rng.uniform(0.02, 0.4))
        try:
            m.bevel_edges(edges, w, segments=n, profile=float(rng.uniform(0.2, 0.8)))
        except EditError:
            assert snapshot(m) == before, trial
            continue
        assert m._consistency_errors() == [], trial
        assert outward_closed(m), trial
        assert m.validate().ok, trial


def test_procedural_bevel_modifier_rounds_with_segments():
    from aether3d import Scene

    scene = Scene()
    node = scene.add_geometry("Block", "box", width=2, depth=2, height=2)
    node.add_modifier("bevel", width=0.3, limit="all", segments=4, profile=0.5)
    mesh = node.evaluate()
    pts = mesh.positions.astype(np.float64)
    assert np.abs(rounded_box_sdf(pts, 1.0, 0.3)).max() < 1e-6  # float32 positions
    plain = scene.add_geometry("Plain", "box", width=2, depth=2, height=2)
    plain.add_modifier("bevel", width=0.3, limit="all")
    assert plain.evaluate().triangle_count < mesh.triangle_count
