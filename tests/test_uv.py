"""UV mapping: unwrapping, parameterization, packing, projections, island tools, report."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Mesh, TexturingError, uv
from aether3d.primitives import create_cube, create_plane, create_sphere
from aether3d.uv import param, sparse
from aether3d.uv.topology import Topology


def grid(nx: int, ny: int, z=None) -> Mesh:
    xs, ys = np.meshgrid(np.linspace(0.0, 2.0, nx), np.linspace(0.0, 1.0, ny))
    zs = np.zeros_like(xs) if z is None else z(xs, ys)
    p = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    t = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            t += [[a, a + 1, a + nx + 1], [a, a + nx + 1, a + nx]]
    return Mesh(p, t)


def open_cylinder(radius=0.5, height=2.0, segments=48, rings=12) -> Mesh:
    ang = np.arange(segments) * 2 * np.pi / segments
    h = np.linspace(0.0, height, rings + 1)
    a, z = np.meshgrid(ang, h, indexing="ij")
    p = np.stack([radius * np.cos(a), radius * np.sin(a), z], -1).reshape(-1, 3)
    t = []
    for i in range(segments):
        for j in range(rings):
            v0 = i * (rings + 1) + j
            v1 = ((i + 1) % segments) * (rings + 1) + j
            t += [[v0, v1, v1 + 1], [v0, v1 + 1, v0 + 1]]
    return Mesh(p, t)


def torus(R=1.0, r=0.35, nu=32, nv=12) -> Mesh:
    u, v = np.meshgrid(
        np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij"
    )
    p = np.stack(
        [(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], -1
    ).reshape(-1, 3)
    t = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            t += [[a, b, c], [a, c, d]]
    return Mesh(p, t)


def clean(report: dict) -> None:
    assert report["flipped_triangles"] == 0
    assert report["degenerate_triangles"] == 0
    assert report["overlapping_triangles"] == 0
    assert report["islands_outside_unit_square"] == 0


# --------------------------------------------------------------------------- #
# Sparse solver
# --------------------------------------------------------------------------- #


def _laplacian(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3))
    rows, cols, vals = [], [], []
    for i in range(n):
        d = ((pts - pts[i]) ** 2).sum(axis=1)
        for j in np.argsort(d)[1:6]:
            w = -float(rng.random())
            rows += [i, j]
            cols += [j, i]
            vals += [w, w]
    off = sparse.assemble(n, rows, cols, vals)
    col = np.repeat(np.arange(n), np.diff(off.indptr))
    diag = -np.bincount(col, off.data, n) + 0.05
    m = sparse.assemble(n, np.r_[rows, np.arange(n)], np.r_[cols, np.arange(n)], np.r_[vals, diag])
    return m, pts


def test_ldl_solves_spd_systems_exactly_enough():
    m, pts = _laplacian(400)
    b = np.random.default_rng(1).random((400, 3))
    x = sparse.spd_solve(m, b, pts)
    dense = np.zeros((400, 400))
    dense[m.indices, np.repeat(np.arange(400), np.diff(m.indptr))] = m.data
    assert np.abs(dense @ x - b).max() < 1e-9
    # The structure: ascending row patterns, every row k of L inside columns i < k.
    s = sparse.analyze(m, sparse.nested_dissection(pts, m.indptr, m.indices))
    for k in range(0, 400, 37):
        cols = s.rj[s.rp[k] : s.rp[k + 1]]
        assert np.all(np.diff(cols) > 0) and np.all(cols < k)
        assert np.all(s.li[s.rpos[s.rp[k] : s.rp[k + 1]]] == k)


def test_nested_dissection_reduces_fill_on_a_grid():
    n = 40
    idx = np.arange(n * n).reshape(n, n)
    a = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    b = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    size = n * n
    m = sparse.assemble(
        size,
        np.r_[a, b, np.arange(size)],
        np.r_[b, a, np.arange(size)],
        np.r_[-np.ones(2 * a.size), np.full(size, 4.01)],
    )
    ys, xs = np.divmod(np.arange(size), n)
    pts = np.stack([xs, ys, np.zeros(size)], 1).astype(float)
    nd = sparse.analyze(m, sparse.nested_dissection(pts, m.indptr, m.indices)).li.size
    natural = sparse.analyze(m, np.arange(size)).li.size
    assert nd < 0.6 * natural


def test_indefinite_systems_are_refused():
    m = sparse.assemble(2, [0, 0, 1, 1], [0, 1, 0, 1], [1.0, 2.0, 2.0, 1.0])
    with pytest.raises(TexturingError) as exc:
        sparse.spd_solve(m, np.ones(2), np.zeros((2, 3)))
    assert exc.value.code == "SOLVER_FAILED"


# --------------------------------------------------------------------------- #
# Parameterization
# --------------------------------------------------------------------------- #


def _chartset(mesh: Mesh) -> param.ChartSet:
    t = mesh.triangles.astype(np.int64)
    return param.ChartSet(
        t,
        mesh.positions.astype(np.float64),
        np.zeros(len(t), int),
        np.zeros(mesh.vertex_count, int),
        1,
    )


def test_lscm_of_a_plane_is_a_similarity():
    cs = _chartset(grid(9, 6))
    bd = param.boundary(cs)
    out = param.lscm(cs, bd, np.array([True]))
    from aether3d.uv.metrics import stretch

    l2, linf = stretch(cs, out)
    # Exact up to the 1e-10 regularization of the LSCM system.
    assert abs(l2[0] - 1.0) < 1e-7 and abs(linf[0] - 1.0) < 1e-7


def test_slim_flattens_a_developable_surface_isometrically():
    cs = _chartset(grid(20, 10, z=lambda x, y: 0.25 * np.sin(2.5 * x)))
    tut = param.tutte(cs, np.array([True]))
    assert (param.uv_det(tut, cs.tri) > 0).all()  # Tutte: injective
    out, energy = param.slim(cs, tut, np.array([True]), iterations=100, tolerance=1e-12)
    assert (param.uv_det(out, cs.tri) > 0).all()  # never inverted on the way
    assert energy[0] < 4.0 + 1e-8  # 4 is an isometry
    # Edge lengths are kept: the quads of this surface are planar, so it is developable.
    e3 = np.linalg.norm(cs.points[cs.tri[:, 1]] - cs.points[cs.tri[:, 0]], axis=1)
    e2 = np.linalg.norm(out[cs.tri[:, 1]] - out[cs.tri[:, 0]], axis=1)
    assert np.allclose(e2, e3, rtol=1e-4)


def test_svd2_reconstructs_its_matrix():
    j = np.random.default_rng(4).normal(size=(500, 2, 2))
    u, s1, s2, v = param.svd2(j)
    back = np.einsum("fij,fj,fkj->fik", u, np.stack([s1, s2], 1), v)
    assert np.allclose(back, j, atol=1e-12)
    assert np.all(s1 >= np.abs(s2) - 1e-12)
    assert np.allclose(np.sign(s2), np.sign(np.linalg.det(j)))


# --------------------------------------------------------------------------- #
# Unwrap
# --------------------------------------------------------------------------- #


def test_cube_unwraps_into_six_undistorted_islands():
    result = uv.unwrap(create_cube(1.0), resolution=256, padding=4)
    rep = uv.uv_report(result.mesh, resolution=256)
    clean(rep)
    assert result.report["charts"] == rep["islands"] == 6
    assert rep["stretch_l2"]["max"] == pytest.approx(1.0, abs=1e-9)
    assert rep["min_gap_texels"] >= 4
    assert rep["texel_density"]["spread"] < 1e-9  # one density for every island
    # Geometry, normals and groups are untouched.
    cube = create_cube(1.0)
    assert result.mesh.triangle_count == cube.triangle_count
    assert np.array_equal(
        result.mesh.positions[result.mesh.triangles], cube.positions[cube.triangles]
    )
    assert np.array_equal(result.mesh.normals[result.mesh.triangles], cube.normals[cube.triangles])


def test_sphere_unwrap_has_bounded_stretch_and_no_flips():
    result = uv.unwrap(create_sphere(1.0, 32, 16), resolution=512, max_stretch=1.1)
    rep = uv.uv_report(result.mesh, resolution=512)
    clean(rep)
    assert rep["stretch_l2"]["max"] <= 1.1
    assert rep["islands"] >= 2  # a closed surface must be cut


def test_torus_is_cut_into_disks():
    result = uv.unwrap(torus(), resolution=512)
    rep = uv.uv_report(result.mesh, resolution=512)
    clean(rep)
    topo = Topology.from_mesh(result.mesh)
    labels, count = uv.islands.island_labels(topo) if hasattr(uv, "islands") else (None, 0)
    from aether3d.uv.islands import island_labels

    labels, count = island_labels(topo)
    for i in range(count):
        tri = topo.triangles[labels == i]
        v = np.unique(tri).size
        e = np.unique(np.sort(np.stack([tri, tri[:, [1, 2, 0]]], -1).reshape(-1, 2), 1), axis=0)
        assert v - len(e) + len(tri) == 1  # Euler characteristic of a disk


def test_cylinder_side_unwraps_without_stretch():
    result = uv.unwrap(open_cylinder(), resolution=512, split_by=())
    rep = uv.uv_report(result.mesh, resolution=512)
    clean(rep)
    assert rep["stretch_l2"]["max"] < 1.0 + 2e-3


def test_explicit_seams_cut_the_map():
    plane = grid(9, 5)
    nx = 9
    seam = [[j * nx + 4, (j + 1) * nx + 4] for j in range(4)]  # a vertical line
    result = uv.unwrap(plane, seams=seam, resolution=256)
    assert result.report["charts"] == 2
    with pytest.raises(TexturingError) as exc:
        uv.unwrap(plane, seams=[[0, 40]])
    assert exc.value.code == "INVALID_SEAM"


def test_face_groups_can_split_charts():
    plane = grid(9, 5)
    groups = (plane.positions[plane.triangles].mean(axis=1)[:, 0] > 1.0).astype(np.uint16)
    plane = plane.with_groups(groups)
    assert uv.unwrap(plane, resolution=256).report["charts"] == 1
    split = uv.unwrap(plane, resolution=256, split_by=["groups"])
    assert split.report["charts"] == 2
    # Each island holds one group.
    for i in range(2):
        assert np.unique(groups[split.face_chart == i]).size == 1


def test_partial_unwrap_keeps_other_islands_shapes():
    base = uv.unwrap(create_cube(1.0), resolution=256)
    mesh = base.mesh
    faces = np.arange(4)
    result = uv.unwrap(mesh, faces=faces, resolution=256)
    rep = uv.uv_report(result.mesh, resolution=256)
    clean(rep)
    # The untouched triangles keep their UV shapes up to a similarity (repacked).
    old = mesh.uvs[mesh.triangles[4:]]
    new = result.mesh.uvs[result.mesh.triangles[4:]]
    d_old = np.linalg.norm(old[:, 1] - old[:, 0], axis=1)
    d_new = np.linalg.norm(new[:, 1] - new[:, 0], axis=1)
    ratio = d_new / d_old
    assert np.allclose(ratio, ratio[0], rtol=1e-4)


def test_unwrap_is_deterministic():
    a = uv.unwrap(create_sphere(1.0, 24, 12), resolution=256)
    b = uv.unwrap(create_sphere(1.0, 24, 12), resolution=256)
    assert a.mesh == b.mesh


def test_conformal_method_preserves_angles_on_a_plane():
    result = uv.unwrap(grid(7, 5), method="conformal", resolution=128)
    rep = uv.uv_report(result.mesh, resolution=128)
    assert rep["stretch_l2"]["max"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"angle_limit": 0}, "INVALID_ARGUMENT"),
        ({"angle_limit": float("nan")}, "INVALID_ARGUMENT"),
        ({"split_by": ["bogus"]}, "INVALID_ARGUMENT"),
        ({"method": "abf"}, "INVALID_ARGUMENT"),
        ({"max_stretch": 0.5}, "INVALID_ARGUMENT"),
        ({"resolution": 0}, "INVALID_ARGUMENT"),
        ({"resolution": True}, "INVALID_ARGUMENT"),
        ({"padding": -1}, "INVALID_ARGUMENT"),
        ({"texels_per_unit": -2}, "INVALID_ARGUMENT"),
        ({"faces": [10_000]}, "INVALID_ARGUMENT"),
        ({"iterations": 1.5}, "INVALID_ARGUMENT"),
    ],
)
def test_invalid_unwrap_arguments(kwargs, code):
    with pytest.raises(TexturingError) as exc:
        uv.unwrap(create_cube(1.0), **kwargs)
    assert exc.value.code == code


def test_atlas_full_with_a_fixed_density():
    with pytest.raises(TexturingError) as exc:
        uv.unwrap(create_cube(1.0), resolution=64, texels_per_unit=500)
    assert exc.value.code == "ATLAS_FULL"


def test_fixed_density_is_honored():
    result = uv.unwrap(create_cube(1.0), resolution=512, texels_per_unit=100, padding=2)
    rep = uv.uv_report(result.mesh, resolution=512)
    assert rep["texel_density"]["mean"] == pytest.approx(100.0, rel=1e-6)


def test_cancellation():
    with pytest.raises(TexturingError) as exc:
        uv.unwrap(create_sphere(1.0, 16, 8), should_stop=lambda: True)
    assert exc.value.code == "CANCELLED"


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #


def test_planar_projection_of_a_plane_is_exact():
    plane = create_plane(2.0)
    result = uv.project(plane, "planar", axis="z", pack=False)
    p = plane.positions[plane.triangles][:, :, :2]
    q = result.mesh.uvs[result.mesh.triangles]
    lo, span = p.reshape(-1, 2).min(0), np.ptp(p.reshape(-1, 2), 0).max()
    assert np.allclose(q, (p - lo) / span, atol=1e-6)


def test_box_projection_of_a_cube():
    result = uv.project(create_cube(1.0), "box", resolution=256)
    rep = uv.uv_report(result.mesh, resolution=256)
    clean(rep)
    assert rep["islands"] == 6 and rep["stretch_l2"]["max"] == pytest.approx(1.0, abs=1e-9)


def test_cylindrical_projection_of_a_cylinder_is_a_rectangle():
    result = uv.project(open_cylinder(), "cylindrical", axis="z", resolution=512)
    rep = uv.uv_report(result.mesh, resolution=512)
    clean(rep)
    assert rep["islands"] == 1
    # Chord-vs-arc: a 48-gon's sides are within 0.1% of the arc they replace.
    assert rep["stretch_l2"]["max"] < 1.002


def test_spherical_projection_reports_its_poles():
    result = uv.project(create_sphere(1.0, 32, 16), "spherical", axis="z", resolution=512)
    rep = uv.uv_report(result.mesh, resolution=512)
    clean(rep)
    assert [d.code for d in result.diagnostics] == ["UV_SINGULARITY"]


# --------------------------------------------------------------------------- #
# Island tools
# --------------------------------------------------------------------------- #


def test_transform_islands_exactly():
    mesh = uv.unwrap(create_cube(1.0), resolution=256).mesh
    moved = uv.transform_islands(mesh, [0], translate=(0.25, 0.0), pivot=(0.0, 0.0)).mesh
    labels = moved.uvs[moved.triangles] - mesh.uvs[mesh.triangles]
    from aether3d.uv.islands import island_labels

    lab, _ = island_labels(Topology.from_mesh(mesh))
    assert np.allclose(labels[lab == 0], [0.25, 0.0], atol=1e-7)
    assert np.allclose(labels[lab != 0], 0.0)
    rotated = uv.transform_islands(mesh, [1], rotate=90).mesh
    a = mesh.uvs[mesh.triangles][lab == 1].reshape(-1, 2)
    b = rotated.uvs[rotated.triangles][lab == 1].reshape(-1, 2)
    c = 0.5 * (a.min(0) + a.max(0))
    assert np.allclose(b - c, (a - c) @ np.array([[0, 1], [-1, 0]]), atol=1e-6)


def test_mirrored_islands_are_reported_not_flipped():
    mesh = uv.unwrap(create_cube(1.0), resolution=256).mesh
    mirrored = uv.transform_islands(mesh, [2], mirror="u").mesh
    rep = uv.uv_report(mirrored, resolution=256)
    assert rep["mirrored_islands"] == 1
    assert rep["flipped_triangles"] == 0


def test_stacked_and_overlapping_islands():
    mesh = uv.unwrap(create_cube(1.0), resolution=256, rotate=False).mesh
    rep0 = uv.uv_report(mesh, resolution=256)
    assert rep0["stacked_pair_count"] == 0
    # Move island 1 exactly onto island 0 (same size: a stacked copy).
    from aether3d.uv.islands import island_labels

    lab, _ = island_labels(Topology.from_mesh(mesh))
    a = mesh.uvs[mesh.triangles][lab == 0].reshape(-1, 2)
    b = mesh.uvs[mesh.triangles][lab == 1].reshape(-1, 2)
    shift = a.min(0) - b.min(0)
    stacked = uv.transform_islands(mesh, [1], translate=tuple(shift)).mesh
    rep = uv.uv_report(stacked, resolution=256)
    assert rep["stacked_pairs"] == [[0, 1]] and rep["overlapping_pair_count"] == 0
    assert rep["overlapping_triangles"] == 0 and rep["stacked_triangles"] > 0
    half = uv.transform_islands(mesh, [1], translate=tuple(shift + 0.01), scale=1.3).mesh
    rep = uv.uv_report(half, resolution=256)
    assert [0, 1] in rep["overlapping_pairs"]
    assert rep["overlapping_triangles"] > 0 and rep["stacked_triangles"] == 0


def test_normalize_density_and_repack():
    mesh = uv.unwrap(create_sphere(1.0, 24, 12), resolution=256).mesh
    dens = uv.set_texel_density(mesh, 40.0, resolution=256)
    rep = uv.uv_report(dens.mesh, resolution=256)
    assert rep["texel_density"]["mean"] == pytest.approx(40.0, rel=1e-6)
    packed = uv.pack_islands(dens.mesh, resolution=256, padding=3)
    rep = uv.uv_report(packed.mesh, resolution=256)
    clean(rep)
    assert rep["min_gap_texels"] >= 3
    norm = uv.normalize_islands(uv.transform_islands(mesh, scale=3.0).mesh)
    b = norm.mesh.uvs
    assert b.min() == pytest.approx(0.0, abs=1e-7) and b.max() == pytest.approx(1.0, abs=1e-6)


def test_stitch_joins_two_halves_of_a_plane():
    plane = grid(9, 5)
    nx = 9
    seam = [[j * nx + 4, (j + 1) * nx + 4] for j in range(4)]
    cut = uv.unwrap(plane, seams=seam, resolution=256).mesh
    rep = uv.uv_report(cut, resolution=256)
    assert rep["islands"] == 2
    # The seam edges, in the cut mesh's vertex numbering (positions are kept).
    pos = cut.positions
    target = plane.positions
    pairs = []
    for a, b in seam:
        ia = int(np.flatnonzero(np.all(pos == target[a], axis=1))[0])
        ib = int(np.flatnonzero(np.all(pos == target[b], axis=1))[0])
        pairs.append([ia, ib])
    joined = uv.stitch(cut, pairs).mesh
    rep = uv.uv_report(joined, resolution=256)
    assert rep["islands"] == 1 and rep["flipped_triangles"] == 0
    assert rep["overlapping_triangles"] == 0


# --------------------------------------------------------------------------- #
# Editing sessions and scene objects
# --------------------------------------------------------------------------- #


def test_editable_mesh_unwrap_keeps_polygons_whole_and_undoes():
    from aether3d.editing import EditableMesh

    m = EditableMesh.from_mesh(create_cube(1.0))
    before = m.to_mesh()
    result = m.unwrap(resolution=256)
    assert result.data["faces"] == 6
    mesh = m.to_mesh()
    rep = uv.uv_report(mesh, resolution=256)
    clean(rep)
    assert rep["islands"] == 6
    m.undo()
    assert m.to_mesh() == before
    boxed = m.project_uvs("box", resolution=256)
    assert boxed.data["report"]["islands"] == 6


def test_editable_mesh_unwrap_with_seams_and_partial_selection():
    from aether3d.editing import EditableMesh

    m = EditableMesh.from_mesh(grid(5, 3))
    edges = [e for e in m.edge_ids() if np.allclose(m.positions(m.edge_vertices(e))[:, 0], 1.0)]
    assert len(edges) == 2
    result = m.unwrap(seams=edges, resolution=128)
    assert result.data["report"]["charts"] == 2
    # A partial unwrap: only the faces right of x = 1 get a new map.
    right = [f for f in m.face_ids() if m.positions(m.face_vertices(f))[:, 0].min() >= 1.0 - 1e-9]
    before = m.to_mesh()
    m.unwrap(faces=right, resolution=128)
    assert uv.uv_report(m.to_mesh(), resolution=128)["overlapping_triangles"] == 0
    assert m.to_mesh().triangle_count == before.triangle_count


def test_scene_object_uv_steps_are_undoable():
    from aether3d import Scene

    scene = Scene()
    box = scene.add_cube("Box", size=1)
    original = box.mesh
    report = box.unwrap(resolution=256)
    assert report["charts"] == 6
    assert scene.history.entries[-1].label == "Unwrap UVs of 'Box'"
    box.edit_uvs("transform", islands=[0], rotate=90)
    box.project_uvs("box", resolution=256)
    assert box.uv_report(resolution=256)["islands"] == 6
    scene.history.undo(3)
    assert box.mesh is original
    with pytest.raises(TexturingError):
        box.edit_uvs("bogus")


def test_layout_images():
    mesh = uv.unwrap(create_sphere(1.0, 24, 12), resolution=256).mesh
    for mode in ("islands", "density", "stretch", "overlaps"):
        image = uv.layout_image(mesh, 128, mode=mode)
        assert image.shape == (128, 128, 3) and image.dtype == np.uint8
    clean = uv.layout_image(mesh, 128, mode="overlaps")
    red = (clean[..., 0] > 200) & (clean[..., 1] < 80)
    assert not red.any()
    uvs = mesh.uvs.copy()
    uvs[mesh.triangles[5]] = uvs[mesh.triangles[50]] + 0.005
    bad = Mesh(mesh.positions, mesh.triangles, normals=mesh.normals, uvs=uvs)
    image = uv.layout_image(bad, 128, mode="overlaps")
    assert ((image[..., 0] > 200) & (image[..., 1] < 80)).any()
    tex = np.zeros((8, 8, 4))
    tex[..., 1] = 1.0
    tex[..., 3] = 1.0
    shown = uv.layout_image(mesh, 64, mode="texture", texture=tex, edges=False)
    assert (shown[..., 1] == 255).all()
    for kwargs in ({"size": 8}, {"mode": "rainbow"}, {"mode": "texture"}, {"texture": tex}):
        with pytest.raises(TexturingError):
            uv.layout_image(mesh, **{"size": 64, **kwargs})
