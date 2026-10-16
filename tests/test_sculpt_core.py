"""The sculpt mesh and its brushes: exact conversion, analytic brush displacements,
masks, symmetry, transforms, projections, undo, atomicity and sessions."""

import math

import numpy as np
import pytest

from aether3d import EditError, Mesh, Scene, SculptError, ValidationError, create_sphere
from aether3d.editing import EditableMesh
from aether3d.sculpt import Brush, SculptMesh, Stroke, check_mesh
from aether3d.sculpt.brushes import FALLOFFS, kelvinlet
from aether3d.sculpt.stroke import dab_positions


def grid_mesh(n=20, size=2.0) -> Mesh:
    em = EditableMesh()
    em.add_grid(n, n, size)
    return em.to_mesh()


def stroke(kind, path, **brush):
    brush.setdefault("radius", 0.5)
    return {"brush": {"kind": kind, **brush}, "path": path}


def state(sm: SculptMesh) -> tuple:
    n = sm._nv
    return (
        sm._pos[:n].tobytes(),
        sm._mask[:n].tobytes(),
        sm._dirty[:n].tobytes(),
        sm._vsum[:n].tobytes(),
        sm._cross[: sm._nt].tobytes(),
    )


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def test_conversion_welds_seams_and_is_lossless():
    mesh = create_sphere(1.0, 32, 16)
    sm = SculptMesh.from_mesh(mesh)
    # The UV seam column and the poles are welded: 32 x 15 rings + 2 poles.
    assert sm.vertex_count == 32 * 15 + 2 and sm.triangle_count == mesh.triangle_count
    assert sm.to_mesh() is mesh  # nothing changed: the very same object
    assert sm._consistency_errors() == []
    sm.stroke(stroke("draw", [[0, 0, 1]], strength=0.2))
    out = sm.to_mesh()
    assert out is not mesh and out.triangles is mesh.triangles and out.uvs is mesh.uvs
    moved = np.any(out.positions != mesh.positions, axis=1)
    assert moved.any() and not moved.all()
    # Unmoved vertices far from the stroke keep their normals bit for bit.
    far = mesh.positions[:, 2] < 0
    assert np.array_equal(out.normals[far], mesh.normals[far])
    sm.undo()
    assert sm.to_mesh() is mesh


def test_seam_and_hard_edges_stay_as_they_were():
    from aether3d import create_cube

    cube = create_cube(2)
    sm = SculptMesh.from_mesh(cube)
    assert sm.vertex_count == 8  # welded corners; 24 render vertices stay
    sm.stroke(stroke("draw", [[1, 1, 1]], radius=0.5, strength=0.1))
    out = sm.to_mesh()
    assert out.vertex_count == 24
    # The moved corner keeps three render vertices with three different normals: the
    # cube's edges stay hard (each face's normal group is recomputed on its own).
    corner = np.flatnonzero(np.all(out.positions > 1, axis=1))
    assert len(corner) == 3
    normals = out.normals[corner]
    assert np.all(np.abs(normals @ normals.T - np.eye(3)) < 0.1)
    # The opposite corner's triangles did not change: its normals stay bit for bit.
    far = np.all(cube.positions < 0, axis=1)
    assert far.sum() == 3 and np.array_equal(out.normals[far], cube.normals[far])


# --------------------------------------------------------------------------- #
# Brushes, analytically
# --------------------------------------------------------------------------- #


def test_falloff_curves():
    x = np.linspace(0, 1, 101)
    for name in FALLOFFS:
        f = Brush("draw", 1.0, falloff=name).weights(x)
        assert f[0] == pytest.approx(1.0)
        if name != "constant":
            assert f[-1] == pytest.approx(0.0, abs=1e-12)
            assert np.all(np.diff(f) <= 1e-12)
    custom = Brush("draw", 1.0, falloff=((0, 1), (0.5, 1), (1, 0))).weights(x)
    assert custom[50] == 1 and custom[75] == pytest.approx(0.5)
    hard = Brush("draw", 1.0, falloff="linear", hardness=0.5).weights(np.array([0.25, 0.5, 0.75]))
    assert hard.tolist() == pytest.approx([1.0, 1.0, 0.5])


def test_draw_moves_by_strength_times_radius_times_falloff():
    sm = SculptMesh.from_mesh(grid_mesh(20, 2.0))
    before = sm.positions()
    r = sm.stroke(stroke("draw", [[0, 0, 0]], radius=0.5, strength=0.2, falloff="smooth"))
    after = sm.positions()
    d = np.linalg.norm(before[:, :2], axis=1)
    x = np.clip(d / 0.5, 0, 1)
    expected = 0.2 * 0.5 * (1 - (3 * x * x - 2 * x**3))
    assert np.allclose(after[:, 2], np.where(d <= 0.5, expected, 0.0), atol=1e-12)
    assert np.allclose(after[:, :2], before[:, :2])
    assert r.data["max_displacement"] == pytest.approx(0.1)
    zero = SculptMesh.from_mesh(grid_mesh())
    zero.stroke(stroke("draw", [[0, 0, 0]], strength=0.0))
    assert not zero.has_changes


def test_inflate_moves_along_vertex_normals():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 64, 32))
    before = sm.positions()
    sm.stroke(stroke("inflate", [[1, 0, 0]], radius=0.6, strength=0.1))
    d = sm.positions() - before
    moved = np.linalg.norm(d, axis=1) > 0
    radial = before[moved] / np.linalg.norm(before[moved], axis=1)[:, None]
    cos = (d[moved] * radial).sum(axis=1) / np.linalg.norm(d[moved], axis=1)
    assert cos.min() > 0.99


def test_flatten_fill_and_scrape():
    em = EditableMesh()
    em.add_grid(30, 30, 2.0)
    rng = np.random.default_rng(0)
    for v in em.vertex_ids():
        x, y, _ = em.position(v)
        em.set_positions({v: (x, y, 0.05 * rng.standard_normal())})
    mesh = em.to_mesh()
    flat = SculptMesh.from_mesh(mesh)
    flat.stroke(stroke("flatten", [[0, 0, 0]], radius=0.5, strength=1.0, falloff="constant"))
    p = flat.positions()
    inside = np.linalg.norm(p[:, :2], axis=1) < 0.4
    q = p[inside] - p[inside].mean(axis=0)
    assert np.linalg.svd(q, compute_uv=False)[-1] < 1e-9  # all on one plane
    for kind, cmp in (("fill", np.greater_equal), ("scrape", np.less_equal)):
        s = SculptMesh.from_mesh(mesh)
        z0 = s.positions()[:, 2]
        s.stroke(stroke(kind, [[0, 0, 0]], radius=0.5, strength=1.0))
        dz = s.positions()[:, 2] - z0
        assert np.all(cmp(dz, -1e-15) if kind == "fill" else cmp(dz, 1e-15))
        assert np.any(dz != 0)


def test_pinch_and_twist_stay_in_the_plane_of_a_flat_surface():
    for kind in ("pinch", "twist"):
        sm = SculptMesh.from_mesh(grid_mesh(20, 2.0))
        before = sm.positions()
        sm.stroke(stroke(kind, [[0, 0, 0]], radius=0.6, strength=0.5))
        after = sm.positions()
        assert np.allclose(after[:, 2], 0, atol=1e-15)
        r0 = np.linalg.norm(before[:, :2], axis=1)
        r1 = np.linalg.norm(after[:, :2], axis=1)
        if kind == "pinch":
            assert np.all(r1 <= r0 + 1e-12) and np.any(r1 < r0 - 1e-6)
        else:  # a rotation about the normal keeps the distance to the axis
            assert np.allclose(r1, r0, atol=1e-12) and np.any(np.abs(after - before) > 1e-6)


def test_grab_follows_the_stroke_exactly():
    sm = SculptMesh.from_mesh(grid_mesh(20, 2.0))
    before = sm.positions()
    path = [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.3]]
    sm.stroke(
        {
            "brush": {"kind": "grab", "radius": 0.5, "falloff": "linear"},
            "path": path,
            "project": "none",
            "resample": False,
        }
    )
    d = np.linalg.norm(before[:, :2], axis=1)
    w = np.clip(1 - d / 0.5, 0, 1)
    assert np.allclose(sm.positions()[:, 2], 0.3 * w, atol=1e-12)


def test_grab_and_snake_hook_pull_geometry_off_the_surface():
    # With the default projection only the first dab lands on the surface; the later
    # ones follow the path's motion, so the geometry leaves the surface.
    sm = SculptMesh.from_mesh(grid_mesh(20, 2.0))
    before = sm.positions()
    sm.stroke(
        {
            "brush": {"kind": "grab", "radius": 0.5, "falloff": "linear"},
            "path": [[0, 0, 0.2], [0, 0, 0.5]],  # starts 0.2 above the grid: snapped
            "resample": False,
        }
    )
    d = np.linalg.norm(before[:, :2], axis=1)
    w = np.clip(1 - d / 0.5, 0, 1)
    assert np.allclose(sm.positions()[:, 2], 0.3 * w, atol=1e-12)
    ball = SculptMesh.from_mesh(create_sphere(1.0, 32, 16))
    ball.stroke(
        {
            "brush": {"kind": "snake_hook", "radius": 0.3, "strength": 1.0},
            "path": [[0, 0, 1], [0, 0, 1.5]],
        }
    )
    assert ball.positions()[:, 2].max() > 1.45  # a horn half a radius long


def test_kelvinlet_is_normalized_and_divergence_free_when_incompressible():
    delta = np.array([0.0, 0.0, 0.25])
    assert np.allclose(kelvinlet(np.zeros((1, 3)), delta, 0.5, 0.3)[0], delta)
    # Poisson ratio 1/2: the regularized Stokeslet, divergence free everywhere.
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20, 3))
    h = 1e-5
    div = np.zeros(len(pts))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        div += (
            kelvinlet(pts + e, delta, 0.5, 0.4999999)[:, k]
            - kelvinlet(pts - e, delta, 0.5, 0.4999999)[:, k]
        ) / (2 * h)
    assert np.abs(div).max() < 1e-5
    sm = SculptMesh.from_mesh(create_sphere(1.0, 32, 16))
    top = int(np.argmax(sm.positions()[:, 2]))
    start = sm.positions()[top].copy()
    sm.stroke(
        {
            "brush": {"kind": "elastic", "radius": 0.5},
            "path": [start.tolist(), (start + [0, 0, 0.4]).tolist()],
            "project": "none",
            "resample": False,
        }
    )
    assert np.allclose(sm.positions()[top], start + [0, 0, 0.4])


def test_layer_never_exceeds_its_height():
    sm = SculptMesh.from_mesh(grid_mesh(20, 2.0))
    path = [[0, 0, 0]] * 1 + [[0.01 * i, 0, 0] for i in range(10)]
    sm.stroke(
        {
            "brush": {"kind": "layer", "radius": 0.5, "strength": 0.2},
            "path": path,
            "resample": False,
        }
    )
    assert sm.positions()[:, 2].max() == pytest.approx(0.1, abs=1e-12)


def noisy_sphere(seed=2) -> Mesh:
    base = create_sphere(1.0, 48, 24)
    rng = np.random.default_rng(seed)
    sm = SculptMesh.from_mesh(base)
    p = sm._pos[: sm._nv]
    p *= (1 + 0.03 * rng.standard_normal(len(p)))[:, None]
    return Mesh(p[sm._rgeo].astype(np.float32), base.triangles)


def volume(sm: SculptMesh) -> float:
    t = sm.triangles()
    p = sm._pos
    return float((p[t[:, 0]] * np.cross(p[t[:, 1]], p[t[:, 2]])).sum() / 6)


def roughness(sm: SculptMesh) -> float:
    r = np.linalg.norm(sm.positions(), axis=1)
    return float(np.std(r))


def test_smooth_methods():
    mesh = noisy_sphere()
    results = {}
    for method in ("laplacian", "taubin", "relax"):
        sm = SculptMesh.from_mesh(mesh)
        v0, r0 = volume(sm), roughness(sm)
        path = [[x, y, z] for x, y, z in create_sphere(1.0, 8, 4).positions.tolist()]
        sm.stroke(
            {
                "brush": {
                    "kind": "smooth",
                    "radius": 0.9,
                    "strength": 1.0,
                    "smooth_method": method,
                },
                "path": path,
                "resample": False,
            }
        )
        results[method] = (volume(sm) / v0 - 1, roughness(sm) / r0)
    assert results["laplacian"][1] < 0.8 and results["taubin"][1] < 0.9
    # Taubin's inflating step keeps the volume far better than plain Laplacian smoothing.
    assert abs(results["taubin"][0]) < abs(results["laplacian"][0]) / 2
    # Relax moves vertices within the tangent plane: the shape barely changes.
    assert abs(results["relax"][0]) < abs(results["laplacian"][0]) / 2


# --------------------------------------------------------------------------- #
# Masks, automasking, symmetry, transforms
# --------------------------------------------------------------------------- #


def test_masked_vertices_do_not_move():
    sm = SculptMesh.from_mesh(grid_mesh(20, 2.0))
    sm.set_mask(1.0, box=((-1, -1, -1), (0, 1, 1)))
    before = sm.positions()
    sm.stroke(stroke("draw", [[0, 0, 0]], radius=0.6, strength=0.3))
    after = sm.positions()
    left = before[:, 0] <= 0
    assert np.array_equal(after[left], before[left])
    assert np.any(after[~left, 2] > 0)
    sm.invert_mask()
    assert np.all(sm.mask()[left] == 0)
    sm.clear_mask()
    assert not sm.mask().any()
    painted = SculptMesh.from_mesh(grid_mesh())
    painted.stroke(stroke("mask", [[0, 0, 0]], radius=0.5, strength=1.0, falloff="constant"))
    assert painted.mask().max() == 1 and painted.mask().min() == 0
    assert not painted.has_changes  # masks are not geometry


def symmetric_sphere() -> Mesh:
    """A sphere whose positions *and* triangles mirror across x = 0 (a UV sphere's own
    triangulation does not: its quad diagonals all lean one way)."""
    scene = Scene()
    node = scene.add_geometry("S", "uv_sphere", radius=1.0, segments=64, rings=32)
    node.add_modifier("mirror", axes="x", bisect=True)
    return node.evaluate()


def test_symmetric_strokes_keep_a_symmetric_mesh_symmetric():
    sm = SculptMesh.from_mesh(symmetric_sphere())
    p0 = sm.positions()
    index = {tuple(np.round(q, 12)): i for i, q in enumerate(p0)}
    mirror = np.array([index[tuple(np.round(q * [-1, 1, 1], 12))] for q in p0])
    path = [[0.4, -0.5, 0.75], [0.6, -0.7, 0.2]]
    sm.stroke(stroke("draw", path, radius=0.4, strength=0.2, symmetry="x"))
    sm.stroke(stroke("pinch", [[0.05, -0.9, 0.3]], radius=0.5, strength=0.5, symmetry="x"))
    p = sm.positions()
    assert np.abs(p - p[mirror] * [-1, 1, 1]).max() < 1e-12  # up to rounding
    on_plane = p0[:, 0] == 0
    assert on_plane.any() and np.all(p[on_plane, 0] == 0.0)  # exactly on the plane
    assert np.abs(p - p0).max() > 0.01


def test_brushes_measure_in_world_space_under_non_uniform_scale():
    scene = Scene()
    obj = scene.add_object("Plate", mesh=grid_mesh(40, 2.0), scale=(4, 1, 1))
    with obj.sculpt() as s:
        s.stroke(stroke("draw", [[0, 0, 0]], radius=0.5, strength=0.2, falloff="constant"))
    local = obj.mesh.positions.astype(np.float64)
    world = local * [4, 1, 1]
    raised = local[:, 2] > 0
    # A world-space sphere of radius 0.5: |world xy| <= 0.5 exactly where raised.
    inside = np.linalg.norm(world[:, :2], axis=1) <= 0.5 + 1e-9
    assert np.array_equal(raised, inside)


def test_negative_scale_keeps_draw_outward():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 48, 24), scale=(-1, 1, 1))
    with ball.sculpt() as s:
        s.stroke(stroke("draw", [[0, 0, 1]], radius=0.4, strength=0.2))
    assert ball.mesh.positions[:, 2].max() > 1.05  # raised, not pushed in


def test_front_faces_only_and_topology_automask():
    em = EditableMesh()
    em.add_grid(10, 10, 2.0)
    em.add_grid(10, 10, 2.0, center=(0, 0, -0.05))
    em.flip_faces(em.select_faces(box=((-2, -2, -0.1), (2, 2, -0.01))))
    plates = em.to_mesh()
    front = SculptMesh.from_mesh(plates)
    front.stroke(
        {
            "brush": {"kind": "draw", "radius": 0.5, "strength": 0.1, "front_faces_only": True},
            "path": [[0, 0, 0]],
            "view": [0, 0, -1],
        }
    )
    p = front.positions()
    start = SculptMesh.from_mesh(plates).positions()
    lower = start[:, 2] < -0.02
    assert np.array_equal(p[lower], start[lower]) and np.any(p[~lower, 2] > 0)
    topo = SculptMesh.from_mesh(plates)
    topo.stroke(stroke("draw", [[0, 0, 0]], radius=0.5, strength=0.1, automask=["topology"]))
    assert np.array_equal(topo.positions()[lower], start[lower])


def test_geodesic_falloff_stays_on_its_side_of_a_fold():
    em = EditableMesh()
    em.add_grid(20, 20, 2.0)
    em.add_grid(20, 20, 2.0, center=(0, 0, 0.1))  # a second sheet just above, unconnected
    mesh = em.to_mesh()
    sm = SculptMesh.from_mesh(mesh)
    sm.stroke(stroke("draw", [[0, 0, 0]], radius=0.5, strength=0.05, geodesic=True))
    start = SculptMesh.from_mesh(mesh).positions()
    upper = start[:, 2] > 0.05
    assert np.array_equal(sm.positions()[upper], start[upper])
    assert np.any(sm.positions()[~upper, 2] > 0)


def test_projection_modes_and_spacing():
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    dabs, pres = dab_positions(pts, np.array([1.0, 0.0]), 0.25)
    assert len(dabs) == 5 and pres.tolist() == pytest.approx([1, 0.75, 0.5, 0.25, 0])
    sm = SculptMesh.from_mesh(create_sphere(1.0, 32, 16))
    r = sm.stroke(
        {
            "brush": {"kind": "draw", "radius": 0.3, "strength": 0.1},
            "path": [[0, 0, 5], [0, 5, 5]],
            "project": "ray",
            "view": [0, 0, -1],
        }
    )
    assert r.data["skipped_dabs"] > 0 and r.data["applied_dabs"] > 0
    assert "DABS_SKIPPED" in [d.code for d in r.diagnostics]
    with pytest.raises(ValidationError):
        Stroke.from_dict(
            {"brush": {"kind": "draw", "radius": 1}, "path": [[0, 0, 0]], "project": "ray"}
        )


# --------------------------------------------------------------------------- #
# History, atomicity, validation of inputs
# --------------------------------------------------------------------------- #


def test_undo_redo_and_determinism():
    mesh = create_sphere(1.0, 48, 24)
    sm = SculptMesh.from_mesh(mesh)
    start = state(sm)
    st = stroke(
        "crease", [[0.2, -0.8, 0.5], [0.5, -0.8, -0.2]], radius=0.35, strength=0.4, symmetry="x"
    )
    sm.stroke(st)
    sm.stroke(stroke("inflate", [[0, 0, 1]], radius=0.5, strength=0.1))
    done = state(sm)
    sm.undo()
    sm.undo()
    assert state(sm) == start
    sm.redo()
    sm.redo()
    assert state(sm) == done
    other = SculptMesh.from_mesh(mesh)
    other.stroke(st)
    other.stroke(stroke("inflate", [[0, 0, 1]], radius=0.5, strength=0.1))
    assert state(other) == done
    assert sm._consistency_errors() == []


def test_a_failing_stroke_changes_nothing(monkeypatch):
    sm = SculptMesh.from_mesh(create_sphere(1.0, 32, 16))
    sm.stroke(stroke("draw", [[0, 0, 1]], strength=0.1))
    before = state(sm)
    from aether3d.sculpt import stroke as stroke_module

    original = stroke_module.StrokeRunner.dab
    calls = {"n": 0}

    def failing(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("simulated failure in the middle of a stroke")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(stroke_module.StrokeRunner, "dab", failing)
    with pytest.raises(RuntimeError):
        sm.stroke(stroke("draw", [[0, 0, 1], [0, 1, 0]], radius=0.3, strength=0.2))
    assert state(sm) == before and len(sm.history) == 1


def test_invalid_brushes_and_strokes_are_refused():
    for bad in (
        {"kind": "paint", "radius": 1},
        {"kind": "draw", "radius": 0},
        {"kind": "draw", "radius": 1, "strength": -1},
        {"kind": "draw", "radius": 1, "direction": 2},
        {"kind": "draw", "radius": 1, "falloff": "wobbly"},
        {"kind": "draw", "radius": 1, "symmetry": "xx"},
        {"kind": "draw", "radius": 1, "automask": ["nose"]},
        {"kind": "draw", "radius": 1, "extra": True},
        {"kind": "smooth", "radius": 1, "strength": 2},
        {"kind": "draw", "radius": float("nan")},
    ):
        with pytest.raises(ValidationError):
            Brush.from_dict(bad)
    for path in ([], [[0, 0]], [[0, 0, float("inf")]], [{"position": [0, 0, 0], "pressure": 2}]):
        with pytest.raises(ValidationError):
            Stroke.from_dict({"brush": {"kind": "draw", "radius": 1}, "path": path})
    sm = SculptMesh.from_mesh(create_sphere(1.0, 16, 8))
    with pytest.raises(SculptError, match="TOO_MANY_DABS"):
        sm.stroke(stroke("draw", [[0, 0, 0], [1e9, 0, 0]], radius=0.1, spacing=0.01))
    assert not sm.has_changes


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def test_session_commits_one_history_step_and_refuses_procedural_objects():
    scene = Scene()
    mesh = create_sphere(1.0, 32, 16)
    obj = scene.add_object("Head", mesh=mesh)
    other = scene.add_object("Twin", mesh=mesh)
    steps = len(scene.history.entries)
    with obj.sculpt() as s:
        s.stroke(stroke("draw", [[0, 0, 1]], strength=0.2))
        s.stroke(stroke("smooth", [[0, 0, 1]], strength=0.5))
    assert len(scene.history.entries) == steps + 1
    assert obj.mesh is not mesh and other.mesh is mesh
    assert s.result.report.ok and s.result.report.summary["closed"]
    scene.history.undo()
    assert obj.mesh is mesh
    stale = obj.sculpt()
    stale.stroke(stroke("draw", [[0, 0, 1]], strength=0.2))
    obj.mesh = create_sphere(1.0, 8, 4)
    with pytest.raises(SculptError, match="STALE_SESSION"):
        stale.commit()
    proc = scene.add_procedural("Box", "box")
    with pytest.raises(SculptError, match="PROCEDURAL_OBJECT"):
        proc.sculpt()
    assert isinstance(SculptError("X", "y"), EditError)


def test_checkpoint_commits_and_source_sculpting():
    scene = Scene()
    obj = scene.add_object("Head", mesh=create_sphere(1.0, 32, 16))
    s = obj.sculpt()
    s.stroke(stroke("draw", [[0, 0, 1]], strength=0.2))
    first = s.commit(close=False)
    assert first.changed and not s.closed
    assert not s.commit(close=False).changed  # nothing new
    s.stroke(stroke("draw", [[1, 0, 0]], strength=0.2))
    second = s.commit()
    assert second.changed and second.mesh is obj.mesh and s.closed
    node = scene.add_geometry("Rock", "uv_sphere", radius=1.0)
    node.convert_source_to_mesh()
    node.add_modifier("transform", translation=(0, 0, 2))
    user = scene.add_object("RockObject", geometry=node)
    with node.sculpt_source() as src:
        src.stroke(stroke("draw", [[0, 0, 1]], strength=0.3))
    assert user.mesh.positions[:, 2].max() > 3.1  # sculpted, then moved by the modifier


def test_commit_reports_validity():
    scene = Scene()
    obj = scene.add_object("Ball", mesh=create_sphere(1.0, 32, 16))
    with obj.sculpt() as s:
        s.stroke(stroke("draw", [[0, 0, 1]], radius=0.4, strength=2.5))  # drives the cap through
    report = s.result.report
    assert "self_intersecting_pairs" in report.summary
    assert check_mesh(obj.mesh.positions, obj.mesh.triangles).summary["closed"]
    assert math.isfinite(report.summary["min_angle_degrees"])
