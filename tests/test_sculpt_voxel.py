"""Voxel remeshing: the generated marching-cubes table, exact narrow-band distances, the
union of overlapping parts, cavities, shells, refusals before any change, undo."""

import math

import numpy as np
import pytest

from aether3d import Mesh, SculptError, ValidationError, create_cube, create_plane, create_sphere
from aether3d.sculpt import SculptMesh, check_mesh
from aether3d.sculpt import voxel as V
from aether3d.sculpt.validate import weld
from aether3d.spatial.surface import TriangleSurface


def closed_ok(positions, triangles) -> dict:
    s = check_mesh(positions, triangles, welded=True).summary
    assert s["closed"] and s["edge_manifold"] and s["vertex_manifold"], s
    assert s["consistently_oriented"] and s["degenerate_triangle_count"] == 0, s
    return s


def grid_surface(values: np.ndarray):
    """Marching cubes of a dense grid ``values[x, y, z]`` (spacing 1)."""
    n = np.array(values.shape)
    keys = np.arange(values.size)
    flat = values.transpose(2, 1, 0).ravel()  # key = (z * ny + y) * nx + x
    return V.marching_cubes(keys, flat, n, np.zeros(3), 1.0)


def test_every_case_of_the_table_closes_and_faces_out():
    for case in range(1, 255):
        v = np.ones((4, 4, 4))
        for corner in range(8):
            if case >> corner & 1:
                v[1 + (corner & 1), 1 + (corner >> 1 & 1), 1 + (corner >> 2 & 1)] = -1.0
        s = closed_ok(*grid_surface(v))
        assert s["volume"] > 0


def test_random_fields_give_closed_manifold_surfaces():
    # Ambiguous faces everywhere: neighbouring cells must agree on every shared face.
    rng = np.random.default_rng(7)
    for _ in range(60):
        v = rng.normal(size=(8, 8, 8))
        v[[0, -1], :, :] = v[:, [0, -1], :] = v[:, :, [0, -1]] = 1.0
        positions, triangles = grid_surface(v)
        assert closed_ok(positions, triangles)["volume"] > 0


def test_the_band_width_does_not_change_the_surface(monkeypatch):
    mesh = create_sphere(1.0, 16, 8)
    p, t = weld(mesh.positions, mesh.triangles)
    p = p.astype(np.float64)
    outs = []
    for band in (V.BAND, 2.5):
        monkeypatch.setattr(V, "BAND", band)
        outs.append(V.voxel_surface(p, t, 0.07)[:2])
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])


def test_band_distances_are_exact():
    mesh = create_sphere(1.0, 12, 6)
    p, t = weld(mesh.positions, mesh.triangles)
    p = p.astype(np.float64)
    h = 0.1
    origin, dims = V.grid_for(p, h, 0.0)
    keys, values, nearest = V.band_values(p, t, h, origin, dims, shell=None, max_voxels=10**7)
    q = (
        origin
        + np.stack(
            [keys % dims[0], (keys // dims[0]) % dims[1], keys // (dims[0] * dims[1])], axis=1
        )
        * h
    )
    reference = TriangleSurface(p, t).nearest(q)
    assert np.array_equal(np.abs(values), reference.distance)
    # Every grid point within the band is there, and the sign is inside/outside.
    radius = np.linalg.norm(q, axis=1)
    assert (values[radius < 0.8] < 0).all() and (values[radius > 1.0] > 0).all()
    all_points = (
        origin
        + np.stack(np.meshgrid(*(np.arange(d) for d in dims), indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        * h
    )
    within = TriangleSurface(p, t).nearest(all_points).distance <= V.BAND * h
    assert within.sum() == len(keys)


def test_sphere_remesh_is_closed_and_follows_the_surface():
    source = create_sphere(1.0, 48, 24)
    sm = SculptMesh.from_mesh(source)
    r = sm.voxel_remesh(0.05)
    out = sm.to_mesh()
    s = closed_ok(out.positions, out.triangles)
    assert s["euler_characteristic"] == 2 and r.data["output"]["components"] == 1
    assert r.data["output"]["volume"] == pytest.approx(r.data["input"]["volume"], rel=0.005)
    p, t = weld(source.positions, source.triangles)
    d = TriangleSurface(p, t).nearest(out.positions).distance
    assert d.max() < 0.1 * 0.05  # vertices on the source surface within a tenth of a voxel
    assert out.uvs is None and out.normals is not None
    assert {d.code for d in r.diagnostics} == {"VOXEL_UVS_DROPPED", "VOXEL_NORMALS_RECOMPUTED"}
    report = check_mesh(out.positions, out.triangles, self_intersections=True)
    assert report.summary["self_intersecting_pairs"] == 0


def two_spheres(offset: float, *, inner: float | None = None) -> Mesh:
    a = create_sphere(1.0, 48, 24)
    pa, ta = weld(a.positions, a.triangles)
    if inner is None:
        pb, tb = pa + np.float32([offset, 0, 0]), ta  # a second sphere, overlapping
    else:
        pb, tb = pa * np.float32(inner), ta[:, ::-1]  # an inward-facing sphere inside
    return Mesh(np.concatenate([pa, pb]), np.concatenate([ta, tb + len(pa)]))


def test_overlapping_parts_fuse_into_their_union():
    # (Exactly one radius apart, rings of the two spheres would share welded vertices:
    # non-manifold, refused.)
    sm = SculptMesh.from_mesh(two_spheres(1.05))
    r = sm.voxel_remesh(0.04)
    out = sm.to_mesh()
    s = closed_ok(out.positions, out.triangles)
    assert r.data["input"]["components"] == 2 and r.data["output"]["components"] == 1
    assert s["euler_characteristic"] == 2
    # The union of two unit spheres 1.05 apart: twice a sphere minus the lens
    # pi (4R + d)(2R - d)^2 / 12 (the faceted input is 0.7% smaller than the spheres).
    union = 2 * 4 / 3 * math.pi - math.pi * (4 + 1.05) * (2 - 1.05) ** 2 / 12
    assert s["volume"] == pytest.approx(union, rel=0.015)
    # No buried surface survives: every vertex is on the outer boundary.
    p = out.positions.astype(np.float64)
    nearest_centre = np.minimum(np.linalg.norm(p, axis=1), np.linalg.norm(p - [1.05, 0, 0], axis=1))
    assert np.abs(nearest_centre - 1.0).max() < 0.02


def test_inward_parts_are_cavities():
    sm = SculptMesh.from_mesh(two_spheres(0.0, inner=0.5))
    r = sm.voxel_remesh(0.04)
    out = sm.to_mesh()
    s = closed_ok(out.positions, out.triangles)
    assert r.data["output"]["components"] == 2
    assert s["volume"] == pytest.approx(4.15897 * (1 - 0.125), rel=0.01)


def test_shells_close_open_meshes():
    sm = SculptMesh.from_mesh(create_plane(2.0))
    r = sm.voxel_remesh(0.05, mode="shell", thickness=0.2)
    out = sm.to_mesh()
    s = closed_ok(out.positions, out.triangles)
    assert s["euler_characteristic"] == 2 and r.data["input"]["closed"] is False
    p = create_plane(2.0)
    d = TriangleSurface(p.positions, p.triangles).nearest(out.positions).distance
    assert np.abs(d - 0.1).max() < 0.005


def test_refusals_change_nothing():
    plane = SculptMesh.from_mesh(create_plane(2.0))
    with pytest.raises(SculptError, match="boundary edges") as err:
        plane.voxel_remesh(0.1)
    assert err.value.code == "VOXEL_NOT_WATERTIGHT"
    cube = create_cube(2)
    flipped = SculptMesh.from_mesh(Mesh(cube.positions, cube.triangles[:, ::-1]))
    with pytest.raises(SculptError) as err:
        flipped.voxel_remesh(0.1)
    assert err.value.code == "VOXEL_INSIDE_OUT"
    tiny = SculptMesh.from_mesh(create_sphere(0.01, 8, 4))
    with pytest.raises(SculptError) as err:
        tiny.voxel_remesh(0.5)
    assert err.value.code == "VOXEL_EMPTY"
    big = SculptMesh.from_mesh(create_sphere(1.0, 16, 8))
    with pytest.raises(SculptError) as err:
        big.voxel_remesh(0.01, max_voxels=10_000)
    assert err.value.code == "VOXEL_GRID_TOO_LARGE"
    for kwargs in (
        {"voxel_size": 0},
        {"voxel_size": float("nan")},
        {"voxel_size": True},
        {"voxel_size": 0.1, "mode": "hollow"},
        {"voxel_size": 0.1, "mode": "shell"},
        {"voxel_size": 0.1, "mode": "shell", "thickness": 0.1},
        {"voxel_size": 0.1, "thickness": 0.5},
        {"voxel_size": 0.1, "max_voxels": 10},
    ):
        with pytest.raises(ValidationError):
            big.voxel_remesh(**kwargs)
    for sm in (plane, flipped, tiny, big):
        assert not sm.has_changes and sm.to_mesh() is sm._source


def test_undo_redo_and_determinism():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 24, 12))
    sm.stroke({"brush": {"kind": "draw", "radius": 0.5, "strength": 0.2}, "path": [[0, 0, 1]]})
    before = sm.to_mesh()
    sm.voxel_remesh(0.08)
    after = sm.to_mesh()
    sm.undo()
    assert sm.to_mesh() == before
    sm.redo()
    assert sm.to_mesh() == after
    a, b = (SculptMesh.from_mesh(before) for _ in range(2))
    a.voxel_remesh(0.08)
    b.voxel_remesh(0.08)
    assert a.to_mesh() == b.to_mesh()


def test_groups_and_mask_follow_the_nearest_source_point():
    source = create_sphere(1.0, 32, 16)
    groups = (source.positions[source.triangles].mean(axis=1)[:, 2] > 0).astype(np.int64)
    sm = SculptMesh.from_mesh(Mesh(source.positions, source.triangles, groups=groups))
    sm.set_mask(1.0, box=((-2, -2, 0.5), (2, 2, 2)))
    sm.voxel_remesh(0.06)
    out = sm.to_mesh()
    centre_z = out.positions[out.triangles].mean(axis=1)[:, 2]
    assert (out.groups[centre_z > 0.05] == 1).all() and (out.groups[centre_z < -0.05] == 0).all()
    mask = sm.mask()
    z = sm.positions()[:, 2]
    assert (mask[z > 0.75] == 1).all() and (mask[z < 0.3] == 0).all()
    assert 0 < mask.mean() < 1


def test_voxel_size_is_in_world_units():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 24, 12))
    sm.world_matrix = np.diag([2.0, 2.0, 2.0, 1.0])
    r = sm.voxel_remesh(0.1)
    # Metrics are in world units: the sphere of radius 2 the object shows.
    assert r.data["input"]["volume"] == pytest.approx(8 * 4.0, rel=0.05)
    assert r.data["output"]["volume"] == pytest.approx(r.data["input"]["volume"], rel=0.01)
    # The grid spans the world-space sphere (radius 2) at 0.1: about 40 cells across.
    assert 40 <= r.data["grid"][0] <= 50
    local = sm.positions()
    assert np.abs(np.linalg.norm(local, axis=1) - 1.0).max() < 0.02
