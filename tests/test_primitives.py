from collections import Counter

import numpy as np
import pytest

from aether3d import ValidationError, create_cube, create_plane, create_sphere


def face_normals(mesh):
    p = mesh.positions.astype(np.float64)
    a, b, c = (p[mesh.triangles[:, i]] for i in range(3))
    return np.cross(b - a, c - a)


def assert_consistent(mesh):
    """Unit normals, non-degenerate triangles, winding agrees with vertex normals."""
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-6)
    fn = face_normals(mesh)
    assert np.all(np.linalg.norm(fn, axis=1) > 1e-9), "degenerate triangle"
    vertex_normal = mesh.normals[mesh.triangles].astype(np.float64).mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", fn, vertex_normal) > 0), "triangle wound inwards"
    assert np.all((mesh.uvs >= 0) & (mesh.uvs <= 1))


def assert_closed(mesh):
    """Every edge (after welding equal positions) is shared by exactly two triangles."""
    keys = [tuple(np.round(p, 5)) for p in mesh.positions]
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    welded = [[ids[keys[v]] for v in tri] for tri in mesh.triangles]
    edges = Counter(tuple(sorted((t[i], t[(i + 1) % 3]))) for t in welded for i in range(3))
    assert set(edges.values()) == {2}


def test_cube():
    mesh = create_cube(2)
    assert (mesh.vertex_count, mesh.triangle_count) == (24, 12)
    assert mesh.bounds() == ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    assert_consistent(mesh)
    assert_closed(mesh)
    # Face normals point away from the center.
    centers = mesh.positions[mesh.triangles].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", face_normals(mesh), centers) > 0)


def test_cube_default_size():
    assert create_cube().bounds() == ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


@pytest.mark.parametrize("segments, rings", [(32, 16), (3, 2), (7, 5)])
def test_sphere(segments, rings):
    mesh = create_sphere(radius=1.5, segments=segments, rings=rings)
    assert mesh.vertex_count == (segments + 1) * (rings + 1)
    assert mesh.triangle_count == 2 * segments * (rings - 1)
    np.testing.assert_allclose(np.linalg.norm(mesh.positions, axis=1), 1.5, rtol=1e-6)
    np.testing.assert_allclose(mesh.positions / 1.5, mesh.normals, atol=1e-6)
    assert_consistent(mesh)
    assert_closed(mesh)


def test_sphere_poles_on_z_axis_and_area():
    mesh = create_sphere(radius=2, segments=64, rings=32)
    lo, hi = mesh.bounds()
    assert lo[2] == -2.0 and hi[2] == 2.0
    area = np.linalg.norm(face_normals(mesh), axis=1).sum() / 2
    assert area == pytest.approx(4 * np.pi * 4, rel=0.01)


def test_plane():
    mesh = create_plane(4)
    assert (mesh.vertex_count, mesh.triangle_count) == (4, 2)
    assert mesh.bounds() == ((-2.0, -2.0, 0.0), (2.0, 2.0, 0.0))
    assert_consistent(mesh)
    assert np.all(face_normals(mesh)[:, 2] > 0)


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (create_cube, {"size": 0}),
        (create_cube, {"size": -1}),
        (create_cube, {"size": "2"}),
        (create_plane, {"size": float("inf")}),
        (create_sphere, {"radius": 0}),
        (create_sphere, {"segments": 2}),
        (create_sphere, {"rings": 1}),
        (create_sphere, {"segments": 8.5}),
    ],
)
def test_invalid_parameters(factory, kwargs):
    with pytest.raises(ValidationError):
        factory(**kwargs)
