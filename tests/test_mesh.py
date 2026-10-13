import numpy as np
import pytest

from aether3d import Mesh, ValidationError

TRIANGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]


def test_basic_mesh():
    mesh = Mesh(TRIANGLE, [(0, 1, 2)])
    assert mesh.vertex_count == 3
    assert mesh.triangle_count == 1
    assert mesh.positions.dtype == np.float32
    assert mesh.triangles.dtype == np.uint32
    assert mesh.normals is None and mesh.uvs is None
    assert mesh.bounds() == ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))


def test_flat_inputs_are_reshaped():
    mesh = Mesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2], normals=[0, 0, 1] * 3, uvs=[0, 0] * 3)
    assert mesh.positions.shape == (3, 3)
    assert mesh.triangles.shape == (1, 3)
    assert mesh.normals.shape == (3, 3)
    assert mesh.uvs.shape == (3, 2)


def test_arrays_are_read_only_copies():
    source = np.array(TRIANGLE, dtype=np.float32)
    mesh = Mesh(source, [(0, 1, 2)])
    source[0, 0] = 99
    assert mesh.positions[0, 0] == 0
    with pytest.raises(ValueError):
        mesh.positions[0, 0] = 5


def test_empty_mesh():
    mesh = Mesh([], [])
    assert mesh.vertex_count == 0 and mesh.triangle_count == 0
    assert mesh.bounds() == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"positions": [(0, 0)], "triangles": []}, "positions"),
        ({"positions": TRIANGLE, "triangles": [(0, 1, 3)]}, "indices"),
        ({"positions": TRIANGLE, "triangles": [(0, -1, 2)]}, "indices"),
        ({"positions": TRIANGLE, "triangles": [(0.0, 1.0, 2.0)]}, "integer"),
        ({"positions": TRIANGLE, "triangles": [(0, 1)]}, "triangles"),
        ({"positions": [(0, 0, float("inf"))] * 3, "triangles": []}, "finite"),
        ({"positions": TRIANGLE, "triangles": [], "normals": [(0, 0, 1)]}, "normals"),
        ({"positions": TRIANGLE, "triangles": [], "uvs": [(0, 0, 0)] * 3}, "uvs"),
        ({"positions": "abc", "triangles": []}, "positions"),
    ],
)
def test_invalid_meshes_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        Mesh(**kwargs)


def test_equality():
    a = Mesh(TRIANGLE, [(0, 1, 2)])
    assert a == Mesh(TRIANGLE, [(0, 1, 2)])
    assert a != Mesh(TRIANGLE, [(0, 2, 1)])
    assert a != Mesh(TRIANGLE, [(0, 1, 2)], normals=[(0, 0, 1)] * 3)
