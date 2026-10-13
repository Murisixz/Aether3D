"""The Z-up <-> Y-up conversion is a proper rotation applied consistently."""

import numpy as np
import pytest

from aether3d import Transform, math3d
from aether3d.gltf import axes

C = axes.AXIS_MATRIX
C4 = np.eye(4)
C4[:3, :3] = C


def test_axis_matrix_is_a_proper_rotation_mapping_up_to_up():
    np.testing.assert_array_equal(C @ C.T, np.eye(3))
    assert np.linalg.det(C) == 1.0  # no handedness change -> winding is preserved
    np.testing.assert_array_equal(C @ (0, 0, 1), (0, 1, 0))  # Aether up -> glTF up
    np.testing.assert_array_equal(C @ (1, 0, 0), (1, 0, 0))  # right stays right
    np.testing.assert_array_equal(C @ (0, -1, 0), (0, 0, 1))  # Aether -Y (front) -> glTF +Z


def test_vector_functions_match_the_matrix_and_invert_exactly():
    rng = np.random.default_rng(3)
    for v in rng.normal(size=(50, 3)) * 100:
        v = tuple(v.tolist())
        g = axes.vector_to_gltf(v)
        np.testing.assert_array_equal(g, C @ v)
        assert axes.vector_from_gltf(g) == v
    arr = rng.normal(size=(20, 3)).astype(np.float32)
    np.testing.assert_array_equal(axes.vectors_to_gltf(arr), arr @ C.T.astype(np.float32))
    np.testing.assert_array_equal(axes.vectors_from_gltf(axes.vectors_to_gltf(arr)), arr)


def test_conversion_never_emits_negative_zero():
    assert str(axes.vector_to_gltf((0.0, 0.0, 0.0))) == "(0.0, 0.0, 0.0)"
    assert str(axes.rotation_to_gltf((0.0, 0.0, 0.0, 1.0))) == "(0.0, 0.0, 0.0, 1.0)"


@pytest.mark.parametrize("seed", range(20))
def test_transform_conversion_equals_conjugation(seed):
    """C M C^T is again exactly TRS, with the documented component mapping."""
    rng = np.random.default_rng(seed)
    t = Transform(
        position=rng.normal(size=3) * 10,
        rotation=rng.normal(size=4),
        scale=rng.uniform(0.1, 4, size=3) * rng.choice([-1, 1], size=3),
    )
    expected = C4 @ t.matrix @ C4.T
    converted = math3d.compose_matrix(
        axes.vector_to_gltf(t.position),
        axes.rotation_to_gltf(t.rotation),
        axes.scale_to_gltf(t.scale),
    )
    np.testing.assert_allclose(converted, expected, atol=1e-12)
    back = Transform(
        axes.vector_from_gltf(axes.vector_to_gltf(t.position)),
        axes.rotation_from_gltf(axes.rotation_to_gltf(t.rotation)),
        axes.scale_from_gltf(axes.scale_to_gltf(t.scale)),
    )
    assert back == t  # bit-exact


def test_hierarchy_conversion_is_consistent():
    """Converting each local matrix converts the world matrix: C(AB)C^T = (CAC^T)(CBC^T)."""
    rng = np.random.default_rng(7)
    mats = [
        Transform(rng.normal(size=3), rng.normal(size=4), rng.uniform(0.5, 2, size=3)).matrix
        for _ in range(4)
    ]
    world = mats[0] @ mats[1] @ mats[2] @ mats[3]
    converted = [C4 @ m @ C4.T for m in mats]
    np.testing.assert_allclose(
        converted[0] @ converted[1] @ converted[2] @ converted[3], C4 @ world @ C4.T, atol=1e-10
    )


def test_uv_flip():
    uvs = np.array([[0.0, 0.0], [0.25, 0.1], [1.0, 1.0]], dtype=np.float32)
    flipped = axes.flip_uvs(uvs)
    np.testing.assert_allclose(flipped, [[0, 1], [0.25, 0.9], [1, 0]], atol=1e-7)
    assert np.abs(axes.flip_uvs(flipped) - uvs).max() <= 2**-23  # within one float32 ulp of 1.0
