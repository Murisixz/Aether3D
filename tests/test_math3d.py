import math

import numpy as np
import pytest

from aether3d import ValidationError, math3d


def rotate(q, v):
    return math3d.quat_to_matrix3(q) @ np.asarray(v, dtype=float)


def test_quat_is_normalized_and_canonical():
    q = math3d.quat((0, 0, 0, -2))
    assert q == (0.0, 0.0, 0.0, 1.0)
    q = math3d.quat((1, 1, 1, 1))
    assert math.isclose(np.linalg.norm(q), 1.0)


@pytest.mark.parametrize(
    "bad", [(0, 0, 0, 0), (1, 2, 3), "abcd", (1, 2, float("nan"), 1), None, (1, 2, "x", 4)]
)
def test_quat_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        math3d.quat(bad)


@pytest.mark.parametrize("bad", [(1, 2), (1, 2, 3, 4), (0, float("inf"), 0), "xyz", 5])
def test_vec3_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        math3d.vec3(bad)


def test_scale3_accepts_scalar():
    assert math3d.scale3(2) == (2.0, 2.0, 2.0)
    assert math3d.scale3((1, 2, 3)) == (1.0, 2.0, 3.0)
    with pytest.raises(ValidationError):
        math3d.scale3(True)


def test_axis_angle_rotates_counter_clockwise():
    q = math3d.quat_from_axis_angle((0, 0, 1), 90)
    np.testing.assert_allclose(rotate(q, (1, 0, 0)), (0, 1, 0), atol=1e-12)
    q = math3d.quat_from_axis_angle((1, 0, 0), 90)
    np.testing.assert_allclose(rotate(q, (0, 1, 0)), (0, 0, 1), atol=1e-12)
    with pytest.raises(ValidationError):
        math3d.quat_from_axis_angle((0, 0, 0), 10)


def test_euler_order_is_x_then_y_then_z():
    q = math3d.quat_from_euler((90, 90, 0))
    # X first: +Y -> +Z; then Y: +Z -> +X.
    np.testing.assert_allclose(rotate(q, (0, 1, 0)), (1, 0, 0), atol=1e-12)
    rx = math3d.quat_to_matrix3(math3d.quat_from_axis_angle((1, 0, 0), 10))
    ry = math3d.quat_to_matrix3(math3d.quat_from_axis_angle((0, 1, 0), 20))
    rz = math3d.quat_to_matrix3(math3d.quat_from_axis_angle((0, 0, 1), 30))
    np.testing.assert_allclose(
        math3d.quat_to_matrix3(math3d.quat_from_euler((10, 20, 30))), rz @ ry @ rx, atol=1e-12
    )


def test_euler_round_trip():
    rng = np.random.default_rng(1)
    for angles in rng.uniform(-179, 179, size=(200, 3)):
        angles[1] = angles[1] / 2.1  # keep Y inside (-90, 90) where Euler angles are unique
        back = math3d.quat_to_euler(math3d.quat_from_euler(angles))
        np.testing.assert_allclose(back, angles, atol=1e-7)


def test_euler_gimbal_lock_still_represents_rotation():
    q = math3d.quat_from_euler((30, 90, 40))
    back = math3d.quat_from_euler(math3d.quat_to_euler(q))
    np.testing.assert_allclose(math3d.quat_to_matrix3(back), math3d.quat_to_matrix3(q), atol=1e-9)


def test_matrix_quat_round_trip():
    rng = np.random.default_rng(2)
    for raw in rng.normal(size=(200, 4)):
        q = math3d.quat(raw)
        back = math3d.matrix3_to_quat(math3d.quat_to_matrix3(q))
        np.testing.assert_allclose(back, q, atol=1e-12)


def test_quat_multiply_matches_matrix_product():
    a = math3d.quat_from_euler((10, 50, -20))
    b = math3d.quat_from_euler((-70, 5, 33))
    np.testing.assert_allclose(
        math3d.quat_to_matrix3(math3d.quat_multiply(a, b)),
        math3d.quat_to_matrix3(a) @ math3d.quat_to_matrix3(b),
        atol=1e-12,
    )


def test_compose_applies_scale_then_rotation_then_translation():
    m = math3d.compose_matrix((10, 0, 0), math3d.quat_from_axis_angle((0, 0, 1), 90), (2, 1, 1))
    # (1,0,0) -> scale (2,0,0) -> rotate (0,2,0) -> translate (10,2,0)
    np.testing.assert_allclose(math3d.transform_point(m, (1, 0, 0)), (10, 2, 0), atol=1e-12)


@pytest.mark.parametrize("scale", [(1, 1, 1), (2, 3, 4), (-1, 2, 3), (0.5, 0.5, -0.25)])
def test_decompose_inverts_compose(scale):
    q = math3d.quat_from_euler((15, -40, 120))
    m = math3d.compose_matrix((1, -2, 3), q, scale)
    position, rotation, back_scale = math3d.decompose_matrix(m)
    np.testing.assert_allclose(math3d.compose_matrix(position, rotation, back_scale), m, atol=1e-9)
    np.testing.assert_allclose(position, (1, -2, 3), atol=1e-12)


def test_decompose_rejects_singular_matrix():
    with pytest.raises(ValidationError):
        math3d.decompose_matrix(math3d.compose_matrix(scale=(1, 0, 1)))
    with pytest.raises(ValidationError):
        math3d.decompose_matrix(np.eye(3))
