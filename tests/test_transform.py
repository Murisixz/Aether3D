import numpy as np
import pytest

from aether3d import Transform, ValidationError


def test_defaults_are_identity():
    t = Transform()
    assert t.position == (0.0, 0.0, 0.0)
    assert t.rotation == (0.0, 0.0, 0.0, 1.0)
    assert t.scale == (1.0, 1.0, 1.0)
    np.testing.assert_array_equal(t.matrix, np.eye(4))


def test_components_are_plain_float_tuples():
    t = Transform(position=[1, 2, 3])
    assert t.position == (1.0, 2.0, 3.0)
    assert all(type(c) is float for c in t.position)
    t.position = np.array([4, 5, 6])
    assert t.position == (4.0, 5.0, 6.0)


def test_uniform_scale_shorthand():
    t = Transform()
    t.scale = 3
    assert t.scale == (3.0, 3.0, 3.0)


def test_rotation_is_normalized_and_euler_view():
    t = Transform()
    t.rotation = (0, 0, 2, 2)
    np.testing.assert_allclose(t.rotation, (0, 0, 2**-0.5, 2**-0.5))
    np.testing.assert_allclose(t.rotation_euler, (0, 0, 90), atol=1e-9)
    t.rotation_euler = (0, 0, -90)
    np.testing.assert_allclose(t.matrix[:3, 0], (0, -1, 0), atol=1e-12)


def test_matrix_translation_column():
    t = Transform(position=(1, 2, 3), scale=2)
    np.testing.assert_allclose(t.matrix[:3, 3], (1, 2, 3))
    np.testing.assert_allclose(np.diag(t.matrix)[:3], (2, 2, 2))


def test_from_matrix_round_trip():
    t = Transform(position=(1, 2, 3), scale=(1, 2, 3))
    t.rotation_euler = (10, 20, 30)
    assert Transform.from_matrix(t.matrix).is_close(t)


def test_invalid_values_raise_and_keep_state():
    t = Transform(position=(1, 1, 1))
    with pytest.raises(ValidationError, match="position"):
        t.position = (1, 2)
    with pytest.raises(ValidationError, match="scale"):
        t.scale = (1, float("nan"), 1)
    with pytest.raises(ValidationError, match="rotation"):
        t.rotation = (0, 0, 0, 0)
    assert t.position == (1.0, 1.0, 1.0)


def test_equality_and_copy():
    a = Transform(position=(1, 2, 3))
    b = a.copy()
    assert a == b and a is not b
    b.position = (0, 0, 0)
    assert a != b
    assert a.position == (1.0, 2.0, 3.0)
