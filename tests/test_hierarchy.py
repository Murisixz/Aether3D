import numpy as np
import pytest

from aether3d import HierarchyError, Scene, ValidationError, math3d


def approx(values, abs=1e-9):
    return pytest.approx(tuple(values), abs=abs)


@pytest.fixture
def scene():
    return Scene("Hierarchy")


def test_parenting_and_children_order(scene):
    parent = scene.add_object("Parent")
    a = scene.add_cube("A", parent=parent)
    b = scene.add_cube("B", parent=parent.id)  # by id works too
    assert a.parent is parent and b.parent is parent
    assert parent.children == (a, b)
    assert scene.roots == (parent,)
    assert list(parent.iter_descendants()) == [a, b]
    assert list(a.iter_ancestors()) == [parent]


def test_iteration_is_depth_first(scene):
    r1 = scene.add_object("R1")
    r2 = scene.add_object("R2")
    c1 = scene.add_object("C1", parent=r1)
    g1 = scene.add_object("G1", parent=c1)
    c2 = scene.add_object("C2", parent=r1)
    assert scene.objects == (r1, c1, g1, c2, r2)


def test_world_translation_composes(scene):
    parent = scene.add_object("P", position=(1, 2, 3))
    child = scene.add_object("C", parent=parent, position=(10, 0, 0))
    grandchild = scene.add_object("G", parent=child, position=(0, 0, 5))
    assert grandchild.world_position == approx((11, 2, 8))
    assert grandchild.position == (0.0, 0.0, 5.0)  # local is untouched


def test_parent_rotation_rotates_child_offset(scene):
    parent = scene.add_object("P", position=(5, 0, 0))
    parent.rotation_euler = (0, 0, 90)
    child = scene.add_object("C", parent=parent, position=(1, 0, 0))
    assert child.world_position == approx((5, 1, 0))
    assert child.world_rotation == approx(math3d.quat_from_euler((0, 0, 90)))


def test_parent_scale_scales_child(scene):
    parent = scene.add_object("P", scale=2)
    child = scene.add_object("C", parent=parent, position=(1, 1, 1), scale=3)
    assert child.world_position == approx((2, 2, 2))
    assert child.world_scale == approx((6, 6, 6))


def test_world_matrix_is_product_of_local_matrices(scene):
    a = scene.add_object("A", position=(1, 2, 3), scale=(1, 2, 1))
    a.rotation_euler = (10, 20, 30)
    b = scene.add_object("B", parent=a, position=(-1, 0, 4), scale=0.5)
    b.rotation_euler = (-45, 0, 90)
    c = scene.add_object("C", parent=b, position=(0, 3, 0))
    c.rotation_euler = (0, 60, 0)
    np.testing.assert_allclose(
        c.world_matrix, a.local_matrix @ b.local_matrix @ c.local_matrix, atol=1e-12
    )
    np.testing.assert_allclose(c.world_position, c.world_matrix[:3, 3])


def test_reparent_keeps_local_by_default(scene):
    p1 = scene.add_object("P1", position=(10, 0, 0))
    p2 = scene.add_object("P2", position=(0, 10, 0))
    child = scene.add_object("C", parent=p1, position=(1, 0, 0))
    child.set_parent(p2)
    assert child.position == (1.0, 0.0, 0.0)
    assert child.world_position == approx((1, 10, 0))
    assert p1.children == () and p2.children == (child,)


def test_reparent_keep_world_transform(scene):
    p1 = scene.add_object("P1", position=(10, 0, 0), scale=2)
    p1.rotation_euler = (0, 0, 45)
    p2 = scene.add_object("P2", position=(0, -3, 1), scale=0.5)
    p2.rotation_euler = (30, 0, -60)
    child = scene.add_object("C", parent=p1, position=(1, 2, 3))
    child.rotation_euler = (5, 10, 15)
    before = child.world_matrix
    child.set_parent(p2, keep_world_transform=True)
    np.testing.assert_allclose(child.world_matrix, before, atol=1e-9)
    child.set_parent(None, keep_world_transform=True)
    assert child.parent is None and child in scene.roots
    np.testing.assert_allclose(child.local_matrix, before, atol=1e-9)


def test_keep_world_transform_under_zero_scale_parent_fails_cleanly(scene):
    flat = scene.add_object("Flat", scale=(1, 1, 0))
    child = scene.add_object("C", position=(1, 1, 1))
    with pytest.raises(ValidationError):
        child.set_parent(flat, keep_world_transform=True)
    assert child.parent is None and child.position == (1.0, 1.0, 1.0)


def test_cycles_are_rejected(scene):
    a = scene.add_object("A")
    b = scene.add_object("B", parent=a)
    c = scene.add_object("C", parent=b)
    with pytest.raises(HierarchyError):
        a.set_parent(a)
    with pytest.raises(HierarchyError):
        a.set_parent(c)
    assert a.parent is None and c.parent is b


def test_moving_parent_moves_descendants(scene):
    parent = scene.add_object("P")
    child = scene.add_cube("C", parent=parent, position=(0, 0, 1))
    parent.position = (4, 4, 4)
    assert child.world_position == approx((4, 4, 5))
