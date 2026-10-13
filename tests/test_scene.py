import uuid

import pytest

from aether3d import (
    AetherError,
    DetachedObjectError,
    Mesh,
    NameConflictError,
    ObjectNotFoundError,
    Scene,
    ValidationError,
    create_cube,
)


def test_scene_name():
    scene = Scene("My Scene")
    assert scene.name == "My Scene"
    assert Scene().name == "Untitled Scene"
    with pytest.raises(ValidationError):
        Scene("  ")
    with pytest.raises(ValidationError):
        scene.name = ""


def test_add_and_get_objects():
    scene = Scene()
    cube = scene.add_cube(name="Cube", size=2)
    sphere = scene.add_sphere(name="Sphere", radius=1)
    plane = scene.add_plane(name="Plane")
    empty = scene.add_object(name="Empty")
    assert len(scene) == 4
    assert scene.get(cube.id) is cube
    assert scene.get_by_name("Sphere") is sphere
    assert cube.mesh.vertex_count == 24
    assert plane.mesh.triangle_count == 2
    assert empty.mesh is None
    assert cube in scene and cube.id in scene and "nope" not in scene
    assert scene.objects == (cube, sphere, plane, empty)
    assert cube.scene is scene


def test_ids_are_unique_uuids():
    scene = Scene()
    ids = {scene.add_cube().id for _ in range(50)}
    assert len(ids) == 50
    for object_id in ids:
        assert str(uuid.UUID(object_id)) == object_id


def test_explicit_object_id():
    scene = Scene()
    obj = scene.add_object("A", object_id="custom-id")
    assert scene.get("custom-id") is obj
    with pytest.raises(ValidationError):
        scene.add_object("B", object_id="custom-id")


def test_default_names_are_unique():
    scene = Scene()
    names = [scene.add_cube().name for _ in range(3)]
    assert names == ["Cube", "Cube.001", "Cube.002"]
    assert scene.add_object().name == "Object"
    assert scene.add_object(mesh=create_cube()).name == "Mesh"


def test_explicit_duplicate_name_is_an_error():
    scene = Scene()
    scene.add_cube(name="Box")
    with pytest.raises(NameConflictError):
        scene.add_sphere(name="Box")
    assert len(scene) == 1


def test_rename():
    scene = Scene()
    a = scene.add_cube(name="A")
    scene.add_cube(name="B")
    a.name = "C"
    assert scene.get_by_name("C") is a
    with pytest.raises(ObjectNotFoundError):
        scene.get_by_name("A")
    with pytest.raises(NameConflictError):
        a.name = "B"
    a.name = "C"  # renaming to its own name is a no-op
    with pytest.raises(ValidationError):
        a.name = ""


def test_lookup_errors_are_descriptive():
    scene = Scene()
    with pytest.raises(ObjectNotFoundError, match="no object with id 'missing'"):
        scene.get("missing")
    with pytest.raises(KeyError):  # also a KeyError for generic handling
        scene.get_by_name("missing")
    with pytest.raises(AetherError):
        scene.get(None)


def test_creation_kwargs_set_transform():
    scene = Scene()
    obj = scene.add_cube(position=(1, 2, 3), rotation=(0, 0, 1, 0), scale=2)
    assert obj.position == (1.0, 2.0, 3.0)
    assert obj.rotation == (0.0, 0.0, 1.0, 0.0)
    assert obj.scale == (2.0, 2.0, 2.0)


def test_failed_creation_does_not_modify_scene():
    scene = Scene()
    with pytest.raises(ValidationError):
        scene.add_cube(name="Bad", position=(1, 2))
    with pytest.raises(ObjectNotFoundError):
        scene.add_cube(name="Bad", parent="no-such-parent")
    with pytest.raises(ValidationError):
        scene.add_object(name="Bad", mesh="not a mesh")
    assert len(scene) == 0


def test_mesh_assignment():
    scene = Scene()
    obj = scene.add_object("Holder")
    mesh = create_cube()
    obj.mesh = mesh
    assert obj.mesh is mesh
    obj.mesh = None
    with pytest.raises(ValidationError):
        obj.mesh = [1, 2, 3]
    shared = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    a = scene.add_object("A", mesh=shared)
    b = scene.add_object("B", mesh=shared)
    assert a.mesh is b.mesh


def test_remove_recursive():
    scene = Scene()
    root = scene.add_object("Root")
    child = scene.add_cube("Child", parent=root)
    grandchild = scene.add_cube("Grandchild", parent=child)
    other = scene.add_cube("Other")
    scene.remove(child)
    assert len(scene) == 2
    assert root.children == ()
    assert child.scene is None and grandchild.scene is None
    assert child.id not in scene
    # Names of removed objects become available again.
    scene.add_cube("Child")
    scene.remove(other.id)
    assert [o.name for o in scene] == ["Root", "Child"]


def test_remove_non_recursive_keeps_children_in_place():
    scene = Scene()
    root = scene.add_object("Root", position=(10, 0, 0))
    first = scene.add_object("First", parent=root)
    middle = scene.add_object("Middle", parent=root, position=(1, 0, 0), scale=2)
    last = scene.add_object("Last", parent=root)
    kid = scene.add_cube("Kid", parent=middle, position=(1, 0, 0))
    before = kid.world_position
    scene.remove(middle, recursive=False)
    assert kid.parent is root
    assert root.children == (first, kid, last)
    assert kid.world_position == pytest.approx(before)
    assert kid.scale == pytest.approx((2, 2, 2))


def test_detached_objects_cannot_be_used_in_hierarchy():
    scene = Scene()
    obj = scene.add_cube("A")
    scene.remove(obj)
    with pytest.raises(DetachedObjectError):
        obj.set_parent(None)
    with pytest.raises(DetachedObjectError):
        obj.name = "B"
    with pytest.raises(ObjectNotFoundError):
        scene.remove(obj)


def test_objects_from_other_scenes_are_rejected():
    a, b = Scene("A"), Scene("B")
    foreign = b.add_object("Foreign")
    with pytest.raises(ObjectNotFoundError):
        a.add_cube(parent=foreign)
    with pytest.raises(ObjectNotFoundError):
        a.remove(foreign)
    assert foreign not in a


def test_repr():
    scene = Scene("S")
    obj = scene.add_cube("C")
    assert "S" in repr(scene) and obj.id in repr(obj)
