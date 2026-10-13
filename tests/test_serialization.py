import copy
import json

import pytest

from aether3d import FORMAT_VERSION, Mesh, ProjectFormatError, Scene, load_scene, serialization


def build_scene():
    scene = Scene("Round Trip")
    cube = scene.add_cube(name="Cube", size=2, position=(0, 0, 1))
    cube.rotation_euler = (10, 20, 30)
    pivot = scene.add_object(name="Pivot", position=(0.1, -0.2, 0.3), scale=(1, 2, 3))
    scene.add_sphere(name="Moon", radius=0.25, segments=8, rings=4, parent=pivot)
    scene.add_plane(name="Ground", size=10, parent=pivot)
    scene.add_object(name="Nested Empty", parent=cube)
    return scene


def assert_scenes_equal(a, b):
    assert a.name == b.name
    assert [o.id for o in a] == [o.id for o in b]
    for x, y in zip(a, b, strict=True):
        assert x.name == y.name
        assert (x.parent and x.parent.id) == (y.parent and y.parent.id)
        assert [c.id for c in x.children] == [c.id for c in y.children]
        assert x.transform == y.transform  # exact, not approximate
        assert (x.mesh is None) == (y.mesh is None)
        if x.mesh is not None:
            assert x.mesh == y.mesh


def test_example_from_readme(tmp_path):
    scene = Scene("First Scene")
    cube = scene.add_cube(name="Cube", size=2)
    cube.position = (0, 0, 1)
    sphere = scene.add_sphere(name="Sphere", radius=1)
    sphere.position = (3, 0, 1)
    scene.save(tmp_path / "example.aether")
    loaded_scene = Scene.load(tmp_path / "example.aether")
    assert_scenes_equal(scene, loaded_scene)
    assert loaded_scene.get(cube.id).position == (0.0, 0.0, 1.0)
    assert loaded_scene.get_by_name("Sphere").position == (3.0, 0.0, 1.0)


def test_round_trip_preserves_everything(tmp_path):
    scene = build_scene()
    path = tmp_path / "scene.aether"
    scene.save(path)
    assert_scenes_equal(scene, Scene.load(path))
    assert_scenes_equal(scene, load_scene(str(path)))


def test_save_is_deterministic_and_stable_across_round_trips(tmp_path):
    scene = build_scene()
    scene.save(tmp_path / "a.aether")
    Scene.load(tmp_path / "a.aether").save(tmp_path / "b.aether")
    assert (tmp_path / "a.aether").read_bytes() == (tmp_path / "b.aether").read_bytes()


def test_document_structure():
    doc = build_scene().to_dict()
    assert doc["format"] == "aether3d.project"
    assert doc["format_version"] == FORMAT_VERSION
    objects = doc["scene"]["objects"]
    assert [o["name"] for o in objects] == ["Cube", "Nested Empty", "Pivot", "Moon", "Ground"]
    assert objects[1]["parent"] == objects[0]["id"]
    assert objects[1]["mesh"] is None
    assert len(doc["meshes"]) == 3
    json.dumps(doc, allow_nan=False)  # strictly JSON-compatible


def test_float32_values_are_written_readably(tmp_path):
    scene = Scene()
    scene.add_object(mesh=Mesh([(0.1, 0.2, 0.3)] * 3, [(0, 1, 2)]))
    text = serialization.dumps(scene.to_dict())
    assert "[0.1, 0.2, 0.3]" in text


def test_shared_meshes_are_stored_once(tmp_path):
    scene = Scene()
    mesh = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    scene.add_object("A", mesh=mesh)
    scene.add_object("B", mesh=mesh)
    doc = scene.to_dict()
    assert len(doc["meshes"]) == 1
    loaded = Scene.from_dict(doc)
    assert loaded.get_by_name("A").mesh is loaded.get_by_name("B").mesh


def test_empty_scene_round_trip(tmp_path):
    Scene("Empty").save(tmp_path / "e.aether")
    loaded = Scene.load(tmp_path / "e.aether")
    assert loaded.name == "Empty" and len(loaded) == 0


def test_children_may_precede_parents_in_file():
    doc = build_scene().to_dict()
    doc["scene"]["objects"].reverse()
    loaded = Scene.from_dict(doc)
    assert loaded.get_by_name("Moon").parent.name == "Pivot"
    # Sibling order follows file order.
    assert [c.name for c in loaded.get_by_name("Pivot").children] == ["Ground", "Moon"]


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "s.aether"
    build_scene().save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["s.aether"]


def test_save_failure_keeps_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "s.aether"
    build_scene().save(path)
    original = path.read_bytes()

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(serialization.os, "replace", boom)
    with pytest.raises(OSError):
        Scene("Other").save(path)
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["s.aether"]


# --------------------------------------------------------------------------- #
# Validation of malformed documents
# --------------------------------------------------------------------------- #


def mutated(fn):
    doc = copy.deepcopy(build_scene().to_dict())
    fn(doc)
    return doc


def obj(doc, name):
    return next(o for o in doc["scene"]["objects"] if o["name"] == name)


@pytest.mark.parametrize(
    "mutate, location, message",
    [
        (lambda d: d.update(format="something.else"), None, "not an Aether3D project"),
        (lambda d: d.update(format_version=FORMAT_VERSION + 1), "format_version", "upgrade"),
        (lambda d: d.update(format_version="1"), "format_version", "positive integer"),
        (lambda d: d.update(extra=1), "", "unknown key"),
        (lambda d: d.pop("meshes"), "", "missing required"),
        (lambda d: d["scene"].update(name=""), "scene.name", "non-empty string"),
        (lambda d: d["scene"]["objects"][0].pop("transform"), "scene.objects[0]", "missing"),
        (
            lambda d: d["scene"]["objects"][0]["transform"].update(postion=[0, 0, 0]),
            "scene.objects[0].transform",
            "unknown key",
        ),
        (
            lambda d: d["scene"]["objects"][0]["transform"].update(scale=[1, 1]),
            "scene.objects[0].transform.scale",
            "3 numbers",
        ),
        (
            lambda d: d["scene"]["objects"][0]["transform"].update(rotation=[0, 0, 0, 0]),
            "scene.objects[0].transform",
            "non-zero",
        ),
        (lambda d: obj(d, "Moon").update(parent="ghost"), "scene.objects[3].parent", "ghost"),
        (
            lambda d: obj(d, "Moon").update(id=obj(d, "Cube")["id"]),
            "scene.objects[3].id",
            "duplicate",
        ),
        (lambda d: obj(d, "Moon").update(name="Cube"), "scene.objects[3].name", "duplicate"),
        (lambda d: obj(d, "Moon").update(mesh=99), "scene.objects[3].mesh", "index"),
        (lambda d: obj(d, "Moon").update(mesh=True), "scene.objects[3].mesh", "index"),
        (
            lambda d: obj(d, "Pivot").update(parent=obj(d, "Moon")["id"]),
            "scene.objects[2].parent",
            "cycle",
        ),
        (
            lambda d: obj(d, "Cube").update(parent=obj(d, "Cube")["id"]),
            "scene.objects[0].parent",
            "cycle",
        ),
        (lambda d: d["meshes"][0]["triangles"].append([0, 1, 9999]), "meshes[0]", "indices"),
        (lambda d: d["meshes"][0]["triangles"].append([0, 1, 2**70]), "meshes[0]", "range"),
        (lambda d: d["meshes"][0]["positions"].pop(), "meshes[0]", "normals"),
        (lambda d: d["meshes"][0]["positions"][2].append(1), "meshes[0].positions[2]", "3 numbers"),
        (
            lambda d: d["meshes"][0]["triangles"].append([0, 1, 2.0]),
            "meshes[0].triangles[12]",
            "integers",
        ),
        (lambda d: d["meshes"][0].update(colors=[]), "meshes[0]", "unknown key"),
    ],
)
def test_invalid_documents_are_rejected_with_location(mutate, location, message):
    with pytest.raises(ProjectFormatError, match=message) as info:
        Scene.from_dict(mutated(mutate))
    assert info.value.location == location or (location in (None, "") and not info.value.location)


@pytest.mark.parametrize(
    "text, message",
    [
        ("{not json", "invalid JSON"),
        ("[]", "top level"),
        ('{"a": 1, "a": 2}', "duplicate key"),
        ('{"format": NaN}', "NaN"),
    ],
)
def test_invalid_json_is_rejected(tmp_path, text, message):
    path = tmp_path / "bad.aether"
    path.write_text(text)
    with pytest.raises(ProjectFormatError, match=message):
        Scene.load(path)


def test_non_utf8_file_is_rejected(tmp_path):
    path = tmp_path / "bad.aether"
    path.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ProjectFormatError, match="UTF-8"):
        Scene.load(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Scene.load(tmp_path / "nope.aether")


def test_older_versions_are_migrated(monkeypatch):
    """The migration chain upgrades documents one version at a time."""
    doc = build_scene().to_dict()
    old = copy.deepcopy(doc)
    old["format_version"] = 1
    old["scene"]["title"] = old["scene"].pop("name")  # pretend v1 called it "title"

    def v1_to_v2(d):
        d = copy.deepcopy(d)
        d["scene"]["name"] = d["scene"].pop("title")
        return d

    monkeypatch.setattr(serialization, "FORMAT_VERSION", 2)
    monkeypatch.setitem(serialization.MIGRATIONS, 1, v1_to_v2)
    loaded = serialization.scene_from_dict(old)
    assert loaded.name == "Round Trip"
    assert old["format_version"] == 1  # the input document is not modified
