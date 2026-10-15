"""Object tags and metadata (0.10.0): validation, history, persistence, glTF, scripts, CLI."""

from __future__ import annotations

import json

import pytest

import aether3d as a3
from aether3d import Scene, ValidationError
from aether3d.serialization import FORMAT_VERSION, scene_from_dict, scene_to_dict


def tagged_scene() -> Scene:
    s = Scene("Tags")
    keep = s.add_cube(name="Keep")
    keep.tags = ["tower", "stone"]
    keep.metadata = {"floors": 3, "role": "keep", "doors": [{"side": "south"}]}
    s.add_sphere(name="Ball", parent=keep)
    return s


def test_tags_are_sorted_unique_and_validated():
    s = Scene()
    c = s.add_cube(name="C")
    c.tags = ["b", "a", "b"]
    assert c.tags == ("a", "b") and c.has_tag("a") and not c.has_tag("A")
    c.add_tags("c", "a")
    c.remove_tags("b", "missing")
    assert c.tags == ("a", "c")
    for bad in ("tower", ["ok", ""], [" padded"], ["x" * 65], [3], [f"t{i}" for i in range(65)]):
        with pytest.raises(ValidationError):
            c.tags = bad
    assert c.tags == ("a", "c")


def test_metadata_is_json_and_returned_as_a_copy():
    s = Scene()
    c = s.add_cube(name="C")
    c.set_metadata("floors", 3)
    c.set_metadata("rooms", ["hall", "kitchen"])
    data = c.metadata
    data["floors"] = 99  # a copy
    assert c.metadata == {"floors": 3, "rooms": ["hall", "kitchen"]}
    c.remove_metadata("rooms")
    c.remove_metadata("not there")
    assert c.metadata == {"floors": 3}
    for bad in (
        {1: "x"},
        {"": 1},
        {"nan": float("nan")},
        {"set": {1, 2}},
        [],
        {"big": "x" * 20000},
    ):
        with pytest.raises(ValidationError):
            c.metadata = bad
    assert c.metadata == {"floors": 3}


def test_tags_and_metadata_are_undoable_steps():
    s = Scene()
    c = s.add_cube(name="C")
    s.history.clear()
    c.add_tags("tower")
    c.set_metadata("floors", 2)
    c.tags = ["keep"]
    assert [e.label for e in s.history.entries] == [
        "Tag 'C' 'tower'",
        "Set metadata 'floors' of 'C'",
        "Set tags of 'C'",
    ]
    s.history.undo(3)
    assert c.tags == () and c.metadata == {}
    s.history.redo(3)
    assert c.tags == ("keep",) and c.metadata == {"floors": 2}
    c.tags = ["keep"]  # unchanged: no step
    assert len(s.history) == 3


def test_projects_store_tags_and_metadata(tmp_path):
    s = tagged_scene()
    s.save(tmp_path / "t.aether", history=True)
    doc = json.loads((tmp_path / "t.aether").read_text())
    assert doc["format_version"] == FORMAT_VERSION == 6
    record = next(o for o in doc["scene"]["objects"] if o["name"] == "Keep")
    assert list(record)[:4] == ["id", "name", "tags", "metadata"]
    assert record["tags"] == ["stone", "tower"] and record["metadata"]["floors"] == 3
    loaded = Scene.load(tmp_path / "t.aether", history=True)
    keep = loaded.get_by_name("Keep")
    assert keep.tags == ("stone", "tower") and keep.metadata["doors"] == [{"side": "south"}]
    loaded.history.undo(len(loaded.history))  # the history file restores them too
    assert keep.tags == () and keep.metadata == {}
    loaded.history.redo(len(loaded.history.redo_entries))
    assert keep.tags == ("stone", "tower")


def test_version_5_projects_load_without_tags():
    doc = scene_to_dict(tagged_scene())
    doc["format_version"] = 5
    for record in doc["scene"]["objects"]:
        del record["tags"], record["metadata"]
    old = scene_from_dict(doc)
    assert all(o.tags == () and o.metadata == {} for o in old)
    assert scene_to_dict(old)["format_version"] == 6


def test_invalid_stored_tags_are_reported_with_their_location():
    doc = scene_to_dict(tagged_scene())
    doc["scene"]["objects"][0]["tags"] = ["ok", ""]
    with pytest.raises(a3.ProjectFormatError) as exc:
        scene_from_dict(doc)
    assert exc.value.location == "scene.objects[0].tags"
    doc["scene"]["objects"][0]["tags"] = "tower"
    with pytest.raises(a3.ProjectFormatError):
        scene_from_dict(doc)


def test_gltf_carries_tags_and_metadata(tmp_path):
    s = tagged_scene()
    s.export(tmp_path / "t.glb")
    back = Scene.import_file(tmp_path / "t.glb")
    assert back.get_by_name("Keep").tags == ("stone", "tower")
    assert back.get_by_name("Keep").metadata == s.get_by_name("Keep").metadata
    assert back.get_by_name("Ball").tags == ()
    # Invalid values in a file are reported, not imported.
    s.export(tmp_path / "t.gltf")
    data = json.loads((tmp_path / "t.gltf").read_text())
    node = next(n for n in data["nodes"] if n.get("name") == "Keep")
    node["extras"]["aether3d"]["tags"] = [""]
    (tmp_path / "t.gltf").write_text(json.dumps(data))
    result = a3.import_gltf(tmp_path / "t.gltf")
    assert result.scene.get_by_name("Keep").tags == ()
    assert "STORED_ANNOTATIONS_REJECTED" in {d.code for d in result.diagnostics}


def test_duplicates_and_merges_copy_tags_and_metadata():
    s = tagged_scene()
    copy = s.duplicate("Keep")
    assert copy.tags == ("stone", "tower") and copy.metadata["floors"] == 3
    copy.set_metadata("floors", 4)
    assert s.get_by_name("Keep").metadata["floors"] == 3
    merged = s.merge(tagged_scene())
    assert merged[0].tags == ("stone", "tower")


def test_scripts_and_the_cli_set_tags_and_metadata(tmp_path, capsys):
    from aether3d.cli import main

    s = tagged_scene()
    result = a3.run_scene_script(
        s,
        {
            "format": "aether3d.scene-script",
            "version": 1,
            "operations": [
                {
                    "op": "set_object",
                    "object": "Ball",
                    "add_tags": ["prop"],
                    "metadata": {"weight": 2.5},
                },
                {"op": "set_object", "object": "Keep", "remove_tags": ["stone"]},
                {"op": "set_object", "object": "Keep", "metadata": {"role": None}},
            ],
        },
    )
    assert result["operations"][0]["tags"] == ["prop"]
    assert s.get_by_name("Keep").tags == ("tower",) and "role" not in s.get_by_name("Keep").metadata
    path = tmp_path / "t.aether"
    tagged_scene().save(path)
    argv = ["object", "set", str(path), "Ball", "--add-tag", "prop", "--metadata", "weight=2.5"]
    assert main([*argv, "--metadata", "label=red ball", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["operations"][0]["tags"] == ["prop"]
    ball = Scene.load(path).get_by_name("Ball")
    assert ball.tags == ("prop",) and ball.metadata == {"weight": 2.5, "label": "red ball"}
    assert main(["object", "set", str(path), "Ball", "--tags", ""]) == 0
    assert Scene.load(path).get_by_name("Ball").tags == ()
