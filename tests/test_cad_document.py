"""The CAD document: items, dependencies, incremental regeneration, links to scene
objects, transactions, history (undo/redo, history files), persistence and guards."""

from __future__ import annotations

import json
import math

import pytest

import aether3d
from aether3d import history as history_module
from aether3d.errors import AetherError, CadError

pytest.importorskip("manifold3d")


def bracket(scene=None):
    scene = scene or aether3d.Scene("bracket")
    doc = scene.cad
    with doc.batch("Bracket"):
        doc.parameter("width", "80 mm")
        doc.parameter("hole_d", "8 mm")
        sk = doc.sketch("Profile", plane="XY")
        lines = sk.rectangle((0, 0), (80, 40))
        sk.constrain("fix", "p1")
        sk.dimension("length", lines[0], value="=width", id="w")
        sk.dimension("length", lines[1], value="40 mm", id="h")
        part = doc.part("Bracket")
        part.extrude("Profile", 10, id="base")
        part.hole("base:end", [[20, 20, 10], [60, 20, 10]], "=hole_d", id="holes")
    return scene, doc, sk, part


def test_bracket_volume_is_exact_and_follows_its_parameters():
    scene, doc, sk, part = bracket()
    assert sk.solve()["status"] == "fully_constrained"
    v = part.measure()["volume"]
    assert v == pytest.approx(80 * 40 * 10 - 2 * math.pi * 16 * 10, rel=1e-13)
    doc.parameter("width", "100 mm")
    assert sk.measure("w") == pytest.approx(100.0, abs=1e-9)
    assert part.measure()["volume"] == pytest.approx(
        100 * 40 * 10 - 2 * math.pi * 16 * 10, rel=1e-13
    )
    doc.parameter("hole_d", "10 mm")
    assert part.measure()["volume"] == pytest.approx(
        100 * 40 * 10 - 2 * math.pi * 25 * 10, rel=1e-13
    )


def test_only_affected_items_regenerate():
    scene, doc, sk, part = bracket()
    other = doc.sketch("Other")
    other.rectangle((0, 0), (5, 5))
    p2 = doc.part("Other part")
    p2.extrude("Other", 3)
    doc.result(part.item)
    doc.result(p2.item)
    before = doc.stats["evaluations"]
    doc.parameter("hole_d", "6 mm")  # the parameter and the bracket only
    assert doc.stats["evaluations"] - before == 2


def test_links_show_meshes_that_follow_edits_and_undo():
    scene, doc, sk, part = bracket()
    obj = doc.show(part)
    assert obj.cad_item is part.item
    n0 = obj.mesh.triangle_count
    lo, hi = obj.mesh.bounds()
    assert hi[0] == pytest.approx(0.08, abs=1e-6)
    doc.parameter("width", "120 mm")
    assert obj.mesh.bounds()[1][0] == pytest.approx(0.12, abs=1e-6)
    scene.history.undo()
    assert obj.mesh.bounds()[1][0] == pytest.approx(0.08, abs=1e-6)
    assert part.measure()["volume"] == pytest.approx(
        80 * 40 * 10 - 2 * math.pi * 16 * 10, rel=1e-13
    )
    scene.history.redo()
    assert obj.mesh.bounds()[1][0] == pytest.approx(0.12, abs=1e-6)
    assert obj.mesh.triangle_count >= n0 - 8


def test_driven_objects_refuse_direct_edits_until_converted():
    scene, doc, sk, part = bracket()
    obj = doc.show(part)
    for action in (lambda: setattr(obj, "mesh", None), obj.edit_mesh, obj.sculpt):
        with pytest.raises(CadError) as e:
            action()
        assert e.value.code == "CAD_DRIVEN"
    report = doc.convert_to_mesh(obj)
    assert report["item"] == "Bracket" and report["lost"]
    assert obj.cad_item is None
    mesh = obj.mesh
    doc.parameter("width", "90 mm")
    assert obj.mesh is mesh  # no longer follows
    obj.mesh = None  # and may be edited


def test_failures_are_atomic_and_located():
    scene, doc, sk, part = bracket()
    obj = doc.show(part)
    steps = len(scene.history.entries)
    mesh = obj.mesh
    with pytest.raises(CadError) as e:
        sk.dimension("length", "r3", value="70 mm", id="bad")
    assert e.value.code == "SKETCH_CONFLICT"
    assert e.value.details["conflicts"][0]["constraint"] == "bad"
    assert "bad" not in json.dumps(doc.item("Profile").definition)
    assert len(scene.history.entries) == steps and obj.mesh is mesh
    with pytest.raises(CadError) as e:
        part.fillet([{"edge": ["base:end", "base:side/missing"]}], 2)
    assert e.value.code == "REFERENCE_LOST"
    assert "features[2]" in e.value.location
    with pytest.raises(CadError) as e:
        doc.parameter("hole_d", "90 mm")  # a hole wider than the plate: hole off the face
    assert len(scene.history.entries) == steps


def test_cycles_unknown_items_and_items_in_use():
    scene, doc, sk, part = bracket()
    with pytest.raises(CadError) as e:
        doc.parameter("a", "=b + 1 mm")
    assert e.value.code == "UNKNOWN_PARAMETER"
    doc.parameter("a", "1 mm")
    doc.parameter("b", "=a * 2")
    with pytest.raises(CadError) as e:
        doc.parameter("a", "=b + 1 mm")
    assert e.value.code == "DEPENDENCY_CYCLE"
    with pytest.raises(CadError) as e:
        doc.remove("Profile")
    assert e.value.code == "ITEM_IN_USE"
    with pytest.raises(CadError) as e:
        doc.rename("width", "w2")
    assert e.value.code == "PARAMETER_IN_USE"


def test_strict_history_round_trips_every_step():
    history_module.STRICT = True
    try:
        scene, doc, sk, part = bracket()
        obj = doc.show(part)
        doc.parameter("width", "95 mm")
        part.fillet([{"edge": ["base:end", "base:side/r2"]}], 3, id="round")
        states = []
        while scene.history.can_undo:
            states.append(json.dumps(doc.to_json(), sort_keys=True))
            scene.history.undo()
        assert not doc.items()
        while scene.history.can_redo:
            scene.history.redo()
        assert part.measure()["valid"]
        assert json.dumps(doc.to_json(), sort_keys=True) == states[0]
        assert obj.cad_item is part.item
    finally:
        history_module.STRICT = False


def test_save_load_and_history_files(tmp_path):
    scene, doc, sk, part = bracket()
    obj = doc.show(part)
    path = tmp_path / "bracket.aether"
    scene.save(path, history=True)
    data = json.loads(path.read_text())
    assert data["format_version"] == 11
    assert {i["kind"] for i in data["scene"]["cad"]["items"]} == {"parameter", "sketch", "part"}
    loaded = aether3d.Scene.load(path, history=True)
    ldoc = loaded.cad
    lobj = loaded.get_by_name("Bracket")
    assert lobj.cad_item is not None and lobj.mesh.triangle_count == obj.mesh.triangle_count
    assert ldoc.result("Bracket").bodies["main"].summary()["faces"] == len(part.body().faces)
    ldoc.parameter("width", "70 mm")
    assert loaded.get_by_name("Bracket").mesh.bounds()[1][0] == pytest.approx(0.07, abs=1e-6)
    loaded.history.undo()
    assert loaded.get_by_name("Bracket").mesh.bounds()[1][0] == pytest.approx(0.08, abs=1e-6)
    while loaded.history.can_undo:
        loaded.history.undo()
    assert not loaded.cad.items()


def test_version_7_projects_migrate_to_an_empty_document(tmp_path):
    scene = aether3d.Scene("old")
    scene.add_cube(name="Cube")
    path = tmp_path / "old.aether"
    scene.save(path)
    data = json.loads(path.read_text())
    data["format_version"] = 7
    del data["scene"]["cad"]
    path.write_text(json.dumps(data))
    loaded = aether3d.Scene.load(path)
    assert loaded.cad.items() == []
    assert loaded.get_by_name("Cube").mesh is not None


def test_malformed_cad_sections_are_refused(tmp_path):
    scene, doc, sk, part = bracket()
    doc.show(part)
    path = tmp_path / "x.aether"
    scene.save(path)
    good = json.loads(path.read_text())
    for mutate in (
        lambda d: d["scene"]["cad"]["items"][0].update(kind="spaceship"),
        lambda d: d["scene"]["cad"]["items"][0].update(id="not-a-uuid"),
        lambda d: d["scene"]["cad"]["items"][-1].update(
            links=["00000000-0000-4000-8000-000000000000"]
        ),
        lambda d: d["scene"]["cad"]["items"][3]["definition"]["features"][0].update(type="warp"),
        lambda d: d["scene"]["cad"].update(items=5),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(AetherError):
            aether3d.Scene.load(path)


def test_removing_a_linked_object_unlinks_it_and_removing_items_removes_objects():
    scene, doc, sk, part = bracket()
    obj = doc.show(part)
    scene.remove(obj)
    assert part.item.links == ()
    obj2 = doc.show(part)
    with doc.batch("Remove the bracket"):
        doc.remove(part.item)
    assert obj2.scene is None
    scene.history.undo()
    assert obj2.scene is scene and obj2.cad_item is part.item
