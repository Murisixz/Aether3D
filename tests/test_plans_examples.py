"""The example plans shipped in the package, and the guide's examples, build what they say."""

import json
import re
from importlib import resources

import pytest

from aether3d import PlanError, Scene, validate_plan
from aether3d.plans import select_objects

PACKAGE = resources.files("aether3d.plans")
EXAMPLES = {"basics": 44, "castle": 2190, "village": 11270}  # objects each one makes


def example(name):
    return json.loads((PACKAGE / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def test_the_package_ships_every_example_and_nothing_else():
    shipped = {p.name for p in (PACKAGE / "examples").iterdir() if p.name.endswith(".json")}
    assert shipped == {f"{name}.json" for name in EXAMPLES}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_examples_build_cleanly(name):
    plan = example(name)
    static = validate_plan(plan)
    assert static.ok and [w["code"] for w in static.warnings] == ["NOT_BOUND"]
    assert static.estimate["objects"] >= EXAMPLES[name]

    scene = Scene(name)
    result = scene.execute_plan(plan)
    assert result.ok and result.status == "applied" and not result.warnings
    assert result.counts["objects"] == len(scene) == EXAMPLES[name]
    assert all("error" not in q for q in result.queries), result.queries
    assert scene.active_camera is not None
    # one undo step, and the scene comes back
    assert len(scene.history.undo()) == 1 and len(scene) == 0
    assert len(scene.history.redo()) == 1 and len(scene) == EXAMPLES[name]


def test_the_castle_is_made_of_shared_blocks_with_real_openings():
    scene = Scene("Castle")
    result = scene.execute_plan(example("castle"))
    overlaps, gate = result.queries
    assert overlaps["result"] == []  # no courtyard tree grows into the keep
    assert gate["result"]["object"]["name"] == "keep/door"  # seen through the gate
    parts = ("front", "back", "left", "right", "crenels", "wall_", "battlements", "gate_lintel")
    blocks = [o for o in scene if any(part in o.name for part in parts)]
    assert len(blocks) > 2000 and len({id(o.mesh) for o in blocks}) == 1  # one shared cube
    assert result.counts["mesh_objects_sharing_a_mesh"] >= 2000
    names = {o.name for o in scene}
    assert "Tower SW/front 38" not in names  # a window, skipped by "where"
    assert "Tower SW/front 37" in names
    assert not [o.name for o in scene if not o.visible]


def test_the_village_reuses_structures():
    scene = Scene("Village")
    result = scene.execute_plan(example("village"))
    overlaps, nearest = result.queries
    assert overlaps["result"] == []  # no car hits a lamp post
    assert len(nearest["result"]) == 3
    assert result.counts["meshes"] == 20  # 11k objects, 20 meshes
    assert len(select_objects(scene, {"tags": ["house"]})) == 62 * 9
    fences = select_objects(scene, {"tags": ["fence"]})
    assert 62 * 9 * 7 <= len(fences) < 62 * 9 * 9  # every fence has a gate
    assert len(select_objects(scene, {"tags": ["car"]})) == 38 * 9 - 69  # (7x + 3y) % 5 == 0: free
    # deterministic: the same plan makes the same village
    again = Scene("Again")
    again.execute_plan(example("village"))
    assert [(o.name, o.position) for o in again] == [(o.name, o.position) for o in scene]


def guide_blocks():
    text = (PACKAGE / "guide.md").read_text(encoding="utf-8")
    return re.findall(r"```json\n(.*?)```", text, flags=re.S)


def test_the_guides_examples_work():
    first, prefab = guide_blocks()
    scene = Scene("Guide")
    result = scene.execute_plan(json.loads(first))
    assert result.ok and result.queries[0]["result"] == []
    fragment = json.loads("{" + prefab + "}")  # the prefab snippet, with what it refers to
    plan = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "meshes": {"box": {"primitive": "box", "size": [6, 6, 3]}, "roof": {"primitive": "cone"}},
        "materials": {"plaster": {"base_color": "#e8dcc4"}, "tiles": {"base_color": "#9a3b2c"}},
        **fragment,
        "queries": [{"query": "overlaps", "objects": "@street/roof"}],
    }
    result = Scene("Street").execute_plan(plan)
    assert result.counts["objects"] == 12 * 3


def test_a_prefab_is_not_an_object():
    plan = {
        "format": "aether3d.scene-plan",
        "version": 1,
        "prefabs": {"hut": {"children": {"roof": {"mesh": {"primitive": "cone"}}}}},
        "objects": {"huts": {"prefab": "@hut", "repeat": 2}},
        "operations": [{"op": "update", "select": "@hut/roof", "set": {"tags": ["x"]}}],
        "queries": [{"query": "overlaps", "objects": "@huts/roof", "against": ["@hut/roof"]}],
    }
    expected = {
        ("INVALID_REFERENCE", "/operations/0/select/refs/0"),
        ("INVALID_REFERENCE", "/queries/0/against/0"),  # found before anything is built
    }
    static = validate_plan(plan)
    assert not static.ok and {(i["code"], i["pointer"]) for i in static.issues} == expected
    assert all("@object/child" in i["hint"] for i in static.issues)
    scene = Scene("Huts")
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan)
    assert {(i["code"], i["pointer"]) for i in info.value.issues} == expected
    assert len(scene) == 0
