"""Declarative JSON edit scripts: execution, selections, errors with locations, atomicity."""

import copy

import pytest

from aether3d import Scene, ScriptError, inspect_mesh, run_edit_script


def script(*operations, obj="Block", **extra):
    return {
        "format": "aether3d.edit-script",
        "version": 1,
        "edits": [{"object": obj, "operations": list(operations), **extra}],
    }


@pytest.fixture
def scene():
    s = Scene()
    s.add_cube("Block", size=2)
    s.add_object("Twin", mesh=s.get_by_name("Block").mesh, position=(3, 0, 0))
    return s


def test_full_script(scene):
    result = run_edit_script(
        scene,
        script(
            {"op": "select", "kind": "faces", "normal": [[0, 0, 1], 1], "expect": 1, "as": "top"},
            {
                "op": "extrude_faces",
                "faces": {"selection": "top"},
                "distance": 1,
                "save": {"walls": "side_faces"},
            },
            {"op": "inset_faces", "faces": {"selection": "top"}, "thickness": 0.25},
            {"op": "translate", "elements": {"selection": "top"}, "offset": [0, 0, 0.5]},
            {
                "op": "bevel_edges",
                "edges": {"select": {"kind": "edges", "nearest": [1, 1, -0.5]}},
                "width": 0.1,
            },
            {"op": "summary"},
            {"op": "validate", "require_ok": True},
        ),
    )
    (edit,) = result.edits
    assert edit["object"]["name"] == "Block"
    assert [o["op"] for o in edit["operations"]] == [
        "select",
        "extrude_faces",
        "inset_faces",
        "translate",
        "bevel_edges",
        "summary",
        "validate",
    ]
    assert len(edit["selections"]["walls"]["faces"]) == 4
    assert edit["commit"]["changed"] and edit["commit"]["validation"]["status"] == "ok"
    block, twin = scene.get_by_name("Block"), scene.get_by_name("Twin")
    assert block.mesh is not twin.mesh  # default scope: only the target object
    assert inspect_mesh(block.mesh).topology["volume"] > 8


def test_shared_scope_and_create(scene):
    run_edit_script(
        scene,
        script(
            {"op": "translate", "elements": {"select": {"kind": "vertices"}}, "offset": [0, 0, 1]},
            scope="shared",
        ),
    )
    assert scene.get_by_name("Block").mesh is scene.get_by_name("Twin").mesh
    run_edit_script(
        scene,
        script({"op": "add_box", "size": [1, 2, 3]}, obj="Fresh", create=True),
    )
    assert inspect_mesh(scene.get_by_name("Fresh").mesh).topology["volume"] == pytest.approx(6)


def test_selection_queries_and_combine(scene):
    result = run_edit_script(
        scene,
        script(
            {"op": "select_all", "as": "all"},
            {
                "op": "select",
                "kind": "faces",
                "normal": {"direction": [0, 0, 1], "max_angle": 1},
                "as": "top",
            },
            {"op": "region_boundary", "faces": {"selection": "top"}, "as": "rim"},
            {"op": "edge_ring", "edge": 0, "as": "ring"},
            {"op": "combine", "difference": ["all", "top"], "as": "rest"},
            {"op": "to_vertices", "selection": {"selection": "top"}, "as": "top_verts"},
            {"op": "describe", "kinds": ["faces"], "limit": 2},
        ),
    )
    sels = result.edits[0]["selections"]
    assert len(sels["rim"]["edges"]) == 4 and len(sels["top_verts"]["vertices"]) == 4
    assert len(sels["rest"]["faces"]) == 5 and len(sels["ring"]["edges"]) == 4
    assert len(result.edits[0]["operations"][-1]["elements"]["faces"]) == 2


def test_undo_redo_steps(scene):
    result = run_edit_script(
        scene,
        script(
            {"op": "translate", "elements": {"select": {"kind": "vertices"}}, "offset": [5, 0, 0]},
            {"op": "undo"},
        ),
    )
    assert not result.edits[0]["commit"]["changed"]
    assert scene.get_by_name("Block").mesh is scene.get_by_name("Twin").mesh


@pytest.mark.parametrize(
    "bad, location, code",
    [
        ({"op": "explode"}, "edits[0].operations[0].op", "UNKNOWN_OPERATION"),
        (
            {"op": "extrude_faces", "faces": [0], "distanse": 1},
            "edits[0].operations[0].distanse",
            "INVALID_ARGUMENT",
        ),
        (
            {"op": "extrude_faces", "faces": [99], "distance": 1},
            "edits[0].operations[0]",
            "INVALID_ELEMENT",
        ),
        (
            {"op": "extrude_faces", "faces": {"selection": "nope"}, "distance": 1},
            "edits[0].operations[0].faces.selection",
            "UNKNOWN_SELECTION",
        ),
        (
            {"op": "loop_cut", "edge": {"select": {"kind": "edges"}}},
            "edits[0].operations[0].edge",
            "AMBIGUOUS_SELECTION",
        ),
        (
            {"op": "select", "kind": "faces", "nearest": [0, 0, 0], "as": "x"},
            "edits[0].operations[0]",
            "AMBIGUOUS_SELECTION",
        ),
        (
            {"op": "select", "kind": "faces", "as": "x", "colour": "red"},
            "edits[0].operations[0]",
            "INVALID_SCRIPT",
        ),
        (
            {"op": "inset_faces", "faces": [0], "thickness": 10},
            "edits[0].operations[0]",
            "INSET_TOO_LARGE",
        ),
        (
            {"op": "extrude_faces", "faces": [0], "distance": "far"},
            "edits[0].operations[0]",
            "INVALID_ARGUMENT",
        ),
        (
            {"op": "validate", "require_ok": True, "extra": 1},
            "edits[0].operations[0]",
            "INVALID_SCRIPT",
        ),
        ("not an object", "edits[0].operations[0]", "INVALID_SCRIPT"),
    ],
)
def test_errors_have_locations_and_codes(scene, bad, location, code):
    with pytest.raises(ScriptError) as info:
        run_edit_script(scene, script(bad))
    assert info.value.location == location and info.value.code == code


@pytest.mark.parametrize(
    "mutate, location",
    [
        (lambda s: s.update(format="other"), "script.format"),
        (lambda s: s.update(version=2), "script.version"),
        (lambda s: s.update(edits=[]), "script.edits"),
        (lambda s: s["edits"][0].pop("object"), "edits[0]"),
        (lambda s: s["edits"][0].update(object="Missing"), "edits[0].object"),
        (lambda s: s["edits"][0].update(scope="galaxy"), "edits[0].scope"),
        (lambda s: s.update(extra=True), "script"),
    ],
)
def test_malformed_scripts(scene, mutate, location):
    doc = copy.deepcopy(script({"op": "summary"}))
    mutate(doc)
    with pytest.raises(ScriptError) as info:
        run_edit_script(scene, doc)
    assert info.value.location == location


def test_script_is_all_or_nothing(scene):
    block = scene.get_by_name("Block")
    original = block.mesh
    count = len(scene)
    doc = {
        "format": "aether3d.edit-script",
        "version": 1,
        "edits": [
            {
                "object": "Block",
                "operations": [
                    {
                        "op": "translate",
                        "elements": {"select": {"kind": "vertices"}},
                        "offset": [1, 0, 0],
                    }
                ],
            },
            {"object": "New", "create": True, "operations": [{"op": "add_box"}]},
            {"object": "Twin", "operations": [{"op": "explode"}]},
        ],
    }
    with pytest.raises(ScriptError) as info:
        run_edit_script(scene, doc)
    assert info.value.location == "edits[2].operations[0].op"
    assert block.mesh is original and len(scene) == count  # earlier edits rolled back


def test_scripts_never_evaluate_code(scene):
    for name in ("__class__", "_replace_face", "_rollback", "copy", "to_mesh", "commit"):
        with pytest.raises(ScriptError, match="UNKNOWN_OPERATION"):
            run_edit_script(scene, script({"op": name}))
