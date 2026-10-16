"""Sculpt scripts (atomic, limits, structured errors), validity tiers at commit,
symmetrize and cancellation."""

import copy

import numpy as np
import pytest

from aether3d import Mesh, Scene, SculptError, create_plane, create_sphere
from aether3d.sculpt import SculptMesh, check_mesh
from aether3d.sculpt.script import SculptScriptError, check_script, run_sculpt_script
from aether3d.sculpt.validate import TIERS

STROKE = {
    "op": "stroke",
    "brush": {"kind": "draw", "radius": 0.4, "strength": 0.3},
    "path": [[0.3, -0.95, 0.2], [0.3, -0.95, 0.4]],
}


def scene_with_ball() -> Scene:
    scene = Scene()
    scene.add_object("Ball", mesh=create_sphere(1.0, 32, 16))
    scene.add_object("Sheet", mesh=create_plane(2.0))
    return scene


def script(*operations, obj="Ball", **entry) -> dict:
    return {
        "format": "aether3d.sculpt-script",
        "version": 1,
        "sculpts": [{"object": obj, "operations": list(operations), **entry}],
    }


def test_a_script_is_one_undo_step_with_results():
    scene = scene_with_ball()
    before = scene.get_by_name("Ball").mesh
    out = run_sculpt_script(
        scene,
        script(
            STROKE,
            {"op": "set_mask", "value": 1.0, "box": [[-2, -2, -2], [2, 2, 0]]},
            {**STROKE, "brush": {**STROKE["brush"], "symmetry": "x"}},
            {"op": "check"},
            {"op": "undo"},
            {"op": "redo"},
            require="closed",
        ),
    )
    entry = out["sculpts"][0]
    assert entry["object"]["name"] == "Ball"
    assert [s["op"] for s in entry["operations"]] == [
        "stroke",
        "set_mask",
        "stroke",
        "check",
        "undo",
        "redo",
    ]
    assert entry["operations"][0]["data"]["moved_vertices"] > 0
    assert entry["operations"][3]["report"]["summary"]["tier"] == "closed"
    commit = entry["commit"]
    assert commit["changed"] and commit["report"]["summary"]["tier"] == "closed"
    ball = scene.get_by_name("Ball")
    assert ball.mesh is not before
    assert scene.history.entries[-1].label.startswith("Sculpt script")
    scene.history.undo()
    assert ball.mesh is before


def test_failures_change_nothing_and_say_where():
    scene = scene_with_ball()
    ball = scene.get_by_name("Ball")
    before, steps = ball.mesh, len(scene.history)
    bad = copy.deepcopy(STROKE)
    bad["brush"]["radius"] = -1
    cases = [
        (script(STROKE, bad), "sculpts[0].operations[1]", None),
        (
            script(STROKE, {"op": "remesh", "target_length": 0.05, "colour": 1}),
            None,
            "INVALID_ARGUMENT",
        ),
        (
            script(STROKE, {"op": "voxel_remesh", "voxel_size": 0.1}, obj="Sheet"),
            None,
            "VOXEL_NOT_WATERTIGHT",
        ),
        (script(STROKE, obj="Nobody"), "sculpts[0].object", "OBJECT_NOT_FOUND"),
        (script({"op": "explode"}), "sculpts[0].operations[0].op", "INVALID_SCRIPT"),
        (script(STROKE, require="solid", obj="Sheet"), "sculpts[0].require", "BELOW_VALIDITY_TIER"),
    ]
    for doc, location, code in cases:
        with pytest.raises(SculptScriptError) as err:
            run_sculpt_script(scene, doc)
        if location:
            assert err.value.location.startswith(location), err.value.location
        if code:
            assert err.value.code == code, err.value.to_dict()
        assert ball.mesh is before and len(scene.history) == steps
    # A failure in the second object rolls back the first one too.
    two = script(STROKE)
    two["sculpts"].append(
        {"object": "Sheet", "operations": [{"op": "voxel_remesh", "voxel_size": 0.1}]}
    )
    with pytest.raises(SculptScriptError):
        run_sculpt_script(scene, two)
    assert ball.mesh is before and len(scene.history) == steps


def test_limits_stop_the_script_and_roll_back():
    scene = scene_with_ball()
    before = scene.get_by_name("Ball").mesh
    many = {**STROKE, "path": [[0.3, -0.95, z] for z in np.linspace(-0.5, 0.5, 50).tolist()]}
    doc = script(many)
    doc["limits"] = {"max_dabs": 5}
    with pytest.raises(SculptScriptError) as err:
        run_sculpt_script(scene, doc)
    assert err.value.code == "BUDGET_EXCEEDED"
    doc = script({"op": "remesh", "target_length": 0.05})
    doc["limits"] = {"max_triangles": 1000}
    with pytest.raises(SculptScriptError) as err:
        run_sculpt_script(scene, doc)
    assert err.value.code == "BUDGET_EXCEEDED"
    doc = script(many)
    doc["limits"] = {"max_seconds": 1e-9}
    with pytest.raises(SculptScriptError) as err:
        run_sculpt_script(scene, doc)
    assert err.value.code == "BUDGET_EXCEEDED"
    assert scene.get_by_name("Ball").mesh is before
    for bad in ({"max_dabs": 0}, {"max_dabs": 1.5}, {"max_seconds": float("inf")}, {"fast": 1}):
        doc = script(STROKE)
        doc["limits"] = bad
        with pytest.raises(SculptScriptError):
            check_script(doc)


def test_scripts_are_data_only():
    # Strings that look like code are only ever names or rejected values.
    scene = scene_with_ball()
    for evil in ("__import__('os').system('true')", "lambda: 0"):
        with pytest.raises(SculptScriptError):
            run_sculpt_script(scene, script({"op": evil}))
        with pytest.raises(SculptScriptError):
            run_sculpt_script(
                scene,
                script(
                    {"op": "stroke", "brush": {"kind": evil, "radius": 0.1}, "path": [[0, 0, 1]]}
                ),
            )
        with pytest.raises(SculptScriptError):
            run_sculpt_script(scene, script({"op": "symmetrize", "axis": evil}))


def test_tiers_are_ordered_and_measured():
    assert TIERS == ("valid", "manifold", "closed", "solid")
    ball = create_sphere(1.0, 16, 8)
    assert check_mesh(ball.positions, ball.triangles).summary["tier"] == "closed"
    report = check_mesh(ball.positions, ball.triangles, self_intersections=True)
    assert report.summary["tier"] == "solid"
    plane = create_plane(1.0)
    assert (
        check_mesh(plane.positions, plane.triangles, self_intersections=True).summary["tier"]
        == "manifold"
    )
    flipped = check_mesh(ball.positions, ball.triangles[:, ::-1], self_intersections=True)
    assert flipped.summary["tier"] == "closed"  # inside out: no positive volume
    p = np.concatenate([ball.positions, ball.positions + np.float32([0.5, 0, 0])])
    t = np.concatenate([ball.triangles, ball.triangles + len(ball.positions)])
    crossing = check_mesh(p, t, self_intersections=True)
    assert crossing.summary["tier"] == "closed"


def test_symmetrize_copies_one_side_exactly():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 32, 16))
    sm.stroke({"brush": {"kind": "draw", "radius": 0.5, "strength": 0.5}, "path": [[0.7, -0.7, 0]]})
    r = sm.symmetrize("x", direction="+")
    assert r.data["projected"] == 0 and r.data["paired"] > 0
    p = sm.positions()
    mirrored = p * [-1, 1, 1]
    # Every vertex has an exact mirror image among the vertices.
    keys = {tuple(x) for x in np.round(p, 12).tolist()}
    assert all(tuple(x) in keys for x in np.round(mirrored, 12).tolist())
    sm.undo()
    assert not sm.has_changes or sm.history[-1]["operation"] == "stroke"


def test_symmetrize_projects_when_the_topology_differs():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 24, 12))
    sm.stroke(
        {
            "brush": {
                "kind": "draw",
                "radius": 0.4,
                "strength": 0.3,
                "dyntopo": {"detail": 0.05, "mode": "constant"},
            },
            "path": [[0.7, -0.7, 0.1]],
        }
    )
    r = sm.symmetrize("x", direction="-")
    assert r.data["projected"] > 0
    assert any(d.code == "SYMMETRIZE_PROJECTED" for d in r.diagnostics)
    out = sm.to_mesh()
    assert check_mesh(out.positions, out.triangles).ok


def test_operations_stop_when_asked_and_roll_back():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 32, 16))
    calls = []

    def stop_at(n):
        calls.clear()
        sm.should_stop = lambda: calls.append(1) or len(calls) >= n

    stop_at(4)  # in the middle of the stroke, after three dabs moved vertices
    with pytest.raises(SculptError) as err:
        sm.stroke({"brush": {"kind": "draw", "radius": 0.3}, "path": [[0, 0, 1], [1, 0, 0]]})
    assert err.value.code == "CANCELLED"
    assert not sm.has_changes
    stop_at(3)
    with pytest.raises(SculptError):
        sm.remesh(0.05)
    stop_at(2)
    with pytest.raises(SculptError):
        sm.voxel_remesh(0.03)
    assert not sm.has_changes and sm.to_mesh() is sm._source and not sm.history


def test_commit_refuses_results_below_the_required_tier():
    scene = Scene()
    obj = scene.add_object("Sheet", mesh=create_plane(2.0))
    before = obj.mesh
    session = obj.sculpt()
    session.stroke({"brush": {"kind": "draw", "radius": 2.0}, "path": [[0, 0, 0]]})
    with pytest.raises(SculptError) as err:
        session.commit(require="closed")
    assert err.value.code == "BELOW_VALIDITY_TIER"
    assert obj.mesh is before and not session.closed
    assert session.commit(require="manifold").changed


def test_mesh_inputs_with_problems_report_their_tier():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0]], dtype=np.float32)
    t = np.array([[0, 1, 2], [0, 1, 3]])
    s = check_mesh(Mesh(p[:3], t[:1]).positions, t[:1]).summary
    assert s["tier"] == "manifold"
    s = check_mesh(p, t).summary
    assert s["tier"] is None  # not even valid: a position is not finite
