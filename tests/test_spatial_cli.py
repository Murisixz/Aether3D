"""Stage 10: spatial queries from the command line and scene scripts give the Python results."""

from __future__ import annotations

import json

import pytest

import aether3d as a3
from aether3d import Scene
from aether3d.cli import main
from aether3d.spatial import Box, Sphere, run_query
from aether3d.spatial.request import QUERIES, parse_value, request_from_arguments, to_json


def query_scene() -> Scene:
    s = Scene("Queries")
    base = s.add_cube(name="Base", size=2)
    base.tags = ["castle"]
    base.metadata = {"floors": 2}
    s.add_sphere(name="Ball", radius=0.5, position=(0, 0, 2.0), parent=base)
    s.add_cube(name="Shed", size=1, position=(3, 0, 0.5))
    s.add_plane(name="Ground", size=20, position=(0, 0, -1))
    s.add_camera("Cam", position=(0, -10, 3), look_at=(0, 0, 0.5))
    return s


# Each case: the Python call, and the same query as CLI KEY=VALUE arguments.
CASES = [
    (lambda s: s.describe(), "describe", []),
    (lambda s: s.bounds("Base", children=True), "bounds", ["object=Base", "children=true"]),
    (lambda s: s.bounds(), "bounds", []),
    (lambda s: s.find_objects(tags="castle"), "find", ["tags=castle"]),
    (lambda s: s.find_objects(name="B*", kind="mesh"), "find", ["name=B*", "kind=mesh"]),
    (
        lambda s: s.raycast((0, 0, 10), (0, 0, -1)),
        "raycast",
        ["origin=0,0,10", "direction=0,0,-1"],
    ),
    (
        lambda s: s.raycast_all((0.1, 0.2, 10), (0, 0, -1), limit=3),
        "raycast_all",
        ["origin=[0.1,0.2,10]", "direction=0,0,-1", "limit=3"],
    ),
    (
        lambda s: s.raycast((0, 0, 10), (0, 0, -1), exclude=["Ball", "Base"]),
        "raycast",
        ["origin=0,0,10", "direction=0,0,-1", "exclude=Ball,Base"],
    ),
    (lambda s: s.closest_point((5, 5, 5)), "closest_point", ["point=5,5,5"]),
    (lambda s: s.nearest_objects((2, 0, 0), count=2), "nearest", ["point=2,0,0", "count=2"]),
    (lambda s: s.contains_point("Base", (0, 0, 0)), "contains", ["object=Base", "point=0,0,0"]),
    (lambda s: s.distance("Base", "Shed"), "distance", ["a=Base", "b=Shed"]),
    (lambda s: s.overlaps(), "overlaps", []),
    (lambda s: s.overlaps(exact=False), "overlaps", ["exact=false"]),
    (
        lambda s: s.query_region(Box((-1, -1, -1), (1, 1, 1))),
        "region",
        ['region={"box":{"min":[-1,-1,-1],"max":[1,1,1]}}'],
    ),
    (
        lambda s: s.query_region(Sphere((3, 0, 0.5), 0.2), mode="inside"),
        "region",
        ['region={"sphere":{"center":[3,0,0.5],"radius":0.2}}', "mode=inside"],
    ),
    (
        lambda s: s.camera_ray(10, 20, width=64, height=48),
        "camera_ray",
        ["x=10", "y=20", "width=64", "height=48"],
    ),
    (
        lambda s: s.pick(32, 24, width=64, height=48),
        "pick",
        ["x=32", "y=24", "width=64", "height=48"],
    ),
    (
        lambda s: s.project([(0, 0, 0), (3, 0, 0.5)], width=64, height=48),
        "project",
        ["points=[[0,0,0],[3,0,0.5]]", "width=64", "height=48"],
    ),
    (lambda s: s.objects_in_view(width=64, height=48), "in_view", ["width=64", "height=48"]),
    (
        lambda s: s.line_of_sight((0, -5, 0.5), (0, 5, 0.5)),
        "line_of_sight",
        ["start=0,-5,0.5", "end=0,5,0.5"],
    ),
    (
        lambda s: s.visible_from("Shed", "Cam", samples=16, width=64, height=48),
        "visible_from",
        ["target=Shed", "observer=Cam", "samples=16", "width=64", "height=48"],
    ),
    (
        lambda s: s.relation("Ball", "Base", near_distance=1.0, viewer="Cam"),
        "relation",
        ["a=Ball", "b=Base", "near_distance=1", "viewer=Cam"],
    ),
]


def test_every_query_has_a_case():
    assert {name for _, name, _ in CASES} == set(QUERIES)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_cli_and_python_agree(index, tmp_path, capsys):
    call, name, arguments = CASES[index]
    scene = query_scene()
    path = tmp_path / "q.aether"
    scene.save(path)
    expected = to_json(call(Scene.load(path)))
    assert main(["query", name, str(path), *arguments, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] and data["query"] == name
    assert data["result"] == expected  # (describe: both loaded scenes are at revision 0)
    # The same request through run_query and through a scene script.
    request = request_from_arguments(name, arguments)
    assert run_query(Scene.load(path), request) == {"query": name, "result": expected}
    loaded = Scene.load(path)
    revision = loaded.revision
    script = {
        "format": "aether3d.scene-script",
        "version": 1,
        "operations": [{"op": "query", **request}],
    }
    out = a3.run_scene_script(loaded, script)
    assert out["operations"][0]["result"] == expected
    assert loaded.revision == revision and len(loaded.history) == 0  # read-only: no step


def test_cli_text_output_and_batches(tmp_path, capsys):
    path = tmp_path / "q.aether"
    query_scene().save(path)
    assert main(["query", "raycast", str(path), "origin=0,0,10", "direction=0,0,-1"]) == 0
    assert capsys.readouterr().out.startswith("raycast: Ball at (0, 0, 2.5), distance 7.5")
    requests = tmp_path / "requests.json"
    requests.write_text(
        json.dumps(
            [
                {"query": "distance", "a": "Base", "b": "Shed"},
                {"query": "find", "metadata": {"floors": 2}},
            ]
        )
    )
    assert main(["query", "run", str(path), "--request", str(requests), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["query"] for r in data["results"]] == ["distance", "find"]
    found = data["results"][1]["result"]
    assert [(o["name"], o["kind"]) for o in found] == [("Base", "mesh")]


def test_cli_reports_query_errors(tmp_path, capsys):
    path = tmp_path / "q.aether"
    query_scene().save(path)
    assert main(["query", "contains", str(path), "object=Nope", "point=0,0,0", "--json"]) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "type": "QueryError",
        "message": "no object with id or name 'Nope' in this scene",
        "location": "object",
        "code": "OBJECT_NOT_FOUND",
    }
    assert main(["query", "raycast", str(path), "origin=0,0,10", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_REQUEST"
    assert main(["query", "raycast", str(path), "origin", "--json"]) == 1
    assert "KEY=VALUE" in json.loads(capsys.readouterr().out)["error"]["message"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"query": "bounds"}, {"query": "warp"}]))
    assert main(["query", "run", str(path), "--request", str(bad), "--json"]) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["location"] == "requests[1].query" and error["code"] == "INVALID_REQUEST"
    assert main(["query", "bounds", str(path), "--request", str(bad), "--json"]) == 1
    capsys.readouterr()


def test_queries_leave_the_project_file_untouched(tmp_path, capsys):
    path = tmp_path / "q.aether"
    query_scene().save(path)
    before = path.read_bytes()
    for _, name, arguments in CASES:
        assert main(["query", name, str(path), *arguments]) == 0
    capsys.readouterr()
    assert path.read_bytes() == before
    assert not (tmp_path / "q.aether.history").exists()


def test_command_line_values():
    assert parse_value("origin", "1,2,3") == [1.0, 2.0, 3.0]
    assert parse_value("origin", "[1, 2, 3]") == [1, 2, 3]
    assert parse_value("object", "1") == "1"  # an object may be named "1"
    assert parse_value("object", "Tower, east") == "Tower, east"
    assert parse_value("objects", "A,B") == ["A", "B"]
    assert parse_value("observer", "Cam") == "Cam"
    assert parse_value("limit", "3") == 3 and parse_value("exact", "false") is False
    assert parse_value("name", "Tower*") == "Tower*"
    with pytest.raises(a3.QueryError):
        request_from_arguments("raycast", ["origin=1,2,3", "origin=1,2,3"])
