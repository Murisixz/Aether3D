"""A scene is freed when the last reference to it goes, whatever it was used for.

Caches that outlive one call (the spatial index, plan idempotency records, render and
traversal caches) must never keep a scene alive: a long-running process -- the MCP
server, a notebook -- opens and drops many scenes. Each case builds a fresh scene, uses
it once, drops it and checks a weak reference after a full collection.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from aether3d import Expr, Scene, create_cube, create_sphere
from aether3d.spatial import Box, Sphere


def build() -> Scene:
    s = Scene("Lifecycle")
    s.add_object("Base", mesh=create_cube(2.0))
    s.add_object("Ball", mesh=create_sphere(0.5, 16, 8), position=(0.0, 0.0, 1.5))
    s.add_object("Shed", mesh=create_cube(1.0), position=(3.0, 0.0, 0.5))
    s.add_directional_light("Sun", direction=(-0.3, 0.5, -1.0))
    s.add_camera("Cam", position=(0.0, -8.0, 2.0)).frame(aspect=64 / 48)
    return s


def assert_freed(use) -> None:
    scene = build()
    use(scene)
    ref = weakref.ref(scene)
    del scene
    gc.collect()
    if ref() is not None:
        holders = sorted({type(r).__name__ for r in gc.get_referrers(ref())})
        pytest.fail(f"the scene is still alive; held by: {', '.join(holders)}")


QUERIES = {
    "describe": lambda s: s.describe(),
    "bounds": lambda s: s.bounds("Base", children=True),
    "find": lambda s: s.find_objects(name="B*", kind="mesh"),
    "raycast": lambda s: s.raycast((0, 0, 10), (0, 0, -1)),
    "raycast_all": lambda s: s.raycast_all((0.1, 0.2, 10), (0, 0, -1), limit=3),
    "closest_point": lambda s: s.closest_point((5, 5, 5)),
    "nearest": lambda s: s.nearest_objects((2, 0, 0), count=2),
    "contains": lambda s: s.contains_point("Base", (0, 0, 0)),
    "distance": lambda s: s.distance("Base", "Shed"),
    "overlaps": lambda s: s.overlaps(),
    "region": lambda s: s.query_region(Box((-1, -1, -1), (1, 1, 1))),
    "region_sphere": lambda s: s.query_region(Sphere((3, 0, 0.5), 0.2), mode="inside"),
    "camera_ray": lambda s: s.camera_ray(10, 20, width=64, height=48),
    "pick": lambda s: s.pick(32, 24, width=64, height=48),
    "project": lambda s: s.project([(0, 0, 0), (3, 0, 0.5)], width=64, height=48),
    "in_view": lambda s: s.objects_in_view(width=64, height=48),
    "line_of_sight": lambda s: s.line_of_sight((0, -5, 0.5), (0, 5, 0.5)),
    "visible_from": lambda s: s.visible_from("Shed", "Cam", samples=16, width=64, height=48),
    "relation": lambda s: s.relation("Ball", "Base", near_distance=1.0, viewer="Cam"),
}


@pytest.mark.parametrize("name", sorted(QUERIES))
def test_a_queried_scene_is_freed(name):
    assert_freed(QUERIES[name])


def test_a_scene_queried_at_many_revisions_is_freed():
    def use(s):
        for k in range(3):
            s.raycast((0, 0, 10), (0, 0, -1))
            s.get_by_name("Ball").position = (0.0, float(k), 1.5)
        s.history.undo()
        s.raycast((0, 0, 10), (0, 0, -1))

    assert_freed(use)


def test_json_queries_free_the_scene():
    from aether3d.spatial.request import run_query

    assert_freed(
        lambda s: run_query(s, {"query": "raycast", "origin": [0, 0, 10], "direction": [0, 0, -1]})
    )


PLAN = {
    "format": "aether3d.scene-plan",
    "version": 1,
    "meshes": {"block": {"primitive": "cube", "size": 0.5}},
    "objects": {"blocks": {"mesh": "@block", "grid": {"count": [3, 3]}, "position": [0, 0, 3]}},
}


def test_plans_free_the_scene():
    assert_freed(lambda s: s.execute_plan(PLAN))


def test_plans_with_an_idempotency_key_and_queries_free_the_scene():
    plan = {
        **PLAN,
        "idempotency_key": "lifecycle-1",
        "queries": [{"query": "raycast", "origin": [0, 0, 10], "direction": [0, 0, -1]}],
    }

    def use(s):
        s.execute_plan(plan)
        s.execute_plan(plan)  # recognized as already applied

    assert_freed(use)


def test_rendering_frees_the_scene():
    assert_freed(lambda s: s.render(width=24, height=16, backend="cpu"))


def test_path_tracing_frees_the_scene():
    assert_freed(
        lambda s: s.render(width=8, height=6, backend="path", path={"samples": 1, "bounces": 1})
    )


def test_editing_frees_the_scene():
    def use(s):
        with s.get_by_name("Base").edit_mesh() as edit:
            edit.extrude_faces(edit.select_faces(normal=((0, 0, 1), 1), expect=1), 0.1)

    assert_freed(use)


def test_procedural_geometry_frees_the_scene():
    def use(s):
        s.procedural.add_parameter("height", 2.0, min=1, max=5)
        geometry = s.add_geometry("Tower", "cylinder", radius=0.5, depth=Expr("height"))
        geometry.add_modifier("transform", translation=(4.0, 0.0, 0.0))
        s.procedural.set_parameter("height", 3.0)
        s.raycast((0, 0, 10), (0, 0, -1))

    assert_freed(use)


def test_history_frees_the_scene():
    def use(s):
        with s.transaction("Move"):
            s.get_by_name("Ball").position = (1.0, 1.0, 1.0)
        s.history.undo()
        s.history.redo()
        s.history.checkpoint("here")

    assert_freed(use)


def test_saving_loading_and_gltf_free_the_scene(tmp_path):
    def use(s):
        s.save(tmp_path / "a.aether")
        s.export(tmp_path / "a.glb")

    assert_freed(use)
    loaded = Scene.load(tmp_path / "a.aether")
    ref = weakref.ref(loaded)
    loaded.raycast((0, 0, 10), (0, 0, -1))
    del loaded
    gc.collect()
    assert ref() is None


def test_gpu_rendering_frees_the_scene(gpu):
    assert_freed(lambda s: gpu.render(s, width=24, height=16))


def test_gpu_path_tracing_frees_the_scene(gpu):
    assert_freed(
        lambda s: s.render(
            width=8,
            height=6,
            backend="gpu_path",
            adapter=gpu.adapter,
            path={"samples": 1, "bounces": 1},
        )
    )
