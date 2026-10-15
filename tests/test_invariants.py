"""Invariants the performance work must keep: identity, exact undo/redo, coherent caches.

* **Identity.** Along a random chain of changes, undoing back to any state gives the
  same Python objects for every object and material that existed then (not equal
  copies), and redoing gives them again.
* **Coherent caches.** Everything a scene caches about itself (the spatial index, and
  any render state kept between frames) answers exactly as a fresh copy of the scene
  -- one rebuilt from its document, with no caches -- after every change, undo and
  redo.
* **Determinism.** World matrices are bitwise the same whichever way they are
  computed; extracting a frame twice gives identical render scenes.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest
from parity import assert_bitwise, assert_render_scenes_equal, query_fingerprint

import aether3d as a3
from aether3d import Scene, create_cube, create_sphere
from aether3d.rendering.scene_data import _mat4_mul, extract_render_scene, world_matrices

QUERIES = [
    {"query": "describe"},
    {"query": "raycast", "origin": [0.3, -20.0, 1.1], "direction": [0.0, 1.0, 0.0]},
    {"query": "raycast_all", "origin": [0.1, 0.2, 20.0], "direction": [0.0, 0.0, -1.0], "limit": 4},
    {"query": "nearest", "point": [1.0, 1.0, 1.0], "count": 3},
    {"query": "overlaps"},
    {"query": "bounds"},
]


IGNORE = ("revision",)
"""Keys that differ between equal states: ``scene.revision`` increases with every change,
undo and redo included, and a copy rebuilt from the document starts at 0."""


def doc(scene: Scene) -> str:
    return json.dumps(scene.to_dict(), sort_keys=True)


def fresh(scene: Scene) -> Scene:
    """The same scene rebuilt from its document: no caches, no history."""
    return Scene.from_dict(scene.to_dict())


def start_scene() -> Scene:
    s = Scene("Invariants")
    cube = s.add_cube(name="Cube", size=2, position=(0, 0, 1))
    s.add_sphere(name="Ball", parent=cube, position=(0, 0, 2))
    s.add_plane(name="Ground", size=20)
    s.add_camera(name="Cam", position=(8, -8, 5), look_at=(0, 0, 0))
    s.add_point_light(name="Lamp", position=(2, 2, 6))
    stone = s.create_material("Stone", base_color=(0.5, 0.5, 0.5))
    s.assign_material("Cube", stone)
    s.add_procedural("Tower", "cylinder", radius=1, depth=4, segments=8, position=(5, 0, 0))
    s.procedural.add_parameter("height", 3.0, min=1, max=10)
    s.history.clear()
    return s


def snapshot(scene: Scene) -> dict:
    camera = scene.active_camera
    render = None
    if camera is not None:
        render = extract_render_scene(scene, camera, aspect=4 / 3)
    return {
        "doc": doc(scene),
        "objects": {o.id: o for o in scene},
        "materials": {m.id: m for m in scene.materials.materials},
        "queries": query_fingerprint(scene, QUERIES, IGNORE),
        "render": render,
    }


def check_state(scene: Scene, expected: dict, against_fresh: bool) -> None:
    assert doc(scene) == expected["doc"]
    for oid, obj in expected["objects"].items():
        assert scene.get(oid) is obj, f"object {oid} is not the same Python object"
    for mid, material in expected["materials"].items():
        assert scene.materials.material(mid) is material
    assert query_fingerprint(scene, QUERIES, IGNORE) == expected["queries"]
    camera = scene.active_camera
    render = None if camera is None else extract_render_scene(scene, camera, aspect=4 / 3)
    if expected["render"] is not None:
        assert_render_scenes_equal(render, expected["render"], "extraction vs the recorded one")
    if against_fresh:
        copy = fresh(scene)
        assert query_fingerprint(copy, QUERIES, IGNORE) == expected["queries"], "cached vs fresh"
        if render is not None:
            assert_render_scenes_equal(
                render,
                extract_render_scene(copy, copy.active_camera, aspect=4 / 3),
                "extraction vs a fresh copy",
            )


def random_actions(s: Scene, rng: random.Random) -> list:
    def pick():
        return rng.choice(list(s))

    def move_many():
        with s.transaction("move many"):
            for obj in rng.sample(list(s), min(4, len(s))):
                obj.position = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 3))

    def plan():
        s.execute_plan(
            {
                "format": "aether3d.scene-plan",
                "version": 1,
                "meshes": {"m": {"primitive": "cube", "size": 0.4}},
                "objects": {
                    f"g{rng.randrange(10**6)}": {
                        "mesh": "@m",
                        "grid": {"count": [2, 2]},
                        "position": [rng.uniform(-4, 4), rng.uniform(-4, 4), 0.5],
                    }
                },
            }
        )

    def edit():
        target = rng.choice([o for o in s if o.mesh is not None and o.geometry is None] or [pick()])
        with target.edit_mesh() as e:
            faces = e.face_ids()
            if faces:
                e.extrude_faces([faces[0]], 0.1)

    return [
        lambda: s.add_cube(position=(rng.uniform(-5, 5), rng.uniform(-5, 5), 0)),
        lambda: s.remove(pick()),
        lambda: s.duplicate(pick()),
        lambda: setattr(pick(), "position", (rng.random(), rng.random(), rng.random())),
        lambda: setattr(pick(), "rotation_euler", (rng.uniform(0, 90), rng.uniform(0, 90), 0)),
        lambda: setattr(pick(), "scale", (rng.uniform(0.5, 2), rng.uniform(-2, -0.5), 1.0)),
        lambda: setattr(pick(), "name", f"N{rng.randrange(10**6)}"),
        lambda: pick().set_parent(rng.choice([None, pick()])),
        lambda: setattr(pick(), "visible", rng.random() < 0.5),
        lambda: setattr(pick(), "color", (rng.random(), rng.random(), rng.random())),
        lambda: s.assign_material(pick(), rng.choice([None, "Stone"])),
        lambda: setattr(s.materials["Stone"], "roughness", rng.random()),
        lambda: setattr(s.world, "background", (rng.random(), rng.random(), rng.random())),
        lambda: s.procedural.set_parameter("height", rng.uniform(1, 10)),
        lambda: s.procedural.geometry("Tower").set_source(segments=rng.randrange(3, 12)),
        lambda: setattr(pick(), "mesh", create_cube(rng.uniform(0.5, 2))),
        move_many,
        plan,
        edit,
    ]


@pytest.mark.parametrize("seed", [1, 2])
def test_identity_and_coherent_caches_along_a_random_chain(seed):
    rng = random.Random(seed)
    s = start_scene()
    actions = random_actions(s, rng)
    states = [snapshot(s)]
    steps = 0
    while steps < 30:
        before = doc(s)
        try:
            rng.choice(actions)()
        except a3.AetherError:
            assert doc(s) == before
            continue
        if doc(s) == before:
            continue
        steps += 1
        states.append(snapshot(s))
        check_state(s, states[-1], against_fresh=steps % 3 == 0)
    for k in range(steps, 0, -1):
        s.history.undo()
        check_state(s, states[k - 1], against_fresh=k % 4 == 0)
    for k in range(1, steps + 1):
        s.history.redo()
        check_state(s, states[k], against_fresh=k % 5 == 0)


def random_forest(n: int, seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    s = Scene("Forest")
    objs = []
    for k in range(n):
        parent = None
        if objs and rng.random() < 0.85:
            parent = objs[int(rng.integers(0, len(objs)))]
            if _depth(parent) >= 8:
                parent = None
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        scale = rng.uniform(0.3, 2.5, 3) * rng.choice([-1.0, 1.0], 3)
        obj = s.add_object(
            f"o{k}",
            mesh=None,
            parent=parent,
            position=tuple(float(v) for v in rng.uniform(-50, 50, 3)),
            rotation=tuple(float(v) for v in q),
            scale=tuple(float(v) for v in scale),
        )
        objs.append(obj)
    return s


def _depth(obj) -> int:
    d = 0
    while obj.parent is not None:
        obj = obj.parent
        d += 1
    return d


def renderer_reference(obj) -> np.ndarray:
    """``world_matrices``' arithmetic for one object: ``(root @ ...) @ local``, top-down,
    with the explicit fixed-order products of ``_mat4_mul``."""
    chain = [obj]
    while chain[-1].parent is not None:
        chain.append(chain[-1].parent)
    world = chain[-1].local_matrix
    for node in reversed(chain[:-1]):
        world = _mat4_mul(world, node.local_matrix)
    return world


@pytest.mark.parametrize("seed", [3, 4])
def test_world_matrices_are_bitwise_reproducible(seed):
    """The renderer's world matrices (renders, queries, picks) equal a per-object
    reference with the same arithmetic, bit for bit, also after reparenting.

    ``obj.world_matrix`` (the public property) multiplies with ``@`` from the object up
    (``root @ (parent @ local)``), a different association: it agrees to within a few
    ulps, not bitwise. Both are kept as they are; neither may change silently.
    """
    s = random_forest(300, seed)
    for _ in range(2):
        worlds = world_matrices(s)
        for obj in s:
            assert_bitwise(worlds[obj.id], renderer_reference(obj), f"world matrix of {obj.name}")
            np.testing.assert_allclose(worlds[obj.id], obj.world_matrix, rtol=1e-12, atol=1e-12)
        with s.transaction("reparent"):
            for obj in list(s)[::17]:
                obj.set_parent(None, keep_world_transform=True)


def test_extraction_is_deterministic_and_matches_a_fresh_copy():
    s = start_scene()
    s.add_object("Mirrored", mesh=create_sphere(0.5, 12, 6), scale=(-1.0, 1.0, 2.0))
    camera = s.get_by_name("Cam")
    first = extract_render_scene(s, camera, aspect=16 / 9)
    assert_render_scenes_equal(first, extract_render_scene(s, camera, aspect=16 / 9))
    copy = fresh(s)
    assert_render_scenes_equal(
        first, extract_render_scene(copy, copy.get_by_name("Cam"), aspect=16 / 9)
    )
    assert not math.isnan(float(first.instances[0].world[0, 0]))
