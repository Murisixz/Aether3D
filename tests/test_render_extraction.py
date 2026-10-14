"""Render-scene extraction computes matrices in batches, bitwise equal to the per-object rules.

``extract_render_scene`` composes the local matrices of all objects at once,
multiplies parents level by level, and inverts all normal matrices in one LAPACK
call. These tests compare every result, bit for bit, with the original
per-object computation (kept below as the reference): ``obj.local_matrix``,
the explicit 4x4 product ``_mat4_mul``, ``normal_matrix`` and ``to_linear``.
"""

import numpy as np
import pytest

from aether3d import Scene, create_sphere
from aether3d.rendering.color import to_linear
from aether3d.rendering.scene_data import (
    _mat4_mul,
    extract_render_scene,
    normal_matrix,
    resolve_camera,
    world_matrices,
)


def reference_world_matrices(scene):
    result = {}
    for obj in scene:
        local = obj.local_matrix
        parent = obj.parent
        result[obj.id] = local if parent is None else _mat4_mul(result[parent.id], local)
    return result


def random_scene(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    scene = Scene(f"Random {seed}")
    mesh = create_sphere(0.5, 12, 6)
    objects = []
    for k in range(120):
        parent = objects[rng.integers(len(objects))] if objects and rng.random() < 0.7 else None
        obj = scene.add_object(f"O{k}", mesh=mesh if rng.random() < 0.8 else None, parent=parent)
        obj.position = rng.normal(0, 50, 3)
        if rng.random() < 0.5:
            obj.rotation_euler = rng.uniform(-180, 180, 3)
        else:
            obj.rotation = rng.normal(0, 1, 4)  # normalized on assignment
        scale = rng.uniform(0.2, 3.0, 3) * rng.choice([-1.0, 1.0], 3)
        if rng.random() < 0.05:
            scale[rng.integers(3)] = 0.0  # a collapsed object
        obj.scale = scale
        obj.color = tuple(rng.random(3))
        obj.visible = rng.random() < 0.9
        objects.append(obj)
    pivot = scene.add_object("Pivot", position=(3, 4, 5), scale=(2, -1, 0.5))
    pivot.rotation_euler = (10, 20, 30)
    scene.add_directional_light("Sun", parent=pivot, direction=(0.2, 0.5, -1))
    scene.add_point_light("Bulb", parent=pivot, position=(1, 2, 3))
    scene.add_camera("Cam", position=(0, -200, 50), look_at=(0, 0, 0))
    return scene


@pytest.mark.parametrize("seed", range(6))
def test_batched_extraction_is_bitwise_equal_to_the_per_object_rules(seed):
    scene = random_scene(seed)
    reference = reference_world_matrices(scene)
    batched = world_matrices(scene)
    assert batched.keys() == reference.keys()
    for key, matrix in reference.items():
        assert np.array_equal(batched[key], matrix), key  # bit for bit, not approximately
    rs = extract_render_scene(scene, resolve_camera(scene, None))
    by_id = {obj.id: obj for obj in scene}
    expected = [
        obj
        for obj in scene
        if obj.mesh is not None
        and obj.visible
        and normal_matrix(reference[obj.id][:3, :3]) is not None
    ]
    assert [i.object_id for i in rs.instances] == [o.id for o in expected]
    for inst in rs.instances:
        world = reference[inst.object_id]
        assert np.array_equal(inst.world, world)
        assert np.array_equal(inst.normal_matrix, normal_matrix(world[:3, :3]))
        assert inst.mirrored == bool(np.linalg.det(world[:3, :3]) < 0.0)
        assert np.array_equal(inst.base_color, to_linear(by_id[inst.object_id].appearance.color))
    collapsed = [
        o.name
        for o in scene
        if o.mesh is not None and o.visible and normal_matrix(reference[o.id][:3, :3]) is None
    ]
    zero = [d.location for d in rs.diagnostics if d.code == "ZERO_WORLD_SCALE"]
    assert zero == [f"objects[{name}]" for name in collapsed]  # same diagnostics, same order


def test_large_scenes_extract_quickly():
    import time

    scene = Scene()
    cube = create_sphere(0.3, 8, 4)
    for i in range(5000):
        scene.add_object(f"C{i}", mesh=cube, position=(i % 100, i // 100, 0))
    scene.add_camera("Cam", position=(50, -100, 60), look_at=(50, 25, 0))
    t0 = time.perf_counter()
    rs = extract_render_scene(scene, resolve_camera(scene, None))
    elapsed = time.perf_counter() - t0
    assert len(rs.instances) == 5000
    assert elapsed < 2.0  # about 0.1 s on a laptop; the per-object version took ~0.2 s/1000
