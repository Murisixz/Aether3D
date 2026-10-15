"""Vectorized rewrites give bitwise the results of the loops they replace.

Each test keeps the replaced per-item loop as a reference and compares it with the
vectorized code on random inputs (rotations, non-uniform and mirrored scales, singular
and non-finite transforms, empty meshes), byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest
from parity import assert_bitwise

from aether3d import Scene, create_cube, create_sphere
from aether3d.rendering.raytrace.accel import BLASCache, SceneAccel, _instance_spaces
from aether3d.rendering.scene_data import extract_render_scene


def _reference_instance_spaces(rs, mesh_blas):
    """``SceneAccel.build``'s per-instance loop before vectorization (0.11)."""
    n_inst = len(rs.instances)
    to_object = np.zeros((n_inst, 3, 4))
    inst_lo = np.full((n_inst, 3), np.inf)
    inst_hi = np.full((n_inst, 3), -np.inf)
    usable = np.zeros(n_inst, dtype=bool)
    for k, inst in enumerate(rs.instances):
        blas = mesh_blas[inst.mesh]
        m = inst.world
        if blas.bvh.order.size == 0 or not np.all(np.isfinite(m)):
            continue
        linear = m[:3, :3]
        det = float(np.linalg.det(linear))
        if not np.isfinite(det) or abs(det) < 1e-300:
            continue
        inverse = np.linalg.inv(linear)
        to_object[k, :, :3] = inverse
        to_object[k, :, 3] = -inverse @ m[:3, 3]
        lo, hi = blas.bvh.lo, blas.bvh.hi
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        world = corners @ linear.T + m[:3, 3]
        if not np.all(np.isfinite(world)):
            continue
        inst_lo[k] = np.nextafter(world.min(axis=0), -np.inf)
        inst_hi[k] = np.nextafter(world.max(axis=0), np.inf)
        usable[k] = True
    return to_object, inst_lo, inst_hi, usable


class _Inst:
    __slots__ = ("mesh", "world")

    def __init__(self, mesh, world):
        self.mesh, self.world = mesh, world


class _RS:
    def __init__(self, meshes, instances):
        self.meshes, self.instances = meshes, instances


def random_instances(seed: int, n: int = 400):
    rng = np.random.default_rng(seed)
    scene = Scene("meshes")
    meshes = [create_cube(1.3), create_sphere(0.7, 10, 5), create_sphere(2.0, 24, 12)]
    for k, mesh in enumerate(meshes):
        scene.add_object(f"m{k}", mesh=mesh)
    rs = extract_render_scene(scene, scene.add_camera("c", position=(0, -9, 0)))
    instances = []
    for k in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x, y, z, w = q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        scale = rng.uniform(0.05, 40, 3) * rng.choice([-1.0, 1.0], 3)
        m = np.eye(4)
        m[:3, :3] = rot * scale
        m[:3, 3] = rng.uniform(-1e4, 1e4, 3)
        if k % 37 == 0:
            m[:3, 0] = 0.0  # singular
        if k % 53 == 0:
            m[1, 3] = np.inf  # non-finite
        if k % 61 == 0:
            m[:3, :3] *= 1e-120  # determinant below 1e-300
        instances.append(_Inst(int(rng.integers(0, len(rs.meshes))), m))
    return _RS(rs.meshes, instances)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_instance_spaces_are_bitwise_those_of_the_loop(seed):
    rs = random_instances(seed)
    cache = BLASCache()
    mesh_blas = [cache.get(m) for m in rs.meshes]
    got = _instance_spaces(rs, mesh_blas)
    ref = _reference_instance_spaces(rs, mesh_blas)
    for name, a, b in zip(("to_object", "inst_lo", "inst_hi", "usable"), got, ref, strict=True):
        assert_bitwise(a, b, name)
    assert 0 < int(ref[3].sum()) < len(rs.instances)


def test_scene_accel_of_a_rendered_scene_is_unchanged():
    """The whole structure, through the public build, against the loop's spaces."""
    s = Scene("s")
    rng = np.random.default_rng(5)
    for k in range(200):
        obj = s.add_object(
            f"o{k}",
            mesh=create_cube(0.5) if k % 2 else create_sphere(0.4, 8, 4),
            position=tuple(rng.uniform(-20, 20, 3).tolist()),
            scale=tuple((rng.uniform(0.2, 3, 3) * rng.choice([-1, 1], 3)).tolist()),
        )
        obj.rotation_euler = tuple(rng.uniform(0, 360, 3).tolist())
    rs = extract_render_scene(s, s.add_camera("c", position=(0, -60, 0)))
    accel = SceneAccel.build(rs, BLASCache())
    mesh_blas = list(accel.blas)
    to_object, inst_lo, inst_hi, usable = _reference_instance_spaces(rs, mesh_blas)
    assert_bitwise(accel.to_object, to_object, "to_object")
    assert_bitwise(accel.inst_lo, inst_lo, "inst_lo")
    assert_bitwise(accel.inst_hi, inst_hi, "inst_hi")
