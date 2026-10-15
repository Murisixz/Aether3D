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


# ---------------------------------------------------------------------------------------
# The GPU backend's per-frame packing (0.11 loops kept as references)


def _reference_draw_order(rs, cull, keep=None, masked=None):
    def winding(i: int) -> bool:
        return rs.instances[i].mirrored if cull != "none" else False

    def alpha(i: int) -> bool:
        return bool(masked[i]) if masked is not None else False

    indices = [i for i in range(len(rs.instances)) if keep is None or keep[i]]
    keyed = sorted(indices, key=lambda i: (rs.instances[i].mesh, winding(i), alpha(i), i))
    draws = []
    for position, index in enumerate(keyed):
        mesh, mirrored = rs.instances[index].mesh, winding(index)
        if draws and draws[-1][0] == mesh and draws[-1][1] == mirrored:
            draws[-1] = (mesh, mirrored, draws[-1][2], draws[-1][3] + 1)
        else:
            draws.append((mesh, mirrored, position, 1))
    return np.array(keyed, dtype=np.int64), draws


def _reference_pack_instances(rs, order, eye):
    from aether3d.rendering.gpu.resources import (
        HAS_GROUPS,
        HAS_NORMALS,
        HAS_UVS,
        INSTANCE_FLOATS,
        MIRRORED,
    )

    n = len(order)
    data = np.zeros((max(n, 1), INSTANCE_FLOATS), dtype=np.float32)
    if n == 0:
        return data
    words = data.view(np.uint32)
    instances = [rs.instances[i] for i in order]
    world = np.stack([inst.world for inst in instances])
    model = world[:, :3, :4].copy()
    model[:, :, 3] -= eye
    data[:, 0:12] = model.reshape(n, 12)
    normal = np.stack([inst.normal_matrix for inst in instances])
    for row in range(3):
        data[:, 4 * (3 + row) : 4 * (3 + row) + 3] = normal[:, row]
    data[:, 24:27] = np.stack([inst.base_color for inst in instances])
    data[:, 27] = [inst.specular for inst in instances]
    data[:, 28] = [inst.shininess for inst in instances]
    has_normals = np.array([m.normals is not None for m in rs.meshes], dtype=bool)
    has_uvs = np.array([m.uvs is not None for m in rs.meshes], dtype=bool)
    has_groups = np.array([m.groups is not None for m in rs.meshes], dtype=bool)
    mesh = np.fromiter((inst.mesh for inst in instances), dtype=np.int64, count=n)
    mirrored = np.fromiter((inst.mirrored for inst in instances), dtype=bool, count=n)
    words[:n, 29] = (
        np.where(mirrored, MIRRORED, 0)
        | np.where(has_normals[mesh], HAS_NORMALS, 0)
        | np.where(has_uvs[mesh], HAS_UVS, 0)
        | np.where(has_groups[mesh], HAS_GROUPS, 0)
    )
    words[:n, 30] = order
    return data


def _reference_corners(rs, bounds):
    from aether3d.rendering.shadows import _CORNERS

    n = len(rs.instances)
    mesh = np.fromiter((inst.mesh for inst in rs.instances), dtype=np.int64, count=n)
    world = np.stack([inst.world for inst in rs.instances])
    boxes = np.where(_CORNERS[None], bounds[mesh, 1][:, None, :], bounds[mesh, 0][:, None, :])
    return np.einsum("nij,nkj->nki", world[:, :3, :3], boxes) + world[:, None, :3, 3]


def _reference_instance_table(rs, order):
    from aether3d.rendering.gpu.materials import INSTANCE_WORDS

    table = np.zeros((max(len(order), 1), INSTANCE_WORDS), dtype=np.uint32)
    groups: list[int] = []
    for row, index in enumerate(order):
        inst = rs.instances[int(index)]
        table[row, 0] = inst.material
        if inst.group_materials:
            count = max(g for g, _ in inst.group_materials) + 1
            entries = [inst.material] * count
            for g, m in inst.group_materials:
                entries[g] = m
            table[row, 1] = len(groups)
            table[row, 2] = count
            groups.extend(entries)
    return table, np.array(groups or [0], dtype=np.uint32)


def packing_scene(seed: int) -> Scene:
    from aether3d import Mesh

    rng = np.random.default_rng(seed)
    s = Scene("Packing")
    sphere = create_sphere(0.5, 12, 6)
    groups = np.arange(sphere.triangle_count, dtype=np.uint16) % 3
    grouped = Mesh(sphere.positions, sphere.triangles, sphere.normals, sphere.uvs, groups)
    meshes = [create_cube(1.0), sphere, grouped]
    leaf = s.create_material("Leaf", alpha_mode="mask", alpha_cutoff=0.4)
    stone = s.create_material("Stone", base_color="#887766")
    for k in range(300):
        obj = s.add_object(
            f"o{k}",
            mesh=meshes[k % 3],
            position=tuple(rng.uniform(-30, 30, 3).tolist()),
            scale=tuple((rng.uniform(0.3, 2, 3) * rng.choice([-1, 1], 3)).tolist()),
        )
        obj.rotation_euler = tuple(rng.uniform(0, 360, 3).tolist())
        if k % 5 == 0:
            obj.material = leaf if k % 10 == 0 else stone
        if k % 3 == 2 and k % 4 == 0:
            s.assign_material(obj, stone, group=1)
            s.assign_material(obj, leaf, group=2)
        if k % 7 == 0 and obj.material is None:
            obj.color = tuple(rng.random(3).tolist())
    return s


@pytest.mark.parametrize("seed", [0, 1])
def test_render_scene_arrays_are_the_instances_values(seed):
    from aether3d.rendering.scene_data import InstanceArrays

    s = packing_scene(seed)
    rs = extract_render_scene(s, s.add_camera("c", position=(0, -80, 10)))
    gathered = InstanceArrays.gather(rs.instances)
    for name in InstanceArrays.__dataclass_fields__:
        assert_bitwise(getattr(rs.arrays, name), getattr(gathered, name), name)
    expected = sum(int(rs.meshes[i.mesh].triangles.shape[0]) for i in rs.instances)
    assert rs.triangle_count == expected


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("cull", ["back", "none"])
def test_packing_is_bitwise_that_of_the_loops(seed, cull):
    from aether3d.rendering.gpu.materials import HEADER_GROUPS, HEADER_INSTANCES, pack_material_data
    from aether3d.rendering.gpu.resources import draw_order, instance_masks, pack_instances
    from aether3d.rendering.shadows import instance_corners, mesh_bounds

    s = packing_scene(seed)
    rs = extract_render_scene(s, s.add_camera("c", position=(0, -80, 10)))
    rng = np.random.default_rng(seed + 10)
    masked = instance_masks(rs)
    assert masked.any() and not masked.all()
    for keep in (None, rng.random(len(rs.instances)) < 0.8):
        order, draws = draw_order(rs, cull, keep, masked)
        ref_order, ref_draws = _reference_draw_order(rs, cull, keep, masked)
        assert_bitwise(order, ref_order, "draw order")
        assert draws == ref_draws
        eye = np.array([1.25, -70.5, 9.0])
        assert_bitwise(pack_instances(rs, order, eye), _reference_pack_instances(rs, order, eye))
        words = pack_material_data(rs, order).words
        table, groups = _reference_instance_table(rs, order)
        start = int(words[HEADER_INSTANCES])
        assert_bitwise(words[start : start + table.size], table.reshape(-1), "instance table")
        first = int(words[HEADER_GROUPS])
        assert_bitwise(words[first : first + groups.size], groups, "groups")
    bounds = mesh_bounds(rs)
    assert_bitwise(instance_corners(rs, bounds), _reference_corners(rs, bounds), "corners")
