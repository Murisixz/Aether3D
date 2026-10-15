"""The ray tracing core: BVH construction and two-level traversal.

Hits are checked against an independent brute-force reference written here
(Moller-Trumbore over every world-space triangle, no BVH, no instancing): the
acceleration structure must find exactly the same nearest triangle, at the
same distance, for random rays through instanced scenes with rotations,
non-uniform and negative (mirroring) scales and shared meshes.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import aether3d as a3
from aether3d import primitives
from aether3d.mesh import Mesh
from aether3d.rendering.raytrace import BLASCache, SceneAccel, build_bvh
from aether3d.rendering.raytrace.bvh import LEAF_SIZE, triangle_bounds
from aether3d.rendering.scene_data import extract_render_scene

# --------------------------------------------------------------------------- #
# Independent reference
# --------------------------------------------------------------------------- #


def world_triangles(rs):
    """``(T, 3, 3)`` world triangles with owner instance and mesh triangle of each."""
    tris, owner, prim = [], [], []
    for k, inst in enumerate(rs.instances):
        mesh = rs.meshes[inst.mesh]
        p = mesh.positions.astype(np.float64) @ inst.world[:3, :3].T + inst.world[:3, 3]
        t = mesh.triangles.astype(np.int64)
        tris.append(p[t])
        owner.append(np.full(t.shape[0], k))
        prim.append(np.arange(t.shape[0]))
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(tris), np.concatenate(owner), np.concatenate(prim)


def brute_force(tris, origins, directions, tmax=np.inf):
    """Nearest hit of every ray over every triangle (Moller-Trumbore): ``(t, index)``."""
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    best = np.full(origins.shape[0], np.inf)
    index = np.full(origins.shape[0], -1)
    for r in range(origins.shape[0]):
        p = np.cross(directions[r], e2)
        det = (e1 * p).sum(axis=1)
        ok = det != 0.0
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origins[r] - v0
        u = (s * p).sum(axis=1) * inv
        q = np.cross(s, e1)
        v = (q @ directions[r]) * inv
        t = (e2 * q).sum(axis=1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < tmax)
        if hit.any():
            k = np.flatnonzero(hit)[np.argmin(t[hit])]
            best[r], index[r] = t[k], k
    return best, index


def random_scene(seed: int, count: int = 24) -> a3.Scene:
    rng = np.random.default_rng(seed)
    scene = a3.Scene()
    shared = [primitives.create_sphere(0.7, 18, 9), primitives.create_cube(1.0)]
    for _ in range(count):
        scale = rng.uniform(0.3, 1.5, 3) * np.where(rng.random(3) < 0.25, -1.0, 1.0)
        obj = scene.add_object(
            mesh=shared[int(rng.integers(0, 2))], position=rng.uniform(-4, 4, 3), scale=scale
        )
        obj.transform.rotation_euler = rng.uniform(-180, 180, 3)
    scene.add_camera(position=(0, -12, 0))
    return scene


def accel_for(scene: a3.Scene, cache: BLASCache | None = None):
    rs = extract_render_scene(scene, scene.active_camera)
    return rs, SceneAccel.build(rs, cache or BLASCache())


def height_field(n: int = 40, seed: int = 0) -> Mesh:
    """A bumpy grid whose triangles share vertices exactly (closed under refinement)."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    zs = rng.uniform(-0.05, 0.05, xs.shape)
    positions = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return Mesh(positions, tris)


# --------------------------------------------------------------------------- #
# BVH construction
# --------------------------------------------------------------------------- #


def test_bvh_leaves_hold_every_primitive_once_inside_their_parents_boxes():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(15000, 3)) * rng.uniform(0.01, 2.0, (15000, 1))
    lo, hi = triangle_bounds(v, np.arange(15000).reshape(5000, 3))
    bvh = build_bvh(lo, hi)
    assert np.array_equal(np.sort(bvh.order), np.arange(5000))
    leaves = np.flatnonzero(bvh.child[:, 0] < 0)
    assert bvh.count[leaves].sum() == 5000
    assert bvh.count[leaves].max() <= LEAF_SIZE
    assert np.all(bvh.count[bvh.child[:, 0] >= 0] == 0)
    # Every node's primitives lie inside the box its parent stores for it.
    node_lo = np.full((bvh.node_count, 3), np.nan)
    node_hi = np.full((bvh.node_count, 3), np.nan)
    node_lo[0], node_hi[0] = bvh.lo, bvh.hi
    inner = np.flatnonzero(bvh.child[:, 0] >= 0)
    for side in (0, 1):
        node_lo[bvh.child[inner, side]] = bvh.child_lo[inner, side]
        node_hi[bvh.child[inner, side]] = bvh.child_hi[inner, side]
    for leaf in leaves:
        prims = bvh.order[bvh.first[leaf] : bvh.first[leaf] + bvh.count[leaf]]
        assert np.all(lo[prims] >= node_lo[leaf]) and np.all(hi[prims] <= node_hi[leaf])
    for node in inner:  # children inside their parent
        for side in (0, 1):
            assert np.all(bvh.child_lo[node, side] >= node_lo[node])
            assert np.all(bvh.child_hi[node, side] <= node_hi[node])
    assert bvh.depth <= 2 * int(np.log2(5000)) + 2


def test_bvh_build_is_deterministic_and_handles_identical_centroids():
    rng = np.random.default_rng(4)
    v = rng.normal(size=(777 * 3, 3)).astype(np.float32)
    lo, hi = triangle_bounds(v, np.arange(777 * 3).reshape(777, 3))
    assert lo.dtype == np.float32
    a, b = build_bvh(lo, hi), build_bvh(lo.copy(), hi.copy())
    for name in ("child", "child_lo", "child_hi", "first", "count", "order"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert a.child_lo.dtype == np.float32  # boxes keep the bounds' precision
    # 1000 identical boxes: no SAH split exists, the median split still terminates.
    same = build_bvh(np.zeros((1000, 3)), np.ones((1000, 3)))
    assert np.array_equal(np.sort(same.order), np.arange(1000))
    assert same.depth <= 12
    empty = build_bvh(np.zeros((0, 3)), np.zeros((0, 3)))
    assert empty.node_count == 1 and empty.order.size == 0


# --------------------------------------------------------------------------- #
# Traversal against the brute-force reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_closest_hits_match_brute_force_in_instanced_mirrored_scenes(seed):
    scene = random_scene(seed)
    rs, accel = accel_for(scene)
    assert accel.info["meshes"] == 2 and accel.info["blas_built"] == 2  # shared meshes
    rng = np.random.default_rng(seed + 100)
    n = 1500
    origins = rng.uniform(-6, 6, (n, 3))
    targets = rng.uniform(-4, 4, (n, 3))  # aim into the cluster: most rays hit
    directions = (targets - origins) * rng.uniform(0.2, 3.0, (n, 1))
    hits = accel.closest_hit(origins, directions)
    tris, owner, prim = world_triangles(rs)
    t_ref, i_ref = brute_force(tris, origins, directions)
    found = np.isfinite(t_ref)
    assert found.sum() > n // 3
    assert np.array_equal(hits.hit, found)
    np.testing.assert_allclose(hits.t[found], t_ref[found], rtol=1e-9)
    same = (owner[i_ref[found]] == hits.inst[found]) & (prim[i_ref[found]] == hits.prim[found])
    assert same.all()
    # The barycentrics locate the hit point on the reported triangle.
    k = np.flatnonzero(found)
    offsets = np.cumsum([0] + [rs.meshes[i.mesh].triangles.shape[0] for i in rs.instances])
    corners = tris[offsets[hits.inst[k]] + hits.prim[k]]
    on_triangle = np.einsum("nk,nkc->nc", hits.b[k], corners)
    np.testing.assert_allclose(on_triangle, origins[k] + directions[k] * hits.t[k, None], atol=1e-9)
    assert np.all(hits.b[k] >= 0) and np.allclose(hits.b[k].sum(axis=1), 1.0)
    # Any-hit queries agree, and nothing lies before the nearest hit.
    assert np.array_equal(accel.occluded(origins, directions), found)
    before = accel.occluded(origins, directions, tmax=np.where(found, t_ref * (1 - 1e-9), np.inf))
    assert not before[found].any()


def test_hits_respect_tmin_tmax_and_the_skipped_triangle():
    scene = a3.Scene()
    scene.add_plane(size=4.0, position=(0, 0, 0))
    scene.add_plane(size=4.0, position=(0, 0, 1))
    scene.add_camera(position=(0, -5, 0))
    rs, accel = accel_for(scene)
    o = np.array([[0.1, 0.2, 3.0]] * 4)
    d = np.array([[0.0, 0.0, -1.0]] * 4)
    hits = accel.closest_hit(o, d, tmin=np.array([0, 2.5, 0, 0]), tmax=np.array([9, 9, 1.5, 2.0]))
    assert hits.t.tolist() == pytest.approx([2.0, 3.0, np.inf, np.inf], abs=1e-12)  # tmax: open
    assert hits.inst.tolist() == [1, 0, -1, -1]
    # A ray leaving the upper plane's triangle skips it and finds the lower plane.
    first = accel.closest_hit(o[:1], d[:1])
    again = accel.closest_hit(
        o[:1] + d[:1] * first.t[0], d[:1], skip_inst=first.inst, skip_tri=first.prim
    )
    assert again.inst[0] == 0 and again.t[0] == pytest.approx(1.0)


@pytest.mark.parametrize("rotation", [(0, 0, 0), (33, 12, 71), (180, 0, 0)])
def test_rays_through_shared_vertices_and_edges_never_slip_through(rotation):
    """Watertightness: rays crossing the surface exactly at shared vertices/edges hit."""
    mesh = height_field()
    scene = a3.Scene()
    obj = scene.add_object(mesh=mesh, scale=(1.3, -0.7, 2.1))
    obj.transform.rotation_euler = rotation
    scene.add_camera(position=(0, -5, 0))
    rs, accel = accel_for(scene)
    w = rs.instances[0].world
    p = mesh.positions.astype(np.float64) @ w[:3, :3].T + w[:3, 3]
    t = mesh.triangles.astype(np.int64)
    g = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    up = g.mean(axis=0) / np.linalg.norm(g.mean(axis=0))
    local = mesh.positions.astype(np.float64)
    border = (np.abs(local[:, 0]) > 0.999) | (np.abs(local[:, 1]) > 0.999)
    rng = np.random.default_rng(7)
    total = 0
    for _ in range(4):
        direction = -(up + rng.normal(size=3) * 0.5)
        direction /= np.linalg.norm(direction)
        facing = g @ direction
        front, back = facing < -0.05, facing > 0.05
        # Only targets where the ray really crosses the surface: every triangle around the
        # vertex (or both beside the edge) faces the ray the same way (no grazing contact).
        all_front = np.ones(p.shape[0], bool)
        all_back = np.ones(p.shape[0], bool)
        for j in range(3):
            np.logical_and.at(all_front, t[:, j], front)
            np.logical_and.at(all_back, t[:, j], back)
        targets = [p[(all_front | all_back) & ~border]]
        edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        owner = np.tile(np.arange(t.shape[0]), 3)
        key = np.sort(edges, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        pair = np.all(key[order][1:] == key[order][:-1], axis=1)
        t1, t2 = owner[order[:-1][pair]], owner[order[1:][pair]]
        crossing = (front[t1] & front[t2]) | (back[t1] & back[t2])
        e = key[order[:-1][pair]][crossing]
        for f in (0.5, 0.25, 0.9):
            targets.append(p[e[:, 0]] * f + p[e[:, 1]] * (1 - f))
        x = np.concatenate(targets)
        o = x - direction * 3.0
        hits = accel.closest_hit(o, x - o)
        assert hits.hit.all()
        assert np.all(hits.t <= 1.0 + 1e-12)
        total += x.shape[0]
    assert total > 10_000


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_degenerate_triangles_are_never_hit_and_do_not_break_the_build():
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 0), (1, 1, 0)]
    # zero-area (collinear), repeated vertex, and one valid triangle
    triangles = [(0, 1, 2), (0, 0, 3), (4, 1, 3), (1, 5, 3)]
    scene = a3.Scene()
    scene.add_object(mesh=Mesh(positions, triangles))
    scene.add_camera(position=(0, 0, 5))
    rs, accel = accel_for(scene)
    o = np.array([[0.5, 0.0, 1.0], [1.5, 0.0, 1.0], [0.0, 0.5, 1.0], [0.8, 0.8, 1.0]])
    d = np.tile([0.0, 0.0, -1.0], (4, 1))
    hits = accel.closest_hit(o, d)
    # rays 0/1 cross only the collinear triangle's line (and edges of real triangles)
    tris, owner, prim = world_triangles(rs)
    t_ref, _ = brute_force(tris, o, d)
    assert np.array_equal(hits.hit, np.isfinite(t_ref))
    assert hits.hit[3] and hits.prim[3] == 3


def test_grazing_rays_in_a_triangles_plane_miss_it():
    scene = a3.Scene()
    scene.add_plane(size=2.0)
    scene.add_camera(position=(0, -5, 0))
    _, accel = accel_for(scene)
    o = np.array([[-3.0, 0.0, 0.0], [-3.0, 0.3, 0.0], [0.0, -3.0, 0.0]])
    d = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = accel.closest_hit(o, d)
    assert not hits.hit.any()


def test_axis_parallel_rays_on_a_box_face_plane_still_reach_its_edges():
    # A ray along -z through x = 1 runs exactly in the plane of the cube's +x face (and of
    # every box bounding it there); it must still hit the top and bottom faces' edges, on
    # both sides of the box and in both directions (boxes are closed, 0.10.0).
    scene = a3.Scene()
    scene.add_cube(size=2.0)
    scene.add_camera(position=(0, -5, 0))
    rs, accel = accel_for(scene)
    o = np.array(
        [[1.0, 0.0, 5.0], [-1.0, 0.0, 5.0], [0.0, 1.0, -5.0], [0.0, -1.0, -5.0], [1.0, 1.0, 5.0]]
    )
    d = np.array([[0.0, 0.0, -1.0]] * 2 + [[0.0, 0.0, 1.0]] * 2 + [[0.0, 0.0, -1.0]])
    hits = accel.closest_hit(o, d)
    assert hits.hit.all() and np.allclose(hits.t, 4.0)
    tris, _, _ = world_triangles(rs)
    want, _ = brute_force(tris, o, d)
    assert np.allclose(want, 4.0)  # the reference agrees (edges belong to the triangles)
    assert accel.occluded(o, d).all()


def test_rays_from_inside_closed_meshes_hit_the_inside():
    scene = a3.Scene()
    scene.add_sphere(radius=1.0, segments=48, rings=24)
    scene.add_camera(position=(0, -5, 0))
    _, accel = accel_for(scene)
    rng = np.random.default_rng(5)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    hits = accel.closest_hit(np.zeros((500, 3)), d)
    assert hits.hit.all()
    assert np.all((hits.t > 0.98) & (hits.t <= 1.0 + 1e-6))


def test_overlapping_coplanar_and_near_contact_surfaces():
    scene = a3.Scene()
    scene.add_plane(size=2.0)
    scene.add_plane(size=2.0)  # exactly coplanar duplicate
    scene.add_plane(size=2.0, position=(0, 0, 1e-7))  # near contact above both
    scene.add_camera(position=(0, -5, 0))
    rs, accel = accel_for(scene)
    o = np.array([[0.2, 0.1, 1.0], [0.2, 0.1, -1.0]])
    d = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    hits = accel.closest_hit(o, d)
    assert hits.inst[0] == 2 and hits.t[0] == pytest.approx(1.0 - 1e-7, abs=1e-12)
    assert hits.inst[1] in (0, 1) and hits.t[1] == pytest.approx(1.0, abs=1e-12)
    again = accel.closest_hit(o, d)
    assert np.array_equal(again.inst, hits.inst) and np.array_equal(again.prim, hits.prim)


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_extreme_scales_keep_relative_precision(scale):
    scene = random_scene(9, count=8)
    for obj in scene:
        if obj.mesh is not None:
            obj.transform.position = np.asarray(obj.transform.position) * scale
            obj.transform.scale = np.asarray(obj.transform.scale) * scale
    rs, accel = accel_for(scene)
    rng = np.random.default_rng(11)
    origins = rng.uniform(-6, 6, (400, 3)) * scale
    directions = (rng.uniform(-4, 4, (400, 3)) * scale - origins) / scale
    hits = accel.closest_hit(origins, directions)
    tris, owner, prim = world_triangles(rs)
    t_ref, i_ref = brute_force(tris, origins, directions)
    found = np.isfinite(t_ref)
    assert found.any()
    assert np.array_equal(hits.hit, found)
    np.testing.assert_allclose(hits.t[found], t_ref[found], rtol=1e-9)


def test_non_finite_rays_and_empty_scenes_report_no_hits():
    scene = a3.Scene()
    scene.add_camera(position=(0, -5, 0))
    rs, accel = accel_for(scene)
    hits = accel.closest_hit(np.zeros((3, 3)), np.ones((3, 3)))
    assert not hits.hit.any() and accel.info["instances"] == 0
    scene.add_cube()
    _, accel = accel_for(scene)
    o = np.array([[np.nan, 0, -5], [0, 0, -5], [0, 0, -5]])
    d = np.array([[0, 0, 1.0], [0, 0, np.inf], [0, 0, 1.0]])
    assert accel.closest_hit(o, d).hit.tolist() == [False, False, True]


# --------------------------------------------------------------------------- #
# Bottom-level structures: sharing, reuse and invalidation
# --------------------------------------------------------------------------- #


def test_blas_cache_shares_reuses_and_invalidates_meshes():
    cache = BLASCache()
    scene = a3.Scene()
    sphere = primitives.create_sphere(1.0, 16, 8)
    a = scene.add_object(mesh=sphere)
    scene.add_object(mesh=sphere, position=(3, 0, 0))
    scene.add_camera(position=(0, -5, 0))
    rs, first = accel_for(scene, cache)
    # One mesh, one BLAS: both instances enter the same bottom-level root.
    assert first.info["meshes"] == 1 and first.info["blas_built"] == 1
    assert first.inst_root[0] == first.inst_root[1]
    rs, second = accel_for(scene, cache)
    assert second.info["blas_built"] == 0 and second.info["blas_reused"] == 1
    assert second.blas[0] is first.blas[0]
    # Editing geometry makes a new mesh: a new BLAS is built for it.
    a.mesh = primitives.create_cube(1.0)
    rs, third = accel_for(scene, cache)
    assert third.info["blas_built"] == 1 and third.info["meshes"] == 2
    # Meshes that disappear take their BLAS with them.
    del scene, a, sphere, rs, first, second, third
    gc.collect()
    assert cache.stats()["meshes"] == 0 and cache.bytes == 0


def test_blas_cache_evicts_least_recently_used_beyond_its_budget():
    meshes = [primitives.create_sphere(1.0, 24, 12) for _ in range(4)]
    cache = BLASCache(budget=1)  # every entry exceeds the budget
    scene = a3.Scene()
    for m in meshes:
        scene.add_object(mesh=m)
    scene.add_camera(position=(0, -5, 0))
    _, accel = accel_for(scene, cache)
    # The meshes of the render being built are kept; nothing else is.
    assert cache.stats()["meshes"] == 4
    other = a3.Scene()
    other.add_object(mesh=primitives.create_cube(1.0))
    other.add_camera(position=(0, -5, 0))
    accel_for(other, cache)
    assert cache.stats()["meshes"] == 1 and cache.evictions == 4
    assert accel.closest_hit(np.array([[0.0, -5, 0]]), np.array([[0.0, 1, 0]])).hit[0]
