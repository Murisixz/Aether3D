"""Skin weights (Stage 18): geodesic voxel binding, policies, rigid parts, editing with
selections, brushes and locks, mirroring, transfer and inspection -- against
independent expectations (touching limbs, symmetric meshes, analytic falloffs)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d import Scene, _native, create_cube
from aether3d.animation import voxels as V
from aether3d.errors import AnimationError
from aether3d.mesh import Mesh


def tube(a, b, r, n_ring=12, rings=24, caps=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    w = axis / np.linalg.norm(axis)
    u = np.cross(w, [0.31, 0.47, 0.83])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    ang = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
    pts = [a + f * axis + r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
           for f in np.linspace(0, 1, rings)]  # fmt: skip
    pos = np.concatenate(pts)
    tris = []
    for k in range(rings - 1):
        for i in range(n_ring):
            p0, p1 = k * n_ring + i, k * n_ring + (i + 1) % n_ring
            tris += [[p0, p1, p1 + n_ring], [p0, p1 + n_ring, p0 + n_ring]]
    if caps:
        c0, c1 = len(pos), len(pos) + 1
        pos = np.vstack([pos, a, b])
        last = (rings - 1) * n_ring
        for i in range(n_ring):
            tris += [[c0, (i + 1) % n_ring, i], [c1, last + i, last + (i + 1) % n_ring]]
    return pos, np.array(tris)


def merge(*parts):
    pos, tris, base = [], [], 0
    for p, t in parts:
        pos.append(p)
        tris.append(t + base)
        base += len(p)
    return Mesh(np.concatenate(pos).astype(np.float32), np.concatenate(tris).astype(np.uint32))


def legs_scene(gap=0.006):
    """Two legs (one mesh, two tubes) whose surfaces are ``gap`` apart, a pelvis bar."""
    s = Scene("Legs")
    arm = s.add_object("Arm")
    r = 0.06
    x = r + gap / 2
    s.animation.add_rig(
        arm,
        [
            {"name": "Hips", "position": [0, 0, 1.0], "tail": [0, 0, 0.1]},
            {"name": "Thigh.L", "parent": "Hips", "position": [x, 0, 0], "tail": [0, 0, -0.45],
             "role": "thigh.L"},
            {"name": "Shin.L", "parent": "Thigh.L", "position": [0, 0, -0.45], "role": "shin.L",
             "tail": [0, 0, -0.45]},
            {"name": "Thigh.R", "parent": "Hips", "position": [-x, 0, 0], "tail": [0, 0, -0.45],
             "role": "thigh.R"},
            {"name": "Shin.R", "parent": "Thigh.R", "position": [0, 0, -0.45], "role": "shin.R",
             "tail": [0, 0, -0.45]},
        ],
    )  # fmt: skip
    mesh = merge(
        tube([x, 0, 1.0], [x, 0, 0.1], r),
        tube([-x, 0, 1.0], [-x, 0, 0.1], r),
        tube([-0.2, 0, 1.0], [0.2, 0, 1.0], 0.07),
    )
    body = s.add_object("Legs", mesh=mesh)
    return s, arm, body


def dense(s, body):
    sk = s.animation._owned[("skin", body.id)]._definition
    rig = s.animation._items[sk.rig]._definition
    names = [rig.joints[rig.index_of_id(j)].name for j in sk.joints]
    return sk.dense(), names


def test_geodesic_weights_do_not_leak_between_touching_legs():
    s, arm, body = legs_scene()
    a = s.animation
    rep = a.generate_weights(body, arm, resolution=400)  # (voxels finer than the gap)
    assert rep["unreached_vertices"] == 0 and rep["technique"].startswith("geodesic")
    w, names = dense(s, body)
    p = body.mesh.positions
    left_leg = (p[:, 0] > 0.01) & (p[:, 2] < 0.9)
    right_cols = [names.index(n) for n in ("Thigh.R", "Shin.R")]
    assert w[left_leg][:, right_cols].max() == 0.0  # not a trace of the other leg
    assert np.allclose(w.sum(axis=1), 1.0)
    # Stage 17's Euclidean envelopes, by contrast, pull the other leg (6 mm apart).
    a.skin(body, arm, weights="envelope", influences=4)
    w2, names2 = dense(s, body)
    cols2 = [names2.index(n) for n in ("Thigh.R", "Shin.R")]
    assert w2[left_leg][:, cols2].max() > 0.2
    assert any(f["code"] == "CROSS_SIDE_WEIGHTS" for f in a.inspect_weights(body)["findings"])
    a.generate_weights(body, arm, resolution=400)
    assert not any(f["code"] == "CROSS_SIDE_WEIGHTS" for f in a.inspect_weights(body)["findings"])
    # Without the skeleton rule, the other thigh reaches the inner thigh through the
    # pelvis (geodesically near); with too coarse voxels the legs fuse.
    a.generate_weights(body, arm, resolution=400, hops=0, sides="off")
    w3, n3 = dense(s, body)
    assert w3[left_leg][:, [n3.index("Thigh.R")]].max() > 0.01
    a.generate_weights(body, arm, resolution=60, hops=0, sides="off")
    w4, n4 = dense(s, body)
    assert w4[left_leg & (p[:, 2] < 0.5)][:, [n4.index("Shin.R")]].max() > 0.01
    # The skeleton and side rules hold even then.
    a.generate_weights(body, arm, resolution=60)
    w5, n5 = dense(s, body)
    assert w5[left_leg][:, [n5.index("Shin.R"), n5.index("Thigh.R")]].max() == 0.0


def test_geodesic_distances_are_exact_shortest_paths(monkeypatch):
    """The chamfer distances equal a brute-force Dijkstra on the same graph, natively and
    in NumPy (an L-shaped solid: paths must go round the corner)."""
    import heapq

    solid = np.zeros((14, 12, 6), bool)
    solid[1:13, 1:4, 1:5] = True
    solid[1:4, 1:11, 1:5] = True
    grid = V.VoxelGrid(np.zeros(3), 1.0, solid, solid.copy())
    nx, ny, nz = solid.shape
    flat = grid.solid_flat()
    src = np.array([grid.flat(np.array([[11, 2, 2]]))[0]])
    queries = np.flatnonzero(flat)
    got = V.geodesic(grid, [src], queries)[:, 0]
    # Independent Dijkstra.
    coords = {int(f): (f % nx, (f // nx) % ny, f // (nx * ny)) for f in queries}
    dist = {int(src[0]): 0}
    heap = [(0, int(src[0]))]
    while heap:
        d, f = heapq.heappop(heap)
        if d > dist[f]:
            continue
        i, j, k = coords[f]
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    g = (i + dx) + nx * ((j + dy) + ny * (k + dz))
                    if g in coords:
                        nd = d + 2 + (dx != 0) + (dy != 0) + (dz != 0)
                        if nd < dist.get(g, 1 << 30):
                            dist[g] = nd
                            heapq.heappush(heap, (nd, g))
    expect = np.array([dist[int(f)] for f in queries])
    assert np.array_equal(got, expect)
    monkeypatch.setattr(_native, "module", lambda: None)
    assert np.array_equal(V.geodesic(grid, [src], queries)[:, 0], expect)
    capped = V.geodesic(grid, [src], queries, cutoff=20)[:, 0]
    assert np.array_equal(capped, np.where(expect <= 20, expect, -1))


def test_policies_parts_and_fallbacks():
    s, arm, body = legs_scene(gap=0.05)
    a = s.animation
    hat = s.add_object("Hat", mesh=merge(tube([0, 0, 1.8], [0, 0, 1.9], 0.1)))
    a.generate_weights(body, arm, policy="rigid")
    w, names = dense(s, body)
    assert np.all(np.sort(w, axis=1)[:, -1] == 1.0)  # one joint per vertex
    rep = a.generate_weights(hat, arm, resolution=64)  # no bone runs through the hat
    assert rep["unreached_vertices"] == hat.mesh.vertex_count and rep["rigid_parts"] == 1
    wh, nh = dense(s, hat)
    assert np.all(wh[:, nh.index("Hips")] == 1.0)
    # The stylized policy blends over a shorter span than the organic one.
    a.generate_weights(body, arm, policy="organic")
    wo, _ = dense(s, body)
    a.generate_weights(body, arm, policy="stylized")
    ws, _ = dense(s, body)
    assert (wo > 0.05).sum(axis=1).mean() > (ws > 0.05).sum(axis=1).mean()
    with pytest.raises(AnimationError) as e:
        a.generate_weights(body, arm, policy="painterly")
    assert e.value.code == "INVALID_ARGUMENT"


def test_open_meshes_are_closed_and_generation_is_deterministic(monkeypatch):
    s = Scene("Open")
    arm = s.add_object("Arm")
    s.animation.add_rig(arm, [{"name": "A", "position": [0, 0, 0], "tail": [0, 0, 0.5]},
                              {"name": "B", "parent": "A", "position": [0, 0, 0.5],
                               "tail": [0, 0, 0.5]}])  # fmt: skip
    open_tube = merge(tube([0, 0, 0], [0, 0, 1.0], 0.1, caps=False))
    body = s.add_object("Sleeve", mesh=open_tube)
    rep = s.animation.generate_weights(body, arm, resolution=48)
    assert rep["unreached_vertices"] == 0
    w1, _ = dense(s, body)
    monkeypatch.setattr(_native, "module", lambda: None)
    s.animation.generate_weights(body, arm, resolution=48)
    w2, _ = dense(s, body)
    assert np.array_equal(w1, w2)  # NumPy and native agree, bit for bit


def test_editing_with_brushes_locks_and_one_undo_step():
    s, arm, body = legs_scene(gap=0.05)
    a = s.animation
    a.generate_weights(body, arm)
    w0, names = dense(s, body)
    p = body.mesh.positions.astype(float)
    n_hist = len(s.history)
    knee = [0.085, 0.0, 0.55]
    rep = a.edit_weights(body, [
        {"op": "lock", "joints": ["Hips"]},
        {"op": "assign", "joint": "Shin.L", "value": 1.0,
         "falloff": {"center": knee, "radius": 0.1, "curve": "linear"}},
    ])  # fmt: skip
    assert len(s.history) == n_hist + 1
    assert rep["locked_joints"] == ["Hips"]
    w1, names1 = dense(s, body)
    d = np.linalg.norm(p - knee, axis=1)
    c = names1.index("Shin.L")
    expect = w0[:, c] + (1 - d / 0.1).clip(0) * (1 - w0[:, c])
    inside = d < 0.1
    hips = names1.index("Hips")
    # (where the locked hips weigh in, the shin gets at most what is left)
    free = w0[:, hips] < 1e-9
    assert np.allclose(w1[inside & free, c], expect[inside & free], atol=1e-9)
    assert np.allclose(w1[:, hips], w0[:, hips], atol=1e-12)
    assert np.allclose(w1.sum(axis=1), 1.0)
    assert np.array_equal(w1[~inside], w0[~inside])
    s.history.undo()
    assert np.array_equal(dense(s, body)[0], w0)
    # Persistent vertex locks: smoothing and scaling leave locked vertices alone.
    a.edit_weights(body, [{"op": "lock", "select": {"sphere": {"center": knee, "radius": 0.05}}}])
    a.edit_weights(body, [{"op": "smooth", "iterations": 20},
                          {"op": "scale", "joint": "Shin.L", "factor": 0.5}])  # fmt: skip
    w2, _ = dense(s, body)
    locked = np.linalg.norm(p - knee, axis=1) <= 0.05
    assert np.array_equal(w2[locked], w0[locked]) and not np.array_equal(w2[~locked], w0[~locked])


def test_smoothing_never_crosses_disconnected_parts():
    s, arm, body = legs_scene(gap=0.05)
    a = s.animation
    a.generate_weights(body, arm, policy="rigid", rigid_parts="all")
    w0, names = dense(s, body)
    a.edit_weights(body, [{"op": "smooth", "iterations": 50, "strength": 1.0}])
    w1, _ = dense(s, body)
    assert np.array_equal(w0, w1)  # each part is rigid: nothing to average with


def test_prune_cap_replace_rigid_and_validation():
    s, arm, body = legs_scene(gap=0.05)
    a = s.animation
    a.generate_weights(body, arm, influences=4)
    a.edit_weights(body, [{"op": "cap", "influences": 2}, {"op": "prune", "threshold": 0.2}])
    w, names = dense(s, body)
    assert ((w > 0).sum(axis=1) <= 2).all() and (w[w > 0] >= 0.2 - 1e-12).all()
    a.edit_weights(body, [{"op": "replace", "from": "Shin.L", "to": "Thigh.L"}])
    w, names = dense(s, body)
    assert w[:, names.index("Shin.L")].max() == 0.0
    a.edit_weights(body, [{"op": "rigid", "select": {"component_of": 0}, "joint": "Hips"}])
    w, names = dense(s, body)
    for bad in (
        [{"op": "paint"}],
        [{"op": "assign", "joint": "Nope"}],
        [{"op": "assign", "joint": "Hips", "value": 2}],
        [{"op": "assign", "joint": "Hips", "select": {"sphere": {"center": [0, 0]}}}],
        [{"op": "smooth", "iterations": 0}],
        [{"op": "lock", "joints": ["Hips"]}, {"op": "assign", "joint": "Hips"}],
    ):
        with pytest.raises(AnimationError):
            a.edit_weights(body, bad)


def test_mirroring_weights_with_joint_pairs():
    s, arm, body = legs_scene(gap=0.05)
    a = s.animation
    a.generate_weights(body, arm)
    w0, names = dense(s, body)
    box = {"box": {"min": [-1, -1, 0.3], "max": [0, 1, 0.4]}}
    a.edit_weights(body, [{"op": "assign", "joint": "Shin.R", "value": 1.0, "select": box}])
    rep = a.edit_weights(body, [{"op": "mirror", "from": "+x"}])
    assert rep["operations"][0]["matched"] > 0
    w1, names1 = dense(s, body)
    p = body.mesh.positions.astype(float)
    for side_l, side_r in (("Thigh.L", "Thigh.R"), ("Shin.L", "Shin.R")):
        li, ri = names1.index(side_l), names1.index(side_r)
        left = p[:, 0] > 0.02
        right_of = {tuple(np.round(q * [-1, 1, 1], 5)): i for i, q in enumerate(p)}
        for i in np.flatnonzero(left)[:200]:
            j = right_of.get(tuple(np.round(p[i], 5)))
            if j is not None:
                assert w1[j, ri] == pytest.approx(w1[i, li], abs=1e-9)


def test_transfer_gives_clothing_the_body_weights_and_reports_far_parts():
    s, arm, body = legs_scene(gap=0.05)
    a = s.animation
    a.generate_weights(body, arm)
    # A trouser leg 1 cm outside the left leg, and a sword far away.
    x = 0.06 + 0.025
    trousers = merge(
        tube([x, 0, 0.95], [x, 0, 0.2], 0.07), tube([0.6, 0, 1.0], [0.6, 0, 0.2], 0.02)
    )
    pants = s.add_object("Pants", mesh=trousers)
    rep = a.transfer_weights(pants, body, max_distance=0.05)
    assert len(rep["unmatched_parts"]) == 1 and rep["unmatched_parts"][0]["vertices"] > 100
    trouser = next(part for part in rep["parts"] if part["matched"])
    assert trouser["median_distance"] == pytest.approx(0.01, abs=0.003)
    w, names = dense(s, pants)
    wb, nb = dense(s, body)
    pp = pants.mesh.positions.astype(float)
    leg = (pp[:, 0] < 0.3) & (pp[:, 2] < 0.9) & (pp[:, 2] > 0.25)
    shin = names.index("Shin.L")
    # Weights follow the body's along the leg (within the blend of neighbouring rings).
    pb = body.mesh.positions.astype(float)
    for z in (0.3, 0.55, 0.8):
        near_p = leg & (np.abs(pp[:, 2] - z) < 0.02)
        near_b = (pb[:, 0] > 0.03) & (np.abs(pb[:, 2] - z) < 0.02)
        assert w[near_p, shin].mean() == pytest.approx(wb[near_b, nb.index("Shin.L")].mean(),
                                                         abs=0.08)  # fmt: skip
    sword = ~leg & (pp[:, 0] > 0.3)
    assert np.all(np.sort(w[sword], axis=1)[:, -1] == 1.0)  # rigid
    with pytest.raises(AnimationError) as e:
        a.transfer_weights(pants, body, max_distance=0.05, unmatched="refuse")
    assert e.value.code == "UNMATCHED_PARTS"


def test_time_limits_cancel_generation():
    s, arm, body = legs_scene()
    with pytest.raises(AnimationError) as e:
        s.animation.generate_weights(body, arm, max_seconds=0.0)
    assert e.value.code == "CANCELLED"
    assert ("skin", body.id) not in s.animation._owned  # nothing changed


def test_blocky_boxes_bind_part_by_part():
    s = Scene("Blocky")
    arm = s.add_object("Arm")
    s.animation.add_rig(arm, [
        {"name": "Body", "position": [0, 0, 0.75], "tail": [0, 0, 0.75]},
        {"name": "Head", "parent": "Body", "position": [0, 0, 0.75], "tail": [0, 0, 0.5]},
        {"name": "Arm.L", "parent": "Body", "position": [0.375, 0, 0.75], "tail": [0, 0, -0.75]},
    ])  # fmt: skip
    parts = []
    for c, size in (((0, 0, 1.125), (0.5, 0.25, 0.75)), ((0, 0, 1.75), (0.5, 0.5, 0.5)),
                    ((0.375 + 0.125, 0, 1.125), (0.25, 0.25, 0.75))):  # fmt: skip
        b = create_cube(1.0)
        parts.append((b.positions.astype(float) * size + c, b.triangles.astype(np.int64)))
    body = s.add_object("Blocks", mesh=merge(*parts))
    s.animation.generate_weights(body, arm, policy="rigid", resolution=96)
    w, names = dense(s, body)
    p = body.mesh.positions
    assert np.all(w[p[:, 2] > 1.55, names.index("Head")] == 1.0)
    assert np.all(w[p[:, 0] > 0.3, names.index("Arm.L")] == 1.0)


def test_auto_rigged_weights_have_no_seams_at_bone_changes():
    """Where the nearest bone changes (armpits, the neck, the lower back) the weights stay
    smooth: no edge between two vertices whose weights differ by more than half, and
    twisting the torso or turning the head stretches no edge more than 5 times."""
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    s = Scene("Seams")
    body = s.add_object("Body", mesh=implicit_surface(humanoid_shapes(), 0.016))
    a = s.animation
    a.autorig(body, name="Rig")
    w, _names = dense(s, body)
    tris = np.asarray(body.mesh.triangles)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                                              tris[:, [2, 0]]]), axis=1), axis=0)  # fmt: skip
    jump = np.abs(w[edges[:, 0]] - w[edges[:, 1]]).sum(axis=1) / 2
    assert np.percentile(jump, 99.9) < 0.5, np.percentile(jump, 99.9)
    rep = a.diagnose_rig("Rig", poses=["torso_twist_45", "head_turn_60", "spine_bend_45"],
                         animation=False)  # fmt: skip
    for pose, rec in rep["poses"].items():
        assert rec["Body"]["worst_stretch"] < 5.0, (pose, rec["Body"]["worst_stretch"])


def test_smoothing_adds_neighbours_in_scatter_order():
    """smooth_rows' per-column bincounts give exactly the bits of the plain unbuffered
    scatter (np.add.at) it replaced, with pinned rows and columns kept as they are."""
    from aether3d.animation.weights import mesh_edges, smooth_rows

    def reference(w, a, b, iterations, strength, rows, cols):
        n = w.shape[0]
        deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
        out = w.copy()
        for _ in range(iterations):
            acc = np.zeros_like(out)
            np.add.at(acc, a, out[b])
            np.add.at(acc, b, out[a])
            mean = np.where(deg[:, None] > 0, acc / np.maximum(deg, 1)[:, None], out)
            new = out + strength * (mean - out)
            if cols is not None:
                new[:, cols] = out[:, cols]
                fixed = new[:, cols].sum(axis=1)
                free = np.clip(new[:, ~cols], 0.0, None)
                total = free.sum(axis=1)
                room = np.clip(1.0 - fixed, 0.0, None)
                scale = np.where(total > 0, room / np.where(total > 0, total, 1.0), 0.0)
                new[:, ~cols] = free * scale[:, None]
            if rows is not None:
                new[rows] = out[rows]
            out = new
        return out

    rng = np.random.default_rng(5)
    tris = rng.integers(0, 400, size=(900, 3))
    a, b = mesh_edges(tris)
    w = rng.random((400, 9)) * (rng.random((400, 9)) < 0.5)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    cols = np.zeros(9, bool)
    cols[[1, 4, 6]] = True
    rows = rng.random(400) < 0.1
    for args in ((5, 0.5, None, None), (7, 0.3, rows, None), (9, 0.5, None, cols),
                 (4, 0.7, rows, cols)):  # fmt: skip
        assert np.array_equal(smooth_rows(w, a, b, *args), reference(w, a, b, *args)), args
