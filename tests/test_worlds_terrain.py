"""Terrain worlds: tiles, seams, level of detail, water, queries, caves and holes."""

import math

import numpy as np
import pytest

from aether3d import Scene
from aether3d.errors import WorldError
from aether3d.worlds.tiles import STITCH_BITS, TileKey, View, restrict, stitch_masks
from aether3d.worlds.volumes import CHUNK as VCHUNK
from aether3d.worlds.volumes import volumes_of

SMALL = {"hydrology": {"cell": 32.0}}


@pytest.fixture(scope="module")
def world():
    scene = Scene("Terrain")
    return scene.worlds.create("Valley", "temperate", seed=13, size=2048.0, overrides=SMALL)


def edge(td, side):
    h = td.heights
    return {"-x": h[:, 0], "+x": h[:, -1], "-y": h[0, :], "+y": h[-1, :]}[side]


def test_neighbouring_tiles_share_their_edges_exactly(world):
    rt = world.runtime
    for level in (0, 2):
        a = rt.tile(TileKey(level, 0, 0))
        b = rt.tile(TileKey(level, 1, 0))
        c = rt.tile(TileKey(level, 0, 1))
        assert np.array_equal(edge(a, "+x"), edge(b, "-x"))
        assert np.array_equal(edge(a, "+y"), edge(c, "-y"))


def test_tile_vertices_are_the_authoritative_heights(world):
    rt = world.runtime
    key = TileKey(1, -1, 2)
    td = rt.tile(key)
    xs, ys = rt.grid().coords(key)
    X, Y = np.meshgrid(xs, ys)
    exact = np.asarray(world.height(X.reshape(-1), Y.reshape(-1))).reshape(X.shape)
    assert np.array_equal(td.heights, exact)
    # The reported error bounds the mesh against the authoritative surface at midpoints.
    mx = (xs[:-1] + xs[1:]) / 2
    my = (ys[:-1] + ys[1:]) / 2
    MX, MY = np.meshgrid(mx, my)
    mid = np.asarray(world.height(MX.reshape(-1), MY.reshape(-1))).reshape(MX.shape)
    interp = (td.heights[:-1, :-1] + td.heights[1:, 1:]) / 2  # (on a diagonal of each quad)
    assert np.abs(mid - interp).max() <= max(4.0 * td.error, 0.05) + 1e-9


def test_stitching_makes_a_finer_edge_follow_its_coarser_neighbour(world):
    """The drawn edge of a finer tile next to a coarser one: its even vertices are the
    coarse vertices, its odd ones on the coarse edge's segments (no cracks)."""
    rt = world.runtime
    r = rt.grid().resolution
    fine, coarse = TileKey(0, 3, 0), TileKey(1, 2, 0)  # fine's +x side borders coarse
    stitched = rt.tile(fine, STITCH_BITS["+x"])
    size = rt.grid().size(0)
    p = stitched.mesh.positions[: (r + 1) * (r + 1)].astype(np.float64)  # (skirts follow)
    on_edge = np.abs(p[:, 0] - size) < 1e-6
    edge_f = p[on_edge][np.argsort(p[on_edge][:, 1]), 2]
    edge_c = edge(rt.tile(coarse), "-x")[: r // 2 + 1].astype(np.float32).astype(np.float64)
    assert np.array_equal(edge_f[::2], edge_c)
    assert np.allclose(edge_f[1::2], (edge_c[:-1] + edge_c[1:]) / 2, atol=1e-4)
    # The authoritative heights of the tile stay the generated ones.
    assert np.array_equal(stitched.heights, rt.tile(fine).heights)


def test_restricted_quadtree_and_masks():
    tiles = [TileKey(3, 0, 0), TileKey(0, 8, 0)]
    out = restrict(tiles)
    by = {(k.i, k.j, k.level) for k in out}
    for k in out:  # every edge neighbour differs by at most one level
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = k.i + di, k.j + dj
            for other in out:
                span = 2 ** (other.level - k.level) if other.level >= k.level else None
                if (
                    span
                    and other.i * span <= ni < (other.i + 1) * span
                    and other.j * span <= nj < (other.j + 1) * span
                ):
                    assert other.level - k.level <= 1
    masks = stitch_masks(out)
    assert all(0 <= m < 16 for m in masks.values())
    assert by  # (non-empty)


def test_lod_selection_refines_near_the_camera_and_respects_limits(world):
    rt = world.runtime
    eye = np.array([0.0, -900.0, 400.0])
    view = View(eye, np.array([0.0, 0.8, -0.6]) / 1.0, 900.0, 5000.0, None)
    tiles = rt.select(view)
    near = min(tiles, key=lambda t: math.dist(rt.grid().bounds(t[0])[:2], eye[:2]))
    far = max(tiles, key=lambda t: math.dist(rt.grid().bounds(t[0])[:2], eye[:2]))
    assert near[0].level <= far[0].level
    # A tighter error budget never gives fewer tiles.
    world.configure(streaming={"lod_error": 1.0}, label="finer")
    finer = rt.select(view)
    assert len(finer) >= len(tiles)
    world.configure(streaming={"lod_error": 3.0, "max_tiles": 64}, label="limited")
    assert len(rt.select(view)) <= 64 + 64  # (coarser tiles kept beyond the limit; restricted)
    world._system.scene.history.undo()
    world._system.scene.history.undo()


def test_rivers_and_lakes_follow_the_drainage():
    scene = Scene("Water")
    w = scene.worlds.create("Alps", "alpine_valley", seed=7, size=4096.0, overrides=SMALL)
    reg = w.runtime.regional()
    summary = reg.summary()
    assert summary["rivers"] > 0
    for r in reg.rivers:  # water levels never rise downstream
        lev = r["level"]
        assert np.all(np.diff(lev) <= 1e-9)
    # Water at river points lies at or above the bed, below the banks' top.
    seg = reg.segments[0]
    p = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
    s = w.sample([list(p)], fields=["height", "water", "water_kind"])
    assert s["water"][0] is not None and s["water"][0] >= s["height"][0] - 1e-6


def test_queries(world):
    s = world.sample([[0, 0], [100.5, -20]], fields=["height", "slope", "biome", "normal", "water"])
    assert len(s["height"]) == 2 and 0 <= s["slope"][0] <= 90
    assert s["biome"][0] in world.info()["biomes"]
    n = np.asarray(s["normal"][0])
    assert abs(np.linalg.norm(n) - 1) < 1e-5 and n[2] > 0
    p = world.profile([[-500, 0], [500, 0]], samples=17)
    assert len(p["height"]) == 17 and p["distance"][-1] == pytest.approx(1000.0)
    assert world.height(0.0, 0.0) == pytest.approx(s["height"][0])
    with pytest.raises(WorldError) as info:
        world.sample([[0, 0]], fields=["nonsense"])
    assert info.value.code in ("INVALID_ARGUMENT", "UNKNOWN_FIELD")
    with pytest.raises(WorldError):
        world.sample([[float("nan"), 0]])
    with pytest.raises(WorldError):
        world.sample([[5e6, 0]])  # outside a bounded world


def test_features_change_the_ground_locally(world):
    scene = world._system.scene
    before = world.height(300.0, 300.0)
    far_before = world.height(-800.0, -800.0)
    world.add_feature(
        {"type": "flatten", "id": "yard", "shape": {"circle": [300, 300, 40]}, "height": 123.0}
    )
    assert world.height(300.0, 300.0) == pytest.approx(123.0)
    assert world.height(-800.0, -800.0) == far_before  # elsewhere: the same bits
    world.add_feature(
        {
            "type": "lake",
            "id": "pond",
            "shape": {"circle": [-300, 300, 50]},
            "level": 50.0,
            "depth": 3.0,
        }
    )
    s = world.sample([[-300, 300]], fields=["water", "water_kind"])
    assert s["water"][0] == 50.0 and s["water_kind"][0] == "authored"
    scene.history.undo()
    scene.history.undo()
    assert world.height(300.0, 300.0) == before


def test_cave_entrances_cut_the_tiles_and_meet_them():
    scene = Scene("Caves")
    w = scene.worlds.create("C", "cavernous_highlands", seed=3, size=1024.0, overrides=SMALL)
    rt = w.runtime
    vol = volumes_of(rt)
    found = None
    for ci in range(-8, 8):
        for cj in range(-8, 8):
            for ck, flags in vol.column(ci, cj):
                if flags & 1:
                    found = (ci, cj, ck)
    assert found is not None
    ci, cj, ck = found
    mesh = vol.cube(ci, cj, ck, 0)
    assert mesh is not None and mesh.triangle_count > 0
    # Ground vertices of the volume on the footprint's edge lie on the authoritative ground
    # (the tiles' vertices there), so the two surfaces meet.
    p = mesh.positions.astype(np.float64)
    on_edge = (np.abs(p[:, 0]) < 1e-6) & (np.abs(p[:, 1] - np.round(p[:, 1])) < 1e-6)
    x = p[on_edge, 0] + ci * VCHUNK
    y = p[on_edge, 1] + cj * VCHUNK
    if x.size:
        h = np.asarray(w.height(x, y))
        z = p[on_edge, 2]
        smp = rt.model().evaluate(x, y)
        rock = vol.cave_field(x, y, z - 0.5, smp["height"], ~np.isnan(smp["water"])) > 1.0
        near_ground = (np.abs(z - h) < 0.2) & rock  # (where no cave air reaches the edge)
        assert near_ground.any()
        assert np.abs(z[near_ground] - h[near_ground]).max() < 0.02  # 1% of a voxel
    # Points in cave air are not solid; points deep in rock are.
    assert not vol.solid([[0.0, 0.0, 10_000.0]])[0]
    assert vol.solid([[0.0, 0.0, w.height(0.0, 0.0) - 500.0]])[0]


def test_cave_features_carve_tunnels():
    scene = Scene("Tunnel")
    w = scene.worlds.create("T", "temperate", seed=4, size=1024.0, overrides=SMALL)
    g = w.height(0.0, 0.0)
    w.add_feature(
        {
            "type": "cave",
            "id": "t1",
            "kind": "tunnel",
            "radius": 4.0,
            "noise": 0.0,
            "shape": {"line": [[-60, 0, g - 20], [60, 0, g - 20]], "width": 8},
        }
    )
    vol = volumes_of(w.runtime)
    assert vol.enabled()
    assert not vol.solid([[0.0, 0.0, g - 20.0]])[0]  # inside the tunnel
    assert vol.solid([[0.0, 0.0, g - 30.0]])[0]  # below it
    assert vol.solid([[0.0, 30.0, g - 20.0]])[0]  # beside it


def test_neighbouring_overhang_cubes_meet_on_their_shared_faces():
    """Overhang warps fade by the distance to the footprints left to the tiles, so two
    drawn neighbours compute the same field on their shared face: the same vertices there
    (no crack), whatever their other neighbours are."""
    from aether3d.worlds.volumes import NEIGHBOURS

    scene = Scene("Mesa")
    w = scene.worlds.create("M", "cavernous_highlands", seed=3, size=1024.0, overrides=SMALL)
    top = w.height(100.0, 60.0) + 28.0
    w.add_feature(
        {"type": "pad", "id": "mesa", "shape": {"rect": [60.0, 32.0, 140.0, 88.0]},
         "height": round(top, 2), "falloff": 5.0}
    )  # fmt: skip
    vol = volumes_of(w.runtime)
    cut = {
        (ci, cj)
        for ci in range(-2, 8)
        for cj in range(-2, 6)
        if any(flags for _, flags in vol.column(ci, cj))
    }

    def mask(ci, cj):
        return sum(1 << b for b, (a, c) in enumerate(NEIGHBOURS) if (ci + a, cj + c) in cut)

    def face(mesh, axis, value):
        p = mesh.positions.astype(np.float64)
        on = np.abs(p[:, axis] - value) < 1e-9
        return {tuple(np.round(q, 9)) for q in np.delete(p[on], axis, axis=1)}

    compared = 0
    for ci, cj in sorted(cut):
        for a, b, axis in ((1, 0, 0), (0, 1, 1)):
            if (ci + a, cj + b) not in cut:
                continue
            ks = {k for k, f in vol.column(ci, cj) if f & 2} & {
                k for k, f in vol.column(ci + a, cj + b) if f & 2
            }
            for ck in sorted(ks):
                m1 = vol.cube(ci, cj, ck, mask(ci, cj))
                m2 = vol.cube(ci + a, cj + b, ck, mask(ci + a, cj + b))
                if m1 is None or m2 is None:
                    continue
                f1, f2 = face(m1, axis, VCHUNK), face(m2, axis, 0.0)
                assert f1 == f2, (ci, cj, ck, axis, len(f1 ^ f2))
                compared += len(f1)
    assert compared > 50  # the mesa's walls cross footprint borders


def test_cave_surface_columns_draw_every_cube_the_ground_crosses():
    scene = Scene("Caves")
    w = scene.worlds.create("C", "cavernous_highlands", seed=3, size=1024.0, overrides=SMALL)
    vol = volumes_of(w.runtime)
    m = w.runtime.model()
    checked = 0
    for ci in range(-8, 8):
        for cj in range(-8, 8):
            col = vol.column(ci, cj)
            if not any(f & 1 for _, f in col):
                continue
            g = np.arange(0.0, VCHUNK + 1.0)
            X, Y = np.meshgrid(ci * VCHUNK + g, cj * VCHUNK + g)
            h = m.evaluate(X.ravel(), Y.ravel())["height"]
            need = range(math.floor(h.min() / VCHUNK), math.floor(h.max() / VCHUNK) + 1)
            assert set(need) <= {k for k, f in col if f}, (ci, cj)
            checked += 1
    assert checked > 0


def test_caches_keep_their_budgets_and_regenerate_the_same_bits():
    """``streaming.memory_mb`` bounds what stays cached between views (tiles 60 %, plants
    25 %, cave volumes 15 %); least recently used entries go first, and an evicted tile
    comes back bit for bit."""
    from aether3d.worlds.runtime import TILE_SHARE
    from aether3d.worlds.vegetation import PLANT_SHARE, vegetation_of

    scene = Scene("Budget")
    w = scene.worlds.create(
        "B", "temperate", seed=5, size=4096.0,
        overrides={**SMALL, "streaming": {"memory_mb": 16.0, "view_distance": 1500.0}},
    )  # fmt: skip
    rt = w.runtime
    veg = vegetation_of(rt)
    first = None
    for x in np.linspace(-1600.0, 1600.0, 9):
        eye = np.array([x, -1200.0, w.height(x, -1200.0) + 60.0])
        view = View(eye, np.array([0.0, 0.96, -0.28]), 900.0, 1500.0, None)
        tiles = rt.select(view)
        if first is None:
            first_key, _, data = tiles[0]
            first = data.heights.copy()
        veg.visible(view)
        budget = 16 * 2**20
        assert rt.stats()["tile_bytes"] <= max(TILE_SHARE * budget, sum(t[2].nbytes for t in tiles))
        stats = veg.stats()
        assert stats["bytes"] <= PLANT_SHARE * budget or stats["tiles_cached"] == 1
    assert rt.counters["tiles_evicted"] > 0 and veg.stats()["tiles_evicted"] > 0
    assert np.array_equal(rt.tile(first_key).heights, first)
