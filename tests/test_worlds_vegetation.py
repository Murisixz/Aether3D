"""Vegetation and structures: placement rules, stable ids, nested thinning, clearings;
structure items, pads that follow the natural ground, settlements, roads and bridges."""

import math

import numpy as np
import pytest

from aether3d import Scene
from aether3d.errors import WorldError
from aether3d.worlds.assets import LAYERS, asset
from aether3d.worlds.structures import footprint
from aether3d.worlds.tiles import View
from aether3d.worlds.vegetation import CLASSES, SPECIES_ORDER, THIN, vegetation_of

SMALL = {"hydrology": {"cell": 32.0}}


@pytest.fixture(scope="module")
def world():
    return Scene("Plants").worlds.create("W", "temperate", seed=7, size=2048.0, overrides=SMALL)


def test_every_species_has_three_levels_of_detail():
    for species in LAYERS:
        counts = [asset(species, 0, lod, 0).mesh.triangle_count for lod in range(3)]
        assert counts[0] >= counts[1] >= counts[2] > 0, species
        a = asset(species, 1, 0, 0)
        assert a.mesh.colors is not None and a.height > 0
        assert set(np.unique(a.mesh.groups)) <= {0, 1}


def test_plants_never_stand_in_water_on_roads_or_steep_slopes(world):
    veg = vegetation_of(world.runtime)
    for cls in CLASSES:
        for i in range(-3, 3):
            ps = veg.tile(i, i, cls, 0)
            if not ps.count:
                continue
            s = world.runtime.model().evaluate(ps.pos[:, 0], ps.pos[:, 1])
            names = np.array(SPECIES_ORDER)[ps.species]
            wet = ~np.isnan(s["water"]) & (s["water"] > s["height"] - 0.2)
            assert not np.any(wet & (names != "reeds")), cls
            assert np.all(s["road"] < 0.3)
            assert np.allclose(ps.pos[:, 2], s["height"] - 0.15)  # sunk a little


def test_ids_are_stable_and_unique(world):
    veg = vegetation_of(world.runtime)
    ids = []
    for cls in CLASSES:
        ids += veg.tile(0, 0, cls, 0).ids()
    assert len(ids) == len(set(ids))
    again = Scene("Again").worlds.create("W", "temperate", seed=7, size=2048.0, overrides=SMALL)
    assert (
        vegetation_of(again.runtime).tile(0, 0, "trees", 0).ids()
        == veg.tile(0, 0, "trees", 0).ids()
    )


def test_thinning_levels_are_nested(world):
    """A plant drawn far away is drawn at every shorter distance: no popping in."""
    veg = vegetation_of(world.runtime)
    full = set(veg.tile(1, -2, "trees", 0).ids())
    previous = full
    for level in range(1, 4):
        thinned = set(veg.tile(1, -2, "trees", level).ids())
        assert thinned <= previous
        previous = thinned
    assert len(previous) < len(full)


def test_visible_plants_thin_with_distance(world):
    veg = vegetation_of(world.runtime)
    eye = np.array([0.0, 0.0, world.height(0.0, 0.0) + 30.0])
    view = View(eye, np.array([0.0, 1.0, 0.0]), 900.0, 3000.0, None)
    vis = veg.visible(view)
    d = np.linalg.norm(vis["pos"][:, :2] - eye[:2], axis=1)
    trees = np.isin(
        np.array(SPECIES_ORDER)[vis["species"]],
        ["oak", "beech", "birch", "maple", "spruce", "pine"],
    )
    near = np.sum(trees & (d < THIN["trees"])) / (math.pi * THIN["trees"] ** 2)
    far = np.sum(trees & (d > 2 * THIN["trees"]) & (d < 3 * THIN["trees"])) / (
        math.pi * 5 * THIN["trees"] ** 2
    )
    assert near > 3 * far > 0
    assert set(np.unique(vis["lod"]).tolist()) <= {0, 1, 2}


def test_clear_features_and_pads_clear_the_ground():
    scene = Scene("Clear")
    w = scene.worlds.create("W", "temperate", seed=7, size=2048.0, overrides=SMALL)
    veg = vegetation_of(w.runtime)
    before = veg.tile(0, 0, "trees", 0)
    assert before.count > 0
    w.add_feature(
        {
            "type": "clear",
            "id": "glade",
            "shape": {"rect": [0, 0, 128, 128]},
            "what": ["trees"],
            "falloff": 0.5,
        }
    )
    after = vegetation_of(w.runtime).tile(0, 0, "trees", 0)
    inside = (
        (after.pos[:, 0] > 2)
        & (after.pos[:, 0] < 126)
        & (after.pos[:, 1] > 2)
        & (after.pos[:, 1] < 126)
    )
    assert not np.any(inside)
    scene.history.undo()
    assert vegetation_of(w.runtime).tile(0, 0, "trees", 0).ids() == before.ids()


def test_structure_pads_follow_the_natural_ground():
    scene = Scene("Pads")
    w = scene.worlds.create("W", "temperate", seed=7, size=2048.0, overrides=SMALL)
    item = w.place_structure(
        {
            "id": "inn",
            "kind": "house",
            "at": [200.0, -150.0],
            "rotation": 30.0,
            "size": [8.0, 6.0],
            "floors": 2,
        }
    )
    model = w.runtime.model()
    pts = np.asarray(footprint(item, -0.5))
    centre = pts.mean(axis=0)
    samples = np.concatenate([pts, (pts + centre) / 2.0, centre[None]])
    expect = float(np.median(model.natural(samples[:, 0], samples[:, 1])))
    assert model.pad_height("structure:inn") == pytest.approx(expect, abs=1e-4)
    assert w.height(200.0, -150.0) == pytest.approx(round(expect, 4), abs=1e-6)  # flattened
    # Regenerate the world under it: the pad follows the new ground.
    w.configure(seed=8)
    model = w.runtime.model()
    expect2 = float(np.median(model.natural(samples[:, 0], samples[:, 1])))
    assert expect2 != pytest.approx(expect, abs=0.5)
    assert w.height(200.0, -150.0) == pytest.approx(round(expect2, 4), abs=1e-6)
    with pytest.raises(WorldError) as info:
        w.place_structure({"id": "inn", "kind": "tower", "at": [0, 0]})
    assert info.value.code == "DUPLICATE_ID"


def test_settlements_are_valid_and_undo_as_one_step():
    scene = Scene("Villages")
    w = scene.worlds.create("W", "rolling_plains", seed=4, size=4096.0, overrides=SMALL)
    steps = len(scene.history.entries)
    summary = w.build_settlements(2, houses=[6, 10])
    assert len(scene.history.entries) == steps + 1
    assert len(summary["settlements"]) >= 1
    items = w.structures("house")
    assert items
    model = w.runtime.model()
    polys = [np.asarray(footprint(i, 0.0)) for i in items]
    from aether3d.worlds.structures import _overlap

    for a in range(len(polys)):  # no two houses overlap
        for b in range(a + 1, len(polys)):
            assert not _overlap(polys[a].tolist(), polys[b].tolist())
    for poly in polys:  # never on water
        s = model.evaluate(poly[:, 0], poly[:, 1])
        assert np.all(np.isnan(s["water"]))
    roads = [f for f in w.features("road")]
    assert roads and all(f["shape"]["width"] > 0 for f in roads)
    scene.history.undo()
    assert w.structures() == [] and w.features("road") == []


def test_roads_bridge_rivers():
    scene = Scene("Roads")
    w = scene.worlds.create("W", "alpine_valley", seed=7, size=4096.0, overrides=SMALL)
    reg = w.runtime.regional()
    inside = [s for s in reg.segments if max(abs(s[0]), abs(s[1]), abs(s[2]), abs(s[3])) < 1500]
    seg = max(inside, key=lambda s: max(s[6], s[7]))  # the widest river segment inside
    mx, my = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    n = math.hypot(dx, dy)
    nx, ny = -dy / n, dx / n
    a = [mx - 80 * nx, my - 80 * ny]
    b = [mx + 80 * nx, my + 80 * ny]
    plan = w.route(a, b, cell=8.0)
    assert plan["length"] > 150
    assert any(i["kind"] == "bridge" for i in plan["items"])


def test_landmarks_and_the_bridges_setting():
    scene = Scene("Landmarks")
    w = scene.worlds.create(
        "W",
        "rolling_plains",
        seed=4,
        size=4096.0,
        overrides={**SMALL, "structures": {"towers": 1, "ruins": 1, "castles": 1}},
    )
    summary = w.build_settlements(1, houses=[4, 6])
    kinds = {i["kind"] for i in w.structures()}
    assert {"castle", "tower", "ruin"} <= kinds and summary["towers"] and summary["ruins"]
    alps = Scene("Alps").worlds.create(
        "A",
        "alpine_valley",
        seed=7,
        size=4096.0,
        overrides={**SMALL, "structures": {"bridges": False}},
    )
    reg = alps.runtime.regional()
    inside = [s for s in reg.segments if max(abs(s[0]), abs(s[1]), abs(s[2]), abs(s[3])) < 1500]
    seg = max(inside, key=lambda s: max(s[6], s[7]))
    mx, my = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
    n = math.hypot(seg[2] - seg[0], seg[3] - seg[1])
    nx, ny = -(seg[3] - seg[1]) / n, (seg[2] - seg[0]) / n
    from aether3d.worlds.structures import plan_road

    with pytest.raises(WorldError) as info:
        plan_road(
            alps.runtime.model(),
            [mx - 80 * nx, my - 80 * ny],
            [mx + 80 * nx, my + 80 * ny],
            prefix="r",
            cell=8.0,
            bridges=False,
        )
    assert info.value.code == "NO_ROUTE"
