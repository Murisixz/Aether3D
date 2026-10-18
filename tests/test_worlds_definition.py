"""World definitions: presets, validation with located errors, defaults, digests."""

import copy
import json

import pytest

from aether3d import Scene
from aether3d.errors import WorldError
from aether3d.worlds import definition as D


def test_every_preset_builds_a_valid_definition_that_checks_as_itself():
    for name in D.PRESETS:
        d = D.build_definition(name, seed=11)
        assert d["seed"] == 11
        again = D.check_definition(json.loads(json.dumps(d)))
        assert again == d, name  # normalizing a normalized definition changes nothing
        assert D.digest(again) == D.digest(d)
    assert set(D.describe_presets()) == set(D.PRESETS)


def test_build_definition_options():
    d = D.build_definition("temperate", seed=3, size=2000.0)
    assert d["extent"] == {"min": [-1000.0, -1000.0], "max": [1000.0, 1000.0]}
    d = D.build_definition("blocky_adventure", seed=1)
    assert d["mode"] == "blocks"
    d = D.build_definition(overrides={"terrain": {"mountains": {"height": 1234.0}}})
    assert d["terrain"]["mountains"]["height"] == 1234.0
    assert d["terrain"]["hills"] == D.DEFAULTS["terrain"]["hills"]  # merged, not replaced
    with pytest.raises(WorldError) as info:
        D.build_definition("nowhere")
    assert info.value.code in ("UNKNOWN_PRESET", "INVALID_DEFINITION")


@pytest.mark.parametrize(
    "patch, where, message",
    [
        ({"terrain": {"mountains": {"height": -1}}}, "terrain.mountains.height", "outside"),
        ({"terrain": {"mountains": {"height": "tall"}}}, "terrain.mountains.height", "number"),
        ({"terrain": {"nope": 1}}, "terrain.nope", "unknown key"),
        ({"seed": 2**60}, "seed", "outside"),
        ({"seed": 1.5}, "seed", "integer"),
        ({"sea_level": float("nan")}, "sea_level", "finite"),
        ({"mode": "voxels"}, "mode", "expected one of"),
        ({"streaming": {"chunk": 48.0}}, "streaming.chunk", "power of two"),
        ({"blocks": {"min_height": 0, "max_height": 8}}, "blocks.max_height", "at least 16"),
        ({"biomes": {"disabled": ["atlantis"]}}, "biomes.disabled[0]", "unknown biome"),
        ({"structures": {"biomes": ["moon"]}}, "structures.biomes[0]", "unknown biome"),
        (
            {"features": [{"type": "lake", "id": "a", "shape": {"circle": [0, 0, -1]}}]},
            "features[0].shape.circle[2]",
            "positive",
        ),
        (
            {"features": [{"type": "volcano", "id": "a", "shape": {"circle": [0, 0, 1]}}]},
            "features[0].type",
            "unknown feature type",
        ),
        ({"features": [{"type": "lake", "shape": {"circle": [0, 0, 1]}}]}, "features[0].id", "id"),
        (
            {
                "features": [
                    {"type": "lake", "id": "x", "shape": {"circle": [0, 0, 5]}},
                    {"type": "lake", "id": "x", "shape": {"circle": [9, 0, 5]}},
                ]
            },
            "features[1].id",
            "duplicate",
        ),
        (
            {"structures": {"items": [{"id": "h", "kind": "house"}]}},
            "structures.items[0].at",
            "position",
        ),
        (
            {"structures": {"items": [{"id": "h", "kind": "spaceship", "at": [0, 0]}]}},
            "structures.items[0].kind",
            "unknown structure kind",
        ),
        (
            {"structures": {"items": [{"id": "b", "kind": "bridge", "line": [[0, 0], [1000, 0]]}]}},
            "structures.items[0].line",
            "400",
        ),
        ({"extent": {"min": [0, 0], "max": [0, 10]}}, "extent", ""),
        ({"hydrology": {"rivers": {"threshold": 0}}}, "hydrology.rivers.threshold", "outside"),
    ],
)
def test_invalid_definitions_are_rejected_with_their_location(patch, where, message):
    base = D.build_definition("temperate")
    with pytest.raises(WorldError) as info:
        D.check_definition(D.merge(base, patch), "definition")
    assert info.value.code in ("INVALID_DEFINITION", "UNKNOWN_BIOME")
    assert info.value.location.startswith(f"definition.{where}"), info.value.location
    assert message in info.value.message


def test_an_unbounded_drainage_world_needs_a_hydrology_extent():
    with pytest.raises(WorldError) as info:
        D.build_definition("temperate", extent=None)
    assert "hydrology.extent" in info.value.location
    ok = D.build_definition("temperate", extent=None, overrides={"hydrology": {"mode": "contour"}})
    assert ok["extent"] is None
    ok = D.build_definition(
        "temperate", extent=None, overrides={"hydrology": {"extent": [-2000, -2000, 2000, 2000]}}
    )
    assert ok["extent"] is None and ok["hydrology"]["mode"] == "drainage"
    assert D.build_definition("infinite_highlands")["extent"] is None


def test_digest_is_a_content_key():
    a = D.build_definition("temperate", seed=1)
    b = copy.deepcopy(a)
    assert D.digest(a) == D.digest(b)
    b["terrain"]["hills"]["height"] += 1.0
    assert D.digest(a) != D.digest(b)
    # Key order does not matter.
    assert D.digest(json.loads(json.dumps(a, sort_keys=True))) == D.digest(a)


def test_world_handles_report_and_configure_atomically():
    scene = Scene("Defs")
    w = scene.worlds.create("Valley", "temperate", seed=5, size=1500)
    before = w.definition
    with pytest.raises(WorldError) as info:
        w.configure({"terrain": {"mountains": {"height": -3}}})
    assert info.value.location.startswith("worlds[Valley].definition.terrain.mountains.height")
    assert w.definition == before and len(scene.history.entries) == 1  # only the creation
    w.configure(terrain={"mountains": {"height": 1500.0}})
    assert w.definition["terrain"]["mountains"]["height"] == 1500.0
    with pytest.raises(WorldError):
        w.configure(mode="blocks")  # a world's mode is fixed
    scene.history.undo()
    assert w.definition == before
    info = w.info()
    assert info["mode"] == "terrain" and info["seed"] == 5 and info["area_km2"] == 2.25
    assert "temperate_forest" in info["biomes"]
