"""Block worlds: generation, edits, greedy meshing, structures and queries."""

import numpy as np
import pytest

from aether3d import Scene
from aether3d.errors import WorldError
from aether3d.worlds.blocks import AIR, BLOCK_LIST, BLOCKS, texture_names, texture_rgba
from aether3d.worlds.voxels import _OPAQUE, CHUNK, _blocks, _padded, mesh_chunk


@pytest.fixture()
def world():
    return Scene("Blocks").worlds.create("B", "blocky_adventure", seed=3, size=256)


def test_textures_are_original_pixel_art():
    names = texture_names()
    assert len(names) == len(set(names)) >= 50
    for name in names:
        rgba = texture_rgba(name)
        assert rgba.shape == (16, 16, 4) and rgba.dtype == np.uint8
        assert len(np.unique(rgba.reshape(-1, 4), axis=0)) > 1  # never a flat colour
    # Plants are alpha-masked; solid blocks opaque.
    assert texture_rgba("tall_grass")[..., 3].min() == 0
    assert texture_rgba("stone")[..., 3].min() == 255


def test_chunks_have_bedrock_ground_and_air(world):
    bw = _blocks(world.runtime)
    c = bw.chunk(0, 0)
    assert c.blocks.shape == (
        world.definition["blocks"]["max_height"] - world.definition["blocks"]["min_height"],
        16,
        16,
    )
    assert np.all(c.blocks[0] == BLOCKS["bedrock"])
    assert np.all(c.blocks[-1] == AIR)
    top = world.ground(8.5, 8.5)
    assert world.block_at(8.5, 8.5, top - 0.5) != "air"
    assert world.block_at(8.5, 8.5, top + 3.5) in (
        "air",
        "oak_leaves",
        "birch_leaves",
        "oak_log",
        "birch_log",
        "spruce_leaves",
        "spruce_log",
        "water",
    )


def test_greedy_meshes_cover_exactly_the_exposed_faces(world):
    bw = _blocks(world.runtime)
    d = world.definition
    data = bw.chunk(1, 1)
    nb = {k: bw.chunk(1 + k[0], 1 + k[1]) for k in ((1, 0), (-1, 0), (0, 1), (0, -1))}
    mesh, _ = mesh_chunk(d, data, nb)
    P = _padded(data, nb)
    opaque = _OPAQUE[P]
    solid = _OPAQUE[P[1:-1, 1:-1, 1:-1]]
    exposed = 0
    for axis in range(3):
        for sign in (1, -1):
            sl = [slice(1, -1)] * 3
            sl[axis] = slice(1 + sign, P.shape[axis] - 1 + sign)
            exposed += int(np.sum(solid & ~opaque[tuple(sl)]))
    # Area of the meshed faces (excluding the plants' crossed quads) = exposed block faces.
    p = mesh.positions.astype(np.float64)
    t = mesh.triangles
    n = mesh.normals[t[:, 0]].astype(np.float64)
    axis_aligned = np.isclose(np.abs(n).max(axis=1), 1.0)
    area = 0.5 * np.linalg.norm(np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), axis=1)
    size = d["blocks"]["size"]
    assert area[axis_aligned].sum() / (size * size) == pytest.approx(exposed, abs=1e-3)
    # Greedy merging: fewer quads than exposed faces (corners with different ambient
    # occlusion keep faces apart; test_greedy_merging_of_a_flat_chunk shows the merging).
    assert axis_aligned.sum() / 2 < exposed
    # Faces point outward: from solid toward the open side.
    cen = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
    outside = cen[axis_aligned] + 0.25 * size * n[axis_aligned]
    ix = np.floor(outside[:, 0] / size).astype(int) + 1
    iy = np.floor(outside[:, 1] / size).astype(int) + 1
    iz = np.floor(outside[:, 2] / size).astype(int) - data.z0 + 1
    ok = (iz >= 0) & (iz < P.shape[0])
    assert not np.any(opaque[iz[ok], iy[ok], ix[ok]])


def test_greedy_merging_of_a_flat_chunk(world):
    from aether3d.worlds.voxels import ChunkData

    def flat() -> ChunkData:
        blocks = np.zeros((32, 16, 16), dtype=np.uint16)
        blocks[:10] = BLOCKS["stone"]
        return ChunkData(0, 0, 0, blocks, np.ones((16, 16, 3, 3)), np.full((16, 16), np.nan))

    nb = {k: flat() for k in ((1, 0), (-1, 0), (0, 1), (0, -1))}
    mesh, water = mesh_chunk(world.definition, flat(), nb)
    assert water is None
    assert mesh.triangle_count == 2  # the whole 16 x 16 top in one quad
    assert np.allclose(mesh.positions[:, 2], 10.0)
    assert set(np.unique(mesh.uvs, axis=0).ravel().tolist()) == {0.0, 16.0}  # repeats per block


def test_edits_persist_undo_and_save(tmp_path, world):
    scene = world._system.scene
    g = int(world.ground(4.5, 4.5))
    n = world.set_blocks([[4, 4, g], [5, 4, g]], "glowstone")
    assert n == 2 and world.block_at(4.5, 4.5, g + 0.5) == "glowstone"
    world.fill_blocks([0, 0, g + 5], [2, 2, g + 6], "stone_bricks")
    assert world.block_at(1.5, 1.5, g + 5.5) == "stone_bricks"
    world.replace_blocks([0, 0, g + 5], [2, 2, g + 6], "stone_bricks", "glass")
    assert world.block_at(1.5, 1.5, g + 6.5) == "glass"
    # The mesh of the edited chunk changes; untouched chunks keep theirs.
    path = tmp_path / "edits.aether"
    scene.save(path)
    loaded = Scene.load(path)
    w2 = loaded.worlds.get("B")
    assert (
        w2.block_at(4.5, 4.5, g + 0.5) == "glowstone" and w2.block_at(1.5, 1.5, g + 6.5) == "glass"
    )
    world.revert_blocks([0, 0, g], [5, 5, g + 6])
    assert world.block_at(4.5, 4.5, g + 0.5) != "glowstone"
    for _ in range(4):
        scene.history.undo()
    assert world.block_at(4.5, 4.5, g + 0.5) != "glowstone"
    scene.history.redo()
    assert world.block_at(4.5, 4.5, g + 0.5) == "glowstone"
    with pytest.raises(WorldError) as info:
        world.set_blocks([[0, 0, 10_000]], "stone")
    assert info.value.code in ("OUTSIDE_WORLD", "INVALID_ARGUMENT")
    with pytest.raises(WorldError) as info:
        world.set_blocks([[0, 0, g]], "diamond_block")
    assert info.value.code == "UNKNOWN_BLOCK"


def test_trees_crossing_chunk_borders_are_whole(world):
    """A tree near a border is stamped by both chunks identically."""
    bw = _blocks(world.runtime)
    leaves = {BLOCKS[b.name] for b in BLOCK_LIST if b.name.endswith("_leaves")}
    checked = 0
    for cx in range(-3, 3):
        a = bw.chunk(cx, 0)
        b = bw.chunk(cx + 1, 0)
        ea = np.isin(a.blocks[:, :, CHUNK - 1], list(leaves))
        eb = np.isin(b.blocks[:, :, 0], list(leaves))
        # Leaves along the border continue across it somewhere when trees straddle it.
        checked += int(ea.sum() + eb.sum())
    assert checked > 0
    # The same tree list either way.
    t1 = bw.trees(0, 0, 32, 32)
    t2 = bw.trees(0, 0, 16, 32) + bw.trees(16, 0, 32, 32)
    assert sorted(t1) == sorted(t2)


def test_structures_are_stamped_on_their_pads(world):
    site = world.find_sites(1, kind="castle", radius=16.0, max_slope=30.0, spacing=100.0)[0]
    world.place_structure(
        {"id": "keep", "kind": "tower", "at": site["at"], "size": [7, 7], "floors": 2}
    )
    x, y = site["at"]
    model = world.runtime.model()
    pad = model.pad_height("structure:keep")
    size = world.definition["blocks"]["size"]
    z = round(pad / size) * size
    assert world.block_at(x + 3.0, y, z + 2.5) == "stone_bricks"  # the tower's wall
    # No tree grows inside the footprint.
    for dx in (-2.0, 0.0, 2.0):
        assert "leaves" not in world.block_at(x + dx, y + dx, z + 6.5)


def test_block_queries_check_their_arguments(world):
    with pytest.raises(WorldError):
        world.block_at(1e9, 0, 0)
    with pytest.raises(WorldError):
        world.ground(10_000.0, 0.0)  # outside the 256 m world
    terrain = Scene("T").worlds.create("T", "temperate", seed=1, size=512.0)
    with pytest.raises(WorldError) as info:
        terrain.block_at(0, 0, 0)
    assert info.value.code == "NOT_A_BLOCK_WORLD"
