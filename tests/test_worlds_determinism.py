"""World generation is deterministic: golden hashes, and results that never depend on the
order of generation, batching, threads or the native kernels.

The golden hashes are of exact float64 bits. Generation uses only correctly rounded
operations (``+ - * /``, ``sqrt``, ``floor``) and integer hashing (docs/WORLDS.md), so the
same definition gives the same bits on every platform; CI runs these tests on Linux,
macOS and Windows, with and without the native package. A change that alters what a
definition generates must bump ``definition.GENERATOR`` and these values together.
"""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from aether3d import Scene
from aether3d.worlds.vegetation import vegetation_of
from aether3d.worlds.voxels import _blocks

CASES = {
    "temperate": ("temperate", {"size": 1024.0}, {"hydrology": {"cell": 32.0}}),
    "alpine": ("alpine_valley", {"size": 2048.0}, {"hydrology": {"cell": 32.0}}),
    "contour": ("infinite_highlands", {}, {}),
    "archipelago": ("archipelago", {"size": 2048.0}, {"hydrology": {"cell": 32.0}}),
}
GOLDEN = {
    "temperate": "978008d45a922bffab846242",
    "alpine": "186afc3984af4515e74748f8",
    "contour": "8a35dd1172ea4386a48e9936",
    "archipelago": "102d53658cd26932e6dd1936",
    "blocks": "c58da7a65b36898e6f8839e7",
    "block_mesh": "f745cbe7913a602db946f605",
    "plants": "00ade4c7f787384506fc49bb",
}


def fields_hash(preset, kw, overrides, seed=7):
    scene = Scene("G")
    w = scene.worlds.create("W", preset, seed=seed, overrides=overrides, **kw)
    e = w.extent or (-1500.0, -1500.0, 1500.0, 1500.0)
    x, y = np.meshgrid(np.linspace(e[0], e[2], 33), np.linspace(e[1], e[3], 33))
    s = w.runtime.model().evaluate(x.reshape(-1), y.reshape(-1))
    blob = b"".join(
        np.ascontiguousarray(s[k], dtype="<f8").tobytes()
        for k in ("height", "temperature", "moisture", "weights")
    )
    blob += np.nan_to_num(np.ascontiguousarray(s["water"], dtype="<f8"), nan=-1e300).tobytes()
    return hashlib.sha256(blob).hexdigest()[:24]


@pytest.mark.parametrize("name", list(CASES))
def test_golden_fields(name):
    assert fields_hash(*CASES[name]) == GOLDEN[name]


def test_golden_blocks_and_plants():
    scene = Scene("B")
    b = scene.worlds.create("B", "blocky_adventure", seed=3, size=256)
    bw = _blocks(b.runtime)
    chunk = bw.chunk(1, -2)
    assert hashlib.sha256(chunk.blocks.astype("<u2").tobytes()).hexdigest()[:24] == GOLDEN["blocks"]
    mesh, _ = bw.mesh(1, -2)
    blob = mesh.positions.astype("<f4").tobytes() + mesh.triangles.astype("<i8").tobytes()
    assert hashlib.sha256(blob).hexdigest()[:24] == GOLDEN["block_mesh"]
    v = scene.worlds.create(
        "V", "temperate", seed=7, size=1024.0, overrides={"hydrology": {"cell": 32.0}}
    )
    ps = vegetation_of(v.runtime).tile(0, 0, "trees", 0)
    blob = "|".join(ps.ids()).encode() + ps.pos.astype("<f8").tobytes()
    assert hashlib.sha256(blob).hexdigest()[:24] == GOLDEN["plants"]


def test_the_numpy_twins_give_the_golden_values_too():
    """The same hashes without the native package (a fresh process)."""
    code = (
        "import sys; sys.path.insert(0, 'tests');"
        "from test_worlds_determinism import fields_hash, CASES, GOLDEN;"
        "assert fields_hash(*CASES['alpine']) == GOLDEN['alpine'], 'alpine';"
        "print('ok')"
    )
    env = dict(os.environ, AETHER3D_NATIVE="0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_points_do_not_depend_on_the_batch_they_are_evaluated_in():
    scene = Scene("P")
    w = scene.worlds.create(
        "W", "temperate", seed=2, size=1024.0, overrides={"hydrology": {"cell": 32.0}}
    )
    m = w.runtime.model()
    rng = np.random.default_rng(4)
    x, y = rng.uniform(-500, 500, 400), rng.uniform(-500, 500, 400)
    whole = m.evaluate(x, y)["height"]
    parts = np.concatenate(
        [m.evaluate(x[k : k + 37], y[k : k + 37])["height"] for k in range(0, 400, 37)]
    )
    assert np.array_equal(whole, parts)
    single = np.array(
        [m.evaluate(x[k : k + 1], y[k : k + 1])["height"][0] for k in range(0, 400, 50)]
    )
    assert np.array_equal(single, whole[::50])


def test_tiles_do_not_depend_on_generation_order():
    from aether3d.worlds.tiles import TileKey

    def heights(order):
        scene = Scene("T")
        w = scene.worlds.create(
            "W", "temperate", seed=9, size=1024.0, overrides={"hydrology": {"cell": 32.0}}
        )
        rt = w.runtime
        return {k: rt.tile(k).heights.copy() for k in order}

    keys = [TileKey(0, 1, 1), TileKey(2, 0, 0), TileKey(1, -1, 0), TileKey(0, -3, 2)]
    a = heights(keys)
    b = heights(keys[::-1])
    assert all(np.array_equal(a[k], b[k]) for k in keys)


def test_vegetation_batches_and_single_tiles_agree():
    scene = Scene("V")
    w = scene.worlds.create(
        "W", "temperate", seed=7, size=1024.0, overrides={"hydrology": {"cell": 32.0}}
    )
    veg = vegetation_of(w.runtime)
    requests = [("trees", i, j, 0) for i in (-2, -1, 0) for j in (0, 1)] + [("cover", 0, 0, 0)]
    batch = veg.tiles(requests)
    other = Scene("V2").worlds.create(
        "W", "temperate", seed=7, size=1024.0, overrides={"hydrology": {"cell": 32.0}}
    )
    veg2 = vegetation_of(other.runtime)
    for req, ps in zip(requests, batch, strict=True):
        alone = veg2.tile(req[1], req[2], req[0], req[3])
        assert alone.ids() == ps.ids()
        assert np.array_equal(alone.pos, ps.pos) and np.array_equal(alone.scale, ps.scale)


def test_block_chunks_do_not_depend_on_threads_or_batches():
    def build(threads, coords):
        scene = Scene("B")
        b = scene.worlds.create(
            "B", "blocky_adventure", seed=5, size=256, overrides={"streaming": {"threads": threads}}
        )
        bw = _blocks(b.runtime)
        chunks = bw.chunks(coords)
        meshes = bw.meshes(coords)
        return [c.blocks for c in chunks], [
            m[0].positions if m[0] is not None else None for m in meshes
        ]

    coords = [(0, 0), (1, 0), (0, 1), (-1, -1)]
    a_blocks, a_mesh = build(1, coords)
    b_blocks, b_mesh = build(3, coords[::-1])
    for x, y in zip(a_blocks, b_blocks[::-1], strict=True):
        assert np.array_equal(x, y)
    for x, y in zip(a_mesh, b_mesh[::-1], strict=True):
        assert (x is None and y is None) or np.array_equal(x, y)


def test_seeds_give_different_worlds_and_the_generator_version_is_recorded():
    a = fields_hash("temperate", {"size": 1024.0}, {"hydrology": {"cell": 32.0}}, seed=1)
    b = fields_hash("temperate", {"size": 1024.0}, {"hydrology": {"cell": 32.0}}, seed=2)
    assert a != b
    w = Scene("S").worlds.create("W", "temperate", seed=1, size=1024.0)
    assert w.definition["generator"] == 1


def test_exact_angle_and_logarithm_helpers_match_the_math_library():
    """``atan2_deg`` and ``log10`` (orientation of settlements, wetness colours) use only
    correctly rounded operations, yet agree with the platform's functions."""
    import math

    from aether3d.worlds import noise as nz

    rng = np.random.default_rng(3)
    pairs = rng.normal(size=(5000, 2)) * 10.0 ** rng.uniform(-6.0, 6.0, (5000, 2))
    worst = max(abs(nz.atan2_deg(y, x) - math.degrees(math.atan2(y, x))) for y, x in pairs)
    assert worst < 1e-12
    assert nz.atan2_deg(0.0, 0.0) == 0.0
    assert [round(nz.atan2_deg(y, x), 9) for y, x in ((0, 1), (1, 0), (0, -1), (-1, 0))] == [
        0.0,
        90.0,
        180.0,
        -90.0,
    ]
    v = 10.0 ** rng.uniform(-10.0, 12.0, 200_000)
    assert np.max(np.abs(nz.log10(v) - np.log10(v))) < 1e-14
    assert nz.log10(np.array([1.0, 10.0, 1000.0])).tolist() == pytest.approx([0.0, 1.0, 3.0])
