"""Procedural texture graphs: hashing, node semantics, validation, and compiled programs."""

import math

import numpy as np
import pytest

from aether3d.errors import MaterialError
from aether3d.materials import noise as nz
from aether3d.materials.graph import (
    MAX_NODES,
    NODE_TYPES,
    EvalContext,
    TextureGraph,
    compile_graph,
    evaluate,
    node_types,
    run_program,
)
from aether3d.rendering.color import srgb_to_linear

M = 0xFFFFFFFF


def pcg3d_reference(x: int, y: int, z: int) -> tuple[int, int, int]:
    """pcg3d with Python integers (an independent check of the NumPy uint32 version)."""
    x, y, z = ((v * 1664525 + 1013904223) & M for v in (x, y, z))
    x = (x + y * z) & M
    y = (y + z * x) & M
    z = (z + x * y) & M
    x, y, z = x ^ (x >> 16), y ^ (y >> 16), z ^ (z >> 16)
    x = (x + y * z) & M
    y = (y + z * x) & M
    z = (z + x * y) & M
    return x, y, z


def ctx(points, sample_image=None) -> EvalContext:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return EvalContext(p, np.zeros_like(p), np.zeros_like(p), sample_image)


def run(graph, points, **kw) -> np.ndarray:
    return evaluate(TextureGraph.from_dict(graph), ctx(points, **kw))


def test_the_hash_is_pcg3d_with_wrapping_uint32_arithmetic():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**32, (200, 3), dtype=np.uint64).astype(np.uint32)
    hx, hy, hz = nz.pcg3d(values[:, 0], values[:, 1], values[:, 2])
    for row, a, b, c in zip(values.tolist(), hx.tolist(), hy.tolist(), hz.tolist(), strict=True):
        assert (a, b, c) == pcg3d_reference(*row)
    # Fixed vectors: changing the hash changes every procedural texture, on purpose only.
    assert pcg3d_reference(0, 0, 0) == tuple(
        int(v[0]) for v in nz.pcg3d(*(np.zeros(1, np.uint32),) * 3)
    )
    assert nz.unit(np.array([0xFFFFFFFF], np.uint32))[0] == (2**24 - 1) / 2**24


def test_cells_use_floor_and_clamped_coordinates():
    bits, frac = nz.cell_index(np.array([[-0.5, 1.25, -3.0], [2e9, -2e9, 0.0]]))
    assert bits[0].tolist() == [0xFFFFFFFF, 1, 0xFFFFFFFD]
    assert frac[0].tolist() == [0.5, 0.25, 0.0]
    assert bits[1].tolist() == [int(nz.COORD_LIMIT), (-int(nz.COORD_LIMIT)) & M, 0]


def test_noise_is_reproducible_bounded_smooth_and_seeded():
    rng = np.random.default_rng(1)
    p = rng.uniform(-50, 50, (4000, 3))
    for kind in ("perlin", "value"):
        a = nz.noise(p, 7, kind)
        assert np.array_equal(a, nz.noise(p.copy(), 7, kind))
        assert a.min() >= 0.0 and a.max() <= 1.0 and a.std() > 0.05
        assert not np.allclose(a, nz.noise(p, 8, kind))
        step = nz.noise(p + 1e-4, 7, kind)
        assert np.abs(step - a).max() < 0.01  # continuous
    # Value noise equals the hash at lattice points; gradient noise is 0.5 there.
    lattice = np.array([[3.0, -2.0, 5.0]])
    bits, _ = nz.cell_index(lattice)
    h, _, _ = nz.pcg3d(bits[:, 0], bits[:, 1], bits[:, 2] ^ nz.seed_bits(4))
    assert nz.value_noise(lattice, 4)[0] == nz.unit(h)[0]
    assert nz.gradient_noise(lattice, 4)[0] == 0.5


def test_fbm_is_normalized_and_voronoi_is_regular_without_randomness():
    p = np.random.default_rng(2).uniform(-10, 10, (3000, 3))
    f = nz.fbm(p, 3, "perlin", 6, 2.0, 0.5)
    assert f.min() >= 0.0 and f.max() <= 1.0
    v = nz.voronoi(np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [2.5, 7.5, 9.0]]), 0, 0.0, 2)
    assert v[0, 0] == 0.0 and v[2, 0] == 0.0  # cell centers are the feature points
    assert v[1, 0] == pytest.approx(math.sqrt(0.5))
    assert v[1, 1] == pytest.approx(0.0)  # equidistant from four centers: on an edge


def test_checker_stripes_and_gradients():
    pts = [[0.25, 0.25, 0], [1.25, 0.25, 0], [-0.5, 0.5, 0], [1.5, 1.5, 0]]
    out = run({"type": "checker", "color1": 0.0, "color2": 1.0}, pts)
    assert out[:, 0].tolist() == [0.0, 1.0, 1.0, 0.0] and out[:, 3].tolist() == [0.0, 1.0, 1.0, 0.0]
    stripes = run(
        {"type": "stripes", "width": 0.25, "color1": 0.0, "color2": 1.0},
        [[0.1, 0, 0], [0.3, 0, 0], [1.2, 0, 0]],
    )
    assert stripes[:, 0].tolist() == [0.0, 1.0, 0.0]
    for kind, point, expected in [
        ("linear", [0.3, 0, 0], 0.3),
        ("quadratic", [0.5, 0, 0], 0.25),
        ("easing", [0.5, 0, 0], 0.5),
        ("diagonal", [0.2, 0.4, 0], 0.3),
        ("radial", [-1.0, 0.0, 0], 1.0),
        ("spherical", [0.0, 0.6, 0.0], 0.4),
        ("linear", [2.0, 0, 0], 1.0),
    ]:
        assert run({"type": "gradient", "kind": kind}, [point])[0, 0] == pytest.approx(expected)


def test_pattern_scale_and_offset_map_the_coordinate():
    out = run({"type": "coordinates", "scale": [2, 3, 4], "offset": [1, 0, -1]}, [[1.0, 1.0, 1.0]])
    assert out[0, :3].tolist() == [3.0, 3.0, 3.0]


def test_bricks_mortar_and_row_offset():
    graph = {
        "type": "bricks",
        "brick_width": 1.0,
        "brick_height": 0.5,
        "mortar": 0.1,
        "smoothness": 0.0,
        "variation": 0.0,
        "color1": 0.2,
        "mortar_color": 0.9,
    }
    pts = [[0.5, 0.25, 0], [0.01, 0.25, 0], [0.5, 0.01, 0], [0.99, 0.75, 0], [0.5, 0.75, 0]]
    out = run(graph, pts)
    assert out[:, 3].tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
    ]  # the second row is offset by half a brick
    assert out[0, 0] == pytest.approx(0.2) and out[1, 0] == pytest.approx(0.9)
    tiles = run(
        {"type": "tiles", "tile_width": 1.0, "tile_height": 1.0, "mortar": 0.1, "smoothness": 0.0},
        [[0.5, 1.5, 0], [0.01, 1.5, 0]],
    )
    assert tiles[:, 3].tolist() == [0.0, 1.0]


def test_colors_are_display_colors_and_numbers_are_linear():
    out = run({"type": "constant", "value": "#808080"}, [[0, 0, 0]])
    assert out[0, 0] == pytest.approx(srgb_to_linear(128 / 255))
    assert run({"type": "constant", "value": 0.5}, [[0, 0, 0]])[0, :3].tolist() == [0.5] * 3


def test_mix_modes_ramp_and_math():
    def mix(mode, a=0.25, b=0.5, f=1.0):
        return run({"type": "mix", "a": a, "b": b, "factor": f, "mode": mode}, [[0, 0, 0]])[0, 0]

    assert mix("mix", f=0.5) == pytest.approx(0.375)
    assert mix("multiply") == pytest.approx(0.125)
    assert mix("add") == pytest.approx(0.75)
    assert mix("subtract") == pytest.approx(-0.25)
    assert mix("screen") == pytest.approx(1 - 0.75 * 0.5)
    assert mix("difference") == pytest.approx(0.25)
    assert mix("minimum") == pytest.approx(0.25) and mix("maximum") == pytest.approx(0.5)
    assert mix("overlay") == pytest.approx(2 * 0.25 * 0.5)
    ramp = {"type": "ramp", "stops": [[0.2, 0.0], [0.6, 1.0], [1.0, 0.5]]}
    factors = {"type": "gradient"}
    graph = {"nodes": {"g": factors, "r": {**ramp, "factor": {"node": "g"}}}, "output": "r"}
    x = [[0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0], [0.8, 0, 0], [1.0, 0, 0]]
    assert run(graph, x)[:, 0] == pytest.approx([0.0, 0.0, 0.5, 1.0, 0.75, 0.5])
    graph["nodes"]["r"]["interpolation"] = "constant"
    # (Not exactly at a stop: positions are float32, like on the GPU, and 0.6 rounds up.)
    x[3][0] = 0.61
    assert run(graph, x)[:, 0] == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    graph["nodes"]["r"]["interpolation"] = "smooth"
    assert run(graph, [[0.3, 0, 0]])[0, 0] == pytest.approx(0.15625)

    def math_op(op, a, b=1.0, c=1.0):
        return run({"type": "math", "op": op, "a": a, "b": b, "c": c}, [[0, 0, 0]])[0, 0]

    assert math_op("divide", 3.0, 0.0) == 0.0 and math_op("divide", 3.0, 2.0) == 1.5
    assert math_op("power", 0.0, 0.0) == 1.0 and math_op("power", 0.0, -1.0) == 0.0
    assert math_op("power", -2.0, 2.0) == 0.0 and math_op("power", 2.0, 3.0) == 8.0
    assert math_op("clamp", 5.0, 0.0, 2.0) == 2.0 and math_op("invert", 0.25) == 0.75
    assert (
        math_op("smoothstep", 0.5, 0.0, 1.0) == 0.5 and math_op("smoothstep", 0.5, 1.0, 1.0) == 0.0
    )
    assert math_op("remap", 3.0, 2.0, 6.0) == 0.25 and math_op("remap", 3.0, 2.0, 2.0) == 0.0
    assert math_op("absolute", -3.0) == 3.0 and math_op("subtract", 1.0, 3.0) == -2.0


def test_references_outputs_and_conversions():
    graph = {
        "nodes": {
            "b": {
                "type": "bricks",
                "color1": "#ff0000",
                "color2": "#ff0000",
                "mortar_color": "#00ff00",
            },
            "lum": {"type": "math", "op": "add", "a": {"node": "b"}, "b": 0.0},
            "mask": {"type": "math", "op": "add", "a": {"node": "b", "output": "mortar"}, "b": 0.0},
            "gray": {"type": "mix", "a": {"node": "mask"}, "b": 0.0, "factor": 0.0},
        },
        "output": "gray",
    }
    g = TextureGraph.from_dict(graph)
    p = [[0.25, 0.125, 0.0]]  # inside a brick
    assert evaluate(g, ctx(p))[0, :3].tolist() == [0.0, 0.0, 0.0]
    graph["output"] = "lum"
    assert run(graph, p)[0, 0] == pytest.approx(0.2126)  # pure red: Rec. 709 luminance
    graph["output"] = {"node": "b", "output": "a"}
    assert run(graph, [[0.0, 0.0, 0.0]])[0, :3].tolist() == [1.0, 1.0, 1.0]


def test_bump_turns_a_height_slope_into_a_tilted_normal():
    graph = {
        "nodes": {
            "h": {"type": "gradient"},
            "n": {"type": "bump", "height": {"node": "h"}, "depth": 0.5},
        },
        "output": "n",
    }
    out = run(graph, [[0.3, 0.5, 0.0]])[0, :3] * 2.0 - 1.0
    expected = np.array([-0.5, 0.0, 1.0]) / math.sqrt(1.25)  # slope 0.5 along x
    assert out == pytest.approx(expected, abs=1e-6)
    flat = run({"type": "bump", "height": 0.7}, [[0, 0, 0]])[0, :3]
    assert flat.tolist() == [0.5, 0.5, 1.0]


def test_image_nodes_call_the_sampler_with_scaled_coordinates():
    calls = []

    def sampler(ref, space, p, dx, dy):
        calls.append((ref, space, p.copy(), dx.copy()))
        return np.tile([0.1, 0.2, 0.3, 0.4], (p.shape[0], 1))

    out = run(
        {"type": "image", "texture": "Bricks", "scale": 2.0, "color_space": "linear"},
        [[0.5, 0.25, 0.0]],
        sample_image=sampler,
    )
    assert out[0].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert calls[0][0] == "Bricks" and calls[0][1] == "linear"
    assert calls[0][2][0].tolist() == [1.0, 0.5, 0.0]


@pytest.mark.parametrize(
    "graph,code,where",
    [
        ({"type": "nope"}, "UNKNOWN_NODE_TYPE", "graph.nodes.out.type"),
        ({"type": "noise", "octaves": 3}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out"),
        ({"type": "fbm", "octaves": 40}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out.octaves"),
        ({"type": "fbm", "octaves": 2.5}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out.octaves"),
        ({"type": "noise", "scale": [1, 2]}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out.scale"),
        (
            {"type": "noise", "scale": float("nan")},
            "INVALID_TEXTURE_GRAPH",
            "graph.nodes.out.scale",
        ),
        ({"type": "checker", "color1": "#zzz"}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out.color1"),
        ({"type": "gradient", "kind": "conic"}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out.kind"),
        (
            {"type": "ramp", "stops": [[0, "#000"]]},
            "INVALID_TEXTURE_GRAPH",
            "graph.nodes.out.stops",
        ),
        (
            {"type": "ramp", "stops": [[0.5, 0.0], [0.2, 1.0]]},
            "INVALID_TEXTURE_GRAPH",
            "graph.nodes.out.stops[1]",
        ),
        (
            {"type": "ramp", "stops": [[0, 0.0], [1, "#fff"]]},
            "INVALID_TEXTURE_GRAPH",
            "graph.nodes.out.stops",
        ),
        ({"type": "image"}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out"),
        ({"type": "mix", "a": {"node": "missing"}}, "INVALID_TEXTURE_GRAPH", "graph.nodes.out.a"),
        ({"type": "mix", "a": {"node": "out"}}, "GRAPH_CYCLE", "graph.nodes.out"),
        (
            {"nodes": {"a": {"type": "noise"}}, "output": "b"},
            "INVALID_TEXTURE_GRAPH",
            "graph.output",
        ),
        (
            {"nodes": {"bad id!": {"type": "noise"}}, "output": "bad id!"},
            "INVALID_TEXTURE_GRAPH",
            "graph.nodes.bad id!",
        ),
        (
            {"nodes": {"a": {"type": "noise"}}, "output": {"node": "a", "output": "mortar"}},
            "INVALID_TEXTURE_GRAPH",
            "graph.output",
        ),
        ({"nodes": {}, "output": "a"}, "INVALID_TEXTURE_GRAPH", "graph.nodes"),
        ([1, 2], "INVALID_TEXTURE_GRAPH", "graph"),
        (
            {"nodes": {"a": {"type": "noise"}}, "output": "a", "extra": 1},
            "INVALID_TEXTURE_GRAPH",
            "graph",
        ),
    ],
)
def test_invalid_graphs_are_rejected_with_a_location(graph, code, where):
    with pytest.raises(MaterialError) as info:
        TextureGraph.from_dict(graph)
    assert info.value.code == code and info.value.location == where


def test_cycles_are_reported_with_their_path():
    graph = {
        "nodes": {
            "a": {"type": "mix", "a": {"node": "b"}},
            "b": {"type": "mix", "a": {"node": "c"}},
            "c": {"type": "mix", "a": {"node": "a"}},
        },
        "output": "a",
    }
    with pytest.raises(MaterialError, match="a -> b -> c -> a"):
        TextureGraph.from_dict(graph)


def test_size_and_depth_limits():
    nodes = {f"n{i}": {"type": "noise", "seed": i} for i in range(MAX_NODES + 1)}
    with pytest.raises(MaterialError) as info:
        TextureGraph.from_dict({"nodes": nodes, "output": "n0"})
    assert info.value.code == "GRAPH_TOO_COMPLEX"
    deep = {"nodes": {"h": {"type": "noise"}}, "output": "b3"}
    previous = "h"
    for k in range(1, 4):
        deep["nodes"][f"b{k}"] = {"type": "bump", "height": {"node": previous, "output": "r"}}
        previous = f"b{k}"
    with pytest.raises(MaterialError) as info:
        TextureGraph.from_dict(deep)
    assert info.value.code == "GRAPH_TOO_COMPLEX"
    # Too many values alive at once for the GPU's registers.
    wide = {f"v{i}": {"type": "noise", "seed": i} for i in range(20)}
    chain = "v0"
    for i in range(1, 20):
        wide[f"s{i}"] = {"type": "math", "op": "add", "a": {"node": chain}, "b": {"node": f"v{i}"}}
        chain = f"s{i}"
    TextureGraph.from_dict({"nodes": wide, "output": chain})  # sums as it goes: fine
    many = {f"v{i}": {"type": "noise", "seed": i} for i in range(16)}
    many["m"] = {"type": "mix", "a": {"node": "v0"}, "b": {"node": "v1"}, "factor": {"node": "v2"}}
    graph = TextureGraph.from_dict({"nodes": many, "output": "m"})
    compile_graph(graph)


def random_graph(rng: np.random.Generator) -> dict:
    """A random valid graph using every node type (for compiler checks)."""
    kinds = [k for k in NODE_TYPES if k != "image"]
    nodes: dict[str, dict] = {}
    for i in range(int(rng.integers(3, 12))):
        kind = str(rng.choice(kinds))
        spec = NODE_TYPES[kind]
        node: dict = {"type": kind}
        for p in spec.params:
            if p.connectable and nodes and rng.random() < 0.6:
                ref = str(rng.choice(list(nodes)))
                outputs = [o for o, _ in NODE_TYPES[nodes[ref]["type"]].outputs] + ["r", "a"]
                node[p.name] = {"node": ref, "output": str(rng.choice(outputs))}
            elif p.kind == "int" and p.name == "seed":
                node[p.name] = int(rng.integers(0, 1000))
        if spec.pattern:
            node["scale"] = float(rng.uniform(0.5, 6.0))
        if kind == "bump":
            if not nodes:
                continue
            node["height"] = {"node": str(rng.choice(list(nodes))), "output": "r"}
        nodes[f"n{i}"] = node
    try:
        g = TextureGraph.from_dict({"nodes": nodes, "output": list(nodes)[-1]})
    except MaterialError:
        return {"type": "noise"}
    return g.to_dict()


@pytest.mark.parametrize("seed", range(25))
def test_compiled_programs_compute_what_the_evaluator_computes(seed):
    rng = np.random.default_rng(seed)
    graph = TextureGraph.from_dict(random_graph(rng))
    points = rng.uniform(-4, 4, (2000, 3))
    c = ctx(points)
    reference = evaluate(graph, c)
    program = compile_graph(graph)
    assert program.registers <= 16
    assert np.allclose(run_program(program, c), reference, atol=2e-5)


def test_every_node_type_compiles_and_matches():
    p = np.random.default_rng(9).uniform(-3, 3, (1000, 3))
    c = ctx(
        p,
        sample_image=lambda ref, space, q, dx, dy: np.stack(
            [q[:, 0], q[:, 1], q[:, 2], np.ones(q.shape[0])], 1
        ),
    )
    for name in NODE_TYPES:
        node = {"type": name}
        if name == "image":
            node["texture"] = "T"
        if name == "bump":
            node["height"] = 0.25
        g = TextureGraph.from_dict(node)
        assert np.allclose(run_program(compile_graph(g), c), evaluate(g, c), atol=2e-5), name


def test_graphs_round_trip_and_schemas_cover_every_type():
    g = TextureGraph.from_dict(random_graph(np.random.default_rng(4)))
    again = TextureGraph.from_dict(g.to_dict())
    assert again == g and again.order == g.order
    schemas = {s["type"]: s for s in node_types()}
    assert set(schemas) == set(NODE_TYPES)
    assert {p["name"] for p in schemas["bricks"]["params"]} >= {"scale", "offset", "mortar", "seed"}
    assert schemas["voronoi"]["outputs"][1] == {"name": "edge", "channels": "g"}
