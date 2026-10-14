"""Procedural texture graphs: node types, validation, CPU evaluation and GPU programs.

A procedural texture is a small directed acyclic graph of *nodes*. Each node
has a ``type`` (see :data:`NODE_TYPES`) and parameters; *inputs* marked
connectable take either a literal value or the output of another node::

    {
      "output": "color",
      "nodes": {
        "grain": {"type": "fbm", "scale": 6, "octaves": 4, "seed": 3},
        "color": {"type": "ramp", "factor": {"node": "grain"},
                  "stops": [[0, "#6d6d6a"], [1, "#a5a39b"]]}
      }
    }

A single node can be written on its own: ``{"type": "checker", "scale": 8}``.

Values
    Every node produces four numbers (``r, g, b, a``). A node's *outputs* name
    what they mean -- ``color`` (``rgb``), ``value`` (``r``), ``mask``
    (``a``), ... -- and every node also has the generic outputs ``r``, ``g``,
    ``b`` and ``a``. A reference is ``{"node": id}`` (the node's first output)
    or ``{"node": id, "output": name}``. A color used where a number is
    expected gives its luminance; a number used as a color gives gray.

Colors and numbers
    Colors written as ``"#rrggbb"`` or ``[r, g, b]`` are display (sRGB)
    colors, like everywhere in Aether3D, and are converted to linear light.
    A plain number given for a color input is a linear gray value (so
    ``0.5`` means half reflectance, not the display gray ``#808080``).

Coordinates
    Nodes are evaluated at a texture coordinate ``p = (x, y, z)`` chosen by
    the material binding (UV, object or world space; see
    :mod:`aether3d.materials.material`). Pattern nodes first map it with
    their own ``scale`` (a number or ``[x, y, z]``) and ``offset``:
    ``p' = p * scale + offset``.

Reproducibility
    All randomness comes from integer hashes of lattice cells and the
    ``seed`` parameters (:mod:`aether3d.materials.noise`), so a graph gives
    the same result on every run, machine and backend.

Evaluation
    :func:`evaluate` runs a graph on the CPU with NumPy. :func:`compile_graph`
    turns it into a :class:`Program` -- a list of fixed instructions with
    numeric operands -- that the GPU shader's interpreter executes. Graphs are
    data: they are never turned into shader or Python code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import MaterialError
from ..rendering.color import parse_color, srgb_to_linear
from . import noise as nz

MAX_NODES = 64
MAX_INSTRUCTIONS = 512
MAX_REGISTERS = 16
MAX_RAMP_STOPS = 8
MAX_BUMP_DEPTH = 2
NODE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
"""Rec. 709 luminance weights (linear RGB), used when a color is read as a number."""

GENERIC_OUTPUTS = {"r": 0, "g": 1, "b": 2, "a": 3}


# --------------------------------------------------------------------------- #
# Node type specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Param:
    """One node parameter.

    ``kind``: ``"float"``, ``"int"``, ``"vec3"`` (a number or three numbers),
    ``"enum"``, ``"color"`` (connectable color input), ``"value"`` (connectable
    number input), ``"stops"`` (ramp stops) or ``"texture"`` (an image
    texture name or id).
    """

    name: str
    kind: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    description: str = ""

    @property
    def connectable(self) -> bool:
        return self.kind in ("color", "value")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "kind": self.kind, "default": self.default}
        if self.lo is not None:
            data["min"] = self.lo
        if self.hi is not None:
            data["max"] = self.hi
        if self.choices:
            data["choices"] = list(self.choices)
        if self.connectable:
            data["connectable"] = True
        data["description"] = self.description
        return data


@dataclass(frozen=True)
class NodeType:
    name: str
    opcode: int
    params: tuple[Param, ...]
    outputs: tuple[tuple[str, str], ...]  # (name, "color" | "r" | "g" | "b" | "a")
    description: str
    pattern: bool = True  # has scale/offset and reads the coordinate

    def param(self, name: str) -> Param | None:
        return next((p for p in self.all_params if p.name == name), None)

    @property
    def all_params(self) -> tuple[Param, ...]:
        if not self.pattern:
            return self.params
        return (
            Param("scale", "vec3", 1.0, -1e6, 1e6, description="coordinate scale (p * scale)"),
            Param("offset", "vec3", 0.0, -1e6, 1e6, description="added after scaling"),
            *self.params,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "description": self.description,
            "params": [p.to_dict() for p in self.all_params],
            "outputs": [{"name": n, "channels": c} for n, c in self.outputs],
        }


def _seed() -> Param:
    return Param("seed", "int", 0, 0, 2**31 - 1, description="random seed (any integer)")


_NOISE_KIND = Param(
    "kind", "enum", "perlin", choices=("perlin", "value"), description="gradient or value noise"
)

NODE_TYPES: dict[str, NodeType] = {
    t.name: t
    for t in (
        NodeType(
            "constant",
            1,
            (Param("value", "color", 0.8, description="a color, or a number (linear gray)"),),
            (("color", "color"), ("value", "r")),
            "a constant color or number",
            pattern=False,
        ),
        NodeType(
            "coordinates",
            2,
            (),
            (("vector", "color"), ("x", "r"), ("y", "g"), ("z", "b")),
            "the texture coordinate itself, after scale and offset",
        ),
        NodeType(
            "checker",
            3,
            (
                Param("color1", "color", "#202020"),
                Param("color2", "color", "#e0e0e0"),
                Param("dimensions", "int", 2, 2, 3, description="2: squares, 3: cubes"),
            ),
            (("color", "color"), ("mask", "a")),
            "alternating squares of color1 and color2, one per unit (mask: 0 or 1)",
        ),
        NodeType(
            "gradient",
            4,
            (
                Param(
                    "kind",
                    "enum",
                    "linear",
                    choices=("linear", "quadratic", "easing", "diagonal", "radial", "spherical"),
                ),
            ),
            (("value", "r"),),
            "a 0..1 ramp: linear/quadratic/easing along x, diagonal (x + y) / 2, radial "
            "(angle around z) or spherical (1 - distance from the origin)",
        ),
        NodeType(
            "stripes",
            5,
            (
                Param("color1", "color", "#202020"),
                Param("color2", "color", "#e0e0e0"),
                Param("width", "float", 0.5, 0.0, 1.0, description="fraction that is color1"),
                Param("axis", "enum", "x", choices=("x", "y", "z", "diagonal")),
            ),
            (("color", "color"), ("mask", "a")),
            "parallel stripes, one period per unit along the axis",
        ),
        NodeType(
            "noise",
            6,
            (_NOISE_KIND, _seed()),
            (("value", "r"),),
            "smooth 3D noise in 0..1, features about one unit apart",
        ),
        NodeType(
            "fbm",
            7,
            (
                _NOISE_KIND,
                _seed(),
                Param("octaves", "int", 5, 1, 12),
                Param("lacunarity", "float", 2.0, 1.0, 8.0, description="frequency step"),
                Param("gain", "float", 0.5, 0.0, 1.0, description="amplitude step"),
            ),
            (("value", "r"),),
            "fractal Brownian motion: octaves of noise summed and normalized to 0..1",
        ),
        NodeType(
            "voronoi",
            8,
            (
                _seed(),
                Param("randomness", "float", 1.0, 0.0, 1.0),
                Param("dimensions", "int", 2, 2, 3),
            ),
            (("distance", "r"), ("edge", "g"), ("cell", "b")),
            "cellular (Worley) noise: distance to the nearest feature point, F2 - F1 "
            "(0 on cell borders) and a random value per cell",
        ),
        NodeType(
            "bricks",
            9,
            (
                Param("color1", "color", "#8e4a35"),
                Param("color2", "color", "#a8624a"),
                Param("mortar_color", "color", "#bdb6a8"),
                Param("brick_width", "float", 0.5, 1e-6, 1e6),
                Param("brick_height", "float", 0.25, 1e-6, 1e6),
                Param("mortar", "float", 0.02, 0.0, 1e6, description="mortar thickness"),
                Param("smoothness", "float", 0.2, 0.0, 10.0, description="edge rounding"),
                Param("row_offset", "float", 0.5, 0.0, 1.0, description="shift per row"),
                Param("variation", "float", 0.5, 0.0, 1.0, description="color1 -> color2"),
                _seed(),
            ),
            (("color", "color"), ("mortar", "a")),
            "running-bond bricks in the xy plane (mortar: 1 in the joints, 0 on bricks)",
        ),
        NodeType(
            "tiles",
            10,
            (
                Param("color1", "color", "#d8d4cc"),
                Param("color2", "color", "#c4beb2"),
                Param("mortar_color", "color", "#6f6a62"),
                Param("tile_width", "float", 0.25, 1e-6, 1e6),
                Param("tile_height", "float", 0.25, 1e-6, 1e6),
                Param("mortar", "float", 0.01, 0.0, 1e6, description="grout thickness"),
                Param("smoothness", "float", 0.2, 0.0, 10.0),
                Param("variation", "float", 0.5, 0.0, 1.0),
                _seed(),
            ),
            (("color", "color"), ("mortar", "a")),
            "a grid of tiles in the xy plane (mortar: 1 in the grout, 0 on tiles)",
        ),
        NodeType(
            "wood",
            11,
            (
                Param("color1", "color", "#c8955c"),
                Param("color2", "color", "#8a5a30"),
                Param("rings", "float", 10.0, 0.0, 1e4, description="rings per unit radius"),
                Param("distortion", "float", 0.4, 0.0, 10.0),
                Param("grain", "float", 0.25, 0.0, 1.0),
                Param("axis", "enum", "z", choices=("x", "y", "z")),
                _seed(),
            ),
            (("color", "color"), ("rings", "a")),
            "concentric growth rings around an axis, distorted by noise, with fine grain",
        ),
        NodeType(
            "mix",
            12,
            (
                Param("a", "color", 0.0),
                Param("b", "color", 1.0),
                Param("factor", "value", 0.5),
                Param(
                    "mode",
                    "enum",
                    "mix",
                    choices=(
                        "mix",
                        "multiply",
                        "add",
                        "subtract",
                        "screen",
                        "difference",
                        "minimum",
                        "maximum",
                        "overlay",
                    ),
                ),
            ),
            (("color", "color"), ("value", "r")),
            "blends a with b by factor (in linear light)",
            pattern=False,
        ),
        NodeType(
            "ramp",
            13,
            (
                Param("factor", "value", 0.5),
                Param(
                    "stops",
                    "stops",
                    [[0.0, "#000000"], [1.0, "#ffffff"]],
                    description=f"2-{MAX_RAMP_STOPS} [position, color or number] pairs",
                ),
                Param("interpolation", "enum", "linear", choices=("linear", "constant", "smooth")),
            ),
            (("color", "color"), ("value", "r")),
            "maps a 0..1 factor to colors (or numbers) between stops",
            pattern=False,
        ),
        NodeType(
            "math",
            14,
            (
                Param(
                    "op",
                    "enum",
                    "multiply",
                    choices=(
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                        "power",
                        "minimum",
                        "maximum",
                        "absolute",
                        "invert",
                        "clamp",
                        "smoothstep",
                        "remap",
                    ),
                ),
                Param("a", "value", 0.0),
                Param("b", "value", 1.0),
                Param("c", "value", 1.0),
            ),
            (("value", "r"),),
            "arithmetic on numbers: clamp a to [b, c], smoothstep(b, c, a), remap a from "
            "[b, c] to [0, 1], invert = 1 - a",
            pattern=False,
        ),
        NodeType(
            "bump",
            15,
            (
                Param("height", "value", 0.0, description="height field (usually a reference)"),
                Param("depth", "float", 0.02, -1e3, 1e3, description="height of 1, in coords"),
                Param("step", "float", 1e-3, 1e-7, 1.0, description="finite-difference step"),
            ),
            (("normal", "color"),),
            "a tangent-space normal map from a height field (by finite differences in "
            "texture coordinates); use it in a material's normal slot",
            pattern=False,
        ),
        NodeType(
            "image",
            16,
            (
                Param("texture", "texture", None, description="an image texture (name or id)"),
                Param("color_space", "enum", "srgb", choices=("srgb", "linear")),
            ),
            (("color", "color"), ("alpha", "a")),
            "samples an image texture of the scene (using its wrap and filter settings)",
        ),
    )
}
OPCODES = {t.opcode: t.name for t in NODE_TYPES.values()}
OP_SET_COORD = 32  # coordinate = base + offset (bump sampling)
OP_OUTPUT = 33  # final output selection


def node_types() -> list[dict[str, Any]]:
    """JSON-compatible schemas of every node type."""
    return [t.to_dict() for t in NODE_TYPES.values()]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ref:
    """A reference to another node's output."""

    node: str
    output: str | None = None

    def to_json(self) -> dict[str, str]:
        return (
            {"node": self.node}
            if self.output is None
            else {"node": self.node, "output": self.output}
        )


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    params: dict[str, Any]  # normalized values (literals or Ref)

    @property
    def spec(self) -> NodeType:
        return NODE_TYPES[self.type]


def _err(message: str, location: str, code: str = "INVALID_TEXTURE_GRAPH", **kw) -> MaterialError:
    return MaterialError(code, message, location, **kw)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _number(value: Any, p: Param, loc: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise _err(f"{p.name} must be a finite number, got {value!r}", loc)
    v = float(value)
    if (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi):
        raise _err(f"{p.name} must be in [{p.lo:g}, {p.hi:g}], got {v:g}", loc)
    return v


def _color_literal(value: Any, loc: str) -> Any:
    """A color input literal: a number (linear gray) or a display color (kept as given)."""
    if _is_number(value):
        v = float(value)
        if not math.isfinite(v) or not -1e6 <= v <= 1e6:
            raise _err(f"expected a finite number or a color, got {value!r}", loc)
        return v
    try:
        parse_color(value, loc)
    except Exception:
        raise _err(
            f"expected a color ('#rrggbb' or [r, g, b] in 0..1) or a number, got {value!r}", loc
        ) from None
    return list(value) if not isinstance(value, str) else value


def _ref(value: dict, loc: str) -> Ref:
    unknown = sorted(set(value) - {"node", "output"})
    if unknown or not isinstance(value.get("node"), str):
        raise _err('a connection must be {"node": id} or {"node": id, "output": name}', loc)
    output = value.get("output")
    if output is not None and not isinstance(output, str):
        raise _err("output must be a string", loc)
    return Ref(value["node"], output)


def _normalize_param(p: Param, value: Any, loc: str) -> Any:
    if p.connectable:
        if isinstance(value, dict):
            return _ref(value, loc)
        if p.kind == "value":
            return _number(value, Param(p.name, "float", 0, -1e6, 1e6), loc)
        return _color_literal(value, loc)
    if p.kind == "float":
        return _number(value, p, loc)
    if p.kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise _err(f"{p.name} must be an integer, got {value!r}", loc)
        v = int(value)
        if (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi):
            raise _err(f"{p.name} must be in [{p.lo:g}, {p.hi:g}], got {v}", loc)
        return v
    if p.kind == "vec3":
        if _is_number(value):
            return _number(value, p, loc)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [_number(v, p, loc) for v in value]
        raise _err(f"{p.name} must be a number or three numbers, got {value!r}", loc)
    if p.kind == "enum":
        if value not in p.choices:
            raise _err(f"{p.name} must be one of {', '.join(p.choices)}, got {value!r}", loc)
        return value
    if p.kind == "texture":
        if not isinstance(value, str) or not value:
            raise _err(f"{p.name} must name an image texture, got {value!r}", loc)
        return value
    if p.kind == "stops":
        return _stops(value, loc)
    raise AssertionError(p.kind)  # pragma: no cover


def _stops(value: Any, loc: str) -> list[list[Any]]:
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= MAX_RAMP_STOPS:
        raise _err(f"stops must be a list of 2-{MAX_RAMP_STOPS} [position, value] pairs", loc)
    out, kinds, last = [], set(), -math.inf
    for i, stop in enumerate(value):
        sloc = f"{loc}[{i}]"
        if not isinstance(stop, (list, tuple)) or len(stop) != 2:
            raise _err("each stop must be [position, color or number]", sloc)
        position = _number(stop[0], Param("position", "float", 0, 0.0, 1.0), sloc)
        if position < last:
            raise _err("stop positions must be in increasing order", sloc)
        last = position
        literal = _color_literal(stop[1], sloc)
        kinds.add("number" if isinstance(literal, float) else "color")
        out.append([position, literal])
    if len(kinds) > 1:
        raise _err("stops must be all colors or all numbers", loc)
    return out


@dataclass(frozen=True)
class TextureGraph:
    """A validated procedural texture graph (immutable; see the module docstring)."""

    nodes: dict[str, Node]
    output: Ref
    order: tuple[str, ...] = field(default=())  # topological order (inputs first)

    @classmethod
    def from_dict(cls, data: Any, location: str = "graph") -> TextureGraph:
        if isinstance(data, TextureGraph):
            return data
        if not isinstance(data, dict):
            raise _err("a texture graph must be an object", location)
        if "nodes" not in data:  # a single node
            if "type" not in data:
                raise _err('expected {"nodes": {...}, "output": ...} or a single node', location)
            data = {"nodes": {"out": dict(data)}, "output": "out"}
        unknown = sorted(set(data) - {"nodes", "output"})
        if unknown:
            raise _err(f"unknown key(s): {', '.join(unknown)}", location)
        raw_nodes = data["nodes"]
        if not isinstance(raw_nodes, dict) or not raw_nodes:
            raise _err("nodes must be a non-empty object of {id: node}", f"{location}.nodes")
        if len(raw_nodes) > MAX_NODES:
            raise _err(
                f"a texture graph has at most {MAX_NODES} nodes, got {len(raw_nodes)}",
                f"{location}.nodes",
                "GRAPH_TOO_COMPLEX",
            )
        nodes: dict[str, Node] = {}
        for node_id, raw in raw_nodes.items():
            nloc = f"{location}.nodes.{node_id}"
            if not isinstance(node_id, str) or not NODE_ID.match(node_id):
                raise _err(
                    "node ids are 1-64 letters, digits, '_', '.' or '-', starting with a letter",
                    nloc,
                )
            nodes[node_id] = _node(node_id, raw, nloc)
        output = data.get("output")
        if isinstance(output, str):
            output = Ref(output)
        elif isinstance(output, dict):
            output = _ref(output, f"{location}.output")
        else:
            raise _err("output must name a node (or be {node, output})", f"{location}.output")
        graph = cls(nodes, output)
        order = graph._check(location)
        object.__setattr__(graph, "order", order)
        return graph

    def _check(self, location: str) -> tuple[str, ...]:
        for node in self.nodes.values():
            for name, value in node.params.items():
                if isinstance(value, Ref):
                    self._check_ref(value, f"{location}.nodes.{node.id}.{name}")
        self._check_ref(self.output, f"{location}.output")
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(node_id: str, path: list[str]) -> None:
            mark = state.get(node_id)
            if mark == 2:
                return
            if mark == 1:
                cycle = path[path.index(node_id) :] + [node_id]
                raise _err(
                    "the graph has a cycle: " + " -> ".join(cycle),
                    f"{location}.nodes.{node_id}",
                    "GRAPH_CYCLE",
                )
            state[node_id] = 1
            for value in self.nodes[node_id].params.values():
                if isinstance(value, Ref):
                    visit(value.node, [*path, node_id])
            state[node_id] = 2
            order.append(node_id)

        for node_id in self.nodes:
            visit(node_id, [])
        depth = self._bump_depth()
        if depth > MAX_BUMP_DEPTH:
            raise _err(
                f"bump nodes are nested {depth} deep (at most {MAX_BUMP_DEPTH})",
                location,
                "GRAPH_TOO_COMPLEX",
            )
        return tuple(order)

    def _check_ref(self, ref: Ref, loc: str) -> None:
        if ref.node not in self.nodes:
            raise _err(f"no node {ref.node!r} in this graph", loc)
        if ref.output is not None:
            target = self.nodes[ref.node].spec
            if ref.output not in dict(target.outputs) and ref.output not in GENERIC_OUTPUTS:
                names = [n for n, _ in target.outputs] + list(GENERIC_OUTPUTS)
                raise _err(
                    f"node {ref.node!r} ({target.name}) has no output {ref.output!r} "
                    f"(outputs: {', '.join(names)})",
                    loc,
                )

    def _bump_depth(self) -> int:
        memo: dict[str, int] = {}

        def depth(node_id: str) -> int:
            if node_id not in memo:
                node = self.nodes[node_id]
                inner = max(
                    (depth(v.node) for v in node.params.values() if isinstance(v, Ref)), default=0
                )
                memo[node_id] = inner + (1 if node.type == "bump" else 0)
            return memo[node_id]

        return max((depth(n) for n in self.nodes), default=0)

    # -- inspection ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        nodes = {}
        for node_id, node in self.nodes.items():
            entry: dict[str, Any] = {"type": node.type}
            for name, value in node.params.items():
                entry[name] = value.to_json() if isinstance(value, Ref) else value
            nodes[node_id] = entry
        output: Any = self.output.node if self.output.output is None else self.output.to_json()
        return {"nodes": nodes, "output": output}

    def image_references(self) -> list[str]:
        """Names or ids of the image textures used by ``image`` nodes."""
        return [n.params["texture"] for n in self.nodes.values() if n.type == "image"]

    def output_kind(self) -> str:
        """``"color"`` or ``"value"``: what the graph's output is."""
        return _output_kind(self.nodes[self.output.node], self.output.output)

    def uses_coordinates(self) -> bool:
        return any(n.spec.pattern for n in self.nodes.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextureGraph):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]


def _node(node_id: str, raw: Any, loc: str) -> Node:
    if not isinstance(raw, dict):
        raise _err("a node must be an object with a type", loc)
    kind = raw.get("type")
    if kind not in NODE_TYPES:
        raise _err(
            f"unknown node type {kind!r}",
            f"{loc}.type",
            "UNKNOWN_NODE_TYPE",
            hint="node types: " + ", ".join(NODE_TYPES),
        )
    spec = NODE_TYPES[kind]
    names = {p.name for p in spec.all_params}
    unknown = sorted(set(raw) - names - {"type"})
    if unknown:
        raise _err(
            f"unknown parameter(s) for {kind}: {', '.join(unknown)}",
            loc,
            hint="parameters: " + ", ".join(sorted(names)),
        )
    params: dict[str, Any] = {}
    for p in spec.all_params:
        if p.name in raw:
            params[p.name] = _normalize_param(p, raw[p.name], f"{loc}.{p.name}")
        elif p.kind == "texture":
            raise _err(f"{kind} needs a {p.name}", loc)
        else:
            params[p.name] = p.default
    return Node(node_id, kind, params)


def _output_kind(node: Node, output: str | None) -> str:
    spec = node.spec
    if output is None:
        channels = spec.outputs[0][1]
        if node.type == "constant" and _is_number(node.params["value"]):
            return "value"
        if node.type == "ramp" and isinstance(node.params["stops"][0][1], float):
            return "value"
        if node.type == "mix" and all(_is_number(node.params[k]) for k in ("a", "b")):
            return "value"
        return "color" if channels == "color" else "value"
    if output in GENERIC_OUTPUTS:
        return "value"
    return "color" if dict(spec.outputs)[output] == "color" else "value"


def _output_channels(node: Node, output: str | None) -> str:
    """``"color"`` or a single channel letter for a reference."""
    if output is None:
        return node.spec.outputs[0][1]
    if output in GENERIC_OUTPUTS:
        return output
    return dict(node.spec.outputs)[output]


# --------------------------------------------------------------------------- #
# Literal conversion (shared by CPU and GPU)
# --------------------------------------------------------------------------- #


def linear_color(value: Any) -> tuple[float, float, float, float]:
    """A color input literal as linear RGBA (numbers are linear gray)."""
    if type(value).__name__ == "_LinearRGB":
        return (value[0], value[1], value[2], 1.0)
    if isinstance(value, float) or _is_number(value):
        v = float(value)
        return (v, v, v, 1.0)
    rgb = srgb_to_linear(parse_color(value))
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def vec3(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        return (float(value[0]), float(value[1]), float(value[2]))
    v = float(value)
    return (v, v, v)


def _f32(value: Any) -> Any:
    """Parameters as the GPU receives them: numbers rounded to float32 (ints unchanged)."""
    if isinstance(value, float):
        return float(np.float32(value))
    if isinstance(value, list) and value and all(isinstance(v, float) for v in value):
        return [float(np.float32(v)) for v in value]
    return value


def _f32_rgba(value: Any) -> np.ndarray:
    return np.array(linear_color(value), dtype=np.float32).astype(np.float64)


# --------------------------------------------------------------------------- #
# CPU evaluation
# --------------------------------------------------------------------------- #

ImageSampler = Callable[[str, str, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""``(texture ref, color_space, p, dpdx, dpdy) -> (N, 4)`` linear RGBA."""


@dataclass
class EvalContext:
    """Where a graph is evaluated: ``(N, 3)`` coordinates and their screen derivatives."""

    p: np.ndarray
    dpdx: np.ndarray
    dpdy: np.ndarray
    sample_image: ImageSampler | None = None


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, 0] * LUMINANCE[0] + rgb[:, 1] * LUMINANCE[1] + rgb[:, 2] * LUMINANCE[2]


def _smoothstep(e0: Any, e1: Any, x: np.ndarray) -> np.ndarray:
    span = np.asarray(e1, dtype=np.float64) - e0
    safe = np.where(span == 0.0, 1.0, span)
    t = np.clip((x - e0) / safe, 0.0, 1.0)
    t = np.where(span == 0.0, (x >= e1).astype(np.float64), t)
    return t * t * (3.0 - 2.0 * t)


def evaluate(graph: TextureGraph, ctx: EvalContext) -> np.ndarray:
    """Evaluate ``graph`` at ``ctx``'s points: ``(N, 4)`` float64 (linear RGB + alpha)."""
    n = ctx.p.shape[0]
    memo: dict[tuple[str, tuple[float, float, float]], np.ndarray] = {}

    def value_of(node_id: str, shift: tuple[float, float, float]) -> np.ndarray:
        key = (node_id, shift)
        if key not in memo:
            node = graph.nodes[node_id]
            inputs: dict[str, np.ndarray] = {}
            if node.type == "bump":
                step = node.params["step"]
                height = node.params["height"]
                inputs["height"] = read(height, "value", shift)
                inputs["height_x"] = read(height, "value", (shift[0] + step, shift[1], shift[2]))
                inputs["height_y"] = read(height, "value", (shift[0], shift[1] + step, shift[2]))
            else:
                for prm in node.spec.params:
                    if prm.connectable:
                        inputs[prm.name] = read(node.params[prm.name], prm.kind, shift)
            p = ctx.p if shift == (0.0, 0.0, 0.0) else ctx.p + np.array(shift)
            memo[key] = node_value(node, p, ctx.dpdx, ctx.dpdy, inputs, ctx.sample_image)
        return memo[key]

    def read(value: Any, kind: str, shift) -> np.ndarray:
        """An input: ``(N, 4)`` for colors, ``(N,)`` for numbers."""
        if isinstance(value, Ref):
            return select(value_of(value.node, shift), graph.nodes[value.node], value.output, kind)
        if kind == "value":
            return np.full(n, float(np.float32(value)))
        return np.tile(_f32_rgba(value), (n, 1))

    root = graph.output
    result = value_of(root.node, (0.0, 0.0, 0.0))
    return select(result, graph.nodes[root.node], root.output, "color")


def select(result: np.ndarray, node: Node, output: str | None, kind: str) -> np.ndarray:
    """A node's ``(N, 4)`` result read through one of its outputs, as a color or a number."""
    channels = _output_channels(node, output)
    if kind == "value":
        return _luma(result) if channels == "color" else result[:, "rgba".index(channels)]
    if channels == "color":
        return result
    v = result[:, "rgba".index(channels)]
    return np.stack([v, v, v, np.ones_like(v)], axis=1)


def node_value(
    node: Node,
    p: np.ndarray,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    inputs: dict[str, np.ndarray],
    sample_image: ImageSampler | None = None,
) -> np.ndarray:
    """One node's ``(N, 4)`` result at coordinates ``p`` (before its own scale and offset).

    ``inputs`` holds the node's connectable inputs, already evaluated:
    ``(N, 4)`` for colors and ``(N,)`` for numbers (``bump`` also receives
    ``height_x`` and ``height_y``, its height one step along x and y).
    """
    n = p.shape[0]
    # Numbers are used as float32, exactly as the GPU interpreter receives them, so both
    # backends place pattern edges at the same coordinates.
    prm = {k: _f32(v) for k, v in node.params.items()}
    node = Node(node.id, node.type, prm)
    t = node.type
    out = np.zeros((n, 4))
    out[:, 3] = 1.0
    if t == "constant":
        return inputs["value"]
    if t == "mix":
        return _mix(inputs["a"], inputs["b"], inputs["factor"][:, None], prm["mode"])
    if t == "ramp":
        return _ramp(inputs["factor"], prm["stops"], prm["interpolation"])
    if t == "math":
        v = _math(prm["op"], inputs["a"], inputs["b"], inputs["c"])
        out[:, 0] = out[:, 1] = out[:, 2] = v
        return out
    if t == "bump":
        return _bump(
            inputs["height"], inputs["height_x"], inputs["height_y"], prm["depth"], prm["step"]
        )
    scale = np.array(vec3(prm["scale"]))
    q = p * scale + np.array(vec3(prm["offset"]))
    if t == "coordinates":
        out[:, :3] = q
        return out
    if t == "checker":
        c = np.floor(q[:, 0]) + np.floor(q[:, 1])
        if prm["dimensions"] == 3:
            c = c + np.floor(q[:, 2])
        mask = c - 2.0 * np.floor(c * 0.5)
        out = _lerp(inputs["color1"], inputs["color2"], mask)
        out[:, 3] = mask
        return out
    if t == "gradient":
        v = _gradient(q, prm["kind"])
        out[:, 0] = out[:, 1] = out[:, 2] = v
        return out
    if t == "stripes":
        axis = prm["axis"]
        a = q[:, 0] + q[:, 1] if axis == "diagonal" else q[:, "xyz".index(axis)]
        mask = ((a - np.floor(a)) >= prm["width"]).astype(np.float64)
        out = _lerp(inputs["color1"], inputs["color2"], mask)
        out[:, 3] = mask
        return out
    if t == "noise":
        v = nz.noise(q, prm["seed"], prm["kind"])
        out[:, 0] = out[:, 1] = out[:, 2] = v
        return out
    if t == "fbm":
        v = nz.fbm(q, prm["seed"], prm["kind"], prm["octaves"], prm["lacunarity"], prm["gain"])
        out[:, 0] = out[:, 1] = out[:, 2] = v
        return out
    if t == "voronoi":
        out[:, :3] = nz.voronoi(q, prm["seed"], prm["randomness"], prm["dimensions"])
        return out
    if t in ("bricks", "tiles"):
        return _bricks(node, q, inputs)
    if t == "wood":
        return _wood(node, q, inputs)
    if t == "image":
        if sample_image is None:
            raise _err("image nodes need an image sampler", f"nodes.{node.id}")
        return sample_image(prm["texture"], prm["color_space"], q, dpdx * scale, dpdy * scale)
    raise AssertionError(t)  # pragma: no cover


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t[:, None]


def _gradient(p: np.ndarray, kind: str) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    if kind == "linear":
        v = x
    elif kind == "quadratic":
        v = np.maximum(x, 0.0) ** 2
    elif kind == "easing":
        c = np.clip(x, 0.0, 1.0)
        v = c * c * (3.0 - 2.0 * c)
    elif kind == "diagonal":
        v = (x + y) * 0.5
    elif kind == "radial":
        v = np.arctan2(y, x) / (2.0 * math.pi) + 0.5
    else:  # spherical
        v = np.maximum(1.0 - np.sqrt((p * p).sum(axis=1)), 0.0)
    return np.clip(v, 0.0, 1.0)


MIX_MODES = (
    "mix",
    "multiply",
    "add",
    "subtract",
    "screen",
    "difference",
    "minimum",
    "maximum",
    "overlay",
)


def _mix(a: np.ndarray, b: np.ndarray, f: np.ndarray, mode: str) -> np.ndarray:
    out = np.empty_like(a)
    ca, cb = a[:, :3], b[:, :3]
    if mode == "mix":
        blend = cb
    elif mode == "multiply":
        blend = ca * cb
    elif mode == "add":
        blend = ca + cb
    elif mode == "subtract":
        blend = ca - cb
    elif mode == "screen":
        blend = 1.0 - (1.0 - ca) * (1.0 - cb)
    elif mode == "difference":
        blend = np.abs(ca - cb)
    elif mode == "minimum":
        blend = np.minimum(ca, cb)
    elif mode == "maximum":
        blend = np.maximum(ca, cb)
    else:  # overlay
        blend = np.where(ca < 0.5, 2.0 * ca * cb, 1.0 - 2.0 * (1.0 - ca) * (1.0 - cb))
    out[:, :3] = ca + (blend - ca) * f
    out[:, 3] = a[:, 3] + (b[:, 3] - a[:, 3]) * f[:, 0]
    return out


def ramp_table(stops: list) -> tuple[np.ndarray, np.ndarray]:
    """Stop positions ``(k,)`` and linear values ``(k, 4)`` of a ramp (float32-exact)."""
    positions = np.array([float(s[0]) for s in stops], dtype=np.float32).astype(np.float64)
    values = np.array([_f32_rgba(s[1]) for s in stops], dtype=np.float64)
    return positions, values


def _ramp(f: np.ndarray, stops: list, interpolation: str) -> np.ndarray:
    positions, values = ramp_table(stops)
    count = len(positions)
    x = np.clip(f, 0.0, 1.0)
    # The segment [k, k+1] with positions[k] <= x < positions[k+1] (clamped to the ends).
    k = np.clip(np.searchsorted(positions, x, side="right") - 1, 0, count - 1)
    out = values[k].copy()
    if interpolation == "constant":
        return out
    nxt = np.minimum(k + 1, count - 1)
    span = positions[nxt] - positions[k]
    t = np.where(span > 0.0, (x - positions[k]) / np.where(span > 0.0, span, 1.0), 0.0)
    t = np.where(x < positions[0], 0.0, t)
    if interpolation == "smooth":
        t = t * t * (3.0 - 2.0 * t)
    return out + (values[nxt] - out) * t[:, None]


MATH_OPS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "minimum",
    "maximum",
    "absolute",
    "invert",
    "clamp",
    "smoothstep",
    "remap",
)


def _math(op: str, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        return np.where(b != 0.0, a / np.where(b != 0.0, b, 1.0), 0.0)
    if op == "power":
        base = np.maximum(a, 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.power(base, b)
        return np.where(np.isfinite(r), r, 0.0)
    if op == "minimum":
        return np.minimum(a, b)
    if op == "maximum":
        return np.maximum(a, b)
    if op == "absolute":
        return np.abs(a)
    if op == "invert":
        return 1.0 - a
    if op == "clamp":
        return np.minimum(np.maximum(a, b), c)
    if op == "smoothstep":
        return _smoothstep(b, c, a)
    # remap a from [b, c] to [0, 1]
    span = c - b
    return np.where(span != 0.0, (a - b) / np.where(span != 0.0, span, 1.0), 0.0)


def _bump(h: np.ndarray, hx: np.ndarray, hy: np.ndarray, depth: float, step: float) -> np.ndarray:
    sx = -depth * (hx - h) / step
    sy = -depth * (hy - h) / step
    inv = 1.0 / np.sqrt(sx * sx + sy * sy + 1.0)
    out = np.empty((h.shape[0], 4))
    out[:, 0] = sx * inv * 0.5 + 0.5
    out[:, 1] = sy * inv * 0.5 + 0.5
    out[:, 2] = inv * 0.5 + 0.5
    out[:, 3] = 1.0
    return out


def _bricks(node: Node, p: np.ndarray, inputs: dict[str, np.ndarray]) -> np.ndarray:
    prm = node.params
    if node.type == "bricks":
        w, h, row_offset = prm["brick_width"], prm["brick_height"], prm["row_offset"]
    else:
        w, h, row_offset = prm["tile_width"], prm["tile_height"], 0.0
    row = np.floor(p[:, 1] / h)
    shifted = row * row_offset
    x = p[:, 0] + (shifted - np.floor(shifted)) * w
    col = np.floor(x / w)
    lx = x - col * w
    ly = p[:, 1] - row * h
    d = np.minimum(np.minimum(lx, w - lx), np.minimum(ly, h - ly))
    half = prm["mortar"] * 0.5
    mortar = 1.0 - _smoothstep(half, half + prm["mortar"] * prm["smoothness"], d)
    tint = nz.cell_random(col, row, prm["seed"]) * prm["variation"]
    brick = _lerp(inputs["color1"], inputs["color2"], tint)
    out = _lerp(brick, inputs["mortar_color"], mortar)
    out[:, 3] = mortar
    return out


def _wood(node: Node, p: np.ndarray, inputs: dict[str, np.ndarray]) -> np.ndarray:
    prm = node.params
    axis = "xyz".index(prm["axis"])
    a, b = [i for i in range(3) if i != axis]
    stretch = np.full(3, 2.0)
    stretch[axis] = 0.5
    warp = nz.fbm(p * stretch, prm["seed"], "value", 3, 2.0, 0.5)
    r = np.sqrt(p[:, a] * p[:, a] + p[:, b] * p[:, b]) + prm["distortion"] * (warp - 0.5)
    ring = r * prm["rings"]
    ring = ring - np.floor(ring)
    fine = np.full(3, 40.0)
    fine[axis] = 2.0
    grain = nz.value_noise(p * fine, prm["seed"] + 1)
    t = np.clip(ring * ring + prm["grain"] * (grain - 0.5), 0.0, 1.0)
    out = _lerp(inputs["color1"], inputs["color2"], t)
    out[:, 3] = t
    return out


# --------------------------------------------------------------------------- #
# Compilation to GPU programs
# --------------------------------------------------------------------------- #

INSTRUCTION_WORDS = 8
REGISTER_BIT = 0x80000000
SELECT_ALL, SELECT_R, SELECT_G, SELECT_B, SELECT_A, SELECT_LUMA = range(6)
_SELECT = {"color": SELECT_ALL, "r": SELECT_R, "g": SELECT_G, "b": SELECT_B, "a": SELECT_A}
GRADIENT_KINDS = ("linear", "quadratic", "easing", "diagonal", "radial", "spherical")
STRIPE_AXES = ("x", "y", "z", "diagonal")
RAMP_MODES = ("linear", "constant", "smooth")


@dataclass
class Program:
    """A compiled texture graph: fixed-format instructions with numeric operands.

    Each instruction has :data:`INSTRUCTION_WORDS` ``uint32`` words:
    ``[opcode | aux << 8, destination register, operand 0..3, parameter offset, extra]``.
    An operand is a register (``REGISTER_BIT | register | selector << 8``) or
    the offset of a constant ``vec4`` in :attr:`params`. ``extra`` holds a seed
    or an image index (into :attr:`images`). The result is in register 0 after
    the last instruction (``OP_OUTPUT``).
    """

    words: np.ndarray  # (instructions * 8,) uint32
    params: np.ndarray  # (k,) float32
    images: list[tuple[str, str]]  # (texture reference, color space) per image index
    registers: int

    @property
    def instructions(self) -> int:
        return int(self.words.shape[0] // INSTRUCTION_WORDS)


@dataclass
class _Instr:
    op: int
    aux: int = 0
    dst: int = -1
    ins: list[tuple[str, Any, int]] = field(
        default_factory=list
    )  # ("reg", vreg, sel) / ("const", vec4)
    params: list[float] = field(default_factory=list)
    extra: int = 0


def compile_graph(graph: TextureGraph, location: str = "graph") -> Program:
    """Compile ``graph`` for the GPU interpreter (see :class:`Program`)."""
    instrs: list[_Instr] = []
    memo: dict[tuple[str, tuple[float, float, float]], int] = {}
    images: list[tuple[str, str]] = []
    state = {"shift": (0.0, 0.0, 0.0), "vregs": 0}

    def new_vreg() -> int:
        state["vregs"] += 1
        return state["vregs"] - 1

    def operand(value: Any, kind: str, shift) -> tuple[str, Any, int]:
        if isinstance(value, Ref):
            reg = emit(value.node, shift)
            channels = _output_channels(graph.nodes[value.node], value.output)
            if kind == "value" and channels == "color":
                return ("reg", reg, SELECT_LUMA)
            return ("reg", reg, _SELECT[channels])
        if kind == "value":
            v = float(value)
            return ("const", (v, v, v, 1.0), 0)
        return ("const", linear_color(value), 0)

    def emit(node_id: str, shift) -> int:
        key = (node_id, shift)
        if key in memo:
            return memo[key]
        node = graph.nodes[node_id]
        spec, prm = node.spec, node.params
        ins: list[tuple[str, Any, int]] = []
        params: list[float] = []
        aux = extra = 0
        kinds = {p.name: p.kind for p in spec.all_params}
        if node.type == "bump":
            step = prm["step"]
            for s in (
                shift,
                (shift[0] + step, shift[1], shift[2]),
                (shift[0], shift[1] + step, shift[2]),
            ):
                ins.append(operand(prm["height"], "value", s))
            params = [prm["depth"], step]
        else:
            for p in spec.params:
                if p.connectable:
                    ins.append(operand(prm[p.name], kinds[p.name], shift))
        if spec.pattern:
            params = [*vec3(prm["scale"]), *vec3(prm["offset"]), *params]
        t = node.type
        if t == "checker":
            aux = prm["dimensions"]
        elif t == "gradient":
            aux = GRADIENT_KINDS.index(prm["kind"])
        elif t == "stripes":
            params.append(prm["width"])
            aux = STRIPE_AXES.index(prm["axis"])
        elif t in ("noise", "fbm"):
            aux = 0 if prm["kind"] == "perlin" else 1
            extra = prm["seed"]
            if t == "fbm":
                aux |= prm["octaves"] << 4
                params += [prm["lacunarity"], prm["gain"]]
        elif t == "voronoi":
            params.append(prm["randomness"])
            aux = prm["dimensions"]
            extra = prm["seed"]
        elif t in ("bricks", "tiles"):
            if t == "bricks":
                size = [prm["brick_width"], prm["brick_height"]]
                row_offset = prm["row_offset"]
            else:
                size = [prm["tile_width"], prm["tile_height"]]
                row_offset = 0.0
            params += [*size, prm["mortar"], prm["smoothness"], row_offset, prm["variation"]]
            extra = prm["seed"]
        elif t == "wood":
            params += [prm["rings"], prm["distortion"], prm["grain"]]
            aux = "xyz".index(prm["axis"])
            extra = prm["seed"]
        elif t == "mix":
            aux = MIX_MODES.index(prm["mode"])
        elif t == "ramp":
            stops = prm["stops"]
            aux = RAMP_MODES.index(prm["interpolation"]) | (len(stops) << 4)
            for position, value in stops:
                params += [position, *linear_color(value)[:3]]
        elif t == "math":
            aux = MATH_OPS.index(prm["op"])
        elif t == "image":
            aux = 0 if prm["color_space"] == "srgb" else 1
            extra = len(images)
            images.append((prm["texture"], prm["color_space"]))
        if spec.pattern and state["shift"] != shift:
            instrs.append(_Instr(OP_SET_COORD, params=list(shift)))
            state["shift"] = shift
        reg = new_vreg()
        instrs.append(_Instr(spec.opcode, aux, reg, ins, params, extra))
        memo[key] = reg
        return reg

    final = operand(graph.output, "color", (0.0, 0.0, 0.0))
    instrs.append(_Instr(OP_OUTPUT, 0, -2, [final]))
    if len(instrs) > MAX_INSTRUCTIONS:
        raise _err(
            f"the graph compiles to {len(instrs)} instructions (limit {MAX_INSTRUCTIONS})",
            location,
            "GRAPH_TOO_COMPLEX",
            hint="use fewer nodes, octaves or nested bump nodes",
        )
    return _assemble(instrs, images, location)


def _assemble(instrs: list[_Instr], images: list[tuple[str, str]], location: str) -> Program:
    # Liveness: the last instruction reading each virtual register.
    last_use: dict[int, int] = {}
    for i, ins in enumerate(instrs):
        for kind, value, _ in ins.ins:
            if kind == "reg":
                last_use[value] = i
    physical: dict[int, int] = {}
    free = list(range(MAX_REGISTERS - 1, 0, -1))  # register 0 is reserved for the output
    used = 1
    words: list[int] = []
    pool: list[float] = []
    for i, ins in enumerate(instrs):
        encoded = []
        for kind, value, sel in ins.ins:
            if kind == "reg":
                encoded.append(REGISTER_BIT | physical[value] | (sel << 8))
            else:
                encoded.append(len(pool))
                pool.extend(value)
        for kind, value, _ in ins.ins:  # operands read for the last time free their register
            if kind == "reg" and last_use.get(value) == i and value in physical:
                free.append(physical.pop(value))
        if ins.dst == -2:
            dst = 0
        elif ins.dst >= 0:
            if not free:
                raise _err(
                    f"the graph needs more than {MAX_REGISTERS - 1} values at once",
                    location,
                    "GRAPH_TOO_COMPLEX",
                    hint="simplify the graph (fewer branches evaluated at the same time)",
                )
            dst = free.pop()
            physical[ins.dst] = dst
            used = max(used, dst + 1)
        else:
            dst = 0
        param_offset = len(pool)
        pool.extend(ins.params)
        encoded += [0] * (4 - len(encoded))
        words += [ins.op | (ins.aux << 8), dst, *encoded, param_offset, ins.extra & 0xFFFFFFFF]
    return Program(
        np.array(words, dtype=np.uint32),
        np.array(pool, dtype=np.float32),
        images,
        used,
    )


def run_program(program: Program, ctx: EvalContext) -> np.ndarray:
    """Execute a compiled program with NumPy, instruction by instruction (``(N, 4)`` float64).

    This is the GPU interpreter's algorithm on the CPU; tests use it to check
    the compiler (register allocation, operand encoding) against :func:`evaluate`.
    """
    n = ctx.p.shape[0]
    regs = [np.zeros((n, 4)) for _ in range(MAX_REGISTERS)]
    params = program.params.astype(np.float64)
    coord = ctx.p
    words = program.words.reshape(-1, INSTRUCTION_WORDS)

    def load(word: int) -> np.ndarray:
        if word & REGISTER_BIT:
            value = regs[word & 0xF]
            sel = (word >> 8) & 0x7
            if sel == SELECT_ALL:
                return value
            v = _luma(value) if sel == SELECT_LUMA else value[:, sel - 1]
            return np.stack([v, v, v, np.ones(n)], axis=1)
        return np.tile(params[word : word + 4], (n, 1))

    for row in words:
        op, aux = int(row[0]) & 0xFF, int(row[0]) >> 8
        dst, base, extra = int(row[1]), int(row[6]), int(row[7])
        if op == OP_SET_COORD:
            coord = ctx.p + params[base : base + 3]
            continue
        operands = [load(int(w)) for w in row[2:6]]
        if op == OP_OUTPUT:
            regs[0] = operands[0]
            continue
        node = _node_from_instruction(OPCODES[op], aux, params, base, extra, program)
        if node.type == "bump":
            names = ["height", "height_x", "height_y"]
            inputs = {k: operands[i][:, 0] for i, k in enumerate(names)}
        else:
            connectable = [p for p in node.spec.params if p.connectable]
            inputs = {
                p.name: operands[i][:, 0] if p.kind == "value" else operands[i]
                for i, p in enumerate(connectable)
            }
        regs[dst] = node_value(node, coord, ctx.dpdx, ctx.dpdy, inputs, ctx.sample_image)
    return regs[0]


def _node_from_instruction(
    name: str, aux: int, params: np.ndarray, base: int, extra: int, program: Program
) -> Node:
    """Rebuild a node's parameters from an instruction (for :func:`run_program`)."""
    spec = NODE_TYPES[name]
    prm: dict[str, Any] = {}
    i = base
    if spec.pattern:
        prm["scale"] = params[i : i + 3].tolist()
        prm["offset"] = params[i + 3 : i + 6].tolist()
        i += 6
    if name == "checker":
        prm["dimensions"] = aux
    elif name == "gradient":
        prm["kind"] = GRADIENT_KINDS[aux]
    elif name == "stripes":
        prm["width"], prm["axis"] = float(params[i]), STRIPE_AXES[aux]
    elif name in ("noise", "fbm"):
        prm["kind"] = "perlin" if aux & 0xF == 0 else "value"
        prm["seed"] = extra
        if name == "fbm":
            prm["octaves"] = aux >> 4
            prm["lacunarity"], prm["gain"] = float(params[i]), float(params[i + 1])
    elif name == "voronoi":
        prm["randomness"], prm["dimensions"], prm["seed"] = float(params[i]), aux, extra
    elif name in ("bricks", "tiles"):
        w, h, mortar, smooth, row_offset, variation = params[i : i + 6].tolist()
        prm.update(mortar=mortar, smoothness=smooth, variation=variation, seed=extra)
        if name == "bricks":
            prm.update(brick_width=w, brick_height=h, row_offset=row_offset)
        else:
            prm.update(tile_width=w, tile_height=h)
    elif name == "wood":
        prm["rings"], prm["distortion"], prm["grain"] = params[i : i + 3].tolist()
        prm["axis"], prm["seed"] = "xyz"[aux], extra
    elif name == "mix":
        prm["mode"] = MIX_MODES[aux]
    elif name == "ramp":
        count = aux >> 4
        prm["interpolation"] = RAMP_MODES[aux & 0xF]
        # Stop values were converted to linear light when compiled: pass them as numbers.
        prm["stops"] = [
            [float(params[i + 4 * k]), _LinearRGB(params[i + 4 * k + 1 : i + 4 * k + 4])]
            for k in range(count)
        ]
    elif name == "math":
        prm["op"] = MATH_OPS[aux]
    elif name == "bump":
        prm["depth"], prm["step"] = float(params[i]), float(params[i + 1])
    elif name == "image":
        prm["texture"], prm["color_space"] = program.images[extra]
    return Node("instruction", name, prm)


class _LinearRGB(tuple):
    """A ramp stop value that is already linear RGB (used by :func:`run_program`)."""

    def __new__(cls, values):
        return super().__new__(cls, (float(v) for v in values))
