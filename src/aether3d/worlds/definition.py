"""World definitions: the validated JSON document that *is* a world, defaults and presets.

A world is fully described by its definition (plus its sparse edits, see
:mod:`aether3d.worlds.edits`). The definition is plain JSON -- numbers, strings, lists and
objects -- checked here key by key against :data:`SPEC`, completed with defaults, and
never mutated afterwards (a change replaces it), so the history keeps definitions by
reference and caches key their results by its content.

Sections (every number in metres, degrees, degrees Celsius or a stated fraction):

* ``seed`` (0 .. 2^53 - 1), ``generator`` (the algorithm version, :data:`GENERATOR`),
  ``mode`` (``"terrain"``: continuous surfaces; ``"blocks"``: a block world),
  ``extent`` (``{"min": [x, y], "max": [x, y]}`` or ``null`` for an unbounded world),
  ``sea_level``;
* ``terrain`` -- the layered height recipe: ``continents`` (macro land/ocean field and its
  height curve), ``ranges`` (where mountain chains run), ``mountains`` (ridged multifractal
  peaks), ``hills``, ``valleys``, ``plateaus`` (terraced mesas), ``canyons``, ``dunes``
  (in desert biomes), ``detail``, ``warp`` (domain warping) and an optional final
  ``curve``;
* ``climate`` -- temperature and moisture fields (mean, variation, gradients, lapse rate,
  water influence, rain shadow, wind);
* ``biomes`` -- the biome table (``preset`` plus ``table`` overrides), ``blend`` width;
* ``surface`` -- beaches, cliffs, snow and tree lines;
* ``hydrology`` -- ``mode`` (``drainage``: erosion, lakes and rivers from a regional
  simulation; ``contour``: rivers along noise valleys, no regional stage; ``none``), the
  regional grid, erosion, rivers and lakes;
* ``caves``, ``vegetation``, ``structures``, ``blocks``, ``materials``, ``streaming``;
* ``features`` -- authored terrain features and overrides, applied in order (see
  :data:`FEATURE_TYPES`).

:func:`build_definition` makes a complete definition from a preset and overrides;
:func:`check_definition` validates one (unknown keys, wrong types, out-of-range values and
oversized lists are errors naming their location).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

from ..errors import WorldError

GENERATOR = 1
"""The world-generation algorithm version: every result is keyed by it; a later Aether3D
that changes what a definition generates bumps it and keeps generating old versions or
refuses them."""

MAX_SEED = 2**53 - 1
MAX_COORDINATE = 1.0e7
"""Largest world coordinate (10,000 km): float64 keeps sub-millimetre steps there."""
MAX_EXTENT = 2.0e6
"""Largest side of a bounded world (2,000 km)."""
MAX_FEATURES = 4096
MAX_FEATURE_POINTS = 4096
MAX_BIOMES = 64
MAX_SPECIES = 32
MAX_CURVE_POINTS = 32
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\- ]{0,63}$")
FEATURE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

MODES = ("terrain", "blocks")
HYDROLOGY_MODES = ("drainage", "contour", "none")


def _err(message: str, location: str, code: str = "INVALID_DEFINITION", hint: str | None = None):
    return WorldError(code, message, location, hint=hint)


# --------------------------------------------------------------------------------------- #
# Value checks
# --------------------------------------------------------------------------------------- #


def _number(v: Any, lo: float, hi: float, loc: str, integer: bool = False) -> float | int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _err(f"expected a number, got {type(v).__name__}", loc)
    if integer:
        if isinstance(v, float):
            if not v.is_integer():
                raise _err(f"expected an integer, got {v}", loc)
            v = int(v)
    else:
        v = float(v)
        if not math.isfinite(v):
            raise _err(f"{v} is not finite", loc)
    if not lo <= v <= hi:
        raise _err(f"{v} is outside [{lo}, {hi}]", loc)
    return v


_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _color(v: Any, loc: str) -> str:
    """A display colour as ``#rrggbb`` (from ``#rrggbb`` or ``[r, g, b]`` in 0..1)."""
    if isinstance(v, str) and _HEX.match(v):
        return v.lower()
    if isinstance(v, (list, tuple)) and len(v) == 3:
        rgb = [_number(c, 0.0, 1.0, f"{loc}[{i}]") for i, c in enumerate(v)]
        return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)
    raise _err("expected a colour: '#rrggbb' or [r, g, b] in 0..1", loc)


def _curve(v: Any, loc: str) -> list[list[float]]:
    if not isinstance(v, (list, tuple)) or not 2 <= len(v) <= MAX_CURVE_POINTS:
        raise _err(f"expected 2..{MAX_CURVE_POINTS} [input, output] pairs", loc)
    out = []
    for i, p in enumerate(v):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise _err("expected an [input, output] pair", f"{loc}[{i}]")
        out.append(
            [
                _number(p[0], -1e7, 1e7, f"{loc}[{i}][0]"),
                _number(p[1], -1e7, 1e7, f"{loc}[{i}][1]"),
            ]
        )
    for i in range(1, len(out)):
        if not out[i][0] > out[i - 1][0]:
            raise _err("inputs must increase", f"{loc}[{i}][0]")
    return out


def _vec(v: Any, n: int, lo: float, hi: float, loc: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise _err(f"expected {n} numbers", loc)
    return [_number(c, lo, hi, f"{loc}[{i}]") for i, c in enumerate(v)]


def _name(v: Any, loc: str) -> str:
    if not isinstance(v, str) or not NAME.match(v):
        raise _err("expected a name (letters, digits, _ . - and spaces, at most 64)", loc)
    return v


# A spec is a callable check(value, location) -> normalized value, or a dict of specs
# (an object with exactly those keys, missing ones taken from the defaults).


def F(lo: float, hi: float):
    return lambda v, loc: _number(v, lo, hi, loc)


def I(lo: int, hi: int):  # noqa: E743 (a spec constructor, like F)
    return lambda v, loc: _number(v, lo, hi, loc, integer=True)


def B(v: Any, loc: str) -> bool:
    if not isinstance(v, bool):
        raise _err(f"expected true or false, got {type(v).__name__}", loc)
    return v


def E(*choices: str):
    def check(v: Any, loc: str) -> str:
        if v not in choices:
            raise _err(f"expected one of {', '.join(map(repr, choices))}, got {v!r}", loc)
        return v

    return check


def NULL(spec):
    return lambda v, loc: None if v is None else spec(v, loc)


def LIST(spec, max_len: int, min_len: int = 0):
    def check(v: Any, loc: str) -> list:
        if not isinstance(v, (list, tuple)) or not min_len <= len(v) <= max_len:
            raise _err(f"expected a list of {min_len}..{max_len} items", loc)
        return [spec(x, f"{loc}[{i}]") for i, x in enumerate(v)]

    return check


COLOR = _color
CURVE = _curve
NAMES = LIST(_name, MAX_BIOMES)


def _normalize(value: Any, spec: Any, default: Any, loc: str) -> Any:
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            raise _err(f"expected an object, got {type(value).__name__}", loc)
        out = {}
        for key in value:
            if key not in spec:
                raise _err(
                    f"unknown key {key!r}",
                    f"{loc}.{key}",
                    hint="keys: " + ", ".join(spec),
                )
        for key, sub in spec.items():
            sub_default = default.get(key) if isinstance(default, dict) else None
            if key in value:
                out[key] = _normalize(value[key], sub, sub_default, f"{loc}.{key}")
            else:
                out[key] = copy.deepcopy(sub_default)
        return out
    return spec(value, loc)


# --------------------------------------------------------------------------------------- #
# Biomes
# --------------------------------------------------------------------------------------- #

SPECIES = (
    # trees
    "oak", "birch", "beech", "maple", "pine", "spruce", "fir", "larch", "palm", "acacia",
    "baobab", "jungle", "willow", "dead_tree", "cactus",
    # small plants and ground cover
    "bush", "shrub", "fern", "grass", "tall_grass", "flowers", "reeds", "mushroom",
    "dead_bush", "heather",
    # rocks and debris
    "rock", "boulder", "pebbles", "log",
)  # fmt: skip
"""Procedural species the vegetation system can generate (``worlds/assets.py``)."""

SPECIES_SPEC = {
    "species": E(*SPECIES),
    "density": F(0.0, 50_000.0),  # per hectare
    "cluster": F(0.0, 1.0),  # 0: even spread, 1: tight groves
    "scale": F(0.1, 10.0),  # size multiplier
    "slope": F(0.0, 90.0),  # steepest slope it grows on (degrees)
}
SPECIES_DEFAULT = {"species": "oak", "density": 10.0, "cluster": 0.4, "scale": 1.0, "slope": 30.0}


def _species(v: Any, loc: str) -> dict:
    return _normalize(v, SPECIES_SPEC, SPECIES_DEFAULT, loc)


GROUNDS = ("grass", "dirt", "sand", "red_sand", "snow", "gravel", "mud", "rock", "ice")

BIOME_SPEC = {
    "temperature": lambda v, loc: _range(v, -60.0, 60.0, loc),
    "moisture": lambda v, loc: _range(v, 0.0, 1.0, loc),
    "ground": E(*GROUNDS),
    "colors": {
        "grass": COLOR,
        "soil": COLOR,
        "rock": COLOR,
        "sand": COLOR,
        "foliage": COLOR,
        "water": COLOR,
    },
    "terrain": {"dunes": F(0.0, 1.0), "flatten": F(0.0, 1.0), "roughness": F(0.0, 4.0)},
    "vegetation": LIST(_species, MAX_SPECIES),
    "blocks": {
        "top": lambda v, loc: _block_name(v, loc),
        "filler": lambda v, loc: _block_name(v, loc),
        "depth": I(0, 16),
        "underwater": lambda v, loc: _block_name(v, loc),
    },
}


def _range(v: Any, lo: float, hi: float, loc: str) -> list[float]:
    pair = _vec(v, 2, lo, hi, loc)
    if not pair[1] > pair[0]:
        raise _err("expected [low, high] with low < high", loc)
    return pair


def _block_name(v: Any, loc: str) -> str:
    from .blocks import BLOCKS

    if v not in BLOCKS:
        raise _err(f"unknown block {v!r}", loc, hint="blocks: " + ", ".join(sorted(BLOCKS)))
    return v


def _biome(name: str, temperature, moisture, ground, colors, vegetation, *, terrain=None,
           blocks=None) -> tuple[str, dict]:  # fmt: skip
    return name, {
        "temperature": list(temperature),
        "moisture": list(moisture),
        "ground": ground,
        "colors": colors,
        "terrain": {"dunes": 0.0, "flatten": 0.0, "roughness": 1.0, **(terrain or {})},
        "vegetation": [
            {**SPECIES_DEFAULT, **({"species": s, "density": d} if isinstance(s, str) else s)}
            for s, d in vegetation
        ],
        "blocks": {
            "top": "grass_block",
            "filler": "dirt",
            "depth": 3,
            "underwater": "sand",
            **(blocks or {}),
        },
    }


def _colors(grass, soil="#6b5134", rock="#807a72", sand="#d8c79a", foliage=None, water="#2f6f8f"):
    return {
        "grass": grass,
        "soil": soil,
        "rock": rock,
        "sand": sand,
        "foliage": foliage or grass,
        "water": water,
    }


def _v(species: str, density: float, **extra: Any) -> tuple[Any, float]:
    return ({"species": species, "density": density, **extra}, density)


EARTH_BIOMES: dict[str, dict] = dict(
    [
        _biome("ice_sheet", (-60, -12), (0.0, 1.0), "snow",
               _colors("#e8eef2", "#9aa3a8", "#8a8f94", "#d9dde0", water="#3a6a86"), [],
               blocks={"top": "snow_block", "filler": "packed_ice", "underwater": "gravel"}),
        _biome("tundra", (-12, -2), (0.0, 1.0), "grass",
               _colors("#7e8a60", "#5d5040", "#7e7a74", "#c9bf9c", "#64704a"),
               [_v("heather", 400.0, cluster=0.6), _v("rock", 6.0), _v("pebbles", 30.0)],
               blocks={"top": "snowy_grass_block"}),
        _biome("snowy_taiga", (-8, 1), (0.35, 1.0), "snow",
               _colors("#7d9474", "#4f4334", "#7a7670", "#cbbf9e", "#3f5f45"),
               [_v("spruce", 220.0, cluster=0.5), _v("fir", 60.0), _v("rock", 4.0)],
               blocks={"top": "snowy_grass_block"}),
        _biome("taiga", (-2, 7), (0.35, 1.0), "grass",
               _colors("#566f47", "#4f4130", "#77736c", "#c8bc98", "#34553a"),
               [_v("spruce", 260.0, cluster=0.45), _v("pine", 60.0), _v("fern", 600.0),
                _v("rock", 5.0), _v("log", 3.0)]),
        _biome("alpine_meadow", (-4, 6), (0.0, 0.35), "grass",
               _colors("#7a9152", "#5a4a36", "#86817a", "#cbbf9c", "#5a7a3c"),
               [_v("flowers", 900.0, cluster=0.7), _v("grass", 2500.0), _v("rock", 12.0),
                _v("larch", 6.0, cluster=0.8)]),
        _biome("temperate_forest", (6, 18), (0.5, 0.82), "grass",
               _colors("#4f7434", "#5c4430", "#7a766f", "#d1c49c", "#36602a"),
               [_v("oak", 120.0, cluster=0.4), _v("beech", 70.0), _v("birch", 30.0),
                _v("bush", 160.0), _v("fern", 500.0), _v("mushroom", 40.0), _v("log", 4.0),
                _v("rock", 3.0)]),
        _biome("dense_woodland", (6, 16), (0.82, 1.0), "grass",
               _colors("#41662c", "#4a3826", "#6e6a63", "#c9bc94", "#2b5222"),
               [_v("beech", 200.0, cluster=0.3), _v("oak", 120.0), _v("maple", 60.0),
                _v("fern", 1500.0), _v("mushroom", 120.0), _v("log", 8.0)]),
        _biome("birch_forest", (4, 13), (0.42, 0.65), "grass",
               _colors("#5f8540", "#5c4632", "#7d7972", "#d3c6a0", "#4f7a34"),
               [_v("birch", 180.0, cluster=0.5), _v("bush", 80.0), _v("flowers", 300.0),
                _v("grass", 1500.0)]),
        _biome("plains", (6, 22), (0.22, 0.5), "grass",
               _colors("#6f9142", "#6a5236", "#857f76", "#d6c79c", "#557c30"),
               [_v("grass", 3500.0, cluster=0.3), _v("flowers", 500.0, cluster=0.75),
                _v("oak", 2.5, cluster=0.85), _v("bush", 10.0, cluster=0.8), _v("rock", 1.5)]),
        _biome("meadow", (4, 14), (0.5, 0.75), "grass",
               _colors("#69924a", "#5d4732", "#837e77", "#d1c39b", "#527a36"),
               [_v("grass", 4000.0), _v("flowers", 1400.0, cluster=0.65),
                _v("birch", 4.0, cluster=0.9)]),
        _biome("savanna", (22, 32), (0.15, 0.45), "grass",
               _colors("#a59a58", "#8a6a44", "#8f8274", "#d9c28e", "#717c36"),
               [_v("acacia", 6.0, cluster=0.6), _v("baobab", 0.6, cluster=0.2),
                _v("tall_grass", 2500.0, cluster=0.4), _v("bush", 12.0)],
               blocks={"top": "dry_grass_block"}),
        _biome("desert", (18, 45), (0.0, 0.15), "sand",
               _colors("#c9b26a", "#c9a46c", "#b48a62", "#e2c992", "#8f8f4a", "#3a7f8c"),
               [_v("cactus", 4.0, cluster=0.3), _v("dead_bush", 12.0), _v("rock", 2.0)],
               terrain={"dunes": 1.0, "roughness": 0.6},
               blocks={"top": "sand", "filler": "sand", "depth": 4, "underwater": "sand"}),
        _biome("badlands", (14, 40), (0.0, 0.2), "red_sand",
               _colors("#b07a4a", "#a35f3a", "#a5603e", "#c97a46", "#8a7a3a"),
               [_v("dead_bush", 8.0), _v("cactus", 1.0), _v("rock", 6.0)],
               terrain={"roughness": 1.2},
               blocks={"top": "red_sand", "filler": "terracotta", "depth": 6,
                       "underwater": "red_sand"}),
        _biome("tropical_forest", (22, 45), (0.6, 1.0), "grass",
               _colors("#3a7a2a", "#4f3a26", "#6f6a62", "#d8c99a", "#2a6a22"),
               [_v("jungle", 160.0, cluster=0.3), _v("palm", 30.0), _v("fern", 2000.0),
                _v("bush", 300.0), _v("tall_grass", 600.0)]),
        _biome("swamp", (8, 30), (0.78, 1.0), "mud",
               _colors("#506232", "#3f3626", "#666258", "#a99a72", "#44562a", "#3c4f3a"),
               [_v("willow", 50.0, cluster=0.5), _v("reeds", 1200.0, cluster=0.7),
                _v("mushroom", 60.0), _v("log", 6.0)],
               terrain={"flatten": 0.85, "roughness": 0.4},
               blocks={"top": "grass_block", "filler": "mud", "underwater": "mud"}),
        _biome("rocky_highlands", (-2, 12), (0.0, 0.42), "gravel",
               _colors("#7b8456", "#5e5446", "#8a857d", "#c4b898"),
               [_v("rock", 40.0, cluster=0.5), _v("boulder", 6.0), _v("pine", 12.0),
                _v("heather", 300.0)],
               terrain={"roughness": 1.4},
               blocks={"top": "gravel", "filler": "stone", "depth": 2}),
    ]
)  # fmt: skip
"""The ``"earth"`` biome table: Whittaker-style climate envelopes (temperature in degrees
Celsius, moisture 0..1) with colours, ground, terrain modifiers, vegetation and blocks."""

BIOME_PRESETS = {"earth": EARTH_BIOMES}


def _biome_table(v: Any, loc: str) -> dict[str, dict]:
    if not isinstance(v, dict):
        raise _err("expected an object of biomes by name", loc)
    if len(v) > MAX_BIOMES:
        raise _err(f"at most {MAX_BIOMES} biomes", loc)
    out = {}
    for name, spec in v.items():
        _name(name, f"{loc}.{name}")
        base = EARTH_BIOMES.get(name) or EARTH_BIOMES["plains"]
        out[name] = _normalize(spec, BIOME_SPEC, base, f"{loc}.{name}")
    return out


# --------------------------------------------------------------------------------------- #
# Features (authored overrides)
# --------------------------------------------------------------------------------------- #

SHAPE_STAGE = ("raise", "lower", "scale", "mountain", "ridge", "valley")
FINAL_STAGE = ("flatten", "pad", "road", "river", "lake", "smooth", "biome", "clear", "cave")
FEATURE_TYPES = SHAPE_STAGE + FINAL_STAGE
"""Feature types. *Shape* features change landforms before the regional stage (rivers and
erosion respond to them); *final* features apply after it, locally (pads, roads, authored
rivers and lakes, smoothing, biome overrides, vegetation clearing, caves)."""


def _points(v: Any, loc: str, minimum: int = 1) -> list[list[float]]:
    if not isinstance(v, (list, tuple)) or not minimum <= len(v) <= MAX_FEATURE_POINTS:
        raise _err(f"expected {minimum}..{MAX_FEATURE_POINTS} points", loc)
    out = []
    for i, p in enumerate(v):
        if not isinstance(p, (list, tuple)) or len(p) not in (2, 3):
            raise _err("expected [x, y] or [x, y, z]", f"{loc}[{i}]")
        out.append([_number(c, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}[{i}][{k}]")
                    for k, c in enumerate(p)])  # fmt: skip
    return out


def _shape(v: Any, loc: str) -> dict:
    """``{"circle": [x, y, r]}``, ``{"rect": [x0, y0, x1, y1]}``, ``{"polygon": [[x, y]...]}``
    or ``{"line": [[x, y]...], "width": w}``."""
    if not isinstance(v, dict) or not v:
        raise _err("expected a shape: circle, rect, polygon or line", loc)
    keys = set(v) - {"width"}
    if len(keys) != 1:
        raise _err("a shape has exactly one of circle, rect, polygon, line", loc)
    kind = keys.pop()
    if kind == "circle":
        c = _vec(v["circle"], 3, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}.circle")
        if not c[2] > 0:
            raise _err("the radius must be positive", f"{loc}.circle[2]")
        out: dict = {"circle": c}
    elif kind == "rect":
        r = _vec(v["rect"], 4, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}.rect")
        if not (r[2] > r[0] and r[3] > r[1]):
            raise _err("expected [x0, y0, x1, y1] with x1 > x0 and y1 > y0", f"{loc}.rect")
        out = {"rect": r}
    elif kind == "polygon":
        out = {"polygon": [p[:2] for p in _points(v["polygon"], f"{loc}.polygon", 3)]}
    elif kind == "line":
        out = {"line": _points(v["line"], f"{loc}.line", 2)}
    else:
        raise _err(f"unknown shape {kind!r}", f"{loc}.{kind}", hint="circle, rect, polygon, line")
    if "width" in v:
        if kind != "line":
            raise _err("width applies to a line", f"{loc}.width")
        out["width"] = _number(v["width"], 0.01, 1e5, f"{loc}.width")
    elif kind == "line":
        out["width"] = 8.0
    return out


_FEATURE_FIELDS: dict[str, dict[str, Any]] = {
    "raise": {"amount": F(-1e4, 1e4), "falloff": F(0.0, 1e5)},
    "lower": {"amount": F(-1e4, 1e4), "falloff": F(0.0, 1e5)},
    "scale": {"factor": F(0.0, 10.0), "falloff": F(0.0, 1e5), "base": NULL(F(-1e5, 1e5))},
    "mountain": {"height": F(-1e4, 1e4), "radius": F(1.0, 1e5), "sharpness": F(0.0, 1.0)},
    "ridge": {"height": F(-1e4, 1e4), "falloff": F(1.0, 1e5), "sharpness": F(0.0, 1.0)},
    "valley": {"depth": F(0.0, 1e4), "falloff": F(1.0, 1e5)},
    "flatten": {"height": NULL(F(-1e5, 1e5)), "falloff": F(0.0, 1e5)},
    "pad": {"height": F(-1e5, 1e5), "falloff": F(0.0, 1e4), "structure": NULL(_name)},
    "road": {"falloff": F(0.0, 1e3), "material": E("gravel", "dirt", "cobble", "asphalt", "path")},
    "river": {"depth": F(0.1, 1e3), "level": NULL(LIST(F(-1e5, 1e5), MAX_FEATURE_POINTS, 2))},
    "lake": {"level": F(-1e5, 1e5), "depth": F(0.1, 1e3), "falloff": F(0.0, 1e4)},
    "smooth": {"factor": F(0.0, 1.0), "falloff": F(0.0, 1e5)},
    "biome": {"biome": _name, "strength": F(0.0, 1.0), "falloff": F(0.0, 1e5)},
    "clear": {
        "what": LIST(E("trees", "plants", "rocks", "all"), 4, 1),
        "falloff": F(0.0, 1e4),
    },
    "cave": {
        "kind": E("tunnel", "cavern", "arch", "hole"),
        "radius": F(0.5, 500.0),
        "radius_end": NULL(F(0.5, 500.0)),
        "noise": F(0.0, 1.0),
    },
}
_FEATURE_DEFAULTS: dict[str, dict[str, Any]] = {
    "raise": {"amount": 50.0, "falloff": 200.0},
    "lower": {"amount": 50.0, "falloff": 200.0},
    "scale": {"factor": 1.25, "falloff": 300.0, "base": None},
    "mountain": {"height": 800.0, "radius": 1500.0, "sharpness": 0.6},
    "ridge": {"height": 400.0, "falloff": 800.0, "sharpness": 0.6},
    "valley": {"depth": 80.0, "falloff": 400.0},
    "flatten": {"height": None, "falloff": 20.0},
    "pad": {"height": 0.0, "falloff": 6.0, "structure": None},
    "road": {"falloff": 4.0, "material": "gravel"},
    "river": {"depth": 2.0, "level": None},
    "lake": {"level": 0.0, "depth": 4.0, "falloff": 30.0},
    "smooth": {"factor": 0.2, "falloff": 50.0},
    "biome": {"biome": "temperate_forest", "strength": 1.0, "falloff": 200.0},
    "clear": {"what": ["all"], "falloff": 2.0},
    "cave": {"kind": "tunnel", "radius": 4.0, "radius_end": None, "noise": 0.3},
}


def check_feature(v: Any, loc: str) -> dict:
    """One feature: ``{"type", "id", "shape" (or "at" for mountains), ...fields}``."""
    if not isinstance(v, dict):
        raise _err("expected a feature object", loc)
    kind = v.get("type")
    if kind not in _FEATURE_FIELDS:
        raise _err(
            f"unknown feature type {kind!r}",
            f"{loc}.type",
            hint="types: " + ", ".join(FEATURE_TYPES),
        )
    fid = v.get("id")
    if not isinstance(fid, str) or not FEATURE_ID.match(fid):
        raise _err("every feature needs an id (letters, digits, _ . -, at most 64)", f"{loc}.id")
    fields = _FEATURE_FIELDS[kind]
    allowed = {"type", "id", "shape", "enabled", *fields}
    for key in v:
        if key not in allowed:
            raise _err(f"unknown key {key!r}", f"{loc}.{key}", hint="keys: " + ", ".join(allowed))
    if "shape" not in v:
        raise _err("a feature needs a shape", f"{loc}.shape")
    out: dict = {"type": kind, "id": fid, "shape": _shape(v["shape"], f"{loc}.shape")}
    if kind == "cave":
        shape = out["shape"]
        if "line" not in shape and "circle" not in shape:
            raise _err("a cave is a line (tunnel, arch) or a circle (cavern, hole)", f"{loc}.shape")
        pts = shape.get("line") or []
        if any(len(p) != 3 for p in pts):
            raise _err("cave lines need [x, y, z] points", f"{loc}.shape.line")
    out["enabled"] = B(v.get("enabled", True), f"{loc}.enabled")
    defaults = _FEATURE_DEFAULTS[kind]
    for key, spec in fields.items():
        out[key] = spec(v[key], f"{loc}.{key}") if key in v else copy.deepcopy(defaults[key])
    return out


def _structure_items(v: Any, loc: str) -> list[dict]:
    from .structures import check_items

    return check_items(v, loc)


def _features(v: Any, loc: str) -> list[dict]:
    if not isinstance(v, (list, tuple)) or len(v) > MAX_FEATURES:
        raise _err(f"expected a list of at most {MAX_FEATURES} features", loc)
    out = [check_feature(f, f"{loc}[{i}]") for i, f in enumerate(v)]
    seen: set[str] = set()
    for i, f in enumerate(out):
        if f["id"] in seen:
            raise _err(f"duplicate feature id {f['id']!r}", f"{loc}[{i}].id")
        seen.add(f["id"])
    return out


# --------------------------------------------------------------------------------------- #
# The specification and the defaults
# --------------------------------------------------------------------------------------- #


def _extent(v: Any, loc: str) -> dict | None:
    if v is None:
        return None
    if not isinstance(v, dict) or set(v) != {"min", "max"}:
        raise _err('expected {"min": [x, y], "max": [x, y]} or null (unbounded)', loc)
    lo = _vec(v["min"], 2, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}.min")
    hi = _vec(v["max"], 2, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}.max")
    for k in (0, 1):
        if not hi[k] - lo[k] >= 16.0:
            raise _err("each side must be at least 16 m", f"{loc}.max[{k}]")
        if hi[k] - lo[k] > MAX_EXTENT:
            raise _err(f"each side is at most {MAX_EXTENT:g} m", f"{loc}.max[{k}]")
    return {"min": lo, "max": hi}


def _rect(v: Any, loc: str) -> list[float] | None:
    if v is None:
        return None
    r = _vec(v, 4, -MAX_COORDINATE, MAX_COORDINATE, loc)
    if not (r[2] > r[0] and r[3] > r[1]):
        raise _err("expected [x0, y0, x1, y1] with x1 > x0 and y1 > y0", loc)
    return r


def _seed(v: Any, loc: str) -> int:
    return _number(v, 0, MAX_SEED, loc, integer=True)


def _materials(v: Any, loc: str) -> dict[str, str]:
    if not isinstance(v, dict) or len(v) > 256:
        raise _err("expected an object of layer -> material name or id", loc)
    out = {}
    for k, m in v.items():
        if not isinstance(k, str) or not re.match(r"^[a-z][a-z0-9_]{0,63}$", k):
            raise _err("layer names are lowercase identifiers", f"{loc}.{k}")
        if not isinstance(m, str) or not 0 < len(m) <= 128:
            raise _err("expected a material name or id", f"{loc}.{k}")
        out[k] = m
    return out


SPEC: dict[str, Any] = {
    "seed": _seed,
    "generator": I(1, GENERATOR),
    "mode": E(*MODES),
    "extent": _extent,
    "sea_level": F(-1e4, 1e4),
    "terrain": {
        "scale": F(0.01, 100.0),
        "height_scale": F(0.0, 100.0),
        "continents": {
            "scale": F(100.0, 1e6),
            "octaves": I(1, 8),
            "land": F(0.0, 1.0),
            "warp": F(0.0, 2.0),
            "edge": F(0.0, 1e5),
            "curve": CURVE,
        },
        "ranges": {
            "scale": F(100.0, 1e6),
            "coverage": F(0.0, 1.0),
            "warp": F(0.0, 2.0),
            "sharpness": F(0.0, 1.0),
        },
        "mountains": {
            "height": F(0.0, 2e4),
            "scale": F(10.0, 1e6),
            "octaves": I(1, 12),
            "ridged": F(0.0, 1.0),
            "sharpness": F(0.0, 8.0),
            "roughness": F(0.1, 0.9),
            "warp": F(0.0, 2.0),
        },
        "hills": {"height": F(0.0, 1e4), "scale": F(10.0, 1e6), "octaves": I(1, 12)},
        "valleys": {"depth": F(0.0, 1e4), "scale": F(10.0, 1e6), "width": F(0.001, 1.0)},
        "plateaus": {
            "coverage": F(0.0, 1.0),
            "scale": F(10.0, 1e6),
            "height": F(0.0, 1e4),
            "step": F(0.0, 1e4),
            "sharpness": F(0.0, 1.0),
        },
        "canyons": {
            "coverage": F(0.0, 1.0),
            "scale": F(10.0, 1e6),
            "depth": F(0.0, 1e4),
            "width": F(0.001, 1.0),
        },
        "dunes": {"height": F(0.0, 500.0), "scale": F(1.0, 1e5), "direction": F(-360.0, 360.0)},
        "detail": {"height": F(0.0, 1e3), "scale": F(0.5, 1e5), "octaves": I(1, 10)},
        "warp": {"amplitude": F(0.0, 1e5), "scale": F(10.0, 1e6)},
        "curve": NULL(CURVE),
    },
    "climate": {
        "temperature": F(-60.0, 60.0),
        "variation": F(0.0, 60.0),
        "gradient": lambda v, loc: _vec(v, 2, -100.0, 100.0, loc),
        "lapse_rate": F(0.0, 60.0),  # degrees per km (block worlds compress heights)
        "moisture": F(0.0, 1.0),
        "moisture_variation": F(0.0, 1.0),
        "moisture_gradient": lambda v, loc: _vec(v, 2, -10.0, 10.0, loc),
        "scale": F(100.0, 1e6),
        "water_influence": F(0.0, 1.0),
        "water_distance": F(1.0, 1e6),
        "rain_shadow": F(0.0, 1.0),
        "wind": F(-360.0, 360.0),
        "boundary_noise": F(0.0, 1.0),
    },
    "biomes": {
        "preset": E(*BIOME_PRESETS),
        "blend": F(0.01, 1.0),
        "table": _biome_table,
        "disabled": NAMES,
    },
    "surface": {
        "beach_height": F(0.0, 100.0),
        "cliff_slope": F(5.0, 89.0),
        "snow_temperature": F(-60.0, 30.0),
        "treeline_temperature": F(-60.0, 30.0),
    },
    "hydrology": {
        "mode": E(*HYDROLOGY_MODES),
        "extent": _rect,
        "cell": F(1.0, 1000.0),
        "max_cells": I(1024, 16_777_216),
        "erosion": {
            "iterations": I(0, 1000),
            "strength": F(0.0, 100.0),
            "deposition": F(0.0, 1.0),
            "thermal": F(0.0, 1.0),
            "talus": F(5.0, 80.0),
            "uplift": F(0.0, 10.0),
        },
        "rivers": {
            "threshold": F(1e3, 1e12),
            "width": F(0.0, 1.0),
            "max_width": F(0.5, 2000.0),
            "depth": F(0.0, 100.0),
            "contour_width": F(0.0005, 0.2),
        },
        "lakes": {
            "enabled": B,
            "max": I(0, 10_000),
            "min_area": F(0.0, 1e10),
            "min_depth": F(0.0, 1e3),
        },
    },
    "caves": {
        "enabled": B,
        "tunnels": F(0.0, 1.0),
        "caverns": F(0.0, 1.0),
        "overhangs": F(0.0, 1.0),
        "radius": F(0.5, 100.0),
        "depth": lambda v, loc: _range(v, -1e4, 1e4, loc),
        "entrances": F(0.0, 1.0),
        "scale": F(4.0, 1e5),
    },
    "vegetation": {
        "density": F(0.0, 10.0),
        "clutter": F(0.0, 10.0),
        "rocks": F(0.0, 10.0),
        "distance": F(0.0, 1e5),
        "detail_distance": F(0.0, 1e4),
        "variants": I(1, 16),
    },
    "structures": {
        "settlements": I(0, 256),
        "size": lambda v, loc: [int(x) for x in _range(v, 1, 400, loc)],
        "spacing": F(10.0, 1e6),
        "castles": I(0, 64),
        "towers": I(0, 256),
        "ruins": I(0, 256),
        "roads": B,
        "bridges": B,
        "biomes": NAMES,
        "max_slope": F(1.0, 60.0),
        "items": lambda v, loc: _structure_items(v, loc),
    },
    "blocks": {
        "size": F(0.0625, 64.0),
        "min_height": I(-4096, 4096),
        "max_height": I(-4096, 4096),
        "trees": B,
        "ores": B,
        "plants": B,
    },
    "materials": _materials,
    "streaming": {
        "chunk": F(4.0, 4096.0),
        "resolution": I(4, 256),
        "lod_error": F(0.1, 64.0),
        "view_distance": F(16.0, 1e6),
        "volume_distance": F(0.0, 1e5),
        "memory_mb": F(16.0, 1e6),
        "max_tiles": I(16, 1_000_000),
        "threads": I(0, 256),
    },
    "features": _features,
}

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "generator": GENERATOR,
    "mode": "terrain",
    "extent": {"min": [-2048.0, -2048.0], "max": [2048.0, 2048.0]},
    "sea_level": 0.0,
    "terrain": {
        "scale": 1.0,
        "height_scale": 1.0,
        "continents": {
            "scale": 14000.0,
            "octaves": 4,
            "land": 0.62,
            "warp": 0.35,
            "edge": 0.0,
            "curve": [
                [-1.0, -420.0],
                [-0.3, -60.0],
                [-0.06, -6.0],
                [0.0, 1.5],
                [0.12, 10.0],
                [0.45, 55.0],
                [1.0, 180.0],
            ],  # fmt: skip
        },
        "ranges": {"scale": 9000.0, "coverage": 0.45, "warp": 0.5, "sharpness": 0.5},
        "mountains": {
            "height": 1100.0,
            "scale": 3000.0,
            "octaves": 7,
            "ridged": 0.75,
            "sharpness": 2.0,
            "roughness": 0.47,
            "warp": 0.25,
        },
        "hills": {"height": 55.0, "scale": 1100.0, "octaves": 6},
        "valleys": {"depth": 0.0, "scale": 4000.0, "width": 0.06},
        "plateaus": {
            "coverage": 0.0,
            "scale": 6000.0,
            "height": 110.0,
            "step": 22.0,
            "sharpness": 0.8,
        },  # fmt: skip
        "canyons": {"coverage": 0.0, "scale": 3200.0, "depth": 70.0, "width": 0.05},
        "dunes": {"height": 5.0, "scale": 80.0, "direction": 30.0},
        "detail": {"height": 2.5, "scale": 45.0, "octaves": 4},
        "warp": {"amplitude": 320.0, "scale": 2600.0},
        "curve": None,
    },
    "climate": {
        "temperature": 13.0,
        "variation": 5.0,
        "gradient": [0.0, 0.0],
        "lapse_rate": 6.5,
        "moisture": 0.55,
        "moisture_variation": 0.22,
        "moisture_gradient": [0.0, 0.0],
        "scale": 7000.0,
        "water_influence": 0.2,
        "water_distance": 1200.0,
        "rain_shadow": 0.3,
        "wind": 180.0,
        "boundary_noise": 0.12,
    },
    "biomes": {"preset": "earth", "blend": 0.12, "table": {}, "disabled": []},
    "surface": {
        "beach_height": 2.0,
        "cliff_slope": 40.0,
        "snow_temperature": -2.5,
        "treeline_temperature": 1.5,
    },
    "hydrology": {
        "mode": "drainage",
        "extent": None,
        "cell": 16.0,
        "max_cells": 4_194_304,
        "erosion": {
            "iterations": 40,
            "strength": 2.0,
            "deposition": 0.3,
            "thermal": 0.3,
            "talus": 42.0,
            "uplift": 0.0,
        },
        "rivers": {
            "threshold": 1.5e6,
            "width": 0.008,
            "max_width": 60.0,
            "depth": 1.5,
            "contour_width": 0.012,
        },
        "lakes": {"enabled": True, "max": 6, "min_area": 40000.0, "min_depth": 2.0},
    },
    "caves": {
        "enabled": False,
        "tunnels": 0.4,
        "caverns": 0.2,
        "overhangs": 0.0,
        "radius": 3.5,
        "depth": [-80.0, -4.0],
        "entrances": 0.35,
        "scale": 120.0,
    },
    "vegetation": {
        "density": 1.0,
        "clutter": 1.0,
        "rocks": 1.0,
        "distance": 2500.0,
        "detail_distance": 90.0,
        "variants": 6,
    },
    "structures": {
        "settlements": 0,
        "size": [6, 16],
        "spacing": 1500.0,
        "castles": 0,
        "towers": 0,
        "ruins": 0,
        "roads": True,
        "bridges": True,
        "biomes": ["plains", "meadow", "temperate_forest", "birch_forest", "savanna", "taiga"],
        "max_slope": 14.0,
        "items": [],
    },
    "blocks": {
        "size": 1.0,
        "min_height": -64,
        "max_height": 192,
        "trees": True,
        "ores": True,
        "plants": True,
    },
    "materials": {},
    "streaming": {
        "chunk": 64.0,
        "resolution": 64,
        "lod_error": 3.0,
        "view_distance": 12000.0,
        "volume_distance": 400.0,
        "memory_mb": 1024.0,
        "max_tiles": 20000,
        "threads": 1,
    },
    "features": [],
}


def _pow2(v: float) -> bool:
    m, e = math.frexp(v)
    return m == 0.5


def _cross_checks(d: dict, loc: str) -> None:
    s = d["streaming"]
    if not _pow2(s["chunk"]):
        raise _err(
            "chunk must be a power of two (exact vertex coordinates)", f"{loc}.streaming.chunk"
        )
    if not _pow2(float(s["resolution"])):
        raise _err("resolution must be a power of two", f"{loc}.streaming.resolution")
    b = d["blocks"]
    if not b["max_height"] - b["min_height"] >= 16:
        raise _err("max_height must exceed min_height by at least 16", f"{loc}.blocks.max_height")
    if b["max_height"] - b["min_height"] > 4096:
        raise _err("at most 4096 blocks of height", f"{loc}.blocks.max_height")
    if d["mode"] == "blocks" and not _pow2(b["size"]):
        raise _err("block size must be a power of two (1 = one metre)", f"{loc}.blocks.size")
    h = d["hydrology"]
    if h["mode"] == "drainage" and d["extent"] is None and h["extent"] is None:
        raise _err(
            "an unbounded world needs hydrology.extent for drainage rivers",
            f"{loc}.hydrology.extent",
            hint="give [x0, y0, x1, y1], or use mode 'contour' (rivers along noise valleys)",
        )
    biomes = biome_table(d)
    for i, name in enumerate(d["biomes"]["disabled"]):
        if name not in {**BIOME_PRESETS[d["biomes"]["preset"]], **d["biomes"]["table"]}:
            raise _err(f"unknown biome {name!r}", f"{loc}.biomes.disabled[{i}]", "UNKNOWN_BIOME")
    if not biomes:
        raise _err("every biome is disabled", f"{loc}.biomes.disabled")
    for i, f in enumerate(d["features"]):
        if f["type"] == "biome" and f["biome"] not in biomes:
            raise _err(
                f"unknown biome {f['biome']!r}",
                f"{loc}.features[{i}].biome",
                "UNKNOWN_BIOME",
                hint="biomes: " + ", ".join(biomes),
            )
    for i, name in enumerate(d["structures"]["biomes"]):
        if name not in biomes:
            raise _err(f"unknown biome {name!r}", f"{loc}.structures.biomes[{i}]", "UNKNOWN_BIOME")


def check_definition(value: Any, location: str = "definition") -> dict:
    """A complete, validated copy of ``value`` (missing keys take the defaults)."""
    if not isinstance(value, dict):
        raise _err(f"expected an object, got {type(value).__name__}", location)
    out = _normalize(value, SPEC, DEFAULTS, location)
    _cross_checks(out, location)
    return out


def biome_table(d: dict) -> dict[str, dict]:
    """The biomes of a definition: its preset's table with ``table`` merged over it,
    without ``disabled`` ones (name -> biome, in table order)."""
    table = dict(BIOME_PRESETS[d["biomes"]["preset"]])
    table.update(d["biomes"]["table"])
    for name in d["biomes"]["disabled"]:
        table.pop(name, None)
    return table


def merge(base: Any, patch: Any) -> Any:
    """``patch`` deep-merged over ``base``: objects merge key by key, anything else
    (numbers, lists, ``null``) replaces."""
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            out[k] = merge(base.get(k), v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(patch)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(*parts: Any) -> str:
    """A content key: SHA-256 (hex, 32 characters) of the canonical JSON of ``parts``."""
    return hashlib.sha256(canonical(list(parts))).hexdigest()[:32]


# --------------------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------------------- #

PRESETS: dict[str, dict[str, Any]] = {
    "temperate": {},
    "alpine_valley": {
        "extent": {"min": [-4096.0, -4096.0], "max": [4096.0, 4096.0]},
        "terrain": {
            "continents": {"land": 0.95, "curve": [[-1.0, 300.0], [0.0, 600.0], [1.0, 900.0]]},
            "ranges": {"scale": 4500.0, "coverage": 0.85, "sharpness": 0.45},
            "mountains": {
                "height": 3200.0,
                "scale": 4200.0,
                "ridged": 0.45,
                "octaves": 9,
                "roughness": 0.45,
            },  # fmt: skip
            "hills": {"height": 90.0, "scale": 900.0},
            "valleys": {"depth": 220.0, "scale": 3200.0, "width": 0.09},
            "detail": {"height": 2.0, "scale": 40.0},
            "warp": {"amplitude": 420.0, "scale": 2400.0},
        },
        "climate": {"temperature": 11.0, "moisture": 0.62, "variation": 3.0},
        "hydrology": {
            "cell": 16.0,
            "erosion": {
                "iterations": 60,
                "strength": 3.0,
                "deposition": 0.3,
                "talus": 36.0,
                "thermal": 0.5,
            },  # fmt: skip
            "rivers": {"threshold": 3.0e6, "width": 0.01, "max_width": 45.0},
            "lakes": {"max": 3},
        },
    },
    "forested_mountains": {
        "extent": {"min": [-4096.0, -4096.0], "max": [4096.0, 4096.0]},
        "terrain": {
            "continents": {"land": 0.92, "curve": [[-1.0, 20.0], [0.0, 120.0], [1.0, 300.0]]},
            "ranges": {"coverage": 0.6, "sharpness": 0.35},
            "mountains": {"height": 1200.0, "ridged": 0.45, "scale": 2200.0},
            "hills": {"height": 120.0, "scale": 800.0},
        },
        "climate": {"temperature": 12.0, "moisture": 0.78, "variation": 3.0},
        "vegetation": {"density": 1.2},
    },
    "archipelago": {
        "extent": {"min": [-6144.0, -6144.0], "max": [6144.0, 6144.0]},
        "terrain": {
            "continents": {
                "scale": 5200.0,
                "land": 0.38,
                "warp": 0.6,
                "edge": 2500.0,
                "curve": [
                    [-1.0, -300.0],
                    [-0.25, -40.0],
                    [-0.04, -3.0],
                    [0.0, 1.2],
                    [0.2, 25.0],
                    [1.0, 140.0],
                ],  # fmt: skip
            },
            "ranges": {"scale": 4000.0, "coverage": 0.35},
            "mountains": {"height": 480.0, "scale": 1500.0, "ridged": 0.55},
            "hills": {"height": 35.0, "scale": 700.0},
        },
        "climate": {"temperature": 27.0, "moisture": 0.7, "variation": 2.0},
        "surface": {"beach_height": 3.5},
    },
    "canyon_desert": {
        "extent": {"min": [-4096.0, -4096.0], "max": [4096.0, 4096.0]},
        "terrain": {
            "continents": {"land": 0.95, "curve": [[-1.0, 60.0], [0.0, 180.0], [1.0, 320.0]]},
            "ranges": {"coverage": 0.25},
            "mountains": {"height": 500.0, "ridged": 0.6},
            "hills": {"height": 30.0},
            "plateaus": {"coverage": 0.55, "height": 140.0, "step": 18.0, "sharpness": 0.9},
            "canyons": {"coverage": 0.7, "depth": 120.0, "width": 0.045, "scale": 2600.0},
            "dunes": {"height": 7.0},
        },
        "climate": {
            "temperature": 30.0,
            "moisture": 0.08,
            "moisture_variation": 0.08,
            "variation": 4.0,
        },  # fmt: skip
        "hydrology": {
            "erosion": {"iterations": 30, "strength": 0.6, "thermal": 0.7},
            "rivers": {"threshold": 2.0e7},
        },  # fmt: skip
    },
    "snowy_wilderness": {
        "extent": {"min": [-4096.0, -4096.0], "max": [4096.0, 4096.0]},
        "terrain": {
            "continents": {"land": 0.9, "curve": [[-1.0, 0.0], [0.0, 160.0], [1.0, 380.0]]},
            "ranges": {"coverage": 0.55},
            "mountains": {"height": 1700.0, "ridged": 0.8},
        },
        "climate": {"temperature": -1.0, "moisture": 0.55, "variation": 4.0},
        "surface": {"snow_temperature": -4.0, "treeline_temperature": -3.5},
    },
    "rolling_plains": {
        "terrain": {
            "ranges": {"coverage": 0.1},
            "mountains": {"height": 250.0},
            "hills": {"height": 40.0, "scale": 1500.0},
        },
        "climate": {"temperature": 16.0, "moisture": 0.45},
    },
    "biome_continent": {
        "extent": {"min": [-16384.0, -16384.0], "max": [16384.0, 16384.0]},
        "terrain": {
            "continents": {"scale": 22000.0, "land": 0.68, "edge": 3000.0},
            "ranges": {"scale": 12000.0, "coverage": 0.35},
            "mountains": {"height": 1500.0, "scale": 3200.0},
        },
        "climate": {
            "temperature": 12.0,
            "gradient": [0.0, -1.3],
            "moisture": 0.5,
            "moisture_gradient": [0.03, 0.0],
            "variation": 3.0,
        },
        "hydrology": {"cell": 32.0, "erosion": {"iterations": 40}},
    },
    "cavernous_highlands": {
        "extent": {"min": [-2048.0, -2048.0], "max": [2048.0, 2048.0]},
        "terrain": {
            "continents": {"land": 0.95, "curve": [[-1.0, 60.0], [0.0, 150.0], [1.0, 260.0]]},
            "mountains": {"height": 700.0, "ridged": 0.7},
            "plateaus": {"coverage": 0.3, "height": 60.0, "step": 12.0},
        },
        "caves": {
            "enabled": True,
            "tunnels": 0.6,
            "caverns": 0.4,
            "overhangs": 0.6,
            "depth": [-70.0, -3.0],
            "entrances": 0.3,
        },
    },
    "infinite_highlands": {
        "extent": None,
        "terrain": {
            "continents": {"scale": 30000.0, "land": 0.75},
            "valleys": {"depth": 60.0, "scale": 5000.0},
        },
        "hydrology": {"mode": "contour"},
    },
    "blocky_adventure": {
        "mode": "blocks",
        "extent": {"min": [-512.0, -512.0], "max": [512.0, 512.0]},
        "terrain": {
            "continents": {
                "land": 0.8,
                "scale": 3000.0,
                "curve": [[-1.0, -30.0], [-0.1, -6.0], [0.0, 1.0], [0.2, 8.0], [1.0, 30.0]],
            },  # fmt: skip
            "ranges": {"scale": 1400.0, "coverage": 0.55, "sharpness": 0.4},
            "mountains": {"height": 95.0, "scale": 420.0, "ridged": 0.7, "octaves": 6},
            "hills": {"height": 12.0, "scale": 160.0},
            "detail": {"height": 1.2, "scale": 24.0},
            "warp": {"amplitude": 60.0, "scale": 500.0},
            "dunes": {"height": 2.0, "scale": 30.0},
        },
        "hydrology": {
            "cell": 4.0,
            "erosion": {"iterations": 25, "strength": 0.7},
            "rivers": {"threshold": 3.0e5, "width": 0.006, "max_width": 9.0, "depth": 1.5},
        },
        "climate": {"temperature": 12.0, "moisture": 0.58, "scale": 1800.0, "lapse_rate": 30.0},
        "caves": {"enabled": True, "depth": [-60.0, 10.0], "scale": 40.0},
        "streaming": {
            "chunk": 16.0,
            "resolution": 16,
            "view_distance": 2000.0,
            "volume_distance": 0.0,
        },  # fmt: skip
        "vegetation": {"distance": 2000.0},
    },
}
"""Starting points; each is a patch over :data:`DEFAULTS` (see docs/WORLDS.md)."""


def build_definition(
    preset: str | None = None,
    *,
    seed: int | None = None,
    mode: str | None = None,
    size: float | None = None,
    extent: Any = "keep",
    overrides: dict | None = None,
    location: str = "definition",
) -> dict:
    """A complete definition: ``preset`` (default ``"temperate"``), then ``seed``, ``mode``,
    ``size`` (a square extent of that side centred on the origin), ``extent`` and
    ``overrides`` (a deep patch, see :func:`merge`), validated."""
    name = "temperate" if preset is None else preset
    if name not in PRESETS:
        raise _err(
            f"unknown preset {name!r}",
            f"{location}.preset",
            "UNKNOWN_PRESET",
            hint="presets: " + ", ".join(PRESETS),
        )
    d = merge(DEFAULTS, PRESETS[name])
    if seed is not None:
        d["seed"] = seed
    if mode is not None:
        d["mode"] = mode
    if size is not None:
        half = _number(size, 16.0, MAX_EXTENT, f"{location}.size") / 2.0
        d["extent"] = {"min": [-half, -half], "max": [half, half]}
    if extent != "keep":
        d["extent"] = extent
    if overrides:
        if not isinstance(overrides, dict):
            raise _err("overrides is an object (a patch of the definition)", f"{location}")
        d = merge(d, overrides)
    return check_definition(d, location)


def describe_presets() -> dict[str, str]:
    """A one-line description of each preset (for guides, the CLI and MCP)."""
    return {
        "temperate": "the defaults: mixed lowlands and a mountain range, temperate climate",
        "alpine_valley": "8 km of high alpine ranges with eroded valleys, rivers and lakes",
        "forested_mountains": "rounded wooded mountains in a wet temperate climate",
        "archipelago": "12 km of tropical islands in an ocean, beaches and palms",
        "canyon_desert": "hot terraced mesas cut by canyons, dunes in the lowlands",
        "snowy_wilderness": "cold taiga and tundra under snowy peaks, frozen lakes",
        "rolling_plains": "gentle grassland hills with scattered trees",
        "biome_continent": "32 km continent with north-south temperature and east-west "
        "moisture gradients: every biome",
        "cavernous_highlands": "highlands with caves, tunnels and overhanging cliffs",
        "infinite_highlands": "an unbounded world with noise-valley rivers (no regional stage)",
        "blocky_adventure": "a Minecraft-style block world: mountains, rivers, forests, caves",
    }
