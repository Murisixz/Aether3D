"""Material presets: stone, brick, wood, metal, grass and fabric, built from procedural textures.

Every preset is an ordinary ``"standard"`` material whose texture slots are
bound to procedural textures created alongside it (named
``"<material> base_color"``, ``"<material> roughness"``, ...). Nothing is a
bitmap: the patterns are node graphs (see :mod:`aether3d.materials.graph`)
that can be inspected and edited like any other texture.

Patterns are sized in scene units (meters): a brick is 25 cm long, stones
are about 30 cm across. By default they are projected from object space
(``coordinates="object"``, triplanar), which needs no UVs and keeps the
pattern attached to the object. Options shared by all presets:

* ``scale`` -- feature size multiplier (2 = twice as large);
* ``seed`` -- variation of the random patterns;
* ``coordinates`` -- ``"object"`` (default), ``"world"`` or ``"uv"``;
* ``projection`` -- for object/world coordinates (default ``"triplanar"``);
* ``color`` (and preset-specific colors) -- the main color;

and any material parameter (``roughness``, ``metallic``, ...) overrides the
preset's value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..errors import MaterialError
from ..rendering.color import color_to_hex, parse_color

if TYPE_CHECKING:
    from .library import MaterialLibrary
    from .material import Material

COMMON_OPTIONS = ("scale", "seed", "coordinates", "projection", "color")


def _hex(value: Any, location: str) -> str:
    return color_to_hex(parse_color(value, location))


def _stone(o: dict[str, Any]) -> dict[str, Any]:
    color = o.get("color", "#8c8a83")
    seed = o["seed"]
    return {
        "params": {"roughness": 1.0, "specular": 0.4},
        "textures": {
            "base_color": {
                "output": "color",
                "nodes": {
                    "cells": {"type": "voronoi", "scale": 3.2, "randomness": 0.85, "seed": seed},
                    "inside": {
                        "type": "math",
                        "op": "smoothstep",
                        "a": {"node": "cells", "output": "edge"},
                        "b": 0.015,
                        "c": 0.09,
                    },
                    "tint": {
                        "type": "ramp",
                        "factor": {"node": "cells", "output": "cell"},
                        "stops": [
                            [0.0, _shade(color, 0.78)],
                            [0.55, color],
                            [1.0, _shade(color, 1.18)],
                        ],
                    },
                    "detail": {"type": "fbm", "scale": 9.0, "octaves": 4, "seed": seed + 1},
                    "grain": {
                        "type": "ramp",
                        "factor": {"node": "detail"},
                        "stops": [[0.25, 0.72], [0.75, 1.0]],
                    },
                    "stones": {
                        "type": "mix",
                        "a": {"node": "tint"},
                        "b": {"node": "grain"},
                        "factor": 1.0,
                        "mode": "multiply",
                    },
                    "color": {
                        "type": "mix",
                        "a": o.get("mortar_color", _shade(color, 0.55)),
                        "b": {"node": "stones"},
                        "factor": {"node": "inside"},
                    },
                },
            },
            "roughness": {
                "output": "rough",
                "nodes": {
                    "cells": {"type": "voronoi", "scale": 3.2, "randomness": 0.85, "seed": seed},
                    "detail": {"type": "fbm", "scale": 9.0, "octaves": 3, "seed": seed + 1},
                    "rough": {
                        "type": "ramp",
                        "factor": {"node": "mixed"},
                        "stops": [[0.0, 0.95], [1.0, 0.72]],
                    },
                    "mixed": {
                        "type": "math",
                        "op": "multiply",
                        "a": {"node": "inside"},
                        "b": {"node": "detail"},
                    },
                    "inside": {
                        "type": "math",
                        "op": "smoothstep",
                        "a": {"node": "cells", "output": "edge"},
                        "b": 0.015,
                        "c": 0.09,
                    },
                },
            },
            "normal": {
                "output": "bump",
                "nodes": {
                    "cells": {"type": "voronoi", "scale": 3.2, "randomness": 0.85, "seed": seed},
                    "inside": {
                        "type": "math",
                        "op": "smoothstep",
                        "a": {"node": "cells", "output": "edge"},
                        "b": 0.0,
                        "c": 0.12,
                    },
                    "detail": {"type": "fbm", "scale": 14.0, "octaves": 3, "seed": seed + 2},
                    "rough": {"type": "math", "op": "multiply", "a": {"node": "detail"}, "b": 0.25},
                    "height": {
                        "type": "math",
                        "op": "add",
                        "a": {"node": "inside"},
                        "b": {"node": "rough"},
                    },
                    "bump": {
                        "type": "bump",
                        "height": {"node": "height"},
                        "depth": 0.012,
                        "step": 0.002,
                    },
                },
            },
        },
    }


def _brick(o: dict[str, Any]) -> dict[str, Any]:
    c1 = o.get("color", "#8f4632")
    c2 = o.get("color2", _shade(c1, 1.25))
    mortar = o.get("mortar_color", "#b8b0a2")
    seed = o["seed"]
    width, height = o.get("brick_width", 0.25), o.get("brick_height", 0.075)
    bricks = {
        "type": "bricks",
        "brick_width": width,
        "brick_height": height,
        "mortar": o.get("mortar", 0.012),
        "smoothness": 0.6,
        "variation": 0.7,
        "color1": c1,
        "color2": c2,
        "mortar_color": mortar,
        "seed": seed,
    }
    return {
        "params": {"roughness": 1.0},
        "textures": {
            "base_color": {
                "output": "color",
                "nodes": {
                    "bricks": bricks,
                    "dirt": {"type": "fbm", "scale": 4.0, "octaves": 4, "seed": seed + 1},
                    "grime": {
                        "type": "ramp",
                        "factor": {"node": "dirt"},
                        "stops": [[0.3, 0.7], [0.7, 1.0]],
                    },
                    "color": {
                        "type": "mix",
                        "a": {"node": "bricks"},
                        "b": {"node": "grime"},
                        "factor": 1.0,
                        "mode": "multiply",
                    },
                },
            },
            "roughness": {
                "output": "rough",
                "nodes": {
                    "bricks": bricks,
                    "rough": {
                        "type": "ramp",
                        "factor": {"node": "bricks", "output": "mortar"},
                        "stops": [[0.0, 0.78], [1.0, 0.96]],
                    },
                },
            },
            "normal": {
                "output": "bump",
                "nodes": {
                    "bricks": bricks,
                    "solid": {
                        "type": "math",
                        "op": "invert",
                        "a": {"node": "bricks", "output": "mortar"},
                    },
                    "detail": {"type": "fbm", "scale": 25.0, "octaves": 3, "seed": seed + 2},
                    "pits": {"type": "math", "op": "multiply", "a": {"node": "detail"}, "b": 0.12},
                    "height": {
                        "type": "math",
                        "op": "add",
                        "a": {"node": "solid"},
                        "b": {"node": "pits"},
                    },
                    "bump": {
                        "type": "bump",
                        "height": {"node": "height"},
                        "depth": 0.006,
                        "step": 0.001,
                    },
                },
            },
        },
    }


def _wood(o: dict[str, Any]) -> dict[str, Any]:
    c1 = o.get("color", "#c08a55")
    c2 = o.get("color2", _shade(c1, 0.62))
    seed = o["seed"]
    wood = {
        "type": "wood",
        "scale": 6.0,
        "rings": 1.6,
        "distortion": 0.6,
        "grain": 0.35,
        "axis": o.get("axis", "x"),
        "color1": c1,
        "color2": c2,
        "seed": seed,
    }
    return {
        "projection": "volume",
        "params": {"roughness": 1.0},
        "textures": {
            "base_color": {"output": "w", "nodes": {"w": wood}},
            "roughness": {
                "output": "rough",
                "nodes": {
                    "w": wood,
                    "rough": {
                        "type": "ramp",
                        "factor": {"node": "w", "output": "rings"},
                        "stops": [[0.0, 0.5], [1.0, 0.68]],
                    },
                },
            },
        },
    }


def _metal(o: dict[str, Any]) -> dict[str, Any]:
    seed = o["seed"]
    brushed = {"type": "fbm", "scale": [3.0, 3.0, 160.0], "octaves": 3, "seed": seed}
    return {
        "params": {"metallic": 1.0, "roughness": 1.0, "base_color": o.get("color", "#c4c8cc")},
        "textures": {
            "roughness": {
                "output": "rough",
                "nodes": {
                    "smudge": {"type": "fbm", "scale": 2.5, "octaves": 4, "seed": seed + 1},
                    "lines": brushed,
                    "both": {
                        "type": "mix",
                        "a": {"node": "smudge"},
                        "b": {"node": "lines"},
                        "factor": 0.5,
                    },
                    "rough": {
                        "type": "ramp",
                        "factor": {"node": "both"},
                        "stops": [
                            [0.2, o.get("roughness_low", 0.22)],
                            [0.8, o.get("roughness_high", 0.42)],
                        ],
                    },
                },
            },
            "normal": {
                "output": "bump",
                "nodes": {
                    "lines": brushed,
                    "bump": {
                        "type": "bump",
                        "height": {"node": "lines"},
                        "depth": 0.0004,
                        "step": 0.0005,
                    },
                },
            },
        },
    }


def _grass(o: dict[str, Any]) -> dict[str, Any]:
    color = o.get("color", "#4c8a2e")
    seed = o["seed"]
    return {
        "params": {"roughness": 0.92, "specular": 0.35},
        "textures": {
            "base_color": {
                "output": "color",
                "nodes": {
                    "patches": {"type": "fbm", "scale": 0.8, "octaves": 5, "seed": seed},
                    "blades": {
                        "type": "voronoi",
                        "scale": 55.0,
                        "randomness": 1.0,
                        "seed": seed + 1,
                    },
                    "base": {
                        "type": "ramp",
                        "factor": {"node": "patches"},
                        "stops": [
                            [0.25, _shade(color, 0.72)],
                            [0.55, color],
                            [0.8, o.get("color2", "#8aa83e")],
                        ],
                        "interpolation": "smooth",
                    },
                    "light": {
                        "type": "ramp",
                        "factor": {"node": "blades", "output": "cell"},
                        "stops": [[0.0, 0.75], [1.0, 1.1]],
                    },
                    "color": {
                        "type": "mix",
                        "a": {"node": "base"},
                        "b": {"node": "light"},
                        "factor": 1.0,
                        "mode": "multiply",
                    },
                },
            },
            "normal": {
                "output": "bump",
                "nodes": {
                    "blades": {
                        "type": "voronoi",
                        "scale": 55.0,
                        "randomness": 1.0,
                        "seed": seed + 1,
                    },
                    "bump": {
                        "type": "bump",
                        "height": {"node": "blades", "output": "edge"},
                        "depth": 0.003,
                        "step": 0.0008,
                    },
                },
            },
        },
    }


def _fabric(o: dict[str, Any]) -> dict[str, Any]:
    color = o.get("color", "#3d5a8a")
    threads = {
        "warp": {
            "type": "stripes",
            "axis": "x",
            "width": 0.5,
            "scale": 90.0,
            "color1": 0.0,
            "color2": 1.0,
        },
        "weft": {
            "type": "stripes",
            "axis": "y",
            "width": 0.5,
            "scale": 90.0,
            "color1": 0.0,
            "color2": 1.0,
        },
        "pick": {"type": "checker", "scale": 90.0, "color1": 0.0, "color2": 1.0},
        "weave": {
            "type": "mix",
            "a": {"node": "warp", "output": "mask"},
            "b": {"node": "weft", "output": "mask"},
            "factor": {"node": "pick", "output": "mask"},
        },
    }
    return {
        "params": {"roughness": 0.95, "specular": 0.3},
        "textures": {
            "base_color": {
                "output": "color",
                "nodes": {
                    **threads,
                    "shade": {
                        "type": "ramp",
                        "factor": {"node": "weave"},
                        "stops": [[0.0, 0.78], [1.0, 1.0]],
                    },
                    "fuzz": {"type": "fbm", "scale": 30.0, "octaves": 3, "seed": o["seed"]},
                    "fiber": {
                        "type": "ramp",
                        "factor": {"node": "fuzz"},
                        "stops": [[0.2, 0.9], [0.8, 1.05]],
                    },
                    "lit": {
                        "type": "math",
                        "op": "multiply",
                        "a": {"node": "shade"},
                        "b": {"node": "fiber"},
                    },
                    "color": {
                        "type": "mix",
                        "a": color,
                        "b": {"node": "lit"},
                        "factor": 1.0,
                        "mode": "multiply",
                    },
                },
            },
            "normal": {
                "output": "bump",
                "nodes": {
                    **threads,
                    "bump": {
                        "type": "bump",
                        "height": {"node": "weave"},
                        "depth": 0.0008,
                        "step": 0.002,
                    },
                },
            },
        },
    }


def _shade(color: Any, factor: float) -> str:
    """A display color made lighter (``factor > 1``) or darker, in linear light."""
    from ..rendering.color import linear_to_srgb, srgb_to_linear

    rgb = srgb_to_linear(parse_color(color, "color")) * factor
    return color_to_hex(tuple(float(c) for c in linear_to_srgb(rgb)))


PRESETS: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str, tuple[str, ...]]] = {
    "stone": (
        _stone,
        "irregular stones with dark joints (voronoi cells, fbm detail, bump)",
        ("mortar_color",),
    ),
    "brick": (
        _brick,
        "running-bond bricks with mortar grooves and grime",
        ("color2", "mortar_color", "brick_width", "brick_height", "mortar"),
    ),
    "wood": (
        _wood,
        "solid wood with growth rings and grain (3D, no UVs needed)",
        ("color2", "axis"),
    ),
    "metal": (
        _metal,
        "brushed metal (metallic, varying roughness)",
        ("roughness_low", "roughness_high"),
    ),
    "grass": (_grass, "grass: patchy greens and blade clumps", ("color2",)),
    "fabric": (_fabric, "woven fabric: warp and weft threads", ()),
}


def preset_names() -> list[str]:
    return list(PRESETS)


def preset_info() -> list[dict[str, Any]]:
    """Name, description and options of every preset."""
    return [
        {"name": name, "description": desc, "options": [*COMMON_OPTIONS, *extra]}
        for name, (_, desc, extra) in PRESETS.items()
    ]


def build_preset(
    library: MaterialLibrary,
    preset: str,
    name: str | None = None,
    *,
    material_id: str | None = None,
    **options: Any,
) -> Material:
    """Create a preset material (and its procedural textures) in ``library``."""
    from .material import PARAM_SPECS

    if preset not in PRESETS:
        raise MaterialError(
            "UNKNOWN_PRESET",
            f"unknown material preset {preset!r}",
            "preset",
            hint="presets: " + ", ".join(PRESETS),
        )
    builder, _, extra = PRESETS[preset]
    params = {k: options.pop(k) for k in list(options) if k in PARAM_SPECS and k not in ("color",)}
    unknown = sorted(set(options) - set(COMMON_OPTIONS) - set(extra))
    if unknown:
        raise MaterialError(
            "INVALID_MATERIAL",
            f"unknown option(s) for the {preset} preset: {', '.join(unknown)}",
            "preset",
            hint="options: " + ", ".join([*COMMON_OPTIONS, *extra]) + ", or material parameters",
        )
    scale = options.pop("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 1e-4 <= scale <= 1e4:
        raise MaterialError(
            "INVALID_MATERIAL", f"scale must be a number in [0.0001, 10000], got {scale!r}", "scale"
        )
    seed = options.pop("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31 - 16:
        raise MaterialError(
            "INVALID_MATERIAL", f"seed must be a non-negative integer, got {seed!r}", "seed"
        )
    coordinates = options.pop("coordinates", "object")
    for key in ("color", "color2", "mortar_color"):
        if key in options:
            options[key] = _hex(options[key], key)
    spec = builder({**options, "seed": seed})
    projection = options.pop("projection", spec.get("projection", "triplanar"))
    name = library._unique_material_name(preset.capitalize()) if name is None else name
    values = {**spec["params"], **params}
    material = library.create_material(name, model="standard", material_id=material_id, **values)
    for slot, graph in spec["textures"].items():
        binding: dict[str, Any] = {"coordinates": coordinates, "scale": 1.0 / scale}
        if coordinates != "uv":
            binding["projection"] = projection
        texture = library.add_procedural_texture(
            library._unique_texture_name(f"{name} {slot}"), graph
        )
        material.set_texture(slot, texture, **binding)
    return material
