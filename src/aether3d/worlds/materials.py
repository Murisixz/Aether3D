"""The scene materials a world draws with, created with the world (one history step).

A world's meshes are drawn with ordinary materials of the scene's library, so they can be
inspected, edited, saved and exported like any other; the world's definition maps its
*layers* to them (``definition["materials"]``, layer -> material id):

* terrain worlds: ``terrain`` (white base colour times the vertex colours, with a world-space
  triplanar detail texture and bump), ``water`` (smooth, specular), ``rock``, ``bark``,
  ``birch_bark``, ``wood``, ``leaves``, ``needles``, ``palm``, ``grass``, ``flowers``,
  ``cactus``, ``mushroom``, ``road``, ``stone``, ``plaster``, ``roof``, ``timber``;
* block worlds: ``block_<texture>`` for every block texture (an embedded 16 x 16 PNG
  sampled with ``nearest`` filtering, mipmaps and repeat wrapping; plants alpha-masked and
  double-sided, lava and glowstone emissive) and ``water``.

Renaming or editing these materials changes how the world looks; deleting one makes the
world draw that layer with a plain grey fallback (``WORLD_MATERIAL_MISSING``).
"""

from __future__ import annotations

from typing import Any

from .blocks import BLOCK_LIST, has_alpha, texture_names, texture_png

DETAIL_GRAPH = {
    "output": "tone",
    "nodes": {
        "coarse": {"type": "fbm", "scale": 0.35, "octaves": 4, "seed": 11},
        "fine": {"type": "fbm", "scale": 2.2, "octaves": 3, "seed": 12},
        "mixed": {"type": "mix", "a": {"node": "coarse"}, "b": {"node": "fine"}, "factor": 0.45},
        "tone": {
            "type": "ramp",
            "factor": {"node": "mixed"},
            "stops": [[0.2, 0.74], [0.5, 0.95], [0.8, 1.08]],
        },
    },
}
"""Grey detail around 1 (multiplies the vertex colours): ground texture at 0.5-3 m."""

BUMP_GRAPH = {
    "output": "bump",
    "nodes": {
        "h": {"type": "fbm", "scale": 1.6, "octaves": 4, "seed": 13},
        "bump": {"type": "bump", "height": {"node": "h"}, "depth": 0.035},
    },
}

TERRAIN_LAYERS: dict[str, dict[str, Any]] = {
    "water": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.04, "specular": 0.7},
    "rock": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.9},
    "bark": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.95},
    "birch_bark": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.8},
    "wood": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.85},
    "leaves": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.8},
    "needles": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.85},
    "palm": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.75},
    "grass": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.9},
    "flowers": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.7},
    "cactus": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.7},
    "mushroom": {"base_color": (1.0, 1.0, 1.0), "roughness": 0.6},
    "road": {"base_color": (0.42, 0.38, 0.32), "roughness": 0.95},
    "stone": {"base_color": (0.6, 0.58, 0.54), "roughness": 0.88},
    "plaster": {"base_color": (0.86, 0.82, 0.74), "roughness": 0.9},
    "roof": {"base_color": (0.45, 0.2, 0.14), "roughness": 0.8},
    "timber": {"base_color": (0.3, 0.2, 0.12), "roughness": 0.85},
}
"""Terrain-world layers besides ``terrain``. Plants and rocks carry their colours as vertex
colours, so their layers are white (a layer's base colour then tints every plant using it);
structures use flat colours."""


def _unique(library: Any, name: str) -> str:
    if name not in library._material_names:
        return name
    k = 2
    while f"{name} {k}" in library._material_names:
        k += 1
    return f"{name} {k}"


def _unique_texture(library: Any, name: str) -> str:
    if name not in library._texture_names:
        return name
    k = 2
    while f"{name} {k}" in library._texture_names:
        k += 1
    return f"{name} {k}"


def create_materials(scene: Any, world_name: str, mode: str, seed: int = 1) -> dict[str, str]:
    """Create the world's materials (inside the caller's history operation); returns the
    layer -> material id map."""
    lib = scene.materials
    out: dict[str, str] = {}
    if mode == "terrain":
        terrain = lib.create_material(
            _unique(lib, f"{world_name} terrain"), roughness=0.93, specular=0.3
        )
        terrain.base_color = (1.0, 1.0, 1.0)
        detail = lib.add_procedural_texture(
            _unique_texture(lib, f"{world_name} ground detail"), DETAIL_GRAPH
        )
        terrain.set_texture("base_color", detail, coordinates="world", projection="triplanar")
        bump = lib.add_procedural_texture(
            _unique_texture(lib, f"{world_name} ground bump"), BUMP_GRAPH
        )
        terrain.set_texture("normal", bump, coordinates="world", projection="triplanar")
        out["terrain"] = terrain.id
        for layer, params in TERRAIN_LAYERS.items():
            m = lib.create_material(_unique(lib, f"{world_name} {layer}"), **params)
            out[layer] = m.id
        return out
    # Block worlds: one material per texture.
    for name in texture_names():
        tex = lib.add_image_texture(
            _unique_texture(lib, f"{world_name} {name}"),
            data=texture_png(name, seed),
            embed=True,
            filter="nearest",
            mipmaps=True,
            wrap="repeat",
        )
        params: dict[str, Any] = {"roughness": 0.9, "specular": 0.25}
        block = next((b for b in BLOCK_LIST if name in (b.top, b.side, b.bottom)), None)
        if block is not None:
            params["roughness"] = block.roughness
            if block.emission > 0.0:
                params["emissive"] = (1.0, 0.8, 0.5)
                params["emissive_strength"] = block.emission
        if has_alpha(name):
            params["alpha_mode"] = "mask"
            params["alpha_cutoff"] = 0.5
            params["double_sided"] = True
        m = lib.create_material(_unique(lib, f"{world_name} {name}"), **params)
        m.base_color = (1.0, 1.0, 1.0)
        m.set_texture("base_color", tex)
        if block is not None and block.emission > 0.0:
            m.set_texture("emissive", tex)
        out[f"block_{name}"] = m.id
    water = lib.create_material(_unique(lib, f"{world_name} water"), roughness=0.05, specular=0.7)
    out["water"] = water.id
    return out
