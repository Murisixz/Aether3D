"""Texturing (``docs/TEXTURING.md``): texel rasterization, padding, baking and painting.

UV maps live in :mod:`aether3d.uv`; painted (layered) textures in
:mod:`aether3d.materials.painted`. This package works on them:

* :mod:`.raster` -- which surface point every texel sample stands for (exact coverage);
* :mod:`.dilate` -- padding: island values grown into the empty texels around them;
* :mod:`.bake` -- normal, AO, curvature, thickness, height, ID and material maps from
  high-resolution sources;
* :mod:`.paint` -- brushes and strokes on painted textures (surface, view and UV space),
  fills and gradients;
* :mod:`.transfer` -- carry painted textures onto a remeshed copy (unwrap + bake);
* :mod:`.script` -- ``aether3d.texture-script``: all of the above as safe JSON.
"""

from .bake import MAPS, SLOT_OF_MAP, BakeResult, apply_bake, bake
from .dilate import dilate
from .paint import (
    BRUSH_KINDS,
    Brush,
    PaintView,
    Target,
    fill,
    gradient,
    stroke,
    texel_map,
)
from .raster import Coverage, rasterize_uv
from .script import run_texture_script
from .transfer import transfer_textures

__all__ = [
    "BRUSH_KINDS",
    "MAPS",
    "SLOT_OF_MAP",
    "BakeResult",
    "Brush",
    "Coverage",
    "PaintView",
    "Target",
    "apply_bake",
    "bake",
    "dilate",
    "fill",
    "gradient",
    "rasterize_uv",
    "run_texture_script",
    "stroke",
    "texel_map",
    "transfer_textures",
]
