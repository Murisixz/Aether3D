"""UV mapping (``docs/TEXTURING.md``): unwrapping, projections, island tools, packing and
diagnostics.

Quick start::

    from aether3d import uv
    from aether3d.primitives import create_sphere

    result = uv.unwrap(create_sphere(1.0, 64, 32), resolution=1024, padding=4)
    mesh = result.mesh                     # the same surface with new UVs
    print(result.report["charts"], result.report["stretch_l2"])
    print(uv.uv_report(mesh)["flipped_triangles"])   # 0

Modules: :mod:`.topology` (welded surface, seams), :mod:`.charts` (segmentation and
splitting), :mod:`.param` (LSCM, Tutte, SLIM), :mod:`.sparse` (LDL^T solver, native when
installed), :mod:`.pack` (atlas packing), :mod:`.projection`, :mod:`.ops` (island tools)
and :mod:`.report` (diagnostics).
"""

from .ops import (
    align_islands,
    normalize_islands,
    pack_islands,
    set_texel_density,
    stitch,
    transform_islands,
)
from .projection import KINDS as PROJECTIONS
from .projection import project
from .report import uv_report
from .unwrap import UnwrapResult, unwrap

__all__ = [
    "PROJECTIONS",
    "UnwrapResult",
    "align_islands",
    "normalize_islands",
    "pack_islands",
    "project",
    "set_texel_density",
    "stitch",
    "transform_islands",
    "unwrap",
    "uv_report",
]
