"""Sculpting: brushes, strokes, masks, sessions and the sculpt mesh (docs/MODELING.md)."""

from .brushes import FALLOFFS, KINDS, Brush
from .mesh import SculptMesh, SculptResult
from .script import SculptScriptError, check_script, run_sculpt_script
from .session import SculptCommit, SculptSession, SourceSculptSession
from .stroke import Stroke
from .validate import TIERS, MeshReport, check_mesh

__all__ = [
    "FALLOFFS",
    "KINDS",
    "TIERS",
    "Brush",
    "MeshReport",
    "SculptCommit",
    "SculptMesh",
    "SculptResult",
    "SculptScriptError",
    "SculptSession",
    "SourceSculptSession",
    "Stroke",
    "check_mesh",
    "check_script",
    "run_sculpt_script",
]
