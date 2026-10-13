"""Programmatic polygon mesh editing (see docs/EDITING.md)."""

from .core import CornerLayer, OperationResult, Selection
from .mesh import EditableMesh
from .session import CommitResult, EditSession
from .validation import TopologyReport

__all__ = [
    "CommitResult",
    "CornerLayer",
    "EditSession",
    "EditableMesh",
    "OperationResult",
    "Selection",
    "TopologyReport",
]
