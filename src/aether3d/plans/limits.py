"""Resource limits of scene plans (checked before anything is built)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from ..errors import PlanError


@dataclass(frozen=True)
class PlanLimits:
    """Budgets a plan must fit in; every one is checked before the scene changes.

    The defaults suit an interactive session on a workstation. A server that takes
    plans from agents (``aether3d mcp serve``) can lower or raise them.
    """

    max_plan_bytes: int = 16 * 2**20
    """Size of the plan document as JSON."""
    max_objects: int = 100_000
    """Objects one plan may create (every pattern instance and child counts)."""
    max_scene_objects: int = 1_000_000
    """Objects the scene may hold after the plan."""
    max_pattern_count: int = 100_000
    """Instances of one pattern."""
    max_depth: int = 32
    """Nesting of children and prefabs."""
    max_operations: int = 10_000
    """Entries of ``operations``."""
    max_queries: int = 100
    """Entries of ``queries``."""
    max_segments: int = 512
    """Segments and rings of primitive meshes (the engine itself allows 1024)."""
    max_inline_vertices: int = 200_000
    """Vertices of one inline mesh (``{"data": ...}``)."""
    max_triangles: int = 50_000_000
    """Triangles of all new objects (instances count once each, shared or not)."""
    max_new_mesh_bytes: int = 1 * 2**30
    """Memory of the new meshes the plan builds (shared meshes count once)."""
    max_materials: int = 1_000
    """Materials and textures one plan may create."""

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlanError(
                    "INVALID_VALUE", f"limit {f.name} must be a non-negative integer, got {value!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanLimits:
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise PlanError("INVALID_VALUE", f"unknown limit(s): {', '.join(unknown)}")
        return cls(**data)
