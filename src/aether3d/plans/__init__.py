"""Scene plans: build or change a whole scene with one declarative, checked document.

A plan (``{"format": "aether3d.scene-plan", "version": 1, ...}``) declares meshes,
materials, textures, procedural geometry, reusable prefabs and objects -- single ones or
thousands placed by patterns -- plus ordered operations on existing objects, camera and
lighting setup, and read-only spatial queries to answer afterwards. It is data: nothing
in it is executed as code (expressions use the procedural engine's safe whitelist).

* :func:`execute_plan` / ``scene.execute_plan(plan)`` -- check, then build: one undo
  step, all or nothing (opt-in chunks for very large plans).
* :func:`validate_plan` -- check and estimate without building.
* :func:`plan_schema` -- the JSON Schema of the format.

See ``docs/SCENE_PLANS.md``.
"""

from __future__ import annotations

from .executor import (
    ID_NAMESPACE,
    PlanResult,
    execute_plan,
    select_objects,
    validate_plan,
    with_selections,
)
from .limits import PlanLimits
from .model import FORMAT_ID, VERSION, parse_plan
from .schema import plan_schema

__all__ = [
    "FORMAT_ID",
    "ID_NAMESPACE",
    "VERSION",
    "PlanLimits",
    "PlanResult",
    "execute_plan",
    "parse_plan",
    "plan_schema",
    "select_objects",
    "with_selections",
    "validate_plan",
]
