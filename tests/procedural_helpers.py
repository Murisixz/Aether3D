"""Shared helpers for the procedural geometry tests."""

from __future__ import annotations

import numpy as np

from aether3d import inspect_mesh
from aether3d.procedural.modifiers import MODIFIER_TYPES, ModifierContext
from aether3d.procedural.params import validate_values
from aether3d.procedural.primitives import PRIMITIVES


def prim(name: str, **values):
    """Generate a primitive PolyMesh (schema defaults filled in)."""
    t = PRIMITIVES[name]
    return t.generate(validate_values(t.params, values, name), name)


def apply(name: str, mesh, *, operand_mesh=None, **params):
    """Run one modifier directly; returns (PolyMesh, diagnostics).

    ``operand_mesh`` is what a boolean's operand reference evaluates to.
    """
    t = MODIFIER_TYPES[name]
    values = validate_values(t.params, params, name)
    t.check(values, name)
    return t.apply(mesh, values, ModifierContext(name, lambda spec, loc: operand_mesh))


def topology(poly) -> dict:
    mesh, _ = poly.to_mesh()
    return inspect_mesh(mesh).topology


def volume(poly) -> float:
    return topology(poly)["volume"]


def is_solid(poly) -> bool:
    t = topology(poly)
    return bool(
        t["closed"]
        and t["edge_manifold"]
        and t["vertex_manifold"]
        and t["consistently_oriented"]
        and (t["volume"] or 0) > 0
    )


def codes(diagnostics) -> list[str]:
    return [d.code for d in diagnostics]


def same_mesh(a, b) -> bool:
    return (
        np.array_equal(a.positions, b.positions)
        and np.array_equal(a.triangles, b.triangles)
        and ((a.uvs is None and b.uvs is None) or np.array_equal(a.uvs, b.uvs))
        and ((a.normals is None and b.normals is None) or np.array_equal(a.normals, b.normals))
    )
