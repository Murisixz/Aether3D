"""The boolean modifier with the Manifold backend (skipped when manifold3d is absent)."""

import math

import numpy as np
import pytest
from procedural_helpers import apply, codes, is_solid, prim, volume

from aether3d import ProceduralError
from aether3d.procedural.modifiers import reverse_corners

pytest.importorskip("manifold3d")


def boolean(op, a, b, **params):
    return apply("boolean", a, operand_mesh=b, operation=op, operand={"geometry": "B"}, **params)


def _rewound(mesh):
    return reverse_corners(mesh)


@pytest.mark.parametrize(
    ("op", "expected"), [("union", 15), ("difference", 7), ("intersection", 1)]
)
def test_boolean_volumes_are_exact(op, expected):
    a = prim("box", width=2, depth=2, height=2)
    out, diags = boolean(op, a, a, operand_translation=(1, 1, 1))
    assert volume(out) == pytest.approx(expected, rel=1e-6)
    assert is_solid(out) and diags == []
    assert np.all(out.face_sizes == 3)


def test_boolean_operand_transform_and_curved_inputs():
    cyl = prim("cylinder", radius=0.5, depth=4, segments=24)
    box = prim("box", width=2, depth=2, height=2)
    # A rotated cylinder drilled through the box (axis along X).
    out, _ = boolean("difference", box, cyl, operand_rotation=(0, 90, 0))
    hole = 0.5 * 24 * 0.25 * math.sin(2 * math.pi / 24) * 2
    assert volume(out) == pytest.approx(8 - hole, rel=1e-6)
    assert is_solid(out)
    assert out.validate().summary["euler_characteristic"] == 0  # one through-hole (genus 1)


def test_boolean_carries_uvs_and_shading_flags():
    a = prim("box", width=2, depth=2, height=2)
    b = prim("uv_sphere", radius=1.2, segments=16, rings=8)
    out, _ = boolean("difference", a, b)
    assert out.uvs is not None and np.all(np.isfinite(out.uvs))
    # Faces from the sphere are smooth, faces from the box flat.
    assert out.smooth.any() and not out.smooth.all()
    no_uv = b.with_(uvs=None)
    out, diags = boolean("union", a, no_uv)
    assert out.uvs is None and "UVS_DROPPED" in codes(diags)


def test_boolean_empty_result_and_disjoint_union():
    a = prim("box")
    out, diags = boolean("intersection", a, a, operand_translation=(5, 0, 0))
    assert out.face_count == 0 and "BOOLEAN_EMPTY_RESULT" in codes(diags)
    out, _ = boolean("union", a, a, operand_translation=(5, 0, 0))
    assert volume(out) == pytest.approx(2 * volume(a))


@pytest.mark.parametrize("which", ["input", "operand"])
def test_boolean_rejects_non_solids(which):
    open_mesh = prim("grid")
    inside_out, _ = apply("solidify", prim("grid"), thickness=0.2)
    inside_out = inside_out.with_(face_vertices=inside_out.face_vertices[_rewound(inside_out)])
    for bad in (open_mesh, inside_out):
        args = (bad, prim("box")) if which == "input" else (prim("box"), bad)
        with pytest.raises(ProceduralError) as info:
            boolean("union", *args)
        assert info.value.code == "BOOLEAN_INPUT_NOT_SOLID"
        assert info.value.location.endswith(".operand") == (which == "operand")
