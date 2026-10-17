"""Assemblies: mates solved over rigid motions, degrees of freedom, conflicts, nesting,
instanced display and clash detection by exact B-rep intersection."""

from __future__ import annotations

import numpy as np
import pytest

import aether3d
from aether3d.cad.assembly import clashes
from aether3d.errors import CadError

pytest.importorskip("manifold3d")


def parts(doc):
    with doc.batch("Parts"):
        sk = doc.sketch("Plate profile")
        sk.rectangle((0, 0), (80, 40))
        plate = doc.part("Plate")
        plate.extrude("Plate profile", 10, id="base")
        plate.hole("base:end", [[20, 20, 10], [60, 20, 10]], 8, id="h")
        pin = doc.part("Pin")
        pin.add_feature(
            {
                "type": "cylinder",
                "id": "shaft",
                "radius": 4,
                "height": 30,
                "frame": {"origin": [0, 0, 15]},
            }
        )
        pin.add_feature(
            {
                "type": "cylinder",
                "id": "head",
                "radius": 7,
                "height": 4,
                "frame": {"origin": [0, 0, 32]},
                "operation": "add",
            }
        )
    return plate, pin


def mates(extra=()):
    return [
        {
            "id": "c1",
            "type": "concentric",
            "a": {"component": "plate", "face": "h.0:rev/wall.0"},
            "b": {"component": "pin1", "face": "shaft:side/wall.0"},
        },
        {
            "id": "f1",
            "type": "coincident",
            "a": {"component": "plate", "face": "base:end"},
            "b": {"component": "pin1", "face": "head:start"},
        },
        {
            "id": "c2",
            "type": "concentric",
            "a": {"component": "plate", "face": "h.1:rev/wall.0"},
            "b": {"component": "pin2", "face": "shaft:side/wall.0"},
        },
        *extra,
    ]


def make(doc, extra=()):
    return doc.add(
        "assembly",
        "Hinge",
        {
            "components": [
                {"id": "plate", "source": "Plate", "fixed": True},
                {"id": "pin1", "source": "Pin", "placement": {"origin": [25, 15, 30]}},
                {
                    "id": "pin2",
                    "source": "Pin",
                    "placement": {"origin": [55, 25, 40], "z": [0.1, 0, 1]},
                },
            ],
            "mates": mates(extra),
        },
    )


def test_pins_are_placed_by_their_mates_and_keep_free_motions():
    scene = aether3d.Scene("asm")
    doc = scene.cad
    parts(doc)
    asm = make(doc)
    info = doc.info(asm)
    assert info["status"] == "under_constrained"
    assert info["dof"] == {"plate": 0, "pin1": 1, "pin2": 2}  # spin; spin and slide
    np.testing.assert_allclose(info["components"]["pin1"]["origin"], [20, 20, -20], atol=1e-9)
    np.testing.assert_allclose(info["components"]["pin2"]["origin"][:2], [60, 20], atol=1e-9)
    np.testing.assert_allclose(info["components"]["pin2"]["z"], [0, 0, 1], atol=1e-12)
    assert {r["mate"] for r in info["redundant"]} == {"f1"}  # the axis is held twice, consistently


def test_conflicting_mates_are_refused_and_blamed():
    scene = aether3d.Scene("asm")
    doc = scene.cad
    parts(doc)
    asm = make(doc)
    bad = {
        "id": "bad",
        "type": "coincident",
        "a": {"component": "plate", "face": "base:start"},
        "b": {"component": "pin1", "face": "head:start"},
    }
    d = doc.item(asm).definition
    d["mates"].append(bad)
    with pytest.raises(CadError) as e:
        doc.set_definition(asm, d)
    assert e.value.code == "ASSEMBLY_CONFLICT"
    assert any(c["mate"] == "bad" for c in e.value.details["conflicts"])


def test_instances_share_meshes_and_follow_part_edits():
    scene = aether3d.Scene("asm")
    doc = scene.cad
    plate, pin = parts(doc)
    asm = make(doc)
    obj = doc.show(asm)
    kids = {c.name.split("/")[-1]: c for c in obj.children}
    assert kids["pin1"].mesh is kids["pin2"].mesh
    plate.set("h", diameter=10)  # holes and pins no longer match radii: still concentric
    assert doc.info(asm)["dof"]["pin1"] == 1


def test_clashes_by_exact_intersection():
    scene = aether3d.Scene("asm")
    doc = scene.cad
    parts(doc)
    asm = make(doc)
    out = clashes(doc, asm)
    assert out == [{"a": "plate", "b": "pin1", "kind": "contact_or_near", "volume": 0.0}]
    # Fatter pins interfere: the overlap is a ring of known volume.
    doc.part("Pin").set("shaft", radius=5)
    out = clashes(doc, asm)
    hit = next(c for c in out if c["b"] == "pin1")
    assert hit["kind"] == "interference"
    assert hit["volume"] == pytest.approx(np.pi * (25 - 16) * 10, rel=1e-9)


def test_nested_assemblies_and_cycles():
    scene = aether3d.Scene("asm")
    doc = scene.cad
    parts(doc)
    make(doc)
    top = doc.add(
        "assembly",
        "Two hinges",
        {
            "components": [
                {"id": "a", "source": "Hinge", "fixed": True},
                {"id": "b", "source": "Hinge", "fixed": True, "placement": {"origin": [0, 100, 0]}},
            ],
            "mates": [],
        },
    )
    assert len(doc.result(top).leaves) == 6
    d = doc.item("Hinge").definition
    d["components"].append({"id": "loop", "source": "Two hinges"})
    with pytest.raises(CadError) as e:
        doc.set_definition("Hinge", d)
    assert e.value.code == "DEPENDENCY_CYCLE"
