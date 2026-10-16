"""The bulk creation path (``Scene._create_many``) and automatic object names.

``_create_many`` is what scene plans build scenes with. It must give exactly what the
per-object API gives -- the same project document, the same history behaviour -- while
costing O(n) for n objects. Automatic names must stay "the first free ``base.NNN``"
however objects come and go, now that ``_unique_name`` remembers where it stopped.
"""

from __future__ import annotations

import pytest
from timing import seconds

from aether3d import Scene, create_cube, create_sphere
from aether3d.errors import NameConflictError, ObjectNotFoundError, ValidationError
from aether3d.history import check_scene
from aether3d.math3d import quat_from_euler
from aether3d.rendering.settings import Appearance
from aether3d.scene import NewObject


def _doc(scene: Scene) -> dict:
    return scene.to_dict()


# --------------------------------------------------------------------------- #
# Automatic names
# --------------------------------------------------------------------------- #


def _reference_name(scene: Scene, base: str) -> str:
    """The documented rule, computed by brute force: base, else the first free base.NNN."""
    taken = {o.name for o in scene}
    if base not in taken:
        return base
    k = 1
    while f"{base}.{k:03d}" in taken:
        k += 1
    return f"{base}.{k:03d}"


def test_automatic_names_are_the_first_free_suffix_through_removals_renames_and_undo():
    scene = Scene("Names")
    cubes = [scene.add_cube() for _ in range(12)]
    assert [c.name for c in cubes[:3]] == ["Cube", "Cube.001", "Cube.002"]
    assert cubes[-1].name == "Cube.011"
    scene.remove(cubes[4])  # frees Cube.004
    assert scene.unique_name("Cube") == _reference_name(scene, "Cube") == "Cube.004"
    assert scene.add_cube().name == "Cube.004"
    assert scene.unique_name("Cube") == "Cube.012"
    cubes[2].name = "Renamed"  # frees Cube.002
    assert scene.unique_name("Cube") == "Cube.002"
    scene.add_cube(name="Cube.002")
    scene.history.undo()  # the explicit Cube.002 is gone again
    assert scene.unique_name("Cube") == _reference_name(scene, "Cube") == "Cube.002"
    scene.history.undo()  # the rename is undone: Cube.002 is taken again
    assert scene.unique_name("Cube") == _reference_name(scene, "Cube") == "Cube.012"
    scene.history.undo(2)  # Cube.004 (re-added) is removed, then the removed one is back
    assert scene.unique_name("Cube") == _reference_name(scene, "Cube")


def test_automatic_names_match_the_rule_in_a_random_sequence_of_changes():
    import random

    rng = random.Random(7)
    scene = Scene("Random names")
    bases = ("Cube", "Cube.001", "Wall", "Wall.010", "Object", "Mesh")
    for _ in range(400):
        action = rng.random()
        objects = scene.objects
        if action < 0.55 or not objects:
            base = rng.choice(bases)
            name = scene.unique_name(base)
            assert name == _reference_name(scene, base)
            parent = rng.choice(objects) if objects and rng.random() < 0.3 else None
            scene.add_object(name, parent=parent)
        elif action < 0.7:
            scene.remove(rng.choice(objects), recursive=rng.random() < 0.5)
        elif action < 0.8:
            rng.choice(objects).name = scene.unique_name(rng.choice(bases))
        elif action < 0.9 and scene.history.can_undo:
            scene.history.undo()
        elif scene.history.can_redo:
            scene.history.redo()
        for base in bases:
            assert scene.unique_name(base) == _reference_name(scene, base)
    check_scene(scene)


def test_asking_for_a_free_name_does_not_rescan_the_taken_ones():
    def calls(n: int) -> float:
        scene = Scene("Many")
        names = ["Crate"] + [f"Crate.{k:03d}" for k in range(1, n)]
        scene._create_many([NewObject(name) for name in names])
        assert scene.unique_name("Crate") == f"Crate.{n:03d}"

        def ask() -> None:
            for _ in range(2_000):
                scene.unique_name("Crate")

        return seconds(ask)

    small, large = min(calls(1_000) for _ in range(2)), min(calls(16_000) for _ in range(2))
    # A scan from .001 would make every call 16x slower in the larger scene.
    assert large < 4 * small + 0.01, (small, large)


# --------------------------------------------------------------------------- #
# _create_many
# --------------------------------------------------------------------------- #


def _same_scene_by_api(records: list[NewObject], meshes: dict) -> Scene:
    """What _create_many must produce, built one object at a time with the public API."""
    scene = Scene("Bulk")
    created = []
    for rec in records:
        parent = created[rec.parent] if isinstance(rec.parent, int) else rec.parent
        obj = scene.add_object(
            rec.name,
            mesh=rec.mesh,
            parent=parent,
            position=rec.position,
            rotation=rec.rotation,
            scale=rec.scale,
            object_id=rec.object_id,
        )
        if rec.appearance is not None:
            obj.appearance = rec.appearance
        obj.visible = rec.visible
        obj.tags = rec.tags
        if rec.metadata is not None:
            obj.metadata = rec.metadata
        created.append(obj)
    return scene


def _records(mesh_a, mesh_b) -> list[NewObject]:
    red = Appearance(color="#c0392b", specular=0.4, shininess=64)
    return [
        NewObject("Group", object_id="id-group", position=(1.0, 2.0, 0.0)),
        NewObject(
            "Crate",
            object_id="id-crate",
            parent=0,
            mesh=mesh_a,
            position=(0.5, 0.0, 0.25),
            rotation=quat_from_euler((0.0, 0.0, 30.0)),
            scale=(1.0, 2.0, 0.5),
            appearance=red,
            tags=["prop", "crate"],
            metadata={"mass": 12.5, "label": "Ünïcødé"},
        ),
        NewObject("Ball", object_id="id-ball", parent=0, mesh=mesh_b, visible=False),
        NewObject("Crate 2", object_id="id-crate-2", parent=1, mesh=mesh_a, tags=("prop",)),
        NewObject("Loose", object_id="id-loose", mesh=mesh_b, position=(-3.0, 0.0, 1.0)),
    ]


def test_create_many_gives_the_document_of_the_per_object_api():
    a, b = create_cube(1.0), create_sphere(0.5, 12, 6)
    records = _records(a, b)
    reference = _same_scene_by_api(records, {})
    scene = Scene("Bulk")
    created = scene._create_many(records, "Bulk build")
    assert [o.name for o in created] == ["Group", "Crate", "Ball", "Crate 2", "Loose"]
    assert _doc(scene) == _doc(reference)
    assert created[1].mesh is created[3].mesh is a  # shared, never copied
    assert created[3].parent is created[1] and created[1].parent is created[0]
    assert created[1].tags == ("crate", "prop") and created[1].metadata["label"] == "Ünïcødé"
    assert created[1].appearance == Appearance(color="#c0392b", specular=0.4, shininess=64)
    assert created[1].appearance is not records[1].appearance  # a copy owned by the object
    check_scene(scene)


def test_create_many_is_one_undoable_step():
    a, b = create_cube(1.0), create_sphere(0.5, 12, 6)
    scene = Scene("Bulk")
    scene.add_cube(name="Existing")
    before = _doc(scene)
    created = scene._create_many(_records(a, b), "Bulk build")
    after = _doc(scene)
    assert [e.label for e in scene.history.entries] == ["Add object 'Existing'", "Bulk build"]
    scene.history.undo()
    assert _doc(scene) == before and len(scene) == 1
    assert all(o.scene is None for o in created)
    scene.history.redo()
    assert _doc(scene) == after
    assert scene.get("id-crate") is created[1]  # the same Python objects come back
    created[1].color = "#00ff00"  # and they keep recording
    scene.history.undo()
    assert created[1].color == Appearance(color="#c0392b").color


def test_create_many_inside_a_transaction_rolls_back_with_it():
    scene = Scene("Bulk")
    with pytest.raises(RuntimeError), scene.transaction("Plan"):
        scene._create_many([NewObject(f"C{i}", mesh=create_cube(1.0)) for i in range(5)])
        raise RuntimeError("later step failed")
    assert len(scene) == 0 and not scene.history.can_undo


@pytest.mark.parametrize(
    "bad, error, message",
    [
        (lambda recs: setattr(recs[2], "name", "Existing"), NameConflictError, "already exists"),
        (lambda recs: setattr(recs[3], "name", "Crate"), NameConflictError, "two new objects"),
        (lambda recs: setattr(recs[2], "visible", 1), ValidationError, "True or False"),
        (lambda recs: setattr(recs[2], "name", "  "), ValidationError, "non-empty"),
        (lambda recs: setattr(recs[4], "object_id", "id-crate"), ValidationError, "already"),
        (lambda recs: setattr(recs[1], "parent", 3), ValidationError, "earlier record"),
        (lambda recs: setattr(recs[1], "parent", -1), ValidationError, "earlier record"),
        (lambda recs: setattr(recs[1], "parent", "no-such-id"), ObjectNotFoundError, "no object"),
        (lambda recs: setattr(recs[1], "scale", (1.0, float("nan"), 1.0)), ValidationError, "fin"),
        (lambda recs: setattr(recs[1], "position", (0, 0, 0)), ValidationError, "finite floats"),
        (lambda recs: setattr(recs[1], "rotation", (0.0, 0.0, 0.0, 2.0)), ValidationError, "unit"),
        (lambda recs: setattr(recs[1], "material", "nope"), ValidationError, "no material"),
        (lambda recs: setattr(recs[1], "tags", ["ok", " bad"]), ValidationError, "tag"),
        (lambda recs: setattr(recs[1], "metadata", {"x": float("inf")}), ValidationError, "finite"),
        (lambda recs: setattr(recs[1], "mesh", "cube"), ValidationError, "Mesh"),
    ],
)
def test_create_many_checks_everything_before_changing_the_scene(bad, error, message):
    scene = Scene("Bulk")
    scene.add_cube(name="Existing")
    before, revision = _doc(scene), scene.revision
    records = _records(create_cube(1.0), create_sphere(0.5, 12, 6))
    bad(records)
    with pytest.raises(error, match=message):
        scene._create_many(records)
    assert _doc(scene) == before and scene.revision == revision and len(scene.history) == 1


def test_create_many_attaches_to_existing_parents_in_order():
    scene = Scene("Bulk")
    base = scene.add_cube(name="Base")
    scene.add_cube(name="First child", parent=base)
    mesh = create_cube(0.5)
    created = scene._create_many(
        [NewObject(f"Brick {i}", parent=base, mesh=mesh) for i in range(3)]
        + [NewObject("Top", parent=1, mesh=mesh)]
    )
    assert [c.name for c in base.children] == ["First child", "Brick 0", "Brick 1", "Brick 2"]
    assert created[3].parent is created[1]
    scene.history.undo()
    assert [c.name for c in base.children] == ["First child"]
    check_scene(scene)


def test_create_many_scales_linearly():
    mesh = create_cube(1.0)

    def build(n: int) -> float:
        scene = Scene("Bulk")
        records = [NewObject(f"C{i}", mesh=mesh, position=(float(i), 0.0, 0.0)) for i in range(n)]
        return seconds(lambda: scene._create_many(records))

    build(500)  # warm up
    small, large = min(build(2_000) for _ in range(2)), min(build(16_000) for _ in range(2))
    assert large < 16 * small + 0.05, (small, large)  # 8x the objects: well under 64x


def test_batched_ids_are_random_version_4_uuids_in_their_format():
    import uuid

    from aether3d.scene import new_object_id, new_object_ids

    ids = new_object_ids(5000)
    assert len(set(ids)) == 5000 and new_object_ids(0) == []
    for text in ids[:500] + [new_object_id()]:
        parsed = uuid.UUID(text)
        assert parsed.version == 4 and parsed.variant == uuid.RFC_4122
        assert str(parsed) == text  # lowercase, 8-4-4-4-12


def test_batch_transform_checks_accept_exactly_what_the_per_record_checks_accept():
    import math
    import random
    from collections import namedtuple

    import numpy as np

    from aether3d.errors import ValidationError
    from aether3d.scene import NewObject, _finite3, _valid_transforms

    def per_record(records) -> bool:
        try:
            for rec in records:
                _finite3(rec.position, "p")
                _finite3(rec.scale, "s")
                q = rec.rotation
                if (
                    not isinstance(q, tuple)
                    or len(q) != 4
                    or not all(type(v) is float and math.isfinite(v) for v in q)
                    or abs(sum(v * v for v in q) - 1.0) > 1e-9
                ):
                    return False
        except ValidationError:
            return False
        return True

    rng = random.Random(4)
    Vec = namedtuple("Vec", "x y z")
    near = math.sqrt(1.0 + 1e-9)  # a quaternion right at the tolerance
    oddities = [
        ("position", [0.0, 0.0, 0.0]),
        ("position", (0.0, 0.0)),
        ("position", (0, 0.0, 0.0)),
        ("position", (np.float64(1.0), 0.0, 0.0)),
        ("position", (math.nan, 0.0, 0.0)),
        ("scale", (math.inf, 1.0, 1.0)),
        ("scale", Vec(1.0, 1.0, 1.0)),  # a tuple subclass: accepted one by one
        ("rotation", (0.0, 0.0, 0.0, 1.0 + 2e-9)),
        ("rotation", (0.0, 0.0, 0.0, near)),
        ("rotation", (0.0, 0.0, 0.0, 1.0, 0.0)),
    ]
    for trial in range(200):
        records = []
        for k in range(rng.randrange(1, 6)):
            q = [rng.gauss(0, 1) for _ in range(4)]
            n = math.sqrt(sum(v * v for v in q))
            records.append(
                NewObject(
                    f"o{k}",
                    position=tuple(rng.uniform(-9, 9) for _ in range(3)),
                    rotation=tuple(v / n for v in q),
                    scale=(rng.uniform(0.1, 3), 1.0, -2.0),
                )
            )
        if trial % 2:
            field, value = rng.choice(oddities)
            setattr(records[rng.randrange(len(records))], field, value)
        fast = _valid_transforms(records)
        assert not fast or per_record(records), records  # never accepts an invalid batch
        assert fast == per_record(records) or isinstance(
            next((r.scale for r in records if type(r.scale) is not tuple), None), Vec
        )
