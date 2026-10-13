"""The procedural system: modifier stacks, transactions, dependencies, caching and objects."""

import numpy as np
import pytest
from procedural_helpers import same_mesh

from aether3d import (
    EditError,
    Expr,
    Mesh,
    ProceduralError,
    Scene,
    ValidationError,
    create_cube,
    inspect_mesh,
)
from aether3d.procedural import Modifier


def volume(mesh):
    return inspect_mesh(mesh).topology["volume"]


@pytest.fixture
def scene():
    scene = Scene("Procedural")
    scene.procedural.add_parameter("size", 2.0, min=0.5, max=10)
    return scene


# --------------------------------------------------------------------------- #
# Modifier stacks
# --------------------------------------------------------------------------- #


def test_stack_ids_names_order_and_enabled(scene):
    geo = scene.add_geometry("Block", "box", width=Expr("size"))
    a = geo.add_modifier("array", count=3)
    b = geo.add_modifier("array", count=2, relative_offset=(0, 1, 0))
    c = geo.add_modifier("mirror", name="Flip", axes="z", index=0)
    assert [m.name for m in geo.modifiers] == ["Flip", "Array", "Array.001"]
    assert len({m.id for m in geo.modifiers}) == 3
    ids = [m.id for m in geo.modifiers]
    geo.move_modifier("Flip", 2)
    assert [m.name for m in geo.modifiers] == ["Array", "Array.001", "Flip"]
    assert sorted(m.id for m in geo.modifiers) == sorted(ids)  # ids are stable
    assert geo.modifier(a.id) is a and geo.modifier("Array.001") is b and geo.modifier(-1) is c
    full = geo.evaluate()
    assert full.triangle_count == 12 * 6 * 2
    b.set(enabled=False)
    assert geo.evaluate().triangle_count == 12 * 3 * 2
    b.set(enabled=True, count=4, name="Rows")
    assert geo.evaluate().triangle_count == 12 * 3 * 4 * 2 and b.name == "Rows"
    geo.remove_modifier("Rows")
    assert [m.name for m in geo.modifiers] == ["Array", "Flip"]
    with pytest.raises(ProceduralError) as info:
        geo.modifier("Rows")
    assert info.value.code == "UNKNOWN_MODIFIER"


def test_evaluate_upto_a_modifier(scene):
    geo = scene.add_geometry("Block", "box")
    geo.add_modifier("array", count=3)
    geo.add_modifier("array", name="Rows", count=2, relative_offset=(0, 1, 0))
    assert geo.evaluate(upto="Array").triangle_count == 36
    assert geo.evaluate(upto=0).triangle_count == 36
    assert geo.evaluate().triangle_count == 72


def test_invalid_modifier_changes_are_rejected_atomically(scene):
    geo = scene.add_geometry("Block", "box")
    arr = geo.add_modifier("array", count=3)
    before = geo.evaluate()
    cases = [
        (lambda: arr.set(count=0), "PARAMETER_OUT_OF_RANGE"),
        (lambda: arr.set(counts=2), "UNKNOWN_PARAMETER"),
        (lambda: arr.set(count="many"), "INVALID_PARAMETER"),
        (lambda: arr.set(count=Expr("nope * 2")), "UNKNOWN_PARAMETER"),
        (lambda: arr.set(enabled="yes"), "INVALID_PARAMETER"),
        (lambda: geo.add_modifier("melt"), "UNKNOWN_MODIFIER"),
        (lambda: geo.add_modifier("array", name="Array"), "DUPLICATE_NAME"),
        (lambda: geo.add_modifier("subdivision", levels=9), "PARAMETER_OUT_OF_RANGE"),
        (lambda: geo.move_modifier("Array", 5), "INVALID_INDEX"),
        (lambda: geo.add_modifier("transform", scale=(0, 1, 1)), "DEGENERATE_TRANSFORM"),
    ]
    revision = scene.procedural.revision
    for change, code in cases:
        with pytest.raises(ProceduralError) as info:
            change()
        assert info.value.code == code
        assert info.value.location is None or info.value.location.startswith(
            ("geometry[", "parameters")
        )
    assert scene.procedural.revision == revision
    assert len(geo.modifiers) == 1 and arr.params == {"count": 3}
    assert geo.evaluate() is before  # cached result, nothing changed


def test_evaluation_failure_rolls_back(scene):
    geo = scene.add_geometry("Sheet", "grid")
    solid = geo.add_modifier("solidify", thickness=0.1)
    obj = scene.add_object("Sheet", geometry=geo)
    good = obj.mesh
    # A bevel wider than the faces fails during evaluation: the whole change is undone.
    with pytest.raises(ProceduralError) as info:
        geo.add_modifier("bevel", width=5.0, limit="all")
    assert info.value.code.startswith("BEVEL")
    assert info.value.location.startswith("geometry[Sheet].modifiers[Bevel]")
    assert [m.name for m in geo.modifiers] == ["Solidify"] and obj.mesh is good
    with pytest.raises(ProceduralError):
        solid.set(thickness=0)
    assert solid.params == {"thickness": 0.1}


def test_base_definition_is_never_modified():
    cube = create_cube(2)
    positions = cube.positions.copy()
    scene = Scene()
    geo = scene.add_geometry("Imported", mesh=cube)
    geo.add_modifier("subdivision", levels=2)
    geo.add_modifier("transform", scale=(2, 2, 2))
    obj = scene.add_object("Blob", geometry=geo)
    assert obj.mesh.triangle_count == 12 * 16
    assert geo.source.mesh is cube and np.array_equal(cube.positions, positions)
    geo.remove_modifier(0)
    geo.remove_modifier(0)
    assert same_mesh(obj.mesh, geo.evaluate()) and obj.mesh.triangle_count == 12


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def test_undo_redo_and_batch(scene):
    geo = scene.add_geometry("Block", "box", width=Expr("size"))
    obj = scene.add_object("Block", geometry=geo)
    v0 = volume(obj.mesh)
    arr = geo.add_modifier("array", count=2, relative_offset=(2, 0, 0))  # separate copies
    with scene.procedural.batch():
        scene.procedural.set_parameter("size", 4.0)
        arr.set(count=3)
    v2 = volume(obj.mesh)
    assert v2 == pytest.approx(v0 * 2 * 3)
    scene.procedural.undo()  # the batch is one step
    assert volume(obj.mesh) == pytest.approx(v0 * 2) and arr.params["count"] == 2
    scene.procedural.undo()
    assert geo.modifiers == [] and volume(obj.mesh) == pytest.approx(v0)
    scene.procedural.redo()
    scene.procedural.redo()
    assert volume(obj.mesh) == pytest.approx(v2)
    assert geo.modifiers[0] is arr  # handles survive undo/redo
    assert not scene.procedural.can_redo
    with pytest.raises(ProceduralError) as info:
        scene.procedural.redo()
    assert info.value.code == "NOTHING_TO_REDO"


def test_undo_restores_object_assignments(scene):
    geo = scene.add_geometry("Block", "box")
    obj = scene.add_object("Thing", mesh=create_cube(1))
    static = obj.mesh
    obj.set_geometry(geo)
    assert obj.is_procedural and obj.mesh is not static
    scene.procedural.undo()
    assert not obj.is_procedural and obj.mesh is static
    obj.set_geometry(geo)
    obj.bake()
    assert not obj.is_procedural and same_mesh(obj.mesh, geo.evaluate())
    scene.procedural.undo()
    assert obj.is_procedural


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def test_geometry_references_overrides_and_after(scene):
    pillar = scene.add_geometry(
        "Pillar", "cylinder", parameters={"h": {"value": 2.0, "min": 0.1}}, segments=8
    )
    pillar.set_source(depth=Expr("h"))
    pillar.add_modifier("transform", translation=Expr("(0, 0, h / 2)"))
    pillar.add_modifier("array", name="Row", count=3, relative_offset=(2, 0, 0))
    tall = scene.add_geometry("Tall", source=pillar, overrides={"h": 5.0})
    single = scene.add_geometry("Single", source=pillar, after="Transform")
    assert tall.evaluate().positions[:, 2].max() == pytest.approx(5.0)
    assert pillar.evaluate().positions[:, 2].max() == pytest.approx(2.0)
    assert single.evaluate().triangle_count * 3 == pillar.evaluate().triangle_count
    assert [g.name for g in scene.procedural.dependencies(tall)] == ["Pillar"]
    assert {g.name for g in scene.procedural.dependents(pillar)} == {"Tall", "Single"}
    # Changing the referenced geometry updates its dependents.
    pillar.set_parameter("h", 3.0)
    assert single.evaluate().positions[:, 2].max() == pytest.approx(3.0)
    assert tall.evaluate().positions[:, 2].max() == pytest.approx(5.0)
    with pytest.raises(ProceduralError) as info:
        scene.add_geometry("Bad", source=pillar, overrides={"height": 1.0})
    assert info.value.code == "UNKNOWN_PARAMETER"
    with pytest.raises(ProceduralError) as info:
        scene.procedural.remove_geometry(pillar)
    assert info.value.code == "GEOMETRY_IN_USE"


def test_dependency_cycles_are_rejected():
    pytest.importorskip("manifold3d")
    scene = Scene()
    a = scene.add_geometry("A", "box")
    b = scene.add_geometry("B", source=a)
    c = scene.add_geometry("C", "box")
    c.add_modifier("boolean", operation="union", operand={"geometry": "B"})
    with pytest.raises(ProceduralError) as info:
        a.add_modifier("boolean", operation="union", operand={"geometry": "C"})
    assert info.value.code == "DEPENDENCY_CYCLE"
    assert info.value.details["cycle"][0] == info.value.details["cycle"][-1]
    with pytest.raises(ProceduralError) as info:
        a.add_modifier("boolean", operation="union", operand={"geometry": "A"})
    assert info.value.code == "DEPENDENCY_CYCLE"
    assert a.modifiers == [] and b.evaluate() is not None


def test_boolean_operand_by_name_with_parameter_overrides():
    pytest.importorskip("manifold3d")
    scene = Scene()
    hole = scene.add_geometry("Hole", "cylinder", parameters={"r": 0.25}, segments=16, depth=4)
    hole.set_source(radius=Expr("r"))
    block = scene.add_geometry("Block", "box", width=2, depth=2, height=2)
    drill = block.add_modifier(
        "boolean", operation="difference", operand={"geometry": "Hole", "parameters": {"r": 0.5}}
    )
    assert drill.params["operand"]["geometry"] == hole.id  # stored by id: renames are safe
    scene.procedural.rename_geometry(hole, "Drill")
    small = 8 - 0.5 * 16 * 0.25**2 * np.sin(2 * np.pi / 16) * 2
    big = 8 - 0.5 * 16 * 0.5**2 * np.sin(2 * np.pi / 16) * 2
    assert volume(block.evaluate()) == pytest.approx(big, rel=1e-6)
    drill.set(operand={"geometry": "Drill"})
    assert volume(block.evaluate()) == pytest.approx(small, rel=1e-6)
    with pytest.raises(ProceduralError) as info:
        drill.set(operand={"geometry": "Missing"})
    assert info.value.code == "UNKNOWN_GEOMETRY"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_cache_reuses_unchanged_prefixes_and_invalidates_dependents(scene):
    proc = scene.procedural
    proc.add_parameter("count", 3, min=1)
    geo = scene.add_geometry("Block", "box", width=Expr("size"))
    geo.add_modifier("subdivision", levels=2)
    geo.add_modifier("array", count=Expr("count"))
    other = scene.add_geometry("Other", "box")
    other.add_modifier("subdivision", levels=3)
    first = geo.evaluate()
    other.evaluate()
    stats = dict(proc.stats.evaluations)
    proc.set_parameter("count", 4)  # only the array step (and the final mesh) recompute
    second = geo.evaluate()
    delta = {k: proc.stats.evaluations.get(k, 0) - stats.get(k, 0) for k in proc.stats.evaluations}
    assert delta == {"primitive": 0, "subdivision": 0, "array": 1, "mesh": 1}
    assert second.triangle_count == first.triangle_count * 4 // 3
    proc.set_parameter("count", 3)  # back to the old key: served from the cache
    assert geo.evaluate() is first
    stats = dict(proc.stats.evaluations)
    proc.set_parameter("size", 3.0)  # source changed: the whole stack recomputes, Other does not
    geo.evaluate()
    other.evaluate()
    delta = {k: proc.stats.evaluations.get(k, 0) - stats.get(k, 0) for k in proc.stats.evaluations}
    assert delta == {"primitive": 1, "subdivision": 1, "array": 1, "mesh": 1}


def test_cache_is_bounded_and_results_stay_correct(scene):
    proc = scene.procedural
    proc.cache_limit_bytes = 200_000
    geo = scene.add_geometry("Ball", "uv_sphere", radius=Expr("size"), segments=32, rings=16)
    for size in (1.0, 2.0, 3.0, 4.0, 5.0):
        proc.set_parameter("size", size)
        r = np.linalg.norm(geo.evaluate().positions, axis=1).max()
        assert r == pytest.approx(size, rel=1e-6)
    assert proc.cache_info["bytes"] <= 200_000 or proc.cache_info["entries"] <= 1
    proc.clear_cache()
    assert proc.cache_info["entries"] == 0
    assert np.linalg.norm(geo.evaluate().positions, axis=1).max() == pytest.approx(5.0, rel=1e-6)


def test_evaluation_is_deterministic():
    def build():
        s = Scene()
        g = s.add_geometry("T", "torus", major_segments=12, minor_segments=8)
        g.add_modifier("subdivision", levels=1)
        g.add_modifier("mirror", axes="x", origin=(3, 0, 0))
        return g.evaluate(), s.procedural.evaluate(g).key

    (m1, k1), (m2, k2) = build(), build()
    assert k1 == k2 and same_mesh(m1, m2)


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #


def test_objects_share_evaluated_meshes(scene):
    geo = scene.add_geometry("Crate", "box")
    objs = [scene.add_object(f"Crate {i}", geometry=geo, position=(i * 2, 0, 0)) for i in range(5)]
    assert all(o.mesh is objs[0].mesh for o in objs)
    assert sorted(geo.users) == sorted(o.id for o in objs)
    report = scene.inspect_geometry()
    assert len(report.to_dict()["meshes"]) == 1


def test_procedural_objects_protect_their_mesh(scene):
    obj = scene.add_procedural("Tower", "cylinder", segments=8)
    assert obj.geometry.name == "Tower" and obj.is_procedural
    with pytest.raises(ProceduralError) as info:
        obj.mesh = create_cube(1)
    assert info.value.code == "PROCEDURAL_OBJECT" and "bake" in info.value.hint
    with pytest.raises(EditError) as info:
        obj.edit_mesh()
    assert info.value.code == "PROCEDURAL_OBJECT"
    # The object name is taken, so the new geometry node gets a suffixed name.
    obj.name = "Keep"
    second = scene.add_procedural("Tower", "box")
    assert second.geometry.name == "Tower.001"
    with pytest.raises(ValidationError):
        scene.add_object("Both", mesh=create_cube(1), geometry=obj.geometry)


def test_bake_and_detach(scene):
    obj = scene.add_procedural("Tower", "cylinder", segments=8)
    twin = scene.add_object("Twin", geometry=obj.geometry)
    evaluated = obj.mesh
    baked = obj.bake()
    assert baked is evaluated and obj.mesh is evaluated and not obj.is_procedural
    assert twin.is_procedural and obj.geometry is None
    obj.mesh = create_cube(1)  # static again: assignable
    twin.set_geometry(None)
    assert twin.mesh is None
    with pytest.raises(ProceduralError):
        twin.set_geometry("Nope")


def test_edit_source_updates_every_user():
    scene = Scene()
    geo = scene.add_geometry("Wall", "box", width=4, depth=1, height=2)
    geo.add_modifier("array", count=2, relative_offset=(0, 0, 1))
    a = scene.add_object("A", geometry=geo)
    b = scene.add_object("B", geometry=geo, position=(0, 5, 0))
    with pytest.raises(ProceduralError) as info:
        geo.edit_source()
    assert info.value.code == "SOURCE_NOT_EDITABLE"
    geo.convert_source_to_mesh()
    assert geo.source.kind == "mesh"
    with geo.edit_source() as edit:
        top = edit.select_faces(normal=((0, 0, 1), 1))
        edit.extrude_faces(top, distance=1.0)
    assert edit.result.changed and set(edit.result.objects) == {a.id, b.id}
    # The base now spans z in [-1, 2]; the array stacks a second copy 3 units higher.
    assert a.mesh is b.mesh and a.mesh.positions[:, 2].max() == pytest.approx(5.0)
    assert geo.evaluate_polygons().validate().ok
    scene.procedural.undo()
    assert geo.source.kind == "mesh" and a.mesh.positions[:, 2].max() == pytest.approx(3.0)


def test_info_and_types_are_json_serializable(scene):
    import json

    geo = scene.add_geometry("Block", "box", width=Expr("size"))
    geo.add_modifier("array", count=2, relative_offset=(2, 0, 0))
    scene.add_object("Block", geometry=geo)
    info = scene.procedural.info(evaluate=True)
    json.dumps(info)
    json.dumps(scene.procedural.types())
    entry = info["geometry"][0]
    assert entry["source"] == {
        "type": "primitive",
        "primitive": "box",
        "params": {"width": {"expr": "size"}},
    }
    assert entry["modifiers"][0]["resolved"]["relative_offset"] == [2, 0, 0]
    assert entry["evaluated"]["mesh"]["triangle_count"] == 24
    validation = scene.procedural.validate()
    json.dumps(validation)
    assert validation["status"] == "ok"


def test_scene_level_add_geometry_validation():
    scene = Scene()
    with pytest.raises(ProceduralError):
        scene.add_geometry("X")  # no source
    with pytest.raises(ProceduralError):
        scene.add_geometry("X", "box", mesh=Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]))
    with pytest.raises(ProceduralError) as info:
        scene.add_geometry("X", "pyramid")
    assert info.value.code == "UNKNOWN_PRIMITIVE"
    scene.add_geometry("X", "box")
    with pytest.raises(ProceduralError) as info:
        scene.add_geometry("X", "box")
    assert info.value.code == "DUPLICATE_NAME"
    assert len(scene.procedural.geometries) == 1


def test_modifier_handle_type():
    scene = Scene()
    m = scene.add_geometry("X", "box").add_modifier("array")
    assert isinstance(m, Modifier) and m.resolved()["count"] == 2
    assert m.info()["type"] == "array"


def test_undo_never_touches_unrelated_static_meshes():
    scene = Scene()
    static = scene.add_object("Static", mesh=create_cube(1))
    geo = scene.add_geometry("Block", "box")
    replacement = create_cube(3)
    static.mesh = replacement  # an edit outside the procedural history
    scene.procedural.undo()  # undoes add_geometry only
    assert static.mesh is replacement and scene.procedural.geometries == []
    scene.procedural.redo()
    assert scene.procedural.geometry("Block") is geo and static.mesh is replacement


def test_object_assignment_is_undoable_and_history_conflicts_are_refused():
    scene = Scene()
    geo = scene.add_geometry("Block", "box")
    obj = scene.add_object("Block", geometry=geo)
    scene.procedural.undo()  # undoes the assignment made by add_object
    assert not obj.is_procedural and obj.mesh is None
    scene.procedural.redo()
    assert obj.geometry is geo
    # Simulate an object that refers to the node outside the history (e.g. a tool that
    # wrote the reference directly): undoing the node's creation must not dangle it.
    other = scene.add_object("Other")
    other._geometry = geo.id
    scene.procedural.undo()  # the assignment of "Block"
    revision = scene.procedural.revision
    with pytest.raises(ProceduralError) as info:
        scene.procedural.undo()  # would remove "Block" while "Other" uses it
    assert info.value.code == "HISTORY_CONFLICT"
    assert scene.procedural.geometry("Block") is geo and scene.procedural.revision == revision
    assert other.mesh is not None


def test_unexpected_modifier_failures_are_reported_and_rolled_back(monkeypatch):
    from aether3d.procedural.modifiers import MODIFIER_TYPES

    scene = Scene()
    geo = scene.add_geometry("Block", "box")

    def broken(self, mesh, values, ctx):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(type(MODIFIER_TYPES["mirror"]), "apply", broken)
    with pytest.raises(ProceduralError) as info:
        geo.add_modifier("mirror")
    assert info.value.code == "MODIFIER_FAILED"
    assert info.value.message == "RuntimeError: backend exploded"
    assert info.value.location == "geometry[Block].modifiers[Mirror]"
    assert geo.modifiers == []
