"""The history's change journal (``History._changes_since``).

Caches derived from a whole scene (the retained render world) ask which parts changed
since the revision they were computed at. The journal must name every changed part --
commits, undo, redo, transactions -- under the revision that changed it, nothing for
changes rolled back or still inside an open operation, and ``None`` (recompute
everything) whenever it cannot reach back far enough.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from aether3d import Mesh, Scene, create_cube, history
from aether3d.history import ALIVE, APPEARANCE, ROOTS, TRANSFORM


def started(scene: Scene) -> int:
    """Start the scene's journal; returns the revision it starts at."""
    assert scene.history._changes_since(scene.revision) == []
    return scene.revision


def parts(scene: Scene, revision: int) -> set[tuple[str, int]]:
    """``(aspect name, id(target))`` of the parts changed after ``revision``."""
    found = scene.history._changes_since(revision)
    assert found is not None
    return {(aspect.name, id(target)) for aspect, target in found}


def part(name: str, target: object) -> tuple[str, int]:
    return (name, id(target))


def test_nothing_is_kept_until_a_cache_asks():
    s = Scene("J")
    cube = s.add_cube("C")
    assert s.history._journal is None
    cube.position = (1.0, 0.0, 0.0)
    assert s.history._journal is None
    assert s.history._changes_since(s.revision - 1) is None  # unknown: recompute
    assert s.history._changes_since(s.revision) == []  # nothing changed since now


def test_commits_undo_and_redo_name_what_they_changed():
    s = Scene("J")
    cube = s.add_cube("C")
    start = started(s)
    cube.position = (1.0, 2.0, 3.0)
    assert parts(s, start) == {part("transform", cube)}
    moved = s.revision
    cube.color = (0.1, 0.2, 0.3)
    assert parts(s, moved) == {part("appearance", cube)}
    colored = s.revision
    s.history.undo()
    assert s.revision == colored + 1
    assert {name for name, _ in parts(s, colored)} == {"appearance"}
    s.history.undo()
    assert parts(s, colored + 1) == {part("transform", cube)}
    s.history.redo()
    assert parts(s, start) >= {part("transform", cube)}
    # Everything since the start, oldest first, a part once per revision.
    found = s.history._changes_since(start)
    assert [a.name for a, _ in found] == [
        "transform",
        "appearance",
        "appearance",
        "transform",
        "transform",
    ]


def test_a_transaction_is_one_revision_and_a_rollback_leaves_nothing():
    s = Scene("J")
    a, b = s.add_cube("A"), s.add_cube("B")
    start = started(s)
    with s.transaction("Both"):
        a.position = (1.0, 0.0, 0.0)
        b.position = (2.0, 0.0, 0.0)
        assert s.history.in_progress
        assert s.history._changes_since(start) == []  # not counted until it ends
    assert s.revision == start + 1
    assert parts(s, start) == {part("transform", a), part("transform", b)}
    after = s.revision
    with pytest.raises(RuntimeError), s.transaction("Fails"):
        a.position = (5.0, 0.0, 0.0)
        raise RuntimeError("rolled back")
    with s.transaction("Explicit rollback") as tx:
        b.position = (7.0, 0.0, 0.0)
        tx.rollback()
    assert s.revision == after and s.history._changes_since(after) == []
    assert a.position == (1.0, 0.0, 0.0) and b.position == (2.0, 0.0, 0.0)


def test_structure_changes_are_named():
    s = Scene("J")
    start = started(s)
    cube = s.add_cube("C")
    names = {name for name, _ in parts(s, start)}
    assert {"alive", "roots"} <= names
    added = s.revision
    s.remove(cube)
    assert part("alive", cube) in parts(s, added)


def test_a_disabled_history_still_journals():
    s = Scene("J")
    cube = s.add_cube("C")
    s.history.enabled = False
    start = started(s)
    cube.position = (0.0, 0.0, 4.0)
    assert parts(s, start) == {part("transform", cube)}


def test_the_journal_is_bounded(monkeypatch):
    monkeypatch.setattr(history, "JOURNAL_ENTRIES", 3)
    s = Scene("J")
    cube = s.add_cube("C")
    start = started(s)
    for k in range(5):
        cube.position = (float(k), 0.0, 0.0)
    assert s.history._changes_since(start) is None  # trimmed: recompute everything
    assert len(s.history._changes_since(s.revision - 3)) == 3
    assert s.history._changes_since(s.revision - 4) is None


def test_a_change_larger_than_the_journal_empties_it(monkeypatch):
    monkeypatch.setattr(history, "JOURNAL_CHANGES", 4)  # the transaction changes 5 parts
    s = Scene("J")
    start = started(s)
    with s.transaction("Many"):
        for k in range(4):
            s.add_cube(f"C{k}")
    assert s.history._changes_since(start) is None
    assert s.history._changes_since(s.revision) == []
    later = s.revision
    s.get_by_name("C0").position = (1.0, 0.0, 0.0)
    assert parts(s, later) == {part("transform", s.get_by_name("C0"))}


def test_the_journal_keeps_no_states():
    """It names the parts that changed; their before and after states (meshes, images)
    belong to the history's steps, which may be dropped."""

    class Tracked(Mesh):
        __slots__ = ("__weakref__",)

    s = Scene("J")
    s.history.enabled = False  # no step keeps the old mesh either
    cube = create_cube(1.0)
    obj = s.add_object("Obj", mesh=Tracked(cube.positions, cube.triangles))
    started(s)
    old = weakref.ref(obj.mesh)
    obj.mesh = create_cube(2.0)
    gc.collect()
    assert old() is None
    assert s.history._journal  # the change is journaled


def test_aspect_objects_are_the_history_aspects():
    s = Scene("J")
    start = started(s)
    cube = s.add_cube("C")
    cube.position = (0.0, 1.0, 0.0)
    cube.color = (1.0, 0.0, 0.0)
    aspects = {aspect for aspect, _ in s.history._changes_since(start)}
    assert {ALIVE, ROOTS, TRANSFORM, APPEARANCE} <= aspects
