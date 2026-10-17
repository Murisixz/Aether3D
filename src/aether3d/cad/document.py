"""The CAD document: parametric items, their dependencies, regeneration and display.

A scene owns one :class:`CadDocument` (``scene.cad``). It holds **items** -- parameters,
frames, sketches, parts, assemblies, building elements, drawings -- each with

* a stable id (UUID4) and a unique name;
* a **definition**: plain JSON values, validated against the item kind's rules and never
  mutated in place (a change replaces it), so the history keeps definitions by reference;
* **references** to other items (computed from the definition), forming a directed
  acyclic graph -- a change that would close a cycle is refused (``DEPENDENCY_CYCLE``);
* optional **links**: scene objects that show the item's display geometry.

**Regeneration is incremental and content-addressed.** Every item's result is keyed by a
SHA-256 of its kind, definition, the document settings and the keys of the results it
reads; results live in a bounded cache. A change re-evaluates the changed items and the
items downstream of them only (a reverse-dependency index), in dependency order; an
item whose key did not change is not evaluated again and does not dirty its dependents.

**Transactions are atomic.** Every public mutation is one step of the scene's history
(``with doc.batch("label"):`` groups several): definitions change, the affected items
regenerate, linked objects receive their new meshes (as ordinary ``DATA`` changes, so
renderers, spatial queries and glTF see plain meshes and update incrementally). If any
affected item fails to evaluate -- an over-constrained sketch, a lost reference, an
invalid boolean -- the error names the item and the whole step is rolled back.

Linked objects are *driven*: their meshes cannot be edited directly (``CAD_DRIVEN``);
``convert_to_mesh`` detaches them deliberately, reporting the parametric data that the
object no longer follows.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import CadError
from .tolerance import TessellationTolerance
from .units import ANGLE, LENGTH, RATIO, Dimension, Quantity, unit

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

MAX_ITEMS = 100_000
MAX_DEFINITION_DEPTH = 24
MAX_DEFINITION_NODES = 200_000
CACHE_ENTRIES = 20_000

DEFAULT_SETTINGS: dict[str, Any] = {
    "length_unit": "mm",
    "angle_unit": "deg",
    "precision": 3,
    "display": {"chord": 0.0005, "angle_deg": 12.0, "relative": 0.002},
}


def _new_id() -> str:
    return str(uuid.uuid4())


def check_json(value: Any, location: str) -> Any:
    """A deep copy of ``value`` as plain JSON values (lists for tuples), bounded and finite."""
    count = 0

    def walk(v: Any, depth: int, loc: str) -> Any:
        nonlocal count
        count += 1
        if count > MAX_DEFINITION_NODES:
            raise CadError("LIMIT_EXCEEDED", "the definition is too large", location)
        if depth > MAX_DEFINITION_DEPTH:
            raise CadError("LIMIT_EXCEEDED", "the definition is nested too deeply", loc)
        if v is None or isinstance(v, (bool, str)):
            if isinstance(v, str) and len(v) > 10_000:
                raise CadError("LIMIT_EXCEEDED", "a string is longer than 10,000 characters", loc)
            return v
        if isinstance(v, int):
            if abs(v) > 2**53:
                raise CadError("INVALID_VALUE", "integers must be within +-2^53", loc)
            return v
        if isinstance(v, float):
            if not math.isfinite(v):
                raise CadError("INVALID_VALUE", f"{v} is not finite", loc)
            return v
        if isinstance(v, (list, tuple)):
            return [walk(x, depth + 1, f"{loc}[{i}]") for i, x in enumerate(v)]
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if not isinstance(k, str):
                    raise CadError("INVALID_VALUE", "object keys must be strings", loc)
                out[k] = walk(x, depth + 1, f"{loc}.{k}")
            return out
        if hasattr(v, "tolist"):  # NumPy scalars and arrays
            return walk(v.tolist(), depth, loc)
        raise CadError("INVALID_VALUE", f"{type(v).__name__} is not a JSON value", loc)

    return walk(value, 0, location)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_key(*parts: Any) -> str:
    return hashlib.sha256(canonical(list(parts)).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Items and kinds
# --------------------------------------------------------------------------- #


class Item:
    """A handle on a document item. Read ``id``, ``kind``, ``name``, ``definition`` (a
    copy), ``links`` and ``result``; change it through the document or a kind's handle.
    Handles stay valid across undo and redo."""

    def __init__(self, doc: CadDocument, iid: str, kind: str, name: str, definition: dict):
        self._doc = doc
        self.id = iid
        self.kind = kind
        self._name = name
        self._definition = definition
        self._alive = True
        self._links: tuple[str, ...] = ()
        self._serial = doc._next_serial()
        self._refs: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self._name

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def definition(self) -> dict:
        return copy.deepcopy(self._definition)

    @property
    def links(self) -> tuple[str, ...]:
        return self._links

    @property
    def result(self) -> Any:
        return self._doc.result(self)

    def __repr__(self) -> str:
        return f"<{self.kind} {self._name!r}>"


@dataclass
class Context:
    """What a kind's ``evaluate`` may use."""

    doc: CadDocument
    item: Item
    inputs: dict[str, Any]
    location: str
    deadline: float | None = None
    should_stop: Callable[[], bool] | None = None
    previous: Any = None  # the item's previous result (for continuation), if known

    def check(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise CadError("CANCELLED", "regeneration was cancelled; nothing was changed")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise CadError("TIMEOUT", "regeneration exceeded its time budget; nothing changed")

    def value(self, raw: Any, expected: Dimension, loc: str) -> float:
        """A definition value: number (document unit), quantity text or ``"=expr"``."""
        return self.doc.evaluate_value(raw, expected, f"{self.location}.{loc}")

    def length(self, raw: Any, loc: str) -> float:
        return self.value(raw, LENGTH, loc)

    def angle(self, raw: Any, loc: str) -> float:
        return self.value(raw, ANGLE, loc)


@dataclass
class DisplayPart:
    """One piece of an item's display: a mesh shown by a linked object."""

    mesh: Any
    name: str = ""
    materials: dict[int, str] = field(default_factory=dict)  # face group -> material name
    color: tuple[float, float, float] | None = None
    transform: Any = None  # 4x4 world placement of the mesh (None: identity)


class ItemKind:
    """Base class of item kinds (registered in :data:`KINDS`)."""

    name = "item"
    displays = False

    def check(self, doc: CadDocument, definition: dict, location: str) -> None:
        """Validate a definition completely (raise ``CadError`` with a location)."""

    def references(self, doc: CadDocument, definition: dict) -> list[str]:
        """Ids of the items this definition reads."""
        return []

    def evaluate(self, ctx: Context) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def display(self, doc: CadDocument, item: Item, result: Any) -> list[DisplayPart]:
        return []

    def info(self, doc: CadDocument, item: Item, result: Any) -> dict[str, Any]:
        return {}

    def post_evaluate(self, doc: CadDocument, item: Item, result: Any) -> dict | None:
        """A definition to store after a successful evaluation (e.g. solved sketch
        coordinates), or ``None``."""
        return None


KINDS: dict[str, ItemKind] = {}


def register(kind):
    """Register an item kind (a class, instantiated once, or an instance)."""
    instance = kind() if isinstance(kind, type) else kind
    KINDS[instance.name] = instance
    return kind


def _load_kinds() -> None:
    from . import kinds  # noqa: F401  (registers parameter, frame, sketch, part)

    for module in (
        "aether3d.cad.assembly",
        "aether3d.arch.kinds",
        "aether3d.cad.interop",
        "aether3d.cad.drawing.kind",
    ):
        try:
            __import__(module)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional parts of the stage
            if not module.startswith(exc.name or "\0"):
                raise


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


class CadDocument:
    def __init__(self, scene: Scene) -> None:
        _load_kinds()
        self.scene = scene
        self._items: dict[str, Item] = {}  # every item ever created here (alive or not)
        self._by_name: dict[str, Item] = {}
        self._dependents: dict[str, set[str]] = defaultdict(set)
        self._settings: dict = copy.deepcopy(DEFAULT_SETTINGS)
        self._serial = 0
        self._depth = 0
        self._dirty: set[str] = set()
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._keys: dict[str, str] = {}
        self._results: dict[str, Any] = {}
        self._shown: dict[tuple[str, str], str] = {}
        self._meshes: OrderedDict[str, Any] = OrderedDict()
        self._drivers: dict[str, str] = {}  # object id -> item id
        self._writing = False
        self._deadline: float | None = None
        self._should_stop: Callable[[], bool] | None = None
        self.revision = 0
        self.stats = {"evaluations": 0, "cache_hits": 0, "regenerations": 0}

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial

    # -- discovery --------------------------------------------------------------------

    @property
    def settings(self) -> dict:
        return copy.deepcopy(self._settings)

    @property
    def length_unit(self) -> str:
        return self._settings["length_unit"]

    def items(self, kind: str | None = None) -> list[Item]:
        out = [i for i in self._items.values() if i._alive and (kind is None or i.kind == kind)]
        out.sort(key=lambda i: i._serial)
        return out

    def item(self, ref: Any, kind: str | tuple[str, ...] | None = None) -> Item:
        """An item by handle, id or name (optionally of a given kind)."""
        found = None
        if not isinstance(ref, (Item, str)) and isinstance(getattr(ref, "item", None), Item):
            ref = ref.item  # a kind's handle (SketchHandle, PartHandle, ...)
        if isinstance(ref, Item):
            if self._items.get(ref.id) is ref and ref._alive:
                found = ref
        elif isinstance(ref, str):
            found = self._by_name.get(ref)
            if found is None:
                cand = self._items.get(ref)
                found = cand if cand is not None and cand._alive else None
        if found is None:
            raise CadError(
                "UNKNOWN_ITEM",
                f"no CAD item {ref!r}",
                hint="items: " + (", ".join(i.name for i in self.items()[:30]) or "none"),
            )
        if kind is not None:
            kinds = (kind,) if isinstance(kind, str) else kind
            if found.kind not in kinds:
                raise CadError(
                    "WRONG_ITEM_KIND",
                    f"{found.name!r} is a {found.kind}, not a {' or '.join(kinds)}",
                )
        return found

    def __contains__(self, ref: Any) -> bool:
        try:
            self.item(ref)
        except CadError:
            return False
        return True

    def _driver(self, obj: SceneObject) -> Item | None:
        """The item an object shows: linked directly, or a display part (a child named
        ``<linked object>/<part>``)."""
        iid = self._drivers.get(obj.id)
        if (
            iid is None
            and obj._parent is not None
            and obj._name.startswith(obj._parent._name + "/")
        ):
            iid = self._drivers.get(obj._parent.id)
        return None if iid is None else self._items.get(iid)

    # -- values -------------------------------------------------------------------------

    def parameter_value(self, name: str) -> Quantity:
        item = self._by_name.get(name)
        if item is None or item.kind != "parameter":
            raise KeyError(name)
        return self.result(item)

    def evaluate_value(self, raw: Any, expected: Dimension, location: str) -> float:
        from .expr import evaluate_value

        default = (
            self._settings["length_unit"]
            if expected == LENGTH
            else self._settings["angle_unit"]
            if expected == ANGLE
            else None
        )
        try:
            q = evaluate_value(raw, self.parameter_value, expected=expected, default_unit=default)
        except CadError as exc:
            raise CadError(exc.code, exc.message, location, hint=exc.hint) from None
        if expected == RATIO and q.dimension != RATIO:
            raise CadError("UNIT_MISMATCH", "a plain number is expected", location)
        return q.value

    # -- transactions -------------------------------------------------------------------

    @contextmanager
    def batch(self, label: str = "CAD change", *, deadline: float | None = None, should_stop=None):
        """Several changes as one atomic step; regeneration runs once, at the end."""
        history = self.scene._history
        with history._op(label):
            self._depth += 1
            outer = self._depth == 1
            if outer:
                self._deadline = deadline
                self._should_stop = should_stop
            try:
                yield self
                if outer:
                    self._regenerate()
            except BaseException:
                if outer:
                    self._dirty.clear()
                    self._keys.clear()
                    self._results.clear()
                    self._shown.clear()
                raise
            finally:
                self._depth -= 1
                if outer:
                    self._deadline = None
                    self._should_stop = None

    def _touch(self, item: Item) -> None:
        from ..history import CAD_ITEM

        self.scene._history._touch(CAD_ITEM, item)

    def _set(self, item: Item, *, definition=None, name=None, links=None, alive=None) -> None:
        self._touch(item)
        if definition is not None:
            item._definition = definition
            self._dirty.add(item.id)
        if name is not None:
            item._name = name
        if links is not None:
            item._links = tuple(links)
            self._dirty.add(item.id)
        if alive is not None:
            item._alive = alive
            self._dirty.add(item.id)
        self._reindex(item)
        self.revision += 1

    def _reindex(self, item: Item) -> None:
        for name, it in list(self._by_name.items()):
            if it is item:
                del self._by_name[name]
        if item._alive:
            other = self._by_name.get(item._name)
            if other is not None and other is not item:
                raise CadError("DUPLICATE_NAME", f"a CAD item is already named {item._name!r}")
            self._by_name[item._name] = item
        for oid, iid in list(self._drivers.items()):
            if iid == item.id:
                del self._drivers[oid]
        if item._alive:
            for oid in item._links:
                self._drivers[oid] = item.id

    def _restored(self) -> None:
        """After undo, redo or a rollback: rebuild indexes; forget derived results."""
        self._by_name = {}
        self._drivers = {}
        self._dependents = defaultdict(set)
        for item in self._items.values():
            if item._alive:
                self._by_name[item._name] = item
                for oid in item._links:
                    self._drivers[oid] = item.id
        for item in self._items.values():
            if item._alive:
                kind = KINDS[item.kind]
                try:
                    item._refs = tuple(kind.references(self, item._definition))
                except CadError:
                    item._refs = ()
                for r in item._refs:
                    self._dependents[r].add(item.id)
        self._keys.clear()
        self._results.clear()
        self._shown.clear()
        self._dirty.clear()
        self.revision += 1

    # -- mutations ----------------------------------------------------------------------

    def _unique_name(self, base: str) -> str:
        if base not in self._by_name:
            return base
        k = 1
        while f"{base}.{k:03d}" in self._by_name:
            k += 1
        return f"{base}.{k:03d}"

    def add(
        self, kind: str, name: str | None, definition: dict, *, label: str | None = None
    ) -> Item:
        """Add an item (validated; regenerated before this returns)."""
        if kind not in KINDS:
            raise CadError(
                "UNKNOWN_KIND", f"unknown item kind {kind!r}", hint=", ".join(sorted(KINDS))
            )
        if sum(1 for i in self._items.values() if i._alive) >= MAX_ITEMS:
            raise CadError("LIMIT_EXCEEDED", f"a document holds at most {MAX_ITEMS} items")
        if name is not None:
            _check_name(name)
            if name in self._by_name:
                raise CadError("DUPLICATE_NAME", f"a CAD item is already named {name!r}")
        name = name or self._unique_name(kind.capitalize())
        defn = check_json(definition, f"{kind}[{name}]")
        KINDS[kind].check(self, defn, f"{kind}[{name}]")
        with self.batch(label or f"Add {kind} {name!r}"):
            item = Item(self, _new_id(), kind, name, defn)
            item._alive = False
            self._items[item.id] = item
            self._set(item, alive=True, definition=defn)
            self._update_refs(item)
        return item

    def set_definition(self, ref: Any, definition: dict, *, label: str | None = None) -> None:
        item = self.item(ref)
        defn = check_json(definition, f"{item.kind}[{item.name}]")
        KINDS[item.kind].check(self, defn, f"{item.kind}[{item.name}]")
        with self.batch(label or f"Change {item.kind} {item.name!r}"):
            self._set(item, definition=defn)
            self._update_refs(item)

    def update(self, ref: Any, changes: dict, *, label: str | None = None) -> None:
        """Change top-level keys of a definition (``None`` removes a key)."""
        item = self.item(ref)
        defn = copy.deepcopy(item._definition)
        for k, v in changes.items():
            if v is None:
                defn.pop(k, None)
            else:
                defn[k] = v
        self.set_definition(item, defn, label=label)

    def rename(self, ref: Any, name: str) -> None:
        item = self.item(ref)
        _check_name(name)
        if name == item.name:
            return
        if name in self._by_name:
            raise CadError("DUPLICATE_NAME", f"a CAD item is already named {name!r}")
        if item.kind == "parameter" and any(item.id in i._refs for i in self.items()):
            raise CadError(
                "PARAMETER_IN_USE",
                f"parameter {item.name!r} is used by expressions; renaming it would break them",
            )
        with self.batch(f"Rename {item.name!r} to {name!r}"):
            self._set(item, name=name)

    def remove(self, ref: Any, *, remove_objects: bool = True) -> None:
        item = self.item(ref)
        users = [self._items[d] for d in self._dependents.get(item.id, ()) if self._items[d]._alive]
        if users:
            raise CadError(
                "ITEM_IN_USE",
                f"{item.name!r} is used by {', '.join(u.name for u in users[:10])}",
                hint="remove or change those items first",
            )
        with self.batch(f"Remove {item.kind} {item.name!r}"):
            links = list(item._links)
            self._set(item, alive=False, links=())
            self._update_refs(item)
            if remove_objects:
                for oid in links:
                    if oid in self.scene._objects:
                        self.scene.remove(oid)

    def _update_refs(self, item: Item) -> None:
        for r in item._refs:
            self._dependents[r].discard(item.id)
        if not item._alive:
            item._refs = ()
            return
        refs = tuple(dict.fromkeys(KINDS[item.kind].references(self, item._definition)))
        for r in refs:
            target = self._items.get(r)
            if target is None or not target._alive:
                raise CadError("UNKNOWN_ITEM", f"{item.name!r} references a missing item {r!r}")
        item._refs = refs
        for r in refs:
            self._dependents[r].add(item.id)
        self._check_cycle(item)

    def _check_cycle(self, start: Item) -> None:
        # Is ``start`` reachable from its own references?
        stack = list(start._refs)
        seen: set[str] = set()
        parent: dict[str, str] = {r: start.id for r in start._refs}
        while stack:
            cur = stack.pop()
            if cur == start.id:
                path = [start.name]
                node = parent.get(start.id)
                while node is not None and node != start.id and len(path) < 50:
                    path.append(self._items[node].name)
                    node = parent.get(node)
                raise CadError(
                    "DEPENDENCY_CYCLE",
                    "the references would form a cycle: " + " -> ".join(path + [start.name]),
                )
            if cur in seen:
                continue
            seen.add(cur)
            for r in self._items[cur]._refs:
                if r not in seen:
                    parent.setdefault(r, cur)
                    stack.append(r)

    # -- links -------------------------------------------------------------------------

    def show(self, ref: Any, *, name: str | None = None, parent: Any = None) -> SceneObject:
        """Show an item's display geometry in the scene: a new linked object (the item's
        display parts become its children when there are several)."""
        item = self.item(ref)
        if not KINDS[item.kind].displays:
            raise CadError("NOT_DISPLAYABLE", f"a {item.kind} has no display geometry")
        with self.batch(f"Show {item.name!r}"):
            obj = self.scene.add_object(name or self.scene._unique_name(item.name), parent=parent)
            self._set(item, links=(*item._links, obj.id))
        return obj

    def unlink(self, obj: Any) -> None:
        """Stop an object from following its item (its current mesh stays as a copy)."""
        o = self.scene._resolve(obj)
        item = self._driver(o)
        if item is None:
            return
        with self.batch(f"Unlink {o.name!r}"):
            self._set(item, links=tuple(x for x in item._links if x != o.id))

    def convert_to_mesh(self, obj: Any) -> dict[str, Any]:
        """Detach a linked object: it keeps its current tessellation as a plain mesh, and
        stops following its CAD item. Returns what the object no longer has."""
        o = self.scene._resolve(obj)
        item = self._driver(o)
        if item is None:
            raise CadError("NOT_CAD_DRIVEN", f"object {o.name!r} does not show a CAD item")
        self.unlink(o)
        return {
            "object": o.name,
            "item": item.name,
            "kind": item.kind,
            "lost": [
                "exact geometry (the mesh is a tessellation within the display tolerance)",
                "parametric updates (changes of the item no longer reach this object)",
                "persistent face and edge names",
            ],
            "kept": "the CAD item and its definition remain in the document",
        }

    def _objects_removed(self, objects: list) -> None:
        ids = {o.id for o in objects}
        for item in list(self._items.values()):
            if item._alive and any(oid in ids for oid in item._links):
                self._set(item, links=tuple(x for x in item._links if x not in ids))
                self._dirty.discard(item.id)

    # -- regeneration -------------------------------------------------------------------

    def result(self, ref: Any) -> Any:
        """The item's current result (evaluated on demand, with what it reads)."""
        item = self.item(ref)
        return self._evaluate_item(item)

    def _evaluate_item(self, item: Item) -> Any:
        # Inputs first (iteratively, deepest first).
        order = self._closure_up(item)
        for it in order:
            self._ensure(it)
        return self._results[item.id]

    def _closure_up(self, item: Item) -> list[Item]:
        """The item and everything it reads, in evaluation order."""
        out: list[Item] = []
        seen: set[str] = set()
        stack: list[tuple[Item, bool]] = [(item, False)]
        while stack:
            it, done = stack.pop()
            if done:
                out.append(it)
                continue
            if it.id in seen:
                continue
            seen.add(it.id)
            stack.append((it, True))
            for r in it._refs:
                if r not in seen:
                    stack.append((self._items[r], False))
        return out

    def _key(self, item: Item) -> str:
        return content_key(
            item.kind,
            item._definition,
            [self._keys[r] for r in item._refs],
            {k: self._settings[k] for k in ("length_unit", "angle_unit")},
            [self._items[r]._name for r in item._refs if self._items[r].kind == "parameter"],
        )

    def _ensure(self, item: Item) -> bool:
        """Evaluate if its key changed; True if the result changed."""
        key = self._key(item)
        if self._keys.get(item.id) == key and item.id in self._results:
            return False
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            result = hit
        else:
            kind = KINDS[item.kind]
            ctx = Context(
                self,
                item,
                {r: self._results[r] for r in item._refs},
                f"{item.kind}[{item._name}]",
                self._deadline,
                self._should_stop,
                self._results.get(item.id),
            )
            ctx.check()
            self.stats["evaluations"] += 1
            try:
                result = kind.evaluate(ctx)
            except CadError:
                raise
            except Exception as exc:  # a numerical or backend failure: report, don't leak
                raise CadError(
                    "EVALUATION_FAILED", f"{type(exc).__name__}: {exc}", ctx.location
                ) from exc
            self._cache[key] = result
            if len(self._cache) > CACHE_ENTRIES:
                self._cache.popitem(last=False)
        self._keys[item.id] = key
        self._results[item.id] = result
        return True

    def _regenerate(self) -> None:
        if not self._dirty:
            return
        self.stats["regenerations"] += 1
        dirty = {d for d in self._dirty if d in self._items}
        self._dirty.clear()
        # Affected: the changed items and everything downstream, in dependency order.
        affected: set[str] = set()
        queue = deque(dirty)
        while queue:
            cur = queue.popleft()
            if cur in affected:
                continue
            affected.add(cur)
            queue.extend(self._dependents.get(cur, ()))
        live = [self._items[i] for i in affected if self._items[i]._alive]
        order = _topological(live, self._items)
        changed: list[Item] = []
        for item in order:
            for r in item._refs:
                if r not in self._results:
                    self._evaluate_item(self._items[r])
            if self._ensure(item) or item.id in dirty:
                changed.append(item)
                # Results that rewrite their own definition (solved sketches).
                new_defn = KINDS[item.kind].post_evaluate(self, item, self._results[item.id])
                if new_defn is not None and new_defn != item._definition:
                    self._touch(item)
                    item._definition = new_defn
                    self._keys[item.id] = self._key(item)
                    self._cache[self._keys[item.id]] = self._results[item.id]
        for item in changed:
            if item._links:
                self._update_display(item)

    def _update_display(self, item: Item) -> None:
        from ..history import DATA, TRANSFORM

        kind = KINDS[item.kind]
        parts = kind.display(self, item, self._results[item.id])
        scene = self.scene
        self._writing = True
        try:
            for oid in item._links:
                obj = scene._objects.get(oid)
                if obj is None:
                    continue
                if self._shown.get((item.id, oid)) == self._keys[item.id]:
                    continue
                self._apply_parts(obj, parts, DATA, TRANSFORM)
                self._shown[(item.id, oid)] = self._keys[item.id]
        finally:
            self._writing = False

    def _apply_parts(self, obj, parts: list[DisplayPart], data_aspect, transform_aspect) -> None:
        if len(parts) == 1 and parts[0].transform is None:
            part = parts[0]
            if obj._mesh is not part.mesh:
                obj.mesh = part.mesh
            self._apply_look(obj, part)
            return
        # Several parts: children named after their part, created or reused in order.
        children = {c.name: c for c in obj.children}
        wanted = []
        for k, part in enumerate(parts):
            cname = f"{obj.name}/{part.name or k}"
            child = children.get(cname)
            if child is None:
                child = self.scene.add_object(cname, parent=obj)
            wanted.append(child)
            if child._mesh is not part.mesh:
                child.mesh = part.mesh
            if part.transform is not None:
                from ..transform import Transform

                t = Transform.from_matrix(part.transform)
                if child.transform != t:
                    child.transform = t
            self._apply_look(child, part)
        keep = {c.id for c in wanted}
        for c in list(obj.children):
            if c.id not in keep and c.name.startswith(f"{obj.name}/"):
                self.scene.remove(c)
        if obj._mesh is not None:
            obj.mesh = None

    def _apply_look(self, obj, part: DisplayPart) -> None:
        lib = self.scene._materials
        for group, mname in part.materials.items():
            mat = lib._material_names.get(mname) or lib._materials.get(mname)
            if mat is None:
                continue
            if group == -1:
                if obj.material is not mat:
                    self.scene.assign_material(obj, mat)
            elif obj.material_for_group(group) is not mat:
                self.scene.assign_material(obj, mat, group=group)
        if (
            part.color is not None
            and obj.material is None
            and tuple(round(c, 6) for c in obj.color) != tuple(round(c, 6) for c in part.color)
        ):
            obj.color = part.color

    def regenerate_all(self) -> dict[str, Any]:
        """Re-evaluate every item and refresh every linked object (one history step)."""
        with self.batch("Regenerate CAD document"):
            self._keys.clear()
            self._results.clear()
            self._shown.clear()
            for item in self.items():
                self._dirty.add(item.id)
        return {"items": len(self.items()), **self.stats}

    def set_units(self, length: str | None = None, angle: str | None = None) -> None:
        from ..history import CAD_SETTINGS

        new = copy.deepcopy(self._settings)
        if length is not None:
            if unit(length).dimension != LENGTH:
                raise CadError("UNIT_MISMATCH", f"{length!r} is not a length unit")
            new["length_unit"] = unit(length).symbol
        if angle is not None:
            if unit(angle).dimension != ANGLE:
                raise CadError("UNIT_MISMATCH", f"{angle!r} is not an angle unit")
            new["angle_unit"] = unit(angle).symbol
        with self.batch("Set CAD units"):
            self.scene._history._touch(CAD_SETTINGS, self)
            self._settings = new
            self._dirty.update(i.id for i in self.items())

    def set_display_tolerance(
        self,
        chord: float | None = None,
        angle_deg: float | None = None,
        relative: float | None = None,
    ) -> None:
        from ..history import CAD_SETTINGS

        new = copy.deepcopy(self._settings)
        d = dict(new["display"])
        if chord is not None:
            d["chord"] = float(chord)
        if angle_deg is not None:
            d["angle_deg"] = float(angle_deg)
        if relative is not None:
            d["relative"] = float(relative)
        TessellationTolerance(d["chord"], math.radians(d["angle_deg"]), d["relative"])
        new["display"] = d
        with self.batch("Set CAD display tolerance"):
            self.scene._history._touch(CAD_SETTINGS, self)
            self._settings = new
            self._shown.clear()
            for item in self.items():
                if item._links:
                    self._dirty.add(item.id)
                    self._keys.pop(item.id, None)

    def mesh_for(self, key: str, build: Callable[[], Any]) -> Any:
        """A display mesh by content key: identical geometry gives the identical Mesh
        object, so linked objects are not rewritten and instances share one mesh."""
        cache = self._meshes
        full = content_key(key, self._settings["display"])
        hit = cache.get(full)
        if hit is not None:
            cache.move_to_end(full)
            return hit
        mesh = build()
        cache[full] = mesh
        if len(cache) > CACHE_ENTRIES:
            cache.popitem(last=False)
        return mesh

    def body_mesh(self, part: Any, body: str = "main") -> Any:
        """The display mesh of a part's body (shared by every instance of it)."""
        from .brep.tessellate import tessellate

        item = self.item(part, "part")
        result = self.result(item)
        b = result.body(body)
        return self.mesh_for(
            f"body:{self._keys[item.id]}:{body}",
            lambda: tessellate(b, self.display_tolerance()).to_mesh(
                groups=np.array([f.group for f in b.faces])
            ),
        )

    def display_tolerance(self) -> TessellationTolerance:
        d = self._settings["display"]
        return TessellationTolerance(d["chord"], math.radians(d["angle_deg"]), d["relative"])

    # -- reports ------------------------------------------------------------------------

    def info(self, ref: Any = None, *, evaluate: bool = True) -> dict[str, Any]:
        if ref is None:
            return {
                "settings": self.settings,
                "items": [self.info(i, evaluate=evaluate) for i in self.items()],
                "stats": dict(self.stats),
            }
        item = self.item(ref)
        out: dict[str, Any] = {
            "id": item.id,
            "kind": item.kind,
            "name": item.name,
            "references": [self._items[r].name for r in item._refs],
            "used_by": sorted(
                self._items[d].name
                for d in self._dependents.get(item.id, ())
                if self._items[d]._alive
            ),
            "links": [self.scene._objects[o].name for o in item._links if o in self.scene._objects],
        }
        if evaluate:
            try:
                out.update(KINDS[item.kind].info(self, item, self.result(item)))
            except CadError as exc:
                out["error"] = exc.to_dict()
        return out

    def iter_links(self) -> Iterator[tuple[Item, str]]:
        for item in self.items():
            for oid in item._links:
                yield item, oid

    # -- persistence --------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "settings": copy.deepcopy(self._settings),
            "items": [
                {
                    "id": i.id,
                    "kind": i.kind,
                    "name": i.name,
                    "definition": copy.deepcopy(i._definition),
                    "links": list(i._links),
                }
                for i in self.items()
            ],
        }

    def _load(self, data: dict, location: str = "cad") -> None:
        """Fill an empty document from saved data (no history, no regeneration)."""
        from ..errors import ProjectFormatError

        if not isinstance(data, dict):
            raise ProjectFormatError("must be an object", location)
        settings = data.get("settings", DEFAULT_SETTINGS)
        try:
            settings = check_json(settings, f"{location}.settings")
            unit(settings["length_unit"])
            unit(settings["angle_unit"])
        except (CadError, KeyError, TypeError) as exc:
            raise ProjectFormatError(
                f"invalid CAD settings: {exc}", f"{location}.settings"
            ) from None
        merged = copy.deepcopy(DEFAULT_SETTINGS)
        merged.update(settings)
        self._settings = merged
        items = data.get("items", [])
        if not isinstance(items, list):
            raise ProjectFormatError("items must be a list", f"{location}.items")
        for k, raw in enumerate(items):
            loc = f"{location}.items[{k}]"
            if not isinstance(raw, dict):
                raise ProjectFormatError("an item must be an object", loc)
            try:
                kind = raw["kind"]
                if kind not in KINDS:
                    raise CadError("UNKNOWN_KIND", f"unknown item kind {kind!r}")
                iid = str(uuid.UUID(raw["id"]))
                name = raw["name"]
                _check_name(name)
                defn = check_json(raw.get("definition", {}), f"{loc}.definition")
                links = [str(uuid.UUID(x)) for x in raw.get("links", [])]
            except (CadError, KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ProjectFormatError(f"invalid CAD item: {exc}", loc) from None
            if iid in self._items or name in self._by_name:
                raise ProjectFormatError("duplicate CAD item id or name", loc)
            item = Item(self, iid, kind, name, defn)
            item._links = tuple(links)
            self._items[iid] = item
            self._by_name[name] = item
        for item in self.items():
            loc = f"{location}.items[{item.name}]"
            try:
                KINDS[item.kind].check(self, item._definition, f"{item.kind}[{item.name}]")
                self._update_refs(item)
            except CadError as exc:
                raise ProjectFormatError(f"invalid CAD item: {exc}", loc) from None
            for oid in item._links:
                self._drivers[oid] = item.id


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise CadError("INVALID_NAME", f"invalid item name {name!r} (1..128 characters)")
    if any(ch in name for ch in "\n\r\t\0/[]"):
        raise CadError("INVALID_NAME", "item names cannot contain control characters, / or []")


def _topological(items: list[Item], all_items: dict[str, Item]) -> list[Item]:
    ids = {i.id for i in items}
    indeg = {i.id: 0 for i in items}
    out_edges: dict[str, list[str]] = defaultdict(list)
    for i in items:
        for r in i._refs:
            if r in ids:
                indeg[i.id] += 1
                out_edges[r].append(i.id)
    ready = sorted((i for i in items if indeg[i.id] == 0), key=lambda x: x._serial)
    order: list[Item] = []
    queue = deque(ready)
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for d in sorted(out_edges.get(cur.id, ()), key=lambda x: all_items[x]._serial):
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(all_items[d])
    if len(order) != len(items):
        raise CadError("DEPENDENCY_CYCLE", "the CAD items' references form a cycle")
    return order
