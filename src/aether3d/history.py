"""Operations history: undo, redo, transactions and checkpoints (``scene.history``).

Every change to a scene -- through the Python API, a script or the command line --
is recorded as one **step** of its history, which :meth:`History.undo` and
:meth:`History.redo` reverse and re-apply exactly::

    cube = scene.add_cube(name="Cube")      # step 1: "Add object 'Cube'"
    cube.position = (0, 0, 2)               # step 2: "Set position of 'Cube'"
    scene.history.undo()                    # the cube is back at the origin
    scene.history.redo()                    # ... and at (0, 0, 2) again

How it works (see ``docs/HISTORY.md``):

* **Operations.** Every public mutating call is one operation. Calls made inside
  it (``add_camera`` creates an object, aims it and activates it) belong to it,
  so a step is always a whole user-level action. :meth:`History.transaction`
  groups several calls into one step.
* **Aspects.** The scene's state is divided into small parts -- an object's
  transform, its camera, its place in the hierarchy, a material, the procedural
  definitions, ... Before an operation first writes a part, the part's current
  state is captured; when the operation ends, the new state is captured too.
  A step stores just these before/after pairs of the parts it changed: never a
  copy of the scene. Meshes, images and texture graphs are immutable and are
  shared by reference, never copied.
* **Atomicity.** If an operation (or a transaction) raises, every part it
  changed is restored before the exception propagates; nested transactions are
  savepoints. Failed operations leave no step.
* **Identity.** Undo and redo restore the same Python objects: a removed object
  that comes back is the same :class:`~aether3d.SceneObject` (same id, same
  handle), and so are materials, textures and procedural handles.
* **Caches.** Renderer caches (GPU mesh buffers, ray tracing BVHs, decoded
  images, procedural evaluations) are keyed by immutable content, so they stay
  valid; undo and redo bump ``scene.revision``, ``scene.materials.version`` and
  ``scene.procedural.revision`` like any other change.

Undo and redo never record themselves; a new change after undoing discards the
undone steps (the redo branch).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import HistoryError

if TYPE_CHECKING:
    from .scene import Scene, SceneObject

DEFAULT_MAX_ENTRIES = 256
"""Default number of undo steps kept (``history.max_entries``)."""

DEFAULT_MAX_BYTES = 256 * 2**20
"""Default memory budget of the history in bytes (``history.max_bytes``)."""

STRICT = False
"""Testing aid: check the scene's invariants after every reversal, and fail on
changes recorded outside an operation (an internal error)."""

# Fix-ups after restoring states.
_NAMES = "names"  # rebuild the scene's name index
_LIBRARY = "library"  # rebuild the material library's name indexes, bump its version
_PROCEDURAL = "procedural"  # bump the procedural revision

_UNSET = object()
_UNRECORDED = contextlib.nullcontext()


def recording(value: Any, fields: Iterable[str]) -> Any:
    """The operation of an in-place change of ``fields`` of a value object (an object's
    camera, light or appearance; the world, its environment, the render settings).

    Value objects that belong to no scene record nothing.
    """
    owner = getattr(value, "_owner", None)
    if owner is None:
        return _UNRECORDED
    holder, aspect = owner
    return holder._edit_value(aspect, fields)


def _same(a: Any, b: Any) -> bool:
    return a is b or a == b


def _same_items(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------- #
# Aspects: how a part of the scene is captured, restored and described
# --------------------------------------------------------------------------- #


class Aspect:
    """One part of the scene's state (see the module docstring)."""

    __slots__ = ("name", "fixups")

    def __init__(self, name: str, fixups: tuple[str, ...] = ()) -> None:
        self.name = name
        self.fixups = fixups

    def capture(self, target: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply(self, target: Any, state: Any) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def same(self, a: Any, b: Any) -> bool:
        return _same(a, b)

    def payloads(self, state: Any) -> tuple:
        """Large immutable objects held by ``state`` (meshes, image bytes)."""
        return ()

    def size(self, state: Any) -> int:
        """Rough size of ``state`` in bytes, payloads excluded."""
        return 120

    def path(self, target: Any) -> str:
        return f"objects[{target._id}].{self.name}"

    def __repr__(self) -> str:
        return f"<aspect {self.name}>"


class _Attr(Aspect):
    """An aspect held in one attribute with an immutable value."""

    __slots__ = ("attr", "root")

    def __init__(self, name: str, attr: str, fixups: tuple[str, ...] = (), root: str = "") -> None:
        super().__init__(name, fixups)
        self.attr = attr
        self.root = root

    def capture(self, target: Any) -> Any:
        return getattr(target, self.attr)

    def apply(self, target: Any, state: Any) -> None:
        setattr(target, self.attr, state)

    def same(self, a: Any, b: Any) -> bool:
        return a is b or (type(a) is type(b) and a == b)

    def path(self, target: Any) -> str:
        return f"{self.root}.{self.name}" if self.root else super().path(target)


class _Transform(Aspect):
    __slots__ = ()

    def capture(self, obj: SceneObject) -> tuple:
        t = obj._transform
        return (t._position, t._rotation, t._scale)

    def apply(self, obj: SceneObject, state: tuple) -> None:
        t = obj._transform
        t._position, t._rotation, t._scale = state


class _Fields(Aspect):
    """A value object (appearance, camera, light, render settings) stored field by field.

    ``attr`` is the attribute of the owner that holds the value object, ``factory``
    builds an empty one when a state has to be restored onto ``None``.
    """

    __slots__ = ("attr", "fields", "factory", "root")

    def __init__(
        self, name: str, attr: str, fields: tuple[str, ...], factory: Callable, root: str = ""
    ) -> None:
        super().__init__(name)
        self.attr = attr
        self.fields = tuple(f"_{f}" for f in fields)
        self.factory = factory
        self.root = root

    def capture(self, owner: Any) -> tuple | None:
        value = getattr(owner, self.attr)
        if value is None:
            return None
        return tuple(getattr(value, f) for f in self.fields)

    def apply(self, owner: Any, state: tuple | None) -> None:
        if state is None:
            setattr(owner, self.attr, None)
            return
        value = getattr(owner, self.attr)
        if value is None:
            value = self.factory()
            setattr(owner, self.attr, value)
        for f, v in zip(self.fields, state, strict=True):
            object.__setattr__(value, f, v)

    def path(self, target: Any) -> str:
        return f"{self.root}.{self.name}" if self.root else super().path(target)


class _World(Aspect):
    """The world with its environment (both kept as the same instances)."""

    __slots__ = ()
    _FIELDS = ("_background", "_ambient_color", "_ambient_intensity")

    def capture(self, scene: Scene) -> tuple:
        world = scene._world
        env = world._environment
        return (
            *(getattr(world, f) for f in self._FIELDS),
            tuple(getattr(env, f"_{f}") for f in env._FIELDS),
        )

    def apply(self, scene: Scene, state: tuple) -> None:
        world = scene._world
        for f, v in zip(self._FIELDS, state[:3], strict=True):
            object.__setattr__(world, f, v)
        env = world._environment
        for f, v in zip(env._FIELDS, state[3], strict=True):
            object.__setattr__(env, f"_{f}", v)

    def path(self, target: Any) -> str:
        return "world"


class _Data(Aspect):
    """An object's geometry: its static mesh and its procedural geometry node."""

    __slots__ = ()

    def capture(self, obj: SceneObject) -> tuple:
        return (obj._mesh, obj._geometry)

    def apply(self, obj: SceneObject, state: tuple) -> None:
        obj._mesh, obj._geometry = state

    def same(self, a: tuple, b: tuple) -> bool:
        return a[0] is b[0] and a[1] == b[1]

    def payloads(self, state: tuple) -> tuple:
        return () if state[0] is None else (state[0],)


class _Annotations(Aspect):
    """An object's tags and metadata (metadata kept as canonical JSON text)."""

    __slots__ = ()

    def capture(self, obj: SceneObject) -> tuple:
        return (obj._tags, obj._metadata)

    def apply(self, obj: SceneObject, state: tuple) -> None:
        obj._tags, obj._metadata = state


class _Materials(Aspect):
    """An object's material and per-face-group materials."""

    __slots__ = ()

    def capture(self, obj: SceneObject) -> tuple:
        return (obj._material, tuple(sorted(obj._group_materials.items())))

    def apply(self, obj: SceneObject, state: tuple) -> None:
        obj._material = state[0]
        obj._group_materials = dict(state[1])


class _Members(Aspect):
    """An ordered list of objects held in one attribute (children, roots)."""

    __slots__ = ("attr", "root")

    def __init__(self, name: str, attr: str, root: str = "") -> None:
        super().__init__(name)
        self.attr = attr
        self.root = root

    def capture(self, target: Any) -> tuple:
        return tuple(getattr(target, self.attr))

    def apply(self, target: Any, state: tuple) -> None:
        setattr(target, self.attr, list(state))

    def same(self, a: tuple, b: tuple) -> bool:
        return _same_items(a, b)

    def size(self, state: tuple) -> int:
        return 64 + 8 * len(state)

    def path(self, target: Any) -> str:
        return f"{self.root}.{self.name}" if self.root else super().path(target)


class _Alive(Aspect):
    """Whether an object is in its scene (created / removed)."""

    __slots__ = ()

    def capture(self, obj: SceneObject) -> bool:
        # The object table, not ``obj._scene``: a new object knows its scene before it
        # is added to it.
        return obj._home._objects.get(obj._id) is obj

    def apply(self, obj: SceneObject, state: bool) -> None:
        home = obj._home
        if state:
            obj._scene = home
            home._objects[obj._id] = obj
        else:
            obj._scene = None
            home._objects.pop(obj._id, None)

    def path(self, obj: SceneObject) -> str:
        return f"objects[{obj._id}]"


class _Library(Aspect):
    """The ordered materials (or textures) of the material library."""

    __slots__ = ("attr",)

    def __init__(self, name: str, attr: str) -> None:
        super().__init__(name, (_LIBRARY,))
        self.attr = attr

    def capture(self, library: Any) -> tuple:
        return tuple(getattr(library, self.attr).values())

    def apply(self, library: Any, state: tuple) -> None:
        kept = {id(item) for item in state}
        for item in getattr(library, self.attr).values():
            if id(item) not in kept:
                item._library = None
        for item in state:
            item._library = library
        setattr(library, self.attr, {item._id: item for item in state})

    def same(self, a: tuple, b: tuple) -> bool:
        return _same_items(a, b)

    def size(self, state: tuple) -> int:
        return 64 + 8 * len(state)

    def path(self, target: Any) -> str:
        return self.name


class _Material(Aspect):
    """One material: name, model, parameters and texture bindings."""

    __slots__ = ()

    def capture(self, m: Any) -> tuple:
        return (m._name, m._model, dict(m._values), dict(m._textures))

    def apply(self, m: Any, state: tuple) -> None:
        m._name, m._model = state[0], state[1]
        m._values, m._textures = dict(state[2]), dict(state[3])

    def size(self, state: tuple) -> int:
        return 200 + 48 * (len(state[2]) + len(state[3]))

    def path(self, m: Any) -> str:
        return f"materials[{m._id}]"


class _Texture(Aspect):
    """One texture: name and source (file or embedded bytes, sampler, or graph)."""

    __slots__ = ()

    def capture(self, t: Any) -> tuple:
        from .materials.library import _texture_state

        return (t._name, _texture_state(t))

    def apply(self, t: Any, state: tuple) -> None:
        from .materials.library import _set_texture_state

        t._name = state[0]
        _set_texture_state(t, state[1])

    def same(self, a: tuple, b: tuple) -> bool:
        return (
            a[0] == b[0]
            and a[1].keys() == b[1].keys()
            and all(_same(a[1][k], b[1][k]) for k in a[1])
        )

    def payloads(self, state: tuple) -> tuple:
        data = state[1].get("_data")
        return () if data is None else (data,)

    def path(self, t: Any) -> str:
        return f"textures[{t._id}]"


class _Procedural(Aspect):
    """The procedural definitions: parameters, rules and geometry nodes."""

    __slots__ = ()

    def capture(self, system: Any) -> tuple:
        return (system._revision, system._defs_state())

    def apply(self, system: Any, state: tuple) -> None:
        system._restore_defs(state[1])

    def same(self, a: tuple, b: tuple) -> bool:
        # Every successful change bumps the revision, even one that sets a value to
        # what it was: then the definitions are compared (dataclasses, by value).
        return a[0] == b[0] or a[1] == b[1]

    def payloads(self, state: tuple) -> tuple:
        meshes = []
        for g in state[1]["geometries"]:
            mesh = g["source"].mesh
            if mesh is not None:
                meshes.append(mesh)
        return tuple(meshes)

    def size(self, state: tuple) -> int:
        defs = state[1]
        return 256 + 128 * len(defs["parameters"]) + 768 * len(defs["geometries"])

    def path(self, target: Any) -> str:
        return "procedural"


def _appearance() -> Any:
    from .rendering.settings import Appearance

    return Appearance.__new__(Appearance)


def _camera() -> Any:
    from .rendering.camera import Camera

    return Camera.__new__(Camera)


def _light() -> Any:
    from .rendering.light import Light

    return Light.__new__(Light)


def _render_settings() -> Any:
    from .rendering.settings import RenderSettings

    return RenderSettings.__new__(RenderSettings)


# Object aspects.
NAME = _Attr("name", "_name", (_NAMES,))
TRANSFORM = _Transform("transform")
VISIBLE = _Attr("visible", "_visible")
APPEARANCE = _Fields("appearance", "_appearance", ("color", "specular", "shininess"), _appearance)
CAMERA = _Fields(
    "camera",
    "_camera",
    ("projection", "fov", "fov_axis", "ortho_scale", "near", "far", "aspect_ratio"),
    _camera,
)
LIGHT = _Fields(
    "light",
    "_light",
    ("type", "color", "intensity", "range", "falloff", "radius", "angle"),
    _light,
)
DATA = _Data("geometry")
MATERIALS = _Materials("material", (_LIBRARY,))
PARENT = _Attr("parent", "_parent")
CHILDREN = _Members("children", "_children")
ALIVE = _Alive("alive", (_NAMES,))
ANNOTATIONS = _Annotations("annotations")
# Scene aspects.
SCENE_NAME = _Attr("name", "_name", root="scene")
ROOTS = _Members("roots", "_roots", root="scene")
ACTIVE_CAMERA = _Attr("active_camera", "_active_camera", root="scene")
WORLD = _World("world")
RENDER = _Fields(
    "render",
    "_render_settings",
    ("width", "height", "samples", "exposure", "cull", "default_lights"),
    _render_settings,
    root="scene",
)
# Material library and procedural aspects.
LIBRARY_MATERIALS = _Library("materials", "_materials")
LIBRARY_TEXTURES = _Library("textures", "_textures")
MATERIAL = _Material("material", (_LIBRARY,))
TEXTURE = _Texture("texture", (_LIBRARY,))
PROCEDURAL = _Procedural("procedural", (_PROCEDURAL,))


# --------------------------------------------------------------------------- #
# Steps, frames and the public views
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HistoryEntry:
    """A step of the history, as reported by :attr:`History.entries` and friends.

    ``index`` identifies the step (it increases with every new step and is never
    reused in a session), ``label`` describes it, ``changes`` counts the parts of
    the scene it changed and ``paths`` names them (``objects[<id>].transform``,
    ``materials[<id>]``, ``world``, ``procedural``, ...). ``bytes`` estimates the
    size of the step: its states and every mesh or image they refer to (also those
    the scene or other steps share; :attr:`History.nbytes` counts each once, and
    only those the scene no longer uses).
    """

    index: int
    label: str
    changes: int
    paths: tuple[str, ...]
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "changes": self.changes,
            "paths": list(self.paths),
            "bytes": self.bytes,
        }


class _Change:
    __slots__ = ("aspect", "target", "before", "after")

    def __init__(self, aspect: Aspect, target: Any, before: Any, after: Any) -> None:
        self.aspect, self.target, self.before, self.after = aspect, target, before, after


class _Step:
    __slots__ = ("serial", "parent", "label", "changes", "small", "payloads")

    def __init__(self, serial: int, parent: int, label: str, changes: list[_Change]) -> None:
        self.serial = serial
        self.parent = parent
        self.label = label
        self.changes = changes
        small = 96
        payloads: dict[int, Any] = {}
        for c in changes:
            small += 48 + c.aspect.size(c.before) + c.aspect.size(c.after)
            for state in (c.before, c.after):
                for p in c.aspect.payloads(state):
                    payloads[id(p)] = p
        self.small = small
        self.payloads = list(payloads.values())


class _Frame:
    __slots__ = ("label", "touched", "implicit")

    def __init__(self, label: str | None, implicit: bool = False) -> None:
        self.label = label
        self.touched: dict[tuple[Aspect, int], tuple[Aspect, Any, Any]] = {}
        self.implicit = implicit


class _Op:
    """Context manager of one operation (see :meth:`History._op`)."""

    __slots__ = ("_history", "_label", "_aspect", "_target", "_frame")

    def __init__(
        self, history: History, label: str | None, aspect: Aspect | None, target: Any
    ) -> None:
        self._history = history
        self._label = label
        self._aspect = aspect
        self._target = target

    def __enter__(self) -> _Op:
        history = self._history
        self._frame = history._begin(self._label)
        if self._aspect is not None:
            history._touch(self._aspect, self._target)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._history._end(self._frame, exc_type is None)
        return False


class Transaction:
    """The handle of ``with scene.history.transaction(label) as tx:``.

    After the block, :attr:`entry` is the recorded step (``None`` if nothing
    changed, or if the history is disabled). :meth:`rollback` undoes everything
    done in the transaction so far; the block may go on making changes.
    """

    def __init__(self, history: History, label: str) -> None:
        self._history = history
        self.label = label
        self._frame: _Frame | None = None
        self._entry: HistoryEntry | None = None

    @property
    def entry(self) -> HistoryEntry | None:
        return self._entry

    def rollback(self) -> None:
        """Restore the scene as it was when the transaction started."""
        frame = self._frame
        history = self._history
        if frame is None or frame not in history._frames:
            raise HistoryError("TRANSACTION_CLOSED", "this transaction is not open")
        if history._frames[-1] is not frame:
            raise HistoryError(
                "TRANSACTION_OPEN", "an operation nested in this transaction is still open"
            )
        history._restore_frame(frame)
        frame.touched.clear()


# --------------------------------------------------------------------------- #
# The history
# --------------------------------------------------------------------------- #


class History:
    """Undo/redo history of one scene (``scene.history``); see the module docstring."""

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._undo: list[_Step] = []
        self._redo: list[_Step] = []
        self._frames: list[_Frame] = []
        self._applying = False
        self._building = 0
        self._enabled = True
        self._max_entries = DEFAULT_MAX_ENTRIES
        self._max_bytes = DEFAULT_MAX_BYTES
        self._serial = 0
        self._current = 0  # the serial of the step that produced the current state
        self._checkpoints: dict[str, int] = {}
        self._payloads: dict[int, list] = {}  # id -> [object, bytes, references]
        self._payload_bytes = 0
        self._small_bytes = 0
        self._revision = 0
        self.dropped = 0
        """Steps discarded so far to respect :attr:`max_entries` and :attr:`max_bytes`."""

    # ------------------------------------------------------------------ #
    # Recording (used by every mutating method of the scene)
    # ------------------------------------------------------------------ #

    def _op(self, label: str | None, aspect: Aspect | None = None, target: Any = None) -> _Op:
        """An operation: ``with history._op("Add object", ALIVE, obj): ...``."""
        return _Op(self, label, aspect, target)

    def _begin(self, label: str | None) -> _Frame | None:
        if self._building:
            return None
        if self._applying:
            raise HistoryError(
                "HISTORY_BUSY", "the scene cannot change while its history restores a step"
            )
        frames = self._frames
        if frames and frames[-1].implicit:
            self._settle()
        frame = _Frame(label)
        frames.append(frame)
        return frame

    def _touch(self, aspect: Aspect, target: Any) -> None:
        """Capture ``aspect`` of ``target`` before an operation first writes it."""
        if self._building or self._applying:
            return
        frames = self._frames
        if not frames:
            if STRICT:
                raise AssertionError(f"{aspect!r} changed outside an operation")
            frames.append(_Frame(None, implicit=True))
        key = (aspect, id(target))
        state = _UNSET
        for frame in reversed(frames):
            touched = frame.touched
            if key in touched:
                break
            if state is _UNSET:
                state = aspect.capture(target)
            touched[key] = (aspect, target, state)

    def _end(self, frame: _Frame | None, ok: bool) -> _Step | None:
        if frame is None:
            return None
        frames = self._frames
        if not frames or frames[-1] is not frame:  # pragma: no cover - internal error
            raise HistoryError("HISTORY_CORRUPT", "operations ended out of order")
        frames.pop()
        if not ok:
            self._restore_frame(frame)
            return None
        if frames:
            return None  # the enclosing operation already holds every change
        return self._commit(frame)

    def _settle(self) -> None:
        """Commit a pending implicit operation (a change made outside any operation)."""
        frames = self._frames
        if frames and frames[-1].implicit:
            self._commit(frames.pop())

    def _commit(self, frame: _Frame) -> _Step | None:
        changes = []
        for aspect, target, before in frame.touched.values():
            after = aspect.capture(target)
            if not aspect.same(before, after):
                changes.append(_Change(aspect, target, before, after))
        if not changes:
            return None
        self._revision += 1
        self._discard_redo()
        if not self._enabled:
            return None
        self._serial += 1
        label = frame.label or _default_label(changes)
        step = _Step(self._serial, self._current, label, changes)
        self._undo.append(step)
        self._account(step, 1)
        self._current = step.serial
        self._trim()
        return step

    def _restore_frame(self, frame: _Frame) -> None:
        """Roll back a failed operation: no net change, so no revision changes either."""
        changes = [_Change(a, t, before, None) for a, t, before in frame.touched.values()]
        if changes:
            self._apply(changes, forward=False, rollback=True)

    def _apply(self, changes: list[_Change], forward: bool, rollback: bool = False) -> None:
        """Restore every change's after (``forward``) or before state, then fix indexes.

        If that fails (``HISTORY_CONFLICT``), the states it wrote are put back before
        the error propagates: a step is applied whole or not at all.
        """
        self._applying = True
        try:
            try:
                flags = self._write(changes, forward)
                if rollback:
                    flags.discard(_PROCEDURAL)  # its revision counts successful changes only
                self._fixup(flags)
            except BaseException:
                if not rollback:  # a rollback has no states to go back to
                    self._fixup(self._write(changes, not forward), strict=False)
                raise
        finally:
            self._applying = False
        if STRICT:
            check_scene(self._scene)

    @staticmethod
    def _write(changes: list[_Change], forward: bool) -> set[str]:
        flags: set[str] = set()
        for c in changes if forward else reversed(changes):
            c.aspect.apply(c.target, c.after if forward else c.before)
            flags.update(c.aspect.fixups)
        return flags

    def _fixup(self, flags: set[str], strict: bool = True) -> None:
        scene = self._scene
        if _NAMES in flags:
            by_name: dict[str, Any] = {}
            for obj in scene._objects.values():
                if obj._name in by_name and strict:
                    raise HistoryError(
                        "HISTORY_CONFLICT",
                        f"two objects would be named {obj._name!r}",
                        hint="the scene was changed outside its history (private attributes?)",
                    )
                by_name[obj._name] = obj
            scene._by_name = by_name
        if _LIBRARY in flags:
            library = scene._materials
            library._material_names = {m._name: m for m in library._materials.values()}
            library._texture_names = {t._name: t for t in library._textures.values()}
            library._changed()
        if _PROCEDURAL in flags and scene._procedural is not None:
            scene._procedural._revision += 1

    # ------------------------------------------------------------------ #
    # Memory and limits
    # ------------------------------------------------------------------ #

    def _account(self, step: _Step, sign: int) -> None:
        self._small_bytes += sign * step.small
        payloads = self._payloads
        for p in step.payloads:
            key = id(p)
            entry = payloads.get(key)
            if sign > 0:
                if entry is None:
                    size = _payload_size(p)
                    payloads[key] = [p, size, 1]
                    self._payload_bytes += size
                else:
                    entry[2] += 1
            else:
                entry[2] -= 1
                if entry[2] == 0:
                    del payloads[key]
                    self._payload_bytes -= entry[1]

    def _live_payloads(self) -> set[int]:
        """Ids of the meshes and image bytes the scene itself uses."""
        scene = self._scene
        live: set[int] = set()
        for obj in scene._objects.values():
            if obj._mesh is not None:
                live.add(id(obj._mesh))
        system = scene._procedural
        if system is not None:
            for g in system._geometries.values():
                if g.source.mesh is not None:
                    live.add(id(g.source.mesh))
        for t in scene._materials._textures.values():
            data = getattr(t, "_data", None)
            if data is not None:
                live.add(id(data))
        return live

    def _history_bytes(self, live: set[int] | None = None) -> int:
        if live is None:
            live = self._live_payloads()
        own = sum(e[1] for key, e in self._payloads.items() if key not in live)
        return self._small_bytes + own

    def _drop(self, step: _Step) -> None:
        self._account(step, -1)
        self.dropped += 1

    def _trim(self) -> None:
        undo = self._undo
        while len(undo) > self._max_entries:
            self._drop(undo.pop(0))
        if self._small_bytes + self._payload_bytes > self._max_bytes:
            live = self._live_payloads()
            while (undo or self._redo) and self._history_bytes(live) > self._max_bytes:
                # The farthest steps go first: the oldest undo steps, then the last redo ones.
                self._drop(undo.pop(0) if undo else self._redo.pop(0))
        self._prune_checkpoints()

    def _discard_redo(self) -> None:
        for step in self._redo:
            self._account(step, -1)
        self._redo.clear()
        self._prune_checkpoints()

    def _reachable(self) -> set[int]:
        states = {self._current}
        if self._undo:
            states.add(self._undo[0].parent)
        states.update(s.serial for s in self._undo)
        states.update(s.serial for s in self._redo)
        return states

    def _prune_checkpoints(self) -> None:
        if self._checkpoints:
            reachable = self._reachable()
            for name in [n for n, s in self._checkpoints.items() if s not in reachable]:
                del self._checkpoints[name]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def enabled(self) -> bool:
        """Whether changes are recorded. Disabling clears the history (changes made
        while it is disabled cannot be undone, so nothing before them can be either)."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise HistoryError("INVALID_ARGUMENT", f"enabled must be True or False, got {value!r}")
        self._settle()
        if not value:
            self.clear()
        self._enabled = value

    @contextmanager
    def disabled(self) -> Iterator[History]:
        """``with scene.history.disabled():`` -- change the scene without recording
        (for example to build a large scene); the history is cleared."""
        previous = self._enabled
        self.enabled = False
        try:
            yield self
        finally:
            self._enabled = previous

    @property
    def max_entries(self) -> int:
        """How many undo steps are kept (the oldest are dropped first)."""
        return self._max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        self.configure(max_entries=value)

    @property
    def max_bytes(self) -> int:
        """Memory budget in bytes; the oldest steps are dropped to respect it."""
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        self.configure(max_bytes=value)

    def configure(self, *, max_entries: int | None = None, max_bytes: int | None = None) -> None:
        """Change the limits; steps beyond them are dropped at once."""
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise HistoryError(
                    "INVALID_ARGUMENT", f"{name} must be a non-negative integer, got {value!r}"
                )
        self._settle()
        if max_entries is not None:
            self._max_entries = max_entries
        if max_bytes is not None:
            self._max_bytes = max_bytes
        self._trim()

    @property
    def nbytes(self) -> int:
        """Estimated memory held only by the history: its states, and meshes and images
        the scene no longer uses."""
        self._settle()
        return self._history_bytes()

    @property
    def revision(self) -> int:
        """Increases with every change of the scene, undo and redo included."""
        return self._revision

    @property
    def can_undo(self) -> bool:
        self._settle()
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        self._settle()
        return bool(self._redo)

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        """The steps that :meth:`undo` reverses, oldest first (the last is undone first)."""
        self._settle()
        return tuple(self._entry(s) for s in self._undo)

    @property
    def redo_entries(self) -> tuple[HistoryEntry, ...]:
        """The undone steps that :meth:`redo` re-applies, the next one first."""
        self._settle()
        return tuple(self._entry(s) for s in reversed(self._redo))

    def __len__(self) -> int:
        self._settle()
        return len(self._undo)

    def _entry(self, step: _Step) -> HistoryEntry:
        paths = tuple(dict.fromkeys(c.aspect.path(c.target) for c in step.changes))
        size = step.small + sum(_payload_size(p) for p in step.payloads)
        return HistoryEntry(step.serial, step.label, len(step.changes), paths, size)

    def _check_idle(self, what: str) -> None:
        if self._frames:
            raise HistoryError(
                "TRANSACTION_OPEN",
                f"cannot {what} while an operation or transaction is in progress",
                hint="finish the transaction first",
            )

    @staticmethod
    def _check_steps(steps: Any) -> int:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise HistoryError(
                "INVALID_ARGUMENT", f"steps must be a positive integer, got {steps!r}"
            )
        return steps

    def undo(self, steps: int = 1) -> list[HistoryEntry]:
        """Reverse the last ``steps`` steps; returns them, the most recent first.

        Raises ``NOTHING_TO_UNDO`` (and changes nothing) if fewer steps are recorded.
        """
        self._settle()
        self._check_idle("undo")
        steps = self._check_steps(steps)
        if steps > len(self._undo):
            raise HistoryError(
                "NOTHING_TO_UNDO",
                f"cannot undo {steps} step(s): {len(self._undo)} recorded",
                details={"available": len(self._undo)},
            )
        return self._move(self._undo, self._redo, steps, forward=False)

    def redo(self, steps: int = 1) -> list[HistoryEntry]:
        """Re-apply the last ``steps`` undone steps; returns them, in order.

        Raises ``NOTHING_TO_REDO`` (and changes nothing) if fewer steps were undone.
        """
        self._settle()
        self._check_idle("redo")
        steps = self._check_steps(steps)
        if steps > len(self._redo):
            raise HistoryError(
                "NOTHING_TO_REDO",
                f"cannot redo {steps} step(s): {len(self._redo)} undone",
                details={"available": len(self._redo)},
            )
        return self._move(self._redo, self._undo, steps, forward=True)

    def _move(
        self, source: list[_Step], target: list[_Step], steps: int, forward: bool
    ) -> list[HistoryEntry]:
        moved: list[_Step] = []
        try:
            for _ in range(steps):
                step = source[-1]
                self._apply(step.changes, forward)
                source.pop()
                target.append(step)
                self._current = step.serial if forward else step.parent
                self._revision += 1
                moved.append(step)
        except BaseException:
            # All or nothing: put back the steps already moved.
            for step in reversed(moved):
                self._apply(step.changes, not forward)
                target.pop()
                source.append(step)
                self._current = step.parent if forward else step.serial
                self._revision += 1
            raise
        return [self._entry(s) for s in moved]

    @contextmanager
    def transaction(self, label: str = "Transaction") -> Iterator[Transaction]:
        """Group every change made in the block into one step, atomically.

        If the block raises, everything it changed is restored and the exception
        propagates. Transactions nest; an inner one that fails is rolled back
        alone (a savepoint) if the outer block handles its exception.
        """
        if not isinstance(label, str) or not label.strip():
            raise HistoryError(
                "INVALID_ARGUMENT", f"label must be a non-empty string, got {label!r}"
            )
        self._settle()
        tx = Transaction(self, label)
        frame = self._begin(label)
        tx._frame = frame
        try:
            yield tx
        except BaseException:
            if frame is not None and frame in self._frames:
                self._end(frame, False)
            raise
        step = self._end(frame, True)
        tx._entry = None if step is None else self._entry(step)

    def clear(self) -> None:
        """Forget every step and checkpoint (the scene is unchanged)."""
        self._settle()
        for step in self._undo:
            self._account(step, -1)
        self._undo.clear()
        self._discard_redo()
        self._checkpoints.clear()

    # -- checkpoints ----------------------------------------------------------------

    def checkpoint(self, name: str) -> None:
        """Name the current state; :meth:`goto` returns to it while it is reachable.

        A checkpoint costs no memory: it marks a position in the history. It is
        forgotten when the steps leading to it are dropped (limits) or discarded
        (a new change after undoing past it).
        """
        if not isinstance(name, str) or not name.strip():
            raise HistoryError("INVALID_ARGUMENT", f"name must be a non-empty string, got {name!r}")
        self._settle()
        self._checkpoints[name] = self._current

    @property
    def checkpoints(self) -> dict[str, int]:
        """Checkpoint names and the index of the step that produced their state
        (0: the state before any recorded step)."""
        self._settle()
        return dict(self._checkpoints)

    def remove_checkpoint(self, name: str) -> None:
        self._settle()
        if name not in self._checkpoints:
            raise HistoryError("UNKNOWN_CHECKPOINT", f"no checkpoint named {name!r}")
        del self._checkpoints[name]

    def goto(self, name: str) -> list[HistoryEntry]:
        """Undo or redo until the scene is in the state of checkpoint ``name``.

        Returns the steps undone or redone. All or nothing.
        """
        self._settle()
        self._check_idle("go to a checkpoint")
        if name not in self._checkpoints:
            raise HistoryError(
                "UNKNOWN_CHECKPOINT",
                f"no checkpoint named {name!r}",
                hint=f"checkpoints: {', '.join(sorted(self._checkpoints)) or 'none'}",
            )
        target = self._checkpoints[name]
        if target == self._current:
            return []
        for k in range(len(self._undo) - 1, -1, -1):
            if self._undo[k].parent == target:
                return self._move(self._undo, self._redo, len(self._undo) - k, forward=False)
        for k in range(len(self._redo) - 1, -1, -1):
            if self._redo[k].serial == target:
                return self._move(self._redo, self._undo, len(self._redo) - k, forward=True)
        raise HistoryError(  # pragma: no cover - pruned checkpoints are removed
            "UNKNOWN_CHECKPOINT", f"checkpoint {name!r} can no longer be reached"
        )

    # -- inspection -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible summary: steps, undone steps, checkpoints, limits, memory."""
        self._settle()
        return {
            "enabled": self._enabled,
            "entries": [e.to_dict() for e in self.entries],
            "redo": [e.to_dict() for e in self.redo_entries],
            "checkpoints": self.checkpoints,
            "max_entries": self._max_entries,
            "max_bytes": self._max_bytes,
            "bytes": self.nbytes,
            "dropped": self.dropped,
            "revision": self._revision,
        }

    def __repr__(self) -> str:
        return f"History(entries={len(self._undo)}, redo={len(self._redo)})"

    # -- files --------------------------------------------------------------------------

    def _project_dir(self, project_dir: Any) -> Any:
        from pathlib import Path

        if project_dir is not None:
            return Path(project_dir).absolute()
        return (self._scene.materials.base_dir or Path.cwd()).absolute()

    def save(self, path: Any, *, project_dir: Any = None) -> Any:
        """Write the history to ``path`` (JSON, atomically); returns the path.

        ``scene.save(project, history=True)`` saves the project and its history
        together (``<project>.history``). Texture files in states are stored relative
        to ``project_dir`` (default: the scene's project directory).
        """
        from ._history_file import save_history

        return save_history(self._scene, path, self._project_dir(project_dir))

    def load(self, path: Any, *, project_dir: Any = None, verify: bool = True) -> None:
        """Replace this history with the one saved in ``path``.

        The file must belong to the scene as it is now (``HISTORY_MISMATCH``
        otherwise). With ``verify`` it is first replayed on a copy of the scene --
        every step undone and redone -- and refused (``INVALID_HISTORY``) unless the
        copy comes back to the same document. Nothing changes if loading fails.
        """
        from ._history_file import load_history

        self._check_idle("load a history")
        load_history(self._scene, path, self._project_dir(project_dir), verify=verify)

    def verify(self) -> dict[str, Any]:
        """Check that the history replays exactly: save it, load it into a copy of the
        scene, undo and redo every step there (checking the scene's invariants after
        each), and compare the copy with the scene. The scene itself is not touched.

        Returns ``{"steps", "undone_and_redone", "redone_and_undone", "base", "ok"}``.
        """
        from ._history_file import history_document, replay
        from .serialization import scene_from_dict, scene_to_dict

        self._settle()
        self._check_idle("verify the history")
        project_dir = self._project_dir(None)
        document = history_document(self._scene, project_dir)
        scratch = scene_from_dict(scene_to_dict(self._scene, project_dir), project_dir)
        return replay(scratch, document, project_dir)

    # -- scene construction (loaders) -------------------------------------------------

    @contextmanager
    def _suspended(self) -> Iterator[None]:
        """Record nothing (loading or importing a scene: its history starts empty)."""
        self._building += 1
        try:
            yield
        finally:
            self._building -= 1


def _default_label(changes: list[_Change]) -> str:
    c = changes[0]
    name = getattr(c.target, "_name", None)
    what = f"{c.aspect.name} of {name!r}" if isinstance(name, str) else c.aspect.name
    more = f" (+{len(changes) - 1} more)" if len(changes) > 1 else ""
    return f"Change {what}{more}"


def _payload_size(p: Any) -> int:
    if isinstance(p, (bytes, bytearray)):
        return len(p)
    size = 0
    for name in ("positions", "normals", "uvs", "triangles", "groups"):
        array = getattr(p, name, None)
        if array is not None:
            size += int(array.nbytes)
    return size or sys.getsizeof(p)


# --------------------------------------------------------------------------- #
# Invariants (tests, and STRICT mode)
# --------------------------------------------------------------------------- #


def check_scene(scene: Scene) -> None:
    """Raise ``AssertionError`` if the scene's internal indexes are inconsistent."""
    seen: dict[str, Any] = {}

    def visit(obj: Any, parent: Any) -> None:
        assert obj._scene is scene, f"{obj!r} is in the hierarchy but not in the scene"
        assert obj._parent is parent, f"{obj!r} has the wrong parent"
        assert obj._id not in seen, f"{obj!r} appears twice in the hierarchy"
        seen[obj._id] = obj
        for child in obj._children:
            visit(child, obj)

    for root in scene._roots:
        visit(root, None)
    assert seen.keys() == scene._objects.keys(), "the hierarchy and the object table differ"
    assert all(scene._objects[k] is v for k, v in seen.items())
    assert scene._by_name == {o._name: o for o in seen.values()}, "the name index is stale"
    assert scene._active_camera is None or (
        scene._active_camera in seen and seen[scene._active_camera]._camera is not None
    ), "the active camera is not a camera of the scene"
    library = scene._materials
    assert library._material_names == {m._name: m for m in library._materials.values()}
    assert library._texture_names == {t._name: t for t in library._textures.values()}
    assert all(m._library is library for m in library._materials.values())
    assert all(t._library is library for t in library._textures.values())
    for obj in seen.values():
        for ref in (obj._material, *obj._group_materials.values()):
            assert ref is None or ref in library._materials, f"{obj!r} uses a missing material"
        if obj._geometry is not None:
            system = scene._procedural
            assert system is not None and obj._geometry in system._geometries, (
                f"{obj!r} uses a missing geometry node"
            )
